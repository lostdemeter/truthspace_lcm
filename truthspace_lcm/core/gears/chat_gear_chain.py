"""
ChatGearChain - Unified Chat Entry Point with Knowledge Persistence

Wraps the chat application as a GearChain with integrated knowledge store.
Implements Design 091: Position Is Everything.

Features:
- Loads knowledge store on startup
- Saves knowledge store on shutdown
- Uses position-based learning for all interactions
- Concepts that are frequently successful persist
- Concepts that are rarely used fade

The knowledge store learns from:
- Intent detection patterns
- Successful/failed query responses
- User feedback (explicit or implicit)

Author: Lesley Gushurst
License: GPLv3
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from truthspace_lcm.core.gear import Gear, GearChain, GearState
from truthspace_lcm.core.knowledge import GeometricKnowledgeStore, Concept, CRITICAL_LINE


# Default paths
DEFAULT_KNOWLEDGE_PATH = Path.home() / ".truthspace" / "knowledge.json"


@dataclass
class ChatConfig:
    """Configuration for ChatGearChain."""
    llm_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "qwen2:latest"
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH
    auto_save: bool = False  # Save on exit, not every query
    prune_on_save: bool = True  # Prune when saving
    debug: bool = False


class KnowledgeLearningGear(Gear):
    """
    A gear that learns from interactions using the knowledge store.
    
    This gear sits in the chain and:
    1. Creates concepts from queries
    2. Updates concept positions based on success/failure
    3. Provides learned patterns to downstream gears
    
    Position anchors define the "attractor basins" in the space:
    - Successful queries move toward their intent anchor
    - Failed queries move away
    - Concepts past the critical line persist
    """
    
    # Intent position anchors (4D space)
    INTENT_ANCHORS = {
        'KNOWLEDGE': (0.8, 0.0, 0.0, 0.0),
        'CHAT': (0.8, 0.0, 0.0, 0.0),  # Same as KNOWLEDGE
        'TOOL_CALL': (0.0, 0.8, 0.0, 0.0),
        'EXECUTE': (0.0, 0.8, 0.0, 0.0),  # Same as TOOL_CALL
        'CODE_GENERATION': (0.0, 0.0, 0.8, 0.0),
        'ORCHESTRATOR': (0.0, 0.0, 0.0, 0.8),
    }
    
    def __init__(self, store: GeometricKnowledgeStore = None):
        super().__init__(name="KnowledgeLearningGear", ratio=1.0)
        self._knowledge_store = store or GeometricKnowledgeStore(
            name="chat_knowledge",
            dims=4
        )
        self._current_concept_id: Optional[str] = None
    
    @property
    def knowledge_store(self) -> GeometricKnowledgeStore:
        return self._knowledge_store
    
    def forward(self, state: GearState) -> GearState:
        """
        Process input and create/update concept.
        
        1. Find or create concept for this query
        2. Store concept ID in state for later feedback
        3. Add any relevant learned patterns to context
        """
        query = state.entity
        
        # Find existing concept or create new one
        matches = self._knowledge_store.query(query, top_k=1)
        
        if matches and matches[0][1] > 0.6:  # Good match
            concept = matches[0][0]
        else:
            # Create new concept
            concept = self._knowledge_store.add_from_text(query, source="chat")
        
        self._current_concept_id = concept.id
        
        # Add to state metadata
        state.metadata['knowledge_concept_id'] = concept.id
        state.metadata['knowledge_concept_magnitude'] = concept.magnitude
        state.metadata['knowledge_concept_persists'] = concept.persists
        
        # Add learned patterns from similar persisting concepts
        similar = self._knowledge_store.query(query, top_k=5)
        learned_patterns = [
            {
                'words': list(c.words)[:10],
                'magnitude': c.magnitude,
                'persists': c.persists,
            }
            for c, score in similar if c.persists and score > 0.3
        ]
        state.metadata['learned_patterns'] = learned_patterns
        
        return state
    
    def feedback(self, concept_id: str, intent: str, success: bool) -> bool:
        """
        Provide feedback on a query result.
        
        Moves the concept toward/away from the intent anchor.
        """
        anchor = self.INTENT_ANCHORS.get(intent.upper())
        if anchor is None:
            return False
        
        return self._knowledge_store.use(concept_id, anchor, success)
    
    def feedback_current(self, intent: str, success: bool) -> bool:
        """Provide feedback on the most recent query."""
        if self._current_concept_id is None:
            return False
        return self.feedback(self._current_concept_id, intent, success)


class ChatGearChain(GearChain):
    """
    Unified chat entry point with knowledge persistence.
    
    This chain wraps the chat application and integrates:
    - Knowledge store for learning from interactions
    - Intent detection
    - Query routing (knowledge, tools, code)
    - Feedback loop for position-based learning
    
    Usage:
        chain = ChatGearChain.create()
        chain.load_knowledge()
        
        # Process queries
        response = chain.chat("Who is George Washington?")
        
        # Provide feedback
        chain.feedback(success=True)
        
        # Save on shutdown
        chain.save_knowledge()
    """
    
    def __init__(self, config: ChatConfig = None):
        super().__init__(name="ChatGearChain")
        self.config = config or ChatConfig()
        
        # Knowledge learning gear
        self._learning_gear = KnowledgeLearningGear()
        self.add(self._learning_gear)
        
        # Intent detector (optional, added if available)
        self._intent_gear = None
        try:
            from truthspace_lcm.core.gears.intent_detector_gear import IntentDetectorGear
            self._intent_gear = IntentDetectorGear()
            # Note: IntentDetectorGear is not a Gear subclass, so we use it directly
        except ImportError:
            pass
        
        # Conversational chain (optional, added if available)
        self._conv_chain = None
        try:
            from truthspace_lcm.core import ConversationalChain
            self._conv_chain = ConversationalChain()
            self._conv_chain.configure_llm(self.config.llm_url, self.config.llm_model)
        except ImportError:
            pass
        
        # Python code gear (optional)
        self._code_gear = None
        try:
            from truthspace_lcm.core.gears.python_code_gear import PythonCodeGear
            self._code_gear = PythonCodeGear()
            self._code_gear.configure_llm(self.config.llm_url, self.config.llm_model)
        except ImportError:
            pass
        
        # Orchestrator (optional)
        self._orchestrator = None
        try:
            from truthspace_lcm.core.orchestrators.gear_orchestrator import GearOrchestrator
            self._orchestrator = GearOrchestrator()
            self._orchestrator.configure_llm(self.config.llm_url, self.config.llm_model)
        except ImportError:
            pass
        
        # Track last intent for feedback
        self._last_intent: Optional[str] = None
    
    @classmethod
    def create(cls, config: ChatConfig = None) -> 'ChatGearChain':
        """Factory method to create a configured ChatGearChain."""
        return cls(config)
    
    @property
    def knowledge_store(self) -> GeometricKnowledgeStore:
        """Get the knowledge store."""
        return self._learning_gear.knowledge_store
    
    def load_knowledge(self, path: Path = None) -> bool:
        """
        Load knowledge store from file.
        
        Returns True if loaded, False if file doesn't exist.
        """
        path = path or self.config.knowledge_path
        
        if not path.exists():
            if self.config.debug:
                print(f"[DEBUG] Knowledge file not found: {path}")
            return False
        
        try:
            store = GeometricKnowledgeStore.load(str(path))
            self._learning_gear._knowledge_store = store
            
            if self.config.debug:
                print(f"[DEBUG] Loaded knowledge: {len(store)} concepts")
                print(f"[DEBUG] Persisting: {len(store.get_persisting_concepts())}")
            
            return True
        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Failed to load knowledge: {e}")
            return False
    
    def save_knowledge(self, path: Path = None) -> bool:
        """Save knowledge store to file."""
        path = path or self.config.knowledge_path
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Optionally prune before saving
            if self.config.prune_on_save:
                pruned = self.knowledge_store.prune()
                if self.config.debug and pruned > 0:
                    print(f"[DEBUG] Pruned {pruned} fading concepts")
            
            self.knowledge_store.save(str(path))
            
            if self.config.debug:
                print(f"[DEBUG] Saved knowledge: {len(self.knowledge_store)} concepts")
            
            return True
        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Failed to save knowledge: {e}")
            return False
    
    def chat(self, query: str) -> str:
        """
        Process a chat query with knowledge learning.
        
        1. Create/update concept for query
        2. Detect intent
        3. Route to appropriate handler
        4. Return response
        
        Call feedback() after to indicate success/failure.
        """
        # Create GearState
        state = GearState(entity=query)
        
        # Process through learning gear
        state = self._learning_gear.forward(state)
        
        # Detect intent
        intent = "KNOWLEDGE"  # Default
        if self._intent_gear:
            result = self._intent_gear.detect(query)
            intent = result.intent.name
            state.metadata['intent'] = intent
            state.metadata['intent_confidence'] = result.confidence
        
        self._last_intent = intent
        
        # Route based on intent
        response = self._route_query(query, intent, state)
        
        return response
    
    def _route_query(self, query: str, intent: str, state: GearState) -> str:
        """Route query to appropriate handler based on intent."""
        
        if intent in ('CHAT', 'KNOWLEDGE', 'UNKNOWN'):
            # Knowledge query
            if self._conv_chain:
                return self._conv_chain.chat(query)
            return "Knowledge chain not available."
        
        elif intent == 'CODE_GENERATION':
            # Code generation
            if self._code_gear:
                result = self._code_gear.generate_from_text(query)
                if result.success:
                    response = f"```python\n{result.code}\n```"
                    if result.verified:
                        response += "\n✓ Code verified"
                    return response
                return f"Code generation failed: {result.error}"
            return "Code generator not available."
        
        elif intent in ('TOOL_CALL', 'ORCHESTRATOR'):
            # Tool/orchestrator
            if self._orchestrator:
                result = self._orchestrator.execute(query, dry_run=True)
                if result['commands']:
                    cmd_list = '\n'.join([f"  $ {cmd}" for cmd in result['commands']])
                    return f"Commands to execute:\n{cmd_list}"
                return "No commands generated."
            return "Orchestrator not available."
        
        return f"Unknown intent: {intent}"
    
    def feedback(self, success: bool) -> bool:
        """
        Provide feedback on the last query.
        
        This is THE learning operation - moves the concept
        toward/away from the intent anchor based on success.
        """
        if self._last_intent is None:
            return False
        
        return self._learning_gear.feedback_current(self._last_intent, success)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the chain and knowledge store."""
        store = self.knowledge_store
        return {
            'total_concepts': len(store),
            'persisting_concepts': len(store.get_persisting_concepts()),
            'fading_concepts': len(store.get_fading_concepts()),
            'critical_line': CRITICAL_LINE,
            'has_conv_chain': self._conv_chain is not None,
            'has_code_gear': self._code_gear is not None,
            'has_orchestrator': self._orchestrator is not None,
        }
    
    def __enter__(self):
        """Context manager entry - load knowledge."""
        self.load_knowledge()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save knowledge."""
        self.save_knowledge()
        return False


def create_chat_chain(config: ChatConfig = None) -> ChatGearChain:
    """Convenience function to create a ChatGearChain."""
    return ChatGearChain.create(config)
