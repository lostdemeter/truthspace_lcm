"""
ChatPipeline - HyperMapping-based Chat System

Replaces ChatGearChain with a cleaner, more geometric architecture.

The key insight: Chat routing IS a HyperMapping problem:
- Input = user query
- Output = response or action
- Position = geometric encoding of intent/meaning

Design Principles:
- Intent detection via bootstrapped templates (not hardcoded rules)
- Knowledge retrieval via position-based matching
- Learning through position reinforcement
- No magic numbers - all thresholds are geometric (critical line)

Author: Lesley Gushurst
License: GPLv3
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

# Import from hypermapping package
import sys
# Add parent of hypermapping to path so we can import hypermapping as a package
hypermapping_parent = Path(__file__).parent.parent.parent
if str(hypermapping_parent) not in sys.path:
    sys.path.insert(0, str(hypermapping_parent))

from hypermapping import (
    HyperMapping, HyperPipeline, Mapping, MatchResult, CRITICAL_LINE,
    TextEncoder
)

from .knowledge_space import KnowledgeSpace


class Intent(Enum):
    """Intent categories for chat routing."""
    KNOWLEDGE = auto()      # Knowledge/information query
    TOOL_CALL = auto()      # Execute a command/tool
    CODE_GENERATION = auto() # Generate code
    CLARIFICATION = auto()  # Need more information
    UNKNOWN = auto()        # Cannot determine intent


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: Intent
    confidence: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatConfig:
    """Configuration for ChatPipeline."""
    knowledge_path: Optional[Path] = None
    auto_save: bool = False
    prune_on_save: bool = True
    debug: bool = False
    dims: int = 8


class IntentSpace(HyperMapping):
    """
    Intent detection using HyperMapping with TextEncoder.
    
    Uses semantic text matching for intent classification.
    Bootstrap patterns define the "attractor basins" for each intent.
    After bootstrapping, all detection is geometric (position-based).
    
    Intent positions are bootstrapped, then refined through use.
    """
    
    def __init__(self, dims: int = 8):
        encoder = TextEncoder(dims=dims)
        super().__init__(dims=dims, encoder=encoder, name="intent")
        self._bootstrap_encoder()
        self._bootstrap_intents()
    
    def _bootstrap_encoder(self) -> None:
        """Learn word positions from bootstrap patterns."""
        # Collect all bootstrap patterns for learning
        all_patterns = []
        
        # Knowledge queries
        all_patterns.extend([
            "what is", "who is", "how does", "why does",
            "tell me about", "explain", "describe",
            "what are", "when did", "where is",
        ])
        
        # Tool calls
        all_patterns.extend([
            "create", "delete", "run", "execute",
            "make a", "set up", "install", "remove",
            "start", "stop", "restart",
        ])
        
        # Code generation
        all_patterns.extend([
            "write code", "python function", "javascript",
            "implement", "code that", "program to",
            "script that", "function that",
        ])
        
        # Clarification
        all_patterns.extend([
            "help", "what do you mean",
            "I don't understand", "clarify",
        ])
        
        # Learn from patterns
        self.encoder.learn(all_patterns)
    
    def _bootstrap_intents(self) -> None:
        """
        Bootstrap intent detection with template patterns.
        
        These are the ONLY hardcoded patterns - they bootstrap the space.
        After bootstrapping, all detection is geometric.
        """
        # Knowledge queries (questions)
        knowledge_patterns = [
            "what is", "who is", "how does", "why does",
            "tell me about", "explain", "describe",
            "what are", "when did", "where is",
        ]
        for pattern in knowledge_patterns:
            self.bootstrap(pattern, Intent.KNOWLEDGE.name)
        
        # Tool calls (commands)
        tool_patterns = [
            "create", "delete", "run", "execute",
            "make a", "set up", "install", "remove",
            "start", "stop", "restart",
        ]
        for pattern in tool_patterns:
            self.bootstrap(pattern, Intent.TOOL_CALL.name)
        
        # Code generation
        code_patterns = [
            "write code", "python function", "javascript",
            "implement", "code that", "program to",
            "script that", "function that",
        ]
        for pattern in code_patterns:
            self.bootstrap(pattern, Intent.CODE_GENERATION.name)
        
        # Clarification needed
        clarification_patterns = [
            "help", "?", "what do you mean",
            "I don't understand", "clarify",
        ]
        for pattern in clarification_patterns:
            self.bootstrap(pattern, Intent.CLARIFICATION.name)
    
    def detect(self, query: str) -> IntentResult:
        """
        Detect intent from query.
        
        Two-phase detection:
        1. Check for prefix matches (bootstrap patterns)
        2. Fall back to geometric matching
        
        This ensures bootstrap patterns work exactly, while still
        allowing geometric generalization for novel queries.
        """
        query_lower = query.lower().strip()
        
        # Phase 1: Check for prefix matches against templates
        # This is the BOOTSTRAP phase - exact pattern matching
        if hasattr(self, '_templates'):
            for template_key, intent_name in self._templates.items():
                if query_lower.startswith(template_key.lower()):
                    try:
                        intent = Intent[intent_name]
                        return IntentResult(
                            intent=intent,
                            confidence=1.0,
                            reason=f"Prefix match: {template_key}",
                            metadata={'matched_template': template_key, 'match_type': 'prefix'}
                        )
                    except (KeyError, ValueError):
                        pass
        
        # Phase 2: Geometric matching (for novel queries)
        result = self.forward(query)
        
        if result is None:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reason="No mappings in intent space"
            )
        
        # Parse intent from output
        try:
            intent = Intent[result.output]
        except (KeyError, ValueError):
            intent = Intent.UNKNOWN
        
        # Confidence is similarity (geometric)
        confidence = max(0.0, result.similarity)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            reason=f"Geometric match: {result.input}",
            metadata={'matched_template': result.input, 'match_type': 'geometric'}
        )
    
    def learn_intent(self, query: str, correct_intent: Intent) -> None:
        """
        Learn from a correction.
        
        Updates the space so similar queries map to the correct intent.
        """
        self.learn(query, correct_intent.name)


class ChatPipeline:
    """
    Chat pipeline using HyperMapping.
    
    Replaces ChatGearChain with a cleaner, more geometric architecture.
    
    Pipeline stages:
    1. Intent detection (IntentSpace)
    2. Knowledge retrieval (KnowledgeSpace)
    3. Response generation (template-based)
    
    Usage:
        pipeline = ChatPipeline()
        pipeline.load_knowledge("knowledge.json")
        
        response = pipeline.chat("What is the capital of France?")
        pipeline.feedback(success=True)
        
        pipeline.save_knowledge("knowledge.json")
    """
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        
        # Intent detection space
        self.intent_space = IntentSpace()
        
        # Knowledge space
        self.knowledge_space = KnowledgeSpace(
            name="chat_knowledge",
            dims=self.config.dims
        )
        
        # Response templates (bootstrapped)
        self.response_space = HyperMapping(
            dims=self.config.dims,
            name="responses"
        )
        self._bootstrap_responses()
        
        # Build pipeline
        self.pipeline = HyperPipeline(name="chat")
        self.pipeline.add("intent", self.intent_space)
        self.pipeline.add("knowledge", self.knowledge_space)
        self.pipeline.add("responses", self.response_space)
        
        # Track last query for feedback
        self._last_query: Optional[str] = None
        self._last_intent: Optional[Intent] = None
        self._last_mapping: Optional[Mapping] = None
    
    def _bootstrap_responses(self) -> None:
        """Bootstrap response templates."""
        # These are fallback responses when no knowledge matches
        self.response_space.bootstrap(
            "no_knowledge",
            "I don't have information about that topic."
        )
        self.response_space.bootstrap(
            "clarification_needed",
            "Could you please clarify what you're asking?"
        )
        self.response_space.bootstrap(
            "tool_not_available",
            "Tool execution is not available in this mode."
        )
        self.response_space.bootstrap(
            "code_not_available",
            "Code generation is not available in this mode."
        )
    
    def chat(self, query: str) -> str:
        """
        Process a chat query.
        
        1. Detect intent (geometric)
        2. Route to appropriate handler
        3. Return response
        
        Call feedback() after to indicate success/failure.
        """
        self._last_query = query
        
        # Detect intent
        intent_result = self.intent_space.detect(query)
        self._last_intent = intent_result.intent
        
        if self.config.debug:
            print(f"[DEBUG] Query: {query}")
            print(f"[DEBUG] Intent: {intent_result.intent.name} ({intent_result.confidence:.2f})")
            print(f"[DEBUG] Reason: {intent_result.reason}")
        
        # Route based on intent
        if intent_result.intent == Intent.KNOWLEDGE:
            return self._handle_knowledge(query)
        elif intent_result.intent == Intent.TOOL_CALL:
            return self._handle_tool(query)
        elif intent_result.intent == Intent.CODE_GENERATION:
            return self._handle_code(query)
        elif intent_result.intent == Intent.CLARIFICATION:
            return self.response_space.compose("clarification_needed")
        else:
            # Unknown - try knowledge anyway
            return self._handle_knowledge(query)
    
    def _handle_knowledge(self, query: str) -> str:
        """Handle knowledge query."""
        results = self.knowledge_space.query_text(query, top_k=3)
        
        if results and results[0].similarity > 0.1:
            self._last_mapping = results[0].mapping
            
            if self.config.debug:
                print(f"[DEBUG] Knowledge match: {results[0].similarity:.3f}")
            
            # Return the matched knowledge
            return results[0].output
        
        # No good match - return fallback
        return self.response_space.compose("no_knowledge")
    
    def _handle_tool(self, query: str) -> str:
        """Handle tool call request."""
        # For now, just indicate tools aren't available
        # This will be extended when we integrate tool execution
        return self.response_space.compose("tool_not_available")
    
    def _handle_code(self, query: str) -> str:
        """Handle code generation request."""
        # For now, just indicate code gen isn't available
        # This will be extended when we integrate code generation
        return self.response_space.compose("code_not_available")
    
    def feedback(self, success: bool) -> bool:
        """
        Provide feedback on the last response.
        
        This is THE learning operation:
        - Success: Reinforce the mapping that was used
        - Failure: Weaken the mapping
        
        Returns True if feedback was recorded.
        """
        if self._last_mapping is None:
            return False
        
        self.knowledge_space.use(self._last_mapping, success)
        
        if self.config.debug:
            print(f"[DEBUG] Feedback: {'success' if success else 'failure'}")
            print(f"[DEBUG] Mapping magnitude: {self._last_mapping.magnitude:.3f}")
        
        return True
    
    def add_knowledge(self, text: str, source: str = "user") -> Mapping:
        """Add knowledge to the space."""
        return self.knowledge_space.add_text(text, source)
    
    def learn_intent(self, query: str, correct_intent: Intent) -> None:
        """Correct intent detection."""
        self.intent_space.learn_intent(query, correct_intent)
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def load_knowledge(self, path: Optional[str] = None) -> bool:
        """Load knowledge from file."""
        path = path or (str(self.config.knowledge_path) if self.config.knowledge_path else None)
        
        if not path or not Path(path).exists():
            if self.config.debug:
                print(f"[DEBUG] Knowledge file not found: {path}")
            return False
        
        try:
            self.knowledge_space = KnowledgeSpace.load(path)
            
            if self.config.debug:
                print(f"[DEBUG] Loaded knowledge: {len(self.knowledge_space)} concepts")
            
            return True
        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Failed to load knowledge: {e}")
            return False
    
    def save_knowledge(self, path: Optional[str] = None) -> bool:
        """Save knowledge to file."""
        path = path or (str(self.config.knowledge_path) if self.config.knowledge_path else None)
        
        if not path:
            return False
        
        try:
            if self.config.prune_on_save:
                pruned = self.knowledge_space.prune()
                if self.config.debug and pruned > 0:
                    print(f"[DEBUG] Pruned {pruned} fading concepts")
            
            self.knowledge_space.save(path)
            
            if self.config.debug:
                print(f"[DEBUG] Saved knowledge: {len(self.knowledge_space)} concepts")
            
            return True
        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Failed to save knowledge: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        return {
            'pipeline': self.pipeline.get_stats(),
            'knowledge': self.knowledge_space.get_stats(),
            'intent_templates': len(self.intent_space),
            'response_templates': len(self.response_space.templates),
        }
    
    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------
    
    def __enter__(self):
        """Context manager entry - load knowledge."""
        if self.config.knowledge_path:
            self.load_knowledge()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save knowledge."""
        if self.config.auto_save and self.config.knowledge_path:
            self.save_knowledge()
        return False
    
    def __repr__(self) -> str:
        return f"ChatPipeline(knowledge={len(self.knowledge_space)}, intents={len(self.intent_space)})"


def create_chat_pipeline(config: Optional[ChatConfig] = None) -> ChatPipeline:
    """Convenience function to create a ChatPipeline."""
    return ChatPipeline(config)
