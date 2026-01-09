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
from .code_space import CodeSpace
from .plot_space import PlotSpace
from .ollama_space import OllamaSpace
from .bootstrap_knowledge import get_bootstrap_knowledge, get_bootstrap_synonyms
from .quaternion_encoder import QuaternionEncoder, QuaternionPosition
from .dynamic_dimensions import DynamicDimensionRegistry
from .learned_knowledge import LearnedKnowledge, extract_llm_response


class Intent(Enum):
    """Intent categories for chat routing."""
    KNOWLEDGE = auto()      # Knowledge/information query
    TOOL_CALL = auto()      # Execute a command/tool
    CODE_GENERATION = auto() # Generate code
    PLOT_GENERATION = auto() # Generate matplotlib plot
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
    use_phi_lattice: bool = True   # Design 099, 047: Use φ-lattice coordinates with intrinsic/functional
    use_quaternion: bool = True    # Design 104-105: Use quaternion encoding


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
        
        # Plot generation
        all_patterns.extend([
            "plot", "graph", "chart", "visualize",
            "sine wave", "cosine wave", "histogram",
            "scatter plot", "bar chart", "pie chart",
            "create a plot", "make a graph", "draw a chart",
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
        
        # Plot generation (matplotlib)
        plot_patterns = [
            "plot", "graph", "chart", "visualize",
            "sine wave", "cosine wave", "histogram",
            "scatter plot", "bar chart", "pie chart",
            "create a plot", "make a graph", "draw a chart",
        ]
        for pattern in plot_patterns:
            self.bootstrap(pattern, Intent.PLOT_GENERATION.name)
        
        # Clarification / help / about (routes to identity/capabilities)
        clarification_patterns = [
            "help", "?", "what do you mean",
            "I don't understand", "clarify",
            "what can you do", "what are you", "who are you",
            "what are your capabilities", "introduce yourself",
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
        
        # Phase 0a: Check for KNOWLEDGE query patterns FIRST
        # "what is X" and "tell me about X" are knowledge queries even if X contains plot words
        knowledge_prefixes = ['what is', 'what are', 'who is', 'who are', 
                              'tell me about', 'explain', 'describe', 'define']
        if any(query_lower.startswith(prefix) for prefix in knowledge_prefixes):
            return IntentResult(
                intent=Intent.KNOWLEDGE,
                confidence=1.0,
                reason=f"Knowledge query prefix detected",
                metadata={'match_type': 'knowledge_prefix'}
            )
        
        # Phase 0b: Check for plot-related keywords
        # This takes priority over generic "create" which would match TOOL_CALL
        # But NOT if it's a knowledge query about plotting tools
        plot_keywords = ['plot', 'graph', 'chart', 'sine', 'cosine', 'histogram', 
                         'scatter', 'bar chart', 'pie chart', 'visualize', 'wave']
        if any(kw in query_lower for kw in plot_keywords):
            return IntentResult(
                intent=Intent.PLOT_GENERATION,
                confidence=1.0,
                reason=f"Plot keyword detected",
                metadata={'match_type': 'keyword'}
            )
        
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
            dims=self.config.dims,
            use_phi_lattice=self.config.use_phi_lattice
        )
        
        # Code generation space
        self.code_space = CodeSpace(
            name="code_generation",
            dims=self.config.dims
        )
        
        # Plot generation space
        self.plot_space = PlotSpace(
            name="plot_generation",
            dims=self.config.dims
        )
        
        # LLM knowledge acquisition (optional)
        self.ollama_space: Optional[OllamaSpace] = None
        try:
            self.ollama_space = OllamaSpace(
                name="ollama",
                dims=self.config.dims
            )
        except Exception:
            pass  # Ollama not available
        
        # Response templates (bootstrapped)
        self.response_space = HyperMapping(
            dims=self.config.dims,
            name="responses"
        )
        self._bootstrap_responses()
        
        # Bootstrap knowledge
        self._bootstrap_knowledge()
        
        # Build pipeline
        self.pipeline = HyperPipeline(name="chat")
        self.pipeline.add("intent", self.intent_space)
        self.pipeline.add("knowledge", self.knowledge_space)
        self.pipeline.add("code", self.code_space)
        self.pipeline.add("plot", self.plot_space)
        self.pipeline.add("responses", self.response_space)
        
        # Track last query for feedback
        self._last_query: Optional[str] = None
        self._last_intent: Optional[Intent] = None
        self._last_mapping: Optional[Mapping] = None
        
        # Quaternion encoder for dynamic dimensions (Design 104-105)
        self._dimension_registry: Optional[DynamicDimensionRegistry] = None
        self._quaternion_encoder: Optional[QuaternionEncoder] = None
        if self.config.use_quaternion:
            self._dimension_registry = DynamicDimensionRegistry()
            self._quaternion_encoder = QuaternionEncoder(self._dimension_registry)
            if self.config.debug:
                print(f"[DEBUG] Quaternion encoder: {self._quaternion_encoder.summary()}")
        
        # Learned knowledge - separate from bootstrap, grows through conversation
        self._learned_knowledge = LearnedKnowledge()
        self._load_learned_into_knowledge_space()
    
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
    
    def _bootstrap_knowledge(self) -> None:
        """Bootstrap knowledge from predefined topics."""
        knowledge_items = get_bootstrap_knowledge()
        synonyms = get_bootstrap_synonyms()
        
        # Add synonyms to the knowledge space encoder
        if hasattr(self.knowledge_space, 'encoder'):
            self.knowledge_space.encoder.add_synonyms(synonyms)
        
        # Train encoder on all knowledge texts and keywords
        all_texts = []
        for item in knowledge_items:
            all_texts.append(item["text"])
            all_texts.extend(item.get("keywords", []))
        
        if hasattr(self.knowledge_space, 'encoder'):
            self.knowledge_space.encoder.learn(all_texts)
        
        # Add each knowledge item
        for item in knowledge_items:
            # Add the main text
            mapping = self.knowledge_space.add_text(
                item["text"],
                source="bootstrap"
            )
            
            # Store metadata from bootstrap JSON
            if "phi_levels" in item:
                mapping.metadata["phi_levels"] = item["phi_levels"]
            if "keywords" in item:
                mapping.metadata["keywords"] = item["keywords"]
            if "topic" in item:
                mapping.metadata["topic"] = item["topic"]
            
            # Register keywords as primitives (Design 103: Self-Assembling Primitives)
            # This transforms keywords to geometry at bootstrap time
            if self.config.use_phi_lattice and "keywords" in item and "phi_levels" in item:
                self.knowledge_space._primitive_registry.register_from_bootstrap(
                    text=item["text"],
                    keywords=item["keywords"],
                    phi_levels=item["phi_levels"]
                )
        
        if self.config.debug:
            print(f"[DEBUG] Bootstrapped {len(knowledge_items)} knowledge items")
            if self.config.use_phi_lattice:
                stats = self.knowledge_space._primitive_registry.stats
                print(f"[DEBUG] Registered {stats['total_count']} primitives ({stats['single_word_count']} single, {stats['multi_word_count']} multi)")
    
    def _load_learned_into_knowledge_space(self) -> None:
        """Load learned facts into the knowledge space."""
        for fact in self._learned_knowledge.all_facts():
            self.knowledge_space.add_text(
                fact.content,
                source="learned",
                reproject=False  # Don't reproject for each - do once at end
            )
        
        # Reproject once after loading all
        if len(self._learned_knowledge) > 0 and len(self.knowledge_space) > 1:
            self.knowledge_space.reproject()
        
        if self.config.debug and len(self._learned_knowledge) > 0:
            print(f"[DEBUG] Loaded {len(self._learned_knowledge)} learned facts")
    
    def learn_from_response(self, query: str, response: str) -> Optional[str]:
        """
        Auto-learn from an LLM response.
        
        Extracts the content from the response and adds it to learned knowledge.
        
        Args:
            query: The original query
            response: The LLM response
            
        Returns:
            The topic learned, or None if nothing was learned
        """
        content = extract_llm_response(response)
        if not content:
            return None
        
        topic = self._extract_topic(query)
        if not topic:
            return None
        
        # Learn the fact
        self._learned_knowledge.learn(topic, content, query)
        
        # Also add to knowledge space for immediate use
        self.knowledge_space.add_text(content, source="learned")
        
        if self.config.debug:
            print(f"[DEBUG] Learned about '{topic}' ({len(content)} chars)")
        
        return topic
    
    def forget_topic(self, topic: str) -> bool:
        """
        Forget a learned topic.
        
        Args:
            topic: The topic to forget
            
        Returns:
            True if topic was forgotten, False if not found
        """
        return self._learned_knowledge.forget(topic)
    
    def learned_topics(self) -> List[str]:
        """Get list of learned topics."""
        return self._learned_knowledge.topics()
    
    def learned_stats(self) -> Dict[str, Any]:
        """Get statistics about learned knowledge."""
        return self._learned_knowledge.stats()
    
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
        elif intent_result.intent == Intent.PLOT_GENERATION:
            return self._handle_plot(query)
        elif intent_result.intent == Intent.CLARIFICATION:
            # Handle help/about queries by looking up identity knowledge
            results = self.knowledge_space.query_text(query, top_k=1)
            if results and results[0].similarity > 0.5:
                return results[0].output
            return self.response_space.compose("clarification_needed")
        else:
            # Unknown - try knowledge anyway
            return self._handle_knowledge(query)
    
    def _handle_knowledge(self, query: str, use_llm: bool = True) -> str:
        """
        Handle knowledge query.
        
        1. Try geometric matching in KnowledgeSpace
        2. If no match and Ollama available, query LLM
        3. If still no match, return "I don't know"
        """
        # Geometric matching in KnowledgeSpace
        results = self.knowledge_space.query_text(query, top_k=3)
        
        # Check if we have a good match
        if results and results[0].similarity > 0.3:
            self._last_mapping = results[0].mapping
            
            if self.config.debug:
                print(f"[DEBUG] Knowledge match: {results[0].similarity:.3f}")
            
            # Return the matched knowledge
            return results[0].output
        
        # No good match - try LLM if available
        if use_llm and self.ollama_space and self.ollama_space.is_available():
            if self.config.debug:
                print(f"[DEBUG] No knowledge match, querying LLM...")
            
            llm_result = self.ollama_space.query(query)
            
            if llm_result.success:
                # Add to knowledge space for future queries
                self.knowledge_space.add_text(
                    llm_result.content,
                    source=f"ollama:{llm_result.model}"
                )
                
                if self.config.debug:
                    print(f"[DEBUG] LLM response added to knowledge")
                
                return llm_result.content
        
        # No match and no LLM - return "I don't know"
        topic = self._extract_topic(query)
        return f"I don't have information about '{topic}'. You can teach me by adding knowledge with /add or by enabling LLM knowledge acquisition."
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        query_lower = query.lower()
        
        prefixes = [
            'what is ', 'what are ', 'who is ', 'who are ',
            'tell me about ', 'explain ', 'describe ',
            'how does ', 'how do ', 'why does ', 'why do ',
        ]
        
        topic = query
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                topic = query[len(prefix):]
                break
        
        return topic.rstrip('?!.').strip()
    
    def _handle_tool(self, query: str) -> str:
        """Handle tool call request."""
        # For now, just indicate tools aren't available
        # This will be extended when we integrate tool execution
        return self.response_space.compose("tool_not_available")
    
    def _handle_code(self, query: str) -> str:
        """Handle code generation request using CodeSpace."""
        result = self.code_space.generate(query, verify=True)
        
        if not result.success:
            return f"Failed to generate code: {result.error}"
        
        # Format response
        response = f"```python\n{result.code}\n```"
        
        if result.pattern_name:
            response += f"\n\n*Pattern: {result.pattern_name}*"
        
        if result.verified:
            response += "\n✓ Code verified - runs successfully"
            if result.output:
                output_preview = result.output.strip()[:200]
                response += f"\nOutput: {output_preview}"
        elif result.error:
            response += f"\n⚠ Verification failed: {result.error}"
        
        return response
    
    def _handle_plot(self, query: str) -> str:
        """Handle plot generation request using PlotSpace."""
        result = self.plot_space.generate(query)
        
        if not result.success:
            return f"Failed to generate plot: {result.error}"
        
        # Verify the code
        result = self.plot_space.verify(result)
        
        # Format response
        response = f"```python\n{result.code}\n```"
        
        if result.plot_type:
            response += f"\n\n*Plot type: {result.plot_type}*"
        
        if result.modifiers:
            mods = ", ".join(f"{k}={v}" for k, v in result.modifiers.items())
            response += f"\n*Modifiers: {mods}*"
        
        if result.verified:
            response += "\n✓ Code verified - syntax OK"
        elif result.error:
            response += f"\n⚠ Verification failed: {result.error}"
        
        return response
    
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
        stats = {
            'pipeline': self.pipeline.get_stats(),
            'knowledge': self.knowledge_space.get_stats(),
            'intent_templates': len(self.intent_space),
            'response_templates': len(self.response_space.templates),
        }
        
        # Add quaternion encoder stats if enabled
        if self._quaternion_encoder:
            stats['quaternion'] = self._quaternion_encoder.summary()
            stats['dimensions'] = self._dimension_registry.summary()
        
        return stats
    
    # -------------------------------------------------------------------------
    # Quaternion Encoding (Design 104-105)
    # -------------------------------------------------------------------------
    
    def encode_quaternion(self, text: str) -> Optional[QuaternionPosition]:
        """
        Encode text to quaternion position with dynamic z-layer.
        
        Returns None if quaternion encoding is disabled.
        """
        if not self._quaternion_encoder:
            return None
        return self._quaternion_encoder.encode(text)
    
    def encode_quaternion_with_description(self, text: str) -> Optional[Tuple[QuaternionPosition, Dict[str, Any]]]:
        """
        Encode text and return position with human-readable description.
        
        Returns (position, description) or None if disabled.
        """
        if not self._quaternion_encoder:
            return None
        return self._quaternion_encoder.encode_with_description(text)
    
    def get_text_dimensions(self, text: str) -> Dict[str, float]:
        """
        Get active dynamic dimensions for text.
        
        Returns dict of dimension_name -> level.
        """
        if not self._dimension_registry:
            return {}
        vec = self._dimension_registry.encode_text(text)
        return self._dimension_registry.describe_vector(vec)
    
    def quaternion_similarity(self, text1: str, text2: str) -> float:
        """
        Compute quaternion-based similarity between two texts.
        
        Uses all layers: semantic (w), grammatical (x), contextual (y), dynamic (z).
        """
        if not self._quaternion_encoder:
            return 0.0
        return self._quaternion_encoder.similarity(text1, text2)
    
    def ingest_corpus(self, text: str) -> None:
        """
        Ingest a corpus to build the dimension registry.
        
        This enables entity discovery and dimension expansion.
        """
        if self._dimension_registry:
            self._dimension_registry.ingest_text(text)
            if self.config.debug:
                print(f"[DEBUG] Ingested corpus, registry: {self._dimension_registry}")
    
    def discover_entities(self) -> List[Tuple[str, float, float]]:
        """
        Discover entities (proper nouns) in ingested text.
        
        Returns list of (entity, score, dimension_density).
        """
        if not self._dimension_registry:
            return []
        return self._dimension_registry.discover_entities()
    
    @property
    def dimension_names(self) -> List[str]:
        """Get list of registered dimension names."""
        if not self._dimension_registry:
            return []
        return self._dimension_registry.dimension_names
    
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
