"""
Gear Message - Universal Communication Protocol for Gears

A simple, standardized message format that all gears can understand.
This enables clean inter-gear communication without complex feedback loops.

The key insight: gears don't need bidirectional feedback, they just need
a common language. GearMessage provides that language.

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum


class MessageIntent(Enum):
    """
    Seed intents - these bootstrap the emergent intent space.
    
    The EmergentIntentSpace can discover new intents beyond these,
    but these provide initial structure for the system to learn from.
    """
    QUERY = "query"           # Ask for information
    TRANSFORM = "transform"   # Transform the content
    VALIDATE = "validate"     # Check if content is valid
    ENHANCE = "enhance"       # Improve/enrich the content
    FILTER = "filter"         # Remove unwanted content
    ROUTE = "route"           # Decide where to send next
    EXECUTE = "execute"       # Perform an action
    REPORT = "report"         # Return status/results


class EmergentIntentSpace:
    """
    Discovers message intents from usage patterns rather than hardcoding.
    
    The key insight: intents are clusters in message-behavior space.
    Messages that lead to similar gear behaviors should have similar intents.
    
    This class:
    1. Starts with seed intents (the MessageIntent enum)
    2. Observes message → gear → outcome patterns
    3. Discovers new intent clusters when patterns don't fit existing intents
    4. Refines intent boundaries based on gear success/failure
    """
    
    def __init__(self):
        # Seed intents from enum
        self.intents: Dict[str, Dict[str, Any]] = {}
        for intent in MessageIntent:
            self.intents[intent.value] = {
                'name': intent.value,
                'patterns': [],      # Regex patterns that match this intent
                'keywords': set(),   # Keywords associated with this intent
                'gear_affinities': {},  # Which gears handle this intent well
                'success_rate': 0.5,    # How often this intent leads to success
                'usage_count': 0,
            }
        
        # Initialize with some seed patterns
        self._seed_patterns()
        
        # Track observations for learning
        self.observations: List[Dict[str, Any]] = []
        self.max_observations = 1000
    
    def _seed_patterns(self):
        """Initialize with basic patterns for seed intents."""
        self.intents['query']['keywords'] = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'does', 'tell', 'explain', 'describe'}
        self.intents['query']['patterns'] = [r'\?$', r'^(who|what|where|when|why|how)\b']
        
        self.intents['transform']['keywords'] = {'convert', 'change', 'make', 'turn', 'translate', 'format'}
        self.intents['transform']['patterns'] = [r'\b(convert|change|turn|translate)\b']
        
        self.intents['validate']['keywords'] = {'check', 'verify', 'valid', 'correct', 'right', 'wrong'}
        self.intents['validate']['patterns'] = [r'\b(check|verify|is this)\b']
        
        self.intents['enhance']['keywords'] = {'improve', 'better', 'enhance', 'expand', 'elaborate', 'more'}
        self.intents['enhance']['patterns'] = [r'\b(improve|enhance|expand|more detail)\b']
        
        self.intents['filter']['keywords'] = {'remove', 'filter', 'exclude', 'only', 'just', 'without'}
        self.intents['filter']['patterns'] = [r'\b(remove|filter|exclude|without)\b']
        
        self.intents['route']['keywords'] = {'send', 'forward', 'pass', 'redirect', 'use'}
        self.intents['route']['patterns'] = [r'\b(send to|forward to|use the)\b']
        
        self.intents['execute']['keywords'] = {'create', 'make', 'run', 'execute', 'do', 'build', 'delete', 'write'}
        self.intents['execute']['patterns'] = [r'\b(create|make|run|execute|build|delete|write)\b']
        
        self.intents['report']['keywords'] = {'status', 'result', 'output', 'show', 'display', 'list'}
        self.intents['report']['patterns'] = [r'\b(status|result|show|display|list)\b']
    
    def detect(self, message: str) -> Tuple[str, float]:
        """
        Detect the intent of a message.
        
        Returns (intent_name, confidence).
        """
        import re
        message_lower = message.lower()
        words = set(message_lower.split())
        
        scores = {}
        for intent_name, intent_data in self.intents.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = len(words & intent_data['keywords'])
            score += keyword_matches * 0.3
            
            # Pattern matching
            for pattern in intent_data['patterns']:
                if re.search(pattern, message_lower):
                    score += 0.4
                    break
            
            # Gear affinity bonus (if we know which gears work well)
            if intent_data['success_rate'] > 0.7:
                score += 0.1
            
            scores[intent_name] = score
        
        if not scores or max(scores.values()) == 0:
            return 'query', 0.3  # Default to query with low confidence
        
        best_intent = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_intent])
        
        return best_intent, confidence
    
    def observe(self, message: str, intent_used: str, gear_name: str, 
                success: bool, outcome: str = ""):
        """
        Record an observation of message → intent → gear → outcome.
        
        This is how the system learns which intents work best.
        """
        observation = {
            'message': message,
            'intent': intent_used,
            'gear': gear_name,
            'success': success,
            'outcome': outcome[:100],  # Truncate
            'words': set(message.lower().split()),
        }
        
        self.observations.append(observation)
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]
        
        # Update intent stats
        if intent_used in self.intents:
            intent = self.intents[intent_used]
            intent['usage_count'] += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1
            intent['success_rate'] = (1 - alpha) * intent['success_rate'] + alpha * (1.0 if success else 0.0)
            
            # Update gear affinity
            if gear_name not in intent['gear_affinities']:
                intent['gear_affinities'][gear_name] = {'successes': 0, 'failures': 0}
            if success:
                intent['gear_affinities'][gear_name]['successes'] += 1
            else:
                intent['gear_affinities'][gear_name]['failures'] += 1
            
            # Learn new keywords from successful HIGH-CONFIDENCE messages only
            # Low-confidence messages might be misclassified, so don't pollute keywords
            detected_intent, detected_conf = self.detect(message)
            if success and detected_conf >= 0.6:
                for word in observation['words']:
                    if len(word) > 3:  # Skip short words
                        intent['keywords'].add(word)
    
    def discover_new_intent(self, min_observations: int = 10) -> Optional[str]:
        """
        Analyze observations to discover potential new intents.
        
        Looks for clusters of messages that:
        1. Don't match existing intents well
        2. Have similar word patterns
        3. Lead to similar outcomes
        
        Returns the name of a discovered intent, or None.
        """
        # Find low-confidence observations (use <= 0.4 to catch edge cases)
        low_conf_obs = []
        for obs in self.observations:
            detected, conf = self.detect(obs['message'])
            if conf <= 0.5:  # Include borderline cases
                low_conf_obs.append(obs)
        
        if len(low_conf_obs) < min_observations:
            return None
        
        # Find common words in low-confidence observations
        word_counts: Dict[str, int] = {}
        for obs in low_conf_obs:
            for word in obs['words']:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find words that appear frequently
        common_words = {w for w, c in word_counts.items() if c >= min_observations // 2}
        
        if not common_words:
            return None
        
        # Create a new intent based on common patterns
        # Name it after the most common word
        intent_name = max(common_words, key=lambda w: word_counts[w])
        
        if intent_name not in self.intents:
            self.intents[intent_name] = {
                'name': intent_name,
                'patterns': [],
                'keywords': common_words,
                'gear_affinities': {},
                'success_rate': 0.5,
                'usage_count': 0,
                'emergent': True,  # Mark as discovered, not seeded
            }
            return intent_name
        
        return None
    
    def get_intent(self, name: str) -> Optional[Dict[str, Any]]:
        """Get intent data by name."""
        return self.intents.get(name)
    
    def list_intents(self) -> List[str]:
        """List all known intents."""
        return list(self.intents.keys())
    
    def get_best_gear(self, intent_name: str) -> Optional[str]:
        """Get the gear with highest success rate for an intent."""
        intent = self.intents.get(intent_name)
        if not intent or not intent['gear_affinities']:
            return None
        
        best_gear = None
        best_rate = 0.0
        
        for gear_name, stats in intent['gear_affinities'].items():
            total = stats['successes'] + stats['failures']
            if total > 0:
                rate = stats['successes'] / total
                if rate > best_rate:
                    best_rate = rate
                    best_gear = gear_name
        
        return best_gear
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving."""
        return {
            'intents': {
                name: {
                    **data,
                    'keywords': list(data['keywords']),
                }
                for name, data in self.intents.items()
            },
            'observation_count': len(self.observations),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmergentIntentSpace':
        """Load from saved data."""
        space = cls()
        for name, intent_data in data.get('intents', {}).items():
            space.intents[name] = {
                **intent_data,
                'keywords': set(intent_data.get('keywords', [])),
            }
        return space


# Global emergent intent space (can be replaced with instance-based)
_intent_space: Optional[EmergentIntentSpace] = None

def get_intent_space() -> EmergentIntentSpace:
    """Get or create the global emergent intent space."""
    global _intent_space
    if _intent_space is None:
        _intent_space = EmergentIntentSpace()
    return _intent_space


@dataclass
class GearMessage:
    """
    Universal message format for gear communication.
    
    This is the "common language" that all gears speak. It's intentionally
    simple - just enough structure to enable clean communication without
    being overly prescriptive.
    
    Attributes:
        content: The main payload (usually text)
        source: Name of the gear that created/last modified this message
        intent: What the sender wants (query, transform, etc.)
        context: Additional structured data (original input, metadata, etc.)
        history: Chain of gears that have processed this message
        confidence: How confident the source is in this content (0.0-1.0)
        errors: Any errors encountered during processing
    """
    content: str
    source: str = ""
    intent: MessageIntent = MessageIntent.QUERY
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure history includes source if provided."""
        if self.source and (not self.history or self.history[-1] != self.source):
            self.history = list(self.history) + [self.source]
    
    @classmethod
    def from_string(cls, text: str, source: str = "user") -> 'GearMessage':
        """Create a message from a simple string."""
        return cls(content=text, source=source, context={'original_input': text})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GearMessage':
        """Create a message from a dictionary (for compatibility)."""
        content = data.get('content') or data.get('output') or data.get('input') or str(data)
        return cls(
            content=content,
            source=data.get('source', ''),
            intent=MessageIntent(data.get('intent', 'query')) if isinstance(data.get('intent'), str) else data.get('intent', MessageIntent.QUERY),
            context=data.get('context', {}),
            history=data.get('history', []),
            confidence=data.get('confidence', 1.0),
            errors=data.get('errors', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with dict-based gears."""
        return {
            'content': self.content,
            'output': self.content,  # Alias for compatibility
            'input': self.context.get('original_input', self.content),
            'source': self.source,
            'intent': self.intent.value,
            'context': self.context,
            'history': self.history,
            'confidence': self.confidence,
            'errors': self.errors,
        }
    
    def forward(self, new_source: str, new_content: str = None, 
                intent: MessageIntent = None) -> 'GearMessage':
        """
        Create a new message forwarding this one through another gear.
        
        This is the primary way gears pass messages along the chain.
        """
        return GearMessage(
            content=new_content if new_content is not None else self.content,
            source=new_source,
            intent=intent if intent is not None else self.intent,
            context={**self.context, 'previous_content': self.content},
            history=self.history + [new_source],
            confidence=self.confidence,
            errors=list(self.errors),
        )
    
    def with_error(self, error: str) -> 'GearMessage':
        """Add an error to this message."""
        return GearMessage(
            content=self.content,
            source=self.source,
            intent=self.intent,
            context=self.context,
            history=self.history,
            confidence=self.confidence * 0.5,  # Reduce confidence on error
            errors=self.errors + [error],
        )
    
    def with_context(self, key: str, value: Any) -> 'GearMessage':
        """Add context to this message."""
        new_context = {**self.context, key: value}
        return GearMessage(
            content=self.content,
            source=self.source,
            intent=self.intent,
            context=new_context,
            history=self.history,
            confidence=self.confidence,
            errors=self.errors,
        )
    
    def get_original_input(self) -> str:
        """Get the original user input from context."""
        return self.context.get('original_input', self.content)
    
    def chain_length(self) -> int:
        """How many gears have processed this message."""
        return len(self.history)
    
    def was_processed_by(self, gear_name: str) -> bool:
        """Check if a specific gear has already processed this message."""
        return gear_name in self.history
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        chain = " → ".join(self.history) if self.history else "no history"
        return f"GearMessage({self.content[:50]}... | {chain})"


def normalize_input(data: Union[str, dict, GearMessage]) -> GearMessage:
    """
    Convert any input format to a GearMessage.
    
    This is the key function for gear compatibility - it allows gears
    to accept strings, dicts, or GearMessages and always work with
    a consistent format internally.
    """
    if isinstance(data, GearMessage):
        return data
    elif isinstance(data, dict):
        return GearMessage.from_dict(data)
    elif isinstance(data, str):
        return GearMessage.from_string(data)
    else:
        return GearMessage.from_string(str(data))


def normalize_output(message: GearMessage, format: str = "message") -> Union[str, dict, GearMessage]:
    """
    Convert a GearMessage to the desired output format.
    
    Args:
        message: The message to convert
        format: One of "message", "dict", "string"
    """
    if format == "message":
        return message
    elif format == "dict":
        return message.to_dict()
    elif format == "string":
        return message.content
    else:
        return message


class GearProtocol:
    """
    Standard protocol that all gears should implement.
    
    This defines the contract for gear communication:
    1. receive() - Accept any input format, normalize to GearMessage
    2. process_message() - Core logic, GearMessage in, GearMessage out
    3. send() - Forward message to next gear
    4. process() - Compatibility wrapper returning dict
    5. chat() - Convenience wrapper for string I/O
    
    Gears can inherit from this or implement the protocol duck-typed.
    """
    name: str = "UnnamedGear"
    
    def receive(self, data: Union[str, dict, 'GearMessage', Any]) -> 'GearMessage':
        """
        Normalize any input to GearMessage.
        
        Accepts:
        - str: Plain text
        - dict: {'input': ..., 'output': ..., ...}
        - GearMessage: Pass through
        - GearState: Extract entity/metadata
        - Any: Convert to string
        """
        # Import here to avoid circular imports
        try:
            from .base import GearState
            if isinstance(data, GearState):
                msg = GearMessage(
                    content=data.entity,
                    source=self.name,
                    context={
                        'actions': data.actions,
                        'targets': data.targets,
                        'metadata': data.metadata,
                        'original_input': data.entity,
                    }
                )
                return msg
        except ImportError:
            pass
        
        msg = normalize_input(data)
        if 'original_input' not in msg.context:
            msg = msg.with_context('original_input', msg.content)
        return msg
    
    def send(self, message: 'GearMessage', content: str = None, 
             intent: MessageIntent = None) -> 'GearMessage':
        """Create outgoing message with this gear as source."""
        return message.forward(
            new_source=self.name,
            new_content=content,
            intent=intent
        )
    
    def process_message(self, message: 'GearMessage') -> 'GearMessage':
        """
        Core gear logic. Override in subclasses.
        
        Args:
            message: Normalized input message
            
        Returns:
            Processed output message
        """
        # Default: pass through unchanged
        return self.send(message)
    
    def process(self, data: Union[str, dict, 'GearMessage']) -> Dict[str, Any]:
        """
        Main entry point - handles any input format.
        
        Returns a dict for compatibility with existing gears.
        """
        message = self.receive(data)
        result = self.process_message(message)
        return result.to_dict()
    
    def chat(self, text: str) -> str:
        """Convenience method for string in, string out."""
        message = self.receive(text)
        result = self.process_message(message)
        return result.content
    
    def __call__(self, data: Union[str, dict, 'GearMessage']) -> 'GearMessage':
        """Allow gear to be called directly."""
        message = self.receive(data)
        return self.process_message(message)


# Alias for backward compatibility
MessageAwareGear = GearProtocol


def adapt_to_gear_state(message: GearMessage) -> 'Any':
    """
    Convert GearMessage to GearState for gears that use the formal protocol.
    
    This bridges the two systems.
    """
    try:
        from .base import GearState
        return GearState(
            entity=message.content,
            actions=message.context.get('actions', []),
            targets=message.context.get('targets', []),
            metadata={
                **message.context.get('metadata', {}),
                'gear_message': message.to_dict(),
            }
        )
    except ImportError:
        return message.to_dict()


def adapt_from_gear_state(state: Any, source: str = "") -> GearMessage:
    """
    Convert GearState back to GearMessage.
    """
    try:
        from .base import GearState
        if isinstance(state, GearState):
            # Check if there's a gear_message in metadata
            if 'gear_message' in state.metadata:
                base_msg = GearMessage.from_dict(state.metadata['gear_message'])
                return base_msg.forward(source, state.entity)
            
            return GearMessage(
                content=state.entity,
                source=source,
                context={
                    'actions': state.actions,
                    'targets': state.targets,
                    'metadata': state.metadata,
                }
            )
    except ImportError:
        pass
    
    if isinstance(state, dict):
        return GearMessage.from_dict(state)
    return GearMessage.from_string(str(state), source)
