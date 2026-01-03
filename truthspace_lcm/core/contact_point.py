"""
Contact Point System for Inter-Gear Communication

Inspired by chromosomal "kissing" (Non-Homologous Chromosomal Contacts),
this module provides a minimal shared vocabulary for gears to communicate.

Key insight: Gears don't need to fully understand each other - they just
need to understand the contact point vocabulary. Like chromosomes that
only interact at specific loci, not along their entire length.

The contact point vocabulary is intentionally minimal:
- Verbs: CREATE, READ, TRANSFORM, OUTPUT
- Nouns: TEXT, NUMBER, SEQUENCE, FILE
- Structure: REPEAT, BRANCH, COMPOSE

Each gear has its own rich internal representation (territory), but they
"kiss" through these shared contact points.

Reference: "Interchromosomal interactions: A genomic love story of kissing 
chromosomes" (Maass et al., 2019) - PMC6314556

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any

# Golden ratio for geometric encoding
PHI = (1 + np.sqrt(5)) / 2


class ContactVerb(Enum):
    """What action to perform - the "verb" of the contact."""
    CREATE = auto()     # Make something new
    READ = auto()       # Get existing data
    TRANSFORM = auto()  # Change/process data
    OUTPUT = auto()     # Produce result (print, return, write)


class ContactNoun(Enum):
    """What kind of data - the "noun" of the contact."""
    TEXT = auto()       # String/text data
    NUMBER = auto()     # Numeric data (int, float)
    SEQUENCE = auto()   # List/iterable/collection
    FILE = auto()       # File system object
    BOOLEAN = auto()    # True/False condition
    NONE = auto()       # No specific data type


class ContactStructure(Enum):
    """How to organize operations - the "structure" of the contact."""
    SINGLE = auto()     # Single operation
    REPEAT = auto()     # Iteration needed (for, while)
    BRANCH = auto()     # Conditional needed (if/else)
    COMPOSE = auto()    # Combine multiple operations


@dataclass
class ContactPoint:
    """
    A minimal unit of inter-gear communication.
    
    This is the "kiss" between gears - the point where their territories
    overlap and information can transfer.
    
    Attributes:
        verb: What action to perform
        noun: What kind of data is involved
        structure: How operations are organized
        params: Additional parameters (names, values, etc.)
        encoding: Geometric encoding in φ-space
    """
    verb: ContactVerb
    noun: ContactNoun = ContactNoun.NONE
    structure: ContactStructure = ContactStructure.SINGLE
    params: Dict[str, Any] = field(default_factory=dict)
    encoding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute geometric encoding after initialization."""
        if self.encoding is None:
            self.encoding = self._compute_encoding()
    
    def _compute_encoding(self) -> np.ndarray:
        """
        Encode this contact point into φ-space.
        
        The encoding uses golden ratio powers to create a unique
        geometric signature for each combination of verb/noun/structure.
        """
        # Base dimensions: verb, noun, structure
        verb_phase = self.verb.value * PHI
        noun_phase = self.noun.value * (PHI ** 2)
        struct_phase = self.structure.value * (PHI ** 3)
        
        # Create encoding vector
        encoding = np.array([
            np.cos(verb_phase),
            np.sin(verb_phase),
            np.cos(noun_phase),
            np.sin(noun_phase),
            np.cos(struct_phase),
            np.sin(struct_phase),
        ])
        
        # Normalize to unit sphere
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        
        return encoding
    
    def similarity(self, other: 'ContactPoint') -> float:
        """
        Compute similarity between two contact points.
        
        This is the "kiss strength" - how well the shapes match.
        Returns a value between -1 (opposite) and 1 (identical).
        """
        if self.encoding is None or other.encoding is None:
            return 0.0
        return float(np.dot(self.encoding, other.encoding))
    
    def kisses(self, other: 'ContactPoint', threshold: float = 0.8) -> bool:
        """
        Check if two contact points "kiss" (have matching shapes).
        
        Args:
            other: Another contact point
            threshold: Minimum similarity for a kiss (default 0.8)
        
        Returns:
            True if the contact points kiss (shapes match)
        """
        return self.similarity(other) >= threshold
    
    @classmethod
    def from_text(cls, text: str) -> 'ContactPoint':
        """
        Parse natural language into a contact point.
        
        This is how the orchestrator translates user intent into
        the shared vocabulary.
        """
        text_lower = text.lower()
        
        # Detect verb - order matters (more specific first)
        verb = ContactVerb.CREATE  # default
        if any(w in text_lower for w in ['print', 'output', 'display', 'show', 'hello']):
            verb = ContactVerb.OUTPUT
        elif any(w in text_lower for w in ['read', 'get', 'load', 'open', 'fetch']):
            verb = ContactVerb.READ
        elif any(w in text_lower for w in ['write', 'save', 'append']):
            verb = ContactVerb.OUTPUT
        elif any(w in text_lower for w in ['transform', 'convert', 'change', 'process', 'calculate', 'compute', 'sum', 'add', 'multiply']):
            verb = ContactVerb.TRANSFORM
        elif any(w in text_lower for w in ['create', 'make', 'generate', 'build', 'define']):
            verb = ContactVerb.CREATE
        
        # Detect noun
        noun = ContactNoun.NONE
        if any(w in text_lower for w in ['string', 'text', 'word', 'sentence', 'message']):
            noun = ContactNoun.TEXT
        elif any(w in text_lower for w in ['number', 'integer', 'float', 'count', 'sum', 'total']):
            noun = ContactNoun.NUMBER
        elif any(w in text_lower for w in ['list', 'array', 'sequence', 'collection', 'items']):
            noun = ContactNoun.SEQUENCE
        elif any(w in text_lower for w in ['file', 'path', 'directory', 'folder']):
            noun = ContactNoun.FILE
        elif any(w in text_lower for w in ['true', 'false', 'boolean', 'condition', 'check']):
            noun = ContactNoun.BOOLEAN
        
        # Detect structure
        structure = ContactStructure.SINGLE
        if any(w in text_lower for w in ['each', 'every', 'all', 'loop', 'iterate', 'for each', 'repeat']):
            structure = ContactStructure.REPEAT
        elif any(w in text_lower for w in ['if', 'when', 'condition', 'check', 'whether']):
            structure = ContactStructure.BRANCH
        elif any(w in text_lower for w in ['and then', 'then', 'after', 'sequence', 'steps']):
            structure = ContactStructure.COMPOSE
        
        # Extract params (simple extraction for now)
        params = {}
        
        # Try to extract quoted strings
        import re
        quotes = re.findall(r'"([^"]*)"', text) + re.findall(r"'([^']*)'", text)
        if quotes:
            params['values'] = quotes
            # Quoted strings are TEXT
            if noun == ContactNoun.NONE:
                noun = ContactNoun.TEXT
        
        # Try to extract numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            params['numbers'] = [float(n) if '.' in n else int(n) for n in numbers]
        
        # For OUTPUT verb, try to extract the full message to output
        if verb == ContactVerb.OUTPUT and not quotes:
            output_message = cls._extract_output_message(text)
            if output_message:
                params['values'] = [output_message]
                if noun == ContactNoun.NONE:
                    noun = ContactNoun.TEXT
        
        # Detect unknown concepts (proper nouns, capitalized phrases not in our vocabulary)
        # These are potential placeholder concepts
        unknown_concepts = cls._extract_unknown_concepts(text)
        if unknown_concepts:
            params['unknown_concepts'] = unknown_concepts
            # If we have unknown concepts and no values yet, use them
            if noun == ContactNoun.NONE:
                noun = ContactNoun.TEXT
            if 'values' not in params:
                params['values'] = unknown_concepts
        
        return cls(verb=verb, noun=noun, structure=structure, params=params)
    
    @staticmethod
    def _extract_output_message(text: str) -> Optional[str]:
        """
        Extract the message to output from text like:
        - "prints hello George Washington"
        - "says goodbye"
        - "outputs the result"
        
        Returns the extracted message or None.
        """
        import re
        text_lower = text.lower()
        
        # Patterns for extracting what to print/output
        patterns = [
            r'prints?\s+(.+?)(?:\s*$|\s+and\s+|\s+then\s+)',
            r'says?\s+(.+?)(?:\s*$|\s+and\s+|\s+then\s+)',
            r'outputs?\s+(.+?)(?:\s*$|\s+and\s+|\s+then\s+)',
            r'displays?\s+(.+?)(?:\s*$|\s+and\s+|\s+then\s+)',
            r'shows?\s+(.+?)(?:\s*$|\s+and\s+|\s+then\s+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                message = match.group(1).strip()
                # Clean up common trailing words
                message = re.sub(r'\s+(to|into|in|on)\s+\w+$', '', message)
                if message and len(message) > 1:
                    return message
        
        return None
    
    @staticmethod
    def _extract_unknown_concepts(text: str) -> List[str]:
        """
        Extract potential unknown concepts from text.
        
        These are typically:
        - Capitalized words/phrases (proper nouns)
        - Quoted strings
        - Words that don't match our vocabulary
        
        Returns list of unknown concept strings.
        """
        import re
        
        # Known vocabulary words (stop words + our vocabulary)
        known_words = {
            # Stop words
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            # Programming/action words
            'write', 'create', 'make', 'generate', 'build', 'print', 'output',
            'display', 'show', 'read', 'get', 'load', 'open', 'fetch', 'save',
            'file', 'program', 'code', 'script', 'function', 'python', 'hello',
            'world', 'string', 'text', 'number', 'integer', 'list', 'array',
            'loop', 'iterate', 'count', 'calculate', 'sum', 'add', 'multiply',
            'that', 'prints', 'says', 'outputs', 'displays', 'returns',
        }
        
        unknown = []
        
        # Find capitalized phrases (2+ words starting with capitals)
        # e.g., "George Washington", "New York"
        cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for phrase in cap_phrases:
            if phrase.lower() not in known_words:
                unknown.append(phrase)
        
        # Find single capitalized words that aren't at sentence start
        # and aren't in our vocabulary
        words = text.split()
        for i, word in enumerate(words):
            # Skip first word (might just be sentence start)
            if i == 0:
                continue
            # Check if capitalized and not known
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and clean_word.lower() not in known_words:
                # Check it's not part of an already-found phrase
                if not any(clean_word in phrase for phrase in unknown):
                    unknown.append(clean_word)
        
        return unknown
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'verb': self.verb.name,
            'noun': self.noun.name,
            'structure': self.structure.name,
            'params': self.params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContactPoint':
        """Create from dictionary."""
        return cls(
            verb=ContactVerb[data['verb']],
            noun=ContactNoun[data['noun']],
            structure=ContactStructure[data['structure']],
            params=data.get('params', {}),
        )
    
    def __str__(self) -> str:
        parts = [self.verb.name]
        if self.noun != ContactNoun.NONE:
            parts.append(self.noun.name)
        if self.structure != ContactStructure.SINGLE:
            parts.append(f"[{self.structure.name}]")
        if self.params:
            parts.append(str(self.params))
        return ' '.join(parts)


@dataclass
class ContactMessage:
    """
    A message passed between gears through contact points.
    
    This is the full "kiss" - the contact point plus any additional
    context needed for the receiving gear.
    """
    contact: ContactPoint
    source_gear: str
    target_gear: str
    context: Dict[str, Any] = field(default_factory=dict)
    response: Optional[str] = None
    success: bool = False
    
    def with_response(self, response: str, success: bool = True) -> 'ContactMessage':
        """Create a response message."""
        return ContactMessage(
            contact=self.contact,
            source_gear=self.target_gear,  # Swap source/target
            target_gear=self.source_gear,
            context=self.context,
            response=response,
            success=success,
        )


class ContactRegistry:
    """
    Registry of contact points that gears can understand.
    
    This is like the "transcription factory" in the nucleus -
    a place where gears can register what contacts they respond to.
    """
    
    def __init__(self):
        self.handlers: Dict[str, Dict[Tuple[ContactVerb, ContactNoun], callable]] = {}
    
    def register(self, gear_name: str, verb: ContactVerb, noun: ContactNoun, 
                 handler: callable):
        """Register a handler for a specific contact type."""
        if gear_name not in self.handlers:
            self.handlers[gear_name] = {}
        self.handlers[gear_name][(verb, noun)] = handler
    
    def find_handler(self, gear_name: str, contact: ContactPoint) -> Optional[callable]:
        """Find a handler for a contact point."""
        if gear_name not in self.handlers:
            return None
        
        # Exact match
        key = (contact.verb, contact.noun)
        if key in self.handlers[gear_name]:
            return self.handlers[gear_name][key]
        
        # Try with NONE noun (generic handler)
        key = (contact.verb, ContactNoun.NONE)
        if key in self.handlers[gear_name]:
            return self.handlers[gear_name][key]
        
        return None
    
    def can_handle(self, gear_name: str, contact: ContactPoint) -> bool:
        """Check if a gear can handle a contact point."""
        return self.find_handler(gear_name, contact) is not None


# Global registry for contact points
_global_registry = ContactRegistry()


def get_registry() -> ContactRegistry:
    """Get the global contact registry."""
    return _global_registry


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_intent(text: str) -> ContactPoint:
    """Parse natural language into a contact point."""
    return ContactPoint.from_text(text)


def contact_similarity(a: ContactPoint, b: ContactPoint) -> float:
    """Compute similarity between two contact points."""
    return a.similarity(b)


def contacts_kiss(a: ContactPoint, b: ContactPoint, threshold: float = 0.8) -> bool:
    """Check if two contact points kiss."""
    return a.kisses(b, threshold)


# =============================================================================
# PREDEFINED CONTACT POINTS (common patterns)
# =============================================================================

# Common contact points that gears might use
CONTACTS = {
    # Creation
    'create_text': ContactPoint(ContactVerb.CREATE, ContactNoun.TEXT),
    'create_number': ContactPoint(ContactVerb.CREATE, ContactNoun.NUMBER),
    'create_sequence': ContactPoint(ContactVerb.CREATE, ContactNoun.SEQUENCE),
    'create_file': ContactPoint(ContactVerb.CREATE, ContactNoun.FILE),
    
    # Reading
    'read_text': ContactPoint(ContactVerb.READ, ContactNoun.TEXT),
    'read_number': ContactPoint(ContactVerb.READ, ContactNoun.NUMBER),
    'read_sequence': ContactPoint(ContactVerb.READ, ContactNoun.SEQUENCE),
    'read_file': ContactPoint(ContactVerb.READ, ContactNoun.FILE),
    
    # Transformation
    'transform_text': ContactPoint(ContactVerb.TRANSFORM, ContactNoun.TEXT),
    'transform_number': ContactPoint(ContactVerb.TRANSFORM, ContactNoun.NUMBER),
    'transform_sequence': ContactPoint(ContactVerb.TRANSFORM, ContactNoun.SEQUENCE),
    
    # Output
    'output_text': ContactPoint(ContactVerb.OUTPUT, ContactNoun.TEXT),
    'output_number': ContactPoint(ContactVerb.OUTPUT, ContactNoun.NUMBER),
    'output_sequence': ContactPoint(ContactVerb.OUTPUT, ContactNoun.SEQUENCE),
    
    # Iteration patterns
    'iterate_sequence': ContactPoint(ContactVerb.TRANSFORM, ContactNoun.SEQUENCE, ContactStructure.REPEAT),
    'iterate_file': ContactPoint(ContactVerb.READ, ContactNoun.FILE, ContactStructure.REPEAT),
    
    # Conditional patterns
    'branch_boolean': ContactPoint(ContactVerb.TRANSFORM, ContactNoun.BOOLEAN, ContactStructure.BRANCH),
}


if __name__ == "__main__":
    # Test the contact point system
    print("=== Contact Point System Test ===\n")
    
    # Test parsing
    test_phrases = [
        "create a function that prints hello",
        "read a file and print each line",
        "calculate the sum of numbers in a list",
        "if the number is greater than 10, print it",
        "write 'hello world' to output.txt",
    ]
    
    print("Parsing test phrases:")
    for phrase in test_phrases:
        contact = ContactPoint.from_text(phrase)
        print(f"  '{phrase[:40]}...'")
        print(f"    → {contact}")
        print()
    
    # Test similarity
    print("Similarity tests:")
    c1 = ContactPoint.from_text("print a message")
    c2 = ContactPoint.from_text("output some text")
    c3 = ContactPoint.from_text("read a file")
    
    print(f"  'print a message' vs 'output some text': {c1.similarity(c2):.3f}")
    print(f"  'print a message' vs 'read a file': {c1.similarity(c3):.3f}")
    print(f"  Kiss threshold 0.8: {c1.kisses(c2)} / {c1.kisses(c3)}")
    
    # Test predefined contacts
    print("\nPredefined contacts:")
    for name, contact in list(CONTACTS.items())[:5]:
        print(f"  {name}: {contact}")
