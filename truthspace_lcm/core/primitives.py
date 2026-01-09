"""
Primitives for φ-Lattice Encoding

Primitives are semantic anchors that map words to φ-lattice positions.
Inspired by the old TruthSpace implementation (temp/old_core/truthspace.py).

Each primitive has:
- name: Identifier
- dimension: Which semantic dimension it activates
- level: What φ-level it activates
- keywords: Words that trigger this primitive

Design Principles:
- Primitives are bootstrapped (initial seed)
- They are immediately transformed to geometry
- The keywords are just the seed - positions are what matter
- MAX aggregation per dimension (Sierpinski property)

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class Primitive:
    """
    A semantic anchor in the φ-lattice.
    
    Maps keywords to a specific (dimension, level) position.
    """
    name: str
    dimension: int
    level: int
    keywords: List[str]
    
    def __hash__(self):
        return hash(self.name)


# =============================================================================
# DOMAIN PRIMITIVES (dimension 0)
# What area of knowledge?
# =============================================================================

DOMAIN_PRIMITIVES = [
    # Hard science (level 3-4)
    Primitive("PHYSICS", 0, 3, [
        "physics", "quantum", "relativity", "mechanics", "thermodynamics",
        "electromagnetism", "particle", "wave", "energy", "force"
    ]),
    Primitive("MATH", 0, 3, [
        "math", "mathematics", "calculus", "algebra", "geometry",
        "theorem", "proof", "equation", "formula", "number"
    ]),
    Primitive("CHEMISTRY", 0, 3, [
        "chemistry", "chemical", "molecule", "atom", "reaction",
        "element", "compound", "bond"
    ]),
    Primitive("BIOLOGY", 0, 3, [
        "biology", "cell", "dna", "gene", "organism", "evolution"
    ]),
    
    # Technology (level 2)
    Primitive("PROGRAMMING", 0, 2, [
        "programming", "code", "python", "software", "algorithm",
        "function", "variable", "class", "method", "api"
    ]),
    Primitive("TECHNOLOGY", 0, 2, [
        "technology", "computer", "digital", "system", "network",
        "data", "machine", "artificial", "intelligence"
    ]),
    Primitive("ENGINEERING", 0, 2, [
        "engineering", "design", "build", "construct", "develop"
    ]),
    
    # General knowledge (level 1)
    Primitive("GENERAL_KNOWLEDGE", 0, 1, [
        "knowledge", "information", "learn", "understand", "know",
        "fact", "topic", "subject", "area"
    ]),
    Primitive("SCIENCE", 0, 1, [
        "science", "scientific", "research", "study", "experiment"
    ]),
    
    # Meta/Identity (level 0)
    Primitive("IDENTITY", 0, 0, [
        "you", "your", "yourself", "hyperchat", "assistant",
        "who", "name", "chatbot", "bot", "ai"
    ]),
    Primitive("CAPABILITY", 0, 0, [
        "able", "capability", "help", "assist", "abilities"
    ]),
    
    # Social (level -1)
    Primitive("GREETING", 0, -1, [
        "hello", "hi", "hey", "greetings", "howdy", "morning",
        "afternoon", "evening"
    ]),
    Primitive("FAREWELL", 0, -1, [
        "goodbye", "bye", "farewell", "later", "see"
    ]),
    Primitive("THANKS", 0, -1, [
        "thanks", "thank", "appreciate", "grateful"
    ]),
    
    # Filler (level -2)
    Primitive("ACKNOWLEDGMENT", 0, -2, [
        "ok", "okay", "alright", "sure", "yes", "yeah", "yep",
        "got", "understood", "right"
    ]),
]

# =============================================================================
# SPECIFICITY PRIMITIVES (dimension 1)
# How specific is the concept?
# =============================================================================

SPECIFICITY_PRIMITIVES = [
    # Very specific (level 3-4)
    Primitive("HIGHLY_SPECIFIC", 1, 4, [
        "specifically", "exactly", "precisely", "particular"
    ]),
    Primitive("VERY_SPECIFIC", 1, 3, [
        "quantum", "neural", "differential", "integral",
        "recursive", "polymorphic", "asynchronous"
    ]),
    
    # Specific (level 2)
    Primitive("SPECIFIC", 1, 2, [
        "physics", "python", "machine", "learning", "network",
        "algorithm", "function", "class"
    ]),
    
    # General (level 1)
    Primitive("GENERAL", 1, 1, [
        "science", "programming", "technology", "knowledge",
        "information", "concept", "idea"
    ]),
    
    # Very general (level 0)
    Primitive("VERY_GENERAL", 1, 0, [
        "what", "how", "why", "explain", "tell", "about",
        "thing", "stuff"
    ]),
    
    # Vague (level -1)
    Primitive("VAGUE", 1, -1, [
        "something", "anything", "whatever", "somehow"
    ]),
]

# =============================================================================
# INTENT PRIMITIVES (dimension 2)
# What kind of response is expected?
# =============================================================================

INTENT_PRIMITIVES = [
    # Deep explanation (level 3)
    Primitive("DEEP_EXPLAIN", 2, 3, [
        "derive", "prove", "demonstrate", "detail", "thoroughly"
    ]),
    
    # Explanation (level 2)
    Primitive("EXPLAIN", 2, 2, [
        "explain", "describe", "teach", "how", "why",
        "work", "works", "meaning"
    ]),
    
    # Information (level 1)
    Primitive("INFORM", 2, 1, [
        "what", "tell", "show", "is", "are", "define",
        "definition", "mean", "means"
    ]),
    
    # Acknowledgment (level 0)
    Primitive("ACKNOWLEDGE", 2, 0, [
        "ok", "yes", "sure", "got", "understood", "right"
    ]),
    
    # Social response (level -1)
    Primitive("SOCIAL_RESPONSE", 2, -1, [
        "hello", "hi", "thanks", "bye", "goodbye"
    ]),
]

# =============================================================================
# FORMALITY PRIMITIVES (dimension 3)
# How formal is the context?
# =============================================================================

FORMALITY_PRIMITIVES = [
    # Academic (level 2)
    Primitive("ACADEMIC", 3, 2, [
        "theory", "hypothesis", "analysis", "methodology",
        "furthermore", "therefore", "consequently"
    ]),
    
    # Professional (level 1)
    Primitive("PROFESSIONAL", 3, 1, [
        "please", "could", "would", "kindly", "request",
        "assist", "provide"
    ]),
    
    # Casual (level 0)
    Primitive("CASUAL", 3, 0, [
        "can", "want", "need", "like", "get", "make"
    ]),
    
    # Informal (level -1)
    Primitive("INFORMAL", 3, -1, [
        "hey", "yo", "cool", "awesome", "gonna", "wanna"
    ]),
]

# =============================================================================
# INTRINSIC/FUNCTIONAL PRIMITIVES (dimension 4)
# What type of knowledge: structural properties vs relational/purpose
# Design 047: Discovered through pure geometric matching
# =============================================================================

INTRINSIC_FUNCTIONAL_PRIMITIVES = [
    # Strongly functional (level 2) - context, history, purpose
    Primitive("STRONGLY_FUNCTIONAL", 4, 2, [
        "created", "invented", "founded", "developed", "designed",
        "history", "origin", "background", "context"
    ]),
    
    # Functional (level 1) - purpose, known as, used for
    Primitive("FUNCTIONAL", 4, 1, [
        "known", "called", "named", "purpose", "used", "for",
        "designed", "meant", "intended", "serves"
    ]),
    
    # Balanced (level 0) - neutral
    Primitive("BALANCED", 4, 0, [
        "about", "regarding", "concerning"
    ]),
    
    # Intrinsic (level -1) - properties, structure
    Primitive("INTRINSIC", 4, -1, [
        "is", "has", "contains", "includes", "features",
        "properties", "characteristics", "attributes"
    ]),
    
    # Strongly intrinsic (level -2) - raw structure, composition
    Primitive("STRONGLY_INTRINSIC", 4, -2, [
        "made", "consists", "composed", "built", "comprises",
        "structure", "components", "parts"
    ]),
]

# =============================================================================
# ACTION PRIMITIVES (dimension 5)
# User-agent relationship: query vs create vs execute
# Design 048: Agent-grounded interpretation
# =============================================================================

ACTION_PRIMITIVES = [
    # Strongly query (level -2) - deep questions about concepts
    Primitive("STRONGLY_QUERY", 5, -2, [
        "what", "why", "how", "when", "where", "who"
    ]),
    
    # Query (level -1) - requests for explanation/description
    Primitive("QUERY", 5, -1, [
        "explain", "describe", "tell", "define", "meaning",
        "understand", "clarify", "elaborate"
    ]),
    
    # Create (level +1) - requests to generate output
    Primitive("CREATE", 5, 1, [
        "create", "make", "generate", "plot", "draw", "build",
        "write", "produce", "design", "construct"
    ]),
    
    # Execute (level +2) - requests to perform tool actions
    Primitive("EXECUTE", 5, 2, [
        "list", "read", "show", "find", "search", "run",
        "execute", "delete", "remove", "open", "close",
        "start", "stop", "install", "update"
    ]),
]

# =============================================================================
# ALL PRIMITIVES
# =============================================================================

ALL_PRIMITIVES = (
    DOMAIN_PRIMITIVES +
    SPECIFICITY_PRIMITIVES +
    INTENT_PRIMITIVES +
    FORMALITY_PRIMITIVES +
    INTRINSIC_FUNCTIONAL_PRIMITIVES +
    ACTION_PRIMITIVES
)


def build_keyword_map() -> Dict[str, Primitive]:
    """
    Build keyword → primitive mapping.
    
    If multiple primitives have the same keyword, prefer the one
    with higher level (more specific meaning).
    """
    keyword_map = {}
    for prim in ALL_PRIMITIVES:
        for kw in prim.keywords:
            kw_lower = kw.lower()
            if kw_lower not in keyword_map or prim.level > keyword_map[kw_lower].level:
                keyword_map[kw_lower] = prim
    return keyword_map


def get_primitives_for_dimension(dimension: int) -> List[Primitive]:
    """Get all primitives for a specific dimension."""
    return [p for p in ALL_PRIMITIVES if p.dimension == dimension]


def get_keywords_for_level(dimension: int, level: int) -> Set[str]:
    """Get all keywords that map to a specific (dimension, level)."""
    keywords = set()
    for prim in ALL_PRIMITIVES:
        if prim.dimension == dimension and prim.level == level:
            keywords.update(kw.lower() for kw in prim.keywords)
    return keywords


# Prebuilt keyword map for efficiency
KEYWORD_MAP = build_keyword_map()
