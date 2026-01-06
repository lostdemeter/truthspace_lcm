"""
Semantic Dimension Definitions for Knowledge Matching

Defines the semantic axes of the φ-lattice:
- DOMAIN: What area of knowledge
- SPECIFICITY: How specific is the concept
- INTENT: What kind of response expected
- FORMALITY: How formal is the context

Each dimension has φ-levels with semantic meanings.

Design Principles (from Design 099):
- Dimensions have clear semantic interpretation
- Levels map to φ^k values (absolute coordinates)
- Weights follow old TruthSpace pattern (actions > domains > relations)

Author: Lesley Gushurst
License: GPLv3
"""

from .phi_lattice import SemanticDimension, PHI

# =============================================================================
# DOMAIN DIMENSION (index 0)
# What area of knowledge does this concept belong to?
# =============================================================================

DOMAIN = SemanticDimension(
    index=0,
    name='domain',
    description='What area of knowledge',
    level_meanings={
        4: 'specialized_science',   # Quantum mechanics, topology
        3: 'hard_science',          # Physics, Math, Chemistry
        2: 'technology',            # Programming, Engineering
        1: 'general_knowledge',     # General facts, trivia
        0: 'meta',                  # Identity, self-reference, capabilities
        -1: 'social',               # Greetings, thanks, farewells
        -2: 'filler',               # Acknowledgments, fillers
    },
    weight=PHI ** 2  # Highest weight - domain is most important
)

# =============================================================================
# SPECIFICITY DIMENSION (index 1)
# How specific or general is the concept?
# =============================================================================

SPECIFICITY = SemanticDimension(
    index=1,
    name='specificity',
    description='How specific is the concept',
    level_meanings={
        4: 'highly_specific',       # Specific theorem, exact algorithm
        3: 'very_specific',         # Quantum mechanics, neural networks
        2: 'specific',              # Physics, Python, machine learning
        1: 'general',               # Science, programming, learning
        0: 'very_general',          # Knowledge, information, things
        -1: 'vague',                # Something, anything, stuff
    },
    weight=PHI  # Second highest weight
)

# =============================================================================
# INTENT DIMENSION (index 2)
# What kind of response is expected?
# =============================================================================

INTENT = SemanticDimension(
    index=2,
    name='intent',
    description='What kind of response is expected',
    level_meanings={
        3: 'deep_explanation',      # Detailed teaching, derivation
        2: 'explanation',           # Explain, describe, how does X work
        1: 'information',           # What is X, tell me about X
        0: 'acknowledgment',        # Confirm, acknowledge, ok
        -1: 'social_response',      # Greeting response, thanks response
    },
    weight=1.0  # Neutral weight
)

# =============================================================================
# FORMALITY DIMENSION (index 3)
# How formal is the context?
# =============================================================================

FORMALITY = SemanticDimension(
    index=3,
    name='formality',
    description='How formal is the context',
    level_meanings={
        2: 'academic',              # Technical, formal, precise
        1: 'professional',          # Business, clear, polite
        0: 'casual',                # Everyday, relaxed
        -1: 'informal',             # Friendly, chatty
        -2: 'very_informal',        # Slang, abbreviations
    },
    weight=PHI ** -1  # Lowest weight - formality matters least
)

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_DIMENSIONS = [DOMAIN, SPECIFICITY, INTENT, FORMALITY]

DEFAULT_DIMENSION_COUNT = len(DEFAULT_DIMENSIONS)

# Weights as array for distance calculations
DEFAULT_WEIGHTS = [d.weight for d in DEFAULT_DIMENSIONS]

# Quick lookup by name
DIMENSION_BY_NAME = {d.name: d for d in DEFAULT_DIMENSIONS}

# Quick lookup by index
DIMENSION_BY_INDEX = {d.index: d for d in DEFAULT_DIMENSIONS}


def get_dimension(name_or_index) -> SemanticDimension:
    """Get a dimension by name or index."""
    if isinstance(name_or_index, str):
        return DIMENSION_BY_NAME.get(name_or_index)
    return DIMENSION_BY_INDEX.get(name_or_index)


def describe_levels(levels: list) -> dict:
    """Get semantic description for a list of levels."""
    return {
        DEFAULT_DIMENSIONS[i].name: DEFAULT_DIMENSIONS[i].get_meaning(level)
        for i, level in enumerate(levels)
        if i < len(DEFAULT_DIMENSIONS)
    }
