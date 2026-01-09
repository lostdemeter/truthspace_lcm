"""
Perspective System for HyperChat (Design 111 + 112)

Perspectives are delta vectors that shift word positions in semantic space.
Output emerges from find_nearest(position + delta) - the Music Box Principle.

No word->word mappings. The music emerges from the geometry.

Usage:
    from truthspace_lcm.core.perspective_geometric import GeometricPerspective, PERSPECTIVES
    
    perspective = PERSPECTIVES['warhammer40k']
    styled_response = perspective.transform_response(base_response)

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from .geometric_vocabulary import (
    GeometricVocabulary, 
    get_default_vocabulary, 
    get_perspective_delta,
    PERSPECTIVE_DELTAS
)


@dataclass
class GeometricPerspective:
    """
    A perspective is a delta vector in semantic space (Design 112).
    
    The delta shifts word positions. Output emerges from find_nearest.
    No word->word mappings - the music emerges from the geometry.
    
    The prefix/suffix are kept for framing but could also be geometric
    in a more complete implementation.
    """
    name: str
    description: str
    delta: np.ndarray  # [tense, formality, domain, intensity]
    prefix: str = ""
    suffix: str = ""
    
    def transform_response(self, response: str, vocab: Optional[GeometricVocabulary] = None) -> str:
        """
        Transform response text using geometric vocabulary lookup.
        
        For each word:
        1. Get word's position (read the drum)
        2. Apply delta (rotation of the drum)
        3. Find nearest word at new position (comb produces sound)
        
        The music emerges from the geometry.
        """
        if vocab is None:
            vocab = get_default_vocabulary()
        
        # Skip transformation if delta is zero
        if np.allclose(self.delta, 0):
            result = response
        else:
            # Transform each word
            result = self._transform_text(response, vocab)
        
        # Add prefix if specified
        if self.prefix:
            result = f"{self.prefix}\n\n{result}"
        
        # Add suffix if specified
        if self.suffix:
            result = f"{result}\n\n{self.suffix}"
        
        return result
    
    def _transform_text(self, text: str, vocab: GeometricVocabulary) -> str:
        """Transform text word by word using geometric lookup."""
        # Split into words while preserving punctuation and whitespace
        tokens = re.findall(r'\b[\w\'-]+\b|[^\w\s]+|\s+', text)
        
        result = []
        for token in tokens:
            # Skip whitespace and punctuation
            if not token.strip() or not token[0].isalnum():
                result.append(token)
                continue
            
            # Try to transform the word
            transformed = vocab.transform(token, self.delta)
            
            if transformed and transformed.lower() != token.lower():
                # Preserve original capitalization pattern
                if token.isupper():
                    transformed = transformed.upper()
                elif token[0].isupper():
                    transformed = transformed.capitalize()
                result.append(transformed)
            else:
                result.append(token)
        
        return ''.join(result)


# =============================================================================
# PREDEFINED PERSPECTIVES (using deltas, not word mappings)
# =============================================================================

DEFAULT_PERSPECTIVE = GeometricPerspective(
    name="default",
    description="Standard HyperChat assistant - helpful, clear, technical",
    delta=np.array([0, 0, 0, 0]),
    prefix="",
    suffix="",
)

WARHAMMER_40K_PERSPECTIVE = GeometricPerspective(
    name="warhammer40k",
    description="Grimdark Warhammer 40,000 narrator - dramatic, gothic, zealous",
    delta=np.array([0, 2, 2, 0.5]),  # archaic + sacred + intensity
    prefix="*In the grim darkness of the far future, there is only code...*\n\nHearken, supplicant, to the sacred knowledge:",
    suffix="*The Omnissiah protects. The Machine God provides.*",
)

PIRATE_PERSPECTIVE = GeometricPerspective(
    name="pirate",
    description="Swashbuckling pirate captain - nautical, adventurous, colorful",
    delta=np.array([0, -1, -1, 0]),  # casual + mundane
    prefix="Ahoy there, matey!",
    suffix="Now ye know, savvy? Fair winds to ye!",
)

SHAKESPEARE_PERSPECTIVE = GeometricPerspective(
    name="shakespeare",
    description="Elizabethan playwright - poetic, dramatic, archaic",
    delta=np.array([0, 2, 0, 0]),  # archaic formality
    prefix="Hark! Attend well to these words of wisdom:",
    suffix="*Exeunt*",
)

# Registry of all perspectives
PERSPECTIVES: Dict[str, GeometricPerspective] = {
    "default": DEFAULT_PERSPECTIVE,
    "warhammer40k": WARHAMMER_40K_PERSPECTIVE,
    "wh40k": WARHAMMER_40K_PERSPECTIVE,
    "grimdark": WARHAMMER_40K_PERSPECTIVE,
    "pirate": PIRATE_PERSPECTIVE,
    "shakespeare": SHAKESPEARE_PERSPECTIVE,
    "bard": SHAKESPEARE_PERSPECTIVE,
}


def get_perspective(name: str) -> GeometricPerspective:
    """Get a perspective by name, defaulting to DEFAULT if not found."""
    return PERSPECTIVES.get(name.lower(), DEFAULT_PERSPECTIVE)


def list_perspectives() -> List[str]:
    """List all available perspective names."""
    seen = set()
    result = []
    for name, persp in PERSPECTIVES.items():
        if persp.name not in seen:
            result.append(f"- **{persp.name}**: {persp.description}")
            seen.add(persp.name)
    return result
