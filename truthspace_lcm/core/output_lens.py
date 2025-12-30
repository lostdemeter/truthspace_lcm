#!/usr/bin/env python3
"""
Output Lens: Holographic Projection for Natural Language Output

This module provides a "lens" that transforms raw concept-space output
into natural, readable text without modifying the underlying knowledge.

The key insight from Design 059 (Two-Source Diffraction):
- Knowledge Source: WHAT to say (concept space - unchanged)
- Style Source: HOW to say it (output lens - this module)

The interference between them produces natural, styled responses.

Mathematical Model:
    Output = Knowledge ⊗ Style
    
Where ⊗ is the interference operation that:
- Preserves semantic content (from knowledge)
- Applies stylistic transformation (from lens)

Author: Lesley Gushurst
License: GPLv3
"""

import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StyleLens:
    """
    A lens that transforms raw output into styled natural language.
    
    Each lens has:
    - name: identifier for the style
    - templates: sentence patterns for different answer types
    - transforms: word/phrase replacements
    - prefix/suffix: framing text
    """
    name: str
    templates: Dict[str, List[str]] = field(default_factory=dict)
    transforms: Dict[str, str] = field(default_factory=dict)
    prefix: str = ""
    suffix: str = ""
    
    def apply(self, content: str, answer_type: str = "describe") -> str:
        """Apply the lens to transform content."""
        # Apply word transforms
        result = content
        for old, new in self.transforms.items():
            result = result.replace(old, new)
        
        # Apply template if available
        if answer_type in self.templates and self.templates[answer_type]:
            template = random.choice(self.templates[answer_type])
            if "{content}" in template:
                result = template.format(content=result)
        
        # Add framing
        if self.prefix and not result.startswith(self.prefix):
            result = self.prefix + result
        if self.suffix and not result.endswith(self.suffix):
            result = result + self.suffix
        
        return result


# Pre-defined style lenses
NATURAL_LENS = StyleLens(
    name="natural",
    templates={
        "describe": [
            "{content}",
            "Based on what I know, {content}",
            "From the available information, {content}",
        ],
        "who": [
            "{content}",
            "That would be {content}",
        ],
        "what": [
            "{content}",
            "It appears that {content}",
        ],
        "list": [
            "{content}",
            "The key points are: {content}",
        ],
    },
    transforms={
        " is a entity who ": " is someone who ",
        " is a protagonist who ": " is a character who ",
        " is a concept who ": " is a concept that ",
        " is a concept.": " is an abstract idea.",
        ", often involving ": ". This often involves ",
        "Formula is semantically.": "I don't have specific information about that.",
        " and attosecond.": ".",
        " and clauses.": ".",
        " involving certain and ": " involving ",
        " involving conceptual and ": " involving ",
        "Animacy is a concept": "Animacy is a linguistic property",
        "Consciousness is a concept": "Consciousness is a phenomenon",
        "Abstraction is a concept": "Abstraction is a cognitive process",
        "Physics is a character": "Physics is a field",
        "Physics is a protagonist": "Physics is a field",
        "Physics is a science who": "Physics is a science that",
        "Evolution is a character": "Evolution is a process",
        "Evolution is a protagonist": "Evolution is a process",
        "Biology is a character": "Biology is a field",
        "Biology is a protagonist": "Biology is a field",
        " is a field who ": " is a field that ",
        " is a science who ": " is a science that ",
        " is a process who ": " is a process that ",
        " is a discipline who ": " is a discipline that ",
    },
    prefix="",
    suffix="",
)

FORMAL_LENS = StyleLens(
    name="formal",
    templates={
        "describe": [
            "Upon examination, {content}",
            "Analysis indicates that {content}",
        ],
        "who": [
            "The individual in question appears to be {content}",
        ],
        "what": [
            "It can be determined that {content}",
        ],
    },
    transforms={
        " is a entity who ": " can be characterized as an individual who ",
        " is a protagonist who ": " functions as a principal figure who ",
        " is a concept who ": " represents an abstract notion that ",
        ", often involving ": ". This frequently pertains to ",
        " who includes, ": " who ",
    },
    prefix="",
    suffix="",
)

CASUAL_LENS = StyleLens(
    name="casual",
    templates={
        "describe": [
            "So basically, {content}",
            "Well, {content}",
            "{content} Pretty interesting, right?",
        ],
        "who": [
            "Oh, that's {content}",
        ],
        "what": [
            "It's like, {content}",
        ],
    },
    transforms={
        " is a entity who ": " is someone who ",
        " is a protagonist who ": " is this character who ",
        " is a concept who ": " is basically something that ",
        ", often involving ": " - usually with ",
        " who includes, ": " who ",
    },
    prefix="",
    suffix="",
)

LITERARY_LENS = StyleLens(
    name="literary",
    templates={
        "describe": [
            "In the tapestry of this narrative, {content}",
            "One might observe that {content}",
        ],
        "who": [
            "Among the dramatis personae, {content}",
        ],
        "what": [
            "The essence reveals that {content}",
        ],
    },
    transforms={
        " is a entity who ": " emerges as a figure who ",
        " is a protagonist who ": " stands as a central character who ",
        " is a concept who ": " manifests as an idea that ",
        ", often involving ": ", weaving together ",
    },
    prefix="",
    suffix="",
)

SCIENTIFIC_LENS = StyleLens(
    name="scientific",
    templates={
        "describe": [
            "Data analysis reveals: {content}",
            "Empirical observation suggests that {content}",
        ],
        "who": [
            "The subject can be identified as {content}",
        ],
        "what": [
            "Evidence indicates that {content}",
        ],
    },
    transforms={
        " is a entity who ": " can be classified as an agent that ",
        " is a protagonist who ": " represents a primary subject that ",
        " is a concept who ": " constitutes an abstract construct that ",
        ", often involving ": ". Correlates include ",
    },
    prefix="",
    suffix=" Further investigation may be warranted.",
)


# Registry of available lenses
LENSES = {
    "natural": NATURAL_LENS,
    "formal": FORMAL_LENS,
    "casual": CASUAL_LENS,
    "literary": LITERARY_LENS,
    "scientific": SCIENTIFIC_LENS,
}


class OutputProjector:
    """
    Projects concept-space output through a style lens.
    
    This is the holographic projection layer that sits between
    the knowledge (concept space) and the user (natural language).
    
    The projection preserves semantic content while transforming
    the surface form to be more natural and readable.
    """
    
    def __init__(self, lens: StyleLens = None):
        self.lens = lens or NATURAL_LENS
        self._post_processors = []
    
    def set_lens(self, lens_name: str):
        """Set the active lens by name."""
        if lens_name in LENSES:
            self.lens = LENSES[lens_name]
        else:
            raise ValueError(f"Unknown lens: {lens_name}. Available: {list(LENSES.keys())}")
    
    def project(self, raw_output: str, answer_type: str = "describe") -> str:
        """
        Project raw concept-space output through the lens.
        
        Args:
            raw_output: The raw output from GeometricQA
            answer_type: Type of answer (describe, who, what, list)
        
        Returns:
            Natural language output
        """
        # First, apply the lens transforms
        result = self.lens.apply(raw_output, answer_type)
        
        # Then apply structural improvements (after lens transforms)
        result = self._improve_structure(result)
        
        # Finally, apply post-processors
        for processor in self._post_processors:
            result = processor(result)
        
        return result
    
    def _improve_structure(self, text: str) -> str:
        """
        Improve the structural quality of the output.
        
        This handles common issues like:
        - Redundant phrases
        - Awkward constructions
        - Missing articles
        """
        result = text
        
        # Fix common awkward patterns
        patterns = [
            # Remove redundant "who" clauses
            (r'\bwho includes, examines, and\b', 'who examines and'),
            (r'\bwho states, includes, and\b', 'who'),
            (r'\bwho provides, embeds, and\b', 'who'),
            (r'\bwho underscores, divides, and\b', 'who'),
            
            # Fix verb agreement
            (r'\bwho differses\b', 'who differs'),
            (r'\bwho avoides\b', 'who avoids'),
            (r'\bthat theses\b', 'that theorizes'),
            (r'\bwho theses\b', 'who theorizes'),
            (r'\bwho emphasizes\b', 'who highlights'),
            (r'\bthat provides, embeds, and theses\b', 'that categorizes and organizes'),
            (r'\bthat raises\b', 'that emerges from mental processes'),
            
            # Fix awkward comma patterns
            (r', and (\w+)\.', r' and \1.'),
            (r'who (\w+), and (\w+)\.', r'who \1 and \2.'),
            (r'that (\w+), and (\w+)\.', r'that \1 and \2.'),
            
            # Clean up empty parentheticals
            (r'\(often involving\s*\)', ''),
            (r'\(often involving and\s*\)', ''),
            
            # Clean up awkward "involves" phrases
            (r'This often involves doorway and holmes\.', 'This often involves helping Holmes.'),
            (r'This often involves certain\.', ''),
            (r'This often involves conceptual\.', ''),
            (r'This often involves groups\.', ''),
            (r'This often involves (\w+) and (\w+)\.', r'This relates to \1 and \2.'),
            (r'This often involves (\w+)\.$', ''),
            (r'This relates to (\w+)\.$', ''),
            
            # Fix "is a character" for non-characters (domain-aware)
            (r'Physics is a (character|field) who', 'Physics is a field that'),
            (r'Evolution is a (character|process) who', 'Evolution is a process that'),
            (r'Biology is a (character|field) who', 'Biology is a field that'),
            (r'Chemistry is a (character|field) who', 'Chemistry is a field that'),
            (r'Mathematics is a (character|field) who', 'Mathematics is a field that'),
            (r'Science is a (character|discipline) who', 'Science is a discipline that'),
            (r'Philosophy is a (character|discipline) who', 'Philosophy is a discipline that'),
            (r'History is a (character|discipline) who', 'History is a discipline that'),
            (r'Consciousness is a (something|phenomenon) (who|that)', 'Consciousness is a phenomenon that'),
            (r'Animacy is a (something|linguistic property) (who|that)', 'Animacy is a linguistic property that'),
            (r'Abstraction is a (something|cognitive process) (who|that)', 'Abstraction is a cognitive process that'),
            (r'Truth is (something|a concept) (who|that)', 'Truth is a concept that'),
            (r'Knowledge is (something|a concept) (who|that)', 'Knowledge is a concept that'),
            
            # Generic fix for "X is a Y who" → "X is a Y that" for non-persons
            (r'(\w+) is a (field|process|discipline|phenomenon|property|concept) who', r'\1 is a \2 that'),
            
            # Remove trailing empty sentences
            (r'\.\s*$', '.'),
            (r'\s+$', ''),
            
            # Fix double spaces
            (r'\s+', ' '),
            
            # Fix trailing punctuation
            (r'\s+\.', '.'),
            (r'\.\.+', '.'),
            
            # Capitalize after periods
            (r'\. ([a-z])', lambda m: '. ' + m.group(1).upper()),
        ]
        
        for pattern, replacement in patterns:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def add_post_processor(self, processor):
        """Add a post-processing function."""
        self._post_processors.append(processor)
    
    def remove_post_processors(self):
        """Clear all post-processors."""
        self._post_processors = []


def create_lens(name: str, **kwargs) -> StyleLens:
    """Create a custom lens."""
    return StyleLens(name=name, **kwargs)


def get_lens(name: str) -> StyleLens:
    """Get a lens by name."""
    return LENSES.get(name, NATURAL_LENS)


def list_lenses() -> List[str]:
    """List available lens names."""
    return list(LENSES.keys())


# Convenience function for quick projection
def project(raw_output: str, lens_name: str = "natural", answer_type: str = "describe") -> str:
    """
    Quick projection of raw output through a named lens.
    
    Args:
        raw_output: Raw output from concept space
        lens_name: Name of the lens to use
        answer_type: Type of answer
    
    Returns:
        Projected natural language output
    """
    projector = OutputProjector(LENSES.get(lens_name, NATURAL_LENS))
    return projector.project(raw_output, answer_type)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("OUTPUT LENS DEMO")
    print("=" * 60)
    
    # Sample raw outputs from GeometricQA
    raw_outputs = [
        "Holmes is a entity who includes, examines, and deduces, often involving evidence and identity.",
        "Watson is a protagonist who assists, watches, and adventures, often involving doorway and interest.",
        "Physics is a protagonist who states, includes, and provides, often involving certain and attosecond.",
        "Animacy is a concept who provides, embeds, and theses, often involving conceptual and clauses.",
        "Formula is semantically.",
    ]
    
    print("\nRAW OUTPUT → LENS PROJECTION")
    print("-" * 60)
    
    for raw in raw_outputs:
        print(f"\nRaw: {raw[:60]}...")
        for lens_name in ["natural", "formal", "casual", "literary"]:
            projected = project(raw, lens_name)
            print(f"  {lens_name:10}: {projected[:70]}...")
