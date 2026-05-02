"""
Relationships: Virtual Abstraction for Semantic Transformations
================================================================

A Relationship defines HOW concepts relate to each other.
Each relationship type has its own behavior and path structure.

Examples:
- OppositeRelationship: hot→cold, big→small
- GenderRelationship: king→queen, man→woman
- TenseRelationship: run→ran, eat→ate
- IntensityRelationship: warm→hot, cool→cold

Relationships are abstract - they define the interface.
Paths are concrete - they store the actual transformations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch


@dataclass
class RelationshipMetadata:
    """Metadata about a relationship type."""
    name: str
    description: str
    symmetric: bool  # If A→B, does B→A also hold?
    examples: List[Tuple[str, str]]


class Relationship(ABC):
    """
    Abstract base class for semantic relationships.
    
    A Relationship defines:
    1. What kind of transformation this is
    2. How to discover paths for this relationship
    3. How to validate that a path is correct
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this relationship type."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @property
    def symmetric(self) -> bool:
        """Is this relationship symmetric? (A→B implies B→A)"""
        return True
    
    @abstractmethod
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        """
        Get a prompt for the model to generate example pairs.
        
        Used during path discovery to get training examples.
        """
        pass
    
    @abstractmethod
    def get_validation_prompt(self, source: str) -> str:
        """
        Get a prompt to validate a transformation.
        
        Used to verify that a discovered path is correct.
        """
        pass
    
    def get_metadata(self) -> RelationshipMetadata:
        """Get metadata about this relationship."""
        return RelationshipMetadata(
            name=self.name,
            description=self.description,
            symmetric=self.symmetric,
            examples=[]
        )


class OppositeRelationship(Relationship):
    """
    Opposite/antonym relationship.
    
    Examples: hot→cold, big→small, fast→slow
    """
    
    def __init__(self, domain: Optional[str] = None):
        """
        Args:
            domain: Optional domain specifier (e.g., "temperature", "size")
        """
        self.domain = domain
    
    @property
    def name(self) -> str:
        if self.domain:
            return f"opposite_{self.domain}"
        return "opposite"
    
    @property
    def description(self) -> str:
        if self.domain:
            return f"Opposite relationship in the {self.domain} domain"
        return "General opposite/antonym relationship"
    
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        if self.domain:
            return f"""List {n_pairs} pairs of {self.domain}-related opposites.
Use simple, common, single English words.
Format: word1, word2
One pair per line."""
        return f"""List {n_pairs} pairs of opposite words (antonyms).
Use simple, common, single English words.
Format: word1, word2
One pair per line."""
    
    def get_validation_prompt(self, source: str) -> str:
        if self.domain:
            return f"What is the {self.domain} opposite of '{source}'? Reply with just one word."
        return f"What is the opposite of '{source}'? Reply with just one word."


class GenderRelationship(Relationship):
    """
    Gender counterpart relationship.
    
    Examples: king→queen, man→woman, actor→actress
    """
    
    @property
    def name(self) -> str:
        return "gender"
    
    @property
    def description(self) -> str:
        return "Male/female counterpart relationship"
    
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        return f"""List {n_pairs} pairs of common nouns where one is male and one is female.
Examples: king/queen, man/woman, boy/girl, father/mother.
Do NOT use proper names like John/Jane.
Format: male_word, female_word
One pair per line."""
    
    def get_validation_prompt(self, source: str) -> str:
        return f"What is the female counterpart of '{source}'? Reply with just one word."


class TenseRelationship(Relationship):
    """
    Verb tense relationship.
    
    Examples: run→ran, eat→ate, go→went
    """
    
    def __init__(self, from_tense: str = "present", to_tense: str = "past"):
        self.from_tense = from_tense
        self.to_tense = to_tense
    
    @property
    def name(self) -> str:
        return f"tense_{self.from_tense}_to_{self.to_tense}"
    
    @property
    def description(self) -> str:
        return f"Verb tense: {self.from_tense} → {self.to_tense}"
    
    @property
    def symmetric(self) -> bool:
        return False  # run→ran doesn't imply ran→run in the same way
    
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        return f"""List {n_pairs} pairs of verbs in {self.from_tense} and {self.to_tense} tense.
Examples: run/ran, eat/ate, go/went.
Format: {self.from_tense}_form, {self.to_tense}_form
One pair per line."""
    
    def get_validation_prompt(self, source: str) -> str:
        return f"What is the {self.to_tense} tense of '{source}'? Reply with just one word."


class IntensityRelationship(Relationship):
    """
    Intensity/degree relationship.
    
    Examples: warm→hot, cool→cold, like→love
    """
    
    def __init__(self, direction: str = "increase"):
        """
        Args:
            direction: "increase" or "decrease"
        """
        self.direction = direction
    
    @property
    def name(self) -> str:
        return f"intensity_{self.direction}"
    
    @property
    def description(self) -> str:
        return f"Intensity {self.direction} relationship"
    
    @property
    def symmetric(self) -> bool:
        return False
    
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        if self.direction == "increase":
            return f"""List {n_pairs} pairs where the second word is a more intense version of the first.
Examples: warm/hot, like/love, big/huge.
Format: mild_word, intense_word
One pair per line."""
        else:
            return f"""List {n_pairs} pairs where the second word is a less intense version of the first.
Examples: hot/warm, love/like, huge/big.
Format: intense_word, mild_word
One pair per line."""
    
    def get_validation_prompt(self, source: str) -> str:
        if self.direction == "increase":
            return f"What is a more intense word for '{source}'? Reply with just one word."
        return f"What is a less intense word for '{source}'? Reply with just one word."


class SpatialRelationship(Relationship):
    """
    Spatial opposite relationship.
    
    Examples: up→down, left→right, near→far
    """
    
    def __init__(self, axis: str = "vertical"):
        """
        Args:
            axis: "vertical", "horizontal", or "distance"
        """
        self.axis = axis
    
    @property
    def name(self) -> str:
        return f"spatial_{self.axis}"
    
    @property
    def description(self) -> str:
        return f"Spatial opposite on {self.axis} axis"
    
    def get_discovery_prompt(self, n_pairs: int = 15) -> str:
        if self.axis == "vertical":
            examples = "up/down, above/below, top/bottom"
        elif self.axis == "horizontal":
            examples = "left/right, east/west"
        else:
            examples = "near/far, close/distant"
        
        return f"""List {n_pairs} pairs of spatial opposites on the {self.axis} axis.
Examples: {examples}.
Format: word1, word2
One pair per line."""
    
    def get_validation_prompt(self, source: str) -> str:
        return f"What is the spatial opposite of '{source}'? Reply with just one word."


# Registry of all relationship types
RELATIONSHIP_REGISTRY: Dict[str, type] = {
    'opposite': OppositeRelationship,
    'gender': GenderRelationship,
    'tense': TenseRelationship,
    'intensity': IntensityRelationship,
    'spatial': SpatialRelationship,
}


def get_relationship(name: str, **kwargs) -> Relationship:
    """Factory function to create relationship instances."""
    if name not in RELATIONSHIP_REGISTRY:
        raise ValueError(f"Unknown relationship type: {name}")
    return RELATIONSHIP_REGISTRY[name](**kwargs)
