#!/usr/bin/env python3
"""
Learnable Structure: Gradient-Free Knowledge Graph Learning

This module implements error-driven structure learning for the LCM system.
Instead of gradient descent on weights, we learn by analyzing errors and
adding discrete structure to a knowledge graph.

The key insight: Error = Construction Blueprint
- Each error points to missing structure
- We add the structure
- The model improves
- No gradients needed

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


# =============================================================================
# VOCABULARY CATEGORIES
# =============================================================================

# Role vocabulary - words that describe what an entity IS
ROLE_VOCABULARY = {
    # Professions
    'detective', 'doctor', 'lawyer', 'professor', 'inspector', 'captain',
    'soldier', 'servant', 'butler', 'maid', 'nurse', 'teacher', 'writer',
    'artist', 'musician', 'scientist', 'engineer', 'merchant', 'banker',
    # Social roles
    'gentleman', 'lady', 'lord', 'duke', 'duchess', 'king', 'queen',
    'prince', 'princess', 'knight', 'squire', 'peasant', 'commoner',
    # Relationship roles
    'friend', 'companion', 'partner', 'ally', 'enemy', 'rival', 'lover',
    'husband', 'wife', 'father', 'mother', 'son', 'daughter', 'brother',
    'sister', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
    # Character types
    'hero', 'villain', 'protagonist', 'antagonist', 'narrator', 'witness',
}

# Quality vocabulary - words that describe HOW an entity is
QUALITY_VOCABULARY = {
    # Intellectual
    'brilliant', 'clever', 'intelligent', 'wise', 'cunning', 'shrewd',
    'observant', 'perceptive', 'analytical', 'logical', 'rational',
    # Emotional
    'kind', 'gentle', 'compassionate', 'loving', 'caring', 'warm',
    'cold', 'cruel', 'harsh', 'stern', 'strict', 'severe',
    # Social
    'proud', 'humble', 'arrogant', 'modest', 'shy', 'bold', 'brave',
    'cowardly', 'loyal', 'faithful', 'treacherous', 'deceitful',
    # Physical
    'strong', 'weak', 'tall', 'short', 'handsome', 'beautiful', 'ugly',
    'graceful', 'clumsy', 'agile', 'slow', 'quick', 'fast',
    # Personality
    'witty', 'charming', 'charismatic', 'mysterious', 'enigmatic',
    'eccentric', 'peculiar', 'strange', 'normal', 'ordinary',
}

# Action vocabulary - words that describe what an entity DOES
ACTION_VOCABULARY = {
    # Investigation
    'investigates', 'deduces', 'solves', 'discovers', 'uncovers',
    'examines', 'analyzes', 'observes', 'watches', 'studies',
    # Communication
    'speaks', 'talks', 'says', 'tells', 'explains', 'describes',
    'narrates', 'chronicles', 'records', 'writes', 'reads',
    # Emotion
    'loves', 'hates', 'fears', 'admires', 'respects', 'despises',
    'envies', 'pities', 'sympathizes', 'empathizes',
    # Action
    'fights', 'battles', 'struggles', 'challenges', 'confronts',
    'helps', 'assists', 'aids', 'supports', 'protects', 'defends',
    'attacks', 'threatens', 'intimidates',
    # Movement
    'travels', 'journeys', 'visits', 'arrives', 'departs', 'leaves',
    'returns', 'escapes', 'flees', 'chases', 'pursues', 'follows',
}


# =============================================================================
# LEARNABLE STRUCTURE
# =============================================================================

@dataclass
class EntityProfile:
    """Profile of learned information about an entity."""
    role: Optional[str] = None
    qualities: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    source: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'qualities': self.qualities,
            'actions': self.actions,
            'relations': self.relations,
            'source': self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EntityProfile':
        return cls(
            role=data.get('role'),
            qualities=data.get('qualities', []),
            actions=data.get('actions', []),
            relations=data.get('relations', []),
            source=data.get('source'),
        )


class LearnableStructure:
    """
    Gradient-free learnable knowledge structure.
    
    Learns by analyzing errors between generated and target text,
    then adding discrete structure to fill the gaps.
    """
    
    def __init__(self):
        self.profiles: Dict[str, EntityProfile] = {}
        self.known_entities: Set[str] = set()
        
        # Vocabulary for classification
        self.role_vocab = ROLE_VOCABULARY
        self.quality_vocab = QUALITY_VOCABULARY
        self.action_vocab = ACTION_VOCABULARY
    
    def add_known_entity(self, entity: str):
        """Register an entity as known (for relation detection)."""
        self.known_entities.add(entity.lower())
    
    def add_known_entities(self, entities: List[str]):
        """Register multiple entities."""
        for entity in entities:
            self.add_known_entity(entity)
    
    def get_profile(self, entity: str) -> EntityProfile:
        """Get or create profile for entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.profiles:
            self.profiles[entity_lower] = EntityProfile()
        return self.profiles[entity_lower]
    
    def learn_from_error(self, entity: str, target: str, generated: str) -> List[str]:
        """
        Learn structure from the gap between target and generated.
        
        Returns list of what was learned.
        """
        # Tokenize and normalize
        target_words = set(self._tokenize(target))
        gen_words = set(self._tokenize(generated))
        
        # Find missing words
        missing = target_words - gen_words
        
        # Get or create profile
        profile = self.get_profile(entity)
        entity_lower = entity.lower()
        
        learned = []
        
        for word in missing:
            word_lower = word.lower()
            
            # Classify and add
            if word_lower in self.role_vocab:
                if profile.role != word_lower:
                    profile.role = word_lower
                    learned.append(f'role:{word_lower}')
            
            elif word_lower in self.quality_vocab:
                if word_lower not in profile.qualities:
                    profile.qualities.append(word_lower)
                    learned.append(f'quality:{word_lower}')
            
            elif word_lower in self.action_vocab:
                if word_lower not in profile.actions:
                    profile.actions.append(word_lower)
                    learned.append(f'action:{word_lower}')
            
            elif word_lower in self.known_entities and word_lower != entity_lower:
                if word_lower not in profile.relations:
                    profile.relations.append(word_lower)
                    learned.append(f'relation:{word_lower}')
        
        return learned
    
    def learn_from_corpus(self, entity: str, texts: List[str]):
        """
        Learn structure by extracting from corpus texts.
        
        This is unsupervised learning from raw text.
        """
        profile = self.get_profile(entity)
        entity_lower = entity.lower()
        
        for text in texts:
            words = self._tokenize(text)
            
            for word in words:
                word_lower = word.lower()
                
                if word_lower in self.role_vocab and profile.role is None:
                    profile.role = word_lower
                
                elif word_lower in self.quality_vocab:
                    if word_lower not in profile.qualities and len(profile.qualities) < 3:
                        profile.qualities.append(word_lower)
                
                elif word_lower in self.action_vocab:
                    if word_lower not in profile.actions and len(profile.actions) < 3:
                        profile.actions.append(word_lower)
                
                elif word_lower in self.known_entities and word_lower != entity_lower:
                    if word_lower not in profile.relations and len(profile.relations) < 3:
                        profile.relations.append(word_lower)
    
    def generate(self, entity: str, source: str = 'the story', 
                 style: str = 'neutral') -> str:
        """
        Generate answer using learned structure.
        
        Args:
            entity: The entity to describe
            source: The source text name
            style: 'formal', 'neutral', or 'casual'
        """
        profile = self.get_profile(entity)
        
        # Build parts
        parts = [entity.title()]
        
        # Copula
        parts.append('is')
        
        # Article
        parts.append('a')
        
        # Qualities (before role)
        if profile.qualities:
            qualities = profile.qualities[:2]  # Max 2 qualities
            parts.append(' '.join(qualities))
        
        # Role
        role = profile.role or 'character'
        parts.append(role)
        
        # Source
        parts.append(f'from {source}')
        
        # Actions
        if profile.actions:
            action = profile.actions[0]
            parts.append(f'who {action}')
        
        # Relations
        if profile.relations:
            relation = profile.relations[0]
            # Grammar: "who loves Elizabeth" not "who loves with Elizabeth"
            if profile.actions and profile.actions[0] in ('loves', 'admires', 'hates', 'fears'):
                parts.append(relation.title())
            else:
                parts.append(f'with {relation.title()}')
        
        sentence = ' '.join(parts) + '.'
        
        # Clean up double spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        
        return sentence
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def save(self, path: str):
        """Save learned structure to JSON file."""
        data = {
            'profiles': {k: v.to_dict() for k, v in self.profiles.items()},
            'known_entities': list(self.known_entities),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load learned structure from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.profiles = {
            k: EntityProfile.from_dict(v) 
            for k, v in data.get('profiles', {}).items()
        }
        self.known_entities = set(data.get('known_entities', []))
    
    def get_stats(self) -> dict:
        """Get statistics about learned structure."""
        total_roles = sum(1 for p in self.profiles.values() if p.role)
        total_qualities = sum(len(p.qualities) for p in self.profiles.values())
        total_actions = sum(len(p.actions) for p in self.profiles.values())
        total_relations = sum(len(p.relations) for p in self.profiles.values())
        
        return {
            'entities': len(self.profiles),
            'known_entities': len(self.known_entities),
            'roles_learned': total_roles,
            'qualities_learned': total_qualities,
            'actions_learned': total_actions,
            'relations_learned': total_relations,
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_from_examples(structure: LearnableStructure, 
                        examples: List[Tuple[str, str, str]],
                        max_epochs: int = 10,
                        target_overlap: float = 0.95) -> dict:
    """
    Train structure from (entity, source, target_answer) examples.
    
    Args:
        structure: The LearnableStructure to train
        examples: List of (entity, source, target_answer) tuples
        max_epochs: Maximum training epochs
        target_overlap: Stop when average overlap reaches this
    
    Returns:
        Training statistics
    """
    history = []
    
    for epoch in range(max_epochs):
        total_overlap = 0
        all_learned = []
        
        for entity, source, target in examples:
            # Generate with current structure
            generated = structure.generate(entity, source)
            
            # Compute overlap
            target_words = set(structure._tokenize(target))
            gen_words = set(structure._tokenize(generated))
            
            if target_words | gen_words:
                overlap = len(target_words & gen_words) / len(target_words | gen_words)
            else:
                overlap = 0
            
            total_overlap += overlap
            
            # Learn from error
            learned = structure.learn_from_error(entity, target, generated)
            all_learned.extend(learned)
        
        avg_overlap = total_overlap / len(examples) if examples else 0
        history.append({
            'epoch': epoch + 1,
            'avg_overlap': avg_overlap,
            'learned': all_learned,
        })
        
        # Check convergence
        if avg_overlap >= target_overlap:
            break
        
        # Check if no learning happened (converged)
        if not all_learned:
            break
    
    return {
        'epochs': len(history),
        'final_overlap': history[-1]['avg_overlap'] if history else 0,
        'history': history,
    }


def compute_overlap(target: str, generated: str) -> float:
    """Compute word overlap between target and generated."""
    target_words = set(re.findall(r'\b\w+\b', target.lower()))
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    
    if not (target_words | gen_words):
        return 0
    
    return len(target_words & gen_words) / len(target_words | gen_words)


# =============================================================================
# TEST
# =============================================================================

def test_learnable_structure():
    """Test the learnable structure."""
    print("=== LearnableStructure Test ===\n")
    
    # Create structure
    structure = LearnableStructure()
    
    # Add known entities
    structure.add_known_entities([
        'holmes', 'watson', 'moriarty', 'lestrade',
        'darcy', 'elizabeth', 'jane', 'bingley', 'wickham', 'charlotte',
    ])
    
    # Training examples
    examples = [
        ('holmes', 'Sherlock Holmes', 
         'Holmes is a brilliant detective from Sherlock Holmes who investigates with Watson.'),
        ('watson', 'Sherlock Holmes', 
         'Watson is a loyal doctor from Sherlock Holmes who assists Holmes.'),
        ('darcy', 'Pride and Prejudice', 
         'Darcy is a proud gentleman from Pride and Prejudice who loves Elizabeth.'),
        ('elizabeth', 'Pride and Prejudice', 
         'Elizabeth is a witty lady from Pride and Prejudice who challenges Darcy.'),
    ]
    
    # Train
    print("Training...")
    result = train_from_examples(structure, examples)
    print(f"Epochs: {result['epochs']}")
    print(f"Final overlap: {result['final_overlap']:.1%}")
    print()
    
    # Show results
    print("Learned profiles:")
    for entity in ['holmes', 'watson', 'darcy', 'elizabeth']:
        profile = structure.get_profile(entity)
        print(f"  {entity}: role={profile.role}, qualities={profile.qualities}, "
              f"actions={profile.actions}, relations={profile.relations}")
    print()
    
    # Generate
    print("Generated answers:")
    for entity, source, target in examples:
        generated = structure.generate(entity, source)
        overlap = compute_overlap(target, generated)
        print(f"  {entity}: \"{generated}\" (overlap: {overlap:.1%})")
    print()
    
    # Stats
    print("Stats:", structure.get_stats())


if __name__ == '__main__':
    test_learnable_structure()
