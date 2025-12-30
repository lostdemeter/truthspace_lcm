#!/usr/bin/env python3
"""
Structural Priority: Guide Self-Improvement Based on Geometric Gaps

This module analyzes the TruthSpace structure to identify:
1. Missing transforms (axes not covered)
2. Concept gaps (positions that should exist but don't)
3. Priority topics (what to fetch next to fill gaps)

The self-improvement daemon uses this to prioritize exploration
based on structural needs rather than random selection.

Author: Lesley Gushurst
License: GPLv3
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from .semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class Transform:
    """A self-similar transformation."""
    name: str
    delta: Tuple[float, float, float, float]
    examples: List[Tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ConceptGap:
    """A predicted but unfilled concept position."""
    position: Tuple[float, float, float, float]
    predicted_by: str
    priority: float
    suggested_keywords: List[str] = field(default_factory=list)


@dataclass
class StructuralPriority:
    """Priority information for guiding exploration."""
    least_covered_axis: str
    axis_coverage: Dict[str, int]
    missing_transforms: List[str]
    top_gaps: List[ConceptGap]
    suggested_topics: List[str]
    priority_keywords: List[str]


class StructuralAnalyzer:
    """
    Analyzes TruthSpace structure to find gaps and priorities.
    
    Used by the self-improvement daemon to guide exploration.
    """
    
    def __init__(self, concepts: Dict[str, SemanticQuaternion] = None):
        self.concepts = concepts or dict(DEFAULT_SEMANTIC_FEATURES)
        self.transforms: Dict[str, Transform] = {}
        self._init_known_transforms()
    
    def _init_known_transforms(self):
        """Initialize with known self-similar transforms."""
        self.transforms = {
            'gender_flip': Transform(
                name='gender_flip',
                delta=(-2.0, 0.0, 0.0, 0.0),
                examples=[('king', 'queen'), ('man', 'woman'), ('boy', 'girl')],
                confidence=1.0,
            ),
            'age_decrease': Transform(
                name='age_decrease',
                delta=(0.0, -2.0, 0.0, 0.0),
                examples=[('man', 'boy'), ('woman', 'girl')],
                confidence=1.0,
            ),
            'agency_decrease': Transform(
                name='agency_decrease',
                delta=(0.0, 0.0, -0.5, 0.0),
                examples=[('king', 'man')],
                confidence=0.9,
            ),
        }
    
    def update_concepts(self, concepts: Dict[str, SemanticQuaternion]):
        """Update the concept space (called when corpus changes)."""
        self.concepts = concepts
    
    def analyze_axis_coverage(self) -> Dict[str, int]:
        """Analyze how well each axis is covered by transforms."""
        coverage = {'x': 0, 'y': 0, 'z': 0, 'w': 0}
        
        for t in self.transforms.values():
            if abs(t.delta[0]) > 0.1: coverage['x'] += 1
            if abs(t.delta[1]) > 0.1: coverage['y'] += 1
            if abs(t.delta[2]) > 0.1: coverage['z'] += 1
            if abs(t.delta[3]) > 0.1: coverage['w'] += 1
        
        return coverage
    
    def find_least_covered_axis(self) -> str:
        """Find the axis with least transform coverage."""
        coverage = self.analyze_axis_coverage()
        axis_names = {'x': 'gender', 'y': 'age', 'z': 'agency', 'w': 'animacy'}
        
        min_axis = min(coverage, key=coverage.get)
        return axis_names[min_axis]
    
    def find_concept_gaps(self, max_gaps: int = 20) -> List[ConceptGap]:
        """Find positions where concepts should exist but don't."""
        gaps = []
        seen_positions = set()
        
        for concept_name, sq in self.concepts.items():
            for t_name, transform in self.transforms.items():
                # Apply transform
                new_x = sq.x + transform.delta[0]
                new_y = sq.y + transform.delta[1]
                new_z = sq.z + transform.delta[2]
                new_w = sq.w + transform.delta[3]
                
                new_pos = (round(new_x, 1), round(new_y, 1), round(new_z, 1), round(new_w, 1))
                
                if new_pos in seen_positions:
                    continue
                seen_positions.add(new_pos)
                
                # Check if position exists
                exists = False
                for other_sq in self.concepts.values():
                    if (abs(new_pos[0] - other_sq.x) < 0.15 and
                        abs(new_pos[1] - other_sq.y) < 0.15 and
                        abs(new_pos[2] - other_sq.z) < 0.15 and
                        abs(new_pos[3] - other_sq.w) < 0.15):
                        exists = True
                        break
                
                if not exists:
                    # Generate keywords for this gap
                    keywords = self._generate_gap_keywords(new_pos)
                    
                    gaps.append(ConceptGap(
                        position=new_pos,
                        predicted_by=f"{t_name}({concept_name})",
                        priority=transform.confidence,
                        suggested_keywords=keywords,
                    ))
        
        # Sort by priority and return top gaps
        gaps.sort(key=lambda g: -g.priority)
        return gaps[:max_gaps]
    
    def _generate_gap_keywords(self, position: Tuple[float, float, float, float]) -> List[str]:
        """Generate search keywords for a gap position."""
        x, y, z, w = position
        keywords = []
        
        # Gender axis
        if x > 0.5:
            keywords.extend(['male', 'man', 'boy', 'father', 'king', 'prince'])
        elif x < -0.5:
            keywords.extend(['female', 'woman', 'girl', 'mother', 'queen', 'princess'])
        
        # Age axis
        if y > 0.5:
            keywords.extend(['adult', 'mature', 'elder', 'senior'])
        elif y < -0.5:
            keywords.extend(['young', 'child', 'youth', 'junior'])
        
        # Agency axis
        if z > 0.5:
            keywords.extend(['leader', 'ruler', 'authority', 'power', 'control'])
        elif z < -0.5:
            keywords.extend(['servant', 'follower', 'subject', 'subordinate'])
        
        # Animacy axis
        if w > 0.5:
            keywords.extend(['person', 'human', 'individual', 'being'])
        elif w < -0.5:
            keywords.extend(['concept', 'abstract', 'idea', 'notion', 'principle'])
        
        return keywords[:5]  # Top 5 keywords
    
    def suggest_priority_topics(self) -> List[str]:
        """Suggest Grokipedia topics based on structural gaps."""
        gaps = self.find_concept_gaps(max_gaps=20)
        least_covered = self.find_least_covered_axis()
        
        topics = []
        
        # EXPANDED: Topics based on least covered axis
        axis_topics = {
            'gender': [
                'Gender_studies', 'Masculinity', 'Femininity', 'Gender_role',
                'Patriarchy', 'Matriarchy', 'Gender_identity', 'Sexism',
                'Feminism', 'Men\'s_rights_movement', 'Gender_equality',
                'Transgender', 'Non-binary_gender', 'Sexual_dimorphism',
                'Male', 'Female', 'Man', 'Woman', 'Boy', 'Girl',
                'Father', 'Mother', 'Brother', 'Sister', 'Son', 'Daughter',
                'Husband', 'Wife', 'King', 'Queen', 'Prince', 'Princess',
                'Actor', 'Actress', 'Waiter', 'Waitress', 'Hero', 'Heroine',
            ],
            'age': [
                'Childhood', 'Adolescence', 'Adulthood', 'Old_age',
                'Life_stage', 'Coming_of_age', 'Gerontology', 'Aging',
                'Infant', 'Toddler', 'Child', 'Teenager', 'Young_adult',
                'Middle_age', 'Senior_citizen', 'Centenarian', 'Youth',
                'Puberty', 'Menopause', 'Life_expectancy', 'Longevity',
                'Generation', 'Baby_boomer', 'Generation_X', 'Millennial',
                'Generation_Z', 'Silent_Generation', 'Greatest_Generation',
            ],
            'agency': [
                'Agency_(philosophy)', 'Free_will', 'Autonomy', 'Power_(social_and_political)',
                'Authority', 'Leadership', 'Hierarchy', 'Control_(management)',
                'Dominance_(ethology)', 'Submission', 'Obedience', 'Rebellion',
                'Revolution', 'Dictatorship', 'Democracy', 'Monarchy',
                'Aristocracy', 'Oligarchy', 'Anarchy', 'Totalitarianism',
                'Boss', 'Employee', 'Manager', 'Worker', 'Slave', 'Master',
                'Ruler', 'Subject_(philosophy)', 'Citizen', 'Peasant',
                'Noble', 'Commoner', 'Elite', 'Proletariat', 'Bourgeoisie',
            ],
            'animacy': [
                'Animacy', 'Personification', 'Abstraction', 'Concept',
                'Reification', 'Anthropomorphism', 'Abstract_and_concrete',
                'Consciousness', 'Sentience', 'Sapience', 'Intelligence',
                'Artificial_intelligence', 'Robot', 'Android', 'Cyborg',
                'Animal', 'Plant', 'Organism', 'Life', 'Death', 'Soul',
                'Mind', 'Body', 'Spirit', 'Ghost', 'Zombie', 'Vampire',
                'God', 'Deity', 'Angel', 'Demon', 'Monster', 'Creature',
                'Object_(philosophy)', 'Thing', 'Entity', 'Being', 'Existence',
                'Reality', 'Illusion', 'Dream', 'Imagination', 'Fantasy',
            ],
        }
        
        topics.extend(axis_topics.get(least_covered, []))
        
        # Add topics from ALL axes (not just least covered) for diversity
        for axis, axis_topic_list in axis_topics.items():
            if axis != least_covered:
                topics.extend(axis_topic_list[:10])  # Top 10 from each other axis
        
        # Topics based on gap keywords
        for gap in gaps[:10]:
            for keyword in gap.suggested_keywords[:3]:
                # Convert keyword to potential topic
                topic = keyword.replace(' ', '_').title()
                if topic not in topics:
                    topics.append(topic)
        
        return topics[:100]  # Expanded to 100 topics
    
    def get_priority_keywords(self) -> List[str]:
        """Get keywords to prioritize in sentence scoring."""
        gaps = self.find_concept_gaps(max_gaps=10)
        keywords = set()
        
        for gap in gaps:
            keywords.update(gap.suggested_keywords)
        
        # Add axis-specific keywords
        least_covered = self.find_least_covered_axis()
        axis_keywords = {
            'gender': ['male', 'female', 'man', 'woman', 'masculine', 'feminine'],
            'age': ['young', 'old', 'child', 'adult', 'youth', 'elder'],
            'agency': ['leader', 'follower', 'power', 'authority', 'control', 'serve'],
            'animacy': ['person', 'thing', 'abstract', 'concrete', 'human', 'concept'],
        }
        keywords.update(axis_keywords.get(least_covered, []))
        
        return list(keywords)[:20]
    
    def get_structural_priority(self) -> StructuralPriority:
        """Get complete structural priority information."""
        coverage = self.analyze_axis_coverage()
        least_covered = self.find_least_covered_axis()
        gaps = self.find_concept_gaps(max_gaps=10)
        
        # Find missing transforms
        missing = []
        if coverage['w'] == 0:
            missing.append('w_transform (animacy)')
        if coverage['z'] < 2:
            missing.append('more z_transforms (agency)')
        
        return StructuralPriority(
            least_covered_axis=least_covered,
            axis_coverage=coverage,
            missing_transforms=missing,
            top_gaps=gaps,
            suggested_topics=self.suggest_priority_topics(),
            priority_keywords=self.get_priority_keywords(),
        )
    
    def discover_transform_from_examples(self, examples: List[Tuple[str, str]]) -> Optional[Transform]:
        """Try to discover a new transform from concept pairs."""
        if len(examples) < 2:
            return None
        
        deltas = []
        for a_name, b_name in examples:
            if a_name in self.concepts and b_name in self.concepts:
                a = self.concepts[a_name]
                b = self.concepts[b_name]
                delta = (b.x - a.x, b.y - a.y, b.z - a.z, b.w - a.w)
                deltas.append(delta)
        
        if len(deltas) < 2:
            return None
        
        # Check consistency
        variance = sum(np.var([d[i] for d in deltas]) for i in range(4))
        if variance > 0.1:
            return None  # Not consistent
        
        avg_delta = tuple(float(np.mean([d[i] for d in deltas])) for i in range(4))
        
        return Transform(
            name='discovered',
            delta=avg_delta,
            examples=examples,
            confidence=max(0, 1 - variance),
        )
    
    def add_transform(self, name: str, transform: Transform):
        """Add a discovered transform."""
        transform.name = name
        self.transforms[name] = transform


# Global instance for easy access
_analyzer: Optional[StructuralAnalyzer] = None


def get_analyzer() -> StructuralAnalyzer:
    """Get or create the global structural analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = StructuralAnalyzer()
    return _analyzer


def get_priority_topics() -> List[str]:
    """Get priority topics for the self-improvement daemon."""
    return get_analyzer().suggest_priority_topics()


def get_priority_keywords() -> List[str]:
    """Get priority keywords for sentence scoring."""
    return get_analyzer().get_priority_keywords()


def get_structural_priority() -> StructuralPriority:
    """Get complete structural priority information."""
    return get_analyzer().get_structural_priority()
