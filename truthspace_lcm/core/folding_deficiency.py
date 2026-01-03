"""
Folding-Based Deficiency Detection

A geometric approach to deficiency detection based on DNA-like folding structures.
Information is encoded in SHAPE (fold patterns), not content.

Key principles:
1. Folds occur where the sequence references itself (repeated words)
2. Shape is the curvature pattern created by folds
3. Similar shapes = similar meaning, regardless of content
4. Content can have errors and still work (error tolerance)

This replaces the pattern-matching approach with a geometric approach.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from enum import Enum
import re


# φ (golden ratio) - fundamental constant
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class FoldPoint:
    """
    A point where the structure folds.
    
    Like a zinc finger in DNA - an access point that brings
    distant parts of the sequence into contact.
    """
    position: int
    fold_to: int
    strength: float = 1.0
    label: str = ""
    
    @property
    def distance(self) -> int:
        """Linear distance bridged by this fold."""
        return abs(self.fold_to - self.position)


class FoldingStructure:
    """
    A structure that encodes information through FOLDING.
    
    Key principles:
    1. Linear sequence of tokens (like DNA bases)
    2. Fold points bring distant parts into contact
    3. Shape (curvature between folds) encodes meaning
    4. Content can have error - shape is what matters
    """
    
    # Common words that create weak folds
    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'it', 'this', 'that', 'and', 'or', 'but', 'if', 'then',
                 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'your'}
    
    def __init__(self):
        self.sequence: List[str] = []
        self.fold_points: List[FoldPoint] = []
        self.contact_map: Dict[int, Set[int]] = defaultdict(set)
        self.access_points: Set[int] = set()
    
    @classmethod
    def from_text(cls, text: str) -> 'FoldingStructure':
        """Create a FoldingStructure from text."""
        structure = cls()
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if tokens:
            structure.add_sequence(tokens)
        return structure
    
    def add_sequence(self, tokens: List[str]):
        """Add tokens to the linear sequence."""
        self.sequence.extend(tokens)
        self._detect_fold_points()
    
    def _detect_fold_points(self):
        """
        Detect natural fold points in the sequence.
        
        Folds happen where the sequence references itself.
        Weight folds by word importance (Zipf-aware).
        """
        if len(self.sequence) < 3:
            return
        
        # Clear existing folds
        self.fold_points = []
        self.contact_map = defaultdict(set)
        self.access_points = set()
        
        # Count word frequencies for Zipf weighting
        word_counts: Dict[str, int] = defaultdict(int)
        for word in self.sequence:
            word_counts[word] += 1
        
        total_words = len(self.sequence)
        
        # Find repeated words
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, word in enumerate(self.sequence):
            word_positions[word].append(i)
        
        # Words that appear multiple times create folds
        for word, positions in word_positions.items():
            if len(positions) >= 2:
                # Calculate fold strength based on word importance
                freq_ratio = word_counts[word] / total_words
                
                if word in self.STOPWORDS:
                    base_strength = 0.1  # Weak fold for stopwords
                elif freq_ratio > 0.1:
                    base_strength = 0.3  # Medium fold for frequent words
                else:
                    base_strength = 1.0  # Strong fold for rare/specific words
                
                # Create folds between occurrences
                for i in range(len(positions) - 1):
                    pos1 = positions[i]
                    pos2 = positions[i + 1]
                    
                    if pos2 - pos1 >= 2:  # Only fold if there's distance
                        strength = base_strength / (pos2 - pos1)
                        
                        fold = FoldPoint(
                            position=pos2,
                            fold_to=pos1,
                            strength=strength,
                            label=f"self-ref:{word}"
                        )
                        self.fold_points.append(fold)
                        
                        self.contact_map[pos1].add(pos2)
                        self.contact_map[pos2].add(pos1)
                        
                        if strength > 0.1:
                            self.access_points.add(pos1)
                            self.access_points.add(pos2)
    
    def compute_shape(self) -> np.ndarray:
        """
        Compute the shape of the structure.
        
        Shape is represented as curvature at each position.
        High curvature = fold point, Low curvature = straight segment.
        """
        n = len(self.sequence)
        if n == 0:
            return np.array([])
        
        curvature = np.zeros(n)
        
        for fold in self.fold_points:
            if fold.position < n:
                curvature[fold.position] += fold.strength
            if fold.fold_to < n:
                curvature[fold.fold_to] += fold.strength * 0.5
        
        # Smooth the curvature
        if len(curvature) >= 3:
            smoothed = np.convolve(curvature, [0.25, 0.5, 0.25], mode='same')
            return smoothed
        
        return curvature
    
    def shape_similarity(self, other: 'FoldingStructure') -> float:
        """
        Compare shapes of two structures.
        
        Similar shapes = similar meaning, even if content differs.
        """
        shape1 = self.compute_shape()
        shape2 = other.compute_shape()
        
        if len(shape1) == 0 or len(shape2) == 0:
            return 0.0
        
        # Resample to same length
        target_len = min(len(shape1), len(shape2))
        if len(shape1) > target_len:
            indices = np.linspace(0, len(shape1)-1, target_len).astype(int)
            shape1 = shape1[indices]
        if len(shape2) > target_len:
            indices = np.linspace(0, len(shape2)-1, target_len).astype(int)
            shape2 = shape2[indices]
        
        if np.std(shape1) < 1e-10 or np.std(shape2) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(shape1, shape2)[0, 1]
        return max(0.0, correlation)
    
    def shape_mismatch(self, expected_shape: np.ndarray) -> float:
        """Calculate shape mismatch score."""
        actual_shape = self.compute_shape()
        
        if len(actual_shape) == 0 or len(expected_shape) == 0:
            return 1.0
        
        target_len = min(len(actual_shape), len(expected_shape))
        if len(actual_shape) > target_len:
            indices = np.linspace(0, len(actual_shape)-1, target_len).astype(int)
            actual_shape = actual_shape[indices]
        if len(expected_shape) > target_len:
            indices = np.linspace(0, len(expected_shape)-1, target_len).astype(int)
            expected_shape = expected_shape[indices]
        
        diff = np.abs(actual_shape - expected_shape)
        return np.mean(diff)
    
    def get_fold_words(self) -> Set[str]:
        """Get the words that create folds."""
        fold_words = set()
        for fold in self.fold_points:
            if fold.position < len(self.sequence):
                fold_words.add(self.sequence[fold.position])
        return fold_words


class ShapeDeficiencyType(Enum):
    """
    Deficiency types that EMERGE from shape analysis.
    
    Unlike the old hardcoded enum, these are derived from
    geometric properties of the shape mismatch.
    """
    NONE = "none"                    # Shape matches well
    INCOMPLETE = "incomplete"        # Too short, missing structure
    MISSING_STRUCTURE = "missing_structure"  # No self-references
    WRONG_STRUCTURE = "wrong_structure"      # Different fold pattern
    PARTIAL = "partial"              # Some mismatch


@dataclass
class ShapeDeficiency:
    """
    A deficiency detected via shape analysis.
    
    This replaces the old pattern-matching Deficiency class.
    """
    type: ShapeDeficiencyType
    severity: float  # 0.0 to 1.0
    shape_similarity: float
    shape_mismatch: float
    fold_ratio: float
    length_ratio: float
    missing_fold_words: Set[str] = field(default_factory=set)
    description: str = ""
    suggested_fix: str = ""


class FoldingDeficiencyDetector:
    """
    Detects deficiencies using shape-based analysis.
    
    This is the geometric replacement for DeficiencyDetectorGear.
    """
    
    def __init__(self):
        self.name = "FoldingDeficiencyDetector"
        
        # Learned shape templates
        self.shape_templates: Dict[str, np.ndarray] = {}
        self.structure_templates: Dict[str, FoldingStructure] = {}
    
    def learn_template(self, name: str, text: str):
        """Learn a shape template from example text."""
        structure = FoldingStructure.from_text(text)
        self.structure_templates[name] = structure
        self.shape_templates[name] = structure.compute_shape()
    
    def detect(self, expected: str, actual: str) -> ShapeDeficiency:
        """
        Detect deficiency between expected and actual output.
        
        Returns a ShapeDeficiency with geometric metrics.
        """
        expected_struct = FoldingStructure.from_text(expected)
        actual_struct = FoldingStructure.from_text(actual)
        
        expected_shape = expected_struct.compute_shape()
        
        # Compute metrics
        shape_sim = expected_struct.shape_similarity(actual_struct)
        shape_mismatch = actual_struct.shape_mismatch(expected_shape)
        
        expected_folds = len(expected_struct.fold_points)
        actual_folds = len(actual_struct.fold_points)
        fold_ratio = actual_folds / max(expected_folds, 1)
        
        expected_len = len(expected_struct.sequence)
        actual_len = len(actual_struct.sequence)
        length_ratio = actual_len / max(expected_len, 1)
        
        # Find missing fold words
        expected_fold_words = expected_struct.get_fold_words()
        actual_fold_words = actual_struct.get_fold_words()
        missing_fold_words = expected_fold_words - actual_fold_words
        
        # Classify deficiency type based on shape analysis
        if shape_sim > 0.9:
            deficiency_type = ShapeDeficiencyType.NONE
            severity = 0.0
            description = "Shape matches well"
            suggested_fix = ""
        elif length_ratio < 0.5:
            deficiency_type = ShapeDeficiencyType.INCOMPLETE
            severity = 0.8 * (1.0 - length_ratio)
            description = f"Output too short (length ratio: {length_ratio:.2f})"
            suggested_fix = "Expand output to include more content"
        elif fold_ratio < 0.3:
            deficiency_type = ShapeDeficiencyType.MISSING_STRUCTURE
            severity = 0.7 * (1.0 - fold_ratio)
            description = f"Missing self-references (fold ratio: {fold_ratio:.2f})"
            suggested_fix = f"Add self-references for: {missing_fold_words}"
        elif shape_sim < 0.5:
            deficiency_type = ShapeDeficiencyType.WRONG_STRUCTURE
            severity = 0.6 * (1.0 - shape_sim)
            description = f"Different structure pattern (similarity: {shape_sim:.2f})"
            suggested_fix = "Restructure to match expected narrative pattern"
        else:
            deficiency_type = ShapeDeficiencyType.PARTIAL
            severity = 0.4 * (1.0 - shape_sim)
            description = f"Partial shape mismatch (similarity: {shape_sim:.2f})"
            suggested_fix = f"Improve self-references, missing: {missing_fold_words}"
        
        return ShapeDeficiency(
            type=deficiency_type,
            severity=severity,
            shape_similarity=shape_sim,
            shape_mismatch=shape_mismatch,
            fold_ratio=fold_ratio,
            length_ratio=length_ratio,
            missing_fold_words=missing_fold_words,
            description=description,
            suggested_fix=suggested_fix
        )
    
    def detect_from_template(self, template_name: str, actual: str) -> Optional[ShapeDeficiency]:
        """Detect deficiency against a learned template."""
        if template_name not in self.structure_templates:
            return None
        
        expected_struct = self.structure_templates[template_name]
        actual_struct = FoldingStructure.from_text(actual)
        
        expected_shape = self.shape_templates[template_name]
        
        shape_sim = expected_struct.shape_similarity(actual_struct)
        shape_mismatch = actual_struct.shape_mismatch(expected_shape)
        
        expected_folds = len(expected_struct.fold_points)
        actual_folds = len(actual_struct.fold_points)
        fold_ratio = actual_folds / max(expected_folds, 1)
        
        expected_len = len(expected_struct.sequence)
        actual_len = len(actual_struct.sequence)
        length_ratio = actual_len / max(expected_len, 1)
        
        missing_fold_words = expected_struct.get_fold_words() - actual_struct.get_fold_words()
        
        # Classify
        if shape_sim > 0.9:
            deficiency_type = ShapeDeficiencyType.NONE
            severity = 0.0
        elif length_ratio < 0.5:
            deficiency_type = ShapeDeficiencyType.INCOMPLETE
            severity = 0.8 * (1.0 - length_ratio)
        elif fold_ratio < 0.3:
            deficiency_type = ShapeDeficiencyType.MISSING_STRUCTURE
            severity = 0.7 * (1.0 - fold_ratio)
        elif shape_sim < 0.5:
            deficiency_type = ShapeDeficiencyType.WRONG_STRUCTURE
            severity = 0.6 * (1.0 - shape_sim)
        else:
            deficiency_type = ShapeDeficiencyType.PARTIAL
            severity = 0.4 * (1.0 - shape_sim)
        
        return ShapeDeficiency(
            type=deficiency_type,
            severity=severity,
            shape_similarity=shape_sim,
            shape_mismatch=shape_mismatch,
            fold_ratio=fold_ratio,
            length_ratio=length_ratio,
            missing_fold_words=missing_fold_words,
            description=f"Template: {template_name}",
            suggested_fix=f"Missing fold words: {missing_fold_words}" if missing_fold_words else ""
        )


def compare_detection_methods(expected: str, actual: str, 
                               expected_contains: List[str] = None) -> Dict:
    """
    Compare old pattern-matching vs new shape-based detection.
    
    Returns comparison metrics for both methods.
    """
    from .gear_improvement_loop import DeficiencyDetectorGear, TestCase
    
    # Old method (pattern matching)
    old_detector = DeficiencyDetectorGear()
    test_case = TestCase(
        input=expected,
        expected_contains=expected_contains or []
    )
    old_deficiencies = old_detector.detect(actual, test_case)
    
    # New method (shape-based)
    new_detector = FoldingDeficiencyDetector()
    new_deficiency = new_detector.detect(expected, actual)
    
    return {
        'old_method': {
            'deficiency_count': len(old_deficiencies),
            'deficiencies': [
                {
                    'type': d.type.value,
                    'severity': d.severity,
                    'description': d.description
                }
                for d in old_deficiencies
            ],
            'max_severity': max((d.severity for d in old_deficiencies), default=0.0)
        },
        'new_method': {
            'type': new_deficiency.type.value,
            'severity': new_deficiency.severity,
            'shape_similarity': new_deficiency.shape_similarity,
            'fold_ratio': new_deficiency.fold_ratio,
            'missing_fold_words': list(new_deficiency.missing_fold_words),
            'description': new_deficiency.description,
            'suggested_fix': new_deficiency.suggested_fix
        }
    }
