#!/usr/bin/env python3
"""
Error Correction Quaternion (Q4)

Hypothesis: Error detection and correction can be treated as a quaternion operation.

The four quaternions:
  Q1 (Concept):  What fits together (polyomino fitting)
  Q2 (Output):   How to express it (style, certainty)
  Q3 (Morpho):   How words transform (conjugation)
  Q4 (Error):    What's wrong and how to fix it

Q4 axes:
  X4: Semantic Error    (-1 = wrong meaning, 0 = correct, +1 = overcorrected)
  Y4: Syntactic Error   (-1 = ungrammatical, 0 = correct, +1 = overcorrected)
  Z4: Coherence Error   (-1 = disconnected, 0 = coherent, +1 = overcorrected)
  W4: Fit Error         (-1 = doesn't fit, 0 = perfect fit, +1 = forced fit)

The W4 axis (Fit Error) measures distance from the critical line (σ = 0.5).
Error = WHERE to add structure.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.polyomino_generator import PolyominoGenerator, Frame
from experiments.morphological_quaternion import MorphoQuaternion, MorphologicalTransformer

PHI = 1.618034


@dataclass
class ErrorQuaternion:
    """
    Q4: Error space quaternion.
    
    Measures how far the output is from "correct" on each axis.
    Values near 0 = no error, values away from 0 = error detected.
    
    X: Semantic error   (wrong entity/action pairing)
    Y: Syntactic error  (wrong grammar/conjugation)
    Z: Coherence error  (disconnected/illogical)
    W: Fit error        (polyomino pieces don't fit)
    """
    x: float = 0.0  # Semantic
    y: float = 0.0  # Syntactic
    z: float = 0.0  # Coherence
    w: float = 0.0  # Fit (symmetry axis)
    
    @property
    def total_error(self) -> float:
        """Total error magnitude (distance from origin)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    @property
    def dominant_error(self) -> str:
        """Which error type is largest?"""
        errors = {
            'semantic': abs(self.x),
            'syntactic': abs(self.y),
            'coherence': abs(self.z),
            'fit': abs(self.w),
        }
        return max(errors, key=errors.get)
    
    @property
    def needs_correction(self) -> bool:
        """Is the error significant enough to correct?"""
        return self.total_error > 0.3
    
    def describe(self) -> str:
        return f"Q4(sem={self.x:.2f}, syn={self.y:.2f}, coh={self.z:.2f}, fit={self.w:.2f})"


@dataclass
class Correction:
    """A suggested correction based on Q4 analysis."""
    target_quaternion: str  # 'Q1', 'Q2', 'Q3'
    axis: str               # Which axis to adjust
    adjustment: float       # How much to adjust
    reason: str             # Why this correction


class ErrorDetector:
    """
    Detect errors in generated text using quaternion analysis.
    
    The key insight: errors are deviations from the critical line (σ = 0.5).
    Each error type maps to a specific quaternion that needs adjustment.
    """
    
    def __init__(self, concept_generator: PolyominoGenerator):
        self.concept_gen = concept_generator
        self.morpho = MorphologicalTransformer()
        
        # Known good patterns (learned from corpus)
        self.valid_actor_actions: Dict[str, set] = {}
        self.valid_action_targets: Dict[str, set] = {}
        
        # Build from concept generator's learned patterns
        self._build_valid_patterns()
    
    def _build_valid_patterns(self):
        """Build valid pattern sets from learned data."""
        for actor, actions in self.concept_gen.actor_actions.items():
            self.valid_actor_actions[actor] = set(actions.keys())
        
        for action, targets in self.concept_gen.action_targets.items():
            self.valid_action_targets[action] = set(targets.keys())
    
    def detect_semantic_error(self, frame: Frame) -> float:
        """
        Detect semantic errors (wrong entity/action pairing).
        
        Returns error magnitude: 0 = no error, ±1 = severe error
        """
        actor = frame.actor
        action = frame.action
        target = frame.target
        
        error = 0.0
        
        # Check if actor-action pairing is valid
        if actor in self.valid_actor_actions:
            if action not in self.valid_actor_actions[actor]:
                # Unknown pairing - might be wrong
                error -= 0.5
        else:
            # Unknown actor - uncertain
            error -= 0.3
        
        # Check if action-target pairing is valid
        if target and action in self.valid_action_targets:
            if target not in self.valid_action_targets[action]:
                error -= 0.3
        
        # Check direction compatibility (polyomino fit)
        if actor in self.concept_gen.concepts and action in self.concept_gen.concepts:
            dir1 = self.concept_gen.concepts[actor].direction
            dir2 = self.concept_gen.concepts[action].direction
            
            # Should have opposite directions
            if dir1 * dir2 >= 0:
                error -= 0.5  # Same direction = semantic mismatch
        
        return max(-1.0, min(1.0, error))
    
    def detect_syntactic_error(self, sentence: str) -> float:
        """
        Detect syntactic errors (grammar/conjugation issues).
        
        Returns error magnitude: 0 = no error, ±1 = severe error
        """
        error = 0.0
        
        # Check for common conjugation errors
        bad_patterns = [
            (r'\b(\w+)s\s+(\w+)s\b', -0.3),  # Double -s (verbs verbs)
            (r'\b(is|are|was|were)\s+\w+ed\b', -0.2),  # is examined (passive ok, but check)
            (r'\b\w+ss\b', -0.1),  # Double s at end (might be ok:lass, boss)
            (r'\bhe\s+\w+[^s]\b', -0.2),  # he examine (missing -s)
            (r'\bthey\s+\w+s\b', -0.2),  # they examines (wrong number)
        ]
        
        for pattern, penalty in bad_patterns:
            if re.search(pattern, sentence.lower()):
                error += penalty
        
        # Check for article errors
        if re.search(r'\ba\s+[aeiou]', sentence.lower()):
            error -= 0.3  # "a apple" should be "an apple"
        
        # Check for missing articles before common nouns
        common_nouns = {'evidence', 'room', 'journal', 'newspaper', 'garden', 
                       'building', 'window', 'scene', 'tea', 'hole'}
        for noun in common_nouns:
            if re.search(rf'\b{noun}\b', sentence.lower()):
                if not re.search(rf'\b(the|a|an)\s+{noun}\b', sentence.lower()):
                    error -= 0.2
        
        return max(-1.0, min(1.0, error))
    
    def detect_coherence_error(self, sentences: List[str]) -> float:
        """
        Detect coherence errors (disconnected/illogical flow).
        
        Returns error magnitude: 0 = no error, ±1 = severe error
        """
        if len(sentences) < 2:
            return 0.0
        
        error = 0.0
        
        # Check for entity continuity
        prev_entities = set()
        for sentence in sentences:
            # Extract capitalized words (likely entities)
            entities = set(re.findall(r'\b[A-Z][a-z]+\b', sentence))
            
            if prev_entities and not entities.intersection(prev_entities):
                # No shared entities - might be disconnected
                error -= 0.3
            
            prev_entities = entities
        
        # Check for repetition (same sentence repeated)
        if len(sentences) != len(set(sentences)):
            error -= 0.5
        
        return max(-1.0, min(1.0, error))
    
    def detect_fit_error(self, frame: Frame) -> float:
        """
        Detect fit errors (polyomino pieces don't fit).
        
        This is the W4 axis - the symmetry axis.
        Returns error magnitude: 0 = perfect fit, ±1 = no fit
        """
        if frame.actor not in self.concept_gen.concepts:
            return -0.5  # Unknown actor
        
        if frame.action not in self.concept_gen.concepts:
            return -0.5  # Unknown action
        
        # Get directions
        dir_actor = self.concept_gen.concepts[frame.actor].direction
        dir_action = self.concept_gen.concepts[frame.action].direction
        
        # Compute fit: opposite directions should multiply to negative
        fit_product = dir_actor * dir_action
        
        if fit_product < 0:
            # Good fit - opposite directions
            return 0.0
        elif fit_product == 0:
            # One is neutral - uncertain fit
            return -0.3
        else:
            # Bad fit - same directions
            return -abs(fit_product)
    
    def analyze(self, frame: Frame, sentence: str, 
                context: List[str] = None) -> ErrorQuaternion:
        """
        Analyze a generated output and return the error quaternion.
        """
        x = self.detect_semantic_error(frame)
        y = self.detect_syntactic_error(sentence)
        z = self.detect_coherence_error(context or [sentence])
        w = self.detect_fit_error(frame)
        
        return ErrorQuaternion(x=x, y=y, z=z, w=w)
    
    def suggest_correction(self, q4: ErrorQuaternion) -> Optional[Correction]:
        """
        Based on the error quaternion, suggest which quaternion to adjust.
        """
        if not q4.needs_correction:
            return None
        
        dominant = q4.dominant_error
        
        if dominant == 'semantic':
            return Correction(
                target_quaternion='Q1',
                axis='fitting',
                adjustment=-q4.x,
                reason=f"Semantic error ({q4.x:.2f}): regenerate with different concept pairing"
            )
        
        elif dominant == 'syntactic':
            return Correction(
                target_quaternion='Q3',
                axis='morphology',
                adjustment=-q4.y,
                reason=f"Syntactic error ({q4.y:.2f}): adjust conjugation parameters"
            )
        
        elif dominant == 'coherence':
            return Correction(
                target_quaternion='Q2',
                axis='depth',
                adjustment=-q4.z,
                reason=f"Coherence error ({q4.z:.2f}): adjust output continuity"
            )
        
        else:  # fit
            return Correction(
                target_quaternion='Q1',
                axis='direction',
                adjustment=-q4.w,
                reason=f"Fit error ({q4.w:.2f}): find concepts with opposite φ-directions"
            )


class ErrorCorrectionLoop:
    """
    Feedback loop that uses Q4 to correct Q1-Q3.
    
    The loop:
    1. Generate with Q1 → Q3 → Q2
    2. Analyze with Q4
    3. If error detected, adjust the indicated quaternion
    4. Regenerate
    """
    
    def __init__(self, concept_gen: PolyominoGenerator):
        self.concept_gen = concept_gen
        self.detector = ErrorDetector(concept_gen)
        self.morpho = MorphologicalTransformer()
        self.max_iterations = 3
    
    def generate_with_correction(self, seed: str, 
                                  morpho_q: MorphoQuaternion) -> Tuple[str, List[ErrorQuaternion]]:
        """
        Generate text with error correction loop.
        
        Returns the final sentence and the error history.
        """
        error_history = []
        
        for iteration in range(self.max_iterations):
            # Generate frame (Q1)
            frame = self.concept_gen.generate_frame(seed)
            
            # Transform verb (Q3)
            base_action = self.morpho._get_base(frame.action)
            verb = self.morpho.transform(base_action, morpho_q)
            
            # Build sentence (simplified Q2)
            actor = frame.actor.title()
            target = frame.target
            
            if target:
                # Filter bad targets
                bad_targets = {'tall', 'small', 'confused', 'angrily', 'intently',
                              'gracefully', 'wildly', 'slowly', 'quickly', 'him', 'her'}
                if target in bad_targets or target.endswith('ly'):
                    target = None
            
            if target:
                if target in {'evidence', 'room', 'journal', 'newspaper', 'garden',
                             'building', 'window', 'scene', 'tea', 'hole'}:
                    sentence = f"{actor} {verb} the {target}."
                else:
                    sentence = f"{actor} {verb} {target.title()}."
            else:
                sentence = f"{actor} {verb}."
            
            # Analyze with Q4
            q4 = self.detector.analyze(frame, sentence)
            error_history.append(q4)
            
            # Check if correction needed
            if not q4.needs_correction:
                return sentence, error_history
            
            # Get correction suggestion
            correction = self.detector.suggest_correction(q4)
            
            if correction:
                # Apply correction by trying a different seed or regenerating
                if correction.target_quaternion == 'Q1':
                    # Try a different starting point
                    if frame.target and frame.target in self.concept_gen.concepts:
                        seed = frame.target
                    else:
                        # Pick a random entity
                        if self.concept_gen.entities:
                            import random
                            seed = random.choice(self.concept_gen.entities)
        
        # Return best effort after max iterations
        return sentence, error_history


def run_experiment():
    """Test the error correction quaternion."""
    print("=" * 70)
    print("ERROR CORRECTION QUATERNION (Q4) EXPERIMENT")
    print("=" * 70)
    print()
    print("Hypothesis: Error detection/correction is a quaternion operation.")
    print()
    print("Q4 axes:")
    print("  X: Semantic error   (wrong entity/action pairing)")
    print("  Y: Syntactic error  (grammar/conjugation)")
    print("  Z: Coherence error  (disconnected flow)")
    print("  W: Fit error        (polyomino pieces don't fit)")
    print()
    
    # Training corpus
    corpus = """
    Holmes examined the evidence carefully. Watson watched from the doorway.
    The detective studied the footprints. He noticed something unusual.
    Holmes said to Watson that the case was elementary.
    Watson replied that he did not understand.
    The inspector arrived at the scene. Lestrade questioned the witnesses.
    Holmes observed the room methodically. He found a clue near the window.
    Watson wrote in his journal. The doctor recorded every detail.
    Holmes deduced the killer identity. He explained his reasoning.
    The criminal fled through the garden. Holmes pursued him quickly.
    Watson called for help. The police surrounded the building.
    Holmes captured the villain. Justice was served.
    Alice fell down the rabbit hole. She wondered where she was going.
    The Queen shouted angrily. Alice felt confused and scared.
    The Cheshire Cat smiled mysteriously. He disappeared slowly.
    Alice grew very tall. She shrank very small.
    The Mad Hatter laughed wildly. He poured more tea.
    """
    
    # Create generator and learn
    concept_gen = PolyominoGenerator()
    concept_gen.learn_from_text(corpus)
    
    detector = ErrorDetector(concept_gen)
    
    print(f"Learned {len(concept_gen.concepts)} concepts")
    print()
    
    # Test error detection on various frames
    print("=" * 70)
    print("ERROR DETECTION TESTS")
    print("=" * 70)
    print()
    
    test_cases = [
        # (actor, action, target, sentence, description)
        ('holmes', 'examined', 'evidence', 'Holmes examined the evidence.', 'Good sentence'),
        ('holmes', 'examined', 'watson', 'Holmes examined Watson.', 'Unusual pairing'),
        ('evidence', 'examined', 'holmes', 'Evidence examined Holmes.', 'Wrong direction'),
        ('holmes', 'examine', 'evidence', 'Holmes examine the evidence.', 'Missing -s'),
        ('holmes', 'examined', None, 'Holmes examined.', 'No target'),
    ]
    
    for actor, action, target, sentence, desc in test_cases:
        frame = Frame(actor=actor, action=action, target=target, fit_score=0)
        q4 = detector.analyze(frame, sentence)
        
        print(f"{desc}:")
        print(f"  Frame: {actor} → {action} → {target}")
        print(f"  Sentence: \"{sentence}\"")
        print(f"  {q4.describe()}")
        print(f"  Total error: {q4.total_error:.3f}")
        print(f"  Dominant: {q4.dominant_error}")
        print(f"  Needs correction: {q4.needs_correction}")
        
        if q4.needs_correction:
            correction = detector.suggest_correction(q4)
            if correction:
                print(f"  Suggestion: {correction.reason}")
        print()
    
    # Test error correction loop
    print("=" * 70)
    print("ERROR CORRECTION LOOP")
    print("=" * 70)
    print()
    
    loop = ErrorCorrectionLoop(concept_gen)
    morpho_q = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd sing present simple
    
    for seed in ['holmes', 'watson', 'alice', 'evidence']:
        print(f"Seed: {seed}")
        sentence, error_history = loop.generate_with_correction(seed, morpho_q)
        
        print(f"  Final: \"{sentence}\"")
        print(f"  Iterations: {len(error_history)}")
        
        for i, q4 in enumerate(error_history):
            status = "✓" if not q4.needs_correction else "→"
            print(f"    [{i+1}] {status} error={q4.total_error:.3f} ({q4.dominant_error})")
        print()
    
    # Analyze error distribution
    print("=" * 70)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Generate many frames and analyze error distribution
    errors = {'semantic': [], 'syntactic': [], 'coherence': [], 'fit': []}
    
    for _ in range(50):
        frame = concept_gen.generate_frame()
        base = MorphologicalTransformer()._get_base(frame.action)
        verb = MorphologicalTransformer().transform(base, morpho_q)
        
        actor = frame.actor.title()
        if frame.target:
            sentence = f"{actor} {verb} {frame.target}."
        else:
            sentence = f"{actor} {verb}."
        
        q4 = detector.analyze(frame, sentence)
        
        errors['semantic'].append(abs(q4.x))
        errors['syntactic'].append(abs(q4.y))
        errors['coherence'].append(abs(q4.z))
        errors['fit'].append(abs(q4.w))
    
    print("Average error by type (lower = better):")
    for error_type, values in errors.items():
        avg = np.mean(values)
        print(f"  {error_type:12}: {avg:.3f} {'⚠️' if avg > 0.3 else '✓'}")
    
    print()
    
    # Test Q4 on deliberately bad frames vs good frames
    print("=" * 70)
    print("Q4 DISCRIMINATES GOOD vs BAD FRAMES")
    print("=" * 70)
    print()
    
    # Generate good frames (using polyomino fitting)
    good_errors = []
    for _ in range(20):
        frame = concept_gen.generate_frame()
        base = MorphologicalTransformer()._get_base(frame.action)
        verb = MorphologicalTransformer().transform(base, morpho_q)
        actor = frame.actor.title()
        sentence = f"{actor} {verb}."
        q4 = detector.analyze(frame, sentence)
        good_errors.append(q4.total_error)
    
    # Generate bad frames (random pairings, ignoring fit)
    bad_errors = []
    import random
    all_words = list(concept_gen.concepts.keys())
    for _ in range(20):
        # Random pairing (likely wrong direction)
        actor = random.choice(all_words)
        action = random.choice(all_words)
        target = random.choice(all_words) if random.random() > 0.5 else None
        frame = Frame(actor=actor, action=action, target=target, fit_score=0)
        sentence = f"{actor.title()} {action} {target or ''}."
        q4 = detector.analyze(frame, sentence)
        bad_errors.append(q4.total_error)
    
    avg_good = np.mean(good_errors)
    avg_bad = np.mean(bad_errors)
    
    print(f"Average error for GOOD frames (polyomino fit): {avg_good:.3f}")
    print(f"Average error for BAD frames (random pairing): {avg_bad:.3f}")
    print(f"Discrimination ratio: {avg_bad/avg_good:.2f}x" if avg_good > 0 else "")
    print()
    
    if avg_bad > avg_good * 1.5:
        print("✅ Q4 DISCRIMINATES between good and bad frames!")
        print("   Error quaternion successfully detects when pieces don't fit.")
    else:
        print("⚠️  Q4 discrimination is weak - may need refinement.")
    
    return detector, loop


if __name__ == "__main__":
    detector, loop = run_experiment()
