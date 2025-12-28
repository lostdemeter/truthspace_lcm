#!/usr/bin/env python3
"""
Polyomino Symmetry Test

Hypothesis: Concepts that co-occur (fit together) have symmetric φ-relationships.

If language is a cascading polyomino puzzle, then:
1. Actor-action pairs should have complementary φ-scores (they "fit")
2. Action-target pairs should have complementary φ-scores
3. Non-co-occurring pairs should NOT have this symmetry

The "fitting" rule: φ^+n × φ^-n ≈ 1 (the inversion horizon)

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.symmetry_encoder import SymmetryEncoder, SymmetrySignature

PHI = 1.618034


@dataclass
class ConceptPair:
    """A pair of concepts with their φ-relationship."""
    word1: str
    word2: str
    phi_score1: float  # φ^+n (word-level symmetry)
    phi_score2: float  # φ^+n (word-level symmetry)
    product: float     # φ^+n × φ^+n - should be ≈ 1 for fitting pairs
    direction1: float  # φ-direction (positive=entity, negative=action)
    direction2: float  # φ-direction
    direction_product: float  # Should be NEGATIVE for fitting pairs (opposite directions)
    co_occurs: bool    # Do they appear together in frames?
    relationship: str  # 'actor-action', 'action-target', 'none'


class PolyominoSymmetryTester:
    """
    Test if co-occurring concepts have symmetric φ-relationships.
    
    The polyomino hypothesis: concepts that "fit" together have
    complementary symmetry scores, like puzzle pieces.
    """
    
    def __init__(self):
        self.encoder = SymmetryEncoder()
        self.frames: List[Dict] = []
        self.co_occurrences: Dict[str, Set[str]] = defaultdict(set)
        self.pair_types: Dict[Tuple[str, str], str] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _is_content_word(self, word: str) -> bool:
        """Filter out function words."""
        function_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be',
                         'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                         'he', 'she', 'it', 'they', 'his', 'her', 'its',
                         'that', 'this', 'from', 'not', 'did', 'do', 'does',
                         'and', 'or', 'but', 'if', 'then', 'so', 'as',
                         'very', 'more', 'down', 'up', 'out', 'about'}
        return word not in function_words and len(word) > 2
    
    def _compute_phi_score(self, word: str) -> float:
        """
        Compute φ-score for a word.
        
        This represents the word's position on the φ-scale:
        - High score = more "outward" (entity-like)
        - Low score = more "inward" (action-like)
        """
        sig = self.encoder.encode(word)
        
        # Combine symmetry features into a single φ-score
        # Using compression and vowel balance as primary indicators
        score = sig.compression * (1 + sig.vowel_balance)
        
        # Normalize to roughly 0.5-2.0 range (around φ-inversion point of 1.0)
        return score * PHI
    
    def _compute_phi_direction(self, word: str) -> float:
        """
        Compute φ-DIRECTION for a word based on its ROLE in frames.
        
        Positive = outward (entity-like, φ^+n) - appears as actor/target
        Negative = inward (action-like, φ^-n) - appears as action
        Zero = at the joint (appears in multiple roles)
        
        The polyomino hypothesis: fitting pairs have OPPOSITE directions.
        """
        # This will be computed AFTER frames are extracted
        # For now, return 0 (will be updated in _compute_role_directions)
        return 0.0
    
    def _compute_role_directions(self):
        """
        Compute φ-directions based on actual roles in frames.
        
        Words that appear as actors/targets → positive (entity)
        Words that appear as actions → negative (action)
        """
        self.role_counts = defaultdict(lambda: {'actor': 0, 'action': 0, 'target': 0})
        
        for frame in self.frames:
            self.role_counts[frame['actor']]['actor'] += 1
            self.role_counts[frame['action']]['action'] += 1
            if frame['target']:
                self.role_counts[frame['target']]['target'] += 1
        
        # Compute direction for each word
        self.word_directions = {}
        for word, counts in self.role_counts.items():
            entity_count = counts['actor'] + counts['target']
            action_count = counts['action']
            total = entity_count + action_count
            
            if total > 0:
                # Direction: +1 = pure entity, -1 = pure action, 0 = mixed
                direction = (entity_count - action_count) / total
            else:
                direction = 0.0
            
            self.word_directions[word] = direction
    
    def get_word_direction(self, word: str) -> float:
        """Get the role-based φ-direction for a word."""
        return self.word_directions.get(word, 0.0)
    
    def extract_frames(self, text: str):
        """Extract actor-action-target frames from text."""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            content = [t for t in tokens if self._is_content_word(t)]
            
            if len(content) >= 2:
                # Simple frame extraction: first content = actor, second = action
                actor = content[0]
                action = content[1]
                target = content[2] if len(content) > 2 else None
                
                frame = {'actor': actor, 'action': action, 'target': target}
                self.frames.append(frame)
                
                # Record co-occurrences
                self.co_occurrences[actor].add(action)
                self.co_occurrences[action].add(actor)
                self.pair_types[(actor, action)] = 'actor-action'
                self.pair_types[(action, actor)] = 'actor-action'
                
                if target:
                    self.co_occurrences[action].add(target)
                    self.co_occurrences[target].add(action)
                    self.pair_types[(action, target)] = 'action-target'
                    self.pair_types[(target, action)] = 'action-target'
    
    def analyze_pair(self, word1: str, word2: str) -> ConceptPair:
        """Analyze the φ-relationship between two words."""
        phi1 = self._compute_phi_score(word1)
        phi2 = self._compute_phi_score(word2)
        dir1 = self.get_word_direction(word1)  # Role-based direction
        dir2 = self.get_word_direction(word2)
        
        # The product should be ≈ 1 for "fitting" pairs (φ-inversion)
        product = phi1 * phi2
        
        # Normalize product to be relative to 1.0
        normalized_product = product / (PHI * PHI)
        
        # Direction product: NEGATIVE means opposite directions (fitting!)
        direction_product = dir1 * dir2
        
        co_occurs = word2 in self.co_occurrences.get(word1, set())
        relationship = self.pair_types.get((word1, word2), 'none')
        
        return ConceptPair(
            word1=word1,
            word2=word2,
            phi_score1=phi1,
            phi_score2=phi2,
            product=normalized_product,
            direction1=dir1,
            direction2=dir2,
            direction_product=direction_product,
            co_occurs=co_occurs,
            relationship=relationship,
        )
    
    def test_polyomino_hypothesis(self) -> Dict:
        """
        Test if co-occurring pairs have more symmetric φ-relationships.
        
        Returns statistics comparing co-occurring vs non-co-occurring pairs.
        """
        # Get all unique words
        all_words = set()
        for frame in self.frames:
            all_words.add(frame['actor'])
            all_words.add(frame['action'])
            if frame['target']:
                all_words.add(frame['target'])
        
        all_words = list(all_words)
        
        # Analyze all pairs
        co_occurring_pairs = []
        non_co_occurring_pairs = []
        
        for i, w1 in enumerate(all_words):
            for w2 in all_words[i+1:]:
                pair = self.analyze_pair(w1, w2)
                
                if pair.co_occurs:
                    co_occurring_pairs.append(pair)
                else:
                    non_co_occurring_pairs.append(pair)
        
        # Compute statistics
        def distance_from_unity(pairs):
            """How far are the products from 1.0 (the inversion horizon)?"""
            if not pairs:
                return float('inf')
            return np.mean([abs(p.product - 1.0) for p in pairs])
        
        def product_stats(pairs):
            if not pairs:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            products = [p.product for p in pairs]
            return {
                'mean': np.mean(products),
                'std': np.std(products),
                'min': np.min(products),
                'max': np.max(products),
            }
        
        def direction_stats(pairs):
            """Stats on direction products - negative = opposite directions = fitting."""
            if not pairs:
                return {'mean': 0, 'negative_ratio': 0}
            dir_products = [p.direction_product for p in pairs]
            negative_count = sum(1 for d in dir_products if d < 0)
            return {
                'mean': np.mean(dir_products),
                'negative_ratio': negative_count / len(pairs),
            }
        
        return {
            'co_occurring': {
                'count': len(co_occurring_pairs),
                'distance_from_unity': distance_from_unity(co_occurring_pairs),
                'stats': product_stats(co_occurring_pairs),
                'direction_stats': direction_stats(co_occurring_pairs),
                'pairs': co_occurring_pairs[:10],
            },
            'non_co_occurring': {
                'count': len(non_co_occurring_pairs),
                'distance_from_unity': distance_from_unity(non_co_occurring_pairs),
                'stats': product_stats(non_co_occurring_pairs),
                'direction_stats': direction_stats(non_co_occurring_pairs),
                'pairs': non_co_occurring_pairs[:10],
            },
        }


def run_experiment():
    """Test the polyomino symmetry hypothesis."""
    print("=" * 70)
    print("POLYOMINO SYMMETRY TEST")
    print("=" * 70)
    print()
    print("Hypothesis: Co-occurring concepts have symmetric φ-relationships.")
    print("            Their φ-products should be closer to 1.0 (inversion horizon).")
    print()
    
    # Test corpus
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
    Darcy looked at Elizabeth proudly. She ignored him completely.
    Elizabeth danced gracefully. Darcy watched her intently.
    Mr Bennet read his newspaper. Mrs Bennet worried about her daughters.
    Jane smiled sweetly. Bingley fell in love immediately.
    """
    
    # Run test
    tester = PolyominoSymmetryTester()
    tester.extract_frames(corpus)
    tester._compute_role_directions()  # Compute role-based directions
    
    print(f"Extracted {len(tester.frames)} frames")
    print()
    
    # Show sample frames
    print("Sample frames:")
    for frame in tester.frames[:5]:
        print(f"  {frame['actor']} → {frame['action']} → {frame['target']}")
    print()
    
    # Test hypothesis
    results = tester.test_polyomino_hypothesis()
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    print("CO-OCCURRING PAIRS (should fit together):")
    print(f"  Count: {results['co_occurring']['count']}")
    print(f"  Distance from unity (1.0): {results['co_occurring']['distance_from_unity']:.4f}")
    print(f"  Direction stats: {results['co_occurring']['direction_stats']}")
    print(f"    (negative_ratio = pairs with OPPOSITE directions = fitting)")
    print()
    
    print("NON-CO-OCCURRING PAIRS (should NOT fit):")
    print(f"  Count: {results['non_co_occurring']['count']}")
    print(f"  Distance from unity (1.0): {results['non_co_occurring']['distance_from_unity']:.4f}")
    print(f"  Direction stats: {results['non_co_occurring']['direction_stats']}")
    print()
    
    # Sample pairs
    print("Sample CO-OCCURRING pairs:")
    for pair in results['co_occurring']['pairs'][:5]:
        dir_sign = "←→" if pair.direction_product < 0 else "→→"
        print(f"  {pair.word1:12} {dir_sign} {pair.word2:12} dir={pair.direction_product:+.3f} ({pair.relationship})")
    print()
    
    print("Sample NON-CO-OCCURRING pairs:")
    for pair in results['non_co_occurring']['pairs'][:5]:
        dir_sign = "←→" if pair.direction_product < 0 else "→→"
        print(f"  {pair.word1:12} {dir_sign} {pair.word2:12} dir={pair.direction_product:+.3f}")
    print()
    
    # Evaluate hypothesis
    print("=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    print()
    
    co_neg_ratio = results['co_occurring']['direction_stats']['negative_ratio']
    non_neg_ratio = results['non_co_occurring']['direction_stats']['negative_ratio']
    
    print("TEST 1: Direction Product (opposite directions = fitting)")
    print(f"  Co-occurring pairs with opposite directions: {co_neg_ratio:.1%}")
    print(f"  Non-co-occurring pairs with opposite directions: {non_neg_ratio:.1%}")
    print()
    
    if co_neg_ratio > non_neg_ratio:
        ratio = co_neg_ratio / non_neg_ratio if non_neg_ratio > 0 else float('inf')
        print(f"✅ POLYOMINO HYPOTHESIS SUPPORTED!")
        print(f"   Co-occurring pairs are {ratio:.2f}x more likely to have opposite directions.")
        print()
        print("   This suggests concepts that 'fit' together have COMPLEMENTARY")
        print("   φ-directions, like polyomino pieces with matching edges.")
    else:
        print(f"⚠️  Direction test inconclusive or not supported.")
        print(f"   Co-occurring pairs don't show more opposite directions.")
    
    print()
    
    # Additional analysis: by relationship type
    print("=" * 70)
    print("ANALYSIS BY RELATIONSHIP TYPE")
    print("=" * 70)
    print()
    
    actor_action = [p for p in results['co_occurring']['pairs'] if p.relationship == 'actor-action']
    action_target = [p for p in results['co_occurring']['pairs'] if p.relationship == 'action-target']
    
    if actor_action:
        aa_dist = np.mean([abs(p.product - 1.0) for p in actor_action])
        print(f"Actor-Action pairs: distance from unity = {aa_dist:.4f}")
        for p in actor_action[:3]:
            print(f"  {p.word1:12} → {p.word2:12}: φ₁={p.phi_score1:.2f} φ₂={p.phi_score2:.2f} product={p.product:.3f}")
    
    print()
    
    if action_target:
        at_dist = np.mean([abs(p.product - 1.0) for p in action_target])
        print(f"Action-Target pairs: distance from unity = {at_dist:.4f}")
        for p in action_target[:3]:
            print(f"  {p.word1:12} → {p.word2:12}: φ₁={p.phi_score1:.2f} φ₂={p.phi_score2:.2f} product={p.product:.3f}")
    
    return tester, results


if __name__ == "__main__":
    tester, results = run_experiment()
