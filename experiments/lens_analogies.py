#!/usr/bin/env python3
"""
Lens-Aware Analogies

Key insight: Analogies are the same structural relationship viewed through different lenses.

For "holmes:detective :: watson:?":
- LITERAL lens: What category is Watson? → doctor
- BEHAVIORAL lens: What acts like Watson at detective's φ? → teacher
- RELATIONAL lens: What connects to Watson like detective connects to Holmes? → companion
- INTRINSIC lens: What has Watson's intrinsic properties shifted by the A→B delta? → assistant

The "correct" answer depends on which LENS you're using to view the analogy.

Author: Lesley Gushurst
License: GPLv3
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.distilled_lcm import DistilledLCM, Concept
from truthspace_lcm.core.semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES


class AnalogyLens(Enum):
    """Different perspectives for viewing analogies."""
    LITERAL = "literal"         # Category-based: A is-a B, C is-a ?
    BEHAVIORAL = "behavioral"   # Action-based: A acts like B, C acts like ?
    RELATIONAL = "relational"   # Connection-based: A relates to B, C relates to ?
    INTRINSIC = "intrinsic"     # Property-based: A has properties of B, C has properties of ?


@dataclass
class LensAnalogyResult:
    """Result of an analogy through a specific lens."""
    lens: AnalogyLens
    answer: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]]


class LensAnalogies:
    """
    Solve analogies through different perspective lenses.
    
    The same analogy A:B :: C:? can have different answers depending
    on which lens you view it through.
    """
    
    def __init__(self, distilled_path: str = "truthspace_lcm/concepts_distilled.json"):
        self.distilled = DistilledLCM()
        self.distilled.load(distilled_path)
        
        # Category words for literal lens
        self.category_words = {
            'detective', 'doctor', 'scientist', 'teacher', 'writer',
            'philosopher', 'artist', 'leader', 'hero', 'villain',
            'companion', 'assistant', 'friend', 'enemy', 'partner',
            'physicist', 'biologist', 'chemist', 'mathematician',
            'chronicler', 'narrator', 'observer', 'participant',
            'science', 'field', 'discipline', 'study', 'theory',
        }
    
    def analogy_literal(self, a: str, b: str, c: str, k: int = 5) -> LensAnalogyResult:
        """
        LITERAL lens: Category-based analogy.
        
        "A is-a B, so C is-a ?"
        
        Looks for explicit category relationships.
        """
        concept_c = self.distilled.get_concept(c.lower())
        if not concept_c:
            return LensAnalogyResult(
                lens=AnalogyLens.LITERAL,
                answer="(unknown)",
                confidence=0.0,
                reasoning=f"Concept '{c}' not found",
                alternatives=[]
            )
        
        # Look for category words in C's targets
        categories = []
        for target, count in concept_c.targets:
            if target in self.category_words:
                categories.append((target, count))
        
        # Also check related concepts
        related = self.distilled.find_related(c.lower(), top_k=20)
        for word, score in related:
            if word in self.category_words and word not in [cat for cat, _ in categories]:
                categories.append((word, score))
        
        if categories:
            categories.sort(key=lambda x: -x[1])
            answer = categories[0][0]
            confidence = min(1.0, categories[0][1] / 3)
            reasoning = f"'{c}' has category '{answer}' in corpus"
        else:
            answer = "(no category found)"
            confidence = 0.0
            reasoning = f"No explicit category for '{c}' in corpus"
        
        return LensAnalogyResult(
            lens=AnalogyLens.LITERAL,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=categories[:k]
        )
    
    def analogy_behavioral(self, a: str, b: str, c: str, k: int = 5) -> LensAnalogyResult:
        """
        BEHAVIORAL lens: Action-based analogy.
        
        "A acts at φ_A, B acts at φ_B, so C should relate to something at φ_C + (φ_B - φ_A)"
        
        Uses φ-direction to find behaviorally similar concepts.
        """
        concept_a = self.distilled.get_concept(a.lower())
        concept_b = self.distilled.get_concept(b.lower())
        concept_c = self.distilled.get_concept(c.lower())
        
        if not all([concept_a, concept_b, concept_c]):
            return LensAnalogyResult(
                lens=AnalogyLens.BEHAVIORAL,
                answer="(unknown)",
                confidence=0.0,
                reasoning="One or more concepts not found",
                alternatives=[]
            )
        
        # Calculate φ delta
        phi_delta = concept_b.phi_direction - concept_a.phi_direction
        expected_phi = concept_c.phi_direction + phi_delta
        expected_phi = max(-1, min(1, expected_phi))
        
        # Find concepts near expected φ
        candidates = []
        for word, concept in self.distilled.concepts.items():
            if word in {a.lower(), b.lower(), c.lower()}:
                continue
            if concept.frequency > 500:  # Skip very common words
                continue
            
            phi_diff = abs(concept.phi_direction - expected_phi)
            if phi_diff < 0.5:
                # Bonus for category words
                score = 1.0 - phi_diff
                if word in self.category_words:
                    score += 0.5
                candidates.append((word, score, concept.phi_direction))
        
        candidates.sort(key=lambda x: -x[1])
        
        if candidates:
            answer = candidates[0][0]
            confidence = candidates[0][1]
            reasoning = f"φ({a})={concept_a.phi_direction:.2f}, φ({b})={concept_b.phi_direction:.2f}, " \
                       f"φ({c})={concept_c.phi_direction:.2f} → expected φ={expected_phi:.2f}, " \
                       f"'{answer}' has φ={candidates[0][2]:.2f}"
        else:
            answer = "(no match)"
            confidence = 0.0
            reasoning = f"No concepts found near expected φ={expected_phi:.2f}"
        
        return LensAnalogyResult(
            lens=AnalogyLens.BEHAVIORAL,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[(w, s) for w, s, _ in candidates[:k]]
        )
    
    def analogy_relational(self, a: str, b: str, c: str, k: int = 5) -> LensAnalogyResult:
        """
        RELATIONAL lens: Connection-based analogy.
        
        "A relates to B, so C relates to ?"
        
        Looks at what C connects to, filtered by the A→B relationship type.
        """
        concept_a = self.distilled.get_concept(a.lower())
        concept_b = self.distilled.get_concept(b.lower())
        concept_c = self.distilled.get_concept(c.lower())
        
        if not all([concept_a, concept_b, concept_c]):
            return LensAnalogyResult(
                lens=AnalogyLens.RELATIONAL,
                answer="(unknown)",
                confidence=0.0,
                reasoning="One or more concepts not found",
                alternatives=[]
            )
        
        # What kind of relationship is A→B?
        # Check if B is in A's targets, actions, or related
        a_targets = set(t for t, _ in concept_a.targets[:10])
        a_actions = set(act for act, _ in concept_a.actions[:10])
        
        relationship_type = "related"
        if b.lower() in a_targets:
            relationship_type = "target"
        elif b.lower() in a_actions:
            relationship_type = "action"
        
        # Find C's connections of the same type
        c_targets = [(t, c) for t, c in concept_c.targets[:10]]
        c_related = self.distilled.find_related(c.lower(), top_k=10)
        
        # Filter by category words (often what we're looking for)
        candidates = []
        for target, count in c_targets:
            if target in self.category_words:
                candidates.append((target, count * 2))  # Boost categories
            else:
                candidates.append((target, count))
        
        for word, score in c_related:
            if word not in [t for t, _ in candidates]:
                if word in self.category_words:
                    candidates.append((word, score * 2))
                else:
                    candidates.append((word, score))
        
        # Filter out input words
        candidates = [(w, s) for w, s in candidates if w not in {a.lower(), b.lower(), c.lower()}]
        candidates.sort(key=lambda x: -x[1])
        
        if candidates:
            answer = candidates[0][0]
            confidence = min(1.0, candidates[0][1] / 3)
            reasoning = f"'{c}' connects to '{answer}' (relationship type: {relationship_type})"
        else:
            answer = "(no connection)"
            confidence = 0.0
            reasoning = f"No connections found for '{c}'"
        
        return LensAnalogyResult(
            lens=AnalogyLens.RELATIONAL,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=candidates[:k]
        )
    
    def analogy_intrinsic(self, a: str, b: str, c: str, k: int = 5) -> LensAnalogyResult:
        """
        INTRINSIC lens: Property-based analogy.
        
        "A has intrinsic properties, B has intrinsic properties, 
         so C should relate to something with C's properties + (B - A) delta"
        
        Uses SemanticQuaternion for intrinsic property matching.
        """
        # Get intrinsic properties
        sq_a = DEFAULT_SEMANTIC_FEATURES.get(a.lower())
        sq_b = DEFAULT_SEMANTIC_FEATURES.get(b.lower())
        sq_c = DEFAULT_SEMANTIC_FEATURES.get(c.lower())
        
        if not all([sq_a, sq_b, sq_c]):
            # Fall back to behavioral if intrinsic not defined
            missing = [w for w, sq in [(a, sq_a), (b, sq_b), (c, sq_c)] if sq is None]
            return LensAnalogyResult(
                lens=AnalogyLens.INTRINSIC,
                answer="(not defined)",
                confidence=0.0,
                reasoning=f"Intrinsic properties not defined for: {missing}",
                alternatives=[]
            )
        
        # Calculate property delta
        delta = sq_b - sq_a
        expected = sq_c + delta
        
        # Find concepts with similar intrinsic properties
        candidates = []
        for word, sq in DEFAULT_SEMANTIC_FEATURES.items():
            if word in {a.lower(), b.lower(), c.lower()}:
                continue
            
            # Distance in 4D quaternion space
            distance = expected.distance(sq)
            if distance < 2.0:  # Threshold
                similarity = 1.0 / (1.0 + distance)
                candidates.append((word, similarity, distance))
        
        candidates.sort(key=lambda x: -x[1])
        
        if candidates:
            answer = candidates[0][0]
            confidence = candidates[0][1]
            reasoning = f"SQ({a})→SQ({b}) delta applied to SQ({c}) → closest is '{answer}' (dist={candidates[0][2]:.2f})"
        else:
            answer = "(no match)"
            confidence = 0.0
            reasoning = "No concepts with similar intrinsic properties"
        
        return LensAnalogyResult(
            lens=AnalogyLens.INTRINSIC,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[(w, s) for w, s, _ in candidates[:k]]
        )
    
    def analogy_all_lenses(self, a: str, b: str, c: str) -> Dict[AnalogyLens, LensAnalogyResult]:
        """Solve an analogy through all lenses."""
        return {
            AnalogyLens.LITERAL: self.analogy_literal(a, b, c),
            AnalogyLens.BEHAVIORAL: self.analogy_behavioral(a, b, c),
            AnalogyLens.RELATIONAL: self.analogy_relational(a, b, c),
            AnalogyLens.INTRINSIC: self.analogy_intrinsic(a, b, c),
        }
    
    def combine_lenses(self, a: str, b: str, c: str, 
                        strategy: str = "weighted") -> Tuple[str, float, Dict]:
        """
        Combine all lenses to get a consensus answer.
        
        Strategies:
        - "voting": Each lens votes, majority wins
        - "weighted": Weight by confidence, sum scores
        - "consensus": Only return if multiple lenses agree
        - "best": Return highest confidence answer
        - "intrinsic_priority": Prefer intrinsic if high confidence
        """
        results = self.analogy_all_lenses(a, b, c)
        
        # Special case: if intrinsic lens has very high confidence, trust it
        if strategy == "intrinsic_priority":
            intrinsic = results.get(AnalogyLens.INTRINSIC)
            if intrinsic and intrinsic.confidence >= 0.9:
                return intrinsic.answer, intrinsic.confidence, {
                    "sources": ["intrinsic"],
                    "strategy": "intrinsic_priority",
                    "reason": "High-confidence intrinsic match"
                }
            # Fall through to weighted if intrinsic not confident
            strategy = "weighted"
        
        # Collect all answers with their scores
        answer_scores: Dict[str, float] = {}
        answer_sources: Dict[str, List[str]] = {}
        
        # Lens weights - intrinsic gets bonus when confident
        lens_weights = {
            AnalogyLens.LITERAL: 1.0,
            AnalogyLens.BEHAVIORAL: 0.8,  # Slightly lower - can be noisy
            AnalogyLens.RELATIONAL: 0.7,  # Can pick up noise
            AnalogyLens.INTRINSIC: 1.5,   # Bonus for semantic properties
        }
        
        for lens, result in results.items():
            if result.confidence > 0:
                answer = result.answer.lower()
                lens_weight = lens_weights.get(lens, 1.0)
                
                if strategy == "voting":
                    score = 1.0 * lens_weight
                elif strategy == "weighted":
                    score = result.confidence * lens_weight
                elif strategy == "best":
                    score = result.confidence
                else:
                    score = result.confidence * lens_weight
                
                if answer not in answer_scores:
                    answer_scores[answer] = 0
                    answer_sources[answer] = []
                
                answer_scores[answer] += score
                answer_sources[answer].append(lens.value)
                
                # Also add alternatives with reduced weight (only for primary answers)
                for alt, alt_score in result.alternatives[:2]:
                    alt_lower = alt.lower()
                    if alt_lower not in answer_scores:
                        answer_scores[alt_lower] = 0
                        answer_sources[alt_lower] = []
                    answer_scores[alt_lower] += alt_score * 0.2 * lens_weight
        
        if not answer_scores:
            return "(no answer)", 0.0, {"sources": []}
        
        if strategy == "consensus":
            # Only return if 2+ lenses agree
            for answer, sources in answer_sources.items():
                if len(sources) >= 2:
                    return answer, answer_scores[answer], {
                        "sources": sources,
                        "strategy": "consensus",
                        "agreement": len(sources)
                    }
            return "(no consensus)", 0.0, {"sources": []}
        
        elif strategy == "best":
            # Return highest single-lens confidence
            best_answer = None
            best_score = 0
            best_lens = None
            for lens, result in results.items():
                if result.confidence > best_score:
                    best_score = result.confidence
                    best_answer = result.answer
                    best_lens = lens.value
            return best_answer or "(none)", best_score, {
                "sources": [best_lens] if best_lens else [],
                "strategy": "best"
            }
        
        else:  # voting or weighted
            # Sort by score
            sorted_answers = sorted(answer_scores.items(), key=lambda x: -x[1])
            best_answer = sorted_answers[0][0]
            best_score = sorted_answers[0][1]
            
            return best_answer, best_score, {
                "sources": answer_sources.get(best_answer, []),
                "strategy": strategy,
                "all_scores": dict(sorted_answers[:5])
            }
    
    def compare_lenses(self, a: str, b: str, c: str) -> None:
        """Print comparison of analogy through all lenses."""
        results = self.analogy_all_lenses(a, b, c)
        
        print(f"\n{'='*70}")
        print(f"ANALOGY: {a}:{b} :: {c}:?")
        print(f"{'='*70}")
        
        for lens, result in results.items():
            print(f"\n{lens.value.upper()} LENS:")
            print(f"  Answer: {result.answer}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reasoning: {result.reasoning}")
            if result.alternatives:
                alts = ", ".join([f"{w}({s:.2f})" for w, s in result.alternatives[:3]])
                print(f"  Alternatives: {alts}")
        
        # Show combined results
        print(f"\n--- COMBINED RESULTS ---")
        for strategy in ["voting", "weighted", "consensus", "best", "intrinsic_priority"]:
            answer, score, meta = self.combine_lenses(a, b, c, strategy)
            sources = meta.get('sources', [])
            print(f"  {strategy.upper():18} → {answer} (score={score:.2f}, sources={sources})")


def demo():
    """Demonstrate lens-aware analogies."""
    print("=" * 70)
    print("LENS-AWARE ANALOGIES WITH COMBINATION")
    print("=" * 70)
    print("""
    Key insight: Analogies are the same structural relationship
    viewed through different LENSES.
    
    Combination strategies:
    - VOTING:    Each lens gets 1 vote, majority wins
    - WEIGHTED:  Weight by confidence, sum scores
    - CONSENSUS: Only return if 2+ lenses agree
    - BEST:      Return highest confidence single answer
    """)
    
    la = LensAnalogies()
    
    # Test cases
    test_cases = [
        ('holmes', 'detective', 'watson'),
        ('king', 'queen', 'man'),
        ('physics', 'science', 'biology'),
        ('einstein', 'physicist', 'darwin'),
        ('man', 'woman', 'king'),
        ('father', 'mother', 'son'),
    ]
    
    for a, b, c in test_cases:
        la.compare_lenses(a, b, c)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: COMBINED LENS RESULTS")
    print("=" * 70)
    
    print(f"\n{'Analogy':<35} {'Intrinsic Pri':<15} {'Weighted':<15} {'Best Lens':<15}")
    print("-" * 80)
    
    for a, b, c in test_cases:
        analogy_str = f"{a}:{b} :: {c}:?"
        
        intrinsic_pri, ip_score, _ = la.combine_lenses(a, b, c, "intrinsic_priority")
        weighted, w_score, _ = la.combine_lenses(a, b, c, "weighted")
        best, b_score, meta = la.combine_lenses(a, b, c, "best")
        best_lens = meta.get('sources', ['?'])[0] if meta.get('sources') else '?'
        
        print(f"{analogy_str:<35} {intrinsic_pri:<15} {weighted:<15} {best}({best_lens})")
    
    # Show which analogies the intrinsic lens solves perfectly
    print("\n" + "=" * 70)
    print("INTRINSIC LENS PERFECT MATCHES (confidence >= 0.9)")
    print("=" * 70)
    
    for a, b, c in test_cases:
        results = la.analogy_all_lenses(a, b, c)
        intrinsic = results.get(AnalogyLens.INTRINSIC)
        if intrinsic and intrinsic.confidence >= 0.9:
            print(f"  {a}:{b} :: {c}:? → {intrinsic.answer} (conf={intrinsic.confidence:.2f})")
            print(f"    Reasoning: {intrinsic.reasoning}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
