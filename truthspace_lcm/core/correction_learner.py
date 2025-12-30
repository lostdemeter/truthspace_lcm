#!/usr/bin/env python3
"""
Correction Learner: Train on Output Corrections

This module enables the GeometricLCM to learn from corrected outputs.
When the model produces "Sherlock Holmes is a teacher" but we want
"Sherlock Holmes is a consulting detective", we can apply that correction
directly to the geometric structure.

Key insight: Corrections are frame injections with high confidence.
We don't need gradients - we adjust relationship weights directly.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truthspace_lcm.core.geometric import HolographicGeometricQA


@dataclass
class Correction:
    """A single correction record."""
    question: str
    wrong_answer: str
    correct_answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False
    
    # Extracted frames
    wrong_frame: Dict = field(default_factory=dict)
    correct_frame: Dict = field(default_factory=dict)


@dataclass
class CorrectionDelta:
    """Changes to apply from a correction."""
    concept: str
    weaken_actions: List[str] = field(default_factory=list)
    strengthen_actions: List[str] = field(default_factory=list)
    weaken_targets: List[str] = field(default_factory=list)
    strengthen_targets: List[str] = field(default_factory=list)
    phi_adjustment: float = 0.0


class CorrectionLearner:
    """
    Learn from output corrections by adjusting geometric relationships.
    
    Usage:
        learner = CorrectionLearner()
        learner.load_corpus('truthspace_lcm/corpus_self_improved.json')
        
        # Apply a correction
        learner.correct(
            question="Who is Holmes?",
            wrong="Holmes is a teacher",
            correct="Holmes is a consulting detective"
        )
        
        # Save updated corpus
        learner.save_corpus('truthspace_lcm/corpus_self_improved.json')
    """
    
    # Correction weight multiplier (higher = corrections stick more)
    CORRECTION_WEIGHT = 2.0
    
    # Words to skip when extracting frames
    SKIP_WORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                  'who', 'what', 'where', 'when', 'why', 'how', 'which',
                  'very', 'really', 'quite', 'just', 'also', 'and', 'or'}
    
    def __init__(self):
        self.corpus_data: Dict = {}
        self.corrections: List[Correction] = []
        self.correction_log_path = Path('truthspace_lcm/corrections.json')
        self.qa: Optional[HolographicGeometricQA] = None
        self.corpus_path: Optional[str] = None
    
    def load_corpus(self, path: str):
        """Load the frame-based corpus."""
        self.corpus_path = path
        with open(path) as f:
            self.corpus_data = json.load(f)
        
        # Also load the QA system for testing
        self.qa = HolographicGeometricQA()
        self.qa.load_corpus(path)
        print(f"Loaded corpus: {len(self.corpus_data.get('frames', []))} frames")
    
    def save_corpus(self, path: str):
        """Save the updated corpus."""
        with open(path, 'w') as f:
            json.dump(self.corpus_data, f, indent=2)
        print(f"Saved corpus to {path}")
    
    def load_corrections(self):
        """Load previous corrections from log."""
        if self.correction_log_path.exists():
            with open(self.correction_log_path) as f:
                data = json.load(f)
                self.corrections = [Correction(**c) for c in data]
            print(f"Loaded {len(self.corrections)} previous corrections")
    
    def save_corrections(self):
        """Save corrections to log."""
        with open(self.correction_log_path, 'w') as f:
            json.dump([asdict(c) for c in self.corrections], f, indent=2)
    
    def extract_frame(self, sentence: str) -> Dict[str, str]:
        """
        Extract a simple frame from a sentence.
        
        Returns: {initiator, mediator, receiver}
        """
        # Clean and tokenize
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        words = [w for w in sentence.split() if w not in self.SKIP_WORDS]
        
        if len(words) < 2:
            return {'initiator': '', 'mediator': '', 'receiver': ''}
        
        # Simple heuristic: first content word = initiator, last = receiver
        # Middle words are potential mediators (look for verbs)
        initiator = words[0]
        receiver = words[-1] if len(words) > 1 else ''
        
        # Find mediator (verb-like word in middle)
        mediator = ''
        verb_endings = ('ing', 'ed', 's', 'es')
        for word in words[1:-1] if len(words) > 2 else words[1:]:
            if word.endswith(verb_endings) or word in ('consult', 'detect', 'teach', 'work', 'live'):
                mediator = word
                break
        
        # If no verb found, use 'be' as default mediator
        if not mediator:
            mediator = 'be'
        
        return {
            'initiator': initiator,
            'mediator': mediator,
            'receiver': receiver,
        }
    
    def compute_delta(self, wrong_frame: Dict, correct_frame: Dict) -> CorrectionDelta:
        """
        Compute what needs to change between wrong and correct frames.
        """
        delta = CorrectionDelta(concept=correct_frame.get('initiator', ''))
        
        # Same subject - different predicate/object
        if wrong_frame.get('initiator') == correct_frame.get('initiator'):
            # Weaken wrong relationships
            if wrong_frame.get('mediator'):
                delta.weaken_actions.append(wrong_frame['mediator'])
            if wrong_frame.get('receiver'):
                delta.weaken_targets.append(wrong_frame['receiver'])
            
            # Strengthen correct relationships
            if correct_frame.get('mediator'):
                delta.strengthen_actions.append(correct_frame['mediator'])
            if correct_frame.get('receiver'):
                delta.strengthen_targets.append(correct_frame['receiver'])
        
        return delta
    
    def apply_delta(self, delta: CorrectionDelta):
        """
        Apply a correction delta to the corpus.
        
        This adds correction frames that will influence the concept's
        geometric properties when re-processed.
        """
        frames = self.corpus_data.get('frames', [])
        
        # Add strengthening frames (with high weight via repetition)
        for action in delta.strengthen_actions:
            for _ in range(int(self.CORRECTION_WEIGHT)):
                frames.append({
                    'initiator': delta.concept,
                    'mediator': action,
                    'receiver': delta.strengthen_targets[0] if delta.strengthen_targets else action,
                    'source': 'Correction',
                    'text': f"[CORRECTION] {delta.concept} {action} {delta.strengthen_targets[0] if delta.strengthen_targets else ''}"
                })
        
        # For weakening, we could add negative frames or just rely on
        # the strengthening to outweigh. For now, we add counter-frames.
        # In a more sophisticated system, we'd track negative evidence.
        
        self.corpus_data['frames'] = frames
        print(f"Applied delta: +{len(delta.strengthen_actions)} actions, +{len(delta.strengthen_targets)} targets for '{delta.concept}'")
    
    def correct(self, question: str, wrong: str, correct: str) -> CorrectionDelta:
        """
        Apply a correction: the model said 'wrong' but should have said 'correct'.
        
        Args:
            question: The question that was asked
            wrong: The wrong answer the model produced
            correct: The correct answer we want
        
        Returns:
            The CorrectionDelta that was applied
        """
        # Extract frames
        wrong_frame = self.extract_frame(wrong)
        correct_frame = self.extract_frame(correct)
        
        print(f"\nCorrection:")
        print(f"  Question: {question}")
        print(f"  Wrong:    {wrong}")
        print(f"  Correct:  {correct}")
        print(f"  Wrong frame:   {wrong_frame}")
        print(f"  Correct frame: {correct_frame}")
        
        # Compute and apply delta
        delta = self.compute_delta(wrong_frame, correct_frame)
        self.apply_delta(delta)
        
        # Log correction
        correction = Correction(
            question=question,
            wrong_answer=wrong,
            correct_answer=correct,
            wrong_frame=wrong_frame,
            correct_frame=correct_frame,
            applied=True,
        )
        self.corrections.append(correction)
        
        return delta
    
    def batch_correct(self, corrections: List[Tuple[str, str, str]]):
        """
        Apply multiple corrections.
        
        Args:
            corrections: List of (question, wrong_answer, correct_answer) tuples
        """
        print(f"\nApplying {len(corrections)} corrections...")
        
        for question, wrong, correct in corrections:
            self.correct(question, wrong, correct)
        
        print(f"\nBatch complete. {len(corrections)} corrections applied.")
    
    def get_correction_stats(self) -> Dict:
        """Get statistics about applied corrections."""
        if not self.corrections:
            return {'total': 0}
        
        concepts_corrected = Counter()
        for c in self.corrections:
            if c.correct_frame:
                concepts_corrected[c.correct_frame.get('initiator', 'unknown')] += 1
        
        return {
            'total': len(self.corrections),
            'concepts_corrected': dict(concepts_corrected.most_common(10)),
            'latest': self.corrections[-1].timestamp if self.corrections else None,
        }
    
    def test_answer(self, question: str) -> str:
        """Test what the model currently answers for a question."""
        if not self.qa:
            return "[QA not loaded]"
        return self.qa.ask(question)
    
    def reload_qa(self):
        """Reload QA system after corpus changes."""
        if self.corpus_path:
            self.qa = HolographicGeometricQA()
            self.qa.load_corpus(self.corpus_path)
    
    def correct_and_verify(self, question: str, wrong: str, correct: str, 
                           save: bool = False) -> Dict:
        """
        Apply a correction and verify improvement.
        
        Returns dict with before/after answers and whether it improved.
        """
        # Get answer before correction
        before = self.test_answer(question)
        
        # Apply correction
        delta = self.correct(question, wrong, correct)
        
        # Save and reload to see effect
        if save and self.corpus_path:
            self.save_corpus(self.corpus_path)
            self.reload_qa()
            after = self.test_answer(question)
        else:
            after = "[save=False, not reloaded]"
        
        # Check if correct answer terms appear more in 'after'
        correct_terms = set(correct.lower().split()) - self.SKIP_WORDS
        before_matches = sum(1 for t in correct_terms if t in before.lower())
        after_matches = sum(1 for t in correct_terms if t in after.lower()) if save else -1
        
        return {
            'question': question,
            'before': before,
            'after': after,
            'target': correct,
            'before_matches': before_matches,
            'after_matches': after_matches,
            'improved': after_matches > before_matches if save else None,
        }


def demo():
    """Demonstrate correction learning."""
    learner = CorrectionLearner()
    learner.load_corpus('truthspace_lcm/corpus_self_improved.json')
    
    print("\n" + "=" * 60)
    print("CORRECTION LEARNING DEMO")
    print("=" * 60)
    
    # First, show current answers
    test_questions = [
        "Who is Holmes?",
        "What does Watson do?",
        "What is physics?",
    ]
    
    print("\n--- Current Answers (Before Correction) ---")
    for q in test_questions:
        answer = learner.test_answer(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:100]}..." if len(answer) > 100 else f"  A: {answer}")
        print()
    
    # Example corrections
    corrections = [
        ("Who is Holmes?", 
         "Holmes is a teacher", 
         "Holmes is a consulting detective"),
        
        ("What does Watson do?",
         "Watson is a cook",
         "Watson is a doctor and companion"),
        
        ("Where does Holmes live?",
         "Holmes lives in Paris",
         "Holmes lives at Baker Street London"),
    ]
    
    learner.batch_correct(corrections)
    
    # Show stats
    stats = learner.get_correction_stats()
    print(f"\nCorrection Stats:")
    print(f"  Total corrections: {stats['total']}")
    print(f"  Concepts corrected: {stats['concepts_corrected']}")
    
    # Save (commented out to not modify corpus in demo)
    # learner.save_corpus('truthspace_lcm/corpus_self_improved.json')
    # learner.save_corrections()
    
    print("\n" + "=" * 60)
    print("Corrections applied to in-memory corpus.")
    print("Call save_corpus() to persist changes.")
    print("=" * 60)


if __name__ == '__main__':
    demo()
