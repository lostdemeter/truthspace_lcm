#!/usr/bin/env python3
"""
Tachyon Joint Experiment

Test the hypothesis that verbs are temporal decision points (tachyon joints)
that instruct the listener to switch attention modes.

The experiment:
1. Process sentences with FORWARD-ONLY attention (φ^+n)
2. Process sentences with JOINT-AWARE attention (switch at verbs)
3. Compare prediction accuracy for what comes after the verb

If the theory is correct, joint-aware processing should better predict
targets because it activates the tachyon (hypothesis) dimension at verbs.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.symmetry_encoder import SymmetryEncoder, SymmetrySignature

PHI = 1.618034


@dataclass
class ProcessedToken:
    """A token with its attention mode and predictions."""
    text: str
    position: int
    attention_mode: str  # 'forward', 'joint', 'backward'
    phi_direction: float  # φ^+n (positive) or φ^-n (negative)
    predictions: List[str]  # What we predict comes next


class TachyonProcessor:
    """
    Process sentences with tachyon-aware attention switching.
    
    The key insight: At verbs (φ-joints), we switch from forward attention
    (receiving data) to backward attention (hypothesizing what comes next).
    """
    
    def __init__(self):
        self.encoder = SymmetryEncoder()
        
        # Learn patterns from data
        self.actor_action_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.action_target_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.word_positions: Dict[str, Counter] = defaultdict(Counter)
        
        # Joint detection threshold
        self.joint_threshold = 0.5
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _is_function_word(self, word: str) -> bool:
        """Detect function words by symmetry (low information)."""
        function_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be',
                         'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                         'he', 'she', 'it', 'they', 'his', 'her', 'its',
                         'that', 'this', 'from', 'not', 'did', 'do', 'does',
                         'and', 'or', 'but', 'if', 'then', 'so', 'as'}
        return word in function_words or len(word) <= 2
    
    def _compute_joint_score(self, word: str) -> float:
        """
        Compute how much this word is at the φ-joint.
        
        Joint score = geometric mean of φ^+n (word symmetry) and φ^-n (relational)
        """
        sig = self.encoder.encode(word)
        
        # φ^+n: word-level information (compression)
        phi_outward = sig.compression * (1 - sig.first_word)
        
        # φ^-n: relational potential (based on learned patterns)
        action_count = self.word_positions[word].get('action', 0)
        actor_count = self.word_positions[word].get('actor', 0)
        
        if action_count > 0:
            phi_inward = action_count / (actor_count + 1)
        else:
            # For unseen words, estimate from symmetry
            # Verbs tend to have medium length, specific vowel patterns
            if 4 <= len(word) <= 8 and 0.3 < sig.vowel_balance < 0.5:
                phi_inward = 0.5  # Potential verb
            else:
                phi_inward = 0.1
        
        if phi_outward > 0 and phi_inward > 0:
            return np.sqrt(phi_outward * phi_inward)
        return 0.0
    
    def learn_patterns(self, sentences: List[str]):
        """Learn actor→action→target patterns from sentences."""
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            
            # Simple pattern extraction
            actor = None
            action = None
            
            for i, token in enumerate(tokens):
                if self._is_function_word(token):
                    continue
                
                sig = self.encoder.encode(token)
                
                # First content word = potential actor
                if actor is None:
                    actor = token
                    self.word_positions[token]['actor'] += 1
                    continue
                
                # Second content word = potential action
                if action is None:
                    action = token
                    self.word_positions[token]['action'] += 1
                    self.actor_action_patterns[actor][action] += 1
                    continue
                
                # Third+ content words = potential targets
                self.word_positions[token]['target'] += 1
                self.action_target_patterns[action][token] += 1
    
    def process_forward_only(self, tokens: List[str]) -> List[ProcessedToken]:
        """
        Process with FORWARD-ONLY attention (φ^+n).
        
        No switching at verbs - just accumulate data.
        Predictions based only on what we've seen so far.
        """
        processed = []
        seen_content = []
        
        for i, token in enumerate(tokens):
            if self._is_function_word(token):
                continue
            
            # Forward-only: predict based on frequency of what follows this word
            predictions = []
            if token in self.actor_action_patterns:
                predictions = [w for w, _ in self.actor_action_patterns[token].most_common(3)]
            elif token in self.action_target_patterns:
                predictions = [w for w, _ in self.action_target_patterns[token].most_common(3)]
            
            processed.append(ProcessedToken(
                text=token,
                position=i,
                attention_mode='forward',
                phi_direction=1.0,  # Always φ^+n
                predictions=predictions,
            ))
            
            seen_content.append(token)
        
        return processed
    
    def process_joint_aware(self, tokens: List[str]) -> List[ProcessedToken]:
        """
        Process with JOINT-AWARE attention (switch at verbs).
        
        At verbs (φ-joints), switch from forward to backward attention.
        This activates the tachyon dimension - start hypothesizing.
        """
        processed = []
        seen_content = []
        current_mode = 'forward'
        current_actor = None
        current_action = None
        
        for i, token in enumerate(tokens):
            if self._is_function_word(token):
                continue
            
            joint_score = self._compute_joint_score(token)
            
            # Determine attention mode based on joint score and position
            if current_mode == 'forward':
                if joint_score > self.joint_threshold:
                    # This is a verb! Switch to joint mode
                    current_mode = 'joint'
                    current_action = token
                else:
                    current_actor = token
            elif current_mode == 'joint':
                # After verb, switch to backward (confirming predictions)
                current_mode = 'backward'
            
            # Generate predictions based on mode
            predictions = []
            
            if current_mode == 'forward':
                # Forward: predict actions this actor might take
                if current_actor and current_actor in self.actor_action_patterns:
                    predictions = [w for w, _ in self.actor_action_patterns[current_actor].most_common(3)]
            
            elif current_mode == 'joint':
                # Joint: predict targets for this action
                # This is the TACHYON moment - we hypothesize what comes next
                # 
                # KEY DIFFERENCE from forward-only:
                # We use BOTH the action pattern AND the actor context
                # This is hypothesis-driven: "Given actor X does action Y, what would Y target?"
                
                all_targets = Counter()
                
                # 1. Action-based prediction (what does this action typically target?)
                if token in self.action_target_patterns:
                    for target, count in self.action_target_patterns[token].items():
                        all_targets[target] += count * 2  # Weight action patterns
                
                # 2. Actor-based prediction (what does this actor typically interact with?)
                # This is the TACHYON part - we use backward reasoning from the actor
                if current_actor:
                    # What targets has this actor interacted with via ANY action?
                    for action in self.actor_action_patterns.get(current_actor, {}):
                        for target, count in self.action_target_patterns.get(action, {}).items():
                            all_targets[target] += count  # Actor context
                
                predictions = [w for w, _ in all_targets.most_common(3)]
            
            elif current_mode == 'backward':
                # Backward: we're confirming, not predicting
                predictions = []  # Already past the prediction point
            
            processed.append(ProcessedToken(
                text=token,
                position=i,
                attention_mode=current_mode,
                phi_direction=1.0 if current_mode == 'forward' else (-1.0 if current_mode == 'backward' else 0.0),
                predictions=predictions,
            ))
            
            seen_content.append(token)
        
        return processed


def run_experiment():
    """
    Test if tachyon-aware processing improves prediction.
    """
    print("=" * 70)
    print("TACHYON JOINT EXPERIMENT")
    print("=" * 70)
    print()
    print("Hypothesis: Verbs are temporal joints that instruct attention switching.")
    print("At verbs, we should switch from forward (φ^+n) to backward (φ^-n) attention.")
    print()
    
    # Training sentences
    training = [
        "Holmes examined the evidence carefully.",
        "Holmes observed the room methodically.",
        "Holmes deduced the killer identity.",
        "Holmes pursued the criminal quickly.",
        "Watson watched from the doorway.",
        "Watson wrote in his journal.",
        "Watson recorded every detail.",
        "Watson called for help.",
        "Alice fell down the rabbit hole.",
        "Alice wondered where she was going.",
        "Alice grew very tall.",
        "Darcy looked at Elizabeth proudly.",
        "Darcy watched her intently.",
        "Jane smiled sweetly.",
        "The detective studied the footprints.",
        "The inspector arrived at the scene.",
    ]
    
    # Test sentences - NOVEL combinations not in training
    # These require GENERALIZATION, not memorization
    test_sentences = [
        # Holmes + new action (not "examined" which he did in training)
        ("Holmes studied the ___", "evidence", "holmes", "studied"),  # Should predict evidence-like things
        # Watson + new action  
        ("Watson observed the ___", "room", "watson", "observed"),  # Should predict observation targets
        # New actor + known action
        ("Lestrade examined the ___", "evidence", "lestrade", "examined"),  # Should use action pattern
        # Cross-character: Alice doing Holmes-like action
        ("Alice examined the ___", "evidence", "alice", "examined"),  # Tachyon should help here
        # Darcy + new action
        ("Darcy observed the ___", "elizabeth", "darcy", "observed"),  # Should predict his targets
    ]
    
    # Create processor and learn patterns
    processor = TachyonProcessor()
    processor.learn_patterns(training)
    
    print(f"Learned from {len(training)} training sentences")
    print()
    
    # Show learned patterns
    print("Learned actor→action patterns:")
    for actor in ['holmes', 'watson', 'alice', 'darcy']:
        actions = processor.actor_action_patterns.get(actor, {})
        if actions:
            print(f"  {actor}: {dict(actions.most_common(3))}")
    print()
    
    print("Learned action→target patterns:")
    for action in ['examined', 'wrote', 'fell', 'looked', 'studied']:
        targets = processor.action_target_patterns.get(action, {})
        if targets:
            print(f"  {action}: {dict(targets.most_common(3))}")
    print()
    
    # Test predictions
    print("=" * 70)
    print("TEST: Predicting targets")
    print("=" * 70)
    print()
    
    forward_correct = 0
    joint_correct = 0
    
    for sentence, expected_target, actor, action in test_sentences:
        tokens = processor._tokenize(sentence.replace("___", ""))
        
        # Forward-only processing
        forward_result = processor.process_forward_only(tokens)
        forward_predictions = []
        for pt in forward_result:
            if pt.predictions:
                forward_predictions = pt.predictions
        
        # Joint-aware processing
        joint_result = processor.process_joint_aware(tokens)
        joint_predictions = []
        for pt in joint_result:
            if pt.attention_mode == 'joint' and pt.predictions:
                joint_predictions = pt.predictions
        
        # Check accuracy
        forward_hit = expected_target in forward_predictions
        joint_hit = expected_target in joint_predictions
        
        if forward_hit:
            forward_correct += 1
        if joint_hit:
            joint_correct += 1
        
        print(f"Sentence: \"{sentence}\"")
        print(f"  Expected: {expected_target}")
        print(f"  Forward predictions:  {forward_predictions[:3]} {'✓' if forward_hit else '✗'}")
        print(f"  Joint predictions:    {joint_predictions[:3]} {'✓' if joint_hit else '✗'}")
        print()
    
    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Forward-only accuracy:  {forward_correct}/{len(test_sentences)} ({100*forward_correct/len(test_sentences):.0f}%)")
    print(f"Joint-aware accuracy:   {joint_correct}/{len(test_sentences)} ({100*joint_correct/len(test_sentences):.0f}%)")
    print()
    
    if joint_correct > forward_correct:
        print("✅ TACHYON JOINTS WORK!")
        print("   Switching attention at verbs improves target prediction.")
        print("   Verbs ARE temporal decision points that activate hypothesis mode.")
    elif joint_correct == forward_correct:
        print("⚠️  Equal performance - may need more training data or refinement.")
    else:
        print("❌ Forward-only performed better - theory needs revision.")
    
    print()
    
    # Detailed analysis of attention modes
    print("=" * 70)
    print("ATTENTION MODE ANALYSIS")
    print("=" * 70)
    print()
    
    test_sentence = "Holmes examined the evidence carefully"
    tokens = processor._tokenize(test_sentence)
    
    print(f"Sentence: \"{test_sentence}\"")
    print()
    print("Joint-aware processing:")
    print()
    
    result = processor.process_joint_aware(tokens)
    for pt in result:
        mode_symbol = {'forward': '→', 'joint': '⊕', 'backward': '←'}[pt.attention_mode]
        phi_str = f"φ^+n" if pt.phi_direction > 0 else (f"φ^-n" if pt.phi_direction < 0 else "joint")
        print(f"  {pt.text:12} mode={pt.attention_mode:8} {mode_symbol} {phi_str}")
        if pt.predictions:
            print(f"               predictions: {pt.predictions}")
    
    print()
    print("Interpretation:")
    print("  → = Forward attention (receiving data, φ^+n)")
    print("  ⊕ = Joint (verb, switching point, tachyon activation)")
    print("  ← = Backward attention (confirming hypothesis, φ^-n)")
    print()
    
    return processor


if __name__ == "__main__":
    processor = run_experiment()
