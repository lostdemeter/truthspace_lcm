#!/usr/bin/env python3
"""
Wave-Based Geometric Projection

The key insight: Words are WAVES with amplitude and phase.
- Amplitude = importance (frequency, TF-IDF-like)
- Phase = position in sentence (where the word appears)

Projection works by:
1. Encode truth as wave (complex vector with amplitude + phase)
2. Learn the TRANSFER FUNCTION from truth waves to signal waves
3. Apply transfer function to new truth waves
4. Decode back to text

The transfer function captures HOW the signal corpus transforms:
- Which words get amplified (constructive interference)
- Which words get suppressed (destructive interference)  
- How positions shift (phase transformation)

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import cmath
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class WaveProjector:
    """
    Projects truth to signal using wave interference.
    
    Each word is a wave: z = amplitude * e^(i*phase)
    - amplitude = sqrt(frequency) normalized
    - phase = 2π * (position / sentence_length)
    
    The transfer function H(ω) transforms truth waves to signal waves.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
        self.signal_frames = {}
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
        
        # Build vocabulary with frequency info
        self.vocab_freq = self._build_vocab_freq()
        
        # Learn transfer function
        self.transfer_function = self._learn_transfer_function()
    
    def _build_vocab_freq(self) -> Dict[str, float]:
        """Build vocabulary with frequency weights."""
        freq = Counter()
        
        for text in self.signal_frames.values():
            words = re.findall(r'\b\w+\b', text.lower())
            freq.update(words)
        
        # Normalize to [0, 1]
        max_freq = max(freq.values()) if freq else 1
        return {w: c / max_freq for w, c in freq.items()}
    
    def _encode_wave(self, text: str) -> Dict[str, complex]:
        """
        Encode text as wave vector.
        
        Each word gets a complex number:
        z = amplitude * e^(i * phase)
        
        amplitude = sqrt(global_freq) * local_count
        phase = 2π * (first_position / total_words)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {}
        
        total = len(words)
        wave = {}
        
        # Track first position of each word
        first_pos = {}
        word_counts = Counter()
        
        for i, w in enumerate(words):
            if w not in first_pos:
                first_pos[w] = i
            word_counts[w] += 1
        
        for w, count in word_counts.items():
            # Amplitude: based on frequency and local count
            global_freq = self.vocab_freq.get(w, 0.01)
            amplitude = math.sqrt(global_freq) * count
            
            # Phase: based on position (0 to 2π)
            phase = 2 * math.pi * (first_pos[w] / total)
            
            # Complex wave
            wave[w] = amplitude * cmath.exp(1j * phase)
        
        return wave
    
    def _decode_wave(self, wave: Dict[str, complex], reference_structure: List[str]) -> str:
        """
        Decode wave back to text.
        
        Use reference structure for word order, but filter/weight by wave amplitudes.
        """
        if not wave:
            return ""
        
        # Sort words by amplitude (importance)
        sorted_words = sorted(wave.items(), key=lambda x: abs(x[1]), reverse=True)
        important_words = {w for w, z in sorted_words[:30] if abs(z) > 0.01}
        
        # Reconstruct using reference structure
        result = []
        for w in reference_structure:
            w_lower = w.lower()
            if w_lower in important_words or w_lower not in wave:
                result.append(w)
        
        return ' '.join(result)
    
    def _learn_transfer_function(self) -> Dict[str, complex]:
        """
        Learn transfer function H(word) from truth→signal pairs.
        
        For each word, H(word) = avg(signal_wave[word] / truth_wave[word])
        
        This captures:
        - Amplitude change (amplification or suppression)
        - Phase shift (position change)
        """
        # Accumulate transfer values for each word
        transfer_sum = defaultdict(complex)
        transfer_count = defaultdict(int)
        
        for concept, signal_text in self.signal_frames.items():
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            truth_wave = self._encode_wave(truth_text)
            signal_wave = self._encode_wave(signal_text)
            
            # For words in both, compute transfer ratio
            for w in truth_wave:
                if w in signal_wave and abs(truth_wave[w]) > 0.001:
                    h = signal_wave[w] / truth_wave[w]
                    transfer_sum[w] += h
                    transfer_count[w] += 1
            
            # For words only in signal, they're "added" - high transfer
            for w in signal_wave:
                if w not in truth_wave:
                    transfer_sum[w] += signal_wave[w] * 2  # Boost
                    transfer_count[w] += 1
        
        # Average transfer function
        transfer = {}
        for w in transfer_sum:
            if transfer_count[w] > 0:
                transfer[w] = transfer_sum[w] / transfer_count[w]
        
        return transfer
    
    def project(self, concept: str) -> str:
        """
        Project truth through wave transfer function.
        """
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # If we have direct signal, return it
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Encode truth as wave
        truth_wave = self._encode_wave(truth)
        
        # Apply transfer function
        projected_wave = {}
        for w, z in truth_wave.items():
            if w in self.transfer_function:
                projected_wave[w] = z * self.transfer_function[w]
            else:
                # Unknown word - keep with reduced amplitude
                projected_wave[w] = z * 0.5
        
        # Add high-transfer words that might be missing
        for w, h in self.transfer_function.items():
            if w not in projected_wave and abs(h) > 1.5:
                # This word is often added in signal
                projected_wave[w] = h * 0.3
        
        # Decode back to text
        truth_words = re.findall(r'\b\w+\b', truth)
        return self._decode_wave(projected_wave, truth_words)


class SequenceProjector:
    """
    Sequence-aware projection that preserves word order.
    
    Instead of bag-of-words, we model the SEQUENCE:
    - Learn which positions in truth map to which positions in signal
    - Learn word substitutions at each position type
    
    Position types:
    - START: First few words (entity name, "It appears", etc.)
    - ROLE: The role/category word
    - ACTION: Verb phrases
    - TARGET: Object/target phrases
    - END: Concluding phrases
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load corpora
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        self.signal_frames = {}
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
        
        # Learn sequence transformations
        self.start_patterns = self._learn_start_patterns()
        self.role_transforms = self._learn_role_transforms()
        self.action_transforms = self._learn_action_transforms()
        self.connector_patterns = self._learn_connector_patterns()
    
    def _learn_start_patterns(self) -> List[Tuple[str, float]]:
        """Learn how sentences start in signal corpus."""
        starts = Counter()
        
        for text in self.signal_frames.values():
            # Get first 3-4 words
            words = text.split()[:4]
            start = ' '.join(words)
            starts[start] += 1
        
        total = sum(starts.values())
        return [(s, c/total) for s, c in starts.most_common(20)]
    
    def _learn_role_transforms(self) -> Dict[str, Counter]:
        """Learn how roles are expressed in signal."""
        transforms = defaultdict(Counter)
        
        for concept, signal_text in self.signal_frames.items():
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            # Extract role from truth
            truth_match = re.search(r'is a (\w+)', truth_text.lower())
            if not truth_match:
                continue
            truth_role = truth_match.group(1)
            
            # Extract role expression from signal
            signal_match = re.search(r'is a[n]? (\w+)', signal_text.lower())
            if signal_match:
                transforms[truth_role][signal_match.group(1)] += 1
            
            # Also check for "seems to be a"
            signal_match = re.search(r'seems to be a[n]? (\w+)', signal_text.lower())
            if signal_match:
                transforms[truth_role][f"seems to be a {signal_match.group(1)}"] += 1
        
        return dict(transforms)
    
    def _learn_action_transforms(self) -> Dict[str, str]:
        """Learn how actions are transformed."""
        # Common transformations
        return {
            'investigates': 'investigating',
            'studies': 'studying',
            'examines': 'examining',
            'explores': 'exploring',
            'analyzes': 'analyzing',
            'solves': 'solving',
            'deduces': 'deducing',
            'assists': 'assisting',
            'supports': 'supporting',
            'changes': 'changing',
            'develops': 'developing',
            'adapts': 'adapting',
        }
    
    def _learn_connector_patterns(self) -> Dict[str, str]:
        """Learn connector transformations."""
        return {
            'who': 'that',
            'This relates to': 'particularly involving',
            'often involving': 'particularly',
            ', and': ', and',
        }
    
    def project(self, concept: str) -> str:
        """Project using sequence-aware transformation."""
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # If we have direct signal, return it
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Parse and transform
        return self._transform_sequence(truth, concept)
    
    def _transform_sequence(self, truth: str, concept: str) -> str:
        """Transform truth sequence to signal-like sequence."""
        # Parse truth into segments
        segments = self._parse_segments(truth, concept)
        
        # Transform each segment
        result_parts = []
        
        # Entity + role
        entity = segments.get('entity', concept.title())
        role = segments.get('role', 'entity')
        
        # Transform role if we have learned transformation
        if role in self.role_transforms:
            candidates = self.role_transforms[role]
            if candidates:
                best_role = candidates.most_common(1)[0][0]
                if 'seems to be' in best_role:
                    result_parts.append(f"{entity} {best_role}")
                else:
                    result_parts.append(f"{entity} is a {best_role}")
            else:
                result_parts.append(f"{entity} is a {role}")
        else:
            result_parts.append(f"{entity} is a {role}")
        
        # Actions
        actions = segments.get('actions', [])
        if actions:
            # Transform to gerunds
            gerunds = []
            for a in actions:
                if a in self.action_transforms:
                    gerunds.append(self.action_transforms[a])
                elif a.endswith('s'):
                    gerunds.append(a[:-1] + 'ing')
                else:
                    gerunds.append(a + 'ing')
            
            if len(gerunds) == 1:
                action_str = gerunds[0]
            elif len(gerunds) == 2:
                action_str = f"{gerunds[0]} and {gerunds[1]}"
            else:
                action_str = f"{gerunds[0]}, {gerunds[1]}, and {gerunds[2]}"
            
            result_parts.append(f"that involves {action_str}")
        
        # Targets
        targets = segments.get('targets', [])
        if targets:
            target_str = ' and '.join(targets[:2])
            result_parts.append(f"particularly {target_str}")
        
        # Join with appropriate punctuation
        if len(result_parts) == 1:
            return result_parts[0] + "."
        elif len(result_parts) == 2:
            return f"{result_parts[0]} {result_parts[1]}."
        else:
            return f"{result_parts[0]} {result_parts[1]}, {result_parts[2]}."
    
    def _parse_segments(self, truth: str, concept: str) -> Dict:
        """Parse truth into semantic segments."""
        segments = {
            'entity': concept.title(),
            'role': 'entity',
            'actions': [],
            'targets': [],
        }
        
        truth_lower = truth.lower()
        
        # Extract role
        role_match = re.search(r'is a[n]? (\w+)', truth_lower)
        if role_match:
            segments['role'] = role_match.group(1)
        
        # Extract actions
        action_match = re.search(r'who (\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if action_match:
            segments['actions'] = [a for a in action_match.groups() if a]
        
        # Extract targets
        target_match = re.search(r'(?:involving|relates to)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if target_match:
            segments['targets'] = [t for t in target_match.groups() if t]
        
        return segments


def demo():
    """Demo the wave and sequence projectors."""
    print("=" * 70)
    print("WAVE & SEQUENCE PROJECTION")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    wave_proj = WaveProjector(truth_path, signal_path)
    seq_proj = SequenceProjector(truth_path, signal_path)
    
    # Find concepts NOT in signal
    test_concepts = []
    for concept in wave_proj.truth_qa.knowledge.concepts:
        if concept not in wave_proj.signal_frames:
            c = wave_proj.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 10:
            break
    
    print(f"\nTesting {len(test_concepts)} concepts NOT in signal corpus:\n")
    
    for concept in test_concepts[:6]:
        truth = wave_proj.truth_qa.ask(f"What is {concept}?")
        wave_result = wave_proj.project(concept)
        seq_result = seq_proj.project(concept)
        
        print(f"{concept.upper()}")
        print(f"  TRUTH:    {truth}")
        print(f"  WAVE:     {wave_result}")
        print(f"  SEQUENCE: {seq_result}")
        print()


if __name__ == "__main__":
    demo()
