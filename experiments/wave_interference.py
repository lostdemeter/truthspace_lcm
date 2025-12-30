#!/usr/bin/env python3
"""
Wave Interference Polisher: True Geometric Approach

Each word is encoded as a complex wave:
    w = A · e^(iθ)

Where:
    A (amplitude) = sqrt(frequency) - how important/common the word is
    θ (phase) = 2π · (position / sentence_length) - where it appears

Interference between Truth beam and Signal beam:
    combined = truth_wave + signal_wave
    
    |combined|² = |truth|² + |signal|² + 2·|truth|·|signal|·cos(θ_truth - θ_signal)

When phases align (same position): cos(0) = 1 → CONSTRUCTIVE
When phases oppose: cos(π) = -1 → DESTRUCTIVE

This is pure geometry - no morphology, no templates, just wave math.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import cmath
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class WordWave:
    """A word encoded as a complex wave."""
    word: str
    amplitude: float
    phase: float  # radians, 0 to 2π
    
    def to_complex(self) -> complex:
        """Convert to complex number."""
        return self.amplitude * cmath.exp(1j * self.phase)
    
    @staticmethod
    def from_complex(word: str, z: complex) -> 'WordWave':
        """Create from complex number."""
        return WordWave(
            word=word,
            amplitude=abs(z),
            phase=cmath.phase(z)
        )


class WaveEncoder:
    """Encodes sentences as collections of word waves."""
    
    def __init__(self):
        # Global word frequencies for amplitude normalization
        self.word_freqs = Counter()
        self.max_freq = 1
    
    def learn_frequencies(self, sentences: List[str]):
        """Learn word frequencies from a corpus of sentences."""
        for sentence in sentences:
            words = self._tokenize(sentence)
            self.word_freqs.update(words)
        
        if self.word_freqs:
            self.max_freq = max(self.word_freqs.values())
    
    def _tokenize(self, sentence: str) -> List[str]:
        """Simple tokenization."""
        import re
        return [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
    
    def encode_sentence(self, sentence: str) -> List[WordWave]:
        """
        Encode a sentence as a list of word waves.
        
        Each word gets:
        - amplitude = sqrt(frequency / max_freq) - normalized importance
        - phase = 2π * (position / length) - position encoding
        """
        words = self._tokenize(sentence)
        if not words:
            return []
        
        waves = []
        for i, word in enumerate(words):
            # Amplitude from frequency (use sqrt for compression)
            freq = self.word_freqs.get(word, 1)
            amplitude = math.sqrt(freq / self.max_freq) if self.max_freq > 0 else 0.5
            
            # Phase from position (0 at start, 2π at end)
            phase = 2 * math.pi * (i / len(words))
            
            waves.append(WordWave(word=word, amplitude=amplitude, phase=phase))
        
        return waves
    
    def encode_to_vector(self, sentence: str) -> Dict[str, complex]:
        """
        Encode sentence as a dictionary of word -> complex number.
        
        If a word appears multiple times, sum the waves.
        """
        waves = self.encode_sentence(sentence)
        
        word_vectors = defaultdict(complex)
        for wave in waves:
            word_vectors[wave.word] += wave.to_complex()
        
        return dict(word_vectors)


class WaveInterference:
    """
    Computes interference between two wave sources.
    
    Truth beam: raw geometric knowledge
    Signal beam: polished phrasing
    
    The interference pattern reveals which words/phrases
    are reinforced (constructive) or cancelled (destructive).
    """
    
    def __init__(self):
        self.encoder = WaveEncoder()
    
    def compute_interference(self, truth_vector: Dict[str, complex], 
                            signal_vector: Dict[str, complex]) -> Dict[str, Tuple[float, str]]:
        """
        Compute interference between truth and signal vectors.
        
        Returns: {word: (combined_amplitude, interference_type)}
        """
        all_words = set(truth_vector.keys()) | set(signal_vector.keys())
        
        results = {}
        for word in all_words:
            truth_wave = truth_vector.get(word, 0j)
            signal_wave = signal_vector.get(word, 0j)
            
            # Interference: just add the complex numbers!
            combined = truth_wave + signal_wave
            combined_amp = abs(combined)
            
            # Determine interference type
            truth_amp = abs(truth_wave)
            signal_amp = abs(signal_wave)
            
            if truth_amp > 0 and signal_amp > 0:
                # Both present - check if constructive or destructive
                # Constructive if combined > max(individual)
                # Destructive if combined < min(individual)
                max_individual = max(truth_amp, signal_amp)
                min_individual = min(truth_amp, signal_amp)
                
                if combined_amp > max_individual * 1.2:
                    itype = 'constructive'
                elif combined_amp < min_individual * 0.8:
                    itype = 'destructive'
                else:
                    itype = 'partial'
            elif truth_amp > 0:
                itype = 'truth_only'
            elif signal_amp > 0:
                itype = 'signal_only'
            else:
                itype = 'none'
            
            results[word] = (combined_amp, itype)
        
        return results
    
    def reconstruct_sentence(self, interference: Dict[str, Tuple[float, str]],
                            original_order: List[str]) -> str:
        """
        Reconstruct a sentence from interference pattern.
        
        Keep words with constructive/partial interference.
        Filter words with destructive interference.
        """
        output_words = []
        
        for word in original_order:
            word_lower = word.lower()
            if word_lower in interference:
                amp, itype = interference[word_lower]
                
                # Keep constructive and partial, filter destructive
                if itype in ('constructive', 'partial', 'signal_only'):
                    output_words.append(word)
                elif itype == 'truth_only' and amp > 0.3:
                    # Keep truth-only if strong enough
                    output_words.append(word)
                # else: filter out (destructive)
            else:
                output_words.append(word)
        
        return ' '.join(output_words)


class GeometricPolisher:
    """
    Polishes output using pure wave interference.
    
    No templates, no morphology - just geometry.
    
    KEY INSIGHT: We don't need a signal for EVERY concept.
    We learn the GLOBAL interference pattern from all truth/signal pairs,
    then apply that pattern to ANY truth output.
    
    The interference pattern tells us:
    - Which words are "structural" (high constructive interference globally)
    - Which words are "noise" (high destructive interference globally)
    - Which phrasings are preferred (signal-only words that should be added)
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.truth_path = truth_corpus_path
        self.signal_path = signal_corpus_path
        
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal frames
        self.signal_frames = {}  # concept -> polished text
        self.signal_texts = []   # all signal texts for global learning
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
                        self.signal_texts.append(text)
        
        # Build encoder with frequencies from both corpora
        self.encoder = WaveEncoder()
        self._learn_frequencies()
        
        # Interference calculator
        self.interference = WaveInterference()
        self.interference.encoder = self.encoder
        
        # LEARN GLOBAL INTERFERENCE PATTERN
        self.global_pattern = self._learn_global_interference()
    
    def _learn_frequencies(self):
        """Learn word frequencies from signal corpus only (fast)."""
        # Just use signal texts - no slow querying
        self.encoder.learn_frequencies(self.signal_texts)
    
    def _learn_global_interference(self) -> Dict[str, Tuple[float, float]]:
        """
        Learn global interference pattern from all truth/signal pairs.
        
        For each word, compute:
        - Average amplitude in truth beam
        - Average amplitude in signal beam
        - Average phase difference
        
        This tells us which words are "good" (appear in both, aligned)
        vs "bad" (appear in only one, or misaligned).
        """
        # Aggregate vectors across all pairs
        truth_totals = defaultdict(complex)
        signal_totals = defaultdict(complex)
        pair_count = 0
        
        for concept, signal_text in self.signal_frames.items():
            # Get truth for this concept
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            # Encode both
            truth_vec = self.encoder.encode_to_vector(truth_text)
            signal_vec = self.encoder.encode_to_vector(signal_text)
            
            # Accumulate
            for word, z in truth_vec.items():
                truth_totals[word] += z
            for word, z in signal_vec.items():
                signal_totals[word] += z
            
            pair_count += 1
        
        if pair_count == 0:
            return {}
        
        # Compute global pattern
        all_words = set(truth_totals.keys()) | set(signal_totals.keys())
        pattern = {}
        
        for word in all_words:
            truth_z = truth_totals.get(word, 0j) / pair_count
            signal_z = signal_totals.get(word, 0j) / pair_count
            
            truth_amp = abs(truth_z)
            signal_amp = abs(signal_z)
            
            # Phase difference
            if truth_amp > 0 and signal_amp > 0:
                phase_diff = abs(cmath.phase(truth_z) - cmath.phase(signal_z))
            else:
                phase_diff = math.pi  # Maximum difference if one is missing
            
            pattern[word] = (truth_amp, signal_amp, phase_diff)
        
        return pattern
    
    def polish(self, concept: str) -> Dict[str, str]:
        """
        Polish output for a concept using wave interference.
        
        Uses GLOBAL interference pattern to transform ANY truth output,
        even without a direct signal for this concept.
        """
        concept_lower = concept.lower()
        
        # Get truth beam (raw output)
        truth_text = self.truth_qa.ask(f"What is {concept}?")
        
        # Get signal beam if available (for comparison)
        signal_text = self.signal_frames.get(concept_lower, '')
        
        # Apply global interference pattern to truth
        interference_text = self._apply_global_interference(truth_text)
        
        # Also compute local pattern if we have signal
        local_pattern = {}
        if signal_text:
            truth_vector = self.encoder.encode_to_vector(truth_text)
            signal_vector = self.encoder.encode_to_vector(signal_text)
            local_pattern = self.interference.compute_interference(truth_vector, signal_vector)
        
        return {
            'truth': truth_text,
            'signal': signal_text if signal_text else '(no direct signal)',
            'interference': interference_text,
            'pattern': local_pattern,
        }
    
    def _apply_global_interference(self, truth_text: str) -> str:
        """
        Apply global interference pattern to transform truth text.
        
        KEY INSIGHT: 
        - STRUCTURE words (is, a, that, the) appear in MANY signals → high global amplitude
        - CONTENT words (physics, studies, holmes) appear in FEW signals → low global amplitude
        
        So we use interference to:
        1. KEEP structure words that have constructive interference
        2. KEEP content words (low global frequency = unique = important)
        3. FILTER noise words (medium frequency, destructive interference)
        """
        import re
        
        words = re.findall(r'\b\w+\b', truth_text)
        output_words = []
        
        # Identify structure words (high frequency in signal)
        structure_threshold = 0.15  # Words appearing in >15% of signals
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in self.global_pattern:
                truth_amp, signal_amp, phase_diff = self.global_pattern[word_lower]
                
                # Is this a structure word (high signal frequency)?
                is_structure = signal_amp > structure_threshold
                
                # Is this a content word (low signal frequency = unique)?
                is_content = signal_amp < 0.05 and len(word) > 3
                
                if is_content:
                    # Content words: ALWAYS keep (they carry meaning)
                    output_words.append(word)
                elif is_structure:
                    # Structure words: keep if constructive interference
                    alignment = math.cos(phase_diff) if phase_diff < math.pi else -1
                    if alignment > -0.5:  # Not strongly destructive
                        output_words.append(word)
                else:
                    # Middle ground: use interference score
                    if truth_amp > 0 and signal_amp > 0:
                        alignment = math.cos(phase_diff)
                        score = (truth_amp + signal_amp) * (0.5 + 0.5 * alignment)
                    elif signal_amp > 0:
                        score = signal_amp
                    else:
                        score = truth_amp * 0.5
                    
                    if score > 0.03:
                        output_words.append(word)
            else:
                # Unknown word - likely content, keep it
                output_words.append(word)
        
        return ' '.join(output_words)
    
    def _reconstruct_from_interference(self, signal_text: str, 
                                        signal_words: List[str],
                                        pattern: Dict[str, Tuple[float, str]]) -> str:
        """Reconstruct sentence keeping constructive interference words."""
        # For now, just return signal if we have good interference
        # Count constructive vs destructive
        constructive = sum(1 for _, (_, t) in pattern.items() if t == 'constructive')
        destructive = sum(1 for _, (_, t) in pattern.items() if t == 'destructive')
        
        if constructive > destructive:
            # Good interference - use signal
            return signal_text
        else:
            # Poor interference - blend truth and signal
            # Take structure from signal, content from truth
            return signal_text  # For now, just use signal
    
    def analyze_interference(self, concept: str) -> None:
        """Print detailed interference analysis."""
        result = self.polish(concept)
        
        print(f"\n{'='*60}")
        print(f"INTERFERENCE ANALYSIS: {concept.upper()}")
        print('='*60)
        print(f"\nTRUTH BEAM:  {result['truth']}")
        print(f"SIGNAL BEAM: {result['signal']}")
        print(f"INTERFERENCE: {result['interference']}")
        
        if result['pattern']:
            print(f"\nWORD INTERFERENCE PATTERN:")
            
            # Group by type
            by_type = defaultdict(list)
            for word, (amp, itype) in result['pattern'].items():
                by_type[itype].append((word, amp))
            
            for itype in ['constructive', 'partial', 'destructive', 'truth_only', 'signal_only']:
                if itype in by_type:
                    words = sorted(by_type[itype], key=lambda x: -x[1])[:5]
                    word_str = ', '.join(f"{w}({a:.2f})" for w, a in words)
                    print(f"  {itype:12}: {word_str}")


def demo():
    """Demo the wave interference polisher."""
    print("=" * 70)
    print("WAVE INTERFERENCE POLISHER")
    print("Pure geometric approach - no templates")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal.json"
    
    if not os.path.exists(signal_path):
        print("\nSignal corpus not found. Please run:")
        print("  python3 experiments/two_beam_polisher.py --create-signal --concepts 300")
        return
    
    polisher = GeometricPolisher(truth_path, signal_path)
    
    print(f"\nLoaded {len(polisher.signal_frames)} signal frames")
    print(f"Learned frequencies from {len(polisher.encoder.word_freqs)} unique words")
    
    # Analyze several concepts
    for concept in ['physics', 'holmes', 'evolution', 'biology', 'consciousness']:
        polisher.analyze_interference(concept)


if __name__ == "__main__":
    demo()
