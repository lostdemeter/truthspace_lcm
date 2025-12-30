#!/usr/bin/env python3
"""
Interference Polisher: Two-Source Geometric Output Polishing

Uses constructive/destructive interference between two corpora:
1. TRUTH CORPUS - Raw geometric knowledge (what we know)
2. SIGNAL CORPUS - Qwen2-polished versions (how to say it naturally)

The interference pattern:
- Words that appear in BOTH with similar context → CONSTRUCTIVE (keep)
- Words that appear in only ONE → DESTRUCTIVE (filter)
- Phrasing patterns that align → REINFORCED (emerge as templates)

This allows us to:
1. Train the signal corpus ONCE with Qwen2
2. Use pure geometric interference at inference time
3. No LLM needed for polishing after training!

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import math
import cmath
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA, GeometricKnowledge


@dataclass
class WordVector:
    """A word encoded as amplitude and phase."""
    word: str
    amplitude: float  # How strongly it appears (frequency)
    phase: float      # Context/position encoding (0 to 2π)
    
    def to_complex(self) -> complex:
        """Convert to complex number for interference."""
        return self.amplitude * cmath.exp(1j * self.phase)


@dataclass
class InterferenceResult:
    """Result of interference between two sources."""
    word: str
    truth_amplitude: float
    signal_amplitude: float
    combined_amplitude: float
    interference_type: str  # 'constructive', 'destructive', 'partial'


class InterferencePolisher:
    """
    Polishes output using interference between truth and signal corpora.
    
    The Two-Beam Model:
    - TRUTH BEAM: Raw geometric knowledge (content - WHAT to say)
    - SIGNAL BEAM: Qwen2-polished phrasing (style - HOW to say it)
    
    Interference mechanics:
    - Each word is encoded as amplitude (frequency) + phase (position/context)
    - Words appearing in similar contexts in BOTH beams → constructive
    - Words appearing in different contexts → destructive
    - The interference pattern reveals natural phrasing templates
    
    Key insight: We're not just filtering words, we're extracting
    PHRASE PATTERNS that appear in both beams.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str = None):
        self.truth_path = truth_corpus_path
        self.signal_path = signal_corpus_path or truth_corpus_path.replace('.json', '_signal.json')
        
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_knowledge = self.truth_qa.knowledge
        
        # Load signal corpus (polished versions)
        self.signal_frames = []
        if os.path.exists(self.signal_path):
            with open(self.signal_path, 'r') as f:
                data = json.load(f)
                self.signal_frames = data.get('frames', [])
        
        # Build phrase patterns from signal corpus
        self.phrase_patterns = self._extract_phrase_patterns()
        
        # Build word co-occurrence maps for interference
        self.truth_cooccur = self._build_cooccurrence(self.truth_knowledge)
        self.signal_cooccur = self._build_signal_cooccurrence()
        
        # Compute interference
        self.interference_patterns = self._compute_interference()
    
    def _build_frequency_map(self, knowledge: GeometricKnowledge) -> Dict[str, Counter]:
        """Build word frequency map from corpus."""
        if not knowledge:
            return {}
        
        # Count word frequencies per concept
        concept_words = defaultdict(Counter)
        
        for name, concept in knowledge.concepts.items():
            # Count actions
            if concept.actions:
                for action, count in concept.actions.items():
                    concept_words[name][action] += count
            
            # Count targets
            if concept.targets:
                for target, count in concept.targets.items():
                    concept_words[name][target] += count
        
        return dict(concept_words)
    
    def _compute_interference(self) -> Dict[str, InterferenceResult]:
        """Compute interference patterns between truth and signal."""
        patterns = {}
        
        if not self.signal_freqs:
            return patterns
        
        # Get all words from both corpora
        all_words = set()
        for concept_words in self.truth_freqs.values():
            all_words.update(concept_words.keys())
        for concept_words in self.signal_freqs.values():
            all_words.update(concept_words.keys())
        
        # Compute interference for each word
        for word in all_words:
            # Sum frequencies across all concepts
            truth_total = sum(cw.get(word, 0) for cw in self.truth_freqs.values())
            signal_total = sum(cw.get(word, 0) for cw in self.signal_freqs.values())
            
            # Normalize to amplitudes (0-1)
            max_truth = max(sum(cw.values()) for cw in self.truth_freqs.values()) if self.truth_freqs else 1
            max_signal = max(sum(cw.values()) for cw in self.signal_freqs.values()) if self.signal_freqs else 1
            
            truth_amp = truth_total / max_truth if max_truth > 0 else 0
            signal_amp = signal_total / max_signal if max_signal > 0 else 0
            
            # Compute interference
            # If both have similar amplitude → constructive
            # If one is much stronger → partial
            # If only one has it → destructive (for the missing one)
            
            if truth_amp > 0 and signal_amp > 0:
                # Both present - compute combined amplitude
                # Use cosine similarity of frequencies as phase alignment
                ratio = min(truth_amp, signal_amp) / max(truth_amp, signal_amp)
                combined = (truth_amp + signal_amp) * ratio  # Constructive if aligned
                
                if ratio > 0.7:
                    itype = 'constructive'
                elif ratio > 0.3:
                    itype = 'partial'
                else:
                    itype = 'destructive'
            elif truth_amp > 0:
                combined = truth_amp * 0.5  # Partial - only in truth
                itype = 'truth_only'
            elif signal_amp > 0:
                combined = signal_amp * 0.5  # Partial - only in signal
                itype = 'signal_only'
            else:
                combined = 0
                itype = 'none'
            
            patterns[word] = InterferenceResult(
                word=word,
                truth_amplitude=truth_amp,
                signal_amplitude=signal_amp,
                combined_amplitude=combined,
                interference_type=itype,
            )
        
        return patterns
    
    def get_constructive_words(self, threshold: float = 0.5) -> List[str]:
        """Get words with constructive interference (appear strongly in both)."""
        return [
            p.word for p in self.interference_patterns.values()
            if p.interference_type == 'constructive' and p.combined_amplitude > threshold
        ]
    
    def get_destructive_words(self) -> List[str]:
        """Get words with destructive interference (should be filtered)."""
        return [
            p.word for p in self.interference_patterns.values()
            if p.interference_type in ('truth_only', 'signal_only')
        ]
    
    def polish_sentence(self, sentence: str, concept: str = None) -> str:
        """
        Polish a sentence using interference patterns.
        
        Words with constructive interference are kept.
        Words with destructive interference are filtered or replaced.
        """
        words = sentence.split()
        polished = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            
            if word_lower in self.interference_patterns:
                pattern = self.interference_patterns[word_lower]
                
                if pattern.interference_type == 'constructive':
                    # Keep - strong in both
                    polished.append(word)
                elif pattern.interference_type == 'partial':
                    # Keep but maybe modify
                    polished.append(word)
                elif pattern.interference_type == 'truth_only':
                    # Only in truth - might be noise, keep but flag
                    polished.append(word)
                elif pattern.interference_type == 'signal_only':
                    # Only in signal - good phrasing word, definitely keep
                    polished.append(word)
                else:
                    polished.append(word)
            else:
                # Unknown word - keep as is
                polished.append(word)
        
        return ' '.join(polished)
    
    def generate_polished(self, concept: str) -> str:
        """
        Generate polished output for a concept using interference.
        
        This combines truth (what we know) with signal (how to say it).
        """
        if concept not in self.truth_knowledge.concepts:
            return f"I don't have information about {concept}."
        
        truth_concept = self.truth_knowledge.concepts[concept]
        
        # Get truth components
        role = self._get_role(truth_concept)
        actions = self._get_constructive_actions(concept)
        targets = self._get_constructive_targets(concept)
        
        # Build sentence using interference-selected words
        if actions:
            action_str = ', '.join(actions[:2]) + (' and ' + actions[2] if len(actions) > 2 else '')
            if targets:
                sentence = f"{concept.title()} is a {role} that {action_str} {targets[0]}."
            else:
                sentence = f"{concept.title()} is a {role} that {action_str}."
        else:
            sentence = f"{concept.title()} is a {role}."
        
        return sentence
    
    def _get_role(self, concept) -> str:
        """Get role from concept."""
        category_words = {'detective', 'doctor', 'scientist', 'science', 'field', 
                         'discipline', 'study', 'process', 'phenomenon', 'concept'}
        if concept.targets:
            for target, count in concept.targets.most_common(10):
                if target in category_words and count >= 2:
                    return target
        return "concept"
    
    def _get_constructive_actions(self, concept: str) -> List[str]:
        """Get actions that have constructive interference."""
        if concept not in self.truth_freqs:
            return []
        
        actions = []
        truth_words = self.truth_freqs[concept]
        
        # Good verbs to look for
        good_verbs = {
            'studies', 'examines', 'investigates', 'explores', 'analyzes',
            'describes', 'explains', 'discovers', 'observes', 'measures',
            'solves', 'deduces', 'helps', 'supports', 'provides',
            'creates', 'develops', 'transforms', 'changes', 'adapts',
        }
        
        for word, count in truth_words.most_common(20):
            if word in good_verbs:
                # Check if constructive
                if word in self.interference_patterns:
                    pattern = self.interference_patterns[word]
                    if pattern.interference_type in ('constructive', 'partial', 'signal_only'):
                        actions.append(word)
                else:
                    actions.append(word)
        
        return actions[:3]
    
    def _get_constructive_targets(self, concept: str) -> List[str]:
        """Get targets that have constructive interference."""
        if concept not in self.truth_freqs:
            return []
        
        targets = []
        truth_words = self.truth_freqs[concept]
        
        for word, count in truth_words.most_common(20):
            if len(word) > 3 and word in self.truth_knowledge.concepts:
                # Check if constructive
                if word in self.interference_patterns:
                    pattern = self.interference_patterns[word]
                    if pattern.interference_type in ('constructive', 'partial'):
                        targets.append(word)
                else:
                    targets.append(word)
        
        return targets[:3]


def create_signal_corpus(truth_path: str, output_path: str, num_concepts: int = 100):
    """
    Create a signal corpus by polishing truth corpus with Qwen2.
    
    This is done ONCE to create the signal source.
    After this, no LLM is needed for polishing.
    """
    from experiments.ollama_corpus_refiner import OllamaClient
    
    print("Creating signal corpus using Qwen2...")
    print(f"Truth corpus: {truth_path}")
    print(f"Output: {output_path}")
    
    # Load truth corpus
    qa = GeometricQA()
    qa.load_corpus(truth_path)
    qa.set_output_lens('natural')
    
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama not available!")
        return
    
    # Get concepts to polish
    concepts = []
    for name, concept in qa.knowledge.concepts.items():
        if concept.is_content_word and concept.actions:
            concepts.append(name)
    
    concepts = concepts[:num_concepts]
    print(f"Polishing {len(concepts)} concepts...")
    
    # Generate polished frames
    signal_frames = []
    
    for i, concept in enumerate(concepts):
        # Get raw answer
        raw = qa.ask(f"What is {concept}?")
        
        if "don't know" in raw.lower():
            continue
        
        # Polish with Qwen2
        prompt = f"""Rewrite this to be more natural and grammatically correct. Only output the rewritten sentence:

"{raw}"

Rewritten:"""
        
        polished = ollama.generate(prompt, temperature=0.3)
        
        if polished and len(polished) > 10:
            # Add polished version as frame
            signal_frames.append({
                'text': polished.strip(),
                'source': 'signal',
                'agent': concept,
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Polished {i + 1}/{len(concepts)} concepts")
    
    # Save signal corpus
    with open(output_path, 'w') as f:
        json.dump({'frames': signal_frames}, f, indent=2)
    
    print(f"Saved signal corpus with {len(signal_frames)} frames")


def demo():
    """Demonstrate interference polisher."""
    print("=" * 70)
    print("INTERFERENCE POLISHER DEMO")
    print("=" * 70)
    print()
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal.json"
    
    # Check if signal corpus exists
    if not os.path.exists(signal_path):
        print("Signal corpus not found. Creating it...")
        create_signal_corpus(truth_path, signal_path, num_concepts=50)
        print()
    
    # Create polisher
    polisher = InterferencePolisher(truth_path, signal_path)
    
    print("Constructive words (appear strongly in both):")
    constructive = polisher.get_constructive_words(threshold=0.3)
    print(f"  {constructive[:20]}")
    print()
    
    print("Testing polished output:")
    print("-" * 60)
    
    for concept in ['physics', 'evolution', 'consciousness', 'holmes']:
        output = polisher.generate_polished(concept)
        print(f"{concept}: {output}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interference Polisher")
    parser.add_argument("--create-signal", action="store_true", help="Create signal corpus")
    parser.add_argument("--concepts", type=int, default=100, help="Number of concepts for signal")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    
    args = parser.parse_args()
    
    if args.create_signal:
        create_signal_corpus(
            "truthspace_lcm/corpus_experimental.json",
            "truthspace_lcm/corpus_signal.json",
            num_concepts=args.concepts
        )
    elif args.demo:
        demo()
    else:
        demo()
