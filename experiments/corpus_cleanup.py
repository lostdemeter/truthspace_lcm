#!/usr/bin/env python3
"""
Corpus Cleanup Tool: Using Qwen2 to Fix Bad Frames

This tool identifies and fixes problematic frames in the corpus:
1. Frames with nouns used as verbs ("ecologies", "observables")
2. Frames with broken grammar
3. Frames with nonsensical content

Uses Qwen2 to:
- Validate if a frame is grammatically correct
- Suggest fixes for broken frames
- Identify frames that should be removed

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ollama_corpus_refiner import OllamaClient
from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class FrameAnalysis:
    """Analysis of a single frame."""
    text: str
    is_valid: bool
    issue: Optional[str] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.0


@dataclass 
class ConceptAnalysis:
    """Analysis of a concept's quality."""
    name: str
    total_frames: int
    bad_actions: List[str]
    bad_targets: List[str]
    suggested_actions: List[str]
    suggested_targets: List[str]


class CorpusAnalyzer:
    """Analyzes corpus for quality issues."""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.qa = GeometricQA()
        self.qa.load_corpus(corpus_path)
        self.knowledge = self.qa.knowledge
        
        # Known bad patterns
        self.noun_verbs = {
            'ecologies', 'ecology', 'systems', 'system', 'observables',
            'categorize', 'analyze', 'safeguard', 'safeguards', 'across',
            'acknowledges', 'cuneiform', 'extraordinary', 'various',
            'nuance', 'nuances', 'complexity', 'indispensable',
            'adaptability', 'essential', 'precision', 'systematic',
            'observable', 'necessary', 'understand', 'categorizes',
            'mores', 'philosophies', 'ideological', 'utilization',
            'curation', 'enhancement', 'relevance', 'diverse',
        }
        
        # Known good verbs
        self.good_verbs = {
            'studies', 'study', 'examines', 'examine', 'investigates', 'investigate',
            'explores', 'explore', 'analyzes', 'analyze', 'describes', 'describe',
            'explains', 'explain', 'understands', 'understand', 'discovers', 'discover',
            'observes', 'observe', 'measures', 'measure', 'tests', 'test',
            'solves', 'solve', 'deduces', 'deduce', 'reasons', 'reason',
            'assists', 'assist', 'helps', 'help', 'supports', 'support',
            'documents', 'document', 'records', 'record', 'provides', 'provide',
            'includes', 'include', 'involves', 'involve', 'creates', 'create',
            'develops', 'develop', 'produces', 'produce', 'generates', 'generate',
            'processes', 'process', 'transforms', 'transform', 'converts', 'convert',
        }
    
    def analyze_concept(self, name: str) -> ConceptAnalysis:
        """Analyze a single concept for quality issues."""
        if name not in self.knowledge.concepts:
            return None
        
        concept = self.knowledge.concepts[name]
        
        # Find bad actions
        bad_actions = []
        good_actions = []
        if concept.actions:
            for action, count in concept.actions.most_common(10):
                if action.lower() in self.noun_verbs:
                    bad_actions.append(action)
                elif action.lower() in self.good_verbs:
                    good_actions.append(action)
        
        # Find bad targets
        bad_targets = []
        good_targets = []
        if concept.targets:
            for target, count in concept.targets.most_common(10):
                # Targets that are verbs or noise
                if target.lower() in self.good_verbs or target.lower() in self.noun_verbs:
                    bad_targets.append(target)
                elif len(target) > 2 and target[0].islower():
                    good_targets.append(target)
        
        return ConceptAnalysis(
            name=name,
            total_frames=concept.initiator_count + concept.mediator_count + concept.receiver_count,
            bad_actions=bad_actions,
            bad_targets=bad_targets,
            suggested_actions=good_actions[:3],
            suggested_targets=good_targets[:3],
        )
    
    def find_problematic_concepts(self, min_bad_actions: int = 1) -> List[ConceptAnalysis]:
        """Find all concepts with quality issues."""
        problems = []
        
        for name in self.knowledge.concepts:
            analysis = self.analyze_concept(name)
            if analysis and (len(analysis.bad_actions) >= min_bad_actions or len(analysis.bad_targets) >= 2):
                problems.append(analysis)
        
        # Sort by number of issues
        problems.sort(key=lambda x: len(x.bad_actions) + len(x.bad_targets), reverse=True)
        return problems


class CorpusCleaner:
    """
    Cleans the corpus using Qwen2 to validate and fix frames.
    """
    
    def __init__(self, corpus_path: str, model: str = "qwen2:latest"):
        self.corpus_path = corpus_path
        self.ollama = OllamaClient(model=model)
        self.analyzer = CorpusAnalyzer(corpus_path)
        
        # Load raw corpus for modification
        with open(corpus_path, 'r') as f:
            self.corpus_data = json.load(f)
        
        self.stats = {
            'frames_analyzed': 0,
            'frames_fixed': 0,
            'frames_removed': 0,
            'concepts_cleaned': 0,
        }
    
    def validate_frame(self, frame_text: str) -> FrameAnalysis:
        """Use Qwen2 to validate a frame."""
        prompt = f"""Is this sentence grammatically correct and meaningful? Answer only "yes" or "no", then briefly explain why.

Sentence: "{frame_text}"

Answer:"""
        
        response = self.ollama.generate(prompt, temperature=0.1)
        
        if response:
            is_valid = response.lower().startswith('yes')
            return FrameAnalysis(
                text=frame_text,
                is_valid=is_valid,
                issue=response if not is_valid else None,
                confidence=0.9 if is_valid else 0.7,
            )
        
        return FrameAnalysis(text=frame_text, is_valid=True, confidence=0.5)
    
    def fix_frame(self, frame_text: str) -> Optional[str]:
        """Use Qwen2 to fix a broken frame."""
        prompt = f"""Fix this sentence to be grammatically correct and meaningful. Only output the fixed sentence, nothing else.

Original: "{frame_text}"

Fixed:"""
        
        response = self.ollama.generate(prompt, temperature=0.3)
        
        if response and len(response) > 5:
            # Clean up the response
            fixed = response.strip().strip('"').strip("'")
            # Make sure it's actually different and not too long
            if fixed != frame_text and len(fixed) < 200:
                return fixed
        
        return None
    
    def clean_concept_actions(self, concept_name: str) -> Dict:
        """Clean bad actions from a concept using Qwen2."""
        analysis = self.analyzer.analyze_concept(concept_name)
        if not analysis:
            return {'status': 'not_found'}
        
        if not analysis.bad_actions:
            return {'status': 'clean', 'concept': concept_name}
        
        # Ask Qwen2 what verbs this concept should use
        prompt = f"""What are the 3 most appropriate verbs to describe what "{concept_name}" does? 
Only list the verbs in base form, separated by commas. No explanation.

Example for "physics": studies, investigates, explains
Example for "detective": investigates, solves, deduces

Verbs for "{concept_name}":"""
        
        response = self.ollama.generate(prompt, temperature=0.3)
        
        suggested_verbs = []
        if response:
            # Parse the response
            verbs = re.findall(r'\b\w+\b', response.lower())
            for v in verbs:
                if v in self.analyzer.good_verbs or len(v) > 4:
                    suggested_verbs.append(v)
        
        return {
            'status': 'needs_cleaning',
            'concept': concept_name,
            'bad_actions': analysis.bad_actions,
            'suggested_verbs': suggested_verbs[:3],
            'current_good_actions': analysis.suggested_actions,
        }
    
    def apply_concept_fix(self, concept_name: str, good_verbs: List[str], strength: int = 20) -> int:
        """Apply fixes by adding good frames and marking bad ones for removal."""
        frames_added = 0
        
        # Add reinforcement frames with good verbs
        for verb in good_verbs:
            frame_text = f"{concept_name.title()} {verb}."
            # Add to corpus data
            self.corpus_data['frames'].append({
                'text': frame_text,
                'source': 'cleanup',
                'agent': concept_name,
            })
            frames_added += strength
            
            # Add multiple times for reinforcement
            for _ in range(strength - 1):
                self.corpus_data['frames'].append({
                    'text': frame_text,
                    'source': 'cleanup',
                    'agent': concept_name,
                })
        
        self.stats['frames_fixed'] += frames_added
        self.stats['concepts_cleaned'] += 1
        
        return frames_added
    
    def remove_bad_frames(self, patterns: List[str]) -> int:
        """Remove frames matching bad patterns."""
        original_count = len(self.corpus_data['frames'])
        
        # Filter out bad frames
        good_frames = []
        for frame in self.corpus_data['frames']:
            text = frame.get('text', '').lower()
            is_bad = False
            
            for pattern in patterns:
                if pattern.lower() in text:
                    is_bad = True
                    break
            
            if not is_bad:
                good_frames.append(frame)
        
        removed = original_count - len(good_frames)
        self.corpus_data['frames'] = good_frames
        self.stats['frames_removed'] += removed
        
        return removed
    
    def clean_corpus(self, concepts: List[str] = None, auto_fix: bool = True) -> Dict:
        """
        Clean the corpus by fixing problematic concepts.
        
        Args:
            concepts: List of concepts to clean (None = find automatically)
            auto_fix: Whether to automatically apply fixes
        """
        results = {
            'analyzed': [],
            'fixed': [],
            'skipped': [],
        }
        
        # Find problematic concepts if not specified
        if concepts is None:
            problems = self.analyzer.find_problematic_concepts()
            concepts = [p.name for p in problems[:20]]  # Top 20 worst
        
        print(f"Cleaning {len(concepts)} concepts...")
        
        for concept in concepts:
            print(f"\nAnalyzing: {concept}")
            
            # Get cleaning suggestions
            suggestion = self.clean_concept_actions(concept)
            results['analyzed'].append(suggestion)
            
            if suggestion['status'] == 'needs_cleaning':
                print(f"  Bad actions: {suggestion['bad_actions']}")
                print(f"  Suggested verbs: {suggestion['suggested_verbs']}")
                
                if auto_fix and suggestion['suggested_verbs']:
                    # Apply the fix
                    frames_added = self.apply_concept_fix(
                        concept, 
                        suggestion['suggested_verbs'],
                        strength=20
                    )
                    print(f"  ✓ Added {frames_added} reinforcement frames")
                    results['fixed'].append(concept)
                else:
                    results['skipped'].append(concept)
            else:
                print(f"  ✓ Already clean")
        
        return results
    
    def save_corpus(self, path: str = None):
        """Save the cleaned corpus."""
        path = path or self.corpus_path
        
        with open(path, 'w') as f:
            json.dump(self.corpus_data, f, indent=2)
        
        print(f"\nSaved cleaned corpus to {path}")
        print(f"  Total frames: {len(self.corpus_data['frames'])}")
    
    def print_stats(self):
        """Print cleanup statistics."""
        print("\n" + "=" * 60)
        print("CLEANUP STATISTICS")
        print("=" * 60)
        print(f"Concepts cleaned: {self.stats['concepts_cleaned']}")
        print(f"Frames fixed/added: {self.stats['frames_fixed']}")
        print(f"Frames removed: {self.stats['frames_removed']}")
        print("=" * 60)


def demo():
    """Demonstrate the corpus cleanup tool."""
    print("=" * 70)
    print("CORPUS CLEANUP TOOL")
    print("Using Qwen2 to fix bad frames")
    print("=" * 70)
    print()
    
    # Check Ollama
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama is not running!")
        return
    
    print("✓ Ollama is available")
    print()
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    
    # Analyze first
    print("ANALYZING CORPUS FOR PROBLEMS:")
    print("-" * 60)
    
    analyzer = CorpusAnalyzer(corpus_path)
    problems = analyzer.find_problematic_concepts()
    
    print(f"Found {len(problems)} concepts with issues:")
    for p in problems[:10]:
        print(f"  {p.name}: bad_actions={p.bad_actions[:3]}, bad_targets={p.bad_targets[:2]}")
    
    print()
    print("CLEANING TOP PROBLEMATIC CONCEPTS:")
    print("-" * 60)
    
    # Clean the worst offenders
    cleaner = CorpusCleaner(corpus_path)
    
    # Target specific broken concepts
    target_concepts = [
        'evolution', 'consciousness', 'matter', 'energy',
        'mathematics', 'biology', 'chemistry', 'watson',
    ]
    
    results = cleaner.clean_corpus(concepts=target_concepts, auto_fix=True)
    
    cleaner.print_stats()
    
    # Save
    print("\nSaving cleaned corpus...")
    cleaner.save_corpus()
    
    print("\nDone!")


def full_cleanup():
    """Run a full corpus cleanup."""
    print("=" * 70)
    print("FULL CORPUS CLEANUP")
    print("=" * 70)
    
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama is not running!")
        return
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    cleaner = CorpusCleaner(corpus_path)
    
    # Find and clean all problematic concepts
    results = cleaner.clean_corpus(concepts=None, auto_fix=True)
    
    cleaner.print_stats()
    cleaner.save_corpus()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Corpus Cleanup Tool")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--full", action="store_true", help="Run full cleanup")
    
    args = parser.parse_args()
    
    if args.full:
        full_cleanup()
    else:
        demo()
