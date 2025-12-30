#!/usr/bin/env python3
"""
Automated Gear Chain Refinement with Qwen2

Uses Qwen2 via Ollama to automatically:
1. Evaluate projection quality
2. Suggest corrections
3. Propagate corrections back through gear chain
4. Update the knowledge corpus

This creates a self-improving loop:
  Project → Evaluate → Correct → Update → Project (improved)

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gear_chain_feedback import FeedbackGearChain, Correction
from experiments.ollama_corpus_refiner import OllamaClient


class AutoRefiner:
    """
    Automatically refines gear chain output using Qwen2.
    
    Process:
    1. Project concept through gear chain
    2. Ask Qwen2 to evaluate and correct the output
    3. Parse Qwen2's correction
    4. Propagate correction back through gears
    5. Optionally save to corpus
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        self.chain = FeedbackGearChain(truth_corpus_path, signal_corpus_path)
        self.ollama = OllamaClient()
        
        if not self.ollama.is_available():
            print("WARNING: Ollama not available. Auto-refinement disabled.")
        
        # Track refinement stats
        self.stats = {
            'evaluated': 0,
            'corrected': 0,
            'unchanged': 0,
            'errors': 0,
        }
    
    def evaluate_output(self, concept: str, output: str) -> Tuple[float, str]:
        """
        Ask Qwen2 to evaluate the output quality.
        
        Returns (score, feedback) where score is 0-10.
        """
        prompt = f"""Evaluate this sentence describing "{concept}":

"{output}"

Rate it 0-10 on:
- Grammatical correctness
- Natural phrasing
- Appropriate role/category for the concept

Respond with ONLY:
SCORE: [number]
FEEDBACK: [one sentence explanation]"""

        response = self.ollama.generate(prompt, temperature=0.1)
        
        if not response:
            return 5.0, "No response from evaluator"
        
        # Parse score
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response)
        score = float(score_match.group(1)) if score_match else 5.0
        
        # Parse feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else response
        
        return min(10, max(0, score)), feedback
    
    def suggest_correction(self, concept: str, output: str, feedback: str) -> Optional[str]:
        """
        Ask Qwen2 to suggest a corrected version.
        
        Returns corrected output or None if no correction needed.
        """
        prompt = f"""The following sentence describes "{concept}":

"{output}"

Feedback: {feedback}

If this sentence needs improvement, provide a corrected version.
Keep the same structure but fix any issues with:
- Role/category (e.g., "entity" should be "concept" for abstract things)
- Grammar
- Natural phrasing

Respond with ONLY the corrected sentence, or "NO CHANGE" if it's fine.
Do not add explanations."""

        response = self.ollama.generate(prompt, temperature=0.2)
        
        if not response:
            return None
        
        response = response.strip().strip('"')
        
        if 'NO CHANGE' in response.upper():
            return None
        
        # Validate response looks like a sentence
        if len(response) < 10 or not response[0].isupper():
            return None
        
        return response
    
    def refine_concept(self, concept: str, auto_save: bool = False) -> Dict[str, Any]:
        """
        Refine a single concept through the full pipeline.
        
        Returns dict with refinement results.
        """
        result = {
            'concept': concept,
            'original_output': None,
            'score': None,
            'feedback': None,
            'corrected_output': None,
            'corrections': [],
            'saved': False,
        }
        
        # Project
        output = self.chain.project(concept)
        result['original_output'] = output
        
        # Skip if direct signal match
        if concept.lower() in self.chain.signal_frames:
            result['score'] = 10.0
            result['feedback'] = "Direct signal match"
            self.stats['unchanged'] += 1
            return result
        
        self.stats['evaluated'] += 1
        
        # Evaluate
        score, feedback = self.evaluate_output(concept, output)
        result['score'] = score
        result['feedback'] = feedback
        
        # If score is high enough, no correction needed
        if score >= 8.0:
            self.stats['unchanged'] += 1
            return result
        
        # Get correction suggestion
        corrected = self.suggest_correction(concept, output, feedback)
        
        if not corrected or corrected == output:
            self.stats['unchanged'] += 1
            return result
        
        result['corrected_output'] = corrected
        
        # Apply correction through gear chain
        corrections = self.chain.correct(corrected)
        result['corrections'] = [
            {'field': c.field, 'old': c.old_value, 'new': c.new_value}
            for c in corrections
        ]
        
        if corrections:
            self.stats['corrected'] += 1
            
            if auto_save:
                self.chain.apply_corrections(save=True)
                result['saved'] = True
            else:
                # Clear pending without saving
                self.chain.pending_corrections = []
        
        return result
    
    def refine_batch(self, concepts: List[str], auto_save: bool = False, 
                     delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        Refine a batch of concepts.
        
        Args:
            concepts: List of concepts to refine
            auto_save: Whether to save corrections to corpus
            delay: Delay between API calls (seconds)
        
        Returns list of refinement results.
        """
        results = []
        
        for i, concept in enumerate(concepts):
            print(f"[{i+1}/{len(concepts)}] Refining: {concept}")
            
            try:
                result = self.refine_concept(concept, auto_save=auto_save)
                results.append(result)
                
                # Print summary
                if result['corrected_output']:
                    print(f"  Score: {result['score']:.1f} → Corrected")
                    for c in result['corrections']:
                        print(f"    {c['field']}: {c['old']} → {c['new']}")
                else:
                    print(f"  Score: {result['score']:.1f} → OK")
                
            except Exception as e:
                print(f"  Error: {e}")
                self.stats['errors'] += 1
                results.append({
                    'concept': concept,
                    'error': str(e)
                })
            
            if delay > 0 and i < len(concepts) - 1:
                time.sleep(delay)
        
        return results
    
    def print_stats(self):
        """Print refinement statistics."""
        print("\n" + "=" * 50)
        print("REFINEMENT STATISTICS")
        print("=" * 50)
        print(f"Evaluated:  {self.stats['evaluated']}")
        print(f"Corrected:  {self.stats['corrected']}")
        print(f"Unchanged:  {self.stats['unchanged']}")
        print(f"Errors:     {self.stats['errors']}")
        
        if self.stats['evaluated'] > 0:
            correction_rate = self.stats['corrected'] / self.stats['evaluated'] * 100
            print(f"Correction rate: {correction_rate:.1f}%")


def demo():
    """Demo the auto-refiner."""
    print("=" * 70)
    print("AUTOMATED GEAR CHAIN REFINEMENT")
    print("Using Qwen2 to evaluate and correct projections")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    refiner = AutoRefiner(truth_path, signal_path)
    
    if not refiner.ollama.is_available():
        print("Ollama not available. Exiting.")
        return
    
    # Find test concepts
    test_concepts = []
    for concept in refiner.chain.truth_qa.knowledge.concepts:
        if concept not in refiner.chain.signal_frames:
            c = refiner.chain.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 10:
            break
    
    print(f"\nRefining {len(test_concepts)} concepts...")
    print()
    
    # Refine batch (don't auto-save for demo)
    results = refiner.refine_batch(test_concepts, auto_save=False, delay=0.3)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for r in results:
        if 'error' in r:
            continue
        
        print(f"\n{r['concept'].upper()}")
        print(f"  Original:  {r['original_output']}")
        print(f"  Score:     {r['score']:.1f}")
        print(f"  Feedback:  {r['feedback']}")
        
        if r['corrected_output']:
            print(f"  Corrected: {r['corrected_output']}")
            for c in r['corrections']:
                print(f"    → {c['field']}: {c['old']} → {c['new']}")
    
    refiner.print_stats()


def refine_all(save: bool = False, limit: int = 100):
    """
    Refine all concepts not in signal corpus.
    
    Args:
        save: Whether to save corrections to corpus
        limit: Maximum number of concepts to process
    """
    print("=" * 70)
    print("FULL CORPUS REFINEMENT")
    print(f"Save: {save}, Limit: {limit}")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    refiner = AutoRefiner(truth_path, signal_path)
    
    if not refiner.ollama.is_available():
        print("Ollama not available. Exiting.")
        return
    
    # Find all concepts not in signal
    concepts = []
    for concept in refiner.chain.truth_qa.knowledge.concepts:
        if concept not in refiner.chain.signal_frames:
            c = refiner.chain.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions:
                concepts.append(concept)
        if len(concepts) >= limit:
            break
    
    print(f"Found {len(concepts)} concepts to refine")
    
    # Refine
    results = refiner.refine_batch(concepts, auto_save=save, delay=0.3)
    
    # Save results
    output_file = "logs/refinement_results.json"
    os.makedirs("logs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    refiner.print_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-refine gear chain projections")
    parser.add_argument("--full", action="store_true", help="Refine all concepts")
    parser.add_argument("--save", action="store_true", help="Save corrections to corpus")
    parser.add_argument("--limit", type=int, default=100, help="Max concepts to process")
    
    args = parser.parse_args()
    
    if args.full:
        refine_all(save=args.save, limit=args.limit)
    else:
        demo()
