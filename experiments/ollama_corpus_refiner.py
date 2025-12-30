#!/usr/bin/env python3
"""
Ollama Corpus Refiner: Using LLMs to Improve Geometric Knowledge

This experiment uses Ollama (with qwen2:latest) to refine our corpus by:
1. Generating prattle output from our geometric knowledge
2. Asking Qwen2 to rewrite it more clearly
3. Parsing the rewritten answer
4. Using geometric reinforcement learning to update the corpus

The key insight: We can use a larger LLM as a "teacher" to improve our
geometric model's outputs, then distill those improvements back into
the corpus structure.

This is a form of knowledge distillation where:
- Teacher: Qwen2 (7B parameters, trained on massive data)
- Student: Our geometric corpus (structured knowledge)
- Signal: Rewritten answers that are clearer and more natural

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.karplus_strong_output import KarplusStrongQA
from experiments.geometric_reinforcement import GeometricRL, Correction, CorpusModification


@dataclass
class RewriteResult:
    """Result of an Ollama rewrite."""
    original: str
    rewritten: str
    entity: str
    query: str
    success: bool
    error: Optional[str] = None


class OllamaClient:
    """Simple client for Ollama API."""
    
    def __init__(self, model: str = "qwen2:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Generate a response from Ollama."""
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Ollama error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class CorpusRefiner:
    """
    Refines the corpus using Ollama rewrites.
    
    The workflow:
    1. Pick a concept from the corpus
    2. Generate a prattle about it
    3. Ask Qwen2 to rewrite it more clearly
    4. Parse the rewritten answer for structure
    5. Apply geometric reinforcement learning
    6. Repeat
    """
    
    def __init__(self, corpus_path: str, model: str = "qwen2:latest"):
        self.corpus_path = corpus_path
        
        # Initialize components
        self.ollama = OllamaClient(model=model)
        self.ks_qa = KarplusStrongQA(corpus_path)
        self.grl = GeometricRL(corpus_path)
        
        # Track refinements
        self.refinements = []
        self.stats = {
            'total_rewrites': 0,
            'successful_rewrites': 0,
            'frames_added': 0,
            'concepts_refined': set(),
        }
    
    def rewrite_prompt(self, query: str, answer: str) -> str:
        """Create a prompt for Qwen2 to rewrite the answer."""
        return f"""Given the question "{query}" and the answer "{answer}", rewrite the answer to make it as clear and natural as possible while keeping the same meaning. Only reply with the rewritten answer, nothing else."""
    
    def rewrite(self, query: str, answer: str, entity: str) -> RewriteResult:
        """Ask Ollama to rewrite an answer."""
        prompt = self.rewrite_prompt(query, answer)
        
        rewritten = self.ollama.generate(prompt)
        
        if rewritten:
            self.stats['total_rewrites'] += 1
            self.stats['successful_rewrites'] += 1
            return RewriteResult(
                original=answer,
                rewritten=rewritten,
                entity=entity,
                query=query,
                success=True,
            )
        else:
            self.stats['total_rewrites'] += 1
            return RewriteResult(
                original=answer,
                rewritten="",
                entity=entity,
                query=query,
                success=False,
                error="Ollama failed to generate response",
            )
    
    def refine_concept(self, entity: str, query_template: str = "What is {entity}?") -> Dict:
        """
        Refine a single concept.
        
        Args:
            entity: The concept to refine
            query_template: Template for the query (use {entity} placeholder)
        
        Returns:
            Results of the refinement
        """
        query = query_template.format(entity=entity)
        
        # 1. Generate prattle output
        prattle = self.ks_qa.prattle(query, sentences=3)
        
        if "don't have information" in prattle.lower():
            return {'status': 'skipped', 'reason': 'unknown entity'}
        
        # 2. Get Qwen2 rewrite
        result = self.rewrite(query, prattle, entity)
        
        if not result.success:
            return {'status': 'failed', 'reason': result.error}
        
        # 3. Apply geometric reinforcement
        grl_results = self.grl.correct(query, result.rewritten)
        
        # 4. Track
        self.refinements.append({
            'entity': entity,
            'query': query,
            'original': result.original,
            'rewritten': result.rewritten,
            'modifications': grl_results,
        })
        
        self.stats['frames_added'] += grl_results.get('frames_added', 0)
        self.stats['concepts_refined'].add(entity)
        
        return {
            'status': 'success',
            'entity': entity,
            'original': result.original,
            'rewritten': result.rewritten,
            'frames_added': grl_results.get('frames_added', 0),
        }
    
    def refine_batch(self, entities: List[str], query_templates: List[str] = None) -> List[Dict]:
        """Refine a batch of concepts."""
        if query_templates is None:
            query_templates = [
                "What is {entity}?",
                "What does {entity} do?",
                "Tell me about {entity}",
            ]
        
        results = []
        
        for entity in entities:
            # Use different query templates for variety
            for template in query_templates:
                result = self.refine_concept(entity, template)
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✓ Refined {entity} ({template.format(entity='...')})")
                    print(f"    Original: {result['original'][:60]}...")
                    print(f"    Rewritten: {result['rewritten'][:60]}...")
                    print(f"    Frames added: {result['frames_added']}")
                elif result['status'] == 'skipped':
                    print(f"  - Skipped {entity}: {result['reason']}")
                else:
                    print(f"  ✗ Failed {entity}: {result['reason']}")
        
        return results
    
    def get_refinable_concepts(self, min_count: int = 5, max_count: int = 100) -> List[str]:
        """Get concepts that are good candidates for refinement."""
        concepts = []
        
        for name, concept in self.ks_qa.knowledge.concepts.items():
            # Skip noise
            if len(name) < 3:
                continue
            if name[0].isdigit():
                continue
            if not concept.is_content_word:
                continue
            
            # Must have some actions (we have something to say about it)
            if not concept.actions:
                continue
            
            # Must have reasonable attestation
            total_count = concept.initiator_count + concept.mediator_count + concept.receiver_count
            if total_count < min_count or total_count > max_count:
                continue
            
            concepts.append(name)
        
        return concepts
    
    def save_corpus(self, path: str = None):
        """Save the refined corpus."""
        self.grl.save_corpus(path)
    
    def print_stats(self):
        """Print refinement statistics."""
        print("\n" + "=" * 60)
        print("REFINEMENT STATISTICS")
        print("=" * 60)
        print(f"Total rewrites attempted: {self.stats['total_rewrites']}")
        print(f"Successful rewrites: {self.stats['successful_rewrites']}")
        print(f"Frames added: {self.stats['frames_added']}")
        print(f"Concepts refined: {len(self.stats['concepts_refined'])}")
        print("=" * 60)


def demo():
    """Demonstrate the Ollama corpus refiner."""
    print("=" * 70)
    print("OLLAMA CORPUS REFINER")
    print("Using Qwen2 to improve geometric knowledge")
    print("=" * 70)
    print()
    
    # Check Ollama availability
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        return
    
    print("✓ Ollama is available")
    print()
    
    # Initialize refiner
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    refiner = CorpusRefiner(corpus_path)
    
    # Test with a few concepts
    test_concepts = ['physics', 'holmes', 'watson']
    
    print("BEFORE REFINEMENT:")
    print("-" * 60)
    for entity in test_concepts:
        query = f"What is {entity}?"
        answer = refiner.ks_qa.prattle(query, sentences=2)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print()
    
    print("REFINING WITH QWEN2:")
    print("-" * 60)
    results = refiner.refine_batch(test_concepts, query_templates=["What is {entity}?"])
    
    print("\nAFTER REFINEMENT:")
    print("-" * 60)
    for entity in test_concepts:
        query = f"What is {entity}?"
        answer = refiner.grl.generate(query)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print()
    
    refiner.print_stats()
    
    # Ask about saving
    print("\nWould you like to save the refined corpus? (y/n)")


def continuous_refinement(cycles: int = 10, concepts_per_cycle: int = 5):
    """Run continuous refinement cycles."""
    print("=" * 70)
    print("CONTINUOUS CORPUS REFINEMENT")
    print(f"Running {cycles} cycles with {concepts_per_cycle} concepts each")
    print("=" * 70)
    print()
    
    # Check Ollama
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama is not running!")
        return
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    refiner = CorpusRefiner(corpus_path)
    
    # Get refinable concepts
    all_concepts = refiner.get_refinable_concepts()
    print(f"Found {len(all_concepts)} refinable concepts")
    print()
    
    import random
    
    for cycle in range(cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}/{cycles}")
        print(f"{'='*60}")
        
        # Pick random concepts
        if len(all_concepts) < concepts_per_cycle:
            concepts = all_concepts
        else:
            concepts = random.sample(all_concepts, concepts_per_cycle)
        
        print(f"Refining: {', '.join(concepts)}")
        print()
        
        # Refine
        refiner.refine_batch(concepts)
        
        # Remove refined concepts from pool
        for c in concepts:
            if c in all_concepts:
                all_concepts.remove(c)
    
    refiner.print_stats()
    
    # Save
    print("\nSaving refined corpus...")
    refiner.save_corpus()
    print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Corpus Refiner")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--continuous", action="store_true", help="Run continuous refinement")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles for continuous mode")
    parser.add_argument("--concepts", type=int, default=5, help="Concepts per cycle")
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_refinement(cycles=args.cycles, concepts_per_cycle=args.concepts)
    else:
        demo()
