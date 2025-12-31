#!/usr/bin/env python3
"""
Continuous Training Experiment

Validates the idea of continuous autonomous training for emergent gear chains.
Includes benchmarking to measure improvement over training cycles.

Key concepts:
1. Generate new training data using LLM
2. Incrementally train the gear chains
3. Measure quality via benchmarks
4. Repeat until convergence or improvement stops

Benchmarks:
- Semantic coherence: Do similar concepts cluster together?
- Dimension stability: Are dimensions consistent across cycles?
- Response quality: Are responses improving?
- Coverage: How much of the concept space is covered?
"""

import json
import numpy as np
import requests
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.gears.core import SemanticChain, LinguisticChain


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    cycle: int
    timestamp: str
    metrics: Dict[str, float]
    dimension_count: int
    item_count: int
    group_count: int


@dataclass
class TrainingState:
    """State of the continuous training process."""
    cycles_completed: int = 0
    total_items_generated: int = 0
    benchmark_history: List[BenchmarkResult] = field(default_factory=list)
    best_score: float = 0.0
    cycles_without_improvement: int = 0


class ContinuousTrainer:
    """
    Continuous autonomous training for emergent gear chains.
    
    The training loop:
    1. Evaluate current performance (benchmark)
    2. Identify gaps in coverage
    3. Generate new training data for gaps
    4. Retrain chains with new data
    5. Evaluate again
    6. Repeat until convergence
    """
    
    def __init__(self, 
                 semantic_chain: SemanticChain,
                 linguistic_chain: LinguisticChain,
                 model: str = "qwen2:latest"):
        
        self.semantic = semantic_chain
        self.linguistic = linguistic_chain
        self.model = model
        
        self.state = TrainingState()
        
        # Ground truth for benchmarking (known relationships)
        self.ground_truth = {
            'similar_pairs': [
                ('holmes', 'watson'),
                ('holmes', 'detective_work'),
                ('villain', 'moriarty'),
                ('hero', 'brave'),
                ('king', 'queen'),
                ('sage', 'wisdom'),
            ],
            'opposite_pairs': [
                ('hero', 'villain'),
                ('king', 'servant'),
                ('sage', 'child'),
                ('good_vs_evil', 'villain'),
            ],
            'clusters': {
                'detectives': ['holmes', 'watson', 'detective_work', 'sherlock_holmes'],
                'villains': ['villain', 'moriarty', 'spy', 'politician'],
                'royalty': ['king', 'queen', 'leadership'],
                'nature': ['storm', 'fire', 'river', 'tree'],
            }
        }
    
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Call LLM for data generation."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.8}
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"LLM error: {e}")
            return ""
    
    def benchmark(self) -> BenchmarkResult:
        """
        Evaluate current chain performance.
        
        Metrics:
        - similar_accuracy: How well do we identify known similar pairs?
        - opposite_accuracy: How well do we identify known opposites?
        - cluster_coherence: How tight are known clusters?
        - dimension_quality: Variance explained by dimensions
        - coverage: Fraction of ground truth concepts we know
        """
        metrics = {}
        
        # 1. Similar pair accuracy
        similar_correct = 0
        similar_total = 0
        for c1, c2 in self.ground_truth['similar_pairs']:
            if self.semantic.find_group(c1) and self.semantic.find_group(c2):
                similar = self.semantic.find_similar(c1, k=5)
                similar_names = [s[0] for s in similar]
                if any(c2 in s or s in c2 for s in similar_names):
                    similar_correct += 1
                similar_total += 1
        
        metrics['similar_accuracy'] = similar_correct / max(similar_total, 1)
        
        # 2. Opposite pair accuracy
        opposite_correct = 0
        opposite_total = 0
        for c1, c2 in self.ground_truth['opposite_pairs']:
            if self.semantic.find_group(c1) and self.semantic.find_group(c2):
                result = self.semantic.find_opposite(c1)
                if result:
                    opposite = result[0]
                    if c2 in opposite or opposite in c2:
                        opposite_correct += 1
                opposite_total += 1
        
        metrics['opposite_accuracy'] = opposite_correct / max(opposite_total, 1)
        
        # 3. Cluster coherence (average intra-cluster distance)
        cluster_scores = []
        for cluster_name, members in self.ground_truth['clusters'].items():
            known_members = [m for m in members if self.semantic.find_group(m)]
            if len(known_members) >= 2:
                positions = [self.semantic.get_position(m) for m in known_members]
                positions = [p for p in positions if p is not None]
                if len(positions) >= 2:
                    # Calculate average pairwise distance
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            distances.append(np.linalg.norm(positions[i] - positions[j]))
                    avg_dist = np.mean(distances)
                    # Lower distance = better coherence, convert to 0-1 score
                    coherence = 1.0 / (1.0 + avg_dist)
                    cluster_scores.append(coherence)
        
        metrics['cluster_coherence'] = np.mean(cluster_scores) if cluster_scores else 0.0
        
        # 4. Dimension quality (total variance explained)
        total_variance = sum(d.variance for d in self.semantic.dimensions)
        metrics['dimension_quality'] = min(total_variance, 1.0)
        
        # 5. Coverage (fraction of ground truth concepts we know)
        all_concepts = set()
        for pair in self.ground_truth['similar_pairs']:
            all_concepts.update(pair)
        for pair in self.ground_truth['opposite_pairs']:
            all_concepts.update(pair)
        for members in self.ground_truth['clusters'].values():
            all_concepts.update(members)
        
        known_concepts = sum(1 for c in all_concepts if self.semantic.find_group(c))
        metrics['coverage'] = known_concepts / len(all_concepts)
        
        # 6. Composite score
        metrics['composite'] = (
            metrics['similar_accuracy'] * 0.25 +
            metrics['opposite_accuracy'] * 0.20 +
            metrics['cluster_coherence'] * 0.25 +
            metrics['dimension_quality'] * 0.15 +
            metrics['coverage'] * 0.15
        )
        
        result = BenchmarkResult(
            cycle=self.state.cycles_completed,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            dimension_count=len(self.semantic.dimensions),
            item_count=len(self.semantic.items),
            group_count=len(self.semantic.groups),
        )
        
        return result
    
    def identify_gaps(self) -> List[str]:
        """Identify concepts that need more training data."""
        gaps = []
        
        # Check ground truth concepts we don't know
        all_concepts = set()
        for pair in self.ground_truth['similar_pairs']:
            all_concepts.update(pair)
        for pair in self.ground_truth['opposite_pairs']:
            all_concepts.update(pair)
        for members in self.ground_truth['clusters'].values():
            all_concepts.update(members)
        
        for concept in all_concepts:
            if not self.semantic.find_group(concept):
                gaps.append(concept)
        
        # Also check for concepts with low item counts
        for group in self.semantic.groups:
            count = self.semantic.group_counts.get(group, 0)
            if count < 5:
                gaps.append(group)
        
        return list(set(gaps))
    
    def generate_training_data(self, concepts: List[str], n_per_concept: int = 5) -> List[Dict]:
        """Generate new training data for specified concepts."""
        frames = []
        
        # Get context from existing similar concepts
        for concept in concepts[:10]:
            concept_clean = concept.replace('_', ' ').title()
            
            # Find similar known concepts for context
            similar_context = ""
            for group in self.semantic.groups[:5]:
                if group != concept:
                    similar_context += f"- {group.replace('_', ' ').title()}\n"
            
            prompt = f"""Generate {n_per_concept} behavioral sentences for "{concept_clean}".

Context - similar concepts in our system:
{similar_context}

Rules:
1. EVERY sentence MUST start with "{concept_clean}"
2. Second word should be a verb (action word)
3. Format: "{concept_clean} [verb] [rest of sentence]"
4. Use verbs that show characteristic behavior
5. Keep sentences 8-15 words

Examples of good format:
- "{concept_clean} investigates the mysterious case carefully"
- "{concept_clean} commands the troops with authority"

Generate exactly {n_per_concept} sentences, one per line:"""

            response = self._call_llm(prompt)
            
            if response:
                lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
                for line in lines[:n_per_concept]:
                    line = line.lstrip('0123456789.-) ')
                    # Verify it starts with the concept
                    if line.lower().startswith(concept_clean.lower()) and len(line) > 15:
                        frames.append({
                            'text': line,
                            'agent': concept.lower().replace(' ', '_'),
                            'source': 'continuous_training',
                        })
            
            time.sleep(0.3)
        
        return frames
    
    def generate_relationship_data(self, n_pairs: int = 5) -> List[Dict]:
        """Generate data that reinforces known relationships."""
        frames = []
        
        # Generate sentences that show similar concepts together
        for c1, c2 in self.ground_truth['similar_pairs'][:n_pairs]:
            c1_clean = c1.replace('_', ' ').title()
            c2_clean = c2.replace('_', ' ').title()
            
            prompt = f"""Generate 3 sentences showing {c1_clean} and {c2_clean} working together or having similar traits.

Rules:
1. Show them as allies or similar in nature
2. Each sentence should mention both
3. Keep sentences clear and simple

Generate 3 sentences:"""

            response = self._call_llm(prompt)
            if response:
                lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
                for line in lines[:3]:
                    line = line.lstrip('0123456789.-) ')
                    if len(line) > 15:
                        # Add frame for both agents
                        frames.append({'text': line, 'agent': c1.lower(), 'source': 'relationship'})
                        frames.append({'text': line, 'agent': c2.lower(), 'source': 'relationship'})
            
            time.sleep(0.3)
        
        return frames
    
    def train_cycle(self, generate_new_data: bool = True) -> BenchmarkResult:
        """
        Run one training cycle.
        
        1. Benchmark current state
        2. Identify gaps
        3. Generate new data (if enabled)
        4. Generate relationship data to reinforce structure
        5. Retrain chains
        6. Benchmark again
        """
        self.state.cycles_completed += 1
        print(f"\n{'='*60}")
        print(f"TRAINING CYCLE {self.state.cycles_completed}")
        print(f"{'='*60}")
        
        # Pre-training benchmark
        pre_benchmark = self.benchmark()
        print(f"\nPre-training metrics:")
        for k, v in pre_benchmark.metrics.items():
            print(f"  {k}: {v:.3f}")
        
        if generate_new_data:
            # Identify gaps
            gaps = self.identify_gaps()
            print(f"\nIdentified {len(gaps)} gaps: {gaps[:5]}...")
            
            total_new = 0
            
            if gaps:
                # Generate new training data for gaps
                print(f"Generating gap-filling data...")
                new_frames = self.generate_training_data(gaps, n_per_concept=5)
                print(f"  Generated {len(new_frames)} gap frames")
                
                for frame in new_frames:
                    self.semantic.ingest_item(frame)
                    self.linguistic.ingest_item(frame)
                total_new += len(new_frames)
            
            # Generate relationship data to reinforce structure
            print(f"Generating relationship data...")
            rel_frames = self.generate_relationship_data(n_pairs=3)
            print(f"  Generated {len(rel_frames)} relationship frames")
            
            for frame in rel_frames:
                self.semantic.ingest_item(frame)
                self.linguistic.ingest_item(frame)
            total_new += len(rel_frames)
            
            self.state.total_items_generated += total_new
        
        # Retrain dimensions
        print(f"\nRetraining dimensions...")
        semantic_dims = self.semantic.learn_dimensions()
        linguistic_dims = self.linguistic.learn_dimensions()
        print(f"  Semantic: {semantic_dims} dimensions")
        print(f"  Linguistic: {linguistic_dims} dimensions")
        
        # Post-training benchmark
        post_benchmark = self.benchmark()
        print(f"\nPost-training metrics:")
        for k, v in post_benchmark.metrics.items():
            print(f"  {k}: {v:.3f}")
        
        # Track improvement
        improvement = post_benchmark.metrics['composite'] - pre_benchmark.metrics['composite']
        print(f"\nImprovement: {improvement:+.3f}")
        
        if post_benchmark.metrics['composite'] > self.state.best_score:
            self.state.best_score = post_benchmark.metrics['composite']
            self.state.cycles_without_improvement = 0
            print("  ✓ New best score!")
        else:
            self.state.cycles_without_improvement += 1
            print(f"  No improvement for {self.state.cycles_without_improvement} cycles")
        
        self.state.benchmark_history.append(post_benchmark)
        
        return post_benchmark
    
    def train_until_convergence(self, 
                                max_cycles: int = 10,
                                patience: int = 3,
                                generate_new_data: bool = True) -> List[BenchmarkResult]:
        """
        Train until convergence or max cycles reached.
        
        Args:
            max_cycles: Maximum training cycles
            patience: Stop after this many cycles without improvement
            generate_new_data: Whether to generate new LLM data
            
        Returns:
            List of benchmark results
        """
        print("\n" + "=" * 70)
        print("CONTINUOUS TRAINING EXPERIMENT")
        print("=" * 70)
        print(f"Max cycles: {max_cycles}")
        print(f"Patience: {patience}")
        print(f"Generate new data: {generate_new_data}")
        
        results = []
        
        for cycle in range(max_cycles):
            result = self.train_cycle(generate_new_data=generate_new_data)
            results.append(result)
            
            if self.state.cycles_without_improvement >= patience:
                print(f"\nConverged after {cycle + 1} cycles (no improvement for {patience} cycles)")
                break
        
        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Cycles completed: {self.state.cycles_completed}")
        print(f"Total items generated: {self.state.total_items_generated}")
        print(f"Final items: {len(self.semantic.items)}")
        print(f"Final groups: {len(self.semantic.groups)}")
        print(f"Final dimensions: {len(self.semantic.dimensions)}")
        print(f"Best composite score: {self.state.best_score:.3f}")
        
        # Show score progression
        print("\nScore progression:")
        for i, r in enumerate(results):
            print(f"  Cycle {i+1}: {r.metrics['composite']:.3f}")
        
        return results
    
    def save_state(self, path: str):
        """Save training state."""
        state = {
            'cycles_completed': self.state.cycles_completed,
            'total_items_generated': self.state.total_items_generated,
            'best_score': self.state.best_score,
            'benchmark_history': [
                {
                    'cycle': b.cycle,
                    'timestamp': b.timestamp,
                    'metrics': b.metrics,
                    'dimension_count': b.dimension_count,
                    'item_count': b.item_count,
                    'group_count': b.group_count,
                }
                for b in self.state.benchmark_history
            ],
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)


def run_experiment():
    """Run the continuous training experiment."""
    print("=" * 70)
    print("CONTINUOUS TRAINING EXPERIMENT")
    print("=" * 70)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is running")
    except:
        print("WARNING: Ollama not running - will skip data generation")
        generate_data = False
    else:
        generate_data = True
    
    # Create chains
    semantic = SemanticChain("Understanding")
    linguistic = LinguisticChain("Output")
    
    # Load initial corpus
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if corpus_path.exists():
        print(f"\nLoading initial corpus: {corpus_path}")
        semantic.ingest_corpus(str(corpus_path))
        linguistic.ingest_corpus(str(corpus_path))
        print(f"  Loaded {len(semantic.items)} items")
    
    # Initial training
    print("\nInitial dimension learning...")
    semantic.learn_dimensions()
    linguistic.learn_dimensions()
    print(f"  Semantic: {len(semantic.dimensions)} dimensions")
    print(f"  Linguistic: {len(linguistic.dimensions)} dimensions")
    
    # Create trainer
    trainer = ContinuousTrainer(semantic, linguistic)
    
    # Run training
    results = trainer.train_until_convergence(
        max_cycles=5,
        patience=2,
        generate_new_data=generate_data,
    )
    
    # Save state
    state_path = Path(__file__).parent / "continuous_training_state.json"
    trainer.save_state(str(state_path))
    print(f"\nState saved to: {state_path}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = run_experiment()
