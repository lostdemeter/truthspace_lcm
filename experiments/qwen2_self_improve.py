#!/usr/bin/env python3
"""
Qwen2 Self-Improvement Daemon

A long-running process that continuously improves the experimental corpus by:
1. Generating answers from the corpus (using Natural Lens)
2. Sending them to Qwen2 for rewriting
3. Parsing and ingesting the improvements
4. Periodically cleaning up bad frames
5. Saving progress regularly

This creates a feedback loop where Qwen2 acts as a "teacher" to improve
our geometric knowledge corpus over time.

Usage:
    python3 experiments/qwen2_self_improve.py --cycles 100 --cleanup-every 20

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ollama_corpus_refiner import OllamaClient
from experiments.corpus_cleanup import CorpusAnalyzer, CorpusCleaner
from experiments.geometric_reinforcement import GeometricRL
from truthspace_lcm.core.geometric import GeometricQA


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/qwen2_self_improve.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ImprovementStats:
    """Track improvement statistics."""
    cycles_completed: int = 0
    concepts_refined: int = 0
    frames_added: int = 0
    frames_removed: int = 0
    cleanups_performed: int = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'cycles_completed': self.cycles_completed,
            'concepts_refined': self.concepts_refined,
            'frames_added': self.frames_added,
            'frames_removed': self.frames_removed,
            'cleanups_performed': self.cleanups_performed,
            'errors': self.errors,
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),
        }


class Qwen2SelfImprover:
    """
    Long-running self-improvement daemon using Qwen2.
    
    The workflow:
    1. Pick concepts that need improvement
    2. Generate current answer using Natural Lens
    3. Ask Qwen2 to rewrite it better
    4. Parse and ingest the improvements
    5. Periodically clean up bad frames
    6. Save progress regularly
    """
    
    def __init__(self, corpus_path: str, model: str = "qwen2:latest"):
        self.corpus_path = corpus_path
        self.model = model
        
        # Initialize components
        self.ollama = OllamaClient(model=model)
        self.qa = GeometricQA()
        self.qa.load_corpus(corpus_path)
        self.qa.set_output_lens('natural')
        
        self.grl = GeometricRL(corpus_path)
        
        # Track which concepts we've refined
        self.refined_concepts: Set[str] = set()
        self.stats = ImprovementStats()
        
        # Known bad patterns to clean
        self.bad_patterns = [
            'ecologies', 'observables', 'categorizes', 'safeguards',
            'cuneiform', 'nuances', 'mores', 'philosophies',
            'grappls', 'snappings', 'betweens', 'childrens',
            'haves ', 'fosteres', 'monitores', 'continus',
            'alteres', 'curtailes', 'bes ', 'enabls',
            ' across ', ' various ', ' extraordinary ',
        ]
        
        # Good verbs for reinforcement
        self.good_verbs = {
            'studies', 'examines', 'investigates', 'explores', 'analyzes',
            'describes', 'explains', 'discovers', 'observes', 'measures',
            'solves', 'deduces', 'reasons', 'assists', 'helps', 'supports',
            'documents', 'records', 'provides', 'includes', 'involves',
            'creates', 'develops', 'produces', 'generates', 'processes',
            'transforms', 'changes', 'adapts', 'evolves', 'grows',
            'calculates', 'proves', 'demonstrates', 'shows', 'reveals',
            'powers', 'flows', 'exists', 'composes', 'forms',
        }
    
    def get_refinable_concepts(self, batch_size: int = 10) -> List[str]:
        """Get concepts that are good candidates for refinement."""
        # Priority concepts - important ones we want to improve first
        priority_concepts = [
            'physics', 'biology', 'chemistry', 'mathematics', 'science',
            'evolution', 'consciousness', 'matter', 'energy', 'nature',
            'holmes', 'watson', 'detective', 'doctor', 'mystery',
            'brain', 'mind', 'thought', 'intelligence', 'knowledge',
            'force', 'gravity', 'light', 'time', 'space',
            'life', 'death', 'love', 'truth', 'beauty',
            'philosophy', 'psychology', 'sociology', 'economics', 'history',
            'art', 'music', 'literature', 'language', 'culture',
            'technology', 'computer', 'internet', 'machine', 'robot',
            'human', 'animal', 'plant', 'cell', 'gene',
            'atom', 'molecule', 'electron', 'proton', 'neutron',
            'earth', 'sun', 'moon', 'star', 'planet',
            'water', 'air', 'fire', 'metal', 'wood',
        ]
        
        candidates = []
        
        # First, add priority concepts that haven't been refined
        for name in priority_concepts:
            if name in self.refined_concepts:
                continue
            if name in self.qa.knowledge.concepts:
                candidates.append(name)
        
        # Then add other concepts with bad actions
        for name, concept in self.qa.knowledge.concepts.items():
            if name in self.refined_concepts or name in candidates:
                continue
            
            # Skip noise
            if len(name) < 4 or name[0].isdigit():
                continue
            if not concept.is_content_word:
                continue
            
            # Must have some actions
            if not concept.actions:
                continue
            
            # Must have reasonable attestation (not obscure)
            total_count = concept.initiator_count + concept.mediator_count
            if total_count < 5:
                continue
            
            # Check if it has bad actions that need fixing
            has_bad_actions = False
            for action, _ in concept.actions.most_common(5):
                if action.lower() not in self.good_verbs:
                    has_bad_actions = True
                    break
            
            if has_bad_actions:
                candidates.append(name)
        
        # Return a batch
        return candidates[:batch_size]
    
    def generate_answer(self, entity: str) -> str:
        """Generate current answer using Natural Lens."""
        query = f"What is {entity}?"
        return self.qa.ask(query)
    
    def rewrite_with_qwen2(self, query: str, answer: str) -> Optional[str]:
        """Ask Qwen2 to rewrite the answer."""
        prompt = f"""Given the question "{query}" and the answer "{answer}", rewrite the answer to make it as clear and natural as possible while keeping the same meaning. Only reply with the rewritten answer, nothing else."""
        
        try:
            response = self.ollama.generate(prompt, temperature=0.3)
            return response
        except Exception as e:
            logger.error(f"Qwen2 error: {e}")
            self.stats.errors += 1
            return None
    
    def refine_concept(self, entity: str) -> bool:
        """Refine a single concept."""
        query = f"What is {entity}?"
        
        # Get current answer
        current_answer = self.generate_answer(entity)
        
        if "don't know" in current_answer.lower():
            return False
        
        # Get Qwen2 rewrite
        rewritten = self.rewrite_with_qwen2(query, current_answer)
        
        if not rewritten:
            return False
        
        # Apply geometric reinforcement
        try:
            results = self.grl.correct(query, rewritten)
            self.stats.frames_added += results.get('frames_added', 0)
            self.refined_concepts.add(entity)
            self.stats.concepts_refined += 1
            return True
        except Exception as e:
            logger.error(f"Reinforcement error for {entity}: {e}")
            self.stats.errors += 1
            return False
    
    def run_cleanup(self) -> int:
        """Run cleanup to remove bad frames."""
        logger.info("Running cleanup...")
        
        cleaner = CorpusCleaner(self.corpus_path)
        removed = cleaner.remove_bad_frames(self.bad_patterns)
        
        self.stats.frames_removed += removed
        self.stats.cleanups_performed += 1
        
        cleaner.save_corpus()
        
        # Reload the corpus after cleanup
        self.qa.load_corpus(self.corpus_path)
        self.grl = GeometricRL(self.corpus_path)
        
        logger.info(f"Cleanup complete: removed {removed} bad frames")
        return removed
    
    def save_progress(self, deduplicate: bool = True):
        """Save current progress with optional deduplication."""
        # Save corpus
        self.grl.save_corpus(self.corpus_path)
        
        # Deduplicate to save space
        if deduplicate:
            self._deduplicate_corpus()
        
        # Save stats
        stats_path = 'logs/qwen2_improve_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        logger.info(f"Progress saved: {self.stats.concepts_refined} concepts refined")
    
    def _deduplicate_corpus(self):
        """Deduplicate corpus by storing counts instead of duplicates."""
        from collections import defaultdict
        
        with open(self.corpus_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        original_count = len(frames)
        
        # Group by text
        frame_counts = defaultdict(lambda: {'count': 0, 'sources': set()})
        for frame in frames:
            text = frame.get('text', '')
            source = frame.get('source', '')
            # Handle existing count field
            count = frame.get('count', 1)
            if text:
                frame_counts[text]['count'] += count
                if source:
                    frame_counts[text]['sources'].add(source)
        
        # Create deduplicated frames
        deduped_frames = []
        for text, info in frame_counts.items():
            sources = list(info['sources'])
            source = sources[0] if len(sources) == 1 else ('mixed' if sources else '')
            
            frame = {'text': text, 'source': source}
            if info['count'] > 1:
                frame['count'] = info['count']
            deduped_frames.append(frame)
        
        # Save
        with open(self.corpus_path, 'w') as f:
            json.dump({'frames': deduped_frames}, f, indent=2)
        
        if original_count != len(deduped_frames):
            logger.info(f"Deduplicated: {original_count} -> {len(deduped_frames)} frames")
    
    def run(self, cycles: int = 100, concepts_per_cycle: int = 5, 
            cleanup_every: int = 20, save_every: int = 10):
        """
        Run the self-improvement loop.
        
        Args:
            cycles: Number of improvement cycles to run
            concepts_per_cycle: Concepts to refine per cycle
            cleanup_every: Run cleanup every N cycles
            save_every: Save progress every N cycles
        """
        logger.info(f"Starting Qwen2 self-improvement: {cycles} cycles")
        logger.info(f"  Concepts per cycle: {concepts_per_cycle}")
        logger.info(f"  Cleanup every: {cleanup_every} cycles")
        logger.info(f"  Save every: {save_every} cycles")
        
        # Check Ollama
        if not self.ollama.is_available():
            logger.error("Ollama is not running!")
            return
        
        for cycle in range(1, cycles + 1):
            logger.info(f"=== Cycle {cycle}/{cycles} ===")
            
            # Get concepts to refine
            concepts = self.get_refinable_concepts(concepts_per_cycle)
            
            if not concepts:
                logger.info("No more concepts to refine, resetting...")
                self.refined_concepts.clear()
                concepts = self.get_refinable_concepts(concepts_per_cycle)
            
            # Refine each concept
            for entity in concepts:
                success = self.refine_concept(entity)
                if success:
                    logger.info(f"  ✓ Refined: {entity}")
                else:
                    logger.info(f"  - Skipped: {entity}")
            
            self.stats.cycles_completed = cycle
            
            # Periodic cleanup
            if cycle % cleanup_every == 0:
                self.run_cleanup()
            
            # Periodic save
            if cycle % save_every == 0:
                self.save_progress()
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)
        
        # Final save
        self.save_progress()
        self.run_cleanup()
        
        logger.info("Self-improvement complete!")
        logger.info(f"Final stats: {self.stats.to_dict()}")
    
    def print_sample_outputs(self):
        """Print sample outputs to show current quality."""
        print("\nSample outputs after improvement:")
        print("-" * 60)
        
        test_queries = [
            "What is physics?",
            "What does Holmes do?",
            "What is evolution?",
            "What is consciousness?",
        ]
        
        for query in test_queries:
            answer = self.qa.ask(query)
            print(f"Q: {query}")
            print(f"A: {answer}")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2 Self-Improvement Daemon")
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles (0 = infinite)")
    parser.add_argument("--concepts", type=int, default=5, help="Concepts per cycle")
    parser.add_argument("--cleanup-every", type=int, default=20, help="Cleanup frequency")
    parser.add_argument("--save-every", type=int, default=10, help="Save frequency")
    parser.add_argument("--corpus", type=str, default="truthspace_lcm/corpus_experimental.json",
                        help="Path to corpus")
    parser.add_argument("--model", type=str, default="qwen2:latest", help="Ollama model")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--hours", type=float, default=0, help="Run for N hours (overrides --cycles)")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Calculate cycles if hours specified
    if args.hours > 0:
        # Estimate ~3 seconds per concept
        concepts_per_hour = 3600 / 3 / args.concepts
        args.cycles = int(args.hours * concepts_per_hour)
        print(f"Running for {args.hours} hours (~{args.cycles} cycles)")
    
    if not args.quiet:
        print("=" * 70)
        print("QWEN2 SELF-IMPROVEMENT DAEMON")
        print("=" * 70)
        print()
        print(f"Corpus: {args.corpus}")
        print(f"Model: {args.model}")
        print(f"Cycles: {args.cycles if args.cycles > 0 else 'infinite'}")
        print(f"Concepts per cycle: {args.concepts}")
        print(f"Cleanup every: {args.cleanup_every} cycles")
        print()
    
    improver = Qwen2SelfImprover(args.corpus, model=args.model)
    
    if not args.quiet:
        print("BEFORE improvement:")
        improver.print_sample_outputs()
    
    try:
        # Run improvement (cycles=0 means infinite)
        if args.cycles == 0:
            logger.info("Running in infinite mode - press Ctrl+C to stop")
            cycle = 0
            while True:
                cycle += 1
                # Run one cycle at a time
                improver.run(
                    cycles=1,
                    concepts_per_cycle=args.concepts,
                    cleanup_every=1 if cycle % args.cleanup_every == 0 else 999999,
                    save_every=1 if cycle % args.save_every == 0 else 999999,
                )
        else:
            improver.run(
                cycles=args.cycles,
                concepts_per_cycle=args.concepts,
                cleanup_every=args.cleanup_every,
                save_every=args.save_every,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user - saving progress...")
        improver.save_progress()
        logger.info("Progress saved!")
    
    if not args.quiet:
        print("\nAFTER improvement:")
        improver.print_sample_outputs()


if __name__ == "__main__":
    main()
