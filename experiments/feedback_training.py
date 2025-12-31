#!/usr/bin/env python3
"""
Feedback-Corrected Training

Uses LLM feedback to correct emergent outputs and backpropagate
corrections to adjust the dimensional space.

The key insight: When the system says "opposite of hero is fire",
an LLM can correct this to "villain". We then:
1. Generate synthetic data reinforcing hero↔villain opposition
2. Adjust dimensional positions to reflect the correction
3. Retrain with the corrected relationships

This is RLAIF (Reinforcement Learning from AI Feedback) for emergent systems.
"""

import json
import numpy as np
import requests
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fully_emergent_chains import (
    FullyEmergentSemanticChain,
    FullyEmergentChatbot,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"


@dataclass
class Correction:
    """A correction from LLM feedback."""
    query_type: str  # 'opposite', 'similar', 'trait'
    concept: str
    original_answer: str
    corrected_answer: str
    confidence: float
    reasoning: str = ""


@dataclass
class FeedbackStats:
    """Statistics from feedback loop."""
    queries_evaluated: int = 0
    corrections_made: int = 0
    corrections_applied: int = 0
    synthetic_frames_generated: int = 0


class FeedbackTrainer:
    """
    Trains emergent chains with LLM feedback correction.
    
    Flow:
    1. Generate outputs from current model
    2. Ask LLM to evaluate and correct outputs
    3. Generate synthetic data from corrections
    4. Backpropagate by adjusting positions/retraining
    """
    
    def __init__(self, semantic_chain: FullyEmergentSemanticChain):
        self.semantic = semantic_chain
        self.corrections: List[Correction] = []
        self.stats = FeedbackStats()
        
        # Track relationship corrections for backprop
        self.opposite_corrections: Dict[str, str] = {}  # concept -> correct_opposite
        self.similar_corrections: Dict[str, List[str]] = defaultdict(list)
        self.trait_corrections: Dict[str, List[str]] = defaultdict(list)
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call Ollama API."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM error: {e}")
        return None
    
    def evaluate_opposite(self, concept: str) -> Optional[Correction]:
        """Ask LLM to evaluate an opposite relationship."""
        result = self.semantic.find_opposite(concept)
        if not result:
            return None
        
        original = result[0]
        
        # Get list of known concepts to constrain the answer
        known_concepts = [g.replace('_', ' ').title() for g in self.semantic.groups[:30]]
        known_str = ', '.join(known_concepts)
        
        prompt = f"""Evaluate this semantic relationship:

"The opposite of {concept.title()} is {original.title()}"

Is this correct? If not, what would be a better opposite from this list of known concepts?

KNOWN CONCEPTS: {known_str}

Rules:
1. Consider semantic/behavioral opposition
2. IMPORTANT: Your answer MUST be from the known concepts list above
3. Answer in this exact format:
   CORRECT: yes/no
   BETTER_OPPOSITE: [concept from list, or "none"]
   REASONING: [brief explanation]

Examples:
- Opposite of "hero" should be "villain" (behavioral opposition)
- Opposite of "king" could be "queen" or "servant" (role opposition)
- Opposite of "sage" could be "child" (wisdom opposition)

Evaluate:"""

        response = self._call_llm(prompt)
        if not response:
            return None
        
        self.stats.queries_evaluated += 1
        
        # Parse response
        lines = response.strip().split('\n')
        correct = True
        better = None
        reasoning = ""
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('correct:'):
                correct = 'yes' in line_lower
            elif line_lower.startswith('better_opposite:'):
                better = line.split(':', 1)[1].strip().lower()
                if better in ['none', 'n/a', '']:
                    better = None
            elif line_lower.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if not correct and better and better != original.lower():
            correction = Correction(
                query_type='opposite',
                concept=concept,
                original_answer=original,
                corrected_answer=better,
                confidence=0.8,
                reasoning=reasoning
            )
            self.corrections.append(correction)
            self.opposite_corrections[concept] = better
            self.stats.corrections_made += 1
            return correction
        
        return None
    
    def evaluate_similar(self, concept: str) -> Optional[Correction]:
        """Ask LLM to evaluate similarity relationships."""
        similar = self.semantic.find_similar(concept, k=5)
        if not similar:
            return None
        
        similar_names = [s[0] for s in similar]
        
        prompt = f"""Evaluate these semantic similarities:

"{concept.title()} is similar to: {', '.join([s.title() for s in similar_names])}"

Are these good similarity matches? Which ones are wrong, and what would be better?

Rules:
1. Similar concepts should share behavioral or role characteristics
2. Answer in this exact format:
   WRONG: [list of wrong matches, or "none"]
   BETTER_SIMILAR: [list of better matches, or "none"]
   REASONING: [brief explanation]

Evaluate:"""

        response = self._call_llm(prompt)
        if not response:
            return None
        
        self.stats.queries_evaluated += 1
        
        # Parse response
        lines = response.strip().split('\n')
        wrong = []
        better = []
        reasoning = ""
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('wrong:'):
                wrong_str = line.split(':', 1)[1].strip()
                if wrong_str.lower() not in ['none', 'n/a', '']:
                    wrong = [w.strip().lower() for w in wrong_str.split(',')]
            elif line_lower.startswith('better_similar:'):
                better_str = line.split(':', 1)[1].strip()
                if better_str.lower() not in ['none', 'n/a', '']:
                    better = [b.strip().lower() for b in better_str.split(',')]
            elif line_lower.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if better:
            correction = Correction(
                query_type='similar',
                concept=concept,
                original_answer=', '.join(similar_names),
                corrected_answer=', '.join(better),
                confidence=0.7,
                reasoning=reasoning
            )
            self.corrections.append(correction)
            self.similar_corrections[concept].extend(better)
            self.stats.corrections_made += 1
            return correction
        
        return None
    
    def evaluate_traits(self, concept: str) -> Optional[Correction]:
        """Ask LLM to evaluate trait descriptions."""
        traits = self.semantic.describe_traits(concept)
        if not traits:
            return None
        
        prompt = f"""Evaluate these character traits:

"{concept.title()} exhibits {', '.join(traits)} qualities"

Are these good trait descriptions? What traits would better describe {concept.title()}?

Rules:
1. Traits should be behavioral/personality descriptors
2. Answer in this exact format:
   ACCURATE: yes/no
   BETTER_TRAITS: [list of better traits, or "none"]
   REASONING: [brief explanation]

Common good traits: analytical, heroic, wise, cunning, loyal, brave, scheming, nurturing, authoritative, playful

Evaluate:"""

        response = self._call_llm(prompt)
        if not response:
            return None
        
        self.stats.queries_evaluated += 1
        
        # Parse response
        lines = response.strip().split('\n')
        accurate = True
        better = []
        reasoning = ""
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('accurate:'):
                accurate = 'yes' in line_lower
            elif line_lower.startswith('better_traits:'):
                better_str = line.split(':', 1)[1].strip()
                if better_str.lower() not in ['none', 'n/a', '']:
                    better = [b.strip().lower() for b in better_str.split(',')]
            elif line_lower.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if not accurate and better:
            correction = Correction(
                query_type='trait',
                concept=concept,
                original_answer=', '.join(traits),
                corrected_answer=', '.join(better),
                confidence=0.6,
                reasoning=reasoning
            )
            self.corrections.append(correction)
            self.trait_corrections[concept].extend(better)
            self.stats.corrections_made += 1
            return correction
        
        return None
    
    def generate_correction_data(self) -> List[Dict]:
        """Generate synthetic training data from corrections."""
        frames = []
        
        # Generate data for opposite corrections
        for concept, correct_opposite in self.opposite_corrections.items():
            # Generate contrastive sentences
            prompt = f"""Generate 3 sentences showing {concept.title()} and {correct_opposite.title()} as opposites.

Rules:
1. Show them in opposition or conflict
2. Highlight their contrasting behaviors
3. Keep sentences 8-15 words
4. Start each sentence with one of the names

Generate 3 sentences:"""

            response = self._call_llm(prompt, max_tokens=300)
            if response:
                for line in response.strip().split('\n'):
                    line = line.strip().lstrip('0123456789.-) ')
                    if len(line) > 15:
                        # Determine which agent the sentence is about
                        if line.lower().startswith(concept.lower()):
                            frames.append({
                                'text': line,
                                'agent': concept.lower(),
                                'source': 'feedback_correction',
                                'correction_type': 'opposite',
                            })
                        elif line.lower().startswith(correct_opposite.lower()):
                            frames.append({
                                'text': line,
                                'agent': correct_opposite.lower(),
                                'source': 'feedback_correction',
                                'correction_type': 'opposite',
                            })
            
            time.sleep(0.2)
        
        # Generate data for similar corrections
        for concept, similar_list in self.similar_corrections.items():
            for similar in similar_list[:3]:  # Limit to top 3
                prompt = f"""Generate 2 sentences showing {concept.title()} and {similar.title()} as similar.

Rules:
1. Show them working together or having similar traits
2. Keep sentences 8-15 words
3. Start each sentence with one of the names

Generate 2 sentences:"""

                response = self._call_llm(prompt, max_tokens=200)
                if response:
                    for line in response.strip().split('\n'):
                        line = line.strip().lstrip('0123456789.-) ')
                        if len(line) > 15:
                            if line.lower().startswith(concept.lower()):
                                frames.append({
                                    'text': line,
                                    'agent': concept.lower(),
                                    'source': 'feedback_correction',
                                    'correction_type': 'similar',
                                })
                            elif line.lower().startswith(similar.lower()):
                                frames.append({
                                    'text': line,
                                    'agent': similar.lower(),
                                    'source': 'feedback_correction',
                                    'correction_type': 'similar',
                                })
                
                time.sleep(0.2)
        
        # Generate data for trait corrections
        for concept, traits in self.trait_corrections.items():
            trait_str = ', '.join(traits[:3])
            prompt = f"""Generate 3 sentences showing {concept.title()} being {trait_str}.

Rules:
1. Each sentence should demonstrate one of the traits
2. Start each sentence with "{concept.title()}"
3. Keep sentences 8-15 words

Generate 3 sentences:"""

            response = self._call_llm(prompt, max_tokens=300)
            if response:
                for line in response.strip().split('\n'):
                    line = line.strip().lstrip('0123456789.-) ')
                    if len(line) > 15 and line.lower().startswith(concept.lower()):
                        frames.append({
                            'text': line,
                            'agent': concept.lower(),
                            'source': 'feedback_correction',
                            'correction_type': 'trait',
                        })
            
            time.sleep(0.2)
        
        self.stats.synthetic_frames_generated = len(frames)
        return frames
    
    def backpropagate_corrections(self, frames: List[Dict]) -> int:
        """
        Backpropagate corrections by:
        1. Directly adjusting dimensional positions for opposite corrections
        2. Adding synthetic data and retraining
        
        The direct adjustment is key - we modify U matrix positions
        to enforce corrected relationships.
        """
        if not frames:
            return 0
        
        applied = 0
        
        # DIRECT BACKPROP: Adjust positions for opposite corrections
        # Use a bounded approach - push apart but keep within reasonable range
        if self.semantic.U is not None and len(self.semantic.dimensions) > 0:
            adjustments_made = []
            
            for concept, correct_opposite in self.opposite_corrections.items():
                concept_idx = None
                opposite_idx = None
                
                # Find indices - handle multi-word corrections
                correct_opposite_clean = correct_opposite.split(',')[0].strip().lower()
                correct_opposite_clean = correct_opposite_clean.strip('"\'')
                
                for i, g in enumerate(self.semantic.groups):
                    if g == concept:
                        concept_idx = i
                    if g == correct_opposite_clean or correct_opposite_clean in g:
                        opposite_idx = i
                
                if concept_idx is not None and opposite_idx is not None:
                    # Avoid duplicate adjustments (hero↔villain and villain↔hero)
                    pair = tuple(sorted([concept_idx, opposite_idx]))
                    if pair in adjustments_made:
                        continue
                    adjustments_made.append(pair)
                    
                    print(f"    Adjusting {concept} ↔ {self.semantic.groups[opposite_idx]}")
                    
                    # Use bounded adjustment - push to opposite signs with fixed magnitude
                    n_dims = min(3, self.semantic.U.shape[1])  # Only adjust first 3 dims
                    target_mag = 0.5  # Fixed, bounded magnitude
                    
                    for d in range(n_dims):
                        # Push to opposite sides
                        self.semantic.U[concept_idx, d] = target_mag
                        self.semantic.U[opposite_idx, d] = -target_mag
                        applied += 1
                    
                    print(f"      Set to ±{target_mag:.2f} on {n_dims} dimensions")
                else:
                    if opposite_idx is None:
                        print(f"    Skipping: {correct_opposite_clean} not in vocabulary")
            
            # Normalize U matrix to prevent runaway values
            if applied > 0:
                max_val = np.max(np.abs(self.semantic.U))
                if max_val > 1.0:
                    self.semantic.U = self.semantic.U / max_val
                    print(f"    Normalized U matrix (was max {max_val:.2f})")
        
        # SOFT BACKPROP: Ingest correction frames for future training
        for frame in frames:
            self.semantic.stopword_chain.ingest_item(frame)
            self.semantic.label_chain.ingest_item(frame)
            self.semantic.template_chain.ingest_item(frame)
            self.semantic.ingest_item(frame)
        
        # Retrain sub-chains (but keep adjusted U matrix for main chain)
        self.semantic.stopword_chain.learn_dimensions(min_variance=0.01, max_dims=5)
        self.semantic.stopword_chain.discover_stopwords(spread_threshold=0.2, freq_threshold=0.005)
        self.semantic.label_chain.learn_dimensions(min_variance=0.03, max_dims=10)
        self.semantic.template_chain.learn_dimensions(min_variance=0.05, max_dims=6)
        
        self.stats.corrections_applied = applied + len(frames)
        return applied + len(frames)
    
    def feedback_cycle(self, concepts: List[str] = None) -> FeedbackStats:
        """
        Run one feedback cycle:
        1. Evaluate outputs for given concepts
        2. Collect corrections
        3. Generate synthetic data
        4. Backpropagate
        """
        if concepts is None:
            concepts = self.semantic.groups[:20]  # Sample of concepts
        
        print(f"\n{'='*60}")
        print("FEEDBACK CYCLE")
        print(f"{'='*60}")
        
        # Reset stats
        self.stats = FeedbackStats()
        self.corrections = []
        self.opposite_corrections = {}
        self.similar_corrections = defaultdict(list)
        self.trait_corrections = defaultdict(list)
        
        # Evaluate outputs
        print(f"\nEvaluating {len(concepts)} concepts...")
        for concept in concepts:
            print(f"  Evaluating: {concept}")
            
            # Evaluate opposite
            corr = self.evaluate_opposite(concept)
            if corr:
                print(f"    ✗ Opposite: {corr.original_answer} → {corr.corrected_answer}")
            
            # Evaluate similar (sample)
            if np.random.random() < 0.3:  # 30% chance to evaluate similar
                corr = self.evaluate_similar(concept)
                if corr:
                    print(f"    ✗ Similar: corrected")
            
            # Evaluate traits (sample)
            if np.random.random() < 0.3:  # 30% chance to evaluate traits
                corr = self.evaluate_traits(concept)
                if corr:
                    print(f"    ✗ Traits: {corr.original_answer} → {corr.corrected_answer}")
            
            time.sleep(0.1)
        
        print(f"\nCorrections found: {self.stats.corrections_made}")
        
        # Generate correction data
        if self.stats.corrections_made > 0:
            print(f"\nGenerating correction data...")
            frames = self.generate_correction_data()
            print(f"  Generated {len(frames)} frames")
            
            # Backpropagate
            print(f"\nBackpropagating corrections...")
            applied = self.backpropagate_corrections(frames)
            print(f"  Applied {applied} corrections")
        
        return self.stats


class ContinuousFeedbackTrainer:
    """
    Continuous training with periodic feedback correction.
    
    Combines:
    1. Data generation (forward pass)
    2. Feedback evaluation (quality check)
    3. Backpropagation (correction)
    """
    
    def __init__(self):
        self.semantic = FullyEmergentSemanticChain()
        self.feedback = None
        self.cycle_count = 0
        self.total_corrections = 0
        self.history: List[Dict] = []
    
    def load_corpus(self, corpus_path: str) -> int:
        count = self.semantic.ingest_corpus(corpus_path)
        return count
    
    def initial_train(self):
        """Initial training pass."""
        self.semantic.learn_dimensions()
        self.feedback = FeedbackTrainer(self.semantic)
    
    def _generate_new_data(self, concepts: List[str], n_per_concept: int = 3) -> List[Dict]:
        """Generate new training data for concepts."""
        frames = []
        
        for concept in concepts[:5]:
            prompt = f"""Generate {n_per_concept} behavioral sentences for "{concept.title()}".

Rules:
1. Start each sentence with "{concept.title()}"
2. Second word should be a verb
3. Keep sentences 8-15 words
4. Show characteristic behavior

Generate {n_per_concept} sentences:"""

            response = self.feedback._call_llm(prompt, max_tokens=300)
            if response:
                for line in response.strip().split('\n'):
                    line = line.strip().lstrip('0123456789.-) ')
                    if len(line) > 15 and line.lower().startswith(concept.lower()):
                        frames.append({
                            'text': line,
                            'agent': concept.lower(),
                            'source': 'continuous_training',
                        })
            time.sleep(0.2)
        
        return frames
    
    def training_cycle(self, 
                       generate_data: bool = True,
                       run_feedback: bool = True,
                       sample_size: int = 10) -> Dict:
        """
        Run one training cycle:
        1. Sample concepts to evaluate
        2. Optionally generate new data
        3. Run feedback evaluation and correction
        4. Report metrics
        """
        self.cycle_count += 1
        
        print(f"\n{'='*60}")
        print(f"TRAINING CYCLE {self.cycle_count}")
        print(f"{'='*60}")
        
        # Sample concepts
        concepts = self.semantic.groups[:sample_size]
        
        # Pre-cycle metrics
        pre_metrics = self._compute_metrics(concepts)
        print(f"\nPre-cycle metrics:")
        for k, v in pre_metrics.items():
            print(f"  {k}: {v}")
        
        frames_generated = 0
        corrections_made = 0
        
        # Generate new data
        if generate_data:
            print(f"\nGenerating new data...")
            new_frames = self._generate_new_data(concepts)
            frames_generated = len(new_frames)
            print(f"  Generated {frames_generated} frames")
            
            for frame in new_frames:
                self.semantic.ingest_item(frame)
        
        # Run feedback correction
        if run_feedback:
            print(f"\nRunning feedback evaluation...")
            stats = self.feedback.feedback_cycle(concepts)
            corrections_made = stats.corrections_made
            self.total_corrections += corrections_made
        
        # Post-cycle metrics
        post_metrics = self._compute_metrics(concepts)
        print(f"\nPost-cycle metrics:")
        for k, v in post_metrics.items():
            print(f"  {k}: {v}")
        
        # Record history
        cycle_record = {
            'cycle': self.cycle_count,
            'frames_generated': frames_generated,
            'corrections_made': corrections_made,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
        }
        self.history.append(cycle_record)
        
        return cycle_record
    
    def _compute_metrics(self, concepts: List[str]) -> Dict:
        """Compute quality metrics for concepts."""
        # Count how many have "reasonable" opposites (not fire/storm/etc)
        reasonable_opposites = 0
        has_traits = 0
        
        nature_concepts = {'fire', 'storm', 'river', 'tree', 'dog', 'fox', 'hearts'}
        
        for concept in concepts:
            opposite = self.semantic.find_opposite(concept)
            if opposite and opposite[0] not in nature_concepts:
                reasonable_opposites += 1
            
            traits = self.semantic.describe_traits(concept)
            if traits:
                has_traits += 1
        
        return {
            'reasonable_opposites': f"{reasonable_opposites}/{len(concepts)}",
            'has_traits': f"{has_traits}/{len(concepts)}",
            'total_items': len(self.semantic.items),
            'dimensions': len(self.semantic.dimensions),
        }
    
    def run_training(self, n_cycles: int = 3):
        """Run multiple training cycles."""
        print("=" * 70)
        print("CONTINUOUS FEEDBACK TRAINING")
        print("=" * 70)
        
        for i in range(n_cycles):
            self.training_cycle(
                generate_data=(i % 2 == 0),  # Generate data every other cycle
                run_feedback=True,
            )
        
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Total corrections: {self.total_corrections}")
        print(f"  Final items: {len(self.semantic.items)}")
        print(f"  Final dimensions: {len(self.semantic.dimensions)}")
        
        # Show final outputs
        print("\nFinal outputs:")
        for concept in ['hero', 'villain', 'holmes', 'watson', 'sage', 'king']:
            opposite = self.semantic.find_opposite(concept)
            traits = self.semantic.describe_traits(concept)
            opp_str = opposite[0] if opposite else "none"
            print(f"  {concept}: opposite={opp_str}, traits={traits[:2] if traits else []}")


def test_feedback_training():
    """Test the feedback training system."""
    print("=" * 70)
    print("FEEDBACK-CORRECTED TRAINING")
    print("=" * 70)
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama not running!")
            return
        print("Ollama is running")
    except:
        print("Ollama not available!")
        return
    
    # Create and train initial model
    chain = FullyEmergentSemanticChain()
    
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    print(f"\nLoading corpus: {corpus_path}")
    count = chain.ingest_corpus(str(corpus_path))
    print(f"  Loaded {count} items")
    
    print("\nInitial training...")
    chain.learn_dimensions()
    print(f"  Dimensions: {len(chain.dimensions)}")
    
    # Show initial outputs
    print("\n" + "-" * 60)
    print("BEFORE FEEDBACK")
    print("-" * 60)
    
    test_concepts = ['hero', 'villain', 'holmes', 'watson', 'sage']
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        traits = chain.describe_traits(concept)
        opp_str = opposite[0] if opposite else "none"
        print(f"  {concept}: opposite={opp_str}, traits={traits}")
    
    # Run feedback cycle
    trainer = FeedbackTrainer(chain)
    stats = trainer.feedback_cycle(test_concepts)
    
    # Show corrected outputs
    print("\n" + "-" * 60)
    print("AFTER FEEDBACK")
    print("-" * 60)
    
    for concept in test_concepts:
        opposite = chain.find_opposite(concept)
        traits = chain.describe_traits(concept)
        opp_str = opposite[0] if opposite else "none"
        print(f"  {concept}: opposite={opp_str}, traits={traits}")
    
    print("\n" + "-" * 60)
    print("FEEDBACK STATS")
    print("-" * 60)
    print(f"  Queries evaluated: {stats.queries_evaluated}")
    print(f"  Corrections made: {stats.corrections_made}")
    print(f"  Frames generated: {stats.synthetic_frames_generated}")
    print(f"  Corrections applied: {stats.corrections_applied}")
    
    return trainer


def test_continuous_training():
    """Test continuous training with feedback."""
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama not running!")
            return
        print("Ollama is running")
    except:
        print("Ollama not available!")
        return
    
    trainer = ContinuousFeedbackTrainer()
    
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    print(f"\nLoading corpus: {corpus_path}")
    count = trainer.load_corpus(str(corpus_path))
    print(f"  Loaded {count} items")
    
    print("\nInitial training...")
    trainer.initial_train()
    
    # Run continuous training
    trainer.run_training(n_cycles=3)
    
    return trainer


if __name__ == "__main__":
    # Run continuous training by default
    trainer = test_continuous_training()
