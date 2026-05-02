#!/usr/bin/env python3
"""
Self-Improving Planner

Can the planner learn from its experiences to improve action prediction?

Approach:
1. Start with a basic layer 3 classifier
2. Run planning tasks and record outcomes
3. When an action leads to success, reinforce that pattern
4. When an action leads to failure, adjust the classifier
5. Track improvement over time

This connects to the Self-Improving TruthSpace (memory from Dec 29, 2024):
- ANALYZE: What do we have? What's missing?
- PREDICT: Where should new concepts/transforms be?
- SEARCH: Look for evidence of predictions
- VERIFY: Use probe extraction to confirm
- INTEGRATE: Add verified discoveries
- REPEAT
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from concept_search import ConceptSearcher

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Experience:
    """A single planning experience."""
    context: str
    action_taken: str
    layer3_embedding: np.ndarray
    outcome: str  # 'success', 'failure', 'partial'
    next_state: Optional[str] = None


@dataclass
class PlannerMemory:
    """Memory of past experiences."""
    experiences: List[Experience] = field(default_factory=list)
    action_success_rates: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def record(self, exp: Experience):
        self.experiences.append(exp)
        
        # Track success rates per action
        if exp.action_taken not in self.action_success_rates:
            self.action_success_rates[exp.action_taken] = {'success': 0, 'failure': 0, 'partial': 0}
        self.action_success_rates[exp.action_taken][exp.outcome] += 1
    
    def get_success_rate(self, action: str) -> float:
        if action not in self.action_success_rates:
            return 0.5  # Unknown
        stats = self.action_success_rates[action]
        total = sum(stats.values())
        if total == 0:
            return 0.5
        return stats['success'] / total


class SelfImprovingPlanner:
    """
    Planner that learns from experience.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading Self-Improving Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.searcher = ConceptSearcher()
        self.device = next(self.model.parameters()).device
        
        self.memory = PlannerMemory()
        self.classifier = None
        self.training_data = []  # (embedding, action, weight)
        
        print("✓ Planner loaded!\n")
    
    def _get_layer3_embedding(self, text: str) -> np.ndarray:
        """Get layer 3 embedding."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
            layer3 = outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
        
        return layer3
    
    def _predict_action(self, context: str) -> Tuple[str, Dict[str, float]]:
        """Predict action using current classifier."""
        embedding = self._get_layer3_embedding(context)
        
        if self.classifier is None:
            # No classifier yet - use heuristics
            if "[Created:" in context:
                return "done", {"done": 0.8, "generate": 0.1, "search": 0.1}
            elif "[Searched:" in context or "[Found:" in context:
                return "generate", {"generate": 0.7, "search": 0.2, "done": 0.1}
            else:
                return "search", {"search": 0.8, "generate": 0.1, "done": 0.1}
        
        probs = self.classifier.predict_proba([embedding])[0]
        classes = self.classifier.classes_
        prob_dict = {c: float(p) for c, p in zip(classes, probs)}
        predicted = self.classifier.predict([embedding])[0]
        
        return predicted, prob_dict
    
    def _execute_action(self, action: str, goal: str, knowledge: List[str], 
                       artifacts: Dict[str, str]) -> Tuple[str, List[str], Dict[str, str]]:
        """Execute an action and return outcome."""
        
        if action == "search":
            results = self.searcher.search(goal, max_results=3)
            if results:
                for r in results:
                    if r.excerpts:
                        knowledge.append(r.excerpts[0][:200])
                return "success", knowledge, artifacts
            return "failure", knowledge, artifacts
        
        elif action == "generate":
            if not knowledge:
                return "failure", knowledge, artifacts
            
            # Generate content
            content = f"# {goal}\n\n" + "\n\n".join(knowledge[:3])
            artifacts["output.md"] = content
            return "success", knowledge, artifacts
        
        elif action == "done":
            if artifacts:
                return "success", knowledge, artifacts
            return "failure", knowledge, artifacts
        
        return "failure", knowledge, artifacts
    
    def _update_classifier(self):
        """Update classifier based on experiences."""
        if len(self.memory.experiences) < 5:
            return  # Not enough data
        
        # Build training data from experiences
        X = []
        y = []
        weights = []
        
        for exp in self.memory.experiences:
            X.append(exp.layer3_embedding)
            y.append(exp.action_taken)
            # Weight by outcome
            if exp.outcome == "success":
                weights.append(1.0)
            elif exp.outcome == "partial":
                weights.append(0.5)
            else:
                weights.append(0.1)  # Still learn from failures
        
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        
        # Train classifier
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X, y, sample_weight=weights)
    
    def solve(self, goal: str, max_steps: int = 5) -> Dict:
        """Solve a goal while learning from experience."""
        print(f"🎯 Goal: {goal}")
        
        knowledge = []
        artifacts = {}
        actions_taken = []
        success = False
        
        base_context = f"""You are completing a goal step by step.

GOAL: {goal}

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete"""
        
        for step in range(max_steps):
            # Build current context
            context = base_context
            if knowledge:
                context += f"\n\n[Searched: {goal}]\nFound: {len(knowledge)} excerpts"
            if artifacts:
                context += f"\n\n[Created: {list(artifacts.keys())[0]}]"
            context += "\n\nWhat is your next action?"
            
            # Get layer 3 embedding
            embedding = self._get_layer3_embedding(context)
            
            # Predict action
            action, probs = self._predict_action(context)
            actions_taken.append(action)
            
            print(f"  Step {step+1}: {action} (probs: {probs})")
            
            # Execute action
            outcome, knowledge, artifacts = self._execute_action(
                action, goal, knowledge, artifacts
            )
            
            # Record experience
            exp = Experience(
                context=context,
                action_taken=action,
                layer3_embedding=embedding,
                outcome=outcome
            )
            self.memory.record(exp)
            
            # Check if done
            if action == "done" and outcome == "success":
                success = True
                print(f"  ✓ Goal achieved!")
                break
            
            # Update classifier periodically
            if len(self.memory.experiences) % 3 == 0:
                self._update_classifier()
        
        return {
            "success": success,
            "steps": len(actions_taken),
            "actions": actions_taken,
            "artifacts": list(artifacts.keys())
        }
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about learning progress."""
        return {
            "total_experiences": len(self.memory.experiences),
            "action_success_rates": {
                action: self.memory.get_success_rate(action)
                for action in self.memory.action_success_rates
            },
            "classifier_trained": self.classifier is not None
        }


def run_self_improvement_experiment():
    """Run the self-improvement experiment."""
    planner = SelfImprovingPlanner()
    
    # Goals to solve - mix of easy and harder
    goals = [
        # Round 1: Basic goals
        "Write a summary about the φ-computer proof",
        "Explain the transformer disentanglement",
        "Summarize boom-newton attention",
        
        # Round 2: More goals (should improve)
        "Describe the holographic encoding mechanism",
        "Explain the safe dial mechanism",
        "Summarize the scaffolding vs content finding",
        
        # Round 3: Even more (should be better)
        "Analyze the φ-Zipf distribution",
        "Describe the autoregression eigenvalue discovery",
        "Explain the bulge geodesic deviation",
    ]
    
    print("=" * 60)
    print("SELF-IMPROVEMENT EXPERIMENT")
    print("=" * 60)
    
    results_by_round = {1: [], 2: [], 3: []}
    
    for i, goal in enumerate(goals):
        round_num = (i // 3) + 1
        print(f"\n--- Round {round_num}, Goal {(i % 3) + 1} ---")
        
        result = planner.solve(goal)
        results_by_round[round_num].append(result)
        
        # Show learning stats
        stats = planner.get_learning_stats()
        print(f"  Learning: {stats['total_experiences']} experiences, classifier={stats['classifier_trained']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)
    
    for round_num, results in results_by_round.items():
        successes = sum(1 for r in results if r["success"])
        avg_steps = np.mean([r["steps"] for r in results])
        print(f"\nRound {round_num}:")
        print(f"  Success rate: {successes}/{len(results)}")
        print(f"  Avg steps: {avg_steps:.1f}")
    
    # Final stats
    final_stats = planner.get_learning_stats()
    print(f"\nFinal learning stats:")
    print(f"  Total experiences: {final_stats['total_experiences']}")
    print(f"  Action success rates: {final_stats['action_success_rates']}")
    
    # Did we improve?
    r1_success = sum(1 for r in results_by_round[1] if r["success"])
    r3_success = sum(1 for r in results_by_round[3] if r["success"])
    
    if r3_success >= r1_success:
        print("\n✓ Performance maintained or improved through learning!")
    else:
        print("\n✗ Performance degraded - need to investigate")
    
    return planner, results_by_round


def run_adversarial_experiment():
    """Test with adversarial/ambiguous scenarios where learning matters."""
    planner = SelfImprovingPlanner()
    
    print("=" * 60)
    print("ADVERSARIAL LEARNING EXPERIMENT")
    print("=" * 60)
    
    # Scenarios designed to be ambiguous
    scenarios = [
        # Should search but context looks like it has knowledge
        ("Explain quantum computing basics", 
         "[Note: This is a new topic not in our docs]", 
         "search"),
        
        # Should generate but minimal knowledge
        ("Write one sentence about φ",
         "[Searched: φ]\nFound: φ = 1.618...",
         "generate"),
        
        # Should be done but output is minimal
        ("Create a title",
         "[Created: title.md - 10 chars]",
         "done"),
    ]
    
    # Run multiple rounds to see learning
    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        
        for goal, state_hint, expected in scenarios:
            # Build context with the hint
            context = f"""You are completing a goal step by step.

GOAL: {goal}

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

{state_hint}

What is your next action?"""
            
            predicted, probs = planner._predict_action(context)
            correct = predicted == expected
            
            # Record as experience
            embedding = planner._get_layer3_embedding(context)
            outcome = "success" if correct else "failure"
            
            exp = Experience(
                context=context,
                action_taken=predicted,
                layer3_embedding=embedding,
                outcome=outcome
            )
            planner.memory.record(exp)
            
            status = "✓" if correct else "✗"
            print(f"  {status} {goal[:30]}... → {predicted} (expected {expected})")
        
        # Update classifier after each round
        planner._update_classifier()
        print(f"  Classifier updated with {len(planner.memory.experiences)} experiences")
    
    # Final stats
    stats = planner.get_learning_stats()
    print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    run_self_improvement_experiment()
    print("\n\n")
    run_adversarial_experiment()
