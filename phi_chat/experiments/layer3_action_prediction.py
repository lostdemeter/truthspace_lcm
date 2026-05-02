#!/usr/bin/env python3
"""
Layer 3 Action Prediction

Can we predict the needed action from layer 3 embeddings alone,
without running the full forward pass?

If yes, this means:
1. We can stop computation at layer 3 for planning decisions
2. The "click point" contains all the information we need
3. Planning becomes O(3 layers) instead of O(28 layers)

Approach:
1. Collect layer 3 embeddings for different states
2. Train a simple classifier (or find geometric patterns)
3. Test if we can predict actions from layer 3 alone
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


@dataclass
class StateExample:
    """A state with its correct action."""
    context: str
    action: str  # 'search', 'generate', 'done'
    layer3_embedding: np.ndarray = None
    phi_level: float = 0.0


class Layer3ActionPredictor:
    """
    Predict actions from layer 3 embeddings.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
        
        self.classifier = None
        self.action_centroids = {}
    
    def _get_layer3_embedding(self, text: str) -> np.ndarray:
        """Get layer 3 embedding for text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
            # Layer 3 is index 3 (0=embedding, 1-28=layers)
            layer3 = outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
        
        return layer3
    
    def _compute_phi_level(self, embedding: np.ndarray) -> float:
        """Compute mean φ-level."""
        magnitudes = np.abs(embedding)
        magnitudes = magnitudes[magnitudes > 1e-10]
        phi_levels = np.log(magnitudes) / LOG_PHI
        return float(np.mean(phi_levels))
    
    def generate_training_data(self) -> List[StateExample]:
        """Generate training examples for different states."""
        
        base_template = """You are completing a goal step by step.

GOAL: {goal}

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

{state_description}

What is your next action?"""
        
        goals = [
            "Write a summary about the φ-computer proof",
            "Explain the transformer disentanglement discovery",
            "Summarize the boom-newton attention findings",
            "Describe the holographic encoding mechanism",
            "Analyze the boom attention speedup",
        ]
        
        examples = []
        
        for goal in goals:
            # START state → search
            context = base_template.format(
                goal=goal,
                state_description="Current state: No knowledge gathered yet."
            )
            examples.append(StateExample(context=context, action="search"))
            
            # HAS_KNOWLEDGE state → generate
            context = base_template.format(
                goal=goal,
                state_description=f"[Searched: {goal}]\nFound: Doc 191 - Relevant information...\n\nCurrent state: Knowledge gathered."
            )
            examples.append(StateExample(context=context, action="generate"))
            
            # HAS_OUTPUT state → done
            context = base_template.format(
                goal=goal,
                state_description=f"[Searched: {goal}]\n[Created: summary.md]\n\nCurrent state: Output created."
            )
            examples.append(StateExample(context=context, action="done"))
        
        # Get embeddings
        print(f"Generating embeddings for {len(examples)} examples...")
        for i, ex in enumerate(examples):
            ex.layer3_embedding = self._get_layer3_embedding(ex.context)
            ex.phi_level = self._compute_phi_level(ex.layer3_embedding)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(examples)}")
        
        return examples
    
    def train_classifier(self, examples: List[StateExample]):
        """Train a simple classifier on layer 3 embeddings."""
        X = np.array([ex.layer3_embedding for ex in examples])
        y = np.array([ex.action for ex in examples])
        
        # Train logistic regression
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X, y)
        
        # Compute centroids for each action
        for action in ['search', 'generate', 'done']:
            action_embeddings = [ex.layer3_embedding for ex in examples if ex.action == action]
            self.action_centroids[action] = np.mean(action_embeddings, axis=0)
        
        # Training accuracy
        y_pred = self.classifier.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\nTraining accuracy: {acc:.2%}")
        print(classification_report(y, y_pred))
        
        return acc
    
    def predict_action(self, context: str) -> Tuple[str, Dict[str, float]]:
        """Predict action from context using layer 3 embedding."""
        embedding = self._get_layer3_embedding(context)
        
        if self.classifier is not None:
            # Use trained classifier
            probs = self.classifier.predict_proba([embedding])[0]
            classes = self.classifier.classes_
            prob_dict = {c: float(p) for c, p in zip(classes, probs)}
            predicted = self.classifier.predict([embedding])[0]
        else:
            # Use centroid distance
            distances = {}
            for action, centroid in self.action_centroids.items():
                dist = np.linalg.norm(embedding - centroid)
                distances[action] = dist
            
            predicted = min(distances, key=distances.get)
            prob_dict = {a: 1.0 / (d + 1e-10) for a, d in distances.items()}
        
        return predicted, prob_dict
    
    def test_on_new_goals(self) -> Dict:
        """Test on goals not seen during training."""
        
        base_template = """You are completing a goal step by step.

GOAL: {goal}

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

{state_description}

What is your next action?"""
        
        # New goals not in training
        test_goals = [
            "Explain the φ-Zipf distribution in attention",
            "Describe the scaffolding vs content distinction",
            "Summarize the autoregression as eigenvalue finding",
        ]
        
        test_cases = []
        
        for goal in test_goals:
            # START → search
            context = base_template.format(
                goal=goal,
                state_description="Current state: No knowledge gathered yet."
            )
            test_cases.append((context, "search", goal, "START"))
            
            # HAS_KNOWLEDGE → generate
            context = base_template.format(
                goal=goal,
                state_description=f"[Searched: {goal}]\nFound: Relevant docs...\n\nCurrent state: Knowledge gathered."
            )
            test_cases.append((context, "generate", goal, "HAS_KNOWLEDGE"))
            
            # HAS_OUTPUT → done
            context = base_template.format(
                goal=goal,
                state_description=f"[Searched: {goal}]\n[Created: output.md]\n\nCurrent state: Output created."
            )
            test_cases.append((context, "done", goal, "HAS_OUTPUT"))
        
        # Test
        correct = 0
        results = []
        
        print("\nTesting on new goals:")
        for context, expected, goal, state in test_cases:
            predicted, probs = self.predict_action(context)
            is_correct = predicted == expected
            correct += int(is_correct)
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {state}: predicted={predicted}, expected={expected}")
            
            results.append({
                'goal': goal[:30],
                'state': state,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'probs': probs
            })
        
        accuracy = correct / len(test_cases)
        print(f"\nTest accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_cases),
            'results': results
        }
    
    def analyze_geometry(self, examples: List[StateExample]):
        """Analyze the geometric structure of action embeddings."""
        
        print("\n" + "=" * 60)
        print("GEOMETRIC ANALYSIS")
        print("=" * 60)
        
        # Group by action
        by_action = {}
        for ex in examples:
            if ex.action not in by_action:
                by_action[ex.action] = []
            by_action[ex.action].append(ex)
        
        # Compute statistics
        print("\n1. φ-LEVEL BY ACTION")
        print("-" * 40)
        for action, exs in by_action.items():
            phi_levels = [ex.phi_level for ex in exs]
            print(f"  {action:10} φ-level: {np.mean(phi_levels):.3f} ± {np.std(phi_levels):.3f}")
        
        # Compute pairwise distances between centroids
        print("\n2. CENTROID DISTANCES")
        print("-" * 40)
        actions = list(self.action_centroids.keys())
        for i, a1 in enumerate(actions):
            for a2 in actions[i+1:]:
                dist = np.linalg.norm(self.action_centroids[a1] - self.action_centroids[a2])
                cos = np.dot(self.action_centroids[a1], self.action_centroids[a2]) / (
                    np.linalg.norm(self.action_centroids[a1]) * np.linalg.norm(self.action_centroids[a2])
                )
                print(f"  {a1} ↔ {a2}: distance={dist:.2f}, cosine={cos:.3f}")
        
        # Within-action variance
        print("\n3. WITHIN-ACTION VARIANCE")
        print("-" * 40)
        for action, exs in by_action.items():
            embeddings = np.array([ex.layer3_embedding for ex in exs])
            centroid = self.action_centroids[action]
            distances = [np.linalg.norm(e - centroid) for e in embeddings]
            print(f"  {action:10} mean_dist_to_centroid: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        
        # Can we separate actions with a simple rule?
        print("\n4. SIMPLE SEPARATION RULES")
        print("-" * 40)
        
        # Try φ-level thresholds
        all_phi = [(ex.phi_level, ex.action) for ex in examples]
        all_phi.sort()
        
        print("  φ-levels by action:")
        for action in ['search', 'generate', 'done']:
            levels = [p for p, a in all_phi if a == action]
            print(f"    {action:10} range: [{min(levels):.3f}, {max(levels):.3f}]")


def run_layer3_experiments():
    """Run layer 3 action prediction experiments."""
    predictor = Layer3ActionPredictor()
    
    # Generate training data
    print("=" * 60)
    print("LAYER 3 ACTION PREDICTION")
    print("=" * 60)
    
    print("\n1. GENERATING TRAINING DATA")
    print("-" * 40)
    examples = predictor.generate_training_data()
    
    # Train classifier
    print("\n2. TRAINING CLASSIFIER")
    print("-" * 40)
    train_acc = predictor.train_classifier(examples)
    
    # Analyze geometry
    predictor.analyze_geometry(examples)
    
    # Test on new goals
    print("\n3. TESTING ON NEW GOALS")
    print("-" * 40)
    test_results = predictor.test_on_new_goals()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Test accuracy: {test_results['accuracy']:.2%}")
    
    if test_results['accuracy'] == 1.0:
        print("\n✓ PERFECT GENERALIZATION!")
        print("  Layer 3 embeddings contain all information needed for action prediction.")
        print("  We can stop at layer 3 for planning decisions.")
    
    return predictor, examples, test_results


def test_complex_scenarios():
    """Test on more complex and varied scenarios."""
    predictor = Layer3ActionPredictor()
    
    # Train on basic examples first
    examples = predictor.generate_training_data()
    predictor.train_classifier(examples)
    
    print("\n" + "=" * 60)
    print("COMPLEX SCENARIO TESTING")
    print("=" * 60)
    
    # More complex scenarios
    scenarios = [
        # Multi-step goals
        {
            "name": "Multi-step research",
            "context": """You are completing a goal step by step.

GOAL: Write a comprehensive analysis comparing φ-computer proof with transformer disentanglement

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

Current state: No knowledge gathered yet.

What is your next action?""",
            "expected": "search"
        },
        {
            "name": "Partial knowledge",
            "context": """You are completing a goal step by step.

GOAL: Explain the connection between boom attention and the safe dial mechanism

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

[Searched: boom attention]
Found: Doc 192 - Boom-Newton Attention...

Current state: Some knowledge gathered, but may need more.

What is your next action?""",
            "expected": "search"  # Might need more info
        },
        {
            "name": "Ready to generate",
            "context": """You are completing a goal step by step.

GOAL: Summarize the key findings about φ-geometry in transformers

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

[Searched: φ-geometry transformers]
Found: Doc 191 - The φ-Computer Proof
Found: Doc 160 - Unified Geometric Theory
Found: Doc 141 - Irreducible Shape

Current state: Comprehensive knowledge gathered from multiple sources.

What is your next action?""",
            "expected": "generate"
        },
        {
            "name": "Output exists but incomplete",
            "context": """You are completing a goal step by step.

GOAL: Write a detailed paper about TruthSpace

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

[Searched: TruthSpace]
[Created: draft.md - 500 words]

Current state: Draft created but may need revision.

What is your next action?""",
            "expected": "done"  # Or could be generate for revision
        },
        {
            "name": "Completely done",
            "context": """You are completing a goal step by step.

GOAL: Create a summary of the holographic encoding

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

[Searched: holographic encoding]
[Created: summary.md - 1500 words]
[Verified: Content is complete and accurate]

Current state: Task fully completed.

What is your next action?""",
            "expected": "done"
        },
    ]
    
    correct = 0
    for scenario in scenarios:
        predicted, probs = predictor.predict_action(scenario["context"])
        is_correct = predicted == scenario["expected"]
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} {scenario['name']}")
        print(f"   Expected: {scenario['expected']}, Predicted: {predicted}")
        print(f"   Probs: {probs}")
    
    print(f"\n\nComplex scenario accuracy: {correct}/{len(scenarios)} ({100*correct/len(scenarios):.0f}%)")
    
    return correct, len(scenarios)


if __name__ == "__main__":
    run_layer3_experiments()
    test_complex_scenarios()
