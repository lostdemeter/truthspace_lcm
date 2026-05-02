#!/usr/bin/env python3
"""
Hint → Geometry Investigation

The hypothesis: Discrete hints ("NEXT: use search") work because they create
specific geometric patterns at the click point (layer 3). If we understand
this conversion, we might not need hints at all.

Questions to answer:
1. What's the geometric signature of "NEXT: use search" vs "NEXT: use generate"?
2. How does the hint change the trajectory through layers?
3. Can we predict the needed hint from the current geometric state?
4. Can we inject the geometry directly without the hint text?
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


@dataclass
class HintSignature:
    """Geometric signature of a hint."""
    hint_text: str
    layer3_embedding: np.ndarray  # The click point
    layer27_embedding: np.ndarray  # The bottleneck
    phi_levels: List[float]  # φ-level at each layer
    trajectory_direction: np.ndarray  # Direction of change from no-hint


class HintGeometryAnalyzer:
    """
    Analyze how hints become geometry.
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
    
    def _get_all_hidden_states(self, text: str) -> List[np.ndarray]:
        """Get hidden states at all layers."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
        
        # Get last token's hidden state at each layer
        states = []
        for h in outputs.hidden_states:
            state = h[0, -1, :].float().cpu().numpy()
            states.append(state)
        
        return states
    
    def _compute_phi_level(self, state: np.ndarray) -> float:
        """Compute mean φ-level of a hidden state."""
        magnitudes = np.abs(state)
        magnitudes = magnitudes[magnitudes > 1e-10]
        phi_levels = np.log(magnitudes) / LOG_PHI
        return float(np.mean(phi_levels))
    
    def analyze_hint(self, base_context: str, hint: str) -> HintSignature:
        """Analyze the geometric signature of a hint."""
        # Get states without hint
        no_hint_states = self._get_all_hidden_states(base_context)
        
        # Get states with hint
        with_hint = base_context + "\n\n" + hint
        hint_states = self._get_all_hidden_states(with_hint)
        
        # Compute φ-levels
        phi_levels = [self._compute_phi_level(s) for s in hint_states]
        
        # Compute trajectory direction (how hint changes the state)
        direction = hint_states[3] - no_hint_states[3]  # Change at layer 3
        
        return HintSignature(
            hint_text=hint,
            layer3_embedding=hint_states[3],
            layer27_embedding=hint_states[27] if len(hint_states) > 27 else hint_states[-1],
            phi_levels=phi_levels,
            trajectory_direction=direction
        )
    
    def compare_hints(self, base_context: str, hints: List[str]) -> Dict:
        """Compare geometric signatures of different hints."""
        signatures = {}
        
        for hint in hints:
            sig = self.analyze_hint(base_context, hint)
            signatures[hint] = sig
            print(f"Analyzed: {hint[:40]}...")
        
        # Compute pairwise similarities
        similarities = {}
        hint_list = list(signatures.keys())
        
        for i, h1 in enumerate(hint_list):
            for h2 in hint_list[i+1:]:
                s1 = signatures[h1]
                s2 = signatures[h2]
                
                # Cosine similarity at layer 3 (click point)
                cos_l3 = np.dot(s1.layer3_embedding, s2.layer3_embedding) / (
                    np.linalg.norm(s1.layer3_embedding) * np.linalg.norm(s2.layer3_embedding) + 1e-10
                )
                
                # Cosine similarity at layer 27 (bottleneck)
                cos_l27 = np.dot(s1.layer27_embedding, s2.layer27_embedding) / (
                    np.linalg.norm(s1.layer27_embedding) * np.linalg.norm(s2.layer27_embedding) + 1e-10
                )
                
                # Direction similarity (how similarly they change the state)
                dir_sim = np.dot(s1.trajectory_direction, s2.trajectory_direction) / (
                    np.linalg.norm(s1.trajectory_direction) * np.linalg.norm(s2.trajectory_direction) + 1e-10
                )
                
                similarities[(h1[:20], h2[:20])] = {
                    'layer3_cos': float(cos_l3),
                    'layer27_cos': float(cos_l27),
                    'direction_cos': float(dir_sim)
                }
        
        return {
            'signatures': signatures,
            'similarities': similarities
        }
    
    def find_hint_directions(self, base_context: str, hints: List[str]) -> Dict[str, np.ndarray]:
        """
        Find the geometric direction each hint pushes the state.
        
        If we can identify these directions, we might be able to:
        1. Detect which direction the state needs to go
        2. Inject that direction without the hint text
        """
        # Get baseline (no hint)
        baseline_states = self._get_all_hidden_states(base_context)
        baseline_l3 = baseline_states[3]
        
        directions = {}
        
        for hint in hints:
            sig = self.analyze_hint(base_context, hint)
            
            # Direction = how the hint changes layer 3
            direction = sig.layer3_embedding - baseline_l3
            
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            
            directions[hint] = direction
            
            print(f"Hint: {hint[:30]}...")
            print(f"  Direction magnitude: {norm:.2f}")
            print(f"  φ-level at L3: {sig.phi_levels[3]:.3f}")
            print(f"  φ-level at L27: {sig.phi_levels[27]:.3f}")
        
        return directions
    
    def predict_needed_hint(self, current_state: List[np.ndarray], 
                           goal_state: List[np.ndarray],
                           hint_directions: Dict[str, np.ndarray]) -> str:
        """
        Predict which hint is needed based on geometric state.
        
        The idea: find which hint direction best aligns with the
        direction from current state to goal state.
        """
        # Direction we need to go (at layer 3)
        needed_direction = goal_state[3] - current_state[3]
        needed_norm = np.linalg.norm(needed_direction)
        
        if needed_norm < 1e-10:
            return "Already at goal"
        
        needed_direction = needed_direction / needed_norm
        
        # Find best matching hint
        best_hint = None
        best_alignment = -1
        
        for hint, direction in hint_directions.items():
            alignment = np.dot(needed_direction, direction)
            if alignment > best_alignment:
                best_alignment = alignment
                best_hint = hint
        
        return best_hint


def analyze_state_needs():
    """
    Instead of predicting hints based on direction to goal,
    analyze what the current state "needs" geometrically.
    
    Hypothesis: The state itself encodes what action is needed.
    We don't need to compare to a goal - the geometry tells us.
    """
    analyzer = HintGeometryAnalyzer()
    
    base_context = """You are completing a goal step by step.

GOAL: Write a summary about the φ-computer proof

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete"""
    
    # Different states
    states = {
        "START": base_context + "\n\nCurrent state: No knowledge gathered yet.",
        "HAS_KNOWLEDGE": base_context + "\n\n[Searched: φ-computer proof]\nFound: Doc 191 - The φ-Computer Proof...\n\nCurrent state: Knowledge gathered.",
        "HAS_OUTPUT": base_context + "\n\n[Searched: φ-computer proof]\n[Created: summary.md]\n\nCurrent state: Output created.",
    }
    
    expected_actions = {
        "START": "search",
        "HAS_KNOWLEDGE": "generate",
        "HAS_OUTPUT": "done",
    }
    
    print("\n" + "=" * 60)
    print("STATE NEEDS ANALYSIS")
    print("=" * 60)
    
    # Get embeddings for each state
    state_embeddings = {}
    for name, context in states.items():
        embeddings = analyzer._get_all_hidden_states(context)
        state_embeddings[name] = {
            'layer3': embeddings[3],
            'layer27': embeddings[27] if len(embeddings) > 27 else embeddings[-1],
            'phi_l3': analyzer._compute_phi_level(embeddings[3]),
            'phi_l27': analyzer._compute_phi_level(embeddings[27] if len(embeddings) > 27 else embeddings[-1]),
        }
        print(f"\n{name}:")
        print(f"  φ-level L3: {state_embeddings[name]['phi_l3']:.3f}")
        print(f"  φ-level L27: {state_embeddings[name]['phi_l27']:.3f}")
        print(f"  Expected action: {expected_actions[name]}")
    
    # Compute differences between states
    print("\n" + "-" * 40)
    print("STATE TRANSITIONS")
    print("-" * 40)
    
    transitions = [
        ("START", "HAS_KNOWLEDGE", "search"),
        ("HAS_KNOWLEDGE", "HAS_OUTPUT", "generate"),
    ]
    
    for from_state, to_state, action in transitions:
        from_emb = state_embeddings[from_state]['layer3']
        to_emb = state_embeddings[to_state]['layer3']
        
        direction = to_emb - from_emb
        magnitude = np.linalg.norm(direction)
        
        print(f"\n{from_state} → {to_state} (action: {action})")
        print(f"  Direction magnitude: {magnitude:.2f}")
        print(f"  φ-level change: {state_embeddings[to_state]['phi_l3'] - state_embeddings[from_state]['phi_l3']:.3f}")
    
    # Key question: Is there a pattern in the state that predicts the needed action?
    print("\n" + "-" * 40)
    print("STATE → ACTION PREDICTION")
    print("-" * 40)
    
    # Get action hint embeddings
    action_hints = {
        "search": "NEXT: Use 'search' to gather knowledge",
        "generate": "NEXT: Use 'generate_and_save' to create output",
        "done": "NEXT: Use 'done' to complete",
    }
    
    action_directions = {}
    for action, hint in action_hints.items():
        sig = analyzer.analyze_hint(base_context, hint)
        baseline = analyzer._get_all_hidden_states(base_context)
        direction = sig.layer3_embedding - baseline[3]
        action_directions[action] = direction / (np.linalg.norm(direction) + 1e-10)
    
    # For each state, which action direction aligns best with its "needs"?
    # The "need" is the direction from current state to the NEXT state
    print("\nPrediction based on transition direction:")
    
    for from_state, to_state, expected_action in transitions:
        from_emb = state_embeddings[from_state]['layer3']
        to_emb = state_embeddings[to_state]['layer3']
        
        needed_direction = to_emb - from_emb
        needed_direction = needed_direction / (np.linalg.norm(needed_direction) + 1e-10)
        
        # Find best matching action
        best_action = None
        best_alignment = -1
        
        for action, direction in action_directions.items():
            alignment = np.dot(needed_direction, direction)
            if alignment > best_alignment:
                best_alignment = alignment
                best_action = action
        
        correct = "✓" if best_action == expected_action else "✗"
        print(f"\n  {from_state}: predicted={best_action}, expected={expected_action} {correct}")
        print(f"    Alignment: {best_alignment:.3f}")
    
    return state_embeddings, action_directions


def run_hint_geometry_analysis():
    """Analyze how different hints create different geometry."""
    analyzer = HintGeometryAnalyzer()
    
    # Base context (a planning scenario)
    base_context = """You are completing a goal step by step.

GOAL: Write a summary about the φ-computer proof

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

Current state: No knowledge gathered yet."""
    
    # Different hints to compare
    hints = [
        "NEXT: Use 'search' to gather knowledge",
        "NEXT: Use 'generate_and_save' to create output",
        "NEXT: Use 'done' to complete",
        "You should search for information first",
        "You should create the output now",
        "You are ready to complete the task",
    ]
    
    print("=" * 60)
    print("HINT GEOMETRY ANALYSIS")
    print("=" * 60)
    
    # Find directions for each hint
    print("\n1. HINT DIRECTIONS")
    print("-" * 40)
    directions = analyzer.find_hint_directions(base_context, hints)
    
    # Compare hints
    print("\n2. HINT SIMILARITIES")
    print("-" * 40)
    results = analyzer.compare_hints(base_context, hints)
    
    print("\nPairwise similarities:")
    for (h1, h2), sims in results['similarities'].items():
        print(f"\n  {h1}... vs {h2}...")
        print(f"    Layer 3 (click): {sims['layer3_cos']:.3f}")
        print(f"    Layer 27 (bottleneck): {sims['layer27_cos']:.3f}")
        print(f"    Direction: {sims['direction_cos']:.3f}")
    
    # Test prediction
    print("\n3. HINT PREDICTION TEST")
    print("-" * 40)
    
    # Simulate different states
    states_to_test = [
        ("No knowledge", base_context),
        ("Has knowledge", base_context + "\n\n[Searched: φ-computer proof]\nFound: Doc 191..."),
        ("Has output", base_context + "\n\n[Created: summary.md]"),
    ]
    
    for state_name, state_context in states_to_test:
        state = analyzer._get_all_hidden_states(state_context)
        
        # Goal state (completed)
        goal_context = base_context + "\n\n[Searched: φ-computer proof]\n[Created: summary.md]\n[DONE]"
        goal_state = analyzer._get_all_hidden_states(goal_context)
        
        predicted = analyzer.predict_needed_hint(state, goal_state, directions)
        print(f"\n  State: {state_name}")
        print(f"  Predicted hint: {predicted[:50]}...")
    
    # Key insight: can we find a "universal" direction for each action?
    print("\n4. ACTION DIRECTIONS")
    print("-" * 40)
    
    # Group hints by action
    search_hints = [h for h in hints if 'search' in h.lower()]
    generate_hints = [h for h in hints if 'generate' in h.lower() or 'create' in h.lower() or 'output' in h.lower()]
    done_hints = [h for h in hints if 'done' in h.lower() or 'complete' in h.lower()]
    
    # Average direction for each action type
    for action_name, action_hints in [("SEARCH", search_hints), ("GENERATE", generate_hints), ("DONE", done_hints)]:
        if action_hints:
            avg_direction = np.mean([directions[h] for h in action_hints], axis=0)
            avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)
            
            # Check consistency
            consistencies = []
            for h in action_hints:
                cos = np.dot(avg_direction, directions[h])
                consistencies.append(cos)
            
            print(f"\n  {action_name}:")
            print(f"    Hints: {len(action_hints)}")
            print(f"    Consistency: {np.mean(consistencies):.3f} ± {np.std(consistencies):.3f}")
    
    return results


if __name__ == "__main__":
    # First run the state needs analysis
    analyze_state_needs()
    
    print("\n\n")
    
    # Then run the full hint geometry analysis
    run_hint_geometry_analysis()
