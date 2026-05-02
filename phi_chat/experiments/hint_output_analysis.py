#!/usr/bin/env python3
"""
Hint → Output Analysis

The previous experiment showed that hints don't move the state toward the goal.
Instead, they might change how the model interprets the state.

Hypothesis: Hints change the OUTPUT DISTRIBUTION, not the state position.
The hint "NEXT: use search" makes "search" more likely in the output,
without necessarily moving the hidden state toward a "search state".

This is like the safe dial: the hint rotates the "plates" (context),
which changes what "clicks" when the model generates output.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = (1 + np.sqrt(5)) / 2


class HintOutputAnalyzer:
    """Analyze how hints change output probabilities."""
    
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
        
        # Get token IDs for key words
        self.action_tokens = {
            'search': self.tokenizer.encode('search', add_special_tokens=False),
            'generate': self.tokenizer.encode('generate', add_special_tokens=False),
            'done': self.tokenizer.encode('done', add_special_tokens=False),
            'TOOL': self.tokenizer.encode('TOOL', add_special_tokens=False),
        }
        print(f"Action tokens: {self.action_tokens}")
    
    def get_next_token_probs(self, text: str) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
        """Get probability distribution over next token."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0, -1, :]  # Last token's logits
            probs = F.softmax(logits, dim=-1)
        
        # Get top tokens
        top_k = 20
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_tokens = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx])
            top_tokens.append((token, float(prob)))
        
        return probs, top_tokens
    
    def get_action_probs(self, text: str) -> Dict[str, float]:
        """Get probabilities for action-related tokens."""
        probs, _ = self.get_next_token_probs(text)
        
        action_probs = {}
        for action, token_ids in self.action_tokens.items():
            # Sum probabilities for all tokens in the action
            total_prob = sum(float(probs[tid]) for tid in token_ids)
            action_probs[action] = total_prob
        
        return action_probs
    
    def compare_with_without_hint(self, base_context: str, hint: str) -> Dict:
        """Compare output probabilities with and without a hint."""
        # Without hint
        no_hint_probs, no_hint_top = self.get_next_token_probs(base_context)
        no_hint_actions = self.get_action_probs(base_context)
        
        # With hint
        with_hint = base_context + "\n\n" + hint
        hint_probs, hint_top = self.get_next_token_probs(with_hint)
        hint_actions = self.get_action_probs(with_hint)
        
        # Compute changes
        changes = {}
        for action in self.action_tokens:
            before = no_hint_actions[action]
            after = hint_actions[action]
            ratio = after / (before + 1e-10)
            changes[action] = {
                'before': before,
                'after': after,
                'ratio': ratio,
                'log_ratio': np.log(ratio + 1e-10)
            }
        
        return {
            'no_hint_top': no_hint_top[:10],
            'hint_top': hint_top[:10],
            'action_changes': changes
        }


def run_output_analysis():
    """Analyze how hints change output probabilities."""
    analyzer = HintOutputAnalyzer()
    
    base_context = """You are completing a goal step by step.

GOAL: Write a summary about the φ-computer proof

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

Use: TOOL: {"tool": "name", "param": "value"}

Current state: No knowledge gathered yet.

What is your next action?"""
    
    hints = [
        "NEXT: Use 'search' to gather knowledge",
        "NEXT: Use 'generate_and_save' to create output",
        "NEXT: Use 'done' to complete",
    ]
    
    print("=" * 60)
    print("HINT → OUTPUT ANALYSIS")
    print("=" * 60)
    
    # Baseline (no hint)
    print("\n1. BASELINE (no hint)")
    print("-" * 40)
    _, baseline_top = analyzer.get_next_token_probs(base_context)
    baseline_actions = analyzer.get_action_probs(base_context)
    
    print("Top tokens:")
    for token, prob in baseline_top[:10]:
        print(f"  {repr(token):15} {prob:.4f}")
    
    print("\nAction probabilities:")
    for action, prob in baseline_actions.items():
        print(f"  {action:10} {prob:.6f}")
    
    # With each hint
    print("\n2. WITH HINTS")
    print("-" * 40)
    
    for hint in hints:
        print(f"\nHint: {hint}")
        result = analyzer.compare_with_without_hint(base_context, hint)
        
        print("  Top tokens after hint:")
        for token, prob in result['hint_top'][:5]:
            print(f"    {repr(token):15} {prob:.4f}")
        
        print("  Action probability changes:")
        for action, changes in result['action_changes'].items():
            ratio = changes['ratio']
            direction = "↑" if ratio > 1 else ("↓" if ratio < 1 else "→")
            print(f"    {action:10} {changes['before']:.6f} → {changes['after']:.6f} ({ratio:.2f}x) {direction}")
    
    # Key test: Does the hint for "search" increase "search" probability?
    print("\n3. HINT EFFECTIVENESS")
    print("-" * 40)
    
    for hint in hints:
        # Extract the action from the hint
        if 'search' in hint.lower():
            target_action = 'search'
        elif 'generate' in hint.lower():
            target_action = 'generate'
        elif 'done' in hint.lower():
            target_action = 'done'
        else:
            continue
        
        result = analyzer.compare_with_without_hint(base_context, hint)
        target_change = result['action_changes'][target_action]
        
        effective = "✓" if target_change['ratio'] > 1.5 else "✗"
        print(f"\n  Hint for '{target_action}': {effective}")
        print(f"    Probability change: {target_change['before']:.6f} → {target_change['after']:.6f}")
        print(f"    Ratio: {target_change['ratio']:.2f}x")
    
    return analyzer


def analyze_state_dependent_hints():
    """
    Test if hints work differently depending on the current state.
    
    Hypothesis: The same hint might have different effects depending
    on what state we're in. The geometry of the state interacts with
    the geometry of the hint.
    """
    analyzer = HintOutputAnalyzer()
    
    base = """You are completing a goal step by step.

GOAL: Write a summary about the φ-computer proof

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete

Use: TOOL: {"tool": "name", "param": "value"}"""
    
    states = {
        "START": base + "\n\nCurrent state: No knowledge gathered yet.\n\nWhat is your next action?",
        "HAS_KNOWLEDGE": base + "\n\n[Searched: φ-computer proof]\nFound: Doc 191...\n\nCurrent state: Knowledge gathered.\n\nWhat is your next action?",
        "HAS_OUTPUT": base + "\n\n[Searched: φ-computer proof]\n[Created: summary.md]\n\nCurrent state: Output created.\n\nWhat is your next action?",
    }
    
    print("\n" + "=" * 60)
    print("STATE-DEPENDENT HINT ANALYSIS")
    print("=" * 60)
    
    for state_name, state_context in states.items():
        print(f"\n{state_name}")
        print("-" * 40)
        
        # Get baseline for this state
        baseline_actions = analyzer.get_action_probs(state_context)
        print("Baseline action probs:")
        for action, prob in baseline_actions.items():
            print(f"  {action:10} {prob:.6f}")
        
        # Which action is most likely without any hint?
        most_likely = max(baseline_actions, key=baseline_actions.get)
        print(f"Most likely action: {most_likely}")


if __name__ == "__main__":
    run_output_analysis()
    analyze_state_dependent_hints()
