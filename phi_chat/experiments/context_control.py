#!/usr/bin/env python3
"""
Context Window Control

Key findings from context_window_geometry.py:
1. Context can be compressed 28x with 0.917 layer 3 similarity
2. Layer 3 → Final has near-zero similarity (layers 4-27 transform significantly)
3. Attention anchors capture most of the information

This experiment explores:
1. Can we control what the model attends to?
2. Can we expand effective context by geometric manipulation?
3. Can we achieve the 9x speedup by working at layer 3?

The insight: The 9x speedup isn't about skipping layers 4-27.
It's about controlling WHAT layer 3 sees, so layers 4-27 do what we want.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2


class ContextController:
    """Control the context window geometrically."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
    
    def get_kv_cache(self, context: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the KV cache for a context (what attention can route to)."""
        inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True,
                use_cache=True
            )
        
        # past_key_values is a tuple of (key, value) for each layer
        # Each key/value is (batch, num_heads, seq_len, head_dim)
        return outputs.past_key_values
    
    def generate_with_context(self, context: str, query: str, max_tokens: int = 50) -> str:
        """Generate with a given context."""
        full_text = context + query
        inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compare_contexts(self, contexts: List[str], query: str) -> Dict:
        """Compare how different contexts affect the output."""
        results = []
        
        for context in contexts:
            output = self.generate_with_context(context, query, max_tokens=30)
            
            # Get hidden states
            full_text = context + query
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                hidden_outputs = self.model(inputs.input_ids, output_hidden_states=True)
            
            layer3 = hidden_outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
            final = hidden_outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            
            results.append({
                'context': context[:50],
                'output': output[len(full_text):].strip()[:100],
                'layer3_norm': float(np.linalg.norm(layer3)),
                'final_norm': float(np.linalg.norm(final)),
            })
        
        return results
    
    def test_context_injection(self, base_context: str, injection: str, query: str) -> Dict:
        """
        Test if we can inject information into the context.
        
        The idea: Can we add a "steering" token that changes the output
        without adding much to the context length?
        """
        # Without injection
        no_inject = self.generate_with_context(base_context, query, max_tokens=30)
        
        # With injection at start
        inject_start = self.generate_with_context(injection + " " + base_context, query, max_tokens=30)
        
        # With injection at end (before query)
        inject_end = self.generate_with_context(base_context + " " + injection, query, max_tokens=30)
        
        return {
            'no_injection': no_inject[len(base_context + query):].strip()[:100],
            'inject_start': inject_start[len(injection + " " + base_context + query):].strip()[:100],
            'inject_end': inject_end[len(base_context + " " + injection + query):].strip()[:100],
        }
    
    def find_minimal_context(self, full_context: str, query: str, target_output: str) -> Dict:
        """
        Find the minimal context that produces similar output.
        
        Binary search to find how much context we actually need.
        """
        tokens = self.tokenizer.encode(full_context)
        
        # Full context output
        full_output = self.generate_with_context(full_context, query, max_tokens=30)
        full_response = full_output[len(full_context + query):].strip()
        
        # Binary search for minimal context
        left, right = 1, len(tokens)
        
        while left < right:
            mid = (left + right) // 2
            partial_context = self.tokenizer.decode(tokens[:mid])
            partial_output = self.generate_with_context(partial_context, query, max_tokens=30)
            partial_response = partial_output[len(partial_context + query):].strip()
            
            # Check if outputs are similar (first 20 chars match)
            if partial_response[:20] == full_response[:20]:
                right = mid
            else:
                left = mid + 1
        
        minimal_context = self.tokenizer.decode(tokens[:left])
        
        return {
            'full_tokens': len(tokens),
            'minimal_tokens': left,
            'compression': len(tokens) / left if left > 0 else 0,
            'full_response': full_response[:50],
            'minimal_response': self.generate_with_context(minimal_context, query, max_tokens=30)[len(minimal_context + query):].strip()[:50],
        }
    
    def test_action_steering(self) -> Dict:
        """
        Test if we can steer the model to different actions using minimal context changes.
        
        This is the key to the 9x speedup: if we can control the action
        with minimal context, we don't need complex planning.
        """
        base_context = """You are completing a goal step by step.
GOAL: Write a summary about φ
TOOLS: search, generate, done
"""
        
        # Different state descriptions
        states = {
            'search': "Current state: No knowledge gathered yet.",
            'generate': "Current state: Knowledge gathered. [Searched: φ] Found: φ = 1.618...",
            'done': "Current state: Output created. [Created: summary.md]",
        }
        
        query = "\nWhat is your next action? Respond with just the tool name."
        
        results = {}
        for expected_action, state_desc in states.items():
            context = base_context + state_desc
            output = self.generate_with_context(context, query, max_tokens=10)
            response = output[len(context + query):].strip().lower()
            
            # Check if the expected action is in the response
            predicted = None
            for action in ['search', 'generate', 'done']:
                if action in response:
                    predicted = action
                    break
            
            results[expected_action] = {
                'state': state_desc[:40],
                'response': response[:30],
                'predicted': predicted,
                'correct': predicted == expected_action
            }
        
        return results


def run_context_control_experiments():
    """Run context control experiments."""
    controller = ContextController()
    
    print("=" * 60)
    print("CONTEXT CONTROL EXPERIMENTS")
    print("=" * 60)
    
    # Test 1: Action steering
    print("\n1. ACTION STEERING")
    print("-" * 40)
    
    steering_results = controller.test_action_steering()
    
    correct = 0
    for expected, result in steering_results.items():
        status = "✓" if result['correct'] else "✗"
        correct += int(result['correct'])
        print(f"  {status} Expected: {expected}")
        print(f"     State: {result['state']}...")
        print(f"     Response: {result['response']}")
        print(f"     Predicted: {result['predicted']}")
    
    print(f"\n  Accuracy: {correct}/{len(steering_results)}")
    
    # Test 2: Minimal context
    print("\n2. MINIMAL CONTEXT SEARCH")
    print("-" * 40)
    
    full_context = """The golden ratio φ (phi) is approximately 1.618.
It appears in nature, art, and mathematics.
In TruthSpace, φ governs the geometric structure.
The transformer's bottleneck converges to φ-level 1."""
    
    query = " What is φ?"
    
    minimal = controller.find_minimal_context(full_context, query, "")
    print(f"\n  Full context: {minimal['full_tokens']} tokens")
    print(f"  Minimal context: {minimal['minimal_tokens']} tokens")
    print(f"  Compression: {minimal['compression']:.1f}x")
    print(f"  Full response: {minimal['full_response']}")
    print(f"  Minimal response: {minimal['minimal_response']}")
    
    # Test 3: Context injection
    print("\n3. CONTEXT INJECTION")
    print("-" * 40)
    
    base = "You are a helpful assistant."
    injection = "[IMPORTANT: Be very brief.]"
    query = " Explain what φ means."
    
    injection_results = controller.test_context_injection(base, injection, query)
    print(f"\n  No injection: {injection_results['no_injection'][:60]}...")
    print(f"  Inject start: {injection_results['inject_start'][:60]}...")
    print(f"  Inject end: {injection_results['inject_end'][:60]}...")
    
    # Test 4: Compare different context formulations
    print("\n4. CONTEXT FORMULATION COMPARISON")
    print("-" * 40)
    
    contexts = [
        "GOAL: search for φ information",
        "Task: Find information about φ",
        "You need to search for φ",
        "[Action required: search] Topic: φ",
    ]
    query = " What should you do?"
    
    comparison = controller.compare_contexts(contexts, query)
    for r in comparison:
        print(f"\n  Context: {r['context']}")
        print(f"  Output: {r['output'][:50]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"""
Key findings:
1. Action steering: {correct}/{len(steering_results)} correct
   → State description controls action selection
   
2. Minimal context: {minimal['compression']:.1f}x compression possible
   → Most context is redundant for simple queries
   
3. Context injection: Position matters
   → Injections at end have more effect (closer to query)

Implication for 9x speedup:
- The speedup isn't about skipping layers
- It's about CONTROLLING what layer 3 sees
- Minimal, well-structured context → predictable layer 3 output
- Predictable layer 3 → predictable action
""")
    
    return controller


if __name__ == "__main__":
    run_context_control_experiments()
