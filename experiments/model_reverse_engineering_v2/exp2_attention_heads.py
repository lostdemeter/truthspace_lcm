#!/usr/bin/env python3
"""
Experiment 2: Attention Head Classification

Qwen2-7B has 28 attention heads per layer (GQA: 28 Q heads, 4 KV heads).
Each head operates on a 128-dim subspace (3584 / 28 = 128).

This experiment:
1. Hooks into each attention layer to capture per-head input/output
2. Runs ContinuousPhaseDiscovery on each head's transformation
3. Classifies heads by archetype
4. Looks for patterns: do heads within a layer share structure?
   Do heads at the same position across layers share structure?

Focus on the peak geometric layers (4-6) plus transition (3) and
control layers (0, 14, 27).
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phi_geometric.core.continuous_discovery import (
    ContinuousPhaseDiscovery,
    to_phi_levels,
)

PHI = (1 + np.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# Attention head extraction via hooks
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Capture per-head attention inputs and outputs."""
    
    def __init__(self, model, target_layers: List[int]):
        self.target_layers = target_layers
        self.captures = {}  # (layer, 'pre'|'post') → list of tensors
        self.hooks = []
        
        # Register hooks on attention modules
        for layer_idx in target_layers:
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            
            # Hook: capture input to attention (pre) and output (post)
            pre_hook = attn.register_forward_pre_hook(
                self._make_pre_hook(layer_idx),
                with_kwargs=True,
            )
            post_hook = attn.register_forward_hook(
                self._make_post_hook(layer_idx)
            )
            self.hooks.extend([pre_hook, post_hook])
    
    def _make_pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            # hidden_states can be positional or keyword
            if args:
                hs = args[0]
            elif 'hidden_states' in kwargs:
                hs = kwargs['hidden_states']
            else:
                return
            key = (layer_idx, 'pre')
            if key not in self.captures:
                self.captures[key] = []
            self.captures[key].append(hs.detach().cpu().float())
        return hook
    
    def _make_post_hook(self, layer_idx):
        def hook(module, args, output):
            # output is a tuple; first element is attention output
            attn_out = output[0] if isinstance(output, tuple) else output
            key = (layer_idx, 'post')
            if key not in self.captures:
                self.captures[key] = []
            self.captures[key].append(attn_out.detach().cpu().float())
        return hook
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_pairs(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (input, output) arrays for a layer's attention.
        Returns (N_tokens, hidden_dim) arrays.
        """
        pre_list = self.captures.get((layer_idx, 'pre'), [])
        post_list = self.captures.get((layer_idx, 'post'), [])
        
        if not pre_list or not post_list:
            return np.array([]), np.array([])
        
        # Concat across prompts: each is (1, seq_len, hidden_dim)
        pre = torch.cat(pre_list, dim=1)[0].numpy()   # (total_tokens, hidden_dim)
        post = torch.cat(post_list, dim=1)[0].numpy()
        
        return pre, post


# ---------------------------------------------------------------------------
# Per-head analysis
# ---------------------------------------------------------------------------

def split_into_heads(data: np.ndarray, num_heads: int = 28) -> List[np.ndarray]:
    """Split (N, hidden_dim) into per-head (N, head_dim) arrays."""
    # Qwen2-7B: hidden_dim=3584, num_heads=28, head_dim=128
    N, D = data.shape
    head_dim = D // num_heads
    heads = []
    for h in range(num_heads):
        heads.append(data[:, h * head_dim : (h + 1) * head_dim])
    return heads


def analyze_head(
    head_in: np.ndarray,
    head_out: np.ndarray,
    head_idx: int,
    layer_idx: int,
    phi_scale: int = 64,
    context_radius: int = 2,
) -> Dict:
    """Run ContinuousPhaseDiscovery on a single attention head."""
    N, D = head_in.shape
    
    # Subsample tokens for speed
    n_sample = min(N, 80)
    indices = np.random.choice(N, n_sample, replace=False)
    
    cpd = ContinuousPhaseDiscovery(
        phi_scale=phi_scale,
        context_radius=context_radius,
        identity_threshold=1.0,
        affine_threshold=0.7,
    )
    
    for i in indices:
        cpd.add_pair(head_in[i], head_out[i])
    
    result = cpd.discover()
    
    rd = result.rule_distribution
    total = sum(rd.values())
    
    return {
        'layer': layer_idx,
        'head': head_idx,
        'archetype': result.archetype,
        'r_squared': float(result.mean_r_squared),
        'identity_pct': rd.get('identity', 0) / total,
        'scale_pct': rd.get('scale', 0) / total,
        'affine_pct': rd.get('affine', 0) / total,
        'context_pct': rd.get('context', 0) / total,
        'collapse_pct': rd.get('collapse', 0) / total,
        'unstructured_pct': rd.get('unstructured', 0) / total,
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "I went to the store and",
    "She said that she would",
    "The book is on the",
    "The capital of France is",
    "The largest planet is",
    "Water boils at",
    "Albert Einstein developed the",
    "The speed of light is",
    "In the beginning there was",
    "Once upon a time in a",
    "The quick brown fox jumps",
    "To be or not to be",
    "All that glitters is not",
    "The meaning of life is",
    "A journey of a thousand miles",
]

# Layers to analyze
TARGET_LAYERS = [0, 2, 3, 4, 5, 6, 9, 11, 14, 20, 25, 27]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    print("=" * 70)
    print("Experiment 2: Attention Head Classification")
    print("=" * 70)
    
    model_name = "Qwen/Qwen2-7B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    
    # Get model config
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    print(f"  Q heads: {num_heads}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  Target layers: {TARGET_LAYERS}")
    
    # Set up hooks
    capture = AttentionCapture(model, TARGET_LAYERS)
    
    # Run prompts
    print(f"\nRunning {len(PROMPTS)} prompts...")
    for idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        if (idx + 1) % 5 == 0:
            print(f"  {idx + 1}/{len(PROMPTS)}")
    
    capture.remove_hooks()
    
    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Analyze each head at each layer
    print(f"\n{'='*70}")
    print("PER-HEAD ANALYSIS")
    print(f"{'='*70}")
    
    all_results = []
    
    for layer_idx in TARGET_LAYERS:
        pre, post = capture.get_pairs(layer_idx)
        
        if pre.size == 0:
            print(f"\n  Layer {layer_idx}: No data captured")
            continue
        
        # Split into heads
        heads_in = split_into_heads(pre, num_heads)
        heads_out = split_into_heads(post, num_heads)
        
        zone = ""
        if layer_idx <= 2:
            zone = "DRUM"
        elif layer_idx == 3:
            zone = "TRANSITION"
        elif layer_idx <= 6:
            zone = "COMB-early"
        elif layer_idx <= 25:
            zone = "COMB-late"
        else:
            zone = "MUSIC"
        
        print(f"\n--- Layer {layer_idx} ({zone}) ---")
        print(f"  {'Head':>4} {'Archetype':<20} {'R²':>6} "
              f"{'id%':>5} {'scl%':>5} {'aff%':>5} {'ctx%':>5} {'unk%':>5}")
        print(f"  {'-'*65}")
        
        layer_results = []
        for h in range(num_heads):
            result = analyze_head(
                heads_in[h], heads_out[h],
                head_idx=h, layer_idx=layer_idx,
                phi_scale=64, context_radius=2,
            )
            layer_results.append(result)
            all_results.append(result)
            
            print(f"  {h:>4} {result['archetype']:<20} {result['r_squared']:>6.3f} "
                  f"{result['identity_pct']:>5.0%} {result['scale_pct']:>5.0%} "
                  f"{result['affine_pct']:>5.0%} {result['context_pct']:>5.0%} "
                  f"{result['unstructured_pct']:>5.0%}")
        
        # Layer summary
        mean_r2 = np.mean([r['r_squared'] for r in layer_results])
        mean_affine = np.mean([r['affine_pct'] for r in layer_results])
        arch_counts = Counter(r['archetype'] for r in layer_results)
        
        print(f"\n  Layer {layer_idx} summary:")
        print(f"    Mean R²: {mean_r2:.3f}")
        print(f"    Mean affine%: {mean_affine:.0%}")
        print(f"    Archetypes: {dict(arch_counts)}")
    
    # Cross-layer analysis
    print(f"\n{'='*70}")
    print("CROSS-LAYER SUMMARY")
    print(f"{'='*70}")
    
    # Per-layer averages
    by_layer = defaultdict(list)
    for r in all_results:
        by_layer[r['layer']].append(r)
    
    print(f"\n{'Layer':>5} {'Zone':<12} {'Mean R²':>8} {'Mean aff%':>10} {'Dominant arch':<20}")
    print("-" * 60)
    
    for layer_idx in sorted(by_layer.keys()):
        results = by_layer[layer_idx]
        mean_r2 = np.mean([r['r_squared'] for r in results])
        mean_aff = np.mean([r['affine_pct'] for r in results])
        archs = Counter(r['archetype'] for r in results)
        dominant = archs.most_common(1)[0][0]
        
        zone = ""
        if layer_idx <= 2: zone = "DRUM"
        elif layer_idx == 3: zone = "TRANSITION"
        elif layer_idx <= 6: zone = "COMB-early"
        elif layer_idx <= 25: zone = "COMB-late"
        else: zone = "MUSIC"
        
        print(f"{layer_idx:>5} {zone:<12} {mean_r2:>8.3f} {mean_aff:>10.0%} {dominant:<20}")
    
    # Head position analysis: do heads at same position share structure?
    print(f"\nPer-head-position analysis (across layers):")
    by_head = defaultdict(list)
    for r in all_results:
        by_head[r['head']].append(r)
    
    print(f"\n{'Head':>4} {'Mean R²':>8} {'Mean aff%':>10} {'Archetypes'}")
    print("-" * 60)
    for h in sorted(by_head.keys()):
        results = by_head[h]
        mean_r2 = np.mean([r['r_squared'] for r in results])
        mean_aff = np.mean([r['affine_pct'] for r in results])
        archs = Counter(r['archetype'] for r in results)
        print(f"{h:>4} {mean_r2:>8.3f} {mean_aff:>10.0%} {dict(archs)}")
    
    # Save
    results_file = output_dir / "exp2_attention_heads.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_file}")


if __name__ == "__main__":
    main()
