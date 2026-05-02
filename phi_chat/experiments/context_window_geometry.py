#!/usr/bin/env python3
"""
Context Window Geometry

What is the context window geometrically?
- It's the set of K,V vectors that attention can route to
- The "window" is the span of positions the model can attend to
- Geometrically: it's a subspace defined by the V vectors

Questions:
1. What is the effective dimensionality of the context window?
2. How does attention distribute across the window?
3. Can we compress the context without losing information?
4. Can we expand the effective context by geometric manipulation?

Key insight from Doc 189: The "click" happens at layer 3.
- Before layer 3: context is being integrated
- At layer 3: the critical decision is made
- After layer 3: the path is determined

For the 9x speedup: if we can predict the layer 3 output from
a compressed context representation, we skip layers 4-27.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


@dataclass
class ContextGeometry:
    """Geometric properties of a context window."""
    num_tokens: int
    effective_dim: int  # How many dimensions capture 90% variance
    attention_entropy: float  # How spread out is attention
    attention_concentration: float  # What fraction goes to top-k tokens
    layer3_phi_level: float
    bottleneck_phi_level: float


class ContextWindowAnalyzer:
    """Analyze the geometry of context windows."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"  # Required for attention output
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
    
    def _get_attention_and_hidden(self, text: str) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """Get attention weights and hidden states for all layers."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True,
                output_attentions=True
            )
        
        # Attention: list of (batch, heads, seq, seq) per layer
        attentions = [a.float().cpu() for a in outputs.attentions]
        
        # Hidden states: list of (batch, seq, hidden) per layer
        hidden_states = [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
        
        return attentions, hidden_states
    
    def _compute_phi_level(self, state: np.ndarray) -> float:
        """Compute mean φ-level."""
        magnitudes = np.abs(state)
        magnitudes = magnitudes[magnitudes > 1e-10]
        phi_levels = np.log(magnitudes) / LOG_PHI
        return float(np.mean(phi_levels))
    
    def analyze_context(self, text: str) -> ContextGeometry:
        """Analyze the geometric properties of a context."""
        attentions, hidden_states = self._get_attention_and_hidden(text)
        
        num_tokens = len(self.tokenizer.encode(text))
        
        # Analyze layer 3 attention (the click point)
        layer3_attn = attentions[3][0]  # (heads, seq, seq)
        
        # Average attention from last token to all positions
        last_token_attn = layer3_attn[:, -1, :].mean(dim=0).numpy()  # (seq,)
        
        # Attention entropy
        attn_probs = last_token_attn / (last_token_attn.sum() + 1e-10)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-10))
        max_entropy = np.log(num_tokens)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Attention concentration (top-3)
        top_k = min(3, num_tokens)
        top_k_attn = np.sort(attn_probs)[-top_k:].sum()
        
        # Effective dimensionality of context
        # Stack all hidden states and compute SVD
        if num_tokens > 1:
            # Get hidden states for all tokens at layer 3
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
            
            layer3_all = outputs.hidden_states[3][0].float().cpu().numpy()  # (seq, hidden)
            
            # SVD to find effective dimensionality
            U, S, Vt = np.linalg.svd(layer3_all, full_matrices=False)
            cumvar = np.cumsum(S**2) / np.sum(S**2)
            effective_dim = np.searchsorted(cumvar, 0.9) + 1
        else:
            effective_dim = 1
        
        return ContextGeometry(
            num_tokens=num_tokens,
            effective_dim=effective_dim,
            attention_entropy=normalized_entropy,
            attention_concentration=top_k_attn,
            layer3_phi_level=self._compute_phi_level(hidden_states[3]),
            bottleneck_phi_level=self._compute_phi_level(hidden_states[27] if len(hidden_states) > 27 else hidden_states[-1])
        )
    
    def compare_contexts(self, short_context: str, long_context: str) -> Dict:
        """Compare geometry of short vs long context."""
        short_geom = self.analyze_context(short_context)
        long_geom = self.analyze_context(long_context)
        
        return {
            'short': short_geom,
            'long': long_geom,
            'token_ratio': long_geom.num_tokens / short_geom.num_tokens,
            'dim_ratio': long_geom.effective_dim / short_geom.effective_dim,
            'entropy_change': long_geom.attention_entropy - short_geom.attention_entropy,
        }
    
    def find_attention_anchors(self, text: str) -> List[Tuple[int, float, str]]:
        """Find which tokens receive the most attention (boom positions)."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_attentions=True)
        
        # Average attention across layers and heads
        all_attn = torch.stack([a[0].float().cpu().mean(dim=0) for a in outputs.attentions])  # (layers, seq, seq)
        
        # Attention received by each position (from all other positions)
        attn_received = all_attn.mean(dim=0).sum(dim=0).numpy()  # (seq,)
        
        # Normalize
        attn_received = attn_received / attn_received.sum()
        
        # Find anchors (positions with above-average attention)
        mean_attn = 1.0 / len(tokens)
        anchors = []
        for i, (attn, token) in enumerate(zip(attn_received, tokens)):
            if attn > mean_attn * 1.5:  # 1.5x average
                anchors.append((i, float(attn), token))
        
        return sorted(anchors, key=lambda x: -x[1])
    
    def test_context_compression(self, full_context: str, query: str) -> Dict:
        """
        Test if we can compress context without losing information.
        
        Approach: Keep only the attention anchors and see if output changes.
        """
        full_text = full_context + " " + query
        
        # Get anchors
        anchors = self.find_attention_anchors(full_text)
        anchor_positions = set(a[0] for a in anchors[:5])  # Top 5 anchors
        
        # Get full output
        inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            full_outputs = self.model(inputs.input_ids, output_hidden_states=True)
        full_layer3 = full_outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
        full_final = full_outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        # Create compressed context (keep only anchor tokens)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        compressed_tokens = [tokens[i] for i in sorted(anchor_positions)]
        compressed_text = self.tokenizer.convert_tokens_to_string(compressed_tokens) + " " + query
        
        # Get compressed output
        comp_inputs = self.tokenizer(compressed_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            comp_outputs = self.model(comp_inputs.input_ids, output_hidden_states=True)
        comp_layer3 = comp_outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
        comp_final = comp_outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        # Compare
        layer3_cos = np.dot(full_layer3, comp_layer3) / (
            np.linalg.norm(full_layer3) * np.linalg.norm(comp_layer3) + 1e-10
        )
        final_cos = np.dot(full_final, comp_final) / (
            np.linalg.norm(full_final) * np.linalg.norm(comp_final) + 1e-10
        )
        
        return {
            'full_tokens': len(tokens),
            'compressed_tokens': len(compressed_tokens),
            'compression_ratio': len(tokens) / len(compressed_tokens),
            'layer3_similarity': float(layer3_cos),
            'final_similarity': float(final_cos),
            'anchors': anchors[:5]
        }
    
    def test_layer3_speedup(self, contexts: List[str]) -> Dict:
        """
        Test the 9x speedup hypothesis.
        
        If we can predict layer 3 output accurately, we can skip layers 4-27.
        This tests how much information is in layer 3 vs the final layer.
        """
        results = []
        
        for context in contexts:
            inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
            
            layer3 = outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
            final = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            
            # How similar are they?
            cos_sim = np.dot(layer3, final) / (
                np.linalg.norm(layer3) * np.linalg.norm(final) + 1e-10
            )
            
            # φ-level change
            layer3_phi = self._compute_phi_level(layer3)
            final_phi = self._compute_phi_level(final)
            
            results.append({
                'context': context[:50],
                'layer3_final_cos': float(cos_sim),
                'layer3_phi': layer3_phi,
                'final_phi': final_phi,
                'phi_change': final_phi - layer3_phi
            })
        
        return {
            'results': results,
            'mean_similarity': np.mean([r['layer3_final_cos'] for r in results]),
            'mean_phi_change': np.mean([r['phi_change'] for r in results])
        }


def run_context_geometry_analysis():
    """Analyze context window geometry."""
    analyzer = ContextWindowAnalyzer()
    
    print("=" * 60)
    print("CONTEXT WINDOW GEOMETRY ANALYSIS")
    print("=" * 60)
    
    # Test 1: Compare short vs long context
    print("\n1. SHORT VS LONG CONTEXT")
    print("-" * 40)
    
    short = "What is φ?"
    long = """The golden ratio φ (phi) equals (1 + √5) / 2 ≈ 1.618.
It appears throughout nature, art, and mathematics.
In TruthSpace, φ is the fundamental constant that governs
the geometric structure of semantic space.
What is φ?"""
    
    comparison = analyzer.compare_contexts(short, long)
    
    print(f"\nShort context ({comparison['short'].num_tokens} tokens):")
    print(f"  Effective dim: {comparison['short'].effective_dim}")
    print(f"  Attention entropy: {comparison['short'].attention_entropy:.3f}")
    print(f"  Layer 3 φ-level: {comparison['short'].layer3_phi_level:.3f}")
    
    print(f"\nLong context ({comparison['long'].num_tokens} tokens):")
    print(f"  Effective dim: {comparison['long'].effective_dim}")
    print(f"  Attention entropy: {comparison['long'].attention_entropy:.3f}")
    print(f"  Layer 3 φ-level: {comparison['long'].layer3_phi_level:.3f}")
    
    print(f"\nRatios:")
    print(f"  Token ratio: {comparison['token_ratio']:.1f}x")
    print(f"  Dim ratio: {comparison['dim_ratio']:.1f}x")
    print(f"  Entropy change: {comparison['entropy_change']:+.3f}")
    
    # Test 2: Find attention anchors
    print("\n2. ATTENTION ANCHORS (Boom Positions)")
    print("-" * 40)
    
    test_text = """You are completing a goal step by step.
GOAL: Write a summary about the φ-computer proof
TOOLS: search, generate, done
Current state: No knowledge gathered yet.
What is your next action?"""
    
    anchors = analyzer.find_attention_anchors(test_text)
    print(f"\nTop attention anchors in planning context:")
    for pos, attn, token in anchors[:10]:
        print(f"  Position {pos:2d}: {attn:.3f} - '{token}'")
    
    # Test 3: Context compression
    print("\n3. CONTEXT COMPRESSION")
    print("-" * 40)
    
    full_context = """The φ-computer proof shows that transformers are geometric computers.
Layer 3 is the click point where context is integrated.
The bottleneck at layer 27 converges to φ-level 1.
This means the transformer's intelligence is in its geometry."""
    
    query = "What is the key insight?"
    
    compression = analyzer.test_context_compression(full_context, query)
    print(f"\nFull context: {compression['full_tokens']} tokens")
    print(f"Compressed: {compression['compressed_tokens']} tokens")
    print(f"Compression ratio: {compression['compression_ratio']:.1f}x")
    print(f"Layer 3 similarity: {compression['layer3_similarity']:.3f}")
    print(f"Final similarity: {compression['final_similarity']:.3f}")
    
    # Test 4: Layer 3 speedup potential
    print("\n4. LAYER 3 SPEEDUP POTENTIAL")
    print("-" * 40)
    
    test_contexts = [
        "What is 2 + 2?",
        "Explain the concept of φ in mathematics.",
        "Write a summary about transformers.",
        "GOAL: Search for information. Current state: No knowledge.",
        "GOAL: Generate output. Current state: Knowledge gathered.",
    ]
    
    speedup = analyzer.test_layer3_speedup(test_contexts)
    
    print(f"\nLayer 3 → Final layer analysis:")
    for r in speedup['results']:
        print(f"  {r['context'][:40]}...")
        print(f"    Cosine: {r['layer3_final_cos']:.3f}, φ-change: {r['phi_change']:+.3f}")
    
    print(f"\nMean similarity (L3 → Final): {speedup['mean_similarity']:.3f}")
    print(f"Mean φ-level change: {speedup['mean_phi_change']:+.3f}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if speedup['mean_similarity'] > 0.8:
        print("\n✓ HIGH SIMILARITY between layer 3 and final layer")
        print("  → Layers 4-27 are refinement, not transformation")
        print("  → 9x speedup is FEASIBLE for action prediction")
    elif speedup['mean_similarity'] > 0.5:
        print("\n~ MODERATE SIMILARITY between layer 3 and final layer")
        print("  → Some information is added in layers 4-27")
        print("  → Partial speedup possible with approximation")
    else:
        print("\n✗ LOW SIMILARITY between layer 3 and final layer")
        print("  → Layers 4-27 significantly transform the representation")
        print("  → Full computation may be needed")
    
    return analyzer


if __name__ == "__main__":
    run_context_geometry_analysis()
