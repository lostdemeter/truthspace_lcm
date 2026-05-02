#!/usr/bin/env python3
"""
Attention AS the Shape Change
==============================

Key finding: The crossing pattern is unpredictable from h(A) and h(B).

But the transformer has ATTENTION which explicitly computes relationships.
What if the attention pattern IS the shape change?

From Doc 180: Relationships are rotations toward Platonic Ideals.
The attention mechanism might be computing WHICH Platonic Ideal to rotate toward.

Hypothesis: The attention weights encode the "type" of shape change.
If we can characterize attention patterns, we can precompute shape changes.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AttentionShapeAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        # Load with eager attention to get attention weights
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Heads: {self.n_heads}")
    
    def get_hidden_and_attention(self, token_ids: List[int]):
        """Get final hidden state and attention weights."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                output_hidden_states=True,
                output_attentions=True
            )
            
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            
            # Attention: list of (batch, heads, seq, seq) per layer
            # For 2-token sequence, we want attention from pos 1 to pos 0
            attentions = []
            for layer_attn in outputs.attentions:
                # (heads, 2, 2) -> attention from pos 1 to pos 0
                attn_to_first = layer_attn[0, :, -1, 0].float().cpu().numpy()
                attentions.append(attn_to_first)
            
            return hidden, np.array(attentions)  # (layers, heads)
    
    def analyze_attention_patterns(self, n_samples: int = 100):
        """
        Analyze attention patterns for 2-token sequences.
        
        Key question: Do attention patterns cluster into types?
        """
        print(f"\n--- Attention Pattern Analysis ({n_samples} pairs) ---")
        
        attention_patterns = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_AB, attn = self.get_hidden_and_attention([A, B])
                attention_patterns.append(attn.flatten())  # (layers * heads,)
                h_AB_list.append(h_AB)
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        attention_patterns = np.array(attention_patterns)
        h_AB = np.array(h_AB_list)
        
        print(f"\n  Attention pattern shape: {attention_patterns.shape}")
        
        # SVD of attention patterns (handle NaN/Inf)
        attention_patterns = np.nan_to_num(attention_patterns, nan=0.0, posinf=1.0, neginf=0.0)
        try:
            _, S, Vt = np.linalg.svd(attention_patterns, full_matrices=False)
        except:
            print("    SVD failed, using truncated SVD")
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(50, attention_patterns.shape[0]-1))
            svd.fit(attention_patterns)
            S = svd.singular_values_
            Vt = svd.components_
        
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Attention pattern SVD:")
        print(f"    Top 10 singular values: {S[:10].round(3)}")
        print(f"    Dims for 50% variance: {np.searchsorted(cumvar, 0.50) + 1}")
        print(f"    Dims for 90% variance: {np.searchsorted(cumvar, 0.90) + 1}")
        print(f"    Dims for 99% variance: {np.searchsorted(cumvar, 0.99) + 1}")
        
        # Can we predict h_AB from attention pattern?
        print(f"\n  Testing: h_AB = f(attention_pattern)")
        
        for k in [5, 10, 20, 50]:
            # Project attention to k dimensions
            attn_k = attention_patterns @ Vt[:k].T
            
            # Solve for W: h_AB ≈ attn_k @ W
            W, _, _, _ = np.linalg.lstsq(attn_k, h_AB, rcond=None)
            
            pred = attn_k @ W
            
            correct = sum(np.argmax(self.lm_head @ h_AB[i]) == np.argmax(self.lm_head @ pred[i]) 
                         for i in range(len(h_AB)))
            
            print(f"    k={k}: {correct}/{len(h_AB)} = {correct/len(h_AB)*100:.1f}%")
        
        return {
            'attention_patterns': attention_patterns,
            'h_AB': h_AB,
            'Vt': Vt,
            'S': S,
        }
    
    def analyze_per_layer_attention(self, n_samples: int = 50):
        """
        Analyze attention at each layer separately.
        
        Which layers have the most informative attention?
        """
        print(f"\n--- Per-Layer Attention Analysis ---")
        
        layer_attentions = {l: [] for l in range(self.n_layers)}
        h_AB_list = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_AB, attn = self.get_hidden_and_attention([A, B])
                
                for l in range(self.n_layers):
                    layer_attentions[l].append(attn[l])  # (heads,)
                
                h_AB_list.append(h_AB)
            except:
                continue
        
        h_AB = np.array(h_AB_list)
        
        print(f"\n  Per-layer attention to first token:")
        
        for l in range(self.n_layers):
            attn_l = np.array(layer_attentions[l])
            mean_attn = attn_l.mean()
            std_attn = attn_l.std()
            print(f"    Layer {l}: mean={mean_attn:.3f}, std={std_attn:.3f}")
    
    def test_attention_based_caching(self, n_samples: int = 100):
        """
        Test: Can we use attention patterns to index into a cache?
        
        If attention patterns cluster, we can:
        1. Identify the cluster for a new (A, B) pair
        2. Look up the precomputed h_AB for that cluster
        """
        print(f"\n--- Attention-Based Caching Test ---")
        
        # Collect data
        attention_patterns = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_AB, attn = self.get_hidden_and_attention([A, B])
                attention_patterns.append(attn.flatten())
                h_AB_list.append(h_AB)
            except:
                continue
        
        attention_patterns = np.array(attention_patterns)
        h_AB = np.array(h_AB_list)
        
        # Simple clustering: k-means on attention patterns
        from sklearn.cluster import KMeans
        
        for n_clusters in [10, 50, 100, 200]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(attention_patterns)
            
            # For each cluster, compute mean h_AB
            cluster_means = np.zeros((n_clusters, self.hidden_dim))
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    cluster_means[c] = h_AB[mask].mean(axis=0)
            
            # Test: predict h_AB using cluster mean
            correct = 0
            for i in range(len(h_AB)):
                pred = cluster_means[labels[i]]
                
                true_token = np.argmax(self.lm_head @ h_AB[i])
                pred_token = np.argmax(self.lm_head @ pred)
                
                if true_token == pred_token:
                    correct += 1
            
            accuracy = correct / len(h_AB)
            print(f"    {n_clusters} clusters: {correct}/{len(h_AB)} = {accuracy*100:.1f}%")


def main():
    print("=" * 70)
    print("ATTENTION AS THE SHAPE CHANGE")
    print("=" * 70)
    print("""
Key question: Does the attention pattern encode the shape change?

If attention patterns cluster into types, we can:
1. Identify the type for a new (A, B) pair
2. Apply the precomputed shape change for that type
""")
    
    analyzer = AttentionShapeAnalyzer()
    
    # 1. Attention pattern analysis
    attn_results = analyzer.analyze_attention_patterns(n_samples=100)
    
    # 2. Per-layer analysis
    analyzer.analyze_per_layer_attention(n_samples=50)
    
    # 3. Attention-based caching
    analyzer.test_attention_based_caching(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
