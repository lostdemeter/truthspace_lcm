#!/usr/bin/env python3
"""
Attention from Embeddings: The Safe Dial Mechanism
====================================================

Key insight: Attention is computed from embeddings:
    attention = softmax(Q @ K.T / sqrt(d))
    Q = W_q @ emb(current)
    K = W_k @ emb(context)

This means attention weights are DETERMINISTIC given the embeddings!
We can precompute Q and K for all tokens.

The "rotary plates" (context) don't change shape randomly -
they change based on the geometric relationship between Q and K.

If we can characterize this relationship, we can predict the shape change.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AttentionFromEmbeddingsAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_dim // self.n_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        # Extract Q, K, V projection matrices from layer 0
        self.extract_qkv_matrices()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Heads: {self.n_heads}")
        print(f"  Head dim: {self.head_dim}")
    
    def extract_qkv_matrices(self):
        """Extract Q, K, V projection matrices from the model."""
        # Qwen2 uses fused QKV projection
        # The weight matrix is (3 * hidden_dim, hidden_dim)
        
        layer0 = self.model.model.layers[0]
        
        # Get the Q, K, V projections
        # In Qwen2, these might be separate or fused
        if hasattr(layer0.self_attn, 'q_proj'):
            self.W_q = layer0.self_attn.q_proj.weight.data.float().cpu().numpy()
            self.W_k = layer0.self_attn.k_proj.weight.data.float().cpu().numpy()
            self.W_v = layer0.self_attn.v_proj.weight.data.float().cpu().numpy()
            print(f"  W_q shape: {self.W_q.shape}")
            print(f"  W_k shape: {self.W_k.shape}")
            print(f"  W_v shape: {self.W_v.shape}")
        else:
            print("  Could not find separate Q, K, V projections")
            self.W_q = None
    
    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Get token embeddings."""
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([token_ids]).to(device)
        
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(input_ids)
            return embeddings[0].float().cpu().numpy()
    
    def compute_attention_from_embeddings(self, emb_A: np.ndarray, emb_B: np.ndarray) -> float:
        """
        Compute attention weight from B to A using embeddings.
        
        attention = softmax(Q_B @ K_A.T / sqrt(d))
        """
        if self.W_q is None:
            return None
        
        # Q from B, K from A
        Q_B = emb_B @ self.W_q.T  # (hidden_dim,)
        K_A = emb_A @ self.W_k.T  # (hidden_dim,)
        
        # Reshape to heads
        Q_B = Q_B.reshape(self.n_heads, self.head_dim)
        K_A = K_A.reshape(self.n_heads, self.head_dim)
        
        # Attention score per head
        scores = np.sum(Q_B * K_A, axis=1) / np.sqrt(self.head_dim)  # (n_heads,)
        
        # For 2-token sequence, softmax is over [A, B]
        # We need K_B too
        K_B = emb_B @ self.W_k.T
        K_B = K_B.reshape(self.n_heads, self.head_dim)
        
        score_to_A = np.sum(Q_B * K_A, axis=1) / np.sqrt(self.head_dim)
        score_to_B = np.sum(Q_B * K_B, axis=1) / np.sqrt(self.head_dim)
        
        # Softmax
        scores = np.stack([score_to_A, score_to_B], axis=1)  # (n_heads, 2)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # Return attention to A, averaged across heads
        return attn[:, 0].mean()
    
    def test_attention_prediction(self, n_samples: int = 100):
        """
        Test: Can we predict the actual attention from embeddings?
        """
        print(f"\n--- Attention Prediction Test ({n_samples} pairs) ---")
        
        if self.W_q is None:
            print("  Cannot test - Q, K matrices not extracted")
            return
        
        device = next(self.model.parameters()).device
        
        predicted_attns = []
        actual_attns = []
        
        # Need to reload model with eager attention
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
        config._attn_implementation = "eager"
        
        model_eager = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get embeddings
                emb = self.get_embeddings([A, B])
                emb_A, emb_B = emb[0], emb[1]
                
                # Predict attention from embeddings
                pred_attn = self.compute_attention_from_embeddings(emb_A, emb_B)
                
                # Get actual attention from model
                input_ids = torch.tensor([[A, B]]).to(device)
                with torch.no_grad():
                    outputs = model_eager(input_ids, output_attentions=True)
                    # Layer 0 attention, from position 1 to position 0
                    actual_attn = outputs.attentions[0][0, :, 1, 0].mean().item()
                
                predicted_attns.append(pred_attn)
                actual_attns.append(actual_attn)
            except Exception as e:
                continue
        
        predicted_attns = np.array(predicted_attns)
        actual_attns = np.array(actual_attns)
        
        # Correlation
        correlation = np.corrcoef(predicted_attns, actual_attns)[0, 1]
        
        print(f"\n  Results:")
        print(f"    Predicted mean: {predicted_attns.mean():.4f}")
        print(f"    Actual mean: {actual_attns.mean():.4f}")
        print(f"    Correlation: {correlation:.4f}")
        
        # Error
        error = np.abs(predicted_attns - actual_attns).mean()
        print(f"    Mean absolute error: {error:.4f}")
        
        del model_eager
        
        return {
            'predicted': predicted_attns,
            'actual': actual_attns,
            'correlation': correlation,
        }
    
    def precompute_qk_for_vocab(self, n_tokens: int = 1000):
        """
        Precompute Q and K vectors for a sample of the vocabulary.
        
        If we can precompute Q and K for all tokens, we can compute
        attention weights without running the model!
        """
        print(f"\n--- Precomputing Q, K for {n_tokens} tokens ---")
        
        if self.W_q is None:
            print("  Cannot precompute - Q, K matrices not extracted")
            return
        
        device = next(self.model.parameters()).device
        
        Q_cache = []
        K_cache = []
        
        for i in range(n_tokens):
            if i % 200 == 0:
                print(f"  Token {i}/{n_tokens}...")
            
            token_id = i  # Just use first n_tokens
            
            try:
                emb = self.get_embeddings([token_id])[0]
                
                Q = emb @ self.W_q.T
                K = emb @ self.W_k.T
                
                Q_cache.append(Q)
                K_cache.append(K)
            except:
                Q_cache.append(np.zeros(self.hidden_dim))
                K_cache.append(np.zeros(self.hidden_dim))
        
        Q_cache = np.array(Q_cache)
        K_cache = np.array(K_cache)
        
        print(f"\n  Q cache shape: {Q_cache.shape}")
        print(f"  K cache shape: {K_cache.shape}")
        
        # Storage estimate for full vocab
        vocab_size = self.tokenizer.vocab_size
        storage_per_token = 2 * self.hidden_dim * 2  # Q and K, float16
        total_storage = vocab_size * storage_per_token / (1024**3)
        
        print(f"\n  Storage estimate for full vocab ({vocab_size} tokens):")
        print(f"    Q + K cache: {total_storage:.2f} GB")
        
        return {
            'Q_cache': Q_cache,
            'K_cache': K_cache,
        }
    
    def test_attention_based_generation(self, n_samples: int = 50):
        """
        Test: Can we predict the next token using precomputed attention?
        
        Strategy:
        1. Compute attention weights from Q, K cache
        2. Use attention to weight the V vectors
        3. Apply remaining layers (or use cached final hidden states)
        """
        print(f"\n--- Attention-Based Generation Test ---")
        
        # This is complex because we need to handle all layers
        # For now, let's just verify that layer 0 attention is predictable
        
        print("  (Testing layer 0 attention predictability only)")
        
        # The key question: If we can predict layer 0 attention,
        # can we use that to adjust the single-token cache?
        
        # Hypothesis: h_final(A,B) ≈ attn_to_A * h_final(A) + attn_to_B * h_final(B)
        
        device = next(self.model.parameters()).device
        
        # Load model with eager attention
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
        config._attn_implementation = "eager"
        
        model_eager = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        correct = 0
        
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(device)
                
                with torch.no_grad():
                    # Get actual outputs
                    outputs = model_eager(input_ids, output_hidden_states=True, output_attentions=True)
                    h_AB = outputs.hidden_states[-1][0, 1, :].float().cpu().numpy()
                    
                    # Get attention at layer 0
                    attn_layer0 = outputs.attentions[0][0, :, 1, :].mean(dim=0).cpu().numpy()  # (2,)
                    attn_to_A = attn_layer0[0]
                    attn_to_B = attn_layer0[1]
                    
                    # Get single-token hidden states
                    out_A = model_eager(torch.tensor([[A]]).to(device), output_hidden_states=True)
                    out_B = model_eager(torch.tensor([[B]]).to(device), output_hidden_states=True)
                    
                    h_A = out_A.hidden_states[-1][0, 0, :].float().cpu().numpy()
                    h_B = out_B.hidden_states[-1][0, 0, :].float().cpu().numpy()
                    
                    # Predict: h_AB ≈ attn_to_A * h_A + attn_to_B * h_B
                    h_pred = attn_to_A * h_A + attn_to_B * h_B
                    
                    # Compare tokens
                    true_token = np.argmax(self.lm_head @ h_AB)
                    pred_token = np.argmax(self.lm_head @ h_pred)
                    
                    if true_token == pred_token:
                        correct += 1
            except Exception as e:
                continue
        
        accuracy = correct / n_samples
        print(f"\n  Attention-weighted combination accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        
        del model_eager
        
        return accuracy


def main():
    print("=" * 70)
    print("ATTENTION FROM EMBEDDINGS: THE SAFE DIAL MECHANISM")
    print("=" * 70)
    print("""
The safe dial analogy:
- Dial = current token (deterministic path)
- Rotary plates = context tokens (their K vectors)
- Shape change = attention weights (Q @ K.T)

Key insight: Attention is DETERMINISTIC given embeddings!
We can precompute Q and K for all tokens.
""")
    
    analyzer = AttentionFromEmbeddingsAnalyzer()
    
    # 1. Test attention prediction from embeddings
    attn_results = analyzer.test_attention_prediction(n_samples=100)
    
    # 2. Precompute Q, K for vocab sample
    qk_cache = analyzer.precompute_qk_for_vocab(n_tokens=1000)
    
    # 3. Test attention-based generation
    accuracy = analyzer.test_attention_based_generation(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
