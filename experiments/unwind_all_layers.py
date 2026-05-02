#!/usr/bin/env python3
"""
Unwind All 28 Layers: Full Model Inference
============================================

We proved layer 3 can be unwound with 99.96% accuracy.
Now extend to all 28 layers and test full inference.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class UnwoundTransformer:
    """A fully unwound transformer that computes each layer explicitly."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        self.n_layers = self.model.config.num_hidden_layers
        self.heads_per_kv = self.n_heads // self.n_kv_heads
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        # Extract all layer weights
        self.extract_all_layers()
        
        # Extract final layer norm
        self.final_ln_weight = self.model.model.norm.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Heads: {self.n_heads}, KV heads: {self.n_kv_heads}")
    
    def extract_all_layers(self):
        """Extract weights for all 28 layers."""
        print(f"  Extracting weights for {self.n_layers} layers...")
        
        self.layers = []
        
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            
            layer_weights = {
                # Attention weights
                'W_q': attn.q_proj.weight.data.float().cpu().numpy(),
                'W_k': attn.k_proj.weight.data.float().cpu().numpy(),
                'W_v': attn.v_proj.weight.data.float().cpu().numpy(),
                'W_o': attn.o_proj.weight.data.float().cpu().numpy(),
                
                # Attention biases
                'b_q': attn.q_proj.bias.data.float().cpu().numpy() if attn.q_proj.bias is not None else None,
                'b_k': attn.k_proj.bias.data.float().cpu().numpy() if attn.k_proj.bias is not None else None,
                'b_v': attn.v_proj.bias.data.float().cpu().numpy() if attn.v_proj.bias is not None else None,
                
                # Layer norms
                'ln_attn': layer.input_layernorm.weight.data.float().cpu().numpy(),
                'ln_mlp': layer.post_attention_layernorm.weight.data.float().cpu().numpy(),
                
                # MLP weights
                'W_gate': layer.mlp.gate_proj.weight.data.float().cpu().numpy(),
                'W_up': layer.mlp.up_proj.weight.data.float().cpu().numpy(),
                'W_down': layer.mlp.down_proj.weight.data.float().cpu().numpy(),
            }
            
            # Reshape per-head weights
            layer_weights['W_q_heads'] = layer_weights['W_q'].reshape(self.n_heads, self.head_dim, self.hidden_dim)
            layer_weights['W_k_heads'] = layer_weights['W_k'].reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
            layer_weights['W_v_heads'] = layer_weights['W_v'].reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
            
            if layer_weights['b_q'] is not None:
                layer_weights['b_q_heads'] = layer_weights['b_q'].reshape(self.n_heads, self.head_dim)
            if layer_weights['b_k'] is not None:
                layer_weights['b_k_heads'] = layer_weights['b_k'].reshape(self.n_kv_heads, self.head_dim)
            if layer_weights['b_v'] is not None:
                layer_weights['b_v_heads'] = layer_weights['b_v'].reshape(self.n_kv_heads, self.head_dim)
            
            self.layers.append(layer_weights)
        
        # Extract RoPE (same for all layers)
        layer0 = self.model.model.layers[0]
        if hasattr(layer0.self_attn, 'rotary_emb'):
            rotary = layer0.self_attn.rotary_emb
            if hasattr(rotary, 'inv_freq'):
                self.inv_freq = rotary.inv_freq.float().cpu().numpy()
            else:
                base = 10000.0
                self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        else:
            base = 10000.0
            self.inv_freq = 1.0 / (base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        
        print(f"  Extracted {len(self.layers)} layers")
    
    def compute_rope_embeddings(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RoPE cos and sin for positions 0 to seq_len-1."""
        positions = np.arange(seq_len)
        freqs = np.outer(positions, self.inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Apply RoPE to a vector."""
        x1 = x[:self.head_dim//2]
        x2 = x[self.head_dim//2:]
        x_rotated = np.concatenate([-x2, x1])
        return x * cos + x_rotated * sin
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply RMS normalization."""
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation."""
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_layer(self, layer_idx: int, hidden_states: np.ndarray, 
                      cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """
        Compute a single layer for a 2-token sequence.
        
        hidden_states: (2, hidden_dim) - hidden states for tokens A and B
        cos, sin: (2, head_dim) - RoPE embeddings for positions 0 and 1
        
        Returns: (2, hidden_dim) - output hidden states
        """
        L = self.layers[layer_idx]
        
        # We compute for both positions, but focus on position 1 (token B)
        # which attends to both position 0 (token A) and itself
        
        h_A, h_B = hidden_states[0], hidden_states[1]
        
        # Layer norm
        h_A_norm = self.rms_norm(h_A, L['ln_attn'])
        h_B_norm = self.rms_norm(h_B, L['ln_attn'])
        
        # Compute attention output for both positions
        attn_output = np.zeros((2, self.hidden_dim))
        
        # Position 0 (token A) - only attends to itself (causal)
        for h in range(self.n_heads):
            kv_idx = h // self.heads_per_kv
            
            q_A = h_A_norm @ L['W_q_heads'][h].T
            k_A = h_A_norm @ L['W_k_heads'][kv_idx].T
            v_A = h_A_norm @ L['W_v_heads'][kv_idx].T
            
            if L['b_q'] is not None:
                q_A = q_A + L['b_q_heads'][h]
            if L['b_k'] is not None:
                k_A = k_A + L['b_k_heads'][kv_idx]
            if L['b_v'] is not None:
                v_A = v_A + L['b_v_heads'][kv_idx]
            
            # Apply RoPE
            q_A_rope = self.apply_rope(q_A, cos[0], sin[0])
            k_A_rope = self.apply_rope(k_A, cos[0], sin[0])
            
            # Attention (only to self for position 0)
            score = np.dot(q_A_rope, k_A_rope) / np.sqrt(self.head_dim)
            attn = 1.0  # softmax of single element is 1
            
            attn_output[0, h*self.head_dim:(h+1)*self.head_dim] = attn * v_A
        
        # Position 1 (token B) - attends to both A and B
        for h in range(self.n_heads):
            kv_idx = h // self.heads_per_kv
            
            q_B = h_B_norm @ L['W_q_heads'][h].T
            k_A = h_A_norm @ L['W_k_heads'][kv_idx].T
            k_B = h_B_norm @ L['W_k_heads'][kv_idx].T
            v_A = h_A_norm @ L['W_v_heads'][kv_idx].T
            v_B = h_B_norm @ L['W_v_heads'][kv_idx].T
            
            if L['b_q'] is not None:
                q_B = q_B + L['b_q_heads'][h]
            if L['b_k'] is not None:
                k_A = k_A + L['b_k_heads'][kv_idx]
                k_B = k_B + L['b_k_heads'][kv_idx]
            if L['b_v'] is not None:
                v_A = v_A + L['b_v_heads'][kv_idx]
                v_B = v_B + L['b_v_heads'][kv_idx]
            
            # Apply RoPE
            q_B_rope = self.apply_rope(q_B, cos[1], sin[1])
            k_A_rope = self.apply_rope(k_A, cos[0], sin[0])
            k_B_rope = self.apply_rope(k_B, cos[1], sin[1])
            
            # Attention scores
            score_to_A = np.dot(q_B_rope, k_A_rope) / np.sqrt(self.head_dim)
            score_to_B = np.dot(q_B_rope, k_B_rope) / np.sqrt(self.head_dim)
            
            scores = np.array([score_to_A, score_to_B])
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            
            v_out = attn[0] * v_A + attn[1] * v_B
            attn_output[1, h*self.head_dim:(h+1)*self.head_dim] = v_out
        
        # Output projection
        attn_output[0] = attn_output[0] @ L['W_o'].T
        attn_output[1] = attn_output[1] @ L['W_o'].T
        
        # Residual
        h_post_attn = hidden_states + attn_output
        
        # MLP for both positions
        mlp_output = np.zeros((2, self.hidden_dim))
        for pos in range(2):
            h_norm = self.rms_norm(h_post_attn[pos], L['ln_mlp'])
            gate = h_norm @ L['W_gate'].T
            up = h_norm @ L['W_up'].T
            mlp_out = self.silu(gate) * up
            mlp_output[pos] = mlp_out @ L['W_down'].T
        
        # Residual
        output = h_post_attn + mlp_output
        
        return output
    
    def forward_unwound(self, token_A: int, token_B: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run full forward pass using unwound computation.
        
        Returns: (final_hidden_B, all_hidden_states)
        """
        # Get embeddings
        h = np.stack([self.embeddings[token_A], self.embeddings[token_B]])
        
        # Compute RoPE for 2 positions
        cos, sin = self.compute_rope_embeddings(2)
        
        all_hidden = [h.copy()]
        
        # Run all layers
        for layer_idx in range(self.n_layers):
            h = self.compute_layer(layer_idx, h, cos, sin)
            all_hidden.append(h.copy())
        
        # Final layer norm
        h_final = np.stack([
            self.rms_norm(h[0], self.final_ln_weight),
            self.rms_norm(h[1], self.final_ln_weight)
        ])
        
        return h_final[1], all_hidden
    
    def get_actual_hidden_states(self, token_A: int, token_B: int) -> List[np.ndarray]:
        """Get actual hidden states from the model."""
        input_ids = torch.tensor([[token_A, token_B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0].float().cpu().numpy() for h in outputs.hidden_states]
    
    def test_layer_by_layer(self, n_samples: int = 50):
        """Test each layer's accuracy."""
        print(f"\n--- Testing Layer-by-Layer Accuracy ({n_samples} pairs) ---")
        
        layer_cosines = {i: [] for i in range(self.n_layers + 1)}
        
        for sample in range(n_samples):
            if sample % 10 == 0:
                print(f"  {sample}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get actual hidden states
                actual_hidden = self.get_actual_hidden_states(A, B)
                
                # Get unwound hidden states
                _, unwound_hidden = self.forward_unwound(A, B)
                
                # Compare each layer
                for layer_idx in range(self.n_layers + 1):
                    actual = actual_hidden[layer_idx][1]  # Position 1 (token B)
                    unwound = unwound_hidden[layer_idx][1]
                    
                    cos = np.dot(actual, unwound) / (
                        np.linalg.norm(actual) * np.linalg.norm(unwound) + 1e-10)
                    layer_cosines[layer_idx].append(cos)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Layer-by-layer cosine similarity:")
        for layer_idx in range(self.n_layers + 1):
            mean_cos = np.mean(layer_cosines[layer_idx])
            print(f"    Layer {layer_idx:2d}: {mean_cos:.4f}")
        
        return layer_cosines
    
    def test_final_token_prediction(self, n_samples: int = 100):
        """Test final token prediction accuracy."""
        print(f"\n--- Testing Final Token Prediction ({n_samples} pairs) ---")
        
        correct = 0
        final_cosines = []
        
        for sample in range(n_samples):
            if sample % 20 == 0:
                print(f"  {sample}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Get actual final hidden state
                input_ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits_actual = outputs.logits[0, 1].float().cpu().numpy()
                
                actual_token = np.argmax(logits_actual)
                
                # Get unwound final hidden state
                h_final_unwound, _ = self.forward_unwound(A, B)
                
                # Compute logits
                logits_unwound = self.lm_head @ h_final_unwound
                unwound_token = np.argmax(logits_unwound)
                
                if actual_token == unwound_token:
                    correct += 1
                
                # Cosine of final hidden
                actual_hidden = self.get_actual_hidden_states(A, B)[-1][1]
                cos = np.dot(actual_hidden, h_final_unwound) / (
                    np.linalg.norm(actual_hidden) * np.linalg.norm(h_final_unwound) + 1e-10)
                final_cosines.append(cos)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        accuracy = correct / n_samples
        mean_cos = np.mean(final_cosines)
        
        print(f"\n  Results:")
        print(f"    Token prediction accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        print(f"    Final hidden cosine: {mean_cos:.4f}")
        
        return accuracy, mean_cos


def main():
    print("=" * 70)
    print("UNWIND ALL 28 LAYERS: FULL MODEL INFERENCE")
    print("=" * 70)
    print("""
Layer 3 unwinding achieved 99.96% accuracy.
Now extending to all 28 layers for full inference.
""")
    
    transformer = UnwoundTransformer()
    
    # 1. Test layer-by-layer accuracy
    layer_cosines = transformer.test_layer_by_layer(n_samples=50)
    
    # 2. Test final token prediction
    accuracy, final_cos = transformer.test_final_token_prediction(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if accuracy > 0.9:
        print(f"  ✓ Full unwinding successful: {accuracy*100:.1f}% token accuracy")
    else:
        print(f"  ⚠ Accuracy lower than expected: {accuracy*100:.1f}%")
        print(f"    Investigating layer-by-layer degradation...")


if __name__ == "__main__":
    main()
