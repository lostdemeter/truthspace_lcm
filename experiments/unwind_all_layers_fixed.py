#!/usr/bin/env python3
"""
Unwind All 28 Layers (Fixed): Full Model Inference
====================================================

Fixed issues:
1. hidden_states has 29 entries (embeddings + 28 layers)
2. hidden_states[-1] is AFTER final norm, not before
3. Need to compare layer outputs correctly

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class UnwoundTransformerFixed:
    """A fully unwound transformer with correct hidden state handling."""
    
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
        self.final_ln_weight = self.model.model.norm.weight.data.float().cpu().numpy()
        
        # Extract all layer weights
        self.extract_all_layers()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def extract_all_layers(self):
        """Extract weights for all layers."""
        print(f"  Extracting weights for {self.n_layers} layers...")
        
        self.layers = []
        
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            
            layer_weights = {
                'W_q': attn.q_proj.weight.data.float().cpu().numpy(),
                'W_k': attn.k_proj.weight.data.float().cpu().numpy(),
                'W_v': attn.v_proj.weight.data.float().cpu().numpy(),
                'W_o': attn.o_proj.weight.data.float().cpu().numpy(),
                'b_q': attn.q_proj.bias.data.float().cpu().numpy() if attn.q_proj.bias is not None else None,
                'b_k': attn.k_proj.bias.data.float().cpu().numpy() if attn.k_proj.bias is not None else None,
                'b_v': attn.v_proj.bias.data.float().cpu().numpy() if attn.v_proj.bias is not None else None,
                'ln_attn': layer.input_layernorm.weight.data.float().cpu().numpy(),
                'ln_mlp': layer.post_attention_layernorm.weight.data.float().cpu().numpy(),
                'W_gate': layer.mlp.gate_proj.weight.data.float().cpu().numpy(),
                'W_up': layer.mlp.up_proj.weight.data.float().cpu().numpy(),
                'W_down': layer.mlp.down_proj.weight.data.float().cpu().numpy(),
            }
            
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
        
        # Extract RoPE
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
        positions = np.arange(seq_len)
        freqs = np.outer(positions, self.inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        x1 = x[:self.head_dim//2]
        x2 = x[self.head_dim//2:]
        x_rotated = np.concatenate([-x2, x1])
        return x * cos + x_rotated * sin
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * weight
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_layer(self, layer_idx: int, hidden_states: np.ndarray, 
                      cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Compute a single layer for a 2-token sequence."""
        L = self.layers[layer_idx]
        h_A, h_B = hidden_states[0], hidden_states[1]
        
        h_A_norm = self.rms_norm(h_A, L['ln_attn'])
        h_B_norm = self.rms_norm(h_B, L['ln_attn'])
        
        attn_output = np.zeros((2, self.hidden_dim))
        
        # Position 0 (token A) - only attends to itself
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
            
            q_A_rope = self.apply_rope(q_A, cos[0], sin[0])
            k_A_rope = self.apply_rope(k_A, cos[0], sin[0])
            
            attn_output[0, h*self.head_dim:(h+1)*self.head_dim] = v_A
        
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
            
            q_B_rope = self.apply_rope(q_B, cos[1], sin[1])
            k_A_rope = self.apply_rope(k_A, cos[0], sin[0])
            k_B_rope = self.apply_rope(k_B, cos[1], sin[1])
            
            score_to_A = np.dot(q_B_rope, k_A_rope) / np.sqrt(self.head_dim)
            score_to_B = np.dot(q_B_rope, k_B_rope) / np.sqrt(self.head_dim)
            
            scores = np.array([score_to_A, score_to_B])
            exp_scores = np.exp(scores - scores.max())
            attn = exp_scores / exp_scores.sum()
            
            v_out = attn[0] * v_A + attn[1] * v_B
            attn_output[1, h*self.head_dim:(h+1)*self.head_dim] = v_out
        
        attn_output[0] = attn_output[0] @ L['W_o'].T
        attn_output[1] = attn_output[1] @ L['W_o'].T
        
        h_post_attn = hidden_states + attn_output
        
        mlp_output = np.zeros((2, self.hidden_dim))
        for pos in range(2):
            h_norm = self.rms_norm(h_post_attn[pos], L['ln_mlp'])
            gate = h_norm @ L['W_gate'].T
            up = h_norm @ L['W_up'].T
            mlp_out = self.silu(gate) * up
            mlp_output[pos] = mlp_out @ L['W_down'].T
        
        return h_post_attn + mlp_output
    
    def forward_unwound(self, token_A: int, token_B: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run full forward pass using unwound computation."""
        h = np.stack([self.embeddings[token_A], self.embeddings[token_B]])
        cos, sin = self.compute_rope_embeddings(2)
        
        all_hidden = [h.copy()]
        
        for layer_idx in range(self.n_layers):
            h = self.compute_layer(layer_idx, h, cos, sin)
            all_hidden.append(h.copy())
        
        # Apply final layer norm
        h_final = np.stack([
            self.rms_norm(h[0], self.final_ln_weight),
            self.rms_norm(h[1], self.final_ln_weight)
        ])
        
        return h_final, all_hidden
    
    def test_layer_by_layer(self, n_samples: int = 30):
        """Test each layer's accuracy."""
        print(f"\n--- Testing Layer-by-Layer Accuracy ({n_samples} pairs) ---")
        
        # hidden_states[0] = embeddings
        # hidden_states[i] = after layer i-1 (for i = 1 to n_layers)
        layer_cosines = {i: [] for i in range(self.n_layers + 1)}
        
        for sample in range(n_samples):
            if sample % 10 == 0:
                print(f"  {sample}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                
                actual_hidden = [h[0].float().cpu().numpy() for h in outputs.hidden_states]
                _, unwound_hidden = self.forward_unwound(A, B)
                
                # Compare embeddings (layer 0)
                for i in range(self.n_layers + 1):
                    actual = actual_hidden[i][1]  # Position 1
                    unwound = unwound_hidden[i][1]
                    
                    cos = np.dot(actual, unwound) / (
                        np.linalg.norm(actual) * np.linalg.norm(unwound) + 1e-10)
                    layer_cosines[i].append(cos)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\n  Layer-by-layer cosine similarity:")
        for i in range(self.n_layers + 1):
            mean_cos = np.mean(layer_cosines[i])
            layer_name = "embeddings" if i == 0 else f"layer {i-1}"
            print(f"    {layer_name:12s}: {mean_cos:.4f}")
        
        return layer_cosines
    
    def test_final_token_prediction(self, n_samples: int = 50):
        """Test final token prediction accuracy."""
        print(f"\n--- Testing Final Token Prediction ({n_samples} pairs) ---")
        
        correct = 0
        final_cosines = []
        
        for sample in range(n_samples):
            if sample % 10 == 0:
                print(f"  {sample}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                
                actual_token = torch.argmax(outputs.logits[0, 1]).item()
                
                # Get unwound final hidden (after final norm)
                h_final_unwound, _ = self.forward_unwound(A, B)
                
                # Compute logits
                logits_unwound = self.lm_head @ h_final_unwound[1]
                unwound_token = np.argmax(logits_unwound)
                
                if actual_token == unwound_token:
                    correct += 1
                
                # Compare final hidden states
                # Note: outputs.hidden_states[-1] is BEFORE final norm in HF
                # We need to compare with our post-norm output
                actual_final = outputs.hidden_states[-1][0, 1].float().cpu().numpy()
                actual_final_normed = self.rms_norm(actual_final, self.final_ln_weight)
                
                cos = np.dot(actual_final_normed, h_final_unwound[1]) / (
                    np.linalg.norm(actual_final_normed) * np.linalg.norm(h_final_unwound[1]) + 1e-10)
                final_cosines.append(cos)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        accuracy = correct / n_samples
        mean_cos = np.mean(final_cosines)
        
        print(f"\n  Results:")
        print(f"    Token prediction accuracy: {correct}/{n_samples} = {accuracy*100:.1f}%")
        print(f"    Final hidden cosine (after norm): {mean_cos:.4f}")
        
        return accuracy, mean_cos


def main():
    print("=" * 70)
    print("UNWIND ALL 28 LAYERS (FIXED)")
    print("=" * 70)
    
    transformer = UnwoundTransformerFixed()
    
    # Test layer-by-layer
    layer_cosines = transformer.test_layer_by_layer(n_samples=30)
    
    # Test final token prediction
    accuracy, final_cos = transformer.test_final_token_prediction(n_samples=50)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if accuracy > 0.9:
        print(f"  ✓ Full unwinding successful: {accuracy*100:.1f}% token accuracy")
        print(f"  ✓ Final hidden cosine: {final_cos:.4f}")
    else:
        print(f"  ⚠ Accuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
