#!/usr/bin/env python3
"""
Final Fix: Correct indexing and token prediction
=================================================

The issue:
- actual_hidden has 29 entries: [embeddings, layer0, layer1, ..., layer27]
- unwound_hidden has 29 entries: [embeddings, layer0, layer1, ..., layer27]
- But we were iterating to n_layers+1 = 29, causing index 28 to fail

Also need to verify the final norm is being applied correctly.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class UnwoundTransformerFinal:
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
        
        self.extract_all_layers()
        
        print(f"  Hidden dim: {self.hidden_dim}, Layers: {self.n_layers}")
    
    def extract_all_layers(self):
        print(f"  Extracting {self.n_layers} layers...")
        self.layers = []
        
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            
            L = {
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
            
            L['W_q_heads'] = L['W_q'].reshape(self.n_heads, self.head_dim, self.hidden_dim)
            L['W_k_heads'] = L['W_k'].reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
            L['W_v_heads'] = L['W_v'].reshape(self.n_kv_heads, self.head_dim, self.hidden_dim)
            
            if L['b_q'] is not None:
                L['b_q_heads'] = L['b_q'].reshape(self.n_heads, self.head_dim)
            if L['b_k'] is not None:
                L['b_k_heads'] = L['b_k'].reshape(self.n_kv_heads, self.head_dim)
            if L['b_v'] is not None:
                L['b_v_heads'] = L['b_v'].reshape(self.n_kv_heads, self.head_dim)
            
            self.layers.append(L)
        
        # RoPE
        layer0 = self.model.model.layers[0]
        if hasattr(layer0.self_attn, 'rotary_emb') and hasattr(layer0.self_attn.rotary_emb, 'inv_freq'):
            self.inv_freq = layer0.self_attn.rotary_emb.inv_freq.float().cpu().numpy()
        else:
            self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.head_dim, 2) / self.head_dim))
    
    def rope_embed(self, seq_len):
        freqs = np.outer(np.arange(seq_len), self.inv_freq)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(self, x, cos, sin):
        x1, x2 = x[:self.head_dim//2], x[self.head_dim//2:]
        return x * cos + np.concatenate([-x2, x1]) * sin
    
    def rms_norm(self, x, w, eps=1e-6):
        return (x / np.sqrt(np.mean(x**2) + eps)) * w
    
    def silu(self, x):
        return x * (1 / (1 + np.exp(-np.clip(x, -20, 20))))
    
    def compute_layer(self, idx, h, cos, sin):
        L = self.layers[idx]
        h_A, h_B = h[0], h[1]
        
        h_A_n = self.rms_norm(h_A, L['ln_attn'])
        h_B_n = self.rms_norm(h_B, L['ln_attn'])
        
        attn_out = np.zeros((2, self.hidden_dim))
        
        # Position 0
        for hd in range(self.n_heads):
            kv = hd // self.heads_per_kv
            v_A = h_A_n @ L['W_v_heads'][kv].T
            if L['b_v'] is not None:
                v_A += L['b_v_heads'][kv]
            attn_out[0, hd*self.head_dim:(hd+1)*self.head_dim] = v_A
        
        # Position 1
        for hd in range(self.n_heads):
            kv = hd // self.heads_per_kv
            
            q_B = h_B_n @ L['W_q_heads'][hd].T
            k_A = h_A_n @ L['W_k_heads'][kv].T
            k_B = h_B_n @ L['W_k_heads'][kv].T
            v_A = h_A_n @ L['W_v_heads'][kv].T
            v_B = h_B_n @ L['W_v_heads'][kv].T
            
            if L['b_q'] is not None:
                q_B += L['b_q_heads'][hd]
            if L['b_k'] is not None:
                k_A += L['b_k_heads'][kv]
                k_B += L['b_k_heads'][kv]
            if L['b_v'] is not None:
                v_A += L['b_v_heads'][kv]
                v_B += L['b_v_heads'][kv]
            
            q_B_r = self.apply_rope(q_B, cos[1], sin[1])
            k_A_r = self.apply_rope(k_A, cos[0], sin[0])
            k_B_r = self.apply_rope(k_B, cos[1], sin[1])
            
            s_A = np.dot(q_B_r, k_A_r) / np.sqrt(self.head_dim)
            s_B = np.dot(q_B_r, k_B_r) / np.sqrt(self.head_dim)
            
            scores = np.array([s_A, s_B])
            attn = np.exp(scores - scores.max())
            attn = attn / attn.sum()
            
            attn_out[1, hd*self.head_dim:(hd+1)*self.head_dim] = attn[0]*v_A + attn[1]*v_B
        
        attn_out[0] = attn_out[0] @ L['W_o'].T
        attn_out[1] = attn_out[1] @ L['W_o'].T
        
        h_post = h + attn_out
        
        mlp_out = np.zeros((2, self.hidden_dim))
        for p in range(2):
            h_n = self.rms_norm(h_post[p], L['ln_mlp'])
            mlp_out[p] = (self.silu(h_n @ L['W_gate'].T) * (h_n @ L['W_up'].T)) @ L['W_down'].T
        
        return h_post + mlp_out
    
    def forward(self, A, B):
        h = np.stack([self.embeddings[A], self.embeddings[B]])
        cos, sin = self.rope_embed(2)
        
        all_h = [h.copy()]
        for i in range(self.n_layers):
            h = self.compute_layer(i, h, cos, sin)
            all_h.append(h.copy())
        
        # Final norm
        h_normed = np.stack([self.rms_norm(h[0], self.final_ln_weight),
                            self.rms_norm(h[1], self.final_ln_weight)])
        
        return h_normed, all_h
    
    def test_quick(self, n=20):
        """Quick test of layer accuracy and token prediction."""
        print(f"\n--- Quick Test ({n} pairs) ---")
        
        layer_cos = []
        token_correct = 0
        
        for i in range(n):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Actual
                ids = torch.tensor([[A, B]]).to(self.device)
                with torch.no_grad():
                    out = self.model(ids, output_hidden_states=True)
                
                actual_token = torch.argmax(out.logits[0, 1]).item()
                actual_h = [x[0].float().cpu().numpy() for x in out.hidden_states]
                
                # Unwound
                h_final, all_h = self.forward(A, B)
                
                # Compare last layer (before final norm)
                # actual_h[-1] is after layer 27, before final norm
                # all_h[-1] is after layer 27, before final norm
                cos = np.dot(actual_h[-1][1], all_h[-1][1]) / (
                    np.linalg.norm(actual_h[-1][1]) * np.linalg.norm(all_h[-1][1]) + 1e-10)
                layer_cos.append(cos)
                
                # Token prediction
                logits = self.lm_head @ h_final[1]
                pred_token = np.argmax(logits)
                
                if pred_token == actual_token:
                    token_correct += 1
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"  Last layer cosine: {np.mean(layer_cos):.4f}")
        print(f"  Token accuracy: {token_correct}/{n} = {token_correct/n*100:.1f}%")
        
        return np.mean(layer_cos), token_correct/n


def main():
    print("=" * 70)
    print("UNWIND FINAL FIX")
    print("=" * 70)
    
    model = UnwoundTransformerFinal()
    
    # Quick test
    cos, acc = model.test_quick(n=20)
    
    if acc > 0.8:
        print(f"\n✓ SUCCESS: {acc*100:.0f}% token accuracy")
    else:
        print(f"\n⚠ Low accuracy: {acc*100:.0f}%")
        print("  Investigating...")
        
        # Debug single pair
        A, B = 100, 200
        ids = torch.tensor([[A, B]]).to(model.device)
        
        with torch.no_grad():
            out = model.model(ids, output_hidden_states=True)
        
        actual_logits = out.logits[0, 1].float().cpu().numpy()
        actual_token = np.argmax(actual_logits)
        
        h_final, _ = model.forward(A, B)
        pred_logits = model.lm_head @ h_final[1]
        pred_token = np.argmax(pred_logits)
        
        print(f"\n  Debug (A={A}, B={B}):")
        print(f"    Actual token: {actual_token} = '{model.tokenizer.decode([actual_token])}'")
        print(f"    Pred token:   {pred_token} = '{model.tokenizer.decode([pred_token])}'")
        
        # Compare logits
        logits_cos = np.dot(actual_logits, pred_logits) / (
            np.linalg.norm(actual_logits) * np.linalg.norm(pred_logits) + 1e-10)
        print(f"    Logits cosine: {logits_cos:.4f}")
        
        # Compare final hidden
        actual_h_last = out.hidden_states[-1][0, 1].float().cpu().numpy()
        actual_h_normed = model.rms_norm(actual_h_last, model.final_ln_weight)
        
        h_cos = np.dot(actual_h_normed, h_final[1]) / (
            np.linalg.norm(actual_h_normed) * np.linalg.norm(h_final[1]) + 1e-10)
        print(f"    Final hidden cosine (after norm): {h_cos:.4f}")


if __name__ == "__main__":
    main()
