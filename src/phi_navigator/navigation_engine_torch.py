#!/usr/bin/env python3
"""
PyTorch-Optimized Navigation Inference Engine
==============================================

Fast navigation inference using PyTorch for GPU/CPU acceleration.

Loads φ-encoded weights from cache and decodes them to PyTorch tensors
at load time for fast inference.

Usage:
    from navigation_engine_torch import TorchNavigationEngine
    
    engine = TorchNavigationEngine(device='cuda')  # or 'cpu'
    engine.load_from_cache()
    
    text = engine.generate("Hello!", max_tokens=50)
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
PHI_SCALE = 8192


def decode_phi_tensor(path: str, device: str = 'cpu') -> torch.Tensor:
    """Load φ-encoded tensor and decode to PyTorch tensor (float32)."""
    data = np.load(path)
    signs = data['signs'].astype(np.float32)
    exps = data['exps'].astype(np.float32)
    shape = tuple(data['shape'])
    
    # Compute in float32
    phi_f32 = np.float32(PHI)
    scale_f32 = np.float32(PHI_SCALE)
    values = signs * (phi_f32 ** (exps / scale_f32))
    return torch.from_numpy(values.reshape(shape)).to(device)


class TorchNavigationEngine:
    """
    PyTorch-optimized navigation inference engine.
    
    Decodes φ-tensors to PyTorch tensors at load time for fast inference.
    """
    
    def __init__(self, cache_dir: str = None, device: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phi_navigation")
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model components (PyTorch tensors)
        self.embeddings: Optional[torch.Tensor] = None
        self.lm_head: Optional[torch.Tensor] = None
        self.norm_weight: Optional[torch.Tensor] = None
        self.layers: List[Dict] = []
        
        # Config
        self.hidden_dim = 3584
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.vocab_size = 152064
        
        # Tokenizer
        self.tokenizer = None
    
    def load_from_cache(self, max_layers: int = None):
        """Load pre-converted model from cache as PyTorch tensors."""
        from transformers import AutoTokenizer
        
        print(f"Loading from cache: {self.cache_dir}")
        print(f"Device: {self.device}")
        
        start = time.time()
        
        # Load config
        config = np.load(os.path.join(self.cache_dir, 'config.npz'))
        n_layers = min(int(config['num_layers']), max_layers or 999)
        self.hidden_dim = int(config['hidden_dim'])
        self.num_heads = int(config['num_heads'])
        self.num_kv_heads = int(config['num_kv_heads'])
        self.head_dim = int(config['head_dim'])
        self.vocab_size = int(config['vocab_size'])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        
        # Load embeddings
        print("  Loading embeddings...")
        self.embeddings = decode_phi_tensor(
            os.path.join(self.cache_dir, 'embeddings.npz'), self.device
        )
        
        # Load LM head
        print("  Loading LM head...")
        self.lm_head = decode_phi_tensor(
            os.path.join(self.cache_dir, 'lm_head.npz'), self.device
        )
        
        # Load norm weight
        self.norm_weight = torch.from_numpy(
            np.load(os.path.join(self.cache_dir, 'norm_weight.npy')).astype(np.float32)
        ).to(self.device)
        
        # Load layers
        self.layers = []
        for layer_idx in range(n_layers):
            print(f"  Loading layer {layer_idx}...")
            layer_dir = os.path.join(self.cache_dir, f'layer_{layer_idx:02d}')
            
            # Load MESH matrices (stack into single tensor for efficiency)
            mesh_list = []
            for h in range(self.num_heads):
                mesh = decode_phi_tensor(
                    os.path.join(layer_dir, f'mesh_{h:02d}.npz'), self.device
                )
                mesh_list.append(mesh)
            
            # Load cross terms
            cross_data = np.load(os.path.join(layer_dir, 'cross_terms.npz'))
            
            # Load projections
            layer_data = {
                'mesh': mesh_list,  # List of (hidden_dim, hidden_dim) tensors
                'cross_qk': torch.from_numpy(cross_data['cross_qk'].astype(np.float32)).to(self.device),
                'cross_kq': torch.from_numpy(cross_data['cross_kq'].astype(np.float32)).to(self.device),
                'bias_term': torch.from_numpy(cross_data['bias_term'].astype(np.float32)).to(self.device),
                'W_v': decode_phi_tensor(os.path.join(layer_dir, 'W_v.npz'), self.device),
                'W_o': decode_phi_tensor(os.path.join(layer_dir, 'W_o.npz'), self.device),
                'b_v': torch.from_numpy(np.load(os.path.join(layer_dir, 'b_v.npy')).astype(np.float32)).to(self.device),
                'W_gate': decode_phi_tensor(os.path.join(layer_dir, 'W_gate.npz'), self.device),
                'W_up': decode_phi_tensor(os.path.join(layer_dir, 'W_up.npz'), self.device),
                'W_down': decode_phi_tensor(os.path.join(layer_dir, 'W_down.npz'), self.device),
            }
            
            # Load LayerNorm
            ln_data = np.load(os.path.join(layer_dir, 'layernorm.npz'))
            layer_data['ln1'] = torch.from_numpy(ln_data['ln1'].astype(np.float32)).to(self.device)
            layer_data['ln2'] = torch.from_numpy(ln_data['ln2'].astype(np.float32)).to(self.device)
            
            self.layers.append(layer_data)
        
        load_time = time.time() - start
        print(f"Loaded {len(self.layers)} layers in {load_time:.1f}s")
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMS normalization."""
        variance = (x ** 2).mean(dim=-1, keepdim=True)
        return (x / torch.sqrt(variance + eps)) * weight
    
    def navigate_attention(self, hidden: torch.Tensor, layer: Dict) -> torch.Tensor:
        """Compute attention using pre-computed MESH matrices."""
        batch_size, seq_len, _ = hidden.shape
        
        # RMS Norm
        normed = self._rms_norm(hidden, layer['ln1'])
        
        # V projection
        V = F.linear(normed, layer['W_v']) + layer['b_v']
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(1, 2)  # (batch, kv_heads, seq, head_dim)
        
        # Expand V for GQA
        heads_per_kv = self.num_heads // self.num_kv_heads
        V = V.repeat_interleave(heads_per_kv, dim=1)  # (batch, num_heads, seq, head_dim)
        
        # Compute attention scores using MESH
        all_attn_outputs = []
        
        for h in range(self.num_heads):
            MESH = layer['mesh'][h]
            cross_qk = layer['cross_qk'][h]  # (hidden_dim,)
            cross_kq = layer['cross_kq'][h]  # (hidden_dim,)
            bias = layer['bias_term'][h]     # scalar
            
            # scores = x @ MESH @ x.T + x @ cross_qk + cross_kq @ x.T + bias
            # Shape: (batch, seq_q, seq_k)
            scores = torch.einsum('bsd,de,bte->bst', normed, MESH, normed)
            scores = scores + (normed @ cross_qk).unsqueeze(-1)  # (batch, seq, 1)
            scores = scores + (normed @ cross_kq).unsqueeze(1)   # (batch, 1, seq)
            scores = scores + bias
            
            # Scale
            scores = scores / np.sqrt(self.head_dim)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply to V
            V_head = V[:, h, :, :]  # (batch, seq, head_dim)
            attn_output = torch.bmm(attn_weights, V_head)  # (batch, seq, head_dim)
            all_attn_outputs.append(attn_output)
        
        # Concatenate heads
        attn_output = torch.cat(all_attn_outputs, dim=-1)  # (batch, seq, num_heads * head_dim)
        
        # Output projection
        output = F.linear(attn_output, layer['W_o'])
        
        return hidden + output
    
    def navigate_mlp(self, hidden: torch.Tensor, layer: Dict, linearized: bool = False) -> torch.Tensor:
        """Compute MLP using φ-encoded weights."""
        # RMS Norm
        normed = self._rms_norm(hidden, layer['ln2'])
        
        # Gate and up projections
        gate = F.linear(normed, layer['W_gate'])
        up = F.linear(normed, layer['W_up'])
        
        if linearized:
            # Linearized SiLU: (gate * up) / 2
            mlp_hidden = (gate * up) / 2
        else:
            # Full SiLU
            mlp_hidden = F.silu(gate) * up
        
        # Down projection
        mlp_output = F.linear(mlp_hidden, layer['W_down'])
        
        return hidden + mlp_output
    
    @torch.no_grad()
    def navigate_forward(self, token_ids: List[int], linearized_mlp: bool = False) -> torch.Tensor:
        """Full forward pass using geometric navigation."""
        # Embed tokens
        token_tensor = torch.tensor(token_ids, device=self.device)
        hidden = self.embeddings[token_tensor].unsqueeze(0)  # (1, seq, hidden)
        
        # Navigate through layers
        for layer in self.layers:
            hidden = self.navigate_attention(hidden, layer)
            hidden = self.navigate_mlp(hidden, layer, linearized=linearized_mlp)
        
        # Final norm
        hidden = self._rms_norm(hidden, self.norm_weight)
        
        # LM head
        logits = F.linear(hidden, self.lm_head)
        
        return logits
    
    def sample_token(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return int(torch.argmax(logits).item())
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        probs = F.softmax(logits, dim=-1)
        
        # Top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff
        cutoff_mask = cumsum <= top_p
        cutoff_mask[0] = True  # Always include at least one token
        
        # Zero out tokens beyond cutoff
        filtered_probs = sorted_probs * cutoff_mask.float()
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        # Sample
        chosen_idx = torch.multinomial(filtered_probs, 1).item()
        return int(sorted_indices[chosen_idx].item())
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chat_format: bool = True,
    ) -> str:
        """Generate text using navigation inference."""
        # Apply chat template if requested
        if chat_format:
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted = prompt
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        generated_ids = []
        
        for i in range(max_tokens):
            # Forward pass
            all_ids = input_ids + generated_ids
            logits = self.navigate_forward(all_ids)
            
            # Sample next token
            next_logits = logits[0, -1, :]
            next_token = self.sample_token(next_logits, temperature, top_p)
            
            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            if next_token == 151645:  # <|im_end|>
                break
            
            generated_ids.append(next_token)
        
        # Decode
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    print("="*60)
    print("TORCH NAVIGATION ENGINE TEST")
    print("="*60)
    
    engine = TorchNavigationEngine()
    engine.load_from_cache()
    
    print("\n" + "="*60)
    print("GENERATING...")
    print("="*60)
    
    start = time.time()
    text = engine.generate("Hello!", max_tokens=30, temperature=0.0)
    gen_time = time.time() - start
    
    print(f"\nGenerated: {text}")
    print(f"Time: {gen_time:.2f}s")
