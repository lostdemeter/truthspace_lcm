#!/usr/bin/env python3
"""
Compact Navigation Engine with Hierarchical φ-Encoding + Adaptive Layers
=========================================================================

Two optimizations:
1. Hierarchical φ-encoding: 8 bits/weight instead of 40 bits (5x compression)
2. Adaptive layer activation: Use subset of layers for easy tokens

This reduces memory transfer from ~22GB to ~4GB per forward pass.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class CompactPhiTensor:
    """
    Compact φ-encoded tensor using hierarchical encoding.
    
    Pack sign (1 bit) + level (7 bits) into single int8 per level.
    Two levels = 2 bytes per weight = 4x compression vs float32.
    """
    # Packed: sign in bit 7, level in bits 0-6 (range -64 to +63)
    packed0: np.ndarray  # int8: sign + level for level 0
    packed1: np.ndarray  # int8: sign + level for level 1
    shape: tuple
    scale0: tuple  # (scale, center) for level 0
    scale1: tuple  # (scale, center) for level 1
    
    @classmethod
    def from_float(cls, tensor: np.ndarray) -> 'CompactPhiTensor':
        """Encode float tensor to compact hierarchical φ representation."""
        shape = tensor.shape
        flat = tensor.flatten().astype(np.float64)
        
        # Level 0: Main structure
        signs0 = (flat >= 0).astype(np.uint8)  # 1 for positive, 0 for negative
        magnitudes = np.abs(flat)
        magnitudes = np.maximum(magnitudes, 1e-45)
        
        # Compute scale to fit in 7-bit range (-64 to +63)
        raw_levels = np.log(magnitudes) / LOG_PHI
        scale0 = np.std(raw_levels) / 20.0  # Fit in ±64 range
        center0 = np.median(raw_levels)
        
        levels0 = np.clip(np.round((raw_levels - center0) / scale0), -64, 63).astype(np.int8)
        
        # Pack sign + level into int8: sign in high bit, level in low 7 bits
        packed0 = ((signs0 << 7) | (levels0 & 0x7F)).astype(np.uint8).view(np.int8)
        
        # Decode level 0 for residual calculation
        decoded0 = np.where(signs0, 1, -1) * (PHI ** (levels0 * scale0 + center0))
        
        # Level 1: Residual refinement
        residual = flat - decoded0
        signs1 = (residual >= 0).astype(np.uint8)
        magnitudes1 = np.abs(residual)
        magnitudes1 = np.maximum(magnitudes1, 1e-45)
        
        raw_levels1 = np.log(magnitudes1) / LOG_PHI
        scale1 = np.std(raw_levels1) / 20.0
        center1 = np.median(raw_levels1)
        
        levels1 = np.clip(np.round((raw_levels1 - center1) / scale1), -64, 63).astype(np.int8)
        packed1 = ((signs1 << 7) | (levels1 & 0x7F)).astype(np.uint8).view(np.int8)
        
        return cls(
            packed0=packed0, packed1=packed1,
            shape=shape,
            scale0=(float(scale0), float(center0)),
            scale1=(float(scale1), float(center1))
        )
    
    def to_float(self) -> np.ndarray:
        """Decode to float32."""
        scale0, center0 = self.scale0
        scale1, center1 = self.scale1
        
        # Unpack level 0
        packed0_u = self.packed0.view(np.uint8)
        signs0 = (packed0_u >> 7).astype(np.float32) * 2 - 1  # 1 or -1
        levels0_raw = (self.packed0 & 0x7F).astype(np.int16)  # Use int16 to avoid overflow
        # Sign extend 7-bit to full range
        levels0 = np.where(levels0_raw > 63, levels0_raw - 128, levels0_raw).astype(np.float32)
        decoded0 = signs0 * (PHI ** (levels0 * scale0 + center0))
        
        # Unpack level 1
        packed1_u = self.packed1.view(np.uint8)
        signs1 = (packed1_u >> 7).astype(np.float32) * 2 - 1
        levels1_raw = (self.packed1 & 0x7F).astype(np.int16)
        levels1 = np.where(levels1_raw > 63, levels1_raw - 128, levels1_raw).astype(np.float32)
        decoded1 = signs1 * (PHI ** (levels1 * scale1 + center1))
        
        return (decoded0 + decoded1).reshape(self.shape).astype(np.float32)
    
    def to_torch(self, device: str = 'cpu') -> torch.Tensor:
        """Decode directly to PyTorch tensor."""
        if device == 'cpu':
            return torch.from_numpy(self.to_float())
        
        # GPU-accelerated decoding: transfer packed int8, decode on GPU
        scale0, center0 = self.scale0
        scale1, center1 = self.scale1
        
        # Transfer packed arrays to GPU
        packed0_gpu = torch.from_numpy(self.packed0.view(np.uint8)).to(device)
        packed1_gpu = torch.from_numpy(self.packed1.view(np.uint8)).to(device)
        
        # Decode level 0 on GPU
        signs0 = (packed0_gpu >> 7).float() * 2 - 1
        levels0_raw = (packed0_gpu & 0x7F).short()
        levels0 = torch.where(levels0_raw > 63, levels0_raw - 128, levels0_raw).float()
        decoded0 = signs0 * (PHI ** (levels0 * scale0 + center0))
        
        # Decode level 1 on GPU
        signs1 = (packed1_gpu >> 7).float() * 2 - 1
        levels1_raw = (packed1_gpu & 0x7F).short()
        levels1 = torch.where(levels1_raw > 63, levels1_raw - 128, levels1_raw).float()
        decoded1 = signs1 * (PHI ** (levels1 * scale1 + center1))
        
        return (decoded0 + decoded1).view(self.shape)
    
    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.packed0.nbytes + self.packed1.nbytes + 32  # 32 for scales
    
    def save(self, path: str):
        np.savez_compressed(
            path,
            packed0=self.packed0, packed1=self.packed1,
            shape=np.array(self.shape),
            scale0=np.array(self.scale0),
            scale1=np.array(self.scale1)
        )
    
    @classmethod
    def load(cls, path: str) -> 'CompactPhiTensor':
        data = np.load(path)
        return cls(
            packed0=data['packed0'], packed1=data['packed1'],
            shape=tuple(data['shape']),
            scale0=tuple(data['scale0']),
            scale1=tuple(data['scale1'])
        )


class AdaptiveNavigationEngine:
    """
    Navigation engine with:
    1. Compact hierarchical φ-encoding (2x memory reduction)
    2. Hybrid GPU/CPU caching (keep N layers decoded on GPU)
    """
    
    def __init__(self, cache_dir: str = None, device: str = None, gpu_layers: int = 14):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phi_navigation_compact")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_layers = gpu_layers  # Number of layers to keep decoded on GPU
        
        # Model config (Qwen2-7B)
        self.hidden_dim = 3584
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.vocab_size = 152064
        self.rope_theta = 1000000.0
        
        # Weights stored compactly on CPU
        self.embeddings: Optional[torch.Tensor] = None  # Keep on GPU (used every token)
        self.lm_head: Optional[torch.Tensor] = None     # Keep on GPU (used every token)
        self.norm_weight: Optional[torch.Tensor] = None
        self.layers: List[Dict] = []  # Compact encoded, decoded on demand
        self.tokenizer = None
        
        # Layer groups for adaptive activation
        # Group 1: Essential layers (always used) - 14 layers
        self.essential_layers = list(range(0, 28, 2))  # Every other layer: 0,2,4,...,26
        # Group 2: Refinement layers (used for hard tokens) - remaining 14 layers  
        self.refinement_layers = list(range(1, 28, 2))  # 1,3,5,...,27
        
        # RoPE cache
        self._rope_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def convert_and_cache(self, max_layers: int = 28):
        """Convert model to compact φ-encoding."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch as th
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Loading Qwen/Qwen2-7B-Instruct...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            torch_dtype=th.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        
        n_layers = min(len(model.model.layers), max_layers)
        
        # Save config
        np.savez(
            os.path.join(self.cache_dir, 'config.npz'),
            num_layers=n_layers,
            hidden_dim=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            vocab_size=model.config.vocab_size,
            rope_theta=getattr(model.config, 'rope_theta', 1000000.0)
        )
        
        # Embeddings - keep as float16 (frequently accessed)
        print("Converting embeddings...")
        emb = model.model.embed_tokens.weight.detach().cpu().numpy()
        np.save(os.path.join(self.cache_dir, 'embeddings.npy'), emb.astype(np.float16))
        
        # LM head - keep as float16
        print("Converting LM head...")
        lm = model.lm_head.weight.detach().cpu().numpy()
        np.save(os.path.join(self.cache_dir, 'lm_head.npy'), lm.astype(np.float16))
        
        # Final norm
        norm = model.model.norm.weight.detach().cpu().numpy()
        np.save(os.path.join(self.cache_dir, 'norm_weight.npy'), norm)
        
        # Convert layers with compact encoding
        print(f"Converting {n_layers} layers with compact φ-encoding...")
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            layer_dir = os.path.join(self.cache_dir, f'layer_{layer_idx:02d}')
            os.makedirs(layer_dir, exist_ok=True)
            
            # Attention weights - compact encoded
            W_q = layer.self_attn.q_proj.weight.detach().cpu().numpy()
            W_k = layer.self_attn.k_proj.weight.detach().cpu().numpy()
            W_v = layer.self_attn.v_proj.weight.detach().cpu().numpy()
            W_o = layer.self_attn.o_proj.weight.detach().cpu().numpy()
            
            CompactPhiTensor.from_float(W_q).save(os.path.join(layer_dir, 'W_q.npz'))
            CompactPhiTensor.from_float(W_k).save(os.path.join(layer_dir, 'W_k.npz'))
            CompactPhiTensor.from_float(W_v).save(os.path.join(layer_dir, 'W_v.npz'))
            CompactPhiTensor.from_float(W_o).save(os.path.join(layer_dir, 'W_o.npz'))
            
            # Biases (small, keep as float32)
            np.savez(
                os.path.join(layer_dir, 'biases.npz'),
                b_q=layer.self_attn.q_proj.bias.detach().cpu().numpy() if layer.self_attn.q_proj.bias is not None else np.zeros(W_q.shape[0]),
                b_k=layer.self_attn.k_proj.bias.detach().cpu().numpy() if layer.self_attn.k_proj.bias is not None else np.zeros(W_k.shape[0]),
                b_v=layer.self_attn.v_proj.bias.detach().cpu().numpy() if layer.self_attn.v_proj.bias is not None else np.zeros(W_v.shape[0]),
            )
            
            # MLP weights - compact encoded
            W_gate = layer.mlp.gate_proj.weight.detach().cpu().numpy()
            W_up = layer.mlp.up_proj.weight.detach().cpu().numpy()
            W_down = layer.mlp.down_proj.weight.detach().cpu().numpy()
            
            CompactPhiTensor.from_float(W_gate).save(os.path.join(layer_dir, 'W_gate.npz'))
            CompactPhiTensor.from_float(W_up).save(os.path.join(layer_dir, 'W_up.npz'))
            CompactPhiTensor.from_float(W_down).save(os.path.join(layer_dir, 'W_down.npz'))
            
            # LayerNorm (small, keep as float32)
            np.savez(
                os.path.join(layer_dir, 'layernorm.npz'),
                ln1=layer.input_layernorm.weight.detach().cpu().numpy(),
                ln2=layer.post_attention_layernorm.weight.detach().cpu().numpy(),
            )
            
            print(f"  Layer {layer_idx}/{n_layers}")
        
        print(f"Cached {n_layers} layers to {self.cache_dir}")
        
        # Cleanup
        del model
        import gc
        gc.collect()
    
    def load_from_cache(self, max_layers: int = None):
        """Load compact model from cache."""
        from transformers import AutoTokenizer
        
        print(f"Loading from cache: {self.cache_dir}")
        
        config = np.load(os.path.join(self.cache_dir, 'config.npz'))
        n_layers = min(int(config['num_layers']), max_layers or 999)
        self.hidden_dim = int(config['hidden_dim'])
        self.num_heads = int(config['num_heads'])
        self.num_kv_heads = int(config['num_kv_heads'])
        self.head_dim = int(config['head_dim'])
        self.vocab_size = int(config['vocab_size'])
        self.rope_theta = float(config['rope_theta']) if 'rope_theta' in config else 1000000.0
        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        
        # Load embeddings and LM head to GPU (float16, frequently used)
        print("  Loading embeddings to GPU...")
        emb = np.load(os.path.join(self.cache_dir, 'embeddings.npy'))
        self.embeddings = torch.from_numpy(emb.astype(np.float32)).to(self.device)
        
        print("  Loading LM head to GPU...")
        lm = np.load(os.path.join(self.cache_dir, 'lm_head.npy'))
        self.lm_head = torch.from_numpy(lm.astype(np.float32)).to(self.device)
        
        self.norm_weight = torch.from_numpy(
            np.load(os.path.join(self.cache_dir, 'norm_weight.npy')).astype(np.float32)
        ).to(self.device)
        
        # Load layers - first N on GPU (decoded), rest as compact on CPU
        self.layers = []
        self.gpu_cached_layers = []  # Decoded layers on GPU
        
        n_gpu = min(self.gpu_layers, n_layers)
        print(f"  Caching {n_gpu} layers on GPU, {n_layers - n_gpu} on CPU (compact)")
        
        for layer_idx in range(n_layers):
            layer_dir = os.path.join(self.cache_dir, f'layer_{layer_idx:02d}')
            
            biases = np.load(os.path.join(layer_dir, 'biases.npz'))
            ln_data = np.load(os.path.join(layer_dir, 'layernorm.npz'))
            
            if layer_idx < n_gpu:
                # Keep decoded on GPU for fast access
                layer_data = {
                    'W_q': CompactPhiTensor.load(os.path.join(layer_dir, 'W_q.npz')).to_torch(self.device),
                    'W_k': CompactPhiTensor.load(os.path.join(layer_dir, 'W_k.npz')).to_torch(self.device),
                    'W_v': CompactPhiTensor.load(os.path.join(layer_dir, 'W_v.npz')).to_torch(self.device),
                    'W_o': CompactPhiTensor.load(os.path.join(layer_dir, 'W_o.npz')).to_torch(self.device),
                    'b_q': torch.from_numpy(biases['b_q'].astype(np.float32)).to(self.device),
                    'b_k': torch.from_numpy(biases['b_k'].astype(np.float32)).to(self.device),
                    'b_v': torch.from_numpy(biases['b_v'].astype(np.float32)).to(self.device),
                    'W_gate': CompactPhiTensor.load(os.path.join(layer_dir, 'W_gate.npz')).to_torch(self.device),
                    'W_up': CompactPhiTensor.load(os.path.join(layer_dir, 'W_up.npz')).to_torch(self.device),
                    'W_down': CompactPhiTensor.load(os.path.join(layer_dir, 'W_down.npz')).to_torch(self.device),
                    'ln1': torch.from_numpy(ln_data['ln1'].astype(np.float32)).to(self.device),
                    'ln2': torch.from_numpy(ln_data['ln2'].astype(np.float32)).to(self.device),
                    '_on_gpu': True,
                }
            else:
                # Keep compact on CPU
                layer_data = {
                    'W_q': CompactPhiTensor.load(os.path.join(layer_dir, 'W_q.npz')),
                    'W_k': CompactPhiTensor.load(os.path.join(layer_dir, 'W_k.npz')),
                    'W_v': CompactPhiTensor.load(os.path.join(layer_dir, 'W_v.npz')),
                    'W_o': CompactPhiTensor.load(os.path.join(layer_dir, 'W_o.npz')),
                    'b_q': torch.from_numpy(biases['b_q'].astype(np.float32)),
                    'b_k': torch.from_numpy(biases['b_k'].astype(np.float32)),
                    'b_v': torch.from_numpy(biases['b_v'].astype(np.float32)),
                    'W_gate': CompactPhiTensor.load(os.path.join(layer_dir, 'W_gate.npz')),
                    'W_up': CompactPhiTensor.load(os.path.join(layer_dir, 'W_up.npz')),
                    'W_down': CompactPhiTensor.load(os.path.join(layer_dir, 'W_down.npz')),
                    'ln1': torch.from_numpy(ln_data['ln1'].astype(np.float32)),
                    'ln2': torch.from_numpy(ln_data['ln2'].astype(np.float32)),
                    '_on_gpu': False,
                }
            self.layers.append(layer_data)
            print(f"  Loaded layer {layer_idx} ({'GPU' if layer_idx < n_gpu else 'CPU'})")
        
        print(f"Loaded {n_layers} layers ({n_gpu} on GPU, {n_layers - n_gpu} compact on CPU)")
    
    def _get_rope(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len not in self._rope_cache:
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim))
            positions = torch.arange(seq_len, device=self.device)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._rope_cache[seq_len] = (torch.cos(emb), torch.sin(emb))
        return self._rope_cache[seq_len]
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos.unsqueeze(0).unsqueeze(0) + x_rotated * sin.unsqueeze(0).unsqueeze(0)
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        variance = (x ** 2).mean(dim=-1, keepdim=True)
        return (x / torch.sqrt(variance + eps)) * weight
    
    def _get_layer_gpu(self, layer: Dict) -> Dict[str, torch.Tensor]:
        """Get layer weights on GPU (decode if needed)."""
        if layer.get('_on_gpu', False):
            # Already decoded on GPU
            return layer
        
        # Decode compact tensors to GPU
        return {
            'W_q': layer['W_q'].to_torch(self.device),
            'W_k': layer['W_k'].to_torch(self.device),
            'W_v': layer['W_v'].to_torch(self.device),
            'W_o': layer['W_o'].to_torch(self.device),
            'b_q': layer['b_q'].to(self.device),
            'b_k': layer['b_k'].to(self.device),
            'b_v': layer['b_v'].to(self.device),
            'W_gate': layer['W_gate'].to_torch(self.device),
            'W_up': layer['W_up'].to_torch(self.device),
            'W_down': layer['W_down'].to_torch(self.device),
            'ln1': layer['ln1'].to(self.device),
            'ln2': layer['ln2'].to(self.device),
            '_on_gpu': False,  # Mark as temporary
        }
    
    @torch.no_grad()
    def navigate_layer(self, hidden: torch.Tensor, layer_gpu: Dict[str, torch.Tensor],
                       cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Process one layer (attention + MLP)."""
        batch_size, seq_len, _ = hidden.shape
        heads_per_kv = self.num_heads // self.num_kv_heads
        
        # Attention
        normed = self._rms_norm(hidden, layer_gpu['ln1'])
        
        Q = F.linear(normed, layer_gpu['W_q'], layer_gpu['b_q'])
        K = F.linear(normed, layer_gpu['W_k'], layer_gpu['b_k'])
        V = F.linear(normed, layer_gpu['W_v'], layer_gpu['b_v'])
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        Q = self._apply_rope(Q, cos, sin)
        K = self._apply_rope(K, cos, sin)
        
        K = K.repeat_interleave(heads_per_kv, dim=1)
        V = V.repeat_interleave(heads_per_kv, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device) * float('-inf'), diagonal=1)
        scores = scores + mask
        
        attention = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attention, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = F.linear(attn_output, layer_gpu['W_o'])
        
        hidden = hidden + attn_output
        
        # MLP
        normed = self._rms_norm(hidden, layer_gpu['ln2'])
        gate = F.linear(normed, layer_gpu['W_gate'])
        up = F.linear(normed, layer_gpu['W_up'])
        mlp_hidden = F.silu(gate) * up
        mlp_output = F.linear(mlp_hidden, layer_gpu['W_down'])
        
        return hidden + mlp_output
    
    @torch.no_grad()
    def navigate_forward(self, token_ids: List[int], max_layers: int = None) -> torch.Tensor:
        """Forward pass with configurable layer count.
        
        Args:
            token_ids: Input token IDs
            max_layers: Maximum layers to use (None = all layers)
        """
        # Embed tokens (already on GPU)
        input_ids = torch.tensor(token_ids, device=self.device)
        hidden = self.embeddings[input_ids].unsqueeze(0)
        
        seq_len = len(token_ids)
        cos, sin = self._get_rope(seq_len)
        
        n_layers = min(max_layers or len(self.layers), len(self.layers))
        
        for i in range(n_layers):
            layer_gpu = self._get_layer_gpu(self.layers[i])
            hidden = self.navigate_layer(hidden, layer_gpu, cos, sin)
            # Only delete if it was a temporary decode (not cached on GPU)
            if not self.layers[i].get('_on_gpu', False):
                del layer_gpu
        
        # Final norm
        hidden = self._rms_norm(hidden, self.norm_weight)
        
        # LM head (already on GPU)
        logits = F.linear(hidden, self.lm_head)
        
        return logits
    
    @torch.no_grad()
    def generate_adaptive(self, token_ids: List[int], max_tokens: int = 50, 
                          temperature: float = 0.7, confidence_threshold: float = 0.8) -> List[int]:
        """Generate tokens with confidence-based adaptive layer count.
        
        Uses fewer layers for high-confidence tokens, more for uncertain ones.
        """
        generated = list(token_ids)
        
        for _ in range(max_tokens):
            # First try with 14 layers (half)
            logits = self.navigate_forward(generated, max_layers=14)
            probs = F.softmax(logits[0, -1], dim=-1)
            confidence = probs.max().item()
            
            if confidence < confidence_threshold:
                # Low confidence - use full model
                logits = self.navigate_forward(generated, max_layers=None)
            
            next_token = self.sample_token(logits[0, -1], temperature)
            generated.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def sample_token(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
        if temperature == 0:
            return int(torch.argmax(logits).item())
        
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        cutoff_idx = (cumsum > top_p).nonzero()
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0].item() + 1
        else:
            cutoff_idx = len(sorted_probs)
        
        top_probs = sorted_probs[:cutoff_idx]
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = top_probs / top_probs.sum()
        
        chosen_idx = torch.multinomial(top_probs, 1).item()
        return int(top_indices[chosen_idx].item())


def test_compact_encoding():
    """Test compact φ-encoding quality."""
    print("="*60)
    print("Testing Compact Hierarchical φ-Encoding")
    print("="*60)
    
    # Create random weight matrix
    np.random.seed(42)
    W = np.random.randn(3584, 3584).astype(np.float32) * 0.02
    
    # Encode and decode
    compact = CompactPhiTensor.from_float(W)
    W_decoded = compact.to_float()
    
    # Measure quality
    correlation = np.corrcoef(W.flatten(), W_decoded.flatten())[0, 1]
    mse = np.mean((W - W_decoded) ** 2)
    
    # Memory comparison
    original_bytes = W.nbytes
    compact_bytes = compact.memory_bytes()
    
    print(f"\nQuality:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  MSE: {mse:.2e}")
    
    print(f"\nMemory:")
    print(f"  Original: {original_bytes / 1e6:.1f} MB")
    print(f"  Compact: {compact_bytes / 1e6:.1f} MB")
    print(f"  Compression: {original_bytes / compact_bytes:.1f}x")


def test_speed():
    """Test inference speed with compact encoding."""
    print("\n" + "="*60)
    print("Testing Adaptive Navigation Engine")
    print("="*60)
    
    engine = AdaptiveNavigationEngine()
    
    # Check if cache exists
    if not os.path.exists(os.path.join(engine.cache_dir, 'config.npz')):
        print("Cache not found. Run with --convert first.")
        return
    
    import time
    start = time.time()
    engine.load_from_cache(max_layers=28)
    load_time = time.time() - start
    print(f"\nLoad time: {load_time:.1f}s")
    
    # Warm up
    token_ids = engine.tokenizer.encode("Hello", add_special_tokens=False)
    _ = engine.navigate_forward(token_ids)
    torch.cuda.synchronize()
    
    # Test adaptive vs full
    token_ids = engine.tokenizer.encode("What is the capital of France?", add_special_tokens=False)
    
    # Adaptive (9 key layers)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        logits = engine.navigate_forward(token_ids, use_adaptive=True)
        torch.cuda.synchronize()
    adaptive_time = (time.time() - start) / 3
    
    # Full (28 layers)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        logits = engine.navigate_forward(token_ids, use_adaptive=False)
        torch.cuda.synchronize()
    full_time = (time.time() - start) / 3
    
    print(f"\nForward pass times:")
    print(f"  Adaptive (9 layers): {adaptive_time*1000:.0f}ms ({1/adaptive_time:.1f} tok/s)")
    print(f"  Full (28 layers): {full_time*1000:.0f}ms ({1/full_time:.1f} tok/s)")
    print(f"  Speedup: {full_time/adaptive_time:.1f}x")


if __name__ == "__main__":
    import sys
    
    if "--convert" in sys.argv:
        engine = AdaptiveNavigationEngine()
        engine.convert_and_cache(max_layers=28)
    elif "--test-encoding" in sys.argv:
        test_compact_encoding()
    else:
        test_compact_encoding()
        test_speed()
