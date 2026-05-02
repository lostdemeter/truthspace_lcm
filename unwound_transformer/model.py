"""
Unwound Qwen2-7B Model
======================

Complete implementation of Qwen2-7B using explicit matrix operations.
Designed for geometric analysis of the computation chain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

try:
    from .ops import rms_norm, compute_rope_embeddings, apply_rope, silu, softmax, gated_mlp
except ImportError:
    from ops import rms_norm, compute_rope_embeddings, apply_rope, silu, softmax, gated_mlp


@dataclass
class LayerTrace:
    """Trace of a single layer's computation for geometric analysis."""
    layer_idx: int
    input_hidden: np.ndarray
    post_norm_attn: np.ndarray
    attention_weights: Dict[int, np.ndarray]  # head -> weights
    attention_output: np.ndarray
    post_attn_residual: np.ndarray
    post_norm_mlp: np.ndarray
    mlp_gate: np.ndarray
    mlp_up: np.ndarray
    mlp_output: np.ndarray
    output_hidden: np.ndarray


@dataclass
class ForwardTrace:
    """Complete trace of forward pass for geometric analysis."""
    token_A: int
    token_B: int
    embedding_A: np.ndarray
    embedding_B: np.ndarray
    layer_traces: List[LayerTrace]
    final_hidden: np.ndarray
    logits: np.ndarray
    predicted_token: int


class UnwoundQwen2:
    """
    Qwen2-7B computed through explicit matrix operations.
    
    All computation is transparent and traceable for geometric analysis.
    """
    
    # Model constants
    HIDDEN_DIM = 3584
    N_LAYERS = 28
    N_HEADS = 28
    N_KV_HEADS = 4
    HEAD_DIM = 128
    HEADS_PER_KV = 7  # 28 // 4
    INTERMEDIATE_DIM = 18944
    VOCAB_SIZE = 152064
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """Load and extract weights from the model."""
        print(f"Loading {model_name}...")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,  # bfloat16 avoids NaN issues
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = next(self.hf_model.parameters()).device
        
        self._extract_weights()
        print(f"  Loaded {self.N_LAYERS} layers, {self.HIDDEN_DIM} hidden dim")
    
    def _extract_weights(self):
        """Extract all weights to float64 numpy arrays."""
        model = self.hf_model
        
        # Global weights
        self.embeddings = model.model.embed_tokens.weight.data.float().cpu().numpy().astype(np.float64)
        self.final_ln = model.model.norm.weight.data.float().cpu().numpy().astype(np.float64)
        self.lm_head = model.lm_head.weight.data.float().cpu().numpy().astype(np.float64)
        
        # Per-layer weights
        self.layers = []
        for i in range(self.N_LAYERS):
            layer = model.model.layers[i]
            attn = layer.self_attn
            
            L = {
                'W_q': attn.q_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'W_k': attn.k_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'W_v': attn.v_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'W_o': attn.o_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'b_q': attn.q_proj.bias.data.float().cpu().numpy().astype(np.float64),
                'b_k': attn.k_proj.bias.data.float().cpu().numpy().astype(np.float64),
                'b_v': attn.v_proj.bias.data.float().cpu().numpy().astype(np.float64),
                'ln_attn': layer.input_layernorm.weight.data.float().cpu().numpy().astype(np.float64),
                'ln_mlp': layer.post_attention_layernorm.weight.data.float().cpu().numpy().astype(np.float64),
                'W_gate': layer.mlp.gate_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'W_up': layer.mlp.up_proj.weight.data.float().cpu().numpy().astype(np.float64),
                'W_down': layer.mlp.down_proj.weight.data.float().cpu().numpy().astype(np.float64),
            }
            
            # Reshape for per-head access
            L['W_q_heads'] = L['W_q'].reshape(self.N_HEADS, self.HEAD_DIM, self.HIDDEN_DIM)
            L['W_k_heads'] = L['W_k'].reshape(self.N_KV_HEADS, self.HEAD_DIM, self.HIDDEN_DIM)
            L['W_v_heads'] = L['W_v'].reshape(self.N_KV_HEADS, self.HEAD_DIM, self.HIDDEN_DIM)
            L['b_q_heads'] = L['b_q'].reshape(self.N_HEADS, self.HEAD_DIM)
            L['b_k_heads'] = L['b_k'].reshape(self.N_KV_HEADS, self.HEAD_DIM)
            L['b_v_heads'] = L['b_v'].reshape(self.N_KV_HEADS, self.HEAD_DIM)
            
            self.layers.append(L)
        
        # RoPE frequencies
        layer0 = model.model.layers[0]
        if hasattr(layer0.self_attn, 'rotary_emb') and hasattr(layer0.self_attn.rotary_emb, 'inv_freq'):
            self.inv_freq = layer0.self_attn.rotary_emb.inv_freq.float().cpu().numpy().astype(np.float64)
        else:
            self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.HEAD_DIM, 2, dtype=np.float64) / self.HEAD_DIM))
    
    def compute_layer(self, layer_idx: int, h: np.ndarray, 
                      cos: np.ndarray, sin: np.ndarray,
                      trace: bool = False) -> Tuple[np.ndarray, Optional[LayerTrace]]:
        """
        Compute one transformer layer for 2-token input.
        
        Args:
            layer_idx: Layer index (0-27)
            h: Hidden states of shape (2, hidden_dim)
            cos, sin: RoPE embeddings
            trace: Whether to record computation trace
        
        Returns:
            Updated hidden states and optional trace
        """
        L = self.layers[layer_idx]
        h_A, h_B = h[0], h[1]
        
        # Pre-attention layer norm
        h_A_n = rms_norm(h_A, L['ln_attn'])
        h_B_n = rms_norm(h_B, L['ln_attn'])
        
        attn_out = np.zeros((2, self.HIDDEN_DIM))
        attn_weights_trace = {}
        
        # Position 0: self-attention only (causal)
        for head in range(self.N_HEADS):
            kv = head // self.HEADS_PER_KV
            v_A = h_A_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
            attn_out[0, head*self.HEAD_DIM:(head+1)*self.HEAD_DIM] = v_A
            if trace:
                attn_weights_trace[(0, head)] = np.array([1.0])
        
        # Position 1: attends to both
        for head in range(self.N_HEADS):
            kv = head // self.HEADS_PER_KV
            
            # Projections with bias
            q_B = h_B_n @ L['W_q_heads'][head].T + L['b_q_heads'][head]
            k_A = h_A_n @ L['W_k_heads'][kv].T + L['b_k_heads'][kv]
            k_B = h_B_n @ L['W_k_heads'][kv].T + L['b_k_heads'][kv]
            v_A = h_A_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
            v_B = h_B_n @ L['W_v_heads'][kv].T + L['b_v_heads'][kv]
            
            # RoPE
            q_B_r = apply_rope(q_B, cos[1], sin[1])
            k_A_r = apply_rope(k_A, cos[0], sin[0])
            k_B_r = apply_rope(k_B, cos[1], sin[1])
            
            # Attention scores
            s_A = np.dot(q_B_r, k_A_r) / np.sqrt(self.HEAD_DIM)
            s_B = np.dot(q_B_r, k_B_r) / np.sqrt(self.HEAD_DIM)
            weights = softmax(np.array([s_A, s_B]))
            
            if trace:
                attn_weights_trace[(1, head)] = weights
            
            v_out = weights[0] * v_A + weights[1] * v_B
            attn_out[1, head*self.HEAD_DIM:(head+1)*self.HEAD_DIM] = v_out
        
        # Output projection
        attn_out[0] = attn_out[0] @ L['W_o'].T
        attn_out[1] = attn_out[1] @ L['W_o'].T
        
        # Residual
        h_post = h + attn_out
        
        # MLP
        mlp_out = np.zeros((2, self.HIDDEN_DIM))
        mlp_gates = []
        mlp_ups = []
        
        for p in range(2):
            h_n = rms_norm(h_post[p], L['ln_mlp'])
            gate = h_n @ L['W_gate'].T
            up = h_n @ L['W_up'].T
            mlp_out[p] = (silu(gate) * up) @ L['W_down'].T
            if trace:
                mlp_gates.append(gate)
                mlp_ups.append(up)
        
        output = h_post + mlp_out
        
        if trace:
            layer_trace = LayerTrace(
                layer_idx=layer_idx,
                input_hidden=h[1].copy(),
                post_norm_attn=h_B_n.copy(),
                attention_weights=attn_weights_trace,
                attention_output=attn_out[1].copy(),
                post_attn_residual=h_post[1].copy(),
                post_norm_mlp=rms_norm(h_post[1], L['ln_mlp']),
                mlp_gate=mlp_gates[1] if mlp_gates else None,
                mlp_up=mlp_ups[1] if mlp_ups else None,
                mlp_output=mlp_out[1].copy(),
                output_hidden=output[1].copy()
            )
            return output, layer_trace
        
        return output, None
    
    def forward(self, token_A: int, token_B: int) -> int:
        """
        Forward pass returning predicted next token.
        
        Args:
            token_A: First token ID
            token_B: Second token ID
        
        Returns:
            Predicted next token ID
        """
        h = np.stack([self.embeddings[token_A], self.embeddings[token_B]])
        cos, sin = compute_rope_embeddings(2, self.inv_freq)
        
        for i in range(self.N_LAYERS):
            h, _ = self.compute_layer(i, h, cos, sin, trace=False)
        
        h_final = rms_norm(h[1], self.final_ln)
        logits = self.lm_head @ h_final
        
        return int(np.argmax(logits))
    
    def forward_with_trace(self, token_A: int, token_B: int) -> ForwardTrace:
        """
        Forward pass with full computation trace for geometric analysis.
        
        Args:
            token_A: First token ID
            token_B: Second token ID
        
        Returns:
            ForwardTrace containing all intermediate values
        """
        emb_A = self.embeddings[token_A].copy()
        emb_B = self.embeddings[token_B].copy()
        
        h = np.stack([emb_A, emb_B])
        cos, sin = compute_rope_embeddings(2, self.inv_freq)
        
        layer_traces = []
        for i in range(self.N_LAYERS):
            h, trace = self.compute_layer(i, h, cos, sin, trace=True)
            layer_traces.append(trace)
        
        h_final = rms_norm(h[1], self.final_ln)
        logits = self.lm_head @ h_final
        predicted = int(np.argmax(logits))
        
        return ForwardTrace(
            token_A=token_A,
            token_B=token_B,
            embedding_A=emb_A,
            embedding_B=emb_B,
            layer_traces=layer_traces,
            final_hidden=h_final,
            logits=logits,
            predicted_token=predicted
        )
    
    def validate_against_model(self, n_samples: int = 20, verbose: bool = False) -> Tuple[int, int]:
        """
        Validate unwound computation against HuggingFace model.
        
        Returns:
            (correct, total) counts
        """
        correct = 0
        valid = 0
        np.random.seed(42)
        
        for i in range(n_samples * 3):  # Try more to get enough valid samples
            if valid >= n_samples:
                break
                
            # With bfloat16, we can use a wider range
            A = np.random.randint(100, 10000)
            B = np.random.randint(100, 10000)
            
            # HF model prediction
            ids = torch.tensor([[A, B]]).to(self.device)
            with torch.no_grad():
                out = self.hf_model(ids, output_hidden_states=True)
            
            # Check for NaN in hidden states (float16 overflow)
            last_hidden = out.hidden_states[-1][0, 1].float().cpu().numpy()
            if np.isnan(last_hidden).any():
                if verbose:
                    print(f"  Skipping ({A}, {B}) - NaN in model output")
                continue
            
            valid += 1
            actual = torch.argmax(out.logits[0, 1]).item()
            
            # Our prediction
            pred = self.forward(A, B)
            
            if actual == pred:
                correct += 1
            elif verbose:
                print(f"  Mismatch ({A}, {B}): actual={actual}, pred={pred}")
        
        return correct, valid
    
    def decode_token(self, token_id: int) -> str:
        """Decode a token ID to string."""
        return self.tokenizer.decode([token_id])
