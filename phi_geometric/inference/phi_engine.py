"""
PhiQwen2Engine: Complete φ-integer inference engine for Qwen2-7B.

Phase 3: End-to-end autoregressive generation with KV caching.

Components:
  - φ-matmul (sign XOR + exp ADD + LUT, or hybrid decode+numpy)
  - φ-softmax (φ^(x/T) / Σ φ^(x/T) — exact equivalent)
  - φ-SiLU (x × sigmoid(x))
  - RMSNorm (float — magnitude operation, not structural)
  - RoPE (rotary position embeddings)
  - KV cache (incremental decode — process one token at a time)

Architecture: Qwen2-7B
  - 28 layers, 28 Q heads, 4 KV heads (GQA 7:1)
  - hidden_dim=3584, head_dim=128, intermediate=18944
  - vocab=152064, rope_theta=1_000_000

Usage:
    engine = PhiQwen2Engine.load("path/to/phi_model")
    logits = engine.forward([token_id_1, token_id_2, ...])
    tokens = engine.generate(prompt_ids, max_new_tokens=50)
"""

import os
import json
import time
import numpy as np
from typing import List, Optional, Callable

from .phi_types import PhiEncoded
from .phi_components import PhiEmbedding, PhiLMHead, rms_norm
from .phi_attention import PhiAttention, RoPE, KVCache
from .phi_mlp import PhiMLP


class PhiTransformerLayer:
    """One transformer layer: attention + MLP."""

    def __init__(self, attention: PhiAttention, mlp: PhiMLP, layer_idx: int):
        self.attention = attention
        self.mlp = mlp
        self.layer_idx = layer_idx

    def __call__(self, hidden: np.ndarray, pure: bool = False,
                 kv_cache: Optional[KVCache] = None) -> np.ndarray:
        hidden = self.attention(hidden, pure=pure,
                                kv_cache=kv_cache, layer_idx=self.layer_idx)
        hidden = self.mlp(hidden, pure=pure)
        return hidden

    def warm_weights(self):
        """Pre-decode all φ-encoded weights for this layer."""
        for w in [self.attention.W_q, self.attention.W_k,
                  self.attention.W_v, self.attention.W_o,
                  self.mlp.W_gate, self.mlp.W_up, self.mlp.W_down]:
            w.warm_cache()

    def clear_weight_cache(self):
        """Free cached decoded weights for this layer."""
        for w in [self.attention.W_q, self.attention.W_k,
                  self.attention.W_v, self.attention.W_o,
                  self.mlp.W_gate, self.mlp.W_up, self.mlp.W_down]:
            w.clear_cache()


class PhiQwen2Engine:
    """
    Complete Qwen2-7B inference engine using φ-encoded weights.

    No GPU required. No PyTorch required. Pure NumPy.

    Memory strategy:
      Phase 2 (single forward):
        - Decode weights on-the-fly, discard after use
      Phase 3 (generation with KV cache):
        - warm_weights() pre-decodes all weights (~28 GB float32)
        - KV cache grows incrementally (~4 MB/layer per 1024 tokens)
        - After generation, clear_weight_cache() frees decoded weights

    For machines with <40 GB RAM, use without warm_weights() —
    each layer will decode per call (slower but uses less memory).
    """

    def __init__(self):
        self.embedding: Optional[PhiEmbedding] = None
        self.layers: List[PhiTransformerLayer] = []
        self.final_norm_weight: Optional[np.ndarray] = None
        self.lm_head: Optional[PhiLMHead] = None
        self.rope: Optional[RoPE] = None

        # Config
        self.hidden_dim = 3584
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_dim = 128
        self.intermediate_size = 18944
        self.vocab_size = 152064
        self.num_layers = 28
        self.rope_theta = 1_000_000.0

    @classmethod
    def load(cls, model_dir: str, max_layers: int = None,
             max_seq_len: int = 4096, verbose: bool = True) -> 'PhiQwen2Engine':
        """
        Load φ-encoded Qwen2-7B from disk.

        Args:
            model_dir: path to phi_model/ directory
            max_layers: load only first N layers (for testing)
            max_seq_len: max sequence length for RoPE pre-computation
            verbose: print progress

        Returns:
            Loaded engine ready for inference
        """
        engine = cls()

        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        engine.hidden_dim = config['hidden_size']
        engine.num_heads = config['num_attention_heads']
        engine.num_kv_heads = config['num_key_value_heads']
        engine.head_dim = config['head_dim']
        engine.intermediate_size = config['intermediate_size']
        engine.vocab_size = config['vocab_size']
        engine.num_layers = config['num_hidden_layers']
        engine.rope_theta = config['rope_theta']

        n_layers = max_layers if max_layers is not None else engine.num_layers

        if verbose:
            print(f"Loading φ-encoded Qwen2-7B from {model_dir}")
            print(f"  Config: {engine.hidden_dim}d, {engine.num_heads}h, "
                  f"{engine.num_kv_heads}kv, {n_layers} layers")

        # Initialize RoPE (shared across all layers)
        engine.rope = RoPE(engine.head_dim, engine.rope_theta, max_seq_len)
        if verbose:
            print(f"  RoPE: theta={engine.rope_theta:.0f}, max_seq={max_seq_len}")

        # Load embeddings
        t0 = time.time()
        emb_enc = PhiEncoded.load(os.path.join(model_dir, 'embed_tokens.npz'))
        engine.embedding = PhiEmbedding(emb_enc)
        if verbose:
            print(f"  Embeddings: {engine.embedding.vocab_size}×{engine.embedding.hidden_dim} "
                  f"({time.time()-t0:.1f}s)")

        # Load layers
        for i in range(n_layers):
            t0 = time.time()
            layer_dir = os.path.join(model_dir, f'layer_{i:02d}')

            # Load φ-encoded weights
            W_q = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
            W_k = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
            W_v = PhiEncoded.load(os.path.join(layer_dir, 'v_proj.npz'))
            W_o = PhiEncoded.load(os.path.join(layer_dir, 'o_proj.npz'))
            W_gate = PhiEncoded.load(os.path.join(layer_dir, 'gate_proj.npz'))
            W_up = PhiEncoded.load(os.path.join(layer_dir, 'up_proj.npz'))
            W_down = PhiEncoded.load(os.path.join(layer_dir, 'down_proj.npz'))

            # Load float biases and norms
            biases = np.load(os.path.join(layer_dir, 'biases.npz'))
            norms = np.load(os.path.join(layer_dir, 'norms.npz'))

            b_q = biases['q_proj_bias']
            b_k = biases['k_proj_bias']
            b_v = biases['v_proj_bias']
            ln1_weight = norms['input_layernorm']
            ln2_weight = norms['post_attention_layernorm']

            # Build attention block
            attention = PhiAttention(
                W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
                b_q=b_q, b_k=b_k, b_v=b_v,
                norm_weight=ln1_weight, rope=engine.rope,
                num_heads=engine.num_heads,
                num_kv_heads=engine.num_kv_heads,
                head_dim=engine.head_dim,
            )

            # Build MLP block
            mlp = PhiMLP(
                W_gate=W_gate, W_up=W_up, W_down=W_down,
                norm_weight=ln2_weight,
            )

            layer = PhiTransformerLayer(attention, mlp, i)
            engine.layers.append(layer)

            if verbose:
                print(f"  Layer {i:2d}: loaded ({time.time()-t0:.1f}s)")

        # Load final norm
        final_norm_data = np.load(os.path.join(model_dir, 'final_norm.npz'))
        engine.final_norm_weight = final_norm_data['weight']

        # Load LM head
        t0 = time.time()
        lm_head_enc = PhiEncoded.load(os.path.join(model_dir, 'lm_head.npz'))
        engine.lm_head = PhiLMHead(lm_head_enc)
        if verbose:
            print(f"  LM Head: {engine.vocab_size}×{engine.hidden_dim} ({time.time()-t0:.1f}s)")
            print(f"  Done! {n_layers} layers loaded.")

        return engine

    def warm_weights(self, verbose: bool = True):
        """
        Pre-decode all φ-encoded weights to float32.

        This caches decoded weights so subsequent forward passes skip decoding.
        Uses ~28 GB additional RAM for full 28-layer model.
        Call before generate() for best performance.
        """
        t0 = time.time()
        if verbose:
            print("Warming weight caches...")
        for layer in self.layers:
            layer.warm_weights()
            if verbose:
                print(f"  Layer {layer.layer_idx:2d}: warmed")
        self.lm_head.weight.warm_cache()
        if verbose:
            print(f"  LM Head: warmed")
            print(f"  Done! ({time.time()-t0:.1f}s)")

    def clear_weight_cache(self):
        """Free all cached decoded weights."""
        for layer in self.layers:
            layer.clear_weight_cache()
        self.lm_head.weight.clear_cache()

    def forward(self, token_ids: List[int], pure: bool = False,
                verbose: bool = False,
                kv_cache: Optional[KVCache] = None) -> np.ndarray:
        """
        Full forward pass: tokens → logits.

        Args:
            token_ids: list of token IDs
            pure: use pure φ-integer matmul (slow but proves integer arith)
            verbose: print per-layer timing
            kv_cache: if provided, use/update KV cache

        Returns:
            logits: (1, seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        total_start = time.time()

        # Embedding lookup (no matmul needed)
        hidden = self.embedding(token_ids)  # (seq_len, hidden_dim)
        hidden = hidden[np.newaxis, :, :]   # (1, seq_len, hidden_dim)

        if verbose:
            print(f"  Embed: {hidden.shape}")

        # Transformer layers
        for layer in self.layers:
            t0 = time.time()
            hidden = layer(hidden, pure=pure, kv_cache=kv_cache)
            if verbose:
                dt = time.time() - t0
                h_norm = np.linalg.norm(hidden)
                print(f"  Layer {layer.layer_idx:2d}: {dt:.2f}s  |hidden|={h_norm:.1f}")

        # Final RMSNorm
        hidden = rms_norm(hidden, self.final_norm_weight)

        # LM head: hidden → logits
        t0 = time.time()
        logits = self.lm_head(hidden, pure=pure)
        if verbose:
            print(f"  LM Head: {time.time()-t0:.2f}s")
            print(f"  Total: {time.time()-total_start:.2f}s")

        return logits

    def forward_with_hidden_states(self, token_ids: List[int],
                                   pure: bool = False) -> List[np.ndarray]:
        """
        Forward pass that returns hidden states after each layer.

        Returns:
            List of (1, seq_len, hidden_dim) arrays.
            Index 0 = post-embedding, index i+1 = post-layer-i.
        """
        hidden = self.embedding(token_ids)
        hidden = hidden[np.newaxis, :, :]

        states = [hidden.copy()]
        for layer in self.layers:
            hidden = layer(hidden, pure=pure)
            states.append(hidden.copy())

        return states

    def predict_next(self, token_ids: List[int], pure: bool = False,
                     top_k: int = 5) -> List[tuple]:
        """
        Predict next token probabilities.

        Returns:
            List of (token_id, logit_value) tuples, sorted by logit descending.
        """
        logits = self.forward(token_ids, pure=pure)
        last_logits = logits[0, -1, :]  # (vocab_size,)

        top_indices = np.argsort(last_logits)[-top_k:][::-1]
        return [(int(idx), float(last_logits[idx])) for idx in top_indices]

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 20,
                 pure: bool = False, verbose: bool = False,
                 eos_token_id: int = 151643,
                 token_callback: Optional[Callable[[int, int], None]] = None
                 ) -> List[int]:
        """
        Autoregressive text generation with KV caching.

        Two phases:
          1. Prefill:  process entire prompt, build KV cache
          2. Decode:   one token at a time, extend KV cache

        After prefill, each decode step only runs 1 token through the model.
        Weight decode caching (decode_cached) means weights are decoded once.

        Args:
            prompt_ids: list of prompt token IDs
            max_new_tokens: maximum tokens to generate
            pure: use pure φ-integer matmul
            verbose: print per-token timing
            eos_token_id: stop generation at this token
            token_callback: fn(step, token_id) called after each decode step

        Returns:
            Full sequence including prompt + generated tokens
        """
        n_layers = len(self.layers)
        kv_cache = KVCache(n_layers, self.num_kv_heads, self.head_dim)

        generated = list(prompt_ids)

        # Phase 1: Prefill — process entire prompt to build KV cache
        t0 = time.time()
        logits = self.forward(prompt_ids, pure=pure, kv_cache=kv_cache)
        prefill_time = time.time() - t0

        next_token = int(np.argmax(logits[0, -1, :]))

        if verbose:
            print(f"  Prefill: {len(prompt_ids)} tokens in {prefill_time:.1f}s "
                  f"({prefill_time/len(prompt_ids):.2f}s/tok)")

        if next_token == eos_token_id:
            return generated

        generated.append(next_token)
        if token_callback:
            token_callback(0, next_token)

        # Phase 2: Decode — one token at a time with KV cache
        for step in range(1, max_new_tokens):
            t0 = time.time()

            # Only process the LAST token — KV cache has everything before it
            logits = self.forward([next_token], pure=pure, kv_cache=kv_cache)
            next_token = int(np.argmax(logits[0, -1, :]))
            dt = time.time() - t0

            if verbose:
                print(f"  Decode step {step}: token={next_token} ({dt:.2f}s)")

            if next_token == eos_token_id:
                break

            generated.append(next_token)
            if token_callback:
                token_callback(step, next_token)

        return generated
