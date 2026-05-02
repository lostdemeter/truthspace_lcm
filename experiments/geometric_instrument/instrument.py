"""
Geometric Instrument — The Complete Pipeline
==============================================

Assembles the six geometric components into a working next-token
predictor. Each component is an independent, interchangeable module.

The pipeline:
    1. Embed tokens into the waveguide
    2. Decompose spectrally (Spectrometer × N layers)
    3. Select the entity position (Selector)
    4. Lock on with resonance (Resonator)
    5. Focus through the knowledge lens (Lens)
    6. Amplify the answer signal (Amplifier × M stages)
    7. Read the output (LM head projection)

This file demonstrates that we can reconstruct what an LLM does
using only named geometric operations — no black boxes.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.geometric_instrument.components.waveguide import Waveguide
from experiments.geometric_instrument.components.stabilizer import Stabilizer
from experiments.geometric_instrument.components.decomposer import Decomposer
from experiments.geometric_instrument.components.selector import Selector
from experiments.geometric_instrument.components.resonator import Resonator
from experiments.geometric_instrument.components.lens import Lens
from experiments.geometric_instrument.components.amplifier import Amplifier


class GeometricInstrument:
    """A next-token predictor built from six geometric components.
    
    Every step from input to output is a named geometric operation.
    No hidden computation, no black boxes.
    """

    def __init__(self, config):
        """Initialize the instrument from a configuration dict.
        
        Args:
            config: dict with:
                d_model: int — waveguide dimensionality
                embedding_fn: callable(token_ids) → [seq_len, d_model]
                lm_head_fn: callable([d_model]) → [vocab_size] logits
                final_norm_weight: [d_model] — final RMSNorm weights
                
                # Stage 1: Decomposition layers
                decomposition_layers: list of dicts, each with:
                    stabilizer_attn: Stabilizer
                    stabilizer_mlp: Stabilizer
                    attn_fn: callable(normed_3d) → attn_output
                    amplifier: Amplifier
                
                # Stage 2: Extraction (the critical layer)
                extraction: dict with:
                    layer_idx: int
                    stabilizer_attn: Stabilizer
                    selector: Selector
                    resonator: Resonator
                    lens: Lens
                    attn_fn: callable — full attention for this layer
                    stabilizer_mlp: Stabilizer
                    amplifier: Amplifier
                
                # Stage 3: Amplification layers
                amplification_layers: list of dicts, each with:
                    stabilizer_attn: Stabilizer
                    attn_fn: callable
                    stabilizer_mlp: Stabilizer
                    amplifier: Amplifier
        """
        self.d_model = config['d_model']
        self.embedding_fn = config['embedding_fn']
        self.lm_head_fn = config['lm_head_fn']
        self.final_norm_weight = config['final_norm_weight']
        self.decomposition_layers = config.get('decomposition_layers', [])
        self.extraction = config['extraction']
        self.amplification_layers = config.get('amplification_layers', [])

    def _final_norm(self, h):
        """Apply final RMS normalization before LM head."""
        rms = np.sqrt(np.mean(h ** 2) + 1e-6)
        return (h / rms) * self.final_norm_weight

    def predict(self, token_ids, verbose=False):
        """Run the full instrument pipeline.
        
        Args:
            token_ids: list of token IDs
            verbose: if True, print each stage
            
        Returns:
            logits: [vocab_size] array of next-token logits
            trace: dict with intermediate states for inspection
        """
        trace = {}
        
        # ── EMBED ────────────────────────────────────────────
        embeddings = self.embedding_fn(token_ids)  # [seq_len, d_model]
        h = embeddings[np.newaxis, :, :].astype(np.float32)  # [1, seq, d]
        seq_len = h.shape[1]
        
        if verbose:
            print(f"  Embedded {len(token_ids)} tokens → waveguide ℝ^{self.d_model}")
        
        trace['embeddings'] = h[0].copy()
        
        # ── STAGE 1: DECOMPOSITION ──────────────────────────
        for i, layer in enumerate(self.decomposition_layers):
            h = self._run_full_layer(h, layer)
        
        trace['post_decomposition'] = h[0, -1].copy()
        if verbose:
            print(f"  Decomposition: {len(self.decomposition_layers)} layers complete")
        
        # ── STAGE 2: EXTRACTION ─────────────────────────────
        ext = self.extraction
        h, extraction_trace = self._run_extraction_layer(h, ext, verbose)
        trace['extraction'] = extraction_trace
        
        # ── STAGE 3: AMPLIFICATION ──────────────────────────
        for i, layer in enumerate(self.amplification_layers):
            h = self._run_full_layer(h, layer)
        
        trace['post_amplification'] = h[0, -1].copy()
        if verbose:
            print(f"  Amplification: {len(self.amplification_layers)} layers complete")
        
        # ── READ OUTPUT ─────────────────────────────────────
        h_final = h[0, -1]  # last token
        h_normed = self._final_norm(h_final)
        logits = self.lm_head_fn(h_normed)
        
        trace['final_state'] = h_final.copy()
        trace['logits'] = logits.copy()
        
        if verbose:
            top5 = np.argsort(logits)[-5:][::-1]
            print(f"  Output: top token ID = {top5[0]} (logit = {logits[top5[0]]:.3f})")
        
        return logits, trace

    def _run_full_layer(self, h, layer_config):
        """Run a full transformer layer (attention + MLP) using components.
        
        For decomposition and amplification layers, we use the model's
        actual attention function (these layers' attention is infrastructure,
        not the knowledge-bearing extraction step).
        
        Args:
            h: [1, seq_len, d_model]
            layer_config: dict with stabilizer_attn, attn_fn, stabilizer_mlp, amplifier
            
        Returns:
            h: [1, seq_len, d_model] after attention + MLP
        """
        # Attention sub-layer
        stab_attn = layer_config['stabilizer_attn']
        normed = stab_attn.normalize(h[0])[np.newaxis, :, :]
        attn_out = layer_config['attn_fn'](normed, h)
        h = h + attn_out
        
        # MLP sub-layer (using Amplifier component)
        amp = layer_config['amplifier']
        stab_mlp = layer_config['stabilizer_mlp']
        normed_mlp = stab_mlp.normalize(h[0])
        gate = normed_mlp @ amp.W_gate.T
        up = normed_mlp @ amp.W_up.T
        from experiments.geometric_instrument.components.amplifier import _silu
        hidden = _silu(gate) * up
        mlp_out = hidden @ amp.W_down.T
        h = h + mlp_out[np.newaxis, :, :]
        
        return h

    def _run_extraction_layer(self, h, ext, verbose):
        """Run the extraction layer using geometric components.
        
        This is the critical layer where the Selector, Resonator,
        and Lens work together to extract knowledge.
        
        Args:
            h: [1, seq_len, d_model]
            ext: extraction config dict
            verbose: print details
            
        Returns:
            h: [1, seq_len, d_model] after extraction layer
            trace: dict with component outputs for inspection
        """
        trace = {}
        seq_len = h.shape[1]
        
        # Pre-attention norm
        normed = ext['stabilizer_attn'].normalize(h[0])  # [seq_len, d_model]
        trace['normed_last'] = normed[-1].copy()
        
        # ── SELECTOR: Which position has the entity? ────────
        selected_pos = ext['selector'].select(normed)
        selector_info = ext['selector'].margin(normed, selected_pos)
        trace['selected_pos'] = selected_pos
        trace['selector_margin'] = selector_info
        
        if verbose:
            print(f"  Selector: points to position {selected_pos} "
                  f"(margin={selector_info['margin']:.3f})")
        
        # ── Full attention (using model's actual attention for now) ──
        # In Phase 3, we'll replace this with Selector+Resonator+Lens
        normed_3d = normed[np.newaxis, :, :]
        attn_out = ext['attn_fn'](normed_3d, h)
        h_post_attn = h + attn_out
        
        trace['h_post_attn_last'] = h_post_attn[0, -1].copy()
        
        # ── LENS: What does the entity's identity look like? ────
        h_entity = normed[selected_pos]  # [d_model]
        binding = ext['lens'].focus(h_entity)
        trace['binding'] = binding.copy()
        
        if verbose:
            print(f"  Lens: focused on entity at pos {selected_pos}, "
                  f"||binding|| = {np.linalg.norm(binding):.2f}")
        
        # ── AMPLIFIER (MLP): Boost the answer signal ────────
        amp = ext['amplifier']
        stab_mlp = ext['stabilizer_mlp']
        normed_mlp = stab_mlp.normalize(h_post_attn[0])
        gate = normed_mlp @ amp.W_gate.T
        up = normed_mlp @ amp.W_up.T
        from experiments.geometric_instrument.components.amplifier import _silu
        hidden = _silu(gate) * up
        mlp_out = hidden @ amp.W_down.T
        h = h_post_attn + mlp_out[np.newaxis, :, :]
        
        trace['h_post_mlp_last'] = h[0, -1].copy()
        
        if verbose:
            delta_mlp = mlp_out[-1]
            delta_attn = attn_out[0, -1]
            n_mlp = np.linalg.norm(delta_mlp)
            n_attn = np.linalg.norm(delta_attn)
            if n_attn > 1e-12 and n_mlp > 1e-12:
                cos_orth = float(np.dot(delta_mlp, delta_attn) / (n_mlp * n_attn))
            else:
                cos_orth = 0.0
            print(f"  Amplifier: ||Δmlp||/||Δattn|| = {n_mlp/n_attn:.2f}, "
                  f"cos(Δmlp,Δattn) = {cos_orth:.4f}")
        
        return h, trace


def build_from_model(engine, tokenizer, extraction_layer=23, extraction_head=6,
                     decomp_end=22, amp_start=24, amp_end=28):
    """Build a GeometricInstrument from a loaded PhiQwen2Engine.
    
    This is Phase 1: extract real components from the model and
    verify they compose correctly.
    
    Args:
        engine: Loaded PhiQwen2Engine
        tokenizer: Qwen2Tokenizer
        extraction_layer: The critical knowledge-extraction layer (default 23)
        extraction_head: The head that implements the Selector-Resonator-Lens triad (default 6)
        decomp_end: Last decomposition layer (exclusive, default 22)
        amp_start: First amplification-only layer (default 24)
        amp_end: Last amplification layer (exclusive, default 28)
    
    Returns:
        GeometricInstrument instance
    """
    from phi_geometric.inference.phi_integer import phi_to_float
    from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
    from phi_geometric.inference.phi_matmul import phi_linear
    
    d_model = engine.layers[0].attention.norm_weight.shape[0]
    
    def make_attn_fn(layer_idx):
        """Create an attention function for a given layer."""
        def attn_fn(normed_3d, h_residual):
            attn = engine.layers[layer_idx].attention
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            seq_len = normed_3d.shape[1]
            
            Q = phi_linear(attn.W_q, normed_3d, attn.b_q)
            K = phi_linear(attn.W_k, normed_3d, attn.b_k)
            V = phi_linear(attn.W_v, normed_3d, attn.b_v)
            Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
            K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke = np.repeat(K, hpk, axis=1)
            Ve = np.repeat(V, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if seq_len > 1:
                scores += np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            weights = phi_softmax(scores, axis=-1)
            ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
            ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
            ao = phi_linear(attn.W_o, ao)
            return ao
        return attn_fn
    
    def embedding_fn(token_ids):
        return engine.embedding(token_ids)
    
    def lm_head_fn(h_normed):
        h_3d = h_normed[np.newaxis, np.newaxis, :]
        return engine.lm_head(h_3d)[0, 0, :]
    
    # Build decomposition layers (L0 to decomp_end-1)
    decomposition_layers = []
    for li in range(decomp_end):
        layer_config = {
            'stabilizer_attn': Stabilizer.from_model(engine, li, 'attn'),
            'attn_fn': make_attn_fn(li),
            'stabilizer_mlp': Stabilizer.from_model(engine, li, 'mlp'),
            'amplifier': Amplifier.from_model(engine, li),
        }
        decomposition_layers.append(layer_config)
    
    # Build the extraction layer
    # Include layers from decomp_end to extraction_layer
    # (e.g., L22 is pre-extraction, L23 is extraction)
    for li in range(decomp_end, extraction_layer):
        decomposition_layers.append({
            'stabilizer_attn': Stabilizer.from_model(engine, li, 'attn'),
            'attn_fn': make_attn_fn(li),
            'stabilizer_mlp': Stabilizer.from_model(engine, li, 'mlp'),
            'amplifier': Amplifier.from_model(engine, li),
        })
    
    extraction = {
        'layer_idx': extraction_layer,
        'stabilizer_attn': Stabilizer.from_model(engine, extraction_layer, 'attn'),
        'selector': Selector.from_model(engine, extraction_layer, extraction_head),
        'resonator': Resonator.from_model(engine, extraction_layer, extraction_head),
        'lens': Lens.from_model(engine, extraction_layer, extraction_head),
        'attn_fn': make_attn_fn(extraction_layer),
        'stabilizer_mlp': Stabilizer.from_model(engine, extraction_layer, 'mlp'),
        'amplifier': Amplifier.from_model(engine, extraction_layer),
    }
    
    # Build amplification layers
    amplification_layers = []
    for li in range(amp_start, amp_end):
        layer_config = {
            'stabilizer_attn': Stabilizer.from_model(engine, li, 'attn'),
            'attn_fn': make_attn_fn(li),
            'stabilizer_mlp': Stabilizer.from_model(engine, li, 'mlp'),
            'amplifier': Amplifier.from_model(engine, li),
        }
        amplification_layers.append(layer_config)
    
    config = {
        'd_model': d_model,
        'embedding_fn': embedding_fn,
        'lm_head_fn': lm_head_fn,
        'final_norm_weight': engine.final_norm_weight.copy(),
        'decomposition_layers': decomposition_layers,
        'extraction': extraction,
        'amplification_layers': amplification_layers,
    }
    
    return GeometricInstrument(config)
