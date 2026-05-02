"""
Module 4: Selector — Geometric Spatial Filter
===============================================

Directional selection. Points the instrument at a specific position
in the input sequence by projecting all hidden states onto a single
direction vector and taking the argmax.

Optical analog: Spatial filter / aperture stop
Characteristic dimensionality: 1 (a single direction)

Specification:
    Selection rule:  argmax(h_i · d_k) over positions i
    Direction:       Single d-dimensional unit vector d_k
    Alignment:       cos(d_q, d_k) = 1.0 (same-feature detector)
    Accuracy:        Correct position for ALL entities in domain
    Storage:         1 direction vector (d params)
                     Ideal: 1 bit (all-negative → constant direction)
    Compute cost:    d multiplies + 1 argmax (2,869× cheaper than full attn)
"""

import numpy as np


class Selector:
    """Directional selection. Points the instrument at one position."""

    def __init__(self, d_k):
        """Initialize with the selection direction vector.
        
        Args:
            d_k: Direction vector [d_model]. Will be stored as-is
                 (not necessarily unit — the raw geometric direction).
        """
        self.d_k = np.asarray(d_k, dtype=np.float32).copy()
        self.d_model = self.d_k.shape[0]
        self._d_k_unit = self.d_k / np.linalg.norm(self.d_k)

    def select(self, hidden_states):
        """Select the position with maximum projection onto d_k.
        
        Args:
            hidden_states: [seq_len, d_model] hidden states at all positions.
            
        Returns:
            Index of the selected position (int).
        """
        scores = self.scores(hidden_states)
        return int(np.argmax(scores))

    def scores(self, hidden_states):
        """Compute raw selection scores for all positions.
        
        Args:
            hidden_states: [seq_len, d_model]
            
        Returns:
            [seq_len] array of projection scores.
        """
        return hidden_states @ self.d_k

    def margin(self, hidden_states, target_pos):
        """Measure selection margin: score(target) - score(next_best).
        
        Args:
            hidden_states: [seq_len, d_model]
            target_pos: Expected correct position
            
        Returns:
            dict with selected_pos, target_score, margin, correct
        """
        s = self.scores(hidden_states)
        selected = int(np.argmax(s))
        target_score = float(s[target_pos])
        
        # Margin = target score minus best non-target score
        mask = np.ones(len(s), dtype=bool)
        mask[target_pos] = False
        if mask.any():
            next_best = float(np.max(s[mask]))
        else:
            next_best = 0.0
        
        return {
            'selected_pos': selected,
            'target_score': target_score,
            'next_best_score': next_best,
            'margin': target_score - next_best,
            'correct': selected == target_pos,
        }

    @classmethod
    def from_model(cls, engine, layer_idx, head_idx):
        """Extract the Selector direction from a real model.
        
        The selector direction d_k is derived from the rank-1 SVD
        of the MESH (combined QK score matrix including biases).
        
        The MESH is computed from bias-inclusive weight matrices:
            W_q_b = W_q + b_q[:, None]  (bias added to every column)
            W_k_b = W_k + b_k[:, None]
            MESH = W_q_b @ W_k_b.T      (head_dim × head_dim)
        
        The bias outer product dominates (F45: 99.99% of MESH).
        d_k = W_k_b.T @ v₁ where v₁ is the top right singular vector.
        
        For the known capital-city circuit (L23 H6), d_k is
        effectively all-negative — meaning 1 bit suffices.
        
        Args:
            engine: PhiQwen2Engine instance
            layer_idx: Layer index (e.g. 23)
            head_idx: Head index (e.g. 6)
            
        Returns:
            Selector instance with d_k extracted from the model.
        """
        from phi_geometric.inference.phi_integer import phi_to_float
        
        attn = engine.layers[layer_idx].attention
        hd = attn.head_dim
        nh = attn.num_heads
        nkv = attn.num_kv_heads
        kv = head_idx // (nh // nkv)
        
        # Extract Q weights + bias for this head
        W_q = phi_to_float(attn.W_q.signs, attn.W_q.exponents)
        W_q_h = W_q[head_idx * hd:(head_idx + 1) * hd, :]  # [head_dim, d_model]
        b_q_h = attn.b_q[head_idx * hd:(head_idx + 1) * hd]  # [head_dim]
        
        # Extract K weights + bias for this head's KV group
        W_k = phi_to_float(attn.W_k.signs, attn.W_k.exponents)
        W_k_h = W_k[kv * hd:(kv + 1) * hd, :]  # [head_dim, d_model]
        b_k_h = attn.b_k[kv * hd:(kv + 1) * hd]  # [head_dim]
        
        # Bias-inclusive weight matrices (identity probing equivalent)
        # phi_linear(W, e_j, b) = W[:, j] + b for each basis vector e_j
        # So W_b = W + b[:, None] — bias added to every column
        W_q_b = W_q_h + b_q_h[:, None]  # [head_dim, d_model]
        W_k_b = W_k_h + b_k_h[:, None]  # [head_dim, d_model]
        
        # MESH in head space (includes bias — this is where rank-1 lives)
        MESH = W_q_b @ W_k_b.T  # [head_dim, head_dim]
        _, _, Vt = np.linalg.svd(MESH)
        v1 = Vt[0]  # top right singular vector [head_dim]
        
        # Project back to hidden space through bias-inclusive K
        d_k = W_k_b.T @ v1  # [d_model]
        
        return cls(d_k)

    @classmethod
    def from_one_bit(cls, d_model):
        """Create the ideal 1-bit selector (all-negative direction).
        
        F45 showed that d_k is all-negative for the capital-city circuit.
        This is the maximally compressed selector: 1 bit of information.
        
        Args:
            d_model: Dimensionality (e.g. 3584)
            
        Returns:
            Selector with d_k = all -1s.
        """
        return cls(-np.ones(d_model, dtype=np.float32))

    def spec(self):
        """Return specification measurements for this selector."""
        return {
            'd_model': self.d_model,
            'norm': float(np.linalg.norm(self.d_k)),
            'frac_negative': float(np.mean(self.d_k < 0)),
            'frac_positive': float(np.mean(self.d_k > 0)),
        }
