"""
Geometric Analysis Tools for Unwound Transformer
=================================================

Tools for examining the geometric structure of transformer computation:
- Hidden state trajectories
- Attention pattern geometry
- Weight matrix structure
- φ-level analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618


@dataclass
class GeometricAnalysis:
    """Results of geometric analysis on a forward trace."""
    # Hidden state trajectory
    hidden_norms: List[float]
    hidden_directions: List[np.ndarray]  # Unit vectors
    direction_changes: List[float]  # Cosine between consecutive
    
    # Attention geometry
    attention_entropy: Dict[int, List[float]]  # layer -> per-head entropy
    attention_focus: Dict[int, List[float]]  # layer -> per-head max weight
    
    # φ-level analysis
    phi_levels: List[int]  # Which φ-level each layer output is at
    phi_residuals: List[float]  # Distance from nearest φ-level


def analyze_trace(trace) -> GeometricAnalysis:
    """
    Perform geometric analysis on a forward trace.
    
    Args:
        trace: ForwardTrace from model.forward_with_trace()
    
    Returns:
        GeometricAnalysis with all computed metrics
    """
    # Hidden state trajectory
    hidden_norms = []
    hidden_directions = []
    
    # Start with embedding
    emb_norm = np.linalg.norm(trace.embedding_B)
    hidden_norms.append(emb_norm)
    hidden_directions.append(trace.embedding_B / (emb_norm + 1e-10))
    
    for lt in trace.layer_traces:
        norm = np.linalg.norm(lt.output_hidden)
        hidden_norms.append(norm)
        hidden_directions.append(lt.output_hidden / (norm + 1e-10))
    
    # Direction changes (cosine similarity between consecutive)
    direction_changes = []
    for i in range(1, len(hidden_directions)):
        cos = np.dot(hidden_directions[i-1], hidden_directions[i])
        direction_changes.append(cos)
    
    # Attention geometry
    attention_entropy = {}
    attention_focus = {}
    
    for lt in trace.layer_traces:
        layer_entropy = []
        layer_focus = []
        
        for head in range(28):
            weights = lt.attention_weights.get((1, head), np.array([1.0]))
            
            # Entropy: -Σ p log p
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            layer_entropy.append(entropy)
            
            # Focus: max weight
            layer_focus.append(np.max(weights))
        
        attention_entropy[lt.layer_idx] = layer_entropy
        attention_focus[lt.layer_idx] = layer_focus
    
    # φ-level analysis
    phi_levels = []
    phi_residuals = []
    
    for norm in hidden_norms:
        if norm > 0:
            # What power of φ is this closest to?
            log_phi = np.log(norm) / np.log(PHI)
            level = int(round(log_phi))
            residual = abs(log_phi - level)
            phi_levels.append(level)
            phi_residuals.append(residual)
        else:
            phi_levels.append(0)
            phi_residuals.append(0)
    
    return GeometricAnalysis(
        hidden_norms=hidden_norms,
        hidden_directions=hidden_directions,
        direction_changes=direction_changes,
        attention_entropy=attention_entropy,
        attention_focus=attention_focus,
        phi_levels=phi_levels,
        phi_residuals=phi_residuals
    )


def compute_trajectory_curvature(directions: List[np.ndarray]) -> List[float]:
    """
    Compute curvature of the hidden state trajectory.
    
    Curvature = rate of change of direction.
    High curvature = sharp turn in hidden space.
    """
    curvatures = []
    for i in range(1, len(directions) - 1):
        # Second derivative approximation
        d1 = directions[i] - directions[i-1]
        d2 = directions[i+1] - directions[i]
        curvature = np.linalg.norm(d2 - d1)
        curvatures.append(curvature)
    return curvatures


def find_attention_anchors(analysis: GeometricAnalysis, threshold: float = 0.8) -> Dict[int, List[int]]:
    """
    Find layers where attention is highly focused (potential anchors).
    
    Args:
        analysis: GeometricAnalysis from analyze_trace()
        threshold: Focus threshold (max attention weight)
    
    Returns:
        Dict mapping layer -> list of focused heads
    """
    anchors = {}
    for layer, focus_values in analysis.attention_focus.items():
        focused_heads = [h for h, f in enumerate(focus_values) if f > threshold]
        if focused_heads:
            anchors[layer] = focused_heads
    return anchors


def compute_layer_similarity_matrix(model) -> np.ndarray:
    """
    Compute similarity between layers based on weight structure.
    
    Returns:
        (28, 28) matrix of layer similarities
    """
    similarities = np.zeros((28, 28))
    
    for i in range(28):
        for j in range(28):
            # Compare W_q matrices (flattened)
            w_i = model.layers[i]['W_q'].flatten()
            w_j = model.layers[j]['W_q'].flatten()
            
            cos = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j) + 1e-10)
            similarities[i, j] = cos
    
    return similarities


def analyze_weight_spectrum(weight: np.ndarray) -> Dict:
    """
    Analyze the singular value spectrum of a weight matrix.
    
    Returns:
        Dict with spectrum statistics
    """
    U, S, Vt = np.linalg.svd(weight, full_matrices=False)
    
    # Normalize singular values
    S_norm = S / S[0]
    
    # Find effective rank (where singular values drop below threshold)
    threshold = 0.01
    effective_rank = np.sum(S_norm > threshold)
    
    # φ-level analysis of singular values
    phi_aligned = []
    for s in S[:20]:  # Top 20
        log_phi = np.log(s + 1e-10) / np.log(PHI)
        level = round(log_phi)
        residual = abs(log_phi - level)
        phi_aligned.append(residual < 0.1)
    
    return {
        'singular_values': S,
        'effective_rank': effective_rank,
        'condition_number': S[0] / (S[-1] + 1e-10),
        'phi_aligned_count': sum(phi_aligned),
        'top_20_phi_aligned': phi_aligned
    }


def project_to_subspace(hidden: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Project hidden state onto a subspace defined by basis vectors.
    
    Args:
        hidden: Hidden state vector
        basis: (k, d) matrix of k basis vectors
    
    Returns:
        (projection, residual_norm)
    """
    # Orthonormalize basis
    Q, R = np.linalg.qr(basis.T)
    
    # Project
    coords = Q.T @ hidden
    projection = Q @ coords
    residual = hidden - projection
    
    return projection, np.linalg.norm(residual)


def find_geometric_invariants(traces: List) -> Dict:
    """
    Find geometric properties that are invariant across different inputs.
    
    Args:
        traces: List of ForwardTrace objects
    
    Returns:
        Dict of invariant properties
    """
    analyses = [analyze_trace(t) for t in traces]
    
    # Check if direction changes follow a pattern
    all_changes = np.array([a.direction_changes for a in analyses])
    mean_changes = np.mean(all_changes, axis=0)
    std_changes = np.std(all_changes, axis=0)
    
    # Layers with consistent direction change
    consistent_layers = np.where(std_changes < 0.05)[0]
    
    # Check if norm growth follows φ pattern
    all_norms = np.array([a.hidden_norms for a in analyses])
    norm_ratios = all_norms[:, 1:] / (all_norms[:, :-1] + 1e-10)
    mean_ratios = np.mean(norm_ratios, axis=0)
    
    # Check if ratios are close to φ or 1/φ
    phi_ratios = np.abs(mean_ratios - PHI) < 0.1
    inv_phi_ratios = np.abs(mean_ratios - 1/PHI) < 0.1
    
    return {
        'consistent_direction_layers': consistent_layers.tolist(),
        'mean_direction_changes': mean_changes.tolist(),
        'mean_norm_ratios': mean_ratios.tolist(),
        'phi_aligned_layers': np.where(phi_ratios)[0].tolist(),
        'inv_phi_aligned_layers': np.where(inv_phi_ratios)[0].tolist()
    }


def print_analysis_summary(analysis: GeometricAnalysis):
    """Print a summary of geometric analysis."""
    print("\n=== Geometric Analysis Summary ===")
    
    print("\n--- Hidden State Trajectory ---")
    print(f"  Initial norm: {analysis.hidden_norms[0]:.4f}")
    print(f"  Final norm: {analysis.hidden_norms[-1]:.4f}")
    print(f"  Growth factor: {analysis.hidden_norms[-1] / (analysis.hidden_norms[0] + 1e-10):.2f}x")
    
    print("\n--- Direction Changes (cosine) ---")
    for i in range(0, len(analysis.direction_changes), 7):
        layer_range = analysis.direction_changes[i:i+7]
        print(f"  Layers {i}-{i+len(layer_range)-1}: {[f'{c:.3f}' for c in layer_range]}")
    
    print("\n--- φ-Level Analysis ---")
    print(f"  φ-levels: {analysis.phi_levels}")
    mean_residual = np.mean(analysis.phi_residuals)
    print(f"  Mean φ-residual: {mean_residual:.4f} (0 = perfect φ-alignment)")
    
    print("\n--- Attention Focus ---")
    high_focus_layers = []
    for layer, focus in analysis.attention_focus.items():
        max_focus = max(focus)
        if max_focus > 0.9:
            high_focus_layers.append((layer, max_focus))
    
    if high_focus_layers:
        print(f"  High focus layers (>0.9): {high_focus_layers}")
    else:
        print("  No layers with focus > 0.9")
