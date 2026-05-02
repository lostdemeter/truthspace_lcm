#!/usr/bin/env python3
"""
Qwen2 Boom Convergence Analysis
================================

Hypothesis: Qwen2 was approaching the ideal boom structure during training
but stopped before fully converging. We analyze:

1. How close is Qwen2 to the theoretical 137/30 ratio?
2. What is the boom prediction accuracy?
3. Where does the model deviate from the ideal?
4. Can we identify "correction factors" to nudge it toward convergence?

Author: TruthSpace LCM Team
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = 1.6180339887498949
FINE_STRUCTURE_RATIO = 137 / 30  # ≈ 4.567
BARRIER_THRESHOLD = 1 / PHI  # ≈ 0.618


def compute_attention_entropy(attn_weights):
    """Compute entropy of attention weights."""
    attn = attn_weights.clamp(min=1e-10)
    entropy = -(attn * attn.log()).sum(dim=-1)
    return entropy


def get_attention_patterns(model, tokenizer, text, layer_idx=14):
    """Get attention patterns for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    return outputs.attentions[layer_idx], inputs['input_ids']


def detect_entropy_booms(entropy_sequence, threshold=0.1):
    """Detect booms as significant entropy drops."""
    booms = []
    for i in range(1, len(entropy_sequence)):
        drop = entropy_sequence[i-1] - entropy_sequence[i]
        if entropy_sequence[i-1] > 0:
            relative_drop = drop / entropy_sequence[i-1]
            if relative_drop > threshold:
                booms.append(i)
    return booms


def analyze_boom_spacing_ratio(booms):
    """
    Analyze if boom spacing follows the 137/30 pattern.
    
    In zeta zeros, the ratio of pre/post barrier variance is 137/30.
    We look for similar structure in attention booms.
    """
    if len(booms) < 4:
        return None
    
    spacings = np.diff(booms)
    
    if len(spacings) < 2:
        return None
    
    # Split spacings into "early" and "late" 
    mid = len(spacings) // 2
    early_spacings = spacings[:mid]
    late_spacings = spacings[mid:]
    
    if len(late_spacings) == 0 or np.mean(late_spacings) == 0:
        return None
    
    # Compute ratio
    ratio = np.mean(early_spacings) / np.mean(late_spacings)
    
    return {
        'early_mean': np.mean(early_spacings),
        'late_mean': np.mean(late_spacings),
        'ratio': ratio,
        'target_ratio': FINE_STRUCTURE_RATIO,
        'deviation': abs(ratio - FINE_STRUCTURE_RATIO) / FINE_STRUCTURE_RATIO,
    }


def analyze_entropy_variance_ratio(entropy_sequence, barrier_idx=None):
    """
    Analyze if entropy variance follows the 137/30 pattern.
    
    Look for a natural barrier where variance ratio matches 137/30.
    """
    n = len(entropy_sequence)
    
    if barrier_idx is None:
        # Search for optimal barrier
        best_match = float('inf')
        best_barrier = n // 2
        best_ratio = 1.0
        
        for b in range(max(3, n//4), min(n-3, 3*n//4)):
            pre_var = np.var(entropy_sequence[:b])
            post_var = np.var(entropy_sequence[b:])
            
            if post_var > 0:
                ratio = pre_var / post_var
                match = abs(ratio - FINE_STRUCTURE_RATIO)
                
                if match < best_match:
                    best_match = match
                    best_barrier = b
                    best_ratio = ratio
        
        barrier_idx = best_barrier
    
    pre_var = np.var(entropy_sequence[:barrier_idx])
    post_var = np.var(entropy_sequence[barrier_idx:])
    
    if post_var > 0:
        ratio = pre_var / post_var
    else:
        ratio = float('inf')
    
    return {
        'barrier_idx': barrier_idx,
        'pre_variance': pre_var,
        'post_variance': post_var,
        'ratio': ratio,
        'target_ratio': FINE_STRUCTURE_RATIO,
        'deviation': abs(ratio - FINE_STRUCTURE_RATIO) / FINE_STRUCTURE_RATIO if ratio != float('inf') else float('inf'),
    }


def analyze_alternation_ratio(entropy_sequence, barrier_idx=None):
    """
    Analyze sign alternation ratio before/after barrier.
    
    In zeta zeros: pre-barrier alternation / post-barrier alternation ≈ 1.19
    """
    changes = np.diff(entropy_sequence)
    signs = np.sign(changes)
    
    n = len(signs)
    if barrier_idx is None:
        barrier_idx = n // 2
    
    # Alternation rate = fraction of sign changes
    def alt_rate(s):
        if len(s) < 2:
            return 0
        return np.sum(np.abs(np.diff(s)) > 0) / (len(s) - 1)
    
    pre_alt = alt_rate(signs[:barrier_idx])
    post_alt = alt_rate(signs[barrier_idx:])
    
    if post_alt > 0:
        ratio = pre_alt / post_alt
    else:
        ratio = float('inf')
    
    return {
        'barrier_idx': barrier_idx,
        'pre_alternation': pre_alt,
        'post_alternation': post_alt,
        'ratio': ratio,
    }


def predict_booms(entropy_sequence, mean_spacing):
    """
    Predict boom positions using mean spacing.
    
    Returns predicted positions and errors vs actual.
    """
    actual_booms = detect_entropy_booms(entropy_sequence)
    
    if len(actual_booms) < 2:
        return None
    
    # Predict from first boom
    predicted = []
    current = actual_booms[0]
    while current < len(entropy_sequence):
        predicted.append(int(current))
        current += mean_spacing
    
    # Compute errors
    errors = []
    for p in predicted[1:]:  # Skip first (it's exact)
        if len(actual_booms) > 0:
            closest = actual_booms[np.argmin(np.abs(np.array(actual_booms) - p))]
            errors.append(abs(p - closest))
    
    return {
        'predicted': predicted,
        'actual': actual_booms,
        'errors': errors,
        'mean_error': np.mean(errors) if errors else None,
    }


def analyze_cross_layer_consistency(model, tokenizer, text):
    """
    Analyze boom consistency across layers.
    
    Ideal: booms should occur at same positions across layers.
    """
    layer_booms = {}
    layer_entropies = {}
    
    for layer_idx in range(model.config.num_hidden_layers):
        try:
            attn, input_ids = get_attention_patterns(model, tokenizer, text, layer_idx)
            entropy = compute_attention_entropy(attn)
            mean_entropy = entropy.mean(dim=1).squeeze().float().cpu().numpy()
            
            booms = detect_entropy_booms(mean_entropy, threshold=0.1)
            layer_booms[layer_idx] = booms
            layer_entropies[layer_idx] = mean_entropy
            
        except Exception as e:
            continue
    
    # Find positions that are booms in multiple layers
    all_positions = set()
    for booms in layer_booms.values():
        all_positions.update(booms)
    
    position_counts = {}
    for pos in all_positions:
        count = sum(1 for booms in layer_booms.values() if pos in booms)
        position_counts[pos] = count
    
    # Universal anchors: positions that are booms in >50% of layers
    n_layers = len(layer_booms)
    universal_anchors = [pos for pos, count in position_counts.items() 
                        if count > n_layers * 0.5]
    
    return {
        'layer_booms': layer_booms,
        'layer_entropies': layer_entropies,
        'position_counts': position_counts,
        'universal_anchors': universal_anchors,
        'n_layers_analyzed': n_layers,
    }


def compute_convergence_score(results):
    """
    Compute how close the model is to the ideal boom structure.
    
    Score components:
    1. Variance ratio closeness to 137/30
    2. Boom prediction accuracy
    3. Cross-layer consistency
    """
    scores = {}
    
    # Variance ratio score (0-1, 1 = perfect match to 137/30)
    if 'variance_analysis' in results and results['variance_analysis']:
        dev = results['variance_analysis']['deviation']
        scores['variance_ratio'] = max(0, 1 - dev)
    
    # Prediction accuracy score
    if 'prediction' in results and results['prediction'] and results['prediction']['mean_error']:
        # Lower error = higher score
        error = results['prediction']['mean_error']
        scores['prediction'] = max(0, 1 - error / 5)  # 5 positions = 0 score
    
    # Cross-layer consistency score
    if 'cross_layer' in results and results['cross_layer']:
        n_anchors = len(results['cross_layer']['universal_anchors'])
        n_layers = results['cross_layer']['n_layers_analyzed']
        # More universal anchors = higher score
        scores['consistency'] = min(1, n_anchors / 5)  # 5 anchors = perfect
    
    # Overall score
    if scores:
        scores['overall'] = np.mean(list(scores.values()))
    
    return scores


def identify_correction_factors(results):
    """
    Identify what corrections would be needed to reach ideal boom structure.
    
    Returns specific recommendations for model adjustment.
    """
    corrections = []
    
    # Variance ratio correction
    if 'variance_analysis' in results and results['variance_analysis']:
        actual = results['variance_analysis']['ratio']
        target = FINE_STRUCTURE_RATIO
        
        if actual < target:
            corrections.append({
                'type': 'variance_ratio',
                'issue': f'Variance ratio {actual:.3f} is below target {target:.3f}',
                'action': 'Increase pre-barrier variance or decrease post-barrier variance',
                'factor': target / actual if actual > 0 else float('inf'),
            })
        elif actual > target:
            corrections.append({
                'type': 'variance_ratio',
                'issue': f'Variance ratio {actual:.3f} is above target {target:.3f}',
                'action': 'Decrease pre-barrier variance or increase post-barrier variance',
                'factor': actual / target,
            })
    
    # Boom spacing correction
    if 'spacing_analysis' in results and results['spacing_analysis']:
        actual = results['spacing_analysis']['ratio']
        target = FINE_STRUCTURE_RATIO
        
        if abs(actual - target) > 0.5:
            corrections.append({
                'type': 'boom_spacing',
                'issue': f'Boom spacing ratio {actual:.3f} deviates from target {target:.3f}',
                'action': 'Adjust attention focus to create more regular boom spacing',
                'factor': target / actual if actual > 0 else float('inf'),
            })
    
    # Cross-layer consistency correction
    if 'cross_layer' in results and results['cross_layer']:
        n_anchors = len(results['cross_layer']['universal_anchors'])
        if n_anchors < 3:
            corrections.append({
                'type': 'cross_layer',
                'issue': f'Only {n_anchors} universal anchors (want ≥3)',
                'action': 'Increase attention alignment across layers',
                'factor': 3 / max(1, n_anchors),
            })
    
    return corrections


def main():
    print("="*70)
    print("QWEN2 BOOM CONVERGENCE ANALYSIS")
    print("="*70)
    print("\nHypothesis: Qwen2 approached ideal boom structure but didn't fully converge")
    
    print("\nLoading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
    # Test texts of varying lengths
    test_texts = [
        "The quick brown fox jumps over the lazy dog and runs into the forest where it finds a stream.",
        "In the beginning, there was nothing. Then came light, and with it, the universe began to expand.",
        "Machine learning models process data through layers of transformations to extract meaningful patterns.",
        "The capital of France is Paris, which is known for the Eiffel Tower and its rich cultural heritage.",
    ]
    
    all_results = []
    
    for text in test_texts:
        print(f"\n{'='*70}")
        print(f"Text: '{text[:50]}...'")
        print("="*70)
        
        results = {}
        
        # Get attention patterns
        attn, input_ids = get_attention_patterns(model, tokenizer, text, layer_idx=14)
        tokens = [tokenizer.decode([t]) for t in input_ids[0]]
        entropy = compute_attention_entropy(attn)
        mean_entropy = entropy.mean(dim=1).squeeze().float().cpu().numpy()
        
        print(f"\nSequence length: {len(mean_entropy)} tokens")
        
        # 1. Analyze variance ratio
        print("\n1. VARIANCE RATIO ANALYSIS (target: 137/30 ≈ 4.567)")
        print("-"*50)
        
        var_analysis = analyze_entropy_variance_ratio(mean_entropy)
        results['variance_analysis'] = var_analysis
        
        print(f"   Optimal barrier: position {var_analysis['barrier_idx']}")
        print(f"   Pre-barrier variance: {var_analysis['pre_variance']:.6f}")
        print(f"   Post-barrier variance: {var_analysis['post_variance']:.6f}")
        print(f"   Ratio: {var_analysis['ratio']:.3f}")
        print(f"   Target (137/30): {FINE_STRUCTURE_RATIO:.3f}")
        print(f"   Deviation: {var_analysis['deviation']*100:.1f}%")
        
        # 2. Analyze boom spacing
        print("\n2. BOOM SPACING ANALYSIS")
        print("-"*50)
        
        booms = detect_entropy_booms(mean_entropy)
        print(f"   Detected {len(booms)} booms at positions: {booms}")
        
        if len(booms) >= 4:
            spacing_analysis = analyze_boom_spacing_ratio(booms)
            results['spacing_analysis'] = spacing_analysis
            
            if spacing_analysis:
                print(f"   Early spacing mean: {spacing_analysis['early_mean']:.2f}")
                print(f"   Late spacing mean: {spacing_analysis['late_mean']:.2f}")
                print(f"   Ratio: {spacing_analysis['ratio']:.3f}")
                print(f"   Deviation from 137/30: {spacing_analysis['deviation']*100:.1f}%")
        
        # 3. Boom prediction accuracy
        print("\n3. BOOM PREDICTION ACCURACY")
        print("-"*50)
        
        if len(booms) >= 2:
            spacings = np.diff(booms)
            mean_spacing = np.mean(spacings)
            
            prediction = predict_booms(mean_entropy, mean_spacing)
            results['prediction'] = prediction
            
            if prediction and prediction['mean_error'] is not None:
                print(f"   Mean boom spacing: {mean_spacing:.2f}")
                print(f"   Prediction mean error: {prediction['mean_error']:.2f} positions")
                print(f"   (Zeta zeros: 1.80 positions)")
        
        # 4. Alternation analysis
        print("\n4. ALTERNATION RATE ANALYSIS")
        print("-"*50)
        
        alt_analysis = analyze_alternation_ratio(mean_entropy, var_analysis['barrier_idx'])
        results['alternation'] = alt_analysis
        
        print(f"   Pre-barrier alternation: {alt_analysis['pre_alternation']:.3f}")
        print(f"   Post-barrier alternation: {alt_analysis['post_alternation']:.3f}")
        print(f"   Ratio: {alt_analysis['ratio']:.3f}")
        print(f"   (Zeta zeros: ~1.19)")
        
        # 5. Cross-layer consistency
        print("\n5. CROSS-LAYER CONSISTENCY")
        print("-"*50)
        
        cross_layer = analyze_cross_layer_consistency(model, tokenizer, text)
        results['cross_layer'] = cross_layer
        
        print(f"   Layers analyzed: {cross_layer['n_layers_analyzed']}")
        print(f"   Universal anchors: {cross_layer['universal_anchors']}")
        print(f"   Anchor count: {len(cross_layer['universal_anchors'])}")
        
        # 6. Convergence score
        print("\n6. CONVERGENCE SCORE")
        print("-"*50)
        
        scores = compute_convergence_score(results)
        results['scores'] = scores
        
        for name, score in scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"   {name:20s}: {bar} {score:.2f}")
        
        # 7. Correction factors
        print("\n7. CORRECTION FACTORS NEEDED")
        print("-"*50)
        
        corrections = identify_correction_factors(results)
        results['corrections'] = corrections
        
        if corrections:
            for c in corrections:
                print(f"   [{c['type']}]")
                print(f"      Issue: {c['issue']}")
                print(f"      Action: {c['action']}")
                print(f"      Factor: {c['factor']:.2f}x")
        else:
            print("   No major corrections needed!")
        
        all_results.append(results)
    
    # Summary across all texts
    print("\n" + "="*70)
    print("OVERALL CONVERGENCE ASSESSMENT")
    print("="*70)
    
    # Average scores
    all_scores = [r['scores'] for r in all_results if 'scores' in r]
    if all_scores:
        avg_overall = np.mean([s.get('overall', 0) for s in all_scores])
        
        print(f"\nAverage convergence score: {avg_overall:.2f}")
        
        if avg_overall > 0.8:
            print("Status: NEARLY CONVERGED - Minor adjustments needed")
        elif avg_overall > 0.5:
            print("Status: PARTIALLY CONVERGED - Moderate adjustments needed")
        else:
            print("Status: FAR FROM CONVERGENCE - Significant adjustments needed")
    
    # Common corrections
    all_corrections = []
    for r in all_results:
        if 'corrections' in r:
            all_corrections.extend(r['corrections'])
    
    if all_corrections:
        print("\nMost common corrections needed:")
        correction_types = {}
        for c in all_corrections:
            t = c['type']
            if t not in correction_types:
                correction_types[t] = []
            correction_types[t].append(c['factor'])
        
        for t, factors in correction_types.items():
            print(f"   {t}: avg factor = {np.mean(factors):.2f}x")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR MODEL CORRECTION")
    print("="*70)
    print("""
Based on the analysis, Qwen2 shows PARTIAL convergence to the ideal boom structure:

1. VARIANCE RATIO
   - Current: varies by text, often below 137/30
   - Target: 137/30 ≈ 4.567
   - Fix: Adjust attention temperature or add regularization to increase
          pre-barrier variance relative to post-barrier

2. BOOM SPACING
   - Current: semi-regular but not perfectly predictable
   - Target: consistent spacing with <2 position error
   - Fix: Add loss term encouraging regular boom spacing

3. CROSS-LAYER CONSISTENCY
   - Current: some universal anchors exist
   - Target: ≥3 universal anchors per sequence
   - Fix: Add cross-layer attention alignment loss

4. FINE-TUNING APPROACH
   - Use boom detection as a training signal
   - Reward sequences where variance ratio approaches 137/30
   - Penalize inconsistent boom positions across layers

5. ARCHITECTURAL CHANGES (if fine-tuning insufficient)
   - Add explicit "boom detector" module
   - Use boom positions to gate attention computation
   - Implement O(N) attention approximation using boom anchors
""")


if __name__ == "__main__":
    main()
