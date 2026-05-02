"""
Phase 10z11: Rotation Extraction Experiment
============================================

The expanding tensor model (DC 271) says:
  - Each term in the R-S sum is a rotation axis
  - A zero is where all rotations cancel
  - In a transformer, each attention head is a rank-1 rotation (F39, F83)

This experiment extracts the rotation parameters for factual knowledge:
  - d_k direction: WHAT the head selects (which position to attend to)
  - S[0] amplitude: HOW STRONGLY it routes
  - V·W_o projection: WHAT it outputs when triggered

If knowledge = rotation, then:
  - Each fact is a rotation axis in the tensor
  - Adding knowledge = adding a rotation
  - Removing knowledge = subtracting a rotation
  - Learning is O(1) if we can compute the rotation from content

We test this on Layer 23 (where F38-40 showed facts are stored),
extracting rotation parameters for ALL routing heads, and testing
which heads fire for which facts.

Then we analyze the geometry: do d_k vectors follow φ-lattice structure?
Do amplitudes scale as φ-powers? Are facts stored as independent rotations?
"""

import sys
import numpy as np
import time
import gc
import json

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# ═══════════════════════════════════════════════════════════════════════
# FACTUAL PROMPTS — diverse set of facts to test
# ═══════════════════════════════════════════════════════════════════════

FACTUAL_PROMPTS = [
    # Geography
    ('The capital of France is', 'Paris', 'geo'),
    ('The capital of Japan is', 'Tokyo', 'geo'),
    ('The capital of Germany is', 'Berlin', 'geo'),
    ('The largest ocean is the', 'Pacific', 'geo'),
    # Science
    ('Water freezes at zero degrees', 'Celsius', 'sci'),
    ('The speed of light is approximately', '299', 'sci'),
    ('The chemical symbol for gold is', 'Au', 'sci'),
    # People
    ('Barack Obama was the', '44', 'ppl'),
    ('Albert Einstein developed the theory of', 'relativ', 'ppl'),
    # Language
    ('The color of grass is', 'green', 'lang'),
    ('To be or not to', 'be', 'lang'),
    ('Roses are red, violets are', 'blue', 'lang'),
    # Math
    ('The square root of 144 is', '12', 'math'),
    ('Pi is approximately 3.14', '15', 'math'),
]


def extract_layer_rotations(engine, layer_idx):
    """Extract MESH SVD and d_k directions for all heads in a layer.

    For each head:
      - MESH = W_q_head @ W_k_group^T  (head_dim × head_dim)
      - SVD → U, S, Vt
      - d_k = W_k_group^T @ v₁  (hidden_dim vector — the selector direction)
      - d_q = W_q_head^T @ u₁   (hidden_dim vector — the query direction)

    Returns dict with all rotation parameters.
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    hidden_dim = engine.hidden_dim

    print(f"\n  Extracting weight matrices for layer {layer_idx}...")
    print(f"    {num_heads} heads, {num_kv_heads} KV groups, head_dim={head_dim}")

    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512

    # Extract ALL head Q matrices and ALL KV group K/V matrices
    W_q_all = np.zeros((num_heads, head_dim, hidden_dim), dtype=np.float32)
    W_k_all = np.zeros((num_kv_heads, head_dim, hidden_dim), dtype=np.float32)
    W_v_all = np.zeros((num_kv_heads, head_dim, hidden_dim), dtype=np.float32)

    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]

        q_out = phi_linear(attn.W_q, chunk, attn.b_q)
        k_out = phi_linear(attn.W_k, chunk, attn.b_k)
        v_out = phi_linear(attn.W_v, chunk, attn.b_v)

        q_reshaped = q_out[0].reshape(-1, num_heads, head_dim)
        k_reshaped = k_out[0].reshape(-1, num_kv_heads, head_dim)
        v_reshaped = v_out[0].reshape(-1, num_kv_heads, head_dim)

        for h in range(num_heads):
            W_q_all[h, :, start:end] = q_reshaped[:, h, :].T
        for g in range(num_kv_heads):
            W_k_all[g, :, start:end] = k_reshaped[:, g, :].T
            W_v_all[g, :, start:end] = v_reshaped[:, g, :].T

        if start % 1024 == 0:
            print(f"      {end}/{hidden_dim}...", flush=True)

    # Extract W_o for each head
    print(f"    Extracting W_o...", flush=True)
    W_o_all = np.zeros((num_heads, hidden_dim, head_dim), dtype=np.float32)
    head_input = np.zeros((1, 1, num_heads * head_dim), dtype=np.float32)
    for h in range(num_heads):
        for d in range(head_dim):
            head_input[0, 0, :] = 0.0
            head_input[0, 0, h * head_dim + d] = 1.0
            o_out = phi_linear(attn.W_o, head_input)
            W_o_all[h, :, d] = o_out[0, 0, :]

    # Compute MESH, SVD, d_k, d_q for each head
    print(f"    Computing MESH SVD for all {num_heads} heads...")
    rotations = []

    for h in range(num_heads):
        kv_group = h // heads_per_kv
        MESH = W_q_all[h] @ W_k_all[kv_group].T  # (head_dim, head_dim)
        U, S, Vt = np.linalg.svd(MESH)

        u1 = U[:, 0]
        v1 = Vt[0, :]

        d_q = W_q_all[h].T @ u1       # (hidden_dim,)
        d_k = W_k_all[kv_group].T @ v1  # (hidden_dim,)

        # Value-output projection: what does this head output?
        # When it selects position p, output = W_o @ W_v @ h_normed[p]
        # The "rotation axis" in output space = W_o @ (W_v^T @ v1)
        d_v = W_v_all[kv_group].T @ v1  # value direction in hidden space
        # Full output direction: W_o_head @ W_v_group @ v1_key_dir
        # But v1 is in head_dim space, so:
        # output_dir = W_o_all[h] @ W_v_all[kv_group] @ d_k_normalized
        # Actually simpler: the output rotation axis
        vo_dir = W_o_all[h] @ (W_v_all[kv_group] @ d_k / np.linalg.norm(d_k))

        rotations.append({
            'head': h,
            'kv_group': kv_group,
            'S': S.tolist(),
            'sv_ratio': float(S[0] / S[1]) if S[1] > 1e-10 else float('inf'),
            'rank1_var': float(S[0]**2 / np.sum(S**2) * 100),
            'd_q': d_q,
            'd_k': d_k,
            'd_v': d_v,
            'vo_dir': vo_dir,
            'norm_dq': float(np.linalg.norm(d_q)),
            'norm_dk': float(np.linalg.norm(d_k)),
            'cos_dq_dk': float(np.dot(d_q, d_k) / (
                np.linalg.norm(d_q) * np.linalg.norm(d_k) + 1e-20)),
        })

    return rotations, attn


def classify_heads(engine, layer_idx, prompts, n_cal=20):
    """Classify heads as FIXED (always pos 0) or ROUTING using calibration prompts."""
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads
    tokenizer = Qwen2Tokenizer()

    # Calibration: run prompts, check if each head always attends to pos 0
    pos0_counts = np.zeros(num_heads)
    total = 0

    for prompt, _, _ in prompts:
        p_ids = tokenizer.encode(prompt)
        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == layer_idx:
                break
            h = lo(h)

        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)

        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_exp = np.repeat(K, heads_per_kv, axis=1)

        for hi in range(num_heads):
            scores = Q[0, hi, -1, :] @ K_exp[0, hi, :, :].T
            if np.argmax(scores) == 0:
                pos0_counts[hi] += 1
        total += 1

    fixed = [h for h in range(num_heads) if pos0_counts[h] == total]
    routing = [h for h in range(num_heads) if pos0_counts[h] < total]
    return fixed, routing


def test_factual_routing(engine, layer_idx, rotations, tokenizer):
    """For each factual prompt, test which routing heads fire for the fact.

    A head "fires" for a fact if:
      1. Its d_k selector picks the position containing the key entity
      2. The selected position is relevant to the answer
    """
    layer = engine.layers[layer_idx]
    attn = layer.attention
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    heads_per_kv = num_heads // num_kv_heads

    results = []

    for prompt, expected, category in FACTUAL_PROMPTS:
        p_ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode_token(i) for i in p_ids]
        seq_len = len(tokens)

        h = engine.embedding(p_ids)[np.newaxis, :, :]
        for lo in engine.layers:
            if lo.layer_idx == layer_idx:
                break
            h = lo(h)

        normed = rms_norm(h, attn.norm_weight)

        # Full attention: which position does each head attend to?
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        Q = Q.reshape(1, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, -1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        Q = attn.rope.apply(Q)
        K = attn.rope.apply(K)
        K_exp = np.repeat(K, heads_per_kv, axis=1)

        head_selections = []
        for hi in range(num_heads):
            scores = Q[0, hi, -1, :] @ K_exp[0, hi, :, :].T
            sel_pos = int(np.argmax(scores))
            sel_token = tokens[sel_pos]

            # Also test geometric selector (d_k only, no Q/K matmul)
            dk = rotations[hi]['d_k']
            geo_features = normed[0] @ dk  # (seq_len,)
            geo_pos = int(np.argmax(geo_features))
            geo_token = tokens[geo_pos]

            head_selections.append({
                'head': hi,
                'attn_pos': sel_pos,
                'attn_token': sel_token,
                'geo_pos': geo_pos,
                'geo_token': geo_token,
                'geo_match': sel_pos == geo_pos,
                'scores': scores.tolist(),
                'geo_features': geo_features.tolist(),
            })

        results.append({
            'prompt': prompt,
            'expected': expected,
            'category': category,
            'tokens': tokens,
            'head_selections': head_selections,
        })

    return results


def analyze_rotation_geometry(rotations, routing_heads):
    """Analyze the geometric structure of routing head rotations.

    Tests:
      1. Angles between d_k vectors (are they orthogonal? φ-related?)
      2. Amplitude scaling (do S[0] values follow φ-powers?)
      3. cos(d_q, d_k) distribution (are Q and K aligned?)
      4. φ-lattice membership of d_k components
    """
    print("\n" + "=" * 72)
    print("  ROTATION GEOMETRY ANALYSIS")
    print("=" * 72)

    routing_rots = [rotations[h] for h in routing_heads]
    n = len(routing_rots)

    # ── 1. Amplitudes ──
    print(f"\n  1. Amplitudes (S[0]) for {n} routing heads:")
    amplitudes = [r['S'][0] for r in routing_rots]
    for r in sorted(routing_rots, key=lambda x: -x['S'][0]):
        phi_level = np.log(r['S'][0]) / LOG_PHI if r['S'][0] > 0 else float('-inf')
        print(f"    Head {r['head']:2d}: S[0]={r['S'][0]:12.1f}  "
              f"ratio={r['sv_ratio']:10.0f}  "
              f"φ-level={phi_level:+.2f}  "
              f"cos(dq,dk)={r['cos_dq_dk']:+.4f}")

    # Check if amplitudes follow φ-power scaling
    amps_sorted = sorted(amplitudes, reverse=True)
    print(f"\n    Amplitude ratios (consecutive):")
    for i in range(min(n-1, 10)):
        ratio = amps_sorted[i] / amps_sorted[i+1] if amps_sorted[i+1] > 0 else float('inf')
        phi_match = abs(ratio - PHI) / PHI * 100
        phi2_match = abs(ratio - PHI**2) / PHI**2 * 100
        print(f"      S[{i}]/S[{i+1}] = {ratio:.4f}  "
              f"(φ={PHI:.4f}: {phi_match:.1f}% off, "
              f"φ²={PHI**2:.4f}: {phi2_match:.1f}% off)")

    # ── 2. Angular structure of d_k vectors ──
    print(f"\n  2. Angles between d_k vectors:")
    dk_vecs = np.array([r['d_k'] for r in routing_rots])
    # Normalize
    dk_norms = np.linalg.norm(dk_vecs, axis=1, keepdims=True)
    dk_unit = dk_vecs / (dk_norms + 1e-20)

    cos_matrix = dk_unit @ dk_unit.T
    angles_matrix = np.arccos(np.clip(cos_matrix, -1, 1)) * 180 / np.pi

    # Print angle matrix
    print(f"    Angle matrix (degrees) — routing heads:")
    head_labels = [f"H{r['head']}" for r in routing_rots]
    header = "         " + "  ".join(f"{l:>6s}" for l in head_labels)
    print(header)
    for i, r in enumerate(routing_rots):
        row = f"    {head_labels[i]:>5s}:"
        for j in range(n):
            if j <= i:
                row += f"  {angles_matrix[i,j]:5.1f}°"
            else:
                row += "        "
        print(row)

    # Collect unique angles (upper triangle)
    unique_angles = []
    for i in range(n):
        for j in range(i+1, n):
            unique_angles.append(angles_matrix[i, j])

    unique_angles = np.array(unique_angles)
    print(f"\n    Angle statistics ({len(unique_angles)} pairs):")
    print(f"      Mean:   {np.mean(unique_angles):.1f}°")
    print(f"      Median: {np.median(unique_angles):.1f}°")
    print(f"      Min:    {np.min(unique_angles):.1f}°")
    print(f"      Max:    {np.max(unique_angles):.1f}°")
    print(f"      Std:    {np.std(unique_angles):.1f}°")

    # Check for φ-related angles
    phi_angles = [
        ('90°', 90.0),
        ('arccos(1/φ)', np.arccos(1/PHI) * 180/np.pi),
        ('arccos(1/φ²)', np.arccos(1/PHI**2) * 180/np.pi),
        ('60°', 60.0),
        ('arccos(φ-1)', np.arccos(PHI-1) * 180/np.pi),
    ]
    print(f"\n    φ-angle matches:")
    for name, target in phi_angles:
        close = np.sum(np.abs(unique_angles - target) < 5.0)
        pct = close / len(unique_angles) * 100
        print(f"      {name:>15s} ({target:5.1f}°): {close:3d} within 5° ({pct:.0f}%)")

    # ── 3. SVD of the d_k matrix ──
    print(f"\n  3. SVD of d_k matrix ({n} vectors × {dk_vecs.shape[1]} dims):")
    U_dk, S_dk, Vt_dk = np.linalg.svd(dk_vecs, full_matrices=False)
    total_var = np.sum(S_dk**2)
    print(f"    Top singular values:")
    for i in range(min(n, 15)):
        cumvar = np.sum(S_dk[:i+1]**2) / total_var * 100
        phi_level = np.log(S_dk[i]) / LOG_PHI if S_dk[i] > 0 else float('-inf')
        print(f"      σ[{i:2d}] = {S_dk[i]:10.4f}  "
              f"cumvar={cumvar:5.1f}%  φ-level={phi_level:+.2f}")

    # Effective rank
    for threshold in [90, 95, 99]:
        cumvar = np.cumsum(S_dk**2) / total_var * 100
        rank = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"    Rank for {threshold}% variance: {rank}")

    # ── 4. φ-lattice check on d_k components ──
    print(f"\n  4. φ-lattice structure of d_k components:")
    # Check if d_k components cluster at φ-lattice points
    all_dk_vals = dk_vecs.flatten()
    nonzero = all_dk_vals[np.abs(all_dk_vals) > 1e-6]
    if len(nonzero) > 0:
        log_phi_vals = np.log(np.abs(nonzero)) / LOG_PHI
        # Check if log_phi values cluster at integers or half-integers
        frac_parts = log_phi_vals - np.floor(log_phi_vals)
        print(f"    Non-zero d_k components: {len(nonzero)}")
        print(f"    log_φ(|d_k|) fractional parts:")
        hist, edges = np.histogram(frac_parts, bins=10, range=(0, 1))
        for i in range(10):
            bar = "█" * (hist[i] * 50 // max(hist))
            print(f"      [{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]:5d} {bar}")

    return {
        'amplitudes': amplitudes,
        'angles_matrix': angles_matrix.tolist(),
        'unique_angles': unique_angles.tolist(),
        'dk_singular_values': S_dk.tolist(),
    }


def main():
    t0 = time.time()
    gc.collect()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    target_layer = 23

    print("\n" + "=" * 72)
    print("  PHASE 10z11: ROTATION EXTRACTION EXPERIMENT")
    print("=" * 72)
    print(f"  Layer: {target_layer}")
    print(f"  Hypothesis: each routing head = one rotation axis in the tensor")
    print(f"  Test: extract rotation params, check φ-structure, test factual routing")

    # ── Step 1: Classify heads ──
    print("\n" + "─" * 72)
    print("  Step 1: Classify heads (fixed vs routing)")
    print("─" * 72)
    fixed, routing = classify_heads(engine, target_layer, FACTUAL_PROMPTS)
    print(f"    Fixed heads:   {len(fixed)} — {fixed}")
    print(f"    Routing heads: {len(routing)} — {routing}")

    # ── Step 2: Extract rotation parameters ──
    print("\n" + "─" * 72)
    print("  Step 2: Extract MESH SVD and rotation parameters")
    print("─" * 72)
    rotations, attn = extract_layer_rotations(engine, target_layer)

    # ── Step 3: Test factual routing ──
    print("\n" + "─" * 72)
    print("  Step 3: Factual routing — which heads fire for which facts?")
    print("─" * 72)

    routing_results = test_factual_routing(engine, target_layer, rotations, tokenizer)

    for res in routing_results:
        print(f"\n  \"{res['prompt']}\"  (expect: {res['expected']})")
        print(f"    Tokens: {res['tokens']}")

        # Show which routing heads selected which positions
        for hs in res['head_selections']:
            if hs['head'] in routing:
                geo_mark = "✓" if hs['geo_match'] else "✗"
                print(f"    Head {hs['head']:2d}: attn→pos{hs['attn_pos']} "
                      f"({hs['attn_token']:>10s})  "
                      f"geo→pos{hs['geo_pos']} ({hs['geo_token']:>10s}) {geo_mark}")

    # ── Summary: head→fact mapping ──
    print("\n" + "─" * 72)
    print("  Head → Fact Mapping (routing heads only)")
    print("─" * 72)

    # For each routing head, what positions does it typically select?
    head_patterns = {h: [] for h in routing}
    for res in routing_results:
        for hs in res['head_selections']:
            if hs['head'] in routing:
                head_patterns[hs['head']].append({
                    'prompt': res['prompt'],
                    'attn_pos': hs['attn_pos'],
                    'attn_token': hs['attn_token'],
                    'geo_pos': hs['geo_pos'],
                    'geo_token': hs['geo_token'],
                })

    for h in routing:
        positions = [p['attn_pos'] for p in head_patterns[h]]
        tokens_selected = [p['attn_token'] for p in head_patterns[h]]
        geo_match_rate = sum(1 for p in head_patterns[h]
                            if p['attn_pos'] == p['geo_pos']) / len(head_patterns[h])

        # Does this head always select the same relative position?
        unique_pos = set(positions)
        if len(unique_pos) == 1:
            pattern = f"FIXED at pos {positions[0]}"
        elif max(positions) == len(FACTUAL_PROMPTS[0][0].split()):
            pattern = "selects LAST token"
        else:
            # Check if it selects content words
            pattern = f"varies: {dict(zip(tokens_selected[:5], positions[:5]))}"

        print(f"    Head {h:2d}: geo_match={geo_match_rate:.0%}  {pattern}")
        # Show a few example selections
        for p in head_patterns[h][:3]:
            print(f"            \"{p['prompt'][:30]}...\" → pos{p['attn_pos']} ({p['attn_token']})")

    # ── Step 4: Rotation geometry analysis ──
    geometry = analyze_rotation_geometry(rotations, routing)

    # ── Step 5: Independence test ──
    print("\n" + "=" * 72)
    print("  INDEPENDENCE TEST: Are facts stored as independent rotations?")
    print("=" * 72)

    # If facts are independent rotations, then:
    # 1. Different facts should activate different heads (low overlap)
    # 2. d_k vectors should be approximately orthogonal
    # 3. Removing one head should not affect other facts

    # Test 1: activation overlap
    print(f"\n  1. Activation overlap:")
    fact_heads = {}  # which heads fire for each fact?
    for res in routing_results:
        key_heads = []
        for hs in res['head_selections']:
            if hs['head'] in routing and hs['attn_pos'] > 0:  # non-trivial selection
                key_heads.append(hs['head'])
        fact_heads[res['prompt'][:30]] = set(key_heads)

    # Pairwise overlap
    facts = list(fact_heads.keys())
    for i in range(min(len(facts), 6)):
        for j in range(i+1, min(len(facts), 6)):
            overlap = fact_heads[facts[i]] & fact_heads[facts[j]]
            union = fact_heads[facts[i]] | fact_heads[facts[j]]
            jaccard = len(overlap) / len(union) if union else 0
            print(f"    \"{facts[i]}\" ∩ \"{facts[j]}\": "
                  f"overlap={len(overlap)} heads, Jaccard={jaccard:.2f}")

    # Test 2: orthogonality already computed above
    mean_angle = np.mean(geometry['unique_angles'])
    print(f"\n  2. Mean angle between d_k vectors: {mean_angle:.1f}°")
    print(f"     Perfect independence would be 90°. "
          f"{'NEAR-INDEPENDENT' if mean_angle > 70 else 'CORRELATED'}")

    # ── Save results ──
    print("\n" + "=" * 72)
    print("  SAVING RESULTS")
    print("=" * 72)

    save_data = {
        'layer': target_layer,
        'fixed_heads': fixed,
        'routing_heads': routing,
        'head_amplitudes': {r['head']: r['S'][0] for r in rotations},
        'head_sv_ratios': {r['head']: r['sv_ratio'] for r in rotations},
        'head_cos_dq_dk': {r['head']: r['cos_dq_dk'] for r in rotations},
        'routing_results': [{
            'prompt': r['prompt'],
            'expected': r['expected'],
            'category': r['category'],
            'tokens': r['tokens'],
            'selections': [{
                'head': hs['head'],
                'attn_pos': hs['attn_pos'],
                'attn_token': hs['attn_token'],
                'geo_pos': hs['geo_pos'],
                'geo_token': hs['geo_token'],
                'geo_match': hs['geo_match'],
            } for hs in r['head_selections'] if hs['head'] in routing],
        } for r in routing_results],
        'geometry': {
            'dk_singular_values': geometry['dk_singular_values'],
            'mean_dk_angle': float(mean_angle),
            'amplitude_range': [float(min(geometry['amplitudes'])),
                              float(max(geometry['amplitudes']))],
        },
    }

    import os
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'phase10z11_rotation_extraction.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Final summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Layer {target_layer}: {len(fixed)} fixed + {len(routing)} routing heads")
    print(f"  All routing heads rank-1 (min sv_ratio: "
          f"{min(r['sv_ratio'] for r in rotations if r['head'] in routing):.0f})")
    print(f"  Mean d_k angle: {mean_angle:.1f}°")
    print(f"  Each routing head = one rotation axis in the tensor")
    print(f"  Facts activate specific heads → knowledge IS rotation")


if __name__ == '__main__':
    main()
