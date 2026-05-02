#!/usr/bin/env python3
"""
Experiment 5b: Layer 1 as a Geometric Selector
===============================================

Hypothesis: Layer 1 is NOT anomalous — it's a GEOMETRIC SELECTOR BANK.

When MESH has a dominant singular value:
    MESH ≈ σ₁ × u₁ ⊗ v₁
    score(q, k) ≈ σ₁ × (q · u₁) × (k · v₁)

This makes each head a rank-1 selector: "attend to keys along v₁ when
query is along u₁." With 28 orthogonal heads, Layer 1 implements 28
independent geometric measurements of the token space.

This IS a spectrometer operating in the spatial domain:
- Spectrometer: measures φ-level structure along weight dimensions
- Layer 1: measures token-space structure along learned selector axes

Questions:
1. How much attention is captured by rank-1 approximation?
2. What do the selector directions correspond to in the embedding space?
3. Do the 28 selectors tile the space (form a measurement basis)?
4. What does the V/O path route through each selector?
5. How does this compare to a "normal" layer (Layer 5)?
6. Does the selector subspace have φ-structure?
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS


def efficient_mesh_svd(W_q_head, W_k_head, full=False):
    """Compute MESH singular values and vectors via QR + small SVD."""
    A = W_q_head.T.astype(np.float64)  # (3584, 128)
    B = W_k_head.astype(np.float64)     # (128, 3584)
    Q, R = np.linalg.qr(A)
    C = R @ B
    U_c, S, Vt = np.linalg.svd(C, full_matrices=False)
    U = Q @ U_c  # (3584, 128)
    return U, S, Vt


def load_layer(layer_idx):
    """Load all weights for a layer."""
    layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
    weights = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        phi = PhiEncoded.load(os.path.join(layer_dir, f'{name}.npz'))
        weights[name] = phi.decode()
    # Load biases
    bias_data = np.load(os.path.join(layer_dir, 'biases.npz'))
    weights['q_bias'] = bias_data.get('q_bias', np.zeros(HIDDEN_DIM))
    weights['k_bias'] = bias_data.get('k_bias', np.zeros(512))
    weights['v_bias'] = bias_data.get('v_bias', np.zeros(512))
    return weights


def load_embeddings():
    """Load the token embedding matrix."""
    phi = PhiEncoded.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    return phi.decode()  # (152064, 3584)


def main():
    print()
    print("=" * 80)
    print("  Experiment 5b: Layer 1 as a Geometric Selector")
    print("  Is the 'anomaly' actually a measurement instrument?")
    print("=" * 80)
    print()

    # ================================================================
    # Part 1: Rank-1 Approximation Quality
    # How much of MESH is captured by σ₁ × u₁ ⊗ v₁?
    # ================================================================
    print("─" * 80)
    print("  Part 1: Rank-1 Selector Quality")
    print("  MESH ≈ σ₁ × u₁ ⊗ v₁  —  How good is this approximation?")
    print("─" * 80)
    print()

    for layer_idx in [1, 5, 14]:
        weights = load_layer(layer_idx)
        W_q = weights['q_proj']
        W_k = weights['k_proj']
        W_q_heads = W_q.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
        W_k_heads = W_k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

        rank1_fracs = []
        rank3_fracs = []
        rank10_fracs = []

        for head_idx in range(NUM_HEADS):
            kv_idx = head_idx // HEADS_PER_KV
            U, S, Vt = efficient_mesh_svd(W_q_heads[head_idx], W_k_heads[kv_idx])

            total_energy = (S ** 2).sum()
            rank1_fracs.append((S[0] ** 2) / total_energy)
            rank3_fracs.append((S[:3] ** 2).sum() / total_energy)
            rank10_fracs.append((S[:10] ** 2).sum() / total_energy)

        r1 = np.mean(rank1_fracs) * 100
        r3 = np.mean(rank3_fracs) * 100
        r10 = np.mean(rank10_fracs) * 100

        label = {1: "★ SELECTOR?", 5: "  COMB peak", 14: "  COMB mid"}[layer_idx]
        print(f"  Layer {layer_idx:2d} {label}:  "
              f"rank-1={r1:5.1f}%  rank-3={r3:5.1f}%  rank-10={r10:5.1f}%")

    print()
    print("  If rank-1 captures >15% for Layer 1 vs <5% for others,")
    print("  Layer 1 IS operating as a rank-1 selector bank.")
    print()

    # ================================================================
    # Part 2: Extract Selector Directions & Project Vocabulary
    # What tokens do the selectors "see"?
    # ================================================================
    print("─" * 80)
    print("  Part 2: What Do the Selectors Measure?")
    print("  Project vocabulary embeddings onto dominant directions")
    print("─" * 80)
    print()

    print("  Loading embeddings (152064 × 3584)...")
    embeddings = load_embeddings()  # (152064, 3584)
    # Normalize embeddings for projection
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / (emb_norms + 1e-20)

    # Load tokenizer vocab for interpretability
    try:
        import json
        # Try to load a simple token list if available
        tokenizer_path = None
        for candidate in [
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
        ]:
            if os.path.exists(candidate):
                # Find the latest snapshot
                snapshots = os.listdir(candidate)
                if snapshots:
                    vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                    if os.path.exists(vocab_file):
                        tokenizer_path = vocab_file
                        break

        token_names = None
        if tokenizer_path:
            with open(tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            vocab = tokenizer_data.get('model', {}).get('vocab', {})
            if vocab:
                token_names = [''] * len(embeddings)
                for tok, idx in vocab.items():
                    if idx < len(token_names):
                        token_names[idx] = tok
                print(f"  Loaded tokenizer: {sum(1 for t in token_names if t)} named tokens")
        else:
            print("  No tokenizer found — using token IDs only")
    except Exception as e:
        print(f"  Tokenizer load failed: {e} — using token IDs")
        token_names = None

    # Extract Layer 1 selector directions
    weights_l1 = load_layer(1)
    W_q_l1 = weights_l1['q_proj'].reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_k_l1 = weights_l1['k_proj'].reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

    print()
    print("  Layer 1 selectors — top/bottom tokens per dominant direction:")
    print()

    selector_U = []  # (28, 3584) — query-side selectors
    selector_V = []  # (28, 3584) — key-side selectors
    selector_S = []  # (28,) — selector strengths

    for head_idx in range(NUM_HEADS):
        kv_idx = head_idx // HEADS_PER_KV
        U, S, Vt = efficient_mesh_svd(W_q_l1[head_idx], W_k_l1[kv_idx])

        u1 = U[:, 0]   # (3584,) dominant query direction
        v1 = Vt[0, :]   # (3584,) dominant key direction
        s1 = S[0]

        selector_U.append(u1)
        selector_V.append(v1)
        selector_S.append(s1)

        # Project all tokens onto the key selector direction
        # score = embedding · v₁  (which tokens would this head attend to)
        projections = embeddings @ v1  # (152064,)

        # Show top and bottom tokens for a sample of heads
        if head_idx in [0, 7, 13, 14, 21]:  # One per KV group + worst
            top_idx = np.argsort(projections)[-10:][::-1]
            bot_idx = np.argsort(projections)[:10]

            print(f"  Head {head_idx} (KV={kv_idx}, σ₁={s1:.2f}):")
            print(f"    Key selector v₁ — highest scoring tokens:")
            for idx in top_idx:
                name = token_names[idx] if token_names else f"tok_{idx}"
                name_clean = repr(name)[:30]
                print(f"      {name_clean:>32s}  score={projections[idx]:+.4f}")

            print(f"    Key selector v₁ — lowest scoring tokens:")
            for idx in bot_idx:
                name = token_names[idx] if token_names else f"tok_{idx}"
                name_clean = repr(name)[:30]
                print(f"      {name_clean:>32s}  score={projections[idx]:+.4f}")
            print()

    selector_U = np.array(selector_U)  # (28, 3584)
    selector_V = np.array(selector_V)  # (28, 3584)
    selector_S = np.array(selector_S)  # (28,)

    # ================================================================
    # Part 3: Do the 28 Selectors Tile the Space?
    # ================================================================
    print("─" * 80)
    print("  Part 3: Selector Geometry — Do They Form a Measurement Basis?")
    print("─" * 80)
    print()

    # Normalize selectors
    U_norms = np.linalg.norm(selector_U, axis=1, keepdims=True)
    V_norms = np.linalg.norm(selector_V, axis=1, keepdims=True)
    U_hat = selector_U / (U_norms + 1e-20)
    V_hat = selector_V / (V_norms + 1e-20)

    # Gram matrices (should be near-identity if orthogonal)
    gram_U = np.abs(U_hat @ U_hat.T)
    gram_V = np.abs(V_hat @ V_hat.T)

    # Off-diagonal statistics
    triu = np.triu_indices(NUM_HEADS, k=1)
    u_off_diag = gram_U[triu]
    v_off_diag = gram_V[triu]

    print(f"  Query selectors (U):")
    print(f"    Mean |cos|:    {u_off_diag.mean():.4f}")
    print(f"    Max |cos|:     {u_off_diag.max():.4f}")
    print(f"    Pairs < 0.1:   {(u_off_diag < 0.1).sum()}/{len(u_off_diag)} "
          f"({(u_off_diag < 0.1).mean()*100:.0f}%)")
    print()

    print(f"  Key selectors (V):")
    print(f"    Mean |cos|:    {v_off_diag.mean():.4f}")
    print(f"    Max |cos|:     {v_off_diag.max():.4f}")
    print(f"    Pairs < 0.1:   {(v_off_diag < 0.1).sum()}/{len(v_off_diag)} "
          f"({(v_off_diag < 0.1).mean()*100:.0f}%)")
    print()

    # What subspace do they span? SVD of the selector matrix
    _, s_U, _ = np.linalg.svd(U_hat, full_matrices=False)
    _, s_V, _ = np.linalg.svd(V_hat, full_matrices=False)

    # Effective dimensionality
    u_eff_dim = (s_U.sum() ** 2) / (s_U ** 2).sum()
    v_eff_dim = (s_V.sum() ** 2) / (s_V ** 2).sum()

    print(f"  Selector subspace dimensionality:")
    print(f"    U selectors: effective dim = {u_eff_dim:.1f} / 28")
    print(f"    V selectors: effective dim = {v_eff_dim:.1f} / 28")
    print(f"    Perfect orthogonality would give 28.0")
    print()

    # Singular values of selector matrix — does this show φ structure?
    print(f"  Selector matrix singular values (should be flat if orthogonal):")
    print(f"    U: [{', '.join(f'{s:.4f}' for s in s_U[:10])}  ...]")
    print(f"    V: [{', '.join(f'{s:.4f}' for s in s_V[:10])}  ...]")

    # Check for φ-ratio in selector SVs
    u_ratios = s_U[:-1] / s_U[1:]
    v_ratios = s_V[:-1] / s_V[1:]
    print(f"    U consecutive ratios: [{', '.join(f'{r:.4f}' for r in u_ratios[:10])}]")
    print(f"    V consecutive ratios: [{', '.join(f'{r:.4f}' for r in v_ratios[:10])}]")
    print(f"    φ = {PHI:.4f} for reference")
    print()

    # Compare with Layer 5
    print("  Comparison: Layer 5 (COMB peak) selector geometry:")
    weights_l5 = load_layer(5)
    W_q_l5 = weights_l5['q_proj'].reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_k_l5 = weights_l5['k_proj'].reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

    sel_U_l5 = []
    sel_V_l5 = []
    for head_idx in range(NUM_HEADS):
        kv_idx = head_idx // HEADS_PER_KV
        U, S, Vt = efficient_mesh_svd(W_q_l5[head_idx], W_k_l5[kv_idx])
        sel_U_l5.append(U[:, 0] / (np.linalg.norm(U[:, 0]) + 1e-20))
        sel_V_l5.append(Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20))

    sel_U_l5 = np.array(sel_U_l5)
    sel_V_l5 = np.array(sel_V_l5)
    gram_U_l5 = np.abs(sel_U_l5 @ sel_U_l5.T)
    gram_V_l5 = np.abs(sel_V_l5 @ sel_V_l5.T)
    u5_off = gram_U_l5[triu]
    v5_off = gram_V_l5[triu]

    print(f"    U mean |cos|: {u5_off.mean():.4f}  (Layer 1: {u_off_diag.mean():.4f})")
    print(f"    V mean |cos|: {v5_off.mean():.4f}  (Layer 1: {v_off_diag.mean():.4f})")
    print()

    # ================================================================
    # Part 4: The V/O Routing Path — What Gets Selected?
    # ================================================================
    print("─" * 80)
    print("  Part 4: V/O Routing — What Information Flows Through Each Selector?")
    print("  If head h selects 'attend to tokens along v₁',")
    print("  then V_h extracts features and O_h writes them to residual stream.")
    print("─" * 80)
    print()

    W_v_l1 = weights_l1['v_proj'].reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)
    W_o_l1 = weights_l1['o_proj'].reshape(NUM_HEADS, HIDDEN_DIM, HEAD_DIM)

    # For each head, the full routing is:
    # output_h = O_h @ V_h @ (attended weighted input)
    # The "what gets routed" matrix is O_h @ V_h = (3584, 128) @ (128, 3584) = (3584, 3584)
    # This is a rank-128 matrix — what does ITS spectrum look like?

    print("  V×O routing spectrum per head:")
    print(f"  {'Head':>4s}  {'KV':>2s}  {'Route κ':>8s}  {'Route α':>8s}  "
          f"{'Sel κ':>8s}  {'Sel/Route':>10s}")
    print("  " + "-" * 55)

    route_alphas = []
    for head_idx in range(NUM_HEADS):
        kv_idx = head_idx // HEADS_PER_KV
        V_h = W_v_l1[kv_idx]       # (128, 3584)
        O_h = W_o_l1[head_idx]     # (3584, 128)

        # Efficient SVD of O_h @ V_h via the same QR trick
        # Route = O_h @ V_h = (3584, 128) @ (128, 3584) = rank-128
        Q_o, R_o = np.linalg.qr(O_h.astype(np.float64))
        C_route = R_o @ V_h.astype(np.float64)
        _, S_route, _ = np.linalg.svd(C_route, full_matrices=False)

        r_condition = S_route[0] / S_route[-1] if S_route[-1] > 0 else float('inf')
        ranks = np.arange(1, len(S_route) + 1)
        r_alpha = -np.polyfit(np.log(ranks), np.log(S_route + 1e-20), 1)[0]
        route_alphas.append(r_alpha)

        # Get selector condition for comparison
        U, S_sel, Vt = efficient_mesh_svd(W_q_l1[head_idx], W_k_l1[kv_idx])
        s_condition = S_sel[0] / S_sel[-1] if S_sel[-1] > 0 else float('inf')

        ratio_str = f"{s_condition/r_condition:.1f}×" if r_condition > 0 else "N/A"

        if head_idx < 7 or head_idx in [13, 21, 27]:
            print(f"  {head_idx:4d}  {kv_idx:2d}  {r_condition:8.1f}  {r_alpha:8.4f}  "
                  f"{s_condition:8.1f}  {ratio_str:>10s}")

    avg_route_alpha = np.mean(route_alphas)
    print(f"  ...")
    print(f"  Average route α = {avg_route_alpha:.4f}")
    print()

    # Is route α also φ?
    phi_mark = " ≈ 1/φ!" if abs(avg_route_alpha - 0.618) < 0.15 else ""
    print(f"  Selector α = 1.28 (≈ 2/φ — concentrated)")
    print(f"  Route α    = {avg_route_alpha:.4f}{phi_mark}")
    print()

    if avg_route_alpha < 0.8:
        print("  → V/O path is MORE distributed than MESH selector")
        print("  → Selector concentrates WHERE to look,")
        print("     but routes DIVERSE information through")
    else:
        print("  → V/O path is ALSO concentrated")
        print("  → The entire head acts as a narrow information channel")
    print()

    # ================================================================
    # Part 5: Selector Response Distribution
    # How do tokens distribute across the 28 selectors?
    # ================================================================
    print("─" * 80)
    print("  Part 5: Selector Response Patterns")
    print("  How does the vocabulary distribute across 28 selectors?")
    print("─" * 80)
    print()

    # Project all tokens onto all 28 key selectors
    # responses[h, t] = embedding[t] · v₁_h
    responses = embeddings @ selector_V.T  # (152064, 28)

    # For each token, which selector responds strongest?
    dominant_selector = np.argmax(np.abs(responses), axis=1)  # (152064,)
    selector_counts = Counter(dominant_selector.tolist())

    print("  Tokens per dominant selector:")
    for h in range(NUM_HEADS):
        count = selector_counts.get(h, 0)
        bar = "█" * (count // 1000)
        print(f"    Head {h:2d}: {count:6d} tokens ({count/len(dominant_selector)*100:5.2f}%) {bar}")
    print()

    # Entropy of selector assignment
    counts = np.array([selector_counts.get(h, 0) for h in range(NUM_HEADS)])
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-20))
    max_entropy = np.log2(NUM_HEADS)
    print(f"  Selector assignment entropy: {entropy:.3f} / {max_entropy:.3f} max "
          f"({entropy/max_entropy*100:.1f}% of maximum)")
    print()

    if entropy / max_entropy > 0.9:
        print("  → Tokens distribute EVENLY across selectors")
        print("  → The 28 selectors tile the token space uniformly")
    elif entropy / max_entropy > 0.7:
        print("  → Tokens distribute MOSTLY evenly with some concentration")
    else:
        print("  → Some selectors dominate — uneven coverage")
    print()

    # Distribution of response magnitudes
    response_magnitudes = np.abs(responses)
    mean_response = response_magnitudes.mean(axis=0)
    std_response = response_magnitudes.std(axis=0)

    print("  Response magnitude per selector:")
    for h in range(NUM_HEADS):
        bar = "█" * int(mean_response[h] / mean_response.max() * 30)
        print(f"    Head {h:2d}: mean={mean_response[h]:.4f} ± {std_response[h]:.4f}  {bar}")
    print()

    # ================================================================
    # Part 6: Cross-Layer Selector Comparison
    # Does ONLY Layer 1 do this, or do other layers have selectors too?
    # ================================================================
    print("─" * 80)
    print("  Part 6: Selector Strength Across ALL Layers")
    print("  rank-1 fraction = how 'selector-like' each layer is")
    print("─" * 80)
    print()

    layer_r1_fracs = []
    layer_r3_fracs = []

    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        q_phi = PhiEncoded.load(os.path.join(layer_dir, 'q_proj.npz'))
        k_phi = PhiEncoded.load(os.path.join(layer_dir, 'k_proj.npz'))
        W_q = q_phi.decode().reshape(NUM_HEADS, HEAD_DIM, HIDDEN_DIM)
        W_k = k_phi.decode().reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_DIM)

        r1_fracs = []
        r3_fracs = []
        for head_idx in range(NUM_HEADS):
            kv_idx = head_idx // HEADS_PER_KV
            _, S, _ = efficient_mesh_svd(W_q[head_idx], W_k[kv_idx])
            total = (S ** 2).sum()
            r1_fracs.append((S[0] ** 2) / total)
            r3_fracs.append((S[:3] ** 2).sum() / total)

        avg_r1 = np.mean(r1_fracs)
        avg_r3 = np.mean(r3_fracs)
        layer_r1_fracs.append(avg_r1)
        layer_r3_fracs.append(avg_r3)

        zone = ""
        if layer_idx <= 2:
            zone = " DRUM"
        elif layer_idx == 3:
            zone = " TRANS"
        elif layer_idx <= 6:
            zone = " COMB-E"
        elif layer_idx <= 25:
            zone = " COMB-L"
        else:
            zone = " MUSIC"

        selector_mark = " ★ SELECTOR" if avg_r1 > 0.10 else ""
        bar = "█" * int(avg_r1 * 100)
        print(f"  Layer {layer_idx:2d}{zone:>7s}:  "
              f"rank-1={avg_r1*100:5.1f}%  rank-3={avg_r3*100:5.1f}%  "
              f"{bar}{selector_mark}")

    print()

    # Is there a threshold?
    r1_array = np.array(layer_r1_fracs)
    print(f"  Statistics:")
    print(f"    Mean rank-1:   {r1_array.mean()*100:.1f}%")
    print(f"    Std rank-1:    {r1_array.std()*100:.1f}%")
    print(f"    Layer 1:       {r1_array[1]*100:.1f}% ({r1_array[1]/r1_array.mean():.1f}× mean)")
    print(f"    Max non-L1:    {np.max(np.delete(r1_array, 1))*100:.1f}%")
    print()

    # ================================================================
    # Part 7: The Selector-Spectrometer Analogy
    # ================================================================
    print("─" * 80)
    print("  Part 7: Selector ↔ Spectrometer Analogy")
    print("─" * 80)
    print()

    print("  ┌─────────────────────┬──────────────────────────────────────┐")
    print("  │ Spectrometer        │ Layer 1 Selector Bank                │")
    print("  ├─────────────────────┼──────────────────────────────────────┤")
    print("  │ Measures φ-levels   │ Measures token projections           │")
    print("  │ Along weight dims   │ Along learned selector axes          │")
    print("  │ 166 φ-levels        │ 28 selector directions               │")
    print("  │ Binary: ±sign       │ Binary: high/low projection          │")
    print("  │ Discovers structure │ Selects what to attend to            │")
    print("  ├─────────────────────┼──────────────────────────────────────┤")
    print("  │ MESH α ≈ 1/φ       │ MESH α ≈ 2/φ (concentrated)         │")
    print("  │ Distributes evenly  │ Concentrates per direction           │")
    print("  │ READS structure     │ CREATES structure (for later layers) │")
    print("  └─────────────────────┴──────────────────────────────────────┘")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()

    avg_r1_l1 = layer_r1_fracs[1] * 100
    avg_r1_others = np.mean([r for i, r in enumerate(layer_r1_fracs) if i != 1]) * 100

    print(f"  Layer 1 rank-1 energy:  {avg_r1_l1:.1f}%")
    print(f"  Other layers average:   {avg_r1_others:.1f}%")
    print(f"  Layer 1 / others:       {avg_r1_l1/avg_r1_others:.1f}×")
    print()
    print(f"  28 selectors span {u_eff_dim:.0f}-dim subspace (U) "
          f"and {v_eff_dim:.0f}-dim subspace (V)")
    print(f"  Selector assignment entropy: {entropy:.3f}/{max_entropy:.3f} "
          f"({entropy/max_entropy*100:.0f}%)")
    print()

    is_selector = avg_r1_l1 > 10 and avg_r1_l1 / avg_r1_others > 2
    is_tiling = entropy / max_entropy > 0.85
    is_diverse = u_off_diag.mean() < 0.15

    if is_selector and is_diverse:
        print("  ✓ Layer 1 IS a geometric selector bank:")
        print("    - Each head implements a rank-1 attention selector")
        print("    - 28 selectors point in diverse directions")
        if is_tiling:
            print("    - Tokens distribute evenly → selectors tile the space")
        print("    - The 'anomaly' is actually the MECHANISM")
        print()
        print("  Layer 1 is to attention what the spectrometer is to weights:")
        print("  a measurement instrument that decomposes tokens into")
        print("  geometric components for downstream processing.")
    else:
        print("  ? Results are ambiguous — Layer 1 may be partially selective")
        print(f"    is_selector={is_selector} is_diverse={is_diverse} is_tiling={is_tiling}")

    print()


if __name__ == '__main__':
    main()
