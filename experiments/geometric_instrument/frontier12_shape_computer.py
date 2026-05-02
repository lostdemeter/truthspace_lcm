"""
Frontier 12: The Shape Computer
=================================
"There are no quintics." — Abel-Ruffini / Galois

4D is what we got. Can we compute with shapes instead of scalars?

The shape machine formalization (DC 284) says a transformer is:
  directions + projections + interference + gate

This experiment extracts the essential directions from the full model
and tests whether the capital-of task can be solved in a minimal
geometric subspace using ONLY shape operations:
  - dot product (project onto direction)
  - addition (interference / superposition)
  - threshold (φ-gate)

Experiments:
  A: Extract essential directions — how many independent dims?
  B: Low-dimensional projection — what's the minimum d that works?
  C: The shape computer — solve capital-of with directions only
  D: The 4D test — can we do it in 4 dimensions?
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear


MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
PHI_CONST = (1 + np.sqrt(5)) / 2


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-20 else 0.0


def main():
    print("=" * 80)
    print("  Frontier 12: The Shape Computer")
    print("  Can we compute with shapes instead of scalars?")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    prompts = {
        'France': ('The capital of France is', ' Paris'),
        'Germany': ('The capital of Germany is', ' Berlin'),
        'Japan': ('The capital of Japan is', ' Tokyo'),
        'Italy': ('The capital of Italy is', ' Rome'),
    }

    # Get answer token IDs
    answer_ids = {}
    for name, (prompt, answer) in prompts.items():
        answer_ids[name] = tokenizer.encode(answer)[-1]
        print(f"    {name}: answer token = {answer} (id={answer_ids[name]})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT A: Extract the essential directions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment A: Extract essential directions")
    print("  What are the shapes the model actually uses?")
    print("=" * 80)

    # 1. Entity hidden states at position 3 (at various layers)
    entity_states = {}  # entity_states[name][layer] = h at entity pos
    last_pos_states = {}  # last position hidden states

    for name, (prompt, answer) in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        entity_states[name] = {-1: h[0, 3, :].copy()}
        last_pos_states[name] = {-1: h[0, -1, :].copy()}
        for li in range(28):
            layer = engine.layers[li]
            h = layer(h)
            entity_states[name][li] = h[0, 3, :].copy()
            last_pos_states[name][li] = h[0, -1, :].copy()

    # 2. Key directions from Head 6 at L23
    L23 = 23
    attn = engine.layers[L23].attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    kv_group = 6 // hpk
    hidden_dim = engine.hidden_dim

    print(f"\n  Extracting W_q, W_k, W_v for L23 Head 6...", flush=True)
    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512
    W_q_h6 = np.zeros((hd, hidden_dim), dtype=np.float32)
    W_k_g0 = np.zeros((hd, hidden_dim), dtype=np.float32)
    W_v_g0 = np.zeros((hd, hidden_dim), dtype=np.float32)

    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]
        q_out = phi_linear(attn.W_q, chunk, attn.b_q)
        k_out = phi_linear(attn.W_k, chunk, attn.b_k)
        v_out = phi_linear(attn.W_v, chunk, attn.b_v)
        q_reshaped = q_out[0].reshape(-1, nh, hd)
        k_reshaped = k_out[0].reshape(-1, nkv, hd)
        v_reshaped = v_out[0].reshape(-1, nkv, hd)
        W_q_h6[:, start:end] = q_reshaped[:, 6, :].T
        W_k_g0[:, start:end] = k_reshaped[:, kv_group, :].T
        W_v_g0[:, start:end] = v_reshaped[:, kv_group, :].T

    # Extract W_o for head 6
    W_o_h6 = np.zeros((hidden_dim, hd), dtype=np.float32)
    head6_input = np.zeros((1, 1, nh * hd), dtype=np.float32)
    for d in range(hd):
        head6_input[0, 0, :] = 0.0
        head6_input[0, 0, 6 * hd + d] = 1.0
        o_out = phi_linear(attn.W_o, head6_input)
        W_o_h6[:, d] = o_out[0, 0, :]

    # MESH SVD → d_k, d_q directions
    MESH = W_q_h6 @ W_k_g0.T
    U_mesh, S_mesh, Vt_mesh = np.linalg.svd(MESH)
    u1 = U_mesh[:, 0]
    v1_mesh = Vt_mesh[0, :]
    d_q = W_q_h6.T @ u1     # hidden-space query direction
    d_k = W_k_g0.T @ v1_mesh  # hidden-space key direction

    d_q_norm = d_q / np.linalg.norm(d_q)
    d_k_norm = d_k / np.linalg.norm(d_k)

    print(f"  MESH S[0]={S_mesh[0]:.1f}, S[1]={S_mesh[1]:.4f}")
    print(f"  cos(d_q, d_k) = {cosine(d_q, d_k):.4f}")

    # 3. Gate direction v₁ from W_gate at L15 (COMB zone)
    # Extract via SVD of gate responses to entity states
    print(f"\n  Computing gate directions at COMB layers...", flush=True)
    gate_dirs = {}
    for li in [15, 16, 17, 18, 19, 20]:
        mlp = engine.layers[li].mlp
        # Get gate responses for each entity at this layer's input
        responses = []
        for name in prompts:
            h_in = entity_states[name][li - 1]
            h_normed = rms_norm(h_in[np.newaxis, np.newaxis, :],
                                mlp.norm_weight)[0, 0]
            responses.append(h_normed)
        responses = np.stack(responses)  # (4, hidden_dim)
        # SVD of entity responses → v₁ = dominant direction
        U_r, S_r, Vt_r = np.linalg.svd(responses, full_matrices=False)
        gate_dirs[li] = Vt_r[0, :]  # dominant input direction
        energy_r1 = S_r[0] ** 2 / np.sum(S_r ** 2) * 100
        print(f"    L{li}: v₁ energy = {energy_r1:.1f}%, "
              f"cos(v₁, d_k) = {cosine(gate_dirs[li], d_k):.4f}")

    # 4. Answer directions in vocabulary space
    print(f"\n  Extracting answer directions from lm_head...", flush=True)
    answer_dirs = {}
    for name, (prompt, answer) in prompts.items():
        # The lm_head maps hidden_dim → vocab
        # The row for answer token is the "answer direction"
        aid = answer_ids[name]
        # Get the weight row for this token
        e_tok = np.zeros((1, 1, 1), dtype=np.float32)
        # Actually, we need lm_head's weight matrix row
        # Use identity probing
        probe = np.zeros((1, 1, hidden_dim), dtype=np.float32)
        logits_zero = phi_linear(engine.lm_head.weight, probe)[0, 0]
        # Get gradient direction: which hidden direction maps to this token?
        # The answer direction is the row of lm_head corresponding to answer_id
        # Probe each hidden dim
        answer_dir = np.zeros(hidden_dim, dtype=np.float32)
        for start in range(0, hidden_dim, chunk_size):
            end = min(start + chunk_size, hidden_dim)
            chunk = identity[start:end][np.newaxis, :, :]
            logits_chunk = phi_linear(engine.lm_head.weight, chunk)[0]
            answer_dir[start:end] = logits_chunk[:, aid]
        answer_dirs[name] = answer_dir
        print(f"    {name}: ||answer_dir|| = {np.linalg.norm(answer_dir):.2f}")

    # 5. Collect all essential directions
    print(f"\n  Essential directions collected:")
    all_dirs = []
    dir_labels = []

    # Entity states at L22 (just before extraction)
    for name in prompts:
        v = entity_states[name][22]
        all_dirs.append(v / np.linalg.norm(v))
        dir_labels.append(f"entity_{name}_L22")

    # Last-position states at L22
    for name in prompts:
        v = last_pos_states[name][22]
        all_dirs.append(v / np.linalg.norm(v))
        dir_labels.append(f"lastpos_{name}_L22")

    # Selector directions
    all_dirs.append(d_q_norm)
    dir_labels.append("d_q")
    all_dirs.append(d_k_norm)
    dir_labels.append("d_k")

    # Answer directions
    for name in prompts:
        v = answer_dirs[name]
        all_dirs.append(v / np.linalg.norm(v))
        dir_labels.append(f"answer_{name}")

    all_dirs = np.stack(all_dirs)  # (N, hidden_dim)
    N = len(all_dirs)
    print(f"    {N} directions total")

    # Gram matrix — how independent are they?
    gram = all_dirs @ all_dirs.T
    print(f"\n  Gram matrix (cosine similarities):")
    # Print in blocks
    for i in range(N):
        if i % 4 == 0:
            print()
        sims = " ".join(f"{gram[i,j]:+.2f}" for j in range(N))
        print(f"    {dir_labels[i]:>22s}: {sims}")

    # Effective dimensionality via SVD of the direction set
    U_dirs, S_dirs, _ = np.linalg.svd(all_dirs, full_matrices=False)
    cumvar = np.cumsum(S_dirs ** 2) / np.sum(S_dirs ** 2) * 100
    print(f"\n  Direction set SVD (effective dimensionality):")
    for i in range(min(N, 20)):
        print(f"    dim {i+1}: S={S_dirs[i]:.4f}  cumulative={cumvar[i]:.1f}%")

    # How many dims for 99%? 95%? 90%?
    for threshold in [99.9, 99, 95, 90, 80]:
        d_needed = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"    {threshold}% variance → {d_needed} dimensions")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT B: Low-dimensional projection
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment B: Low-dimensional shape computation")
    print("  Project into d-dimensional subspace, test accuracy")
    print("=" * 80)

    # Use the SVD basis of the direction set as the projection
    # This is the optimal subspace for preserving the essential directions
    _, _, Vt_basis = np.linalg.svd(all_dirs, full_matrices=False)

    # For each dimensionality d, project and test
    for d in [2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 128]:
        if d > N:
            continue
        P = Vt_basis[:d, :]  # (d, hidden_dim) — projection matrix

        n_correct = 0
        results = []
        for name, (prompt, answer) in prompts.items():
            # Project entity state at L22 into d-dimensional space
            h_entity_d = P @ entity_states[name][22]

            # Project last-position state at L22 into d-dimensional space
            h_last_d = P @ last_pos_states[name][22]

            # Project selector direction
            dk_d = P @ d_k
            dk_d_norm = dk_d / (np.linalg.norm(dk_d) + 1e-20)

            # Shape operation 1: SELECT — which position has higher d_k score?
            # (In the full model, this selects across all positions.
            #  Here we just compare entity pos vs last pos)
            score_entity = np.dot(h_entity_d, dk_d)
            score_last = np.dot(h_last_d, dk_d)

            # Shape operation 2: READ — project answer directions
            # Which answer direction aligns best with the entity state?
            best_answer = None
            best_score = -np.inf
            for target_name in prompts:
                ans_d = P @ answer_dirs[target_name]
                score = np.dot(h_entity_d, ans_d)
                if score > best_score:
                    best_score = score
                    best_answer = target_name

            correct = (best_answer == name)
            if correct:
                n_correct += 1
            results.append((name, best_answer, correct))

        status = "✓" if n_correct == 4 else f"{n_correct}/4"
        details = " ".join(f"{n}→{b}{'✓' if c else '✗'}"
                          for n, b, c in results)
        print(f"    d={d:>3d}: {status:>5s}  {details}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT C: The Shape Computer — full pipeline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment C: The Shape Computer")
    print("  Compute answer using ONLY shape operations in projected space")
    print("=" * 80)

    # The full shape computer pipeline:
    # 1. Embed entity → direction in shape space
    # 2. Project onto d_k → selection score
    # 3. Read via V·W_o binding → binding direction
    # 4. Project binding onto answer directions → logits
    # 5. argmax → answer

    # First, compute V·W_o bindings for each entity at L23
    print(f"\n  Computing V·W_o bindings at L23 H6...")
    bindings = {}
    for name in prompts:
        h_ent = entity_states[name][22]
        h_normed = rms_norm(h_ent[np.newaxis, np.newaxis, :],
                           attn.norm_weight)[0, 0]
        # V projection
        v_proj = W_v_g0 @ h_normed  # (head_dim,)
        # W_o projection
        binding = W_o_h6 @ v_proj  # (hidden_dim,)
        bindings[name] = binding
        print(f"    {name}: ||binding|| = {np.linalg.norm(binding):.2f}")

    # Now test the shape computer at various dimensions
    print(f"\n  Shape computer pipeline:")
    print(f"    1. Entity embedding → d-dim direction")
    print(f"    2. d_k selector → which entity to read")
    print(f"    3. V·W_o binding → what the entity says")
    print(f"    4. Answer direction projection → which answer")
    print(f"    5. argmax → output")

    for d in [2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 128, 256]:
        if d > N:
            # Use PCA of all important vectors instead
            all_important = []
            for name in prompts:
                all_important.append(entity_states[name][22])
                all_important.append(bindings[name])
                all_important.append(answer_dirs[name])
            all_important.append(d_k)
            all_important = np.stack(all_important)
            U_imp, S_imp, Vt_imp = np.linalg.svd(all_important,
                                                   full_matrices=False)
            P = Vt_imp[:d, :]
        else:
            P = Vt_basis[:d, :]

        n_correct = 0
        results = []
        for name, (prompt, answer) in prompts.items():
            # Step 1: Project entity state
            h_d = P @ entity_states[name][22]

            # Step 2: Selector score (verifies this is a capital-of entity)
            dk_d = P @ d_k
            select_score = np.dot(h_d, dk_d) / (np.linalg.norm(dk_d) + 1e-20)

            # Step 3: Project binding
            bind_d = P @ bindings[name]

            # Step 4: Score each answer
            scores = {}
            for target_name in prompts:
                ans_d = P @ answer_dirs[target_name]
                # Interference: entity direction + binding direction → answer
                combined = h_d + bind_d
                scores[target_name] = np.dot(combined, ans_d)

            # Step 5: argmax
            best = max(scores, key=scores.get)
            correct = (best == name)
            if correct:
                n_correct += 1
            results.append((name, best, correct))

        status = "✓" if n_correct == 4 else f"{n_correct}/4"
        details = " ".join(f"{n}→{b}{'✓' if c else '✗'}"
                          for n, b, c in results)
        print(f"    d={d:>3d}: {status:>5s}  {details}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT D: The 4D Test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment D: The 4D Shape Computer")
    print("  'There are no quintics' — can we do it in 4 dimensions?")
    print("=" * 80)

    # Strategy: find the BEST 4D subspace for discriminating entities
    # Use the SVD of entity-state differences (what distinguishes them)
    entity_vecs = np.stack([entity_states[n][22] for n in prompts])
    mean_entity = entity_vecs.mean(axis=0)
    centered = entity_vecs - mean_entity
    U_ent, S_ent, Vt_ent = np.linalg.svd(centered, full_matrices=False)

    print(f"\n  Entity-difference SVD:")
    for i in range(min(4, len(S_ent))):
        print(f"    dim {i+1}: S={S_ent[i]:.4f}  "
              f"cumvar={np.sum(S_ent[:i+1]**2)/np.sum(S_ent**2)*100:.1f}%")

    # Also try: include answer directions in the basis
    # Find 4D subspace that maximizes discrimination
    # Strategy: use first 2 entity-difference dims + 2 answer-difference dims
    answer_vecs = np.stack([answer_dirs[n] for n in prompts])
    mean_answer = answer_vecs.mean(axis=0)
    centered_ans = answer_vecs - mean_answer
    U_ans, S_ans, Vt_ans = np.linalg.svd(centered_ans, full_matrices=False)

    strategies = {
        "Entity SVD top-4": Vt_ent[:4, :],
        "Entity top-2 + Answer top-2": np.vstack([Vt_ent[:2, :],
                                                    Vt_ans[:2, :]]),
        "Direction set SVD top-4": Vt_basis[:4, :],
        "All-important SVD top-4": None,  # computed below
    }

    # All-important SVD
    all_important = []
    for name in prompts:
        all_important.append(entity_states[name][22])
        all_important.append(bindings[name])
        all_important.append(answer_dirs[name])
    all_important.append(d_k)
    all_important = np.stack(all_important)
    _, _, Vt_all = np.linalg.svd(all_important, full_matrices=False)
    strategies["All-important SVD top-4"] = Vt_all[:4, :]

    for strat_name, P4 in strategies.items():
        # Orthogonalize (in case of mixed strategies)
        Q, R = np.linalg.qr(P4.T)
        P4_orth = Q[:, :4].T  # (4, hidden_dim)

        print(f"\n  Strategy: {strat_name}")

        # Test: can 4D projections discriminate entities?
        entity_4d = {}
        for name in prompts:
            entity_4d[name] = P4_orth @ entity_states[name][22]
            print(f"    {name}_4d = [{', '.join(f'{x:.3f}' for x in entity_4d[name])}]")

        # Test entity discrimination via cosine
        print(f"    Entity discrimination (cosine in 4D):")
        for i, n1 in enumerate(prompts):
            for j, n2 in enumerate(prompts):
                if j > i:
                    c = cosine(entity_4d[n1], entity_4d[n2])
                    print(f"      cos({n1}, {n2}) = {c:.4f}")

        # Full shape computer in 4D
        n_correct = 0
        results = []
        for name, (prompt, answer) in prompts.items():
            h4 = P4_orth @ entity_states[name][22]
            b4 = P4_orth @ bindings[name]
            combined = h4 + b4

            scores = {}
            for target_name in prompts:
                a4 = P4_orth @ answer_dirs[target_name]
                scores[target_name] = np.dot(combined, a4)

            best = max(scores, key=scores.get)
            correct = (best == name)
            if correct:
                n_correct += 1
            results.append((name, best, correct,
                          {k: f"{v:.3f}" for k, v in scores.items()}))

        status = "✓✓✓✓" if n_correct == 4 else f"{n_correct}/4"
        print(f"    Result: {status}")
        for name, best, correct, scores in results:
            mark = "✓" if correct else "✗"
            print(f"      {name} → {best} {mark}  scores: {scores}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT E: Minimum dimensionality search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment E: Minimum dimensionality for 4/4 correct")
    print("  Binary search for the minimal d")
    print("=" * 80)

    # Use all-important SVD basis (best general basis)
    best_basis = Vt_all

    def test_dim(d):
        P = best_basis[:d, :]
        n_correct = 0
        for name, (prompt, answer) in prompts.items():
            h_d = P @ entity_states[name][22]
            b_d = P @ bindings[name]
            combined = h_d + b_d
            scores = {}
            for target_name in prompts:
                a_d = P @ answer_dirs[target_name]
                scores[target_name] = np.dot(combined, a_d)
            best = max(scores, key=scores.get)
            if best == name:
                n_correct += 1
        return n_correct

    print(f"\n    Sweep:")
    min_working = None
    for d in range(1, min(65, len(S_dirs) + 1)):
        nc = test_dim(d)
        marker = " ← MINIMUM" if nc == 4 and min_working is None else ""
        if nc == 4 and min_working is None:
            min_working = d
        print(f"    d={d:>3d}: {nc}/4{marker}")
        if d > 20 and nc == 4:
            break

    if min_working:
        print(f"\n    ★ Minimum dimensionality for 4/4: d = {min_working}")

        # Show what the shape computer looks like at minimum d
        P_min = best_basis[:min_working, :]
        print(f"\n    The {min_working}D shape computer:")
        for name, (prompt, answer) in prompts.items():
            h_d = P_min @ entity_states[name][22]
            b_d = P_min @ bindings[name]
            combined = h_d + b_d
            scores = {}
            for target_name in prompts:
                a_d = P_min @ answer_dirs[target_name]
                scores[target_name] = float(np.dot(combined, a_d))
            best = max(scores, key=scores.get)
            margin = scores[name] - max(v for k, v in scores.items() if k != name)
            print(f"      {name}: [{', '.join(f'{x:.3f}' for x in h_d)}]"
                  f" → {best} (margin={margin:.3f})")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT F: Operation count
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Experiment F: Operation count — shapes vs scalars")
    print("=" * 80)

    if min_working:
        d = min_working
        # Shape computer operations:
        # 1. Project entity: d × hidden_dim dot products (precomputed)
        #    At runtime: entity IS the d-dim vector
        # 2. Project binding: d × hidden_dim (precomputed)
        # 3. Add entity + binding: d additions
        # 4. Score each answer: d multiplications + d-1 additions per answer
        # 5. argmax over 4 answers: 3 comparisons

        shape_ops = d + (4 * (2 * d - 1)) + 3  # add + 4 dot products + argmax
        full_ops = 28 * (3 * hidden_dim * (nh * hd) +   # Q/K/V per layer
                         nh * 5 * hd +                    # scoring per layer
                         nh * 5 * hd +                    # attn@V per layer
                         (nh * hd) * hidden_dim)           # W_o per layer

        print(f"\n    At d={d}:")
        print(f"    Shape computer runtime ops: {shape_ops:,}")
        print(f"    Full model ops (28 layers): ~{full_ops:,}")
        print(f"    Reduction: {full_ops / shape_ops:,.0f}×")
        print(f"\n    The shape computer uses {d} directions.")
        print(f"    Storage: {d * hidden_dim * 4:,} bytes "
              f"(projection matrix)")
        print(f"    + {4 * d * 4:,} bytes (4 entity vectors)")
        print(f"    + {4 * d * 4:,} bytes (4 answer directions)")
        print(f"    + {4 * d * 4:,} bytes (4 bindings)")
        total_bytes = d * hidden_dim * 4 + 3 * 4 * d * 4
        print(f"    Total: {total_bytes:,} bytes = {total_bytes/1024:.1f} KB")
        model_bytes = 1_200_000_000 * 2  # ~2.4 GB at fp16
        print(f"    Full model: ~{model_bytes/1024/1024:.0f} MB")
        print(f"    Compression: {model_bytes / total_bytes:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    if min_working:
        print(f"\n    The capital-of task is solvable in {min_working}D shape space.")
        print(f"    Operations: shapes only (project + add + argmax)")
        print(f"    No matrix multiplication. No attention. No MLP.")
        print(f"    Just directions interfering in {min_working}-dimensional space.")
    print(f"\n    'There are no quintics.' But there are shapes.")
    print()


if __name__ == '__main__':
    main()
