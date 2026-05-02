#!/usr/bin/env python3
"""
TruthSpace Structure-Aware Gating — Validating the Hypothesis
================================================================

The hypothesis: our gate experiments fail because we treat 3584 dims as
homogeneous. But they're a SUPERPOSITION of 7 geometric structures (DC 276,
DC 279), each needing different operations. Gates applied to the RIGHT
structure should outperform universal VA.

This script:
1. Identifies the Lens subspace (~66d) via SVD of W_o_h for L23 H6
2. Identifies Spectrometer dims via F62 sign rules
3. Decomposes entity→answer deltas into structural subspaces
4. Tests structure-aware composition vs uniform VA

Key question: are relationship deltas MORE CONSISTENT in the Lens subspace
than in the full space? If yes → structure-aware gating should work.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "phi_model"
RULES_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "results" / "phase4_rules_full"
sys.path.insert(0, str(PROJECT_ROOT))

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
PHI_GRID = 128
D_MODEL = 3584
HEAD_DIM = 128
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS


def phi_to_float(signs, exponents):
    """Decode phi-encoded weights to float."""
    return (
        signs.astype(np.float64)
        * PHI ** (exponents.astype(np.float64) / PHI_GRID)
    ).astype(np.float32)


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Concept:
    name: str
    token_id: int
    token_str: str
    embedding: np.ndarray


class VocabSearcher:
    def __init__(self, embeddings, id_to_token):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb_normed = embeddings / (norms + 1e-20)
        self.id_to_token = id_to_token
        self.n_vocab = embeddings.shape[0]

    def rank_of(self, composed, target_tid, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        return int(np.sum(sims > sims[target_tid]))

    def top_k(self, composed, k=5, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self.id_to_token.get(int(i), f"?{i}"), float(sims[i]))
                for i in top_idx]


def load_embeddings_and_concepts():
    """Load output embeddings and build concepts."""
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings...", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file, "r") as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    token_to_id = {tok: idx for tok, idx in vocab.items()}

    def find_token_id(word):
        candidates = [
            word, word.lower(), word.capitalize(), word.upper(),
            f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
            f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}",
            f" {word}", f" {word.lower()}", f" {word.capitalize()}",
        ]
        for c in candidates:
            if c in token_to_id:
                return token_to_id[c], c
        return None, None

    all_words = [
        "France", "Paris", "Germany", "Berlin", "Japan", "Tokyo",
        "China", "Beijing", "Egypt", "Cairo",
        "Australia", "Canberra", "Thailand", "Bangkok", "Poland", "Warsaw",
        "Norway", "Oslo", "Sweden", "Stockholm", "India", "Delhi",
        "Brazil", "Brasilia", "Korea", "Seoul",
        "French", "German", "Japanese", "Chinese", "Spanish",
        "Italy", "Italian", "Portugal", "Portuguese", "Russia", "Russian",
        "Spain",
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "brother", "sister", "son", "daughter",
        "husband", "wife",
    ]

    concepts = {}
    for word in all_words:
        tid, tok_str = find_token_id(word)
        if tid is not None:
            concepts[word] = Concept(
                name=word, token_id=tid, token_str=tok_str,
                embedding=embeddings[tid],
            )

    searcher = VocabSearcher(embeddings, id_to_token)
    print(f"  {len(concepts)} concepts, {searcher.n_vocab} vocab tokens", flush=True)
    return concepts, searcher, embeddings


def load_lens_weights(layer_idx=23, head_idx=6):
    """Load W_v and W_o for a specific head directly from npz files."""
    print(f"Loading Lens weights (L{layer_idx} H{head_idx})...", flush=True)
    layer_dir = MODEL_DIR / f"layer_{layer_idx:02d}"

    v_data = np.load(str(layer_dir / "v_proj.npz"))
    W_v = phi_to_float(v_data['signs'], v_data['exponents'])  # (512, 3584)

    o_data = np.load(str(layer_dir / "o_proj.npz"))
    W_o = phi_to_float(o_data['signs'], o_data['exponents'])  # (3584, 3584)

    b_data = np.load(str(layer_dir / "biases.npz"))
    b_v = b_data['v_proj_bias']  # (512,)

    # Extract head-specific slices
    kv_group = head_idx // HEADS_PER_KV
    W_v_h = W_v[kv_group * HEAD_DIM:(kv_group + 1) * HEAD_DIM, :]  # (128, 3584)
    b_v_h = b_v[kv_group * HEAD_DIM:(kv_group + 1) * HEAD_DIM]     # (128,)
    W_o_h = W_o[:, head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM]  # (3584, 128)

    print(f"  W_v_h: {W_v_h.shape}, W_o_h: {W_o_h.shape}, b_v_h: {b_v_h.shape}", flush=True)
    return W_v_h, W_o_h, b_v_h


def load_f62_rules():
    """Load F62 spectrometer rules for all layers."""
    print("Loading F62 rules...", flush=True)
    all_rules = {}
    for layer_idx in range(28):
        rule_file = RULES_DIR / f"layer_{layer_idx:02d}.json"
        if rule_file.exists():
            with open(rule_file) as f:
                all_rules[layer_idx] = json.load(f)
    print(f"  Loaded rules for {len(all_rules)} layers", flush=True)
    return all_rules


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_lens_subspace(W_o_h):
    """
    Compute the Lens output subspace via SVD of W_o_h.T.

    W_o_h.T is (128, 3584): maps from value space to output space.
    SVD gives: W_o_h.T = U_o @ diag(S_o) @ Vt_o
    where Vt_o rows are the output-space directions.

    Returns:
        Vt_o: (128, 3584) — rows are Lens output directions in ℝ^3584
        S_o: (128,) — singular values (importance of each direction)
        energy: (128,) — cumulative energy fraction
    """
    W_o_T = W_o_h.T  # (128, 3584)
    U_o, S_o, Vt_o = np.linalg.svd(W_o_T, full_matrices=False)
    energy = np.cumsum(S_o**2) / np.sum(S_o**2)
    return Vt_o, S_o, energy


def project_to_subspace(vectors, basis, rank_k):
    """
    Project vectors onto the subspace spanned by basis[:rank_k].

    Args:
        vectors: (..., D) array
        basis: (N, D) orthonormal rows
        rank_k: number of basis vectors to use

    Returns:
        projected: (..., D) — component in subspace
        orthogonal: (..., D) — component orthogonal to subspace
    """
    B = basis[:rank_k]  # (k, D)
    # Project: x_proj = B^T @ B @ x
    coeffs = vectors @ B.T  # (..., k)
    projected = coeffs @ B   # (..., D)
    orthogonal = vectors - projected
    return projected, orthogonal


def compute_subspace_cv(deltas, basis, rank_k):
    """
    Compute the coefficient of variation of deltas projected into a subspace.

    Returns per-dimension CV in the subspace basis coordinates.
    """
    B = basis[:rank_k]  # (k, D)
    # Project deltas into subspace coordinates
    coords = deltas @ B.T  # (n_pairs, k)
    mean_coords = np.mean(coords, axis=0)
    std_coords = np.std(coords, axis=0)
    cv = std_coords / (np.abs(mean_coords) + 1e-20)
    return cv, mean_coords, std_coords


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    t0 = time.time()

    print("=" * 72)
    print("  STRUCTURE-AWARE GATING: VALIDATING THE HYPOTHESIS")
    print("  Do relationship deltas concentrate in the Lens subspace?")
    print("=" * 72)
    print()

    # Load all data
    concepts, searcher, embeddings = load_embeddings_and_concepts()
    W_v_h, W_o_h, b_v_h = load_lens_weights(layer_idx=23, head_idx=6)
    f62_rules = load_f62_rules()

    # ══════════════════════════════════════════════════════════════════
    # PART 1: Lens Subspace Identification
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 1: Lens Subspace Identification (L23 H6)")
    print("─" * 72)

    Vt_o, S_o, energy = compute_lens_subspace(W_o_h)

    print(f"\n  W_o_h SVD — singular values:")
    print(f"    S[0:5]:  {S_o[:5].round(3)}")
    print(f"    S[60:70]: {S_o[60:70].round(3)}")
    print(f"    S[120:128]: {S_o[120:].round(3)}")

    for k in [5, 10, 20, 30, 40, 50, 66, 80, 100, 128]:
        pct = energy[min(k-1, 127)] * 100
        print(f"    rank-{k:3d}: {pct:6.2f}% energy")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: Delta Decomposition by Structure
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 2: Delta Decomposition — Lens vs Orthogonal")
    print("─" * 72)

    relationships = {
        'capital': [
            ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
            ("China", "Beijing"), ("Egypt", "Cairo"),
        ],
        'language': [
            ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
            ("China", "Chinese"), ("Spain", "Spanish"),
            ("Italy", "Italian"), ("Portugal", "Portuguese"), ("Russia", "Russian"),
        ],
        'gender': [
            ("king", "queen"), ("man", "woman"), ("boy", "girl"),
            ("father", "mother"), ("brother", "sister"),
            ("husband", "wife"),
        ],
    }

    test_sets = {
        'capital': [
            ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
            ("Brazil", "Brasilia"),
        ],
        'language': [
            ("Australia", "Canberra"),  # placeholder — language test is below
        ],
        'gender': [
            ("son", "daughter"),
        ],
    }

    # For each relationship, decompose deltas into Lens vs orthogonal
    lens_ranks_to_test = [10, 20, 30, 50, 66, 80, 100, 128]

    for rel_name, pairs in relationships.items():
        train_pairs = [(concepts[e], concepts[a]) for e, a in pairs
                       if e in concepts and a in concepts]
        if len(train_pairs) < 2:
            continue

        deltas = np.array([a.embedding - e.embedding for e, a in train_pairs])
        n_pairs = len(deltas)

        print(f"\n  === {rel_name.upper()} ({n_pairs} pairs) ===")

        # Full-space statistics
        mean_delta = np.mean(deltas, axis=0)
        std_delta = np.std(deltas, axis=0)
        full_cv = std_delta / (np.abs(mean_delta) + 1e-20)
        print(f"\n  Full space (3584d):")
        print(f"    Mean |delta|:   {np.mean(np.abs(mean_delta)):.6f}")
        print(f"    Mean CV:        {np.mean(full_cv):.2f}")
        print(f"    Median CV:      {np.median(full_cv):.2f}")
        print(f"    CV < 0.5:       {np.sum(full_cv < 0.5)} dims ({np.sum(full_cv < 0.5)/D_MODEL*100:.1f}%)")
        print(f"    CV < 1.0:       {np.sum(full_cv < 1.0)} dims ({np.sum(full_cv < 1.0)/D_MODEL*100:.1f}%)")

        # Decompose into Lens subspace at various ranks
        print(f"\n  {'Rank':>6s}  {'LensCV_mean':>11s}  {'LensCV_med':>10s}  "
              f"{'OrthCV_mean':>11s}  {'OrthCV_med':>10s}  "
              f"{'Lens_low':>8s}  {'Orth_low':>8s}  "
              f"{'Lens_E%':>7s}")
        print("  " + "─" * 90)

        for rank_k in lens_ranks_to_test:
            # Project deltas into Lens subspace and orthogonal complement
            lens_deltas, orth_deltas = project_to_subspace(deltas, Vt_o, rank_k)

            # CV in Lens subspace coordinates
            lens_cv, lens_mean, lens_std = compute_subspace_cv(deltas, Vt_o, rank_k)

            # CV in orthogonal complement (raw dims)
            orth_mean_delta = np.mean(orth_deltas, axis=0)
            orth_std_delta = np.std(orth_deltas, axis=0)
            orth_cv = orth_std_delta / (np.abs(orth_mean_delta) + 1e-20)

            # How much of the delta energy is in the Lens subspace?
            lens_energy = np.mean(np.sum(lens_deltas**2, axis=1))
            total_energy = np.mean(np.sum(deltas**2, axis=1))
            lens_frac = lens_energy / total_energy * 100

            # Count low-CV dims
            n_lens_low = np.sum(lens_cv < 1.0)
            n_orth_low = np.sum(orth_cv < 1.0)

            print(f"  {rank_k:6d}  {np.mean(lens_cv):11.2f}  {np.median(lens_cv):10.2f}  "
                  f"{np.mean(orth_cv):11.2f}  {np.median(orth_cv):10.2f}  "
                  f"{n_lens_low:>5d}/{rank_k:<3d}  {n_orth_low:>5d}/{D_MODEL-rank_k:<4d}  "
                  f"{lens_frac:6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: Structure-Aware Composition — The Hypothesis Test
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 3: Structure-Aware Composition vs Uniform VA")
    print("─" * 72)

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    capital_test = [
        ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
        ("Brazil", "Brasilia"),
    ]

    language_train = [
        ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
        ("China", "Chinese"), ("Spain", "Spanish"),
    ]
    language_test = [
        ("Italy", "Italian"), ("Portugal", "Portuguese"), ("Russia", "Russian"),
    ]

    for rel_name, train_set, test_set in [
        ("capital", capital_train, capital_test),
        ("language", language_train, language_test),
    ]:
        train_pairs = [(concepts[e], concepts[a]) for e, a in train_set
                       if e in concepts and a in concepts]
        if len(train_pairs) < 2:
            continue

        deltas = np.array([a.embedding - e.embedding for e, a in train_pairs])
        avg_delta = np.mean(deltas, axis=0)

        print(f"\n  === {rel_name.upper()} PREDICTION ===")
        print(f"  Training: {len(train_pairs)} pairs")

        # For each Lens rank, try:
        # 1. VA (uniform): entity + avg_delta
        # 2. Lens-only: project avg_delta into Lens subspace, apply only that
        # 3. Lens-gate + VA-orth: learned delta in Lens subspace + VA in orthogonal
        # 4. Lens-gate + entity-orth: learned delta in Lens + keep entity in orth
        # 5. Lens-gate + zero-orth: only Lens component (no orthogonal)

        # Pre-compute Lens and Orth components of avg_delta
        print(f"\n  {'Method':>30s}  {'Rank':>4s}", end="")
        for e_name, _ in test_set:
            if e_name in concepts:
                print(f"  {e_name[:5]:>5s}", end="")
        print(f"  {'Mean':>6s}  {'Med':>5s}")
        print("  " + "─" * (40 + 7 * min(len(test_set), 7) + 15))

        for rank_k in [10, 20, 30, 50, 66, 100, 128]:
            # Decompose avg_delta
            delta_lens, delta_orth = project_to_subspace(
                avg_delta[np.newaxis, :], Vt_o, rank_k
            )
            delta_lens = delta_lens[0]
            delta_orth = delta_orth[0]

            # Also compute per-dim mean delta in Lens subspace coordinates
            lens_coords = deltas @ Vt_o[:rank_k].T  # (n_pairs, rank_k)
            lens_mean_coords = np.mean(lens_coords, axis=0)  # (rank_k,)
            lens_cv_coords = np.std(lens_coords, axis=0) / (np.abs(lens_mean_coords) + 1e-20)

            methods = {}

            # Method 0: VA baseline (same for all ranks)
            if rank_k == 10:  # only compute once
                va_ranks_all = []
                for e_name, a_name in test_set:
                    e, a = concepts.get(e_name), concepts.get(a_name)
                    if not e or not a:
                        continue
                    pred = e.embedding + avg_delta
                    va_ranks_all.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
                methods['VA (uniform)'] = va_ranks_all

            # Method 1: Lens-only delta (zero orthogonal component)
            ranks_lens_only = []
            for e_name, a_name in test_set:
                e, a = concepts.get(e_name), concepts.get(a_name)
                if not e or not a:
                    continue
                pred = e.embedding + delta_lens
                ranks_lens_only.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
            methods['Lens-only'] = ranks_lens_only

            # Method 2: Lens delta + VA orthogonal
            ranks_lens_va = []
            for e_name, a_name in test_set:
                e, a = concepts.get(e_name), concepts.get(a_name)
                if not e or not a:
                    continue
                pred = e.embedding + delta_lens + delta_orth
                ranks_lens_va.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
            methods['Lens + VA-orth'] = ranks_lens_va

            # Method 3: Lens delta + keep entity on orthogonal
            # (don't add any delta on the orthogonal component)
            ranks_lens_entity = []
            for e_name, a_name in test_set:
                e, a = concepts.get(e_name), concepts.get(a_name)
                if not e or not a:
                    continue
                # Project entity into Lens and orth
                e_lens, e_orth = project_to_subspace(
                    e.embedding[np.newaxis, :], Vt_o, rank_k
                )
                # Prediction: shift in Lens, keep entity in orth
                pred = (e_lens[0] + delta_lens) + e_orth[0]
                ranks_lens_entity.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
            methods['Lens-shift + entity-orth'] = ranks_lens_entity

            # Method 4: Per-dim learned delta in Lens coords (low-CV only) + VA orth
            # Use only Lens coordinates with CV < 1.0
            confident_mask = lens_cv_coords < 1.0
            ranks_lens_confident = []
            for e_name, a_name in test_set:
                e, a = concepts.get(e_name), concepts.get(a_name)
                if not e or not a:
                    continue
                # Project entity into Lens coords
                e_coords = e.embedding @ Vt_o[:rank_k].T  # (rank_k,)
                # Shift only confident coordinates
                new_coords = e_coords.copy()
                new_coords[confident_mask] += lens_mean_coords[confident_mask]
                # Reconstruct Lens component
                pred_lens = new_coords @ Vt_o[:rank_k]  # (3584,)
                # Add orthogonal component (VA delta on orth)
                _, e_orth = project_to_subspace(
                    e.embedding[np.newaxis, :], Vt_o, rank_k
                )
                pred = pred_lens + e_orth[0] + delta_orth
                ranks_lens_confident.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
            methods[f'Lens-confident(CV<1) + VA-orth'] = ranks_lens_confident

            # Print results
            for method_name, ranks in methods.items():
                if not ranks:
                    continue
                rk_str = str(rank_k) if method_name != 'VA (uniform)' else " all"
                print(f"  {method_name:>30s}  {rk_str:>4s}", end="")
                for r in ranks[:7]:
                    print(f"  {r:5d}", end="")
                print(f"  {np.mean(ranks):6.1f}  {np.median(ranks):5.0f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: F62 Spectrometer Dims — Structural Cross-Reference
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 4: F62 Spectrometer × Lens Subspace Overlap")
    print("─" * 72)

    # Classify each dim by its F62 consensus rule across layers
    sign_rules = ['sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate']

    # For each dim, count how many layers classify it as each rule type
    dim_rule_counts = np.zeros((D_MODEL, len(sign_rules)), dtype=int)
    dim_affine_count = np.zeros(D_MODEL, dtype=int)

    for layer_idx, rules in f62_rules.items():
        layer_idx = int(layer_idx)
        # Rules stored as 'dim_rules' list, each entry has 'rule_type'
        dim_rules_list = rules.get('dim_rules', [])
        if not dim_rules_list:
            continue
        for dim_info in dim_rules_list:
            d = dim_info.get('global_dim', dim_info.get('local_dim', -1))
            if d < 0 or d >= D_MODEL:
                continue
            rt = dim_info.get('rule_type', 'unstructured')
            if rt in sign_rules:
                dim_rule_counts[d, sign_rules.index(rt)] += 1
            elif rt == 'affine':
                dim_affine_count[d] += 1

    # Consensus: dim's most common sign rule
    has_sign_rule = np.sum(dim_rule_counts, axis=1) > 0
    consensus_rule = np.argmax(dim_rule_counts, axis=1)
    max_sign_count = np.max(dim_rule_counts, axis=1)

    print(f"\n  F62 dim classification summary:")
    print(f"    Dims with any sign rule: {np.sum(has_sign_rule)} / {D_MODEL}")
    for i, rule in enumerate(sign_rules):
        n = np.sum(consensus_rule[has_sign_rule] == i)
        print(f"    Consensus {rule:>15s}: {n} dims")
    print(f"    Dims with affine count > 0: {np.sum(dim_affine_count > 0)}")

    # Key question: how much of the Lens subspace energy is on sign-rule dims?
    # Project each Vt_o row onto the sign-rule dim mask
    sign_preserve_dims = np.where(
        has_sign_rule & (consensus_rule == sign_rules.index('sign_preserve'))
    )[0]
    sign_flip_dims = np.where(
        has_sign_rule & (consensus_rule == sign_rules.index('sign_flip'))
    )[0]
    no_sign_dims = np.where(~has_sign_rule)[0]

    print(f"\n  Lens subspace energy on F62-classified dims:")
    for rank_k in [10, 30, 66, 128]:
        basis = Vt_o[:rank_k]  # (k, 3584)
        # Energy of basis on sign_preserve dims
        sp_energy = np.sum(basis[:, sign_preserve_dims]**2) / np.sum(basis**2) * 100
        sf_energy = np.sum(basis[:, sign_flip_dims]**2) / np.sum(basis**2) * 100
        ns_energy = np.sum(basis[:, no_sign_dims]**2) / np.sum(basis**2) * 100
        print(f"    rank-{rank_k:3d}: sign_preserve={sp_energy:5.1f}%, "
              f"sign_flip={sf_energy:5.1f}%, no_sign={ns_energy:5.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: Combined Structure-Aware Gate
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 5: Combined — Lens Gate + Sign Rules on Spectrometer Dims")
    print("─" * 72)

    # For capital relationship:
    # 1. In Lens subspace: use learned delta (confident coords only)
    # 2. On sign_flip dims (outside Lens): flip sign + VA magnitude
    # 3. On sign_preserve dims (outside Lens): preserve sign + VA magnitude
    # 4. Remainder: VA delta

    capital_train_pairs = [(concepts[e], concepts[a]) for e, a in capital_train
                           if e in concepts and a in concepts]
    cap_deltas = np.array([a.embedding - e.embedding for e, a in capital_train_pairs])
    cap_avg_delta = np.mean(cap_deltas, axis=0)

    # Sign flip/keep rates from training data
    entity_signs = np.array([np.sign(e.embedding) for e, _ in capital_train_pairs])
    answer_signs = np.array([np.sign(a.embedding) for _, a in capital_train_pairs])
    sign_products = entity_signs * answer_signs
    flip_rate = np.mean(sign_products < 0, axis=0)
    keep_rate = np.mean(sign_products > 0, axis=0)

    print(f"\n  {'Method':>40s}", end="")
    for e_name, _ in capital_test:
        if e_name in concepts:
            print(f"  {e_name[:5]:>5s}", end="")
    print(f"  {'Mean':>6s}  {'Med':>5s}")
    print("  " + "─" * (45 + 7 * 7 + 15))

    for rank_k in [30, 50, 66, 100]:
        # Decompose avg_delta
        delta_lens, delta_orth = project_to_subspace(
            cap_avg_delta[np.newaxis, :], Vt_o, rank_k
        )
        delta_lens = delta_lens[0]
        delta_orth = delta_orth[0]

        # Lens subspace coordinates
        lens_coords_all = cap_deltas @ Vt_o[:rank_k].T
        lens_mean_coords = np.mean(lens_coords_all, axis=0)
        lens_cv = np.std(lens_coords_all, axis=0) / (np.abs(lens_mean_coords) + 1e-20)
        confident = lens_cv < 1.0

        # Combined structure-aware gate
        ranks_combined = []
        for e_name, a_name in capital_test:
            e, a = concepts.get(e_name), concepts.get(a_name)
            if not e or not a:
                continue
            ex = {e.token_id}

            # Start with VA as base
            pred = e.embedding + cap_avg_delta

            # Override Lens subspace: use confident learned shifts
            e_lens_coords = e.embedding @ Vt_o[:rank_k].T
            new_lens_coords = e_lens_coords.copy()
            new_lens_coords[confident] += lens_mean_coords[confident]
            # Don't shift non-confident Lens coords — keep entity value
            pred_lens = new_lens_coords @ Vt_o[:rank_k]

            # Replace Lens component in prediction
            _, pred_orth = project_to_subspace(pred[np.newaxis, :], Vt_o, rank_k)
            pred = pred_lens + pred_orth[0]

            # On sign dims outside Lens: enforce sign from relationship
            # (Only on dims where flip_rate > 0.7 — the safe threshold)
            flip_dims = np.where(flip_rate >= 0.7)[0]
            keep_dims_sign = np.where(keep_rate >= 0.7)[0]
            pred[flip_dims] = np.abs(pred[flip_dims]) * (-np.sign(e.embedding[flip_dims] + 1e-20))
            pred[keep_dims_sign] = np.abs(pred[keep_dims_sign]) * np.sign(e.embedding[keep_dims_sign] + 1e-20)

            ranks_combined.append(searcher.rank_of(pred, a.token_id, ex))

        # VA baseline for comparison
        va_ranks = []
        for e_name, a_name in capital_test:
            e, a = concepts.get(e_name), concepts.get(a_name)
            if not e or not a:
                continue
            pred = e.embedding + cap_avg_delta
            va_ranks.append(searcher.rank_of(pred, a.token_id, {e.token_id}))

        if rank_k == 30:
            print(f"  {'VA (uniform baseline)':>40s}  ", end="")
            for r in va_ranks[:7]:
                print(f"  {r:5d}", end="")
            print(f"  {np.mean(va_ranks):6.1f}  {np.median(va_ranks):5.0f}")

        print(f"  {f'Combined (Lens-{rank_k} + sign + VA)':>40s}  ", end="")
        for r in ranks_combined[:7]:
            print(f"  {r:5d}", end="")
        print(f"  {np.mean(ranks_combined):6.1f}  {np.median(ranks_combined):5.0f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 6: Relationship's OWN Subspace (SVD of deltas)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART 6: Relationship Subspace via SVD of Deltas")
    print("  (If the Lens doesn't capture it, what DOES?)")
    print("─" * 72)

    for rel_name, train_set, test_set in [
        ("capital", capital_train, capital_test),
        ("language", language_train, language_test),
    ]:
        train_pairs = [(concepts[e], concepts[a]) for e, a in train_set
                       if e in concepts and a in concepts]
        if len(train_pairs) < 2:
            continue

        deltas = np.array([a.embedding - e.embedding for e, a in train_pairs])
        avg_delta = np.mean(deltas, axis=0)
        n_train = len(deltas)

        # SVD of the delta matrix: find the subspace where deltas live
        # deltas is (n_pairs, 3584)
        U_d, S_d, Vt_d = np.linalg.svd(deltas, full_matrices=False)
        # Vt_d rows are the principal directions of variation in deltas
        # S_d values indicate how much variance in each direction

        delta_energy = np.cumsum(S_d**2) / np.sum(S_d**2)

        print(f"\n  === {rel_name.upper()} ===  ({n_train} training pairs)")
        print(f"  Delta matrix SVD:")
        print(f"    Singular values: {S_d.round(4)}")
        print(f"    Energy: {(delta_energy * 100).round(1)}")

        # The key question: does projecting into the delta's OWN subspace
        # capture the relationship better than raw dims?
        print(f"\n  {'Method':>35s}", end="")
        for e_name, _ in test_set:
            if e_name in concepts:
                print(f"  {e_name[:5]:>5s}", end="")
        print(f"  {'Mean':>6s}  {'Med':>5s}")
        print("  " + "─" * (40 + 7 * min(len(test_set), 7) + 15))

        # VA baseline
        va_ranks = []
        for e_name, a_name in test_set:
            e, a = concepts.get(e_name), concepts.get(a_name)
            if not e or not a:
                continue
            pred = e.embedding + avg_delta
            va_ranks.append(searcher.rank_of(pred, a.token_id, {e.token_id}))
        print(f"  {'VA (uniform, all 3584d)':>35s}", end="")
        for r in va_ranks[:7]:
            print(f"  {r:5d}", end="")
        print(f"  {np.mean(va_ranks):6.1f}  {np.median(va_ranks):5.0f}")

        # Test: use avg_delta projected into delta-subspace of rank k
        for rank_k in range(1, min(n_train + 1, 6)):
            # Project avg_delta into the top-k delta directions
            B = Vt_d[:rank_k]  # (k, 3584)
            delta_proj = (avg_delta @ B.T) @ B  # project avg_delta into delta subspace
            delta_orth = avg_delta - delta_proj

            # Method A: use only projected delta
            ranks_proj = []
            for e_name, a_name in test_set:
                e, a = concepts.get(e_name), concepts.get(a_name)
                if not e or not a:
                    continue
                pred = e.embedding + delta_proj
                ranks_proj.append(searcher.rank_of(pred, a.token_id, {e.token_id}))

            print(f"  {f'Delta-subspace rank-{rank_k} only':>35s}", end="")
            for r in ranks_proj[:7]:
                print(f"  {r:5d}", end="")
            print(f"  {np.mean(ranks_proj):6.1f}  {np.median(ranks_proj):5.0f}")

        # Also test: what if we remove the top-1 direction (the "mean" direction)
        # and gate in the remaining variance directions?
        # Top-1 is the average direction. Top-2+ are the entity-specific variation.
        B1 = Vt_d[:1]  # (1, 3584) — mean direction
        mean_component = (avg_delta @ B1.T) @ B1
        residual_delta = avg_delta - mean_component

        print(f"\n  Delta SVD direction 1 (mean dir):")
        print(f"    |mean_component| = {np.linalg.norm(mean_component):.4f}")
        print(f"    |residual|       = {np.linalg.norm(residual_delta):.4f}")
        print(f"    |avg_delta|      = {np.linalg.norm(avg_delta):.4f}")
        print(f"    cos(avg_delta, dir1) = {np.dot(avg_delta, Vt_d[0]) / (np.linalg.norm(avg_delta) + 1e-20):.4f}")

        # How much of Lens subspace overlaps with delta subspace?
        for rank_k in [1, 2, min(n_train, 5)]:
            B_delta = Vt_d[:rank_k]  # (k, 3584)
            for lens_rank in [10, 66, 128]:
                B_lens = Vt_o[:lens_rank]  # (lens_rank, 3584)
                # Compute overlap: how much of delta-subspace lies within Lens-subspace?
                # Project delta basis onto Lens basis
                overlap_matrix = B_delta @ B_lens.T  # (k, lens_rank)
                # Frobenius norm of overlap / max possible
                overlap_frac = np.sum(overlap_matrix**2) / rank_k * 100
                if rank_k == 1:
                    print(f"    Overlap(delta-{rank_k}, lens-{lens_rank}): {overlap_frac:.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"\n  Key questions answered:")
    print(f"    1. Are deltas more consistent in Lens subspace? → See Part 2")
    print(f"    2. Does structure-aware composition beat VA?    → See Part 3 & 5")
    print(f"    3. Do F62 dims overlap with Lens subspace?      → See Part 4")
    print(f"    4. What subspace DO deltas live in?             → See Part 6")
    print()


if __name__ == "__main__":
    main()
