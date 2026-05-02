#!/usr/bin/env python3
"""
TruthSpace Platonic Residual Test
===================================

THE QUESTION: Are concepts purely compounds of platonic ideals?

If so, a concept's embedding should be fully reconstructable from its
projections along truth axes. The residual (original - reconstruction)
should be unstructured noise.

If NOT, the residual will contain structured information — geometric
content that truth axes alone don't capture.

Test:
1. Load embeddings and compute anchor directions (truth axes)
2. For each concept, project onto all truth axes → reconstruction
3. Residual = original - reconstruction
4. Analyze: is the residual structured or noise?

Metrics:
- Variance explained: how much of the embedding's energy lives in truth axes?
- Residual structure: can residuals predict concept identity? (if yes → structured)
- Residual similarity: do concepts with similar meanings have similar residuals?
- Dimensionality: what is the effective rank of the residual space?
- Nearest neighbor: can residuals alone retrieve the correct concept?
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse infrastructure from the delta readability script
from explore_delta_readability import (
    load_embeddings_and_concepts,
    define_relationships,
    compute_delta,
    FieldNotes,
    VocabSearcher,
)

NOTES_PATH = SCRIPT_DIR / "platonic_residual_notes.md"


# =============================================================================
# ANCHOR COMPUTATION (reproduced from explore_delta_readability Part 3)
# =============================================================================

def compute_anchors(concepts, searcher):
    """Compute anchor directions and thresholds for all truth axes."""

    anchors_def = {
        "is_european_country": {
            "positive": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Greece", "Ireland",
                         "Finland", "Denmark", "Austria", "Belgium", "Netherlands",
                         "Switzerland", "Russia"],
            "negative": ["Japan", "China", "Egypt", "Australia", "Thailand",
                         "India", "Brazil", "Korea", "Turkey", "Nigeria",
                         "Kenya", "Morocco", "Israel", "Iran", "Vietnam",
                         "Indonesia", "Philippines", "Mexico", "Canada",
                         "Argentina", "Chile", "Colombia", "Peru"],
        },
        "is_asian_country": {
            "positive": ["Japan", "China", "Thailand", "India", "Korea",
                         "Vietnam", "Indonesia", "Philippines", "Malaysia",
                         "Singapore", "Iran", "Iraq", "Israel"],
            "negative": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Egypt", "Australia",
                         "Brazil", "Nigeria", "Kenya", "Morocco",
                         "Mexico", "Canada", "Argentina"],
        },
        "is_capital_city": {
            "positive": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                         "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
                         "Delhi", "Seoul", "Rome", "Lisbon", "Moscow",
                         "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
                         "Copenhagen", "Vienna", "Brussels", "Amsterdam",
                         "Ottawa", "Lima", "Tehran", "Baghdad", "Hanoi"],
            "negative": ["France", "Germany", "Japan", "China", "Egypt",
                         "Australia", "Thailand", "Poland", "Norway", "Sweden",
                         "India", "Korea", "Italy", "Portugal", "Russia",
                         "Spain", "Greece", "Turkey", "Ireland",
                         "Brazil", "Mexico", "Canada", "Argentina"],
        },
        "is_romance_language": {
            "positive": ["French", "Italian", "Portuguese", "Spanish"],
            "negative": ["German", "Japanese", "Chinese", "Arabic", "English",
                         "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                         "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Danish", "Persian", "Vietnamese"],
        },
        "is_germanic_language": {
            "positive": ["German", "English", "Dutch", "Norwegian", "Swedish",
                         "Danish"],
            "negative": ["French", "Italian", "Portuguese", "Spanish",
                         "Japanese", "Chinese", "Arabic", "Korean",
                         "Polish", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Thai", "Persian", "Vietnamese"],
        },
        "is_female_gendered": {
            "positive": ["queen", "woman", "girl", "mother", "sister",
                         "daughter", "wife", "aunt", "princess", "actress",
                         "waitress", "heroine"],
            "negative": ["king", "man", "boy", "father", "brother",
                         "son", "husband", "uncle", "prince", "actor",
                         "waiter", "hero"],
        },
    }

    anchor_directions = {}

    for anchor_name, anchor_def in anchors_def.items():
        pos_tids = []
        for word in anchor_def["positive"]:
            if word in concepts:
                pos_tids.append(concepts[word].token_id)

        neg_tids = []
        for word in anchor_def["negative"]:
            if word in concepts:
                neg_tids.append(concepts[word].token_id)

        if len(pos_tids) < 2 or len(neg_tids) < 2:
            continue

        pos_embs = np.array([searcher.embeddings[tid] for tid in pos_tids])
        neg_embs = np.array([searcher.embeddings[tid] for tid in neg_tids])
        anchor_dir = np.mean(pos_embs, axis=0) - np.mean(neg_embs, axis=0)
        anchor_dir_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-20)

        # Compute threshold
        pos_projs = [np.dot(searcher.embeddings[tid], anchor_dir_norm) for tid in pos_tids]
        neg_projs = [np.dot(searcher.embeddings[tid], anchor_dir_norm) for tid in neg_tids]
        threshold = (np.mean(pos_projs) + np.mean(neg_projs)) / 2

        anchor_directions[anchor_name] = {
            "direction": anchor_dir_norm,
            "raw_direction": anchor_dir,
            "threshold": threshold,
            "pos_tids": pos_tids,
            "neg_tids": neg_tids,
        }

    return anchor_directions


# =============================================================================
# PART 1: RECONSTRUCTION TEST
# =============================================================================

def part1_reconstruction(concepts, searcher, anchor_directions, notes):
    """Project each concept onto truth axes and measure reconstruction quality."""

    notes.section("1. Reconstruction Test — How Much Do Truth Axes Explain?")

    anchor_names = sorted(anchor_directions.keys())
    n_anchors = len(anchor_names)

    # Build the truth-axis basis matrix: (n_anchors, d_model)
    basis = np.array([anchor_directions[name]["direction"] for name in anchor_names])
    notes.observe(f"Truth axis basis: {n_anchors} directions in R^{basis.shape[1]}")

    # Gram matrix — how orthogonal are these axes really?
    gram = basis @ basis.T
    notes.observe(f"\nGram matrix (should be near-identity if orthogonal):")
    header = [""] + [a.replace("is_", "")[:10] for a in anchor_names]
    for i, name_i in enumerate(anchor_names):
        row = [name_i.replace("is_", "")[:10]]
        for j in range(n_anchors):
            row.append(f"{gram[i,j]:.3f}")
        notes.observe(f"  {'  '.join(row)}")

    # Orthogonalize the basis using Gram-Schmidt for fair comparison
    # (non-orthogonal basis inflates explained variance via shared components)
    Q, R_qr = np.linalg.qr(basis.T)  # Q: (d_model, n_anchors), orthonormal
    notes.observe(f"\nOrthogonalized basis: {Q.shape[1]} orthonormal directions")

    # For each concept: project, reconstruct, measure residual
    test_groups = {
        "Countries": ["France", "Germany", "Japan", "China", "Egypt",
                       "Australia", "India", "Brazil", "Korea", "Italy",
                       "Spain", "Russia", "Poland", "Norway", "Sweden",
                       "Turkey", "Greece", "Ireland", "Finland", "Denmark",
                       "Mexico", "Canada", "Argentina", "Nigeria", "Kenya"],
        "Capitals": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                      "Canberra", "Delhi", "Seoul", "Rome", "Lisbon",
                      "Moscow", "Madrid", "Athens", "Ankara", "Dublin",
                      "Helsinki", "Copenhagen", "Vienna", "Warsaw", "Oslo",
                      "Stockholm", "Ottawa", "Lima"],
        "Languages": ["French", "German", "Japanese", "Chinese", "Spanish",
                       "Italian", "Portuguese", "Russian", "Arabic", "English",
                       "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                       "Dutch", "Greek", "Turkish", "Hindi", "Finnish"],
        "Gender (M)": ["king", "man", "boy", "father", "brother", "son",
                        "husband", "uncle", "prince", "actor"],
        "Gender (F)": ["queen", "woman", "girl", "mother", "sister",
                        "daughter", "wife", "aunt", "princess", "actress"],
    }

    all_results = []
    group_results = {}

    for group_name, words in test_groups.items():
        notes.observe(f"\n### {group_name}")
        rows = []
        group_vars = []

        for word in words:
            if word not in concepts:
                continue

            emb = concepts[word].embedding
            emb_norm_sq = np.dot(emb, emb)

            # Project onto orthonormal basis
            coeffs = Q.T @ emb  # (n_anchors,) — projection coefficients
            reconstruction = Q @ coeffs  # (d_model,) — reconstructed embedding
            residual = emb - reconstruction

            # Metrics
            recon_norm_sq = np.dot(reconstruction, reconstruction)
            resid_norm_sq = np.dot(residual, residual)
            var_explained = recon_norm_sq / (emb_norm_sq + 1e-20)

            rows.append({
                "word": word,
                "emb_norm": np.sqrt(emb_norm_sq),
                "recon_norm": np.sqrt(recon_norm_sq),
                "resid_norm": np.sqrt(resid_norm_sq),
                "var_explained": var_explained,
                "coeffs": coeffs,
                "residual": residual,
                "embedding": emb,
            })
            group_vars.append(var_explained)
            all_results.append(rows[-1])

        # Summary table
        for r in rows:
            notes.observe(f"  {r['word']:15s}  ||emb||={r['emb_norm']:.3f}  "
                         f"||recon||={r['recon_norm']:.3f}  ||resid||={r['resid_norm']:.3f}  "
                         f"var_explained={r['var_explained']*100:.2f}%")

        mean_var = np.mean(group_vars) if group_vars else 0
        notes.observe(f"  **Group mean variance explained: {mean_var*100:.2f}%**")
        group_results[group_name] = mean_var

    # Overall summary
    all_vars = [r["var_explained"] for r in all_results]
    notes.observe(f"\n### Overall Reconstruction Summary")
    notes.observe(f"  Concepts tested: {len(all_results)}")
    notes.observe(f"  Mean variance explained by {n_anchors} truth axes: {np.mean(all_vars)*100:.3f}%")
    notes.observe(f"  Median: {np.median(all_vars)*100:.3f}%")
    notes.observe(f"  Min: {np.min(all_vars)*100:.3f}%  Max: {np.max(all_vars)*100:.3f}%")

    notes.observe(f"\n  Per group:")
    for g, v in group_results.items():
        notes.observe(f"    {g:15s}: {v*100:.3f}%")

    # This is the key number: what fraction of embedding energy lives in truth axes?
    notes.finding(f"{n_anchors} truth axes explain {np.mean(all_vars)*100:.3f}% of embedding variance. "
                  f"The remaining {(1-np.mean(all_vars))*100:.3f}% is in the residual. "
                  f"If concepts are purely platonic compounds, this residual should be "
                  f"unstructured noise or just 'more platonic ideals we haven't found yet'.")

    return all_results, Q, anchor_names


# =============================================================================
# PART 2: RESIDUAL STRUCTURE ANALYSIS
# =============================================================================

def part2_residual_structure(all_results, Q, anchor_names, concepts, searcher, notes):
    """Is the residual structured or noise?"""

    notes.section("2. Residual Structure — Is What's Left Structured or Noise?")

    # Collect all residual vectors
    residuals = np.array([r["residual"] for r in all_results])
    embeddings_orig = np.array([r["embedding"] for r in all_results])
    words = [r["word"] for r in all_results]
    n_concepts = len(words)

    notes.observe(f"Residual matrix: {residuals.shape}")

    # -------------------------------------------------------
    # TEST 2a: Residual SVD — effective dimensionality
    # -------------------------------------------------------
    notes.observe(f"\n### 2a. Residual SVD — Effective Dimensionality")

    # Center residuals
    residual_mean = np.mean(residuals, axis=0)
    residuals_centered = residuals - residual_mean

    U, S, Vt = np.linalg.svd(residuals_centered, full_matrices=False)
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / total_var

    # Effective rank at various thresholds
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        rank = int(np.searchsorted(cumvar, thresh)) + 1
        notes.observe(f"  {thresh*100:.0f}% cumulative variance in {rank} dimensions")

    notes.observe(f"\n  Top 20 singular values:")
    for i in range(min(20, len(S))):
        notes.observe(f"    S[{i:2d}] = {S[i]:.4f}  cumvar = {cumvar[i]*100:.2f}%")

    # Compare to random baseline
    random_residuals = np.random.randn(*residuals.shape) * np.std(residuals)
    _, S_rand, _ = np.linalg.svd(random_residuals - np.mean(random_residuals, axis=0),
                                  full_matrices=False)
    cumvar_rand = np.cumsum(S_rand**2) / np.sum(S_rand**2)
    for thresh in [0.5, 0.8, 0.9]:
        rank_rand = int(np.searchsorted(cumvar_rand, thresh)) + 1
        rank_real = int(np.searchsorted(cumvar, thresh)) + 1
        notes.observe(f"  {thresh*100:.0f}% variance: real={rank_real} dims, random={rank_rand} dims")

    if int(np.searchsorted(cumvar, 0.9)) + 1 < int(np.searchsorted(cumvar_rand, 0.9)) + 1:
        notes.finding("Residuals are LOWER-DIMENSIONAL than random noise — they are STRUCTURED. "
                      "The residual space is not noise; it contains geometric information "
                      "that truth axes don't capture.")
    else:
        notes.finding("Residuals have similar dimensionality to random noise — they may be "
                      "unstructured. This would support concepts being purely platonic compounds.")

    # -------------------------------------------------------
    # TEST 2b: Nearest-neighbor retrieval from residuals alone
    # -------------------------------------------------------
    notes.observe(f"\n### 2b. Nearest-Neighbor Test — Can Residuals Identify Concepts?")

    # Can we find semantically similar concepts from residuals alone?
    # Cosine similarity matrix over residuals
    norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    resid_normed = residuals / (norms + 1e-20)
    sim_matrix = resid_normed @ resid_normed.T

    # For comparison: cosine similarity over original embeddings
    orig_norms = np.linalg.norm(embeddings_orig, axis=1, keepdims=True)
    orig_normed = embeddings_orig / (orig_norms + 1e-20)
    orig_sim_matrix = orig_normed @ orig_normed.T

    # For each concept, find k-nearest neighbors in residual space vs original space
    k = 5
    notes.observe(f"\nTop-{k} nearest neighbors: residual space vs original space")
    notes.observe(f"{'Concept':15s} | {'Residual NN':50s} | {'Original NN':50s}")
    notes.observe("-" * 120)

    # Sample a diverse set for display
    display_words = ["France", "Paris", "French", "king", "queen",
                     "Japan", "Tokyo", "Japanese", "man", "woman",
                     "Germany", "Berlin", "German", "boy", "girl"]

    for word in display_words:
        if word not in words:
            continue
        idx = words.index(word)

        # Residual neighbors
        resid_sims = sim_matrix[idx].copy()
        resid_sims[idx] = -999
        resid_nn_idx = np.argsort(resid_sims)[-k:][::-1]
        resid_nn = [(words[j], f"{resid_sims[j]:.3f}") for j in resid_nn_idx]

        # Original neighbors
        orig_sims = orig_sim_matrix[idx].copy()
        orig_sims[idx] = -999
        orig_nn_idx = np.argsort(orig_sims)[-k:][::-1]
        orig_nn = [(words[j], f"{orig_sims[j]:.3f}") for j in orig_nn_idx]

        resid_str = ", ".join(f"{w}({s})" for w, s in resid_nn)
        orig_str = ", ".join(f"{w}({s})" for w, s in orig_nn)
        notes.observe(f"{word:15s} | {resid_str:50s} | {orig_str:50s}")

    # -------------------------------------------------------
    # TEST 2c: Category clustering in residual space
    # -------------------------------------------------------
    notes.observe(f"\n### 2c. Category Clustering — Do Semantic Groups Cluster in Residual Space?")

    # Assign categories
    categories = {}
    cat_lists = {
        "country": ["France", "Germany", "Japan", "China", "Egypt", "Australia",
                     "India", "Brazil", "Korea", "Italy", "Spain", "Russia",
                     "Poland", "Norway", "Sweden", "Turkey", "Greece", "Ireland",
                     "Finland", "Denmark", "Mexico", "Canada", "Argentina",
                     "Nigeria", "Kenya"],
        "capital": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo", "Canberra",
                    "Delhi", "Seoul", "Rome", "Lisbon", "Moscow", "Madrid",
                    "Athens", "Ankara", "Dublin", "Helsinki", "Copenhagen",
                    "Vienna", "Warsaw", "Oslo", "Stockholm", "Ottawa", "Lima"],
        "language": ["French", "German", "Japanese", "Chinese", "Spanish",
                     "Italian", "Portuguese", "Russian", "Arabic", "English",
                     "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                     "Dutch", "Greek", "Turkish", "Hindi", "Finnish"],
        "male": ["king", "man", "boy", "father", "brother", "son",
                 "husband", "uncle", "prince", "actor"],
        "female": ["queen", "woman", "girl", "mother", "sister",
                   "daughter", "wife", "aunt", "princess", "actress"],
    }
    for cat, word_list in cat_lists.items():
        for w in word_list:
            categories[w] = cat

    # Within-category vs between-category cosine similarity
    for space_name, sim_mat in [("Residual", sim_matrix), ("Original", orig_sim_matrix)]:
        within_sims = []
        between_sims = []

        for i in range(n_concepts):
            for j in range(i+1, n_concepts):
                w_i, w_j = words[i], words[j]
                if w_i not in categories or w_j not in categories:
                    continue
                if categories[w_i] == categories[w_j]:
                    within_sims.append(sim_mat[i, j])
                else:
                    between_sims.append(sim_mat[i, j])

        if within_sims and between_sims:
            notes.observe(f"\n  **{space_name} space:**")
            notes.observe(f"    Within-category mean cos: {np.mean(within_sims):.4f} "
                         f"(std {np.std(within_sims):.4f})")
            notes.observe(f"    Between-category mean cos: {np.mean(between_sims):.4f} "
                         f"(std {np.std(between_sims):.4f})")
            separation = np.mean(within_sims) - np.mean(between_sims)
            notes.observe(f"    Separation (within - between): {separation:.4f}")

    # -------------------------------------------------------
    # TEST 2d: Residual prediction via relationship deltas
    # -------------------------------------------------------
    notes.observe(f"\n### 2d. Residual Relationship Coherence")
    notes.observe("If France_residual + capital_residual_delta ≈ Paris_residual,")
    notes.observe("then the residual encodes relational structure beyond truth axes.\n")

    # Compute residual deltas for relationships
    word_to_residual = {r["word"]: r["residual"] for r in all_results}

    for rel_name, pairs in [
        ("capital", [("France", "Paris"), ("Germany", "Berlin"),
                     ("Japan", "Tokyo"), ("China", "Beijing"), ("Egypt", "Cairo")]),
        ("gender", [("king", "queen"), ("man", "woman"),
                    ("boy", "girl"), ("father", "mother")]),
        ("language", [("France", "French"), ("Germany", "German"),
                      ("Japan", "Japanese"), ("China", "Chinese"), ("Spain", "Spanish")]),
    ]:
        valid = [(a, b) for a, b in pairs if a in word_to_residual and b in word_to_residual]
        if len(valid) < 2:
            continue

        # Compute mean residual delta (train on first N-1, test on last)
        train = valid[:-1]
        test_pair = valid[-1]

        resid_deltas = np.array([word_to_residual[b] - word_to_residual[a]
                                 for a, b in train])
        mean_resid_delta = np.mean(resid_deltas, axis=0)

        # Test: predict test target's residual from test source's residual + mean delta
        src_resid = word_to_residual[test_pair[0]]
        predicted_resid = src_resid + mean_resid_delta
        actual_resid = word_to_residual[test_pair[1]]

        cos_pred = (np.dot(predicted_resid, actual_resid) /
                    (np.linalg.norm(predicted_resid) * np.linalg.norm(actual_resid) + 1e-20))

        # Baseline: random residual similarity
        random_cos = np.mean([np.dot(actual_resid, word_to_residual[w]) /
                              (np.linalg.norm(actual_resid) * np.linalg.norm(word_to_residual[w]) + 1e-20)
                              for w in word_to_residual if w != test_pair[1]][:20])

        notes.observe(f"  {rel_name}: {test_pair[0]} + delta → {test_pair[1]}")
        notes.observe(f"    Predicted↔Actual residual cos: {cos_pred:.4f}")
        notes.observe(f"    Random baseline cos: {random_cos:.4f}")

        # Also test: how consistent are the residual deltas across pairs?
        if len(resid_deltas) >= 2:
            delta_norms = [d / (np.linalg.norm(d) + 1e-20) for d in resid_deltas]
            pair_cos = []
            for i in range(len(delta_norms)):
                for j in range(i+1, len(delta_norms)):
                    pair_cos.append(float(np.dot(delta_norms[i], delta_norms[j])))
            notes.observe(f"    Residual delta consistency (mean pairwise cos): {np.mean(pair_cos):.4f}")

    return residuals, words


# =============================================================================
# PART 3: SCALE ANALYSIS — HOW MANY AXES WOULD WE NEED?
# =============================================================================

def part3_scale_analysis(all_results, Q, anchor_names, concepts, searcher,
                         anchor_directions, notes):
    """Estimate how many truth axes would be needed to fully explain concepts."""

    notes.section("3. Scale Analysis — How Many Platonic Ideals Exist?")

    n_anchors = len(anchor_names)

    # Current state
    all_vars = [r["var_explained"] for r in all_results]
    notes.observe(f"Current: {n_anchors} axes → {np.mean(all_vars)*100:.3f}% variance explained")

    # Key insight: if we need K orthogonal axes to explain P% of variance,
    # then the "platonic ideal count" is approximately K.
    # But our 6 hand-picked axes may not be the optimal 6.

    # Test: what if we find the OPTIMAL K directions via PCA of concept embeddings?
    embeddings_matrix = np.array([r["embedding"] for r in all_results])
    emb_mean = np.mean(embeddings_matrix, axis=0)
    emb_centered = embeddings_matrix - emb_mean

    U_pca, S_pca, Vt_pca = np.linalg.svd(emb_centered, full_matrices=False)
    total_var_pca = np.sum(S_pca**2)
    cumvar_pca = np.cumsum(S_pca**2) / total_var_pca

    notes.observe(f"\n### PCA of Concept Embeddings (optimal K directions)")
    for k in [1, 2, 3, 6, 10, 15, 20, 30, 50, 80]:
        if k <= len(cumvar_pca):
            notes.observe(f"  {k:3d} PCA dims → {cumvar_pca[k-1]*100:.2f}% variance explained")

    # How many PCA dims to match our 6 truth axes?
    our_var = np.mean(all_vars)
    match_k = int(np.searchsorted(cumvar_pca, our_var)) + 1
    notes.observe(f"\n  Our {n_anchors} truth axes explain {our_var*100:.3f}% variance")
    notes.observe(f"  Equivalent to {match_k} PCA dimensions")

    # Compare truth axes to PCA axes: are truth axes aligned with top PCA components?
    notes.observe(f"\n### Truth Axes vs PCA Components")
    basis = np.array([anchor_directions[name]["direction"] for name in anchor_names])

    for i, aname in enumerate(anchor_names):
        # Project truth axis onto PCA space
        pca_coeffs = Vt_pca @ basis[i]
        top_pca = np.argsort(np.abs(pca_coeffs))[::-1][:5]
        notes.observe(f"  {aname}:")
        for pc_idx in top_pca:
            notes.observe(f"    PC{pc_idx}: coeff={pca_coeffs[pc_idx]:.4f} "
                         f"(cumvar={cumvar_pca[pc_idx]*100:.1f}%)")

    # Key question: what would happen with MORE anchors?
    # We can simulate by using PCA components as "discovered truth axes"
    notes.observe(f"\n### Simulated Scaling: Variance Explained vs Number of Axes")
    notes.observe("Using PCA as a proxy for 'optimal truth axis discovery':\n")

    for n_axes in [6, 10, 17, 30, 50, 88]:
        if n_axes <= len(cumvar_pca):
            notes.observe(f"  {n_axes:3d} axes → {cumvar_pca[n_axes-1]*100:.2f}% variance")

    notes.finding(f"PCA analysis reveals the intrinsic dimensionality of the concept space. "
                  f"If ~K PCA dims explain >99% of variance, then ~K platonic ideals "
                  f"would suffice to fully describe all concepts. The gap between our "
                  f"{n_anchors} truth axes and optimal PCA tells us how many more "
                  f"platonic ideals remain to be discovered.")

    return cumvar_pca, Vt_pca


# =============================================================================
# PART 4: THE CRITICAL TEST — RESIDUAL NEAREST-NEIGHBOR RETRIEVAL
# =============================================================================

def part4_retrieval_test(all_results, concepts, searcher, notes):
    """The definitive test: can you identify a concept from its residual alone?

    If concepts are PURELY platonic compounds, the residual should be noise,
    and nearest-neighbor retrieval from residuals should be at chance level.

    If the residual is structured, retrieval should work — meaning there IS
    content beyond the platonic decomposition."""

    notes.section("4. The Definitive Test — Concept Identity from Residuals Alone")

    residuals = np.array([r["residual"] for r in all_results])
    embeddings = np.array([r["embedding"] for r in all_results])
    words = [r["word"] for r in all_results]
    n = len(words)

    # Full-vocabulary retrieval from residuals
    # For each concept, project its residual against all vocab embeddings
    # If the residual is noise, the best-matching vocab token should be random
    # If structured, it should match semantically related tokens

    notes.observe("For each concept, we find the closest vocab token to its RESIDUAL vector.")
    notes.observe("If residuals are noise, the matches should be random.")
    notes.observe("If structured, matches should be semantically meaningful.\n")

    for word in ["France", "Paris", "French", "Japan", "Tokyo", "Japanese",
                 "king", "queen", "man", "woman", "boy", "girl",
                 "Germany", "Berlin", "German", "India", "Delhi"]:
        if word not in [r["word"] for r in all_results]:
            continue
        idx = [r["word"] for r in all_results].index(word)
        resid = all_results[idx]["residual"]

        top_matches = searcher.top_k(resid, k=10,
                                      exclude_tids=[concepts[word].token_id])
        clean = lambda t: t.replace("Ġ", "").replace("▁", "").strip()
        match_str = ", ".join(f"{clean(tok)}({sim:.3f})" for tok, sim in top_matches[:5])
        notes.observe(f"  {word:12s} residual → {match_str}")

    # Quantitative: self-retrieval accuracy
    # Use residual to find the concept's own embedding in the full vocabulary
    notes.observe(f"\n### Self-Retrieval from Residuals")
    notes.observe("Can the residual alone find the original concept in the full vocabulary?")

    ranks = []
    for r in all_results:
        rank = searcher.rank_of(r["residual"], concepts[r["word"]].token_id)
        ranks.append(rank)

    notes.observe(f"\n  Mean rank: {np.mean(ranks):.1f} (out of ~150K vocab tokens)")
    notes.observe(f"  Median rank: {np.median(ranks):.1f}")
    notes.observe(f"  In top-10: {sum(1 for r in ranks if r < 10)}/{len(ranks)}")
    notes.observe(f"  In top-100: {sum(1 for r in ranks if r < 100)}/{len(ranks)}")
    notes.observe(f"  In top-1000: {sum(1 for r in ranks if r < 1000)}/{len(ranks)}")

    # Show worst cases
    ranked_results = sorted(zip(ranks, [r["word"] for r in all_results]))
    notes.observe(f"\n  Best retrievals (lowest rank = most identifiable from residual):")
    for rank, word in ranked_results[:10]:
        notes.observe(f"    {word:15s} rank={rank}")
    notes.observe(f"\n  Worst retrievals (highest rank = least identifiable):")
    for rank, word in ranked_results[-10:]:
        notes.observe(f"    {word:15s} rank={rank}")

    if np.median(ranks) < 1000:
        notes.finding(f"Residuals can retrieve concepts at median rank {np.median(ranks):.0f} "
                      f"out of ~150K tokens. This means the residual is HIGHLY STRUCTURED — "
                      f"it encodes concept identity far beyond what 6 truth axes capture. "
                      f"The residual is NOT noise. It is 'more platonic ideals we haven't "
                      f"named yet'.")
    else:
        notes.finding(f"Residuals retrieve concepts at median rank {np.median(ranks):.0f}. "
                      f"The residual contains limited concept-specific information — "
                      f"most identity is captured by the truth axes.")

    return ranks


# =============================================================================
# PART 5: SYNTHESIS
# =============================================================================

def part5_synthesis(all_results, ranks, cumvar_pca, notes):
    """Pull it all together."""

    notes.section("5. Synthesis — Are Concepts Purely Platonic Compounds?")

    all_vars = [r["var_explained"] for r in all_results]
    mean_var = np.mean(all_vars)
    median_rank = np.median(ranks)

    notes.observe(f"### Key Numbers")
    notes.observe(f"  Truth axes (6): explain {mean_var*100:.3f}% of embedding variance")
    notes.observe(f"  Residual retrieval: median rank {median_rank:.0f} / ~150K")

    # Find how many PCA dims for 99%
    dims_99 = int(np.searchsorted(cumvar_pca, 0.99)) + 1
    dims_95 = int(np.searchsorted(cumvar_pca, 0.95)) + 1
    dims_90 = int(np.searchsorted(cumvar_pca, 0.90)) + 1

    notes.observe(f"  PCA dims for 90% variance: {dims_90}")
    notes.observe(f"  PCA dims for 95% variance: {dims_95}")
    notes.observe(f"  PCA dims for 99% variance: {dims_99}")

    notes.observe(f"\n### Interpretation")

    if median_rank < 100:
        notes.observe("The residual is EXTREMELY structured. After removing the projection")
        notes.observe("onto 6 truth axes, what remains can still identify the concept")
        notes.observe(f"at median rank {median_rank:.0f} in a 150K vocabulary.")
        notes.observe("")
        notes.observe("This means one of two things:")
        notes.observe("  (a) There are MANY more platonic ideals (truth axes) that we")
        notes.observe("      haven't discovered yet, and the residual is just the")
        notes.observe("      compound of those undiscovered ideals. OR")
        notes.observe("  (b) There is NON-PLATONIC structure — something about concepts")
        notes.observe("      that cannot be decomposed into binary truth axes.")
        notes.observe("")
        notes.observe(f"The PCA analysis suggests ~{dims_95} dimensions capture 95% of")
        notes.observe(f"concept variance. If each dimension corresponds to a platonic ideal,")
        notes.observe(f"then ~{dims_95} ideals would nearly fully describe all concepts.")
        notes.observe(f"This is still a finite, tractable number.")
    else:
        notes.observe("The residual is weakly structured. Most concept identity lives")
        notes.observe("in the truth axes. The platonic compound hypothesis is supported.")

    notes.observe(f"\n### The Answer")
    notes.observe("")
    if mean_var < 0.05 and median_rank < 100:
        notes.observe("**6 truth axes explain very little variance, yet residuals are structured.**")
        notes.observe("This strongly suggests MANY more platonic ideals exist. The concepts")
        notes.observe("ARE compounds — we've just only found 6 of the ~{} ideals needed.".format(dims_95))
        notes.observe("The 'non-platonic residual' is actually just undiscovered platonic structure.")
    elif mean_var > 0.5:
        notes.observe("**Truth axes explain >50% of variance.** The platonic decomposition is")
        notes.observe("working well even with just 6 axes. More axes would explain more.")
    else:
        notes.observe(f"**6 truth axes explain {mean_var*100:.1f}% of variance.** The remaining")
        notes.observe(f"{(1-mean_var)*100:.1f}% is structured (residual retrieval works).")
        notes.observe(f"This is consistent with concepts being compounds of ~{dims_95} platonic ideals,")
        notes.observe(f"of which we've identified 6. The residual is 'undiscovered platonic structure',")
        notes.observe(f"not non-platonic noise.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  TruthSpace Platonic Residual Test")
    print("  Are concepts purely compounds of platonic ideals?")
    print("=" * 70)
    print()

    notes = FieldNotes(NOTES_PATH)
    t_start = time.time()

    # Load everything
    concepts, searcher, embeddings, token_to_id, id_to_token = load_embeddings_and_concepts()
    relationships = define_relationships(concepts)
    anchor_directions = compute_anchors(concepts, searcher)

    notes.observe(f"Loaded {len(concepts)} concepts, {searcher.n_vocab} vocab tokens")
    notes.observe(f"Computed {len(anchor_directions)} anchor directions (truth axes)")
    for name, data in anchor_directions.items():
        notes.observe(f"  {name}: {len(data['pos_tids'])}+ / {len(data['neg_tids'])}-")

    # Part 1: Reconstruction
    all_results, Q, anchor_names = part1_reconstruction(
        concepts, searcher, anchor_directions, notes)

    # Part 2: Residual structure
    residuals, words = part2_residual_structure(
        all_results, Q, anchor_names, concepts, searcher, notes)

    # Part 3: Scale analysis
    cumvar_pca, Vt_pca = part3_scale_analysis(
        all_results, Q, anchor_names, concepts, searcher, anchor_directions, notes)

    # Part 4: Retrieval test
    ranks = part4_retrieval_test(all_results, concepts, searcher, notes)

    # Part 5: Synthesis
    part5_synthesis(all_results, ranks, cumvar_pca, notes)

    elapsed = time.time() - t_start
    notes.observe(f"\n\nCompleted in {elapsed:.1f}s")
    notes.close()
    print(f"\nField notes written to {NOTES_PATH}")


if __name__ == "__main__":
    main()
