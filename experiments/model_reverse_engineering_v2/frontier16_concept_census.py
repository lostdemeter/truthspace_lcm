#!/usr/bin/env python3
"""
Frontier 16: Concept Census — How Many Concepts Does the Model Know?
=====================================================================

Four-part experiment on the embedding matrix (152064 × 3584):

Part 1: SVD Energy Profile
    What is the effective rank of the embedding space?
    How many dimensions carry semantic signal vs noise?

Part 2: K-Means Concept Clustering
    How many natural clusters exist in embedding space?
    Sweep k from 100 to 20000, track inertia elbow.

Part 3: Reconstruction Test
    Compress embeddings via (cluster_center in M-dim SVD subspace).
    Compare lm_head predictions on original vs reconstructed embeddings.
    Does compression preserve the model's output behavior?

Part 4: Concept Labeling
    For the best-performing configuration, show representative tokens
    per cluster to see what each "concept" is.

DC 290, F160
"""

import numpy as np
import os
import sys
import json
import gc
import time

PHI = (1 + np.sqrt(5)) / 2
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')
GRID = 128


def decode_phi(path, dtype=np.float64):
    """Decode φ-encoded weight matrix."""
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def load_tokenizer():
    """Load tokenizer vocabulary."""
    for candidate in [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
    ]:
        if os.path.exists(candidate):
            snapshots = os.listdir(candidate)
            if snapshots:
                vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                if os.path.exists(vocab_file):
                    with open(vocab_file, 'r') as f:
                        tokenizer_data = json.load(f)
                    vocab = tokenizer_data.get('model', {}).get('vocab', {})
                    id_to_token = {idx: tok for tok, idx in vocab.items()}
                    return id_to_token
    return None


# ─── Part 1: SVD Energy Profile ──────────────────────────────────────

def part1_svd_profile(embeddings):
    """
    SVD of the embedding matrix to find effective dimensionality.
    Uses covariance approach: E^T @ E is (3584 × 3584).
    """
    print()
    print("=" * 80)
    print("  Part 1: SVD Energy Profile")
    print("  How many dimensions carry semantic signal?")
    print("=" * 80)
    print()

    n_tokens, n_dims = embeddings.shape
    print(f"  Embedding matrix: {n_tokens} tokens × {n_dims} dimensions")

    # Center the embeddings (mean-subtract) for PCA
    print("  Computing mean embedding...")
    mean_emb = np.mean(embeddings, axis=0)
    print(f"  ||mean||={np.linalg.norm(mean_emb):.4f}")

    # Covariance matrix (3584 × 3584) — feasible
    print("  Computing covariance matrix (3584 × 3584)...")
    t0 = time.time()
    # Use float32 for the matmul to save memory, cast to float64 for eigh
    emb_centered = (embeddings - mean_emb).astype(np.float32)
    cov = emb_centered.T @ emb_centered  # (3584, 3584) float32
    cov = cov.astype(np.float64) / n_tokens
    del emb_centered
    gc.collect()
    print(f"  Covariance computed in {time.time()-t0:.1f}s")

    print("  Eigendecomposition...")
    t0 = time.time()
    eigenvalues, V = np.linalg.eigh(cov)
    print(f"  Eigendecomposition in {time.time()-t0:.1f}s")
    del cov
    gc.collect()

    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    V = V[:, idx_sort]

    # Clip any negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = np.sum(eigenvalues)

    # Cumulative variance
    cumvar = np.cumsum(eigenvalues) / total_var * 100

    print(f"\n  Total variance: {total_var:.4f}")
    print(f"  Top eigenvalue: {eigenvalues[0]:.4f} ({eigenvalues[0]/total_var*100:.2f}%)")
    print(f"  Bottom eigenvalue: {eigenvalues[-1]:.6f}")

    # Energy thresholds
    thresholds = [50, 75, 80, 85, 90, 95, 99, 99.5, 99.9]
    print(f"\n  {'Variance %':>12s}  {'Dims needed':>12s}  {'% of 3584':>10s}")
    print("  " + "-" * 40)

    effective_dims = {}
    for thresh in thresholds:
        k = int(np.searchsorted(cumvar, thresh)) + 1
        k = min(k, n_dims)
        effective_dims[thresh] = k
        print(f"  {thresh:11.1f}%  {k:12d}  {k/n_dims*100:9.1f}%")

    # Zipf analysis: how do eigenvalues decay?
    print(f"\n  Eigenvalue decay (Zipf analysis):")
    checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3584]
    print(f"  {'Rank':>6s}  {'Eigenvalue':>12s}  {'Ratio to #1':>12s}  {'Cumul %':>8s}")
    print("  " + "-" * 45)
    for k in checkpoints:
        if k <= n_dims:
            ev = eigenvalues[k-1]
            ratio = ev / eigenvalues[0] if eigenvalues[0] > 0 else 0
            cv = cumvar[k-1]
            print(f"  {k:6d}  {ev:12.4f}  {ratio:12.6f}  {cv:7.2f}%")

    print()
    sys.stdout.flush()

    return eigenvalues, V, mean_emb, effective_dims


# ─── Part 2: K-Means Concept Clustering ──────────────────────────────

def part2_clustering(embeddings, V, mean_emb, effective_dims):
    """
    K-means clustering on SVD-projected embeddings.
    Uses sklearn MiniBatchKMeans for speed.
    """
    from sklearn.cluster import MiniBatchKMeans
    print()
    print("=" * 80)
    print("  Part 2: K-Means Concept Clustering")
    print("  How many natural concept clusters exist?")
    print("=" * 80)
    print()

    # Project into the 90% variance subspace for clustering
    m_dim = effective_dims[90]
    print(f"  Projecting into top {m_dim} SVD dimensions (90% variance)")

    # Cast to float32 BEFORE matmul to halve memory and speed up
    V_sub = V[:, :m_dim].astype(np.float32)
    emb_f32 = embeddings.astype(np.float32) - mean_emb.astype(np.float32)
    emb_projected = emb_f32 @ V_sub  # (152064, m_dim) float32
    del emb_f32, V_sub
    gc.collect()
    print(f"  Projected shape: {emb_projected.shape}")

    # Sweep k values — use fewer large k values to keep runtime reasonable
    k_values = [100, 500, 1000, 2000, 5000, 10000]
    results = []

    print(f"\n  {'k':>7s}  {'Inertia':>14s}  {'Inertia/k':>12s}  {'Time':>7s}")
    print("  " + "-" * 50)
    sys.stdout.flush()

    best_labels = None
    best_centers = None
    best_k = None

    for k in k_values:
        t0 = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(10000, max(k * 10, 1000)),
            max_iter=100,
            random_state=42,
            n_init=1,
        )
        kmeans.fit(emb_projected)
        elapsed = time.time() - t0

        inertia = kmeans.inertia_
        inertia_per_k = inertia / k
        results.append((k, inertia, inertia_per_k))

        print(f"  {k:7d}  {inertia:14.1f}  {inertia_per_k:12.1f}  {elapsed:6.1f}s")
        sys.stdout.flush()

        # Save k=5000 as default for reconstruction test
        if k == 5000:
            best_labels = kmeans.labels_.copy()
            best_centers = kmeans.cluster_centers_.copy()
            best_k = k

    # Find elbow: where does adding more clusters stop helping much?
    print(f"\n  Inertia reduction ratios:")
    for i in range(1, len(results)):
        k_prev, inertia_prev, _ = results[i-1]
        k_curr, inertia_curr, _ = results[i]
        reduction = (inertia_prev - inertia_curr) / inertia_prev * 100
        per_new_cluster = (inertia_prev - inertia_curr) / (k_curr - k_prev)
        print(f"    k={k_prev}→{k_curr}: {reduction:.1f}% reduction, "
              f"{per_new_cluster:.1f} inertia/new_cluster")

    print()
    sys.stdout.flush()

    return emb_projected, best_labels, best_centers, best_k, V, results


# ─── Part 3: Reconstruction Test ─────────────────────────────────────

def part3_reconstruction(embeddings, emb_projected, labels, centers,
                         k, V, mean_emb, effective_dims, id_to_token):
    """
    Replace each embedding with its cluster center, reconstruct to full
    space, and compare lm_head predictions.
    
    Test multiple configurations:
    - Cluster-only (no residual)
    - Cluster + low-rank residual within cluster
    - Various SVD dimension counts
    """
    print()
    print("=" * 80)
    print("  Part 3: Reconstruction Test")
    print("  Does compression preserve model predictions?")
    print("=" * 80)
    print()

    n_tokens = embeddings.shape[0]
    m_dim = effective_dims[90]

    # Load lm_head
    print("  Loading lm_head (float32)...")
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'), dtype=np.float32)
    print(f"  lm_head shape: {lm_head.shape}")

    # Compute original logits for a sample of tokens
    # Use 1000 evenly-spaced tokens for efficiency
    sample_size = 1000
    sample_indices = np.linspace(0, n_tokens - 1, sample_size, dtype=int)
    # Also add some known tokens
    known_words = ['dragon', 'the', 'is', 'computer', 'love', 'hello']
    if id_to_token:
        token_to_id = {}
        for idx, tok in id_to_token.items():
            token_to_id[tok] = idx
            token_to_id[tok.lower()] = idx
        for w in known_words:
            for variant in [w, w.capitalize(), f"Ġ{w}", f"Ġ{w.lower()}"]:
                if variant in token_to_id:
                    sample_indices = np.append(sample_indices, token_to_id[variant])
                    break

    sample_indices = np.unique(sample_indices)
    sample_size = len(sample_indices)
    print(f"  Testing {sample_size} sample tokens")

    # Original predictions
    print("  Computing original logits for sample tokens...")
    original_embs = embeddings[sample_indices].astype(np.float32)
    original_logits = original_embs @ lm_head.T  # (sample, 152064)
    original_top10 = np.argsort(original_logits, axis=1)[:, -10:]
    original_top1 = np.argmax(original_logits, axis=1)

    # Test configurations
    configs = []

    # Config A: Cluster centers only (projected back to full space)
    print("\n  --- Config A: Cluster centers only ---")
    V_f32 = V.astype(np.float32)
    mean_f32 = mean_emb.astype(np.float32)
    reconstructed_proj = centers[labels[sample_indices]]  # (sample, m_dim)
    reconstructed_full = (reconstructed_proj @ V_f32[:, :m_dim].T) + mean_f32
    recon_logits = reconstructed_full @ lm_head.T
    top1_match_a = np.mean(np.argmax(recon_logits, axis=1) == original_top1) * 100

    # Top-10 overlap
    recon_top10 = np.argsort(recon_logits, axis=1)[:, -10:]
    top10_overlaps = []
    for i in range(sample_size):
        overlap = len(set(recon_top10[i]) & set(original_top10[i]))
        top10_overlaps.append(overlap)
    top10_avg_a = np.mean(top10_overlaps)

    # Cosine similarity of logit vectors
    cos_sims = []
    for i in range(sample_size):
        o = original_logits[i]
        r = recon_logits[i]
        cos = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-10)
        cos_sims.append(cos)
    cos_avg_a = np.mean(cos_sims)

    print(f"    k={k}, dims={m_dim}")
    print(f"    Top-1 match:   {top1_match_a:.1f}%")
    print(f"    Top-10 overlap: {top10_avg_a:.2f}/10")
    print(f"    Logit cosine:   {cos_avg_a:.6f}")
    configs.append(('cluster-only', k, m_dim, top1_match_a, top10_avg_a, cos_avg_a))
    del reconstructed_proj, reconstructed_full, recon_logits
    gc.collect()

    # Config B: Vary SVD dimensions with cluster centers
    for m in [50, 100, 200, 500, 1000, 2000, 3584]:
        if m > V.shape[1]:
            continue
        print(f"\n  --- Config B: Cluster centers, {m} SVD dims ---")

        # Use existing labels (from 90%-dim clustering) but project centers
        # into the m-dim space for reconstruction
        if m >= m_dim:
            # Pad existing centers with zeros
            padded_centers = np.zeros((k, m), dtype=np.float32)
            padded_centers[:, :m_dim] = centers
            recon_proj = padded_centers[labels[sample_indices]]
        else:
            # Truncate existing centers
            recon_proj = centers[labels[sample_indices]][:, :m]

        recon_full = (recon_proj @ V_f32[:, :m].T) + mean_f32
        recon_logits = recon_full @ lm_head.T

        top1_match = np.mean(np.argmax(recon_logits, axis=1) == original_top1) * 100
        recon_top10 = np.argsort(recon_logits, axis=1)[:, -10:]
        top10_overlaps = []
        for i in range(sample_size):
            overlap = len(set(recon_top10[i]) & set(original_top10[i]))
            top10_overlaps.append(overlap)
        top10_avg = np.mean(top10_overlaps)

        cos_sims = []
        for i in range(sample_size):
            o = original_logits[i]
            r = recon_logits[i]
            cos = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-10)
            cos_sims.append(cos)
        cos_avg = np.mean(cos_sims)

        print(f"    k={k}, dims={m}")
        print(f"    Top-1 match:   {top1_match:.1f}%")
        print(f"    Top-10 overlap: {top10_avg:.2f}/10")
        print(f"    Logit cosine:   {cos_avg:.6f}")
        configs.append(('cluster+svd', k, m, top1_match, top10_avg, cos_avg))

        del recon_proj, recon_full, recon_logits
        gc.collect()

    # Config C: No clustering, just SVD truncation (for comparison)
    print(f"\n  --- Config C: SVD-only (no clustering) ---")
    for m in [50, 100, 200, 500, 1000, 2000]:
        emb_centered = embeddings[sample_indices].astype(np.float32) - mean_f32
        emb_m = emb_centered @ V_f32[:, :m]  # project down
        recon_full = (emb_m @ V_f32[:, :m].T) + mean_f32  # project up
        recon_logits = recon_full @ lm_head.T

        top1_match = np.mean(np.argmax(recon_logits, axis=1) == original_top1) * 100
        recon_top10 = np.argsort(recon_logits, axis=1)[:, -10:]
        top10_overlaps = []
        for i in range(sample_size):
            overlap = len(set(recon_top10[i]) & set(original_top10[i]))
            top10_overlaps.append(overlap)
        top10_avg = np.mean(top10_overlaps)

        cos_sims = []
        for i in range(sample_size):
            o = original_logits[i]
            r = recon_logits[i]
            cos = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-10)
            cos_sims.append(cos)
        cos_avg = np.mean(cos_sims)

        orig_size = 152064 * 3584
        compressed_size = 152064 * m + m * 3584 + 3584  # projections + basis + mean
        ratio = orig_size / compressed_size

        print(f"    dims={m}: top1={top1_match:.1f}%, top10={top10_avg:.2f}/10, "
              f"cos={cos_avg:.6f}, compression={ratio:.1f}×")
        configs.append(('svd-only', 0, m, top1_match, top10_avg, cos_avg))

        del emb_centered, emb_m, recon_full, recon_logits
        gc.collect()

    del lm_head
    gc.collect()

    # Summary table
    print(f"\n  {'Config':>18s}  {'k':>6s}  {'dims':>5s}  {'Top1%':>6s}  {'Top10':>6s}  {'Cosine':>8s}")
    print("  " + "-" * 60)
    for name, k_val, m_val, t1, t10, cos in configs:
        print(f"  {name:>18s}  {k_val:6d}  {m_val:5d}  {t1:5.1f}%  {t10:5.2f}  {cos:8.6f}")

    print()
    sys.stdout.flush()
    return configs


# ─── Part 4: Concept Labeling ────────────────────────────────────────

def part4_labeling(embeddings, emb_projected, labels, centers, k,
                   id_to_token, V, mean_emb):
    """
    For the k=5000 clustering, show representative tokens per cluster
    to understand what each concept is.
    """
    print()
    print("=" * 80)
    print("  Part 4: Concept Labeling")
    print("  What are the concept clusters?")
    print("=" * 80)
    print()

    if id_to_token is None:
        print("  ERROR: No tokenizer loaded, can't label clusters")
        return

    n_tokens = embeddings.shape[0]

    # Compute cluster sizes
    cluster_sizes = np.bincount(labels, minlength=k)
    size_order = np.argsort(cluster_sizes)[::-1]

    print(f"  Total clusters: {k}")
    print(f"  Cluster size stats:")
    print(f"    Mean: {np.mean(cluster_sizes):.1f} tokens/cluster")
    print(f"    Median: {np.median(cluster_sizes):.1f}")
    print(f"    Min: {np.min(cluster_sizes)}, Max: {np.max(cluster_sizes)}")
    print(f"    Singletons: {np.sum(cluster_sizes == 1)}")
    print(f"    Size > 100: {np.sum(cluster_sizes > 100)}")
    print(f"    Size > 1000: {np.sum(cluster_sizes > 1000)}")

    # Show top 50 largest clusters with representative tokens
    print(f"\n  Top 50 largest clusters (representative tokens):")
    print(f"  {'#':>4s}  {'Cluster':>8s}  {'Size':>6s}  Representative tokens")
    print("  " + "-" * 70)

    for rank_idx in range(min(50, k)):
        cluster_id = size_order[rank_idx]
        size = cluster_sizes[cluster_id]
        members = np.where(labels == cluster_id)[0]

        # Find tokens closest to cluster center
        member_vecs = emb_projected[members]
        center = centers[cluster_id]
        dists = np.linalg.norm(member_vecs - center, axis=1)
        closest_order = np.argsort(dists)

        # Get up to 8 representative tokens
        rep_tokens = []
        for j in closest_order[:8]:
            token_id = members[j]
            tok = id_to_token.get(token_id, f"tok_{token_id}")
            rep_tokens.append(tok)

        reps_str = ", ".join(f"'{t}'" for t in rep_tokens)
        print(f"  {rank_idx+1:4d}  {cluster_id:8d}  {size:6d}  {reps_str}")

    # Show 20 random mid-size clusters
    mid_mask = (cluster_sizes >= 10) & (cluster_sizes <= 50)
    mid_clusters = np.where(mid_mask)[0]
    if len(mid_clusters) > 0:
        print(f"\n  20 random mid-size clusters (10-50 tokens):")
        print(f"  {'Cluster':>8s}  {'Size':>6s}  Representative tokens")
        print("  " + "-" * 70)

        rng = np.random.RandomState(42)
        sample_mid = rng.choice(mid_clusters, size=min(20, len(mid_clusters)), replace=False)

        for cluster_id in sample_mid:
            size = cluster_sizes[cluster_id]
            members = np.where(labels == cluster_id)[0]
            member_vecs = emb_projected[members]
            center = centers[cluster_id]
            dists = np.linalg.norm(member_vecs - center, axis=1)
            closest_order = np.argsort(dists)

            rep_tokens = []
            for j in closest_order[:8]:
                token_id = members[j]
                tok = id_to_token.get(token_id, f"tok_{token_id}")
                rep_tokens.append(tok)

            reps_str = ", ".join(f"'{t}'" for t in rep_tokens)
            print(f"  {cluster_id:8d}  {size:6d}  {reps_str}")

    # Show some small clusters (potential "pure concepts")
    small_mask = (cluster_sizes >= 2) & (cluster_sizes <= 5)
    small_clusters = np.where(small_mask)[0]
    if len(small_clusters) > 0:
        print(f"\n  20 random small clusters (2-5 tokens) — potential pure concepts:")
        print(f"  {'Cluster':>8s}  {'Size':>6s}  All tokens")
        print("  " + "-" * 70)

        rng = np.random.RandomState(123)
        sample_small = rng.choice(small_clusters, size=min(20, len(small_clusters)), replace=False)

        for cluster_id in sample_small:
            size = cluster_sizes[cluster_id]
            members = np.where(labels == cluster_id)[0]
            all_tokens = []
            for token_id in members:
                tok = id_to_token.get(token_id, f"tok_{token_id}")
                all_tokens.append(tok)

            toks_str = ", ".join(f"'{t}'" for t in all_tokens[:10])
            print(f"  {cluster_id:8d}  {size:6d}  {toks_str}")

    print()
    sys.stdout.flush()


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 80)
    print("  Frontier 16: Concept Census")
    print("  How many concepts does Qwen2-7B know?")
    print("=" * 80)
    print()

    # Load tokenizer
    print("  Loading tokenizer...")
    id_to_token = load_tokenizer()
    if id_to_token:
        print(f"  Vocabulary: {len(id_to_token)} tokens")
    else:
        print("  WARNING: Could not load tokenizer (labeling will be skipped)")

    # Load embeddings
    print("  Loading embeddings...")
    t0 = time.time()
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"  Embeddings: {embeddings.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Part 1: SVD
    eigenvalues, V, mean_emb, effective_dims = part1_svd_profile(embeddings)

    # Part 2: Clustering
    emb_projected, labels, centers, best_k, V, cluster_results = \
        part2_clustering(embeddings, V, mean_emb, effective_dims)

    # Part 3: Reconstruction
    configs = part3_reconstruction(
        embeddings, emb_projected, labels, centers, best_k,
        V, mean_emb, effective_dims, id_to_token
    )

    # Part 4: Labeling
    part4_labeling(embeddings, emb_projected, labels, centers, best_k,
                   id_to_token, V, mean_emb)

    # Final summary
    print("=" * 80)
    print("  SUMMARY — Frontier 16: Concept Census")
    print("=" * 80)
    print()
    print(f"  Embedding matrix: {embeddings.shape[0]} tokens × {embeddings.shape[1]} dims")
    print(f"  Effective rank (90% var): {effective_dims[90]} dims")
    print(f"  Effective rank (95% var): {effective_dims[95]} dims")
    print(f"  Effective rank (99% var): {effective_dims[99]} dims")
    print()
    print(f"  Best reconstruction configs:")
    for name, k_val, m_val, t1, t10, cos in configs:
        if t1 > 50 or cos > 0.99:
            flag = " ★" if t1 > 90 else ""
            print(f"    {name} k={k_val} d={m_val}: "
                  f"top1={t1:.1f}%, top10={t10:.2f}/10, cos={cos:.6f}{flag}")
    print()

    # Compression analysis
    original_params = 152064 * 3584
    print(f"  Original embedding parameters: {original_params:,} ({original_params*8/1e9:.2f} GB)")
    for name, k_val, m_val, t1, t10, cos in configs:
        if t1 > 80:
            if name == 'svd-only':
                compressed = 152064 * m_val + m_val * 3584 + 3584
            elif name == 'cluster-only' or name == 'cluster+svd':
                compressed = k_val * m_val + 152064 + m_val * 3584 + 3584
                # k centers × m_dim + assignments + basis + mean
            else:
                continue
            ratio = original_params / compressed
            print(f"    {name} k={k_val} d={m_val}: {compressed:,} params "
                  f"({ratio:.1f}× compression) — top1={t1:.1f}%")

    print()


if __name__ == '__main__':
    main()
