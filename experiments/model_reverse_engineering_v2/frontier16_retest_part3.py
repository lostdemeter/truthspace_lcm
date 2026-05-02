#!/usr/bin/env python3
"""Quick retest of Part 3 reconstruction with fixed top-10 computation."""

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
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def load_tokenizer():
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


def evaluate_reconstruction(original_logits, original_top1, original_top10,
                            recon_logits, sample_size, label):
    """Compute top-1 match, top-10 overlap, cosine similarity."""
    top1_match = np.mean(np.argmax(recon_logits, axis=1) == original_top1) * 100

    recon_top10 = np.argsort(recon_logits, axis=1)[:, -10:]
    top10_overlaps = []
    for i in range(sample_size):
        overlap = len(set(recon_top10[i].tolist()) & set(original_top10[i].tolist()))
        top10_overlaps.append(overlap)
    top10_avg = np.mean(top10_overlaps)

    # Cosine similarity of logit vectors
    o_norms = np.linalg.norm(original_logits, axis=1, keepdims=True)
    r_norms = np.linalg.norm(recon_logits, axis=1, keepdims=True)
    cos_sims = np.sum(original_logits * recon_logits, axis=1) / (
        (o_norms.ravel() * r_norms.ravel()) + 1e-10)
    cos_avg = np.mean(cos_sims)

    print(f"    {label}")
    print(f"    Top-1 match:    {top1_match:.1f}%")
    print(f"    Top-10 overlap: {top10_avg:.2f}/10")
    print(f"    Logit cosine:   {cos_avg:.6f}")
    return top1_match, top10_avg, cos_avg


def main():
    print("\n  Frontier 16 Part 3 RETEST (fixed top-10)\n")

    id_to_token = load_tokenizer()

    # Load embeddings
    print("  Loading embeddings...")
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    n_tokens, n_dims = embeddings.shape
    print(f"  Shape: {embeddings.shape}")

    # SVD
    print("  Computing SVD (covariance approach)...")
    mean_emb = np.mean(embeddings, axis=0)
    emb_centered = (embeddings - mean_emb).astype(np.float32)
    cov = emb_centered.T @ emb_centered
    cov = cov.astype(np.float64) / n_tokens
    del emb_centered; gc.collect()

    eigenvalues, V = np.linalg.eigh(cov)
    del cov; gc.collect()

    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx_sort], 0)
    V = V[:, idx_sort]
    V_f32 = V.astype(np.float32)
    mean_f32 = mean_emb.astype(np.float32)

    # Load lm_head
    print("  Loading lm_head (float32)...")
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'), dtype=np.float32)

    # Sample tokens
    sample_size = 2000
    sample_indices = np.linspace(0, n_tokens - 1, sample_size, dtype=int)
    if id_to_token:
        token_to_id = {}
        for idx, tok in id_to_token.items():
            token_to_id[tok] = idx
            token_to_id[tok.lower()] = idx
        for w in ['dragon', 'the', 'is', 'computer', 'love', 'hello', 'king', 'queen']:
            for variant in [w, w.capitalize(), f"Ġ{w}", f"Ġ{w.lower()}"]:
                if variant in token_to_id:
                    sample_indices = np.append(sample_indices, token_to_id[variant])
                    break
    sample_indices = np.unique(sample_indices)
    sample_size = len(sample_indices)
    print(f"  Testing {sample_size} sample tokens")

    # Original predictions
    print("  Computing original logits...")
    original_embs = embeddings[sample_indices].astype(np.float32)
    original_logits = original_embs @ lm_head.T
    original_top10 = np.argsort(original_logits, axis=1)[:, -10:]
    original_top1 = np.argmax(original_logits, axis=1)

    configs = []

    # SVD-only at various dimensions
    print("\n  === SVD-only reconstruction ===")
    for m in [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3584]:
        emb_c = embeddings[sample_indices].astype(np.float32) - mean_f32
        emb_m = emb_c @ V_f32[:, :m]
        recon = (emb_m @ V_f32[:, :m].T) + mean_f32
        recon_logits = recon @ lm_head.T

        orig_size = 152064 * 3584
        comp_size = 152064 * m + m * 3584 + 3584
        ratio = orig_size / comp_size

        t1, t10, cos = evaluate_reconstruction(
            original_logits, original_top1, original_top10,
            recon_logits, sample_size, f"dims={m} ({ratio:.1f}× compression)")
        configs.append(('svd-only', 0, m, t1, t10, cos, ratio))

        del emb_c, emb_m, recon, recon_logits; gc.collect()

    # K-means at k=5000 in 500-dim SVD space (faster than 1435-dim)
    print("\n  === K-means + SVD reconstruction ===")
    from sklearn.cluster import MiniBatchKMeans

    for m_cluster in [200, 500]:
        print(f"\n  Clustering in {m_cluster}-dim SVD space...")
        emb_c = (embeddings - mean_emb).astype(np.float32)
        emb_proj = emb_c @ V_f32[:, :m_cluster]
        del emb_c; gc.collect()

        for k in [1000, 5000, 10000, 20000]:
            t0 = time.time()
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000,
                                     max_iter=100, random_state=42, n_init=1)
            kmeans.fit(emb_proj)
            elapsed = time.time() - t0
            print(f"    k={k} in {m_cluster}d: {elapsed:.1f}s")

            # Reconstruct sample tokens
            sample_proj = emb_proj[sample_indices]
            sample_labels = kmeans.predict(sample_proj)
            recon_proj = kmeans.cluster_centers_[sample_labels]
            recon = (recon_proj @ V_f32[:, :m_cluster].T) + mean_f32
            recon_logits = recon @ lm_head.T

            comp_size = k * m_cluster + 152064 * int(np.ceil(np.log2(k) / 8)) + m_cluster * 3584 + 3584
            ratio = orig_size / comp_size

            t1, t10, cos = evaluate_reconstruction(
                original_logits, original_top1, original_top10,
                recon_logits, sample_size,
                f"k={k}, dims={m_cluster} ({ratio:.1f}× compression)")
            configs.append(('cluster+svd', k, m_cluster, t1, t10, cos, ratio))

            del recon, recon_logits; gc.collect()
            sys.stdout.flush()

        del emb_proj; gc.collect()

    # Summary
    print(f"\n  {'Config':>15s}  {'k':>6s}  {'dims':>5s}  {'Top1%':>6s}  {'Top10':>6s}  {'Cosine':>8s}  {'Compr':>6s}")
    print("  " + "-" * 65)
    for name, k_val, m_val, t1, t10, cos, ratio in configs:
        flag = " ★" if t1 > 90 else ""
        print(f"  {name:>15s}  {k_val:6d}  {m_val:5d}  {t1:5.1f}%  {t10:5.2f}  {cos:8.6f}  {ratio:5.1f}×{flag}")

    print()


if __name__ == '__main__':
    main()
