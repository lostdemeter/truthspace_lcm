#!/usr/bin/env python3
"""
Quick MESH SVD check for Qwen2-1.5B-Instruct.
Compare rank structure to tensors_are_shapes 7B findings.

MESH_h = W_q_h.T @ W_k_h  (128×128 per head)
7B finding: S[0]/S[1] = 368,000:1 at L23 H6 (rank-1 Resonator)
1.5B claim:  effective rank 99-122, ratio <= 1.35:1 (full-rank)
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
PROBE_LAYERS = [1, 5, 10, 14, 20, 23, 27]

print(f"Loading {MODEL_ID} ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

cfg = model.config
n_layers   = cfg.num_hidden_layers
n_heads    = cfg.num_attention_heads
n_kv_heads = cfg.num_key_value_heads
head_dim   = cfg.hidden_size // n_heads
kv_per_q   = n_heads // n_kv_heads   # Q-heads per KV group

print(f"  n_layers={n_layers}  n_heads={n_heads}  n_kv_heads={n_kv_heads}")
print(f"  head_dim={head_dim}  Q-heads/KV-group={kv_per_q}")
print()

def mesh_svd(layer_idx, head_idx):
    attn = model.model.layers[layer_idx].self_attn
    Wq   = attn.q_proj.weight.detach().numpy()   # (n_heads*head_dim, hidden)
    Wk   = attn.k_proj.weight.detach().numpy()   # (n_kv_heads*head_dim, hidden)
    bq   = attn.q_proj.bias.detach().numpy() if attn.q_proj.bias is not None else None
    bk   = attn.k_proj.bias.detach().numpy() if attn.k_proj.bias is not None else None

    kv_group = head_idx // kv_per_q
    Wq_h = Wq[head_idx*head_dim:(head_idx+1)*head_dim, :].T    # (hidden, head_dim)
    Wk_h = Wk[kv_group*head_dim:(kv_group+1)*head_dim, :].T    # (hidden, head_dim)

    MESH_ww = Wq_h.T @ Wk_h       # (head_dim, head_dim) — weights only

    # Full MESH including bias outer product
    if bq is not None and bk is not None:
        bq_h = bq[head_idx*head_dim:(head_idx+1)*head_dim]
        bk_h = bk[kv_group*head_dim:(kv_group+1)*head_dim]
        MESH_bb = np.outer(bq_h, bk_h) * head_dim  # D × (bq ⊗ bk)
        MESH_full = MESH_ww + MESH_bb
    else:
        MESH_bb   = None
        MESH_full = MESH_ww

    # SVD
    _, S_w, _    = np.linalg.svd(MESH_ww, full_matrices=False)
    _, S_full, _ = np.linalg.svd(MESH_full, full_matrices=False)

    ratio_w    = S_w[0]/S_w[1]    if len(S_w)>1 and S_w[1]>1e-10 else float('inf')
    ratio_full = S_full[0]/S_full[1] if len(S_full)>1 and S_full[1]>1e-10 else float('inf')

    # Effective rank (99% variance)
    cum_w    = np.cumsum(S_w**2)    / (S_w**2).sum()
    cum_full = np.cumsum(S_full**2) / (S_full**2).sum()
    eff_w    = int(np.searchsorted(cum_w,    0.99)) + 1
    eff_full = int(np.searchsorted(cum_full, 0.99)) + 1

    # Bias fraction of full MESH
    if MESH_bb is not None:
        frac_bb = float(np.linalg.norm(MESH_bb)**2 /
                        (np.linalg.norm(MESH_ww)**2 + np.linalg.norm(MESH_bb)**2))
    else:
        frac_bb = 0.0

    return ratio_w, ratio_full, eff_w, eff_full, frac_bb, S_full

print("="*72)
print("All heads at L23 — the key routing layer")
print("="*72)
print(f"  {'Head':<6} {'S0/S1 (w only)':<18} {'S0/S1 (full)':<16} {'eff_rank_full':<15} bias_frac")
print(f"  {'-'*65}")
for h in range(n_heads):
    rw, rf, ew, ef, fb, Sf = mesh_svd(23, h)
    kv_grp = h // kv_per_q
    marker = " ←" if rf > 100 else ""
    print(f"  H{h:02d}(KV{kv_grp})  {rw:>12.1f}          {rf:>10.3f}       {ef:>5d}         {fb:.4f}{marker}")

print()
print("="*72)
print("Max S0/S1 across all heads, by layer")
print("="*72)
print(f"  {'Layer':<8} {'max S0/S1':<14} {'mean S0/S1':<14} {'best head':<10} {'mean eff_rank'}")
print(f"  {'-'*55}")
for L in PROBE_LAYERS:
    ratios, eff_ranks = [], []
    for h in range(n_heads):
        _, rf, _, ef, _, _ = mesh_svd(L, h)
        ratios.append(rf)
        eff_ranks.append(ef)
    best_h = int(np.argmax(ratios))
    marker = " *** RESONATOR?" if max(ratios) > 100 else ""
    print(f"  L{L:<7} {max(ratios):>10.2f}     {np.mean(ratios):>10.2f}     H{best_h:<7}  {np.mean(eff_ranks):.1f}{marker}")

print()
print("="*72)
print("Bias outer product fraction at L23 (all heads)")
print("="*72)
fracs = []
for h in range(n_heads):
    _, _, _, _, fb, _ = mesh_svd(23, h)
    fracs.append(fb)
    print(f"  H{h:02d}: bias_frac = {fb:.6f}  ({'dominant' if fb>0.5 else 'minor'})")
print(f"\n  Mean bias_frac at L23: {np.mean(fracs):.6f}")
print(f"  7B finding: bias_frac = 0.9994 (99.94% from b_q ⊗ b_k)")

print("\nDone.")
