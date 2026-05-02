#!/usr/bin/env python3
"""
Phase 10z7: Darwin II — Architecture Exploration
=================================================

F110 showed φ-geometry is EMERGENT (appears only after training).
Now we ask: WHERE does it emerge? Which architectural component enables it?

Test matrix:
  A. Standard transformer (8L, residual + attn + FFN)  — BASELINE (from F110)
  B. No residual connections (attn + FFN, no skip)
  C. MLP-only (no attention, just FFN layers with residual)
  D. Attention-only (no FFN, just attention with residual)
  E. Deep MLP (16 layers, no attention, no residual)
  F. Linear transformer (no nonlinearity in FFN)

For each: measure SVD power-law α, Processor zone α, sign changes,
and inter-zone angles. Compare to the 2/φ and 2/φ² universals.

HYPOTHESIS: Residual connections are necessary for φ-geometry.
Without them, the "Dirichlet series" structure (additive terms)
doesn't exist, so there's nothing to form power laws from.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os

PHI = (1 + math.sqrt(5)) / 2
P = 97  # prime modulus
VOCAB = P + 1


# ============================================================================
# DATASET
# ============================================================================

def make_mod_dataset(p=97, split='train', train_frac=0.7):
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    pairs = all_pairs[:n_train] if split == 'train' else all_pairs[n_train:]
    inputs, targets = [], []
    for a, b in pairs:
        inputs.append([a, b, p])
        targets.append((a + b) % p)
    return torch.tensor(inputs), torch.tensor(targets)


# ============================================================================
# ARCHITECTURE VARIANTS
# ============================================================================

class StandardAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)


# --- A: Standard Transformer (baseline) ---
class StandardLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = StandardAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# --- B: No Residual ---
class NoResidualLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = StandardAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x):
        x = self.attn(self.ln1(x))   # NO skip connection
        x = self.ffn(self.ln2(x))     # NO skip connection
        return x


# --- C: MLP-only (residual, no attention) ---
class MLPOnlyLayer(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x):
        return x + self.ffn(self.ln(x))


# --- D: Attention-only (residual, no FFN) ---
class AttnOnlyLayer(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = StandardAttention(d, n_heads)

    def forward(self, x):
        return x + self.attn(self.ln(x))


# --- E: Deep MLP (no attention, no residual) ---
class DeepMLPLayer(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x):
        return self.ffn(self.ln(x))  # No residual


# --- F: Linear Transformer (no nonlinearity) ---
class LinearLayer(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = StandardAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.Linear(d_ff, d))  # No GELU

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# GENERIC MODEL WRAPPER
# ============================================================================

class ArchModel(nn.Module):
    def __init__(self, layer_fn, n_layers, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(4, d_model)
        self.layers = nn.ModuleList([layer_fn() for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, x, return_intermediates=False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        if return_intermediates:
            intermediates = [h.detach().clone()]
            for layer in self.layers:
                h = layer(h)
                intermediates.append(h.detach().clone())
            h = self.ln_final(h)
            return self.head(h), intermediates
        else:
            for layer in self.layers:
                h = layer(h)
            h = self.ln_final(h)
            return self.head(h)


# ============================================================================
# TRAINING AND ANALYSIS
# ============================================================================

def train_model(model, train_inputs, train_targets, epochs=10000, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    n = len(train_inputs)
    batch_size = 512
    best_train_acc = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, correct, total = 0, 0, 0

        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            inp = train_inputs[batch_idx]
            tgt = train_targets[batch_idx]
            logits = model(inp)
            loss = criterion(logits[:, -1, :], tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
            pred = logits[:, -1, :].argmax(dim=-1)
            correct += (pred == tgt).sum().item()
            total += len(batch_idx)

        scheduler.step()
        acc = correct / total
        best_train_acc = max(best_train_acc, acc)

        if (epoch + 1) % 2000 == 0:
            print(f"      Epoch {epoch+1:5d}: loss={total_loss/total:.4f}, acc={acc:.4f}")

        if acc > 0.999:
            print(f"      Converged at epoch {epoch+1}")
            break

    return best_train_acc


def analyze_geometry(model, inputs, targets, label=""):
    """Focused geometry analysis: SVD power law, projections, zones."""
    n_layers = model.n_layers
    d_model = model.d_model

    n_sample = min(200, len(inputs))
    idx = np.random.choice(len(inputs), n_sample, replace=False)
    sample_inputs = inputs[idx]
    sample_targets = targets[idx]

    all_additions = []
    all_pred_dirs = []

    with torch.no_grad():
        for i in range(n_sample):
            inp = sample_inputs[i:i+1]
            tgt = sample_targets[i].item()
            logits, intermediates = model(inp, return_intermediates=True)
            pred_dir = model.head.weight[tgt].detach().float()
            pred_dir = pred_dir / pred_dir.norm()
            all_pred_dirs.append(pred_dir.numpy())

            additions = []
            for li in range(n_layers):
                h_in = intermediates[li][0, -1].float()
                h_out = intermediates[li+1][0, -1].float()
                additions.append((h_out - h_in).numpy())
            all_additions.append(np.array(additions))

    all_additions = np.array(all_additions)
    all_pred_dirs = np.array(all_pred_dirs)
    mean_add = np.mean(all_additions, axis=0)
    mean_pred = np.mean(all_pred_dirs, axis=0)
    mean_pred /= np.linalg.norm(mean_pred) + 1e-20

    # SVD
    U, S, Vt = np.linalg.svd(mean_add, full_matrices=False)

    # Power-law fit
    k_vals = np.arange(1, len(S) + 1)
    valid = S > S[0] * 1e-6
    if np.sum(valid) > 2:
        log_k = np.log(k_vals[valid])
        log_S = np.log(S[valid])
        coeffs = np.polyfit(log_k, log_S, 1)
        alpha = -coeffs[0]
        R2 = 1 - np.var(log_S - np.polyval(coeffs, log_k)) / np.var(log_S)
    else:
        alpha, R2 = 0.0, 0.0

    # Projections
    full_proj = mean_add @ mean_pred
    signs = np.sign(full_proj)
    sign_changes = int(np.sum(np.abs(np.diff(signs)) > 0))

    # Zone analysis (quarters)
    n_comp = max(n_layers // 4, 1)
    n_targ = max(n_layers // 4, 1)
    n_proc = n_layers - n_comp - n_targ

    proc_layers = list(range(n_comp, n_comp + n_proc))
    if len(proc_layers) >= 2:
        proc_adds = mean_add[proc_layers]
        pU, pS, pVt = np.linalg.svd(proc_adds, full_matrices=False)
        pk = np.arange(1, len(pS) + 1)
        pvalid = pS > pS[0] * 1e-4
        if np.sum(pvalid) > 2:
            pc = np.polyfit(np.log(pk[pvalid]), np.log(pS[pvalid]), 1)
            proc_alpha = -pc[0]
        else:
            proc_alpha = 0.0
    else:
        proc_alpha = 0.0

    # Inter-zone angles
    zones = {
        "Comp": list(range(n_comp)),
        "Proc": proc_layers,
        "Targ": list(range(n_comp + n_proc, n_layers)),
    }
    zone_vecs = {}
    for zn, lrs in zones.items():
        if len(lrs) > 0:
            zone_vecs[zn] = np.mean(mean_add[lrs], axis=0)

    angles = {}
    znames = list(zone_vecs.keys())
    for i in range(len(znames)):
        for j in range(i+1, len(znames)):
            v1, v2 = zone_vecs[znames[i]], zone_vecs[znames[j]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles[f"{znames[i]}-{znames[j]}"] = float(np.degrees(np.arccos(cos_a)))

    # φ-matches
    phi_matches = {}
    for name, val in [("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("2/φ", 2/PHI), ("1", 1.0)]:
        if alpha > 0.01:
            phi_matches[name] = (1 - abs(alpha - val) / val) * 100
    best_phi = max(phi_matches.items(), key=lambda x: x[1]) if phi_matches else ("none", 0)

    proc_phi_matches = {}
    for name, val in [("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("2/φ", 2/PHI), ("1", 1.0)]:
        if proc_alpha > 0.01:
            proc_phi_matches[name] = (1 - abs(proc_alpha - val) / val) * 100
    best_proc_phi = max(proc_phi_matches.items(), key=lambda x: x[1]) if proc_phi_matches else ("none", 0)

    return {
        "alpha": float(alpha),
        "R2": float(R2),
        "proc_alpha": float(proc_alpha),
        "sign_changes": sign_changes,
        "projections": full_proj.tolist(),
        "angles": angles,
        "best_phi": best_phi,
        "best_proc_phi": best_proc_phi,
        "sv_top5": S[:5].tolist(),
    }


def evaluate(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        return (logits[:, -1, :].argmax(dim=-1) == targets).float().mean().item()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 10z7: DARWIN II — ARCHITECTURE EXPLORATION")
    print("=" * 70)

    train_inputs, train_targets = make_mod_dataset(P, 'train')
    test_inputs, test_targets = make_mod_dataset(P, 'test')
    print(f"\n  Task: (a+b) mod {P}, train={len(train_inputs)}, test={len(test_inputs)}")

    D = 64
    HEADS = 4
    DFF = 256
    NLAYERS = 8

    architectures = {
        "A_standard": lambda: StandardLayer(D, HEADS, DFF),
        "B_no_residual": lambda: NoResidualLayer(D, HEADS, DFF),
        "C_mlp_only": lambda: MLPOnlyLayer(D, DFF),
        "D_attn_only": lambda: AttnOnlyLayer(D, HEADS),
        "E_deep_mlp": lambda: DeepMLPLayer(D, DFF),
        "F_linear": lambda: LinearLayer(D, HEADS, DFF),
    }

    results = {}

    for arch_name, layer_fn in architectures.items():
        print(f"\n{'='*70}")
        print(f"  ARCHITECTURE: {arch_name}")
        print(f"{'='*70}")

        torch.manual_seed(42)
        model = ArchModel(layer_fn, NLAYERS, D)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        # Train
        print(f"    Training...")
        best_acc = train_model(model, train_inputs, train_targets, epochs=10000, lr=3e-4)
        model.eval()
        train_acc = evaluate(model, train_inputs, train_targets)
        test_acc = evaluate(model, test_inputs, test_targets)
        print(f"    Final: train={train_acc:.4f}, test={test_acc:.4f}")

        # Analyze
        geo = analyze_geometry(model, test_inputs, test_targets, arch_name)

        results[arch_name] = {
            "n_params": n_params,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "geometry": geo,
        }

        alpha = geo["alpha"]
        proc_alpha = geo["proc_alpha"]
        bp = geo["best_phi"]
        bpp = geo["best_proc_phi"]
        sc = geo["sign_changes"]

        print(f"    α = {alpha:.4f} (best φ: {bp[0]} at {bp[1]:.1f}%)")
        print(f"    Proc α = {proc_alpha:.4f} (best φ: {bpp[0]} at {bpp[1]:.1f}%)")
        print(f"    Sign changes: {sc}/{NLAYERS-1}")
        if geo["angles"]:
            for k, v in geo["angles"].items():
                print(f"    Angle {k}: {v:.1f}°")

    # ═════════════════════════════════════════════
    # COMPARISON TABLE
    # ═════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n  {'Arch':15s} {'Params':>8s} {'TrAcc':>7s} {'TeAcc':>7s} "
          f"{'α':>7s} {'ProcA':>7s} {'SignΔ':>6s} {'Best φ':>10s} {'P.φ':>10s}")
    print(f"  {'─'*15} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*10} {'─'*10}")

    for arch_name, r in results.items():
        g = r["geometry"]
        bp = g["best_phi"]
        bpp = g["best_proc_phi"]
        print(f"  {arch_name:15s} {r['n_params']:>8,} {r['train_acc']:>7.3f} {r['test_acc']:>7.3f} "
              f"{g['alpha']:>7.3f} {g['proc_alpha']:>7.3f} {g['sign_changes']:>6d} "
              f"{bp[0]:>5s} {bp[1]:>4.0f}% {bpp[0]:>5s} {bpp[1]:>4.0f}%")

    # Qwen reference
    print(f"  {'Qwen2.5-7B':15s} {'7B':>8s} {'—':>7s} {'—':>7s} "
          f"{'1.223':>7s} {'0.769':>7s} {'many':>6s} {'2/φ':>5s} {'99':>4s}% {'2/φ²':>5s} {'99':>4s}%")

    # ═════════════════════════════════════════════
    # VERDICTS
    # ═════════════════════════════════════════════
    print(f"\n  VERDICTS:")

    # Which architectures developed φ-geometry?
    phi_arches = []
    no_phi_arches = []
    for arch_name, r in results.items():
        g = r["geometry"]
        bp = g["best_phi"]
        if bp[1] > 80:
            phi_arches.append((arch_name, bp))
        else:
            no_phi_arches.append((arch_name, bp))

    if phi_arches:
        print(f"\n    φ-geometry EMERGED in:")
        for name, (expr, match) in phi_arches:
            print(f"      {name}: α ≈ {expr} ({match:.1f}%)")

    if no_phi_arches:
        print(f"\n    φ-geometry DID NOT emerge in:")
        for name, (expr, match) in no_phi_arches:
            g = results[name]["geometry"]
            print(f"      {name}: α = {g['alpha']:.3f} (best {expr} at {match:.1f}%)")

    # What's necessary?
    print(f"\n    ARCHITECTURAL REQUIREMENTS:")
    has_residual = {"A_standard", "C_mlp_only", "D_attn_only", "F_linear"}
    has_attn = {"A_standard", "B_no_residual", "D_attn_only", "F_linear"}
    has_gelu = {"A_standard", "B_no_residual", "C_mlp_only", "D_attn_only", "E_deep_mlp"}

    phi_names = {n for n, _ in phi_arches}

    if phi_names <= has_residual and phi_names:
        residual_only = phi_names - (phi_names - has_residual)
        if residual_only:
            print(f"    → Residual connections: LIKELY NECESSARY")
    if phi_names <= has_attn and phi_names:
        print(f"    → Attention: may be necessary")
    if phi_names <= has_gelu and phi_names:
        print(f"    → GELU nonlinearity: may be necessary")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/phase10z7_darwin_architectures.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to results/phase10z7_darwin_architectures.json")


if __name__ == "__main__":
    main()
