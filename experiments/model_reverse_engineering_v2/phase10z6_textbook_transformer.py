#!/usr/bin/env python3
"""
Phase 10z6: Textbook Transformer — Architectural vs Emergent Geometry
======================================================================

The Qwen2.5-7B results (F107-F109) showed ζ-like structure:
- Three-stage pipeline (Compressor/Processor/Targeter)
- φ-governed power laws (2/φ, 1/φ, 2/φ²)
- Conditional convergence (oscillating projections)
- Rank-1 prediction dominance (91.8%)

BUT: Is this the ARCHITECTURE talking, or the TRAINING talking?

TEST: Build a minimal textbook transformer (8 layers, 64-dim, 4 heads).
1. Analyze UNTRAINED (random init) — is geometry already there?
2. Train on modular arithmetic (a + b) mod 97
3. Analyze TRAINED — does geometry sharpen?
4. Compare to Qwen findings

If φ-geometry appears UNTRAINED → it's architectural (inherent to residual stream)
If φ-geometry appears only TRAINED → it's emergent (learned through optimization)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os

PHI = (1 + math.sqrt(5)) / 2

# ============================================================================
# MINIMAL TEXTBOOK TRANSFORMER
# ============================================================================

class TextbookAttention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

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


class TextbookFFN(nn.Module):
    """Standard FFN with GELU activation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


class TextbookLayer(nn.Module):
    """Pre-norm transformer layer (like GPT-2/Qwen)."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TextbookAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TextbookFFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TextbookTransformer(nn.Module):
    """
    Minimal textbook transformer for modular arithmetic.
    Input: 3 tokens [a, b, =]
    Output: prediction of (a + b) mod p at the = position
    """
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=8, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4, d_model)  # max 4 positions
        self.layers = nn.ModuleList([
            TextbookLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

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
# MODULAR ARITHMETIC DATASET
# ============================================================================

def make_mod_dataset(p=97, split='train', train_frac=0.7):
    """Generate (a + b) mod p dataset."""
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.seed(42)
    np.random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_frac)
    if split == 'train':
        pairs = all_pairs[:n_train]
    else:
        pairs = all_pairs[n_train:]

    # Token format: [a, b, =] → predict (a+b) mod p at position 2
    # Vocabulary: 0..p-1 for numbers, p for '='
    inputs = []
    targets = []
    for a, b in pairs:
        inputs.append([a, b, p])  # p is the '=' token
        targets.append((a + b) % p)

    return torch.tensor(inputs), torch.tensor(targets)


# ============================================================================
# GEOMETRY ANALYSIS (same measurements as Qwen)
# ============================================================================

def analyze_geometry(model, inputs, targets, label=""):
    """
    Run the same geometric analysis as on Qwen:
    1. SVD of addition matrix
    2. Power-law decay
    3. Cumulative projection / conditional convergence
    4. Zone-by-zone analysis
    5. Angle structure
    """
    print(f"\n{'='*70}")
    print(f"  GEOMETRY ANALYSIS: {label}")
    print(f"{'='*70}")

    n_layers = model.n_layers
    d_model = model.d_model

    # Sample a batch of inputs
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

            # Prediction direction: weight vector for correct answer
            pred_dir = model.head.weight[tgt].detach().float()
            pred_dir = pred_dir / pred_dir.norm()
            all_pred_dirs.append(pred_dir.numpy())

            # Per-layer additions (last token position, index 2 = '=')
            additions = []
            for li in range(n_layers):
                h_in = intermediates[li][0, -1].float()    # before layer li
                h_out = intermediates[li+1][0, -1].float()  # after layer li
                add = (h_out - h_in).numpy()
                additions.append(add)

            all_additions.append(np.array(additions))

    all_additions = np.array(all_additions)    # (n_sample, n_layers, d_model)
    all_pred_dirs = np.array(all_pred_dirs)     # (n_sample, d_model)

    # Mean addition matrix
    mean_add = np.mean(all_additions, axis=0)   # (n_layers, d_model)
    mean_pred = np.mean(all_pred_dirs, axis=0)
    mean_pred /= np.linalg.norm(mean_pred)

    # ── SVD ──
    U, S, Vt = np.linalg.svd(mean_add, full_matrices=False)
    print(f"\n  SVD: ({n_layers}, {d_model}) → S = {S[:5].round(3)}")

    # Power-law fit
    k_vals = np.arange(1, len(S) + 1)
    valid = S > S[0] * 1e-6
    if np.sum(valid) > 2:
        log_k = np.log(k_vals[valid])
        log_S = np.log(S[valid])
        coeffs = np.polyfit(log_k, log_S, 1)
        alpha = -coeffs[0]
        R2 = 1 - np.var(log_S - np.polyval(coeffs, log_k)) / np.var(log_S)

        print(f"  Power law: σ_k ~ k^(-{alpha:.4f}), R² = {R2:.4f}")

        phi_matches = [
            ("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("2/φ", 2/PHI),
            ("1/φ²", 1/PHI**2), ("φ/2", PHI/2), ("1", 1.0), ("φ", PHI),
        ]
        print(f"  φ-matches for α = {alpha:.4f}:")
        for name, val in phi_matches:
            match = (1 - abs(alpha - val) / val) * 100
            marker = " ★" if match > 95 else ""
            print(f"    {name:8s} = {val:.4f}: {match:.1f}%{marker}")
    else:
        alpha = None
        R2 = None
        print(f"  SVD: insufficient valid singular values for power law fit")

    # ── Projection onto prediction direction ──
    full_proj = mean_add @ mean_pred  # (n_layers,)

    print(f"\n  Per-layer projection onto prediction direction:")
    cumsum = 0
    max_abs = max(abs(p) for p in full_proj) if max(abs(p) for p in full_proj) > 0 else 1
    for li in range(n_layers):
        cumsum += full_proj[li]
        bar_len = int(abs(full_proj[li]) / max_abs * 25)
        sign = "+" if full_proj[li] >= 0 else "-"
        bar = "█" * bar_len
        print(f"    L{li:02d}: {full_proj[li]:+8.4f}  cum={cumsum:+8.4f}  {sign}{bar}")

    # Check conditional convergence: do projections oscillate?
    signs = np.sign(full_proj)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
    monotonic = np.all(np.diff(np.cumsum(full_proj)) >= 0) or np.all(np.diff(np.cumsum(full_proj)) <= 0)

    print(f"\n  Sign changes: {sign_changes}/{n_layers-1}")
    print(f"  Monotonic cumulative: {'YES' if monotonic else 'NO'}")
    print(f"  Conditional convergence: {'YES (oscillating)' if sign_changes > n_layers//3 else 'NO (mostly monotonic)'}")

    # ── SVD projection onto prediction ──
    Vt_pred = Vt @ mean_pred
    sv_contribs = np.abs(S * Vt_pred)
    total_sv = np.sum(sv_contribs**2)

    print(f"\n  SVD components projected onto prediction:")
    cumul = 0
    for k in range(min(len(S), n_layers)):
        cumul += sv_contribs[k]**2
        pct = cumul / total_sv * 100 if total_sv > 0 else 0
        bar = "█" * int(sv_contribs[k] / (sv_contribs[0] + 1e-20) * 20)
        cryst = " ★ 99%" if pct > 99 and (k == 0 or (cumul - sv_contribs[k]**2) / total_sv * 100 < 99) else ""
        print(f"    SV{k:02d}: {sv_contribs[k]:8.4f}  cumul={pct:6.1f}%  {bar}{cryst}")

    # Crystallization rank
    cumul_pct = np.cumsum(sv_contribs**2) / (total_sv + 1e-20)
    cryst_95 = np.searchsorted(cumul_pct, 0.95) + 1
    cryst_99 = np.searchsorted(cumul_pct, 0.99) + 1
    print(f"\n  Crystallization: 95% at rank {cryst_95}, 99% at rank {cryst_99}")

    # ── Zone analysis (thirds for 8 layers) ──
    n_comp = max(n_layers // 4, 1)      # first quarter
    n_targ = max(n_layers // 4, 1)      # last quarter
    n_proc = n_layers - n_comp - n_targ  # middle half

    zones = {
        f"Compressor (L0-{n_comp-1})": list(range(n_comp)),
        f"Processor (L{n_comp}-{n_comp+n_proc-1})": list(range(n_comp, n_comp+n_proc)),
        f"Targeter (L{n_comp+n_proc}-{n_layers-1})": list(range(n_comp+n_proc, n_layers)),
    }

    print(f"\n  Zone-by-zone:")
    zone_means = {}
    for zname, layers in zones.items():
        zone_adds = mean_add[layers]
        zone_U, zone_S, zone_Vt = np.linalg.svd(zone_adds, full_matrices=False)

        # Zone power law
        zk = np.arange(1, len(zone_S) + 1)
        zvalid = zone_S > zone_S[0] * 1e-4
        if np.sum(zvalid) > 2:
            zcoeffs = np.polyfit(np.log(zk[zvalid]), np.log(zone_S[zvalid]), 1)
            zalpha = -zcoeffs[0]
            zR2 = 1 - np.var(np.log(zone_S[zvalid]) - np.polyval(zcoeffs, np.log(zk[zvalid]))) / np.var(np.log(zone_S[zvalid]))
        else:
            zalpha = 0
            zR2 = 0

        # Zone projection
        zone_proj = np.mean(full_proj[layers])

        zone_means[zname] = zone_adds

        print(f"    {zname}:")
        print(f"      α = {zalpha:.4f} (R² = {zR2:.3f}), mean_proj = {zone_proj:+.4f}")

        # Best φ-match
        best_name, best_val, best_match = "", 0, -999
        for pname, pval in [("1/φ", 1/PHI), ("2/φ²", 2/PHI**2), ("2/φ", 2/PHI), ("1", 1.0)]:
            m = (1 - abs(zalpha - pval) / pval) * 100
            if m > best_match:
                best_name, best_val, best_match = pname, pval, m
        print(f"      Best φ: α ≈ {best_name} = {best_val:.4f} ({best_match:.1f}%)")

    # ── Inter-zone angles ──
    zone_names = list(zones.keys())
    zone_vecs = {}
    for zname, layers in zones.items():
        zone_vecs[zname] = np.mean(mean_add[layers], axis=0)  # mean over layers → (d_model,)

    print(f"\n  Inter-zone angles:")
    for i in range(len(zone_names)):
        for j in range(i+1, len(zone_names)):
            v1 = zone_vecs[zone_names[i]]
            v2 = zone_vecs[zone_names[j]]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-20)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            marker = ""
            if abs(angle - 70.53) < 5: marker = " ≈ arccos(1/3) tetrahedral"
            elif abs(angle - 72) < 5: marker = " ≈ 72° pentagonal"
            elif abs(angle - 90) < 5: marker = " ≈ 90° orthogonal"
            elif abs(angle - 60) < 5: marker = " ≈ 60° hexagonal"
            print(f"    {zone_names[i][:12]} ↔ {zone_names[j][:12]}: {angle:.1f}°{marker}")

    # ── Collect results ──
    return {
        "alpha": float(alpha) if alpha is not None else None,
        "R2": float(R2) if R2 is not None else None,
        "projections": full_proj.tolist(),
        "sign_changes": int(sign_changes),
        "cryst_95": int(cryst_95),
        "cryst_99": int(cryst_99),
        "sv_top5": S[:5].tolist(),
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_inputs, train_targets, epochs=5000, lr=1e-3, log_every=500):
    """Train on modular arithmetic."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    n = len(train_inputs)
    batch_size = 512
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            inp = train_inputs[batch_idx]
            tgt = train_targets[batch_idx]

            logits = model(inp)
            loss = criterion(logits[:, -1, :], tgt)  # predict at '=' position

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

        if (epoch + 1) % log_every == 0 or acc > 0.99:
            print(f"    Epoch {epoch+1:5d}: loss={total_loss/total:.4f}, acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc

        if acc > 0.999:
            print(f"    Converged at epoch {epoch+1}!")
            break

    return best_acc


def evaluate_model(model, inputs, targets):
    """Evaluate accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        pred = logits[:, -1, :].argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
    return acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 10z6: TEXTBOOK TRANSFORMER — ARCHITECTURAL vs EMERGENT GEOMETRY")
    print("=" * 70)

    P = 97  # prime modulus
    vocab_size = P + 1  # 0..96 for numbers, 97 for '='
    d_model = 64
    n_heads = 4
    n_layers = 8
    d_ff = 256

    print(f"\n  Task: (a + b) mod {P}")
    print(f"  Architecture: {n_layers} layers, d={d_model}, {n_heads} heads, d_ff={d_ff}")
    print(f"  Vocab: {vocab_size} ({P} numbers + '=')")

    # Create dataset
    train_inputs, train_targets = make_mod_dataset(P, 'train')
    test_inputs, test_targets = make_mod_dataset(P, 'test')
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    # Build model
    torch.manual_seed(42)
    model = TextbookTransformer(vocab_size, d_model, n_heads, n_layers, d_ff)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ═══════════════════════════════════════════════
    # PHASE 1: UNTRAINED ANALYSIS
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: UNTRAINED (random initialization)")
    print(f"{'='*70}")

    model.eval()
    untrained_acc = evaluate_model(model, test_inputs, test_targets)
    print(f"\n  Test accuracy: {untrained_acc:.4f} (chance = {1/P:.4f})")

    untrained_results = analyze_geometry(model, test_inputs, test_targets, "UNTRAINED")

    # ═══════════════════════════════════════════════
    # PHASE 2: TRAINING
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: TRAINING")
    print(f"{'='*70}")

    best_acc = train_model(model, train_inputs, train_targets,
                           epochs=15000, lr=3e-4, log_every=1000)

    train_acc = evaluate_model(model, train_inputs, train_targets)
    test_acc = evaluate_model(model, test_inputs, test_targets)
    print(f"\n  Final: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # ═══════════════════════════════════════════════
    # PHASE 3: TRAINED ANALYSIS
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: TRAINED")
    print(f"{'='*70}")

    model.eval()
    trained_results = analyze_geometry(model, test_inputs, test_targets, "TRAINED")

    # ═══════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  COMPARISON: UNTRAINED vs TRAINED vs QWEN")
    print(f"{'='*70}")

    qwen_vals = {
        "alpha": 1.223,
        "cryst_99": 3,
        "sign_changes": "many (conditional convergence)",
    }

    print(f"\n  {'Metric':25s}  {'Untrained':>12}  {'Trained':>12}  {'Qwen 7B':>12}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*12}")

    for key, qval in [
        ("SV decay α", ("alpha", qwen_vals["alpha"])),
        ("Crystallization (99%)", ("cryst_99", qwen_vals["cryst_99"])),
        ("Sign changes", ("sign_changes", qwen_vals["sign_changes"])),
    ]:
        uval = untrained_results.get(qval[0], "—")
        tval = trained_results.get(qval[0], "—")
        if isinstance(uval, float):
            uval = f"{uval:.4f}"
        if isinstance(tval, float):
            tval = f"{tval:.4f}"
        print(f"  {key:25s}  {str(uval):>12}  {str(tval):>12}  {str(qval[1]):>12}")

    # Verdict
    print(f"\n  VERDICT:")
    if untrained_results.get("alpha") and trained_results.get("alpha"):
        u_alpha = untrained_results["alpha"]
        t_alpha = trained_results["alpha"]
        q_alpha = 1.223

        if abs(u_alpha - q_alpha) / q_alpha < 0.2:
            print(f"    φ-power law: ARCHITECTURAL (present untrained, α={u_alpha:.3f})")
        elif abs(t_alpha - q_alpha) / q_alpha < 0.2:
            print(f"    φ-power law: EMERGENT (only after training, α={t_alpha:.3f})")
        else:
            print(f"    φ-power law: NEITHER (untrained α={u_alpha:.3f}, trained α={t_alpha:.3f}, Qwen α={q_alpha:.3f})")

    u_sc = untrained_results.get("sign_changes", 0)
    t_sc = trained_results.get("sign_changes", 0)
    if u_sc > n_layers // 3:
        print(f"    Conditional convergence: ARCHITECTURAL (oscillating untrained)")
    elif t_sc > n_layers // 3:
        print(f"    Conditional convergence: EMERGENT (oscillating only after training)")
    else:
        print(f"    Conditional convergence: NOT OBSERVED (sign changes: untrained={u_sc}, trained={t_sc})")

    # Save
    save_data = {
        "architecture": {
            "n_layers": n_layers, "d_model": d_model,
            "n_heads": n_heads, "d_ff": d_ff, "n_params": n_params
        },
        "task": {"modulus": P, "train_size": len(train_inputs), "test_size": len(test_inputs)},
        "untrained": untrained_results,
        "trained": trained_results,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/phase10z6_textbook_transformer.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to results/phase10z6_textbook_transformer.json")


if __name__ == "__main__":
    main()
