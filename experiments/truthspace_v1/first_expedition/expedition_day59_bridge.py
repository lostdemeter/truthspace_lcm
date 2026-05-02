#!/usr/bin/env python3
"""
Expedition Day 59 — The Bridge: L14 → L27

Day 58 established the corrected architecture:
  Token → W_in → [pre-semantic] → L14 (Zone C) → [elaboration+knowledge] →
  L27 → W_out = W_in.T → next token

Zone C is the thinking space. To emit tokens from Zone C positions we need
a bridge from L14 hidden states to L27 hidden states.

The residual stream predicts this should be approximately linear:
  h_L27 = h_L14 + Σ(residual contributions L15→L27)

For isolated tokens processed without rich context, the residual contribution
is a deterministic function of h_L14 alone — because the model sees the same
minimal context each time (BOS + token). So a linear M: h_L14 → h_L27
should be learnable and may generalise.

Tests:
  B1  Learn M: h_L14 → h_L27 (ridge regression). Measure R² and cosine
      similarity of predicted vs actual h_L27 on held-out words.

  B2  Token recovery via bridge: does W_out @ M(h_L14(w)) place w in top-k?
      This answers: "can we emit any Zone C concept as a token?"

  B3  Bridge quality for contextual sequences: take a fill prompt, extract
      contextual h_L14 at last position, apply M, compare h_L27 cosine vs
      actual. Does the generation improve over raw φ-L14?

  B4  Full LCM pipeline via bridge:
        (a) Target a Zone C word (e.g. "Paris" for "capital of France")
        (b) Look up h_L14("Paris") from cache
        (c) h_L27_pred = M @ h_L14("Paris") + b
        (d) logit = h_L27_pred @ W_out.T → argmax → predicted token
      This is the closed LCM loop: concept address → output token.
"""

import json, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR   = Path(__file__).parent
CACHE_FILE   = str(SCRIPT_DIR / "day27_hs_cache.npz")
L27_CACHE    = str(SCRIPT_DIR / "day59_hs_27_cache.npz")
ATLAS_FILE   = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE  = str(SCRIPT_DIR / "day59_bridge.json")
MODEL_ID     = "Qwen/Qwen2-1.5B-Instruct"
BRIDGE_LAYER = 14
OUTPUT_LAYER = 27

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

FILL_TESTS = [
    ("The capital of France is",            "Paris"),
    ("The opposite of hot is",              "cold"),
    ("Dogs are known for their ability to", "bark"),
    ("Water freezes and turns into",        "ice"),
    ("A female horse is called a",          "mare"),
    ("The plural of cat is",                "cats"),
    ("She is a great singer and he is a great", "dancer"),
    ("The sun rises in the east and sets in the", "west"),
    ("A baby dog is called a",              "puppy"),
    ("The colour of grass is",              "green"),
    ("The opposite of tall is",             "short"),
    ("Kings and",                           "queens"),
    ("Boys and",                            "girls"),
    ("The past tense of walk is",           "walked"),
    ("The comparative form of big is",      "bigger"),
    ("The adverb form of quick is",         "quickly"),
    ("An adult female cat is called a",     "queen"),
    ("A group of wolves is called a",       "pack"),
    ("The colour of the sky on a clear day is", "blue"),
    ("The opposite of ancient is",          "modern"),
]

print("=" * 70)
print("  Expedition Day 59 — The Bridge: L14 → L27")
print("=" * 70)

# ── Load L14 cache ─────────────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
hs23_all  = npz['hs_23'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}
N         = len(words_all)

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# ── Load model ─────────────────────────────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers  = model.config.num_hidden_layers   # 28
hidden_sz = model.config.hidden_size          # 1536
vocab_sz  = model.config.vocab_size
W_out     = model.lm_head.weight.detach().numpy().astype(np.float64)  # [V, H]
print(f"  n_layers={n_layers}, hidden={hidden_sz}, vocab={vocab_sz}")
assert OUTPUT_LAYER <= n_layers, f"OUTPUT_LAYER {OUTPUT_LAYER} exceeds model depth {n_layers}"


# ── Extract / load L27 hidden states ──────────────────────────────────────
if Path(L27_CACHE).exists():
    print(f"\n  L27 cache found: {L27_CACHE}")
    c27   = np.load(L27_CACHE)
    hs27_all = c27['hs_27'].astype(np.float64)
    print(f"  Loaded {len(hs27_all)} L27 hidden states")
else:
    print(f"\n  Extracting L27 hidden states for {N} words (batched) ...")
    hs27_all = np.zeros((N, hidden_sz), dtype=np.float32)
    BATCH    = 64
    t0       = time.time()
    # Pre-tokenise; all words in cache are single-token with space prefix,
    # so each input is [BOS, token_id] (length 2) — uniform length, no padding needed.
    with torch.no_grad():
        for b_start in range(0, N, BATCH):
            b_end   = min(b_start + BATCH, N)
            batch_w = words_all[b_start:b_end]

            # Build input_ids: list of [bos, token_id]
            rows = []
            valid = []
            for i, w in enumerate(batch_w):
                ids = tok.encode(w, add_special_tokens=False)
                if ids:
                    rows.append(ids[-1:])    # take last subtoken if multi-token
                    valid.append(i)

            if not rows:
                continue

            # Pad to same length (most rows are length 1 after BOS is added by model)
            max_len = max(len(r) for r in rows)
            padded  = [r + [tok.pad_token_id or 0] * (max_len - len(r)) for r in rows]
            inp     = tok(batch_w.tolist() if hasattr(batch_w, 'tolist') else list(batch_w),
                         return_tensors='pt', padding=True, add_special_tokens=True)
            out     = model(**inp, output_hidden_states=True)
            # Extract last non-pad token position for each item
            attn    = inp['attention_mask']
            last_pos = attn.sum(dim=1) - 1   # [B]
            for j in range(len(batch_w)):
                pos = int(last_pos[j].item())
                hs27_all[b_start + j] = out.hidden_states[OUTPUT_LAYER][j, pos, :].numpy()

            if (b_start + BATCH) % 2000 < BATCH:
                done    = b_start + BATCH
                elapsed = (time.time() - t0) / 60
                eta     = elapsed / max(done, 1) * (N - done)
                print(f"  [{done:5d}/{N}]  {elapsed:.1f} min  ETA {eta:.1f} min")

    hs27_all = hs27_all.astype(np.float64)
    np.savez_compressed(L27_CACHE, words=words_all, hs_27=hs27_all)
    print(f"  Cached: {L27_CACHE}")


# ── Build φ-space tools ────────────────────────────────────────────────────
def build_z2(pairs, hs_dict):
    ds = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx+a, pfx+b
            if wa in hs_dict and wb in hs_dict:
                d  = hs_dict[wb] - hs_dict[wa]
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    _, _, Vt = np.linalg.svd(np.stack(ds), full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])

z2 = build_z2(KILLING_PAIRS, {w: hs14_all[w2i[w]] for w in words_all if w in w2i})

def to_phi(h, z2):
    h    = h.astype(np.float64)
    hn   = h / (np.linalg.norm(h) + 1e-20)
    perp = hn - np.dot(hn, z2) * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

phi14_all = np.stack([to_phi(hs14_all[i], z2) for i in range(N)])


# ═══════════════════════════════════════════════════════════════════════════
# B1 — Learn Linear Bridge M: h_L14 → h_L27
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"B1 — Linear Bridge  h_L14 → h_L27")
print(f"{'='*70}\n")

# 80/20 train/test split
rng      = np.random.default_rng(42)
idx      = rng.permutation(N)
n_train  = int(0.8 * N)
tr_idx   = idx[:n_train]
te_idx   = idx[n_train:]

X_tr = hs14_all[tr_idx]   # [n_train, H]
Y_tr = hs27_all[tr_idx]
X_te = hs14_all[te_idx]
Y_te = hs27_all[te_idx]

# Ridge regression: Y = X @ M.T + b
# Closed form: M.T = (X.T X + λI)^{-1} X.T Y
print(f"  Train size: {n_train}   Test size: {len(te_idx)}")
print(f"  Fitting bridge (ridge, λ=1e-3) ...")

from numpy.linalg import solve

lambda_reg = 1e-3
X_tr_b = np.hstack([X_tr, np.ones((len(X_tr), 1))])   # add bias column
X_te_b = np.hstack([X_te, np.ones((len(X_te), 1))])

A  = X_tr_b.T @ X_tr_b
A += lambda_reg * np.eye(A.shape[0])
B  = X_tr_b.T @ Y_tr
W  = solve(A, B)   # shape [H+1, H]

M_bridge = W[:-1, :]   # [H, H]  — the linear map
b_bridge = W[-1:, :]   # [1, H]  — bias

# Evaluate
Y_te_pred = X_te_b @ W
cos_te    = np.array([cosine(Y_te_pred[i], Y_te[i]) for i in range(len(te_idx))])
l2_te     = np.linalg.norm(Y_te_pred - Y_te, axis=1)
l2_raw    = np.linalg.norm(Y_te, axis=1)

# R² per dimension
ss_res = np.sum((Y_te - Y_te_pred)**2, axis=0)
ss_tot = np.sum((Y_te - Y_te.mean(0))**2, axis=0)
r2_per_dim = 1 - ss_res / (ss_tot + 1e-20)
r2_mean    = float(np.mean(r2_per_dim))

print(f"\n  Bridge quality (test set, {len(te_idx)} words):")
print(f"    Mean R² (per dimension):     {r2_mean:.6f}")
print(f"    Mean cos(pred, actual):      {np.mean(cos_te):.6f}")
print(f"    Median cos(pred, actual):    {np.median(cos_te):.6f}")
print(f"    % cos > 0.99:                {np.mean(cos_te > 0.99)*100:.1f}%")
print(f"    % cos > 0.95:                {np.mean(cos_te > 0.95)*100:.1f}%")
print(f"    % cos > 0.90:                {np.mean(cos_te > 0.90)*100:.1f}%")
print(f"    Mean L2 error / ||h_L27||:   {np.mean(l2_te / (l2_raw + 1e-20)):.6f}")

# Best/worst recovered words
best_words  = [words_all[te_idx[i]] for i in np.argsort(-cos_te)[:5]]
worst_words = [words_all[te_idx[i]] for i in np.argsort(cos_te)[:5]]
print(f"\n  Best recovered:  {best_words}")
print(f"  Worst recovered: {worst_words}")

# Also test the L14→L23 bridge for comparison
print(f"\n  Comparison — L14→L23 bridge (same method):")
Y23_tr   = hs23_all[tr_idx]; Y23_te = hs23_all[te_idx]
A23      = X_tr_b.T @ X_tr_b + lambda_reg * np.eye(X_tr_b.shape[1])
B23      = X_tr_b.T @ Y23_tr
W23      = solve(A23, B23)
Y23_pred = X_te_b @ W23
cos23    = np.array([cosine(Y23_pred[i], Y23_te[i]) for i in range(len(te_idx))])
ss_res23 = np.sum((Y23_te - Y23_pred)**2, axis=0)
ss_tot23 = np.sum((Y23_te - Y23_te.mean(0))**2, axis=0)
r2_23    = float(np.mean(1 - ss_res23 / (ss_tot23 + 1e-20)))
print(f"    Mean R² L14→L23: {r2_23:.6f}   Mean cos: {np.mean(cos23):.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# B2 — Token Recovery via Bridge
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"B2 — Token Recovery:  W_out @ M(h_L14(w))  →  w?")
print(f"{'='*70}\n")
print(f"  For each test-set word, decode bridge-predicted h_L27 via W_out.")
print(f"  Does it find the word itself in top-k?\n")

# Use full test set
X_te_b_all = np.hstack([hs14_all[te_idx], np.ones((len(te_idx), 1))])
h27_pred_te = X_te_b_all @ W      # [n_test, H]

# Decode: logits = h27_pred_te @ W_out.T
# We'll check top-1, top-5, top-10
hits_1 = 0; hits_5 = 0; hits_10 = 0

examples = []
for i in range(len(te_idx)):
    w   = words_all[te_idx[i]]
    logits = h27_pred_te[i] @ W_out.T    # [V]
    top10  = np.argsort(-logits)[:10]
    top10_strs = [tok.decode([t]).strip().lower() for t in top10]

    w_clean = w.strip().lower()
    hit1  = w_clean == top10_strs[0]
    hit5  = w_clean in top10_strs[:5]
    hit10 = w_clean in top10_strs[:10]
    if hit1:  hits_1  += 1
    if hit5:  hits_5  += 1
    if hit10: hits_10 += 1

    if len(examples) < 20:
        mark = '✓' if hit5 else '✗'
        examples.append((w.strip(), top10_strs[0], top10_strs[:5], mark))

n_te = len(te_idx)
print(f"  Top-1  accuracy: {hits_1}/{n_te} = {hits_1/n_te:.4f}")
print(f"  Top-5  accuracy: {hits_5}/{n_te} = {hits_5/n_te:.4f}")
print(f"  Top-10 accuracy: {hits_10}/{n_te} = {hits_10/n_te:.4f}")
print(f"\n  Sample (first 20 test words):")
print(f"  {'Word':<18}  {'Top-1 pred':<18}  {'Hit@5':<6}  Top-5 predictions")
for w, p1, p5, m in examples[:20]:
    print(f"  {w:<18}  {p1:<18}  {m:<6}  {p5}")

# Also compare: direct W_out @ h_L27_actual (ceiling)
hits_1_real = 0; hits_5_real = 0
for i in range(len(te_idx)):
    w       = words_all[te_idx[i]]
    logits  = hs27_all[te_idx[i]] @ W_out.T
    top5    = [tok.decode([t]).strip().lower() for t in np.argsort(-logits)[:5]]
    w_clean = w.strip().lower()
    if w_clean == top5[0]: hits_1_real += 1
    if w_clean in top5:    hits_5_real += 1
print(f"\n  Ceiling (W_out @ actual h_L27):")
print(f"    Top-1: {hits_1_real/n_te:.4f}   Top-5: {hits_5_real/n_te:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# B3 — Bridge Quality for Contextual h_L14
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"B3 — Contextual Bridge:  M(contextual h_L14) → h_L27")
print(f"{'='*70}\n")
print(f"  For fill prompts: extract contextual h_L14 at last position,")
print(f"  apply bridge → h_L27_pred, compare to actual h_L27.\n")

def run_forward(prompt):
    inputs = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {L: out.hidden_states[L][0, -1, :].numpy().astype(np.float64)
          for L in [BRIDGE_LAYER, OUTPUT_LAYER]}
    logits = out.logits[0, -1, :].numpy()
    return hs, logits

def nearest_zone_c(h_l14_or_phi, use_phi=False):
    if use_phi:
        sims = phi14_all @ h_l14_or_phi
    else:
        phi_q = to_phi(h_l14_or_phi, z2)
        sims  = phi14_all @ phi_q
    best = np.argmax(sims)
    return words_all[best], float(sims[best])

print(f"  {'Prompt':<43}  {'Expected':<10}  "
      f"{'LM @1':<10}  cos(bridge,act)  {'Bridge @1'}")
print(f"  {'-'*95}")

b3_results = []
for prompt, expected in FILL_TESTS:
    hs, lm_logits = run_forward(prompt)
    h14_ctx   = hs[BRIDGE_LAYER]
    h27_act   = hs[OUTPUT_LAYER]

    # Apply bridge to contextual h_L14
    x_b = np.append(h14_ctx, 1.0).reshape(1, -1)   # [1, H+1]
    h27_pred  = (x_b @ W)[0]                         # [H]

    cos_bridge_actual = cosine(h27_pred, h27_act)

    # Decode predictions
    lm_top1  = tok.decode([np.argmax(lm_logits)]).strip()
    br_top5  = np.argsort(-(h27_pred @ W_out.T))[:5]
    br_top1  = tok.decode([br_top5[0]]).strip()
    br_words = [tok.decode([t]).strip().lower() for t in br_top5]

    exp_lower = expected.lower()
    lm_hit  = lm_top1.lower() == exp_lower
    br_hit  = exp_lower in br_words

    lm_mark = '✓' if lm_hit else '✗'
    br_mark = '✓' if br_hit else '✗'

    print(f"  {prompt[:41]:<43}  {expected:<10}  "
          f"{lm_mark}{lm_top1:<9}  {cos_bridge_actual:.4f}           "
          f"{br_mark}{br_top1}")

    b3_results.append({
        'prompt': prompt, 'expected': expected,
        'lm_top1': lm_top1, 'lm_hit': lm_hit,
        'bridge_top1': br_top1, 'bridge_hit': br_hit,
        'cos_bridge_actual': float(cos_bridge_actual),
    })

lm_acc = sum(r['lm_hit']     for r in b3_results) / len(b3_results)
br_acc = sum(r['bridge_hit'] for r in b3_results) / len(b3_results)
cos_mean = np.mean([r['cos_bridge_actual'] for r in b3_results])
print(f"\n  LM-head accuracy:       {lm_acc:.3f}")
print(f"  Bridge accuracy (@5):   {br_acc:.3f}")
print(f"  Mean cos(bridge, actual h_L27):  {cos_mean:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# B4 — Full LCM Pipeline via Bridge
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"B4 — Full LCM Loop:  φ-navigate → h_L14 → bridge → W_out → token")
print(f"{'='*70}\n")
print(f"  The closed LCM loop:")
print(f"  1. Know target concept (e.g. 'Paris')")
print(f"  2. Look up h_L14('Paris') from Zone C cache")
print(f"  3. Apply bridge → h_L27_pred")
print(f"  4. Decode via W_out → predicted token")
print(f"  This tests: can Zone C concept address → output token?\n")

print(f"  {'Target concept':<18}  {'h_L14 → Bridge → W_out top-5'}")
print(f"  {'-'*70}")

LCM_TARGETS = [
    'Paris', 'cold', 'bark', 'ice', 'cats', 'queen', 'queens', 'bigger',
    'quickly', 'puppy', 'green', 'short', 'blue', 'walked', 'dancer',
    'west', 'pack', 'mare', 'girls', 'modern',
]

b4_results = []
for target in LCM_TARGETS:
    found = False
    for pfx in [' ', '']:
        wk = pfx + target.lower()
        if wk in w2i:
            h14_target = hs14_all[w2i[wk]]
            x_b        = np.append(h14_target, 1.0).reshape(1, -1)
            h27_pred   = (x_b @ W)[0]
            top5_ids   = np.argsort(-(h27_pred @ W_out.T))[:5]
            top5_words = [tok.decode([t]).strip() for t in top5_ids]
            hit = target.lower() in [w.lower() for w in top5_words]
            mark = '✓' if hit else '✗'
            print(f"  {target:<18}  {mark} {top5_words}")
            b4_results.append({'target': target, 'top5': top5_words, 'hit': hit})
            found = True
            break
    if not found:
        print(f"  {target:<18}  NOT IN CACHE")

b4_acc = sum(r['hit'] for r in b4_results) / len(b4_results) if b4_results else 0
print(f"\n  B4 LCM accuracy (top-5): {b4_acc:.3f}  ({sum(r['hit'] for r in b4_results)}/{len(b4_results)})")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Day 59: The Bridge")
print(f"{'='*70}")
print(f"""
  B1  Linear bridge  h_L14 → h_L27:
      R² (mean per dim):         {r2_mean:.6f}
      Mean cos(pred, actual):    {np.mean(cos_te):.6f}
      % cos > 0.95:              {np.mean(cos_te > 0.95)*100:.1f}%
      L14→L23 R² (comparison):  {r2_23:.6f}

  B2  Token recovery  bridge → W_out → word:
      Top-1:   {hits_1/n_te:.4f}
      Top-5:   {hits_5/n_te:.4f}
      Top-10:  {hits_10/n_te:.4f}
      Ceiling (actual h_L27):
        Top-1: {hits_1_real/n_te:.4f}   Top-5: {hits_5_real/n_te:.4f}

  B3  Contextual bridge (fill prompts):
      LM-head accuracy:       {lm_acc:.3f}
      Bridge accuracy (@5):   {br_acc:.3f}
      Mean cos(bridge, actual h_L27): {cos_mean:.4f}

  B4  Full LCM loop (concept → token):
      Accuracy (top-5):  {b4_acc:.3f}  ({sum(r['hit'] for r in b4_results)}/{len(b4_results)})

  INTERPRETATION:
    R² close to 1.0:  bridge is linear — the LCM loop closes
    R² < 0.5:         bridge needs a non-linear step
    B4 high (>0.5):   Zone C address → token WORKS — LCM is feasible
    B4 low (<0.3):    need contextual bridge, not just isolated-token bridge
""")

# Save
def to_py(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    return x

output = {
    'b1': {'r2_mean': r2_mean, 'cos_mean': float(np.mean(cos_te)),
            'pct_cos_095': float(np.mean(cos_te > 0.95)),
            'r2_l14_l23': r2_23},
    'b2': {'top1': hits_1/n_te, 'top5': hits_5/n_te, 'top10': hits_10/n_te,
            'ceiling_top1': hits_1_real/n_te, 'ceiling_top5': hits_5_real/n_te},
    'b3': {'lm_acc': lm_acc, 'bridge_acc': br_acc, 'cos_mean': float(cos_mean),
            'results': b3_results},
    'b4': {'accuracy': b4_acc, 'results': b4_results},
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 59 complete.")
