#!/usr/bin/env python3
"""
Day 68 — T2 Sign Flip and Negative Zero

Hypothesis: The cascade geodesic crosses a sign boundary in T2-projection space.
The conformal crossing ("共" at α=300, cos≈0.09) is the equator — but the
SIGN FLIP (where T2-projection passes through zero) occurs somewhere in the
dense gap α=220→300. Everything on the far side has NEGATIVE T2 projection.

"Negative zero" = the token at the sign-flip crossing point. Same magnitude as
the positive crossing, but arrived from the negative side — IEEE 754 -0.0 vs +0.0.

Measurements:
  S1: For every rank-0 token at each α, measure:
        proj_T2 = dot(h_token, T2)        # T2-axis projection
        cos_T2  = proj_T2 / (||h|| * ||T2||)  # cosine in T2 direction only
      Track the sign of this projection across the sweep.

  S2: Dense sweep α=200–400 (step=10) to locate the exact sign-flip point.

  S3: For the top-15 tokens at each α, compute their T2 projections.
      Visualise the distribution: does the entire top-15 cluster cross together,
      or do individual tokens flip one by one (cascade)?

  S4: Measure T2 projections of key vocabulary:
        dogs, cats, tech, 共, not, apparently, ,, than, slow, faster
      Find which ones have positive vs negative T2 projection at baseline.
      This maps the T2 axis in vocabulary space.

  S5: The "negative zero" token — find the token with T2 projection closest
      to zero (from below, i.e. proj_T2 just < 0). This is the semantic equator.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day68_sign_flip.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

CTX_PAIRS_COMP = [
    ("The fast car won the race",    "The faster car won the race"),
    ("The big dog barked loudly",    "The bigger dog barked loudly"),
    ("A small bird sang at dawn",    "A smaller bird sang at dawn"),
    ("The tall tree swayed gently",  "The taller tree swayed gently"),
    ("A cold wind swept the plain",  "A colder wind swept the plain"),
    ("The old house still stands",   "The older house still stands"),
    ("A young child played outside", "A younger child played outside"),
    ("The strong man lifted it",     "The stronger man lifted it"),
]

# Dense sweep to find the sign flip, plus extended range to see full geodesic
ALPHA_COARSE = [0, 10, 20, 30, 40, 50, 60, 75, 100, 130, 170, 220]
ALPHA_DENSE  = list(range(220, 410, 10))   # dense around conformal crossing
ALPHA_FAR    = [450, 500, 600, 700, 800, 1000, 1500, 2000]
ALL_ALPHA    = ALPHA_COARSE + ALPHA_DENSE + ALPHA_FAR

# Tokens to map on T2 axis
VOCAB_PROBE = [
    "dogs", "cats", "not", "apparently", "tech", "than",
    "slow", "fast", "faster", "slower", "bigger", "big",
    "smaller", "small", "quickly", "quickly",
    "positive", "negative", "zero",
]

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers   = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"  n_layers={n_layers}  hidden={hidden_dim}\n")

def get_hs_last(text, layer):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy().astype(np.float32)

def get_hs_word(word, layer):
    return get_hs_last(" " + word.strip(), layer)

def get_logits_steered(prompt, direction_np, alpha, layer):
    inputs = tok(prompt, return_tensors="pt")
    if alpha == 0:
        with torch.no_grad():
            return model(**inputs).logits[0, -1, :].numpy()
    d_t = torch.tensor(direction_np, dtype=torch.float32)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][0, -1, :] += alpha * d_t; return out
        out[0, -1, :] += alpha * d_t; return out
    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].numpy()
    finally:
        h.remove()
    return logits

def build_ctx_t2(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        h1 = get_hs_last(s1, layer); h2 = get_hs_last(s2, layer)
        d = h2 - h1; n = np.linalg.norm(d)
        if n > 1e-6: diffs.append(d / n)
    v = np.mean(diffs, axis=0)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

def t2_proj(h, t2):
    """Signed scalar projection of h onto T2 direction."""
    return float(np.dot(h, t2))  # t2 is already unit vector

def token_rank(logits, word):
    ids = tok.encode(" " + word, add_special_tokens=False) or \
          tok.encode(word, add_special_tokens=False)
    tid = ids[0]
    return int((logits > logits[tid]).sum())

print("Building T2 direction (L14) ...")
T2 = build_ctx_t2(CTX_PAIRS_COMP, 14)
print(f"  ||T2|| = {np.linalg.norm(T2):.4f}  (should be 1.0)\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("S4 — Vocabulary T2 projections (baseline)")
print("=" * 70)
print("  Mapping where key tokens sit on the T2 axis at rest.\n")

vocab_projs = {}
print(f"  {'token':>15}  {'proj_T2':>10}  {'sign':>6}")
print(f"  {'-'*38}")
for word in VOCAB_PROBE:
    try:
        h = get_hs_word(word, 14)
        p = t2_proj(h, T2)
        vocab_projs[word] = p
        sign_str = "+pos" if p > 0 else "-neg"
        print(f"  {word:>15}  {p:>10.2f}  {sign_str}")
    except Exception as e:
        print(f"  {word:>15}  ERROR: {e}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S1/S2 — T2 projection of rank-0 token across full α sweep")
print("=" * 70)
print("  Tracking sign of T2 projection as geodesic traverses the manifold.\n")

prompt    = "The plural of dog is"
target    = "dogs"
best_L    = 14

print(f"  Prompt: '{prompt}' → {target}  (T2 at L{best_L})\n")
print(f"  {'α':>6}  {'rank0_tok':>16}  {'proj_T2':>10}  {'sign':>5}  {'tgt_rank':>9}  {'note'}")
print(f"  {'-'*70}")

sweep_results = []
prev_sign = None
sign_flip_alpha = None

for a in ALL_ALPHA:
    logits    = get_logits_steered(prompt, T2, a, best_L)
    tgt_rank  = token_rank(logits, target)

    # Get rank-0 token and its T2 projection
    top_id    = int(np.argmax(logits))
    top_str   = tok.decode([top_id]).strip()
    h_top     = get_hs_word(top_str if top_str else "the", 14)
    proj      = t2_proj(h_top, T2)
    sign      = "+" if proj >= 0 else "-"

    # Detect sign flip
    note = ""
    if prev_sign is not None and sign != prev_sign:
        note = "<<< SIGN FLIP"
        if sign_flip_alpha is None:
            sign_flip_alpha = a

    print(f"  {a:>6}  {top_str:>16}  {proj:>10.2f}  {sign:>5}  {tgt_rank:>9}  {note}")

    sweep_results.append({
        "alpha": a, "rank0_token": top_str, "rank0_proj_T2": proj,
        "sign": sign, "target_rank": tgt_rank
    })
    prev_sign = sign

print(f"\n  First sign flip detected at: α = {sign_flip_alpha}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S2 — Dense sweep around conformal crossing (α=220–400, step 10)")
print("=" * 70)
print("  Resolving the exact sign-flip α.\n")

print(f"  {'α':>6}  {'rank0_tok':>16}  {'proj_T2':>10}  {'sign':>5}  {'tgt_rank':>9}")
print(f"  {'-'*60}")

# Already covered in ALL_ALPHA (ALPHA_DENSE), extract from sweep_results
dense_steps = [r for r in sweep_results if 210 <= r["alpha"] <= 420]
for step in dense_steps:
    print(f"  {step['alpha']:>6}  {step['rank0_token']:>16}  "
          f"{step['rank0_proj_T2']:>10.2f}  {step['sign']:>5}  "
          f"{step['target_rank']:>9}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S3 — Top-15 T2 projections across the cascade")
print("=" * 70)
print("  Do tokens cross the sign boundary together (cluster flip) or one-by-one?\n")

selected_alphas = [0, 50, 100, 130, 220, 250, 280, 300, 330, 360, 400, 600, 1000]

for a in selected_alphas:
    if a not in [r["alpha"] for r in sweep_results]:
        continue
    logits = get_logits_steered(prompt, T2, a, best_L)
    top15_ids = np.argsort(logits)[::-1][:15]
    projs = []
    toks_and_projs = []
    for tid in top15_ids:
        tok_str = tok.decode([int(tid)]).strip()
        try:
            h = get_hs_word(tok_str if tok_str else "the", 14)
            p = t2_proj(h, T2)
            projs.append(p)
            toks_and_projs.append((tok_str, p))
        except Exception:
            toks_and_projs.append((tok_str, float("nan")))

    n_pos = sum(1 for _, p in toks_and_projs if not np.isnan(p) and p > 0)
    n_neg = sum(1 for _, p in toks_and_projs if not np.isnan(p) and p < 0)
    mean_p = float(np.nanmean([p for _, p in toks_and_projs]))
    top3 = "  ".join(f"[{t[:8]}:{p:+.0f}]" for t, p in toks_and_projs[:3])
    tgt_rank = token_rank(logits, target)
    print(f"  α={a:>5}  pos={n_pos:2d} neg={n_neg:2d}  mean_proj={mean_p:+8.1f}"
          f"  tgt={tgt_rank:>6}  top3: {top3}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("S5 — Finding 'negative zero': token with T2 projection nearest to 0 (from below)")
print("=" * 70)
print("  Sampling 500 most common tokens, finding the semantic equator.\n")

# Get top-500 most common tokens by unigram frequency proxy (logit at α=0)
baseline_logits = get_logits_steered(prompt, T2, 0, best_L)
top500_ids = np.argsort(baseline_logits)[::-1][:500]

equator_tokens = []
for tid in top500_ids:
    tok_str = tok.decode([int(tid)]).strip()
    if not tok_str:
        continue
    try:
        h = get_hs_word(tok_str, 14)
        p = t2_proj(h, T2)
        equator_tokens.append((tok_str, p, int(tid)))
    except Exception:
        pass

equator_tokens.sort(key=lambda x: abs(x[1]))

print(f"  Tokens with T2 projection nearest to zero (equator residents):\n")
print(f"  {'token':>18}  {'proj_T2':>10}  {'side':>8}")
print(f"  {'-'*42}")
for tok_str, p, tid in equator_tokens[:20]:
    side = "+pos" if p >= 0 else "-neg (neg-zero side)"
    print(f"  {tok_str:>18}  {p:>10.3f}  {side}")

# First token with negative T2 projection from this list
neg_zero_tok = next((t for t, p, _ in equator_tokens if p < 0), None)
pos_zero_tok = next((t for t, p, _ in equator_tokens if p >= 0), None)
print(f"\n  'Positive zero' (nearest to 0 from above): {pos_zero_tok}")
print(f"  'Negative zero' (nearest to 0 from below): {neg_zero_tok}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SYNTHESIS — Sign Flip and Semantic Topology")
print("=" * 70)

# Classify vocabulary
pos_tokens = [(t, p) for t, p in vocab_projs.items() if p > 0]
neg_tokens = [(t, p) for t, p in vocab_projs.items() if p < 0]
pos_tokens.sort(key=lambda x: -x[1])
neg_tokens.sort(key=lambda x: x[1])

print(f"\n  T2-positive vocabulary (same side as comparative direction):")
for t, p in pos_tokens:
    print(f"    {t:>15}  proj={p:+.1f}")
print(f"\n  T2-negative vocabulary (anti-comparative direction):")
for t, p in neg_tokens:
    print(f"    {t:>15}  proj={p:+.1f}")

print(f"""
  Sign flip α: {sign_flip_alpha}
  Interpretation:
    Before sign flip: rank-0 tokens live in T2-positive space
    After sign flip:  rank-0 tokens live in T2-negative space
    "Negative zero":  the crossing point — a token with proj_T2 ≈ -0

  IEEE 754 analogy:
    +0.0 = equator approached from the positive T2 side
    -0.0 = equator approached from the negative T2 side
    They are the same semantic location but different origins.
    The cascade carries the sign information even at magnitude ≈ 0.
""")

# Save
results = {
    "sweep": sweep_results,
    "sign_flip_alpha": sign_flip_alpha,
    "vocab_projs": vocab_projs,
    "equator_tokens": [(t, p) for t, p, _ in equator_tokens[:30]],
    "neg_zero_token": neg_zero_tok,
    "pos_zero_token": pos_zero_tok,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: {OUTPUT_FILE}")
print("Day 68 complete.")
