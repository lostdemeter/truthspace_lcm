#!/usr/bin/env python3
"""
Expedition Day 58 — Output Decode: Is Generation = φ-Space Navigation?

The ENCODE=DECODE hypothesis says:
    TEXT IN → φ-space → TEXT OUT

We've proven the ENCODE half: words have stable φ-addresses, T2 operators
are Killing vectors, ENCODE=DECODE is exact in ambient space.

Now the DECODE half: how does a hidden state at the final layer (L27) produce
the next token?

In Qwen2-1.5B:
    logit_t = h_L27 · W_out[t]     (standard LM head)

If ENCODE=DECODE holds for generation:
    argmax_t logit_t  ≈  argmax_t cos(φ(h_L27), φ(W_out[t]))

i.e., the most probable next token is the one whose φ-address is nearest
to the current hidden state's φ-address. Generation = navigation in φ-space.

Tests:
  O1  Embedding geometry: what is the φ-space structure of W_in (token embeddings)?
      Are they in Zone C structure? Do T2 pairs align with known T2 operators?

  O2  Tied weights check: is W_out ≈ W_in.T? (does the model encode and decode
      through the same geometric space?)

  O3  ENCODE=DECODE test for generation:
      For a set of fill-in-the-blank prompts with known correct answers,
      compare: argmax logit vs argmax φ-cosine.
      This is the central test: can φ-space navigation replace the LM head?

  O4  Residual stream evolution: how does the φ-address of the final token
      change layer by layer? Where does the "correct answer" first appear?

  O5  The full LCM pipeline: can we generate a coherent response using ONLY
      φ-space navigation (no LM head, no softmax)?
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day58_output_decode.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# Fill-in-the-blank prompts with unambiguous single-word answers
FILL_TESTS = [
    # (prompt, answer)
    ("The capital of France is",          "Paris"),
    ("The opposite of hot is",             "cold"),
    ("Dogs are known for their ability to", "bark"),
    ("Water freezes and turns into",       "ice"),
    ("A female horse is called a",         "mare"),
    ("The plural of cat is",               "cats"),
    ("She ran faster than everyone else and won the",  "race"),
    ("He is bigger than his brother but smaller than his", "father"),
    ("The sun rises in the east and sets in the",         "west"),
    ("A baby dog is called a",             "puppy"),
    ("The colour of grass is",             "green"),
    ("The opposite of tall is",            "short"),
    ("She is a great singer and he is a great",          "dancer"),
    ("A group of wolves is called a",      "pack"),
    ("The past tense of walk is",          "walked"),
    ("An adult female cat is called a",    "queen"),
    ("The comparative form of big is",     "bigger"),
    ("The adverb form of quick is",        "quickly"),
    ("Boys and",                           "girls"),
    ("Kings and",                          "queens"),
]

print("=" * 70)
print("  Expedition Day 58 — Output Decode")
print("  Is generation φ-space navigation?")
print("=" * 70)


# ── Load baseline data ────────────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

def build_z2(pairs, hs_dict):
    ds = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx+a, pfx+b
            if wa in hs_dict and wb in hs_dict:
                d = hs_dict[wb].astype(np.float64) - hs_dict[wa].astype(np.float64)
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    _, _, Vt = np.linalg.svd(np.stack(ds), full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])

z2 = build_z2(KILLING_PAIRS, {w: hs14_all[w2i[w]] for w in words_all if w in w2i})

def to_phi_v(h, z2):
    h    = h.astype(np.float64)
    hn   = h / (np.linalg.norm(h) + 1e-20)
    perp = hn - np.dot(hn, z2) * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers  = model.config.num_hidden_layers
hidden_sz = model.config.hidden_size
vocab_sz  = model.config.vocab_size
print(f"  Layers: {n_layers}, hidden: {hidden_sz}, vocab: {vocab_sz}")


# ── O1: Embedding geometry ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"O1 — Embedding Geometry: W_in φ-structure")
print(f"{'='*70}")

W_in  = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_out = model.lm_head.weight.detach().numpy().astype(np.float64)

print(f"\n  W_in  shape: {W_in.shape}")
print(f"  W_out shape: {W_out.shape}")

# Check tied weights
tied = np.allclose(W_in, W_out, atol=1e-5)
cos_in_out = float(np.mean([
    cosine(W_in[i], W_out[i]) for i in range(min(1000, vocab_sz))
]))
print(f"\n  Weights tied (W_in == W_out):  {tied}")
print(f"  Mean cos(W_in[t], W_out[t]):   {cos_in_out:.6f}  (1.0 = same direction)")

# Compute φ-vectors for Zone C words using W_in (embedding, not L14)
print(f"\n  Computing Z2 for EMBEDDING space ...")
# Z2 built from L14 hidden states — but we need to project W_in rows to L14-φ-space
# Actually, we can't directly compare W_in dimensions to L14 hidden states
# unless model.embed_tokens has the same dimension as L14 hidden states (it does: both 1536)
# The Z2 axis was built from L14 hidden states — not from embeddings
# Instead, build a separate Z2 from W_in for the Killing pair words

emb_hs_dict = {}
for w in words_all:
    ids = tok.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        emb_hs_dict[w] = W_in[ids[0]]

z2_emb = build_z2(KILLING_PAIRS, emb_hs_dict)
print(f"  Z2 embedding axis built from {len(emb_hs_dict)} words")

# How aligned is Z2_emb with Z2_L14?
cos_z2 = cosine(z2, z2_emb)
print(f"  cos(Z2_L14, Z2_emb) = {cos_z2:.6f}  "
      f"({'ALIGNED' if abs(cos_z2) > 0.7 else 'DIFFERENT'})")

# Test T2 operators in embedding space
print(f"\n  T2 operators in embedding space:")
print(f"  {'Pair':<25}  cos(Δ_emb, Δ_L14)  magnitude")
T2_SEEDS = {
    'male_female':     [(' king',' queen'),(' man',' woman'),(' boy',' girl')],
    'singular_plural': [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees')],
    'base_comp':       [(' big',' bigger'),(' fast',' faster')],
}

def build_t2_emb(seeds):
    ds = []
    for a, b in seeds:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in emb_hs_dict and wb in emb_hs_dict:
                pa = to_phi_v(emb_hs_dict[wa], z2_emb)
                pb = to_phi_v(emb_hs_dict[wb], z2_emb)
                d  = pb - pa; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

phi14_all = np.stack([to_phi_v(hs14_all[i], z2) for i in range(len(words_all))])

def build_t2_l14(seeds):
    ds = []
    for a, b in seeds:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                pa = phi14_all[w2i[wa]]; pb = phi14_all[w2i[wb]]
                d  = pb - pa; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

o1_t2_align = {}
for t2_name, seeds in T2_SEEDS.items():
    t2_emb = build_t2_emb(seeds)
    t2_l14 = build_t2_l14(seeds)
    if t2_emb is None or t2_l14 is None:
        print(f"  {t2_name}: MISSING")
        continue
    c = cosine(t2_emb, t2_l14)
    print(f"  {t2_name:<25}  {c:+.6f}  "
          f"({'ALIGNED' if abs(c) > 0.7 else 'ORTHOGONAL' if abs(c) < 0.3 else 'PARTIAL'})")
    o1_t2_align[t2_name] = float(c)


# ── O2: Tied weights and spectral relationship ────────────────────────────────
print(f"\n{'='*70}")
print(f"O2 — W_in vs W_out: Encode = Decode Matrix Relationship")
print(f"{'='*70}")

# Compute SVD of both to compare spectral structure
print(f"\n  Comparing spectral structure (sample of {min(vocab_sz, 10000)} rows)...")
idx     = np.random.choice(vocab_sz, min(10000, vocab_sz), replace=False)
W_in_s  = W_in[idx]
W_out_s = W_out[idx]

# Normalised rows
W_in_n  = W_in_s  / (np.linalg.norm(W_in_s,  axis=1, keepdims=True) + 1e-20)
W_out_n = W_out_s / (np.linalg.norm(W_out_s, axis=1, keepdims=True) + 1e-20)

# Per-token cosine similarity W_in[t] vs W_out[t]
per_token_cos = np.sum(W_in_n * W_out_n, axis=1)
print(f"  Per-token cos(W_in, W_out):")
print(f"    Mean:    {np.mean(per_token_cos):.6f}")
print(f"    Median:  {np.median(per_token_cos):.6f}")
print(f"    Std:     {np.std(per_token_cos):.6f}")
print(f"    % > 0.9: {np.mean(per_token_cos > 0.9)*100:.1f}%")
print(f"    % > 0.5: {np.mean(per_token_cos > 0.5)*100:.1f}%")
print(f"    % < 0.0: {np.mean(per_token_cos < 0.0)*100:.1f}%")

# Frobenius norm ratio
W_in_frob  = float(np.linalg.norm(W_in))
W_out_frob = float(np.linalg.norm(W_out))
print(f"\n  Frobenius norms: ||W_in||={W_in_frob:.2f}  ||W_out||={W_out_frob:.2f}")
print(f"  Ratio: {W_out_frob/W_in_frob:.6f}  (1.0 = same scale)")


# ── O3: ENCODE=DECODE test for generation ────────────────────────────────────
print(f"\n{'='*70}")
print(f"O3 — ENCODE=DECODE for Generation")
print(f"  For each prompt, compare:")
print(f"  A) LM head: argmax(h_L27 · W_out.T)")
print(f"  B) φ-cosine: argmax cos(φ(h_L27), φ(W_out[t]))")
print(f"  C) φ-cosine with Z2_emb: using embedding-space Z2")
print(f"{'='*70}\n")

# Pre-compute φ-vectors for W_out rows (only for Zone C vocabulary words to save time)
# Map token_id → φ-vector for all Zone C tokens
print(f"  Pre-computing φ-vectors for vocabulary tokens ...")
# Build vocab of single-token Zone C words
zonec_tok_phi = {}    # token_id → (word, phi_L14)
zonec_tok_phi_emb = {}  # token_id → (word, phi_emb)

for w in words_all:
    # check if this word is single-token
    for pfx in [' ', '']:
        wk = pfx + w.lstrip()
        ids_w = tok.encode(wk, add_special_tokens=False)
        if len(ids_w) == 1:
            tid = ids_w[0]
            # φ from L14
            phi_l14 = phi14_all[w2i[wk]] if wk in w2i else None
            if phi_l14 is not None:
                zonec_tok_phi[tid] = (wk, phi_l14)
            # φ from embedding
            emb_vec = W_in[tid]
            phi_emb_v = to_phi_v(emb_vec, z2_emb)
            zonec_tok_phi_emb[tid] = (wk, phi_emb_v)
            break

print(f"  Zone C tokens mapped: {len(zonec_tok_phi)} (L14 φ), {len(zonec_tok_phi_emb)} (emb φ)")

# Vectorise for fast similarity computation
zc_tids  = np.array(list(zonec_tok_phi.keys()))
zc_phi14 = np.stack([zonec_tok_phi[t][1] for t in zc_tids])

zc_tids_e  = np.array(list(zonec_tok_phi_emb.keys()))
zc_phi_emb = np.stack([zonec_tok_phi_emb[t][1] for t in zc_tids_e])

print(f"\n  {'Prompt':<45}  {'Expected':<12}  {'LM-head':<12}  {'φ@5?':<6}  {'φ-top1'}")
print(f"  {'-'*90}")

o3_results = []
for prompt, expected in FILL_TESTS:
    inputs = tok(prompt, return_tensors='pt')

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # LM head top-1
    logits   = out.logits[0, -1, :].numpy()
    lm_top1  = tok.decode([np.argmax(logits)]).strip()
    lm_top10 = [tok.decode([i]).strip() for i in np.argsort(-logits)[:10]]

    # φ-cosine search (L14 φ of the last token's L14 hidden state)
    h_last_14 = out.hidden_states[14][0, -1, :].numpy().astype(np.float64)
    phi_last  = to_phi_v(h_last_14, z2)
    sims_phi  = zc_phi14 @ phi_last
    phi_top5_idx = np.argsort(-sims_phi)[:5]
    phi_top1_word = zonec_tok_phi[zc_tids[phi_top5_idx[0]]][0].strip()
    phi_top5_words = [zonec_tok_phi[zc_tids[i]][0].strip() for i in phi_top5_idx]

    # φ-cosine using FINAL layer (L27) hidden state
    h_last_27 = out.hidden_states[-1][0, -1, :].numpy().astype(np.float64)
    phi_last27 = to_phi_v(h_last_27, z2)
    sims27     = zc_phi14 @ phi_last27
    phi27_top1 = zonec_tok_phi[zc_tids[np.argmax(sims27)]][0].strip()

    exp_lower = expected.lower()
    lm_hit    = exp_lower in [w.lower() for w in lm_top10[:1]]
    phi_hit   = exp_lower in [w.lower() for w in phi_top5_words]

    lm_mark  = '✓' if lm_hit  else '✗'
    phi_mark = '✓' if phi_hit else '✗'

    prompt_short = prompt[:43]
    print(f"  {prompt_short:<45}  {expected:<12}  "
          f"{lm_mark}{lm_top1:<11}  {phi_mark}{phi_hit!s:<5}  {phi_top1_word}")

    o3_results.append({
        'prompt': prompt, 'expected': expected,
        'lm_top1': lm_top1, 'lm_hit': lm_hit,
        'phi14_top1': phi_top1_word, 'phi14_hit': phi_hit,
        'phi27_top1': phi27_top1,
        'phi14_top5': phi_top5_words,
    })

lm_acc   = sum(r['lm_hit']   for r in o3_results) / len(o3_results)
phi14_acc = sum(r['phi14_hit'] for r in o3_results) / len(o3_results)
phi27_acc = sum(r.get('phi27_top1','').lower() == r['expected'].lower() for r in o3_results) / len(o3_results)

print(f"\n  LM-head   accuracy (top-1): {lm_acc:.3f}")
print(f"  φ-L14     accuracy (top-5): {phi14_acc:.3f}")
print(f"  φ-L27     accuracy (top-1): {phi27_acc:.3f}")


# ── O4: Residual stream evolution — at which layer does the answer "appear"? ──
print(f"\n{'='*70}")
print(f"O4 — Residual Stream Evolution")
print(f"  At which layer does the correct answer first appear in φ-space?")
print(f"{'='*70}\n")

TRACK_PROMPTS = FILL_TESTS[:6]   # test first 6
print(f"  {'Prompt':<40}  L14φ  L20φ  L24φ  L27φ  correct")
print(f"  {'-'*75}")
o4_results = []
for prompt, expected in TRACK_PROMPTS:
    inputs = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    layer_hits = {}
    for L in [7, 10, 14, 17, 20, 24, 27]:
        h_L   = out.hidden_states[L][0, -1, :].numpy().astype(np.float64)
        phi_L = to_phi_v(h_L, z2)
        sims  = zc_phi14 @ phi_L
        top1  = zonec_tok_phi[zc_tids[np.argmax(sims)]][0].strip()
        layer_hits[L] = top1

    exp = expected.lower()
    marks = {L: '✓' if w.lower()==exp else '·' for L, w in layer_hits.items()}
    print(f"  {prompt[:38]:<40}  "
          f"{marks[14]}{layer_hits[14][:4]:<5}  "
          f"{marks[20]}{layer_hits[20][:4]:<5}  "
          f"{marks[24]}{layer_hits[24][:4]:<5}  "
          f"{marks[27]}{layer_hits[27][:4]:<5}  {expected}")
    o4_results.append({'prompt': prompt, 'expected': expected, 'by_layer': layer_hits})


# ── O5: φ-LCM pipeline ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"O5 — φ-LCM Generation Pipeline")
print(f"  Generate a token sequence using ONLY φ-space navigation:")
print(f"  h_L14 → φ-nearest-neighbour → next token (no LM head)")
print(f"{'='*70}\n")

GEN_PROMPTS = [
    "The cat sat on the",
    "The capital of France is",
    "The opposite of hot is",
    "She is a great singer and he is a great",
    "Water freezes and turns into",
]

print(f"  Generating up to 5 tokens per prompt using φ-navigation (L14):")
print(f"  (Greedy: pick highest-cosine Zone C token each step)\n")
o5_results = []
for prompt in GEN_PROMPTS:
    current = prompt
    generated = []

    for step in range(5):
        inputs = tok(current, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        h_L14  = out.hidden_states[14][0, -1, :].numpy().astype(np.float64)
        phi_h  = to_phi_v(h_L14, z2)
        sims   = zc_phi14 @ phi_h
        best_i = np.argmax(sims)
        next_w = zonec_tok_phi[zc_tids[best_i]][0]
        generated.append(next_w.strip())
        current = current + next_w

    lm_gen = []
    current_lm = prompt
    for step in range(5):
        inputs = tok(current_lm, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        next_tid = int(torch.argmax(logits).item())
        next_w_lm = tok.decode([next_tid])
        lm_gen.append(next_w_lm.strip())
        current_lm = current_lm + next_w_lm

    print(f"  Prompt:    \"{prompt}\"")
    print(f"  φ-LCM:     {' '.join(generated)}")
    print(f"  LM-head:   {' '.join(lm_gen)}")
    print()
    o5_results.append({'prompt': prompt, 'phi_gen': generated, 'lm_gen': lm_gen})


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — Day 58")
print(f"{'='*70}")
print(f"""
  O1  Embedding geometry:
      cos(Z2_L14, Z2_emb)     = {cos_z2:.6f}
      Weights tied (W_in=W_out)? {tied}
      Mean cos(W_in[t],W_out[t])= {cos_in_out:.6f}
      T2 alignment (emb vs L14): {o1_t2_align}

  O2  W_in vs W_out:
      Mean per-token cos = {np.mean(per_token_cos):.6f}
      % > 0.9:  {np.mean(per_token_cos > 0.9)*100:.1f}%
      % > 0.5:  {np.mean(per_token_cos > 0.5)*100:.1f}%

  O3  Generation accuracy:
      LM-head @1:  {lm_acc:.3f}
      φ-L14   @5:  {phi14_acc:.3f}
      φ-L27   @1:  {phi27_acc:.3f}

  VERDICT:
      If φ-L14 @5 ≥ 0.5: φ-space navigation predicts next token at
                           moderate accuracy — generation IS φ-navigation
      If φ-L14 @5 < 0.3: φ-space does not directly predict generation
                           — an additional decode step is needed
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
    'o1_embedding_geometry': {
        'z2_l14_vs_z2_emb': float(cos_z2),
        'weights_tied': bool(tied),
        'mean_cos_in_out': float(cos_in_out),
        't2_alignment': o1_t2_align,
    },
    'o2_spectral': {
        'mean_per_token_cos': float(np.mean(per_token_cos)),
        'pct_gt_0p9': float(np.mean(per_token_cos > 0.9)),
        'pct_gt_0p5': float(np.mean(per_token_cos > 0.5)),
    },
    'o3_generation': {'results': o3_results, 'lm_acc': lm_acc, 'phi14_acc': phi14_acc},
    'o4_layer_evolution': o4_results,
    'o5_phi_lcm': o5_results,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 58 complete.")
