#!/usr/bin/env python3
"""
Expedition Day 60 — Anticipation Test + Contextual Bridge

Day 59 proved that the bridge approach using isolated-token h_L27 is wrong.
The conceptual error: W_out @ h_L27(isolated w) = "what comes AFTER w", not w.

The revised question: does contextual h_L14 (at the last prompt token)
ANTICIPATE the answer in Zone C φ-space?

If "The capital of France is" → h_L14(last) is near φ("Paris") in Zone C,
then Zone C is the ANTICIPATION SPACE. The model assembles its answer
geometrically at L14 before the output layers (L15-L27) decode it.

Tests:
  A1  Anticipation test:
      cos(ctx_h_L14_last, phi14(expected_answer))  vs  random baseline
      Per-layer sweep: at which layer does anticipation peak?

  A2  Contextual bridge:
      From many (prompt, answer) pairs learn M_ctx: ctx_h_L14 → ctx_h_L27
      Where ctx_h is the LAST POSITION hidden state in the full sequence.
      Accuracy: does M_ctx + W_out give correct answers?

  A3  T2 steering:
      Apply T2 operator to ctx_h_L14 mid-inference.
      Run the prompt "The king sat on the throne" → steer h_L14 by
      Δ_male_female → does the output shift toward "queen" responses?
      This is the most direct test of Zone C's role in LCM steering.
"""

import json, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
L27_CACHE   = str(SCRIPT_DIR / "day59_hs_27_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day60_anticipation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# --- Fill prompts for A1 anticipation test ---
# Format: (prompt, expected_answer)
FILL_PROMPTS = [
    ("The capital of France is",              "Paris"),
    ("The capital of Germany is",             "Berlin"),
    ("The capital of Japan is",               "Tokyo"),
    ("The capital of Italy is",               "Rome"),
    ("The capital of Spain is",               "Madrid"),
    ("The opposite of hot is",                "cold"),
    ("The opposite of tall is",               "short"),
    ("The opposite of fast is",               "slow"),
    ("The opposite of old is",                "young"),
    ("The opposite of dark is",               "light"),
    ("The plural of cat is",                  "cats"),
    ("The plural of dog is",                  "dogs"),
    ("The plural of tree is",                 "trees"),
    ("The plural of bird is",                 "birds"),
    ("The plural of house is",                "houses"),
    ("The male version of queen is",          "king"),
    ("The female version of king is",         "queen"),
    ("The female version of boy is",          "girl"),
    ("The female version of man is",          "woman"),
    ("Water freezes and turns into",          "ice"),
    ("A baby dog is called a",                "puppy"),
    ("A female horse is called a",            "mare"),
    ("A group of wolves is called a",         "pack"),
    ("The colour of grass is",                "green"),
    ("The colour of the sky is",              "blue"),
    ("The past tense of walk is",             "walked"),
    ("The past tense of jump is",             "jumped"),
    ("The comparative of big is",             "bigger"),
    ("The comparative of fast is",            "faster"),
    ("The adverb form of quick is",           "quickly"),
    ("She is a great singer and he is a great", "dancer"),
    ("The sun rises in the east and sets in the", "west"),
    ("Kings and",                             "queens"),
    ("Boys and",                              "girls"),
    ("Cats and",                              "dogs"),
    ("Day and",                               "night"),
    ("Black and",                             "white"),
    ("Hot and",                               "cold"),
    ("Left and",                              "right"),
    ("Up and",                                "down"),
]

# --- T2 steering test prompts ---
STEERING_TESTS = [
    {
        'base_prompt':   "The king ruled his",
        'steered_toward': 'queen',
        't2': 'male_female',
        'expected_shift': "The queen ruled her",
    },
    {
        'base_prompt':   "The boy played",
        'steered_toward': 'girl',
        't2': 'male_female',
        'expected_shift': "The girl played",
    },
    {
        'base_prompt':   "One cat sat on the",
        'steered_toward': 'cats',
        't2': 'singular_plural',
        'expected_shift': "Two cats sat on the",
    },
    {
        'base_prompt':   "She walks to",
        'steered_toward': 'walked',
        't2': 'base_past',
        'expected_shift': "She walked to",
    },
]

T2_SEEDS = {
    'male_female':     [(' king',' queen'),(' man',' woman'),(' boy',' girl'),
                        (' actor',' actress'),(' uncle',' aunt')],
    'singular_plural': [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees'),
                        (' bird',' birds'),(' house',' houses')],
    'base_comp':       [(' big',' bigger'),(' fast',' faster'),(' old',' older'),
                        (' tall',' taller'),(' small',' smaller')],
}

print("=" * 70)
print("  Expedition Day 60 — Anticipation Test + Contextual Bridge")
print("=" * 70)


# ── Load caches ──────────────────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}
N         = len(words_all)

c27       = np.load(L27_CACHE)
hs27_all  = c27['hs_27'].astype(np.float64)

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Build φ-space
def build_z2(pairs, hs_dict):
    ds = []
    for a, b in pairs:
        for pfx in [' ','']:
            wa, wb = pfx+a, pfx+b
            if wa in hs_dict and wb in hs_dict:
                d = hs_dict[wb]-hs_dict[wa]; nm=np.linalg.norm(d)
                if nm>1e-20: ds.append(d/nm)
                break
    _,_,Vt = np.linalg.svd(np.stack(ds),full_matrices=False)
    return Vt[0]/np.linalg.norm(Vt[0])

z2 = build_z2(KILLING_PAIRS, {w: hs14_all[w2i[w]] for w in words_all if w in w2i})

def to_phi(h, z2):
    h = h.astype(np.float64)
    hn = h/(np.linalg.norm(h)+1e-20)
    p  = hn - np.dot(hn,z2)*z2; pm=np.linalg.norm(p)
    return p/(pm+1e-20)

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20))

phi14_all = np.stack([to_phi(hs14_all[i],z2) for i in range(N)])

def phi14_of(word):
    for pfx in [' ','']:
        wk = pfx + word.strip()
        if wk in w2i: return phi14_all[w2i[wk]]
    return None


# ── Load model ───────────────────────────────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_out = model.lm_head.weight.detach().numpy().astype(np.float64)
print(f"  Loaded. Layers={model.config.num_hidden_layers}, H={model.config.hidden_size}")


def run_all_layers(prompt):
    """Return all hidden states [L+1, H] at the last token position."""
    inputs = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    n = inputs['input_ids'].shape[1]
    last = n - 1
    hs = np.stack([out.hidden_states[L][0,last,:].numpy().astype(np.float64)
                   for L in range(len(out.hidden_states))])   # [29, H]
    logits = out.logits[0,-1,:].numpy()
    return hs, logits


# ═══════════════════════════════════════════════════════════════════════════════
# A1 — Anticipation Test
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"A1 — Anticipation Test: does ctx h_L (last) anticipate answer in Zone C?")
print(f"{'='*70}\n")
print(f"  For each fill prompt, measure cos(ctx_h_L_last, phi14(answer))")
print(f"  Layer sweep: L0 through L27. Where does anticipation PEAK?\n")

# Layer-by-layer anticipation: mean cos(h_L, phi14(answer)) vs random baseline
n_layers = model.config.num_hidden_layers  # 28
layer_anticipation = np.zeros((len(FILL_PROMPTS), n_layers+1))
layer_random       = np.zeros((len(FILL_PROMPTS), n_layers+1))
lm_correct         = []

# Random baseline: mean cos across 200 random Zone C words
rng     = np.random.default_rng(0)
rnd_idx = rng.integers(0, N, 200)
rnd_phi = phi14_all[rnd_idx]   # [200, H]

print(f"  {'Prompt':<45}  {'Exp':<10}  L14-cos  L23-cos  LM-top1")
print(f"  {'-'*80}")

a1_rows = []
for pi, (prompt, expected) in enumerate(FILL_PROMPTS):
    hs_all, logits = run_all_layers(prompt)   # [29, H]

    phi_ans = phi14_of(expected)
    if phi_ans is None:
        continue   # expected word not in cache

    for L in range(n_layers+1):
        phi_L = to_phi(hs_all[L], z2)
        layer_anticipation[pi, L] = cosine(phi_L, phi_ans)
        layer_random[pi, L]       = float(np.mean(rnd_phi @ phi_L))

    lm_top1 = tok.decode([np.argmax(logits)]).strip()
    lm_hit  = lm_top1.lower() == expected.lower()
    lm_correct.append(lm_hit)

    print(f"  {prompt[:43]:<45}  {expected:<10}  "
          f"{layer_anticipation[pi,14]:.4f}   {layer_anticipation[pi,23]:.4f}   "
          f"{'✓' if lm_hit else '✗'}{lm_top1}")

    a1_rows.append({'prompt': prompt, 'expected': expected,
                    'lm_top1': lm_top1, 'lm_hit': lm_hit,
                    'by_layer': layer_anticipation[pi].tolist()})

# Aggregate
valid = [i for i,r in enumerate(a1_rows) if r is not None]
ant   = layer_anticipation[:len(a1_rows)]   # [valid, L+1]
rnd   = layer_random[:len(a1_rows)]

mean_ant = ant.mean(0)  # [L+1]
mean_rnd = rnd.mean(0)

peak_L  = int(np.argmax(mean_ant))
gap     = mean_ant - mean_rnd   # signal above random

print(f"\n  Per-layer anticipation (mean cos across {len(a1_rows)} prompts):")
print(f"  Layer  mean_cos(answer)  mean_cos(random)  gap(signal)")
for L in [0,5,7,10,12,14,17,20,23,24,27]:
    if L <= n_layers:
        marker = '  ← PEAK' if L == peak_L else ''
        print(f"    L{L:<3}  {mean_ant[L]:.6f}         {mean_rnd[L]:.6f}        "
              f"{gap[L]:+.6f}{marker}")

print(f"\n  PEAK anticipation layer: L{peak_L}  (mean cos = {mean_ant[peak_L]:.6f})")
print(f"  L14 anticipation cos:    {mean_ant[14]:.6f}  (random: {mean_rnd[14]:.6f})")
print(f"  LM-head accuracy:        {sum(lm_correct)/len(lm_correct):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A2 — Contextual Bridge M_ctx: ctx_h_L14 → ctx_h_L27
# ═══════════════════════════════════════════════════════════════════════════════
BRIDGE_SRC = 14   # always use L14 as source for the LCM bridge
print(f"\n{'='*70}")
print(f"A2 — Contextual Bridge:  ctx_h_L{BRIDGE_SRC} → ctx_h_L27")
print(f"{'='*70}\n")

# Build a larger corpus of contextual (h_L_src, h_L27) pairs
# using the Zone C vocabulary to generate prompts
PROMPT_TEMPLATES = [
    "The plural of {w} is",
    "A single {w} or multiple",
    "The opposite of {w} is",
    "She wanted a {w} or perhaps a",
    "It looked like a {w} and also a",
    "He said the word {w} then continued with",
    "The {w} was beautiful and so was the",
    "First a {w} and then a",
]

# Sample Zone C words from atlas
zonec_words = []
for w, meta in wmap.items():
    b = meta.get('L14_body','')
    if b and b not in ('B000','B001') and w in w2i:
        zonec_words.append(w.strip())
zonec_words = list(set(zonec_words))[:400]

print(f"  Building contextual corpus from {len(PROMPT_TEMPLATES)} templates × {len(zonec_words)} words...")
print(f"  Plus {len(FILL_PROMPTS)} fill prompts for validation.\n")

ctx_h_src  = []   # [N_ctx, H]
ctx_h_L27  = []   # [N_ctx, H]
ctx_labels = []   # for debugging

# Process fill prompts (known answers)
for prompt, expected in FILL_PROMPTS:
    hs_all, _ = run_all_layers(prompt)
    ctx_h_src.append(hs_all[14])
    ctx_h_L27.append(hs_all[27])
    ctx_labels.append(('fill', expected))

# Process Zone C template prompts
n_template = 0
t0 = time.time()
for wi, word in enumerate(zonec_words):
    tmpl = PROMPT_TEMPLATES[wi % len(PROMPT_TEMPLATES)]
    prompt = tmpl.format(w=word)
    hs_all, _ = run_all_layers(prompt)
    ctx_h_src.append(hs_all[14])
    ctx_h_L27.append(hs_all[27])
    ctx_labels.append(('template', word))
    n_template += 1
    if (wi+1) % 100 == 0:
        elapsed = (time.time()-t0)/60
        eta     = elapsed/(wi+1)*(len(zonec_words)-(wi+1))
        print(f"  [{wi+1}/{len(zonec_words)}] templates  {elapsed:.1f} min  ETA {eta:.1f} min")

ctx_h_src = np.stack(ctx_h_src)   # [N_ctx, H]
ctx_h_L27 = np.stack(ctx_h_L27)   # [N_ctx, H]
N_ctx      = len(ctx_h_src)
print(f"\n  Context corpus built: {N_ctx} samples")

# Train contextual bridge (80/20 split)
rng2   = np.random.default_rng(42)
idx    = rng2.permutation(N_ctx)
n_tr   = int(0.8 * N_ctx)
tr_idx = idx[:n_tr]; te_idx = idx[n_tr:]

X_tr = ctx_h_src[tr_idx]; Y_tr = ctx_h_L27[tr_idx]
X_te = ctx_h_src[te_idx]; Y_te = ctx_h_L27[te_idx]

from numpy.linalg import solve
lambda_reg = 1e-3
X_tr_b = np.hstack([X_tr, np.ones((len(X_tr),1))])
X_te_b = np.hstack([X_te, np.ones((len(X_te),1))])
A  = X_tr_b.T @ X_tr_b + lambda_reg * np.eye(X_tr_b.shape[1])
B  = X_tr_b.T @ Y_tr
W_ctx = solve(A, B)   # [H+1, H]

Y_te_pred = X_te_b @ W_ctx
cos_ctx = np.array([cosine(Y_te_pred[i], Y_te[i]) for i in range(len(te_idx))])

ss_res = np.sum((Y_te - Y_te_pred)**2, axis=0)
ss_tot = np.sum((Y_te - Y_te.mean(0))**2, axis=0)
r2_ctx = float(np.mean(1 - ss_res / (ss_tot + 1e-20)))

print(f"\n  Contextual bridge quality:")
print(f"    R² (mean per dim):        {r2_ctx:.6f}")
print(f"    Mean cos(pred, actual):   {np.mean(cos_ctx):.6f}")
print(f"    % cos > 0.95:             {np.mean(cos_ctx > 0.95)*100:.1f}%")
print(f"    % cos > 0.90:             {np.mean(cos_ctx > 0.90)*100:.1f}%")

# Token recovery via contextual bridge
hits_1=0; hits_5=0
for i in range(len(te_idx)):
    logits   = Y_te_pred[i] @ W_out.T
    top5     = [tok.decode([t]).strip().lower() for t in np.argsort(-logits)[:5]]
    label    = ctx_labels[te_idx[i]][1]
    if label.lower() == top5[0]: hits_1 += 1
    if label.lower() in top5:    hits_5 += 1

n_te = len(te_idx)
print(f"\n  Token recovery via ctx bridge → W_out:")
print(f"    Top-1: {hits_1}/{n_te} = {hits_1/n_te:.4f}")
print(f"    Top-5: {hits_5}/{n_te} = {hits_5/n_te:.4f}")


# Full accuracy test on fill prompts via contextual bridge
print(f"\n  Fill-prompt accuracy via contextual bridge:")
print(f"  {'Prompt':<43}  {'Exp':<10}  LM   Bridge-top1")
print(f"  {'-'*75}")
ctx_bridge_hits = 0
for prompt, expected in FILL_PROMPTS[:20]:
    hs_all, logits = run_all_layers(prompt)
    h_src = hs_all[14]
    x_b   = np.append(h_src, 1.0).reshape(1,-1)
    h27p  = (x_b @ W_ctx)[0]

    lm_top1 = tok.decode([np.argmax(logits)]).strip()
    br_top5_ids = np.argsort(-(h27p @ W_out.T))[:5]
    br_top5 = [tok.decode([t]).strip().lower() for t in br_top5_ids]
    br_top1 = tok.decode([br_top5_ids[0]]).strip()
    hit = expected.lower() in br_top5

    lm_hit = lm_top1.lower() == expected.lower()
    if hit: ctx_bridge_hits += 1
    print(f"  {prompt[:41]:<43}  {expected:<10}  "
          f"{'✓' if lm_hit else '✗'}    {'✓' if hit else '✗'}{br_top1}")

ctx_bridge_acc = ctx_bridge_hits / min(20, len(FILL_PROMPTS))
print(f"\n  Contextual bridge accuracy (top-5): {ctx_bridge_acc:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A3 — T2 Steering Test
# ═══════════════════════════════════════════════════════════════════════════════
STEER_LAYER = 14   # steer at Zone C layer (T2 operators live here)
print(f"\n{'='*70}")
print(f"A3 — T2 Steering: apply T2 operator to ctx h_L{STEER_LAYER}, observe output shift")
print(f"{'='*70}\n")
print(f"  Can we steer the model output by patching h_L{STEER_LAYER} with a T2 vector?")
print(f"  Method: pre-hook on L{STEER_LAYER+1}, replace hidden states with T2-shifted version.\n")

# Build T2 operators from isolated tokens
def build_t2_phi(seeds):
    ds = []
    for a, b in seeds:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                pa = phi14_all[w2i[wa]]; pb = phi14_all[w2i[wb]]
                d  = pb - pa; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d/nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return m/(nm+1e-20)

# Build T2 vectors in HIDDEN SPACE (not φ-space) for patching
def build_t2_hidden(seeds):
    ds = []
    for a, b in seeds:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                ha = hs14_all[w2i[wa]]; hb = hs14_all[w2i[wb]]
                d  = hb - ha; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d/nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return torch.tensor(m/(nm+1e-20), dtype=torch.float32)

t2_vectors = {name: build_t2_hidden(seeds) for name, seeds in T2_SEEDS.items()}
t2_norms   = {name: float(v.norm().item()) for name, v in t2_vectors.items() if v is not None}
print(f"  T2 vectors built:")
for name, v in t2_vectors.items():
    if v is not None:
        print(f"    {name}: ‖t2‖ = {float(v.norm()):.4f}")

# Steering via forward hook
def steer_and_run(prompt, t2_vec, alpha=5.0, layer=STEER_LAYER):
    """
    Run prompt through model. At `layer` boundary (pre-hook on layer+1),
    add alpha * t2_vec to the last-position residual stream.
    """
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)

    with torch.no_grad():
        out_orig = model(**inputs, use_cache=False)
    logits_orig = out_orig.logits[0, last_pos, :].detach().numpy()

    patch = (float(alpha) * t2_vec).float()

    # Pre-hook on layer (layer+1): its INPUT is the output of `layer`
    next_layer_idx = min(layer + 1, model.config.num_hidden_layers - 1)

    def pre_hook_fn(module, args):
        # args[0] = hidden_states [B, seq, H] or [seq, H]
        if not isinstance(args, tuple) or len(args) == 0:
            return
        h_in = args[0]
        h = h_in.clone().float()
        p  = patch.to(h.device)
        if isinstance(h_in, tuple):  # named tuple
            h = h_in._replace(hidden_states=h)
        else:
            if h.dim() == 3:
                h[0, last_pos] = h[0, last_pos] + p
            elif h.dim() == 2:
                h[last_pos] = h[last_pos] + p
            else:
                return   # unexpected, leave unchanged
        return (h.to(h_in.dtype),) + args[1:]

    handle = model.model.layers[next_layer_idx].register_forward_pre_hook(pre_hook_fn)
    with torch.no_grad():
        out_steered = model(**inputs, use_cache=False)
    handle.remove()
    logits_steered = out_steered.logits[0, last_pos, :].detach().numpy()
    return logits_orig, logits_steered

print(f"  Steering experiments (α = 5.0, patching at L{STEER_LAYER} → pre-hook on L{STEER_LAYER+1}):\n")
a3_results = []
for test in STEERING_TESTS:
    prompt  = test['base_prompt']
    t2_name = test['t2']

    t2_vec = t2_vectors.get(t2_name)
    if t2_vec is None:
        print(f"  [{t2_name}] T2 vector not built, skipping")
        continue

    logits_orig, logits_steer = steer_and_run(prompt, t2_vec, alpha=5.0)

    orig_top5  = [tok.decode([i]).strip() for i in np.argsort(-logits_orig)[:5]]
    steer_top5 = [tok.decode([i]).strip() for i in np.argsort(-logits_steer)[:5]]

    target = test['steered_toward']
    orig_hit  = target.lower() in [w.lower() for w in orig_top5]
    steer_hit = target.lower() in [w.lower() for w in steer_top5]

    # Rank shift for target word
    t_ids_orig  = [tok.decode([i]).strip().lower() for i in np.argsort(-logits_orig)]
    t_ids_steer = [tok.decode([i]).strip().lower() for i in np.argsort(-logits_steer)]
    rank_orig   = next((i for i,w in enumerate(t_ids_orig)  if w == target.lower()), 9999)
    rank_steer  = next((i for i,w in enumerate(t_ids_steer) if w == target.lower()), 9999)

    moved = rank_orig - rank_steer   # positive = moved up (toward target)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  T2: {t2_name}  →  target: '{target}'")
    print(f"  Original  top-5: {orig_top5}")
    print(f"  Steered   top-5: {steer_top5}")
    print(f"  Rank of '{target}':  orig={rank_orig}  steered={rank_steer}  "
          f"Δrank={moved:+d}  {'▲ MOVED UP' if moved > 0 else '▼ moved down' if moved < 0 else '='}")
    print()

    a3_results.append({
        'prompt': prompt, 't2': t2_name, 'target': target,
        'orig_top5': orig_top5, 'steered_top5': steer_top5,
        'rank_orig': rank_orig, 'rank_steer': rank_steer, 'delta_rank': moved,
    })

# Sweep alpha to find optimal steering strength
print(f"  Alpha sweep (male_female T2 on 'The king sat on the'):")
print(f"  {'alpha':<8}  top-1 pred  rank('queen')")
sweep_prompt = "The king sat on the"
t2_vec_mf = t2_vectors.get('male_female')
if t2_vec_mf is not None:
    for alpha in [0, 1, 2, 5, 10, 20, 50]:
        _, ls = steer_and_run(sweep_prompt, t2_vec_mf, alpha=float(alpha), layer=STEER_LAYER)
        top1 = tok.decode([np.argmax(ls)]).strip()
        rank = next((i for i,w in enumerate([tok.decode([t]).strip().lower()
                     for t in np.argsort(-ls)]) if w=='queen'), 9999)
        print(f"  α={alpha:<6}   {top1:<12}  rank_queen={rank}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Day 60")
print(f"{'='*70}")

print(f"""
  A1  Anticipation (peak_L={peak_L}, Zone C L14 shown separately):
      Mean cos(ctx_h_L_last, phi14(answer)):  {mean_ant[peak_L]:.6f}
      Mean cos(ctx_h_L_last, random):         {mean_rnd[peak_L]:.6f}
      Signal above random:                    {gap[peak_L]:+.6f}
      L14 anticipation cos:                   {mean_ant[14]:.6f}
      L14 random baseline:                    {mean_rnd[14]:.6f}
      LM-head accuracy:                       {sum(lm_correct)/len(lm_correct):.3f}

      INTERPRETATION:
        gap > 0.05:  Zone C anticipates answer (LCM feasible via steering)
        gap < 0.01:  No anticipation — Zone C reads current token not next

  A2  Contextual bridge L{peak_L}→L27:
      R² (mean per dim):        {r2_ctx:.6f}
      Mean cos(pred, actual):   {np.mean(cos_ctx):.6f}
      Fill prompt accuracy (@5): {ctx_bridge_acc:.3f}

  A3  T2 Steering: see per-test rank shifts above.
      A positive Δrank = steering moved the target word UP toward output.
""")

# Save
def to_py(x):
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k,v in x.items()}
    return x

output = {
    'a1': {
        'peak_layer': peak_L,
        'mean_anticipation_by_layer': mean_ant.tolist(),
        'mean_random_by_layer': mean_rnd.tolist(),
        'gap_by_layer': gap.tolist(),
        'lm_accuracy': float(sum(lm_correct)/len(lm_correct)),
        'per_prompt': a1_rows,
    },
    'a2': {
        'r2': r2_ctx, 'cos_mean': float(np.mean(cos_ctx)),
        'bridge_acc_top5': ctx_bridge_acc,
    },
    'a3': {'results': a3_results},
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 60 complete.")
