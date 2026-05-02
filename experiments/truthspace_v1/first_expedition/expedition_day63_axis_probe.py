#!/usr/bin/env python3
"""
Expedition Day 63 — Probing the Concept-Sharpening Axis

Day 62 revealed: the singular_plural contextual T2 direction at L23 (α=20)
boosts @5 accuracy from 65% to 87.5% across ALL 40 fill-prompt categories.
The direction is anti-aligned (cos=-0.078) with the isolated singular_plural
T2 — so it is NOT encoding plurality. It encodes something else.

PROBES:
  P1  Vocabulary projection: project all 16K word L23 hidden states onto
      the direction. What semantic category do the top/bottom words form?
      If top = concrete nouns, bottom = function words → "concreteness axis".
      If top = high-frequency, bottom = low-frequency → "frequency axis".

  P2  Certainty axis: compute h_L23 for all 40 fill prompts, correlate
      with LM log-probability of the correct answer.  The direction that
      separates "model knows the answer" from "model doesn't" is the
      "certainty axis". Compare with singular_plural contextual direction.

  P3  Combined axis: pool ALL four contextual T2 directions into a
      weighted mean. Does pooling improve beyond 87.5%?

  P4  Held-out generalisation: test the best direction on 20 NEW
      fill prompts not in the Day 62 training set.

  P5  Alpha refinement on best direction: fine sweep α ∈ {10,15,20,25,30}
      on held-out set to confirm the sweet-spot is truly at α=20.
"""

import json, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
L27_CACHE   = str(SCRIPT_DIR / "day59_hs_27_cache.npz")
OUTPUT_FILE = str(SCRIPT_DIR / "day63_axis_probe.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Contextual T2 templates (recomputed from Day 62) ─────────────────────────
CTX_T2_TEMPLATES = {
    'singular_plural': [
        ("One cat sat on the",        "Two cats sat on the"),
        ("One dog ran to the",        "Two dogs ran to the"),
        ("The book was on the",       "The books were on the"),
        ("The car parked near the",   "The cars parked near the"),
        ("One bird flew over the",    "Two birds flew over the"),
        ("The tree fell in the",      "The trees fell in the"),
        ("The house stood on the",    "The houses stood on the"),
        ("The door opened to the",    "The doors opened to the"),
    ],
    'male_female': [
        ("The king walked to the",    "The queen walked to the"),
        ("The king sat on the",       "The queen sat on the"),
        ("The boy played in the",     "The girl played in the"),
        ("The man worked at the",     "The woman worked at the"),
        ("The uncle visited the",     "The aunt visited the"),
        ("The son left the",          "The daughter left the"),
        ("The father cooked the",     "The mother cooked the"),
        ("The husband drove the",     "The wife drove the"),
        ("The brother met the",       "The sister met the"),
    ],
    'base_past': [
        ("Every day I walk to the",   "Yesterday I walked to the"),
        ("Every day I jump over the", "Yesterday I jumped over the"),
        ("Every day I talk to the",   "Yesterday I talked to the"),
        ("Every day I play in the",   "Yesterday I played in the"),
        ("Every day I work at the",   "Yesterday I worked at the"),
    ],
    'base_comp': [
        ("The big dog ran to the",    "The bigger dog ran to the"),
        ("The fast car drove to the", "The faster car drove to the"),
        ("The tall building near the","The taller building near the"),
        ("The old man walked to the", "The older man walked to the"),
        ("The small cat sat on the",  "The smaller cat sat on the"),
    ],
}

# ── Original 40 fill prompts (training set for the direction) ────────────────
FILL_TRAIN = [
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

# ── 20 held-out prompts (NEW — not in training set) ──────────────────────────
FILL_HOLDOUT = [
    ("The capital of Australia is",           "Canberra"),
    ("The capital of China is",               "Beijing"),
    ("The capital of Russia is",              "Moscow"),
    ("The capital of Brazil is",              "Brasilia"),
    ("The capital of India is",               "Delhi"),
    ("The opposite of wet is",                "dry"),
    ("The opposite of loud is",               "quiet"),
    ("The opposite of heavy is",              "light"),
    ("The opposite of north is",              "south"),
    ("The plural of tooth is",                "teeth"),
    ("The plural of mouse is",                "mice"),
    ("The plural of child is",                "children"),
    ("A baby cat is called a",                "kitten"),
    ("A female lion is called a",             "lioness"),
    ("The female version of uncle is",        "aunt"),
    ("The past tense of run is",              "ran"),
    ("The past tense of see is",              "saw"),
    ("The comparative of tall is",            "taller"),
    ("Ice melts and turns into",              "water"),
    ("The colour of blood is",                "red"),
]

# Word categories for P1 interpretation
WORD_CATEGORIES = {
    'function':  ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
                  'and', 'in', 'it', 'that', 'for', 'on', 'with', 'as', 'at'],
    'concrete':  ['cat', 'dog', 'tree', 'house', 'car', 'book', 'bird', 'chair',
                  'table', 'water', 'fire', 'stone', 'wood', 'bread', 'fish',
                  'horse', 'river', 'mountain', 'flower', 'grass'],
    'abstract':  ['freedom', 'justice', 'love', 'truth', 'beauty', 'idea',
                  'thought', 'theory', 'concept', 'belief', 'hope', 'fear',
                  'time', 'space', 'mind', 'soul', 'life', 'death'],
    'proper':    ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Spain',
                  'France', 'Japan', 'Germany', 'Italy', 'China', 'Russia'],
    'morpho':    ['cats', 'dogs', 'trees', 'houses', 'ran', 'walked', 'bigger',
                  'faster', 'quickly', 'jumped', 'played', 'worked'],
}

print("=" * 70)
print("  Expedition Day 63 — Probing the Concept-Sharpening Axis")
print("=" * 70)

# ── Load caches ──────────────────────────────────────────────────────────────
npz      = np.load(CACHE_FILE, allow_pickle=True)
words_all= list(npz['words'])
hs14_all = npz['hs_14'].astype(np.float64)
hs23_all = npz['hs_23'].astype(np.float64)
w2i      = {w: i for i, w in enumerate(words_all)}
N        = len(words_all)

print(f"  Cache: {N} words loaded\n")

# ── Load model ───────────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"  Loaded. n_layers={n_layers}\n")


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_hs(prompt, layers=(14, 23, 27)):
    inputs = tok(prompt, return_tensors='pt')
    last   = inputs['input_ids'].shape[1] - 1
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return {L: out.hidden_states[L][0, last, :].numpy().astype(np.float64)
            for L in layers}


def get_logits(prompt):
    inputs = tok(prompt, return_tensors='pt')
    last   = inputs['input_ids'].shape[1] - 1
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    return out.logits[0, last, :].detach().numpy()


def token_rank(logits, word):
    order = np.argsort(-logits)
    toks  = [tok.decode([i]).strip().lower() for i in order]
    return next((i for i, w in enumerate(toks) if w == word.lower()), 9999)


def token_top5(logits):
    return [tok.decode([i]).strip() for i in np.argsort(-logits)[:5]]


def steer_run(prompt, direction_np, alpha, layer=23):
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)
    with torch.no_grad():
        base_out = model(**inputs, use_cache=False)
    base_logits = base_out.logits[0, last_pos, :].detach().numpy()
    if alpha == 0:
        return base_logits, base_logits
    patch    = torch.tensor(alpha * direction_np, dtype=torch.float32)
    next_idx = min(layer + 1, n_layers - 1)
    def pre_hook(module, args):
        if not isinstance(args, tuple) or not args: return
        h = args[0]
        if not isinstance(h, torch.Tensor) or h.dim() < 2: return
        h2 = h.clone().float()
        p  = patch.to(h2.device)
        if h2.dim() == 3: h2[0, last_pos] += p
        else:              h2[last_pos]    += p
        return (h2.to(h.dtype),) + args[1:]
    handle = model.model.layers[next_idx].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        st_out = model(**inputs, use_cache=False)
    handle.remove()
    return base_logits, st_out.logits[0, last_pos, :].detach().numpy()


def eval_direction(direction_np, prompts, alpha=20, layer=23):
    """Returns (top1_acc, top5_acc, mean_rank_improvement)."""
    top1, top5_h, deltas = 0, 0, []
    for prompt, expected in prompts:
        bl, sl = steer_run(prompt, direction_np, alpha, layer)
        t1_bl = tok.decode([np.argmax(bl)]).strip().lower() == expected.lower()
        t5_sl = expected.lower() in [w.lower() for w in token_top5(sl)]
        if t1_bl: top1 += 1
        if t5_sl: top5_h += 1
        deltas.append(token_rank(bl, expected) - token_rank(sl, expected))
    n = len(prompts)
    return top1/n, top5_h/n, float(np.mean(deltas))


# ════════════════════════════════════════════════════════════════════════════
# Rebuild the key contextual T2 directions (from Day 62)
# ════════════════════════════════════════════════════════════════════════════
print(f"  Rebuilding contextual T2 directions at L23 ...")
t0 = time.time()
CTX_T2_L23 = {}
for t2_name, pairs in CTX_T2_TEMPLATES.items():
    diffs = []
    for sent_a, sent_b in pairs:
        ha = get_hs(sent_a)[23]
        hb = get_hs(sent_b)[23]
        d  = hb - ha
        nm = np.linalg.norm(d)
        if nm > 1e-20: diffs.append(d / nm)
    if diffs:
        mean_vec = np.stack(diffs).mean(0)
        nm = np.linalg.norm(mean_vec)
        if nm > 1e-20:
            CTX_T2_L23[t2_name] = mean_vec / nm
print(f"  Done ({time.time()-t0:.1f}s). Directions: {list(CTX_T2_L23.keys())}\n")

SP_DIR = CTX_T2_L23['singular_plural']   # the 87.5% direction


# ════════════════════════════════════════════════════════════════════════════
# P1 — Vocabulary Projection onto SP Direction
# ════════════════════════════════════════════════════════════════════════════
print(f"{'='*70}")
print(f"P1 — Vocabulary Projection: What does SP direction select?")
print(f"{'='*70}\n")

projections = hs23_all @ SP_DIR          # shape (N,)
order_high  = np.argsort(-projections)
order_low   = np.argsort(projections)

K = 40
print(f"  TOP {K} words (high SP projection):")
top_words = []
for i in range(K):
    idx = order_high[i]
    top_words.append(words_all[idx])
print("  " + "  ".join(top_words))

print(f"\n  BOTTOM {K} words (low SP projection):")
bot_words = []
for i in range(K):
    idx = order_low[i]
    bot_words.append(words_all[idx])
print("  " + "  ".join(bot_words))

# Check how known category words project
print(f"\n  Category mean projections (normalised to N(0,1)):")
mu, sig = projections.mean(), projections.std()
for cat_name, word_list in WORD_CATEGORIES.items():
    vals = []
    for w in word_list:
        for pfx in ['', ' ']:
            key = pfx + w
            if key in w2i:
                vals.append((projections[w2i[key]] - mu) / sig)
                break
    if vals:
        print(f"    {cat_name:<12}: mean={np.mean(vals):+.3f}  "
              f"min={np.min(vals):+.3f}  max={np.max(vals):+.3f}  n={len(vals)}")

# ════════════════════════════════════════════════════════════════════════════
# P2 — Certainty Axis: h_L23 vs LM confidence on fill prompts
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"P2 — Certainty Axis: h_L23 correlates with LM confidence?")
print(f"{'='*70}\n")

print(f"  Extracting h_L23 and confidence for {len(FILL_TRAIN)} fill prompts ...")
H_mat   = []   # (40, hidden_dim)
log_conf= []   # log probability of correct answer

for prompt, expected in FILL_TRAIN:
    hs  = get_hs(prompt, layers=(23,))
    H_mat.append(hs[23])
    lo  = get_logits(prompt)
    probs = torch.softmax(torch.tensor(lo, dtype=torch.float32), dim=0).numpy()
    # Find probability of ANY tokenisation of expected word
    best_prob = 0.0
    for pfx in [' ', '']:
        tids = tok.encode(pfx + expected, add_special_tokens=False)
        if len(tids) == 1:
            best_prob = max(best_prob, float(probs[tids[0]]))
    log_conf.append(float(np.log(best_prob + 1e-20)))

H_mat    = np.stack(H_mat)           # (40, D)
log_conf = np.array(log_conf)        # (40,)

# Fit h → log_conf via least squares (1-dim projection)
# certainty_axis = H.T @ log_conf / ||...||
cert_raw = H_mat.T @ log_conf
cert_nm  = np.linalg.norm(cert_raw)
cert_axis = cert_raw / cert_nm if cert_nm > 1e-20 else cert_raw

cos_cert_sp = cosine(cert_axis, SP_DIR)
print(f"  cos(certainty_axis, SP_direction) = {cos_cert_sp:.6f}")
print(f"  Interpretation: {'highly aligned' if abs(cos_cert_sp) > 0.7 else 'weakly aligned' if abs(cos_cert_sp) > 0.3 else 'orthogonal'}")

# Correlation between projection on SP_DIR and log_conf
proj_on_sp = H_mat @ SP_DIR
r_sp  = float(np.corrcoef(proj_on_sp, log_conf)[0, 1])
r_cert= float(np.corrcoef(H_mat @ cert_axis, log_conf)[0, 1])
print(f"  Pearson r(proj_SP, log_conf)    = {r_sp:.6f}")
print(f"  Pearson r(proj_cert, log_conf)  = {r_cert:.6f}")

# Show which prompts have high/low projection on SP_DIR
sp_proj_train = proj_on_sp
order_sp = np.argsort(-sp_proj_train)
print(f"\n  Fill prompts with HIGHEST SP projection (SP 'prefers' these):")
for i in range(8):
    idx = order_sp[i]
    p, e = FILL_TRAIN[idx]
    lo   = get_logits(p)
    rank = token_rank(lo, e)
    print(f"    proj={sp_proj_train[idx]:+.3f}  rank={rank:4d}  '{p[:40]}' → {e}")

print(f"\n  Fill prompts with LOWEST SP projection (SP 'avoids' these):")
for i in range(8):
    idx = order_sp[-(i+1)]
    p, e = FILL_TRAIN[idx]
    lo   = get_logits(p)
    rank = token_rank(lo, e)
    print(f"    proj={sp_proj_train[idx]:+.3f}  rank={rank:4d}  '{p[:40]}' → {e}")


# ════════════════════════════════════════════════════════════════════════════
# P3 — Combined Directions
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"P3 — Combined Directions: Does Pooling Beat 87.5%?")
print(f"{'='*70}\n")

# Build candidates
candidates = {}
candidates['sp_ctx']       = SP_DIR
candidates['cert_axis']    = cert_axis
candidates['pool_all_ctx'] = None
candidates['sp_cert_mean'] = None

# Pool of all contextual T2 directions
all_ctx = list(CTX_T2_L23.values())
pool_mean = np.stack(all_ctx).mean(0)
pool_nm = np.linalg.norm(pool_mean)
candidates['pool_all_ctx'] = pool_mean / pool_nm if pool_nm > 1e-20 else pool_mean

# SP + certainty axis mean
sc = SP_DIR + cert_axis
sc_nm = np.linalg.norm(sc)
candidates['sp_cert_mean'] = sc / sc_nm if sc_nm > 1e-20 else sc

# Isolated SP direction at L23 (for comparison)
iso_seeds = [('cat','cats'), ('dog','dogs'), ('tree','trees'),
             ('bird','birds'), ('house','houses'),
             ('book','books'), ('car','cars'), ('door','doors')]
iso_vecs = []
for a, b in iso_seeds:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs23_all[w2i[wb]] - hs23_all[w2i[wa]]
            nm = np.linalg.norm(d)
            if nm > 1e-20: iso_vecs.append(d/nm)
            break
if iso_vecs:
    iso_mean = np.stack(iso_vecs).mean(0)
    iso_nm = np.linalg.norm(iso_mean)
    candidates['sp_iso']  = iso_mean / iso_nm if iso_nm > 1e-20 else iso_mean
    candidates['sp_iso_neg'] = -candidates['sp_iso']

# Cross-cosines
print(f"  Cosines between candidate directions:")
cnames = list(candidates.keys())
for i, n1 in enumerate(cnames):
    for n2 in cnames[i+1:]:
        d1, d2 = candidates[n1], candidates[n2]
        if d1 is not None and d2 is not None:
            print(f"    cos({n1}, {n2}) = {cosine(d1, d2):+.6f}")

# Evaluate each candidate on training set (α=20)
print(f"\n  Evaluating on training set (α=20, L23):")
print(f"  {'Direction':<20}  top1    top5    Δrank")
print(f"  {'-'*52}")
p3_results = {}
for name, direction in candidates.items():
    if direction is None: continue
    t1, t5, dr = eval_direction(direction, FILL_TRAIN, alpha=20, layer=23)
    print(f"  {name:<20}  {t1:.3f}   {t5:.3f}   {dr:+.1f}")
    p3_results[name] = {'top1': t1, 'top5': t5, 'mean_delta': dr}


# ════════════════════════════════════════════════════════════════════════════
# P4 — Held-Out Generalisation Test
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"P4 — Held-Out Generalisation (20 NEW prompts)")
print(f"{'='*70}\n")

# Find the best direction from P3 for held-out evaluation
best_name = max(p3_results, key=lambda k: p3_results[k]['top5'])
best_dir  = candidates[best_name]
print(f"  Best from P3: '{best_name}' (top5={p3_results[best_name]['top5']:.3f})\n")

print(f"  Held-out prompts (α=20, L23) — comparing LM vs best direction:")
print(f"  {'Prompt':<45}  {'Exp':<10}  LM  Best  Δrank")
print(f"  {'-'*75}")

# Also test SP for comparison
ho_lm, ho_sp, ho_best = 0, 0, 0
ho_rows = []
for prompt, expected in FILL_HOLDOUT:
    bl, sl_sp   = steer_run(prompt, SP_DIR,   20, 23)
    _,  sl_best = steer_run(prompt, best_dir, 20, 23)
    lm_hit   = tok.decode([np.argmax(bl)]).strip().lower() == expected.lower()
    sp_hit   = expected.lower() in [w.lower() for w in token_top5(sl_sp)]
    best_hit = expected.lower() in [w.lower() for w in token_top5(sl_best)]
    if lm_hit:   ho_lm   += 1
    if sp_hit:   ho_sp   += 1
    if best_hit: ho_best += 1

    rank_bl    = token_rank(bl, expected)
    rank_best  = token_rank(sl_best, expected)
    delta      = rank_bl - rank_best
    lm_m   = '✓' if lm_hit else '✗'
    best_m = '★' if best_hit else ('↑' if delta > 0 else '✗')
    print(f"  {prompt[:43]:<45}  {expected:<10}  {lm_m}   {best_m}     {delta:+d}")
    ho_rows.append({'prompt': prompt, 'expected': expected,
                    'lm_hit': lm_hit, 'sp_hit': sp_hit, 'best_hit': best_hit,
                    'delta_rank': int(delta)})

n = len(FILL_HOLDOUT)
print(f"\n  LM baseline (@1):            {ho_lm/n:.3f}  ({ho_lm}/{n})")
print(f"  SP ctx direction (@5):       {ho_sp/n:.3f}  ({ho_sp}/{n})")
print(f"  Best ctx direction (@5):     {ho_best/n:.3f}  ({ho_best}/{n})")


# ════════════════════════════════════════════════════════════════════════════
# P5 — Alpha Refinement on Best Direction (held-out)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"P5 — Alpha Refinement on Best Direction (held-out set)")
print(f"{'='*70}\n")

REFINE_ALPHAS = [5, 10, 15, 20, 25, 30, 40, 50]
print(f"  Direction: {best_name}")
print(f"  {'alpha':<8}  {'top1 (HO)':<12}  {'top5 (HO)':<12}  {'top1 (train)':<14}  top5 (train)")
print(f"  {'-'*60}")
p5_results = {}
for alpha in REFINE_ALPHAS:
    t1h, t5h, _ = eval_direction(best_dir, FILL_HOLDOUT, alpha=alpha, layer=23)
    t1t, t5t, _ = eval_direction(best_dir, FILL_TRAIN,   alpha=alpha, layer=23)
    marker = ' ★' if t5h > ho_sp/n else ''
    print(f"  α={alpha:<6}  {t1h:.3f}         {t5h:.3f}         "
          f"{t1t:.3f}            {t5t:.3f}{marker}")
    p5_results[alpha] = {'top1_ho': t1h, 'top5_ho': t5h,
                         'top1_train': t1t, 'top5_train': t5t}


# ════════════════════════════════════════════════════════════════════════════
# P1 extra: semantic interpretation summary
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"INTERPRETATION SUMMARY")
print(f"{'='*70}\n")

# Compute category mean projections with normalised SP projections
proj_norm = (projections - projections.mean()) / projections.std()
cat_stats = {}
for cat, words in WORD_CATEGORIES.items():
    vals = []
    for w in words:
        for pfx in ['', ' ']:
            if pfx + w in w2i:
                vals.append(proj_norm[w2i[pfx+w]])
                break
    if vals:
        cat_stats[cat] = float(np.mean(vals))

# Sort categories by projection
sorted_cats = sorted(cat_stats.items(), key=lambda x: -x[1])
print(f"  Category ordering on SP direction (high = direction moves toward):")
for cat, score in sorted_cats:
    bar = '█' * int(abs(score) * 10) if score > 0 else '░' * int(abs(score) * 10)
    print(f"    {cat:<14}: {score:+.3f}  {bar}")

print(f"\n  cos(certainty_axis, SP_direction) = {cos_cert_sp:.4f}")
print(f"  r(SP_projection, log_confidence)  = {r_sp:.4f}")

# Interpret
interp = []
if sorted_cats[0][0] in ('concrete', 'proper', 'morpho'):
    interp.append("Direction selects CONCRETE/SPECIFIC vocabulary")
if sorted_cats[-1][0] in ('function', 'abstract'):
    interp.append("Direction avoids FUNCTION/ABSTRACT words")
if abs(cos_cert_sp) > 0.5:
    interp.append(f"Substantially aligned with certainty axis (cos={cos_cert_sp:.3f})")
elif abs(cos_cert_sp) > 0.2:
    interp.append(f"Weakly aligned with certainty axis (cos={cos_cert_sp:.3f})")
else:
    interp.append(f"Orthogonal to certainty axis (cos={cos_cert_sp:.3f})")
if abs(r_sp) > 0.4:
    interp.append(f"SP projection positively correlated with model confidence (r={r_sp:.3f})")

print(f"\n  Interpretation:")
for line in interp:
    print(f"    • {line}")

print(f"\n  NAMING THE AXIS:")
if sorted_cats[0][0] in ('concrete', 'proper'):
    axis_name = "CONCRETENESS / REFERENTIAL SPECIFICITY"
elif abs(cos_cert_sp) > 0.5:
    axis_name = "MODEL CONFIDENCE / ANSWER CERTAINTY"
elif sorted_cats[0][0] == 'morpho':
    axis_name = "MORPHOLOGICAL SPECIFICITY"
else:
    axis_name = "UNKNOWN — inspect vocabulary list above"
print(f"    AXIS NAME: {axis_name}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Day 63")
print(f"{'='*70}")

best_train = p3_results.get(best_name, {}).get('top5', 0)
best_alpha_ho = max(p5_results, key=lambda a: p5_results[a]['top5_ho'])
print(f"""
  P1  SP direction vocabulary profile: [see top/bottom word lists above]
      Category ordering: {' > '.join(c for c,_ in sorted_cats)}

  P2  Certainty axis:
      cos(certainty_axis, SP) = {cos_cert_sp:.4f}
      r(SP_proj, log_conf)    = {r_sp:.4f}

  P3  Best combined direction: {best_name}  (train top5={best_train:.3f})

  P4  Held-out generalisation:
      LM baseline:           {ho_lm/n:.3f}
      SP ctx direction @5:   {ho_sp/n:.3f}
      Best direction @5:     {ho_best/n:.3f}

  P5  Best alpha (held-out): α={best_alpha_ho}
      top5_ho={p5_results[best_alpha_ho]['top5_ho']:.3f}  top5_train={p5_results[best_alpha_ho]['top5_train']:.3f}

  AXIS NAME: {axis_name}
""")

# Save
def to_py(x):
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {str(k): to_py(v) for k, v in x.items()}
    return x

output = {
    'p1': {'top_words': top_words[:40], 'bot_words': bot_words[:40],
           'category_means': cat_stats},
    'p2': {'cos_cert_sp': float(cos_cert_sp), 'r_sp': float(r_sp),
           'r_cert': float(r_cert)},
    'p3': p3_results,
    'p4': {'lm': float(ho_lm/n), 'sp': float(ho_sp/n), 'best': float(ho_best/n),
           'best_direction': best_name, 'rows': ho_rows},
    'p5': to_py(p5_results),
    'axis_name': axis_name,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 63 complete.")
