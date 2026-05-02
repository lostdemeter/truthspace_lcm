#!/usr/bin/env python3
"""
Expedition Day 61 — Steer at L23: T2 Operators at the Knowledge Layer

Day 60 showed:
  Layer  anticipation gap above random
  L14    +0.021   (weak — Zone C semantic)
  L23    +0.056   (stronger — knowledge layer)
  L27    +0.087   (peak — output layer)

T2 steering at L14 moved target ranks (+26 to +159) but didn't reach top-5.
Hypothesis: steering at L23 or L27 should be more effective because:
  1. Higher anticipation gap = hidden state is already more answer-aligned
  2. Fewer subsequent layers to "undo" the perturbation

Key insight from Day 60: T2 vectors were built from L14 isolated hidden states.
Steering at L23 with an L14 T2 vector is misaligned — we're injecting a
perturbation calibrated for L14 curvature into L23 space.

THIS DAY:
  S1  Build T2 vectors at L14, L23, L27 separately (from cached hidden states).
      Measure T2 norms and angular consistency at each layer.

  S2  Layer sweep: steer at {L14, L17, L20, L23, L24, L27} using the
      layer-matched T2 vector. For each layer × alpha grid:
        - Mean rank of target across fill prompts
        - % reaching top-5, top-10, top-20

  S3  Find optimal (layer, alpha, T2) → first configuration reaching top-5.

  S4  Cascade steering: apply T2 at L14 AND L23 simultaneously.
      Does combining both produce stronger effects?

  S5  Full fill-prompt accuracy: at the best (layer, alpha) found in S2/S3,
      run all 40 fill prompts. How close does steered generation get to 74%?
"""

import json, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
L27_CACHE   = str(SCRIPT_DIR / "day59_hs_27_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day61_steer_l23.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

T2_SEEDS = {
    'male_female':     [('king','queen'), ('man','woman'), ('boy','girl'),
                        ('actor','actress'), ('uncle','aunt'),
                        ('brother','sister'), ('father','mother'),
                        ('son','daughter'), ('husband','wife')],
    'singular_plural': [('cat','cats'), ('dog','dogs'), ('tree','trees'),
                        ('bird','birds'), ('house','houses'),
                        ('book','books'), ('car','cars'), ('door','doors')],
    'base_comp':       [('big','bigger'), ('fast','faster'), ('old','older'),
                        ('tall','taller'), ('small','smaller'),
                        ('new','newer'), ('young','younger')],
    'base_past':       [('walk','walked'), ('jump','jumped'), ('talk','talked'),
                        ('play','played'), ('work','worked'), ('look','looked')],
}

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

# Specific T2 steering tests: (prompt, T2_name, target_word)
T2_TESTS = [
    ("The king sat on the",       'male_female',     'queen'),
    ("The boy played in the",     'male_female',     'girl'),
    ("The man walked to the",     'male_female',     'woman'),
    ("The cat sat on the",        'singular_plural', 'cats'),
    ("One book on the shelf, two",'singular_plural', 'books'),
    ("She walked to the",         'base_past',       'walk'),
    ("He is bigger than the",     'base_comp',       'big'),
]

STEER_LAYERS   = [14, 17, 20, 23, 24, 27]
ALPHA_VALUES   = [0, 1, 5, 10, 20, 50, 100, 200]

print("=" * 70)
print("  Expedition Day 61 — T2 Steering at L23 (and Layer Sweep)")
print("=" * 70)


# ── Load caches ──────────────────────────────────────────────────────────────
npz      = np.load(CACHE_FILE, allow_pickle=True)
words_all= list(npz['words'])
hs14_all = npz['hs_14'].astype(np.float64)
hs23_all = npz['hs_23'].astype(np.float64)
c27      = np.load(L27_CACHE)
hs27_all = c27['hs_27'].astype(np.float64)
w2i      = {w: i for i, w in enumerate(words_all)}
N        = len(words_all)

hs_by_layer = {14: hs14_all, 23: hs23_all, 27: hs27_all}

print(f"  Cache: {N} words, layers 14/23/27 loaded")

with open(ATLAS_FILE) as f: atlas = json.load(f)

# ── Load model ───────────────────────────────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_out     = model.lm_head.weight.detach().numpy().astype(np.float64)
n_layers  = model.config.num_hidden_layers
hidden_sz = model.config.hidden_size
print(f"  Loaded. n_layers={n_layers}")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


# ═══════════════════════════════════════════════════════════════════════════════
# S1 — Build Layer-Matched T2 Vectors at L14, L23, L27
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"S1 — Layer-Matched T2 Vectors at L14, L23, L27")
print(f"{'='*70}\n")

def build_t2_at_layer(seeds, layer_hs, threshold=0.3):
    """Build T2 direction in hidden-space at a given layer."""
    vecs = []
    for a, b in seeds:
        for pfx in [' ', '']:
            wa, wb = pfx + a.strip(), pfx + b.strip()
            if wa in w2i and wb in w2i:
                ha = layer_hs[w2i[wa]]
                hb = layer_hs[w2i[wb]]
                d  = hb - ha
                nm = np.linalg.norm(d)
                if nm > 1e-20:
                    vecs.append(d / nm)
                break
    if not vecs:
        return None, 0.0
    mat = np.stack(vecs)
    mean_vec = mat.mean(0)
    nm = np.linalg.norm(mean_vec)
    if nm < 1e-20:
        return None, 0.0
    direction = mean_vec / nm
    # Consistency: mean cosine between individual vecs and direction
    consistency = float(np.mean(mat @ direction))
    return direction, consistency

T2_VECTORS = {}   # {(t2_name, layer): np.ndarray}
T2_TORCH   = {}   # {(t2_name, layer): torch.Tensor}

print(f"  {'T2 name':<20}  {'L14 consist':<13}  {'L23 consist':<13}  {'L27 consist'}")
print(f"  {'-'*65}")
for t2_name, seeds in T2_SEEDS.items():
    row = [f"  {t2_name:<20}"]
    for layer in [14, 23, 27]:
        hs = hs_by_layer.get(layer)
        if hs is None:
            row.append(f"  {'N/A':<13}")
            continue
        vec, cons = build_t2_at_layer(seeds, hs)
        if vec is not None:
            T2_VECTORS[(t2_name, layer)] = vec
            T2_TORCH[(t2_name, layer)]   = torch.tensor(vec, dtype=torch.float32)
        row.append(f"  {cons:.6f}     " if vec is not None else "  FAILED       ")
    print(''.join(row))

print(f"\n  Cross-layer cosine (T2 direction at L14 vs L23 vs L27):")
for t2_name in T2_SEEDS:
    v14 = T2_VECTORS.get((t2_name, 14))
    v23 = T2_VECTORS.get((t2_name, 23))
    v27 = T2_VECTORS.get((t2_name, 27))
    if v14 is not None and v23 is not None and v27 is not None:
        c1423 = cosine(v14, v23)
        c1427 = cosine(v14, v27)
        c2327 = cosine(v23, v27)
        print(f"    {t2_name:<20}  cos(L14,L23)={c1423:.4f}  cos(L14,L27)={c1427:.4f}  cos(L23,L27)={c2327:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# S2 — Layer × Alpha Sweep on T2 Steering Tests
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"S2 — Layer × Alpha Sweep")
print(f"{'='*70}\n")

def steer_and_run(prompt, t2_torch, alpha, steer_layer):
    """Steer at `steer_layer` by injecting alpha * t2 into the last-token
    residual stream via pre-hook on layer steer_layer+1."""
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)

    # baseline
    with torch.no_grad():
        out_orig = model(**inputs, use_cache=False)
    logits_orig = out_orig.logits[0, last_pos, :].detach().numpy()

    if alpha == 0:
        return logits_orig, logits_orig

    patch = (float(alpha) * t2_torch).float()
    next_idx = min(steer_layer + 1, n_layers - 1)

    def pre_hook(module, args):
        if not isinstance(args, tuple) or len(args) == 0:
            return
        h_in = args[0]
        if not isinstance(h_in, torch.Tensor) or h_in.dim() < 2:
            return
        h = h_in.clone().float()
        p = patch.to(h.device)
        if h.dim() == 3:
            h[0, last_pos] = h[0, last_pos] + p
        else:
            h[last_pos] = h[last_pos] + p
        return (h.to(h_in.dtype),) + args[1:]

    handle = model.model.layers[next_idx].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        out_st = model(**inputs, use_cache=False)
    handle.remove()
    return logits_orig, out_st.logits[0, last_pos, :].detach().numpy()


def token_rank(logits, target_word):
    tokens = [tok.decode([i]).strip().lower() for i in np.argsort(-logits)]
    return next((i for i, w in enumerate(tokens) if w == target_word.lower()), 9999)


def token_top5(logits):
    return [tok.decode([i]).strip() for i in np.argsort(-logits)[:5]]


print(f"  Sweep: {len(STEER_LAYERS)} layers × {len(ALPHA_VALUES)} alphas × {len(T2_TESTS)} prompts\n")

# Results: [steer_layer][alpha] → list of (rank_orig, rank_steered) per test
sweep_results = {}   # (layer, alpha, t2_name) → mean_rank_improvement

for t2_name in ['male_female', 'singular_plural', 'base_past']:
    print(f"  T2: {t2_name}")
    print(f"  {'Layer':<8}", end='')
    for alpha in ALPHA_VALUES:
        print(f"  α={alpha:<5}", end='')
    print(f"  (mean Δrank | % top-5)")

    tests_for_t2 = [(p, t, tgt) for p, t, tgt in T2_TESTS if t == t2_name]
    if not tests_for_t2:
        print(f"  [no tests for {t2_name}]")
        continue

    for steer_layer in STEER_LAYERS:
        # Use layer-matched T2 if available, else fall back to L14
        t2_key = (t2_name, steer_layer) if (t2_name, steer_layer) in T2_TORCH \
                 else (t2_name, 14)
        t2_torch = T2_TORCH.get(t2_key)
        if t2_torch is None:
            continue

        print(f"  L{steer_layer:<6}", end='')
        for alpha in ALPHA_VALUES:
            improvements = []
            top5_hits    = 0
            for prompt, _, target in tests_for_t2:
                lo, ls = steer_and_run(prompt, t2_torch, alpha, steer_layer)
                rank_o = token_rank(lo, target)
                rank_s = token_rank(ls, target)
                improvements.append(rank_o - rank_s)
                if rank_s < 5: top5_hits += 1
            mean_imp = np.mean(improvements)
            pct5     = top5_hits / len(tests_for_t2) * 100
            marker   = '★' if top5_hits > 0 else ' '
            print(f"  {mean_imp:+.0f}/{pct5:.0f}%{marker}", end='')
            sweep_results[(steer_layer, alpha, t2_name)] = {
                'mean_improvement': float(mean_imp),
                'pct_top5': float(pct5),
                'top5_hits': int(top5_hits),
            }
        print()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# S3 — Find Optimal Configuration + Detailed Results
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"S3 — Best Configurations (top-5 achievers)")
print(f"{'='*70}\n")

top5_achievers = [(k, v) for k, v in sweep_results.items() if v['top5_hits'] > 0]
top5_achievers.sort(key=lambda x: (-x[1]['top5_hits'], -x[1]['mean_improvement']))

if top5_achievers:
    print(f"  Configurations that achieved top-5:")
    print(f"  {'Layer':<8}  {'alpha':<8}  {'T2':<20}  {'top5_hits':<12}  mean_Δrank")
    for (layer, alpha, t2_name), v in top5_achievers[:20]:
        print(f"  L{layer:<7}  {alpha:<8}  {t2_name:<20}  {v['top5_hits']:>2}/{len([t for p,t,_ in T2_TESTS if t==t2_name]):<8}  "
              f"{v['mean_improvement']:+.1f}")
else:
    print(f"  No configuration reached top-5 with T2 steering alone.")
    print(f"  Best configurations by mean rank improvement:")
    best = sorted(sweep_results.items(), key=lambda x: -x[1]['mean_improvement'])[:10]
    for (layer, alpha, t2_name), v in best:
        print(f"  L{layer}, α={alpha:<5}, {t2_name:<20}  Δrank={v['mean_improvement']:+.1f}  top5={v['pct_top5']:.0f}%")

# Best overall for male_female (most tests)
print(f"\n  Detailed: male_female T2, best alpha at each layer:")
for steer_layer in STEER_LAYERS:
    best_alpha = max(
        [(a, sweep_results.get((steer_layer, a, 'male_female'), {'mean_improvement': -9999}))
         for a in ALPHA_VALUES],
        key=lambda x: x[1]['mean_improvement']
    )
    v = best_alpha[1]
    print(f"  L{steer_layer}: best α={best_alpha[0]:<5}  Δrank={v['mean_improvement']:+.1f}  top5={v['pct_top5']:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# S4 — Cascade Steering: L14 + L23 simultaneously
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"S4 — Cascade Steering: L14 + L23 simultaneously")
print(f"{'='*70}\n")

def steer_cascade(prompt, t2_torch_14, t2_torch_23, alpha):
    """Steer at both L14 and L23 simultaneously."""
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)

    with torch.no_grad():
        out_orig = model(**inputs, use_cache=False)
    logits_orig = out_orig.logits[0, last_pos, :].detach().numpy()

    if alpha == 0:
        return logits_orig, logits_orig

    p14 = (float(alpha) * t2_torch_14).float()
    p23 = (float(alpha) * t2_torch_23).float()

    def make_pre_hook(patch):
        def pre_hook(module, args):
            if not isinstance(args, tuple) or len(args) == 0: return
            h_in = args[0]
            if not isinstance(h_in, torch.Tensor) or h_in.dim() < 2: return
            h = h_in.clone().float()
            p = patch.to(h.device)
            if h.dim() == 3: h[0, last_pos] = h[0, last_pos] + p
            else:             h[last_pos] = h[last_pos] + p
            return (h.to(h_in.dtype),) + args[1:]
        return pre_hook

    h14 = model.model.layers[15].register_forward_pre_hook(make_pre_hook(p14))
    h23 = model.model.layers[24].register_forward_pre_hook(make_pre_hook(p23))
    with torch.no_grad():
        out_st = model(**inputs, use_cache=False)
    h14.remove(); h23.remove()
    return logits_orig, out_st.logits[0, last_pos, :].detach().numpy()


t2_mf_14 = T2_TORCH.get(('male_female', 14))
t2_mf_23 = T2_TORCH.get(('male_female', 23))
t2_sp_14 = T2_TORCH.get(('singular_plural', 14))
t2_sp_23 = T2_TORCH.get(('singular_plural', 23))

print(f"  Test: cascade L14+L23 steering (male_female + singular_plural)")
print(f"  {'alpha':<8}  {'prompt':<35}  {'target':<10}  "
      f"{'rank_orig':<10}  {'rank_L14':<10}  {'rank_L23':<10}  {'rank_cascade'}")
print(f"  {'-'*100}")

cascade_results = []
for prompt, t2_name, target in T2_TESTS[:6]:
    t2_14 = t2_mf_14 if t2_name == 'male_female' else t2_sp_14
    t2_23 = t2_mf_23 if t2_name == 'male_female' else t2_sp_23
    if t2_14 is None or t2_23 is None:
        continue

    for alpha in [20, 50, 100]:
        lo, ls14     = steer_and_run(prompt, t2_14, alpha, 14)
        _,  ls23     = steer_and_run(prompt, t2_23, alpha, 23)
        _,  ls_casc  = steer_cascade(prompt, t2_14, t2_23, alpha)

        rank_o = token_rank(lo, target)
        rank_14 = token_rank(ls14, target)
        rank_23 = token_rank(ls23, target)
        rank_c  = token_rank(ls_casc, target)

        best = min(rank_o, rank_14, rank_23, rank_c)
        marker = '★' if best < 5 else ('↑' if best < rank_o else ' ')
        print(f"  α={alpha:<6}  {prompt[:33]:<35}  {target:<10}  "
              f"{rank_o:<10}  {rank_14:<10}  {rank_23:<10}  {rank_c:<5} {marker}")

        cascade_results.append({
            'prompt': prompt, 'target': target, 't2': t2_name, 'alpha': alpha,
            'rank_orig': rank_o, 'rank_L14': rank_14,
            'rank_L23': rank_23, 'rank_cascade': rank_c,
        })
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# S5 — Full Fill-Prompt Accuracy at Best Configuration
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"S5 — Full Fill-Prompt Evaluation at Best Steering Config")
print(f"{'='*70}\n")

# Find the overall best (layer, alpha) for male_female
best_layer, best_alpha_mf, best_t2 = None, None, None
best_score = -9999
for (layer, alpha, t2_name), v in sweep_results.items():
    if alpha == 0: continue
    if v['mean_improvement'] > best_score:
        best_score = v['mean_improvement']
        best_layer, best_alpha_mf, best_t2 = layer, alpha, t2_name

print(f"  Best overall config: Layer=L{best_layer}, α={best_alpha_mf}, T2={best_t2}")
print(f"  (mean Δrank = {best_score:+.1f})\n")

if best_layer is not None:
    t2_key = (best_t2, best_layer) if (best_t2, best_layer) in T2_TORCH \
             else (best_t2, 14)
    t2_best = T2_TORCH.get(t2_key)

    print(f"  Running all {len(FILL_PROMPTS)} fill prompts (LM baseline vs steered)...")
    print(f"  {'Prompt':<43}  {'Exp':<10}  LM     Steered-top1  Δrank")
    print(f"  {'-'*85}")

    lm_hits = 0; st_hits5 = 0; rank_improvements = []
    s5_rows = []
    for prompt, expected in FILL_PROMPTS:
        lo, ls = steer_and_run(prompt, t2_best, best_alpha_mf, best_layer)
        lm_top1  = tok.decode([np.argmax(lo)]).strip()
        st_top5  = token_top5(ls)
        st_top1  = st_top5[0]
        rank_o   = token_rank(lo, expected)
        rank_s   = token_rank(ls, expected)
        delta    = rank_o - rank_s

        lm_hit = lm_top1.lower() == expected.lower()
        st_hit = expected.lower() in [w.lower() for w in st_top5]
        if lm_hit:  lm_hits  += 1
        if st_hit:  st_hits5 += 1
        rank_improvements.append(delta)

        lm_m = '✓' if lm_hit else '✗'
        st_m = '✓' if st_hit else '✗'
        print(f"  {prompt[:41]:<43}  {expected:<10}  {lm_m}      {st_m}{st_top1:<14}  {delta:+d}")
        s5_rows.append({'prompt': prompt, 'expected': expected,
                        'lm_top1': lm_top1, 'lm_hit': lm_hit,
                        'steered_top5': st_top5, 'steered_hit': st_hit,
                        'rank_orig': int(rank_o), 'rank_steered': int(rank_s),
                        'delta_rank': int(delta)})

    n = len(FILL_PROMPTS)
    lm_acc = lm_hits  / n
    st_acc = st_hits5 / n
    mean_ri = float(np.mean(rank_improvements))
    pct_improved = float(np.mean([r > 0 for r in rank_improvements]))
    print(f"\n  LM-head accuracy (@1):   {lm_acc:.3f}  ({lm_hits}/{n})")
    print(f"  Steered accuracy (@5):   {st_acc:.3f}  ({st_hits5}/{n})")
    print(f"  Mean rank improvement:   {mean_ri:+.1f}")
    print(f"  % prompts improved:      {pct_improved:.3f}")
else:
    s5_rows = []; lm_acc = st_acc = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Day 61")
print(f"{'='*70}")

print(f"""
  S1  T2 vectors built at L14, L23, L27 from cached hidden states.
      Cross-layer cosine (L14 vs L23 direction):
      [see above table for per-T2 consistency and cross-layer cos]

  S2/S3  Layer × Alpha sweep:
      Best config: L{best_layer}, α={best_alpha_mf}, T2={best_t2}
      Mean Δrank = {best_score:+.1f}

  S4  Cascade L14+L23 steering: see table above.

  S5  Full fill-prompt evaluation at best config:
      LM baseline:   {lm_acc:.3f}
      Steered @5:    {st_acc:.3f}

  VERDICT:
    If steered @5 > 0.5:  T2 steering is a viable LCM generation mechanism
    If steered @5 < 0.2:  T2 operators from isolated tokens don\'t survive
                          contextual inference at any layer
""")

# Save
def to_py(x):
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    return x

output = {
    't2_consistency': {
        f"{k[0]}_L{k[1]}": float(float(np.dot(T2_VECTORS[k], T2_VECTORS[k])))
        for k in T2_VECTORS
    },
    'sweep': {f"L{k[0]}_a{k[1]}_{k[2]}": to_py(v) for k, v in sweep_results.items()},
    'cascade': cascade_results,
    's5': {'lm_acc': lm_acc, 'steered_acc': st_acc, 'rows': s5_rows},
    'best_config': {'layer': best_layer, 'alpha': best_alpha_mf, 't2': best_t2,
                    'mean_improvement': best_score},
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 61 complete.")
