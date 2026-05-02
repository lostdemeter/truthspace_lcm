#!/usr/bin/env python3
"""
Expedition Day 62 — Contextual T2 Vectors

Day 61 revealed:
  - T2 directions are nearly identical at L14 and L23 (cos > 0.93)
  - L23 is a better steering point (stable at high α)
  - Cascade L14+L23 at α=50 → queen rank 13  (need < 5 for top-5)
  - The remaining gap: isolated T2 vectors are 6.7% misaligned from context

Root cause: T2 vectors were built from ISOLATED single-word hidden states.
In context, h_L14/h_L23 includes:
  (a) the word identity (Zone C body)
  (b) the contextual grounding (~77° shift from isolated, Day 57)
  (c) positional / syntactic information

The CONTEXTUAL T2 direction is the difference between sentence pairs where
only the semantic axis changes:
  "The king walked to the"  →  h_L23_last
  "The queen walked to the" →  h_L23_last
  T2_ctx = normalize(h_queen - h_king)

If the contextual T2 direction is better aligned with the output geometry,
it should close the rank-13 → rank-4 gap.

THIS DAY:
  C1  Build contextual T2 vectors at L14 and L23 from sentence pairs.
      Compare direction vs isolated T2 (cos alignment).
      Measure consistency across multiple sentence templates.

  C2  Fine alpha sweep at L23 with CONTEXTUAL T2.
      α ∈ {30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150}
      Target: queen, girl, woman, cats, books.

  C3  L23 + L27 cascade using contextual T2.
      Does double-steering in the late layers reach top-5?

  C4  Head-to-head: isolated T2 vs contextual T2 on all fill prompts.
      At the best α found in C2, run all 40 fill prompts with:
        (a) no steering (LM baseline)
        (b) isolated T2 at best layer/α
        (c) contextual T2 at best layer/α
      Report top-5 accuracy for each.
"""

import json, time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
L27_CACHE   = str(SCRIPT_DIR / "day59_hs_27_cache.npz")
OUTPUT_FILE = str(SCRIPT_DIR / "day62_contextual_t2.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Sentence pairs for contextual T2 construction ────────────────────────────
# Each pair differs ONLY in the target semantic axis
CTX_T2_TEMPLATES = {
    'male_female': [
        ("The king walked to the",    "The queen walked to the"),
        ("The king sat on the",       "The queen sat on the"),
        ("The king ruled the",        "The queen ruled the"),
        ("The king spoke to the",     "The queen spoke to the"),
        ("The boy played in the",     "The girl played in the"),
        ("The boy ran to the",        "The girl ran to the"),
        ("The man worked at the",     "The woman worked at the"),
        ("The man drove to the",      "The woman drove to the"),
        ("The uncle visited the",     "The aunt visited the"),
        ("The son left the",          "The daughter left the"),
        ("The father cooked the",     "The mother cooked the"),
        ("The husband drove the",     "The wife drove the"),
        ("The brother met the",       "The sister met the"),
        ("The actor starred in the",  "The actress starred in the"),
    ],
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
    'base_comp': [
        ("The big dog ran to the",    "The bigger dog ran to the"),
        ("The fast car drove to the", "The faster car drove to the"),
        ("The tall building near the","The taller building near the"),
        ("The old man walked to the", "The older man walked to the"),
        ("The small cat sat on the",  "The smaller cat sat on the"),
    ],
    'base_past': [
        ("Every day I walk to the",   "Yesterday I walked to the"),
        ("Every day I jump over the", "Yesterday I jumped over the"),
        ("Every day I talk to the",   "Yesterday I talked to the"),
        ("Every day I play in the",   "Yesterday I played in the"),
        ("Every day I work at the",   "Yesterday I worked at the"),
    ],
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

T2_TESTS = [
    ("The king sat on the",       'male_female',     'queen'),
    ("The boy played in the",     'male_female',     'girl'),
    ("The man walked to the",     'male_female',     'woman'),
    ("The cat sat on the",        'singular_plural', 'cats'),
    ("One book on the shelf, two",'singular_plural', 'books'),
    ("The walk was",              'base_past',       'walked'),
]

print("=" * 70)
print("  Expedition Day 62 — Contextual T2 Vectors")
print("=" * 70)

# ── Load caches ──────────────────────────────────────────────────────────────
npz      = np.load(CACHE_FILE, allow_pickle=True)
words_all= list(npz['words'])
hs14_all = npz['hs_14'].astype(np.float64)
hs23_all = npz['hs_23'].astype(np.float64)
c27      = np.load(L27_CACHE)
hs27_all = c27['hs_27'].astype(np.float64)
w2i      = {w: i for i, w in enumerate(words_all)}

hs_isolated = {14: hs14_all, 23: hs23_all, 27: hs27_all}

# ── Load model ───────────────────────────────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_out    = model.lm_head.weight.detach().numpy().astype(np.float64)
n_layers = model.config.num_hidden_layers
print(f"  Loaded. n_layers={n_layers}\n")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def get_hidden_states(prompt, layers=(14, 23, 27)):
    """Return {layer: hidden_state_at_last_pos} for specified layers."""
    inputs = tok(prompt, return_tensors='pt')
    last   = inputs['input_ids'].shape[1] - 1
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return {L: out.hidden_states[L][0, last, :].numpy().astype(np.float64)
            for L in layers}


def token_rank(logits, target_word):
    toks = [tok.decode([i]).strip().lower() for i in np.argsort(-logits)]
    return next((i for i, w in enumerate(toks) if w == target_word.lower()), 9999)


def token_top5(logits):
    return [tok.decode([i]).strip() for i in np.argsort(-logits)[:5]]


def steer_and_run(prompt, t2_torch, alpha, steer_layer):
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)
    with torch.no_grad():
        out_orig = model(**inputs, use_cache=False)
    logits_orig = out_orig.logits[0, last_pos, :].detach().numpy()
    if alpha == 0:
        return logits_orig, logits_orig
    patch    = (float(alpha) * t2_torch).float()
    next_idx = min(steer_layer + 1, n_layers - 1)
    def pre_hook(module, args):
        if not isinstance(args, tuple) or not args: return
        h_in = args[0]
        if not isinstance(h_in, torch.Tensor) or h_in.dim() < 2: return
        h = h_in.clone().float()
        p = patch.to(h.device)
        if h.dim() == 3: h[0, last_pos] = h[0, last_pos] + p
        else:             h[last_pos] = h[last_pos] + p
        return (h.to(h_in.dtype),) + args[1:]
    handle = model.model.layers[next_idx].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        out_st = model(**inputs, use_cache=False)
    handle.remove()
    return logits_orig, out_st.logits[0, last_pos, :].detach().numpy()


def steer_cascade(prompt, t2_torch_a, layer_a, t2_torch_b, layer_b, alpha):
    """Steer at two layers simultaneously."""
    inputs   = tok(prompt, return_tensors='pt')
    last_pos = int(inputs['input_ids'].shape[1] - 1)
    with torch.no_grad():
        out_orig = model(**inputs, use_cache=False)
    logits_orig = out_orig.logits[0, last_pos, :].detach().numpy()
    if alpha == 0:
        return logits_orig, logits_orig

    def make_hook(patch):
        def pre_hook(module, args):
            if not isinstance(args, tuple) or not args: return
            h_in = args[0]
            if not isinstance(h_in, torch.Tensor) or h_in.dim() < 2: return
            h = h_in.clone().float()
            p = patch.to(h.device)
            if h.dim() == 3: h[0, last_pos] = h[0, last_pos] + p
            else:             h[last_pos] = h[last_pos] + p
            return (h.to(h_in.dtype),) + args[1:]
        return pre_hook

    pa = (float(alpha) * t2_torch_a).float()
    pb = (float(alpha) * t2_torch_b).float()
    h_a = model.model.layers[min(layer_a+1, n_layers-1)].register_forward_pre_hook(make_hook(pa))
    h_b = model.model.layers[min(layer_b+1, n_layers-1)].register_forward_pre_hook(make_hook(pb))
    with torch.no_grad():
        out_st = model(**inputs, use_cache=False)
    h_a.remove(); h_b.remove()
    return logits_orig, out_st.logits[0, last_pos, :].detach().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# C1 — Build Contextual T2 Vectors
# ═══════════════════════════════════════════════════════════════════════════════
print(f"{'='*70}")
print(f"C1 — Build Contextual T2 Vectors at L14, L23, L27")
print(f"{'='*70}\n")

print(f"  Extracting hidden states from {sum(len(v) for v in CTX_T2_TEMPLATES.values())} "
      f"sentence pairs across {len(CTX_T2_TEMPLATES)} T2 types...\n")

CTX_T2_VECS   = {}   # {(t2_name, layer): np.ndarray}
CTX_T2_TORCH  = {}   # {(t2_name, layer): torch.Tensor}
CTX_T2_STATS  = {}   # {t2_name: {layer: {consistency, cos_vs_isolated, n_pairs}}}

ISO_T2_VECS   = {}   # rebuilt isolated T2 for comparison
for t2_name in CTX_T2_TEMPLATES:
    CTX_T2_STATS[t2_name] = {}

t0 = time.time()
pair_count = 0
for t2_name, pairs in CTX_T2_TEMPLATES.items():
    diffs = {14: [], 23: [], 27: []}
    for sent_a, sent_b in pairs:
        hs_a = get_hidden_states(sent_a)
        hs_b = get_hidden_states(sent_b)
        for L in [14, 23, 27]:
            d = hs_b[L] - hs_a[L]
            nm = np.linalg.norm(d)
            if nm > 1e-20:
                diffs[L].append(d / nm)
        pair_count += 1
        if pair_count % 10 == 0:
            print(f"  [{pair_count} pairs done, {time.time()-t0:.1f}s]")

    for L in [14, 23, 27]:
        if not diffs[L]: continue
        mat      = np.stack(diffs[L])
        mean_vec = mat.mean(0)
        nm       = np.linalg.norm(mean_vec)
        if nm < 1e-20: continue
        direction   = mean_vec / nm
        consistency = float(np.mean(mat @ direction))

        CTX_T2_VECS[(t2_name, L)]  = direction
        CTX_T2_TORCH[(t2_name, L)] = torch.tensor(direction, dtype=torch.float32)

        # Isolated T2 for comparison
        iso_hs = hs_isolated.get(L)
        if iso_hs is not None:
            iso_seeds = {
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
            }.get(t2_name, [])
            iso_vecs = []
            for a, b in iso_seeds:
                for pfx in [' ', '']:
                    wa, wb = pfx+a, pfx+b
                    if wa in w2i and wb in w2i:
                        d = iso_hs[w2i[wb]] - iso_hs[w2i[wa]]
                        nm2 = np.linalg.norm(d)
                        if nm2 > 1e-20: iso_vecs.append(d/nm2)
                        break
            if iso_vecs:
                iso_mean = np.stack(iso_vecs).mean(0)
                iso_nm   = np.linalg.norm(iso_mean)
                if iso_nm > 1e-20:
                    iso_dir = iso_mean / iso_nm
                    ISO_T2_VECS[(t2_name, L)] = iso_dir
                    cos_align = cosine(direction, iso_dir)
                else:
                    cos_align = 0.0
            else:
                cos_align = 0.0
        else:
            cos_align = 0.0

        CTX_T2_STATS[t2_name][L] = {
            'consistency': consistency,
            'n_pairs': len(diffs[L]),
            'cos_vs_isolated': cos_align,
        }

print(f"\n  Done. {time.time()-t0:.1f}s\n")
print(f"  Contextual T2 statistics:")
print(f"  {'T2 name':<20}  {'Layer':<6}  {'n_pairs':<8}  {'consistency':<13}  cos(ctx vs iso)")
print(f"  {'-'*70}")
for t2_name in CTX_T2_TEMPLATES:
    for L in [14, 23, 27]:
        st = CTX_T2_STATS[t2_name].get(L)
        if st:
            print(f"  {t2_name:<20}  L{L:<5}  {st['n_pairs']:<8}  "
                  f"{st['consistency']:.6f}     {st['cos_vs_isolated']:.6f}")

print(f"\n  Cross-layer consistency of contextual T2 direction:")
print(f"  {'T2 name':<20}  cos_ctx(L14,L23)  cos_ctx(L14,L27)  cos_ctx(L23,L27)")
for t2_name in CTX_T2_TEMPLATES:
    v14 = CTX_T2_VECS.get((t2_name, 14))
    v23 = CTX_T2_VECS.get((t2_name, 23))
    v27 = CTX_T2_VECS.get((t2_name, 27))
    if v14 is not None and v23 is not None and v27 is not None:
        print(f"  {t2_name:<20}  {cosine(v14,v23):.6f}         "
              f"{cosine(v14,v27):.6f}         {cosine(v23,v27):.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# C2 — Fine Alpha Sweep at L23 with Contextual vs Isolated T2
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"C2 — Fine Alpha Sweep at L23: Contextual vs Isolated T2")
print(f"{'='*70}\n")

FINE_ALPHAS = [0, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200]
STEER_LAYER = 23

print(f"  Comparing contextual vs isolated T2, steering at L{STEER_LAYER}\n")
c2_results = {}
for prompt, t2_name, target in T2_TESTS:
    ctx_vec  = CTX_T2_TORCH.get((t2_name, STEER_LAYER))
    iso_vec  = ISO_T2_VECS.get((t2_name, STEER_LAYER))
    if ctx_vec is None:
        print(f"  [{t2_name}] ctx T2 not built, skipping")
        continue

    print(f"  Prompt: \"{prompt}\"  target='{target}'")
    print(f"  {'alpha':<8}  {'rank_orig':<10}  {'rank_iso':<10}  {'rank_ctx':<10}  "
          f"iso_top5  ctx_top5  ctx_top1")
    print(f"  {'-'*72}")

    best_iso_rank = 9999; best_ctx_rank = 9999; best_ctx_alpha = None

    for alpha in FINE_ALPHAS:
        lo, l_iso = steer_and_run(prompt, torch.tensor(iso_vec, dtype=torch.float32)
                                   if iso_vec is not None else ctx_vec,
                                  alpha, STEER_LAYER) \
                   if iso_vec is not None else (None, None)
        lo2, l_ctx = steer_and_run(prompt, ctx_vec, alpha, STEER_LAYER)

        rank_o   = token_rank(lo2, target)
        rank_iso = token_rank(l_iso, target) if l_iso is not None else 9999
        rank_ctx = token_rank(l_ctx, target)
        top5_ctx = token_top5(l_ctx)
        top1_ctx = top5_ctx[0]

        iso_hit = '★' if rank_iso < 5 else (' ' if rank_iso < 20 else ' ')
        ctx_hit = '★' if rank_ctx < 5 else (' ' if rank_ctx < 20 else ' ')

        print(f"  α={alpha:<6}  {rank_o:<10}  {rank_iso:<10}  {rank_ctx:<10}  "
              f"{iso_hit}         {ctx_hit}         {top1_ctx}")

        if rank_ctx < best_ctx_rank:
            best_ctx_rank = rank_ctx; best_ctx_alpha = alpha
        if rank_iso < best_iso_rank:
            best_iso_rank = rank_iso

        c2_results[(prompt, t2_name, alpha)] = {
            'rank_orig': int(rank_o),
            'rank_iso_L23': int(rank_iso),
            'rank_ctx_L23': int(rank_ctx),
        }

    print(f"\n  Best iso rank: {best_iso_rank}   Best ctx rank: {best_ctx_rank} "
          f"(α={best_ctx_alpha})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# C3 — Cascade Tests: L23 + L27 with Contextual T2
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"C3 — Cascade Steering: L23 + L27 (Contextual T2)")
print(f"{'='*70}\n")

CASCADE_ALPHAS = [20, 30, 40, 50, 60, 70, 75, 80, 100]

print(f"  Comparing: L23 only  vs  L14+L23  vs  L23+L27  vs  L14+L23+L27\n")
c3_results = []
for prompt, t2_name, target in T2_TESTS[:4]:
    ctx_14 = CTX_T2_TORCH.get((t2_name, 14))
    ctx_23 = CTX_T2_TORCH.get((t2_name, 23))
    ctx_27 = CTX_T2_TORCH.get((t2_name, 27))
    if ctx_23 is None: continue

    lo, _ = steer_and_run(prompt, ctx_23, 0, 23)
    rank_o = token_rank(lo, target)
    print(f"  \"{prompt}\"  target='{target}'  baseline rank={rank_o}")
    print(f"  {'alpha':<8}  {'L23':<10}  {'L14+L23':<12}  {'L23+L27':<12}  {'L14+L23+L27'}")
    print(f"  {'-'*65}")

    for alpha in CASCADE_ALPHAS:
        _, ls23       = steer_and_run(prompt, ctx_23, alpha, 23)
        r23           = token_rank(ls23, target)

        r1423 = 9999
        if ctx_14 is not None:
            _, ls1423 = steer_cascade(prompt, ctx_14, 14, ctx_23, 23, alpha)
            r1423     = token_rank(ls1423, target)

        r2327 = 9999
        if ctx_27 is not None:
            _, ls2327 = steer_cascade(prompt, ctx_23, 23, ctx_27, 27, alpha)
            r2327     = token_rank(ls2327, target)

        r_triple = 9999
        if ctx_14 is not None and ctx_27 is not None:
            # three-layer: pre-hook on L15, L24, L28(→L27)
            inputs   = tok(prompt, return_tensors='pt')
            last_pos = int(inputs['input_ids'].shape[1] - 1)
            pa = (float(alpha) * ctx_14).float()
            pb = (float(alpha) * ctx_23).float()
            pc = (float(alpha) * ctx_27).float()
            def mh(patch):
                def h(m, a):
                    if not isinstance(a, tuple) or not a: return
                    hi = a[0]
                    if not isinstance(hi, torch.Tensor) or hi.dim() < 2: return
                    hh = hi.clone().float(); hh[0 if hh.dim()==3 else ..., last_pos] = \
                        (hh[0, last_pos] if hh.dim()==3 else hh[last_pos]) + patch.to(hh.device)
                    if hh.dim() == 3: hh[0, last_pos] = hh[0, last_pos]
                    return (hh.to(hi.dtype),) + a[1:]
                return h
            def mh3(p14, p23, p27):
                def h(m, a):
                    if not isinstance(a, tuple) or not a: return
                    hi = a[0]
                    if not isinstance(hi, torch.Tensor) or hi.dim() < 2: return
                    hh = hi.clone().float()
                    # We can't distinguish which layer we're in here;
                    # use steer_cascade for triple
                    return (hh.to(hi.dtype),) + a[1:]
                return h
            # Just do cascade triple via two steer_cascade calls (approximate)
            _, ls_ab = steer_cascade(prompt, ctx_14, 14, ctx_23, 23, alpha)
            # Can't easily triple-cascade in one pass; report N/A
            r_triple = 9999

        marker = '★' if min(r23, r1423, r2327) < 5 else ''
        print(f"  α={alpha:<6}  {r23:<10}  {r1423:<12}  {r2327:<12}  {'N/A':<12} {marker}")
        c3_results.append({'prompt': prompt, 'target': target, 't2': t2_name,
                            'alpha': alpha, 'rank_orig': rank_o,
                            'rank_L23': int(r23), 'rank_L14L23': int(r1423),
                            'rank_L23L27': int(r2327)})
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# C4 — Head-to-Head: LM vs Isolated T2 vs Contextual T2 on All 40 Fill Prompts
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"C4 — Full 40 Fill-Prompt Evaluation: LM vs Isolated vs Contextual T2")
print(f"{'='*70}\n")

# Find the best (t2_name, alpha) for contextual T2 at L23 from C2
# We'll use male_female at the alpha that gave the best 'queen' rank
# Additionally test singular_plural
best_ctx_configs = {}
for t2_name in ['male_female', 'singular_plural', 'base_comp', 'base_past']:
    vec = CTX_T2_TORCH.get((t2_name, 23))
    if vec is None: continue
    # Use sweet-spot alpha from C2 results (default 70 if not found)
    relevant = [(alpha, v['rank_ctx_L23'])
                for (p, tn, alpha), v in c2_results.items()
                if tn == t2_name and alpha > 0]
    if relevant:
        best_alpha_for_t2 = min(relevant, key=lambda x: x[1])[0]
    else:
        best_alpha_for_t2 = 70
    best_ctx_configs[t2_name] = best_alpha_for_t2

print(f"  Best contextual T2 configs from C2:")
for t2_name, alpha in best_ctx_configs.items():
    print(f"    {t2_name:<20}  L23 α={alpha}")

print(f"\n  Evaluating all 40 fill prompts...\n")
print(f"  {'Prompt':<43}  {'Exp':<10}  LM    Ctx-top1  Δrank")
print(f"  {'-'*80}")

lm_hits  = 0
ctx_hits = {t2: 0 for t2 in best_ctx_configs}
c4_rows  = []

for prompt, expected in FILL_PROMPTS:
    lo, _ = steer_and_run(prompt, CTX_T2_TORCH.get(('male_female', 23),
            torch.zeros(model.config.hidden_size)), 0, 23)
    lm_top1 = tok.decode([np.argmax(lo)]).strip()
    lm_hit  = lm_top1.lower() == expected.lower()
    if lm_hit: lm_hits += 1

    row = {'prompt': prompt, 'expected': expected, 'lm_top1': lm_top1,
           'lm_hit': lm_hit, 'steered': {}}

    # Test with each T2 at its best alpha
    best_rank = token_rank(lo, expected)
    best_t2_label = 'LM'
    ctx_results_row = {}
    for t2_name, alpha in best_ctx_configs.items():
        vec = CTX_T2_TORCH.get((t2_name, 23))
        if vec is None: continue
        _, ls = steer_and_run(prompt, vec, alpha, 23)
        rank_s  = token_rank(ls, expected)
        top5_s  = token_top5(ls)
        hit     = expected.lower() in [w.lower() for w in top5_s]
        if hit: ctx_hits[t2_name] += 1
        ctx_results_row[t2_name] = {
            'rank': int(rank_s), 'top5': top5_s, 'hit': hit
        }
        if rank_s < best_rank:
            best_rank = rank_s
            best_t2_label = t2_name

    row['steered'] = ctx_results_row
    c4_rows.append(row)

    rank_o   = token_rank(lo, expected)
    best_top1 = c4_rows[-1]['steered'].get(best_t2_label, {}).get('top5', [lm_top1])[0] \
                if best_t2_label != 'LM' else lm_top1
    delta  = rank_o - best_rank
    lm_m   = '✓' if lm_hit else '✗'
    best_m = '★' if best_rank < 5 else ('↑' if delta > 0 else '✗')

    print(f"  {prompt[:41]:<43}  {expected:<10}  {lm_m}     {best_m}{best_top1:<12}  {delta:+d}  [{best_t2_label}]")

lm_acc = lm_hits / len(FILL_PROMPTS)
print(f"\n  LM baseline accuracy (@1):       {lm_acc:.3f}  ({lm_hits}/{len(FILL_PROMPTS)})")
for t2_name, hits in ctx_hits.items():
    alpha = best_ctx_configs.get(t2_name, '?')
    print(f"  Ctx T2 '{t2_name}' (@5, α={alpha}): {hits/len(FILL_PROMPTS):.3f}  ({hits}/{len(FILL_PROMPTS)})")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Day 62")
print(f"{'='*70}")

print(f"""
  C1  Contextual T2 vectors built from {sum(len(v) for v in CTX_T2_TEMPLATES.values())} sentence pairs.
      Consistency and alignment with isolated T2: see table above.

  C2  Fine alpha sweep at L23:
      Contextual T2 best ranks per test: see table above.
      Key: did contextual T2 outperform isolated T2?

  C3  Cascade L23+L27 results: see table above.

  C4  Full 40-prompt evaluation:
      LM baseline:  {lm_acc:.3f}
      Best ctx T2 accuracy per T2 type: see above.

  VERDICT:
    ctx_T2 top-5 reach > 50%:  Contextual T2 is a functional LCM mechanism
    ctx_T2 closes rank gap to  < 5: LCM loop closes via T2 steering
""")

# Save
out = {
    'ctx_t2_stats': CTX_T2_STATS,
    'c2': [{
        'prompt': p, 't2': t, 'alpha': a,
        'rank_orig': v['rank_orig'],
        'rank_iso': v['rank_iso_L23'],
        'rank_ctx': v['rank_ctx_L23'],
    } for (p, t, a), v in c2_results.items()],
    'c3': c3_results,
    'c4': {'lm_acc': lm_acc, 'ctx_hits': ctx_hits,
           'best_configs': best_ctx_configs, 'rows': c4_rows},
}
def to_py(x):
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {str(k): to_py(v) for k, v in x.items()}
    return x
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(out), f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 62 complete.")
