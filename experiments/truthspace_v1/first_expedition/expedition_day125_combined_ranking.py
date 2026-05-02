#!/usr/bin/env python3
"""
Day 125 — T2 + d_k Combined Ranking

Day 124 found:
  - T2 (isolated): good CATEGORY filter (MRR=0.540, above random 0.314)
  - T2 (in-context): fails for semantic categories (MRR=0.244, below random)
  - T2 works for syntactic categories (tense/gender: 100%)

QUESTION: Can d_k distinguish within-category candidates?
  e.g., for "The capital of France is", does d_k project more strongly
  for "Paris" than for "London" or "Berlin" when each is appended?

d_k background (Finding 40, Days 117-119):
  - d_k = W_k^T @ v1 from L23 H6 attention head
  - Activates 4.3x at entity position in retrieval context
  - At last-token: activates ~3x for any multi-word prompt

HYPOTHESIS:
  In "The capital of France is Paris", d_k at the LAST TOKEN ("Paris")
  should be higher than in "The capital of France is London" because:
  - "Paris" is the correct entity, consistent with the preceding context
  - "London" is semantically inconsistent → different d_k signature

COMBINED STRATEGY:
  Score(candidate) = α × T2_category_sim + (1-α) × d_k_last_token
  Find optimal α that maximizes MRR across all 12 cases.

Also test: d_k at entity/candidate token position vs last token.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day125_combined_ranking.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# T2 axis configuration
DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
        ("The father worked to feed family","The mother worked to feed family"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in field","Dogs played happily in field"),
        ("The cat sat quietly by window","The cats sat quietly by window"),
        ("A bird sang softly in mist","Birds sang softly in mist"),
        ("The tree fell down in storm","The trees fell down in storm"),
        ("A book sat open on desk","Books sat open on desk"),
        ("The car drove slowly down road","The cars drove slowly down road"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy","The burden is too heavy"),
        ("The long road leads","The long journey leads"),
        ("The high wall blocks","The high barrier blocks"),
        ("The flame fades away","The hope fades away"),
        ("The root grips soil","The base grips earth"),
        ("The bridge connects","The bond connects"),
    ],
    "past_tense": [
        ("I walk every morning","I walked every morning"),
        ("She runs through park","She ran through park"),
        ("He eats before leaving","He ate before leaving"),
        ("They build the wall","They built the wall"),
        ("We swim in lake","We swam in lake"),
        ("She writes a letter","She wrote a letter"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is first","It is last"),
    ],
    "passive": [
        ("The cat chased mouse","The mouse was chased"),
        ("John broke window","The window was broken"),
        ("The chef cooked meal","The meal was cooked"),
        ("The storm destroyed house","The house was destroyed"),
        ("The artist painted picture","The picture was painted"),
        ("The king signed document","The document was signed"),
    ],
    "causation": [
        ("The rain falls down","The ground gets wet"),
        ("The fire burns long","The wood turns to ash"),
        ("The sun heats earth","The ice starts melting"),
        ("The wind blows branches","The leaves start falling"),
        ("The child cries loud","The mother comes running"),
        ("The glass breaks","The water spills out"),
    ],
    "question": [
        ("She is tired today","Is she tired today"),
        ("He can swim well","Can he swim well"),
        ("They went to market","Did they go to market"),
        ("The dog is hungry","Is the dog hungry"),
        ("She wrote the letter","Did she write letter"),
        ("He knows the answer","Does he know answer"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
    ],
}

# Same 12 cases as Day 124
RANKING_CASES = [
    {"category":"capitals","prompt":"The capital city of France is",
     "correct":["Paris"],"plausible":["London","Rome","Berlin","Madrid"],
     "wrong":["banana","running","very","because"]},
    {"category":"capitals","prompt":"The capital city of Japan is",
     "correct":["Tokyo"],"plausible":["Osaka","Beijing","Seoul","Bangkok"],
     "wrong":["quietly","ocean","third","green"]},
    {"category":"capitals","prompt":"The capital city of Germany is",
     "correct":["Berlin"],"plausible":["Frankfurt","Vienna","Warsaw","Amsterdam"],
     "wrong":["walked","stone","never","that"]},
    {"category":"languages","prompt":"The official language of Brazil is",
     "correct":["Portuguese"],"plausible":["Spanish","Italian","French","English"],
     "wrong":["purple","running","before","cold"]},
    {"category":"languages","prompt":"The official language of Egypt is",
     "correct":["Arabic"],"plausible":["Hebrew","Turkish","Persian","Urdu"],
     "wrong":["quickly","stone","later","because"]},
    {"category":"hypernyms","prompt":"A poodle is a type of",
     "correct":["dog","animal","mammal"],"plausible":["cat","rabbit","horse","pet"],
     "wrong":["quickly","stone","because","red"]},
    {"category":"hypernyms","prompt":"A rose is a type of",
     "correct":["flower","plant","bloom"],"plausible":["tree","bush","grass","weed"],
     "wrong":["capital","walked","never","cold"]},
    {"category":"hypernyms","prompt":"A hammer is a type of",
     "correct":["tool","instrument","implement"],"plausible":["machine","device","appliance","weapon"],
     "wrong":["flower","language","quickly","green"]},
    {"category":"antonyms","prompt":"The opposite of hot is",
     "correct":["cold","cool","frigid"],"plausible":["warm","lukewarm","mild","chilly"],
     "wrong":["Paris","running","because","stone"]},
    {"category":"antonyms","prompt":"The opposite of happy is",
     "correct":["sad","unhappy","miserable"],"plausible":["angry","bored","worried","tired"],
     "wrong":["capital","walked","flower","green"]},
    {"category":"tense","prompt":"Yesterday he",
     "correct":["walked","ran","ate","built","saw"],"plausible":["walk","run","eat","build","see"],
     "wrong":["Paris","flower","because","cold"]},
    {"category":"gender","prompt":"The king and",
     "correct":["queen","princess","duchess","empress"],"plausible":["prince","knight","lord","earl"],
     "wrong":["banana","quickly","walked","stone"]},
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers

# ── Extract d_k direction (L23 H6, same as Finding 40 / Days 116-119) ──────────
print("Extracting d_k from L23 H6 ...")
L_dk, H_dk = 23, 6
try:
    layer_module = model.model.layers[L_dk]
    W_k_full     = layer_module.self_attn.k_proj.weight.data.cpu().numpy()
    head_dim     = model.config.hidden_size // model.config.num_attention_heads
    n_kv_heads   = model.config.num_key_value_heads
    n_q_heads    = model.config.num_attention_heads
    kv_per_group = n_q_heads // n_kv_heads
    kv_head_idx  = H_dk // kv_per_group
    W_k_head = W_k_full[kv_head_idx*head_dim:(kv_head_idx+1)*head_dim, :]
    U_k, S_k, Vt_k = np.linalg.svd(W_k_head, full_matrices=False)
    v1 = Vt_k[0, :]  # right singular vector corresponding to largest singular value
    d_k = v1.astype(np.float32)
    print(f"  d_k extracted: ||d_k||={np.linalg.norm(d_k):.4f}")
except Exception as e:
    print(f"  WARNING: could not extract d_k — {e}")
    d_k = np.zeros(hidden_size, np.float32)

# ── Compute T2 axes ────────────────────────────────────────────────────────────
print("Computing T2 axes ...")
t2_axes = {}
for ax_name in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax_name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(ax_name, []):
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            h2 = o2.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            d = h2-h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d/n)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
    nv = np.linalg.norm(v)
    t2_axes[ax_name] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

def get_hidden_states(text):
    """Get all layer hidden states for last token position."""
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                for L in range(n_layers + 1)}
    except:
        return {}

def t2_vector(hs_dict):
    """Compute 12D T2 projection from hidden state dict."""
    v = np.zeros(12, np.float32)
    for k, ax_name in enumerate(AXIS_NAMES_12):
        L = DAY78_LAYERS[ax_name]
        h = hs_dict.get(L, np.zeros(hidden_size))
        hn = normed(h)
        v[k] = float(np.dot(hn, t2_axes[ax_name]))
    return v

def dk_score(hs_dict):
    """d_k projection of normed hidden state at L23."""
    h = hs_dict.get(L_dk, np.zeros(hidden_size))
    return float(abs(np.dot(normed(h), d_k)))

def t2_cosine(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ── Run ranking experiment ────────────────────────────────────────────────────
print("=" * 72)
print("Exp 1: d_k score at candidate position in context")
print("       Hypothesis: correct entity has higher d_k in retrieval context")
print("=" * 72)
print()

all_results = []
for case in RANKING_CASES:
    prompt    = case["prompt"]
    correct   = case["correct"]
    plausible = case["plausible"]
    wrong     = case["wrong"]
    all_cands = correct + plausible + wrong

    # Context anchor T2 (last token of prompt)
    ctx_hs = get_hidden_states(prompt)
    ctx_t2 = t2_vector(ctx_hs)

    # Isolated T2 (Day 124 Method A — known to work as category filter)
    iso_t2  = {w: t2_vector(get_hidden_states(" " + w)) for w in all_cands}
    iso_sim = {w: t2_cosine(iso_t2[w], ctx_t2) for w in all_cands}

    # d_k at last token of "[prompt] [candidate]"
    cand_hs  = {w: get_hidden_states(prompt + " " + w) for w in all_cands}
    dk_scores = {w: dk_score(cand_hs[w]) for w in all_cands}

    # T2 of candidate-in-context (Day 124 Method B)
    ctx_t2_cand = {w: t2_vector(cand_hs[w]) for w in all_cands}
    ctx_sim     = {w: t2_cosine(ctx_t2_cand[w], ctx_t2) for w in all_cands}

    def best_rank(scores_dict, correct_list, higher_is_better=True):
        ranked = sorted(all_cands, key=lambda w: (-scores_dict[w] if higher_is_better else scores_dict[w]))
        for i, w in enumerate(ranked):
            if w in correct_list: return i + 1
        return len(all_cands) + 1

    rank_iso  = best_rank(iso_sim, correct)
    rank_dk   = best_rank(dk_scores, correct)
    rank_ctx  = best_rank(ctx_sim, correct)

    # Combined: T2_iso + dk (normalize each to [0,1] first)
    iso_vals = np.array([iso_sim[w] for w in all_cands])
    dk_vals  = np.array([dk_scores[w] for w in all_cands])
    iso_n = (iso_vals - iso_vals.min()) / (iso_vals.max()-iso_vals.min() + 1e-8)
    dk_n  = (dk_vals  - dk_vals.min())  / (dk_vals.max()-dk_vals.min()  + 1e-8)
    # Alpha sweep
    best_alpha = 0.5; best_rank_combo = len(all_cands) + 1
    for alpha in np.arange(0.0, 1.05, 0.1):
        combo = alpha * iso_n + (1-alpha) * dk_n
        combo_scores = {w: float(combo[i]) for i, w in enumerate(all_cands)}
        r = best_rank(combo_scores, correct)
        if r < best_rank_combo:
            best_rank_combo = r; best_alpha = float(alpha)

    # Mean scores by tier
    mean_cor_dk  = np.mean([dk_scores[w] for w in correct])
    mean_pla_dk  = np.mean([dk_scores[w] for w in plausible])
    mean_wrg_dk  = np.mean([dk_scores[w] for w in wrong])
    dk_order     = bool(mean_cor_dk > mean_pla_dk > mean_wrg_dk)

    print(f"  [{case['category']}] {prompt!r}")
    print(f"    d_k: correct={mean_cor_dk:.4f}  plausible={mean_pla_dk:.4f}  wrong={mean_wrg_dk:.4f}  "
          f"order={'✓' if dk_order else '✗'}  rank={rank_dk}")
    print(f"    T2_iso rank={rank_iso}  d_k rank={rank_dk}  ctx rank={rank_ctx}  "
          f"best_combo rank={best_rank_combo} (α={best_alpha:.1f})")
    print()

    all_results.append({
        "category": case["category"], "prompt": prompt,
        "rank_iso": rank_iso, "rank_dk": rank_dk,
        "rank_ctx": rank_ctx, "rank_combo": best_rank_combo,
        "best_alpha": best_alpha,
        "dk_order": dk_order,
        "dk_means": {"correct": float(mean_cor_dk),
                     "plausible": float(mean_pla_dk),
                     "wrong": float(mean_wrg_dk)},
        "iso_sims": {w: float(iso_sim[w]) for w in all_cands},
        "dk_scores": {w: float(dk_scores[w]) for w in all_cands},
    })

# ── Aggregate ───────────────────────────────────────────────────────────────────
n  = len(all_results)
mrr = lambda ranks: float(np.mean([1/r for r in ranks]))
ranks_iso   = [r["rank_iso"]   for r in all_results]
ranks_dk    = [r["rank_dk"]    for r in all_results]
ranks_ctx   = [r["rank_ctx"]   for r in all_results]
ranks_combo = [r["rank_combo"] for r in all_results]
random_mrr  = sum(1/i for i in range(1, len(RANKING_CASES[0]["correct"]) +
                  len(RANKING_CASES[0]["plausible"]) + len(RANKING_CASES[0]["wrong"]) + 1)) / \
              (len(RANKING_CASES[0]["correct"]) + len(RANKING_CASES[0]["plausible"]) +
               len(RANKING_CASES[0]["wrong"]))

n_dk_order = sum(1 for r in all_results if r["dk_order"])

print("=" * 72)
print("Aggregate Results — MRR Comparison")
print("=" * 72)
print(f"""
  Method              MRR     mean_rank   correct_order
  T2_iso (Day124)    {mrr(ranks_iso):.4f}    {np.mean(ranks_iso):.2f}        {sum(1 for r in ranks_iso if r==1)}/{n}=rank-1
  d_k only           {mrr(ranks_dk):.4f}    {np.mean(ranks_dk):.2f}        {n_dk_order}/{n} order correct
  T2_ctx (Day124)    {mrr(ranks_ctx):.4f}    {np.mean(ranks_ctx):.2f}
  T2+d_k combined   {mrr(ranks_combo):.4f}    {np.mean(ranks_combo):.2f}
  Random baseline    {random_mrr:.4f}    {(len(RANKING_CASES[0]['correct'])+len(RANKING_CASES[0]['plausible'])+len(RANKING_CASES[0]['wrong'])+1)/2:.2f}
""")

print("Per-category (d_k ordering):")
print(f"  {'category':>12}  {'dk_order':>9}  {'rank_dk':>8}  {'rank_combo':>12}  {'dk_cor':>8}  {'dk_pla':>8}  {'dk_wrg':>8}")
print(f"  {'-'*70}")
cats_seen = []
for r in all_results:
    if r["category"] not in cats_seen: cats_seen.append(r["category"])
for cat in cats_seen:
    cr = [r for r in all_results if r["category"] == cat]
    print(f"  {cat:>12}  {sum(1 for r in cr if r['dk_order'])}/{len(cr):>3}        "
          f"{np.mean([r['rank_dk'] for r in cr]):>8.2f}  "
          f"{np.mean([r['rank_combo'] for r in cr]):>12.2f}  "
          f"{np.mean([r['dk_means']['correct'] for r in cr]):>8.4f}  "
          f"{np.mean([r['dk_means']['plausible'] for r in cr]):>8.4f}  "
          f"{np.mean([r['dk_means']['wrong'] for r in cr]):>8.4f}")

# ── Optimal alpha analysis ──────────────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Global alpha sweep (T2_iso + d_k combined across all cases)")
print("=" * 72)
print()
all_cors  = [case["correct"] for case in RANKING_CASES]
all_cands_list = [case["correct"]+case["plausible"]+case["wrong"] for case in RANKING_CASES]

print(f"  alpha  MRR_global  mean_rank")
print(f"  {'─'*36}")
best_global_mrr = 0; best_global_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.1):
    ranks = []
    for i, r in enumerate(all_results):
        cands = all_cands_list[i]
        iso_v = np.array([float(r["iso_sims"][w])  for w in cands])
        dk_v  = np.array([float(r["dk_scores"][w]) for w in cands])
        iso_n = (iso_v - iso_v.min()) / (iso_v.max()-iso_v.min() + 1e-8)
        dk_n  = (dk_v  - dk_v.min())  / (dk_v.max()-dk_v.min()  + 1e-8)
        combo = alpha * iso_n + (1-alpha) * dk_n
        ranked = sorted(cands, key=lambda w: -combo[cands.index(w)])
        rank = next((j+1 for j,w in enumerate(ranked) if w in all_cors[i]), len(cands)+1)
        ranks.append(rank)
    m = float(np.mean([1/r for r in ranks]))
    if m > best_global_mrr: best_global_mrr = m; best_global_alpha = float(alpha)
    print(f"  {alpha:.1f}    {m:.4f}      {np.mean(ranks):.2f}")

print(f"\n  Best global alpha: {best_global_alpha:.1f}  MRR: {best_global_mrr:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 125 Summary — T2 + d_k Combined Ranking")
print("=" * 72)

dk_useful     = mrr(ranks_dk) > random_mrr
combo_useful  = best_global_mrr > mrr(ranks_iso)
dk_adds_value = combo_useful

print(f"""
  d_k as standalone ranker:   MRR={mrr(ranks_dk):.4f}  {'ABOVE' if dk_useful else 'BELOW'} random ({random_mrr:.4f})
  T2_iso as standalone:       MRR={mrr(ranks_iso):.4f}  (Day 124 confirmed above random)
  T2+d_k combined (α={best_global_alpha:.1f}):    MRR={best_global_mrr:.4f}

  Does d_k ADD VALUE to T2?  {'YES — combination improves MRR' if combo_useful else 'NO — T2 alone is better'}
  Is d_k useful alone?       {'YES — above random' if dk_useful else 'NO — below or at random'}

  d_k ordering (correct > plausible > wrong): {n_dk_order}/{n} cases

  VERDICT:
  {'→ d_k distinguishes candidates within semantic categories: retrieval signal is specific' if dk_useful and n_dk_order >= n//2 else
   '→ d_k does NOT distinguish candidates within semantic categories at last token' if not dk_useful else
   '→ d_k partially useful: above random but ordering inconsistent'}

  TruthSpace Two-Structure Picture:
  {'→ T2 = category filter + d_k = within-category selector: CONFIRMED' if dk_useful and combo_useful else
   '→ T2 = category filter only. d_k at candidate last-token does not provide additional within-category signal.' if not dk_useful else
   '→ Partial: T2 + d_k combination is better than T2 alone, but d_k signal is weak'}

  Note: Day 117 showed d_k is retrieval-specific at ENTITY POSITION (not last-token).
  {'→ The entity-position d_k (not measured here) may still provide within-category signal.' if not dk_useful else ''}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "mrr_iso": mrr(ranks_iso), "mrr_dk": mrr(ranks_dk),
        "mrr_ctx": mrr(ranks_ctx), "mrr_combo": best_global_mrr,
        "random_mrr": random_mrr,
        "best_global_alpha": best_global_alpha,
        "n_dk_order": n_dk_order,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 125 complete.")
