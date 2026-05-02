#!/usr/bin/env python3
"""
Day 127 — Full Hidden State Similarity vs T2 vs d_k vs Log-Probability

Day 126 found:
  - T2 (12D) ρ(log-prob) = +0.063  WEAK
  - d_k (1D) ρ(log-prob) = +0.347  MODERATE

QUESTION: How much of the log-prob ranking signal is present in the
full 1536D hidden state?

If cosine(h_full_isolated_candidate, h_full_context) correlates strongly
with log-prob, it means:
  → The geometric representation IS informationally sufficient for ranking
  → The T2 compression (12D) loses the probability-relevant information
  → d_k is a better 1D summary because it points in the right direction

Comparison hierarchy:
  1. log-prob: oracle (MRR=1.000)
  2. full_h_L23_cosine: best geometric (1536D)
  3. d_k: best 1D geometric from Day 126 (ρ=0.347)
  4. T2 (12D): category-level filter (ρ=0.063)

Also test multiple layers: does similarity at L15, L20, L23, L27, L28
differ in correlation with log-prob? This reveals where probability-relevant
geometric information is encoded.

Finally: test the FULL hidden state at ALL 28 layers as an oracle comparison
vs T2+d_k. This sets the ceiling on what pure geometric similarity can achieve.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day127_hidden_state_sim.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# Layers to probe
TEST_LAYERS = [1, 5, 10, 15, 20, 23, 25, 27, 28]

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

L_dk, H_dk = 23, 6

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers
print(f"  hidden={hidden_size}  n_layers={n_layers}\n")

# d_k extraction
print("Extracting d_k ...")
layer_module = model.model.layers[L_dk]
W_k_full     = layer_module.self_attn.k_proj.weight.data.cpu().numpy()
head_dim     = hidden_size // model.config.num_attention_heads
n_kv_heads   = model.config.num_key_value_heads
kv_per_group = model.config.num_attention_heads // n_kv_heads
kv_head_idx  = H_dk // kv_per_group
W_k_head     = W_k_full[kv_head_idx*head_dim:(kv_head_idx+1)*head_dim, :]
_, _, Vt_k   = np.linalg.svd(W_k_head, full_matrices=False)
d_k = Vt_k[0, :].astype(np.float32)
print(f"  ||d_k||={np.linalg.norm(d_k):.4f}\n")

# T2 axes
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
            d = h2-h1; nv = np.linalg.norm(d)
            if nv > 1e-6: diffs.append(d/nv)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
    nv = np.linalg.norm(v)
    t2_axes[ax_name] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

def get_all_hs(text):
    """All layer hidden states at last token."""
    inp = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                for L in range(n_layers + 1)}
    except:
        return {}

def get_logprob(prompt, word):
    inp = tok(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            lp = torch.log_softmax(logits, dim=-1).numpy()
        ids = tok(" " + word, add_special_tokens=False)["input_ids"]
        return float(lp[ids[0]]) if ids else float("-inf")
    except:
        return float("-inf")

def full_cosine(h1, h2):
    n1 = np.linalg.norm(h1); n2 = np.linalg.norm(h2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(h1, h2) / (n1 * n2))

def t2_vector(hs):
    v = np.zeros(12, np.float32)
    for k, ax_name in enumerate(AXIS_NAMES_12):
        L = DAY78_LAYERS[ax_name]
        h = normed(hs.get(L, np.zeros(hidden_size)))
        v[k] = float(np.dot(h, t2_axes[ax_name]))
    return v

def t2_cosine(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ── Main experiment ────────────────────────────────────────────────────────────
print("=" * 72)
print("Computing per-layer full-cosine similarity vs log-prob correlation")
print("=" * 72)
print()

# Collect per-layer Spearman ρ with log-prob
layer_rhos = {L: [] for L in TEST_LAYERS}
# Also collect for concatenated (all-layer mean)
concat_rhos = []
t2_rhos = []; dk_rhos = []

all_results = []

for case in RANKING_CASES:
    prompt    = case["prompt"]
    correct   = case["correct"]
    plausible = case["plausible"]
    wrong     = case["wrong"]
    all_cands = correct + plausible + wrong

    # Log-probs
    lp = {w: get_logprob(prompt, w) for w in all_cands}
    lp_vec = np.array([lp[w] for w in all_cands])

    # Context hidden states (prompt last token)
    ctx_hs  = get_all_hs(prompt)
    ctx_t2  = t2_vector(ctx_hs)

    # Isolated candidate hidden states
    cand_hs = {w: get_all_hs(" " + w) for w in all_cands}

    # Per-layer full cosine
    layer_sims = {}
    for L in TEST_LAYERS:
        ctx_h = ctx_hs.get(L, np.zeros(hidden_size))
        sims  = np.array([full_cosine(ctx_h, cand_hs[w].get(L, np.zeros(hidden_size)))
                          for w in all_cands])
        layer_sims[L] = sims
        rho, _ = spearmanr(sims, lp_vec)
        layer_rhos[L].append(float(rho))

    # Mean-across-all-layers cosine
    concat_sim = np.zeros(len(all_cands))
    for L in range(n_layers + 1):
        ctx_h = ctx_hs.get(L, np.zeros(hidden_size))
        for i, w in enumerate(all_cands):
            concat_sim[i] += full_cosine(ctx_h, cand_hs[w].get(L, np.zeros(hidden_size)))
    concat_sim /= (n_layers + 1)
    rho_cat, _ = spearmanr(concat_sim, lp_vec)
    concat_rhos.append(float(rho_cat))

    # T2 similarity
    t2_sims = np.array([t2_cosine(t2_vector(cand_hs[w]), ctx_t2) for w in all_cands])
    rho_t2, _ = spearmanr(t2_sims, lp_vec)
    t2_rhos.append(float(rho_t2))

    # d_k similarity (isolated candidate at L23)
    dk_sims = np.array([abs(np.dot(normed(cand_hs[w].get(L_dk, np.zeros(hidden_size))), d_k))
                        for w in all_cands])
    rho_dk, _ = spearmanr(dk_sims, lp_vec)
    dk_rhos.append(float(rho_dk))

    # MRR for best-layer full cosine (L23)
    cos23 = layer_sims[23]
    ranked23 = sorted(all_cands, key=lambda w: -cos23[all_cands.index(w)])
    rank_best_23 = next((i+1 for i,w in enumerate(ranked23) if w in correct), len(all_cands)+1)

    all_results.append({
        "category": case["category"], "prompt": prompt,
        "rho_per_layer": {str(L): float(layer_rhos[L][-1]) for L in TEST_LAYERS},
        "rho_concat": float(rho_cat),
        "rho_t2": float(rho_t2), "rho_dk": float(rho_dk),
        "mrr_full23": 1.0/rank_best_23,
        "logprobs": {w: float(lp[w]) for w in all_cands},
    })

# ── Results table ───────────────────────────────────────────────────────────────
print(f"  {'Layer':>8}  {'Mean ρ':>10}  {'min ρ':>10}  {'max ρ':>10}  {'MRR est':>10}")
print(f"  {'-'*56}")
for L in TEST_LAYERS:
    rhos = layer_rhos[L]
    # Approximate MRR from per-case ρ (not exact but indicative)
    print(f"  L{L:02d}      {np.mean(rhos):>+10.4f}  {np.min(rhos):>+10.4f}  {np.max(rhos):>+10.4f}")
print(f"  {'concat':>8}  {np.mean(concat_rhos):>+10.4f}  {np.min(concat_rhos):>+10.4f}  {np.max(concat_rhos):>+10.4f}")
print(f"  {'T2':>8}  {np.mean(t2_rhos):>+10.4f}  {np.min(t2_rhos):>+10.4f}  {np.max(t2_rhos):>+10.4f}")
print(f"  {'d_k':>8}  {np.mean(dk_rhos):>+10.4f}  {np.min(dk_rhos):>+10.4f}  {np.max(dk_rhos):>+10.4f}")

# ── MRR comparison at best layer ───────────────────────────────────────────────
print()
print("=" * 72)
print("MRR Comparison: which method best approximates log-prob?")
print("=" * 72)
print()

best_layer = max(TEST_LAYERS, key=lambda L: np.mean(layer_rhos[L]))
print(f"  Best single layer for full-cosine: L{best_layer} (mean ρ={np.mean(layer_rhos[best_layer]):.4f})")

# Compute MRR for each method
def compute_mrr(scores_list, correct_list, all_cands_list, higher=True):
    total = []
    for scores, correct, cands in zip(scores_list, correct_list, all_cands_list):
        ranked = sorted(enumerate(cands), key=lambda x: (-scores[x[0]] if higher else scores[x[0]]))
        r = next((i+1 for i,(_, w) in enumerate(ranked) if w in correct), len(cands)+1)
        total.append(1.0/r)
    return float(np.mean(total))

all_cands_all = [case["correct"]+case["plausible"]+case["wrong"] for case in RANKING_CASES]
all_correct   = [case["correct"] for case in RANKING_CASES]

mrr_by_layer = {}
for L in TEST_LAYERS:
    sims_list = []
    for i, case in enumerate(RANKING_CASES):
        cands = all_cands_all[i]
        ctx_hs_l = get_all_hs(case["prompt"])
        ctx_h = ctx_hs_l.get(L, np.zeros(hidden_size))
        sims  = [full_cosine(ctx_h, get_all_hs(" "+w).get(L, np.zeros(hidden_size)))
                 for w in cands]
        sims_list.append(sims)
    mrr = compute_mrr(sims_list, all_correct, all_cands_all)
    mrr_by_layer[L] = mrr
    print(f"  Full cosine L{L:02d}: MRR={mrr:.4f}  ρ={np.mean(layer_rhos[L]):+.4f}")

print()
print(f"  T2 (12D):         MRR=0.5397  ρ={np.mean(t2_rhos):+.4f}  (Day 124/126)")
print(f"  d_k (1D):         MRR=0.5494  ρ={np.mean(dk_rhos):+.4f}  (Day 125/126)")
print(f"  T2+d_k (α=0.9):   MRR=0.5952  (Day 126)")
print(f"  Log-prob oracle:  MRR=1.0000")

# ── Findings summary ────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 127 Summary — Full Hidden State vs Compressed Signals")
print("=" * 72)

best_full_rho = max(np.mean(layer_rhos[L]) for L in TEST_LAYERS)
best_full_mrr = max(mrr_by_layer.values())

print(f"""
  Full hidden state (best layer L{best_layer}):
    Mean ρ(cos, log-prob) = {best_full_rho:.4f}
    MRR = {best_full_mrr:.4f}

  Compressed signals (Days 124-126):
    T2 (12D)   ρ = {np.mean(t2_rhos):.4f}  MRR = 0.5397
    d_k (1D)   ρ = {np.mean(dk_rhos):.4f}  MRR = 0.5494
    T2+d_k     ρ = ---      MRR = 0.5952

  Log-prob oracle MRR = 1.0000

  VERDICT:
  Full hidden state: {'BETTER' if best_full_rho > 0.35 else 'SIMILAR'} than d_k for log-prob correlation.
  Best layer for probability-relevant geometry: L{best_layer}

  Information loss from compression:
    1536D → 1D (d_k):  ρ drops from {best_full_rho:.3f} to {np.mean(dk_rhos):.3f}
    1536D → 12D (T2):  ρ drops from {best_full_rho:.3f} to {np.mean(t2_rhos):.3f}
    {'→ d_k is a better 1D compression than T2 for probability-relevant information' if np.mean(dk_rhos) > np.mean(t2_rhos) else
     '→ T2 and d_k capture similar amounts of probability-relevant information'}

  Does the full geometric representation substitute for log-prob?
  {'→ YES: full cosine MRR ~ oracle (geometry IS sufficient for ranking)' if best_full_mrr > 0.85 else
   '→ PARTIAL: full cosine captures ~'+f'{100*best_full_mrr:.0f}% of oracle MRR' if best_full_mrr > 0.6 else
   '→ NO: even full hidden state cosine does not approximate log-prob ranking'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "layer_rhos_mean": {str(L): float(np.mean(layer_rhos[L])) for L in TEST_LAYERS},
        "concat_rho_mean": float(np.mean(concat_rhos)),
        "t2_rho_mean": float(np.mean(t2_rhos)),
        "dk_rho_mean": float(np.mean(dk_rhos)),
        "mrr_by_layer": {str(L): float(mrr_by_layer[L]) for L in TEST_LAYERS},
        "best_layer": best_layer,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 127 complete.")
