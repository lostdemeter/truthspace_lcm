#!/usr/bin/env python3
"""
Day 124 — T2 Semantic Ranking: Can T2 Distinguish Correct vs Wrong Completions?

QUESTION: Can T2 addresses be used as a SEMANTIC FILTER for generation?
Given a prompt, do correct completion tokens cluster closer in T2 space
than incorrect ones?

HYPOTHESIS (TruthSpace): The T2 12D address encodes the semantic role
a token should play. In a factual retrieval context, the correct answer
(Paris) should have a T2 address similar to other correct answers in
the same category (capitals), while wrong answers (banana) should have
different T2 addresses.

TEST STRUCTURE: For 5 factual question categories:
  capitals:    "The capital of France is ___"
  languages:   "The official language of Brazil is ___"
  colors:      "The color of a ripe tomato is ___"
  opposites:   "The opposite of hot is ___"
  categories:  "A poodle is a type of ___"

For each prompt, present 3 candidate tiers:
  Tier 1 (correct):    the actual correct answer
  Tier 2 (plausible):  semantically related but wrong (another capital, another language)
  Tier 3 (wrong):      semantically unrelated (completely different category)

Metrics:
  1. T2 cosine similarity between candidate and context last-token
  2. T2 Euclidean distance between candidates in T2 space
  3. Rank correlation: does T2 rank correct > plausible > wrong?

Also test: does conditioning on the CONTEXT (last-token of the prompt)
vs the candidate IN ISOLATION produce different rankings?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day124_t2_semantic_ranking.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

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

# Ranking test cases
# Each entry: prompt, correct, plausible_wrong, semantic_wrong
RANKING_CASES = [
    # ── Capitals ────────────────────────────────────────────────────────────────
    {
        "category": "capitals",
        "prompt": "The capital city of France is",
        "correct":   ["Paris"],
        "plausible": ["London","Rome","Berlin","Madrid"],
        "wrong":     ["banana","running","very","because"],
    },
    {
        "category": "capitals",
        "prompt": "The capital city of Japan is",
        "correct":   ["Tokyo"],
        "plausible": ["Osaka","Beijing","Seoul","Bangkok"],
        "wrong":     ["quietly","ocean","third","green"],
    },
    {
        "category": "capitals",
        "prompt": "The capital city of Germany is",
        "correct":   ["Berlin"],
        "plausible": ["Frankfurt","Vienna","Warsaw","Amsterdam"],
        "wrong":     ["walked","stone","never","that"],
    },
    # ── Languages ───────────────────────────────────────────────────────────────
    {
        "category": "languages",
        "prompt": "The official language of Brazil is",
        "correct":   ["Portuguese"],
        "plausible": ["Spanish","Italian","French","English"],
        "wrong":     ["purple","running","before","cold"],
    },
    {
        "category": "languages",
        "prompt": "The official language of Egypt is",
        "correct":   ["Arabic"],
        "plausible": ["Hebrew","Turkish","Persian","Urdu"],
        "wrong":     ["quickly","stone","later","because"],
    },
    # ── Hypernyms ────────────────────────────────────────────────────────────────
    {
        "category": "hypernyms",
        "prompt": "A poodle is a type of",
        "correct":   ["dog","animal","mammal"],
        "plausible": ["cat","rabbit","horse","pet"],
        "wrong":     ["quickly","stone","because","red"],
    },
    {
        "category": "hypernyms",
        "prompt": "A rose is a type of",
        "correct":   ["flower","plant","bloom"],
        "plausible": ["tree","bush","grass","weed"],
        "wrong":     ["capital","walked","never","cold"],
    },
    {
        "category": "hypernyms",
        "prompt": "A hammer is a type of",
        "correct":   ["tool","instrument","implement"],
        "plausible": ["machine","device","appliance","weapon"],
        "wrong":     ["flower","language","quickly","green"],
    },
    # ── Antonyms ──────────────────────────────────────────────────────────────────
    {
        "category": "antonyms",
        "prompt": "The opposite of hot is",
        "correct":   ["cold","cool","frigid"],
        "plausible": ["warm","lukewarm","mild","chilly"],
        "wrong":     ["Paris","running","because","stone"],
    },
    {
        "category": "antonyms",
        "prompt": "The opposite of happy is",
        "correct":   ["sad","unhappy","miserable"],
        "plausible": ["angry","bored","worried","tired"],
        "wrong":     ["capital","walked","flower","green"],
    },
    # ── Tense ────────────────────────────────────────────────────────────────────
    {
        "category": "tense",
        "prompt": "Yesterday he",
        "correct":   ["walked","ran","ate","built","saw"],
        "plausible": ["walk","run","eat","build","see"],
        "wrong":     ["Paris","flower","because","cold"],
    },
    # ── Gender ─────────────────────────────────────────────────────────────────
    {
        "category": "gender",
        "prompt": "The king and",
        "correct":   ["queen","princess","duchess","empress"],
        "plausible": ["prince","knight","lord","earl"],
        "wrong":     ["banana","quickly","walked","stone"],
    },
]

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

def phi_bin(x):
    if   x >  INV_PHI:  return "H"
    elif x < -INV_PHI2: return "L"
    else: return "U"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

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

def get_t2_vector(text, use_last_token=True, context_prefix=None):
    """Get 12D T2 projection vector for a word/phrase.
    If context_prefix is given, the word is placed after the prefix.
    """
    if context_prefix is not None:
        full_text = context_prefix + " " + text
    else:
        full_text = " " + text
    inp = tok(full_text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        t2_vec = np.zeros(12, dtype=np.float32)
        for k, ax_name in enumerate(AXIS_NAMES_12):
            L  = DAY78_LAYERS[ax_name]
            h  = normed(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
            t2_vec[k] = float(np.dot(h, t2_axes[ax_name]))
        return t2_vec
    except:
        return np.zeros(12, dtype=np.float32)

def get_context_t2(prompt_text):
    """Get T2 vector of the last token of the prompt (context anchor)."""
    inp = tok(prompt_text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        t2_vec = np.zeros(12, dtype=np.float32)
        for k, ax_name in enumerate(AXIS_NAMES_12):
            L = DAY78_LAYERS[ax_name]
            h = normed(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
            t2_vec[k] = float(np.dot(h, t2_axes[ax_name]))
        return t2_vec
    except:
        return np.zeros(12, dtype=np.float32)

def t2_cosine(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def t2_euclidean(v1, v2):
    return float(np.linalg.norm(v1 - v2))

# ── Main experiment ────────────────────────────────────────────────────────────
print("=" * 72)
print("T2 Semantic Ranking Experiment")
print("Method A: Isolated word T2 vectors")
print("Method B: Word-in-context T2 vectors (appended to prompt)")
print("=" * 72)
print()

all_results = []
correct_ranks_A = []; correct_ranks_B = []  # rank of correct in full candidate list

for case in RANKING_CASES:
    prompt    = case["prompt"]
    correct   = case["correct"]
    plausible = case["plausible"]
    wrong     = case["wrong"]
    all_cands = correct + plausible + wrong

    # Context anchor: last-token T2 of the prompt
    ctx_t2 = get_context_t2(prompt)

    # Method A: isolated word T2 vectors
    iso_vecs = {w: get_t2_vector(w) for w in all_cands}
    # Method B: word in context T2 vectors
    ctx_vecs = {w: get_t2_vector(w, context_prefix=prompt) for w in all_cands}

    # Compute similarities to context anchor
    iso_sims = {w: t2_cosine(iso_vecs[w], ctx_t2) for w in all_cands}
    ctx_sims = {w: t2_cosine(ctx_vecs[w], ctx_t2) for w in all_cands}

    # Rank all candidates by similarity (higher = more similar to context)
    iso_ranked = sorted(all_cands, key=lambda w: -iso_sims[w])
    ctx_ranked = sorted(all_cands, key=lambda w: -ctx_sims[w])

    # Rank of BEST correct answer in each ranking
    def best_rank(ranked_list, correct_list):
        for i, w in enumerate(ranked_list):
            if w in correct_list: return i + 1  # 1-indexed
        return len(ranked_list) + 1

    rank_A = best_rank(iso_ranked, correct)
    rank_B = best_rank(ctx_ranked, correct)
    correct_ranks_A.append(rank_A)
    correct_ranks_B.append(rank_B)

    # Mean similarity by tier
    mean_correct_A   = np.mean([iso_sims[w] for w in correct])
    mean_plausible_A = np.mean([iso_sims[w] for w in plausible])
    mean_wrong_A     = np.mean([iso_sims[w] for w in wrong])
    mean_correct_B   = np.mean([ctx_sims[w] for w in correct])
    mean_plausible_B = np.mean([ctx_sims[w] for w in plausible])
    mean_wrong_B     = np.mean([ctx_sims[w] for w in wrong])

    # Rank order check: correct > plausible > wrong?
    order_A = (mean_correct_A > mean_plausible_A > mean_wrong_A)
    order_B = (mean_correct_B > mean_plausible_B > mean_wrong_B)

    print(f"  [{case['category']}] {prompt!r}")
    print(f"    Correct:   {correct[:2]}")
    print(f"    Plausible: {plausible[:3]}")
    print(f"    Wrong:     {wrong[:2]}")
    print(f"    ── Method A (isolated) ──────────────────────────────")
    print(f"    correct={mean_correct_A:+.4f}  plausible={mean_plausible_A:+.4f}  wrong={mean_wrong_A:+.4f}  "
          f"order={'✓' if order_A else '✗'}  best_rank={rank_A}")
    print(f"    ── Method B (in-context) ────────────────────────────")
    print(f"    correct={mean_correct_B:+.4f}  plausible={mean_plausible_B:+.4f}  wrong={mean_wrong_B:+.4f}  "
          f"order={'✓' if order_B else '✗'}  best_rank={rank_B}")
    print()

    all_results.append({
        "category": case["category"],
        "prompt": prompt,
        "iso_sims": iso_sims, "ctx_sims": ctx_sims,
        "rank_A": rank_A, "rank_B": rank_B,
        "order_A": bool(order_A), "order_B": bool(order_B),
        "means_A": {"correct": mean_correct_A, "plausible": mean_plausible_A, "wrong": mean_wrong_A},
        "means_B": {"correct": mean_correct_B, "plausible": mean_plausible_B, "wrong": mean_wrong_B},
    })

# ── Aggregate statistics ────────────────────────────────────────────────────────
print("=" * 72)
print("Aggregate Results")
print("=" * 72)

n_cases   = len(all_results)
n_order_A = sum(1 for r in all_results if r["order_A"])
n_order_B = sum(1 for r in all_results if r["order_B"])

mean_rank_A = float(np.mean(correct_ranks_A))
mean_rank_B = float(np.mean(correct_ranks_B))
n_cands     = len(RANKING_CASES[0]["correct"]) + len(RANKING_CASES[0]["plausible"]) + len(RANKING_CASES[0]["wrong"])

# MRR (Mean Reciprocal Rank)
mrr_A = float(np.mean([1/r for r in correct_ranks_A]))
mrr_B = float(np.mean([1/r for r in correct_ranks_B]))

print(f"""
  Cases tested: {n_cases}
  Total candidates per case: {len(RANKING_CASES[0]['correct'])}C + {len(RANKING_CASES[0]['plausible'])}P + {len(RANKING_CASES[0]['wrong'])}W

  Method A (isolated word T2 vs context anchor):
    Correct ordering (C>P>W):  {n_order_A}/{n_cases} ({100*n_order_A/n_cases:.1f}%)
    Mean rank of best correct: {mean_rank_A:.2f} (random baseline: {(n_cands+1)/2:.1f})
    MRR:                       {mrr_A:.4f} (random baseline: {sum(1/i for i in range(1,n_cands+1))/n_cands:.4f})

  Method B (in-context T2 vs context anchor):
    Correct ordering (C>P>W):  {n_order_B}/{n_cases} ({100*n_order_B/n_cases:.1f}%)
    Mean rank of best correct: {mean_rank_B:.2f} (random baseline: {(n_cands+1)/2:.1f})
    MRR:                       {mrr_B:.4f} (random baseline: {sum(1/i for i in range(1,n_cands+1))/n_cands:.4f})
""")

# Per-category analysis
print("  Per-category (Method B — in-context):")
print(f"  {'category':>12}  {'order':>6}  {'rank_B':>8}  {'C_sim':>8}  {'P_sim':>8}  {'W_sim':>8}")
print(f"  {'-'*58}")
cats_seen = []
for r in all_results:
    if r["category"] not in cats_seen:
        cats_seen.append(r["category"])
for cat in cats_seen:
    cat_res = [r for r in all_results if r["category"] == cat]
    avg_order = sum(1 for r in cat_res if r["order_B"]) / len(cat_res)
    avg_rank  = np.mean([r["rank_B"] for r in cat_res])
    avg_C = np.mean([r["means_B"]["correct"] for r in cat_res])
    avg_P = np.mean([r["means_B"]["plausible"] for r in cat_res])
    avg_W = np.mean([r["means_B"]["wrong"] for r in cat_res])
    print(f"  {cat:>12}  {avg_order:>6.1%}  {avg_rank:>8.2f}  "
          f"{avg_C:>8.4f}  {avg_P:>8.4f}  {avg_W:>8.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 124 Summary — T2 Semantic Ranking")
print("=" * 72)
random_mrr = sum(1/i for i in range(1, n_cands+1)) / n_cands
t2_useful_A = mrr_A > 1.2 * random_mrr
t2_useful_B = mrr_B > 1.2 * random_mrr

print(f"""
  T2 as semantic filter for candidate ranking:

  Method A (isolated): MRR={mrr_A:.4f}, correct_order={100*n_order_A/n_cases:.0f}%
  Method B (context):  MRR={mrr_B:.4f}, correct_order={100*n_order_B/n_cases:.0f}%
  Random baseline MRR: {random_mrr:.4f}

  VERDICT:
  Method A (isolated): {'USEFUL — T2 carries semantic signal above random' if t2_useful_A else 'NOT USEFUL — T2 does not rank correct answers better than random'}
  Method B (context):  {'USEFUL — contextual T2 carries semantic signal above random' if t2_useful_B else 'NOT USEFUL — contextual T2 does not rank correct answers better than random'}

  BEST METHOD: {'B (context)' if mrr_B > mrr_A else 'A (isolated)' if mrr_A > mrr_B else 'tied'}

  IMPLICATION FOR TRUTHSPACE:
  {'→ T2 addresses CAN serve as semantic filters. Correct answers are distinguishable from wrong ones using geometric proximity in T2 space.' if t2_useful_B else
   '→ T2 addresses cannot reliably distinguish correct from wrong completions. The 12D projection does not carry enough semantic ranking signal.'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "results": all_results,
        "n_order_A": n_order_A, "n_order_B": n_order_B,
        "mrr_A": mrr_A, "mrr_B": mrr_B,
        "random_mrr": random_mrr,
        "mean_rank_A": mean_rank_A, "mean_rank_B": mean_rank_B,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 124 complete.")
