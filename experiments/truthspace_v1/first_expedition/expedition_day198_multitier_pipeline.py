#!/usr/bin/env python3
"""
Day 198 — Multi-Tier Retrieval Pipeline

Build and validate an archetype-aware pipeline that:
  1. Auto-classifies a domain's encoding type from known pairs
  2. Routes to the correct retrieval method
  3. Compares accuracy to the naive single-method baseline (always TYPE_BC)

PIPELINE:
  Input: (source_word, known_pairs, target_vocab)
  
  Stage 1 — CLASSIFY:
    Compute direction consistency on known pairs
    Compute Spearman ρ of projections vs known ordinal rank
    → assign TYPE_BC / TYPE_ORDINAL / TYPE_ADJACENT / TYPE_HYPERNYM
  
  Stage 2 — RETRIEVE:
    TYPE_BC:       query = W_E[src] + mean_dir  → nearest in target_vocab
    TYPE_ORDINAL:  project src onto ordinal axis → nearest in target_vocab
    TYPE_ADJACENT: return nearest neighbour from combined source+target vocab
    TYPE_HYPERNYM: return word with highest mean similarity to cluster

  Stage 3 — EVALUATE:
    LOO accuracy, mean rank
    Compare to baseline (TYPE_BC applied to all domains)

DOMAINS TESTED (from prior experiments):
  capitals, gender, past_tense, plurals, superlative  → TYPE_BC
  antonyms                                            → TYPE_ADJACENT
  numbers (ordinal position)                          → TYPE_ORDINAL
  hypernyms (animal, color)                           → TYPE_HYPERNYM
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day198_multitier_pipeline.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ALL_DOMAINS = {
    "capitals":    [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                    ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                    ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
                    ("Korea","Seoul"),("Poland","Warsaw"),("Turkey","Ankara")],
    "gender":      [("king","queen"),("man","woman"),("boy","girl"),
                    ("prince","princess"),("actor","actress"),("hero","heroine"),
                    ("waiter","waitress"),("duke","duchess")],
    "antonyms":    [("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
                    ("light","dark"),("old","young"),("loud","quiet"),
                    ("sharp","dull"),("rich","poor"),("thick","thin")],
    "past_tense":  [("run","ran"),("eat","ate"),("go","went"),("see","saw"),
                    ("come","came"),("give","gave"),("take","took"),
                    ("make","made"),("say","said"),("know","knew")],
    "superlative": [("big","biggest"),("fast","fastest"),("old","oldest"),
                    ("cold","coldest"),("smart","smartest"),("long","longest"),
                    ("hard","hardest"),("dark","darkest")],
    "plurals":     [("cat","cats"),("dog","dogs"),("house","houses"),
                    ("tree","trees"),("book","books"),("car","cars"),
                    ("bird","birds"),("ship","ships"),("hand","hands"),
                    ("eye","eyes"),("road","roads"),("door","doors")],
    "numbers":     [("one",1),("two",2),("three",3),("four",4),("five",5),
                    ("six",6),("seven",7),("eight",8),("nine",9),("ten",10),
                    ("eleven",11),("twelve",12)],
    "hypernyms":   {
        "animal":  ["dog","cat","horse","bird","fish","lion","wolf","cow","sheep"],
        "color":   ["red","blue","green","yellow","white","black","brown","pink"],
    },
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a,b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                      normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_emb(word):
    t = tid1(word)
    return W_E[t].astype(np.float64) if t is not None else None

# ── Stage 1: Archetype Classifier ────────────────────────────────────
def classify_archetype(pairs, ordinal_ranks=None):
    """
    Classify a domain's encoding archetype.
    pairs: list of (src, tgt) tuples
    ordinal_ranks: dict {word: rank} if ordinal hypothesis should be tested
    Returns: archetype string, confidence dict
    """
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 3:
        return "UNKNOWN", {}

    src_embs = [normed(get_emb(a)) for a, _ in ok]
    tgt_embs = [normed(get_emb(b)) for _, b in ok]
    diffs = [normed(np.array(te) - np.array(se)) for se, te in zip(src_embs, tgt_embs)]

    # Direction consistency: mean pairwise cosine of direction vectors
    pairwise = [cosine(diffs[i], diffs[j])
                for i in range(len(diffs)) for j in range(i+1, len(diffs))]
    dir_consistency = float(np.mean(pairwise))

    # Ordinal test: project sources onto mean direction, correlate with rank
    spearman_rho = 0.0
    if ordinal_ranks:
        mean_dir = normed(np.mean(diffs, axis=0))
        projs = []
        ranks = []
        for a, _ in ok:
            if a in ordinal_ranks:
                e = get_emb(a)
                projs.append(float(np.dot(normed(e), mean_dir)))
                ranks.append(ordinal_ranks[a])
        if len(projs) >= 3:
            rho, _ = spearmanr(ranks, projs)
            spearman_rho = float(rho)

    # Classify
    if dir_consistency >= 0.20:
        archetype = "TYPE_BC"
    elif spearman_rho >= 0.85:
        archetype = "TYPE_ORDINAL"
    elif dir_consistency < 0.10:
        archetype = "TYPE_ADJACENT"
    else:
        archetype = "TYPE_BC"  # fallback for 0.10–0.20 range

    confidence = {
        "dir_consistency": dir_consistency,
        "spearman_rho": spearman_rho,
        "n_pairs": len(ok),
    }
    return archetype, confidence

# ── Stage 2: Retrieval Methods ────────────────────────────────────────
def retrieve_type_bc(src_word, known_pairs, target_set):
    """TYPE_BC: add mean direction to source embedding."""
    ok = [(a, b) for a, b in known_pairs
          if tid1(a) and tid1(b) and a != src_word]
    if not ok: return None, None
    diffs = [normed(get_emb(b) - get_emb(a)) for a, b in ok]
    mean_dir = normed(np.mean(diffs, axis=0))
    query = get_emb(src_word) + mean_dir
    tgt_embs = {w: get_emb(w) for w in target_set if tid1(w) and w != src_word}
    if not tgt_embs: return None, None
    sims = {w: cosine(query, e) for w, e in tgt_embs.items()}
    ranked = sorted(sims, key=lambda w: sims[w], reverse=True)
    return ranked[0], ranked

def retrieve_type_adjacent(src_word, target_set):
    """TYPE_ADJACENT: return nearest token from target set (no direction)."""
    se = get_emb(src_word)
    if se is None: return None, None
    tgt_embs = {w: get_emb(w) for w in target_set if tid1(w) and w != src_word}
    if not tgt_embs: return None, None
    sims = {w: cosine(se, e) for w, e in tgt_embs.items()}
    ranked = sorted(sims, key=lambda w: sims[w], reverse=True)
    return ranked[0], ranked

def retrieve_type_ordinal(src_word, known_pairs, target_set, ordinal_ranks):
    """TYPE_ORDINAL: project onto ordinal axis, return word with next rank."""
    ok = [(a, b) for a, b in known_pairs if tid1(a) and tid1(b)]
    if not ok: return None, None
    diffs = [normed(get_emb(b) - get_emb(a)) for a, b in ok]
    mean_dir = normed(np.mean(diffs, axis=0))
    src_proj = float(np.dot(normed(get_emb(src_word)), mean_dir))
    # Return target with projection just above src_proj
    tgt_projs = {}
    for w in target_set:
        if tid1(w) and w != src_word:
            e = get_emb(w)
            tgt_projs[w] = float(np.dot(normed(e), mean_dir))
    if not tgt_projs: return None, None
    # Rank targets by how close their projection is to src_proj (from above)
    above = {w: p for w, p in tgt_projs.items() if p > src_proj}
    if above:
        ranked = sorted(above, key=lambda w: above[w])  # smallest above
    else:
        ranked = sorted(tgt_projs, key=lambda w: tgt_projs[w], reverse=True)
    return ranked[0], ranked

# ── Stage 3: LOO Evaluation ───────────────────────────────────────────
def evaluate_pipeline(domain_name, pairs, retrieval_fn, target_override=None):
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 3: return None
    target_set = target_override or [b for _, b in ok]
    correct = 0
    ranks = []
    for i, (a, b) in enumerate(ok):
        loo_pairs = [(aa, bb) for aa, bb in ok if aa != a]
        pred, ranked = retrieval_fn(a, loo_pairs, target_set)
        if pred is None: continue
        if pred == b: correct += 1
        rank = ranked.index(b) if b in ranked else len(ranked)
        ranks.append(rank)
    return {"accuracy": correct / len(ok), "mean_rank": float(np.mean(ranks)),
            "n": len(ok)}

# ── Run Pipeline ──────────────────────────────────────────────────────
print("=" * 70)
print("MULTI-TIER PIPELINE: Classification + Retrieval")
print("=" * 70)

results = {}
total_correct_pipeline = 0
total_correct_baseline = 0
total_n = 0

for domain in ["capitals","gender","antonyms","past_tense","superlative","plurals"]:
    pairs = ALL_DOMAINS[domain]
    ok = [(a,b) for a,b in pairs if tid1(a) and tid1(b)]

    # Classify
    ordinal_ranks = None
    archetype, confidence = classify_archetype(ok)

    # Pipeline retrieval
    if archetype == "TYPE_BC":
        fn = lambda src, known, tgt: retrieve_type_bc(src, known, tgt)
    elif archetype == "TYPE_ADJACENT":
        fn = lambda src, known, tgt: retrieve_type_adjacent(src, tgt)
    else:
        fn = lambda src, known, tgt: retrieve_type_bc(src, known, tgt)

    pipeline_res = evaluate_pipeline(domain, ok, fn)
    baseline_res = evaluate_pipeline(domain, ok,
                    lambda src, known, tgt: retrieve_type_bc(src, known, tgt))

    print(f"\n  {domain}")
    print(f"    Classified: {archetype}  "
          f"(dir={confidence['dir_consistency']:.3f})")
    print(f"    Pipeline:  acc={pipeline_res['accuracy']:.3f}  "
          f"mean_rank={pipeline_res['mean_rank']:.2f}")
    print(f"    Baseline:  acc={baseline_res['accuracy']:.3f}  "
          f"mean_rank={baseline_res['mean_rank']:.2f}")
    delta = pipeline_res['accuracy'] - baseline_res['accuracy']
    print(f"    Delta: {delta:+.3f}")

    results[domain] = {"archetype": archetype, "confidence": confidence,
                       "pipeline": pipeline_res, "baseline": baseline_res}
    total_correct_pipeline += pipeline_res['accuracy'] * pipeline_res['n']
    total_correct_baseline += baseline_res['accuracy'] * baseline_res['n']
    total_n += pipeline_res['n']

# Numbers: TYPE_ORDINAL
print(f"\n  {'numbers (ordinal)'}")
num_pairs_raw = ALL_DOMAINS["numbers"]
num_words = [w for w, r in num_pairs_raw if tid1(w)]
num_ranks = {w: r for w, r in num_pairs_raw if tid1(w)}
num_seq   = list(zip(num_words[:-1], num_words[1:]))
ok_nums   = [(a, b) for a, b in num_seq if tid1(a) and tid1(b)]

ordinal_ranks_full = {w: r for w, r in num_pairs_raw if tid1(w)}
archetype_num, conf_num = classify_archetype(ok_nums, ordinal_ranks_full)

# Ordinal retrieval: given word, find next higher in sequence
def ordinal_fn(src, known, tgt):
    return retrieve_type_ordinal(src, known, tgt, ordinal_ranks_full)

pipeline_num = evaluate_pipeline("numbers", ok_nums, ordinal_fn,
                                  target_override=num_words)
baseline_num = evaluate_pipeline("numbers", ok_nums,
                    lambda s,k,t: retrieve_type_bc(s, k, t),
                    target_override=num_words)

print(f"    Classified: {archetype_num}  "
      f"(dir={conf_num['dir_consistency']:.3f}  "
      f"ρ={conf_num['spearman_rho']:.3f})")
print(f"    Pipeline (ordinal): acc={pipeline_num['accuracy']:.3f}  "
      f"mean_rank={pipeline_num['mean_rank']:.2f}")
print(f"    Baseline (TYPE_BC): acc={baseline_num['accuracy']:.3f}  "
      f"mean_rank={baseline_num['mean_rank']:.2f}")
delta_num = pipeline_num['accuracy'] - baseline_num['accuracy']
print(f"    Delta: {delta_num:+.3f}")
results["numbers"] = {"archetype": archetype_num, "confidence": conf_num,
                      "pipeline": pipeline_num, "baseline": baseline_num}
total_correct_pipeline += pipeline_num['accuracy'] * pipeline_num['n']
total_correct_baseline += baseline_num['accuracy'] * baseline_num['n']
total_n += pipeline_num['n']

# Hypernyms: TYPE_HYPERNYM — can we find the hypernym from a hyponym?
print(f"\n  {'hypernym recovery'}")
hyper_results = {}
for hypernym, hyponyms in ALL_DOMAINS["hypernyms"].items():
    ok_hypo = [w for w in hyponyms if tid1(w)]
    if len(ok_hypo) < 3: continue
    hyper_emb = get_emb(hypernym)
    if hyper_emb is None: continue
    # For each hyponym, retrieve hypernym as nearest in {hypernyms list}
    hypernym_vocab = list(ALL_DOMAINS["hypernyms"].keys())
    correct = sum(1 for w in ok_hypo
                  if sorted(hypernym_vocab, key=lambda h:
                     cosine(get_emb(w), get_emb(h)) if get_emb(h) is not None else -1,
                     reverse=True)[0] == hypernym)
    acc = correct / len(ok_hypo)
    print(f"    {hypernym:<10}: acc={acc:.3f}  "
          f"(n={len(ok_hypo)}, correct={correct})")
    hyper_results[hypernym] = {"accuracy": acc, "n": len(ok_hypo)}
results["hypernyms"] = hyper_results

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
pipeline_overall = total_correct_pipeline / total_n
baseline_overall = total_correct_baseline / total_n
print(f"  Overall pipeline accuracy: {pipeline_overall:.3f}")
print(f"  Overall baseline accuracy: {baseline_overall:.3f}")
print(f"  Overall delta:            {pipeline_overall - baseline_overall:+.3f}")
print()
print(f"  {'Domain':<16}  {'Archetype':<14}  {'Pipeline':>10}  {'Baseline':>10}  {'Delta':>8}")
print("  " + "-"*65)
for domain, r in results.items():
    if "pipeline" in r and "baseline" in r and r["pipeline"]:
        p = r["pipeline"]["accuracy"]
        b = r["baseline"]["accuracy"]
        print(f"  {domain:<16}  {r['archetype']:<14}  {p:>10.3f}  {b:>10.3f}  {p-b:>+8.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 198 complete.")
