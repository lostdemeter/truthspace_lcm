#!/usr/bin/env python3
"""
Day 196 — Relational Encoding Archetypes

HYPOTHESIS: Beyond TYPE_BC (directional translation), other geometric
encoding patterns exist in W_E. We test four candidate archetypes:

  TYPE_BC (directional): source + mean_direction → target
    - Known: capitals, gender, antonyms, languages
    - LOO accuracy ~0.90
    - Step size consistent (~4% std)

  TYPE_CLUSTER (cluster proximity): related words are nearest neighbours
    without a consistent directional translation
    - Candidate: singular→plural, noun→verb form
    - Test: is the plural nearer to the singular than a direction predicts?

  TYPE_ORDINAL (rank-encoded): magnitude along a direction encodes order
    - Candidate: number words (one < two < three)
    - Test: do projections onto number-direction give monotonic order?

  TYPE_HYPERNYM (containment): hyponyms cluster within a hypernym region
    - Candidate: dog/cat/horse → animal, red/blue/green → color
    - Test: hypernym centroid nearer to hyponyms than random words?

  TYPE_ANALOGY (2D): source differs along two independent dimensions
    - Candidate: man:woman :: king:queen (both direction AND identity)
    - Test: does applying gender direction to king give queen better than LOO?

EXPERIMENTS:
  For each archetype, measure:
    - LOO accuracy using the directional approach
    - Centroid-proximity accuracy
    - Rank-correlation for ordinal encoding
    - Containment measure for hypernyms
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day196_encoding_archetypes.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    # TYPE_BC — established directional domains
    "capitals":   [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                   ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
                   ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
                   ("Korea","Seoul"),("Poland","Warsaw"),("Turkey","Ankara")],
    "gender":     [("king","queen"),("man","woman"),("boy","girl"),
                   ("prince","princess"),("actor","actress"),("hero","heroine"),
                   ("waiter","waitress"),("duke","duchess")],
    "antonyms":   [("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
                   ("light","dark"),("old","young"),("loud","quiet"),
                   ("sharp","dull"),("rich","poor"),("thick","thin")],
    # TYPE_ORDINAL — candidate
    "number_seq": [("one",1),("two",2),("three",3),("four",4),("five",5),
                   ("six",6),("seven",7),("eight",8),("nine",9),("ten",10),
                   ("eleven",11),("twelve",12)],
    # TYPE_CLUSTER/ANALOGY — singular→plural (morphological)
    "plurals":    [("cat","cats"),("dog","dogs"),("house","houses"),
                   ("tree","trees"),("book","books"),("car","cars"),
                   ("bird","birds"),("ship","ships"),("hand","hands"),
                   ("eye","eyes"),("road","roads"),("door","doors")],
    # TYPE_HYPERNYM — candidate
    "hypernyms":  {
        "animal":  ["dog","cat","horse","bird","fish","lion","wolf","cow","sheep"],
        "color":   ["red","blue","green","yellow","white","black","brown","pink"],
        "country": ["France","Germany","Italy","Spain","Japan","China","Russia"],
        "number":  ["one","two","three","four","five","six","seven"],
    },
    # TYPE_BC tense — verb past tense
    "past_tense": [("run","ran"),("eat","ate"),("go","went"),("see","saw"),
                   ("come","came"),("give","gave"),("take","took"),
                   ("make","made"),("say","said"),("know","knew")],
    # TYPE_BC superlative
    "superlative":[("big","biggest"),("fast","fastest"),("old","oldest"),
                   ("cold","coldest"),("smart","smartest"),("long","longest")],
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

# ── LOO directional accuracy for pair-based domains ──────────────────
def loo_accuracy(pairs, vocab_filter=None):
    """Leave-one-out directional retrieval accuracy."""
    ok = [(a, b) for a, b in pairs if tid1(a) and tid1(b)]
    if len(ok) < 3: return None, None, ok
    # Build target vocabulary from all pairs
    tgt_vocab = {b: get_emb(b) for _, b in ok}
    correct = 0
    ranks = []
    for i, (a, b) in enumerate(ok):
        loo_diffs = [normed(get_emb(bb) - get_emb(aa))
                     for aa, bb in ok if aa != a]
        mean_dir = normed(np.mean(loo_diffs, axis=0))
        query = get_emb(a) + mean_dir
        sims = {w: cosine(query, v) for w, v in tgt_vocab.items() if w != a}
        ranked = sorted(sims, key=lambda w: sims[w], reverse=True)
        if ranked[0] == b: correct += 1
        ranks.append(ranked.index(b) if b in ranked else len(ranked))
    acc = correct / len(ok)
    return acc, float(np.mean(ranks)), ok

# ── TYPE_BC Domains ──────────────────────────────────────────────────
print("=" * 60)
print("TYPE_BC: Directional Domains")
print("=" * 60)
results = {}
for domain in ["capitals", "gender", "antonyms", "past_tense", "superlative", "plurals"]:
    pairs = DOMAINS[domain]
    acc, mean_rank, ok = loo_accuracy(pairs)
    if acc is None:
        print(f"  {domain:<16}: INSUFFICIENT DATA (n={len(ok)})")
        continue
    # Step consistency
    steps = [normed(get_emb(b) - get_emb(a)) for a, b in ok]
    step_mags = [float(np.linalg.norm(get_emb(b) - get_emb(a))) for a, b in ok]
    mean_step = normed(np.mean(steps, axis=0))
    inter_cos = [cosine(steps[i], steps[j])
                 for i in range(len(steps)) for j in range(i+1, len(steps))]
    direction_consistency = float(np.mean(inter_cos))
    print(f"  {domain:<16}  n={len(ok):<3}  acc={acc:.3f}  "
          f"mean_rank={mean_rank:.2f}  "
          f"dir_consistency={direction_consistency:.4f}  "
          f"step_mag={np.mean(step_mags):.3f}±{np.std(step_mags):.3f}")
    results[domain] = {"type": "TYPE_BC", "n": len(ok), "accuracy": acc,
                       "mean_rank": mean_rank,
                       "direction_consistency": direction_consistency,
                       "step_mag_mean": float(np.mean(step_mags)),
                       "step_mag_std": float(np.std(step_mags))}
print()

# ── TYPE_ORDINAL: Number sequence ────────────────────────────────────
print("=" * 60)
print("TYPE_ORDINAL: Number Sequence Rank Encoding")
print("=" * 60)
num_pairs = DOMAINS["number_seq"]
ok_nums = [(w, rank) for w, rank in num_pairs if tid1(w)]
if ok_nums:
    words = [w for w, r in ok_nums]
    ranks = [r for w, r in ok_nums]
    embs  = np.array([get_emb(w) for w in words])
    # Find the direction that best explains ordinal order
    # Use first-difference direction: two-one, three-two, ...
    diffs = [normed(embs[i+1] - embs[i]) for i in range(len(embs)-1)]
    mean_dir = normed(np.mean(diffs, axis=0))
    # Project each word onto mean direction
    projs = [float(np.dot(normed(embs[i]), mean_dir)) for i in range(len(embs))]
    # Spearman rank correlation: do projections give monotonic order?
    rho, p = spearmanr(ranks, projs)
    print(f"  Number words: {words}")
    print(f"  Projections:  {[f'{p:.3f}' for p in projs]}")
    print(f"  Spearman ρ = {rho:.4f}  (p={p:.4f})")
    print()
    # Also test: LOO for number→next
    num_seq_pairs = list(zip(words[:-1], words[1:]))
    acc_num, mr_num, ok_seq = loo_accuracy(num_seq_pairs)
    print(f"  LOO accuracy (num→next): {acc_num:.3f}  mean_rank={mr_num:.2f}")
    results["number_ordinal"] = {"type": "TYPE_ORDINAL", "spearman_rho": float(rho),
                                  "projections": projs,
                                  "loo_accuracy_sequential": acc_num}
print()

# ── TYPE_HYPERNYM: Containment ───────────────────────────────────────
print("=" * 60)
print("TYPE_HYPERNYM: Hypernym Containment")
print("=" * 60)
hypernym_results = {}
for hypernym, hyponyms in DOMAINS["hypernyms"].items():
    ok_hypo = [(w, get_emb(w)) for w in hyponyms if tid1(w)]
    hypo_emb = get_emb(hypernym)
    if not ok_hypo or hypo_emb is None: continue
    # Centroid of hyponyms
    hypo_vecs = np.array([e for _, e in ok_hypo])
    centroid = normed(np.mean(hypo_vecs / (np.linalg.norm(hypo_vecs,axis=1,keepdims=True)+1e-8), axis=0))
    # How similar is hypernym to hyponym centroid?
    hyper_cos = cosine(hypo_emb, centroid)
    # Mean similarity of hypernym to individual hyponyms
    hyper_hypo_cos = [cosine(hypo_emb, e) for _, e in ok_hypo]
    # Intra-hyponym similarity
    nv = hypo_vecs / (np.linalg.norm(hypo_vecs,axis=1,keepdims=True)+1e-8)
    cos_m = nv @ nv.T
    mask = ~np.eye(len(ok_hypo), dtype=bool)
    intra = float(np.mean(cos_m[mask]))
    print(f"  {hypernym:<10}  n_hypo={len(ok_hypo):<3}  "
          f"hyper→centroid={hyper_cos:.4f}  "
          f"hyper→hypo_mean={np.mean(hyper_hypo_cos):.4f}  "
          f"intra_hypo={intra:.4f}")
    hypernym_results[hypernym] = {
        "hyper_to_centroid": float(hyper_cos),
        "hyper_to_hypo_mean": float(np.mean(hyper_hypo_cos)),
        "intra_hyponym_cos": intra,
    }
results["hypernyms"] = hypernym_results
print()

# Direction consistency comparison table
print("=" * 60)
print("Direction Consistency Summary (all tested domains)")
print("=" * 60)
print(f"  {'Domain':<16}  {'acc':>6}  {'dir_consistency':>16}  {'step_mag':>10}  type")
print("  " + "-"*70)
for domain, r in results.items():
    if "accuracy" in r:
        print(f"  {domain:<16}  {r['accuracy']:>6.3f}  "
              f"{r.get('direction_consistency',0):>16.4f}  "
              f"{r.get('step_mag_mean',0):>10.3f}  {r['type']}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 196 complete.")
