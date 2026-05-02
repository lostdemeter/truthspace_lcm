#!/usr/bin/env python3
"""
Day 178 — The Relational Encoding Boundary in W_E

Day 176 showed: geographic/linguistic relations encode as directions (viable chains),
but thematic/associative relations do not (animal→sound, metal→property, season→weather).

WHY? What geometric signal distinguishes encoded from non-encoded relations?

FOUR HYPOTHESES:
  H1: Direction consistency — encoded relations have consistent difference vectors
      (high mean inter-pair cosine); non-encoded have random/noisy differences
  H2: Target cluster compactness — encoded targets cluster together;
      non-encoded targets are spread across W_E
  H3: Source cluster compactness — encoded sources cluster together;
      non-encoded sources are scattered
  H4: Displacement magnitude consistency — encoded pairs have uniform ||Y-X||;
      non-encoded pairs have variable displacement magnitudes

PREDICTION: H1 (direction consistency) is the primary distinguisher.
  High direction consistency → relation is directionally encoded → viable for chains
  Low direction consistency → relation is not directionally encoded → chains fail

DOMAINS TESTED:
  ENCODED (known to work):  capitals, languages, gender, category(metal)
  NON-ENCODED (known to fail): animal sounds, metal properties, seasonal weather
  BORDERLINE (unknown):  insect→category, planet→type, color→temperature
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day178_relational_boundary.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── All domains ─────────────────────────────────────────────────
DOMAINS = {
    # ENCODED (expected high consistency)
    "capitals": [
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        ("Russia","Moscow"),("Greece","Athens"),("Poland","Warsaw"),
        ("Sweden","Stockholm"),("Korea","Seoul"),("Brazil","Brasilia"),
    ],
    "languages": [
        ("France","French"),("Germany","German"),("Italy","Italian"),
        ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
        ("Russia","Russian"),("Greece","Greek"),("Sweden","Swedish"),
        ("Korea","Korean"),("Poland","Polish"),
    ],
    "gender": [
        ("king","queen"),("man","woman"),("boy","girl"),
        ("prince","princess"),("lord","lady"),("actor","actress"),
        ("waiter","waitress"),("hero","heroine"),
    ],
    "metal_to_category": [
        ("iron","metal"),("copper","metal"),("aluminum","metal"),
        ("tin","metal"),("zinc","metal"),("lead","metal"),
        ("gold","metal"),("silver","metal"),("steel","metal"),
    ],
    # NON-ENCODED (expected low consistency)
    "animal_sound": [
        ("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),
        ("lion","roar"),("bird","tweet"),("frog","croak"),("bee","buzz"),
        ("horse","neigh"),("snake","hiss"),
    ],
    "metal_property": [
        ("iron","magnetic"),("copper","conductive"),("gold","malleable"),
        ("silver","reflective"),("aluminum","lightweight"),("lead","heavy"),
        ("steel","strong"),("tin","soft"),
    ],
    "season_weather": [
        ("winter","snow"),("summer","heat"),("spring","rain"),("autumn","wind"),
        ("winter","cold"),("summer","hot"),("spring","mild"),("autumn","cool"),
    ],
    "number_parity": [
        ("one","odd"),("two","even"),("three","odd"),("four","even"),
        ("five","odd"),("six","even"),("seven","odd"),("eight","even"),
    ],
    # BORDERLINE (unknown encoding status)
    "insect_category": [
        ("ant","insect"),("bee","insect"),("fly","insect"),
        ("moth","insect"),("wasp","insect"),("beetle","insect"),
    ],
    "planet_type": [
        ("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky"),
        ("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas"),
    ],
    "color_temperature": [
        ("red","warm"),("orange","warm"),("yellow","warm"),
        ("blue","cool"),("green","cool"),("purple","cool"),
    ],
    "antonym_hot": [
        ("hot","cold"),("big","small"),("fast","slow"),("hard","soft"),
        ("light","dark"),("old","young"),("loud","quiet"),("rich","poor"),
    ],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
print(f"  H={W_E.shape[1]}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def analyze_domain(name, pairs):
    ok = [(a, b) for a, b in pairs if tid(a) and tid(b)]
    if len(ok) < 3:
        return None

    srcs = np.array([W_E[tid(a)] for a, b in ok])
    tgts = np.array([W_E[tid(b)] for a, b in ok])
    diffs = np.array([normed(W_E[tid(b)] - W_E[tid(a)]) for a, b in ok])

    # H1: Direction consistency (mean inter-pair cosine of diff vectors)
    n = len(diffs)
    if n > 1:
        cos_pairs = [cosine(diffs[i], diffs[j])
                     for i in range(n) for j in range(i+1, n)]
        h1_consistency = float(np.mean(cos_pairs))
    else:
        h1_consistency = 0.0

    # H2: Target cluster compactness (mean pairwise cosine of targets)
    tgt_norms = np.array([normed(t) for t in tgts])
    if n > 1:
        tgt_pairs = [float(np.dot(tgt_norms[i], tgt_norms[j]))
                     for i in range(n) for j in range(i+1, n)]
        h2_target_compact = float(np.mean(tgt_pairs))
    else:
        h2_target_compact = 0.0

    # H3: Source cluster compactness (mean pairwise cosine of sources)
    src_norms = np.array([normed(s) for s in srcs])
    if n > 1:
        src_pairs = [float(np.dot(src_norms[i], src_norms[j]))
                     for i in range(n) for j in range(i+1, n)]
        h3_source_compact = float(np.mean(src_pairs))
    else:
        h3_source_compact = 0.0

    # H4: Displacement magnitude consistency (CV of ||Y-X||)
    mags = [float(np.linalg.norm(W_E[tid(b)] - W_E[tid(a)])) for a, b in ok]
    h4_mag_cv = float(np.std(mags) / (np.mean(mags) + 1e-8))  # lower = more consistent

    # Bonus: LOO retrieval accuracy using mean direction
    mean_dir = normed(np.mean(diffs, axis=0))
    # Build target vocabulary (just the targets in this domain)
    tgt_vocab = {b: W_E[tid(b)] for _, b in ok}
    nc = 0
    for a, b in ok:
        eid = tid(a)
        # LOO direction: exclude current pair
        loo_diffs = [normed(W_E[tid(bb)] - W_E[tid(aa)])
                     for aa, bb in ok if aa != a]
        if not loo_diffs: continue
        loo_dir = normed(np.mean(loo_diffs, axis=0))
        query = W_E[eid] + loo_dir
        cands = {w: cosine(query, v) for w, v in tgt_vocab.items() if w != a}
        if not cands: continue
        pred = max(cands, key=lambda w: cands[w])
        if pred == b: nc += 1
    loo_acc = nc / len(ok)

    return {
        "n_pairs": len(ok),
        "H1_direction_consistency": h1_consistency,
        "H2_target_compactness": h2_target_compact,
        "H3_source_compactness": h3_source_compact,
        "H4_magnitude_cv": h4_mag_cv,
        "LOO_retrieval_acc": loo_acc,
    }

print(f"{'Domain':>22}  {'H1_dir':>8}  {'H2_tgt':>8}  {'H3_src':>8}  "
      f"{'H4_cv':>7}  {'LOO_acc':>8}  {'n':>4}")
print("-"*80)

results = {}
for name, pairs in DOMAINS.items():
    r = analyze_domain(name, pairs)
    if r is None:
        print(f"  {name}: insufficient single-token pairs, skip")
        continue
    results[name] = r
    print(f"  {name:>20}  {r['H1_direction_consistency']:>8.3f}  "
          f"{r['H2_target_compactness']:>8.3f}  "
          f"{r['H3_source_compactness']:>8.3f}  "
          f"{r['H4_magnitude_cv']:>7.3f}  "
          f"{r['LOO_retrieval_acc']:>8.3f}  "
          f"{r['n_pairs']:>4}")

# ─── Correlation analysis ─────────────────────────────────────────
print()
print("="*80)
print("CORRELATION: Which hypothesis best predicts LOO_acc?")
print("="*80)
print()

metrics = ["H1_direction_consistency", "H2_target_compactness",
           "H3_source_compactness", "H4_magnitude_cv"]
accs = [results[d]["LOO_retrieval_acc"] for d in results]

for metric in metrics:
    vals = [results[d][metric] for d in results]
    if metric == "H4_magnitude_cv":
        vals = [-v for v in vals]  # lower CV = better consistency
    corr = np.corrcoef(vals, accs)[0, 1]
    print(f"  cor(LOO_acc, {metric:>28}): {corr:+.3f}")

# ─── Threshold analysis ───────────────────────────────────────────
print()
print("="*80)
print("THRESHOLD: What H1 value separates encoded from non-encoded?")
print("="*80)
print()

sorted_by_h1 = sorted(results.items(), key=lambda x: -x[1]["H1_direction_consistency"])
print(f"  {'Domain':>22}  {'H1':>6}  {'LOO':>6}  {'Encoded?':>10}")
for name, r in sorted_by_h1:
    expected = "yes" if name in {"capitals","languages","gender","metal_to_category"} else \
               "no"  if name in {"animal_sound","metal_property","season_weather"} else "?"
    print(f"  {name:>22}  {r['H1_direction_consistency']:>6.3f}  "
          f"{r['LOO_retrieval_acc']:>6.3f}  {expected:>10}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 178 complete.")
