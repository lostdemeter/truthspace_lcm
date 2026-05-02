#!/usr/bin/env python3
"""
Day 118 — T2 Analogy Arithmetic

The TruthSpace self-similarity principle states:
  "Gender flip is always Δx = -2.0 (king→queen, man→woman, boy→girl)"
  "The same transformations work identically at every scale"

This predicts: in the T2 continuous projection space, analogy arithmetic works:
  proj(king) - proj(man) + proj(woman) ≈ proj(queen)

Where proj(w)[k] = h_w(L_k) · t2_axis_k  (per-axis correct layer)

Day 115 confirmed that φ-thresholding amplifies weak signals:
  - gender axis Cramér's V = 0.090 (30× amplification)
  - past_tense axis Cramér's V = 0.182

If the continuous projections are consistent WITHIN each category pair,
analogy arithmetic should work even with small absolute deltas.

EXPERIMENT:
  A. Gender analogies: king/queen, man/woman, boy/girl, actor/actress
     Test: proj(king) - proj(man) + proj(woman) ≈ proj(queen)?
     Metric: cosine(result, proj(queen)) > cosine(result, all others)

  B. Tense analogies: walk/walked, run/ran, eat/ate, see/saw
     Test: proj(walked) - proj(walk) + proj(run) ≈ proj(ran)?

  C. Plurality analogies: dog/dogs, cat/cats, tree/trees
     Test: proj(dogs) - proj(dog) + proj(cat) ≈ proj(cats)?

  D. Comparative analogies: fast/faster, big/bigger, old/older
     Test: proj(faster) - proj(fast) + proj(big) ≈ proj(bigger)?

  E. Antonym analogies: hot/cold, fast/slow, happy/sad
     Test: proj(cold) - proj(hot) + proj(fast) ≈ proj(slow)?

For each analogy, check if nearest neighbor in the vocabulary is correct.
Measure rank of correct answer among all vocabulary tokens.

SELF-SIMILARITY TEST:
  For each axis, compute Δ = proj(A') - proj(A) for all pairs on that axis.
  If self-similar: these Δs should be CONSTANT (low std / mean ratio).
  The TruthSpace hypothesis predicts constant Δ across all pairs.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day118_t2_analogy.json")
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
        ("A son was born in the winter","A daughter was born in the winter"),
        ("The prince rode across the land","The princess rode across the land"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
        ("The dark room","The darker room"),("A hard rock","A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The oak crashed in the storm","The tree crashed in the storm"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The soldier marched into fight","The person marched into fight"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk","Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road","The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky","Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text","The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He spoke loudly","He spoke noisily"),
        ("It is hard","It is difficult"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The iron chain has broken now","The bond between them has broken"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
        ("The strong root grips the soil","The strong base grips the earth"),
        ("The bridge connects two banks","The bond connects two communities"),
        ("The small key opens the door","The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days","We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend","She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire","They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The light is on","The dark is on"),("The news is good","The news is bad"),
        ("It is hard","It is soft"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is the first","It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("John broke the window","The window was broken by John"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The dog bit the man","The man was bitten by the dog"),
        ("The teacher helped the student","The student was helped by the teacher"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
        ("The king signed the document","The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The sun heats the cold earth","The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly","The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student","The student feels very proud"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The car broke down again","Did the car break down again"),
        ("The dog is hungry now","Is the dog hungry now"),
        ("She wrote the letter herself","Did she write the letter herself"),
        ("He knows the right answer","Does he know the right answer"),
        ("The house looks very old","Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today","It will not rain today"),
    ],
}

# Analogy sets: (A, A', B) → B'
# A is to A' as B is to B'
# axis_name tells which axis to use (determines layer)
ANALOGY_SETS = {
    "gender": [
        ("king",  "queen",  "man",   "woman"),
        ("man",   "woman",  "boy",   "girl"),
        ("man",   "woman",  "king",  "queen"),
        ("actor", "actress","man",   "woman"),
        ("father","mother", "son",   "daughter"),
        ("brother","sister","king",  "queen"),
    ],
    "past_tense": [
        ("walk",  "walked", "run",   "ran"),
        ("run",   "ran",    "walk",  "walked"),
        ("eat",   "ate",    "see",   "saw"),
        ("see",   "saw",    "write", "wrote"),
        ("build", "built",  "break", "broke"),
        ("swim",  "swam",   "fly",   "flew"),
    ],
    "plural": [
        ("dog",   "dogs",   "cat",   "cats"),
        ("cat",   "cats",   "dog",   "dogs"),
        ("tree",  "trees",  "bird",  "birds"),
        ("man",   "men",    "woman", "women"),
        ("hand",  "hands",  "eye",   "eyes"),
    ],
    "comparative": [
        ("fast",  "faster", "big",   "bigger"),
        ("big",   "bigger", "fast",  "faster"),
        ("good",  "better", "bad",   "worse"),
        ("old",   "older",  "cold",  "colder"),
    ],
    "antonym": [
        ("hot",   "cold",   "fast",  "slow"),
        ("good",  "bad",    "hot",   "cold"),
        ("happy", "sad",    "strong","weak"),
        ("old",   "new",    "big",   "small"),
    ],
    "synonym": [
        ("big",   "large",  "small", "tiny"),
        ("fast",  "quick",  "cold",  "frigid"),
        ("happy", "joyful", "old",   "aged"),
    ],
}

# Vocabulary for analogy resolution (all unique words in ANALOGY_SETS)
ANALOGY_VOCAB = list(dict.fromkeys([
    w for qs in ANALOGY_SETS.values() for quad in qs for w in quad
]))
# Add extra vocabulary for disambiguation
EXTRA_VOCAB = [
    "prince","princess","girl","horse","run","write","fly","jump","swim",
    "cat","bird","birds","cats","dogs","men","women","hands","eyes","trees",
    "faster","bigger","better","worse","older","colder","smaller",
    "cold","hot","fast","slow","happy","sad","good","bad","strong","weak",
    "old","new","big","small","large","tiny","quick","frigid","joyful","aged",
    "ran","walked","built","broke","swam","flew","saw","wrote","ate",
    "walk","eat","see","build","break",
    "son","daughter","brother","sister","mother","father","actor","actress",
    "king","queen","man","woman","boy",
]
VOCAB = list(dict.fromkeys(ANALOGY_VOCAB + EXTRA_VOCAB))

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}, vocab={len(VOCAB)} words\n")

def get_h_at_layer(word, layer):
    inp = tok(" " + word.strip(), return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            h2 = o2.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            d = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

print("Extracting vocabulary hidden states (all required layers) ...")
word_hs = {}  # {word: {layer: vec}}
for word in VOCAB:
    inp = tok(" " + word.strip(), return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        word_hs[word] = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                         for L in ALL_LAYERS}
    except: pass

valid_vocab = [w for w in VOCAB if w in word_hs]
print(f"  {len(valid_vocab)} valid words\n")

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def t2_vec(word):
    """12D T2 continuous projection, per-axis correct layer."""
    if word not in word_hs: return None
    return np.array([
        float(np.dot(normed(word_hs[word][DAY78_LAYERS[ax]]), t2_axes[ax]))
        for ax in AXIS_NAMES_12
    ], dtype=np.float32)

# Precompute T2 vectors for all vocab
t2_vecs = {w: t2_vec(w) for w in valid_vocab}

def nearest_by_t2(query_vec, exclude=(), top_k=5):
    """Find nearest vocab words by T2 cosine similarity."""
    sims = []
    for w in valid_vocab:
        if w in exclude: continue
        v = t2_vecs[w]
        if v is None: continue
        vn = np.linalg.norm(v); qn = np.linalg.norm(query_vec)
        cos = float(np.dot(v, query_vec) / (vn * qn + 1e-8))
        sims.append((w, cos))
    return sorted(sims, key=lambda x: -x[1])[:top_k]

def nearest_by_hidden(query_h, layer, exclude=(), top_k=5):
    """Find nearest vocab words by hidden-state cosine similarity at given layer."""
    sims = []
    qn = normed(query_h)
    for w in valid_vocab:
        if w in exclude: continue
        v = normed(word_hs[w][layer])
        sims.append((w, float(np.dot(qn, v))))
    return sorted(sims, key=lambda x: -x[1])[:top_k]

# ── Exp 1: T2 analogy arithmetic (3CosAdd) ────────────────────────────────────
print("=" * 72)
print("Exp 1: T2 Analogy Arithmetic (A:A' :: B:B')")
print("       Method: proj(B') ≈ proj(A') - proj(A) + proj(B)")
print("=" * 72)
print()

analogy_results = {}
total_correct_t2 = 0; total_analogy = 0
total_correct_h  = 0

for axis_name, analogies in ANALOGY_SETS.items():
    L = DAY78_LAYERS[axis_name]
    axis_results = []
    n_correct_t2 = 0; n_correct_h = 0
    print(f"  {axis_name} axis [L{L}]:")
    print(f"  {'A':>10}  {'A_prime':>10}  {'B':>10}  {'target':>10}  "
          f"{'T2_rank':>8}  {'H_rank':>7}  {'T2_top3':>30}")
    print(f"  {'-'*90}")

    for A, Ap, B, Bp in analogies:
        if any(w not in word_hs for w in [A, Ap, B, Bp]): continue

        # T2 method: query = t2(A') - t2(A) + t2(B)
        tA  = t2_vecs.get(A);  tAp = t2_vecs.get(Ap)
        tB  = t2_vecs.get(B);  tBp = t2_vecs.get(Bp)
        if any(v is None for v in [tA, tAp, tB, tBp]): continue

        query_t2 = tAp - tA + tB
        top5_t2  = nearest_by_t2(query_t2, exclude=(A, Ap, B))
        top_words_t2 = [w for w,_ in top5_t2]
        rank_t2 = top_words_t2.index(Bp) + 1 if Bp in top_words_t2 else ">5"
        correct_t2 = (rank_t2 == 1)

        # Hidden-state method at the axis layer: h(A') - h(A) + h(B)
        hA  = normed(word_hs[A][L]);  hAp = normed(word_hs[Ap][L])
        hB  = normed(word_hs[B][L])
        query_h = hAp - hA + hB
        top5_h  = nearest_by_hidden(query_h, L, exclude=(A, Ap, B))
        top_words_h = [w for w,_ in top5_h]
        rank_h = top_words_h.index(Bp) + 1 if Bp in top_words_h else ">5"
        correct_h = (rank_h == 1)

        n_correct_t2 += correct_t2; n_correct_h += correct_h
        top3_str = ", ".join(f"{w}({s:.2f})" for w,s in top5_t2[:3])
        print(f"  {A:>10}  {Ap:>10}  {B:>10}  {Bp:>10}  "
              f"{'#'+str(rank_t2):>8}  {'#'+str(rank_h):>7}  {top3_str:>30}")

        axis_results.append({
            "A": A, "Ap": Ap, "B": B, "Bp": Bp,
            "t2_rank": rank_t2, "h_rank": rank_h,
            "t2_correct": correct_t2, "h_correct": correct_h,
            "t2_top5": top_words_t2,
        })
        total_analogy += 1

    n_analogies = len(axis_results)
    if n_analogies > 0:
        print(f"  T2 accuracy: {n_correct_t2}/{n_analogies} = "
              f"{100*n_correct_t2/n_analogies:.1f}%  |  "
              f"Hidden accuracy: {n_correct_h}/{n_analogies} = "
              f"{100*n_correct_h/n_analogies:.1f}%")
    total_correct_t2 += n_correct_t2; total_correct_h += n_correct_h
    analogy_results[axis_name] = axis_results
    print()

# ── Exp 2: Self-similarity — is Δ constant within each axis? ─────────────────
print("=" * 72)
print("Exp 2: Self-Similarity Test — Is Δ = proj(A') - proj(A) Constant?")
print("       (TruthSpace predicts: constant Δ per axis)")
print("=" * 72)
print()

self_sim_results = {}
for axis_name, analogies in ANALOGY_SETS.items():
    deltas = []
    for A, Ap, B, Bp in analogies:
        if A not in t2_vecs or Ap not in t2_vecs: continue
        tA = t2_vecs[A]; tAp = t2_vecs[Ap]
        if tA is None or tAp is None: continue
        # Use only the axis-specific dimension
        k = AXIS_NAMES_12.index(axis_name)
        delta_k = float(tAp[k] - tA[k])  # scalar Δ on axis dimension
        deltas.append(delta_k)

    if not deltas: continue
    mean_d = float(np.mean(deltas)); std_d = float(np.std(deltas))
    cv = abs(std_d / mean_d) if abs(mean_d) > 1e-6 else float("inf")
    consistent = (cv < 0.5)
    self_sim_results[axis_name] = {"mean_delta": mean_d, "std_delta": std_d,
                                    "cv": cv, "n": len(deltas)}
    print(f"  {axis_name:>14}:  delta_mean={mean_d:+.4f}  std={std_d:.4f}  "
          f"CV={cv:.3f}  {'CONSISTENT' if consistent else 'VARIABLE'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 118 Summary — T2 Analogy Arithmetic")
print("=" * 72)

t2_acc   = 100 * total_correct_t2 / max(total_analogy, 1)
h_acc    = 100 * total_correct_h  / max(total_analogy, 1)
n_consistent = sum(1 for r in self_sim_results.values() if r["cv"] < 0.5)
n_total_ss = len(self_sim_results)

# Best and worst axes for analogy
per_axis_t2 = {}
for ax, results in analogy_results.items():
    n = len(results); nc = sum(r["t2_correct"] for r in results)
    if n > 0: per_axis_t2[ax] = 100*nc/n
best_ax  = max(per_axis_t2, key=per_axis_t2.get) if per_axis_t2 else "N/A"
worst_ax = min(per_axis_t2, key=per_axis_t2.get) if per_axis_t2 else "N/A"

print(f"""
  Overall T2 analogy accuracy:     {t2_acc:.1f}% ({total_correct_t2}/{total_analogy})
  Overall hidden-state accuracy:   {h_acc:.1f}% ({total_correct_h}/{total_analogy})
  Best axis (T2):  {best_ax} ({per_axis_t2.get(best_ax,0):.1f}%)
  Worst axis (T2): {worst_ax} ({per_axis_t2.get(worst_ax,0):.1f}%)

  Self-similarity (constant Δ per axis):
    Consistent axes (CV < 0.5): {n_consistent}/{n_total_ss}
    Per-axis: {', '.join(f'{ax}({r["cv"]:.2f})' for ax, r in self_sim_results.items())}

  VERDICT:
  {'→ T2 analogy arithmetic WORKS (>50% accuracy): self-similarity CONFIRMED' if t2_acc > 50 else
   '→ T2 analogy arithmetic PARTIAL (25-50%)' if t2_acc > 25 else
   '→ T2 analogy arithmetic FAILS (<25%): self-similarity not confirmed in T2 space'}

  KEY FINDING:
  {'→ The T2 continuous projection space supports word2vec-style analogy arithmetic' if t2_acc > 50 else
   '→ Analogy arithmetic works better in the full hidden-state space than T2 alone' if h_acc > t2_acc + 10 else
   '→ T2 space partially captures relational structure; full hidden state is similar quality'}

  SELF-SIMILARITY:
  {'→ CONFIRMED: axis Δ is approximately constant (CV < 0.5) for ' + str(n_consistent) + '/' + str(n_total_ss) + ' axes' if n_consistent > n_total_ss//2 else
   '→ PARTIAL: some axes show constant Δ, others are variable' if n_consistent > 0 else
   '→ NOT CONFIRMED: axis Δ is variable (CV > 0.5) for most axes'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "analogy_results": analogy_results,
        "self_sim_results": self_sim_results,
        "overall_t2_acc": t2_acc,
        "overall_h_acc": h_acc,
        "per_axis_t2_acc": per_axis_t2,
        "total_correct_t2": total_correct_t2,
        "total_correct_h": total_correct_h,
        "total_analogy": total_analogy,
        "n_consistent": n_consistent,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 118 complete.")
