#!/usr/bin/env python3
"""
Day 122 — PCA Independent Axis Validation

QUESTION: Are the T2 axes intrinsic to the LM, or artifacts of the
specific sentence pairs used to construct them?

METHOD: For each semantic axis, independently derive the separating
direction from LM hidden states using:
  1. Mean-difference (MD): axis = mean(class_B) - mean(class_A), normed
  2. LDA direction: Fisher discriminant — maximize inter-class / intra-class
  3. PCA on class-mean-centered data

Then measure: cosine(T2_axis, MD_axis) and cosine(T2_axis, LDA_axis)

If T2 axes are INTRINSIC to the LM:
  → High alignment (cos > 0.5) between T2 sentence-pair axis and independently
    derived class-mean / LDA direction

If T2 axes are CONSTRUCTION ARTIFACTS:
  → Low alignment (cos near 0) — the specific sentence pairs determine direction,
    not the underlying LM geometry

Also test: cross-axis independence
  → MD_axis for gender should NOT align with MD_axis for plural (they're different!)

PREDICTION (TruthSpace hypothesis):
  High alignment between sentence-pair T2 axes and independently-derived
  LDA directions, because the LM encodes these categories as geometric
  subspaces that any derivation method would find.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day122_pca_axis_validation.json")
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
        ("The prince rode across the land","The princess rode across the land"),
        ("The uncle visited the old house","The aunt visited the old house"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
        ("The dark room","The darker room"),("A hard rock","A harder rock"),
    ],
    "hypernym": [
        ("The dog ran from danger","The animal ran from danger"),
        ("A rose bloomed in garden","A flower bloomed in garden"),
        ("The car sped past sign","The vehicle sped past sign"),
        ("The eagle soared above","The bird soared above"),
        ("The ruby gleamed bright","The gem gleamed bright"),
        ("The hammer struck nail","The tool struck nail"),
        ("The oak fell in storm","The tree fell in storm"),
        ("The salmon swam upstream","The fish swam upstream"),
    ],
    "plural": [
        ("A dog played in field","Dogs played in field"),
        ("The cat sat by window","The cats sat by window"),
        ("A bird sang in mist","Birds sang in mist"),
        ("The tree fell in storm","The trees fell in storm"),
        ("A book sat on desk","Books sat on desk"),
        ("The car drove down road","The cars drove down road"),
        ("A star shone in sky","Stars shone in sky"),
        ("The word appeared in text","The words appeared in text"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He spoke loud","He spoke noisy"),
        ("It is hard","It is difficult"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is heavy","The burden is heavy"),
        ("The iron chain broke","The bond between broke"),
        ("The long road leads","The long journey leads"),
        ("The high wall blocks","The high barrier blocks"),
        ("The flame fades away","The hope fades away"),
        ("The root grips soil","The base grips earth"),
        ("The bridge connects banks","The bond connects them"),
        ("The key opens door","The answer opens path"),
    ],
    "past_tense": [
        ("I walk every morning","I walked every morning"),
        ("She runs through park","She ran through park"),
        ("He eats before leaving","He ate before leaving"),
        ("They build the wall","They built the wall"),
        ("We swim in lake","We swam in lake"),
        ("She writes a letter","She wrote a letter"),
        ("He speaks at meeting","He spoke at meeting"),
        ("They sing at fire","They sang at fire"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The light is on","The dark is on"),("The news is good","The news is bad"),
        ("It is hard","It is soft"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is first","It is last"),
    ],
    "passive": [
        ("The cat chased mouse","The mouse was chased"),
        ("John broke window","The window was broken"),
        ("The chef cooked meal","The meal was cooked"),
        ("The dog bit man","The man was bitten"),
        ("The teacher helped student","The student was helped"),
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
        ("The ball rolls off edge","The ball falls down"),
        ("The teacher praises work","The student feels proud"),
        ("The glass breaks on stone","The water spills out"),
    ],
    "question": [
        ("She is tired today","Is she tired today"),
        ("He can swim well","Can he swim well"),
        ("They went to market","Did they go to market"),
        ("The car broke down","Did the car break down"),
        ("The dog is hungry","Is the dog hungry"),
        ("She wrote the letter","Did she write letter"),
        ("He knows the answer","Does he know answer"),
        ("The house looks old","Does the house look old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today","It will not rain today"),
    ],
}

# Independent category word sets for each axis
# class_A = "base" class, class_B = "transformed" class
CATEGORY_WORDS = {
    "gender": {
        "A": ["king","man","boy","father","son","actor","brother","uncle",
              "prince","husband","grandfather","nephew","monk","wizard","lord"],
        "B": ["queen","woman","girl","mother","daughter","actress","sister","aunt",
              "princess","wife","grandmother","niece","nun","witch","lady"],
    },
    "comparative": {
        "A": ["fast","big","old","cold","tall","bright","dark","hard",
              "small","warm","fresh","clean","soft","long","strong"],
        "B": ["faster","bigger","older","colder","taller","brighter","darker","harder",
              "smaller","warmer","fresher","cleaner","softer","longer","stronger"],
    },
    "hypernym": {
        "A": ["dog","rose","car","eagle","ruby","hammer","oak","salmon",
              "tulip","hawk","wrench","pine","trout","emerald","van"],
        "B": ["animal","flower","vehicle","bird","gem","tool","tree","fish",
              "plant","raptor","instrument","plant","vertebrate","mineral","transport"],
    },
    "plural": {
        "A": ["dog","cat","tree","bird","book","car","star","hand",
              "eye","man","woman","child","thought","year","house"],
        "B": ["dogs","cats","trees","birds","books","cars","stars","hands",
              "eyes","men","women","children","thoughts","years","houses"],
    },
    "synonym": {
        "A": ["big","small","fast","cold","happy","hard","sad","old",
              "tired","smart","angry","brave","clean","rich","dark"],
        "B": ["large","tiny","quick","frigid","joyful","difficult","unhappy","aged",
              "exhausted","intelligent","furious","courageous","spotless","wealthy","dim"],
    },
    "concrete": {
        "A": ["stone","road","wall","flame","chain","bridge","root","key",
              "hammer","sword","mountain","river","door","lock","iron"],
        "B": ["burden","journey","barrier","hope","bond","connection","foundation","solution",
              "force","power","challenge","flow","opportunity","security","strength"],
    },
    "past_tense": {
        "A": ["walk","run","eat","see","build","swim","write","fly",
              "speak","break","take","give","make","go","come"],
        "B": ["walked","ran","ate","saw","built","swam","wrote","flew",
              "spoke","broke","took","gave","made","went","came"],
    },
    "antonym": {
        "A": ["hot","fast","good","happy","strong","old","big","light",
              "open","high","early","rich","clean","loud","alive"],
        "B": ["cold","slow","bad","sad","weak","young","small","dark",
              "closed","low","late","poor","dirty","quiet","dead"],
    },
    "passive": {
        "A": ["chases","breaks","cooks","destroys","helps","paints","writes","builds"],
        "B": ["chased","broken","cooked","destroyed","helped","painted","written","built"],
    },
    "causation": {
        "A": ["rain","fire","heat","wind","pressure","friction","gravity","impact"],
        "B": ["flood","ash","melt","fall","collapse","spark","fall","crack"],
    },
    "question": {
        "A": ["is","can","does","was","will","has","are","did"],
        "B": ["Is","Can","Does","Was","Will","Has","Are","Did"],
    },
    "negation": {
        "A": ["fast","good","strong","happy","clean","loud","alive","open"],
        "B": ["slow","bad","weak","sad","dirty","quiet","dead","closed"],
    },
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def lda_direction(class_a_vecs, class_b_vecs):
    """Fisher LDA direction: maximize inter-class / intra-class variance."""
    a = np.array(class_a_vecs); b = np.array(class_b_vecs)
    mu_a = np.mean(a, axis=0); mu_b = np.mean(b, axis=0)
    # Between-class: (mu_b - mu_a)
    mean_diff = mu_b - mu_a
    # Within-class scatter: S_w = cov_a + cov_b
    S_a = np.cov(a.T) if len(a) > 1 else np.eye(a.shape[1]) * 1e-6
    S_b = np.cov(b.T) if len(b) > 1 else np.eye(b.shape[1]) * 1e-6
    S_w = S_a + S_b + np.eye(S_a.shape[0]) * 1e-4  # regularize
    try:
        w = np.linalg.solve(S_w, mean_diff)
    except np.linalg.LinAlgError:
        w = mean_diff  # fallback to mean-diff
    n = np.linalg.norm(w)
    return (w / n if n > 1e-8 else mean_diff / (np.linalg.norm(mean_diff)+1e-8)).astype(np.float32)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}\n")

# Step 1: Compute T2 sentence-pair axes (reference)
print("Computing T2 sentence-pair axes ...")
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

# Step 2: Extract hidden states for category word sets
print("Extracting category word hidden states ...")
cat_hs = {}  # {ax_name: {"A": [...], "B": [...]}}
for ax_name in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax_name]
    cat_hs[ax_name] = {"A": [], "B": []}
    for cls in ["A", "B"]:
        for w in CATEGORY_WORDS[ax_name][cls]:
            inp = tok(" " + w, return_tensors="pt")
            try:
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                pos = inp["input_ids"].shape[1] - 1
                h = out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                cat_hs[ax_name][cls].append(h)
            except: pass
print("  Done.\n")

# Step 3: Compute independent axes via mean-difference and LDA
print("Computing independent axes (mean-diff and LDA) ...")
md_axes  = {}
lda_axes = {}
for ax_name in AXIS_NAMES_12:
    a_vecs = cat_hs[ax_name]["A"]; b_vecs = cat_hs[ax_name]["B"]
    if len(a_vecs) < 2 or len(b_vecs) < 2:
        md_axes[ax_name]  = np.zeros(hidden_size, np.float32)
        lda_axes[ax_name] = np.zeros(hidden_size, np.float32)
        continue
    mu_a = np.mean(a_vecs, axis=0); mu_b = np.mean(b_vecs, axis=0)
    diff = mu_b - mu_a; n = np.linalg.norm(diff)
    md_axes[ax_name] = (diff/n if n > 1e-8 else diff).astype(np.float32)
    # LDA: too expensive in 1536D, use projected LDA on top-50 PCA
    all_vecs = np.array(a_vecs + b_vecs)
    all_c = all_vecs - np.mean(all_vecs, axis=0)
    try:
        U, S, Vt = np.linalg.svd(all_c, full_matrices=False)
        proj_basis = Vt[:50, :]  # top 50 principal components
        a_proj = [normed(v) @ proj_basis.T for v in a_vecs]
        b_proj = [normed(v) @ proj_basis.T for v in b_vecs]
        lda_dir_proj = lda_direction(a_proj, b_proj)  # 50D LDA
        lda_dir_full = (lda_dir_proj @ proj_basis).astype(np.float32)
        n2 = np.linalg.norm(lda_dir_full)
        lda_axes[ax_name] = lda_dir_full/n2 if n2 > 1e-8 else md_axes[ax_name]
    except:
        lda_axes[ax_name] = md_axes[ax_name]
print("  Done.\n")

# ── Exp 1: T2 vs MD alignment ─────────────────────────────────────────────────
print("=" * 72)
print("Exp 1: T2 sentence-pair axis vs Mean-Difference (MD) axis cosine")
print("       (intrinsic test: do both methods find the SAME direction?)")
print("=" * 72)
print(f"\n  {'axis':>14}  {'opt_L':>6}  {'cos(T2,MD)':>12}  {'cos(T2,LDA)':>13}  "
      f"{'cos(MD,LDA)':>13}  {'verdict':>12}")
print(f"  {'-'*75}")

alignment_results = {}
t2_md_coss  = []
t2_lda_coss = []
for ax_name in AXIS_NAMES_12:
    L      = DAY78_LAYERS[ax_name]
    t2     = t2_axes[ax_name]
    md     = md_axes[ax_name]
    lda    = lda_axes[ax_name]
    c_t2_md  = float(abs(np.dot(t2, md)))
    c_t2_lda = float(abs(np.dot(t2, lda)))
    c_md_lda = float(abs(np.dot(md, lda)))
    t2_md_coss.append(c_t2_md); t2_lda_coss.append(c_t2_lda)
    verdict = ("ALIGNED" if c_t2_md > 0.5 else
               "PARTIAL" if c_t2_md > 0.2 else "DIVERGENT")
    alignment_results[ax_name] = {
        "L": L, "cos_t2_md": c_t2_md, "cos_t2_lda": c_t2_lda, "cos_md_lda": c_md_lda
    }
    print(f"  {ax_name:>14}  {f'L{L:02d}':>6}  {c_t2_md:>12.4f}  {c_t2_lda:>13.4f}  "
          f"{c_md_lda:>13.4f}  {verdict:>12}")

print(f"\n  Mean cos(T2, MD):  {np.mean(t2_md_coss):.4f}")
print(f"  Mean cos(T2, LDA): {np.mean(t2_lda_coss):.4f}")

# ── Exp 2: Cross-axis independence check ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Cross-Axis Independence — MD axes for different categories")
print("       (aligned = bad: axes bleed into each other)")
print("=" * 72)
print()
print(f"  Off-diagonal cosines (|cos|) between MD axes:")
print(f"  (diagonal = 1.000 by definition, lower off-diag = more independent)")
print()

md_matrix = np.stack([md_axes[ax] for ax in AXIS_NAMES_12])
G_md = md_matrix @ md_matrix.T
upper_md = np.abs(G_md[np.triu_indices(12, k=1)])
print(f"  Mean off-diagonal |cos|:  {np.mean(upper_md):.4f}")
print(f"  Max off-diagonal |cos|:   {np.max(upper_md):.4f}")
print()
# Print top 5 most-aligned MD pairs
pairs_sorted = sorted(
    [(AXIS_NAMES_12[i], AXIS_NAMES_12[j], float(abs(G_md[i,j])))
     for i in range(12) for j in range(i+1,12)],
    key=lambda x: -x[2]
)
print(f"  Top 5 most-aligned MD pairs:")
for a, b, c in pairs_sorted[:5]:
    print(f"    {a:>14} / {b:>14}: |cos| = {c:.4f}")

# ── Exp 3: Can MD axis reproduce T2 trie LOO? ─────────────────────────────────
print()
print("=" * 72)
print("Exp 3: MD Axis Classification Accuracy")
print("       (can MD direction separate the category words correctly?)")
print("=" * 72)
print(f"\n  {'axis':>14}  {'MD_acc':>8}  {'T2_acc':>8}  {'n_words':>9}")
print(f"  {'-'*40}")

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

def phi_bin(x):
    if x >  INV_PHI:  return "H"
    elif x < -INV_PHI2: return "L"
    else: return "U"

class_acc_results = {}
for ax_name in AXIS_NAMES_12:
    L  = DAY78_LAYERS[ax_name]
    md = md_axes[ax_name]; t2 = t2_axes[ax_name]
    a_vecs = cat_hs[ax_name]["A"]; b_vecs = cat_hs[ax_name]["B"]
    if not a_vecs or not b_vecs:
        print(f"  {ax_name:>14}  {'N/A':>8}  {'N/A':>8}  {0:>9}"); continue
    n_words = len(a_vecs) + len(b_vecs)
    # MD: project each class onto MD axis, check sign
    a_md = [float(np.dot(normed(v), md)) for v in a_vecs]
    b_md = [float(np.dot(normed(v), md)) for v in b_vecs]
    md_sign = 1 if np.mean(b_md) > np.mean(a_md) else -1
    md_correct = sum(1 for p in a_md if md_sign * p < 0) + \
                 sum(1 for p in b_md if md_sign * p > 0)
    md_acc = md_correct / n_words
    # T2: same check with T2 axis
    a_t2 = [float(np.dot(normed(v), t2)) for v in a_vecs]
    b_t2 = [float(np.dot(normed(v), t2)) for v in b_vecs]
    t2_sign = 1 if np.mean(b_t2) > np.mean(a_t2) else -1
    t2_correct = sum(1 for p in a_t2 if t2_sign * p < 0) + \
                 sum(1 for p in b_t2 if t2_sign * p > 0)
    t2_acc = t2_correct / n_words
    class_acc_results[ax_name] = {"md_acc": md_acc, "t2_acc": t2_acc}
    print(f"  {ax_name:>14}  {100*md_acc:>7.1f}%  {100*t2_acc:>7.1f}%  {n_words:>9}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 122 Summary — PCA Independent Axis Validation")
print("=" * 72)

n_aligned = sum(1 for ax, r in alignment_results.items() if r["cos_t2_md"] > 0.5)
n_partial  = sum(1 for ax, r in alignment_results.items() if 0.2 < r["cos_t2_md"] <= 0.5)
n_diverg   = sum(1 for ax, r in alignment_results.items() if r["cos_t2_md"] <= 0.2)
mean_t2_md = float(np.mean(t2_md_coss))
best_aligned  = max(alignment_results, key=lambda ax: alignment_results[ax]["cos_t2_md"])
worst_aligned = min(alignment_results, key=lambda ax: alignment_results[ax]["cos_t2_md"])
md_gram_mean  = float(np.mean(upper_md))

valid_md = {ax: r for ax, r in class_acc_results.items() if isinstance(r.get("md_acc"), float)}
mean_md_acc = float(np.mean([r["md_acc"] for r in valid_md.values()])) if valid_md else 0
mean_t2_acc = float(np.mean([r["t2_acc"] for r in valid_md.values()])) if valid_md else 0

print(f"""
  T2 sentence-pair axis vs MD axis alignment:
    Aligned (cos > 0.5): {n_aligned}/12
    Partial (0.2-0.5):   {n_partial}/12
    Divergent (< 0.2):   {n_diverg}/12
    Mean cos(T2, MD):    {mean_t2_md:.4f}
    Best:   {best_aligned} ({alignment_results[best_aligned]['cos_t2_md']:.4f})
    Worst:  {worst_aligned} ({alignment_results[worst_aligned]['cos_t2_md']:.4f})

  MD axis classification accuracy:
    Mean MD acc:  {100*mean_md_acc:.1f}%
    Mean T2 acc:  {100*mean_t2_acc:.1f}%

  MD inter-axis independence:
    Mean off-diagonal |cos|: {md_gram_mean:.4f}
    (T2 gram offdiag mean was 0.0616 at Day78 optimal layers)

  VERDICT:
  {'→ T2 axes ARE intrinsic to LM: high alignment (cos > 0.5) for ' + str(n_aligned) + '/12 axes' if n_aligned >= 6 else
   '→ T2 axes PARTIALLY intrinsic: some alignment (mean cos=' + f'{mean_t2_md:.3f}' + ')' if mean_t2_md > 0.2 else
   '→ T2 axes are CONSTRUCTION-SPECIFIC: low alignment with independently-derived directions'}

  KEY FINDING:
  {'→ Different derivation methods find the SAME geometric direction: semantic axes are universal LM properties' if n_aligned >= 8 else
   '→ Partial alignment: T2 sentence-pair method captures some but not all of the intrinsic axis direction' if mean_t2_md > 0.2 else
   '→ T2 sentence-pair axes are specific to the construction sentences, not intrinsic LM geometry'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "alignment_results": alignment_results,
        "class_acc_results": class_acc_results,
        "mean_t2_md": mean_t2_md,
        "mean_t2_lda": float(np.mean(t2_lda_coss)),
        "md_gram_offdiag_mean": md_gram_mean,
        "n_aligned": n_aligned,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 122 complete.")
