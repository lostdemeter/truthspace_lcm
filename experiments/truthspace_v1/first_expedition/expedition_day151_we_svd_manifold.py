#!/usr/bin/env python3
"""
Day 151 — SVD Analysis of W_E Vocabulary Manifold

Days 133-150 established: W_E encodes factual/relational knowledge via:
  - Proximity: France≈Paris, hot≈cold
  - Directional: gender_dir, capital_dir, antonym_dir
  - Arithmetic: France+(Japan-Germany)→Tokyo

Question: What is the GLOBAL structure of W_E for our curated vocabulary?

METHOD:
  1. Build matrix M[i,j] = W_E[vocab_i] — shape (N_vocab × H)
  2. SVD: M = U Σ V^T
  3. Analyze top components (V columns = directions in W_E space)
  4. Project vocab words onto each component
  5. Are the top components semantically interpretable?
     Do they match our known universal directions (capital, gender, antonym)?

HYPOTHESIS:
  If the TruthSpace hypothesis is correct, the top SVD components of W_E
  should correspond to semantic categories and match the T2 axes discovered
  in earlier experiments.

  Specifically:
    V1 ≈ ?  (dominant axis in vocabulary embedding)
    V2 ≈ gender direction?
    V3 ≈ capital direction?
    V4 ≈ antonym direction?

ALSO:
  - Compute cosine between SVD components and our known universal directions
  - Project specific word groups (countries, antonyms, gender pairs) onto each component
  - Visualize semantic structure
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day151_we_svd_manifold.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

VOCAB_CURATED = [
    "walked","ran","ate","built","wrote","read","said","went","came","took",
    "made","got","saw","gave","knew","thought","found","told","became","left",
    "brought","bought","taught","caught","fought","heard","held","kept","sent",
    "fell","felt","grew","slept","spent","stood","wore","won",
    "drove","flew","swam","sang","sat","laid","paid","played","stayed","opened",
    "turned","looked","stopped","asked","started","tried","closed","moved","lived",
    "walk","run","eat","build","write","say","come","take","make","get","see",
    "give","know","think","find","tell","become","leave",
    "cold","hot","big","small","fast","slow","dark","light","happy","sad",
    "good","bad","strong","weak","young","old","loud","quiet","easy","hard",
    "clean","dirty","rich","poor","safe","early","late",
    "Paris","London","Rome","Berlin","Madrid","Tokyo","Moscow","Beijing",
    "Sydney","Ottawa","Canberra","Brasilia","Cairo","Delhi","Seoul","Bangkok",
    "Vienna","Warsaw","Athens","Lisbon","Brussels","Amsterdam","Oslo","Stockholm",
    "English","French","Spanish","German","Italian","Portuguese","Arabic",
    "Mandarin","Japanese","Korean","Hindi","Russian","Turkish","Persian",
    "Bengali","Tamil","Urdu","Polish","Dutch","Swedish","Greek",
    "animal","plant","tool","vehicle","food","music","sport","color","number",
    "language","country","city","flower","tree","bird","fish","dog","cat","horse",
    "instrument","weapon","machine","device","metal","mineral","crystal","gem",
    "king","queen","prince","princess","duke","duchess","emperor","empress",
    "father","mother","brother","sister","son","daughter","uncle","aunt",
    "man","woman","boy","girl","actor","actress","hero","heroine",
    "east","west","north","south","morning","evening","night","water","fire",
    "door","house","book","table","chair","window","street","road","park",
    "school","office","market","store","church","castle","palace","bridge",
    "then","also","soon","just","very","still","again","always","never",
    "first","last","next","before","after","here","there",
]

COUNTRY_CAPITAL = {
    "France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid",
    "Japan":"Tokyo","China":"Beijing","Russia":"Moscow","Brazil":"Brasilia",
    "Egypt":"Cairo","Greece":"Athens","Poland":"Warsaw","Sweden":"Stockholm",
}
ANTONYM_PAIRS = [
    ("hot","cold"),("big","small"),("fast","slow"),("dark","light"),
    ("good","bad"),("young","old"),("rich","poor"),("clean","dirty"),
    ("loud","quiet"),("strong","weak"),("early","late"),("easy","hard"),
]
GENDER_PAIRS = [
    ("king","queen"),("prince","princess"),("actor","actress"),
    ("son","daughter"),("father","mother"),("brother","sister"),
    ("man","woman"),("boy","girl"),
]
LANGUAGE_PAIRS = [
    ("Germany","German"),("France","French"),("Spain","Spanish"),
    ("Japan","Japanese"),("Italy","Italian"),("Greece","Greek"),
    ("Poland","Polish"),("Sweden","Swedish"),("Russia","Russian"),
    ("China","Mandarin"),("Korea","Korean"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
print(f"  hidden={H}\n")

def get_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build vocab matrix
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = []
for w in VOCAB:
    if get_tid(w): vocab_ok.append(w)
N = len(vocab_ok)
print(f"Vocabulary: {N} words\n")

M = np.array([W_E[get_tid(w)] for w in vocab_ok], dtype=np.float32)  # (N, H)
M_centered = M - M.mean(axis=0)
M_normed = np.array([normed(row) for row in M], dtype=np.float32)

print("="*66)
print("PART 1: SVD of W_E Vocabulary Matrix")
print("="*66)
print()

# SVD of centered and normed matrices
print("Computing SVD ...")
U_c, S_c, Vt_c = np.linalg.svd(M_centered, full_matrices=False)
U_n, S_n, Vt_n = np.linalg.svd(M_normed, full_matrices=False)
print(f"  Centered:  singular values S[0:6] = {S_c[:6].round(1)}")
print(f"  Normed:    singular values S[0:6] = {S_n[:6].round(1)}")

# Energy explained
total_c = float(np.sum(S_c**2))
print(f"\n  Centered: top-k energy explained:")
for k in [1, 2, 5, 10, 20]:
    e = float(np.sum(S_c[:k]**2)) / total_c
    print(f"    k={k:>3}: {e:.3f}")

print()

# ─────────────────────────────────────────────────────────────
print("="*66)
print("PART 2: Build Universal Directions")
print("="*66)

def make_dir(pairs):
    dirs = []
    for a, b in pairs:
        ea = W_E[get_tid(a)] if get_tid(a) else None
        eb = W_E[get_tid(b)] if get_tid(b) else None
        if ea is not None and eb is not None:
            dirs.append(normed(eb - ea))
    return normed(np.mean(dirs, axis=0))

cap_dir     = make_dir(list(COUNTRY_CAPITAL.items()))
gender_dir  = make_dir(GENDER_PAIRS)
antonym_dir = make_dir(ANTONYM_PAIRS)
lang_dir    = make_dir(LANGUAGE_PAIRS)

known_dirs = {
    "cap_dir":     cap_dir,
    "gender_dir":  gender_dir,
    "antonym_dir": antonym_dir,
    "lang_dir":    lang_dir,
}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 3: Align SVD Components with Known Directions")
print("="*66)
print()

# For each of top-20 SVD components, compute cosine with each known direction
TOP_K = 20
print(f"  Cosine(SVD_k, known_dir) for top-{TOP_K} components:")
print(f"  {'k':>4}  {'cap_dir':>10}  {'gender_dir':>12}  {'antonym_dir':>13}  {'lang_dir':>10}")
print(f"  {'-'*60}")
svd_alignments = []
for k in range(TOP_K):
    comp = Vt_c[k]  # k-th right singular vector (H-dim)
    aligns = {name: cosine(comp, d) for name, d in known_dirs.items()}
    svd_alignments.append({"k": k, "S": float(S_c[k]), **aligns,
                            "best": max(aligns, key=lambda n: abs(aligns[n]))})
    print(f"  {k:>4}: {aligns['cap_dir']:>10.3f}  {aligns['gender_dir']:>12.3f}  "
          f"{aligns['antonym_dir']:>13.3f}  {aligns['lang_dir']:>10.3f}")

# Best alignments
print()
for dir_name in known_dirs:
    best_k = max(range(TOP_K), key=lambda k: abs(svd_alignments[k][dir_name]))
    best_cos = svd_alignments[best_k][dir_name]
    print(f"  Best SVD alignment with {dir_name}: k={best_k}, cos={best_cos:.3f}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 4: Word Projections onto Top SVD Components")
print("="*66)
print()

# Project each word onto top-5 SVD components and identify semantic clusters
word_projections = {}
for i, w in enumerate(vocab_ok):
    word_projections[w] = {
        k: float(U_c[i, k] * S_c[k]) for k in range(5)
    }

# For each component, show top-5 and bottom-5 words
for k in range(5):
    scores = [(w, word_projections[w][k]) for w in vocab_ok]
    scores.sort(key=lambda x: -x[1])
    top5  = [w for w,_ in scores[:5]]
    bot5  = [w for w,_ in scores[-5:]]
    # Find category composition of extremes
    print(f"  Component {k} (S={S_c[k]:.1f}):")
    print(f"    Positive: {top5}")
    print(f"    Negative: {bot5}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 5: Semantic Category Separation on Top-2 Components")
print("="*66)
print()

# Project CATEGORY GROUPS onto top-2 SVD components and check separation
GROUPS = {
    "cities":     list(COUNTRY_CAPITAL.values()),
    "countries":  list(COUNTRY_CAPITAL.keys()),
    "languages":  [b for _,b in LANGUAGE_PAIRS],
    "antonym_A":  [a for a,b in ANTONYM_PAIRS],
    "antonym_B":  [b for a,b in ANTONYM_PAIRS],
    "masc":       [a for a,b in GENDER_PAIRS],
    "fem":        [b for a,b in GENDER_PAIRS],
    "past_verbs": ["walked","ran","ate","said","went","came","took","made"],
    "pres_verbs": ["walk","run","eat","say","come","take","make","get"],
}

print(f"  {'Group':>14}  {'n':>4}  {'PC0_mean':>10}  {'PC1_mean':>10}  {'PC0_std':>10}")
group_stats = {}
for gname, gwords in GROUPS.items():
    gw_ok = [w for w in gwords if w in vocab_ok]
    if not gw_ok: continue
    projs = np.array([[word_projections[w][k] for k in range(2)] for w in gw_ok])
    group_stats[gname] = {
        "pc0_mean": float(projs[:,0].mean()),
        "pc1_mean": float(projs[:,1].mean()),
        "pc0_std":  float(projs[:,0].std()),
    }
    print(f"  {gname:>14}: n={len(gw_ok):>3}  PC0={projs[:,0].mean():>+8.1f}  "
          f"PC1={projs[:,1].mean():>+8.1f}  std={projs[:,0].std():>8.1f}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 6: Reconstruct Universal Directions from SVD")
print("="*66)
print()

# How many SVD components are needed to reconstruct gender_dir at 90% cosine?
for dir_name, dir_vec in known_dirs.items():
    cumsum_proj = np.zeros(H, dtype=np.float64)
    cos_at_k = []
    for k in range(50):
        comp = Vt_c[k].astype(np.float64)
        proj = float(np.dot(dir_vec.astype(np.float64), comp))
        cumsum_proj += proj * comp
        cos_k = cosine(cumsum_proj, dir_vec.astype(np.float64))
        cos_at_k.append(cos_k)

    k90 = next((k for k,c in enumerate(cos_at_k) if c >= 0.90), 50)
    k95 = next((k for k,c in enumerate(cos_at_k) if c >= 0.95), 50)
    print(f"  {dir_name}: k@90%={k90}  k@95%={k95}  "
          f"cos@5={cos_at_k[4]:.3f}  cos@20={cos_at_k[19]:.3f}  cos@50={cos_at_k[49]:.3f}")

print()
print("="*66)
print("Summary")
print("="*66)

# Key question: are top SVD components interpretable?
best_aligned = {name: max(range(TOP_K), key=lambda k: abs(svd_alignments[k][name]))
                for name in known_dirs}
print(f"""
  SVD of W_E curated vocabulary:
    N_words = {N},  H = {H}
    Top singular values: {S_c[:5].round(1).tolist()}

  Universal direction alignment with SVD:
""")
for name in known_dirs:
    k = best_aligned[name]
    c = svd_alignments[k][name]
    print(f"    {name:>14}: best at k={k}, cos={c:.3f}  {'(ALIGNED)' if abs(c)>=0.3 else '(weak)'}")

print()
print(f"  Category group PC0 range:")
pc0_vals = sorted(group_stats.items(), key=lambda x: x[1]["pc0_mean"])
for gname, s in pc0_vals:
    print(f"    {gname:>14}: PC0={s['pc0_mean']:>+7.1f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "svd_alignments": svd_alignments,
        "group_stats": group_stats,
        "singular_values": S_c[:20].tolist(),
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 151 complete.")
