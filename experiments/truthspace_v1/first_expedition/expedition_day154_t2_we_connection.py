#!/usr/bin/env python3
"""
Day 154 — T2 ↔ W_E Connection: Are L25 Axes the Same as W_E SVD Axes?

T2 coordinate system (Days 73-132): axes derived from L25 hidden states
  in full-context prompts. Achieved 97-100% categorical accuracy.

W_E SVD (Day 151): PC0=named-entity, PC1=verb-tense, PC2=royalty,
  PC3=capital-direction. Found via SVD of raw embedding matrix.

QUESTION: Are the principal directions of L25 hidden states (in context)
  the same as the W_E SVD directions? Does the geometry PRESERVE across layers?

METHOD:
  1. Build W_E matrix (N × H) and compute SVD → Vt_0
  2. Build L25 matrix: for each vocab word, run a minimal context prompt
     "[BOS] word" or "The word is [word]" and take L25 entity hidden state
  3. Build CATEGORY-SPECIFIC L25 matrix:
     - capital words: " France", " Germany", ... at L25 (last token of " Country")
     - antonym words: at L25 via " hot", " cold", etc.
  4. Compute alignment: cosine(Vt_0[k], Vt_25[j]) for all k, j
  5. Report maximum cross-layer alignment scores
  6. Compare: do L25 SVD components align with W_E cap/gender/antonym directions?

HYPOTHESIS A (T2 preserves W_E): cos(Vt_25[k], Vt_0[k]) ≈ 0.5-0.9 for top-k
HYPOTHESIS B (T2 independent): cos(Vt_25[k], Vt_0[k]) ≈ 0  (different axes)
HYPOTHESIS C (T2 sharpens): Vt_25 aligns MORE strongly with known directions
  than Vt_0 does (cos(Vt_25, cap_dir) > cos(Vt_0, cap_dir) = 0.445)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day154_t2_we_connection.json")
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

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
n_layers = model.config.num_hidden_layers
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
print(f"  hidden={H}, layers={n_layers}\n")

def get_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_hidden_at_layer(word, layer):
    """Get hidden state of word token at given layer, single-token context."""
    inp = tok(" "+word, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

def get_hidden_full_context(template, word, layer):
    """Get hidden state of word token at given layer in full context."""
    prompt = template.replace("[WORD]", word)
    inp = tok(prompt, return_tensors="pt")
    # Find position of word token in the input
    word_id = get_tid(word)
    if word_id is None: return None
    ids = inp["input_ids"][0].tolist()
    # Find last occurrence of word_id
    positions = [i for i, t in enumerate(ids) if t == word_id]
    if not positions: return None
    pos = positions[-1]
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

# ─────────────────────────────────────────────────────────────
# Build vocab
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = [w for w in VOCAB if get_tid(w)]
N = len(vocab_ok)
print(f"Vocabulary: {N} words\n")

# Build W_E matrix
M_0 = np.array([W_E[get_tid(w)] for w in vocab_ok], dtype=np.float32)
M_0_centered = M_0 - M_0.mean(axis=0)
mean_0 = M_0.mean(axis=0)

print("Computing W_E SVD ...")
_, S_0, Vt_0 = np.linalg.svd(M_0_centered, full_matrices=False)
print(f"  S_0[:6] = {S_0[:6].round(3)}\n")

# Build universal directions
def make_dir(pairs):
    ds = [normed(W_E[get_tid(b)] - W_E[get_tid(a)])
          for a, b in pairs if get_tid(a) and get_tid(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

cap_dir     = make_dir(list(COUNTRY_CAPITAL.items()))
gender_dir  = make_dir(GENDER_PAIRS)
antonym_dir = make_dir(ANTONYM_PAIRS)
known_dirs  = {"cap_dir": cap_dir, "gender_dir": gender_dir, "antonym_dir": antonym_dir}

# ─────────────────────────────────────────────────────────────
print("="*66)
print("PART 1: Single-Token L25 Hidden States (vs W_E)")
print("="*66)
print()

print("Computing L25 hidden states for all vocab words (single-token) ...")
M_25 = np.array([get_hidden_at_layer(w, n_layers) for w in vocab_ok], dtype=np.float32)
M_25_centered = M_25 - M_25.mean(axis=0)

_, S_25, Vt_25 = np.linalg.svd(M_25_centered, full_matrices=False)
print(f"  S_25[:6] = {S_25[:6].round(3)}\n")

# Cross-alignment: W_E SVD vs L25 SVD
print("Cross-alignment matrix (top-10 W_E vs top-10 L25 components):")
print(f"  {'':>6}  " + "  ".join(f"L25_{j:>2}" for j in range(10)))
cross_aligns = np.zeros((10, 10))
for k in range(10):
    row = []
    for j in range(10):
        c = cosine(Vt_0[k], Vt_25[j])
        cross_aligns[k, j] = abs(c)
        row.append(f"{c:>+6.3f}")
    print(f"  W_E_{k:>2}:  " + "  ".join(row))

print()
print("  Max cross-alignment per W_E component:")
for k in range(10):
    best_j = int(np.argmax(cross_aligns[k]))
    print(f"  W_E_{k}: max_cos={cross_aligns[k, best_j]:.3f} at L25_{best_j}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 2: L25 SVD Alignment with Known Directions")
print("="*66)
print()

print("  L25 SVD alignment with universal directions:")
print(f"  {'k':>4}  {'cap_dir':>10}  {'gender_dir':>12}  {'antonym_dir':>13}")
l25_aligns = []
for k in range(20):
    aligns = {name: cosine(Vt_25[k], d) for name, d in known_dirs.items()}
    l25_aligns.append({"k": k, "S": float(S_25[k]), **aligns})
    print(f"  {k:>4}: {aligns['cap_dir']:>10.3f}  {aligns['gender_dir']:>12.3f}  {aligns['antonym_dir']:>13.3f}")

print()
for dir_name in known_dirs:
    best_k_l0  = max(range(20), key=lambda k: abs(cosine(Vt_0[k],  known_dirs[dir_name])))
    best_k_l25 = max(range(20), key=lambda k: abs(cosine(Vt_25[k], known_dirs[dir_name])))
    c_l0  = cosine(Vt_0[best_k_l0],   known_dirs[dir_name])
    c_l25 = cosine(Vt_25[best_k_l25], known_dirs[dir_name])
    diff = abs(c_l25) - abs(c_l0)
    arrow = "↑ STRONGER" if diff > 0.05 else "↓ WEAKER" if diff < -0.05 else "≈ SAME"
    print(f"  {dir_name:>14}: W_E best={c_l0:>+.3f} at k={best_k_l0}  "
          f"L25 best={c_l25:>+.3f} at k={best_k_l25}  {arrow}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 3: Full-Context L25 Hidden States")
print("="*66)
print()

# Use context prompts that reveal relational structure
TEMPLATES = {
    "capital":  "The capital city of [WORD] is",
    "language": "The official language of [WORD] is",
    "antonym":  "The opposite of [WORD] is",
    "gender":   "The [WORD] and the",
    "hypernym": "[WORD] is a type of",
    "neutral":  "[WORD]",
}

# Build full-context L25 matrices for each template (on subset of vocab)
SUBSETS = {
    "capital":  list(COUNTRY_CAPITAL.keys()),
    "language": list(COUNTRY_CAPITAL.keys()),
    "antonym":  [a for a,b in ANTONYM_PAIRS] + [b for a,b in ANTONYM_PAIRS],
    "gender":   [a for a,b in GENDER_PAIRS]  + [b for a,b in GENDER_PAIRS],
}

full_ctx_results = {}
for task, template in [("capital","The capital city of [WORD] is"),
                        ("antonym","The opposite of [WORD] is"),
                        ("gender","The [WORD] and the")]:
    words = SUBSETS.get(task, vocab_ok[:20])
    words = [w for w in words if get_tid(w)]
    
    hs_list = []
    for w in words:
        h = get_hidden_full_context(template, w, n_layers)
        if h is not None: hs_list.append((w, h))

    if len(hs_list) < 4:
        print(f"  {task}: too few words ({len(hs_list)}), skipping")
        continue

    ws, hs = zip(*hs_list)
    M_task = np.array(hs, dtype=np.float32)
    M_task_c = M_task - M_task.mean(axis=0)

    _, S_t, Vt_t = np.linalg.svd(M_task_c, full_matrices=False)

    # Alignment with known directions
    aligns = {}
    for dir_name, dir_vec in known_dirs.items():
        best_k = max(range(min(10, len(S_t))),
                     key=lambda k: abs(cosine(Vt_t[k], dir_vec)))
        aligns[dir_name] = cosine(Vt_t[best_k], dir_vec)

    # Word projections on PC0
    pc0_projs = [(w, float(M_task_c[i] @ Vt_t[0])) for i,w in enumerate(ws)]
    pc0_projs.sort(key=lambda x: -x[1])

    print(f"  Task={task} ({len(ws)} words), S[:4]={S_t[:4].round(2)}")
    print(f"    cap_dir cos={aligns['cap_dir']:.3f}  gender_dir cos={aligns['gender_dir']:.3f}  antonym cos={aligns['antonym_dir']:.3f}")
    print(f"    PC0 top: {[w for w,_ in pc0_projs[:3]]}  bottom: {[w for w,_ in pc0_projs[-3:]]}")

    full_ctx_results[task] = {"words": list(ws), "S": S_t[:6].tolist(), "aligns": aligns}

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 4: Residual Stream Alignment Across All Layers")
print("="*66)
print()

# For a set of key words, track how their L0 proximity is preserved across layers
PROBE_WORDS = [
    ("France", "Paris"),
    ("Germany", "German"),
    ("hot", "cold"),
    ("king", "queen"),
]
PROBE_LAYERS = [0, 3, 6, 10, 14, 20, 25]

print("  Cosine similarity between key pairs across layers:")
print(f"  {'Pair':>20}  " + "  ".join(f"L{L:>2}" for L in PROBE_LAYERS))
pair_results = {}
for wa, wb in PROBE_WORDS:
    row = []
    for L in PROBE_LAYERS:
        ha = get_hidden_at_layer(wa, L)
        hb = get_hidden_at_layer(wb, L)
        c = cosine(ha, hb)
        row.append(c)
    pair_results[f"{wa}-{wb}"] = dict(zip(PROBE_LAYERS, row))
    row_str = "  ".join(f"{c:>+.3f}" for c in row)
    print(f"  {wa:>8}↔{wb:<10}: {row_str}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("Summary: T2 ↔ W_E Connection")
print("="*66)
print()

# Top cross-alignment
best_cross = {}
for k in range(10):
    best_j = int(np.argmax(cross_aligns[k]))
    best_cross[k] = (best_j, float(cross_aligns[k, best_j]))

max_cross = max(best_cross.values(), key=lambda x: x[1])
print(f"  Max W_E↔L25 SVD alignment:  {max_cross[1]:.3f}  (W_E_k → L25_{max_cross[0]})")

# Direction alignment comparison
print(f"\n  Known direction alignment (top-20 SVD components):")
print(f"  {'Direction':>14}  {'W_E best':>12}  {'L25 best':>12}  verdict")
for dir_name, dir_vec in known_dirs.items():
    c_l0  = max(abs(cosine(Vt_0[k],  dir_vec)) for k in range(20))
    c_l25 = max(abs(cosine(Vt_25[k], dir_vec)) for k in range(20))
    verdict = "L25 STRONGER" if c_l25 > c_l0 + 0.05 else \
              "W_E STRONGER" if c_l0  > c_l25 + 0.05 else "SIMILAR"
    print(f"  {dir_name:>14}: W_E={c_l0:.3f}  L25={c_l25:.3f}  {verdict}")

print()
print("  Pair cosine at L0 vs L25 (proxy for T2 preservation):")
for pair, vals in pair_results.items():
    c_l0  = vals[0]
    c_l25 = vals[PROBE_LAYERS[-1]]
    delta = c_l25 - c_l0
    print(f"  {pair:>20}: L0={c_l0:.3f}  L25={c_l25:.3f}  Δ={delta:>+.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "cross_aligns": cross_aligns.tolist(),
        "l25_aligns": l25_aligns,
        "full_ctx_results": full_ctx_results,
        "pair_results": {k: {str(l): v for l,v in vals.items()}
                         for k, vals in pair_results.items()},
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 154 complete.")
