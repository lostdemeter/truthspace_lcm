#!/usr/bin/env python3
"""
Day 152 — SVD-Projected entity_excl: Does Subspace Projection Help?

Day 151: SVD of W_E vocabulary. PC3 = capital direction (cos=0.445).
Question: Does projecting embeddings onto top-K SVD components BEFORE
computing cosine similarity improve entity_excl accuracy?

Intuition:
  - Full 1536D has noise dimensions
  - Top-K SVD captures semantic variance
  - Projecting onto semantic subspace might sharpen proximity signals

SWEEP: K in [2, 5, 10, 20, 50, 100, 200, 1536]
METHOD: entity_excl using cosine in K-dimensional SVD subspace

Compare per-category and overall vs Day 148 baseline (82.8% routed).

Also test: CATEGORY-SPECIFIC SVD
  For capitals: project onto components best aligned with cap_dir
  For gender:   project onto components best aligned with gender_dir
  For antonyms: project onto components best aligned with antonym_dir
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day152_svd_projected_excl.json")
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

HELD_OUT = [
    ("The capital city of Russia is",      "Russia",   {"Russia","Russian"},  "Moscow",    "capitals"),
    ("The capital city of China is",       "China",    {"China","Chinese"},   "Beijing",   "capitals"),
    ("The capital city of Australia is",   "Australia",{"Australia"},         "Canberra",  "capitals"),
    ("The capital city of Greece is",      "Greece",   {"Greece","Greek"},    "Athens",    "capitals"),
    ("The capital city of Poland is",      "Poland",   {"Poland","Polish"},   "Warsaw",    "capitals"),
    ("The capital city of Sweden is",      "Sweden",   {"Sweden","Swedish"},  "Stockholm", "capitals"),
    ("The official language of Germany is","Germany",  {"Germany"},           "German",    "languages"),
    ("The official language of Japan is",  "Japan",    {"Japan"},             "Japanese",  "languages"),
    ("The official language of Korea is",  "Korea",    {"Korea"},             "Korean",    "languages"),
    ("A hammer is a type of",   "hammer", {"hammer"}, "tool",       "hypernyms"),
    ("A ruby is a type of",     "ruby",   {"ruby"},   "gem",        "hypernyms"),
    ("A whale is a type of",    "whale",  {"whale"},  "animal",     "hypernyms"),
    ("A violin is a type of",   "violin", {"violin"}, "instrument", "hypernyms"),
    ("The opposite of good is",   "good",   {"good"},   "bad",    "antonyms"),
    ("The opposite of fast is",   "fast",   {"fast"},   "slow",   "antonyms"),
    ("The opposite of clean is",  "clean",  {"clean"},  "dirty",  "antonyms"),
    ("The opposite of rich is",   "rich",   {"rich"},   "poor",   "antonyms"),
    ("The opposite of loud is",   "loud",   {"loud"},   "quiet",  "antonyms"),
    ("The opposite of strong is", "strong", {"strong"}, "weak",   "antonyms"),
    ("The prince and", "prince", {"prince"},  "princess","gender"),
    ("The duke and",   "duke",   {"duke"},    "duchess", "gender"),
    ("The actor and",  "actor",  {"actor"},   "actress", "gender"),
    ("The son and",    "son",    {"son"},     "daughter","gender"),
    ("The opposite of happy is",  "happy",  {"happy"},  "sad",    "antonyms_extra"),
    ("The opposite of early is",  "early",  {"early"},  "late",   "antonyms_extra"),
    ("The opposite of old is",    "old",    {"old"},    "young",  "antonyms_extra"),
    ("The opposite of easy is",   "easy",   {"easy"},   "hard",   "antonyms_extra"),
    ("Last month she",  "she",  {"she"},  "went", "tense"),
    ("Last week they",  "they", {"they"}, "went", "tense"),
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

# Build vocab
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = [w for w in VOCAB if get_tid(w)]
N = len(vocab_ok)
vocab_embs = {w: W_E[get_tid(w)] for w in vocab_ok}

# Build full vocab matrix and compute SVD (on all vocab words)
M = np.array([vocab_embs[w] for w in vocab_ok], dtype=np.float32)
M_centered = M - M.mean(axis=0)
print("Computing SVD ...")
U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
mean_vec = M.mean(axis=0)
print(f"  S[:6] = {S[:6].round(2)}\n")

# Build universal directions (for category-specific projection)
def make_dir(pairs):
    ds = [normed(W_E[get_tid(b)] - W_E[get_tid(a)])
          for a, b in pairs if get_tid(a) and get_tid(b)]
    return normed(np.mean(ds, axis=0))

cap_dir     = make_dir(list(COUNTRY_CAPITAL.items()))
gender_dir  = make_dir(GENDER_PAIRS)
antonym_dir = make_dir(ANTONYM_PAIRS)
lang_dir    = make_dir(LANGUAGE_PAIRS)

# Find best-aligned SVD components for each direction
def best_components(direction, top_n=20, threshold=0.05):
    aligns = [(k, cosine(Vt[k], direction)) for k in range(min(200, len(S)))]
    aligns.sort(key=lambda x: -abs(x[1]))
    return [k for k, c in aligns[:top_n] if abs(c) >= threshold]

cap_comps     = best_components(cap_dir,     top_n=30)
gender_comps  = best_components(gender_dir,  top_n=30)
antonym_comps = best_components(antonym_dir, top_n=30)
lang_comps    = best_components(lang_dir,    top_n=30)
print(f"Category-specific components:")
print(f"  cap    ({len(cap_comps)}): {cap_comps[:8]}")
print(f"  gender ({len(gender_comps)}): {gender_comps[:8]}")
print(f"  antonym({len(antonym_comps)}): {antonym_comps[:8]}")
print(f"  lang   ({len(lang_comps)}): {lang_comps[:8]}")
print()

# Build oracle from logprob (LM head)
import torch as _torch
print("Loading model again for oracle logprobs ...")
from transformers import AutoModelForCausalLM as _AMLM
_model = _AMLM.from_pretrained(MODEL_ID, dtype=_torch.float32)
_model.eval()
vocab_tids = {w: get_tid(w) for w in vocab_ok}

def get_oracle(prompt, exclude):
    inp = tok(prompt, return_tensors="pt")
    with _torch.no_grad():
        lp = _torch.log_softmax(_model(**inp).logits[0,-1,:], dim=-1).numpy()
    excl = [w for w in vocab_ok if w not in exclude]
    return sorted(excl, key=lambda w: -lp[vocab_tids[w]])[0]

print("Computing oracle answers ...")
oracle_answers = {}
for prompt, entity, exclude, expected, cat in HELD_OUT:
    oracle_answers[prompt] = get_oracle(prompt, exclude)
del _model
print()

# ─────────────────────────────────────────────────────────────
def project_onto_components(emb, components):
    """Project embedding onto selected SVD components."""
    centered = (emb - mean_vec).astype(np.float64)
    proj = np.zeros(len(components))
    for i, k in enumerate(components):
        proj[i] = float(np.dot(centered, Vt[k].astype(np.float64)))
    return proj

def svd_cosine(emb_a, emb_b, components):
    pa = project_onto_components(emb_a, components)
    pb = project_onto_components(emb_b, components)
    return cosine(pa, pb)

def entity_excl_svd(entity_word, exclude, components):
    eid = get_tid(entity_word)
    if eid is None: return vocab_ok[0]
    entity_e = W_E[eid]
    excl = [w for w in vocab_ok if w not in exclude]
    scores = {w: svd_cosine(entity_e, vocab_embs[w], components) for w in excl}
    return sorted(excl, key=lambda w: -scores[w])[0]

def entity_excl_topk(entity_word, exclude, K):
    """Standard entity_excl projected onto top-K SVD components."""
    comps = list(range(K))
    return entity_excl_svd(entity_word, exclude, comps)

# ─────────────────────────────────────────────────────────────
print("="*66)
print("PART 1: Top-K SVD Projection Sweep")
print("="*66)

K_VALUES = [2, 5, 10, 20, 50, 100, 200, N-1]  # N-1 = full rank of vocab matrix

results_by_k = {}
for K in K_VALUES:
    n_agree = 0
    for prompt, entity, exclude, expected, cat in HELD_OUT:
        oracle = oracle_answers[prompt]
        pred = entity_excl_topk(entity, exclude, min(K, len(S)-1))
        if pred == oracle: n_agree += 1
    results_by_k[K] = n_agree
    print(f"  K={K:>5}: {n_agree}/29 = {n_agree/29:.3f}")

best_K = max(K_VALUES, key=lambda k: results_by_k[k])
print(f"\n  Best K: {best_K} → {results_by_k[best_K]}/29 = {results_by_k[best_K]/29:.3f}")
print(f"  Full (K={H}): baseline entity_excl = 23/29 = 0.793")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 2: Category-Specific Component Projection")
print("="*66)
print()

CAT_COMPS = {
    "capitals":      cap_comps[:20],
    "languages":     lang_comps[:20],
    "antonyms":      antonym_comps[:20],
    "antonyms_extra":antonym_comps[:20],
    "gender":        gender_comps[:20],
    "hypernyms":     list(range(20)),
    "tense":         list(range(20)),
}

n_cat_agree = 0
cat_results = []
for prompt, entity, exclude, expected, cat in HELD_OUT:
    oracle = oracle_answers[prompt]
    comps = CAT_COMPS.get(cat, list(range(20)))
    pred_cat = entity_excl_svd(entity, exclude, comps)
    pred_full = entity_excl_topk(entity, exclude, len(S)-1)  # full rank
    agree_cat = (pred_cat == oracle)
    agree_full = (pred_full == oracle)
    if agree_cat: n_cat_agree += 1
    cat_results.append({"cat": cat, "entity": entity, "oracle": oracle,
                         "pred_cat": pred_cat, "pred_full": pred_full,
                         "agree_cat": agree_cat, "agree_full": agree_full})
    e = "✓" if agree_cat else "✗"
    print(f"  [{cat:>15}] {entity:>12}: {e} {pred_cat:<12}  oracle={oracle}")

print(f"\n  Category-specific projection: {n_cat_agree}/29 = {n_cat_agree/29:.3f}")
print(f"  Day 148 routed (full W_E):     24/29 = 0.828")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("PART 3: Category-Specific Accuracy at Best K")
print("="*66)
print()

for cat in sorted(set(r["cat"] for r in cat_results)):
    cr = [r for r in cat_results if r["cat"]==cat]
    ac = sum(r["agree_cat"] for r in cr)/len(cr)
    af = sum(r["agree_full"] for r in cr)/len(cr)
    # also at best_K
    n_bk = sum(1 for r in cr if entity_excl_topk(r["entity"],
        next(excl for p,e,excl,_,c in HELD_OUT if e==r["entity"] and c==cat),
        best_K) == r["oracle"])
    print(f"  {cat:>16}: cat_proj={ac:.2f}  full_rank={af:.2f}  best_K({best_K})={n_bk/len(cr):.2f}")

# ─────────────────────────────────────────────────────────────
print()
print("="*66)
print("Summary")
print("="*66)
best_any = max(results_by_k.values())
print(f"""
  K-sweep best:              {best_K} comps → {best_any}/29 = {best_any/29:.3f}
  Category-specific proj:    {n_cat_agree}/29 = {n_cat_agree/29:.3f}
  Day 148 full-W_E routed:   24/29 = 0.828  (baseline)
  Day 141 entity_excl:       23/29 = 0.793

  SVD projection {"IMPROVES" if best_any > 24 else "TIES" if best_any==24 else "DEGRADES"} over full-W_E entity_excl (K-sweep)
  Cat-specific   {"IMPROVES" if n_cat_agree > 24 else "TIES" if n_cat_agree==24 else "DEGRADES"} over Day 148 routed baseline
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"k_sweep": results_by_k, "cat_results": cat_results,
               "best_K": best_K, "n_cat": n_cat_agree}, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 152 complete.")
