#!/usr/bin/env python3
"""
Day 158 — Cross-Model Universality: Does GPT-2 Have the Same W_E Structure?

Qwen2-1.5B results:
  - PC3 = capital direction (cos 0.445)
  - PC0 = named-entity axis
  - gender_dir: 100% universal, cap_dir: 91%, antonym: 75%
  - entity_excl: 79.3% baseline, routed: 82.8%

QUESTION: Is this structure universal across model architectures?
  Test model: GPT-2 (openai-community/gpt2)
    hidden_size = 768  (vs 1536 for Qwen2)
    vocabulary  = 50,257 BPE tokens
    architecture = GPT-2 (older, different tokenizer)

METHOD:
  1. Find which curated words are single tokens in GPT-2
  2. Build W_E matrix for those words in GPT-2
  3. Compute SVD — do top components match Qwen2's?
  4. Build universal directions in GPT-2 space
  5. Test entity_excl on held-out set
  6. Compare SVD structure and accuracy to Qwen2

HYPOTHESIS A (Universal):
  The same semantic structure emerges in different models because it
  reflects the structure of language, not the specific model architecture.
  PC3 = capital direction should appear in GPT-2 too.

HYPOTHESIS B (Model-Specific):
  The structure is an artifact of Qwen2's specific training/tokenization.
  GPT-2 has different axes, possibly weaker structure.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR   = Path(__file__).parent
OUTPUT_FILE  = str(SCRIPT_DIR / "day158_cross_model_universality.json")

QWEN_ID = "Qwen/Qwen2-1.5B-Instruct"
GPT2_ID = "openai-community/gpt2"

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

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

# ─────────────────────────────────────────────────────────────
# Load BOTH models
print(f"Loading {QWEN_ID} ...")
q_tok   = AutoTokenizer.from_pretrained(QWEN_ID)
q_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.float32)
q_model.eval()
W_E_Q = q_model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
H_Q   = W_E_Q.shape[1]
del q_model
print(f"  Qwen2: H={H_Q}, vocab={W_E_Q.shape[0]}\n")

print(f"Loading {GPT2_ID} ...")
g_tok   = AutoTokenizer.from_pretrained(GPT2_ID)
g_model = AutoModelForCausalLM.from_pretrained(GPT2_ID, dtype=torch.float32)
g_model.eval()
W_E_G = g_model.transformer.wte.weight.detach().numpy().astype(np.float32)
H_G   = W_E_G.shape[1]
del g_model
print(f"  GPT-2: H={H_G}, vocab={W_E_G.shape[0]}\n")

def q_tid(word):
    ids = q_tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def g_tid(word):
    ids = g_tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build vocabulary — words that are single tokens in BOTH models
seen = set()
ALL_VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_both = [w for w in ALL_VOCAB if q_tid(w) and g_tid(w)]
vocab_q    = [w for w in ALL_VOCAB if q_tid(w)]
vocab_g    = [w for w in ALL_VOCAB if g_tid(w)]
N_both = len(vocab_both)
N_q    = len(vocab_q)
N_g    = len(vocab_g)
print(f"Vocabulary coverage:")
print(f"  Qwen2 single-token words: {N_q}/{len(ALL_VOCAB)}")
print(f"  GPT-2 single-token words: {N_g}/{len(ALL_VOCAB)}")
print(f"  Both single-token:        {N_both}/{len(ALL_VOCAB)}\n")

# ─────────────────────────────────────────────────────────────
print("="*68)
print("PART 1: SVD Structure Comparison")
print("="*68)
print()

def build_svd(words, W_E, tid_fn):
    M = np.array([W_E[tid_fn(w)] for w in words], dtype=np.float32)
    M_c = M - M.mean(axis=0)
    _, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    return M, M_c, S, Vt, M.mean(axis=0)

M_q, M_qc, S_q, Vt_q, mean_q = build_svd(vocab_both, W_E_Q, q_tid)
M_g, M_gc, S_g, Vt_g, mean_g = build_svd(vocab_both, W_E_G, g_tid)

print(f"  Qwen2 S[:6]: {S_q[:6].round(2)}")
print(f"  GPT-2 S[:6]: {S_g[:6].round(2)}\n")

# Top-5 components for each model — what words load highest?
print(f"  Top-5 SVD components (both models, vocab={N_both} shared words):\n")
for k in range(5):
    q_scores = [(vocab_both[i], float(M_qc[i] @ Vt_q[k])) for i in range(N_both)]
    g_scores = [(vocab_both[i], float(M_gc[i] @ Vt_g[k])) for i in range(N_both)]
    q_scores.sort(key=lambda x: -x[1])
    g_scores.sort(key=lambda x: -x[1])
    q_top = [w for w,_ in q_scores[:4]]
    q_bot = [w for w,_ in q_scores[-3:]]
    g_top = [w for w,_ in g_scores[:4]]
    g_bot = [w for w,_ in g_scores[-3:]]
    print(f"  PC{k}  Qwen2: +{q_top}  -{q_bot}")
    print(f"  PC{k}  GPT-2: +{g_top}  -{g_bot}")
    print()

# ─────────────────────────────────────────────────────────────
print("="*68)
print("PART 2: Universal Directions in Both Models")
print("="*68)
print()

def make_dir_model(pairs, W_E, tid_fn):
    ds = [normed(W_E[tid_fn(b)] - W_E[tid_fn(a)])
          for a, b in pairs if tid_fn(a) and tid_fn(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

q_cap_dir    = make_dir_model(list(COUNTRY_CAPITAL.items()), W_E_Q, q_tid)
q_gender_dir = make_dir_model(GENDER_PAIRS, W_E_Q, q_tid)
q_antonym_dir= make_dir_model(ANTONYM_PAIRS, W_E_Q, q_tid)
q_lang_dir   = make_dir_model(LANGUAGE_PAIRS, W_E_Q, q_tid)

g_cap_dir    = make_dir_model(list(COUNTRY_CAPITAL.items()), W_E_G, g_tid)
g_gender_dir = make_dir_model(GENDER_PAIRS, W_E_G, g_tid)
g_antonym_dir= make_dir_model(ANTONYM_PAIRS, W_E_G, g_tid)
g_lang_dir   = make_dir_model(LANGUAGE_PAIRS, W_E_G, g_tid)

# Alignment of each model's direction with its own SVD
print("  Qwen2 universal direction → SVD alignment (top-20):")
q_dirs = {"cap":q_cap_dir,"gender":q_gender_dir,"antonym":q_antonym_dir,"lang":q_lang_dir}
q_dir_align = {}
for dname, dvec in q_dirs.items():
    if dvec is None: continue
    aligns = [(k, cosine(Vt_q[k], dvec)) for k in range(20)]
    best_k, best_c = max(aligns, key=lambda x: abs(x[1]))
    q_dir_align[dname] = (best_k, best_c)
    print(f"    {dname:>12}: best PC{best_k}, cos={best_c:.3f}")

print()
print("  GPT-2 universal direction → SVD alignment (top-20):")
g_dirs = {"cap":g_cap_dir,"gender":g_gender_dir,"antonym":g_antonym_dir,"lang":g_lang_dir}
g_dir_align = {}
for dname, dvec in g_dirs.items():
    if dvec is None: continue
    aligns = [(k, cosine(Vt_g[k], dvec)) for k in range(20)]
    best_k, best_c = max(aligns, key=lambda x: abs(x[1]))
    g_dir_align[dname] = (best_k, best_c)
    print(f"    {dname:>12}: best PC{best_k}, cos={best_c:.3f}")

# Are the directions themselves aligned between models?
# (after projecting to shared word space via per-word cosine)
print()
print("  Cross-model direction similarity (Qwen2 vs GPT-2):")
print("  Measured by: for each direction, does word A rank above word B in both models?")
for dname in ["cap","gender","antonym","lang"]:
    q_dvec = q_dirs.get(dname)
    g_dvec = g_dirs.get(dname)
    if q_dvec is None or g_dvec is None: continue

    # Agreement on which direction increases a score
    # Test: for each pair (a, b) in direction's training set,
    # does adding the direction improve ranking of b from a in both models?
    pairs_map = {"cap":list(COUNTRY_CAPITAL.items()),"gender":GENDER_PAIRS,
                 "antonym":ANTONYM_PAIRS,"lang":LANGUAGE_PAIRS}
    test_pairs = [(a,b) for a,b in pairs_map[dname]
                  if q_tid(a) and q_tid(b) and g_tid(a) and g_tid(b)]

    q_correct = 0; g_correct = 0
    for a, b in test_pairs:
        qa = W_E_Q[q_tid(a)]; qb = W_E_Q[q_tid(b)]
        ga = W_E_G[g_tid(a)]; gb = W_E_G[g_tid(b)]
        # rank b from a+dir vs without dir
        q_with = cosine(qa + q_dvec, qb)
        q_base = cosine(qa, qb)
        g_with = cosine(ga + g_dvec, gb)
        g_base = cosine(ga, gb)
        if q_with > q_base: q_correct += 1
        if g_with > g_base: g_correct += 1

    n = len(test_pairs)
    print(f"    {dname:>12}: Qwen2={q_correct}/{n}={q_correct/n:.2f}  GPT-2={g_correct}/{n}={g_correct/n:.2f}")

# ─────────────────────────────────────────────────────────────
print()
print("="*68)
print("PART 3: entity_excl Accuracy on Held-Out Set")
print("="*68)
print()

CAT_DIR_Q = {"gender": q_gender_dir, "capitals": q_cap_dir}
CAT_DIR_G = {"gender": g_gender_dir, "capitals": g_cap_dir}

def entity_excl_model(entity_word, direction, exclude, W_E, tid_fn, vocab):
    eid = tid_fn(entity_word)
    if eid is None: return None, 0.0
    e = W_E[eid].copy()
    if direction is not None:
        e = e + direction
    excl = [w for w in vocab if w not in exclude and tid_fn(w)]
    if not excl: return None, 0.0
    scores = {w: cosine(e, W_E[tid_fn(w)]) for w in excl}
    top1 = max(excl, key=lambda w: scores[w])
    return top1, scores[top1]

# Use only words available in each model's vocabulary
print(f"  {'Prompt entity':>12}  {'cat':>14}  {'Q-pred':>12}  {'G-pred':>12}  oracle")
q_agree = 0; g_agree = 0
ho_results = []
for prompt, entity, exclude, expected, cat in HELD_OUT:
    q_dir = CAT_DIR_Q.get(cat)
    g_dir = CAT_DIR_G.get(cat)
    q_pred, q_score = entity_excl_model(entity, q_dir, exclude, W_E_Q, q_tid, vocab_q)
    g_pred, g_score = entity_excl_model(entity, g_dir, exclude, W_E_G, g_tid, vocab_g)

    # Oracle from Qwen2 logprobs
    q_model2 = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.float32)
    q_model2.eval()
    inp = q_tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(q_model2(**inp).logits[0,-1,:], dim=-1).numpy()
    del q_model2
    excl_set = {w for w in exclude}
    oracle = max([w for w in vocab_q if w not in excl_set],
                  key=lambda w: float(lp[q_tid(w)]) if q_tid(w) else -1e9)

    qa = "✓" if q_pred == oracle else "✗"
    ga = "✓" if g_pred == oracle else "✗"
    if q_pred == oracle: q_agree += 1
    if g_pred == oracle: g_agree += 1
    ho_results.append({"entity": entity, "cat": cat, "oracle": oracle,
                        "q_pred": q_pred, "g_pred": g_pred,
                        "agree_q": q_pred==oracle, "agree_g": g_pred==oracle})
    print(f"  {entity:>12}  {cat:>14}  {qa}{q_pred or '?':<12}  {ga}{g_pred or '?':<12}  {oracle}")

print(f"\n  Qwen2 entity_excl: {q_agree}/29 = {q_agree/29:.3f}")
print(f"  GPT-2 entity_excl: {g_agree}/29 = {g_agree/29:.3f}")
print(f"  (Qwen2 routed Day 148 baseline: 24/29 = 82.8%)")

# ─────────────────────────────────────────────────────────────
print()
print("="*68)
print("PART 4: Cross-Model SVD Alignment")
print("="*68)
print()
# Map each model's PC to a shared 'word score' vector and compare
# Shared word scores for PC k: [U[i,k] * S[k] for i in shared vocab]
# Correlation of these scores between Qwen2 and GPT-2

print("  Pearson correlation of word PC scores (Qwen2 vs GPT-2 top-10 PCs):")
print(f"  {'Qwen2 PC':>10}  " + "  ".join(f"GPT_{j}" for j in range(5)))
for k in range(5):
    q_word_scores = np.array([float(M_qc[i] @ Vt_q[k]) for i in range(N_both)])
    row = []
    for j in range(5):
        g_word_scores = np.array([float(M_gc[i] @ Vt_g[j]) for i in range(N_both)])
        # Pearson correlation
        c = float(np.corrcoef(q_word_scores, g_word_scores)[0,1])
        row.append(f"{c:>+7.3f}")
    print(f"  Qwen_PC{k}:  " + "  ".join(row))

# ─────────────────────────────────────────────────────────────
print()
print("="*68)
print("Summary: Cross-Model Universality")
print("="*68)

print(f"""
  Model comparison:
    Qwen2-1.5B: H={H_Q}, vocab=151936, shared words={N_both}
    GPT-2:      H={H_G}, vocab=50257,  GPT-2 words={N_g}

  SVD direction alignment:
    Qwen2 cap_dir  → own SVD: PC{q_dir_align.get('cap',(0,0))[0]}, cos={q_dir_align.get('cap',(0,0))[1]:.3f}
    GPT-2 cap_dir  → own SVD: PC{g_dir_align.get('cap',(0,0))[0]}, cos={g_dir_align.get('cap',(0,0))[1]:.3f}

    Qwen2 gender   → own SVD: PC{q_dir_align.get('gender',(0,0))[0]}, cos={q_dir_align.get('gender',(0,0))[1]:.3f}
    GPT-2 gender   → own SVD: PC{g_dir_align.get('gender',(0,0))[0]}, cos={g_dir_align.get('gender',(0,0))[1]:.3f}

  entity_excl accuracy:
    Qwen2: {q_agree}/29 = {q_agree/29:.3f}
    GPT-2: {g_agree}/29 = {g_agree/29:.3f}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"q_dir_align": q_dir_align, "g_dir_align": g_dir_align,
               "ho_results": ho_results, "q_agree": q_agree, "g_agree": g_agree,
               "N_both": N_both, "N_q": N_q, "N_g": N_g}, f, indent=2, default=str)
print(f"Saved: {OUTPUT_FILE}")
print("Day 158 complete.")
