#!/usr/bin/env python3
"""
Day 134 — Free-Form Generation with Curated Vocabulary

Day 133 failed due to subword fragment contamination in the vocabulary.
L25 cosine attracted BPE fragments ('orem','ena','isen') that dominated rankings.

FIX: Use a hand-curated vocabulary of real English words:
  - Common verbs (especially past tense for tense prompts)
  - Common nouns (cities, languages, categories)
  - Function words / connectives
  - NO subword fragments (all words must be in standard English dictionary)

TEST:
  A) T2 method on syntactic prompts (tense/gender) — known to work in Day 132
  B) L25 method on factual prompts — check if subword removal helps
  C) Struct_axis on antonym prompts

METRIC: Top-1 agreement, rank of LM top-1, overlap@K
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day134_curated_vocab_gen.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
L_BEST      = 25

# ── Curated vocabulary ────────────────────────────────────────────────────────
# Real English words, organized by semantic type
VOCAB_CURATED = [
    # Past-tense verbs (critical for tense T2 method)
    "walked","ran","ate","built","wrote","read","said","went","came","took",
    "made","got","saw","gave","knew","thought","found","told","became","left",
    "brought","bought","taught","caught","fought","heard","held","kept","sent",
    "fell","felt","grew","slept","spent","stood","understood","wore","won",
    "drove","flew","swam","sang","sat","laid","paid","played","stayed","opened",
    "turned","looked","stopped","asked","started","tried","closed","moved","lived",
    # Present-tense verbs
    "walk","run","eat","build","write","read","say","go","come","take",
    "make","get","see","give","know","think","find","tell","become","leave",
    # Common adjectives / antonyms
    "cold","hot","big","small","fast","slow","dark","light","happy","sad",
    "good","bad","strong","weak","young","old","loud","quiet","easy","hard",
    "clean","dirty","rich","poor","safe","dangerous","early","late",
    # Nouns — cities / capitals
    "Paris","London","Rome","Berlin","Madrid","Tokyo","Moscow","Beijing",
    "Sydney","Ottawa","Canberra","Brasilia","Cairo","Delhi","Seoul","Bangkok",
    "Vienna","Warsaw","Athens","Lisbon","Brussels","Amsterdam","Oslo","Stockholm",
    # Nouns — languages
    "English","French","Spanish","German","Italian","Portuguese","Arabic",
    "Mandarin","Japanese","Korean","Hindi","Russian","Turkish","Persian",
    "Bengali","Punjabi","Tamil","Urdu","Polish","Dutch","Swedish","Greek",
    # Nouns — categories / hypernyms
    "animal","plant","tool","vehicle","food","music","sport","color","number",
    "language","country","city","flower","tree","bird","fish","dog","cat","horse",
    "instrument","weapon","machine","device","metal","mineral","crystal","gem",
    # Nouns — people / gender
    "king","queen","prince","princess","duke","duchess","emperor","empress",
    "father","mother","brother","sister","son","daughter","uncle","aunt",
    "man","woman","boy","girl","actor","actress","hero","heroine",
    # Common content words
    "east","west","north","south","morning","evening","night","water","fire",
    "door","house","book","table","chair","window","street","road","park",
    "school","office","market","store","church","castle","palace","bridge",
    # Connectives / function words (some useful for completion)
    "then","also","soon","just","very","still","again","always","never",
    "first","last","next","before","after","here","there",
]

# Deduplicate while preserving order
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]

TEST_PROMPTS = [
    # TENSE — T2 method
    {"prompt": "Yesterday he",             "cat": "tense",    "note": "past tense"},
    {"prompt": "Yesterday she",            "cat": "tense",    "note": "past tense"},
    {"prompt": "Yesterday they",           "cat": "tense",    "note": "past tense"},
    {"prompt": "She opened the door and",  "cat": "tense",    "note": "continuation"},
    {"prompt": "He read the book and then","cat": "tense",    "note": "continuation"},
    {"prompt": "Last year he",             "cat": "tense",    "note": "past tense"},
    # GENDER — T2 method
    {"prompt": "The king and",             "cat": "gender",   "note": "gender pair"},
    {"prompt": "The queen and",            "cat": "gender",   "note": "gender pair"},
    {"prompt": "The father and his",       "cat": "gender",   "note": "gender relation"},
    # ANTONYMS — struct_axis
    {"prompt": "The opposite of hot is",   "cat": "antonyms", "note": "antonym"},
    {"prompt": "The opposite of large is", "cat": "antonyms", "note": "antonym"},
    {"prompt": "The opposite of dark is",  "cat": "antonyms", "note": "antonym"},
    {"prompt": "The opposite of young is", "cat": "antonyms", "note": "antonym"},
    # CAPITALS — L25
    {"prompt": "The capital city of France is",  "cat": "capitals", "note": "factual"},
    {"prompt": "The capital city of Japan is",   "cat": "capitals", "note": "factual"},
    {"prompt": "The capital city of Germany is", "cat": "capitals", "note": "factual"},
    # HYPERNYMS — L25
    {"prompt": "A poodle is a type of",    "cat": "hypernyms","note": "hypernym"},
    {"prompt": "A rose is a type of",      "cat": "hypernyms","note": "hypernym"},
    {"prompt": "An eagle is a type of",    "cat": "hypernyms","note": "hypernym"},
    # FREE FORM — routing depends on T2 classification
    {"prompt": "The sun rises in the",     "cat": "free",     "note": "direction"},
    {"prompt": "The cat sat on the",       "cat": "free",     "note": "location"},
]

ROUTING_TABLE = {
    "tense": "t2", "gender": "t2",
    "antonyms": "struct_axis",
    "hypernyms": "l25", "capitals": "l25", "languages": "l25",
    "unknown": "l25",
}
TRAIN_CENTROIDS = {
    "antonyms": ["The opposite of hot is","The opposite of happy is","The opposite of bright is",
                 "The opposite of fast is","The opposite of large is","The opposite of right is"],
    "capitals": ["The capital city of France is","The capital city of Japan is","The capital city of Germany is",
                 "The capital city of Spain is","The capital city of Italy is","The capital city of Russia is"],
    "languages":["The official language of Brazil is","The official language of Egypt is",
                 "The official language of China is","The official language of India is"],
    "hypernyms": ["A poodle is a type of","A rose is a type of","A hammer is a type of",
                  "An eagle is a type of","A ruby is a type of"],
    "tense":     ["Yesterday he","Yesterday she","Yesterday they"],
    "gender":    ["The king and","The queen and","The father and"],
}
ANTONYM_TRAIN = [
    ("cold", ["warm","lukewarm"], "The opposite of hot is"),
    ("sad",  ["angry","bored"],  "The opposite of happy is"),
    ("dark", ["bright","sunny"], "The opposite of bright is"),
    ("slow", ["quick","rapid"],  "The opposite of fast is"),
    ("small",["big","huge"],     "The opposite of large is"),
    ("wrong",["correct","accurate"],"The opposite of right is"),
]
DAY78_LAYERS = {"gender":27,"comparative":15,"hypernym":28,"plural":1,"synonym":28,"concrete":28,
                "past_tense":28,"antonym":28,"passive":28,"causation":28,"question":28,"negation":28}
AXIS_NAMES_12 = ["gender","comparative","hypernym","plural","synonym","concrete",
                 "past_tense","antonym","passive","causation","question","negation"]
AXIS_PAIRS = {
    "gender":[("The king ruled","The queen ruled"),("A man walked","A woman walked"),("His brother arrived","His sister arrived")],
    "comparative":[("The fast car","The faster car"),("A big dog","A bigger dog"),("The tall tree","The taller tree")],
    "hypernym":[("The dog ran","The animal ran"),("A rose bloomed","A flower bloomed"),("The car sped","The vehicle sped")],
    "plural":[("A dog played","Dogs played"),("The cat sat","The cats sat")],
    "synonym":[("He is big","He is large"),("She is small","She is tiny"),("It is cold","It is frigid")],
    "concrete":[("The stone is heavy","The burden is heavy"),("The long road","The long journey")],
    "past_tense":[("I walk every morning","I walked every morning"),("She runs through park","She ran through park"),("He eats","He ate")],
    "antonym":[("It is hot","It is cold"),("He runs fast","He runs slow"),("She is happy","She is sad")],
    "passive":[("The cat chased mouse","The mouse was chased"),("John broke window","The window was broken")],
    "causation":[("The rain falls","The ground gets wet"),("The fire burns","The wood turns to ash")],
    "question":[("She is tired","Is she tired"),("He can swim","Can he swim")],
    "negation":[("The dog is fast","The dog is not fast"),("She can swim","She cannot swim")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
T2_LAYERS  = sorted(set(DAY78_LAYERS.values()))
ALL_LAYERS = sorted(set([L_BEST] + T2_LAYERS))
print(f"  hidden={H}  vocab_curated={len(VOCAB)}\n")

def get_hs(text, layers=None):
    if layers is None: layers = ALL_LAYERS
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

def get_logits_top(prompt, k=20):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
    top_ids = torch.argsort(lp, descending=True)[:k].tolist()
    top_ws  = [tok.decode([i]).strip() for i in top_ids]
    return top_ids, top_ws, lp.numpy()

# Build T2 axes
print("Building T2 axes ...")
t2_axes = {}
for ax in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax]; diffs = []
    for s1, s2 in AXIS_PAIRS.get(ax, []):
        inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
        with torch.no_grad():
            h1 = model(**inp1, output_hidden_states=True).hidden_states[L][0,-1,:].numpy().astype(np.float32)
            h2 = model(**inp2, output_hidden_states=True).hidden_states[L][0,-1,:].numpy().astype(np.float32)
        d = h2-h1; nv = np.linalg.norm(d)
        if nv > 1e-6: diffs.append(d/nv)
    v = np.mean(diffs, axis=0) if diffs else np.zeros(H, np.float32)
    nv = np.linalg.norm(v); t2_axes[ax] = (v/nv if nv>1e-6 else v).astype(np.float32)
print("  Done.\n")

def t2_vec(hs):
    v = np.zeros(12, np.float32)
    for k, ax in enumerate(AXIS_NAMES_12):
        h = normed(hs.get(DAY78_LAYERS[ax], np.zeros(H))); v[k] = float(np.dot(h, t2_axes[ax]))
    return v

# Build struct_axis
print("Building struct_axis (antonyms) ...")
ant_deltas = []
for correct, wrong_pool, _ in ANTONYM_TRAIN:
    h_c = get_hs(" "+correct, [L_BEST])[L_BEST]
    h_w = np.mean([get_hs(" "+w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0).astype(np.float32)
    d = h_c - h_w; nv = np.linalg.norm(d)
    if nv > 1e-6: ant_deltas.append(d/nv)
struct_axis = normed(np.mean(ant_deltas, axis=0).astype(np.float32))
print("  Done.\n")

# Build T2 centroids
print("Building category T2 centroids ...")
cat_centroids = {}
for cat, prompts in TRAIN_CENTROIDS.items():
    vecs = [t2_vec(get_hs(p, T2_LAYERS)) for p in prompts]
    m = np.mean(vecs, axis=0)
    cat_centroids[cat] = normed(m.astype(np.float32))
print("  Done.\n")

def classify_prompt(prompt):
    v = t2_vec(get_hs(prompt, T2_LAYERS))
    scores = {c: cosine(v, cat_centroids[c]) for c in cat_centroids}
    return max(scores, key=lambda c: scores[c])

# Pre-compute vocab HS — only keep words that tokenize as single tokens
print(f"Pre-computing vocab hidden states ...")
vocab_hs = {}
vocab_token_ids = {}
single_token_words = []
for w in VOCAB:
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:  # single token only
        single_token_words.append(w)
        vocab_token_ids[w] = ids[0]

print(f"  Single-token words: {len(single_token_words)} / {len(VOCAB)}")
for i, w in enumerate(single_token_words):
    if i % 50 == 0: print(f"  {i}/{len(single_token_words)} ...", end="\r", flush=True)
    vocab_hs[w] = get_hs(" " + w, ALL_LAYERS)
print(f"\n  Done. {len(vocab_hs)} words cached.\n")

def rank_vocab_words(method, ctx_hs):
    words = list(vocab_hs.keys())
    if method == "t2":
        ctx_t2 = t2_vec(ctx_hs)
        scores = {w: cosine(t2_vec(vocab_hs[w]), ctx_t2) for w in words}
    elif method == "struct_axis":
        scores = {w: float(np.dot(normed(vocab_hs[w][L_BEST]), struct_axis)) for w in words}
    else:  # l25
        h = ctx_hs[L_BEST]
        scores = {w: cosine(h, vocab_hs[w][L_BEST]) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

# ── Main test ─────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 134: Free-Form Generation with Curated Vocabulary")
print("=" * 72)
print()

all_results = []
N_VOCAB = len(vocab_hs)

for case in TEST_PROMPTS:
    prompt = case["prompt"]
    cat_true = case["cat"]
    note = case["note"]

    ctx_hs = get_hs(prompt, ALL_LAYERS)
    cat_pred = classify_prompt(prompt)
    method   = ROUTING_TABLE.get(cat_pred, "l25")

    geo_ranked, geo_scores = rank_vocab_words(method, ctx_hs)
    geo_top1 = geo_ranked[0]

    _, lm_top_words, lp_arr = get_logits_top(prompt, k=50)
    lm_top1 = lm_top_words[0]

    # LM probability scores for vocab words (only single-token ones)
    lm_scores = {w: float(lp_arr[vocab_token_ids[w]]) for w in vocab_hs}
    lm_ranked_vocab = sorted(vocab_hs.keys(), key=lambda w: -lm_scores[w])
    lm_top1_in_vocab = lm_ranked_vocab[0]

    # Rank of lm_top1_in_vocab in geo ranking
    geo_rank_of_lm_vocab_top1 = next((i+1 for i,w in enumerate(geo_ranked) if w==lm_top1_in_vocab), N_VOCAB+1)

    # Overlap at K
    for K in [5, 10, 20]:
        geo_K = set(geo_ranked[:K])
        lm_K  = set(lm_ranked_vocab[:K])
        overlap = len(geo_K & lm_K)
        expected = K * K / N_VOCAB  # expected random overlap

    # Top-1 agreement (within vocab)
    agree_vocab = (geo_top1 == lm_top1_in_vocab)

    overlap5  = len(set(geo_ranked[:5])  & set(lm_ranked_vocab[:5]))
    overlap10 = len(set(geo_ranked[:10]) & set(lm_ranked_vocab[:10]))
    overlap20 = len(set(geo_ranked[:20]) & set(lm_ranked_vocab[:20]))
    expected5  = 5*5/N_VOCAB
    expected10 = 10*10/N_VOCAB
    expected20 = 20*20/N_VOCAB

    all_results.append({
        "prompt": prompt, "cat_true": cat_true, "cat_pred": cat_pred,
        "method": method, "agree_vocab": agree_vocab,
        "geo_top1": geo_top1, "lm_top1_vocab": lm_top1_in_vocab, "lm_top1": lm_top1,
        "geo_rank_of_lm_vocab_top1": geo_rank_of_lm_vocab_top1,
        "overlap5": overlap5, "overlap10": overlap10, "overlap20": overlap20,
        "expected5": expected5, "expected10": expected10, "expected20": expected20,
        "lm_top5_vocab": lm_ranked_vocab[:5],
        "geo_top5": geo_ranked[:5],
    })

    print(f"  [{cat_pred:>10}|{method:>11}] {prompt!r}")
    print(f"    geo={geo_ranked[:5]}")
    print(f"    lm(vocab)={lm_ranked_vocab[:5]}  lm(full)={lm_top_words[:3]}")
    print(f"    agree={agree_vocab}  geo_rank(lm_top1)={geo_rank_of_lm_vocab_top1}  "
          f"overlap@5={overlap5}(exp{expected5:.1f})  @10={overlap10}(exp{expected10:.1f})")
    print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 72)
print("Summary")
print("=" * 72)
n = len(all_results)
n_agree  = sum(1 for r in all_results if r["agree_vocab"])
mean_rank= float(np.mean([r["geo_rank_of_lm_vocab_top1"] for r in all_results]))
mean_o5  = float(np.mean([r["overlap5"]  for r in all_results]))
mean_o10 = float(np.mean([r["overlap10"] for r in all_results]))
mean_o20 = float(np.mean([r["overlap20"] for r in all_results]))
exp5     = float(np.mean([r["expected5"]  for r in all_results]))
exp10    = float(np.mean([r["expected10"] for r in all_results]))
exp20    = float(np.mean([r["expected20"] for r in all_results]))

print(f"""
  Vocab size: {N_VOCAB} single-token real words
  Top-1 agreement (within vocab): {n_agree}/{n} = {n_agree/n:.3f}
  Mean geo-rank of LM vocab top-1: {mean_rank:.1f} (out of {N_VOCAB})
  
  Overlap@K vs random expectation:
    @5:  {mean_o5:.2f} (expected {exp5:.2f})   ratio={mean_o5/max(exp5,0.001):.2f}x
    @10: {mean_o10:.2f} (expected {exp10:.2f})  ratio={mean_o10/max(exp10,0.001):.2f}x
    @20: {mean_o20:.2f} (expected {exp20:.2f})  ratio={mean_o20/max(exp20,0.001):.2f}x
""")

print("  Per-category:")
for cat in sorted(set(r["cat_true"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat_true"] == cat]
    na = sum(1 for r in cat_r if r["agree_vocab"])
    mr = float(np.mean([r["geo_rank_of_lm_vocab_top1"] for r in cat_r]))
    mo10 = float(np.mean([r["overlap10"] for r in cat_r]))
    meth = ROUTING_TABLE.get(cat, "l25")
    print(f"    {cat:>12} [{meth:>11}]: agree={na}/{len(cat_r)}  rank={mr:.1f}  overlap@10={mo10:.2f}")

print()
signal_ratio = mean_o10 / max(exp10, 0.001)
print(f"  Signal ratio @10: {signal_ratio:.2f}x random  "
      f"({'STRONG signal' if signal_ratio>3 else 'moderate signal' if signal_ratio>1.5 else 'weak/no signal'})")
top1_rate = n_agree/n
print(f"  Top-1 agreement: {top1_rate:.3f}  "
      f"({'STRONG' if top1_rate>0.3 else 'MODERATE' if top1_rate>0.1 else 'WEAK'})")

print(f"\n  VERDICT:")
if top1_rate > 0.3 or signal_ratio > 3:
    print(f"  → Geometric pipeline has STRONG signal for free-form generation")
    print(f"  → T2/geometric structure approximates LM next-token distribution")
elif top1_rate > 0.1 or signal_ratio > 1.5:
    print(f"  → Geometric pipeline has MODERATE signal")
    print(f"  → Some categories (likely tense/gender) drive the agreement")
else:
    print(f"  → Geometric pipeline has WEAK signal for free-form generation")
    print(f"  → Candidate-set ranking (Days 124-132) works; unconstrained doesn't")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "n_agree": n_agree, "n_total": n,
        "top1_agreement": n_agree/n,
        "mean_rank": mean_rank,
        "mean_overlap5": mean_o5, "mean_overlap10": mean_o10, "mean_overlap20": mean_o20,
        "expected5": exp5, "expected10": exp10, "expected20": exp20,
        "vocab_size": N_VOCAB,
        "signal_ratio_10": signal_ratio,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 134 complete.")
