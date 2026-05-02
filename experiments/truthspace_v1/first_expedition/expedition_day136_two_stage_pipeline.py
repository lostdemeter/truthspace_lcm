#!/usr/bin/env python3
"""
Day 136 — Two-Stage Pipeline: T2 Filter → L25 Rank

Day 135 confirmed complementary roles:
  T2  (12D): category filter — 2.34x random for free-form vocab selection
  L25 (1536D): within-category ranker — 62% oracle for constrained candidates

HYPOTHESIS: Combining them two-stage should beat either alone:
  Stage 1: T2 selects top-K words from vocab (K = 10, 20, 50)
  Stage 2: L25 ranks within those K words
  Final: top-1 from the L25-ranked T2-filtered set

Also test:
  - T2-filter → struct_axis rank (for antonym prompts)
  - T2-filter → T2-rank (single-stage T2, top-1 from T2)
  - Oracle: LM log-prob top-1 within vocab

Compare to:
  - T2 alone (top-1 from T2 ranking of full vocab)
  - L25 alone (top-1 from L25 ranking of full vocab)
  - LM oracle within vocab
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day136_two_stage_pipeline.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
L_BEST      = 25
K_VALUES    = [10, 20, 50, 100]

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

TEST_PROMPTS = [
    {"prompt": "Yesterday he",              "cat": "tense"},
    {"prompt": "Yesterday she",             "cat": "tense"},
    {"prompt": "She opened the door and",   "cat": "tense"},
    {"prompt": "He read the book and then", "cat": "tense"},
    {"prompt": "Last year he",              "cat": "tense"},
    {"prompt": "The children played",       "cat": "tense"},
    {"prompt": "The king and",              "cat": "gender"},
    {"prompt": "The queen and",             "cat": "gender"},
    {"prompt": "The father and his",        "cat": "gender"},
    {"prompt": "The opposite of hot is",    "cat": "antonyms"},
    {"prompt": "The opposite of large is",  "cat": "antonyms"},
    {"prompt": "The opposite of dark is",   "cat": "antonyms"},
    {"prompt": "The opposite of young is",  "cat": "antonyms"},
    {"prompt": "The capital city of France is",   "cat": "capitals"},
    {"prompt": "The capital city of Japan is",    "cat": "capitals"},
    {"prompt": "The capital city of Germany is",  "cat": "capitals"},
    {"prompt": "A poodle is a type of",     "cat": "hypernyms"},
    {"prompt": "A rose is a type of",       "cat": "hypernyms"},
    {"prompt": "An eagle is a type of",     "cat": "hypernyms"},
    {"prompt": "The sun rises in the",      "cat": "free"},
    {"prompt": "The cat sat on the",        "cat": "free"},
]

ANTONYM_TRAIN = [
    ("cold", ["warm","lukewarm"]),
    ("sad",  ["angry","bored"]),
    ("dark", ["bright","sunny"]),
    ("slow", ["quick","rapid"]),
    ("small",["big","huge"]),
    ("wrong",["correct","accurate"]),
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
print(f"  hidden={H}\n")

def get_hs(text, layers=None):
    if layers is None: layers = ALL_LAYERS
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

def get_lp_vocab(prompt, vocab_tok_ids):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1).numpy()
    return {w: float(lp[vocab_tok_ids[w]]) for w in vocab_tok_ids}

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
print("Building struct_axis ...")
ant_deltas = []
for correct, wrong_pool in ANTONYM_TRAIN:
    h_c = get_hs(" "+correct, [L_BEST])[L_BEST]
    h_w = np.mean([get_hs(" "+w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0).astype(np.float32)
    d = h_c - h_w; nv = np.linalg.norm(d)
    if nv > 1e-6: ant_deltas.append(d/nv)
struct_axis = normed(np.mean(ant_deltas, axis=0).astype(np.float32))
print("  Done.\n")

# Vocab
print("Building vocab cache ...")
seen = set(); VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_hs = {}; vocab_tok_id = {}
for w in VOCAB:
    ids = tok(" "+w, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        vocab_hs[w] = get_hs(" "+w, ALL_LAYERS)
        vocab_tok_id[w] = ids[0]
N_VOCAB = len(vocab_hs)
print(f"  {N_VOCAB} words cached.\n")

vocab_t2 = {w: t2_vec(vocab_hs[w]) for w in vocab_hs}

def score_words_t2(ctx_t2, words):
    return {w: cosine(vocab_t2[w], ctx_t2) for w in words}

def score_words_l25(ctx_h25, words):
    return {w: cosine(ctx_h25, vocab_hs[w][L_BEST]) for w in words}

def score_words_struct(words):
    return {w: float(np.dot(normed(vocab_hs[w][L_BEST]), struct_axis)) for w in words}

print("="*72)
print("Day 136: Two-Stage Pipeline T2→L25")
print("="*72)
print()

all_results = []
for case in TEST_PROMPTS:
    prompt = case["prompt"]; cat = case["cat"]
    ctx_hs  = get_hs(prompt, ALL_LAYERS)
    ctx_t2  = t2_vec(ctx_hs)
    ctx_h25 = ctx_hs[L_BEST]
    lp_scores = get_lp_vocab(prompt, vocab_tok_id)
    all_words = list(vocab_hs.keys())

    # Oracle within vocab
    oracle_ranked = sorted(all_words, key=lambda w: -lp_scores[w])
    oracle_top1   = oracle_ranked[0]

    # Stage 1: T2 full ranking
    t2_scores  = score_words_t2(ctx_t2, all_words)
    t2_ranked  = sorted(all_words, key=lambda w: -t2_scores[w])
    t2_top1    = t2_ranked[0]

    # Stage 1: L25 full ranking
    l25_scores = score_words_l25(ctx_h25, all_words)
    l25_ranked = sorted(all_words, key=lambda w: -l25_scores[w])
    l25_top1   = l25_ranked[0]

    # Stage 1: struct_axis full
    sa_scores  = score_words_struct(all_words)
    sa_ranked  = sorted(all_words, key=lambda w: -sa_scores[w])
    sa_top1    = sa_ranked[0]

    # Two-stage: T2 filter K → L25 rank
    two_stage = {}
    for K in K_VALUES:
        t2_topK = t2_ranked[:K]
        l25_within = score_words_l25(ctx_h25, t2_topK)
        final_top1 = max(t2_topK, key=lambda w: l25_within[w])
        two_stage[K] = final_top1

    # Two-stage: T2 filter K → struct_axis rank
    two_stage_sa = {}
    for K in K_VALUES:
        t2_topK  = t2_ranked[:K]
        sa_within = score_words_struct(t2_topK)
        final_top1 = max(t2_topK, key=lambda w: sa_within[w])
        two_stage_sa[K] = final_top1

    # rank of oracle top-1 in each method
    t2_rank  = next((i+1 for i,w in enumerate(t2_ranked) if w==oracle_top1), N_VOCAB+1)
    l25_rank = next((i+1 for i,w in enumerate(l25_ranked) if w==oracle_top1), N_VOCAB+1)

    row = {
        "prompt": prompt, "cat": cat,
        "oracle_top1": oracle_top1,
        "t2_top1": t2_top1, "l25_top1": l25_top1, "sa_top1": sa_top1,
        "t2_rank": t2_rank, "l25_rank": l25_rank,
        "two_stage": two_stage, "two_stage_sa": two_stage_sa,
        "t2_agree": t2_top1 == oracle_top1,
        "l25_agree": l25_top1 == oracle_top1,
    }
    for K in K_VALUES:
        row[f"ts_{K}_agree"] = two_stage[K] == oracle_top1
        row[f"ts_sa_{K}_agree"] = two_stage_sa[K] == oracle_top1
    all_results.append(row)

    print(f"  [{cat:>10}] {prompt!r}")
    print(f"    oracle={oracle_top1}  t2={t2_top1}(r{t2_rank})  l25={l25_top1}(r{l25_rank})")
    ts_str = "  ".join(f"K{K}={two_stage[K]}{'✓' if two_stage[K]==oracle_top1 else ''}" for K in K_VALUES)
    print(f"    T2→L25: {ts_str}")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*72)
print("Summary — Day 136")
print("="*72)
n = len(all_results)
exp_random = 1/N_VOCAB

def agree_rate(key): return sum(1 for r in all_results if r[key]) / n

print(f"\n  {'Method':>20}  {'top-1 agree':>12}  {'ratio vs random'}")
print(f"  {'-'*52}")
print(f"  {'t2_alone':>20}  {agree_rate('t2_agree'):>5.3f} ({sum(r['t2_agree'] for r in all_results)}/{n})  "
      f"  {agree_rate('t2_agree')/exp_random:.1f}x")
print(f"  {'l25_alone':>20}  {agree_rate('l25_agree'):>5.3f} ({sum(r['l25_agree'] for r in all_results)}/{n})  "
      f"  {agree_rate('l25_agree')/exp_random:.1f}x")
for K in K_VALUES:
    ar_ts   = agree_rate(f"ts_{K}_agree")
    ar_tssa = agree_rate(f"ts_sa_{K}_agree")
    print(f"  {'T2→L25 K='+str(K):>20}  {ar_ts:>5.3f} ({sum(r[f'ts_{K}_agree'] for r in all_results)}/{n})"
          f"   {ar_ts/exp_random:.1f}x   |  T2→SA: {ar_tssa:.3f} ({ar_tssa/exp_random:.1f}x)")

print(f"\n  Random baseline: {exp_random:.4f}")

# Per-category top-1 agreement for best two-stage K
best_K = max(K_VALUES, key=lambda K: agree_rate(f"ts_{K}_agree"))
print(f"\n  Best two-stage K={best_K}  Per-category agree:")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat]
    ts_agree  = sum(r[f"ts_{best_K}_agree"] for r in cat_r) / len(cat_r)
    t2_agree  = sum(r["t2_agree"]  for r in cat_r) / len(cat_r)
    l25_agree = sum(r["l25_agree"] for r in cat_r) / len(cat_r)
    oracle_t1 = cat_r[0]["oracle_top1"] if cat_r else "?"
    print(f"    {cat:>12}: T2→L25={ts_agree:.3f}  T2={t2_agree:.3f}  L25={l25_agree:.3f}  "
          f"{'✓ improves' if ts_agree > max(t2_agree, l25_agree) else '~ same' if ts_agree == max(t2_agree, l25_agree) else '✗ worse'}")

# Mean rank analysis
print(f"\n  Mean rank of oracle top-1:")
print(f"    T2 alone: {float(np.mean([r['t2_rank'] for r in all_results])):.1f}")
print(f"    L25 alone: {float(np.mean([r['l25_rank'] for r in all_results])):.1f}")

print(f"\n  VERDICT:")
best_ts_rate = max(agree_rate(f"ts_{K}_agree") for K in K_VALUES)
if best_ts_rate > 0.3:
    print(f"  → Two-stage pipeline ACHIEVES > 30% top-1 agreement")
    print(f"  → T2 category filtering + L25 within-category ranking = viable generation")
elif best_ts_rate > 0.1:
    print(f"  → Two-stage pipeline achieves 10-30% top-1 agreement (moderate)")
else:
    print(f"  → Two-stage pipeline does NOT achieve reliable top-1 agreement")
    print(f"  → Free-form generation requires a different approach (e.g. log-prob)")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": [{k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                         for k, v in r.items()} for r in all_results],
        "summary": {
            "t2_agree": agree_rate("t2_agree"),
            "l25_agree": agree_rate("l25_agree"),
            **{f"ts_{K}_agree": agree_rate(f"ts_{K}_agree") for K in K_VALUES},
        },
        "vocab_size": N_VOCAB, "random_baseline": exp_random,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 136 complete.")
