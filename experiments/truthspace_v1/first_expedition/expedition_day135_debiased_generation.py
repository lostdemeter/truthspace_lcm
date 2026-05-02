#!/usr/bin/env python3
"""
Day 135 — Frequency-Debiased L25 Similarity for Free-Form Generation

Day 134 problem: 'Portuguese','Canberra','Oslo' dominate geo-top5 for all
prompts because proper nouns cluster near ALL "is"-final contexts at L25.

FIX: Center the representations to remove shared "frequency" direction.
  μ_vocab = mean(h(w) for w in vocab)  [mean word representation]
  h_centered(w) = h(w) - μ_vocab
  h_centered(ctx) = h(ctx) - μ_vocab
  score = cosine(h_centered(ctx), h_centered(w))

This is equivalent to removing the first principal component (the "mean")
from the L25 similarity computation. After centering, the similarity becomes
sensitive to DIFFERENCES from the average word, not absolute positions.

Also test:
  A) PMI-style debiasing: score = log P_geo(w|ctx) - log P_geo(w)
     = log sim(ctx, w) - log sim(mean_ctx, w)
  B) Multiple debiasing strategies (subtract vocab mean vs subtract context mean)
  C) Whether T2 method also benefits from debiasing
  D) Combined: debiased L25 + T2 routing
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day135_debiased_generation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
L_BEST      = 25

# Same curated vocabulary as Day 134
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

def get_logits_top(prompt, k=50):
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
print("Building struct_axis ...")
ant_deltas = []
for correct, wrong_pool in ANTONYM_TRAIN:
    h_c = get_hs(" "+correct, [L_BEST])[L_BEST]
    h_w = np.mean([get_hs(" "+w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0).astype(np.float32)
    d = h_c - h_w; nv = np.linalg.norm(d)
    if nv > 1e-6: ant_deltas.append(d/nv)
struct_axis = normed(np.mean(ant_deltas, axis=0).astype(np.float32))
print("  Done.\n")

# Build vocab — single-token only
print("Building vocab cache ...")
seen = set(); VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_hs = {}; vocab_tok_id = {}
for w in VOCAB:
    ids = tok(" "+w, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        vocab_hs[w] = get_hs(" "+w, ALL_LAYERS)
        vocab_tok_id[w] = ids[0]
N_VOCAB = len(vocab_hs)
print(f"  {N_VOCAB} single-token words cached.\n")

# Compute vocab mean at L25 (the "frequency bias" direction)
vocab_vecs_L25 = np.stack([vocab_hs[w][L_BEST] for w in vocab_hs])  # (N, H)
mu_vocab = vocab_vecs_L25.mean(axis=0)  # (H,) — mean word representation

# Also compute T2 vocab mean
vocab_t2_vecs = np.stack([t2_vec(vocab_hs[w]) for w in vocab_hs])   # (N, 12)
mu_t2 = vocab_t2_vecs.mean(axis=0)  # (12,)

print(f"  Vocab mean L25 ||μ|| = {np.linalg.norm(mu_vocab):.4f}")
print(f"  Vocab mean T2  ||μ|| = {np.linalg.norm(mu_t2):.4f}\n")

def rank_words_centered(ctx_hs):
    """L25 cosine after subtracting vocab mean (frequency debiasing)."""
    ctx_c = ctx_hs[L_BEST] - mu_vocab
    words = list(vocab_hs.keys())
    scores = {}
    for w in words:
        w_c = vocab_hs[w][L_BEST] - mu_vocab
        scores[w] = cosine(ctx_c, w_c)
    return sorted(words, key=lambda w: -scores[w]), scores

def rank_words_raw_l25(ctx_hs):
    words = list(vocab_hs.keys())
    ctx_h = ctx_hs[L_BEST]
    scores = {w: cosine(ctx_h, vocab_hs[w][L_BEST]) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

def rank_words_t2(ctx_hs):
    ctx_t2 = t2_vec(ctx_hs)
    words = list(vocab_hs.keys())
    scores = {w: cosine(t2_vec(vocab_hs[w]), ctx_t2) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

def rank_words_t2_centered(ctx_hs):
    ctx_t2 = t2_vec(ctx_hs) - mu_t2
    words = list(vocab_hs.keys())
    scores = {w: cosine(t2_vec(vocab_hs[w]) - mu_t2, ctx_t2) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

def rank_words_struct(ctx_hs):
    words = list(vocab_hs.keys())
    scores = {w: float(np.dot(normed(vocab_hs[w][L_BEST]), struct_axis)) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

# ── Main test ─────────────────────────────────────────────────────────────────
print("="*72)
print("Day 135: Frequency-Debiased L25 Similarity")
print("="*72)
print()

all_results = []
for case in TEST_PROMPTS:
    prompt = case["prompt"]; cat = case["cat"]
    ctx_hs = get_hs(prompt, ALL_LAYERS)
    _, lm_ws, lp_arr = get_logits_top(prompt, k=50)
    lm_vocab_ranked = sorted(vocab_hs.keys(), key=lambda w: -lp_arr[vocab_tok_id[w]])
    lm_top1v = lm_vocab_ranked[0]

    methods = {
        "l25_raw":     rank_words_raw_l25(ctx_hs),
        "l25_centered":rank_words_centered(ctx_hs),
        "t2_raw":      rank_words_t2(ctx_hs),
        "t2_centered": rank_words_t2_centered(ctx_hs),
        "struct_axis": rank_words_struct(ctx_hs),
    }

    row = {"prompt": prompt, "cat": cat, "lm_top1_vocab": lm_top1v}
    for mname, (ranked, scores) in methods.items():
        rank_of_lm = next((i+1 for i,w in enumerate(ranked) if w==lm_top1v), N_VOCAB+1)
        ov10 = len(set(ranked[:10]) & set(lm_vocab_ranked[:10]))
        row[f"{mname}_top1"] = ranked[0]
        row[f"{mname}_rank"] = rank_of_lm
        row[f"{mname}_ov10"] = ov10

    all_results.append(row)

    print(f"  [{cat:>10}] {prompt!r}")
    print(f"    lm_top1_vocab={lm_top1v}  lm_top5={lm_vocab_ranked[:5]}")
    for mname in ["l25_raw","l25_centered","t2_raw","t2_centered","struct_axis"]:
        r = row[f"{mname}_rank"]; ov = row[f"{mname}_ov10"]
        t1 = row[f"{mname}_top1"]
        print(f"    {mname:>14}: top1={t1:<12}  rank={r:>3}  ov@10={ov}")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*72)
print("Summary — All Methods, All Prompts")
print("="*72)
n = len(all_results)

methods = ["l25_raw","l25_centered","t2_raw","t2_centered","struct_axis"]
exp10 = 10*10/N_VOCAB

print(f"\n  {'Method':>16}  {'top1_agree':>12}  {'mean_rank':>10}  {'overlap@10':>12}  {'vs_random':>10}")
print(f"  {'-'*70}")
for mname in methods:
    n_agree = sum(1 for r in all_results if r[f"{mname}_top1"]==r["lm_top1_vocab"])
    mean_rank = float(np.mean([r[f"{mname}_rank"] for r in all_results]))
    mean_ov   = float(np.mean([r[f"{mname}_ov10"] for r in all_results]))
    ratio = mean_ov / max(exp10, 0.001)
    print(f"  {mname:>16}  {n_agree:>5}/{n:<5} ({n_agree/n:.3f})  {mean_rank:>10.1f}  "
          f"{mean_ov:>10.2f}      {ratio:>8.2f}x")

# Per-category
print(f"\n  Per-category overlap@10 (best method per category):")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat]
    best_ov = max(float(np.mean([r[f"{m}_ov10"] for r in cat_r])) for m in methods)
    best_m  = max(methods, key=lambda m: float(np.mean([r[f"{m}_ov10"] for r in cat_r])))
    best_rank = float(np.mean([r[f"{best_m}_rank"] for r in cat_r]))
    print(f"    {cat:>12}: best_overlap@10={best_ov:.2f} ({best_m})  mean_rank={best_rank:.1f}")

# Improvement from centering
print(f"\n  Debiasing improvement:")
l25_raw_ov10 = float(np.mean([r["l25_raw_ov10"] for r in all_results]))
l25_cen_ov10 = float(np.mean([r["l25_centered_ov10"] for r in all_results]))
t2_raw_ov10  = float(np.mean([r["t2_raw_ov10"] for r in all_results]))
t2_cen_ov10  = float(np.mean([r["t2_centered_ov10"] for r in all_results]))
print(f"    L25 raw={l25_raw_ov10:.3f} → centered={l25_cen_ov10:.3f}  "
      f"delta={l25_cen_ov10-l25_raw_ov10:+.3f}  "
      f"{'✓ improved' if l25_cen_ov10 > l25_raw_ov10 else '✗ worse'}")
print(f"    T2  raw={t2_raw_ov10:.3f}  → centered={t2_cen_ov10:.3f}   "
      f"delta={t2_cen_ov10-t2_raw_ov10:+.3f}  "
      f"{'✓ improved' if t2_cen_ov10 > t2_raw_ov10 else '✗ worse'}")

print(f"\n  VERDICT:")
if l25_cen_ov10 > 2*exp10:
    print(f"  → L25 centered: STRONG signal ({l25_cen_ov10/exp10:.1f}x random)")
elif l25_cen_ov10 > 1.5*exp10:
    print(f"  → L25 centered: MODERATE signal ({l25_cen_ov10/exp10:.1f}x random)")
else:
    print(f"  → L25 centered: WEAK signal ({l25_cen_ov10/exp10:.1f}x random)")

if l25_cen_ov10 > l25_raw_ov10:
    print(f"  → Debiasing HELPS L25 cosine similarity for free-form generation")
else:
    print(f"  → Debiasing does NOT help — frequency bias is not the main issue")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "summary": {
            m: {
                "top1_agreement": sum(1 for r in all_results if r[f"{m}_top1"]==r["lm_top1_vocab"])/n,
                "mean_rank": float(np.mean([r[f"{m}_rank"] for r in all_results])),
                "mean_ov10": float(np.mean([r[f"{m}_ov10"] for r in all_results])),
                "ratio_vs_random": float(np.mean([r[f"{m}_ov10"] for r in all_results])) / max(exp10, 0.001),
            } for m in methods
        },
        "exp10": exp10, "vocab_size": N_VOCAB,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 135 complete.")
