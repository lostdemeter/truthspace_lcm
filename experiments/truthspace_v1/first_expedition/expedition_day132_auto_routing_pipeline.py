#!/usr/bin/env python3
"""
Day 132 — T2-Guided Automatic Routing Pipeline

Day 131: T2 classifies query categories at 97.1% LOO accuracy.
Day 130: naive routing MRR=0.494 (below L25=0.515) due to wrong category assignments.

FIX: Use T2 centroid nearest-neighbor to assign category, then route
to category-optimal method from Days 124-130:
  tense/gender → T2 axis ranking
  antonyms     → struct_axis (mean-diff, from training pairs)
  hypernyms/capitals/languages → L25 full cosine
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day132_auto_routing_pipeline.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
L_BEST = 25

ROUTING_TABLE = {
    "tense": "t2", "gender": "t2",
    "antonyms": "struct_axis",
    "hypernyms": "l25", "capitals": "l25", "languages": "l25",
}

TRAIN_CASES = {
    "antonyms": [
        ("cold",  ["warm","lukewarm"],  "The opposite of hot is",    ["warm","lukewarm","mild","chilly","Paris","running"]),
        ("sad",   ["angry","bored"],    "The opposite of happy is",  ["angry","bored","worried","tired","capital","walked"]),
        ("dark",  ["bright","sunny"],   "The opposite of bright is", ["bright","sunny","light","clear","flower","never"]),
        ("slow",  ["quick","rapid"],    "The opposite of fast is",   ["quick","rapid","swift","fast","capital","walked"]),
        ("small", ["big","huge"],       "The opposite of large is",  ["big","huge","enormous","great","banana","running"]),
        ("wrong", ["correct","accurate"],"The opposite of right is", ["correct","accurate","true","right","stone","never"]),
    ],
    "tense": [
        (["walked","ran","ate","built","saw"],["walk","run","eat","build","see","Paris","flower","because","cold"],"Yesterday he"),
        (["walked","ran","wrote","ate","saw"],["walk","run","write","eat","see","Paris","banana","because","cold"],"Yesterday she"),
        (["walked","ran","ate","built","saw"],["walk","run","eat","build","see","capital","flower","never","stone"],"Yesterday they"),
    ],
    "gender": [
        (["queen","princess","duchess","empress"],["prince","knight","lord","earl","banana","quickly","walked","stone"],"The king and"),
        (["king","prince","duke","emperor"],["princess","lady","duchess","countess","banana","quickly","walked","stone"],"The queen and"),
        (["mother","sister","daughter","aunt"],["brother","uncle","nephew","cousin","banana","quickly","stone","cold"],"The father and"),
    ],
    "capitals": [
        (["Paris"],  ["London","Rome","Berlin","Madrid","banana","running","very","because"],"The capital city of France is"),
        (["Tokyo"],  ["Osaka","Beijing","Seoul","Bangkok","quietly","ocean","third","green"],"The capital city of Japan is"),
        (["Berlin"], ["Frankfurt","Vienna","Warsaw","Amsterdam","walked","stone","never","that"],"The capital city of Germany is"),
        (["Madrid"], ["Lisbon","Rome","Paris","Brussels","banana","quickly","cold","stone"],"The capital city of Spain is"),
        (["Rome"],   ["Milan","Paris","Vienna","Athens","walked","never","quickly","because"],"The capital city of Italy is"),
        (["Moscow"], ["Petersburg","Kiev","Minsk","Warsaw","banana","stone","cold","quietly"],"The capital city of Russia is"),
    ],
    "languages": [
        (["Portuguese"],["Spanish","Italian","French","English","purple","running","before","cold"],"The official language of Brazil is"),
        (["Arabic"],    ["Hebrew","Turkish","Persian","Urdu","quickly","stone","later","because"],"The official language of Egypt is"),
        (["Mandarin"],  ["Japanese","Korean","Cantonese","Thai","capital","walked","never","cold"],"The official language of China is"),
        (["Hindi"],     ["Urdu","Bengali","Tamil","Punjabi","banana","running","because","stone"],"The official language of India is"),
    ],
    "hypernyms": [
        (["dog","animal","mammal"],["cat","rabbit","horse","pet","quickly","stone","because","red"],"A poodle is a type of"),
        (["flower","plant","bloom"],["tree","bush","grass","weed","capital","walked","never","cold"],"A rose is a type of"),
        (["tool","instrument","implement"],["machine","device","appliance","weapon","flower","language","quickly","green"],"A hammer is a type of"),
        (["bird","animal","vertebrate"],["insect","reptile","fish","mammal","banana","stone","cold","quietly"],"An eagle is a type of"),
        (["gem","mineral","stone"],["metal","crystal","rock","glass","walked","never","because","purple"],"A ruby is a type of"),
    ],
}

TEST_HELD_OUT = [
    ("antonyms", "The opposite of young is",   ["old","elderly","aged"],    ["young","new","fresh","recent","Paris","quickly","banana","stone"]),
    ("antonyms", "The opposite of weak is",    ["strong","powerful"],       ["fragile","feeble","frail","delicate","Paris","quietly","ocean","third"]),
    ("antonyms", "The opposite of quiet is",   ["loud","noisy"],            ["silent","calm","peaceful","gentle","capital","flower","stone","never"]),
    ("capitals", "The capital city of Australia is",["Canberra"],           ["Sydney","Melbourne","Perth","Brisbane","banana","running","because","cold"]),
    ("capitals", "The capital city of Canada is",  ["Ottawa"],              ["Toronto","Vancouver","Montreal","Calgary","walked","stone","never","quickly"]),
    ("capitals", "The capital city of Brazil is",  ["Brasilia"],            ["Rio","Sao Paulo","Buenos Aires","Lima","banana","cold","stone","quickly"]),
    ("languages","The official language of Mexico is",["Spanish"],          ["English","French","Portuguese","Italian","running","banana","because","cold"]),
    ("languages","The official language of Japan is", ["Japanese"],         ["Chinese","Korean","Mandarin","Thai","quickly","stone","never","walked"]),
    ("hypernyms","A salmon is a type of",  ["fish","animal","vertebrate"],  ["bird","insect","plant","tree","capital","walked","never","cold"]),
    ("hypernyms","A piano is a type of",   ["instrument","tool","device"],  ["weapon","machine","vehicle","toy","flower","language","quickly","green"]),
    ("tense",    "Yesterday we",   ["walked","ran","ate","built","saw"],    ["walk","run","eat","build","see","Paris","flower","because","cold"]),
    ("gender",   "The prince and", ["princess","queen","duchess","empress"],["king","duke","lord","earl","banana","quickly","walked","stone"]),
]

DAY78_LAYERS = {"gender":27,"comparative":15,"hypernym":28,"plural":1,"synonym":28,"concrete":28,
                "past_tense":28,"antonym":28,"passive":28,"causation":28,"question":28,"negation":28}
AXIS_NAMES_12 = ["gender","comparative","hypernym","plural","synonym","concrete",
                 "past_tense","antonym","passive","causation","question","negation"]
AXIS_PAIRS = {
    "gender":[("The king ruled","The queen ruled"),("A man walked","A woman walked"),("His brother arrived","His sister arrived")],
    "comparative":[("The fast car","The faster car"),("A big dog","A bigger dog"),("The tall tree","The taller tree")],
    "hypernym":[("The dog ran","The animal ran"),("A rose bloomed","A flower bloomed"),("The car sped","The vehicle sped")],
    "plural":[("A dog played","Dogs played"),("The cat sat","The cats sat"),("A bird sang","Birds sang")],
    "synonym":[("He is big","He is large"),("She is small","She is tiny"),("It is cold","It is frigid")],
    "concrete":[("The stone is heavy","The burden is heavy"),("The long road","The long journey"),("The high wall","The high barrier")],
    "past_tense":[("I walk every morning","I walked every morning"),("She runs through park","She ran through park"),("He eats","He ate")],
    "antonym":[("It is hot","It is cold"),("He runs fast","He runs slow"),("She is happy","She is sad")],
    "passive":[("The cat chased mouse","The mouse was chased"),("John broke window","The window was broken")],
    "causation":[("The rain falls","The ground gets wet"),("The fire burns","The wood turns to ash")],
    "question":[("She is tired","Is she tired"),("He can swim","Can he swim"),("They went","Did they go")],
    "negation":[("The dog is fast","The dog is not fast"),("She can swim","She cannot swim")],
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
T2_LAYERS = sorted(set(DAY78_LAYERS.values()))
ALL_LAYERS = sorted(set([L_BEST] + T2_LAYERS))
print(f"  hidden={H}\n")

def get_hs(text, layers=None):
    if layers is None: layers = ALL_LAYERS
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

def get_logprob(prompt, word):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1).numpy()
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return float(lp[ids[0]]) if ids else float("-inf")

# Build T2 axes
print("Building T2 axes ...")
t2_axes = {}
for ax in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax]
    diffs = []
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

# Build struct_axis for antonyms
print("Building struct_axis (antonyms) ...")
ant_deltas = []
for correct, wrong_pool, prompt, _ in TRAIN_CASES["antonyms"]:
    h_c = get_hs(" "+correct, [L_BEST])[L_BEST]
    h_w = np.mean([get_hs(" "+w, [L_BEST])[L_BEST] for w in wrong_pool], axis=0).astype(np.float32)
    d = h_c - h_w; nv = np.linalg.norm(d)
    if nv > 1e-6: ant_deltas.append(d/nv)
struct_axis = normed(np.mean(ant_deltas, axis=0).astype(np.float32))
print("  Done.\n")

# Build T2 category centroids from training prompts
print("Building category T2 centroids ...")
cat_centroids = {}
for cat, cases in TRAIN_CASES.items():
    vecs = [t2_vec(get_hs(p if isinstance(p, str) else cases[0][2], T2_LAYERS))
            for *_, p, __ in [(c[2], None) for c in cases]]
    # fix: extract prompts properly
    prompts = [c[2] for c in cases]
    vecs = [t2_vec(get_hs(p, T2_LAYERS)) for p in prompts]
    m = np.mean(vecs, axis=0)
    cat_centroids[cat] = normed(m.astype(np.float32))
print("  Done.\n")

def classify_prompt(prompt):
    v = t2_vec(get_hs(prompt, T2_LAYERS))
    scores = {cat: cosine(v, cat_centroids[cat]) for cat in cat_centroids}
    return max(scores, key=lambda c: scores[c]), scores

def rank_candidates(prompt, all_cands, method, ctx_hs=None):
    if ctx_hs is None:
        ctx_hs = get_hs(prompt, ALL_LAYERS)
    cand_hs = {w: get_hs(" "+w, ALL_LAYERS) for w in all_cands}
    if method == "t2":
        ctx_t2 = t2_vec(ctx_hs)
        scores = {w: cosine(t2_vec(cand_hs[w]), ctx_t2) for w in all_cands}
    elif method == "struct_axis":
        scores = {w: float(np.dot(normed(cand_hs[w][L_BEST]), struct_axis)) for w in all_cands}
    else:  # l25
        ctx_h = ctx_hs[L_BEST]
        scores = {w: cosine(ctx_h, cand_hs[w][L_BEST]) for w in all_cands}
    return sorted(all_cands, key=lambda w: -scores[w]), scores

def mrr(ranked, correct):
    r = next((i+1 for i,w in enumerate(ranked) if w in correct), len(ranked)+1)
    return 1.0/r

print("="*72)
print("Day 132: T2-Guided Automatic Routing Pipeline")
print("="*72)
print()

all_results = []
for cat_true, prompt, correct, wrong in TEST_HELD_OUT:
    all_cands = list(correct) + list(wrong)
    cat_pred, cat_scores = classify_prompt(prompt)
    method = ROUTING_TABLE[cat_pred]
    ranked, scores = rank_candidates(prompt, all_cands, method)
    mrr_auto = mrr(ranked, correct)

    # Baselines
    ranked_l25, _  = rank_candidates(prompt, all_cands, "l25")
    ranked_t2,  _  = rank_candidates(prompt, all_cands, "t2")
    lp = {w: get_logprob(prompt, w) for w in all_cands}
    ranked_lp = sorted(all_cands, key=lambda w: -lp[w])
    mrr_lp  = mrr(ranked_lp,  correct)
    mrr_l25 = mrr(ranked_l25, correct)
    mrr_t2  = mrr(ranked_t2,  correct)

    correct_cat = cat_pred == cat_true
    all_results.append({
        "cat_true": cat_true, "cat_pred": cat_pred,
        "correct_routing": correct_cat, "method": method,
        "mrr_auto": mrr_auto, "mrr_lp": mrr_lp,
        "mrr_l25": mrr_l25, "mrr_t2": mrr_t2,
        "prompt": prompt,
    })
    print(f"  [{cat_true:>12}] pred={cat_pred:>12} {'✓' if correct_cat else '✗'}  "
          f"method={method:>11}  auto={mrr_auto:.3f}  L25={mrr_l25:.3f}  "
          f"T2={mrr_t2:.3f}  lp={mrr_lp:.3f}")

print()
n = len(all_results)
m_auto = float(np.mean([r["mrr_auto"] for r in all_results]))
m_l25  = float(np.mean([r["mrr_l25"]  for r in all_results]))
m_t2   = float(np.mean([r["mrr_t2"]   for r in all_results]))
m_lp   = float(np.mean([r["mrr_lp"]   for r in all_results]))
n_correct_cat = sum(1 for r in all_results if r["correct_routing"])

print("="*72)
print("Summary")
print("="*72)
print(f"""
  Category classification accuracy: {n_correct_cat}/{n} = {n_correct_cat/n:.3f}

  Method              MRR      % oracle
  ─────────────────────────────────────────
  log-prob (oracle)  {m_lp:.4f}    100%
  Auto-routed        {m_auto:.4f}    {100*m_auto/m_lp:.1f}%
  L25 cosine         {m_l25:.4f}    {100*m_l25/m_lp:.1f}%
  T2 isolated        {m_t2:.4f}    {100*m_t2/m_lp:.1f}%
  Random baseline    0.1667

  Auto-routed vs L25: {m_auto-m_l25:+.4f}  {'✓ routing helps' if m_auto > m_l25 else '✗ routing does not help'}
""")

print("  Per-category auto-routed MRR:")
for cat_name in ROUTING_TABLE:
    cat_r = [r for r in all_results if r["cat_true"] == cat_name]
    if not cat_r: continue
    m = float(np.mean([r["mrr_auto"] for r in cat_r]))
    ml = float(np.mean([r["mrr_lp"]  for r in cat_r]))
    mth = ROUTING_TABLE[cat_name]
    print(f"    {cat_name:>12}: MRR={m:.4f}  oracle={ml:.4f}  method={mth}  "
          f"{'✓' if m > 0.5 else '~' if m > 0.3 else '✗'}")

verdict = (f"→ Auto-routing SUCCEEDS: MRR={m_auto:.4f} > L25={m_l25:.4f}"
           if m_auto > m_l25 else
           f"→ Auto-routing FAILS: MRR={m_auto:.4f} <= L25={m_l25:.4f}")
print(f"\n  {verdict}")
print(f"  TruthSpace geometric pipeline: {100*m_auto/m_lp:.0f}% of oracle MRR")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"all_results": all_results,
               "mrr_auto": m_auto, "mrr_l25": m_l25, "mrr_t2": m_t2, "mrr_lp": m_lp,
               "cat_acc": n_correct_cat/n}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 132 complete.")
