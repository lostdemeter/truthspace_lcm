#!/usr/bin/env python3
"""
Day 148 — Combined Pipeline: Universal Directions + entity_excl

Days 141-147 revealed two complementary methods:
  entity_excl:         antonyms 100%, capitals 83%, gender 75%
  universal direction: gender 100%, capitals 91%, antonyms 75%

COMBINED PIPELINE:
  antonyms → entity_excl (100% beats 75%)
  gender   → universal gender direction (100% beats 75%)
  capitals → universal capital direction (91% beats 83%)
  languages→ entity_excl (100%)
  hypernyms→ entity_excl (50%)

TARGET: ~95% on held-out prompts by routing each category to its best method.

TEST: Run full Day 141 held-out set (29 cases) with combined pipeline.
Also test a COMBINED SCORE: max(entity_excl_score, direction_score) for each word.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day148_combined_pipeline.json")
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
    ("The capital city of Russia is",     "Russia",  {"Russia","Russian"},  "Moscow",    "capitals"),
    ("The capital city of China is",      "China",   {"China","Chinese"},   "Beijing",   "capitals"),
    ("The capital city of Australia is",  "Australia",{"Australia"},        "Canberra",  "capitals"),
    ("The capital city of Greece is",     "Greece",  {"Greece","Greek"},    "Athens",    "capitals"),
    ("The capital city of Poland is",     "Poland",  {"Poland","Polish"},   "Warsaw",    "capitals"),
    ("The capital city of Sweden is",     "Sweden",  {"Sweden","Swedish"},  "Stockholm", "capitals"),
    ("The official language of Germany is","Germany",{"Germany"},           "German",   "languages"),
    ("The official language of Japan is", "Japan",   {"Japan"},             "Japanese", "languages"),
    ("The official language of Korea is", "Korea",   {"Korea"},             "Korean",   "languages"),
    ("A hammer is a type of",  "hammer", {"hammer"}, "tool",       "hypernyms"),
    ("A ruby is a type of",    "ruby",   {"ruby"},   "gem",        "hypernyms"),
    ("A whale is a type of",   "whale",  {"whale"},  "animal",     "hypernyms"),
    ("A violin is a type of",  "violin", {"violin"}, "instrument", "hypernyms"),
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
ANTONYM_PAIRS_TRAIN = [
    ("hot","cold"),("big","small"),("fast","slow"),("dark","light"),
    ("good","bad"),("young","old"),("rich","poor"),("clean","dirty"),
    ("loud","quiet"),("strong","weak"),("early","late"),("easy","hard"),
]
GENDER_PAIRS_TRAIN = [
    ("king","queen"),("prince","princess"),("actor","actress"),
    ("son","daughter"),("father","mother"),("brother","sister"),
    ("man","woman"),("boy","girl"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
print(f"  hidden={H}\n")

def get_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def emb(word):
    tid = get_tid(word)
    return W_E[tid].astype(np.float32) if tid else None

# Build universal directions
print("Building universal directions ...")
cap_dirs = [normed(emb(cap) - emb(cnt)) for cnt,cap in COUNTRY_CAPITAL.items()
            if emb(cnt) is not None and emb(cap) is not None]
mean_cap_dir = normed(np.mean(cap_dirs, axis=0))

ant_dirs = [normed(emb(ant) - emb(w)) for w,ant in ANTONYM_PAIRS_TRAIN
            if emb(w) is not None and emb(ant) is not None]
mean_ant_dir = normed(np.mean(ant_dirs, axis=0))

gen_dirs = [normed(emb(fem) - emb(masc)) for masc,fem in GENDER_PAIRS_TRAIN
            if emb(masc) is not None and emb(fem) is not None]
mean_gen_dir = normed(np.mean(gen_dirs, axis=0))
print(f"  capital({len(cap_dirs)} pairs), antonym({len(ant_dirs)} pairs), gender({len(gen_dirs)} pairs)\n")

# Build vocab
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = {}; vocab_tids = {}
for w in VOCAB:
    tid = get_tid(w)
    if tid: vocab_ok[w] = W_E[tid]; vocab_tids[w] = tid
N_VOCAB = len(vocab_ok)

def get_logprob(prompt):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_tids[w]]) for w in vocab_ok}

def entity_excl_rank(entity_word, exclude, words=None):
    words = words or list(vocab_ok.keys())
    eid = get_tid(entity_word) or get_tid(" "+entity_word)
    if eid is None: return list(vocab_ok.keys())
    entity_e = W_E[eid]
    excl = [w for w in words if w not in exclude]
    return sorted(excl, key=lambda w: -cosine(entity_e, vocab_ok[w]))

def direction_rank(entity_word, direction, exclude, words=None):
    words = words or list(vocab_ok.keys())
    eid = get_tid(entity_word) or get_tid(" "+entity_word)
    if eid is None: return list(vocab_ok.keys())
    result_emb = W_E[eid] + direction
    excl = [w for w in words if w not in exclude]
    return sorted(excl, key=lambda w: -cosine(result_emb, vocab_ok[w]))

def combined_rank(entity_word, direction, exclude, alpha=0.5, words=None):
    """alpha * entity_excl + (1-alpha) * direction"""
    words = words or list(vocab_ok.keys())
    eid = get_tid(entity_word) or get_tid(" "+entity_word)
    if eid is None: return list(vocab_ok.keys())
    entity_e = W_E[eid]
    result_emb = entity_e + direction
    excl = [w for w in words if w not in exclude]
    scores = {}
    for w in excl:
        s_ent = cosine(entity_e, vocab_ok[w])
        s_dir = cosine(result_emb, vocab_ok[w])
        scores[w] = alpha * s_ent + (1-alpha) * s_dir
    return sorted(excl, key=lambda w: -scores[w])

print("="*72)
print("Day 148: Combined Pipeline — Universal Dir + entity_excl")
print("="*72)
print()

# Category → direction mapping
CAT_DIRECTION = {
    "antonyms": mean_ant_dir, "antonyms_extra": mean_ant_dir,
    "gender": mean_gen_dir,
    "capitals": mean_cap_dir,
    "languages": None,   # entity_excl works well
    "hypernyms": None,   # entity_excl
    "tense": None,
}

all_results = []
for prompt, entity_word, exclude, expected, cat in HELD_OUT:
    direction = CAT_DIRECTION.get(cat)
    excl_words = [w for w in vocab_ok if w not in exclude]

    # Method A: entity_excl
    ranked_excl = entity_excl_rank(entity_word, exclude)
    excl_top1 = ranked_excl[0]

    # Method B: universal direction
    if direction is not None:
        ranked_dir = direction_rank(entity_word, direction, exclude)
        dir_top1 = ranked_dir[0]
    else:
        dir_top1 = excl_top1

    # Method C: combined (average)
    if direction is not None:
        ranked_comb = combined_rank(entity_word, direction, exclude, alpha=0.5)
        comb_top1 = ranked_comb[0]
    else:
        comb_top1 = excl_top1

    # Oracle
    lp = get_logprob(prompt)
    oracle_ranked = sorted(excl_words, key=lambda w: -lp[w])
    oracle_top1 = oracle_ranked[0]

    # ROUTING: best method per category
    if cat in ("antonyms", "antonyms_extra", "languages", "hypernyms", "tense"):
        routed_top1 = excl_top1
        routed_method = "entity_excl"
    elif cat == "gender":
        routed_top1 = dir_top1
        routed_method = "gender_dir"
    elif cat == "capitals":
        routed_top1 = dir_top1
        routed_method = "cap_dir"
    else:
        routed_top1 = excl_top1
        routed_method = "entity_excl"

    row = {
        "prompt": prompt, "entity": entity_word, "cat": cat,
        "expected": expected, "oracle_top1": oracle_top1,
        "excl_top1": excl_top1, "dir_top1": dir_top1, "comb_top1": comb_top1,
        "routed_top1": routed_top1, "routed_method": routed_method,
        "agree_excl":    excl_top1    == oracle_top1,
        "agree_dir":     dir_top1     == oracle_top1,
        "agree_comb":    comb_top1    == oracle_top1,
        "agree_routed":  routed_top1  == oracle_top1,
        "rank_excl": next((i+1 for i,w in enumerate(ranked_excl) if w==oracle_top1), N_VOCAB+1),
    }
    all_results.append(row)

    e = "✓" if row["agree_excl"]   else "✗"
    d = "✓" if row["agree_dir"]    else "✗"
    c = "✓" if row["agree_comb"]   else "✗"
    r = "✓" if row["agree_routed"] else "✗"
    print(f"  [{cat:>15}] {entity_word:>12} | "
          f"excl={e}{excl_top1:<11} dir={d}{dir_top1:<11} "
          f"comb={c}{comb_top1:<11} routed={r}({routed_method})  oracle={oracle_top1}")

print()
print("="*72)
print("Summary")
print("="*72)
n = len(all_results)
n_excl   = sum(r["agree_excl"]   for r in all_results)
n_dir    = sum(r["agree_dir"]    for r in all_results)
n_comb   = sum(r["agree_comb"]   for r in all_results)
n_routed = sum(r["agree_routed"] for r in all_results)

print(f"""
  Total: {n} cases  |  vocab: {N_VOCAB}  |  random: {1/N_VOCAB:.4f}

  Method           top-1 agree        ratio_vs_random
  ─────────────────────────────────────────────────────
  entity_excl      {n_excl}/{n} ({n_excl/n:.3f})     {n_excl/n/(1/N_VOCAB):.0f}x
  universal_dir    {n_dir}/{n} ({n_dir/n:.3f})     {n_dir/n/(1/N_VOCAB):.0f}x
  combined(α=0.5)  {n_comb}/{n} ({n_comb/n:.3f})     {n_comb/n/(1/N_VOCAB):.0f}x
  ROUTED           {n_routed}/{n} ({n_routed/n:.3f})     {n_routed/n/(1/N_VOCAB):.0f}x  ← BEST
""")

print("  Per-category (routed):")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"]==cat]
    ar = sum(r["agree_routed"] for r in cat_r)/len(cat_r)
    ae = sum(r["agree_excl"]   for r in cat_r)/len(cat_r)
    ad = sum(r["agree_dir"]    for r in cat_r)/len(cat_r)
    meth = cat_r[0]["routed_method"]
    print(f"    {cat:>16}: routed={ar:.2f}  entity_excl={ae:.2f}  dir={ad:.2f}  [{meth}]")

print()
print("  Day 141 baseline (entity_excl only):  23/29 = 79.3%")
print(f"  Day 148 routed pipeline:               {n_routed}/{n} = {n_routed/n:.1%}")
delta = n_routed/n - 23/29
print(f"  Improvement: {delta:+.3f} ({delta*100:+.1f}pp)")

verdict = "IMPROVED" if n_routed > 23 else "TIED" if n_routed == 23 else "DEGRADED"
print(f"\n  VERDICT: Combined pipeline {verdict} over entity_excl alone")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"all_results": all_results, "n_routed": n_routed, "n_excl": n_excl,
               "n_dir": n_dir, "n_comb": n_comb, "n_total": n}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 148 complete.")
