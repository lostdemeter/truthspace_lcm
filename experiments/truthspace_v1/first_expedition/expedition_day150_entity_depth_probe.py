#!/usr/bin/env python3
"""
Day 150 — Entity Hidden States at Depth: L0 vs L6 vs L12 vs L20 vs L25

W_E (L0) is at 82.8% ceiling. The remaining 17.2% failures are:
  - Obscure capitals (Australia→Canberra: L0 gives Sydney)
  - Co-occurrence ambiguity (hammer→tool vs hammer→weapon)
  - Tense (irreducible)

HYPOTHESIS: Running the entity word through deeper transformer layers
enriches the encoding with more specific factual information.

For "Australia", a single-token forward pass to L12 might encode:
  L0:  Australia ≈ Sydney, Melbourne, New Zealand (proximity)
  L12: Australia ≈ Canberra (after 12 layers of self-enrichment?)

METHOD:
  1. Run single-token prompt " Australia" through model to get hidden states at L0...L25
  2. Use each layer's entity hidden state to rank vocab words (entity_excl)
  3. Compare to oracle for held-out prompts

KEY QUESTION: Does any intermediate layer outperform L0 for the hard cases?

Also compare: single-token vs full-context entity hidden state.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day150_entity_depth_probe.json")
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

PROBE_LAYERS = [0, 6, 12, 16, 20, 25]  # L0=W_E, then DRUM, COMB, MUSIC zones

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

def get_entity_hidden_states(word, layers):
    """Run single-token prompt through model, return hidden states at each layer."""
    inp = tok(" "+word, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in layers}

def get_logprob_vocab(prompt, vocab_ids):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_ids[w]]) for w in vocab_ids}

# Build vocab
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = {}; vocab_tids = {}
for w in VOCAB:
    tid = get_tid(w)
    if tid: vocab_ok[w] = W_E[tid]; vocab_tids[w] = tid
N = len(vocab_ok)

# Pre-compute vocab hidden states at each probe layer
print("Computing vocab hidden states at probe layers ...")
vocab_hs = {L: {} for L in PROBE_LAYERS}
for w in vocab_ok:
    hs = get_entity_hidden_states(w, PROBE_LAYERS)
    for L in PROBE_LAYERS:
        vocab_hs[L][w] = hs[L]
print(f"  Done. {N} words × {len(PROBE_LAYERS)} layers\n")

print("="*72)
print(f"Day 150: Entity Hidden States at Depth (Layers {PROBE_LAYERS})")
print("="*72)
print()

all_results = []
for prompt, entity_word, exclude, expected, cat in HELD_OUT:
    excl_words = [w for w in vocab_ok if w not in exclude]

    # Oracle
    lp = get_logprob_vocab(prompt, vocab_tids)
    oracle_top1 = sorted(excl_words, key=lambda w: -lp[w])[0]

    # Entity hidden states at each layer
    entity_hs = get_entity_hidden_states(entity_word, PROBE_LAYERS)

    row = {"prompt": prompt, "entity": entity_word, "cat": cat,
           "expected": expected, "oracle_top1": oracle_top1}

    per_layer = {}
    for L in PROBE_LAYERS:
        e_vec = entity_hs[L]
        scores = {w: cosine(e_vec, vocab_hs[L][w]) for w in excl_words}
        ranked = sorted(excl_words, key=lambda w: -scores[w])
        top1 = ranked[0]
        rank = next((i+1 for i,w in enumerate(ranked) if w==oracle_top1), N+1)
        per_layer[L] = {"top1": top1, "rank": rank, "agree": top1==oracle_top1}

    row["layers"] = per_layer
    all_results.append(row)

    layer_str = "  ".join(
        f"L{L}={'✓' if per_layer[L]['agree'] else '✗'}{per_layer[L]['top1']:<10}"
        for L in PROBE_LAYERS
    )
    print(f"  [{cat:>15}] {entity_word:>12}: {layer_str}  oracle={oracle_top1}")

print()
print("="*72)
print("Summary by layer")
print("="*72)
n = len(all_results)
print()
print(f"  {'Layer':>6}  {'top-1 agree':>12}  {'mean_rank':>10}")
print(f"  {'-'*40}")
for L in PROBE_LAYERS:
    n_agree = sum(r["layers"][L]["agree"] for r in all_results)
    mean_r  = float(np.mean([r["layers"][L]["rank"] for r in all_results]))
    print(f"  L{L:>4}   {n_agree}/{n} ({n_agree/n:.3f})    {mean_r:>8.1f}")

print()
print("  Best layer per category:")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"]==cat]
    best_L = max(PROBE_LAYERS, key=lambda L: sum(r["layers"][L]["agree"] for r in cat_r))
    best_acc = sum(r["layers"][best_L]["agree"] for r in cat_r)/len(cat_r)
    all_accs = {L: sum(r["layers"][L]["agree"] for r in cat_r)/len(cat_r) for L in PROBE_LAYERS}
    print(f"    {cat:>16}: best=L{best_L}({best_acc:.2f})  " +
          "  ".join(f"L{L}={all_accs[L]:.2f}" for L in PROBE_LAYERS))

# Focus on hard cases
print()
print("  Hard cases (L0 fails) at each layer:")
hard_cases = [r for r in all_results if not r["layers"][0]["agree"]]
for r in hard_cases:
    layer_str = "  ".join(
        f"L{L}={'✓' if r['layers'][L]['agree'] else '✗'}({r['layers'][L]['top1']})"
        for L in PROBE_LAYERS
    )
    print(f"    {r['entity']:>12}: {layer_str}  oracle={r['oracle_top1']}")

print()
best_layer_overall = max(PROBE_LAYERS, key=lambda L: sum(r["layers"][L]["agree"] for r in all_results))
best_n = sum(r["layers"][best_layer_overall]["agree"] for r in all_results)
l0_n   = sum(r["layers"][0]["agree"] for r in all_results)
print(f"  Best layer: L{best_layer_overall} ({best_n}/{n} = {best_n/n:.1%})")
print(f"  L0 baseline: {l0_n}/{n} = {l0_n/n:.1%}")
print(f"  Delta: {best_n-l0_n:+d} ({(best_n-l0_n)/n:+.1%})")
if best_n > l0_n:
    print(f"  VERDICT: Deeper entity encoding HELPS ({best_layer_overall} > L0)")
elif best_n == l0_n:
    print(f"  VERDICT: No improvement from deeper entity encoding")
else:
    print(f"  VERDICT: Deeper encoding HURTS (L0 is best)")

with open(OUTPUT_FILE, "w") as f:
    json.dump([{k: (v if not isinstance(v,dict) else {str(kk):vv for kk,vv in v.items()})
                for k,v in r.items()} for r in all_results], f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 150 complete.")
