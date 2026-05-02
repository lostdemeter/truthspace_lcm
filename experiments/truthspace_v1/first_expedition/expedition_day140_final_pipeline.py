#!/usr/bin/env python3
"""
Day 140 — Final Optimal Pipeline: Held-Out Evaluation

DC 339 synthesizes Days 137-139:
  Factual/relational: entity_excl L0 cosine → 52% top-1
  Syntactic/tense: T2 axis → 62% oracle MRR for constrained sets

FINAL PIPELINE:
  Detect entity in prompt
  If temporal/syntactic marker → T2 past_tense axis
  Else → entity_excl L0 cosine (exclude entity + variants)

TEST: 25 FRESH held-out prompts not seen in any Day 124-139 experiment.
  - New capital cities (not France/Japan/Germany/Spain/Italy)
  - New languages (not Brazil/Egypt/China)
  - New hypernyms (not poodle/rose/eagle)
  - New antonyms (not hot/large/dark/young)
  - New gender pairs
  - New tense prompts

EVALUATION:
  entity_excl top-1 agreement vs oracle log-prob
  Per-category breakdown
  Comparison to Day 139 in-sample results
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day140_final_pipeline.json")
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

# HELD-OUT TEST CASES — none appeared in Days 124-139
# (prompt, entity_word, exclude_variants, oracle_expected, category)
HELD_OUT = [
    # New capitals (not France/Japan/Germany/Spain/Italy)
    ("The capital city of Russia is",     "Russia",    {"Russia","Russian"},        "Moscow",    "capitals"),
    ("The capital city of China is",      "China",     {"China","Chinese","Mandarin"},"Beijing",  "capitals"),
    ("The capital city of Australia is",  "Australia", {"Australia","Australian"},   "Canberra",  "capitals"),
    ("The capital city of Greece is",     "Greece",    {"Greece","Greek"},           "Athens",    "capitals"),
    ("The capital city of Poland is",     "Poland",    {"Poland","Polish"},          "Warsaw",    "capitals"),
    ("The capital city of Sweden is",     "Sweden",    {"Sweden","Swedish"},         "Stockholm", "capitals"),

    # New languages
    ("The official language of Germany is",  "Germany", {"Germany","German"},      "German",   "languages"),
    ("The official language of Japan is",    "Japan",   {"Japan","Japanese"},      "Japanese", "languages"),
    ("The official language of Korea is",    "Korea",   {"Korea","Korean"},        "Korean",   "languages"),

    # New hypernyms
    ("A hammer is a type of",  "hammer",   {"hammer"},  "tool",      "hypernyms"),
    ("A ruby is a type of",    "ruby",     {"ruby"},    "gem",       "hypernyms"),
    ("A whale is a type of",   "whale",    {"whale"},   "animal",    "hypernyms"),
    ("A violin is a type of",  "violin",   {"violin"},  "instrument","hypernyms"),

    # New antonyms
    ("The opposite of good is",  "good",  {"good"},    "bad",    "antonyms"),
    ("The opposite of fast is",  "fast",  {"fast"},    "slow",   "antonyms"),
    ("The opposite of clean is", "clean", {"clean"},   "dirty",  "antonyms"),
    ("The opposite of rich is",  "rich",  {"rich"},    "poor",   "antonyms"),
    ("The opposite of loud is",  "loud",  {"loud"},    "quiet",  "antonyms"),
    ("The opposite of strong is","strong",{"strong"},  "weak",   "antonyms"),

    # New gender pairs
    ("The prince and",   "prince",   {"prince","princess"}, "princess","gender"),
    ("The duke and",     "duke",     {"duke"},              "duchess", "gender"),
    ("The actor and",    "actor",    {"actor"},             "actress", "gender"),
    ("The son and",      "son",      {"son"},               "daughter","gender"),

    # Tense (uses T2 — for reference/comparison)
    ("Last month she",   "she",      {"she"},   "went",   "tense"),
    ("Last week they",   "they",     {"they"},  "went",   "tense"),
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
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
T2_LAYERS = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={H}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

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

def get_hs_last(text):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1]-1
    return {L: out.hidden_states[L][0,pos,:].numpy().astype(np.float32) for L in T2_LAYERS}

def t2_vec(hs):
    v = np.zeros(12, np.float32)
    for k, ax in enumerate(AXIS_NAMES_12):
        h = normed(hs.get(DAY78_LAYERS[ax], np.zeros(H))); v[k] = float(np.dot(h, t2_axes[ax]))
    return v

# Build vocab
print("Building vocab ...")
seen = set(); VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = {}; vocab_tok_id = {}
for w in VOCAB:
    tid = get_token_id(w)
    if tid is not None:
        vocab_ok[w] = W_E[tid]; vocab_tok_id[w] = tid

# T2 vecs for vocab
vocab_t2 = {}
for w in vocab_ok:
    hs = get_hs_last(" "+w)
    vocab_t2[w] = t2_vec(hs)
N_VOCAB = len(vocab_ok)
print(f"  {N_VOCAB} words.\n")

def get_logprob_vocab(prompt):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_tok_id[w]]) for w in vocab_ok}

print("="*72)
print("Day 140: Final Pipeline — Held-Out Evaluation")
print("="*72)
print()

all_results = []
for prompt, entity_word, exclude, expected, cat in HELD_OUT:
    all_words = list(vocab_ok.keys())
    excl_words = [w for w in all_words if w not in exclude]

    # Entity L0 embedding (self-exclusion)
    eid = get_token_id(entity_word) or get_token_id(" "+entity_word)
    if eid is None:
        # Try getting entity from prompt tokenization
        inp_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        dec = [tok.decode([t]).strip() for t in inp_ids]
        eid = next((inp_ids[i] for i,d in enumerate(dec) if d.lower()==entity_word.lower()), inp_ids[-1])
    entity_emb = W_E[eid]

    scores_entity = {w: cosine(entity_emb, vocab_ok[w]) for w in excl_words}
    ranked_entity = sorted(excl_words, key=lambda w: -scores_entity[w])
    entity_top1 = ranked_entity[0]

    # T2 method (for tense)
    ctx_hs = get_hs_last(prompt)
    ctx_t2  = t2_vec(ctx_hs)
    scores_t2 = {w: cosine(vocab_t2[w], ctx_t2) for w in excl_words}
    ranked_t2  = sorted(excl_words, key=lambda w: -scores_t2[w])
    t2_top1 = ranked_t2[0]

    # Final routing: tense → T2, rest → entity_excl
    final_top1 = t2_top1 if cat == "tense" else entity_top1
    method_used = "t2" if cat == "tense" else "entity_excl"

    # Oracle
    lp = get_logprob_vocab(prompt)
    oracle_ranked = sorted(excl_words, key=lambda w: -lp[w])
    oracle_top1   = oracle_ranked[0]

    rank_entity = next((i+1 for i,w in enumerate(ranked_entity) if w==oracle_top1), N_VOCAB+1)
    ov10 = len(set(ranked_entity[:10]) & set(oracle_ranked[:10]))

    agree_entity = entity_top1 == oracle_top1
    agree_final  = final_top1  == oracle_top1
    expected_hit = final_top1  == expected

    all_results.append({
        "prompt": prompt, "entity": entity_word, "cat": cat,
        "expected": expected,
        "oracle_top1": oracle_top1, "oracle_top3": oracle_ranked[:3],
        "entity_top1": entity_top1, "entity_top3": ranked_entity[:3],
        "t2_top1": t2_top1,
        "final_top1": final_top1, "method": method_used,
        "rank_entity": rank_entity, "ov10": ov10,
        "agree_entity": agree_entity, "agree_final": agree_final,
        "expected_hit": expected_hit,
    })

    status = "✓" if agree_final else ("~" if final_top1==expected else "✗")
    print(f"  [{cat:>10}|{method_used:>11}] {entity_word:>10} | {status} "
          f"final={final_top1:<12} oracle={oracle_top1:<12} rank={rank_entity:>3}  ov@10={ov10}")

print()
print("="*72)
print("Summary — Day 140 Held-Out Results")
print("="*72)
n = len(all_results)
n_agree  = sum(r["agree_final"] for r in all_results)
n_entity = sum(r["agree_entity"] for r in all_results)
exp10 = 10*10/N_VOCAB
mean_rank = float(np.mean([r["rank_entity"] for r in all_results]))
mean_ov10 = float(np.mean([r["ov10"] for r in all_results]))

print(f"""
  Held-out test cases: {n}  (none seen in Days 124-139)
  Vocab: {N_VOCAB} words | random baseline: top1={1/N_VOCAB:.4f}

  Method           top-1 agree        ratio_vs_random
  ─────────────────────────────────────────────────────
  Final pipeline   {n_agree}/{n} ({n_agree/n:.3f})     {n_agree/n/(1/N_VOCAB):.0f}x random
  entity_excl only {n_entity}/{n} ({n_entity/n:.3f})     {n_entity/n/(1/N_VOCAB):.0f}x random

  Mean rank of oracle in entity_excl: {mean_rank:.1f}
  Mean overlap@10: {mean_ov10:.2f} vs random {exp10:.2f} (ratio {mean_ov10/exp10:.1f}x)
""")

print("  Per-category:")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"]==cat]
    af = sum(r["agree_final"] for r in cat_r)/len(cat_r)
    ae = sum(r["agree_entity"] for r in cat_r)/len(cat_r)
    re = float(np.mean([r["rank_entity"] for r in cat_r]))
    meth = "t2" if cat=="tense" else "entity_excl"
    print(f"    {cat:>12} [{meth:>11}]: {sum(r['agree_final'] for r in cat_r)}/{len(cat_r)}"
          f"  ({af:.3f})  entity_rank={re:.1f}")

print()
print("  Hits:")
for r in all_results:
    if r["agree_final"]:
        print(f"    [{r['cat']}] {r['prompt']!r} → {r['final_top1']} ✓")

print()
print("  Misses (entity_excl):")
for r in all_results:
    if not r["agree_entity"] and r["cat"] != "tense":
        print(f"    [{r['cat']}] {r['entity']:>10} | got={r['entity_top1']:<12} oracle={r['oracle_top1']:<12} "
              f"expected={r['expected']}")

# Comparison to Day 139 in-sample
day139_rate = 0.522
print(f"""
  Generalization check:
    Day 139 (in-sample):    52.2%  (12/23)
    Day 140 (held-out):     {n_agree/n:.1%}  ({n_agree}/{n})
    Delta:                  {n_agree/n - day139_rate:+.3f}
    {'✓ Generalizes!' if n_agree/n >= 0.4 else '✗ Overfits in-sample' if n_agree/n < 0.3 else '~ Partial generalization'}
""")

verdict = "CONFIRMED" if n_agree/n >= 0.4 else "PARTIAL" if n_agree/n >= 0.25 else "FAILS"
print(f"  FINAL VERDICT: Geometric pipeline on held-out = {verdict}")
print(f"  entity_excl L0: {n_agree/n:.0%} top-1 generalization on unseen prompts")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "n_agree": n_agree, "n_total": n,
        "top1_final": n_agree/n,
        "top1_entity": n_entity/n,
        "mean_rank": mean_rank,
        "mean_ov10": mean_ov10,
        "day139_rate": day139_rate,
        "generalization_delta": n_agree/n - day139_rate,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 140 complete.")
