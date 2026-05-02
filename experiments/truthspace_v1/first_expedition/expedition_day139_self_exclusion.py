#!/usr/bin/env python3
"""
Day 139 — Self-Exclusion Fix for L0 Entity Embedding

Day 138: Entity L0 achieves 13% top-1 (3/23), 6.41x random.
  Capitals: oracle at rank 2 (rank 1 = entity itself)
  Antonyms: oracle at rank 2 (rank 1 = entity or related form)

FIX: Exclude the entity word and its morphological variants from rankings.
  E.g., for entity="France", exclude: French, France
  For entity="hot", exclude: hot, hotter, hottest, hotly

ALSO TEST: Combined pipeline
  tense/gender → T2 axis (best for syntactic, Day 132)
  capitals/languages/hypernyms/antonyms → L0 entity cosine (exclude self)

PREDICT: Overall top-1 agreement should reach 30-50% with self-exclusion.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day139_self_exclusion.json")
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

# Exclusion sets per entity (entity itself + morphological neighbors in vocab)
ENTITY_EXCLUDE = {
    "France":    {"French","France"},
    "Japan":     {"Japanese","Japan"},
    "Germany":   {"German","Germany"},
    "Spain":     {"Spanish","Spain"},
    "Italy":     {"Italian","Italy"},
    "Brazil":    {"Portuguese","Brazil"},  # NOTE: don't exclude Portuguese (that's the answer!)
    "Egypt":     {"Arabic","Egypt"},
    "China":     {"Mandarin","Chinese","China"},
    "poodle":    {"poodle"},
    "rose":      {"rose"},
    "eagle":     {"eagle"},
    "hot":       {"hot"},
    "large":     {"large","big"},  # don't exclude the antonym (small)
    "dark":      {"dark"},
    "young":     {"young"},
    "king":      {"king"},
    "queen":     {"queen"},
    "father":    {"father"},
    "Yesterday": {"Yesterday","yesterday"},
    "sun":       {"sun"},
    "cat":       {"cat"},
}
# Fix: Brazil exclusion should NOT include Portuguese
ENTITY_EXCLUDE["Brazil"] = {"Brazil"}
ENTITY_EXCLUDE["Egypt"]  = {"Egypt"}
ENTITY_EXCLUDE["China"]  = {"China","Chinese"}

TEST_CASES = [
    ("The capital city of France is",   "France",   "capitals"),
    ("The capital city of Japan is",    "Japan",    "capitals"),
    ("The capital city of Germany is",  "Germany",  "capitals"),
    ("The capital city of Spain is",    "Spain",    "capitals"),
    ("The capital city of Italy is",    "Italy",    "capitals"),
    ("The official language of Brazil is",  "Brazil",   "languages"),
    ("The official language of Egypt is",   "Egypt",    "languages"),
    ("The official language of China is",   "China",    "languages"),
    ("A poodle is a type of",  "poodle",  "hypernyms"),
    ("A rose is a type of",    "rose",    "hypernyms"),
    ("An eagle is a type of",  "eagle",   "hypernyms"),
    ("The opposite of hot is",   "hot",   "antonyms"),
    ("The opposite of large is", "large", "antonyms"),
    ("The opposite of dark is",  "dark",  "antonyms"),
    ("The opposite of young is", "young", "antonyms"),
    ("Yesterday he",   "Yesterday",  "tense"),
    ("Yesterday she",  "Yesterday",  "tense"),
    ("Yesterday they", "Yesterday",  "tense"),
    ("The king and",    "king",    "gender"),
    ("The queen and",   "queen",   "gender"),
    ("The father and",  "father",  "gender"),
    ("The sun rises in the",  "sun",   "free"),
    ("The cat sat on the",    "cat",   "free"),
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

T2_CATS = {"tense", "gender"}  # use T2 axis for these

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
T2_LAYERS = sorted(set(DAY78_LAYERS.values()))
ALL_T2_LAYERS = T2_LAYERS
print(f"  hidden={H}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_entity_embedding(prompt, entity_word):
    inp = tok(prompt, return_tensors="pt")
    tokens = inp["input_ids"][0].tolist()
    decoded = [tok.decode([t]).strip() for t in tokens]
    for i, d in enumerate(decoded):
        if d.lower() == entity_word.lower(): return W_E[tokens[i]], i
        if entity_word.lower() in d.lower(): return W_E[tokens[i]], i
    eid = get_token_id(entity_word)
    if eid: return W_E[eid], -1
    return W_E[tokens[-1]], len(tokens)-1

def get_logprob_vocab(prompt, vocab_tok_ids):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
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

def get_hs_last(text, layers):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1]-1
    return {L: out.hidden_states[L][0,pos,:].numpy().astype(np.float32) for L in layers}

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
N_VOCAB = len(vocab_ok)

# Precompute T2 vecs for vocab words
vocab_t2 = {}
for w in vocab_ok:
    hs = get_hs_last(" "+w, T2_LAYERS)
    vocab_t2[w] = t2_vec(hs)
print(f"  {N_VOCAB} vocab words, T2 vecs computed.\n")

print("="*72)
print("Day 139: Self-Exclusion Fix + Combined Pipeline")
print("="*72)
print()

all_results = []
for prompt, entity_word, cat in TEST_CASES:
    entity_emb, entity_pos = get_entity_embedding(prompt, entity_word)
    exclude = ENTITY_EXCLUDE.get(entity_word, {entity_word})
    all_words = list(vocab_ok.keys())
    excl_words = [w for w in all_words if w not in exclude]

    # Oracle within vocab
    lp = get_logprob_vocab(prompt, vocab_tok_id)
    oracle_ranked = sorted(excl_words, key=lambda w: -lp[w])
    oracle_top1 = oracle_ranked[0]

    # Method A: L0 entity cosine, with self-exclusion
    scores_entity = {w: cosine(entity_emb, vocab_ok[w]) for w in excl_words}
    ranked_entity = sorted(excl_words, key=lambda w: -scores_entity[w])
    entity_top1 = ranked_entity[0]

    # Method B: L0 entity cosine, WITHOUT self-exclusion (Day 138 baseline)
    scores_entity_full = {w: cosine(entity_emb, vocab_ok[w]) for w in all_words}
    ranked_entity_full = sorted(all_words, key=lambda w: -scores_entity_full[w])
    entity_top1_full = ranked_entity_full[0]

    # Method C: T2 axis (best for tense/gender)
    ctx_hs = get_hs_last(prompt, T2_LAYERS)
    ctx_t2  = t2_vec(ctx_hs)
    scores_t2 = {w: cosine(vocab_t2[w], ctx_t2) for w in excl_words}
    ranked_t2 = sorted(excl_words, key=lambda w: -scores_t2[w])
    t2_top1 = ranked_t2[0]

    # Combined: T2 for tense/gender, L0 entity for others
    combined_top1 = t2_top1 if cat in T2_CATS else entity_top1

    rank_entity = next((i+1 for i,w in enumerate(ranked_entity) if w==oracle_top1), N_VOCAB+1)
    ov10_entity = len(set(ranked_entity[:10]) & set(oracle_ranked[:10]))
    ov10_t2     = len(set(ranked_t2[:10])     & set(oracle_ranked[:10]))

    agree_entity   = entity_top1   == oracle_top1
    agree_t2       = t2_top1       == oracle_top1
    agree_combined = combined_top1 == oracle_top1

    all_results.append({
        "prompt": prompt, "entity": entity_word, "cat": cat,
        "oracle_top1": oracle_top1, "oracle_top5": oracle_ranked[:5],
        "entity_excl_top1": entity_top1, "entity_excl_top5": ranked_entity[:5],
        "entity_full_top1": entity_top1_full,
        "t2_top1": t2_top1, "combined_top1": combined_top1,
        "rank_entity_excl": rank_entity,
        "ov10_entity": ov10_entity, "ov10_t2": ov10_t2,
        "agree_entity": agree_entity, "agree_t2": agree_t2,
        "agree_combined": agree_combined,
        "excluded": list(exclude),
    })

    print(f"  [{cat:>10}|{entity_word:>12}] {prompt!r}")
    print(f"    oracle={oracle_top1}  top5={oracle_ranked[:5]}")
    print(f"    entity_excl: top1={entity_top1:>14} r={rank_entity:>3}  ov@10={ov10_entity}  {'✓' if agree_entity else ''}")
    print(f"    t2:          top1={t2_top1:>14}                          {'✓' if agree_t2 else ''}")
    print(f"    combined:    top1={combined_top1:>14}  ({'entity_excl' if cat not in T2_CATS else 't2'})  {'✓' if agree_combined else ''}")
    print()

# Summary
print("="*72)
print("Summary")
print("="*72)
n = len(all_results)
exp10 = 10*10/N_VOCAB

def ar(key): return sum(r[key] for r in all_results)/n

print(f"""
  Vocab: {N_VOCAB} words
  Random baseline: top1={1/N_VOCAB:.4f}  overlap@10={exp10:.2f}

  Method           top1_agree   ratio_vs_random
  ───────────────────────────────────────────────
  entity_excl      {sum(r['agree_entity'] for r in all_results)}/{n}  ({ar('agree_entity'):.3f})  {ar('agree_entity')/(1/N_VOCAB):.0f}x random
  t2_only          {sum(r['agree_t2'] for r in all_results)}/{n}  ({ar('agree_t2'):.3f})  {ar('agree_t2')/(1/N_VOCAB):.0f}x random
  combined         {sum(r['agree_combined'] for r in all_results)}/{n}  ({ar('agree_combined'):.3f})  {ar('agree_combined')/(1/N_VOCAB):.0f}x random
""")

print("  Per-category (combined):")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat]
    ae = sum(r["agree_entity"]   for r in cat_r)/len(cat_r)
    at = sum(r["agree_t2"]       for r in cat_r)/len(cat_r)
    ac = sum(r["agree_combined"] for r in cat_r)/len(cat_r)
    re = float(np.mean([r["rank_entity_excl"] for r in cat_r]))
    meth = "t2" if cat in T2_CATS else "entity_excl"
    print(f"    {cat:>12} [{meth:>11}]: entity={ae:.3f}  t2={at:.3f}  combined={ac:.3f}  rank={re:.1f}")

print()
n_hits = sum(r["agree_combined"] for r in all_results)
print("  Combined pipeline hits:")
for r in all_results:
    if r["agree_combined"]:
        meth = "t2" if r["cat"] in T2_CATS else "entity_excl"
        print(f"    [{r['cat']}|{meth}] {r['prompt']!r} → {r['combined_top1']} ✓")

print()
ar_combined = ar("agree_combined")
if ar_combined > 0.4:
    print(f"  VERDICT: Combined pipeline achieves {ar_combined:.0%} top-1 — STRONG")
    print(f"  → Entity L0 embedding + T2 axis = viable free-form generation baseline")
elif ar_combined > 0.2:
    print(f"  VERDICT: Combined pipeline achieves {ar_combined:.0%} top-1 — MODERATE")
elif ar_combined > 0:
    print(f"  VERDICT: Combined pipeline achieves {ar_combined:.0%} top-1 — WEAK but non-zero")
else:
    print(f"  VERDICT: Combined pipeline achieves 0% — generation requires more than embeddings")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "n_total": n,
        "agree_entity_excl": ar("agree_entity"),
        "agree_t2": ar("agree_t2"),
        "agree_combined": ar("agree_combined"),
        "vocab_size": N_VOCAB, "random_top1": 1/N_VOCAB,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 139 complete.")
