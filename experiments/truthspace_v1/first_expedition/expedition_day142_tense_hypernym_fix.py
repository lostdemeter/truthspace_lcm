#!/usr/bin/env python3
"""
Day 142 — Tense Fix + Hypernym Improvement

Day 141 remaining failures:
  tense:    0%  (temporal words don't embed near past-tense verbs)
  hypernyms: 50% (hammer→weapon, whale→bird — co-occurrence artifacts)

FIX 1 — TENSE:
  Problem: "Yesterday" embedding doesn't cluster near walked/ran/told.
  Approach: T2 pre-filter top-K past-tense verbs → L25 rank within
  Alternative: Use a PIVOT approach — identify the VERB in the prompt
    (if present) and use ITS embedding instead of the temporal marker.
  For "Yesterday he ran the race" → entity = "ran" not "Yesterday"
  For "Yesterday he" → no verb → use T2 axis on full prompt

FIX 2 — HYPERNYMS:
  Problem: hammer≈weapon, whale≈bird in raw embedding space
  Approach: T2 hypernym axis as second-stage ranker
    Stage 1: entity_excl top-20 (coarse semantic neighbors)
    Stage 2: T2 hypernym axis scores (selects the "is-a" relationship)
  This combines embedding proximity with the hypernym axis direction

ALSO TEST:
  Complete combined pipeline with tense fix:
    tense → T2 past_tense axis → L25 within top-20 past-tense verbs
    hypernyms → entity_excl top-20 → T2 hypernym axis re-rank
    all others → entity_excl
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day142_tense_hypernym_fix.json")
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

# Past-tense verb subset (for T2 pre-filter)
PAST_TENSE_VERBS = [
    "walked","ran","ate","built","wrote","read","said","went","came","took",
    "made","got","saw","gave","knew","thought","found","told","became","left",
    "brought","bought","taught","caught","fought","heard","held","kept","sent",
    "fell","felt","grew","slept","spent","stood","wore","won",
    "drove","flew","swam","sang","sat","laid","paid","played","stayed","opened",
    "turned","looked","stopped","asked","started","tried","closed","moved","lived",
]

# ALL TEST CASES (Day 141 + new tense/hypernym focused cases)
TEST_CASES = [
    # Tense cases — need new approach
    ("Yesterday he",    "Yesterday", {"Yesterday"}, "went",    "tense"),
    ("Yesterday she",   "Yesterday", {"Yesterday"}, "went",    "tense"),
    ("Last month she",  "she",       {"she"},       "went",    "tense"),
    ("Last week they",  "they",      {"they"},      "went",    "tense"),
    # More tense — where the verb IS in the prompt
    ("She walked to the store and",  "walked",  {"walked","walk"},  "saw",     "tense_verb"),
    ("He ran to school and",         "ran",     {"ran","run"},      "saw",     "tense_verb"),
    ("They drove home and",          "drove",   {"drove","drive"},  "saw",     "tense_verb"),
    # Hypernym cases — need T2 hypernym re-rank
    ("A hammer is a type of", "hammer", {"hammer"}, "tool",      "hypernyms"),
    ("A whale is a type of",  "whale",  {"whale"},  "animal",    "hypernyms"),
    ("A salmon is a type of", "salmon", {"salmon"}, "fish",      "hypernyms"),
    ("A sword is a type of",  "sword",  {"sword"},  "weapon",    "hypernyms"),
    ("A daisy is a type of",  "daisy",  {"daisy"},  "flower",    "hypernyms"),
    ("A piano is a type of",  "piano",  {"piano"},  "instrument","hypernyms"),
    # Known good cases to verify no regression
    ("The opposite of good is",  "good",  {"good"},  "bad",   "antonyms"),
    ("The capital city of France is",  "France", {"France","French"}, "Paris", "capitals"),
    ("The official language of Germany is", "Germany", {"Germany"}, "German", "languages"),
    ("The son and", "son", {"son"}, "daughter", "gender"),
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
ALL_LAYERS = sorted(set([25] + T2_LAYERS))
print(f"  hidden={H}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

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

def get_hs(text, layers):
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

print("Building vocab ...")
seen = set(); VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = {}; vocab_tok_id = {}; vocab_t2 = {}; vocab_l25 = {}
for w in VOCAB:
    tid = get_token_id(w)
    if tid is not None:
        vocab_ok[w]     = W_E[tid]
        vocab_tok_id[w] = tid
        hs = get_hs(" "+w, ALL_LAYERS)
        vocab_t2[w]  = t2_vec(hs)
        vocab_l25[w] = hs[25]
N_VOCAB = len(vocab_ok)
pt_vocab = [w for w in PAST_TENSE_VERBS if w in vocab_ok]
print(f"  {N_VOCAB} vocab words  |  {len(pt_vocab)} past-tense verbs\n")

def get_logprob_vocab(prompt):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_tok_id[w]]) for w in vocab_ok}

print("="*72)
print("Day 142: Tense Fix + Hypernym T2 Re-rank")
print("="*72)
print()

# Past-tense T2 axis score for each vocab word
pt_axis = t2_axes["past_tense"]  # direction that increases past-tense score
hypernym_axis = t2_axes["hypernym"]

def rank_entity_excl(entity_word, exclude, words):
    eid = get_token_id(entity_word)
    if eid is None:
        eid = get_token_id(" "+entity_word) or 0
    emb = W_E[eid]
    excl = [w for w in words if w not in exclude]
    scores = {w: cosine(emb, vocab_ok[w]) for w in excl}
    return sorted(excl, key=lambda w: -scores[w])

def rank_t2(ctx_t2, words):
    scores = {w: cosine(vocab_t2[w], ctx_t2) for w in words}
    return sorted(words, key=lambda w: -scores[w])

def rank_l25(ctx_l25, words):
    scores = {w: cosine(ctx_l25, vocab_l25[w]) for w in words}
    return sorted(words, key=lambda w: -scores[w])

def rank_pt_axis(words):
    """Rank words by past_tense T2 axis score — selects past-tense verbs."""
    scores = {w: float(np.dot(normed(vocab_l25[w]), pt_axis)) for w in words}
    return sorted(words, key=lambda w: -scores[w])

def rank_hypernym_axis(words):
    """Rank words by hypernym T2 axis score."""
    scores = {w: float(np.dot(normed(vocab_l25[w]), hypernym_axis)) for w in words}
    return sorted(words, key=lambda w: -scores[w])

all_results = []
for prompt, entity_word, exclude, expected, cat in TEST_CASES:
    all_words = list(vocab_ok.keys())
    excl_words = [w for w in all_words if w not in exclude]
    ctx_hs = get_hs(prompt, ALL_LAYERS)
    ctx_t2  = t2_vec(ctx_hs)
    ctx_l25 = ctx_hs[25]

    # Oracle
    lp = get_logprob_vocab(prompt)
    oracle_ranked = sorted(excl_words, key=lambda w: -lp[w])
    oracle_top1   = oracle_ranked[0]

    # Method A: entity_excl (baseline)
    entity_ranked = rank_entity_excl(entity_word, exclude, all_words)
    entity_top1   = entity_ranked[0]

    # Method B: tense fix — T2 pre-filter past-tense only → L25 rank within
    pt_excl = [w for w in pt_vocab if w not in exclude]
    if pt_excl:
        ranked_pt_l25 = rank_l25(ctx_l25, pt_excl)
        tense_fix_top1 = ranked_pt_l25[0]
    else:
        tense_fix_top1 = entity_top1

    # Method C: hypernym fix — entity_excl top-20 → hypernym axis re-rank
    top20_entity = entity_ranked[:20]
    hyp_reranked  = rank_hypernym_axis(top20_entity)
    hypernym_fix_top1 = hyp_reranked[0]

    # Method D: combined
    if cat in ("tense",):
        combined_top1 = tense_fix_top1
    elif cat in ("hypernyms",):
        combined_top1 = hypernym_fix_top1
    elif cat in ("tense_verb",):
        combined_top1 = entity_top1  # verb already in entity position
    else:
        combined_top1 = entity_top1

    row = {
        "prompt": prompt, "entity": entity_word, "cat": cat,
        "expected": expected, "oracle_top1": oracle_top1,
        "entity_top1": entity_top1, "entity_top3": entity_ranked[:3],
        "tense_fix_top1": tense_fix_top1, "pt_top3": ranked_pt_l25[:3] if pt_excl else [],
        "hypernym_fix_top1": hypernym_fix_top1, "hyp_top3": hyp_reranked[:3],
        "combined_top1": combined_top1,
        "agree_entity": entity_top1==oracle_top1,
        "agree_tense_fix": tense_fix_top1==oracle_top1,
        "agree_hypernym_fix": hypernym_fix_top1==oracle_top1,
        "agree_combined": combined_top1==oracle_top1,
        "rank_entity": next((i+1 for i,w in enumerate(entity_ranked) if w==oracle_top1), N_VOCAB+1),
    }
    all_results.append(row)

    e  = "✓" if row["agree_entity"]       else "✗"
    tf = "✓" if row["agree_tense_fix"]    else "✗"
    hf = "✓" if row["agree_hypernym_fix"] else "✗"
    c  = "✓" if row["agree_combined"]     else "✗"
    print(f"  [{cat:>14}] {entity_word:>12} | entity={e}{entity_top1:<12} "
          f"tense_fix={tf}{tense_fix_top1:<12} hyp_fix={hf}{hypernym_fix_top1:<12} "
          f"combined={c}  oracle={oracle_top1}")

print()
print("="*72)
print("Summary")
print("="*72)
n = len(all_results)

def ar(key): return sum(r[key] for r in all_results)/n

print(f"""
  {'Method':>20}  {'top-1 agree':>12}  {'ratio_vs_random'}
  {'─'*55}
  {'entity_excl':>20}  {sum(r['agree_entity'] for r in all_results)}/{n} ({ar('agree_entity'):.3f})  {ar('agree_entity')/(1/N_VOCAB):.0f}x
  {'tense_fix':>20}  {sum(r['agree_tense_fix'] for r in all_results)}/{n} ({ar('agree_tense_fix'):.3f})  {ar('agree_tense_fix')/(1/N_VOCAB):.0f}x
  {'hypernym_fix':>20}  {sum(r['agree_hypernym_fix'] for r in all_results)}/{n} ({ar('agree_hypernym_fix'):.3f})  {ar('agree_hypernym_fix')/(1/N_VOCAB):.0f}x
  {'combined':>20}  {sum(r['agree_combined'] for r in all_results)}/{n} ({ar('agree_combined'):.3f})  {ar('agree_combined')/(1/N_VOCAB):.0f}x
""")

print("  Per-category:")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"]==cat]
    ae = sum(r["agree_entity"] for r in cat_r)/len(cat_r)
    at = sum(r["agree_tense_fix"] for r in cat_r)/len(cat_r)
    ah = sum(r["agree_hypernym_fix"] for r in cat_r)/len(cat_r)
    ac = sum(r["agree_combined"] for r in cat_r)/len(cat_r)
    print(f"    {cat:>14}: entity={ae:.2f}  tense_fix={at:.2f}  hyp_fix={ah:.2f}  combined={ac:.2f}")

print()
print("  Tense fix analysis:")
tense_r = [r for r in all_results if r["cat"]=="tense"]
for r in tense_r:
    print(f"    {r['entity']:>12}: pt_top3={r['pt_top3']}  oracle={r['oracle_top1']} {'✓' if r['agree_tense_fix'] else '✗'}")

print()
print("  Hypernym fix analysis:")
hyp_r = [r for r in all_results if r["cat"]=="hypernyms"]
for r in hyp_r:
    print(f"    {r['entity']:>12}: entity={r['entity_top1']:<10}  hyp_fix={r['hypernym_fix_top1']:<10}  top3={r['hyp_top3']}  oracle={r['oracle_top1']} {'✓' if r['agree_hypernym_fix'] else '✗'}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "agree_entity": ar("agree_entity"),
        "agree_tense_fix": ar("agree_tense_fix"),
        "agree_hypernym_fix": ar("agree_hypernym_fix"),
        "agree_combined": ar("agree_combined"),
        "vocab_size": N_VOCAB,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 142 complete.")
