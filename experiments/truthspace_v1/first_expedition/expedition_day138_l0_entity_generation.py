#!/usr/bin/env python3
"""
Day 138 — L0 Entity-Position Free-Form Generation

Day 137: L0 entity HS achieves 97% oracle MRR for constrained candidate sets.
Day 133-136: L25 last-token fails for free-form vocab (0.78x random).

HYPOTHESIS: Use L0 embedding of entity token to rank the 237-word vocab.
Since L0 has no "is" bias (it's just the static embedding), France's L0
representation should directly cluster near Paris in embedding space.

PROCEDURE:
  For each prompt:
    1. Identify entity position (France, poodle, hot, etc.)
    2. Extract L0 hidden state at entity position = raw token embedding
    3. Extract L0 hidden state for each vocab word = raw token embedding
    4. Rank vocab by cosine(entity_L0, word_L0)
    5. Compare top-1 to LM oracle top-1

Also test L0 of LAST TOKEN (no context) vs L0 entity.

Also test a KEY VARIANT: is there an even simpler approach?
  Direct embedding lookup: cos(W_E[entity_token_id], W_E[cand_token_id])
  where W_E is the embedding matrix — purely geometric, no forward pass needed
  for the candidate words.

This would be the purest TruthSpace test: can STATIC embeddings predict
factual associations?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day138_l0_entity_generation.json")
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

# Prompts with entity position marked
# (prompt, entity_word, category)
TEST_CASES = [
    # Factual — entity = country
    ("The capital city of France is",   "France",   "capitals"),
    ("The capital city of Japan is",    "Japan",    "capitals"),
    ("The capital city of Germany is",  "Germany",  "capitals"),
    ("The capital city of Spain is",    "Spain",    "capitals"),
    ("The capital city of Italy is",    "Italy",    "capitals"),
    ("The official language of Brazil is",  "Brazil",   "languages"),
    ("The official language of Egypt is",   "Egypt",    "languages"),
    ("The official language of China is",   "China",    "languages"),
    # Hypernyms — entity = specific instance
    ("A poodle is a type of",  "poodle",  "hypernyms"),
    ("A rose is a type of",    "rose",    "hypernyms"),
    ("An eagle is a type of",  "eagle",   "hypernyms"),
    # Antonyms — entity = the word being opposed
    ("The opposite of hot is",   "hot",   "antonyms"),
    ("The opposite of large is", "large", "antonyms"),
    ("The opposite of dark is",  "dark",  "antonyms"),
    ("The opposite of young is", "young", "antonyms"),
    # Tense — entity = verb root / context word
    ("Yesterday he",   "Yesterday",  "tense"),
    ("Yesterday she",  "Yesterday",  "tense"),
    ("Yesterday they", "Yesterday",  "tense"),
    # Gender — entity = the gender word
    ("The king and",    "king",    "gender"),
    ("The queen and",   "queen",   "gender"),
    ("The father and",  "father",  "gender"),
    # Free form
    ("The sun rises in the",  "sun",   "free"),
    ("The cat sat on the",    "cat",   "free"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size

# Extract embedding matrix W_E directly (layer 0 = embedding layer)
# In Qwen2, the embedding is model.model.embed_tokens
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)  # (V, H)
print(f"  hidden={H}  embed_shape={W_E.shape}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_entity_embedding(prompt, entity_word):
    """Return the raw embedding (L0) of the entity token position."""
    inp = tok(prompt, return_tensors="pt")
    tokens = inp["input_ids"][0].tolist()
    decoded = [tok.decode([t]).strip() for t in tokens]
    # Find entity position
    for i, d in enumerate(decoded):
        if d.lower() == entity_word.lower(): return W_E[tokens[i]], i
        if entity_word.lower() in d.lower(): return W_E[tokens[i]], i
    # fallback: entity's own token id embedding
    eid = get_token_id(entity_word)
    if eid: return W_E[eid], -1
    return W_E[tokens[-1]], len(tokens)-1

def get_last_embedding(prompt):
    """Return the raw embedding (L0) of the last token."""
    inp = tok(prompt, return_tensors="pt")
    last_id = inp["input_ids"][0, -1].item()
    return W_E[last_id]

def get_logprob_vocab(prompt, vocab_tok_ids):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    return {w: float(lp[vocab_tok_ids[w]]) for w in vocab_tok_ids}

# Build vocab
print("Building vocab (single-token words) ...")
seen = set(); VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = {}; vocab_tok_id = {}
for w in VOCAB:
    tid = get_token_id(w)
    if tid is not None:
        vocab_ok[w] = W_E[tid]  # raw embedding
        vocab_tok_id[w] = tid
N_VOCAB = len(vocab_ok)
print(f"  {N_VOCAB} single-token vocab words\n")

def rank_by_embedding(query_emb, words):
    scores = {w: cosine(query_emb, vocab_ok[w]) for w in words}
    return sorted(words, key=lambda w: -scores[w]), scores

print("="*72)
print("Day 138: L0 Entity-Position Free-Form Generation")
print("="*72)
print()

all_results = []
for prompt, entity_word, cat in TEST_CASES:
    entity_emb, entity_pos = get_entity_embedding(prompt, entity_word)
    last_emb   = get_last_embedding(prompt)
    entity_tid = get_token_id(entity_word)

    all_words = list(vocab_ok.keys())

    # Method A: cosine(entity_L0, word_L0) — entity embedding vs word embeddings
    ranked_entity, scores_entity = rank_by_embedding(entity_emb, all_words)

    # Method B: cosine(last_L0, word_L0) — last-token embedding vs word embeddings
    ranked_last, scores_last = rank_by_embedding(last_emb, all_words)

    # Method C: direct embedding lookup cos(W_E[entity], W_E[word]) — same as A
    # (already computed above)

    # Oracle: LM log-prob
    lp = get_logprob_vocab(prompt, vocab_tok_id)
    oracle_ranked = sorted(all_words, key=lambda w: -lp[w])
    oracle_top1   = oracle_ranked[0]

    entity_top1 = ranked_entity[0]
    last_top1   = ranked_last[0]

    # Rank of oracle top-1 in each method
    rank_entity = next((i+1 for i,w in enumerate(ranked_entity) if w==oracle_top1), N_VOCAB+1)
    rank_last   = next((i+1 for i,w in enumerate(ranked_last)   if w==oracle_top1), N_VOCAB+1)

    ov10_entity = len(set(ranked_entity[:10]) & set(oracle_ranked[:10]))
    ov10_last   = len(set(ranked_last[:10])   & set(oracle_ranked[:10]))

    agree_entity = entity_top1 == oracle_top1
    agree_last   = last_top1   == oracle_top1

    all_results.append({
        "prompt": prompt, "entity": entity_word, "cat": cat,
        "oracle_top1": oracle_top1, "oracle_top5": oracle_ranked[:5],
        "entity_top1": entity_top1, "entity_top5": ranked_entity[:5],
        "last_top1":   last_top1,   "last_top5":   ranked_last[:5],
        "rank_entity": rank_entity, "rank_last": rank_last,
        "ov10_entity": ov10_entity, "ov10_last": ov10_last,
        "agree_entity": agree_entity, "agree_last": agree_last,
        "entity_pos": entity_pos,
    })

    print(f"  [{cat:>10}|{entity_word:>12}] {prompt!r}")
    print(f"    oracle={oracle_top1}  oracle_top5={oracle_ranked[:5]}")
    print(f"    entity: top1={entity_top1:>12}  rank={rank_entity:>3}  ov@10={ov10_entity}  {'✓' if agree_entity else ''}")
    print(f"    last:   top1={last_top1:>12}  rank={rank_last:>3}  ov@10={ov10_last}  {'✓' if agree_last else ''}")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*72)
print("Summary — Day 138")
print("="*72)
n = len(all_results)
exp10 = 10*10/N_VOCAB

n_agree_entity = sum(1 for r in all_results if r["agree_entity"])
n_agree_last   = sum(1 for r in all_results if r["agree_last"])
mean_rank_entity = float(np.mean([r["rank_entity"] for r in all_results]))
mean_rank_last   = float(np.mean([r["rank_last"]   for r in all_results]))
mean_ov10_entity = float(np.mean([r["ov10_entity"] for r in all_results]))
mean_ov10_last   = float(np.mean([r["ov10_last"]   for r in all_results]))

print(f"""
  Vocab: {N_VOCAB} single-token words
  Random baseline: top1={1/N_VOCAB:.4f}  overlap@10={exp10:.2f}

  Method       top1_agree     mean_rank   overlap@10   ratio
  ─────────────────────────────────────────────────────────────
  entity_L0    {n_agree_entity}/{n} ({n_agree_entity/n:.3f})  {mean_rank_entity:>9.1f}   {mean_ov10_entity:>9.2f}   {mean_ov10_entity/exp10:.2f}x
  last_L0      {n_agree_last}/{n} ({n_agree_last/n:.3f})  {mean_rank_last:>9.1f}   {mean_ov10_last:>9.2f}   {mean_ov10_last/exp10:.2f}x
""")

print("  Per-category breakdown (entity_L0 vs last_L0):")
for cat in sorted(set(r["cat"] for r in all_results)):
    cat_r = [r for r in all_results if r["cat"] == cat]
    ae = sum(r["agree_entity"] for r in cat_r) / len(cat_r)
    al = sum(r["agree_last"]   for r in cat_r) / len(cat_r)
    re = float(np.mean([r["rank_entity"] for r in cat_r]))
    oe = float(np.mean([r["ov10_entity"] for r in cat_r]))
    print(f"    {cat:>12}: entity_agree={ae:.3f}  last_agree={al:.3f}  "
          f"entity_rank={re:.1f}  entity_ov@10={oe:.2f}  "
          f"{'entity wins ✓' if ae > al or oe > float(np.mean([r['ov10_last'] for r in cat_r])) else 'last wins'}")

print()
if n_agree_entity > 0:
    print(f"  HITS (entity_L0 == oracle):")
    for r in all_results:
        if r["agree_entity"]:
            print(f"    [{r['cat']}] {r['prompt']!r} → {r['entity_top1']} ✓")

print()
if n_agree_entity / n > 0.3:
    print("  VERDICT: Entity L0 achieves >30% top-1 agreement — STRONG")
elif n_agree_entity / n > 0.1:
    print("  VERDICT: Entity L0 achieves 10-30% top-1 agreement — MODERATE")
elif n_agree_entity > 0:
    print("  VERDICT: Entity L0 achieves some hits — WEAK but non-zero")
else:
    print("  VERDICT: Entity L0 achieves 0% top-1 — embedding similarity insufficient")

if mean_ov10_entity > 2*exp10:
    print(f"  SIGNAL: overlap@10 = {mean_ov10_entity:.2f} = {mean_ov10_entity/exp10:.1f}x random — strong signal")
elif mean_ov10_entity > exp10:
    print(f"  SIGNAL: overlap@10 = {mean_ov10_entity:.2f} = {mean_ov10_entity/exp10:.1f}x random — above random")
else:
    print(f"  SIGNAL: overlap@10 = {mean_ov10_entity:.2f} = {mean_ov10_entity/exp10:.1f}x random — below/at random")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "n_agree_entity": n_agree_entity,
        "n_agree_last": n_agree_last,
        "n_total": n,
        "top1_entity": n_agree_entity/n,
        "top1_last": n_agree_last/n,
        "mean_rank_entity": mean_rank_entity,
        "mean_rank_last": mean_rank_last,
        "mean_ov10_entity": mean_ov10_entity,
        "mean_ov10_last": mean_ov10_last,
        "vocab_size": N_VOCAB,
        "random_top1": 1/N_VOCAB,
        "random_ov10": exp10,
    }, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 138 complete.")
