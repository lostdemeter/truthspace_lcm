#!/usr/bin/env python3
"""
Day 137 — Entity-Position Factual Probe

DC 338 identified the root cause of L25's free-form failure:
  "is" token bias — prompts ending in "is" have a generic L25 state
  that clusters near tokens like 'Canberra','Oslo','Portuguese'

HYPOTHESIS: The entity token's hidden state (e.g. "France" in "The capital
city of France is") contains factual information at L22/L25.

From Finding 154: entity identity at pos 3 (swap → different answer).
From Finding 117: L22 is the "BIG jump" layer where entity gets read.

TEST:
  For each prompt, extract the hidden state at the ENTITY POSITION
  (not the last "is" position) at layers L0, L10, L15, L20, L22, L25, L27.

  Use h(entity_pos, L) to rank candidates:
    cosine(h_ctx_entity, h_candidate_last) vs log-prob oracle
    cosine(h_ctx_entity, h_candidate_last) vs cosine(h_last, h_candidate)

  PROMPTS: factual (capitals, hypernyms, languages)
  Also test: relational (antonyms) — entity position is the antonym word

KEY QUESTIONS:
  1. Does the entity position's HS predict the correct candidate?
  2. Which layer is best: L22 (entity read) or L25 (used in Days 124-132)?
  3. Does this explain why Days 124-132 worked: correct candidate is at
     correct entity-type semantic location?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day137_entity_position_probe.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PROBE_LAYERS = [0, 10, 15, 20, 22, 25, 27]

# Each case: (prompt, entity_word, correct_candidates, wrong_candidates)
# entity_word = the semantic entity in the prompt
CASES = [
    # Capitals — entity = country name
    ("The capital city of France is",  "France",
     ["Paris"],  ["London","Rome","Berlin","Madrid"]),
    ("The capital city of Japan is",   "Japan",
     ["Tokyo"],  ["Osaka","Beijing","Seoul","Bangkok"]),
    ("The capital city of Germany is", "Germany",
     ["Berlin"], ["Frankfurt","Vienna","Warsaw","Amsterdam"]),
    ("The capital city of Spain is",   "Spain",
     ["Madrid"], ["Lisbon","Rome","Paris","Brussels"]),
    ("The capital city of Italy is",   "Italy",
     ["Rome"],   ["Milan","Paris","Vienna","Athens"]),
    # Languages — entity = country name
    ("The official language of Brazil is",  "Brazil",
     ["Portuguese"], ["Spanish","Italian","French","English"]),
    ("The official language of Egypt is",   "Egypt",
     ["Arabic"],     ["Hebrew","Turkish","Persian","Urdu"]),
    ("The official language of China is",   "China",
     ["Mandarin"],   ["Japanese","Korean","Cantonese","Thai"]),
    # Hypernyms — entity = specific instance
    ("A poodle is a type of", "poodle",
     ["dog","animal"], ["cat","rabbit","horse","flower"]),
    ("A rose is a type of",   "rose",
     ["flower","plant"], ["tree","bush","grass","metal"]),
    ("An eagle is a type of", "eagle",
     ["bird","animal"], ["insect","reptile","fish","flower"]),
    # Antonyms — entity = the word being negated
    ("The opposite of hot is",   "hot",
     ["cold","cool"],  ["warm","lukewarm","mild","Paris"]),
    ("The opposite of large is", "large",
     ["small","tiny"], ["big","huge","enormous","Paris"]),
    ("The opposite of dark is",  "dark",
     ["light","bright"],["sunny","clear","warm","Paris"]),
    # Tense — entity = verb (the action being cast to past tense)
    ("Yesterday he walked",  "walked",
     ["walked"],  ["walks","walk","walking","run"]),
    ("Yesterday she ran",    "ran",
     ["ran"],     ["runs","run","running","walked"]),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
print(f"  hidden={H}\n")

def get_all_hs(text):
    """Return hidden states for ALL positions at all probe layers."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    tokens = tok.convert_ids_to_tokens(inp["input_ids"][0].tolist())
    n_pos = inp["input_ids"].shape[1]
    hs = {}
    for L in PROBE_LAYERS:
        hs[L] = out.hidden_states[L][0, :, :].numpy().astype(np.float32)  # (n_pos, H)
    return tokens, n_pos, hs

def get_last_hs(text):
    """Return hidden states for LAST position at all probe layers."""
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32) for L in PROBE_LAYERS}

def get_logprob(prompt, word):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0, -1, :], dim=-1).numpy()
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return float(lp[ids[0]]) if ids else float("-inf")

def find_entity_pos(tokens, entity_word):
    """Find the position of entity_word in the tokenized prompt."""
    entity_tokens = tok(" "+entity_word, add_special_tokens=False)["input_ids"]
    entity_toks_dec = [tok.decode([i]).strip() for i in entity_tokens]
    decoded = [t.lstrip("▁Ġ ") for t in tokens]
    for i in range(len(decoded)):
        if decoded[i].lower() == entity_word.lower(): return i
        # multi-token match
        if i + len(entity_toks_dec) <= len(decoded):
            if all(decoded[i+j].lower() == entity_toks_dec[j].lower()
                   for j in range(len(entity_toks_dec))):
                return i
    # fallback: find by partial match
    for i, t in enumerate(decoded):
        if entity_word.lower() in t.lower(): return i
    return len(tokens) - 1  # fallback to last

print("="*72)
print("Day 137: Entity-Position Factual Probe")
print("="*72)
print()

all_results = []

for prompt, entity_word, correct, wrong in CASES:
    all_cands = list(correct) + list(wrong)
    tokens, n_pos, ctx_hs_all = get_all_hs(prompt)
    last_hs = {L: ctx_hs_all[L][-1] for L in PROBE_LAYERS}

    # Find entity position
    entity_pos = find_entity_pos(tokens, entity_word)
    entity_hs  = {L: ctx_hs_all[L][entity_pos] for L in PROBE_LAYERS}

    # LM log-prob ranking (oracle)
    lp = {w: get_logprob(prompt, w) for w in all_cands}
    oracle_ranked = sorted(all_cands, key=lambda w: -lp[w])
    oracle_rank1  = oracle_ranked[0]
    oracle_mrr    = 1.0 / next((i+1 for i,w in enumerate(oracle_ranked) if w in correct), len(all_cands)+1)

    # Candidate last-token HS at each layer
    cand_hs = {w: get_last_hs(" "+w) for w in all_cands}

    # Score using entity_pos HS vs last_pos HS per layer
    layer_results = {}
    for L in PROBE_LAYERS:
        h_entity = entity_hs[L]
        h_last   = last_hs[L]
        scores_entity = {w: cosine(h_entity, cand_hs[w][L]) for w in all_cands}
        scores_last   = {w: cosine(h_last,   cand_hs[w][L]) for w in all_cands}

        ranked_entity = sorted(all_cands, key=lambda w: -scores_entity[w])
        ranked_last   = sorted(all_cands, key=lambda w: -scores_last[w])

        def mrr(ranked, correct_set):
            r = next((i+1 for i,w in enumerate(ranked) if w in correct_set), len(ranked)+1)
            return 1.0/r

        layer_results[L] = {
            "entity_mrr": mrr(ranked_entity, set(correct)),
            "last_mrr":   mrr(ranked_last,   set(correct)),
            "entity_top1": ranked_entity[0],
            "last_top1":   ranked_last[0],
        }

    # Best layer for entity probe
    best_L_entity = max(PROBE_LAYERS, key=lambda L: layer_results[L]["entity_mrr"])
    best_L_last   = max(PROBE_LAYERS, key=lambda L: layer_results[L]["last_mrr"])

    row = {
        "prompt": prompt, "entity_word": entity_word,
        "correct": list(correct), "entity_pos": entity_pos,
        "tokens": tokens[:10],
        "oracle_mrr": oracle_mrr, "oracle_top1": oracle_rank1,
        "layer_results": {str(L): layer_results[L] for L in PROBE_LAYERS},
        "best_L_entity": best_L_entity,
        "best_entity_mrr": layer_results[best_L_entity]["entity_mrr"],
        "best_L_last": best_L_last,
        "best_last_mrr":   layer_results[best_L_last]["last_mrr"],
    }
    all_results.append(row)

    e_mrr_25 = layer_results[25]["entity_mrr"]
    l_mrr_25 = layer_results[25]["last_mrr"]
    e_mrr_22 = layer_results[22]["entity_mrr"]
    l_mrr_22 = layer_results[22]["last_mrr"]
    print(f"  [{entity_word:>12}] pos={entity_pos}  oracle={oracle_mrr:.3f}  "
          f"entity_L22={e_mrr_22:.3f}  entity_L25={e_mrr_25:.3f}  "
          f"last_L25={l_mrr_25:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("="*72)
print("Summary — Layer-by-layer MRR comparison")
print("="*72)
print()

n = len(all_results)
oracle_mrr_mean = float(np.mean([r["oracle_mrr"] for r in all_results]))

print(f"  Oracle MRR: {oracle_mrr_mean:.4f}")
print()
print(f"  {'Layer':>6}  {'entity_MRR':>12}  {'last_MRR':>12}  {'entity>last?':>14}")
print(f"  {'-'*52}")
for L in PROBE_LAYERS:
    em = float(np.mean([r["layer_results"][str(L)]["entity_mrr"] for r in all_results]))
    lm = float(np.mean([r["layer_results"][str(L)]["last_mrr"]   for r in all_results]))
    better = "entity ✓" if em > lm else ("tie" if abs(em-lm)<0.005 else "last ✓")
    print(f"  L{L:>2}:   {em:>12.4f}  {lm:>12.4f}  {better:>14}  "
          f"  ({100*em/oracle_mrr_mean:.0f}% oracle  vs  {100*lm/oracle_mrr_mean:.0f}%)")

# Per-category breakdown
print()
print("  Per-category entity_L22 vs last_L25 vs oracle:")
cat_map = {
    "capitals":  ["France","Japan","Germany","Spain","Italy"],
    "languages": ["Brazil","Egypt","China"],
    "hypernyms": ["poodle","rose","eagle"],
    "antonyms":  ["hot","large","dark"],
    "tense":     ["walked","ran"],
}
for cat, entities in cat_map.items():
    cat_r = [r for r in all_results if r["entity_word"] in entities]
    if not cat_r: continue
    em22 = float(np.mean([r["layer_results"]["22"]["entity_mrr"] for r in cat_r]))
    em25 = float(np.mean([r["layer_results"]["25"]["entity_mrr"] for r in cat_r]))
    lm25 = float(np.mean([r["layer_results"]["25"]["last_mrr"]   for r in cat_r]))
    om   = float(np.mean([r["oracle_mrr"] for r in cat_r]))
    print(f"    {cat:>12}: entity_L22={em22:.3f}  entity_L25={em25:.3f}  "
          f"last_L25={lm25:.3f}  oracle={om:.3f}  "
          f"{'entity wins ✓' if em25 > lm25 else 'last wins'}")

# Best entity layer
best_entity_L = max(PROBE_LAYERS, key=lambda L:
    float(np.mean([r["layer_results"][str(L)]["entity_mrr"] for r in all_results])))
best_entity_mrr = float(np.mean([r["layer_results"][str(best_entity_L)]["entity_mrr"] for r in all_results]))
print()
print(f"  Best entity layer: L{best_entity_L} (MRR={best_entity_mrr:.4f}  "
      f"= {100*best_entity_mrr/oracle_mrr_mean:.0f}% oracle)")

print()
if best_entity_mrr > float(np.mean([r["layer_results"]["25"]["last_mrr"] for r in all_results])):
    print("  VERDICT: Entity position HS BEATS last-token HS for factual ranking!")
    print("  → The 'is' bias in last-token is real and bypassing it helps.")
    print("  → Factual knowledge encoded in entity token's evolution through layers.")
else:
    print("  VERDICT: Entity position HS does NOT beat last-token HS.")
    print("  → The factual signal requires the full prompt context propagation.")
    print("  → Entity position alone doesn't encode the association.")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 137 complete.")
