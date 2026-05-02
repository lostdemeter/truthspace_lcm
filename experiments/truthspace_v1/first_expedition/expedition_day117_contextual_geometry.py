#!/usr/bin/env python3
"""
Day 117 — Contextual vs Isolated Token Geometry

Day 116 found: d_k does NOT fire on isolated proper nouns (Cohen's d=0.25).
Hypothesis: d_k is a CONTEXT-DEPENDENT direction — it fires only when the
LM processes entity-retrieval prompts.

EXPERIMENT: Compare hidden states of the SAME token in different contexts.

Context types:
  A. Isolated:    " France"
  B. Entity pos:  "The capital of France is ___"  [at "France" position]
  C. Last token:  "The capital of France is ___"  [at "is" — last token]
  D. Neutral ctx: "France is a large country in western Europe, and"
                  [at "France" position]
  E. Query last:  "The capital of France is"  [at last token "is"]

Measurements per context × token:
  1. d_k projection (entity selector fires?)
  2. T2 coordinate (semantic address changes?)
  3. Cosine similarity to paired token
     (France/Paris closer in context A vs C?)

PREDICTIONS (two-structure theory):
  - d_k projection: C ("is" in retrieval) >> A (isolated) ≈ D (neutral)
  - T2 coordinate: stable across contexts (intrinsic property)
  - Entity pair cosim: B+C context brings France/Paris closer in d_k space
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day117_contextual_geometry.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The child cries very loudly","The mother comes running in"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The dog is hungry now","Is the dog hungry now"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
    ],
}

# Test entities (country → capital pairs, from Finding 40)
TEST_ENTITIES = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("China", "Beijing"),
    ("Russia", "Moscow"),
    ("Canada", "Ottawa"),
]

# Context templates for extracting hidden states
# Each returns (prompt, token_to_measure, position_type)
def make_contexts(country, capital):
    return [
        # A: isolated country
        (f" {country}", country, "isolated_country"),
        # B: country inside factual prompt (at country's position)
        (f"The capital of {country} is", country, "entity_in_query"),
        # C: last token of factual query
        (f"The capital of {country} is", "is", "query_last_token"),
        # D: neutral context (country mentioned as subject)
        (f"{country} is a country in the world and", country, "neutral_ctx"),
        # E: isolated capital
        (f" {capital}", capital, "isolated_capital"),
        # F: capital in sentence context
        (f"The capital city is {capital} and", capital, "capital_in_ctx"),
    ]

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_heads     = model.config.num_attention_heads
n_kv_heads  = model.config.num_key_value_heads
head_dim    = hidden_size // n_heads
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
L_CANON     = 28  # primary layer for d_k (L23 output ≈ L28 input)
print(f"  hidden={hidden_size}\n")

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            h2 = o2.hidden_states[L][0, -1, :].numpy().astype(np.float32)
            d  = h2 - h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d / n)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)
t2_matrix = np.stack([t2_axes[ax] for ax in AXIS_NAMES_12], axis=0)

print("Computing d_k (H6 L23) ...")
L23    = model.model.layers[22]
W_k_L = L23.self_attn.k_proj.weight.data.float().numpy()
kv_g   = n_heads // n_kv_heads
kvi    = 6 // kv_g
h6k    = W_k_L[kvi*head_dim : (kvi+1)*head_dim, :]
Uk,_,_ = np.linalg.svd(h6k, full_matrices=False)
d_k    = (h6k.T @ Uk[:, 0]).astype(np.float32)
d_k   /= np.linalg.norm(d_k)
print()

def get_h_at_token(prompt, target_word, layer):
    """Get hidden state at the position of target_word in prompt at given layer."""
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    tokens = tok.convert_ids_to_tokens(inp["input_ids"][0])
    # Find last occurrence of target_word subword
    target_enc = tok.encode(" " + target_word.strip(), add_special_tokens=False)
    target_first = tok.encode(target_word.strip(), add_special_tokens=False)
    input_ids = inp["input_ids"][0].tolist()
    # Find where target starts (try both with and without space)
    pos = -1
    for i in range(len(input_ids)):
        if input_ids[i:i+len(target_enc)] == target_enc:
            pos = i + len(target_enc) - 1  # last subword token
        if input_ids[i:i+len(target_first)] == target_first:
            if pos == -1:
                pos = i + len(target_first) - 1
    if pos == -1:
        pos = inp["input_ids"].shape[1] - 1  # fallback to last token
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32), pos

def normed(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def t2_proj_12(h):
    hn = normed(h)
    return np.array([float(np.dot(hn, t2_axes[ax])) for ax in AXIS_NAMES_12], dtype=np.float32)

print("Running contextual geometry experiments ...")
print("=" * 72)
print("Per-entity: d_k and T2 projections across context types")
print("=" * 72)

all_results = {}
context_types = ["isolated_country", "entity_in_query", "query_last_token",
                 "neutral_ctx", "isolated_capital", "capital_in_ctx"]

# Collect d_k and T2 for all contexts × entities
ctx_dk   = {ct: [] for ct in context_types}
ctx_t2   = {ct: [] for ct in context_types}

for country, capital in TEST_ENTITIES:
    contexts = make_contexts(country, capital)
    entity_result = {}
    for prompt, token, ctx_type in contexts:
        try:
            h, pos = get_h_at_token(prompt, token, 23)  # L23 for d_k
            dk_val = float(abs(np.dot(normed(h), d_k)))

            # For T2: use the correct layer per axis
            # Use L28 approximation for speed (single forward pass at L23)
            # Re-run to get L-specific hidden states for T2
            inp = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            # Find position again
            input_ids = inp["input_ids"][0].tolist()
            target_enc  = tok.encode(" " + token.strip(), add_special_tokens=False)
            target_first = tok.encode(token.strip(), add_special_tokens=False)
            tok_pos = -1
            for i in range(len(input_ids)):
                if input_ids[i:i+len(target_enc)] == target_enc:
                    tok_pos = i + len(target_enc) - 1
                if input_ids[i:i+len(target_first)] == target_first:
                    if tok_pos == -1:
                        tok_pos = i + len(target_first) - 1
            if tok_pos == -1: tok_pos = len(input_ids) - 1

            # T2: per-axis correct layer
            t2_coord = np.zeros(12, dtype=np.float32)
            for k, ax_name in enumerate(AXIS_NAMES_12):
                L_ax = DAY78_LAYERS[ax_name]
                h_ax = out.hidden_states[L_ax][0, tok_pos, :].numpy().astype(np.float32)
                t2_coord[k] = float(np.dot(normed(h_ax), t2_axes[ax_name]))
            t2_mag = float(np.linalg.norm(t2_coord))

            entity_result[ctx_type] = {"dk": dk_val, "t2_mag": t2_mag,
                                        "t2_coord": t2_coord.tolist(), "pos": tok_pos}
            ctx_dk[ctx_type].append(dk_val)
            ctx_t2[ctx_type].append(t2_mag)
        except Exception as e:
            entity_result[ctx_type] = {"error": str(e)}

    all_results[f"{country}/{capital}"] = entity_result
    iso = entity_result.get("isolated_country", {})
    qlt = entity_result.get("query_last_token", {})
    eiq = entity_result.get("entity_in_query", {})
    ntx = entity_result.get("neutral_ctx", {})
    print(f"  {country:>10}/{capital:<12}  "
          f"isolated_dk={iso.get('dk',0):.4f}  query_last_dk={qlt.get('dk',0):.4f}  "
          f"entity_in_q_dk={eiq.get('dk',0):.4f}  neutral_dk={ntx.get('dk',0):.4f}")

# ── Exp 1: d_k by context type ────────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 1: d_k Mean by Context Type (is entity selector context-dependent?)")
print("=" * 72)
print(f"\n  {'context_type':>22}  {'dk_mean':>9}  {'dk_std':>9}  "
      f"{'vs_isolated':>12}  {'verdict':>15}")
print(f"  {'-'*70}")

iso_dk_mean = float(np.mean(ctx_dk["isolated_country"])) if ctx_dk["isolated_country"] else 0
for ct in context_types:
    vals = ctx_dk[ct]
    if not vals: continue
    m = float(np.mean(vals)); s = float(np.std(vals))
    delta = m - iso_dk_mean
    verdict = ("HIGHER (+)" if delta > 0.003 else "LOWER (-)" if delta < -0.003 else "≈ SAME")
    print(f"  {ct:>22}  {m:>9.4f}  {s:>9.4f}  {delta:>+12.4f}  {verdict:>15}")

# ── Exp 2: T2 stability across contexts ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: T2 Magnitude Stability Across Contexts")
print("=" * 72)
print(f"\n  {'context_type':>22}  {'t2_mean':>9}  {'t2_std':>9}  {'vs_isolated':>12}")
print(f"  {'-'*55}")

iso_t2_mean = float(np.mean(ctx_t2["isolated_country"])) if ctx_t2["isolated_country"] else 0
for ct in context_types:
    vals = ctx_t2[ct]
    if not vals: continue
    m = float(np.mean(vals)); s = float(np.std(vals))
    delta = m - iso_t2_mean
    print(f"  {ct:>22}  {m:>9.4f}  {s:>9.4f}  {delta:>+12.4f}")

# ── Exp 3: Entity pair cosim by context ──────────────────────────────────────
print()
print("=" * 72)
print("Exp 3: Country/Capital T2 Cosine Similarity by Context")
print("=" * 72)
print(f"\n  {'pair':>15}  {'iso_country':>12}  {'iso_capital':>12}  "
      f"{'iso_cosim':>10}  {'query_cosim':>11}")
print(f"  {'-'*65}")

cosim_results = {}
for country, capital in TEST_ENTITIES:
    key = f"{country}/{capital}"
    if key not in all_results: continue
    er = all_results[key]
    ic  = er.get("isolated_country", {})
    icap = er.get("isolated_capital", {})
    qlt = er.get("query_last_token", {})
    eiq = er.get("entity_in_query", {})
    if "t2_coord" not in ic or "t2_coord" not in icap: continue
    t2_ic   = np.array(ic["t2_coord"])
    t2_icap = np.array(icap["t2_coord"])
    cos_iso = float(np.dot(t2_ic, t2_icap) / (np.linalg.norm(t2_ic)*np.linalg.norm(t2_icap)+1e-8))
    # query context: country at entity position vs isolated capital
    if "t2_coord" in eiq:
        t2_eiq = np.array(eiq["t2_coord"])
        cos_q   = float(np.dot(t2_eiq, t2_icap) / (np.linalg.norm(t2_eiq)*np.linalg.norm(t2_icap)+1e-8))
    else:
        cos_q = float("nan")
    cosim_results[key] = {"iso_cosim": cos_iso, "query_cosim": cos_q}
    print(f"  {key:>15}  {ic.get('dk',0):>12.4f}  {icap.get('dk',0):>12.4f}  "
          f"{cos_iso:>10.4f}  {cos_q:>11.4f}")

# ── Exp 4: T2 coordinate drift from isolated to contextual ───────────────────
print()
print("=" * 72)
print("Exp 4: T2 Coordinate Drift — How Much Does Context Change the Address?")
print("=" * 72)
print(f"\n  {'pair':>15}  {'iso→entity_cosim':>18}  {'iso→neutral_cosim':>18}  "
      f"{'iso→query_last':>15}")
print(f"  {'-'*70}")

drift_results = {}
for country, capital in TEST_ENTITIES:
    key = f"{country}/{capital}"
    if key not in all_results: continue
    er = all_results[key]
    ic  = er.get("isolated_country", {})
    eiq = er.get("entity_in_query", {})
    ntx = er.get("neutral_ctx", {})
    qlt = er.get("query_last_token", {})
    if "t2_coord" not in ic: continue
    t2_base = np.array(ic["t2_coord"])
    def cosim_to_base(other):
        if "t2_coord" not in other: return float("nan")
        v = np.array(other["t2_coord"])
        return float(np.dot(t2_base, v) / (np.linalg.norm(t2_base)*np.linalg.norm(v)+1e-8))
    c_eiq = cosim_to_base(eiq)
    c_ntx = cosim_to_base(ntx)
    c_qlt = cosim_to_base(qlt)
    drift_results[key] = {"entity_drift": 1-c_eiq, "neutral_drift": 1-c_ntx,
                          "query_drift": 1-c_qlt}
    print(f"  {key:>15}  {c_eiq:>18.4f}  {c_ntx:>18.4f}  {c_qlt:>15.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 117 Summary — Contextual vs Isolated Token Geometry")
print("=" * 72)

valid_ct = {ct: ctx_dk[ct] for ct in context_types if ctx_dk[ct]}
best_dk_ct  = max(valid_ct, key=lambda ct: np.mean(valid_ct[ct]))
best_dk_val = float(np.mean(valid_ct[best_dk_ct]))
iso_mean    = float(np.mean(ctx_dk["isolated_country"])) if ctx_dk["isolated_country"] else 0
query_mean  = float(np.mean(ctx_dk["query_last_token"])) if ctx_dk["query_last_token"] else 0

mean_iso_cosim   = float(np.mean([v["iso_cosim"]   for v in cosim_results.values() if not math.isnan(v["iso_cosim"])]))
mean_query_cosim = float(np.mean([v["query_cosim"] for v in cosim_results.values() if not math.isnan(v.get("query_cosim",float("nan")))]))

print(f"""
  d_k projections by context:
    isolated_country:  {iso_mean:.4f}
    query_last_token:  {query_mean:.4f}
    Best context:      {best_dk_ct} ({best_dk_val:.4f})
    Delta (best-iso):  {best_dk_val - iso_mean:+.4f}

  T2 country/capital cosim:
    Isolated context:  {mean_iso_cosim:.4f}
    Entity-in-query:   {mean_query_cosim:.4f}

  VERDICT:
  {'→ d_k IS context-dependent: query_last_token >> isolated (CONFIRMED)' if query_mean > iso_mean + 0.003 else
   '→ d_k NOT strongly context-dependent in this test (delta < 0.003)'}

  {'→ T2 cosim stable across contexts (intrinsic property CONFIRMED)' if abs(mean_query_cosim - mean_iso_cosim) < 0.05 else
   '→ T2 cosim changes in context (context modulates T2 coordinate)'}

  KEY FINDING:
  The d_k entity selector {'activates more strongly' if query_mean > iso_mean + 0.003 else 'does not clearly activate more'} in retrieval context
  versus isolated token context {'(' + f'delta={query_mean-iso_mean:+.4f})' if True else ''}.
  {'This validates DC 331 revised: d_k is a query-time pointer.' if query_mean > iso_mean + 0.003 else 'Further investigation needed.'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "ctx_dk_means": {ct: float(np.mean(v)) for ct, v in valid_ct.items()},
        "ctx_t2_means": {ct: float(np.mean(v)) for ct, v in ctx_t2.items() if v},
        "cosim_results": cosim_results,
        "drift_results": drift_results,
        "iso_dk_mean": iso_mean,
        "query_last_dk_mean": query_mean,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 117 complete.")
