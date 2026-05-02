#!/usr/bin/env python3
"""
Day 119 — What Primes d_k? Characterizing the Retrieval-Priming Context

Day 117 found:
  - Neutral context "France is a country" → d_k stays 0.0167 (dormant)
  - Retrieval context "The capital of France is" → d_k jumps to 0.0713 (4.3x)

QUESTION: Is d_k activation specific to capital-retrieval prompts,
or does it activate for ANY factual retrieval?

EXPERIMENT: Test d_k projection for many prompt types:
  Group 1: Capital retrieval (the known trigger)
    "The capital of X is"
  Group 2: Other fact types (general retrieval test)
    "The official language of X is"
    "The population of X is approximately"
    "X is located in"
    "The president of X is"
    "The currency used in X is"
  Group 3: Non-retrieval structures (should stay dormant)
    "X is a country in"
    "People in X often"
    "The culture of X is known for"
    "X was founded in"
  Group 4: Question forms (different syntactic structure)
    "What is the capital of X?"
    "Where is X located?"
  Group 5: Completion with entity as LAST token (entity primed position)
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of X is"
    (few-shot format — strong retrieval priming)

PREDICTION:
  - If d_k is GENERAL retrieval: all Group 2 prompts should activate it
  - If d_k is SPECIFIC to capital: only Group 1 activates it
  - Group 3 should stay dormant (controls)
  - Group 4 (question form) may or may not activate
  - Group 5 (few-shot) should show HIGHEST activation (strongest retrieval cue)

Also measure: which TOKEN POSITION has highest d_k?
  - The entity position ("France")
  - The last token ("is")
  - Both? Neither?

This characterizes whether there's a THIRD geometric structure:
  Structure 3: Retrieval-priming context (geometric signature of "I need to retrieve a fact")
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day119_dk_priming.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ENTITIES = [
    ("France",    "Paris",   "French",    "euro"),
    ("Germany",   "Berlin",  "German",    "euro"),
    ("Japan",     "Tokyo",   "Japanese",  "yen"),
    ("Italy",     "Rome",    "Italian",   "euro"),
    ("Spain",     "Madrid",  "Spanish",   "euro"),
    ("China",     "Beijing", "Mandarin",  "yuan"),
    ("Russia",    "Moscow",  "Russian",   "ruble"),
    ("Canada",    "Ottawa",  "English",   "dollar"),
]

# Prompt templates: (name, template_with_{entity}_placeholder, target_word)
# target_word = what the answer would be (used for context, not measured)
PROMPT_GROUPS = {
    "capital_retrieval": [
        ("capital_std",      "The capital of {entity} is",         "city"),
        ("capital_long",     "The capital city of {entity} is",    "city"),
        ("capital_question", "What is the capital of {entity}?",   "city"),
    ],
    "other_fact_retrieval": [
        ("language",         "The official language of {entity} is",           "lang"),
        ("location",         "{entity} is located in",                          "region"),
        ("currency",         "The currency used in {entity} is",               "cur"),
        ("president",        "The current president of {entity} is",           "person"),
        ("population",       "The population of {entity} is approximately",    "num"),
    ],
    "non_retrieval": [
        ("neutral_country",  "{entity} is a country in",                       None),
        ("culture",          "The culture of {entity} is known for",           None),
        ("history",          "{entity} was founded in",                        None),
        ("people",           "People in {entity} often",                       None),
        ("geography",        "{entity} has many",                              None),
    ],
    "few_shot": [
        ("few_shot_2",
         "The capital of France is Paris. The capital of Germany is Berlin. The capital of {entity} is",
         "city"),
        ("few_shot_1",
         "The capital of France is Paris. The capital of {entity} is",
         "city"),
        ("analogy_form",
         "France: Paris, Germany: Berlin, {entity}:",
         "city"),
    ],
    "isolated": [
        ("isolated",         " {entity}",                                      None),
    ],
}

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_heads     = model.config.num_attention_heads
n_kv_heads  = model.config.num_key_value_heads
head_dim    = hidden_size // n_heads
print(f"  hidden={hidden_size}\n")

print("Computing d_k (H6 L23) ...")
L23    = model.model.layers[22]
W_k_L  = L23.self_attn.k_proj.weight.data.float().numpy()
kv_g   = n_heads // n_kv_heads
kvi    = 6 // kv_g
h6k    = W_k_L[kvi*head_dim : (kvi+1)*head_dim, :]
Uk,_,_ = np.linalg.svd(h6k, full_matrices=False)
d_k    = (h6k.T @ Uk[:, 0]).astype(np.float32)
d_k   /= np.linalg.norm(d_k)

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_all_hidden(prompt, layer=23):
    """Returns hidden states at given layer for ALL positions."""
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer][0].numpy().astype(np.float32), inp["input_ids"][0].tolist()

def find_entity_pos(input_ids, entity):
    """Find the last subword position of entity in input_ids."""
    enc_with_space = tok.encode(" " + entity.strip(), add_special_tokens=False)
    enc_bare       = tok.encode(entity.strip(), add_special_tokens=False)
    ids = input_ids
    pos = -1
    for i in range(len(ids)):
        if ids[i:i+len(enc_with_space)] == enc_with_space:
            pos = i + len(enc_with_space) - 1
        elif ids[i:i+len(enc_bare)] == enc_bare:
            if pos == -1:
                pos = i + len(enc_bare) - 1
    return pos

def dk_at_pos(hs, pos):
    """d_k projection at a given token position."""
    return float(abs(np.dot(normed(hs[pos]), d_k)))

print("Computing d_k activation for all prompt types × entities ...")
print()

# Results structure: {group: {template_name: {entity: {last_dk, entity_dk}}}}
results = {}
group_dk_last   = {}  # {group: {template: [dk_last values]}}
group_dk_entity = {}  # {group: {template: [dk_entity values]}}

for group_name, templates in PROMPT_GROUPS.items():
    results[group_name] = {}
    group_dk_last[group_name]   = {}
    group_dk_entity[group_name] = {}
    for tmpl_name, template, _ in templates:
        results[group_name][tmpl_name]         = {}
        group_dk_last[group_name][tmpl_name]   = []
        group_dk_entity[group_name][tmpl_name] = []
        for entity, capital, language, currency in ENTITIES:
            prompt = template.replace("{entity}", entity)
            try:
                hs, input_ids = get_all_hidden(prompt)
                last_pos    = len(input_ids) - 1
                entity_pos  = find_entity_pos(input_ids, entity)
                dk_last     = dk_at_pos(hs, last_pos)
                dk_entity   = dk_at_pos(hs, entity_pos) if entity_pos >= 0 else float("nan")
                results[group_name][tmpl_name][entity] = {
                    "dk_last": dk_last, "dk_entity": dk_entity,
                    "entity_pos": entity_pos, "seq_len": len(input_ids),
                }
                group_dk_last[group_name][tmpl_name].append(dk_last)
                if entity_pos >= 0:
                    group_dk_entity[group_name][tmpl_name].append(dk_entity)
            except Exception as e:
                results[group_name][tmpl_name][entity] = {"error": str(e)}

# ── Exp 1: d_k at last token by prompt group/template ────────────────────────
print("=" * 72)
print("Exp 1: d_k at LAST TOKEN by Prompt Group and Template")
print("=" * 72)
print(f"\n  {'group':>20}  {'template':>22}  {'dk_last_mean':>14}  "
      f"{'dk_entity_mean':>16}  {'vs_isolated':>12}")
print(f"  {'-'*90}")

iso_mean = float(np.mean(group_dk_last["isolated"]["isolated"])) \
           if group_dk_last["isolated"].get("isolated") else 0.0167

group_summary = {}  # {group: {mean_last, mean_entity}}
for group_name in PROMPT_GROUPS:
    g_last_all = []; g_ent_all = []
    for tmpl_name, template, _ in PROMPT_GROUPS[group_name]:
        vals_last   = group_dk_last[group_name].get(tmpl_name, [])
        vals_entity = group_dk_entity[group_name].get(tmpl_name, [])
        m_last   = float(np.mean(vals_last))   if vals_last   else 0.0
        m_entity = float(np.mean(vals_entity)) if vals_entity else 0.0
        g_last_all.extend(vals_last); g_ent_all.extend(vals_entity)
        delta = m_last - iso_mean
        print(f"  {group_name:>20}  {tmpl_name:>22}  {m_last:>14.4f}  "
              f"{m_entity:>16.4f}  {delta:>+12.4f}")
    group_summary[group_name] = {
        "mean_last": float(np.mean(g_last_all)) if g_last_all else 0,
        "mean_entity": float(np.mean(g_ent_all)) if g_ent_all else 0,
    }
    print()

# ── Exp 2: Group-level summary ────────────────────────────────────────────────
print("=" * 72)
print("Exp 2: Group-Level Summary (mean d_k across all templates in group)")
print("=" * 72)
print(f"\n  {'group':>22}  {'dk_last_mean':>14}  {'dk_entity_mean':>16}  "
      f"{'factor_vs_iso':>14}  {'verdict':>15}")
print(f"  {'-'*85}")

for group_name in PROMPT_GROUPS:
    g = group_summary[group_name]
    factor = g["mean_last"] / max(iso_mean, 1e-8)
    if g["mean_last"] > iso_mean + 0.003:
        verdict = "PRIMES d_k"
    elif g["mean_last"] < iso_mean - 0.001:
        verdict = "SUPPRESSES"
    else:
        verdict = "NO CHANGE"
    print(f"  {group_name:>22}  {g['mean_last']:>14.4f}  {g['mean_entity']:>16.4f}  "
          f"{factor:>14.2f}x  {verdict:>15}")

# ── Exp 3: Per-entity breakdown for capital vs language ───────────────────────
print()
print("=" * 72)
print("Exp 3: Capital vs Language retrieval — per entity comparison")
print("=" * 72)
print(f"\n  {'entity':>10}  {'capital_dk':>12}  {'language_dk':>13}  "
      f"{'location_dk':>13}  {'neutral_dk':>12}  {'isolated_dk':>12}")
print(f"  {'-'*75}")

per_entity_results = {}
for entity, capital, language, currency in ENTITIES:
    cap_dk = results.get("capital_retrieval", {}).get("capital_std", {}).get(entity, {}).get("dk_last", float("nan"))
    lan_dk = results.get("other_fact_retrieval", {}).get("language", {}).get(entity, {}).get("dk_last", float("nan"))
    loc_dk = results.get("other_fact_retrieval", {}).get("location", {}).get(entity, {}).get("dk_last", float("nan"))
    neu_dk = results.get("non_retrieval", {}).get("neutral_country", {}).get(entity, {}).get("dk_last", float("nan"))
    iso_dk = results.get("isolated", {}).get("isolated", {}).get(entity, {}).get("dk_last", float("nan"))
    per_entity_results[entity] = {"cap": cap_dk, "lang": lan_dk, "loc": loc_dk,
                                   "neutral": neu_dk, "iso": iso_dk}
    print(f"  {entity:>10}  {cap_dk:>12.4f}  {lan_dk:>13.4f}  "
          f"{loc_dk:>13.4f}  {neu_dk:>12.4f}  {iso_dk:>12.4f}")

# ── Exp 4: Few-shot amplification ─────────────────────────────────────────────
print()
print("=" * 72)
print("Exp 4: Few-Shot Amplification — Does More Context Prime Stronger?")
print("=" * 72)
print(f"\n  {'template':>20}  {'dk_last_mean':>14}  {'factor_vs_iso':>14}")
print(f"  {'-'*55}")

for tmpl_name in ["isolated", "capital_std", "few_shot_1", "few_shot_2", "analogy_form"]:
    group = "isolated" if tmpl_name == "isolated" else \
            "capital_retrieval" if tmpl_name == "capital_std" else "few_shot"
    vals = group_dk_last.get(group, {}).get(tmpl_name, [])
    if not vals: continue
    m = float(np.mean(vals))
    factor = m / max(iso_mean, 1e-8)
    print(f"  {tmpl_name:>20}  {m:>14.4f}  {factor:>14.2f}x")

# ── Exp 5: Is entity position or last position higher? ────────────────────────
print()
print("=" * 72)
print("Exp 5: Entity Position vs Last Token — Which Has Higher d_k?")
print("=" * 72)
print(f"\n  (For capital_std template)")
print(f"\n  {'entity':>10}  {'entity_pos_dk':>15}  {'last_tok_dk':>13}  {'which_higher':>14}")
print(f"  {'-'*55}")

for entity, capital, language, currency in ENTITIES:
    r = results.get("capital_retrieval", {}).get("capital_std", {}).get(entity, {})
    dk_e = r.get("dk_entity", float("nan"))
    dk_l = r.get("dk_last",   float("nan"))
    if math.isnan(dk_e) or math.isnan(dk_l): continue
    which = "entity" if dk_e > dk_l else "last"
    print(f"  {entity:>10}  {dk_e:>15.4f}  {dk_l:>13.4f}  {which:>14}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Day 119 Summary — What Primes d_k?")
print("=" * 72)

cap_mean   = group_summary.get("capital_retrieval", {}).get("mean_last", 0)
other_mean = group_summary.get("other_fact_retrieval", {}).get("mean_last", 0)
neutral_m  = group_summary.get("non_retrieval", {}).get("mean_last", 0)
few_shot_m = group_summary.get("few_shot", {}).get("mean_last", 0)

print(f"""
  d_k activation levels:
    isolated:            {iso_mean:.4f}  (baseline)
    non_retrieval:       {neutral_m:.4f}  ({'primes' if neutral_m > iso_mean+0.003 else 'dormant'})
    other_fact_retr:     {other_mean:.4f}  ({'primes' if other_mean > iso_mean+0.003 else 'does NOT prime'})
    capital_retrieval:   {cap_mean:.4f}  ({'primes' if cap_mean > iso_mean+0.003 else 'does NOT prime'})
    few_shot:            {few_shot_m:.4f}  ({'amplifies further' if few_shot_m > cap_mean+0.003 else 'similar to capital'})

  KEY FINDING:
  {'→ d_k is GENERAL: activates for all fact retrieval types' if other_mean > iso_mean+0.003 else
   '→ d_k is SPECIFIC: activates mainly for capital/known retrieval structures' if cap_mean > iso_mean+0.003 and other_mean < iso_mean+0.003 else
   '→ d_k is context-dependent but not strongly selective between retrieval types'}

  {'→ Few-shot AMPLIFIES: stronger context = stronger d_k activation' if few_shot_m > cap_mean+0.003 else
   '→ Few-shot does not further amplify d_k beyond single-shot retrieval'}

  IMPLICATION:
  {'→ Structure 3: Retrieval-priming is a GENERAL geometric property shared across fact types' if other_mean > iso_mean+0.003 else
   '→ d_k specificity: the entity selector is tuned to specific linguistic patterns (e.g., "capital of X is")' if cap_mean > iso_mean+0.003 else
   '→ Further investigation needed: d_k activation is noisy in these conditions'}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "group_summary": group_summary,
        "iso_mean": iso_mean,
        "per_entity_results": per_entity_results,
        "group_dk_last_means": {
            g: {t: float(np.mean(v)) if v else 0 for t,v in tdict.items()}
            for g, tdict in group_dk_last.items()
        },
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 119 complete.")
