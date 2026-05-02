#!/usr/bin/env python3
"""
Expedition Day 45 — Does the Gate Fire for English Phrases?

Day 44 established:
  - English gerunds (' walking') are B001 as single tokens
  - English phrases ('is walking') reach Zone C(0.515) — same body as 走着(0.484)
  - Rotation angle is ~90° for ALL forms (no torque difference)
  - ONE mechanism: B001 → Zone C via attention, different scope

The decisive question:
  走着 → H01(L23) drops to 0.10 (gate OPEN)
  Is 'is walking' → H01(L23) drops below 0.55 (gate OPEN)?

If YES: the gate is language-agnostic at PHRASE granularity. Day 41 was
        correct in spirit, wrong in scope. The gate fires whenever the last
        token makes a B001→Zone C rotation, regardless of whether that
        happens within a Chinese word or an English phrase.

If NO: the gate is pattern-matching Chinese morphology specifically,
       not detecting the general semantic completion event.

The test is clean because we have matched pairs with comparable sim_C:
  走着  → sim_C=0.484, gate OPEN (H01=0.104)
  'is walking' → sim_C=0.515, gate ???

Additionally test gradient: weaker/stronger English phrases, to see if
there's a sim_C THRESHOLD at which the gate transitions from CLOSED to OPEN.

Architecture: Qwen2-1.5B-Instruct, 28 layers, 12 Q-heads, 2 KV-heads
"""

import os, json
import numpy as np

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day45_phrase_gate.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

GATE_LAYER = 23
GATE_HEAD  = 1    # head index 1 (Day 41's H01)
GATE_THRESHOLD = 0.55

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (text, note, n_tokens_expected, last_pos_is_content_token)
# We measure H01 backward attention from LAST token to FIRST token at L23.
# For each phrase we also check sim_C at L14 for the last token.

# Chinese reference (Day 41 established baselines)
CHINESE_REF = [
    ("走着",  "walk+着 (Day 41: gate OPEN)",    2),
    ("跑着",  "run+着",                         2),
    ("吃着",  "eat+着",                         2),
    ("唱着",  "sing+着 (Day 41: borderline)",   2),
    ("做着",  "make+着",                        2),
    ("走了",  "walk+了 (Day 41: gate CLOSED)",  1),  # should be 1-tok
]

# English phrases that Day 44 showed reach Zone C (sim_C 0.46-0.58)
# Prediction: gate OPEN (same mechanism as Chinese)
ENGLISH_PHRASES_STRONG = [
    ("is walking",       "copula+gerund         Day44: sim_C=0.515"),
    ("was singing",      "past-copula+gerund    Day44: sim_C=0.572"),
    ("keep walking",     "aspectual+gerund      Day44: sim_C=0.581"),
    ("start singing",    "inceptive+gerund      Day44: sim_C=0.553"),
    ("go swimming",      "motion+gerund         Day44: sim_C=0.549"),
    ("her singing",      "possessive+gerund     Day44: sim_C=0.573"),
    ("loves singing",    "mental-verb+gerund    Day44: sim_C=0.488"),
    ("I am walking",     "pronoun+cop+gerund    Day44: sim_C=0.462"),
]

# English phrases at the boundary — just below the Chinese threshold
ENGLISH_PHRASES_WEAK = [
    ("she was singing",  "3-token, sim_C=0.466"),
    ("the singing",      "det+gerund, B000 in Day44"),
    ("their walking",    "possessive+gerund (test)"),
    ("all swimming",     "quantifier+gerund (test)"),
]

# English single-token gerunds — B001 baseline, gate N/A (1-tok)
# Included for sim_C reference
ENGLISH_SINGLE_TOK = [
    (" walking",   "single-tok, B001, sim_C=0.240"),
    (" singing",   "single-tok, B001, sim_C=0.249"),
    (" running",   "single-tok, B001, sim_C=0.242"),
    (" swimming",  "single-tok, B001, sim_C=0.232"),
]

# Longer English phrases — does extending context improve gate probability?
ENGLISH_LONG = [
    ("she is walking",       "3-tok, copula+gerund"),
    ("he was singing",       "3-tok, past-cop+gerund"),
    ("they keep walking",    "3-tok, aspectual"),
    ("she loves singing",    "3-tok, mental"),
    ("I love swimming",      "3-tok, mental"),
    ("we keep walking",      "3-tok, aspectual"),
]

PROBE_LAYERS = [0, 1, 5, 10, 14, 20, 23, 27]

print("=" * 70)
print("  Expedition Day 45 — Does the Gate Fire for English Phrases?")
print("=" * 70)

# ── Build φ-space and zone machinery ─────────────────────────────────────────
print("\n── Load atlas ──────────────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}
with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        if pfx+a in w2i and pfx+b in w2i:
            d = hs14_all[w2i[pfx+b]] - hs14_all[w2i[pfx+a]]
            dm = np.linalg.norm(d)
            if dm > 1e-20:
                deltas.append(d / dm)
            break
_, _, Vt = np.linalg.svd(np.stack(deltas), full_matrices=False)
z2 = Vt[0].astype(np.float64)

def phi_single(h):
    hn   = h.astype(np.float64) / (np.linalg.norm(h) + 1e-20)
    proj = float(hn @ z2)
    perp = hn - proj * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)

def batch_phi(hs):
    H   = hs.astype(np.float64)
    nm  = np.linalg.norm(H, axis=1, keepdims=True)
    Hn  = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

wmap_words = [w for w in wmap.keys() if w in w2i]
wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx])

zone_c_idx = [i for i, w in enumerate(wmap_words)
              if wmap[w]['phase']==2 and wmap[w].get('L14_body') not in ('B000','B001',None)]
b000_idx   = [i for i, w in enumerate(wmap_words)
              if wmap[w]['phase']==2 and wmap[w].get('L14_body') == 'B000']
b001_idx   = [i for i, w in enumerate(wmap_words)
              if wmap[w]['phase']==2 and wmap[w].get('L14_body') == 'B001']
ab_idx     = [i for i, w in enumerate(wmap_words) if wmap[w]['phase'] != 2]

c_C  = wmap_phi[zone_c_idx].mean(0); c_C  /= np.linalg.norm(c_C)
c_B0 = wmap_phi[b000_idx].mean(0);   c_B0 /= np.linalg.norm(c_B0)
c_B1 = wmap_phi[b001_idx].mean(0);   c_B1 /= np.linalg.norm(c_B1)
c_AB = wmap_phi[ab_idx].mean(0);     c_AB /= np.linalg.norm(c_AB)

body_centroids = {}
body_labels    = {}
for i, w in enumerate(wmap_words):
    v = wmap[w]
    if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None):
        bd = v['L14_body']
        body_centroids.setdefault(bd, []).append(wmap_phi[i])
        body_labels[bd] = v.get('L14_label', bd)
body_cvecs = {bd: (lambda c: c/np.linalg.norm(c))(np.stack(v).mean(0))
              for bd, v in body_centroids.items()}

def assign_zone(phi_v):
    sims = {'C': float(phi_v@c_C), 'B000': float(phi_v@c_B0),
            'B001': float(phi_v@c_B1), 'A/B': float(phi_v@c_AB)}
    top_body     = max(body_cvecs, key=lambda bd: float(phi_v@body_cvecs[bd]))
    top_body_sim = float(phi_v @ body_cvecs[top_body])
    top_body_lbl = body_labels[top_body]
    return max(sims, key=sims.get), sims, top_body, top_body_lbl, top_body_sim

print(f"  Zone C={len(zone_c_idx)} B000={len(b000_idx)} B001={len(b001_idx)}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\n── Load model ──────────────────────────────────────────────────────────")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map='cpu',
    output_hidden_states=True, output_attentions=True,
    attn_implementation='eager')
model.eval()
num_heads = model.config.num_attention_heads
print(f"  num_Q_heads={num_heads}")


def run_text(text):
    inputs     = tok(text, return_tensors='pt', add_special_tokens=False)
    ids        = inputs['input_ids'][0]
    token_strs = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    hs   = [out.hidden_states[L][0].numpy().astype(np.float64)
            for L in range(len(out.hidden_states))]
    attn = [out.attentions[L][0].numpy() for L in range(len(out.attentions))]
    return token_strs, hs, attn


def analyse(text):
    """Full analysis: tokenization, sim_C@L14, H01 profile, gate state."""
    token_strs, hs, attn = run_text(text)
    n = len(token_strs)

    # Zone/sim_C for last token at L14
    phi_last = phi_single(hs[14][-1])
    zone, sims, body_id, body_lbl, body_sim = assign_zone(phi_last)

    # H01 backward attention (last→first) at all probe layers
    h01_profile = {}
    if n >= 2:
        for L in PROBE_LAYERS:
            if L < len(attn):
                h01_profile[L] = float(attn[L][GATE_HEAD, n-1, 0])

    h01_l23    = h01_profile.get(GATE_LAYER, None)
    gate_state = ('OPEN' if h01_l23 < GATE_THRESHOLD else 'CLOSED') \
                  if h01_l23 is not None else '1-tok'

    # Also: per-layer sim_C profile for last token
    sim_c_path = {}
    for L in PROBE_LAYERS:
        if L < len(hs):
            phi_v = phi_single(hs[L][-1])
            s = assign_zone(phi_v)[1]
            sim_c_path[L] = round(float(s['C']), 4)

    return {
        'text': text, 'tokens': token_strs, 'n': n,
        'zone': zone, 'sim_C': round(float(sims['C']), 4),
        'body': body_id, 'body_lbl': body_lbl,
        'H01_profile': {str(k): round(v, 5) for k, v in h01_profile.items()},
        'H01_L23': round(h01_l23, 5) if h01_l23 is not None else None,
        'gate_state': gate_state,
        'sim_C_path': sim_c_path,
    }


# ── Run all cases ─────────────────────────────────────────────────────────────
all_cases   = []
all_results = {}

for cases, group in [
    (CHINESE_REF,           'ZH-ref'),
    (ENGLISH_PHRASES_STRONG,'EN-strong'),
    (ENGLISH_PHRASES_WEAK,  'EN-weak'),
    (ENGLISH_SINGLE_TOK,    'EN-single'),
    (ENGLISH_LONG,          'EN-long'),
]:
    for item in cases:
        text = item[0]
        note = item[1]
        r    = analyse(text)
        r['group'] = group
        r['note']  = note
        all_cases.append(r)
        all_results[text] = r


# ── Section 1: The decisive gate test ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — The Decisive Gate Test")
print(f"  H01@L23 for Chinese refs vs English phrases")
print(f"  Gate threshold: {GATE_THRESHOLD}")
print(f"{'='*70}")

def display_row(r):
    tok_s  = str(r['tokens'])[:22]
    h01    = f"{r['H01_L23']:.3f}" if r['H01_L23'] is not None else "  — "
    gate   = r['gate_state']
    zone   = f"{r['zone']}({r['sim_C']:.3f}C)"
    body   = r['body_lbl'][:28]
    return f"  {repr(r['text']):<22s} {tok_s:<22s}  H01={h01}  {gate:<7s}  {zone:<16s}  {body}"

print(f"\n  Chinese references:")
for r in [x for x in all_cases if x['group'] == 'ZH-ref']:
    print(display_row(r))

print(f"\n  English strong phrases (predicted OPEN):")
for r in [x for x in all_cases if x['group'] == 'EN-strong']:
    print(display_row(r))

print(f"\n  English weak phrases (predicted borderline):")
for r in [x for x in all_cases if x['group'] == 'EN-weak']:
    print(display_row(r))

print(f"\n  English single-token baseline (1-tok, no gate):")
for r in [x for x in all_cases if x['group'] == 'EN-single']:
    print(display_row(r))

print(f"\n  English longer phrases (3-token):")
for r in [x for x in all_cases if x['group'] == 'EN-long']:
    print(display_row(r))


# ── Section 2: sim_C vs gate state correlation ────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — sim_C at L14 vs Gate State")
print(f"  Is there a sim_C threshold below which gate is CLOSED?")
print(f"{'='*70}")

multi_tok = [r for r in all_cases if r['n'] >= 2]
multi_tok.sort(key=lambda x: x['sim_C'])

print(f"\n  All multi-token cases sorted by sim_C (last token, L14):")
print(f"  {'sim_C':>7s}  Gate     Group       Text")
print(f"  {'-'*60}")
for r in multi_tok:
    print(f"  {r['sim_C']:>7.3f}  {r['gate_state']:<7s}  {r['group']:<12s}  {repr(r['text'])}")

# Find the sim_C boundary
open_sims   = [r['sim_C'] for r in multi_tok if r['gate_state'] == 'OPEN']
closed_sims = [r['sim_C'] for r in multi_tok if r['gate_state'] == 'CLOSED']
print(f"\n  OPEN gates:   sim_C range [{min(open_sims):.3f}, {max(open_sims):.3f}]  n={len(open_sims)}"
      if open_sims else "\n  OPEN gates: none")
print(f"  CLOSED gates: sim_C range [{min(closed_sims):.3f}, {max(closed_sims):.3f}]  n={len(closed_sims)}"
      if closed_sims else "  CLOSED gates: none")


# ── Section 3: H01 layer profile for matched pairs ───────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — H01 Layer Profile: Chinese vs English Phrases")
print(f"  Show the full L0→L27 evolution of backward attention")
print(f"{'='*70}")

# Pick 2 matched pairs: walk-concept and sing-concept
showcase_texts = ["走着", "is walking", "keep walking",
                  "唱着", "was singing", "start singing"]
showcase = [r for r in all_cases if r['text'] in showcase_texts]
showcase.sort(key=lambda x: showcase_texts.index(x['text']) if x['text'] in showcase_texts else 99)

print(f"\n  H01 backward attention [last→first] across layers:")
print(f"  {'Text':<22s} " + "  ".join(f"L{L:02d}" for L in PROBE_LAYERS) + "  Gate  sim_C")
print(f"  {'-'*90}")
for r in showcase:
    prof = r['H01_profile']
    vals = [f"{prof[str(L)]:.3f}" if str(L) in prof else " — " for L in PROBE_LAYERS]
    print(f"  {repr(r['text']):<22s} " + "  ".join(vals) +
          f"  {r['gate_state']:<6s}  {r['sim_C']:.3f}")


# ── Section 4: Does H01 track sim_C as context grows? ────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Context Strength Gradient")
print(f"  From isolated gerund through phrase, does H01 decrease as sim_C rises?")
print(f"{'='*70}")

# Group by concept: walking, singing, swimming
for concept_gerund, concept_label in [('walking','walk'), ('singing','sing'), ('swimming','swim')]:
    print(f"\n  Concept: {concept_label}")
    print(f"  {'Text':<22s}  {'n':>2s}  {'sim_C':>7s}  H01@L23   Gate")
    print(f"  {'-'*60}")
    concept_cases = [r for r in all_cases
                     if concept_gerund in r['text'] or concept_gerund.replace('ing','') in r['text']]
    concept_cases.sort(key=lambda x: x['sim_C'])
    for r in concept_cases:
        h01 = f"{r['H01_L23']:.3f}" if r['H01_L23'] is not None else "  — "
        print(f"  {repr(r['text']):<22s}  {r['n']:>2d}  {r['sim_C']:>7.3f}  {h01}    {r['gate_state']}")


# ── Section 5: The verdict ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Verdict")
print(f"{'='*70}")

n_open  = sum(1 for r in multi_tok if r['gate_state'] == 'OPEN')
n_close = sum(1 for r in multi_tok if r['gate_state'] == 'CLOSED')
zh_open = sum(1 for r in multi_tok if r['gate_state'] == 'OPEN' and r['group'] == 'ZH-ref')
en_open = sum(1 for r in multi_tok if r['gate_state'] == 'OPEN' and 'EN' in r['group'])

print(f"\n  Multi-token cases: {len(multi_tok)}")
print(f"  Gate OPEN total: {n_open} ({zh_open} ZH + {en_open} EN)")
print(f"  Gate CLOSED:     {n_close}")
print()
if en_open > 0:
    en_open_texts = [r['text'] for r in multi_tok if r['gate_state'] == 'OPEN' and 'EN' in r['group']]
    print(f"  RESULT: Gate fires for {en_open} English phrase(s): {en_open_texts}")
    print(f"  INTERPRETATION: Gate IS language-agnostic at phrase granularity.")
    print(f"  The gate detects B001→Zone C rotation regardless of scope (word/phrase).")
else:
    print(f"  RESULT: Gate fires for 0 English phrases.")
    print(f"  INTERPRETATION: Gate is Chinese-specific. Even when English phrases")
    print(f"  achieve equal sim_C as Chinese compounds, the gate does not fire.")
    print(f"  The gate reads a Chinese-specific hidden-state signature, not sim_C.")
    print()
    print(f"  Chinese OPEN gates: H01 values = "
          + str([round(r['H01_L23'],3) for r in multi_tok
                 if r['gate_state'] == 'OPEN' and r['group'] == 'ZH-ref']))
    print(f"  English best phrases: H01 values = "
          + str([round(r['H01_L23'],3) for r in sorted(multi_tok, key=lambda x: x['sim_C'])
                 if 'EN' in r['group'] and r['H01_L23'] is not None][-5:]))


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {
        'experiment': 'Day 45 — Phrase-Level Gate Test',
        'gate_layer': GATE_LAYER, 'gate_head': GATE_HEAD,
        'gate_threshold': GATE_THRESHOLD,
    },
    'results': to_json({r['text']: r for r in all_cases}),
    'summary': to_json({
        'n_open': n_open, 'n_closed': n_close,
        'zh_open': zh_open, 'en_open': en_open,
        'open_sims': open_sims if open_sims else [],
        'closed_sims': closed_sims if closed_sims else [],
    }),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 45 complete.")
