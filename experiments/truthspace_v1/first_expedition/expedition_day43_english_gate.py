#!/usr/bin/env python3
"""
Expedition Day 43 — Does the Completeness Gate Fire for English?

Day 41 showed:
  - Chinese 着-forms:  H01 backward attention drops 0.96 → 0.31 (gate OPEN)
  - English -ing fragments: H01 holds at 0.96 (gate CLOSED)
  - Gate is described as "semantic completeness" — language-agnostic

The question: what about English 2-token words where the second token is
itself a real, meaningful word (not a phonemic fragment)?

  "singing"  → ["s", "inging"]   ← "inging" is NOT a word — gate should HOLD
  "birthday" → ["birth", "day"]  ← "day" IS a word — gate should OPEN?
  "cannot"   → ["can", "not"]    ← "not" IS a word — gate should OPEN?
  "because"  → ?                 ← tokenization unknown

Hypothesis: the gate tests whether token[1] has become a self-contained
semantic concept, irrespective of language. If "day" absorbs "birth" context
and becomes a Zone C concept (birthdays / celebrations), H01 drops.
If "inging" never becomes a Zone C concept (it is pure phonology), H01 holds.

Three categories:
  A. Semantic compounds  — both tokens are real English words
  B. Morphological forms — second token is a bound morpheme (not a word)
  C. Functional compounds — second token is a function word ("to", "in", etc.)

If the hypothesis is correct:
  A → gate OPEN (H01 drops, Zone C at last position)
  B → gate CLOSED (H01 holds, stays B001)
  C → gate OPEN or ambiguous (function words have some semantic content)

Architecture: Qwen2-1.5B-Instruct, 28 layers, 12 Q-heads, 2 KV-heads
"""

import os, json
import numpy as np

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day43_english_gate.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# ── Test vocabulary ────────────────────────────────────────────────────────────
# Format: (word, category, note, expected_gate)
# expected_gate: 'OPEN' (H01 should drop), 'CLOSED' (H01 should hold), '?' (unknown)
TEST_WORDS = [
    # ── Category A: Semantic compounds (both tokens real words) ──────────────
    # Expected: gate OPEN — second token absorbs semantic context → Zone C
    ("birthday",   "A-compound", "birth+day — celebration concept",       "OPEN"),
    ("notebook",   "A-compound", "note+book — writing object",            "OPEN"),
    ("bedroom",    "A-compound", "bed+room — domestic space",             "OPEN"),
    ("keyboard",   "A-compound", "key+board — tech/instrument object",    "OPEN"),
    ("blackbird",  "A-compound", "black+bird — specific animal",          "OPEN"),
    ("sunlight",   "A-compound", "sun+light — natural phenomenon",        "OPEN"),
    ("downtown",   "A-compound", "down+town — location concept",          "OPEN"),
    ("boyfriend",  "A-compound", "boy+friend — relationship concept",     "OPEN"),
    ("greenhouse", "A-compound", "green+house — specific structure",      "OPEN"),
    ("cannot",     "A-compound", "can+not — modal negation",              "OPEN"),
    ("something",  "A-compound", "some+thing — indefinite object",        "?"),
    ("everyone",   "A-compound", "every+one — collective pronoun",        "?"),
    ("without",    "A-compound", "with+out — preposition compound",       "?"),
    # ── Category B: Morphological fragments (second token NOT a word) ────────
    # Expected: gate CLOSED — second token is phonemic, not semantic
    ("singing",    "B-morpho",  "s+inging — phonemic split",              "CLOSED"),
    ("killing",    "B-morpho",  "k+illing — phonemic split",              "CLOSED"),
    ("bigger",     "B-morpho",  "big+ger — comparative morpheme",        "CLOSED"),
    ("fastest",    "B-morpho",  "fast+est — superlative morpheme",       "CLOSED"),
    ("quickly",    "B-morpho",  "quick+ly — adverb morpheme",            "CLOSED"),
    ("walked",     "B-morpho",  "walk+ed — past tense morpheme",         "CLOSED"),
    ("taller",     "B-morpho",  "tall+er — comparative morpheme",        "CLOSED"),
    ("deepest",    "B-morpho",  "deep+est — superlative morpheme",       "CLOSED"),
    ("loudly",     "B-morpho",  "loud+ly — adverb morpheme",             "CLOSED"),
    ("painted",    "B-morpho",  "paint+ed — past tense morpheme",        "CLOSED"),
    # ── Category C: Functional/abstract compounds ─────────────────────────────
    # These are interesting borderline cases
    ("today",      "C-func",   "to+day — temporal (day IS a word)",       "?"),
    ("because",    "C-func",   "be+cause — causal connector",             "?"),
    ("before",     "C-func",   "be+fore — temporal preposition",          "?"),
    ("inside",     "C-func",   "in+side — spatial compound",              "?"),
    ("outside",    "C-func",   "out+side — spatial compound",             "?"),
    ("anyone",     "C-func",   "any+one — indefinite pronoun",            "?"),
    ("nothing",    "C-func",   "no+thing — negation+object",              "?"),
    ("something",  "C-func",   "some+thing — indefinite (duplicate)",     "?"),
    # ── Reference: Chinese from Day 41 (control group) ────────────────────────
    ("走着",       "ZH-ref",   "walk+着 — semantic compound (Day 41)",    "OPEN"),
    ("唱着",       "ZH-ref",   "sing+着 — semantic compound (Day 41)",    "OPEN"),
    ("走了",       "ZH-ref",   "walk+了 — completion marker (Day 41)",    "CLOSED"),
]

# Threshold: H01 value below this = gate OPEN, above = gate CLOSED
# From Day 41: 走着 → 0.31 (OPEN), singing → 0.96 (CLOSED)
# Use midpoint as threshold
GATE_THRESHOLD = 0.55

# The gate heads from Day 41
GATE_LAYER = 23
GATE_HEADS = [0, 1]   # H01/H02 (0-indexed: heads 0 and 1)

PROBE_LAYERS = [0, 1, 5, 10, 14, 20, 23, 27]

# ── Load atlas + build zone centroids ─────────────────────────────────────────
print("=" * 70)
print("  Expedition Day 43 — The English Completeness Gate")
print("=" * 70)

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
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
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
    top_body = max(body_cvecs, key=lambda bd: float(phi_v@body_cvecs[bd]))
    top_body_sim = float(phi_v @ body_cvecs[top_body])
    top_body_lbl = body_labels[top_body]
    return max(sims, key=sims.get), sims, top_body_lbl, top_body_sim

print(f"  Zone C={len(zone_c_idx)} B000={len(b000_idx)} B001={len(b001_idx)} A/B={len(ab_idx)}")

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
num_heads = model.config.num_attention_heads   # 12
num_kv    = model.config.num_key_value_heads   # 2
head_dim  = model.config.hidden_size // num_heads
print(f"  num_Q_heads={num_heads}  num_KV_heads={num_kv}  head_dim={head_dim}")


def run_word(text):
    """Forward pass; return token_strs, hidden states per layer per pos, attention."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    token_strs = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    hs   = [out.hidden_states[L][0].numpy().astype(np.float64)
            for L in range(len(out.hidden_states))]
    attn = [out.attentions[L][0].numpy()
            for L in range(len(out.attentions))]
    return token_strs, hs, attn


# ── Section 1: Tokenization + Zone assignment ─────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Tokenization + Zone by Position")
print(f"{'='*70}")
print(f"\n  {'Word':<14s} {'Cat':<12s} {'Tokens':<30s} {'n':>2s}  L14 zone (per pos)")
print(f"  {'-'*90}")

results = {}
seen_words = set()

for word, cat, note, expected in TEST_WORDS:
    if word in seen_words:
        continue
    seen_words.add(word)

    token_strs, hs, attn = run_word(word)
    n = len(token_strs)

    zones_by_pos = []
    for pos in range(n):
        phi_v = phi_single(hs[14][pos])
        z, sims, top_lbl, top_sim = assign_zone(phi_v)
        zones_by_pos.append({
            'pos': pos, 'zone': z,
            'sim_C': round(float(sims['C']), 4),
            'sim_B001': round(float(sims['B001']), 4),
            'top_body_label': top_lbl,
            'top_body_sim': round(top_sim, 4),
        })

    # Collect H01/H02 backward attention at gate layer (last token → first token)
    gate_attn = {}
    if n >= 2:
        last_pos  = n - 1
        first_pos = 0
        for L in PROBE_LAYERS:
            if L < len(attn):
                a = attn[L]   # (num_heads, n, n)
                gate_attn[L] = {h: round(float(a[h, last_pos, first_pos]), 5)
                                for h in range(num_heads)}

    h01_at_gate = gate_attn.get(GATE_LAYER, {}).get(0, None)   # head 0
    h02_at_gate = gate_attn.get(GATE_LAYER, {}).get(1, None)   # head 1

    if n >= 2 and h01_at_gate is not None:
        gate_state = 'OPEN' if h01_at_gate < GATE_THRESHOLD else 'CLOSED'
    else:
        gate_state = 'N/A (1-tok)'

    results[word] = {
        'category': cat, 'note': note, 'expected': expected,
        'tokens': token_strs, 'n_tokens': n,
        'zones_by_pos': zones_by_pos,
        'gate_attn_profile': gate_attn,
        'H01_at_L23': h01_at_gate,
        'H02_at_L23': h02_at_gate,
        'gate_state': gate_state,
        'gate_correct': (gate_state == expected) if expected != '?' else None,
    }

    # Display
    tok_str = str(token_strs)[:28]
    zone_str = '  '.join(f"p{z['pos']}:{z['zone']}({z['sim_C']:.2f}C)" for z in zones_by_pos)
    gate_str = f"  H01@L23={h01_at_gate:.3f} → {gate_state}" if h01_at_gate is not None else ""
    corr_str = " ✓" if results[word]['gate_correct'] else (" ✗" if results[word]['gate_correct'] is False else "")
    print(f"  {word:<14s} {cat:<12s} {tok_str:<30s} {n:>2d}  {zone_str}{gate_str}{corr_str}")


# ── Section 2: Gate summary by category ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Gate Summary by Category")
print(f"{'='*70}")

from collections import defaultdict
by_cat = defaultdict(list)
for word, r in results.items():
    by_cat[r['category']].append((word, r))

for cat in sorted(by_cat.keys()):
    entries = by_cat[cat]
    print(f"\n  Category {cat}:")
    print(f"  {'Word':<14s}  {'n':>2s}  {'Tokens':<22s}  {'H01@L23':>8s}  {'Gate':>7s}  "
          f"{'Exp':>7s}  {'OK?':>4s}  Zone(last-pos)")
    print(f"  {'-'*95}")
    for word, r in entries:
        if r['n_tokens'] < 2:
            tok_s = str(r['tokens'])[:20]
            zone_last = r['zones_by_pos'][-1]['zone'] + f"({r['zones_by_pos'][-1]['sim_C']:.2f})"
            print(f"  {word:<14s}  {r['n_tokens']:>2d}  {tok_s:<22s}  {'—':>8s}  {'1-tok':>7s}  "
                  f"{r['expected']:>7s}  {'—':>4s}  {zone_last}")
        else:
            tok_s   = str(r['tokens'])[:20]
            h01     = f"{r['H01_at_L23']:.3f}" if r['H01_at_L23'] is not None else "—"
            gate    = r['gate_state']
            exp     = r['expected']
            ok      = "✓" if r['gate_correct'] else ("✗" if r['gate_correct'] is False else "?")
            zone_last = r['zones_by_pos'][-1]['zone'] + f"({r['zones_by_pos'][-1]['sim_C']:.2f})"
            print(f"  {word:<14s}  {r['n_tokens']:>2d}  {tok_s:<22s}  {h01:>8s}  {gate:>7s}  "
                  f"{exp:>7s}  {ok:>4s}  {zone_last}")


# ── Section 3: The cliff — H01 evolution across layers ────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — H01 Backward Attention Across Layers (the 'cliff')")
print(f"  Show evolution for key words: do English compounds cliff at L23?")
print(f"{'='*70}")

# Pick interesting cases: a couple from each cat that are 2-token
showcase = []
for word, r in results.items():
    if r['n_tokens'] == 2 and r['H01_at_L23'] is not None:
        showcase.append((word, r))

# Sort: OPEN cases first, then CLOSED
showcase.sort(key=lambda x: (x[1]['gate_state'] != 'OPEN', x[0]))

print(f"\n  H01 backward attention [last→first] across layers:")
print(f"  {'Word':<14s} " + "  ".join(f"L{L:02d}" for L in PROBE_LAYERS) + "  Gate")
print(f"  {'-'*90}")
for word, r in showcase:
    prof = r['gate_attn_profile']
    vals = [f"{prof[L][0]:.3f}" if L in prof else "  — " for L in PROBE_LAYERS]
    gate = r['gate_state']
    print(f"  {word:<14s} " + "  ".join(vals) + f"  {gate}")


# ── Section 4: H01 vs H02 comparison ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — H01 vs H02 at L23 (gate head comparison)")
print(f"{'='*70}")
print(f"\n  {'Word':<14s} {'Cat':<12s}  H00@L23  H01@L23  H02@L23  H03@L23  Gate")
print(f"  {'-'*80}")
for word, r in sorted(results.items(), key=lambda x: x[1]['category']):
    if r['n_tokens'] < 2:
        continue
    prof = r['gate_attn_profile'].get(GATE_LAYER, {})
    h_vals = [f"{prof.get(h, 0.0):.3f}" for h in [0, 1, 2, 3]]
    print(f"  {word:<14s} {r['category']:<12s}  " +
          "  ".join(h_vals) + f"  {r['gate_state']}")


# ── Section 5: Zone correlation with gate state ───────────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Gate State vs Zone Assignment Correlation")
print(f"  Does gate OPEN correlate with Zone C at last position?")
print(f"{'='*70}")

print(f"\n  2-token words only:")
print(f"  Gate=OPEN  →  Zone(last):")
open_zones = [r['zones_by_pos'][-1]['zone']
              for _, r in results.items()
              if r['n_tokens'] == 2 and r['gate_state'] == 'OPEN']
print(f"    {open_zones}")

print(f"  Gate=CLOSED →  Zone(last):")
closed_zones = [r['zones_by_pos'][-1]['zone']
                for _, r in results.items()
                if r['n_tokens'] == 2 and r['gate_state'] == 'CLOSED']
print(f"    {closed_zones}")

from collections import Counter
print(f"\n  Zone distribution for OPEN gates:  {dict(Counter(open_zones))}")
print(f"  Zone distribution for CLOSED gates: {dict(Counter(closed_zones))}")

# ── Section 6: Is gate OPEN predictive of Zone C? ────────────────────────────
n_open_C   = sum(1 for z in open_zones   if z == 'C')
n_closed_C = sum(1 for z in closed_zones if z == 'C')
print(f"\n  Of {len(open_zones)} OPEN gates:   {n_open_C} land in Zone C "
      f"({100*n_open_C/max(len(open_zones),1):.0f}%)")
print(f"  Of {len(closed_zones)} CLOSED gates: {n_closed_C} land in Zone C "
      f"({100*n_closed_C/max(len(closed_zones),1):.0f}%)")

# ── Section 7: Accuracy summary ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 7 — Hypothesis Test Summary")
print(f"{'='*70}")

definite = [(w, r) for w, r in results.items()
            if r['n_tokens'] == 2 and r['expected'] != '?' and r['gate_correct'] is not None]
correct  = sum(1 for _, r in definite if r['gate_correct'])
total    = len(definite)

print(f"\n  Definite predictions (A + B categories, 2-token): {total}")
print(f"  Correct: {correct}/{total}  ({100*correct/max(total,1):.0f}%)")
print()
print(f"  Breakdown by category:")
for cat in ['A-compound', 'B-morpho']:
    cat_items = [(w, r) for w, r in definite if r['category'] == cat]
    cat_corr  = sum(1 for _, r in cat_items if r['gate_correct'])
    print(f"    {cat}: {cat_corr}/{len(cat_items)}")

print(f"\n  Category C (functional compounds, no expected label):")
for word, r in results.items():
    if r['category'] == 'C-func' and r['n_tokens'] == 2:
        print(f"    {word:<14s} H01={r['H01_at_L23']:.3f}  gate={r['gate_state']}  "
              f"zone_last={r['zones_by_pos'][-1]['zone']}")

print(f"\n  Chinese reference (Day 41 replication):")
for word, r in results.items():
    if r['category'] == 'ZH-ref' and r['n_tokens'] >= 2:
        print(f"    {word:<10s} H01={r['H01_at_L23']:.3f}  gate={r['gate_state']}  "
              f"expected={r['expected']}  ✓" if r['gate_correct'] else
              f"    {word:<10s} H01={r['H01_at_L23']:.3f}  gate={r['gate_state']}  "
              f"expected={r['expected']}  ✗")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {
        'experiment': 'Day 43 — English Completeness Gate',
        'gate_threshold': GATE_THRESHOLD,
        'gate_layer': GATE_LAYER,
        'gate_heads': GATE_HEADS,
    },
    'results': to_json(results),
    'summary': {
        'total_definite': total,
        'correct': correct,
        'accuracy_pct': round(100*correct/max(total,1), 1),
        'open_zone_dist': dict(Counter(open_zones)),
        'closed_zone_dist': dict(Counter(closed_zones)),
    }
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 43 complete.")
