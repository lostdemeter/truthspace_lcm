#!/usr/bin/env python3
"""
Expedition Day 44 — Same Destination, Different Paths

The Day 43 puzzle: Chinese 走着 fires the L23 gate. English 'singing' (2-tok)
does not. Day 41 called the gate "language-agnostic." Day 43 revised that.

Now we discover the real story with one tokenization check:
  'singing'  (no space, standalone) → 2 tokens ['s', 'inging']  → B000
  ' singing' (space-prefixed, in context) → 1 token ['Ġsinging'] → Zone C?

Chinese: 走着 → ALWAYS 2 tokens in any context → runtime attention composition
English: ' singing' → 1 token in context → tokenizer-time composition

Hypothesis: English and Chinese reach the SAME Zone C bodies, but via
fundamentally different mechanisms:

  Chinese (runtime path):
    token₀ (B001) + token₁ (B001)
    → within-word attention at L14
    → token₁ rotates into Zone C
    → gate at L23 H01 fires (B001→C rotation confirmed)

  English (tokenizer path):
    ' singing' → single token → direct Zone C embedding
    No runtime composition. No gate needed. Already there.

The LLM's gate is not a universal semantic completeness detector.
It is specifically the confirmation signal for RUNTIME composition events.
English doesn't trigger it because English composition happens before
the model even starts.

This experiment:
  1. Confirm: space-prefixed English -ing words are Zone C single tokens
  2. Cross-lingual body matching: ' singing' vs 唱着 — same body?
  3. Rotation angle: 走着 (runtime) vs ' singing' (tokenizer) vs English morphemes
  4. Layer-by-layer path: when does each form arrive at Zone C?
  5. Phrase-level: English multi-token phrases ('is walking') — where does Zone C emerge?
  6. The "torque" gradient: is the rotation angle directly proportional to
     "how far from Zone C the token started"?

Architecture: Qwen2-1.5B-Instruct, 28 layers, 12 Q-heads, 2 KV-heads
"""

import os, json
import numpy as np

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day44_convergence_paths.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# ── Matched cross-lingual pairs ────────────────────────────────────────────────
# (chinese_form, english_context_form, english_standalone, concept_label)
MATCHED_PAIRS = [
    ("走着",  " walking",  "walking",  "physical locomotion (walk)"),
    ("跑着",  " running",  "running",  "physical locomotion (run)"),
    ("吃着",  " eating",   "eating",   "consumption (eat)"),
    ("唱着",  " singing",  "singing",  "musical production (sing)"),
    ("做着",  " making",   "making",   "creation/production (make)"),
]

# Extra English cases: context form vs standalone, to measure the tokenization effect
ENGLISH_CONTEXT_VS_STANDALONE = [
    (" singing",  "singing",  "s+inging (phonemic)"),
    (" swimming", "swimming", "sw+imming (phonemic)"),
    (" walking",  "walking",  "walking (single tok both forms)"),
    (" running",  "running",  "running (single tok both forms)"),
    (" quickly",  "quickly",  "quick+ly (root+suffix)"),
    (" bigger",   "bigger",   "b+igger (phonemic)"),
    (" fastest",  "fastest",  "fast+est (root+suffix)"),
]

# English multi-token phrases — where does Zone C emerge?
ENGLISH_PHRASES = [
    ("is walking",       "copula + gerund"),
    ("was singing",      "past copula + gerund"),
    ("I am walking",     "pronoun + copula + gerund"),
    ("she was singing",  "pronoun + past copula + gerund"),
    ("keep walking",     "aspectual verb + gerund"),
    ("start singing",    "inceptive verb + gerund"),
    ("loves singing",    "mental verb + gerund"),
    ("go swimming",      "motion verb + gerund"),
    ("the singing",      "determiner + gerund"),
    ("her singing",      "possessive + gerund"),
]

PROBE_LAYERS = [0, 1, 5, 10, 14, 20, 23, 27]

print("=" * 70)
print("  Expedition Day 44 — Same Destination, Different Paths")
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
num_heads = model.config.num_attention_heads
num_kv    = model.config.num_key_value_heads
head_dim  = model.config.hidden_size // num_heads
print(f"  num_Q_heads={num_heads}  num_KV_heads={num_kv}  head_dim={head_dim}")


def run_text(text):
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

def rotation_angle_deg(h0, hL):
    """Angle between hidden states at layer 0 and layer L (in degrees)."""
    v0 = h0 / (np.linalg.norm(h0) + 1e-20)
    vL = hL / (np.linalg.norm(hL) + 1e-20)
    cos_sim = np.clip(float(v0 @ vL), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))

def zone_path(token_strs, hs, pos=-1):
    """Zone assignment at each layer for a given token position."""
    if pos == -1:
        pos = len(token_strs) - 1
    path = {}
    for L in PROBE_LAYERS:
        if L < len(hs):
            phi_v = phi_single(hs[L][pos])
            z, sims, body_id, body_lbl, body_sim = assign_zone(phi_v)
            path[L] = {'zone': z, 'sim_C': round(float(sims['C']), 4),
                       'sim_B001': round(float(sims['B001']), 4),
                       'body': body_id, 'body_lbl': body_lbl[:25],
                       'body_sim': round(body_sim, 4)}
    return path

def h01_backward(token_strs, attn, pos_from=-1, pos_to=0):
    """Backward attention from last token to first at GATE_LAYER, head index 1."""
    if pos_from == -1:
        pos_from = len(token_strs) - 1
    if pos_from <= pos_to or len(token_strs) < 2:
        return None
    return {L: round(float(attn[L][1, pos_from, pos_to]), 5)
            for L in PROBE_LAYERS if L < len(attn)}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Tokenization + Zone C for context-form English words
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Context form (' word') vs standalone ('word')")
print(f"  The key question: do space-prefixed forms tokenize as single tokens")
print(f"  and land in Zone C directly?")
print(f"{'='*70}")
print(f"\n  {'Form':<20s}  n   tokens                Zone(last)     Body")
print(f"  {'-'*75}")

sec1_results = {}
for ctx_form, standalone, note in ENGLISH_CONTEXT_VS_STANDALONE:
    for form in [ctx_form, standalone]:
        token_strs, hs, attn = run_text(form)
        n = len(token_strs)
        last_phi = phi_single(hs[14][-1])
        z, sims, body_id, body_lbl, body_sim = assign_zone(last_phi)
        rot = rotation_angle_deg(hs[0][-1], hs[14][-1])
        sec1_results[form] = {
            'tokens': token_strs, 'n': n,
            'zone': z, 'sim_C': round(float(sims['C']), 4),
            'body': body_id, 'body_lbl': body_lbl[:30],
            'rotation_L0_L14': round(rot, 2),
        }
        tok_str = str(token_strs)[:28]
        zone_str = f"{z}({sims['C']:.2f}C)"
        print(f"  {repr(form):<20s}  {n}   {tok_str:<28s}  {zone_str:<15s}  {body_lbl[:25]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Cross-lingual body matching
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Cross-Lingual Body Matching")
print(f"  Do Chinese 着-forms and English context-form -ing words")
print(f"  land in the SAME Zone C body?")
print(f"{'='*70}")
print(f"\n  {'Concept':<30s}  {'Form':<16s}  n  Zone    Body (truncated)")
print(f"  {'-'*80}")

sec2_results = {}
for zh_form, en_ctx, en_alone, concept_lbl in MATCHED_PAIRS:
    for form, form_type in [(zh_form, 'ZH-2tok'),
                             (en_ctx,  'EN-ctx'),
                             (en_alone,'EN-alone')]:
        token_strs, hs, attn = run_text(form)
        n = len(token_strs)
        last_phi  = phi_single(hs[14][-1])
        z, sims, body_id, body_lbl, body_sim = assign_zone(last_phi)
        rot = rotation_angle_deg(hs[0][-1], hs[14][-1])
        h01 = h01_backward(token_strs, attn) if n >= 2 else None
        h01_l23 = h01[23] if h01 and 23 in h01 else None
        gate = ('OPEN' if h01_l23 < 0.55 else 'CLOSED') if h01_l23 is not None else 'N/A'

        sec2_results[f"{concept_lbl}|{form_type}"] = {
            'form': form, 'form_type': form_type, 'concept': concept_lbl,
            'tokens': token_strs, 'n': n,
            'zone': z, 'sim_C': round(float(sims['C']), 4),
            'body': body_id, 'body_lbl': body_lbl,
            'rotation_L0_L14': round(rot, 2),
            'H01_L23': h01_l23,
            'gate': gate,
        }
        tok_str = str(token_strs)[:14]
        zone_str = f"{z}({sims['C']:.2f})"
        rot_str  = f"rot={rot:.1f}°"
        gate_str = f"gate={gate}" if h01_l23 is not None else ""
        print(f"  {concept_lbl[:30]:<30s}  {form_type:<10s} {n}  {zone_str:<10s}  {body_lbl[:28]}  {rot_str}  {gate_str}")
    print()

# Body overlap: do matched pairs land in same body?
print(f"\n  Body overlap analysis (ZH ↔ EN-ctx same body?):")
for zh_form, en_ctx, en_alone, concept_lbl in MATCHED_PAIRS:
    zh_key  = f"{concept_lbl}|ZH-2tok"
    en_key  = f"{concept_lbl}|EN-ctx"
    zh_body = sec2_results.get(zh_key, {}).get('body', '?')
    en_body = sec2_results.get(en_key, {}).get('body', '?')
    match   = "✓ SAME" if zh_body == en_body else "✗ DIFF"
    zh_lbl  = sec2_results.get(zh_key, {}).get('body_lbl', '?')[:25]
    en_lbl  = sec2_results.get(en_key, {}).get('body_lbl', '?')[:25]
    # Cross-body cosine
    if zh_body in body_cvecs and en_body in body_cvecs:
        cos_bodies = float(body_cvecs[zh_body] @ body_cvecs[en_body])
    else:
        cos_bodies = 0.0
    print(f"  {concept_lbl:<30s}  ZH:{zh_lbl:<25s}  EN:{en_lbl:<25s}  {match}  body_cos={cos_bodies:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Rotation angle — the torque gradient
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — The Torque Gradient")
print(f"  Rotation angle (L0→L14) for the LAST token, ordered by magnitude")
print(f"  Hypothesis: ZH aspect-forms > EN context-forms > EN morpho-forms")
print(f"{'='*70}")
print(f"\n  Collecting rotation angles across all test cases...")

rotation_data = []

# All pairs from section 2
for k, r in sec2_results.items():
    rotation_data.append({
        'text': r['form'], 'type': r['form_type'], 'concept': r['concept'],
        'tokens': r['tokens'], 'n': r['n'],
        'zone': r['zone'], 'rotation': r['rotation_L0_L14'],
    })

# English context forms from section 1
for form, r in sec1_results.items():
    form_type = 'EN-ctx' if form.startswith(' ') else 'EN-alone'
    rotation_data.append({
        'text': form, 'type': form_type, 'concept': '(en form test)',
        'tokens': r['tokens'], 'n': r['n'],
        'zone': r['zone'], 'rotation': r['rotation_L0_L14'],
    })

# Sort by rotation angle descending
rotation_data.sort(key=lambda x: -x['rotation'])

print(f"\n  {'Text':<20s} {'Type':<12s} {'n':>2s}  {'Rot(L0→14)':>12s}  {'Zone'}")
print(f"  {'-'*65}")
seen = set()
for r in rotation_data:
    key = (r['text'], r['type'])
    if key in seen:
        continue
    seen.add(key)
    tok_str = str(r['tokens'])[:18]
    zone = f"{r['zone']}({r['zone']})"
    print(f"  {repr(r['text']):<20s} {r['type']:<12s} {r['n']:>2d}  {r['rotation']:>10.1f}°   {r['zone']}")

# Summary statistics by type
from collections import defaultdict
by_type = defaultdict(list)
for r in rotation_data:
    by_type[r['type']].append(r['rotation'])
print(f"\n  Mean rotation by form type:")
for t in ['ZH-2tok', 'EN-ctx', 'EN-alone']:
    vals = by_type.get(t, [])
    if vals:
        print(f"    {t:<12s}  mean={np.mean(vals):6.1f}°  min={min(vals):5.1f}°  max={max(vals):5.1f}°  n={len(vals)}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Layer-by-layer path — when does Zone C emerge?
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Layer-by-Layer Path to Zone C")
print(f"  Show sim_C at each layer for ZH compounds vs EN context forms")
print(f"{'='*70}")

path_cases = []
for zh_form, en_ctx, en_alone, concept_lbl in MATCHED_PAIRS[:3]:  # first 3 concepts
    for form, ftype in [(zh_form, 'ZH'), (en_ctx, 'EN-ctx')]:
        token_strs, hs, attn = run_text(form)
        path = zone_path(token_strs, hs, pos=-1)
        path_cases.append({'form': form, 'type': ftype, 'concept': concept_lbl[:15],
                           'tokens': token_strs, 'path': path})

print(f"\n  sim_C (cosine to Zone C centroid) at each layer for last token:")
print(f"  {'Form':<16s} {'Type':<8s} " + "  ".join(f"L{L:02d}" for L in PROBE_LAYERS))
print(f"  {'-'*80}")
for case in path_cases:
    vals = [f"{case['path'][L]['sim_C']:+.3f}" if L in case['path'] else "  —  "
            for L in PROBE_LAYERS]
    print(f"  {repr(case['form']):<16s} {case['type']:<8s} " + "  ".join(vals))
    # Mark the first layer where zone becomes C
    first_c = next((L for L in PROBE_LAYERS if case['path'].get(L, {}).get('zone') == 'C'), None)
    body_at_14 = case['path'].get(14, {}).get('body_lbl', '?')[:30]
    print(f"  {'':16s} {'':8s} first_C_at=L{first_c if first_c is not None else '??'}  body@L14={body_at_14}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: English phrase-level anchoring
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — English Phrase-Level Zone C Anchoring")
print(f"  For English phrases like 'is walking', track Zone C sim for each")
print(f"  token position at L14. Where does Zone C first appear?")
print(f"{'='*70}")

print(f"\n  {'Phrase':<25s} {'Tokens':<35s}")
print(f"  {'':25s} {'pos':>3s}→zone  (sim_C) at L14")
print(f"  {'-'*80}")

sec5_results = {}
for phrase, desc in ENGLISH_PHRASES:
    token_strs, hs, attn = run_text(phrase)
    n = len(token_strs)
    pos_zones = []
    for pos in range(n):
        phi_v = phi_single(hs[14][pos])
        z, sims, body_id, body_lbl, body_sim = assign_zone(phi_v)
        pos_zones.append({'pos': pos, 'token': token_strs[pos],
                          'zone': z, 'sim_C': round(float(sims['C']), 4),
                          'body': body_id, 'body_lbl': body_lbl[:25]})

    sec5_results[phrase] = {
        'desc': desc, 'tokens': token_strs, 'pos_zones': pos_zones
    }

    tok_str = str(token_strs)[:32]
    print(f"  {repr(phrase):<25s} {tok_str}")
    for pz in pos_zones:
        z_str = f"{pz['zone']}({pz['sim_C']:+.3f}C)"
        body_str = pz['body_lbl'][:28] if pz['zone'] == 'C' else ''
        print(f"  {'':25s} p{pz['pos']}[{pz['token']}] → {z_str}  {body_str}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: The synthesis — two paths, one destination
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 6 — Synthesis: Quantifying the Two Paths")
print(f"{'='*70}")

print(f"\n  For each matched pair, compare:")
print(f"  1. Zone C sim at L14 (destination — should be similar)")
print(f"  2. Zone C sim at L0  (starting point — should differ)")
print(f"  3. Rotation angle    (path length)")
print(f"  4. Gate state        (runtime composition confirmation)")
print()
print(f"  {'Concept':<30s} {'Form':<10s} {'Z@L0':>7s}  {'Z@L14':>7s}  {'Δangle':>8s}  Gate")
print(f"  {'-'*80}")

for zh_form, en_ctx, en_alone, concept_lbl in MATCHED_PAIRS:
    for form, ftype in [(zh_form, 'ZH-2tok'), (en_ctx, 'EN-ctx'), (en_alone, 'EN-alone')]:
        token_strs, hs, attn = run_text(form)
        n = len(token_strs)
        z_L0  = assign_zone(phi_single(hs[0][-1]))[1]  # sims dict
        z_L14 = assign_zone(phi_single(hs[14][-1]))[1]
        rot   = rotation_angle_deg(hs[0][-1], hs[14][-1])
        h01_bw = h01_backward(token_strs, attn)
        h01_l23 = h01_bw[23] if h01_bw and 23 in h01_bw else None
        gate = ('OPEN' if h01_l23 < 0.55 else 'CLOSED') if h01_l23 is not None else '1-tok'
        print(f"  {concept_lbl[:30]:<30s} {ftype:<10s} {z_L0['C']:>+7.3f}  {z_L14['C']:>+7.3f}  {rot:>8.1f}°  {gate}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {'experiment': 'Day 44 — Cross-Lingual Convergence Paths'},
    'section1_tokenization': to_json(sec1_results),
    'section2_body_matching': to_json(sec2_results),
    'section4_phrase_paths': to_json({c['form']: c['path'] for c in path_cases}),
    'section5_english_phrases': to_json(sec5_results),
    'rotation_summary': to_json({
        t: {'mean': float(np.mean(v)), 'min': float(min(v)), 'max': float(max(v))}
        for t, v in by_type.items()
    }),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 44 complete.")
