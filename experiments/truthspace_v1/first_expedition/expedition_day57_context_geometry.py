#!/usr/bin/env python3
"""
Expedition Day 57 — Context Geometry

All Zone C work to date has used isolated single tokens: the model processes
" king" as a standalone input. In practice, the model ALWAYS processes tokens
in context. Does context change where a word lands in φ-space?

Three questions:
  C1  Does a polysemous word move to different Zone C bodies under different meanings?
      "bank" (financial) vs "bank" (river) — same φ-address or different?

  C2  How far does the φ-vector shift between isolated and contextual processing?
      Is Zone C a static dictionary or a dynamic contextual manifold?

  C3  Does context produce a consistent DIRECTION of shift?
      If "bank" (financial) is always shifted in direction Δ_finance relative to
      the isolated "bank", then context encoding is itself a T2-like operator.

If C1 confirms that polysemous words move to their correct sense body under
context, Zone C is NOT a static dictionary — it is a contextual manifold.
This changes the LCM picture fundamentally: words don't have fixed addresses,
they have REGIONS, and context navigates within that region.
"""

import json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day57_context_geometry.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
TARGET_LAYER = 14

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# Polysemous words with clearly different meanings, each with 2 context sentences
POLYSEMY_TESTS = [
    {
        'word':    'bank',
        'senses': [
            {'label': 'financial',  'sentence': 'She went to the bank to deposit her savings into the account.', 'target': 'bank'},
            {'label': 'river',      'sentence': 'He sat on the grassy bank watching the river flow past.', 'target': 'bank'},
        ]
    },
    {
        'word':    'spring',
        'senses': [
            {'label': 'season',     'sentence': 'The flowers bloom beautifully every spring after the cold winter.', 'target': 'spring'},
            {'label': 'coil',       'sentence': 'The broken spring in the mattress was poking through the fabric.', 'target': 'spring'},
            {'label': 'jump',       'sentence': 'The cat will spring from the shelf to catch the mouse.', 'target': 'spring'},
        ]
    },
    {
        'word':    'light',
        'senses': [
            {'label': 'illumination', 'sentence': 'The morning light streamed through the bedroom window at dawn.', 'target': 'light'},
            {'label': 'not_heavy',    'sentence': 'The feather was so light that it floated gently on the breeze.', 'target': 'light'},
        ]
    },
    {
        'word':    'bark',
        'senses': [
            {'label': 'dog_sound',  'sentence': 'The dog began to bark loudly at the stranger by the gate.', 'target': 'bark'},
            {'label': 'tree',       'sentence': 'The rough bark of the oak tree was covered in green moss.', 'target': 'bark'},
        ]
    },
    {
        'word':    'match',
        'senses': [
            {'label': 'game',       'sentence': 'They watched the football match in the stadium on Saturday afternoon.', 'target': 'match'},
            {'label': 'fire',       'sentence': 'He struck a match to light the candle on the dinner table.', 'target': 'match'},
            {'label': 'equal',      'sentence': 'The new paint colour does not match the original shade on the wall.', 'target': 'match'},
        ]
    },
    {
        'word':    'bat',
        'senses': [
            {'label': 'animal',     'sentence': 'A bat flew silently through the dark cave hunting insects.', 'target': 'bat'},
            {'label': 'cricket',    'sentence': 'He swung the bat and hit the ball clean over the boundary rope.', 'target': 'bat'},
        ]
    },
    {
        'word':    'rock',
        'senses': [
            {'label': 'stone',      'sentence': 'She picked up a flat rock from the beach and skimmed it across the water.', 'target': 'rock'},
            {'label': 'music',      'sentence': 'The band played heavy rock music that shook the entire venue.', 'target': 'rock'},
            {'label': 'verb_move',  'sentence': 'She began to rock the baby gently in her arms until it fell asleep.', 'target': 'rock'},
        ]
    },
    {
        'word':    'pool',
        'senses': [
            {'label': 'swimming',   'sentence': 'The children splashed happily in the outdoor swimming pool all afternoon.', 'target': 'pool'},
            {'label': 'billiards',  'sentence': 'He challenged his friend to a game of pool in the basement bar.', 'target': 'pool'},
            {'label': 'resource',   'sentence': 'The company decided to pool their resources to fund the new project.', 'target': 'pool'},
        ]
    },
    {
        'word':    'bear',
        'senses': [
            {'label': 'animal',     'sentence': 'The bear emerged from the forest and sniffed at the campfire remains.', 'target': 'bear'},
            {'label': 'endure',     'sentence': 'She could not bear the thought of leaving her family behind forever.', 'target': 'bear'},
        ]
    },
    {
        'word':    'run',
        'senses': [
            {'label': 'jog',        'sentence': 'He decided to run five miles every morning to improve his fitness.', 'target': 'run'},
            {'label': 'operate',    'sentence': 'She was chosen to run the entire department after the manager resigned.', 'target': 'run'},
            {'label': 'ski',        'sentence': 'The ski run was covered in fresh powder after the overnight snowfall.', 'target': 'run'},
        ]
    },
    {
        'word':    'mean',
        'senses': [
            {'label': 'intend',     'sentence': 'I did not mean to hurt your feelings when I said that yesterday.', 'target': 'mean'},
            {'label': 'unkind',     'sentence': 'The mean teacher gave impossible homework every single night of the week.', 'target': 'mean'},
            {'label': 'average',    'sentence': 'The mean temperature for July in this region is around twenty degrees.', 'target': 'mean'},
        ]
    },
    {
        'word':    'plant',
        'senses': [
            {'label': 'organism',   'sentence': 'The plant grew tall and green on the kitchen windowsill all summer.', 'target': 'plant'},
            {'label': 'factory',    'sentence': 'Workers at the manufacturing plant voted to go on strike next week.', 'target': 'plant'},
            {'label': 'verb_place', 'sentence': 'They decided to plant apple trees along the entire length of the fence.', 'target': 'plant'},
        ]
    },
]

print("=" * 70)
print("  Expedition Day 57 — Context Geometry")
print("  Does context shift word positions in φ-space?")
print("=" * 70)


# ── Load baseline φ-vectors (isolated tokens) ────────────────────────────────
npz      = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i      = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

def build_z2(pairs, hs_dict):
    ds = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx+a, pfx+b
            if wa in hs_dict and wb in hs_dict:
                d = hs_dict[wb].astype(np.float64) - hs_dict[wa].astype(np.float64)
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    _, _, Vt = np.linalg.svd(np.stack(ds), full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])

z2 = build_z2(KILLING_PAIRS, {w: hs14_all[w2i[w]] for w in words_all if w in w2i})

def to_phi(h, z2):
    h   = h.astype(np.float64)
    nm  = np.linalg.norm(h)
    hn  = h / (nm + 1e-20)
    perp = hn - np.dot(hn, z2) * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)

phi_all = np.stack([to_phi(hs14_all[i], z2) for i in range(len(words_all))])

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

def get_body(word):
    for pfx in [' ', '']:
        wk = pfx + word.lstrip()
        if wk in wmap:
            return wmap[wk].get('L14_body', '?'), wmap[wk].get('L14_label', '?')
    return '?', '?'

def isolated_phi(word):
    for pfx in [' ', '']:
        wk = pfx + word.lstrip()
        if wk in w2i:
            return phi_all[w2i[wk]]
    return None

def nearest_body(phi_v, top_k=1):
    body_centroids = {}
    for w, meta in wmap.items():
        b = meta.get('L14_body', '')
        if not b or b in ('B000','B001',None): continue
        if w not in w2i: continue
        body_centroids.setdefault(b, []).append(phi_all[w2i[w]])
    centroids = {b: np.mean(np.stack(vs), 0) for b, vs in body_centroids.items() if len(vs) >= 3}
    sims = {b: cosine(phi_v, c) for b, c in centroids.items()}
    top = sorted(sims.items(), key=lambda x: -x[1])[:top_k]
    return [(b, s, wmap.get(b, {}).get('label', '?')) for b, s in top]


# ── Load model for contextual extraction ─────────────────────────────────────
print(f"\n  Loading {MODEL_ID} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
print(f"  Model loaded. Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")


def get_contextual_hs(sentence, target_word, layer):
    """
    Run sentence through model, return hidden state at layer for the LAST
    occurrence of target_word in the token sequence.
    """
    inputs   = tok(sentence, return_tensors='pt')
    ids      = inputs['input_ids'][0].tolist()
    id_str   = [tok.decode([t]) for t in ids]

    # Find token positions that match the target word (case-insensitive, strip spaces)
    tw_lower = target_word.lower()
    positions = [i for i, s in enumerate(id_str)
                 if tw_lower in s.lower().strip()]

    if not positions:
        # Try finding any token that is part of the target word
        tw_ids = tok.encode(' ' + target_word, add_special_tokens=False)
        if not tw_ids: tw_ids = tok.encode(target_word, add_special_tokens=False)
        # Find the sequence in ids
        for start in range(len(ids)):
            if ids[start:start+len(tw_ids)] == tw_ids:
                positions = list(range(start, start+len(tw_ids)))
                break

    if not positions:
        return None, None   # target not found

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # Use the position of the first/only target token
    pos = positions[0]
    hs = out.hidden_states[layer][0, pos, :].numpy().astype(np.float64)
    return hs, pos


# ── C1 + C2: Polysemy and shift magnitude ────────────────────────────────────
print(f"\n{'='*70}")
print(f"C1 + C2 — Polysemy: Does context shift φ-address? How far?")
print(f"{'='*70}\n")

results = {}
for test in POLYSEMY_TESTS:
    word = test['word']
    phi_iso = isolated_phi(word)
    if phi_iso is None:
        print(f"  {word}: NOT in isolated vocabulary — skipping")
        continue

    body_iso, label_iso = get_body(word)
    print(f"  Word: {word}")
    print(f"  Isolated:  body={body_iso}  label={label_iso[:40] if label_iso else '?'}")
    print(f"  {'Sense':<14}  {'cos(iso,ctx)':<14}  {'angle°':<10}  body→  label")
    print(f"  {'-'*80}")

    word_results = {'isolated': {'body': body_iso, 'label': label_iso}, 'senses': []}

    for sense in test['senses']:
        hs_ctx, pos = get_contextual_hs(sense['sentence'], sense['target'], TARGET_LAYER)
        if hs_ctx is None:
            print(f"  {sense['label']:<14}: target not found in sentence")
            continue

        phi_ctx = to_phi(hs_ctx, z2)
        cos_val = cosine(phi_iso, phi_ctx)
        ang_val = float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

        # Find nearest body for contextual φ-vector
        top1 = nearest_body(phi_ctx, top_k=1)
        ctx_body = top1[0][0] if top1 else '?'
        ctx_label = wmap.get(word + '_ctx', {}).get('label', top1[0][2] if top1 else '?')
        ctx_sim  = top1[0][1] if top1 else 0.0

        # Get actual label from atlas
        body_label = '?'
        for w_, meta_ in wmap.items():
            if meta_.get('L14_body') == ctx_body and meta_.get('L14_label'):
                body_label = meta_['L14_label']
                break

        same_body = '✓' if ctx_body == body_iso else '✗'
        print(f"  {sense['label']:<14}  {cos_val:<14.6f}  {ang_val:<10.4f}°  "
              f"{same_body}{ctx_body}  {body_label[:40]}")

        word_results['senses'].append({
            'label': sense['label'],
            'cos_iso_ctx': float(cos_val),
            'angle_deg': float(ang_val),
            'ctx_body': ctx_body,
            'same_body_as_iso': ctx_body == body_iso,
            'ctx_body_label': body_label,
        })

    print()
    results[word] = word_results


# ── C3: Context shift direction — is it a T2-like operator? ──────────────────
print(f"\n{'='*70}")
print(f"C3 — Context Shift Direction: Is context encoding a T2-like operator?")
print(f"{'='*70}")
print(f"\n  For each word, compute Δ = φ_context - φ_isolated.")
print(f"  Test: do same-sense contexts produce consistent Δ direction?\n")

shift_results = {}
for word, res in results.items():
    phi_iso = isolated_phi(word)
    if phi_iso is None: continue

    # Group by same sense
    sense_shifts = {}
    for s in res['senses']:
        if s['cos_iso_ctx'] < 0.999:   # only if there was meaningful shift
            pass
        sense_shifts.setdefault(s['label'], [])

    # Compute shift vectors for each sense
    for test in POLYSEMY_TESTS:
        if test['word'] != word: continue
        shifts_by_sense = {}
        for sense in test['senses']:
            hs_ctx, pos = get_contextual_hs(sense['sentence'], sense['target'], TARGET_LAYER)
            if hs_ctx is None: continue
            phi_ctx = to_phi(hs_ctx, z2)
            delta = phi_ctx - phi_iso
            nm    = np.linalg.norm(delta)
            if nm > 1e-20:
                shifts_by_sense[sense['label']] = delta / nm

        if len(shifts_by_sense) < 2:
            continue

        # How different are the shift directions between senses?
        senses = list(shifts_by_sense.keys())
        print(f"  {word}:")
        for i in range(len(senses)):
            for j in range(i+1, len(senses)):
                si, sj = senses[i], senses[j]
                cos_delta = cosine(shifts_by_sense[si], shifts_by_sense[sj])
                print(f"    cos(Δ_{si}, Δ_{sj}) = {cos_delta:+.4f}  "
                      f"({'same direction' if cos_delta > 0.7 else 'different direction' if cos_delta < -0.3 else 'orthogonal'})")

        shift_results[word] = {k: v.tolist() for k, v in shifts_by_sense.items()}
        break


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — Day 57")
print(f"{'='*70}")

# Aggregate statistics
all_senses = [s for res in results.values() for s in res['senses']]
n_total    = len(all_senses)
n_same     = sum(1 for s in all_senses if s['same_body_as_iso'])
n_diff     = n_total - n_same
angles     = [s['angle_deg'] for s in all_senses]

print(f"""
  Polysemous words tested:  {len(results)}
  Total contextual senses:  {n_total}

  Body assignment:
    Same body as isolated:  {n_same}/{n_total} = {n_same/n_total:.3f}
    Different body:         {n_diff}/{n_total} = {n_diff/n_total:.3f}

  φ-shift magnitude:
    Mean shift angle:       {np.mean(angles):.3f}°
    Min shift angle:        {np.min(angles):.3f}°
    Max shift angle:        {np.max(angles):.3f}°
    Std shift angle:        {np.std(angles):.3f}°

  Interpretation:
    If mean angle < 5°:   Zone C is a STATIC dictionary (context barely moves words)
    If mean angle 5-20°:  Zone C has SOFT context sensitivity
    If mean angle > 20°:  Zone C is a DYNAMIC contextual manifold
    If diff body > 30%:   Polysemy IS encoded geometrically (senses are spatially separated)
""")

# Save
def to_py(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    return x

output = {
    'polysemy_results': to_py(results),
    'shift_directions': to_py(shift_results),
    'summary': {
        'n_words': len(results),
        'n_senses': n_total,
        'n_same_body': n_same,
        'n_diff_body': n_diff,
        'mean_angle': float(np.mean(angles)) if angles else 0,
        'max_angle':  float(np.max(angles))  if angles else 0,
    }
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")
print(f"\nDay 57 complete.")
