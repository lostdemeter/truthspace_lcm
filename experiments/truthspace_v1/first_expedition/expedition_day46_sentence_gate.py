#!/usr/bin/env python3
"""
Expedition Day 46 — Nested Schmitt Triggers: The Sentence-Level Gate

Day 41 found a WORD-level Schmitt trigger: H01@L23 fires for Chinese 着-forms.
  - Assembles (H01 rises L0→L14) then releases (H01 drops L14→L23)
  - Asymmetric: slow rise, fast fall
  - Latches: stays low after the drop

Hypothesis (Day 45 discussion): the transformer implements NESTED Schmitt
triggers at every level of linguistic composition. If self-similarity holds:

  Word-level gate:     H01@L23    — fires when 2-token compound is complete
  Phrase-level gate:   ???@???    — fires when a phrase-level unit is complete
  Sentence-level gate: ???@???    — fires when a complete clause is processed

This experiment:
  1. HEAD SCAN: measure backward attention (last→first) for ALL 28×12 heads
     across complete sentences vs fragments. Find heads with maximum COMPLETE
     vs FRAGMENT separation.

  2. SCHMITT PROFILE: for top candidate heads, verify the full rise→peak→fall
     signature across layers. Check asymmetry (fall faster than rise).

  3. LATCH TEST: after the gate fires at sentence end (the '.'), does the
     attention stay low in subsequent tokens?

  4. HIERARCHY: for a Chinese 着-form WITHIN a sentence, do we observe BOTH
     the word-level gate (H01@L23) AND a sentence-level gate simultaneously?

  5. CLAUSE BOUNDARY: for compound sentences ("She sang and he danced"),
     does the sentence-level gate fire at the clause boundary ('and')?
     If yes: it's a genuine clause-completion detector, not just an
     end-of-sentence detector.

Architecture: Qwen2-1.5B-Instruct, 28 layers, 12 Q-heads, 2 KV-heads
"""

import os, json
import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day46_sentence_gate.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

PROBE_LAYERS = [0, 1, 3, 5, 7, 10, 14, 17, 20, 23, 25, 27]

# ── Test corpus ────────────────────────────────────────────────────────────────
# Section 1 & 2: COMPLETE vs FRAGMENT (the core gate test)
COMPLETE_SENTENCES = [
    ("The cat chases the mouse.",            "simple present"),
    ("She was singing beautifully.",         "past progressive"),
    ("He ran quickly through the park.",     "simple past + adverbial"),
    ("The old man sat quietly by the fire.", "complex with PP"),
    ("Dogs love to run and play.",           "verb + infinitive"),
    ("The sun rises every morning.",         "habitual present"),
    ("She smiled and left the room.",        "compound verb"),
    ("Water flows downhill.",                "simple fact"),
]

FRAGMENTS = [
    ("The big brown",                        "incomplete NP"),
    ("She was",                              "incomplete aux"),
    ("He ran quickly through",               "incomplete PP"),
    ("The old man sat quietly by",           "incomplete PP"),
    ("Dogs love to",                         "incomplete infinitive"),
    ("The sun rises every",                  "incomplete NP-adjunct"),
    ("She smiled and",                       "incomplete compound"),
    ("Water",                                "single noun"),
]

# Section 3: Latch test — extend a complete sentence with more tokens
# If gate fires at '.', subsequent tokens should still show it latched
LATCH_SENTENCES = [
    ("The cat chases the mouse. The dog",    "complete + continuation start"),
    ("She was singing beautifully. Then",    "complete + then"),
    ("Water flows downhill. This",           "complete + demonstrative"),
]

# Section 4: Hierarchy — Chinese 着-form inside a sentence
# The word-level gate should fire for 着-form AND sentence-level gate at sentence end
HIERARCHY_SENTENCES = [
    ("他走着去学校",      "He walks-asp to school (ZH, complete)"),
    ("她唱着歌",          "She sings-asp song (ZH, complete)"),
    ("他走着",            "He walks-asp (ZH, fragment — no object)"),
    ("走着",              "walks-asp (ZH, just the compound)"),   # baseline
]

# Section 5: Clause boundary test
COMPOUND_SENTENCES = [
    ("She sang and he danced.",              "compound: clause1 + and + clause2"),
    ("The dog runs and the cat sleeps.",     "compound: two simple clauses"),
    ("He worked hard but she gave up.",      "compound: contrast"),
    ("I think therefore I am.",              "compound: logical"),
    ("She sang",                             "clause1 only (incomplete compound)"),
    ("She sang and",                         "up to coordinator"),
    ("She sang and he",                      "coordinator + subject"),
    ("She sang and he danced",               "complete compound no period"),
    ("She sang and he danced.",              "complete compound with period"),
]

print("=" * 70)
print("  Expedition Day 46 — Nested Schmitt Triggers: Sentence-Level Gate")
print("=" * 70)

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
num_layers = model.config.num_hidden_layers
num_heads  = model.config.num_attention_heads
num_kv     = model.config.num_key_value_heads
print(f"  layers={num_layers}  Q-heads={num_heads}  KV-heads={num_kv}")

# GQA: Q-head i maps to KV-head i // (num_heads // num_kv)
heads_per_kv = num_heads // num_kv


def run_text(text):
    inputs     = tok(text, return_tensors='pt', add_special_tokens=False)
    ids        = inputs['input_ids'][0]
    token_strs = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    # out.attentions: tuple of (1, num_kv_heads, seq, seq)
    attn = [out.attentions[L][0].numpy() for L in range(len(out.attentions))]
    return token_strs, attn


def backward_last_to_first(attn_layer, num_q, num_kv, seq_len):
    """Get backward attention (last→first) for all Q-heads at this layer."""
    if seq_len < 2:
        return np.zeros(num_q)
    result = np.zeros(num_q)
    for q in range(num_q):
        kv = q // (num_q // num_kv)
        result[q] = attn_layer[kv, seq_len - 1, 0]
    return result


# ── Section 1: Full head scan ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Full Head Scan (all {num_layers}×{num_heads} heads)")
print(f"  backward(last→first) at each layer, for complete vs fragment")
print(f"{'='*70}")

# Collect mean backward attention per (layer, head) for complete vs fragment
print(f"\n  Processing {len(COMPLETE_SENTENCES)} complete sentences...")
complete_matrix = np.zeros((num_layers, num_heads))  # mean over sentences
for text, label in COMPLETE_SENTENCES:
    token_strs, attn = run_text(text)
    n = len(token_strs)
    for L in range(num_layers):
        bw = backward_last_to_first(attn[L], num_heads, num_kv, n)
        complete_matrix[L] += bw
complete_matrix /= len(COMPLETE_SENTENCES)

print(f"  Processing {len(FRAGMENTS)} fragments...")
fragment_matrix = np.zeros((num_layers, num_heads))
for text, label in FRAGMENTS:
    token_strs, attn = run_text(text)
    n = len(token_strs)
    for L in range(num_layers):
        bw = backward_last_to_first(attn[L], num_heads, num_kv, n)
        fragment_matrix[L] += bw
fragment_matrix /= len(FRAGMENTS)

# Difference: fragment - complete (positive = fragment stays high, complete drops = gate)
diff_matrix = fragment_matrix - complete_matrix

# Find top-N head-layer pairs by gate-like difference
flat_diff = diff_matrix.flatten()
top_k = 20
top_idx = np.argsort(flat_diff)[::-1][:top_k]

print(f"\n  Top {top_k} head-layer pairs showing fragment > complete (gate candidates):")
print(f"  {'Layer':>6s}  {'Head':>5s}  {'Fragment':>10s}  {'Complete':>10s}  {'Diff':>8s}")
print(f"  {'-'*50}")
top_candidates = []
for idx in top_idx:
    L  = idx // num_heads
    H  = idx % num_heads
    fc = fragment_matrix[L, H]
    cc = complete_matrix[L, H]
    df = diff_matrix[L, H]
    top_candidates.append((L, H, fc, cc, df))
    print(f"  L{L:02d}          H{H:02d}     {fc:>10.4f}  {cc:>10.4f}  {df:>+8.4f}")

# Compare to known word-level gate
print(f"\n  Known word-level gate (L23, H01):")
print(f"  L23 H01:  fragment={fragment_matrix[23,1]:.4f}  complete={complete_matrix[23,1]:.4f}  "
      f"diff={diff_matrix[23,1]:+.4f}")


# ── Section 2: Schmitt profile for top candidates ────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Schmitt Trigger Profile for Top Candidates")
print(f"  Show full layer-by-layer evolution for top 5 candidate heads")
print(f"{'='*70}")

# Take top 5 unique (L,H) pairs
seen_lh = set()
top5 = []
for L, H, fc, cc, df in top_candidates:
    if (L, H) not in seen_lh:
        seen_lh.add((L, H))
        top5.append((L, H, df))
    if len(top5) == 5:
        break

# Include the word-level gate for comparison
if (23, 1) not in seen_lh:
    top5.append((23, 1, diff_matrix[23, 1]))

print(f"\n  Mean backward(last→first) across all probe layers:")
print(f"  {'Candidate':<14s} " + " ".join(f"L{L:02d}" for L in PROBE_LAYERS))
print(f"  {'-'*100}")

for L_gate, H_gate, diff in top5:
    label = f"L{L_gate:02d}/H{H_gate:02d}(diff={diff:+.3f})"
    comp_vals = [f"{complete_matrix[L, H_gate]:>6.3f}" if L < num_layers else " — "
                 for L in PROBE_LAYERS]
    frag_vals = [f"{fragment_matrix[L, H_gate]:>6.3f}" if L < num_layers else " — "
                 for L in PROBE_LAYERS]
    print(f"  {label:<20s} COMPLETE: " + " ".join(comp_vals))
    print(f"  {'':20s} FRAGMENT: " + " ".join(frag_vals))

    # Asymmetry check: find peak layer and gate layer
    comp_profile = [complete_matrix[L, H_gate] for L in range(num_layers)]
    peak_L = int(np.argmax(comp_profile))
    # Find where it drops most steeply after peak
    if peak_L < num_layers - 1:
        drops = [comp_profile[l] - comp_profile[l+1] for l in range(peak_L, num_layers-1)]
        drop_L = peak_L + int(np.argmax(drops))
        rise_rate = (comp_profile[peak_L] - comp_profile[0]) / max(peak_L, 1)
        fall_rate = max(drops)
        asymmetry = fall_rate / (rise_rate + 1e-9)
    else:
        drop_L = peak_L
        asymmetry = 0.0
    print(f"  {'':20s} peak@L{peak_L:02d}={comp_profile[peak_L]:.3f}  "
          f"max_drop@L{drop_L:02d}  rise_rate={rise_rate:.4f}  "
          f"fall_rate={fall_rate:.4f}  asymmetry(fall/rise)={asymmetry:.2f}x")
    print()


# ── Section 3: Latch test ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — Latch Test")
print(f"  After a complete sentence ends ('.'), does the gate stay latched?")
print(f"  Measure backward(last→first) at each token position for top gate head")
print(f"{'='*70}")

if top5:
    L_gate, H_gate, _ = top5[0]
    print(f"\n  Using top candidate: L{L_gate:02d}/H{H_gate:02d}")

    for text, note in LATCH_SENTENCES:
        token_strs, attn = run_text(text)
        n = len(token_strs)
        print(f"\n  '{text}'  ({note})")
        print(f"  Tokens: {token_strs}")
        print(f"  backward(pos→pos0) at L{L_gate:02d}/H{H_gate:02d}:")
        vals = []
        for pos in range(n):
            if pos == 0:
                vals.append('  — ')
            else:
                kv_idx = H_gate // heads_per_kv
                v = attn[L_gate][kv_idx, pos, 0]
                vals.append(f"{v:>5.3f}")
        tok_labels = [f"[{t[:6]}]" for t in token_strs]
        print("  " + "  ".join(tok_labels))
        print("  " + "  ".join(vals))


# ── Section 4: Hierarchy test ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Hierarchy Test: Word-Level AND Sentence-Level Gates")
print(f"  Chinese 着-form within a sentence → both gates should fire")
print(f"{'='*70}")

WORD_GATE_L, WORD_GATE_H = 23, 1  # known word-level gate
if top5:
    SENT_GATE_L, SENT_GATE_H, _ = top5[0]
else:
    SENT_GATE_L, SENT_GATE_H = None, None

print(f"\n  Word-level gate: L{WORD_GATE_L:02d}/H{WORD_GATE_H:02d}")
if SENT_GATE_L is not None:
    print(f"  Sentence-level gate candidate: L{SENT_GATE_L:02d}/H{SENT_GATE_H:02d}")

print()
for text, note in HIERARCHY_SENTENCES:
    token_strs, attn = run_text(text)
    n = len(token_strs)
    print(f"  '{text}' ({note})")
    print(f"  Tokens ({n}): {token_strs}")

    # Word-level gate: last→first within 走着 (pos 1→0) at word-gate layer
    if n >= 2:
        kv_wg = WORD_GATE_H // heads_per_kv
        wg_val = attn[WORD_GATE_L][kv_wg, -1, 0]
        wg_state = 'OPEN' if wg_val < 0.55 else 'CLOSED'
    else:
        wg_val, wg_state = None, 'N/A'

    # Sentence-level gate: last→first at sentence gate
    if SENT_GATE_L is not None and n >= 2:
        kv_sg = SENT_GATE_H // heads_per_kv
        sg_val = attn[SENT_GATE_L][kv_sg, -1, 0]
        # Determine threshold dynamically from section 1 (midpoint between means)
        sg_mid = (complete_matrix[SENT_GATE_L, SENT_GATE_H] +
                  fragment_matrix[SENT_GATE_L, SENT_GATE_H]) / 2
        sg_state = 'OPEN' if sg_val < sg_mid else 'CLOSED'
    else:
        sg_val, sg_state, sg_mid = None, 'N/A', 0

    wg_str = f"{wg_val:.4f}" if wg_val is not None else "—"
    print(f"  Word-gate  L{WORD_GATE_L:02d}/H{WORD_GATE_H:02d}: "
          f"H={wg_str}  → {wg_state}")
    if SENT_GATE_L is not None:
        sg_str = f"{sg_val:.4f}" if sg_val is not None else "—"
        print(f"  Sent-gate  L{SENT_GATE_L:02d}/H{SENT_GATE_H:02d}: "
              f"H={sg_str}  → {sg_state}  (threshold={sg_mid:.3f})")
    print()


# ── Section 5: Clause boundary ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Clause Boundary Test")
print(f"  Does the gate fire at clause boundaries inside compound sentences?")
print(f"{'='*70}")

if top5:
    L_gate, H_gate, _ = top5[0]
    kv_idx = H_gate // heads_per_kv
    sg_mid_val = (complete_matrix[L_gate, H_gate] +
                  fragment_matrix[L_gate, H_gate]) / 2

    print(f"\n  Sentence-gate candidate: L{L_gate:02d}/H{H_gate:02d}  threshold={sg_mid_val:.3f}")
    print(f"\n  {'Text':<40s}  {'Tokens':<30s}  Gate(last→first)")
    print(f"  {'-'*90}")

    for text, note in COMPOUND_SENTENCES:
        token_strs, attn = run_text(text)
        n = len(token_strs)
        sg_val = attn[L_gate][kv_idx, -1, 0] if n >= 2 else None
        sg_state = ('OPEN' if sg_val < sg_mid_val else 'CLOSED') if sg_val is not None else 'N/A'
        tok_str = str(token_strs)[:28]
        sg_str2 = f"{sg_val:.4f}" if sg_val is not None else "—"
        print(f"  {repr(text):<40s}  {tok_str:<30s}  {sg_str2}  {sg_state}  ({note})")


# ── Section 6: Individual sentence profiles for top gate head ─────────────────
print(f"\n{'='*70}")
print(f"Section 6 — Individual Profiles: Complete vs Fragment")
print(f"  Every test sentence through top gate candidate")
print(f"{'='*70}")

if top5:
    L_gate, H_gate, _ = top5[0]
    kv_idx = H_gate // heads_per_kv
    sg_mid_val = (complete_matrix[L_gate, H_gate] +
                  fragment_matrix[L_gate, H_gate]) / 2
    print(f"\n  Gate: L{L_gate:02d}/H{H_gate:02d}  threshold={sg_mid_val:.3f}")
    print(f"\n  {'Text':<42s}  {'n':>3s}  H_gate   State   Category")
    print(f"  {'-'*80}")

    for sentences, cat in [(COMPLETE_SENTENCES, 'COMPLETE'),
                           (FRAGMENTS,          'FRAGMENT')]:
        for text, note in sentences:
            token_strs, attn = run_text(text)
            n = len(token_strs)
            val = attn[L_gate][kv_idx, -1, 0] if n >= 2 else None
            state = ('OPEN' if val < sg_mid_val else 'CLOSED') if val is not None else '1-tok'
            flag = ''
            if cat == 'COMPLETE' and state == 'OPEN':  flag = '✓'
            if cat == 'FRAGMENT' and state == 'CLOSED': flag = '✓'
            if cat == 'COMPLETE' and state == 'CLOSED': flag = '✗ MISS'
            if cat == 'FRAGMENT' and state == 'OPEN':   flag = '✗ FALSE'
            print(f"  {repr(text):<42s}  {n:>3d}  "
                  f"{(f'{val:.4f}' if val is not None else '—'):<8s}  {state:<7s}  {cat}  {flag}")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

# Summarise: for each candidate head, its complete/fragment means across all layers
candidates_summary = {}
for L_gate, H_gate, diff in top5:
    candidates_summary[f"L{L_gate:02d}_H{H_gate:02d}"] = {
        'layer': L_gate, 'head': H_gate, 'diff': diff,
        'complete_profile': {str(L): round(complete_matrix[L, H_gate], 5)
                             for L in PROBE_LAYERS if L < num_layers},
        'fragment_profile': {str(L): round(fragment_matrix[L, H_gate], 5)
                             for L in PROBE_LAYERS if L < num_layers},
    }

output = {
    'meta': {'experiment': 'Day 46 — Sentence-Level Gate (Nested Schmitt Trigger)'},
    'num_layers': num_layers, 'num_heads': num_heads, 'num_kv': num_kv,
    'diff_matrix_top20': to_json([(int(L), int(H), float(df))
                                   for L, H, fc, cc, df in top_candidates]),
    'gate_candidates': to_json(candidates_summary),
    'word_level_gate_ref': {
        'layer': WORD_GATE_L, 'head': WORD_GATE_H,
        'fragment': round(fragment_matrix[WORD_GATE_L, WORD_GATE_H], 5),
        'complete': round(complete_matrix[WORD_GATE_L, WORD_GATE_H], 5),
        'diff': round(diff_matrix[WORD_GATE_L, WORD_GATE_H], 5),
    },
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 46 complete.")
