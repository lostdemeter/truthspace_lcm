#!/usr/bin/env python3
"""
Expedition Day 47 — Filling the Hierarchy: Phrase Gate + Punctuation Test

Confirmed gates so far:
  Chinese word  L23 H01   着 morpheme       asymmetry 12.68×
  English sent  L12 KV1   '.' punctuation   asymmetry  9.26×

Two open questions:

  Q1: Is there a phrase-level gate BETWEEN L12 and L23?
      Linguistic phrases (NP/PP/VP) are intermediate units — above single
      words, below full clauses. If the nesting is truly fractal, a gate
      should exist in the L13–L22 range that fires when the HEAD of a phrase
      is reached (completing "the big red ball", "across the wide river",
      "runs very quickly") but NOT at a sentence-ending period.

  Q2: Do '?' and '!' fire the same L12 KV1 gate as '.'?
      If the gate detects punctuation-as-closure-token rather than just '.':
        - Same gate: all three share the closure mechanism
        - Different gates: each punctuation type has its own gate head/layer
        - This maps to the fractal: mode of closure (declarative/interrogative/
          exclamatory) encoded at different positions in the hierarchy

Architecture: Qwen2-1.5B-Instruct, 28 layers, 12 Q-heads, 2 KV-heads (GQA)
Known: L12 KV-group 1 = H06-H11 (all share KV-head 1), threshold=0.498
       L23 H01 = KV-group 0 (shares with H00-H05), threshold=0.55
"""

import os, json
import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day47_phrase_gate.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

# Dense probe: every layer from L12 to L23, plus L10/L11 as left-anchors
# and L24/L25 as right-anchors
DENSE_LAYERS = list(range(10, 27))   # L10..L26 — 17 layers

# ── Test corpus ────────────────────────────────────────────────────────────────

# Q1: Phrase-level gate test
# Pairs: (complete phrase, matched truncated phrase, phrase-type)
# RULE: truncated = drop the HEAD (the last content word that closes the phrase)
NP_PAIRS = [
    ("the big red ball",         "the big red",           "NP: adj+noun"),
    ("a very old wooden table",  "a very old wooden",     "NP: adj chain"),
    ("the small brown dog",      "the small brown",       "NP: adj+noun"),
    ("an ancient stone tower",   "an ancient stone",      "NP: N+N compound"),
    ("her beautiful singing voice","her beautiful singing","NP: gerund head"),
    ("the three grey wolves",    "the three grey",        "NP: num+adj+noun"),
]

PP_PAIRS = [
    ("across the wide river",    "across the wide",       "PP: P+NP head"),
    ("through the dark forest",  "through the dark",      "PP: P+adj+N"),
    ("beside the old stone wall","beside the old stone",  "PP: P+adj+N+N"),
    ("beneath the fallen leaves","beneath the fallen",    "PP: P+adj+N"),
    ("into the deep blue ocean", "into the deep blue",    "PP: P+adj+adj+N"),
    ("along the narrow path",    "along the narrow",      "PP: P+adj+N"),
]

VP_PAIRS = [
    ("runs very quickly",        "runs very",             "VP: V+adv+adv"),
    ("spoke extremely softly",   "spoke extremely",       "VP: V+adv+adv"),
    ("worked very hard indeed",  "worked very hard",      "VP: V+adv+adv"),
    ("moved surprisingly fast",  "moved surprisingly",    "VP: V+adv+adv"),
    ("arrived quite late",       "arrived quite",         "VP: V+adv+adj"),
    ("smiled warmly and gently", "smiled warmly and",     "VP: V+adv+coord"),
]

ALL_PHRASE_PAIRS = (
    [(c, f, t) for c, f, t in NP_PAIRS] +
    [(c, f, t) for c, f, t in PP_PAIRS] +
    [(c, f, t) for c, f, t in VP_PAIRS]
)

# Q2: Punctuation gate test
# Same sentences with '.', '?', '!', and no punctuation
PUNCT_BASE_SENTENCES = [
    "The cat chases the mouse",
    "She was singing beautifully",
    "Water flows downhill",
    "The old man sat quietly by the fire",
    "Dogs love to run",
]
PUNCTUATIONS = ['.', '?', '!', '']

# Q3: Hierarchy specificity
# A complete phrase (no punctuation) vs a complete sentence (with '.')
# The phrase gate should fire for BOTH, but the sentence gate should only
# fire for the sentence (the '.' token)
HIERARCHY_TESTS = [
    # (phrase-only, full sentence, label)
    ("the big red ball",         "the big red ball.",       "NP only vs sentence with NP"),
    ("across the wide river",    "across the wide river.",  "PP only vs sentence with PP"),
    ("runs very quickly",        "runs very quickly.",      "VP only vs sentence with VP"),
    ("She sang and he danced",   "She sang and he danced.", "compound no punct vs with period"),
]

print("=" * 70)
print("  Expedition Day 47 — Phrase Gate + Punctuation Hierarchy")
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
heads_per_kv = num_heads // num_kv
print(f"  layers={num_layers}  Q-heads={num_heads}  KV-heads={num_kv}")

# Known gates
SENT_GATE_L, SENT_GATE_KV = 12, 1    # KV-group 1 → H06-H11, threshold=0.498
SENT_GATE_THRESH = 0.498
WORD_GATE_L, WORD_GATE_H  = 23, 1    # H01 = KV-group 0, threshold=0.550
WORD_GATE_THRESH = 0.550


def run_text(text):
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    toks   = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    attn = [out.attentions[L][0].numpy() for L in range(num_layers)]
    return toks, attn


def bw_kv(attn_layer, kv_idx, seq_len):
    """Backward attention last→first for a specific KV-head index."""
    if seq_len < 2:
        return None
    return float(attn_layer[kv_idx, seq_len - 1, 0])


def all_kv_bw(attn_layers, seq_len):
    """backward(last→first) for all KV heads, at all dense probe layers."""
    result = {}
    for L in DENSE_LAYERS:
        result[L] = {}
        for kv in range(num_kv):
            result[L][kv] = bw_kv(attn_layers[L], kv, seq_len)
    return result


# ── Section 1: Dense scan for phrase-level gate ───────────────────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Dense Scan L10–L26: Phrase Complete vs Truncated")
print(f"  NP/PP/VP pairs, measuring backward(last→first) at ALL {len(DENSE_LAYERS)} layers")
print(f"{'='*70}")

# Accumulate: for each (layer, kv_head), mean over complete phrases and truncated
complete_means = {L: {kv: [] for kv in range(num_kv)} for L in DENSE_LAYERS}
truncated_means = {L: {kv: [] for kv in range(num_kv)} for L in DENSE_LAYERS}

for complete_text, trunc_text, label in ALL_PHRASE_PAIRS:
    c_toks, c_attn = run_text(complete_text)
    t_toks, t_attn = run_text(trunc_text)
    for L in DENSE_LAYERS:
        for kv in range(num_kv):
            cv = bw_kv(c_attn[L], kv, len(c_toks))
            tv = bw_kv(t_attn[L], kv, len(t_toks))
            if cv is not None: complete_means[L][kv].append(cv)
            if tv is not None: truncated_means[L][kv].append(tv)

# Compute means and diffs
print(f"\n  Mean backward(last→first): trunc−complete (positive = phrase gate pattern)")
print(f"  {'Layer':>6s}  {'KV0':>8s}  {'KV1':>8s}  {'Best':>6s}")
print(f"  {'':>6s}  {'diff(c/t)':>8s}  {'diff(c/t)':>8s}")
print(f"  {'-'*40}")

layer_kv_diffs = {}
for L in DENSE_LAYERS:
    row = {}
    for kv in range(num_kv):
        cm = np.mean(complete_means[L][kv])  if complete_means[L][kv]  else 0
        tm = np.mean(truncated_means[L][kv]) if truncated_means[L][kv] else 0
        row[kv] = (tm - cm, cm, tm)
    layer_kv_diffs[L] = row
    best_kv = max(row, key=lambda k: row[k][0])
    flag = ' ← KNOWN SENT GATE' if L == SENT_GATE_L and best_kv == SENT_GATE_KV else ''
    flag += ' ← KNOWN WORD GATE' if L == WORD_GATE_L else ''
    print(f"  L{L:02d}      {row[0][0]:>+7.4f}   {row[1][0]:>+7.4f}  "
          f"KV{best_kv}(diff={row[best_kv][0]:+.4f}){flag}")

# Find the BEST layer between L13-L22 (phrase window, excluding known gates)
phrase_window = [L for L in DENSE_LAYERS if 13 <= L <= 22]
best_phrase_L = max(phrase_window, key=lambda L: max(layer_kv_diffs[L][kv][0] for kv in range(num_kv)))
best_phrase_kv = max(range(num_kv), key=lambda kv: layer_kv_diffs[best_phrase_L][kv][0])
best_diff = layer_kv_diffs[best_phrase_L][best_phrase_kv][0]
best_cm   = layer_kv_diffs[best_phrase_L][best_phrase_kv][1]
best_tm   = layer_kv_diffs[best_phrase_L][best_phrase_kv][2]

print(f"\n  Best phrase-window candidate (L13–L22):")
print(f"  L{best_phrase_L:02d} KV{best_phrase_kv}:  "
      f"complete={best_cm:.4f}  truncated={best_tm:.4f}  diff={best_diff:+.4f}")

# Phrase threshold = midpoint
phrase_thresh = (best_cm + best_tm) / 2
print(f"  Threshold (midpoint): {phrase_thresh:.4f}")


# ── Section 2: Individual phrase pair results ─────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Individual Results at Best Phrase Gate: "
      f"L{best_phrase_L:02d}/KV{best_phrase_kv}")
print(f"{'='*70}")

print(f"\n  {'Phrase type':<30s}  {'Text':<35s}  {'n':>3s}  Gate    State")
print(f"  {'-'*100}")
for complete_text, trunc_text, label in ALL_PHRASE_PAIRS:
    for text, cat in [(complete_text, 'COMPLETE'), (trunc_text, 'TRUNC')]:
        toks, attn = run_text(text)
        n   = len(toks)
        val = bw_kv(attn[best_phrase_L], best_phrase_kv, n)
        if val is None:
            state = '1-tok'
        else:
            state = 'OPEN' if val < phrase_thresh else 'CLOSED'
        flag = ''
        if cat == 'COMPLETE' and state == 'OPEN':   flag = '✓'
        if cat == 'TRUNC'    and state == 'CLOSED': flag = '✓'
        if cat == 'COMPLETE' and state == 'CLOSED': flag = '✗ MISS'
        if cat == 'TRUNC'    and state == 'OPEN':   flag = '✗ FALSE'
        val_str = f"{val:.4f}" if val is not None else "—"
        print(f"  {label:<30s}  {repr(text):<35s}  {n:>3d}  {val_str:<7s}  {state:<7s}  {cat}  {flag}")
    print()


# ── Section 3: Punctuation gate test ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — Punctuation Gate Test")
print(f"  Do '.', '?', '!' all fire the known L12 KV1 gate?")
print(f"  Or does each punctuation type have its own gate?")
print(f"{'='*70}")

# For each base sentence + each punctuation, measure at:
#   L12 KV1 (known sentence gate)
#   All other (layer, kv) pairs at dense layers — find the MAXIMUM diff per punctuation type

# Collect per-punct type backward attentions at ALL dense layers
punct_data = {p: {L: {kv: [] for kv in range(num_kv)} for L in DENSE_LAYERS}
              for p in PUNCTUATIONS}

for base in PUNCT_BASE_SENTENCES:
    for p in PUNCTUATIONS:
        text  = base + p
        toks, attn = run_text(text)
        n = len(toks)
        for L in DENSE_LAYERS:
            for kv in range(num_kv):
                val = bw_kv(attn[L], kv, n)
                if val is not None:
                    punct_data[p][L][kv].append(val)

# Section 3a: L12 KV1 response per punctuation
print(f"\n  3a) L12 KV1 (known sentence gate, threshold={SENT_GATE_THRESH}) per sentence ending:")
print(f"\n  {'Base sentence':<40s}  {'punct':>5s}  L12 KV1  State    Tokens")
print(f"  {'-'*90}")
for base in PUNCT_BASE_SENTENCES:
    for p in PUNCTUATIONS:
        text  = base + p
        toks, attn = run_text(text)
        n = len(toks)
        val = bw_kv(attn[SENT_GATE_L], SENT_GATE_KV, n)
        val_str = f"{val:.4f}" if val is not None else "—"
        state = ('OPEN' if val < SENT_GATE_THRESH else 'CLOSED') if val is not None else 'N/A'
        last_tok = toks[-1] if toks else '?'
        p_disp = repr(p) if p else '(none)'
        print(f"  {repr(base):<40s}  {p_disp:>6s}  {val_str:<8s}  {state:<7s}  last={last_tok}")

# Section 3b: Mean response at L12 KV1 per punct, plus best alternative
print(f"\n  3b) Mean L12 KV1 per punctuation type:")
for p in PUNCTUATIONS:
    vals = punct_data[p][SENT_GATE_L][SENT_GATE_KV]
    mean_val = np.mean(vals) if vals else float('nan')
    state = ('OPEN' if mean_val < SENT_GATE_THRESH else 'CLOSED') if not np.isnan(mean_val) else 'N/A'
    p_disp = repr(p) if p else '(no punct)'
    print(f"  {p_disp:>12s}: mean={mean_val:.4f}  → {state}")

# Section 3c: Scan — for '?' and '!', find their OWN best gate head (if different from L12 KV1)
print(f"\n  3c) Best gate head per punctuation type (scan L10–L26):")
print(f"      (comparing with-punct vs without-punct mean, no-punct as baseline)")

no_punct_means = {L: {kv: np.mean(punct_data[''][L][kv]) if punct_data[''][L][kv] else 0
                      for kv in range(num_kv)} for L in DENSE_LAYERS}

for p in ['.', '?', '!']:
    # Find (L, kv) pair that maximises: mean(p_vals) vs mean(no_punct_vals) difference
    # i.e., no_punct - p_val (positive = gate fires FOR this punctuation more than no punct)
    best_lkv = None
    best_sep  = -999
    for L in DENSE_LAYERS:
        for kv in range(num_kv):
            p_vals = punct_data[p][L][kv]
            pm = np.mean(p_vals) if p_vals else 0
            nm = no_punct_means[L][kv]
            sep = nm - pm  # positive: punct reduces backward attn (gate opens for punct)
            if sep > best_sep:
                best_sep = sep
                best_lkv = (L, kv)
    bL, bkv = best_lkv
    p_mean = np.mean(punct_data[p][bL][bkv]) if punct_data[p][bL][bkv] else float('nan')
    n_mean = no_punct_means[bL][bkv]
    same_as_sent = '← SAME AS SENT GATE' if bL == SENT_GATE_L and bkv == SENT_GATE_KV else ''
    print(f"  '{p}': best gate = L{bL:02d}/KV{bkv}  "
          f"  with_punct={p_mean:.4f}  no_punct={n_mean:.4f}  "
          f"sep={best_sep:+.4f}  {same_as_sent}")

# Section 3d: Layer profile for each punctuation at L12 KV1
print(f"\n  3d) Mean L12 KV1 per punctuation — full dense layer profile comparison:")
print(f"  Layer " + "".join(f"  {p!r:>8s}" if p else f"  {'none':>8s}" for p in PUNCTUATIONS))
print(f"  " + "-" * 60)
for L in DENSE_LAYERS:
    vals = [np.mean(punct_data[p][L][SENT_GATE_KV]) if punct_data[p][L][SENT_GATE_KV]
            else float('nan') for p in PUNCTUATIONS]
    flag = ' ← SENT GATE' if L == SENT_GATE_L else ''
    print(f"  L{L:02d}  " + "  ".join(f"{v:>8.4f}" if not np.isnan(v) else f"  {'—':>6s}" for v in vals) + flag)


# ── Section 4: Hierarchy specificity ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Hierarchy Specificity")
print(f"  Phrase alone vs same phrase + period: phrase gate AND sentence gate")
print(f"{'='*70}")

print(f"\n  Best phrase gate: L{best_phrase_L:02d}/KV{best_phrase_kv}  "
      f"threshold={phrase_thresh:.4f}")
print(f"  Sentence gate:    L{SENT_GATE_L:02d}/KV{SENT_GATE_KV}  "
      f"threshold={SENT_GATE_THRESH:.4f}")
print()
print(f"  {'Text':<50s}  PhrGate  PhrState  SentGate  SentState")
print(f"  {'-'*95}")

for phrase_text, sent_text, label in HIERARCHY_TESTS:
    for text, ttype in [(phrase_text, 'phrase'), (sent_text, 'sentence')]:
        toks, attn = run_text(text)
        n = len(toks)
        pg = bw_kv(attn[best_phrase_L], best_phrase_kv, n)
        sg = bw_kv(attn[SENT_GATE_L],   SENT_GATE_KV,   n)
        pg_str = f"{pg:.4f}" if pg is not None else "—"
        sg_str = f"{sg:.4f}" if sg is not None else "—"
        pg_state = ('OPEN' if pg < phrase_thresh else 'CLOSED') if pg is not None else 'N/A'
        sg_state = ('OPEN' if sg < SENT_GATE_THRESH else 'CLOSED') if sg is not None else 'N/A'
        print(f"  {repr(text):<50s}  {pg_str:<7s}  {pg_state:<9s}  {sg_str:<8s}  {sg_state}  ({ttype})")
    print()


# ── Section 5: Schmitt asymmetry at best phrase gate ─────────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Schmitt Trigger Profile at Best Phrase Gate")
print(f"  L{best_phrase_L:02d}/KV{best_phrase_kv}: full layer evolution for complete vs truncated")
print(f"{'='*70}")

# Collect mean at ALL probe layers for complete vs truncated
all_layers_c = {L: [] for L in range(num_layers)}
all_layers_t = {L: [] for L in range(num_layers)}

for complete_text, trunc_text, _ in ALL_PHRASE_PAIRS:
    c_toks, c_attn = run_text(complete_text)
    t_toks, t_attn = run_text(trunc_text)
    for L in range(num_layers):
        cv = bw_kv(c_attn[L], best_phrase_kv, len(c_toks))
        tv = bw_kv(t_attn[L], best_phrase_kv, len(t_toks))
        if cv is not None: all_layers_c[L].append(cv)
        if tv is not None: all_layers_t[L].append(tv)

c_profile = [np.mean(all_layers_c[L]) if all_layers_c[L] else 0 for L in range(num_layers)]
t_profile = [np.mean(all_layers_t[L]) if all_layers_t[L] else 0 for L in range(num_layers)]

print(f"\n  Probe layers L0–L27 (COMPLETE = phrase complete, TRUNC = phrase truncated):")
# Print in groups of 7
probe_cols = list(range(28))
header = "  " + "".join(f"  L{L:02d}" for L in probe_cols[:14])
print(header)
print("  COMPLETE " + "".join(f"{c_profile[L]:>6.3f}" for L in probe_cols[:14]))
print("  TRUNC    " + "".join(f"{t_profile[L]:>6.3f}" for L in probe_cols[:14]))
print()
header2 = "  " + "".join(f"  L{L:02d}" for L in probe_cols[14:])
print(header2)
print("  COMPLETE " + "".join(f"{c_profile[L]:>6.3f}" for L in probe_cols[14:]))
print("  TRUNC    " + "".join(f"{t_profile[L]:>6.3f}" for L in probe_cols[14:]))

# Find peak layer
peak_L = int(np.argmax(c_profile))
# Find steepest single-layer fall after peak
drops = [(c_profile[l] - c_profile[l+1], l) for l in range(peak_L, num_layers-1)]
max_drop, drop_L = max(drops)
rise_rate = (c_profile[peak_L] - c_profile[0]) / max(peak_L, 1)
asymmetry = max_drop / (rise_rate + 1e-9)

print(f"\n  Peak: L{peak_L:02d}={c_profile[peak_L]:.4f}")
print(f"  Max single-layer drop: L{drop_L:02d}→L{drop_L+1:02d}"
      f"  ({c_profile[drop_L]:.4f}→{c_profile[drop_L+1]:.4f}  Δ={max_drop:.4f})")
print(f"  Rise rate  (L0→L{peak_L:02d}): {rise_rate:.4f}")
print(f"  Fall rate  (max drop):    {max_drop:.4f}")
print(f"  Asymmetry (fall/rise):    {asymmetry:.2f}×")


# ── Section 6: Summary table ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 6 — Gate Hierarchy Summary")
print(f"{'='*70}")

print(f"""
  Granularity      Gate             Fires on           Threshold  Asymmetry
  ─────────────────────────────────────────────────────────────────────────
  Chinese word     L23 H01          着 morpheme         0.550      12.68×
  English sentence L12 KV-group 1   '.' punctuation     0.498       9.26×
  Phrase (NP/PP/VP)L{best_phrase_L:02d} KV-group {best_phrase_kv}   head noun/adverb    {phrase_thresh:.3f}      {asymmetry:.2f}×
""")

punct_means_l12 = {p: np.mean(punct_data[p][SENT_GATE_L][SENT_GATE_KV])
                   if punct_data[p][SENT_GATE_L][SENT_GATE_KV] else float('nan')
                   for p in PUNCTUATIONS}
print(f"  L12 KV1 mean response by punctuation:")
for p in PUNCTUATIONS:
    pd = repr(p) if p else '(none)'
    m  = punct_means_l12[p]
    s  = ('OPEN' if m < SENT_GATE_THRESH else 'CLOSED') if not np.isnan(m) else 'N/A'
    print(f"    {pd:>10s}: {m:.4f}  → {s}")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {'experiment': 'Day 47 — Phrase Gate + Punctuation Hierarchy'},
    'best_phrase_gate': {
        'layer': best_phrase_L, 'kv': best_phrase_kv,
        'complete_mean': best_cm, 'truncated_mean': best_tm,
        'diff': best_diff, 'threshold': phrase_thresh,
    },
    'phrase_gate_profile': {
        'complete': to_json(c_profile),
        'truncated': to_json(t_profile),
        'peak_layer': peak_L,
        'asymmetry': asymmetry,
    },
    'punct_l12_kv1_means': to_json(punct_means_l12),
    'layer_kv_diffs': to_json(layer_kv_diffs),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 47 complete.")
