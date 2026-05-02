#!/usr/bin/env python3
"""
Expedition Day 48 — The Closure Intensity Signal: Definition and Application

Confirmed (Day 47): all three terminal punctuation marks fire L12 KV1 with
values '.':0.271 < '?':0.254 < '!':0.245. The ordering maps to prosodic
intensity. But is this relationship merely a property of the token embedding,
or is it content-modulated — i.e., does the sentence itself influence the value
within the OPEN state?

Four questions:

  Q1: Is the signal content-modulated?
      Same punctuation, different semantic intensity content.
      If "She saved the child!" ≠ "She saved the newspaper!" at L12 KV1,
      the signal is richer than a simple token lookup.

  Q2: What does partial closure look like?
      Comma, semicolon, ellipsis, dash — are they between CLOSED(0.89) and
      OPEN(0.25), i.e., "partial release"? If so, the gate is an analog
      signal, not binary.

  Q3: Does stacking amplify?
      '!!' or '?!' — does the second punctuation token push the gate further?

  Q4: Can we USE this for punctuation prediction?
      Before the punctuation token is appended, does the gate value at the
      LAST CONTENT TOKEN predict which punctuation should follow?
      If so: gate value at last word → predicted punctuation type.
      Application: LCM generation can select punctuation by targeting a
      desired gate value (declarative = 0.27, question = 0.25, emphasis = 0.24).

Q5: Token-level causality
      What specifically about '!' causes lower gate value vs '.'?
      Compare: token embedding cosines, logit distributions, hidden state at
      L12 just before vs after the punctuation token.
"""

import os, json
import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day48_closure_intensity.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

SENT_GATE_L     = 12
SENT_GATE_KV    = 1
SENT_GATE_THRESH = 0.498

print("=" * 70)
print("  Expedition Day 48 — Closure Intensity Signal")
print("=" * 70)

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


def run_text(text):
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    toks   = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    attn    = [out.attentions[L][0].numpy() for L in range(num_layers)]
    hiddens = [out.hidden_states[L][0].numpy() for L in range(num_layers + 1)]
    return toks, attn, hiddens, out.logits[0].numpy()


def gate_val(attn_layers, seq_len, L=SENT_GATE_L, kv=SENT_GATE_KV):
    if seq_len < 2:
        return None
    return float(attn_layers[L][kv, seq_len - 1, 0])


def state(v):
    if v is None: return 'N/A'
    return 'OPEN' if v < SENT_GATE_THRESH else 'CLOSED'


# ── Q1: Content Modulation ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q1 — Is the gate value content-modulated within a punctuation type?")
print(f"{'='*70}")
print(f"  Same punctuation, graded semantic intensity. If values differ")
print(f"  significantly within a group, the signal is content-sensitive.")

# Each group: same punctuation, ascending semantic intensity
# Measured at L12 KV1 (last token = punctuation)
INTENSITY_GROUPS = {
    '!': [
        "She filed the form!",           # low intensity, punctuation mismatched
        "She missed the bus!",           # mild inconvenience
        "She finished the report!",      # modest achievement
        "She won the competition!",      # significant achievement
        "She survived the crash!",       # dramatic
        "She saved the child!",          # heroic
    ],
    '?': [
        "Is it raining?",                # neutral inquiry
        "Do you know the time?",         # mild request
        "Why are you here?",             # probing
        "Where did she go?",             # concern
        "What happened to him?",         # alarm
        "Will she survive?",             # dramatic concern
    ],
    '.': [
        "The form was filed.",           # low salience
        "The bus was late.",             # minor event
        "The report was finished.",      # neutral completion
        "She won the competition.",      # achievement (flat delivery)
        "She survived the crash.",       # dramatic (flat delivery)
        "She saved the child.",          # heroic (flat delivery)
    ],
}

# For the '.' and '!' versions of the SAME content, we can directly compare:
MATCHED_PAIRS = [
    ("She won the competition.", "She won the competition!"),
    ("She survived the crash.",  "She survived the crash!"),
    ("She saved the child.",     "She saved the child!"),
    ("Is it raining.",           "Is it raining?"),
    ("Where did she go.",        "Where did she go?"),
    ("Will she survive.",        "Will she survive?"),
]

q1_results = {}
for punct, sentences in INTENSITY_GROUPS.items():
    print(f"\n  Punctuation: '{punct}'")
    vals = []
    for sent in sentences:
        toks, attn, _, _ = run_text(sent)
        v = gate_val(attn, len(toks))
        vals.append(v)
        vstr = f"{v:.4f}" if v is not None else "—"
        print(f"    {vstr}  {sent}")
    non_none = [v for v in vals if v is not None]
    spread = max(non_none) - min(non_none) if len(non_none) > 1 else 0
    q1_results[punct] = {'values': vals, 'spread': spread,
                          'mean': float(np.mean(non_none)),
                          'sentences': sentences}
    print(f"  → spread = {spread:.4f}  mean = {np.mean(non_none):.4f}")

print(f"\n  Matched-pair test (same content, different punctuation):")
print(f"  {'Text':<45s}  Gate     State")
print(f"  {'-'*70}")
matched_results = []
for sent_dot, sent_other in MATCHED_PAIRS:
    for s in [sent_dot, sent_other]:
        toks, attn, _, _ = run_text(s)
        v = gate_val(attn, len(toks))
        vstr = f"{v:.4f}" if v is not None else "—"
        last_p = toks[-1] if toks else '?'
        print(f"  {repr(s):<45s}  {vstr:<7s}  {state(v):<7s}  last={last_p}")
        matched_results.append({'text': s, 'gate': v})
    print()


# ── Q2: Partial Closure Markers ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q2 — Partial closure: comma, semicolon, ellipsis, dash")
print(f"     Expected: OPEN < partial < CLOSED  (analog between states)")
print(f"{'='*70}")

BASE_FOR_PARTIAL = "She was walking through the park"

PARTIAL_MARKERS = [
    ('.',   BASE_FOR_PARTIAL + '.'),
    ('?',   BASE_FOR_PARTIAL + '?'),
    ('!',   BASE_FOR_PARTIAL + '!'),
    (',',   BASE_FOR_PARTIAL + ','),
    (';',   BASE_FOR_PARTIAL + ';'),
    (':',   BASE_FOR_PARTIAL + ':'),
    ('...',  BASE_FOR_PARTIAL + '...'),
    ('—',   BASE_FOR_PARTIAL + '—'),
    ('-',   BASE_FOR_PARTIAL + '-'),
    ('(no punct)', BASE_FOR_PARTIAL),
]

print(f"\n  Base: '{BASE_FOR_PARTIAL}'")
print(f"\n  {'Marker':<12s}  {'Gate':>7s}  State    Tokens (last)")
print(f"  {'-'*50}")
partial_results = []
for marker, text in PARTIAL_MARKERS:
    toks, attn, _, _ = run_text(text)
    v = gate_val(attn, len(toks))
    vstr = f"{v:.4f}" if v is not None else "—"
    last_tok = toks[-1] if toks else '?'
    st = state(v)
    partial_results.append({'marker': marker, 'gate': v, 'text': text})
    print(f"  {marker:<12s}  {vstr:>7s}  {st:<7s}  last={last_tok}")

# Also test partial markers on multiple base sentences
MULTI_BASE = [
    "Dogs love to run",
    "The cat chases the mouse",
    "She was singing beautifully",
]
print(f"\n  Mean gate values across 3 base sentences per marker:")
print(f"  {'Marker':<12s}  " + "  ".join(f"{b[:15]:<15s}" for b in MULTI_BASE) + "  Mean")
multi_partial = {}
for marker, p_char in [('.', '.'), ('?', '?'), ('!', '!'), (',', ','),
                        (';', ';'), ('...', '...'), ('(none)', '')]:
    row_vals = []
    for base in MULTI_BASE:
        text = base + p_char
        toks, attn, _, _ = run_text(text)
        v = gate_val(attn, len(toks))
        row_vals.append(v)
    mean_v = np.mean([v for v in row_vals if v is not None])
    multi_partial[marker] = {'mean': float(mean_v), 'values': row_vals}
    print(f"  {marker:<12s}  " +
          "  ".join(f"{v:.4f}" if v else "—   " for v in row_vals) +
          f"  {mean_v:.4f}")


# ── Q3: Stacked Punctuation ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3 — Stacked punctuation: do '!!' or '?!' amplify?")
print(f"{'='*70}")

BASE_STACK = "She saved the child"
STACKED = [
    ('.',     BASE_STACK + '.'),
    ('!',     BASE_STACK + '!'),
    ('!!',    BASE_STACK + '!!'),
    ('!!!',   BASE_STACK + '!!!'),
    ('?',     BASE_STACK + '?'),
    ('??',    BASE_STACK + '??'),
    ('?!',    BASE_STACK + '?!'),
    ('!?',    BASE_STACK + '!?'),
]

print(f"\n  Base: '{BASE_STACK}'")
print(f"\n  {'Punct':<8s}  {'Gate (last tok)':>15s}  Tokens  State")
print(f"  {'-'*55}")
stacked_results = []
for marker, text in STACKED:
    toks, attn, _, _ = run_text(text)
    v = gate_val(attn, len(toks))
    vstr = f"{v:.4f}" if v is not None else "—"
    last_tok = toks[-1] if toks else '?'
    n = len(toks)
    stacked_results.append({'marker': marker, 'gate': v, 'n_tokens': n})
    print(f"  {marker:<8s}  {vstr:>15s}  n={n:<4d}  {state(v):<7s}  last={last_tok}")


# ── Q4: Punctuation Prediction from Pre-Punctuation Gate Value ────────────────
print(f"\n{'='*70}")
print(f"Q4 — Punctuation prediction: gate value at LAST CONTENT TOKEN")
print(f"     (before any punctuation is appended)")
print(f"     Can we predict which punctuation should follow?")
print(f"{'='*70}")

# These sentences have "natural" punctuation. We measure the gate at the last
# content word, then see if that value predicts the correct terminal marker.
PUNCT_PREDICTION_CORPUS = [
    # (sentence WITHOUT punctuation, "correct" punctuation, explanation)
    ("She saved the child",           "!",  "heroic action → exclamation"),
    ("She filed the form",            ".",  "mundane action → period"),
    ("Is anyone there",               "?",  "question → question mark"),
    ("The building was on fire",      "!",  "urgent → exclamation"),
    ("The report was submitted",      ".",  "completion → period"),
    ("Where did everyone go",         "?",  "inquiry → question mark"),
    ("She screamed in terror",        "!",  "intense emotion → exclamation"),
    ("The door was locked",           ".",  "neutral state → period"),
    ("Can you help me",               "?",  "request → question mark"),
    ("He jumped off the bridge",      "!",  "dramatic → exclamation"),
    ("The meeting starts at nine",    ".",  "factual → period"),
    ("What is the capital of France", "?",  "knowledge question → question mark"),
    ("She finally understood",        "!",  "revelation → exclamation"),
    ("The package arrived today",     ".",  "delivery → period"),
    ("Did you see that",              "?",  "surprise inquiry → question mark"),
]

print(f"\n  Measuring gate at LAST CONTENT TOKEN (no punctuation appended):")
print(f"  {'Text (no punct)':<40s}  Gate     Predicted  Correct  Match")
print(f"  {'-'*90}")

# First, establish gate value ranges for each punctuation type from Q1
# Use Q1 data: mean('.') ≈ 0.271, mean('?') ≈ 0.254, mean('!') ≈ 0.245
# All are OPEN — but the VALUE distinguishes them. Nearest centroid:
centroids = {
    '.': q1_results['.']['mean'],
    '?': q1_results['?']['mean'],
    '!': q1_results['!']['mean'],
}
print(f"\n  Centroids from Q1: '.': {centroids['.']:.4f}  '?': {centroids['?']:.4f}  '!': {centroids['!']:.4f}")
print()

prediction_results = []
correct_count = 0
for text, correct_punct, note in PUNCT_PREDICTION_CORPUS:
    toks, attn, _, _ = run_text(text)
    v = gate_val(attn, len(toks))
    if v is None:
        predicted = 'N/A'
        match = False
    else:
        # Nearest centroid (all are in OPEN range)
        predicted = min(centroids, key=lambda p: abs(centroids[p] - v))
        match = (predicted == correct_punct)
    if match: correct_count += 1
    vstr = f"{v:.4f}" if v is not None else "—"
    flag = '✓' if match else '✗'
    print(f"  {repr(text):<40s}  {vstr:<7s}  {predicted!r:<9s}  {correct_punct!r:<7s}  {flag}  {note}")
    prediction_results.append({'text': text, 'gate': v, 'predicted': predicted,
                                'correct': correct_punct, 'match': match})

accuracy = correct_count / len(PUNCT_PREDICTION_CORPUS)
print(f"\n  Accuracy: {correct_count}/{len(PUNCT_PREDICTION_CORPUS)} = {accuracy:.1%}")


# ── Q5: Token-level causality — what makes '!' lower than '.'? ────────────────
print(f"\n{'='*70}")
print(f"Q5 — Token-level causality")
print(f"     Compare: embedding of '.', '?', '!' in model space")
print(f"     Does the token embedding itself predict gate response?")
print(f"{'='*70}")

# Get token IDs for the three punctuation marks
punct_tokens = {'.': None, '?': None, '!': None, ',': None, ';': None}
for p in punct_tokens:
    ids = tok(p, add_special_tokens=False)['input_ids']
    if ids:
        punct_tokens[p] = ids[-1]
        print(f"  '{p}' → token id {ids[-1]}  "
              f"({tok.convert_ids_to_tokens(ids[-1])})")

embed_weight = model.model.embed_tokens.weight.detach().numpy()

print(f"\n  Embedding pairwise cosines (input embeddings):")
punct_list = [p for p in punct_tokens if punct_tokens[p] is not None]
for i, p1 in enumerate(punct_list):
    for p2 in punct_list[i+1:]:
        id1, id2 = punct_tokens[p1], punct_tokens[p2]
        e1 = embed_weight[id1]
        e2 = embed_weight[id2]
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        print(f"  cos('{p1}', '{p2}') = {cos:.4f}")

# Norms
print(f"\n  Embedding norms:")
for p in punct_list:
    if punct_tokens[p] is not None:
        norm = float(np.linalg.norm(embed_weight[punct_tokens[p]]))
        print(f"  ||embed('{p}')|| = {norm:.4f}")

# Project each onto the L12 KV attention direction to see which is closer to
# the "gate-open" direction. We can do this by looking at the L12 KV weight
# and projecting each punctuation embedding through it.
# W_k at L12: model.model.layers[12].self_attn.k_proj.weight  [kv_dim, hidden]
try:
    k_proj = model.model.layers[SENT_GATE_L].self_attn.k_proj.weight.detach().numpy()
    print(f"\n  K-projection scores for punctuation (L12 k_proj direction):")
    print(f"  (larger negative = more likely to serve as gate-open key)")
    kp_scores = {}
    for p in punct_list:
        if punct_tokens[p] is not None:
            e = embed_weight[punct_tokens[p]]
            # Project through k_proj — score is L2 norm of resulting key vector
            k_vec = k_proj @ e
            kp_scores[p] = float(np.linalg.norm(k_vec))
            print(f"  ||k('{p}')|| = {kp_scores[p]:.4f}")
    # Also the direction: cosine between key vectors
    print(f"\n  Cosine between key vectors at L12:")
    for i, p1 in enumerate(punct_list):
        for p2 in punct_list[i+1:]:
            if punct_tokens[p1] and punct_tokens[p2]:
                k1 = k_proj @ embed_weight[punct_tokens[p1]]
                k2 = k_proj @ embed_weight[punct_tokens[p2]]
                cos = float(np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2)))
                print(f"  cos(k('{p1}'), k('{p2}')) = {cos:.4f}")
except Exception as e:
    print(f"  [K-proj analysis skipped: {e}]")

# Logit comparison: if we force each punctuation as next token, what probability
# does the model assign to each given a neutral incomplete sentence?
print(f"\n  Next-token logit for each punctuation after neutral sentence base:")
BASE_LOGIT = "The cat sat on the mat"
base_toks, _, _, base_logits = run_text(BASE_LOGIT)
# base_logits shape: [seq_len, vocab]
last_logits = base_logits[-1]  # logits at the last position
last_probs  = np.exp(last_logits - np.max(last_logits))
last_probs /= last_probs.sum()
print(f"  Context: '{BASE_LOGIT}'")
for p in ['.', '?', '!', ',', ';']:
    if punct_tokens.get(p) is not None:
        pid  = punct_tokens[p]
        logit = float(last_logits[pid])
        prob  = float(last_probs[pid])
        print(f"  P(next='{p}') = {prob:.6f}  logit={logit:.4f}")


# ── Q6: Application summary — the closure intensity scale ────────────────────
print(f"\n{'='*70}")
print(f"Q6 — Application: defining the Closure Intensity Scale")
print(f"{'='*70}")

print(f"""
  Measured gate values at L12 KV1 (lower = more complete/emphatic):

  Marker        Mean gate  Interpretation
  ──────────────────────────────────────────────────────────────────────
  '!'           {multi_partial.get('!', {}).get('mean', float('nan')):.4f}      Strong emphatic closure
  '?'           {multi_partial.get('?', {}).get('mean', float('nan')):.4f}      Interrogative closure
  '.'           {multi_partial.get('.', {}).get('mean', float('nan')):.4f}      Declarative closure
  ';'           {multi_partial.get(';', {}).get('mean', float('nan')):.4f}      Partial clause boundary
  ','           {multi_partial.get(',', {}).get('mean', float('nan')):.4f}      Soft pause
  (none)        {multi_partial.get('(none)', {}).get('mean', float('nan')):.4f}      No closure (still assembling)
  '...'         {multi_partial.get('...', {}).get('mean', float('nan')):.4f}      Suspension / trailing off
""")

print(f"  Punctuation prediction accuracy (nearest-centroid on pre-punct gate): "
      f"{accuracy:.1%} ({correct_count}/{len(PUNCT_PREDICTION_CORPUS)})")

print(f"""
  APPLICATIONS:

  1. Punctuation selection in generation (LCM):
     - Compute gate value BEFORE appending any punctuation
     - Select punctuation whose centroid is nearest to that value
     - Requires: gate value → punct type mapping (learned from corpus)

  2. Prosodic annotation for TTS:
     - Gate value at terminal token → pitch contour selection
     - OPEN + low (~0.24): rising-fall or emphasis contour
     - OPEN + mid (~0.27): falling (declarative) contour
     - CLOSED: do not terminate prosodic phrase here

  3. Sentence intensity scoring:
     - Within '!' sentences: spread from Q1 ≈ {q1_results['!']['spread']:.4f}
     - Within '?' sentences: spread ≈ {q1_results['?']['spread']:.4f}
     - Within '.' sentences: spread ≈ {q1_results['.']['spread']:.4f}
     - Non-zero spread = sentence CONTENT modulates value within punct type

  4. Completeness probe (cheap):
     - Run only to L12 (43% of full pass depth)
     - gate < 0.498: sentence complete
     - gate value within [0, 0.498]: intensity / mode of closure
""")


# ── Save ─────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {'experiment': 'Day 48 — Closure Intensity Signal'},
    'q1_content_modulation': to_json(q1_results),
    'q2_partial_markers': to_json(multi_partial),
    'q2_partial_single': to_json(partial_results),
    'q3_stacked': to_json(stacked_results),
    'q4_prediction_accuracy': accuracy,
    'q4_predictions': to_json(prediction_results),
    'q5_centroids': to_json(centroids),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 48 complete.")
