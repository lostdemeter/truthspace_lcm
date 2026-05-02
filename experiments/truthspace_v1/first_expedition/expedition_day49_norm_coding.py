#!/usr/bin/env python3
"""
Expedition Day 49 — Norm Coding: Structural Principle or L12 Curiosity?

Day 48 found that emphatic punctuation tokens have smaller K-projection norms
at layer 12 than declarative punctuation, and that this predicts gate openness.
The hypothesis: this is NORM CODING — the model encodes closure strength in the
L2 norm of the key vector at the gate layer.

If norm coding is structural, the same principle should hold at all three gate
layers:
  L12 KV1: sentence closures  '.','?','!'  have low ||k|| vs content words
  L18 KV0: phrase heads (noun/adverb) have low ||k|| vs modifiers (adj/det)
  L23 KV0: aspect morpheme 着 has low ||k|| vs bare verb characters

If it is structural AND layer-specific, then:
  - '.' has low ||k|| at L12 but NORMAL ||k|| at L18, L23
  - 着  has low ||k|| at L23 but NORMAL ||k|| at L12, L18
  - Phrase heads have low ||k|| at L18 but NORMAL ||k|| at L12, L23

That three-way specificity would confirm: norm coding is the mechanism by which
each gate layer selects its linguistic scale. The hierarchy is encoded in where
each token's key "goes quiet", not just in its direction.

Architecture: Qwen2-1.5B-Instruct — 28 layers, 12 Q-heads, 2 KV-heads (GQA)
  KV-group 0 (H00-H05): L18/L23 phrase and word gates
  KV-group 1 (H06-H11): L12 sentence gate
"""

import os, json
import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day49_norm_coding.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

GATE_LAYERS = {
    'sentence': (12, 1),   # L12 KV1
    'phrase':   (18, 0),   # L18 KV0
    'word':     (23, 0),   # L23 KV0  (H01 ∈ KV0)
}

print("=" * 70)
print("  Expedition Day 49 — Norm Coding")
print("=" * 70)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map='cpu',
    output_hidden_states=True, output_attentions=True,
    attn_implementation='eager')
model.eval()

num_layers   = model.config.num_hidden_layers
num_heads    = model.config.num_attention_heads
num_kv       = model.config.num_key_value_heads
head_dim     = model.config.hidden_size // num_heads
kv_head_dim  = head_dim  # same head dim for KV

embed_weight = model.model.embed_tokens.weight.detach().numpy()  # [vocab, hidden]

# Extract all k_proj weights per layer per KV-group
# k_proj shape: [num_kv * head_dim, hidden]  → split to [num_kv, head_dim, hidden]
def get_k_proj(layer_idx, kv_group):
    W = model.model.layers[layer_idx].self_attn.k_proj.weight.detach().numpy()
    # W: [num_kv * head_dim, hidden]
    W_split = W.reshape(num_kv, kv_head_dim, -1)
    return W_split[kv_group]   # [head_dim, hidden]

# Pre-cache all k_proj matrices for gate layers
k_projs = {}
for name, (L, kv) in GATE_LAYERS.items():
    k_projs[name] = get_k_proj(L, kv)

# Also cache all layers for the full-scan section
k_projs_all = {}
for L in range(num_layers):
    k_projs_all[L] = {}
    for kv in range(num_kv):
        k_projs_all[L][kv] = get_k_proj(L, kv)

def static_k_norm(token_str, layer, kv_group):
    """Norm of k_proj @ embed for a single token string (static, no context)."""
    ids = tok(token_str, add_special_tokens=False)['input_ids']
    if not ids:
        return None
    e = embed_weight[ids[-1]]   # use last subtoken if multi-token
    k = k_projs_all[layer][kv_group] @ e
    return float(np.linalg.norm(k)), ids[-1]


def contextual_k_norm_last(text, layer, kv_group):
    """Norm of actual key vector at 'layer/kv_group' for the LAST token, in context."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    toks   = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    # hidden state BEFORE the attention of 'layer' is hidden_states[layer]
    h = out.hidden_states[layer][0, -1].numpy()    # [hidden]
    W_k = k_projs_all[layer][kv_group]             # [head_dim, hidden]
    k   = W_k @ h
    return float(np.linalg.norm(k)), toks[-1]


# ── Section 1: Static K-norm across ALL 28 layers per token ───────────────────
print(f"\n{'='*70}")
print(f"Section 1 — Static K-norm: full layer scan for each token type")
print(f"  Uses: k_proj[L,kv] @ embed(token)")
print(f"  KV0 = gate layers L18/L23.  KV1 = gate layer L12.")
print(f"{'='*70}")

# Token sets
SENT_CLOSURE   = ['.', '?', '!']
PARTIAL_CLOSE  = [',', ';', '...']
ZH_CLOSURE     = ['着', '了']   # 了 = completion marker (different from 着?)
ZH_CONTENT     = ['走', '跑', '吃', '看', '在', '是', '的', '有']
EN_CONTENT     = ['cat', 'run', 'the', 'red', 'very', 'ball', 'river']
PHRASE_HEADS   = ['ball', 'river', 'forest', 'tower', 'wolves', 'path', 'ocean']
PHRASE_MODIFIERS = ['red', 'wide', 'dark', 'old', 'grey', 'narrow', 'deep']
DETERMINERS    = ['the', 'a', 'an']

ALL_PROBE_TOKENS = (
    [(t, 'sent_closure')    for t in SENT_CLOSURE] +
    [(t, 'partial_close')   for t in PARTIAL_CLOSE] +
    [(t, 'zh_closure')      for t in ZH_CLOSURE] +
    [(t, 'zh_content')      for t in ZH_CONTENT] +
    [(t, 'phrase_head_en')  for t in PHRASE_HEADS] +
    [(t, 'phrase_mod_en')   for t in PHRASE_MODIFIERS] +
    [(t, 'determiner_en')   for t in DETERMINERS]
)

# For each token, compute static K-norm at every layer for KV0 and KV1
# Show the layer at which each token type reaches minimum norm
print(f"\n  Token   Type               MinNorm  @ Layer/KV  "
      f"  L12/KV1  L18/KV0  L23/KV0")
print(f"  {'-'*80}")

norm_data = {}
for token_str, ttype in ALL_PROBE_TOKENS:
    norms_kv0 = []
    norms_kv1 = []
    for L in range(num_layers):
        n0, _ = static_k_norm(token_str, L, 0)
        n1, _ = static_k_norm(token_str, L, 1)
        norms_kv0.append(n0)
        norms_kv1.append(n1)

    # Find global minimum
    all_norms = [(norms_kv0[L], L, 0) for L in range(num_layers)] + \
                [(norms_kv1[L], L, 1) for L in range(num_layers)]
    min_norm, min_L, min_kv = min(all_norms)

    # Values at gate layers
    n_L12_kv1, _ = static_k_norm(token_str, 12, 1)
    n_L18_kv0, _ = static_k_norm(token_str, 18, 0)
    n_L23_kv0, _ = static_k_norm(token_str, 23, 0)

    # Flag if minimum is at a gate layer
    gate_hit = ''
    if min_L == 12 and min_kv == 1: gate_hit = '← SENT GATE'
    if min_L == 18 and min_kv == 0: gate_hit = '← PHRASE GATE'
    if min_L == 23 and min_kv == 0: gate_hit = '← WORD GATE'

    norm_data[token_str] = {
        'type': ttype,
        'norms_kv0': norms_kv0,
        'norms_kv1': norms_kv1,
        'min_norm': min_norm,
        'min_layer': min_L,
        'min_kv': min_kv,
        'L12kv1': n_L12_kv1,
        'L18kv0': n_L18_kv0,
        'L23kv0': n_L23_kv0,
    }
    print(f"  {token_str!r:<8s}{ttype:<20s}{min_norm:.4f}  L{min_L:02d}/KV{min_kv}  "
          f"  {n_L12_kv1:.4f}   {n_L18_kv0:.4f}   {n_L23_kv0:.4f}  {gate_hit}")


# ── Section 2: Group means at each gate layer ─────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 2 — Group means at each gate layer (static)")
print(f"  Does each group have lowest norm at ITS gate layer?")
print(f"{'='*70}")

groups = {
    'sent_closure':   SENT_CLOSURE,
    'partial_close':  PARTIAL_CLOSE,
    'zh_closure':     ZH_CLOSURE,
    'zh_content':     ZH_CONTENT,
    'phrase_head_en': PHRASE_HEADS,
    'phrase_mod_en':  PHRASE_MODIFIERS,
    'determiner_en':  DETERMINERS,
}

print(f"\n  Group               n  L12/KV1  L18/KV0  L23/KV0  Min-at")
print(f"  {'-'*70}")
group_means = {}
for gname, tokens in groups.items():
    vals = {'L12kv1': [], 'L18kv0': [], 'L23kv0': []}
    for t in tokens:
        if t in norm_data:
            vals['L12kv1'].append(norm_data[t]['L12kv1'])
            vals['L18kv0'].append(norm_data[t]['L18kv0'])
            vals['L23kv0'].append(norm_data[t]['L23kv0'])
    means = {k: np.mean(v) for k, v in vals.items() if v}
    min_key = min(means, key=means.get)
    group_means[gname] = means
    print(f"  {gname:<20s}{len(tokens):>2d}  "
          f"{means.get('L12kv1', 0):.4f}   {means.get('L18kv0', 0):.4f}   "
          f"{means.get('L23kv0', 0):.4f}   {min_key}")


# ── Section 3: Cross-level specificity (contextual) ──────────────────────────
print(f"\n{'='*70}")
print(f"Section 3 — Cross-level specificity (contextual K-norms)")
print(f"  For each gate layer, measure contextual ||k|| for the")
print(f"  RIGHT closure token vs WRONG closure tokens (other levels)")
print(f"{'='*70}")

# Test sentences: each ends with a closure token for a SPECIFIC level
CONTEXTUAL_TESTS = {
    'sentence': [
        ("The cat chases the mouse.", ".", "sent"),
        ("The cat chases the mouse?", "?", "sent"),
        ("The cat chases the mouse!", "!", "sent"),
        ("The cat chases the mouse,", ",", "partial"),
        ("The cat chases the mouse", None, "none"),
    ],
    'phrase_head': [
        ("the big red ball", "ball", "phrase_head"),
        ("the big red",      "red",  "phrase_mod"),
        ("across the wide river", "river", "phrase_head"),
        ("across the wide",       "wide",  "phrase_mod"),
        ("runs very quickly",     "quickly", "phrase_head"),
        ("runs very",             "very",    "phrase_mod"),
    ],
    'zh_word': [
        ("走着",  "着",  "zh_aspect"),
        ("走",    "走",  "zh_verb"),
        ("看着",  "着",  "zh_aspect"),
        ("看",    "看",  "zh_verb"),
        ("吃着",  "着",  "zh_aspect"),
        ("吃",    "吃",  "zh_verb"),
    ],
}

# For each test, measure contextual K-norm at ALL THREE gate layers
print(f"\n  {'Text':<30s}  LastTok  Type          L12/KV1  L18/KV0  L23/KV0")
print(f"  {'-'*90}")

contextual_data = {}
for section, tests in CONTEXTUAL_TESTS.items():
    print(f"\n  --- {section} ---")
    contextual_data[section] = []
    for args in tests:
        text, last_desc, ttype = args
        row = {'text': text, 'type': ttype}
        norms = {}
        for gname, (L, kv) in GATE_LAYERS.items():
            n, last_tok = contextual_k_norm_last(text, L, kv)
            norms[gname] = n
            row[gname] = n
        row['last_tok'] = last_tok

        # Flag which gate layer has lowest norm for this token
        min_gate = min(norms, key=norms.get)
        flag = ''
        expected = {'sentence': 'sentence', 'phrase_head': 'phrase',
                    'zh_word': 'word'}
        if ttype in ('sent', 'partial') and min_gate == 'sentence': flag = '✓'
        elif ttype == 'phrase_head'     and min_gate == 'phrase':   flag = '✓'
        elif ttype == 'zh_aspect'       and min_gate == 'word':     flag = '✓'
        else: flag = '✗' if ttype not in ('none', 'zh_verb', 'phrase_mod') else '—'

        print(f"  {repr(text):<30s}  {last_tok!r:<8s}{ttype:<14s}"
              f"{norms['sentence']:.4f}  {norms['phrase']:.4f}  "
              f"{norms['word']:.4f}  {flag}")
        contextual_data[section].append(row)


# ── Section 4: Specificity summary — ratio test ───────────────────────────────
print(f"\n{'='*70}")
print(f"Section 4 — Specificity ratio")
print(f"  For each token type: (norm at WRONG gate) / (norm at RIGHT gate)")
print(f"  Ratio >> 1 confirms norm coding; ratio ≈ 1 means it's global")
print(f"{'='*70}")

# Expected: sent_closure should have low norm at L12, higher at L18/L23
# phrase_head should have low norm at L18, higher at L12/L23
# zh_closure should have low norm at L23, higher at L12/L18
print(f"\n  Token type       Right gate  Right-norm  Mean-other-norms  Ratio")
print(f"  {'-'*70}")

for section in CONTEXTUAL_TESTS:
    rows = contextual_data[section]
    if section == 'sentence':
        right_rows  = [r for r in rows if r['type'] in ('sent',)]
        right_gate  = 'sentence'
        other_gates = ['phrase', 'word']
    elif section == 'phrase_head':
        right_rows  = [r for r in rows if r['type'] == 'phrase_head']
        right_gate  = 'phrase'
        other_gates = ['sentence', 'word']
    else:  # zh_word
        right_rows  = [r for r in rows if r['type'] == 'zh_aspect']
        right_gate  = 'word'
        other_gates = ['sentence', 'phrase']

    if not right_rows:
        continue
    right_norm = np.mean([r[right_gate] for r in right_rows])
    other_norm = np.mean([r[g] for r in right_rows for g in other_gates])
    ratio = other_norm / right_norm if right_norm > 0 else float('nan')
    label = section.replace('_', ' ')
    print(f"  {label:<18s}{right_gate:<12s}{right_norm:.4f}      "
          f"{other_norm:.4f}           {ratio:.2f}×")


# ── Section 5: Full layer profiles for prototype tokens ──────────────────────
print(f"\n{'='*70}")
print(f"Section 5 — Full layer profile: where does each closure token go quiet?")
print(f"  Static K-norm at every layer for prototype tokens of each level")
print(f"{'='*70}")

PROTOTYPES = {
    'sentence (.)'     : ('.',  1),   # KV1 for sentence gate
    'sentence (?)'     : ('?',  1),
    'sentence (!)'     : ('!',  1),
    'comma (,)'        : (',',  1),
    'zh_aspect (着)'   : ('着', 0),   # KV0 for word gate
    'zh_verb (走)'     : ('走', 0),
    'phrase_head (ball)': ('ball',  0),  # KV0 for phrase gate
    'phrase_mod (red)' : ('red',   0),
    'zh_completion (了)': ('了', 0),   # is 了 more like 着 or 走?
}

# Print profile rows, highlighting gate layers
layers_to_show = [0, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27]
header = f"  {'Token':<22s}" + "".join(f"  L{L:02d}" for L in layers_to_show)
print(f"\n{header}")
print(f"  {'(KV group in parens)':22s}" +
      "".join("  ----" for _ in layers_to_show))

profile_data = {}
for label, (token_str, kv_group) in PROTOTYPES.items():
    norms = []
    for L in layers_to_show:
        n, _ = static_k_norm(token_str, L, kv_group)
        norms.append(n)
    profile_data[label] = {'token': token_str, 'kv': kv_group, 'norms': norms}

    # Highlight gate layers
    row = f"  {label:<22s}"
    for i, (L, n) in enumerate(zip(layers_to_show, norms)):
        marker = '*' if L in (12, 18, 23) else ' '
        row += f" {marker}{n:.3f}"
    print(row)

print(f"\n  (* = gate layer for that KV group)")
print(f"  Gate layers: L12=sentence(KV1), L18=phrase(KV0), L23=word(KV0)")


# ── Section 6: Verdict ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Section 6 — Verdict: Feature, Structure, or Curiosity?")
print(f"{'='*70}")

# Summarize the evidence:
print(f"""
  Evidence checklist for NORM CODING as a structural principle:

  [?] Each token type has lowest K-norm at ITS OWN gate layer (not others)
      → Check Section 2 min-at column and Section 4 ratio

  [?] The ratio (wrong gate norm) / (right gate norm) >> 1 for each type
      → Ratio > 1.5× would be strong evidence of layer-specificity

  [?] Closure token norms at 'wrong' gate layers match non-closure norms
      → Punct tokens should look ordinary at L18/L23

  [?] 了 (Chinese completion) — does it behave like 着 or like a content word?
      → If norm coding is semantic, 了 should show reduced norm at L23 too
      → If it's morpheme-specific, 了 may not (it marks completion, not aspect)
""")

# Print the specificity ratios again cleanly
print(f"  Specificity ratios (other-gate-norm / right-gate-norm):")
for section in CONTEXTUAL_TESTS:
    rows = contextual_data[section]
    if section == 'sentence':
        right_rows  = [r for r in rows if r['type'] == 'sent']
        right_gate  = 'sentence'; other_gates = ['phrase', 'word']
    elif section == 'phrase_head':
        right_rows  = [r for r in rows if r['type'] == 'phrase_head']
        right_gate  = 'phrase';   other_gates = ['sentence', 'word']
    else:
        right_rows  = [r for r in rows if r['type'] == 'zh_aspect']
        right_gate  = 'word';     other_gates = ['sentence', 'phrase']
    if not right_rows: continue
    right_norm = np.mean([r[right_gate] for r in right_rows])
    other_norm = np.mean([r[g] for r in right_rows for g in other_gates])
    ratio = other_norm / right_norm if right_norm > 0 else float('nan')
    verdict = ('STRUCTURAL (>1.5×)' if ratio > 1.5 else
               'MODERATE (1.2-1.5×)' if ratio > 1.2 else
               'WEAK (≤1.2×) → likely curiosity')
    print(f"    {section:<20s}: {ratio:.2f}×  → {verdict}")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_json(obj):
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    return obj

output = {
    'meta': {'experiment': 'Day 49 — Norm Coding Hypothesis'},
    'static_norms': to_json(norm_data),
    'group_means':  to_json(group_means),
    'contextual':   to_json(contextual_data),
    'profiles':     to_json(profile_data),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 49 complete.")
