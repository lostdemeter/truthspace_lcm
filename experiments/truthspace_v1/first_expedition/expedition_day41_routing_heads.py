#!/usr/bin/env python3
"""
Expedition Day 41 — Routing Head Analysis

Day 40 Zone-C escapes were ALL multi-token words. Day 40 used last_token_pos,
so for 走着 (=[走,着] 2 tokens) we measured 着's hidden state *in context of 走*.
For 走了 (1 token), we measured the merged token directly.

The pattern is exact:
  2-token forms → last piece attends to first → Zone C possible
  1-token forms → no within-word cross-attention → stays B001/B000

This experiment characterises the ROUTING MECHANISM:
  1. Confirm: for 走着, which position holds the Zone C representation — pos 0 or pos 1?
  2. Which attention heads at L14/L23 transfer semantic content from first to last token?
  3. Is the routing MESH rank-1 (narrow-beam selector, like finding 40)?
  4. Cross-lingual: do the same heads route English split words (singing, killing)?
  5. What is the head activation signature that predicts Zone-C promotion?

Architecture: Qwen2-1.5B, 28 layers, 12 Q-heads, 2 KV-heads (GQA), hidden_size=1536
"""

import os, json
import numpy as np

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day41_routing_heads.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"

KILLING_PAIRS = [
    ('cat','cats'),('dog','dogs'),('tree','trees'),('bird','birds'),
    ('house','houses'),('man','woman'),('king','queen'),('boy','girl'),
    ('big','bigger'),('fast','faster'),('old','older'),
]

# Test cases — (text, description, expected_zone)
# For multi-token words: zone prediction for the LAST token position
CASES = [
    # Chinese 2-token: verb + 着 (predict Zone C for last position)
    ("走着",  "walk+着 (2-tok): physical-ongoing → Zone C?", "C"),
    ("跑着",  "run+着  (2-tok): physical-ongoing → Zone C?", "C"),
    ("吃着",  "eat+着  (2-tok): physical-ongoing → Zone C?", "C"),
    ("唱着",  "sing+着 (2-tok): physical-ongoing → Zone C?", "C"),
    ("做着",  "make+着 (2-tok): physical-ongoing → Zone C?", "C"),
    # Chinese 1-token: same verbs + 了 (predict B001)
    ("走了",  "walk+了 (1-tok): completed → B001?",          "B001"),
    ("说了",  "say+了  (1-tok): completed → B001?",          "B001"),
    ("说着",  "say+着  (1-tok combined!): → B001?",           "B001"),
    # Chinese bare chars (predict B001)
    ("走",   "walk bare (1-tok) → B001?",                   "B001"),
    ("着",   "着 alone  (1-tok) → B001?",                   "B001"),
    # English 2-token: morph split predicts Zone C for last position
    ("singing",  "singing  (2-tok [s,inging])  → Zone C?",  "C"),
    ("killing",  "killing  (2-tok [k,illing])  → Zone C?",  "C"),
    ("driving",  "driving  (check n_tokens first)",          "?"),
    ("writing",  "writing  (check n_tokens first)",          "?"),
    # English 1-token: predict B001/B000
    ("walking",  "walking  (1-tok) → B001?",                "B001"),
    ("running",  "running  (1-tok) → B001?",                "B001"),
    ("eating",   "eating   (check n_tokens first)",          "?"),
    ("building", "building (check n_tokens first)",          "?"),
]

# Which layers to probe for attention weights
PROBE_LAYERS = [0, 1, 5, 10, 14, 20, 23, 27]

print("── Load atlas + build zone centroids ─────────────────────────────────")
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
    top_body = max({bd: float(phi_v@cv) for bd,cv in body_cvecs.items()}, key=lambda x: {bd:float(phi_v@body_cvecs[bd]) for bd in body_cvecs}[x])
    top_body_sim = float(phi_v @ body_cvecs[top_body])
    top_body_lbl = body_labels[top_body]
    return max(sims, key=sims.get), sims, top_body_lbl, top_body_sim

print(f"  Zone C={len(zone_c_idx)} B000={len(b000_idx)} B001={len(b001_idx)} A/B={len(ab_idx)}")

print(f"\n── Load model ──────────────────────────────────────────────────────────")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map='cpu',
    output_hidden_states=True, output_attentions=True,
    attn_implementation='eager')
model.eval()
num_heads   = model.config.num_attention_heads   # 12
num_kv      = model.config.num_key_value_heads   # 2
head_dim    = model.config.hidden_size // num_heads
print(f"  num_Q_heads={num_heads}  num_KV_heads={num_kv}  head_dim={head_dim}")

# ── Section 1: Tokenization check + position-by-position zone analysis ───────
print(f"\n{'='*65}")
print(f"Section 1 — Tokenization + Zone by Position")
print(f"{'='*65}")

def extract_all_positions(text):
    """Run model, return (token_strs, hs_per_layer_per_pos, attn_per_layer).
    Uses add_special_tokens=False so token positions are 0-indexed directly."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    token_strs = tok.convert_ids_to_tokens(ids)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    # hidden_states: tuple of (num_layers+1) tensors of shape (1, seq_len, hidden)
    hs = [out.hidden_states[L][0].numpy().astype(np.float64)
          for L in range(len(out.hidden_states))]  # hs[layer][pos]
    # attentions: tuple of num_layers tensors of shape (1, num_heads, seq_len, seq_len)
    attn = [out.attentions[L][0].numpy()   # (num_heads, seq_len, seq_len)
            for L in range(len(out.attentions))]
    return token_strs, hs, attn

section1_results = {}
print(f"\n  {'Text':<10s} {'Tokens':<30s} {'L14:pos→zone'}")
print(f"  {'-'*80}")

for text, desc, pred_zone in CASES:
    token_strs, hs, attn = extract_all_positions(text)
    n_word_toks = len(token_strs)  # no BOS — all positions are word tokens
    zones_by_pos = []
    for pos in range(len(token_strs)):  # all positions
        phi_v = phi_single(hs[14][pos])
        z, sims, top_lbl, top_sim = assign_zone(phi_v)
        zones_by_pos.append((pos, z, sims, top_lbl, top_sim))
    tok_display = str(token_strs)[:28]
    zones_str = '  '.join(f"pos{p-1}→{z}({sims['C']:.2f}C/{sims['B001']:.2f}B1)"
                           for p, z, sims, _, _ in zones_by_pos)
    pred_ok = (any(z == pred_zone for _, z, _, _, _ in zones_by_pos)
               or pred_zone == '?')
    section1_results[text] = {
        'tokens': token_strs,
        'n_word_tokens': n_word_toks,
        'zones_by_pos': [(p, z, sims, lbl, s)
                         for p, z, sims, lbl, s in zones_by_pos],
        'predicted': pred_zone,
        'correct': pred_ok,
        'desc': desc,
    }
    print(f"  {text:<10s} {tok_display:<30s} {zones_str}")

# ── Section 2: Attention weight analysis for 2-token cases ───────────────────
print(f"\n{'='*65}")
print(f"Section 2 — Attention Weights: which heads transfer content?")
print(f"  Focus: A[head, last_word_tok, first_word_tok] at each layer")
print(f"  Contrast: Zone-C-escaping pairs vs B001-staying pairs")
print(f"{'='*65}")

# Separate into Zone-C escaping and B001-staying 2-token cases
escaping_2tok = [(text, data) for text, data in section1_results.items()
                 if data['n_word_tokens'] == 2
                 and any(z == 'C' for _, z, _, _, _ in data['zones_by_pos'])]
staying_2tok  = [(text, data) for text, data in section1_results.items()
                 if data['n_word_tokens'] == 2
                 and all(z != 'C' for _, z, _, _, _ in data['zones_by_pos'])]
# Single-token cases for contrast
single_tok    = [(text, data) for text, data in section1_results.items()
                 if data['n_word_tokens'] == 1]

print(f"\n  Zone-C-escaping 2-token cases: {[t for t,_ in escaping_2tok]}")
print(f"  B001-staying 2-token cases:    {[t for t,_ in staying_2tok]}")
print(f"  Single-token cases:            {[t for t,_ in single_tok[:5]]}")

# Re-run to get attention weights for each case
def get_attn_last_to_first(text):
    """For a 2-token word, get attention from last word token to first word token,
       across all heads and layers. Returns array (num_layers, num_heads).
       Uses add_special_tokens=False so positions are 0-indexed directly."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=False)
    ids    = inputs['input_ids'][0]
    n_toks = ids.shape[0]
    if n_toks < 2:
        return None
    last_pos  = n_toks - 1  # last token (e.g. 着)
    first_pos = 0            # first token (e.g. 走)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    attn_profile = np.zeros((len(out.attentions), num_heads))
    for L, attn_L in enumerate(out.attentions):
        a = attn_L[0].numpy()  # (num_heads, seq_len, seq_len)
        attn_profile[L] = a[:, last_pos, first_pos]  # attention from last to first
    return attn_profile  # (num_layers, num_heads)

print(f"\n  Computing attention profiles...")
attn_profiles = {}

all_2tok = escaping_2tok + staying_2tok
for text, data in all_2tok:
    if data['n_word_tokens'] >= 2:
        profile = get_attn_last_to_first(text)
        if profile is not None:
            attn_profiles[text] = profile

# Mean attention profile for escaping vs staying
if escaping_2tok:
    esc_texts  = [t for t, _ in escaping_2tok if t in attn_profiles]
    stay_texts = [t for t, _ in staying_2tok  if t in attn_profiles]
    if esc_texts:
        mean_esc  = np.stack([attn_profiles[t] for t in esc_texts]).mean(0)
    if stay_texts:
        mean_stay = np.stack([attn_profiles[t] for t in stay_texts]).mean(0)

    # Differential: escaping - staying
    print(f"\n  Mean attention A[last→first] for Zone-C-escaping vs B001-staying:")
    print(f"  Layer  " + "  ".join(f"H{h:02d}" for h in range(num_heads)))
    print(f"  {'─'*80}")
    for L in PROBE_LAYERS:
        if L >= len(mean_esc):
            continue
        esc_row  = mean_esc[L]  if esc_texts  else np.zeros(num_heads)
        stay_row = mean_stay[L] if stay_texts else np.zeros(num_heads)
        diff_row = esc_row - stay_row
        esc_str  = "  ".join(f"{v:.3f}" for v in esc_row)
        diff_str = "  ".join(f"{v:+.3f}" for v in diff_row)
        print(f"  L{L:02d} (esc) {esc_str}")
        if stay_texts:
            print(f"  L{L:02d} (Δesc-stay) {diff_str}")
        print()

    # Top routing heads by differential at L14 and L23
    print(f"\n  Top routing heads (largest esc-stay differential):")
    for L in [14, 23]:
        if L >= len(mean_esc) or not stay_texts:
            continue
        diff = mean_esc[L] - mean_stay[L]
        ranked = sorted(range(num_heads), key=lambda h: -diff[h])
        print(f"  L{L}: " + "  ".join(f"H{h}={diff[h]:+.4f}" for h in ranked[:6]))

# ── Section 3: MESH analysis for top routing head ────────────────────────────
print(f"\n{'='*65}")
print(f"Section 3 — MESH Rank Analysis for Routing Head")
print(f"  Q×K^T rank: is the routing head a narrow-beam selector?")
print(f"{'='*65}")

def mesh_analysis(layer_idx, head_idx):
    """Extract W_q and W_k for a specific head, compute MESH = W_q^T @ W_k,
       return singular values and ratio."""
    # Qwen2 layer structure
    layer = model.model.layers[layer_idx]
    attn  = layer.self_attn

    # W_q: (hidden, num_heads * head_dim), W_k: (num_kv_heads * head_dim, hidden)
    W_q = attn.q_proj.weight.detach().numpy()  # (num_heads*head_dim, hidden)
    W_k = attn.k_proj.weight.detach().numpy()  # (num_kv_heads*head_dim, hidden)

    # Extract this head's slice
    # KV head index for Q head h: h // (num_heads // num_kv)
    kv_group = num_heads // num_kv   # = 6
    kv_head  = head_idx // kv_group

    q_start = head_idx * head_dim
    q_end   = q_start + head_dim
    k_start = kv_head * head_dim
    k_end   = k_start + head_dim

    Wq_h = W_q[q_start:q_end, :]   # (head_dim, hidden)
    Wk_h = W_k[k_start:k_end, :]   # (head_dim, hidden)

    # MESH = Wq_h.T @ Wk_h  shape: (hidden, hidden)  — too large for SVD
    # Instead: SVD of Wq_h and Wk_h separately, then their "cross" singular values
    # MESH singular values ≈ σ_q * σ_k for aligned directions
    _, sq, Uq = np.linalg.svd(Wq_h, full_matrices=False)  # Uq: (head_dim, head_dim)
    _, sk, Uk = np.linalg.svd(Wk_h, full_matrices=False)

    # Cross-alignment: top-1 vs rest
    # MESH top SV ≈ sq[0] * sk[0] * |cos(Uq[:,0], Uk[:,0])|
    cos_top = abs(float(np.dot(Uq[0], Uk[0])))
    ratio_sq = sq[0] / sq[1] if len(sq) > 1 else float('inf')
    ratio_sk = sk[0] / sk[1] if len(sk) > 1 else float('inf')

    return {
        'sq': sq[:8].tolist(), 'sk': sk[:8].tolist(),
        'cos_top_dirs': cos_top,
        'sq_ratio_s0_s1': ratio_sq,
        'sk_ratio_s0_s1': ratio_sk,
        'effective_rank_q': float(np.exp(-np.sum((sq/sq.sum()) * np.log(sq/sq.sum() + 1e-20)))),
    }

# Analyse routing heads at L14 and L23
print(f"\n  All heads at L14 and L23 (cos alignment of top Q and K directions):")
for L in [14, 23]:
    print(f"\n  Layer {L}:")
    print(f"  Head  cos(q0,k0)  sq_ratio  sk_ratio  eff_rank_q")
    for h in range(num_heads):
        m = mesh_analysis(L, h)
        print(f"  H{h:02d}   {m['cos_top_dirs']:.4f}     {m['sq_ratio_s0_s1']:.2f}      "
              f"{m['sk_ratio_s0_s1']:.2f}      {m['effective_rank_q']:.1f}")

# ── Section 4: Cross-lingual routing signature ───────────────────────────────
print(f"\n{'='*65}")
print(f"Section 4 — Cross-Lingual Routing Signature")
print(f"  Do English split-token words use the same routing heads?")
print(f"{'='*65}")

# English 2-token test cases — check their tokenization and zone
eng_cases = ['singing', 'killing', 'driving', 'writing', 'running', 'walking',
             'eating', 'building', 'playing', 'reading']
print(f"\n  English token splits and zone assignments:")
for word in eng_cases:
    ids = tok(word, add_special_tokens=False)['input_ids']
    n   = len(ids)
    token_strs_w = tok.convert_ids_to_tokens(ids)
    token_strs_w_full, hs_w, _ = extract_all_positions(word)
    if n >= 2:
        zones_pos = []
        for pos in range(len(token_strs_w_full)):  # all positions, no BOS skip
            phi_v = phi_single(hs_w[14][pos])
            z, sims, lbl, sim = assign_zone(phi_v)
            zones_pos.append(f"pos{pos}→{z}({sims['C']:.2f}C)")
        print(f"  {word:<12s} n={n}  tokens={token_strs_w}  zones: {' | '.join(zones_pos)}")
        profile = get_attn_last_to_first(word)
        if profile is not None:
            attn_profiles[word] = profile
    else:
        # single token — position 0 is the word
        phi_v = phi_single(hs_w[14][0])
        z, sims, lbl, sim = assign_zone(phi_v)
        print(f"  {word:<12s} n={n}  tokens={token_strs_w}  zone: {z}({sims['C']:.2f}C)")

# Compare routing signature: Chinese 着-forms vs English split -ing forms
eng_2tok_esc  = [w for w in eng_cases
                 if len(tok(w, add_special_tokens=False)['input_ids']) == 2
                 and w in attn_profiles]
print(f"\n  English 2-token words with routing profiles: {eng_2tok_esc}")

if eng_2tok_esc and escaping_2tok:
    mean_zh_esc  = mean_esc if esc_texts else None
    mean_eng_esc = np.stack([attn_profiles[w] for w in eng_2tok_esc]).mean(0)

    print(f"\n  Correlation of attention profiles (Chinese 着-escape vs English split-word):")
    print(f"  Layer  corr(ZH_esc, EN_esc)")
    for L in PROBE_LAYERS:
        if L >= mean_eng_esc.shape[0] or mean_zh_esc is None:
            continue
        if mean_zh_esc.shape[0] <= L:
            continue
        r = np.corrcoef(mean_zh_esc[L], mean_eng_esc[L])[0, 1]
        print(f"  L{L:02d}   r={r:.4f}")

    # Head-by-head comparison at L14 and L23
    print(f"\n  Head-by-head attention [last→first] at L14 and L23:")
    print(f"  {'Head':<6s} {'ZH_esc':<10s} {'EN_esc':<10s}  diff")
    for L in [14, 23]:
        print(f"  [Layer {L}]")
        if L < mean_zh_esc.shape[0] and L < mean_eng_esc.shape[0]:
            for h in range(num_heads):
                zh_v  = mean_zh_esc[L][h]  if mean_zh_esc is not None else 0
                en_v  = mean_eng_esc[L][h]
                diff  = zh_v - en_v
                print(f"  H{h:02d}   {zh_v:.4f}    {en_v:.4f}    {diff:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj

output = {
    'meta': {'experiment': 'Day 41 — Routing Head Analysis'},
    'section1': to_serializable(section1_results),
    'attn_profiles': to_serializable(attn_profiles),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 41 complete.")
