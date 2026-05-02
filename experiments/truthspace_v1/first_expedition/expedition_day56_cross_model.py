#!/usr/bin/env python3
"""
Expedition Day 56 — Cross-Model Universality

The central question of DC 318: is the Zone C geometry universal?
Do the same semantic bodies appear in a different model's φ-space?

If the geometry is intrinsic to language (not the specific model), then:
  - The inter-body distance matrix in Model A should correlate strongly
    with the inter-body distance matrix in Model B
  - T2 operators should point in similar directions across models
  - The body centroid topology should be preserved

Two comparison models:
  M1: Qwen2-1.5B-Instruct  L14  (baseline — Zone C was built here)
  M2: Qwen2-0.5B            L12  (same family, different size — strongest test)
  M3: microsoft/phi-2       L16  (different architecture — hardest test)

Strategy: use the SAME body assignments from Day 27 (word→body mapping).
For each model, compute body centroids in that model's φ-space.
Compare: Spearman correlation of inter-body distance matrices.
"""

import json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day56_cross_model.json")

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

COMPARISON_MODELS = [
    # (model_id, layer, cache_suffix, label)
    ("Qwen/Qwen2-0.5B",                 12, "qwen05_L12",  "Qwen2-0.5B  L12"),
    ("microsoft/phi-2",                  16, "phi2_L16",    "Phi-2       L16"),
]

print("=" * 70)
print("  Expedition Day 56 — Cross-Model Universality")
print("=" * 70)


# ── Helper functions ──────────────────────────────────────────────────────────
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

def build_z2(pairs, hs_dict):
    ds = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx+a, pfx+b
            if wa in hs_dict and wb in hs_dict:
                d  = hs_dict[wb].astype(np.float64) - hs_dict[wa].astype(np.float64)
                nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    _, _, Vt = np.linalg.svd(np.stack(ds), full_matrices=False)
    return Vt[0] / np.linalg.norm(Vt[0])

def to_phi_batch(hs_mat, z2):
    H   = hs_mat.astype(np.float64)
    nm  = np.linalg.norm(H, axis=1, keepdims=True)
    Hn  = H / (nm + 1e-20)
    perp = Hn - (Hn @ z2)[:, None] * z2
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def compute_body_centroids(words, bodies, phi_mat, w2i, min_members=3):
    body_members = defaultdict(list)
    for w, b in zip(words, bodies):
        if w in w2i: body_members[b].append(w2i[w])
    centroids = {}
    for body, idxs in body_members.items():
        if len(idxs) < min_members: continue
        v  = phi_mat[idxs].mean(0)
        nm = np.linalg.norm(v)
        if nm > 1e-20: centroids[body] = v / nm
    return centroids

def inter_body_distance_matrix(centroids):
    bodies  = sorted(centroids.keys())
    n       = len(bodies)
    D       = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            c = cosine(centroids[bodies[i]], centroids[bodies[j]])
            D[i, j] = D[j, i] = 1.0 - c
    return bodies, D

def extract_hidden_states(model_id, layer, words_with_prefix, cache_path):
    """
    Extract hidden state at `layer` for each word.
    Uses the word's single-token representation (space-prefixed if needed).
    Returns dict: word_key → hidden_state_vector
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if cache_path.exists():
        print(f"  Cache found: {cache_path}")
        npz = np.load(str(cache_path), allow_pickle=True)
        return dict(zip(list(npz['words']), npz['hs']))

    print(f"  Loading model {model_id} ...")
    tok   = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    hs_dict = {}
    n       = len(words_with_prefix)
    t0      = time.time()

    for i, word_key in enumerate(words_with_prefix):
        if i % 300 == 0 and i > 0:
            elapsed = time.time() - t0
            eta     = (n - i) / (i / elapsed)
            print(f"  [{i:>5}/{n}]  {elapsed/60:.1f} min  ETA {eta/60:.1f} min")

        ids = tok.encode(word_key, add_special_tokens=False)
        if len(ids) != 1:
            continue   # skip multi-token words in this model

        inputs = tok(word_key, return_tensors='pt')
        id_list = inputs['input_ids'][0]
        pos = next((j for j, t in enumerate(id_list) if t.item() == ids[0]),
                   len(id_list) - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        n_layers = len(out.hidden_states) - 1   # exclude embedding layer
        actual_layer = min(layer, n_layers)
        hs_dict[word_key] = out.hidden_states[actual_layer][0, pos, :].numpy().astype(np.float32)

    print(f"  Extracted {len(hs_dict)}/{n} words in {(time.time()-t0)/60:.1f} min")
    del model   # free memory

    words_arr = list(hs_dict.keys())
    hs_arr    = np.stack([hs_dict[w] for w in words_arr])
    np.savez_compressed(str(cache_path), words=words_arr, hs=hs_arr)
    print(f"  Cached: {cache_path}")
    return hs_dict


# ── Load M1 baseline (Qwen2-1.5B L14) ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"M1 — Baseline: Qwen2-1.5B L14")
print(f"{'='*70}")

npz       = np.load(CACHE_FILE, allow_pickle=True)
m1_words  = list(npz['words'])
m1_hs     = npz['hs_14'].astype(np.float64)
m1_w2i    = {w: i for i, w in enumerate(m1_words)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words  = [w for w, v in wmap.items()
                 if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None)
                 and w in m1_w2i]
zone_c_bodies = [wmap[w]['L14_body'] for w in zone_c_words]

print(f"  Zone C words: {len(zone_c_words)}")
print(f"  Bodies: {len(set(zone_c_bodies))}")

# Build M1 φ-space
m1_z2 = build_z2(KILLING_PAIRS, {w: m1_hs[m1_w2i[w]] for w in m1_words
                                   if w in m1_w2i})
m1_phi = to_phi_batch(m1_hs, m1_z2)

m1_zc_idx = np.array([m1_w2i[w] for w in zone_c_words])
m1_centroids = compute_body_centroids(
    zone_c_words, zone_c_bodies,
    m1_phi, {w: i for i, w in enumerate(zone_c_words)},
    min_members=4
)
m1_bodies_list, m1_D = inter_body_distance_matrix(m1_centroids)
print(f"  Bodies with ≥4 members: {len(m1_centroids)}")
print(f"  Inter-body distance matrix: {len(m1_bodies_list)} × {len(m1_bodies_list)}")


# ── Build T2 operators for M1 ────────────────────────────────────────────────
T2_SEEDS = {
    'male_female':     [(' king',' queen'),(' man',' woman'),(' boy',' girl'),
                        (' actor',' actress'),(' prince',' princess')],
    'singular_plural': [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees'),
                        (' bird',' birds'),(' book',' books')],
    'base_comp':       [(' big',' bigger'),(' fast',' faster'),(' old',' older'),
                        (' small',' smaller'),(' tall',' taller')],
}

def build_t2_from_phi(seeds, phi_mat, w2i):
    ds = []
    for a, b in seeds:
        for pfx in ['', ' ']:
            wa, wb = pfx+a.strip(), pfx+b.strip()
            if wa in w2i and wb in w2i:
                pa = phi_mat[w2i[wa]]; pb = phi_mat[w2i[wb]]
                d  = pb - pa; nm = np.linalg.norm(d)
                if nm > 1e-20: ds.append(d / nm)
                break
    if not ds: return None
    m = np.stack(ds).mean(0); nm = np.linalg.norm(m)
    return m / nm if nm > 1e-20 else None

m1_t2 = {k: build_t2_from_phi(v, m1_phi, m1_w2i) for k, v in T2_SEEDS.items()}
print(f"  M1 T2 operators built: {sum(1 for v in m1_t2.values() if v is not None)}")


# ── Compare with each model ───────────────────────────────────────────────────
all_results = {}

for model_id, layer, cache_sfx, label in COMPARISON_MODELS:
    print(f"\n{'='*70}")
    print(f"  {label}  ({model_id})")
    print(f"{'='*70}")

    cache_path = SCRIPT_DIR / f"day56_{cache_sfx}_cache.npz"

    # Extract or load hidden states
    words_to_extract = list({w for w in zone_c_words}
                             | {a for pairs in KILLING_PAIRS for a in pairs}
                             | {b for pairs in KILLING_PAIRS for b in pairs})
    # Add space-prefixed variants
    words_to_extract_full = list({pfx + w.lstrip()
                                   for w in words_to_extract
                                   for pfx in [' ', '']})

    hs_dict = extract_hidden_states(model_id, layer, words_to_extract_full, cache_path)

    if not hs_dict:
        print(f"  No hidden states extracted — skipping")
        continue

    # Build Z2 for this model
    mx_z2 = build_z2(KILLING_PAIRS, hs_dict)
    if mx_z2 is None:
        print(f"  Could not build Z2 axis — skipping")
        continue

    # Map zone C words to this model
    mx_words_available = []
    mx_hs_available    = []
    mx_bodies_available = []
    for w, b in zip(zone_c_words, zone_c_bodies):
        found = None
        for pfx in ['', ' ']:
            wk = pfx + w.lstrip()
            if wk in hs_dict: found = wk; break
        if found:
            mx_words_available.append(found)
            mx_hs_available.append(hs_dict[found].astype(np.float64))
            mx_bodies_available.append(b)

    coverage = len(mx_words_available) / len(zone_c_words)
    print(f"  Coverage: {len(mx_words_available)}/{len(zone_c_words)} = {coverage:.3f}")

    if len(mx_words_available) < 100:
        print(f"  Coverage too low — skipping")
        continue

    # Build φ-vectors for available words
    mx_hs_mat = np.stack(mx_hs_available)
    mx_phi    = to_phi_batch(mx_hs_mat, mx_z2)
    mx_w2i    = {w: i for i, w in enumerate(mx_words_available)}

    # Build body centroids for this model
    mx_centroids = compute_body_centroids(
        mx_words_available, mx_bodies_available,
        mx_phi, mx_w2i, min_members=4
    )
    print(f"  Bodies reconstructed: {len(mx_centroids)}")

    # Find common bodies
    common_bodies = sorted(set(m1_bodies_list) & set(mx_centroids.keys()))
    print(f"  Common bodies (≥4 members in both): {len(common_bodies)}")

    if len(common_bodies) < 4:
        print(f"  Too few common bodies — skipping")
        continue

    # Build aligned distance matrices
    m1_D_aligned = np.array([[m1_centroids[a] @ m1_centroids[b] /
                               (np.linalg.norm(m1_centroids[a]) * np.linalg.norm(m1_centroids[b]) + 1e-20)
                               for b in common_bodies] for a in common_bodies])
    mx_D_aligned = np.array([[mx_centroids[a] @ mx_centroids[b] /
                               (np.linalg.norm(mx_centroids[a]) * np.linalg.norm(mx_centroids[b]) + 1e-20)
                               for b in common_bodies] for a in common_bodies])

    # Upper triangle (no diagonal) as flat vector
    n_cb = len(common_bodies)
    idx_i, idx_j = np.triu_indices(n_cb, k=1)
    m1_flat = m1_D_aligned[idx_i, idx_j]
    mx_flat = mx_D_aligned[idx_i, idx_j]

    rho_spear, p_spear   = spearmanr(m1_flat, mx_flat)
    rho_pearson, p_pearson = pearsonr(m1_flat, mx_flat)

    print(f"\n  UNIVERSALITY METRICS (inter-body cosine similarity):")
    print(f"  Spearman  ρ = {rho_spear:.4f}  p = {p_spear:.2e}")
    print(f"  Pearson   r = {rho_pearson:.4f}  p = {p_pearson:.2e}")

    # T2 operator direction comparison
    print(f"\n  T2 OPERATOR ALIGNMENT (cos between M1 and {label} T2 vectors):")
    mx_t2  = {k: build_t2_from_phi(v, mx_phi, mx_w2i) for k, v in T2_SEEDS.items()}
    t2_alignments = {}
    for t2_name, m1_vec in m1_t2.items():
        mx_vec = mx_t2.get(t2_name)
        if m1_vec is None or mx_vec is None:
            continue
        # Project both to same dimensionality via the common φ-space directions
        # (we compare directions in the respective φ-spaces using test words)
        # Instead: compare the T2 operator's EFFECT on test words
        effects_m1, effects_mx = [], []
        for a, b in T2_SEEDS[t2_name][:4]:
            for pfx in ['', ' ']:
                wa = pfx + a.strip()
                wb = pfx + b.strip()
                if wa in m1_w2i and wb in m1_w2i:
                    effects_m1.append(cosine(m1_phi[m1_w2i[wa]], m1_phi[m1_w2i[wb]]))
                    break
            for pfx in ['', ' ']:
                wa = pfx + a.strip()
                wb = pfx + b.strip()
                if wa in mx_w2i and wb in mx_w2i:
                    effects_mx.append(cosine(mx_phi[mx_w2i[wa]], mx_phi[mx_w2i[wb]]))
                    break

        if len(effects_m1) >= 2 and len(effects_mx) >= 2:
            n_common = min(len(effects_m1), len(effects_mx))
            rho_t2, _ = spearmanr(effects_m1[:n_common], effects_mx[:n_common])
            mean_m1   = float(np.mean(effects_m1))
            mean_mx   = float(np.mean(effects_mx))
            print(f"    {t2_name:<20s}: M1 mean_cos={mean_m1:.4f}  "
                  f"{label[:12]} mean_cos={mean_mx:.4f}  "
                  f"pair_effect_ρ={rho_t2:.4f}")
            t2_alignments[t2_name] = {'mean_m1': mean_m1, 'mean_mx': mean_mx}

    # Per-body consistency: which bodies are most/least preserved?
    print(f"\n  PER-BODY DISTANCE PRESERVATION (top 5 most / least consistent):")
    body_errors = []
    for body in common_bodies:
        m1_row  = np.array([m1_centroids[body] @ m1_centroids[b] /
                             (np.linalg.norm(m1_centroids[body]) * np.linalg.norm(m1_centroids[b]) + 1e-20)
                             for b in common_bodies if b != body])
        mx_row  = np.array([mx_centroids[body] @ mx_centroids[b] /
                             (np.linalg.norm(mx_centroids[body]) * np.linalg.norm(mx_centroids[b]) + 1e-20)
                             for b in common_bodies if b != body])
        rho_b, _ = spearmanr(m1_row, mx_row)
        # Get body label from atlas
        label_b = '?'
        for w in zone_c_words:
            if wmap[w].get('L14_body') == body:
                lb = wmap[w].get('L14_label', '?')
                if lb and lb != '?': label_b = lb; break
        body_errors.append((body, float(rho_b), label_b))

    body_errors.sort(key=lambda x: -x[1])
    for body, rho_b, label_b in body_errors[:5]:
        print(f"    BEST  {body}  ρ={rho_b:.4f}  {label_b}")
    print(f"    ...")
    for body, rho_b, label_b in body_errors[-5:]:
        print(f"    WORST {body}  ρ={rho_b:.4f}  {label_b}")

    all_results[cache_sfx] = {
        'model':           model_id,
        'layer':           layer,
        'coverage':        float(coverage),
        'n_common_bodies': len(common_bodies),
        'spearman_rho':    float(rho_spear),
        'spearman_p':      float(p_spear),
        'pearson_r':       float(rho_pearson),
        'pearson_p':       float(p_pearson),
        't2_alignments':   t2_alignments,
        'per_body':        body_errors,
    }


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — Cross-Model Universality")
print(f"{'='*70}")
print(f"\n  Model                  Layer  Coverage  N_bodies  Spearman ρ  Pearson r")
print(f"  {'-'*70}")
print(f"  Qwen2-1.5B (baseline)   L14    1.000    {len(m1_centroids):>6d}   (reference)")
for sfx, res in all_results.items():
    label_s = res['model'].split('/')[-1][:22]
    print(f"  {label_s:<22s}  L{res['layer']:<2d}   {res['coverage']:.3f}   "
          f"{res['n_common_bodies']:>6d}   {res['spearman_rho']:>8.4f}    {res['pearson_r']:>8.4f}")

print(f"\n  Interpretation:")
for sfx, res in all_results.items():
    rho = res['spearman_rho']
    if rho > 0.9:
        verdict = "STRONGLY UNIVERSAL (> 0.9 Spearman)"
    elif rho > 0.7:
        verdict = "MODERATELY UNIVERSAL (0.7–0.9)"
    elif rho > 0.4:
        verdict = "WEAKLY UNIVERSAL (0.4–0.7)"
    else:
        verdict = "NOT UNIVERSAL (< 0.4)"
    print(f"  {res['model'].split('/')[-1]:<22s}: {verdict}")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_py(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, tuple): return list(x)
    return x

output = {
    'm1_baseline': {
        'model': 'Qwen/Qwen2-1.5B-Instruct',
        'layer': 14,
        'n_zone_c': len(zone_c_words),
        'n_bodies': len(m1_centroids),
    },
    'comparisons': to_py(all_results),
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 56 complete.")
