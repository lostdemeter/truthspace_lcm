#!/usr/bin/env python3
"""
Expedition Day 31 — Zone D Structure at L23

Day 27 mapped the verb ocean B000 at L14 (8,778 words) and L23 (9,528 words).
Day 28 sub-clustered B000 at L14 but coherence was not stored.

Day 31 asks: does Zone D compress at L23?
  - Are tighter sub-bodies achievable at L23 vs L14?
  - Does the large residual ocean shrink or grow?
  - Do the same thematic sub-bodies survive to L23?

Approach (pure matrix ops — no forward passes except Z2 build):
  1. Load all 16,978 L14 + L23 hidden states from day27_hs_cache.npz
  2. Build Z2 axes from cached hidden states (no forward pass needed)
  3. Compute phi-vectors at L14 and L23
  4. Identify degenerate pole; isolate Phase 2 words
  5. Cluster Phase 2 at both layers; compare body structure
  6. Sub-cluster Zone D at L23 with two merge thresholds
  7. Direct coherence comparison L14 vs L23 for same word groups
  8. Label top sub-bodies with Ollama
"""

import sys, os, json, time
import numpy as np
import urllib.request

SMALL_MODEL     = "Qwen/Qwen2-1.5B-Instruct"
CACHE_FILE      = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE      = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE     = os.path.join(os.path.dirname(__file__), "day31_zone_d_l23.json")
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5:14b"

POLE_COS_THRESH = 0.90
K_FULL          = 150
K_ZONED         = 250
MERGE_COS_LOW   = 0.82
MERGE_COS_HIGH  = 0.88
TOP_N           = 20
N_LABEL         = 30

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-20))


def batch_phi(hs_matrix, z2):
    H  = hs_matrix.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)


def coherence_of(phi_rows):
    V  = np.array(phi_rows)
    c  = V.mean(axis=0)
    nm = np.linalg.norm(c)
    if nm < 1e-20:
        return 0.0
    return float(V @ (c / nm) / len(V) * len(V))


def coherence_of(phi_rows):
    V  = np.array(phi_rows)
    c  = V.mean(axis=0)
    nm = np.linalg.norm(c)
    if nm < 1e-20:
        return 0.0
    c /= nm
    return float(np.mean(V @ c))


def cluster_merge(X, words, k, merge_cos):
    from sklearn.cluster import MiniBatchKMeans
    k_eff = min(k, max(2, len(words) // 2))
    km = MiniBatchKMeans(n_clusters=k_eff, random_state=42,
                         n_init=5, batch_size=2048, max_iter=300)
    lbl = km.fit_predict(X.astype(np.float32))

    bodies = []
    for ci in range(k_eff):
        idx = np.where(lbl == ci)[0]
        if not len(idx):
            continue
        vecs = X[idx]
        c    = vecs.mean(axis=0)
        nm   = np.linalg.norm(c)
        if nm < 1e-20:
            continue
        c /= nm
        bodies.append({'words': [words[j] for j in idx],
                       'centroid': c, 'coherence': float(np.mean(vecs @ c))})

    changed = True
    while changed:
        changed = False
        merged  = [False] * len(bodies)
        nxt     = []
        for i in range(len(bodies)):
            if merged[i]:
                continue
            cur = dict(bodies[i]); cur['centroid'] = cur['centroid'].copy()
            for j in range(i + 1, len(bodies)):
                if merged[j]:
                    continue
                if cos_sim(cur['centroid'], bodies[j]['centroid']) >= merge_cos:
                    ni, nj      = len(cur['words']), len(bodies[j]['words'])
                    cur['words'] += bodies[j]['words']
                    cur['centroid'] = (ni * cur['centroid'] +
                                       nj * bodies[j]['centroid']) / (ni + nj)
                    nm = np.linalg.norm(cur['centroid'])
                    if nm > 1e-20: cur['centroid'] /= nm
                    merged[j] = True; changed = True
            merged[i] = True; nxt.append(cur)
        bodies = nxt

    for b in bodies:
        idx   = [words.index(w) for w in b['words']]
        vecs  = X[idx]
        c     = vecs.mean(axis=0); nm = np.linalg.norm(c)
        if nm < 1e-20: nm = 1.0
        c /= nm
        sims  = list(vecs @ c)
        order = np.argsort(sims)[::-1]
        b['coherence'] = float(np.mean(vecs @ c))
        b['centroid']  = c
        b['top_words'] = [b['words'][o] for o in order[:TOP_N]]
        b['size']      = len(b['words'])

    bodies.sort(key=lambda b: -b['size'])
    return bodies


def body_stats(bodies, label):
    sizes = [b['size'] for b in bodies]
    cohs  = sorted([b['coherence'] for b in bodies])
    total = sum(sizes)
    print(f"\n  {label}:")
    print(f"    n_bodies={len(bodies)}  total_words={total}  "
          f"largest={sizes[0] if sizes else 0} ({100*sizes[0]/total:.1f}% if {total} else 0)")
    if cohs:
        print(f"    coh: min={cohs[0]:.3f}  median={cohs[len(cohs)//2]:.3f}  "
              f"max={cohs[-1]:.3f}")
        print(f"    coh>0.85: {sum(1 for c in cohs if c>0.85)}  "
              f"coh>0.90: {sum(1 for c in cohs if c>0.90)}  "
              f"coh>0.95: {sum(1 for c in cohs if c>0.95)}")


def ollama_label(words, fallback):
    prompt = (
        f"These {len(words)} words are in the same semantic cluster:\n"
        f"{', '.join(words[:TOP_N])}\n"
        f"Give a SHORT label (2-5 words). Reply ONLY with the label."
    )
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt,
                           "stream": False,
                           "options": {"temperature": 0.0}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"].strip().strip('"\'')
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
t_start = time.time()

# Step 1: Load cache
print(f"\n── Step 1: Load cache ───────────────────────────────────────────")
npz      = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14     = npz['hs_14'].astype(np.float64)
hs23     = npz['hs_23'].astype(np.float64)
w2i      = {w: i for i, w in enumerate(words_all)}
print(f"  {len(words_all)} words  hs14={hs14.shape}  hs23={hs23.shape}")

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap        = atlas['word_map']
phase1_set  = {w for w, v in wmap.items() if v['phase'] == 1}
print(f"  Phase 1 words: {len(phase1_set)}")


# Step 2: Z2 from cache (Killing pairs already in cache)
print(f"\n── Step 2: Z2 axes from cache ───────────────────────────────────")

def z2_from_cache(hs_mat, w2i, pairs, name):
    deltas = []
    for a, b in pairs:
        for pfx in [' ', '']:
            wa, wb = pfx + a, pfx + b
            if wa in w2i and wb in w2i:
                d  = hs_mat[w2i[wb]] - hs_mat[w2i[wa]]
                dm = np.linalg.norm(d)
                if dm > 1e-20:
                    deltas.append(d / dm)
                break
    D = np.stack(deltas)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    z2  = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-20)
    pct = 100 * sv[0]**2 / (np.sum(sv**2) + 1e-20)
    print(f"  Z2 {name}: {pct:.1f}%  ({len(deltas)} deltas)")
    return z2

z2_14 = z2_from_cache(hs14, w2i, KILLING_PAIRS, "L14")
z2_23 = z2_from_cache(hs23, w2i, KILLING_PAIRS, "L23")


# Step 3: phi-vectors
print(f"\n── Step 3: phi-vectors ──────────────────────────────────────────")
phi14 = batch_phi(hs14, z2_14)
phi23 = batch_phi(hs23, z2_23)
print(f"  phi14={phi14.shape}  phi23={phi23.shape}")


# Step 4: Degenerate pole
print(f"\n── Step 4: Degenerate pole ──────────────────────────────────────")
p1_idx = [w2i[w] for w in phase1_set if w in w2i]
pole14 = phi14[p1_idx].mean(axis=0); pole14 /= np.linalg.norm(pole14)
pole23 = phi23[p1_idx].mean(axis=0); pole23 /= np.linalg.norm(pole23)

cos14  = phi14 @ pole14
cos23  = phi23 @ pole23

p2m14  = cos14 < POLE_COS_THRESH
p2m23  = cos23 < POLE_COS_THRESH
print(f"  L14: pole={int((~p2m14).sum())}  phase2={int(p2m14.sum())}")
print(f"  L23: pole={int((~p2m23).sum())}  phase2={int(p2m23.sum())}")


# Step 5: Full Phase 2 clustering at both layers
print(f"\n── Step 5: Full Phase 2 clustering (k={K_FULL}, merge≥{MERGE_COS_HIGH}) ──")

def run_cluster(phi_mat, mask, k, merge_cos, layer):
    idx   = np.where(mask)[0]
    X     = phi_mat[idx].astype(np.float32)
    wlist = [words_all[j] for j in idx]
    print(f"  {layer}: clustering {len(wlist)} words...")
    t0    = time.time()
    bods  = cluster_merge(X, wlist, k, merge_cos)
    print(f"  {layer}: {len(bods)} bodies ({time.time()-t0:.1f}s)")
    return bods

b14_full = run_cluster(phi14, p2m14, K_FULL, MERGE_COS_HIGH, "L14")
b23_full = run_cluster(phi23, p2m23, K_FULL, MERGE_COS_HIGH, "L23")

body_stats(b14_full, f"Full Phase 2 clustering L14 (merge≥{MERGE_COS_HIGH})")
body_stats(b23_full, f"Full Phase 2 clustering L23 (merge≥{MERGE_COS_HIGH})")


# Step 6: Zone D identification and overlap
print(f"\n── Step 6: Zone D identification ────────────────────────────────")
zd14 = b14_full[0]
zd23 = b23_full[0]
print(f"  Zone D L14: {zd14['size']} words  coh={zd14['coherence']:.4f}")
print(f"    top: {', '.join(zd14['top_words'][:10])}")
print(f"  Zone D L23: {zd23['size']} words  coh={zd23['coherence']:.4f}")
print(f"    top: {', '.join(zd23['top_words'][:10])}")

zd14_set  = set(zd14['words'])
zd23_set  = set(zd23['words'])
escaped   = zd14_set - zd23_set
fell_in   = zd23_set - zd14_set
print(f"\n  Zone D membership:")
print(f"    Both layers: {len(zd14_set & zd23_set)}")
print(f"    L14 only (escaped to Zone C at L23): {len(escaped)}")
print(f"    L23 only (fell from Zone C into Zone D at L23): {len(fell_in)}")

if escaped:
    print(f"\n  Escaped words (sample): {', '.join(sorted(escaped)[:25])}")
if fell_in:
    print(f"  Fell-in words (sample):  {', '.join(sorted(fell_in)[:25])}")


# Step 7: Zone D sub-clustering
print(f"\n── Step 7: Zone D sub-clustering ────────────────────────────────")

def subcluster_zd(zd_body, phi_mat, w2i, k, merge_cos, layer):
    wlist = [w for w in zd_body['words'] if w in w2i]
    idx   = [w2i[w] for w in wlist]
    X     = phi_mat[idx].astype(np.float32)
    print(f"  Sub-clustering {len(wlist)} words ({layer}, k={k}, merge≥{merge_cos})...")
    t0    = time.time()
    subs  = cluster_merge(X, wlist, k, merge_cos)
    print(f"  → {len(subs)} sub-bodies ({time.time()-t0:.1f}s)")
    return subs

subs14h = subcluster_zd(zd14, phi14, w2i, K_ZONED, MERGE_COS_HIGH, "L14 strict")
subs23h = subcluster_zd(zd23, phi23, w2i, K_ZONED, MERGE_COS_HIGH, "L23 strict")
subs23l = subcluster_zd(zd23, phi23, w2i, K_ZONED, MERGE_COS_LOW,  "L23 permissive")

body_stats(subs14h, f"Zone D sub-cluster L14 (merge≥{MERGE_COS_HIGH})")
body_stats(subs23h, f"Zone D sub-cluster L23 (merge≥{MERGE_COS_HIGH})")
body_stats(subs23l, f"Zone D sub-cluster L23 (merge≥{MERGE_COS_LOW})")


# Step 8: Direct coherence comparison for same word groups
print(f"\n── Step 8: Coherence comparison L14 vs L23 for same groups ─────")
comparison = []
# Use non-residual sub-bodies from L14 as reference groups
ref_bodies = [b for b in subs14h if b['size'] < zd14['size'] * 0.3]
for bi, b in enumerate(ref_bodies[:60]):
    wset = [w for w in b['top_words'] if w in w2i]
    if len(wset) < 3:
        continue
    idx = [w2i[w] for w in wset]
    coh14 = coherence_of(list(phi14[idx]))
    coh23 = coherence_of(list(phi23[idx]))
    comparison.append({'size': b['size'], 'coh14': coh14, 'coh23': coh23,
                        'delta': coh23 - coh14, 'words': wset[:6]})

comparison.sort(key=lambda x: -abs(x['delta']))
print(f"\n  Groups with largest coherence change L14→L23 (top 20):")
print(f"  {'n':>5}  {'coh14':>6}  {'coh23':>6}  {'Δ':>7}  Words")
print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*35}")
for c in comparison[:20]:
    sym = "▲" if c['delta'] > 0 else "▼"
    print(f"  {c['size']:>5}  {c['coh14']:>6.3f}  {c['coh23']:>6.3f}  "
          f"{sym}{abs(c['delta']):>6.3f}  {', '.join(c['words'])}")

if comparison:
    deltas = [c['delta'] for c in comparison]
    print(f"\n  Summary over {len(comparison)} groups:")
    print(f"    Improved (Δ>0.01):  {sum(1 for d in deltas if d>0.01)}")
    print(f"    Degraded (Δ<-0.01): {sum(1 for d in deltas if d<-0.01)}")
    print(f"    Mean Δ: {np.mean(deltas):+.4f}")


# Step 9: Label top L23 sub-bodies with Ollama
print(f"\n── Step 9: Ollama labelling ─────────────────────────────────────")
to_label = sorted([b for b in subs23h if b['size'] < zd23['size'] * 0.3],
                   key=lambda b: -b['size'])[:N_LABEL]
for i, b in enumerate(to_label):
    b['label'] = ollama_label(b['top_words'], f"ZD23-{i:03d}")
    if i < 8 or i % 5 == 0:
        print(f"  [{i+1:>3}/{len(to_label)}] n={b['size']:>4} "
              f"coh={b['coherence']:.3f}: {b['label']}")


# Step 10: Save
print(f"\n── Step 10: Save ────────────────────────────────────────────────")

def b2d(bodies):
    return [{'size': b['size'], 'coherence': round(b['coherence'], 4),
             'label': b.get('label', ''), 'top_words': b['top_words']}
            for b in bodies]

result = {
    "meta": {
        "experiment": "Day 31 — Zone D structure at L23",
        "n_words_total": len(words_all),
        "pole_cos_thresh": POLE_COS_THRESH,
        "n_phase2_L14": int(p2m14.sum()),
        "n_phase2_L23": int(p2m23.sum()),
    },
    "full_phase2_clustering": {
        "L14": {"n_bodies": len(b14_full),
                "zone_d_size": zd14['size'],
                "zone_d_coh": round(zd14['coherence'], 4),
                "n_coh_over_085": sum(1 for b in b14_full if b['coherence'] > 0.85)},
        "L23": {"n_bodies": len(b23_full),
                "zone_d_size": zd23['size'],
                "zone_d_coh": round(zd23['coherence'], 4),
                "n_coh_over_085": sum(1 for b in b23_full if b['coherence'] > 0.85)},
    },
    "zone_d_overlap": {
        "both_layers": len(zd14_set & zd23_set),
        "escaped_to_zone_c_at_L23": len(escaped),
        "fell_into_zone_d_at_L23": len(fell_in),
        "escaped_sample": sorted(escaped)[:40],
        "fell_in_sample": sorted(fell_in)[:40],
    },
    "sub_clustering": {
        "L14_strict":      b2d(subs14h),
        "L23_strict":      b2d(subs23h),
        "L23_permissive":  b2d(subs23l),
    },
    "coherence_comparison": [
        {"size": c['size'], "coh14": round(c['coh14'], 4),
         "coh23": round(c['coh23'], 4), "delta": round(c['delta'], 4),
         "words": c['words']} for c in comparison
    ],
    "labelled_L23_sub_bodies": b2d(to_label),
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {OUTPUT_FILE}")

print(f"\n{'='*65}")
print(f"DAY 31 SUMMARY — Zone D at L23")
print(f"{'='*65}")
print(f"  Phase 2 words:  L14={int(p2m14.sum())}  L23={int(p2m23.sum())}")
print(f"  Zone D size:    L14={zd14['size']}  L23={zd23['size']}")
print(f"  Zone D coh:     L14={zd14['coherence']:.4f}  "
      f"L23={zd23['coherence']:.4f}")
print(f"  Sub-bodies (strict merge):  L14={len(subs14h)}  L23={len(subs23h)}")
print(f"  Sub-bodies (permissive):    L23={len(subs23l)}")
if comparison:
    deltas = [c['delta'] for c in comparison]
    print(f"  Mean coh change L14→L23: {np.mean(deltas):+.4f}")
print(f"\nDay 31 complete in {(time.time()-t_start)/60:.1f} min")
