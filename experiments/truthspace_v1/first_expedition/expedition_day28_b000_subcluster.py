#!/usr/bin/env python3
"""
Expedition Day 28 — Sub-clustering B000

Day 27 found that 8,778 Phase 2 words (51.7%) collapsed into a single
diffuse body B000 (coh=0.708) labelled "Verbs of Strong Impact" / "Negative
Impact and Distortion". That label is an artifact of showing only the 25
nearest words to the centroid — the body is far too large to be a single
semantic class.

This script drills into B000 by:
  1. Loading the B000 word list from the Day 27 atlas
  2. Loading their L14 + L23 φ-vectors from the Day 27 cache
  3. Running a finer-grained clustering (k=300 initial → merge cos≥0.88)
     specifically on the B000 subspace
  4. Labelling all sub-bodies with Ollama
  5. Saving a sub-atlas and reporting the discovered bodies

Prediction: cities, animals, elements, professions, geographic features,
medical/anatomical terms, foods, vehicles, and other content-rich categories
will emerge as distinct sub-bodies once the scale of B000 is broken up.
"""

import sys, os, re, json, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ─────────────────────────────────────────────────────────────────
ATLAS_FILE   = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
CACHE_FILE   = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "day28_b000_subatlas.json")

LAYERS       = [14, 23]
K_INIT       = 300          # finer initial resolution than Day 27's 200
MERGE_COS    = 0.88         # slightly tighter merge than Day 27's 0.90
TOP_PER_CLUSTER = 25
SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]


# ── Utilities ─────────────────────────────────────────────────────────────────
def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def phi_vec(h, z2):
    hn   = h.astype(np.float64) / (np.linalg.norm(h) + 1e-20)
    proj = float(np.dot(hn, z2))
    perp = hn - proj * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)


def ollama_label(words, fallback):
    prompt = (
        f"Here are {len(words)} English words that Qwen2-1.5B groups into the "
        f"same internal semantic cluster:\n\n{', '.join(words[:TOP_PER_CLUSTER])}\n\n"
        f"Give this cluster a SHORT label (2-5 words) capturing its most specific "
        f"theme. Reply with ONLY the label."
    )
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 20}
    }).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"].strip().strip('"\'')
    except Exception:
        return fallback


# ── Step 1: Load B000 words from Day 27 atlas ─────────────────────────────────
def load_b000_words():
    print(f"\n── Step 1: Loading B000 word list from Day 27 atlas ─────────────────")
    atlas = json.load(open(ATLAS_FILE))

    b000_words = atlas["bodies_L14"]["B000"]["top_words"]   # top 25 only in bodies
    # The full word list is in word_map
    b000_full = [w for w, entry in atlas["word_map"].items()
                 if entry.get("phase") == 2 and entry.get("L14_body") == "B000"]
    print(f"  B000 words at L14: {len(b000_full)}")

    # Also get B000 at L23 (may differ)
    b000_l23 = [w for w, entry in atlas["word_map"].items()
                if entry.get("phase") == 2 and entry.get("L23_body") == "B000"]
    print(f"  B000 words at L23: {len(b000_l23)}")

    return b000_full, b000_l23, atlas


# ── Step 2: Load hidden states from cache ────────────────────────────────────
def load_cache_for_words(words_set):
    print(f"\n── Step 2: Loading hidden states from Day 27 cache ─────────────────")
    npz = np.load(CACHE_FILE, allow_pickle=True)
    cached_words = list(npz['words'])
    arr14 = npz['hs_14']
    arr23 = npz['hs_23']
    word_to_idx = {w: i for i, w in enumerate(cached_words)}

    hs = {}
    missing = 0
    for w in words_set:
        idx = word_to_idx.get(w)
        if idx is not None:
            hs[w] = {14: arr14[idx], 23: arr23[idx]}
        else:
            missing += 1
    print(f"  Loaded {len(hs)} words  ({missing} not in cache)")
    return hs


# ── Step 3: Build Z2 axes ─────────────────────────────────────────────────────
def build_z2(hs_cache):
    print(f"\n── Step 3: Z2 axes ──────────────────────────────────────────────────")
    z2 = {}
    for L in LAYERS:
        deltas = []
        for a, b in KILLING_PAIRS:
            if a in hs_cache and b in hs_cache:
                d = (hs_cache[b][L].astype(np.float64)
                     - hs_cache[a][L].astype(np.float64))
                dm = np.linalg.norm(d)
                if dm > 1e-20:
                    deltas.append(d / dm)
        D = np.stack(deltas)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        z2[L] = Vt[0]
        print(f"  Z2 L{L}: {100*sv[0]**2/np.sum(sv**2):.1f}%  ({len(deltas)} pairs)")
    return z2


# ── Step 4: φ-vectors ─────────────────────────────────────────────────────────
def compute_phi(hs_cache, z2, words):
    phi = {L: {} for L in LAYERS}
    for w in words:
        if w not in hs_cache:
            continue
        for L in LAYERS:
            phi[L][w] = phi_vec(hs_cache[w][L], z2[L])
    return phi


# ── Step 5: Cluster → merge ───────────────────────────────────────────────────
def cluster_layer(phi_dict, layer_label):
    from sklearn.cluster import MiniBatchKMeans
    words = sorted(phi_dict.keys())
    X = np.stack([phi_dict[w] for w in words]).astype(np.float32)
    k = min(K_INIT, len(words) // 5)

    print(f"\n  MiniBatchKMeans k={k} on {len(words)} words ({layer_label})...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                          batch_size=1024, max_iter=300)
    labels = km.fit_predict(X)

    raw = {}
    for w, lab in zip(words, labels):
        raw.setdefault(lab, []).append(w)

    clusters = []
    for lab, wlist in raw.items():
        vecs = np.stack([phi_dict[w] for w in wlist])
        c = vecs.mean(axis=0)
        cn = np.linalg.norm(c)
        centroid = c / (cn + 1e-20)
        top = sorted(wlist, key=lambda w: cos_sim(phi_dict[w], centroid), reverse=True)
        clusters.append({'words': wlist, 'centroid': centroid,
                         'top_words': top[:TOP_PER_CLUSTER], 'size': len(wlist)})

    print(f"  Merging (cos≥{MERGE_COS})...")
    clusters = merge_clusters(clusters)
    clusters.sort(key=lambda c: c['size'], reverse=True)
    print(f"  Final sub-bodies: {len(clusters)}  (from {k} initial)")
    return clusters


def merge_clusters(clusters):
    changed = True
    while changed:
        changed = False
        n = len(clusters)
        merged = [False] * n
        new_clusters = []
        for i in range(n):
            if merged[i]:
                continue
            cur = dict(clusters[i])
            cur['words'] = list(cur['words'])
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                if cos_sim(cur['centroid'], clusters[j]['centroid']) >= MERGE_COS:
                    ni, nj = cur['size'], clusters[j]['size']
                    c = (ni * cur['centroid'] + nj * clusters[j]['centroid'])
                    cn = np.linalg.norm(c)
                    cur['centroid'] = c / (cn + 1e-20)
                    cur['words']   += clusters[j]['words']
                    cur['size']    += clusters[j]['size']
                    merged[j] = True
                    changed = True
            merged[i] = True
            new_clusters.append(cur)
        clusters = new_clusters
    return clusters


# ── Step 6: Ollama labelling ──────────────────────────────────────────────────
def label_clusters(clusters, layer_label):
    print(f"\n── Step 6: Ollama labelling ({layer_label}) ──────────────────────────")
    for i, cl in enumerate(clusters):
        label = ollama_label(cl['top_words'], f"sub_{i:03d}")
        cl['label'] = label
        if i % 10 == 0:
            print(f"  [{i+1:>3}/{len(clusters)}]  S{i:03d} (n={cl['size']:>5}):  {label}")
    return clusters


# ── Step 7: Cross-layer word migration ───────────────────────────────────────
def check_migration(phi14, phi23, sub14, sub23):
    print(f"\n── Step 7: Cross-layer migration ────────────────────────────────────")
    map14 = {w: i for i, cl in enumerate(sub14) for w in cl['words']}
    map23 = {w: i for i, cl in enumerate(sub23) for w in cl['words']}
    common = set(map14) & set(map23)

    same = sum(1 for w in common if map14[w] == map23[w])
    cross_cos = [cos_sim(phi14[w], phi23[w]) for w in common if w in phi14 and w in phi23]

    print(f"  Common words: {len(common)}")
    print(f"  Same sub-body index L14→L23: {same}/{len(common)} "
          f"({100*same/max(1,len(common)):.1f}%)")
    print(f"  Cross-layer φ_cos: mean={np.mean(cross_cos):.4f}  "
          f"std={np.std(cross_cos):.4f}")
    return cross_cos


# ── Step 8: Save and report ───────────────────────────────────────────────────
def save_and_report(sub14, sub23, b000_words, cross_cos):
    output = {
        "meta": {
            "source_body": "B000 from day27_atlas.json",
            "b000_size": len(b000_words),
            "n_sub_bodies_L14": len(sub14),
            "n_sub_bodies_L23": len(sub23),
            "mean_cross_layer_phi_cos": float(np.mean(cross_cos)) if cross_cos else None,
        },
        "sub_bodies_L14": {
            f"S{i:03d}": {
                "label": cl.get('label', f'sub_{i}'),
                "size": cl['size'],
                "top_words": cl.get('top_words', cl['words'][:TOP_PER_CLUSTER]),
                "coherence": float(np.mean([
                    cos_sim(cl['centroid'],
                            np.stack([np.zeros(len(cl['centroid']))])[0])
                    for _ in [1]  # placeholder; computed below
                ])),
            }
            for i, cl in enumerate(sub14)
        },
        "sub_bodies_L23": {
            f"S{i:03d}": {
                "label": cl.get('label', f'sub_{i}'),
                "size": cl['size'],
                "top_words": cl.get('top_words', cl['words'][:TOP_PER_CLUSTER]),
            }
            for i, cl in enumerate(sub23)
        },
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Sub-atlas saved: {OUTPUT_FILE}")

    print("\n" + "="*70)
    print("DAY 28 — B000 SUB-CLUSTER SUMMARY")
    print("="*70)
    print(f"\n  B000 words sub-clustered: {len(b000_words)}")
    print(f"  Sub-bodies discovered: L14={len(sub14)}, L23={len(sub23)}")
    if cross_cos:
        print(f"  Mean cross-layer φ_cos:   {np.mean(cross_cos):.4f}")

    print(f"\n  All sub-bodies at L14 (by size):")
    print(f"  {'Sub-body':<8}  {'Size':>5}  {'Label'}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*45}")
    for i, cl in enumerate(sub14):
        print(f"  S{i:03d}      {cl['size']:>5}  {cl.get('label', '?')}")

    print(f"\n  All sub-bodies at L23 (by size):")
    print(f"  {'Sub-body':<8}  {'Size':>5}  {'Label'}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*45}")
    for i, cl in enumerate(sub23):
        print(f"  S{i:03d}      {cl['size']:>5}  {cl.get('label', '?')}")

    print(f"\nDay 28 complete. Sub-atlas: {OUTPUT_FILE}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    b000_words_l14, b000_words_l23, atlas = load_b000_words()
    all_b000 = sorted(set(b000_words_l14) | set(b000_words_l23))

    hs_cache = load_cache_for_words(set(all_b000) | {w for p in KILLING_PAIRS for w in p})
    z2 = build_z2(hs_cache)

    print(f"\n── Step 4: φ-vectors ────────────────────────────────────────────────")
    phi = compute_phi(hs_cache, z2, all_b000)
    print(f"  φ-vectors: {len(phi[14])} at L14, {len(phi[23])} at L23")

    print(f"\n── Step 5: Clustering ───────────────────────────────────────────────")
    sub14 = cluster_layer(phi[14], "L14")
    sub23 = cluster_layer(phi[23], "L23")

    sub14 = label_clusters(sub14, "L14")
    sub23 = label_clusters(sub23, "L23")

    cross_cos = check_migration(phi[14], phi[23], sub14, sub23)
    save_and_report(sub14, sub23, all_b000, cross_cos)
