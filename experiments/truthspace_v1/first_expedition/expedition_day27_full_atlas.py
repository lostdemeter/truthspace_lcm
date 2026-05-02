#!/usr/bin/env python3
"""
Expedition Day 27 — Full Content Mapping

Map ALL clean English single-token words to gravitational bodies at
L14 (mid-COMB) and L23 (knowledge layer). Compare body membership
across layers to test whether the φ-geography is stable across COMB.

Pipeline:
  1. Load ~8000 clean English words from /usr/share/dict/words
  2. Phase pre-screen: syllables ≤ 1 → Phase 1 (common-word pole)
                       syllables > 1 → Phase 2 (semantic body zone)
  3. Run L14 + L23 forward passes for Phase 2 words (cached to disk)
  4. Build Z2 axis from Killing pairs at both layers
  5. Compute φ-vectors at L14 and L23 for all Phase 2 words
  6. Cluster Phase 2 φ-vectors: k=200 initial → merge (cos≥0.90)
  7. Label final bodies with Ollama (qwen2.5:14b)
  8. Save atlas to day27_atlas.json
  9. Report: body count, sizes, L14 vs L23 agreement, top bodies
"""

import sys, os, re, json, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ─────────────────────────────────────────────────────────────────
SMALL_MODEL     = "Qwen/Qwen2-1.5B-Instruct"
LAYERS          = [14, 23]
CACHE_FILE      = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE      = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
K_INIT          = 200
MERGE_COS       = 0.90
MIN_WORD_LEN    = 3
MAX_WORD_LEN    = 14
CRYST_LAYER     = 2
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5:14b"
TOP_PER_CLUSTER = 25
PHASE1_SYL      = 1        # syllables ≤ this → Phase 1

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]
KILLING_WORDS = {w for pair in KILLING_PAIRS for w in pair}


# ── Utility ───────────────────────────────────────────────────────────────────
def count_syllables(word):
    return max(1, len(re.findall(r'[aeiou]+', word.lower())))


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def phi_vec(h, z2):
    """Project h onto perp-Z2 sphere."""
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


# ── Section 1: Vocabulary ─────────────────────────────────────────────────────
def load_vocab(tok):
    print("\n── Section 1: Loading vocabulary ────────────────────────────────────")
    with open('/usr/share/dict/words') as f:
        dict_words = {
            w.strip().lower() for w in f
            if MIN_WORD_LEN <= len(w.strip()) <= MAX_WORD_LEN
            and "'" not in w.strip()
            and w.strip().isalpha()
            and w.strip() == w.strip().lower()
        }
    dict_words |= KILLING_WORDS        # ensure Killing pairs are always present
    print(f"  Dictionary words after filter: {len(dict_words)}")

    clean = []
    for word in sorted(dict_words):
        for variant in (' ' + word, word):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                clean.append((ids[0], word))
                break

    seen, deduped = set(), []
    for tid, word in clean:
        if tid not in seen:
            seen.add(tid)
            deduped.append((tid, word))
    deduped.sort(key=lambda x: x[0])

    print(f"  Single-token clean words: {len(deduped)}")
    return deduped   # [(token_id, word), ...]


# ── Section 2: Phase screening ────────────────────────────────────────────────
def screen_phases(vocab):
    print("\n── Section 2: Phase screening ───────────────────────────────────────")
    phase1, phase2 = [], []
    for item in vocab:
        if count_syllables(item[1]) <= PHASE1_SYL:
            phase1.append(item)
        else:
            phase2.append(item)
    print(f"  Phase 1 (syllables≤{PHASE1_SYL}): {len(phase1)} words")
    print(f"  Phase 2 (syllables>{PHASE1_SYL}): {len(phase2)} words")
    return phase1, phase2


# ── Section 3: Forward passes (with disk cache) ───────────────────────────────
def load_or_build_cache(model, tok, phase2):
    print("\n── Section 3: Hidden-state extraction ───────────────────────────────")
    words_p2 = [w for _, w in phase2]

    if os.path.exists(CACHE_FILE):
        print(f"  Found cache: {CACHE_FILE}")
        npz = np.load(CACHE_FILE, allow_pickle=True)
        cached_words = list(npz['words'])
        hs_by_word = {}
        arr14 = npz['hs_14']
        arr23 = npz['hs_23']
        for i, w in enumerate(cached_words):
            hs_by_word[w] = {14: arr14[i], 23: arr23[i]}
        missing = [item for item in phase2 if item[1] not in hs_by_word]
        if missing:
            print(f"  {len(missing)} words missing from cache — extracting...")
            new_cache = _extract(model, tok, missing)
            hs_by_word.update(new_cache)
            _save_cache(hs_by_word, words_p2)
        else:
            print(f"  Cache complete ({len(hs_by_word)} words).")
        return hs_by_word

    print(f"  No cache found. Extracting {len(phase2)} words...")
    hs_by_word = _extract(model, tok, phase2)
    _save_cache(hs_by_word, words_p2)
    return hs_by_word


def _extract(model, tok, words_list):
    import torch
    cache = {}
    n = len(words_list)
    t0 = time.time()
    for i, (tid, word) in enumerate(words_list):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = (n - i) / (i / elapsed)
            print(f"  [{i:>5}/{n}]  {elapsed/60:.1f} min elapsed  ETA {eta/60:.1f} min")
        inputs = tok(word, return_tensors='pt')
        id_list = inputs['input_ids'][0]
        pos = next((j for j, t in enumerate(id_list) if t.item() == tid),
                   len(id_list) - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        cache[word] = {L: out.hidden_states[L][0, pos, :].numpy().astype(np.float32)
                       for L in LAYERS}
    print(f"  Extracted {len(cache)} words in {(time.time()-t0)/60:.1f} min")
    return cache


def _save_cache(hs_by_word, words_p2):
    present = [w for w in words_p2 if w in hs_by_word]
    arr14 = np.stack([hs_by_word[w][14] for w in present])
    arr23 = np.stack([hs_by_word[w][23] for w in present])
    np.savez_compressed(CACHE_FILE, words=present, hs_14=arr14, hs_23=arr23)
    print(f"  Cache saved: {CACHE_FILE}  ({arr14.nbytes//1024//1024 * 2} MB)")


# ── Section 4: Z2 axes ────────────────────────────────────────────────────────
def build_z2_axes(hs_by_word):
    print("\n── Section 4: Z2 axis ───────────────────────────────────────────────")
    z2 = {}
    for L in LAYERS:
        deltas = []
        for a, b in KILLING_PAIRS:
            if a in hs_by_word and b in hs_by_word:
                d = (hs_by_word[b][L].astype(np.float64)
                     - hs_by_word[a][L].astype(np.float64))
                dm = np.linalg.norm(d)
                if dm > 1e-20:
                    deltas.append(d / dm)
        D = np.stack(deltas)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        z2[L] = Vt[0]
        var_pct = 100 * sv[0]**2 / np.sum(sv**2)
        print(f"  Z2 L{L}: {var_pct:.1f}% first singular vector  "
              f"({len(deltas)} delta vectors)")
    return z2


# ── Section 5: φ-vectors ──────────────────────────────────────────────────────
def compute_phi(hs_by_word, z2_axes, words_p2):
    print("\n── Section 5: φ-vector computation ─────────────────────────────────")
    phi = {L: {} for L in LAYERS}
    for word in words_p2:
        if word not in hs_by_word:
            continue
        for L in LAYERS:
            phi[L][word] = phi_vec(hs_by_word[word][L], z2_axes[L])
    print(f"  φ-vectors computed: {len(phi[14])} at L14, {len(phi[23])} at L23")
    return phi


# ── Section 6: Clustering ─────────────────────────────────────────────────────
def cluster_phi(phi_dict, layer_label):
    from sklearn.cluster import MiniBatchKMeans
    words = sorted(phi_dict.keys())
    X = np.stack([phi_dict[w] for w in words]).astype(np.float32)
    n = len(words)
    k = min(K_INIT, n // 5)

    print(f"\n  MiniBatchKMeans k={k} on {n} words ({layer_label})...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                          batch_size=1024, max_iter=300)
    labels = km.fit_predict(X)

    raw_clusters = {}
    for w, lab in zip(words, labels):
        raw_clusters.setdefault(lab, []).append(w)

    print(f"  Computing centroids and merging (cos≥{MERGE_COS})...")
    clusters = _build_clusters(raw_clusters, phi_dict)
    clusters = _merge_clusters(clusters, MERGE_COS)
    clusters.sort(key=lambda c: c['size'], reverse=True)
    print(f"  Final bodies: {len(clusters)}  (from {k} initial)")
    return clusters


def _build_clusters(raw_clusters, phi_dict):
    out = []
    for lab, wlist in raw_clusters.items():
        vecs = np.stack([phi_dict[w] for w in wlist if w in phi_dict])
        c = vecs.mean(axis=0)
        cn = np.linalg.norm(c)
        centroid = c / (cn + 1e-20)
        top = sorted(wlist, key=lambda w: cos_sim(phi_dict[w], centroid), reverse=True)
        out.append({'words': wlist, 'centroid': centroid,
                    'top_words': top[:TOP_PER_CLUSTER], 'size': len(wlist)})
    return out


def _merge_clusters(clusters, merge_cos):
    changed = True
    while changed:
        changed = False
        n = len(clusters)
        merged = [False] * n
        new_clusters = []
        for i in range(n):
            if merged[i]:
                continue
            current = dict(clusters[i])
            current['words'] = list(current['words'])
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                if cos_sim(current['centroid'], clusters[j]['centroid']) >= merge_cos:
                    ni, nj = current['size'], clusters[j]['size']
                    c = (ni * current['centroid'] + nj * clusters[j]['centroid'])
                    cn = np.linalg.norm(c)
                    current['centroid'] = c / (cn + 1e-20)
                    current['words'] += clusters[j]['words']
                    current['size'] += clusters[j]['size']
                    merged[j] = True
                    changed = True
            merged[i] = True
            new_clusters.append(current)
        clusters = new_clusters
    return clusters


# ── Section 7: Ollama labeling ────────────────────────────────────────────────
def label_clusters(clusters, layer_label):
    print(f"\n── Section 7: Ollama labeling ({layer_label}) ──────────────────────")
    for i, cl in enumerate(clusters):
        label = ollama_label(cl['top_words'], f"body_{i:03d}")
        cl['label'] = label
        if i % 10 == 0:
            print(f"  [{i+1:>3}/{len(clusters)}] B{i:03d}: {label:<35s}  "
                  f"(n={cl['size']})")
    return clusters


# ── Section 8: L14 vs L23 agreement ──────────────────────────────────────────
def layer_agreement(phi14, phi23, clusters14, clusters23):
    print("\n── Section 8: L14 vs L23 agreement ─────────────────────────────────")

    def assign(phi_dict, clusters):
        mapping = {}
        for i, cl in enumerate(clusters):
            for w in cl['words']:
                mapping[w] = i
        return mapping

    map14 = assign(phi14, clusters14)
    map23 = assign(phi23, clusters23)

    common_words = set(map14) & set(map23)
    body14_idx_of_w = {w: map14[w] for w in common_words}
    body23_idx_of_w = {w: map23[w] for w in common_words}
    body14_of_w = {w: clusters14[map14[w]]['label'] for w in common_words}
    body23_of_w = {w: clusters23[map23[w]]['label'] for w in common_words}
    exact_agree = 0  # not meaningful: labels come from independent Ollama runs

    # Cross-layer φ-cos per word
    cross_cos = [cos_sim(phi14[w], phi23[w]) for w in common_words
                 if w in phi14 and w in phi23]

    print(f"  Common words: {len(common_words)}")
    print(f"  Exact label agreement: {exact_agree}/{len(common_words)} "
          f"({100*exact_agree/max(1,len(common_words)):.1f}%)")
    print(f"  Cross-layer φ_cos: mean={np.mean(cross_cos):.4f}  "
          f"std={np.std(cross_cos):.4f}  min={np.min(cross_cos):.4f}")

    # Words that switch bodies
    switches = [(w, body14_of_w[w], body23_of_w[w])
                for w in common_words if body14_of_w[w] != body23_of_w[w]]
    print(f"  Words switching body L14→L23: {len(switches)}")
    for w, b14, b23 in sorted(switches, key=lambda x: x[0])[:20]:
        print(f"    {w:<16s}  L14: {b14:<25s}  L23: {b23}")

    return cross_cos, switches


# ── Section 9: Build and save atlas ──────────────────────────────────────────
def build_atlas(vocab, phase1, phase2, phi, clusters14, clusters23,
                cross_cos, switches):
    print("\n── Section 9: Building atlas ────────────────────────────────────────")

    def make_body_map(clusters):
        m = {}
        for i, cl in enumerate(clusters):
            for w in cl['words']:
                m[w] = i
        return m

    bmap14 = make_body_map(clusters14)
    bmap23 = make_body_map(clusters23)

    atlas = {
        "meta": {
            "model": SMALL_MODEL,
            "layers": LAYERS,
            "total_clean_words": len(vocab),
            "phase1_count": len(phase1),
            "phase2_count": len(phase2),
            "n_bodies_L14": len(clusters14),
            "n_bodies_L23": len(clusters23),
            "mean_cross_layer_phi_cos": float(np.mean(cross_cos)) if cross_cos else None,
        },
        "phase1_words": sorted([w for _, w in phase1]),
        "bodies_L14": {
            f"B{i:03d}": {
                "label": cl.get('label', f'body_{i}'),
                "size": cl['size'],
                "top_words": cl.get('top_words', cl['words'][:TOP_PER_CLUSTER]),
            }
            for i, cl in enumerate(clusters14)
        },
        "bodies_L23": {
            f"B{i:03d}": {
                "label": cl.get('label', f'body_{i}'),
                "size": cl['size'],
                "top_words": cl.get('top_words', cl['words'][:TOP_PER_CLUSTER]),
            }
            for i, cl in enumerate(clusters23)
        },
        "word_map": {},
    }

    for tid, word in phase1:
        atlas["word_map"][word] = {
            "phase": 1, "token_id": tid,
            "syllables": count_syllables(word),
        }

    for tid, word in phase2:
        if word not in phi[14] or word not in phi[23]:
            continue
        b14_idx = bmap14.get(word)
        b23_idx = bmap23.get(word)
        atlas["word_map"][word] = {
            "phase": 2, "token_id": tid,
            "syllables": count_syllables(word),
            "L14_body": f"B{b14_idx:03d}" if b14_idx is not None else None,
            "L14_label": clusters14[b14_idx].get('label') if b14_idx is not None else None,
            "L14_phi_cos": float(cos_sim(phi[14][word],
                                          clusters14[b14_idx]['centroid']))
                           if b14_idx is not None else None,
            "L23_body": f"B{b23_idx:03d}" if b23_idx is not None else None,
            "L23_label": clusters23[b23_idx].get('label') if b23_idx is not None else None,
            "L23_phi_cos": float(cos_sim(phi[23][word],
                                          clusters23[b23_idx]['centroid']))
                           if b23_idx is not None else None,
        }

    with open(ATLAS_FILE, 'w') as f:
        json.dump(atlas, f, indent=2)
    print(f"  Atlas saved: {ATLAS_FILE}")
    return atlas


# ── Section 10: Summary report ────────────────────────────────────────────────
def report(atlas, clusters14, clusters23):
    meta = atlas['meta']
    print("\n" + "="*70)
    print("DAY 27 — FULL CONTENT MAP SUMMARY")
    print("="*70)
    print(f"\n  Total clean English words : {meta['total_clean_words']}")
    print(f"  Phase 1 (common-word pole): {meta['phase1_count']} words")
    print(f"  Phase 2 (semantic bodies) : {meta['phase2_count']} words")
    print(f"\n  Gravitational bodies discovered:")
    print(f"    L14: {meta['n_bodies_L14']} bodies")
    print(f"    L23: {meta['n_bodies_L23']} bodies")
    if meta['mean_cross_layer_phi_cos']:
        print(f"  Mean cross-layer φ_cos    : {meta['mean_cross_layer_phi_cos']:.4f}")

    print(f"\n  Top 30 bodies at L14 (by size):")
    print(f"  {'Body':<6}  {'Size':>5}  {'Label'}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*40}")
    for i, cl in enumerate(clusters14[:30]):
        label = cl.get('label', f'body_{i}')
        print(f"  B{i:03d}    {cl['size']:>5}  {label}")

    print(f"\n  Top 30 bodies at L23 (by size):")
    print(f"  {'Body':<6}  {'Size':>5}  {'Label'}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*40}")
    for i, cl in enumerate(clusters23[:30]):
        label = cl.get('label', f'body_{i}')
        print(f"  B{i:03d}    {cl['size']:>5}  {label}")

    # Body coherence (mean intra-body φ_cos to centroid) for L14
    print(f"\n  L14 body coherence (mean φ_cos to centroid):")
    for i, cl in enumerate(clusters14[:15]):
        label = cl.get('label', f'body_{i}')
        wmap = atlas['word_map']
        coh_vals = [wmap[w]['L14_phi_cos'] for w in cl['words']
                    if w in wmap and wmap[w].get('L14_phi_cos') is not None]
        if coh_vals:
            print(f"  B{i:03d}  coh={np.mean(coh_vals):.3f}  n={cl['size']}  {label}")

    print(f"\nDay 27 complete. Atlas: {ATLAS_FILE}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer: {SMALL_MODEL}")
    tok = AutoTokenizer.from_pretrained(SMALL_MODEL)

    # Section 1 & 2: vocabulary + phase screening
    vocab   = load_vocab(tok)
    phase1, phase2 = screen_phases(vocab)

    # Load or build hidden-state cache
    needs_model = not os.path.exists(CACHE_FILE) or \
        len(np.load(CACHE_FILE, allow_pickle=True)['words']) < len(phase2) * 0.95

    hs_by_word = {}
    if needs_model:
        print(f"\n  Loading model: {SMALL_MODEL}...")
        model = AutoModelForCausalLM.from_pretrained(
            SMALL_MODEL, dtype=torch.float32, device_map='cpu')
        model.eval()
        print(f"  Loaded: {model.config.num_hidden_layers} layers, "
              f"hidden={model.config.hidden_size}")
        hs_by_word = load_or_build_cache(model, tok, phase2)
        del model
        import gc; gc.collect()
    else:
        hs_by_word = load_or_build_cache(None, tok, phase2)

    # Section 4–5: Z2 axes and φ-vectors
    z2_axes = build_z2_axes(hs_by_word)
    words_p2 = [w for _, w in phase2]
    phi = compute_phi(hs_by_word, z2_axes, words_p2)

    # Section 6: Clustering at both layers
    print("\n── Section 6: Clustering ────────────────────────────────────────────")
    clusters14 = cluster_phi(phi[14], "L14")
    clusters23 = cluster_phi(phi[23], "L23")

    # Section 7: Ollama labeling
    clusters14 = label_clusters(clusters14, "L14")
    clusters23 = label_clusters(clusters23, "L23")

    # Section 8: Cross-layer agreement
    cross_cos, switches = layer_agreement(phi[14], phi[23], clusters14, clusters23)

    # Section 9: Build and save atlas
    atlas = build_atlas(vocab, phase1, phase2, phi,
                        clusters14, clusters23, cross_cos, switches)

    # Section 10: Report
    report(atlas, clusters14, clusters23)
