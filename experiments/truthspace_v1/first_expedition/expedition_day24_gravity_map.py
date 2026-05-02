#!/usr/bin/env python3
"""
Expedition Day 24 — Gravitational Body Discovery

Hypothesis:
  The φ-space clusters found in Day 23 are gravitational bodies — conceptual
  attractors formed by training-corpus co-occurrence mass. Words are pulled
  toward these bodies during COMB processing and settle into their final orbit
  at L27.

Strategy:
  1. Extract L0 embeddings for ~30K clean English tokens directly from the
     embedding matrix (no forward passes — milliseconds, not hours).
  2. Cluster the normalised embedding vectors with k-means to discover the
     gravitational bodies.
  3. Feed each cluster's top-20 most-central tokens to Ollama (qwen2.5:14b)
     to get a one-phrase human-readable label.
  4. Build a gravitational force model:
         force(word, body) = body_mass / angular_distance(φ_word, φ_centroid)²
  5. Verify: assign the 67 Day-23 words to their strongest gravitational body.
     Do the assignments match the semantic labels we already know?
  6. Explore: for unknown words, does the gravity model give sensible assignments?

Key insight:
  We are NOT running forward passes on all 151K tokens. The L0 embedding
  matrix is already in RAM as a 151K × 1536 weight matrix. Clustering it
  directly gives the raw semantic geography before COMB processing.
  The COMB rotation refines positions but the coarse cluster structure is
  already present at L0 — this is what Day 24 tests.
"""

import sys, os, re, json, time
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL   = "Qwen/Qwen2-1.5B-Instruct"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:14b"
N_CLUSTERS    = 32
MIN_WORD_LEN  = 3
MAX_WORD_LEN  = 14
SAMPLE_CAP    = 8000    # max words to cluster (speed vs coverage tradeoff)
TOP_PER_CLUSTER = 25   # words shown to LLM for naming


# ── Day 23 words for ground-truth verification ────────────────────────────────
DAY23_LABELS = {
    'berlin': 'European capital', 'paris': 'European capital',
    'madrid': 'European capital', 'vienna': 'European capital',
    'london': 'European capital', 'rome':   'European capital',
    'tokyo': 'Asian capital', 'beijing': 'Asian capital',
    'seoul': 'Asian capital', 'mumbai': 'Asian city',
    'bangkok': 'Asian capital', 'cairo': 'African/Mediterranean city',
    'sydney': 'Oceanian city', 'nairobi': 'African city',
    'elephant': 'large land mammal', 'rhinoceros': 'large land mammal',
    'hippopotamus': 'large land mammal', 'giraffe': 'large land mammal',
    'chimpanzee': 'primate', 'gorilla': 'primate', 'orangutan': 'primate',
    'dolphin': 'marine animal', 'whale': 'marine animal', 'octopus': 'marine animal',
    'penguin': 'bird', 'eagle': 'bird', 'parrot': 'bird',
    'crocodile': 'reptile', 'python': 'reptile/language', 'iguana': 'reptile',
    'helium': 'noble gas', 'neon': 'noble gas', 'argon': 'noble gas',
    'nitrogen': 'atmospheric gas', 'oxygen': 'atmospheric gas',
    'carbon': 'nonmetal element', 'silicon': 'nonmetal element',
    'sulfur': 'nonmetal element', 'iron': 'metal', 'copper': 'metal',
    'gold': 'metal', 'silver': 'metal', 'hydrogen': 'reactive element',
    'sodium': 'reactive element', 'potassium': 'reactive element',
    'cat': 'common animal', 'dog': 'common animal', 'tree': 'plant',
    'bird': 'common animal', 'house': 'common noun',
    'man': 'person', 'woman': 'person', 'king': 'royalty', 'queen': 'royalty',
    'boy': 'person', 'girl': 'person', 'big': 'adjective', 'fast': 'adjective',
    'old': 'adjective', 'tall': 'adjective', 'cats': 'plural',
    'dogs': 'plural', 'trees': 'plural', 'birds': 'plural', 'houses': 'plural',
    'bigger': 'comparative', 'faster': 'comparative', 'older': 'comparative',
    'taller': 'comparative', 'prince': 'royalty', 'princess': 'royalty',
}


def ollama_label(words, cluster_id):
    """Ask Ollama to name a cluster given its top representative words."""
    word_list = ', '.join(words[:TOP_PER_CLUSTER])
    prompt = (
        f"I have a cluster of words that share a semantic theme. "
        f"Here are the most representative words:\n{word_list}\n\n"
        f"What single short phrase (2-5 words) best describes the semantic "
        f"category these words belong to? Answer ONLY with the category phrase, "
        f"nothing else. Examples of good answers: 'European capital cities', "
        f"'chemical elements', 'physical actions', 'emotions and feelings'."
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 20}
        }, timeout=30)
        if r.status_code == 200:
            label = r.json().get('response', '').strip()
            label = label.split('\n')[0].strip().strip('"').strip("'")
            return label[:60]
    except Exception as e:
        pass
    return f"cluster_{cluster_id}"


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.cluster import MiniBatchKMeans

    print(f"  Loading tokenizer and embedding matrix from {SMALL_MODEL}...")
    tok = AutoTokenizer.from_pretrained(SMALL_MODEL)

    # Load only the embedding layer to save memory
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    emb_matrix = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
    n_layers    = model.config.num_hidden_layers
    print(f"  Embedding matrix: {emb_matrix.shape}  ({emb_matrix.nbytes/1e9:.2f} GB)")
    del model
    import gc; gc.collect()

    # ── Step 1: Build English word list from system dictionary ─────────────────
    print(f"\n  Loading English word list from /usr/share/dict/words...")
    with open('/usr/share/dict/words') as f:
        dict_words = {w.strip().lower() for w in f
                      if MIN_WORD_LEN <= len(w.strip()) <= MAX_WORD_LEN
                      and w.strip().isalpha()
                      and w.strip().isascii()}
    print(f"  Dictionary words (ASCII, len {MIN_WORD_LEN}-{MAX_WORD_LEN}): {len(dict_words)}")

    # For each dictionary word, find its token ID in Qwen (single-token words only)
    print(f"  Finding single-token encodings in Qwen tokenizer...")
    clean_words = []
    for word in dict_words:
        # Try with leading space (word-boundary token)
        ids = tok.encode(' ' + word, add_special_tokens=False)
        if len(ids) == 1:
            clean_words.append((ids[0], word))
        else:
            # Try without leading space
            ids2 = tok.encode(word, add_special_tokens=False)
            if len(ids2) == 1:
                clean_words.append((ids2[0], word))

    # Deduplicate by token ID (keep first)
    seen_ids = set()
    deduped = []
    for tid, word in clean_words:
        if tid not in seen_ids:
            seen_ids.add(tid)
            deduped.append((tid, word))
    clean_words = deduped
    print(f"  Single-token English words: {len(clean_words)}")

    # Sample if too large
    if len(clean_words) > SAMPLE_CAP:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(clean_words), SAMPLE_CAP, replace=False)
        sampled = [clean_words[i] for i in sorted(idx)]
        print(f"  Sampled {len(sampled)} words (cap={SAMPLE_CAP})")
    else:
        sampled = clean_words
        print(f"  Using all {len(sampled)} words")

    token_ids   = np.array([x[0] for x in sampled])
    word_labels = [x[1] for x in sampled]

    # ── Step 2: Extract and normalise L0 embeddings ──────────────────────────
    print(f"  Extracting L0 embeddings...")
    E = emb_matrix[token_ids]                               # (N, 1536)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / (norms + 1e-20)                           # unit vectors

    # ── Step 3: k-means clustering on L0 embeddings ──────────────────────────
    print(f"  Running MiniBatchKMeans (k={N_CLUSTERS})...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                         batch_size=2048, n_init=5, max_iter=300)
    labels_km = km.fit_predict(E_norm)
    print(f"  Clustering done in {time.time()-t0:.1f}s")

    # Cluster centroids (already in E_norm space)
    centroids = km.cluster_centers_
    cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / (cent_norms + 1e-20)

    # ── Step 4: Top words per cluster (by cosine to centroid) ────────────────
    cluster_words = {}
    cluster_sims  = {}
    for k in range(N_CLUSTERS):
        mask = labels_km == k
        members = [(cos_sim(E_norm[i], centroids_norm[k]), word_labels[i])
                   for i in np.where(mask)[0]]
        members.sort(reverse=True)
        cluster_words[k] = [w for _, w in members[:TOP_PER_CLUSTER]]
        cluster_sims[k]  = [s for s, _ in members[:TOP_PER_CLUSTER]]

    cluster_sizes = {k: int(np.sum(labels_km == k)) for k in range(N_CLUSTERS)}

    # ── Step 5: LLM labeling ─────────────────────────────────────────────────
    print(f"\n  Calling Ollama ({OLLAMA_MODEL}) to label {N_CLUSTERS} clusters...")
    cluster_labels = {}
    for k in range(N_CLUSTERS):
        words_for_llm = cluster_words[k][:TOP_PER_CLUSTER]
        label = ollama_label(words_for_llm, k)
        cluster_labels[k] = label
        print(f"  C{k:02d} ({cluster_sizes[k]:4d} words) [{label}]")
        print(f"       top-10: {', '.join(cluster_words[k][:10])}")

    # ── Step 6: Gravitational force model ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DAY 24 — Gravitational Force Model")
    print(f"{'='*70}")

    def gravity_score(word_vec_norm, cluster_id, alpha=2.0):
        """
        Force = mass / distance^alpha
        mass     = cluster size (number of words bound to this body)
        distance = 1 - cos(word, centroid)  (angular distance on sphere)
        """
        mass     = cluster_sizes[cluster_id]
        ang_dist = max(1e-6, 1.0 - cos_sim(word_vec_norm, centroids_norm[cluster_id]))
        return mass / (ang_dist ** alpha)

    def top_clusters(word_vec_norm, top=3):
        scores = [(gravity_score(word_vec_norm, k), k) for k in range(N_CLUSTERS)]
        scores.sort(reverse=True)
        return scores[:top]

    # ── Step 7: Verify with Day 23 ground-truth words ────────────────────────
    print(f"\n── Section A: Day 23 ground-truth verification ───────────────────────")
    print(f"  Assigning {len(DAY23_LABELS)} known words to gravitational bodies.\n")
    print(f"  {'Word':<16} {'Known label':<26} {'Gravity cluster':<32} {'Match?'}")
    print("  " + "─"*85)

    hits, total = 0, 0
    miss_list   = []
    for word, known_label in sorted(DAY23_LABELS.items()):
        ids = tok.encode(' ' + word, add_special_tokens=False)
        if not ids: continue
        tid = ids[0]
        if tid >= len(emb_matrix): continue
        wvec = emb_matrix[tid]
        wvec_n = wvec / (np.linalg.norm(wvec) + 1e-20)
        top = top_clusters(wvec_n, top=1)
        _, best_k = top[0]
        gravity_label = cluster_labels[best_k]
        # Soft match: check if known_label words appear in gravity_label or vice versa
        known_words   = set(known_label.lower().split())
        gravity_words = set(gravity_label.lower().split())
        match = bool(known_words & gravity_words) or known_label.lower() in gravity_label.lower() \
                or gravity_label.lower() in known_label.lower()
        symbol = '✓' if match else '?'
        hits  += int(match); total += 1
        print(f"  {symbol} {word:<16} {known_label:<26} {gravity_label:<32}")
        if not match:
            miss_list.append((word, known_label, gravity_label, cluster_words[best_k][:8]))

    print(f"\n  Semantic match rate: {hits}/{total} = {100*hits/total:.1f}%")
    if miss_list:
        print(f"\n  Words where gravity cluster doesn't semantically match known label:")
        for word, known, grav, examples in miss_list:
            print(f"    {word}: known='{known}' gravity='{grav}'")
            print(f"           cluster examples: {examples}")

    # ── Step 8: Cross-cluster gravity paths ───────────────────────────────────
    print(f"\n── Section B: Within-group gravity coherence ─────────────────────────")
    print(f"  Do semantically related words share the same gravitational body?\n")

    GROUPS = {
        'European capitals': ['berlin', 'paris', 'madrid', 'london', 'rome', 'vienna'],
        'Asian capitals':    ['tokyo', 'beijing', 'seoul', 'bangkok', 'mumbai'],
        'Metals':            ['iron', 'gold', 'silver', 'copper'],
        'Noble gases':       ['helium', 'neon', 'argon'],
        'Large mammals':     ['elephant', 'rhinoceros', 'hippopotamus', 'giraffe'],
        'Primates':          ['chimpanzee', 'gorilla', 'orangutan'],
        'Marine animals':    ['dolphin', 'whale', 'octopus'],
        'Birds':             ['penguin', 'eagle', 'parrot'],
        'Plurals':           ['cats', 'dogs', 'trees', 'birds', 'houses'],
        'Comparatives':      ['bigger', 'faster', 'older', 'taller'],
    }

    for group_name, words in GROUPS.items():
        assignments = {}
        for w in words:
            ids = tok.encode(' ' + w, add_special_tokens=False)
            if not ids: continue
            wvec = emb_matrix[ids[0]] if ids[0] < len(emb_matrix) else None
            if wvec is None: continue
            wvec_n = wvec / (np.linalg.norm(wvec) + 1e-20)
            _, best_k = top_clusters(wvec_n)[0]
            assignments[w] = best_k

        cluster_counts = {}
        for w, k in assignments.items():
            cluster_counts[k] = cluster_counts.get(k, 0) + 1
        dominant_k  = max(cluster_counts, key=cluster_counts.get)
        dom_count   = cluster_counts[dominant_k]
        coherence   = dom_count / len(assignments) if assignments else 0
        label       = cluster_labels[dominant_k]

        verdict = '✓ COHERENT' if coherence >= 0.6 else \
                  ('~ MIXED' if coherence >= 0.4 else '✗ SCATTERED')
        print(f"  {verdict}  {group_name:<22} → [{label}]  "
              f"({dom_count}/{len(assignments)} in dominant body)")
        if coherence < 1.0:
            for w, k in assignments.items():
                if k != dominant_k:
                    print(f"    outlier: {w} → [{cluster_labels[k]}]")

    # ── Step 9: Exploratory gravity probing ───────────────────────────────────
    print(f"\n── Section C: Probing unknown words ─────────────────────────────────")
    print(f"  What gravitational body does the model assign to unfamiliar words?\n")

    PROBE_WORDS = [
        'volcano', 'democracy', 'cathedral', 'chromosome', 'hurricane',
        'philosopher', 'telescope', 'constellation', 'renaissance', 'monastery',
        'algorithm', 'parliament', 'archipelago', 'metabolism', 'symphony',
        'glacier', 'hemisphere', 'lithosphere', 'peninsula', 'meridian',
    ]

    print(f"  {'Word':<18} {'Top gravity body':<35} {'2nd body':<30}")
    print("  " + "─"*85)
    for w in PROBE_WORDS:
        ids = tok.encode(' ' + w, add_special_tokens=False)
        if not ids:
            print(f"  {w:<18} (not in vocab)")
            continue
        if ids[0] >= len(emb_matrix):
            continue
        wvec   = emb_matrix[ids[0]]
        wvec_n = wvec / (np.linalg.norm(wvec) + 1e-20)
        top = top_clusters(wvec_n, top=2)
        s1, k1 = top[0]; s2, k2 = top[1]
        l1 = cluster_labels[k1]; l2 = cluster_labels[k2]
        print(f"  {w:<18} [{l1:<33}]  [{l2:<28}]")

    # ── Step 10: Full cluster inventory ───────────────────────────────────────
    print(f"\n── Section D: Full gravitational body inventory ──────────────────────")
    print(f"  All {N_CLUSTERS} gravitational bodies, sorted by size.\n")
    print(f"  {'#':<4} {'Size':<6} {'Cohesion':<10} {'Label':<35} Top-8 examples")
    print("  " + "─"*100)
    sorted_clusters = sorted(range(N_CLUSTERS), key=lambda k: -cluster_sizes[k])
    for k in sorted_clusters:
        size      = cluster_sizes[k]
        cohesion  = float(np.mean(cluster_sims[k][:10])) if cluster_sims[k] else 0
        label     = cluster_labels[k]
        top8      = ', '.join(cluster_words[k][:8])
        print(f"  C{k:02d}  {size:<6} {cohesion:<10.4f} {label:<35} {top8}")

    # ── Section E: Gravity model quality summary ──────────────────────────────
    print(f"\n── Section E: Summary ────────────────────────────────────────────────")
    sizes = list(cluster_sizes.values())
    cohesions = [float(np.mean(cluster_sims[k][:10])) for k in range(N_CLUSTERS)
                 if cluster_sims[k]]
    print(f"""
  Words clustered:     {len(sampled)}
  Gravitational bodies:{N_CLUSTERS}
  Largest body:        {max(sizes)} words  [{cluster_labels[sorted_clusters[0]]}]
  Smallest body:       {min(sizes)} words  [{cluster_labels[sorted_clusters[-1]]}]
  Mean body size:      {np.mean(sizes):.0f} words
  Mean cohesion:       {np.mean(cohesions):.4f}  (higher = tighter body)
  Day-23 match rate:   {hits}/{total} = {100*hits/max(total,1):.1f}%

  The gravitational bodies are named by Qwen2.5-14B examining only the
  top-{TOP_PER_CLUSTER} most central words in each cluster. No human labelling.
  The match rate measures how well L0 embedding clusters predict the
  semantic category of Day-23 words (which were verified at L14).
    """)

    print(f"{'='*70}")
    print(f"Day 24 complete.")
    print(f"{'='*70}")
