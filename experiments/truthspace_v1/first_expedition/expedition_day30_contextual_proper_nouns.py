#!/usr/bin/env python3
"""
Expedition Day 30 — Contextual φ-Extraction for Proper Nouns

Day 29 showed that ALL capitalised proper nouns degenerate to the same
φ-pole in isolation (coherence = 0.999, cos to common-word pole = 0.9982).
In isolation, the COMB layers see only "proper noun in unmarked syntactic
position" — the same signal for Berlin, Einstein, and Carbon.

Day 30 tests whether sentence context recovers semantic differentiation:

  Q1: Does "Berlin is a city." move φ(Berlin) away from the pole?
  Q2: Do all city contexts cluster together (intra-category coherence)?
  Q3: Do city/scientist/element contexts form DISTINCT bodies?
  Q4: How much does context "lift" each word from the degenerate pole?

Approach:
  1. Categorised proper-noun list with one declarative sentence per category
  2. L14 forward pass on full sentence; extract hidden state at target position
  3. φ-vector via Day 27/29 Z2 axis
  4. Analysis:
       a. Per-word: isolation φ_cos vs contextual φ_cos to degenerate pole
       b. Per-category: intra-category coherence (contextual vs isolation)
       c. Unsupervised clustering: do categories emerge as distinct bodies?
       d. Cross-category φ_cos: are categories separating from each other?
"""

import sys, os, json, time
import numpy as np
import urllib.request

SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "day30_contextual_atlas.json")
CTX_CACHE    = os.path.join(os.path.dirname(__file__), "day30_ctx_cache.npz")
PN_CACHE     = os.path.join(os.path.dirname(__file__), "day29_pn_cache.npz")
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"

K_INIT       = 60
MERGE_COS    = 0.88
TOP_N        = 20
LAYER        = 14

KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]

# ── Categorised proper-noun list with sentence templates ──────────────────────
# Format: (word, category, sentence_with_word_capitalised)
# The sentence must contain the word as a single token.
# Template kept simple and declarative — maximum context signal, minimum noise.

# NOTE: Templates must place {word} LAST.
# Qwen2 is a causal (decoder-only) transformer: each token's hidden state
# is computed from its own embedding + ALL PREVIOUS tokens only.
# If {word} is first in the sentence, it sees BOS + itself — zero category
# information. With {word} last, it has attended to the full description.

WORDS_BY_CATEGORY = {
    'city_europe': {
        'template': 'An example of a major European capital city is {word}.',
        'words': [
            'Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Brussels',
            'Budapest', 'Copenhagen', 'Dublin', 'Helsinki', 'Lisbon',
            'London', 'Madrid', 'Moscow', 'Oslo', 'Paris',
            'Prague', 'Rome', 'Sofia', 'Stockholm', 'Vienna',
            'Warsaw', 'Zagreb',
        ],
    },
    'city_asia': {
        'template': 'An example of a major Asian capital city is {word}.',
        'words': [
            'Baghdad', 'Bangkok', 'Beijing', 'Chennai', 'Delhi',
            'Dubai', 'Hanoi', 'Istanbul', 'Jakarta', 'Kabul',
            'Karachi', 'Manila', 'Mumbai', 'Riyadh', 'Seoul',
            'Shanghai', 'Singapore', 'Taipei', 'Tehran', 'Tokyo',
        ],
    },
    'city_americas': {
        'template': 'An example of a major city in the Americas is {word}.',
        'words': [
            'Boston', 'Chicago', 'Dallas', 'Denver', 'Detroit',
            'Houston', 'Lima', 'Miami', 'Montreal', 'Ottawa',
            'Seattle', 'Toronto', 'Vancouver', 'Washington',
        ],
    },
    'city_africa_oceania': {
        'template': 'An example of a major city in Africa or Oceania is {word}.',
        'words': [
            'Auckland', 'Canberra', 'Dakar', 'Harare', 'Kampala',
            'Lagos', 'Lusaka', 'Nairobi', 'Perth', 'Sydney', 'Wellington',
        ],
    },
    'country': {
        'template': 'An example of a sovereign nation or country is {word}.',
        'words': [
            'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium',
            'Bolivia', 'Brazil', 'Bulgaria', 'Canada', 'Chile',
            'China', 'Colombia', 'Croatia', 'Cuba', 'Denmark',
            'Ecuador', 'Egypt', 'Ethiopia', 'Finland', 'France',
            'Germany', 'Ghana', 'Greece', 'Hungary', 'Iceland',
            'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
            'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan',
            'Kenya', 'Latvia', 'Lithuania', 'Malaysia', 'Mexico',
            'Mongolia', 'Morocco', 'Nepal', 'Nigeria', 'Norway',
            'Pakistan', 'Peru', 'Philippines', 'Poland', 'Portugal',
            'Romania', 'Russia', 'Serbia', 'Slovakia', 'Slovenia',
            'Somalia', 'Spain', 'Sudan', 'Sweden', 'Switzerland',
            'Syria', 'Taiwan', 'Tanzania', 'Thailand', 'Tunisia',
            'Turkey', 'Uganda', 'Ukraine', 'Venezuela', 'Vietnam',
            'Yemen', 'Zambia', 'Zimbabwe',
        ],
    },
    'scientist': {
        'template': 'An example of a famous scientist or physicist is {word}.',
        'words': [
            'Bohr', 'Celsius', 'Copernicus', 'Curie', 'Darwin',
            'Descartes', 'Edison', 'Einstein', 'Euler', 'Faraday',
            'Feynman', 'Galileo', 'Hawking', 'Heisenberg', 'Kepler',
            'Leibniz', 'Lorentz', 'Maxwell', 'Mendel', 'Mendeleev',
            'Newton', 'Pascal', 'Pasteur', 'Planck', 'Ptolemy',
            'Turing',
        ],
    },
    'historical_figure': {
        'template': 'An example of a famous historical philosopher, writer, or composer is {word}.',
        'words': [
            'Caesar', 'Confucius', 'Dante', 'Dickens', 'Dostoevsky',
            'Goethe', 'Homer', 'Kant', 'Lenin', 'Lincoln',
            'Luther', 'Machiavelli', 'Marx', 'Mozart', 'Napoleon',
            'Nietzsche', 'Plato', 'Rousseau', 'Socrates', 'Tchaikovsky',
            'Tolstoy', 'Voltaire', 'Wagner',
        ],
    },
    'element': {
        'template': 'An example of a chemical element from the periodic table is {word}.',
        'words': [
            'Aluminum', 'Argon', 'Barium', 'Beryllium', 'Bismuth',
            'Boron', 'Bromine', 'Cadmium', 'Calcium', 'Carbon',
            'Cerium', 'Cesium', 'Chlorine', 'Chromium', 'Cobalt',
            'Copper', 'Europium', 'Fluorine', 'Gadolinium', 'Gallium',
            'Germanium', 'Hafnium', 'Helium', 'Hydrogen', 'Indium',
            'Iodine', 'Iridium', 'Iron', 'Krypton', 'Lanthanum',
            'Lead', 'Lithium', 'Magnesium', 'Manganese', 'Mercury',
            'Neon', 'Nickel', 'Nitrogen', 'Osmium', 'Oxygen',
            'Palladium', 'Phosphorus', 'Platinum', 'Potassium', 'Radium',
            'Radon', 'Rhodium', 'Rubidium', 'Ruthenium', 'Scandium',
            'Selenium', 'Silicon', 'Silver', 'Sodium', 'Strontium',
            'Sulfur', 'Tantalum', 'Tellurium', 'Tin', 'Titanium',
            'Tungsten', 'Uranium', 'Vanadium', 'Xenon', 'Yttrium',
            'Zinc', 'Zirconium',
        ],
    },
    'language': {
        'template': 'An example of a world language spoken by millions is {word}.',
        'words': [
            'Arabic', 'Bengali', 'Bulgarian', 'Croatian', 'Czech',
            'Danish', 'Dutch', 'English', 'Estonian', 'Finnish',
            'French', 'German', 'Greek', 'Hebrew', 'Hindi',
            'Hungarian', 'Indonesian', 'Italian', 'Japanese', 'Korean',
            'Latvian', 'Lithuanian', 'Malay', 'Norwegian', 'Persian',
            'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian',
            'Slovak', 'Spanish', 'Swedish', 'Tamil', 'Turkish',
            'Ukrainian', 'Vietnamese',
        ],
    },
    'nationality': {
        'template': 'An example of a word describing a nationality or ethnic group is {word}.',
        'words': [
            'African', 'American', 'Asian', 'Australian', 'Brazilian',
            'British', 'Canadian', 'Chinese', 'Egyptian', 'European',
            'French', 'German', 'Greek', 'Indian', 'Iranian',
            'Irish', 'Italian', 'Japanese', 'Korean', 'Mexican',
            'Norwegian', 'Polish', 'Portuguese', 'Russian', 'Scottish',
            'Spanish', 'Swedish', 'Turkish',
        ],
    },
    'geographical': {
        'template': 'An example of a geographical region, continent, or body of water is {word}.',
        'words': [
            'Africa', 'Alaska', 'Alps', 'Amazon', 'Antarctica',
            'Arctic', 'Asia', 'Atlantic', 'Australia', 'Baltic',
            'Caspian', 'Caucasus', 'Danube', 'Europe', 'Ganges',
            'Mediterranean', 'Nile', 'Oceania', 'Pacific', 'Sahara',
            'Scandinavia', 'Siberia', 'Volga',
        ],
    },
    'tech_brand': {
        'template': 'An example of a well-known technology company is {word}.',
        'words': [
            'Amazon', 'Apple', 'Facebook', 'Google', 'Huawei',
            'Intel', 'Microsoft', 'Nvidia', 'Oracle', 'Samsung',
            'Tesla', 'Twitter',
        ],
    },
}

# Flatten to list of (word, category, sentence) for processing
ALL_ITEMS = []
for cat, spec in WORDS_BY_CATEGORY.items():
    tmpl = spec['template']
    for w in spec['words']:
        sentence = tmpl.format(word=w)
        ALL_ITEMS.append((w, cat, sentence))

# Deduplicate words that appear in multiple categories (keep first occurrence)
seen_words = set()
DEDUPED = []
for item in ALL_ITEMS:
    if item[0] not in seen_words:
        seen_words.add(item[0])
        DEDUPED.append(item)


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
        f"Here are {len(words)} words that Qwen2 groups into the same "
        f"internal semantic cluster:\n\n{', '.join(words[:TOP_N])}\n\n"
        f"Give this cluster a SHORT label (2-5 words) capturing its most "
        f"specific theme. Reply with ONLY the label."
    )
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0},
    }).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"].strip().strip('"\'')
    except Exception:
        return fallback


# ── Step 1: Load model ────────────────────────────────────────────────────────
def load_model():
    print(f"\n── Step 1: Loading {SMALL_MODEL} ─────────────────────────────────")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu',
        output_hidden_states=True)
    model.eval()
    print(f"  Loaded: {model.config.num_hidden_layers} layers, "
          f"hidden={model.config.hidden_size}")
    return tok, model


# ── Step 2: Filter to single-token words ─────────────────────────────────────
def filter_single_token(tok, items):
    print(f"\n── Step 2: Filtering to single-token words ─────────────────────")
    valid = []
    skipped = []
    for word, cat, sentence in items:
        for prefix in [' ', '']:
            ids = tok.encode(prefix + word, add_special_tokens=False)
            if len(ids) == 1:
                valid.append((word, cat, sentence, ids[0]))
                break
        else:
            skipped.append(word)
    print(f"  Items: {len(items)}  single-token: {len(valid)}  "
          f"multi-token skipped: {len(skipped)}")
    if skipped:
        print(f"  Skipped examples: {', '.join(skipped[:12])}")
    return valid


# ── Step 3: Contextual L14 hidden states ─────────────────────────────────────
def get_contextual_hs(tok, model, valid_items):
    import torch
    print(f"\n── Step 3: Contextual L{LAYER} hidden states (with cache) ─────────")

    # Load cache
    if os.path.exists(CTX_CACHE):
        npz   = np.load(CTX_CACHE, allow_pickle=True)
        keys  = list(npz['keys'])
        c_hs  = {k: npz['hs'][i] for i, k in enumerate(keys)}
        print(f"  Cache: {len(c_hs)} entries found")
    else:
        c_hs  = {}
        print(f"  No cache yet.")

    # Cache key = word + sentence (to allow same word in different contexts)
    missing = [(w, cat, sent, tid) for w, cat, sent, tid in valid_items
               if (w + '|' + sent) not in c_hs]
    print(f"  Missing: {len(missing)}")

    if missing:
        new_hs = {}
        t0 = time.time()
        fails = []
        for i, (word, cat, sentence, tid) in enumerate(missing):
            cache_key = word + '|' + sentence
            # Tokenize full sentence
            inp  = tok(sentence, return_tensors='pt')
            tids = inp['input_ids'][0].tolist()
            # Find position of target token
            pos = None
            for j, t in enumerate(tids):
                if t == tid:
                    pos = j
                    break
            if pos is None:
                # Try without leading space (sentence-initial capitalized word)
                alt_ids = tok.encode(word, add_special_tokens=False)
                for j, t in enumerate(tids):
                    if t == alt_ids[0]:
                        pos = j
                        break
            if pos is None:
                fails.append(word)
                continue
            with torch.no_grad():
                out = model(**inp)
            new_hs[cache_key] = out.hidden_states[LAYER + 1][0, pos, :].numpy()
            if (i + 1) % 100 == 0:
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (i + 1) * (len(missing) - i - 1)
                print(f"  [{i+1:>4}/{len(missing)}]  "
                      f"{elapsed:.1f} min elapsed  ETA {eta:.1f} min")

        if fails:
            print(f"  WARNING: could not locate token in sentence for: "
                  f"{', '.join(fails[:10])}")

        c_hs.update(new_hs)
        all_keys = list(c_hs.keys())
        all_hs   = np.stack([c_hs[k] for k in all_keys])
        np.savez(CTX_CACHE, keys=all_keys, hs=all_hs)
        print(f"  Cache saved: {len(all_keys)} entries")

    # Return only items that succeeded
    result = []
    for word, cat, sentence, tid in valid_items:
        key = word + '|' + sentence
        if key in c_hs:
            result.append((word, cat, sentence, c_hs[key]))
    print(f"  Contextual hidden states ready: {len(result)}")
    return result


# ── Step 4: Z2 axis ───────────────────────────────────────────────────────────
def build_z2(tok, model):
    import torch
    print(f"\n── Step 4: Z2 axis at L{LAYER} ──────────────────────────────────")
    deltas = []
    for a, b in KILLING_PAIRS:
        for prefix in [' ', '']:
            ia = tok.encode(prefix + a, add_special_tokens=False)
            ib = tok.encode(prefix + b, add_special_tokens=False)
            if len(ia) == 1 and len(ib) == 1:
                def gh(w):
                    inp = tok(w, return_tensors='pt')
                    with torch.no_grad():
                        out = model(**inp)
                    return out.hidden_states[LAYER + 1][0, -1, :].numpy().astype(np.float64)
                d = gh(prefix + b) - gh(prefix + a)
                dm = np.linalg.norm(d)
                if dm > 1e-20:
                    deltas.append(d / dm)
                break
    D = np.stack(deltas)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    z2 = Vt[0]
    pct = 100 * sv[0]**2 / np.sum(sv**2)
    print(f"  Z2 L{LAYER}: {pct:.1f}%  ({len(deltas)} delta vectors)")
    return z2 / (np.linalg.norm(z2) + 1e-20)


# ── Step 5: φ-vectors ─────────────────────────────────────────────────────────
def compute_phi(ctx_data, z2):
    print(f"\n── Step 5: φ-vectors ────────────────────────────────────────────")
    phi = {}
    cat_map = {}
    for word, cat, sentence, h in ctx_data:
        phi[word] = phi_vec(h, z2)
        cat_map[word] = cat
    print(f"  φ-vectors: {len(phi)}")
    return phi, cat_map


# ── Step 6: Load isolation φ-vectors from Day 29 for comparison ───────────────
def load_isolation_phi(words, z2):
    print(f"\n── Step 6: Loading Day 29 isolation φ-vectors ───────────────────")
    if not os.path.exists(PN_CACHE):
        print(f"  WARNING: {PN_CACHE} not found. Skipping isolation comparison.")
        return {}
    npz     = np.load(PN_CACHE, allow_pickle=True)
    pn_idx  = {w: i for i, w in enumerate(npz['words'])}
    iso_phi = {}
    for w in words:
        if w in pn_idx:
            iso_phi[w] = phi_vec(npz['hs'][pn_idx[w]], z2)
    print(f"  Isolation φ-vectors loaded: {len(iso_phi)}")
    return iso_phi


# ── Step 7: Category coherence analysis ───────────────────────────────────────
def analyse_categories(phi, cat_map, iso_phi):
    print(f"\n── Step 7: Category coherence analysis ──────────────────────────")

    # Build degenerate-pole centroid from isolation φ
    if iso_phi:
        iso_vecs = np.stack(list(iso_phi.values()))
        pole = iso_vecs.mean(axis=0)
        pole /= np.linalg.norm(pole)
    else:
        pole = None

    cats = sorted(set(cat_map.values()))
    print(f"\n  {'Category':<24}  {'n':>4}  {'Coh':>5}  "
          f"{'cos_pole_ctx':>12}  {'cos_pole_iso':>12}  {'Δ_lift':>7}")
    print(f"  {'─'*24}  {'─'*4}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*7}")

    results = {}
    for cat in cats:
        words_in_cat = [w for w, c in cat_map.items() if c == cat and w in phi]
        if len(words_in_cat) < 2:
            continue
        vecs = np.stack([phi[w] for w in words_in_cat])
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        coh = float(np.mean([cos_sim(phi[w], centroid) for w in words_in_cat]))

        # Cosine to degenerate pole (contextual)
        cos_pole_ctx = float(np.mean([cos_sim(phi[w], pole)
                                       for w in words_in_cat])) if pole is not None else float('nan')

        # Cosine to degenerate pole (isolation)
        iso_in_cat = [w for w in words_in_cat if w in iso_phi]
        cos_pole_iso = float(np.mean([cos_sim(iso_phi[w], pole)
                                       for w in iso_in_cat])) if iso_in_cat else float('nan')

        delta = (cos_pole_iso - cos_pole_ctx) if not (np.isnan(cos_pole_ctx) or np.isnan(cos_pole_iso)) else float('nan')

        results[cat] = {
            'n': len(words_in_cat),
            'coherence': coh,
            'cos_pole_ctx': cos_pole_ctx,
            'cos_pole_iso': cos_pole_iso,
            'lift': delta,
            'centroid': centroid,
            'words': words_in_cat,
        }

        print(f"  {cat:<24}  {len(words_in_cat):>4}  {coh:.3f}  "
              f"{cos_pole_ctx:>12.4f}  {cos_pole_iso:>12.4f}  "
              f"{delta:>+7.4f}")

    # Cross-category separation
    print(f"\n  Cross-category φ_cos (lower = better separation):")
    cat_names = list(results.keys())
    print(f"  {'':24}", end='')
    for c in cat_names[:8]:
        print(f"  {c[:8]:>8}", end='')
    print()
    for i, ci in enumerate(cat_names[:8]):
        print(f"  {ci:<24}", end='')
        for j, cj in enumerate(cat_names[:8]):
            v = cos_sim(results[ci]['centroid'], results[cj]['centroid'])
            print(f"  {v:>8.3f}", end='')
        print()

    return results


# ── Step 8: Unsupervised clustering ───────────────────────────────────────────
def cluster(phi):
    from sklearn.cluster import MiniBatchKMeans
    words = sorted(phi.keys())
    X = np.stack([phi[w] for w in words]).astype(np.float32)
    k = min(K_INIT, len(words) // 3)

    print(f"\n── Step 8: Unsupervised clustering ──────────────────────────────")
    print(f"  MiniBatchKMeans k={k} on {len(words)} words...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                          batch_size=1024, max_iter=300)
    labels = km.fit_predict(X)

    clusters = []
    for ci in range(k):
        idx = [j for j, l in enumerate(labels) if l == ci]
        if not idx:
            continue
        vecs = X[idx]
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-20
        coh = float(np.mean([cos_sim(vecs[j], centroid) for j in range(len(idx))]))
        cluster_words = [words[j] for j in idx]
        clusters.append({
            'words': cluster_words,
            'size': len(cluster_words),
            'coherence': coh,
            'centroid': centroid,
        })

    # Merge
    clusters.sort(key=lambda c: -c['size'])
    print(f"  Merging (cos≥{MERGE_COS})...")
    changed = True
    while changed:
        changed = False
        merged = [False] * len(clusters)
        new_clusters = []
        for i in range(len(clusters)):
            if merged[i]:
                continue
            cur = dict(clusters[i])
            cur['centroid'] = cur['centroid'].copy()
            for j in range(i + 1, len(clusters)):
                if merged[j]:
                    continue
                if cos_sim(cur['centroid'], clusters[j]['centroid']) >= MERGE_COS:
                    ni, nj = cur['size'], clusters[j]['size']
                    cur['words'] += clusters[j]['words']
                    cur['size']  += nj
                    cur['centroid'] = (ni * cur['centroid'] +
                                       nj * clusters[j]['centroid']) / (ni + nj)
                    cur['centroid'] /= np.linalg.norm(cur['centroid']) + 1e-20
                    cur['coherence'] = (ni * cur['coherence'] +
                                        nj * clusters[j]['coherence']) / (ni + nj)
                    merged[j] = True
                    changed = True
            merged[i] = True
            new_clusters.append(cur)
        clusters = new_clusters

    # Add top_words
    for cl in clusters:
        phi_cl = np.stack([phi[w] for w in cl['words']])
        cent = phi_cl.mean(axis=0); cent /= np.linalg.norm(cent) + 1e-20
        sims = [cos_sim(phi[w], cent) for w in cl['words']]
        cl['top_words'] = [cl['words'][j] for j in np.argsort(sims)[::-1][:TOP_N]]

    print(f"  Final bodies: {len(clusters)}")
    return clusters


# ── Step 9: Label clusters ────────────────────────────────────────────────────
def label_clusters(clusters):
    print(f"\n── Step 9: Ollama labelling ─────────────────────────────────────")
    for i, cl in enumerate(clusters):
        label = ollama_label(cl['top_words'], f"C{i:03d}")
        cl['label'] = label
        if i < 5 or i % 10 == 0:
            print(f"  [{i+1:>3}/{len(clusters)}]  C{i:03d} "
                  f"(n={cl['size']:>4}, coh={cl['coherence']:.3f}):  {label}")
    return clusters


# ── Step 10: Category purity of unsupervised clusters ────────────────────────
def category_purity(clusters, cat_map):
    print(f"\n── Step 10: Category purity of unsupervised clusters ────────────")
    print(f"  Cluster assignment vs known categories:\n")
    print(f"  {'Cluster':<8}  {'n':>4}  {'Coh':>5}  "
          f"{'Dominant category':>24}  {'Purity':>6}  Label")
    print(f"  {'─'*8}  {'─'*4}  {'─'*5}  {'─'*24}  {'─'*6}  {'─'*30}")

    total_pure = 0
    total_words = 0
    for i, cl in enumerate(clusters):
        cat_counts = {}
        for w in cl['words']:
            c = cat_map.get(w, 'unknown')
            cat_counts[c] = cat_counts.get(c, 0) + 1
        dominant = max(cat_counts, key=cat_counts.get)
        purity = cat_counts[dominant] / cl['size']
        total_pure  += cat_counts[dominant]
        total_words += cl['size']
        print(f"  C{i:03d}     {cl['size']:>4}  {cl['coherence']:.3f}  "
              f"{dominant:>24}  {purity:>5.1%}  {cl.get('label', '?')}")

    overall = total_pure / total_words if total_words else 0
    print(f"\n  Overall purity (fraction assigned to dominant category): "
          f"{overall:.1%}")
    return overall


# ── Step 11: Save atlas ───────────────────────────────────────────────────────
def save_atlas(clusters, cat_results, cat_map, phi):
    output = {
        "meta": {
            "experiment": "Day 30 — Contextual proper-noun φ-extraction",
            "layer": f"L{LAYER}",
            "n_words": len(phi),
            "n_bodies": len(clusters),
        },
        "category_analysis": {
            cat: {
                "n": r['n'],
                "coherence": r['coherence'],
                "cos_pole_contextual": r['cos_pole_ctx'],
                "cos_pole_isolation": r['cos_pole_iso'],
                "pole_lift": r['lift'],
                "sample_words": r['words'][:10],
            }
            for cat, r in cat_results.items()
        },
        "bodies": {
            f"C{i:03d}": {
                "label": cl.get('label', f'C{i}'),
                "size": cl['size'],
                "coherence": cl['coherence'],
                "top_words": cl.get('top_words', cl['words'][:TOP_N]),
            }
            for i, cl in enumerate(clusters)
        },
        "word_map": {w: f"C{i:03d}" for i, cl in enumerate(clusters)
                     for w in cl['words']},
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Atlas saved: {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()

    tok, model = load_model()
    valid      = filter_single_token(tok, DEDUPED)
    ctx_data   = get_contextual_hs(tok, model, valid)
    z2         = build_z2(tok, model)
    del model

    phi, cat_map  = compute_phi(ctx_data, z2)
    iso_phi       = load_isolation_phi(list(phi.keys()), z2)
    cat_results   = analyse_categories(phi, cat_map, iso_phi)
    clusters      = cluster(phi)
    clusters      = label_clusters(clusters)
    purity        = category_purity(clusters, cat_map)
    save_atlas(clusters, cat_results, cat_map, phi)

    print(f"\n{'='*70}")
    print(f"DAY 30 — CONTEXTUAL PROPER-NOUN EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Words processed: {len(phi)}")
    print(f"  Unsupervised bodies: {len(clusters)}")
    print(f"  Overall category purity: {purity:.1%}")
    n_lifted = sum(1 for r in cat_results.values()
                   if not np.isnan(r['lift']) and r['lift'] > 0.01)
    print(f"  Categories significantly lifted from pole: "
          f"{n_lifted}/{len(cat_results)}")
    print(f"\nDay 30 complete. Atlas: {OUTPUT_FILE}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
