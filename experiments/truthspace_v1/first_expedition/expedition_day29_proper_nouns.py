#!/usr/bin/env python3
"""
Expedition Day 29 — Proper-Noun Body Map (L14, curated list)

Days 27-28 confirmed that:
  - The φ-geometry (Z2 axis, gravitational bodies) emerges at L14, NOT at L0
  - Z2 explains only 20.1% of variance at L0 vs 82.1% at L14
  - Clustering 16,500 capitalized tokens at L0 fails: all end up in one blob
    because Qwen2's embedding space does not differentiate proper-noun categories

The correct approach:
  1. Define a CURATED list of ~1,500 proper nouns across known categories:
     world cities, countries, historical/scientific persons, chemical elements,
     nationalities, languages, geographical features, currencies
  2. Filter to single-token words in Qwen2 (most multi-word names are multi-token)
  3. Run L14 forward passes (reuses Day 27 Z2 axis from cache)
  4. Cluster + label → proper-noun atlas

This approach takes ~3 minutes (vs >50 min for all 16,500 proper-noun tokens).
"""

import sys, os, re, json, time
import numpy as np
import urllib.request

SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "day29_proper_noun_atlas.json")
PN_CACHE     = os.path.join(os.path.dirname(__file__), "day29_pn_cache.npz")
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"

K_INIT       = 100
MERGE_COS    = 0.88
TOP_N        = 25
LAYER        = 14      # reuses Day 27 Z2 axis

# ── Curated proper-noun list ──────────────────────────────────────────────────
CURATED = [
    # World cities — Europe
    'Amsterdam','Athens','Barcelona','Berlin','Brussels','Bucharest','Budapest',
    'Copenhagen','Dublin','Helsinki','Lisbon','Ljubljana','London','Luxembourg',
    'Madrid','Minsk','Monaco','Moscow','Nicosia','Oslo','Paris','Prague','Reykjavik',
    'Riga','Rome','Sarajevo','Skopje','Sofia','Stockholm','Tallinn','Tirana',
    'Valletta','Vienna','Vilnius','Warsaw','Zagreb',
    # World cities — Asia
    'Astana','Baghdad','Baku','Bangkok','Beijing','Chennai','Colombo','Damascus',
    'Delhi','Dhaka','Dubai','Hanoi','Istanbul','Jakarta','Kabul','Karachi',
    'Kathmandu','Kuwait','Lahore','Manila','Mumbai','Naypyidaw','Riyadh','Seoul',
    'Shanghai','Singapore','Taipei','Tehran','Tokyo','Ulaanbaatar','Yangon',
    # World cities — Americas
    'Bogota','Boston','Brasilia','Buenos','Calgary','Chicago','Dallas','Denver',
    'Detroit','Houston','Lima','Managua','Miami','Montreal','Nassau','Ottawa',
    'Quito','Seattle','Toronto','Vancouver','Washington',
    # World cities — Africa & Oceania
    'Abuja','Accra','Addis','Auckland','Canberra','Dakar','Harare','Kampala',
    'Kinshasa','Lagos','Lusaka','Nairobi','Perth','Sydney','Wellington',
    # Countries
    'Afghanistan','Albania','Algeria','Angola','Argentina','Armenia','Australia',
    'Austria','Azerbaijan','Bangladesh','Belarus','Belgium','Bolivia','Brazil',
    'Bulgaria','Cambodia','Canada','Chile','China','Colombia','Croatia','Cuba',
    'Denmark','Ecuador','Egypt','Ethiopia','Finland','France','Georgia','Germany',
    'Ghana','Greece','Guatemala','Hungary','Iceland','India','Indonesia','Iran',
    'Iraq','Ireland','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan',
    'Kenya','Kuwait','Latvia','Lebanon','Libya','Lithuania','Malaysia','Mexico',
    'Moldova','Mongolia','Morocco','Myanmar','Nepal','Netherlands','Nicaragua',
    'Nigeria','Norway','Pakistan','Panama','Paraguay','Peru','Philippines',
    'Poland','Portugal','Romania','Russia','Rwanda','Saudi','Serbia','Slovakia',
    'Slovenia','Somalia','Spain','Sudan','Sweden','Switzerland','Syria','Taiwan',
    'Tanzania','Thailand','Tunisia','Turkey','Uganda','Ukraine','Uruguay',
    'Venezuela','Vietnam','Yemen','Zambia','Zimbabwe',
    # Famous persons — scientists
    'Archimedes','Aristotle','Bohr','Celsius','Copernicus','Curie','Darwin',
    'Descartes','Edison','Einstein','Euler','Faraday','Feynman','Fibonacci',
    'Galileo','Hawking','Heisenberg','Kepler','Leibniz','Linnaeus','Lorentz',
    'Maxwell','Mendel','Mendeleev','Newton','Pascal','Pasteur','Planck','Ptolemy',
    'Pythagoras','Turing',
    # Famous persons — philosophers / writers / composers
    'Aristotle','Caesar','Cervantes','Confucius','Dante','Descartes','Dickens',
    'Dostoevsky','Goethe','Homer','Kant','Lenin','Lincoln','Luther','Machiavelli',
    'Marx','Milton','Moliere','Mozart','Napoleon','Nietzsche','Plato','Rousseau',
    'Socrates','Tchaikovsky','Tolstoy','Voltaire','Wagner',
    # Chemical elements (by full name, capitalized)
    'Actinium','Aluminum','Americium','Antimony','Argon','Arsenic','Astatine',
    'Barium','Berkelium','Beryllium','Bismuth','Bohrium','Boron','Bromine',
    'Cadmium','Calcium','Carbon','Cerium','Cesium','Chlorine','Chromium',
    'Cobalt','Copper','Curium','Dysprosium','Einsteinium','Erbium','Europium',
    'Fermium','Fluorine','Francium','Gadolinium','Gallium','Germanium','Hafnium',
    'Hassium','Helium','Holmium','Hydrogen','Indium','Iodine','Iridium','Iron',
    'Krypton','Lanthanum','Lead','Lithium','Lutetium','Magnesium','Manganese',
    'Meitnerium','Mercury','Molybdenum','Neodymium','Neon','Neptunium','Nickel',
    'Niobium','Nitrogen','Nobelium','Osmium','Oxygen','Palladium','Phosphorus',
    'Platinum','Plutonium','Polonium','Potassium','Praseodymium','Promethium',
    'Protactinium','Radium','Radon','Rhenium','Rhodium','Roentgenium','Rubidium',
    'Ruthenium','Samarium','Scandium','Seaborgium','Selenium','Silicon','Silver',
    'Sodium','Strontium','Sulfur','Tantalum','Technetium','Tellurium','Terbium',
    'Thallium','Thorium','Thulium','Tin','Titanium','Tungsten','Uranium',
    'Vanadium','Xenon','Ytterbium','Yttrium','Zinc','Zirconium',
    # Languages
    'Arabic','Bengali','Bulgarian','Catalan','Croatian','Czech','Danish','Dutch',
    'English','Estonian','Finnish','French','German','Greek','Hebrew','Hindi',
    'Hungarian','Indonesian','Italian','Japanese','Korean','Latvian','Lithuanian',
    'Malay','Norwegian','Persian','Polish','Portuguese','Romanian','Russian',
    'Serbian','Slovak','Slovenian','Spanish','Swedish','Tamil','Turkish','Ukrainian',
    'Vietnamese',
    # Nationalities / adjectives
    'African','American','Asian','Australian','Brazilian','British','Canadian',
    'Chinese','Dutch','Egyptian','European','French','German','Greek','Indian',
    'Iranian','Irish','Italian','Japanese','Korean','Mexican','Norwegian','Polish',
    'Portuguese','Russian','Scottish','Spanish','Swedish','Turkish',
    # Geographical
    'Africa','Alaska','Alps','Amazon','Andes','Antarctica','Arctic','Asia',
    'Atlantic','Australia','Balkans','Baltic','Caspian','Caucasus','Danube',
    'Euphrates','Europe','Ganges','Himalaya','Himalayas','Mediterranean','Nile',
    'Oceania','Pacific','Sahara','Scandinavia','Siberia','Tigris','Volga',
    # Religions and belief
    'Buddhism','Catholic','Christian','Christianity','Confucianism','Hinduism',
    'Islam','Judaism','Protestant','Taoism','Zoroastrianism',
    # Currencies and institutions
    'Bitcoin','Dollar','Euro','Ethereum','Franc','Pound','Ruble','Rupee','Won','Yen',
    'Congress','Parliament','Senate','Kremlin','Pentagon','Vatican','Interpol',
    # Tech brands (single-token likely)
    'Google','Apple','Microsoft','Amazon','Facebook','Twitter','Intel','Oracle',
    'Samsung','Huawei','Tesla','Nvidia',
]
# Deduplicate
CURATED = sorted(set(CURATED))


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


KILLING_PAIRS = [
    ('cat', 'cats'), ('dog', 'dogs'), ('tree', 'trees'), ('bird', 'birds'),
    ('house', 'houses'), ('man', 'woman'), ('king', 'queen'), ('boy', 'girl'),
    ('big', 'bigger'), ('fast', 'faster'), ('old', 'older'),
]


# ── Step 1: Load model + tokenizer ────────────────────────────────────────────
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


# ── Step 2: Filter curated list to single-token words ─────────────────────────
def filter_single_token(tok):
    print(f"\n── Step 2: Filtering curated list to single-token words ────────")
    single = []
    multi  = []
    for w in CURATED:
        for prefix in [' ', '']:
            ids = tok.encode(prefix + w, add_special_tokens=False)
            if len(ids) == 1:
                single.append(w)
                break
        else:
            multi.append(w)
    print(f"  Curated: {len(CURATED)}  single-token: {len(single)}  "
          f"multi-token (skipped): {len(multi)}")
    if multi:
        print(f"  Multi-token examples: {', '.join(multi[:15])}")
    return single


# ── Step 3: L14 hidden states with cache ──────────────────────────────────────
def get_hidden_states(tok, model, words):
    import torch
    print(f"\n── Step 3: L14 hidden states (with cache) ──────────────────────")
    if os.path.exists(PN_CACHE):
        npz = np.load(PN_CACHE, allow_pickle=True)
        cached = {w: npz['hs'][i] for i, w in enumerate(npz['words'])}
        missing = [w for w in words if w not in cached]
        print(f"  Cache found: {len(cached)} words, {len(missing)} missing")
    else:
        cached = {}
        missing = words
        print(f"  No cache. Extracting {len(missing)} words...")

    if missing:
        new_hs = {}
        t0 = time.time()
        for i, w in enumerate(missing):
            for prefix in [' ', '']:
                ids = tok.encode(prefix + w, add_special_tokens=False)
                if len(ids) == 1:
                    inp = tok(prefix + w, return_tensors='pt')
                    with torch.no_grad():
                        out = model(**inp)
                    pos = inp['input_ids'].shape[1] - 1
                    new_hs[w] = out.hidden_states[LAYER + 1][0, pos, :].numpy()
                    break
            if (i + 1) % 100 == 0:
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (i + 1) * (len(missing) - i - 1)
                print(f"  [{i+1:>4}/{len(missing)}]  "
                      f"{elapsed:.1f} min elapsed  ETA {eta:.1f} min")
        cached.update(new_hs)
        # Save/update cache
        all_words = list(cached.keys())
        all_hs    = np.stack([cached[w] for w in all_words])
        np.savez(PN_CACHE, words=all_words, hs=all_hs)
        print(f"  Cache saved: {len(all_words)} words")

    hs = {w: cached[w] for w in words if w in cached}
    print(f"  Hidden states ready: {len(hs)} words at L{LAYER}")
    return hs


# ── Step 4: Z2 axis (same Killing pairs as Day 27) ────────────────────────────
def build_z2(tok, model):
    import torch
    print(f"\n── Step 4: Z2 axis at L{LAYER} ─────────────────────────────────")
    deltas = []
    for a, b in KILLING_PAIRS:
        for prefix in [' ', '']:
            ids_a = tok.encode(prefix + a, add_special_tokens=False)
            ids_b = tok.encode(prefix + b, add_special_tokens=False)
            if len(ids_a) == 1 and len(ids_b) == 1:
                def get_h(word_str):
                    inp = tok(word_str, return_tensors='pt')
                    with torch.no_grad():
                        out = model(**inp)
                    pos = inp['input_ids'].shape[1] - 1
                    return out.hidden_states[LAYER + 1][0, pos, :].numpy()
                ha = get_h(prefix + a)
                hb = get_h(prefix + b)
                d = hb.astype(np.float64) - ha.astype(np.float64)
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
def compute_phi(hs, z2):
    print(f"\n── Step 5: φ-vectors ────────────────────────────────────────────")
    phi = {w: phi_vec(h, z2) for w, h in hs.items()}
    print(f"  φ-vectors: {len(phi)}")
    return phi


# ── Step 6: Cluster → merge ───────────────────────────────────────────────────
def cluster(phi):
    from sklearn.cluster import MiniBatchKMeans
    words = sorted(phi.keys())
    X = np.stack([phi[w] for w in words]).astype(np.float32)
    k = min(K_INIT, len(words) // 5)

    print(f"\n── Step 5: Clustering ───────────────────────────────────────────")
    print(f"  MiniBatchKMeans k={k} on {len(words)} proper nouns...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                          batch_size=1024, max_iter=300)
    labels = km.fit_predict(X)

    raw = {}
    for w, lab in zip(words, labels):
        raw.setdefault(lab, []).append(w)

    clusters = []
    for lab, wlist in raw.items():
        vecs = np.stack([phi[w] for w in wlist])
        c = vecs.mean(axis=0)
        cn = np.linalg.norm(c)
        centroid = c / (cn + 1e-20)
        top = sorted(wlist, key=lambda w: cos_sim(phi[w], centroid), reverse=True)
        coh = float(np.mean([cos_sim(phi[w], centroid) for w in wlist]))
        clusters.append({'words': wlist, 'centroid': centroid,
                         'top_words': top[:TOP_N], 'size': len(wlist),
                         'coherence': coh})

    print(f"  Merging (cos≥{MERGE_COS})...")
    clusters = merge_clusters(clusters)
    clusters.sort(key=lambda c: c['size'], reverse=True)
    print(f"  Final bodies: {len(clusters)}  (from {k} initial)")
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
                    cur['coherence'] = (ni * cur['coherence'] +
                                        nj * clusters[j]['coherence']) / (ni + nj)
                    merged[j] = True
                    changed = True
            merged[i] = True
            new_clusters.append(cur)
        clusters = new_clusters
    return clusters


# ── Step 7: Label ─────────────────────────────────────────────────────────────
def label_clusters(clusters):
    print(f"\n── Step 6: Ollama labelling ─────────────────────────────────────")
    for i, cl in enumerate(clusters):
        label = ollama_label(cl['top_words'], f"P{i:03d}")
        cl['label'] = label
        if i < 5 or i % 20 == 0:
            print(f"  [{i+1:>3}/{len(clusters)}]  P{i:03d} (n={cl['size']:>5},"
                  f" coh={cl['coherence']:.3f}):  {label}")
    return clusters


# ── Step 8: Spot-check known proper nouns ─────────────────────────────────────
def spot_check(clusters, phi):
    print(f"\n── Step 7: Spot-check ───────────────────────────────────────────")
    test = {
        'cities_eu':  ['Berlin', 'Paris', 'Madrid', 'Rome', 'London',
                        'Vienna', 'Amsterdam', 'Brussels', 'Lisbon', 'Prague'],
        'cities_asia':['Tokyo', 'Beijing', 'Seoul', 'Bangkok', 'Singapore',
                        'Mumbai', 'Jakarta', 'Shanghai', 'Dubai', 'Istanbul'],
        'countries':  ['Germany', 'France', 'Japan', 'China', 'Russia',
                        'Brazil', 'India', 'Canada', 'Australia', 'Mexico'],
        'persons':    ['Einstein', 'Newton', 'Darwin', 'Shakespeare', 'Mozart',
                        'Beethoven', 'Napoleon', 'Caesar', 'Lincoln', 'Gandhi'],
        'elements':   ['Hydrogen', 'Helium', 'Lithium', 'Carbon', 'Nitrogen',
                        'Oxygen', 'Sodium', 'Calcium', 'Iron', 'Uranium'],
    }
    map_word_to_body = {w: i for i, cl in enumerate(clusters) for w in cl['words']}

    for category, words in test.items():
        found = [(w, map_word_to_body.get(w)) for w in words if w in phi]
        missing = [w for w in words if w not in phi]
        print(f"\n  [{category}]")
        for w, bid in found:
            label = clusters[bid]['label'] if bid is not None else '?'
            print(f"    {w:<16} → P{bid:03d}  {label}")
        if missing:
            print(f"    missing: {', '.join(missing)}")


# ── Step 9: Save and report ───────────────────────────────────────────────────
def save_and_report(clusters, single_words):
    output = {
        "meta": {
            "n_proper_nouns_curated": len(single_words),
            "n_bodies": len(clusters),
            "layer": f"L{LAYER} (hidden states)",
        },
        "bodies": {
            f"P{i:03d}": {
                "label": cl.get('label', f'P{i}'),
                "size": cl['size'],
                "coherence": cl['coherence'],
                "top_words": cl.get('top_words', cl['words'][:TOP_N]),
            }
            for i, cl in enumerate(clusters)
        },
        "word_map": {w: f"P{i:03d}" for i, cl in enumerate(clusters)
                     for w in cl['words']},
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Atlas saved: {OUTPUT_FILE}")

    print("\n" + "="*70)
    print("DAY 29 — PROPER-NOUN BODY MAP SUMMARY")
    print("="*70)
    print(f"\n  Curated proper nouns (single-token): {len(single_words)}")
    print(f"  Bodies discovered: {len(clusters)}")
    print(f"\n  All bodies (by size):")
    print(f"  {'Body':<8}  {'Size':>5}  {'Coh':>5}  Label")
    print(f"  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*45}")
    for i, cl in enumerate(clusters):
        print(f"  P{i:03d}     {cl['size']:>5}  "
              f"{cl['coherence']:.3f}  {cl.get('label', '?')}")
    print(f"\nDay 29 complete. Atlas: {OUTPUT_FILE}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    tok, model  = load_model()
    single      = filter_single_token(tok)
    hs          = get_hidden_states(tok, model, single)
    z2          = build_z2(tok, model)
    del model   # free RAM before clustering
    phi         = compute_phi(hs, z2)
    clusters    = cluster(phi)
    clusters    = label_clusters(clusters)
    spot_check(clusters, phi)
    save_and_report(clusters, single)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
