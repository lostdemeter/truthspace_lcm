#!/usr/bin/env python3
"""
Expedition Day 25 — Seed-and-Extend Gravitational Body Mapping

Approach (Option C from Day 24):
  Use the Day-23 L14 φ-centroids as known gravitational seeds. Run forward
  passes for ~150 new extension words to get their L14 hidden states (not
  L0 embeddings — Day 24 showed L0 is too coarse). For each extension word,
  compute its φ vector at L14 and find the nearest seed centroid. Words with
  no close seed form new gravitational bodies. Label everything with Ollama.

This bridges the gap established by Day 24:
  - L0 embeddings: coarse, all content words in same basin
  - L14 hidden states: fine, semantic sub-classes separated
  - Day 25: use known L14 seeds to attract new L14 words

Key questions:
  1. Which new cities are attracted to the city seeds?
  2. Do food words form their own gravitational body?
  3. Do body parts, instruments, vehicles each form distinct bodies?
  4. What is the attraction-threshold boundary (min cos to nearest seed)?
  5. Can Ollama correctly label newly formed bodies from geometry alone?
"""

import sys, os, json, time
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"
MID_COMB     = 14
CRYST_LAYER  = 2
NEW_BODY_THR = 0.30    # max φ-cos to nearest seed below this → new body

# ── Day-23 seed words (known categories) ─────────────────────────────────────
SEEDS = {
    'city_europe':   ['berlin', 'paris', 'madrid', 'vienna', 'london', 'rome'],
    'city_asia':     ['tokyo', 'beijing', 'seoul', 'mumbai', 'bangkok'],
    'city_other':    ['cairo', 'sydney', 'nairobi'],
    'animal_large':  ['elephant', 'rhinoceros', 'hippopotamus', 'giraffe'],
    'animal_primate':['chimpanzee', 'gorilla', 'orangutan'],
    'animal_marine': ['dolphin', 'whale', 'octopus'],
    'animal_bird':   ['penguin', 'eagle', 'parrot'],
    'animal_reptile':['crocodile', 'python', 'iguana'],
    'elem_noble':    ['helium', 'neon', 'argon'],
    'elem_atm':      ['nitrogen', 'oxygen'],
    'elem_solid':    ['carbon', 'silicon', 'sulfur'],
    'elem_metal':    ['iron', 'copper', 'gold', 'silver'],
    'elem_reactive': ['hydrogen', 'sodium', 'potassium'],
    'plural':        ['cats', 'dogs', 'trees', 'birds', 'houses'],
    'gender_pair':   ['man', 'woman', 'king', 'queen', 'boy', 'girl'],
    'comparative':   ['bigger', 'faster', 'older'],
}

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

# ── Extension words to probe (new territory) ─────────────────────────────────
EXTENSIONS = {
    # MORE CITIES — should be attracted to city seeds
    'probe_city_eu2':  ['amsterdam', 'lisbon', 'athens', 'budapest', 'warsaw',
                        'brussels', 'oslo', 'copenhagen', 'helsinki', 'zurich'],
    'probe_city_as2':  ['istanbul', 'dubai', 'singapore', 'karachi', 'jakarta',
                        'manila', 'taipei', 'hanoi', 'tehran', 'riyadh'],
    'probe_city_am':   ['toronto', 'montreal', 'chicago', 'houston', 'miami',
                        'seattle', 'santiago', 'lima', 'bogota', 'havana'],
    # MORE ANIMALS — should be attracted to animal seeds
    'probe_mammal':    ['fox', 'wolf', 'bear', 'lion', 'tiger', 'leopard',
                        'cheetah', 'jaguar', 'panda', 'koala'],
    'probe_fish':      ['salmon', 'tuna', 'shark', 'herring', 'trout',
                        'carp', 'mackerel', 'sardine', 'anchovy', 'eel'],
    # FOOD — new body expected
    'probe_food':      ['bread', 'pasta', 'cheese', 'butter', 'tomato',
                        'potato', 'garlic', 'pepper', 'lemon', 'onion',
                        'sugar', 'honey', 'vinegar', 'mustard', 'cinnamon'],
    # BODY PARTS — new body expected
    'probe_anatomy':   ['liver', 'kidney', 'lung', 'heart', 'brain',
                        'spine', 'muscle', 'artery', 'stomach', 'thyroid',
                        'pancreas', 'bladder', 'intestine', 'trachea'],
    # MUSICAL INSTRUMENTS — new body expected
    'probe_instrument':['piano', 'violin', 'guitar', 'trumpet', 'flute',
                        'cello', 'clarinet', 'saxophone', 'oboe', 'harp',
                        'trombone', 'bassoon', 'mandolin', 'banjo'],
    # VEHICLES — new body expected
    'probe_vehicle':   ['train', 'bicycle', 'motorcycle', 'helicopter',
                        'submarine', 'canoe', 'yacht', 'tractor', 'trolley',
                        'gondola', 'catamaran', 'zeppelin'],
    # PROFESSIONS — new body expected
    'probe_profession':['surgeon', 'dentist', 'architect', 'economist',
                        'geologist', 'biologist', 'chemist', 'historian',
                        'diplomat', 'astronomer', 'philosopher', 'linguist'],
    # MORE ELEMENTS — should be attracted to element seeds
    'probe_elem2':     ['lithium', 'calcium', 'magnesium', 'manganese',
                        'zinc', 'tin', 'lead', 'mercury', 'uranium',
                        'fluorine', 'chlorine', 'bromine', 'iodine'],
    # ABSTRACT CONCEPTS — new body expected
    'probe_abstract':  ['freedom', 'justice', 'courage', 'wisdom', 'mercy',
                        'tyranny', 'democracy', 'sovereignty', 'equality',
                        'liberty', 'dignity', 'solidarity'],
    # WEATHER / NATURE — new body expected
    'probe_nature':    ['volcano', 'glacier', 'tsunami', 'hurricane',
                        'monsoon', 'avalanche', 'earthquake', 'tornado',
                        'drought', 'wildfire', 'blizzard', 'cyclone'],
}


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20: return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_hidden_states(model, tok, word):
    import torch
    for variant in (' ' + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            target_id = ids[0]; break
    else:
        return None
    inputs  = tok(word, return_tensors='pt')
    id_list = inputs['input_ids'][0]
    pos = next((i for i, t in enumerate(id_list) if t.item() == target_id),
               len(id_list) - 1)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return np.stack([hs[0, pos, :].numpy() for hs in out.hidden_states])


def phi_vec(h, z2_axis):
    hn   = h / (np.linalg.norm(h) + 1e-20)
    z2v  = float(np.dot(hn, z2_axis))
    perp = hn - z2v * z2_axis
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20), pm, z2v


def ollama_label(words, fallback='unknown'):
    word_list = ', '.join(words[:20])
    prompt = (
        f"These words share a semantic theme. What single short phrase (2-5 words) "
        f"best describes their semantic category?\n{word_list}\n"
        f"Answer ONLY with the category phrase."
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 20}
        }, timeout=30)
        if r.status_code == 200:
            label = r.json().get('response', '').strip().split('\n')[0]
            return label.strip('"').strip("'")[:60]
    except Exception:
        pass
    return fallback


if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {SMALL_MODEL}...")
    tok   = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32, device_map='cpu')
    model.eval()
    n_layers = model.config.num_hidden_layers

    # ── Cache all words ───────────────────────────────────────────────────────
    all_words = set()
    for words in SEEDS.values():    all_words |= set(words)
    for a, b in KILLING_PAIRS:      all_words |= {a, b}
    for words in EXTENSIONS.values(): all_words |= set(words)

    print(f"  Caching {len(all_words)} words at L{MID_COMB}...")
    cache = {}
    for w in sorted(all_words):
        hs = get_hidden_states(model, tok, w)
        if hs is not None:
            cache[w] = hs
    print(f"  Cached {len(cache)} words.")

    # ── Build Z2 axis ─────────────────────────────────────────────────────────
    comb_deltas = []
    for a, b in KILLING_PAIRS:
        for L in range(CRYST_LAYER, n_layers - 2):
            if a in cache and b in cache:
                d = cache[b][L].astype(np.float64) - cache[a][L].astype(np.float64)
                comb_deltas.append(d / (np.linalg.norm(d) + 1e-20))
    _, sv, Vt = np.linalg.svd(np.stack(comb_deltas), full_matrices=False)
    z2_axis = Vt[0]
    print(f"  Z2 axis: {100*sv[0]**2/np.sum(sv**2):.2f}% variance\n")

    # ── Compute φ for all cached words ────────────────────────────────────────
    phi_cache = {}
    for w in cache:
        p, pm, z2v = phi_vec(cache[w][MID_COMB].astype(np.float64), z2_axis)
        phi_cache[w] = (p, pm, z2v)

    # ── Build seed φ-centroids ────────────────────────────────────────────────
    seed_centroids = {}
    for seed_name, words in SEEDS.items():
        vecs = [phi_cache[w][0] for w in words if w in phi_cache]
        if vecs:
            centroid = np.mean(vecs, axis=0)
            seed_centroids[seed_name] = centroid / (np.linalg.norm(centroid) + 1e-20)

    print(f"{'='*70}")
    print(f"DAY 25 — Seed-and-Extend Gravitational Body Mapping")
    print(f"{'='*70}")

    # ── Section 1: Self-consistency check — do seeds re-attract their own words?
    print(f"\n── Section 1: Seed self-consistency ─────────────────────────────────")
    print(f"  Each seed word should be nearest to its own centroid.\n")
    print(f"  {'Word':<16} {'True seed':<22} {'Nearest seed':<22} {'φ_cos':<8} OK?")
    print("  " + "─"*70)

    self_hits, self_total = 0, 0
    for seed_name, words in SEEDS.items():
        for w in words:
            if w not in phi_cache: continue
            sims = {s: cos_sim(phi_cache[w][0], centroid)
                    for s, centroid in seed_centroids.items() if s != seed_name}
            # Also compare to own centroid (without self-contamination)
            other_vecs = [phi_cache[x][0] for x in words if x != w and x in phi_cache]
            if not other_vecs: continue
            own_c = np.mean(other_vecs, axis=0)
            own_c /= (np.linalg.norm(own_c) + 1e-20)
            own_sim = cos_sim(phi_cache[w][0], own_c)
            sims[seed_name + '*'] = own_sim

            best = max(sims, key=sims.get)
            best_sim = sims[best]
            ok = best == seed_name + '*'
            symbol = '✓' if ok else '✗'
            self_hits += int(ok); self_total += 1
            if not ok or w in ['rome', 'python', 'gold', 'chimpanzee', 'neon']:
                print(f"  {symbol} {w:<16} {seed_name:<22} {best:<22} {own_sim:.4f} {'' if ok else '← MISPLACE'}")

    print(f"\n  Self-consistency: {self_hits}/{self_total} = {100*self_hits/max(self_total,1):.1f}%")

    # ── Section 2: Extension word assignment ──────────────────────────────────
    print(f"\n── Section 2: Extension word assignment ─────────────────────────────")
    print(f"  Assigning new words to nearest seed centroid.")
    print(f"  Threshold for 'new body': max φ_cos < {NEW_BODY_THR}\n")

    assignments = {}      # word → (seed_name, cos_sim) or ('new_body', cos_sim)
    new_body_words = {}   # probe_group → [words not matching any seed]

    for probe_group, words in EXTENSIONS.items():
        group_new = []
        print(f"  ── {probe_group} ──")
        for w in words:
            if w not in phi_cache:
                print(f"    {w}: NOT CACHED")
                continue
            phi_w = phi_cache[w][0]
            sims = {s: cos_sim(phi_w, c) for s, c in seed_centroids.items()}
            best_seed = max(sims, key=sims.get)
            best_sim  = sims[best_seed]

            if best_sim < NEW_BODY_THR:
                status = f"NEW BODY  (best={best_seed} {best_sim:.3f})"
                assignments[w] = ('new_body', best_sim, best_seed)
                group_new.append(w)
            else:
                status = f"→ {best_seed} ({best_sim:.3f})"
                assignments[w] = (best_seed, best_sim, best_seed)

            print(f"    {w:<16} {status}")

        if group_new:
            new_body_words[probe_group] = group_new
        print()

    # ── Section 3: Seed attraction summary ────────────────────────────────────
    print(f"\n── Section 3: Gravitational attraction summary ───────────────────────")
    print(f"  How many new words were attracted to each seed?\n")

    attracted_to = {s: [] for s in seed_centroids}
    new_body_all  = []
    for w, (seed, sim, _) in assignments.items():
        if seed == 'new_body':
            new_body_all.append(w)
        else:
            attracted_to[seed].append((sim, w))

    for seed_name in sorted(attracted_to, key=lambda s: -len(attracted_to[s])):
        words_attracted = attracted_to[seed_name]
        if not words_attracted: continue
        words_attracted.sort(reverse=True)
        top5 = ', '.join(w for _, w in words_attracted[:5])
        rest = f" (+{len(words_attracted)-5} more)" if len(words_attracted) > 5 else ""
        mean_sim = np.mean([s for s, _ in words_attracted])
        print(f"  {seed_name:<22} ← {len(words_attracted):3d} words  "
              f"mean_φ_cos={mean_sim:.3f}   [{top5}{rest}]")

    print(f"\n  New body words ({len(new_body_all)}): "
          f"{', '.join(new_body_all[:20])}"
          f"{' ...' if len(new_body_all) > 20 else ''}")

    # ── Section 4: Cluster the new-body words ────────────────────────────────
    print(f"\n── Section 4: Clustering new gravitational bodies ────────────────────")
    if len(new_body_all) >= 4:
        from sklearn.cluster import AgglomerativeClustering

        new_body_phi = np.stack([phi_cache[w][0] for w in new_body_all if w in phi_cache])
        nb_words = [w for w in new_body_all if w in phi_cache]

        # Choose k based on probe groups that contributed new-body words
        contributing_groups = [g for g, ws in new_body_words.items() if ws]
        n_new_clusters = max(2, min(len(contributing_groups), len(nb_words) // 3))

        clustering = AgglomerativeClustering(
            n_clusters=n_new_clusters, metric='cosine', linkage='average')
        nb_labels = clustering.fit_predict(new_body_phi)

        new_bodies = {}
        for i, w in enumerate(nb_words):
            k = int(nb_labels[i])
            new_bodies.setdefault(k, []).append((cos_sim(new_body_phi[i],
                new_body_phi[nb_labels == k].mean(axis=0)), w))

        print(f"  Found {n_new_clusters} new gravitational bodies in {len(nb_words)} words:\n")
        new_body_labels = {}
        for k, members in sorted(new_bodies.items()):
            members.sort(reverse=True)
            top_words = [w for _, w in members[:8]]
            cohesion  = float(np.mean([s for s, _ in members[:8]]))
            label = ollama_label(top_words, fallback=f'new_body_{k}')
            new_body_labels[k] = label
            print(f"  NB{k}: [{label}]  cohesion={cohesion:.3f}  ({len(members)} words)")
            print(f"        {', '.join(top_words)}")
    else:
        print(f"  Only {len(new_body_all)} new-body words — all attracted to known seeds.")
        new_bodies = {}; new_body_labels = {}

    # ── Section 5: Label the seed bodies via Ollama ───────────────────────────
    print(f"\n── Section 5: Seed body labeling (seeds + attracted words) ──────────")
    print(f"  Ollama labels each gravitational body from its full member list.\n")

    for seed_name in sorted(attracted_to, key=lambda s: -len(attracted_to[s])):
        seed_words  = SEEDS.get(seed_name, [])
        new_members = [w for _, w in sorted(attracted_to[seed_name], reverse=True)[:10]]
        all_members = list(set(seed_words + new_members))
        if not new_members: continue
        label = ollama_label(all_members)
        top_new = ', '.join(w for _, w in sorted(attracted_to[seed_name], reverse=True)[:6])
        mean_sim = np.mean([s for s, _ in attracted_to[seed_name]]) if attracted_to[seed_name] else 0
        print(f"  {seed_name:<22} [{label}]")
        print(f"    seeds:       {', '.join(seed_words[:5])}")
        print(f"    attracted:   {top_new}  (mean φ_cos={mean_sim:.3f})")
        print()

    # ── Section 6: The gravitational atlas ────────────────────────────────────
    print(f"\n── Section 6: Gravitational atlas ────────────────────────────────────")
    print(f"  Complete map of bodies and their member counts at L{MID_COMB}.\n")
    print(f"  {'Body':<28} {'Type':<8} {'Members':>7}  {'Avg φ_cos':>10}  Label")
    print("  " + "─"*80)

    for seed_name in sorted(seed_centroids):
        n  = len(attracted_to[seed_name]) + len(SEEDS.get(seed_name, []))
        ms = (np.mean([s for s, _ in attracted_to[seed_name]])
              if attracted_to[seed_name] else 0.0)
        print(f"  {seed_name:<28} {'seed':<8} {n:>7}  {ms:>10.3f}")

    for k, members in sorted(new_bodies.items()):
        ms = float(np.mean([s for s, _ in members]))
        lbl = new_body_labels.get(k, f'new_body_{k}')
        print(f"  NB{k:<26} {'new':<8} {len(members):>7}  {ms:>10.3f}  [{lbl}]")

    total_mapped = (sum(len(attracted_to[s]) + len(SEEDS.get(s,[])) for s in seed_centroids)
                    + sum(len(m) for m in new_bodies.values()))
    total_words  = len(cache)
    print(f"\n  Total words: {total_words}   Mapped: {total_mapped}   "
          f"Coverage: {100*total_mapped/max(total_words,1):.1f}%")

    # ── Section 7: Summary ────────────────────────────────────────────────────
    print(f"\n── Section 7: Summary ────────────────────────────────────────────────")
    city_attracted = sum(len(attracted_to[s]) for s in attracted_to
                         if s.startswith('city_'))
    animal_attracted = sum(len(attracted_to[s]) for s in attracted_to
                           if s.startswith('animal_'))
    elem_attracted = sum(len(attracted_to[s]) for s in attracted_to
                         if s.startswith('elem_'))
    print(f"""
  Gravitational seeds from Day 23:  {len(seed_centroids)} bodies
  New bodies discovered today:       {len(new_bodies)}
  New words attracted to city seeds: {city_attracted}
  New words attracted to animal seeds:{animal_attracted}
  New words attracted to element seeds:{elem_attracted}
  New body words (no seed match):    {len(new_body_all)}

  The seed-and-extend method bridges Day 23 (L14 fine structure) and
  Day 24 (L0 coarse structure) by running targeted L14 forward passes
  for new words, then comparing directly at L14 resolution.
  Known gravitational bodies attract semantically related new words
  while genuinely novel categories form new bodies.
    """)

    print(f"{'='*70}")
    print(f"Day 25 complete.")
    print(f"{'='*70}")
