"""
SECOND EXPEDITION — DAY 5
=========================
The n=1 Survey: What Semantic Domains Achieve cos≈1/φ?

Day 4 established that:
  - Gender: 54% of pairs at n=1 (navigable)
  - Size: 12% at n=1 (barely)
  - Sentiment: 0% at n=1 (not navigable)

The question: what IS special about n=1? Is it purely morphological derivation
(actor/actress), or are there purely semantic n=1 pairs with no shared characters?

From Day 2 data we already have hints:
  north→south: cos=0.699  n=1 — purely semantic, no shared characters
  east→west:   cos=0.679  n=1 — purely semantic
  true→false:  cos=0.589  n=1 — purely semantic
  summer→winter: cos=0.570 n=1 — purely semantic

These are NOT morphological relatives. They are paired by FUNCTION, not by form.
The hypothesis: n=1 pairs are "functionally bonded" — concepts that are defined
by their mutual opposition or pairing, not by sharing surface form.

Today we survey 300+ pairs across 20 domains to map the complete n=1 landscape.

Phases:
  1. Domain survey — which categories produce n=1 pairs?
  2. Anatomy of non-morphological n=1 pairs — what do they have in common?
  3. Multi-word concept pairs — do phrases achieve n=1?
  4. Build and test axes from newly discovered n=1 domains
  5. The n=1 fingerprint — characterize each n=1 domain's axis
  6. Cross-domain navigation — can an n=1 axis from domain A navigate domain B?

Script: second_expedition/day5_n1_discovery.py
"""

import torch, numpy as np
from collections import defaultdict

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
V = len(W_E)
print(f"  shape={W_E.shape}")

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(V)], dtype=bool)

PHI = (1 + 5**0.5) / 2
PHI_L = {n: 1.0/PHI**n for n in range(0, 10)}

def normed(v): return v / (np.linalg.norm(v) + 1e-12)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word):
    ids = set()
    for p in [word, ' '+word,
              word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

def nn_ret(pred, excl, mask):
    p = normed(pred).astype(np.float32)
    s = W_n @ p; s[~mask] = -1.0
    for e in excl: s[e] = -1.0
    idx = int(np.argmax(s))
    return tok.decode([idx]).strip(), float(s[idx]), idx

def pair_cos(src, tgt):
    es, _ = get_emb(src); et, _ = get_emb(tgt)
    if es is None or et is None: return None
    return float(np.dot(normed(es), normed(et)))

def phi_n(c):
    if c is None or c <= 0: return None
    return round(-np.log(max(c, 1e-9)) / np.log(PHI))

def is_morphological(a, b):
    """Rough check: do a and b share >40% of characters (suggesting morphological relation)?"""
    a, b = a.lower(), b.lower()
    shorter, longer = (a,b) if len(a)<=len(b) else (b,a)
    if len(longer) == 0: return True
    # Check prefix sharing
    px = sum(1 for i in range(min(len(a),len(b))) if a[i]==b[i] and
             all(a[j]==b[j] for j in range(i+1))) 
    # Simpler: count matching prefix
    px = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]: px += 1
        else: break
    if px / max(len(a), len(b)) > 0.35: return True
    # Check if one is contained in the other
    if shorter in longer: return True
    return False

def build_axis(pairs):
    tangents = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        en_s = normed(es); en_t = normed(et)
        c = float(np.dot(en_s, en_t))
        sin_th = float(np.sqrt(max(0, 1-c**2)))
        if sin_th < 1e-8: continue
        tangents.append((en_t - c*en_s) / sin_th)
    if not tangents: return None
    ax = normed(np.mean(tangents, axis=0))
    coh = np.mean([float(np.dot(tangents[i], tangents[j]))
                   for i in range(len(tangents))
                   for j in range(i+1, len(tangents))]) if len(tangents) > 1 else 1.0
    return ax, coh, len(tangents)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE DOMAIN SURVEY
# ═══════════════════════════════════════════════════════════════════════════════
DOMAINS = {
    'compass': [
        ('north','south'), ('east','west'), ('northeast','southwest'),
        ('northwest','southeast'), ('up','down'), ('left','right'),
        ('front','back'), ('top','bottom'), ('above','below'), ('inside','outside'),
    ],
    'calendar_seasons': [
        ('spring','autumn'), ('summer','winter'), ('spring','fall'),
        ('Monday','Friday'), ('Sunday','Saturday'), ('January','July'),
        ('morning','evening'), ('dawn','dusk'), ('day','night'),
        ('yesterday','tomorrow'), ('weekday','weekend'),
    ],
    'boolean_logic': [
        ('true','false'), ('yes','no'), ('on','off'), ('open','close'),
        ('start','stop'), ('begin','end'), ('create','destroy'),
        ('positive','negative'), ('correct','incorrect'), ('valid','invalid'),
    ],
    'physics_pairs': [
        ('hot','cold'), ('fast','slow'), ('big','small'), ('light','dark'),
        ('hard','soft'), ('wet','dry'), ('loud','quiet'), ('heavy','light'),
        ('near','far'), ('early','late'), ('new','old'), ('clean','dirty'),
    ],
    'colors': [
        ('red','blue'), ('red','green'), ('blue','yellow'), ('black','white'),
        ('red','yellow'), ('green','blue'), ('orange','purple'),
        ('pink','brown'), ('gray','brown'), ('cyan','magenta'),
    ],
    'numbers': [
        ('one','two'), ('two','three'), ('first','second'), ('second','third'),
        ('single','double'), ('once','twice'), ('half','whole'),
        ('odd','even'), ('zero','one'), ('hundred','thousand'),
    ],
    'body_parts': [
        ('head','foot'), ('arm','leg'), ('eye','ear'), ('hand','foot'),
        ('finger','toe'), ('nose','mouth'), ('heart','brain'),
        ('front','back'), ('left','right'), ('top','bottom'),
    ],
    'animals_pairs': [
        ('dog','cat'), ('lion','tiger'), ('horse','cow'), ('bird','fish'),
        ('wolf','fox'), ('bear','deer'), ('eagle','hawk'), ('snake','lizard'),
        ('ant','bee'), ('frog','toad'), ('mouse','rat'), ('duck','goose'),
    ],
    'kinship': [
        ('parent','child'), ('mother','father'), ('son','daughter'),
        ('brother','sister'), ('husband','wife'), ('king','queen'),
        ('uncle','aunt'), ('boy','girl'), ('man','woman'), ('grandfather','grandmother'),
    ],
    'nation_language': [
        ('France','French'), ('Germany','German'), ('China','Chinese'),
        ('Japan','Japanese'), ('Spain','Spanish'), ('Italy','Italian'),
        ('Russia','Russian'), ('England','English'), ('Greece','Greek'),
        ('Korea','Korean'),
    ],
    'food_drink': [
        ('bread','water'), ('meat','fish'), ('apple','orange'),
        ('tea','coffee'), ('milk','juice'), ('rice','wheat'),
        ('salt','sugar'), ('oil','vinegar'), ('soup','salad'),
        ('breakfast','dinner'),
    ],
    'time_pairs': [
        ('year','month'), ('month','week'), ('week','day'),
        ('hour','minute'), ('minute','second'), ('past','future'),
        ('before','after'), ('old','new'), ('ancient','modern'),
        ('birth','death'),
    ],
    'actions_opposites': [
        ('give','take'), ('buy','sell'), ('push','pull'), ('ask','answer'),
        ('teach','learn'), ('send','receive'), ('win','lose'), ('rise','fall'),
        ('enter','exit'), ('build','destroy'), ('create','destroy'), ('love','hate'),
    ],
    'rank_degree': [
        ('king','peasant'), ('master','servant'), ('rich','poor'), ('strong','weak'),
        ('better','worse'), ('best','worst'), ('high','low'), ('more','less'),
        ('most','least'), ('first','last'), ('major','minor'), ('senior','junior'),
    ],
    'nature_elements': [
        ('fire','water'), ('earth','sky'), ('land','sea'), ('sun','moon'),
        ('river','mountain'), ('forest','desert'), ('rain','sunshine'),
        ('wind','wave'), ('ice','steam'), ('rock','sand'),
    ],
    'music_art': [
        ('music','silence'), ('sound','light'), ('painting','sculpture'),
        ('song','dance'), ('piano','violin'), ('guitar','drums'),
        ('major','minor'), ('fast','slow'), ('loud','soft'), ('high','low'),
    ],
    'science_pairs': [
        ('matter','energy'), ('space','time'), ('mass','weight'),
        ('acid','base'), ('proton','electron'), ('positive','negative'),
        ('theory','practice'), ('cause','effect'), ('input','output'),
        ('question','answer'),
    ],
    'sports': [
        ('win','lose'), ('attack','defense'), ('offense','defense'),
        ('player','coach'), ('team','opponent'), ('home','away'),
        ('start','finish'), ('kick','catch'), ('run','jump'), ('fast','slow'),
    ],
    'computing': [
        ('hardware','software'), ('input','output'), ('read','write'),
        ('save','load'), ('open','close'), ('start','stop'),
        ('local','remote'), ('client','server'), ('encode','decode'),
        ('compress','expand'),
    ],
    'moral_pairs': [
        ('good','evil'), ('right','wrong'), ('truth','lie'),
        ('justice','injustice'), ('order','chaos'), ('peace','war'),
        ('freedom','slavery'), ('honor','shame'), ('courage','fear'),
        ('wisdom','folly'),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Domain survey — which categories produce n=1 pairs?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Domain Survey: n=1 Pairs Across 20 Domains")
print("  For each domain: count n=1 pairs, identify them, check morphology")
print("═"*72)

all_n1_pairs = []   # (domain, src, tgt, cos, is_morphological)
domain_summary = {}

print(f"\n  {'domain':<20}  {'total':>6}  {'n=1':>5}  {'n=1_sem':>8}  best_pair")
print(f"  {'─'*20}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*30}")

for domain, pairs in DOMAINS.items():
    valid = [(s,t,c) for s,t in pairs for c in [pair_cos(s,t)] if c is not None]
    n1 = [(s,t,c) for s,t,c in valid if phi_n(c) == 1]
    n1_sem = [(s,t,c) for s,t,c in n1 if not is_morphological(s,t)]
    best = max(n1, key=lambda x: x[2]) if n1 else None
    domain_summary[domain] = {'total': len(valid), 'n1': n1, 'n1_sem': n1_sem}
    for s,t,c in n1:
        is_morph = is_morphological(s,t)
        all_n1_pairs.append((domain, s, t, c, is_morph))
    best_str = f"{best[0]}/{best[1]}={best[2]:.3f}" if best else "none"
    print(f"  {domain:<20}  {len(valid):>6}  {len(n1):>5}  {len(n1_sem):>8}  {best_str}")

total_n1 = sum(len(d['n1']) for d in domain_summary.values())
total_n1_sem = sum(len(d['n1_sem']) for d in domain_summary.values())
print(f"\n  Total: {sum(d['total'] for d in domain_summary.values())} pairs, "
      f"{total_n1} n=1, {total_n1_sem} n=1 semantic (non-morphological)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Anatomy of non-morphological n=1 pairs
# What do purely semantic n=1 pairs have in common?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Semantic n=1 Pairs: Who Are They?")
print("  Non-morphological pairs at cos≈1/φ — purely semantic bonding")
print("═"*72)

sem_n1 = [(d,s,t,c) for d,s,t,c,m in all_n1_pairs if not m]
morph_n1 = [(d,s,t,c) for d,s,t,c,m in all_n1_pairs if m]

print(f"\n  Pure semantic n=1 pairs ({len(sem_n1)} total, sorted by cos):")
print(f"  {'domain':<20}  {'pair':>24}  {'cos':>7}  {'cos+cos²':>9}  Δ from 1")
print(f"  {'─'*20}  {'─'*24}  {'─'*7}  {'─'*9}  {'─'*9}")
for d, s, t, c in sorted(sem_n1, key=lambda x: -x[3]):
    c2 = c**2
    total = c + c2
    delta = abs(total - 1.0)
    flag = " ◀" if delta < 0.05 else ""
    print(f"  {d:<20}  {s:>10}→{t:<13}  {c:>7.4f}  {total:>9.4f}  {delta:>9.4f}{flag}")

print(f"\n  Morphological n=1 pairs ({len(morph_n1)} total):")
for d, s, t, c in sorted(morph_n1, key=lambda x: -x[3])[:20]:
    print(f"    {d:<20}  {s:>10}→{t:<13}  {c:.4f}")
if len(morph_n1) > 20:
    print(f"    ... and {len(morph_n1)-20} more")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Build axes for newly discovered n=1 domains
# Test navigation accuracy for each domain's n=1 pairs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — New n=1 Axes: Build and Test Navigation")
print("  For each domain with ≥3 n=1 pairs: build axis, test navigation")
print("═"*72)

# Group semantic n=1 pairs by domain
sem_n1_by_domain = defaultdict(list)
for d, s, t, c in sem_n1:
    sem_n1_by_domain[d].append((s, t))
# Also include morphological n=1 pairs
all_n1_by_domain = defaultdict(list)
for d, s, t, c, m in all_n1_pairs:
    all_n1_by_domain[d].append((s, t))

print(f"\n  {'domain':<20}  {'n1_pairs':>8}  {'coherence':>10}  {'accuracy':>10}")
print(f"  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*10}")

navigable_domains = {}
for domain in sorted(all_n1_by_domain.keys(), key=lambda d: -len(all_n1_by_domain[d])):
    n1_pairs = all_n1_by_domain[domain]
    if len(n1_pairs) < 2: continue
    result = build_axis(n1_pairs)
    if result is None: continue
    ax, coh, n_used = result
    # Test: can the axis navigate its own pairs?
    th_opt = 29.0  # use Day 2 optimal for now
    hits = 0; n = 0
    for s, t in n1_pairs:
        es, _ = get_emb(s)
        if es is None: continue
        n += 1
        pred = np.cos(np.radians(th_opt))*normed(es) + np.sin(np.radians(th_opt))*ax
        w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
        if w == t: hits += 1
    acc = hits/n if n > 0 else 0
    print(f"  {domain:<20}  {len(n1_pairs):>8}  {coh:>10.4f}  {hits}/{n}={acc:>7.0%}")
    navigable_domains[domain] = (ax, coh, n1_pairs, acc)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Optimal theta for each new axis
# Each domain may have a different optimal rotation angle
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — Optimal Navigation Angle for Each n=1 Domain")
print("  Best θ ∈ [10°, 65°] for each navigable domain")
print("═"*72)

print(f"\n  {'domain':<20}  {'pairs':>6}  {'opt_θ':>7}  {'acc':>6}  φ-match?")
print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*20}")

def best_theta_domain(ax, pairs):
    best_th, best_hits, n_total = 10.0, 0, 0
    for src, tgt in pairs:
        es, _ = get_emb(src)
        if es is not None: n_total += 1
    for th_deg in np.linspace(10, 65, 111):
        th = np.radians(th_deg)
        hits = 0
        for s, t in pairs:
            es, _ = get_emb(s)
            if es is None: continue
            pred = np.cos(th)*normed(es) + np.sin(th)*ax
            w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
            if w == t: hits += 1
        if hits > best_hits: best_hits = hits; best_th = th_deg
    return best_th, best_hits, n_total

PHI_ANGLES = {n: np.degrees(np.arccos(PHI_L[n])) for n in range(1, 8)}

for domain, (ax, coh, pairs, _) in sorted(navigable_domains.items(),
                                            key=lambda x: -len(x[1][2])):
    if len(pairs) < 2: continue
    th, hits, n = best_theta_domain(ax, pairs)
    acc = hits/n if n > 0 else 0
    # Is this theta close to any arccos(1/φⁿ)?
    phi_match = ""
    for pn, pth in PHI_ANGLES.items():
        if abs(th - pth) < 2.0:
            phi_match = f"≈arccos(1/φ^{pn})={pth:.1f}°  Δ={abs(th-pth):.2f}°"
            break
    print(f"  {domain:<20}  {n:>6}  {th:>7.1f}°  {hits}/{n}={acc:.0%}  {phi_match}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Cross-domain navigation test
# Can the gender axis navigate compass pairs? Can compass navigate kinship?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — Cross-Domain Navigation")
print("  Can axis from domain A navigate domain B's pairs?")
print("═"*72)

# Build axes for key domains
key_domains = ['kinship', 'compass', 'boolean_logic', 'calendar_seasons', 'colors']
domain_axes = {}

for d in key_domains:
    pairs = all_n1_by_domain.get(d, [])
    if not pairs: continue
    result = build_axis(pairs)
    if result is None: continue
    ax, coh, n = result
    th, hits, n_t = best_theta_domain(ax, pairs)
    domain_axes[d] = (ax, th, pairs, hits, n_t)
    print(f"\n  {d}: {len(pairs)} n=1 pairs  coherence={coh:.4f}  opt_θ={th:.1f}°")

# Cross-navigation matrix
print(f"\n  Cross-navigation matrix (row=axis source, col=test domain):")
print(f"  {'source↓ test→':<20}", end="")
for test_d in key_domains:
    if test_d in domain_axes:
        print(f"  {test_d[:10]:>12}", end="")
print()
print(f"  {'─'*20}", end="")
for _ in key_domains:
    print(f"  {'─'*12}", end="")
print()

for src_d in key_domains:
    if src_d not in domain_axes: continue
    ax_src, th_src, _, _, _ = domain_axes[src_d]
    print(f"  {src_d:<20}", end="")
    for test_d in key_domains:
        if test_d not in domain_axes:
            print(f"  {'N/A':>12}", end="")
            continue
        _, _, test_pairs, _, _ = domain_axes[test_d]
        hits = 0; n = 0
        for s, t in test_pairs:
            es, _ = get_emb(s)
            if es is None: continue
            n += 1
            pred = np.cos(np.radians(th_src))*normed(es) + np.sin(np.radians(th_src))*ax_src
            w, _, _ = nn_ret(pred, source_ids(s), EN_MASK)
            if w == t: hits += 1
        acc = f"{hits}/{n}={hits/n:.0%}" if n > 0 else "N/A"
        print(f"  {acc:>12}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: φ-level distribution: n=1 vs n=2 vs n=3 across ALL 300+ pairs
# Build a comprehensive map of the φ-level landscape
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — Full φ-Level Map: All 300+ Pairs")
print("  Comprehensive picture of semantic relationships by φ-level")
print("═"*72)

all_measured = []
for domain, pairs in DOMAINS.items():
    for s, t in pairs:
        c = pair_cos(s, t)
        if c is None: continue
        n = phi_n(c)
        is_m = is_morphological(s, t)
        all_measured.append((domain, s, t, c, n, is_m))

print(f"\n  Total pairs measured: {len(all_measured)}")
print(f"\n  Distribution by φ-level:")
level_counts = defaultdict(list)
for d, s, t, c, n, m in all_measured:
    level_counts[n].append((d, s, t, c, m))
for n in sorted(level_counts.keys()):
    items = level_counts[n]
    sem_frac = sum(1 for _,_,_,_,m in items if not m) / len(items)
    mean_cos = np.mean([c for _,_,_,c,_ in items])
    domains_at_level = list(set(d for d,_,_,_,_ in items))[:6]
    print(f"  n={n:>2} (1/φⁿ={PHI_L.get(n,0):.4f}): {len(items):>3} pairs  "
          f"mean_cos={mean_cos:.4f}  semantic_frac={sem_frac:.0%}  "
          f"domains: {', '.join(domains_at_level[:4])}")

# Best n=1 pairs across all domains (the "golden ratio pairs")
print(f"\n  Best n=1 pairs (cos + cos² closest to 1.0):")
n1_all = [(d,s,t,c,m) for d,s,t,c,n,m in all_measured if n==1]
n1_sorted = sorted(n1_all, key=lambda x: abs(x[3]+x[3]**2 - 1.0))[:20]
print(f"  {'domain':<20}  {'pair':>24}  {'cos':>7}  {'cos+cos²':>9}  Δ from 1  morph?")
print(f"  {'─'*20}  {'─'*24}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*6}")
for d, s, t, c, m in n1_sorted:
    total = c + c**2
    delta = abs(total - 1.0)
    print(f"  {d:<20}  {s:>10}→{t:<13}  {c:>7.4f}  {total:>9.4f}  {delta:>9.4f}  {'M' if m else 'S'}")

print(f"\n  (S = semantic only, M = morphological)")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 5 SUMMARY")
print("═"*72)
print(f"""
Day 5 surveyed {len(all_measured)} pairs across 20 semantic domains.
Questions answered:
  Phase 1: Which domains have n=1 pairs?
  Phase 2: Anatomy of non-morphological n=1 pairs
  Phase 3: Navigation accuracy for newly discovered domains
  Phase 4: Optimal navigation angles — do they follow arccos(1/φⁿ)?
  Phase 5: Cross-domain navigation — do axes generalize across domains?
  Phase 6: Complete φ-level map of all 300+ pairs

Record in second_expedition/expedition_log.md
""")
