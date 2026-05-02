#!/usr/bin/env python3
"""
Expedition Day 53 — Mathematical Quantum Properties of Zone C

Hypothesis: φ-vectors on the unit hypersphere ARE a mathematical Hilbert
space. Quantum-like properties exist in the mathematical structure itself,
not in physical quantum hardware. Any classical system (brain, silicon)
that processes this structure exploits those properties implicitly.

Four tests:
  P1  Three H0 super-clusters — what are the three macro-domains of Zone C?
  P2  T2 non-commutativity  — do morphological operators commute?
  P3  Fourier duality        — is axis space the Fourier conjugate of T2 space?
  P4  Von Neumann entropy    — do bodies have quantum-mixed-state character?
"""

import json, random, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
CACHE_FILE  = str(SCRIPT_DIR / "day27_hs_cache.npz")
ATLAS_FILE  = str(SCRIPT_DIR / "day27_atlas.json")
OUTPUT_FILE = str(SCRIPT_DIR / "day53_mathematical_quantum.json")

N_AXES      = 43
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("  Expedition Day 53 — Mathematical Quantum Properties of Zone C")
print("=" * 70)


# ── Rebuild geometry ──────────────────────────────────────────────────────────
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c_words  = [w for w, v in wmap.items()
                 if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None)
                 and w in w2i]
zone_c_bodies = {w: wmap[w]['L14_body'] for w in zone_c_words}

KILLING_PAIRS = [
    ('cat','cats'), ('dog','dogs'), ('tree','trees'), ('bird','birds'),
    ('house','houses'), ('man','woman'), ('king','queen'), ('boy','girl'),
    ('big','bigger'), ('fast','faster'), ('old','older'),
]

def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20: deltas.append(d / dm)
            break
_, _, Vt_d = np.linalg.svd(np.stack(deltas), full_matrices=False)
z2 = Vt_d[0] / np.linalg.norm(Vt_d[0])

zone_c_idx  = np.array([w2i[w] for w in zone_c_words])
phi_c14     = batch_phi(hs14_all[zone_c_idx], z2)

body_members = defaultdict(list)
for i, w in enumerate(zone_c_words):
    body_members[zone_c_bodies[w]].append(i)

def build_centroids(bm, min_m=3):
    C = {}
    for body, idxs in bm.items():
        if len(idxs) < min_m: continue
        v = phi_c14[idxs].mean(0)
        nm = np.linalg.norm(v)
        if nm > 1e-20: C[body] = v / nm
    return C

centroids_full = build_centroids(body_members)
bodies_list    = sorted(centroids_full.keys())
C_mat          = np.stack([centroids_full[b] for b in bodies_list])

_, _, Vt_c = np.linalg.svd(C_mat, full_matrices=False)
N_AXES_ACT = min(N_AXES, Vt_c.shape[0])
AXES_MAT   = Vt_c[:N_AXES_ACT]

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


# ── P1: Three H0 Super-Clusters ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P1 — Three H0 Super-Clusters (connected components at cosine dist = 0.30)")
print(f"{'='*70}")

n_b  = len(bodies_list)
D_bb = np.zeros((n_b, n_b))
for i in range(n_b):
    for j in range(n_b):
        D_bb[i,j] = 1.0 - cosine(C_mat[i], C_mat[j])

BRIDGE_THRESHOLD = 0.30

parent = list(range(n_b))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for i in range(n_b):
    for j in range(i+1, n_b):
        if D_bb[i,j] <= BRIDGE_THRESHOLD:
            union(i, j)

components = defaultdict(list)
for i, b in enumerate(bodies_list):
    components[find(i)].append(b)

component_list = sorted(components.values(), key=lambda x: -len(x))
print(f"\n  {len(component_list)} connected components at threshold={BRIDGE_THRESHOLD}")

p1_results = []
for ci, comp in enumerate(component_list):
    comp_idxs = []
    for body in comp:
        comp_idxs.extend(body_members.get(body, []))

    # Representative label for each body
    rep_labels = []
    for body in comp:
        for w in zone_c_words:
            if zone_c_bodies.get(w) == body:
                lab = wmap[w].get('L14_label', '')
                if lab and lab != '?':
                    rep_labels.append(lab)
                    break

    if comp_idxs:
        sc_centroid = phi_c14[comp_idxs].mean(0)
        sc_centroid /= (np.linalg.norm(sc_centroid) + 1e-20)
        sims = phi_c14[comp_idxs] @ sc_centroid
        top_local = np.argsort(-sims)[:15]
        top_words = [zone_c_words[comp_idxs[j]] for j in top_local]
    else:
        top_words = []

    print(f"\n  Super-cluster {ci+1}:  {len(comp)} bodies  |  {len(comp_idxs)} words")
    unique_labels = list(dict.fromkeys(l for l in rep_labels if l))[:12]
    print(f"    Body labels: {', '.join(unique_labels)}")
    print(f"    Top words:   {', '.join(w.strip() for w in top_words[:15])}")

    p1_results.append({
        'cluster': ci + 1,
        'n_bodies': len(comp),
        'n_words': len(comp_idxs),
        'bodies': comp,
        'body_labels': unique_labels,
        'top_words': [w.strip() for w in top_words],
    })


# ── P2: T2 Non-Commutativity ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P2 — T2 Operator Non-Commutativity  [T2_A, T2_B] = ?")
print(f"  Classical: operators commute (AB = BA).")
print(f"  Quantum-like: operators do NOT commute (AB ≠ BA).")
print(f"{'='*70}")

T2_SEEDS = {
    'singular_plural':  [(' cat',' cats'),(' dog',' dogs'),(' tree',' trees'),
                         (' bird',' birds'),(' book',' books')],
    'male_female':      [(' king',' queen'),(' man',' woman'),(' boy',' girl'),
                         (' actor',' actress'),(' prince',' princess')],
    'base_comp':        [(' big',' bigger'),(' fast',' faster'),(' old',' older'),
                         (' small',' smaller'),(' tall',' taller')],
    'base_adverb':      [(' quick',' quickly'),(' slow',' slowly'),
                         (' clear',' clearly'),(' soft',' softly'),
                         (' quiet',' quietly')],
    'base_gerund':      [(' run',' running'),(' walk',' walking'),
                         (' talk',' talking'),(' think',' thinking'),
                         (' play',' playing')],
}

def build_t2(pairs):
    deltas_t = []
    for a, b in pairs:
        wa, wb = a, b
        if wa not in w2i: wa = a.strip()
        if wb not in w2i: wb = b.strip()
        if wa in w2i and wb in w2i:
            pa = batch_phi(hs14_all[[w2i[wa]]], z2)[0]
            pb = batch_phi(hs14_all[[w2i[wb]]], z2)[0]
            d = pb - pa
            nm = np.linalg.norm(d)
            if nm > 1e-20: deltas_t.append(d / nm)
    if not deltas_t: return None
    mean_d = np.stack(deltas_t).mean(0)
    nm = np.linalg.norm(mean_d)
    return mean_d / (nm + 1e-20) if nm > 1e-20 else None

t2_vecs = {}
for name, pairs in T2_SEEDS.items():
    v = build_t2(pairs)
    if v is not None:
        t2_vecs[name] = v
        print(f"  Built T2[{name}]")

def apply_t2(phi_v, t2_delta):
    result = phi_v + t2_delta
    nm = np.linalg.norm(result)
    return result / (nm + 1e-20)

TEST_WORDS = [' king', ' man', ' boy', ' actor', ' prince']
gender_op = t2_vecs.get('male_female')
plural_op = t2_vecs.get('singular_plural')

p2_results = []
if gender_op is not None and plural_op is not None:
    print(f"\n  gender then plural  vs  plural then gender:")
    print(f"  {'Word':<10s}  {'A=g→p top':>12s}  {'B=p→g top':>12s}  cos(A,B)  commutes?")
    print(f"  {'-'*65}")
    for w in TEST_WORDS:
        wk = w if w in w2i else w.strip()
        if wk not in w2i: continue
        phi_w  = batch_phi(hs14_all[[w2i[wk]]], z2)[0]
        phi_ab = apply_t2(apply_t2(phi_w, gender_op), plural_op)
        phi_ba = apply_t2(apply_t2(phi_w, plural_op), gender_op)
        c = cosine(phi_ab, phi_ba)
        sims_ab = phi_c14 @ phi_ab
        sims_ba = phi_c14 @ phi_ba
        top_ab = zone_c_words[int(np.argmax(sims_ab))].strip()
        top_ba = zone_c_words[int(np.argmax(sims_ba))].strip()
        commutes = c > 0.9995
        print(f"  {wk.strip():<10s}  {top_ab:>12s}  {top_ba:>12s}  {c:.4f}    {'yes' if commutes else 'NO'}")
        p2_results.append({'word': wk.strip(), 'cos_ab_ba': float(c),
                           'top_ab': top_ab, 'top_ba': top_ba})

    mean_c = float(np.mean([r['cos_ab_ba'] for r in p2_results]))
    comm_str = 'NON-COMMUTING (quantum-like)' if mean_c < 0.9995 else 'COMMUTING (classical)'
    print(f"\n  Mean cos(AB,BA) = {mean_c:.4f}  → {comm_str}")
else:
    mean_c = 1.0
    comm_str = 'could not test'
    print("  Operators not available.")

comp_op   = t2_vecs.get('base_comp')
gerund_op = t2_vecs.get('base_gerund')
TEST_ADJ  = [' big', ' fast', ' old', ' slow', ' tall']
p2b_results = []
if comp_op is not None and gerund_op is not None:
    print(f"\n  comp then gerund  vs  gerund then comp:")
    vals = []
    for w in TEST_ADJ:
        wk = w if w in w2i else w.strip()
        if wk not in w2i: continue
        phi_w  = batch_phi(hs14_all[[w2i[wk]]], z2)[0]
        phi_ab = apply_t2(apply_t2(phi_w, comp_op), gerund_op)
        phi_ba = apply_t2(apply_t2(phi_w, gerund_op), comp_op)
        c = cosine(phi_ab, phi_ba)
        vals.append(c)
        p2b_results.append({'word': wk.strip(), 'cos_ab_ba': float(c)})
        print(f"  {wk.strip():<10s}  cos(AB,BA)={c:.4f}")
    print(f"  Mean: {float(np.mean(vals)):.4f}")


# ── P3: Fourier Duality ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P3 — Axis ↔ T2 Fourier Duality and Uncertainty Analog")
print(f"{'='*70}")

t2_names = list(t2_vecs.keys())
T2_BASIS = np.stack([t2_vecs[n] for n in t2_names])

n_sample  = min(500, len(zone_c_words))
sample_idx = np.random.choice(len(zone_c_words), n_sample, replace=False)

A_coords = np.zeros((n_sample, N_AXES_ACT))
T_coords = np.zeros((n_sample, len(t2_names)))
for k, i in enumerate(sample_idx):
    phi = phi_c14[i]
    A_coords[k] = AXES_MAT @ phi
    T_coords[k] = T2_BASIS @ phi

A_n  = A_coords - A_coords.mean(0)
T_n  = T_coords - T_coords.mean(0)
A_std = np.std(A_n, axis=0) + 1e-20
T_std = np.std(T_n, axis=0) + 1e-20

cross_cor = (A_n.T @ T_n) / (n_sample * np.outer(A_std, T_std))

print(f"\n  Axis–T2 cross-correlation (max|r| per operator):")
for j, name in enumerate(t2_names):
    col = cross_cor[:, j]
    print(f"    {name:<25s}: max|r|={np.abs(col).max():.4f}  "
          f"(axis {int(np.argmax(np.abs(col)))+1})")

max_xcor = float(np.abs(cross_cor).max())
mean_xcor = float(np.abs(cross_cor).mean())
print(f"\n  Max cross-correlation:  {max_xcor:.4f}")
print(f"  Mean cross-correlation: {mean_xcor:.4f}")
print(f"  (Near 0 = independent subspaces = Fourier-conjugate-like)")
print(f"  (Near 1 = same information = not conjugate)")

# Variance in T2 explained by axis coords
print(f"\n  T2 variance explained by top-k axis coordinates:")
U, s, Vt = np.linalg.svd(A_n, full_matrices=False)
for k in [5, 10, 20, N_AXES]:
    A_k    = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    denom  = np.var(T_n) + 1e-20
    resid  = np.var(T_n - (A_k @ np.linalg.lstsq(A_k, T_n, rcond=None)[0]))
    r2     = float(1.0 - resid / denom)
    print(f"  Top-{k:>2d} axes: T2 var explained = {r2:.4f}")

# Uncertainty analog
axis1_vals = np.abs(A_coords[:, 0])
T2_spread  = np.std(T_coords, axis=1)
rho_unc, p_unc = spearmanr(axis1_vals, T2_spread)
print(f"\n  Uncertainty principle analog:")
print(f"  Spearman ρ(|axis1_coord|, T2_spread) = {rho_unc:.4f}  p={p_unc:.4f}")
if rho_unc > 0.1 and p_unc < 0.05:
    print(f"  → POSITIVE: well-localised in axis → more spread in T2  (quantum-like)")
elif rho_unc < -0.1 and p_unc < 0.05:
    print(f"  → NEGATIVE: localised in axis → also localised in T2  (classical)")
else:
    print(f"  → NONE: axis and T2 localisation are independent")


# ── P4: Von Neumann Entropy ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"P4 — Von Neumann Entropy of Body Density Matrices")
print(f"  rho = (1/n) sum_i |phi_i><phi_i|   in 43-dim axis space")
print(f"  S = -Tr(rho log rho)")
print(f"{'='*70}")

vn_results = []
for body in sorted(body_members.keys()):
    idxs = body_members[body]
    if len(idxs) < 4: continue
    vecs   = phi_c14[idxs]
    coords = vecs @ AXES_MAT.T
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / (norms + 1e-20)

    rho     = (coords.T @ coords) / len(idxs)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-15, None)
    eigvals = eigvals / eigvals.sum()
    S       = float(-np.sum(eigvals * np.log(eigvals)))
    S_max   = math.log(min(len(idxs), N_AXES_ACT))
    purity  = float(np.sum(eigvals**2))

    label = '?'
    for w in zone_c_words:
        if zone_c_bodies.get(w) == body:
            lab = wmap[w].get('L14_label', '?')
            if lab and lab != '?':
                label = lab
                break

    vn_results.append({
        'body': body, 'label': label, 'n_words': len(idxs),
        'entropy': S, 'entropy_norm': S / S_max if S_max > 0 else 0.0,
        'purity': purity,
    })

vn_results.sort(key=lambda x: -x['entropy'])

print(f"\n  Top-15 bodies by Von Neumann entropy  (high S = broad/mixed concept):")
print(f"  {'Body':<8s}  {'N':>4s}  {'S':>6s}  {'S/Smax':>7s}  {'Purity':>8s}  Label")
print(f"  {'-'*72}")
for r in vn_results[:15]:
    print(f"  {r['body']:<8s}  {r['n_words']:>4d}  {r['entropy']:>6.3f}  "
          f"{r['entropy_norm']:>7.4f}  {r['purity']:>8.5f}  {r['label']}")

print(f"\n  Bottom-10 bodies  (low S = narrow/'pure' concept):")
print(f"  {'Body':<8s}  {'N':>4s}  {'S':>6s}  {'S/Smax':>7s}  {'Purity':>8s}  Label")
print(f"  {'-'*72}")
for r in vn_results[-10:]:
    print(f"  {r['body']:<8s}  {r['n_words']:>4d}  {r['entropy']:>6.3f}  "
          f"{r['entropy_norm']:>7.4f}  {r['purity']:>8.5f}  {r['label']}")

entropies = [r['entropy'] for r in vn_results]
purities  = [r['purity']  for r in vn_results]
n_words_l = [r['n_words'] for r in vn_results]
rho_se, _ = spearmanr(n_words_l, entropies)
rho_sn, _ = spearmanr(n_words_l, [r['entropy_norm'] for r in vn_results])

print(f"\n  Entropy:  mean={float(np.mean(entropies)):.3f}  "
      f"std={float(np.std(entropies)):.3f}  "
      f"range=[{min(entropies):.3f},{max(entropies):.3f}]")
print(f"  Purity:   mean={float(np.mean(purities)):.5f}  "
      f"std={float(np.std(purities)):.5f}")
print(f"  ρ(n_words, S) = {rho_se:.3f}  "
      f"{'(scales with size)' if abs(rho_se) > 0.7 else '(independent of size)'}")
print(f"  ρ(n_words, S/Smax) = {rho_sn:.3f}  "
      f"{'(still size-dependent)' if abs(rho_sn) > 0.5 else '(truly independent of size)'}")


# ── Final Verdict ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"VERDICT — Mathematical Quantum Properties")
print(f"{'='*70}")

mean_comm = float(np.mean([r['cos_ab_ba'] for r in p2_results])) if p2_results else 1.0
comm_verdict = mean_comm < 0.9995
mean_purity  = float(np.mean(purities)) if purities else 1.0
mixed_verdict = mean_purity < 0.5
unc_verdict  = bool(rho_unc > 0.1 and p_unc < 0.05)
indep_verdict = max_xcor < 0.3

print(f"""
  Property           Result                             Quantum-like?
  ─────────────────────────────────────────────────────────────────────
  P1 Super-clusters  {len(component_list)} macro-domains at t=0.30              —
  P2 Non-commutat.   cos(AB,BA)={mean_comm:.4f}                      {'YES' if comm_verdict else 'NO'}
  P3 Independence    max cross-cor(axis,T2)={max_xcor:.4f}           {'YES (subspaces separate)' if indep_verdict else 'NO (correlated)'}
     Uncertainty     rho_unc={rho_unc:.4f} p={p_unc:.4f}             {'YES' if unc_verdict else 'NO'}
  P4 Von Neumann     mean purity={mean_purity:.5f}                   {'MIXED STATES' if mixed_verdict else 'NEAR-PURE'}
  ─────────────────────────────────────────────────────────────────────
""")

n_pass = sum([comm_verdict, indep_verdict, unc_verdict, mixed_verdict])
if n_pass >= 3:
    final = "MATHEMATICAL QUANTUM — multiple quantum-like properties confirmed"
elif n_pass >= 2:
    final = "PARTIALLY QUANTUM-LIKE — some properties present"
else:
    final = "CLASSICAL — structure is classically characterised"

print(f"  {final}")


# ── Save ──────────────────────────────────────────────────────────────────────
def to_py(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, list): return [to_py(v) for v in x]
    if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
    return x

output = {
    'p1_super_clusters': p1_results,
    'p2_non_commutativity': {
        'gender_plural_pairs': p2_results,
        'comp_gerund_pairs': p2b_results,
        'mean_cos_ab_ba': mean_comm,
        'verdict': 'NON-COMMUTING' if comm_verdict else 'COMMUTING',
    },
    'p3_fourier_duality': {
        'max_cross_cor': max_xcor,
        'mean_cross_cor': mean_xcor,
        'uncertainty_rho': float(rho_unc),
        'uncertainty_p': float(p_unc),
    },
    'p4_von_neumann': {
        'bodies': vn_results,
        'mean_entropy': float(np.mean(entropies)),
        'mean_purity': mean_purity,
        'verdict': 'MIXED' if mixed_verdict else 'NEAR-PURE',
    },
    'final_verdict': final,
    'n_quantum_properties': n_pass,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(to_py(output), f, indent=2)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 53 complete.")
