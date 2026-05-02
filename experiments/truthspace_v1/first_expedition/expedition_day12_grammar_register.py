#!/usr/bin/env python3
"""
Expedition Day 12 — Grammar-Register Correlation

Day 8 found that the Day 4 axis labels were wrong:
  - Ax15 was labeled "comparative degree" but its vocabulary is Romance language words
  - Ax18 was labeled "plural" but its vocabulary is 3rd-person verbs vs ALL-CAPS labels
  - Ax40 was labeled "tense" but has modal/political words

Yet the functional deltas (gender, plural, comparative) DO work (Day 3: LOO rank ≤ 5),
AND they load primarily onto those specific axes (Day 4: highest loading per delta).

The question: WHY does the comparative delta (big→bigger) load onto Ax15 (Romance words)?

Hypothesis 1: The delta IS primarily grammatical but the axis serves multiple roles —
the SVD mixed grammar and register because they correlate in the training data.

Hypothesis 2: The delta loading onto Ax15 is an ARTIFACT. The real comparative
transformation uses register signals (Romance words appear in comparative contexts).

Hypothesis 3: The deltas load on MULTIPLE axes; the single-axis summary (Day 4) was
misleading. The grammar information is distributed, not concentrated.

Tests:
  1. Decompose each delta in the full IRD axis basis — what fraction lands on
     grammar axes vs register axes vs uncategorised axes?
  2. For the comparative delta: how much loads on Ax15 vs Ax9 (abstract/concrete)?
     Are comparative adjectives disproportionately Romance-derived in English?
  3. Test WITHIN-type coherence: do all comparative deltas agree in axis loading,
     or does big→bigger load differently from old→older?
  4. Compute the correlation matrix between all 9 deltas and all 500 axis projections.
     Visualise which axes each delta type preferentially activates.
  5. Project each word in the top/bottom of Ax15 through the comparative delta.
     Does the comparative delta move Romance words differently from Germanic words?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

FUNCTIONAL_RELS = {
    'gender':      [('king','queen'),('man','woman'),('boy','girl'),('actor','actress'),
                    ('prince','princess'),('son','daughter'),('father','mother'),
                    ('brother','sister'),('husband','wife'),('uncle','aunt')],
    'capital':     [('france','paris'),('germany','berlin'),('italy','rome'),
                    ('spain','madrid'),('japan','tokyo'),('china','beijing'),
                    ('russia','moscow'),('brazil','brasilia')],
    'plural':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                    ('book','books'),('tree','trees'),('bird','birds'),
                    ('child','children'),('mouse','mice'),('foot','feet')],
    'past':        [('run','ran'),('walk','walked'),('eat','ate'),('write','wrote'),
                    ('speak','spoke'),('take','took'),('go','went'),('see','saw')],
    'comparative': [('big','bigger'),('small','smaller'),('fast','faster'),
                    ('old','older'),('young','younger'),('strong','stronger')],
    'hypernym':    [('dog','animal'),('apple','fruit'),('paris','city'),
                    ('piano','instrument'),('tiger','animal'),('oak','tree')],
    'antonym':     [('hot','cold'),('happy','sad'),('fast','slow'),
                    ('big','small'),('old','young'),('light','dark')],
    'verb_agent':  [('swim','swimmer'),('teach','teacher'),('write','writer'),
                    ('read','reader'),('drive','driver'),('build','builder')],
}

# Day 8 axis labels
GRAMMAR_AXES  = {2, 5, 7, 15, 17, 18, 40, 54}
REGISTER_AXES = {9, 110, 168, 171, 236, 307, 374, 375}

# Top words on Ax15 (Romance language axis) from Day 8
AX15_POSITIVE = ['ahora','oltre','quiero','Lorenzo','tiene','dopo','cuando','tienen',
                 'porque','Antonio','permite','Rivera','puede','tiempo','Garrett']
AX15_NEGATIVE = ['informational','LAB','philosophical','bib','lic','phys',
                 'memberships','LAND','Anglic','administrations']

# Germanic-derived English comparatives vs Romance-derived
GERMANIC_COMPARATIVES = [('big','bigger'),('old','older'),('fast','faster'),
                          ('small','smaller'),('tall','taller'),('cold','colder')]
ROMANCE_COMPARATIVES  = [('large','larger'),('fine','finer'),('rare','rarer'),
                          ('dense','denser'),('pure','purer'),('grave','graver')]


def learn_delta(lcm, pairs):
    vecs = []
    for a, b in pairs:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            vecs.append(pb.astype(np.float64) - pa.astype(np.float64))
        except RuntimeError:
            pass
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def axis_decomposition(delta, axis_vectors, axis_names, top_n=10):
    """
    Project delta onto each IRD axis, return sorted (axis_name, loading).
    """
    loadings = []
    for i, (av, an) in enumerate(zip(axis_vectors, axis_names)):
        av = av.astype(np.float64)
        av /= (np.linalg.norm(av) + 1e-20)
        loading = float(np.dot(delta, av))
        loadings.append((i, an, loading))
    loadings.sort(key=lambda x: abs(x[2]), reverse=True)
    return loadings


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)
    # Projections are already in the 500-dim axis space.
    # delta[i] = loading on axis i directly — no A_norm needed.
    n_axes = P.shape[1]  # 500

    print(f"\n{'='*65}")
    print(f"DAY 12 — Grammar-Register Correlation")
    print(f"{'='*65}")

    deltas = {}
    for rel, pairs in FUNCTIONAL_RELS.items():
        d = learn_delta(lcm, pairs)
        if d is not None:
            deltas[rel] = d

    axis_idx = list(range(n_axes))

    # ── Section 1: Decompose each delta in the IRD axis basis ─────────────────
    print(f"\n── Section 1: Axis decomposition of each delta (top 8 loadings) ──")
    delta_loadings = {}
    for rel, delta in deltas.items():
        # delta is shape (500,) — each component IS the loading on that axis
        loadings = delta.copy()  # (n_axes,)
        top_idx = np.argsort(np.abs(loadings))[-8:][::-1]
        top_list = [(i, lcm.axis_names[i], float(loadings[i])) for i in top_idx]
        delta_loadings[rel] = loadings

        print(f"\n  Δ_{rel} (||Δ||={np.linalg.norm(delta):.4f}):")
        for ax_i, ax_name, load in top_list:
            tag = " [GRAMMAR]" if ax_i in GRAMMAR_AXES else \
                  " [REGISTER]" if ax_i in REGISTER_AXES else ""
            short_name = ax_name[:40]
            print(f"    Ax{ax_i:<4} {load:+.4f}  {short_name}{tag}")

        # Summary fractions
        gram_mass = float(np.sum(loadings[list(GRAMMAR_AXES)]**2))
        reg_mass  = float(np.sum(loadings[list(REGISTER_AXES)]**2))
        total_sq  = float(np.sum(loadings**2))
        print(f"    Grammar axes energy: {100*gram_mass/total_sq:.1f}%  "
              f"Register axes energy: {100*reg_mass/total_sq:.1f}%  "
              f"Other: {100*(total_sq-gram_mass-reg_mass)/total_sq:.1f}%")

    # ── Section 2: Per-pair loading variance (within-delta coherence) ─────────
    print(f"\n── Section 2: Within-delta coherence (Ax15 loading per pair) ──")
    ax15_idx = 15  # the "Romance language" axis
    ax5_idx  = 5   # the gender axis

    for rel, pairs in FUNCTIONAL_RELS.items():
        loadings_per_pair = []
        for a, b in pairs:
            try:
                pa, _ = lcm._get_proj(a)
                pb, _ = lcm._get_proj(b)
                d = pb.astype(np.float64) - pa.astype(np.float64)
                ax15_load = float(d[ax15_idx])  # direct component in proj space
                loadings_per_pair.append((a, b, ax15_load))
            except RuntimeError:
                pass
        if loadings_per_pair:
            vals = [v for _, _, v in loadings_per_pair]
            print(f"\n  Δ_{rel} Ax15 loading per pair:")
            for a, b, v in loadings_per_pair:
                print(f"    {a:<10}→{b:<12}  Ax15={v:+.4f}")
            print(f"    mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
                  f"range=[{min(vals):+.4f}, {max(vals):+.4f}]")

    # ── Section 3: Germanic vs Romance comparative delta loading ──────────────
    print(f"\n── Section 3: Germanic vs Romance comparative loading on Ax15 ──")
    print(f"  (Does 'big→bigger' load Ax15 more than 'large→larger'?)\n")

    print(f"  Germanic comparatives:")
    germ_loads = []
    for a, b in GERMANIC_COMPARATIVES:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            d = pb.astype(np.float64) - pa.astype(np.float64)
            ax15 = float(d[ax15_idx])
            germ_loads.append(ax15)
            print(f"    {a}→{b}: Ax15={ax15:+.4f}")
        except RuntimeError:
            print(f"    {a}→{b}: not in vocab")

    print(f"  Romance comparatives:")
    rom_loads = []
    for a, b in ROMANCE_COMPARATIVES:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            d = pb.astype(np.float64) - pa.astype(np.float64)
            ax15 = float(d[ax15_idx])
            rom_loads.append(ax15)
            print(f"    {a}→{b}: Ax15={ax15:+.4f}")
        except RuntimeError:
            print(f"    {a}→{b}: not in vocab")

    if germ_loads and rom_loads:
        print(f"\n  Germanic mean Ax15: {np.mean(germ_loads):+.4f}")
        print(f"  Romance mean Ax15:  {np.mean(rom_loads):+.4f}")
        diff = np.mean(rom_loads) - np.mean(germ_loads)
        print(f"  Difference (Romance - Germanic): {diff:+.4f}  "
              f"({'Romance loads MORE on Ax15' if diff > 0 else 'Germanic loads MORE on Ax15'})")

    # ── Section 4: Do Ax15 top words project differently under comparative Δ? ──
    print(f"\n── Section 4: Does comparative Δ move Romance words differently? ──")
    comp_delta = deltas.get('comparative')
    if comp_delta is None:
        print("  (comparative delta not available)")
    else:
        print(f"  Applying Δ_comparative to top Ax15 words:")
        for word in AX15_POSITIVE[:8]:
            try:
                proj, idx = lcm._get_proj(word)
                proj = proj.astype(np.float64)
                after = proj + comp_delta
                # Normalize for retrieval
                after_norm = after / (np.linalg.norm(after) + 1e-20)
                sims = P @ after_norm
                if idx is not None:
                    sims[idx] = -9999
                top3_idx = np.argsort(sims)[-3:][::-1]
                top3 = [(lcm.words[i], float(sims[i])) for i in top3_idx]
                # Ax15 loading: direct component in proj space
                proj_norm = proj / (np.linalg.norm(proj) + 1e-20)
                ax15_orig    = float(proj_norm[ax15_idx])
                ax15_derived = float(after_norm[ax15_idx])
                print(f"    {word:<12} Ax15: {ax15_orig:+.3f} → {ax15_derived:+.3f}  "
                      f"top1: {top3[0][0]}({top3[0][1]:.3f})")
            except RuntimeError:
                pass

    # ── Section 5: Cross-delta correlation matrix ─────────────────────────────
    print(f"\n── Section 5: Cross-delta cosine similarity matrix ──────────")
    rel_list = list(deltas.keys())
    print(f"  {'':>12}" + "".join(f"  {r[:8]:>10}" for r in rel_list))
    for r1 in rel_list:
        d1 = deltas[r1] / (np.linalg.norm(deltas[r1]) + 1e-20)
        row = f"  {r1:>12}"
        for r2 in rel_list:
            d2 = deltas[r2] / (np.linalg.norm(deltas[r2]) + 1e-20)
            cos = float(np.dot(d1, d2))
            row += f"  {cos:+.4f}    "
        print(row)

    # ── Section 6: Axis energy by layer (grammar/register/other) per delta ────
    print(f"\n── Section 6: Energy distribution summary ───────────────────")
    print(f"  {'Delta':<14}  {'Grammar %':<12}  {'Register %':<12}  {'Other %':<10}  Top axis")
    print("  " + "─" * 60)
    for rel, loadings in delta_loadings.items():
        gram_e  = float(np.sum(loadings[list(GRAMMAR_AXES)]**2))
        reg_e   = float(np.sum(loadings[list(REGISTER_AXES)]**2))
        total_e = float(np.sum(loadings**2))
        other_e = total_e - gram_e - reg_e
        top_ax  = int(np.argmax(np.abs(loadings)))
        print(f"  {rel:<14}  {100*gram_e/max(total_e,1e-20):<12.1f}  "
              f"{100*reg_e/max(total_e,1e-20):<12.1f}  {100*other_e/max(total_e,1e-20):<10.1f}  "
              f"Ax{top_ax}({loadings[top_ax]:+.3f})")
