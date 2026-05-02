#!/usr/bin/env python3
"""
Expedition Day 11 — Delta Algebra: Composing Killing Vectors

The IRD has ~9 Killing vectors (functional deltas). Are they a genuine algebra?
In a true Lie group structure:
  - Vectors should compose: Δ_gender ∘ Δ_plural should take king → queens
  - Composition should be (approximately) commutative for orthogonal generators
  - The algebra should be closed: composed vector should land in the same space

Tests:
  A. Sequential composition:    king + Δ_gender = queen; queen + Δ_plural = queens?
                                 (vs)  king + Δ_gender + Δ_plural in one step = queens?
  B. Reverse-composition test:  queens - Δ_plural - Δ_gender = king?
  C. Commutativity test:        Δ_gender + Δ_plural vs Δ_plural + Δ_gender (same result?)
  D. Multi-hop factual:         france + Δ_capital = paris; paris + Δ_capital = ? (chain fails gracefully?)
  E. Cross-type composition:    man + Δ_gender + Δ_plural = women?
  F. Null composition:          king + Δ_gender + (-Δ_gender) = king? (round-trip test)
  G. Scaling test:              king + α*Δ_gender for α in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
                                Does scaled delta produce an interpolation?

The HYPOTHESIS is: if IRD axes are genuine Killing vectors of a symmetric space,
then composition of orthogonal Killing vectors should produce exact traversal
to the combined target.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_library import build_lcm

FUNCTIONAL_RELS = {
    'gender':    [('king','queen'),('man','woman'),('boy','girl'),('actor','actress'),
                  ('prince','princess'),('son','daughter'),('father','mother'),
                  ('brother','sister'),('husband','wife'),('uncle','aunt')],
    'plural':    [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                  ('book','books'),('tree','trees'),('bird','birds'),
                  ('child','children'),('mouse','mice'),('foot','feet')],
    'capital':   [('france','paris'),('germany','berlin'),('italy','rome'),
                  ('spain','madrid'),('japan','tokyo'),('china','beijing'),
                  ('russia','moscow'),('brazil','brasilia')],
    'past':      [('run','ran'),('walk','walked'),('eat','ate'),('write','wrote'),
                  ('speak','spoke'),('take','took'),('go','went'),('see','saw')],
    'comparative': [('big','bigger'),('small','smaller'),('fast','faster'),
                    ('old','older'),('young','younger'),('strong','stronger')],
}


def learn_delta(lcm, pairs):
    vecs = []
    for a, b in pairs:
        try:
            pa, _ = lcm._get_proj(a)
            pb, _ = lcm._get_proj(b)
            vecs.append(pb.astype(np.float64) - pa.astype(np.float64))
        except RuntimeError:
            pass
    return np.mean(vecs, axis=0) if vecs else None


def apply_delta(lcm, word, delta, P, exclude_src=True):
    try:
        src, idx = lcm._get_proj(word)
    except RuntimeError:
        return None, None
    src = src.astype(np.float64)
    derived = src + delta
    derived /= (np.linalg.norm(derived) + 1e-20)
    sims = P @ derived
    if exclude_src and idx is not None:
        sims[idx] = -9999
    top5 = np.argsort(sims)[-5:][::-1]
    return [(lcm.words[i], float(sims[i])) for i in top5], idx


def rank_of(results, target):
    words = [w.lower() for w, _ in results]
    return words.index(target.lower()) + 1 if target.lower() in words else None


if __name__ == '__main__':
    print("Loading LCM...")
    lcm = build_lcm()
    P   = lcm.projections.astype(np.float64)

    print(f"\n{'='*65}")
    print(f"DAY 11 — Delta Algebra: Composing Killing Vectors")
    print(f"{'='*65}")

    deltas = {k: learn_delta(lcm, v) for k, v in FUNCTIONAL_RELS.items()}
    for k, d in deltas.items():
        print(f"  Δ_{k}: ||Δ||={np.linalg.norm(d):.4f}")

    # ── Test A: Sequential vs simultaneous composition ────────────────────────
    print(f"\n── Test A: Sequential vs simultaneous composition ───────────")
    print(f"  king + Δ_gender → queen, then + Δ_plural → queens")
    print(f"  vs king + (Δ_gender + Δ_plural) → queens in one step\n")

    composed = deltas['gender'] + deltas['plural']

    # Sequential
    r1, _ = apply_delta(lcm, 'king', deltas['gender'], P)
    print(f"  Step 1 (king + Δ_gender): {r1[:3]}")
    if r1:
        top1 = r1[0][0]
        r2, _ = apply_delta(lcm, top1, deltas['plural'], P)
        print(f"  Step 2 ({top1} + Δ_plural): {r2[:3]}")

    # Simultaneous
    r3, _ = apply_delta(lcm, 'king', composed, P)
    print(f"  Simultaneous (king + Δ_gender+Δ_plural): {r3[:3]}")

    # More combinations
    combos = [
        ('man',    deltas['gender'],   deltas['plural'],   'women'),
        ('cat',    deltas['plural'],   None,               'cats'),
        ('actor',  deltas['gender'],   deltas['plural'],   'actresses'),
        ('father', deltas['gender'],   None,               'mother'),
        ('france', deltas['capital'],  None,               'paris'),
    ]
    print(f"\n  {'Base':<10}  {'Δ1':<12}  {'Δ2':<12}  {'Target':<12}  Result")
    print("  " + "─" * 60)
    for base, d1, d2, target in combos:
        delta = d1 if d2 is None else (d1 + d2)
        d1_name = [k for k, v in deltas.items() if np.allclose(v, d1)][0]
        d2_name = [k for k, v in deltas.items() if np.allclose(v, d2)][0] if d2 is not None else "—"
        res, _ = apply_delta(lcm, base, delta, P)
        if res is None:
            print(f"  {base:<10}  {d1_name:<12}  {d2_name:<12}  {target:<12}  (base not found)")
            continue
        hit = "✓" if rank_of(res, target) else "✗"
        rk  = rank_of(res, target)
        top1 = f"{res[0][0]}({res[0][1]:.3f})"
        print(f"  {base:<10}  {d1_name:<12}  {d2_name:<12}  {target:<12}  {hit} rank={rk}  → {top1}")

    # ── Test B: Reverse composition ───────────────────────────────────────────
    print(f"\n── Test B: Reverse composition ──────────────────────────────")
    print(f"  queens - Δ_plural - Δ_gender = king?")
    reverse_both = -(deltas['plural'] + deltas['gender'])
    reverse_tests = [
        ('queens',   -(deltas['plural'] + deltas['gender']),  'king'),
        ('actresses',-(deltas['plural'] + deltas['gender']),  'actor'),
        ('women',    -deltas['gender'],                        'man'),
        ('ran',      -deltas['past'],                          'run'),
        ('bigger',   -deltas['comparative'],                   'big'),
    ]
    for src, d, target in reverse_tests:
        res, _ = apply_delta(lcm, src, d, P)
        if res is None:
            print(f"  {src} → (not found)")
            continue
        hit = "✓" if rank_of(res, target) else "✗"
        top3 = ", ".join(f"{w}({s:.3f})" for w, s in res[:3])
        print(f"  {src:<12} → target={target:<10}  {hit} rank={rank_of(res, target)}  | {top3}")

    # ── Test C: Commutativity ─────────────────────────────────────────────────
    print(f"\n── Test C: Commutativity ─────────────────────────────────────")
    print(f"  Δ_gender + Δ_plural vs Δ_plural + Δ_gender (should be same since addition commutes)")
    d_gp = deltas['gender'] + deltas['plural']
    d_pg = deltas['plural'] + deltas['gender']
    cos_comm = float(np.dot(d_gp / np.linalg.norm(d_gp), d_pg / np.linalg.norm(d_pg)))
    print(f"  Cosine(Δ_g+Δ_p, Δ_p+Δ_g) = {cos_comm:.6f}  (1.0 = perfectly commutative)")
    for base in ['king', 'man', 'actor', 'cat', 'dog']:
        r_gp, _ = apply_delta(lcm, base, d_gp, P)
        r_pg, _ = apply_delta(lcm, base, d_pg, P)
        if r_gp and r_pg:
            same = r_gp[0][0] == r_pg[0][0]
            print(f"  {base}: gp→{r_gp[0][0]}  pg→{r_pg[0][0]}  {'same' if same else 'DIFFERENT'}")

    # ── Test D: Multi-hop factual chain ───────────────────────────────────────
    print(f"\n── Test D: Multi-hop chain ──────────────────────────────────")
    print(f"  france +capital→ paris +capital→ ??? (should fail gracefully)\n")
    chain = ['france']
    for _ in range(3):
        prev = chain[-1]
        res, _ = apply_delta(lcm, prev, deltas['capital'], P)
        if res:
            chain.append(res[0][0])
            print(f"  {prev} +capital→ {res[0][0]}  (top5: {[w for w,_ in res[:5]]})")
        else:
            print(f"  {prev} → (not found)")
            break

    # ── Test F: Round-trip (null composition) ─────────────────────────────────
    print(f"\n── Test F: Round-trip (null composition) ────────────────────")
    print(f"  word + Δ - Δ = word?  (should recover original)\n")
    for word in ['king', 'france', 'cat', 'run', 'big']:
        try:
            orig, _ = lcm._get_proj(word)
        except RuntimeError:
            continue
        orig = orig.astype(np.float64)
        for rel, d in deltas.items():
            after = orig + d
            back  = after - d
            cos_rt = float(np.dot(back / np.linalg.norm(back), orig / np.linalg.norm(orig)))
            if abs(cos_rt - 1.0) > 1e-8:
                print(f"  {word} +{rel} -Δ: cos_roundtrip={cos_rt:.6f}  (should be 1.0)")
            break  # just check once per word
        after = orig + deltas['gender']
        back  = after - deltas['gender']
        cos_rt = float(np.dot(back/np.linalg.norm(back), orig/np.linalg.norm(orig)))
        print(f"  {word}: cos(original, after_roundtrip) = {cos_rt:.6f}")

    # ── Test G: Scaled delta ──────────────────────────────────────────────────
    print(f"\n── Test G: Scaled delta — does α*Δ interpolate? ─────────────")
    print(f"  king + α*Δ_gender for α in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]\n")
    print(f"  {'α':<6}  Top-5 results")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        res, _ = apply_delta(lcm, 'king', alpha * deltas['gender'], P)
        if res:
            top5 = ", ".join(f"{w}({s:.3f})" for w, s in res[:5])
            print(f"  {alpha:<6.2f}  {top5}")
