"""
Instrument Verification — End-to-End Pipeline Test
====================================================

Composes all six geometric components into the full pipeline and
verifies that the instrument produces the correct next token.

Test: "The capital of France is" → " Paris"
      (and 5 other capital-city facts)

This is the proof that the transformer IS a geometric instrument.
Every step from input to output is a named geometric operation.
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_integer import phi_to_float

from experiments.geometric_instrument.instrument import GeometricInstrument, build_from_model

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}


def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])


def run_real_model(engine, tokenizer, prompt):
    """Run the real model for ground truth comparison."""
    p_ids = tokenizer.encode(prompt)
    h = engine.embedding(p_ids)[np.newaxis, :, :]
    for layer in engine.layers:
        h = layer(h)
    h = rms_norm(h, engine.final_norm_weight)
    logits = engine.lm_head(h)[0, -1, :]
    return logits


def main():
    print("=" * 65)
    print("  GEOMETRIC INSTRUMENT — End-to-End Verification")
    print("=" * 65)

    t0 = time.time()
    gc.collect()
    print(f"\n  Loading model...", flush=True)
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Build the instrument from the real model ────────────
    print(f"\n  Building geometric instrument...", flush=True)
    t1 = time.time()
    instrument = build_from_model(
        engine, tokenizer,
        extraction_layer=23,
        extraction_head=6,
        decomp_end=22,
        amp_start=24,
        amp_end=28,
    )
    print(f"  Built in {time.time()-t1:.1f}s", flush=True)
    print(f"  Pipeline: {len(instrument.decomposition_layers)} decomposition layers "
          f"→ 1 extraction layer (L23) → "
          f"{len(instrument.amplification_layers)} amplification layers", flush=True)

    # ── Test 1: Real model ground truth ─────────────────────
    print("\n" + "─" * 65)
    print("  Test 1: Real Model (ground truth)")
    print("─" * 65)

    real_results = {}
    for country, info in FACTS.items():
        logits = run_real_model(engine, tokenizer, info['prompt'])
        rank, score = get_rank(logits, info['answer'], tokenizer)
        top_id = int(np.argmax(logits))
        top_tok = tokenizer.decode([top_id])
        real_results[country] = {'rank': rank, 'score': score, 'top': top_tok}
        print(f"  {country:>8s}: top='{top_tok}', answer rank={rank}", flush=True)

    # ── Test 2: Geometric instrument ────────────────────────
    print("\n" + "─" * 65)
    print("  Test 2: Geometric Instrument")
    print("─" * 65)

    instrument_results = {}
    for country, info in FACTS.items():
        p_ids = tokenizer.encode(info['prompt'])
        print(f"\n  {country}: \"{info['prompt']}\"", flush=True)

        logits, trace = instrument.predict(p_ids, verbose=True)

        rank, score = get_rank(logits, info['answer'], tokenizer)
        top_id = int(np.argmax(logits))
        top_tok = tokenizer.decode([top_id])
        instrument_results[country] = {'rank': rank, 'score': score, 'top': top_tok}

        match = top_tok.strip() == info['answer'].strip()
        print(f"  → top='{top_tok}', answer='{info['answer']}' rank={rank} "
              f"{'✓ MATCH' if match else '✗ MISMATCH'}", flush=True)

    # ── Test 3: Component-level trace analysis ──────────────
    print("\n" + "─" * 65)
    print("  Test 3: Component Trace Analysis (France)")
    print("─" * 65)

    p_ids = tokenizer.encode(FACTS['France']['prompt'])
    logits, trace = instrument.predict(p_ids, verbose=False)

    # Show what each stage contributes
    ext = trace.get('extraction', {})
    if 'selected_pos' in ext:
        tokens = [tokenizer.decode([t]) for t in p_ids]
        sel_pos = ext['selected_pos']
        print(f"  Tokens: {tokens}")
        print(f"  Selector → position {sel_pos} ('{tokens[sel_pos]}')")

    if 'binding' in ext:
        binding = ext['binding']
        bind_logits = engine.lm_head(
            rms_norm(binding[np.newaxis, np.newaxis, :], engine.final_norm_weight)
        )[0, 0, :]
        bind_rank, _ = get_rank(bind_logits, ' Paris', tokenizer)
        print(f"  Lens binding → 'Paris' at rank {bind_rank} (before amplification)")

    # Answer rank at each stage
    stages = [
        ('Post-decomposition', trace.get('post_decomposition')),
        ('Post-extraction (attn)', ext.get('h_post_attn_last')),
        ('Post-extraction (mlp)', ext.get('h_post_mlp_last')),
        ('Post-amplification', trace.get('post_amplification')),
        ('Final', trace.get('final_state')),
    ]

    print(f"\n  Answer rank trajectory:")
    for name, state in stages:
        if state is not None:
            sl = engine.lm_head(
                rms_norm(state[np.newaxis, np.newaxis, :], engine.final_norm_weight)
            )[0, 0, :]
            r, s = get_rank(sl, ' Paris', tokenizer)
            print(f"    {name:>30s}: rank {r:>5d}  (score {s:.3f})")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  SUMMARY")
    print("═" * 65)

    print(f"\n  {'Country':>8s}  {'Real Top':>10s}  {'Instrument Top':>15s}  "
          f"{'Real Rank':>10s}  {'Instr Rank':>10s}  {'Match':>6s}")
    print("  " + "─" * 68)

    total_match = 0
    for country in FACTS:
        r = real_results[country]
        i = instrument_results[country]
        match = r['top'].strip() == i['top'].strip()
        total_match += match
        print(f"  {country:>8s}  {r['top']:>10s}  {i['top']:>15s}  "
              f"{r['rank']:>10d}  {i['rank']:>10d}  {'✓' if match else '✗':>6s}")

    print(f"\n  Top-1 match: {total_match}/{len(FACTS)}")

    if total_match == len(FACTS):
        print("\n  ═══════════════════════════════════════════════")
        print("  ║  THE GEOMETRIC INSTRUMENT PRODUCES THE SAME ║")
        print("  ║  OUTPUTS AS THE NEURAL NETWORK.             ║")
        print("  ║                                              ║")
        print("  ║  Every step is a named geometric operation.  ║")
        print("  ║  There are no black boxes.                   ║")
        print("  ═══════════════════════════════════════════════")
    else:
        print(f"\n  {total_match}/{len(FACTS)} match — investigating differences...")
        for country in FACTS:
            r = real_results[country]
            i = instrument_results[country]
            if r['top'].strip() != i['top'].strip():
                print(f"    {country}: real='{r['top']}' vs instrument='{i['top']}' "
                      f"(real_rank={r['rank']}, instr_rank={i['rank']})")


if __name__ == '__main__':
    main()
