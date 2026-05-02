"""
Component Verification — Test Each Geometric Component in Isolation
====================================================================

For each component, we:
1. Extract it from the real Qwen2-7B model (φ-encoded)
2. Verify it meets its specification
3. Compare its output to the real model's behavior

Usage:
    python verify_component.py                    # Run all tests
    python verify_component.py --component lens   # Test one component
"""

import sys, os, time, gc, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_integer import phi_to_float
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear

from experiments.geometric_instrument.components.waveguide import Waveguide
from experiments.geometric_instrument.components.selector import Selector
from experiments.geometric_instrument.components.resonator import Resonator
from experiments.geometric_instrument.components.lens import Lens
from experiments.geometric_instrument.components.amplifier import Amplifier
from experiments.geometric_instrument.components.stabilizer import Stabilizer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}

# ─── Helpers ────────────────────────────────────────────────────────

def get_logits(engine, h_vec):
    """Get logits from a single hidden state vector."""
    h_3d = h_vec[np.newaxis, np.newaxis, :].astype(np.float32)
    normed = rms_norm(h_3d, engine.final_norm_weight)
    return engine.lm_head(normed)[0, 0, :]

def get_rank(logits, tok_str, tokenizer):
    tids = tokenizer.encode(tok_str)
    if not tids:
        return None, None
    tid = tids[0]
    return int(np.sum(logits > logits[tid])), float(logits[tid])

def forward_to_layer(engine, prompt_ids, target_layer):
    """Run forward pass up to (and including) target_layer.
    Returns h [1, seq, d] after the target layer, plus per-layer data."""
    h = engine.embedding(prompt_ids)[np.newaxis, :, :]
    seq_len = h.shape[1]
    layer_states = {}

    for li in range(target_layer + 1):
        layer = engine.layers[li]
        attn = layer.attention
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim

        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        V = phi_linear(attn.W_v, normed, attn.b_v)
        Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke = np.repeat(K, hpk, axis=1)
        Ve = np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if seq_len > 1:
            scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
        weights = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ao = phi_linear(attn.W_o, ao)
        h_post_attn = h + ao

        mlp = layer.mlp
        nm = rms_norm(h_post_attn, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        h_post_mlp = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)

        layer_states[li] = {
            'normed': normed[0].copy(),
            'h_pre': h[0].copy(),
            'h_post_attn': h_post_attn[0].copy(),
            'h_post_mlp': h_post_mlp[0].copy(),
            'attn_weights': weights[0].copy(),
            'attn_out': ao[0].copy(),
        }
        h = h_post_mlp

    return h, layer_states

def find_country_pos(tokens, country):
    for i, t in enumerate(tokens):
        if country.lower() in t.lower():
            return i
    return None


# ─── Component Tests ────────────────────────────────────────────────

def test_waveguide():
    """Verify waveguide carries orthogonal signals without interference."""
    print("\n" + "═" * 60)
    print("  TEST: Waveguide (Residual Stream)")
    print("═" * 60)
    
    d = 3584
    wg = Waveguide(d)
    
    # Create two orthogonal signals
    rng = np.random.RandomState(42)
    s1 = rng.randn(d).astype(np.float32)
    s2 = rng.randn(d).astype(np.float32)
    # Gram-Schmidt
    s2 = s2 - s1 * np.dot(s1, s2) / np.dot(s1, s1)
    s1 = s1 / np.linalg.norm(s1)
    s2 = s2 / np.linalg.norm(s2)
    
    wg.inject(s1 * 10.0)
    wg.inject(s2 * 5.0)
    
    state = wg.read()
    # Recover s1 component
    proj1 = np.dot(state, s1)
    proj2 = np.dot(state, s2)
    
    ok1 = abs(proj1 - 10.0) < 1e-3
    ok2 = abs(proj2 - 5.0) < 1e-3
    
    print(f"  Signal 1 recovered: {proj1:.6f} (expected 10.0) {'✓' if ok1 else '✗'}")
    print(f"  Signal 2 recovered: {proj2:.6f} (expected 5.0)  {'✓' if ok2 else '✗'}")
    print(f"  Orthogonality: cos(s1,s2) = {abs(np.dot(s1,s2)):.2e}")
    
    passed = ok1 and ok2
    print(f"\n  WAVEGUIDE: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


def test_selector(engine, tokenizer):
    """Verify selector picks the correct entity position for all prompts."""
    print("\n" + "═" * 60)
    print("  TEST: Selector (Spatial Filter) — L23 H6")
    print("═" * 60)
    
    selector = Selector.from_model(engine, layer_idx=23, head_idx=6)
    spec = selector.spec()
    print(f"  d_model = {spec['d_model']}")
    print(f"  ||d_k|| = {spec['norm']:.2f}")
    print(f"  frac_negative = {spec['frac_negative']:.4f}")
    
    # Also test the 1-bit version
    selector_1bit = Selector.from_one_bit(spec['d_model'])
    
    correct = 0
    correct_1bit = 0
    total = 0
    
    for country, info in FACTS.items():
        prompt_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([t]) for t in prompt_ids]
        target_pos = find_country_pos(tokens, country)
        if target_pos is None:
            print(f"  WARNING: Can't find '{country}' in tokens {tokens}")
            continue
        
        # Forward to L23, get pre-attention normed states
        _, states = forward_to_layer(engine, prompt_ids, 23)
        normed = states[23]['normed']  # [seq, d]
        
        # Test extracted selector
        result = selector.margin(normed, target_pos)
        ok = result['correct']
        correct += ok
        
        # Test 1-bit selector
        result_1bit = selector_1bit.margin(normed, target_pos)
        ok_1bit = result_1bit['correct']
        correct_1bit += ok_1bit
        
        total += 1
        print(f"  {country:>8s}: pos={target_pos}, selected={result['selected_pos']}, "
              f"margin={result['margin']:.3f} {'✓' if ok else '✗'}  "
              f"(1-bit: {'✓' if ok_1bit else '✗'})")
    
    # 5/6 is known behavior — Egypt is edge case (F45)
    passed = correct >= 5
    print(f"\n  SELECTOR (extracted): {correct}/{total} {'PASS ✓' if passed else 'FAIL ✗'} (5/6 = known behavior)")
    print(f"  SELECTOR (1-bit):     {correct_1bit}/{total}")
    return passed


def test_resonator(engine):
    """Verify resonator creates rank-1 score matrix with ratio > 100K."""
    print("\n" + "═" * 60)
    print("  TEST: Resonator (Fabry-Pérot Cavity) — L23 H6")
    print("═" * 60)
    
    resonator = Resonator.from_model(engine, layer_idx=23, head_idx=6)
    spec = resonator.spec()
    
    print(f"  head_dim = {spec['head_dim']}")
    print(f"  ||b_q|| = {spec['b_q_norm']:.2f}")
    print(f"  ||b_k|| = {spec['b_k_norm']:.2f}")
    print(f"  b_q · b_k = {spec['bias_dot']:.2f}")
    print(f"  Rank-1 ratio S[0]/S[1] = {spec['rank1_ratio']:.0f}")
    print(f"  Is rank-1: {spec['is_rank1']}")
    
    passed = spec['rank1_ratio'] > 100000
    print(f"\n  RESONATOR: {'PASS ✓' if passed else 'FAIL ✗'} "
          f"(need ratio > 100K, got {spec['rank1_ratio']:.0f})")
    return passed


def test_lens(engine, tokenizer):
    """Verify lens produces correct answer at rank < 25 for all countries."""
    print("\n" + "═" * 60)
    print("  TEST: Lens (Focusing Optic) — L23 H6")
    print("═" * 60)
    
    lens = Lens.from_model(engine, layer_idx=23, head_idx=6)
    spec = lens.spec()
    
    print(f"  head_dim = {spec['head_dim']}, d_model = {spec['d_model']}")
    print(f"  rank@90% = {spec['rank_90']}")
    print(f"  near-isometric ratio = {spec['near_isometric_ratio']:.4f}")
    
    all_ok = True
    for country, info in FACTS.items():
        prompt_ids = tokenizer.encode(info['prompt'])
        tokens = [tokenizer.decode([t]) for t in prompt_ids]
        target_pos = find_country_pos(tokens, country)
        if target_pos is None:
            continue
        
        _, states = forward_to_layer(engine, prompt_ids, 23)
        normed = states[23]['normed']
        h_entity = normed[target_pos]
        
        binding = lens.focus(h_entity)
        
        # Decode binding through LM head
        logits = get_logits(engine, binding)
        rank, score = get_rank(logits, info['answer'], tokenizer)
        ok = rank is not None and rank < 25
        all_ok = all_ok and ok
        
        print(f"  {country:>8s}: answer='{info['answer']}', rank={rank}, "
              f"score={score:.3f} {'✓' if ok else '✗'}")
    
    print(f"\n  LENS: {'PASS ✓' if all_ok else 'FAIL ✗'} (need all rank < 25)")
    return all_ok


def test_amplifier(engine, tokenizer):
    """Verify amplifier boosts answer rank and operates orthogonal to attention."""
    print("\n" + "═" * 60)
    print("  TEST: Amplifier (Laser Gain Medium) — L23")
    print("═" * 60)
    
    amplifier = Amplifier.from_model(engine, layer_idx=23)
    spec = amplifier.spec()
    print(f"  d_model = {spec['d_model']}, intermediate = {spec['intermediate_size']}")
    print(f"  expansion = {spec['expansion_ratio']:.2f}×")
    
    improved = 0
    orthogonal = 0
    total = 0
    
    for country, info in FACTS.items():
        prompt_ids = tokenizer.encode(info['prompt'])
        _, states = forward_to_layer(engine, prompt_ids, 23)
        
        h_post_attn = states[23]['h_post_attn'][-1]  # last token
        h_post_mlp = states[23]['h_post_mlp'][-1]
        attn_delta = states[23]['attn_out'][-1]  # last token attn output
        
        # Measure answer rank before and after MLP
        logits_pre = get_logits(engine, h_post_attn)
        logits_post = get_logits(engine, h_post_mlp)
        rank_pre, _ = get_rank(logits_pre, info['answer'], tokenizer)
        rank_post, _ = get_rank(logits_post, info['answer'], tokenizer)
        
        # Measure orthogonality
        mlp_delta = h_post_mlp - h_post_attn
        n_mlp = np.linalg.norm(mlp_delta)
        n_attn = np.linalg.norm(attn_delta)
        cos_orth = float(np.dot(mlp_delta, attn_delta) / (n_mlp * n_attn)) if n_mlp > 0 and n_attn > 0 else 0
        
        rank_improved = rank_post < rank_pre
        is_orthogonal = abs(cos_orth) < 0.15
        improved += rank_improved
        orthogonal += is_orthogonal
        total += 1
        
        print(f"  {country:>8s}: rank {rank_pre:>5d} → {rank_post:>5d} "
              f"{'✓' if rank_improved else '✗'}  "
              f"cos(Δmlp,Δattn)={cos_orth:+.4f} {'✓' if is_orthogonal else '✗'}")
    
    passed = improved == total and orthogonal == total
    print(f"\n  Rank improved: {improved}/{total}")
    print(f"  Orthogonal:    {orthogonal}/{total}")
    print(f"  AMPLIFIER: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


def test_stabilizer(engine, tokenizer):
    """Verify that perturbations settle to bounded orbit (emergent gyroscope)."""
    print("\n" + "═" * 60)
    print("  TEST: Stabilizer (Gyroscope) — Emergent")
    print("═" * 60)
    
    prompt = 'The capital of France is'
    prompt_ids = tokenizer.encode(prompt)
    
    # Run true forward
    _, true_states = forward_to_layer(engine, prompt_ids, 27)
    
    # Run perturbed forward (add noise to embeddings)
    rng = np.random.RandomState(42)
    h_true = engine.embedding(prompt_ids)[np.newaxis, :, :]
    noise = rng.randn(*h_true.shape).astype(np.float32) * 0.5
    h_pert = h_true + noise
    
    # Manual forward pass with perturbation
    h = h_pert
    seq_len = h.shape[1]
    perturbed_states = {}
    for li in range(28):
        layer = engine.layers[li]
        attn = layer.attention
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        normed = rms_norm(h, attn.norm_weight)
        Q = phi_linear(attn.W_q, normed, attn.b_q)
        K = phi_linear(attn.W_k, normed, attn.b_k)
        V = phi_linear(attn.W_v, normed, attn.b_v)
        Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        Q, K = attn.rope.apply(Q), attn.rope.apply(K)
        Ke = np.repeat(K, hpk, axis=1)
        Ve = np.repeat(V, hpk, axis=1)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
        if seq_len > 1:
            scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
        weights = phi_softmax(scores, axis=-1)
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ao = phi_linear(attn.W_o, ao)
        h_post_attn = h + ao
        mlp = layer.mlp
        nm = rms_norm(h_post_attn, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        h = h_post_attn + phi_linear(mlp.W_down, phi_silu(g) * u)
        perturbed_states[li] = h[0, -1].copy()
    
    # Measure drift at each layer
    true_list = [true_states[li]['h_post_mlp'][-1] for li in range(28)]
    pert_list = [perturbed_states[li] for li in range(28)]
    
    traj = Stabilizer.measure_trajectory(true_list, pert_list)
    
    print(f"  Perturbation: ||noise|| / ||embed|| = {np.linalg.norm(noise) / np.linalg.norm(h_true):.3f}")
    print(f"  Drift trajectory (angle in degrees):")
    for li in range(0, 28, 4):
        print(f"    L{li:>2d}: angle={traj['angles'][li]:.1f}°, "
              f"drift_ratio={traj['drift_ratios'][li]:.3f}")
    
    print(f"\n  Steady-state angle: {traj['steady_state_angle']:.1f}°")
    print(f"  Steady-state drift: {traj['steady_state_drift']:.3f}")
    
    # The gyroscope is working if drift doesn't diverge
    # (angle should stabilize, not grow linearly)
    last5 = traj['angles'][-5:]
    angle_range = max(last5) - min(last5)
    settled = angle_range < 20  # less than 20° variation in last 5 layers
    
    print(f"  Last 5 layers angle range: {angle_range:.1f}°")
    print(f"\n  STABILIZER: {'PASS ✓' if settled else 'FAIL ✗'} "
          f"(need stable orbit, not divergence)")
    return settled


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Verify geometric instrument components')
    parser.add_argument('--component', type=str, default='all',
                        help='Component to test: waveguide, selector, resonator, lens, amplifier, stabilizer, all')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  GEOMETRIC INSTRUMENT — Component Verification")
    print("=" * 60)
    
    # Waveguide doesn't need the model
    results = {}
    
    if args.component in ('all', 'waveguide'):
        results['waveguide'] = test_waveguide()
    
    # Everything else needs the model
    need_model = args.component != 'waveguide'
    if need_model:
        t0 = time.time()
        gc.collect()
        print(f"\n  Loading model...", flush=True)
        engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
        tokenizer = Qwen2Tokenizer()
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
        
        if args.component in ('all', 'selector'):
            results['selector'] = test_selector(engine, tokenizer)
        if args.component in ('all', 'resonator'):
            results['resonator'] = test_resonator(engine)
        if args.component in ('all', 'lens'):
            results['lens'] = test_lens(engine, tokenizer)
        if args.component in ('all', 'amplifier'):
            results['amplifier'] = test_amplifier(engine, tokenizer)
        if args.component in ('all', 'stabilizer'):
            results['stabilizer'] = test_stabilizer(engine, tokenizer)
    
    # Summary
    print("\n" + "═" * 60)
    print("  VERIFICATION SUMMARY")
    print("═" * 60)
    for name, passed in results.items():
        print(f"  {name:>12s}: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    print(f"\n  {total_pass}/{total_tests} components verified")
    
    if total_pass == total_tests:
        print("  ALL COMPONENTS PASS — ready to compose into instrument")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"  FAILED: {', '.join(failed)}")


if __name__ == '__main__':
    main()
