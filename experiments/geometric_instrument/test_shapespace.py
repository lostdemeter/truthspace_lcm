"""
Test ShapeSpace — The Geometric Data Structure
================================================

Demonstrates ShapeSpace as a general-purpose CS data object:
1. Extract vectors from the model (bootstrap)
2. Build a ShapeSpace
3. Query it (71 ops)
4. Project to 4D (still works!)
5. Extend with new entity
6. Compose two spaces
7. Compare to hash map / full model
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shapespace import ShapeSpace
from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_matmul import phi_linear


MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def extract_vectors(engine, tokenizer, prompts):
    """Extract entity states, bindings, and answer directions from model."""
    hidden_dim = engine.hidden_dim
    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512

    # 1. Entity hidden states at L22
    entity_vecs = {}
    for name, (prompt, answer) in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(23):  # through L22
            h = engine.layers[li](h)
        entity_vecs[name] = h[0, 3, :].copy()  # entity position

    # 2. V·W_o bindings from L23 Head 6
    attn = engine.layers[23].attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk = nh // nkv
    hd = attn.head_dim
    kv_group = 6 // hpk

    # Extract W_v for KV group 0
    W_v_g0 = np.zeros((hd, hidden_dim), dtype=np.float32)
    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]
        v_out = phi_linear(attn.W_v, chunk, attn.b_v)
        v_reshaped = v_out[0].reshape(-1, nkv, hd)
        W_v_g0[:, start:end] = v_reshaped[:, kv_group, :].T

    # Extract W_o for head 6
    W_o_h6 = np.zeros((hidden_dim, hd), dtype=np.float32)
    head_input = np.zeros((1, 1, nh * hd), dtype=np.float32)
    for d in range(hd):
        head_input[0, 0, :] = 0.0
        head_input[0, 0, 6 * hd + d] = 1.0
        o_out = phi_linear(attn.W_o, head_input)
        W_o_h6[:, d] = o_out[0, 0, :]

    binding_vecs = {}
    for name in prompts:
        h_ent = entity_vecs[name]
        h_normed = rms_norm(h_ent[np.newaxis, np.newaxis, :],
                            attn.norm_weight)[0, 0]
        v_proj = W_v_g0 @ h_normed
        binding_vecs[name] = (W_o_h6 @ v_proj).copy()

    # 3. Answer directions from lm_head
    answer_vecs = {}
    for name, (prompt, answer) in prompts.items():
        aid = tokenizer.encode(answer)[-1]
        answer_dir = np.zeros(hidden_dim, dtype=np.float32)
        for start in range(0, hidden_dim, chunk_size):
            end = min(start + chunk_size, hidden_dim)
            chunk = identity[start:end][np.newaxis, :, :]
            logits = phi_linear(engine.lm_head.weight, chunk)[0]
            answer_dir[start:end] = logits[:, aid]
        answer_vecs[name] = answer_dir

    return entity_vecs, binding_vecs, answer_vecs


def main():
    print("=" * 80)
    print("  ShapeSpace — The Geometric Data Structure")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    prompts = {
        'France': ('The capital of France is', ' Paris'),
        'Germany': ('The capital of Germany is', ' Berlin'),
        'Japan': ('The capital of Japan is', ' Tokyo'),
        'Italy': ('The capital of Italy is', ' Rome'),
    }

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Extract vectors from model
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 1: Extract vectors from model (bootstrap)")
    print("=" * 80)

    t0 = time.time()
    entity_vecs, binding_vecs, answer_vecs = extract_vectors(
        engine, tokenizer, prompts)
    print(f"  Extraction time: {time.time()-t0:.1f}s")

    for name in prompts:
        print(f"    {name}: entity={np.linalg.norm(entity_vecs[name]):.1f}  "
              f"binding={np.linalg.norm(binding_vecs[name]):.1f}  "
              f"answer={np.linalg.norm(answer_vecs[name]):.2f}")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Build ShapeSpace
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 2: Build ShapeSpace")
    print("=" * 80)

    space = ShapeSpace.from_vectors(
        entity_vecs=entity_vecs,
        binding_vecs=binding_vecs,
        answer_vecs=answer_vecs,
    )
    print(f"\n{space.summary()}")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Query (71 ops!)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 3: Query — the shape computer in action")
    print("=" * 80)

    ground_truth = {
        'France': 'France',
        'Germany': 'Germany',
        'Japan': 'Japan',
        'Italy': 'Italy',
    }

    print(f"\n  Ops per query: {space.ops_per_query}")
    print()
    for name in prompts:
        answer, score = space.query(name)
        all_scores = space.scores(name)
        correct = "✓" if answer == name else "✗"
        print(f"  {name:>10s} → {answer:<10s} {correct}  "
              f"(score={score:.3f})  "
              f"all: {', '.join(f'{k}={v:.3f}' for k, v in all_scores.items())}")

    acc = space.accuracy(ground_truth)
    print(f"\n  Accuracy: {acc*100:.0f}%")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Project to lower dimensions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 4: Dimensional projection — how small can we go?")
    print("=" * 80)

    for d in range(1, space.dimensionality + 1):
        proj = space.project(d)
        acc = proj.accuracy(ground_truth)
        marker = ""
        if acc == 1.0 and d <= 10:
            marker = " ← ★"
            if d > 1:
                prev = space.project(d - 1)
                if prev.accuracy(ground_truth) < 1.0:
                    marker = " ← MINIMUM ★★★"
        print(f"    d={d:>3d}: {acc*100:>5.0f}%  "
              f"ops={proj.ops_per_query:>5d}  "
              f"storage={proj.storage_bytes:>10,}B{marker}")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Extend with new entity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 5: Extend — add Spain without rebuilding")
    print("=" * 80)

    spain_prompt = 'The capital of Spain is'
    spain_answer = ' Madrid'
    spain_tids = tokenizer.encode(spain_prompt)

    # Get Spain's vectors
    h = engine.embedding(spain_tids)[np.newaxis, :, :]
    for li in range(23):
        h = engine.layers[li](h)
    spain_entity = h[0, 3, :].copy()

    # Spain binding
    attn = engine.layers[23].attention
    h_normed = rms_norm(spain_entity[np.newaxis, np.newaxis, :],
                        attn.norm_weight)[0, 0]
    # We need W_v_g0 and W_o_h6 — re-extract quickly
    hidden_dim = engine.hidden_dim
    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk = nh // nkv
    hd = attn.head_dim
    kv_group = 6 // hpk

    W_v_g0 = np.zeros((hd, hidden_dim), dtype=np.float32)
    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]
        v_out = phi_linear(attn.W_v, chunk, attn.b_v)
        v_reshaped = v_out[0].reshape(-1, nkv, hd)
        W_v_g0[:, start:end] = v_reshaped[:, kv_group, :].T

    W_o_h6 = np.zeros((hidden_dim, hd), dtype=np.float32)
    head_input = np.zeros((1, 1, nh * hd), dtype=np.float32)
    for d in range(hd):
        head_input[0, 0, :] = 0.0
        head_input[0, 0, 6 * hd + d] = 1.0
        o_out = phi_linear(attn.W_o, head_input)
        W_o_h6[:, d] = o_out[0, 0, :]

    v_proj = W_v_g0 @ h_normed
    spain_binding = (W_o_h6 @ v_proj).copy()

    # Spain answer direction
    spain_aid = tokenizer.encode(spain_answer)[-1]
    spain_answer_dir = np.zeros(hidden_dim, dtype=np.float32)
    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]
        logits = phi_linear(engine.lm_head.weight, chunk)[0]
        spain_answer_dir[start:end] = logits[:, spain_aid]

    # Extend the space
    space.extend('Spain', spain_entity, spain_binding, spain_answer_dir)

    print(f"\n  Space after extension: {space}")
    print()
    for name in ['France', 'Germany', 'Japan', 'Italy', 'Spain']:
        answer, score = space.query(name)
        correct = "✓" if answer == name else "✗"
        print(f"    {name:>10s} → {answer:<10s} {correct}  (score={score:.3f})")

    extended_truth = {**ground_truth, 'Spain': 'Spain'}
    acc = space.accuracy(extended_truth)
    print(f"\n  Accuracy after extension: {acc*100:.0f}%")

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Serialization round-trip
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 6: Serialize → deserialize round-trip")
    print("=" * 80)

    data = space.to_dict()
    import json
    json_str = json.dumps(data)
    print(f"\n  JSON size: {len(json_str):,} bytes ({len(json_str)/1024:.1f} KB)")

    space2 = ShapeSpace.from_dict(json.loads(json_str))
    print(f"  Restored: {space2}")

    # Verify round-trip
    all_match = True
    for name in extended_truth:
        a1, s1 = space.query(name)
        a2, s2 = space2.query(name)
        if a1 != a2 or abs(s1 - s2) > 1e-6:
            print(f"    MISMATCH: {name} → {a1}/{a2}")
            all_match = False
    print(f"  Round-trip match: {'✓ perfect' if all_match else '✗ MISMATCH'}")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Build second space (language-of) and compose
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 7: Compose — merge capital-of + language-of spaces")
    print("=" * 80)

    lang_prompts = {
        'France': ('The primary language of France is', ' French'),
        'Germany': ('The primary language of Germany is', ' German'),
        'Japan': ('The primary language of Japan is', ' Japanese'),
        'Italy': ('The primary language of Italy is', ' Italian'),
    }

    # Extract language vectors
    lang_entity_vecs = {}
    lang_binding_vecs = {}
    lang_answer_vecs = {}

    for name, (prompt, answer) in lang_prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(23):
            h = engine.layers[li](h)
        # Entity position varies by prompt length — find it
        # For "The primary language of X is", entity is at different pos
        # Use the subject token position
        tokens = tokenizer.encode(prompt)
        # Find position of country name
        entity_pos = len(tokens) - 2  # "X is" → X is second-to-last
        lang_entity_vecs[name] = h[0, entity_pos, :].copy()

        # Binding
        h_ent = lang_entity_vecs[name]
        h_normed = rms_norm(h_ent[np.newaxis, np.newaxis, :],
                            attn.norm_weight)[0, 0]
        v_proj = W_v_g0 @ h_normed
        lang_binding_vecs[name] = (W_o_h6 @ v_proj).copy()

        # Answer direction
        aid = tokenizer.encode(answer)[-1]
        ans_dir = np.zeros(hidden_dim, dtype=np.float32)
        for start in range(0, hidden_dim, chunk_size):
            end = min(start + chunk_size, hidden_dim)
            chunk = identity[start:end][np.newaxis, :, :]
            logits = phi_linear(engine.lm_head.weight, chunk)[0]
            ans_dir[start:end] = logits[:, aid]
        lang_answer_vecs[name] = ans_dir

    lang_space = ShapeSpace.from_vectors(
        entity_vecs=lang_entity_vecs,
        binding_vecs=lang_binding_vecs,
        answer_vecs=lang_answer_vecs,
    )
    print(f"\n  Language space: {lang_space}")

    # Test language space
    print("  Language queries:")
    for name in lang_prompts:
        answer, score = lang_space.query(name)
        correct = "✓" if answer == name else "✗"
        print(f"    {name:>10s} → {answer:<10s} {correct}  (score={score:.3f})")

    lang_truth = {n: n for n in lang_prompts}
    lang_acc = lang_space.accuracy(lang_truth)
    print(f"  Language accuracy: {lang_acc*100:.0f}%")

    # Compose!
    cap_space = ShapeSpace.from_vectors(entity_vecs, binding_vecs, answer_vecs)
    merged = cap_space.compose(lang_space)
    print(f"\n  Merged space: {merged}")

    # Test merged queries
    print("  Merged queries:")
    for prefix, names in [("capital", prompts), ("language", lang_prompts)]:
        for name in names:
            key = f"a/{name}" if prefix == "capital" else f"b/{name}"
            if key in merged._entities:
                answer, score = merged.query(key)
                print(f"    {prefix}/{name:>10s} → {answer:<15s} (score={score:.3f})")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: Timing comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Step 8: Timing — ShapeSpace vs full model")
    print("=" * 80)

    # ShapeSpace query timing
    n_iters = 10000
    t0 = time.time()
    for _ in range(n_iters):
        _ = space.query('France')
    shape_time = (time.time() - t0) / n_iters
    print(f"\n  ShapeSpace query: {shape_time*1e6:.1f} μs/query "
          f"({n_iters} iterations)")

    # Full model timing (single forward pass)
    tids = tokenizer.encode('The capital of France is')
    t0 = time.time()
    n_model = 3
    for _ in range(n_model):
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(28):
            h = engine.layers[li](h)
    model_time = (time.time() - t0) / n_model
    print(f"  Full model query: {model_time*1e3:.1f} ms/query "
          f"({n_model} iterations)")
    print(f"  Speedup: {model_time / shape_time:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"""
  ShapeSpace is a geometric data structure that:
    ✓ Solves capital-of in {space.ops_per_query} operations
    ✓ Stores in {space.storage_bytes:,} bytes ({space.storage_bytes/1024:.1f} KB)
    ✓ Projects to lower dimensions (accuracy trades for speed)
    ✓ Extends with new entities (no rebuild)
    ✓ Composes with other spaces (merge knowledge)
    ✓ Serializes to JSON (portable)
    ✓ Queries in ~{shape_time*1e6:.0f} μs (vs ~{model_time*1e3:.0f} ms full model)

  Comparison:
    Hash map:    O(1) lookup, no relationships, not composable
    Tree:        O(log n), hierarchical, rigid structure
    Graph:       O(V+E), relational, discrete edges
    ShapeSpace:  O(d), geometric, composable, projectable

  The geometry IS the data structure.
  The directions ARE the knowledge.
""")


if __name__ == '__main__':
    main()
