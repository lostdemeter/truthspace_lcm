"""
Benchmark: ShapeSpace vs Full Model
=====================================

Concrete measurements of:
  - Storage (bytes on disk, in-memory)
  - Memory footprint (RSS)
  - CPU time per query (latency)
  - Throughput (queries/second)
  - Operations per query
  - Scaling with dimensionality d
  - Scaling with number of entities N
"""

import sys, os, time, gc, json, resource
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shapespace import ShapeSpace
from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_matmul import phi_linear


MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'


def get_rss_mb():
    """Current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_model_file_sizes(model_dir):
    """Total size of model files on disk."""
    total = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def extract_vectors(engine, tokenizer, prompts):
    """Extract entity states, bindings, and answer directions."""
    hidden_dim = engine.hidden_dim
    identity = np.eye(hidden_dim, dtype=np.float32)
    chunk_size = 512
    attn = engine.layers[23].attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk = nh // nkv
    hd = attn.head_dim
    kv_group = 6 // hpk

    # Entity hidden states at L22
    entity_vecs = {}
    for name, (prompt, answer) in prompts.items():
        tids = tokenizer.encode(prompt)
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(23):
            h = engine.layers[li](h)
        entity_vecs[name] = h[0, 3, :].copy()

    # W_v, W_o for L23 Head 6
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

    # Bindings
    binding_vecs = {}
    for name in prompts:
        h_normed = rms_norm(entity_vecs[name][np.newaxis, np.newaxis, :],
                            attn.norm_weight)[0, 0]
        v_proj = W_v_g0 @ h_normed
        binding_vecs[name] = (W_o_h6 @ v_proj).copy()

    # Answer directions
    answer_vecs = {}
    for name, (prompt, answer) in prompts.items():
        aid = tokenizer.encode(answer)[-1]
        ans_dir = np.zeros(hidden_dim, dtype=np.float32)
        for start in range(0, hidden_dim, chunk_size):
            end = min(start + chunk_size, hidden_dim)
            chunk = identity[start:end][np.newaxis, :, :]
            logits = phi_linear(engine.lm_head.weight, chunk)[0]
            ans_dir[start:end] = logits[:, aid]
        answer_vecs[name] = ans_dir

    return entity_vecs, binding_vecs, answer_vecs


def main():
    print("=" * 80)
    print("  Benchmark: ShapeSpace vs Full Model")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════
    # 1. MODEL LOADING — measure memory
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  1. Model Loading & Memory")
    print("=" * 80)

    gc.collect()
    rss_before = get_rss_mb()
    print(f"  RSS before model load: {rss_before:.1f} MB")

    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    model_load_time = time.time() - t0

    rss_after_model = get_rss_mb()
    print(f"  RSS after model load:  {rss_after_model:.1f} MB")
    print(f"  Model memory footprint: {rss_after_model - rss_before:.1f} MB")
    print(f"  Model load time: {model_load_time:.1f}s")

    model_disk_bytes = get_model_file_sizes(MODEL_DIR)
    print(f"  Model on disk: {model_disk_bytes:,} bytes "
          f"({model_disk_bytes / 1024 / 1024:.1f} MB)")

    prompts = {
        'France': ('The capital of France is', ' Paris'),
        'Germany': ('The capital of Germany is', ' Berlin'),
        'Japan': ('The capital of Japan is', ' Tokyo'),
        'Italy': ('The capital of Italy is', ' Rome'),
    }

    # ═══════════════════════════════════════════════════════════
    # 2. EXTRACTION — build ShapeSpace
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  2. Vector Extraction & ShapeSpace Construction")
    print("=" * 80)

    t0 = time.time()
    entity_vecs, binding_vecs, answer_vecs = extract_vectors(
        engine, tokenizer, prompts)
    extract_time = time.time() - t0
    print(f"  Extraction time: {extract_time:.1f}s")

    t0 = time.time()
    space = ShapeSpace.from_vectors(entity_vecs, binding_vecs, answer_vecs)
    build_time = time.time() - t0
    print(f"  ShapeSpace build time: {build_time*1000:.2f} ms")
    print(f"  ShapeSpace: {space}")

    # ═══════════════════════════════════════════════════════════
    # 3. STORAGE COMPARISON
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  3. Storage Comparison")
    print("=" * 80)

    # ShapeSpace in-memory
    shape_bytes = space.storage_bytes
    # ShapeSpace serialized
    shape_json = json.dumps(space.to_dict())
    shape_json_bytes = len(shape_json.encode('utf-8'))
    # ShapeSpace binary (numpy)
    shape_npz_path = '/tmp/shapespace_bench.npz'
    data = space.to_dict()
    np.savez_compressed(shape_npz_path,
                        basis=np.array(data['basis']),
                        entities=np.array([data['entities'][k]
                                           for k in sorted(data['entities'])]),
                        bindings=np.array([data['bindings'][k]
                                           for k in sorted(data['bindings'])]),
                        answers=np.array([data['answers'][k]
                                          for k in sorted(data['answers'])]),
                        entity_mean=np.array(data['entity_mean'])
                            if data['entity_mean'] else np.array([]),
                        )
    shape_npz_bytes = os.path.getsize(shape_npz_path)

    print(f"\n  {'Metric':<30s} {'ShapeSpace':>15s} {'Full Model':>15s} {'Ratio':>10s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

    def ratio_str(a, b):
        if a > 0:
            return f"{b/a:,.0f}×"
        return "∞"

    rows = [
        ("In-memory (bytes)", shape_bytes, int(
            (rss_after_model - rss_before) * 1024 * 1024)),
        ("JSON serialized", shape_json_bytes, model_disk_bytes),
        ("Binary compressed", shape_npz_bytes, model_disk_bytes),
    ]
    for label, sv, mv in rows:
        print(f"  {label:<30s} {sv:>15,} {mv:>15,} {ratio_str(sv, mv):>10s}")

    # Per dimension
    print(f"\n  ShapeSpace by dimension:")
    for d in [3, 4, 5, 7]:
        proj = space.project(d) if d < space.dimensionality else space
        gt = {n: n for n in prompts}
        acc = proj.accuracy(gt)
        print(f"    d={d}: {proj.storage_bytes:>10,} bytes  "
              f"({proj.storage_bytes/1024:.1f} KB)  "
              f"acc={acc*100:.0f}%  ops={proj.ops_per_query}")

    # ═══════════════════════════════════════════════════════════
    # 4. LATENCY — per-query timing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  4. Query Latency")
    print("=" * 80)

    # ShapeSpace latency (warm cache)
    n_warmup = 1000
    for _ in range(n_warmup):
        space.query('France')

    n_iters = 100000
    latencies_shape = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        space.query('France')
        latencies_shape.append(time.perf_counter_ns() - t0)

    latencies_shape = np.array(latencies_shape, dtype=np.float64)
    shape_mean = np.mean(latencies_shape)
    shape_p50 = np.percentile(latencies_shape, 50)
    shape_p95 = np.percentile(latencies_shape, 95)
    shape_p99 = np.percentile(latencies_shape, 99)
    shape_min = np.min(latencies_shape)

    print(f"\n  ShapeSpace ({n_iters:,} queries):")
    print(f"    Mean:  {shape_mean/1000:.2f} μs")
    print(f"    P50:   {shape_p50/1000:.2f} μs")
    print(f"    P95:   {shape_p95/1000:.2f} μs")
    print(f"    P99:   {shape_p99/1000:.2f} μs")
    print(f"    Min:   {shape_min/1000:.2f} μs")

    # Full model latency
    tids_france = tokenizer.encode('The capital of France is')
    # Warmup
    h = engine.embedding(tids_france)[np.newaxis, :, :]
    for li in range(28):
        h = engine.layers[li](h)

    n_model = 5
    latencies_model = []
    for _ in range(n_model):
        t0 = time.perf_counter_ns()
        h = engine.embedding(tids_france)[np.newaxis, :, :]
        for li in range(28):
            h = engine.layers[li](h)
        latencies_model.append(time.perf_counter_ns() - t0)

    latencies_model = np.array(latencies_model, dtype=np.float64)
    model_mean = np.mean(latencies_model)

    print(f"\n  Full Model ({n_model} queries):")
    print(f"    Mean:  {model_mean/1e6:.1f} ms")
    print(f"    Min:   {np.min(latencies_model)/1e6:.1f} ms")
    print(f"    Max:   {np.max(latencies_model)/1e6:.1f} ms")

    speedup = model_mean / shape_mean
    print(f"\n  Speedup: {speedup:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    # 5. THROUGHPUT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  5. Throughput (queries/second)")
    print("=" * 80)

    # ShapeSpace throughput — batch all 4 entities
    n_batch = 100000
    entities = list(prompts.keys())
    t0 = time.perf_counter()
    for i in range(n_batch):
        space.query(entities[i % 4])
    shape_qps = n_batch / (time.perf_counter() - t0)

    # Full model throughput
    n_batch_model = 5
    model_prompts = [(tokenizer.encode(p), n)
                     for n, (p, a) in prompts.items()]
    t0 = time.perf_counter()
    for i in range(n_batch_model):
        tids, _ = model_prompts[i % 4]
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(28):
            h = engine.layers[li](h)
    model_qps = n_batch_model / (time.perf_counter() - t0)

    print(f"\n  ShapeSpace: {shape_qps:,.0f} queries/sec")
    print(f"  Full Model: {model_qps:.2f} queries/sec")
    print(f"  Throughput ratio: {shape_qps/model_qps:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    # 6. OPERATIONS COUNT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  6. Operation Count")
    print("=" * 80)

    hidden_dim = engine.hidden_dim
    nh = engine.layers[0].attention.num_heads
    hd = engine.layers[0].attention.head_dim
    n_layers = 28
    seq_len = 5  # "The capital of France is"

    # Full model ops (approximate)
    # Per layer: Q/K/V projections + attention scoring + V@attn + O projection + MLP
    qkv_ops = 3 * seq_len * hidden_dim * hidden_dim  # Q, K, V
    attn_score_ops = nh * seq_len * seq_len * hd      # Q@K^T
    attn_apply_ops = nh * seq_len * seq_len * hd      # attn@V
    o_proj_ops = seq_len * (nh * hd) * hidden_dim     # W_o
    # MLP: gate + up (both hidden→4*hidden) + down (4*hidden→hidden)
    mlp_intermediate = hidden_dim * 4  # approximate
    mlp_ops = seq_len * (2 * hidden_dim * mlp_intermediate +  # gate + up
                         mlp_intermediate * hidden_dim)        # down
    layer_ops = qkv_ops + attn_score_ops + attn_apply_ops + o_proj_ops + mlp_ops
    total_model_ops = n_layers * layer_ops
    # lm_head
    lm_head_ops = hidden_dim * 151936  # vocab size

    shape_ops = space.ops_per_query

    print(f"\n  ShapeSpace:")
    print(f"    d = {space.dimensionality}")
    print(f"    N_answers = {space.n_answers}")
    print(f"    Ops/query = {shape_ops}")
    print(f"    Breakdown:")
    d = space.dimensionality
    na = space.n_answers
    print(f"      Interference (add):  {d} ops")
    print(f"      Dot products:        {na} × {2*d-1} = {na*(2*d-1)} ops")
    print(f"      Argmax:              {na-1} ops")
    print(f"      Total:               {shape_ops} ops")

    print(f"\n  Full Model (approximate):")
    print(f"    Layers: {n_layers}")
    print(f"    Seq length: {seq_len}")
    print(f"    Hidden dim: {hidden_dim}")
    print(f"    Heads: {nh} × {hd}d")
    print(f"    Per-layer ops: ~{layer_ops:,}")
    print(f"    28-layer total: ~{total_model_ops:,}")
    print(f"    + lm_head:      ~{lm_head_ops:,}")
    print(f"    Grand total:    ~{total_model_ops + lm_head_ops:,}")

    print(f"\n  Reduction: {(total_model_ops + lm_head_ops) / shape_ops:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    # 7. SCALING WITH d
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  7. Scaling with Dimensionality d")
    print("=" * 80)

    gt = {n: n for n in prompts}
    print(f"\n  {'d':>3s}  {'Acc':>5s}  {'Ops':>5s}  {'Storage':>10s}  "
          f"{'Latency':>12s}  {'QPS':>12s}")
    print(f"  {'---':>3s}  {'-----':>5s}  {'-----':>5s}  {'----------':>10s}  "
          f"{'------------':>12s}  {'------------':>12s}")

    for d in range(1, space.dimensionality + 1):
        proj = space.project(d)
        acc = proj.accuracy(gt)

        # Timing
        for _ in range(500):
            proj.query('France')
        n_t = 50000
        t0 = time.perf_counter()
        for i in range(n_t):
            proj.query(entities[i % 4])
        elapsed = time.perf_counter() - t0
        lat = elapsed / n_t * 1e6  # μs
        qps = n_t / elapsed

        print(f"  {d:>3d}  {acc*100:>4.0f}%  {proj.ops_per_query:>5d}  "
              f"{proj.storage_bytes:>10,}B  {lat:>9.2f} μs  {qps:>10,.0f}/s")

    # ═══════════════════════════════════════════════════════════
    # 8. CPU USAGE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  8. CPU Usage (user + system time)")
    print("=" * 80)

    # ShapeSpace CPU time
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    n_cpu = 500000
    for i in range(n_cpu):
        space.query(entities[i % 4])
    r1 = resource.getrusage(resource.RUSAGE_SELF)

    shape_user = r1.ru_utime - r0.ru_utime
    shape_sys = r1.ru_stime - r0.ru_stime
    print(f"\n  ShapeSpace ({n_cpu:,} queries):")
    print(f"    User CPU:   {shape_user:.3f}s")
    print(f"    System CPU: {shape_sys:.3f}s")
    print(f"    Total CPU:  {shape_user + shape_sys:.3f}s")
    print(f"    CPU/query:  {(shape_user + shape_sys) / n_cpu * 1e6:.2f} μs")

    # Full model CPU time
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    n_cpu_m = 3
    for i in range(n_cpu_m):
        tids, _ = model_prompts[i % 4]
        h = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(28):
            h = engine.layers[li](h)
    r1 = resource.getrusage(resource.RUSAGE_SELF)

    model_user = r1.ru_utime - r0.ru_utime
    model_sys = r1.ru_stime - r0.ru_stime
    print(f"\n  Full Model ({n_cpu_m} queries):")
    print(f"    User CPU:   {model_user:.3f}s")
    print(f"    System CPU: {model_sys:.3f}s")
    print(f"    Total CPU:  {model_user + model_sys:.3f}s")
    print(f"    CPU/query:  {(model_user + model_sys) / n_cpu_m * 1000:.1f} ms")

    cpu_ratio = ((model_user + model_sys) / n_cpu_m) / \
                ((shape_user + shape_sys) / n_cpu)
    print(f"\n  CPU reduction: {cpu_ratio:,.0f}×")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    model_total_ops = total_model_ops + lm_head_ops
    model_mem_mb = rss_after_model - rss_before

    print(f"""
  ┌─────────────────────┬──────────────────┬──────────────────┬────────────┐
  │ Metric              │ ShapeSpace (d=7) │ Full Model       │ Ratio      │
  ├─────────────────────┼──────────────────┼──────────────────┼────────────┤
  │ Storage (memory)    │ {space.storage_bytes:>11,} B  │ {model_mem_mb:>11.0f} MB  │ {model_mem_mb*1024*1024/space.storage_bytes:>9,.0f}× │
  │ Storage (disk)      │ {shape_npz_bytes:>11,} B  │ {model_disk_bytes/1024/1024:>11.1f} MB  │ {model_disk_bytes/shape_npz_bytes:>9,.0f}× │
  │ Ops/query           │ {shape_ops:>14,}    │ {model_total_ops:>14,}    │ {model_total_ops//shape_ops:>9,}× │
  │ Latency (mean)      │ {shape_mean/1000:>11.2f} μs  │ {model_mean/1e6:>11.1f} ms  │ {speedup:>9,.0f}× │
  │ Throughput          │ {shape_qps:>10,.0f} q/s  │ {model_qps:>11.2f} q/s  │ {shape_qps/model_qps:>9,.0f}× │
  │ CPU/query           │ {(shape_user+shape_sys)/n_cpu*1e6:>11.2f} μs  │ {(model_user+model_sys)/n_cpu_m*1000:>11.1f} ms  │ {cpu_ratio:>9,.0f}× │
  │ Load time           │ {build_time*1000:>11.2f} ms  │ {model_load_time:>11.1f}  s  │ {model_load_time/(build_time):>9,.0f}× │
  └─────────────────────┴──────────────────┴──────────────────┴────────────┘

  At d=3 (minimum for 100%):
  ┌─────────────────────┬──────────────────┐
  │ Storage             │ {space.project(3).storage_bytes:>11,} B  │
  │ Ops/query           │ {space.project(3).ops_per_query:>14,}    │
  └─────────────────────┴──────────────────┘
""")


if __name__ == '__main__':
    main()
