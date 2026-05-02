"""
Phase 4f: Synthetic BOS Pump & Parametric Template Generator
==============================================================

Two experiments:

1. SYNTHETIC BOS PUMP (F134 follow-up):
   L3's MLP explodes BOS along W_down's SV0 (cos=0.9955, scale=7135).
   Can we replace L3's MLP entirely with: h[0] += scale * sv0_direction?
   Also test replacing L26's drain with: h[0] += scale_26 * (-sv0_direction)?

2. PARAMETRIC TEMPLATE GENERATOR (F133 follow-up):
   Attention templates have structure: {BOS, middle, subject, last}.
   Fit T(N) from observed templates and test on held-out lengths.
"""

import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm, phi_softmax, phi_silu
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

FACTS = {
    'France':  {'prompt': 'The capital of France is',  'answer': ' Paris'},
    'Japan':   {'prompt': 'The capital of Japan is',   'answer': ' Tokyo'},
    'Germany': {'prompt': 'The capital of Germany is', 'answer': ' Berlin'},
    'Italy':   {'prompt': 'The capital of Italy is',   'answer': ' Rome'},
    'Spain':   {'prompt': 'The capital of Spain is',   'answer': ' Madrid'},
    'Egypt':   {'prompt': 'The capital of Egypt is',   'answer': ' Cairo'},
}

LENGTH_PROMPTS = {
    'France': [
        ('5tok', 'The capital of France is'),
        ('7tok', 'I know the capital of France is'),
        ('9tok', 'Can you tell me the capital of France is'),
        ('11tok', 'Please can you tell me what the capital of France is'),
    ],
    'Germany': [
        ('5tok', 'The capital of Germany is'),
        ('7tok', 'I know the capital of Germany is'),
        ('9tok', 'Can you tell me the capital of Germany is'),
        ('11tok', 'Please can you tell me what the capital of Germany is'),
    ],
}


def decode_weight(w):
    if isinstance(w, PhiEncoded):
        return w.decode()
    return w


def cos_sim(a, b):
    return float(np.dot(a.ravel(), b.ravel()) / 
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def get_top_prediction(engine, h):
    """Get top-1 token from hidden state."""
    h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
    logits = engine.lm_head(h_final)
    tid = int(np.argmax(logits[0, 0, :]))
    rank = int(np.sum(logits[0, 0, :] > logits[0, 0, tid]))
    return tid, rank


def run_with_synthetic_pump(engine, tokenizer, token_ids, pump_layers, 
                             sv0_dir, pump_scales):
    """Run model replacing specified layers' MLP at BOS with rank-1 injection."""
    h = engine.embedding(token_ids)[np.newaxis, :, :]
    n_layers = len(engine.layers)
    
    for li in range(n_layers):
        if li in pump_layers:
            # Run attention normally
            layer = engine.layers[li]
            attn = layer.attention
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            seq_len = h.shape[1]
            
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
            attn_out = phi_linear(attn.W_o, ao)
            h = h + attn_out
            
            # Run MLP normally for non-BOS positions
            mlp = layer.mlp
            nm = rms_norm(h, mlp.norm_weight)
            g = phi_linear(mlp.W_gate, nm)
            u = phi_linear(mlp.W_up, nm)
            mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
            
            # Replace BOS (position 0) with synthetic rank-1 injection
            scale = pump_scales[li]
            mlp_out[0, 0, :] = scale * sv0_dir
            
            h = h + mlp_out
        else:
            h = engine.layers[li](h)
    
    return h


def extract_last_token_attn(engine, token_ids, layer_indices):
    """Extract last-token attention weights at specified layers."""
    h = engine.embedding(token_ids)[np.newaxis, :, :]
    templates = {}
    
    for li in range(len(engine.layers)):
        layer = engine.layers[li]
        if li in layer_indices:
            attn = layer.attention
            nh, nkv = attn.num_heads, attn.num_kv_heads
            hpk, hd = nh // nkv, attn.head_dim
            seq_len = h.shape[1]
            
            normed = rms_norm(h, attn.norm_weight)
            Q = phi_linear(attn.W_q, normed, attn.b_q)
            K = phi_linear(attn.W_k, normed, attn.b_k)
            V = phi_linear(attn.W_v, normed, attn.b_v)
            Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
            K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
            Q, K = attn.rope.apply(Q), attn.rope.apply(K)
            Ke = np.repeat(K, hpk, axis=1)
            scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn.scale
            if seq_len > 1:
                scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
            weights = phi_softmax(scores, axis=-1)
            
            templates[li] = weights[0, :, -1, :]  # [nh, seq_len] last-token row
        
        h = layer(h)
    
    return templates


def run_with_fixed_template(engine, token_ids, templates):
    """Run model with fixed attention templates at all layers."""
    h = engine.embedding(token_ids)[np.newaxis, :, :]
    
    for li in range(len(engine.layers)):
        layer = engine.layers[li]
        attn = layer.attention
        nh, nkv = attn.num_heads, attn.num_kv_heads
        hpk, hd = nh // nkv, attn.head_dim
        seq_len = h.shape[1]
        
        normed = rms_norm(h, attn.norm_weight)
        V = phi_linear(attn.W_v, normed, attn.b_v)
        V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
        Ve = np.repeat(V, hpk, axis=1)
        
        # Use fixed template
        template = templates[li]  # [nh, seq_len]
        weights = np.zeros((1, nh, seq_len, seq_len), dtype=np.float32)
        weights[0, :, -1, :] = template
        # For non-last positions, use identity (attend to self)
        for p in range(seq_len - 1):
            weights[0, :, p, p] = 1.0
        
        ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
        ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        attn_out = phi_linear(attn.W_o, ao)
        h = h + attn_out
        
        mlp = layer.mlp
        nm = rms_norm(h, mlp.norm_weight)
        g = phi_linear(mlp.W_gate, nm)
        u = phi_linear(mlp.W_up, nm)
        mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
        h = h + mlp_out
    
    return h


def main():
    print("=" * 80)
    print("  Phase 4f: Synthetic BOS Pump & Parametric Template Generator")
    print("=" * 80)
    
    gc.collect()
    print("\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")
    
    n_layers = len(engine.layers)
    all_layers = list(range(n_layers))
    
    # ═══════════════════════════════════════════════════════════
    # PART 1: SYNTHETIC BOS PUMP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 1: SYNTHETIC BOS PUMP")
    print("=" * 80)
    
    # Step 1a: Extract L3's W_down SV0 direction and calibrate scale
    print("\n  Step 1a: Extract pump direction and calibrate scale")
    W_down_3 = decode_weight(engine.layers[3].mlp.W_down)
    U3, S3, Vt3 = np.linalg.svd(W_down_3, full_matrices=False)
    sv0_dir = U3[:, 0].copy()  # [d_model]
    print(f"    L3 W_down SV0: ||sv0|| = {float(np.linalg.norm(sv0_dir)):.4f}")
    print(f"    L3 W_down S[0] = {S3[0]:.4f}, S[0]/S[1] = {S3[0]/S3[1]:.2f}")
    
    # Calibrate: run France to get actual mlp_out at BOS
    france_tids = tokenizer.encode('The capital of France is')
    h_cal = engine.embedding(france_tids)[np.newaxis, :, :]
    for li in range(3):
        h_cal = engine.layers[li](h_cal)
    
    # Get L3's actual MLP output at BOS
    layer3 = engine.layers[3]
    attn3 = layer3.attention
    normed = rms_norm(h_cal, attn3.norm_weight)
    nh, nkv = attn3.num_heads, attn3.num_kv_heads
    hpk, hd = nh // nkv, attn3.head_dim
    seq_len = h_cal.shape[1]
    Q = phi_linear(attn3.W_q, normed, attn3.b_q)
    K = phi_linear(attn3.W_k, normed, attn3.b_k)
    V = phi_linear(attn3.W_v, normed, attn3.b_v)
    Q = Q.reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3)
    K = K.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    V = V.reshape(1, seq_len, nkv, hd).transpose(0, 2, 1, 3)
    Q, K = attn3.rope.apply(Q), attn3.rope.apply(K)
    Ke = np.repeat(K, hpk, axis=1)
    Ve = np.repeat(V, hpk, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, Ke) * attn3.scale
    scores += np.triu(np.full((seq_len, seq_len), -1e9, np.float32), k=1)
    weights = phi_softmax(scores, axis=-1)
    ao = np.einsum('bhqk,bhkd->bhqd', weights, Ve)
    ao = ao.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    attn_out = phi_linear(attn3.W_o, ao)
    h_post_attn = h_cal + attn_out
    
    mlp3 = layer3.mlp
    nm = rms_norm(h_post_attn, mlp3.norm_weight)
    g = phi_linear(mlp3.W_gate, nm)
    u = phi_linear(mlp3.W_up, nm)
    real_mlp_out_bos = phi_linear(mlp3.W_down, phi_silu(g) * u)[0, 0, :]
    
    # Scale = projection of real mlp_out onto sv0
    real_scale = float(np.dot(real_mlp_out_bos, sv0_dir))
    real_norm = float(np.linalg.norm(real_mlp_out_bos))
    recon_norm = abs(real_scale)
    
    print(f"    Real mlp_out[BOS] norm = {real_norm:.1f}")
    print(f"    Projection onto SV0 = {real_scale:.1f}")
    print(f"    Reconstruction error = {(real_norm - recon_norm)/real_norm*100:.2f}%")
    
    # Also extract L26's drain direction and scale
    print("\n  Extracting L26 drain direction...")
    W_down_26 = decode_weight(engine.layers[26].mlp.W_down)
    U26, S26, Vt26 = np.linalg.svd(W_down_26, full_matrices=False)
    sv0_dir_26 = U26[:, 0].copy()
    
    # Get L26's actual mlp_out at BOS
    h_26 = engine.embedding(france_tids)[np.newaxis, :, :]
    for li in range(26):
        h_26 = engine.layers[li](h_26)
    
    layer26 = engine.layers[26]
    attn26 = layer26.attention
    normed26 = rms_norm(h_26, attn26.norm_weight)
    Q26 = phi_linear(attn26.W_q, normed26, attn26.b_q)
    K26 = phi_linear(attn26.W_k, normed26, attn26.b_k)
    V26 = phi_linear(attn26.W_v, normed26, attn26.b_v)
    s26 = h_26.shape[1]
    Q26 = Q26.reshape(1, s26, nh, hd).transpose(0, 2, 1, 3)
    K26 = K26.reshape(1, s26, nkv, hd).transpose(0, 2, 1, 3)
    V26 = V26.reshape(1, s26, nkv, hd).transpose(0, 2, 1, 3)
    Q26, K26 = attn26.rope.apply(Q26), attn26.rope.apply(K26)
    Ke26 = np.repeat(K26, hpk, axis=1)
    Ve26 = np.repeat(V26, hpk, axis=1)
    sc26 = np.einsum('bhqd,bhkd->bhqk', Q26, Ke26) * attn26.scale
    sc26 += np.triu(np.full((s26, s26), -1e9, np.float32), k=1)
    w26 = phi_softmax(sc26, axis=-1)
    ao26 = np.einsum('bhqk,bhkd->bhqd', w26, Ve26)
    ao26 = ao26.transpose(0, 2, 1, 3).reshape(1, s26, -1)
    attn_out26 = phi_linear(attn26.W_o, ao26)
    h_pa26 = h_26 + attn_out26
    mlp26 = layer26.mlp
    nm26 = rms_norm(h_pa26, mlp26.norm_weight)
    g26 = phi_linear(mlp26.W_gate, nm26)
    u26 = phi_linear(mlp26.W_up, nm26)
    real_mlp26_bos = phi_linear(mlp26.W_down, phi_silu(g26) * u26)[0, 0, :]
    
    scale_26 = float(np.dot(real_mlp26_bos, sv0_dir_26))
    print(f"    L26 W_down S[0]/S[1] = {S26[0]/S26[1]:.2f}")
    print(f"    L26 mlp_out[BOS] projection onto L26_SV0 = {scale_26:.1f}")
    print(f"    cos(L3_SV0, L26_SV0) = {cos_sim(sv0_dir, sv0_dir_26):.4f}")
    
    # Step 1b: Test synthetic pump at L3 only
    print("\n" + "-" * 60)
    print("  Step 1b: Synthetic L3 pump (replace MLP at BOS only)")
    print("-" * 60)
    
    correct = 0
    total = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        answer_tid = tokenizer.encode(info['answer'])[0]
        
        h = run_with_synthetic_pump(
            engine, tokenizer, tids,
            pump_layers={3},
            sv0_dir=sv0_dir,
            pump_scales={3: real_scale}
        )
        
        h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
        logits = engine.lm_head(h_final)[0, 0, :]
        pred_tid = int(np.argmax(logits))
        rank = int(np.sum(logits > logits[answer_tid]))
        pred_tok = tokenizer.decode([pred_tid])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct += 1
        total += 1
        print(f"    {country}: '{pred_tok}' {ok}")
    
    print(f"\n  Synthetic L3 pump: {correct}/{total}")
    
    # Step 1c: Calibrate per-prompt (check if scale varies)
    print("\n" + "-" * 60)
    print("  Step 1c: Per-prompt scale calibration")
    print("-" * 60)
    
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h_p = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(3):
            h_p = engine.layers[li](h_p)
        
        dd = _decompose_l3(engine, h_p)
        mlp_bos = dd[0, 0, :]
        proj = float(np.dot(mlp_bos, sv0_dir))
        cos = cos_sim(mlp_bos, sv0_dir)
        print(f"    {country}: scale={proj:.1f}, cos(mlp,sv0)={cos:.4f}, "
              f"||mlp||={float(np.linalg.norm(mlp_bos)):.1f}")
    
    # Step 1d: Try with universal (average) scale
    print("\n" + "-" * 60)
    print("  Step 1d: Test with average scale across prompts")
    print("-" * 60)
    
    scales = []
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        h_p = engine.embedding(tids)[np.newaxis, :, :]
        for li in range(3):
            h_p = engine.layers[li](h_p)
        mlp_bos = _decompose_l3(engine, h_p)[0, 0, :]
        scales.append(float(np.dot(mlp_bos, sv0_dir)))
    
    avg_scale = np.mean(scales)
    print(f"    Scales: {[f'{s:.1f}' for s in scales]}")
    print(f"    Average scale: {avg_scale:.1f}")
    print(f"    Std: {np.std(scales):.1f}")
    
    correct = 0
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        answer_tid = tokenizer.encode(info['answer'])[0]
        
        h = run_with_synthetic_pump(
            engine, tokenizer, tids,
            pump_layers={3},
            sv0_dir=sv0_dir,
            pump_scales={3: avg_scale}
        )
        
        h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
        logits = engine.lm_head(h_final)[0, 0, :]
        rank = int(np.sum(logits > logits[answer_tid]))
        pred_tok = tokenizer.decode([int(np.argmax(logits))])
        ok = "✓" if rank == 0 else f"rank={rank}"
        if rank == 0:
            correct += 1
        print(f"    {country}: '{pred_tok}' {ok}")
    
    print(f"\n  Average-scale pump: {correct}/{len(FACTS)}")
    
    # ═══════════════════════════════════════════════════════════
    # PART 2: PARAMETRIC TEMPLATE GENERATOR
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 2: PARAMETRIC TEMPLATE GENERATOR")
    print("=" * 80)
    
    # Step 2a: Extract templates at multiple lengths
    print("\n  Step 2a: Extracting templates at multiple lengths...")
    
    template_data = {}  # {seq_len: templates_dict}
    
    for country in ['France', 'Germany']:
        for label, prompt in LENGTH_PROMPTS[country]:
            tids = tokenizer.encode(prompt)
            seq_len = len(tids)
            key = f"{country}_{label}_{seq_len}"
            
            templates = extract_last_token_attn(engine, tids, all_layers)
            template_data[key] = {
                'country': country,
                'label': label,
                'seq_len': seq_len,
                'templates': templates,
                'tids': tids,
            }
            print(f"    {key}: {seq_len} tokens")
    
    # Step 2b: Analyze template structure per length
    print("\n  Step 2b: Template structure at L23 per length")
    print(f"  {'Key':>30}  {'Len':>3}  {'BOS':>6}  {'mid_avg':>7}  {'subj':>6}  {'last':>6}")
    print("  " + "─" * 65)
    
    observed = []  # [(seq_len, bos_frac, mid_avg, subj_frac, last_frac)]
    
    for key, data in sorted(template_data.items(), key=lambda x: x[1]['seq_len']):
        t23 = data['templates'][23]  # [nh, seq_len]
        avg_w = t23.mean(axis=0)  # [seq_len] avg across heads
        seq_len = data['seq_len']
        
        bos = float(avg_w[0])
        last = float(avg_w[-1])
        subj = float(avg_w[-2])  # subject is second-to-last
        if seq_len > 2:
            mid = float(avg_w[1:-2].mean()) if seq_len > 3 else 0.0
        else:
            mid = 0.0
        
        print(f"  {key:>30}  {seq_len:>3}  {bos:>6.4f}  {mid:>7.4f}  {subj:>6.4f}  {last:>6.4f}")
        observed.append((seq_len, bos, mid, subj, last))
    
    # Step 2c: Fit parametric model
    print("\n  Step 2c: Fit parametric model T(N)")
    
    # Group by length (average France + Germany)
    len_groups = {}
    for sl, bos, mid, subj, last in observed:
        if sl not in len_groups:
            len_groups[sl] = []
        len_groups[sl].append((bos, mid, subj, last))
    
    avg_by_len = {}
    for sl, vals in sorted(len_groups.items()):
        arr = np.array(vals)
        avg_by_len[sl] = arr.mean(axis=0)  # [bos, mid, subj, last]
    
    print(f"\n  Averaged template parameters by length:")
    print(f"  {'Len':>5}  {'BOS':>6}  {'mid':>6}  {'subj':>6}  {'last':>6}")
    print("  " + "─" * 35)
    for sl, (bos, mid, subj, last) in sorted(avg_by_len.items()):
        print(f"  {sl:>5}  {bos:>6.4f}  {mid:>6.4f}  {subj:>6.4f}  {last:>6.4f}")
    
    # Fit: BOS(N) = a / (1 + b*N)
    # Try simple least squares
    lens = np.array(sorted(avg_by_len.keys()), dtype=float)
    bos_vals = np.array([avg_by_len[int(l)][0] for l in lens])
    subj_vals = np.array([avg_by_len[int(l)][2] for l in lens])
    last_vals = np.array([avg_by_len[int(l)][3] for l in lens])
    
    # BOS(N) ≈ a / (1 + b*N) → rewrite as 1/BOS = (1 + b*N)/a = 1/a + (b/a)*N
    inv_bos = 1.0 / bos_vals
    # Linear fit: inv_bos = c0 + c1 * N
    A = np.column_stack([np.ones_like(lens), lens])
    c0, c1 = np.linalg.lstsq(A, inv_bos, rcond=None)[0]
    a_bos = 1.0 / c0
    b_bos = c1 / c0
    
    print(f"\n  Fitted BOS(N) = {a_bos:.4f} / (1 + {b_bos:.4f} * N)")
    for sl in lens:
        fitted = a_bos / (1 + b_bos * sl)
        actual = avg_by_len[int(sl)][0]
        print(f"    N={int(sl):>2}: actual={actual:.4f}, fitted={fitted:.4f}, "
              f"err={abs(actual-fitted):.4f}")
    
    # Subject and last: try simple fits
    subj_mean = float(subj_vals.mean())
    print(f"\n  Subject: constant ≈ {subj_mean:.4f} (std={float(subj_vals.std()):.4f})")
    
    # Last(N) ≈ a_last / N
    last_fit = np.polyfit(1.0/lens, last_vals, 1)
    print(f"  Last(N) ≈ {last_fit[0]:.4f}/N + {last_fit[1]:.4f}")
    
    # Step 2d: Generate synthetic templates for test lengths
    print("\n" + "-" * 60)
    print("  Step 2d: Generate synthetic templates")
    print("-" * 60)
    
    def generate_template(seq_len, n_heads=28):
        """Generate a synthetic attention template for the last token."""
        N = float(seq_len)
        bos = a_bos / (1 + b_bos * N)
        subj = subj_mean
        last = last_fit[0] / N + last_fit[1]
        
        # Middle positions share remaining weight
        remaining = 1.0 - bos - subj - last
        n_mid = max(seq_len - 3, 0)  # exclude BOS, subject, last
        mid = remaining / n_mid if n_mid > 0 else 0.0
        
        # Build template: [n_heads, seq_len]
        template = np.zeros((n_heads, seq_len), dtype=np.float32)
        template[:, 0] = bos           # BOS
        if n_mid > 0:
            template[:, 1:-2] = mid    # middle positions
        template[:, -2] = subj         # subject (second to last)
        template[:, -1] = last         # last
        
        # Renormalize to sum to 1
        row_sums = template.sum(axis=1, keepdims=True)
        template = template / (row_sums + 1e-12)
        
        return template
    
    # Test: generate for known lengths and compare with real
    print(f"\n  Comparison: synthetic vs real template (L23, avg across heads):")
    for sl in sorted(avg_by_len.keys()):
        synth = generate_template(sl)
        avg_synth = synth.mean(axis=0)
        
        # Get real template from first matching entry
        for key, data in template_data.items():
            if data['seq_len'] == sl:
                real = data['templates'][23].mean(axis=0)
                cos = cos_sim(avg_synth, real)
                print(f"    N={sl}: cos(synth, real) = {cos:.4f}")
                break
    
    # Step 2e: Test synthetic templates on actual predictions
    print("\n" + "-" * 60)
    print("  Step 2e: Test synthetic templates for prediction")
    print("-" * 60)
    
    # Extract one real template set (France 5tok) for reference
    france_5_tids = tokenizer.encode('The capital of France is')
    real_templates_5 = extract_last_token_attn(engine, france_5_tids, all_layers)
    
    # Build synthetic templates for all layers at seq_len=5
    def build_all_layer_synthetic(seq_len, n_heads=28, n_layers=28):
        """Build synthetic templates for all layers."""
        templates = {}
        for li in range(n_layers):
            templates[li] = generate_template(seq_len, n_heads)
        return templates
    
    # Test real templates first (control)
    print(f"\n  Control: Real France templates → Germany predictions:")
    for country in ['Germany', 'Italy', 'Spain']:
        info = FACTS[country]
        tids = tokenizer.encode(info['prompt'])
        answer_tid = tokenizer.encode(info['answer'])[0]
        
        h = run_with_fixed_template(engine, tids, real_templates_5)
        h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
        logits = engine.lm_head(h_final)[0, 0, :]
        rank = int(np.sum(logits > logits[answer_tid]))
        pred_tok = tokenizer.decode([int(np.argmax(logits))])
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country}: '{pred_tok}' {ok}")
    
    # Test synthetic templates
    print(f"\n  Synthetic templates (all layers) → predictions:")
    for country, info in FACTS.items():
        tids = tokenizer.encode(info['prompt'])
        seq_len = len(tids)
        answer_tid = tokenizer.encode(info['answer'])[0]
        
        synth_templates = build_all_layer_synthetic(seq_len)
        
        h = run_with_fixed_template(engine, tids, synth_templates)
        h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
        logits = engine.lm_head(h_final)[0, 0, :]
        rank = int(np.sum(logits > logits[answer_tid]))
        pred_tok = tokenizer.decode([int(np.argmax(logits))])
        ok = "✓" if rank == 0 else f"rank={rank}"
        print(f"    {country} (N={seq_len}): '{pred_tok}' {ok}")
    
    # Step 2f: Test on different-length prompts
    print(f"\n  Synthetic templates → different-length prompts:")
    for country in ['France', 'Germany']:
        for label, prompt in LENGTH_PROMPTS[country]:
            tids = tokenizer.encode(prompt)
            seq_len = len(tids)
            
            # Baseline
            h_base = engine.embedding(tids)[np.newaxis, :, :]
            for li in range(n_layers):
                h_base = engine.layers[li](h_base)
            h_bf = rms_norm(h_base[:, -1:, :], engine.final_norm_weight)
            logits_base = engine.lm_head(h_bf)[0, 0, :]
            base_top = tokenizer.decode([int(np.argmax(logits_base))])
            
            # Synthetic
            synth_templates = build_all_layer_synthetic(seq_len)
            h = run_with_fixed_template(engine, tids, synth_templates)
            h_final = rms_norm(h[:, -1:, :], engine.final_norm_weight)
            logits = engine.lm_head(h_final)[0, 0, :]
            top = tokenizer.decode([int(np.argmax(logits))])
            
            # Check if they agree
            agree = "✓" if int(np.argmax(logits)) == int(np.argmax(logits_base)) else "✗"
            rank_base = int(np.sum(logits_base > logits_base[int(np.argmax(logits_base))]))
            rank_synth = int(np.sum(logits > logits[int(np.argmax(logits_base))]))
            
            print(f"    {country} {label} (N={seq_len}): "
                  f"base='{base_top}' synth='{top}' agree={agree} "
                  f"rank_of_base_in_synth={rank_synth}")
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    print(f"\n  BOS Pump Formula: scale * sv0_dir where:")
    print(f"    sv0_dir = first left singular vector of L3's W_down")
    print(f"    scale = {real_scale:.1f} (calibrated from France)")
    print(f"    avg_scale = {avg_scale:.1f} (averaged across 6 prompts)")
    
    print(f"\n  Template Generator: T(N) =")
    print(f"    BOS(N)  = {a_bos:.4f} / (1 + {b_bos:.4f} * N)")
    print(f"    subj    = {subj_mean:.4f} (constant)")
    print(f"    last(N) = {last_fit[0]:.4f}/N + {last_fit[1]:.4f}")
    print(f"    mid     = (1 - BOS - subj - last) / (N - 3)")
    
    print()


def _decompose_l3(engine, h):
    """Get L3 MLP output only."""
    layer = engine.layers[3]
    attn = layer.attention
    nh, nkv = attn.num_heads, attn.num_kv_heads
    hpk, hd = nh // nkv, attn.head_dim
    seq_len = h.shape[1]
    
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
    attn_out = phi_linear(attn.W_o, ao)
    h_pa = h + attn_out
    
    mlp = layer.mlp
    nm = rms_norm(h_pa, mlp.norm_weight)
    g = phi_linear(mlp.W_gate, nm)
    u = phi_linear(mlp.W_up, nm)
    mlp_out = phi_linear(mlp.W_down, phi_silu(g) * u)
    return mlp_out


if __name__ == '__main__':
    main()
