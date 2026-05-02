#!/usr/bin/env python3
"""
Frontier 15: Three Experiments — Convergence, Composition, Geometry Head
=========================================================================

Experiment 1: Alternation Convergence Test
    Does the MLP gate contribution oscillate with decreasing amplitude
    across 28 layers? (Newtonian error correction hypothesis)

Experiment 2: Converged-State Composition
    Does dragon+shrimp→lobster improve from rank 17 (raw embeddings)
    to rank <5 in the converged hidden state (post-layer-27)?
    龙虾 (lóngxiā) = dragon+shrimp = lobster in Chinese.
    The shapes represent CONCEPTS that transcend language.

Experiment 3: Geometry Head Prototype
    How few SVD dimensions of the output space are needed to identify
    the correct answer token? (BBP spigot analogy)

All experiments use a manual forward pass through φ-encoded weights,
processing one layer at a time to manage memory.

DC 289 §6, F159
"""

import numpy as np
import os
import sys
import json
import time
import gc

PHI = (1 + np.sqrt(5)) / 2
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')
GRID = 128
EPS = 1e-6

# ─── Weight loading ───────────────────────────────────────────────────

def decode_phi(path):
    """Decode φ-encoded weight matrix to float64."""
    d = np.load(path)
    signs = d['signs'].astype(np.float64)
    exponents = d['exponents'].astype(np.float64)
    return signs * (PHI ** (exponents / GRID))


def rms_norm(x, weight):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight"""
    rms = np.sqrt(np.mean(x ** 2) + EPS)
    return (x / rms) * weight.astype(np.float64)


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def load_tokenizer():
    """Load tokenizer vocabulary."""
    for candidate in [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"),
    ]:
        if os.path.exists(candidate):
            snapshots = os.listdir(candidate)
            if snapshots:
                vocab_file = os.path.join(candidate, snapshots[0], "tokenizer.json")
                if os.path.exists(vocab_file):
                    with open(vocab_file, 'r') as f:
                        tokenizer_data = json.load(f)
                    vocab = tokenizer_data.get('model', {}).get('vocab', {})
                    id_to_token = {idx: tok for tok, idx in vocab.items()}
                    token_to_id = {}
                    for tok, idx in vocab.items():
                        token_to_id[tok] = idx
                        token_to_id[tok.lower()] = idx
                    return id_to_token, token_to_id
    return None, None


def find_token_id(word, token_to_id):
    """Find token ID trying various forms."""
    for c in [word, word.lower(), word.capitalize(), word.upper(),
              f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
              f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}"]:
        if c in token_to_id:
            return token_to_id[c], c
    return None, None


# ─── Batched forward pass ─────────────────────────────────────────────

def batched_forward_pass(token_ids, token_names, embeddings):
    """
    Run forward pass for MULTIPLE tokens through all 28 layers.
    
    Key optimization: load each layer's weights ONCE and process ALL
    tokens through it. This is 18x faster than separate passes.
    
    For single token at position 0:
    - Attention is trivial: softmax over 1 position = 1.0
    - Output = V repeated through GQA groups, then O_proj
    
    Returns:
        all_hidden: dict {name: list of 29 hidden states [embed, L0, ..., L27]}
        mlp_contribs: dict {name: list of 28 MLP outputs}
        attn_contribs: dict {name: list of 28 attention outputs}
        finals: dict {name: final normed hidden state}
    """
    N = len(token_ids)
    dim = embeddings.shape[1]
    
    # Initialize hidden states from embeddings
    xs = {}
    all_hidden = {}
    mlp_contribs = {}
    attn_contribs = {}
    
    for tid, name in zip(token_ids, token_names):
        xs[name] = embeddings[tid].astype(np.float64).copy()
        all_hidden[name] = [xs[name].copy()]
        mlp_contribs[name] = []
        attn_contribs[name] = []
    
    config = json.load(open(os.path.join(MODEL_DIR, 'config.json')))
    num_heads = config['num_attention_heads']      # 28
    num_kv_heads = config['num_key_value_heads']   # 4
    head_dim = config['head_dim']                  # 128
    heads_per_kv = num_heads // num_kv_heads       # 7
    
    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        t0 = time.time()
        
        # Load norms and biases (small — keep in memory)
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        input_ln = norms['input_layernorm'].astype(np.float64)
        post_attn_ln = norms['post_attention_layernorm'].astype(np.float64)
        
        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        q_bias = biases['q_proj_bias'].astype(np.float64)
        k_bias = biases['k_proj_bias'].astype(np.float64)
        v_bias = biases['v_proj_bias'].astype(np.float64)
        
        # ─── Attention: load each weight, process ALL tokens, free ───
        # v_proj first (only V matters for single-token attention)
        v_proj = decode_phi(os.path.join(layer_dir, 'v_proj.npz'))
        vs = {}
        for name in xs:
            x_normed = rms_norm(xs[name], input_ln)
            v = v_proj @ x_normed + v_bias
            # GQA expansion: repeat each KV head for its group of Q heads
            attn_out = np.zeros(num_heads * head_dim)
            for h in range(num_heads):
                kv_idx = h // heads_per_kv
                attn_out[h * head_dim:(h + 1) * head_dim] = \
                    v[kv_idx * head_dim:(kv_idx + 1) * head_dim]
            vs[name] = attn_out
        del v_proj; gc.collect()
        
        # We skip q_proj and k_proj — for single token, attention weight
        # is always 1.0 regardless of Q/K values. Only V matters.
        
        o_proj = decode_phi(os.path.join(layer_dir, 'o_proj.npz'))
        for name in xs:
            attn_output = o_proj @ vs[name]
            attn_contribs[name].append(attn_output.copy())
            xs[name] = xs[name] + attn_output
        del o_proj, vs; gc.collect()
        
        # ─── MLP: load each weight, process ALL tokens, free ─────────
        gate_proj = decode_phi(os.path.join(layer_dir, 'gate_proj.npz'))
        gates = {}
        for name in xs:
            x_normed = rms_norm(xs[name], post_attn_ln)
            gates[name] = (gate_proj @ x_normed, x_normed)
        del gate_proj; gc.collect()
        
        up_proj = decode_phi(os.path.join(layer_dir, 'up_proj.npz'))
        intermediates = {}
        for name in xs:
            gate_val, x_normed = gates[name]
            up_val = up_proj @ x_normed
            intermediates[name] = silu(gate_val) * up_val
        del up_proj, gates; gc.collect()
        
        down_proj = decode_phi(os.path.join(layer_dir, 'down_proj.npz'))
        for name in xs:
            mlp_output = down_proj @ intermediates[name]
            mlp_contribs[name].append(mlp_output.copy())
            xs[name] = xs[name] + mlp_output
            all_hidden[name].append(xs[name].copy())
        del down_proj, intermediates; gc.collect()
        
        elapsed = time.time() - t0
        # Print first token's stats as reference
        ref = token_names[0]
        print(f"    Layer {layer_idx:2d}: ||x||={np.linalg.norm(xs[ref]):.2f}  "
              f"||mlp||={np.linalg.norm(mlp_contribs[ref][-1]):.2f}  "
              f"||attn||={np.linalg.norm(attn_contribs[ref][-1]):.2f}  "
              f"({elapsed:.1f}s) [{N} tokens]")
    
    # Final norm
    final_norm_w = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight'].astype(np.float64)
    finals = {}
    for name in xs:
        finals[name] = rms_norm(xs[name], final_norm_w)
    
    return all_hidden, mlp_contribs, attn_contribs, finals


# ─── Experiment 1: Alternation Convergence ────────────────────────────

def experiment1_convergence(mlp_contributions, attn_contributions, hidden_states):
    """
    Test: Do MLP gate contributions oscillate with decreasing amplitude?
    If layers perform Newtonian error correction, we expect:
    - Alternating sign in the dominant direction
    - Decreasing amplitude: ||mlp_L+1|| < ||mlp_L|| (roughly)
    - Cumulative sum converges
    """
    print()
    print("=" * 80)
    print("  Experiment 1: Alternation Convergence Test")
    print("  Does the MLP contribution oscillate like Newtonian error correction?")
    print("=" * 80)
    print()
    
    mlp_norms = [np.linalg.norm(m) for m in mlp_contributions]
    attn_norms = [np.linalg.norm(a) for a in attn_contributions]
    
    # 1a. MLP contribution norms across layers
    print("  1a. MLP contribution norms by layer:")
    print(f"  {'Layer':>5s}  {'||MLP||':>10s}  {'||Attn||':>10s}  {'||hidden||':>10s}  MLP sign pattern")
    print("  " + "-" * 65)
    
    # Track the dominant direction of MLP contributions
    # Use the first MLP contribution as reference
    ref_dir = mlp_contributions[0] / (np.linalg.norm(mlp_contributions[0]) + 1e-20)
    
    mlp_projections = []
    for i in range(28):
        proj = np.dot(mlp_contributions[i], ref_dir)
        mlp_projections.append(proj)
        sign_str = "+" if proj > 0 else "−"
        bar = sign_str * min(int(abs(proj) / max(abs(p) for p in mlp_projections) * 20 + 0.5), 40) if i > 0 else sign_str * 20
        print(f"  {i:5d}  {mlp_norms[i]:10.2f}  {attn_norms[i]:10.2f}  "
              f"{np.linalg.norm(hidden_states[i+1]):10.2f}  {bar}")
    
    # 1b. Consecutive cosine similarities (anti-alternation check)
    print()
    print("  1b. Consecutive MLP cosine similarities:")
    cos_consecutive = []
    for i in range(27):
        cos = np.dot(mlp_contributions[i], mlp_contributions[i + 1]) / \
              (np.linalg.norm(mlp_contributions[i]) * np.linalg.norm(mlp_contributions[i + 1]) + 1e-20)
        cos_consecutive.append(cos)
        sign = "SAME" if cos > 0 else "FLIP"
        print(f"    L{i:02d}→L{i+1:02d}: cos={cos:+.4f}  [{sign}]")
    
    n_flips = sum(1 for c in cos_consecutive if c < 0)
    print(f"\n  Anti-alternation: {n_flips}/27 flips ({n_flips/27*100:.0f}%)")
    
    # 1c. Cumulative MLP contribution — does it converge?
    print()
    print("  1c. Cumulative MLP contribution (convergence test):")
    cumulative = np.zeros_like(mlp_contributions[0])
    prev_norm = 0
    for i in range(28):
        cumulative += mlp_contributions[i]
        cum_norm = np.linalg.norm(cumulative)
        delta = cum_norm - prev_norm
        direction = "↑" if delta > 0 else "↓"
        print(f"    After L{i:02d}: ||cumulative||={cum_norm:10.2f}  "
              f"Δ={delta:+10.2f} {direction}")
        prev_norm = cum_norm
    
    # 1d. Does amplitude decrease? (error correction bound)
    print()
    print("  1d. Amplitude evolution:")
    decreasing = 0
    for i in range(1, 28):
        if mlp_norms[i] < mlp_norms[i - 1]:
            decreasing += 1
    print(f"    Decreasing amplitude: {decreasing}/27 pairs ({decreasing/27*100:.0f}%)")
    print(f"    First layer ||MLP||: {mlp_norms[0]:.2f}")
    print(f"    Last layer ||MLP||:  {mlp_norms[27]:.2f}")
    print(f"    Ratio last/first:    {mlp_norms[27]/mlp_norms[0]:.4f}")
    
    # 1e. Alternating series test: project onto SVD of cumulative
    print()
    print("  1e. Projection onto dominant convergence direction:")
    U, S, Vt = np.linalg.svd(np.array(mlp_contributions), full_matrices=False)
    # Project each MLP contribution onto the top singular vector
    top_dir = Vt[0]
    projections = [np.dot(m, top_dir) for m in mlp_contributions]
    print(f"    Top SV explains {S[0]**2/np.sum(S**2)*100:.1f}% of variance")
    print(f"    Projections onto top SV direction:")
    for i, p in enumerate(projections):
        bar_len = int(abs(p) / max(abs(pp) for pp in projections) * 30)
        bar = ("+" if p > 0 else "−") * bar_len
        print(f"      L{i:02d}: {p:+12.2f}  {bar}")
    
    return {
        'mlp_norms': mlp_norms,
        'cos_consecutive': cos_consecutive,
        'projections': projections,
        'n_flips': n_flips,
    }


# ─── Experiment 2: Converged-State Composition ───────────────────────

def experiment2_converged_composition(hidden_states_dict, embeddings, id_to_token, token_to_id):
    """
    Test: Does dragon+shrimp→lobster improve in converged hidden states?
    
    hidden_states_dict: {word: [h0, h1, ..., h28]} for each concept word
    """
    print()
    print("=" * 80)
    print("  Experiment 2: Converged-State Composition")
    print("  Does 龙虾 (dragon+shrimp=lobster) improve after 28 layers?")
    print("=" * 80)
    print()
    
    compositions = [
        ("dragon", "shrimp", "lobster"),
        ("sun", "flower", "sunflower"),
        ("rain", "bow", "rainbow"),
        ("foot", "ball", "football"),
        ("star", "fish", "starfish"),
        ("sea", "horse", "seahorse"),
    ]
    
    check_layers = [0, 7, 14, 21, 27, 28]
    final_norm_w = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight'].astype(np.float64)
    
    # Pre-compute normalized embeddings for layer 0 comparisons
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / (emb_norms + 1e-20)
    
    # Load lm_head ONCE for all deep-layer comparisons
    print("  Loading lm_head (once for all compositions)...")
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'))  # (152064, 3584)
    print("  lm_head loaded.\n")
    
    for word_a, word_b, expected in compositions:
        if word_a not in hidden_states_dict or word_b not in hidden_states_dict:
            print(f"  SKIP: {word_a} + {word_b} (no hidden states)")
            continue
        
        id_exp, tok_exp = find_token_id(expected, token_to_id)
        if id_exp is None:
            print(f"  SKIP: {expected} not in vocab")
            continue
        
        id_a, _ = find_token_id(word_a, token_to_id)
        id_b, _ = find_token_id(word_b, token_to_id)
        
        print(f"  {word_a} + {word_b} → {expected}:")
        
        for li in check_layers:
            if li >= len(hidden_states_dict[word_a]):
                continue
            
            h_a = hidden_states_dict[word_a][li]
            h_b = hidden_states_dict[word_b][li]
            h_composed = h_a + h_b
            
            if li == 0:
                # Compare against raw embeddings via cosine similarity
                comp_norm = h_composed / (np.linalg.norm(h_composed) + 1e-20)
                sims = emb_normed @ comp_norm
                if id_a is not None: sims[id_a] = -999
                if id_b is not None: sims[id_b] = -999
                
                rank = int(np.sum(sims > sims[id_exp]))
                top5_idx = np.argsort(sims)[-5:][::-1]
                top5 = [id_to_token.get(idx, '?') for idx in top5_idx]
                print(f"    embed: rank={rank:5d}  top5={top5}")
            else:
                # Project through lm_head to get token logits
                if li == 28:
                    h_proj = rms_norm(h_composed, final_norm_w)
                else:
                    h_proj = h_composed
                
                logits = lm_head @ h_proj
                if id_a is not None: logits[id_a] = -1e9
                if id_b is not None: logits[id_b] = -1e9
                
                rank = int(np.sum(logits > logits[id_exp]))
                top5_idx = np.argsort(logits)[-5:][::-1]
                top5 = [id_to_token.get(idx, '?') for idx in top5_idx]
                label = f"L{li-1:02d}" if li < 28 else "final"
                print(f"    {label:>5s}: rank={rank:5d}  top5={top5}")
        
        print()
    
    del lm_head; gc.collect()


# ─── Experiment 3: Geometry Head ──────────────────────────────────────

def experiment3_geometry_head(x_final, correct_token_id, id_to_token, token_to_id):
    """
    Test: How few SVD dimensions of lm_head are needed to identify
    the correct output token?
    
    Approach: SVD of lm_head, project x_final into increasing numbers
    of dimensions, check when the correct token becomes rank 1.
    """
    print()
    print("=" * 80)
    print("  Experiment 3: Geometry Head Prototype")
    print("  How few dimensions to identify the correct token?")
    print("=" * 80)
    print()
    
    print("  Loading lm_head for SVD...")
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'))  # (152064, 3584)
    
    # Full logits first
    full_logits = lm_head @ x_final
    full_top5_idx = np.argsort(full_logits)[-5:][::-1]
    print(f"  Full lm_head top 5:")
    for i, idx in enumerate(full_top5_idx):
        tok = id_to_token.get(idx, f"tok_{idx}")
        print(f"    {i}: {tok!r:>25s}  logit={full_logits[idx]:.4f}")
    
    full_rank = int(np.sum(full_logits > full_logits[correct_token_id]))
    print(f"  Correct token '{id_to_token.get(correct_token_id, '?')}' rank: {full_rank}")
    print()
    
    # SVD of lm_head
    print("  Computing SVD of lm_head (this may take a while)...")
    # lm_head is (152064, 3584) — SVD would be huge
    # Instead: project x_final through progressively more dimensions
    # Use the SVD of lm_head^T (3584, 152064) which gives us
    # the 3584 singular vectors we need
    
    # More efficient: compute lm_head @ x_final in the SVD basis
    # lm_head = U @ S @ Vt, so logits = U @ S @ Vt @ x
    # With k dims: logits_k = U[:,:k] @ S[:k] @ Vt[:k,:] @ x
    # = U[:,:k] @ (S[:k] * (Vt[:k,:] @ x))
    
    # Since lm_head is (152064, 3584), we SVD the transpose for efficiency
    # Actually, for (152064, 3584), thin SVD gives (152064, 3584) @ (3584,) directly
    # Let's use a different approach: SVD of the 3584-dim space
    
    # Compute lm_head^T @ lm_head for the right singular vectors
    print("  Computing covariance matrix (3584 x 3584)...")
    cov = lm_head.T @ lm_head  # (3584, 3584) — this is manageable
    eigenvalues, V = np.linalg.eigh(cov)  # eigendecomposition
    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    V = V[:, idx_sort]  # (3584, 3584) — columns are right singular vectors
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    
    # x_final in the SVD basis
    x_in_basis = V.T @ x_final  # (3584,) — coefficients in SVD basis
    
    # For k dimensions, the approximate logits are:
    # logits_k = lm_head @ V[:,:k] @ V[:,:k].T @ x_final
    # = (lm_head @ V[:,:k]) @ (V[:,:k].T @ x_final)
    # = projected_lm[:,:k] @ x_in_basis[:k]
    
    print("  Projecting lm_head into SVD basis...")
    projected_lm = lm_head @ V  # (152064, 3584) — lm_head in SVD coordinates
    del lm_head, cov; gc.collect()
    
    # Test increasing dimensions
    test_dims = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 3584]
    
    print(f"\n  {'Dims':>6s}  {'Rank':>6s}  {'Top token':>25s}  {'Var explained':>15s}")
    print("  " + "-" * 60)
    
    total_var = np.sum(eigenvalues)
    first_rank1 = None
    
    for k in test_dims:
        if k > 3584:
            k = 3584
        
        # Approximate logits with k dimensions
        logits_k = projected_lm[:, :k] @ x_in_basis[:k]
        
        rank_k = int(np.sum(logits_k > logits_k[correct_token_id]))
        top_idx = np.argmax(logits_k)
        top_tok = id_to_token.get(top_idx, f"tok_{top_idx}")
        var_explained = np.sum(eigenvalues[:k]) / total_var * 100
        
        marker = " ★" if rank_k == 0 else ""
        print(f"  {k:6d}  {rank_k:6d}  {top_tok!r:>25s}  {var_explained:13.1f}%{marker}")
        
        if rank_k == 0 and first_rank1 is None:
            first_rank1 = k
    
    print()
    if first_rank1:
        print(f"  ★ Correct token becomes rank 1 at {first_rank1} dimensions")
        print(f"    (out of 3584 total = {first_rank1/3584*100:.1f}% of dimensions)")
        print(f"    (identifying 1 token out of 152064 = {152064/first_rank1:.0f}x compression)")
    else:
        print(f"  Correct token never reached rank 1 (full rank = {full_rank})")
    
    del projected_lm; gc.collect()
    
    return first_rank1


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 80)
    print("  Frontier 15: Three Experiments")
    print("  Convergence · Composition · Geometry Head")
    print("=" * 80)
    print()
    
    # Load tokenizer
    print("  Loading tokenizer...")
    id_to_token, token_to_id = load_tokenizer()
    if id_to_token is None:
        print("  ERROR: Could not load tokenizer")
        return
    
    # Load embeddings
    print("  Loading embeddings...")
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"  Embeddings: {embeddings.shape}")
    print()
    
    # ─── Batched forward pass for ALL concept words ─────────────────
    # Load each layer's weights ONCE and process all tokens through it.
    
    concept_words = ["dragon", "shrimp", "lobster", "sun", "flower", "sunflower",
                     "rain", "bow", "rainbow", "foot", "ball", "football",
                     "star", "fish", "starfish", "sea", "horse", "seahorse"]
    
    # Resolve token IDs
    token_ids = []
    token_names = []
    for word in concept_words:
        tid, ttok = find_token_id(word, token_to_id)
        if tid is None:
            print(f"  SKIP: {word} (not in vocab)")
            continue
        token_ids.append(tid)
        token_names.append(word)
        print(f"  Token: {word:>12s} → id={tid:6d} ({ttok})")
    
    print(f"\n  Running batched forward pass for {len(token_ids)} tokens...")
    print(f"  Loading each layer ONCE, processing all tokens through it.")
    print()
    
    all_hidden, mlp_contribs, attn_contribs, finals = batched_forward_pass(
        token_ids, token_names, embeddings
    )
    
    # ─── Experiment 1 (uses dragon's forward pass data) ───────────
    exp1_results = experiment1_convergence(
        mlp_contribs["dragon"], attn_contribs["dragon"], all_hidden["dragon"]
    )
    
    # ─── Experiment 2 ─────────────────────────────────────────────
    experiment2_converged_composition(
        all_hidden, embeddings, id_to_token, token_to_id
    )
    
    # ─── Experiment 3 ─────────────────────────────────────────────
    # Use "dragon" final normed state — check how many SVD dims of
    # lm_head are needed to identify the top predicted token
    dragon_id, _ = find_token_id("dragon", token_to_id)
    experiment3_geometry_head(finals["dragon"], dragon_id, id_to_token, token_to_id)
    
    # ─── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  SUMMARY — Frontier 15")
    print("=" * 80)
    print()
    print("  Exp 1 (Convergence): MLP anti-alternation flips =", exp1_results['n_flips'], "/ 27")
    print("  Exp 2 (Composition): Check rank progression above")
    print("  Exp 3 (Geometry Head): Check minimum dimensions above")
    print()
    print("  Chinese insight: 龙虾 (lóngxiā) = dragon+shrimp = lobster")
    print("  The shapes represent CONCEPTS that transcend language.")
    print()


if __name__ == '__main__':
    main()
