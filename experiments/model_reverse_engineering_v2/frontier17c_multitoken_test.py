#!/usr/bin/env python3
"""
Frontier 17c: Multi-Token Contextual Vocabulary Partitioning Test
=================================================================

The DEFINITIVE test: run English sentences through all 28 φ-decoded
transformer layers with FULL attention (Q/K/V + RoPE + causal mask).

For each position in the sequence, compare full-vocab vs reduced-vocab
lm_head predictions. This tests vocabulary partitioning under real
inference conditions with contextual attention.

F17b (single-token) showed 76% top-1 match, with all mismatches caused
by attractor tokens (新浪财经, 就够, ידוע). The hypothesis: multi-token
context will suppress these attractors, giving >95% top-1 match.

The deeper question: does the transformer operate in a language-agnostic
"meaning space" where language is just an I/O adapter?

    Input Language → Input Meaning → Output Meaning → Output Language
    (embed_tokens)    (28 layers)     (28 layers)     (lm_head)

DC 291 §6
"""

import numpy as np
import os
import sys
import json
import gc
import time
import unicodedata

PHI = (1 + np.sqrt(5)) / 2
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'phi_model')
GRID = 128
EPS = 1e-6

# Model config
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_DIM = 3584
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 7
ROPE_THETA = 1000000.0


# ─── Core functions ──────────────────────────────────────────────────

def decode_phi(path, dtype=np.float64):
    """Decode φ-encoded weight matrix."""
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def rms_norm(x, weight):
    """RMSNorm for 1D or 2D input."""
    if x.ndim == 1:
        rms = np.sqrt(np.mean(x ** 2) + EPS)
        return (x / rms) * weight.astype(np.float64)
    else:
        # (seq_len, dim) — norm per position
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + EPS)
        return (x / rms) * weight.astype(np.float64)[np.newaxis, :]


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


# ─── RoPE ────────────────────────────────────────────────────────────

def build_rope_cache(seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """Build RoPE cos/sin cache for given sequence length."""
    half = head_dim // 2
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = np.outer(positions, freqs)  # (seq_len, half)
    # Qwen2 convention: repeat to full head_dim
    cos = np.cos(np.concatenate([angles, angles], axis=-1))  # (seq_len, head_dim)
    sin = np.sin(np.concatenate([angles, angles], axis=-1))
    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply RoPE to x. x shape: (num_heads, seq_len, head_dim)
    cos, sin shape: (seq_len, head_dim) — broadcast over heads.
    Qwen2 convention: rotate_half = [-x[..., half:], x[..., :half]]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    # Broadcast cos/sin: (1, seq_len, head_dim)
    cos_b = cos[np.newaxis, :, :]
    sin_b = sin[np.newaxis, :, :]
    return x * cos_b + rotated * sin_b


# ─── BPE decoder ─────────────────────────────────────────────────────

def build_bpe_byte_decoder():
    bs = list(range(ord('!'), ord('~')+1)) + \
         list(range(ord('\u00a1'), ord('\u00ac')+1)) + \
         list(range(ord('\u00ae'), ord('\u00ff')+1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

BPE_BYTE_DECODER = build_bpe_byte_decoder()

def decode_bpe_token(tok_str):
    try:
        byte_values = bytes([BPE_BYTE_DECODER[c] for c in tok_str])
        return byte_values.decode('utf-8', errors='replace')
    except (KeyError, UnicodeDecodeError):
        return tok_str


# ─── Token classification ────────────────────────────────────────────

def char_to_script(ch):
    cp = ord(ch)
    if cp < 0x0080:
        cat = unicodedata.category(ch)
        return 'Latin' if cat.startswith('L') else 'Common'
    if 0x0080 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return 'Latin'
    if 0x0370 <= cp <= 0x03FF: return 'Greek'
    if 0x0400 <= cp <= 0x052F: return 'Cyrillic'
    if 0x0590 <= cp <= 0x05FF: return 'Hebrew'
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F: return 'Arabic'
    if 0x0900 <= cp <= 0x097F: return 'Devanagari'
    if 0x0E00 <= cp <= 0x0E7F: return 'Thai'
    if 0x3040 <= cp <= 0x309F: return 'Hiragana'
    if 0x30A0 <= cp <= 0x30FF: return 'Katakana'
    if 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return 'Han'
    if 0xAC00 <= cp <= 0xD7AF or 0x3130 <= cp <= 0x318F: return 'Hangul'
    cat = unicodedata.category(ch)
    return 'Unknown' if cat.startswith('L') else 'Common'


def is_english_token(tok_str):
    decoded = decode_bpe_token(tok_str)
    clean = decoded.strip()
    if not clean or clean == '\ufffd':
        return True
    for ch in clean:
        if ch == '\ufffd': continue
        script = char_to_script(ch)
        if script not in ('Latin', 'Common', 'Inherited'):
            return False
    return True


def load_tokenizer_vocab():
    """Load id_to_token mapping from tokenizer.json."""
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
                    return {idx: tok for tok, idx in vocab.items()}
    return None


def get_english_partition(id_to_token):
    english_ids = []
    for tid in range(max(id_to_token.keys()) + 1):
        tok = id_to_token.get(tid, '')
        if not tok or tok.startswith('<|') or tok.startswith('<unused'):
            english_ids.append(tid)
        elif is_english_token(tok):
            english_ids.append(tid)
    return sorted(english_ids)


# ─── Multi-token forward pass ────────────────────────────────────────

def forward_pass_sequence(token_ids, embeddings):
    """
    Full forward pass for a TOKEN SEQUENCE with real attention.

    Args:
        token_ids: list of token IDs (the sequence)
        embeddings: full embedding matrix (vocab_size, hidden_dim)

    Returns:
        final_hidden: (seq_len, hidden_dim) — normed final hidden states
    """
    seq_len = len(token_ids)

    # Initialize hidden state from embeddings
    x = np.stack([embeddings[tid] for tid in token_ids]).astype(np.float64)
    # x shape: (seq_len, hidden_dim)

    # Build RoPE cache
    rope_cos, rope_sin = build_rope_cache(seq_len)

    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        t0 = time.time()

        # Load norms and biases
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        input_ln = norms['input_layernorm'].astype(np.float64)
        post_attn_ln = norms['post_attention_layernorm'].astype(np.float64)

        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        q_bias = biases['q_proj_bias'].astype(np.float64)
        k_bias = biases['k_proj_bias'].astype(np.float64)
        v_bias = biases['v_proj_bias'].astype(np.float64)

        # ─── Attention ───────────────────────────────────────────

        # Pre-attention norm
        x_normed = rms_norm(x, input_ln)  # (seq_len, hidden_dim)

        # Q projection
        q_proj = decode_phi(os.path.join(layer_dir, 'q_proj.npz'))
        q = x_normed @ q_proj.T + q_bias  # (seq_len, num_heads * head_dim)
        del q_proj; gc.collect()

        # K projection
        k_proj = decode_phi(os.path.join(layer_dir, 'k_proj.npz'))
        k = x_normed @ k_proj.T + k_bias  # (seq_len, num_kv_heads * head_dim)
        del k_proj; gc.collect()

        # V projection
        v_proj = decode_phi(os.path.join(layer_dir, 'v_proj.npz'))
        v = x_normed @ v_proj.T + v_bias  # (seq_len, num_kv_heads * head_dim)
        del v_proj; gc.collect()

        # Reshape to multi-head: (num_heads, seq_len, head_dim)
        q = q.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = k.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 0, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # GQA: expand K, V
        k = np.repeat(k, HEADS_PER_KV, axis=0)  # (num_heads, seq_len, head_dim)
        v = np.repeat(v, HEADS_PER_KV, axis=0)

        # Attention scores: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # (num_heads, seq_len, seq_len)

        # Causal mask
        if seq_len > 1:
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + causal_mask[np.newaxis, :, :]

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply attention: (num_heads, seq_len, seq_len) @ (num_heads, seq_len, head_dim)
        attn_out = np.matmul(attn_weights, v)  # (num_heads, seq_len, head_dim)

        # Reshape back: (seq_len, num_heads * head_dim)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, NUM_HEADS * HEAD_DIM)

        # O projection
        o_proj = decode_phi(os.path.join(layer_dir, 'o_proj.npz'))
        attn_output = attn_out @ o_proj.T  # (seq_len, hidden_dim)
        del o_proj; gc.collect()

        # Residual
        x = x + attn_output

        # ─── MLP ─────────────────────────────────────────────────

        x_normed = rms_norm(x, post_attn_ln)

        gate_proj = decode_phi(os.path.join(layer_dir, 'gate_proj.npz'))
        gate = x_normed @ gate_proj.T
        del gate_proj; gc.collect()

        up_proj = decode_phi(os.path.join(layer_dir, 'up_proj.npz'))
        up = x_normed @ up_proj.T
        del up_proj; gc.collect()

        intermediate = silu(gate) * up

        down_proj = decode_phi(os.path.join(layer_dir, 'down_proj.npz'))
        mlp_output = intermediate @ down_proj.T
        del down_proj, intermediate; gc.collect()

        # Residual
        x = x + mlp_output

        elapsed = time.time() - t0
        print(f"    Layer {layer_idx:2d}: ||x||={np.linalg.norm(x[-1]):.2f}  "
              f"({elapsed:.1f}s) [seq_len={seq_len}]")
        sys.stdout.flush()

    # Final norm
    final_norm_w = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight'].astype(np.float64)
    final_hidden = rms_norm(x, final_norm_w)

    return final_hidden


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Frontier 17c: Multi-Token Contextual Vocabulary Partitioning Test")
    print("  DC 291 §6 — Full attention with Q/K/V + RoPE")
    print("=" * 80)
    print()

    # Tokenize test sentences
    print("  Loading tokenizer...")
    sys.stdout.flush()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

    test_sentences = [
        "The cat sat on the mat",
        "Once upon a time there was a",
        "The quick brown fox jumps over the",
        "She walked into the room and said",
        "In the beginning there was nothing but",
    ]

    all_sequences = []
    for sent in test_sentences:
        ids = tokenizer.encode(sent)
        tokens = [tokenizer.decode([i]) for i in ids]
        all_sequences.append((sent, ids, tokens))
        print(f"  Sentence: {repr(sent)}")
        print(f"    Tokens: {tokens}")
        print(f"    IDs:    {ids}")
    print()

    # Load vocabulary for classification
    print("  Loading vocab for English partition...")
    id_to_token = load_tokenizer_vocab()
    english_ids = get_english_partition(id_to_token)
    n_english = len(english_ids)
    english_set = set(english_ids)
    english_idx = np.array(english_ids)
    print(f"  English partition: {n_english:,d} tokens")
    sys.stdout.flush()

    # Load embeddings
    print("  Loading embeddings...")
    t0 = time.time()
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"  Embeddings: {embeddings.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Run forward pass for each sentence
    all_finals = {}
    for sent, ids, tokens in all_sequences:
        print(f"\n  ─── Forward pass: {repr(sent)} ───")
        sys.stdout.flush()
        t_start = time.time()
        final_hidden = forward_pass_sequence(ids, embeddings)
        elapsed = time.time() - t_start
        all_finals[sent] = (ids, tokens, final_hidden)
        print(f"  Done in {elapsed:.1f}s")
        sys.stdout.flush()

    # Free embeddings, load lm_head
    del embeddings
    gc.collect()

    print("\n  Loading lm_head...")
    t0 = time.time()
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'))
    print(f"  lm_head: {lm_head.shape} loaded in {time.time()-t0:.1f}s")
    lm_head_reduced = lm_head[english_idx]
    print(f"  Reduced lm_head: {lm_head_reduced.shape}")
    sys.stdout.flush()

    # ─── Compare predictions at each position ────────────────────
    print()
    print("=" * 80)
    print("  Comparing full-vocab vs reduced-vocab predictions")
    print("  at each position in multi-token sequences")
    print("=" * 80)
    print()

    total_positions = 0
    top1_match = 0
    top10_overlap_sum = 0
    full_top1_english = 0
    cos_sum = 0.0

    for sent, (ids, tokens, final_hidden) in all_finals.items():
        seq_len = len(ids)
        print(f"  Sentence: {repr(sent)}")
        print(f"  {'Pos':>4s}  {'Input':>15s}  {'Full top-1':>18s}  {'Reduced top-1':>18s}  {'Match':>5s}  {'Eng?':>4s}")
        print(f"  " + "-" * 72)

        for pos in range(seq_len):
            hidden = final_hidden[pos].astype(np.float32)

            # Full-vocab logits
            logits_full = hidden @ lm_head.astype(np.float32).T
            full_top1_id = int(np.argmax(logits_full))
            full_top10_ids = set(int(x) for x in np.argsort(logits_full)[-10:])

            # Reduced-vocab logits
            logits_reduced = hidden @ lm_head_reduced.astype(np.float32).T
            red_top1_local = int(np.argmax(logits_reduced))
            red_top1_id = english_ids[red_top1_local]
            red_top10_local = np.argsort(logits_reduced)[-10:]
            red_top10_ids = set(english_ids[int(j)] for j in red_top10_local)

            # Metrics
            match = red_top1_id == full_top1_id
            if match:
                top1_match += 1
            overlap = len(full_top10_ids & red_top10_ids)
            top10_overlap_sum += overlap
            is_eng = full_top1_id in english_set
            if is_eng:
                full_top1_english += 1

            # Cosine
            lf_eng = logits_full[english_idx]
            cos = np.dot(lf_eng, logits_reduced) / (
                np.linalg.norm(lf_eng) * np.linalg.norm(logits_reduced) + 1e-10)
            cos_sum += cos

            total_positions += 1

            # Display
            input_tok = tokens[pos] if pos < len(tokens) else '?'
            full_tok = decode_bpe_token(id_to_token.get(full_top1_id, '???'))
            red_tok = decode_bpe_token(id_to_token.get(red_top1_id, '???'))
            m_sym = "✓" if match else "✗"
            e_sym = "✓" if is_eng else "✗"
            print(f"  {pos:>4d}  {repr(input_tok):>15s}  {repr(full_tok):>18s}  "
                  f"{repr(red_tok):>18s}  {m_sym:>5s}  {e_sym:>4s}")

        print()

    # ─── Summary ─────────────────────────────────────────────────
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print(f"  Total positions tested: {total_positions}")
    print(f"  Top-1 match (full vs reduced):  {top1_match}/{total_positions} = "
          f"{top1_match/total_positions*100:.1f}%")
    print(f"  Top-10 overlap (avg):           {top10_overlap_sum/total_positions:.2f}/10")
    print(f"  Logit cosine (English subset):  {cos_sum/total_positions:.6f}")
    print(f"  Full top-1 is English:          {full_top1_english}/{total_positions} = "
          f"{full_top1_english/total_positions*100:.1f}%")
    print()

    # Comparison table
    print(f"  Progression across tests:")
    print(f"  {'Test':>30s}  {'Top-1 Match':>12s}  {'Top-10':>8s}  {'Cosine':>8s}")
    print(f"  " + "-" * 62)
    print(f"  {'Raw embedding (F17)':>30s}  {'49.4%':>12s}  {'5.17':>8s}  {'0.9995':>8s}")
    print(f"  {'Single-token/28 layers (F17b)':>30s}  {'76.0%':>12s}  {'7.98':>8s}  {'1.0000':>8s}")
    print(f"  {'Multi-token/28 layers (F17c)':>30s}  "
          f"{f'{top1_match/total_positions*100:.1f}%':>12s}  "
          f"{f'{top10_overlap_sum/total_positions:.2f}':>8s}  "
          f"{f'{cos_sum/total_positions:.4f}':>8s}")
    print()

    # Verdict
    pct = top1_match / total_positions * 100
    if pct > 95:
        print("  ★ VERDICT: Multi-token context RESOLVES cross-lingual ambiguity.")
        print("    Vocabulary partitioning is VIABLE for English inference.")
        print("    The transformer operates in language-agnostic meaning space.")
    elif pct > 80:
        print("  ◆ VERDICT: Multi-token context SIGNIFICANTLY helps.")
        print(f"    {pct:.1f}% match — much better than single-token (76%) and raw (49.4%).")
        print("    Context provides language-coherent signal, but attractor tokens persist.")
    else:
        print(f"  ✗ VERDICT: Multi-token context shows {pct:.1f}% match.")
        print("    Cross-lingual entanglement persists even with full context.")
    print()


if __name__ == '__main__':
    main()
