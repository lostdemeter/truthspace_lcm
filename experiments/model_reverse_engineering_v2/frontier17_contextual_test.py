#!/usr/bin/env python3
"""
Frontier 17b: Contextual Vocabulary Partitioning Test
=====================================================

Tests whether vocabulary partitioning works with ACTUAL transformer processing,
not just raw embeddings.

Method:
  - Run 50 English tokens through all 28 φ-decoded transformer layers
    (single-token attention at position 0, where attn weight = 1.0)
  - At the final hidden state, compute logits with:
    a) Full lm_head (152064 tokens)
    b) Reduced lm_head (English-only, ~94550 tokens)
  - Compare: top-1 match, top-10 overlap, rank correlation

This tests the ACTUAL inference scenario (minus multi-token context).
The hypothesis: transformer layers contextualize hidden states to be
language-coherent, so reduced vocab should preserve English predictions.

Expected: >95% top-1 match (vs 49.4% for raw embeddings in F17 Phase 2).

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


def decode_phi(path, dtype=np.float64):
    """Decode φ-encoded weight matrix."""
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def rms_norm(x, weight):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight"""
    rms = np.sqrt(np.mean(x ** 2) + EPS)
    return (x / rms) * weight.astype(np.float64)


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def build_bpe_byte_decoder():
    """Build the GPT-2/Qwen2 byte-level BPE decoder."""
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
    """Decode a BPE token string to actual Unicode text."""
    try:
        byte_values = bytes([BPE_BYTE_DECODER[c] for c in tok_str])
        return byte_values.decode('utf-8', errors='replace')
    except (KeyError, UnicodeDecodeError):
        return tok_str


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
                    return id_to_token
    return None


def char_to_script(ch):
    """Classify a character into a script name using code-point ranges."""
    cp = ord(ch)
    if cp < 0x0080:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            return 'Latin'
        return 'Common'
    if 0x0080 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return 'Latin'
    if 0x0370 <= cp <= 0x03FF:
        return 'Greek'
    if 0x0400 <= cp <= 0x052F:
        return 'Cyrillic'
    if 0x0590 <= cp <= 0x05FF:
        return 'Hebrew'
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return 'Arabic'
    if 0x0900 <= cp <= 0x097F:
        return 'Devanagari'
    if 0x0E00 <= cp <= 0x0E7F:
        return 'Thai'
    if 0x3040 <= cp <= 0x309F:
        return 'Hiragana'
    if 0x30A0 <= cp <= 0x30FF:
        return 'Katakana'
    if 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return 'Han'
    if 0xAC00 <= cp <= 0xD7AF or 0x3130 <= cp <= 0x318F:
        return 'Hangul'
    cat = unicodedata.category(ch)
    if cat.startswith('L'):
        return 'Unknown'
    return 'Common'


def is_english_token(tok_str):
    """Check if a token belongs to the English (latin) partition."""
    decoded = decode_bpe_token(tok_str)
    clean = decoded.strip()
    if not clean or clean == '\ufffd':
        return True  # core tokens count as English
    for ch in clean:
        if ch == '\ufffd':
            continue
        script = char_to_script(ch)
        if script not in ('Latin', 'Common', 'Inherited'):
            return False
    return True


def get_english_partition(id_to_token):
    """Get list of token IDs in the English partition."""
    english_ids = []
    for tid in range(max(id_to_token.keys()) + 1):
        tok = id_to_token.get(tid, '')
        if not tok or tok.startswith('<|') or tok.startswith('<unused'):
            english_ids.append(tid)  # special tokens are core
        elif is_english_token(tok):
            english_ids.append(tid)
    return sorted(english_ids)


# ─── Forward Pass (single-token, position 0) ────────────────────────

def forward_pass_batch(token_ids, embeddings):
    """
    Run forward pass for multiple tokens through all 28 layers.
    Single-token at position 0: attention is trivial (weight = 1.0).
    Only V and O projections needed.

    Returns: dict of {token_id: final_hidden_state}
    """
    N = len(token_ids)
    dim = embeddings.shape[1]

    # Initialize hidden states
    xs = {tid: embeddings[tid].astype(np.float64).copy() for tid in token_ids}

    config = json.load(open(os.path.join(MODEL_DIR, 'config.json')))
    num_heads = config['num_attention_heads']      # 28
    num_kv_heads = config['num_key_value_heads']   # 4
    head_dim = config['head_dim']                  # 128
    heads_per_kv = num_heads // num_kv_heads       # 7

    for layer_idx in range(28):
        layer_dir = os.path.join(MODEL_DIR, f'layer_{layer_idx:02d}')
        t0 = time.time()

        # Load norms and biases
        norms = np.load(os.path.join(layer_dir, 'norms.npz'))
        input_ln = norms['input_layernorm'].astype(np.float64)
        post_attn_ln = norms['post_attention_layernorm'].astype(np.float64)

        biases = np.load(os.path.join(layer_dir, 'biases.npz'))
        v_bias = biases['v_proj_bias'].astype(np.float64)

        # ─── Attention (single-token: only V matters) ───
        v_proj = decode_phi(os.path.join(layer_dir, 'v_proj.npz'))
        vs = {}
        for tid in xs:
            x_normed = rms_norm(xs[tid], input_ln)
            v = v_proj @ x_normed + v_bias
            # GQA expansion
            attn_out = np.zeros(num_heads * head_dim)
            for h in range(num_heads):
                kv_idx = h // heads_per_kv
                attn_out[h * head_dim:(h + 1) * head_dim] = \
                    v[kv_idx * head_dim:(kv_idx + 1) * head_dim]
            vs[tid] = attn_out
        del v_proj; gc.collect()

        o_proj = decode_phi(os.path.join(layer_dir, 'o_proj.npz'))
        for tid in xs:
            attn_output = o_proj @ vs[tid]
            xs[tid] = xs[tid] + attn_output
        del o_proj, vs; gc.collect()

        # ─── MLP ───
        gate_proj = decode_phi(os.path.join(layer_dir, 'gate_proj.npz'))
        gates = {}
        for tid in xs:
            x_normed = rms_norm(xs[tid], post_attn_ln)
            gates[tid] = (gate_proj @ x_normed, x_normed)
        del gate_proj; gc.collect()

        up_proj = decode_phi(os.path.join(layer_dir, 'up_proj.npz'))
        intermediates = {}
        for tid in xs:
            gate_val, x_normed = gates[tid]
            up_val = up_proj @ x_normed
            intermediates[tid] = silu(gate_val) * up_val
        del up_proj, gates; gc.collect()

        down_proj = decode_phi(os.path.join(layer_dir, 'down_proj.npz'))
        for tid in xs:
            mlp_output = down_proj @ intermediates[tid]
            xs[tid] = xs[tid] + mlp_output
        del down_proj, intermediates; gc.collect()

        elapsed = time.time() - t0
        ref_tid = token_ids[0]
        print(f"    Layer {layer_idx:2d}: ||x||={np.linalg.norm(xs[ref_tid]):.2f}  "
              f"({elapsed:.1f}s) [{N} tokens]")
        sys.stdout.flush()

    # Final norm
    final_norm_w = np.load(os.path.join(MODEL_DIR, 'final_norm.npz'))['weight'].astype(np.float64)
    finals = {}
    for tid in xs:
        finals[tid] = rms_norm(xs[tid], final_norm_w)

    return finals


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Frontier 17b: Contextual Vocabulary Partitioning Test")
    print("  DC 291 §6 — Does transformer processing fix the 49.4%?")
    print("=" * 80)
    print()

    # Load tokenizer
    print("  Loading tokenizer...")
    id_to_token = load_tokenizer()
    if id_to_token is None:
        print("  ERROR: Could not load tokenizer")
        return
    print(f"  Tokenizer: {len(id_to_token)} tokens")
    sys.stdout.flush()

    # Get English partition
    print("  Building English partition...")
    english_ids = get_english_partition(id_to_token)
    n_english = len(english_ids)
    english_set = set(english_ids)
    english_idx = np.array(english_ids)
    print(f"  English partition: {n_english:,d} tokens ({n_english/len(id_to_token)*100:.1f}%)")
    sys.stdout.flush()

    # Select test tokens: 50 common English tokens
    # Pick recognizable English words spread across the vocabulary
    test_words = [
        'the', 'of', 'and', 'to', 'in', 'is', 'for', 'that', 'with', 'on',
        'are', 'be', 'was', 'not', 'but', 'have', 'from', 'this', 'they',
        'she', 'his', 'her', 'him', 'out', 'up', 'can', 'new', 'some',
        'time', 'very', 'when', 'come', 'could', 'make', 'like', 'just',
        'over', 'year', 'also', 'back', 'into', 'your', 'good', 'all',
        'world', 'still', 'here', 'than', 'first', 'been',
    ]

    # Find token IDs (try with and without BPE space prefix)
    test_token_ids = []
    for word in test_words:
        # Try raw BPE form
        for tid, tok in id_to_token.items():
            decoded = decode_bpe_token(tok)
            if decoded == word or decoded == ' ' + word:
                test_token_ids.append(tid)
                break
        if len(test_token_ids) >= 50:
            break

    # If we don't have enough, add some random English tokens
    if len(test_token_ids) < 50:
        np.random.seed(42)
        extra = np.random.choice(english_idx, size=100, replace=False)
        for eid in extra:
            if eid not in test_token_ids and eid in english_set:
                test_token_ids.append(int(eid))
            if len(test_token_ids) >= 50:
                break

    print(f"  Test tokens: {len(test_token_ids)}")
    for tid in test_token_ids[:10]:
        tok = id_to_token.get(tid, '???')
        decoded = decode_bpe_token(tok)
        print(f"    ID {tid:>6d}: {repr(decoded)}")
    if len(test_token_ids) > 10:
        print(f"    ... and {len(test_token_ids) - 10} more")
    print()
    sys.stdout.flush()

    # Load embeddings
    print("  Loading embeddings...")
    t0 = time.time()
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"  Embeddings: {embeddings.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # ─── Forward pass through 28 layers ──────────────────────────────
    print()
    print("  Running forward pass through 28 layers...")
    print("  (Single-token attention: weight = 1.0 at position 0)")
    print()
    sys.stdout.flush()

    t_start = time.time()
    finals = forward_pass_batch(test_token_ids, embeddings)
    t_forward = time.time() - t_start
    print(f"\n  Forward pass complete in {t_forward:.1f}s")
    sys.stdout.flush()

    # Free embeddings, load lm_head
    del embeddings
    gc.collect()

    print("  Loading lm_head...")
    t0 = time.time()
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'))
    print(f"  lm_head: {lm_head.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Build reduced lm_head
    lm_head_reduced = lm_head[english_idx]
    print(f"  Reduced lm_head: {lm_head_reduced.shape}")

    # Create reverse mapping
    full_to_reduced = {}
    for ridx, fid in enumerate(english_ids):
        full_to_reduced[fid] = ridx

    # ─── Compare predictions ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("  Comparing full-vocab vs reduced-vocab predictions")
    print("  on transformer-processed hidden states")
    print("=" * 80)
    print()
    sys.stdout.flush()

    top1_match = 0
    top10_overlap_sum = 0
    cos_sum = 0.0
    full_top1_english = 0
    rank_preserved = 0
    total = 0

    for tid in test_token_ids:
        hidden = finals[tid].astype(np.float32)

        # Full-vocab logits
        logits_full = hidden @ lm_head.astype(np.float32).T  # (152064,)

        # Reduced-vocab logits
        logits_reduced = hidden @ lm_head_reduced.astype(np.float32).T  # (n_english,)

        # Full-vocab top-1 and top-10
        full_top1_id = int(np.argmax(logits_full))
        full_top10_ids = set(int(x) for x in np.argsort(logits_full)[-10:])

        # Reduced-vocab top-1 and top-10
        red_top1_local = int(np.argmax(logits_reduced))
        red_top1_id = english_ids[red_top1_local]
        red_top10_local = np.argsort(logits_reduced)[-10:]
        red_top10_ids = set(english_ids[int(j)] for j in red_top10_local)

        # Top-1 match
        if red_top1_id == full_top1_id:
            top1_match += 1

        # Top-10 overlap
        overlap = len(full_top10_ids & red_top10_ids)
        top10_overlap_sum += overlap

        # Is full-vocab top-1 English?
        if full_top1_id in english_set:
            full_top1_english += 1

        # Rank preserved
        if full_top1_id in red_top10_ids:
            rank_preserved += 1

        # Cosine similarity of logit vectors (English subset)
        lf_english = logits_full[english_idx]
        cos = np.dot(lf_english, logits_reduced) / (
            np.linalg.norm(lf_english) * np.linalg.norm(logits_reduced) + 1e-10)
        cos_sum += cos

        total += 1

        # Show details for first 10
        if total <= 10:
            decoded_input = decode_bpe_token(id_to_token.get(tid, '???'))
            full_top1_tok = decode_bpe_token(id_to_token.get(full_top1_id, '???'))
            red_top1_tok = decode_bpe_token(id_to_token.get(red_top1_id, '???'))
            is_eng = "✓" if full_top1_id in english_set else "✗"
            match = "=" if red_top1_id == full_top1_id else "≠"
            print(f"  Input: {repr(decoded_input):20s}  "
                  f"Full→{repr(full_top1_tok):15s} [{is_eng}]  "
                  f"Red→{repr(red_top1_tok):15s} [{match}]  "
                  f"cos={cos:.4f}")

    print()
    print(f"  Results over {total} tokens (transformer-processed hidden states):")
    print(f"  " + "-" * 60)
    print(f"  Top-1 match (full vs reduced):  {top1_match}/{total} = {top1_match/total*100:.1f}%")
    print(f"  Top-10 overlap (avg):           {top10_overlap_sum/total:.2f}/10")
    print(f"  Logit cosine (English subset):  {cos_sum/total:.6f}")
    print(f"  Full top-1 is English:          {full_top1_english}/{total} = {full_top1_english/total*100:.1f}%")
    print(f"  Rank preserved (full top-1 in"
          f" reduced top-10): {rank_preserved}/{total} = {rank_preserved/total*100:.1f}%")
    print()

    # Compare with raw embedding baseline
    print(f"  Comparison with raw embedding test (F17 Phase 2):")
    print(f"    Raw embedding:   49.4% top-1 match, 5.17/10 overlap")
    print(f"    After 28 layers: {top1_match/total*100:.1f}% top-1 match, "
          f"{top10_overlap_sum/total:.2f}/10 overlap")
    print()

    # Verdict
    if top1_match / total > 0.95:
        print("  ★ VERDICT: Transformer processing FIXES the cross-lingual ambiguity.")
        print("    Vocabulary partitioning is VIABLE for inference.")
    elif top1_match / total > 0.80:
        print("  ◆ VERDICT: Transformer processing REDUCES cross-lingual ambiguity.")
        print(f"    {top1_match/total*100:.1f}% > 49.4% (raw), but not perfect.")
    else:
        print("  ✗ VERDICT: Transformer processing does NOT fix cross-lingual ambiguity.")
        print(f"    {top1_match/total*100:.1f}% is not much better than 49.4% (raw).")
    print()

    # Show top-1 prediction details
    print("  Full prediction table:")
    print(f"  {'Input':>20s}  {'Full top-1':>15s}  {'Reduced top-1':>15s}  {'Match':>5s}")
    print("  " + "-" * 62)
    for tid in test_token_ids:
        hidden = finals[tid].astype(np.float32)
        logits_full = hidden @ lm_head.astype(np.float32).T
        logits_reduced = hidden @ lm_head_reduced.astype(np.float32).T

        full_top1_id = int(np.argmax(logits_full))
        red_top1_id = english_ids[int(np.argmax(logits_reduced))]

        decoded_input = decode_bpe_token(id_to_token.get(tid, '???'))
        full_top1_tok = decode_bpe_token(id_to_token.get(full_top1_id, '???'))
        red_top1_tok = decode_bpe_token(id_to_token.get(red_top1_id, '???'))
        match = "✓" if red_top1_id == full_top1_id else "✗"

        print(f"  {repr(decoded_input):>20s}  {repr(full_top1_tok):>15s}  "
              f"{repr(red_top1_tok):>15s}  {match:>5s}")
    print()


if __name__ == '__main__':
    main()
