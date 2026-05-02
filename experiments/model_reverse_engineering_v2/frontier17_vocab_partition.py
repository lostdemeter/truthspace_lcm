#!/usr/bin/env python3
"""
Frontier 17: Vocabulary Partitioning — Active Vocabulary Selection
==================================================================

Four-phase experiment on whether we can partition the 152064-token vocabulary
by language/script and use only the relevant partition without losing
prediction quality.

Phase 1: Vocabulary Census
    Classify all 152064 tokens by Unicode script.
    Count tokens per partition.

Phase 2: Reduced-Vocabulary Prediction Test
    Build English-only (core + latin) embedding/lm_head subsets.
    Compare full-vocab vs reduced-vocab logits for English tokens.
    Measure: top-1 match, top-10 overlap, rank correlation.

Phase 3: Cross-Partition Leakage Test
    For each English token, find its nearest neighbors in full vocab.
    Check how many neighbors are from other partitions.
    Measure: cross-lingual geometric entanglement.

Phase 4: Memory & Speed Benchmarks
    Measure lm_head matmul time: full vocab vs reduced vocab.
    Calculate effective memory savings.

DC 291, F17
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


def decode_phi(path, dtype=np.float64):
    """Decode φ-encoded weight matrix."""
    d = np.load(path)
    signs = d['signs'].astype(dtype)
    exponents = d['exponents'].astype(dtype)
    return signs * (dtype(PHI) ** (exponents / dtype(GRID)))


def build_bpe_byte_decoder():
    """
    Build the GPT-2/Qwen2 byte-level BPE decoder.
    Maps visible characters back to their original byte values.
    """
    bs = list(range(ord('!'), ord('~')+1)) + \
         list(range(ord('¡'), ord('¬')+1)) + \
         list(range(ord('®'), ord('ÿ')+1))
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
    """Classify a single character into a script name using code-point ranges."""
    cp = ord(ch)

    # ASCII
    if cp < 0x0080:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            return 'Latin'
        return 'Common'

    # Latin Extended
    if 0x0080 <= cp <= 0x024F:
        return 'Latin'
    if 0x1E00 <= cp <= 0x1EFF:  # Latin Extended Additional
        return 'Latin'
    if 0x2C60 <= cp <= 0x2C7F:  # Latin Extended-C
        return 'Latin'
    if 0xA720 <= cp <= 0xA7FF:  # Latin Extended-D
        return 'Latin'

    # Greek
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
        return 'Greek'

    # Cyrillic
    if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
        return 'Cyrillic'

    # Armenian
    if 0x0530 <= cp <= 0x058F:
        return 'Armenian'

    # Hebrew
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return 'Hebrew'

    # Arabic
    if (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or
            0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF or
            0xFE70 <= cp <= 0xFEFF):
        return 'Arabic'

    # Devanagari
    if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
        return 'Devanagari'

    # Bengali
    if 0x0980 <= cp <= 0x09FF:
        return 'Bengali'

    # Gurmukhi
    if 0x0A00 <= cp <= 0x0A7F:
        return 'Gurmukhi'

    # Gujarati
    if 0x0A80 <= cp <= 0x0AFF:
        return 'Gujarati'

    # Tamil
    if 0x0B80 <= cp <= 0x0BFF:
        return 'Tamil'

    # Telugu
    if 0x0C00 <= cp <= 0x0C7F:
        return 'Telugu'

    # Kannada
    if 0x0C80 <= cp <= 0x0CFF:
        return 'Kannada'

    # Malayalam
    if 0x0D00 <= cp <= 0x0D7F:
        return 'Malayalam'

    # Thai
    if 0x0E00 <= cp <= 0x0E7F:
        return 'Thai'

    # Lao
    if 0x0E80 <= cp <= 0x0EFF:
        return 'Lao'

    # Tibetan
    if 0x0F00 <= cp <= 0x0FFF:
        return 'Tibetan'

    # Myanmar
    if 0x1000 <= cp <= 0x109F:
        return 'Myanmar'

    # Georgian
    if 0x10A0 <= cp <= 0x10FF or 0x2D00 <= cp <= 0x2D2F:
        return 'Georgian'

    # Hangul Jamo
    if 0x1100 <= cp <= 0x11FF:
        return 'Hangul'

    # Ethiopic
    if 0x1200 <= cp <= 0x137F:
        return 'Ethiopic'

    # Khmer
    if 0x1780 <= cp <= 0x17FF:
        return 'Khmer'

    # CJK Radicals, Kangxi, Ideographic Description
    if 0x2E80 <= cp <= 0x2FDF:
        return 'Han'

    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return 'Hiragana'

    # Katakana
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF65 <= cp <= 0xFF9F:
        return 'Katakana'

    # CJK Unified Ideographs (including extensions)
    if (0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or
            0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2FA1F):
        return 'Han'

    # Hangul Syllables
    if 0xAC00 <= cp <= 0xD7AF or 0x3130 <= cp <= 0x318F:
        return 'Hangul'

    # General punctuation, symbols, etc.
    cat = unicodedata.category(ch)
    if cat.startswith('L'):
        return 'Unknown'
    return 'Common'


def classify_token_script(token_str):
    """
    Classify a token into a script/partition based on its Unicode characters.
    First decodes the byte-level BPE encoding to get actual Unicode text,
    then classifies by dominant script.
    """
    # Decode byte-level BPE to actual Unicode
    decoded = decode_bpe_token(token_str)

    # Strip leading space (from BPE Ġ prefix) and whitespace
    clean = decoded.strip()

    if not clean or clean == '\ufffd':  # empty or replacement char
        return 'core'

    script_counts = {}
    for ch in clean:
        if ch == '\ufffd':  # skip replacement characters
            continue
        script = char_to_script(ch)
        script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return 'core'

    # Common/Inherited characters don't determine partition
    non_common = {s: c for s, c in script_counts.items()
                  if s not in ('Common', 'Inherited')}

    if not non_common:
        return 'core'

    dominant = max(non_common, key=non_common.get)
    return dominant


def script_to_partition(script_name):
    """Map Unicode script name to our partition scheme."""
    mapping = {
        'Latin': 'latin',
        'Common': 'core',
        'Inherited': 'core',
        'Han': 'cjk',
        'Hiragana': 'cjk',
        'Katakana': 'cjk',
        'Hangul': 'hangul',
        'Arabic': 'arabic',
        'Hebrew': 'arabic',
        'Cyrillic': 'cyrillic',
        'Devanagari': 'indic',
        'Bengali': 'indic',
        'Tamil': 'indic',
        'Telugu': 'indic',
        'Gujarati': 'indic',
        'Thai': 'indic',
        'Lao': 'indic',
        'Myanmar': 'indic',
        'Khmer': 'indic',
        'Georgian': 'other',
        'Armenian': 'other',
        'Ethiopic': 'other',
        'Tibetan': 'other',
        'Greek': 'other',
        'Unknown': 'other',
    }
    return mapping.get(script_name, 'other')


# ─── Phase 1: Vocabulary Census ─────────────────────────────────────

def phase1_vocab_census(id_to_token):
    """Classify all tokens by Unicode script and count per partition."""
    print()
    print("=" * 80)
    print("  Phase 1: Vocabulary Census")
    print("  Classifying all tokens by Unicode script")
    print("=" * 80)
    print()
    sys.stdout.flush()

    n_tokens = max(id_to_token.keys()) + 1
    token_partition = {}   # token_id -> partition name
    token_script = {}      # token_id -> raw script name
    partition_ids = {}      # partition -> list of token_ids

    # Special tokens (token_id not in vocab or empty string)
    for tid in range(n_tokens):
        tok = id_to_token.get(tid, '')
        if not tok or tok.startswith('<|') or tok.startswith('<unused'):
            token_partition[tid] = 'core'
            token_script[tid] = 'Special'
        else:
            script = classify_token_script(tok)
            partition = script_to_partition(script)
            token_partition[tid] = partition
            token_script[tid] = script

        if token_partition[tid] not in partition_ids:
            partition_ids[token_partition[tid]] = []
        partition_ids[token_partition[tid]].append(tid)

    # Print census
    print(f"  Total tokens: {n_tokens}")
    print()
    print(f"  {'Partition':<12s}  {'Count':>8s}  {'%':>7s}  {'Example tokens'}")
    print("  " + "-" * 70)

    for part in ['core', 'latin', 'cjk', 'hangul', 'arabic', 'cyrillic', 'indic', 'other']:
        ids = partition_ids.get(part, [])
        pct = len(ids) / n_tokens * 100
        # Show 5 example tokens (decoded from BPE bytes)
        examples = []
        sample_ids = ids[:20]
        for sid in sample_ids:
            tok_raw = id_to_token.get(sid, '???')
            tok_dec = decode_bpe_token(tok_raw)
            if len(tok_dec) > 15:
                tok_dec = tok_dec[:15] + '…'
            examples.append(repr(tok_dec))
            if len(examples) >= 5:
                break
        ex_str = ', '.join(examples) if examples else '(none)'
        print(f"  {part:<12s}  {len(ids):>8,d}  {pct:>6.1f}%  {ex_str}")
    print()

    # Script-level detail
    script_counts = {}
    for tid, sc in token_script.items():
        script_counts[sc] = script_counts.get(sc, 0) + 1

    print(f"  Detailed script breakdown (top 20):")
    print(f"  {'Script':<16s}  {'Count':>8s}  {'%':>7s}")
    print("  " + "-" * 35)
    for sc, cnt in sorted(script_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {sc:<16s}  {cnt:>8,d}  {cnt/n_tokens*100:>6.1f}%")
    print()

    # English partition = core + latin
    english_ids = sorted(set(partition_ids.get('core', []) + partition_ids.get('latin', [])))
    print(f"  English partition (core + latin): {len(english_ids):,d} tokens "
          f"({len(english_ids)/n_tokens*100:.1f}%)")
    print(f"  Reduction: {n_tokens:,d} → {len(english_ids):,d} "
          f"({n_tokens/len(english_ids):.2f}× smaller)")
    print()
    sys.stdout.flush()

    return token_partition, partition_ids, english_ids


# ─── Phase 2: Reduced-Vocabulary Prediction Test ────────────────────

def phase2_prediction_test(embeddings, lm_head, english_ids, id_to_token):
    """
    Compare full-vocab vs reduced-vocab predictions.
    For English tokens, does removing non-English tokens change lm_head output?
    """
    print()
    print("=" * 80)
    print("  Phase 2: Reduced-Vocabulary Prediction Test")
    print("  Does removing non-English tokens change English predictions?")
    print("=" * 80)
    print()
    sys.stdout.flush()

    n_total = embeddings.shape[0]
    n_english = len(english_ids)
    english_set = set(english_ids)
    english_idx = np.array(english_ids)

    # Build reduced lm_head: only English rows
    print(f"  Building reduced lm_head: {n_total} → {n_english} rows")
    lm_head_reduced = lm_head[english_idx]  # (n_english, 3584)
    print(f"  Full lm_head: {lm_head.shape} = {lm_head.nbytes/1e9:.2f} GB")
    print(f"  Reduced lm_head: {lm_head_reduced.shape} = {lm_head_reduced.nbytes/1e9:.2f} GB")
    print(f"  Memory saving: {(lm_head.nbytes - lm_head_reduced.nbytes)/1e9:.2f} GB")
    print()

    # Sample English tokens for testing
    # Pick 2000 english tokens spread across the vocabulary
    np.random.seed(42)
    test_english = np.random.choice(english_idx, size=min(2000, n_english), replace=False)

    # Create reverse mapping: full_id -> reduced_id
    full_to_reduced = {}
    for ridx, fid in enumerate(english_ids):
        full_to_reduced[fid] = ridx

    # Compute logits for test tokens
    print(f"  Testing {len(test_english)} English tokens...")
    print(f"  Computing full-vocab logits (n={n_total})...")
    sys.stdout.flush()

    # Process in batches to manage memory
    batch_size = 200
    top1_match = 0
    top10_overlap_sum = 0
    rank_preserved = 0
    cos_sum = 0.0
    total = 0

    # Track if full-vocab top-1 was English
    full_top1_english = 0
    full_top1_other = 0

    # Track probability mass on English tokens
    prob_mass_english_full = 0.0

    for batch_start in range(0, len(test_english), batch_size):
        batch_ids = test_english[batch_start:batch_start + batch_size]
        batch_embs = embeddings[batch_ids].astype(np.float32)

        # Full-vocab logits
        logits_full = batch_embs @ lm_head.astype(np.float32).T  # (batch, 152064)

        # Reduced-vocab logits
        logits_reduced = batch_embs @ lm_head_reduced.astype(np.float32).T  # (batch, n_english)

        for i in range(len(batch_ids)):
            lf = logits_full[i]
            lr = logits_reduced[i]

            # Full-vocab top-1 and top-10
            full_top1_id = np.argmax(lf)
            full_top10_ids = set(np.argsort(lf)[-10:])

            # Reduced-vocab top-1 and top-10 (map back to full IDs)
            red_top1_local = np.argmax(lr)
            red_top1_id = english_ids[red_top1_local]
            red_top10_local = np.argsort(lr)[-10:]
            red_top10_ids = set(english_ids[j] for j in red_top10_local)

            # Top-1 match: does reduced vocab predict same token?
            if red_top1_id == full_top1_id:
                top1_match += 1

            # Top-10 overlap
            overlap = len(full_top10_ids & red_top10_ids)
            top10_overlap_sum += overlap

            # Is full-vocab top-1 an English token?
            if full_top1_id in english_set:
                full_top1_english += 1
            else:
                full_top1_other += 1

            # Rank preservation: is full-vocab top-1 in reduced-vocab top-10?
            if full_top1_id in red_top10_ids:
                rank_preserved += 1

            # Cosine similarity of logit vectors (over English tokens only)
            # Extract full-vocab logits at English indices
            lf_english = lf[english_idx]
            cos = np.dot(lf_english, lr) / (np.linalg.norm(lf_english) * np.linalg.norm(lr) + 1e-10)
            cos_sum += cos

            # Probability mass on English tokens in full vocab
            # Use log-sum-exp for numerical stability
            lf_max = np.max(lf)
            exp_sum_full = np.sum(np.exp(lf - lf_max))
            exp_sum_eng = np.sum(np.exp(lf[english_idx] - lf_max))
            prob_mass_english_full += exp_sum_eng / exp_sum_full

            total += 1

        del logits_full, logits_reduced, batch_embs
        gc.collect()

        if (batch_start // batch_size) % 2 == 0:
            print(f"    Processed {min(batch_start + batch_size, len(test_english))}/{len(test_english)} tokens...")
            sys.stdout.flush()

    # Results
    print()
    print(f"  Results over {total} English test tokens:")
    print(f"  " + "-" * 55)
    print(f"  Top-1 match (full vs reduced):  {top1_match}/{total} = {top1_match/total*100:.1f}%")
    print(f"  Top-10 overlap (avg):           {top10_overlap_sum/total:.2f}/10")
    print(f"  Rank preserved (full top-1 in"
          f" reduced top-10): {rank_preserved}/{total} = {rank_preserved/total*100:.1f}%")
    print(f"  Logit cosine (English subset):  {cos_sum/total:.6f}")
    print()
    print(f"  Full-vocab top-1 is English:    {full_top1_english}/{total} = {full_top1_english/total*100:.1f}%")
    print(f"  Full-vocab top-1 is non-English: {full_top1_other}/{total} = {full_top1_other/total*100:.1f}%")
    print(f"  Avg softmax mass on English:    {prob_mass_english_full/total*100:.2f}%")
    print()
    sys.stdout.flush()

    del lm_head_reduced
    gc.collect()

    return {
        'top1_match': top1_match / total,
        'top10_overlap': top10_overlap_sum / total,
        'rank_preserved': rank_preserved / total,
        'cosine': cos_sum / total,
        'full_top1_english_pct': full_top1_english / total,
        'softmax_mass_english': prob_mass_english_full / total,
    }


# ─── Phase 3: Cross-Partition Leakage ───────────────────────────────

def phase3_leakage_test(embeddings, english_ids, token_partition, id_to_token):
    """
    For English tokens, check how many nearest neighbors are non-English.
    This measures geometric entanglement between partitions.
    """
    print()
    print("=" * 80)
    print("  Phase 3: Cross-Partition Leakage Test")
    print("  How geometrically entangled are language partitions?")
    print("=" * 80)
    print()
    sys.stdout.flush()

    english_set = set(english_ids)
    n_total = embeddings.shape[0]

    # Sample 500 English tokens
    np.random.seed(123)
    sample_ids = np.random.choice(english_ids, size=min(500, len(english_ids)), replace=False)

    # Precompute norms for cosine distance
    print("  Computing embedding norms...")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    norms = np.maximum(norms, 1e-10)
    sys.stdout.flush()

    k_neighbors = 20
    cross_partition_counts = []
    cross_partition_in_top5 = []
    cross_partition_in_top10 = []

    # Show some examples of cross-partition neighbors
    examples = []

    print(f"  Finding {k_neighbors} nearest neighbors for {len(sample_ids)} English tokens...")
    sys.stdout.flush()

    batch_size = 50
    for batch_start in range(0, len(sample_ids), batch_size):
        batch = sample_ids[batch_start:batch_start + batch_size]
        batch_embs = embeddings[batch].astype(np.float32)
        batch_norms = norms[batch]

        # Cosine similarity to all tokens
        # (batch, 3584) @ (3584, n_total) -> (batch, n_total)
        sims = (batch_embs @ embeddings.astype(np.float32).T) / (
            batch_norms[:, None] * norms[None, :])

        for i, tid in enumerate(batch):
            # Zero out self-similarity
            sims[i, tid] = -1

            # Top-k neighbors
            top_k_ids = np.argsort(sims[i])[-k_neighbors:][::-1]

            # Count non-English in top-k
            non_eng = sum(1 for nid in top_k_ids if nid not in english_set)
            cross_partition_counts.append(non_eng)

            non_eng_5 = sum(1 for nid in top_k_ids[:5] if nid not in english_set)
            cross_partition_in_top5.append(non_eng_5)

            non_eng_10 = sum(1 for nid in top_k_ids[:10] if nid not in english_set)
            cross_partition_in_top10.append(non_eng_10)

            # Collect examples of cross-partition leakage
            if non_eng > 0 and len(examples) < 15:
                tok_str = decode_bpe_token(id_to_token.get(tid, f'[{tid}]'))
                neighbors = []
                for nid in top_k_ids[:10]:
                    ntok = decode_bpe_token(id_to_token.get(int(nid), f'[{nid}]'))
                    npart = token_partition.get(int(nid), '?')
                    neighbors.append((ntok, npart, float(sims[i, nid])))
                examples.append((tok_str, neighbors))

        del sims, batch_embs
        gc.collect()

        if (batch_start // batch_size) % 2 == 0:
            print(f"    Processed {min(batch_start + batch_size, len(sample_ids))}/{len(sample_ids)}...")
            sys.stdout.flush()

    # Results
    cpc = np.array(cross_partition_counts)
    cp5 = np.array(cross_partition_in_top5)
    cp10 = np.array(cross_partition_in_top10)

    print()
    print(f"  Results over {len(sample_ids)} English tokens, {k_neighbors} neighbors each:")
    print(f"  " + "-" * 55)
    print(f"  Non-English in top-5:   mean={np.mean(cp5):.2f}, "
          f"median={np.median(cp5):.0f}, max={np.max(cp5)}")
    print(f"  Non-English in top-10:  mean={np.mean(cp10):.2f}, "
          f"median={np.median(cp10):.0f}, max={np.max(cp10)}")
    print(f"  Non-English in top-20:  mean={np.mean(cpc):.2f}, "
          f"median={np.median(cpc):.0f}, max={np.max(cpc)}")
    print()
    print(f"  Tokens with 0 non-English in top-20: "
          f"{np.sum(cpc == 0)}/{len(cpc)} = {np.sum(cpc == 0)/len(cpc)*100:.1f}%")
    print(f"  Tokens with ≥5 non-English in top-20: "
          f"{np.sum(cpc >= 5)}/{len(cpc)} = {np.sum(cpc >= 5)/len(cpc)*100:.1f}%")
    print()

    # Show examples
    if examples:
        print(f"  Examples of cross-partition leakage:")
        print()
        for tok_str, neighbors in examples[:10]:
            print(f"    Token: {repr(tok_str)}")
            for ntok, npart, sim in neighbors[:7]:
                marker = " ←" if npart not in ('core', 'latin') else ""
                print(f"      {sim:.4f}  {npart:<8s}  {repr(ntok)}{marker}")
            print()

    sys.stdout.flush()
    del norms
    gc.collect()

    return {
        'mean_non_eng_top5': float(np.mean(cp5)),
        'mean_non_eng_top10': float(np.mean(cp10)),
        'mean_non_eng_top20': float(np.mean(cpc)),
        'pct_zero_leakage': float(np.sum(cpc == 0) / len(cpc)),
    }


# ─── Phase 4: Memory & Speed Benchmarks ─────────────────────────────

def phase4_benchmarks(embeddings, lm_head, english_ids):
    """
    Measure actual speed and memory differences between
    full-vocab and reduced-vocab lm_head matmul.
    """
    print()
    print("=" * 80)
    print("  Phase 4: Memory & Speed Benchmarks")
    print("  Full vocab vs reduced vocab performance")
    print("=" * 80)
    print()
    sys.stdout.flush()

    n_total = embeddings.shape[0]
    n_dims = embeddings.shape[1]
    n_english = len(english_ids)
    english_idx = np.array(english_ids)

    # Build reduced matrices
    lm_head_reduced = lm_head[english_idx].astype(np.float32)
    lm_head_full_f32 = lm_head.astype(np.float32)
    emb_reduced = embeddings[english_idx].astype(np.float32)
    emb_full_f32 = embeddings.astype(np.float32)

    # Memory comparison
    full_emb_bytes = n_total * n_dims * 4  # float32
    full_lm_bytes = n_total * n_dims * 4
    red_emb_bytes = n_english * n_dims * 4
    red_lm_bytes = n_english * n_dims * 4

    print(f"  Memory comparison (float32):")
    print(f"    Full embed_tokens:    {full_emb_bytes/1e9:.3f} GB ({n_total:,d} × {n_dims})")
    print(f"    Reduced embed_tokens: {red_emb_bytes/1e9:.3f} GB ({n_english:,d} × {n_dims})")
    print(f"    Full lm_head:         {full_lm_bytes/1e9:.3f} GB ({n_total:,d} × {n_dims})")
    print(f"    Reduced lm_head:      {red_lm_bytes/1e9:.3f} GB ({n_english:,d} × {n_dims})")
    print()
    total_full = full_emb_bytes + full_lm_bytes
    total_red = red_emb_bytes + red_lm_bytes
    print(f"    Total (embed + lm_head):")
    print(f"      Full:    {total_full/1e9:.3f} GB")
    print(f"      Reduced: {total_red/1e9:.3f} GB")
    print(f"      Saving:  {(total_full - total_red)/1e9:.3f} GB "
          f"({(1 - total_red/total_full)*100:.1f}%)")
    print()

    # Speed benchmark: lm_head matmul
    # Simulate a batch of hidden states
    np.random.seed(42)
    batch_sizes = [1, 8, 32, 128]

    print(f"  Speed benchmark: hidden_state @ lm_head.T")
    print(f"  {'Batch':>6s}  {'Full (ms)':>10s}  {'Reduced (ms)':>12s}  {'Speedup':>8s}")
    print("  " + "-" * 42)

    for bs in batch_sizes:
        hidden = np.random.randn(bs, n_dims).astype(np.float32)

        # Warmup
        _ = hidden @ lm_head_full_f32.T
        _ = hidden @ lm_head_reduced.T

        # Time full
        n_iters = 5
        t0 = time.time()
        for _ in range(n_iters):
            _ = hidden @ lm_head_full_f32.T
        t_full = (time.time() - t0) / n_iters * 1000

        # Time reduced
        t0 = time.time()
        for _ in range(n_iters):
            _ = hidden @ lm_head_reduced.T
        t_red = (time.time() - t0) / n_iters * 1000

        speedup = t_full / t_red if t_red > 0 else 0
        print(f"  {bs:>6d}  {t_full:>10.1f}  {t_red:>12.1f}  {speedup:>7.2f}×")

    print()

    # Embedding lookup benchmark
    print(f"  Embedding lookup benchmark:")
    print(f"    Full vocab embedding table:    {emb_full_f32.shape}")
    print(f"    Reduced vocab embedding table:  {emb_reduced.shape}")

    # With remapping overhead
    test_ids = np.random.randint(0, n_english, size=1000)
    full_ids = english_idx[test_ids]

    n_iters = 100
    t0 = time.time()
    for _ in range(n_iters):
        _ = emb_full_f32[full_ids]
    t_full_lookup = (time.time() - t0) / n_iters * 1000

    t0 = time.time()
    for _ in range(n_iters):
        _ = emb_reduced[test_ids]
    t_red_lookup = (time.time() - t0) / n_iters * 1000

    print(f"    Full lookup (1000 tokens):    {t_full_lookup:.3f} ms")
    print(f"    Reduced lookup (1000 tokens): {t_red_lookup:.3f} ms")
    print()
    sys.stdout.flush()

    del lm_head_reduced, lm_head_full_f32, emb_reduced, emb_full_f32
    gc.collect()

    return {
        'full_size_gb': total_full / 1e9,
        'reduced_size_gb': total_red / 1e9,
        'saving_pct': (1 - total_red / total_full) * 100,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Frontier 17: Vocabulary Partitioning — Active Vocabulary Selection")
    print("  DC 291")
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

    # Phase 1: Vocabulary Census
    token_partition, partition_ids, english_ids = phase1_vocab_census(id_to_token)

    # Load embeddings
    print("  Loading embeddings...")
    t0 = time.time()
    embeddings = decode_phi(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    print(f"  Embeddings: {embeddings.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Load lm_head
    print("  Loading lm_head...")
    t0 = time.time()
    lm_head = decode_phi(os.path.join(MODEL_DIR, 'lm_head.npz'))
    print(f"  lm_head: {lm_head.shape} loaded in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Phase 2: Prediction Test
    pred_results = phase2_prediction_test(embeddings, lm_head, english_ids, id_to_token)

    # Phase 3: Leakage Test
    leak_results = phase3_leakage_test(embeddings, english_ids, token_partition, id_to_token)

    # Phase 4: Benchmarks
    bench_results = phase4_benchmarks(embeddings, lm_head, english_ids)

    # ─── Final Summary ───────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  SUMMARY — Frontier 17: Vocabulary Partitioning")
    print("=" * 80)
    print()
    print(f"  Vocabulary:")
    print(f"    Full:    {embeddings.shape[0]:,d} tokens")
    print(f"    English: {len(english_ids):,d} tokens "
          f"({len(english_ids)/embeddings.shape[0]*100:.1f}%)")
    print(f"    Reduction: {embeddings.shape[0]/len(english_ids):.2f}×")
    print()
    print(f"  Prediction Quality (full vs reduced vocab):")
    print(f"    Top-1 match:              {pred_results['top1_match']*100:.1f}%")
    print(f"    Top-10 overlap:           {pred_results['top10_overlap']:.2f}/10")
    print(f"    Logit cosine (English):   {pred_results['cosine']:.6f}")
    print(f"    Full top-1 is English:    {pred_results['full_top1_english_pct']*100:.1f}%")
    print(f"    Softmax mass on English:  {pred_results['softmax_mass_english']*100:.2f}%")
    print()
    print(f"  Cross-Partition Leakage:")
    print(f"    Non-English in top-5 neighbors:  {leak_results['mean_non_eng_top5']:.2f}")
    print(f"    Non-English in top-10 neighbors: {leak_results['mean_non_eng_top10']:.2f}")
    print(f"    Non-English in top-20 neighbors: {leak_results['mean_non_eng_top20']:.2f}")
    print(f"    Tokens with zero leakage:        {leak_results['pct_zero_leakage']*100:.1f}%")
    print()
    print(f"  Memory:")
    print(f"    Full (embed + lm_head):    {bench_results['full_size_gb']:.3f} GB")
    print(f"    Reduced:                   {bench_results['reduced_size_gb']:.3f} GB")
    print(f"    Saving:                    {bench_results['saving_pct']:.1f}%")
    print()

    # Verdict
    if pred_results['top1_match'] > 0.95:
        print("  ★ VERDICT: Vocabulary partitioning is VIABLE.")
        print("    Removing non-English tokens has negligible effect on English predictions.")
    elif pred_results['top1_match'] > 0.80:
        print("  ◆ VERDICT: Vocabulary partitioning is PROMISING but has measurable impact.")
        print(f"    {(1-pred_results['top1_match'])*100:.1f}% of English predictions change.")
    else:
        print("  ✗ VERDICT: Vocabulary partitioning causes SIGNIFICANT prediction changes.")
        print(f"    {(1-pred_results['top1_match'])*100:.1f}% of English predictions change.")
        print("    Cross-lingual entanglement may be too high.")
    print()


if __name__ == '__main__':
    main()
