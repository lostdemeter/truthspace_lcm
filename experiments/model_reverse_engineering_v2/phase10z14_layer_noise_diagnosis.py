"""
Phase 10z14: Layer-by-Layer Noise Diagnosis
=============================================

The full model predicts German tokens (' heiß', ' Gründe', etc.) instead of
correct English answers like 'Paris'. This script diagnoses WHERE and WHY.

Approach:
1. Run inference layer by layer
2. After each layer, project to vocab via LM head → check top predictions
3. Track hidden state statistics (norm, max, min, distribution)
4. Identify the exact layer where noise first appears
5. Compare hidden state statistics to detect anomalies
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_matmul import phi_linear

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

PHI = (1 + np.sqrt(5)) / 2


def probe_vocab(hidden, engine, tokenizer, label, target_tokens=None):
    """Project hidden state through final norm + LM head, report top predictions."""
    normed = rms_norm(hidden, engine.final_norm_weight)
    logits = engine.lm_head(normed)
    last_logits = logits[0, -1, :]

    top5_idx = np.argsort(last_logits)[-5:][::-1]
    top5 = [(tokenizer.decode([idx]), float(last_logits[idx])) for idx in top5_idx]

    target_info = {}
    if target_tokens:
        for tok_str in target_tokens:
            tok_ids = tokenizer.encode(tok_str)
            if tok_ids:
                tid = tok_ids[0]
                rank = int((last_logits > last_logits[tid]).sum())
                target_info[tok_str] = {
                    'rank': rank,
                    'logit': float(last_logits[tid]),
                }

    return {
        'label': label,
        'top5': top5,
        'target_info': target_info,
        'logit_stats': {
            'max': float(last_logits.max()),
            'min': float(last_logits.min()),
            'mean': float(last_logits.mean()),
            'std': float(last_logits.std()),
        }
    }


def analyze_hidden(hidden, label):
    """Analyze hidden state statistics."""
    h = hidden[0, -1, :]  # last token
    stats = {
        'label': label,
        'norm': float(np.linalg.norm(h)),
        'max': float(np.max(h)),
        'min': float(np.min(h)),
        'mean': float(np.mean(h)),
        'std': float(np.std(h)),
        'abs_max': float(np.max(np.abs(h))),
        'num_nan': int(np.sum(np.isnan(h))),
        'num_inf': int(np.sum(np.isinf(h))),
        'num_zero': int(np.sum(h == 0)),
        'skew': float(np.mean(((h - np.mean(h)) / (np.std(h) + 1e-10))**3)),
        'kurtosis': float(np.mean(((h - np.mean(h)) / (np.std(h) + 1e-10))**4) - 3),
    }

    # Distribution: how many values in various magnitude ranges
    abs_h = np.abs(h)
    stats['pct_below_1'] = float(np.mean(abs_h < 1) * 100)
    stats['pct_below_10'] = float(np.mean(abs_h < 10) * 100)
    stats['pct_above_100'] = float(np.mean(abs_h > 100) * 100)
    stats['pct_above_1000'] = float(np.mean(abs_h > 1000) * 100)

    return stats


def main():
    t0 = time.time()

    print("=" * 72)
    print("  PHASE 10z14: LAYER-BY-LAYER NOISE DIAGNOSIS")
    print("=" * 72)

    # Load
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    prompts = [
        ("The capital of France is", "Paris"),
        ("The largest planet is", "Jupiter"),
    ]

    for prompt_text, expected in prompts:
        print(f"\n{'═' * 72}")
        print(f"  Prompt: '{prompt_text}' → expected '{expected}'")
        print(f"{'═' * 72}")

        token_ids = tokenizer.encode(prompt_text)
        print(f"  Tokens: {token_ids}")
        target_tokens = [' ' + expected, expected]

        # Embedding
        hidden = engine.embedding(token_ids)
        hidden = hidden[np.newaxis, :, :]

        h_stats = analyze_hidden(hidden, "Embed")
        v_probe = probe_vocab(hidden, engine, tokenizer, "Embed", target_tokens)
        print(f"\n  {'Layer':<8} {'|h|':>10} {'h_max':>10} {'h_std':>10} "
              f"{'logit_max':>10} {'logit_std':>10}  Top-1 prediction")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*30}")
        top1 = v_probe['top5'][0]
        print(f"  {'Embed':<8} {h_stats['norm']:10.1f} {h_stats['abs_max']:10.3f} "
              f"{h_stats['std']:10.3f} {v_probe['logit_stats']['max']:10.2f} "
              f"{v_probe['logit_stats']['std']:10.2f}  '{top1[0]}' ({top1[1]:.1f})")

        # Layer-by-layer
        prev_hidden = hidden.copy()
        anomaly_layer = None

        for i, layer in enumerate(engine.layers):
            hidden = layer(hidden, pure=False)

            h_stats = analyze_hidden(hidden, f"L{i}")
            v_probe = probe_vocab(hidden, engine, tokenizer, f"L{i}", target_tokens)
            top1 = v_probe['top5'][0]

            # Check for anomalies
            anomaly_flags = []
            if h_stats['num_nan'] > 0:
                anomaly_flags.append("NaN!")
            if h_stats['num_inf'] > 0:
                anomaly_flags.append("Inf!")
            if h_stats['abs_max'] > 1e4:
                anomaly_flags.append(f"HUGE({h_stats['abs_max']:.0e})")
            if h_stats['pct_above_1000'] > 1:
                anomaly_flags.append(f"{h_stats['pct_above_1000']:.1f}%>1K")

            # Cosine similarity with previous hidden
            h_curr = hidden[0, -1, :]
            h_prev = prev_hidden[0, -1, :]
            cos_sim = float(np.dot(h_curr, h_prev) /
                           (np.linalg.norm(h_curr) * np.linalg.norm(h_prev) + 1e-10))

            # Check if target token is in top-5
            target_in_top5 = any(expected.lower() in t[0].lower()
                                 for t in v_probe['top5'])

            flag_str = " ".join(anomaly_flags) if anomaly_flags else ""
            marker = "✓" if target_in_top5 else " "

            print(f"  {'L'+str(i):<8} {h_stats['norm']:10.1f} {h_stats['abs_max']:10.3f} "
                  f"{h_stats['std']:10.3f} {v_probe['logit_stats']['max']:10.2f} "
                  f"{v_probe['logit_stats']['std']:10.2f}  "
                  f"'{top1[0]}' ({top1[1]:.1f}) cos={cos_sim:.4f} {marker} {flag_str}")

            if anomaly_flags and anomaly_layer is None:
                anomaly_layer = i

            prev_hidden = hidden.copy()

        # Final output
        normed = rms_norm(hidden, engine.final_norm_weight)
        logits = engine.lm_head(normed)
        last_logits = logits[0, -1, :]
        top10_idx = np.argsort(last_logits)[-10:][::-1]
        print(f"\n  Final top-10:")
        for rank, idx in enumerate(top10_idx):
            tok = tokenizer.decode([idx])
            print(f"    {rank+1:2d}. '{tok}' ({last_logits[idx]:.2f})")

        # Target token info
        for tok_str in target_tokens:
            tok_ids = tokenizer.encode(tok_str)
            if tok_ids:
                tid = tok_ids[0]
                rank = int((last_logits > last_logits[tid]).sum())
                print(f"  Target '{tok_str}': rank={rank}, logit={last_logits[tid]:.2f}")

        if anomaly_layer is not None:
            print(f"\n  ⚠ First anomaly at Layer {anomaly_layer}")
        else:
            print(f"\n  No anomalies detected in hidden state magnitudes.")

    # ── Detailed analysis of problematic layers ──
    print(f"\n{'═' * 72}")
    print(f"  DETAILED: Per-layer logit shift analysis")
    print(f"{'═' * 72}")

    prompt_text = "The capital of France is"
    token_ids = tokenizer.encode(prompt_text)
    hidden = engine.embedding(token_ids)[np.newaxis, :, :]

    # Track how much each layer changes the logit distribution
    prev_logits = None
    for i, layer in enumerate(engine.layers):
        hidden = layer(hidden, pure=False)

        normed = rms_norm(hidden, engine.final_norm_weight)
        logits = engine.lm_head(normed)
        last_logits = logits[0, -1, :]

        if prev_logits is not None:
            delta = last_logits - prev_logits
            top_delta_idx = np.argsort(np.abs(delta))[-3:][::-1]
            top_delta = [(tokenizer.decode([idx]), float(delta[idx])) for idx in top_delta_idx]

            # Which tokens gain most?
            top_gain_idx = np.argsort(delta)[-3:][::-1]
            top_gain = [(tokenizer.decode([idx]), float(delta[idx])) for idx in top_gain_idx]

            paris_id = tokenizer.encode(' Paris')[0]
            paris_delta = float(delta[paris_id])
            paris_rank = int((last_logits > last_logits[paris_id]).sum())

            print(f"  L{i:2d}: Δmax={np.max(np.abs(delta)):7.2f}  "
                  f"Paris Δ={paris_delta:+7.2f} rank={paris_rank:6d}  "
                  f"Biggest gain: '{top_gain[0][0]}' ({top_gain[0][1]:+.1f})")

        prev_logits = last_logits.copy()

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
