#!/usr/bin/env python3
"""
A/B Test: Level-Weighted vs Unweighted Sign Navigation (Doc 254)

Tests whether weighting sign comparisons by φ-level proximity to zero
improves opposite-word navigation accuracy.

Design:
  - Training pairs: used to learn flip patterns + reference beam (same as production)
  - Held-out test pairs: NEVER seen during training, used for evaluation only
  - Both methods use identical reference beam and flip patterns
  - Only difference: how sign agreement is computed (weighted vs unweighted)

Metrics:
  - Top-1 accuracy: Is the correct opposite the #1 result?
  - Top-5 accuracy: Is the correct opposite in top 5?
  - Top-10 accuracy: Is the correct opposite in top 10?
  - Mean reciprocal rank (MRR): 1/rank of correct answer
  - Mean confidence: Average confidence score of top-1 result
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
import torch
import torch.nn.functional as F
import time
import json
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# ================================================================
# Training pairs (used to learn flip patterns — same as production)
# ================================================================
TRAIN_DIMENSIONS = {
    "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning"), ("icy", "fiery")],
    "size": [("small", "big"), ("tiny", "huge"), ("little", "large"), ("mini", "giant")],
    "speed": [("slow", "fast"), ("sluggish", "quick"), ("gradual", "rapid"), ("leisurely", "swift")],
    "height": [("short", "tall"), ("low", "high"), ("squat", "towering")],
    "brightness": [("dark", "bright"), ("dim", "light"), ("gloomy", "radiant")],
    "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale")],
    "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive"), ("evil", "good")],
    "weight": [("light", "heavy"), ("weightless", "weighty")],
    "hardness": [("soft", "hard"), ("tender", "tough"), ("gentle", "harsh")],
    "moisture": [("dry", "wet"), ("arid", "damp"), ("parched", "moist")],
    "emotion": [("love", "hate"), ("joy", "sorrow"), ("hope", "despair")],
    "wealth": [("rich", "poor"), ("wealthy", "impoverished"), ("affluent", "destitute")],
    "strength": [("strong", "weak"), ("powerful", "feeble"), ("mighty", "frail")],
    "volume": [("loud", "quiet"), ("noisy", "silent"), ("deafening", "mute")],
    "cleanliness": [("clean", "dirty"), ("pure", "filthy"), ("spotless", "grimy")],
    "truth": [("true", "false"), ("real", "fake"), ("genuine", "counterfeit")],
    "beauty": [("beautiful", "ugly"), ("pretty", "hideous"), ("gorgeous", "grotesque")],
    "intelligence": [("smart", "dumb"), ("clever", "stupid"), ("wise", "foolish")],
    "safety": [("safe", "dangerous"), ("secure", "risky"), ("harmless", "harmful")],
    "fullness": [("full", "empty"), ("complete", "incomplete"), ("whole", "partial")],
    "courage": [("brave", "coward"), ("bold", "timid"), ("fearless", "fearful")],
    "kindness": [("kind", "cruel"), ("gentle", "harsh"), ("caring", "callous")],
    "honesty": [("honest", "dishonest"), ("truthful", "deceitful"), ("sincere", "insincere")],
    "calmness": [("calm", "angry"), ("peaceful", "agitated"), ("serene", "furious")],
    "life": [("alive", "dead"), ("living", "deceased"), ("vital", "lifeless")],
    "consciousness": [("awake", "asleep"), ("alert", "drowsy"), ("conscious", "unconscious")],
}

# ================================================================
# Held-out test pairs (NEVER used for training)
# ================================================================
TEST_PAIRS = [
    # Same dimensions as training, different words
    ("warm", "chilly"),      # temperature-adjacent
    ("rapid", "sluggish"),   # speed (reversed from train)
    ("enormous", "minute"),  # size
    ("ancient", "modern"),   # age
    ("shallow", "deep"),     # depth (new dimension)
    ("narrow", "wide"),      # width (new dimension)
    ("smooth", "rough"),     # texture (new dimension)
    ("bitter", "sweet"),     # taste (new dimension)
    ("open", "closed"),      # state (new dimension)
    ("early", "late"),       # time (new dimension)
    ("sharp", "blunt"),      # sharpness (new dimension)
    ("proud", "humble"),     # pride (new dimension)
    ("cheap", "expensive"),  # cost (new dimension)
    ("simple", "complex"),   # complexity (new dimension)
    ("rare", "common"),      # frequency (new dimension)
    ("thick", "thin"),       # thickness (new dimension)
    ("loose", "tight"),      # fit (new dimension)
    ("dull", "sharp"),       # sharpness
    ("tame", "wild"),        # wildness
    ("fresh", "rotten"),     # freshness
    ("rigid", "flexible"),   # flexibility
    ("temporary", "permanent"), # duration
    ("visible", "invisible"),   # visibility
    ("normal", "abnormal"),     # normality
    ("guilty", "innocent"),     # guilt
    ("private", "public"),      # privacy
    ("abstract", "concrete"),   # abstraction
    ("passive", "active"),      # activity
    ("maximum", "minimum"),     # extremity
    ("internal", "external"),   # position
    ("amateur", "professional"), # skill
    ("positive", "negative"),   # polarity (in training but different context)
    ("domestic", "foreign"),    # origin
    ("natural", "artificial"),  # naturalness
    ("voluntary", "mandatory"), # choice
    ("major", "minor"),         # importance
    ("superior", "inferior"),   # quality
    ("optimistic", "pessimistic"), # outlook
    ("generous", "selfish"),    # generosity
    ("patient", "impatient"),   # patience
]

# Collect all training words to ensure no overlap
TRAIN_WORDS = set()
for pairs in TRAIN_DIMENSIONS.values():
    for w1, w2 in pairs:
        TRAIN_WORDS.add(w1.lower())
        TRAIN_WORDS.add(w2.lower())


def main():
    print("=" * 80)
    print("A/B TEST: Level-Weighted vs Unweighted Sign Navigation")
    print("=" * 80)
    print()

    # ================================================================
    # Load model embeddings
    # ================================================================
    print("Loading model embeddings...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    embeds = model.model.embed_tokens.weight.detach().float().cpu()
    hidden_dim = embeds.shape[1]
    vocab_size = embeds.shape[0]

    # Compute signs and levels
    all_signs = torch.sign(embeds).to(torch.int8)
    all_signs[all_signs == 0] = 1

    K = 128
    all_levels = torch.round(
        K * torch.log(torch.abs(embeds) + 1e-10) / LOG_PHI
    ).to(torch.int16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_signs = all_signs.to(device)
    all_levels = all_levels.to(device)

    del model
    torch.cuda.empty_cache()

    print(f"  {vocab_size} tokens, {hidden_dim} dims")

    # ================================================================
    # Helper functions
    # ================================================================
    def get_token_id(word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None

    def compute_level_weights(token_id):
        """φ-geometric decay: w = φ^(-|level|/K)"""
        levels = all_levels[token_id].float()
        return PHI ** (-torch.abs(levels) / K)

    def weighted_agreement(source_signs, target_all_signs, weights):
        """Level-weighted sign agreement."""
        matches = (target_all_signs == source_signs.unsqueeze(0)).float()
        return (matches * weights.unsqueeze(0)).sum(dim=1)

    def unweighted_agreement(source_signs, target_all_signs):
        """Standard unweighted sign agreement."""
        return (target_all_signs == source_signs.unsqueeze(0)).float().sum(dim=1)

    # ================================================================
    # Learn flip patterns from training data
    # ================================================================
    print("\nLearning flip patterns from training data...")
    flip_patterns = {}

    for dim_name, pairs in TRAIN_DIMENSIONS.items():
        flip_counts = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        n = 0
        for neg_word, pos_word in pairs:
            neg_id = get_token_id(neg_word)
            pos_id = get_token_id(pos_word)
            if neg_id is None or pos_id is None:
                continue
            flips = (all_signs[neg_id] != all_signs[pos_id]).float()
            flip_counts += flips
            n += 1
        if n > 0:
            flip_prob = flip_counts / n
            flip_patterns[dim_name] = (flip_prob > 0.5)

    print(f"  Learned {len(flip_patterns)} dimensions")

    # Compute holographic reference beam
    all_flip_probs = []
    for dim_name, pairs in TRAIN_DIMENSIONS.items():
        flip_counts = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        n = 0
        for neg_word, pos_word in pairs:
            neg_id = get_token_id(neg_word)
            pos_id = get_token_id(pos_word)
            if neg_id is None or pos_id is None:
                continue
            flips = (all_signs[neg_id] != all_signs[pos_id]).float()
            flip_counts += flips
            n += 1
        if n > 0:
            all_flip_probs.append(flip_counts / n)

    flip_matrix = torch.stack(all_flip_probs)
    U, S, Vt = torch.linalg.svd(flip_matrix.cpu())
    reference_beam = Vt[0].to(device)
    variance_captured = (S[0]**2 / (S**2).sum() * 100).item()
    print(f"  Reference beam: {variance_captured:.1f}% variance captured")

    # ================================================================
    # Holographic navigation function (both modes)
    # ================================================================
    def navigate_holographic(word, use_weighted, alpha=0.5):
        """
        Navigate to find opposite using holographic projection.
        Returns (result_word, score, rank_of_correct, all_top_words)
        """
        word_id = get_token_id(word)
        if word_id is None:
            return None, 0, -1, []

        source_signs = all_signs[word_id].float()

        # Flip where reference beam is strong
        flip_strength = reference_beam.abs()
        flip_threshold = flip_strength.mean() + flip_strength.std() * alpha
        flip_mask = flip_strength > flip_threshold

        target_signs = source_signs.clone()
        target_signs[flip_mask] *= -1

        # Compute agreement (THE KEY DIFFERENCE)
        if use_weighted:
            weights = compute_level_weights(word_id).to(device)
            agreement = weighted_agreement(target_signs.to(torch.int8), all_signs, weights)
            weight_sum = weights.sum().item()
        else:
            agreement = unweighted_agreement(target_signs.to(torch.int8), all_signs)
            weight_sum = float(hidden_dim)

        agreement[word_id] = -1

        # Get top-20 results
        top_vals, top_idxs = agreement.topk(20)
        top_words = []
        for idx, val in zip(top_idxs, top_vals):
            w = tokenizer.decode([idx.item()]).strip()
            top_words.append((w, val.item() / weight_sum * 100))

        top_word = top_words[0][0] if top_words else ""
        top_score = top_words[0][1] if top_words else 0

        return top_word, top_score, weight_sum, top_words

    # ================================================================
    # Run A/B test on held-out pairs
    # ================================================================
    print("\n" + "=" * 80)
    print("A/B TEST ON HELD-OUT PAIRS")
    print("=" * 80)
    print()

    # Filter test pairs: both words must be single tokens and not in training
    valid_pairs = []
    skipped = []
    for w1, w2 in TEST_PAIRS:
        id1 = get_token_id(w1)
        id2 = get_token_id(w2)
        if id1 is None or id2 is None:
            skipped.append((w1, w2, "token not found"))
            continue
        valid_pairs.append((w1, w2))

    print(f"Valid test pairs: {len(valid_pairs)}/{len(TEST_PAIRS)}")
    if skipped:
        print(f"Skipped: {len(skipped)} (token issues)")
    print()

    # Test both directions for each pair
    results = {"weighted": [], "unweighted": []}

    header = f"{'Word':<15} {'Expected':<15} {'Weighted top-1':<18} {'W conf':<8} {'Unweighted top-1':<18} {'U conf':<8} {'W>U?'}"
    print(header)
    print("-" * len(header))

    for w1, w2 in valid_pairs:
        for source, target in [(w1, w2), (w2, w1)]:
            # Weighted
            w_word, w_score, _, w_top = navigate_holographic(source, use_weighted=True)
            # Unweighted
            u_word, u_score, _, u_top = navigate_holographic(source, use_weighted=False)

            # Find rank of target in each
            w_rank = -1
            u_rank = -1
            target_lower = target.lower()
            for i, (tw, _) in enumerate(w_top):
                if tw.lower() == target_lower:
                    w_rank = i + 1
                    break
            for i, (tw, _) in enumerate(u_top):
                if tw.lower() == target_lower:
                    u_rank = i + 1
                    break

            results["weighted"].append({
                "source": source, "target": target,
                "top1": w_word, "confidence": w_score,
                "rank": w_rank, "top5": [w[0] for w in w_top[:5]],
            })
            results["unweighted"].append({
                "source": source, "target": target,
                "top1": u_word, "confidence": u_score,
                "rank": u_rank, "top5": [w[0] for w in u_top[:5]],
            })

            w_match = "✓" if w_word.lower() == target_lower else ""
            u_match = "✓" if u_word.lower() == target_lower else ""
            better = "★" if w_score > u_score else ""

            print(f"  {source:<13} {target:<13}   {w_word:<15}{w_match:<3}{w_score:>5.1f}%  "
                  f"{u_word:<15}{u_match:<3}{u_score:>5.1f}%  {better}")

    # ================================================================
    # Compute metrics
    # ================================================================
    print()
    print("=" * 80)
    print("METRICS")
    print("=" * 80)
    print()

    for method in ["weighted", "unweighted"]:
        r = results[method]
        n = len(r)

        top1_exact = sum(1 for x in r if x["top1"].lower() == x["target"].lower())
        top5_found = sum(1 for x in r if 0 < x["rank"] <= 5)
        top10_found = sum(1 for x in r if 0 < x["rank"] <= 10)
        top20_found = sum(1 for x in r if 0 < x["rank"] <= 20)

        mrr_vals = [1.0 / x["rank"] if x["rank"] > 0 else 0 for x in r]
        mrr = np.mean(mrr_vals)

        mean_conf = np.mean([x["confidence"] for x in r])

        label = method.upper()
        print(f"  {label}:")
        print(f"    Top-1  exact match:  {top1_exact:3d}/{n} ({top1_exact/n*100:5.1f}%)")
        print(f"    Top-5  contains:     {top5_found:3d}/{n} ({top5_found/n*100:5.1f}%)")
        print(f"    Top-10 contains:     {top10_found:3d}/{n} ({top10_found/n*100:5.1f}%)")
        print(f"    Top-20 contains:     {top20_found:3d}/{n} ({top20_found/n*100:5.1f}%)")
        print(f"    MRR:                 {mrr:.4f}")
        print(f"    Mean confidence:     {mean_conf:.1f}%")
        print()

    # Direct comparison
    print("=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    print()

    w_wins = 0
    u_wins = 0
    ties = 0
    w_better_conf = 0
    u_better_conf = 0

    for wr, ur in zip(results["weighted"], results["unweighted"]):
        w_found = wr["rank"] > 0
        u_found = ur["rank"] > 0
        w_r = wr["rank"] if w_found else 999
        u_r = ur["rank"] if u_found else 999

        if w_r < u_r:
            w_wins += 1
        elif u_r < w_r:
            u_wins += 1
        else:
            ties += 1

        if wr["confidence"] > ur["confidence"]:
            w_better_conf += 1
        elif ur["confidence"] > wr["confidence"]:
            u_better_conf += 1

    n = len(results["weighted"])
    print(f"  Weighted finds target at better rank: {w_wins:3d}/{n} ({w_wins/n*100:.1f}%)")
    print(f"  Unweighted finds target at better rank: {u_wins:3d}/{n} ({u_wins/n*100:.1f}%)")
    print(f"  Ties (same rank):                       {ties:3d}/{n} ({ties/n*100:.1f}%)")
    print()
    print(f"  Weighted has higher confidence:   {w_better_conf:3d}/{n} ({w_better_conf/n*100:.1f}%)")
    print(f"  Unweighted has higher confidence: {u_better_conf:3d}/{n} ({u_better_conf/n*100:.1f}%)")
    print()

    # Verdict
    w_top1 = sum(1 for x in results["weighted"] if x["top1"].lower() == x["target"].lower())
    u_top1 = sum(1 for x in results["unweighted"] if x["top1"].lower() == x["target"].lower())
    w_mrr = np.mean([1.0/x["rank"] if x["rank"] > 0 else 0 for x in results["weighted"]])
    u_mrr = np.mean([1.0/x["rank"] if x["rank"] > 0 else 0 for x in results["unweighted"]])

    print("=" * 80)
    if w_top1 > u_top1 or (w_top1 == u_top1 and w_mrr > u_mrr):
        print(f"VERDICT: WEIGHTED WINS (top-1: {w_top1} vs {u_top1}, MRR: {w_mrr:.4f} vs {u_mrr:.4f})")
    elif u_top1 > w_top1 or (u_top1 == w_top1 and u_mrr > w_mrr):
        print(f"VERDICT: UNWEIGHTED WINS (top-1: {u_top1} vs {w_top1}, MRR: {u_mrr:.4f} vs {w_mrr:.4f})")
    else:
        print(f"VERDICT: TIE (top-1: {w_top1} vs {u_top1}, MRR: {w_mrr:.4f} vs {u_mrr:.4f})")
    print("=" * 80)

    # Save results
    out = {
        "n_test_pairs": len(valid_pairs),
        "n_test_directions": n,
        "weighted": {
            "top1_accuracy": w_top1 / n * 100,
            "mrr": float(w_mrr),
            "mean_confidence": float(np.mean([x["confidence"] for x in results["weighted"]])),
        },
        "unweighted": {
            "top1_accuracy": u_top1 / n * 100,
            "mrr": float(u_mrr),
            "mean_confidence": float(np.mean([x["confidence"] for x in results["unweighted"]])),
        },
        "head_to_head": {
            "weighted_better_rank": w_wins,
            "unweighted_better_rank": u_wins,
            "ties": ties,
        },
    }

    import os
    out_path = '/home/thorin/truthspace-lcm/experiments/model_reverse_engineering_v2/results/ab_test_weighted_nav.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
