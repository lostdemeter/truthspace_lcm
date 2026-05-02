#!/usr/bin/env python3
"""
TruthSpace Gate Exploration — Darwin's Notebook
=================================================

Systematic investigation of four questions arising from truthspace_v1 results:

1. SELECTIVE OVERRIDE: VA baseline + hard-override only confident dims
   (instead of blending which collapses to VA)

2. WHY GATEC1 WINS ON LANGUAGE: What's different about language vs capital
   gate structure that makes per-dim gating win?

3. PRESERVE DETECTION: Why 0 dims are classified as PRESERVE —
   what do the actual distributions look like?

4. ANCHOR SYSTEM: Can F62 sign rules (from model layer analysis)
   inform which dims should be SHIFT/FLIP/PRESERVE?

Each investigation writes findings to field_notes.md
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "phi_model"
RULES_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "results" / "phase4_rules_full"
sys.path.insert(0, str(PROJECT_ROOT))

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
PHI_GRID = 128


def phi_encode(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    signs = np.sign(values).astype(np.int8)
    signs[signs == 0] = 1
    magnitudes = np.abs(values).astype(np.float64) + 1e-20
    exponents = np.round(PHI_GRID * np.log(magnitudes) / LOG_PHI).astype(np.int16)
    return signs, exponents


def phi_decode(signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    return (
        signs.astype(np.float64)
        * PHI ** (exponents.astype(np.float64) / PHI_GRID)
    ).astype(np.float32)


# =============================================================================
# DATA LOADING (reused from truthspace_v1)
# =============================================================================

@dataclass
class Concept:
    name: str
    token_id: int
    token_str: str
    embedding: np.ndarray
    phi_signs: np.ndarray
    phi_exponents: np.ndarray


class VocabSearcher:
    def __init__(self, embeddings, id_to_token):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb_normed = embeddings / (norms + 1e-20)
        self.id_to_token = id_to_token
        self.n_vocab = embeddings.shape[0]

    def rank_of(self, composed, target_tid, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        target_sim = sims[target_tid]
        return int(np.sum(sims > target_sim))

    def top_k(self, composed, k=5, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self.id_to_token.get(int(i), f"?{i}"), float(sims[i]))
                for i in top_idx]


def load_data():
    """Load embeddings, tokenizer, build concepts."""
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings...")
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file, "r") as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    token_to_id = {tok: idx for tok, idx in vocab.items()}

    def find_token_id(word):
        candidates = [
            word, word.lower(), word.capitalize(), word.upper(),
            f"Ġ{word}", f"Ġ{word.lower()}", f"Ġ{word.capitalize()}",
            f"▁{word}", f"▁{word.lower()}", f"▁{word.capitalize()}",
            f" {word}", f" {word.lower()}", f" {word.capitalize()}",
        ]
        for c in candidates:
            if c in token_to_id:
                return token_to_id[c], c
        return None, None

    # Build concepts for all words we need
    all_words = [
        "France", "Paris", "Germany", "Berlin", "Japan", "Tokyo",
        "China", "Beijing", "Egypt", "Cairo",
        "Australia", "Canberra", "Thailand", "Bangkok", "Poland", "Warsaw",
        "Norway", "Oslo", "Sweden", "Stockholm", "India", "Delhi",
        "Brazil", "Brasilia", "Korea", "Seoul",
        "French", "German", "Japanese", "Chinese", "Spanish",
        "Italy", "Italian", "Portugal", "Portuguese", "Russia", "Russian",
        "Spain",
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "brother", "sister", "son", "daughter",
        "husband", "wife",
        "hot", "cold", "big", "small", "fast", "slow",
        "good", "bad", "happy", "sad", "long", "short", "up", "down",
    ]

    concepts = {}
    for word in all_words:
        tid, tok_str = find_token_id(word)
        if tid is not None:
            concepts[word] = Concept(
                name=word, token_id=tid, token_str=tok_str,
                embedding=embeddings[tid],
                phi_signs=phi.signs[tid],
                phi_exponents=phi.exponents[tid],
            )

    searcher = VocabSearcher(embeddings, id_to_token)
    print(f"Loaded {len(concepts)} concepts, {searcher.n_vocab} vocab tokens")
    return concepts, searcher, embeddings, phi.signs, phi.exponents


# =============================================================================
# FIELD NOTES WRITER
# =============================================================================

class FieldNotes:
    """Darwin's notebook — accumulate observations."""

    def __init__(self):
        self.notes = []
        self.section_count = 0

    def section(self, title):
        self.section_count += 1
        self.notes.append(f"\n## {self.section_count}. {title}\n")
        print(f"\n{'='*70}")
        print(f"  INVESTIGATION {self.section_count}: {title}")
        print(f"{'='*70}\n")

    def observe(self, text):
        self.notes.append(text)
        for line in text.split('\n'):
            print(f"  {line}")

    def data_table(self, headers, rows, title=""):
        if title:
            self.notes.append(f"\n**{title}**\n")
            print(f"\n  {title}")
        header_str = "| " + " | ".join(f"{h:>12s}" for h in headers) + " |"
        sep_str = "| " + " | ".join("-" * 12 for _ in headers) + " |"
        self.notes.append(header_str)
        self.notes.append(sep_str)
        print(f"  {header_str}")
        print(f"  {sep_str}")
        for row in rows:
            row_str = "| " + " | ".join(f"{str(v):>12s}" for v in row) + " |"
            self.notes.append(row_str)
            print(f"  {row_str}")

    def finding(self, text):
        self.notes.append(f"\n> **FINDING:** {text}\n")
        print(f"\n  >>> FINDING: {text}\n")

    def save(self, path):
        header = """# TruthSpace Gate Exploration — Field Notes
*Generated by explore_gates.py*

These are systematic observations from investigating why BoomWt collapses
to VecAdd and what geometric structure actually exists in the embedding space.

"""
        with open(path, "w") as f:
            f.write(header + "\n".join(self.notes))
        print(f"\n  Field notes saved to {path}")


# =============================================================================
# INVESTIGATION 1: SELECTIVE OVERRIDE
# =============================================================================

def investigate_selective_override(concepts, searcher, notes):
    """
    Instead of confidence-weighted blending (which collapses to VA),
    try hard override: VA everywhere, but replace specific dims with
    boom values. Vary the confidence threshold for which dims to override.
    """
    notes.section("Selective Override — Hard Replace vs Blend")

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    capital_test = [
        ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
        ("Brazil", "Brasilia"),
    ]

    # Compute training data
    train_pairs = [(concepts[e], concepts[a]) for e, a in capital_train
                   if e in concepts and a in concepts]

    avg_delta = np.mean([a.embedding - e.embedding for e, a in train_pairs], axis=0)

    # Compute per-dim statistics
    deltas = np.array([a.embedding - e.embedding for e, a in train_pairs])
    mean_delta = np.mean(deltas, axis=0)
    std_delta = np.std(deltas, axis=0)

    # Coefficient of variation (lower = more consistent)
    cv = std_delta / (np.abs(mean_delta) + 1e-20)

    # Exponent-space analysis
    entity_exps = np.array([e.phi_exponents.astype(np.int32) for e, _ in train_pairs])
    answer_exps = np.array([a.phi_exponents.astype(np.int32) for _, a in train_pairs])
    exp_deltas = answer_exps - entity_exps
    exp_iqr = np.percentile(exp_deltas, 75, axis=0) - np.percentile(exp_deltas, 25, axis=0)
    exp_median = np.median(exp_deltas, axis=0).astype(np.int16)

    # Sign analysis
    entity_signs = np.array([e.phi_signs for e, _ in train_pairs])
    answer_signs = np.array([a.phi_signs for _, a in train_pairs])
    sign_products = entity_signs * answer_signs
    flip_rate = np.mean(sign_products < 0, axis=0)
    keep_rate = np.mean(sign_products > 0, axis=0)

    notes.observe("Per-dimension statistics across training pairs:")
    notes.observe(f"  Mean |CV|:  {np.mean(cv):.2f}")
    notes.observe(f"  CV < 0.5:   {np.sum(cv < 0.5)} dims ({np.sum(cv < 0.5)/3584*100:.1f}%)")
    notes.observe(f"  CV < 1.0:   {np.sum(cv < 1.0)} dims ({np.sum(cv < 1.0)/3584*100:.1f}%)")
    notes.observe(f"  CV < 2.0:   {np.sum(cv < 2.0)} dims ({np.sum(cv < 2.0)/3584*100:.1f}%)")
    notes.observe(f"  Exp IQR=0:  {np.sum(exp_iqr == 0)} dims ({np.sum(exp_iqr == 0)/3584*100:.1f}%)")
    notes.observe(f"  Exp IQR≤1:  {np.sum(exp_iqr <= 1)} dims ({np.sum(exp_iqr <= 1)/3584*100:.1f}%)")
    notes.observe(f"  Exp IQR≤2:  {np.sum(exp_iqr <= 2)} dims ({np.sum(exp_iqr <= 2)/3584*100:.1f}%)")
    notes.observe(f"  Sign flip≥0.8: {np.sum(flip_rate >= 0.8)} dims")
    notes.observe(f"  Sign keep≥0.8: {np.sum(keep_rate >= 0.8)} dims")

    # Test strategy: VA baseline, override N most confident dims
    # Confidence = inverse of CV (for float) or inverse of IQR (for int)
    float_confidence = 1.0 / (cv + 0.01)
    int_confidence = 1.0 / (exp_iqr + 0.1)

    # Sort dims by confidence (most confident first)
    float_order = np.argsort(-float_confidence)
    int_order = np.argsort(-int_confidence)

    notes.observe("\nSelective override sweep: replace top-N dims in VA result with gate values")

    override_counts = [0, 10, 25, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3584]
    rows = []

    for n_override in override_counts:
        # Float-space selective override
        float_ranks = []
        int_ranks = []
        va_ranks = []

        for entity_name, answer_name in capital_test:
            e = concepts.get(entity_name)
            a = concepts.get(answer_name)
            if not e or not a:
                continue
            ex = {e.token_id}

            # Pure VA
            va_pred = e.embedding + avg_delta
            va_r = searcher.rank_of(va_pred, a.token_id, ex)
            va_ranks.append(va_r)

            if n_override == 0:
                float_ranks.append(va_r)
                int_ranks.append(va_r)
                continue

            # Float override: replace top-N dims with entity + mean_delta
            float_pred = va_pred.copy()
            override_dims = float_order[:n_override]
            float_pred[override_dims] = e.embedding[override_dims] + mean_delta[override_dims]
            float_ranks.append(searcher.rank_of(float_pred, a.token_id, ex))

            # Int override: replace top-N dims with integer-shifted values
            int_pred = va_pred.copy()
            override_dims_int = int_order[:n_override]
            new_signs = e.phi_signs.copy()
            new_exps = e.phi_exponents.copy().astype(np.int32)
            new_exps += exp_median.astype(np.int32)
            # Flip signs where consistent
            flip_dims = np.where(flip_rate >= 0.8)[0]
            new_signs[flip_dims] *= -1
            int_vals = phi_decode(new_signs, new_exps.astype(np.int16))
            int_pred[override_dims_int] = int_vals[override_dims_int]
            int_ranks.append(searcher.rank_of(int_pred, a.token_id, ex))

        rows.append([
            str(n_override),
            f"{np.mean(va_ranks):.1f}",
            f"{np.mean(float_ranks):.1f}",
            f"{np.mean(int_ranks):.1f}",
            f"{np.median(float_ranks):.0f}",
            f"{np.median(int_ranks):.0f}",
        ])

    notes.data_table(
        ["N_override", "VA_mean", "Float_mean", "Int_mean", "Float_med", "Int_med"],
        rows,
        "Override top-N most confident dims (capital test)"
    )

    # Now try the opposite: override only sign-flip dims
    notes.observe("\n--- Sign-only override: flip signs where consistent, keep VA magnitudes ---")

    sign_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sign_rows = []

    for thresh in sign_thresholds:
        flip_dims = np.where(flip_rate >= thresh)[0]
        keep_dims = np.where(keep_rate >= thresh)[0]

        ranks_sign_override = []
        for entity_name, answer_name in capital_test:
            e = concepts.get(entity_name)
            a = concepts.get(answer_name)
            if not e or not a:
                continue
            ex = {e.token_id}

            # VA baseline
            pred = e.embedding + avg_delta

            # Override signs only
            pred[flip_dims] = np.abs(pred[flip_dims]) * (-np.sign(e.embedding[flip_dims] + 1e-20))
            pred[keep_dims] = np.abs(pred[keep_dims]) * np.sign(e.embedding[keep_dims] + 1e-20)

            ranks_sign_override.append(searcher.rank_of(pred, a.token_id, ex))

        sign_rows.append([
            f"{thresh:.1f}",
            str(len(flip_dims)),
            str(len(keep_dims)),
            f"{np.mean(ranks_sign_override):.1f}",
            f"{np.median(ranks_sign_override):.0f}",
        ])

    notes.data_table(
        ["FlipThresh", "N_flip", "N_keep", "Mean_rank", "Median"],
        sign_rows,
        "Sign-only override (VA magnitudes + forced signs)"
    )

    # Key experiment: what if we ONLY use the most confident dims?
    notes.observe("\n--- Extreme test: predict using ONLY top-N dims (zero rest) ---")
    extreme_rows = []
    for n_dims in [50, 100, 200, 500, 1000]:
        dims = int_order[:n_dims]
        ranks = []
        for entity_name, answer_name in capital_test:
            e = concepts.get(entity_name)
            a = concepts.get(answer_name)
            if not e or not a:
                continue
            ex = {e.token_id}

            # Build sparse prediction: only top-N dims
            pred = np.zeros(3584, dtype=np.float32)
            pred[dims] = e.embedding[dims] + mean_delta[dims]
            ranks.append(searcher.rank_of(pred, a.token_id, ex))

        extreme_rows.append([str(n_dims), f"{np.mean(ranks):.1f}", f"{np.median(ranks):.0f}"])

    notes.data_table(
        ["N_dims", "Mean_rank", "Median"],
        extreme_rows,
        "Prediction using ONLY top-N most confident dims"
    )


# =============================================================================
# INVESTIGATION 2: WHY GATEC1 WINS ON LANGUAGE
# =============================================================================

def investigate_gatec1_language(concepts, searcher, notes):
    """
    GateC1 (per-dim gate with comb_scale=1.0) beats VecAdd on language
    but loses on capitals. What structural difference explains this?
    """
    notes.section("Why GateC1 Wins on Language — Structural Analysis")

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    language_train = [
        ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
        ("China", "Chinese"), ("Spain", "Spanish"),
    ]

    def analyze_relationship(name, pairs):
        """Deep per-dim analysis of a relationship."""
        valid = [(concepts[e], concepts[a]) for e, a in pairs
                 if e in concepts and a in concepts]

        deltas = np.array([a.embedding - e.embedding for e, a in valid])
        mean_delta = np.mean(deltas, axis=0)
        std_delta = np.std(deltas, axis=0)
        cv = std_delta / (np.abs(mean_delta) + 1e-20)

        entity_signs = np.array([e.phi_signs for e, _ in valid])
        answer_signs = np.array([a.phi_signs for _, a in valid])
        sign_products = entity_signs * answer_signs
        flip_rate = np.mean(sign_products < 0, axis=0)
        keep_rate = np.mean(sign_products > 0, axis=0)

        entity_exps = np.array([e.phi_exponents.astype(np.int32) for e, _ in valid])
        answer_exps = np.array([a.phi_exponents.astype(np.int32) for _, a in valid])
        exp_deltas = answer_exps - entity_exps
        exp_iqr = np.percentile(exp_deltas, 75, axis=0) - np.percentile(exp_deltas, 25, axis=0)

        # Cosine similarity between entity and answer
        cosines = []
        for e, a in valid:
            cos = np.dot(e.embedding, a.embedding) / (
                np.linalg.norm(e.embedding) * np.linalg.norm(a.embedding) + 1e-20)
            cosines.append(cos)

        return {
            "name": name,
            "n_pairs": len(valid),
            "mean_delta_norm": float(np.linalg.norm(mean_delta)),
            "mean_cv": float(np.mean(cv)),
            "median_cv": float(np.median(cv)),
            "cv_lt_05": int(np.sum(cv < 0.5)),
            "cv_lt_10": int(np.sum(cv < 1.0)),
            "flip_ge_08": int(np.sum(flip_rate >= 0.8)),
            "keep_ge_08": int(np.sum(keep_rate >= 0.8)),
            "exp_iqr_0": int(np.sum(exp_iqr == 0)),
            "exp_iqr_le_1": int(np.sum(exp_iqr <= 1)),
            "mean_entity_answer_cos": float(np.mean(cosines)),
            "delta_magnitude_per_dim": float(np.mean(np.abs(mean_delta))),
            "cv": cv,
            "flip_rate": flip_rate,
            "keep_rate": keep_rate,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
        }

    cap_stats = analyze_relationship("capital-of", capital_train)
    lang_stats = analyze_relationship("language-of", language_train)

    notes.observe("Structural comparison of capital-of vs language-of:")
    notes.data_table(
        ["Metric", "Capital", "Language"],
        [
            ["N pairs", str(cap_stats["n_pairs"]), str(lang_stats["n_pairs"])],
            ["Mean |delta|", f"{cap_stats['mean_delta_norm']:.2f}", f"{lang_stats['mean_delta_norm']:.2f}"],
            ["Mean CV", f"{cap_stats['mean_cv']:.2f}", f"{lang_stats['mean_cv']:.2f}"],
            ["Median CV", f"{cap_stats['median_cv']:.2f}", f"{lang_stats['median_cv']:.2f}"],
            ["CV < 0.5", str(cap_stats["cv_lt_05"]), str(lang_stats["cv_lt_05"])],
            ["CV < 1.0", str(cap_stats["cv_lt_10"]), str(lang_stats["cv_lt_10"])],
            ["Sign flip≥0.8", str(cap_stats["flip_ge_08"]), str(lang_stats["flip_ge_08"])],
            ["Sign keep≥0.8", str(cap_stats["keep_ge_08"]), str(lang_stats["keep_ge_08"])],
            ["Exp IQR=0", str(cap_stats["exp_iqr_0"]), str(lang_stats["exp_iqr_0"])],
            ["Exp IQR≤1", str(cap_stats["exp_iqr_le_1"]), str(lang_stats["exp_iqr_le_1"])],
            ["Entity↔Ans cos", f"{cap_stats['mean_entity_answer_cos']:.4f}", f"{lang_stats['mean_entity_answer_cos']:.4f}"],
            ["|delta|/dim", f"{cap_stats['delta_magnitude_per_dim']:.6f}", f"{lang_stats['delta_magnitude_per_dim']:.6f}"],
        ],
        "Relationship structure comparison"
    )

    # Which dims are confident in language but not capital (and vice versa)?
    cap_confident = set(np.where(cap_stats["cv"] < 0.5)[0])
    lang_confident = set(np.where(lang_stats["cv"] < 0.5)[0])

    both = cap_confident & lang_confident
    cap_only = cap_confident - lang_confident
    lang_only = lang_confident - cap_confident

    notes.observe(f"\nConfident dim overlap (CV < 0.5):")
    notes.observe(f"  Both: {len(both)} dims")
    notes.observe(f"  Capital-only: {len(cap_only)} dims")
    notes.observe(f"  Language-only: {len(lang_only)} dims")

    # What do the shared confident dims look like?
    if both:
        shared = list(both)
        cap_shift_shared = cap_stats["mean_delta"][shared]
        lang_shift_shared = lang_stats["mean_delta"][shared]
        shift_corr = np.corrcoef(cap_shift_shared, lang_shift_shared)[0, 1]
        notes.observe(f"  Shift correlation on shared dims: {shift_corr:.4f}")

    # GateC1 applies: entity + shift for SHIFT/FLIP dims, entity * comb_scale for COMB
    # With comb_scale=1.0, COMB dims keep entity value (identity)
    # So GateC1 is: modify confident dims, leave uncertain dims as entity value
    # VA is: modify ALL dims by avg_delta

    # The key question: is "leave uncertain dims alone" better for language?
    notes.observe("\nWhy GateC1 works for language: decomposing the effect")
    notes.observe("GateC1(comb=1) = modify SHIFT+FLIP dims, keep entity for COMB dims")
    notes.observe("VecAdd = modify ALL dims by avg_delta")
    notes.observe("Difference: on COMB dims, GateC1 keeps entity, VA adds delta")

    # Test: what if we use entity value on COMB dims + VA on SHIFT/FLIP dims?
    # This is the "inverse" of our normal approach
    for rel_name, train_pairs, test_pairs in [
        ("capital", capital_train, [
            ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
            ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
            ("Brazil", "Brasilia"),
        ]),
        ("language", language_train, [
            ("Italy", "Italian"), ("Portugal", "Portuguese"), ("Russia", "Russian"),
        ]),
    ]:
        valid_train = [(concepts[e], concepts[a]) for e, a in train_pairs
                       if e in concepts and a in concepts]
        avg_d = np.mean([a.embedding - e.embedding for e, a in valid_train], axis=0)
        deltas = np.array([a.embedding - e.embedding for e, a in valid_train])
        cv = np.std(deltas, axis=0) / (np.abs(np.mean(deltas, axis=0)) + 1e-20)

        # Different strategies for handling uncertain dims
        methods = {}
        for cv_thresh in [0.3, 0.5, 1.0, 2.0]:
            confident = cv < cv_thresh
            n_conf = np.sum(confident)

            va_ranks = []
            hybrid_ranks = []  # confident=per-dim, uncertain=VA
            entity_ranks = []  # confident=per-dim, uncertain=entity

            for entity_name, answer_name in test_pairs:
                e = concepts.get(entity_name)
                a = concepts.get(answer_name)
                if not e or not a:
                    continue
                ex = {e.token_id}

                # VA
                va_pred = e.embedding + avg_d
                va_ranks.append(searcher.rank_of(va_pred, a.token_id, ex))

                # Hybrid: per-dim shift on confident, VA on uncertain
                mean_d = np.mean(deltas, axis=0)
                hybrid_pred = e.embedding + avg_d  # start with VA
                hybrid_pred[confident] = e.embedding[confident] + mean_d[confident]
                hybrid_ranks.append(searcher.rank_of(hybrid_pred, a.token_id, ex))

                # Entity-preserve: per-dim shift on confident, entity on uncertain
                entity_pred = e.embedding.copy()  # start with entity
                entity_pred[confident] += mean_d[confident]
                entity_ranks.append(searcher.rank_of(entity_pred, a.token_id, ex))

            methods[cv_thresh] = (n_conf, np.mean(va_ranks), np.mean(hybrid_ranks), np.mean(entity_ranks))

        rows = []
        for cv_t, (n_conf, va_m, hyb_m, ent_m) in methods.items():
            rows.append([f"{cv_t:.1f}", str(n_conf),
                        f"{va_m:.1f}", f"{hyb_m:.1f}", f"{ent_m:.1f}"])

        notes.data_table(
            ["CV_thresh", "N_conf", "VA_mean", "Hybrid_mean", "Entity_mean"],
            rows,
            f"{rel_name}: uncertain dim handling comparison"
        )

    return cap_stats, lang_stats


# =============================================================================
# INVESTIGATION 3: PRESERVE DETECTION
# =============================================================================

def investigate_preserve(concepts, notes):
    """
    Why are 0 dims classified as PRESERVE? Look at actual distributions.
    """
    notes.section("PRESERVE Detection — Why Zero Preserved Dims?")

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]

    valid = [(concepts[e], concepts[a]) for e, a in capital_train
             if e in concepts and a in concepts]

    deltas = np.array([a.embedding - e.embedding for e, a in valid])
    entity_mags = np.mean(np.abs(np.array([e.embedding for e, _ in valid])), axis=0)
    mean_abs_delta = np.mean(np.abs(deltas), axis=0)
    relative_delta = mean_abs_delta / (entity_mags + 1e-20)

    notes.observe("PRESERVE criterion: relative_delta < preserve_threshold")
    notes.observe(f"  relative_delta = mean(|answer - entity|) / mean(|entity|)")
    notes.observe(f"")
    notes.observe(f"Distribution of relative_delta across 3584 dims:")
    notes.observe(f"  min:    {np.min(relative_delta):.4f}")
    notes.observe(f"  5th%:   {np.percentile(relative_delta, 5):.4f}")
    notes.observe(f"  25th%:  {np.percentile(relative_delta, 25):.4f}")
    notes.observe(f"  median: {np.median(relative_delta):.4f}")
    notes.observe(f"  75th%:  {np.percentile(relative_delta, 75):.4f}")
    notes.observe(f"  95th%:  {np.percentile(relative_delta, 95):.4f}")
    notes.observe(f"  max:    {np.max(relative_delta):.4f}")

    # How many dims would be PRESERVE at various thresholds?
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    rows = []
    for t in thresholds:
        n = int(np.sum(relative_delta < t))
        rows.append([f"{t:.2f}", str(n), f"{n/3584*100:.1f}%"])
    notes.data_table(
        ["Threshold", "N_preserve", "Pct"],
        rows,
        "PRESERVE count at various thresholds"
    )

    notes.observe(f"\nThe problem: with threshold=0.1, the minimum relative_delta is {np.min(relative_delta):.4f}")
    notes.observe(f"ALL dims change significantly relative to entity magnitude.")

    # But wait — maybe in EXPONENT space things are different
    entity_exps = np.array([e.phi_exponents.astype(np.int32) for e, _ in valid])
    answer_exps = np.array([a.phi_exponents.astype(np.int32) for _, a in valid])
    exp_deltas = answer_exps - entity_exps

    exp_abs_mean = np.mean(np.abs(exp_deltas), axis=0)

    notes.observe(f"\nIn exponent space:")
    notes.observe(f"  Mean |exp_delta| distribution:")
    notes.observe(f"  min:    {np.min(exp_abs_mean):.1f}")
    notes.observe(f"  5th%:   {np.percentile(exp_abs_mean, 5):.1f}")
    notes.observe(f"  25th%:  {np.percentile(exp_abs_mean, 25):.1f}")
    notes.observe(f"  median: {np.median(exp_abs_mean):.1f}")
    notes.observe(f"  75th%:  {np.percentile(exp_abs_mean, 75):.1f}")
    notes.observe(f"  max:    {np.max(exp_abs_mean):.1f}")

    exp_thresholds = [0, 1, 2, 3, 5, 10, 20]
    rows = []
    for t in exp_thresholds:
        n = int(np.sum(exp_abs_mean <= t))
        rows.append([str(t), str(n), f"{n/3584*100:.1f}%"])
    notes.data_table(
        ["MaxExpDelta", "N_preserve", "Pct"],
        rows,
        "PRESERVE in exponent space (mean |exp_delta| ≤ threshold)"
    )

    # What about looking at it differently: which dims have the SAME value
    # across all entities? Those are the "structural" dims.
    entity_embeddings = np.array([e.embedding for e, _ in valid])
    entity_cv = np.std(entity_embeddings, axis=0) / (np.abs(np.mean(entity_embeddings, axis=0)) + 1e-20)

    notes.observe(f"\nEntity-side stability (how much do entities differ on each dim?):")
    notes.observe(f"  Mean entity CV: {np.mean(entity_cv):.2f}")
    notes.observe(f"  Median entity CV: {np.median(entity_cv):.2f}")
    notes.observe(f"  Entity CV < 0.1: {np.sum(entity_cv < 0.1)} dims (possible structural constants)")
    notes.observe(f"  Entity CV < 0.5: {np.sum(entity_cv < 0.5)} dims")

    # Interesting: dims where entities are DIFFERENT but deltas are CONSISTENT
    delta_cv = np.std(deltas, axis=0) / (np.abs(np.mean(deltas, axis=0)) + 1e-20)
    interesting = (entity_cv > 1.0) & (delta_cv < 0.5)
    notes.observe(f"\n  Dims where entities differ (CV>1) but delta is consistent (CV<0.5): {np.sum(interesting)}")
    notes.observe(f"  These are the TRUE SHIFT dimensions — the relationship signal.")

    # And the reverse: dims where entities are similar but deltas vary
    boring = (entity_cv < 0.5) & (delta_cv > 2.0)
    notes.observe(f"  Dims where entities similar (CV<0.5) but delta varies (CV>2): {np.sum(boring)}")
    notes.observe(f"  These are noise dimensions — no relationship signal.")

    # Try a different PRESERVE criterion based on absolute delta vs noise floor
    notes.observe(f"\nAlternative PRESERVE: |mean_delta| < noise_floor")
    noise_floor = np.std(deltas.flatten()) * 0.1  # 10% of global noise
    notes.observe(f"  Global noise std: {np.std(deltas.flatten()):.6f}")
    notes.observe(f"  Noise floor (10%): {noise_floor:.6f}")
    notes.observe(f"  Dims with |mean_delta| < noise_floor: {np.sum(np.abs(np.mean(deltas, axis=0)) < noise_floor)}")


# =============================================================================
# INVESTIGATION 4: ANCHOR SYSTEM (F62 SIGN RULES)
# =============================================================================

def investigate_anchors(concepts, searcher, notes):
    """
    Load F62 sign rules from model layer analysis and see if they
    correlate with which dims are SHIFT/FLIP/PRESERVE in relationship gates.
    """
    notes.section("Anchor System — F62 Sign Rules Meet Relationship Gates")

    if not RULES_DIR.exists():
        notes.observe(f"SKIP: Phase4 rules not found at {RULES_DIR}")
        return

    # Load summary of layer rules
    summary_file = RULES_DIR / "summary.json"
    with open(summary_file) as f:
        layer_summary = json.load(f)

    notes.observe("F62 layer rules summary (28 layers):")
    notes.observe(f"  Available layers: {len(layer_summary)}")

    # Find layers with highest sign rule percentages
    sign_layers = []
    for row in layer_summary:
        sign_pct = sum(row.get(f'{t}_pct', 0) for t in ['sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate'])
        sign_layers.append((row['layer'], sign_pct, row.get('structured_pct', 0)))

    sign_layers.sort(key=lambda x: -x[1])
    rows = []
    for li, sp, struc in sign_layers[:10]:
        rows.append([str(li), f"{sp:.1%}", f"{struc:.1%}"])
    notes.data_table(
        ["Layer", "Sign%", "Struct%"],
        rows,
        "Top 10 layers by sign rule percentage"
    )

    # Load per-dim rules for a few interesting layers
    # Focus on: early (embedding->L1), mid (concept formation), late (output)
    interesting_layers = [1, 3, 13, 27]

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    valid = [(concepts[e], concepts[a]) for e, a in capital_train
             if e in concepts and a in concepts]

    # Relationship statistics
    deltas = np.array([a.embedding - e.embedding for e, a in valid])
    mean_delta = np.mean(deltas, axis=0)
    std_delta = np.std(deltas, axis=0)
    cv = std_delta / (np.abs(mean_delta) + 1e-20)

    entity_signs = np.array([e.phi_signs for e, _ in valid])
    answer_signs = np.array([a.phi_signs for _, a in valid])
    sign_products = entity_signs * answer_signs
    flip_rate = np.mean(sign_products < 0, axis=0)

    for layer_idx in interesting_layers:
        layer_file = RULES_DIR / f"layer_{layer_idx:02d}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        # Build per-dim rule map: global_dim -> rule_type
        dim_rule_map = {}
        for dr in layer_data.get("dim_rules", []):
            dim_rule_map[dr["global_dim"]] = dr["rule_type"]

        # For each rule type in this layer, check correlation with our
        # relationship gate classifications
        rule_types = Counter(dim_rule_map.values())

        notes.observe(f"\nLayer {layer_idx} ({layer_data.get('archetype', '?')}):")
        notes.observe(f"  Rule distribution: {dict(rule_types)}")

        # Compare: dims that F62 says are sign_flip vs dims where our
        # relationship shows sign flip
        f62_sign_flip_dims = set(d for d, r in dim_rule_map.items() if r == 'sign_flip')
        f62_sign_preserve_dims = set(d for d, r in dim_rule_map.items() if r == 'sign_preserve')
        f62_affine_dims = set(d for d, r in dim_rule_map.items() if r in ('identity', 'scale', 'affine'))

        rel_flip_dims = set(np.where(flip_rate >= 0.8)[0])
        rel_shift_dims = set(np.where(cv < 0.5)[0])

        if f62_sign_flip_dims:
            overlap_flip = len(f62_sign_flip_dims & rel_flip_dims)
            notes.observe(f"  F62 sign_flip: {len(f62_sign_flip_dims)} dims")
            notes.observe(f"  Relationship sign_flip≥0.8: {len(rel_flip_dims)} dims")
            notes.observe(f"  Overlap: {overlap_flip} dims")

        if f62_sign_preserve_dims:
            overlap_keep = len(f62_sign_preserve_dims & rel_shift_dims)
            notes.observe(f"  F62 sign_preserve: {len(f62_sign_preserve_dims)} dims")
            notes.observe(f"  Relationship low-CV: {len(rel_shift_dims)} dims")
            notes.observe(f"  Overlap (preserve→shift): {overlap_keep} dims")

        if f62_affine_dims:
            notes.observe(f"  F62 linear/affine: {len(f62_affine_dims)} dims")
            overlap_affine_shift = len(f62_affine_dims & rel_shift_dims)
            notes.observe(f"  Overlap with rel shift: {overlap_affine_shift} dims")

    # Aggregate analysis: use F62 rules as classification prior
    notes.observe(f"\n--- Using F62 as classification prior ---")

    # Load all layers and build a "consensus" dim classification
    dim_votes = {d: Counter() for d in range(3584)}
    for layer_idx in range(28):
        layer_file = RULES_DIR / f"layer_{layer_idx:02d}.json"
        if not layer_file.exists():
            continue
        with open(layer_file) as f:
            layer_data = json.load(f)
        for dr in layer_data.get("dim_rules", []):
            gd = dr["global_dim"]
            if gd < 3584:
                dim_votes[gd][dr["rule_type"]] += 1

    # For each dim, what's the most common rule across layers?
    consensus = {}
    for d in range(3584):
        if dim_votes[d]:
            consensus[d] = dim_votes[d].most_common(1)[0]
        else:
            consensus[d] = ("unstructured", 0)

    # Count consensus types
    consensus_types = Counter(rule for rule, _ in consensus.values())
    notes.observe(f"F62 consensus classification (most common rule across 28 layers):")
    for rule, count in consensus_types.most_common():
        notes.observe(f"  {rule:20s}: {count:5d} dims ({count/3584*100:.1f}%)")

    # Try using F62 consensus to guide gate application
    notes.observe(f"\n--- F62-guided gate: classify dims by model structure ---")

    # Identify dims that F62 says are "structural" (linear, sign-patterned)
    f62_structural = set()
    f62_sign_dims = set()
    for d in range(3584):
        rule, votes = consensus[d]
        if rule in ('identity', 'scale', 'affine', 'sign_preserve', 'sign_flip'):
            f62_structural.add(d)
        if rule in ('sign_preserve', 'sign_flip', 'sign_xor', 'sign_gate'):
            f62_sign_dims.add(d)

    notes.observe(f"  F62 structural dims: {len(f62_structural)}")
    notes.observe(f"  F62 sign-pattern dims: {len(f62_sign_dims)}")

    # Test: use F62 structural dims as the "confident" set
    capital_test = [
        ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
        ("Brazil", "Brasilia"),
    ]

    avg_delta = np.mean([a.embedding - e.embedding for e, a in valid], axis=0)

    va_ranks = []
    f62_hybrid_ranks = []
    f62_sign_ranks = []

    for entity_name, answer_name in capital_test:
        e = concepts.get(entity_name)
        a = concepts.get(answer_name)
        if not e or not a:
            continue
        ex = {e.token_id}

        # VA
        va_pred = e.embedding + avg_delta
        va_ranks.append(searcher.rank_of(va_pred, a.token_id, ex))

        # F62-guided: structural dims use per-dim shift, rest use VA
        f62_pred = va_pred.copy()
        struct_arr = np.array(sorted(f62_structural))
        if len(struct_arr) > 0:
            f62_pred[struct_arr] = e.embedding[struct_arr] + mean_delta[struct_arr]
        f62_hybrid_ranks.append(searcher.rank_of(f62_pred, a.token_id, ex))

        # F62 sign-guided: override signs on F62 sign dims
        f62_sign_pred = va_pred.copy()
        for d in f62_sign_dims:
            rule, _ = consensus[d]
            if rule == 'sign_flip':
                f62_sign_pred[d] = np.abs(f62_sign_pred[d]) * (-np.sign(e.embedding[d] + 1e-20))
            elif rule == 'sign_preserve':
                f62_sign_pred[d] = np.abs(f62_sign_pred[d]) * np.sign(e.embedding[d] + 1e-20)
        f62_sign_ranks.append(searcher.rank_of(f62_sign_pred, a.token_id, ex))

    notes.observe(f"\nF62-guided gate performance (capital test):")
    notes.observe(f"  VecAdd mean rank: {np.mean(va_ranks):.1f}")
    notes.observe(f"  F62 hybrid mean rank: {np.mean(f62_hybrid_ranks):.1f}")
    notes.observe(f"  F62 sign override mean rank: {np.mean(f62_sign_ranks):.1f}")

    return consensus


# =============================================================================
# BONUS: DIMENSION ANATOMY
# =============================================================================

def investigate_dim_anatomy(concepts, notes):
    """
    Deep look at individual dimensions — what do the most consistent
    dims actually encode?
    """
    notes.section("Dimension Anatomy — What Do Consistent Dims Encode?")

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    language_train = [
        ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
        ("China", "Chinese"), ("Spain", "Spanish"),
    ]
    gender_train = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
    ]

    def get_top_dims(name, pairs, n=20):
        valid = [(concepts[e], concepts[a]) for e, a in pairs
                 if e in concepts and a in concepts]
        if not valid:
            return None
        deltas = np.array([a.embedding - e.embedding for e, a in valid])
        mean_d = np.mean(deltas, axis=0)
        std_d = np.std(deltas, axis=0)
        cv = std_d / (np.abs(mean_d) + 1e-20)

        # Top N most consistent dims (lowest CV with non-trivial shift)
        # Filter out near-zero deltas
        mag = np.abs(mean_d)
        score = mag / (cv + 0.01)  # high magnitude, low variance
        top = np.argsort(-score)[:n]

        return {
            "name": name,
            "top_dims": top,
            "top_cv": cv[top],
            "top_shift": mean_d[top],
            "top_score": score[top],
        }

    cap_top = get_top_dims("capital", capital_train)
    lang_top = get_top_dims("language", language_train)
    gen_top = get_top_dims("gender", gender_train)

    # Show top dims for each relationship
    for analysis in [cap_top, lang_top, gen_top]:
        if analysis is None:
            continue
        notes.observe(f"\nTop 20 most informative dims for '{analysis['name']}':")
        rows = []
        for i in range(min(20, len(analysis["top_dims"]))):
            d = analysis["top_dims"][i]
            rows.append([
                str(d),
                f"{analysis['top_shift'][i]:.6f}",
                f"{analysis['top_cv'][i]:.3f}",
                f"{analysis['top_score'][i]:.3f}",
            ])
        notes.data_table(["Dim", "Shift", "CV", "Score"], rows)

    # Cross-relationship comparison: do the same dims appear?
    if cap_top and lang_top and gen_top:
        cap_set = set(cap_top["top_dims"][:20])
        lang_set = set(lang_top["top_dims"][:20])
        gen_set = set(gen_top["top_dims"][:20])

        notes.observe(f"\nTop-20 dim overlap:")
        notes.observe(f"  Capital ∩ Language: {len(cap_set & lang_set)} dims  {sorted(cap_set & lang_set)}")
        notes.observe(f"  Capital ∩ Gender:   {len(cap_set & gen_set)} dims  {sorted(cap_set & gen_set)}")
        notes.observe(f"  Language ∩ Gender:  {len(lang_set & gen_set)} dims  {sorted(lang_set & gen_set)}")
        notes.observe(f"  All three:          {len(cap_set & lang_set & gen_set)} dims")

        notes.finding(
            "If top dims overlap heavily → dims encode 'relationship-ness' generically. "
            "If no overlap → dims encode relationship-specific information."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 70)
    print("  TruthSpace Gate Exploration — Darwin's Notebook")
    print("  Systematic investigation of geometric gate structure")
    print("=" * 70)
    print()

    notes = FieldNotes()

    t0 = time.time()
    concepts, searcher, embeddings, all_signs, all_exponents = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    # Run all investigations
    investigate_selective_override(concepts, searcher, notes)
    cap_stats, lang_stats = investigate_gatec1_language(concepts, searcher, notes)
    investigate_preserve(concepts, notes)
    investigate_anchors(concepts, searcher, notes)
    investigate_dim_anatomy(concepts, notes)

    # Save field notes
    notes.save(SCRIPT_DIR / "field_notes.md")

    print(f"\nTotal exploration time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
