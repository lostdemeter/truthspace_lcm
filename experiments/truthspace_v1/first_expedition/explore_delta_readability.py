#!/usr/bin/env python3
"""
TruthSpace Delta Readability & Anchoring Investigation
========================================================

The question: Can we READ relationship deltas — look at a delta and say
"this encodes the capital-of relationship" — rather than just using them
as statistical black boxes?

And if we can read them: can we find ANCHOR POINTS — verifiable truths
encoded in the geometry — that serve as fixed reference coordinates for
building a model whose foundation is exclusively provably true concepts?

This is the Gödel numbering idea: each concept gets a unique geometric
address, and the address itself encodes the concept's verifiable properties.

Part 1: Dimension Semantics — what do individual dims encode?
Part 2: Delta Direction Interpretation — what does a relationship direction mean?
Part 3: Anchor Discovery — find verifiable binary property separators
Part 4: Gödel Composition — concepts as intersections of verified properties
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2" / "phi_model"
sys.path.insert(0, str(PROJECT_ROOT))

PHI = (1 + np.sqrt(5)) / 2
D_MODEL = 3584

NOTES_PATH = SCRIPT_DIR / "delta_readability_notes.md"


# =============================================================================
# FIELD NOTES
# =============================================================================

class FieldNotes:
    """Darwin-style field notes — append-only, markdown formatted."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w")
        self.f.write("# TruthSpace Delta Readability — Field Notes\n")
        self.f.write("*Can we read relationship deltas and anchor concepts to truth?*\n\n")
        self.f.flush()

    def section(self, title):
        self.f.write(f"\n## {title}\n\n")
        self.f.flush()
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")

    def observe(self, text):
        self.f.write(f"{text}\n")
        self.f.flush()
        print(text)

    def data_table(self, headers, rows, title=""):
        if title:
            self.f.write(f"\n**{title}**\n\n")
            print(f"\n  {title}")
        fmt = " | ".join(f"{{:>{max(12, len(h))}}}" for h in headers)
        sep = " | ".join("-" * max(12, len(h)) for h in headers)
        hdr = fmt.format(*headers)
        self.f.write(f"| {hdr} |\n| {sep} |\n")
        print(f"  | {hdr} |")
        for row in rows:
            line = fmt.format(*[str(x) for x in row])
            self.f.write(f"| {line} |\n")
            print(f"  | {line} |")
        self.f.write("\n")
        self.f.flush()

    def finding(self, text):
        self.f.write(f"\n> **FINDING:** {text}\n\n")
        self.f.flush()
        print(f"\n  >>> FINDING: {text}\n")

    def close(self):
        self.f.close()


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Concept:
    name: str
    token_id: int
    token_str: str
    embedding: np.ndarray


class VocabSearcher:
    def __init__(self, embeddings, id_to_token):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb_normed = embeddings / (norms + 1e-20)
        self.embeddings = embeddings
        self.id_to_token = id_to_token
        self.n_vocab = embeddings.shape[0]

    def rank_of(self, composed, target_tid, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        return int(np.sum(sims > sims[target_tid]))

    def top_k(self, composed, k=5, exclude_tids=None):
        vec_norm = composed / (np.linalg.norm(composed) + 1e-20)
        sims = self.emb_normed @ vec_norm
        if exclude_tids:
            for eid in exclude_tids:
                sims[eid] = -999
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self.id_to_token.get(int(i), f"?{i}"), float(sims[i]))
                for i in top_idx]

    def tokens_by_dim_value(self, dim, k=20):
        """Find tokens with highest and lowest values on a given dimension."""
        vals = self.embeddings[:, dim]
        top_idx = np.argsort(vals)[-k:][::-1]
        bot_idx = np.argsort(vals)[:k]
        top = [(self.id_to_token.get(int(i), f"?{i}"), float(vals[i])) for i in top_idx]
        bot = [(self.id_to_token.get(int(i), f"?{i}"), float(vals[i])) for i in bot_idx]
        return top, bot

    def project_onto_direction(self, direction, k=20):
        """Project all tokens onto a direction, return most aligned/anti-aligned."""
        d_norm = direction / (np.linalg.norm(direction) + 1e-20)
        projections = self.embeddings @ d_norm
        top_idx = np.argsort(projections)[-k:][::-1]
        bot_idx = np.argsort(projections)[:k]
        top = [(self.id_to_token.get(int(i), f"?{i}"), float(projections[i])) for i in top_idx]
        bot = [(self.id_to_token.get(int(i), f"?{i}"), float(projections[i])) for i in bot_idx]
        return top, bot

    def classify_by_direction(self, direction, positive_tids, negative_tids):
        """Test how well a direction separates two sets of tokens."""
        d_norm = direction / (np.linalg.norm(direction) + 1e-20)
        projections = self.embeddings @ d_norm
        pos_projs = [projections[tid] for tid in positive_tids]
        neg_projs = [projections[tid] for tid in negative_tids]

        pos_mean = np.mean(pos_projs)
        neg_mean = np.mean(neg_projs)
        threshold = (pos_mean + neg_mean) / 2

        # Classification accuracy
        correct = 0
        total = len(positive_tids) + len(negative_tids)
        for p in pos_projs:
            if p > threshold:
                correct += 1
        for n in neg_projs:
            if n <= threshold:
                correct += 1

        # Separation margin
        if pos_mean > neg_mean:
            margin = min(pos_projs) - max(neg_projs)
        else:
            margin = min(neg_projs) - max(pos_projs)

        return {
            "accuracy": correct / total if total > 0 else 0,
            "pos_mean": float(pos_mean),
            "neg_mean": float(neg_mean),
            "threshold": float(threshold),
            "margin": float(margin),
            "separable": margin > 0,
        }


def load_embeddings_and_concepts():
    """Load output embeddings and build an extended concept set."""
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings...", flush=True)
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

    # Extended word list — countries, capitals, languages, genders,
    # plus semantic property words for anchor discovery
    all_words = [
        # Countries (European)
        "France", "Germany", "Japan", "China", "Egypt",
        "Australia", "Thailand", "Poland", "Norway", "Sweden",
        "India", "Brazil", "Korea", "Italy", "Portugal",
        "Russia", "Spain", "Greece", "Turkey", "Ireland",
        "Finland", "Denmark", "Austria", "Belgium", "Netherlands",
        "Switzerland", "Mexico", "Canada", "Argentina", "Chile",
        "Colombia", "Peru", "Nigeria", "Kenya", "Morocco",
        "Israel", "Iran", "Iraq", "Vietnam", "Indonesia",
        "Philippines", "Malaysia", "Singapore",
        # Capitals
        "Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
        "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
        "Delhi", "Brasilia", "Seoul", "Rome", "Lisbon",
        "Moscow", "Madrid", "Athens", "Ankara", "Dublin",
        "Helsinki", "Copenhagen", "Vienna", "Brussels", "Amsterdam",
        "Bern", "Mexico", "Ottawa", "Lima", "Santiago",
        "Bogota", "Lagos", "Nairobi", "Tehran", "Baghdad",
        "Hanoi", "Jakarta", "Manila", "Singapore",
        # Languages
        "French", "German", "Japanese", "Chinese", "Spanish",
        "Italian", "Portuguese", "Russian", "Arabic", "English",
        "Korean", "Thai", "Polish", "Norwegian", "Swedish",
        "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
        "Danish", "Irish", "Persian", "Hebrew", "Vietnamese",
        "Indonesian", "Filipino", "Malay",
        # Gender pairs
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "brother", "sister", "son", "daughter",
        "husband", "wife", "uncle", "aunt", "prince", "princess",
        "actor", "actress", "waiter", "waitress", "hero", "heroine",
        # Continents / regions (for geographic anchoring)
        "Europe", "Asia", "Africa", "America", "Oceania",
        "European", "Asian", "African", "American",
        # Language families (for linguistic anchoring)
        "Romance", "Germanic", "Slavic", "Semitic",
        # Semantic properties
        "democratic", "republic", "monarchy", "island", "continental",
        "tropical", "arctic", "desert", "coastal", "landlocked",
        # Compound concepts (from F158)
        "dragon", "shrimp", "lobster",
        "sun", "flower", "sunflower",
        "rain", "bow", "rainbow",
        "foot", "ball", "football",
    ]

    concepts = {}
    for word in all_words:
        tid, tok_str = find_token_id(word)
        if tid is not None and word not in concepts:  # avoid dupes
            concepts[word] = Concept(
                name=word, token_id=tid, token_str=tok_str,
                embedding=embeddings[tid],
            )

    searcher = VocabSearcher(embeddings, id_to_token)
    print(f"  {len(concepts)} concepts, {searcher.n_vocab} vocab tokens", flush=True)
    return concepts, searcher, embeddings, token_to_id, id_to_token


# =============================================================================
# RELATIONSHIP DEFINITIONS
# =============================================================================

def define_relationships(concepts):
    """Define all relationships with train/test splits."""

    capital_train = [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("China", "Beijing"), ("Egypt", "Cairo"),
    ]
    capital_test = [
        ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Poland", "Warsaw"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("India", "Delhi"),
        ("Brazil", "Brasilia"), ("Korea", "Seoul"),
    ]

    language_train = [
        ("France", "French"), ("Germany", "German"), ("Japan", "Japanese"),
        ("China", "Chinese"), ("Spain", "Spanish"),
    ]
    language_test = [
        ("Italy", "Italian"), ("Portugal", "Portuguese"), ("Russia", "Russian"),
    ]

    gender_train = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
    ]
    gender_test = [
        ("brother", "sister"), ("son", "daughter"), ("husband", "wife"),
    ]

    def valid_pairs(pair_list):
        return [(e, a) for e, a in pair_list
                if e in concepts and a in concepts]

    return {
        "capital": {"train": valid_pairs(capital_train), "test": valid_pairs(capital_test)},
        "language": {"train": valid_pairs(language_train), "test": valid_pairs(language_test)},
        "gender": {"train": valid_pairs(gender_train), "test": valid_pairs(gender_test)},
    }


def compute_delta(concepts, pairs):
    """Compute mean delta and per-pair deltas for a relationship."""
    deltas = np.array([
        concepts[a].embedding - concepts[e].embedding
        for e, a in pairs
    ])
    mean_delta = np.mean(deltas, axis=0)
    std_delta = np.std(deltas, axis=0)
    cv = std_delta / (np.abs(mean_delta) + 1e-20)
    return mean_delta, deltas, cv


# =============================================================================
# PART 1: DIMENSION SEMANTICS
# =============================================================================

def part1_dimension_semantics(concepts, searcher, relationships, notes):
    """For each relationship's top informative dims, find what tokens
    have extreme values on those dims. This tells us what each dim encodes."""

    notes.section("1. Dimension Semantics — What Do Individual Dims Encode?")

    for rel_name, rel_data in relationships.items():
        train_pairs = rel_data["train"]
        mean_delta, deltas, cv = compute_delta(concepts, train_pairs)

        # Score = |mean_shift| / CV (signal-to-noise per dim)
        abs_mean = np.abs(mean_delta)
        score = abs_mean / (cv + 1e-10)

        top_dims = np.argsort(score)[-10:][::-1]

        notes.observe(f"\n### Relationship: {rel_name}")
        notes.observe(f"Top 10 most informative dims (by |shift|/CV):\n")

        for rank, dim in enumerate(top_dims):
            notes.observe(f"**Dim {dim}** — shift={mean_delta[dim]:.6f}, "
                         f"CV={cv[dim]:.3f}, score={score[dim]:.4f}")

            # What tokens have extreme values on this dimension?
            top_tokens, bot_tokens = searcher.tokens_by_dim_value(dim, k=15)

            # Clean token display
            def clean(tok):
                return tok.replace("Ġ", " ").replace("▁", " ").strip()

            top_str = ", ".join(f"{clean(t)}({v:.4f})" for t, v in top_tokens[:8])
            bot_str = ", ".join(f"{clean(t)}({v:.4f})" for t, v in bot_tokens[:8])

            notes.observe(f"  HIGH: {top_str}")
            notes.observe(f"  LOW:  {bot_str}")

            # Check where our relationship concepts sit on this dim
            entity_vals = [concepts[e].embedding[dim] for e, _ in train_pairs if e in concepts]
            answer_vals = [concepts[a].embedding[dim] for _, a in train_pairs if a in concepts]

            notes.observe(f"  Entities: mean={np.mean(entity_vals):.4f}, "
                         f"Answers: mean={np.mean(answer_vals):.4f}, "
                         f"Δ={np.mean(answer_vals)-np.mean(entity_vals):.4f}")
            notes.observe("")

        # Summary: are top dims semantically interpretable?
        notes.observe(f"---")


# =============================================================================
# PART 2: DELTA DIRECTION INTERPRETATION
# =============================================================================

def part2_delta_direction(concepts, searcher, relationships, notes):
    """Project all 152K embeddings onto each relationship's delta direction.
    What's 'most capital-like' in the vocabulary?"""

    notes.section("2. Delta Direction Interpretation — What Does Each Delta Mean?")

    for rel_name, rel_data in relationships.items():
        train_pairs = rel_data["train"]
        mean_delta, deltas, cv = compute_delta(concepts, train_pairs)

        notes.observe(f"\n### Relationship: {rel_name}")
        notes.observe(f"Delta norm: {np.linalg.norm(mean_delta):.4f}")

        # Project all vocab onto delta direction
        top_aligned, bot_aligned = searcher.project_onto_direction(mean_delta, k=25)

        def clean(tok):
            return tok.replace("Ġ", " ").replace("▁", " ").strip()

        notes.observe(f"\n**Most aligned with {rel_name} delta direction** (these tokens point where the delta goes):")
        for i, (tok, proj) in enumerate(top_aligned[:15]):
            notes.observe(f"  {i+1:2d}. {clean(tok):20s}  proj={proj:.4f}")

        notes.observe(f"\n**Most anti-aligned with {rel_name} delta direction** (opposite of where delta goes):")
        for i, (tok, proj) in enumerate(bot_aligned[:15]):
            notes.observe(f"  {i+1:2d}. {clean(tok):20s}  proj={proj:.4f}")

        # SVD of the deltas to find relationship subspace
        U, S, Vt = np.linalg.svd(deltas, full_matrices=False)
        notes.observe(f"\n**Delta SVD spectrum** (how many directions the relationship uses):")
        total_var = np.sum(S**2)
        cum_var = 0
        for i, s in enumerate(S):
            cum_var += s**2
            notes.observe(f"  Dir {i}: σ={s:.4f}, var={s**2/total_var*100:.1f}%, "
                         f"cumvar={cum_var/total_var*100:.1f}%")

        # Cosine between mean_delta and SVD direction 1
        cos_with_svd1 = abs(np.dot(mean_delta / np.linalg.norm(mean_delta),
                                    Vt[0] / np.linalg.norm(Vt[0])))
        notes.observe(f"  cos(mean_delta, SVD_dir_1) = {cos_with_svd1:.4f}")

        # Cross-relationship: how similar are the delta directions?
        notes.observe("")

    # Cross-relationship comparison
    notes.observe("\n### Cross-Relationship Delta Comparison")
    delta_dirs = {}
    for rel_name, rel_data in relationships.items():
        mean_delta, _, _ = compute_delta(concepts, rel_data["train"])
        delta_dirs[rel_name] = mean_delta / (np.linalg.norm(mean_delta) + 1e-20)

    names = list(delta_dirs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cos = float(np.dot(delta_dirs[names[i]], delta_dirs[names[j]]))
            notes.observe(f"  cos({names[i]}, {names[j]}) = {cos:.4f}")

    notes.finding("If cross-relationship cosines are near 0, each relationship has "
                  "its own unique direction in ℝ³⁵⁸⁴ — they're orthogonal transforms, "
                  "not variations of a common 'relationship axis'.")


# =============================================================================
# PART 3: ANCHOR DISCOVERY
# =============================================================================

def part3_anchor_discovery(concepts, searcher, relationships, notes):
    """Find verifiable binary properties in the geometry.
    An anchor is a direction in embedding space that reliably separates
    concepts by a verifiable truth."""

    notes.section("3. Anchor Discovery — Verifiable Properties in the Geometry")

    # Define verifiable binary properties with positive and negative examples
    anchors = {
        "is_european_country": {
            "positive": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Greece", "Ireland",
                         "Finland", "Denmark", "Austria", "Belgium", "Netherlands",
                         "Switzerland", "Russia"],
            "negative": ["Japan", "China", "Egypt", "Australia", "Thailand",
                         "India", "Brazil", "Korea", "Turkey", "Nigeria",
                         "Kenya", "Morocco", "Israel", "Iran", "Vietnam",
                         "Indonesia", "Philippines", "Mexico", "Canada",
                         "Argentina", "Chile", "Colombia", "Peru"],
        },
        "is_asian_country": {
            "positive": ["Japan", "China", "Thailand", "India", "Korea",
                         "Vietnam", "Indonesia", "Philippines", "Malaysia",
                         "Singapore", "Iran", "Iraq", "Israel"],
            "negative": ["France", "Germany", "Poland", "Norway", "Sweden",
                         "Italy", "Portugal", "Spain", "Egypt", "Australia",
                         "Brazil", "Nigeria", "Kenya", "Morocco",
                         "Mexico", "Canada", "Argentina"],
        },
        "is_capital_city": {
            "positive": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                         "Canberra", "Bangkok", "Warsaw", "Oslo", "Stockholm",
                         "Delhi", "Seoul", "Rome", "Lisbon", "Moscow",
                         "Madrid", "Athens", "Ankara", "Dublin", "Helsinki",
                         "Copenhagen", "Vienna", "Brussels", "Amsterdam",
                         "Ottawa", "Lima", "Tehran", "Baghdad", "Hanoi"],
            "negative": ["France", "Germany", "Japan", "China", "Egypt",
                         "Australia", "Thailand", "Poland", "Norway", "Sweden",
                         "India", "Korea", "Italy", "Portugal", "Russia",
                         "Spain", "Greece", "Turkey", "Ireland",
                         "Brazil", "Mexico", "Canada", "Argentina"],
        },
        "is_romance_language": {
            "positive": ["French", "Italian", "Portuguese", "Spanish"],
            "negative": ["German", "Japanese", "Chinese", "Arabic", "English",
                         "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                         "Dutch", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Danish", "Persian", "Vietnamese"],
        },
        "is_germanic_language": {
            "positive": ["German", "English", "Dutch", "Norwegian", "Swedish",
                         "Danish"],
            "negative": ["French", "Italian", "Portuguese", "Spanish",
                         "Japanese", "Chinese", "Arabic", "Korean",
                         "Polish", "Greek", "Turkish", "Hindi", "Finnish",
                         "Russian", "Thai", "Persian", "Vietnamese"],
        },
        "is_female_gendered": {
            "positive": ["queen", "woman", "girl", "mother", "sister",
                         "daughter", "wife", "aunt", "princess", "actress",
                         "waitress", "heroine"],
            "negative": ["king", "man", "boy", "father", "brother",
                         "son", "husband", "uncle", "prince", "actor",
                         "waiter", "hero"],
        },
    }

    anchor_directions = {}

    for anchor_name, anchor_def in anchors.items():
        notes.observe(f"\n### Anchor: {anchor_name}")

        # Get token IDs for positive and negative examples
        pos_tids = []
        pos_names = []
        for word in anchor_def["positive"]:
            if word in concepts:
                pos_tids.append(concepts[word].token_id)
                pos_names.append(word)

        neg_tids = []
        neg_names = []
        for word in anchor_def["negative"]:
            if word in concepts:
                neg_tids.append(concepts[word].token_id)
                neg_names.append(word)

        notes.observe(f"  Positive examples: {len(pos_tids)} ({', '.join(pos_names[:8])}{'...' if len(pos_names) > 8 else ''})")
        notes.observe(f"  Negative examples: {len(neg_tids)} ({', '.join(neg_names[:8])}{'...' if len(neg_names) > 8 else ''})")

        if len(pos_tids) < 2 or len(neg_tids) < 2:
            notes.observe(f"  SKIP: not enough examples")
            continue

        # Compute anchor direction: mean(positive embeddings) - mean(negative embeddings)
        pos_embs = np.array([searcher.embeddings[tid] for tid in pos_tids])
        neg_embs = np.array([searcher.embeddings[tid] for tid in neg_tids])
        anchor_dir = np.mean(pos_embs, axis=0) - np.mean(neg_embs, axis=0)
        anchor_dir_norm = anchor_dir / (np.linalg.norm(anchor_dir) + 1e-20)

        # Test classification accuracy
        result = searcher.classify_by_direction(anchor_dir, pos_tids, neg_tids)
        notes.observe(f"  **Classification accuracy: {result['accuracy']*100:.1f}%**")
        notes.observe(f"  Positive mean projection: {result['pos_mean']:.4f}")
        notes.observe(f"  Negative mean projection: {result['neg_mean']:.4f}")
        notes.observe(f"  Margin: {result['margin']:.4f} ({'SEPARABLE' if result['separable'] else 'OVERLAPPING'})")

        # Leave-one-out cross-validation
        loo_correct = 0
        loo_total = 0
        for i in range(len(pos_tids)):
            # Train without this example
            train_pos = [t for j, t in enumerate(pos_tids) if j != i]
            train_neg = neg_tids
            loo_dir = (np.mean([searcher.embeddings[t] for t in train_pos], axis=0) -
                       np.mean([searcher.embeddings[t] for t in train_neg], axis=0))
            proj = np.dot(searcher.embeddings[pos_tids[i]], loo_dir / (np.linalg.norm(loo_dir) + 1e-20))
            threshold = (np.mean([np.dot(searcher.embeddings[t], loo_dir / (np.linalg.norm(loo_dir) + 1e-20)) for t in train_pos]) +
                        np.mean([np.dot(searcher.embeddings[t], loo_dir / (np.linalg.norm(loo_dir) + 1e-20)) for t in train_neg])) / 2
            if proj > threshold:
                loo_correct += 1
            loo_total += 1

        for i in range(len(neg_tids)):
            train_neg = [t for j, t in enumerate(neg_tids) if j != i]
            train_pos = pos_tids
            loo_dir = (np.mean([searcher.embeddings[t] for t in train_pos], axis=0) -
                       np.mean([searcher.embeddings[t] for t in train_neg], axis=0))
            proj = np.dot(searcher.embeddings[neg_tids[i]], loo_dir / (np.linalg.norm(loo_dir) + 1e-20))
            threshold = (np.mean([np.dot(searcher.embeddings[t], loo_dir / (np.linalg.norm(loo_dir) + 1e-20)) for t in train_pos]) +
                        np.mean([np.dot(searcher.embeddings[t], loo_dir / (np.linalg.norm(loo_dir) + 1e-20)) for t in train_neg])) / 2
            if proj <= threshold:
                loo_correct += 1
            loo_total += 1

        loo_acc = loo_correct / loo_total if loo_total > 0 else 0
        notes.observe(f"  **LOO cross-validation: {loo_acc*100:.1f}%** ({loo_correct}/{loo_total})")

        # What does the vocabulary look like projected onto this anchor?
        top_aligned, bot_aligned = searcher.project_onto_direction(anchor_dir, k=15)

        def clean(tok):
            return tok.replace("Ġ", " ").replace("▁", " ").strip()

        notes.observe(f"\n  Top-10 vocab most aligned with '{anchor_name}':")
        for i, (tok, proj) in enumerate(top_aligned[:10]):
            notes.observe(f"    {i+1:2d}. {clean(tok):20s}  proj={proj:.4f}")

        notes.observe(f"  Top-10 vocab most anti-aligned:")
        for i, (tok, proj) in enumerate(bot_aligned[:10]):
            notes.observe(f"    {i+1:2d}. {clean(tok):20s}  proj={proj:.4f}")

        # Store anchor direction for Part 4
        anchor_directions[anchor_name] = {
            "direction": anchor_dir_norm,
            "accuracy": result["accuracy"],
            "loo_accuracy": loo_acc,
            "separable": result["separable"],
            "pos_tids": pos_tids,
            "neg_tids": neg_tids,
        }

    # Cross-anchor comparison
    notes.observe("\n### Cross-Anchor Orthogonality")
    names = list(anchor_directions.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            cos = float(np.dot(anchor_directions[names[i]]["direction"],
                               anchor_directions[names[j]]["direction"]))
            notes.observe(f"  cos({names[i]}, {names[j]}) = {cos:.4f}")

    notes.finding("Anchors with high LOO accuracy AND mutual orthogonality are "
                  "independent verifiable truths — candidate coordinate axes for TruthSpace.")

    # How do anchor directions relate to relationship deltas?
    notes.observe("\n### Anchor-Delta Alignment")
    for rel_name, rel_data in relationships.items():
        mean_delta, _, _ = compute_delta(concepts, rel_data["train"])
        delta_norm = mean_delta / (np.linalg.norm(mean_delta) + 1e-20)
        for anch_name, anch_data in anchor_directions.items():
            cos = float(np.dot(delta_norm, anch_data["direction"]))
            if abs(cos) > 0.1:
                notes.observe(f"  {rel_name} ↔ {anch_name}: cos = {cos:.4f}")

    return anchor_directions


# =============================================================================
# PART 4: GÖDEL COMPOSITION
# =============================================================================

def part4_godel_composition(concepts, searcher, anchor_directions, relationships, notes):
    """Test: can we identify concepts by their anchor coordinates?
    If France = (is_european=YES, is_asian=NO, ...), and these coordinates
    uniquely identify France, we have a Gödel-like addressing scheme."""

    notes.section("4. Gödel Composition — Concepts as Anchor Coordinate Vectors")

    # Only use anchors with reasonable accuracy
    good_anchors = {name: data for name, data in anchor_directions.items()
                    if data["loo_accuracy"] >= 0.7}

    if not good_anchors:
        notes.observe("No anchors with LOO accuracy >= 70%. Cannot proceed with composition.")
        return

    anchor_names = sorted(good_anchors.keys())
    notes.observe(f"Using {len(anchor_names)} anchors with LOO >= 70%:")
    for name in anchor_names:
        notes.observe(f"  {name}: LOO={good_anchors[name]['loo_accuracy']*100:.1f}%")

    # Compute anchor coordinates for every concept
    notes.observe(f"\n### Anchor Coordinates for Known Concepts")

    # Compute thresholds for each anchor
    thresholds = {}
    for name, data in good_anchors.items():
        pos_projs = [np.dot(searcher.embeddings[tid], data["direction"])
                     for tid in data["pos_tids"]]
        neg_projs = [np.dot(searcher.embeddings[tid], data["direction"])
                     for tid in data["neg_tids"]]
        thresholds[name] = (np.mean(pos_projs) + np.mean(neg_projs)) / 2

    # Test concepts — countries, capitals, languages, gender words
    test_groups = {
        "Countries": ["France", "Germany", "Japan", "China", "Egypt",
                       "Australia", "India", "Brazil", "Korea", "Italy",
                       "Spain", "Russia", "Poland", "Norway", "Sweden",
                       "Turkey", "Greece", "Ireland", "Finland", "Denmark",
                       "Mexico", "Canada", "Argentina", "Nigeria", "Kenya"],
        "Capitals": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                      "Canberra", "Delhi", "Seoul", "Rome", "Lisbon",
                      "Moscow", "Madrid", "Athens", "Ankara", "Dublin",
                      "Helsinki", "Copenhagen", "Vienna", "Warsaw", "Oslo",
                      "Stockholm", "Ottawa", "Lima"],
        "Languages": ["French", "German", "Japanese", "Chinese", "Spanish",
                       "Italian", "Portuguese", "Russian", "Arabic", "English",
                       "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                       "Dutch", "Greek", "Turkish", "Hindi", "Finnish"],
        "Gender (M)": ["king", "man", "boy", "father", "brother", "son",
                        "husband", "uncle", "prince", "actor"],
        "Gender (F)": ["queen", "woman", "girl", "mother", "sister",
                        "daughter", "wife", "aunt", "princess", "actress"],
    }

    for group_name, words in test_groups.items():
        notes.observe(f"\n**{group_name}:**")
        header = ["Concept"] + [a.replace("is_", "").replace("_", " ")[:12] for a in anchor_names]
        rows = []
        for word in words:
            if word not in concepts:
                continue
            emb = concepts[word].embedding
            coords = []
            for aname in anchor_names:
                proj = np.dot(emb, good_anchors[aname]["direction"])
                label = "+" if proj > thresholds[aname] else "-"
                coords.append(label)
            rows.append([word] + coords)

        if rows:
            notes.data_table(header, rows, f"{group_name} anchor coordinates")

    # UNIQUENESS TEST: do anchor coordinates uniquely identify concepts?
    notes.observe("\n### Uniqueness Test — Do Coordinates Form Unique Addresses?")

    all_concepts_tested = []
    for group_words in test_groups.values():
        all_concepts_tested.extend(group_words)

    coord_to_concepts = defaultdict(list)
    for word in all_concepts_tested:
        if word not in concepts:
            continue
        emb = concepts[word].embedding
        coord_tuple = tuple(
            "+" if np.dot(emb, good_anchors[aname]["direction"]) > thresholds[aname]
            else "-"
            for aname in anchor_names
        )
        coord_to_concepts[coord_tuple].append(word)

    n_unique = sum(1 for v in coord_to_concepts.values() if len(v) == 1)
    n_total = sum(len(v) for v in coord_to_concepts.values())
    n_addresses = len(coord_to_concepts)

    notes.observe(f"  Total concepts tested: {n_total}")
    notes.observe(f"  Unique addresses: {n_addresses}")
    notes.observe(f"  Concepts with unique address: {n_unique}/{n_total} ({n_unique/n_total*100:.1f}%)")

    # Show collisions (same address, different concepts)
    collisions = {k: v for k, v in coord_to_concepts.items() if len(v) > 1}
    if collisions:
        notes.observe(f"\n  Collisions ({len(collisions)} addresses shared by multiple concepts):")
        for coord, words in sorted(collisions.items(), key=lambda x: -len(x[1])):
            coord_str = "".join(coord)
            notes.observe(f"    {coord_str}: {', '.join(words)}")

    notes.finding(f"With {len(anchor_names)} anchors, {n_unique/n_total*100:.1f}% of concepts "
                  f"have unique addresses. Each additional verified anchor doubles the "
                  f"address space. Need ~log2(N_concepts) anchors for full uniqueness.")

    # COMPOSITION TEST: can we predict a concept's anchor coordinates?
    notes.observe("\n### Composition Test — Predicting Coordinates from Relationships")
    notes.observe("If France→Paris via capital-of, and we know France's coordinates,")
    notes.observe("can we predict Paris's coordinates?\n")

    for rel_name, rel_data in [("capital", relationships["capital"]),
                                ("language", relationships["language"]),
                                ("gender", relationships["gender"])]:
        test_pairs = rel_data["test"]
        train_pairs = rel_data["train"]
        mean_delta, _, _ = compute_delta(concepts, train_pairs)

        notes.observe(f"\n**{rel_name} relationship:**")

        correct_coords = 0
        total_coords = 0
        for entity_name, answer_name in test_pairs:
            if entity_name not in concepts or answer_name not in concepts:
                continue

            # Predict answer embedding via VA
            predicted = concepts[entity_name].embedding + mean_delta

            # Compute predicted vs actual anchor coordinates
            actual_emb = concepts[answer_name].embedding
            match_count = 0
            for aname in anchor_names:
                pred_proj = np.dot(predicted, good_anchors[aname]["direction"])
                actual_proj = np.dot(actual_emb, good_anchors[aname]["direction"])
                pred_label = "+" if pred_proj > thresholds[aname] else "-"
                actual_label = "+" if actual_proj > thresholds[aname] else "-"
                if pred_label == actual_label:
                    match_count += 1
                    correct_coords += 1
                total_coords += 1

            notes.observe(f"  {entity_name} → {answer_name}: "
                         f"{match_count}/{len(anchor_names)} coordinates match")

        if total_coords > 0:
            notes.observe(f"  Overall: {correct_coords}/{total_coords} "
                         f"({correct_coords/total_coords*100:.1f}%) coordinate predictions correct")

    notes.finding("If VA preserves anchor coordinates across relationships, then "
                  "the anchor coordinate system IS compatible with relationship deltas — "
                  "we can reason about both simultaneously. This is the foundation "
                  "for verifiable concept composition.")


# =============================================================================
# PART 5: RECONSTRUCTION / RESIDUAL TEST
# =============================================================================

def part5_reconstruction_residual(concepts, searcher, anchor_directions, notes):
    """Test: are concepts purely compounds of platonic ideals?

    Project each concept embedding onto all anchor directions, reconstruct
    from those projections, and examine what's left (the residual).

    If residuals are small/unstructured → concepts ARE platonic compounds.
    If residuals are large/structured → there's geometry beyond the truth axes."""

    notes.section("5. Reconstruction / Residual Test — Are Concepts Platonic Compounds?")

    # Use all anchors with reasonable accuracy
    good_anchors = {name: data for name, data in anchor_directions.items()
                    if data["loo_accuracy"] >= 0.7}

    if not good_anchors:
        notes.observe("No anchors with LOO accuracy >= 70%. Cannot proceed.")
        return

    anchor_names = sorted(good_anchors.keys())
    n_anchors = len(anchor_names)
    notes.observe(f"Using {n_anchors} anchor directions (LOO >= 70%):")
    for name in anchor_names:
        notes.observe(f"  {name}: LOO={good_anchors[name]['loo_accuracy']*100:.1f}%")

    # Build the anchor basis matrix: shape (n_anchors, D_MODEL)
    # Each row is a normalized anchor direction
    anchor_matrix = np.array([good_anchors[name]["direction"] for name in anchor_names])

    # Gram matrix to check how orthogonal the basis is
    gram = anchor_matrix @ anchor_matrix.T
    notes.observe(f"\n### Anchor Basis Properties")
    notes.observe(f"  Anchor basis shape: {anchor_matrix.shape}")
    notes.observe(f"  Gram matrix diagonal (should be 1.0): {np.diag(gram)}")
    off_diag = gram[np.triu_indices(n_anchors, k=1)]
    notes.observe(f"  Off-diagonal |cos| — mean: {np.mean(np.abs(off_diag)):.4f}, "
                  f"max: {np.max(np.abs(off_diag)):.4f}")
    notes.observe(f"  Anchors are {'nearly orthogonal' if np.max(np.abs(off_diag)) < 0.3 else 'NOT orthogonal'}")

    # Use pseudo-inverse for projection (handles non-orthogonal bases correctly)
    # projection coefficients c = (A A^T)^{-1} A x
    # reconstructed = A^T c = A^T (A A^T)^{-1} A x
    gram_inv = np.linalg.inv(gram)
    proj_operator = gram_inv @ anchor_matrix  # shape (n_anchors, D_MODEL)

    # Collect all test concepts
    test_groups = {
        "Countries": ["France", "Germany", "Japan", "China", "Egypt",
                       "Australia", "India", "Brazil", "Korea", "Italy",
                       "Spain", "Russia", "Poland", "Norway", "Sweden",
                       "Turkey", "Greece", "Ireland", "Finland", "Denmark",
                       "Mexico", "Canada", "Argentina", "Nigeria", "Kenya"],
        "Capitals": ["Paris", "Berlin", "Tokyo", "Beijing", "Cairo",
                      "Canberra", "Delhi", "Seoul", "Rome", "Lisbon",
                      "Moscow", "Madrid", "Athens", "Ankara", "Dublin",
                      "Helsinki", "Copenhagen", "Vienna", "Warsaw", "Oslo",
                      "Stockholm", "Ottawa", "Lima"],
        "Languages": ["French", "German", "Japanese", "Chinese", "Spanish",
                       "Italian", "Portuguese", "Russian", "Arabic", "English",
                       "Korean", "Thai", "Polish", "Norwegian", "Swedish",
                       "Dutch", "Greek", "Turkish", "Hindi", "Finnish"],
        "Gender (M)": ["king", "man", "boy", "father", "brother", "son",
                        "husband", "uncle", "prince", "actor"],
        "Gender (F)": ["queen", "woman", "girl", "mother", "sister",
                        "daughter", "wife", "aunt", "princess", "actress"],
    }

    notes.observe(f"\n### Per-Concept Reconstruction")

    all_residual_norms = []
    all_original_norms = []
    all_reconstruction_ratios = []
    all_residuals = []  # store actual residual vectors for structure analysis
    all_concept_names = []
    group_stats = {}

    for group_name, words in test_groups.items():
        notes.observe(f"\n**{group_name}:**")
        header = ["Concept", "||emb||", "||recon||", "||resid||", "ratio", "% explained"]
        rows = []
        g_residual_norms = []
        g_ratios = []

        for word in words:
            if word not in concepts:
                continue

            emb = concepts[word].embedding  # shape (D_MODEL,)
            emb_norm = np.linalg.norm(emb)

            # Project: coefficients along each anchor
            coeffs = proj_operator @ emb  # shape (n_anchors,)

            # Reconstruct from anchor projections
            reconstructed = anchor_matrix.T @ coeffs  # shape (D_MODEL,)
            recon_norm = np.linalg.norm(reconstructed)

            # Residual
            residual = emb - reconstructed
            resid_norm = np.linalg.norm(residual)

            # Variance explained
            ratio = resid_norm / (emb_norm + 1e-20)
            pct_explained = (1.0 - (resid_norm**2 / (emb_norm**2 + 1e-20))) * 100

            rows.append([word,
                         f"{emb_norm:.3f}",
                         f"{recon_norm:.3f}",
                         f"{resid_norm:.3f}",
                         f"{ratio:.4f}",
                         f"{pct_explained:.2f}%"])

            all_residual_norms.append(resid_norm)
            all_original_norms.append(emb_norm)
            all_reconstruction_ratios.append(ratio)
            all_residuals.append(residual)
            all_concept_names.append(word)
            g_residual_norms.append(resid_norm)
            g_ratios.append(ratio)

        if rows:
            notes.data_table(header, rows, f"{group_name} reconstruction")
            mean_ratio = np.mean(g_ratios)
            mean_pct = (1.0 - np.mean(np.array(g_residual_norms)**2 /
                        (np.array([np.linalg.norm(concepts[w].embedding)
                         for w in words if w in concepts])**2 + 1e-20))) * 100
            notes.observe(f"  Group mean ||resid||/||emb||: {mean_ratio:.4f}")
            group_stats[group_name] = {"mean_ratio": mean_ratio, "mean_pct": mean_pct}

    # === AGGREGATE STATISTICS ===
    notes.observe(f"\n### Aggregate Reconstruction Statistics")
    all_residual_norms = np.array(all_residual_norms)
    all_original_norms = np.array(all_original_norms)
    all_ratios = np.array(all_reconstruction_ratios)

    pct_explained_all = (1.0 - (all_residual_norms**2 / (all_original_norms**2 + 1e-20))) * 100

    notes.observe(f"  Total concepts analyzed: {len(all_ratios)}")
    notes.observe(f"  Embedding dimension: {D_MODEL}")
    notes.observe(f"  Number of anchor axes: {n_anchors}")
    notes.observe(f"  Theoretical max variance explained by {n_anchors} axes: "
                  f"{n_anchors/D_MODEL*100:.3f}% (if random directions)")
    notes.observe(f"")
    notes.observe(f"  ||residual|| / ||embedding||:")
    notes.observe(f"    Mean:   {np.mean(all_ratios):.4f}")
    notes.observe(f"    Median: {np.median(all_ratios):.4f}")
    notes.observe(f"    Std:    {np.std(all_ratios):.4f}")
    notes.observe(f"    Min:    {np.min(all_ratios):.4f}")
    notes.observe(f"    Max:    {np.max(all_ratios):.4f}")
    notes.observe(f"")
    notes.observe(f"  Variance explained (% of ||emb||^2):")
    notes.observe(f"    Mean:   {np.mean(pct_explained_all):.2f}%")
    notes.observe(f"    Median: {np.median(pct_explained_all):.2f}%")
    notes.observe(f"    Min:    {np.min(pct_explained_all):.2f}%")
    notes.observe(f"    Max:    {np.max(pct_explained_all):.2f}%")

    # === RESIDUAL STRUCTURE ANALYSIS ===
    notes.observe(f"\n### Residual Structure Analysis")
    notes.observe("If residuals are random noise, they should:")
    notes.observe("  1. Have no consistent direction (low pairwise cosine)")
    notes.observe("  2. Not cluster by concept type")
    notes.observe("  3. Have low rank (spread across many dimensions)\n")

    residual_matrix = np.array(all_residuals)  # shape (n_concepts, D_MODEL)

    # 1. Pairwise cosine similarity of residuals
    resid_norms_col = np.linalg.norm(residual_matrix, axis=1, keepdims=True)
    resid_normed = residual_matrix / (resid_norms_col + 1e-20)
    cos_sim = resid_normed @ resid_normed.T

    # Overall pairwise stats (upper triangle, excluding diagonal)
    n_concepts = len(all_residuals)
    triu_idx = np.triu_indices(n_concepts, k=1)
    pairwise_cos = cos_sim[triu_idx]
    notes.observe(f"  Pairwise cosine similarity of residuals:")
    notes.observe(f"    Mean: {np.mean(pairwise_cos):.4f}")
    notes.observe(f"    Std:  {np.std(pairwise_cos):.4f}")
    notes.observe(f"    |cos| mean: {np.mean(np.abs(pairwise_cos)):.4f}")
    notes.observe(f"    Max:  {np.max(pairwise_cos):.4f}")
    notes.observe(f"    Min:  {np.min(pairwise_cos):.4f}")

    # 2. Within-group vs between-group cosine similarity
    notes.observe(f"\n  Within-group vs between-group residual similarity:")
    concept_to_group = {}
    group_indices = {}
    idx = 0
    for group_name, words in test_groups.items():
        group_indices[group_name] = []
        for word in words:
            if word in concepts:
                concept_to_group[all_concept_names[idx]] = group_name
                group_indices[group_name].append(idx)
                idx += 1

    for group_name, indices in group_indices.items():
        if len(indices) < 2:
            continue
        # Within-group
        within_cos = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_cos.append(cos_sim[indices[i], indices[j]])
        within_mean = np.mean(within_cos) if within_cos else 0

        # Between-group (this group vs all others)
        other_indices = [i for g, idxs in group_indices.items()
                         if g != group_name for i in idxs]
        between_cos = []
        for i in indices:
            for j in other_indices:
                between_cos.append(cos_sim[i, j])
        between_mean = np.mean(between_cos) if between_cos else 0

        notes.observe(f"    {group_name:15s}: within={within_mean:.4f}, "
                      f"between={between_mean:.4f}, "
                      f"gap={within_mean - between_mean:.4f}")

    # 3. SVD of residual matrix — is there low-rank structure?
    notes.observe(f"\n  SVD of residual matrix ({residual_matrix.shape}):")
    # Center residuals before SVD
    residual_centered = residual_matrix - np.mean(residual_matrix, axis=0)
    U, S, Vt = np.linalg.svd(residual_centered, full_matrices=False)

    # How many components needed to explain 50%, 80%, 90%, 95%?
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / (total_var + 1e-20)
    notes.observe(f"    Total singular values: {len(S)}")
    notes.observe(f"    Top-10 singular values: {S[:10]}")
    notes.observe(f"    Top-1 explains: {S[0]**2/total_var*100:.2f}% of residual variance")
    notes.observe(f"    Top-3 explain:  {cumvar[2]*100:.2f}%")
    notes.observe(f"    Top-5 explain:  {cumvar[4]*100:.2f}%")
    notes.observe(f"    Top-10 explain: {cumvar[min(9, len(cumvar)-1)]*100:.2f}%")

    for threshold in [0.5, 0.8, 0.9, 0.95]:
        n_needed = int(np.searchsorted(cumvar, threshold)) + 1
        notes.observe(f"    Components for {threshold*100:.0f}% variance: {n_needed}")

    # 4. Are the top residual dimensions interpretable?
    notes.observe(f"\n  Top residual principal directions:")
    for k in range(min(3, len(S))):
        direction = Vt[k]  # shape (D_MODEL,)
        # Project all concepts onto this residual direction
        projs = residual_matrix @ direction
        # Find extremes
        top_idx = np.argsort(projs)[-5:]
        bot_idx = np.argsort(projs)[:5]
        notes.observe(f"\n    PC{k+1} (explains {S[k]**2/total_var*100:.1f}%):")
        notes.observe(f"      Most positive: {', '.join(f'{all_concept_names[i]}({projs[i]:.3f})' for i in reversed(top_idx))}")
        notes.observe(f"      Most negative: {', '.join(f'{all_concept_names[i]}({projs[i]:.3f})' for i in bot_idx)}")

    # === COMPARISON TO RANDOM BASELINE ===
    notes.observe(f"\n### Random Baseline Comparison")
    notes.observe("How does reconstruction with truth axes compare to random directions?\n")

    rng = np.random.RandomState(42)
    n_trials = 20
    random_pct_explained = []
    for trial in range(n_trials):
        # Generate random orthonormal basis of same size
        random_dirs = rng.randn(n_anchors, D_MODEL)
        # Orthonormalize via QR
        Q, _ = np.linalg.qr(random_dirs.T)
        random_basis = Q[:, :n_anchors].T  # shape (n_anchors, D_MODEL)

        random_gram = random_basis @ random_basis.T
        random_gram_inv = np.linalg.inv(random_gram)
        random_proj_op = random_gram_inv @ random_basis

        trial_explained = []
        for word in all_concept_names:
            emb = concepts[word].embedding
            coeffs = random_proj_op @ emb
            recon = random_basis.T @ coeffs
            resid = emb - recon
            pct = (1.0 - (np.linalg.norm(resid)**2 / (np.linalg.norm(emb)**2 + 1e-20))) * 100
            trial_explained.append(pct)
        random_pct_explained.append(np.mean(trial_explained))

    random_mean = np.mean(random_pct_explained)
    random_std = np.std(random_pct_explained)
    actual_mean = np.mean(pct_explained_all)

    notes.observe(f"  Truth axes ({n_anchors} dims): {actual_mean:.2f}% variance explained")
    notes.observe(f"  Random axes ({n_anchors} dims): {random_mean:.2f}% ± {random_std:.2f}% (mean ± std over {n_trials} trials)")
    notes.observe(f"  Ratio (truth/random): {actual_mean / (random_mean + 1e-20):.2f}x")

    if actual_mean > random_mean + 3 * random_std:
        notes.finding(f"Truth axes explain {actual_mean:.2f}% of concept variance vs "
                      f"{random_mean:.2f}% for random — {actual_mean/random_mean:.1f}x more! "
                      f"The truth axes capture MEANINGFUL structure, not just random subspace.")
    elif actual_mean > random_mean + random_std:
        notes.finding(f"Truth axes explain somewhat more ({actual_mean:.2f}%) than random "
                      f"({random_mean:.2f}%), suggesting partial platonic structure.")
    else:
        notes.finding(f"Truth axes ({actual_mean:.2f}%) do not explain significantly more "
                      f"than random ({random_mean:.2f}%). Concepts are NOT purely platonic compounds — "
                      f"there is substantial structure beyond the current truth axes.")

    # === FINAL VERDICT ===
    notes.observe(f"\n### Verdict")
    notes.observe(f"  With {n_anchors} truth axes spanning {n_anchors}/{D_MODEL} dimensions:")
    notes.observe(f"  - {actual_mean:.2f}% of concept variance is explained by truth axes")
    notes.observe(f"  - {100 - actual_mean:.2f}% lives in the residual (orthogonal to truth axes)")

    if actual_mean > 50:
        notes.finding("Concepts are PREDOMINANTLY platonic compounds. "
                      "The truth axes capture the majority of concept structure. "
                      "The residual likely encodes fine-grained distinctions.")
    elif actual_mean > 10:
        notes.finding(f"Truth axes capture a significant fraction ({actual_mean:.1f}%) of concept "
                      f"structure given they span only {n_anchors} of {D_MODEL} dimensions. "
                      f"Concepts have a platonic component PLUS additional structure. "
                      f"More truth axes would likely increase coverage.")
    else:
        notes.finding(f"Truth axes capture only {actual_mean:.1f}% — concepts are much richer "
                      f"than their platonic coordinates. Either we need many more truth axes, "
                      f"or the residual encodes fundamentally different structure.")

    return {
        "n_anchors": n_anchors,
        "mean_pct_explained": actual_mean,
        "random_pct_explained": random_mean,
        "residual_matrix": residual_matrix,
        "singular_values": S,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  TruthSpace Delta Readability & Anchoring Investigation")
    print("=" * 70)
    print()

    notes = FieldNotes(NOTES_PATH)
    t_start = time.time()

    # Load everything
    concepts, searcher, embeddings, token_to_id, id_to_token = load_embeddings_and_concepts()
    relationships = define_relationships(concepts)

    for rel_name, rel_data in relationships.items():
        notes.observe(f"  {rel_name}: {len(rel_data['train'])} train, {len(rel_data['test'])} test pairs")

    # Run all five parts
    part1_dimension_semantics(concepts, searcher, relationships, notes)
    part2_delta_direction(concepts, searcher, relationships, notes)
    anchor_directions = part3_anchor_discovery(concepts, searcher, relationships, notes)
    part4_godel_composition(concepts, searcher, anchor_directions, relationships, notes)
    part5_reconstruction_residual(concepts, searcher, anchor_directions, notes)

    # Final synthesis
    notes.section("6. Synthesis — The State of Delta Readability")
    elapsed = time.time() - t_start
    notes.observe(f"Investigation completed in {elapsed:.1f}s\n")

    notes.observe("### What We Now Know")
    notes.observe("")
    notes.observe("Part 1 tells us what individual dimensions encode — whether they")
    notes.observe("carry semantic content (geographic, linguistic, gendered) or noise.")
    notes.observe("")
    notes.observe("Part 2 tells us what relationship delta directions mean — whether")
    notes.observe("the delta direction itself separates source from target concepts.")
    notes.observe("")
    notes.observe("Part 3 tells us which binary properties are geometrically verifiable —")
    notes.observe("these are candidate TruthSpace anchors.")
    notes.observe("")
    notes.observe("Part 4 tells us whether anchor coordinates uniquely address concepts")
    notes.observe("and whether relationships preserve coordinates — the Gödel test.")

    notes.close()
    print(f"\nField notes written to {NOTES_PATH}")


if __name__ == "__main__":
    main()
