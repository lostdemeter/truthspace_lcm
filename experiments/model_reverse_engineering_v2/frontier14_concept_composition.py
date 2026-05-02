#!/usr/bin/env python3
"""
Frontier 14: Concept Composition in Embedding Space
=====================================================

Test: Can geometric shapes compose semantically?

If concepts have geometric positions (shapes), then:
    shape(A) + shape(B) ≈ shape(C) where C = "A-like B"

Examples:
    "dragon" + "shrimp" → "lobster"?  (large armored + small aquatic crustacean)
    "king" - "man" + "woman" → "queen"?  (classic word2vec)
    "ice" + "cream" → "dessert"?  (compound concept)
    "fire" + "fly" → "firefly"?  (literal composition)
    "book" + "worm" → "bookworm"?  (figurative composition)

Also test: relative concepts
    "lobster" is defined by its relationships to other concepts.
    Can we recover concept identity from relational position alone?

DC 289 §4, §6.3
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from phi_geometric.inference.phi_types import PhiEncoded

MODEL_DIR = os.path.join(os.path.dirname(__file__), "phi_model")


def load_embeddings():
    """Load the token embedding matrix."""
    phi = PhiEncoded.load(os.path.join(MODEL_DIR, 'embed_tokens.npz'))
    return phi.decode()  # (152064, 3584)


def load_tokenizer():
    """Load tokenizer vocabulary for name lookups."""
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
                    # Build id→token map
                    id_to_token = {}
                    for tok, idx in vocab.items():
                        id_to_token[idx] = tok
                    # Build token→id map (case insensitive for lookup)
                    token_to_id = {}
                    for tok, idx in vocab.items():
                        token_to_id[tok] = idx
                        token_to_id[tok.lower()] = idx
                    return id_to_token, token_to_id
    return None, None


def find_token_id(word, token_to_id):
    """Find the token ID for a word, trying various capitalizations."""
    candidates = [
        word,           # exact
        word.lower(),   # lowercase
        word.capitalize(),  # Capitalized
        word.upper(),   # UPPER
        f"Ġ{word}",    # with space prefix (BPE)
        f"Ġ{word.lower()}",
        f"Ġ{word.capitalize()}",
        f"▁{word}",    # sentencepiece prefix
        f"▁{word.lower()}",
        f"▁{word.capitalize()}",
    ]
    for c in candidates:
        if c in token_to_id:
            return token_to_id[c], c
    return None, None


def nearest_neighbors(vec, embeddings, id_to_token, top_k=20, exclude_ids=None):
    """Find nearest neighbors by cosine similarity."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-20)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / (emb_norms + 1e-20)
    sims = emb_normed @ vec_norm
    
    if exclude_ids:
        for eid in exclude_ids:
            sims[eid] = -999
    
    top_idx = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_idx:
        token = id_to_token.get(idx, f"tok_{idx}")
        results.append((idx, token, float(sims[idx])))
    return results


def main():
    print()
    print("=" * 80)
    print("  Frontier 14: Concept Composition in Embedding Space")
    print("  Can geometric shapes compose semantically?")
    print("=" * 80)
    print()
    
    print("  Loading embeddings...")
    embeddings = load_embeddings()
    print(f"  Shape: {embeddings.shape}")
    
    print("  Loading tokenizer...")
    id_to_token, token_to_id = load_tokenizer()
    if id_to_token is None:
        print("  ERROR: Could not load tokenizer")
        return
    print(f"  Vocabulary size: {len(id_to_token)}")
    print()
    
    # ================================================================
    # Part 1: Concept Addition
    # shape(A) + shape(B) → nearest neighbor?
    # ================================================================
    print("─" * 80)
    print("  Part 1: Concept Addition — shape(A) + shape(B) → ?")
    print("─" * 80)
    print()
    
    compositions = [
        ("dragon", "shrimp", "lobster", "large armored + small aquatic crustacean"),
        ("fire", "fly", "firefly", "literal compound"),
        ("book", "worm", "bookworm", "figurative compound"),
        ("snow", "man", "snowman", "literal compound"),
        ("sun", "flower", "sunflower", "literal compound"),
        ("water", "fall", "waterfall", "literal compound"),
        ("star", "fish", "starfish", "shape + animal"),
        ("sword", "fish", "swordfish", "shape + animal"),
        ("sea", "horse", "seahorse", "habitat + animal"),
        ("thunder", "storm", "thunderstorm", "compound"),
        ("rain", "bow", "rainbow", "compound"),
        ("foot", "ball", "football", "compound"),
    ]
    
    for word_a, word_b, expected, desc in compositions:
        id_a, tok_a = find_token_id(word_a, token_to_id)
        id_b, tok_b = find_token_id(word_b, token_to_id)
        id_exp, tok_exp = find_token_id(expected, token_to_id)
        
        if id_a is None or id_b is None:
            print(f"  SKIP: {word_a} + {word_b} (token not found)")
            continue
        
        vec_sum = embeddings[id_a] + embeddings[id_b]
        neighbors = nearest_neighbors(
            vec_sum, embeddings, id_to_token, top_k=15, 
            exclude_ids={id_a, id_b}
        )
        
        # Check if expected is in top-k
        found_rank = None
        if id_exp is not None:
            for rank, (nid, ntok, nsim) in enumerate(neighbors):
                if nid == id_exp:
                    found_rank = rank
                    break
        
        exp_str = f" (expected '{expected}' at rank {found_rank})" if found_rank is not None else \
                  f" (expected '{expected}' NOT in top 15)" if id_exp else \
                  f" (expected '{expected}' not in vocab)"
        
        print(f"  {tok_a} + {tok_b} → {desc}")
        print(f"    Top 5:{exp_str}")
        for rank, (nid, ntok, nsim) in enumerate(neighbors[:5]):
            marker = " ★" if nid == id_exp else ""
            print(f"      {rank}: {ntok!r:>25s}  cos={nsim:.4f}{marker}")
        print()
    
    # ================================================================
    # Part 2: Classic Analogies (A - B + C → D)
    # ================================================================
    print("─" * 80)
    print("  Part 2: Analogies — shape(A) - shape(B) + shape(C) → ?")
    print("─" * 80)
    print()
    
    analogies = [
        ("king", "man", "woman", "queen"),
        ("Paris", "France", "Germany", "Berlin"),
        ("Paris", "France", "Japan", "Tokyo"),
        ("big", "small", "hot", "cold"),
        ("cat", "kitten", "dog", "puppy"),
        ("good", "better", "bad", "worse"),
    ]
    
    for word_a, word_b, word_c, expected in analogies:
        id_a, tok_a = find_token_id(word_a, token_to_id)
        id_b, tok_b = find_token_id(word_b, token_to_id)
        id_c, tok_c = find_token_id(word_c, token_to_id)
        id_exp, tok_exp = find_token_id(expected, token_to_id)
        
        if any(x is None for x in [id_a, id_b, id_c]):
            print(f"  SKIP: {word_a} - {word_b} + {word_c} (token not found)")
            continue
        
        vec_analogy = embeddings[id_a] - embeddings[id_b] + embeddings[id_c]
        neighbors = nearest_neighbors(
            vec_analogy, embeddings, id_to_token, top_k=10,
            exclude_ids={id_a, id_b, id_c}
        )
        
        found_rank = None
        if id_exp is not None:
            for rank, (nid, ntok, nsim) in enumerate(neighbors):
                if nid == id_exp:
                    found_rank = rank
                    break
        
        exp_str = f" → rank {found_rank}" if found_rank is not None else " → NOT in top 10"
        print(f"  {tok_a} - {tok_b} + {tok_c} = ?  (expected: {expected}{exp_str})")
        for rank, (nid, ntok, nsim) in enumerate(neighbors[:5]):
            marker = " ★" if nid == id_exp else ""
            print(f"    {rank}: {ntok!r:>25s}  cos={nsim:.4f}{marker}")
        print()
    
    # ================================================================
    # Part 3: Relational Identity
    # Can we identify a concept from its relationships alone?
    # "lobster" = position relative to {dragon, shrimp, fish, crab, ...}
    # ================================================================
    print("─" * 80)
    print("  Part 3: Relational Identity")
    print("  Can we identify a concept from its distances to reference concepts?")
    print("─" * 80)
    print()
    
    # Reference concepts (the "coordinate system")
    reference_words = [
        "animal", "food", "water", "fire", "big", "small",
        "red", "blue", "fast", "slow", "hot", "cold",
        "fish", "bird", "dog", "cat", "tree", "rock",
        "king", "child", "sword", "shield", "land", "sea",
    ]
    
    ref_ids = []
    ref_tokens = []
    for w in reference_words:
        rid, rtok = find_token_id(w, token_to_id)
        if rid is not None:
            ref_ids.append(rid)
            ref_tokens.append(rtok)
    
    ref_vecs = embeddings[ref_ids]  # (N_ref, 3584)
    ref_norms = np.linalg.norm(ref_vecs, axis=1, keepdims=True)
    ref_normed = ref_vecs / (ref_norms + 1e-20)
    
    # Test: can we identify target concepts from relational fingerprint?
    targets = ["lobster", "eagle", "piano", "castle", "diamond", "volcano"]
    
    print(f"  Reference set: {len(ref_ids)} concepts")
    print(f"  Testing relational identity for: {targets}")
    print()
    
    # For each target, compute relational fingerprint (cosine to each reference)
    for target in targets:
        tid, ttok = find_token_id(target, token_to_id)
        if tid is None:
            print(f"  SKIP: {target} (not in vocab)")
            continue
        
        target_vec = embeddings[tid]
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-20)
        
        # Fingerprint: cosine similarity to each reference
        fingerprint = ref_normed @ target_norm  # (N_ref,)
        
        # Now find which token in the ENTIRE vocabulary has the most
        # similar fingerprint (without using the target directly)
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_normed = embeddings / (emb_norms + 1e-20)
        all_fingerprints = emb_normed @ ref_normed.T  # (152064, N_ref)
        
        # Compare each vocab token's fingerprint to the target's fingerprint
        fp_diffs = all_fingerprints - fingerprint[np.newaxis, :]  # (152064, N_ref)
        fp_distances = np.linalg.norm(fp_diffs, axis=1)  # (152064,)
        fp_distances[tid] = 999  # exclude self
        
        # Nearest by fingerprint
        top5 = np.argsort(fp_distances)[:5]
        
        print(f"  Target: {ttok} ({target})")
        print(f"    Top relational fingerprint: {', '.join(f'{ref_tokens[i]}={fingerprint[i]:.3f}' for i in np.argsort(np.abs(fingerprint))[-5:][::-1])}")
        print(f"    Nearest by relational identity:")
        for rank, idx in enumerate(top5):
            tok = id_to_token.get(idx, f"tok_{idx}")
            print(f"      {rank}: {tok!r:>25s}  fp_dist={fp_distances[idx]:.4f}")
        print()
    
    # ================================================================
    # Part 4: The Composition Operator
    # Is addition the right operator, or do we need something else?
    # ================================================================
    print("─" * 80)
    print("  Part 4: Composition Operators — Which Works Best?")
    print("  Test: addition, average, element-wise product, concat-project")
    print("─" * 80)
    print()
    
    test_pairs = [
        ("dragon", "shrimp", "lobster"),
        ("sun", "flower", "sunflower"),
        ("star", "fish", "starfish"),
        ("sea", "horse", "seahorse"),
        ("thunder", "storm", "thunderstorm"),
        ("rain", "bow", "rainbow"),
    ]
    
    operators = {
        "add": lambda a, b: a + b,
        "avg": lambda a, b: (a + b) / 2,
        "multiply": lambda a, b: a * b,
        "max": lambda a, b: np.maximum(a, b),
        "diff+b": lambda a, b: (a - b) + b * 2,  # emphasize b
    }
    
    print(f"  {'Pair':>25s}", end="")
    for op_name in operators:
        print(f"  {op_name:>8s}", end="")
    print()
    print("  " + "-" * 75)
    
    for word_a, word_b, expected in test_pairs:
        id_a, _ = find_token_id(word_a, token_to_id)
        id_b, _ = find_token_id(word_b, token_to_id)
        id_exp, _ = find_token_id(expected, token_to_id)
        
        if any(x is None for x in [id_a, id_b, id_exp]):
            continue
        
        print(f"  {word_a}+{word_b}→{expected}:", end="")
        
        for op_name, op_fn in operators.items():
            vec = op_fn(embeddings[id_a].astype(np.float64), 
                       embeddings[id_b].astype(np.float64))
            neighbors = nearest_neighbors(
                vec, embeddings, id_to_token, top_k=50,
                exclude_ids={id_a, id_b}
            )
            
            found_rank = None
            for rank, (nid, ntok, nsim) in enumerate(neighbors):
                if nid == id_exp:
                    found_rank = rank
                    break
            
            rank_str = f"r={found_rank}" if found_rank is not None else "r>50"
            print(f"  {rank_str:>8s}", end="")
        
        print()
    
    print()
    
    # ================================================================
    # Summary
    # ================================================================
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print("  Key questions answered:")
    print("  1. Do concepts compose by addition in embedding space?")
    print("  2. Do classic analogies work (king - man + woman = queen)?")
    print("  3. Can relational identity recover concept from fingerprint?")
    print("  4. Which composition operator works best?")
    print()
    print("  Note: These are RAW EMBEDDINGS (layer 0). The real shapes")
    print("  emerge after 28 layers of error-correcting convergence.")
    print("  If composition works even in raw embeddings, it should")
    print("  work BETTER in the converged representation.")
    print()


if __name__ == '__main__':
    main()
