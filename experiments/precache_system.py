#!/usr/bin/env python3
"""
Precache System: Build Entity → (First Token, Pattern Type) Lookup Table
=========================================================================

This script precaches all vocabulary tokens to enable 100% geometric generation.

For each token in the vocabulary:
1. Run forward pass: "The capital of [token] is"
2. Generate 6 tokens
3. Store: token_id → (first_output_token, pattern_type, full_response)

The pattern_type is determined by clustering the response patterns.

Usage:
  # Test mode (first 100 tokens)
  python precache_system.py --test
  
  # Full run (all 152K tokens) - run overnight
  python precache_system.py --full
  
  # Resume from checkpoint
  python precache_system.py --resume

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import numpy as np
import json
import os
import time
import argparse
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949

# Output directory
CACHE_DIR = "/home/thorin/truthspace-lcm/data/precache"
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "checkpoint.json")
CACHE_FILE = os.path.join(CACHE_DIR, "entity_cache.json")
PATTERN_FILE = os.path.join(CACHE_DIR, "pattern_templates.json")


@dataclass
class CacheEntry:
    """Single cache entry for an entity."""
    token_id: int
    token_text: str
    first_output_id: int
    first_output_text: str
    pattern_signature: str  # Tokens 1-5 as string for clustering
    full_response: List[int]
    full_response_text: List[str]


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_checkpoint() -> Dict:
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_token_id": -1, "entries": [], "patterns": {}}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def generate_response(model, tokenizer, token_id: int, n_tokens: int = 6) -> Optional[CacheEntry]:
    """Generate response for a single token."""
    
    # Get token text
    token_text = tokenizer.decode([token_id])
    
    # Skip special tokens and very short tokens
    if token_text.strip() == "" or len(token_text.strip()) < 2:
        return None
    
    # Skip tokens that are clearly not entities (punctuation, numbers, etc.)
    if token_text.strip().isdigit():
        return None
    if all(c in '.,;:!?()[]{}"\'-_+=<>@#$%^&*~`|\\/' for c in token_text.strip()):
        return None
    
    # Build prompt
    prompt = f"The capital of{token_text} is"
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    except:
        return None
    
    # Generate tokens
    output_tokens = []
    output_texts = []
    
    try:
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token = outputs.logits[0, -1, :].argmax().item()
                output_tokens.append(next_token)
                output_texts.append(tokenizer.decode([next_token]))
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    except Exception as e:
        print(f"  Error generating for token {token_id} ({token_text!r}): {e}")
        return None
    
    # Create pattern signature (tokens 1-5)
    pattern_signature = "|".join(str(t) for t in output_tokens[1:])
    
    return CacheEntry(
        token_id=token_id,
        token_text=token_text,
        first_output_id=output_tokens[0],
        first_output_text=output_texts[0],
        pattern_signature=pattern_signature,
        full_response=output_tokens,
        full_response_text=output_texts,
    )


def cluster_patterns(entries: List[CacheEntry]) -> Dict[str, int]:
    """Cluster patterns and assign pattern IDs."""
    
    # Count pattern occurrences
    pattern_counts = defaultdict(int)
    for entry in entries:
        pattern_counts[entry.pattern_signature] += 1
    
    # Sort by frequency
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
    
    # Assign IDs (most common = 0)
    pattern_to_id = {}
    for i, (pattern, count) in enumerate(sorted_patterns):
        pattern_to_id[pattern] = i
    
    return pattern_to_id


def run_precache(model, tokenizer, start_id: int = 0, end_id: int = None, 
                 checkpoint_interval: int = 100, verbose: bool = True):
    """Run precaching for a range of token IDs."""
    
    vocab_size = tokenizer.vocab_size
    if end_id is None:
        end_id = vocab_size
    
    print(f"\n{'='*70}")
    print(f"Precaching tokens {start_id} to {end_id} (of {vocab_size})")
    print(f"{'='*70}")
    
    # Load existing checkpoint
    checkpoint = load_checkpoint()
    entries = [CacheEntry(**e) for e in checkpoint.get("entries", [])]
    last_id = checkpoint.get("last_token_id", -1)
    
    if last_id >= start_id:
        print(f"Resuming from token {last_id + 1}")
        start_id = last_id + 1
    
    # Track progress
    start_time = time.time()
    processed = 0
    skipped = 0
    errors = 0
    
    for token_id in range(start_id, end_id):
        # Generate response
        entry = generate_response(model, tokenizer, token_id)
        
        if entry is None:
            skipped += 1
        else:
            entries.append(entry)
            processed += 1
        
        # Progress update
        if verbose and (token_id - start_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (token_id - start_id + 1) / elapsed
            remaining = (end_id - token_id - 1) / rate if rate > 0 else 0
            print(f"  Token {token_id}/{end_id} | {processed} cached, {skipped} skipped | "
                  f"{rate:.1f} tok/s | ETA: {remaining/60:.1f} min")
        
        # Checkpoint
        if (token_id - start_id + 1) % checkpoint_interval == 0:
            checkpoint = {
                "last_token_id": token_id,
                "entries": [asdict(e) for e in entries],
            }
            save_checkpoint(checkpoint)
            if verbose:
                print(f"  [Checkpoint saved at token {token_id}]")
    
    # Final save
    checkpoint = {
        "last_token_id": end_id - 1,
        "entries": [asdict(e) for e in entries],
    }
    save_checkpoint(checkpoint)
    
    # Cluster patterns
    print(f"\nClustering {len(entries)} patterns...")
    pattern_to_id = cluster_patterns(entries)
    
    print(f"Found {len(pattern_to_id)} distinct patterns")
    
    # Show top patterns
    pattern_counts = defaultdict(int)
    for entry in entries:
        pattern_counts[entry.pattern_signature] += 1
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 patterns:")
    for pattern, count in sorted_patterns:
        # Decode pattern
        try:
            tokens = [int(t) for t in pattern.split("|")]
            text = [tokenizer.decode([t]) for t in tokens]
            print(f"  {count:5d}x: {text}")
        except:
            print(f"  {count:5d}x: {pattern}")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Precaching complete!")
    print(f"  Tokens processed: {end_id - start_id}")
    print(f"  Entries cached: {len(entries)}")
    print(f"  Skipped: {skipped}")
    print(f"  Distinct patterns: {len(pattern_to_id)}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {(end_id - start_id) / elapsed:.1f} tokens/second")
    print(f"{'='*70}")
    
    return entries, pattern_to_id


def save_final_cache(entries: List[CacheEntry], pattern_to_id: Dict[str, int], tokenizer):
    """Save final cache files."""
    
    # Build compact cache: token_id → (first_output_id, pattern_id)
    cache = {}
    for entry in entries:
        pattern_id = pattern_to_id.get(entry.pattern_signature, -1)
        cache[entry.token_id] = {
            "first": entry.first_output_id,
            "pattern": pattern_id,
            "text": entry.token_text,
            "first_text": entry.first_output_text,
        }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    
    print(f"Saved cache to {CACHE_FILE}")
    
    # Build pattern templates
    # For each pattern, store the token sequence
    patterns = {}
    for entry in entries:
        pattern_id = pattern_to_id.get(entry.pattern_signature, -1)
        if pattern_id not in patterns:
            patterns[pattern_id] = {
                "tokens": entry.full_response[1:],  # Skip first token
                "text": entry.full_response_text[1:],
                "count": 0,
            }
        patterns[pattern_id]["count"] += 1
    
    with open(PATTERN_FILE, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"Saved {len(patterns)} pattern templates to {PATTERN_FILE}")


def test_cache(model, tokenizer, n_test: int = 10):
    """Test the cache on a few examples."""
    
    print(f"\n{'='*70}")
    print("Testing Cache")
    print(f"{'='*70}")
    
    # Load cache
    if not os.path.exists(CACHE_FILE):
        print("No cache file found. Run precaching first.")
        return
    
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    
    with open(PATTERN_FILE, 'r') as f:
        patterns = json.load(f)
    
    print(f"Loaded cache with {len(cache)} entries and {len(patterns)} patterns")
    
    # Test on some entities
    test_entities = ["France", "Germany", "Italy", "Spain", "Poland", 
                     "Japan", "China", "Brazil", "Canada", "Australia"]
    
    for entity in test_entities[:n_test]:
        # Get token ID
        tokens = tokenizer.encode(entity, add_special_tokens=False)
        if not tokens:
            print(f"  {entity}: No token found")
            continue
        
        token_id = str(tokens[0])
        
        if token_id not in cache:
            print(f"  {entity}: Not in cache")
            continue
        
        entry = cache[token_id]
        pattern_id = str(entry["pattern"])
        
        if pattern_id not in patterns:
            print(f"  {entity}: Pattern {pattern_id} not found")
            continue
        
        pattern = patterns[pattern_id]
        
        # Reconstruct response
        first_token = entry["first_text"]
        pattern_tokens = pattern["text"]
        
        full_response = [first_token] + pattern_tokens
        
        # Compare with actual
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        actual_tokens = []
        for i in range(6):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token = outputs.logits[0, -1, :].argmax().item()
                actual_tokens.append(tokenizer.decode([next_token]))
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        match = sum(1 for a, c in zip(actual_tokens, full_response) if a == c)
        
        print(f"\n  {entity}:")
        print(f"    Cached:  {full_response}")
        print(f"    Actual:  {actual_tokens}")
        print(f"    Match:   {match}/6")


def main():
    parser = argparse.ArgumentParser(description="Precache entity responses")
    parser.add_argument("--test", action="store_true", help="Test mode (first 100 tokens)")
    parser.add_argument("--full", action="store_true", help="Full run (all tokens)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--verify", action="store_true", help="Verify cache with test")
    parser.add_argument("--start", type=int, default=0, help="Start token ID")
    parser.add_argument("--end", type=int, default=None, help="End token ID")
    args = parser.parse_args()
    
    ensure_cache_dir()
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    if args.verify:
        test_cache(model, tokenizer)
        return
    
    if args.test:
        # Test mode: first 1000 tokens
        entries, pattern_to_id = run_precache(model, tokenizer, 
                                               start_id=0, end_id=1000,
                                               checkpoint_interval=100)
        save_final_cache(entries, pattern_to_id, tokenizer)
        test_cache(model, tokenizer)
    
    elif args.full:
        # Full run
        entries, pattern_to_id = run_precache(model, tokenizer,
                                               start_id=args.start,
                                               end_id=args.end,
                                               checkpoint_interval=1000)
        save_final_cache(entries, pattern_to_id, tokenizer)
    
    elif args.resume:
        # Resume from checkpoint
        checkpoint = load_checkpoint()
        last_id = checkpoint.get("last_token_id", -1)
        print(f"Resuming from token {last_id + 1}")
        
        entries, pattern_to_id = run_precache(model, tokenizer,
                                               start_id=last_id + 1,
                                               end_id=args.end,
                                               checkpoint_interval=1000)
        save_final_cache(entries, pattern_to_id, tokenizer)
    
    else:
        # Default: test mode
        print("No mode specified. Use --test, --full, or --resume")
        print("Running test mode (first 1000 tokens)...")
        entries, pattern_to_id = run_precache(model, tokenizer,
                                               start_id=0, end_id=1000,
                                               checkpoint_interval=100)
        save_final_cache(entries, pattern_to_id, tokenizer)
        test_cache(model, tokenizer)


if __name__ == "__main__":
    main()
