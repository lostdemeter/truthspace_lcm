#!/usr/bin/env python3
"""
GPU-Accelerated Precache System v2
===================================

Fixed version that caches by ENTITY NAME, not raw token ID.

Key insight: Tokenization is context-dependent.
- "Poland" alone → [14658, 437] = ['Pol', 'and']
- " Poland" in context → [27602] = [' Poland']

So we cache by entity name string, not token ID.

Usage:
  python precache_gpu_v2.py --test     # Test on known entities
  python precache_gpu_v2.py --full     # Full vocabulary run

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import json
import os
import time
import argparse
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/thorin/truthspace-lcm/data/precache"
CACHE_FILE = os.path.join(CACHE_DIR, "entity_cache.json")
PATTERN_FILE = os.path.join(CACHE_DIR, "pattern_templates.json")
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "checkpoint_v2.json")


@dataclass
class CacheEntry:
    entity: str
    first_output_id: int
    first_output_text: str
    pattern_signature: str
    full_response: List[int]
    full_response_text: List[str]


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_index": -1, "entries": []}


def save_checkpoint(checkpoint: Dict):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def should_skip(text: str) -> bool:
    text = text.strip()
    if len(text) < 2:
        return True
    if text.isdigit():
        return True
    if all(c in '.,;:!?()[]{}"\'-_+=<>@#$%^&*~`|\\/' for c in text):
        return True
    if text[0] in '<>[]{}()':
        return True
    return False


def generate_for_entity(model, tokenizer, entity: str, device: str, n_tokens: int = 6) -> Optional[CacheEntry]:
    """Generate response for an entity name."""
    
    entity = entity.strip()
    if should_skip(entity):
        return None
    
    prompt = f"The capital of {entity} is"
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    except:
        return None
    
    output_tokens = []
    output_texts = []
    
    try:
        for _ in range(n_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token = outputs.logits[0, -1, :].argmax().item()
                output_tokens.append(next_token)
                output_texts.append(tokenizer.decode([next_token]))
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    except Exception as e:
        return None
    
    pattern_signature = "|".join(str(t) for t in output_tokens[1:])
    
    return CacheEntry(
        entity=entity,
        first_output_id=output_tokens[0],
        first_output_text=output_texts[0],
        pattern_signature=pattern_signature,
        full_response=output_tokens,
        full_response_text=output_texts,
    )


def get_entity_list(tokenizer) -> List[str]:
    """
    Get list of entity names to cache.
    
    Strategy: Decode each token and use it as potential entity.
    This covers single-token entities. Multi-token entities
    would need a separate list.
    """
    entities = []
    
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id]).strip()
        
        # Only keep reasonable entity names
        if len(text) >= 2 and text[0].isupper() and text.isalpha():
            entities.append(text)
    
    return entities


def run_precache(model, tokenizer, device: str, entities: List[str], 
                 checkpoint_interval: int = 100, verbose: bool = True):
    """Run precaching for a list of entities."""
    
    print(f"\n{'='*70}")
    print(f"Precaching {len(entities)} entities")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    checkpoint = load_checkpoint()
    cached_entries = [CacheEntry(**e) for e in checkpoint.get("entries", [])]
    last_idx = checkpoint.get("last_index", -1)
    
    if last_idx >= 0:
        print(f"Resuming from index {last_idx + 1}")
    
    start_time = time.time()
    processed = 0
    skipped = 0
    
    for i, entity in enumerate(entities):
        if i <= last_idx:
            continue
        
        entry = generate_for_entity(model, tokenizer, entity, device)
        
        if entry is None:
            skipped += 1
        else:
            cached_entries.append(entry)
            processed += 1
        
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i - last_idx) / elapsed if elapsed > 0 else 0
            remaining = (len(entities) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(entities)} | {processed} cached, {skipped} skipped | "
                  f"{rate:.1f}/s | ETA: {remaining/60:.1f} min")
        
        if (i + 1) % checkpoint_interval == 0:
            checkpoint = {"last_index": i, "entries": [asdict(e) for e in cached_entries]}
            save_checkpoint(checkpoint)
    
    # Final save
    checkpoint = {"last_index": len(entities) - 1, "entries": [asdict(e) for e in cached_entries]}
    save_checkpoint(checkpoint)
    
    # Cluster patterns
    pattern_counts = defaultdict(int)
    for entry in cached_entries:
        pattern_counts[entry.pattern_signature] += 1
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
    pattern_to_id = {p: i for i, (p, _) in enumerate(sorted_patterns)}
    
    print(f"\nFound {len(pattern_to_id)} distinct patterns")
    print("\nTop 10 patterns:")
    for pattern, count in sorted_patterns[:10]:
        try:
            tokens = [int(t) for t in pattern.split("|")]
            text = [tokenizer.decode([t]) for t in tokens]
            print(f"  {count:5d}x: {text}")
        except:
            pass
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Precaching complete!")
    print(f"  Entities: {len(entities)}")
    print(f"  Cached: {len(cached_entries)}")
    print(f"  Patterns: {len(pattern_to_id)}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")
    
    return cached_entries, pattern_to_id


def save_cache(entries: List[CacheEntry], pattern_to_id: Dict[str, int], tokenizer):
    """Save cache files."""
    
    # Entity → (first_token, pattern_id)
    cache = {}
    for entry in entries:
        pattern_id = pattern_to_id.get(entry.pattern_signature, -1)
        cache[entry.entity] = {
            "first": entry.first_output_id,
            "first_text": entry.first_output_text,
            "pattern": pattern_id,
        }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"Saved {len(cache)} entities to {CACHE_FILE}")
    
    # Pattern templates
    patterns = {}
    for entry in entries:
        pattern_id = pattern_to_id.get(entry.pattern_signature, -1)
        if pattern_id not in patterns:
            patterns[pattern_id] = {
                "tokens": entry.full_response[1:],
                "text": entry.full_response_text[1:],
                "count": 0,
            }
        patterns[pattern_id]["count"] += 1
    
    with open(PATTERN_FILE, 'w') as f:
        json.dump(patterns, f, indent=2)
    print(f"Saved {len(patterns)} patterns to {PATTERN_FILE}")


def test_cache(model, tokenizer, device: str):
    """Test the cache on known entities."""
    
    print(f"\n{'='*70}")
    print("Testing Cache")
    print(f"{'='*70}")
    
    if not os.path.exists(CACHE_FILE):
        print("No cache file found.")
        return
    
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    
    with open(PATTERN_FILE, 'r') as f:
        patterns = json.load(f)
    
    print(f"Loaded {len(cache)} entities, {len(patterns)} patterns")
    
    test_entities = ["France", "Germany", "Italy", "Spain", "Poland",
                     "Japan", "China", "Brazil", "Canada", "Australia"]
    
    for entity in test_entities:
        if entity not in cache:
            print(f"\n  {entity}: Not in cache")
            continue
        
        entry = cache[entity]
        pattern_id = str(entry["pattern"])
        
        if pattern_id not in patterns:
            print(f"\n  {entity}: Pattern not found")
            continue
        
        pattern = patterns[pattern_id]
        cached_response = [entry["first_text"]] + pattern["text"]
        
        # Get actual response
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        actual_tokens = []
        for _ in range(6):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token = outputs.logits[0, -1, :].argmax().item()
                actual_tokens.append(tokenizer.decode([next_token]))
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        
        match = sum(1 for a, c in zip(actual_tokens, cached_response) if a == c)
        
        print(f"\n  {entity}:")
        print(f"    Cached: {cached_response}")
        print(f"    Actual: {actual_tokens}")
        print(f"    Match:  {match}/6 {'✓' if match == 6 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test on known entities")
    parser.add_argument("--full", action="store_true", help="Full vocabulary run")
    parser.add_argument("--verify", action="store_true", help="Verify cache")
    args = parser.parse_args()
    
    ensure_cache_dir()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    if args.verify:
        test_cache(model, tokenizer, device)
        return
    
    if args.test:
        # Test on known country names
        entities = [
            "France", "Germany", "Italy", "Spain", "Poland", "Japan", "China",
            "Brazil", "Canada", "Australia", "Mexico", "Argentina", "Egypt",
            "India", "Russia", "Sweden", "Norway", "Denmark", "Finland",
            "Netherlands", "Belgium", "Austria", "Greece", "Portugal",
        ]
        entries, pattern_to_id = run_precache(model, tokenizer, device, entities)
        save_cache(entries, pattern_to_id, tokenizer)
        test_cache(model, tokenizer, device)
    
    elif args.full:
        # Get all capitalized words from vocabulary
        entities = get_entity_list(tokenizer)
        print(f"Found {len(entities)} potential entities in vocabulary")
        entries, pattern_to_id = run_precache(model, tokenizer, device, entities,
                                               checkpoint_interval=1000)
        save_cache(entries, pattern_to_id, tokenizer)
    
    else:
        print("Use --test or --full")


if __name__ == "__main__":
    main()
