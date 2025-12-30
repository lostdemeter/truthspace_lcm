#!/usr/bin/env python3
"""
Signal Corpus Curator

Cleans up the signal corpus by:
1. Removing garbage/nonsensical Qwen2 rewrites
2. Keeping only clean, well-structured sentences
3. Expanding with more high-quality concepts

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


# Words that indicate a bad/nonsensical rewrite
BAD_INDICATORS = {
    'demolishes', 'stubbornly', 'resists', 'confines', 'correlates',
    'poisonous', 'collisions', 'gravity', 'neutrons', 'pressures',
    'formalizing', 'interstellar', 'biochemistry', 'habitat',
    'overlaps with itself', 'goes beyond', 'pertains to both',
    'stars beyond our solar', 'type of individual',
}

# Patterns that indicate good structure
GOOD_PATTERNS = [
    r'seems to be a \w+ (who|that|known for)',
    r'is a \w+ (who|that|which)',
    r'provides \w+, \w+, and \w+',
    r'known for \w+ing',
    r'involves \w+ing',
]

# Minimum quality thresholds
MIN_WORDS = 5
MAX_WORDS = 30


def score_frame(text: str, agent: str) -> Tuple[float, List[str]]:
    """
    Score a signal frame for quality.
    
    Returns (score, reasons) where higher score = better quality.
    """
    score = 0.0
    reasons = []
    
    text_lower = text.lower()
    words = text.split()
    
    # Length check
    if len(words) < MIN_WORDS:
        score -= 5
        reasons.append("too short")
    elif len(words) > MAX_WORDS:
        score -= 2
        reasons.append("too long")
    else:
        score += 1
    
    # Check for bad indicators
    for bad in BAD_INDICATORS:
        if bad in text_lower:
            score -= 10
            reasons.append(f"bad word: {bad}")
    
    # Check for good patterns
    for pattern in GOOD_PATTERNS:
        if re.search(pattern, text_lower):
            score += 3
            reasons.append(f"good pattern")
            break
    
    # Check if agent name appears (should be in the sentence)
    if agent.lower() in text_lower:
        score += 2
        reasons.append("has agent")
    else:
        score -= 1
        reasons.append("missing agent")
    
    # Check for proper sentence structure
    if text[0].isupper() and text.rstrip().endswith('.'):
        score += 1
        reasons.append("proper sentence")
    
    # Bonus for clean role patterns
    role_patterns = ['is a detective', 'is a doctor', 'is a scientist', 
                    'is a science', 'is a study', 'is a concept',
                    'is a character', 'is a field']
    for rp in role_patterns:
        if rp in text_lower:
            score += 2
            reasons.append(f"has role: {rp}")
            break
    
    return score, reasons


def curate_corpus(input_path: str, output_path: str, min_score: float = 0.0):
    """Curate the signal corpus, keeping only good frames."""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    print(f"Input: {len(frames)} frames")
    
    good_frames = []
    bad_frames = []
    
    for frame in frames:
        text = frame.get('text', '')
        agent = frame.get('agent', '')
        
        score, reasons = score_frame(text, agent)
        
        if score >= min_score:
            good_frames.append(frame)
        else:
            bad_frames.append({
                'frame': frame,
                'score': score,
                'reasons': reasons,
            })
    
    print(f"Good frames: {len(good_frames)}")
    print(f"Bad frames: {len(bad_frames)}")
    
    # Show some bad frames for inspection
    print("\nSample bad frames:")
    for bf in bad_frames[:5]:
        print(f"  [{bf['score']:.1f}] {bf['frame'].get('agent', '?')}: {bf['frame'].get('text', '')[:60]}...")
        print(f"       Reasons: {', '.join(bf['reasons'])}")
    
    # Save curated corpus
    with open(output_path, 'w') as f:
        json.dump({'frames': good_frames}, f, indent=2)
    
    print(f"\nSaved {len(good_frames)} curated frames to {output_path}")
    
    return good_frames, bad_frames


def expand_corpus(corpus_path: str, num_new: int = 200):
    """Expand the signal corpus with more high-quality concepts."""
    from experiments.ollama_corpus_refiner import OllamaClient
    
    print(f"\nExpanding corpus with {num_new} new concepts...")
    
    # Load existing corpus
    with open(corpus_path, 'r') as f:
        data = json.load(f)
    existing_frames = data.get('frames', [])
    existing_agents = {f.get('agent', '').lower() for f in existing_frames}
    
    # Load truth corpus
    qa = GeometricQA()
    qa.load_corpus("truthspace_lcm/corpus_experimental.json")
    qa.set_output_lens('natural')
    
    # Find good concepts to add (not already in signal)
    candidates = []
    for name, concept in qa.knowledge.concepts.items():
        if name in existing_agents:
            continue
        if not concept.is_content_word:
            continue
        if not concept.actions:
            continue
        # Score by how much info we have
        info_score = len(concept.actions) + len(concept.targets)
        if info_score >= 3:
            candidates.append((name, info_score))
    
    # Sort by info score (most info first)
    candidates.sort(key=lambda x: -x[1])
    candidates = candidates[:num_new]
    
    print(f"Found {len(candidates)} candidate concepts")
    
    # Generate polished versions
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama not available!")
        return
    
    new_frames = []
    for i, (concept, _) in enumerate(candidates):
        raw = qa.ask(f"What is {concept}?")
        if "don't know" in raw.lower():
            continue
        
        # Use a better prompt for cleaner output
        prompt = f"""Rewrite this sentence to be grammatically correct and natural. 
Keep it simple and direct. Only output the rewritten sentence, nothing else.

Original: "{raw}"

Rewritten:"""
        
        polished = ollama.generate(prompt, temperature=0.2)
        
        if polished:
            polished = polished.strip().strip('"').strip()
            
            # Quick quality check
            score, _ = score_frame(polished, concept)
            if score >= 0:
                new_frames.append({
                    'text': polished,
                    'source': 'signal',
                    'agent': concept,
                })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(candidates)}")
    
    print(f"Generated {len(new_frames)} new quality frames")
    
    # Combine and save
    all_frames = existing_frames + new_frames
    with open(corpus_path, 'w') as f:
        json.dump({'frames': all_frames}, f, indent=2)
    
    print(f"Saved {len(all_frames)} total frames")


def generate_full_corpus(output_path: str, min_score: float = 1.0, max_retries: int = 2):
    """
    Generate a complete high-quality signal corpus from scratch.
    
    For each concept with enough info:
    1. Get raw truth output
    2. Polish with Qwen2
    3. Score the result
    4. Retry if score too low
    5. Only keep high-quality frames
    """
    from experiments.ollama_corpus_refiner import OllamaClient
    
    print("=" * 70)
    print("FULL SIGNAL CORPUS GENERATION")
    print("=" * 70)
    
    # Load truth corpus
    qa = GeometricQA()
    qa.load_corpus("truthspace_lcm/corpus_experimental.json")
    qa.set_output_lens('natural')
    
    # Find all concepts with enough info
    candidates = []
    for name, concept in qa.knowledge.concepts.items():
        if not concept.is_content_word:
            continue
        if not concept.actions:
            continue
        info_score = len(concept.actions) + len(concept.targets)
        if info_score >= 2:  # Lower threshold to get more concepts
            candidates.append((name, info_score))
    
    candidates.sort(key=lambda x: -x[1])
    print(f"Found {len(candidates)} candidate concepts")
    
    # Initialize Ollama
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama not available!")
        return
    
    # Better prompt templates for different roles
    prompts = {
        'detective': """Rewrite to describe a detective character naturally:
Original: "{raw}"
Rewritten (one sentence only):""",
        
        'doctor': """Rewrite to describe a doctor/medical character naturally:
Original: "{raw}"
Rewritten (one sentence only):""",
        
        'science': """Rewrite to describe a scientific field naturally:
Original: "{raw}"
Rewritten (one sentence only):""",
        
        'study': """Rewrite to describe an academic field of study naturally:
Original: "{raw}"
Rewritten (one sentence only):""",
        
        'concept': """Rewrite to describe an abstract concept naturally:
Original: "{raw}"
Rewritten (one sentence only):""",
        
        'default': """Rewrite this to be grammatically correct and natural.
Keep it simple and direct. One sentence only.
Original: "{raw}"
Rewritten:"""
    }
    
    frames = []
    stats = {'total': 0, 'accepted': 0, 'rejected': 0, 'retried': 0}
    
    for i, (concept, _) in enumerate(candidates):
        stats['total'] += 1
        
        # Get raw truth
        raw = qa.ask(f"What is {concept}?")
        if "don't know" in raw.lower():
            continue
        
        # Detect role for better prompting
        raw_lower = raw.lower()
        if 'detective' in raw_lower:
            role = 'detective'
        elif 'doctor' in raw_lower:
            role = 'doctor'
        elif 'science' in raw_lower:
            role = 'science'
        elif 'study' in raw_lower:
            role = 'study'
        elif 'concept' in raw_lower or 'entity' in raw_lower:
            role = 'concept'
        else:
            role = 'default'
        
        prompt_template = prompts[role]
        
        # Try to generate high-quality output
        best_frame = None
        best_score = -100
        
        for attempt in range(max_retries + 1):
            prompt = prompt_template.format(raw=raw)
            
            # Vary temperature slightly on retries
            temp = 0.2 + (attempt * 0.1)
            polished = ollama.generate(prompt, temperature=temp)
            
            if not polished:
                continue
            
            polished = polished.strip().strip('"').strip()
            
            # Score it
            score, reasons = score_frame(polished, concept)
            
            if score > best_score:
                best_score = score
                best_frame = {
                    'text': polished,
                    'source': 'signal',
                    'agent': concept,
                    'role': role,
                }
            
            if score >= min_score:
                break  # Good enough
            
            if attempt < max_retries:
                stats['retried'] += 1
        
        # Accept or reject
        if best_frame and best_score >= min_score:
            frames.append(best_frame)
            stats['accepted'] += 1
        else:
            stats['rejected'] += 1
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(candidates)} - Accepted: {stats['accepted']}, Rejected: {stats['rejected']}")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump({'frames': frames}, f, indent=2)
    
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total processed: {stats['total']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Retried: {stats['retried']}")
    print(f"Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Corpus Curator")
    parser.add_argument("--curate", action="store_true", help="Curate existing corpus")
    parser.add_argument("--expand", type=int, default=0, help="Expand with N new concepts")
    parser.add_argument("--generate", action="store_true", help="Generate full corpus from scratch")
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum quality score")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per concept")
    parser.add_argument("--input", type=str, default="truthspace_lcm/corpus_signal.json")
    parser.add_argument("--output", type=str, default="truthspace_lcm/corpus_signal_curated.json")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_full_corpus(args.output, min_score=args.min_score, max_retries=args.max_retries)
    elif args.curate:
        curate_corpus(args.input, args.output, args.min_score)
    
    if args.expand > 0:
        # Expand the curated corpus (or original if not curated)
        target = args.output if args.curate else args.input
        expand_corpus(target, args.expand)


if __name__ == "__main__":
    main()
