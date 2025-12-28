#!/usr/bin/env python3
"""
Clean Corpus Script

Applies quality filtering to the concept corpus and saves a cleaned version.
Also extracts and saves character lists per source.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.frame_quality import (
    FrameQualityFilter, QualityTier, clean_corpus,
    extract_character_set, ALL_KNOWN_CHARACTERS,
    SHERLOCK_CHARACTERS, PRIDE_PREJUDICE_CHARACTERS,
    ALICE_CHARACTERS, MOBY_DICK_CHARACTERS,
)


# Extended character sets for all sources in corpus
TALE_TWO_CITIES_CHARACTERS = {
    'carton', 'darnay', 'charles', 'lucie', 'manette', 'defarge',
    'madame', 'lorry', 'cruncher', 'pross', 'stryver', 'barsad',
    'gabelle', 'gaspard', 'foulon', 'ernest',
}

GREAT_EXPECTATIONS_CHARACTERS = {
    'pip', 'joe', 'estella', 'havisham', 'magwitch', 'jaggers',
    'wemmick', 'biddy', 'herbert', 'compeyson', 'orlick', 'drummle',
    'pumblechook', 'gargery', 'wopsle', 'hubble', 'camilla',
}

TOM_SAWYER_CHARACTERS = {
    'tom', 'huck', 'huckleberry', 'becky', 'aunt', 'polly', 'sid',
    'joe', 'harper', 'injun', 'muff', 'potter', 'thatcher', 'dobbins',
    'alfred', 'ben', 'jim', 'mary', 'rogers', 'sawyer',
}

FRANKENSTEIN_CHARACTERS = {
    'victor', 'frankenstein', 'elizabeth', 'henry', 'clerval',
    'creature', 'monster', 'walton', 'justine', 'william', 'alphonse',
    'ernest', 'felix', 'agatha', 'safie', 'delacey', 'beaufort',
}

DRACULA_CHARACTERS = {
    'dracula', 'count', 'harker', 'jonathan', 'mina', 'lucy',
    'seward', 'helsing', 'van', 'arthur', 'holmwood', 'quincey',
    'morris', 'renfield', 'hawkins', 'westenra',
}

LES_MISERABLES_CHARACTERS = {
    'valjean', 'jean', 'javert', 'cosette', 'fantine', 'marius',
    'eponine', 'gavroche', 'thenardier', 'bishop', 'myriel',
    'bienvenu', 'enjolras', 'gillenormand', 'magloire', 'baptistine',
}

WHITE_FANG_CHARACTERS = {
    'fang', 'white', 'grey', 'beaver', 'bill', 'henry', 'kiche',
    'scott', 'weedon', 'matt', 'beauty', 'smith', 'cherokee',
}

# Combine all known characters
EXTENDED_CHARACTERS = (
    ALL_KNOWN_CHARACTERS |
    TALE_TWO_CITIES_CHARACTERS |
    GREAT_EXPECTATIONS_CHARACTERS |
    TOM_SAWYER_CHARACTERS |
    FRANKENSTEIN_CHARACTERS |
    DRACULA_CHARACTERS |
    LES_MISERABLES_CHARACTERS |
    WHITE_FANG_CHARACTERS
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean the concept corpus")
    parser.add_argument("--input", "-i", default="truthspace_lcm/concept_corpus.json",
                       help="Input corpus file")
    parser.add_argument("--output", "-o", default="truthspace_lcm/concept_corpus_clean.json",
                       help="Output cleaned corpus file")
    parser.add_argument("--min-tier", "-t", default="bronze",
                       choices=["gold", "silver", "bronze"],
                       help="Minimum quality tier to keep")
    parser.add_argument("--analyze-only", "-a", action="store_true",
                       help="Only analyze, don't save")
    parser.add_argument("--extract-characters", "-c", action="store_true",
                       help="Extract and save character lists")
    
    args = parser.parse_args()
    
    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input, 'r') as f:
        corpus = json.load(f)
    
    frames = corpus['frames']
    print(f"Total frames: {len(frames)}")
    print()
    
    # Map tier name to enum
    tier_map = {
        "gold": QualityTier.GOLD,
        "silver": QualityTier.SILVER,
        "bronze": QualityTier.BRONZE,
    }
    min_tier = tier_map[args.min_tier]
    
    # Create filter with extended character set
    filter = FrameQualityFilter(known_characters=EXTENDED_CHARACTERS)
    
    # Analyze
    print("Analyzing frame quality...")
    report = filter.analyze(frames)
    print(report)
    print()
    
    # Show auto-detected characters not in our set
    print("=== AUTO-DETECTED CHARACTERS (not in known set) ===")
    suggestions = filter.suggest_characters(frames, top_k=50)
    unknown = [(name, count) for name, count in suggestions if name not in EXTENDED_CHARACTERS]
    for name, count in unknown[:20]:
        print(f"  {name}: {count}")
    print()
    
    if args.extract_characters:
        print("=== EXTRACTING CHARACTERS BY SOURCE ===")
        characters_by_source = extract_character_set(frames)
        
        char_file = args.output.replace('.json', '_characters.json')
        with open(char_file, 'w') as f:
            # Convert sets to lists for JSON
            json.dump({k: sorted(v) for k, v in characters_by_source.items()}, f, indent=2)
        print(f"Saved character lists to {char_file}")
        print()
    
    if args.analyze_only:
        print("Analysis only mode - not saving cleaned corpus")
        return
    
    # Filter frames
    print(f"Filtering to {args.min_tier} tier and above...")
    cleaned_frames = filter.filter_frames(frames, min_tier=min_tier)
    print(f"Kept {len(cleaned_frames)} frames ({100*len(cleaned_frames)/len(frames):.1f}%)")
    print()
    
    # Build cleaned corpus
    cleaned_corpus = {
        "version": corpus.get("version", "0.8.0") + "-clean",
        "description": f"Cleaned corpus (min tier: {args.min_tier})",
        "original_frames": len(frames),
        "cleaned_frames": len(cleaned_frames),
        "quality_report": {
            "gold": report.gold_frames,
            "silver": report.silver_frames,
            "bronze": report.bronze_frames,
            "noise_removed": report.noise_frames,
        },
        "sources": corpus.get("sources", []),
        "frames": cleaned_frames,
    }
    
    # Save
    print(f"Saving cleaned corpus to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(cleaned_corpus, f, indent=2)
    
    print("Done!")
    
    # Show sample of cleaned frames
    print()
    print("=== SAMPLE CLEANED FRAMES ===")
    for f in cleaned_frames[:10]:
        agent = f.get('agent', '?')
        action = f.get('action', '?')
        patient = f.get('patient', '?')
        source = f.get('source', '?')
        print(f"  {agent} --{action}--> {patient} [{source}]")


if __name__ == "__main__":
    main()
