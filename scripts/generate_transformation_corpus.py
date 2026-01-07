#!/usr/bin/env python3
"""
Transformation Corpus Generator

Generates sentence transformations using a local LLM (Ollama) to build
a corpus for geometric dimension manipulation.

Usage:
    python scripts/generate_transformation_corpus.py
    python scripts/generate_transformation_corpus.py --dry-run
    python scripts/generate_transformation_corpus.py --dimensions tense,regality

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import json
import requests
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# SEED SENTENCES
# =============================================================================

SEED_SENTENCES = [
    # Classic nursery rhyme - good for testing transformations
    {
        "text": "Jack and Jill went up the hill to fetch a pail of water.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "compound",
    },
    # Simple sentence - minimal structure
    {
        "text": "The cat sat on the mat.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "simple",
    },
    # Present tense with belief
    {
        "text": "She believes that honesty is the best policy.",
        "base_tense": "present",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "complex",
    },
    # Future tense
    {
        "text": "They will arrive at noon tomorrow.",
        "base_tense": "future",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "simple",
    },
    # Descriptive with location
    {
        "text": "The ancient castle stands on the cliff overlooking the sea.",
        "base_tense": "present",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "complex",
    },
    # Action with object
    {
        "text": "The chef prepared a delicious meal for the guests.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "simple",
    },
    # Question structure (declarative form)
    {
        "text": "The scientist discovered a new element in the laboratory.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "simple",
    },
    # Multiple subjects
    {
        "text": "The king and queen ruled the kingdom with wisdom and grace.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "noble",
        "structure": "compound",
    },
    # Abstract concept
    {
        "text": "Knowledge grows when shared freely among people.",
        "base_tense": "present",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "complex",
    },
    # Conditional/hypothetical
    {
        "text": "The traveler walked through the forest searching for shelter.",
        "base_tense": "past",
        "base_formality": "neutral",
        "base_regality": "common",
        "structure": "complex",
    },
]


# =============================================================================
# DIMENSIONS AND VALUES
# =============================================================================

DIMENSIONS = {
    # Grammatical dimensions
    "tense": {
        "values": ["past", "present", "future"],
        "prompt_template": "Rewrite the following sentence in the {value} tense. Only output the rewritten sentence, nothing else.",
    },
    "voice": {
        "values": ["active", "passive"],
        "prompt_template": "Rewrite the following sentence in {value} voice. Only output the rewritten sentence, nothing else.",
    },
    
    # Semantic dimensions
    "regality": {
        "values": ["common", "noble", "royal"],
        "prompt_template": "Rewrite the following sentence to sound {value} (like a {value_desc}). Only output the rewritten sentence, nothing else.",
        "value_descriptions": {
            "common": "common person speaking casually",
            "noble": "nobleman or noblewoman speaking with dignity",
            "royal": "king or queen making a royal proclamation",
        },
    },
    "formality": {
        "values": ["casual", "neutral", "formal"],
        "prompt_template": "Rewrite the following sentence in a {value} tone. Only output the rewritten sentence, nothing else.",
    },
    "certainty": {
        "values": ["uncertain", "neutral", "certain"],
        "prompt_template": "Rewrite the following sentence to express {value} certainty. Only output the rewritten sentence, nothing else.",
        "value_descriptions": {
            "uncertain": "doubt and uncertainty",
            "neutral": "neutral certainty",
            "certain": "absolute certainty and conviction",
        },
    },
    "emotion": {
        "values": ["sad", "neutral", "happy"],
        "prompt_template": "Rewrite the following sentence with a {value} emotional tone. Only output the rewritten sentence, nothing else.",
    },
}

# Combined transformations (dimension pairs)
COMBINED_TRANSFORMATIONS = [
    ("tense", "future", "regality", "royal"),
    ("tense", "past", "formality", "casual"),
    ("formality", "formal", "certainty", "certain"),
    ("regality", "noble", "emotion", "happy"),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Transformation:
    """A single transformation example."""
    source: str
    source_dimensions: Dict[str, str]
    target: str
    target_dimensions: Dict[str, str]
    dimension_delta: Dict[str, Tuple[str, str]]
    prompt_used: str = ""
    generation_time: float = 0.0


@dataclass
class TransformationCorpus:
    """Collection of transformations."""
    version: int = 1
    generated: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, t: Transformation):
        self.transformations.append(asdict(t))
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TransformationCorpus':
        with open(path, 'r') as f:
            data = json.load(f)
        corpus = cls()
        corpus.version = data.get("version", 1)
        corpus.generated = data.get("generated", "")
        corpus.model = data.get("model", "")
        corpus.transformations = data.get("transformations", [])
        return corpus


# =============================================================================
# GENERATOR
# =============================================================================

class TransformationGenerator:
    """Generates transformation corpus using Ollama."""
    
    def __init__(self, model: str = "qwen2.5:14b", ollama_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.corpus = TransformationCorpus(model=model)
        self.dry_run = False
        self.verbose = True
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API and return response."""
        if self.dry_run:
            return f"[DRY RUN] Would generate for: {prompt[:50]}..."
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent output
                        "num_predict": 200,  # Limit output length
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"  [ERROR] Ollama call failed: {e}")
            return None
    
    def _build_prompt(self, sentence: str, dimension: str, value: str) -> str:
        """Build prompt for a transformation."""
        dim_config = DIMENSIONS[dimension]
        template = dim_config["prompt_template"]
        
        # Handle value descriptions
        value_desc = ""
        if "value_descriptions" in dim_config:
            value_desc = dim_config["value_descriptions"].get(value, value)
        
        prompt = template.format(value=value, value_desc=value_desc)
        prompt += f"\n\nOriginal: \"{sentence}\"\nRewritten:"
        
        return prompt
    
    def _build_combined_prompt(self, sentence: str, 
                                dim1: str, val1: str,
                                dim2: str, val2: str) -> str:
        """Build prompt for combined transformation."""
        prompt = f"Rewrite the following sentence to be {val1} ({dim1}) and {val2} ({dim2}). Only output the rewritten sentence, nothing else."
        prompt += f"\n\nOriginal: \"{sentence}\"\nRewritten:"
        return prompt
    
    def generate_single(self, seed: Dict[str, Any], dimension: str, value: str) -> Optional[Transformation]:
        """Generate a single transformation."""
        sentence = seed["text"]
        
        # Skip if already at target value
        base_key = f"base_{dimension}"
        if seed.get(base_key) == value:
            if self.verbose:
                print(f"  [SKIP] Already at {dimension}={value}")
            return None
        
        prompt = self._build_prompt(sentence, dimension, value)
        
        start_time = time.time()
        result = self._call_ollama(prompt)
        gen_time = time.time() - start_time
        
        if not result:
            return None
        
        # Clean up result (remove quotes if present)
        result = result.strip('"\'')
        
        # Build source dimensions from seed
        source_dims = {
            "tense": seed.get("base_tense", "unknown"),
            "formality": seed.get("base_formality", "neutral"),
            "regality": seed.get("base_regality", "common"),
        }
        
        # Build target dimensions
        target_dims = source_dims.copy()
        target_dims[dimension] = value
        
        return Transformation(
            source=sentence,
            source_dimensions=source_dims,
            target=result,
            target_dimensions=target_dims,
            dimension_delta={dimension: (source_dims.get(dimension, "unknown"), value)},
            prompt_used=prompt,
            generation_time=gen_time,
        )
    
    def generate_combined(self, seed: Dict[str, Any],
                          dim1: str, val1: str,
                          dim2: str, val2: str) -> Optional[Transformation]:
        """Generate a combined transformation."""
        sentence = seed["text"]
        prompt = self._build_combined_prompt(sentence, dim1, val1, dim2, val2)
        
        start_time = time.time()
        result = self._call_ollama(prompt)
        gen_time = time.time() - start_time
        
        if not result:
            return None
        
        result = result.strip('"\'')
        
        source_dims = {
            "tense": seed.get("base_tense", "unknown"),
            "formality": seed.get("base_formality", "neutral"),
            "regality": seed.get("base_regality", "common"),
        }
        
        target_dims = source_dims.copy()
        target_dims[dim1] = val1
        target_dims[dim2] = val2
        
        return Transformation(
            source=sentence,
            source_dimensions=source_dims,
            target=result,
            target_dimensions=target_dims,
            dimension_delta={
                dim1: (source_dims.get(dim1, "unknown"), val1),
                dim2: (source_dims.get(dim2, "unknown"), val2),
            },
            prompt_used=prompt,
            generation_time=gen_time,
        )
    
    def generate_all(self, dimensions: Optional[List[str]] = None,
                     include_combined: bool = True) -> TransformationCorpus:
        """Generate all transformations."""
        dims_to_use = dimensions or list(DIMENSIONS.keys())
        
        total_single = len(SEED_SENTENCES) * sum(
            len(DIMENSIONS[d]["values"]) for d in dims_to_use
        )
        total_combined = len(SEED_SENTENCES) * len(COMBINED_TRANSFORMATIONS) if include_combined else 0
        total = total_single + total_combined
        
        print(f"Generating {total} transformations...")
        print(f"  Seeds: {len(SEED_SENTENCES)}")
        print(f"  Dimensions: {dims_to_use}")
        print(f"  Model: {self.model}")
        print()
        
        count = 0
        
        # Single dimension transformations
        for i, seed in enumerate(SEED_SENTENCES):
            print(f"[{i+1}/{len(SEED_SENTENCES)}] Processing: {seed['text'][:40]}...")
            
            for dim in dims_to_use:
                for value in DIMENSIONS[dim]["values"]:
                    count += 1
                    if self.verbose:
                        print(f"  [{count}/{total}] {dim} -> {value}", end="")
                    
                    t = self.generate_single(seed, dim, value)
                    if t:
                        self.corpus.add(t)
                        if self.verbose:
                            print(f" ({t.generation_time:.1f}s)")
                            if not self.dry_run:
                                print(f"    -> {t.target[:60]}...")
                    else:
                        if self.verbose:
                            print(" [skipped]")
        
        # Combined transformations
        if include_combined:
            print("\nGenerating combined transformations...")
            for i, seed in enumerate(SEED_SENTENCES):
                for dim1, val1, dim2, val2 in COMBINED_TRANSFORMATIONS:
                    count += 1
                    if self.verbose:
                        print(f"  [{count}/{total}] {dim1}={val1} + {dim2}={val2}", end="")
                    
                    t = self.generate_combined(seed, dim1, val1, dim2, val2)
                    if t:
                        self.corpus.add(t)
                        if self.verbose:
                            print(f" ({t.generation_time:.1f}s)")
                            if not self.dry_run:
                                print(f"    -> {t.target[:60]}...")
                    else:
                        if self.verbose:
                            print(" [failed]")
        
        print(f"\nGenerated {len(self.corpus.transformations)} transformations")
        return self.corpus


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate transformation corpus")
    parser.add_argument("--dry-run", action="store_true", help="Don't call LLM, just show what would be done")
    parser.add_argument("--dimensions", type=str, help="Comma-separated list of dimensions to generate")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Ollama model to use")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--no-combined", action="store_true", help="Skip combined transformations")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "truthspace_lcm" / "corpus" / "transformation_corpus.json"
    
    # Parse dimensions
    dimensions = None
    if args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",")]
        # Validate
        for d in dimensions:
            if d not in DIMENSIONS:
                print(f"Unknown dimension: {d}")
                print(f"Available: {list(DIMENSIONS.keys())}")
                return 1
    
    # Create generator
    generator = TransformationGenerator(model=args.model)
    generator.dry_run = args.dry_run
    generator.verbose = not args.quiet
    
    # Generate
    corpus = generator.generate_all(
        dimensions=dimensions,
        include_combined=not args.no_combined
    )
    
    # Save
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus.save(output_path)
        print(f"\nSaved to: {output_path}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total transformations: {len(corpus.transformations)}")
    
    # Count by dimension
    dim_counts = {}
    for t in corpus.transformations:
        for dim in t.get("dimension_delta", {}).keys():
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
    
    print("By dimension:")
    for dim, count in sorted(dim_counts.items()):
        print(f"  {dim}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())
