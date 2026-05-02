#!/usr/bin/env python3
"""
Fast Generation via Navigation
===============================

The hypothesis: For queries where we have stored paths,
we can skip the transformer entirely and just navigate.

Traditional:
  Query → 28 layers → Next token → Repeat
  ~50-100ms per token

Navigation:
  Query → Parse intent → Lookup path → Return answer
  ~1ms total

This only works for queries that match our stored paths.
But for those queries, it's 50-100x faster.
"""

import torch
import time
import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.navigator import PhiNavigator, NavigationResult
from phi_navigator.paths import PathStore
from phi_navigator.relationships import (
    OppositeRelationship, GenderRelationship, 
    SpatialRelationship, TenseRelationship
)


@dataclass
class ParsedQuery:
    """A parsed query ready for navigation."""
    query_type: str  # 'opposite', 'gender', 'spatial', etc.
    source: str      # The word to transform
    relationship: str  # The relationship name
    confidence: float  # How confident we are in the parse


class QueryParser:
    """
    Parse natural language queries into navigation instructions.
    
    Examples:
      "What is the opposite of hot?" → (opposite, hot)
      "What is the female version of king?" → (gender, king)
      "What is the past tense of run?" → (tense, run)
    """
    
    PATTERNS = [
        # Opposite patterns
        (r"opposite of ['\"]?(\w+)['\"]?", "opposite"),
        (r"antonym of ['\"]?(\w+)['\"]?", "opposite"),
        (r"what is the opposite of ['\"]?(\w+)['\"]?", "opposite"),
        
        # Gender patterns
        (r"female version of ['\"]?(\w+)['\"]?", "gender"),
        (r"male version of ['\"]?(\w+)['\"]?", "gender"),
        (r"female of ['\"]?(\w+)['\"]?", "gender"),
        (r"male of ['\"]?(\w+)['\"]?", "gender"),
        
        # Spatial patterns
        (r"spatial opposite of ['\"]?(\w+)['\"]?", "spatial"),
        
        # Tense patterns
        (r"past tense of ['\"]?(\w+)['\"]?", "tense_present_to_past"),
        (r"present tense of ['\"]?(\w+)['\"]?", "tense_past_to_present"),
    ]
    
    def parse(self, query: str) -> Optional[ParsedQuery]:
        """Parse a query into navigation instructions."""
        query_lower = query.lower().strip()
        
        for pattern, query_type in self.PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                source = match.group(1)
                return ParsedQuery(
                    query_type=query_type,
                    source=source,
                    relationship=query_type,
                    confidence=0.9,
                )
        
        return None


class FastGenerator:
    """
    Fast token generation via navigation.
    
    For queries that match stored paths:
    1. Parse the query
    2. Look up the path
    3. Return the answer
    
    No transformer inference needed.
    """
    
    def __init__(self, navigator: PhiNavigator):
        self.navigator = navigator
        self.parser = QueryParser()
        
        # Statistics
        self.nav_count = 0
        self.nav_time = 0.0
        self.gen_count = 0
        self.gen_time = 0.0
    
    def generate(self, query: str) -> Tuple[str, str, float]:
        """
        Generate an answer to a query.
        
        Returns: (answer, method, time_ms)
        """
        start = time.perf_counter()
        
        # Try to parse the query
        parsed = self.parser.parse(query)
        
        if parsed:
            # Try navigation first
            result = self.navigator.lookup(parsed.source, parsed.relationship)
            
            if result:
                elapsed = (time.perf_counter() - start) * 1000
                self.nav_count += 1
                self.nav_time += elapsed
                return result.target, "navigation", elapsed
        
        # Fall back to model generation
        # (In a real system, this would call the model)
        elapsed = (time.perf_counter() - start) * 1000
        self.gen_count += 1
        self.gen_time += elapsed
        return "[requires model]", "fallback", elapsed
    
    def stats(self) -> dict:
        """Get generation statistics."""
        return {
            "navigation_count": self.nav_count,
            "navigation_avg_ms": self.nav_time / self.nav_count if self.nav_count else 0,
            "generation_count": self.gen_count,
            "generation_avg_ms": self.gen_time / self.gen_count if self.gen_count else 0,
        }


def benchmark_fast_generation(model, tokenizer):
    """Benchmark fast generation vs model generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("FAST GENERATION BENCHMARK")
    print("="*70)
    
    # Create navigator with pre-discovered paths
    nav = PhiNavigator(model, tokenizer)
    
    # Discover some relationships
    print("\nDiscovering relationships...")
    relationships = [
        OppositeRelationship(),
        GenderRelationship(),
        SpatialRelationship(axis="vertical"),
    ]
    
    for rel in relationships:
        paths = nav.discover_relationship(rel, n_pairs=15)
        print(f"  {rel.name}: {len(paths)} paths")
    
    print(f"\nTotal paths stored: {nav.path_store.count()}")
    
    # Create fast generator
    fast_gen = FastGenerator(nav)
    
    # Test queries
    test_queries = [
        # Should use navigation (if paths exist)
        "What is the opposite of hot?",
        "What is the opposite of big?",
        "What is the opposite of fast?",
        "What is the female version of king?",
        "What is the spatial opposite of up?",
        
        # Might need fallback
        "What is the opposite of beautiful?",
        "What is the capital of France?",
    ]
    
    print("\n" + "-"*70)
    print("TESTING FAST GENERATION")
    print("-"*70)
    
    nav_results = []
    
    for query in test_queries:
        answer, method, time_ms = fast_gen.generate(query)
        nav_results.append((query, answer, method, time_ms))
        print(f"\n  Q: {query}")
        print(f"  A: {answer} ({method}, {time_ms:.2f}ms)")
    
    # Now compare with full model generation
    print("\n" + "-"*70)
    print("COMPARING WITH MODEL GENERATION")
    print("-"*70)
    
    model_times = []
    
    for query in test_queries[:5]:  # Just test first 5
        messages = [{"role": "user", "content": query + " Reply with just one word."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = (time.perf_counter() - start) * 1000
        model_times.append(elapsed)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        answer = response.split()[0] if response.split() else response
        
        print(f"\n  Q: {query}")
        print(f"  A: {answer} (model, {elapsed:.0f}ms)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    nav_times = [t for _, _, m, t in nav_results if m == "navigation"]
    
    if nav_times and model_times:
        avg_nav = sum(nav_times) / len(nav_times)
        avg_model = sum(model_times) / len(model_times)
        speedup = avg_model / avg_nav if avg_nav > 0 else 0
        
        print(f"\nNavigation average: {avg_nav:.2f}ms")
        print(f"Model average:      {avg_model:.0f}ms")
        print(f"Speedup:            {speedup:.0f}x")
        
        print(f"""
KEY INSIGHT:
  For queries with stored paths, navigation is {speedup:.0f}x faster.
  
  Navigation: Parse query → Lookup path → Return answer
  Model:      Tokenize → 28 layers → Decode → Return answer
  
  The transformer is doing O(N × L × D²) work.
  Navigation is doing O(1) lookup.
""")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    benchmark_fast_generation(model, tokenizer)


if __name__ == "__main__":
    main()
