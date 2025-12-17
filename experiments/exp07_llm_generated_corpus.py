#!/usr/bin/env python3
"""
Experiment 7: LLM-Generated Knowledge Corpus

Demonstrates the full dynamic LCM pipeline:
1. Use Ollama (qwen2:latest) to generate knowledge domains
2. Generate knowledge entries for each domain
3. Ingest into DynamicLCM with automatic primitive discovery
4. Generate test queries and evaluate

This proves that the geometric LCM can scale to arbitrary domains
without any manual knowledge curation.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.dynamic_lcm import DynamicLCM
from truthspace_lcm.core.knowledge_generator import KnowledgeGenerator


def run_experiment(
    domain_count: int = 5,
    entries_per_domain: int = 15,
    seed_topic: str = None,
    model: str = "qwen2:latest",
    save_corpus: bool = True
):
    """Run the full LLM-generated corpus experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 7: LLM-Generated Knowledge Corpus")
    print("=" * 70)
    print(f"\nModel: {model}")
    print(f"Domains: {domain_count}")
    print(f"Entries per domain: {entries_per_domain}")
    if seed_topic:
        print(f"Seed topic: {seed_topic}")
    print()
    
    # Initialize
    generator = KnowledgeGenerator(model=model)
    lcm = DynamicLCM(dimensions=20)  # More dimensions for more domains
    
    # Step 1: Generate domains
    print("=" * 70)
    print("STEP 1: Generating Domains")
    print("=" * 70)
    
    domains = generator.generate_domains(seed_topic=seed_topic, count=domain_count)
    print(f"\nGenerated {len(domains)} domains:")
    for d in domains:
        print(f"  - {d}")
    
    if not domains:
        print("ERROR: No domains generated. Check Ollama connection.")
        return None
    
    # Step 2: Generate knowledge for each domain
    print("\n" + "=" * 70)
    print("STEP 2: Generating Knowledge Entries")
    print("=" * 70)
    
    all_entries = []
    for domain in domains:
        print(f"\n--- Generating entries for '{domain}' ---")
        entries = generator.generate_knowledge_for_domain(domain, count=entries_per_domain)
        print(f"Generated {len(entries)} entries")
        
        # Show a few examples
        for entry in entries[:3]:
            print(f"  • {entry.content[:60]}...")
        if len(entries) > 3:
            print(f"  ... and {len(entries) - 3} more")
        
        all_entries.extend(entries)
    
    print(f"\nTotal entries generated: {len(all_entries)}")
    
    # Step 3: Ingest into DynamicLCM
    print("\n" + "=" * 70)
    print("STEP 3: Ingesting into DynamicLCM")
    print("=" * 70)
    
    # Convert to dict format for ingestion
    entry_dicts = [
        {
            "content": e.content,
            "description": e.description,
            "domain": e.domain
        }
        for e in all_entries
    ]
    
    lcm.ingest_batch(entry_dicts)
    
    stats = lcm.stats()
    print(f"\nLCM Statistics:")
    print(f"  Dimensions: {stats['dimensions']}")
    print(f"  Seed primitives: {stats['seed_primitives']}")
    print(f"  Emergent primitives: {stats['emergent_primitives']}")
    print(f"  Total entries: {stats['entries']}")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"\nDomain sizes:")
    for domain, size in stats['domain_sizes'].items():
        print(f"  {domain}: {size} entries")
    
    # Show emergent primitives
    emergent = [p for p in lcm.primitives if not p.is_seed]
    if emergent:
        print(f"\nEmergent primitives discovered ({len(emergent)}):")
        # Group by domain
        by_domain = {}
        for p in emergent:
            d = p.domain or "general"
            if d not in by_domain:
                by_domain[d] = []
            by_domain[d].append(p)
        
        for domain, prims in sorted(by_domain.items()):
            print(f"  {domain}:")
            for p in prims[:5]:  # Show first 5
                print(f"    - {p.name}: {list(p.keywords)[:3]}")
            if len(prims) > 5:
                print(f"    ... and {len(prims) - 5} more")
    
    # Step 4: Generate test queries
    print("\n" + "=" * 70)
    print("STEP 4: Generating Test Queries")
    print("=" * 70)
    
    test_queries = generator.generate_test_queries(domains, queries_per_domain=3)
    print(f"\nGenerated {len(test_queries)} test queries")
    
    # Step 5: Evaluate
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation")
    print("=" * 70)
    
    correct_domain = 0
    total = 0
    
    for item in test_queries:
        if not isinstance(item, dict) or 'query' not in item:
            continue
            
        query = item['query']
        expected_domain = item.get('domain', 'unknown')
        
        detected_domain, entry, similarity = lcm.resolve_with_domain_detection(query)
        
        is_correct = detected_domain == expected_domain
        if is_correct:
            correct_domain += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} \"{query}\"")
        print(f"    Expected: {expected_domain}, Got: {detected_domain}")
        if entry:
            print(f"    Result: {entry.content[:50]}... (sim: {similarity:.2f})")
    
    if total > 0:
        accuracy = correct_domain / total
        print(f"\n{'=' * 70}")
        print(f"RESULTS")
        print(f"{'=' * 70}")
        print(f"Domain detection accuracy: {correct_domain}/{total} ({accuracy:.1%})")
    
    # Save corpus if requested
    if save_corpus:
        corpus_path = Path(__file__).parent / "generated_corpus.json"
        corpus = {
            "metadata": {
                "model": model,
                "seed_topic": seed_topic,
                "domain_count": domain_count,
                "entries_per_domain": entries_per_domain,
            },
            "domains": domains,
            "entries": entry_dicts,
            "test_queries": test_queries,
            "results": {
                "emergent_primitives": stats['emergent_primitives'],
                "domain_accuracy": accuracy if total > 0 else 0,
            }
        }
        
        with open(corpus_path, 'w') as f:
            json.dump(corpus, f, indent=2)
        print(f"\nCorpus saved to: {corpus_path}")
    
    # Save LCM state
    lcm_path = Path(__file__).parent / "generated_lcm.json"
    lcm.save(str(lcm_path))
    print(f"LCM state saved to: {lcm_path}")
    
    return lcm, stats


def run_with_specific_domains(domains: list, entries_per_domain: int = 15, model: str = "qwen2:latest"):
    """Run with specific domains instead of generating them."""
    
    print("=" * 70)
    print("EXPERIMENT 7b: Specific Domain Corpus")
    print("=" * 70)
    print(f"\nDomains: {domains}")
    print(f"Entries per domain: {entries_per_domain}")
    print()
    
    generator = KnowledgeGenerator(model=model)
    lcm = DynamicLCM(dimensions=20)
    
    all_entries = []
    for domain in domains:
        print(f"\n--- Generating entries for '{domain}' ---")
        entries = generator.generate_knowledge_for_domain(domain, count=entries_per_domain)
        print(f"Generated {len(entries)} entries")
        all_entries.extend(entries)
    
    # Ingest
    entry_dicts = [
        {"content": e.content, "description": e.description, "domain": e.domain}
        for e in all_entries
    ]
    lcm.ingest_batch(entry_dicts)
    
    stats = lcm.stats()
    print(f"\n--- Results ---")
    print(f"Total entries: {stats['entries']}")
    print(f"Emergent primitives: {stats['emergent_primitives']}")
    
    # Quick test
    print(f"\n--- Quick Resolution Test ---")
    test_queries = [
        "how do I list files",
        "what's a good recipe",
        "I'm feeling sick",
        "hello there",
    ]
    
    for query in test_queries:
        domain, entry, sim = lcm.resolve_with_domain_detection(query)
        print(f"  \"{query}\" → {domain}: {entry.content[:40] if entry else 'None'}...")
    
    return lcm


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM-powered knowledge corpus")
    parser.add_argument("--domains", type=int, default=5, help="Number of domains to generate")
    parser.add_argument("--entries", type=int, default=15, help="Entries per domain")
    parser.add_argument("--seed", type=str, default=None, help="Seed topic for domain generation")
    parser.add_argument("--model", type=str, default="qwen2:latest", help="Ollama model to use")
    parser.add_argument("--specific", type=str, nargs="+", help="Use specific domains instead of generating")
    
    args = parser.parse_args()
    
    if args.specific:
        run_with_specific_domains(args.specific, args.entries, args.model)
    else:
        run_experiment(
            domain_count=args.domains,
            entries_per_domain=args.entries,
            seed_topic=args.seed,
            model=args.model
        )
