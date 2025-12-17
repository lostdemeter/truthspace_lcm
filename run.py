#!/usr/bin/env python3
"""
TruthSpace LCM - Hypergeometric Resolution

Translates natural language to bash commands using pure geometry.
No training. No keywords. Just φ-MAX encoding and distance.

Usage:
    python run.py                    # Interactive mode
    python run.py "list files"       # Single query mode
"""

import sys
import subprocess
from truthspace_lcm.core import TruthSpace, KnowledgeGapError


def main():
    ts = TruthSpace()
    
    print("=" * 60)
    print("TruthSpace LCM - Hypergeometric Resolution")
    print("=" * 60)
    print()
    
    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        process_query(ts, query)
        return
    
    # Interactive mode
    print("Enter natural language requests (type 'quit' to exit)")
    print()
    
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if query.lower() == 'explain':
            print("Enter a query to explain:")
            eq = input(">>> ").strip()
            if eq:
                print()
                print(ts.explain(eq))
            print()
            continue
        
        process_query(ts, query)
        print()


def process_query(ts: TruthSpace, query: str):
    """Process a single query."""
    print(f"\nQuery: \"{query}\"")
    print("-" * 40)
    
    try:
        output, entry, similarity = ts.resolve(query)
        
        print(f"Command: {output}")
        print(f"Match: {entry.description} (similarity: {similarity:.2f})")
        print()
        
        # Ask to execute
        response = input("Execute? (y/N): ").strip().lower()
        
        if response == 'y':
            print()
            print("Output:")
            print("-" * 40)
            
            try:
                result = subprocess.run(
                    output,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"[stderr] {result.stderr}")
                if result.returncode != 0:
                    print(f"[Exit code: {result.returncode}]")
            except subprocess.TimeoutExpired:
                print("[Timed out after 30s]")
            except Exception as e:
                print(f"[Error: {e}]")
                
    except KnowledgeGapError as e:
        print(f"No match found (best similarity: {e.best_similarity:.2f})")
        print("\nThe system doesn't know how to do this.")
        print("Add knowledge with: ts.store(name, description)")


if __name__ == "__main__":
    main()
