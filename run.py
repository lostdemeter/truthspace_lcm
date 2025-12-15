#!/usr/bin/env python3
"""
TruthSpace LCM - Interactive Runner

A simple interface to test the TruthSpace system.
Translates natural language to code and executes it.

Usage:
    python run.py                    # Interactive mode
    python run.py "list files"       # Single query mode
"""

import sys
from truthspace_lcm import TruthSpace, Resolver, KnowledgeGapError


def main():
    # Initialize
    ts = TruthSpace()
    resolver = Resolver(ts, auto_learn=True)
    
    print("=" * 60)
    print("TruthSpace LCM - Natural Language to Code")
    print("=" * 60)
    print()
    
    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        process_query(resolver, query)
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
        
        process_query(resolver, query)
        print()


def process_query(resolver: Resolver, query: str):
    """Process a single query and display results."""
    print(f"\nRequest: \"{query}\"")
    print("-" * 40)
    
    try:
        # Resolve
        result = resolver.resolve(query)
        
        print(f"Generated ({result.output_type.value}):")
        print(f"  {result.output}")
        print()
        
        # Ask to execute
        if result.output_type.value in ('bash', 'python'):
            response = input("Execute? (y/N): ").strip().lower()
            
            if response == 'y':
                print()
                print("Output:")
                print("-" * 40)
                
                resolution, exec_result = resolver.resolve_and_execute(query)
                
                if exec_result.stdout:
                    print(exec_result.stdout)
                if exec_result.stderr:
                    print(f"[stderr] {exec_result.stderr}")
                
                if not exec_result.success:
                    print(f"[Exit code: {exec_result.return_code}]")
        
        if result.learned:
            print(f"\n[Auto-learned: {result.knowledge.name}]")
            
    except KnowledgeGapError as e:
        print(f"Knowledge gap: {e.query}")
        print(f"Best match similarity: {e.best_match:.2f}")
        print("\nThe system doesn't know how to do this yet.")
        print("You can teach it by adding knowledge to TruthSpace.")


if __name__ == "__main__":
    main()
