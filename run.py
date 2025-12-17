#!/usr/bin/env python3
"""
TruthSpace LCM - Hypergeometric Resolution

Translates natural language to bash commands using pure geometry.
No training. No keywords. Just φ-MAX encoding and distance.

Usage:
    python run.py                    # Interactive mode
    python run.py "list files"       # Single query mode
    python run.py --reset            # Reset learned knowledge
    python run.py --learned          # Show learned knowledge
    python run.py --learn <cmd>      # Learn a command from man page
"""

import sys
import subprocess
from truthspace_lcm.core import TruthSpace, KnowledgeGapError


def main():
    ts = TruthSpace()
    
    # Handle special arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--reset":
            ts.reset()
            print("✓ Database reset. All learned knowledge removed.")
            print(f"  Bootstrap entries: {len(ts.entries)}")
            return
        
        if arg == "--learned":
            learned = ts.list_learned()
            if learned:
                print(f"Learned knowledge ({len(learned)} entries):")
                for cmd, desc in learned:
                    print(f"  {cmd}: \"{desc}\"")
            else:
                print("No learned knowledge yet.")
            return
        
        if arg == "--learn" and len(sys.argv) > 2:
            cmd = sys.argv[2]
            entry = ts.learn_from_man(cmd)
            if entry:
                print(f"✓ Learned: {entry.name} = \"{entry.description}\"")
            else:
                print(f"✗ Could not learn '{cmd}' from man page")
            return
    
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
    """Process a single query using compound resolution."""
    print(f"\nQuery: \"{query}\"")
    print("-" * 40)
    
    # Use compound resolution to extract multiple concepts
    concepts = ts.resolve_with_params(query)
    
    if not concepts:
        print("No concepts found in query.")
        print()
        
        # Try to learn from man pages
        print("Attempting to learn from man pages...")
        entry = ts.try_learn(query)
        
        if entry:
            print(f"✓ Learned: {entry.name} = \"{entry.description}\"")
            # Retry
            concepts = ts.resolve_with_params(query)
        
        if not concepts:
            print("Could not resolve query.")
            return
    
    # Build full commands with parameters
    commands = []
    print("Concepts extracted:")
    for c in concepts:
        cmd = c['command']
        if c['params']:
            cmd += ' ' + ' '.join(c['params'])
        commands.append(cmd)
        print(f"  • \"{c['window']}\" → {cmd} ({c['similarity']:.2f})")
    
    full_command = ' && '.join(commands)
    print()
    print(f"Command: {full_command}")
    print()
    
    # Ask to execute
    response = input("Execute? (y/N): ").strip().lower()
    
    if response == 'y':
        print()
        print("Output:")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                full_command,
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


if __name__ == "__main__":
    main()
