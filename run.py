#!/usr/bin/env python3
"""
TruthSpace LCM - Geometric Language-Code Model

A natural language interface powered by the StackedLCM (128D hierarchical
geometric embeddings). Translates natural language to bash commands and
provides conversational responses.

Usage:
    python run.py                    # Interactive chat mode (default)
    python run.py --legacy           # Legacy TruthSpace mode (12D)
    python run.py "list files"       # Single query mode

The default mode uses the new StackedLCM with:
- 7 hierarchical layers (128 dimensions)
- Intent detection (bash vs chat)
- Bash command execution with safety checks
- No external LLM dependencies
"""

import sys


def main():
    # Check for legacy mode
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
        run_legacy_mode(sys.argv[2:] if len(sys.argv) > 2 else [])
        return
    
    # Default: Run the new chat interface
    from truthspace_lcm.chat import LCMChat
    
    # Single query mode
    if len(sys.argv) > 1:
        chat = LCMChat(safe_mode=False)
        query = " ".join(sys.argv[1:])
        response = chat.process(query)
        if response and response != "EXIT":
            print(response)
        return
    
    # Interactive chat mode
    chat = LCMChat(safe_mode=True)
    chat.run()


def run_legacy_mode(args: list):
    """Run the legacy TruthSpace (12D) mode."""
    import subprocess
    from truthspace_lcm.core import TruthSpace
    
    ts = TruthSpace()
    
    # Handle special arguments
    if args:
        arg = args[0]
        
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
        
        if arg == "--learn" and len(args) > 1:
            cmd = args[1]
            entry = ts.learn_from_man(cmd)
            if entry:
                print(f"✓ Learned: {entry.name} = \"{entry.description}\"")
            else:
                print(f"✗ Could not learn '{cmd}' from man page")
            return
    
    print("=" * 60)
    print("TruthSpace LCM - Legacy Mode (12D)")
    print("=" * 60)
    print()
    
    # Single query mode
    if args:
        query = " ".join(args)
        process_legacy_query(ts, query)
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
        
        process_legacy_query(ts, query)
        print()


def process_legacy_query(ts, query: str):
    """Process a single query using legacy TruthSpace."""
    import subprocess
    
    print(f"\nQuery: \"{query}\"")
    print("-" * 40)
    
    try:
        output, entry, sim = ts.resolve(query)
        print(f"Command: {output}")
        print(f"Match: {entry.description} (similarity: {sim:.2f})")
        print()
        
        response = input("Execute? (y/N): ").strip().lower()
        if response == 'y':
            print()
            print("Output:")
            print("-" * 40)
            try:
                result = subprocess.run(output, shell=True, capture_output=True, text=True, timeout=30)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"[stderr] {result.stderr}")
            except subprocess.TimeoutExpired:
                print("[Timed out after 30s]")
            except Exception as e:
                print(f"[Error: {e}]")
    except Exception as e:
        print(f"Could not resolve: {e}")


if __name__ == "__main__":
    main()
