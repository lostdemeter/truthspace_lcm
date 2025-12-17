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


def get_social_response(social_type: str) -> str:
    """Get an appropriate response prefix based on social context."""
    responses = {
        'GREETING': "Hello! ",
        'ACKNOWLEDGMENT': "Sounds good. ",
        'POLITENESS': "Of course! ",
        'QUERY_INTENT': "Sure, ",
        'FILLER': "",  # Fillers don't need acknowledgment
    }
    return responses.get(social_type, "")


def process_query(ts: TruthSpace, query: str):
    """Process a single query using compound resolution."""
    print(f"\nQuery: \"{query}\"")
    print("-" * 40)
    
    # Use compound resolution to extract multiple concepts
    result = ts.resolve_with_params(query)
    
    # Handle new structured return format
    if isinstance(result, dict):
        concepts = result.get('concepts', [])
        social = result.get('social', {})
    else:
        # Fallback for old format
        concepts = result
        social = {}
    
    # Get social response prefix
    social_prefix = ""
    if social.get('has_social'):
        social_prefix = get_social_response(social['social_type'])
    
    if not concepts:
        # Check if this is a pure social query (greeting only, no command)
        if social.get('has_social') and not social.get('command_content', '').strip():
            # Pure social interaction - respond appropriately
            if social['social_type'] == 'GREETING':
                print("Hello! How can I help you today?")
            elif social['social_type'] == 'ACKNOWLEDGMENT':
                print("Ready when you are.")
            elif social['social_type'] == 'POLITENESS':
                print("You're welcome! What would you like to do?")
            else:
                print("I'm listening. What would you like to do?")
            return
        
        print("No concepts found in query.")
        print()
        
        # Try to learn from man pages
        print("Attempting to learn from man pages...")
        entry = ts.try_learn(query)
        
        if entry:
            print(f"✓ Learned: {entry.name} = \"{entry.description}\"")
            # Retry
            result = ts.resolve_with_params(query)
            if isinstance(result, dict):
                concepts = result.get('concepts', [])
        
        if not concepts:
            print("Could not resolve query.")
            return
    
    # Build full commands with parameters
    commands = []
    print(f"{social_prefix}Here's what I found:")
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
