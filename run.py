#!/usr/bin/env python3
"""
Emergent Chat - Interactive Chat

A conversational AI using emergent intent classification and code generation.

Usage:
    python run.py                        # Interactive chat (no corpus)
    python run.py --demo                 # Interactive chat with demo corpus
    python run.py --demo "What is AI?"   # Single query with demo corpus

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_TOPICS = [
    "artificial intelligence",
    "machine learning", 
    "programming",
    "python",
    "science",
]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Emergent Chat - Interactive Mode"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional query to process (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build demo corpus from default topics before starting",
    )
    parser.add_argument(
        "--topics", "-t",
        nargs="+",
        help="Custom topics to build corpus from (implies --demo)",
    )
    
    args = parser.parse_args()
    
    from truthspace_lcm.practical_applications.chat.chat import EmergentChat
    chat = EmergentChat()
    
    # Build corpus if --demo or --topics specified
    if args.demo or args.topics:
        topics = args.topics or DEFAULT_TOPICS
        print("=" * 60)
        print("EMERGENT CHAT")
        print("=" * 60)
        print(f"Building corpus from topics: {', '.join(topics)}")
        chat.build_knowledge(topics)
        print()
    
    # Single query mode
    if args.query:
        query = " ".join(args.query)
        response = chat.query(query)
        print(response)
        return 0
    
    # Interactive mode - use the built-in run() method
    chat.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
