#!/usr/bin/env python3
"""
Gear Chain System - Interactive Chat

A conversational AI using the modular gear chain architecture.
Each gear adds dimensions that can be swapped, tuned, or replaced at runtime.

Usage:
    python run.py                    # Interactive chat mode
    python run.py "What is evolution?"  # Single query mode
    python run.py --debug            # Debug mode (show gear transformations)

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gear Chain Chat - Modular Language Understanding"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional query to process (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode to see gear transformations",
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default="experimental",
        help="Corpus to use (default: experimental)",
    )
    
    args = parser.parse_args()
    
    # Single query mode
    if args.query:
        query = " ".join(args.query)
        from truthspace_lcm.gears.practical_applications.nlp.chat import GearChat
        chat = GearChat(corpus_name=args.corpus, debug=args.debug)
        response = chat.query(query)
        print(response)
        return 0
    
    # Interactive mode
    from truthspace_lcm.gears.practical_applications.nlp.chat import main as chat_main
    return chat_main(debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())
