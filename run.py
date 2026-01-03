#!/usr/bin/env python3
"""
Emergent Chat - Interactive Chat

A conversational AI using emergent intent classification and code generation.

Usage:
    python run.py                    # Interactive chat mode
    python run.py "What is evolution?"  # Single query mode

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))


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
    
    args = parser.parse_args()
    
    from truthspace_lcm.practical_applications.chat.chat import EmergentChat
    chat = EmergentChat()
    
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
