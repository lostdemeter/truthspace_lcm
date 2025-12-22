#!/usr/bin/env python3
"""
TruthSpace LCM - Holographic Concept Language Model

A conversational AI using holographic concept resolution.
No training, no neural networks - just geometry.

Core Principle: All semantic operations are geometric operations in concept space.

Architecture:
    Surface Text (any language)
            ↓
    Concept Frame (order-free, language-agnostic)
            ↓
    Holographic Projection (fill the gap)
            ↓
    Answer

Usage:
    python run.py                    # Interactive chat mode
    python run.py test               # Run test suite
    python run.py "Who is Darcy?"    # Single query mode
    python run.py --debug            # Debug mode (show concept frames)

Features:
- Language-agnostic concept extraction
- Holographic Q&A resolution
- Cross-language knowledge queries
- 64-dimensional semantic space
"""

import sys
from pathlib import Path


def main():
    # Test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import subprocess
        print("Running test suite...")
        result1 = subprocess.run([sys.executable, "tests/test_core.py"])
        result2 = subprocess.run([sys.executable, "tests/test_chat.py"])
        sys.exit(result1.returncode or result2.returncode)
    
    # Import chat module
    from truthspace_lcm.chat import main as chat_main
    
    # Pass through to chat module
    sys.exit(chat_main())


if __name__ == "__main__":
    main()
