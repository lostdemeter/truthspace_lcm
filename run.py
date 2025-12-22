#!/usr/bin/env python3
"""
TruthSpace LCM - Dynamic Geometric Language Model

A conversational AI using pure geometric operations in semantic space.
No training, no neural networks - just geometry.

Core Principle: Structure IS the data. Learning IS structure update.

Usage:
    python run.py                    # Interactive chat mode
    python run.py demo               # Run quick demo
    python run.py "question"         # Single query mode
    python run.py test               # Run test suite

Features:
- Dynamic learning from natural language
- Relational queries (What is the capital of France?)
- Analogical reasoning (france:paris :: germany:?)
- Multi-hop reasoning (path finding)
- 256-dimensional semantic space
"""

import sys


def main():
    from truthspace_lcm.chat import GeometricChat, demo
    
    # Demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
        return
    
    # Test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import subprocess
        print("Running test suite...")
        result1 = subprocess.run([sys.executable, "tests/test_core.py"])
        result2 = subprocess.run([sys.executable, "tests/test_chat.py"])
        sys.exit(result1.returncode or result2.returncode)
    
    # Single query mode
    if len(sys.argv) > 1:
        chat = GeometricChat()
        query = " ".join(sys.argv[1:])
        response = chat.process(query)
        if response and response != "QUIT":
            print(response)
        return
    
    # Interactive chat mode
    chat = GeometricChat()
    chat.run()


if __name__ == "__main__":
    main()
