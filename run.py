#!/usr/bin/env python3
"""
TruthSpace LCM - Geometric Chat System

A conversational AI using pure geometric operations in semantic space.
No training, no neural networks - just geometry.

Usage:
    python run.py                    # Interactive chat mode
    python run.py demo               # Run quick demo
    python run.py "question"         # Single query mode

Features:
- Q&A using semantic similarity (cosine distance)
- Style extraction, classification, and transfer
- Knowledge ingestion from text
- 64-dimensional semantic space
"""

import sys


def main():
    from truthspace_lcm.chat import GeometricChat, demo
    
    # Demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
        return
    
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
