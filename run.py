#!/usr/bin/env python3
"""
TruthSpace LCM - Geometric Language Model

A conversational AI using fully geometric language understanding.
No training, no neural networks, no hard-coded rules - just geometry.

Core Principle: All semantic operations are geometric operations in concept space.

Architecture:
    Surface Text (any language)
            ↓
    Position-Based Frame Extraction
            ↓
    Holographic Template Projection + Semantic Quaternion
            ↓
    φ-Dial Styled Response
            ↓
    Answer

Usage:
    python run.py                    # Interactive chat mode
    python run.py test               # Run test suite
    python run.py "Who is Darcy?"    # Single query mode
    python run.py --debug            # Debug mode (show concept frames)

Features:
- Geometric stop word detection (no hard-coded lists)
- Position-based frame extraction
- Morphology learned from parallel structures
- Holographic template projection for dynamic responses
- Semantic quaternions for analogies (100% accuracy)
- Two quaternions: φ-dial (output) + semantic (encoding)
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
    
    # Single query mode
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        query = ' '.join(sys.argv[1:])
        from truthspace_lcm.core import HolographicGeometricQA
        qa = HolographicGeometricQA()
        
        # Try multiple corpus locations (relative to script location)
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir / 'truthspace_lcm' / 'sample_corpus_geometric.json',
            script_dir / 'truthspace_lcm' / 'concept_corpus_quality.json',
            script_dir / 'truthspace_lcm' / 'concept_corpus.json',
        ]
        
        loaded = False
        for corpus_path in possible_paths:
            if corpus_path.exists():
                count = qa.load_corpus(str(corpus_path))
                if count > 0:
                    loaded = True
                    break
        
        answer = qa.ask(query)
        print(answer)
        sys.exit(0)
    
    # Default: Interactive chat mode
    from truthspace_lcm.chat import main as chat_main
    sys.exit(chat_main())


if __name__ == "__main__":
    main()
