"""
TruthSpace Gear System

A modular, extensible gear chain architecture for transformation pipelines.

The gear system is organized into:
- core/: Base classes (Gear, GearState, GearChain, Quaternion) and core functionality
- corpus/: Corpus management utilities
- tools/: Corpus pruning, correction, and reinforcement tools
- practical_applications/: Domain-specific implementations
  - chat/: Emergent conversational chat (truly emergent responses)

Usage (Chat):
    from truthspace_lcm.practical_applications.chat import EmergentChat
    
    chat = EmergentChat()
    response = chat.chat("Hello, how are you?")

Usage (API Server):
    python -m truthspace_lcm.practical_applications.chat.run_api --port 8002

Corpus management:
    from truthspace_lcm.corpus import load_corpus, save_corpus
    
Tools:
    from truthspace_lcm.tools import CorpusPruner, CorpusCorrector, CorpusReinforcer
"""

# Re-export core classes for convenience
from .core import Gear, GearState, GearChain, Quaternion

# Re-export chat application
from .practical_applications.chat import EmergentChat

__all__ = [
    # Core classes
    'Gear',
    'GearState', 
    'GearChain',
    'Quaternion',
    # Chat application
    'EmergentChat',
]
