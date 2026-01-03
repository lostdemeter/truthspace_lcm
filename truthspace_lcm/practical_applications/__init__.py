"""
Practical Applications for the Gear Chain System

This module contains domain-specific gear implementations:
- chat/: Emergent conversational chat (truly emergent responses)

Note: nlp/ and data/ were moved to temp/ as legacy code.
"""

# Import from submodules
from .chat.chat import EmergentChat
from .chat.api_server import create_app as create_chat_app

__all__ = [
    'EmergentChat',
    'create_chat_app',
]
