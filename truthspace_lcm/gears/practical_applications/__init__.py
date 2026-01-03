"""
Practical Applications for the Gear Chain System

This module contains domain-specific gear implementations:
- nlp/: Natural language processing gears and applications
- chat/: Emergent conversational chat (truly emergent responses)

Note: data/ was moved to temp/ as it was only for demonstration purposes.
"""

# Import from submodules
from .nlp.chat import GearChat
from .nlp.api_server import create_app as create_nlp_app
from .chat.chat import EmergentChat
from .chat.api_server import create_app as create_chat_app

__all__ = [
    'GearChat', 
    'create_nlp_app',
    'EmergentChat',
    'create_chat_app',
]
