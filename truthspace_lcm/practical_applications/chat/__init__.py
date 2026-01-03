"""
Chat Practical Application

Emergent conversational chat using the ConversationalChain.

Components:
- chat.py: Interactive chat interface
- api_server.py: OpenAI-compatible REST API
- run_api.py: Script to run the API server

Author: Lesley Gushurst
License: GPLv3
"""

from .chat import EmergentChat
from .api_server import create_app

__all__ = [
    'EmergentChat',
    'create_app',
]
