"""
Chat Practical Application

Conversational chat interfaces:
- EmergentChat: Legacy gear-based chat (ConversationalChain)
- HyperChat: New HyperMapping-based chat (ChatPipeline)

Components:
- chat.py: Legacy interactive chat interface
- hyper_chat.py: New HyperMapping-based chat interface
- api_server.py: OpenAI-compatible REST API
- run_api.py: Script to run the API server

Author: Lesley Gushurst
License: GPLv3
"""

from .chat import EmergentChat
from .hyper_chat import HyperChat
from .api_server import create_app

__all__ = [
    'EmergentChat',
    'HyperChat',
    'create_app',
]
