"""
Chat Practical Application

Conversational chat interfaces:
- EmergentChat: Legacy gear-based chat (ConversationalChain)
- HyperChat: New HyperMapping-based chat (ChatPipeline)

Components:
- chat.py: Legacy interactive chat interface
- hyper_chat.py: New HyperMapping-based chat interface
- api_server.py: Legacy OpenAI-compatible REST API
- hyper_api.py: New HyperMapping-based API server
- run_api.py: Script to run the API server

Author: Lesley Gushurst
License: GPLv3
"""

from .chat import EmergentChat
from .hyper_chat import HyperChat
from .api_server import create_app as create_legacy_app
from .hyper_api import create_app as create_hyper_app, HyperChatEngine

__all__ = [
    'EmergentChat',
    'HyperChat',
    'create_legacy_app',
    'create_hyper_app',
    'HyperChatEngine',
]
