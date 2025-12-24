"""
GeometricLCM Handlers - Modular request processing.

Each handler is responsible for a specific type of request:
- KnowledgeHandler: Q&A, reasoning, holographic generation
- CodeHandler: Code generation and explanation
- ToolHandler: Task execution, charts, calculations
- ChatHandler: Conversational responses, greetings, meta-queries
"""

from .base import Handler, HandlerResult
from .knowledge import KnowledgeHandler
from .code import CodeHandler
from .tools import ToolHandler
from .chat import ChatHandler

__all__ = [
    'Handler',
    'HandlerResult',
    'KnowledgeHandler',
    'CodeHandler',
    'ToolHandler',
    'ChatHandler',
]
