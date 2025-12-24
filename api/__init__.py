"""
GeometricLCM API - OpenAI-Compatible API Server

This module provides an OpenAI-compatible API for GeometricLCM,
allowing it to be used with tools like Open WebUI.
"""

from .server import app
from .models import ChatRequest, ChatResponse, ChatMessage

__all__ = ['app', 'ChatRequest', 'ChatResponse', 'ChatMessage']
