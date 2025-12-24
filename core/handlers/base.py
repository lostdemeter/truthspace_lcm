"""
Base handler interface for GeometricLCM.

All handlers implement this interface to provide modular request processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class Intent(Enum):
    """Classification of user intent."""
    QUESTION = "question"
    CODE = "code"
    EXECUTE = "execute"
    CHART = "chart"
    GREETING = "greeting"
    FAREWELL = "farewell"
    META = "meta"
    GENERAL = "general"


@dataclass
class HandlerResult:
    """Result from a handler."""
    success: bool
    response: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def success_result(cls, response: str, confidence: float = 1.0, **metadata) -> 'HandlerResult':
        """Create a successful result."""
        return cls(success=True, response=response, confidence=confidence, metadata=metadata)
    
    @classmethod
    def failure_result(cls, reason: str) -> 'HandlerResult':
        """Create a failure result."""
        return cls(success=False, response=reason, confidence=0.0)


@dataclass
class Context:
    """Context for request processing."""
    message: str
    intent: Intent
    history: list = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Handler(ABC):
    """
    Base class for all handlers.
    
    Handlers are responsible for processing specific types of requests.
    Each handler declares what intents it can handle and processes
    messages accordingly.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging/debugging."""
        pass
    
    @property
    @abstractmethod
    def supported_intents(self) -> list[Intent]:
        """List of intents this handler can process."""
        pass
    
    @abstractmethod
    def can_handle(self, context: Context) -> float:
        """
        Return confidence (0-1) that this handler can process the request.
        
        Higher confidence means this handler is more suitable.
        Return 0 if the handler cannot handle this request.
        """
        pass
    
    @abstractmethod
    def handle(self, context: Context) -> HandlerResult:
        """
        Process the request and return a result.
        
        This is called only if can_handle returned > 0.
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
