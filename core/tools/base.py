"""
Base tool interface and registry for GeometricLCM.

Tools are executable capabilities that extend what the model can do.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def success_result(cls, output: Any, **metadata) -> 'ToolResult':
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata)
    
    @classmethod
    def error_result(cls, error: str) -> 'ToolResult':
        """Create an error result."""
        return cls(success=False, output=None, error=error)


class Tool(ABC):
    """
    Base class for all tools.
    
    Tools are executable capabilities that can be invoked by name.
    Each tool declares what it can do and how to invoke it.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used for invocation)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @property
    def triggers(self) -> list[str]:
        """
        Patterns that trigger this tool.
        
        These are phrases that indicate the user wants to use this tool.
        """
        return []
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> ToolResult:
        """
        Execute the tool with the given query.
        
        Args:
            query: The user's request
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with the output or error
        """
        pass
    
    def matches(self, query: str) -> float:
        """
        Return confidence (0-1) that this tool should handle the query.
        
        Default implementation checks for trigger patterns.
        """
        query_lower = query.lower()
        
        for trigger in self.triggers:
            if trigger in query_lower:
                return 0.9
        
        return 0.0
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


class ToolRegistry:
    """
    Registry of available tools.
    
    The registry maintains a collection of tools and can find
    the best tool for a given query.
    """
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def unregister(self, name: str):
        """Unregister a tool by name."""
        if name in self.tools:
            del self.tools[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def find_best_tool(self, query: str) -> Optional[tuple[Tool, float]]:
        """
        Find the best tool for a query.
        
        Returns:
            Tuple of (tool, confidence) or None if no tool matches
        """
        best_tool = None
        best_confidence = 0.0
        
        for tool in self.tools.values():
            confidence = tool.matches(query)
            if confidence > best_confidence:
                best_confidence = confidence
                best_tool = tool
        
        if best_tool and best_confidence > 0:
            return (best_tool, best_confidence)
        return None
    
    def list_tools(self) -> list[dict]:
        """List all registered tools with their descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "triggers": tool.triggers,
            }
            for tool in self.tools.values()
        ]
    
    def __len__(self) -> int:
        return len(self.tools)
    
    def __iter__(self):
        return iter(self.tools.values())


# Global registry instance
_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
