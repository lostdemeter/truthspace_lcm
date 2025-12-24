"""
Response Templates - Consistent formatting for GeometricLCM responses.

Provides templates for different response types to ensure consistent,
well-formatted output across all handlers.
"""

from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


class ResponseType(Enum):
    """Types of responses the system can generate."""
    ANSWER = "answer"           # Direct answer to a question
    EXPLANATION = "explanation" # Detailed explanation with reasoning
    LIST = "list"               # List of items
    CODE = "code"               # Code output
    CALCULATION = "calculation" # Math result
    ERROR = "error"             # Error message
    GREETING = "greeting"       # Greeting response
    HELP = "help"               # Help/capabilities
    NOT_FOUND = "not_found"     # When information isn't available


@dataclass
class ResponseTemplate:
    """A template for formatting responses."""
    type: ResponseType
    content: str
    metadata: Optional[dict] = None
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.content.format(**kwargs)


class ResponseFormatter:
    """
    Formats responses consistently across the system.
    
    Provides methods for different response types with consistent styling.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> dict[ResponseType, str]:
        """Load response templates."""
        return {
            ResponseType.ANSWER: "{answer}",
            ResponseType.EXPLANATION: "{answer}\n\n**Reasoning:** {reasoning}",
            ResponseType.LIST: "**{title}:**\n{items}",
            ResponseType.CODE: "```{language}\n{code}\n```",
            ResponseType.CALCULATION: "**{expression}** = `{result}`",
            ResponseType.ERROR: "I encountered an issue: {error}",
            ResponseType.GREETING: "{greeting}",
            ResponseType.HELP: "**{title}**\n\n{content}",
            ResponseType.NOT_FOUND: "I don't have information about {topic} in my knowledge base.",
        }
    
    def format_answer(self, answer: str, confidence: Optional[float] = None) -> str:
        """Format a direct answer."""
        result = answer
        if confidence is not None and confidence < 0.5:
            result = f"I'm not entirely certain, but: {answer}"
        return result
    
    def format_explanation(self, answer: str, reasoning: str) -> str:
        """Format an answer with reasoning path."""
        return self.templates[ResponseType.EXPLANATION].format(
            answer=answer,
            reasoning=reasoning
        )
    
    def format_list(self, title: str, items: list[str], numbered: bool = False) -> str:
        """Format a list of items."""
        if numbered:
            formatted_items = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        else:
            formatted_items = "\n".join(f"• {item}" for item in items)
        
        return self.templates[ResponseType.LIST].format(
            title=title,
            items=formatted_items
        )
    
    def format_code(self, code: str, language: str = "python", 
                    description: Optional[str] = None) -> str:
        """Format code output."""
        result = self.templates[ResponseType.CODE].format(
            language=language,
            code=code
        )
        if description:
            result = f"{description}\n\n{result}"
        return result
    
    def format_calculation(self, expression: str, result: Any) -> str:
        """Format a calculation result."""
        return self.templates[ResponseType.CALCULATION].format(
            expression=expression,
            result=result
        )
    
    def format_error(self, error: str, suggestion: Optional[str] = None) -> str:
        """Format an error message."""
        result = self.templates[ResponseType.ERROR].format(error=error)
        if suggestion:
            result += f"\n\n**Suggestion:** {suggestion}"
        return result
    
    def format_not_found(self, topic: str, alternatives: Optional[list[str]] = None) -> str:
        """Format a not-found response."""
        result = self.templates[ResponseType.NOT_FOUND].format(topic=topic)
        if alternatives:
            result += f"\n\nYou might try asking about: {', '.join(alternatives)}"
        return result
    
    def format_entity_info(self, name: str, description: str, 
                           attributes: Optional[dict] = None,
                           relationships: Optional[list[str]] = None) -> str:
        """Format information about an entity (character, concept, etc.)."""
        parts = [f"**{name}** - {description}"]
        
        if attributes:
            attr_lines = [f"• **{k}**: {v}" for k, v in attributes.items()]
            parts.append("\n" + "\n".join(attr_lines))
        
        if relationships:
            parts.append("\n**Relationships:**")
            parts.append("\n".join(f"• {r}" for r in relationships))
        
        return "\n".join(parts)
    
    def format_comparison(self, items: list[dict], title: Optional[str] = None) -> str:
        """
        Format a comparison between items.
        
        Each item should have 'name' and other comparable attributes.
        """
        if not items:
            return "No items to compare."
        
        parts = []
        if title:
            parts.append(f"**{title}**\n")
        
        for item in items:
            name = item.get('name', 'Unknown')
            attrs = {k: v for k, v in item.items() if k != 'name'}
            
            parts.append(f"**{name}:**")
            for k, v in attrs.items():
                parts.append(f"  • {k}: {v}")
            parts.append("")
        
        return "\n".join(parts)
    
    def format_multi_part(self, responses: list[str], separator: str = "\n\n") -> str:
        """Format multiple response parts into one."""
        return separator.join(r for r in responses if r)


# Global instance
_formatter: Optional[ResponseFormatter] = None

def get_formatter() -> ResponseFormatter:
    """Get the global response formatter."""
    global _formatter
    if _formatter is None:
        _formatter = ResponseFormatter()
    return _formatter
