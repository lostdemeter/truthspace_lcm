"""
Chat Handler - Conversational responses, greetings, and meta-queries.

Handles greetings, farewells, questions about capabilities,
and general conversational interactions.
"""

from .base import Handler, HandlerResult, Context, Intent
from ..self_knowledge import get_self_knowledge


class ChatHandler(Handler):
    """Handler for conversational interactions."""
    
    def __init__(self):
        """Initialize chat handler with self-knowledge."""
        self.self_knowledge = get_self_knowledge()
        
        self.farewells = [
            "Goodbye! Feel free to return anytime.",
            "Take care! Happy to help again.",
            "See you later!",
        ]
    
    @property
    def name(self) -> str:
        return "chat"
    
    @property
    def supported_intents(self) -> list[Intent]:
        return [Intent.GREETING, Intent.FAREWELL, Intent.META, Intent.GENERAL]
    
    def can_handle(self, context: Context) -> float:
        """Check if this is a conversational request."""
        if context.intent in [Intent.GREETING, Intent.FAREWELL, Intent.META]:
            return 0.95
        
        if context.intent == Intent.GENERAL:
            return 0.5  # Fallback handler
        
        return 0.1  # Can always provide a fallback
    
    def handle(self, context: Context) -> HandlerResult:
        """Handle conversational request."""
        if context.intent == Intent.GREETING:
            return self._handle_greeting(context)
        elif context.intent == Intent.FAREWELL:
            return self._handle_farewell(context)
        elif context.intent == Intent.META:
            return self._handle_meta(context)
        else:
            return self._handle_general(context)
    
    def _handle_greeting(self, context: Context) -> HandlerResult:
        """Handle greeting."""
        sk = self.self_knowledge
        greeting = (
            f"Hello! I'm {sk.identity['name']}, a {sk.identity['full_name'].lower()}. "
            f"I can answer questions, generate code, create charts, and execute tasks. "
            f"How can I help you today?"
        )
        return HandlerResult.success_result(greeting, confidence=1.0)
    
    def _handle_farewell(self, context: Context) -> HandlerResult:
        """Handle farewell."""
        return HandlerResult.success_result(self.farewells[0], confidence=1.0)
    
    def _handle_meta(self, context: Context) -> HandlerResult:
        """Handle meta-questions about capabilities."""
        # Try self-knowledge first
        answer = self.self_knowledge.answer_meta_question(context.message)
        if answer:
            return HandlerResult.success_result(answer, confidence=1.0)
        
        # Default: full introduction
        return HandlerResult.success_result(
            self.self_knowledge.get_full_introduction(), 
            confidence=0.8
        )
    
    def _handle_general(self, context: Context) -> HandlerResult:
        """Handle general/fallback requests."""
        sk = self.self_knowledge
        return HandlerResult.success_result(
            f"I'm not sure how to help with that specific request. "
            f"I can answer questions about my knowledge base, generate Python code, "
            f"execute calculations, or create charts. What would you like to try?",
            confidence=0.3
        )
