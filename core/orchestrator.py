"""
Orchestrator - Routes requests to appropriate handlers.

The orchestrator is the "brain" of GeometricLCM. It:
1. Classifies user intent
2. Routes to the best handler
3. Manages conversation state
4. Composes the final response
"""

from typing import Optional
from .handlers.base import Handler, HandlerResult, Context, Intent
from .self_knowledge import get_self_knowledge
from .query_resolver import get_query_resolver, ResolvedQuery
from .conversation_context import get_conversation_context


class IntentClassifier:
    """Classify user intent to route to appropriate handler."""
    
    def classify(self, message: str) -> Intent:
        """Classify the intent of a message."""
        msg_lower = message.lower().strip()
        # Remove trailing punctuation for matching
        msg_clean = msg_lower.rstrip('!?.,:;')
        
        # Greeting patterns
        greetings = ["hello", "hi", "hey", "greetings", "good morning", 
                     "good afternoon", "good evening", "howdy"]
        if any(msg_clean == g or msg_lower.startswith(g + " ") or msg_lower.startswith(g + ",") 
               for g in greetings):
            return Intent.GREETING
        
        # Farewell patterns
        farewells = ["bye", "goodbye", "see you", "farewell", "take care", 
                     "thanks", "thank you", "cheers"]
        if any(msg_lower.startswith(f) for f in farewells):
            return Intent.FAREWELL
        
        # Meta patterns (asking about capabilities)
        meta_patterns = ['what can you do', 'what are you', 'who are you', 
                         'help me', 'your capabilities', 'what do you know',
                         'tell me about yourself', 'how do you work']
        if any(p in msg_lower for p in meta_patterns):
            return Intent.META
        
        # Code patterns
        code_patterns = ['write a function', 'create a function', 'generate code',
                         'write code', 'python function', 'write a script',
                         'code that', 'function to', 'function that',
                         'implement a', 'write a program']
        if any(p in msg_lower for p in code_patterns):
            return Intent.CODE
        
        # Chart patterns
        chart_patterns = ['chart', 'graph', 'plot', 'visualize', 'visualization',
                          'bar chart', 'line chart', 'pie chart', 'histogram']
        if any(p in msg_lower for p in chart_patterns):
            return Intent.CHART
        
        # Execute patterns
        execute_patterns = ['calculate', 'compute', 'find the sum', 'find the average',
                            'sum of', 'average of', 'sort', 'filter', 'count']
        if any(p in msg_lower for p in execute_patterns):
            return Intent.EXECUTE
        
        # Question patterns
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 
                          'is ', 'are ', 'does ', 'did ', 'can ', 'could ']
        if any(msg_lower.startswith(q) for q in question_words) or '?' in message:
            return Intent.QUESTION
        
        return Intent.GENERAL


class Orchestrator:
    """
    Routes requests to appropriate handlers and manages conversation flow.
    
    The orchestrator maintains a list of handlers and selects the best one
    for each request based on intent classification and handler confidence.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.handlers: list[Handler] = []
        self.classifier = IntentClassifier()
        self.history: list[dict] = []
        self.self_knowledge = get_self_knowledge()
        self.default_system_prompt = self.self_knowledge.get_system_prompt()
        self.query_resolver = get_query_resolver()
        self.conversation_context = get_conversation_context()
    
    def register_handler(self, handler: Handler):
        """Register a handler."""
        self.handlers.append(handler)
    
    def process(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Process a message and return a response.
        
        Handles compound queries by splitting them and processing each part.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt
            
        Returns:
            The response string
        """
        # Resolve compound queries
        resolved = self.query_resolver.resolve(message)
        
        if resolved.is_compound:
            # Process each sub-query and combine results
            response = self._process_compound(resolved, system_prompt)
        else:
            # Single query - process normally
            response = self._process_single(message, system_prompt)
        
        # Update history
        self.history.append({
            "role": "user",
            "content": message
        })
        self.history.append({
            "role": "assistant", 
            "content": response
        })
        
        # Update conversation context
        self.conversation_context.add_turn(message, response)
        
        # Keep history manageable
        if len(self.history) > 40:
            self.history = self.history[-40:]
        
        return response
    
    def _process_single(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Process a single (non-compound) query."""
        # Classify intent
        intent = self.classifier.classify(message)
        
        # Create context
        context = Context(
            message=message,
            intent=intent,
            history=self.history.copy(),
            system_prompt=system_prompt
        )
        
        # Find best handler
        best_handler = None
        best_confidence = 0.0
        
        for handler in self.handlers:
            confidence = handler.can_handle(context)
            if confidence > best_confidence:
                best_confidence = confidence
                best_handler = handler
        
        # Process with best handler
        if best_handler and best_confidence > 0:
            result = best_handler.handle(context)
        else:
            result = HandlerResult.failure_result(
                "I'm not sure how to help with that. Try asking a question, "
                "requesting code, or asking what I can do."
            )
        
        return result.response
    
    def _process_compound(self, resolved: ResolvedQuery, system_prompt: Optional[str] = None) -> str:
        """
        Process a compound query by handling each sub-query.
        
        Args:
            resolved: The resolved compound query
            system_prompt: Optional system prompt
            
        Returns:
            Combined response string
        """
        responses = []
        
        for sub_query in resolved.sub_queries:
            # Use resolved text (with pronouns replaced) if available
            query_text = sub_query.final_text
            
            # Process this sub-query
            response = self._process_single(query_text, system_prompt)
            responses.append(response)
        
        # Combine responses
        if len(responses) == 1:
            return responses[0]
        
        # Join with double newline separator
        return "\n\n".join(responses)
    
    def clear_history(self):
        """Clear conversation history and context."""
        self.history = []
        self.conversation_context.clear()
    
    def get_context_summary(self) -> dict:
        """Get a summary of the current conversation context."""
        return self.conversation_context.get_context_summary()
    
    def get_handler_for_intent(self, intent: Intent) -> Optional[Handler]:
        """Get the primary handler for an intent."""
        for handler in self.handlers:
            if intent in handler.supported_intents:
                return handler
        return None
