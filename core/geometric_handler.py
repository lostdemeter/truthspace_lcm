"""
Geometric Knowledge Handler for API Server

A simplified handler that uses the fully geometric QA system.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class HandlerResult:
    """Result from a handler."""
    handled: bool
    response: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GeometricKnowledgeHandler:
    """
    Handler for knowledge queries using geometric QA.
    
    Uses fully geometric approach:
    - Geometric stop word detection
    - Position-based frame extraction
    - Learned morphology and conjugation
    """
    
    name = "geometric_knowledge"
    priority = 10
    
    def __init__(self, qa):
        """
        Initialize with GeometricQA instance.
        
        Args:
            qa: GeometricQA instance
        """
        self.qa = qa
    
    def can_handle(self, query: str) -> bool:
        """Check if this handler can process the query."""
        query_lower = query.lower()
        
        # Knowledge questions
        knowledge_patterns = [
            'who is', 'who was', 'who are',
            'what is', 'what was', 'what are',
            'what does', 'what did',
            'where is', 'where was',
            'when did', 'when was',
            'why did', 'why does',
            'how did', 'how does',
            'tell me about', 'describe',
            'who killed', 'who loves', 'who went',
        ]
        
        for pattern in knowledge_patterns:
            if pattern in query_lower:
                return True
        
        # Check if query mentions known entities
        tokens = query_lower.split()
        for token in tokens:
            if token in self.qa.knowledge.concepts:
                c = self.qa.knowledge.concepts[token]
                if c.is_content_word and c.initiator_count > 0:
                    return True
        
        return False
    
    def handle(self, query: str, context: Dict[str, Any] = None) -> HandlerResult:
        """
        Handle a knowledge query.
        
        Args:
            query: The user's query
            context: Optional context from previous turns
            
        Returns:
            HandlerResult with the response
        """
        if not self.can_handle(query):
            return HandlerResult(handled=False)
        
        # Use geometric QA
        result = self.qa.ask_detailed(query)
        
        if result['answers']:
            best = result['answers'][0]
            return HandlerResult(
                handled=True,
                response=best['answer'],
                confidence=best['confidence'],
                metadata={
                    'axis': result['axis'],
                    'entity': result['entity'],
                    'action': result['action'],
                    'source': 'geometric',
                }
            )
        
        return HandlerResult(
            handled=True,
            response="I don't have information about that.",
            confidence=0.3,
            metadata={'source': 'geometric'}
        )


__all__ = ['GeometricKnowledgeHandler', 'HandlerResult']
