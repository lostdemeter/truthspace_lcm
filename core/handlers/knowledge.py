"""
Knowledge Handler - Q&A, reasoning, and holographic generation.

Handles questions about the knowledge base, multi-hop reasoning,
and holographic text generation.
"""

from .base import Handler, HandlerResult, Context, Intent
from ..response_templates import get_formatter


class KnowledgeHandler(Handler):
    """Handler for knowledge-based queries."""
    
    def __init__(self, qa=None, reasoning=None, hologen=None):
        """
        Initialize with GeometricLCM components.
        
        Args:
            qa: ConceptQA instance
            reasoning: ReasoningEngine instance
            hologen: HolographicGenerator instance
        """
        self.qa = qa
        self.reasoning = reasoning
        self.hologen = hologen
        self.formatter = get_formatter()
        
        # Known entities for better responses
        self.known_entities = {
            'holmes': {'name': 'Sherlock Holmes', 'domain': 'Sherlock Holmes stories'},
            'sherlock': {'name': 'Sherlock Holmes', 'domain': 'Sherlock Holmes stories'},
            'watson': {'name': 'Dr. John Watson', 'domain': 'Sherlock Holmes stories'},
            'moriarty': {'name': 'Professor Moriarty', 'domain': 'Sherlock Holmes stories'},
            'irene': {'name': 'Irene Adler', 'domain': 'Sherlock Holmes stories'},
            'mycroft': {'name': 'Mycroft Holmes', 'domain': 'Sherlock Holmes stories'},
            'lestrade': {'name': 'Inspector Lestrade', 'domain': 'Sherlock Holmes stories'},
            'darcy': {'name': 'Mr. Darcy', 'domain': 'Pride and Prejudice'},
            'elizabeth': {'name': 'Elizabeth Bennet', 'domain': 'Pride and Prejudice'},
            'jane': {'name': 'Jane Bennet', 'domain': 'Pride and Prejudice'},
            'bingley': {'name': 'Mr. Bingley', 'domain': 'Pride and Prejudice'},
            'wickham': {'name': 'Mr. Wickham', 'domain': 'Pride and Prejudice'},
            'lydia': {'name': 'Lydia Bennet', 'domain': 'Pride and Prejudice'},
        }
    
    @property
    def name(self) -> str:
        return "knowledge"
    
    @property
    def supported_intents(self) -> list[Intent]:
        return [Intent.QUESTION]
    
    def can_handle(self, context: Context) -> float:
        """Check if this is a knowledge question."""
        if context.intent != Intent.QUESTION:
            return 0.0
        
        # High confidence for question words
        msg_lower = context.message.lower()
        question_starters = ['who', 'what', 'where', 'when', 'why', 'how', 'is ', 'are ', 'does ', 'did ', 'can ', 'could ']
        
        if any(msg_lower.startswith(q) for q in question_starters):
            return 0.9
        
        if '?' in context.message:
            return 0.8
        
        return 0.3
    
    def handle(self, context: Context) -> HandlerResult:
        """Process a knowledge question."""
        if not self.qa:
            return HandlerResult.failure_result("Knowledge system not initialized")
        
        message = context.message
        msg_lower = message.lower()
        
        # Try standard Q&A first
        result = self.qa.ask_detailed(message)
        
        if result['answers'] and result['answers'][0]['confidence'] > 0.3:
            answer = result['answers'][0]['answer']
            confidence = result['answers'][0]['confidence']
            
            # Check if it's a WHY/HOW question for reasoning
            if self.reasoning and (msg_lower.startswith("why") or msg_lower.startswith("how")):
                path = self.reasoning.reason(message)
                if path.steps and len(path.steps) > 1:
                    reasoning_chain = " → ".join([str(s) for s in path.steps[:4]])
                    answer = f"{answer}\n\n**Reasoning path:** {reasoning_chain}"
            
            return HandlerResult.success_result(
                answer,
                confidence=confidence,
                source="qa",
                query=message
            )
        
        # Try holographic generation for entity questions
        if self.hologen:
            entities = ["holmes", "watson", "moriarty", "lestrade", "irene", "mycroft", "darcy", "elizabeth"]
            for entity in entities:
                if entity in msg_lower:
                    try:
                        learnable = self.qa.projector.answer_generator.learnable
                        output = self.hologen.generate(message, entity=entity, learnable=learnable)
                        if output and len(output) > 20:
                            return HandlerResult.success_result(
                                output,
                                confidence=0.7,
                                source="holographic",
                                entity=entity
                            )
                    except Exception:
                        pass
        
        # Fallback - check if we can suggest related topics
        msg_lower = context.message.lower()
        mentioned_entity = None
        for key, info in self.known_entities.items():
            if key in msg_lower:
                mentioned_entity = info
                break
        
        if mentioned_entity:
            return HandlerResult.failure_result(
                self.formatter.format_not_found(
                    topic=f"that specific aspect of {mentioned_entity['name']}",
                    alternatives=[f"general information about {mentioned_entity['name']}",
                                  f"other characters from {mentioned_entity['domain']}"]
                )
            )
        
        return HandlerResult.failure_result(
            self.formatter.format_not_found(
                topic="that",
                alternatives=["Sherlock Holmes characters", "Pride and Prejudice characters"]
            )
        )
