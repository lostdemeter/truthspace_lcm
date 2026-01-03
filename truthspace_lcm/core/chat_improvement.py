"""
Chat Improvement Gear - Automatic Response Quality Enhancement

Uses shape-based (folding) deficiency detection to automatically
improve chat responses before they are returned to the user.

Key features:
1. Learn good response templates from successful interactions
2. Detect structural deficiencies in generated responses
3. Suggest and apply improvements automatically
4. Track improvement effectiveness over time

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import re
import time

from .folding_deficiency import (
    FoldingStructure, FoldingDeficiencyDetector,
    ShapeDeficiency, ShapeDeficiencyType
)


@dataclass
class ResponseTemplate:
    """A learned response template with its shape signature."""
    name: str
    example_text: str
    structure: FoldingStructure
    shape: np.ndarray
    question_type: str  # 'who', 'what', 'how', 'why', 'general'
    success_count: int = 0
    total_uses: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_uses, 1)


@dataclass
class ImprovementResult:
    """Result of an improvement attempt."""
    original: str
    improved: str
    deficiency: Optional[ShapeDeficiency]
    improvement_applied: bool
    improvement_type: str
    shape_similarity_before: float
    shape_similarity_after: float


class ChatImprovementGear:
    """
    Automatic chat response improvement using shape-based detection.
    
    This gear:
    1. Learns good response patterns from examples
    2. Detects deficiencies in generated responses
    3. Applies improvements to fix structural issues
    4. Tracks what works and adapts over time
    """
    
    def __init__(self):
        self.name = "ChatImprovementGear"
        self.detector = FoldingDeficiencyDetector()
        
        # Learned templates by question type
        self.templates: Dict[str, List[ResponseTemplate]] = defaultdict(list)
        
        # Improvement history for learning
        self.improvement_history: List[ImprovementResult] = []
        
        # Effectiveness tracking
        self.improvement_stats = {
            'total_checks': 0,
            'deficiencies_found': 0,
            'improvements_applied': 0,
            'improvements_successful': 0,
        }
        
        # Initialize with default good response templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize with known good response patterns."""
        
        # "Who is" question template - character description
        self.learn_template(
            name="character_description",
            example=(
                "Captain Ahab is the captain of the Pequod. "
                "Ahab is known for his obsession with the white whale. "
                "The captain lost his leg to Moby Dick and seeks revenge."
            ),
            question_type="who"
        )
        
        # "What is" question template - definition
        self.learn_template(
            name="definition",
            example=(
                "The Pequod is a whaling ship. "
                "The ship is commanded by Captain Ahab. "
                "The Pequod sets sail from Nantucket."
            ),
            question_type="what"
        )
        
        # "Tell me about" template - comprehensive
        self.learn_template(
            name="comprehensive",
            example=(
                "Queequeg is a harpooner from the South Pacific. "
                "Queequeg becomes close friends with Ishmael. "
                "The harpooner is known for his skill and loyalty. "
                "Queequeg joins the crew of the Pequod."
            ),
            question_type="who"
        )
        
        # "How" question template - explanation
        self.learn_template(
            name="explanation",
            example=(
                "Whaling involves hunting whales for oil and other products. "
                "The process requires skilled harpooners. "
                "Ships like the Pequod would sail for years. "
                "Whaling was dangerous but profitable."
            ),
            question_type="how"
        )
        
        # "Why" question template - reasoning
        self.learn_template(
            name="reasoning",
            example=(
                "Ahab seeks the whale because Moby Dick took his leg. "
                "The captain's obsession drives the entire voyage. "
                "Revenge motivates Ahab more than profit."
            ),
            question_type="why"
        )
    
    def learn_template(self, name: str, example: str, question_type: str):
        """Learn a new response template."""
        structure = FoldingStructure.from_text(example)
        shape = structure.compute_shape()
        
        template = ResponseTemplate(
            name=name,
            example_text=example,
            structure=structure,
            shape=shape,
            question_type=question_type
        )
        
        self.templates[question_type].append(template)
        
        # Also register with detector
        self.detector.learn_template(name, example)
    
    def detect_question_type(self, question: str) -> str:
        """Detect the type of question being asked."""
        q_lower = question.lower()
        
        if 'who is' in q_lower or 'who was' in q_lower:
            return 'who'
        elif 'what is' in q_lower or 'what are' in q_lower:
            return 'what'
        elif 'tell me about' in q_lower or 'describe' in q_lower:
            return 'who'  # Similar to who questions
        elif 'how' in q_lower:
            return 'how'
        elif 'why' in q_lower:
            return 'why'
        else:
            return 'general'
    
    def get_best_template(self, question_type: str) -> Optional[ResponseTemplate]:
        """Get the best template for a question type."""
        templates = self.templates.get(question_type, [])
        
        if not templates:
            # Fall back to general templates
            templates = self.templates.get('who', [])
        
        if not templates:
            return None
        
        # Return template with best success rate
        return max(templates, key=lambda t: t.success_rate)
    
    def check_response(self, question: str, response: str) -> ShapeDeficiency:
        """Check a response for deficiencies."""
        self.improvement_stats['total_checks'] += 1
        
        question_type = self.detect_question_type(question)
        template = self.get_best_template(question_type)
        
        if template:
            # Check against learned template
            deficiency = self.detector.detect(template.example_text, response)
        else:
            # Check against the question itself (structural echo)
            deficiency = self.detector.detect(question, response)
        
        if deficiency.type != ShapeDeficiencyType.NONE:
            self.improvement_stats['deficiencies_found'] += 1
        
        return deficiency
    
    def improve_response(self, question: str, response: str) -> ImprovementResult:
        """
        Attempt to improve a response based on shape analysis.
        
        Returns the improved response and improvement details.
        """
        # Check for deficiencies
        deficiency = self.check_response(question, response)
        
        original_structure = FoldingStructure.from_text(response)
        question_type = self.detect_question_type(question)
        template = self.get_best_template(question_type)
        
        # Calculate initial shape similarity
        if template:
            shape_before = template.structure.shape_similarity(original_structure)
        else:
            shape_before = 0.0
        
        # If no deficiency or minor, return as-is
        if deficiency.type == ShapeDeficiencyType.NONE or deficiency.severity < 0.2:
            return ImprovementResult(
                original=response,
                improved=response,
                deficiency=deficiency,
                improvement_applied=False,
                improvement_type="none",
                shape_similarity_before=shape_before,
                shape_similarity_after=shape_before
            )
        
        # Apply improvements based on deficiency type
        improved = response
        improvement_type = "none"
        
        if deficiency.type == ShapeDeficiencyType.INCOMPLETE:
            # Response is too short - try to expand
            improved, improvement_type = self._expand_response(response, question, template)
        
        elif deficiency.type == ShapeDeficiencyType.MISSING_STRUCTURE:
            # Missing self-references - add them
            improved, improvement_type = self._add_self_references(response, question, deficiency)
        
        elif deficiency.type == ShapeDeficiencyType.WRONG_STRUCTURE:
            # Wrong structure - try to restructure
            improved, improvement_type = self._restructure_response(response, question, template)
        
        elif deficiency.type == ShapeDeficiencyType.PARTIAL:
            # Partial mismatch - minor adjustments
            improved, improvement_type = self._refine_response(response, question, deficiency)
        
        # Calculate final shape similarity
        improved_structure = FoldingStructure.from_text(improved)
        if template:
            shape_after = template.structure.shape_similarity(improved_structure)
        else:
            shape_after = shape_before
        
        # Track improvement
        if improved != response:
            self.improvement_stats['improvements_applied'] += 1
            if shape_after > shape_before:
                self.improvement_stats['improvements_successful'] += 1
        
        result = ImprovementResult(
            original=response,
            improved=improved,
            deficiency=deficiency,
            improvement_applied=(improved != response),
            improvement_type=improvement_type,
            shape_similarity_before=shape_before,
            shape_similarity_after=shape_after
        )
        
        self.improvement_history.append(result)
        return result
    
    def _expand_response(self, response: str, question: str, 
                         template: Optional[ResponseTemplate]) -> Tuple[str, str]:
        """Expand a response that is too short."""
        # Extract the main topic from the question (proper nouns)
        # Skip question words like Who, What, How, etc.
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'tell', 'me', 'about'}
        words = re.findall(r'\b[A-Z][a-z]+\b', question)
        
        # Filter out question words
        topic_words = [w for w in words if w.lower() not in question_words]
        main_topic = topic_words[0] if topic_words else None
        
        if not main_topic:
            # Try to find any capitalized word in the question after common patterns
            match = re.search(r'(?:who is|what is|tell me about|about)\s+([A-Z][a-z]+)', question, re.IGNORECASE)
            if match:
                main_topic = match.group(1)
        
        if not main_topic:
            return response, "none"
        
        # Check if response mentions the topic
        topic_lower = main_topic.lower()
        if topic_lower not in response.lower():
            # Add topic reference naturally at the start
            response = f"{main_topic}: {response}"
        
        # If response has 2 sentences and topic is mentioned, add closing reference
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) == 2 and topic_lower in response.lower():
            response = f"{response.rstrip('.')}. This is about {main_topic}."
        
        return response, "expand"
    
    def _add_self_references(self, response: str, question: str,
                             deficiency: ShapeDeficiency) -> Tuple[str, str]:
        """Add self-references to create folds."""
        # Get missing fold words
        missing = deficiency.missing_fold_words
        
        if not missing:
            return response, "none"
        
        # Try to add references for missing words
        improved = response
        for word in list(missing)[:2]:  # Add up to 2 references
            # Check if word appears in response
            if word.lower() in improved.lower():
                continue
            
            # Add a sentence referencing the word
            improved = f"{improved} This relates to {word}."
        
        return improved, "add_refs"
    
    def _restructure_response(self, response: str, question: str,
                              template: Optional[ResponseTemplate]) -> Tuple[str, str]:
        """Restructure a response to match expected pattern."""
        if not template:
            return response, "none"
        
        # Extract sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return response, "none"
        
        # Try to create self-references by repeating key nouns
        # Find the main noun in first sentence
        first_words = re.findall(r'\b[A-Z][a-z]+\b', sentences[0])
        if first_words:
            main_noun = first_words[0]
            
            # Check if it appears in other sentences
            refs_found = sum(1 for s in sentences[1:] if main_noun.lower() in s.lower())
            
            if refs_found == 0 and len(sentences) >= 2:
                # Add reference in second sentence
                sentences[1] = f"{main_noun} {sentences[1].lower()}"
        
        improved = '. '.join(sentences) + '.'
        return improved, "restructure"
    
    def _refine_response(self, response: str, question: str,
                         deficiency: ShapeDeficiency) -> Tuple[str, str]:
        """Make minor refinements to improve shape match."""
        # For partial mismatches, try small adjustments
        
        # Ensure response ends with period
        if not response.strip().endswith('.'):
            response = response.strip() + '.'
        
        # Remove redundant whitespace
        response = ' '.join(response.split())
        
        return response, "refine"
    
    def get_stats(self) -> Dict:
        """Get improvement statistics."""
        stats = self.improvement_stats.copy()
        
        if stats['improvements_applied'] > 0:
            stats['success_rate'] = (
                stats['improvements_successful'] / stats['improvements_applied']
            )
        else:
            stats['success_rate'] = 0.0
        
        stats['deficiency_rate'] = (
            stats['deficiencies_found'] / max(stats['total_checks'], 1)
        )
        
        return stats
    
    def feedback(self, was_helpful: bool):
        """
        Provide feedback on the last improvement.
        
        This allows the system to learn what improvements work.
        """
        if not self.improvement_history:
            return
        
        last = self.improvement_history[-1]
        
        # Update template success tracking
        if last.improvement_applied:
            # Find which template was used
            for templates in self.templates.values():
                for template in templates:
                    if template.name == last.improvement_type:
                        template.total_uses += 1
                        if was_helpful:
                            template.success_count += 1
                        break


def create_chat_improvement_gear() -> ChatImprovementGear:
    """Factory function to create a configured ChatImprovementGear."""
    return ChatImprovementGear()
