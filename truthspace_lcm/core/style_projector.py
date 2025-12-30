#!/usr/bin/env python3
"""
Style Projector: Transform Concept Space to Specific Writing Styles

The key insight: We have two quaternions:
1. Semantic Quaternion - encodes WHAT we're saying (meaning)
2. φ-Dial Quaternion - encodes HOW we're saying it (style)

This module implements the φ-Dial for style projection:
- Style (X): formal ↔ casual (word choice)
- Perspective (Y): subjective ↔ meta (framing)
- Depth (Z): terse ↔ elaborate (detail level)
- Certainty (W): definitive ↔ hedged (confidence)

Named Styles (presets):
- "hemingway": terse, definitive, subjective
- "academic": formal, elaborate, hedged
- "journalistic": formal, terse, definitive
- "book_report": formal, elaborate, definitive
- "casual": casual, terse, subjective
- "encyclopedia": formal, elaborate, meta

The projection works by:
1. Taking raw concept-space output (e.g., "Holmes examined evidence")
2. Applying style transformations based on φ-dial settings
3. Producing styled output (e.g., "The detective studied the clues carefully")

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


PHI = 1.618034


@dataclass
class PhiDial:
    """
    4D Quaternion for output styling.
    
    Each axis ranges from -1 to +1:
    - style: -1 = formal, +1 = casual
    - perspective: -1 = subjective, +1 = meta/objective
    - depth: -1 = terse, +1 = elaborate
    - certainty: -1 = hedged, +1 = definitive
    """
    style: float = 0.0       # X: formal ↔ casual
    perspective: float = 0.0  # Y: subjective ↔ meta
    depth: float = 0.0       # Z: terse ↔ elaborate
    certainty: float = 0.0   # W: hedged ↔ definitive
    
    def __post_init__(self):
        # Clamp values to [-1, 1]
        self.style = max(-1, min(1, self.style))
        self.perspective = max(-1, min(1, self.perspective))
        self.depth = max(-1, min(1, self.depth))
        self.certainty = max(-1, min(1, self.certainty))
    
    @classmethod
    def from_preset(cls, name: str) -> 'PhiDial':
        """Create a PhiDial from a named preset."""
        presets = {
            # Hemingway: short, punchy, definitive
            'hemingway': cls(style=0.3, perspective=-0.5, depth=-0.8, certainty=0.9),
            
            # Academic: formal, detailed, hedged
            'academic': cls(style=-0.8, perspective=0.5, depth=0.7, certainty=-0.5),
            
            # Journalistic: formal, concise, factual
            'journalistic': cls(style=-0.5, perspective=0.8, depth=-0.3, certainty=0.7),
            
            # Book report: formal, detailed, definitive
            'book_report': cls(style=-0.6, perspective=0.3, depth=0.5, certainty=0.6),
            
            # Casual: informal, brief, personal
            'casual': cls(style=0.8, perspective=-0.7, depth=-0.4, certainty=0.3),
            
            # Encyclopedia: formal, comprehensive, objective
            'encyclopedia': cls(style=-0.9, perspective=0.9, depth=0.8, certainty=0.5),
            
            # Storyteller: engaging, detailed, personal
            'storyteller': cls(style=0.2, perspective=-0.6, depth=0.6, certainty=0.4),
            
            # Technical: formal, precise, definitive
            'technical': cls(style=-0.7, perspective=0.6, depth=0.3, certainty=0.8),
            
            # Neutral (default)
            'neutral': cls(style=0.0, perspective=0.0, depth=0.0, certainty=0.0),
        }
        return presets.get(name.lower(), presets['neutral'])
    
    def magnitude(self) -> float:
        """Quaternion magnitude."""
        return math.sqrt(
            self.style**2 + self.perspective**2 + 
            self.depth**2 + self.certainty**2
        )


class StyleProjector:
    """
    Project concept-space output to specific writing styles.
    
    Uses the φ-Dial quaternion to transform raw output into
    styled prose.
    """
    
    # Word substitutions for style axis (formal ↔ casual)
    STYLE_SUBSTITUTIONS = {
        # formal -> casual
        'examine': ('examine', 'check out'),
        'investigate': ('investigate', 'look into'),
        'observe': ('observe', 'watch'),
        'deduce': ('deduce', 'figure out'),
        'assist': ('assist', 'help'),
        'companion': ('companion', 'friend'),
        'residence': ('residence', 'place'),
        'individual': ('individual', 'person'),
        'demonstrate': ('demonstrate', 'show'),
        'utilize': ('utilize', 'use'),
        'obtain': ('obtain', 'get'),
        'require': ('require', 'need'),
        'sufficient': ('sufficient', 'enough'),
        'commence': ('commence', 'start'),
        'terminate': ('terminate', 'end'),
        'endeavor': ('endeavor', 'try'),
        'inquire': ('inquire', 'ask'),
        'respond': ('respond', 'answer'),
        'reside': ('reside', 'live'),
        'possess': ('possess', 'have'),
        'acquire': ('acquire', 'get'),
        'comprehend': ('comprehend', 'understand'),
        'perceive': ('perceive', 'see'),
        'notable': ('notable', 'famous'),
        'protagonist': ('protagonist', 'main character'),
        'antagonist': ('antagonist', 'bad guy'),
        'detective': ('detective', 'sleuth'),
        'physician': ('physician', 'doctor'),
        'evidence': ('evidence', 'clues'),
        'subsequently': ('subsequently', 'then'),
        'however': ('however', 'but'),
        'therefore': ('therefore', 'so'),
        'nevertheless': ('nevertheless', 'still'),
        'furthermore': ('furthermore', 'also'),
        'consequently': ('consequently', 'so'),
    }
    
    # Sentence starters for perspective axis (subjective ↔ meta)
    PERSPECTIVE_STARTERS = {
        'subjective': [
            "I think", "It seems", "One could say", "It appears",
            "From what we know", "Looking at this",
        ],
        'meta': [
            "Objectively", "In fact", "It is established that",
            "According to the text", "The evidence shows",
            "Historically", "As documented",
        ],
    }
    
    # Hedging phrases for certainty axis
    CERTAINTY_HEDGES = {
        'hedged': [
            "possibly", "perhaps", "might", "could", "may",
            "it seems", "apparently", "presumably", "likely",
        ],
        'definitive': [
            "certainly", "definitely", "clearly", "undoubtedly",
            "without question", "absolutely", "indeed",
        ],
    }
    
    # Elaboration templates for depth axis
    ELABORATION_TEMPLATES = {
        'entity_description': [
            ", a {role} known for {action},",
            ", who is recognized as a {role},",
            ", the {role} who {action},",
        ],
        'action_elaboration': [
            ", often {action} with great skill,",
            ", frequently engaging in {action},",
            ", known to {action} regularly,",
        ],
        'context_addition': [
            " In this context,",
            " Within the narrative,",
            " As the story shows,",
        ],
    }
    
    def __init__(self, knowledge=None):
        """
        Initialize the style projector.
        
        Args:
            knowledge: Optional GeometricKnowledge for entity info
        """
        self.knowledge = knowledge
        self.dial = PhiDial()
    
    def set_dial(self, dial: PhiDial):
        """Set the φ-dial for style projection."""
        self.dial = dial
    
    def set_style(self, name: str):
        """Set style from a named preset."""
        self.dial = PhiDial.from_preset(name)
    
    def _apply_style_axis(self, text: str) -> str:
        """Apply style transformation (formal ↔ casual)."""
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower()
            # Preserve punctuation
            punct = ''
            if word_lower and word_lower[-1] in '.,!?;:':
                punct = word_lower[-1]
                word_lower = word_lower[:-1]
            
            if word_lower in self.STYLE_SUBSTITUTIONS:
                formal, casual = self.STYLE_SUBSTITUTIONS[word_lower]
                if self.dial.style > 0.3:
                    # Casual
                    replacement = casual
                elif self.dial.style < -0.3:
                    # Formal
                    replacement = formal
                else:
                    replacement = word_lower
                
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + punct)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _apply_perspective_axis(self, text: str) -> str:
        """Apply perspective transformation (subjective ↔ meta)."""
        # Only add perspective framing for strong settings
        if abs(self.dial.perspective) < 0.5:
            return text
        
        sentences = re.split(r'([.!?]+\s*)', text)
        result = []
        added_starter = False
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part.strip():  # Actual sentence content
                sentence = part.strip()
                
                # Add perspective starter to first substantial sentence only
                if not added_starter and len(sentence.split()) > 3:
                    if self.dial.perspective < -0.5:
                        # Subjective - but don't always add
                        pass  # Skip for cleaner output
                    elif self.dial.perspective > 0.5:
                        # Meta/objective - reframe the sentence
                        # Instead of adding a starter, restructure
                        if 'is a' in sentence.lower():
                            # "X is a Y" -> "X can be characterized as a Y"
                            sentence = re.sub(r'(\w+)\s+is\s+a\s+', r'\1 can be characterized as a ', sentence, count=1)
                    added_starter = True
                
                result.append(sentence)
            else:
                result.append(part)
        
        return ''.join(result)
    
    def _apply_depth_axis(self, text: str, entity: str = None) -> str:
        """Apply depth transformation (terse ↔ elaborate)."""
        if self.dial.depth < -0.3:
            # Terse: shorten sentences
            # Remove parenthetical phrases
            text = re.sub(r'\s*,\s*[^,]+,\s*', ' ', text)
            # Remove "who is/was" clauses
            text = re.sub(r'\s+who\s+(is|was|are|were)\s+[^,\.]+[,\.]', '.', text)
            # Remove adverbs
            text = re.sub(r'\s+(very|really|quite|extremely|particularly)\s+', ' ', text)
            
        elif self.dial.depth > 0.3 and entity and self.knowledge:
            # Elaborate: add detail
            entity_lower = entity.lower()
            if entity_lower in self.knowledge.concepts:
                concept = self.knowledge.concepts[entity_lower]
                
                # Add role description
                if concept.actions:
                    top_action = concept.actions.most_common(1)[0][0]
                    
                    # Find a good insertion point (after first mention of entity)
                    pattern = re.compile(re.escape(entity), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        import random
                        template = random.choice(self.ELABORATION_TEMPLATES['entity_description'])
                        
                        # Infer role from phi_direction
                        if concept.phi_direction > 0.3:
                            role = "protagonist"
                        elif concept.phi_direction < -0.3:
                            role = "subject"
                        else:
                            role = "character"
                        
                        elaboration = template.format(role=role, action=top_action)
                        insert_pos = match.end()
                        text = text[:insert_pos] + elaboration + text[insert_pos:]
        
        return text
    
    def _apply_certainty_axis(self, text: str) -> str:
        """Apply certainty transformation (hedged ↔ definitive)."""
        if abs(self.dial.certainty) < 0.3:
            return text
        
        sentences = re.split(r'([.!?]+\s*)', text)
        result = []
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part.strip():
                sentence = part.strip()
                
                if self.dial.certainty < -0.3:
                    # Hedged: add uncertainty markers
                    # Add hedge to "is" statements
                    sentence = re.sub(r'\b(is|are|was|were)\b', 
                                     lambda m: f"appears to {m.group(1).replace('is', 'be').replace('are', 'be').replace('was', 'have been').replace('were', 'have been')}", 
                                     sentence, count=1)
                    
                elif self.dial.certainty > 0.3:
                    # Definitive: strengthen statements
                    import random
                    if 'is' in sentence.lower() and random.random() > 0.5:
                        # Add emphasis
                        emphatic = random.choice(self.CERTAINTY_HEDGES['definitive'])
                        sentence = re.sub(r'\b(is|are)\b', f'{emphatic} \\1', sentence, count=1)
                
                result.append(sentence)
            else:
                result.append(part)
        
        return ''.join(result)
    
    def project(self, text: str, entity: str = None) -> str:
        """
        Project text through the φ-dial to apply style.
        
        Args:
            text: Raw concept-space output
            entity: Main entity (for elaboration)
        
        Returns:
            Styled text
        """
        if not text:
            return text
        
        # Apply transformations in order
        result = text
        
        # 1. Style (word choice)
        result = self._apply_style_axis(result)
        
        # 2. Perspective (framing)
        result = self._apply_perspective_axis(result)
        
        # 3. Depth (detail level)
        result = self._apply_depth_axis(result, entity)
        
        # 4. Certainty (confidence)
        result = self._apply_certainty_axis(result)
        
        # Clean up
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = result.strip()
        
        # Ensure proper capitalization
        if result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def project_with_style(self, text: str, style_name: str, entity: str = None) -> str:
        """
        Project text with a named style preset.
        
        Args:
            text: Raw concept-space output
            style_name: Name of style preset
            entity: Main entity (for elaboration)
        
        Returns:
            Styled text
        """
        self.set_style(style_name)
        return self.project(text, entity)


class StyledGenerator:
    """
    Wrapper that combines geodesic generation with style projection.
    
    This is the main interface for generating styled output.
    """
    
    def __init__(self, qa_system):
        """
        Initialize with a HolographicGeometricQA system.
        
        Args:
            qa_system: HolographicGeometricQA instance
        """
        self.qa = qa_system
        self.projector = StyleProjector(qa_system.knowledge)
        self.current_style = 'neutral'
    
    def set_style(self, style_name: str):
        """Set the output style."""
        self.current_style = style_name
        self.projector.set_style(style_name)
    
    def set_dial(self, style: float = 0, perspective: float = 0, 
                 depth: float = 0, certainty: float = 0):
        """Set the φ-dial directly."""
        self.projector.set_dial(PhiDial(style, perspective, depth, certainty))
    
    def ask(self, question: str) -> str:
        """Ask a question and get a styled response."""
        # Get raw answer from QA system
        raw_answer = self.qa.ask(question)
        
        # Extract entity for elaboration
        entity = self._extract_entity(question)
        
        # Project through style
        return self.projector.project(raw_answer, entity)
    
    def generate_about(self, concept: str, num_sentences: int = 3) -> str:
        """Generate styled text about a concept."""
        raw_text = self.qa.generate_about(concept, num_sentences)
        return self.projector.project(raw_text, concept)
    
    def generate_story(self, concepts: List[str], max_sentences: int = 5) -> str:
        """Generate a styled story connecting concepts."""
        raw_text = self.qa.generate_story(concepts, max_sentences)
        return self.projector.project(raw_text, concepts[0] if concepts else None)
    
    def _extract_entity(self, question: str) -> Optional[str]:
        """Extract the main entity from a question."""
        words = re.findall(r'\b\w+\b', question.lower())
        skip = {'who', 'what', 'where', 'when', 'why', 'how', 
                'is', 'are', 'was', 'were', 'did', 'does', 'do',
                'the', 'a', 'an'}
        
        for word in words:
            if word not in skip and len(word) > 2:
                return word
        return None


def demo():
    """Demonstrate style projection."""
    print("=" * 70)
    print("  STYLE PROJECTOR DEMO")
    print("=" * 70)
    
    # Sample raw output
    raw_texts = [
        "Holmes examined the evidence carefully.",
        "Watson is a loyal companion who assists Holmes.",
        "The detective solved the mystery.",
        "Holmes is a notable protagonist who examines, deduces, and observes.",
    ]
    
    styles = ['hemingway', 'academic', 'journalistic', 'book_report', 'casual']
    
    projector = StyleProjector()
    
    for raw in raw_texts[:2]:
        print(f"\nRaw: {raw}")
        print("-" * 50)
        
        for style in styles:
            projector.set_style(style)
            styled = projector.project(raw, "Holmes")
            print(f"  {style:15}: {styled}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
