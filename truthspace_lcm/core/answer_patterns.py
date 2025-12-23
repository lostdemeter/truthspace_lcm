#!/usr/bin/env python3
"""
Answer Pattern Learning with Noise + Reprojection

Learn answer structures from Q&A training data and apply them to new content.

Key Insight:
    Q&A training teaches STRUCTURE: "[NAME] is a [ROLE] from [PLACE] who [ACTION]"
    Literary data provides CONTENT: Darcy, Pride and Prejudice, speaks, elizabeth
    
    Pattern transfer = Structure from Q&A + Content from literature

Noise + Reprojection:
    1. GEODESIC: Find the direct path (cookie-cutter but accurate)
    2. NOISE: Add controlled variation (templates, word choice)
    3. REPROJECT: Ground back to corpus/geometry (maintain accuracy)
    
    This gives variety without sacrificing accuracy.
"""

import re
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# PHI-DIAL: Unified Control Mechanism
# =============================================================================

class PhiDial:
    """
    1D control mechanism using φ-navigation (horizontal only).
    
    DEPRECATED: Use ComplexPhiDial for 2D control.
    """
    
    PHI = (1 + np.sqrt(5)) / 2
    
    def __init__(self, dial: float = 0.0):
        self.dial = max(-1.0, min(1.0, dial))
    
    def weight(self, value: float) -> float:
        log_val = np.log1p(max(0, value))
        return self.PHI ** (self.dial * log_val)
    
    def get_style(self) -> str:
        if self.dial < -0.3:
            return 'formal'
        elif self.dial > 0.3:
            return 'casual'
        return 'neutral'


class ComplexPhiDial:
    """
    3D control mechanism using φ-navigation.
    
    "We control the horizontal. We control the vertical. We control the depth."
    
    X-AXIS (horizontal): Style
        -1 = formal, specific, rare words
        +1 = casual, universal, common words
        
    Y-AXIS (vertical): Perspective
        -1 = subjective, experiential, personal
         0 = objective, factual, neutral
        +1 = meta, analytical, reflective
        
    Z-AXIS (depth): Elaboration
        -1 = terse, minimal, just the facts
         0 = standard, balanced
        +1 = elaborate, detailed, full context
    
    The MAGNITUDE (φ^x) controls WHAT words we choose.
    The PHASE (y·ln(φ)) controls HOW we frame the content.
    The SCALE (z) controls HOW MUCH detail we include.
    """
    
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initialize the 3D φ-dial.
        
        Args:
            x: Horizontal dial (-1 to +1): Style
               -1 = formal, specific, rare
               +1 = casual, universal, common
            y: Vertical dial (-1 to +1): Perspective
               -1 = subjective, experiential
                0 = objective, factual
               +1 = meta, analytical
            z: Depth dial (-1 to +1): Elaboration
               -1 = terse, minimal
                0 = standard, balanced
               +1 = elaborate, detailed
        """
        self.x = max(-1.0, min(1.0, x))  # Horizontal: style
        self.y = max(-1.0, min(1.0, y))  # Vertical: perspective
        self.z = max(-1.0, min(1.0, z))  # Depth: elaboration
    
    def weight(self, value: float) -> float:
        """Apply horizontal φ-dial weighting (magnitude)."""
        log_val = np.log1p(max(0, value))
        return self.PHI ** (self.x * log_val)
    
    def get_style(self) -> str:
        """Get style from horizontal position."""
        if self.x < -0.3:
            return 'formal'
        elif self.x > 0.3:
            return 'casual'
        return 'neutral'
    
    def get_perspective(self) -> str:
        """Get perspective from vertical position."""
        if self.y < -0.3:
            return 'subjective'
        elif self.y > 0.3:
            return 'meta'
        return 'objective'
    
    def get_depth(self) -> str:
        """Get depth/elaboration level from z position."""
        if self.z < -0.3:
            return 'terse'
        elif self.z > 0.3:
            return 'elaborate'
        return 'standard'
    
    def get_max_actions(self) -> int:
        """How many action verbs to include based on depth."""
        if self.z < -0.3:
            return 1      # Terse: just one
        elif self.z > 0.3:
            return 4      # Elaborate: up to 4
        return 2          # Standard: 2
    
    def include_relationship(self) -> bool:
        """Whether to include relationship info."""
        return self.z > -0.5  # Only skip for very terse
    
    def include_source(self) -> bool:
        """Whether to include source attribution."""
        return self.z > 0.0   # Only for standard+
    
    def include_elaboration(self) -> bool:
        """Whether to add closing elaboration."""
        return self.z > 0.3   # Only for elaborate
    
    def get_quadrant(self) -> Tuple[str, str]:
        """Get (style, perspective) tuple."""
        return (self.get_style(), self.get_perspective())
    
    def get_octant(self) -> Tuple[str, str, str]:
        """Get (style, perspective, depth) tuple."""
        return (self.get_style(), self.get_perspective(), self.get_depth())
    
    def get_quadrant_label(self) -> str:
        """Get human-readable quadrant label (2D, ignoring z)."""
        style, perspective = self.get_quadrant()
        labels = {
            ('formal', 'subjective'): 'Literary/Immersive',
            ('formal', 'objective'): 'Academic/Factual',
            ('formal', 'meta'): 'Scholarly/Analytical',
            ('casual', 'subjective'): 'Conversational/Personal',
            ('casual', 'objective'): 'Informal/Direct',
            ('casual', 'meta'): 'Pop Culture/Commentary',
            ('neutral', 'subjective'): 'Balanced/Personal',
            ('neutral', 'objective'): 'Balanced/Neutral',
            ('neutral', 'meta'): 'Balanced/Reflective',
        }
        return labels.get((style, perspective), f'{style}+{perspective}')
    
    def get_octant_label(self) -> str:
        """Get human-readable octant label (3D)."""
        return f'{self.get_quadrant_label()} ({self.get_depth()})'


# Style-specific vocabulary (horizontal axis)
STYLE_VOCABULARY = {
    'verbs': {
        'speak': {'formal': 'articulated', 'neutral': 'spoke', 'casual': 'said'},
        'think': {'formal': 'contemplated', 'neutral': 'considered', 'casual': 'thought about'},
        'move': {'formal': 'proceeded', 'neutral': 'traveled', 'casual': 'went'},
        'look': {'formal': 'scrutinized', 'neutral': 'observed', 'casual': 'looked at'},
        'act': {'formal': 'executed', 'neutral': 'performed', 'casual': 'did'},
        'possess': {'formal': 'maintained', 'neutral': 'held', 'casual': 'had'},
    },
    'connectors': {
        'regarding': {'formal': 'regarding', 'neutral': 'concerning', 'casual': 'about'},
        'involving': {'formal': 'in conjunction with', 'neutral': 'involving', 'casual': 'with'},
        'from': {'formal': 'originating from', 'neutral': 'from', 'casual': 'from'},
    },
    'descriptors': {
        'character': {'formal': 'literary figure', 'neutral': 'character', 'casual': 'person'},
        'story': {'formal': 'narrative', 'neutral': 'story', 'casual': 'tale'},
    }
}

# Perspective-specific vocabulary (vertical axis)
PERSPECTIVE_VOCABULARY = {
    # Sentence openers that set the perspective
    'openers': {
        'subjective': [
            'I find that',
            'One cannot help but notice that',
            'It strikes me that',
            'From my perspective,',
            'I would say that',
        ],
        'objective': [
            '',  # No opener needed for objective
        ],
        'meta': [
            'In the broader context,',
            'Symbolically speaking,',
            'As an archetype,',
            'From a literary perspective,',
            'Thematically,',
        ],
    },
    # How to describe the subject
    'subject_framing': {
        'subjective': {
            'character': 'this fascinating individual',
            'detective': 'this remarkable investigator',
            'person': 'someone I find quite compelling',
        },
        'objective': {
            'character': 'a character',
            'detective': 'a detective',
            'person': 'an individual',
        },
        'meta': {
            'character': 'an archetypal figure',
            'detective': 'the embodiment of rational deduction',
            'person': 'a representation of',
        },
    },
    # Relationship framing
    'relationship_framing': {
        'subjective': [
            'whose bond with {related} I find particularly moving',
            'and their relationship with {related} is quite remarkable',
            'with {related} playing a deeply significant role',
        ],
        'objective': [
            'often involving {related}',
            'frequently associated with {related}',
            'in connection with {related}',
        ],
        'meta': [
            'with {related} serving as a narrative foil',
            'where {related} represents the complementary archetype',
            'and {related} embodies the counterpoint',
        ],
    },
    # Closing phrases
    'closers': {
        'subjective': [
            ' — which I find quite compelling.',
            ' — and I must say, it\'s rather fascinating.',
            ' — truly a memorable presence.',
        ],
        'objective': [
            '.',
        ],
        'meta': [
            ', representing broader themes of the genre.',
            ', embodying timeless literary patterns.',
            ', a quintessential example of the form.',
        ],
    },
}

# Depth-specific vocabulary (z-axis)
DEPTH_VOCABULARY = {
    # Elaboration phrases added at the end for elaborate mode
    'elaboration': {
        'detective': [
            'Known for remarkable deductive abilities and keen observation.',
            'A master of logical reasoning and forensic analysis.',
            'Renowned for solving the most baffling mysteries.',
        ],
        'character': [
            'A memorable presence in the narrative.',
            'Central to the story\'s development.',
            'Whose actions shape the course of events.',
        ],
        'lover': [
            'Whose romantic journey forms the heart of the story.',
            'A figure of passion and emotional depth.',
        ],
        'thinker': [
            'A contemplative figure given to deep reflection.',
            'Whose thoughts reveal the inner workings of the narrative.',
        ],
        'default': [
            'A significant figure in the work.',
            'Whose presence enriches the narrative.',
        ],
    },
}


# =============================================================================
# ANSWER TEMPLATES (learned from Q&A patterns)
# =============================================================================

# These templates are extracted from analyzing Q&A training data
# Format: [SLOT] markers get filled with content

WHO_IS_TEMPLATES = [
    # From "Einstein is a scientist from Germany who developed..."
    "{name} is a {role} from {source} who {action}",
    # From "Curie worked tirelessly... often involving Pierre"
    "{name} is a character from {source} who {action} often involving {related}",
    # Simpler fallback
    "{name} is a character from {source} who {action}",
    # Minimal
    "{name} appears in {source}",
]

WHAT_DID_TEMPLATES = [
    # From "Einstein thought about physics and developed new theories"
    "{name} {action_past} and {action2_past}",
    # From "Curie discovered... and won..."
    "{name} {action_past} regarding {patient}",
    # Simpler
    "{name} {action_past}",
]

WHERE_IS_TEMPLATES = [
    # From "Paris is located in France along the Seine"
    "{name} is located in {location}",
    # From literary context
    "{name} appears at {location}",
]


# =============================================================================
# ROLE INFERENCE
# =============================================================================

# Infer character roles from their actions
ACTION_TO_ROLE = {
    'SPEAK': ['speaker', 'conversationalist', 'character'],
    'THINK': ['thinker', 'philosopher', 'intellectual'],
    'MOVE': ['traveler', 'wanderer', 'adventurer'],
    'PERCEIVE': ['observer', 'witness', 'watcher'],
    'FEEL': ['emotional character', 'passionate soul', 'sensitive person'],
    'ACT': ['actor', 'doer', 'active character'],
    'POSSESS': ['owner', 'holder', 'possessor'],
    'EXIST': ['character', 'figure', 'person'],
}

# Action to past tense verb
ACTION_TO_PAST = {
    'SPEAK': 'spoke',
    'THINK': 'thought deeply',
    'MOVE': 'traveled',
    'PERCEIVE': 'observed',
    'FEEL': 'felt strongly',
    'ACT': 'took action',
    'POSSESS': 'possessed things',
    'EXIST': 'appeared',
}

# Varied action phrases for noise injection
ACTION_PHRASES_VARIED = {
    'SPEAK': [
        'speaks throughout the narrative',
        'is known for their dialogue',
        'communicates frequently',
        'engages in conversation',
    ],
    'THINK': [
        'thinks deeply',
        'is characterized by reflection',
        'contemplates throughout',
        'reasons carefully',
    ],
    'ACT': [
        'takes decisive action',
        'acts throughout the story',
        'engages actively',
        'drives events forward',
    ],
    'MOVE': [
        'travels extensively',
        'journeys through the narrative',
        'moves through various settings',
        'ventures forth',
    ],
    'PERCEIVE': [
        'observes keenly',
        'watches carefully',
        'notices details others miss',
        'perceives with clarity',
    ],
    'POSSESS': [
        'holds significant influence',
        'maintains important connections',
        'possesses key knowledge',
    ],
    'EXIST': [
        'appears throughout',
        'features prominently',
        'is central to the story',
    ],
}

# Varied relationship phrases
RELATIONSHIP_PHRASES = [
    'often involving {related}',
    'closely connected to {related}',
    'with {related} playing a central role',
    'frequently alongside {related}',
    'whose story intertwines with {related}',
]

# Varied sentence templates for WHO answers
WHO_TEMPLATES_VARIED = [
    '{name} is a {role} from {source} who {action}, {relation}.',
    'In {source}, {name} {action}. {relation_cap}.',
    'A {role} in {source}, {name} {action}, {relation}.',
    '{name} from {source} {action}. {relation_cap}.',
    'The {role} {name}, featured in {source}, {action}, {relation}.',
]

# Capitalized relationship phrases (for sentence starts)
RELATIONSHIP_PHRASES_CAP = [
    '{related} plays a central role in their story',
    'Their defining relationship is with {related}',
    '{related} is closely connected to them',
    'Much of their narrative involves {related}',
]

# More specific action verbs based on context
ACTION_VERBS_DETAILED = {
    'SPEAK': ['spoke', 'said', 'declared', 'exclaimed', 'whispered'],
    'THINK': ['thought', 'pondered', 'considered', 'reflected', 'contemplated'],
    'MOVE': ['traveled', 'walked', 'journeyed', 'ventured', 'went'],
    'PERCEIVE': ['observed', 'noticed', 'saw', 'witnessed', 'watched'],
    'FEEL': ['felt', 'experienced', 'sensed', 'loved', 'feared'],
    'ACT': ['acted', 'did', 'performed', 'executed', 'carried out'],
    'POSSESS': ['had', 'owned', 'held', 'kept', 'maintained'],
}


# =============================================================================
# PATTERN-BASED ANSWER GENERATOR
# =============================================================================

class PatternAnswerGenerator:
    """
    Generate answers using learned patterns from Q&A training.
    
    Now with 3D φ-dial control:
        - X (horizontal): Style (formal ↔ casual)
        - Y (vertical): Perspective (subjective ↔ objective ↔ meta)
        - Z (depth): Elaboration (terse ↔ standard ↔ elaborate)
    
    "We control the horizontal. We control the vertical. We control the depth."
    """
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initialize the pattern generator with 3D φ-dial.
        
        Args:
            x: Horizontal dial (-1 to +1): Style
               -1 = formal, specific, rare
               +1 = casual, universal, common
            y: Vertical dial (-1 to +1): Perspective
               -1 = subjective, experiential
                0 = objective, factual
               +1 = meta, analytical
            z: Depth dial (-1 to +1): Elaboration
               -1 = terse, minimal
                0 = standard, balanced
               +1 = elaborate, detailed
        """
        self.who_templates = WHO_IS_TEMPLATES
        self.what_templates = WHAT_DID_TEMPLATES
        self.where_templates = WHERE_IS_TEMPLATES
        self.dial = ComplexPhiDial(x, y, z)
        
        # Noise words to filter from relationships
        self.noise_words = {
            'evidence', 'chapter', 'part', 'book', 'volume', 'venus', 'allegro',
            'preface', 'introduction', 'contents', 'illustration', 'page',
            'white', 'rabbit', 'red', 'black', 'blue', 'green',
            'aunt', 'uncle', 'mother', 'father',
        }
    
    def set_dial(self, x: float = None, y: float = None, z: float = None):
        """Set dial values. Only updates provided values."""
        new_x = x if x is not None else self.dial.x
        new_y = y if y is not None else self.dial.y
        new_z = z if z is not None else self.dial.z
        self.dial = ComplexPhiDial(new_x, new_y, new_z)
    
    def set_style(self, x: float):
        """Set the horizontal style dial (-1 = formal, +1 = casual)."""
        self.dial = ComplexPhiDial(x, self.dial.y, self.dial.z)
    
    def set_perspective(self, y: float):
        """Set the vertical perspective dial (-1 = subjective, +1 = meta)."""
        self.dial = ComplexPhiDial(self.dial.x, y, self.dial.z)
    
    def set_depth(self, z: float):
        """Set the depth dial (-1 = terse, +1 = elaborate)."""
        self.dial = ComplexPhiDial(self.dial.x, self.dial.y, z)
    
    def _get_depth_elaboration(self, role: str) -> str:
        """Get elaboration phrase based on depth and role."""
        if not self.dial.include_elaboration():
            return ''
        elaborations = DEPTH_VOCABULARY['elaboration'].get(
            role, DEPTH_VOCABULARY['elaboration']['default']
        )
        return ' ' + random.choice(elaborations)
    
    def _get_styled_verb(self, action: str) -> str:
        """Get verb with style applied via φ-dial."""
        style = self.dial.get_style()
        action_lower = action.lower()
        
        # Map action primitives to verb keys
        action_map = {
            'SPEAK': 'speak',
            'THINK': 'think',
            'MOVE': 'move',
            'PERCEIVE': 'look',
            'ACT': 'act',
            'POSSESS': 'possess',
        }
        
        verb_key = action_map.get(action, 'act')
        verbs = STYLE_VOCABULARY['verbs'].get(verb_key, {})
        return verbs.get(style, ACTION_TO_PAST.get(action, 'acted'))
    
    def _get_styled_connector(self, connector: str) -> str:
        """Get connector with style applied via φ-dial."""
        style = self.dial.get_style()
        connectors = STYLE_VOCABULARY['connectors'].get(connector, {})
        return connectors.get(style, connector)
    
    def _get_styled_descriptor(self, descriptor: str) -> str:
        """Get descriptor with style applied via φ-dial."""
        style = self.dial.get_style()
        descriptors = STYLE_VOCABULARY['descriptors'].get(descriptor, {})
        return descriptors.get(style, descriptor)
    
    def _get_perspective_opener(self) -> str:
        """Get sentence opener based on perspective."""
        perspective = self.dial.get_perspective()
        openers = PERSPECTIVE_VOCABULARY['openers'].get(perspective, [''])
        return random.choice(openers)
    
    def _get_perspective_framing(self, role: str) -> str:
        """Get subject framing based on perspective."""
        perspective = self.dial.get_perspective()
        framings = PERSPECTIVE_VOCABULARY['subject_framing'].get(perspective, {})
        return framings.get(role, framings.get('character', 'a character'))
    
    def _get_perspective_relationship(self, related: str) -> str:
        """Get relationship framing based on perspective."""
        perspective = self.dial.get_perspective()
        templates = PERSPECTIVE_VOCABULARY['relationship_framing'].get(perspective, ['involving {related}'])
        template = random.choice(templates)
        return template.format(related=related.title())
    
    def _get_perspective_closer(self) -> str:
        """Get sentence closer based on perspective."""
        perspective = self.dial.get_perspective()
        closers = PERSPECTIVE_VOCABULARY['closers'].get(perspective, ['.'])
        return random.choice(closers)
    
    def generate_who_answer(
        self,
        entity: str,
        actions: List[Tuple[str, int]],
        patients: List[Tuple[str, int]],
        sources: set,
        noise_level: float = 0.0,
    ) -> str:
        """
        Generate a WHO IS answer using learned patterns.
        
        Pattern: "[NAME] is a [ROLE] from [SOURCE] who [ACTION] often involving [RELATED]"
        
        Args:
            noise_level: 0.0 = cookie-cutter (geodesic), 1.0 = maximum variation
        """
        # Get source
        source = list(sources)[0] if sources else "the story"
        
        # Infer role from dominant action
        role = "character"
        top_action = 'EXIST'
        if actions:
            top_action = actions[0][0]
            roles = ACTION_TO_ROLE.get(top_action, ['character'])
            # Add noise: sometimes pick a different role
            if noise_level > 0 and random.random() < noise_level:
                role = random.choice(roles)
            else:
                role = roles[0]
        
        # Get related character (filter noise)
        related = None
        for patient, count in patients:
            if patient.lower() not in self.noise_words and patient != entity and len(patient) > 2:
                related = patient
                break
        
        # GEODESIC PATH (noise_level = 0): Now with 3D φ-dial control!
        # X controls STYLE, Y controls PERSPECTIVE, Z controls DEPTH
        if noise_level == 0:
            # Get styled action descriptions (horizontal axis + depth)
            max_actions = self.dial.get_max_actions()
            action_descs = []
            for action, count in actions:
                if action != 'EXIST' and action != 'POSSESS':
                    desc = self._get_styled_verb(action)
                    if desc not in action_descs:
                        action_descs.append(desc)
                    if len(action_descs) >= max_actions:
                        break
            
            action_desc = " and ".join(action_descs) if action_descs else "appears"
            
            # Get perspective components (vertical axis)
            opener = self._get_perspective_opener()
            closer = self._get_perspective_closer()
            
            # Get styled descriptor (horizontal axis)
            char_desc = self._get_styled_descriptor('character')
            
            # Get elaboration (depth axis)
            elaboration = self._get_depth_elaboration(role)
            
            # Build the answer with all three axes
            if related and self.dial.include_relationship():
                relationship = self._get_perspective_relationship(related)
                
                # Construct based on perspective
                if opener:
                    answer = f"{opener} {entity.title()} is {self._get_perspective_framing(role)} from {source} who {action_desc}, {relationship}{closer}{elaboration}"
                else:
                    answer = f"{entity.title()} is a {char_desc} from {source} who {action_desc}, {relationship}{closer}{elaboration}"
            else:
                if opener:
                    answer = f"{opener} {entity.title()} is {self._get_perspective_framing(role)} from {source} who {action_desc}{closer}{elaboration}"
                else:
                    answer = f"{entity.title()} is a {char_desc} from {source} who {action_desc}{closer}{elaboration}"
            
            # Clean up any double spaces or punctuation issues
            answer = re.sub(r'\s+', ' ', answer)
            answer = re.sub(r',\s*\.', '.', answer)
            answer = re.sub(r'\.\s*\.', '.', answer)
            answer = re.sub(r',\s*,', ',', answer)
            
            return answer.strip()
        
        # NOISE + REPROJECTION (noise_level > 0): varied but grounded
        else:
            # Select action (noise = sometimes pick non-top action)
            available_actions = [a for a, _ in actions if a != 'EXIST']
            if not available_actions:
                available_actions = [top_action]
            
            if random.random() < noise_level and len(available_actions) > 1:
                selected_action = random.choice(available_actions)
            else:
                selected_action = available_actions[0] if available_actions else top_action
            
            # Get varied action phrase
            action_phrases = ACTION_PHRASES_VARIED.get(selected_action, ['appears throughout'])
            action_phrase = random.choice(action_phrases)
            
            # Select template
            template = random.choice(WHO_TEMPLATES_VARIED)
            
            # Build relationship phrase
            if related:
                if '{relation_cap}' in template:
                    relation = ''
                    relation_cap = random.choice(RELATIONSHIP_PHRASES_CAP).format(related=related.title())
                else:
                    relation = random.choice(RELATIONSHIP_PHRASES).format(related=related.title())
                    relation_cap = ''
            else:
                relation = ''
                relation_cap = ''
            
            # Build answer
            answer = template.format(
                name=entity.title(),
                role=role,
                source=source,
                action=action_phrase,
                relation=relation,
                relation_cap=relation_cap,
            )
            
            # Clean up any double spaces or trailing commas
            answer = re.sub(r'\s+', ' ', answer)
            answer = re.sub(r',\s*\.', '.', answer)
            answer = re.sub(r'\.\s*\.', '.', answer)
            
            return answer.strip()
    
    def generate_what_answer(
        self,
        entity: str,
        actions: List[Tuple[str, int]],
        patients: List[Tuple[str, int]],
    ) -> str:
        """
        Generate a WHAT DID answer using learned patterns.
        
        Pattern: "[NAME] [ACTION_PAST] and [ACTION2_PAST] regarding [PATIENT]"
        """
        # Get action verbs (skip EXIST)
        action_verbs = []
        for action, count in actions[:3]:
            if action != 'EXIST':
                verb = ACTION_TO_PAST.get(action, 'acted')
                action_verbs.append(verb)
        
        if not action_verbs:
            return f"{entity.title()} appeared in the story"
        
        # Get patient (filter noise)
        patient = None
        for p, count in patients:
            if p.lower() not in self.noise_words and p != entity and len(p) > 2:
                patient = p
                break
        
        # Build answer
        if len(action_verbs) >= 2:
            result = f"{entity.title()} {action_verbs[0]} and {action_verbs[1]}"
        else:
            result = f"{entity.title()} {action_verbs[0]}"
        
        if patient:
            result += f" regarding {patient}"
        
        return result
    
    def generate_where_answer(
        self,
        entity: str,
        locations: set,
    ) -> str:
        """
        Generate a WHERE IS answer using learned patterns.
        
        Pattern: "[NAME] appears at [LOCATION]"
        """
        # Filter noise from locations
        clean_locs = [loc for loc in locations if loc.lower() not in self.noise_words and len(loc) > 2]
        
        if clean_locs:
            loc_str = ", ".join(clean_locs[:3])
            return f"{entity.title()} appears at {loc_str}"
        
        return f"Location of {entity.title()} is not specified"


def test_pattern_generator():
    """Test the pattern-based answer generator."""
    gen = PatternAnswerGenerator()
    
    print("=== TESTING PATTERN GENERATOR ===")
    print()
    
    # Test WHO IS
    answer = gen.generate_who_answer(
        entity="darcy",
        actions=[("POSSESS", 5), ("SPEAK", 3), ("MOVE", 2)],
        patients=[("elizabeth", 4), ("bingley", 2)],
        sources={"Pride and Prejudice"},
    )
    print(f"WHO IS Darcy? -> {answer}")
    
    # Test WHAT DID
    answer = gen.generate_what_answer(
        entity="holmes",
        actions=[("PERCEIVE", 5), ("THINK", 4), ("ACT", 3)],
        patients=[("watson", 3), ("moriarty", 2)],
    )
    print(f"WHAT DID Holmes do? -> {answer}")
    
    # Test WHERE IS
    answer = gen.generate_where_answer(
        entity="alice",
        locations={"wonderland", "garden", "tea party"},
    )
    print(f"WHERE IS Alice? -> {answer}")


if __name__ == "__main__":
    test_pattern_generator()
