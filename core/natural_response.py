"""
Natural Response Generator for GeometricLCM

Generates longer, more natural-sounding responses by:
1. Using multiple sentence structures
2. Adding contextual details from knowledge base
3. Varying openings and transitions
4. Including relevant relationships and actions

This replaces the single-sentence template approach with
multi-sentence, flowing responses.
"""

import random
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# Entity aliases - map full names and variations to canonical keys
ENTITY_ALIASES = {
    'sherlock holmes': 'holmes',
    'sherlock': 'holmes',
    'mr holmes': 'holmes',
    'dr watson': 'watson',
    'dr. watson': 'watson',
    'john watson': 'watson',
    'doctor watson': 'watson',
    'professor moriarty': 'moriarty',
    'james moriarty': 'moriarty',
    'irene adler': 'irene',
    'the woman': 'irene',
    'inspector lestrade': 'lestrade',
    'g lestrade': 'lestrade',
    'mycroft holmes': 'mycroft',
    'mr darcy': 'darcy',
    'fitzwilliam darcy': 'darcy',
    'elizabeth bennet': 'elizabeth',
    'lizzy': 'elizabeth',
    'lizzy bennet': 'elizabeth',
    'miss bennet': 'elizabeth',
    'jane bennet': 'jane',
    'miss jane': 'jane',
    'mr bingley': 'bingley',
    'charles bingley': 'bingley',
    'mr wickham': 'wickham',
    'george wickham': 'wickham',
    'lydia bennet': 'lydia',
    'miss lydia': 'lydia',
}


def normalize_entity(entity: str) -> str:
    """Normalize an entity name to its canonical form."""
    entity_lower = entity.lower().strip()
    
    # Check aliases first
    if entity_lower in ENTITY_ALIASES:
        return ENTITY_ALIASES[entity_lower]
    
    # Check if it's already a canonical key
    if entity_lower in CHARACTER_PROFILES:
        return entity_lower
    
    # Try partial matching (e.g., "sherlock" in "sherlock holmes")
    for alias, canonical in ENTITY_ALIASES.items():
        if alias in entity_lower or entity_lower in alias:
            return canonical
    
    return entity_lower


# Rich character profiles with detailed information
CHARACTER_PROFILES = {
    'holmes': {
        'full_name': 'Sherlock Holmes',
        'role': 'consulting detective',
        'source': 'the Sherlock Holmes stories by Sir Arthur Conan Doyle',
        'qualities': ['brilliant', 'observant', 'eccentric', 'analytical'],
        'actions': ['solves complex mysteries', 'uses deductive reasoning', 'observes minute details others miss'],
        'relationships': {
            'watson': 'his loyal friend and chronicler Dr. John Watson',
            'moriarty': 'his arch-nemesis Professor Moriarty',
            'mycroft': 'his brother Mycroft Holmes',
            'lestrade': 'Inspector Lestrade of Scotland Yard',
            'irene': 'Irene Adler, the only woman to ever outwit him',
        },
        'notable_traits': [
            'He is known for his remarkable powers of observation and logical reasoning.',
            'He often plays the violin and conducts chemical experiments in his Baker Street flat.',
            'His methods of deduction have become legendary in detective fiction.',
        ],
        'residence': '221B Baker Street, London',
    },
    'sherlock': {  # Alias
        'alias_of': 'holmes',
    },
    'watson': {
        'full_name': 'Dr. John Watson',
        'role': 'physician and chronicler',
        'source': 'the Sherlock Holmes stories by Sir Arthur Conan Doyle',
        'qualities': ['loyal', 'brave', 'practical', 'compassionate'],
        'actions': ['assists Holmes in investigations', 'documents their adventures', 'provides medical expertise'],
        'relationships': {
            'holmes': 'his brilliant friend Sherlock Holmes',
            'mary': 'his wife Mary Morstan',
        },
        'notable_traits': [
            'He served as an army doctor in Afghanistan before meeting Holmes.',
            'His narratives of their cases have made Holmes famous throughout London.',
            'He provides a grounded, human perspective to Holmes\'s cold logic.',
        ],
        'residence': '221B Baker Street, London (initially)',
    },
    'moriarty': {
        'full_name': 'Professor James Moriarty',
        'role': 'criminal mastermind',
        'source': 'the Sherlock Holmes stories by Sir Arthur Conan Doyle',
        'qualities': ['genius', 'ruthless', 'calculating', 'dangerous'],
        'actions': ['orchestrates crimes from the shadows', 'controls a vast criminal network'],
        'relationships': {
            'holmes': 'his intellectual rival Sherlock Holmes',
        },
        'notable_traits': [
            'Holmes considers him the Napoleon of crime.',
            'He is a former mathematics professor with a brilliant but twisted mind.',
            'Their confrontation at Reichenbach Falls is one of literature\'s most famous scenes.',
        ],
    },
    'irene': {
        'full_name': 'Irene Adler',
        'role': 'opera singer and adventuress',
        'source': 'the Sherlock Holmes story "A Scandal in Bohemia"',
        'qualities': ['clever', 'beautiful', 'resourceful', 'independent'],
        'actions': ['outwitted Sherlock Holmes', 'protected herself from powerful enemies'],
        'relationships': {
            'holmes': 'Sherlock Holmes, whom she famously outsmarted',
        },
        'notable_traits': [
            'She is the only person Holmes refers to as "The Woman."',
            'Her intelligence and cunning earned Holmes\'s lasting respect.',
            'She represents a rare defeat for the great detective.',
        ],
    },
    'lestrade': {
        'full_name': 'Inspector G. Lestrade',
        'role': 'Scotland Yard detective',
        'source': 'the Sherlock Holmes stories by Sir Arthur Conan Doyle',
        'qualities': ['determined', 'conventional', 'persistent'],
        'actions': ['investigates crimes for Scotland Yard', 'often consults Holmes on difficult cases'],
        'relationships': {
            'holmes': 'the consulting detective Sherlock Holmes',
        },
        'notable_traits': [
            'He represents the official police force in contrast to Holmes\'s private methods.',
            'Though sometimes frustrated by Holmes, he respects his abilities.',
        ],
    },
    'mycroft': {
        'full_name': 'Mycroft Holmes',
        'role': 'government official',
        'source': 'the Sherlock Holmes stories by Sir Arthur Conan Doyle',
        'qualities': ['brilliant', 'lazy', 'influential', 'observant'],
        'actions': ['works in a crucial government position', 'occasionally assists his brother'],
        'relationships': {
            'holmes': 'his younger brother Sherlock Holmes',
        },
        'notable_traits': [
            'Sherlock admits Mycroft has superior powers of observation.',
            'He prefers the comfort of his club to active investigation.',
            'His position in the British government is never fully explained but clearly important.',
        ],
    },
    'darcy': {
        'full_name': 'Fitzwilliam Darcy',
        'role': 'wealthy gentleman',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['proud', 'reserved', 'honorable', 'wealthy'],
        'actions': ['overcomes his pride for love', 'proves his true character through actions'],
        'relationships': {
            'elizabeth': 'Elizabeth Bennet, whom he comes to love deeply',
            'bingley': 'his close friend Charles Bingley',
            'wickham': 'George Wickham, who wronged his family',
            'georgiana': 'his beloved younger sister Georgiana',
        },
        'notable_traits': [
            'His initial pride and aloofness mask a deeply honorable character.',
            'He owns the grand estate of Pemberley in Derbyshire.',
            'His transformation through love is central to the novel\'s themes.',
        ],
        'residence': 'Pemberley, Derbyshire',
    },
    'elizabeth': {
        'full_name': 'Elizabeth Bennet',
        'role': 'gentleman\'s daughter',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['witty', 'intelligent', 'spirited', 'independent-minded'],
        'actions': ['challenges social conventions', 'overcomes her prejudices', 'stands up for herself'],
        'relationships': {
            'darcy': 'Mr. Darcy, whom she initially misjudges',
            'jane': 'her beloved elder sister Jane',
            'wickham': 'Mr. Wickham, who deceives her',
        },
        'notable_traits': [
            'She is the second of five daughters in the Bennet family.',
            'Her sharp wit and refusal to marry for convenience set her apart.',
            'Her journey from prejudice to understanding mirrors Darcy\'s from pride to humility.',
        ],
        'residence': 'Longbourn, Hertfordshire',
    },
    'jane': {
        'full_name': 'Jane Bennet',
        'role': 'gentleman\'s daughter',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['beautiful', 'kind', 'gentle', 'trusting'],
        'actions': ['sees the best in everyone', 'loves Mr. Bingley faithfully'],
        'relationships': {
            'elizabeth': 'her closest sister Elizabeth',
            'bingley': 'Mr. Bingley, whom she loves',
        },
        'notable_traits': [
            'She is the eldest and most beautiful of the Bennet sisters.',
            'Her gentle nature sometimes leads others to underestimate her feelings.',
        ],
    },
    'bingley': {
        'full_name': 'Charles Bingley',
        'role': 'wealthy gentleman',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['amiable', 'cheerful', 'good-natured', 'easily influenced'],
        'actions': ['falls in love with Jane Bennet', 'rents Netherfield Park'],
        'relationships': {
            'jane': 'Jane Bennet, whom he loves',
            'darcy': 'his close friend Mr. Darcy',
        },
        'notable_traits': [
            'His arrival at Netherfield sets the story in motion.',
            'He is more open and sociable than his reserved friend Darcy.',
        ],
    },
    'wickham': {
        'full_name': 'George Wickham',
        'role': 'militia officer',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['charming', 'deceitful', 'manipulative', 'unscrupulous'],
        'actions': ['spreads lies about Darcy', 'elopes with Lydia Bennet'],
        'relationships': {
            'darcy': 'Mr. Darcy, whom he has wronged',
            'lydia': 'Lydia Bennet, whom he elopes with',
        },
        'notable_traits': [
            'His charming exterior hides a deeply dishonest character.',
            'He attempted to elope with Darcy\'s young sister for her fortune.',
        ],
    },
    'lydia': {
        'full_name': 'Lydia Bennet',
        'role': 'gentleman\'s daughter',
        'source': 'Pride and Prejudice by Jane Austen',
        'qualities': ['flirtatious', 'thoughtless', 'spirited', 'immature'],
        'actions': ['pursues officers', 'elopes with Wickham'],
        'relationships': {
            'wickham': 'Mr. Wickham, whom she elopes with',
            'elizabeth': 'her sister Elizabeth',
        },
        'notable_traits': [
            'She is the youngest of the Bennet sisters.',
            'Her reckless elopement nearly ruins the family\'s reputation.',
        ],
    },
}


# Response templates for variety
OPENING_TEMPLATES = [
    "{name} is {article} {role} from {source}.",
    "In {source}, {name} appears as {article} {role}.",
    "{name}, {article} {role} in {source}, is one of the most memorable characters.",
    "One of the central figures in {source} is {name}, {article} {role}.",
]

QUALITY_TEMPLATES = [
    "Known for being {qualities}, {pronoun} {action}.",
    "{pronoun_cap} is characterized by {pronoun_poss} {qualities} nature.",
    "Described as {qualities}, {pronoun} stands out among the characters.",
]

RELATIONSHIP_TEMPLATES = [
    "{pronoun_cap} has a significant connection to {related}.",
    "Central to {pronoun_poss} story is {pronoun_poss} relationship with {related}.",
    "{pronoun_cap} is closely associated with {related}.",
]

TRAIT_TEMPLATES = [
    "{trait}",
]

CLOSING_TEMPLATES = [
    "{pronoun_cap} remains one of the most {quality} characters in {genre} literature.",
    "Throughout the story, {pronoun} demonstrates {quality}.",
]


class NaturalResponseGenerator:
    """
    Generates natural, multi-sentence responses about characters.
    """
    
    def __init__(self, knowledge=None):
        """
        Initialize with optional knowledge base for additional context.
        """
        self.knowledge = knowledge
        self.profiles = CHARACTER_PROFILES
    
    def get_profile(self, entity: str) -> Optional[Dict]:
        """Get profile for an entity, following aliases."""
        # Normalize the entity name first
        canonical = normalize_entity(entity)
        
        if canonical in self.profiles:
            profile = self.profiles[canonical]
            # Follow alias (for profiles that point to other profiles)
            if 'alias_of' in profile:
                return self.profiles.get(profile['alias_of'])
            return profile
        
        return None
    
    def generate_who_response(self, entity: str, max_sentences: int = 4, 
                               depth: float = 0.0) -> str:
        """
        Generate a natural "Who is X?" response.
        
        Args:
            entity: The entity to describe
            max_sentences: Maximum sentences to include
            depth: -1 (terse) to +1 (elaborate)
        """
        profile = self.get_profile(entity)
        
        if not profile:
            return self._generate_unknown_response(entity)
        
        # Adjust sentence count based on depth
        if depth < -0.3:
            max_sentences = 2
        elif depth > 0.3:
            max_sentences = 6
        
        sentences = []
        
        # 1. Opening sentence (always include)
        opening = self._generate_opening(profile)
        sentences.append(opening)
        
        # 2. Qualities sentence
        if len(sentences) < max_sentences and profile.get('qualities'):
            qualities_sent = self._generate_qualities(profile)
            sentences.append(qualities_sent)
        
        # 3. Notable trait
        if len(sentences) < max_sentences and profile.get('notable_traits'):
            trait = random.choice(profile['notable_traits'])
            sentences.append(trait)
        
        # 4. Key relationship
        if len(sentences) < max_sentences and profile.get('relationships'):
            rel_sent = self._generate_relationship(profile)
            if rel_sent:
                sentences.append(rel_sent)
        
        # 5. Another notable trait (for elaborate mode)
        if len(sentences) < max_sentences and profile.get('notable_traits') and len(profile['notable_traits']) > 1:
            remaining_traits = [t for t in profile['notable_traits'] if t not in sentences]
            if remaining_traits:
                sentences.append(random.choice(remaining_traits))
        
        return ' '.join(sentences)
    
    def _generate_opening(self, profile: Dict) -> str:
        """Generate the opening sentence."""
        name = profile['full_name']
        role = profile['role']
        source = profile['source']
        
        # Determine article
        article = 'an' if role[0].lower() in 'aeiou' else 'a'
        
        template = random.choice(OPENING_TEMPLATES)
        return template.format(
            name=name,
            article=article,
            role=role,
            source=source,
        )
    
    def _generate_qualities(self, profile: Dict) -> str:
        """Generate a sentence about qualities."""
        qualities = profile.get('qualities', [])
        if not qualities:
            return ""
        
        # Format qualities list
        if len(qualities) == 1:
            qualities_str = qualities[0]
        elif len(qualities) == 2:
            qualities_str = f"{qualities[0]} and {qualities[1]}"
        else:
            qualities_str = f"{', '.join(qualities[:-1])}, and {qualities[-1]}"
        
        # Get pronoun based on character (simplified - could be enhanced)
        pronoun = self._get_pronoun(profile)
        pronoun_cap = pronoun.capitalize()
        pronoun_poss = 'his' if pronoun == 'he' else 'her'
        
        # Get an action
        actions = profile.get('actions', ['appears throughout the story'])
        action = actions[0] if actions else 'appears throughout the story'
        
        template = random.choice(QUALITY_TEMPLATES)
        return template.format(
            qualities=qualities_str,
            pronoun=pronoun,
            pronoun_cap=pronoun_cap,
            pronoun_poss=pronoun_poss,
            action=action,
        )
    
    def _generate_relationship(self, profile: Dict) -> str:
        """Generate a sentence about a key relationship."""
        relationships = profile.get('relationships', {})
        if not relationships:
            return ""
        
        # Pick a key relationship
        key, description = random.choice(list(relationships.items()))
        
        pronoun = self._get_pronoun(profile)
        pronoun_cap = pronoun.capitalize()
        pronoun_poss = 'his' if pronoun == 'he' else 'her'
        
        template = random.choice(RELATIONSHIP_TEMPLATES)
        return template.format(
            pronoun=pronoun,
            pronoun_cap=pronoun_cap,
            pronoun_poss=pronoun_poss,
            related=description,
        )
    
    def _get_pronoun(self, profile: Dict) -> str:
        """Determine pronoun for character."""
        # Simple heuristic based on role/name
        female_indicators = ['lady', 'woman', 'daughter', 'sister', 'wife', 'queen', 'princess', 'duchess']
        role = profile.get('role', '').lower()
        
        for indicator in female_indicators:
            if indicator in role:
                return 'she'
        
        # Check name for common female names
        name = profile.get('full_name', '').lower()
        female_names = ['elizabeth', 'jane', 'irene', 'lydia', 'mary', 'georgiana', 'charlotte']
        for fname in female_names:
            if fname in name:
                return 'she'
        
        return 'he'
    
    def _generate_unknown_response(self, entity: str) -> str:
        """Generate response for unknown entity."""
        return (f"I don't have detailed information about {entity.title()} in my knowledge base. "
                f"I can tell you about characters from Sherlock Holmes (like Holmes, Watson, or Moriarty) "
                f"or Pride and Prejudice (like Darcy, Elizabeth, or Jane).")


# Global instance
_generator: Optional[NaturalResponseGenerator] = None


def get_natural_generator(knowledge=None) -> NaturalResponseGenerator:
    """Get the natural response generator."""
    global _generator
    if _generator is None:
        _generator = NaturalResponseGenerator(knowledge)
    return _generator


def generate_character_response(entity: str, depth: float = 0.0) -> Optional[str]:
    """
    Convenience function to generate a character response.
    
    Returns None if entity is not known.
    """
    generator = get_natural_generator()
    profile = generator.get_profile(entity)
    
    if profile:
        return generator.generate_who_response(entity, depth=depth)
    
    return None
