#!/usr/bin/env python3
"""
Holographic Style Projection Layer

Transforms concept space output into natural prose with literary style.

The key insight: Style is a PROJECTION operation.
- Concept space contains WHAT to say (role, traits, actions)
- Style space contains HOW to say it (patterns, rhythm, vocabulary)

We project from concept space through style space to get natural output.

This is analogous to holographic reconstruction:
- Reference beam = concept content
- Object beam = style patterns
- Interference = natural prose
"""

import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class StylePattern:
    """A reusable style pattern extracted from literature."""
    template: str           # Pattern with {slots}
    category: str           # intro, trait, action, relationship, closing
    formality: float        # 0 (casual) to 1 (formal)
    literary_weight: float  # How "literary" vs plain
    source: str             # Where it came from


# Bootstrap style patterns for "high school book report" style
# These capture common literary analysis patterns
BOOK_REPORT_PATTERNS = {
    'intro': [
        "{name} is {article} {role} in {source}, {author}'s {genre} {work_type}.",
        "In {source}, {name} serves as {article} {role} who {key_action}.",
        "One of the most {quality} characters in {source} is {name}, {article} {role}.",
        "{name}, {article} {role} in {source}, {trait_clause}.",
        "The character of {name} in {source} represents {article} {role}.",
    ],
    'trait': [
        "{pronoun_cap} is characterized by {pronoun_poss} {quality} nature.",
        "{pronoun_cap} demonstrates {quality} throughout the narrative.",
        "The reader sees {pronoun_poss} {quality} personality in {pronoun_poss} interactions.",
        "{pronoun_cap} exhibits {quality} behavior, particularly when {action_context}.",
        "What makes {name} distinctive is {pronoun_poss} {quality} approach to {domain}.",
    ],
    'action': [
        "{pronoun_cap} {action_verb} throughout the story, {consequence}.",
        "Throughout the novel, {pronoun} {action_verb}, which {effect}.",
        "{pronoun_cap} is often seen {action_gerund}, {elaboration}.",
        "The narrative shows {pronoun_obj} {action_gerund} in various situations.",
        "{pronoun_poss} tendency to {action_verb} reveals {insight}.",
    ],
    'relationship': [
        "{pronoun_poss} relationship with {other} is central to the plot.",
        "{pronoun_cap} and {other} share {relationship_type}, which {significance}.",
        "The dynamic between {name} and {other} {relationship_verb}.",
        "{pronoun_cap} is closely associated with {other}, {relationship_detail}.",
    ],
    'closing': [
        "{pronoun_cap} remains one of the most {quality} characters in {genre} literature.",
        "Through {name}, {author} explores themes of {theme}.",
        "Ultimately, {name} represents {representation} in the story.",
        "{pronoun_poss} role in {source} demonstrates {lesson}.",
    ],
}

# Literary vocabulary for different concepts
LITERARY_VOCABULARY = {
    'investigator': {
        'qualities': ['analytical', 'perceptive', 'methodical', 'brilliant', 'observant'],
        'actions': ['deduces', 'investigates', 'uncovers', 'solves', 'examines'],
        'gerunds': ['solving mysteries', 'analyzing clues', 'pursuing the truth'],
        'domains': ['crime-solving', 'investigation', 'deduction'],
    },
    'narrator': {
        'qualities': ['reliable', 'thoughtful', 'loyal', 'grounded', 'steadfast'],
        'actions': ['recounts', 'chronicles', 'witnesses', 'describes', 'narrates'],
        'gerunds': ['telling the story', 'recording events', 'chronicling adventures'],
        'domains': ['storytelling', 'narrative', 'companionship'],
    },
    'adventurer': {
        'qualities': ['spirited', 'curious', 'bold', 'imaginative', 'resourceful'],
        'actions': ['explores', 'discovers', 'ventures', 'seeks', 'embarks'],
        'gerunds': ['seeking adventure', 'exploring new places', 'pushing boundaries'],
        'domains': ['adventure', 'exploration', 'discovery'],
    },
    'curious_observer': {
        'qualities': ['inquisitive', 'wondering', 'perceptive', 'innocent', 'questioning'],
        'actions': ['questions', 'wonders', 'observes', 'discovers', 'encounters'],
        'gerunds': ['questioning everything', 'observing the strange', 'navigating the unknown'],
        'domains': ['curiosity', 'wonder', 'discovery'],
    },
    'romantic_figure': {
        'qualities': ['proud', 'reserved', 'honorable', 'complex', 'passionate'],
        'actions': ['struggles', 'transforms', 'reveals', 'overcomes', 'loves'],
        'gerunds': ['overcoming pride', 'revealing true feelings', 'growing as a person'],
        'domains': ['love', 'personal growth', 'social expectations'],
    },
    'protagonist': {
        'qualities': ['determined', 'spirited', 'intelligent', 'independent', 'witty'],
        'actions': ['challenges', 'grows', 'overcomes', 'learns', 'triumphs'],
        'gerunds': ['facing challenges', 'growing stronger', 'finding her way'],
        'domains': ['personal growth', 'self-discovery', 'social navigation'],
    },
}

# Source metadata for richer descriptions
SOURCE_METADATA = {
    'sherlock holmes': {
        'author': 'Arthur Conan Doyle',
        'genre': 'detective',
        'work_type': 'stories',
        'themes': ['logic', 'observation', 'justice'],
    },
    'pride and prejudice': {
        'author': 'Jane Austen',
        'genre': 'romantic',
        'work_type': 'novel',
        'themes': ['love', 'class', 'personal growth'],
    },
    'alice in wonderland': {
        'author': 'Lewis Carroll',
        'genre': 'fantasy',
        'work_type': 'novel',
        'themes': ['identity', 'logic', 'growing up'],
    },
    'tom sawyer': {
        'author': 'Mark Twain',
        'genre': 'adventure',
        'work_type': 'novel',
        'themes': ['childhood', 'freedom', 'morality'],
    },
    'great expectations': {
        'author': 'Charles Dickens',
        'genre': 'coming-of-age',
        'work_type': 'novel',
        'themes': ['ambition', 'identity', 'redemption'],
    },
    'moby dick': {
        'author': 'Herman Melville',
        'genre': 'adventure',
        'work_type': 'novel',
        'themes': ['obsession', 'nature', 'fate'],
    },
}


class HolographicStyleProjector:
    """
    Projects concept space content through style patterns to produce natural prose.
    
    This is a holographic operation:
    1. Content beam: What we want to say (from Tachyon reasoning)
    2. Style beam: How we want to say it (patterns from literature)
    3. Interference: Natural prose output
    """
    
    def __init__(self, style: str = "book_report"):
        """
        Initialize with a target style.
        
        Args:
            style: Target style ("book_report", "literary", "casual")
        """
        self.style = style
        self.patterns = BOOK_REPORT_PATTERNS
        self.vocabulary = LITERARY_VOCABULARY
        self.source_meta = SOURCE_METADATA
    
    def project(self, content: Dict, depth: float = 0.0) -> str:
        """
        Project concept content through style to produce natural prose.
        
        Args:
            content: Dict with keys: name, role, gender, source, key_features, etc.
            depth: -1 (terse) to +1 (elaborate)
            
        Returns:
            Styled natural language output
        """
        # Extract content
        name = content.get('name', 'The character').title()
        role = content.get('role', 'character').replace('_', ' ')
        gender = content.get('gender', 'male')
        source = content.get('source', 'the story')
        key_features = content.get('key_features', [])
        confidence = content.get('confidence', 'medium')
        
        # Determine pronouns
        pronoun = 'she' if gender == 'female' else 'he'
        pronoun_cap = pronoun.capitalize()
        pronoun_poss = 'her' if gender == 'female' else 'his'
        pronoun_obj = 'her' if gender == 'female' else 'him'
        
        # Get source metadata
        source_lower = source.lower()
        meta = self._get_source_metadata(source_lower)
        
        # Get role vocabulary
        vocab = self.vocabulary.get(role.replace(' ', '_'), self.vocabulary.get('protagonist', {}))
        
        # Build sentences based on depth
        sentences = []
        
        # 1. Introduction (always)
        intro = self._generate_intro(name, role, source, meta, vocab, pronoun_poss)
        sentences.append(intro)
        
        # 2. Trait sentence (if not terse)
        if depth >= -0.3:
            trait = self._generate_trait(name, role, vocab, pronoun, pronoun_cap, pronoun_poss)
            sentences.append(trait)
        
        # 3. Action/feature sentence (normal and above)
        if depth >= 0.0 and key_features:
            action = self._generate_action(name, key_features, vocab, pronoun, pronoun_cap, pronoun_poss)
            if action:
                sentences.append(action)
        
        # 4. Closing (elaborate only)
        if depth >= 0.5:
            closing = self._generate_closing(name, role, meta, vocab, pronoun, pronoun_cap, pronoun_poss)
            sentences.append(closing)
        
        return ' '.join(sentences)
    
    def _get_source_metadata(self, source: str) -> Dict:
        """Get metadata for a source, with fallback defaults."""
        source_lower = source.lower()
        
        # Try exact match
        if source_lower in self.source_meta:
            return self.source_meta[source_lower]
        
        # Try partial match - prefer longer matches
        best_match = None
        best_len = 0
        for key, meta in self.source_meta.items():
            if key in source_lower or source_lower in key:
                if len(key) > best_len:
                    best_match = meta
                    best_len = len(key)
        
        if best_match:
            return best_match
        
        # Default
        return {
            'author': 'the author',
            'genre': 'classic',
            'work_type': 'work',
            'themes': ['human nature', 'society'],
        }
    
    def _format_source(self, source: str) -> str:
        """Format source name with proper title case."""
        # Handle special cases
        source_lower = source.lower()
        
        # Known proper titles
        proper_titles = {
            'sherlock holmes': 'Sherlock Holmes',
            'pride and prejudice': 'Pride and Prejudice',
            'alice in wonderland': 'Alice in Wonderland',
            'tom sawyer': 'Tom Sawyer',
            'great expectations': 'Great Expectations',
            'moby dick': 'Moby Dick',
            'tale of two cities': 'A Tale of Two Cities',
            'les miserables': 'Les Misérables',
            'dracula': 'Dracula',
            'frankenstein': 'Frankenstein',
            'white fang': 'White Fang',
            'the valley of fear': 'The Valley of Fear',
            'the hound of the baskervilles': 'The Hound of the Baskervilles',
            'the adventures of sherlock holmes': 'The Adventures of Sherlock Holmes',
            'the sign of the four': 'The Sign of the Four',
        }
        
        for key, proper in proper_titles.items():
            if key in source_lower:
                return proper
        
        # Default: title case but keep small words lowercase
        words = source.split()
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'of', 'in'}
        result = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        return ' '.join(result)
    
    def _get_article(self, word: str) -> str:
        """Get correct article (a/an) for a word."""
        if not word:
            return 'a'
        # Check first letter of the word
        return 'an' if word[0].lower() in 'aeiou' else 'a'
    
    def _generate_intro(self, name: str, role: str, source: str, 
                        meta: Dict, vocab: Dict, pronoun_poss: str) -> str:
        """Generate introduction sentence."""
        article = self._get_article(role)
        
        # Format source properly
        source_fmt = self._format_source(source)
        
        # Pick a quality for the intro
        qualities = vocab.get('qualities', ['notable'])
        quality = random.choice(qualities[:3])  # Prefer first few (most fitting)
        quality_article = self._get_article(quality)
        
        # Pick an action for context
        actions = vocab.get('actions', ['appears'])
        key_action = f"{actions[0]} throughout the narrative"
        
        templates = [
            f"{name} is {article} {role} in {source_fmt}, {meta['author']}'s {meta['genre']} {meta['work_type']}.",
            f"In {source_fmt}, {name} serves as {quality_article} {quality} {role}.",
            f"One of the most {quality} characters in {source_fmt} is {name}, {article} {role}.",
        ]
        
        return random.choice(templates)
    
    def _generate_trait(self, name: str, role: str, vocab: Dict,
                        pronoun: str, pronoun_cap: str, pronoun_poss: str) -> str:
        """Generate trait description sentence."""
        qualities = vocab.get('qualities', ['notable'])
        quality = random.choice(qualities)
        quality_article = self._get_article(quality)
        
        domains = vocab.get('domains', ['the story'])
        domain = random.choice(domains)
        
        templates = [
            f"{pronoun_cap} is characterized by {pronoun_poss} {quality} nature.",
            f"{pronoun_cap} demonstrates {quality_article} {quality} personality throughout the narrative.",
            f"What makes {name} distinctive is {pronoun_poss} {quality} approach to {domain}.",
        ]
        
        return random.choice(templates)
    
    def _generate_action(self, name: str, key_features: List[str], vocab: Dict,
                         pronoun: str, pronoun_cap: str, pronoun_poss: str) -> Optional[str]:
        """Generate action/feature sentence based on Tachyon evidence."""
        if not key_features:
            return None
        
        # Parse the key feature to extract meaning
        feature = key_features[0].lower()
        
        gerunds = vocab.get('gerunds', ['acting in the story'])
        gerund = random.choice(gerunds)
        
        actions = vocab.get('actions', ['acts'])
        action = random.choice(actions)
        
        # Map features to natural descriptions
        if 'authority' in feature or 'authorities' in feature:
            return f"{pronoun_cap} frequently interacts with authority figures, demonstrating {pronoun_poss} professional connections."
        elif 'speaks' in feature or 'frequently' in feature:
            return f"Throughout the narrative, {pronoun} {action}, providing insight into the events."
        elif 'observant' in feature or 'observes' in feature:
            return f"{pronoun_cap} is notably observant, {gerund} with keen attention to detail."
        elif 'unusual' in feature or 'fantastical' in feature:
            return f"{pronoun_cap} encounters unusual and fantastical situations, {gerund}."
        elif 'moves' in feature or 'active' in feature:
            return f"{pronoun_cap} is an active presence in the story, constantly {gerund}."
        elif 'family' in feature or 'friends' in feature:
            return f"{pronoun_poss} relationships with family and friends form a central part of the narrative."
        elif 'described' in feature or 'featured' in feature:
            return f"{pronoun_cap} is prominently featured, with the author devoting significant attention to {pronoun_poss} character."
        else:
            return f"{pronoun_cap} plays a significant role, {gerund}."
    
    def _generate_closing(self, name: str, role: str, meta: Dict, vocab: Dict,
                          pronoun: str, pronoun_cap: str, pronoun_poss: str) -> str:
        """Generate closing sentence."""
        qualities = vocab.get('qualities', ['memorable'])
        quality = random.choice(qualities)
        
        themes = meta.get('themes', ['human nature'])
        theme = random.choice(themes)
        
        # Capitalize possessive properly
        pronoun_poss_cap = pronoun_poss.capitalize()
        
        templates = [
            f"{pronoun_cap} remains one of the most {quality} characters in {meta['genre']} literature.",
            f"Through {name}, {meta['author']} explores themes of {theme}.",
            f"{pronoun_poss_cap} role demonstrates the enduring appeal of the {role} archetype.",
        ]
        
        return random.choice(templates)


def project_with_style(content: Dict, style: str = "book_report", depth: float = 0.0) -> str:
    """
    Convenience function to project content through style.
    
    Args:
        content: Dict with name, role, gender, source, key_features
        style: Target style
        depth: -1 to +1
        
    Returns:
        Styled natural language
    """
    projector = HolographicStyleProjector(style)
    return projector.project(content, depth)
