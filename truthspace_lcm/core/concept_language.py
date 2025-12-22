#!/usr/bin/env python3
"""
Concept Language: Order-Free Semantic Representation

Inspired by Chinese:
- No verb conjugation (走 = walk/walked/walking)
- Context determines tense/aspect
- Word order is flexible
- Meaning from semantic atoms, not morphology

The concept language is a UNIVERSAL INTERLINGUA:
- Surface text (English/Spanish/etc) -> Concept Frame
- Concept Frame -> Storage (language-agnostic vectors)
- Query -> Concept Frame -> Match -> Answer

This sidesteps word order issues (SVO vs VSO vs SOV) by working
with UNORDERED semantic slots.
"""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


# =============================================================================
# CONCEPTUAL PRIMITIVES (Language-Agnostic)
# =============================================================================

# Action primitives - the universal "verbs"
ACTION_PRIMITIVES = {
    'MOVE',      # Motion through space
    'SPEAK',     # Verbal communication
    'THINK',     # Cognitive processes
    'PERCEIVE',  # Sensory perception
    'FEEL',      # Emotional states
    'ACT',       # Physical actions
    'EXIST',     # States of being
    'POSSESS',   # Having/owning
    'CAUSE',     # Causation
    'BECOME',    # Change of state
}

# Semantic roles - the universal "cases"
SEMANTIC_ROLES = {
    'AGENT',      # Who performs the action
    'PATIENT',    # Who/what is affected
    'THEME',      # What is moved/changed
    'LOCATION',   # Where
    'SOURCE',     # From where
    'GOAL',       # To where
    'TIME',       # When
    'MANNER',     # How
    'INSTRUMENT', # With what
    'PURPOSE',    # Why
    'BENEFICIARY',# For whom
}

# Aspect markers (not tense - aspect is more universal)
ASPECTS = {
    'PERFECTIVE',   # Completed action
    'IMPERFECTIVE', # Ongoing action
    'HABITUAL',     # Repeated action
    'STATIVE',      # State (no change)
}


# =============================================================================
# MULTILINGUAL VERB MAPPINGS
# =============================================================================

# English verbs -> Primitives
ENGLISH_VERBS = {
    # MOVE
    'walk': 'MOVE', 'walked': 'MOVE', 'walking': 'MOVE',
    'run': 'MOVE', 'ran': 'MOVE', 'running': 'MOVE',
    'go': 'MOVE', 'went': 'MOVE', 'going': 'MOVE', 'gone': 'MOVE',
    'come': 'MOVE', 'came': 'MOVE', 'coming': 'MOVE',
    'enter': 'MOVE', 'entered': 'MOVE', 'entering': 'MOVE',
    'leave': 'MOVE', 'left': 'MOVE', 'leaving': 'MOVE',
    'turn': 'MOVE', 'turned': 'MOVE', 'turning': 'MOVE',
    'move': 'MOVE', 'moved': 'MOVE', 'moving': 'MOVE',
    'arrive': 'MOVE', 'arrived': 'MOVE',
    'return': 'MOVE', 'returned': 'MOVE',
    
    # SPEAK
    'say': 'SPEAK', 'said': 'SPEAK', 'saying': 'SPEAK',
    'ask': 'SPEAK', 'asked': 'SPEAK', 'asking': 'SPEAK',
    'tell': 'SPEAK', 'told': 'SPEAK', 'telling': 'SPEAK',
    'reply': 'SPEAK', 'replied': 'SPEAK', 'replying': 'SPEAK',
    'answer': 'SPEAK', 'answered': 'SPEAK',
    'cry': 'SPEAK', 'cried': 'SPEAK', 'crying': 'SPEAK',
    'shout': 'SPEAK', 'shouted': 'SPEAK',
    'whisper': 'SPEAK', 'whispered': 'SPEAK',
    'call': 'SPEAK', 'called': 'SPEAK',
    'exclaim': 'SPEAK', 'exclaimed': 'SPEAK',
    
    # THINK
    'think': 'THINK', 'thought': 'THINK', 'thinking': 'THINK',
    'know': 'THINK', 'knew': 'THINK', 'knowing': 'THINK',
    'believe': 'THINK', 'believed': 'THINK',
    'understand': 'THINK', 'understood': 'THINK',
    'remember': 'THINK', 'remembered': 'THINK',
    'forget': 'THINK', 'forgot': 'THINK',
    'wonder': 'THINK', 'wondered': 'THINK',
    'realize': 'THINK', 'realized': 'THINK',
    'consider': 'THINK', 'considered': 'THINK',
    
    # PERCEIVE
    'see': 'PERCEIVE', 'saw': 'PERCEIVE', 'seeing': 'PERCEIVE', 'seen': 'PERCEIVE',
    'look': 'PERCEIVE', 'looked': 'PERCEIVE', 'looking': 'PERCEIVE',
    'watch': 'PERCEIVE', 'watched': 'PERCEIVE',
    'hear': 'PERCEIVE', 'heard': 'PERCEIVE', 'hearing': 'PERCEIVE',
    'notice': 'PERCEIVE', 'noticed': 'PERCEIVE',
    'observe': 'PERCEIVE', 'observed': 'PERCEIVE',
    'find': 'PERCEIVE', 'found': 'PERCEIVE',
    
    # FEEL
    'love': 'FEEL', 'loved': 'FEEL', 'loving': 'FEEL',
    'hate': 'FEEL', 'hated': 'FEEL',
    'fear': 'FEEL', 'feared': 'FEEL',
    'like': 'FEEL', 'liked': 'FEEL',
    'want': 'FEEL', 'wanted': 'FEEL',
    'hope': 'FEEL', 'hoped': 'FEEL',
    'wish': 'FEEL', 'wished': 'FEEL',
    
    # ACT
    'take': 'ACT', 'took': 'ACT', 'taking': 'ACT', 'taken': 'ACT',
    'give': 'ACT', 'gave': 'ACT', 'giving': 'ACT', 'given': 'ACT',
    'make': 'ACT', 'made': 'ACT', 'making': 'ACT',
    'put': 'ACT', 'putting': 'ACT',
    'hold': 'ACT', 'held': 'ACT', 'holding': 'ACT',
    'open': 'ACT', 'opened': 'ACT', 'opening': 'ACT',
    'close': 'ACT', 'closed': 'ACT',
    'write': 'ACT', 'wrote': 'ACT', 'written': 'ACT',
    'read': 'ACT', 'reading': 'ACT',
    
    # EXIST
    'be': 'EXIST', 'is': 'EXIST', 'was': 'EXIST', 'were': 'EXIST',
    'are': 'EXIST', 'been': 'EXIST', 'being': 'EXIST',
    'seem': 'EXIST', 'seemed': 'EXIST',
    'appear': 'EXIST', 'appeared': 'EXIST',
    'become': 'BECOME', 'became': 'BECOME',
    
    # POSSESS
    'have': 'POSSESS', 'had': 'POSSESS', 'having': 'POSSESS',
    'own': 'POSSESS', 'owned': 'POSSESS',
    'belong': 'POSSESS', 'belonged': 'POSSESS',
}

# Spanish verbs -> Primitives (same primitives!)
SPANISH_VERBS = {
    # MOVE
    'caminar': 'MOVE', 'caminó': 'MOVE', 'caminaba': 'MOVE',
    'correr': 'MOVE', 'corrió': 'MOVE', 'corría': 'MOVE',
    'ir': 'MOVE', 'fue': 'MOVE', 'iba': 'MOVE',
    'venir': 'MOVE', 'vino': 'MOVE', 'venía': 'MOVE',
    'entrar': 'MOVE', 'entró': 'MOVE', 'entraba': 'MOVE',
    'salir': 'MOVE', 'salió': 'MOVE', 'salía': 'MOVE',
    'volver': 'MOVE', 'volvió': 'MOVE', 'volvía': 'MOVE',
    'llegar': 'MOVE', 'llegó': 'MOVE', 'llegaba': 'MOVE',
    'pasar': 'MOVE', 'pasó': 'MOVE', 'pasaba': 'MOVE',
    
    # SPEAK
    'decir': 'SPEAK', 'dijo': 'SPEAK', 'decía': 'SPEAK',
    'preguntar': 'SPEAK', 'preguntó': 'SPEAK', 'preguntaba': 'SPEAK',
    'responder': 'SPEAK', 'respondió': 'SPEAK', 'respondía': 'SPEAK',
    'contar': 'SPEAK', 'contó': 'SPEAK', 'contaba': 'SPEAK',
    'gritar': 'SPEAK', 'gritó': 'SPEAK', 'gritaba': 'SPEAK',
    'llamar': 'SPEAK', 'llamó': 'SPEAK', 'llamaba': 'SPEAK',
    'hablar': 'SPEAK', 'habló': 'SPEAK', 'hablaba': 'SPEAK',
    
    # THINK
    'pensar': 'THINK', 'pensó': 'THINK', 'pensaba': 'THINK',
    'saber': 'THINK', 'supo': 'THINK', 'sabía': 'THINK',
    'creer': 'THINK', 'creyó': 'THINK', 'creía': 'THINK',
    'entender': 'THINK', 'entendió': 'THINK', 'entendía': 'THINK',
    'recordar': 'THINK', 'recordó': 'THINK', 'recordaba': 'THINK',
    'olvidar': 'THINK', 'olvidó': 'THINK', 'olvidaba': 'THINK',
    
    # PERCEIVE
    'ver': 'PERCEIVE', 'vio': 'PERCEIVE', 'veía': 'PERCEIVE',
    'mirar': 'PERCEIVE', 'miró': 'PERCEIVE', 'miraba': 'PERCEIVE',
    'oír': 'PERCEIVE', 'oyó': 'PERCEIVE', 'oía': 'PERCEIVE',
    'escuchar': 'PERCEIVE', 'escuchó': 'PERCEIVE', 'escuchaba': 'PERCEIVE',
    'notar': 'PERCEIVE', 'notó': 'PERCEIVE', 'notaba': 'PERCEIVE',
    'encontrar': 'PERCEIVE', 'encontró': 'PERCEIVE', 'encontraba': 'PERCEIVE',
    'hallar': 'PERCEIVE', 'halló': 'PERCEIVE', 'hallaba': 'PERCEIVE',
    
    # FEEL
    'amar': 'FEEL', 'amó': 'FEEL', 'amaba': 'FEEL',
    'odiar': 'FEEL', 'odió': 'FEEL', 'odiaba': 'FEEL',
    'temer': 'FEEL', 'temió': 'FEEL', 'temía': 'FEEL',
    'querer': 'FEEL', 'quiso': 'FEEL', 'quería': 'FEEL',
    'esperar': 'FEEL', 'esperó': 'FEEL', 'esperaba': 'FEEL',
    
    # ACT
    'tomar': 'ACT', 'tomó': 'ACT', 'tomaba': 'ACT',
    'dar': 'ACT', 'dio': 'ACT', 'daba': 'ACT',
    'hacer': 'ACT', 'hizo': 'ACT', 'hacía': 'ACT',
    'poner': 'ACT', 'puso': 'ACT', 'ponía': 'ACT',
    'abrir': 'ACT', 'abrió': 'ACT', 'abría': 'ACT',
    'cerrar': 'ACT', 'cerró': 'ACT', 'cerraba': 'ACT',
    'escribir': 'ACT', 'escribió': 'ACT', 'escribía': 'ACT',
    'leer': 'ACT', 'leyó': 'ACT', 'leía': 'ACT',
    
    # EXIST
    'ser': 'EXIST', 'es': 'EXIST', 'era': 'EXIST', 'fue': 'EXIST',
    'estar': 'EXIST', 'está': 'EXIST', 'estaba': 'EXIST', 'estuvo': 'EXIST',
    'parecer': 'EXIST', 'pareció': 'EXIST', 'parecía': 'EXIST',
    
    # POSSESS
    'tener': 'POSSESS', 'tiene': 'POSSESS', 'tenía': 'POSSESS', 'tuvo': 'POSSESS',
}

# Relation words -> Semantic roles
ENGLISH_RELATIONS = {
    'in': 'LOCATION', 'into': 'GOAL', 'at': 'LOCATION', 'on': 'LOCATION',
    'to': 'GOAL', 'from': 'SOURCE', 'with': 'INSTRUMENT',
    'for': 'BENEFICIARY', 'about': 'THEME', 'of': 'THEME',
    'by': 'AGENT', 'through': 'PATH',
}

SPANISH_RELATIONS = {
    'en': 'LOCATION', 'a': 'GOAL', 'de': 'SOURCE', 'con': 'INSTRUMENT',
    'para': 'BENEFICIARY', 'por': 'AGENT', 'sobre': 'THEME',
    'hacia': 'GOAL', 'desde': 'SOURCE',
}


# =============================================================================
# CONCEPT FRAME
# =============================================================================

@dataclass
class ConceptFrame:
    """
    Language-agnostic semantic frame.
    
    Like Chinese: no word order, just slots filled with concepts.
    The same frame can be expressed in any language.
    """
    agent: Optional[str] = None       # Who performs
    action: Optional[str] = None      # What primitive action
    patient: Optional[str] = None     # Who/what is affected
    theme: Optional[str] = None       # What is moved/discussed
    location: Optional[str] = None    # Where
    source: Optional[str] = None      # From where
    goal: Optional[str] = None        # To where
    instrument: Optional[str] = None  # With what
    manner: Optional[str] = None      # How
    time: Optional[str] = None        # When
    aspect: str = 'PERFECTIVE'        # Completed by default
    
    def to_vector(self, dim: int = 64) -> np.ndarray:
        """
        Encode frame as a vector.
        
        ORDER-INDEPENDENT: We sum contributions from each slot.
        This is the key to language-agnostic representation.
        """
        vec = np.zeros(dim)
        
        # Each filled slot contributes
        slots = [
            ('AGENT', self.agent),
            ('ACTION', self.action),
            ('PATIENT', self.patient),
            ('THEME', self.theme),
            ('LOCATION', self.location),
            ('SOURCE', self.source),
            ('GOAL', self.goal),
        ]
        
        for role, value in slots:
            if value:
                # Combine role and value into a unique vector
                vec += self._hash_to_vec(f'{role}:{value}', dim)
        
        # Aspect contributes less (it's metadata)
        if self.aspect:
            vec += 0.2 * self._hash_to_vec(f'ASPECT:{self.aspect}', dim)
        
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def _hash_to_vec(self, s: str, dim: int) -> np.ndarray:
        """Deterministic hash to unit vector."""
        seed = hash(s) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        return vec / np.linalg.norm(vec)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (non-None values only)."""
        return {k: v for k, v in self.__dict__.items() if v}
    
    def __repr__(self):
        parts = []
        if self.agent: parts.append(f'AGENT:{self.agent}')
        if self.action: parts.append(f'ACTION:{self.action}')
        if self.patient: parts.append(f'PATIENT:{self.patient}')
        if self.theme: parts.append(f'THEME:{self.theme}')
        if self.location: parts.append(f'LOC:{self.location}')
        if self.goal: parts.append(f'GOAL:{self.goal}')
        return '{' + ', '.join(parts) + '}'


# =============================================================================
# CONCEPT EXTRACTOR
# =============================================================================

class ConceptExtractor:
    """
    Extract concept frames from text in any supported language.
    
    The extraction is ORDER-INDEPENDENT:
    1. Find entities (capitalized words)
    2. Find action (verb -> primitive)
    3. Find relations (prepositions -> roles)
    4. Fill frame slots
    
    Word order doesn't matter - we identify components by their TYPE.
    """
    
    def __init__(self, language: str = 'auto'):
        self.language = language
        
        # Combine all verb mappings
        self.verb_map = {}
        self.verb_map.update(ENGLISH_VERBS)
        self.verb_map.update(SPANISH_VERBS)
        
        # Combine all relation mappings
        self.relation_map = {}
        self.relation_map.update(ENGLISH_RELATIONS)
        self.relation_map.update(SPANISH_RELATIONS)
        
        # Skip words (function words)
        self.skip_words = {
            # English
            'the', 'a', 'an', 'and', 'but', 'or', 'if', 'when', 'then',
            'this', 'that', 'these', 'those', 'it', 'its',
            # Spanish
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'y', 'pero', 'o', 'si', 'cuando', 'entonces',
            'este', 'esta', 'estos', 'estas', 'ese', 'esa',
        }
        
        # Pronouns (need special handling)
        self.pronouns = {
            # English
            'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their',
            'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
            # Spanish
            'él', 'ella', 'ellos', 'ellas', 'yo', 'tú', 'nosotros', 'ustedes',
            'le', 'lo', 'la', 'les', 'los', 'las', 'se',
        }
    
    def extract(self, sentence: str) -> Optional[ConceptFrame]:
        """Extract a concept frame from a sentence."""
        # Tokenize (preserve accented characters)
        words = re.findall(r'[\w\u00C0-\u024F]+', sentence)
        if not words:
            return None
        
        frame = ConceptFrame()
        
        # 1. Find action (verb -> primitive)
        action_idx = -1
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.verb_map:
                frame.action = self.verb_map[word_lower]
                action_idx = i
                break
        
        # 2. Find entities (capitalized, not skip words)
        entities = []
        for i, word in enumerate(words):
            if (word[0].isupper() and 
                word.lower() not in self.skip_words and
                word.lower() not in self.pronouns):
                entities.append((word.lower(), i))
        
        # 3. Assign agent (typically before action, or first entity)
        if entities:
            if action_idx >= 0:
                before = [(e, i) for e, i in entities if i < action_idx]
                after = [(e, i) for e, i in entities if i > action_idx]
                
                if before:
                    frame.agent = before[-1][0]  # Closest before
                elif entities:
                    frame.agent = entities[0][0]  # First entity
                
                if after:
                    frame.patient = after[0][0]  # First after
            else:
                frame.agent = entities[0][0]
                if len(entities) > 1:
                    frame.patient = entities[1][0]
        
        # 4. Find relations and their objects
        for i, word in enumerate(words[:-1]):
            word_lower = word.lower()
            if word_lower in self.relation_map:
                role = self.relation_map[word_lower]
                # Find next content word
                for j in range(i + 1, min(i + 4, len(words))):
                    next_word = words[j].lower()
                    if next_word not in self.skip_words:
                        if role == 'LOCATION':
                            frame.location = next_word
                        elif role == 'GOAL':
                            frame.goal = next_word
                        elif role == 'SOURCE':
                            frame.source = next_word
                        elif role == 'THEME':
                            frame.theme = next_word
                        break
        
        return frame if (frame.action or frame.agent) else None
    
    def extract_all(self, text: str) -> List[ConceptFrame]:
        """Extract frames from all sentences in text."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        frames = []
        for sentence in sentences:
            frame = self.extract(sentence)
            if frame:
                frames.append(frame)
        
        return frames


# =============================================================================
# CONCEPT STORE (Language-Agnostic Storage)
# =============================================================================

class ConceptStore:
    """
    Store concept frames as vectors for retrieval.
    
    The key insight: Storage is LANGUAGE-AGNOSTIC.
    Frames from English and Spanish can be stored together
    and queried in either language.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.frames: List[Tuple[ConceptFrame, np.ndarray, str]] = []
    
    def add(self, frame: ConceptFrame, source_text: str = ''):
        """Add a frame to the store."""
        vec = frame.to_vector(self.dim)
        self.frames.append((frame, vec, source_text))
    
    def query(self, query_frame: ConceptFrame, k: int = 5) -> List[Tuple[ConceptFrame, float, str]]:
        """Find similar frames."""
        query_vec = query_frame.to_vector(self.dim)
        
        results = []
        for frame, vec, source in self.frames:
            sim = np.dot(query_vec, vec)
            results.append((frame, sim, source))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def query_by_role(self, role: str, value: str, k: int = 5) -> List[Tuple[ConceptFrame, str]]:
        """Find frames with a specific role filled."""
        results = []
        for frame, vec, source in self.frames:
            frame_dict = frame.to_dict()
            if role.lower() in frame_dict and frame_dict[role.lower()] == value.lower():
                results.append((frame, source))
        return results[:k]


def test_concept_language():
    """Test the concept language system."""
    extractor = ConceptExtractor()
    store = ConceptStore()
    
    print("=== CONCEPT LANGUAGE TEST ===\n")
    
    # Test sentences in multiple languages
    test_sentences = [
        ("Darcy walked into the room", "English"),
        ("Elizabeth looked at him carefully", "English"),
        ("Holmes examined the evidence", "English"),
        ("Quijote entró en la habitación", "Spanish"),
        ("Sancho miró a su amo", "Spanish"),
        ("Don Quijote pensó en Dulcinea", "Spanish"),
    ]
    
    print("Extraction results:")
    print("-" * 70)
    
    for sentence, lang in test_sentences:
        frame = extractor.extract(sentence)
        if frame:
            store.add(frame, sentence)
            print(f"[{lang}] \"{sentence}\"")
            print(f"        -> {frame}")
            print()
    
    # Test cross-language query
    print("\nCross-language query test:")
    print("-" * 70)
    
    # Query in English, should match Spanish
    query = ConceptFrame(action='MOVE', location='room')
    results = store.query(query, k=3)
    
    print(f"Query: {query}")
    print("Results:")
    for frame, sim, source in results:
        print(f"  {sim:.3f}: {frame}")
        print(f"         \"{source}\"")


if __name__ == "__main__":
    test_concept_language()
