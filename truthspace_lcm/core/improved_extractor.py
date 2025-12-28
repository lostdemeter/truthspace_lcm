#!/usr/bin/env python3
"""
Improved Concept Extractor for GeometricLCM

Addresses key limitations of the original extractor:
1. Only captures capitalized words as entities
2. Misses common nouns as patients (the lady, the crowd)
3. No relationship type inference
4. No quality scoring

This extractor uses:
- Part-of-speech-like heuristics (without external dependencies)
- Object detection after verbs
- Relationship pattern matching
- Quality scoring for frames
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from .concept_language import ConceptFrame, ACTION_PRIMITIVES, ENGLISH_VERBS


# Common nouns that can be patients (objects of actions)
COMMON_NOUNS = {
    # People
    'man', 'woman', 'lady', 'gentleman', 'boy', 'girl', 'child', 'children',
    'friend', 'enemy', 'stranger', 'visitor', 'guest', 'servant', 'master',
    'doctor', 'detective', 'inspector', 'captain', 'colonel', 'professor',
    'brother', 'sister', 'father', 'mother', 'uncle', 'aunt', 'cousin',
    'husband', 'wife', 'daughter', 'son', 'family', 'people', 'crowd',
    
    # Objects
    'door', 'window', 'room', 'house', 'letter', 'paper', 'book', 'note',
    'hand', 'head', 'face', 'eye', 'eyes', 'voice', 'word', 'words',
    'money', 'gold', 'jewel', 'treasure', 'key', 'box', 'bag',
    'horse', 'carriage', 'train', 'ship', 'boat',
    
    # Abstract
    'truth', 'secret', 'mystery', 'case', 'matter', 'affair', 'business',
    'question', 'answer', 'reason', 'fact', 'evidence', 'clue',
    'love', 'hate', 'fear', 'hope', 'joy', 'sorrow', 'anger',
}

# Determiners that precede nouns
DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'their', 'our'}

# Prepositions that introduce objects
OBJECT_PREPOSITIONS = {'to', 'at', 'with', 'for', 'about', 'from', 'into', 'upon', 'against'}

# Relationship patterns: (pattern, relation_type, subject_group, object_group)
RELATIONSHIP_PATTERNS = [
    # "X is the Y of Z" -> (X, role, Y), (X, belongs_to, Z)
    (r'\b(\w+)\s+(?:is|was)\s+the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)', 'role_of'),
    
    # "X is a Y" -> (X, is_a, Y)
    (r'\b(\w+)\s+(?:is|was)\s+(?:a|an)\s+(\w+)', 'is_a'),
    
    # "X and Y" when both capitalized -> (X, associated_with, Y)
    (r'\b([A-Z]\w+)\s+and\s+([A-Z]\w+)', 'associated'),
    
    # "X said to Y" -> (X, spoke_to, Y)
    (r'\b(\w+)\s+(?:said|spoke|talked|whispered)\s+to\s+(\w+)', 'spoke_to'),
    
    # "X looked at Y" -> (X, observed, Y)
    (r'\b(\w+)\s+(?:looked|gazed|stared|glanced)\s+at\s+(\w+)', 'observed'),
    
    # "X went to Y" -> (X, traveled_to, Y)
    (r'\b(\w+)\s+(?:went|came|traveled|journeyed|walked|ran)\s+to\s+(\w+)', 'traveled_to'),
]

# Quality indicators in text
QUALITY_WORDS = {
    'brilliant', 'clever', 'intelligent', 'wise', 'cunning', 'shrewd',
    'observant', 'perceptive', 'analytical', 'logical', 'rational',
    'kind', 'gentle', 'compassionate', 'loving', 'caring', 'warm',
    'cold', 'cruel', 'harsh', 'stern', 'strict', 'severe',
    'proud', 'humble', 'arrogant', 'modest', 'shy', 'bold', 'brave',
    'loyal', 'faithful', 'treacherous', 'deceitful',
    'strong', 'weak', 'handsome', 'beautiful', 'ugly',
    'witty', 'charming', 'mysterious', 'eccentric', 'peculiar',
    'tall', 'short', 'thin', 'fat', 'young', 'old',
}

# Role words that indicate character type
ROLE_WORDS = {
    'detective': 'detective',
    'inspector': 'inspector', 
    'doctor': 'doctor',
    'physician': 'doctor',
    'captain': 'captain',
    'colonel': 'military',
    'professor': 'academic',
    'lawyer': 'lawyer',
    'banker': 'banker',
    'merchant': 'merchant',
    'servant': 'servant',
    'butler': 'servant',
    'maid': 'servant',
    'king': 'royalty',
    'queen': 'royalty',
    'prince': 'royalty',
    'princess': 'royalty',
    'lord': 'nobility',
    'lady': 'nobility',
    'duke': 'nobility',
    'earl': 'nobility',
    'villain': 'villain',
    'criminal': 'criminal',
    'thief': 'criminal',
    'murderer': 'criminal',
}


@dataclass
class EnrichedFrame(ConceptFrame):
    """Extended frame with additional metadata."""
    quality_score: float = 0.0
    relationship_type: Optional[str] = None
    qualities: List[str] = field(default_factory=list)
    role: Optional[str] = None
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary including enriched fields."""
        d = super().to_dict()
        if self.quality_score > 0:
            d['quality_score'] = self.quality_score
        if self.relationship_type:
            d['relationship_type'] = self.relationship_type
        if self.qualities:
            d['qualities'] = self.qualities
        if self.role:
            d['role'] = self.role
        return d


class ImprovedExtractor:
    """
    Improved concept frame extractor with better entity and relationship detection.
    """
    
    def __init__(self, known_entities: Set[str] = None):
        """
        Initialize the extractor.
        
        Args:
            known_entities: Set of known entity names (improves extraction)
        """
        self.known_entities = {e.lower() for e in (known_entities or set())}
        
        # Build verb map with additional verbs
        self.verb_map = dict(ENGLISH_VERBS)
        # Add missing action verbs
        self.verb_map.update({
            'dashed': 'MOVE', 'rushed': 'MOVE', 'hurried': 'MOVE', 'ran': 'MOVE',
            'jumped': 'MOVE', 'leaped': 'MOVE', 'climbed': 'MOVE', 'fell': 'MOVE',
            'followed': 'MOVE', 'chased': 'MOVE', 'fled': 'MOVE', 'escaped': 'MOVE',
            'examined': 'PERCEIVE', 'studied': 'PERCEIVE', 'inspected': 'PERCEIVE',
            'investigated': 'PERCEIVE', 'searched': 'PERCEIVE', 'discovered': 'PERCEIVE',
            'grabbed': 'ACT', 'seized': 'ACT', 'caught': 'ACT', 'pulled': 'ACT',
            'pushed': 'ACT', 'struck': 'ACT', 'hit': 'ACT', 'killed': 'ACT',
            'shot': 'ACT', 'stabbed': 'ACT', 'attacked': 'ACT', 'fought': 'ACT',
            'loved': 'FEEL', 'hated': 'FEEL', 'feared': 'FEEL', 'admired': 'FEEL',
            'suspected': 'THINK', 'realized': 'THINK', 'concluded': 'THINK',
            'deduced': 'THINK', 'inferred': 'THINK', 'guessed': 'THINK',
        })
        
        # Skip words for entity detection
        self.skip_words = {
            'the', 'a', 'an', 'and', 'but', 'or', 'if', 'when', 'then',
            'this', 'that', 'these', 'those', 'it', 'its',
            'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of',
            'there', 'here', 'where', 'when', 'why', 'how', 'what', 'which',
            'so', 'as', 'no', 'not', 'yes', 'now', 'then',
            'very', 'much', 'more', 'most', 'less', 'too', 'also',
            'however', 'therefore', 'moreover', 'indeed',
            'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their',
            'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
            'mr', 'mrs', 'miss', 'ms', 'dr', 'sir', 'lord', 'lady',
            'chapter', 'part', 'book', 'volume', 'page',
            'oh', 'ah', 'alas', 'yes', 'no',
        }
    
    def add_known_entities(self, entities: Set[str]):
        """Add known entity names."""
        self.known_entities.update(e.lower() for e in entities)
    
    def extract(self, sentence: str) -> Optional[EnrichedFrame]:
        """
        Extract an enriched concept frame from a sentence.
        
        Improvements over basic extractor:
        1. Captures common nouns as patients
        2. Detects relationship types
        3. Extracts qualities
        4. Scores frame quality
        """
        words = re.findall(r'[\w\u00C0-\u024F]+', sentence)
        if not words:
            return None
        
        frame = EnrichedFrame(raw_text=sentence[:200])
        
        # 1. Find action (verb -> primitive)
        action_idx = -1
        action_word = None
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.verb_map:
                frame.action = self.verb_map[word_lower]
                action_idx = i
                action_word = word_lower
                break
        
        # 2. Find entities (capitalized words + known entities)
        entities = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Capitalized and not a skip word
            if word[0].isupper() and word_lower not in self.skip_words:
                entities.append((word_lower, i, 'proper'))
            # Known entity (even if not capitalized)
            elif word_lower in self.known_entities:
                entities.append((word_lower, i, 'known'))
        
        # 3. Find common noun patients (after verb)
        common_patients = []
        if action_idx >= 0:
            # Look for "verb + determiner + noun" or "verb + preposition + determiner + noun"
            for i in range(action_idx + 1, min(action_idx + 6, len(words))):
                word_lower = words[i].lower()
                
                # Check for determiner + noun pattern
                if word_lower in DETERMINERS and i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    if next_word in COMMON_NOUNS:
                        common_patients.append((next_word, i + 1))
                        break
                
                # Check for direct common noun
                if word_lower in COMMON_NOUNS:
                    common_patients.append((word_lower, i))
                    break
        
        # 4. Assign agent and patient
        if entities:
            if action_idx >= 0:
                before = [(e, i, t) for e, i, t in entities if i < action_idx]
                after = [(e, i, t) for e, i, t in entities if i > action_idx]
                
                if before:
                    frame.agent = before[-1][0]
                elif entities:
                    frame.agent = entities[0][0]
                
                # Prefer proper noun patient, fall back to common noun
                if after:
                    frame.patient = after[0][0]
                elif common_patients:
                    frame.patient = common_patients[0][0]
            else:
                frame.agent = entities[0][0]
                if len(entities) > 1:
                    frame.patient = entities[1][0]
        
        # 5. Extract relationship type from patterns
        frame.relationship_type = self._extract_relationship_type(sentence)
        
        # 6. Extract qualities mentioned
        frame.qualities = self._extract_qualities(sentence, frame.agent)
        
        # 7. Extract role if mentioned
        frame.role = self._extract_role(sentence, frame.agent)
        
        # 8. Calculate quality score
        frame.quality_score = self._calculate_quality(frame)
        
        return frame if (frame.action or frame.agent) else None
    
    def _extract_relationship_type(self, sentence: str) -> Optional[str]:
        """Extract relationship type from sentence patterns."""
        sentence_lower = sentence.lower()
        
        for pattern, rel_type in RELATIONSHIP_PATTERNS:
            if re.search(pattern, sentence_lower):
                return rel_type
        
        return None
    
    def _extract_qualities(self, sentence: str, agent: Optional[str]) -> List[str]:
        """Extract quality words that describe the agent."""
        if not agent:
            return []
        
        qualities = []
        sentence_lower = sentence.lower()
        
        # Look for qualities near the agent mention
        agent_pos = sentence_lower.find(agent)
        if agent_pos >= 0:
            # Check words within 50 chars of agent
            context = sentence_lower[max(0, agent_pos - 50):agent_pos + len(agent) + 50]
            for quality in QUALITY_WORDS:
                if quality in context:
                    qualities.append(quality)
        
        return qualities[:3]  # Limit to 3
    
    def _extract_role(self, sentence: str, agent: Optional[str]) -> Optional[str]:
        """Extract role if mentioned near agent."""
        if not agent:
            return None
        
        sentence_lower = sentence.lower()
        
        # Look for "X, the detective" or "detective X" patterns
        for role_word, role_type in ROLE_WORDS.items():
            # Pattern: "agent, the role" or "the role agent"
            if re.search(rf'\b{agent}\b[,\s]+the\s+{role_word}', sentence_lower):
                return role_type
            if re.search(rf'the\s+{role_word}\s+{agent}', sentence_lower):
                return role_type
            if re.search(rf'{role_word}\s+{agent}', sentence_lower):
                return role_type
        
        return None
    
    def _calculate_quality(self, frame: EnrichedFrame) -> float:
        """Calculate a quality score for the frame."""
        # Immediate disqualifiers
        if frame.agent and frame.patient and frame.agent == frame.patient:
            return 0.0  # Self-reference is noise
        
        score = 0.0
        
        # Has agent
        if frame.agent:
            score += 0.2
            # Agent is known entity
            if frame.agent in self.known_entities:
                score += 0.2
        
        # Has meaningful action
        if frame.action and frame.action not in ['EXIST', 'POSSESS']:
            score += 0.2
        
        # Has patient (different from agent)
        if frame.patient and frame.patient != frame.agent:
            score += 0.2
            # Patient is known entity
            if frame.patient in self.known_entities:
                score += 0.1
        
        # Has relationship type
        if frame.relationship_type:
            score += 0.1
        
        # Has qualities
        if frame.qualities:
            score += 0.1
        
        # Has role
        if frame.role:
            score += 0.1
        
        return min(score, 1.0)
    
    def extract_all(self, text: str, min_quality: float = 0.0) -> List[EnrichedFrame]:
        """
        Extract frames from all sentences in text.
        
        Args:
            text: Input text
            min_quality: Minimum quality score to include
            
        Returns:
            List of enriched frames
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        frames = []
        for sentence in sentences:
            frame = self.extract(sentence)
            if frame and frame.quality_score >= min_quality:
                frames.append(frame)
        
        return frames


def reprocess_corpus(frames: List[Dict], known_entities: Set[str] = None) -> List[Dict]:
    """
    Reprocess existing frames with improved extraction.
    
    Takes the raw text from existing frames and re-extracts with the improved extractor.
    """
    extractor = ImprovedExtractor(known_entities=known_entities)
    
    improved_frames = []
    for f in frames:
        text = f.get('text', '')
        if not text:
            continue
        
        # Re-extract
        new_frame = extractor.extract(text)
        if new_frame and new_frame.quality_score > 0.3:
            # Preserve source
            frame_dict = new_frame.to_dict()
            frame_dict['source'] = f.get('source', '')
            frame_dict['text'] = text
            improved_frames.append(frame_dict)
    
    return improved_frames
