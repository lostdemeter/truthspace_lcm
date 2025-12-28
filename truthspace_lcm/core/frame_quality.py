#!/usr/bin/env python3
"""
Frame Quality Filter for GeometricLCM

Filters and validates concept frames to improve knowledge base quality.

Key Quality Issues Addressed:
1. Empty frames (no agent AND no patient)
2. Self-reference frames (agent == patient)
3. Noise agents (verbs, common words, places as agents)
4. Missing actions (frames with no meaningful action)
5. Gutenberg/metadata pollution

Quality Tiers:
- GOLD: Character interaction with meaningful action
- SILVER: Character mention with some context
- BRONZE: Any entity mention
- NOISE: Should be filtered out
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
from enum import Enum


class QualityTier(Enum):
    """Quality tier for frames."""
    GOLD = "gold"       # Best quality - character interactions
    SILVER = "silver"   # Good quality - character mentions
    BRONZE = "bronze"   # Acceptable - entity mentions
    NOISE = "noise"     # Should be filtered


@dataclass
class QualityReport:
    """Report on frame quality."""
    total_frames: int = 0
    gold_frames: int = 0
    silver_frames: int = 0
    bronze_frames: int = 0
    noise_frames: int = 0
    
    # Breakdown of noise types
    empty_frames: int = 0
    self_ref_frames: int = 0
    noise_agent_frames: int = 0
    metadata_frames: int = 0
    
    def __str__(self):
        return f"""Frame Quality Report:
  Total: {self.total_frames}
  Gold: {self.gold_frames} ({100*self.gold_frames/max(1,self.total_frames):.1f}%)
  Silver: {self.silver_frames} ({100*self.silver_frames/max(1,self.total_frames):.1f}%)
  Bronze: {self.bronze_frames} ({100*self.bronze_frames/max(1,self.total_frames):.1f}%)
  Noise: {self.noise_frames} ({100*self.noise_frames/max(1,self.total_frames):.1f}%)
    - Empty: {self.empty_frames}
    - Self-ref: {self.self_ref_frames}
    - Noise agent: {self.noise_agent_frames}
    - Metadata: {self.metadata_frames}"""


# Words that should NEVER be agents (verbs, function words)
NOISE_AGENTS = {
    # Question words
    'who', 'what', 'how', 'why', 'when', 'where', 'which',
    
    # Common verbs that get extracted as agents
    'come', 'go', 'get', 'make', 'take', 'give', 'see', 'know', 'think',
    'say', 'said', 'tell', 'told', 'ask', 'asked', 'look', 'looked',
    'let', 'put', 'set', 'run', 'find', 'found', 'keep', 'kept',
    'begin', 'began', 'start', 'end', 'turn', 'turned',
    'seem', 'seemed', 'appear', 'appeared', 'become', 'became',
    'feel', 'felt', 'hear', 'heard', 'leave', 'left',
    
    # Gutenberg/metadata
    'start', 'project', 'chapter', 'illustration', 'gutenberg', 'ebook',
    'volume', 'page', 'part', 'contents', 'preface', 'introduction',
    'footnote', 'note', 'appendix', 'index', 'table',
    
    # Publishing
    'publisher', 'press', 'edition', 'copyright', 'printed', 'published',
    
    # Generic words
    'man', 'woman', 'boy', 'girl', 'child', 'person', 'people',
    'thing', 'something', 'nothing', 'everything', 'anything',
    'one', 'two', 'three', 'first', 'second', 'last',
    'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week',
    'way', 'place', 'room', 'house', 'door', 'hand', 'head', 'face', 'eye', 'eyes',
    
    # Adjectives that get capitalized
    'great', 'good', 'little', 'old', 'young', 'new', 'long', 'short',
    'high', 'low', 'large', 'small', 'whole', 'half', 'own',
    
    # Adverbs/conjunctions
    'however', 'therefore', 'moreover', 'meanwhile', 'indeed', 'certainly',
    'perhaps', 'probably', 'suddenly', 'immediately', 'finally',
    'still', 'yet', 'already', 'always', 'never', 'often', 'sometimes',
    'once', 'twice', 'again', 'away', 'back', 'down', 'up', 'out',
    
    # Exclamations
    'oh', 'ah', 'alas', 'yes', 'no',
}

# Places that shouldn't be primary agents (they can be locations)
PLACE_WORDS = {
    'london', 'england', 'paris', 'france', 'america', 'europe',
    'new', 'york', 'street', 'road', 'park', 'hall', 'house', 'castle',
    'church', 'inn', 'hotel', 'station', 'court', 'square',
    'north', 'south', 'east', 'west',
}

# Actions that indicate meaningful interaction
MEANINGFUL_ACTIONS = {
    'SPEAK', 'PERCEIVE', 'MOVE', 'THINK', 'ACT', 'FEEL', 'CAUSE', 'BECOME',
}

# Actions that are often noise
WEAK_ACTIONS = {
    'EXIST', 'POSSESS', None, '', 'None',
}


class FrameQualityFilter:
    """
    Filter and validate concept frames for quality.
    
    Usage:
        filter = FrameQualityFilter()
        quality_frames = filter.filter_frames(frames)
        report = filter.analyze(frames)
    """
    
    def __init__(self, 
                 known_characters: Set[str] = None,
                 min_agent_frequency: int = 3,
                 strict_mode: bool = False):
        """
        Initialize the filter.
        
        Args:
            known_characters: Set of known character names (for GOLD tier)
            min_agent_frequency: Minimum times an agent must appear to be valid
            strict_mode: If True, only return GOLD and SILVER frames
        """
        self.known_characters = known_characters or set()
        self.min_agent_frequency = min_agent_frequency
        self.strict_mode = strict_mode
        
        # Build frequency counts during analysis
        self.agent_counts: Counter = Counter()
        self._analyzed = False
    
    def add_known_characters(self, characters: Set[str]):
        """Add known character names."""
        self.known_characters.update(c.lower() for c in characters)
    
    def analyze(self, frames: List[Dict]) -> QualityReport:
        """
        Analyze frame quality and return a report.
        
        Also builds agent frequency counts for filtering.
        """
        report = QualityReport(total_frames=len(frames))
        
        # Count agent frequencies
        self.agent_counts = Counter()
        for f in frames:
            agent = f.get('agent', '').lower()
            if agent:
                self.agent_counts[agent] += 1
        
        # Classify each frame
        for f in frames:
            tier = self._classify_frame(f)
            
            if tier == QualityTier.GOLD:
                report.gold_frames += 1
            elif tier == QualityTier.SILVER:
                report.silver_frames += 1
            elif tier == QualityTier.BRONZE:
                report.bronze_frames += 1
            else:
                report.noise_frames += 1
                
                # Categorize noise type
                agent = f.get('agent', '').lower()
                patient = f.get('patient', '').lower()
                
                if not agent and not patient:
                    report.empty_frames += 1
                elif agent and agent == patient:
                    report.self_ref_frames += 1
                elif agent in NOISE_AGENTS:
                    report.noise_agent_frames += 1
                elif self._is_metadata(f):
                    report.metadata_frames += 1
        
        self._analyzed = True
        return report
    
    def filter_frames(self, frames: List[Dict], 
                      min_tier: QualityTier = QualityTier.BRONZE) -> List[Dict]:
        """
        Filter frames to only include those at or above min_tier.
        
        Args:
            frames: List of frame dictionaries
            min_tier: Minimum quality tier to include
            
        Returns:
            Filtered list of frames
        """
        # Analyze first if not done
        if not self._analyzed:
            self.analyze(frames)
        
        tier_order = [QualityTier.GOLD, QualityTier.SILVER, QualityTier.BRONZE, QualityTier.NOISE]
        min_idx = tier_order.index(min_tier)
        
        filtered = []
        for f in frames:
            tier = self._classify_frame(f)
            tier_idx = tier_order.index(tier)
            if tier_idx <= min_idx:
                filtered.append(f)
        
        return filtered
    
    def _classify_frame(self, frame: Dict) -> QualityTier:
        """Classify a single frame into a quality tier."""
        agent = frame.get('agent', '').lower()
        patient = frame.get('patient', '').lower()
        action = frame.get('action', '')
        text = frame.get('text', '')
        
        # NOISE: Empty frame
        if not agent and not patient:
            return QualityTier.NOISE
        
        # NOISE: Self-reference
        if agent and agent == patient:
            return QualityTier.NOISE
        
        # NOISE: Agent is a noise word
        if agent in NOISE_AGENTS:
            return QualityTier.NOISE
        
        # NOISE: Metadata/Gutenberg content
        if self._is_metadata(frame):
            return QualityTier.NOISE
        
        # NOISE: Agent appears only once (likely extraction error)
        if agent and self.agent_counts.get(agent, 0) < self.min_agent_frequency:
            return QualityTier.NOISE
        
        # GOLD: Known character with meaningful action
        if agent in self.known_characters and action in MEANINGFUL_ACTIONS:
            return QualityTier.GOLD
        
        # GOLD: Two different known characters interacting
        if agent in self.known_characters and patient in self.known_characters:
            return QualityTier.GOLD
        
        # SILVER: Known character with any action
        if agent in self.known_characters or patient in self.known_characters:
            return QualityTier.SILVER
        
        # SILVER: Meaningful action with frequent agent
        if action in MEANINGFUL_ACTIONS and self.agent_counts.get(agent, 0) >= 5:
            return QualityTier.SILVER
        
        # BRONZE: Any valid agent with some content
        if agent and agent not in PLACE_WORDS:
            return QualityTier.BRONZE
        
        return QualityTier.NOISE
    
    def _is_metadata(self, frame: Dict) -> bool:
        """Check if frame is from metadata/Gutenberg content."""
        text = frame.get('text', '').lower()
        
        # Gutenberg markers
        if 'gutenberg' in text or 'project' in text and 'ebook' in text:
            return True
        
        # Table of contents
        if text.count('...') > 2 or text.count('. . .') > 2:
            return True
        
        # Page numbers / chapter headers
        if re.match(r'^\s*\d+\s*$', text) or re.match(r'^\s*chapter\s+\w+', text, re.I):
            return True
        
        # Very short text with numbers (likely TOC)
        if len(text) < 50 and re.search(r'\d{2,}', text):
            return True
        
        return False
    
    def get_quality_entities(self, frames: List[Dict], min_count: int = 5) -> Set[str]:
        """
        Get entities that appear frequently enough to be considered quality.
        
        These are likely real characters/entities, not extraction noise.
        """
        if not self._analyzed:
            self.analyze(frames)
        
        quality = set()
        for agent, count in self.agent_counts.items():
            if count >= min_count and agent not in NOISE_AGENTS and agent not in PLACE_WORDS:
                quality.add(agent)
        
        return quality
    
    def suggest_characters(self, frames: List[Dict], top_k: int = 50) -> List[Tuple[str, int]]:
        """
        Suggest likely character names based on frequency and patterns.
        
        Returns list of (name, count) tuples.
        """
        if not self._analyzed:
            self.analyze(frames)
        
        candidates = []
        for agent, count in self.agent_counts.most_common(top_k * 2):
            # Skip noise
            if agent in NOISE_AGENTS or agent in PLACE_WORDS:
                continue
            
            # Skip very short names (likely abbreviations)
            if len(agent) < 3:
                continue
            
            # Skip if appears less than 5 times
            if count < 5:
                continue
            
            candidates.append((agent, count))
        
        return candidates[:top_k]


def clean_corpus(frames: List[Dict], 
                 known_characters: Set[str] = None,
                 min_tier: QualityTier = QualityTier.BRONZE) -> Tuple[List[Dict], QualityReport]:
    """
    Clean a corpus of frames, removing noise.
    
    Args:
        frames: List of frame dictionaries
        known_characters: Optional set of known character names
        min_tier: Minimum quality tier to keep
        
    Returns:
        Tuple of (cleaned_frames, quality_report)
    """
    filter = FrameQualityFilter(known_characters=known_characters)
    report = filter.analyze(frames)
    cleaned = filter.filter_frames(frames, min_tier=min_tier)
    
    return cleaned, report


def extract_character_set(frames: List[Dict], sources: List[str] = None) -> Dict[str, Set[str]]:
    """
    Extract likely character names from frames, grouped by source.
    
    Args:
        frames: List of frame dictionaries
        sources: Optional list of sources to analyze (None = all)
        
    Returns:
        Dict mapping source name to set of character names
    """
    # Group frames by source
    by_source: Dict[str, List[Dict]] = {}
    for f in frames:
        source = f.get('source', 'unknown')
        if sources and source not in sources:
            continue
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(f)
    
    # Extract characters per source
    characters: Dict[str, Set[str]] = {}
    for source, source_frames in by_source.items():
        filter = FrameQualityFilter()
        suggestions = filter.suggest_characters(source_frames, top_k=30)
        characters[source] = {name for name, count in suggestions}
    
    return characters


# Pre-defined character sets for known sources
SHERLOCK_CHARACTERS = {
    'holmes', 'sherlock', 'watson', 'moriarty', 'lestrade', 'irene',
    'mycroft', 'hudson', 'adler', 'moran', 'stapleton', 'baskerville',
    'henry', 'mortimer', 'barrymore', 'selden', 'mcmurdo', 'douglas',
    'porlock', 'barker', 'ames', 'mason', 'white', 'baldwin',
}

PRIDE_PREJUDICE_CHARACTERS = {
    'elizabeth', 'darcy', 'bennet', 'bingley', 'jane', 'wickham',
    'collins', 'lydia', 'catherine', 'mary', 'kitty', 'georgiana',
    'fitzwilliam', 'lucas', 'charlotte', 'gardiner', 'phillips',
}

ALICE_CHARACTERS = {
    'alice', 'queen', 'king', 'hatter', 'rabbit', 'cat', 'duchess',
    'dormouse', 'gryphon', 'turtle', 'caterpillar', 'dodo',
}

MOBY_DICK_CHARACTERS = {
    'ahab', 'ishmael', 'queequeg', 'starbuck', 'stubb', 'flask',
    'tashtego', 'daggoo', 'fedallah', 'pip', 'perth', 'elijah',
    'bildad', 'peleg',
}

# Combined set for all known sources
ALL_KNOWN_CHARACTERS = (
    SHERLOCK_CHARACTERS | 
    PRIDE_PREJUDICE_CHARACTERS | 
    ALICE_CHARACTERS | 
    MOBY_DICK_CHARACTERS
)
