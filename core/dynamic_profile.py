"""
Dynamic Profile Builder for GeometricLCM

Builds character/entity profiles on-the-fly from the knowledge base,
replacing hand-coded profiles with dynamically extracted information.

Key insight: The corpus already contains everything we need:
- Frames with agent/patient/action tell us WHO does WHAT to WHOM
- Source tells us WHERE the entity appears
- Frequency and co-occurrence tell us relationships
- Action primitives tell us character roles

This is the scalable approach - any entity in the corpus can be described.
"""

import re
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class DynamicProfile:
    """
    Dynamically built profile for an entity.
    
    All fields are extracted from the knowledge base, not hand-coded.
    """
    entity: str
    canonical_name: str = ""
    role: str = "character"
    source: str = "the story"
    sources: List[str] = field(default_factory=list)
    
    # Extracted from frames
    actions: List[str] = field(default_factory=list)  # What they DO
    qualities: List[str] = field(default_factory=list)  # How they're DESCRIBED
    relationships: Dict[str, str] = field(default_factory=dict)  # WHO they interact with
    
    # Statistics
    frame_count: int = 0
    agent_count: int = 0  # Times as agent (active)
    patient_count: int = 0  # Times as patient (passive)
    
    # Sample text for context
    sample_texts: List[str] = field(default_factory=list)


# Action primitives to natural language descriptions
ACTION_TO_DESCRIPTION = {
    'SPEAK': 'speaks throughout the narrative',
    'THINK': 'is characterized by deep thought and reflection',
    'MOVE': 'travels and moves through the story',
    'PERCEIVE': 'observes and notices details',
    'FEEL': 'experiences strong emotions',
    'ACT': 'takes decisive action',
    'POSSESS': 'holds significant influence',
    'EXIST': 'appears throughout',
}

# Action primitives to role inference (single action)
ACTION_TO_ROLE = {
    'SPEAK': ['speaker', 'conversationalist'],
    'THINK': ['thinker', 'intellectual', 'philosopher'],
    'PERCEIVE': ['observer', 'detective', 'witness'],
    'ACT': ['protagonist', 'active character'],
    'FEEL': ['emotional character', 'romantic figure'],
    'MOVE': ['traveler', 'adventurer'],
    'POSSESS': ['person of influence', 'wealthy individual'],
}

# Action PROFILE signatures - combinations that indicate specific roles
# Format: {role: {action: (min_pct, max_pct), ...}}
# A character matches if their action distribution falls within these ranges
ACTION_PROFILE_SIGNATURES = {
    'investigator': {
        # High PERCEIVE + THINK, moderate SPEAK
        'PERCEIVE': (8, 100),   # At least 8% observing
        'THINK': (5, 100),      # At least 5% thinking
        'SPEAK': (15, 40),      # Moderate speaking
    },
    'narrator': {
        # Very high SPEAK, high PERCEIVE/THINK
        'SPEAK': (30, 100),     # Dominant speaking
        'PERCEIVE': (10, 100),  # High observation
    },
    'adventurer': {
        # High MOVE + ACT
        'MOVE': (15, 100),      # Lots of movement
        'ACT': (10, 100),       # Lots of action
    },
    'romantic figure': {
        # High EXIST (described), moderate FEEL
        'EXIST': (30, 100),     # Heavily described
    },
    'curious observer': {
        # Very high PERCEIVE + THINK
        'PERCEIVE': (20, 100),  # Very high observation
        'THINK': (15, 100),     # Very high thinking
    },
    'protagonist': {
        # Balanced high activity across multiple actions
        'SPEAK': (15, 35),
        'ACT': (10, 100),
        'MOVE': (10, 100),
    },
}

def infer_role_from_action_profile(action_counts: dict, total: int) -> Optional[str]:
    """
    Infer character role from their action distribution.
    
    This is the key insight: literature SHOWS characters doing things
    rather than TELLING us their role. The action profile IS the role.
    
    Args:
        action_counts: Dict mapping action primitives to counts
        total: Total number of actions
        
    Returns:
        Inferred role string, or None if no match
    """
    if total < 10:
        return None
    
    # Calculate percentages
    pcts = {action: 100 * count / total for action, count in action_counts.items()}
    
    # Check each signature
    best_match = None
    best_score = 0
    
    for role, signature in ACTION_PROFILE_SIGNATURES.items():
        matches = 0
        total_criteria = len(signature)
        
        for action, (min_pct, max_pct) in signature.items():
            actual_pct = pcts.get(action, 0)
            if min_pct <= actual_pct <= max_pct:
                matches += 1
        
        # Score = percentage of criteria matched
        score = matches / total_criteria if total_criteria > 0 else 0
        
        if score > best_score and score >= 0.5:  # At least 50% match
            best_score = score
            best_match = role
    
    return best_match

# Quality words to look for in text
QUALITY_WORDS = {
    'brilliant', 'clever', 'intelligent', 'wise', 'cunning', 'shrewd',
    'observant', 'perceptive', 'analytical', 'logical', 'rational',
    'kind', 'gentle', 'compassionate', 'loving', 'caring', 'warm',
    'cold', 'cruel', 'harsh', 'stern', 'strict', 'severe',
    'proud', 'humble', 'arrogant', 'modest', 'shy', 'bold', 'brave',
    'loyal', 'faithful', 'treacherous', 'deceitful',
    'strong', 'weak', 'handsome', 'beautiful',
    'witty', 'charming', 'mysterious', 'eccentric', 'peculiar',
}

# Role words that indicate character type
ROLE_INDICATORS = {
    'detective': ['detective', 'investigator', 'sleuth', 'consulting'],
    'doctor': ['doctor', 'physician', 'surgeon', 'medical'],
    'gentleman': ['gentleman', 'mr.', 'sir', 'lord'],
    'lady': ['lady', 'miss', 'mrs.', 'madam'],
    'villain': ['villain', 'criminal', 'evil', 'wicked'],
    'servant': ['servant', 'butler', 'maid', 'housekeeper'],
    'soldier': ['soldier', 'captain', 'colonel', 'military'],
    'professor': ['professor', 'teacher', 'scholar'],
}

# NOTE: No hard-coded character data - all profiles must be derived from ingested data
# This ensures the system is generalizable and scalable to any literary work

# Common noise words to filter
NOISE_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'this', 'that', 'these', 'those', 'it', 'its',
    'and', 'or', 'but', 'if', 'then', 'else', 'so', 'as',
    'with', 'from', 'to', 'for', 'of', 'in', 'on', 'at', 'by',
    'chapter', 'part', 'book', 'volume', 'page', 'illustration',
    'start', 'end', 'project', 'gutenberg', 'ebook',
}


class DynamicProfileBuilder:
    """
    Builds entity profiles dynamically from the knowledge base.
    
    This replaces hand-coded profiles with on-the-fly extraction.
    """
    
    def __init__(self, knowledge=None):
        """
        Initialize with a ConceptKnowledge instance.
        
        Args:
            knowledge: ConceptKnowledge instance with loaded corpus
        """
        self.knowledge = knowledge
        self._profile_cache: Dict[str, DynamicProfile] = {}
    
    def set_knowledge(self, knowledge):
        """Set or update the knowledge base."""
        self.knowledge = knowledge
        self._profile_cache.clear()
    
    def build_profile(self, entity: str, max_frames: int = 100) -> DynamicProfile:
        """
        Build a profile for an entity from the knowledge base.
        
        Args:
            entity: Entity name to profile
            max_frames: Maximum frames to analyze
            
        Returns:
            DynamicProfile with extracted information
        """
        # Check cache first
        entity_lower = entity.lower()
        if entity_lower in self._profile_cache:
            return self._profile_cache[entity_lower]
        
        if not self.knowledge:
            return DynamicProfile(entity=entity, canonical_name=entity.title())
        
        profile = DynamicProfile(
            entity=entity_lower,
            canonical_name=self._infer_canonical_name(entity_lower),
        )
        
        # Get frames involving this entity
        frames = self._get_entity_frames(entity_lower, max_frames)
        profile.frame_count = len(frames)
        
        if not frames:
            return profile
        
        # Extract information from frames
        self._extract_actions(profile, frames)
        self._extract_sources(profile, frames)
        self._extract_relationships(profile, frames, entity_lower)
        self._extract_qualities(profile, frames)
        self._extract_sample_texts(profile, frames)
        self._infer_role(profile)
        
        # Cache the profile
        self._profile_cache[entity_lower] = profile
        
        return profile
    
    def _get_entity_frames(self, entity: str, max_frames: int) -> List[Dict]:
        """Get frames where entity appears as agent or patient."""
        frames = []
        
        for frame in self.knowledge.frames:
            agent = frame.get('agent', '').lower()
            patient = frame.get('patient', '').lower()
            
            if agent == entity or patient == entity:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
        
        return frames
    
    def _infer_canonical_name(self, entity: str) -> str:
        """Infer the canonical (display) name for an entity."""
        # Check if entity appears with title in text
        if not self.knowledge:
            return entity.title()
        
        # Look for patterns like "Mr. Holmes", "Dr. Watson", etc.
        title_patterns = [
            (r'\b(mr\.?\s+' + entity + r')\b', 'Mr. '),
            (r'\b(mrs\.?\s+' + entity + r')\b', 'Mrs. '),
            (r'\b(miss\s+' + entity + r')\b', 'Miss '),
            (r'\b(dr\.?\s+' + entity + r')\b', 'Dr. '),
            (r'\b(professor\s+' + entity + r')\b', 'Professor '),
            (r'\b(inspector\s+' + entity + r')\b', 'Inspector '),
            (r'\b(captain\s+' + entity + r')\b', 'Captain '),
            (r'\b(colonel\s+' + entity + r')\b', 'Colonel '),
        ]
        
        # Sample some frames to find title usage
        for frame in self.knowledge.frames[:500]:
            text = frame.get('text', '').lower()
            for pattern, title in title_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return title + entity.title()
        
        return entity.title()
    
    def _extract_actions(self, profile: DynamicProfile, frames: List[Dict]):
        """Extract action patterns from frames."""
        action_counts = Counter()
        
        for frame in frames:
            agent = frame.get('agent', '').lower()
            action = frame.get('action', '')
            
            if agent == profile.entity and action:
                action_counts[action] += 1
                profile.agent_count += 1
            elif frame.get('patient', '').lower() == profile.entity:
                profile.patient_count += 1
        
        # Store primitives for role inference
        self._action_primitives = action_counts
        
        # Get top actions (skip EXIST and POSSESS as they're not descriptive)
        skip_actions = {'EXIST', 'POSSESS', 'None', ''}
        for action, count in action_counts.most_common(5):
            if action and action not in skip_actions:
                desc = ACTION_TO_DESCRIPTION.get(action, f'engages in {action.lower()}')
                if desc not in profile.actions:
                    profile.actions.append(desc)
    
    def _extract_sources(self, profile: DynamicProfile, frames: List[Dict]):
        """Extract source information."""
        sources = set()
        for frame in frames:
            source = frame.get('source', '')
            if source and source not in ['Q&A Training']:
                sources.add(source)
        
        profile.sources = list(sources)
        if sources:
            # Pick the most common source as primary
            source_counts = Counter(f.get('source', '') for f in frames)
            profile.source = source_counts.most_common(1)[0][0]
    
    def _extract_relationships(self, profile: DynamicProfile, frames: List[Dict], entity: str):
        """Extract relationships with other entities."""
        # Count co-occurrences with other entities
        related_counts = Counter()
        
        for frame in frames:
            agent = frame.get('agent', '').lower()
            patient = frame.get('patient', '').lower()
            
            if agent == entity and patient and patient not in NOISE_WORDS:
                # Skip self-references
                if patient != entity:
                    related_counts[patient] += 1
            elif patient == entity and agent and agent not in NOISE_WORDS:
                if agent != entity:
                    related_counts[agent] += 1
        
        # Get quality entities from knowledge base if available
        quality_entities = getattr(self.knowledge, 'quality_entities', set())
        
        # Build relationship descriptions for top related entities
        added = 0
        for related, count in related_counts.most_common(10):
            if related == entity or len(related) < 3:
                continue
            
            # Only include quality entities (filters noise) - require at least 3 co-occurrences
            if quality_entities and related not in quality_entities:
                continue
            if count < 3:
                continue
            
            # Determine relationship type based on frame patterns
            rel_type = self._infer_relationship_type(entity, related, frames)
            profile.relationships[related] = rel_type
            added += 1
            if added >= 3:  # Limit to top 3 relationships
                break
    
    def _infer_relationship_type(self, entity: str, related: str, frames: List[Dict]) -> str:
        """Infer the type of relationship between two entities."""
        # Count how they interact
        entity_acts_on_related = 0
        related_acts_on_entity = 0
        
        for frame in frames:
            agent = frame.get('agent', '').lower()
            patient = frame.get('patient', '').lower()
            
            if agent == entity and patient == related:
                entity_acts_on_related += 1
            elif agent == related and patient == entity:
                related_acts_on_entity += 1
        
        # Bidirectional = partnership/friendship
        if entity_acts_on_related > 0 and related_acts_on_entity > 0:
            return f"{related.title()}, a close associate"
        elif entity_acts_on_related > related_acts_on_entity:
            return f"{related.title()}"
        else:
            return f"{related.title()}"
    
    def _extract_qualities(self, profile: DynamicProfile, frames: List[Dict]):
        """Extract quality/adjective descriptions from frame text."""
        # All qualities must be derived from ingested data
        quality_counts = Counter()
        
        entity_pattern = re.compile(r'\b' + re.escape(profile.entity) + r'\b', re.IGNORECASE)
        
        for frame in frames:
            text = frame.get('text', '').lower()
            
            # Only look at frames that mention the entity
            if not entity_pattern.search(text):
                continue
            
            # Find quality words near the entity mention
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if word in QUALITY_WORDS:
                    quality_counts[word] += 1
        
        # Get top qualities
        for quality, count in quality_counts.most_common(4):
            if quality not in profile.qualities:
                profile.qualities.append(quality)
    
    def _extract_sample_texts(self, profile: DynamicProfile, frames: List[Dict]):
        """Extract sample text snippets for context."""
        for frame in frames[:10]:
            text = frame.get('text', '').strip()
            if text and len(text) > 20 and len(text) < 200:
                # Clean up the text
                text = re.sub(r'\s+', ' ', text)
                if text not in profile.sample_texts:
                    profile.sample_texts.append(text)
                if len(profile.sample_texts) >= 3:
                    break
    
    def _infer_role(self, profile: DynamicProfile):
        """Infer the character's role from actions and text."""
        # FIRST: Try action profile inference (most reliable for literature)
        # Literature SHOWS characters doing things rather than labeling them
        if hasattr(self, '_action_primitives') and self._action_primitives:
            action_counts = dict(self._action_primitives)
            total = sum(action_counts.values())
            profile_role = infer_role_from_action_profile(action_counts, total)
            if profile_role:
                profile.role = profile_role
                return
        
        # SECOND: Check for explicit role indicators in sample texts (rare but definitive)
        all_text = ' '.join(profile.sample_texts).lower()
        for role, indicators in ROLE_INDICATORS.items():
            for indicator in indicators:
                if indicator in all_text:
                    profile.role = role
                    return
        
        # THIRD: Infer from dominant single action primitive
        if hasattr(self, '_action_primitives') and self._action_primitives:
            for primitive, count in self._action_primitives.most_common(3):
                if primitive in ACTION_TO_ROLE:
                    roles = ACTION_TO_ROLE[primitive]
                    profile.role = roles[0]
                    return
        
        # Default based on agent/patient ratio
        if profile.agent_count > profile.patient_count * 2:
            profile.role = "notable character"
        elif profile.patient_count > profile.agent_count * 2:
            profile.role = "supporting character"
        else:
            profile.role = "character"
    
    def generate_response(self, entity: str, depth: float = 0.0) -> Optional[str]:
        """
        Generate a natural response about an entity.
        
        Args:
            entity: Entity to describe
            depth: -1 (terse) to +1 (elaborate)
            
        Returns:
            Natural language description, or None if entity unknown
        """
        profile = self.build_profile(entity)
        
        if profile.frame_count == 0:
            return None
        
        # Determine sentence count based on depth
        if depth < -0.3:
            max_sentences = 2
        elif depth > 0.3:
            max_sentences = 5
        else:
            max_sentences = 3
        
        sentences = []
        
        # 1. Opening sentence
        opening = self._generate_opening(profile)
        sentences.append(opening)
        
        # 2. Qualities sentence
        if len(sentences) < max_sentences and profile.qualities:
            qualities_sent = self._generate_qualities_sentence(profile)
            if qualities_sent:
                sentences.append(qualities_sent)
        
        # 3. Actions sentence
        if len(sentences) < max_sentences and profile.actions:
            action_sent = self._generate_action_sentence(profile)
            if action_sent:
                sentences.append(action_sent)
        
        # 4. Relationship sentence
        if len(sentences) < max_sentences and profile.relationships:
            rel_sent = self._generate_relationship_sentence(profile)
            if rel_sent:
                sentences.append(rel_sent)
        
        # 5. Additional context (for elaborate mode)
        if len(sentences) < max_sentences and profile.sample_texts and depth > 0.3:
            context_sent = self._generate_context_sentence(profile)
            if context_sent:
                sentences.append(context_sent)
        
        return ' '.join(sentences)
    
    def _generate_opening(self, profile: DynamicProfile) -> str:
        """Generate opening sentence."""
        name = profile.canonical_name
        role = profile.role
        source = profile.source
        
        # Vary the opening structure
        templates = [
            f"{name} is a {role} from {source}.",
            f"In {source}, {name} appears as a {role}.",
            f"{name}, a {role} in {source}, is a notable character.",
        ]
        
        import random
        return random.choice(templates)
    
    def _generate_qualities_sentence(self, profile: DynamicProfile) -> str:
        """Generate sentence about qualities."""
        if not profile.qualities:
            return ""
        
        qualities = profile.qualities[:3]
        if len(qualities) == 1:
            qualities_str = qualities[0]
        elif len(qualities) == 2:
            qualities_str = f"{qualities[0]} and {qualities[1]}"
        else:
            qualities_str = f"{', '.join(qualities[:-1])}, and {qualities[-1]}"
        
        pronoun = self._get_pronoun(profile)
        return f"{pronoun.capitalize()} is described as {qualities_str}."
    
    def _generate_action_sentence(self, profile: DynamicProfile) -> str:
        """Generate sentence about actions."""
        if not profile.actions:
            return ""
        
        action = profile.actions[0]
        pronoun = self._get_pronoun(profile)
        return f"{pronoun.capitalize()} {action}."
    
    def _generate_relationship_sentence(self, profile: DynamicProfile) -> str:
        """Generate sentence about relationships."""
        if not profile.relationships:
            return ""
        
        # Get the most important relationship
        related, description = next(iter(profile.relationships.items()))
        pronoun = self._get_pronoun(profile)
        
        return f"{pronoun.capitalize()} is closely connected to {description}."
    
    def _generate_context_sentence(self, profile: DynamicProfile) -> str:
        """Generate additional context from sample text."""
        if not profile.sample_texts:
            return ""
        
        # Find a good sample that's not too long
        for text in profile.sample_texts:
            if 30 < len(text) < 150:
                return f'The text describes: "{text[:100]}..."' if len(text) > 100 else f'The text notes: "{text}"'
        
        return ""
    
    def _get_pronoun(self, profile: DynamicProfile) -> str:
        """Determine pronoun for entity."""
        # Infer pronoun from data - check for female indicators in role
        female_roles = ['lady', 'miss', 'mrs', 'woman', 'girl', 'daughter', 'sister', 'wife', 'mother']
        if any(ind in profile.role.lower() for ind in female_roles):
            return 'she'
        if any(ind in profile.canonical_name.lower() for ind in ['miss', 'mrs', 'lady']):
            return 'she'
        
        # Default to 'he' for traditional texts
        return 'he'


# Global instance
_builder: Optional[DynamicProfileBuilder] = None


def get_profile_builder(knowledge=None) -> DynamicProfileBuilder:
    """Get the dynamic profile builder."""
    global _builder
    if _builder is None:
        _builder = DynamicProfileBuilder(knowledge)
    elif knowledge and _builder.knowledge != knowledge:
        _builder.set_knowledge(knowledge)
    return _builder


def generate_dynamic_response(entity: str, knowledge=None, depth: float = 0.0) -> Optional[str]:
    """
    Convenience function to generate a response about an entity.
    
    Args:
        entity: Entity name
        knowledge: ConceptKnowledge instance
        depth: Response depth (-1 to +1)
        
    Returns:
        Natural language response, or None if entity not found
    """
    builder = get_profile_builder(knowledge)
    return builder.generate_response(entity, depth=depth)
