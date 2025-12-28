#!/usr/bin/env python3
"""
Holographic Template Projection: Dynamic Response Templates via Interference

Instead of hard-coded templates, this module projects templates dynamically
from stored Q&A pairs using holographic interference:

1. Store example Q&A pairs
2. For a new query, find similar Q&A pairs
3. Compute interference pattern on responses
4. Structure words (is, a, who) align → keep as literal
5. Content words (Watson, Darcy) have geometric phases → become slots
6. Fill slots with query-specific content

Mathematical Foundation (GEOMETRIC ENCODING):
- Each word encoded as: magnitude × e^(i·phase)
- Structure words: phase = 0 (always align)
- Content words: phase = φ-direction × π (from geometric knowledge)
- Magnitude = role_strength (from initiator/mediator/receiver counts)
- Interference: Σ encodings / N
- High magnitude → keep, Low magnitude → slot

Key Insight: ENCODE = DECODE (same operation, opposite directions)
The phase IS the semantic structure, not an arbitrary hash.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Use cmath for complex operations without numpy dependency
import cmath

PHI = 1.618034


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QAPair:
    """A question-answer pair for template learning."""
    question: str
    answer: str
    question_type: str  # WHO, WHAT, WHERE, etc.
    entity: str  # Main entity in question


@dataclass
class ProjectedTemplate:
    """A template projected from interference."""
    pattern: str  # e.g., "{entity} is a {role} who {action}"
    slots: List[str]  # e.g., ["entity", "role", "action"]
    confidence: float  # How strong the interference was
    source_count: int  # How many Q&A pairs contributed


# =============================================================================
# HOLOGRAPHIC TEMPLATE PROJECTOR
# =============================================================================

class HolographicTemplateProjector:
    """
    Project response templates via holographic interference.
    
    Key insight: Templates are patterns that emerge from similar responses.
    Structure words reinforce (same phase), content words have geometric phases.
    
    GEOMETRIC ENCODING:
    - Phase = φ-direction × π (initiator vs receiver)
    - Magnitude = role_strength (how strongly typed)
    - Structure words: phase = 0 (always align)
    - Content words: phase from geometric knowledge
    """
    
    # Structure words that should always align (phase = 0)
    STRUCTURE_WORDS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'a', 'an', 'the',
        'who', 'that', 'which', 'whom', 'whose',
        'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'as', 'from', 'into', 'through',
    }
    
    # Question type patterns
    QUESTION_PATTERNS = {
        'WHO': ['who is', 'who was', 'who are', 'who were', 'who did', 'who does'],
        'WHAT': ['what is', 'what was', 'what are', 'what did', 'what does'],
        'WHERE': ['where is', 'where was', 'where did', 'where does'],
        'WHEN': ['when did', 'when was', 'when is'],
        'WHY': ['why did', 'why does', 'why is', 'why was'],
        'HOW': ['how did', 'how does', 'how is', 'how was'],
    }
    
    def __init__(self, knowledge=None):
        """
        Initialize the projector.
        
        Args:
            knowledge: Optional GeometricKnowledge for entity info
        """
        self.knowledge = knowledge
        self.qa_pairs: List[QAPair] = []
        self.template_cache: Dict[str, ProjectedTemplate] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question (WHO, WHAT, etc.)."""
        q_lower = question.lower()
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in q_lower:
                    return q_type
        return 'WHAT'  # Default
    
    def _extract_entity(self, question: str) -> str:
        """Extract the main entity from a question."""
        words = self._tokenize(question)
        
        # Skip question words and structure words
        skip_words = {'who', 'what', 'where', 'when', 'why', 'how', 
                      'is', 'are', 'was', 'were', 'did', 'does', 'do',
                      'the', 'a', 'an'}
        
        for word in words:
            if word not in skip_words and len(word) > 2:
                return word
        
        return words[-1] if words else ''
    
    def add_qa_pair(self, question: str, answer: str):
        """
        Add a Q&A pair for template learning.
        
        Args:
            question: The question text
            answer: The answer text
        """
        q_type = self._detect_question_type(question)
        entity = self._extract_entity(question)
        
        pair = QAPair(
            question=question,
            answer=answer,
            question_type=q_type,
            entity=entity
        )
        self.qa_pairs.append(pair)
        
        # Invalidate cache for this question type
        if q_type in self.template_cache:
            del self.template_cache[q_type]
    
    def add_qa_pairs_from_corpus(self, qa_list: List[Tuple[str, str]]):
        """Add multiple Q&A pairs at once."""
        for question, answer in qa_list:
            self.add_qa_pair(question, answer)
    
    def _find_similar_pairs(self, question_type: str, k: int = 5) -> List[QAPair]:
        """Find k Q&A pairs with the same question type."""
        matching = [p for p in self.qa_pairs if p.question_type == question_type]
        return matching[:k]
    
    def _get_word_phase(self, word: str) -> float:
        """
        Get phase for a word using GEOMETRIC encoding.
        
        Structure words → 0 (align)
        Content words → φ-direction × π (from geometric knowledge)
        
        The phase encodes semantic role:
        - Initiators (subjects): phase near 0
        - Receivers (objects): phase near π
        - Mediators (verbs): phase near π/2
        """
        word_lower = word.lower()
        
        if word_lower in self.STRUCTURE_WORDS:
            return 0.0
        
        # Use geometric knowledge if available
        if self.knowledge and word_lower in self.knowledge.concepts:
            concept = self.knowledge.concepts[word_lower]
            
            # φ-direction: [-1, 1] → phase: [0, 2π]
            # Initiators (φ-dir > 0) → phase near 0
            # Receivers (φ-dir < 0) → phase near π
            phi_dir = concept.phi_direction
            
            # Also consider mediator ratio for verbs
            total_roles = concept.initiator_count + concept.mediator_count + concept.receiver_count
            if total_roles > 0:
                mediator_ratio = concept.mediator_count / total_roles
                if mediator_ratio > 0.5:
                    # Verbs get phase near π/2
                    return math.pi / 2
            
            # Map φ-direction to phase: [-1, 1] → [π, 0]
            # Initiators (positive) → small phase
            # Receivers (negative) → large phase
            return (1 - phi_dir) * math.pi / 2
        
        # Fallback: use mean position as phase proxy
        # Words at start of sentences (initiators) → small phase
        # Words at end (receivers) → large phase
        # This is a reasonable heuristic when we don't have knowledge
        return math.pi / 2  # Neutral phase for unknown words
    
    def _get_word_magnitude(self, word: str) -> float:
        """
        Get magnitude (importance) for a word using GEOMETRIC encoding.
        
        Structure words → lower (they're common)
        Content words → role_strength (how strongly typed)
        
        Magnitude encodes HOW MUCH of a concept:
        - High role counts → high magnitude (well-defined concept)
        - Low role counts → low magnitude (weakly defined)
        """
        word_lower = word.lower()
        
        if word_lower in self.STRUCTURE_WORDS:
            return 0.5
        
        # Use geometric knowledge if available
        if self.knowledge and word_lower in self.knowledge.concepts:
            concept = self.knowledge.concepts[word_lower]
            
            # Role strength: how many times this word has appeared in semantic roles
            total_roles = concept.initiator_count + concept.mediator_count + concept.receiver_count
            
            # Normalize: more roles = higher magnitude (up to 1.5)
            # This uses φ-weighting: importance grows with usage
            role_strength = min(1.5, 0.5 + total_roles * 0.1)
            
            return role_strength
        
        # Fallback for unknown words
        return 1.0
    
    def _encode_response(self, response: str) -> List[Tuple[str, complex]]:
        """
        Encode a response as a sequence of (word, complex) pairs.
        
        Each word is encoded as: magnitude × e^(i·phase)
        """
        words = self._tokenize(response)
        encoded = []
        
        for word in words:
            phase = self._get_word_phase(word)
            magnitude = self._get_word_magnitude(word)
            
            # Complex encoding: magnitude × e^(i·phase)
            z = magnitude * cmath.exp(1j * phase)
            encoded.append((word, z))
        
        return encoded
    
    def _compute_interference(self, responses: List[str]) -> List[Tuple[int, str, complex, int, bool]]:
        """
        Compute interference pattern from multiple responses.
        
        IMPROVED: Uses true multi-response synthesis instead of just first response.
        
        Key insight: Structure words should align regardless of exact position.
        Content words (even if repeated like 'Holmes') should become slots.
        
        Returns list of (position, word, complex_value, occurrence_count, is_structure)
        """
        n_responses = len(responses)
        
        # Count word occurrences across all responses
        word_counts: Dict[str, int] = defaultdict(int)
        word_positions: Dict[str, List[float]] = defaultdict(list)
        
        for response in responses:
            words = self._tokenize(response)
            n = len(words)
            seen_in_response = set()
            
            for i, word in enumerate(words):
                pos = i / max(n - 1, 1)
                word_positions[word].append(pos)
                if word not in seen_in_response:
                    word_counts[word] += 1
                    seen_in_response.add(word)
        
        # Build a MERGED template from all responses
        # Group words by position bucket and select best for each
        NUM_BUCKETS = 15
        bucket_candidates: Dict[int, List[Tuple[str, int, float]]] = defaultdict(list)
        
        for word, positions in word_positions.items():
            avg_pos = sum(positions) / len(positions)
            bucket = int(avg_pos * (NUM_BUCKETS - 1))
            count = word_counts[word]
            bucket_candidates[bucket].append((word, count, avg_pos))
        
        # For each bucket, select the best word
        result = []
        for bucket in sorted(bucket_candidates.keys()):
            candidates = bucket_candidates[bucket]
            
            # Prefer structure words, then by frequency
            structure_cands = [(w, c, p) for w, c, p in candidates if w in self.STRUCTURE_WORDS]
            content_cands = [(w, c, p) for w, c, p in candidates if w not in self.STRUCTURE_WORDS]
            
            # Add structure words first (they should appear)
            for word, count, pos in structure_cands:
                frequency = count / n_responses
                if frequency >= 0.5:  # Appears in 50%+ of responses
                    is_structure = True
                    phase = self._get_word_phase(word)
                    magnitude = self._get_word_magnitude(word) * 2.0
                    z = magnitude * cmath.exp(1j * phase)
                    result.append((bucket, word, z, count, is_structure))
            
            # Add content word slot (pick most common for position inference)
            if content_cands:
                # Sort by count descending
                content_cands.sort(key=lambda x: -x[1])
                word, count, pos = content_cands[0]
                is_structure = False
                phase = self._get_word_phase(word)
                magnitude = self._get_word_magnitude(word) * 0.2
                z = magnitude * cmath.exp(1j * phase)
                result.append((bucket, word, z, count, is_structure))
        
        # Sort by bucket position
        result.sort(key=lambda x: x[0])
        
        return result
    
    def _infer_slot_type(self, word: str, position: float) -> str:
        """
        Infer slot type from a cancelled word.
        
        Uses word properties and position to guess slot type.
        
        Typical WHO IS sentence structure:
        "Watson is a loyal doctor who assists Holmes."
         0.0    0.14 0.28 0.43  0.57 0.71 0.86   1.0
         entity      adj   role      action target
        """
        word_lower = word.lower()
        
        # Entity at start
        if position < 0.15:
            return "entity"
        
        # Word-ending heuristics (override position)
        if word_lower.endswith(('ed', 'ing', 'es', 'ies')):
            return "action"
        if word_lower.endswith('s') and position > 0.6:
            return "action"
        if word_lower.endswith(('ly',)):
            return "manner"
        if word_lower.endswith(('er', 'or', 'ist', 'ian', 'man')):
            return "role"
        
        # Check knowledge if available
        if self.knowledge and word_lower in self.knowledge.concepts:
            c = self.knowledge.concepts[word_lower]
            if c.initiator_count > c.receiver_count:
                return "entity"
            elif c.mediator_count > 0:
                return "action"
            else:
                return "target"
        
        # Position-based fallback for WHO IS pattern
        # [entity] is a [adjective] [role] who [action] [target]
        if position > 0.85:
            return "target"
        elif position > 0.65:
            return "action"
        elif position > 0.45:
            return "role"
        elif position > 0.25:
            return "adjective"
        
        return "entity"
    
    def _extract_template(self, interference: List[Tuple[int, str, complex, int, bool]], 
                          num_responses: int,
                          threshold: float = 0.35) -> ProjectedTemplate:
        """
        Extract template from interference pattern.
        
        Structure words → keep as literal
        Content words → become slots
        """
        parts = []
        slots = []
        last_slot_type = None
        
        max_pos = max(pos for pos, _, _, _, _ in interference) if interference else 1
        
        for pos, word, z, count, is_structure in interference:
            position = pos / max(max_pos, 1)
            
            if is_structure:
                # Keep structure words
                parts.append(word)
                last_slot_type = None
            else:
                # Content word → slot
                slot_type = self._infer_slot_type(word, position)
                
                # Avoid duplicate adjacent slots OF THE SAME TYPE
                if slot_type != last_slot_type:
                    parts.append(f"{{{slot_type}}}")
                    slots.append(slot_type)
                last_slot_type = slot_type
        
        # Clean up template
        pattern = " ".join(parts)
        
        # Calculate confidence based on structure word ratio
        kept_count = sum(1 for p in parts if not p.startswith('{'))
        confidence = kept_count / max(len(parts), 1)
        
        return ProjectedTemplate(
            pattern=pattern,
            slots=slots,
            confidence=confidence,
            source_count=num_responses
        )
    
    def project_template(self, query: str, k: int = 5) -> ProjectedTemplate:
        """
        Project a template for the given query.
        
        Args:
            query: The question to generate a template for
            k: Number of similar Q&A pairs to use
            
        Returns:
            ProjectedTemplate with pattern and slots
        """
        q_type = self._detect_question_type(query)
        
        # Check cache
        if q_type in self.template_cache:
            return self.template_cache[q_type]
        
        # Find similar Q&A pairs
        similar = self._find_similar_pairs(q_type, k)
        
        if not similar:
            # Fallback template
            return self._fallback_template(q_type)
        
        # Extract responses
        responses = [pair.answer for pair in similar]
        
        # Compute interference
        interference = self._compute_interference(responses)
        
        # Extract template
        template = self._extract_template(interference, len(responses))
        
        # Cache it
        self.template_cache[q_type] = template
        
        return template
    
    def _fallback_template(self, question_type: str) -> ProjectedTemplate:
        """Fallback templates when no Q&A pairs available."""
        fallbacks = {
            'WHO': ProjectedTemplate("{entity} is a {role} who {action}.", 
                                     ["entity", "role", "action"], 0.3, 0),
            'WHAT': ProjectedTemplate("{entity} {action} {target}.", 
                                      ["entity", "action", "target"], 0.3, 0),
            'WHERE': ProjectedTemplate("{entity} is located in {location}.", 
                                       ["entity", "location"], 0.3, 0),
            'WHEN': ProjectedTemplate("{entity} {action} in {time}.", 
                                      ["entity", "action", "time"], 0.3, 0),
            'WHY': ProjectedTemplate("{entity} {action} because {reason}.", 
                                     ["entity", "action", "reason"], 0.3, 0),
            'HOW': ProjectedTemplate("{entity} {action} by {method}.", 
                                     ["entity", "action", "method"], 0.3, 0),
        }
        return fallbacks.get(question_type, fallbacks['WHAT'])
    
    def fill_template(self, template: ProjectedTemplate, 
                      entity: str, 
                      slot_values: Dict[str, str] = None) -> str:
        """
        Fill a template with values.
        
        Args:
            template: The projected template
            entity: The main entity
            slot_values: Optional dict of slot → value
            
        Returns:
            Filled response string
        """
        result = template.pattern
        
        # Always fill entity
        result = result.replace("{entity}", entity)
        
        # Fill other slots
        if slot_values:
            for slot, value in slot_values.items():
                result = result.replace(f"{{{slot}}}", value)
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def generate(self, query: str, slot_values: Dict[str, str] = None) -> str:
        """
        Generate a response for the query.
        
        Args:
            query: The question
            slot_values: Optional slot values (if not provided, uses entity profile)
            
        Returns:
            Generated response
        """
        # Project template
        template = self.project_template(query)
        
        # Extract entity
        entity = self._extract_entity(query)
        
        # Get slot values from knowledge if not provided
        if slot_values is None and self.knowledge:
            slot_values = self._get_slot_values_from_knowledge(entity, template.slots)
        
        # Fill and return
        return self.fill_template(template, entity, slot_values or {})
    
    def _get_slot_values_from_knowledge(self, entity: str, slots: List[str]) -> Dict[str, str]:
        """Get slot values from knowledge base."""
        values = {}
        
        if not self.knowledge or entity not in self.knowledge.concepts:
            return values
        
        concept = self.knowledge.concepts[entity]
        
        for slot in slots:
            if slot == 'role':
                # Use most common action as role hint
                if concept.actions:
                    most_common = concept.actions.most_common(1)
                    if most_common:
                        values['role'] = most_common[0][0] + "er"  # Simple heuristic
            
            elif slot == 'action':
                if concept.actions:
                    most_common = concept.actions.most_common(1)
                    if most_common:
                        values['action'] = most_common[0][0] + "s"
            
            elif slot == 'target':
                if concept.targets:
                    most_common = concept.targets.most_common(1)
                    if most_common:
                        values['target'] = most_common[0][0]
        
        return values


# =============================================================================
# DEMO / TEST
# =============================================================================

# =============================================================================
# HOLOGRAPHIC RESPONSE SYNTHESIZER
# =============================================================================

class HolographicResponseSynthesizer:
    """
    Synthesize responses by combining multiple source texts via interference.
    
    Unlike template projection (which extracts structure), this synthesizes
    content by finding common elements across relevant source texts.
    
    Applications:
    1. Multi-source answer synthesis
    2. Paraphrase generation
    3. Summary generation
    """
    
    def __init__(self):
        self.sources: List[str] = []
    
    def add_source(self, text: str):
        """Add a source text."""
        self.sources.append(text)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def synthesize(self, query: str, sources: List[str] = None, 
                   top_k: int = 10) -> str:
        """
        Synthesize a response from multiple sources.
        
        Words that appear in multiple sources reinforce.
        Words unique to one source cancel.
        """
        if sources is None:
            sources = self.sources
        
        if not sources:
            return ""
        
        # Count word occurrences across sources
        word_counts: Dict[str, int] = defaultdict(int)
        word_positions: Dict[str, List[float]] = defaultdict(list)
        
        for source in sources:
            words = self._tokenize(source)
            seen = set()
            for i, word in enumerate(words):
                pos = i / max(len(words) - 1, 1)
                word_positions[word].append(pos)
                if word not in seen:
                    word_counts[word] += 1
                    seen.add(word)
        
        # Words appearing in multiple sources are "constructive"
        n_sources = len(sources)
        constructive = []
        
        for word, count in word_counts.items():
            frequency = count / n_sources
            if frequency >= 0.5:  # Appears in 50%+ of sources
                avg_pos = sum(word_positions[word]) / len(word_positions[word])
                constructive.append((word, frequency, avg_pos))
        
        # Sort by position to maintain sentence order
        constructive.sort(key=lambda x: x[2])
        
        # Build synthesized response
        words = [w for w, _, _ in constructive[:top_k]]
        
        return " ".join(words)
    
    def synthesize_with_structure(self, query: str, sources: List[str],
                                   structure_words: Set[str] = None) -> str:
        """
        Synthesize while preserving grammatical structure.
        
        Structure words are kept; content words are selected by interference.
        """
        if structure_words is None:
            structure_words = {
                'is', 'are', 'was', 'were', 'a', 'an', 'the',
                'who', 'that', 'which', 'and', 'or', 'but',
                'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
            }
        
        if not sources:
            return ""
        
        # Use first source as template
        template_words = self._tokenize(sources[0])
        
        # Count content word occurrences across sources
        # Track position-specific counts
        position_words: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for source in sources:
            words = self._tokenize(source)
            n = len(words)
            for i, word in enumerate(words):
                if word not in structure_words:
                    # Map to position bucket (10 buckets)
                    bucket = int((i / max(n - 1, 1)) * 9)
                    position_words[bucket][word] += 1
        
        # Build response
        result = []
        n_template = len(template_words)
        used_words = set()  # Avoid repeating content words
        
        for i, word in enumerate(template_words):
            if word in structure_words:
                result.append(word)
            else:
                # Find best content word for this position
                bucket = int((i / max(n_template - 1, 1)) * 9)
                candidates = position_words[bucket]
                
                # Find best unused word
                best_word = word
                best_count = 0
                
                for cand, count in candidates.items():
                    if cand not in used_words and count > best_count:
                        best_word = cand
                        best_count = count
                
                result.append(best_word)
                used_words.add(best_word)
        
        return " ".join(result)


# =============================================================================
# HOLOGRAPHIC PARAPHRASER
# =============================================================================

class HolographicParaphraser:
    """
    Generate paraphrases using holographic interference.
    
    Given multiple ways to say the same thing, interference reveals
    the common meaning while allowing variation in expression.
    """
    
    def __init__(self):
        # Paraphrase clusters: meaning_id -> list of expressions
        self.clusters: Dict[str, List[str]] = defaultdict(list)
    
    def add_paraphrase_cluster(self, cluster_id: str, expressions: List[str]):
        """Add a cluster of equivalent expressions."""
        self.clusters[cluster_id].extend(expressions)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def paraphrase(self, text: str, style: float = 0.0) -> str:
        """
        Generate a paraphrase of the input text.
        
        style: -1 = more formal, +1 = more casual
        """
        words = self._tokenize(text)
        result = []
        
        for word in words:
            # Check if word is in any cluster
            alternatives = []
            for cluster_id, expressions in self.clusters.items():
                for expr in expressions:
                    if word in self._tokenize(expr):
                        alternatives.extend(expressions)
                        break
            
            if alternatives and len(alternatives) > 1:
                # Select based on style
                # Simple heuristic: shorter = more casual
                alternatives.sort(key=len)
                if style > 0:
                    result.append(self._tokenize(alternatives[0])[0])
                elif style < 0:
                    result.append(self._tokenize(alternatives[-1])[0])
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return " ".join(result)


# =============================================================================
# HOLOGRAPHIC CONCEPT NAVIGATOR
# =============================================================================

class HolographicConceptNavigator:
    """
    Navigate concept space using GEOMETRIC holographic interference.
    
    Key insight: ENCODE = DECODE (same operation, opposite directions)
    
    GEOMETRIC ENCODING (replaces hash-based):
    - Phase = φ-direction × π (semantic role from knowledge)
    - Magnitude = role_strength (how strongly typed)
    - Structure words filtered out
    
    Applications:
    1. Concept similarity via interference
    2. Analogy completion (A:B :: C:?)
    3. Concept interpolation
    4. Relation extraction
    
    NOTE: For full semantic quaternion support (100% analogy accuracy),
    use SemanticQuaternionNavigator from semantic_quaternion.py instead.
    This class provides simpler complex-number encoding for basic operations.
    """
    
    PHI = 1.618034
    
    # Structure words to filter out
    STRUCTURE_WORDS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'a', 'an', 'the', 'who', 'that', 'which', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    }
    
    def __init__(self, knowledge=None):
        """
        Initialize with optional GeometricKnowledge.
        
        Args:
            knowledge: GeometricKnowledge for geometric encoding
        """
        self.knowledge = knowledge
        self.vocabulary: Dict[str, complex] = {}  # word -> complex position
        self.relations: Dict[str, complex] = {}   # relation -> complex vector
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def add_concept(self, concept: str):
        """Add a concept to the vocabulary."""
        z = self.encode_concept(concept)
        self.vocabulary[concept.lower()] = z
    
    def add_concepts(self, concepts: List[str]):
        """Add multiple concepts to the vocabulary."""
        for concept in concepts:
            self.add_concept(concept)
    
    def add_relation(self, name: str, a: str, b: str):
        """
        Learn a relation from an example pair.
        
        The relation vector is B - A.
        e.g., add_relation("capital_of", "France", "Paris")
        """
        z_a = self.encode_concept(a)
        z_b = self.encode_concept(b)
        self.relations[name.lower()] = z_b - z_a
    
    def _get_geometric_encoding(self, word: str) -> Tuple[float, float]:
        """
        Get geometric phase and magnitude for a word.
        
        Returns (phase, magnitude) based on geometric knowledge.
        """
        word_lower = word.lower()
        
        # Structure words: filter out (low magnitude)
        if word_lower in self.STRUCTURE_WORDS:
            return (0.0, 0.1)
        
        # Use geometric knowledge if available
        if self.knowledge and word_lower in self.knowledge.concepts:
            concept = self.knowledge.concepts[word_lower]
            
            # Phase from φ-direction: [-1, 1] → [π, 0]
            phi_dir = concept.phi_direction
            
            # Check if it's a verb (mediator)
            total_roles = concept.initiator_count + concept.mediator_count + concept.receiver_count
            if total_roles > 0:
                mediator_ratio = concept.mediator_count / total_roles
                if mediator_ratio > 0.5:
                    phase = math.pi / 2  # Verbs at π/2
                else:
                    phase = (1 - phi_dir) * math.pi / 2
            else:
                phase = math.pi / 2  # Unknown role
            
            # Magnitude from role strength
            magnitude = min(self.PHI, 0.5 + total_roles * 0.1)
            
            return (phase, magnitude)
        
        # Fallback: neutral encoding
        return (math.pi / 2, 1.0)
    
    def encode_concept(self, text: str) -> complex:
        """
        Encode text to φ-space position using GEOMETRIC encoding.
        
        Phase = φ-direction (semantic role)
        Magnitude = role_strength (how strongly typed)
        """
        words = self._tokenize(text)
        
        if not words:
            return complex(0, 0)
        
        total = complex(0, 0)
        
        for i, word in enumerate(words):
            # Get geometric encoding
            phase, magnitude = self._get_geometric_encoding(word)
            
            # Position in sentence adds small phase offset
            pos_phase = (i / max(len(words), 1)) * math.pi / 8
            phase += pos_phase
            
            # Complex encoding: magnitude × e^(i·phase)
            z = magnitude * cmath.exp(1j * phase)
            total += z
        
        return total
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity via holographic interference.
        
        Uses complex inner product - phases that align reinforce.
        """
        z1 = self.encode_concept(text1)
        z2 = self.encode_concept(text2)
        
        # Complex inner product: real part = cos(phase_difference)
        inner = (z1.conjugate() * z2).real
        
        # Normalize
        norm1 = abs(z1)
        norm2 = abs(z2)
        
        if norm1 > 0 and norm2 > 0:
            return inner / (norm1 * norm2)
        return 0.0
    
    def find_closest(self, z: complex, exclude: Set[str] = None) -> Tuple[str, float]:
        """
        Find the closest concept in vocabulary to a given position.
        
        Returns (concept, distance).
        """
        if exclude is None:
            exclude = set()
        
        best_concept = None
        best_distance = float('inf')
        
        for concept, z_concept in self.vocabulary.items():
            if concept in exclude:
                continue
            
            distance = abs(z - z_concept)
            if distance < best_distance:
                best_distance = distance
                best_concept = concept
        
        return best_concept, best_distance
    
    def find_k_closest(self, z: complex, k: int = 5, exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """
        Find the k closest concepts to a given position.
        
        Returns list of (concept, distance) sorted by distance.
        """
        if exclude is None:
            exclude = set()
        
        distances = []
        for concept, z_concept in self.vocabulary.items():
            if concept in exclude:
                continue
            distance = abs(z - z_concept)
            distances.append((concept, distance))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def complete_analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Complete analogy: A is to B as C is to ?
        
        Uses vector arithmetic in φ-space:
        ? = C + (B - A)
        
        Returns k closest concepts to the target position.
        """
        z_a = self.encode_concept(a)
        z_b = self.encode_concept(b)
        z_c = self.encode_concept(c)
        
        # Analogy vector: the relation from A to B
        relation = z_b - z_a
        
        # Apply relation to C
        z_target = z_c + relation
        
        # Find closest concepts (excluding the inputs)
        exclude = {a.lower(), b.lower(), c.lower()}
        return self.find_k_closest(z_target, k, exclude)
    
    def apply_relation(self, relation_name: str, concept: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Apply a learned relation to a concept.
        
        e.g., apply_relation("capital_of", "Germany") -> "Berlin"
        """
        if relation_name.lower() not in self.relations:
            return []
        
        relation = self.relations[relation_name.lower()]
        z_concept = self.encode_concept(concept)
        z_target = z_concept + relation
        
        exclude = {concept.lower()}
        return self.find_k_closest(z_target, k, exclude)
    
    def extract_relation(self, a: str, b: str) -> complex:
        """
        Extract the relation vector from A to B.
        
        This can be used to find similar relations.
        """
        z_a = self.encode_concept(a)
        z_b = self.encode_concept(b)
        return z_b - z_a
    
    def find_similar_relations(self, a: str, b: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find pairs in vocabulary with similar relations to A:B.
        
        Returns list of (concept1, concept2, similarity).
        """
        target_relation = self.extract_relation(a, b)
        target_mag = abs(target_relation)
        
        if target_mag == 0:
            return []
        
        similar = []
        concepts = list(self.vocabulary.keys())
        
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if c1 == a.lower() and c2 == b.lower():
                    continue
                if c1 == b.lower() and c2 == a.lower():
                    continue
                
                relation = self.vocabulary[c2] - self.vocabulary[c1]
                
                # Similarity via complex inner product
                inner = (target_relation.conjugate() * relation).real
                rel_mag = abs(relation)
                
                if rel_mag > 0:
                    sim = inner / (target_mag * rel_mag)
                    similar.append((c1, c2, sim))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: -x[2])
        return similar[:k]
    
    def interpolate(self, text1: str, text2: str, t: float = 0.5) -> complex:
        """
        Interpolate between two concepts.
        
        t=0 -> text1, t=1 -> text2
        """
        z1 = self.encode_concept(text1)
        z2 = self.encode_concept(text2)
        
        # Linear interpolation in complex plane
        return z1 * (1 - t) + z2 * t
    
    def interpolate_and_find(self, text1: str, text2: str, steps: int = 5) -> List[Tuple[float, str, float]]:
        """
        Interpolate between concepts and find closest vocabulary items.
        
        Returns list of (t, concept, distance) for each step.
        """
        results = []
        exclude = {text1.lower(), text2.lower()}
        
        for i in range(steps + 1):
            t = i / steps
            z = self.interpolate(text1, text2, t)
            concept, distance = self.find_closest(z, exclude)
            results.append((t, concept, distance))
        
        return results


# =============================================================================
# HOLOGRAPHIC SUMMARIZER
# =============================================================================

class HolographicSummarizer:
    """
    Summarize text using holographic interference.
    
    Key insight: Important concepts appear repeatedly.
    Interference amplifies repeated concepts, cancels noise.
    """
    
    STRUCTURE_WORDS = {
        'is', 'are', 'was', 'were', 'a', 'an', 'the',
        'who', 'that', 'which', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'he', 'she', 'it', 'they', 'his', 'her', 'their',
    }
    
    def __init__(self):
        pass
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Summarize text by keeping high-interference words.
        
        ratio: fraction of content to keep (0.3 = 30%)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        # Count word occurrences across sentences
        word_counts: Dict[str, int] = defaultdict(int)
        sentence_words: List[List[str]] = []
        
        for sentence in sentences:
            words = self._tokenize(sentence)
            sentence_words.append(words)
            seen = set()
            for word in words:
                if word not in self.STRUCTURE_WORDS and word not in seen:
                    word_counts[word] += 1
                    seen.add(word)
        
        # Score sentences by sum of word frequencies (interference strength)
        sentence_scores = []
        for i, words in enumerate(sentence_words):
            score = sum(word_counts.get(w, 0) for w in words if w not in self.STRUCTURE_WORDS)
            sentence_scores.append((score, i, sentences[i]))
        
        # Sort by score and take top ratio
        sentence_scores.sort(reverse=True)
        n_keep = max(1, int(len(sentences) * ratio))
        
        # Restore original order
        kept = sorted(sentence_scores[:n_keep], key=lambda x: x[1])
        
        return '. '.join(s for _, _, s in kept) + '.'
    
    def extract_key_concepts(self, text: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Extract key concepts via interference.
        
        Words that appear across multiple sentences = high interference = key.
        """
        sentences = re.split(r'[.!?]+', text)
        
        word_counts: Dict[str, int] = defaultdict(int)
        
        for sentence in sentences:
            words = self._tokenize(sentence)
            seen = set()
            for word in words:
                if word not in self.STRUCTURE_WORDS and len(word) > 3 and word not in seen:
                    word_counts[word] += 1
                    seen.add(word)
        
        # Sort by count
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        
        return sorted_words[:top_k]


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate all holographic capabilities."""
    print("=" * 70)
    print("HOLOGRAPHIC LANGUAGE PROCESSING DEMO")
    print("=" * 70)
    
    # =========================================================================
    # PART 1: TEMPLATE PROJECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: HOLOGRAPHIC TEMPLATE PROJECTION")
    print("=" * 70)
    
    # Create projector
    projector = HolographicTemplateProjector()
    
    # Add Q&A pairs - these are the "reference beams"
    qa_pairs = [
        # WHO IS questions - similar structure, different content
        ("Who is Watson?", "Watson is a loyal doctor who assists Holmes."),
        ("Who is Darcy?", "Darcy is a proud gentleman who loves Elizabeth."),
        ("Who is Moriarty?", "Moriarty is a cunning villain who opposes Holmes."),
        ("Who is Elizabeth?", "Elizabeth is a witty lady who challenges Darcy."),
        ("Who is Lestrade?", "Lestrade is a dedicated inspector who consults Holmes."),
        
        # WHAT DID questions
        ("What did Holmes do?", "Holmes investigated the mysterious case."),
        ("What did Watson do?", "Watson documented the important findings."),
        ("What did Darcy do?", "Darcy proposed to Elizabeth unexpectedly."),
        
        # WHERE questions
        ("Where is Holmes?", "Holmes is located in London."),
        ("Where is Darcy?", "Darcy is located in Derbyshire."),
    ]
    
    projector.add_qa_pairs_from_corpus(qa_pairs)
    
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs as reference beams")
    
    # Test template projection
    print("\n" + "-" * 70)
    print("1. TEMPLATE PROJECTION via Holographic Interference")
    print("-" * 70)
    
    test_queries = [
        "Who is Sherlock?",
        "Who is Jane?",
        "Where is Watson?",
    ]
    
    for query in test_queries:
        template = projector.project_template(query)
        print(f"\nQuery: {query}")
        print(f"  Projected Template: {template.pattern}")
        print(f"  Slots: {template.slots}")
    
    # Test generation with slot values
    print("\n" + "-" * 70)
    print("2. RESPONSE GENERATION (Template + Slot Filling)")
    print("-" * 70)
    
    test_cases = [
        ("Who is Sherlock?", {"adjective": "brilliant", "role": "detective", "action": "solves crimes"}),
        ("Who is Jane?", {"adjective": "kind", "role": "lady", "action": "loves Bingley"}),
        ("Where is Watson?", {"action": "located", "target": "London"}),
    ]
    
    for query, slot_values in test_cases:
        response = projector.generate(query, slot_values)
        print(f"\nQuery: {query}")
        print(f"  Response: {response}")
    
    # Demo response synthesis
    print("\n" + "-" * 70)
    print("3. RESPONSE SYNTHESIS via Multi-Source Interference")
    print("-" * 70)
    
    synthesizer = HolographicResponseSynthesizer()
    
    # Multiple descriptions of Holmes
    holmes_sources = [
        "Holmes is a brilliant detective who solves mysteries.",
        "Holmes is a clever investigator who examines evidence.",
        "Holmes is a famous detective who deduces solutions.",
        "Holmes is a skilled detective who observes details.",
    ]
    
    print("\nSources about Holmes:")
    for s in holmes_sources:
        print(f"  - {s}")
    
    synthesized = synthesizer.synthesize("Who is Holmes?", holmes_sources)
    print(f"\nSynthesized (common words): {synthesized}")
    
    structured = synthesizer.synthesize_with_structure("Who is Holmes?", holmes_sources)
    print(f"Synthesized (with structure): {structured}")
    
    # =========================================================================
    # PART 4: CONCEPT NAVIGATION & ANALOGY COMPLETION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: SEMANTIC QUATERNION NAVIGATION & ANALOGY")
    print("=" * 70)
    
    # NOTE: For analogies, use SemanticQuaternionNavigator (100% accuracy)
    # HolographicConceptNavigator uses geometric encoding but needs knowledge
    # to differentiate concepts. Without knowledge, it falls back to neutral.
    from .semantic_quaternion import SemanticQuaternionNavigator
    navigator = SemanticQuaternionNavigator()
    
    # Build a vocabulary
    vocabulary = [
        # People and roles
        "king", "queen", "man", "woman", "boy", "girl",
        "prince", "princess", "father", "mother", "son", "daughter",
        # Professions
        "doctor", "nurse", "actor", "actress",
        "waiter", "waitress", "host", "hostess",
        # Countries and capitals
        "france", "paris", "germany", "berlin",
        "japan", "tokyo", "italy", "rome",
        "spain", "madrid", "england", "london",
        # Animals
        "dog", "puppy", "cat", "kitten",
        # Actions
        "walk", "walked", "run", "ran",
        "speak", "spoke", "write", "wrote",
        # Sizes
        "big", "small", "tall", "short",
        # Detectives
        "holmes", "watson", "detective", "assistant",
        "moriarty", "villain", "lestrade", "inspector",
    ]
    
    navigator.add_concepts(vocabulary)
    print(f"\nBuilt vocabulary with {len(vocabulary)} concepts")
    
    # Test analogy completion
    print("\n" + "-" * 70)
    print("ANALOGY COMPLETION: A is to B as C is to ?")
    print("-" * 70)
    
    analogies = [
        # Gender analogies
        ("king", "queen", "man"),      # man -> woman
        ("man", "woman", "boy"),       # boy -> girl
        ("father", "mother", "son"),   # son -> daughter
        ("actor", "actress", "waiter"), # waiter -> waitress
        
        # Capital analogies
        ("france", "paris", "germany"),  # germany -> berlin
        ("japan", "tokyo", "italy"),     # italy -> rome
        
        # Role analogies
        ("holmes", "detective", "watson"),  # watson -> assistant
        ("holmes", "detective", "moriarty"), # moriarty -> villain
        
        # Tense analogies
        ("walk", "walked", "run"),    # run -> ran
        ("speak", "spoke", "write"),  # write -> wrote
        
        # Size analogies
        ("dog", "puppy", "cat"),      # cat -> kitten
    ]
    
    correct = 0
    total = len(analogies)
    
    for a, b, c in analogies:
        results = navigator.complete_analogy(a, b, c, k=3)
        top_answer = results[0][0] if results else "?"
        
        print(f"\n  {a} : {b} :: {c} : ?")
        print(f"    Top answers: {[r[0] for r in results[:3]]}")
        
        # Check if expected answer is in top 3
        expected = {
            ("king", "queen", "man"): "woman",
            ("man", "woman", "boy"): "girl",
            ("father", "mother", "son"): "daughter",
            ("actor", "actress", "waiter"): "waitress",
            ("france", "paris", "germany"): "berlin",
            ("japan", "tokyo", "italy"): "rome",
            ("holmes", "detective", "watson"): "assistant",
            ("holmes", "detective", "moriarty"): "villain",
            ("walk", "walked", "run"): "ran",
            ("speak", "spoke", "write"): "wrote",
            ("dog", "puppy", "cat"): "kitten",
        }
        
        exp = expected.get((a, b, c), "?")
        if exp in [r[0] for r in results[:3]]:
            print(f"    ✓ Expected '{exp}' found!")
            correct += 1
        else:
            print(f"    ✗ Expected '{exp}'")
    
    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Learn and apply relations
    print("\n" + "-" * 70)
    print("LEARNED RELATIONS")
    print("-" * 70)
    
    # Learn the "capital_of" relation
    navigator.add_relation("capital_of", "france", "paris")
    navigator.add_relation("gender_swap", "king", "queen")
    
    print("\nLearned relations:")
    print("  - capital_of (from france -> paris)")
    print("  - gender_swap (from king -> queen)")
    
    # Apply relations
    print("\nApplying 'capital_of' to spain:")
    results = navigator.apply_relation("capital_of", "spain", k=3)
    print(f"  Results: {[r[0] for r in results]}")
    
    print("\nApplying 'gender_swap' to prince:")
    results = navigator.apply_relation("gender_swap", "prince", k=3)
    print(f"  Results: {[r[0] for r in results]}")
    
    # Find similar relations
    print("\n" + "-" * 70)
    print("FINDING SIMILAR RELATIONS")
    print("-" * 70)
    
    print("\nPairs with similar relation to 'king' -> 'queen':")
    similar = navigator.find_similar_relations("king", "queen", k=5)
    for c1, c2, sim in similar:
        print(f"  {c1} -> {c2}: similarity = {sim:.3f}")
    
    # Concept interpolation (using quaternion interpolate method)
    print("\n" + "-" * 70)
    print("CONCEPT INTERPOLATION")
    print("-" * 70)
    
    print("\nInterpolating from 'detective' to 'villain':")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z = navigator.interpolate("detective", "villain", t)
        print(f"  t={t:.2f}: quaternion={z}")
    
    # Test similarity with vocabulary
    print("\n" + "-" * 70)
    print("CONCEPT SIMILARITY")
    print("-" * 70)
    
    pairs = [
        ("king", "queen"),
        ("king", "prince"),
        ("king", "dog"),
        ("holmes", "detective"),
        ("holmes", "watson"),
    ]
    
    for c1, c2 in pairs:
        sim = navigator.similarity(c1, c2)
        print(f"  {c1} vs {c2}: {sim:.3f}")
    
    # =========================================================================
    # PART 5: SUMMARIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: HOLOGRAPHIC SUMMARIZATION")
    print("=" * 70)
    
    summarizer = HolographicSummarizer()
    
    text = """
    Holmes examined the evidence carefully. The detective noticed muddy footprints 
    near the window. Watson documented the findings in his notebook. Holmes deduced 
    that the intruder had entered through the garden. The evidence pointed to someone 
    familiar with the house. Watson agreed with Holmes's assessment. The detective 
    concluded that the butler was the prime suspect.
    """
    
    print("\nOriginal text (7 sentences):")
    print(f"  {text.strip()[:100]}...")
    
    summary = summarizer.summarize(text, ratio=0.4)
    print(f"\nSummary (40%):")
    print(f"  {summary}")
    
    key_concepts = summarizer.extract_key_concepts(text, top_k=5)
    print(f"\nKey concepts (via interference):")
    for word, count in key_concepts:
        print(f"  - {word}: appears in {count} sentences")


if __name__ == "__main__":
    demo()
