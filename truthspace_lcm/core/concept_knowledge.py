#!/usr/bin/env python3
"""
Concept Knowledge: Language-Agnostic Knowledge Base with Holographic Projection

This module implements:
1. ConceptKnowledge - stores concept frames from any language
2. HolographicProjector - resolves queries in concept space, projects to target language
3. ConceptQA - Q&A interface using holographic gap-filling

The key insight from holographic stereoscopy:
    Question = Content - Gap    (has missing information)
    Answer   = Content + Fill   (provides missing information)

The gap in the question IS the intent. The answer fills that gap.
"""

import json
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from .concept_language import (
    ConceptFrame, ConceptExtractor, 
    ACTION_PRIMITIVES, SEMANTIC_ROLES,
    ENGLISH_VERBS, SPANISH_VERBS
)
from .answer_patterns import PatternAnswerGenerator
from .spatial_attention import SpatialAttention, get_attention, initialize_attention


# =============================================================================
# QUESTION AXES (from design doc 030)
# =============================================================================

QUESTION_AXES = {
    'WHO': {
        'role': 'AGENT',
        'patterns': ['who', 'whom', 'whose'],
        'fill_slot': 'agent',
    },
    'WHAT': {
        'role': 'PATIENT',
        'patterns': ['what', 'which'],
        'fill_slot': 'patient',
    },
    'WHERE': {
        'role': 'LOCATION',
        'patterns': ['where'],
        'fill_slot': 'location',
    },
    'WHEN': {
        'role': 'TIME',
        'patterns': ['when'],
        'fill_slot': 'time',
    },
    'WHY': {
        'role': 'PURPOSE',
        'patterns': ['why'],
        'fill_slot': 'purpose',
    },
    'HOW': {
        'role': 'MANNER',
        'patterns': ['how'],
        'fill_slot': 'manner',
    },
    'ACTION': {
        'role': 'ACTION',
        'patterns': ['did', 'does', 'do', 'doing'],
        'fill_slot': 'action',
    },
}


# =============================================================================
# CONCEPT KNOWLEDGE BASE
# =============================================================================

class ConceptKnowledge:
    """
    Language-agnostic knowledge base storing concept frames.
    
    Frames from any language can be stored and queried together
    because they share the same conceptual primitives.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.frames: List[Dict] = []
        self.frame_vectors: List[np.ndarray] = []
        self.entities: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
        self.extractor = ConceptExtractor()
        
        # Axis vectors for question types
        self.axis_vectors = {}
        for axis, info in QUESTION_AXES.items():
            self.axis_vectors[axis] = self._hash_to_vec(f'AXIS:{axis}')
    
    def _hash_to_vec(self, s: str) -> np.ndarray:
        """Deterministic hash to unit vector."""
        seed = hash(s) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)
    
    def _frame_to_vector(self, frame_dict: Dict) -> np.ndarray:
        """Convert frame dict to vector (order-independent)."""
        vec = np.zeros(self.dim)
        
        role_mappings = [
            ('agent', 'AGENT'),
            ('action', 'ACTION'),
            ('patient', 'PATIENT'),
            ('theme', 'THEME'),
            ('location', 'LOCATION'),
            ('goal', 'GOAL'),
            ('source', 'SOURCE'),
        ]
        
        for key, role in role_mappings:
            if key in frame_dict and frame_dict[key]:
                vec += self._hash_to_vec(f'{role}:{frame_dict[key]}')
        
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def load_corpus(self, corpus_path: str):
        """Load concept corpus from JSON file."""
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        
        self.frames = corpus.get('frames', [])
        self.entities = corpus.get('entities', {})
        self.relations = corpus.get('relations', {})
        
        # Pre-compute vectors for all frames
        self.frame_vectors = [self._frame_to_vector(f) for f in self.frames]
        
        # Initialize spatial attention with corpus frames and known entities
        initialize_attention(self.frames, set(self.entities.keys()))
        
        return len(self.frames)
    
    def add_frame(self, frame: ConceptFrame, source_text: str = '', source: str = ''):
        """Add a single frame to the knowledge base."""
        frame_dict = frame.to_dict()
        frame_dict['text'] = source_text
        frame_dict['source'] = source
        
        self.frames.append(frame_dict)
        self.frame_vectors.append(self._frame_to_vector(frame_dict))
        
        # Track entity
        if frame.agent:
            if frame.agent not in self.entities:
                self.entities[frame.agent] = {'actions': [], 'source': source}
            if frame.action and frame.action not in self.entities[frame.agent]['actions']:
                self.entities[frame.agent]['actions'].append(frame.action)
    
    def query_by_vector(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Find frames most similar to query vector."""
        if not self.frame_vectors:
            return []
        
        similarities = []
        for i, vec in enumerate(self.frame_vectors):
            sim = np.dot(query_vec, vec)
            similarities.append((self.frames[i], sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def query_by_frame(self, query_frame: ConceptFrame, k: int = 5) -> List[Tuple[Dict, float]]:
        """Find frames similar to query frame."""
        query_vec = self._frame_to_vector(query_frame.to_dict())
        return self.query_by_vector(query_vec, k)
    
    def query_by_entity(self, entity: str, k: int = 10) -> List[Dict]:
        """Find all frames involving an entity."""
        entity_lower = entity.lower()
        results = []
        
        for frame in self.frames:
            if (frame.get('agent', '').lower() == entity_lower or
                frame.get('patient', '').lower() == entity_lower):
                results.append(frame)
                if len(results) >= k:
                    break
        
        return results
    
    def query_by_action(self, action: str, k: int = 10) -> List[Dict]:
        """Find all frames with a specific action primitive."""
        action_upper = action.upper()
        results = []
        
        for frame in self.frames:
            if frame.get('action', '').upper() == action_upper:
                results.append(frame)
                if len(results) >= k:
                    break
        
        return results
    
    def get_entity_info(self, entity: str) -> Optional[Dict]:
        """Get stored information about an entity."""
        return self.entities.get(entity.lower())
    
    def get_relation(self, entity1: str, entity2: str) -> Optional[Dict]:
        """Get relation between two entities."""
        key = f'{entity1.lower()}:{entity2.lower()}'
        return self.relations.get(key)


# =============================================================================
# HOLOGRAPHIC PROJECTOR
# =============================================================================

class HolographicProjector:
    """
    Projects concept frames to natural language (English).
    
    Uses holographic principle: the answer fills the gap in the question.
    
    Architecture:
        Question -> Detect axis (WHO/WHAT/WHERE/etc)
        Query -> Find matching frames in concept space
        Answer -> Project filled slot to English
    """
    
    def __init__(self, knowledge: ConceptKnowledge, style_x: float = 0.0, 
                 perspective_y: float = 0.0, depth_z: float = 0.0):
        """
        Initialize the holographic projector with 3D φ-dial control.
        
        Args:
            knowledge: The concept knowledge base
            style_x: Horizontal dial (-1 to +1): Style
                -1 = formal, specific, rare
                +1 = casual, universal, common
            perspective_y: Vertical dial (-1 to +1): Perspective
                -1 = subjective, experiential
                 0 = objective, factual
                +1 = meta, analytical
            depth_z: Depth dial (-1 to +1): Elaboration
                -1 = terse, minimal
                 0 = standard, balanced
                +1 = elaborate, detailed
        """
        self.knowledge = knowledge
        self.noise_level = 0.0  # 0.0 = geodesic (cookie-cutter), 1.0 = max variation
        
        # 3D φ-dial control: "We control the horizontal. We control the vertical. We control the depth."
        self.style_x = style_x
        self.perspective_y = perspective_y
        self.depth_z = depth_z
        
        # Pattern-based answer generator with 3D φ-dial
        self.answer_generator = PatternAnswerGenerator(x=style_x, y=perspective_y, z=depth_z)
        
        # Concept extractor for parsing questions
        self.extractor = ConceptExtractor()
        
        # Reverse verb mapping: primitive -> English verbs
        self.primitive_to_verbs = {}
        for verb, primitive in ENGLISH_VERBS.items():
            if primitive not in self.primitive_to_verbs:
                self.primitive_to_verbs[primitive] = []
            self.primitive_to_verbs[primitive].append(verb)
    
    def detect_question_axis(self, question: str) -> Tuple[str, str]:
        """
        Detect the question type (axis) and the gap to fill.
        
        Returns: (axis_name, entity_being_asked_about)
        """
        question_lower = question.lower().strip()
        
        # Skip words that aren't real entities
        skip_words = {
            'who', 'what', 'where', 'when', 'why', 'how', 'did', 'does', 
            'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'to', 'of',
            'in', 'on', 'at', 'for', 'with', 'about', 'from', 'by',
            'mr', 'mrs', 'miss', 'ms', 'dr', 'sir', 'lord', 'lady',
        }
        
        # First try: Find capitalized words (entities), excluding question words
        all_caps = re.findall(r'\b([A-Z][a-z]+)\b', question)
        entities = [e for e in all_caps if e.lower() not in skip_words]
        entity = entities[0].lower() if entities else ''
        
        # Second try: If no capitalized entity found, extract from lowercase question
        if not entity:
            # Remove question words and common words, find remaining content words
            words = re.findall(r'\b([a-z]+)\b', question_lower)
            content_words = [w for w in words if w not in skip_words and len(w) > 2]
            if content_words:
                # Take the last content word (usually the subject in "who is X?")
                entity = content_words[-1]
        
        # Check each axis pattern
        for axis, info in QUESTION_AXES.items():
            for pattern in info['patterns']:
                if question_lower.startswith(pattern + ' '):
                    return axis, entity
        
        # Default to WHO axis
        return 'WHO', entity
    
    def extract_question_frame(self, question: str) -> ConceptFrame:
        """Extract a partial concept frame from a question."""
        frame = self.extractor.extract(question)
        if frame:
            return frame
        
        # Manual extraction for questions
        question_lower = question.lower()
        frame = ConceptFrame()
        
        # Find entities
        entities = re.findall(r'\b([A-Z][a-z]+)\b', question)
        if entities:
            frame.agent = entities[0].lower()
        
        # Find action hints
        for verb, primitive in ENGLISH_VERBS.items():
            if verb in question_lower:
                frame.action = primitive
                break
        
        return frame
    
    def project_to_english(self, frame: Dict, axis: str) -> str:
        """
        Project a concept frame to English, emphasizing the axis slot.
        
        This is the "fill" operation - we're providing the missing information.
        """
        agent = frame.get('agent', 'someone')
        action = frame.get('action', 'EXIST')
        patient = frame.get('patient', '')
        location = frame.get('location', '')
        goal = frame.get('goal', '')
        theme = frame.get('theme', '')
        
        # Get English verb for action
        verbs = self.primitive_to_verbs.get(action, ['is'])
        # Prefer past tense for narrative
        past_verbs = [v for v in verbs if v.endswith('ed')]
        verb = past_verbs[0] if past_verbs else verbs[0]
        
        # Action-specific verb selection
        action_verbs = {
            'MOVE': 'went',
            'SPEAK': 'said',
            'THINK': 'thought',
            'PERCEIVE': 'saw',
            'FEEL': 'felt',
            'ACT': 'did',
            'EXIST': 'is',
            'POSSESS': 'has',
            'BECOME': 'became',
        }
        verb = action_verbs.get(action, verb)
        
        # Build answer based on axis
        if axis == 'WHO':
            # Answer is about identity - describe what the entity does/is
            actions_desc = {
                'MOVE': 'travels',
                'SPEAK': 'speaks',
                'THINK': 'thinks',
                'PERCEIVE': 'observes',
                'FEEL': 'feels deeply',
                'ACT': 'acts',
                'EXIST': 'exists',
                'POSSESS': 'possesses',
            }
            action_word = actions_desc.get(action, 'appears')
            
            if patient and patient != agent:
                return f"{agent.title()} {action_word} about {patient}"
            elif theme and theme != agent:
                return f"{agent.title()} {action_word} regarding {theme}"
            else:
                return f"{agent.title()} {action_word} in the story"
        
        elif axis == 'WHAT':
            # Answer is about what happened
            if action == 'MOVE':
                if goal:
                    return f"{agent.title()} went to {goal}"
                elif location:
                    return f"{agent.title()} traveled to {location}"
                else:
                    return f"{agent.title()} moved"
            elif action == 'SPEAK':
                if patient:
                    return f"{agent.title()} spoke to {patient}"
                else:
                    return f"{agent.title()} spoke"
            elif action == 'THINK':
                if patient:
                    return f"{agent.title()} thought about {patient}"
                elif theme:
                    return f"{agent.title()} considered {theme}"
                else:
                    return f"{agent.title()} pondered"
            elif action == 'PERCEIVE':
                if patient:
                    return f"{agent.title()} saw {patient}"
                else:
                    return f"{agent.title()} observed"
            elif action == 'FEEL':
                if patient:
                    return f"{agent.title()} felt something for {patient}"
                else:
                    return f"{agent.title()} experienced emotions"
            elif action == 'POSSESS':
                if patient:
                    return f"{agent.title()} has {patient}"
                else:
                    return f"{agent.title()} possesses something"
            else:
                if patient:
                    return f"{agent.title()} {verb} {patient}"
                else:
                    return f"{agent.title()} {verb}"
        
        elif axis == 'WHERE':
            # Answer is about location
            if location:
                return f"at {location}"
            elif goal:
                return f"going to {goal}"
            else:
                return f"somewhere in the story"
        
        elif axis == 'ACTION':
            # Answer is about what action occurred
            return f"{verb}"
        
        else:
            # Default: full sentence
            parts = [agent.title()]
            parts.append(verb)
            if patient and patient != agent:
                parts.append(patient)
            if location:
                parts.append(f"at {location}")
            return ' '.join(parts)
    
    def resolve(self, question: str, k: int = 3) -> List[Dict]:
        """
        Resolve a question using holographic projection.
        
        1. Detect question axis (the gap)
        2. Extract partial frame from question
        3. Query concept space for matching frames
        4. Aggregate knowledge about entity
        5. Project answers to English (fill the gap)
        
        Returns list of possible answers with confidence.
        """
        # Step 1: Detect axis
        axis, entity = self.detect_question_axis(question)
        
        # Step 2: Extract question frame
        q_frame = self.extract_question_frame(question)
        
        # Step 3: Query concept space - get MORE frames for aggregation
        if entity:
            # Query by entity - get many frames to aggregate
            # Use k=100 to get representative sample of entity's actions
            entity_frames = self.knowledge.query_by_entity(entity, k=100)
            
            # Aggregate actions for this entity
            action_counts = {}
            patients = {}
            locations = set()
            sources = set()
            
            for f in entity_frames:
                action = f.get('action', '')
                if action:
                    action_counts[action] = action_counts.get(action, 0) + 1
                patient = f.get('patient', '')
                if patient and patient != entity:
                    patients[patient] = patients.get(patient, 0) + 1
                loc = f.get('location', '')
                if loc:
                    locations.add(loc)
                sources.add(f.get('source', ''))
            
            # Build aggregated answer based on axis
            # Determine navigation direction based on question axis
            navigation = self._get_navigation_for_axis(axis)
            
            if axis == 'WHO':
                # Describe who this entity is based on their actions
                top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
                
                # Use SPATIAL ATTENTION with question-driven navigation
                attention = get_attention()
                if attention._initialized:
                    # Get attention-weighted important relations using appropriate navigation
                    important_relations = attention.get_important_relations(
                        entity, k=3, navigation=navigation
                    )
                    if important_relations:
                        top_patients = important_relations
                    else:
                        top_patients = sorted(patients.items(), key=lambda x: -x[1])[:2]
                else:
                    top_patients = sorted(patients.items(), key=lambda x: -x[1])[:2]
                
                answer = self._build_who_answer(entity, top_actions, top_patients, sources, 
                                               noise_level=getattr(self, 'noise_level', 0.0))
            elif axis == 'WHAT':
                # WHAT questions use OUTWARD navigation (universal patterns)
                top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
                
                attention = get_attention()
                if attention._initialized:
                    important_relations = attention.get_important_relations(
                        entity, k=3, navigation='outward'
                    )
                    if important_relations:
                        top_patients = important_relations
                    else:
                        top_patients = sorted(patients.items(), key=lambda x: -x[1])[:2]
                else:
                    top_patients = sorted(patients.items(), key=lambda x: -x[1])[:2]
                
                answer = self._build_what_answer(entity, top_actions, top_patients)
            elif axis == 'WHERE':
                answer = self._build_where_answer(entity, locations)
            else:
                # Default: use first frame
                if entity_frames:
                    answer = self.project_to_english(entity_frames[0], axis)
                else:
                    answer = f"Information about {entity} not found"
            
            return [{
                'answer': answer,
                'confidence': 0.8 if entity_frames else 0.0,
                'frame': entity_frames[0] if entity_frames else {},
                'axis': axis,
                'source': ', '.join(list(sources)[:2]),
                'original_text': entity_frames[0].get('text', '')[:100] if entity_frames else '',
                'frame_count': len(entity_frames),
            }]
        else:
            results = self.knowledge.query_by_frame(q_frame, k=k)
        
        # Step 4: Project to English
        answers = []
        for frame, confidence in results:
            english = self.project_to_english(frame, axis)
            answers.append({
                'answer': english,
                'confidence': confidence,
                'frame': frame,
                'axis': axis,
                'source': frame.get('source', 'unknown'),
                'original_text': frame.get('text', '')[:100],
            })
        
        return answers
    
    def _build_who_answer(self, entity: str, actions: List, patients: List, sources: set, 
                          noise_level: float = 0.0) -> str:
        """Build a WHO answer using pattern-based generator with optional noise."""
        return self.answer_generator.generate_who_answer(entity, actions, patients, sources, noise_level)
    
    def _build_what_answer(self, entity: str, actions: List, patients: List) -> str:
        """Build a WHAT answer using pattern-based generator."""
        return self.answer_generator.generate_what_answer(entity, actions, patients)
    
    def _build_where_answer(self, entity: str, locations: set) -> str:
        """Build a WHERE answer using pattern-based generator."""
        return self.answer_generator.generate_where_answer(entity, locations)
    
    def _get_navigation_for_axis(self, axis: str) -> str:
        """
        Map question axis to φ-navigation direction.
        
        The φ-structure supports dual navigation:
            - INWARD (φ^-n): Find specific instances (WHO, WHERE, WHEN)
            - OUTWARD (φ^+n): Find universal patterns (WHAT, HOW)
            - OSCILLATING: Traverse causal chains (WHY)
        
        This is derived from the self-dual property of φ:
            φ^(-n) × φ^(+n) = 1 (always)
        """
        mapping = {
            'WHO': 'inward',      # Find specific individuals
            'WHAT': 'outward',    # Find universal categories
            'WHERE': 'inward',    # Find specific locations
            'WHEN': 'inward',     # Find specific times
            'HOW': 'outward',     # Find general patterns
            'WHY': 'inward',      # TODO: implement oscillating for causal chains
        }
        return mapping.get(axis, 'inward')


# =============================================================================
# CONCEPT Q&A INTERFACE
# =============================================================================

class ConceptQA:
    """
    Q&A interface using holographic concept resolution.
    
    The key insight:
        Question = Content - Gap
        Answer = Content + Fill
    
    We resolve in concept space (language-agnostic),
    then project to the target language (English).
    """
    
    def __init__(self, corpus_path: Optional[str] = None, 
                 style_x: float = 0.0, perspective_y: float = 0.0, depth_z: float = 0.0):
        """
        Initialize the Q&A system with 3D φ-dial control.
        
        Args:
            corpus_path: Optional path to concept corpus
            style_x: Horizontal dial (-1 to +1): Style
                -1 = formal, specific, rare
                +1 = casual, universal, common
            perspective_y: Vertical dial (-1 to +1): Perspective
                -1 = subjective, experiential
                 0 = objective, factual
                +1 = meta, analytical
            depth_z: Depth dial (-1 to +1): Elaboration
                -1 = terse, minimal
                 0 = standard, balanced
                +1 = elaborate, detailed
        """
        self.knowledge = ConceptKnowledge()
        self.projector = HolographicProjector(self.knowledge, style_x, perspective_y, depth_z)
        
        # Store dial settings
        self.style_x = style_x
        self.perspective_y = perspective_y
        self.depth_z = depth_z
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def set_dial(self, x: float = None, y: float = None, z: float = None):
        """
        Set the 3D φ-dial for style, perspective, and depth control.
        
        Args:
            x: Horizontal dial (-1 to +1): Style (formal ↔ casual)
            y: Vertical dial (-1 to +1): Perspective (subjective ↔ meta)
            z: Depth dial (-1 to +1): Elaboration (terse ↔ elaborate)
        """
        if x is not None:
            self.style_x = x
        if y is not None:
            self.perspective_y = y
        if z is not None:
            self.depth_z = z
        
        # Update the answer generator's dial
        self.projector.answer_generator.set_dial(self.style_x, self.perspective_y, self.depth_z)
    
    def set_style(self, x: float):
        """Set horizontal style dial (-1 = formal, +1 = casual)."""
        self.set_dial(x=x)
    
    def set_perspective(self, y: float):
        """Set vertical perspective dial (-1 = subjective, +1 = meta)."""
        self.set_dial(y=y)
    
    def set_depth(self, z: float):
        """Set depth dial (-1 = terse, +1 = elaborate)."""
        self.set_dial(z=z)
    
    def load_corpus(self, corpus_path: str) -> int:
        """Load concept corpus."""
        return self.knowledge.load_corpus(corpus_path)
    
    def ask(self, question: str, k: int = 3, noise_level: float = 0.0) -> str:
        """
        Ask a question and get an answer.
        
        Process:
        1. Parse question to detect axis (the gap)
        2. Query concept space for relevant frames
        3. Project best match to English (fill the gap)
        
        Args:
            noise_level: 0.0 = geodesic (cookie-cutter), 1.0 = max variation
        """
        # Set noise level on projector
        self.projector.noise_level = noise_level
        
        answers = self.projector.resolve(question, k=k)
        
        if not answers:
            return "I don't have information about that."
        
        # Return best answer
        best = answers[0]
        
        # Format response
        response = best['answer']
        
        # Add source attribution based on depth dial
        if best.get('source') and self.projector.answer_generator.dial.include_source():
            response += f" (from {best['source']})"
        
        return response
    
    def ask_detailed(self, question: str, k: int = 5) -> Dict:
        """
        Ask a question and get detailed response with alternatives.
        """
        answers = self.projector.resolve(question, k=k)
        
        axis, entity = self.projector.detect_question_axis(question)
        
        return {
            'question': question,
            'axis': axis,
            'entity': entity,
            'answers': answers,
            'best_answer': answers[0]['answer'] if answers else None,
        }
    
    def ingest_text(self, text: str, source: str = 'unknown'):
        """Ingest new text into the knowledge base."""
        extractor = ConceptExtractor()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        count = 0
        for sent in sentences:
            frame = extractor.extract(sent)
            if frame and frame.action:
                self.knowledge.add_frame(frame, sent, source)
                count += 1
        
        return count
    
    def chat(self):
        """Interactive chat interface."""
        print("=== Concept Q&A (Holographic Resolution) ===")
        print("Ask questions about characters and events from literature.")
        print("Type 'quit' to exit, 'debug' for detailed mode.\n")
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            
            if debug_mode:
                result = self.ask_detailed(user_input)
                print(f"\nAxis: {result['axis']}, Entity: {result['entity']}")
                print(f"Best: {result['best_answer']}")
                print("Alternatives:")
                for ans in result['answers'][:3]:
                    print(f"  [{ans['confidence']:.2f}] {ans['answer']}")
                    print(f"       Source: {ans['source']}")
                print()
            else:
                answer = self.ask(user_input)
                print(f"Bot: {answer}\n")


def test_concept_qa():
    """Test the concept Q&A system."""
    print("=== CONCEPT Q&A TEST ===\n")
    
    # Load corpus
    qa = ConceptQA()
    corpus_path = Path(__file__).parent.parent / 'concept_corpus.json'
    
    if corpus_path.exists():
        count = qa.load_corpus(str(corpus_path))
        print(f"Loaded {count} concept frames\n")
    else:
        print(f"Corpus not found at {corpus_path}")
        return
    
    # Test questions
    questions = [
        "Who is Darcy?",
        "What did Holmes do?",
        "Who is Elizabeth?",
        "What did Dracula do?",
        "Who is Quijote?",
        "What did Alice do?",
    ]
    
    for q in questions:
        result = qa.ask_detailed(q)
        print(f"Q: {q}")
        print(f"A: {result['best_answer']}")
        print(f"   (axis={result['axis']}, entity={result['entity']})")
        print()


if __name__ == "__main__":
    test_concept_qa()
