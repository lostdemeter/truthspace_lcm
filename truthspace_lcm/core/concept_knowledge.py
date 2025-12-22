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
    
    def __init__(self, knowledge: ConceptKnowledge):
        self.knowledge = knowledge
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
        skip_words = {'who', 'what', 'where', 'when', 'why', 'how', 'did', 'does', 
                      'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an'}
        
        # Find capitalized words (entities), excluding question words
        all_caps = re.findall(r'\b([A-Z][a-z]+)\b', question)
        entities = [e for e in all_caps if e.lower() not in skip_words]
        entity = entities[0].lower() if entities else ''
        
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
            entity_frames = self.knowledge.query_by_entity(entity, k=20)
            
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
            if axis == 'WHO':
                # Describe who this entity is based on their actions
                top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
                top_patients = sorted(patients.items(), key=lambda x: -x[1])[:2]
                
                answer = self._build_who_answer(entity, top_actions, top_patients, sources)
            elif axis == 'WHAT':
                # Describe what this entity did
                top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
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
    
    def _build_who_answer(self, entity: str, actions: List, patients: List, sources: set) -> str:
        """Build a WHO answer by aggregating entity knowledge."""
        action_descs = {
            'MOVE': 'travels',
            'SPEAK': 'speaks',
            'THINK': 'contemplates',
            'PERCEIVE': 'observes',
            'FEEL': 'experiences emotions',
            'ACT': 'takes action',
            'EXIST': 'appears',
            'POSSESS': 'possesses things',
        }
        
        parts = [f"{entity.title()} is a character"]
        
        # Add source
        source_list = [s for s in sources if s]
        if source_list:
            parts[0] += f" from {source_list[0]}"
        
        # Add main actions
        if actions:
            action_words = []
            for action, count in actions[:2]:
                desc = action_descs.get(action, 'acts')
                action_words.append(desc)
            if action_words:
                parts.append(f"who {' and '.join(action_words)}")
        
        # Add relationships
        if patients:
            top_patient = patients[0][0]
            parts.append(f"often involving {top_patient}")
        
        return ' '.join(parts)
    
    def _build_what_answer(self, entity: str, actions: List, patients: List) -> str:
        """Build a WHAT answer describing entity's actions."""
        action_verbs = {
            'MOVE': 'traveled',
            'SPEAK': 'spoke',
            'THINK': 'thought',
            'PERCEIVE': 'observed',
            'FEEL': 'felt',
            'ACT': 'acted',
            'EXIST': 'was present',
            'POSSESS': 'had',
        }
        
        if not actions:
            return f"{entity.title()} appears in the story"
        
        # Build action description
        action_parts = []
        for action, count in actions[:2]:
            verb = action_verbs.get(action, 'did something')
            action_parts.append(verb)
        
        result = f"{entity.title()} {', '.join(action_parts)}"
        
        # Add patient if relevant
        if patients:
            top_patient = patients[0][0]
            result += f" regarding {top_patient}"
        
        return result
    
    def _build_where_answer(self, entity: str, locations: set) -> str:
        """Build a WHERE answer."""
        if locations:
            loc_list = list(locations)[:3]
            return f"{entity.title()} appears at: {', '.join(loc_list)}"
        return f"Location of {entity} not specified"


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
    
    def __init__(self, corpus_path: Optional[str] = None):
        self.knowledge = ConceptKnowledge()
        self.projector = HolographicProjector(self.knowledge)
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, corpus_path: str) -> int:
        """Load concept corpus."""
        return self.knowledge.load_corpus(corpus_path)
    
    def ask(self, question: str, k: int = 3) -> str:
        """
        Ask a question and get an answer.
        
        Process:
        1. Parse question to detect axis (the gap)
        2. Query concept space for relevant frames
        3. Project best match to English (fill the gap)
        """
        answers = self.projector.resolve(question, k=k)
        
        if not answers:
            return "I don't have information about that."
        
        # Return best answer
        best = answers[0]
        
        # Format response
        response = best['answer']
        
        # Add source attribution if available
        if best.get('source'):
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
