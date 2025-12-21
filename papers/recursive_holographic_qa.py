#!/usr/bin/env python3
"""
Recursive Holographic Q&A System

The key insight: Q&A projection is RECURSIVE.

Level 1: Raw text → Statements (sentence extraction)
Level 2: Statements → Semantic Triples (subject, predicate, object)
Level 3: Triples → Q&A pairs (question generation)
Level 4: Q&A pairs → Refined answers (answer extraction)

Each level is a GEOMETRIC PROJECTION that extracts more structure.

The HOLOGRAPHIC principle:
  - A question Q defines a DIRECTION (axis) in semantic space
  - The answer A is the PROJECTION of the statement onto that axis
  - Different questions = different projections of the same statement

Mathematical formulation:
  v_S = statement vector
  v_axis = question type axis (WHO, WHAT, WHERE, etc.)
  
  answer_relevance = v_S · v_axis  (dot product = projection magnitude)
  answer_content = component of S along axis direction
"""

import sys
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# SEMANTIC TRIPLE - The atomic unit of meaning
# ============================================================================

class SemanticTriple(NamedTuple):
    """A semantic triple: (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    modifiers: Dict[str, str]  # axis -> value (e.g., 'LOCATION' -> 'in London')
    source: str  # Original sentence


# ============================================================================
# QUESTION AXES - Geometric directions in semantic space
# ============================================================================

class QuestionAxis:
    """
    A question axis defines a DIRECTION in semantic space.
    
    Projecting a statement onto this axis extracts the answer
    for questions of this type.
    """
    
    def __init__(self, name: str, dim: int = 64):
        self.name = name
        self.dim = dim
        
        # Initialize axis direction (deterministic from name)
        np.random.seed(hash(name) % (2**32))
        self.direction = np.random.randn(dim)
        self.direction = self.direction / np.linalg.norm(self.direction)
        np.random.seed(None)
    
    def project(self, vector: np.ndarray) -> float:
        """Project a vector onto this axis. Returns magnitude."""
        return np.dot(vector, self.direction)
    
    def component(self, vector: np.ndarray) -> np.ndarray:
        """Get the component of vector along this axis."""
        mag = self.project(vector)
        return mag * self.direction


# Define the standard question axes
AXES = {
    'SUBJECT': QuestionAxis('SUBJECT'),      # WHO/WHAT is the subject
    'PREDICATE': QuestionAxis('PREDICATE'),  # What ACTION/STATE
    'OBJECT': QuestionAxis('OBJECT'),        # WHO/WHAT is affected
    'LOCATION': QuestionAxis('LOCATION'),    # WHERE
    'TIME': QuestionAxis('TIME'),            # WHEN
    'REASON': QuestionAxis('REASON'),        # WHY
    'METHOD': QuestionAxis('METHOD'),        # HOW
    'ATTRIBUTE': QuestionAxis('ATTRIBUTE'),  # What KIND
}


# ============================================================================
# TRIPLE EXTRACTOR - Level 2 projection
# ============================================================================

def extract_triples(sentence: str) -> List[SemanticTriple]:
    """
    Extract semantic triples from a sentence.
    
    This is the Level 2 projection: Statement → Triples
    """
    sentence = sentence.strip()
    if not sentence or sentence.endswith('?'):
        return []
    
    triples = []
    
    # Pattern 1: Subject + BE-verb + Object
    be_pattern = r'^(.+?)\s+(is|was|are|were|has been|had been)\s+(.+?)([,.]|$)'
    match = re.match(be_pattern, sentence, re.IGNORECASE)
    
    if match:
        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        rest = match.group(3).strip()
        
        # Extract modifiers (prepositional phrases)
        modifiers = {}
        
        # Location: in/at/from/near + noun phrase
        loc_match = re.search(r'\b(in|at|from|near|to)\s+([A-Z][^,]+?)(?=\s+(?:in|at|from|near|to|who|which|that|,|\.|$))', rest, re.IGNORECASE)
        if not loc_match:
            loc_match = re.search(r'\b(in|at|from|near)\s+the\s+(\w+(?:\s+\w+)?)', rest, re.IGNORECASE)
        if loc_match:
            modifiers['LOCATION'] = f"{loc_match.group(1)} {loc_match.group(2)}"
            rest = rest.replace(loc_match.group(0), '').strip()
        
        # Time: in/on/at/during/before/after + time expression
        time_match = re.search(r'\b(in|on|at|during|before|after)\s+(\d{4}|the \w+|yesterday|today|tomorrow)', rest, re.IGNORECASE)
        if time_match:
            modifiers['TIME'] = f"{time_match.group(1)} {time_match.group(2)}"
            rest = rest.replace(time_match.group(0), '').strip()
        
        # Reason: because/since/due to + clause
        reason_match = re.search(r'\b(because|since|due to)\s+(.+?)(?=[,.]|$)', sentence, re.IGNORECASE)
        if reason_match:
            modifiers['REASON'] = reason_match.group(2).strip()
        
        # Method: by/through/using + noun phrase
        method_match = re.search(r'\b(by|through|using)\s+(.+?)(?=\s+(?:in|at|from|,|$))', rest, re.IGNORECASE)
        if method_match:
            modifiers['METHOD'] = f"{method_match.group(1)} {method_match.group(2)}"
            rest = rest.replace(method_match.group(0), '').strip()
        
        # Clean up object
        obj = re.sub(r'\s+', ' ', rest).strip(' ,.')
        
        # Extract attributes (adjectives)
        attr_match = re.findall(r'\b(\w+(?:ous|ive|ful|less|able|ible|al|ic|ed|ing|ant|ent))\b', obj, re.IGNORECASE)
        if attr_match:
            modifiers['ATTRIBUTE'] = ', '.join(attr_match)
        
        triple = SemanticTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            modifiers=modifiers,
            source=sentence
        )
        triples.append(triple)
    
    # Pattern 2: Subject + Action verb + Object
    action_verbs = r'(killed|destroyed|created|found|discovered|became|hunts|hunted|loves|loved|hates|hated|took|gave|made|built|wrote|said)'
    action_pattern = rf'^(.+?)\s+{action_verbs}\s+(.+?)([,.]|$)'
    match = re.match(action_pattern, sentence, re.IGNORECASE)
    
    if match and not triples:  # Only if we didn't already extract
        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        obj = match.group(3).strip(' ,.')
        
        modifiers = {}
        
        # Check for reason in original sentence
        reason_match = re.search(r'\b(because|since|due to)\s+(.+?)(?=[,.]|$)', sentence, re.IGNORECASE)
        if reason_match:
            modifiers['REASON'] = reason_match.group(2).strip()
        
        triple = SemanticTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            modifiers=modifiers,
            source=sentence
        )
        triples.append(triple)
    
    return triples


# ============================================================================
# Q&A GENERATOR - Level 3 projection
# ============================================================================

class QAPair(NamedTuple):
    """A question-answer pair."""
    question: str
    answer: str
    axis: str
    triple: SemanticTriple
    confidence: float


def generate_qa_pairs(triple: SemanticTriple) -> List[QAPair]:
    """
    Generate Q&A pairs from a semantic triple.
    
    This is the Level 3 projection: Triple → Q&A pairs
    
    Each axis generates a different question about the same triple.
    """
    pairs = []
    
    subj = triple.subject
    pred = triple.predicate
    obj = triple.object
    
    # SUBJECT axis: "Who/What is [object]?" → subject
    # (Reverse lookup)
    if obj and subj:
        q = f"Who is {obj}?" if 'person' in obj.lower() or re.match(r'^[A-Z]', obj) else f"What is {obj}?"
        # This doesn't make sense for most cases, skip
    
    # OBJECT axis: "Who/What is [subject]?" → object
    if subj and obj:
        # Determine if subject is a person or thing
        is_person = any(w in subj.lower() for w in ['captain', 'mr', 'mrs', 'dr', 'professor'])
        is_person = is_person or re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', subj)  # Two capitalized words
        
        if is_person:
            q = f"Who is {subj}?"
        else:
            q = f"What is {subj}?"
        
        pairs.append(QAPair(q, obj, 'IDENTITY', triple, 0.9))
        
        # Also add "Tell me about X"
        pairs.append(QAPair(f"Tell me about {subj}", triple.source, 'IDENTITY', triple, 0.7))
    
    # LOCATION axis: "Where is [subject]?" → location modifier
    if 'LOCATION' in triple.modifiers:
        q = f"Where is {subj}?"
        a = triple.modifiers['LOCATION']
        pairs.append(QAPair(q, a, 'LOCATION', triple, 0.85))
        
        # Also: "Where did [subject] [predicate]?"
        if pred not in ['is', 'was', 'are', 'were']:
            q = f"Where did {subj} {pred}?"
            pairs.append(QAPair(q, a, 'LOCATION', triple, 0.8))
    
    # TIME axis: "When did [subject] [predicate]?" → time modifier
    if 'TIME' in triple.modifiers:
        q = f"When did {subj} {pred}?"
        a = triple.modifiers['TIME']
        pairs.append(QAPair(q, a, 'TIME', triple, 0.85))
    
    # REASON axis: "Why did [subject] [predicate]?" → reason modifier
    if 'REASON' in triple.modifiers:
        q = f"Why did {subj} {pred}?"
        a = triple.modifiers['REASON']
        pairs.append(QAPair(q, a, 'REASON', triple, 0.9))
        
        # Also: "Why does [subject] [predicate]?"
        q = f"Why does {subj} {pred}?"
        pairs.append(QAPair(q, a, 'REASON', triple, 0.85))
    
    # METHOD axis: "How did [subject] [predicate]?" → method modifier
    if 'METHOD' in triple.modifiers:
        q = f"How did {subj} {pred}?"
        a = triple.modifiers['METHOD']
        pairs.append(QAPair(q, a, 'METHOD', triple, 0.85))
    
    # ATTRIBUTE axis: "What kind of [subject]?" → attribute modifier
    if 'ATTRIBUTE' in triple.modifiers:
        q = f"What kind of {subj}?"
        a = triple.modifiers['ATTRIBUTE']
        pairs.append(QAPair(q, a, 'ATTRIBUTE', triple, 0.7))
    
    # EVENT axis: "What happened to [subject]?" → predicate + object
    if pred in ['destroyed', 'killed', 'died', 'sank', 'fell', 'broke']:
        q = f"What happened to {subj}?"
        a = f"{pred} {obj}" if obj else pred
        pairs.append(QAPair(q, a, 'EVENT', triple, 0.85))
    
    return pairs


# ============================================================================
# RECURSIVE HOLOGRAPHIC Q&A SPACE
# ============================================================================

class RecursiveHolographicQA:
    """
    A Q&A system that uses recursive holographic projection.
    
    Level 1: Text → Sentences
    Level 2: Sentences → Semantic Triples
    Level 3: Triples → Q&A Pairs
    Level 4: Query → Match Q&A Pairs → Extract Answer
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.triples: List[SemanticTriple] = []
        self.qa_pairs: List[Tuple[np.ndarray, QAPair]] = []
        self.word_positions: Dict[str, np.ndarray] = {}
        self.axes = {name: QuestionAxis(name, dim) for name in AXES}
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_word_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.5
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text as weighted average of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        # Stop words get lower weight
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and'}
        
        pos = np.zeros(self.dim)
        weight_sum = 0
        
        for word in words:
            weight = 0.3 if word in stop_words else 1.0
            pos += weight * self._get_word_position(word)
            weight_sum += weight
        
        if weight_sum > 0:
            pos = pos / weight_sum
        
        return pos
    
    def _encode_question(self, question: str, axis_name: str) -> np.ndarray:
        """
        Encode a question as: content + axis direction.
        
        The axis direction biases the encoding toward questions of that type.
        """
        content = self._encode_text(question)
        
        if axis_name in self.axes:
            axis_dir = self.axes[axis_name].direction
            # Combine content and axis direction
            combined = 0.7 * content + 0.3 * axis_dir
        else:
            combined = content
        
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm
        
        return combined
    
    def _detect_axis(self, question: str) -> str:
        """Detect which axis a question belongs to."""
        q = question.lower()
        
        if q.startswith('who '):
            return 'IDENTITY'
        if q.startswith('what is') or q.startswith('what was') or q.startswith('what are'):
            return 'IDENTITY'
        if q.startswith('what happened') or q.startswith('what did'):
            return 'EVENT'
        if q.startswith('where '):
            return 'LOCATION'
        if q.startswith('when '):
            return 'TIME'
        if q.startswith('why '):
            return 'REASON'
        if q.startswith('how '):
            return 'METHOD'
        if 'kind of' in q or 'type of' in q:
            return 'ATTRIBUTE'
        if q.startswith('tell me about'):
            return 'IDENTITY'
        
        return 'IDENTITY'  # Default
    
    def ingest_sentence(self, sentence: str) -> int:
        """
        Ingest a sentence through recursive projection.
        
        Returns number of Q&A pairs generated.
        """
        # Level 2: Extract triples
        triples = extract_triples(sentence)
        
        total_pairs = 0
        for triple in triples:
            self.triples.append(triple)
            
            # Level 3: Generate Q&A pairs
            qa_pairs = generate_qa_pairs(triple)
            
            for qa in qa_pairs:
                # Encode the question
                q_vec = self._encode_question(qa.question, qa.axis)
                self.qa_pairs.append((q_vec, qa))
                total_pairs += 1
        
        return total_pairs
    
    def ingest_text(self, text: str) -> Tuple[int, int, int]:
        """
        Ingest text through all projection levels.
        
        Returns (sentences, triples, qa_pairs).
        """
        # Level 1: Extract sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        n_sentences = 0
        n_triples = 0
        n_pairs = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            if sentence.endswith('?'):
                continue
            
            n_sentences += 1
            
            # Level 2 & 3
            triples_before = len(self.triples)
            pairs = self.ingest_sentence(sentence)
            
            n_triples += len(self.triples) - triples_before
            n_pairs += pairs
        
        return n_sentences, n_triples, n_pairs
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[QAPair, float]]:
        """
        Query the Q&A space.
        
        The query is encoded and matched against stored question vectors.
        """
        axis = self._detect_axis(question)
        q_vec = self._encode_question(question, axis)
        
        results = []
        for stored_vec, qa in self.qa_pairs:
            # Cosine similarity
            sim = np.dot(q_vec, stored_vec)
            
            # Boost if axis matches
            if qa.axis == axis:
                sim *= 1.3
            
            # Apply confidence
            sim *= qa.confidence
            
            results.append((qa, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def answer(self, question: str) -> Tuple[str, float, str]:
        """Get the best answer."""
        results = self.query(question, top_k=1)
        if results:
            qa, score = results[0]
            return qa.answer, score, qa.axis
        return "I don't know.", 0.0, "NONE"


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("  RECURSIVE HOLOGRAPHIC Q&A SYSTEM")
    print("  Multi-level geometric projection")
    print("=" * 70)
    
    # Test sentences
    sentences = [
        "Captain Ahab is the monomaniacal captain of the Pequod.",
        "Moby Dick is a giant white sperm whale.",
        "The Pequod sailed from Nantucket in Massachusetts.",
        "Ahab hunts Moby Dick because the whale took his leg.",
        "Ishmael is the narrator of the story.",
        "Queequeg is a harpooner from the South Pacific.",
        "The Pequod was destroyed by Moby Dick.",
        "White Fang is a wolf-dog hybrid born in the wild.",
        "Grey Beaver is a Native American who owned White Fang.",
        "Sherlock Holmes is a famous detective in London.",
        "Dr. Watson is Holmes's loyal companion and biographer.",
        "Elizabeth Bennet is the protagonist of Pride and Prejudice.",
        "Mr. Darcy is a wealthy gentleman from Derbyshire.",
    ]
    
    qa = RecursiveHolographicQA(dim=64)
    
    print("\n[1] LEVEL 2: Extracting semantic triples...")
    print("-" * 70)
    
    for sentence in sentences:
        triples = extract_triples(sentence)
        if triples:
            t = triples[0]
            print(f"\n  \"{sentence}\"")
            print(f"    Subject:   {t.subject}")
            print(f"    Predicate: {t.predicate}")
            print(f"    Object:    {t.object}")
            if t.modifiers:
                print(f"    Modifiers: {t.modifiers}")
    
    print("\n\n[2] LEVEL 3: Generating Q&A pairs...")
    print("-" * 70)
    
    for sentence in sentences[:5]:  # Show first 5
        triples = extract_triples(sentence)
        if triples:
            t = triples[0]
            pairs = generate_qa_pairs(t)
            print(f"\n  From: \"{sentence}\"")
            for p in pairs:
                print(f"    [{p.axis}] Q: {p.question}")
                print(f"             A: {p.answer}")
    
    print("\n\n[3] Ingesting all sentences...")
    for sentence in sentences:
        qa.ingest_sentence(sentence)
    
    print(f"    Triples: {len(qa.triples)}")
    print(f"    Q&A pairs: {len(qa.qa_pairs)}")
    
    # Test queries
    print("\n" + "=" * 70)
    print("  QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "Who is Captain Ahab?",
        "What is Moby Dick?",
        "Where did the Pequod sail from?",
        "Why does Ahab hunt Moby Dick?",
        "Who is the narrator?",
        "Tell me about Queequeg",
        "What happened to the Pequod?",
        "Who is Sherlock Holmes?",
        "Where is Sherlock Holmes?",
        "Who is Elizabeth Bennet?",
        "Who is Mr. Darcy?",
    ]
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Q: {query}")
        
        answer, score, axis = qa.answer(query)
        print(f"   [Axis: {axis}, Score: {score:.3f}]")
        print(f"A: {answer}")


if __name__ == "__main__":
    demo()
