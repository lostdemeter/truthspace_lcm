#!/usr/bin/env python3
"""
Holographic Q&A Projection System

The key insight: A statement S can be DECOMPOSED into multiple (Q, A) pairs
through geometric projection onto different question-type axes.

Instead of:
  1. Store statements
  2. Match queries to statements

We do:
  1. Project statements into Q&A space (generate implicit Q&A pairs)
  2. Store (question_vector, answer_text) pairs
  3. Match query vector to question vectors

This is the HOLOGRAPHIC aspect: a single statement contains multiple
"views" (Q&A pairs), just like a hologram contains multiple viewing angles.

The projection is GEOMETRIC:
  - Each question type defines an AXIS in semantic space
  - Projecting a statement onto an axis extracts the answer for that question type
  - The subject, object, and modifiers of a statement map to different axes
"""

import sys
import os
import re
import urllib.request
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import numpy as np

PHI = (1 + math.sqrt(5)) / 2


# ============================================================================
# QUESTION TYPE AXES - Each defines a projection direction
# ============================================================================

class QuestionAxis(NamedTuple):
    """A question type defines a projection axis in semantic space."""
    name: str
    question_templates: List[str]  # How to form questions
    answer_extractors: List[str]   # Syntactic patterns that provide answers
    axis_words: Set[str]           # Words that indicate this axis


QUESTION_AXES = {
    'IDENTITY': QuestionAxis(
        name='IDENTITY',
        question_templates=['Who is {}?', 'Who was {}?', 'Tell me about {}'],
        answer_extractors=['is', 'was', 'named', 'called'],
        axis_words={'who', 'person', 'character', 'man', 'woman', 'he', 'she'},
    ),
    'DEFINITION': QuestionAxis(
        name='DEFINITION',
        question_templates=['What is {}?', 'What was {}?', 'Define {}'],
        answer_extractors=['is', 'was', 'means', 'refers to'],
        axis_words={'what', 'thing', 'object', 'concept', 'it'},
    ),
    'LOCATION': QuestionAxis(
        name='LOCATION',
        question_templates=['Where is {}?', 'Where did {} happen?', 'Where was {}?'],
        answer_extractors=['in', 'at', 'from', 'near', 'to'],
        axis_words={'where', 'place', 'location', 'city', 'country'},
    ),
    'TIME': QuestionAxis(
        name='TIME',
        question_templates=['When did {}?', 'When was {}?', 'What time did {}?'],
        answer_extractors=['in', 'on', 'at', 'during', 'before', 'after', 'year'],
        axis_words={'when', 'time', 'date', 'year', 'day'},
    ),
    'REASON': QuestionAxis(
        name='REASON',
        question_templates=['Why did {}?', 'Why does {}?', 'What caused {}?'],
        answer_extractors=['because', 'since', 'due to', 'reason', 'cause'],
        axis_words={'why', 'reason', 'cause', 'because'},
    ),
    'METHOD': QuestionAxis(
        name='METHOD',
        question_templates=['How did {}?', 'How does {}?', 'How to {}?'],
        answer_extractors=['by', 'through', 'using', 'with'],
        axis_words={'how', 'method', 'way', 'process'},
    ),
    'ATTRIBUTE': QuestionAxis(
        name='ATTRIBUTE',
        question_templates=['What kind of {}?', 'What type of {}?', 'Describe {}'],
        answer_extractors=['is', 'was', 'very', 'quite', 'extremely'],
        axis_words={'kind', 'type', 'sort', 'like', 'such'},
    ),
}


# ============================================================================
# STATEMENT PARSER - Extract semantic components
# ============================================================================

class StatementComponents(NamedTuple):
    """Parsed components of a statement."""
    subject: str
    verb: str
    object: str
    modifiers: List[str]
    prepositional_phrases: List[Tuple[str, str]]  # (preposition, object)
    raw: str


def parse_statement(text: str) -> Optional[StatementComponents]:
    """
    Parse a statement into its semantic components.
    
    "Captain Ahab is the monomaniacal captain of the Pequod"
    → subject: "Captain Ahab"
    → verb: "is"
    → object: "the monomaniacal captain"
    → prepositional_phrases: [("of", "the Pequod")]
    """
    text = text.strip()
    if not text or text.endswith('?'):
        return None
    
    # Simple pattern: Subject + BE-verb + Object/Predicate
    be_verbs = ['is', 'was', 'are', 'were', 'has been', 'had been']
    
    for verb in be_verbs:
        pattern = rf'^(.+?)\s+({verb})\s+(.+)$'
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            verb_found = match.group(2).strip()
            predicate = match.group(3).strip()
            
            # Extract prepositional phrases from predicate
            prep_pattern = r'\b(of|in|at|on|from|to|with|by|for|about)\s+(.+?)(?=\s+(?:of|in|at|on|from|to|with|by|for|about)\b|[,.]|$)'
            prep_phrases = re.findall(prep_pattern, predicate, re.IGNORECASE)
            
            # Remove prepositional phrases from object
            obj = predicate
            for prep, prep_obj in prep_phrases:
                obj = re.sub(rf'\b{prep}\s+{re.escape(prep_obj)}\b', '', obj, flags=re.IGNORECASE)
            obj = obj.strip(' ,.')
            
            # Extract modifiers (adjectives before nouns)
            modifier_pattern = r'\b(very|quite|extremely|most|more|less|rather|somewhat)\b'
            modifiers = re.findall(modifier_pattern, obj, re.IGNORECASE)
            
            return StatementComponents(
                subject=subject,
                verb=verb_found,
                object=obj,
                modifiers=modifiers,
                prepositional_phrases=prep_phrases,
                raw=text
            )
    
    # Try action verb pattern: Subject + Verb + Object
    action_pattern = r'^(.+?)\s+(killed|destroyed|created|found|discovered|became|went|came|said|told|gave|took|made|had|did)\s+(.+)$'
    match = re.match(action_pattern, text, re.IGNORECASE)
    if match:
        return StatementComponents(
            subject=match.group(1).strip(),
            verb=match.group(2).strip(),
            object=match.group(3).strip(),
            modifiers=[],
            prepositional_phrases=[],
            raw=text
        )
    
    return None


# ============================================================================
# Q&A PROJECTION - Generate implicit Q&A pairs from statements
# ============================================================================

class QAPair(NamedTuple):
    """A question-answer pair with its projection axis."""
    question: str
    answer: str
    axis: str
    subject: str
    confidence: float


def project_to_qa(components: StatementComponents) -> List[QAPair]:
    """
    Project a parsed statement onto Q&A axes.
    
    Each axis extracts a different "view" of the statement.
    """
    pairs = []
    
    subject = components.subject
    verb = components.verb
    obj = components.object
    raw = components.raw
    
    # IDENTITY projection: "Who/What is [subject]?"
    if subject and obj:
        # Who is X? → predicate
        q = f"Who is {subject}?"
        a = obj
        pairs.append(QAPair(q, a, 'IDENTITY', subject, 0.9))
        
        # What is X? → predicate
        q = f"What is {subject}?"
        a = obj
        pairs.append(QAPair(q, a, 'DEFINITION', subject, 0.8))
    
    # LOCATION projection: extract from prepositional phrases
    for prep, prep_obj in components.prepositional_phrases:
        if prep.lower() in ['in', 'at', 'from', 'near', 'to']:
            q = f"Where is {subject}?"
            a = f"{prep} {prep_obj}"
            pairs.append(QAPair(q, a, 'LOCATION', subject, 0.7))
            
            # Also: "Where is [prep_obj]?" might be answered by subject
            q = f"What is in {prep_obj}?"
            a = subject
            pairs.append(QAPair(q, a, 'DEFINITION', prep_obj, 0.6))
    
    # ATTRIBUTE projection: if object contains adjectives
    adjectives = re.findall(r'\b(\w+(?:ous|ive|ful|less|able|ible|al|ic|ed|ing))\b', obj, re.IGNORECASE)
    if adjectives:
        q = f"What kind of {subject}?"
        a = ', '.join(adjectives)
        pairs.append(QAPair(q, a, 'ATTRIBUTE', subject, 0.6))
    
    # REASON projection: if statement contains causal words
    if re.search(r'\b(because|since|due to|reason|cause)\b', raw, re.IGNORECASE):
        # Extract the reason part
        reason_match = re.search(r'\b(?:because|since|due to)\s+(.+?)(?:[,.]|$)', raw, re.IGNORECASE)
        if reason_match:
            q = f"Why did {subject} {verb}?"
            a = reason_match.group(1)
            pairs.append(QAPair(q, a, 'REASON', subject, 0.8))
    
    # Full statement as answer to general question
    q = f"Tell me about {subject}"
    a = raw
    pairs.append(QAPair(q, a, 'IDENTITY', subject, 0.5))
    
    return pairs


# ============================================================================
# GEOMETRIC Q&A SPACE
# ============================================================================

class HolographicQASpace:
    """
    A geometric space where Q&A pairs live.
    
    Each question type defines an axis.
    Statements are projected onto these axes to generate Q&A pairs.
    Queries are matched by finding the nearest question vector.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.qa_pairs: List[Tuple[np.ndarray, QAPair]] = []  # (question_vector, qa_pair)
        self.word_positions: Dict[str, np.ndarray] = {}
        self.axis_vectors: Dict[str, np.ndarray] = {}
        
        self._init_axes()
    
    def _init_axes(self):
        """Initialize axis vectors for each question type."""
        for axis_name in QUESTION_AXES.keys():
            np.random.seed(hash(axis_name) % (2**32))
            vec = np.random.randn(self.dim)
            vec = vec / np.linalg.norm(vec)
            self.axis_vectors[axis_name] = vec
            np.random.seed(None)
    
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
        """Encode text as average of word positions."""
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.dim)
        
        pos = np.zeros(self.dim)
        for word in words:
            pos += self._get_word_position(word)
        return pos / len(words)
    
    def _encode_question(self, question: str, axis: str) -> np.ndarray:
        """
        Encode a question as: content_vector + axis_vector
        
        This places questions of the same type near each other,
        while still distinguishing by content.
        """
        content = self._encode_text(question)
        axis_vec = self.axis_vectors.get(axis, np.zeros(self.dim))
        
        # Combine: content determines position, axis determines direction
        combined = 0.6 * content + 0.4 * axis_vec
        
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm
        
        return combined
    
    def ingest_statement(self, text: str) -> int:
        """
        Ingest a statement by projecting it into Q&A space.
        
        Returns number of Q&A pairs generated.
        """
        components = parse_statement(text)
        if components is None:
            return 0
        
        qa_pairs = project_to_qa(components)
        
        for qa in qa_pairs:
            q_vec = self._encode_question(qa.question, qa.axis)
            self.qa_pairs.append((q_vec, qa))
        
        return len(qa_pairs)
    
    def ingest_text(self, text: str) -> Tuple[int, int]:
        """
        Ingest a full text by extracting statements and projecting to Q&A.
        
        Returns (num_statements, num_qa_pairs).
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        total_statements = 0
        total_pairs = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            if sentence.endswith('?'):
                continue
            
            n = self.ingest_statement(sentence)
            if n > 0:
                total_statements += 1
                total_pairs += n
        
        return total_statements, total_pairs
    
    def _detect_query_axis(self, query: str) -> str:
        """Detect which axis a query belongs to."""
        query_lower = query.lower()
        
        if query_lower.startswith('who '):
            return 'IDENTITY'
        if query_lower.startswith('what is') or query_lower.startswith('what was'):
            return 'DEFINITION'
        if query_lower.startswith('where '):
            return 'LOCATION'
        if query_lower.startswith('when '):
            return 'TIME'
        if query_lower.startswith('why '):
            return 'REASON'
        if query_lower.startswith('how '):
            return 'METHOD'
        if 'kind of' in query_lower or 'type of' in query_lower:
            return 'ATTRIBUTE'
        if query_lower.startswith('tell me about'):
            return 'IDENTITY'
        
        return 'DEFINITION'  # Default
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[QAPair, float]]:
        """
        Query the Q&A space.
        
        Returns list of (qa_pair, similarity_score).
        """
        axis = self._detect_query_axis(question)
        q_vec = self._encode_question(question, axis)
        
        results = []
        for stored_vec, qa in self.qa_pairs:
            # Cosine similarity
            sim = np.dot(q_vec, stored_vec)
            
            # Boost if axis matches
            if qa.axis == axis:
                sim *= 1.2
            
            # Boost by confidence
            sim *= qa.confidence
            
            results.append((qa, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def answer(self, question: str) -> Tuple[str, float, str]:
        """Get the best answer to a question."""
        results = self.query(question, top_k=1)
        if results:
            qa, score = results[0]
            return qa.answer, score, qa.axis
        return "I don't know.", 0.0, "NONE"


# ============================================================================
# DEMO
# ============================================================================

def demo_projection():
    """Demonstrate the holographic projection."""
    print("=" * 70)
    print("  HOLOGRAPHIC Q&A PROJECTION DEMO")
    print("  Projecting statements into Q&A space")
    print("=" * 70)
    
    # Test statements
    statements = [
        "Captain Ahab is the monomaniacal captain of the Pequod.",
        "Moby Dick is a giant white sperm whale that bit off Ahab's leg.",
        "The Pequod sailed from Nantucket in Massachusetts.",
        "Ahab hunts Moby Dick because the whale took his leg.",
        "Ishmael is the narrator of the story.",
        "Queequeg is a harpooner from the South Pacific.",
        "The Pequod was destroyed by Moby Dick at the end.",
        "White Fang is a wolf-dog hybrid born in the wild.",
        "Grey Beaver is a Native American who owned White Fang.",
        "Sherlock Holmes is a famous detective who lives at 221B Baker Street.",
    ]
    
    print("\n[1] Projecting statements into Q&A space...")
    print("-" * 70)
    
    space = HolographicQASpace(dim=64)
    
    for stmt in statements:
        components = parse_statement(stmt)
        if components:
            print(f"\nStatement: {stmt}")
            print(f"  Subject: {components.subject}")
            print(f"  Verb: {components.verb}")
            print(f"  Object: {components.object}")
            print(f"  Prep phrases: {components.prepositional_phrases}")
            
            qa_pairs = project_to_qa(components)
            print(f"  Generated {len(qa_pairs)} Q&A pairs:")
            for qa in qa_pairs[:3]:  # Show first 3
                print(f"    [{qa.axis}] Q: {qa.question}")
                print(f"              A: {qa.answer}")
        
        space.ingest_statement(stmt)
    
    print(f"\n\nTotal Q&A pairs in space: {len(space.qa_pairs)}")
    
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
        "Who is White Fang?",
    ]
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Q: {query}")
        
        answer, score, axis = space.answer(query)
        print(f"   [Axis: {axis}, Score: {score:.3f}]")
        print(f"A: {answer}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Holographic Q&A Projection')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--file', type=str, help='Ingest a text file')
    
    args = parser.parse_args()
    
    if args.demo or not args.file:
        demo_projection()
        return
    
    # Interactive mode with file
    print("=" * 70)
    print("  HOLOGRAPHIC Q&A PROJECTION")
    print("=" * 70)
    
    space = HolographicQASpace(dim=64)
    
    print(f"\n[1] Loading {args.file}...")
    with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    print(f"[2] Projecting into Q&A space...")
    n_stmt, n_qa = space.ingest_text(text)
    print(f"    Processed {n_stmt} statements → {n_qa} Q&A pairs")
    
    print("\n" + "=" * 70)
    print("  INTERACTIVE MODE")
    print("  Type 'quit' to exit")
    print("=" * 70)
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        answer, score, axis = space.answer(query)
        print(f"   [Axis: {axis}, Score: {score:.3f}]")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
