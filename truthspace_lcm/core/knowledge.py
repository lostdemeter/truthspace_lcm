"""
Knowledge Base for Geometric Chat System

Implements fact storage, triple extraction, and Q&A pair management as specified in the SDS.

Core operations:
- Semantic search via cosine similarity
- Gap-filling Q&A matching
- Triple extraction from sentences
"""

import numpy as np
import re
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .vocabulary import Vocabulary, tokenize, cosine_similarity


@dataclass
class Fact:
    """A stored fact with its encoding."""
    id: str
    content: str
    encoding: np.ndarray
    source: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Triple:
    """A semantic triple (subject, predicate, object)."""
    id: str
    subject: str
    predicate: str
    object: str
    modifiers: Dict[str, str] = field(default_factory=dict)
    encoding: np.ndarray = None
    source: str = ""


@dataclass
class QAPair:
    """A question-answer pair with encodings."""
    id: str
    question: str
    answer: str
    question_type: str  # WHO, WHAT, WHERE, WHEN, WHY, HOW
    question_encoding: np.ndarray = None
    answer_encoding: np.ndarray = None
    source: str = ""


# Question type patterns
QUESTION_PATTERNS = {
    'WHO': ['who is', 'who was', 'who are', 'who did', 'whose'],
    'WHAT': ['what is', 'what was', 'what are', 'what does', 'define'],
    'WHERE': ['where is', 'where was', 'where did', 'location of'],
    'WHEN': ['when did', 'when was', 'when is', 'what year', 'what time'],
    'WHY': ['why did', 'why does', 'why is', 'reason for', 'cause of'],
    'HOW': ['how did', 'how does', 'how do', 'how to', 'method of'],
}


def detect_question_type(question: str) -> str:
    """
    Detect question type from patterns.
    
    Returns: 'WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', or 'UNKNOWN'
    """
    question_lower = question.lower()
    
    for qtype, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in question_lower:
                return qtype
    
    return 'UNKNOWN'


def extract_triples(sentence: str) -> List[Triple]:
    """
    Extract semantic triples from a sentence.
    
    Simple pattern-based extraction.
    """
    triples = []
    
    # Pattern: Subject + is/was/are + Object
    pattern = r'^([A-Z][a-zA-Z\s]+?)\s+(is|was|are|were|has|had)\s+(.+?)(?:\.|$)'
    match = re.match(pattern, sentence.strip())
    
    if match:
        subject, predicate, rest = match.groups()
        
        # Extract location modifier
        loc_match = re.search(r'\b(in|at|on|near)\s+([^,\.]+)', rest)
        modifiers = {}
        obj = rest
        if loc_match:
            modifiers['location'] = loc_match.group(2).strip()
            obj = rest[:loc_match.start()].strip()
        
        triple_id = hashlib.md5(sentence.encode()).hexdigest()[:12]
        triples.append(Triple(
            id=triple_id,
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=obj.strip(),
            modifiers=modifiers
        ))
    
    return triples


def generate_qa_from_triple(triple: Triple) -> List[Tuple[str, str, str]]:
    """
    Generate Q&A pairs from a triple.
    
    Returns: [(question, answer, question_type), ...]
    """
    pairs = []
    
    # WHO question
    if triple.subject:
        q = f"Who {triple.predicate} {triple.object}?"
        a = f"{triple.subject} {triple.predicate} {triple.object}."
        pairs.append((q, a, 'WHO'))
    
    # WHAT question
    if triple.object:
        q = f"What did {triple.subject} {triple.predicate}?"
        a = f"{triple.subject} {triple.predicate} {triple.object}."
        pairs.append((q, a, 'WHAT'))
    
    # WHERE question
    if 'location' in triple.modifiers:
        loc = triple.modifiers['location']
        q = f"Where did {triple.subject} {triple.predicate}?"
        a = f"{triple.subject} {triple.predicate} in {loc}."
        pairs.append((q, a, 'WHERE'))
    
    return pairs


class KnowledgeBase:
    """
    Stores facts, triples, and Q&A pairs with their encodings.
    
    Provides semantic search and gap-filling Q&A matching.
    """
    
    def __init__(self, vocab: Vocabulary = None):
        self.vocab = vocab or Vocabulary()
        self.facts: Dict[str, Fact] = {}
        self.triples: Dict[str, Triple] = {}
        self.qa_pairs: Dict[str, QAPair] = {}
    
    def add_fact(self, content: str, source: str = "", 
                 metadata: Dict = None) -> Fact:
        """Add a fact to the knowledge base."""
        # Add to vocabulary
        self.vocab.add_text(content)
        
        fact_id = hashlib.md5(content.encode()).hexdigest()[:12]
        encoding = self.vocab.encode(content)
        
        fact = Fact(
            id=fact_id,
            content=content,
            encoding=encoding,
            source=source,
            metadata=metadata or {}
        )
        self.facts[fact_id] = fact
        return fact
    
    def add_triple(self, triple: Triple, source: str = "") -> Triple:
        """Add a triple to the knowledge base."""
        # Create combined text for encoding
        text = f"{triple.subject} {triple.predicate} {triple.object}"
        self.vocab.add_text(text)
        
        triple.encoding = self.vocab.encode(text)
        triple.source = source
        self.triples[triple.id] = triple
        return triple
    
    def add_qa_pair(self, question: str, answer: str, 
                    qtype: str = None, source: str = "") -> QAPair:
        """Add a Q&A pair to the knowledge base."""
        # Add to vocabulary
        self.vocab.add_text(question)
        self.vocab.add_text(answer)
        
        if qtype is None:
            qtype = detect_question_type(question)
        
        pair_id = hashlib.md5(f"{question}:{answer}".encode()).hexdigest()[:12]
        
        qa = QAPair(
            id=pair_id,
            question=question,
            answer=answer,
            question_type=qtype,
            question_encoding=self.vocab.encode(question),
            answer_encoding=self.vocab.encode(answer),
            source=source
        )
        self.qa_pairs[pair_id] = qa
        return qa
    
    def search_facts(self, query: str, k: int = 5) -> List[Tuple[Fact, float]]:
        """
        Find k most similar facts to query.
        
        Uses cosine similarity in semantic space.
        """
        query_vec = self.vocab.encode(query)
        
        results = []
        for fact in self.facts.values():
            sim = cosine_similarity(query_vec, fact.encoding)
            results.append((fact, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def search_qa(self, question: str, k: int = 5) -> List[Tuple[QAPair, float]]:
        """
        Find k most similar Q&A pairs by question similarity.
        
        This is gap-filling: find questions with similar gaps.
        """
        query_vec = self.vocab.encode(question)
        
        results = []
        for qa in self.qa_pairs.values():
            sim = cosine_similarity(query_vec, qa.question_encoding)
            results.append((qa, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def ingest_text(self, text: str, source: str = "") -> Dict[str, int]:
        """
        Ingest raw text into knowledge base.
        
        Pipeline:
        1. Split into sentences
        2. Add each as fact
        3. Extract triples
        4. Generate Q&A pairs
        
        Returns counts of items added.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        facts_added = 0
        triples_added = 0
        qa_added = 0
        
        for sentence in sentences:
            # Add as fact
            self.add_fact(sentence, source)
            facts_added += 1
            
            # Extract triples
            triples = extract_triples(sentence)
            for triple in triples:
                self.add_triple(triple, source)
                triples_added += 1
                
                # Generate Q&A pairs from triple
                qa_pairs = generate_qa_from_triple(triple)
                for q, a, qtype in qa_pairs:
                    self.add_qa_pair(q, a, qtype, source)
                    qa_added += 1
        
        return {
            'facts': facts_added,
            'triples': triples_added,
            'qa_pairs': qa_added
        }
    
    def ingest_qa_pairs(self, pairs: List[Tuple[str, str]], 
                        source: str = "") -> int:
        """
        Ingest structured Q&A pairs.
        
        Returns count of pairs added.
        """
        count = 0
        for question, answer in pairs:
            self.add_qa_pair(question, answer, source=source)
            count += 1
        return count
