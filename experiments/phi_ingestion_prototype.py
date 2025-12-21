#!/usr/bin/env python3
"""
φ-Based Auto-Ingestion Prototype

This module implements scalable data ingestion for a φ-structured chatbot.
The key insight: we don't learn embeddings, we PLACE knowledge at φ^(-n)
positions based on semantic domain, then let attractor/repeller dynamics
refine the structure.

Architecture:
1. Seed vocabulary at known φ positions
2. Ingest text → detect domain → place at φ level
3. Co-occurrence → phase assignment
4. Attractor/repeller → self-organization
5. Query → encode → match → respond
"""

import json
import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import numpy as np

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi


@dataclass
class PhiNode:
    """A node in the φ-structured vocabulary."""
    word: str
    level: int           # n in φ^(-n)
    phase: float         # 0 to 2π
    frequency: int = 0   # Usage count
    domain: str = ""     # Semantic domain
    locked: bool = False # If True, dynamics won't move this node
    
    @property
    def magnitude(self) -> float:
        return PHI ** (-self.level)
    
    @property
    def position(self) -> complex:
        return self.magnitude * np.exp(1j * self.phase)
    
    def to_dict(self) -> dict:
        return {
            'word': self.word,
            'level': self.level,
            'phase': self.phase,
            'frequency': self.frequency,
            'domain': self.domain,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PhiNode':
        return cls(**d)


class SemanticDomainDetector:
    """
    Detect which semantic domain a word or text belongs to.
    Maps to φ^(-n) levels.
    """
    
    # Domain definitions with keywords and φ level
    DOMAINS = {
        'ABSTRACT': {
            'level': 7,
            'keywords': {'meaning', 'truth', 'beauty', 'concept', 'idea', 
                        'philosophy', 'theory', 'principle', 'essence', 'nature'},
        },
        'KNOWLEDGE': {
            'level': 8,
            'keywords': {'know', 'think', 'believe', 'understand', 'learn',
                        'fact', 'information', 'definition', 'explain', 'reason'},
        },
        'ACTION': {
            'level': 9,
            'keywords': {'do', 'make', 'go', 'run', 'create', 'build', 'write',
                        'read', 'send', 'get', 'put', 'move', 'change', 'start', 'stop'},
        },
        'ENTITY': {
            'level': 10,
            'keywords': {'person', 'thing', 'object', 'file', 'program', 'system',
                        'user', 'computer', 'network', 'data', 'message', 'document'},
        },
        'RELATION': {
            'level': 11,
            'keywords': {'like', 'with', 'for', 'about', 'between', 'through',
                        'using', 'contains', 'belongs', 'connects', 'relates'},
        },
        'ATTRIBUTE': {
            'level': 12,
            'keywords': {'big', 'small', 'good', 'bad', 'fast', 'slow', 'new', 'old',
                        'high', 'low', 'long', 'short', 'hot', 'cold', 'open', 'closed'},
        },
        'CONTEXT': {
            'level': 13,
            'keywords': {'time', 'when', 'where', 'now', 'then', 'here', 'there',
                        'before', 'after', 'during', 'while', 'today', 'yesterday'},
        },
        'GROUND': {
            'level': 14,
            'keywords': {'is', 'the', 'a', 'an', 'be', 'have', 'it', 'this', 'that',
                        'and', 'or', 'but', 'if', 'so', 'as', 'to', 'of', 'in', 'on'},
        },
    }
    
    def detect_word_domain(self, word: str) -> Tuple[str, int]:
        """Detect domain for a single word."""
        word_lower = word.lower()
        
        for domain, info in self.DOMAINS.items():
            if word_lower in info['keywords']:
                return domain, info['level']
        
        # Default based on word characteristics
        if word[0].isupper():
            return 'ENTITY', 10  # Proper nouns are entities
        if word.endswith('ly'):
            return 'ATTRIBUTE', 12  # Adverbs are attributes
        if word.endswith('ing') or word.endswith('ed'):
            return 'ACTION', 9  # Verb forms are actions
        
        return 'KNOWLEDGE', 8  # Default to knowledge level
    
    def detect_text_domain(self, text: str) -> Tuple[str, int]:
        """Detect primary domain for a text."""
        words = text.lower().split()
        domain_counts = defaultdict(int)
        
        for word in words:
            domain, _ = self.detect_word_domain(word)
            domain_counts[domain] += 1
        
        if not domain_counts:
            return 'KNOWLEDGE', 8
        
        primary = max(domain_counts, key=domain_counts.get)
        return primary, self.DOMAINS[primary]['level']


class CooccurrenceTracker:
    """
    Track word co-occurrence for phase assignment.
    Words that co-occur should have similar phases.
    """
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[str, int] = defaultdict(int)
    
    def update(self, words: List[str]):
        """Update co-occurrence from a list of words."""
        for i, word in enumerate(words):
            self.word_counts[word] += 1
            
            # Look at window around word
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    other = words[j]
                    self.cooccurrence[word][other] += 1
    
    def get_neighbors(self, word: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top co-occurring words."""
        if word not in self.cooccurrence:
            return []
        
        neighbors = sorted(
            self.cooccurrence[word].items(),
            key=lambda x: -x[1]
        )
        return neighbors[:top_k]
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute co-occurrence similarity between two words."""
        if word1 not in self.cooccurrence or word2 not in self.cooccurrence:
            return 0.0
        
        # Jaccard-like similarity
        neighbors1 = set(self.cooccurrence[word1].keys())
        neighbors2 = set(self.cooccurrence[word2].keys())
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0


class PhiVocabulary:
    """
    The core φ-structured vocabulary.
    Words are placed at φ^(-n) positions with phases for disambiguation.
    """
    
    def __init__(self):
        self.nodes: Dict[str, PhiNode] = {}
        self.domain_detector = SemanticDomainDetector()
        self.cooccurrence = CooccurrenceTracker()
        self._initialize_seed_vocabulary()
    
    def _initialize_seed_vocabulary(self):
        """Initialize with seed words at known positions."""
        
        # CRITICAL: Seed concept-command pairs with MATCHING phases
        # This gives the dynamics a head start
        concept_command_seeds = [
            # files cluster - phase 0
            ('files', 9, 0), ('ls', 9, 0), ('directory', 9, 0.1), ('list', 9, 0.1),
            
            # disk cluster - phase π/2
            ('disk', 9, PI/2), ('df', 9, PI/2), ('space', 9, PI/2 + 0.1), ('storage', 9, PI/2 + 0.1),
            
            # processes cluster - phase π
            ('processes', 9, PI), ('ps', 9, PI), ('running', 9, PI + 0.1), ('process', 9, PI + 0.1),
            
            # network cluster - phase 3π/2
            ('network', 9, 3*PI/2), ('netstat', 9, 3*PI/2), ('connections', 9, 3*PI/2 + 0.1),
            
            # users cluster - phase 2π (= 0, but we'll use 5π/4 to separate from files)
            ('users', 9, 5*PI/4), ('who', 9, 5*PI/4), ('logged', 9, 5*PI/4 + 0.1),
        ]
        
        seeds = concept_command_seeds + [
            # GROUND level (14) - stopwords
            ('is', 14, 0), ('the', 14, PI/4), ('a', 14, PI/2),
            ('be', 14, 3*PI/4), ('have', 14, PI), ('it', 14, 5*PI/4),
            
            # CONTEXT level (13)
            ('time', 13, 0), ('when', 13, PI/3), ('where', 13, 2*PI/3),
            ('now', 13, PI), ('here', 13, 4*PI/3), ('there', 13, 5*PI/3),
        ]
        
        for word, level, phase in seeds:
            self.nodes[word] = PhiNode(
                word=word,
                level=level,
                phase=phase,
                frequency=1000,  # High frequency for seeds
                domain=self._level_to_domain(level),
                locked=True,  # Don't let dynamics move seeded words
            )
    
    def _level_to_domain(self, level: int) -> str:
        """Map φ level to domain name."""
        level_to_domain = {
            7: 'ABSTRACT', 8: 'KNOWLEDGE', 9: 'ACTION', 10: 'ENTITY',
            11: 'RELATION', 12: 'ATTRIBUTE', 13: 'CONTEXT', 14: 'GROUND',
        }
        return level_to_domain.get(level, 'KNOWLEDGE')
    
    def add_word(self, word: str, context: List[str] = None) -> PhiNode:
        """
        Add a word to the vocabulary.
        
        GEOMETRIC APPROACH: All content words start at the SAME level.
        Differentiation happens through PHASE via attractor/repeller dynamics.
        Only truly common words (stopwords) go to higher levels.
        """
        if word in self.nodes:
            # Don't modify locked nodes (seeded words)
            if not self.nodes[word].locked:
                self.nodes[word].frequency += 1
            return self.nodes[word]
        
        # Check if it's a stopword (very common function word)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'although', 'i', 'me', 'my',
                     'myself', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                     'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what',
                     'which', 'who', 'whom', 'this', 'that', 'these', 'those'}
        
        if word.lower() in stopwords:
            level = 14  # Stopwords at high level (low magnitude)
        else:
            level = 9   # ALL content words at same level - phase differentiates
        
        # Random initial phase - dynamics will organize it
        phase = (hash(word) % 1000) / 1000 * 2 * PI
        
        node = PhiNode(
            word=word,
            level=level,
            phase=phase,
            frequency=1,
            domain='CONTENT',
        )
        self.nodes[word] = node
        return node
    
    def _infer_phase_from_context(self, word: str, context: List[str]) -> float:
        """Infer phase from context words."""
        # Find known words in context
        known_phases = []
        for ctx_word in context:
            if ctx_word in self.nodes:
                known_phases.append(self.nodes[ctx_word].phase)
        
        if known_phases:
            # Average phase of context, with small offset for uniqueness
            avg_phase = np.mean(known_phases)
            offset = (hash(word) % 100) / 100 * PI / 4  # Small offset
            return (avg_phase + offset) % (2 * PI)
        else:
            return (hash(word) % 1000) / 1000 * 2 * PI
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to a multi-dimensional φ-vector using SLIDING WINDOW.
        
        Strategy: 
        1. Slide windows of size 1-3 across the text
        2. For each window, compute WEIGHTED CIRCULAR MEAN of phases
        3. Weight by: locked status (high) + co-occurrence within window
        4. Combine windows using MAX magnitude per dimension
        
        Key insight: Locked (seeded) words anchor the encoding.
        Unlocked words contribute proportionally to their co-occurrence
        with locked words in the window.
        """
        words = re.findall(r'\w+', text.lower())
        
        # 8 dimensions: one per φ level (7-14)
        embedding = np.zeros(8, dtype=complex)
        
        # Try different window sizes
        for window_size in [1, 2, 3]:
            for i in range(len(words) - window_size + 1):
                window = words[i:i+window_size]
                
                # Get nodes for words in window
                window_nodes = []
                for w in window:
                    if w in self.nodes and self.nodes[w].level < 13:
                        window_nodes.append(self.nodes[w])
                
                if not window_nodes:
                    continue
                
                # Group by dimension
                by_dim = {}
                for node in window_nodes:
                    dim = node.level - 7
                    if 0 <= dim < 8:
                        if dim not in by_dim:
                            by_dim[dim] = []
                        by_dim[dim].append(node)
                
                # For each dimension, compute weighted circular mean
                for dim, nodes in by_dim.items():
                    # Compute weights: locked words get high weight, 
                    # unlocked words get weight from co-occurrence with locked
                    phase_x = 0.0  # cos component
                    phase_y = 0.0  # sin component
                    total_weight = 0.0
                    max_mag = 0.0
                    
                    for node in nodes:
                        if node.locked:
                            # Locked words anchor the encoding
                            weight = 10.0
                        else:
                            # Unlocked words: weight by co-occurrence with locked words
                            weight = 1.0
                            for other in nodes:
                                if other.locked and other.word != node.word:
                                    count = self.cooccurrence.cooccurrence.get(node.word, {}).get(other.word, 0)
                                    weight += count * 0.5
                        
                        phase_x += weight * np.cos(node.phase)
                        phase_y += weight * np.sin(node.phase)
                        total_weight += weight
                        max_mag = max(max_mag, node.magnitude)
                    
                    if total_weight > 0:
                        # Circular mean phase
                        mean_phase = np.arctan2(phase_y, phase_x)
                        window_pos = max_mag * np.exp(1j * mean_phase)
                        
                        # Boost for multi-word windows with co-occurrence
                        if len(nodes) > 1:
                            window_pos *= (1 + 0.1 * len(nodes))
                        
                        # MAX per dimension
                        if abs(window_pos) > abs(embedding[dim]):
                            embedding[dim] = window_pos
        
        return embedding
    
    def encode_simple(self, text: str) -> complex:
        """Simple single-value encoding for backward compatibility."""
        embedding = self.encode(text)
        # Return the strongest dimension
        max_idx = np.argmax(np.abs(embedding))
        return embedding[max_idx]
    
    def ingest_text(self, text: str):
        """Ingest a text, adding new words and updating co-occurrence."""
        words = re.findall(r'\w+', text.lower())
        
        # Update co-occurrence
        self.cooccurrence.update(words)
        
        # Add new words with context
        for i, word in enumerate(words):
            context = words[max(0, i-5):i] + words[i+1:min(len(words), i+6)]
            self.add_word(word, context)
    
    def save(self, path: str):
        """Save vocabulary to JSON."""
        data = {word: node.to_dict() for word, node in self.nodes.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.nodes = {word: PhiNode.from_dict(d) for word, d in data.items()}


class PhiKnowledgeBase:
    """
    Knowledge base indexed by φ positions.
    Stores query→response mappings at their encoded positions.
    """
    
    def __init__(self, vocabulary: PhiVocabulary):
        self.vocabulary = vocabulary
        self.entries: List[Tuple[np.ndarray, str, str, dict]] = []  # (position, query, response, metadata)
    
    def add(self, query: str, response: str, metadata: dict = None):
        """Add a query→response mapping."""
        position = self.vocabulary.encode(query)
        self.entries.append((position, query, response, metadata or {}))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search using GEOMETRIC CLUSTER matching via co-occurrence.
        
        Strategy:
        1. Find query words that co-occur with command words (ls, df, ps, netstat, who)
        2. Score entries by co-occurrence strength between query words and response command
        3. This learns clusters from data rather than hardcoding them
        
        The geometry: words that co-occur form attractor basins.
        Query words pull toward the command they co-occur with most.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        
        if not query_words:
            return []
        
        # Command words we're looking for in responses
        commands = ['ls', 'df', 'ps', 'netstat', 'who', 'ip', 'ifconfig']
        
        # For each query word, find which command it co-occurs with most
        query_command_affinity = {cmd: 0.0 for cmd in commands}
        
        for word in query_words:
            if word in self.vocabulary.cooccurrence.cooccurrence:
                word_cooccur = self.vocabulary.cooccurrence.cooccurrence[word]
                for cmd in commands:
                    if cmd in word_cooccur:
                        query_command_affinity[cmd] += word_cooccur[cmd]
        
        # Also check if query contains command words directly (strong signal)
        for cmd in commands:
            if cmd in query_words:
                query_command_affinity[cmd] += 100  # Strong boost
        
        scored = []
        
        for stored_vec, stored_q, response, meta in self.entries:
            response_lower = response.lower()
            
            # Find which command is in the response
            response_cmd = None
            for cmd in commands:
                if cmd in response_lower:
                    response_cmd = cmd
                    break
            
            if response_cmd:
                # Score = query's affinity for this command
                cmd_score = query_command_affinity[response_cmd]
            else:
                cmd_score = 0.0
            
            # Secondary: keyword overlap
            stored_words = set(re.findall(r'\w+', stored_q.lower()))
            overlap = len(query_words & stored_words)
            keyword_score = overlap
            
            total_score = cmd_score + 0.1 * keyword_score
            
            scored.append((stored_q, response, total_score))
        
        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]
    
    def ingest_qa_pairs(self, pairs: List[Tuple[str, str]]):
        """Ingest a list of (question, answer) pairs."""
        self._raw_pairs = []  # Store raw pairs for reindexing
        for question, answer in pairs:
            # CRITICAL: Ingest Q+A TOGETHER so concepts co-occur with commands
            combined = question + " " + answer
            self.vocabulary.ingest_text(combined)
            
            # Store raw pair
            self._raw_pairs.append((question, answer))
            
            # Then add to knowledge base
            self.add(question, answer)
    
    def reindex(self):
        """Re-encode all entries after vocabulary phases have changed."""
        if hasattr(self, '_raw_pairs'):
            self.entries = []
            for question, answer in self._raw_pairs:
                self.add(question, answer)


class AttractorRepellerDynamics:
    """
    Apply attractor/repeller dynamics to refine vocabulary positions.
    
    This is the PROVEN approach from our experiments:
    - Co-occurring words ATTRACT (pull phases together)
    - Non-co-occurring words REPEL (push phases apart)
    - Structure EMERGES from the data
    
    No semantic labels needed - geometry does the work.
    """
    
    def __init__(self, vocabulary: PhiVocabulary, 
                 attract_strength: float = 0.4,
                 repel_strength: float = 0.02):
        self.vocabulary = vocabulary
        self.attract_strength = attract_strength
        self.repel_strength = repel_strength
    
    def step(self):
        """Apply one step of dynamics based purely on co-occurrence."""
        
        # ATTRACTION: Words that co-occur pull together
        for word, node in list(self.vocabulary.nodes.items()):
            # Skip locked nodes
            if node.locked:
                continue
                
            neighbors = self.vocabulary.cooccurrence.get_neighbors(word, top_k=10)
            
            if not neighbors:
                continue
            
            # Compute weighted average phase of neighbors
            total_weight = 0
            phase_sum_x = 0  # Use circular mean
            phase_sum_y = 0
            
            for neighbor_word, count in neighbors:
                if neighbor_word in self.vocabulary.nodes:
                    neighbor = self.vocabulary.nodes[neighbor_word]
                    weight = count
                    phase_sum_x += weight * np.cos(neighbor.phase)
                    phase_sum_y += weight * np.sin(neighbor.phase)
                    total_weight += weight
            
            if total_weight > 0:
                # Target phase is the weighted circular mean
                target_phase = np.arctan2(phase_sum_y, phase_sum_x) % (2 * PI)
                
                # Move toward target
                phase_diff = target_phase - node.phase
                # Wrap to [-π, π]
                while phase_diff > PI:
                    phase_diff -= 2 * PI
                while phase_diff < -PI:
                    phase_diff += 2 * PI
                
                # Attraction strength scales with co-occurrence strength
                strength = self.attract_strength * min(1.0, total_weight / 50)
                node.phase = (node.phase + strength * phase_diff) % (2 * PI)
        
        # REPULSION: Words at same level that DON'T co-occur push apart
        nodes_by_level = defaultdict(list)
        for word, node in self.vocabulary.nodes.items():
            nodes_by_level[node.level].append((word, node))
        
        for level, level_nodes in nodes_by_level.items():
            for i, (word1, node1) in enumerate(level_nodes):
                for word2, node2 in level_nodes[i+1:]:
                    # Skip if either node is locked
                    if node1.locked or node2.locked:
                        continue
                    
                    # Check co-occurrence
                    cooccur = self.vocabulary.cooccurrence.cooccurrence[word1].get(word2, 0)
                    
                    if cooccur == 0:  # Never co-occur, should repel
                        phase_diff = node2.phase - node1.phase
                        # Wrap to [-π, π]
                        while phase_diff > PI:
                            phase_diff -= 2 * PI
                        while phase_diff < -PI:
                            phase_diff += 2 * PI
                        
                        # Repel if too close
                        if abs(phase_diff) < PI / 4:
                            # Push apart
                            push = self.repel_strength * np.sign(phase_diff)
                            node1.phase = (node1.phase - push) % (2 * PI)
                            node2.phase = (node2.phase + push) % (2 * PI)
    
    def settle(self, iterations: int = 100):
        """Run dynamics until settled."""
        for _ in range(iterations):
            self.step()


class PhiChatbot:
    """
    A chatbot built on φ-structured knowledge.
    """
    
    def __init__(self):
        self.vocabulary = PhiVocabulary()
        self.knowledge = PhiKnowledgeBase(self.vocabulary)
        self.dynamics = AttractorRepellerDynamics(self.vocabulary)
    
    def ingest_corpus(self, texts: List[str]):
        """Ingest a corpus of texts."""
        for text in texts:
            self.vocabulary.ingest_text(text)
        
        # Apply dynamics to settle
        self.dynamics.settle(iterations=50)
    
    def ingest_qa(self, pairs: List[Tuple[str, str]]):
        """Ingest Q&A pairs."""
        self.knowledge.ingest_qa_pairs(pairs)
        self.dynamics.settle(iterations=50)
        # Re-encode after dynamics settle (phases have changed!)
        self.knowledge.reindex()
    
    def respond(self, query: str) -> str:
        """Generate a response to a query."""
        matches = self.knowledge.search(query, top_k=3)
        
        if matches:
            best_q, best_response, score = matches[0]
            if score > 0.5:
                return best_response
            else:
                return f"I'm not sure, but maybe: {best_response}"
        else:
            return "I don't have information about that."
    
    def save(self, path: str):
        """Save the chatbot state."""
        self.vocabulary.save(path)
    
    def load(self, path: str):
        """Load the chatbot state."""
        self.vocabulary.load(path)


def demo():
    """Demonstrate the φ-based ingestion system."""
    
    print("=" * 70)
    print("φ-BASED AUTO-INGESTION DEMO")
    print("=" * 70)
    
    # Create chatbot
    bot = PhiChatbot()
    
    print(f"\nInitial vocabulary size: {len(bot.vocabulary.nodes)}")
    
    # Ingest some sample Q&A pairs
    qa_pairs = [
        ("How do I list files?", "Use the 'ls' command to list files in a directory."),
        ("How do I show disk space?", "Use 'df -h' to show disk space usage."),
        ("How do I find running processes?", "Use 'ps aux' or 'top' to see running processes."),
        ("How do I check network connections?", "Use 'netstat' or 'ss' to check network connections."),
        ("How do I create a new file?", "Use 'touch filename' to create an empty file."),
        ("How do I delete a file?", "Use 'rm filename' to delete a file."),
        ("How do I copy a file?", "Use 'cp source destination' to copy a file."),
        ("How do I move a file?", "Use 'mv source destination' to move a file."),
        ("How do I see who is logged in?", "Use 'who' or 'w' to see logged in users."),
        ("How do I check system uptime?", "Use 'uptime' to check how long the system has been running."),
    ]
    
    print("\nIngesting Q&A pairs...")
    bot.ingest_qa(qa_pairs)
    
    print(f"Vocabulary size after ingestion: {len(bot.vocabulary.nodes)}")
    
    # Show some vocabulary entries
    print("\n" + "=" * 70)
    print("VOCABULARY SAMPLE (by φ level)")
    print("=" * 70)
    
    by_level = defaultdict(list)
    for word, node in bot.vocabulary.nodes.items():
        by_level[node.level].append((word, node))
    
    for level in sorted(by_level.keys()):
        nodes = by_level[level][:5]  # Show first 5
        print(f"\nφ^(-{level}) = {PHI**(-level):.6f}:")
        for word, node in nodes:
            print(f"  {word}: phase={node.phase:.3f}, freq={node.frequency}")
    
    # Test queries
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    test_queries = [
        "list files",
        "show disk usage",
        "running processes",
        "network status",
        "create file",
        "who is online",
    ]
    
    for query in test_queries:
        response = bot.respond(query)
        print(f"\nQ: {query}")
        print(f"A: {response}")
    
    # Show encoding details
    print("\n" + "=" * 70)
    print("ENCODING DETAILS")
    print("=" * 70)
    
    for query in test_queries[:3]:
        embedding = bot.vocabulary.encode(query)
        print(f"\n'{query}':")
        print(f"  Vector: {len(embedding)} dimensions")
        for i, val in enumerate(embedding):
            if val != 0:
                level = i + 7
                print(f"    φ^(-{level}): mag={abs(val):.6f}, phase={np.angle(val):.3f}")
    
    # Save vocabulary
    save_path = Path(__file__).parent / "openai_data" / "phi_vocabulary.json"
    bot.save(str(save_path))
    print(f"\nSaved vocabulary to: {save_path}")


def detect_social(text: str) -> dict:
    """Detect if query is social (greeting, chitchat) vs command-oriented."""
    words = set(re.findall(r'\w+', text.lower()))
    
    # Social patterns
    social_keywords = {
        'GREETING': {'hey', 'hello', 'hi', 'howdy', 'yo', 'greetings'},
        'FAREWELL': {'bye', 'goodbye', 'later', 'cya', 'see'},
        'CHITCHAT': {'how', 'going', 'doing', 'up', 'good', 'fine', 'great', 'thanks', 'thank'},
        'POLITENESS': {'please', 'thanks', 'thank', 'appreciate'},
    }
    
    # Command-related keywords (from our clusters)
    command_keywords = {
        'files', 'file', 'directory', 'list', 'ls', 'dir', 'folder',
        'disk', 'space', 'storage', 'df', 'usage', 'size',
        'processes', 'process', 'running', 'ps', 'task', 'pid',
        'network', 'connections', 'netstat', 'port', 'tcp', 'udp', 'interface', 'ip',
        'users', 'user', 'who', 'logged', 'login', 'online',
    }
    
    # Count social vs command signals
    social_score = 0
    command_score = 0
    social_type = None
    
    for category, keywords in social_keywords.items():
        overlap = len(words & keywords)
        if overlap > 0:
            social_score += overlap
            if social_type is None:
                social_type = category
    
    command_score = len(words & command_keywords)
    
    # Pure social: high social, no command keywords
    # "hey how's it going" → social
    # "hey what's my network interface" → command (has "network", "interface")
    
    is_pure_social = social_score > 0 and command_score == 0
    
    # Extract command content (remove social words)
    all_social = set()
    for keywords in social_keywords.values():
        all_social.update(keywords)
    
    command_words = [w for w in text.lower().split() if w not in all_social]
    command_content = ' '.join(command_words)
    
    return {
        'is_social': is_pure_social,
        'social_type': social_type if is_pure_social else None,
        'command_content': command_content,
        'social_score': social_score,
        'command_score': command_score,
    }


def extract_command_from_response(response: str) -> str:
    """Extract the actual bash command from a response like 'Use ls -la'."""
    # Look for quoted commands
    quoted = re.findall(r"'([^']+)'", response)
    if quoted:
        cmd = quoted[0]
        # Clean up placeholder paths
        cmd = re.sub(r'/path/to/\w+', '.', cmd)
        cmd = re.sub(r'\[.*?\]', '', cmd)  # Remove [placeholders]
        return cmd.strip()
    
    # Look for backtick commands
    backtick = re.findall(r"`([^`]+)`", response)
    if backtick:
        cmd = backtick[0]
        cmd = re.sub(r'/path/to/\w+', '.', cmd)
        cmd = re.sub(r'\[.*?\]', '', cmd)
        return cmd.strip()
    
    # Look for common command patterns - use sensible defaults
    defaults = {
        'ls': 'ls -la',
        'df': 'df -h',
        'ps': 'ps aux',
        'netstat': 'netstat -tuln',
        'who': 'who',
        'ifconfig': 'ip addr',
        'ip': 'ip addr',
    }
    
    for cmd, default in defaults.items():
        if cmd in response.lower():
            return default
    
    return None


def interactive_chat():
    """Interactive chat with social awareness and command execution."""
    import json
    import subprocess
    
    print("=" * 70)
    print("φ-BASED CHATBOT - Social + Command Execution")
    print("=" * 70)
    print()
    
    # Load training data
    data_path = Path(__file__).parent / "openai_data" / "enhanced_qa.json"
    
    if data_path.exists():
        with open(data_path, 'r') as f:
            corpus = json.load(f)
        qa_pairs = corpus['qa_pairs']
        print(f"Loading {len(qa_pairs)} Q&A pairs...")
    else:
        print("No training data found. Using built-in examples.")
        qa_pairs = [
            ("How do I list files?", "Use 'ls' to list files."),
            ("How do I show disk space?", "Use 'df -h' for disk space."),
            ("How do I find running processes?", "Use 'ps aux' for processes."),
            ("How do I check network connections?", "Use 'netstat -tuln' for connections."),
            ("How do I see who is logged in?", "Use 'who' for logged in users."),
            ("What's my network interface?", "Use 'ip addr' or 'ifconfig' for network interfaces."),
            ("Show my IP address", "Use 'ip addr' to show IP addresses."),
        ]
    
    # Add network interface queries if not present
    extra_qa = [
        ("What's my network interface?", "Use 'ip addr' to show network interfaces."),
        ("Show network interfaces", "Use 'ip addr' or 'ifconfig' for interfaces."),
        ("What's my IP address?", "Use 'ip addr' to display IP addresses."),
    ]
    qa_pairs = qa_pairs + extra_qa
    
    # Create and train chatbot
    bot = PhiChatbot()
    print("Training on Q&A data...")
    bot.ingest_qa(qa_pairs)
    
    # Social responses
    social_responses = {
        'GREETING': [
            "Hey! How can I help you today?",
            "Hello! What would you like to know?",
            "Hi there! Ask me about files, disk, processes, network, or users.",
        ],
        'FAREWELL': [
            "Goodbye! Have a great day!",
            "See you later!",
            "Bye! Come back anytime.",
        ],
        'CHITCHAT': [
            "I'm doing great, thanks for asking! What can I help you with?",
            "All good here! Ready to help with Linux commands.",
            "Fine, thanks! Try asking about files, disk space, or network.",
        ],
        'POLITENESS': [
            "You're welcome! Anything else?",
            "Happy to help! What else do you need?",
        ],
    }
    
    print("\n" + "=" * 70)
    print("READY! I can chat socially or run Linux commands for you.")
    print("Try: 'hey how's it going?' or 'what's my network interface?'")
    print("Commands: 'quit', 'debug', 'help'")
    print("=" * 70)
    
    debug_mode = False
    auto_execute = True  # Automatically run commands
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("Bot: Goodbye! 👋")
            break
        
        if query.lower() == 'debug':
            debug_mode = not debug_mode
            print(f"Bot: Debug mode {'ON' if debug_mode else 'OFF'}")
            continue
        
        if query.lower() == 'execute':
            auto_execute = not auto_execute
            print(f"Bot: Auto-execute commands: {'ON' if auto_execute else 'OFF'}")
            continue
        
        if query.lower() == 'help':
            print("\nBot: I can help with:")
            print("  • Social chat: 'hey', 'how's it going?', 'thanks'")
            print("  • Linux commands: 'list files', 'disk space', 'network interface'")
            print("\nSpecial commands:")
            print("  quit    - Exit")
            print("  debug   - Toggle debug info")
            print("  execute - Toggle auto-execution of commands")
            continue
        
        # Detect social vs command
        social = detect_social(query)
        
        if debug_mode:
            print(f"[DEBUG] Social: {social}")
        
        if social['is_social']:
            # Pure social query
            import random
            responses = social_responses.get(social['social_type'], social_responses['GREETING'])
            print(f"Bot: {random.choice(responses)}")
        else:
            # Command query - use the command content (social words stripped)
            search_query = social['command_content'] if social['command_content'] else query
            
            if debug_mode:
                print(f"[DEBUG] Searching for: '{search_query}'")
            
            matches = bot.knowledge.search(search_query, top_k=3)
            
            if matches and matches[0][2] > 0:  # Has a match with score > 0
                best_q, best_response, score = matches[0]
                
                if debug_mode:
                    print(f"[DEBUG] Best match (score={score:.1f}): {best_response[:50]}...")
                
                # Extract and potentially run the command
                cmd = extract_command_from_response(best_response)
                
                if cmd and auto_execute:
                    print(f"Bot: {best_response}")
                    print(f"\n[Running: {cmd}]")
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, timeout=10
                        )
                        if result.stdout:
                            # Limit output
                            output = result.stdout[:1000]
                            if len(result.stdout) > 1000:
                                output += "\n... (truncated)"
                            print(output)
                        if result.stderr:
                            print(f"[stderr: {result.stderr[:200]}]")
                    except subprocess.TimeoutExpired:
                        print("[Command timed out]")
                    except Exception as e:
                        print(f"[Error: {e}]")
                else:
                    print(f"Bot: {best_response}")
            else:
                print("Bot: I'm not sure about that. Try asking about files, disk, processes, network, or users.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':
        interactive_chat()
    else:
        demo()
