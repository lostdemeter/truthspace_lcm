#!/usr/bin/env python3
"""
Dynamic Geometric LCM

A scalable, generalizable approach to geometric knowledge representation
that can bootstrap from seed primitives and grow through knowledge ingestion.

Core Ideas:
1. SEED PRIMITIVES: Minimal irreducible concepts (actions, relations, structures)
2. EMERGENT PRIMITIVES: Discovered from patterns in ingested knowledge
3. DOMAIN REGIONS: Automatically created clusters in truth space
4. KNOWLEDGE ENTRIES: Positioned relative to primitives (seed + emergent)

This is the foundation for a geometric alternative to traditional LLMs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import hashlib
import re

# The golden ratio - fundamental to our geometry
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Primitive:
    """
    A semantic primitive - an atomic concept in truth space.
    
    Primitives can be:
    - SEED: Hand-coded, irreducible concepts (actions, relations)
    - EMERGENT: Discovered from patterns in ingested knowledge
    """
    name: str
    dimension: int
    level: int  # φ^level determines activation strength
    keywords: Set[str]
    is_seed: bool = True  # False for emergent primitives
    domain: Optional[str] = None  # Which domain this primitive belongs to
    
    @property
    def activation_value(self) -> float:
        return PHI ** self.level


@dataclass
class KnowledgeEntry:
    """A piece of knowledge positioned in truth space."""
    id: str
    content: str  # The actual knowledge (response, command, fact, etc.)
    description: str  # Natural language description for encoding
    domain: str  # Which domain this belongs to
    position: Optional[np.ndarray] = None  # Computed position in truth space
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Domain:
    """
    A knowledge domain - a region in truth space.
    
    Domains are discovered/created as knowledge is ingested.
    Each domain has its own emergent primitives.
    """
    name: str
    description: str
    centroid: Optional[np.ndarray] = None
    primitives: List[Primitive] = field(default_factory=list)  # Domain-specific primitives
    entries: List[KnowledgeEntry] = field(default_factory=list)
    parent: Optional[str] = None  # Parent domain name (for hierarchy)
    children: List[str] = field(default_factory=list)


# =============================================================================
# SEED PRIMITIVES - The Irreducible Core
# =============================================================================

# These are the minimal set of primitives that bootstrap the system.
# They represent fundamental concepts that appear across ALL domains.

SEED_PRIMITIVES = [
    # === ACTIONS (What can be done) ===
    # These are universal - every domain has actions
    Primitive("CREATE", 0, 0, {"create", "make", "new", "generate", "build", "produce"}),
    Primitive("DESTROY", 0, 1, {"destroy", "delete", "remove", "eliminate", "end"}),
    Primitive("TRANSFORM", 0, 2, {"change", "transform", "convert", "modify", "alter"}),
    Primitive("MOVE", 0, 3, {"move", "transfer", "relocate", "shift", "transport"}),
    
    Primitive("READ", 1, 0, {"read", "get", "retrieve", "fetch", "obtain", "see", "view"}),
    Primitive("WRITE", 1, 1, {"write", "set", "store", "save", "record"}),
    Primitive("SEARCH", 1, 2, {"search", "find", "locate", "seek", "look", "query"}),
    Primitive("COMPARE", 1, 3, {"compare", "match", "differ", "contrast", "versus"}),
    
    Primitive("CONNECT", 2, 0, {"connect", "link", "join", "attach", "bind"}),
    Primitive("SEPARATE", 2, 1, {"separate", "split", "divide", "disconnect", "detach"}),
    Primitive("COMBINE", 2, 2, {"combine", "merge", "unite", "aggregate", "collect"}),
    Primitive("FILTER", 2, 3, {"filter", "select", "choose", "pick", "extract"}),
    
    # === RELATIONS (How things relate) ===
    Primitive("INTO", 3, 0, {"into", "to", "toward", "inside"}),
    Primitive("FROM", 3, 1, {"from", "out", "away", "source"}),
    Primitive("WITH", 3, 2, {"with", "using", "by", "through"}),
    Primitive("ABOUT", 3, 3, {"about", "regarding", "concerning", "on"}),
    
    Primitive("BEFORE", 4, 0, {"before", "prior", "first", "earlier", "previous"}),
    Primitive("AFTER", 4, 1, {"after", "then", "next", "later", "following"}),
    Primitive("DURING", 4, 2, {"during", "while", "when", "as"}),
    Primitive("UNTIL", 4, 3, {"until", "till", "up to"}),
    
    # === STRUCTURES (Organizational patterns) ===
    Primitive("SEQUENCE", 5, 0, {"sequence", "list", "series", "order", "chain"}),
    Primitive("HIERARCHY", 5, 1, {"hierarchy", "tree", "nested", "parent", "child"}),
    Primitive("NETWORK", 5, 2, {"network", "graph", "connected", "web", "mesh"}),
    Primitive("COLLECTION", 5, 3, {"collection", "set", "group", "bunch", "batch"}),
    
    # === QUANTITIES (Amounts and measures) ===
    Primitive("ONE", 6, 0, {"one", "single", "individual", "unique", "sole"}),
    Primitive("MANY", 6, 1, {"many", "multiple", "several", "various", "numerous"}),
    Primitive("ALL", 6, 2, {"all", "every", "entire", "complete", "whole"}),
    Primitive("NONE", 6, 3, {"none", "no", "zero", "empty", "nothing"}),
    
    # === SOCIAL (Human interaction - universal across cultures) ===
    Primitive("GREETING", 7, 0, {"hello", "hi", "hey", "greetings", "welcome"}),
    Primitive("GRATITUDE", 7, 1, {"thanks", "thank", "grateful", "appreciate"}),
    Primitive("APOLOGY", 7, 2, {"sorry", "apologize", "regret", "pardon"}),
    Primitive("REQUEST", 7, 3, {"please", "could", "would", "can", "may"}),
    
    # === EPISTEMIC (Knowledge states) ===
    Primitive("KNOW", 8, 0, {"know", "understand", "aware", "realize", "recognize"}),
    Primitive("BELIEVE", 8, 1, {"believe", "think", "suppose", "assume", "expect"}),
    Primitive("DOUBT", 8, 2, {"doubt", "uncertain", "unsure", "question", "wonder"}),
    Primitive("WANT", 8, 3, {"want", "need", "desire", "wish", "hope"}),
    
    # === EVALUATION (Judgments) ===
    Primitive("GOOD", 9, 0, {"good", "great", "excellent", "positive", "beneficial"}),
    Primitive("BAD", 9, 1, {"bad", "poor", "negative", "harmful", "wrong"}),
    Primitive("IMPORTANT", 9, 2, {"important", "significant", "critical", "essential", "key"}),
    Primitive("TRIVIAL", 9, 3, {"trivial", "minor", "unimportant", "negligible"}),
]

# Reserve dimensions 10-15 for emergent domain primitives
EMERGENT_DIM_START = 10
EMERGENT_DIM_END = 15  # Can expand as needed


# =============================================================================
# DYNAMIC LCM CLASS
# =============================================================================

class DynamicLCM:
    """
    A dynamic, scalable geometric language model.
    
    Key capabilities:
    1. Bootstrap from seed primitives
    2. Ingest knowledge and discover patterns
    3. Create emergent primitives from frequent co-occurrences
    4. Organize knowledge into domain hierarchies
    5. Resolve queries through geometric navigation
    """
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.primitives: List[Primitive] = list(SEED_PRIMITIVES)
        self.domains: Dict[str, Domain] = {}
        self.entries: Dict[str, KnowledgeEntry] = {}
        
        # For emergent primitive discovery
        self.word_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.word_frequency: Dict[str, int] = defaultdict(int)
        self.next_emergent_dim = EMERGENT_DIM_START
        self.next_emergent_level = 0
        
        # Build keyword index for fast lookup
        self._rebuild_keyword_index()
    
    def _rebuild_keyword_index(self):
        """Build index from keywords to primitives."""
        self.keyword_to_primitive: Dict[str, List[Primitive]] = defaultdict(list)
        for prim in self.primitives:
            for kw in prim.keywords:
                self.keyword_to_primitive[kw].append(prim)
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into truth space using φ-MAX encoding.
        
        This is the core geometric operation - positioning text in truth space.
        """
        position = np.zeros(self.dimensions)
        words = self._tokenize(text)
        
        for i, word in enumerate(words):
            # Position decay based on word position
            position_decay = PHI ** (-i / 2)
            
            # Find primitives activated by this word
            for prim in self.keyword_to_primitive.get(word, []):
                value = prim.activation_value * position_decay
                # MAX encoding - take maximum, don't sum
                position[prim.dimension] = max(position[prim.dimension], value)
        
        return position
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, preserving important structures."""
        # Preserve filenames, paths, hyphenated words
        tokens = re.findall(r'[\w./]+\.[\w]+|/[\w./]+|\b[\w]+-[\w-]+\b|\b\w+\b', text.lower())
        return tokens
    
    def similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute similarity between two positions in truth space."""
        dist = np.sqrt(np.sum((pos1 - pos2) ** 2))
        return 1.0 / (1.0 + dist)
    
    # =========================================================================
    # KNOWLEDGE INGESTION
    # =========================================================================
    
    def ingest_knowledge(self, content: str, description: str, domain: str,
                        metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """
        Ingest a piece of knowledge into the system.
        
        This:
        1. Creates a knowledge entry
        2. Positions it in truth space
        3. Updates word statistics for emergent primitive discovery
        4. Adds to appropriate domain
        """
        # Generate ID
        entry_id = hashlib.md5(f"{domain}:{content}".encode()).hexdigest()[:12]
        
        # Create entry
        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            description=description,
            domain=domain,
            metadata=metadata or {}
        )
        
        # Encode and position
        entry.position = self.encode(description)
        
        # Update word statistics
        words = self._tokenize(description)
        for word in words:
            self.word_frequency[word] += 1
        for i, w1 in enumerate(words):
            for w2 in words[i+1:i+4]:  # Co-occurrence window of 3
                pair = tuple(sorted([w1, w2]))
                self.word_cooccurrence[pair] += 1
        
        # Ensure domain exists
        if domain not in self.domains:
            self.create_domain(domain, f"Auto-created domain: {domain}")
        
        # Add to domain and global index
        self.domains[domain].entries.append(entry)
        self.entries[entry_id] = entry
        
        return entry
    
    def ingest_batch(self, items: List[Dict[str, str]]) -> List[KnowledgeEntry]:
        """
        Ingest a batch of knowledge items.
        
        Each item should have: content, description, domain
        """
        entries = []
        for item in items:
            entry = self.ingest_knowledge(
                content=item['content'],
                description=item['description'],
                domain=item['domain'],
                metadata=item.get('metadata', {})
            )
            entries.append(entry)
        
        # After batch ingestion, check for emergent primitives
        self._discover_emergent_primitives()
        
        # Re-encode all entries with new primitives
        self._reencode_all_entries()
        
        # Update domain centroids
        self._update_domain_centroids()
        
        return entries
    
    # =========================================================================
    # DOMAIN MANAGEMENT
    # =========================================================================
    
    def create_domain(self, name: str, description: str, parent: str = None) -> Domain:
        """Create a new knowledge domain."""
        domain = Domain(
            name=name,
            description=description,
            parent=parent
        )
        
        if parent and parent in self.domains:
            self.domains[parent].children.append(name)
        
        self.domains[name] = domain
        
        # Create a primitive for the domain name itself
        # This ensures queries containing the domain name route correctly
        domain_keywords = {name.lower(), name.lower().replace('-', ' ')}
        self._create_emergent_primitive(
            keywords=domain_keywords,
            domain=name,
            name_prefix=f"DOMAIN_"
        )
        
        return domain
    
    def _update_domain_centroids(self):
        """Update centroids for all domains based on their entries."""
        for domain in self.domains.values():
            if domain.entries:
                positions = [e.position for e in domain.entries if e.position is not None]
                if positions:
                    domain.centroid = np.mean(positions, axis=0)
    
    def _reencode_all_entries(self):
        """Re-encode all entries after new primitives are discovered."""
        for entry in self.entries.values():
            entry.position = self.encode(entry.description)
    
    # =========================================================================
    # EMERGENT PRIMITIVE DISCOVERY
    # =========================================================================
    
    def _discover_emergent_primitives(self, min_frequency: int = 2, 
                                       min_cooccurrence: int = 2):
        """
        Discover new primitives from patterns in ingested knowledge.
        
        This is the key to scalability - primitives emerge from data,
        not just from hand-coding.
        
        Strategy:
        1. Find frequent words not covered by existing primitives
        2. Find domain-specific words (appear mostly in one domain)
        3. Create new primitives for significant patterns
        """
        # Find words not covered by existing primitives
        covered_words = set()
        for prim in self.primitives:
            covered_words.update(prim.keywords)
        
        # Track which domains each word appears in
        word_domains: Dict[str, Set[str]] = defaultdict(set)
        word_domain_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for entry in self.entries.values():
            words = self._tokenize(entry.description)
            for word in words:
                word_domains[word].add(entry.domain)
                word_domain_freq[word][entry.domain] += 1
        
        # Find domain-specific words (appear predominantly in one domain)
        for word, domains in word_domains.items():
            if word in covered_words:
                continue
            if len(word) < 3:  # Skip very short words
                continue
            
            total_freq = sum(word_domain_freq[word].values())
            if total_freq < min_frequency:
                continue
            
            # Check if word is domain-specific (>70% in one domain)
            for domain, freq in word_domain_freq[word].items():
                if freq / total_freq >= 0.7:
                    # This word is specific to this domain
                    self._create_emergent_primitive(
                        keywords={word},
                        domain=domain,
                        name_prefix=f"{domain.upper()}_"
                    )
                    break
        
        # Also find co-occurring word groups
        uncovered_frequent = []
        for word, freq in self.word_frequency.items():
            if word not in covered_words and freq >= min_frequency and len(word) >= 3:
                uncovered_frequent.append((word, freq))
        
        uncovered_frequent.sort(key=lambda x: -x[1])
        
        word_groups = self._cluster_by_cooccurrence(
            [w for w, _ in uncovered_frequent[:30]],
            min_cooccurrence
        )
        
        for group in word_groups:
            if len(group) >= 2:
                self._create_emergent_primitive(group)
    
    def _cluster_by_cooccurrence(self, words: List[str], 
                                  min_cooccurrence: int) -> List[Set[str]]:
        """Cluster words by co-occurrence patterns."""
        # Simple greedy clustering
        clusters = []
        used = set()
        
        for word in words:
            if word in used:
                continue
            
            cluster = {word}
            used.add(word)
            
            # Find words that co-occur frequently with this word
            for other in words:
                if other in used:
                    continue
                pair = tuple(sorted([word, other]))
                if self.word_cooccurrence[pair] >= min_cooccurrence:
                    cluster.add(other)
                    used.add(other)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _create_emergent_primitive(self, keywords: Set[str], 
                                    domain: str = None,
                                    name_prefix: str = "") -> Primitive:
        """Create a new emergent primitive from discovered keywords."""
        # Generate name from keywords
        name = name_prefix + "_".join(sorted(keywords)[:3]).upper()
        
        # Assign dimension and level
        dim = self.next_emergent_dim
        level = self.next_emergent_level
        
        # Advance to next slot
        self.next_emergent_level += 1
        if self.next_emergent_level > 3:
            self.next_emergent_level = 0
            self.next_emergent_dim += 1
            
            # Expand dimensions if needed
            if self.next_emergent_dim >= self.dimensions:
                self._expand_dimensions()
        
        prim = Primitive(
            name=name,
            dimension=dim,
            level=level,
            keywords=keywords,
            is_seed=False,
            domain=domain
        )
        
        self.primitives.append(prim)
        self._rebuild_keyword_index()
        
        return prim
    
    def _expand_dimensions(self, new_dims: int = 4):
        """Expand truth space dimensions to accommodate more primitives."""
        old_dims = self.dimensions
        self.dimensions += new_dims
        
        # Re-encode all entries with new dimensions
        for entry in self.entries.values():
            old_pos = entry.position
            new_pos = np.zeros(self.dimensions)
            new_pos[:old_dims] = old_pos
            entry.position = new_pos
        
        # Update domain centroids
        self._update_domain_centroids()
    
    # =========================================================================
    # QUERY RESOLUTION
    # =========================================================================
    
    def resolve(self, query: str, domain: str = None) -> Tuple[KnowledgeEntry, float]:
        """
        Resolve a query to the best matching knowledge entry.
        
        If domain is specified, search only within that domain.
        Otherwise, search all domains.
        """
        query_pos = self.encode(query)
        
        best_entry = None
        best_sim = -np.inf
        
        # Determine which entries to search
        if domain and domain in self.domains:
            search_entries = self.domains[domain].entries
        else:
            search_entries = list(self.entries.values())
        
        for entry in search_entries:
            if entry.position is not None:
                sim = self.similarity(query_pos, entry.position)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
        
        return best_entry, best_sim
    
    def resolve_with_domain_detection(self, query: str) -> Tuple[str, KnowledgeEntry, float]:
        """
        First detect the most likely domain, then resolve within it.
        
        Uses a hybrid approach:
        1. Check for direct domain name matches in query
        2. Check for domain-specific primitive activations
        3. Fall back to centroid similarity
        """
        query_lower = query.lower()
        query_words = set(self._tokenize(query))
        query_pos = self.encode(query)
        
        # Stage 1a: Direct domain name match (highest priority)
        for domain_name in self.domains:
            if domain_name.lower() in query_lower or domain_name.lower().replace('-', ' ') in query_lower:
                entry, sim = self.resolve(query, domain=domain_name)
                return domain_name, entry, sim
        
        # Stage 1b: Check domain-specific primitive activations
        domain_scores = {}
        for prim in self.primitives:
            if prim.domain and prim.domain in self.domains:
                # Check if any primitive keywords are in the query
                overlap = prim.keywords & query_words
                if overlap:
                    if prim.domain not in domain_scores:
                        domain_scores[prim.domain] = 0
                    domain_scores[prim.domain] += len(overlap) * prim.activation_value
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            entry, sim = self.resolve(query, domain=best_domain)
            return best_domain, entry, sim
        
        # Stage 1c: Fall back to centroid similarity
        best_domain = None
        best_domain_sim = -np.inf
        
        for domain in self.domains.values():
            if domain.centroid is not None:
                sim = self.similarity(query_pos, domain.centroid)
                if sim > best_domain_sim:
                    best_domain_sim = sim
                    best_domain = domain.name
        
        # Stage 2: Resolve within domain
        if best_domain:
            entry, sim = self.resolve(query, domain=best_domain)
            return best_domain, entry, sim
        else:
            entry, sim = self.resolve(query)
            return "unknown", entry, sim
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def save(self, path: str):
        """Save the LCM state to a file."""
        state = {
            'dimensions': self.dimensions,
            'primitives': [
                {
                    'name': p.name,
                    'dimension': p.dimension,
                    'level': p.level,
                    'keywords': list(p.keywords),
                    'is_seed': p.is_seed,
                    'domain': p.domain
                }
                for p in self.primitives if not p.is_seed  # Only save emergent
            ],
            'domains': {
                name: {
                    'description': d.description,
                    'parent': d.parent,
                    'children': d.children
                }
                for name, d in self.domains.items()
            },
            'entries': [
                {
                    'id': e.id,
                    'content': e.content,
                    'description': e.description,
                    'domain': e.domain,
                    'metadata': e.metadata
                }
                for e in self.entries.values()
            ],
            'word_frequency': dict(self.word_frequency),
            'next_emergent_dim': self.next_emergent_dim,
            'next_emergent_level': self.next_emergent_level
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str):
        """Load LCM state from a file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.dimensions = state['dimensions']
        self.next_emergent_dim = state['next_emergent_dim']
        self.next_emergent_level = state['next_emergent_level']
        
        # Load emergent primitives
        for p in state['primitives']:
            prim = Primitive(
                name=p['name'],
                dimension=p['dimension'],
                level=p['level'],
                keywords=set(p['keywords']),
                is_seed=p['is_seed'],
                domain=p['domain']
            )
            self.primitives.append(prim)
        
        self._rebuild_keyword_index()
        
        # Load domains
        for name, d in state['domains'].items():
            self.domains[name] = Domain(
                name=name,
                description=d['description'],
                parent=d['parent'],
                children=d['children']
            )
        
        # Load entries
        for e in state['entries']:
            entry = KnowledgeEntry(
                id=e['id'],
                content=e['content'],
                description=e['description'],
                domain=e['domain'],
                metadata=e['metadata']
            )
            entry.position = self.encode(entry.description)
            self.entries[entry.id] = entry
            
            if entry.domain in self.domains:
                self.domains[entry.domain].entries.append(entry)
        
        # Load word frequency
        self.word_frequency = defaultdict(int, state['word_frequency'])
        
        # Update centroids
        self._update_domain_centroids()
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics about the LCM."""
        seed_prims = [p for p in self.primitives if p.is_seed]
        emergent_prims = [p for p in self.primitives if not p.is_seed]
        
        return {
            'dimensions': self.dimensions,
            'seed_primitives': len(seed_prims),
            'emergent_primitives': len(emergent_prims),
            'total_primitives': len(self.primitives),
            'domains': len(self.domains),
            'entries': len(self.entries),
            'unique_words': len(self.word_frequency),
            'domain_sizes': {name: len(d.entries) for name, d in self.domains.items()}
        }
    
    def explain_encoding(self, text: str) -> str:
        """Explain how a text is encoded."""
        pos = self.encode(text)
        words = self._tokenize(text)
        
        lines = [f"Text: \"{text}\"", ""]
        
        # Show word -> primitive mappings
        lines.append("Word activations:")
        for word in words:
            prims = self.keyword_to_primitive.get(word, [])
            if prims:
                prim_names = [p.name for p in prims]
                lines.append(f"  '{word}' → {', '.join(prim_names)}")
            else:
                lines.append(f"  '{word}' → (no primitive)")
        
        lines.append("")
        lines.append("Position vector (non-zero dims):")
        for i, v in enumerate(pos):
            if v > 0.01:
                # Find primitive at this dimension
                prim_name = "?"
                for p in self.primitives:
                    if p.dimension == i and abs(p.activation_value - v) < 0.1:
                        prim_name = p.name
                        break
                lines.append(f"  dim {i}: {v:.3f} ({prim_name})")
        
        return "\n".join(lines)
