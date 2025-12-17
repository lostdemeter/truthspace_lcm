#!/usr/bin/env python3
"""
Self-Organizing Geometric LCM

A truly dynamic system that:
1. Ingests knowledge WITHOUT domain labels
2. Discovers structure through geometric clustering
3. Switches context based on trajectory in truth space
4. Creates new "regions" when knowledge doesn't fit existing clusters

This is the geometric equivalent of how LLMs organize knowledge through
attention patterns - but explicit and interpretable.

Key insight: Domains are not pre-defined categories, they are
EMERGENT REGIONS of high density in truth space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import hashlib
import re

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Primitive:
    """A semantic primitive."""
    name: str
    dimension: int
    level: int
    keywords: Set[str]
    
    @property
    def activation_value(self) -> float:
        return PHI ** self.level


@dataclass 
class KnowledgeEntry:
    """A piece of knowledge - NO DOMAIN LABEL."""
    id: str
    content: str
    description: str
    position: np.ndarray = None
    # Instead of domain, we track which cluster it belongs to (discovered, not assigned)
    cluster_id: Optional[int] = None


@dataclass
class Region:
    """
    An emergent region in truth space.
    
    NOT a pre-defined domain - discovered through clustering.
    """
    id: int
    centroid: np.ndarray
    entries: List[str] = field(default_factory=list)  # Entry IDs
    # Emergent name (optional, derived from common words)
    suggested_name: Optional[str] = None
    # Radius of the region (for membership testing)
    radius: float = 0.0


# =============================================================================
# SEED PRIMITIVES (same universal set)
# =============================================================================

SEED_PRIMITIVES = [
    # Actions
    Primitive("CREATE", 0, 0, {"create", "make", "new", "generate", "build"}),
    Primitive("DESTROY", 0, 1, {"destroy", "delete", "remove", "eliminate"}),
    Primitive("TRANSFORM", 0, 2, {"change", "transform", "convert", "modify"}),
    Primitive("MOVE", 0, 3, {"move", "transfer", "relocate", "shift"}),
    
    Primitive("READ", 1, 0, {"read", "get", "retrieve", "fetch", "see", "view", "show"}),
    Primitive("WRITE", 1, 1, {"write", "set", "store", "save", "record"}),
    Primitive("SEARCH", 1, 2, {"search", "find", "locate", "seek", "look"}),
    Primitive("COMPARE", 1, 3, {"compare", "match", "differ", "contrast"}),
    
    Primitive("CONNECT", 2, 0, {"connect", "link", "join", "attach"}),
    Primitive("SEPARATE", 2, 1, {"separate", "split", "divide", "disconnect"}),
    Primitive("COMBINE", 2, 2, {"combine", "merge", "unite", "mix"}),
    Primitive("FILTER", 2, 3, {"filter", "select", "choose", "pick"}),
    
    # Relations
    Primitive("INTO", 3, 0, {"into", "to", "toward", "inside"}),
    Primitive("FROM", 3, 1, {"from", "out", "away", "source"}),
    Primitive("WITH", 3, 2, {"with", "using", "by", "through"}),
    Primitive("ABOUT", 3, 3, {"about", "regarding", "concerning"}),
    
    # Time
    Primitive("BEFORE", 4, 0, {"before", "prior", "first", "earlier"}),
    Primitive("AFTER", 4, 1, {"after", "then", "next", "later"}),
    Primitive("DURING", 4, 2, {"during", "while", "when", "as"}),
    Primitive("UNTIL", 4, 3, {"until", "till"}),
    
    # Structure
    Primitive("SEQUENCE", 5, 0, {"sequence", "list", "series", "order"}),
    Primitive("HIERARCHY", 5, 1, {"hierarchy", "tree", "nested", "parent"}),
    Primitive("NETWORK", 5, 2, {"network", "graph", "connected", "web"}),
    Primitive("COLLECTION", 5, 3, {"collection", "set", "group", "batch"}),
    
    # Quantity
    Primitive("ONE", 6, 0, {"one", "single", "individual", "unique"}),
    Primitive("MANY", 6, 1, {"many", "multiple", "several", "various"}),
    Primitive("ALL", 6, 2, {"all", "every", "entire", "complete"}),
    Primitive("NONE", 6, 3, {"none", "no", "zero", "empty"}),
    
    # Social
    Primitive("GREETING", 7, 0, {"hello", "hi", "hey", "greetings"}),
    Primitive("GRATITUDE", 7, 1, {"thanks", "thank", "grateful", "appreciate"}),
    Primitive("APOLOGY", 7, 2, {"sorry", "apologize", "regret"}),
    Primitive("REQUEST", 7, 3, {"please", "could", "would", "can"}),
    
    # Epistemic
    Primitive("KNOW", 8, 0, {"know", "understand", "aware", "realize"}),
    Primitive("BELIEVE", 8, 1, {"believe", "think", "suppose", "assume"}),
    Primitive("DOUBT", 8, 2, {"doubt", "uncertain", "unsure", "question"}),
    Primitive("WANT", 8, 3, {"want", "need", "desire", "wish"}),
    
    # Evaluation
    Primitive("GOOD", 9, 0, {"good", "great", "excellent", "positive"}),
    Primitive("BAD", 9, 1, {"bad", "poor", "negative", "harmful"}),
    Primitive("IMPORTANT", 9, 2, {"important", "significant", "critical"}),
    Primitive("TRIVIAL", 9, 3, {"trivial", "minor", "unimportant"}),
]


# =============================================================================
# SELF-ORGANIZING LCM
# =============================================================================

class SelfOrganizingLCM:
    """
    A geometric LCM that discovers structure without supervision.
    
    Key differences from DynamicLCM:
    1. No domain labels on ingestion
    2. Clusters emerge from geometric proximity
    3. Context tracked as trajectory through space
    4. New regions created when entries are "far" from existing clusters
    """
    
    def __init__(self, dimensions: int = 16, 
                 cluster_threshold: float = 0.4,
                 min_cluster_size: int = 3):
        self.dimensions = dimensions
        self.cluster_threshold = cluster_threshold  # Similarity threshold for clustering
        self.min_cluster_size = min_cluster_size
        
        self.primitives: List[Primitive] = list(SEED_PRIMITIVES)
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.regions: Dict[int, Region] = {}
        self.next_region_id = 0
        
        # Word statistics for emergent primitives
        self.word_frequency: Dict[str, int] = defaultdict(int)
        self.word_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Context tracking - where are we in truth space?
        self.context_position: Optional[np.ndarray] = None
        self.context_history: List[np.ndarray] = []
        self.context_decay: float = 0.7  # How much old context influences new
        
        # Build keyword index
        self._rebuild_keyword_index()
    
    def _rebuild_keyword_index(self):
        self.keyword_to_primitive: Dict[str, List[Primitive]] = defaultdict(list)
        for prim in self.primitives:
            for kw in prim.keywords:
                self.keyword_to_primitive[kw].append(prim)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text into truth space."""
        position = np.zeros(self.dimensions)
        words = self._tokenize(text)
        
        for i, word in enumerate(words):
            position_decay = PHI ** (-i / 2)
            for prim in self.keyword_to_primitive.get(word, []):
                value = prim.activation_value * position_decay
                position[prim.dimension] = max(position[prim.dimension], value)
        
        return position
    
    def similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute similarity between positions."""
        dist = np.sqrt(np.sum((pos1 - pos2) ** 2))
        return 1.0 / (1.0 + dist)
    
    def distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Euclidean distance."""
        return np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    # =========================================================================
    # INGESTION - No domain labels!
    # =========================================================================
    
    def ingest(self, content: str, description: str = None) -> KnowledgeEntry:
        """
        Ingest knowledge WITHOUT a domain label.
        
        The system will:
        1. Encode the entry
        2. Find nearest existing cluster (if any)
        3. Either add to cluster or create new region
        """
        if description is None:
            description = content
        
        entry_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            description=description,
            position=self.encode(description)
        )
        
        # Update word statistics
        words = self._tokenize(description)
        for word in words:
            self.word_frequency[word] += 1
        for i, w1 in enumerate(words):
            for w2 in words[i+1:i+4]:
                pair = tuple(sorted([w1, w2]))
                self.word_cooccurrence[pair] += 1
        
        self.entries[entry_id] = entry
        
        # Find or create cluster
        self._assign_to_cluster(entry)
        
        return entry
    
    def ingest_batch(self, items: List[Dict[str, str]]) -> List[KnowledgeEntry]:
        """Ingest multiple items, then re-cluster."""
        entries = []
        for item in items:
            content = item.get('content', '')
            description = item.get('description', content)
            entry = self.ingest(content, description)
            entries.append(entry)
        
        # After batch, do full re-clustering for better organization
        self._recluster_all()
        
        # Discover emergent primitives
        self._discover_emergent_primitives()
        
        return entries
    
    # =========================================================================
    # CLUSTERING - Emergent regions
    # =========================================================================
    
    def _assign_to_cluster(self, entry: KnowledgeEntry):
        """Assign entry to nearest cluster or create new one."""
        if not self.regions:
            # First entry - create first region
            self._create_region([entry])
            return
        
        # Find nearest region
        best_region = None
        best_sim = -np.inf
        
        for region in self.regions.values():
            sim = self.similarity(entry.position, region.centroid)
            if sim > best_sim:
                best_sim = sim
                best_region = region
        
        if best_sim >= self.cluster_threshold:
            # Close enough - add to existing region
            best_region.entries.append(entry.id)
            entry.cluster_id = best_region.id
            self._update_region_centroid(best_region)
        else:
            # Too far - create new region
            self._create_region([entry])
    
    def _create_region(self, entries: List[KnowledgeEntry]) -> Region:
        """Create a new region from entries."""
        region_id = self.next_region_id
        self.next_region_id += 1
        
        positions = [e.position for e in entries]
        centroid = np.mean(positions, axis=0)
        
        region = Region(
            id=region_id,
            centroid=centroid,
            entries=[e.id for e in entries]
        )
        
        for entry in entries:
            entry.cluster_id = region_id
        
        self.regions[region_id] = region
        self._update_region_name(region)
        
        return region
    
    def _update_region_centroid(self, region: Region):
        """Update region centroid from its entries."""
        positions = [self.entries[eid].position for eid in region.entries]
        region.centroid = np.mean(positions, axis=0)
        
        # Update radius (max distance from centroid)
        if positions:
            distances = [self.distance(p, region.centroid) for p in positions]
            region.radius = max(distances) if distances else 0
    
    def _update_region_name(self, region: Region):
        """Suggest a name for the region based on common words."""
        word_counts = defaultdict(int)
        
        for entry_id in region.entries:
            entry = self.entries[entry_id]
            words = self._tokenize(entry.description)
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += 1
        
        if word_counts:
            # Get most common word not in primitives
            covered = set()
            for prim in self.primitives:
                covered.update(prim.keywords)
            
            candidates = [(w, c) for w, c in word_counts.items() if w not in covered]
            if candidates:
                candidates.sort(key=lambda x: -x[1])
                region.suggested_name = candidates[0][0].upper()
    
    def _recluster_all(self):
        """Re-cluster all entries for better organization."""
        if len(self.entries) < self.min_cluster_size:
            return
        
        # Simple k-means-like clustering
        # Start with current regions as seeds, then iterate
        
        positions = np.array([e.position for e in self.entries.values()])
        entry_ids = list(self.entries.keys())
        
        if len(self.regions) == 0:
            # Initialize with first entry as centroid
            self._create_region([self.entries[entry_ids[0]]])
        
        # Iterate assignment and update
        for _ in range(5):  # 5 iterations
            # Assign each entry to nearest centroid
            for entry_id in entry_ids:
                entry = self.entries[entry_id]
                best_region = None
                best_sim = -np.inf
                
                for region in self.regions.values():
                    sim = self.similarity(entry.position, region.centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_region = region
                
                if best_sim >= self.cluster_threshold:
                    if entry.cluster_id != best_region.id:
                        # Move to new cluster
                        if entry.cluster_id is not None and entry.cluster_id in self.regions:
                            old_region = self.regions[entry.cluster_id]
                            if entry_id in old_region.entries:
                                old_region.entries.remove(entry_id)
                        best_region.entries.append(entry_id)
                        entry.cluster_id = best_region.id
                else:
                    # Create new region for outlier
                    self._create_region([entry])
            
            # Update centroids
            for region in self.regions.values():
                if region.entries:
                    self._update_region_centroid(region)
                    self._update_region_name(region)
        
        # Remove empty regions
        empty = [rid for rid, r in self.regions.items() if not r.entries]
        for rid in empty:
            del self.regions[rid]
    
    # =========================================================================
    # EMERGENT PRIMITIVES
    # =========================================================================
    
    def _discover_emergent_primitives(self, min_freq: int = 3):
        """Discover primitives from frequent uncovered words."""
        covered = set()
        for prim in self.primitives:
            covered.update(prim.keywords)
        
        # Find frequent uncovered words
        for word, freq in self.word_frequency.items():
            if word not in covered and freq >= min_freq and len(word) > 3:
                # Create primitive
                dim = 10 + (len(self.primitives) - 40) // 4
                level = (len(self.primitives) - 40) % 4
                
                prim = Primitive(
                    name=f"EMERGENT_{word.upper()}",
                    dimension=min(dim, self.dimensions - 1),
                    level=level,
                    keywords={word}
                )
                self.primitives.append(prim)
                covered.add(word)
        
        self._rebuild_keyword_index()
        
        # Re-encode all entries with new primitives
        for entry in self.entries.values():
            entry.position = self.encode(entry.description)
    
    # =========================================================================
    # CONTEXT TRACKING - Where are we in truth space?
    # =========================================================================
    
    def update_context(self, text: str):
        """Update context based on new input."""
        new_pos = self.encode(text)
        
        if self.context_position is None:
            self.context_position = new_pos
        else:
            # Blend old context with new
            self.context_position = (
                self.context_decay * self.context_position + 
                (1 - self.context_decay) * new_pos
            )
        
        self.context_history.append(new_pos.copy())
        if len(self.context_history) > 10:
            self.context_history.pop(0)
    
    def get_context_region(self) -> Optional[Region]:
        """Get the region closest to current context."""
        if self.context_position is None:
            return None
        
        best_region = None
        best_sim = -np.inf
        
        for region in self.regions.values():
            sim = self.similarity(self.context_position, region.centroid)
            if sim > best_sim:
                best_sim = sim
                best_region = region
        
        return best_region
    
    def detect_context_switch(self, text: str) -> Tuple[bool, Optional[Region], Optional[Region]]:
        """
        Detect if input represents a context switch.
        
        Returns: (is_switch, old_region, new_region)
        """
        old_region = self.get_context_region()
        
        # Encode new input
        new_pos = self.encode(text)
        
        # Find region for new input
        new_region = None
        best_sim = -np.inf
        
        for region in self.regions.values():
            sim = self.similarity(new_pos, region.centroid)
            if sim > best_sim:
                best_sim = sim
                new_region = region
        
        is_switch = (old_region is not None and 
                    new_region is not None and 
                    old_region.id != new_region.id)
        
        return is_switch, old_region, new_region
    
    # =========================================================================
    # RESOLUTION - Context-aware
    # =========================================================================
    
    def resolve(self, query: str, use_context: bool = True) -> Tuple[KnowledgeEntry, float, Optional[Region]]:
        """
        Resolve query to best matching entry.
        
        If use_context is True, biases toward current context region.
        """
        query_pos = self.encode(query)
        
        # Detect context switch
        is_switch, old_region, new_region = self.detect_context_switch(query)
        
        if is_switch:
            # Context is switching - update it
            self.update_context(query)
        
        # Get current context region
        context_region = self.get_context_region() if use_context else None
        
        best_entry = None
        best_score = -np.inf
        
        for entry in self.entries.values():
            # Base similarity
            sim = self.similarity(query_pos, entry.position)
            
            # Context bonus - entries in context region get a boost
            if use_context and context_region and entry.cluster_id == context_region.id:
                sim *= 1.2  # 20% boost for context-relevant entries
            
            if sim > best_score:
                best_score = sim
                best_entry = entry
        
        # Update context
        self.update_context(query)
        
        # Get the region of the result
        result_region = None
        if best_entry and best_entry.cluster_id in self.regions:
            result_region = self.regions[best_entry.cluster_id]
        
        return best_entry, best_score, result_region
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            'dimensions': self.dimensions,
            'entries': len(self.entries),
            'regions': len(self.regions),
            'primitives': len(self.primitives),
            'emergent_primitives': len(self.primitives) - len(SEED_PRIMITIVES),
            'region_names': [r.suggested_name for r in self.regions.values()],
            'region_sizes': {r.id: len(r.entries) for r in self.regions.values()},
        }
    
    def visualize_regions(self) -> str:
        """Text visualization of regions."""
        lines = ["=== EMERGENT REGIONS ===\n"]
        
        for region in sorted(self.regions.values(), key=lambda r: -len(r.entries)):
            name = region.suggested_name or f"Region_{region.id}"
            lines.append(f"\n{name} ({len(region.entries)} entries)")
            lines.append("-" * 40)
            
            # Show a few example entries
            for entry_id in region.entries[:3]:
                entry = self.entries[entry_id]
                lines.append(f"  • {entry.content[:60]}...")
            
            if len(region.entries) > 3:
                lines.append(f"  ... and {len(region.entries) - 3} more")
        
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    lcm = SelfOrganizingLCM(dimensions=16, cluster_threshold=0.5)  # Higher threshold = more regions
    
    # Ingest knowledge WITHOUT domain labels
    knowledge = [
        # These will cluster together (cooking-like)
        {"content": "chop onions finely", "description": "chop cut onions knife cooking"},
        {"content": "boil water in pot", "description": "boil water pot heat cooking"},
        {"content": "season with salt", "description": "season salt pepper taste cooking"},
        {"content": "preheat oven", "description": "preheat oven temperature baking"},
        {"content": "simmer for 20 minutes", "description": "simmer heat low cooking time"},
        
        # These will cluster together (tech-like)
        {"content": "ls -la", "description": "list files directory show all"},
        {"content": "mkdir new_folder", "description": "create make directory folder new"},
        {"content": "rm -rf", "description": "delete remove files directory force"},
        {"content": "grep pattern file", "description": "search find pattern text file"},
        {"content": "cat file.txt", "description": "read show file contents display"},
        
        # These will cluster together (social-like)
        {"content": "Hello! How can I help?", "description": "hello greeting help assist"},
        {"content": "Thank you!", "description": "thanks gratitude appreciate"},
        {"content": "I understand how you feel", "description": "understand feeling empathy emotion"},
        {"content": "Take care!", "description": "goodbye care wellbeing"},
    ]
    
    print("Ingesting knowledge (no domain labels)...")
    lcm.ingest_batch(knowledge)
    
    print("\n" + lcm.visualize_regions())
    
    print("\n=== STATS ===")
    stats = lcm.stats()
    print(f"Entries: {stats['entries']}")
    print(f"Emergent regions: {stats['regions']}")
    print(f"Region names: {stats['region_names']}")
    
    print("\n=== RESOLUTION TEST ===")
    queries = [
        "how do I cut vegetables",
        "show me the files",
        "hello there",
        "delete the folder",  # Should stay in tech context
        "thanks for your help",  # Context switch to social
    ]
    
    for query in queries:
        entry, score, region = lcm.resolve(query)
        region_name = region.suggested_name if region else "None"
        print(f"\n\"{query}\"")
        print(f"  → {entry.content[:40]}... (region: {region_name}, score: {score:.2f})")
