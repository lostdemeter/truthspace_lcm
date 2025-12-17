#!/usr/bin/env python3
"""
Stacked Geometric LCM

Recreates the effect of LLM embeddings using hierarchical geometric encoding.
No training required - structure emerges from stacked geometric transformations.

Key insight: LLM embeddings encode relationships at multiple scales.
We can recreate this by stacking geometric layers:

Layer 0: Morphological (word structure, prefixes, suffixes)
Layer 1: Lexical (word → primitive activation)
Layer 2: Compositional (primitive combinations → concepts)
Layer 3: Contextual (concept relationships → domains)
Layer 4: Global (domain structure)

Each layer transforms the representation, adding discriminative power.
The final embedding is the concatenation of all layer outputs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import re

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# LAYER 0: MORPHOLOGICAL ENCODING
# =============================================================================

class MorphologicalLayer:
    """
    Encodes word structure - prefixes, suffixes, character patterns.
    
    This captures structural similarity:
    - "cooking" and "baking" share "-ing" suffix
    - "pre-heat" and "pre-pare" share "pre-" prefix
    - "chop" and "chip" share character patterns
    """
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        
        # Common morphemes
        self.prefixes = {
            'pre': 0, 'un': 1, 're': 2, 'dis': 3, 'over': 4,
            'under': 5, 'out': 6, 'up': 7, 'down': 8, 'sub': 9
        }
        self.suffixes = {
            'ing': 0, 'ed': 1, 'er': 2, 'est': 3, 'ly': 4,
            'tion': 5, 'ness': 6, 'ment': 7, 'able': 8, 'ful': 9
        }
    
    def encode_word(self, word: str) -> np.ndarray:
        """Encode a single word's morphological structure."""
        pos = np.zeros(self.dimensions)
        word = word.lower()
        
        # Prefix detection (dims 0-4)
        for prefix, idx in self.prefixes.items():
            if word.startswith(prefix) and idx < 5:
                pos[idx] = PHI
        
        # Suffix detection (dims 5-9)
        for suffix, idx in self.suffixes.items():
            if word.endswith(suffix) and idx < 5:
                pos[5 + idx] = PHI
        
        # Character n-gram hashing (dims 10-15)
        # Creates unique signature based on character patterns
        for i in range(len(word) - 2):
            trigram = word[i:i+3]
            dim = 10 + (hash(trigram) % 6)
            pos[dim] = max(pos[dim], PHI ** (1 - i/len(word)))
        
        # Word length encoding (dim 15)
        pos[15] = min(len(word) / 10, 1.0)
        
        return pos
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text by averaging word morphologies."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return np.zeros(self.dimensions)
        
        embeddings = [self.encode_word(w) for w in words]
        return np.mean(embeddings, axis=0)


# =============================================================================
# LAYER 1: LEXICAL ENCODING (Primitives)
# =============================================================================

@dataclass
class Primitive:
    name: str
    dimension: int
    level: int
    keywords: Set[str]
    
    @property
    def activation(self) -> float:
        return PHI ** self.level


class LexicalLayer:
    """
    Maps words to semantic primitives.
    
    This is our existing φ-MAX encoding, but as a layer.
    """
    
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
        self.primitives = self._build_primitives()
        self._build_index()
    
    def _build_primitives(self) -> List[Primitive]:
        return [
            # Actions (dims 0-5)
            Primitive("CREATE", 0, 0, {"create", "make", "new", "build", "generate"}),
            Primitive("DESTROY", 0, 1, {"destroy", "delete", "remove", "kill", "end"}),
            Primitive("TRANSFORM", 1, 0, {"change", "transform", "convert", "modify"}),
            Primitive("MOVE", 1, 1, {"move", "transfer", "shift", "relocate"}),
            Primitive("READ", 2, 0, {"read", "get", "view", "show", "see", "list", "display"}),
            Primitive("WRITE", 2, 1, {"write", "set", "save", "store", "record"}),
            Primitive("SEARCH", 3, 0, {"search", "find", "locate", "seek", "look", "grep"}),
            Primitive("COMBINE", 3, 1, {"combine", "merge", "mix", "blend", "join"}),
            
            # Objects (dims 6-11)
            Primitive("FILE", 6, 0, {"file", "files", "document", "data"}),
            Primitive("DIRECTORY", 6, 1, {"directory", "folder", "path", "dir"}),
            Primitive("PROCESS", 7, 0, {"process", "program", "running", "task"}),
            Primitive("SYSTEM", 7, 1, {"system", "computer", "machine", "server"}),
            Primitive("FOOD", 8, 0, {"food", "meal", "dish", "recipe", "ingredient"}),
            Primitive("HEAT", 8, 1, {"heat", "hot", "warm", "temperature", "boil", "bake"}),
            Primitive("CUT", 9, 0, {"cut", "chop", "slice", "dice", "knife"}),
            Primitive("TASTE", 9, 1, {"taste", "flavor", "season", "salt", "pepper", "spice"}),
            
            # Social (dims 12-15)
            Primitive("GREETING", 12, 0, {"hello", "hi", "hey", "greetings", "welcome"}),
            Primitive("GRATITUDE", 12, 1, {"thanks", "thank", "grateful", "appreciate"}),
            Primitive("FEELING", 13, 0, {"feel", "feeling", "emotion", "mood"}),
            Primitive("HELP", 13, 1, {"help", "assist", "support", "aid"}),
            
            # Relations (dims 16-19)
            Primitive("INTO", 16, 0, {"into", "to", "toward", "inside"}),
            Primitive("FROM", 16, 1, {"from", "out", "away", "source"}),
            Primitive("WITH", 17, 0, {"with", "using", "by", "through"}),
            Primitive("ABOUT", 17, 1, {"about", "regarding", "concerning"}),
            
            # Structure (dims 20-23)
            Primitive("SEQUENCE", 20, 0, {"sequence", "list", "series", "order", "step"}),
            Primitive("COLLECTION", 20, 1, {"collection", "set", "group", "all", "every"}),
            Primitive("ONE", 21, 0, {"one", "single", "individual", "a", "an"}),
            Primitive("MANY", 21, 1, {"many", "multiple", "several", "various"}),
        ]
    
    def _build_index(self):
        self.kw_to_prim: Dict[str, List[Primitive]] = defaultdict(list)
        for p in self.primitives:
            for kw in p.keywords:
                self.kw_to_prim[kw].append(p)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using primitive activation."""
        pos = np.zeros(self.dimensions)
        words = re.findall(r'\b\w+\b', text.lower())
        
        for i, word in enumerate(words):
            decay = PHI ** (-i / 3)
            for prim in self.kw_to_prim.get(word, []):
                val = prim.activation * decay
                if prim.dimension < self.dimensions:
                    pos[prim.dimension] = max(pos[prim.dimension], val)
        
        return pos


# =============================================================================
# LAYER 2: COMPOSITIONAL ENCODING
# =============================================================================

class CompositionalLayer:
    """
    Encodes primitive combinations into concept signatures.
    
    Key insight: Concepts are PATTERNS of primitive activation.
    - "cooking" = FOOD + HEAT + CUT + TASTE
    - "terminal" = FILE + DIRECTORY + PROCESS + SYSTEM
    - "social" = GREETING + GRATITUDE + FEELING + HELP
    
    This layer detects these patterns and creates concept-level features.
    """
    
    def __init__(self, input_dim: int = 24, output_dim: int = 16):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define concept patterns (which primitive dimensions activate together)
        # These are discovered from the structure of our primitives
        self.patterns = {
            'cooking_pattern': [8, 9],      # FOOD, HEAT, CUT, TASTE dims
            'tech_pattern': [6, 7],          # FILE, DIRECTORY, PROCESS, SYSTEM dims
            'social_pattern': [12, 13],      # GREETING, GRATITUDE, FEELING, HELP dims
            'action_pattern': [0, 1, 2, 3],  # CREATE, DESTROY, TRANSFORM, etc.
            'relation_pattern': [16, 17],    # INTO, FROM, WITH, ABOUT dims
            'structure_pattern': [20, 21],   # SEQUENCE, COLLECTION, ONE, MANY dims
        }
    
    def encode(self, lexical_embedding: np.ndarray) -> np.ndarray:
        """Transform lexical embedding into compositional features."""
        pos = np.zeros(self.output_dim)
        
        # Detect pattern activations
        for i, (pattern_name, dims) in enumerate(self.patterns.items()):
            if i >= self.output_dim:
                break
            
            # Pattern strength = geometric mean of relevant dimensions
            relevant = [lexical_embedding[d] for d in dims if d < len(lexical_embedding)]
            if relevant:
                # Use product (geometric combination) not sum
                strength = np.prod([v + 0.1 for v in relevant]) ** (1/len(relevant))
                pos[i] = strength
        
        # Cross-pattern interactions (dims 6-11)
        # These capture relationships BETWEEN patterns
        pattern_values = pos[:6]
        for i in range(6):
            for j in range(i+1, 6):
                if i + j < self.output_dim:
                    # Interaction strength
                    interaction = pattern_values[i] * pattern_values[j]
                    pos[6 + (i*5 + j) % 10] = max(pos[6 + (i*5 + j) % 10], interaction)
        
        return pos


# =============================================================================
# LAYER 3: CONTEXTUAL ENCODING
# =============================================================================

class ContextualLayer:
    """
    Encodes relationships between concepts based on co-occurrence.
    
    This layer learns (without training!) which concepts appear together
    by tracking statistics during ingestion.
    
    Key insight: Context is encoded in the RELATIONSHIPS between concepts,
    not just the concepts themselves.
    """
    
    def __init__(self, input_dim: int = 16, output_dim: int = 16):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Co-occurrence statistics (built during ingestion)
        self.cooccurrence = np.zeros((input_dim, input_dim))
        self.occurrence_count = np.zeros(input_dim)
        self.total_samples = 0
    
    def update_statistics(self, compositional_embedding: np.ndarray):
        """Update co-occurrence statistics from a new sample."""
        self.total_samples += 1
        
        # Find active dimensions (above threshold)
        active = np.where(compositional_embedding > 0.1)[0]
        
        # Update occurrence counts
        for dim in active:
            if dim < self.input_dim:
                self.occurrence_count[dim] += 1
        
        # Update co-occurrence
        for i in active:
            for j in active:
                if i < self.input_dim and j < self.input_dim:
                    self.cooccurrence[i, j] += 1
    
    def encode(self, compositional_embedding: np.ndarray) -> np.ndarray:
        """Transform compositional embedding using learned context."""
        pos = np.zeros(self.output_dim)
        
        if self.total_samples < 2:
            # Not enough data - pass through
            return compositional_embedding[:self.output_dim] if len(compositional_embedding) >= self.output_dim else np.pad(compositional_embedding, (0, self.output_dim - len(compositional_embedding)))
        
        # Compute PMI-weighted features
        # PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
        # This captures which concepts are MORE likely to co-occur than chance
        
        active = np.where(compositional_embedding > 0.1)[0]
        
        for idx, dim in enumerate(active):
            if dim >= self.input_dim or idx >= self.output_dim:
                continue
            
            # Base activation
            pos[idx] = compositional_embedding[dim]
            
            # PMI boost from co-occurring concepts
            p_dim = self.occurrence_count[dim] / max(self.total_samples, 1)
            
            for other_dim in active:
                if other_dim >= self.input_dim or other_dim == dim:
                    continue
                
                p_other = self.occurrence_count[other_dim] / max(self.total_samples, 1)
                p_joint = self.cooccurrence[dim, other_dim] / max(self.total_samples, 1)
                
                if p_dim > 0 and p_other > 0 and p_joint > 0:
                    pmi = np.log(p_joint / (p_dim * p_other) + 1e-10)
                    if pmi > 0:  # Positive association
                        pos[idx] += 0.1 * pmi * compositional_embedding[other_dim]
        
        return pos


# =============================================================================
# LAYER 4: GLOBAL STRUCTURE
# =============================================================================

class GlobalLayer:
    """
    Encodes global structure - relationships between domains.
    
    This layer captures the "big picture" - how different regions
    of knowledge space relate to each other.
    """
    
    def __init__(self, input_dim: int = 16, output_dim: int = 8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Domain prototypes (discovered during ingestion)
        self.prototypes: List[np.ndarray] = []
        self.prototype_labels: List[str] = []
        self.max_prototypes = output_dim
    
    def update_prototypes(self, contextual_embedding: np.ndarray, label: str = None):
        """Update domain prototypes based on new sample."""
        if len(self.prototypes) == 0:
            self.prototypes.append(contextual_embedding.copy())
            self.prototype_labels.append(label or f"proto_{len(self.prototypes)}")
            return
        
        # Find nearest prototype
        min_dist = np.inf
        nearest_idx = 0
        for i, proto in enumerate(self.prototypes):
            dist = np.sqrt(np.sum((contextual_embedding - proto) ** 2))
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # If close enough, update prototype (running average)
        if min_dist < 2.0:
            alpha = 0.1  # Learning rate
            self.prototypes[nearest_idx] = (
                (1 - alpha) * self.prototypes[nearest_idx] + 
                alpha * contextual_embedding
            )
        elif len(self.prototypes) < self.max_prototypes:
            # Create new prototype
            self.prototypes.append(contextual_embedding.copy())
            self.prototype_labels.append(label or f"proto_{len(self.prototypes)}")
    
    def encode(self, contextual_embedding: np.ndarray) -> np.ndarray:
        """Encode position relative to global prototypes."""
        pos = np.zeros(self.output_dim)
        
        if not self.prototypes:
            return pos
        
        # Distance to each prototype
        for i, proto in enumerate(self.prototypes):
            if i >= self.output_dim:
                break
            dist = np.sqrt(np.sum((contextual_embedding - proto) ** 2))
            # Similarity (inverse distance)
            pos[i] = 1.0 / (1.0 + dist)
        
        return pos


# =============================================================================
# STACKED GEOMETRIC LCM
# =============================================================================

class StackedGeometricLCM:
    """
    Combines all layers into a unified embedding system.
    
    The final embedding is the concatenation of all layer outputs,
    providing multi-scale representation without training.
    """
    
    def __init__(self):
        # Initialize layers
        self.morphological = MorphologicalLayer(dimensions=16)
        self.lexical = LexicalLayer(dimensions=24)
        self.compositional = CompositionalLayer(input_dim=24, output_dim=16)
        self.contextual = ContextualLayer(input_dim=16, output_dim=16)
        self.global_layer = GlobalLayer(input_dim=16, output_dim=8)
        
        # Total embedding dimension
        self.embedding_dim = 16 + 24 + 16 + 16 + 8  # = 80
        
        # Knowledge storage
        self.points: Dict[str, Tuple[str, np.ndarray]] = {}  # id -> (content, embedding)
        
        # Clustering
        self.clusters: Dict[int, Dict] = {}
        self.cluster_threshold = 0.80  # Balance between too many and too few clusters
    
    def encode(self, text: str, update_stats: bool = True) -> np.ndarray:
        """
        Generate full hierarchical embedding.
        
        Concatenates outputs from all layers.
        """
        # Layer 0: Morphological
        morph = self.morphological.encode(text)
        
        # Layer 1: Lexical
        lex = self.lexical.encode(text)
        
        # Layer 2: Compositional
        comp = self.compositional.encode(lex)
        
        # Layer 3: Contextual (update statistics if training)
        if update_stats:
            self.contextual.update_statistics(comp)
        ctx = self.contextual.encode(comp)
        
        # Layer 4: Global
        if update_stats:
            self.global_layer.update_prototypes(ctx)
        glob = self.global_layer.encode(ctx)
        
        # Concatenate all layers
        embedding = np.concatenate([morph, lex, comp, ctx, glob])
        
        return embedding
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between embeddings."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)
    
    def ingest(self, content: str, description: str = None) -> str:
        """Ingest knowledge and return point ID."""
        if description is None:
            description = content
        
        point_id = hashlib.md5(content.encode()).hexdigest()[:12]
        embedding = self.encode(description, update_stats=True)
        
        self.points[point_id] = (content, embedding)
        return point_id
    
    def ingest_batch(self, items: List[Dict[str, str]]):
        """Ingest multiple items."""
        for item in items:
            self.ingest(item.get('content', ''), item.get('description'))
        
        # Cluster after batch
        self._cluster()
    
    def _cluster(self):
        """Cluster points using agglomerative approach."""
        if len(self.points) < 2:
            return
        
        self.clusters.clear()
        
        # Simple agglomerative clustering
        point_list = list(self.points.items())
        assignments = {pid: i for i, (pid, _) in enumerate(point_list)}
        centroids = {i: emb.copy() for i, (_, (_, emb)) in enumerate(point_list)}
        members = {i: [pid] for i, (pid, _) in enumerate(point_list)}
        
        merged = True
        while merged:
            merged = False
            cluster_ids = list(centroids.keys())
            
            best_sim = -1
            best_pair = None
            
            for i, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[i+1:]:
                    sim = self.cosine_similarity(centroids[c1], centroids[c2])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1, c2)
            
            if best_pair and best_sim >= self.cluster_threshold:
                c1, c2 = best_pair
                members[c1].extend(members[c2])
                
                embeddings = [self.points[pid][1] for pid in members[c1]]
                centroids[c1] = np.mean(embeddings, axis=0)
                
                del centroids[c2]
                del members[c2]
                merged = True
        
        # Create final clusters
        for i, (cid, pids) in enumerate(members.items()):
            # Generate label from common words
            words = defaultdict(int)
            for pid in pids:
                content = self.points[pid][0]
                for w in re.findall(r'\b\w+\b', content.lower()):
                    if len(w) > 3:
                        words[w] += 1
            
            label = max(words, key=words.get) if words else f"cluster_{i}"
            
            self.clusters[i] = {
                'label': label.upper(),
                'centroid': centroids[cid],
                'points': pids
            }
    
    def resolve(self, query: str) -> Tuple[str, float, str]:
        """Resolve query to best matching content."""
        query_emb = self.encode(query, update_stats=False)
        
        best_content = None
        best_sim = -1
        best_cluster = None
        
        for pid, (content, emb) in self.points.items():
            sim = self.cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_content = content
                
                # Find cluster
                for cid, cluster in self.clusters.items():
                    if pid in cluster['points']:
                        best_cluster = cluster['label']
                        break
        
        return best_content, best_sim, best_cluster
    
    def visualize(self) -> str:
        """Text visualization."""
        lines = ["=" * 60, "STACKED GEOMETRIC LCM", "=" * 60, ""]
        lines.append(f"Embedding dimensions: {self.embedding_dim}")
        lines.append(f"  - Morphological: 16")
        lines.append(f"  - Lexical: 24")
        lines.append(f"  - Compositional: 16")
        lines.append(f"  - Contextual: 16")
        lines.append(f"  - Global: 8")
        lines.append(f"\nPoints: {len(self.points)}")
        lines.append(f"Clusters: {len(self.clusters)}")
        lines.append(f"Global prototypes: {len(self.global_layer.prototypes)}")
        
        lines.append("\n--- EMERGENT CLUSTERS ---")
        for cid, cluster in sorted(self.clusters.items(), key=lambda x: -len(x[1]['points'])):
            lines.append(f"\n{cluster['label']} ({len(cluster['points'])} points)")
            for pid in cluster['points'][:3]:
                content = self.points[pid][0]
                lines.append(f"  • {content[:50]}...")
            if len(cluster['points']) > 3:
                lines.append(f"  ... and {len(cluster['points']) - 3} more")
        
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Initializing Stacked Geometric LCM (NO external embeddings)...\n")
    
    lcm = StackedGeometricLCM()
    
    # Knowledge WITHOUT domain labels
    knowledge = [
        # Cooking
        {"content": "chop onions finely", "description": "chop cut onions knife cooking food preparation"},
        {"content": "boil water in pot", "description": "boil water pot heat cooking stove"},
        {"content": "season with salt and pepper", "description": "season salt pepper taste cooking spices"},
        {"content": "preheat oven to 350", "description": "preheat oven temperature heat baking cooking"},
        {"content": "simmer sauce for 20 minutes", "description": "simmer sauce heat low cooking time"},
        {"content": "mix ingredients together", "description": "mix combine ingredients bowl cooking blend"},
        
        # Tech/Bash
        {"content": "ls -la", "description": "list files directory show all terminal command"},
        {"content": "mkdir new_folder", "description": "create make directory folder new terminal command"},
        {"content": "rm -rf directory", "description": "delete remove files directory force terminal"},
        {"content": "grep pattern file", "description": "search find pattern text file terminal"},
        {"content": "cat file.txt", "description": "read show file contents display terminal"},
        {"content": "ps aux", "description": "list show process running system terminal"},
        
        # Social
        {"content": "Hello! How can I help you?", "description": "hello greeting help assist welcome"},
        {"content": "Thank you so much!", "description": "thanks gratitude appreciate thankful"},
        {"content": "I understand how you feel", "description": "understand feeling empathy emotion support"},
        {"content": "Take care of yourself!", "description": "care goodbye feeling help wellbeing"},
    ]
    
    print("Ingesting knowledge (NO DOMAIN LABELS, NO EXTERNAL LLM)...")
    lcm.ingest_batch(knowledge)
    
    print("\n" + lcm.visualize())
    
    print("\n" + "=" * 60)
    print("RESOLUTION TEST")
    print("=" * 60)
    
    queries = [
        "how do I cut vegetables",
        "dice the carrots finely",
        "show me the files in this folder",
        "remove that directory",
        "hi there, how are you",
        "thanks for helping me",
    ]
    
    for query in queries:
        content, sim, cluster = lcm.resolve(query)
        print(f"\n\"{query}\"")
        print(f"  → {content[:40]}... (sim: {sim:.3f})")
        print(f"  Cluster: {cluster}")
    
    # Show embedding breakdown for one example
    print("\n" + "=" * 60)
    print("EMBEDDING BREAKDOWN: 'chop onions finely'")
    print("=" * 60)
    
    text = "chop onions finely"
    morph = lcm.morphological.encode(text)
    lex = lcm.lexical.encode(text)
    comp = lcm.compositional.encode(lex)
    ctx = lcm.contextual.encode(comp)
    glob = lcm.global_layer.encode(ctx)
    
    print(f"Morphological (16D): {np.sum(morph > 0.1)} active dims")
    print(f"Lexical (24D): {np.sum(lex > 0.1)} active dims")
    print(f"Compositional (16D): {np.sum(comp > 0.1)} active dims")
    print(f"Contextual (16D): {np.sum(ctx > 0.1)} active dims")
    print(f"Global (8D): {np.sum(glob > 0.1)} active dims")
