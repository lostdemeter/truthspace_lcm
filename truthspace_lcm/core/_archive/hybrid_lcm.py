#!/usr/bin/env python3
"""
Hybrid Geometric LCM

The insight: We need GOOD EMBEDDINGS for geometric clustering to work.

This hybrid approach:
1. Uses an embedding model (via Ollama) to get high-quality vectors
2. Applies geometric operations (clustering, trajectory) on those vectors
3. Discovers structure without pre-defined domains

This is the bridge between:
- Traditional LLMs (learned embeddings, black box)
- Pure geometric LCM (hand-coded primitives, interpretable)

The embedding provides discrimination, the geometry provides structure.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import requests
import json
import hashlib


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KnowledgePoint:
    """A point in embedding space."""
    id: str
    content: str
    description: str
    embedding: np.ndarray = None
    cluster_id: Optional[int] = None


@dataclass
class Cluster:
    """An emergent cluster - discovered, not defined."""
    id: int
    centroid: np.ndarray
    points: List[str] = field(default_factory=list)
    label: Optional[str] = None
    radius: float = 0.0


# =============================================================================
# HYBRID LCM
# =============================================================================

class HybridLCM:
    """
    Combines LLM embeddings with geometric structure discovery.
    
    The LLM provides the "what" (semantic similarity).
    The geometry provides the "how" (structure, clustering, context).
    """
    
    def __init__(self, 
                 embedding_model: str = "nomic-embed-text:latest",
                 ollama_url: str = "http://localhost:11434"):
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.embed_url = f"{ollama_url}/api/embeddings"
        
        self.points: Dict[str, KnowledgePoint] = {}
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
        
        # Clustering parameters
        self.cluster_threshold = 0.5  # Cosine similarity threshold (lower = more merging)
        self.min_cluster_size = 2
        
        # Context tracking
        self.trajectory: List[np.ndarray] = []
        self.context_window = 5
        
        # Check embedding model
        self._embedding_dim = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama."""
        try:
            response = requests.post(
                self.embed_url,
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            
            if embedding:
                arr = np.array(embedding)
                if self._embedding_dim is None:
                    self._embedding_dim = len(arr)
                return arr
        except Exception as e:
            print(f"Embedding error: {e}")
        
        # Fallback: random embedding (for testing without Ollama)
        if self._embedding_dim:
            return np.random.randn(self._embedding_dim)
        return np.random.randn(768)  # Default dimension
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    # =========================================================================
    # INGESTION
    # =========================================================================
    
    def ingest(self, content: str, description: str = None) -> KnowledgePoint:
        """Ingest knowledge - NO DOMAIN LABEL."""
        if description is None:
            description = content
        
        point_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Get embedding
        embedding = self._get_embedding(description)
        
        point = KnowledgePoint(
            id=point_id,
            content=content,
            description=description,
            embedding=embedding
        )
        
        self.points[point_id] = point
        return point
    
    def ingest_batch(self, items: List[Dict[str, str]]) -> List[KnowledgePoint]:
        """Ingest multiple items, then cluster."""
        points = []
        for item in items:
            point = self.ingest(
                item.get('content', ''),
                item.get('description', item.get('content', ''))
            )
            points.append(point)
            print(f"  Embedded: {item.get('content', '')[:40]}...")
        
        # Cluster after batch
        self._cluster_all()
        
        return points
    
    # =========================================================================
    # CLUSTERING
    # =========================================================================
    
    def _cluster_all(self):
        """Cluster all points using agglomerative approach."""
        if len(self.points) < 2:
            return
        
        # Reset clusters
        self.clusters.clear()
        self.next_cluster_id = 0
        
        # Start with each point as its own cluster
        point_list = list(self.points.values())
        assignments = {p.id: i for i, p in enumerate(point_list)}
        centroids = {i: p.embedding.copy() for i, p in enumerate(point_list)}
        members = {i: [p.id] for i, p in enumerate(point_list)}
        
        # Merge until no more merges possible
        merged = True
        while merged:
            merged = False
            
            # Find most similar pair of clusters
            best_sim = -1
            best_pair = None
            
            cluster_ids = list(centroids.keys())
            for i, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[i+1:]:
                    sim = self.cosine_similarity(centroids[c1], centroids[c2])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1, c2)
            
            # Merge if above threshold
            if best_pair and best_sim >= self.cluster_threshold:
                c1, c2 = best_pair
                
                # Merge c2 into c1
                members[c1].extend(members[c2])
                
                # Update centroid
                embeddings = [self.points[pid].embedding for pid in members[c1]]
                centroids[c1] = np.mean(embeddings, axis=0)
                
                # Update assignments
                for pid in members[c2]:
                    assignments[pid] = c1
                
                # Remove c2
                del centroids[c2]
                del members[c2]
                
                merged = True
        
        # Create final clusters
        for cluster_id, point_ids in members.items():
            if len(point_ids) >= 1:  # Keep all clusters for now
                embeddings = [self.points[pid].embedding for pid in point_ids]
                centroid = np.mean(embeddings, axis=0)
                
                cluster = Cluster(
                    id=self.next_cluster_id,
                    centroid=centroid,
                    points=point_ids
                )
                
                # Compute radius
                if len(point_ids) > 1:
                    distances = [self.euclidean_distance(self.points[pid].embedding, centroid) 
                                for pid in point_ids]
                    cluster.radius = max(distances)
                
                # Generate label from common words
                self._label_cluster(cluster)
                
                # Update point assignments
                for pid in point_ids:
                    self.points[pid].cluster_id = self.next_cluster_id
                
                self.clusters[self.next_cluster_id] = cluster
                self.next_cluster_id += 1
    
    def _label_cluster(self, cluster: Cluster):
        """Generate a label for a cluster based on content."""
        # Simple: use most common significant word
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'in', 'on', 'for', 'with', 'as', 'at',
                     'by', 'from', 'or', 'that', 'this', 'it', 'how', 'what',
                     'can', 'you', 'your', 'i', 'my', 'me', 'we', 'our'}
        
        for pid in cluster.points:
            point = self.points[pid]
            words = point.description.lower().split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 2 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            best_word = max(word_counts, key=word_counts.get)
            cluster.label = best_word.upper()
    
    # =========================================================================
    # CONTEXT TRACKING
    # =========================================================================
    
    def update_context(self, embedding: np.ndarray):
        """Update trajectory through embedding space."""
        self.trajectory.append(embedding.copy())
        if len(self.trajectory) > self.context_window:
            self.trajectory.pop(0)
    
    def get_context_cluster(self) -> Optional[Cluster]:
        """Get cluster nearest to current context."""
        if not self.trajectory or not self.clusters:
            return None
        
        current = self.trajectory[-1]
        best_cluster = None
        best_sim = -1
        
        for cluster in self.clusters.values():
            sim = self.cosine_similarity(current, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        
        return best_cluster
    
    def detect_context_switch(self, new_embedding: np.ndarray) -> Tuple[bool, Optional[Cluster], Optional[Cluster]]:
        """Detect if new input represents a context switch."""
        old_cluster = self.get_context_cluster()
        
        # Find cluster for new embedding
        new_cluster = None
        best_sim = -1
        for cluster in self.clusters.values():
            sim = self.cosine_similarity(new_embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                new_cluster = cluster
        
        is_switch = (old_cluster is not None and 
                    new_cluster is not None and 
                    old_cluster.id != new_cluster.id)
        
        return is_switch, old_cluster, new_cluster
    
    # =========================================================================
    # RESOLUTION
    # =========================================================================
    
    def resolve(self, query: str) -> Tuple[KnowledgePoint, float, Dict[str, Any]]:
        """Resolve query to best matching point."""
        query_embedding = self._get_embedding(query)
        
        # Detect context switch
        is_switch, old_cluster, new_cluster = self.detect_context_switch(query_embedding)
        
        # Find best matching point
        best_point = None
        best_sim = -1
        
        context_cluster = self.get_context_cluster()
        
        for point in self.points.values():
            sim = self.cosine_similarity(query_embedding, point.embedding)
            
            # Context bonus
            if context_cluster and not is_switch:
                if point.cluster_id == context_cluster.id:
                    sim *= 1.1  # 10% bonus for context-relevant
            
            if sim > best_sim:
                best_sim = sim
                best_point = point
        
        # Update context
        self.update_context(query_embedding)
        
        # Debug info
        result_cluster = self.clusters.get(best_point.cluster_id) if best_point else None
        debug = {
            'is_switch': is_switch,
            'old_cluster': old_cluster.label if old_cluster else None,
            'new_cluster': new_cluster.label if new_cluster else None,
            'result_cluster': result_cluster.label if result_cluster else None,
        }
        
        return best_point, best_sim, debug
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize(self) -> str:
        """Text visualization."""
        lines = ["=" * 60, "HYBRID LCM - EMERGENT CLUSTERS", "=" * 60, ""]
        
        lines.append(f"Points: {len(self.points)}")
        lines.append(f"Clusters: {len(self.clusters)}")
        if self._embedding_dim:
            lines.append(f"Embedding dimensions: {self._embedding_dim}")
        lines.append("")
        
        for cluster in sorted(self.clusters.values(), key=lambda c: -len(c.points)):
            label = cluster.label or f"Cluster_{cluster.id}"
            lines.append(f"\n{label} ({len(cluster.points)} points)")
            lines.append("-" * 40)
            
            for pid in cluster.points[:5]:
                point = self.points[pid]
                lines.append(f"  • {point.content[:50]}...")
            
            if len(cluster.points) > 5:
                lines.append(f"  ... and {len(cluster.points) - 5} more")
        
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Initializing Hybrid LCM with nomic-embed-text...")
    lcm = HybridLCM(embedding_model="nomic-embed-text:latest")
    
    # Knowledge WITHOUT domain labels
    knowledge = [
        # Cooking
        {"content": "chop onions finely", "description": "chop cut onions knife cooking food preparation"},
        {"content": "boil water in pot", "description": "boil water pot heat cooking stove"},
        {"content": "season with salt and pepper", "description": "season salt pepper taste cooking spices"},
        {"content": "preheat oven to 350", "description": "preheat oven temperature baking cooking"},
        {"content": "simmer sauce for 20 minutes", "description": "simmer sauce heat low cooking time"},
        
        # Tech/Bash
        {"content": "ls -la", "description": "list files directory show all terminal command"},
        {"content": "mkdir new_folder", "description": "create make directory folder terminal command"},
        {"content": "rm -rf directory", "description": "delete remove files directory force terminal"},
        {"content": "grep pattern file", "description": "search find pattern text file terminal"},
        {"content": "cat file.txt", "description": "read show file contents display terminal"},
        
        # Social
        {"content": "Hello! How can I help you?", "description": "hello greeting help assist welcome"},
        {"content": "Thank you so much!", "description": "thanks gratitude appreciate thankful"},
        {"content": "I understand how you feel", "description": "understand feeling empathy emotion support"},
        {"content": "Take care of yourself!", "description": "care goodbye wellbeing farewell"},
    ]
    
    print("\nIngesting knowledge (NO DOMAIN LABELS)...")
    lcm.ingest_batch(knowledge)
    
    print("\n" + lcm.visualize())
    
    print("\n" + "=" * 60)
    print("RESOLUTION TEST")
    print("=" * 60)
    
    queries = [
        "how do I cut vegetables",
        "dice the carrots",
        "show me the files in this folder",
        "remove that directory",
        "hi there, how are you",
        "thanks for helping me",
    ]
    
    for query in queries:
        point, sim, debug = lcm.resolve(query)
        switch = "⚡" if debug['is_switch'] else ""
        print(f"\n\"{query}\" {switch}")
        print(f"  → {point.content[:40]}... (sim: {sim:.3f})")
        print(f"  Cluster: {debug['result_cluster']}")
