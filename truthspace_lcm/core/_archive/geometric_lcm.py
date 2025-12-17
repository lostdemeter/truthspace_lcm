#!/usr/bin/env python3
"""
Geometric LCM - Truly Dynamic Knowledge Organization

This addresses the four fundamental challenges:
1. OVERLAP: Soft membership - entries exist in multiple regions with weights
2. DYNAMIC CREATION: Online novelty detection creates regions on-the-fly
3. CONTEXT SWITCHING: Trajectory tracking detects discontinuities
4. GEOMETRIC: Everything is distance/density in hyperdimensional space

Key insight: We don't define domains. We define NOTHING.
Structure emerges purely from the geometry of encoded knowledge.
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
# CORE STRUCTURES
# =============================================================================

@dataclass
class Primitive:
    name: str
    dimension: int
    level: int
    keywords: Set[str]
    
    @property
    def activation_value(self) -> float:
        return PHI ** self.level


@dataclass
class KnowledgePoint:
    """
    A point in truth space.
    
    No domain, no cluster assignment - just a position and content.
    Membership in regions is computed dynamically based on distance.
    """
    id: str
    content: str
    description: str
    position: np.ndarray
    
    # Metadata for analysis
    created_at: int = 0  # Ingestion order


@dataclass
class DensityPeak:
    """
    A peak in the density landscape of truth space.
    
    These emerge naturally from the distribution of knowledge points.
    They are NOT pre-defined domains - they are discovered.
    """
    id: int
    position: np.ndarray
    density: float = 0.0
    # Points that are "attracted" to this peak (soft membership)
    attracted_points: Dict[str, float] = field(default_factory=dict)  # id -> weight
    # Emergent label (optional)
    label: Optional[str] = None


# =============================================================================
# SEED PRIMITIVES
# =============================================================================

SEED_PRIMITIVES = [
    Primitive("CREATE", 0, 0, {"create", "make", "new", "generate", "build"}),
    Primitive("DESTROY", 0, 1, {"destroy", "delete", "remove", "eliminate"}),
    Primitive("TRANSFORM", 0, 2, {"change", "transform", "convert", "modify"}),
    Primitive("MOVE", 0, 3, {"move", "transfer", "relocate", "shift"}),
    
    Primitive("READ", 1, 0, {"read", "get", "retrieve", "fetch", "see", "view", "show", "list", "display"}),
    Primitive("WRITE", 1, 1, {"write", "set", "store", "save", "record"}),
    Primitive("SEARCH", 1, 2, {"search", "find", "locate", "seek", "look", "grep"}),
    Primitive("COMPARE", 1, 3, {"compare", "match", "differ", "contrast"}),
    
    Primitive("CONNECT", 2, 0, {"connect", "link", "join", "attach"}),
    Primitive("SEPARATE", 2, 1, {"separate", "split", "divide", "disconnect"}),
    Primitive("COMBINE", 2, 2, {"combine", "merge", "unite", "mix", "blend"}),
    Primitive("FILTER", 2, 3, {"filter", "select", "choose", "pick"}),
    
    Primitive("INTO", 3, 0, {"into", "to", "toward", "inside"}),
    Primitive("FROM", 3, 1, {"from", "out", "away", "source"}),
    Primitive("WITH", 3, 2, {"with", "using", "by", "through"}),
    Primitive("ABOUT", 3, 3, {"about", "regarding", "concerning"}),
    
    Primitive("SEQUENCE", 4, 0, {"sequence", "list", "series", "order", "step"}),
    Primitive("HIERARCHY", 4, 1, {"hierarchy", "tree", "nested", "parent", "child"}),
    Primitive("NETWORK", 4, 2, {"network", "graph", "connected", "web"}),
    Primitive("COLLECTION", 4, 3, {"collection", "set", "group", "batch", "all"}),
    
    Primitive("GREETING", 5, 0, {"hello", "hi", "hey", "greetings", "welcome"}),
    Primitive("GRATITUDE", 5, 1, {"thanks", "thank", "grateful", "appreciate"}),
    Primitive("FEELING", 5, 2, {"feel", "feeling", "emotion", "mood"}),
    Primitive("HELP", 5, 3, {"help", "assist", "support", "aid"}),
    
    Primitive("FOOD", 6, 0, {"food", "cook", "cooking", "eat", "meal", "recipe"}),
    Primitive("HEAT", 6, 1, {"heat", "hot", "warm", "temperature", "boil", "bake"}),
    Primitive("CUT", 6, 2, {"cut", "chop", "slice", "dice", "knife"}),
    Primitive("TASTE", 6, 3, {"taste", "flavor", "season", "salt", "pepper"}),
    
    Primitive("FILE", 7, 0, {"file", "files", "document", "data"}),
    Primitive("DIRECTORY", 7, 1, {"directory", "folder", "path", "dir"}),
    Primitive("PROCESS", 7, 2, {"process", "running", "pid", "kill", "ps"}),
    Primitive("SYSTEM", 7, 3, {"system", "computer", "machine", "server"}),
]


# =============================================================================
# GEOMETRIC LCM
# =============================================================================

class GeometricLCM:
    """
    A purely geometric approach to knowledge organization.
    
    Core principles:
    1. Everything is a point in truth space
    2. Structure emerges from density (clustering of points)
    3. Context is a trajectory through space
    4. Membership is soft (based on distance, not assignment)
    """
    
    def __init__(self, dimensions: int = 12):
        self.dimensions = dimensions
        self.primitives = list(SEED_PRIMITIVES)
        self.points: Dict[str, KnowledgePoint] = {}
        self.peaks: Dict[int, DensityPeak] = {}
        self.next_peak_id = 0
        self.ingestion_counter = 0
        
        # Density parameters
        self.density_radius = 2.0  # Radius for density estimation
        self.novelty_threshold = 0.1  # Below this density = novel point
        self.peak_merge_distance = 1.0  # Peaks closer than this merge
        
        # Context tracking
        self.trajectory: List[np.ndarray] = []
        self.trajectory_window = 5
        self.current_velocity: Optional[np.ndarray] = None
        
        # Word statistics for emergent primitives
        self.word_freq: Dict[str, int] = defaultdict(int)
        
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        self.kw_to_prim: Dict[str, List[Primitive]] = defaultdict(list)
        for p in self.primitives:
            for kw in p.keywords:
                self.kw_to_prim[kw].append(p)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to position in truth space.
        
        Uses TWO encoding strategies:
        1. Primitive-based (dimensions 0-7): Semantic structure
        2. Word-hash-based (dimensions 8+): Content discrimination
        
        The word-hash encoding ensures different content words
        activate different dimensions, providing discrimination.
        """
        pos = np.zeros(self.dimensions)
        words = self._tokenize(text)
        
        for i, word in enumerate(words):
            decay = PHI ** (-i / 2)
            
            # Strategy 1: Primitive activation (semantic structure)
            for prim in self.kw_to_prim.get(word, []):
                val = prim.activation_value * decay
                if prim.dimension < 8:  # First 8 dims for primitives
                    pos[prim.dimension] = max(pos[prim.dimension], val)
            
            # Strategy 2: Word-hash activation (content discrimination)
            # Each unique word activates a dimension based on its hash
            if len(word) > 2:  # Skip very short words
                word_hash = hash(word) % (self.dimensions - 8)
                dim = 8 + word_hash
                if dim < self.dimensions:
                    pos[dim] = max(pos[dim], decay)
        
        return pos
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def similarity(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return 1.0 / (1.0 + self.distance(p1, p2))
    
    # =========================================================================
    # DENSITY ESTIMATION
    # =========================================================================
    
    def estimate_density(self, position: np.ndarray) -> float:
        """
        Estimate local density at a position.
        
        Density = sum of Gaussian weights from nearby points.
        High density = many similar knowledge points nearby.
        """
        if not self.points:
            return 0.0
        
        density = 0.0
        for point in self.points.values():
            dist = self.distance(position, point.position)
            # Gaussian kernel
            weight = np.exp(-(dist ** 2) / (2 * self.density_radius ** 2))
            density += weight
        
        return density
    
    def is_novel(self, position: np.ndarray) -> bool:
        """Check if a position represents novel knowledge."""
        density = self.estimate_density(position)
        return density < self.novelty_threshold
    
    # =========================================================================
    # INGESTION - Truly dynamic
    # =========================================================================
    
    def ingest(self, content: str, description: str = None) -> Tuple[KnowledgePoint, bool, Optional[DensityPeak]]:
        """
        Ingest knowledge and let structure emerge.
        
        Returns: (point, is_novel, nearest_peak)
        """
        if description is None:
            description = content
        
        point_id = hashlib.md5(content.encode()).hexdigest()[:12]
        position = self.encode(description)
        
        # Check novelty BEFORE adding
        is_novel = self.is_novel(position)
        
        # Create point
        point = KnowledgePoint(
            id=point_id,
            content=content,
            description=description,
            position=position,
            created_at=self.ingestion_counter
        )
        self.ingestion_counter += 1
        
        # Update word stats
        for word in self._tokenize(description):
            self.word_freq[word] += 1
        
        # Add to space
        self.points[point_id] = point
        
        # Update density landscape
        if is_novel:
            # This point might become a new peak
            peak = self._maybe_create_peak(position)
        else:
            peak = self._find_nearest_peak(position)
        
        # Update peak attractions
        self._update_attractions()
        
        return point, is_novel, peak
    
    def _maybe_create_peak(self, position: np.ndarray) -> Optional[DensityPeak]:
        """Create a new density peak if position is far from existing peaks."""
        # Check distance to existing peaks
        for peak in self.peaks.values():
            if self.distance(position, peak.position) < self.peak_merge_distance:
                return peak  # Too close to existing peak
        
        # Create new peak
        peak_id = self.next_peak_id
        self.next_peak_id += 1
        
        peak = DensityPeak(
            id=peak_id,
            position=position.copy(),
            density=self.estimate_density(position)
        )
        
        self.peaks[peak_id] = peak
        return peak
    
    def _find_nearest_peak(self, position: np.ndarray) -> Optional[DensityPeak]:
        """Find the peak nearest to a position."""
        if not self.peaks:
            return None
        
        best_peak = None
        best_dist = np.inf
        
        for peak in self.peaks.values():
            dist = self.distance(position, peak.position)
            if dist < best_dist:
                best_dist = dist
                best_peak = peak
        
        return best_peak
    
    def _update_attractions(self):
        """Update which points are attracted to which peaks."""
        for peak in self.peaks.values():
            peak.attracted_points.clear()
            peak.density = self.estimate_density(peak.position)
        
        for point in self.points.values():
            # Compute attraction to each peak (soft membership)
            total_attraction = 0.0
            attractions = {}
            
            for peak in self.peaks.values():
                dist = self.distance(point.position, peak.position)
                # Attraction falls off with distance
                attraction = np.exp(-dist / self.density_radius)
                attractions[peak.id] = attraction
                total_attraction += attraction
            
            # Normalize and assign
            if total_attraction > 0:
                for peak_id, attraction in attractions.items():
                    weight = attraction / total_attraction
                    if weight > 0.1:  # Only track significant attractions
                        self.peaks[peak_id].attracted_points[point.id] = weight
        
        # Label peaks based on common words
        self._label_peaks()
    
    def _label_peaks(self):
        """Generate labels for peaks based on attracted points."""
        covered_words = set()
        for p in self.primitives:
            covered_words.update(p.keywords)
        
        for peak in self.peaks.values():
            word_scores = defaultdict(float)
            
            for point_id, weight in peak.attracted_points.items():
                point = self.points[point_id]
                for word in self._tokenize(point.description):
                    if len(word) > 3 and word not in covered_words:
                        word_scores[word] += weight
            
            if word_scores:
                best_word = max(word_scores, key=word_scores.get)
                peak.label = best_word.upper()
    
    # =========================================================================
    # CONTEXT TRACKING - Trajectory through space
    # =========================================================================
    
    def update_trajectory(self, position: np.ndarray):
        """Update the trajectory through truth space."""
        self.trajectory.append(position.copy())
        
        if len(self.trajectory) > self.trajectory_window:
            self.trajectory.pop(0)
        
        # Compute velocity (direction of movement)
        if len(self.trajectory) >= 2:
            self.current_velocity = self.trajectory[-1] - self.trajectory[-2]
    
    def detect_discontinuity(self, new_position: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if new position represents a discontinuity (context switch).
        
        A discontinuity is a sudden jump in position that doesn't follow
        the current trajectory.
        """
        if len(self.trajectory) < 2:
            return False, 0.0
        
        # Predict where we'd go based on current velocity
        if self.current_velocity is not None:
            predicted = self.trajectory[-1] + self.current_velocity
            prediction_error = self.distance(new_position, predicted)
        else:
            prediction_error = 0.0
        
        # Also check absolute jump distance
        jump_distance = self.distance(new_position, self.trajectory[-1])
        
        # Average recent movement
        recent_movements = []
        for i in range(1, len(self.trajectory)):
            recent_movements.append(self.distance(self.trajectory[i], self.trajectory[i-1]))
        avg_movement = np.mean(recent_movements) if recent_movements else 0.0
        
        # Discontinuity if jump is much larger than average
        is_discontinuity = jump_distance > 2 * (avg_movement + 0.5)
        
        return is_discontinuity, jump_distance
    
    def get_context_peak(self) -> Optional[DensityPeak]:
        """Get the peak we're currently near in the trajectory."""
        if not self.trajectory:
            return None
        
        current_pos = self.trajectory[-1]
        return self._find_nearest_peak(current_pos)
    
    # =========================================================================
    # RESOLUTION - Context-aware, geometry-based
    # =========================================================================
    
    def resolve(self, query: str) -> Tuple[KnowledgePoint, float, Dict[str, Any]]:
        """
        Resolve a query to the best matching knowledge point.
        
        Uses:
        1. Direct geometric similarity
        2. Context (trajectory) bias
        3. Peak membership weighting
        """
        query_pos = self.encode(query)
        
        # Detect context switch
        is_switch, jump_dist = self.detect_discontinuity(query_pos)
        
        # Get current context peak
        context_peak = self.get_context_peak()
        
        best_point = None
        best_score = -np.inf
        
        for point in self.points.values():
            # Base similarity
            sim = self.similarity(query_pos, point.position)
            score = sim
            
            # Context bonus: if we're in a peak and point is attracted to same peak
            if context_peak and not is_switch:
                if point.id in context_peak.attracted_points:
                    attraction = context_peak.attracted_points[point.id]
                    score *= (1 + 0.3 * attraction)  # Up to 30% bonus
            
            if score > best_score:
                best_score = score
                best_point = point
        
        # Update trajectory
        self.update_trajectory(query_pos)
        
        # Build debug info
        result_peak = self._find_nearest_peak(best_point.position) if best_point else None
        debug = {
            'is_context_switch': is_switch,
            'jump_distance': jump_dist,
            'context_peak': context_peak.label if context_peak else None,
            'result_peak': result_peak.label if result_peak else None,
            'query_density': self.estimate_density(query_pos),
        }
        
        return best_point, best_score, debug
    
    # =========================================================================
    # EMERGENT PRIMITIVES
    # =========================================================================
    
    def discover_primitives(self, min_freq: int = 3):
        """Discover new primitives from frequent words."""
        covered = set()
        for p in self.primitives:
            covered.update(p.keywords)
        
        new_prims = []
        for word, freq in self.word_freq.items():
            if word not in covered and freq >= min_freq and len(word) > 3:
                dim = 8 + len(new_prims) // 4
                level = len(new_prims) % 4
                
                if dim < self.dimensions:
                    prim = Primitive(
                        name=f"E_{word.upper()}",
                        dimension=dim,
                        level=level,
                        keywords={word}
                    )
                    new_prims.append(prim)
                    covered.add(word)
        
        self.primitives.extend(new_prims)
        self._build_keyword_index()
        
        # Re-encode all points
        for point in self.points.values():
            point.position = self.encode(point.description)
        
        # Update peaks
        self._update_attractions()
        
        return new_prims
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize(self) -> str:
        """Text visualization of the knowledge landscape."""
        lines = ["=" * 60, "GEOMETRIC KNOWLEDGE LANDSCAPE", "=" * 60, ""]
        
        lines.append(f"Points: {len(self.points)}")
        lines.append(f"Density Peaks: {len(self.peaks)}")
        lines.append(f"Dimensions: {self.dimensions}")
        lines.append("")
        
        # Show peaks
        lines.append("--- EMERGENT PEAKS ---")
        for peak in sorted(self.peaks.values(), key=lambda p: -len(p.attracted_points)):
            label = peak.label or f"Peak_{peak.id}"
            n_points = len(peak.attracted_points)
            lines.append(f"\n{label} ({n_points} attracted points, density={peak.density:.2f})")
            
            # Show top attracted points
            sorted_points = sorted(peak.attracted_points.items(), key=lambda x: -x[1])
            for point_id, weight in sorted_points[:3]:
                point = self.points[point_id]
                lines.append(f"  [{weight:.2f}] {point.content[:50]}...")
        
        # Show trajectory
        if self.trajectory:
            lines.append("\n--- CURRENT CONTEXT ---")
            context_peak = self.get_context_peak()
            if context_peak:
                lines.append(f"Near peak: {context_peak.label or f'Peak_{context_peak.id}'}")
            lines.append(f"Trajectory length: {len(self.trajectory)}")
        
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    lcm = GeometricLCM(dimensions=32)  # More dimensions for word-hash discrimination
    
    print("Ingesting knowledge (NO LABELS)...\n")
    
    # Mix of different "domains" - but we don't tell the system that
    knowledge = [
        # Cooking-ish
        {"content": "chop onions finely", "description": "chop cut onions knife cooking food"},
        {"content": "boil water in pot", "description": "boil water pot heat cooking food"},
        {"content": "season with salt", "description": "season salt pepper taste cooking food"},
        {"content": "preheat oven to 350", "description": "preheat oven temperature heat baking food"},
        {"content": "simmer for 20 minutes", "description": "simmer heat low cooking time food"},
        {"content": "mix ingredients together", "description": "mix combine ingredients bowl cooking"},
        
        # Tech-ish
        {"content": "ls -la", "description": "list files directory show all system"},
        {"content": "mkdir new_folder", "description": "create make directory folder new system"},
        {"content": "rm -rf", "description": "delete remove files directory force system"},
        {"content": "grep pattern file", "description": "search find pattern text file system"},
        {"content": "cat file.txt", "description": "read show file contents display system"},
        {"content": "ps aux", "description": "list show process running system"},
        
        # Social-ish
        {"content": "Hello! How can I help?", "description": "hello greeting help assist"},
        {"content": "Thank you so much!", "description": "thanks gratitude appreciate feeling"},
        {"content": "I understand how you feel", "description": "understand feeling empathy emotion help"},
        {"content": "Take care of yourself!", "description": "care goodbye feeling help"},
        {"content": "That sounds frustrating", "description": "frustrating feeling empathy understand"},
    ]
    
    for item in knowledge:
        point, is_novel, peak = lcm.ingest(item["content"], item["description"])
        status = "NOVEL" if is_novel else "familiar"
        peak_name = peak.label if peak and peak.label else f"Peak_{peak.id}" if peak else "None"
        print(f"  [{status}] {item['content'][:30]}... → {peak_name}")
    
    # Discover emergent primitives
    print("\nDiscovering emergent primitives...")
    new_prims = lcm.discover_primitives(min_freq=2)
    print(f"  Found {len(new_prims)} new primitives")
    
    print("\n" + lcm.visualize())
    
    print("\n" + "=" * 60)
    print("RESOLUTION TEST (with context tracking)")
    print("=" * 60)
    
    queries = [
        "how do I cut vegetables",
        "what about dicing onions",  # Should stay in cooking context
        "show me the files",  # Context switch!
        "delete that folder",  # Should stay in tech context
        "hello there",  # Context switch!
        "thanks for your help",  # Should stay in social context
    ]
    
    for query in queries:
        point, score, debug = lcm.resolve(query)
        switch = "⚡ SWITCH" if debug['is_context_switch'] else ""
        print(f"\n\"{query}\" {switch}")
        print(f"  → {point.content[:40]}...")
        print(f"  Context: {debug['context_peak']} → {debug['result_peak']}")
        print(f"  Score: {score:.2f}, Density: {debug['query_density']:.2f}")
