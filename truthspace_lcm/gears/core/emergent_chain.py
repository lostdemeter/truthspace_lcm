"""
Emergent Dimension Chain

A reusable base class for gear chains that discover their own dimensions
from data via SVD. This is the core abstraction for self-discovering systems.

Both understanding (semantic) and output (linguistic) chains inherit from this.

The key insight: ANY domain can be analyzed by:
1. Extracting features from items
2. Building a feature matrix
3. Discovering dimensions via SVD
4. Using those dimensions for similarity, retrieval, and transformation

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class DimensionInfo:
    """Information about a discovered dimension."""
    index: int
    name: str
    variance: float
    negative_pole: str
    positive_pole: str
    negative_features: List[str]
    positive_features: List[str]
    positions: Dict[str, float]


@dataclass 
class DataItem:
    """A data item with features and position."""
    id: str
    content: Any
    features: Dict[str, float] = field(default_factory=dict)
    position: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergentDimensionChain(ABC):
    """
    Abstract base class for chains that discover dimensions from data.
    
    Subclasses must implement:
    - extract_features(item) -> Dict[str, float]
    - get_item_id(item) -> str
    
    The base class provides:
    - Data ingestion
    - SVD-based dimension discovery
    - Similarity search
    - Position lookup
    """
    
    def __init__(self, name: str = "EmergentChain"):
        self.name = name
        
        # Data storage
        self.items: List[DataItem] = []
        self.item_index: Dict[str, int] = {}  # id -> index
        
        # Feature aggregation (for grouped items like agents)
        self.grouped_features: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.group_counts: Dict[str, int] = defaultdict(int)
        
        # Discovered dimensions
        self.dimensions: List[DimensionInfo] = []
        self.groups: List[str] = []  # Group IDs (e.g., agent names)
        self.feature_names: List[str] = []
        
        # SVD components
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        # Configuration
        self.min_variance: float = 0.02
        self.max_dimensions: int = 15
        self.min_group_count: int = 3
    
    @abstractmethod
    def extract_features(self, item: Any) -> Dict[str, float]:
        """
        Extract features from a data item.
        
        Subclasses implement this to define what features matter for their domain.
        
        Args:
            item: The raw data item
            
        Returns:
            Dictionary of feature_name -> value
        """
        pass
    
    @abstractmethod
    def get_item_id(self, item: Any) -> str:
        """
        Get a unique identifier for grouping items.
        
        For understanding chains, this might be the agent name.
        For output chains, this might be a sentence hash or template ID.
        
        Args:
            item: The raw data item
            
        Returns:
            String identifier
        """
        pass
    
    def get_item_content(self, item: Any) -> Any:
        """
        Get the content to store for an item.
        
        Override in subclasses if needed.
        """
        return item
    
    def ingest_item(self, item: Any) -> Optional[DataItem]:
        """Ingest a single item."""
        item_id = self.get_item_id(item)
        if not item_id:
            return None
        
        features = self.extract_features(item)
        content = self.get_item_content(item)
        
        data_item = DataItem(
            id=item_id,
            content=content,
            features=features,
        )
        
        self.items.append(data_item)
        self.item_index[item_id] = len(self.items) - 1
        
        # Aggregate features by group
        for feat, val in features.items():
            self.grouped_features[item_id][feat] += val
        self.group_counts[item_id] += 1
        
        return data_item
    
    def ingest_batch(self, items: List[Any]) -> int:
        """Ingest a batch of items. Returns count of successfully ingested."""
        count = 0
        for item in items:
            if self.ingest_item(item):
                count += 1
        return count
    
    def ingest_corpus(self, corpus_path: str, frame_key: str = 'frames') -> int:
        """
        Ingest items from a corpus JSON file.
        
        Args:
            corpus_path: Path to JSON file
            frame_key: Key in JSON containing the list of items
            
        Returns:
            Count of ingested items
        """
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        items = corpus.get(frame_key, [])
        return self.ingest_batch(items)
    
    def learn_dimensions(self, 
                         min_variance: Optional[float] = None,
                         max_dims: Optional[int] = None) -> int:
        """
        Discover dimensions from the ingested data via SVD.
        
        Returns:
            Number of dimensions discovered
        """
        if min_variance is not None:
            self.min_variance = min_variance
        if max_dims is not None:
            self.max_dimensions = max_dims
        
        # Filter groups with sufficient data
        valid_groups = {
            g: feats for g, feats in self.grouped_features.items()
            if self.group_counts[g] >= self.min_group_count
        }
        
        if len(valid_groups) < 3:
            return 0
        
        self.groups = list(valid_groups.keys())
        
        # Collect all features
        all_features = set()
        for feats in valid_groups.values():
            all_features.update(feats.keys())
        self.feature_names = sorted(all_features)
        
        n_groups = len(self.groups)
        n_features = len(self.feature_names)
        
        # Build normalized feature matrix
        X = np.zeros((n_groups, n_features))
        for i, group in enumerate(self.groups):
            feats = valid_groups[group]
            total = sum(feats.values())
            if total > 0:
                for j, feat_name in enumerate(self.feature_names):
                    X[i, j] = feats.get(feat_name, 0) / total
        
        # Center and SVD
        X_centered = X - X.mean(axis=0)
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance analysis
        total_var = np.sum(self.S ** 2)
        if total_var < 1e-10:
            return 0
        var_ratios = (self.S ** 2) / total_var
        
        # Discover dimensions
        self.dimensions = []
        cumulative = 0.0
        
        for i in range(min(len(self.S), self.max_dimensions)):
            var = var_ratios[i]
            cumulative += var
            
            if var < self.min_variance:
                break
            
            positions = self.U[:, i]
            min_idx = int(np.argmin(positions))
            max_idx = int(np.argmax(positions))
            
            feature_weights = self.Vt[i]
            sorted_feat_idx = np.argsort(feature_weights)
            neg_features = [self.feature_names[j] for j in sorted_feat_idx[:3]]
            pos_features = [self.feature_names[j] for j in sorted_feat_idx[-3:]]
            
            dim = DimensionInfo(
                index=i,
                name=f"D{i+1}",
                variance=float(var),
                negative_pole=self.groups[min_idx],
                positive_pole=self.groups[max_idx],
                negative_features=neg_features,
                positive_features=pos_features,
                positions={self.groups[j]: float(positions[j]) for j in range(n_groups)},
            )
            self.dimensions.append(dim)
        
        # Update item positions
        self._update_item_positions()
        
        return len(self.dimensions)
    
    def _update_item_positions(self):
        """Update positions for all items based on their group."""
        for item in self.items:
            if item.id in self.groups:
                idx = self.groups.index(item.id)
                item.position = self.U[idx, :len(self.dimensions)]
    
    def get_position(self, group_id: str) -> Optional[np.ndarray]:
        """Get the dimensional position of a group."""
        group_id = group_id.lower() if isinstance(group_id, str) else group_id
        
        # Direct match
        if group_id in self.groups:
            idx = self.groups.index(group_id)
            return self.U[idx, :len(self.dimensions)]
        
        # Partial match
        for group in self.groups:
            if group_id in group or group in group_id:
                idx = self.groups.index(group)
                return self.U[idx, :len(self.dimensions)]
        
        return None
    
    def find_group(self, query: str) -> Optional[str]:
        """Find a group ID matching a query string."""
        query = query.lower()
        
        if query in self.groups:
            return query
        
        for group in self.groups:
            if query in group or group in query:
                return group
        
        return None
    
    def find_similar(self, group_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find groups similar to the given group."""
        pos = self.get_position(group_id)
        if pos is None:
            return []
        
        results = []
        for other in self.groups:
            if other != group_id:
                other_pos = self.get_position(other)
                if other_pos is not None:
                    dist = float(np.linalg.norm(pos - other_pos))
                    results.append((other, dist))
        
        return sorted(results, key=lambda x: x[1])[:k]
    
    def find_opposite(self, group_id: str) -> Optional[Tuple[str, float]]:
        """Find the most opposite group."""
        pos = self.get_position(group_id)
        if pos is None:
            return None
        
        max_dist = 0.0
        opposite = None
        
        for other in self.groups:
            if other != group_id:
                other_pos = self.get_position(other)
                if other_pos is not None:
                    dist = float(np.linalg.norm(pos - other_pos))
                    if dist > max_dist:
                        max_dist = dist
                        opposite = other
        
        return (opposite, max_dist) if opposite else None
    
    def find_items_near(self, target_pos: np.ndarray, k: int = 5) -> List[DataItem]:
        """Find items closest to a target position."""
        scored = []
        
        for item in self.items:
            if item.position is not None:
                dist = float(np.linalg.norm(item.position - target_pos[:len(item.position)]))
                scored.append((dist, item))
        
        scored.sort(key=lambda x: x[0])
        return [item for _, item in scored[:k]]
    
    def find_items_for_groups(self, group_ids: List[str], k: int = 5) -> List[DataItem]:
        """Find items relevant to a list of groups."""
        positions = [self.get_position(g) for g in group_ids]
        positions = [p for p in positions if p is not None]
        
        if not positions:
            # Fallback to text matching
            results = []
            for item in self.items:
                score = sum(1 for g in group_ids if g in str(item.content).lower())
                if score > 0:
                    results.append((score, item))
            results.sort(key=lambda x: -x[0])
            return [item for _, item in results[:k]]
        
        avg_pos = np.mean(positions, axis=0)
        return self.find_items_near(avg_pos, k)
    
    def get_dimension_info(self) -> List[Dict[str, Any]]:
        """Get information about discovered dimensions."""
        return [
            {
                'name': d.name,
                'variance': d.variance,
                'negative_pole': d.negative_pole,
                'positive_pole': d.positive_pole,
                'negative_features': d.negative_features,
                'positive_features': d.positive_features,
            }
            for d in self.dimensions
        ]
    
    def __repr__(self) -> str:
        return f"{self.name}(items={len(self.items)}, groups={len(self.groups)}, dims={len(self.dimensions)})"
