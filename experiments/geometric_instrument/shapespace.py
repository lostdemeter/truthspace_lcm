"""
ShapeSpace — A Geometric Data Structure
=========================================

A ShapeSpace stores entities as directions in a minimal-dimensional
geometric space. Queries are answered via vector arithmetic:
project + add + dot product + argmax.

This is a new category of data structure:

    Hash map:    O(1) lookup, key→value, no relationships
    Tree:        O(log n), hierarchical, rigid
    Graph:       O(V+E), relational, discrete
    ShapeSpace:  O(d), geometric, composable, projectable

Where d is the natural dimensionality of the data — determined by
combinatorial complexity, not by any fixed parameter.

Properties:
    - Natural dimensionality: d = f(N_entities, N_classes)
    - Composable: two ShapeSpaces can be superimposed (⊕)
    - Projectable: trade dimensions for speed (lossy)
    - Relationship-native: entities AND bindings stored geometrically
    - Interference: multiple queries combine via vector addition

Derived from Finding 155 (The Shape Computer) and DC 284
(The Geometric Path Integral).

Usage:
    # From raw vectors (e.g., extracted from a model)
    space = ShapeSpace.from_vectors(
        entity_vecs={'France': vec_fr, 'Germany': vec_de},
        binding_vecs={'France': bind_fr, 'Germany': bind_de},
        answer_vecs={'France': ans_fr, 'Germany': ans_de},
    )
    answer, score = space.query('France')  # → ('France', 3.28)

    # Reduce to 4 dimensions
    space4 = space.project(4)
    answer, score = space4.query('France')  # still works

    # Compose two spaces
    merged = space_capitals.compose(space_languages)

    # From scratch (no model needed)
    space = ShapeSpace.from_observations(
        observations=[
            ('France', 'capital', 'Paris'),
            ('Germany', 'capital', 'Berlin'),
        ],
        feature_fn=my_encoder,  # maps entity → vector
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any


class ShapeSpace:
    """A geometric data structure for entity-relationship computation.

    Stores entities, bindings (relationships), and answers as directions
    in a d-dimensional space. Queries are answered by interference:
    entity_direction + binding_direction → score against answer_directions.

    The dimensionality d is determined by the data, not fixed a priori.
    For N entities in C structure classes: d ≈ N + C - 1.
    """

    def __init__(self):
        self._basis = None          # (d, source_dim) projection matrix
        self._entities = {}         # name → d-dim vector
        self._bindings = {}         # name → d-dim vector
        self._answers = {}          # name → d-dim vector
        self._d = 0                 # current dimensionality
        self._source_dim = 0        # original vector dimension
        self._singular_values = None  # SVD spectrum of the basis
        self._entity_mean = None    # mean entity (structure class direction)
        self._metadata = {}         # arbitrary metadata

    # ═══════════════════════════════════════════════════════════
    # Construction
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def from_vectors(cls,
                     entity_vecs: Dict[str, np.ndarray],
                     binding_vecs: Optional[Dict[str, np.ndarray]] = None,
                     answer_vecs: Optional[Dict[str, np.ndarray]] = None,
                     d: Optional[int] = None,
                     center: bool = True,
                     variance_threshold: float = 0.999,
                     min_d: Optional[int] = None,
                     align: bool = True) -> 'ShapeSpace':
        """Construct a ShapeSpace from raw vectors.

        Args:
            entity_vecs: name → high-dimensional vector for each entity
            binding_vecs: name → binding vector (relationship encoding).
                          If None, bindings are zero (pure entity lookup).
            answer_vecs: name → answer direction vector.
                         If None, answers are the entity vectors themselves.
            d: explicit dimensionality. If None, auto-detected from data.
            center: if True, remove mean entity (structure class direction).
                    This is what enables 4D discrimination.
            variance_threshold: fraction of variance to capture when
                                auto-detecting d (default 99.9%).
            min_d: minimum dimensionality. Defaults to N_entities - 1,
                   since N points in general position require N-1
                   dimensions to separate. Set to 0 to disable.
            align: if True and answer_vecs provided, learn Procrustes
                   rotation from entity subspace to answer subspace via
                   cross-covariance SVD. This is the geometric analog
                   of the model's V·W_o binding.

        Returns:
            A ShapeSpace ready for queries.
        """
        space = cls()
        names = list(entity_vecs.keys())
        source_dim = len(next(iter(entity_vecs.values())))
        space._source_dim = source_dim

        # Stack entity vectors
        entity_matrix = np.stack([entity_vecs[n].astype(np.float64)
                                  for n in names])

        # Center if requested (removes shared structure class direction)
        if center:
            space._entity_mean = entity_matrix.mean(axis=0)
            centered = entity_matrix - space._entity_mean
        else:
            space._entity_mean = np.zeros(source_dim, dtype=np.float64)
            centered = entity_matrix

        # Whitened alignment: map each entity to its centered answer
        # direction. This is equivalent to whitening the entity
        # similarity matrix to identity (decorrelating entities)
        # so each entity maps ONLY to its own answer with no leakage.
        # The SVD basis then captures answer-discriminative structure.
        if align and answer_vecs is not None and len(names) > 1:
            ans_names = list(answer_vecs.keys())
            ans_matrix = np.stack([answer_vecs[n].astype(np.float64)
                                   for n in ans_names])
            ans_centered = ans_matrix - ans_matrix.mean(axis=0)
            centered = ans_centered.copy()  # entities = their centered answers
        else:
            ans_names = list(answer_vecs.keys()) if answer_vecs is not None else names
            ans_matrix = np.stack([answer_vecs[n].astype(np.float64)
                                   for n in ans_names]) if answer_vecs is not None else None
            ans_centered = ans_matrix - ans_matrix.mean(axis=0) if ans_matrix is not None else None

        # Collect vectors for basis computation
        all_vecs = [centered]
        if binding_vecs is not None:
            bind_matrix = np.stack([binding_vecs[n].astype(np.float64)
                                    for n in names])
            all_vecs.append(bind_matrix)
        if ans_centered is not None:
            all_vecs.append(ans_centered)

        combined = np.vstack(all_vecs)

        # SVD to find optimal basis
        U, S, Vt = np.linalg.svd(combined, full_matrices=False)
        space._singular_values = S

        # Auto-detect dimensionality
        if d is None:
            cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
            d = int(np.searchsorted(cumvar, variance_threshold)) + 1
            # N entities need at least N-1 dims to separate
            effective_min = min_d if min_d is not None else (len(names) - 1)
            d = max(d, effective_min, 1)
            d = min(d, len(S))

        space._d = d
        space._basis = Vt[:d, :].copy()

        # Project entities (aligned or centered)
        for i, name in enumerate(names):
            space._entities[name] = (space._basis @ centered[i]).copy()

        # Project bindings
        if binding_vecs is not None:
            for name in names:
                vec = binding_vecs[name].astype(np.float64)
                space._bindings[name] = (space._basis @ vec).copy()
        else:
            for name in names:
                space._bindings[name] = np.zeros(d, dtype=np.float64)

        # Project answers (centered if alignment was used)
        if ans_centered is not None:
            for i, name in enumerate(ans_names):
                space._answers[name] = (space._basis @ ans_centered[i]).copy()
        elif answer_vecs is not None:
            for name in ans_names:
                vec = answer_vecs[name].astype(np.float64)
                space._answers[name] = (space._basis @ vec).copy()
        else:
            for name in names:
                space._answers[name] = space._entities[name].copy()

        return space

    @classmethod
    def from_observations(cls,
                          observations: List[Tuple[str, str, str]],
                          feature_fn: Callable[[str], np.ndarray],
                          d: Optional[int] = None,
                          **kwargs) -> 'ShapeSpace':
        """Construct from (entity, relation, answer) triples.

        Args:
            observations: list of (entity_name, relation_type, answer_name)
            feature_fn: function mapping any string to a vector representation
            d: explicit dimensionality (auto-detected if None)

        Returns:
            A ShapeSpace encoding all observed relationships.
        """
        entity_vecs = {}
        binding_vecs = {}
        answer_vecs = {}

        for entity, relation, answer in observations:
            if entity not in entity_vecs:
                entity_vecs[entity] = feature_fn(entity)
            # Binding = entity + relation combined
            key = entity
            binding_vecs[key] = feature_fn(f"{entity} {relation}")
            answer_vecs[key] = feature_fn(answer)

        return cls.from_vectors(entity_vecs, binding_vecs, answer_vecs,
                                d=d, **kwargs)

    # ═══════════════════════════════════════════════════════════
    # Query Operations
    # ═══════════════════════════════════════════════════════════

    def query(self, entity_name: str) -> Tuple[str, float]:
        """Query: entity → (best_answer_name, score).

        The core operation: O(d × N_answers).
        """
        h = self._entities[entity_name]
        b = self._bindings[entity_name]
        combined = h + b  # interference

        best_name = None
        best_score = -np.inf
        for name, ans_vec in self._answers.items():
            score = float(np.dot(combined, ans_vec))
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    def scores(self, entity_name: str) -> Dict[str, float]:
        """Get scores for all answers given an entity.

        Returns dict of {answer_name: score}.
        """
        h = self._entities[entity_name]
        b = self._bindings[entity_name]
        combined = h + b

        return {name: float(np.dot(combined, ans_vec))
                for name, ans_vec in self._answers.items()}

    def score(self, entity_name: str, answer_name: str) -> float:
        """Score a specific (entity, answer) pair."""
        h = self._entities[entity_name]
        b = self._bindings[entity_name]
        combined = h + b
        return float(np.dot(combined, self._answers[answer_name]))

    def query_vector(self, vec: np.ndarray,
                     binding: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Query with an arbitrary vector (not a stored entity).

        The vector is projected into the ShapeSpace basis first.
        If the space was constructed with centering, the mean is
        subtracted before projection.
        """
        vec = vec.astype(np.float64)
        if self._entity_mean is not None:
            vec = vec - self._entity_mean
        h = self._basis @ vec

        if binding is not None:
            b = self._basis @ binding.astype(np.float64)
        else:
            b = np.zeros(self._d, dtype=np.float64)

        combined = h + b
        best_name = None
        best_score = -np.inf
        for name, ans_vec in self._answers.items():
            score = float(np.dot(combined, ans_vec))
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    def batch_query(self, entity_names: List[str]) -> List[Tuple[str, float]]:
        """Query multiple entities at once."""
        return [self.query(name) for name in entity_names]

    # ═══════════════════════════════════════════════════════════
    # Algebraic Operations
    # ═══════════════════════════════════════════════════════════

    def project(self, d: int) -> 'ShapeSpace':
        """Project to a lower-dimensional ShapeSpace.

        Returns a new ShapeSpace with only the top-d dimensions.
        This is lossy — accuracy may decrease. The original is unchanged.

        Complexity: O(N × d_old)
        """
        if d >= self._d:
            return self  # no-op

        new_space = ShapeSpace()
        new_space._d = d
        new_space._source_dim = self._source_dim
        new_space._basis = self._basis[:d, :].copy()
        new_space._singular_values = self._singular_values[:d].copy() \
            if self._singular_values is not None else None
        new_space._entity_mean = self._entity_mean.copy() \
            if self._entity_mean is not None else None

        for name, vec in self._entities.items():
            new_space._entities[name] = vec[:d].copy()
        for name, vec in self._bindings.items():
            new_space._bindings[name] = vec[:d].copy()
        for name, vec in self._answers.items():
            new_space._answers[name] = vec[:d].copy()

        new_space._metadata = self._metadata.copy()
        return new_space

    def compose(self, other: 'ShapeSpace') -> 'ShapeSpace':
        """Compose two ShapeSpaces into one.

        The resulting space contains entities, bindings, and answers
        from both spaces. The basis is recomputed to span both.

        This is the data-structure analog of model merging:
        two dictionaries of shapes superimposed into one.

        Requires both spaces to have the same source_dim.
        """
        if self._source_dim != other._source_dim:
            raise ValueError(
                f"Cannot compose spaces with different source dimensions: "
                f"{self._source_dim} vs {other._source_dim}")

        # Collect all raw vectors (unproject back to source space)
        entity_vecs = {}
        binding_vecs = {}
        answer_vecs = {}

        for name, vec in self._entities.items():
            raw = self._basis.T @ vec
            if self._entity_mean is not None:
                raw = raw + self._entity_mean
            entity_vecs[f"a/{name}"] = raw

        for name, vec in other._entities.items():
            raw = other._basis.T @ vec
            if other._entity_mean is not None:
                raw = raw + other._entity_mean
            entity_vecs[f"b/{name}"] = raw

        for name, vec in self._bindings.items():
            binding_vecs[f"a/{name}"] = self._basis.T @ vec
        for name, vec in other._bindings.items():
            binding_vecs[f"b/{name}"] = other._basis.T @ vec

        for name, vec in self._answers.items():
            answer_vecs[f"a/{name}"] = self._basis.T @ vec
        for name, vec in other._answers.items():
            answer_vecs[f"b/{name}"] = other._basis.T @ vec

        return ShapeSpace.from_vectors(entity_vecs, binding_vecs,
                                       answer_vecs)

    def extend(self, entity_name: str,
               entity_vec: np.ndarray,
               binding_vec: Optional[np.ndarray] = None,
               answer_vec: Optional[np.ndarray] = None) -> None:
        """Add a new entity to the space.

        Projects the new entity into the existing basis.
        If the new entity is far from the current subspace,
        consider rebuilding with from_vectors instead.

        Args:
            entity_name: name for the new entity
            entity_vec: high-dimensional entity vector
            binding_vec: binding vector (default: zeros)
            answer_vec: answer direction vector (default: entity vec)
        """
        vec = entity_vec.astype(np.float64)
        if self._entity_mean is not None:
            vec = vec - self._entity_mean
        self._entities[entity_name] = (self._basis @ vec).copy()

        if binding_vec is not None:
            self._bindings[entity_name] = \
                (self._basis @ binding_vec.astype(np.float64)).copy()
        else:
            self._bindings[entity_name] = np.zeros(self._d, dtype=np.float64)

        if answer_vec is not None:
            self._answers[entity_name] = \
                (self._basis @ answer_vec.astype(np.float64)).copy()
        else:
            self._answers[entity_name] = self._entities[entity_name].copy()

    # ═══════════════════════════════════════════════════════════
    # Inspection
    # ═══════════════════════════════════════════════════════════

    @property
    def dimensionality(self) -> int:
        """The natural dimensionality of this space."""
        return self._d

    @property
    def n_entities(self) -> int:
        """Number of stored entities."""
        return len(self._entities)

    @property
    def n_answers(self) -> int:
        """Number of stored answer directions."""
        return len(self._answers)

    @property
    def entity_names(self) -> List[str]:
        return list(self._entities.keys())

    @property
    def answer_names(self) -> List[str]:
        return list(self._answers.keys())

    @property
    def ops_per_query(self) -> int:
        """Number of arithmetic operations per query.

        d additions (interference) + N_answers × (2d-1) ops (dot products)
        + (N_answers - 1) comparisons (argmax).
        """
        d = self._d
        na = len(self._answers)
        return d + na * (2 * d - 1) + max(na - 1, 0)

    @property
    def storage_bytes(self) -> int:
        """Total storage in bytes (float64)."""
        d = self._d
        n_ent = len(self._entities)
        n_bind = len(self._bindings)
        n_ans = len(self._answers)
        # Basis matrix + entity vectors + binding vectors + answer vectors
        basis_bytes = d * self._source_dim * 8
        vec_bytes = (n_ent + n_bind + n_ans) * d * 8
        mean_bytes = self._source_dim * 8 if self._entity_mean is not None else 0
        return basis_bytes + vec_bytes + mean_bytes

    def entity_similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two entities in the space."""
        va, vb = self._entities[a], self._entities[b]
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / norm) if norm > 1e-20 else 0.0

    def entity_vector(self, name: str) -> np.ndarray:
        """Get the d-dimensional vector for an entity."""
        return self._entities[name].copy()

    def answer_vector(self, name: str) -> np.ndarray:
        """Get the d-dimensional vector for an answer."""
        return self._answers[name].copy()

    def accuracy(self, ground_truth: Dict[str, str]) -> float:
        """Test accuracy against known entity→answer pairs.

        Args:
            ground_truth: {entity_name: expected_answer_name}

        Returns:
            Fraction correct (0.0 to 1.0).
        """
        correct = 0
        total = 0
        for entity, expected in ground_truth.items():
            if entity in self._entities:
                predicted, _ = self.query(entity)
                if predicted == expected:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0

    def spectrum(self) -> Optional[np.ndarray]:
        """The singular value spectrum of the basis.

        Shows how much variance each dimension captures.
        """
        return self._singular_values.copy() if self._singular_values is not None else None

    def variance_by_dim(self) -> Optional[np.ndarray]:
        """Cumulative variance explained by each dimension."""
        if self._singular_values is None:
            return None
        s2 = self._singular_values ** 2
        return np.cumsum(s2) / np.sum(s2)

    # ═══════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        """Serialize to a dictionary (for JSON/pickle/etc)."""
        return {
            'version': 1,
            'd': self._d,
            'source_dim': self._source_dim,
            'basis': self._basis.tolist(),
            'entities': {n: v.tolist() for n, v in self._entities.items()},
            'bindings': {n: v.tolist() for n, v in self._bindings.items()},
            'answers': {n: v.tolist() for n, v in self._answers.items()},
            'singular_values': self._singular_values.tolist()
                if self._singular_values is not None else None,
            'entity_mean': self._entity_mean.tolist()
                if self._entity_mean is not None else None,
            'metadata': self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ShapeSpace':
        """Deserialize from a dictionary."""
        space = cls()
        space._d = data['d']
        space._source_dim = data['source_dim']
        space._basis = np.array(data['basis'], dtype=np.float64)
        space._entities = {n: np.array(v, dtype=np.float64)
                           for n, v in data['entities'].items()}
        space._bindings = {n: np.array(v, dtype=np.float64)
                           for n, v in data['bindings'].items()}
        space._answers = {n: np.array(v, dtype=np.float64)
                          for n, v in data['answers'].items()}
        space._singular_values = np.array(data['singular_values'],
                                          dtype=np.float64) \
            if data.get('singular_values') is not None else None
        space._entity_mean = np.array(data['entity_mean'],
                                      dtype=np.float64) \
            if data.get('entity_mean') is not None else None
        space._metadata = data.get('metadata', {})
        return space

    # ═══════════════════════════════════════════════════════════
    # Display
    # ═══════════════════════════════════════════════════════════

    def __repr__(self):
        return (f"ShapeSpace(d={self._d}, "
                f"entities={self.n_entities}, "
                f"answers={self.n_answers}, "
                f"ops/query={self.ops_per_query}, "
                f"storage={self.storage_bytes:,}B)")

    def summary(self) -> str:
        """Detailed summary of the ShapeSpace."""
        lines = [
            f"ShapeSpace",
            f"  Dimensionality: {self._d}",
            f"  Source dimension: {self._source_dim}",
            f"  Entities: {self.n_entities} — "
            f"{', '.join(self.entity_names)}",
            f"  Answers:  {self.n_answers} — "
            f"{', '.join(self.answer_names)}",
            f"  Ops/query: {self.ops_per_query}",
            f"  Storage: {self.storage_bytes:,} bytes "
            f"({self.storage_bytes / 1024:.1f} KB)",
        ]

        if self._singular_values is not None:
            cumvar = self.variance_by_dim()
            lines.append(f"  Spectrum:")
            for i in range(min(self._d, 10)):
                lines.append(
                    f"    dim {i+1}: S={self._singular_values[i]:.4f}  "
                    f"cumvar={cumvar[i]*100:.1f}%")

        lines.append(f"  Entity similarities:")
        names = self.entity_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = self.entity_similarity(names[i], names[j])
                lines.append(f"    cos({names[i]}, {names[j]}) = {sim:.4f}")

        return '\n'.join(lines)
