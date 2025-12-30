#!/usr/bin/env python3
"""
Continuous Geometric Projection

The fundamental insight: We need to work in CONTINUOUS space, not discrete.

Approach:
1. Build a simple word embedding from co-occurrence in signal corpus
2. Represent sentences as trajectories in this space
3. Learn the transformation that maps truth trajectories to signal trajectories
4. Apply transformation and decode

The embedding captures:
- Words that appear together are close in space
- The "signal style" defines a REGION in this space
- Projection = moving toward that region while preserving content

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


class WordEmbedding:
    """
    Simple word embedding from co-occurrence.
    
    Words that appear together in signal corpus are close in embedding space.
    """
    
    def __init__(self, dim: int = 50):
        self.dim = dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
    
    def fit(self, texts: List[str], window: int = 3):
        """Build embeddings from texts using co-occurrence."""
        # Build vocabulary
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq.update(words)
        
        # Keep top words
        vocab = [w for w, c in word_freq.most_common(5000) if c >= 2]
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        
        n_vocab = len(vocab)
        
        # Build co-occurrence matrix
        cooccur = np.zeros((n_vocab, n_vocab))
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            indices = [self.word_to_idx[w] for w in words if w in self.word_to_idx]
            
            for i, idx in enumerate(indices):
                for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                    if i != j:
                        cooccur[idx, indices[j]] += 1
        
        # SVD to get embeddings
        # Add small value for numerical stability
        cooccur = np.log1p(cooccur)
        
        U, S, Vt = np.linalg.svd(cooccur, full_matrices=False)
        
        # Take top dimensions
        self.embeddings = U[:, :self.dim] * np.sqrt(S[:self.dim])
    
    def embed_word(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        word = word.lower()
        if word in self.word_to_idx:
            return self.embeddings[self.word_to_idx[word]]
        return None
    
    def embed_sentence(self, text: str) -> np.ndarray:
        """
        Embed sentence as sequence of word vectors.
        
        Returns array of shape (n_words, dim)
        """
        words = re.findall(r'\b\w+\b', text.lower())
        vectors = []
        
        for w in words:
            vec = self.embed_word(w)
            if vec is not None:
                vectors.append(vec)
            else:
                # Unknown word - use zero vector
                vectors.append(np.zeros(self.dim))
        
        if not vectors:
            return np.zeros((1, self.dim))
        
        return np.array(vectors)
    
    def nearest_word(self, vec: np.ndarray, exclude: set = None) -> str:
        """Find nearest word to a vector."""
        if exclude is None:
            exclude = set()
        
        best_word = None
        best_sim = -np.inf
        
        for word, idx in self.word_to_idx.items():
            if word in exclude:
                continue
            
            emb = self.embeddings[idx]
            sim = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb) + 1e-8)
            
            if sim > best_sim:
                best_sim = sim
                best_word = word
        
        return best_word


class ContinuousProjector:
    """
    Projects truth to signal in continuous embedding space.
    
    The transformation is learned as:
    1. Average "direction" from truth embeddings to signal embeddings
    2. This direction captures the "style shift"
    3. Apply this shift to new truth sentences
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
        self.signal_frames = {}
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
        
        # Build embedding from signal corpus
        print("Building word embeddings...")
        self.embedding = WordEmbedding(dim=50)
        self.embedding.fit(list(self.signal_frames.values()))
        print(f"Vocabulary size: {len(self.embedding.word_to_idx)}")
        
        # Learn transformation
        print("Learning transformation...")
        self.transform_direction = self._learn_transform()
        print(f"Transform magnitude: {np.linalg.norm(self.transform_direction):.4f}")
    
    def _learn_transform(self) -> np.ndarray:
        """
        Learn the transformation direction from truth to signal.
        
        For each truth-signal pair:
        - Compute centroid of truth embedding
        - Compute centroid of signal embedding
        - Direction = signal_centroid - truth_centroid
        
        Average across all pairs.
        """
        directions = []
        
        for concept, signal_text in self.signal_frames.items():
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            # Embed both
            truth_emb = self.embedding.embed_sentence(truth_text)
            signal_emb = self.embedding.embed_sentence(signal_text)
            
            # Centroids
            truth_centroid = truth_emb.mean(axis=0)
            signal_centroid = signal_emb.mean(axis=0)
            
            # Direction
            direction = signal_centroid - truth_centroid
            directions.append(direction)
        
        if not directions:
            return np.zeros(self.embedding.dim)
        
        # Average direction
        return np.mean(directions, axis=0)
    
    def project(self, concept: str) -> str:
        """
        Project truth to signal using learned transformation.
        """
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # If we have direct signal, return it
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Embed truth
        truth_emb = self.embedding.embed_sentence(truth)
        
        # Apply transformation to each word vector
        projected_emb = truth_emb + self.transform_direction
        
        # Decode back to words
        return self._decode(projected_emb, truth)
    
    def _decode(self, projected_emb: np.ndarray, reference: str) -> str:
        """
        Decode projected embeddings back to text.
        
        For each position, find the nearest word in vocabulary.
        Use reference structure to guide word order.
        """
        ref_words = re.findall(r'\b\w+\b', reference)
        
        result = []
        used_words = set()
        
        for i, vec in enumerate(projected_emb):
            # Find nearest word
            nearest = self.embedding.nearest_word(vec, exclude=used_words)
            
            if nearest:
                # Check if we should use reference word or nearest
                if i < len(ref_words):
                    ref_word = ref_words[i].lower()
                    ref_vec = self.embedding.embed_word(ref_word)
                    
                    if ref_vec is not None:
                        # Compare distances
                        ref_dist = np.linalg.norm(vec - ref_vec)
                        nearest_vec = self.embedding.embed_word(nearest)
                        nearest_dist = np.linalg.norm(vec - nearest_vec) if nearest_vec is not None else np.inf
                        
                        # Use reference if it's close enough
                        if ref_dist < nearest_dist * 1.5:
                            result.append(ref_words[i])
                            used_words.add(ref_word)
                            continue
                
                result.append(nearest)
                used_words.add(nearest)
        
        return ' '.join(result)


class HybridProjector:
    """
    Hybrid approach: Use continuous projection for style, discrete for structure.
    
    1. Parse truth into semantic components (entity, role, actions, targets)
    2. Use continuous projection to find best signal-style words for each component
    3. Reconstruct using signal-learned structure patterns
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load corpora
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        self.signal_frames = {}
        signal_texts = []
        if os.path.exists(signal_corpus_path):
            with open(signal_corpus_path, 'r') as f:
                data = json.load(f)
                for frame in data.get('frames', []):
                    agent = frame.get('agent', '').lower()
                    text = frame.get('text', '')
                    if agent and text:
                        self.signal_frames[agent] = text
                        signal_texts.append(text)
        
        # Build embedding
        print("Building embeddings...")
        self.embedding = WordEmbedding(dim=30)
        self.embedding.fit(signal_texts)
        
        # Learn component transformations
        self.role_centroids = self._learn_role_centroids()
        self.action_centroids = self._learn_action_centroids()
        self.structure_patterns = self._learn_structure_patterns()
    
    def _learn_role_centroids(self) -> Dict[str, np.ndarray]:
        """Learn centroid embedding for each role type."""
        role_words = defaultdict(list)
        
        for text in self.signal_frames.values():
            # Extract role word
            match = re.search(r'is a[n]? (\w+)', text.lower())
            if match:
                role = match.group(1)
                vec = self.embedding.embed_word(role)
                if vec is not None:
                    role_words[role].append(vec)
        
        centroids = {}
        for role, vecs in role_words.items():
            if vecs:
                centroids[role] = np.mean(vecs, axis=0)
        
        return centroids
    
    def _learn_action_centroids(self) -> Dict[str, np.ndarray]:
        """Learn centroid embedding for action verbs."""
        action_words = defaultdict(list)
        
        for text in self.signal_frames.values():
            # Extract action words (gerunds)
            gerunds = re.findall(r'\b(\w+ing)\b', text.lower())
            for g in gerunds:
                vec = self.embedding.embed_word(g)
                if vec is not None:
                    action_words[g].append(vec)
        
        centroids = {}
        for action, vecs in action_words.items():
            if vecs:
                centroids[action] = np.mean(vecs, axis=0)
        
        return centroids
    
    def _learn_structure_patterns(self) -> List[str]:
        """Learn common structure patterns from signal."""
        patterns = Counter()
        
        for text in self.signal_frames.values():
            # Classify structure
            text_lower = text.lower()
            
            if 'is a' in text_lower and 'that involves' in text_lower:
                patterns['is_a_that_involves'] += 1
            elif 'is a' in text_lower and 'known for' in text_lower:
                patterns['is_a_known_for'] += 1
            elif 'seems to be' in text_lower:
                patterns['seems_to_be'] += 1
            elif 'is a' in text_lower:
                patterns['is_a_simple'] += 1
            else:
                patterns['other'] += 1
        
        return [p for p, _ in patterns.most_common(5)]
    
    def project(self, concept: str) -> str:
        """Project using hybrid approach."""
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # If we have direct signal, return it
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Parse truth
        components = self._parse_truth(truth, concept)
        
        # Transform components using embeddings
        transformed = self._transform_components(components)
        
        # Reconstruct using learned structure
        return self._reconstruct(transformed)
    
    def _parse_truth(self, truth: str, concept: str) -> Dict:
        """Parse truth into components."""
        components = {
            'entity': concept.title(),
            'role': 'entity',
            'actions': [],
            'targets': [],
        }
        
        truth_lower = truth.lower()
        
        # Role
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            components['role'] = match.group(1)
        
        # Actions
        match = re.search(r'who (\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            components['actions'] = [a for a in match.groups() if a]
        
        # Targets
        match = re.search(r'(?:involving|relates to)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            components['targets'] = [t for t in match.groups() if t]
        
        return components
    
    def _transform_components(self, components: Dict) -> Dict:
        """Transform components using embedding space."""
        transformed = components.copy()
        
        # Transform role - find nearest role in signal space
        role = components['role']
        role_vec = self.embedding.embed_word(role)
        
        if role_vec is not None and self.role_centroids:
            # Find nearest role centroid
            best_role = role
            best_dist = np.inf
            
            for signal_role, centroid in self.role_centroids.items():
                dist = np.linalg.norm(role_vec - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_role = signal_role
            
            transformed['role'] = best_role
        
        # Transform actions to gerunds
        new_actions = []
        for action in components['actions']:
            # Convert to gerund
            if action.endswith('s'):
                base = action[:-1]
                if base.endswith('e'):
                    gerund = base[:-1] + 'ing'
                else:
                    gerund = base + 'ing'
            elif action.endswith('e'):
                gerund = action[:-1] + 'ing'
            else:
                gerund = action + 'ing'
            
            new_actions.append(gerund)
        
        transformed['actions'] = new_actions
        
        return transformed
    
    def _reconstruct(self, components: Dict) -> str:
        """Reconstruct sentence using signal-like structure."""
        entity = components['entity']
        role = components['role']
        actions = components['actions']
        targets = components['targets']
        
        # Build action string
        if actions:
            if len(actions) == 1:
                action_str = actions[0]
            elif len(actions) == 2:
                action_str = f"{actions[0]} and {actions[1]}"
            else:
                action_str = f"{actions[0]}, {actions[1]}, and {actions[2]}"
        else:
            action_str = ""
        
        # Build target string
        target_str = ' and '.join(targets[:2]) if targets else ""
        
        # Use most common structure pattern
        if action_str and target_str:
            return f"{entity} is a {role} that involves {action_str}, particularly {target_str}."
        elif action_str:
            return f"{entity} is a {role} that involves {action_str}."
        elif target_str:
            return f"{entity} is a {role} related to {target_str}."
        else:
            return f"{entity} is a {role}."


def demo():
    """Demo the continuous projectors."""
    print("=" * 70)
    print("CONTINUOUS GEOMETRIC PROJECTION")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    print("\n--- Hybrid Projector ---")
    hybrid = HybridProjector(truth_path, signal_path)
    
    # Find test concepts
    test_concepts = []
    for concept in hybrid.truth_qa.knowledge.concepts:
        if concept not in hybrid.signal_frames:
            c = hybrid.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 8:
            break
    
    print(f"\nTesting {len(test_concepts)} concepts NOT in signal corpus:\n")
    
    for concept in test_concepts:
        truth = hybrid.truth_qa.ask(f"What is {concept}?")
        result = hybrid.project(concept)
        
        print(f"{concept.upper()}")
        print(f"  TRUTH:     {truth}")
        print(f"  PROJECTED: {result}")
        print()


if __name__ == "__main__":
    demo()
