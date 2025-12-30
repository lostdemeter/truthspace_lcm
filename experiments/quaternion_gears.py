#!/usr/bin/env python3
"""
Quaternion Gear-Meshing Projection

Instead of wave interference, we model truth and signal as GEARS:
- Each word/concept is a quaternion (4D rotation)
- Truth beam = driving gear
- Signal beam = driven gear
- Gear ratio = transformation between truth style and signal style

The gear metaphor:
- Teeth = discrete words/phrases
- Rotation = semantic flow through sentence
- Gear ratio = how many "truth rotations" produce one "signal rotation"
- Meshing = where truth and signal concepts align

Quaternion advantages:
- No gimbal lock (smooth interpolation)
- Composition is multiplication (chaining transformations)
- Conjugate gives inverse (reversible)
- 4D matches our concept space (x=gender, y=age, z=agency, w=animacy)

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class Quaternion:
    """A quaternion q = w + xi + yj + zk"""
    w: float  # scalar (real) part
    x: float  # i component
    y: float  # j component
    z: float  # k component
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (Hamilton product)."""
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
        )
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate (inverse rotation)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Quaternion magnitude."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Quaternion':
        """Return unit quaternion."""
        n = self.norm()
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between quaternions."""
        # Normalize both
        q1 = self.normalize()
        q2 = other.normalize()
        
        # Compute dot product
        dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        # If dot < 0, negate one to take shorter path
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        # If very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                q1.w + t*(q2.w - q1.w),
                q1.x + t*(q2.x - q1.x),
                q1.y + t*(q2.y - q1.y),
                q1.z + t*(q2.z - q1.z),
            )
            return result.normalize()
        
        # Spherical interpolation
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        return Quaternion(
            s1*q1.w + s2*q2.w,
            s1*q1.x + s2*q2.x,
            s1*q1.y + s2*q2.y,
            s1*q1.z + s2*q2.z,
        )
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)
    
    @staticmethod
    def from_axis_angle(axis: Tuple[float, float, float], angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        ax, ay, az = axis
        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if norm < 1e-10:
            return Quaternion(1, 0, 0, 0)
        
        ax, ay, az = ax/norm, ay/norm, az/norm
        half_angle = angle / 2
        s = math.sin(half_angle)
        
        return Quaternion(
            math.cos(half_angle),
            ax * s,
            ay * s,
            az * s,
        )
    
    @staticmethod
    def identity() -> 'Quaternion':
        return Quaternion(1, 0, 0, 0)


class QuaternionEncoder:
    """
    Encodes words and sentences as quaternions.
    
    Each word maps to a quaternion based on:
    - w: frequency (how common - structural vs content)
    - x: position in sentence (early vs late)
    - y: part-of-speech proxy (verb-like vs noun-like)
    - z: semantic role (agent vs patient vs modifier)
    """
    
    def __init__(self):
        self.word_quaternions: Dict[str, Quaternion] = {}
        self.word_freq: Dict[str, float] = {}
    
    def learn_from_corpus(self, texts: List[str]):
        """Learn word quaternions from corpus."""
        # Count frequencies
        freq = Counter()
        positions = defaultdict(list)  # word -> list of relative positions
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                continue
            
            freq.update(words)
            for i, w in enumerate(words):
                positions[w].append(i / len(words))  # Relative position [0, 1]
        
        # Normalize frequencies
        max_freq = max(freq.values()) if freq else 1
        self.word_freq = {w: c / max_freq for w, c in freq.items()}
        
        # Create quaternion for each word
        for word in freq:
            f = self.word_freq[word]
            
            # w component: frequency (high freq = structural)
            w_comp = f
            
            # x component: average position
            avg_pos = sum(positions[word]) / len(positions[word])
            x_comp = avg_pos * 2 - 1  # Map [0,1] to [-1,1]
            
            # y component: verb-like indicator (ends in -ing, -s, -ed)
            if word.endswith('ing') or word.endswith('ed'):
                y_comp = 0.8
            elif word.endswith('s') and len(word) > 3:
                y_comp = 0.5
            else:
                y_comp = -0.3
            
            # z component: role indicator (based on common patterns)
            if word in {'is', 'are', 'was', 'were', 'be', 'been'}:
                z_comp = 0.0  # Copula - neutral
            elif word in {'a', 'an', 'the', 'this', 'that'}:
                z_comp = -0.5  # Determiner
            elif word in {'who', 'which', 'that', 'what'}:
                z_comp = 0.3  # Relative
            elif word in {'and', 'or', 'but', 'so'}:
                z_comp = -0.8  # Conjunction
            else:
                z_comp = 0.1  # Default
            
            # Create and normalize quaternion
            q = Quaternion(w_comp, x_comp, y_comp, z_comp)
            self.word_quaternions[word] = q.normalize()
    
    def encode_word(self, word: str) -> Quaternion:
        """Get quaternion for a word."""
        word = word.lower()
        if word in self.word_quaternions:
            return self.word_quaternions[word]
        # Unknown word - return identity with slight perturbation
        return Quaternion(0.9, 0.1, 0.1, 0.1).normalize()
    
    def encode_sentence(self, text: str) -> Quaternion:
        """
        Encode sentence as a single quaternion.
        
        Compose word quaternions through multiplication.
        This captures the "rotation" through semantic space.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return Quaternion.identity()
        
        # Compose quaternions
        result = Quaternion.identity()
        for w in words:
            q = self.encode_word(w)
            result = result * q
        
        return result.normalize()


class GearTransformer:
    """
    Transforms truth to signal using gear-meshing metaphor.
    
    The gear ratio determines how truth "rotates" into signal:
    - Ratio > 1: Signal is more verbose/elaborate than truth
    - Ratio < 1: Signal is more concise than truth
    - Ratio = 1: Direct mapping
    
    The meshing quaternion captures the alignment between truth and signal styles.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
        # Load truth corpus
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.truth_qa.set_output_lens('natural')
        
        # Load signal corpus
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
        
        # Build encoder from signal corpus
        print("Learning quaternion encodings...")
        self.encoder = QuaternionEncoder()
        self.encoder.learn_from_corpus(signal_texts)
        
        # Learn gear transformation
        print("Learning gear transformation...")
        self.gear_ratio, self.mesh_quaternion = self._learn_gear_transform()
        
        print(f"Gear ratio: {self.gear_ratio:.3f}")
        print(f"Mesh quaternion: ({self.mesh_quaternion.w:.3f}, {self.mesh_quaternion.x:.3f}, {self.mesh_quaternion.y:.3f}, {self.mesh_quaternion.z:.3f})")
    
    def _learn_gear_transform(self) -> Tuple[float, Quaternion]:
        """
        Learn the gear transformation from truth-signal pairs.
        
        Gear ratio = average(|signal_q| / |truth_q|)
        Mesh quaternion = average(signal_q * truth_q.conjugate())
        """
        ratios = []
        mesh_sum = Quaternion(0, 0, 0, 0)
        count = 0
        
        for concept, signal_text in self.signal_frames.items():
            truth_text = self.truth_qa.ask(f"What is {concept}?")
            if "don't know" in truth_text.lower():
                continue
            
            # Encode both
            truth_q = self.encoder.encode_sentence(truth_text)
            signal_q = self.encoder.encode_sentence(signal_text)
            
            # Compute ratio (based on word count as proxy)
            truth_words = len(re.findall(r'\b\w+\b', truth_text))
            signal_words = len(re.findall(r'\b\w+\b', signal_text))
            if truth_words > 0:
                ratios.append(signal_words / truth_words)
            
            # Compute mesh: how to rotate from truth to signal
            # mesh = signal * truth^-1
            mesh = signal_q * truth_q.conjugate()
            mesh_sum = Quaternion(
                mesh_sum.w + mesh.w,
                mesh_sum.x + mesh.x,
                mesh_sum.y + mesh.y,
                mesh_sum.z + mesh.z,
            )
            count += 1
        
        if count == 0:
            return 1.0, Quaternion.identity()
        
        # Average ratio
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        
        # Average mesh quaternion
        avg_mesh = Quaternion(
            mesh_sum.w / count,
            mesh_sum.x / count,
            mesh_sum.y / count,
            mesh_sum.z / count,
        ).normalize()
        
        return avg_ratio, avg_mesh
    
    def transform(self, concept: str) -> str:
        """
        Transform truth to signal using gear meshing.
        
        1. Encode truth as quaternion
        2. Apply mesh transformation
        3. Decode back to text using signal vocabulary
        """
        concept_lower = concept.lower()
        
        # Get truth
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Direct match
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Encode truth
        truth_q = self.encoder.encode_sentence(truth)
        
        # Apply gear transformation: output = mesh * truth
        output_q = self.mesh_quaternion * truth_q
        
        # Decode: find signal sentence with closest quaternion
        return self._decode_quaternion(output_q, truth, concept)
    
    def _decode_quaternion(self, target_q: Quaternion, truth: str, concept: str) -> str:
        """
        Decode quaternion back to text.
        
        Strategy: Use truth structure but adjust based on quaternion distance
        to signal patterns.
        """
        # Parse truth into components
        entity, role, actions, targets = self._parse_truth(truth, concept)
        
        # Transform actions based on gear ratio
        # Higher ratio = more elaborate phrasing
        if self.gear_ratio > 1.2:
            # Signal is more verbose - use gerunds and add connectors
            transformed_actions = [self._to_gerund(a) for a in actions]
            connector = "that involves"
        elif self.gear_ratio < 0.8:
            # Signal is more concise - use base forms
            transformed_actions = actions
            connector = "who"
        else:
            # Similar verbosity - use gerunds (signal preference)
            transformed_actions = [self._to_gerund(a) for a in actions]
            connector = "that involves"
        
        # Build output
        # Use quaternion components to adjust style
        q = target_q.normalize()
        
        # w component affects formality (high w = more formal)
        if q.w > 0.5:
            prefix = f"{entity} is"
        else:
            prefix = f"{entity} seems to be"
        
        # Determine article
        article = "an" if role[0].lower() in 'aeiou' else "a"
        
        # Build action string
        if transformed_actions:
            if len(transformed_actions) == 1:
                action_str = transformed_actions[0]
            elif len(transformed_actions) == 2:
                action_str = f"{transformed_actions[0]} and {transformed_actions[1]}"
            else:
                action_str = f"{transformed_actions[0]}, {transformed_actions[1]}, and {transformed_actions[2]}"
        else:
            action_str = ""
        
        # Build target string
        if targets:
            target_str = ' and '.join(targets[:2])
        else:
            target_str = ""
        
        # Construct output
        if action_str and target_str:
            return f"{prefix} {article} {role} {connector} {action_str}, particularly {target_str}."
        elif action_str:
            return f"{prefix} {article} {role} {connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {role} related to {target_str}."
        else:
            return f"{prefix} {article} {role}."
    
    def _parse_truth(self, truth: str, concept: str) -> Tuple[str, str, List[str], List[str]]:
        """Parse truth into components."""
        truth_lower = truth.lower()
        
        entity = concept.title()
        role = "entity"
        actions = []
        targets = []
        
        # Role
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            role = match.group(1)
        
        # Actions
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                actions = [a for a in match.groups() if a]
        
        # Targets
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        return entity, role, actions, targets
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund."""
        verb = verb.lower()
        if verb.endswith('ing'):
            return verb
        elif verb.endswith('e') and not verb.endswith('ee'):
            return verb[:-1] + 'ing'
        elif verb.endswith('s') and not verb.endswith('ss'):
            base = verb[:-1]
            if base.endswith('e') and not base.endswith('ee'):
                return base[:-1] + 'ing'
            return base + 'ing'
        else:
            return verb + 'ing'


class AdaptiveGearProjector:
    """
    Adaptive gear projection that adjusts ratio per-concept.
    
    Instead of a single global gear ratio, we compute the optimal
    ratio for each concept based on its quaternion distance to
    known signal patterns.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str):
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
        
        # Build encoder
        self.encoder = QuaternionEncoder()
        self.encoder.learn_from_corpus(signal_texts)
        
        # Index signals by quaternion
        self.signal_index = []  # List of (quaternion, text, agent)
        for agent, text in self.signal_frames.items():
            q = self.encoder.encode_sentence(text)
            self.signal_index.append((q, text, agent))
        
        print(f"Indexed {len(self.signal_index)} signals as quaternions")
    
    def project(self, concept: str) -> str:
        """Project using adaptive gear ratio."""
        concept_lower = concept.lower()
        
        truth = self.truth_qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        if concept_lower in self.signal_frames:
            return self.signal_frames[concept_lower]
        
        # Encode truth
        truth_q = self.encoder.encode_sentence(truth)
        
        # Find nearest signals by quaternion distance
        nearest = self._find_nearest_signals(truth_q, k=3)
        
        # Compute adaptive gear ratio from nearest signals
        truth_words = len(re.findall(r'\b\w+\b', truth))
        ratios = []
        for _, sig_text, _ in nearest:
            sig_words = len(re.findall(r'\b\w+\b', sig_text))
            if truth_words > 0:
                ratios.append(sig_words / truth_words)
        
        adaptive_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        
        # Interpolate between nearest signal structures
        return self._interpolate_output(truth, concept, nearest, adaptive_ratio)
    
    def _find_nearest_signals(self, target_q: Quaternion, k: int = 3) -> List[Tuple[Quaternion, str, str]]:
        """Find k nearest signals by quaternion dot product."""
        distances = []
        
        for q, text, agent in self.signal_index:
            # Quaternion dot product (cosine of half-angle)
            dot = abs(target_q.w*q.w + target_q.x*q.x + target_q.y*q.y + target_q.z*q.z)
            distances.append((dot, q, text, agent))
        
        # Sort by dot product (higher = closer)
        distances.sort(key=lambda x: -x[0])
        
        return [(q, text, agent) for _, q, text, agent in distances[:k]]
    
    def _interpolate_output(self, truth: str, concept: str, 
                           nearest: List[Tuple[Quaternion, str, str]],
                           ratio: float) -> str:
        """
        Interpolate output from nearest signals.
        
        Use SLERP to blend quaternions, then decode.
        """
        # Parse truth
        entity, role, actions, targets = self._parse_truth(truth, concept)
        
        # Transform actions
        transformed_actions = [self._to_gerund(a) for a in actions]
        
        # Determine style from nearest signals
        # Check if nearest signals use "seems to be" or "is"
        formal_count = sum(1 for _, text, _ in nearest if 'is a' in text.lower() and 'seems' not in text.lower())
        informal_count = len(nearest) - formal_count
        
        if formal_count > informal_count:
            prefix = f"{entity} is"
        else:
            prefix = f"{entity} seems to be"
        
        # Determine connector from nearest
        involves_count = sum(1 for _, text, _ in nearest if 'involves' in text.lower() or 'that' in text.lower())
        who_count = sum(1 for _, text, _ in nearest if 'who' in text.lower())
        
        if involves_count > who_count:
            connector = "that involves"
        else:
            connector = "who"
        
        # Article
        article = "an" if role[0].lower() in 'aeiou' else "a"
        
        # Build action string
        if transformed_actions:
            if len(transformed_actions) == 1:
                action_str = transformed_actions[0]
            elif len(transformed_actions) == 2:
                action_str = f"{transformed_actions[0]} and {transformed_actions[1]}"
            else:
                action_str = f"{transformed_actions[0]}, {transformed_actions[1]}, and {transformed_actions[2]}"
        else:
            action_str = ""
        
        # Build target string
        target_str = ' and '.join(targets[:2]) if targets else ""
        
        # Construct
        if action_str and target_str:
            return f"{prefix} {article} {role} {connector} {action_str}, particularly {target_str}."
        elif action_str:
            return f"{prefix} {article} {role} {connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {role} related to {target_str}."
        else:
            return f"{prefix} {article} {role}."
    
    def _parse_truth(self, truth: str, concept: str) -> Tuple[str, str, List[str], List[str]]:
        """Parse truth into components."""
        truth_lower = truth.lower()
        
        entity = concept.title()
        role = "entity"
        actions = []
        targets = []
        
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            role = match.group(1)
        
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                actions = [a for a in match.groups() if a]
        
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        return entity, role, actions, targets
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund."""
        verb = verb.lower()
        if verb.endswith('ing'):
            return verb
        elif verb.endswith('e') and not verb.endswith('ee'):
            return verb[:-1] + 'ing'
        elif verb.endswith('s') and not verb.endswith('ss'):
            base = verb[:-1]
            if base.endswith('e') and not base.endswith('ee'):
                return base[:-1] + 'ing'
            return base + 'ing'
        else:
            return verb + 'ing'


def demo():
    """Demo the quaternion gear projectors."""
    print("=" * 70)
    print("QUATERNION GEAR-MESHING PROJECTION")
    print("Truth and signal as interlocking gears")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    print("\n--- Fixed Gear Transformer ---")
    fixed = GearTransformer(truth_path, signal_path)
    
    print("\n--- Adaptive Gear Projector ---")
    adaptive = AdaptiveGearProjector(truth_path, signal_path)
    
    # Find test concepts
    test_concepts = []
    for concept in fixed.truth_qa.knowledge.concepts:
        if concept not in fixed.signal_frames:
            c = fixed.truth_qa.knowledge.concepts[concept]
            if c.is_content_word and c.actions and len(c.actions) >= 2:
                test_concepts.append(concept)
        if len(test_concepts) >= 8:
            break
    
    print("\n" + "=" * 70)
    print("Testing projection:")
    print("=" * 70)
    
    for concept in test_concepts:
        truth = fixed.truth_qa.ask(f"What is {concept}?")
        fixed_result = fixed.transform(concept)
        adaptive_result = adaptive.project(concept)
        
        print(f"\n{concept.upper()}")
        print(f"  TRUTH:    {truth}")
        print(f"  FIXED:    {fixed_result}")
        print(f"  ADAPTIVE: {adaptive_result}")


if __name__ == "__main__":
    demo()
