#!/usr/bin/env python3
"""
φ Morphology: Emergent Morphological Patterns

Hypothesis: Morphological variants share semantic position but differ in frequency.
The base form is the highest-frequency word in a semantic cluster.

Uses φ-Zipf patterns to discover morphological relationships without hard-coded rules.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

PHI = 1.618034


@dataclass
class MorphoWord:
    """A word with morphological properties."""
    word: str
    frequency: int = 0
    positions: List[float] = field(default_factory=list)
    
    # Morphological cluster
    cluster_id: Optional[int] = None
    is_base_form: bool = False
    base_form: Optional[str] = None
    
    @property
    def mean_position(self) -> float:
        if not self.positions:
            return 0.5
        return sum(self.positions) / len(self.positions)
    
    @property
    def position_variance(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        mean = self.mean_position
        return sum((p - mean) ** 2 for p in self.positions) / len(self.positions)


class PhiMorphology:
    """
    Emergent morphology using φ-Zipf patterns.
    
    Key insight: Words that share semantic position but differ in frequency
    are likely morphological variants. The highest-frequency word is the base.
    """
    
    def __init__(self, position_threshold: float = 0.15):
        self.words: Dict[str, MorphoWord] = {}
        self.clusters: Dict[int, List[str]] = {}
        self.position_threshold = position_threshold
        self.next_cluster_id = 0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def learn(self, text: str):
        """Learn word positions and frequencies from text."""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if len(tokens) < 2:
                continue
            
            for i, word in enumerate(tokens):
                if len(word) < 2:
                    continue
                
                if word not in self.words:
                    self.words[word] = MorphoWord(word=word)
                
                w = self.words[word]
                w.frequency += 1
                pos = i / max(len(tokens) - 1, 1)
                w.positions.append(pos)
        
        # After learning, discover morphological clusters
        self._discover_clusters()
    
    def _discover_clusters(self):
        """
        Discover morphological clusters using semantic position.
        
        Words at similar positions are candidates for the same cluster.
        Within a cluster, the highest-frequency word is the base form.
        """
        # Sort words by mean position
        sorted_words = sorted(self.words.values(), key=lambda w: w.mean_position)
        
        # Cluster words by position proximity
        for word in sorted_words:
            if word.cluster_id is not None:
                continue
            
            # Find or create cluster
            cluster_found = False
            for cluster_id, members in self.clusters.items():
                # Check if this word is close to any cluster member
                for member_name in members:
                    member = self.words[member_name]
                    if abs(word.mean_position - member.mean_position) < self.position_threshold:
                        # Additional check: similar suffix patterns suggest morphological relation
                        if self._might_be_related(word.word, member.word):
                            word.cluster_id = cluster_id
                            self.clusters[cluster_id].append(word.word)
                            cluster_found = True
                            break
                if cluster_found:
                    break
            
            if not cluster_found:
                # Create new cluster
                word.cluster_id = self.next_cluster_id
                self.clusters[self.next_cluster_id] = [word.word]
                self.next_cluster_id += 1
        
        # Within each cluster, identify base form
        # Geometric heuristic: base form is typically
        # 1. Highest frequency (Zipf)
        # 2. Shortest length (simpler form)
        # 3. No common variant suffixes (ed, ing, s, es)
        # 
        # We use a composite score: frequency / (length * suffix_penalty)
        
        variant_suffixes = {'ed', 'ing', 's', 'es', 'er', 'est', 'ly', 'ers'}
        
        for cluster_id, members in self.clusters.items():
            if len(members) < 2:
                continue
            
            # Score each word
            def base_score(name: str) -> float:
                w = self.words[name]
                freq = w.frequency
                length = len(name)
                
                # Suffix penalty: words ending in variant suffixes are less likely base
                suffix_penalty = 1.0
                for suffix in variant_suffixes:
                    if name.endswith(suffix) and len(name) > len(suffix) + 2:
                        suffix_penalty = 2.0  # Penalize variant-like endings
                        break
                
                # Score: higher frequency, shorter length, no variant suffix
                # This is geometric: frequency/length is a ratio, like φ
                return freq / (length * suffix_penalty)
            
            member_words = [(name, base_score(name)) for name in members]
            member_words.sort(key=lambda x: x[1], reverse=True)
            
            # Highest score is base form
            base_name = member_words[0][0]
            self.words[base_name].is_base_form = True
            self.words[base_name].base_form = base_name
            
            # Others are variants
            for name, _ in member_words[1:]:
                self.words[name].is_base_form = False
                self.words[name].base_form = base_name
    
    def _might_be_related(self, word1: str, word2: str) -> bool:
        """
        Check if two words might be morphologically related.
        
        Uses geometric heuristics:
        1. Share a common prefix (at least 3 chars)
        2. One is a suffix of the other
        3. Differ only by common morphological endings
        """
        # Must share significant prefix
        min_len = min(len(word1), len(word2))
        if min_len < 3:
            return False
        
        # Find common prefix length
        prefix_len = 0
        for i in range(min_len):
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break
        
        # Need at least 3 chars in common, or 60% of shorter word
        threshold = max(3, int(min_len * 0.6))
        if prefix_len < threshold:
            return False
        
        # Check for common morphological patterns
        # These are geometric in the sense that they're suffix positions
        suffix1 = word1[prefix_len:]
        suffix2 = word2[prefix_len:]
        
        # Common English morphological suffixes
        # (We could make this emergent too, but for now it's a thin layer)
        morpho_suffixes = {'', 's', 'es', 'ed', 'ing', 'd', 'er', 'est', 'ly'}
        
        return suffix1 in morpho_suffixes or suffix2 in morpho_suffixes
    
    def get_base(self, word: str) -> str:
        """Get the base form of a word."""
        if word not in self.words:
            return word
        
        w = self.words[word]
        if w.base_form:
            return w.base_form
        return word
    
    def get_variants(self, word: str) -> List[str]:
        """Get all variants of a word (including itself)."""
        if word not in self.words:
            return [word]
        
        w = self.words[word]
        if w.cluster_id is None:
            return [word]
        
        return self.clusters.get(w.cluster_id, [word])
    
    def show_clusters(self):
        """Show discovered morphological clusters."""
        print("\nMORPHOLOGICAL CLUSTERS (φ-Zipf Emergent)")
        print("=" * 60)
        
        # Only show clusters with multiple members
        multi_clusters = [(cid, members) for cid, members in self.clusters.items() 
                         if len(members) > 1]
        
        if not multi_clusters:
            print("No multi-word clusters found.")
            return
        
        for cluster_id, members in sorted(multi_clusters, key=lambda x: -len(x[1])):
            # Sort by frequency
            member_info = [(name, self.words[name]) for name in members]
            member_info.sort(key=lambda x: x[1].frequency, reverse=True)
            
            base = member_info[0][0]
            variants = [name for name, _ in member_info[1:]]
            
            print(f"\nCluster {cluster_id}: BASE = '{base}'")
            print(f"  Position: {self.words[base].mean_position:.2f}")
            print(f"  Frequency: {self.words[base].frequency}")
            if variants:
                print(f"  Variants: {', '.join(variants)}")
                for v in variants:
                    w = self.words[v]
                    ratio = self.words[base].frequency / max(w.frequency, 1)
                    print(f"    {v}: freq={w.frequency}, ratio={ratio:.2f} (φ={PHI:.2f})")


# Test corpus
CORPUS = """
Holmes examined the evidence carefully and methodically.
Watson watched from the doorway with great interest.
The detective studied the footprints on the floor.
Holmes deduced the identity of the criminal brilliantly.
Moriarty plotted against Holmes in secret.
Watson assisted Holmes faithfully throughout the case.
Lestrade questioned the witnesses at the scene.
Holmes observed every detail of the crime scene.
The professor schemed against the detective relentlessly.
Watson wrote about their adventures in his journal.
Holmes solved the mystery with remarkable insight.
The criminal fled from the scene in panic.

Alice fell down the rabbit hole unexpectedly.
The Queen shouted at everyone in the garden.
Cheshire smiled mysteriously and vanished slowly.
Alice grew very tall after drinking the potion.
The Hatter laughed wildly at the tea party.
Alice explored Wonderland with great curiosity.
The Caterpillar asked Alice many strange questions.
Alice shrank to a tiny size suddenly.
The Queen ordered everyone to play croquet.
Alice woke from her dream at last.

Hamlet pondered the meaning of existence deeply.
Claudius poisoned the King in the garden.
Ophelia loved Hamlet with all her heart.
Hamlet confronted his mother about her betrayal.
The ghost revealed the truth about the murder.
Hamlet killed Claudius in the final act.
Ophelia drowned in the river tragically.
Horatio witnessed the tragic events unfold.
Laertes sought revenge for his father.
The play ended with many deaths.

Love conquers all obstacles in life.
She loves him deeply and truly.
They loved each other from the start.
Loving someone requires patience and understanding.
The lovers met in secret every night.

Watch the sunset from the hilltop.
He watches the stars every evening.
She watched the children play in the park.
Watching birds is a peaceful hobby.
The watchers waited patiently for dawn.
"""


def demo():
    """Demonstrate emergent morphology."""
    print("φ MORPHOLOGY: Emergent Patterns")
    print("=" * 60)
    print()
    
    morph = PhiMorphology()
    morph.learn(CORPUS)
    
    print(f"Total words: {len(morph.words)}")
    print(f"Total clusters: {len(morph.clusters)}")
    
    morph.show_clusters()
    
    # Test base form lookup
    print("\n" + "=" * 60)
    print("BASE FORM LOOKUP")
    print("=" * 60)
    
    test_words = ['loves', 'loved', 'loving', 'watches', 'watched', 'watching']
    for word in test_words:
        base = morph.get_base(word)
        variants = morph.get_variants(word)
        print(f"  {word} -> base: {base}, variants: {variants}")
    
    return morph


if __name__ == "__main__":
    demo()
