#!/usr/bin/env python3
"""
Stacked Geometric LCM - Primary Core Module

A hierarchical geometric embedding system that generates discriminative
embeddings without training. Uses 7 stacked layers to encode text at
multiple scales:

Architecture (128 dimensions total):
- Layer 0: Morphological (16D) - Word structure (prefixes, suffixes, n-grams)
- Layer 1: Lexical (32D) - Primitive activation (φ-MAX encoding)
- Layer 2: Syntactic (16D) - Word order patterns (bigrams)
- Layer 3: Compositional (24D) - Domain signatures from primitive combinations
- Layer 4: Disambiguation (16D) - Context-dependent meaning resolution
- Layer 5: Contextual (16D) - Co-occurrence statistics (learned from data)
- Layer 6: Global (8D) - Distance to emergent prototypes

Key features:
- No training required - structure emerges from geometric operations
- Interpretable - each dimension has semantic meaning
- Fast - CPU-only, instant encoding
- Extensible - add primitives and patterns as needed

Usage:
    from truthspace_lcm.core.stacked_lcm import StackedLCM
    
    lcm = StackedLCM()
    lcm.ingest("chop onions finely", "cooking food preparation")
    lcm.ingest("ls -la", "list files directory terminal")
    
    content, similarity, cluster = lcm.resolve("how do I cut vegetables")
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import re

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# LAYER 0: MORPHOLOGICAL (16D) - Unchanged but weighted less
# =============================================================================

class MorphologicalLayer:
    """Word structure encoding."""
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.prefixes = {
            'pre': 0, 'un': 1, 're': 2, 'dis': 3, 'over': 4,
            'under': 5, 'out': 6, 'up': 7, 'down': 8, 'sub': 9
        }
        self.suffixes = {
            'ing': 0, 'ed': 1, 'er': 2, 'est': 3, 'ly': 4,
            'tion': 5, 'ness': 6, 'ment': 7, 'able': 8, 'ful': 9
        }
    
    def encode_word(self, word: str) -> np.ndarray:
        pos = np.zeros(self.dimensions)
        word = word.lower()
        
        for prefix, idx in self.prefixes.items():
            if word.startswith(prefix) and idx < 5:
                pos[idx] = PHI
        
        for suffix, idx in self.suffixes.items():
            if word.endswith(suffix) and idx < 5:
                pos[5 + idx] = PHI
        
        for i in range(len(word) - 2):
            trigram = word[i:i+3]
            dim = 10 + (hash(trigram) % 6)
            pos[dim] = max(pos[dim], PHI ** (1 - i/max(len(word), 1)))
        
        pos[15] = min(len(word) / 10, 1.0)
        return pos
    
    def encode(self, text: str) -> np.ndarray:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return np.zeros(self.dimensions)
        return np.mean([self.encode_word(w) for w in words], axis=0)


# =============================================================================
# LAYER 1: LEXICAL (32D) - EXPANDED PRIMITIVES
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
    EXPANDED primitive vocabulary.
    
    Key fix: Added many more keywords including command names,
    abbreviations, and domain-specific terms.
    """
    
    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions
        self.primitives = self._build_primitives()
        self._build_index()
    
    def _build_primitives(self) -> List[Primitive]:
        return [
            # ===== ACTIONS (dims 0-7) =====
            Primitive("CREATE", 0, 0, {
                "create", "make", "new", "build", "generate", "mkdir", "touch",
                "add", "insert", "construct", "produce", "write", "compose"
            }),
            Primitive("DESTROY", 0, 1, {
                "destroy", "delete", "remove", "kill", "end", "rm", "rmdir",
                "erase", "clear", "wipe", "terminate", "stop", "unlink"
            }),
            Primitive("TRANSFORM", 1, 0, {
                "change", "transform", "convert", "modify", "edit", "update",
                "alter", "adjust", "rename", "mv", "sed", "replace"
            }),
            Primitive("MOVE", 1, 1, {
                "move", "transfer", "shift", "relocate", "copy", "cp", "mv",
                "migrate", "send", "put", "place"
            }),
            Primitive("READ", 2, 0, {
                "read", "get", "view", "show", "see", "list", "display", "cat",
                "less", "more", "head", "tail", "ls", "dir", "print", "echo",
                "output", "look", "examine", "inspect", "cut"  # Unix cut command
            }),
            Primitive("WRITE", 2, 1, {
                "write", "set", "save", "store", "record", "log", "append",
                "output", "dump", "export"
            }),
            Primitive("SEARCH", 3, 0, {
                "search", "find", "locate", "seek", "look", "grep", "awk",
                "filter", "query", "match", "scan", "hunt"
            }),
            Primitive("COMBINE", 3, 1, {
                "combine", "merge", "mix", "blend", "join", "concat", "cat",
                "unite", "fuse", "integrate", "aggregate"
            }),
            
            # ===== TECH OBJECTS (dims 8-11) =====
            Primitive("FILE", 8, 0, {
                "file", "files", "document", "data", "txt", "log", "config",
                "script", "code", "source", "binary", "executable"
            }),
            Primitive("DIRECTORY", 8, 1, {
                "directory", "folder", "path", "dir", "directories", "folders",
                "location", "pwd", "cd", "tree"
            }),
            Primitive("PROCESS", 9, 0, {
                "process", "program", "running", "task", "job", "pid", "ps",
                "top", "htop", "daemon", "service", "thread", "execute", "run"
            }),
            Primitive("SYSTEM", 9, 1, {
                "system", "computer", "machine", "server", "host", "os",
                "kernel", "hardware", "cpu", "memory", "disk"
            }),
            
            # ===== COOKING OBJECTS (dims 12-15) =====
            Primitive("FOOD", 12, 0, {
                "food", "meal", "dish", "recipe", "recipes", "ingredient", "ingredients",
                "vegetable", "vegetables", "meat", "fruit", "onion", "onions",
                "carrot", "carrots", "potato", "tomato", "garlic", "chicken",
                "beef", "pork", "fish", "egg", "eggs", "bread", "rice", "pasta",
                "soup", "salad", "sauce", "butter", "oil", "flour", "sugar",
                "kitchen", "cook", "chef", "dinner", "lunch", "breakfast"
            }),
            Primitive("HEAT", 12, 1, {
                "heat", "hot", "warm", "temperature", "boil", "bake", "roast",
                "fry", "grill", "simmer", "sauté", "steam", "oven", "stove",
                "flame", "fire", "cook", "cooking"
            }),
            Primitive("CUT_FOOD", 13, 0, {
                "chop", "slice", "dice", "mince", "julienne", "knife",
                "cutting", "chopping", "slicing", "peel", "grate", "shred"
            }),
            # Note: "cut" alone is ambiguous - handled by context
            Primitive("TASTE", 13, 1, {
                "taste", "flavor", "season", "salt", "pepper", "spice", "spices",
                "sweet", "sour", "bitter", "savory", "umami", "herb", "herbs"
            }),
            
            # ===== SOCIAL (dims 16-19) =====
            Primitive("GREETING", 16, 0, {
                "hello", "hi", "hey", "greetings", "welcome", "howdy",
                "good morning", "good afternoon", "good evening"
            }),
            Primitive("GRATITUDE", 16, 1, {
                "thanks", "thank", "grateful", "appreciate", "appreciation",
                "thankful", "gratitude"
            }),
            Primitive("FEELING", 17, 0, {
                "feel", "feeling", "emotion", "mood", "happy", "sad", "angry",
                "excited", "worried", "anxious", "calm", "understand"
            }),
            Primitive("HELP", 17, 1, {
                "help", "assist", "support", "aid", "care", "helping",
                "assistance", "guidance", "advice"
            }),
            
            # ===== RELATIONS (dims 20-23) =====
            Primitive("INTO", 20, 0, {"into", "to", "toward", "inside", "in"}),
            Primitive("FROM", 20, 1, {"from", "out", "away", "source", "of"}),
            Primitive("WITH", 21, 0, {"with", "using", "by", "through", "via"}),
            Primitive("ABOUT", 21, 1, {"about", "regarding", "concerning", "on"}),
            
            # ===== STRUCTURE (dims 24-27) =====
            Primitive("SEQUENCE", 24, 0, {
                "sequence", "list", "series", "order", "step", "steps",
                "first", "then", "next", "finally", "procedure"
            }),
            Primitive("COLLECTION", 24, 1, {
                "collection", "set", "group", "all", "every", "each",
                "multiple", "batch", "array"
            }),
            Primitive("ONE", 25, 0, {"one", "single", "individual", "a", "an", "this"}),
            Primitive("MANY", 25, 1, {"many", "multiple", "several", "various", "some"}),
            
            # ===== NEW: QUESTION/COMMAND (dims 28-31) =====
            Primitive("QUESTION", 28, 0, {
                "how", "what", "where", "when", "why", "which", "who",
                "can", "could", "would", "should", "is", "are", "do", "does"
            }),
            Primitive("COMMAND", 28, 1, {
                "please", "now", "immediately", "must", "need", "want",
                "let", "make", "have"
            }),
            Primitive("NEGATION", 29, 0, {
                "not", "no", "never", "none", "nothing", "without", "dont",
                "cannot", "cant", "wont", "shouldnt"
            }),
            Primitive("AFFIRMATION", 29, 1, {
                "yes", "yeah", "yep", "sure", "okay", "ok", "right",
                "correct", "true", "indeed"
            }),
        ]
    
    def _build_index(self):
        self.kw_to_prim: Dict[str, List[Primitive]] = defaultdict(list)
        for p in self.primitives:
            for kw in p.keywords:
                self.kw_to_prim[kw].append(p)
    
    def encode(self, text: str) -> np.ndarray:
        pos = np.zeros(self.dimensions)
        words = re.findall(r'\b\w+\b', text.lower())
        
        for i, word in enumerate(words):
            decay = PHI ** (-i / 5)  # Slower decay
            for prim in self.kw_to_prim.get(word, []):
                val = prim.activation * decay
                if prim.dimension < self.dimensions:
                    pos[prim.dimension] = max(pos[prim.dimension], val)
        
        return pos


# =============================================================================
# LAYER 2: SYNTACTIC (16D) - NEW! Word order and structure
# =============================================================================

class SyntacticLayer:
    """
    Encodes word ORDER and syntactic patterns.
    
    Key insight: "cut the file" vs "cut the vegetables" have same words
    but different CONTEXTS. We need to capture word relationships.
    """
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        
        # Bigram patterns that indicate domain
        self.tech_bigrams = {
            ('the', 'file'), ('the', 'files'), ('the', 'directory'), ('the', 'folder'),
            ('the', 'process'), ('the', 'system'), ('the', 'command'), ('the', 'program'),
            ('in', 'terminal'), ('in', 'bash'), ('in', 'linux'), ('in', 'shell'),
            ('run', 'command'), ('execute', 'script'), ('for', 'files'), ('for', 'file'),
            ('cut', 'file'), ('cut', 'the'),  # "cut the file" context
            ('search', 'files'), ('search', 'file'), ('find', 'file'), ('find', 'files'),
            ('delete', 'file'), ('remove', 'file'), ('create', 'file'),
        }
        self.cooking_bigrams = {
            ('the', 'vegetables'), ('the', 'onions'), ('the', 'carrots'), ('the', 'meat'),
            ('the', 'ingredients'), ('the', 'sauce'), ('the', 'pot'), ('the', 'pan'),
            ('in', 'oven'), ('in', 'pan'), ('on', 'stove'), ('in', 'pot'),
            ('with', 'salt'), ('with', 'pepper'), ('with', 'butter'), ('with', 'oil'),
            ('cut', 'vegetables'), ('cut', 'onions'), ('cut', 'carrots'), ('cut', 'meat'),
            ('for', 'recipes'), ('for', 'recipe'), ('for', 'dinner'), ('for', 'cooking'),
            ('search', 'recipes'), ('search', 'recipe'), ('find', 'recipes'), ('find', 'recipe'),
        }
        self.social_bigrams = {
            ('how', 'are'), ('are', 'you'), ('thank', 'you'), ('thanks', 'for'),
            ('can', 'help'), ('i', 'feel'), ('you', 'feel'), ('how', 'you'),
            ('take', 'care'), ('good', 'morning'), ('good', 'evening'), ('good', 'night'),
            ('nice', 'to'), ('pleased', 'to'), ('glad', 'to'),
        }
    
    def encode(self, text: str) -> np.ndarray:
        pos = np.zeros(self.dimensions)
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 2:
            return pos
        
        # Extract bigrams
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        
        # Tech bigram matches (dims 0-3)
        tech_count = sum(1 for bg in bigrams if bg in self.tech_bigrams)
        pos[0] = min(tech_count * PHI / 2, PHI ** 2)
        
        # Cooking bigram matches (dims 4-7)
        cooking_count = sum(1 for bg in bigrams if bg in self.cooking_bigrams)
        pos[4] = min(cooking_count * PHI / 2, PHI ** 2)
        
        # Social bigram matches (dims 8-11)
        social_count = sum(1 for bg in bigrams if bg in self.social_bigrams)
        pos[8] = min(social_count * PHI / 2, PHI ** 2)
        
        # Word position features (dims 12-15)
        # First word type
        if words[0] in {'how', 'what', 'where', 'when', 'why', 'can', 'could'}:
            pos[12] = PHI  # Question
        elif words[0] in {'please', 'let', 'make', 'do', 'run', 'show', 'list'}:
            pos[13] = PHI  # Command
        
        # Sentence length feature
        pos[14] = min(len(words) / 10, 1.0)
        
        # Has technical symbols
        if any(c in text for c in ['-', '/', '.', '_', '*']):
            pos[15] = PHI
        
        return pos


# =============================================================================
# LAYER 3: COMPOSITIONAL (24D) - IMPROVED with domain signatures
# =============================================================================

class CompositionalLayer:
    """
    Detects domain signatures from primitive combinations.
    
    Key improvement: More specific patterns and cross-domain inhibition.
    """
    
    def __init__(self, lexical_dim: int = 32, syntactic_dim: int = 16, output_dim: int = 24):
        self.lexical_dim = lexical_dim
        self.syntactic_dim = syntactic_dim
        self.output_dim = output_dim
        
        # Domain signatures (lexical dimensions that indicate domain)
        self.domain_signatures = {
            'tech': {
                'positive': [0, 2, 3, 8, 9],  # CREATE, READ, SEARCH, FILE, DIRECTORY, PROCESS
                'negative': [12, 13]  # NOT FOOD, NOT CUT (cooking sense)
            },
            'cooking': {
                'positive': [12, 13],  # FOOD, HEAT, CUT, TASTE
                'negative': [8, 9]  # NOT FILE, NOT PROCESS
            },
            'social': {
                'positive': [16, 17],  # GREETING, GRATITUDE, FEELING, HELP
                'negative': [8, 12]  # NOT FILE, NOT FOOD
            }
        }
    
    def encode(self, lexical: np.ndarray, syntactic: np.ndarray) -> np.ndarray:
        pos = np.zeros(self.output_dim)
        
        # Domain scores with inhibition (dims 0-5)
        for i, (domain, sig) in enumerate(self.domain_signatures.items()):
            if i * 2 >= self.output_dim:
                break
            
            # Positive evidence
            pos_score = sum(lexical[d] for d in sig['positive'] if d < len(lexical))
            # Negative evidence (inhibition)
            neg_score = sum(lexical[d] for d in sig['negative'] if d < len(lexical))
            
            # Net score with inhibition
            score = max(0, pos_score - 0.5 * neg_score)
            pos[i * 2] = score
            
            # Confidence (how much positive vs negative)
            if pos_score + neg_score > 0:
                pos[i * 2 + 1] = pos_score / (pos_score + neg_score + 0.1)
        
        # Syntactic domain boost (dims 6-11)
        # Tech syntactic features boost tech domain
        pos[6] = syntactic[0] * pos[0]  # Tech bigrams * tech lexical
        pos[7] = syntactic[4] * pos[2]  # Cooking bigrams * cooking lexical
        pos[8] = syntactic[8] * pos[4]  # Social bigrams * social lexical
        
        # Action type features (dims 12-17)
        # Destructive action
        pos[12] = lexical[0] if 0 < len(lexical) else 0  # DESTROY level
        # Creative action
        pos[13] = lexical[0] if 0 < len(lexical) else 0  # CREATE level
        # Query action
        pos[14] = lexical[2] if 2 < len(lexical) else 0  # READ level
        pos[15] = lexical[3] if 3 < len(lexical) else 0  # SEARCH level
        
        # Cross-domain interaction (dims 18-23)
        # These capture when multiple domains are active (ambiguity)
        pos[18] = pos[0] * pos[2]  # Tech AND cooking (unusual)
        pos[19] = pos[0] * pos[4]  # Tech AND social
        pos[20] = pos[2] * pos[4]  # Cooking AND social
        
        # Purity score - how much ONE domain dominates
        domain_scores = [pos[0], pos[2], pos[4]]
        if sum(domain_scores) > 0:
            max_domain = max(domain_scores)
            pos[21] = max_domain / (sum(domain_scores) + 0.1)
        
        return pos


# =============================================================================
# LAYER 4: DISAMBIGUATION (16D) - Context-dependent meaning
# =============================================================================

class DisambiguationLayer:
    """
    Resolves ambiguous words using context.
    
    Key insight: Some words (cut, run, search) are ambiguous.
    Their meaning depends on WHAT they act on.
    
    This layer:
    1. Identifies ambiguous action words
    2. Looks for unambiguous context words (objects)
    3. Adjusts the embedding based on context
    """
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        
        # Ambiguous words and their possible domains
        self.ambiguous_words = {
            'cut': {'tech': ['file', 'text', 'string', 'output', 'column'],
                    'cooking': ['vegetables', 'onions', 'carrots', 'meat', 'food', 'ingredients']},
            'run': {'tech': ['program', 'script', 'command', 'process', 'code'],
                    'physical': ['store', 'marathon', 'race', 'away', 'fast']},
            'search': {'tech': ['file', 'files', 'directory', 'text', 'pattern'],
                       'cooking': ['recipe', 'recipes', 'ingredients']},
            'find': {'tech': ['file', 'files', 'directory', 'process'],
                     'cooking': ['recipe', 'recipes', 'ingredients']},
            'remove': {'tech': ['file', 'files', 'directory', 'folder'],
                       'cooking': ['seeds', 'skin', 'bones', 'stems']},
            'create': {'tech': ['file', 'directory', 'folder', 'script'],
                       'cooking': ['dish', 'meal', 'recipe']},
        }
        
        # Domain indicators (unambiguous words)
        self.domain_indicators = {
            'tech': {'file', 'files', 'directory', 'folder', 'terminal', 'command',
                     'script', 'program', 'process', 'system', 'bash', 'linux', 'code'},
            'cooking': {'vegetables', 'onions', 'carrots', 'meat', 'recipe', 'recipes',
                        'ingredients', 'oven', 'stove', 'pan', 'pot', 'kitchen'},
            'social': {'hello', 'thanks', 'feel', 'help', 'care', 'friend', 'you'},
            'physical': {'store', 'shop', 'marathon', 'race', 'gym', 'outside', 'park',
                         'street', 'walk', 'jog', 'exercise'},
        }
    
    def encode(self, text: str, compositional: np.ndarray) -> np.ndarray:
        """
        Encode disambiguation features.
        
        Output dimensions:
        0-2: Domain confidence after disambiguation (tech, cooking, social)
        3-5: Ambiguity detected flags
        6-8: Context strength (how strongly context indicates domain)
        9-15: Reserved
        """
        pos = np.zeros(self.dimensions)
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Check for ambiguous words
        ambiguous_found = []
        for word in words:
            if word in self.ambiguous_words:
                ambiguous_found.append(word)
        
        # Check for domain indicators
        domain_scores = {
            'tech': len(words & self.domain_indicators['tech']),
            'cooking': len(words & self.domain_indicators['cooking']),
            'social': len(words & self.domain_indicators['social']),
            'physical': len(words & self.domain_indicators['physical']),
        }
        
        # If ambiguous word found, use context to disambiguate
        if ambiguous_found:
            pos[3] = PHI  # Ambiguity flag
            
            for amb_word in ambiguous_found:
                amb_info = self.ambiguous_words[amb_word]
                
                # Check which domain's context words are present
                for domain, context_words in amb_info.items():
                    if any(cw in words for cw in context_words):
                        if domain == 'tech':
                            pos[0] += PHI
                            pos[6] += PHI  # Strong tech context
                        elif domain == 'cooking':
                            pos[1] += PHI
                            pos[7] += PHI  # Strong cooking context
                        elif domain == 'physical':
                            pos[2] += PHI
                            pos[8] += PHI
        
        # Add domain indicator scores
        pos[0] += domain_scores['tech'] * 0.5
        pos[1] += domain_scores['cooking'] * 0.5
        pos[2] += domain_scores['social'] * 0.5
        pos[10] += domain_scores['physical'] * 0.5  # Physical domain in dim 10
        
        # Boost from compositional layer (reinforce correct domain)
        pos[0] += compositional[0] * 0.3  # Tech from compositional
        pos[1] += compositional[2] * 0.3  # Cooking from compositional
        pos[2] += compositional[4] * 0.3  # Social from compositional
        
        # Domain purity (how much one domain dominates)
        total = pos[0] + pos[1] + pos[2]
        if total > 0:
            pos[9] = max(pos[0], pos[1], pos[2]) / total  # Purity score
        
        return pos


# =============================================================================
# LAYER 5: CONTEXTUAL (16D) - Co-occurrence learning
# =============================================================================

class ContextualLayer:
    """Co-occurrence statistics from ingested data."""
    
    def __init__(self, input_dim: int = 24, output_dim: int = 16):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cooccurrence = np.zeros((input_dim, input_dim))
        self.occurrence_count = np.zeros(input_dim)
        self.total_samples = 0
    
    def update_statistics(self, embedding: np.ndarray):
        self.total_samples += 1
        active = np.where(embedding[:self.input_dim] > 0.1)[0]
        
        for dim in active:
            self.occurrence_count[dim] += 1
        
        for i in active:
            for j in active:
                self.cooccurrence[i, j] += 1
    
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        pos = np.zeros(self.output_dim)
        
        if self.total_samples < 2:
            return embedding[:self.output_dim] if len(embedding) >= self.output_dim else np.pad(embedding, (0, self.output_dim - len(embedding)))
        
        active = np.where(embedding[:self.input_dim] > 0.1)[0]
        
        for idx, dim in enumerate(active[:self.output_dim]):
            pos[idx] = embedding[dim]
            
            p_dim = self.occurrence_count[dim] / max(self.total_samples, 1)
            
            for other_dim in active:
                if other_dim == dim or other_dim >= self.input_dim:
                    continue
                
                p_other = self.occurrence_count[other_dim] / max(self.total_samples, 1)
                p_joint = self.cooccurrence[dim, other_dim] / max(self.total_samples, 1)
                
                if p_dim > 0 and p_other > 0 and p_joint > 0:
                    pmi = np.log(p_joint / (p_dim * p_other) + 1e-10)
                    if pmi > 0:
                        pos[idx] += 0.1 * pmi * embedding[other_dim] if other_dim < len(embedding) else 0
        
        return pos


# =============================================================================
# LAYER 5: GLOBAL (8D) - Prototype distances
# =============================================================================

class GlobalLayer:
    """Distance to emergent prototypes."""
    
    def __init__(self, input_dim: int = 16, output_dim: int = 8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prototypes: List[np.ndarray] = []
        self.prototype_labels: List[str] = []
        self.max_prototypes = output_dim
    
    def update_prototypes(self, embedding: np.ndarray, label: str = None):
        if len(self.prototypes) == 0:
            self.prototypes.append(embedding.copy())
            self.prototype_labels.append(label or f"proto_0")
            return
        
        min_dist = np.inf
        nearest_idx = 0
        for i, proto in enumerate(self.prototypes):
            dist = np.sqrt(np.sum((embedding - proto) ** 2))
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if min_dist < 1.5:
            alpha = 0.1
            self.prototypes[nearest_idx] = (1 - alpha) * self.prototypes[nearest_idx] + alpha * embedding
        elif len(self.prototypes) < self.max_prototypes:
            self.prototypes.append(embedding.copy())
            self.prototype_labels.append(label or f"proto_{len(self.prototypes)}")
    
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        pos = np.zeros(self.output_dim)
        
        if not self.prototypes:
            return pos
        
        for i, proto in enumerate(self.prototypes):
            if i >= self.output_dim:
                break
            dist = np.sqrt(np.sum((embedding - proto) ** 2))
            pos[i] = 1.0 / (1.0 + dist)
        
        return pos


# =============================================================================
# STACKED LCM - PRIMARY CORE CLASS
# =============================================================================

class StackedLCM:
    """
    Stacked Geometric LCM - 128-dimensional hierarchical embedding system.
    
    This is the primary core class for the TruthSpace LCM system.
    
    Layers:
    - Morphological (16D): Word structure encoding
    - Lexical (32D): Primitive activation with φ-MAX
    - Syntactic (16D): Bigram pattern detection
    - Compositional (24D): Domain signature detection
    - Disambiguation (16D): Context-dependent meaning
    - Contextual (16D): Co-occurrence statistics
    - Global (8D): Prototype distances
    """
    
    def __init__(self):
        self.morphological = MorphologicalLayer(dimensions=16)
        self.lexical = LexicalLayer(dimensions=32)
        self.syntactic = SyntacticLayer(dimensions=16)
        self.compositional = CompositionalLayer(lexical_dim=32, syntactic_dim=16, output_dim=24)
        self.disambiguation = DisambiguationLayer(dimensions=16)  # NEW
        self.contextual = ContextualLayer(input_dim=24, output_dim=16)
        self.global_layer = GlobalLayer(input_dim=16, output_dim=8)
        
        # Layer weights (morphological weighted less, disambiguation weighted high)
        self.layer_weights = {
            'morphological': 0.3,  # Reduced further
            'lexical': 1.2,
            'syntactic': 1.0,
            'compositional': 1.2,
            'disambiguation': 2.0,  # NEW - high weight for disambiguation
            'contextual': 1.0,
            'global': 1.0
        }
        
        # Total: 16 + 32 + 16 + 24 + 16 + 16 + 8 = 128D
        self.embedding_dim = 128
        
        self.points: Dict[str, Tuple[str, np.ndarray]] = {}
        self.clusters: Dict[int, Dict] = {}
        self.cluster_threshold = 0.75
    
    def encode(self, text: str, update_stats: bool = True) -> np.ndarray:
        # Layer 0: Morphological
        morph = self.morphological.encode(text) * self.layer_weights['morphological']
        
        # Layer 1: Lexical
        lex = self.lexical.encode(text) * self.layer_weights['lexical']
        
        # Layer 2: Syntactic
        syn = self.syntactic.encode(text) * self.layer_weights['syntactic']
        
        # Layer 3: Compositional (uses both lexical and syntactic)
        comp_raw = self.compositional.encode(lex / self.layer_weights['lexical'], 
                                              syn / self.layer_weights['syntactic'])
        comp = comp_raw * self.layer_weights['compositional']
        
        # Layer 4: Disambiguation (NEW - uses text and compositional)
        disamb = self.disambiguation.encode(text, comp_raw) * self.layer_weights['disambiguation']
        
        # Layer 5: Contextual
        if update_stats:
            self.contextual.update_statistics(comp_raw)
        ctx = self.contextual.encode(comp_raw) * self.layer_weights['contextual']
        
        # Layer 6: Global
        if update_stats:
            self.global_layer.update_prototypes(ctx / self.layer_weights['contextual'])
        glob = self.global_layer.encode(ctx / self.layer_weights['contextual']) * self.layer_weights['global']
        
        return np.concatenate([morph, lex, syn, comp, disamb, ctx, glob])
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)
    
    def ingest(self, content: str, description: str = None) -> str:
        if description is None:
            description = content
        
        point_id = hashlib.md5(content.encode()).hexdigest()[:12]
        embedding = self.encode(description, update_stats=True)
        self.points[point_id] = (content, embedding)
        return point_id
    
    def ingest_batch(self, items: List[Dict[str, str]]):
        for item in items:
            self.ingest(item.get('content', ''), item.get('description'))
        self._cluster()
    
    def _cluster(self):
        if len(self.points) < 2:
            return
        
        self.clusters.clear()
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
        
        for i, (cid, pids) in enumerate(members.items()):
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
        query_emb = self.encode(query, update_stats=False)
        
        best_content = None
        best_sim = -1
        best_cluster = None
        
        for pid, (content, emb) in self.points.items():
            sim = self.cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_content = content
                for cid, cluster in self.clusters.items():
                    if pid in cluster['points']:
                        best_cluster = cluster['label']
                        break
        
        return best_content, best_sim, best_cluster
    
    def visualize(self) -> str:
        lines = ["=" * 60, "STACKED LCM (128D)", "=" * 60, ""]
        lines.append(f"Embedding dimensions: {self.embedding_dim}")
        lines.append(f"  - Morphological: 16 (weight: {self.layer_weights['morphological']})")
        lines.append(f"  - Lexical: 32 (weight: {self.layer_weights['lexical']})")
        lines.append(f"  - Syntactic: 16 (weight: {self.layer_weights['syntactic']})")
        lines.append(f"  - Compositional: 24 (weight: {self.layer_weights['compositional']})")
        lines.append(f"  - Disambiguation: 16 (weight: {self.layer_weights['disambiguation']})")
        lines.append(f"  - Contextual: 16 (weight: {self.layer_weights['contextual']})")
        lines.append(f"  - Global: 8 (weight: {self.layer_weights['global']})")
        lines.append(f"\nPoints: {len(self.points)}")
        lines.append(f"Clusters: {len(self.clusters)}")
        
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
    print("Initializing Stacked LCM (128D)...\n")
    
    lcm = StackedLCM()
    
    knowledge = [
        # Cooking
        {"content": "chop onions finely", "description": "chop cut the onions knife cooking food preparation vegetables"},
        {"content": "boil water in pot", "description": "boil water pot heat cooking stove temperature"},
        {"content": "season with salt and pepper", "description": "season with salt pepper taste cooking spices flavor"},
        {"content": "preheat oven to 350", "description": "preheat oven temperature heat baking cooking"},
        {"content": "simmer sauce for 20 minutes", "description": "simmer sauce heat low cooking time stove"},
        {"content": "mix ingredients together", "description": "mix combine the ingredients bowl cooking blend"},
        
        # Tech/Bash
        {"content": "ls -la", "description": "list show the files in directory terminal command"},
        {"content": "mkdir new_folder", "description": "create make the directory folder new terminal command"},
        {"content": "rm -rf directory", "description": "delete remove the files directory force terminal command"},
        {"content": "grep pattern file", "description": "search find pattern text in the file terminal command"},
        {"content": "cat file.txt", "description": "read show the file contents display terminal command"},
        {"content": "ps aux", "description": "list show the process running system terminal command"},
        
        # Social
        {"content": "Hello! How can I help you?", "description": "hello greeting how are you help assist welcome"},
        {"content": "Thank you so much!", "description": "thank you thanks gratitude appreciate thankful"},
        {"content": "I understand how you feel", "description": "understand how you feel feeling empathy emotion support"},
        {"content": "Take care of yourself!", "description": "take care goodbye feeling help wellbeing"},
    ]
    
    print("Ingesting knowledge...")
    lcm.ingest_batch(knowledge)
    
    print("\n" + lcm.visualize())
    
    print("\n" + "=" * 60)
    print("RESOLUTION TEST")
    print("=" * 60)
    
    queries = [
        "how do I cut the vegetables",
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
    
    print("\n" + "=" * 60)
    print("AMBIGUITY TEST (same word, different context)")
    print("=" * 60)
    
    ambiguous = [
        ("cut the file", "cut the vegetables"),
        ("search for recipes", "search for files"),
        ("run the program", "run to the store"),
    ]
    
    for t1, t2 in ambiguous:
        e1 = lcm.encode(t1, update_stats=False)
        e2 = lcm.encode(t2, update_stats=False)
        sim = lcm.cosine_similarity(e1, e2)
        print(f"\n\"{t1}\" vs \"{t2}\"")
        print(f"  Similarity: {sim:.3f} (should be LOW for different domains)")
