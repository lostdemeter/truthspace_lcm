"""
Natural Language to Bash Translator - Version 2

Key insight: Don't hard-code the encoding. Let the STRUCTURE learn it.

Instead of:
  NL → [hard-coded word mappings] → position

Do:
  NL → [word positions from structure] → average position

The word positions are learned from the training data, not hard-coded.
This is more aligned with the hyperdimensional paradigm.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  "list files"                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                    │
│  │ Word Space  │  ← Words have positions (learned from usage)       │
│  │ (Structure) │                                                    │
│  └─────────────┘                                                    │
│       │                                                             │
│       ▼ average word positions                                      │
│  ┌─────────────┐                                                    │
│  │ NL→Bash     │  ← Mappings have positions                         │
│  │ (Structure) │                                                    │
│  └─────────────┘                                                    │
│       │                                                             │
│       ▼ nearest neighbor                                            │
│      "ls"                                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime
import hashlib

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


# Filler words to ignore
FILLER = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
          'that', 'this', 'is', 'are', 'it', 'be', 'can', 'you', 'i', 'me',
          'my', 'your', 'please', 'could', 'would', 'should', 'do', 'does',
          'what', 'how', 'where', 'when', 'why', 'which', 'who'}


def extract_words(text: str) -> Set[str]:
    """Extract content words from text."""
    words = text.lower().split()
    # Remove punctuation
    words = [''.join(c for c in w if c.isalnum()) for w in words]
    return {w for w in words if w and w not in FILLER and len(w) > 1}


def word_to_seed(word: str) -> int:
    """Convert word to deterministic seed for random position."""
    return int(hashlib.md5(word.encode()).hexdigest()[:8], 16)


class NLToBashTranslatorV2:
    """
    Natural Language to Bash translator using learned word positions.
    
    Key difference from V1: Word positions are LEARNED, not hard-coded.
    
    Three structures:
    1. word_space: Maps words to positions (learned from co-occurrence)
    2. mapping_space: Maps NL→Bash pairs to positions
    3. (optional) bash_space: For reverse translation
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        
        # Word space - words get positions based on co-occurrence
        self.word_space = HyperdimensionalStructure(dims=dims, name="word_space")
        
        # Mapping space - NL→Bash mappings
        self.mapping_space = HyperdimensionalStructure(dims=dims, name="mapping_space")
    
    def _get_word_position(self, word: str) -> np.ndarray:
        """
        Get position for a word.
        
        If word exists in word_space, return its position.
        Otherwise, create a deterministic random position.
        """
        node = self.word_space.get(word)
        if node:
            return node.position
        
        # Create deterministic position from word hash
        np.random.seed(word_to_seed(word))
        pos = np.random.randn(self.dims)
        pos = pos / np.linalg.norm(pos) * CRITICAL_LINE
        return pos
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to position by averaging word positions.
        
        This is the stateless encoding function.
        """
        words = extract_words(text)
        
        if not words:
            return np.zeros(self.dims)
        
        # Average word positions
        positions = [self._get_word_position(w) for w in words]
        avg_position = np.mean(positions, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_position)
        if norm > 1e-10:
            avg_position = avg_position / norm * CRITICAL_LINE
        
        return avg_position
    
    def add_mapping(self, nl_text: str, bash_command: str) -> Node:
        """
        Add a NL → Bash mapping.
        
        Also updates word positions based on co-occurrence.
        """
        # Extract words from both
        nl_words = extract_words(nl_text)
        bash_words = extract_words(bash_command)
        all_words = nl_words | bash_words
        
        # Ensure all words exist in word_space
        for word in all_words:
            if word not in self.word_space:
                pos = self._get_word_position(word)
                self.word_space.add(word, position=pos, data={'word': word})
        
        # Words that appear together should attract
        # This is how word positions become meaningful
        word_list = list(all_words)
        for i, w1 in enumerate(word_list):
            for w2 in word_list[i+1:]:
                # Attract co-occurring words
                pos1 = self.word_space.get(w1).position
                pos2 = self.word_space.get(w2).position
                self.word_space.attract(w1, pos2, strength=0.05)
                self.word_space.attract(w2, pos1, strength=0.05)
        
        # Encode the NL text
        position = self._encode_text(nl_text)
        
        # Add mapping
        node_id = f"map_{len(self.mapping_space)}"
        return self.mapping_space.add(
            node_id=node_id,
            position=position,
            data={'nl': nl_text, 'bash': bash_command}
        )
    
    def translate(self, nl_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Translate natural language to bash command(s)."""
        position = self._encode_text(nl_text)
        matches = self.mapping_space.query_nearest(position, k=top_k)
        
        results = []
        for node, similarity in matches:
            if node.data and 'bash' in node.data:
                results.append((node.data['bash'], similarity))
        
        return results
    
    def feedback(self, nl_text: str, chosen_command: str, success: bool) -> None:
        """
        Provide feedback on a translation.
        
        Updates both the mapping position AND the word positions.
        """
        position = self._encode_text(nl_text)
        words = extract_words(nl_text)
        
        # Find and update the mapping
        for node in self.mapping_space:
            if node.data and node.data.get('bash') == chosen_command:
                self.mapping_space.feedback(node.id, position, success)
                
                # Also update word positions
                # If success: words should move toward the mapping
                # If failure: words should move away
                for word in words:
                    if word in self.word_space:
                        self.word_space.feedback(word, node.position, success,
                                                  attract_strength=0.02,
                                                  repel_strength=0.01)
                break
    
    def seed_common_commands(self) -> None:
        """Seed with common NL → Bash mappings."""
        mappings = [
            # File listing
            ("list files", "ls"),
            ("show files", "ls"),
            ("list all files", "ls -la"),
            ("show hidden files", "ls -a"),
            ("list files with details", "ls -l"),
            ("list files recursively", "ls -R"),
            ("find files", "find . -type f"),
            ("find directories", "find . -type d"),
            ("search for files named", "find . -name"),
            
            # File operations
            ("create file", "touch"),
            ("make directory", "mkdir"),
            ("create directory", "mkdir"),
            ("create folder", "mkdir"),
            ("make folder", "mkdir"),
            ("remove file", "rm"),
            ("delete file", "rm"),
            ("remove directory", "rmdir"),
            ("delete directory recursively", "rm -rf"),
            ("copy file", "cp"),
            ("move file", "mv"),
            ("rename file", "mv"),
            
            # File content
            ("show file contents", "cat"),
            ("display file", "cat"),
            ("read file", "cat"),
            ("show first lines", "head"),
            ("show last lines", "tail"),
            ("count lines", "wc -l"),
            ("count words", "wc -w"),
            
            # Process management
            ("list processes", "ps aux"),
            ("show processes", "ps aux"),
            ("show running processes", "ps aux"),
            ("display processes", "ps aux"),
            ("find process", "pgrep"),
            ("kill process", "kill"),
            ("terminate process", "kill -9"),
            ("stop process", "kill"),
            ("stop all processes named", "pkill"),
            
            # System info
            ("show disk usage", "df -h"),
            ("check disk space", "df -h"),
            ("disk space", "df -h"),
            ("show memory usage", "free -h"),
            ("check memory", "free -h"),
            ("memory usage", "free -h"),
            ("system uptime", "uptime"),
            ("show system info", "uname -a"),
            
            # Network
            ("show network connections", "netstat -tuln"),
            ("list open ports", "ss -tuln"),
            ("show listening ports", "lsof -i"),
            ("network interfaces", "ip addr"),
            ("show ip address", "ip addr"),
            ("ip address", "ip addr"),
        ]
        
        for nl, bash in mappings:
            self.add_mapping(nl, bash)
    
    def stats(self) -> Dict[str, Any]:
        """Get translator statistics."""
        return {
            'dims': self.dims,
            'word_space': self.word_space.stats(),
            'mapping_space': self.mapping_space.stats(),
        }
    
    def save(self, path: str) -> None:
        """Save structures."""
        import json
        from pathlib import Path
        
        data = {
            'type': 'NLToBashTranslatorV2',
            'version': '2.0',
            'dims': self.dims,
            'word_space': self.word_space.to_dict(),
            'mapping_space': self.mapping_space.to_dict(),
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'NLToBashTranslatorV2':
        """Load from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        translator = cls(dims=data.get('dims', 12))
        translator.word_space = HyperdimensionalStructure.from_dict(data['word_space'])
        translator.mapping_space = HyperdimensionalStructure.from_dict(data['mapping_space'])
        
        return translator


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== NL to Bash Translator V2 (Learned Word Positions) ===\n")
    
    # Create and seed translator
    translator = NLToBashTranslatorV2(dims=12)
    translator.seed_common_commands()
    
    print(f"Mappings: {len(translator.mapping_space)}")
    print(f"Words learned: {len(translator.word_space)}")
    print(f"Dimensions: {translator.dims}")
    print()
    
    # Test translations
    test_queries = [
        "list all files in directory",
        "show me the files",
        "what files are here",
        "delete the file",
        "remove this directory",
        "show running processes",
        "list processes",
        "kill the process",
        "how much disk space",
        "check memory usage",
        "show network connections",
        "what's my ip address",
        "create a new folder",
        "make a directory called test",
        "find all python files",
    ]
    
    print("--- Translation Results ---")
    correct = 0
    total = len(test_queries)
    
    expected = {
        "list all files in directory": "ls",
        "show me the files": "ls",
        "what files are here": "ls",
        "delete the file": "rm",
        "remove this directory": "rm",
        "show running processes": "ps",
        "list processes": "ps",
        "kill the process": "kill",
        "how much disk space": "df",
        "check memory usage": "free",
        "show network connections": "netstat",
        "what's my ip address": "ip",
        "create a new folder": "mkdir",
        "make a directory called test": "mkdir",
        "find all python files": "find",
    }
    
    for query in test_queries:
        results = translator.translate(query, top_k=3)
        if results:
            best_cmd, confidence = results[0]
            # Check if correct (base command matches)
            exp = expected.get(query, "")
            is_correct = best_cmd.split()[0] == exp or exp in best_cmd
            mark = "✓" if is_correct else "✗"
            if is_correct:
                correct += 1
            
            alternatives = [cmd for cmd, _ in results[1:3]]
            alt_str = f" (also: {', '.join(alternatives)})" if alternatives else ""
            print(f"  {mark} '{query}'")
            print(f"      → {best_cmd} ({confidence:.2f}){alt_str}")
        else:
            print(f"  ✗ '{query}' → NO MATCH")
        print()
    
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Show word clustering
    print("\n--- Word Space Analysis ---")
    print("Words that should be similar (co-occurred in training):")
    
    # Check similarity between related words
    word_pairs = [
        ("list", "show"),
        ("files", "directory"),
        ("process", "processes"),
        ("kill", "terminate"),
        ("disk", "space"),
        ("memory", "usage"),
    ]
    
    for w1, w2 in word_pairs:
        if w1 in translator.word_space and w2 in translator.word_space:
            pos1 = translator.word_space.get(w1).position
            pos2 = translator.word_space.get(w2).position
            sim = np.dot(pos1, pos2) / (np.linalg.norm(pos1) * np.linalg.norm(pos2))
            print(f"  {w1} ↔ {w2}: {sim:.3f}")
    
    print("\n✓ V2 translator complete!")
    print("\nKey insight: Word positions are LEARNED from co-occurrence,")
    print("not hard-coded. This is more aligned with the paradigm.")
