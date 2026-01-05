"""
Natural Language to Bash Translator - Truly Geometric

Following the Emergent Gear Pattern (Design 086):
1. STRUCTURE: Word Space + Mapping Space
2. BOOTSTRAP: Word positions from co-occurrence
3. MATCH: Query position → nearest mapping position
4. COMPOSE: Return matched command
5. LEARN: Feedback moves positions

NO string comparison at query time.
Word overlap is only used during BOOTSTRAP to build the similarity matrix.
After that, everything is position-based.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


def extract_words(text: str) -> Set[str]:
    """Extract words from text."""
    words = text.lower().split()
    words = [''.join(c for c in w if c.isalnum()) for w in words]
    return {w for w in words if w}


@dataclass
class WordPosition:
    """A word with its position in the geometric space."""
    word: str
    position: np.ndarray
    count: int = 1  # How many times this word appeared in bootstrap


class TrulyGeometricTranslator:
    """
    Truly geometric NL to Bash translator.
    
    Key difference from previous versions:
    - Words have positions (from co-occurrence during bootstrap)
    - Query encoding = average of word positions
    - Matching = nearest neighbor by POSITION distance
    - NO string comparison at query time
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        
        # Word space: word → position
        self.word_positions: Dict[str, WordPosition] = {}
        
        # Mapping space: mappings with positions
        self.mapping_structure = HyperdimensionalStructure(dims=dims, name="mappings")
        self.mappings: List[dict] = []
    
    def load_bootstrap(self, path: str) -> int:
        """
        Load bootstrap mappings and compute word positions from co-occurrence.
        
        The key insight: words that appear in similar contexts get similar positions.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        raw_mappings = data.get('mappings', [])
        
        # Step 1: Extract all unique words and their co-occurrence
        all_words = set()
        word_to_mappings: Dict[str, List[int]] = {}  # word → list of mapping indices
        
        for i, item in enumerate(raw_mappings):
            words = extract_words(item['nl'])
            all_words.update(words)
            for word in words:
                if word not in word_to_mappings:
                    word_to_mappings[word] = []
                word_to_mappings[word].append(i)
        
        word_list = sorted(all_words)
        n_words = len(word_list)
        word_to_idx = {w: i for i, w in enumerate(word_list)}
        
        # Step 2: Build word co-occurrence matrix
        # Two words are similar if they appear in similar mappings
        word_cooccurrence = np.zeros((n_words, n_words))
        
        for i, w1 in enumerate(word_list):
            mappings1 = set(word_to_mappings.get(w1, []))
            for j, w2 in enumerate(word_list):
                mappings2 = set(word_to_mappings.get(w2, []))
                # Jaccard similarity of mapping sets
                if mappings1 or mappings2:
                    intersection = len(mappings1 & mappings2)
                    union = len(mappings1 | mappings2)
                    word_cooccurrence[i, j] = intersection / union if union > 0 else 0
        
        # Step 3: Holographic projection for word positions
        eigenvalues, eigenvectors = np.linalg.eigh(word_cooccurrence)
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        word_positions_matrix = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Store word positions
        for i, word in enumerate(word_list):
            pos = word_positions_matrix[i]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            self.word_positions[word] = WordPosition(
                word=word,
                position=pos,
                count=len(word_to_mappings.get(word, []))
            )
        
        # Step 4: Compute mapping positions as average of word positions
        for i, item in enumerate(raw_mappings):
            words = extract_words(item['nl'])
            
            # Mapping position = average of word positions
            word_pos_list = [self.word_positions[w].position for w in words 
                            if w in self.word_positions]
            
            if word_pos_list:
                mapping_pos = np.mean(word_pos_list, axis=0)
                norm = np.linalg.norm(mapping_pos)
                if norm > 1e-10:
                    mapping_pos = mapping_pos / norm * CRITICAL_LINE
            else:
                mapping_pos = np.zeros(self.dims)
            
            self.mapping_structure.add(
                node_id=f"map_{i}",
                position=mapping_pos,
                data={'nl': item['nl'], 'bash': item['bash'], 'words': list(words)}
            )
            self.mappings.append(item)
        
        return len(self.mappings)
    
    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encode query to position.
        
        Query position = average of word positions.
        Unknown words are IGNORED (not a failure, just less information).
        """
        query_words = extract_words(query)
        
        if not query_words:
            return None
        
        # Get positions for known words
        known_positions = []
        for word in query_words:
            if word in self.word_positions:
                known_positions.append(self.word_positions[word].position)
        
        if not known_positions:
            # No known words - can't encode
            return None
        
        # Average position
        query_pos = np.mean(known_positions, axis=0)
        norm = np.linalg.norm(query_pos)
        if norm > 1e-10:
            query_pos = query_pos / norm * CRITICAL_LINE
        
        return query_pos
    
    def translate(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Translate query to bash command(s).
        
        TRULY GEOMETRIC:
        1. Encode query to position (average of word positions)
        2. Find nearest mappings by position distance
        3. Return top_k matches
        
        NO string comparison here!
        """
        query_pos = self._encode_query(query)
        
        if query_pos is None:
            return []
        
        # Find nearest mappings by position distance
        matches = self.mapping_structure.query_nearest(query_pos, k=top_k * 3)
        
        # Return unique commands
        seen = set()
        results = []
        for node, similarity in matches:
            if node.data and 'bash' in node.data:
                bash = node.data['bash']
                if bash not in seen:
                    seen.add(bash)
                    results.append((bash, similarity))
                    if len(results) >= top_k:
                        break
        
        return results
    
    def feedback(self, query: str, chosen_command: str, success: bool) -> None:
        """
        Provide feedback on a translation.
        
        Updates both word positions and mapping positions.
        """
        query_pos = self._encode_query(query)
        if query_pos is None:
            return
        
        query_words = extract_words(query)
        
        # Find the mapping node
        for node in self.mapping_structure:
            if node.data and node.data.get('bash') == chosen_command:
                # Update mapping position
                self.mapping_structure.feedback(node.id, query_pos, success)
                
                # Update word positions
                for word in query_words:
                    if word in self.word_positions:
                        wp = self.word_positions[word]
                        if success:
                            # Attract word toward successful mapping
                            direction = node.position - wp.position
                            wp.position = wp.position + 0.05 * direction
                        else:
                            # Repel word from failed mapping
                            direction = wp.position - node.position
                            wp.position = wp.position + 0.02 * direction
                        
                        # Renormalize
                        norm = np.linalg.norm(wp.position)
                        if norm > 1e-10:
                            wp.position = wp.position / norm * CRITICAL_LINE
                break
    
    def get_word_similarity(self, word1: str, word2: str) -> float:
        """Get similarity between two words by position distance."""
        if word1 not in self.word_positions or word2 not in self.word_positions:
            return 0.0
        
        pos1 = self.word_positions[word1].position
        pos2 = self.word_positions[word2].position
        
        # Cosine similarity
        dot = np.dot(pos1, pos2)
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            return dot / (norm1 * norm2)
        return 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'dims': self.dims,
            'words': len(self.word_positions),
            'mappings': len(self.mappings),
        }
    
    def save(self, path: str) -> None:
        """Save the translator."""
        data = {
            'type': 'TrulyGeometricTranslator',
            'version': '1.0',
            'dims': self.dims,
            'word_positions': {
                word: {
                    'position': wp.position.tolist(),
                    'count': wp.count
                }
                for word, wp in self.word_positions.items()
            },
            'mapping_structure': self.mapping_structure.to_dict(),
            'mappings': self.mappings,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrulyGeometricTranslator':
        """Load from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        translator = cls(dims=data.get('dims', 12))
        
        # Load word positions
        for word, wp_data in data.get('word_positions', {}).items():
            translator.word_positions[word] = WordPosition(
                word=word,
                position=np.array(wp_data['position']),
                count=wp_data.get('count', 1)
            )
        
        # Load mapping structure
        translator.mapping_structure = HyperdimensionalStructure.from_dict(
            data['mapping_structure']
        )
        translator.mappings = data.get('mappings', [])
        
        return translator


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Truly Geometric NL to Bash Translator ===")
    print("NO string comparison at query time.")
    print("Words have positions. Query = average of word positions.")
    print()
    
    # Load from bootstrap
    bootstrap_path = Path(__file__).parent / "bootstrap" / "nl_bash_mappings.json"
    
    translator = TrulyGeometricTranslator(dims=12)
    count = translator.load_bootstrap(str(bootstrap_path))
    
    print(f"Loaded {count} mappings")
    print(f"Learned {len(translator.word_positions)} word positions")
    print(f"Dimensions: {translator.dims}")
    print()
    
    # Show word similarities (proof that similar words have similar positions)
    print("--- Word Similarities (from co-occurrence) ---")
    word_pairs = [
        ("list", "show"),
        ("list", "display"),
        ("delete", "remove"),
        ("file", "files"),
        ("directory", "folder"),
        ("process", "processes"),
        ("kill", "terminate"),
        ("disk", "memory"),  # Should be somewhat similar (both system)
        ("list", "kill"),    # Should be dissimilar
    ]
    
    for w1, w2 in word_pairs:
        sim = translator.get_word_similarity(w1, w2)
        print(f"  {w1} ↔ {w2}: {sim:.3f}")
    print()
    
    # Test translations
    test_queries = [
        # Exact matches
        "list files",
        "show files",
        "delete file",
        "kill process",
        
        # Harder queries (novel phrasing)
        "display files",       # "display" should be near "show"
        "remove file",         # "remove" should be near "delete"
        "terminate process",   # "terminate" should be near "kill"
        "enumerate files",     # "enumerate" - unknown word
        "erase file",          # "erase" - unknown word
        
        # Indirect
        "files here",
        "running processes",
        "disk space",
        "memory usage",
    ]
    
    expected = {
        "list files": "ls",
        "show files": "ls",
        "delete file": "rm",
        "kill process": "kill",
        "display files": "ls",
        "remove file": "rm",
        "terminate process": "kill",
        "enumerate files": "ls",
        "erase file": "rm",
        "files here": "ls",
        "running processes": "ps",
        "disk space": "df",
        "memory usage": "free",
    }
    
    print("--- Translation Results ---")
    correct = 0
    total = len(test_queries)
    
    for query in test_queries:
        results = translator.translate(query, top_k=3)
        
        if results:
            best_cmd, confidence = results[0]
            base_cmd = best_cmd.split()[0]
            
            exp = expected.get(query, "")
            is_correct = base_cmd == exp or exp in best_cmd
            mark = "✓" if is_correct else "✗"
            if is_correct:
                correct += 1
            
            alts = [f"{cmd}({c:.2f})" for cmd, c in results[1:3]]
            alt_str = f" | {', '.join(alts)}" if alts else ""
            
            print(f"  {mark} '{query}'")
            print(f"      → {best_cmd} ({confidence:.2f}){alt_str}")
        else:
            print(f"  ✗ '{query}' → NO MATCH (no known words)")
        print()
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    
    # Test persistence
    print("\n--- Testing Persistence ---")
    translator.save("/tmp/truly_geometric.json")
    print("Saved to /tmp/truly_geometric.json")
    
    loaded = TrulyGeometricTranslator.load("/tmp/truly_geometric.json")
    print(f"Loaded: {len(loaded.word_positions)} words, {len(loaded.mappings)} mappings")
    
    results = loaded.translate("list files", top_k=1)
    if results:
        print(f"  'list files' → {results[0][0]} ({results[0][1]:.2f})")
    
    print("\n✓ Truly geometric translator complete!")
    print("\nKey differences from previous versions:")
    print("  - Words have positions (from co-occurrence during bootstrap)")
    print("  - Query encoding = average of word positions")
    print("  - Matching = nearest neighbor by POSITION distance")
    print("  - NO string comparison at query time")
