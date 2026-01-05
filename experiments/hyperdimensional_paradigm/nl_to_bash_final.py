"""
Natural Language to Bash Translator - Final Version

Combines all improvements:
1. Truly geometric (no string comparison at query time)
2. Synonym expansion from bootstrap JSON
3. Temporary word injection (Design 085) for unknown words

Following the Emergent Gear Pattern (Design 086):
1. STRUCTURE: Word Space + Mapping Space
2. BOOTSTRAP: Word positions from co-occurrence + synonyms
3. MATCH: Query position → nearest mapping position
4. COMPOSE: Return matched command
5. LEARN: Feedback moves positions + promotes temporary words

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

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
    count: int = 1
    temporary: bool = False  # Temporary words can be promoted or pruned
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.position))
    
    @property
    def persists(self) -> bool:
        """Temporary words persist if they cross the critical line."""
        return self.magnitude >= CRITICAL_LINE


class FinalGeometricTranslator:
    """
    Final geometric NL to Bash translator.
    
    Features:
    1. Synonym expansion during bootstrap
    2. Temporary word injection for unknowns
    3. Learning promotes successful temporary words
    4. Truly geometric matching (no string comparison at query time)
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        
        # Word space
        self.word_positions: Dict[str, WordPosition] = {}
        
        # Synonym groups (for giving unknown words positions)
        self.synonym_groups: List[Set[str]] = []
        
        # Mapping space
        self.mapping_structure = HyperdimensionalStructure(dims=dims, name="mappings")
        self.mappings: List[dict] = []
    
    def load_bootstrap(self, path: str) -> int:
        """
        Load bootstrap with synonym expansion.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        raw_mappings = data.get('mappings', [])
        synonym_lists = data.get('synonyms', [])
        
        # Store synonym groups
        self.synonym_groups = [set(group) for group in synonym_lists]
        
        # Step 1: Extract all words from mappings
        all_words = set()
        word_to_mappings: Dict[str, List[int]] = {}
        
        for i, item in enumerate(raw_mappings):
            words = extract_words(item['nl'])
            all_words.update(words)
            for word in words:
                if word not in word_to_mappings:
                    word_to_mappings[word] = []
                word_to_mappings[word].append(i)
        
        # Step 2: Expand vocabulary with synonyms
        # Words in the same synonym group should appear in the same mappings
        for group in self.synonym_groups:
            # Find all mappings that contain any word from this group
            group_mappings = set()
            for word in group:
                if word in word_to_mappings:
                    group_mappings.update(word_to_mappings[word])
            
            # Add all words in the group to all those mappings
            for word in group:
                all_words.add(word)
                if word not in word_to_mappings:
                    word_to_mappings[word] = []
                word_to_mappings[word] = list(set(word_to_mappings[word]) | group_mappings)
        
        word_list = sorted(all_words)
        n_words = len(word_list)
        word_to_idx = {w: i for i, w in enumerate(word_list)}
        
        # Step 3: Build word co-occurrence matrix
        word_cooccurrence = np.zeros((n_words, n_words))
        
        for i, w1 in enumerate(word_list):
            mappings1 = set(word_to_mappings.get(w1, []))
            for j, w2 in enumerate(word_list):
                mappings2 = set(word_to_mappings.get(w2, []))
                if mappings1 or mappings2:
                    intersection = len(mappings1 & mappings2)
                    union = len(mappings1 | mappings2)
                    word_cooccurrence[i, j] = intersection / union if union > 0 else 0
        
        # Step 4: Holographic projection for word positions
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
                count=len(word_to_mappings.get(word, [])),
                temporary=False
            )
        
        # Step 5: Compute mapping positions
        for i, item in enumerate(raw_mappings):
            words = extract_words(item['nl'])
            
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
    
    def _find_synonym_position(self, unknown_word: str) -> Optional[np.ndarray]:
        """
        Try to find a position for an unknown word via synonyms.
        
        If the unknown word is in a synonym group with known words,
        use the average position of the known synonyms.
        """
        for group in self.synonym_groups:
            if unknown_word in group:
                # Find known words in this group
                known_positions = []
                for word in group:
                    if word in self.word_positions and not self.word_positions[word].temporary:
                        known_positions.append(self.word_positions[word].position)
                
                if known_positions:
                    avg_pos = np.mean(known_positions, axis=0)
                    norm = np.linalg.norm(avg_pos)
                    if norm > 1e-10:
                        avg_pos = avg_pos / norm * CRITICAL_LINE
                    return avg_pos
        
        return None
    
    def _inject_temporary_word(self, word: str, context_words: Set[str]) -> WordPosition:
        """
        Inject a temporary word (Design 085).
        
        Position is computed from context words that ARE known.
        The temporary word starts below the critical line and can be promoted.
        """
        # Get positions of known context words
        context_positions = []
        for ctx_word in context_words:
            if ctx_word in self.word_positions:
                context_positions.append(self.word_positions[ctx_word].position)
        
        if context_positions:
            # Position = average of context, but scaled down (below critical line)
            pos = np.mean(context_positions, axis=0)
            # Start at 80% of critical line (temporary)
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * (CRITICAL_LINE * 0.4)
        else:
            # No context - random position below critical line
            pos = np.random.randn(self.dims)
            pos = pos / np.linalg.norm(pos) * (CRITICAL_LINE * 0.3)
        
        wp = WordPosition(
            word=word,
            position=pos,
            count=0,
            temporary=True
        )
        
        self.word_positions[word] = wp
        return wp
    
    def _encode_query(self, query: str, inject_unknown: bool = True) -> Optional[np.ndarray]:
        """
        Encode query to position.
        
        For unknown words:
        1. Try synonym lookup
        2. If inject_unknown=True, inject temporary word
        3. Otherwise, ignore unknown word
        """
        query_words = extract_words(query)
        
        if not query_words:
            return None
        
        known_positions = []
        unknown_words = []
        
        for word in query_words:
            if word in self.word_positions:
                known_positions.append(self.word_positions[word].position)
            else:
                # Try synonym lookup first
                syn_pos = self._find_synonym_position(word)
                if syn_pos is not None:
                    known_positions.append(syn_pos)
                    # Also inject as temporary with this position
                    if inject_unknown:
                        wp = WordPosition(
                            word=word,
                            position=syn_pos.copy(),
                            count=0,
                            temporary=True
                        )
                        self.word_positions[word] = wp
                else:
                    unknown_words.append(word)
        
        # Inject temporary words for truly unknown words
        if inject_unknown and unknown_words:
            known_context = query_words - set(unknown_words)
            for word in unknown_words:
                wp = self._inject_temporary_word(word, known_context)
                known_positions.append(wp.position)
        
        if not known_positions:
            return None
        
        query_pos = np.mean(known_positions, axis=0)
        norm = np.linalg.norm(query_pos)
        if norm > 1e-10:
            query_pos = query_pos / norm * CRITICAL_LINE
        
        return query_pos
    
    def translate(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Translate query to bash command(s).
        
        Truly geometric with temporary word injection.
        """
        query_pos = self._encode_query(query, inject_unknown=True)
        
        if query_pos is None:
            return []
        
        matches = self.mapping_structure.query_nearest(query_pos, k=top_k * 3)
        
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
        
        Updates word positions and promotes/demotes temporary words.
        """
        query_pos = self._encode_query(query, inject_unknown=False)
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
                            wp.success_count += 1
                            # Attract word toward successful mapping
                            direction = node.position - wp.position
                            wp.position = wp.position + 0.1 * direction
                        else:
                            wp.failure_count += 1
                            # Repel word from failed mapping
                            direction = wp.position - node.position
                            wp.position = wp.position + 0.05 * direction
                        
                        # Renormalize - but allow growth beyond critical line
                        norm = np.linalg.norm(wp.position)
                        if norm > 1e-10:
                            # Successful temporary words grow toward critical line
                            if wp.temporary and success:
                                target_mag = min(CRITICAL_LINE, norm * 1.1)
                            else:
                                target_mag = CRITICAL_LINE
                            wp.position = wp.position / norm * target_mag
                        
                        # Check for promotion
                        if wp.temporary and wp.persists:
                            wp.temporary = False
                break
    
    def prune_temporary(self, min_success: int = 0) -> int:
        """
        Prune temporary words that haven't been successful.
        
        Returns number of words pruned.
        """
        to_prune = []
        for word, wp in self.word_positions.items():
            if wp.temporary and wp.success_count <= min_success:
                to_prune.append(word)
        
        for word in to_prune:
            del self.word_positions[word]
        
        return len(to_prune)
    
    def get_word_similarity(self, word1: str, word2: str) -> float:
        """Get similarity between two words by position distance."""
        if word1 not in self.word_positions or word2 not in self.word_positions:
            return 0.0
        
        pos1 = self.word_positions[word1].position
        pos2 = self.word_positions[word2].position
        
        dot = np.dot(pos1, pos2)
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            return dot / (norm1 * norm2)
        return 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        permanent = sum(1 for wp in self.word_positions.values() if not wp.temporary)
        temporary = sum(1 for wp in self.word_positions.values() if wp.temporary)
        
        return {
            'dims': self.dims,
            'words_permanent': permanent,
            'words_temporary': temporary,
            'words_total': len(self.word_positions),
            'mappings': len(self.mappings),
            'synonym_groups': len(self.synonym_groups),
        }
    
    def save(self, path: str) -> None:
        """Save the translator."""
        data = {
            'type': 'FinalGeometricTranslator',
            'version': '1.0',
            'dims': self.dims,
            'synonym_groups': [list(g) for g in self.synonym_groups],
            'word_positions': {
                word: {
                    'position': wp.position.tolist(),
                    'count': wp.count,
                    'temporary': wp.temporary,
                    'success_count': wp.success_count,
                    'failure_count': wp.failure_count,
                }
                for word, wp in self.word_positions.items()
            },
            'mapping_structure': self.mapping_structure.to_dict(),
            'mappings': self.mappings,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FinalGeometricTranslator':
        """Load from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        translator = cls(dims=data.get('dims', 12))
        translator.synonym_groups = [set(g) for g in data.get('synonym_groups', [])]
        
        for word, wp_data in data.get('word_positions', {}).items():
            translator.word_positions[word] = WordPosition(
                word=word,
                position=np.array(wp_data['position']),
                count=wp_data.get('count', 1),
                temporary=wp_data.get('temporary', False),
                success_count=wp_data.get('success_count', 0),
                failure_count=wp_data.get('failure_count', 0),
            )
        
        translator.mapping_structure = HyperdimensionalStructure.from_dict(
            data['mapping_structure']
        )
        translator.mappings = data.get('mappings', [])
        
        return translator


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Final Geometric NL to Bash Translator ===")
    print("Features: Synonyms + Temporary Injection + Truly Geometric")
    print()
    
    # Load from bootstrap
    bootstrap_path = Path(__file__).parent / "bootstrap" / "nl_bash_mappings.json"
    
    translator = FinalGeometricTranslator(dims=12)
    count = translator.load_bootstrap(str(bootstrap_path))
    
    stats = translator.stats()
    print(f"Loaded {count} mappings")
    print(f"Words: {stats['words_permanent']} permanent, {stats['words_temporary']} temporary")
    print(f"Synonym groups: {stats['synonym_groups']}")
    print()
    
    # Show word similarities (including synonyms)
    print("--- Word Similarities (with synonym expansion) ---")
    word_pairs = [
        ("list", "show"),
        ("list", "enumerate"),
        ("list", "exhibit"),
        ("delete", "remove"),
        ("delete", "erase"),
        ("delete", "destroy"),
        ("kill", "terminate"),
        ("kill", "halt"),
        ("directory", "folder"),
        ("process", "task"),
        ("process", "application"),
    ]
    
    for w1, w2 in word_pairs:
        sim = translator.get_word_similarity(w1, w2)
        print(f"  {w1} ↔ {w2}: {sim:.3f}")
    print()
    
    # Test translations (including previously failing queries)
    test_queries = [
        # Exact matches
        "list files",
        "show files",
        "delete file",
        "kill process",
        
        # Synonym-based (should now work)
        "enumerate files",
        "exhibit files",
        "erase file",
        "destroy file",
        "terminate process",
        "halt process",
        
        # Novel words (will be injected as temporary)
        "obliterate file",
        "annihilate process",
        
        # Harder queries
        "running processes",
        "disk space",
        "memory usage",
        "network connections",
    ]
    
    expected = {
        "list files": "ls",
        "show files": "ls",
        "delete file": "rm",
        "kill process": "kill",
        "enumerate files": "ls",
        "exhibit files": "ls",
        "erase file": "rm",
        "destroy file": "rm",
        "terminate process": "kill",
        "halt process": "kill",
        "obliterate file": "rm",
        "annihilate process": "kill",
        "running processes": "ps",
        "disk space": "df",
        "memory usage": "free",
        "network connections": "netstat",
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
            print(f"  ✗ '{query}' → NO MATCH")
        print()
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    
    # Show temporary words that were injected
    stats = translator.stats()
    print(f"\n--- Temporary Words Injected ---")
    print(f"Total: {stats['words_temporary']}")
    temp_words = [w for w, wp in translator.word_positions.items() if wp.temporary]
    if temp_words:
        print(f"Words: {', '.join(temp_words[:10])}")
    
    # Test learning
    print("\n--- Testing Learning ---")
    print("Providing positive feedback for 'obliterate file' → rm")
    for _ in range(5):
        translator.feedback("obliterate file", "rm", success=True)
    
    # Check if 'obliterate' was promoted
    if 'obliterate' in translator.word_positions:
        wp = translator.word_positions['obliterate']
        print(f"  'obliterate': temporary={wp.temporary}, magnitude={wp.magnitude:.3f}")
        print(f"  success_count={wp.success_count}, persists={wp.persists}")
    
    # Test persistence
    print("\n--- Testing Persistence ---")
    translator.save("/tmp/final_geometric.json")
    print("Saved to /tmp/final_geometric.json")
    
    loaded = FinalGeometricTranslator.load("/tmp/final_geometric.json")
    stats = loaded.stats()
    print(f"Loaded: {stats['words_total']} words, {stats['mappings']} mappings")
    
    results = loaded.translate("erase file", top_k=1)
    if results:
        print(f"  'erase file' → {results[0][0]} ({results[0][1]:.2f})")
    
    print("\n✓ Final geometric translator complete!")
    print("\nKey features:")
    print("  - Synonym expansion from bootstrap JSON")
    print("  - Temporary word injection for unknowns")
    print("  - Learning promotes successful temporary words")
    print("  - Truly geometric matching (no string comparison at query time)")
