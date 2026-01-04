"""
Natural Language to Bash Translator - Pure Geometric

NO morphology. NO word mappings. NO cheating.
Just positions and word overlap similarity.

Philosophy (from PROJECT_OVERVIEW.md):
- Structure IS information
- Geometry IS computation
- The shape IS the knowledge

Bootstrap:
- Load mappings from JSON
- Compute positions via holographic projection (word overlap → similarity → positions)
- Query by finding nearest neighbors

The only "encoding" is word overlap - which is geometric (set intersection).

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


def extract_words(text: str) -> Set[str]:
    """
    Extract words from text.
    
    This is the ONLY text processing we do.
    No stemming, no lemmatization, no morphology.
    Just split on whitespace and lowercase.
    """
    # Split and lowercase
    words = text.lower().split()
    # Remove punctuation from each word
    words = [''.join(c for c in w if c.isalnum()) for w in words]
    # Filter empty strings
    return {w for w in words if w}


def word_overlap(words1: Set[str], words2: Set[str]) -> float:
    """
    Jaccard similarity between word sets.
    
    This is GEOMETRIC - it's set intersection / set union.
    No string matching tricks, just pure set operations.
    """
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


@dataclass
class Mapping:
    """A NL → Bash mapping with its words."""
    nl: str
    bash: str
    words: Set[str]


class PureGeometricTranslator:
    """
    Pure geometric NL to Bash translator.
    
    No morphology. No word mappings. Just:
    1. Word overlap for similarity
    2. Holographic projection for positions
    3. Nearest neighbor for query
    
    Bootstrap from JSON, then positions are computed geometrically.
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        self.structure = HyperdimensionalStructure(dims=dims, name="nl_bash")
        self.mappings: List[Mapping] = []
    
    def load_bootstrap(self, path: str) -> int:
        """
        Load bootstrap mappings from JSON and compute positions.
        
        The JSON contains NL→Bash pairs.
        Positions are computed via holographic projection from word overlap.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract mappings
        for item in data.get('mappings', []):
            nl = item['nl']
            bash = item['bash']
            words = extract_words(nl)
            self.mappings.append(Mapping(nl=nl, bash=bash, words=words))
        
        # Build similarity matrix from word overlap
        n = len(self.mappings)
        S = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                S[i, j] = word_overlap(self.mappings[i].words, self.mappings[j].words)
        
        # Holographic projection: eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Take top dims eigenvectors, scaled by sqrt(eigenvalue)
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        positions = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Add to structure
        for i, mapping in enumerate(self.mappings):
            # Normalize position to critical line
            pos = positions[i]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            
            self.structure.add(
                node_id=f"map_{i}",
                position=pos,
                data={'nl': mapping.nl, 'bash': mapping.bash, 'words': list(mapping.words)}
            )
        
        return len(self.mappings)
    
    def translate(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Translate natural language to bash command.
        
        Pure geometric:
        1. Extract words from query
        2. Compute word overlap with each mapping
        3. Use overlap as "similarity" to find nearest
        
        No position encoding of the query - we use word overlap directly.
        """
        query_words = extract_words(query)
        
        if not query_words:
            return []
        
        # Compute similarity to each mapping
        results = []
        for node in self.structure:
            if node.data and 'words' in node.data:
                mapping_words = set(node.data['words'])
                similarity = word_overlap(query_words, mapping_words)
                if similarity > 0:
                    results.append((node.data['bash'], similarity, node))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k unique commands
        seen = set()
        unique_results = []
        for bash, sim, node in results:
            if bash not in seen:
                seen.add(bash)
                unique_results.append((bash, sim))
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def feedback(self, query: str, chosen_command: str, success: bool) -> None:
        """
        Provide feedback on a translation.
        
        If success: Move matching nodes toward query position
        If failure: Move matching nodes away from query position
        
        But wait - we don't have a query position!
        
        Solution: Use the average position of nodes that match the query words.
        This is still geometric - we're using the structure itself.
        """
        query_words = extract_words(query)
        
        # Find nodes that contributed to this result
        matching_nodes = []
        for node in self.structure:
            if node.data and node.data.get('bash') == chosen_command:
                mapping_words = set(node.data.get('words', []))
                if word_overlap(query_words, mapping_words) > 0:
                    matching_nodes.append(node)
        
        if not matching_nodes:
            return
        
        # Compute query position as average of matching node positions
        query_position = np.mean([n.position for n in matching_nodes], axis=0)
        
        # Apply feedback
        for node in matching_nodes:
            self.structure.feedback(node.id, query_position, success)
    
    def add_mapping(self, nl: str, bash: str) -> Node:
        """
        Add a new mapping.
        
        Position is computed from word overlap with existing mappings.
        """
        words = extract_words(nl)
        
        # Compute similarity to existing mappings
        similarities = []
        for node in self.structure:
            if node.data and 'words' in node.data:
                mapping_words = set(node.data['words'])
                sim = word_overlap(words, mapping_words)
                similarities.append((node, sim))
        
        # Position is weighted average of similar mappings
        if similarities:
            total_weight = sum(sim for _, sim in similarities)
            if total_weight > 0:
                position = np.zeros(self.dims)
                for node, sim in similarities:
                    position += node.position * sim
                position /= total_weight
            else:
                # No overlap - random position
                position = np.random.randn(self.dims)
        else:
            # First mapping - random position
            position = np.random.randn(self.dims)
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 1e-10:
            position = position / norm * CRITICAL_LINE
        
        # Add to structure
        mapping = Mapping(nl=nl, bash=bash, words=words)
        self.mappings.append(mapping)
        
        return self.structure.add(
            node_id=f"map_{len(self.mappings)-1}",
            position=position,
            data={'nl': nl, 'bash': bash, 'words': list(words)}
        )
    
    def save(self, path: str) -> None:
        """Save the structure to JSON."""
        self.structure.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'PureGeometricTranslator':
        """Load from saved structure."""
        translator = cls()
        translator.structure = HyperdimensionalStructure.load(path)
        
        # Rebuild mappings list
        for node in translator.structure:
            if node.data:
                translator.mappings.append(Mapping(
                    nl=node.data.get('nl', ''),
                    bash=node.data.get('bash', ''),
                    words=set(node.data.get('words', []))
                ))
        
        return translator
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'mappings': len(self.mappings),
            'structure': self.structure.stats(),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Pure Geometric NL to Bash Translator ===")
    print("No morphology. No word mappings. Just geometry.")
    print()
    
    # Load from bootstrap JSON
    bootstrap_path = Path(__file__).parent / "bootstrap" / "nl_bash_mappings.json"
    
    translator = PureGeometricTranslator(dims=12)
    count = translator.load_bootstrap(str(bootstrap_path))
    
    print(f"Loaded {count} mappings from bootstrap")
    print(f"Dimensions: {translator.dims}")
    print()
    
    # Test translations
    test_queries = [
        "list files",
        "show files",
        "display files",
        "what files are here",
        "files in directory",
        "delete file",
        "remove file",
        "show processes",
        "running processes",
        "kill process",
        "stop process",
        "disk space",
        "disk usage",
        "memory usage",
        "check memory",
        "network connections",
        "ip address",
        "create folder",
        "make directory",
        "find files",
    ]
    
    # Expected results for accuracy calculation
    expected = {
        "list files": "ls",
        "show files": "ls",
        "display files": "ls",
        "what files are here": "ls",  # Might not match well
        "files in directory": "ls",
        "delete file": "rm",
        "remove file": "rm",
        "show processes": "ps",
        "running processes": "ps",
        "kill process": "kill",
        "stop process": "kill",
        "disk space": "df",
        "disk usage": "df",
        "memory usage": "free",
        "check memory": "free",
        "network connections": "netstat",
        "ip address": "ip",
        "create folder": "mkdir",
        "make directory": "mkdir",
        "find files": "find",
    }
    
    print("--- Translation Results ---")
    correct = 0
    total = 0
    
    for query in test_queries:
        results = translator.translate(query, top_k=3)
        total += 1
        
        if results:
            best_cmd, confidence = results[0]
            base_cmd = best_cmd.split()[0]
            
            # Check correctness
            exp = expected.get(query, "")
            is_correct = base_cmd == exp or exp in best_cmd
            mark = "✓" if is_correct else "✗"
            if is_correct:
                correct += 1
            
            # Format alternatives
            alts = [cmd for cmd, _ in results[1:3]]
            alt_str = f" (also: {', '.join(alts)})" if alts else ""
            
            print(f"  {mark} '{query}'")
            print(f"      → {best_cmd} ({confidence:.2f}){alt_str}")
        else:
            print(f"  ✗ '{query}' → NO MATCH")
        print()
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    
    # Test persistence
    print("\n--- Testing Persistence ---")
    translator.save("/tmp/pure_geometric.json")
    print("Saved to /tmp/pure_geometric.json")
    
    loaded = PureGeometricTranslator.load("/tmp/pure_geometric.json")
    print(f"Loaded: {len(loaded.mappings)} mappings")
    
    results = loaded.translate("list files", top_k=1)
    if results:
        print(f"  'list files' → {results[0][0]} ({results[0][1]:.2f})")
    
    # Test learning
    print("\n--- Testing Learning ---")
    print("Adding new mapping: 'show directory contents' → 'ls -la'")
    translator.add_mapping("show directory contents", "ls -la")
    
    results = translator.translate("show directory contents", top_k=1)
    if results:
        print(f"  'show directory contents' → {results[0][0]} ({results[0][1]:.2f})")
    
    print("\n✓ Pure geometric translator complete!")
    print("\nKey points:")
    print("  - NO morphology (no stemming, lemmatization)")
    print("  - NO word mappings (no hard-coded semantics)")
    print("  - Bootstrap from JSON")
    print("  - Positions from holographic projection")
    print("  - Query by word overlap (geometric)")
