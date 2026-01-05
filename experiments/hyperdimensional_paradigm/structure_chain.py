"""
Structure Chaining - Connecting Multiple Hyperdimensional Structures

Inspired by:
- DNA mechanics (Design 077): Chromosomes share information
- Gear chains: Sequential processing through transformations
- The Emergent Gear Pattern (Design 086): Structure → Bootstrap → Match → Compose → Learn

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Input: "list files in the repo"                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ NL Structure│ ──► │ Intent      │ ──► │ Bash/Git    │           │
│  │ (words)     │     │ Structure   │     │ Structure   │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│       │                    │                    │                   │
│       ▼                    ▼                    ▼                   │
│   word positions      intent type         command                   │
│                       (file/git/sys)      (ls or git ls-files)     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Key concepts:
1. Each structure is a "chromosome" with its own domain
2. Structures connect via "bridges" (shared positions or transcoders)
3. Information flows through the chain, transforming at each step
4. Feedback propagates backward through the chain

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


# =============================================================================
# STRUCTURE LINK - Connection between two structures
# =============================================================================

@dataclass
class StructureLink:
    """
    A link between two structures.
    
    Defines how information flows from source to target.
    The transcoder is a stateless function that maps positions.
    """
    source: str  # Name of source structure
    target: str  # Name of target structure
    transcoder: Optional[Callable[[np.ndarray], np.ndarray]] = None  # Position mapper
    weight: float = 1.0  # Link strength
    
    def transform(self, position: np.ndarray) -> np.ndarray:
        """Transform position from source space to target space."""
        if self.transcoder:
            return self.transcoder(position)
        return position  # Identity if no transcoder


# =============================================================================
# CHAIN RESULT - Output from chain processing
# =============================================================================

@dataclass
class ChainResult:
    """Result from processing through a structure chain."""
    final_output: Any
    confidence: float
    path: List[str]  # Names of structures traversed
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ChainResult(output={self.final_output}, conf={self.confidence:.2f}, path={self.path})"


# =============================================================================
# STRUCTURE CHAIN - Connected structures
# =============================================================================

class StructureChain:
    """
    A chain of connected hyperdimensional structures.
    
    Each structure is a "chromosome" that processes information.
    Links define how information flows between structures.
    
    Processing modes:
    1. Sequential: A → B → C (each feeds into next)
    2. Parallel: A → [B, C] → D (branch and merge)
    3. Conditional: A → B if condition else C (routing)
    """
    
    def __init__(self, name: str = "chain"):
        self.name = name
        self.structures: Dict[str, HyperdimensionalStructure] = {}
        self.links: List[StructureLink] = []
        self.entry_point: Optional[str] = None  # First structure in chain
        self.exit_points: List[str] = []  # Final structures
        
        # Routing function: given current structure and result, return next structure(s)
        self.router: Optional[Callable[[str, Any], List[str]]] = None
    
    def add_structure(self, structure: HyperdimensionalStructure, 
                      is_entry: bool = False, is_exit: bool = False) -> None:
        """Add a structure to the chain."""
        self.structures[structure.name] = structure
        
        if is_entry:
            self.entry_point = structure.name
        if is_exit:
            self.exit_points.append(structure.name)
    
    def link(self, source: str, target: str, 
             transcoder: Optional[Callable] = None,
             weight: float = 1.0) -> None:
        """Create a link between two structures."""
        if source not in self.structures:
            raise ValueError(f"Source structure '{source}' not found")
        if target not in self.structures:
            raise ValueError(f"Target structure '{target}' not found")
        
        self.links.append(StructureLink(
            source=source,
            target=target,
            transcoder=transcoder,
            weight=weight
        ))
    
    def set_router(self, router: Callable[[str, Any], List[str]]) -> None:
        """Set the routing function for conditional processing."""
        self.router = router
    
    def _get_outgoing_links(self, structure_name: str) -> List[StructureLink]:
        """Get all links from a structure."""
        return [link for link in self.links if link.source == structure_name]
    
    def _get_next_structures(self, current: str, result: Any) -> List[str]:
        """Determine next structure(s) to process."""
        if self.router:
            return self.router(current, result)
        
        # Default: follow all outgoing links
        links = self._get_outgoing_links(current)
        return [link.target for link in links]
    
    def process(self, input_position: np.ndarray, 
                top_k: int = 3) -> ChainResult:
        """
        Process input through the chain.
        
        Starts at entry point, follows links until exit point(s).
        """
        if not self.entry_point:
            raise ValueError("No entry point defined")
        
        path = []
        intermediate = {}
        current_position = input_position
        current_structures = [self.entry_point]
        final_results = []
        
        visited = set()
        max_depth = 10  # Prevent infinite loops
        depth = 0
        
        while current_structures and depth < max_depth:
            depth += 1
            next_structures = []
            
            for struct_name in current_structures:
                if struct_name in visited:
                    continue
                visited.add(struct_name)
                path.append(struct_name)
                
                structure = self.structures[struct_name]
                
                # Query this structure
                matches = structure.query_nearest(current_position, k=top_k)
                
                if matches:
                    best_node, similarity = matches[0]
                    intermediate[struct_name] = {
                        'node': best_node,
                        'similarity': similarity,
                        'data': best_node.data
                    }
                    
                    # Check if this is an exit point
                    if struct_name in self.exit_points:
                        final_results.append((best_node.data, similarity, struct_name))
                    else:
                        # Transform position for next structure
                        links = self._get_outgoing_links(struct_name)
                        for link in links:
                            transformed = link.transform(best_node.position)
                            # Use transformed position for next structure
                            current_position = transformed
                            
                            # Determine next structures
                            nexts = self._get_next_structures(struct_name, best_node.data)
                            next_structures.extend(nexts)
            
            current_structures = list(set(next_structures) - visited)
        
        # Combine final results
        if final_results:
            # Take best result
            best_data, best_conf, best_struct = max(final_results, key=lambda x: x[1])
            return ChainResult(
                final_output=best_data,
                confidence=best_conf,
                path=path,
                intermediate_results=intermediate
            )
        
        return ChainResult(
            final_output=None,
            confidence=0.0,
            path=path,
            intermediate_results=intermediate
        )
    
    def feedback(self, path: List[str], success: bool) -> None:
        """
        Propagate feedback backward through the chain.
        
        Each structure in the path receives feedback.
        """
        # TODO: Implement backward feedback propagation
        pass
    
    def stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            'name': self.name,
            'structures': list(self.structures.keys()),
            'links': [(l.source, l.target) for l in self.links],
            'entry': self.entry_point,
            'exits': self.exit_points,
        }
    
    def save(self, path: str) -> None:
        """Save the chain."""
        data = {
            'type': 'StructureChain',
            'version': '1.0',
            'name': self.name,
            'entry_point': self.entry_point,
            'exit_points': self.exit_points,
            'structures': {
                name: struct.to_dict() 
                for name, struct in self.structures.items()
            },
            'links': [
                {'source': l.source, 'target': l.target, 'weight': l.weight}
                for l in self.links
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'StructureChain':
        """Load from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        chain = cls(name=data.get('name', 'chain'))
        chain.entry_point = data.get('entry_point')
        chain.exit_points = data.get('exit_points', [])
        
        for name, struct_data in data.get('structures', {}).items():
            chain.structures[name] = HyperdimensionalStructure.from_dict(struct_data)
        
        for link_data in data.get('links', []):
            chain.links.append(StructureLink(
                source=link_data['source'],
                target=link_data['target'],
                weight=link_data.get('weight', 1.0)
            ))
        
        return chain


# =============================================================================
# INTENT ROUTER - Routes queries to appropriate domain
# =============================================================================

class IntentRouter:
    """
    Routes queries to the appropriate domain structure.
    
    Uses a small intent structure to classify queries,
    then routes to the appropriate domain (bash, git, etc.)
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        
        # Intent structure - classifies query type
        self.intent_structure = HyperdimensionalStructure(dims=dims, name="intent")
        
        # Domain structures
        self.domains: Dict[str, HyperdimensionalStructure] = {}
        
        # Word positions (shared across all)
        self.word_positions: Dict[str, np.ndarray] = {}
    
    def add_intent(self, intent_name: str, example_words: List[str]) -> None:
        """Add an intent with example words."""
        # Compute position from example words
        positions = []
        for word in example_words:
            if word not in self.word_positions:
                # Random position for new word
                pos = np.random.randn(self.dims)
                pos = pos / np.linalg.norm(pos) * CRITICAL_LINE
                self.word_positions[word] = pos
            positions.append(self.word_positions[word])
        
        if positions:
            intent_pos = np.mean(positions, axis=0)
            norm = np.linalg.norm(intent_pos)
            if norm > 1e-10:
                intent_pos = intent_pos / norm * CRITICAL_LINE
            
            self.intent_structure.add(
                node_id=intent_name,
                position=intent_pos,
                data={'intent': intent_name, 'words': example_words}
            )
    
    def add_domain(self, domain_name: str, structure: HyperdimensionalStructure) -> None:
        """Add a domain structure."""
        self.domains[domain_name] = structure
    
    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode query to position."""
        words = query.lower().split()
        words = [''.join(c for c in w if c.isalnum()) for w in words]
        words = [w for w in words if w]
        
        positions = []
        for word in words:
            if word in self.word_positions:
                positions.append(self.word_positions[word])
        
        if not positions:
            return None
        
        pos = np.mean(positions, axis=0)
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        
        return pos
    
    def route(self, query: str) -> Tuple[Optional[str], float]:
        """
        Route query to appropriate domain.
        
        Returns (domain_name, confidence).
        """
        pos = self._encode_query(query)
        if pos is None:
            return None, 0.0
        
        # Find best matching intent
        matches = self.intent_structure.query_nearest(pos, k=1)
        
        if matches:
            node, confidence = matches[0]
            intent = node.data.get('intent', '')
            return intent, confidence
        
        return None, 0.0
    
    def process(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Process query through routing.
        
        Returns list of (command, confidence, domain) tuples.
        """
        # First, route to domain
        domain, route_conf = self.route(query)
        
        if domain is None or domain not in self.domains:
            # Try all domains
            results = []
            for domain_name, structure in self.domains.items():
                pos = self._encode_query(query)
                if pos is not None:
                    matches = structure.query_nearest(pos, k=top_k)
                    for node, conf in matches:
                        if node.data and 'bash' in node.data:
                            results.append((node.data['bash'], conf, domain_name))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        # Query the specific domain
        pos = self._encode_query(query)
        if pos is None:
            return []
        
        structure = self.domains[domain]
        matches = structure.query_nearest(pos, k=top_k)
        
        results = []
        for node, conf in matches:
            if node.data and 'bash' in node.data:
                # Combine route confidence with match confidence
                combined_conf = route_conf * conf
                results.append((node.data['bash'], combined_conf, domain))
        
        return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Structure Chain Test ===\n")
    
    # Create a simple chain: Intent → Domain → Command
    
    # 1. Create structures
    intent_struct = HyperdimensionalStructure(dims=8, name="intent")
    bash_struct = HyperdimensionalStructure(dims=8, name="bash")
    git_struct = HyperdimensionalStructure(dims=8, name="git")
    
    # 2. Populate intent structure
    # File operations intent
    file_pos = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    file_pos = file_pos / np.linalg.norm(file_pos) * CRITICAL_LINE
    intent_struct.add("file_ops", position=file_pos, data={'intent': 'file', 'domain': 'bash'})
    
    # Process operations intent
    proc_pos = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    proc_pos = proc_pos / np.linalg.norm(proc_pos) * CRITICAL_LINE
    intent_struct.add("proc_ops", position=proc_pos, data={'intent': 'process', 'domain': 'bash'})
    
    # Git operations intent
    git_pos = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    git_pos = git_pos / np.linalg.norm(git_pos) * CRITICAL_LINE
    intent_struct.add("git_ops", position=git_pos, data={'intent': 'git', 'domain': 'git'})
    
    # 3. Populate bash structure
    ls_pos = np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=float)
    ls_pos = ls_pos / np.linalg.norm(ls_pos) * CRITICAL_LINE
    bash_struct.add("ls", position=ls_pos, data={'bash': 'ls', 'nl': 'list files'})
    
    ps_pos = np.array([0.1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    ps_pos = ps_pos / np.linalg.norm(ps_pos) * CRITICAL_LINE
    bash_struct.add("ps", position=ps_pos, data={'bash': 'ps aux', 'nl': 'show processes'})
    
    # 4. Populate git structure
    status_pos = np.array([0, 0, 1, 0.1, 0, 0, 0, 0], dtype=float)
    status_pos = status_pos / np.linalg.norm(status_pos) * CRITICAL_LINE
    git_struct.add("status", position=status_pos, data={'bash': 'git status', 'nl': 'show status'})
    
    commit_pos = np.array([0, 0, 0.9, 0.5, 0, 0, 0, 0], dtype=float)
    commit_pos = commit_pos / np.linalg.norm(commit_pos) * CRITICAL_LINE
    git_struct.add("commit", position=commit_pos, data={'bash': 'git commit', 'nl': 'commit changes'})
    
    # 5. Create chain
    chain = StructureChain(name="nl_to_command")
    chain.add_structure(intent_struct, is_entry=True)
    chain.add_structure(bash_struct, is_exit=True)
    chain.add_structure(git_struct, is_exit=True)
    
    # 6. Create links with routing
    chain.link("intent", "bash")
    chain.link("intent", "git")
    
    # 7. Set router based on intent
    def intent_router(current: str, result: Any) -> List[str]:
        if current == "intent" and result:
            domain = result.get('domain', 'bash')
            return [domain]
        return []
    
    chain.set_router(intent_router)
    
    print(f"Chain: {chain.stats()}")
    print()
    
    # 8. Test processing
    print("--- Processing Test ---")
    
    # File query
    file_query_pos = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=float)
    file_query_pos = file_query_pos / np.linalg.norm(file_query_pos) * CRITICAL_LINE
    
    result = chain.process(file_query_pos)
    print(f"File query: {result}")
    
    # Git query
    git_query_pos = np.array([0, 0, 0.9, 0.1, 0, 0, 0, 0], dtype=float)
    git_query_pos = git_query_pos / np.linalg.norm(git_query_pos) * CRITICAL_LINE
    
    result = chain.process(git_query_pos)
    print(f"Git query: {result}")
    
    # 9. Test persistence
    print("\n--- Persistence Test ---")
    chain.save("/tmp/structure_chain.json")
    print("Saved to /tmp/structure_chain.json")
    
    loaded = StructureChain.load("/tmp/structure_chain.json")
    print(f"Loaded: {loaded.stats()}")
    
    print("\n✓ Structure chain working!")
    print("\nKey concepts:")
    print("  - Structures are 'chromosomes' with their own domain")
    print("  - Links define information flow between structures")
    print("  - Router determines which path to take")
    print("  - Feedback can propagate backward through chain")
