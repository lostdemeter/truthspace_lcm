"""
Multi-Domain Translator with Chromosome-like Corpus Sharing

Inspired by DNA mechanics (Design 077):
- Multiple corpuses (chromosomes) can share information
- Each corpus is specialized but can cross-reference
- Shared words create "bridges" between domains

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Query: "show files"                                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ Bash Corpus │ ←─→ │ Shared Words│ ←─→ │ Git Corpus  │           │
│  │ (Chromosome)│     │  (Bridges)  │     │ (Chromosome)│           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│       │                                         │                   │
│       ▼                                         ▼                   │
│      "ls"                                  "git status"             │
│                                                                     │
│  Disambiguation via context or domain hint                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

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


def word_overlap(words1: Set[str], words2: Set[str]) -> float:
    """Jaccard similarity between word sets."""
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


@dataclass
class DomainCorpus:
    """A single domain corpus (chromosome)."""
    name: str
    structure: HyperdimensionalStructure
    mappings: List[dict] = field(default_factory=list)
    word_index: Dict[str, Set[str]] = field(default_factory=dict)  # word → node_ids
    
    def get_words(self) -> Set[str]:
        """Get all words in this corpus."""
        all_words = set()
        for mapping in self.mappings:
            all_words.update(extract_words(mapping['nl']))
        return all_words


class MultiDomainTranslator:
    """
    Multi-domain translator with chromosome-like corpus sharing.
    
    Each domain is a separate corpus (chromosome).
    Shared words create bridges between domains.
    Context or domain hints disambiguate.
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        self.domains: Dict[str, DomainCorpus] = {}
        self.shared_words: Dict[str, Set[str]] = {}  # word → set of domain names
    
    def add_domain(self, name: str, bootstrap_path: str) -> int:
        """
        Add a domain from a bootstrap JSON file.
        
        Returns number of mappings loaded.
        """
        with open(bootstrap_path, 'r') as f:
            data = json.load(f)
        
        structure = HyperdimensionalStructure(dims=self.dims, name=name)
        corpus = DomainCorpus(name=name, structure=structure)
        
        # Load mappings
        mappings = data.get('mappings', [])
        n = len(mappings)
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            words_i = extract_words(mappings[i]['nl'])
            for j in range(n):
                words_j = extract_words(mappings[j]['nl'])
                S[i, j] = word_overlap(words_i, words_j)
        
        # Holographic projection
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        valid_eigenvalues = np.maximum(eigenvalues[idx], 0)
        positions = eigenvectors[:, idx] * np.sqrt(valid_eigenvalues)
        
        # Add to structure
        for i, mapping in enumerate(mappings):
            words = extract_words(mapping['nl'])
            
            pos = positions[i]
            norm = np.linalg.norm(pos)
            if norm > 1e-10:
                pos = pos / norm * CRITICAL_LINE
            
            node_id = f"{name}_{i}"
            structure.add(
                node_id=node_id,
                position=pos,
                data={
                    'nl': mapping['nl'],
                    'bash': mapping['bash'],
                    'words': list(words),
                    'domain': name
                }
            )
            
            # Update word index
            for word in words:
                if word not in corpus.word_index:
                    corpus.word_index[word] = set()
                corpus.word_index[word].add(node_id)
            
            corpus.mappings.append(mapping)
        
        self.domains[name] = corpus
        
        # Update shared words
        self._update_shared_words()
        
        return len(mappings)
    
    def _update_shared_words(self) -> None:
        """Update the shared words index."""
        self.shared_words = {}
        
        for domain_name, corpus in self.domains.items():
            for word in corpus.get_words():
                if word not in self.shared_words:
                    self.shared_words[word] = set()
                self.shared_words[word].add(domain_name)
    
    def get_bridges(self) -> Dict[str, Set[str]]:
        """
        Get words that bridge multiple domains.
        
        These are like shared genes between chromosomes.
        """
        return {word: domains for word, domains in self.shared_words.items()
                if len(domains) > 1}
    
    def translate(self, query: str, domain: Optional[str] = None, 
                  top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Translate query to command(s).
        
        If domain is specified, only search that domain.
        Otherwise, search all domains and return best matches.
        
        Returns list of (command, confidence, domain) tuples.
        """
        query_words = extract_words(query)
        
        if not query_words:
            return []
        
        results = []
        
        # Determine which domains to search
        if domain and domain in self.domains:
            domains_to_search = [self.domains[domain]]
        else:
            domains_to_search = list(self.domains.values())
        
        # Search each domain
        for corpus in domains_to_search:
            for node in corpus.structure:
                if node.data and 'words' in node.data:
                    mapping_words = set(node.data['words'])
                    similarity = word_overlap(query_words, mapping_words)
                    if similarity > 0:
                        results.append((
                            node.data['bash'],
                            similarity,
                            corpus.name,
                            node
                        ))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k unique commands (per domain)
        seen = set()
        unique_results = []
        for bash, sim, domain_name, node in results:
            key = (bash, domain_name)
            if key not in seen:
                seen.add(key)
                unique_results.append((bash, sim, domain_name))
                if len(unique_results) >= top_k * len(self.domains):
                    break
        
        return unique_results[:top_k * len(self.domains)]
    
    def translate_with_context(self, query: str, context: List[str],
                                top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Translate with context from previous queries/commands.
        
        Context helps disambiguate between domains.
        """
        # Determine domain affinity from context
        domain_scores = {name: 0.0 for name in self.domains}
        
        for ctx in context:
            ctx_words = extract_words(ctx)
            for word in ctx_words:
                if word in self.shared_words:
                    for domain_name in self.shared_words[word]:
                        domain_scores[domain_name] += 1
        
        # Get base results
        results = self.translate(query, top_k=top_k * 2)
        
        # Boost by domain affinity
        boosted = []
        for bash, sim, domain_name in results:
            boost = domain_scores.get(domain_name, 0) * 0.1
            boosted.append((bash, sim + boost, domain_name))
        
        # Re-sort
        boosted.sort(key=lambda x: x[1], reverse=True)
        
        return boosted[:top_k]
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        bridges = self.get_bridges()
        return {
            'domains': list(self.domains.keys()),
            'mappings_per_domain': {name: len(c.mappings) for name, c in self.domains.items()},
            'total_mappings': sum(len(c.mappings) for c in self.domains.values()),
            'shared_words': len(bridges),
            'bridge_words': list(bridges.keys())[:20],  # First 20
        }
    
    def save(self, path: str) -> None:
        """Save all domains."""
        data = {
            'type': 'MultiDomainTranslator',
            'version': '1.0',
            'dims': self.dims,
            'domains': {}
        }
        
        for name, corpus in self.domains.items():
            data['domains'][name] = {
                'structure': corpus.structure.to_dict(),
                'mappings': corpus.mappings,
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MultiDomainTranslator':
        """Load from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        translator = cls(dims=data.get('dims', 12))
        
        for name, domain_data in data.get('domains', {}).items():
            structure = HyperdimensionalStructure.from_dict(domain_data['structure'])
            corpus = DomainCorpus(
                name=name,
                structure=structure,
                mappings=domain_data['mappings']
            )
            
            # Rebuild word index
            for mapping in corpus.mappings:
                words = extract_words(mapping['nl'])
                for word in words:
                    if word not in corpus.word_index:
                        corpus.word_index[word] = set()
                    # We don't have node_id here, but that's okay for basic functionality
            
            translator.domains[name] = corpus
        
        translator._update_shared_words()
        return translator


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Multi-Domain Translator ===")
    print("Testing chromosome-like corpus sharing between Bash and Git domains.")
    print()
    
    # Create translator and add domains
    translator = MultiDomainTranslator(dims=12)
    
    bootstrap_dir = Path(__file__).parent / "bootstrap"
    
    bash_count = translator.add_domain("bash", str(bootstrap_dir / "nl_bash_mappings.json"))
    git_count = translator.add_domain("git", str(bootstrap_dir / "git_mappings.json"))
    
    print(f"Loaded {bash_count} Bash mappings")
    print(f"Loaded {git_count} Git mappings")
    print()
    
    # Show shared words (bridges)
    stats = translator.stats()
    print(f"--- Shared Words (Bridges) ---")
    print(f"Total bridge words: {stats['shared_words']}")
    print(f"Examples: {', '.join(stats['bridge_words'][:15])}")
    print()
    
    # Test queries that could go either way
    ambiguous_queries = [
        "show status",      # git status vs system status?
        "show files",       # ls vs git status?
        "add files",        # git add vs ???
        "show changes",     # git diff vs ???
        "show log",         # git log vs system log?
        "create new",       # mkdir vs git init?
        "delete file",      # rm (bash only)
        "push changes",     # git push (git only)
        "show memory",      # free -h (bash only)
        "commit changes",   # git commit (git only)
    ]
    
    print("--- Ambiguous Queries (No Domain Hint) ---")
    for query in ambiguous_queries:
        results = translator.translate(query, top_k=4)
        if results:
            formatted = [f"{cmd}[{dom}]({conf:.2f})" for cmd, conf, dom in results[:4]]
            print(f"  '{query}'")
            print(f"      → {' | '.join(formatted)}")
        else:
            print(f"  '{query}' → NO MATCH")
    print()
    
    # Test with domain hints
    print("--- Queries with Domain Hints ---")
    domain_queries = [
        ("show status", "bash"),
        ("show status", "git"),
        ("show files", "bash"),
        ("add files", "git"),
        ("create new", "bash"),
        ("create new", "git"),
    ]
    
    for query, domain in domain_queries:
        results = translator.translate(query, domain=domain, top_k=2)
        if results:
            formatted = [f"{cmd}({conf:.2f})" for cmd, conf, dom in results[:2]]
            print(f"  '{query}' [domain={domain}] → {' | '.join(formatted)}")
        else:
            print(f"  '{query}' [domain={domain}] → NO MATCH")
    print()
    
    # Test with context
    print("--- Queries with Context ---")
    context_queries = [
        ("show status", ["git clone", "git pull"]),  # Git context
        ("show status", ["ls", "cd", "pwd"]),        # Bash context
        ("show files", ["git add", "git commit"]),   # Git context
        ("show files", ["rm", "mkdir"]),             # Bash context
    ]
    
    for query, context in context_queries:
        results = translator.translate_with_context(query, context, top_k=2)
        if results:
            formatted = [f"{cmd}[{dom}]({conf:.2f})" for cmd, conf, dom in results[:2]]
            ctx_str = ", ".join(context[:2])
            print(f"  '{query}' (context: {ctx_str})")
            print(f"      → {' | '.join(formatted)}")
        else:
            print(f"  '{query}' → NO MATCH")
    print()
    
    # Test persistence
    print("--- Testing Persistence ---")
    translator.save("/tmp/multi_domain.json")
    print("Saved to /tmp/multi_domain.json")
    
    loaded = MultiDomainTranslator.load("/tmp/multi_domain.json")
    print(f"Loaded: {loaded.stats()['total_mappings']} total mappings")
    
    results = loaded.translate("show files", top_k=2)
    if results:
        print(f"  'show files' → {results[0][0]}[{results[0][2]}] ({results[0][1]:.2f})")
    
    print("\n✓ Multi-domain translator complete!")
    print("\nKey insights:")
    print("  - Multiple domains (chromosomes) can coexist")
    print("  - Shared words create bridges between domains")
    print("  - Domain hints or context disambiguate")
    print("  - Each domain maintains its own geometric structure")
