"""
TruthSpace: Geometric Knowledge Storage and Resolution

A unified knowledge system that uses φ-MAX encoding for storage
and geometric distance for queries. This replaces keyword-based
matching with pure hypergeometric resolution.

Design principles:
- Pure geometry: All resolution is φ-weighted Euclidean distance
- φ-MAX encoding: Synonyms collapse, levels separate
- No fallbacks: Query succeeds or fails
- Fail fast: Errors are explicit, not hidden
- Minimal: Only what's needed for geometric resolution

This is a showcase of how hypergeometry can replace trained LLM/LCM
functionality with mathematically-derived resolution.
"""

import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
DIM = 12  # Encoding dimension

# φ-block weights: Actions (φ²), Domains (1), Relations (φ⁻²)
PHI_BLOCK_WEIGHTS = np.array([
    PHI**2, PHI**2, PHI**2, PHI**2,  # Actions: dims 0-3
    1.0, 1.0, 1.0, 1.0,               # Domains: dims 4-7
    PHI**-2, PHI**-2, PHI**-2, PHI**-2  # Relations: dims 8-11
])


# =============================================================================
# PRIMITIVES
# =============================================================================

@dataclass
class Primitive:
    """A semantic anchor in truth space."""
    name: str
    dimension: int
    level: int
    keywords: List[str]


# Bootstrap primitives - the semantic anchors
PRIMITIVES = [
    # Actions (dims 0-3)
    Primitive("CREATE", 0, 0, ["create", "make", "new", "generate", "add", "init"]),
    Primitive("READ", 1, 0, ["read", "get", "show", "display", "view", "print", "cat"]),
    Primitive("LIST", 1, 1, ["list", "ls", "dir"]),
    Primitive("WRITE", 1, 1, ["write", "set", "update", "modify", "change", "edit"]),
    Primitive("DELETE", 1, 2, ["delete", "remove", "destroy", "kill", "rm", "erase", "clear"]),
    Primitive("COPY", 2, 0, ["copy", "duplicate", "clone", "cp"]),
    Primitive("RELOCATE", 2, 1, ["move", "mv", "relocate", "rename"]),
    Primitive("SEARCH", 2, 2, ["search", "find", "locate", "grep", "lookup", "query"]),
    Primitive("EXECUTE", 3, 0, ["run", "execute", "start", "launch", "invoke"]),
    Primitive("CONNECT", 3, 1, ["connect", "link", "join", "attach", "download", "fetch", "curl"]),
    Primitive("TRANSFORM", 3, 2, ["transform", "convert", "parse", "format"]),
    Primitive("COMPRESS", 3, 3, ["compress", "zip", "tar", "archive", "pack"]),
    Primitive("SORT", 3, 4, ["sort", "order", "arrange", "rank"]),
    Primitive("FILTER", 3, 5, ["filter", "unique", "uniq", "distinct", "dedupe"]),
    Primitive("COUNT", 3, 6, ["count", "wc", "tally", "sum"]),
    Primitive("TRACE", 3, 7, ["trace", "strace", "debug", "monitor"]),
    
    # Domains (dims 4-7)
    Primitive("PROCESS", 4, 0, ["process", "processes", "pid", "proc", "task", "job", "running"]),
    Primitive("NETWORK", 4, 1, ["network", "net", "http", "url", "web", "remote", "ssh"]),
    Primitive("USER", 4, 2, ["user", "owner", "permission", "chmod", "whoami"]),
    Primitive("FILE", 5, 0, ["file", "files", "document"]),
    Primitive("DIRECTORY", 5, 2, ["directory", "folder", "dir", "path", "pwd"]),
    Primitive("SYSTEM", 5, 1, ["system", "os", "uname", "kernel"]),
    Primitive("STORAGE", 5, 3, ["storage", "disk", "space", "volume", "df", "size"]),
    Primitive("MEMORY", 5, 4, ["memory", "ram", "free", "usage"]),
    Primitive("TIME", 5, 5, ["time", "date", "clock", "timestamp"]),
    Primitive("HOST", 5, 6, ["host", "hostname", "machine", "computer"]),
    Primitive("UPTIME", 5, 7, ["uptime", "since", "boot"]),
    Primitive("DATA", 6, 1, ["data", "text", "content", "string", "output", "log"]),
    Primitive("RECURSIVE", 7, 0, ["recursive", "tree", "all", "deep"]),
    
    # Relations (dims 8-11)
    Primitive("INTO", 8, 0, ["into", "to", "toward"]),
    Primitive("FROM", 8, 1, ["from", "source", "origin"]),
    Primitive("BEFORE", 9, 0, ["before", "first", "head", "top", "start"]),
    Primitive("AFTER", 9, 1, ["after", "last", "tail", "end", "bottom"]),
    Primitive("DURING", 10, 0, ["during", "while", "follow", "watch", "live"]),
    Primitive("TEST", 11, 0, ["test", "check", "verify", "ping"]),
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class KnowledgeGapError(Exception):
    """Raised when no geometric match is found."""
    def __init__(self, query: str, best_similarity: float = 0.0):
        self.query = query
        self.best_similarity = best_similarity
        super().__init__(f"No match for '{query}' (best: {best_similarity:.2f})")


# =============================================================================
# KNOWLEDGE ENTRY
# =============================================================================

@dataclass
class KnowledgeEntry:
    """A piece of knowledge with its geometric position."""
    name: str
    description: str
    position: np.ndarray
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "position": self.position.tolist(),
            "output": self.output,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            name=data["name"],
            description=data["description"],
            position=np.array(data["position"]),
            output=data.get("output", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# TRUTHSPACE
# =============================================================================

class TruthSpace:
    """
    Geometric knowledge storage and resolution.
    
    Uses φ-MAX encoding for positions and φ-weighted Euclidean
    distance for queries. No keywords, no database - pure geometry.
    """
    
    def __init__(self, knowledge_file: str = None):
        self.dim = DIM
        self.entries: List[KnowledgeEntry] = []
        
        # Build keyword → primitive mapping
        self.keyword_to_primitive: Dict[str, Primitive] = {}
        for prim in PRIMITIVES:
            for kw in prim.keywords:
                self.keyword_to_primitive[kw.lower()] = prim
        
        # Load or initialize knowledge
        if knowledge_file and Path(knowledge_file).exists():
            self.load(knowledge_file)
        else:
            self._load_default_knowledge()
    
    def _encode(self, text: str) -> np.ndarray:
        """
        Encode text using φ-MAX.
        
        - φ^level for each primitive
        - φ^(-i/2) for word position decay
        - MAX per dimension (Sierpinski property)
        """
        words = self._tokenize(text)
        position = np.zeros(self.dim)
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.keyword_to_primitive:
                prim = self.keyword_to_primitive[word_lower]
                value = PHI ** prim.level
                pos_decay = PHI ** (-i * 0.5)
                position[prim.dimension] = max(position[prim.dimension], value * pos_decay)
        
        return position
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """φ-weighted Euclidean distance."""
        diff = (a - b) * PHI_BLOCK_WEIGHTS
        return float(np.linalg.norm(diff))
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Convert distance to similarity (0 to 1)."""
        dist = self._distance(a, b)
        return 1.0 / (1.0 + dist)
    
    def store(self, name: str, description: str, output: str = None, metadata: Dict = None):
        """
        Store knowledge with geometric encoding.
        
        Args:
            name: Command or concept name
            description: Natural language description (encoded to position)
            output: Executable output (defaults to name)
            metadata: Additional data
        """
        position = self._encode(description)
        entry = KnowledgeEntry(
            name=name,
            description=description,
            position=position,
            output=output or name,
            metadata=metadata or {},
        )
        self.entries.append(entry)
    
    def query(self, text: str, threshold: float = 0.3) -> Tuple[KnowledgeEntry, float]:
        """
        Find the nearest knowledge entry to the query.
        
        Args:
            text: Natural language query
            threshold: Minimum similarity required
        
        Returns:
            (entry, similarity)
        
        Raises:
            KnowledgeGapError if no match above threshold
        """
        if not self.entries:
            raise KnowledgeGapError(text, 0.0)
        
        query_pos = self._encode(text)
        
        best_entry = None
        best_sim = 0.0
        
        for entry in self.entries:
            sim = self._similarity(query_pos, entry.position)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        
        if best_sim < threshold:
            raise KnowledgeGapError(text, best_sim)
        
        return best_entry, best_sim
    
    def resolve(self, text: str) -> Tuple[str, KnowledgeEntry, float]:
        """
        Resolve query to executable output.
        
        Returns:
            (output, entry, similarity)
        """
        entry, similarity = self.query(text)
        return entry.output, entry, similarity
    
    def explain(self, text: str) -> str:
        """Explain how a query would be resolved."""
        query_pos = self._encode(text)
        
        lines = [
            f"Query: \"{text}\"",
            "",
            "φ-MAX Encoding:",
        ]
        
        # Show active dimensions
        for i, v in enumerate(query_pos):
            if v > 0.01:
                prim_name = "?"
                for prim in PRIMITIVES:
                    if prim.dimension == i:
                        prim_name = prim.name
                        break
                lines.append(f"  dim{i}: {v:.3f} ({prim_name})")
        
        # Show top matches
        lines.append("")
        lines.append("Top 3 matches:")
        
        matches = [(e, self._similarity(query_pos, e.position)) for e in self.entries]
        matches.sort(key=lambda x: x[1], reverse=True)
        
        for entry, sim in matches[:3]:
            lines.append(f"  {entry.name}: {sim:.3f} (\"{entry.description}\")")
        
        return "\n".join(lines)
    
    def save(self, filepath: str):
        """Save knowledge to JSON."""
        data = [e.to_dict() for e in self.entries]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load knowledge from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.entries = [KnowledgeEntry.from_dict(d) for d in data]
    
    def _load_default_knowledge(self):
        """Load default bash command knowledge."""
        defaults = [
            # Directory
            ("pwd", "directory"),
            ("ls", "list directory"),
            ("ls", "list file"),
            ("mkdir -p", "create directory"),
            
            # File
            ("touch", "create file"),
            ("cat", "read file"),
            ("head", "read file first"),
            ("tail", "read file last"),
            ("rm", "delete file"),
            ("rm -r", "delete directory"),
            
            # Copy/Move
            ("cp", "copy file"),
            ("cp -r", "copy directory"),
            ("mv", "move file"),
            
            # Search
            ("find", "find file"),
            ("grep", "search data"),
            
            # System
            ("df", "read storage"),
            ("du", "read directory storage"),
            ("uname", "read system"),
            ("hostname", "read host"),
            ("date", "read time"),
            ("uptime", "read uptime"),
            ("free", "read memory"),
            
            # Process
            ("ps", "list process"),
            ("top", "read process"),
            ("kill", "delete process"),
            ("lsof", "list file process"),
            ("pstree", "read process recursive"),
            ("strace", "trace system"),
            
            # Network
            ("ifconfig", "read network"),
            ("curl", "download network"),
            ("wget", "download network file"),
            ("ssh", "connect network user"),
            ("scp", "copy file network"),
            ("ping", "test network"),
            
            # User
            ("whoami", "read user"),
            ("chmod", "write file user"),
            
            # Data
            ("wc", "count file"),
            ("sort", "sort data"),
            ("uniq", "filter unique"),
            ("tar", "compress file"),
            
            # Environment
            ("env", "read data"),
            ("echo", "write output"),
        ]
        
        for cmd, desc in defaults:
            self.store(cmd, desc)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    """Demonstrate geometric resolution."""
    ts = TruthSpace()
    
    print("=" * 60)
    print("TRUTHSPACE: Geometric Knowledge Resolution")
    print("=" * 60)
    
    queries = [
        "list directory contents",
        "show disk space",
        "copy files",
        "move files",
        "find files",
        "search text in files",
        "show running processes",
        "kill process",
        "compress files",
        "download from url",
    ]
    
    print("\nResolution Test:")
    print("-" * 40)
    
    for query in queries:
        try:
            output, entry, sim = ts.resolve(query)
            print(f"✓ \"{query}\" → {output} ({sim:.2f})")
        except KnowledgeGapError as e:
            print(f"✗ \"{query}\" → no match ({e.best_similarity:.2f})")
    
    print("\n" + "-" * 40)
    print("\nExplanation:")
    print(ts.explain("show disk space"))
    
    print("\n" + "=" * 60)
    print("Pure geometry. No keywords. No training.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate()
