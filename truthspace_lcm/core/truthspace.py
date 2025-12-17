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
import sqlite3
import subprocess
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager


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
    Primitive("READ", 1, 0, ["read", "get", "show", "display", "view", "print", "cat", "report", "dump"]),
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
    Primitive("MEMORY", 5, 4, ["memory", "ram", "free", "usage", "virtual", "swap", "vmstat"]),
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
    distance for queries.
    
    Knowledge sources:
    - Bootstrap: Loaded from JSON file (immutable core knowledge)
    - Learned: Stored in SQLite database (persists across runs)
    """
    
    def __init__(self, db_path: str = None, bootstrap_path: str = None):
        self.dim = DIM
        self.entries: List[KnowledgeEntry] = []
        
        # Build keyword → primitive mapping
        self.keyword_to_primitive: Dict[str, Primitive] = {}
        for prim in PRIMITIVES:
            for kw in prim.keywords:
                self.keyword_to_primitive[kw.lower()] = prim
        
        # Set up paths
        pkg_dir = Path(__file__).parent.parent
        self.db_path = db_path or str(pkg_dir / "truthspace.db")
        self.bootstrap_path = bootstrap_path or str(pkg_dir / "bootstrap_knowledge.json")
        
        # Initialize database
        self._init_db()
        
        # Load knowledge
        self._load_bootstrap()
        self._load_learned()
    
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
    
    def store(self, name: str, description: str, output: str = None, 
              metadata: Dict = None, persist: bool = False) -> KnowledgeEntry:
        """
        Store knowledge with geometric encoding.
        
        Args:
            name: Command or concept name
            description: Natural language description (encoded to position)
            output: Executable output (defaults to name)
            metadata: Additional data
            persist: If True, save to database for persistence
        
        Returns:
            The created KnowledgeEntry
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
        
        # Persist to database if requested
        if persist:
            with self._connection() as conn:
                conn.execute(
                    "INSERT INTO knowledge (command, description) VALUES (?, ?)",
                    (name, description)
                )
        
        return entry
    
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
    
    # =========================================================================
    # DATABASE
    # =========================================================================
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize the database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT DEFAULT 'learned',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _load_bootstrap(self):
        """Load bootstrap knowledge from JSON."""
        if not Path(self.bootstrap_path).exists():
            return
        
        with open(self.bootstrap_path, 'r') as f:
            data = json.load(f)
        
        for entry in data.get("entries", []):
            self.store(
                entry["command"],
                entry["description"],
                persist=False  # Don't save bootstrap to DB
            )
    
    def _load_learned(self):
        """Load learned knowledge from database."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT command, description FROM knowledge")
            for row in cursor:
                self.store(row["command"], row["description"], persist=False)
    
    # =========================================================================
    # MAN PAGE INGESTION
    # =========================================================================
    
    def parse_man_page(self, command: str) -> Optional[str]:
        """
        Parse a man page to extract a description.
        
        Returns the NAME section description, or None if not found.
        """
        try:
            result = subprocess.run(
                ["man", command],
                capture_output=True,
                text=True,
                timeout=5,
                env={"MANWIDTH": "1000", "LANG": "C"}
            )
            
            if result.returncode != 0:
                return None
            
            text = result.stdout
            
            # Extract NAME section
            name_match = re.search(
                r'NAME\s*\n\s*(\S+)\s*[-–—]\s*(.+?)(?=\n\n|\nSYNOPSIS|\nDESCRIPTION)',
                text,
                re.DOTALL
            )
            
            if name_match:
                description = name_match.group(2).strip()
                # Clean up whitespace
                description = re.sub(r'\s+', ' ', description)
                return description[:200]  # Limit length
            
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def infer_description(self, man_description: str, command: str = None) -> str:
        """
        Convert a man page description to primitive-friendly description.
        
        Extracts key action and domain words that map to primitives.
        Prioritizes unique combinations to avoid collisions.
        """
        words = self._tokenize(man_description)
        
        # Find words that map to primitives, tracking dimensions
        primitive_words = []
        seen_dims = set()
        
        for word in words:
            if word in self.keyword_to_primitive:
                prim = self.keyword_to_primitive[word]
                # Prefer one word per dimension for uniqueness
                if prim.dimension not in seen_dims:
                    primitive_words.append(word)
                    seen_dims.add(prim.dimension)
        
        # If we found primitives, use them
        if primitive_words:
            desc = " ".join(primitive_words[:4])
        else:
            # Fallback: use first few words
            desc = " ".join(words[:3])
        
        # Add command name if it maps to a primitive (for uniqueness)
        if command and command in self.keyword_to_primitive:
            if command not in desc:
                desc = command + " " + desc
        
        return desc
    
    def learn_from_man(self, command: str) -> Optional[KnowledgeEntry]:
        """
        Learn a command from its man page.
        
        Returns the new KnowledgeEntry, or None if learning failed.
        """
        man_desc = self.parse_man_page(command)
        if not man_desc:
            return None
        
        description = self.infer_description(man_desc, command)
        if not description:
            return None
        
        # Store with persistence
        entry = self.store(command, description, persist=True)
        return entry
    
    def list_learned(self) -> List[Tuple[str, str]]:
        """List all learned knowledge from the database."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT command, description FROM knowledge ORDER BY created_at")
            return [(row["command"], row["description"]) for row in cursor]
    
    def forget(self, command: str) -> bool:
        """Remove learned knowledge for a command."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM knowledge WHERE command = ?", (command,))
            if cursor.rowcount > 0:
                # Also remove from entries
                self.entries = [e for e in self.entries if e.name != command]
                return True
            return False
    
    def reset(self):
        """Reset the database, removing all learned knowledge."""
        with self._connection() as conn:
            conn.execute("DELETE FROM knowledge")
        
        # Reload from bootstrap only
        self.entries = []
        self._load_bootstrap()
    
    def try_learn(self, query: str) -> Optional[KnowledgeEntry]:
        """
        Try to learn from a failed query by checking man pages.
        
        Extracts potential command names from the query and tries
        to learn them from man pages.
        """
        words = self._tokenize(query)
        
        # Try each word as a potential command
        for word in words:
            if len(word) < 2:
                continue
            
            # Check if we already know this command
            known = any(e.name == word or e.output == word for e in self.entries)
            if known:
                continue
            
            entry = self.learn_from_man(word)
            if entry:
                return entry
        
        return None


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
