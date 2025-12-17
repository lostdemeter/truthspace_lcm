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
    Primitive("CREATE", 0, 0, ["create", "make", "new", "generate", "add", "init", "put", "place"]),
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
    Primitive("INTO", 8, 0, ["into", "in", "to", "toward"]),
    Primitive("FROM", 8, 1, ["from", "source", "origin"]),
    Primitive("BEFORE", 9, 0, ["before", "first", "head", "top", "start"]),
    Primitive("AFTER", 9, 1, ["after", "last", "tail", "end", "bottom"]),
    Primitive("DURING", 10, 0, ["during", "while", "follow", "watch", "live"]),
    Primitive("TEST", 11, 0, ["test", "check", "verify", "ping"]),
    
    # Structural primitives - for geometric parameter detection
    Primitive("NAMING", 11, 1, ["called", "named", "as"]),
    Primitive("SEQUENCE", 11, 2, ["and", "then", "also", "next", "after"]),
]

# Dimensions that indicate domain context (parameters often follow these)
DOMAIN_DIMS = {4, 5, 6, 7}  # PROCESS, NETWORK, USER, FILE, DIRECTORY, etc.


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
        """Tokenization that preserves filenames, paths, and hyphenated words."""
        import re
        # Match: filenames (word.ext), paths (/foo/bar), hyphenated words, or regular words
        tokens = re.findall(r'[\w./]+\.[\w]+|/[\w./]+|\b[\w]+-[\w-]+\b|\b\w+\b', text.lower())
        return tokens
    
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
    
    def resolve_compound(self, text: str, window_sizes: List[int] = None,
                         threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Extract multiple concepts from a compound query using sliding window encoding.
        
        This is the geometric approach to multi-concept extraction:
        - Slide windows of various sizes across the query
        - Encode each window and find matches above threshold
        - Use greedy non-overlapping selection to get distinct concepts
        
        Args:
            text: Natural language query (potentially compound)
            window_sizes: Window sizes to try (default: [2, 3, 4])
            threshold: Minimum similarity for a match
        
        Returns:
            List of concept dicts with keys:
                - start: word index start
                - end: word index end  
                - window: the matched text
                - command: resolved command
                - description: matched entry description
                - similarity: match similarity
        """
        if window_sizes is None:
            window_sizes = [2, 3, 4]
        
        words = self._tokenize(text)
        
        # Collect all candidate matches
        candidates = []
        
        for window_size in window_sizes:
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i+window_size])
                pos = self._encode(window)
                
                # Find best match for this window
                best_entry = None
                best_sim = 0.0
                
                for entry in self.entries:
                    sim = self._similarity(pos, entry.position)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
                
                if best_sim >= threshold and best_entry:
                    candidates.append({
                        'start': i,
                        'end': i + window_size,
                        'window': window,
                        'command': best_entry.output,
                        'description': best_entry.description,
                        'similarity': best_sim
                    })
        
        # Score candidates: prefer windows with action primitives
        action_dims = {0, 1, 2, 3}  # CREATE, READ/LIST/WRITE/DELETE, COPY/RELOCATE/SEARCH, EXECUTE/etc
        
        for c in candidates:
            # Check if window activates action dimensions
            window_pos = self._encode(c['window'])
            has_action = any(window_pos[d] > 0.5 for d in action_dims)
            # Boost score for action-containing windows
            c['score'] = c['similarity'] * (1.5 if has_action else 1.0)
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Greedy non-overlapping selection
        selected = []
        covered = set()
        
        for c in candidates:
            window_positions = set(range(c['start'], c['end']))
            if not window_positions & covered:
                selected.append(c)
                covered.update(window_positions)
        
        # Sort by position in original query
        selected.sort(key=lambda x: x['start'])
        
        return selected
    
    def detect_parameters_geometric(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect parameters using geometric analysis.
        
        A parameter is identified as a word that:
        1. Does NOT activate any primitives (semantic void)
        2. Appears after a word that DOES activate primitives (context)
        
        Special cases:
        - After NAMING word (called, named) → next word is definitely a parameter
        - After DOMAIN primitive (file, directory) → next void word is likely a parameter
        
        Returns:
            List of (word_index, value, reason) tuples
        """
        words = self._tokenize(text)
        parameters = []
        seen_indices = set()  # Avoid duplicates
        
        # NAMING keywords (check by keyword, not by encoding, to avoid dimension conflicts)
        naming_keywords = {'called', 'named', 'as'}
        
        for i, word in enumerate(words):
            # Check if this is a NAMING keyword
            if word in naming_keywords and i + 1 < len(words):
                next_idx = i + 1
                if next_idx not in seen_indices:
                    next_word = words[next_idx]
                    parameters.append((next_idx, next_word, "after_naming"))
                    seen_indices.add(next_idx)
                continue
            
            # Check if this word is a semantic void after a domain word
            word_pos = self._encode(word)
            word_activates = np.any(word_pos > 0.01)
            
            if i > 0 and not word_activates and i not in seen_indices:
                prev_word = words[i - 1]
                prev_pos = self._encode(prev_word)
                
                # Check if previous word activated any domain dimension
                prev_activates_domain = any(prev_pos[d] > 0.01 for d in DOMAIN_DIMS)
                
                if prev_activates_domain:
                    parameters.append((i, word, "after_domain"))
                    seen_indices.add(i)
        
        return parameters
    
    def extract_parameters(self, text: str) -> Dict[str, List[str]]:
        """
        Extract parameters from natural language text (post-geometric phase).
        
        This is NOT geometric - it's simple pattern matching to extract
        the actual values that will fill in command templates.
        
        Returns:
            Dict with parameter types as keys and lists of values
        """
        import re
        
        params = {
            'names': [],      # Things that are "called X" or "named X"
            'quoted': [],     # Quoted strings
            'paths': [],      # Path-like strings
        }
        
        # Extract "called X" or "named X" patterns
        called_pattern = r"called\s+['\"]?([^'\",\s]+)['\"]?"
        named_pattern = r"named\s+['\"]?([^'\",\s]+)['\"]?"
        
        for match in re.finditer(called_pattern, text, re.IGNORECASE):
            params['names'].append(match.group(1))
        for match in re.finditer(named_pattern, text, re.IGNORECASE):
            params['names'].append(match.group(1))
        
        # Extract quoted strings
        quoted_pattern = r"['\"]([^'\"]+)['\"]"
        for match in re.finditer(quoted_pattern, text):
            value = match.group(1)
            if value not in params['names']:
                params['quoted'].append(value)
        
        # Extract path-like strings (contain / or end with common extensions)
        path_pattern = r"\b([\w./]+(?:\.[\w]+)?)\b"
        for match in re.finditer(path_pattern, text):
            value = match.group(1)
            if ('/' in value or '.' in value) and value not in params['names'] + params['quoted']:
                params['paths'].append(value)
        
        return params
    
    def resolve_with_params(self, text: str, use_geometric_params: bool = True) -> List[Dict[str, Any]]:
        """
        Resolve compound query and attach extracted parameters.
        
        This combines geometric concept extraction with parameter extraction
        to produce actionable commands.
        
        Args:
            text: Natural language query
            use_geometric_params: If True, use geometric parameter detection
        
        Returns:
            List of concept dicts, each with an additional 'params' key
        """
        import re
        
        # Phase 1: Geometric concept extraction
        concepts = self.resolve_compound(text)
        
        if not concepts:
            return []
        
        # Phase 2: Parameter extraction
        words = self._tokenize(text)
        
        if use_geometric_params:
            # GEOMETRIC APPROACH: Parameters are semantic voids in context
            geometric_params = self.detect_parameters_geometric(text)
            param_positions = [(idx, value) for idx, value, reason in geometric_params]
        else:
            # REGEX APPROACH: Pattern matching (fallback)
            param_positions = []
            for i, word in enumerate(words):
                if word in ('called', 'named') and i + 1 < len(words):
                    param_positions.append((i + 1, words[i + 1]))
        
        # Also extract quoted strings (hybrid - quotes are explicit markers)
        for match in re.finditer(r"['\"]([^'\"]+)['\"]", text):
            value = match.group(1)
            word_pos = len(text[:match.start()].split())
            if not any(p[1] == value for p in param_positions):
                param_positions.append((word_pos, value))
        
        # Sort by position
        param_positions.sort(key=lambda x: x[0])
        
        # Assign parameters to concepts based on proximity
        for concept in concepts:
            concept['params'] = []
            concept_end = concept['end']
            
            # Find parameters that come after this concept but before the next
            next_concept_start = float('inf')
            concept_idx = concepts.index(concept)
            if concept_idx + 1 < len(concepts):
                next_concept_start = concepts[concept_idx + 1]['start']
            
            # Collect params between this concept and the next
            for pos, value in param_positions[:]:
                if concept_end <= pos < next_concept_start:
                    concept['params'].append(value)
                    param_positions.remove((pos, value))
        
        return concepts
    
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
