"""
TruthSpace: Unified Knowledge Storage and Query Interface

This is the consolidated knowledge layer that replaces:
- knowledge_db.py
- knowledge_manager.py  
- intent_manager.py

Design principles:
- Single interface for all knowledge operations
- Everything is a KnowledgeEntry (including intents, primitives, patterns)
- No fallbacks - query succeeds or raises KnowledgeGapError
- Fail fast philosophy
"""

import sqlite3
import numpy as np
import json
import os
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager


# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2


class KnowledgeDomain(Enum):
    """Knowledge domains for isolation in TruthSpace."""
    PROGRAMMING = "programming"
    SYSTEM = "system"
    GENERAL = "general"


class EntryType(Enum):
    """Types of knowledge entries."""
    PRIMITIVE = "primitive"      # Semantic anchor (CREATE, READ, FILE, etc.)
    INTENT = "intent"            # NL trigger → command mapping
    COMMAND = "command"          # Executable command knowledge
    PATTERN = "pattern"          # Code/command template
    CONCEPT = "concept"          # General knowledge
    META = "meta"                # Stop words, config, etc.


DOMAIN_OFFSETS = {
    KnowledgeDomain.PROGRAMMING: 1.0,
    KnowledgeDomain.SYSTEM: 2.0,
    KnowledgeDomain.GENERAL: 3.0,
}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class KnowledgeGapError(Exception):
    """Raised when knowledge is not found - triggers learning opportunity."""
    def __init__(self, query: str, best_match: float = 0.0):
        self.query = query
        self.best_match = best_match
        super().__init__(f"Knowledge gap: '{query}' (best match: {best_match:.2f})")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class KnowledgeEntry:
    """A single piece of knowledge in TruthSpace."""
    id: str
    name: str
    entry_type: EntryType
    domain: KnowledgeDomain
    description: str
    position: np.ndarray
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entry_type": self.entry_type.value,
            "domain": self.domain.value,
            "description": self.description,
            "position": self.position.tolist(),
            "keywords": self.keywords,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        return cls(
            id=data["id"],
            name=data["name"],
            entry_type=EntryType(data["entry_type"]),
            domain=KnowledgeDomain(data["domain"]),
            description=data["description"],
            position=np.array(data["position"]),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            version=data.get("version", 1),
        )


@dataclass
class QueryResult:
    """Result of a TruthSpace query."""
    entry: KnowledgeEntry
    similarity: float
    
    
# =============================================================================
# TRUTHSPACE: THE UNIFIED KNOWLEDGE INTERFACE
# =============================================================================

class TruthSpace:
    """
    Unified knowledge storage and query interface.
    
    This is the single source of truth for all knowledge in the system.
    Everything - primitives, intents, commands, patterns - lives here.
    
    Core operations:
    - store(entry) → persist knowledge
    - query(text) → find matching knowledge or raise KnowledgeGapError
    - resolve(text) → query + extract executable output
    """
    
    DIM = 8  # Dimension of φ-space
    SCHEMA_VERSION = 2
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                "..", "truthspace.db"
            )
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encoder will be set externally to avoid circular import
        self._encoder = None
        
        self._init_db()
    
    def set_encoder(self, encoder):
        """Set the φ-encoder (called after encoder is initialized)."""
        self._encoder = encoder
    
    @contextmanager
    def _connection(self):
        """Database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    description TEXT NOT NULL,
                    position BLOB NOT NULL,
                    position_norm REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT UNIQUE NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entry_keywords (
                    entry_id TEXT NOT NULL,
                    keyword_id INTEGER NOT NULL,
                    PRIMARY KEY (entry_id, keyword_id),
                    FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE,
                    FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_domain ON entries(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_name ON entries(name)")
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to φ-position."""
        if self._encoder is None:
            # Fallback: simple keyword-based encoding
            return self._simple_encode(text)
        return self._encoder.encode(text).position
    
    def _simple_encode(self, text: str) -> np.ndarray:
        """Simple encoding when φ-encoder not available (bootstrap)."""
        position = np.zeros(self.DIM)
        words = text.lower().split()
        
        for i, word in enumerate(words[:self.DIM]):
            # Use character sum as simple hash
            position[i % self.DIM] += sum(ord(c) for c in word) / 1000.0
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 0:
            position = position / norm
        
        return position
    
    def _position_to_blob(self, position: np.ndarray) -> bytes:
        return position.astype(np.float64).tobytes()
    
    def _blob_to_position(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float64)
    
    # =========================================================================
    # STORE: Add Knowledge
    # =========================================================================
    
    def store(
        self,
        name: str,
        entry_type: EntryType,
        domain: KnowledgeDomain,
        description: str,
        keywords: List[str] = None,
        metadata: Dict[str, Any] = None,
        position: np.ndarray = None,
    ) -> KnowledgeEntry:
        """
        Store knowledge in TruthSpace.
        
        If position is not provided, it's computed from name + keywords.
        """
        keywords = keywords or []
        metadata = metadata or {}
        
        # Generate ID
        now = datetime.now()
        content = f"{entry_type.value}:{domain.value}:{name}:{now.isoformat()}"
        entry_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Compute position if not provided
        if position is None:
            text = f"{name} {' '.join(keywords)}"
            position = self._encode(text)
            
            # Add domain offset
            domain_offset = DOMAIN_OFFSETS.get(domain, 0)
            position = position * 0.8
            position[0] += domain_offset * 0.2
        
        position_norm = float(np.linalg.norm(position))
        
        with self._connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO entries (id, name, entry_type, domain, description,
                                    position, position_norm, metadata, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id, name, entry_type.value, domain.value, description,
                self._position_to_blob(position), position_norm,
                json.dumps(metadata), now.isoformat(), now.isoformat(), 1
            ))
            
            for kw in keywords:
                cursor.execute("INSERT OR IGNORE INTO keywords (keyword) VALUES (?)", (kw.lower(),))
                cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (kw.lower(),))
                kw_id = cursor.fetchone()[0]
                cursor.execute(
                    "INSERT OR IGNORE INTO entry_keywords (entry_id, keyword_id) VALUES (?, ?)",
                    (entry_id, kw_id)
                )
        
        return KnowledgeEntry(
            id=entry_id,
            name=name,
            entry_type=entry_type,
            domain=domain,
            description=description,
            position=position,
            keywords=keywords,
            metadata=metadata,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            version=1,
        )
    
    # =========================================================================
    # QUERY: Find Knowledge (Fail Fast)
    # =========================================================================
    
    def query(
        self,
        text: str,
        entry_type: EntryType = None,
        domain: KnowledgeDomain = None,
        threshold: float = 0.3,
        top_k: int = 10,
    ) -> List[QueryResult]:
        """
        Query TruthSpace for matching knowledge.
        
        Returns list of results sorted by similarity.
        Raises KnowledgeGapError if no results above threshold.
        """
        query_vec = self._encode(text)
        
        if domain:
            query_vec = query_vec.copy()
            query_vec[0] += DOMAIN_OFFSETS.get(domain, 0) * 0.2
        
        query_norm = np.linalg.norm(query_vec)
        
        results = []
        keywords = set(text.lower().split())
        
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            sql = "SELECT * FROM entries WHERE 1=1"
            params = []
            
            if entry_type:
                sql += " AND entry_type = ?"
                params.append(entry_type.value)
            
            if domain:
                sql += " AND domain = ?"
                params.append(domain.value)
            
            cursor.execute(sql, params)
            
            for row in cursor.fetchall():
                position = self._blob_to_position(row["position"])
                position_norm = row["position_norm"]
                
                # Geometric similarity
                if position_norm > 0 and query_norm > 0:
                    geo_sim = float(np.dot(query_vec, position) / (query_norm * position_norm))
                else:
                    geo_sim = 0.0
                
                # Keyword boost
                cursor.execute("""
                    SELECT k.keyword FROM keywords k
                    JOIN entry_keywords ek ON k.id = ek.keyword_id
                    WHERE ek.entry_id = ?
                """, (row["id"],))
                entry_keywords = set(r[0] for r in cursor.fetchall())
                entry_keywords.add(row["name"].lower())
                
                keyword_boost = 0.0
                for qkw in keywords:
                    if qkw in entry_keywords:
                        keyword_boost += 0.15
                    else:
                        for ekw in entry_keywords:
                            if qkw in ekw or ekw in qkw:
                                keyword_boost += 0.05
                                break
                
                combined_sim = geo_sim + min(keyword_boost, 0.4)
                
                entry = KnowledgeEntry(
                    id=row["id"],
                    name=row["name"],
                    entry_type=EntryType(row["entry_type"]),
                    domain=KnowledgeDomain(row["domain"]),
                    description=row["description"],
                    position=position,
                    keywords=list(entry_keywords),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    version=row["version"],
                )
                
                results.append(QueryResult(entry=entry, similarity=combined_sim))
        
        results.sort(key=lambda x: -x.similarity)
        results = results[:top_k]
        
        # Fail fast: raise if no good matches
        if not results or results[0].similarity < threshold:
            best = results[0].similarity if results else 0.0
            raise KnowledgeGapError(text, best)
        
        return results
    
    def query_safe(
        self,
        text: str,
        entry_type: EntryType = None,
        domain: KnowledgeDomain = None,
        top_k: int = 10,
    ) -> List[QueryResult]:
        """Query without raising KnowledgeGapError (returns empty list instead)."""
        try:
            return self.query(text, entry_type, domain, threshold=0.0, top_k=top_k)
        except KnowledgeGapError:
            return []
    
    # =========================================================================
    # RESOLVE: Query + Extract Output
    # =========================================================================
    
    def resolve(
        self,
        text: str,
        domain: KnowledgeDomain = None,
    ) -> Tuple[str, str, KnowledgeEntry]:
        """
        Resolve natural language to executable output.
        
        Returns: (output, output_type, entry)
        Raises: KnowledgeGapError if no match
        """
        # First try intents (high precision pattern matching)
        try:
            results = self.query(text, entry_type=EntryType.INTENT, domain=domain, threshold=0.5)
            if results:
                entry = results[0].entry
                output = self._extract_output(entry)
                output_type = entry.metadata.get("output_type", "bash")
                return output, output_type, entry
        except KnowledgeGapError:
            pass
        
        # Then try commands
        try:
            results = self.query(text, entry_type=EntryType.COMMAND, domain=domain, threshold=0.4)
            if results:
                entry = results[0].entry
                output = self._extract_output(entry)
                output_type = entry.metadata.get("output_type", "bash")
                return output, output_type, entry
        except KnowledgeGapError:
            pass
        
        # Finally try any knowledge
        results = self.query(text, domain=domain, threshold=0.3)
        entry = results[0].entry
        output = self._extract_output(entry)
        output_type = entry.metadata.get("output_type", "text")
        return output, output_type, entry
    
    def _extract_output(self, entry: KnowledgeEntry) -> str:
        """Extract executable output from entry."""
        # Check metadata fields in order of preference
        for field in ["code", "command", "syntax", "output"]:
            if entry.metadata.get(field):
                return entry.metadata[field]
        
        # For intents, get target command
        if entry.entry_type == EntryType.INTENT:
            commands = entry.metadata.get("target_commands", [])
            if commands:
                return commands[0]
        
        return entry.name
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get entry by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            cursor.execute("""
                SELECT k.keyword FROM keywords k
                JOIN entry_keywords ek ON k.id = ek.keyword_id
                WHERE ek.entry_id = ?
            """, (entry_id,))
            keywords = [r[0] for r in cursor.fetchall()]
            
            return KnowledgeEntry(
                id=row["id"],
                name=row["name"],
                entry_type=EntryType(row["entry_type"]),
                domain=KnowledgeDomain(row["domain"]),
                description=row["description"],
                position=self._blob_to_position(row["position"]),
                keywords=keywords,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                version=row["version"],
            )
    
    def get_by_name(self, name: str, entry_type: EntryType = None) -> Optional[KnowledgeEntry]:
        """Get entry by name."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            if entry_type:
                cursor.execute(
                    "SELECT id FROM entries WHERE name = ? AND entry_type = ?",
                    (name, entry_type.value)
                )
            else:
                cursor.execute("SELECT id FROM entries WHERE name = ?", (name,))
            
            row = cursor.fetchone()
            return self.get(row["id"]) if row else None
    
    def list_by_type(self, entry_type: EntryType) -> List[KnowledgeEntry]:
        """List all entries of a type."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entries WHERE entry_type = ?", (entry_type.value,))
            return [self.get(row["id"]) for row in cursor.fetchall()]
    
    def count(self) -> Dict[str, int]:
        """Count entries by type."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_type, COUNT(*) as count
                FROM entries GROUP BY entry_type
            """)
            return {row["entry_type"]: row["count"] for row in cursor.fetchall()}
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 70)
    print("TRUTHSPACE - Unified Knowledge Interface")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    ts = TruthSpace(db_path)
    
    # Store some knowledge
    print("\nStoring knowledge...")
    
    ts.store(
        name="ls",
        entry_type=EntryType.COMMAND,
        domain=KnowledgeDomain.PROGRAMMING,
        description="List directory contents",
        keywords=["list", "files", "directory", "bash"],
        metadata={"command": "ls -la", "output_type": "bash"}
    )
    
    ts.store(
        name="list_files_intent",
        entry_type=EntryType.INTENT,
        domain=KnowledgeDomain.PROGRAMMING,
        description="List files in directory",
        keywords=["list", "files", "show", "directory"],
        metadata={"target_commands": ["ls -la"], "output_type": "bash"}
    )
    
    print(f"  Stored {ts.count()} entries")
    
    # Query
    print("\nQuerying 'show files in directory'...")
    try:
        results = ts.query("show files in directory")
        for r in results:
            print(f"  {r.similarity:.3f} | {r.entry.name} ({r.entry.entry_type.value})")
    except KnowledgeGapError as e:
        print(f"  Knowledge gap: {e}")
    
    # Resolve
    print("\nResolving 'list all files'...")
    try:
        output, output_type, entry = ts.resolve("list all files")
        print(f"  Output: {output}")
        print(f"  Type: {output_type}")
    except KnowledgeGapError as e:
        print(f"  Knowledge gap: {e}")
    
    # Cleanup
    os.unlink(db_path)
    
    print("\n" + "=" * 70)
    print("TruthSpace test complete!")
    print("=" * 70)
