"""
Knowledge Database: SQLite backend optimized for φ-based vectors.

This replaces the file-based JSON storage with a proper database that:
1. Stores 8-dimensional φ-encoded positions efficiently
2. Supports fast similarity queries using pre-computed norms
3. Maintains domain isolation through indexed columns
4. Provides ACID guarantees for knowledge operations

Schema Design:
- entries: Main knowledge entries with position vectors stored as BLOBs
- keywords: Normalized keyword table for fast keyword-based filtering
- entry_keywords: Many-to-many relationship
- primitives: Cache of φ-encoder primitive decompositions (optional optimization)
"""

import sqlite3
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

from truthspace_lcm.core.phi_encoder import PhiEncoder


# Domain constants for domain isolation in TruthSpace
class KnowledgeDomain(Enum):
    PROGRAMMING = "programming"
    HISTORY = "history"
    SCIENCE = "science"
    GEOGRAPHY = "geography"
    GENERAL = "general"
    CUSTOM = "custom"


DOMAIN_CONSTANTS = {
    KnowledgeDomain.PROGRAMMING: 1.0,
    KnowledgeDomain.HISTORY: 2.0,
    KnowledgeDomain.SCIENCE: 3.0,
    KnowledgeDomain.GEOGRAPHY: 4.0,
    KnowledgeDomain.GENERAL: 5.0,
    KnowledgeDomain.CUSTOM: 6.0,
}


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge."""
    id: str
    name: str
    domain: KnowledgeDomain
    entry_type: str
    description: str
    position: np.ndarray
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain.value,
            "entry_type": self.entry_type,
            "description": self.description,
            "position": self.position.tolist(),
            "keywords": self.keywords,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            domain=KnowledgeDomain(data["domain"]),
            entry_type=data["entry_type"],
            description=data["description"],
            position=np.array(data["position"]),
            keywords=data.get("keywords", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            version=data.get("version", 1)
        )
    
    def similarity_to(self, other: 'KnowledgeEntry') -> float:
        """Compute similarity (only meaningful within same domain)."""
        if self.domain != other.domain:
            return 0.0
        
        n1 = np.linalg.norm(self.position)
        n2 = np.linalg.norm(other.position)
        if n1 > 0 and n2 > 0:
            return float(np.dot(self.position, other.position) / (n1 * n2))
        return 0.0


class KnowledgeDB:
    """
    SQLite-based knowledge storage optimized for φ-based vectors.
    
    Features:
    - Efficient 8-dimensional vector storage as BLOBs
    - Pre-computed norms for fast similarity calculations
    - Indexed domain and keyword columns for filtering
    - Full-text search on descriptions
    - ACID transactions for safe updates
    """
    
    SCHEMA_VERSION = 1
    DIM = 8  # φ-encoder dimension
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                "..", "knowledge.db"
            )
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize φ-encoder
        self._phi_encoder = PhiEncoder()
        
        # Initialize database
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    position BLOB NOT NULL,
                    position_norm REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1
                )
            """)
            
            # Keywords table (normalized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT UNIQUE NOT NULL
                )
            """)
            
            # Entry-keyword relationship
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entry_keywords (
                    entry_id TEXT NOT NULL,
                    keyword_id INTEGER NOT NULL,
                    PRIMARY KEY (entry_id, keyword_id),
                    FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE,
                    FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for fast queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_domain ON entries(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_name ON entries(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword)")
            
            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ("version", str(self.SCHEMA_VERSION))
            )
    
    def _position_to_blob(self, position: np.ndarray) -> bytes:
        """Convert numpy array to blob for storage."""
        return position.astype(np.float64).tobytes()
    
    def _blob_to_position(self, blob: bytes) -> np.ndarray:
        """Convert blob back to numpy array."""
        return np.frombuffer(blob, dtype=np.float64)
    
    def _compute_position(self, name: str, domain: KnowledgeDomain,
                          keywords: List[str]) -> np.ndarray:
        """Compute φ-based position for an entry."""
        text_to_encode = f"{name} {' '.join(keywords)}"
        decomposition = self._phi_encoder.encode(text_to_encode)
        position = decomposition.position.copy()
        
        # Add domain component for isolation
        domain_offset = DOMAIN_CONSTANTS[domain]
        position = position * 0.8
        position[0] += domain_offset * 0.2
        
        return position
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def create(self, name: str, domain: KnowledgeDomain, entry_type: str,
               description: str, keywords: List[str] = None,
               metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """Create a new knowledge entry."""
        keywords = keywords or []
        metadata = metadata or {}
        
        # Generate ID
        now = datetime.now()
        import hashlib
        content = f"{domain.value}:{name}:{now.isoformat()}"
        entry_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Compute position
        position = self._compute_position(name, domain, keywords)
        position_norm = float(np.linalg.norm(position))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert entry
            cursor.execute("""
                INSERT INTO entries (id, name, domain, entry_type, description,
                                    position, position_norm, metadata, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id, name, domain.value, entry_type, description,
                self._position_to_blob(position), position_norm,
                json.dumps(metadata), now.isoformat(), now.isoformat(), 1
            ))
            
            # Insert keywords
            for kw in keywords:
                cursor.execute(
                    "INSERT OR IGNORE INTO keywords (keyword) VALUES (?)",
                    (kw.lower(),)
                )
                cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (kw.lower(),))
                kw_id = cursor.fetchone()[0]
                cursor.execute(
                    "INSERT OR IGNORE INTO entry_keywords (entry_id, keyword_id) VALUES (?, ?)",
                    (entry_id, kw_id)
                )
        
        return KnowledgeEntry(
            id=entry_id,
            name=name,
            domain=domain,
            entry_type=entry_type,
            description=description,
            position=position,
            keywords=keywords,
            metadata=metadata,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            version=1
        )
    
    def read(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Read a single entry by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get keywords
            cursor.execute("""
                SELECT k.keyword FROM keywords k
                JOIN entry_keywords ek ON k.id = ek.keyword_id
                WHERE ek.entry_id = ?
            """, (entry_id,))
            keywords = [r[0] for r in cursor.fetchall()]
            
            return KnowledgeEntry(
                id=row["id"],
                name=row["name"],
                domain=KnowledgeDomain(row["domain"]),
                entry_type=row["entry_type"],
                description=row["description"],
                position=self._blob_to_position(row["position"]),
                keywords=keywords,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                version=row["version"]
            )
    
    def update(self, entry_id: str, **updates) -> Optional[KnowledgeEntry]:
        """Update an existing entry."""
        entry = self.read(entry_id)
        if not entry:
            return None
        
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build update query
            set_clauses = ["updated_at = ?", "version = version + 1"]
            params = [now]
            
            if "name" in updates:
                set_clauses.append("name = ?")
                params.append(updates["name"])
            
            if "description" in updates:
                set_clauses.append("description = ?")
                params.append(updates["description"])
            
            if "entry_type" in updates:
                set_clauses.append("entry_type = ?")
                params.append(updates["entry_type"])
            
            if "metadata" in updates:
                set_clauses.append("metadata = ?")
                params.append(json.dumps(updates["metadata"]))
            
            # Recompute position if name or keywords changed
            if "name" in updates or "keywords" in updates:
                new_name = updates.get("name", entry.name)
                new_keywords = updates.get("keywords", entry.keywords)
                position = self._compute_position(new_name, entry.domain, new_keywords)
                set_clauses.append("position = ?")
                set_clauses.append("position_norm = ?")
                params.append(self._position_to_blob(position))
                params.append(float(np.linalg.norm(position)))
            
            params.append(entry_id)
            
            cursor.execute(
                f"UPDATE entries SET {', '.join(set_clauses)} WHERE id = ?",
                params
            )
            
            # Update keywords if provided
            if "keywords" in updates:
                cursor.execute("DELETE FROM entry_keywords WHERE entry_id = ?", (entry_id,))
                for kw in updates["keywords"]:
                    cursor.execute(
                        "INSERT OR IGNORE INTO keywords (keyword) VALUES (?)",
                        (kw.lower(),)
                    )
                    cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (kw.lower(),))
                    kw_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT INTO entry_keywords (entry_id, keyword_id) VALUES (?, ?)",
                        (entry_id, kw_id)
                    )
        
        return self.read(entry_id)
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def query(self, keywords: List[str], domain: KnowledgeDomain = None,
              top_k: int = 10) -> List[Tuple[float, KnowledgeEntry]]:
        """
        Query knowledge using φ-based semantic similarity.
        
        Combines geometric similarity with keyword boosting.
        """
        # Encode query
        query_text = " ".join(keywords)
        decomposition = self._phi_encoder.encode(query_text)
        query_vec = decomposition.position
        
        if domain:
            query_vec = query_vec.copy()
            query_vec[0] += DOMAIN_CONSTANTS[domain] * 0.2
        
        query_norm = np.linalg.norm(query_vec)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with optional domain filter
            if domain:
                cursor.execute(
                    "SELECT * FROM entries WHERE domain = ?",
                    (domain.value,)
                )
            else:
                cursor.execute("SELECT * FROM entries")
            
            results = []
            keyword_set = set(kw.lower() for kw in keywords)
            
            for row in cursor.fetchall():
                # Get entry position
                position = self._blob_to_position(row["position"])
                position_norm = row["position_norm"]
                
                # Compute geometric similarity
                if position_norm > 0 and query_norm > 0:
                    geo_sim = float(np.dot(query_vec, position) / (query_norm * position_norm))
                else:
                    geo_sim = 0.0
                
                # Get keywords for boosting
                cursor.execute("""
                    SELECT k.keyword FROM keywords k
                    JOIN entry_keywords ek ON k.id = ek.keyword_id
                    WHERE ek.entry_id = ?
                """, (row["id"],))
                entry_keywords = set(r[0] for r in cursor.fetchall())
                entry_keywords.add(row["name"].lower())
                
                # Keyword boost
                keyword_boost = 0.0
                for qkw in keyword_set:
                    if qkw in entry_keywords:
                        keyword_boost += 0.15
                    else:
                        for ekw in entry_keywords:
                            if qkw in ekw or ekw in qkw:
                                keyword_boost += 0.05
                                break
                
                combined_sim = geo_sim + min(keyword_boost, 0.4)
                
                # Build entry
                entry = KnowledgeEntry(
                    id=row["id"],
                    name=row["name"],
                    domain=KnowledgeDomain(row["domain"]),
                    entry_type=row["entry_type"],
                    description=row["description"],
                    position=position,
                    keywords=list(entry_keywords),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    version=row["version"]
                )
                
                results.append((combined_sim, entry))
            
            results.sort(key=lambda x: -x[0])
            return results[:top_k]
    
    def query_by_name(self, name: str, domain: KnowledgeDomain = None) -> Optional[KnowledgeEntry]:
        """Find entry by exact name match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if domain:
                cursor.execute(
                    "SELECT id FROM entries WHERE name = ? AND domain = ?",
                    (name, domain.value)
                )
            else:
                cursor.execute("SELECT id FROM entries WHERE name = ?", (name,))
            
            row = cursor.fetchone()
            if row:
                return self.read(row["id"])
            return None
    
    def list_domain(self, domain: KnowledgeDomain) -> List[KnowledgeEntry]:
        """List all entries in a domain."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entries WHERE domain = ?", (domain.value,))
            return [self.read(row["id"]) for row in cursor.fetchall()]
    
    def count_by_domain(self) -> Dict[str, int]:
        """Count entries by domain."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT domain, COUNT(*) as count
                FROM entries
                GROUP BY domain
            """)
            return {row["domain"]: row["count"] for row in cursor.fetchall()}
    
    def get_all_entries(self) -> Dict[str, KnowledgeEntry]:
        """Get all entries as a dictionary (for compatibility)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entries")
            return {row["id"]: self.read(row["id"]) for row in cursor.fetchall()}
    
    # =========================================================================
    # MIGRATION
    # =========================================================================
    
    def import_from_json_files(self, storage_dir: str):
        """Import entries from JSON file-based storage."""
        storage_path = Path(storage_dir)
        imported = 0
        
        for domain in KnowledgeDomain:
            domain_dir = storage_path / domain.value
            if not domain_dir.exists():
                continue
            
            for json_file in domain_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    entry = KnowledgeEntry.from_dict(data)
                    
                    # Insert directly with existing position
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO entries 
                            (id, name, domain, entry_type, description,
                             position, position_norm, metadata, created_at, updated_at, version)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            entry.id, entry.name, entry.domain.value, entry.entry_type,
                            entry.description, self._position_to_blob(entry.position),
                            float(np.linalg.norm(entry.position)),
                            json.dumps(entry.metadata), entry.created_at,
                            entry.updated_at, entry.version
                        ))
                        
                        # Insert keywords
                        for kw in entry.keywords:
                            cursor.execute(
                                "INSERT OR IGNORE INTO keywords (keyword) VALUES (?)",
                                (kw.lower(),)
                            )
                            cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (kw.lower(),))
                            kw_id = cursor.fetchone()[0]
                            cursor.execute(
                                "INSERT OR IGNORE INTO entry_keywords (entry_id, keyword_id) VALUES (?, ?)",
                                (entry.id, kw_id)
                            )
                    
                    imported += 1
                except Exception as e:
                    print(f"Warning: Could not import {json_file}: {e}")
        
        return imported


# Demonstration
if __name__ == "__main__":
    import tempfile
    
    print("=" * 70)
    print("KNOWLEDGE DATABASE - φ-Based SQLite Backend")
    print("=" * 70)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db = KnowledgeDB(db_path)
    
    # Create some entries
    print("\nCreating entries...")
    
    entry1 = db.create(
        name="ls",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="command",
        description="List directory contents",
        keywords=["bash", "shell", "list", "files", "directory"],
        metadata={"syntax": "ls [options] [path]"}
    )
    print(f"  Created: {entry1.name} (id: {entry1.id})")
    
    entry2 = db.create(
        name="mkdir",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="command",
        description="Create directories",
        keywords=["bash", "shell", "create", "directory", "folder"],
        metadata={"syntax": "mkdir [options] directory"}
    )
    print(f"  Created: {entry2.name} (id: {entry2.id})")
    
    # Query
    print("\nQuerying 'list files'...")
    results = db.query(["list", "files"], domain=KnowledgeDomain.PROGRAMMING, top_k=3)
    for sim, entry in results:
        print(f"  {entry.name}: {sim:.3f}")
    
    print("\nQuerying 'create directory'...")
    results = db.query(["create", "directory"], domain=KnowledgeDomain.PROGRAMMING, top_k=3)
    for sim, entry in results:
        print(f"  {entry.name}: {sim:.3f}")
    
    # Count
    print(f"\nEntries by domain: {db.count_by_domain()}")
    
    # Cleanup
    os.unlink(db_path)
    
    print("\n" + "=" * 70)
    print("Database test complete!")
    print("=" * 70)
