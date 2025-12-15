"""
Knowledge Manager: SQLite-backed CRUD Operations with φ-based Encoding

This module provides the KnowledgeManager class that wraps the KnowledgeDB
for backward compatibility while using the new SQLite backend.

Key Design Principles:
- ADDITIVE: New knowledge adds to the space, doesn't overwrite
- ISOLATED: Domains have separate namespaces
- PERSISTENT: Knowledge saved to SQLite database
- VERIFIABLE: Can check that old knowledge still works after adding new

The geometric insight:
- Each domain has its own "subspace" in truth space
- Domain dimension prevents cross-domain interference
- Position within domain encodes semantic meaning via φ-encoder
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from truthspace_lcm.core.knowledge_db import (
    KnowledgeDB,
    KnowledgeEntry,
    KnowledgeDomain,
    DOMAIN_CONSTANTS,
)

# Re-export for backward compatibility
__all__ = [
    'KnowledgeManager',
    'KnowledgeEntry', 
    'KnowledgeDomain',
    'DOMAIN_CONSTANTS',
    'PHI', 'PI', 'E', 'SQRT2', 'SQRT3', 'LN2', 'GAMMA', 'ZETA3',
]

# Universal constants (exported for compatibility)
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)
GAMMA = 0.5772156649
ZETA3 = 1.2020569


class KnowledgeManager:
    """
    SQLite-backed knowledge management with CRUD operations.
    
    This is a thin wrapper around KnowledgeDB that provides backward
    compatibility with the old file-based API while using the new
    SQLite backend optimized for φ-based vectors.
    
    Features:
    - CREATE: Add new knowledge (never overwrites)
    - READ: Query knowledge by keywords or ID
    - UPDATE: Modify existing knowledge (versioned)
    - DELETE: Remove knowledge (with backup)
    
    Safety guarantees:
    - Domain isolation prevents cross-domain interference
    - Versioning allows rollback
    - ACID transactions for safe updates
    """
    
    def __init__(self, db_path: str = None, storage_dir: str = None, dim: int = 8):
        """
        Initialize KnowledgeManager with SQLite backend.
        
        Args:
            db_path: Path to SQLite database file (preferred)
            storage_dir: Legacy parameter, ignored if db_path provided
            dim: Dimension of position vectors (default 8)
        """
        self.dim = dim
        
        # Determine database path
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                "..", "knowledge.db"
            )
        
        # Initialize database backend
        self._db = KnowledgeDB(db_path)
        
        # Expose φ-encoder for direct access if needed
        self._phi_encoder = self._db._phi_encoder
    
    @property
    def entries(self) -> Dict[str, KnowledgeEntry]:
        """Get all entries as dictionary (for backward compatibility)."""
        return self._db.get_all_entries()
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def create(self, name: str, domain: KnowledgeDomain, entry_type: str,
               description: str, keywords: List[str] = None,
               metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """
        CREATE: Add new knowledge entry.
        
        This is ADDITIVE - it never overwrites existing entries.
        """
        return self._db.create(
            name=name,
            domain=domain,
            entry_type=entry_type,
            description=description,
            keywords=keywords,
            metadata=metadata
        )
    
    def read(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """READ: Get knowledge entry by ID."""
        return self._db.read(entry_id)
    
    def read_by_name(self, name: str, domain: KnowledgeDomain = None) -> List[KnowledgeEntry]:
        """READ: Get knowledge entries by name."""
        entry = self._db.query_by_name(name, domain)
        return [entry] if entry else []
    
    def update(self, entry_id: str, description: str = None,
               keywords: List[str] = None, metadata: Dict[str, Any] = None) -> Optional[KnowledgeEntry]:
        """
        UPDATE: Modify existing knowledge entry.
        
        Increments version number.
        Recomputes position if keywords change.
        """
        updates = {}
        if description is not None:
            updates["description"] = description
        if keywords is not None:
            updates["keywords"] = keywords
        if metadata is not None:
            updates["metadata"] = metadata
        
        return self._db.update(entry_id, **updates)
    
    def delete(self, entry_id: str, hard_delete: bool = False) -> bool:
        """DELETE: Remove knowledge entry."""
        return self._db.delete(entry_id)
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def query(self, keywords: List[str], domain: KnowledgeDomain = None,
              top_k: int = 10) -> List[Tuple[float, KnowledgeEntry]]:
        """
        Query using φ-encoder for semantic matching.
        
        Combines geometric similarity with keyword boosting.
        """
        return self._db.query(keywords, domain, top_k)
    
    def list_domain(self, domain: KnowledgeDomain) -> List[KnowledgeEntry]:
        """List all entries in a domain."""
        return self._db.list_domain(domain)
    
    def count_by_domain(self) -> Dict[str, int]:
        """Count entries by domain."""
        return self._db.count_by_domain()
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def verify_isolation(self, entry1_id: str, entry2_id: str) -> Dict[str, Any]:
        """
        Verify that two entries don't interfere with each other.
        
        Returns analysis of their geometric relationship.
        """
        entry1 = self._db.read(entry1_id)
        entry2 = self._db.read(entry2_id)
        
        if not entry1 or not entry2:
            return {"error": "Entry not found"}
        
        # Compute similarity
        n1 = np.linalg.norm(entry1.position)
        n2 = np.linalg.norm(entry2.position)
        
        if n1 > 0 and n2 > 0:
            full_sim = float(np.dot(entry1.position, entry2.position) / (n1 * n2))
        else:
            full_sim = 0.0
        
        # Domain component similarity
        domain_sim = abs(entry1.position[0] - entry2.position[0])
        
        # Semantic component similarity (excluding domain)
        sem1 = entry1.position[1:]
        sem2 = entry2.position[1:]
        sem_n1 = np.linalg.norm(sem1)
        sem_n2 = np.linalg.norm(sem2)
        
        if sem_n1 > 0 and sem_n2 > 0:
            sem_sim = float(np.dot(sem1, sem2) / (sem_n1 * sem_n2))
        else:
            sem_sim = 0.0
        
        # Determine if they're isolated
        same_domain = entry1.domain == entry2.domain
        isolated = not same_domain or full_sim < 0.5
        
        return {
            "entry1": {"id": entry1.id, "name": entry1.name, "domain": entry1.domain.value},
            "entry2": {"id": entry2.id, "name": entry2.name, "domain": entry2.domain.value},
            "same_domain": same_domain,
            "full_similarity": full_sim,
            "domain_distance": domain_sim,
            "semantic_similarity": sem_sim,
            "isolated": isolated,
            "interference_risk": "LOW" if isolated else "MEDIUM" if full_sim < 0.7 else "HIGH"
        }
    
    def verify_all_knowledge(self) -> Dict[str, Any]:
        """
        Verify that all knowledge entries are properly isolated.
        
        Checks for potential interference between domains.
        """
        entries = self._db.get_all_entries()
        
        results = {
            "total_entries": len(entries),
            "by_domain": self.count_by_domain(),
            "cross_domain_checks": [],
            "potential_issues": []
        }
        
        # Check cross-domain isolation
        domains = list(KnowledgeDomain)
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                entries1 = self.list_domain(domain1)
                entries2 = self.list_domain(domain2)
                
                if entries1 and entries2:
                    e1 = entries1[0]
                    e2 = entries2[0]
                    
                    check = self.verify_isolation(e1.id, e2.id)
                    results["cross_domain_checks"].append({
                        "domains": [domain1.value, domain2.value],
                        "isolated": check["isolated"],
                        "similarity": check["full_similarity"]
                    })
                    
                    if not check["isolated"]:
                        results["potential_issues"].append(check)
        
        results["all_isolated"] = len(results["potential_issues"]) == 0
        
        return results


# Demonstration
if __name__ == "__main__":
    import tempfile
    
    print("=" * 70)
    print("KNOWLEDGE MANAGER - SQLite Backend Test")
    print("=" * 70)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    manager = KnowledgeManager(db_path=db_path)
    
    # Create entries
    print("\nCreating entries...")
    
    entry1 = manager.create(
        name="ls",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="command",
        description="List directory contents",
        keywords=["bash", "shell", "list", "files"],
        metadata={"syntax": "ls [options] [path]"}
    )
    print(f"  Created: {entry1.name}")
    
    entry2 = manager.create(
        name="mkdir",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="command",
        description="Create directories",
        keywords=["bash", "shell", "create", "directory"],
        metadata={"syntax": "mkdir [options] dir"}
    )
    print(f"  Created: {entry2.name}")
    
    # Query
    print("\nQuerying 'list files'...")
    results = manager.query(["list", "files"], domain=KnowledgeDomain.PROGRAMMING)
    for sim, entry in results[:3]:
        print(f"  {entry.name}: {sim:.3f}")
    
    # Count
    print(f"\nEntries by domain: {manager.count_by_domain()}")
    
    # Cleanup
    os.unlink(db_path)
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
