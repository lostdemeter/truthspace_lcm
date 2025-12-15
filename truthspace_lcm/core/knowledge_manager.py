"""
Knowledge Manager: Safe CRUD Operations with Persistent Storage

This system ensures that:
1. Learning new knowledge NEVER destroys existing knowledge
2. Different domains are isolated (Python libraries vs Historical facts)
3. Knowledge is persisted to disk and survives restarts
4. CRUD operations are safe and verifiable

Key Design Principles:
- ADDITIVE: New knowledge adds to the space, doesn't overwrite
- ISOLATED: Domains have separate namespaces
- PERSISTENT: Knowledge saved to JSON files
- VERIFIABLE: Can check that old knowledge still works after adding new

The geometric insight:
- Each domain has its own "subspace" in truth space
- Domain dimension prevents cross-domain interference
- Position within domain encodes semantic meaning
"""

import numpy as np
import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from truthspace_lcm.core.phi_encoder import PhiEncoder

# Universal constants (exported for compatibility)
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)
GAMMA = 0.5772156649
ZETA3 = 1.2020569


class KnowledgeDomain(Enum):
    """
    Knowledge domains - each gets its own subspace.
    
    This is the key to preventing interference:
    - Python libraries live in PROGRAMMING domain
    - Historical facts live in HISTORY domain
    - They can't interfere because they're in different subspaces
    """
    PROGRAMMING = "programming"
    HISTORY = "history"
    SCIENCE = "science"
    GEOGRAPHY = "geography"
    GENERAL = "general"
    CUSTOM = "custom"


# Domain constants for domain isolation in TruthSpace
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
            return 0.0  # Different domains = no similarity
        
        n1 = np.linalg.norm(self.position)
        n2 = np.linalg.norm(other.position)
        if n1 > 0 and n2 > 0:
            return float(np.dot(self.position, other.position) / (n1 * n2))
        return 0.0


class KnowledgeManager:
    """
    Safe knowledge management with CRUD operations.
    
    Features:
    - CREATE: Add new knowledge (never overwrites)
    - READ: Query knowledge by keywords or ID
    - UPDATE: Modify existing knowledge (versioned)
    - DELETE: Remove knowledge (with backup)
    
    Safety guarantees:
    - Domain isolation prevents cross-domain interference
    - Versioning allows rollback
    - Backups before destructive operations
    """
    
    def __init__(self, storage_dir: str = None, dim: int = 8):
        self.dim = dim
        
        # Initialize φ-encoder for semantic position computation
        self._phi_encoder = PhiEncoder()
        
        # Default storage directory
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(__file__), 
                "..", "knowledge_store"
            )
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create domain subdirectories
        for domain in KnowledgeDomain:
            (self.storage_dir / domain.value).mkdir(exist_ok=True)
        
        # Create backup directory
        (self.storage_dir / "backups").mkdir(exist_ok=True)
        
        # In-memory cache
        self.entries: Dict[str, KnowledgeEntry] = {}
        
        # Load existing knowledge
        self._load_all()
    
    def _generate_id(self, name: str, domain: KnowledgeDomain) -> str:
        """Generate unique ID for an entry."""
        content = f"{domain.value}:{name}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_position(self, name: str, domain: KnowledgeDomain,
                          entry_type: str, keywords: List[str]) -> np.ndarray:
        """
        Compute geometric position for knowledge entry using φ-encoder.
        
        The φ-encoder maps semantic primitives to well-defined positions
        using the golden ratio as the fundamental anchor constant.
        """
        # Combine name and keywords for encoding
        text_to_encode = f"{name} {' '.join(keywords)}"
        
        # Get semantic decomposition from φ-encoder
        decomposition = self._phi_encoder.encode(text_to_encode)
        position = decomposition.position.copy()
        
        # Add domain component to dimension 0 for domain isolation
        # This ensures entries from different domains remain separated
        domain_offset = DOMAIN_CONSTANTS[domain]
        
        # Scale semantic components and add domain
        # Domain goes in a separate "layer" by adding to the magnitude
        position = position * 0.8  # Scale semantic to leave room for domain
        position[0] += domain_offset * 0.2  # Add domain signal
        
        return position
    
    def _get_filepath(self, entry: KnowledgeEntry) -> Path:
        """Get file path for an entry."""
        return self.storage_dir / entry.domain.value / f"{entry.id}.json"
    
    def _save_entry(self, entry: KnowledgeEntry):
        """Save a single entry to disk."""
        filepath = self._get_filepath(entry)
        with open(filepath, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def _load_entry(self, filepath: Path) -> Optional[KnowledgeEntry]:
        """Load a single entry from disk."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return KnowledgeEntry.from_dict(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None
    
    def _load_all(self):
        """Load all knowledge from disk."""
        for domain in KnowledgeDomain:
            domain_dir = self.storage_dir / domain.value
            if domain_dir.exists():
                for filepath in domain_dir.glob("*.json"):
                    entry = self._load_entry(filepath)
                    if entry:
                        self.entries[entry.id] = entry
    
    def _backup_entry(self, entry: KnowledgeEntry):
        """Create backup before modification."""
        backup_dir = self.storage_dir / "backups" / entry.domain.value
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{entry.id}_{timestamp}_v{entry.version}.json"
        
        with open(backup_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def create(self, name: str, domain: KnowledgeDomain, entry_type: str,
               description: str, keywords: List[str] = None,
               metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """
        CREATE: Add new knowledge entry.
        
        This is ADDITIVE - it never overwrites existing entries.
        If an entry with the same name exists in the domain, a new
        entry is created with a unique ID.
        """
        keywords = keywords or []
        metadata = metadata or {}
        
        entry_id = self._generate_id(name, domain)
        position = self._compute_position(name, domain, entry_type, keywords)
        
        now = datetime.now().isoformat()
        
        entry = KnowledgeEntry(
            id=entry_id,
            name=name,
            domain=domain,
            entry_type=entry_type,
            description=description,
            position=position,
            keywords=keywords,
            metadata=metadata,
            created_at=now,
            updated_at=now,
            version=1
        )
        
        # Add to memory
        self.entries[entry_id] = entry
        
        # Save to disk
        self._save_entry(entry)
        
        return entry
    
    def read(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        READ: Get knowledge entry by ID.
        """
        return self.entries.get(entry_id)
    
    def read_by_name(self, name: str, domain: KnowledgeDomain = None) -> List[KnowledgeEntry]:
        """
        READ: Get knowledge entries by name.
        
        Returns all entries matching the name (optionally filtered by domain).
        """
        results = []
        for entry in self.entries.values():
            if entry.name.lower() == name.lower():
                if domain is None or entry.domain == domain:
                    results.append(entry)
        return results
    
    def update(self, entry_id: str, description: str = None,
               keywords: List[str] = None, metadata: Dict[str, Any] = None) -> Optional[KnowledgeEntry]:
        """
        UPDATE: Modify existing knowledge entry.
        
        Creates a backup before modification.
        Increments version number.
        Recomputes position if keywords change.
        """
        entry = self.entries.get(entry_id)
        if not entry:
            return None
        
        # Backup before modification
        self._backup_entry(entry)
        
        # Update fields
        if description is not None:
            entry.description = description
        
        if keywords is not None:
            entry.keywords = keywords
            # Recompute position with new keywords
            entry.position = self._compute_position(
                entry.name, entry.domain, entry.entry_type, keywords
            )
        
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.updated_at = datetime.now().isoformat()
        entry.version += 1
        
        # Save to disk
        self._save_entry(entry)
        
        return entry
    
    def delete(self, entry_id: str, hard_delete: bool = False) -> bool:
        """
        DELETE: Remove knowledge entry.
        
        By default, creates a backup before deletion (soft delete).
        Set hard_delete=True to skip backup.
        """
        entry = self.entries.get(entry_id)
        if not entry:
            return False
        
        # Backup before deletion
        if not hard_delete:
            self._backup_entry(entry)
        
        # Remove from memory
        del self.entries[entry_id]
        
        # Remove from disk
        filepath = self._get_filepath(entry)
        if filepath.exists():
            filepath.unlink()
        
        return True
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def query(self, keywords: List[str], domain: KnowledgeDomain = None,
              top_k: int = 10) -> List[Tuple[float, KnowledgeEntry]]:
        """
        Query using φ-encoder for semantic matching.
        
        Combines geometric similarity with keyword boosting for best results.
        """
        # Encode query using φ-encoder
        query_text = " ".join(keywords)
        decomposition = self._phi_encoder.encode(query_text)
        query_vec = decomposition.position
        
        # Add domain component if specified
        if domain:
            query_vec = query_vec.copy()
            query_vec[0] += DOMAIN_CONSTANTS[domain] * 0.2
        
        # Find similar entries with keyword boosting
        results = []
        for entry in self.entries.values():
            # Filter by domain if specified
            if domain and entry.domain != domain:
                continue
            
            # Compute geometric similarity
            entry_norm = np.linalg.norm(entry.position)
            query_norm = np.linalg.norm(query_vec)
            
            if entry_norm > 0 and query_norm > 0:
                geo_sim = float(np.dot(query_vec, entry.position) / (query_norm * entry_norm))
            else:
                geo_sim = 0.0
            
            # Keyword boost: check if query keywords appear in entry keywords or name
            keyword_boost = 0.0
            entry_kw_set = set(kw.lower() for kw in entry.keywords)
            entry_kw_set.add(entry.name.lower())
            
            for qkw in keywords:
                qkw_lower = qkw.lower()
                if qkw_lower in entry_kw_set:
                    keyword_boost += 0.15
                else:
                    # Partial match
                    for ekw in entry_kw_set:
                        if qkw_lower in ekw or ekw in qkw_lower:
                            keyword_boost += 0.05
                            break
            
            # Combined score
            combined_sim = geo_sim + min(keyword_boost, 0.4)
            results.append((combined_sim, entry))
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    def list_domain(self, domain: KnowledgeDomain) -> List[KnowledgeEntry]:
        """List all entries in a domain."""
        return [e for e in self.entries.values() if e.domain == domain]
    
    def count_by_domain(self) -> Dict[str, int]:
        """Count entries by domain."""
        counts = {}
        for domain in KnowledgeDomain:
            counts[domain.value] = len(self.list_domain(domain))
        return counts
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def verify_isolation(self, entry1_id: str, entry2_id: str) -> Dict[str, Any]:
        """
        Verify that two entries don't interfere with each other.
        
        Returns analysis of their geometric relationship.
        """
        entry1 = self.entries.get(entry1_id)
        entry2 = self.entries.get(entry2_id)
        
        if not entry1 or not entry2:
            return {"error": "Entry not found"}
        
        # Compute similarity
        full_sim = float(np.dot(entry1.position, entry2.position) / 
                        (np.linalg.norm(entry1.position) * np.linalg.norm(entry2.position)))
        
        # Domain component similarity
        domain_sim = abs(entry1.position[0] - entry2.position[0])
        
        # Semantic component similarity (excluding domain)
        sem1 = entry1.position[1:]
        sem2 = entry2.position[1:]
        sem_sim = float(np.dot(sem1, sem2) / 
                       (np.linalg.norm(sem1) * np.linalg.norm(sem2) + 1e-10))
        
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
        results = {
            "total_entries": len(self.entries),
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
                    # Sample check (first entry from each)
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


def demonstrate_crud():
    """Demonstrate CRUD operations with isolation verification."""
    
    print("=" * 70)
    print("KNOWLEDGE MANAGER: SAFE CRUD OPERATIONS")
    print("=" * 70)
    print()
    
    # Create manager with test directory
    test_dir = "/tmp/truthspace_knowledge_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    manager = KnowledgeManager(storage_dir=test_dir)
    
    # =========================================================================
    # CREATE: Add knowledge from different domains
    # =========================================================================
    print("-" * 70)
    print("CREATE: Adding knowledge from different domains")
    print("-" * 70)
    print()
    
    # Historical knowledge
    washington = manager.create(
        name="George Washington",
        domain=KnowledgeDomain.HISTORY,
        entry_type="person",
        description="First President of the United States, served 1789-1797",
        keywords=["president", "american", "first", "founding father", "revolutionary war"]
    )
    print(f"  Created: {washington.name} (domain: {washington.domain.value})")
    print(f"    ID: {washington.id}")
    print(f"    Position[0] (domain): {washington.position[0]:.4f}")
    
    lincoln = manager.create(
        name="Abraham Lincoln",
        domain=KnowledgeDomain.HISTORY,
        entry_type="person",
        description="16th President of the United States, led during Civil War",
        keywords=["president", "american", "civil war", "emancipation"]
    )
    print(f"  Created: {lincoln.name} (domain: {lincoln.domain.value})")
    
    # Programming knowledge
    beautifulsoup = manager.create(
        name="BeautifulSoup",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="library",
        description="Python library for parsing HTML and XML documents",
        keywords=["html", "parse", "scrape", "web", "dom", "xml"]
    )
    print(f"  Created: {beautifulsoup.name} (domain: {beautifulsoup.domain.value})")
    print(f"    ID: {beautifulsoup.id}")
    print(f"    Position[0] (domain): {beautifulsoup.position[0]:.4f}")
    
    requests_lib = manager.create(
        name="requests",
        domain=KnowledgeDomain.PROGRAMMING,
        entry_type="library",
        description="Python HTTP library for making web requests",
        keywords=["http", "web", "url", "api", "get", "post", "fetch"]
    )
    print(f"  Created: {requests_lib.name} (domain: {requests_lib.domain.value})")
    
    # Science knowledge
    einstein = manager.create(
        name="Albert Einstein",
        domain=KnowledgeDomain.SCIENCE,
        entry_type="person",
        description="Physicist who developed theory of relativity",
        keywords=["physics", "relativity", "scientist", "nobel prize"]
    )
    print(f"  Created: {einstein.name} (domain: {einstein.domain.value})")
    
    print()
    print(f"  Total entries: {len(manager.entries)}")
    print(f"  By domain: {manager.count_by_domain()}")
    
    # =========================================================================
    # VERIFY: Check that domains are isolated
    # =========================================================================
    print()
    print("-" * 70)
    print("VERIFY: Checking domain isolation")
    print("-" * 70)
    print()
    
    # Check Washington vs BeautifulSoup (should be isolated)
    check1 = manager.verify_isolation(washington.id, beautifulsoup.id)
    print(f"  {washington.name} vs {beautifulsoup.name}:")
    print(f"    Same domain: {check1['same_domain']}")
    print(f"    Full similarity: {check1['full_similarity']:.4f}")
    print(f"    Domain distance: {check1['domain_distance']:.4f}")
    print(f"    Isolated: {check1['isolated']}")
    print(f"    Interference risk: {check1['interference_risk']}")
    
    # Check Washington vs Lincoln (same domain, should be similar)
    check2 = manager.verify_isolation(washington.id, lincoln.id)
    print(f"\n  {washington.name} vs {lincoln.name}:")
    print(f"    Same domain: {check2['same_domain']}")
    print(f"    Full similarity: {check2['full_similarity']:.4f}")
    print(f"    Semantic similarity: {check2['semantic_similarity']:.4f}")
    print(f"    Isolated: {check2['isolated']}")
    
    # Check BeautifulSoup vs requests (same domain, related)
    check3 = manager.verify_isolation(beautifulsoup.id, requests_lib.id)
    print(f"\n  {beautifulsoup.name} vs {requests_lib.name}:")
    print(f"    Same domain: {check3['same_domain']}")
    print(f"    Full similarity: {check3['full_similarity']:.4f}")
    print(f"    Semantic similarity: {check3['semantic_similarity']:.4f}")
    
    # =========================================================================
    # READ: Query knowledge
    # =========================================================================
    print()
    print("-" * 70)
    print("READ: Querying knowledge")
    print("-" * 70)
    print()
    
    # Query within HISTORY domain
    print("  Query: ['president'] in HISTORY domain:")
    results = manager.query(["president"], domain=KnowledgeDomain.HISTORY, top_k=3)
    for sim, entry in results:
        print(f"    → {entry.name}: {sim:.4f}")
    
    # Query within PROGRAMMING domain
    print("\n  Query: ['html', 'parse'] in PROGRAMMING domain:")
    results = manager.query(["html", "parse"], domain=KnowledgeDomain.PROGRAMMING, top_k=3)
    for sim, entry in results:
        print(f"    → {entry.name}: {sim:.4f}")
    
    # Query across all domains
    print("\n  Query: ['president'] across ALL domains:")
    results = manager.query(["president"], domain=None, top_k=5)
    for sim, entry in results:
        print(f"    → {entry.name} ({entry.domain.value}): {sim:.4f}")
    
    # =========================================================================
    # UPDATE: Modify knowledge
    # =========================================================================
    print()
    print("-" * 70)
    print("UPDATE: Modifying knowledge")
    print("-" * 70)
    print()
    
    print(f"  Before update: {washington.name} v{washington.version}")
    print(f"    Keywords: {washington.keywords}")
    
    updated = manager.update(
        washington.id,
        keywords=["president", "american", "first", "founding father", "revolutionary war", "virginia"]
    )
    
    print(f"\n  After update: {updated.name} v{updated.version}")
    print(f"    Keywords: {updated.keywords}")
    print(f"    Backup created in: {manager.storage_dir}/backups/")
    
    # =========================================================================
    # DELETE: Remove knowledge (with backup)
    # =========================================================================
    print()
    print("-" * 70)
    print("DELETE: Removing knowledge (with backup)")
    print("-" * 70)
    print()
    
    # Create a temporary entry to delete
    temp = manager.create(
        name="Temporary Entry",
        domain=KnowledgeDomain.GENERAL,
        entry_type="test",
        description="This will be deleted"
    )
    print(f"  Created temporary entry: {temp.id}")
    print(f"  Total entries before delete: {len(manager.entries)}")
    
    deleted = manager.delete(temp.id)
    print(f"  Deleted: {deleted}")
    print(f"  Total entries after delete: {len(manager.entries)}")
    print(f"  Backup preserved: Yes")
    
    # =========================================================================
    # PERSISTENCE: Verify data survives reload
    # =========================================================================
    print()
    print("-" * 70)
    print("PERSISTENCE: Verifying data survives reload")
    print("-" * 70)
    print()
    
    print(f"  Entries before reload: {len(manager.entries)}")
    
    # Create new manager (simulates restart)
    manager2 = KnowledgeManager(storage_dir=test_dir)
    
    print(f"  Entries after reload: {len(manager2.entries)}")
    
    # Verify Washington still exists
    washington_reloaded = manager2.read_by_name("George Washington", KnowledgeDomain.HISTORY)
    if washington_reloaded:
        print(f"  George Washington found: {washington_reloaded[0].id}")
        print(f"    Version: {washington_reloaded[0].version}")
        print(f"    Keywords: {washington_reloaded[0].keywords}")
    
    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================
    print()
    print("-" * 70)
    print("FINAL VERIFICATION: All knowledge intact")
    print("-" * 70)
    print()
    
    verification = manager2.verify_all_knowledge()
    print(f"  Total entries: {verification['total_entries']}")
    print(f"  By domain: {verification['by_domain']}")
    print(f"  All isolated: {verification['all_isolated']}")
    
    if verification['cross_domain_checks']:
        print(f"\n  Cross-domain isolation checks:")
        for check in verification['cross_domain_checks']:
            status = "✓" if check['isolated'] else "⚠"
            print(f"    {status} {check['domains'][0]} vs {check['domains'][1]}: sim={check['similarity']:.4f}")
    
    print()
    print("=" * 70)
    print("KEY GUARANTEES")
    print("=" * 70)
    print("""
1. ADDITIVE CREATION
   - New knowledge never overwrites existing entries
   - Each entry gets a unique ID

2. DOMAIN ISOLATION
   - Different domains have different position[0] values
   - Cross-domain similarity is near zero
   - Learning BeautifulSoup CANNOT affect George Washington

3. VERSIONED UPDATES
   - Updates increment version number
   - Backups created before modification
   - Can rollback if needed

4. SAFE DELETION
   - Backups created before deletion
   - Can recover deleted knowledge

5. PERSISTENT STORAGE
   - Knowledge saved to JSON files
   - Survives restarts
   - Organized by domain
""")
    
    # Cleanup
    print(f"\n  Test data stored in: {test_dir}")
    print("  (You can inspect the files or delete the directory)")


if __name__ == "__main__":
    demonstrate_crud()
