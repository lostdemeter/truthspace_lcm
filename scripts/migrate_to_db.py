#!/usr/bin/env python3
"""
Migration Script: JSON Files → SQLite Database

Migrates all knowledge entries from the file-based JSON storage
to the new SQLite database optimized for φ-based vectors.
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.knowledge_db import KnowledgeDB, KnowledgeDomain


def migrate_to_database():
    """Migrate all JSON knowledge files to SQLite database."""
    
    print("=" * 70)
    print("KNOWLEDGE MIGRATION: JSON Files → SQLite Database")
    print("=" * 70)
    
    # Paths
    base_dir = Path(__file__).parent.parent / "truthspace_lcm"
    json_storage = base_dir / "knowledge_store"
    db_path = base_dir / "knowledge.db"
    
    # Remove existing database if present
    if db_path.exists():
        print(f"\nRemoving existing database: {db_path}")
        os.unlink(db_path)
    
    # Create new database
    print(f"\nCreating new database: {db_path}")
    db = KnowledgeDB(str(db_path))
    
    # Import from JSON files
    print(f"\nImporting from: {json_storage}")
    imported = db.import_from_json_files(str(json_storage))
    
    print(f"\n✓ Imported {imported} entries")
    
    # Verify
    print("\nVerification:")
    counts = db.count_by_domain()
    for domain, count in counts.items():
        print(f"  {domain}: {count} entries")
    
    # Test queries
    print("\nTesting queries:")
    test_queries = [
        (["show", "network"], "display_network_ifconfig"),
        (["create", "directory"], "mkdir"),
        (["list", "files"], "ls"),
        (["delete", "file"], "rm"),
    ]
    
    for keywords, expected in test_queries:
        results = db.query(keywords, domain=KnowledgeDomain.PROGRAMMING, top_k=3)
        if results:
            best = results[0][1].name
            match = "✓" if expected in best or best in expected else "?"
            print(f"  {match} {keywords} → {best} (sim: {results[0][0]:.3f})")
        else:
            print(f"  ✗ {keywords} → No results")
    
    print("\n" + "=" * 70)
    print("Migration complete!")
    print(f"Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / 1024:.1f} KB")
    print("=" * 70)
    
    return db


if __name__ == "__main__":
    migrate_to_database()
