#!/usr/bin/env python3
"""
Migration Script: Re-encode all knowledge entries with φ-based positions.

This script:
1. Backs up the current knowledge store
2. Re-computes positions for all entries using the φ-encoder
3. Saves the updated entries

Run with: python scripts/migrate_to_phi.py
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain
from truthspace_lcm.core.phi_encoder import PhiEncoder


def migrate_knowledge_to_phi():
    """Migrate all knowledge entries to φ-based positions."""
    
    print("=" * 70)
    print("KNOWLEDGE MIGRATION: Hash-based → φ-based positions")
    print("=" * 70)
    
    # Initialize manager with φ-encoder
    manager = KnowledgeManager()
    
    print(f"\nLoaded {len(manager.entries)} entries")
    
    # Backup current knowledge store
    backup_dir = manager.storage_dir / "backups" / f"pre_phi_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating backup at: {backup_dir}")
    
    for domain in KnowledgeDomain:
        domain_dir = manager.storage_dir / domain.value
        if domain_dir.exists():
            backup_domain_dir = backup_dir / domain.value
            shutil.copytree(domain_dir, backup_domain_dir)
    
    print("Backup complete.\n")
    
    # Re-compute positions for all entries
    migrated = 0
    errors = 0
    
    for entry_id, entry in list(manager.entries.items()):
        try:
            # Recompute position using φ-encoder
            old_position = entry.position.copy()
            new_position = manager._compute_position(
                entry.name, 
                entry.domain, 
                entry.entry_type, 
                entry.keywords
            )
            
            # Update entry
            entry.position = new_position
            entry.updated_at = datetime.now().isoformat()
            
            # Save to disk
            manager._save_entry(entry)
            
            migrated += 1
            
            if migrated % 20 == 0:
                print(f"  Migrated {migrated} entries...")
                
        except Exception as e:
            print(f"  Error migrating {entry.name}: {e}")
            errors += 1
    
    print(f"\n{'=' * 70}")
    print(f"Migration complete!")
    print(f"  Migrated: {migrated} entries")
    print(f"  Errors: {errors}")
    print(f"  Backup: {backup_dir}")
    print(f"{'=' * 70}")
    
    # Verify migration
    print("\nVerification - testing semantic similarity:")
    
    # Test some queries
    test_queries = [
        (["show", "network"], "Should find network-related commands"),
        (["create", "directory"], "Should find mkdir"),
        (["delete", "file"], "Should find rm"),
        (["list", "files"], "Should find ls"),
    ]
    
    for keywords, description in test_queries:
        results = manager.query(keywords, domain=KnowledgeDomain.PROGRAMMING, top_k=3)
        if results:
            top_match = results[0][1].name
            top_sim = results[0][0]
            print(f"  {keywords} → {top_match} (sim: {top_sim:.3f})")
        else:
            print(f"  {keywords} → No results")
    
    return migrated, errors


if __name__ == "__main__":
    migrate_knowledge_to_phi()
