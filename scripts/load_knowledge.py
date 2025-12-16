#!/usr/bin/env python3
"""
Load Knowledge from JSON Files

This script loads knowledge from JSON files in the knowledge/ directory
into the TruthSpace database.

Usage:
    python scripts/load_knowledge.py                    # Load all JSON files
    python scripts/load_knowledge.py bash_commands.json # Load specific file
    python scripts/load_knowledge.py --reset            # Reset DB first, then load all
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    EntryType,
    KnowledgeDomain,
)


# Map string types to EntryType enum
TYPE_MAP = {
    "PRIMITIVE": EntryType.PRIMITIVE,
    "INTENT": EntryType.INTENT,
    "COMMAND": EntryType.COMMAND,
    "PATTERN": EntryType.PATTERN,
    "CONCEPT": EntryType.CONCEPT,
    "META": EntryType.META,
}

# Map string domains to KnowledgeDomain enum
DOMAIN_MAP = {
    "PROGRAMMING": KnowledgeDomain.PROGRAMMING,
    "SYSTEM": KnowledgeDomain.SYSTEM,
    "GENERAL": KnowledgeDomain.GENERAL,
}


def load_json_file(filepath: Path) -> dict:
    """Load and parse a JSON knowledge file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_knowledge_file(ts: TruthSpace, filepath: Path) -> int:
    """
    Load a single knowledge JSON file into TruthSpace.
    
    Returns the number of entries loaded.
    """
    print(f"\nLoading: {filepath.name}")
    print("-" * 40)
    
    data = load_json_file(filepath)
    metadata = data.get("metadata", {})
    entries = data.get("entries", [])
    
    # Get default domain from metadata
    default_domain = DOMAIN_MAP.get(
        metadata.get("domain", "PROGRAMMING"),
        KnowledgeDomain.PROGRAMMING
    )
    
    count = 0
    for entry in entries:
        try:
            # Get entry type
            entry_type = TYPE_MAP.get(
                entry.get("type", "COMMAND"),
                EntryType.COMMAND
            )
            
            # Build metadata dict from entry fields
            entry_metadata = {}
            
            # Copy relevant fields to metadata
            if "output_type" in entry:
                entry_metadata["output_type"] = entry["output_type"]
            if "syntax" in entry:
                entry_metadata["syntax"] = entry["syntax"]
            if "examples" in entry:
                entry_metadata["examples"] = entry["examples"]
            if "flags" in entry:
                entry_metadata["flags"] = entry["flags"]
            if "triggers" in entry:
                entry_metadata["triggers"] = entry["triggers"]
            if "extraction_patterns" in entry:
                entry_metadata["extraction_patterns"] = entry["extraction_patterns"]
            if "validation" in entry:
                entry_metadata["validation"] = entry["validation"]
            if "default" in entry:
                entry_metadata["default"] = entry["default"]
            
            # Set target_commands based on entry type
            if "target_commands" in entry:
                # Explicit target_commands (for intents)
                entry_metadata["target_commands"] = entry["target_commands"]
            elif entry_type == EntryType.COMMAND:
                # For COMMAND type, use syntax or name
                entry_metadata["target_commands"] = [entry.get("syntax", entry["name"])]
            
            # Store in TruthSpace
            ts.store(
                name=entry["name"],
                entry_type=entry_type,
                domain=default_domain,
                description=entry.get("description", ""),
                keywords=entry.get("keywords", []),
                metadata=entry_metadata,
            )
            
            count += 1
            print(f"  ✓ {entry['name']}")
            
        except Exception as e:
            print(f"  ✗ {entry.get('name', 'unknown')}: {e}")
    
    print(f"\nLoaded {count}/{len(entries)} entries from {filepath.name}")
    return count


def load_all_knowledge(ts: TruthSpace, knowledge_dir: Path) -> int:
    """Load all JSON files from the knowledge directory."""
    total = 0
    
    json_files = sorted(knowledge_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {knowledge_dir}")
        return 0
    
    print(f"Found {len(json_files)} knowledge file(s)")
    
    for filepath in json_files:
        total += load_knowledge_file(ts, filepath)
    
    return total


def main():
    # Parse arguments
    reset = "--reset" in sys.argv
    specific_files = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    
    # Paths
    project_root = Path(__file__).parent.parent
    knowledge_dir = project_root / "knowledge"
    
    print("=" * 60)
    print("LOAD KNOWLEDGE - JSON to TruthSpace")
    print("=" * 60)
    
    # Initialize TruthSpace
    ts = TruthSpace()
    
    # Reset if requested
    if reset:
        print("\nResetting database...")
        # Run seed script first
        import subprocess
        subprocess.run([
            sys.executable,
            str(project_root / "scripts" / "seed_truthspace.py")
        ], check=True)
        # Reinitialize after reset
        ts = TruthSpace()
    
    # Load knowledge
    if specific_files:
        total = 0
        for filename in specific_files:
            filepath = knowledge_dir / filename
            if filepath.exists():
                total += load_knowledge_file(ts, filepath)
            else:
                print(f"File not found: {filepath}")
    else:
        total = load_all_knowledge(ts, knowledge_dir)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE - Loaded {total} entries")
    print("=" * 60)


if __name__ == "__main__":
    main()
