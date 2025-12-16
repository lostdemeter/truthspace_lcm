#!/usr/bin/env python3
"""
Seed TruthSpace with Bootstrap Knowledge

This script populates TruthSpace with the minimal knowledge required
for the system to function:

1. Primitives - Semantic anchors (CREATE, READ, FILE, etc.)
2. Core Intents - Basic NL → command mappings
3. Meta Knowledge - Stop words, patterns, etc.

Run this once on first setup or to reset to bootstrap state.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    KnowledgeDomain,
    EntryType,
)
from truthspace_lcm.core.encoder import PrimitiveType, PlasticEncoder


def seed_primitives(ts: TruthSpace):
    """Seed semantic primitives."""
    print("\n1. Seeding Primitives...")
    
    primitives = [
        # Actions (dims 0-3)
        {
            "name": "CREATE",
            "description": "Action: create, make, build something new",
            "keywords": ["create", "make", "new", "add", "build", "mkdir", "touch", "generate", "init"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 0,
                "level": 0,
                "opposite": "DESTROY",
            }
        },
        {
            "name": "DESTROY",
            "description": "Action: destroy, delete, remove something",
            "keywords": ["destroy", "delete", "remove", "rm", "rmdir", "kill", "terminate", "clear"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 0,
                "level": 1,
                "opposite": "CREATE",
            }
        },
        {
            "name": "READ",
            "description": "Action: read, view, display, show information",
            "keywords": ["read", "show", "display", "view", "print", "list", "cat", "get", "see", "look", "check"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 1,
                "level": 0,
                "opposite": "WRITE",
            }
        },
        {
            "name": "WRITE",
            "description": "Action: write, modify, edit, save data",
            "keywords": ["write", "modify", "edit", "change", "set", "update", "save", "store", "put"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 1,
                "level": 1,
                "opposite": "READ",
            }
        },
        {
            "name": "MOVE",
            "description": "Action: move, copy, transfer items",
            "keywords": ["move", "copy", "mv", "cp", "transfer", "send", "relocate"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 2,
                "level": 0,
            }
        },
        {
            "name": "CONNECT",
            "description": "Action: connect, link, establish connection",
            "keywords": ["connect", "link", "join", "attach", "ssh", "ping", "curl", "wget", "fetch"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 3,
                "level": 0,
                "opposite": "DISCONNECT",
            }
        },
        {
            "name": "SEARCH",
            "description": "Action: search, find, locate items",
            "keywords": ["search", "find", "locate", "grep", "look", "seek", "query", "filter"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 2,
                "level": 1,
            }
        },
        {
            "name": "EXECUTE",
            "description": "Action: run, execute, start a process",
            "keywords": ["run", "execute", "start", "launch", "invoke", "call", "spawn"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 3,
                "level": 1,
                "opposite": "STOP",
            }
        },
        {
            "name": "TRANSFORM",
            "description": "Action: transform, convert, parse data",
            "keywords": ["transform", "convert", "parse", "format", "encode", "decode", "compress", "extract"],
            "metadata": {
                "primitive_type": "action",
                "dimension": 2,
                "level": 2,
            }
        },
        
        # Domains (dims 4-6)
        {
            "name": "FILE",
            "description": "Domain: files, directories, filesystem",
            "keywords": ["file", "files", "directory", "folder", "dir", "path", "disk", "storage"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 4,
                "level": 0,
            }
        },
        {
            "name": "PROCESS",
            "description": "Domain: processes, jobs, tasks",
            "keywords": ["process", "processes", "pid", "job", "task", "daemon", "service", "thread"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 5,
                "level": 0,
            }
        },
        {
            "name": "NETWORK",
            "description": "Domain: network, interfaces, connections",
            "keywords": ["network", "interface", "ip", "ethernet", "wifi", "socket", "port", "connection"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 6,
                "level": 0,
            }
        },
        {
            "name": "SYSTEM",
            "description": "Domain: system, kernel, hardware",
            "keywords": ["system", "kernel", "os", "boot", "hardware", "memory", "cpu", "disk"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 4,
                "level": 1,
            }
        },
        {
            "name": "DATA",
            "description": "Domain: data, content, text",
            "keywords": ["data", "json", "xml", "csv", "text", "content", "output", "input"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 5,
                "level": 1,
            }
        },
        {
            "name": "USER",
            "description": "Domain: users, accounts, permissions",
            "keywords": ["user", "users", "account", "permission", "group", "owner", "root", "sudo"],
            "metadata": {
                "primitive_type": "domain",
                "dimension": 6,
                "level": 1,
            }
        },
        
        # Modifiers (dim 7)
        {
            "name": "ALL",
            "description": "Modifier: all, every, complete",
            "keywords": ["all", "every", "each", "entire", "whole", "complete", "full"],
            "metadata": {
                "primitive_type": "modifier",
                "dimension": 7,
                "level": 0,
            }
        },
        {
            "name": "RECURSIVE",
            "description": "Modifier: recursive, deep, nested",
            "keywords": ["recursive", "recursively", "-r", "-R", "deep", "nested"],
            "metadata": {
                "primitive_type": "modifier",
                "dimension": 7,
                "level": 1,
            }
        },
        {
            "name": "FORCE",
            "description": "Modifier: force, override",
            "keywords": ["force", "-f", "override", "overwrite", "ignore"],
            "metadata": {
                "primitive_type": "modifier",
                "dimension": 7,
                "level": 2,
            }
        },
        {
            "name": "VERBOSE",
            "description": "Modifier: verbose, detailed output",
            "keywords": ["verbose", "-v", "detailed", "debug"],
            "metadata": {
                "primitive_type": "modifier",
                "dimension": 7,
                "level": 3,
            }
        },
        
        # Relations (dims 8-11) - NEW in 12D
        # Dim 8: Temporal
        {
            "name": "BEFORE",
            "description": "Relation: before, prior, previous",
            "keywords": ["before", "prior", "previous", "earlier", "first", "initially"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 8,
                "level": 0,
                "opposite": "AFTER",
            }
        },
        {
            "name": "AFTER",
            "description": "Relation: after, then, next",
            "keywords": ["after", "then", "next", "later", "finally", "subsequently"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 8,
                "level": 1,
                "opposite": "BEFORE",
            }
        },
        {
            "name": "DURING",
            "description": "Relation: during, while, simultaneously",
            "keywords": ["during", "while", "when", "as", "simultaneously", "concurrent"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 8,
                "level": 2,
            }
        },
        
        # Dim 9: Causal
        {
            "name": "CAUSE",
            "description": "Relation: because, since, cause",
            "keywords": ["because", "since", "cause", "reason", "due", "therefore"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 9,
                "level": 0,
                "opposite": "EFFECT",
            }
        },
        {
            "name": "EFFECT",
            "description": "Relation: result, effect, outcome",
            "keywords": ["result", "effect", "outcome", "consequence", "leads", "produces"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 9,
                "level": 1,
                "opposite": "CAUSE",
            }
        },
        
        # Dim 10: Conditional
        {
            "name": "IF",
            "description": "Relation: if, when, provided",
            "keywords": ["if", "when", "provided", "assuming", "given", "unless", "retry", "try"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 10,
                "level": 0,
                "opposite": "ELSE",
            }
        },
        {
            "name": "ELSE",
            "description": "Relation: else, otherwise, fallback",
            "keywords": ["else", "otherwise", "alternatively", "instead", "fallback", "fail"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 10,
                "level": 1,
                "opposite": "IF",
            }
        },
        
        # Dim 11: Comparative
        {
            "name": "MORE",
            "description": "Relation: more, greater, increase",
            "keywords": ["more", "greater", "larger", "higher", "increase", "above", "bigger"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 11,
                "level": 0,
                "opposite": "LESS",
            }
        },
        {
            "name": "LESS",
            "description": "Relation: less, fewer, decrease",
            "keywords": ["less", "fewer", "smaller", "lower", "decrease", "below", "reduce"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 11,
                "level": 1,
                "opposite": "MORE",
            }
        },
        {
            "name": "EQUAL",
            "description": "Relation: equal, same, match",
            "keywords": ["equal", "same", "identical", "match", "equivalent", "compare"],
            "metadata": {
                "primitive_type": "relation",
                "dimension": 11,
                "level": 2,
            }
        },
    ]
    
    count = 0
    for p in primitives:
        try:
            ts.store(
                name=p["name"],
                entry_type=EntryType.PRIMITIVE,
                domain=KnowledgeDomain.GENERAL,
                description=p["description"],
                keywords=p["keywords"],
                metadata=p["metadata"],
            )
            count += 1
            print(f"   ✓ {p['name']}")
        except Exception as e:
            print(f"   ✗ {p['name']}: {e}")
    
    print(f"   Seeded {count} primitives")
    return count


def seed_intents(ts: TruthSpace):
    """Seed minimal core intents - most intents come from JSON files."""
    print("\n2. Seeding Core Intents...")
    
    # NOTE: Most intents are loaded from knowledge/*.json files
    # This only seeds the absolute minimum needed before JSON loading
    intents = [
        # Just hello_world as a Python example - bash intents come from JSON
        {
            "name": "hello_world",
            "description": "Print hello world in Python",
            "keywords": ["hello", "world", "python", "print", "program", "write"],
            "metadata": {
                "target_commands": ["print(\"Hello, World!\")"],
                "output_type": "python",
                "triggers": ["hello world", "print hello", "write hello world"],
            }
        },
        # Basic list files as fallback
        {
            "name": "list_files",
            "description": "List files in directory",
            "keywords": ["list", "files", "show", "directory", "contents", "ls"],
            "metadata": {
                "target_commands": ["ls -la"],
                "output_type": "bash",
                "triggers": ["list files", "show files", "what files", "ls"],
            }
        },
    ]
    
    count = 0
    for intent in intents:
        try:
            ts.store(
                name=intent["name"],
                entry_type=EntryType.INTENT,
                domain=KnowledgeDomain.PROGRAMMING,
                description=intent["description"],
                keywords=intent["keywords"],
                metadata=intent["metadata"],
            )
            count += 1
            print(f"   ✓ {intent['name']}")
        except Exception as e:
            print(f"   ✗ {intent['name']}: {e}")
    
    print(f"   Seeded {count} intents")
    print("   (Additional intents loaded from knowledge/*.json)")
    return count


def seed_commands(ts: TruthSpace):
    """Seed minimal commands - most come from JSON files."""
    print("\n3. Seeding Core Commands...")
    
    # NOTE: Most commands are loaded from knowledge/*.json files
    commands = []  # Empty - all commands come from JSON now
    
    if not commands:
        print("   (All commands loaded from knowledge/*.json)")
        return 0
    
    count = 0
    for cmd in commands:
        try:
            ts.store(
                name=cmd["name"],
                entry_type=EntryType.COMMAND,
                domain=KnowledgeDomain.PROGRAMMING,
                description=cmd["description"],
                keywords=cmd["keywords"],
                metadata=cmd["metadata"],
            )
            count += 1
            print(f"   ✓ {cmd['name']}")
        except Exception as e:
            print(f"   ✗ {cmd['name']}: {e}")
    
    print(f"   Seeded {count} commands")
    return count


def main():
    print("=" * 70)
    print("SEED TRUTHSPACE - Bootstrap Knowledge")
    print("=" * 70)
    
    # Initialize TruthSpace
    db_path = os.path.join(project_root, "truthspace_lcm", "truthspace.db")
    
    # Check if already seeded
    if os.path.exists(db_path):
        print(f"\nDatabase exists at: {db_path}")
        response = input("Reset and re-seed? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        os.unlink(db_path)
    
    ts = TruthSpace(db_path)
    
    # Seed knowledge
    primitives = seed_primitives(ts)
    intents = seed_intents(ts)
    commands = seed_commands(ts)
    
    # Summary
    print("\n" + "=" * 70)
    print("SEED COMPLETE")
    print("=" * 70)
    print(f"\nTotal knowledge seeded:")
    print(f"  Primitives: {primitives}")
    print(f"  Intents:    {intents}")
    print(f"  Commands:   {commands}")
    print(f"  TOTAL:      {primitives + intents + commands}")
    print(f"\nDatabase: {db_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
