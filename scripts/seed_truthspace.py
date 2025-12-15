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
from truthspace_lcm.core.encoder import PrimitiveType


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
    """Seed core intents (NL → command mappings)."""
    print("\n2. Seeding Intents...")
    
    intents = [
        # File operations
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
        {
            "name": "create_directory",
            "description": "Create a new directory",
            "keywords": ["create", "make", "directory", "folder", "mkdir"],
            "metadata": {
                "target_commands": ["mkdir -p"],
                "output_type": "bash",
                "triggers": ["create directory", "make folder", "mkdir"],
            }
        },
        {
            "name": "delete_file",
            "description": "Delete a file",
            "keywords": ["delete", "remove", "file", "rm"],
            "metadata": {
                "target_commands": ["rm"],
                "output_type": "bash",
                "triggers": ["delete file", "remove file", "rm"],
            }
        },
        {
            "name": "view_file",
            "description": "View file contents",
            "keywords": ["view", "show", "display", "contents", "cat", "read"],
            "metadata": {
                "target_commands": ["cat"],
                "output_type": "bash",
                "triggers": ["view file", "show contents", "cat", "read file"],
            }
        },
        {
            "name": "copy_file",
            "description": "Copy a file",
            "keywords": ["copy", "duplicate", "cp"],
            "metadata": {
                "target_commands": ["cp"],
                "output_type": "bash",
                "triggers": ["copy file", "duplicate", "cp"],
            }
        },
        {
            "name": "move_file",
            "description": "Move or rename a file",
            "keywords": ["move", "rename", "mv"],
            "metadata": {
                "target_commands": ["mv"],
                "output_type": "bash",
                "triggers": ["move file", "rename", "mv"],
            }
        },
        
        # Search operations
        {
            "name": "search_text",
            "description": "Search for text in files",
            "keywords": ["search", "find", "grep", "text", "pattern"],
            "metadata": {
                "target_commands": ["grep -r"],
                "output_type": "bash",
                "triggers": ["search for", "find text", "grep"],
            }
        },
        {
            "name": "find_files",
            "description": "Find files by name",
            "keywords": ["find", "locate", "files", "name"],
            "metadata": {
                "target_commands": ["find . -name"],
                "output_type": "bash",
                "triggers": ["find files", "locate", "find . -name"],
            }
        },
        
        # Network operations
        {
            "name": "show_network",
            "description": "Show network interfaces",
            "keywords": ["network", "interface", "ip", "show", "display"],
            "metadata": {
                "target_commands": ["ip addr"],
                "output_type": "bash",
                "triggers": ["show network", "network interfaces", "ip address"],
            }
        },
        {
            "name": "download_file",
            "description": "Download a file from URL",
            "keywords": ["download", "fetch", "curl", "wget", "url"],
            "metadata": {
                "target_commands": ["curl -O"],
                "output_type": "bash",
                "triggers": ["download", "fetch url", "curl", "wget"],
            }
        },
        
        # System operations
        {
            "name": "show_processes",
            "description": "Show running processes",
            "keywords": ["process", "processes", "running", "ps", "top"],
            "metadata": {
                "target_commands": ["ps aux"],
                "output_type": "bash",
                "triggers": ["show processes", "running processes", "ps"],
            }
        },
        {
            "name": "disk_usage",
            "description": "Show disk usage",
            "keywords": ["disk", "usage", "space", "df", "du"],
            "metadata": {
                "target_commands": ["df -h"],
                "output_type": "bash",
                "triggers": ["disk usage", "disk space", "df"],
            }
        },
        {
            "name": "system_info",
            "description": "Show system information",
            "keywords": ["system", "info", "uname", "uptime"],
            "metadata": {
                "target_commands": ["uname -a"],
                "output_type": "bash",
                "triggers": ["system info", "uname", "system information"],
            }
        },
        
        # Archive operations
        {
            "name": "compress_files",
            "description": "Compress files into archive",
            "keywords": ["compress", "archive", "tar", "zip", "gzip"],
            "metadata": {
                "target_commands": ["tar -czf"],
                "output_type": "bash",
                "triggers": ["compress", "archive", "tar", "zip"],
            }
        },
        {
            "name": "extract_archive",
            "description": "Extract files from archive",
            "keywords": ["extract", "unzip", "untar", "decompress"],
            "metadata": {
                "target_commands": ["tar -xzf"],
                "output_type": "bash",
                "triggers": ["extract", "unzip", "untar", "decompress"],
            }
        },
        
        # Permission operations
        {
            "name": "make_executable",
            "description": "Make file executable",
            "keywords": ["executable", "chmod", "permission", "run"],
            "metadata": {
                "target_commands": ["chmod +x"],
                "output_type": "bash",
                "triggers": ["make executable", "chmod +x", "permission"],
            }
        },
        
        # Python operations
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
    return count


def seed_commands(ts: TruthSpace):
    """Seed core command knowledge."""
    print("\n3. Seeding Commands...")
    
    commands = [
        {
            "name": "ls",
            "description": "List directory contents",
            "keywords": ["ls", "list", "files", "directory", "bash", "shell"],
            "metadata": {
                "command": "ls",
                "syntax": "ls [options] [path]",
                "output_type": "bash",
                "examples": ["ls -la", "ls -lh /tmp"],
            }
        },
        {
            "name": "cd",
            "description": "Change directory",
            "keywords": ["cd", "change", "directory", "navigate", "bash"],
            "metadata": {
                "command": "cd",
                "syntax": "cd [path]",
                "output_type": "bash",
            }
        },
        {
            "name": "cat",
            "description": "Concatenate and display file contents",
            "keywords": ["cat", "view", "display", "file", "contents", "bash"],
            "metadata": {
                "command": "cat",
                "syntax": "cat [file]",
                "output_type": "bash",
            }
        },
        {
            "name": "grep",
            "description": "Search for patterns in files",
            "keywords": ["grep", "search", "find", "pattern", "text", "bash"],
            "metadata": {
                "command": "grep",
                "syntax": "grep [options] pattern [file]",
                "output_type": "bash",
                "examples": ["grep -r 'pattern' .", "grep -i 'text' file.txt"],
            }
        },
        {
            "name": "find",
            "description": "Find files in directory hierarchy",
            "keywords": ["find", "search", "files", "locate", "bash"],
            "metadata": {
                "command": "find",
                "syntax": "find [path] [expression]",
                "output_type": "bash",
                "examples": ["find . -name '*.py'", "find /tmp -type f"],
            }
        },
    ]
    
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
