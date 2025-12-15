#!/usr/bin/env python3
"""
Migration Script: Hardcoded Intent Patterns → Database

Migrates the hardcoded intent_patterns from bash_generator.py and 
code_generator.py into Intent entries stored in the knowledge database.

This allows the generators to become pure resolvers that query the DB
instead of maintaining their own pattern dictionaries.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.intent_manager import IntentManager, StepType
from truthspace_lcm.core.knowledge_manager import KnowledgeDomain


# Bash intent patterns (from bash_generator.py)
BASH_INTENTS = [
    {
        "name": "create_directory",
        "description": "Create a new directory or folder",
        "triggers": [
            r"create\s+(?:a\s+)?(?:new\s+)?(?:directory|folder|dir)",
            r"make\s+(?:a\s+)?(?:new\s+)?(?:directory|folder|dir)",
            r"mkdir",
        ],
        "keywords": ["create", "make", "directory", "folder", "mkdir", "new"],
        "commands": ["mkdir"],
    },
    {
        "name": "create_file",
        "description": "Create a new empty file",
        "triggers": [
            r"create\s+(?:a\s+)?(?:new\s+)?(?:empty\s+)?file",
            r"touch\s+",
            r"make\s+(?:a\s+)?(?:new\s+)?file",
        ],
        "keywords": ["create", "make", "file", "touch", "new", "empty"],
        "commands": ["touch"],
    },
    {
        "name": "delete_file",
        "description": "Delete or remove files",
        "triggers": [
            r"delete\s+(?:the\s+)?(?:file|files)",
            r"remove\s+(?:the\s+)?(?:file|files)",
            r"rm\s+",
        ],
        "keywords": ["delete", "remove", "file", "rm", "erase"],
        "commands": ["rm"],
    },
    {
        "name": "delete_directory",
        "description": "Delete or remove a directory",
        "triggers": [
            r"delete\s+(?:the\s+)?(?:directory|folder|dir)",
            r"remove\s+(?:the\s+)?(?:directory|folder|dir)",
            r"rmdir",
        ],
        "keywords": ["delete", "remove", "directory", "folder", "rmdir"],
        "commands": ["rm -r", "rmdir"],
    },
    {
        "name": "copy_file",
        "description": "Copy files or directories",
        "triggers": [
            r"copy\s+(?:the\s+)?(?:file|files)",
            r"duplicate\s+",
            r"cp\s+",
        ],
        "keywords": ["copy", "duplicate", "cp", "file"],
        "commands": ["cp"],
    },
    {
        "name": "move_file",
        "description": "Move or rename files and directories",
        "triggers": [
            r"move\s+(?:the\s+)?(?:file|files|directory|folder)",
            r"rename\s+",
            r"mv\s+",
        ],
        "keywords": ["move", "rename", "mv", "file", "directory"],
        "commands": ["mv"],
    },
    {
        "name": "list_files",
        "description": "List files and directory contents",
        "triggers": [
            r"list\s+(?:all\s+)?(?:the\s+)?(?:files|contents|directory)",
            r"show\s+(?:all\s+)?(?:the\s+)?(?:files|contents)",
            r"^ls\s*",
            r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:directory|folder)",
        ],
        "keywords": ["list", "show", "files", "directory", "ls", "contents"],
        "commands": ["ls"],
    },
    {
        "name": "view_file",
        "description": "View or display file contents",
        "triggers": [
            r"(?:view|show|display|print|read)\s+(?:the\s+)?(?:contents?\s+of\s+)",
            r"contents?\s+of\s+",
            r"cat\s+",
            r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?file",
        ],
        "keywords": ["view", "show", "display", "read", "cat", "contents", "file"],
        "commands": ["cat"],
    },
    {
        "name": "search_text",
        "description": "Search for text patterns in files",
        "triggers": [
            r"search\s+for\s+['\"]",
            r"grep\s+",
            r"look\s+for\s+['\"]",
        ],
        "keywords": ["search", "grep", "find", "look", "pattern", "text"],
        "commands": ["grep"],
    },
    {
        "name": "download_file",
        "description": "Download files from URLs",
        "triggers": [
            r"download\s+",
            r"https?://",
            r"curl\s+",
            r"wget\s+",
        ],
        "keywords": ["download", "curl", "wget", "url", "http", "fetch"],
        "commands": ["curl", "wget"],
    },
    {
        "name": "change_permissions",
        "description": "Change file permissions",
        "triggers": [
            r"(?:make|set)\s+(?:it\s+)?executable",
            r"change\s+permissions",
            r"chmod\s+",
        ],
        "keywords": ["chmod", "permissions", "executable", "access", "rights"],
        "commands": ["chmod"],
    },
    {
        "name": "find_files",
        "description": "Find files by name or pattern",
        "triggers": [
            r"find\s+(?:all\s+)?\.\w+\s+files",
            r"find\s+(?:all\s+)?(?:the\s+)?files",
            r"locate\s+",
        ],
        "keywords": ["find", "locate", "search", "files", "pattern"],
        "commands": ["find"],
    },
    {
        "name": "compress_files",
        "description": "Compress files into archives",
        "triggers": [
            r"compress\s+",
            r"(?:create\s+)?(?:a\s+)?(?:tar|zip|archive)",
            r"pack\s+",
        ],
        "keywords": ["compress", "tar", "zip", "archive", "pack", "gzip"],
        "commands": ["tar", "zip", "gzip"],
    },
    {
        "name": "extract_files",
        "description": "Extract files from archives",
        "triggers": [
            r"extract\s+",
            r"unzip\s+",
            r"decompress\s+",
            r"unpack\s+",
        ],
        "keywords": ["extract", "unzip", "decompress", "unpack", "untar"],
        "commands": ["tar -x", "unzip", "gunzip"],
    },
    {
        "name": "show_network",
        "description": "Show network interfaces and IP addresses",
        "triggers": [
            r"(?:show|display|list|get)\s+(?:my\s+)?(?:network|ip|interface)",
            r"ifconfig",
            r"\bip\s+(?:addr|address|link|route)",
            r"network\s+(?:interface|config|status)",
            r"(?:check|view)\s+(?:my\s+)?(?:ip|network)",
        ],
        "keywords": ["network", "interface", "ip", "ifconfig", "address", "ethernet"],
        "commands": ["ip addr", "ifconfig"],
    },
    {
        "name": "show_processes",
        "description": "Show running processes",
        "triggers": [
            r"(?:list|show)\s+(?:running\s+)?process",
            r"\bps\b\s+",
            r"(?:check|view)\s+(?:running\s+)?process",
        ],
        "keywords": ["process", "ps", "running", "pid", "task"],
        "commands": ["ps aux", "ps"],
    },
    {
        "name": "show_system_logs",
        "description": "Show system or kernel logs",
        "triggers": [
            r"\bdmesg\b",
            r"(?:kernel|system)\s+(?:messages?|logs?|ring\s*buffer)",
            r"\bjournalctl\b",
            r"(?:show|view|display)\s+(?:kernel|system)\s+",
        ],
        "keywords": ["dmesg", "kernel", "system", "logs", "messages", "journalctl"],
        "commands": ["dmesg", "journalctl"],
    },
]

# Python intent patterns (from code_generator.py)
PYTHON_INTENTS = [
    {
        "name": "fetch_url",
        "description": "Fetch content from a URL using HTTP",
        "triggers": [
            r"fetch\s+(?:the\s+)?(?:url|webpage|page|website|html)",
            r"get\s+(?:the\s+)?(?:html|content|page)\s+(?:from|of)",
            r"download\s+(?:the\s+)?(?:page|html|content)",
            r"(?:get|fetch)\s+(?:the\s+)?webpage",
        ],
        "keywords": ["fetch", "url", "http", "get", "webpage", "download", "requests"],
        "commands": ["requests.get"],
    },
    {
        "name": "scrape_web",
        "description": "Scrape data from web pages",
        "triggers": [
            r"scrape\s+(?:the\s+)?(?:data|content|elements|titles|links)",
            r"extract\s+(?:the\s+)?(?:data|text|elements|titles|links)\s+from",
            r"parse\s+(?:the\s+)?html",
        ],
        "keywords": ["scrape", "extract", "parse", "html", "beautifulsoup", "web"],
        "commands": ["BeautifulSoup"],
    },
    {
        "name": "read_file_python",
        "description": "Read file contents in Python",
        "triggers": [
            r"read\s+(?:a\s+)?(?:the\s+)?(?:file|contents|data)",
            r"load\s+(?:a\s+)?(?:the\s+)?(?:file|data)",
            r"open\s+(?:and\s+read\s+)?(?:the\s+)?file",
        ],
        "keywords": ["read", "file", "open", "load", "contents"],
        "commands": ["open", "file.read"],
    },
    {
        "name": "write_file_python",
        "description": "Write data to a file in Python",
        "triggers": [
            r"write\s+(?:some\s+)?(?:data\s+)?(?:to\s+)?(?:a\s+)?(?:the\s+)?(?:file|json)",
            r"save\s+(?:the\s+)?(?:data|content|result)",
            r"output\s+to\s+(?:a\s+)?file",
        ],
        "keywords": ["write", "save", "file", "output", "store"],
        "commands": ["open", "file.write"],
    },
    {
        "name": "parse_json",
        "description": "Parse JSON data",
        "triggers": [
            r"parse\s+(?:the\s+)?json",
            r"read\s+(?:a\s+)?(?:the\s+)?json",
            r"load\s+(?:the\s+)?json",
        ],
        "keywords": ["parse", "json", "load", "decode"],
        "commands": ["json.loads", "json.load"],
    },
    {
        "name": "write_json",
        "description": "Write data as JSON",
        "triggers": [
            r"write\s+(?:to\s+)?(?:a\s+)?json",
            r"save\s+(?:to\s+)?(?:a\s+)?json",
            r"(?:to|into)\s+(?:a\s+)?json\s+file",
        ],
        "keywords": ["write", "json", "save", "dump", "encode"],
        "commands": ["json.dumps", "json.dump"],
    },
    {
        "name": "api_request",
        "description": "Make API requests",
        "triggers": [
            r"(?:make\s+)?(?:an?\s+)?api\s+(?:request|call)",
            r"call\s+(?:the\s+)?api",
            r"fetch\s+(?:from\s+)?(?:the\s+)?api",
        ],
        "keywords": ["api", "request", "call", "fetch", "rest", "endpoint"],
        "commands": ["requests.get", "requests.post"],
    },
    {
        "name": "loop_iterate",
        "description": "Loop or iterate over items",
        "triggers": [
            r"loop\s+(?:through|over)",
            r"iterate\s+(?:through|over)",
            r"for\s+each",
            r"\d+\s+to\s+\d+",
        ],
        "keywords": ["loop", "iterate", "for", "each", "range"],
        "commands": ["for", "range"],
    },
    {
        "name": "hello_world",
        "description": "Print hello world",
        "triggers": [
            r"hello\s+world",
            r"print\s+hello",
        ],
        "keywords": ["hello", "world", "print"],
        "commands": ["print"],
    },
]


def migrate_intents():
    """Migrate hardcoded intent patterns to database."""
    
    print("=" * 70)
    print("INTENT MIGRATION: Hardcoded Patterns → Database")
    print("=" * 70)
    
    manager = IntentManager()
    
    # Count existing intents
    existing = len(manager._intents)
    print(f"\nExisting intents in DB: {existing}")
    
    created = 0
    skipped = 0
    
    # Migrate bash intents
    print("\nMigrating Bash intents...")
    for intent_data in BASH_INTENTS:
        # Check if already exists
        existing_names = [i.name for i in manager._intents.values()]
        if intent_data["name"] in existing_names:
            print(f"  Skipped (exists): {intent_data['name']}")
            skipped += 1
            continue
        
        intent = manager.create_intent_for_command(
            command_name=intent_data["commands"][0],
            description=intent_data["description"],
            keywords=intent_data["keywords"],
            step_type=StepType.BASH,
        )
        # Override with our specific triggers
        intent.triggers = intent_data["triggers"]
        intent.name = intent_data["name"]
        intent.target_commands = intent_data["commands"]
        
        print(f"  Created: {intent_data['name']}")
        created += 1
    
    # Migrate python intents
    print("\nMigrating Python intents...")
    for intent_data in PYTHON_INTENTS:
        existing_names = [i.name for i in manager._intents.values()]
        if intent_data["name"] in existing_names:
            print(f"  Skipped (exists): {intent_data['name']}")
            skipped += 1
            continue
        
        intent = manager.create_intent_for_command(
            command_name=intent_data["commands"][0],
            description=intent_data["description"],
            keywords=intent_data["keywords"],
            step_type=StepType.PYTHON,
        )
        intent.triggers = intent_data["triggers"]
        intent.name = intent_data["name"]
        intent.target_commands = intent_data["commands"]
        
        print(f"  Created: {intent_data['name']}")
        created += 1
    
    print("\n" + "=" * 70)
    print(f"Migration complete!")
    print(f"  Created: {created}")
    print(f"  Skipped: {skipped}")
    print(f"  Total intents: {len(manager._intents)}")
    print("=" * 70)
    
    # Test a few queries
    print("\nTesting intent matching:")
    test_queries = [
        ("create a directory called test", StepType.BASH),
        ("show network interfaces", StepType.BASH),
        ("fetch https://api.github.com", StepType.PYTHON),
        ("parse the json response", StepType.PYTHON),
    ]
    
    for query, step_type in test_queries:
        result = manager.get_command_for_request(query, step_type)
        if result:
            cmd, conf, intent = result
            print(f"  '{query}' → {cmd} (conf: {conf:.2f})")
        else:
            print(f"  '{query}' → No match")


if __name__ == "__main__":
    migrate_intents()
