"""
Natural Language to Bash Command Translator

Experiment to test the hyperdimensional paradigm with a practical use case.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  "list files"  ──►  [NL Structure]  ──►  [Bash Structure]  ──►  ls │
│                          │                     │                    │
│                     NL Transcoder         Bash Transcoder           │
│                     (stateless)           (stateless)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

The transcoders are STATELESS - they just define how to encode/decode.
All state lives in the structures.

Dimensions for NL Structure:
- Action type (query, modify, create, delete)
- Target type (file, process, network, system)
- Scope (single, multiple, recursive)
- Verbosity (quiet, normal, verbose)

Dimensions for Bash Structure:
- Command category (file, process, network, system)
- Flags (common patterns)
- Arguments (path, pid, etc.)

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from datetime import datetime

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


# =============================================================================
# STATELESS TRANSCODERS
# =============================================================================

class NLTranscoder:
    """
    Stateless transcoder for natural language.
    
    Like a shader - pure function, no state.
    All it does is define how to map NL → position and position → NL.
    """
    
    # Semantic dimensions for NL
    DIMENSIONS = {
        'action': {
            'query': np.array([1, 0, 0, 0]),      # list, show, display, find
            'modify': np.array([0, 1, 0, 0]),     # change, edit, update
            'create': np.array([0, 0, 1, 0]),     # make, create, new
            'delete': np.array([0, 0, 0, 1]),     # remove, delete, kill
        },
        'target': {
            'file': np.array([1, 0, 0, 0]),
            'process': np.array([0, 1, 0, 0]),
            'network': np.array([0, 0, 1, 0]),
            'system': np.array([0, 0, 0, 1]),
        },
        'scope': {
            'single': np.array([1, 0]),
            'multiple': np.array([0.5, 0.5]),
            'recursive': np.array([0, 1]),
        },
        'verbosity': {
            'quiet': np.array([0]),
            'normal': np.array([0.5]),
            'verbose': np.array([1]),
        },
    }
    
    # Word → dimension mappings
    ACTION_WORDS = {
        'list': 'query', 'show': 'query', 'display': 'query', 'find': 'query',
        'get': 'query', 'what': 'query', 'which': 'query', 'search': 'query',
        'change': 'modify', 'edit': 'modify', 'update': 'modify', 'set': 'modify',
        'rename': 'modify', 'move': 'modify', 'chmod': 'modify',
        'create': 'create', 'make': 'create', 'new': 'create', 'touch': 'create',
        'mkdir': 'create', 'write': 'create',
        'delete': 'delete', 'remove': 'delete', 'kill': 'delete', 'rm': 'delete',
        'stop': 'delete', 'terminate': 'delete',
    }
    
    TARGET_WORDS = {
        'file': 'file', 'files': 'file', 'directory': 'file', 'directories': 'file',
        'folder': 'file', 'folders': 'file', 'path': 'file', 'contents': 'file',
        'line': 'file', 'lines': 'file', 'word': 'file', 'words': 'file',
        'process': 'process', 'processes': 'process', 'pid': 'process',
        'running': 'process', 'task': 'process', 'tasks': 'process', 'job': 'process',
        'jobs': 'process', 'program': 'process', 'programs': 'process',
        'network': 'network', 'connection': 'network', 'connections': 'network',
        'port': 'network', 'ports': 'network', 'ip': 'network', 'socket': 'network',
        'address': 'network', 'interface': 'network', 'interfaces': 'network',
        'system': 'system', 'memory': 'system', 'disk': 'system', 'cpu': 'system',
        'usage': 'system', 'space': 'system', 'uptime': 'system', 'ram': 'system',
        'storage': 'system', 'info': 'system',
    }
    
    SCOPE_WORDS = {
        'all': 'multiple', 'every': 'multiple', 'each': 'multiple',
        'recursive': 'recursive', 'recursively': 'recursive', 'tree': 'recursive',
        'subdirectories': 'recursive', 'nested': 'recursive',
    }
    
    VERBOSITY_WORDS = {
        'quiet': 'quiet', 'silent': 'quiet', 'brief': 'quiet',
        'detailed': 'verbose', 'verbose': 'verbose', 'long': 'verbose',
        'full': 'verbose', 'all': 'verbose',
    }
    
    @classmethod
    def encode(cls, text: str, dims: int) -> np.ndarray:
        """
        Encode natural language text to position.
        
        Stateless - no instance state used.
        """
        words = text.lower().split()
        
        # Detect dimensions
        action = 'query'  # default
        target = 'file'   # default
        scope = 'single'  # default
        verbosity = 'normal'  # default
        
        for word in words:
            if word in cls.ACTION_WORDS:
                action = cls.ACTION_WORDS[word]
            if word in cls.TARGET_WORDS:
                target = cls.TARGET_WORDS[word]
            if word in cls.SCOPE_WORDS:
                scope = cls.SCOPE_WORDS[word]
            if word in cls.VERBOSITY_WORDS:
                verbosity = cls.VERBOSITY_WORDS[word]
        
        # Build position vector
        position = np.concatenate([
            cls.DIMENSIONS['action'][action],
            cls.DIMENSIONS['target'][target],
            cls.DIMENSIONS['scope'][scope],
            cls.DIMENSIONS['verbosity'][verbosity],
        ])
        
        # Pad or truncate to match dims
        if len(position) < dims:
            position = np.concatenate([position, np.zeros(dims - len(position))])
        elif len(position) > dims:
            position = position[:dims]
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 1e-10:
            position = position / norm * CRITICAL_LINE
        
        return position
    
    @classmethod
    def decode(cls, nodes: List[Tuple[Node, float]]) -> str:
        """
        Decode matched nodes back to natural language description.
        
        Stateless - just extracts from node data.
        """
        if not nodes:
            return ""
        
        best_node, _ = nodes[0]
        if best_node.data and 'nl' in best_node.data:
            return best_node.data['nl']
        return ""


class BashTranscoder:
    """
    Stateless transcoder for bash commands.
    
    Maps bash commands to/from positions.
    """
    
    # Command categories
    CATEGORIES = {
        'file_query': np.array([1, 0, 0, 0, 0, 0, 0, 0]),
        'file_modify': np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        'file_create': np.array([0, 0, 1, 0, 0, 0, 0, 0]),
        'file_delete': np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        'process_query': np.array([0, 0, 0, 0, 1, 0, 0, 0]),
        'process_kill': np.array([0, 0, 0, 0, 0, 1, 0, 0]),
        'network_query': np.array([0, 0, 0, 0, 0, 0, 1, 0]),
        'system_query': np.array([0, 0, 0, 0, 0, 0, 0, 1]),
    }
    
    # Command → category mapping
    COMMAND_CATEGORIES = {
        'ls': 'file_query',
        'find': 'file_query',
        'cat': 'file_query',
        'head': 'file_query',
        'tail': 'file_query',
        'less': 'file_query',
        'wc': 'file_query',
        'du': 'file_query',
        'mv': 'file_modify',
        'cp': 'file_modify',
        'chmod': 'file_modify',
        'chown': 'file_modify',
        'touch': 'file_create',
        'mkdir': 'file_create',
        'rm': 'file_delete',
        'rmdir': 'file_delete',
        'ps': 'process_query',
        'top': 'process_query',
        'htop': 'process_query',
        'pgrep': 'process_query',
        'kill': 'process_kill',
        'pkill': 'process_kill',
        'killall': 'process_kill',
        'netstat': 'network_query',
        'ss': 'network_query',
        'lsof': 'network_query',
        'ifconfig': 'network_query',
        'ip': 'network_query',
        'df': 'system_query',
        'free': 'system_query',
        'uptime': 'system_query',
        'uname': 'system_query',
    }
    
    @classmethod
    def encode(cls, command: str, dims: int) -> np.ndarray:
        """
        Encode bash command to position.
        
        Stateless.
        """
        # Extract base command
        parts = command.strip().split()
        base_cmd = parts[0] if parts else ''
        
        # Get category
        category = cls.COMMAND_CATEGORIES.get(base_cmd, 'file_query')
        position = cls.CATEGORIES[category].copy()
        
        # Add flag information
        flags = [p for p in parts[1:] if p.startswith('-')]
        
        # Common flag patterns
        flag_features = np.zeros(4)
        for flag in flags:
            if 'r' in flag or 'R' in flag:  # recursive
                flag_features[0] = 1
            if 'l' in flag:  # long/detailed
                flag_features[1] = 1
            if 'a' in flag:  # all
                flag_features[2] = 1
            if 'v' in flag:  # verbose
                flag_features[3] = 1
        
        position = np.concatenate([position, flag_features])
        
        # Pad or truncate
        if len(position) < dims:
            position = np.concatenate([position, np.zeros(dims - len(position))])
        elif len(position) > dims:
            position = position[:dims]
        
        # Normalize
        norm = np.linalg.norm(position)
        if norm > 1e-10:
            position = position / norm * CRITICAL_LINE
        
        return position
    
    @classmethod
    def decode(cls, nodes: List[Tuple[Node, float]]) -> str:
        """
        Decode matched nodes to bash command.
        
        Stateless.
        """
        if not nodes:
            return ""
        
        best_node, _ = nodes[0]
        if best_node.data and 'bash' in best_node.data:
            return best_node.data['bash']
        return ""


# =============================================================================
# THE TRANSLATOR
# =============================================================================

class NLToBashTranslator:
    """
    Translates natural language to bash commands.
    
    Uses two structures connected by stateless transcoders:
    
    NL Text → [NL Transcoder] → NL Structure → Bash Structure → [Bash Transcoder] → Bash Command
    
    The structures hold the learned mappings.
    The transcoders are stateless encoding/decoding functions.
    """
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        
        # Two structures - one for NL, one for Bash
        # They share the same dimensional space for direct mapping
        self.nl_structure = HyperdimensionalStructure(dims=dims, name="nl_space")
        self.bash_structure = HyperdimensionalStructure(dims=dims, name="bash_space")
    
    def add_mapping(self, nl_text: str, bash_command: str) -> Tuple[Node, Node]:
        """
        Add a NL → Bash mapping.
        
        Creates nodes in both structures at corresponding positions.
        """
        # Encode both
        nl_position = NLTranscoder.encode(nl_text, self.dims)
        bash_position = BashTranscoder.encode(bash_command, self.dims)
        
        # Average the positions for shared space
        # This creates a bridge between the two encodings
        shared_position = (nl_position + bash_position) / 2
        norm = np.linalg.norm(shared_position)
        if norm > 1e-10:
            shared_position = shared_position / norm * CRITICAL_LINE
        
        # Add to both structures
        node_id = f"map_{len(self.nl_structure)}"
        
        nl_node = self.nl_structure.add(
            node_id=node_id,
            position=shared_position,
            data={'nl': nl_text, 'bash': bash_command}
        )
        
        bash_node = self.bash_structure.add(
            node_id=node_id,
            position=shared_position,
            data={'nl': nl_text, 'bash': bash_command}
        )
        
        return nl_node, bash_node
    
    def translate(self, nl_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Translate natural language to bash command(s).
        
        Returns list of (command, confidence) tuples.
        """
        # Encode NL
        nl_position = NLTranscoder.encode(nl_text, self.dims)
        
        # Query NL structure
        matches = self.nl_structure.query_nearest(nl_position, k=top_k)
        
        # Decode to bash
        results = []
        for node, similarity in matches:
            if node.data and 'bash' in node.data:
                results.append((node.data['bash'], similarity))
        
        return results
    
    def feedback(self, nl_text: str, chosen_command: str, success: bool) -> None:
        """
        Provide feedback on a translation.
        
        Moves the mapping toward or away from the query position.
        """
        nl_position = NLTranscoder.encode(nl_text, self.dims)
        
        # Find the node for this command
        for node in self.nl_structure:
            if node.data and node.data.get('bash') == chosen_command:
                self.nl_structure.feedback(node.id, nl_position, success)
                self.bash_structure.feedback(node.id, nl_position, success)
                break
    
    def seed_common_commands(self) -> None:
        """Seed with common NL → Bash mappings."""
        mappings = [
            # File listing
            ("list files", "ls"),
            ("show files", "ls"),
            ("list all files", "ls -la"),
            ("show hidden files", "ls -a"),
            ("list files with details", "ls -l"),
            ("list files recursively", "ls -R"),
            ("find files", "find . -type f"),
            ("find directories", "find . -type d"),
            ("search for files named", "find . -name"),
            
            # File operations
            ("create file", "touch"),
            ("make directory", "mkdir"),
            ("create directory", "mkdir"),
            ("remove file", "rm"),
            ("delete file", "rm"),
            ("remove directory", "rmdir"),
            ("delete directory recursively", "rm -rf"),
            ("copy file", "cp"),
            ("move file", "mv"),
            ("rename file", "mv"),
            
            # File content
            ("show file contents", "cat"),
            ("display file", "cat"),
            ("read file", "cat"),
            ("show first lines", "head"),
            ("show last lines", "tail"),
            ("count lines", "wc -l"),
            ("count words", "wc -w"),
            
            # Process management
            ("list processes", "ps aux"),
            ("show running processes", "ps aux"),
            ("find process", "pgrep"),
            ("kill process", "kill"),
            ("terminate process", "kill -9"),
            ("stop all processes named", "pkill"),
            
            # System info
            ("show disk usage", "df -h"),
            ("check disk space", "df -h"),
            ("show memory usage", "free -h"),
            ("check memory", "free -h"),
            ("system uptime", "uptime"),
            ("show system info", "uname -a"),
            
            # Network
            ("show network connections", "netstat -tuln"),
            ("list open ports", "ss -tuln"),
            ("show listening ports", "lsof -i"),
            ("network interfaces", "ip addr"),
            ("show ip address", "ip addr"),
        ]
        
        for nl, bash in mappings:
            self.add_mapping(nl, bash)
    
    def stats(self) -> Dict[str, Any]:
        """Get translator statistics."""
        return {
            'dims': self.dims,
            'nl_structure': self.nl_structure.stats(),
            'bash_structure': self.bash_structure.stats(),
        }
    
    def save(self, path: str) -> None:
        """Save both structures."""
        import json
        from pathlib import Path
        
        data = {
            'type': 'NLToBashTranslator',
            'version': '1.0',
            'dims': self.dims,
            'nl_structure': self.nl_structure.to_dict(),
            'bash_structure': self.bash_structure.to_dict(),
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'NLToBashTranslator':
        """Load from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        translator = cls(dims=data.get('dims', 12))
        translator.nl_structure = HyperdimensionalStructure.from_dict(data['nl_structure'])
        translator.bash_structure = HyperdimensionalStructure.from_dict(data['bash_structure'])
        
        return translator


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Natural Language to Bash Translator ===\n")
    
    # Create and seed translator
    translator = NLToBashTranslator(dims=12)
    translator.seed_common_commands()
    
    print(f"Seeded with {len(translator.nl_structure)} mappings")
    print(f"Dimensions: {translator.dims}")
    print()
    
    # Test translations
    test_queries = [
        "list all files in directory",
        "show me the files",
        "what files are here",
        "delete the file",
        "remove this directory",
        "show running processes",
        "kill the process",
        "how much disk space",
        "check memory usage",
        "show network connections",
        "what's my ip address",
        "create a new folder",
        "make a directory called test",
        "find all python files",
    ]
    
    print("--- Translation Results ---")
    for query in test_queries:
        results = translator.translate(query, top_k=3)
        if results:
            best_cmd, confidence = results[0]
            alternatives = [cmd for cmd, _ in results[1:3]]
            alt_str = f" (also: {', '.join(alternatives)})" if alternatives else ""
            print(f"  '{query}'")
            print(f"    → {best_cmd} ({confidence:.2f}){alt_str}")
        else:
            print(f"  '{query}' → NO MATCH")
        print()
    
    # Test feedback/learning
    print("--- Testing Learning ---")
    print("Before feedback:")
    results = translator.translate("display files", top_k=1)
    print(f"  'display files' → {results[0][0]} ({results[0][1]:.3f})")
    
    # Provide positive feedback
    translator.feedback("display files", "ls", success=True)
    translator.feedback("display files", "ls", success=True)
    translator.feedback("display files", "ls", success=True)
    
    print("After positive feedback:")
    results = translator.translate("display files", top_k=1)
    print(f"  'display files' → {results[0][0]} ({results[0][1]:.3f})")
    
    # Test persistence
    print("\n--- Testing Persistence ---")
    translator.save("/tmp/nl_to_bash.json")
    print("Saved to /tmp/nl_to_bash.json")
    
    loaded = NLToBashTranslator.load("/tmp/nl_to_bash.json")
    print(f"Loaded: {len(loaded.nl_structure)} mappings")
    
    results = loaded.translate("list files", top_k=1)
    print(f"  'list files' → {results[0][0]} ({results[0][1]:.3f})")
    
    print("\n✓ NL to Bash translator working!")
    print("\nKey insights:")
    print("  - Transcoders are STATELESS (like shaders)")
    print("  - All state lives in the structures")
    print("  - Structures can be saved/loaded independently")
    print("  - Learning happens via position movement in structures")
