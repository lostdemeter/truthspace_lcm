"""
φ-Based TruthSpace Engine

A unified engine that uses φ-anchored primitives for:
1. Natural Language → Code (forward mapping)
2. Code → Natural Language (reverse mapping)
3. Knowledge storage and retrieval
4. Self-extending knowledge base

This replaces the hash-based encoding with proper semantic structure.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from truthspace_lcm.core.phi_encoder import (
    PhiEncoder, 
    SemanticDecomposition,
    Primitive,
    PrimitiveType,
    PHI
)


class OutputType(Enum):
    """Type of output to generate."""
    BASH = "bash"
    PYTHON = "python"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class KnowledgeEntry:
    """A knowledge entry in φ-TruthSpace."""
    id: str
    name: str
    description: str
    
    # Semantic decomposition
    primitives: List[str]           # Primitive names that compose this entry
    position: List[float]           # φ-encoded position
    
    # Executable content
    output_type: str                # bash, python, text
    code: str                       # The actual code/command
    syntax: str = ""                # Syntax template
    examples: List[str] = field(default_factory=list)
    
    # Metadata
    keywords: List[str] = field(default_factory=list)
    residual_keywords: List[str] = field(default_factory=list)  # Keywords not matched to primitives
    confidence: float = 1.0
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""


@dataclass
class QueryResult:
    """Result of querying the knowledge base."""
    success: bool
    entry: Optional[KnowledgeEntry]
    similarity: float
    output: str
    output_type: OutputType
    explanation: str
    alternatives: List[Tuple[float, KnowledgeEntry]] = field(default_factory=list)


class PhiEngine:
    """
    φ-Based TruthSpace Engine.
    
    Uses φ-anchored primitives for semantic encoding.
    Supports bidirectional mapping: NL ↔ position ↔ code.
    Knowledge base grows dynamically.
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "phi_knowledge_store"
            )
        
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.encoder = PhiEncoder()
        self.entries: Dict[str, KnowledgeEntry] = {}
        
        self._load_entries()
        
        # If empty, bootstrap with core knowledge
        if not self.entries:
            self._bootstrap_knowledge()
    
    # =========================================================================
    # FORWARD MAPPING: Natural Language → Code
    # =========================================================================
    
    def query(self, request: str, top_k: int = 5) -> QueryResult:
        """
        Query the knowledge base with natural language.
        
        Pipeline: text → encode → find nearest → decode → output
        """
        # Encode request
        decomposition = self.encoder.encode(request)
        
        if decomposition.confidence == 0:
            return QueryResult(
                success=False,
                entry=None,
                similarity=0.0,
                output="# Could not understand request",
                output_type=OutputType.UNKNOWN,
                explanation="No semantic primitives matched",
            )
        
        # Find nearest entries (pass keywords for boosting)
        query_keywords = self.encoder._tokenize(request)
        results = self._find_nearest(decomposition.position, top_k=top_k, 
                                    query_keywords=query_keywords)
        
        if not results:
            return QueryResult(
                success=False,
                entry=None,
                similarity=0.0,
                output="# No matching knowledge found",
                output_type=OutputType.UNKNOWN,
                explanation=f"Understood as: {[p.name for p, _ in decomposition.primitives]}",
            )
        
        best_sim, best_entry = results[0]
        
        # Determine output type
        output_type = OutputType(best_entry.output_type) if best_entry.output_type in [e.value for e in OutputType] else OutputType.UNKNOWN
        
        return QueryResult(
            success=True,
            entry=best_entry,
            similarity=best_sim,
            output=best_entry.code,
            output_type=output_type,
            explanation=best_entry.description,
            alternatives=results[1:],
        )
    
    def _find_nearest(self, position: np.ndarray, top_k: int = 5, 
                      query_keywords: List[str] = None) -> List[Tuple[float, KnowledgeEntry]]:
        """Find entries nearest to a position, with keyword boosting."""
        results = []
        
        for entry in self.entries.values():
            entry_pos = np.array(entry.position)
            
            # Base similarity from position
            pos_sim = self.encoder.similarity(position, entry_pos)
            
            # Keyword boost: if query keywords match entry keywords, boost score
            keyword_boost = 0.0
            if query_keywords:
                entry_kw_set = set(kw.lower() for kw in entry.keywords)
                entry_kw_set.add(entry.name.lower())
                
                for qkw in query_keywords:
                    if qkw.lower() in entry_kw_set:
                        keyword_boost += 0.1
                    # Partial match
                    for ekw in entry_kw_set:
                        if qkw.lower() in ekw or ekw in qkw.lower():
                            keyword_boost += 0.05
            
            # Combined score (cap keyword boost at 0.3)
            combined_sim = pos_sim + min(keyword_boost, 0.3)
            results.append((combined_sim, entry))
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    # =========================================================================
    # REVERSE MAPPING: Code → Natural Language
    # =========================================================================
    
    def describe(self, code: str) -> str:
        """
        Describe what a piece of code does in natural language.
        
        Pipeline: code → find entry → get description
                  OR code → extract keywords → encode → find nearest → describe
        """
        # First, try to find exact match by code
        for entry in self.entries.values():
            if entry.code.strip() == code.strip():
                return entry.description
        
        # Try to match by command name (first word)
        cmd_parts = code.strip().split()
        if cmd_parts:
            cmd_name = cmd_parts[0]
            for entry in self.entries.values():
                if entry.name.lower() == cmd_name.lower():
                    return entry.description
                # Also check if code starts with the same command
                if entry.code.split()[0].lower() == cmd_name.lower():
                    return entry.description
        
        # Otherwise, encode the code and find nearest entry
        decomposition = self.encoder.encode(code)
        query_keywords = self.encoder._tokenize(code)
        results = self._find_nearest(decomposition.position, top_k=1, 
                                    query_keywords=query_keywords)
        
        if results:
            sim, entry = results[0]
            if sim > 0.5:
                return entry.description
        
        # Fall back to primitive-based description
        return self.encoder.position_to_description(decomposition.position)
    
    def code_to_position(self, code: str) -> np.ndarray:
        """Encode code to a position (for storage or comparison)."""
        decomposition = self.encoder.encode(code)
        return decomposition.position
    
    # =========================================================================
    # KNOWLEDGE MANAGEMENT
    # =========================================================================
    
    def add_knowledge(
        self,
        name: str,
        description: str,
        code: str,
        output_type: str = "bash",
        keywords: List[str] = None,
        syntax: str = "",
        examples: List[str] = None,
    ) -> KnowledgeEntry:
        """
        Add new knowledge to the base.
        
        The position is computed automatically from the description and keywords.
        """
        keywords = keywords or []
        examples = examples or []
        
        # Encode using description + name + keywords
        text_to_encode = f"{name} {description} {' '.join(keywords)}"
        decomposition = self.encoder.encode(text_to_encode)
        
        # Generate ID
        entry_id = f"{name.lower().replace(' ', '_')}_{len(self.entries)}"
        
        now = datetime.now().isoformat()
        
        entry = KnowledgeEntry(
            id=entry_id,
            name=name,
            description=description,
            primitives=[p.name for p, _ in decomposition.primitives],
            position=decomposition.position.tolist(),
            output_type=output_type,
            code=code,
            syntax=syntax,
            examples=examples,
            keywords=keywords,
            residual_keywords=decomposition.residual_keywords,
            confidence=decomposition.confidence,
            created_at=now,
            updated_at=now,
        )
        
        self.entries[entry_id] = entry
        self._save_entry(entry)
        
        return entry
    
    def learn_from_command(self, command: str, description: str = None) -> KnowledgeEntry:
        """
        Learn a new command by analyzing it.
        
        This enables self-extension: the system can learn new commands
        without code changes.
        """
        # Extract command name
        parts = command.strip().split()
        cmd_name = parts[0] if parts else command
        
        # Generate description if not provided
        if not description:
            description = f"Execute the {cmd_name} command"
        
        # Infer keywords from command
        keywords = [cmd_name]
        
        # Add the knowledge
        return self.add_knowledge(
            name=cmd_name,
            description=description,
            code=command,
            output_type="bash",
            keywords=keywords,
        )
    
    # =========================================================================
    # PRIMITIVE GROWTH
    # =========================================================================
    
    def suggest_new_primitives(self) -> List[Dict[str, Any]]:
        """
        Analyze residual keywords to suggest new primitives.
        
        This enables the mesh to grow based on what the current
        primitives don't capture.
        """
        # Collect all residual keywords
        residual_counts: Dict[str, int] = {}
        
        for entry in self.entries.values():
            for kw in entry.residual_keywords:
                residual_counts[kw] = residual_counts.get(kw, 0) + 1
        
        # Find frequently occurring residuals
        suggestions = []
        for kw, count in sorted(residual_counts.items(), key=lambda x: -x[1]):
            if count >= 3:  # Threshold
                suggestions.append({
                    'keyword': kw,
                    'count': count,
                    'suggested_type': self._infer_primitive_type(kw),
                })
        
        return suggestions[:10]
    
    def _infer_primitive_type(self, keyword: str) -> str:
        """Infer what type of primitive a keyword might be."""
        action_indicators = ['ing', 'ate', 'ify', 'ize']
        domain_indicators = ['file', 'data', 'system', 'net', 'proc']
        
        kw_lower = keyword.lower()
        
        for ind in action_indicators:
            if kw_lower.endswith(ind):
                return 'action'
        
        for ind in domain_indicators:
            if ind in kw_lower:
                return 'domain'
        
        return 'unknown'
    
    def add_primitive(self, name: str, ptype: str, keywords: List[str],
                     dimension: int, level: float) -> None:
        """
        Add a new primitive to the encoder.
        
        This grows the semantic mesh.
        """
        ptype_enum = PrimitiveType(ptype)
        self.encoder.add_primitive(
            name=name,
            ptype=ptype_enum,
            keywords=set(keywords),
            dimension=dimension,
            level=level,
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _load_entries(self):
        """Load entries from disk."""
        entries_file = os.path.join(self.storage_dir, "entries.json")
        
        if os.path.exists(entries_file):
            with open(entries_file, 'r') as f:
                data = json.load(f)
                for entry_data in data:
                    entry = KnowledgeEntry(**entry_data)
                    self.entries[entry.id] = entry
    
    def _save_entry(self, entry: KnowledgeEntry):
        """Save all entries to disk."""
        entries_file = os.path.join(self.storage_dir, "entries.json")
        
        data = [asdict(e) for e in self.entries.values()]
        
        with open(entries_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _bootstrap_knowledge(self):
        """Bootstrap with core bash commands."""
        
        core_commands = [
            # File operations
            ("ls", "list directory contents", "ls", ["list", "directory", "files"]),
            ("ls -la", "list all files with details", "ls -la", ["list", "all", "files", "detailed"]),
            ("cat", "display file contents", "cat <file>", ["read", "file", "display", "contents"]),
            ("head", "show first lines of file", "head <file>", ["read", "file", "first", "lines"]),
            ("tail", "show last lines of file", "tail <file>", ["read", "file", "last", "lines"]),
            ("mkdir", "create directory", "mkdir <dir>", ["create", "directory", "folder"]),
            ("rmdir", "remove empty directory", "rmdir <dir>", ["remove", "directory", "folder"]),
            ("rm", "remove file", "rm <file>", ["remove", "delete", "file"]),
            ("rm -rf", "remove directory recursively", "rm -rf <dir>", ["remove", "delete", "directory", "recursive"]),
            ("cp", "copy file", "cp <src> <dst>", ["copy", "file"]),
            ("mv", "move or rename file", "mv <src> <dst>", ["move", "rename", "file"]),
            ("touch", "create empty file", "touch <file>", ["create", "file", "empty"]),
            ("find", "search for files", "find <path> -name <pattern>", ["search", "find", "files"]),
            ("grep", "search text in files", "grep <pattern> <file>", ["search", "text", "file", "pattern"]),
            
            # Process operations
            ("ps", "list processes", "ps aux", ["list", "processes", "running"]),
            ("top", "show running processes", "top", ["show", "processes", "monitor"]),
            ("htop", "interactive process viewer", "htop", ["show", "processes", "interactive"]),
            ("kill", "terminate process", "kill <pid>", ["kill", "terminate", "process"]),
            ("killall", "kill processes by name", "killall <name>", ["kill", "terminate", "process", "name"]),
            
            # Network operations
            ("ping", "test network connectivity", "ping <host>", ["network", "test", "connectivity"]),
            ("ifconfig", "show network interfaces", "ifconfig", ["network", "interface", "show", "ip"]),
            ("ip addr", "show IP addresses", "ip addr", ["network", "ip", "address", "show"]),
            ("curl", "transfer data from URL", "curl <url>", ["network", "download", "url", "http"]),
            ("wget", "download file from URL", "wget <url>", ["network", "download", "file", "url"]),
            ("ssh", "connect to remote host", "ssh <user>@<host>", ["network", "remote", "connect", "ssh"]),
            ("scp", "copy files over SSH", "scp <src> <user>@<host>:<dst>", ["network", "copy", "remote", "file"]),
            
            # System operations
            ("uname", "show system information", "uname -a", ["system", "info", "kernel"]),
            ("uptime", "show system uptime", "uptime", ["system", "uptime", "running"]),
            ("df", "show disk usage", "df -h", ["system", "disk", "usage", "space"]),
            ("du", "show directory size", "du -sh <dir>", ["directory", "size", "disk"]),
            ("free", "show memory usage", "free -h", ["system", "memory", "ram", "usage"]),
            ("dmesg", "show kernel messages", "dmesg", ["system", "kernel", "messages", "log"]),
            ("whoami", "show current user", "whoami", ["user", "current", "name"]),
            ("id", "show user and group IDs", "id", ["user", "group", "id"]),
            
            # Text processing
            ("echo", "print text", "echo <text>", ["print", "text", "output"]),
            ("sort", "sort lines", "sort <file>", ["sort", "lines", "text"]),
            ("uniq", "remove duplicate lines", "uniq <file>", ["unique", "duplicate", "lines"]),
            ("wc", "count lines/words/chars", "wc <file>", ["count", "lines", "words"]),
            ("sed", "stream editor", "sed 's/old/new/g' <file>", ["replace", "text", "edit"]),
            ("awk", "pattern processing", "awk '{print $1}' <file>", ["process", "text", "columns"]),
            
            # Compression
            ("tar", "archive files", "tar -cvf archive.tar <files>", ["archive", "compress", "tar"]),
            ("tar extract", "extract archive", "tar -xvf archive.tar", ["extract", "decompress", "tar"]),
            ("gzip", "compress file", "gzip <file>", ["compress", "gzip"]),
            ("gunzip", "decompress file", "gunzip <file>.gz", ["decompress", "gunzip"]),
            ("zip", "create zip archive", "zip archive.zip <files>", ["compress", "zip", "archive"]),
            ("unzip", "extract zip archive", "unzip archive.zip", ["extract", "unzip", "archive"]),
        ]
        
        for name, description, code, keywords in core_commands:
            self.add_knowledge(
                name=name,
                description=description,
                code=code,
                output_type="bash",
                keywords=keywords,
            )
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        primitive_counts = {}
        for entry in self.entries.values():
            for prim in entry.primitives:
                primitive_counts[prim] = primitive_counts.get(prim, 0) + 1
        
        return {
            'total_entries': len(self.entries),
            'primitive_coverage': primitive_counts,
            'avg_confidence': np.mean([e.confidence for e in self.entries.values()]) if self.entries else 0,
        }


def demonstrate():
    """Demonstrate the φ-engine."""
    print("=" * 70)
    print("φ-BASED TRUTHSPACE ENGINE")
    print("=" * 70)
    
    # Use a temp directory for demo
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PhiEngine(storage_dir=tmpdir)
        
        stats = engine.stats()
        print(f"\nKnowledge base: {stats['total_entries']} entries")
        print(f"Average confidence: {stats['avg_confidence']:.2f}")
        
        # Test forward mapping: NL → Code
        print("\n" + "=" * 70)
        print("FORWARD MAPPING: Natural Language → Code")
        print("=" * 70)
        
        queries = [
            "show network interfaces",
            "list all files in directory",
            "create a new folder",
            "delete file recursively",
            "search for text in files",
            "show running processes",
            "connect to remote server",
            "compress files into archive",
        ]
        
        for query in queries:
            result = engine.query(query)
            status = "✓" if result.success else "✗"
            print(f"\n{status} '{query}'")
            if result.success:
                print(f"   → {result.output}")
                print(f"   Similarity: {result.similarity:.2f}")
            else:
                print(f"   {result.explanation}")
        
        # Test reverse mapping: Code → NL
        print("\n" + "=" * 70)
        print("REVERSE MAPPING: Code → Natural Language")
        print("=" * 70)
        
        codes = [
            "ls -la",
            "grep pattern file.txt",
            "mkdir newfolder",
            "ping google.com",
            "tar -xvf archive.tar",
        ]
        
        for code in codes:
            description = engine.describe(code)
            print(f"\n'{code}'")
            print(f"   → {description}")
        
        # Test learning new command
        print("\n" + "=" * 70)
        print("LEARNING NEW COMMAND")
        print("=" * 70)
        
        new_entry = engine.learn_from_command(
            "ncdu",
            "interactive disk usage analyzer"
        )
        print(f"\nLearned: {new_entry.name}")
        print(f"  Primitives: {new_entry.primitives}")
        print(f"  Position: {new_entry.position[:4]}...")
        
        # Query for the new command
        result = engine.query("analyze disk usage interactively")
        print(f"\nQuery 'analyze disk usage interactively':")
        print(f"  → {result.output} (sim: {result.similarity:.2f})")
        
        # Suggest new primitives
        print("\n" + "=" * 70)
        print("PRIMITIVE GROWTH SUGGESTIONS")
        print("=" * 70)
        
        suggestions = engine.suggest_new_primitives()
        if suggestions:
            print("\nSuggested new primitives based on residual keywords:")
            for s in suggestions[:5]:
                print(f"  '{s['keyword']}' ({s['count']} occurrences) - type: {s['suggested_type']}")
        else:
            print("\nNo new primitives suggested (good coverage!)")
        
        print("\n" + "=" * 70)
        print("φ-engine demonstration complete!")


if __name__ == "__main__":
    demonstrate()
