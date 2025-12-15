"""
Ingestor: Unified Knowledge Acquisition

This is the consolidated ingestion layer that replaces:
- knowledge_ingestor.py
- knowledge_acquisition.py

Design principles:
- Single interface for acquiring knowledge from any source
- Auto-detects source type (man page, pydoc, web, user input)
- Creates properly typed KnowledgeEntry objects
- Fail fast - if we can't parse it, raise an error
"""

import os
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from truthspace_lcm.core.truthspace import (
    TruthSpace,
    KnowledgeEntry,
    KnowledgeDomain,
    EntryType,
    KnowledgeGapError,
)


# =============================================================================
# SOURCE TYPES
# =============================================================================

class SourceType(Enum):
    """Types of knowledge sources."""
    MAN_PAGE = "man_page"
    PYTHON_HELP = "python_help"
    USER_INPUT = "user_input"
    AUTO = "auto"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class IngestionError(Exception):
    """Raised when knowledge cannot be ingested."""
    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"Cannot ingest '{source}': {reason}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParsedKnowledge:
    """Result of parsing a knowledge source."""
    name: str
    description: str
    syntax: str
    examples: List[str]
    keywords: List[str]
    source_type: SourceType
    output_type: str  # bash, python, text
    raw_content: str = ""


# =============================================================================
# INGESTOR
# =============================================================================

class Ingestor:
    """
    Unified knowledge acquisition from various sources.
    
    Supports:
    - Man pages (bash commands)
    - Python help (modules, functions)
    - User-provided knowledge
    - Auto-detection
    """
    
    def __init__(self, truthspace: TruthSpace = None):
        self.ts = truthspace or TruthSpace()
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def ingest(
        self,
        source: str,
        source_type: SourceType = SourceType.AUTO,
    ) -> KnowledgeEntry:
        """
        Ingest knowledge from a source.
        
        Args:
            source: Command name, module name, or raw content
            source_type: Type of source (auto-detected if AUTO)
        
        Returns:
            KnowledgeEntry stored in TruthSpace
        
        Raises:
            IngestionError if source cannot be parsed
        """
        # Auto-detect source type
        if source_type == SourceType.AUTO:
            source_type = self._detect_source_type(source)
        
        # Parse based on source type
        if source_type == SourceType.MAN_PAGE:
            parsed = self._parse_man_page(source)
        elif source_type == SourceType.PYTHON_HELP:
            parsed = self._parse_python_help(source)
        elif source_type == SourceType.USER_INPUT:
            parsed = self._parse_user_input(source)
        else:
            raise IngestionError(source, f"Unknown source type: {source_type}")
        
        # Store in TruthSpace
        return self._store_parsed(parsed)
    
    def ingest_custom(
        self,
        name: str,
        description: str,
        keywords: List[str],
        output_type: str = "bash",
        syntax: str = "",
        examples: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> KnowledgeEntry:
        """
        Ingest custom user-provided knowledge.
        
        This is the direct API for adding knowledge without parsing.
        """
        entry_type = EntryType.COMMAND if output_type in ["bash", "python"] else EntryType.CONCEPT
        domain = KnowledgeDomain.PROGRAMMING if output_type in ["bash", "python"] else KnowledgeDomain.GENERAL
        
        meta = metadata or {}
        meta.update({
            "output_type": output_type,
            "syntax": syntax,
            "examples": examples or [],
            "source": "user_input",
        })
        
        return self.ts.store(
            name=name,
            entry_type=entry_type,
            domain=domain,
            description=description,
            keywords=keywords,
            metadata=meta,
        )
    
    def try_learn(self, request: str) -> Optional[KnowledgeEntry]:
        """
        Try to automatically learn knowledge to fulfill a request.
        
        Called when a KnowledgeGapError is raised.
        Extracts potential command names and tries to ingest them.
        
        Returns:
            KnowledgeEntry if successful, None otherwise
        """
        candidates = self._extract_candidates(request)
        
        for candidate in candidates:
            # Check if we already have this
            existing = self.ts.get_by_name(candidate)
            if existing:
                continue
            
            try:
                return self.ingest(candidate)
            except IngestionError:
                continue
        
        return None
    
    # =========================================================================
    # SOURCE DETECTION
    # =========================================================================
    
    def _detect_source_type(self, source: str) -> SourceType:
        """Auto-detect the source type."""
        # Check if it's a man page
        if self._has_man_page(source):
            return SourceType.MAN_PAGE
        
        # Check if it's a Python module
        if self._is_python_module(source):
            return SourceType.PYTHON_HELP
        
        # Default to user input
        return SourceType.USER_INPUT
    
    def _has_man_page(self, name: str) -> bool:
        """Check if a man page exists for this name."""
        try:
            result = subprocess.run(
                ["man", "-w", name],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except:
            return False
    
    def _is_python_module(self, name: str) -> bool:
        """Check if this is a valid Python module."""
        try:
            result = subprocess.run(
                ["python3", "-c", f"import {name}"],
                capture_output=True,
                timeout=3,
            )
            return result.returncode == 0
        except:
            return False
    
    # =========================================================================
    # PARSING
    # =========================================================================
    
    def _parse_man_page(self, command: str) -> ParsedKnowledge:
        """Parse a man page into structured knowledge."""
        try:
            result = subprocess.run(
                ["man", command],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "MANWIDTH": "80", "PAGER": "cat"},
            )
            
            if result.returncode != 0:
                raise IngestionError(command, "Man page not found")
            
            content = result.stdout
        except subprocess.TimeoutExpired:
            raise IngestionError(command, "Man page timeout")
        except Exception as e:
            raise IngestionError(command, str(e))
        
        # Parse sections
        description = self._extract_section(content, "DESCRIPTION") or \
                      self._extract_section(content, "NAME") or \
                      f"The {command} command"
        
        synopsis = self._extract_section(content, "SYNOPSIS") or command
        examples_section = self._extract_section(content, "EXAMPLES") or ""
        
        # Extract examples
        examples = []
        for line in examples_section.split("\n"):
            line = line.strip()
            if line.startswith(command) or line.startswith("$"):
                examples.append(line.lstrip("$ "))
        
        # Generate keywords
        keywords = self._extract_keywords_from_text(description)
        keywords.extend(["bash", "shell", "command", command])
        keywords = list(set(keywords))[:15]
        
        return ParsedKnowledge(
            name=command,
            description=description[:500],
            syntax=synopsis[:200],
            examples=examples[:5],
            keywords=keywords,
            source_type=SourceType.MAN_PAGE,
            output_type="bash",
            raw_content=content[:2000],
        )
    
    def _parse_python_help(self, module: str) -> ParsedKnowledge:
        """Parse Python help into structured knowledge."""
        try:
            result = subprocess.run(
                ["python3", "-c", f"import {module}; help({module})"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode != 0:
                raise IngestionError(module, "Python module not found")
            
            content = result.stdout
        except subprocess.TimeoutExpired:
            raise IngestionError(module, "Python help timeout")
        except Exception as e:
            raise IngestionError(module, str(e))
        
        # Parse help output
        lines = content.split("\n")
        description = ""
        
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("Help on"):
                description = line.strip()
                break
        
        if not description:
            description = f"Python module: {module}"
        
        # Extract functions/classes mentioned
        keywords = [module, "python", "module"]
        for line in lines:
            if "def " in line or "class " in line:
                match = re.search(r'(?:def|class)\s+(\w+)', line)
                if match:
                    keywords.append(match.group(1))
        
        keywords = list(set(keywords))[:15]
        
        return ParsedKnowledge(
            name=module,
            description=description[:500],
            syntax=f"import {module}",
            examples=[f"import {module}"],
            keywords=keywords,
            source_type=SourceType.PYTHON_HELP,
            output_type="python",
            raw_content=content[:2000],
        )
    
    def _parse_user_input(self, content: str) -> ParsedKnowledge:
        """Parse user-provided content."""
        lines = content.strip().split("\n")
        name = lines[0] if lines else "unknown"
        description = "\n".join(lines[1:]) if len(lines) > 1 else name
        
        keywords = self._extract_keywords_from_text(content)
        
        return ParsedKnowledge(
            name=name,
            description=description[:500],
            syntax="",
            examples=[],
            keywords=keywords[:15],
            source_type=SourceType.USER_INPUT,
            output_type="text",
            raw_content=content,
        )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _extract_section(self, content: str, section: str) -> Optional[str]:
        """Extract a section from man page content."""
        pattern = rf'^{section}\s*\n(.*?)(?=^[A-Z]+\s*$|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        
        if match:
            text = match.group(1).strip()
            # Clean up formatting
            text = re.sub(r'\s+', ' ', text)
            return text[:1000]
        
        return None
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            'this', 'that', 'these', 'those', 'it', 'its',
        }
        
        words = re.findall(r'\b[a-z][a-z0-9_-]{2,15}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Count frequency and return top keywords
        from collections import Counter
        counts = Counter(keywords)
        return [word for word, _ in counts.most_common(20)]
    
    def _extract_candidates(self, request: str) -> List[str]:
        """Extract potential command/module names from a request."""
        candidates = []
        request_lower = request.lower()
        
        # Pattern: "using X" or "with X"
        using_match = re.findall(r'(?:using|with|via)\s+(\w+)', request_lower)
        candidates.extend(using_match)
        
        # Single words that might be commands
        words = re.findall(r'\b([a-z][a-z0-9_-]{1,15})\b', request_lower)
        
        common_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have',
            'show', 'display', 'list', 'get', 'make', 'create', 'delete',
            'file', 'files', 'directory', 'folder', 'all', 'my',
        }
        
        for word in words:
            if word not in common_words and len(word) > 2:
                if self._has_man_page(word) or self._is_python_module(word):
                    candidates.append(word)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        
        return unique[:5]
    
    def _store_parsed(self, parsed: ParsedKnowledge) -> KnowledgeEntry:
        """Store parsed knowledge in TruthSpace."""
        entry_type = EntryType.COMMAND
        domain = KnowledgeDomain.PROGRAMMING
        
        metadata = {
            "output_type": parsed.output_type,
            "syntax": parsed.syntax,
            "examples": parsed.examples,
            "source": parsed.source_type.value,
            "command": parsed.syntax.split()[0] if parsed.syntax else parsed.name,
        }
        
        return self.ts.store(
            name=parsed.name,
            entry_type=entry_type,
            domain=domain,
            description=parsed.description,
            keywords=parsed.keywords,
            metadata=metadata,
        )


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 70)
    print("INGESTOR - Unified Knowledge Acquisition")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    ts = TruthSpace(db_path)
    ingestor = Ingestor(ts)
    
    # Test man page ingestion
    print("\nIngesting 'ls' from man page...")
    try:
        entry = ingestor.ingest("ls")
        print(f"  ✓ {entry.name}: {entry.description[:60]}...")
        print(f"    Keywords: {entry.keywords[:5]}")
    except IngestionError as e:
        print(f"  ✗ {e}")
    
    # Test custom ingestion
    print("\nIngesting custom knowledge...")
    entry = ingestor.ingest_custom(
        name="backup_dir",
        description="Create a timestamped backup of a directory",
        keywords=["backup", "tar", "archive", "directory"],
        output_type="bash",
        syntax="tar -czf backup_$(date +%Y%m%d).tar.gz <dir>",
    )
    print(f"  ✓ {entry.name}")
    
    # Test auto-learn
    print("\nTrying to learn from 'show disk usage using df'...")
    entry = ingestor.try_learn("show disk usage using df")
    if entry:
        print(f"  ✓ Learned: {entry.name}")
    else:
        print("  ✗ Could not learn")
    
    # Cleanup
    os.unlink(db_path)
    
    print("\n" + "=" * 70)
    print("Ingestor test complete!")
    print("=" * 70)
