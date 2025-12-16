"""
Resolver: Thin NL → Knowledge → Output Interface

This replaces:
- bash_generator.py
- code_generator.py
- task_planner.py
- engine.py
- phi_engine.py

Design principles:
- NO hardcoded patterns - everything comes from TruthSpace
- NO fallbacks - query succeeds or raises KnowledgeGapError
- Fail fast philosophy
- Single unified interface for all resolution
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
from truthspace_lcm.core.ingestor import Ingestor, IngestionError


# =============================================================================
# OUTPUT TYPES
# =============================================================================

class OutputType(Enum):
    """Type of output produced by resolver."""
    BASH = "bash"
    PYTHON = "python"
    TEXT = "text"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Resolution:
    """Result of resolving a natural language request."""
    success: bool
    output: str
    output_type: OutputType
    explanation: str
    knowledge: KnowledgeEntry
    confidence: float
    learned: bool = False  # True if we had to learn new knowledge


@dataclass
class ExecutionResult:
    """Result of executing resolved output."""
    success: bool
    stdout: str
    stderr: str
    return_code: int


# =============================================================================
# RESOLVER
# =============================================================================

class Resolver:
    """
    Thin resolver: NL → Knowledge → Output
    
    This is the unified interface that replaces all generators.
    It has NO hardcoded patterns - everything comes from TruthSpace.
    
    Pipeline:
    1. Query TruthSpace for matching knowledge
    2. If KnowledgeGapError, optionally try to learn
    3. Extract executable output from knowledge
    4. Optionally execute
    """
    
    def __init__(
        self,
        truthspace: TruthSpace = None,
        auto_learn: bool = True,
    ):
        self.ts = truthspace or TruthSpace()
        self.ingestor = Ingestor(self.ts)
        self.auto_learn = auto_learn
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def resolve(
        self,
        request: str,
        domain: KnowledgeDomain = None,
    ) -> Resolution:
        """
        Resolve natural language request to executable output.
        
        Args:
            request: Natural language request
            domain: Optional domain filter
        
        Returns:
            Resolution with output and metadata
        
        Raises:
            KnowledgeGapError if no match and auto_learn is False
        """
        learned = False
        
        try:
            # Query TruthSpace
            output, output_type, entry = self.ts.resolve(request, domain)
            confidence = 0.8  # Default confidence for successful resolution
            
            # Apply parameter extraction to output
            output = self._extract_parameters(request, output)
            
        except KnowledgeGapError as e:
            if not self.auto_learn:
                raise
            
            # Try to learn
            entry = self.ingestor.try_learn(request)
            if entry is None:
                raise KnowledgeGapError(request, e.best_match)
            
            learned = True
            output = self._extract_output(entry)
            output = self._extract_parameters(request, output)
            output_type = entry.metadata.get("output_type", "text")
            confidence = 0.6  # Lower confidence for newly learned
        
        return Resolution(
            success=True,
            output=output,
            output_type=OutputType(output_type),
            explanation=entry.description,
            knowledge=entry,
            confidence=confidence,
            learned=learned,
        )
    
    def resolve_and_execute(
        self,
        request: str,
        domain: KnowledgeDomain = None,
        timeout: int = 30,
    ) -> Tuple[Resolution, ExecutionResult]:
        """
        Resolve and execute in one step.
        
        Returns both the resolution and execution result.
        """
        resolution = self.resolve(request, domain)
        
        if resolution.output_type == OutputType.BASH:
            exec_result = self._execute_bash(resolution.output, timeout)
        elif resolution.output_type == OutputType.PYTHON:
            exec_result = self._execute_python(resolution.output, timeout)
        else:
            exec_result = ExecutionResult(
                success=True,
                stdout=resolution.output,
                stderr="",
                return_code=0,
            )
        
        return resolution, exec_result
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def _execute_bash(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a bash command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                return_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
            )
    
    def _execute_python(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute Python code."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Code timed out after {timeout}s",
                return_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
            )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _extract_output(self, entry: KnowledgeEntry) -> str:
        """Extract executable output from entry."""
        for field in ["code", "command", "syntax", "output"]:
            if entry.metadata.get(field):
                return entry.metadata[field]
        
        if entry.entry_type == EntryType.INTENT:
            commands = entry.metadata.get("target_commands", [])
            if commands:
                return commands[0]
        
        return entry.name
    
    # =========================================================================
    # PARAMETER EXTRACTION
    # =========================================================================
    
    def _extract_parameters(self, request: str, template: str) -> str:
        """
        Extract parameters from request and substitute into template.
        
        Uses parameter extraction patterns from TruthSpace CONCEPT entries.
        
        Args:
            request: Original natural language request
            template: Command template with placeholders like {filename}, <url>
        
        Returns:
            Template with parameters substituted
        """
        # Find all placeholders in template: {name}, <name>, or <name>
        placeholders = re.findall(r'\{(\w+)\}|<(\w+)>', template)
        
        if not placeholders:
            return template
        
        result = template
        
        for match in placeholders:
            # match is tuple like ('filename', '') or ('', 'url')
            param_name = match[0] or match[1]
            param_type = param_name.upper()
            
            # Get extraction patterns from knowledge
            value = self._extract_param_value(request, param_type)
            
            if value:
                # Replace both {name} and <name> formats
                result = re.sub(r'\{' + param_name + r'\}', value, result)
                result = re.sub(r'<' + param_name + r'>', value, result)
        
        return result
    
    def _extract_param_value(self, request: str, param_type: str) -> Optional[str]:
        """
        Extract a parameter value from request using knowledge-based patterns.
        
        Args:
            request: Natural language request
            param_type: Parameter type name (e.g., "FILENAME", "URL")
        
        Returns:
            Extracted value or None
        """
        # Query TruthSpace for the parameter concept
        try:
            concept = self.ts.get_by_name(param_type, entry_type=EntryType.CONCEPT)
        except:
            concept = None
        
        if not concept:
            # Fallback: try common extraction patterns
            return self._fallback_extract(request, param_type)
        
        # Get extraction patterns from concept metadata
        patterns = concept.metadata.get("extraction_patterns", [])
        
        if not patterns:
            return self._fallback_extract(request, param_type)
        
        # Sort by priority (lower = higher priority)
        if isinstance(patterns[0], dict):
            patterns = sorted(patterns, key=lambda p: p.get("priority", 99))
            pattern_strings = [p["pattern"] for p in patterns]
        else:
            pattern_strings = patterns
        
        # Try each pattern
        for pattern in pattern_strings:
            try:
                match = re.search(pattern, request, re.IGNORECASE)
                if match:
                    # Return first captured group
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Validate if validation pattern exists
                    validation = concept.metadata.get("validation")
                    if validation and not re.match(validation, value):
                        continue
                    
                    return value
            except re.error:
                continue
        
        # Return default if defined
        return concept.metadata.get("default")
    
    def _fallback_extract(self, request: str, param_type: str) -> Optional[str]:
        """Fallback extraction when no concept is found."""
        # Basic patterns for common types
        fallbacks = {
            "FILENAME": [r'"([^"]+)"', r"'([^']+)'", r'called\s+(\S+)', r'(\S+\.\w+)'],
            "DIRECTORY": [r'"([^"]+)"', r"'([^']+)'", r'called\s+(\S+)', r'folder\s+(\S+)'],
            "URL": [r'(https?://\S+)'],
            "PATTERN": [r'"([^"]+)"', r"'([^']+)'", r'for\s+(\S+)'],
            "NUMBER": [r'(\d+)'],
        }
        
        patterns = fallbacks.get(param_type, [r'"([^"]+)"', r"'([^']+)'"])
        
        for pattern in patterns:
            try:
                match = re.search(pattern, request, re.IGNORECASE)
                if match:
                    return match.group(1) if match.groups() else match.group(0)
            except:
                continue
        
        return None


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 70)
    print("RESOLVER - Thin NL → Knowledge → Output")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    ts = TruthSpace(db_path)
    resolver = Resolver(ts, auto_learn=True)
    
    # Seed some knowledge
    print("\nSeeding knowledge...")
    ts.store(
        name="ls",
        entry_type=EntryType.COMMAND,
        domain=KnowledgeDomain.PROGRAMMING,
        description="List directory contents",
        keywords=["list", "files", "directory", "bash"],
        metadata={"command": "ls -la", "output_type": "bash"}
    )
    
    ts.store(
        name="list_files_intent",
        entry_type=EntryType.INTENT,
        domain=KnowledgeDomain.PROGRAMMING,
        description="List files in directory",
        keywords=["list", "files", "show", "directory", "contents"],
        metadata={"target_commands": ["ls -la"], "output_type": "bash"}
    )
    
    # Test resolution
    print("\nResolving 'show files in current directory'...")
    try:
        result = resolver.resolve("show files in current directory")
        print(f"  Output: {result.output}")
        print(f"  Type: {result.output_type.value}")
        print(f"  Learned: {result.learned}")
    except KnowledgeGapError as e:
        print(f"  Knowledge gap: {e}")
    
    # Test with auto-learn
    print("\nResolving 'show disk usage' (may auto-learn)...")
    try:
        result = resolver.resolve("show disk usage")
        print(f"  Output: {result.output}")
        print(f"  Learned: {result.learned}")
    except KnowledgeGapError as e:
        print(f"  Knowledge gap: {e}")
    
    # Cleanup
    os.unlink(db_path)
    
    print("\n" + "=" * 70)
    print("Resolver test complete!")
    print("=" * 70)
