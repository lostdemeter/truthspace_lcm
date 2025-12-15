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
            
        except KnowledgeGapError as e:
            if not self.auto_learn:
                raise
            
            # Try to learn
            entry = self.ingestor.try_learn(request)
            if entry is None:
                raise KnowledgeGapError(request, e.best_match)
            
            learned = True
            output = self._extract_output(entry)
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
