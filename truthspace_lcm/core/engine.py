"""
TruthSpace Engine - Minimal Bootstrap for Knowledge-Driven Computation

Philosophy:
- Code is just a geometric interpreter
- All logic lives in TruthSpace as knowledge
- The engine only needs to: Encode → Query → Decode → Execute

This is the minimal bootstrap that allows TruthSpace to be self-extending.
New capabilities are added as knowledge, not code.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from truthspace_lcm.core.knowledge_manager import (
    KnowledgeManager, 
    KnowledgeDomain, 
    KnowledgeEntry,
)
from truthspace_lcm.core.intent_manager import IntentManager, StepType
from truthspace_lcm.core.phi_encoder import PhiEncoder


class OutputType(Enum):
    """Type of output to generate."""
    BASH = "bash"
    PYTHON = "python"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class EngineResult:
    """Result from the TruthSpace engine."""
    success: bool
    output: str
    output_type: OutputType
    explanation: str
    knowledge_used: List[KnowledgeEntry]
    confidence: float
    position: np.ndarray = None  # The geometric position of the query
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(8)


class TruthSpaceEngine:
    """
    Minimal bootstrap engine for TruthSpace.
    
    The engine itself contains minimal logic - it's just a geometric
    encoder/decoder. All actual functionality comes from knowledge
    stored in TruthSpace.
    
    Core operations:
    1. encode(text) → position in TruthSpace
    2. query(position) → nearest knowledge entries
    3. decode(knowledge, context) → executable output
    4. execute(output) → result
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        self.storage_dir = storage_dir
        self.manager = KnowledgeManager()  # Uses SQLite backend
        self.intent_manager = IntentManager(storage_dir=storage_dir)
        self.phi_encoder = PhiEncoder()  # φ-based semantic encoder
        self.dim = 8  # Dimensionality of TruthSpace
    
    # =========================================================================
    # ENCODE: Natural Language → Geometric Position
    # =========================================================================
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode natural language text into a geometric position in TruthSpace.
        
        Uses the φ-encoder for semantic positioning based on primitives.
        Similar meanings → nearby positions.
        """
        # Use φ-encoder for semantic decomposition
        decomposition = self.phi_encoder.encode(text)
        return decomposition.position
    
    # =========================================================================
    # QUERY: Find Nearest Knowledge in TruthSpace
    # =========================================================================
    
    def query(
        self, 
        position: np.ndarray, 
        domain: KnowledgeDomain = None,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[float, KnowledgeEntry]]:
        """
        Query TruthSpace for knowledge near a geometric position.
        
        Returns entries sorted by similarity (cosine distance).
        """
        results = []
        
        for entry in self.manager.entries.values():
            # Filter by domain if specified
            if domain and entry.domain != domain:
                continue
            
            # Compute similarity
            similarity = self._similarity(position, entry.position)
            
            if similarity >= threshold:
                results.append((similarity, entry))
        
        # Sort by similarity descending
        results.sort(key=lambda x: -x[0])
        
        return results[:top_k]
    
    def query_by_text(
        self, 
        text: str, 
        domain: KnowledgeDomain = None,
        top_k: int = 10
    ) -> List[Tuple[float, KnowledgeEntry]]:
        """Convenience: encode text and query in one step."""
        position = self.encode(text)
        return self.query(position, domain, top_k)
    
    def _similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute cosine similarity between two positions."""
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(pos1, pos2) / (norm1 * norm2))
    
    # =========================================================================
    # DECODE: Knowledge → Executable Output
    # =========================================================================
    
    def decode(
        self, 
        knowledge: List[Tuple[float, KnowledgeEntry]], 
        context: str
    ) -> Tuple[str, OutputType, str]:
        """
        Decode knowledge entries into executable output.
        
        This is where knowledge transforms into action.
        The decoding rules themselves could be knowledge entries.
        
        Returns: (output, output_type, explanation)
        """
        if not knowledge:
            return "# No matching knowledge", OutputType.UNKNOWN, "No knowledge found"
        
        best_sim, best_entry = knowledge[0]
        
        # Determine output type from entry metadata or type
        output_type = self._determine_output_type(best_entry)
        
        # Extract executable content from entry
        output = self._extract_output(best_entry, context)
        
        explanation = best_entry.description
        
        return output, output_type, explanation
    
    def _determine_output_type(self, entry: KnowledgeEntry) -> OutputType:
        """Determine what type of output this knowledge produces."""
        # Check metadata first
        if entry.metadata.get("step_type") == "bash":
            return OutputType.BASH
        if entry.metadata.get("step_type") == "python":
            return OutputType.PYTHON
        
        # Check entry type
        if entry.entry_type in ["command", "bash_command"]:
            return OutputType.BASH
        if entry.entry_type in ["function", "library", "pattern", "python_module"]:
            return OutputType.PYTHON
        
        # Check keywords
        if "bash" in entry.keywords or "shell" in entry.keywords:
            return OutputType.BASH
        if "python" in entry.keywords:
            return OutputType.PYTHON
        
        # Check for intent entries
        if entry.entry_type == "intent":
            step_type = entry.metadata.get("step_type", "")
            if step_type == "bash":
                return OutputType.BASH
            if step_type == "python":
                return OutputType.PYTHON
        
        return OutputType.UNKNOWN
    
    def _extract_output(self, entry: KnowledgeEntry, context: str) -> str:
        """Extract executable output from a knowledge entry."""
        # For intent entries, get the target command
        if entry.entry_type == "intent":
            commands = entry.metadata.get("target_commands", [])
            if commands:
                return commands[0]
        
        # Check various metadata fields for executable content
        if entry.metadata.get("code"):
            return entry.metadata["code"]
        
        if entry.metadata.get("command"):
            return entry.metadata["command"]
        
        if entry.metadata.get("syntax"):
            # Clean up syntax - take first line
            syntax = entry.metadata["syntax"]
            return syntax.split('\n')[0].strip()
        
        if entry.metadata.get("example"):
            return entry.metadata["example"]
        
        # Fallback to entry name (often the command itself)
        return entry.name
    
    # =========================================================================
    # PROCESS: The Unified Pipeline
    # =========================================================================
    
    def process(self, request: str, domain: KnowledgeDomain = None) -> EngineResult:
        """
        Process a natural language request through TruthSpace.
        
        Pipeline:
        1. Try learned intents first (pattern-based, high precision)
        2. Fall back to geometric similarity search
        3. Decode best match to executable output
        """
        # 1. Encode request to geometric position (for later use)
        position = self.encode(request)
        
        # 2. FIRST: Try learned intents (these have explicit trigger patterns)
        intent_result = self._try_intent_match(request)
        if intent_result:
            return intent_result
        
        # 3. FALLBACK: Query TruthSpace by geometric similarity
        knowledge = self.query(position, domain=domain, top_k=10)
        
        if not knowledge:
            return EngineResult(
                success=False,
                output="# No matching knowledge found",
                output_type=OutputType.UNKNOWN,
                explanation="Request did not match any knowledge in TruthSpace",
                knowledge_used=[],
                confidence=0.0,
                position=position
            )
        
        # 4. Decode knowledge to executable output
        output, output_type, explanation = self.decode(knowledge, request)
        
        best_sim = knowledge[0][0]
        
        return EngineResult(
            success=True,
            output=output,
            output_type=output_type,
            explanation=explanation,
            knowledge_used=[entry for _, entry in knowledge[:3]],
            confidence=best_sim,
            position=position
        )
    
    def _try_intent_match(self, request: str) -> Optional[EngineResult]:
        """
        Try to match request against learned intents.
        
        Intents are knowledge entries with explicit trigger patterns.
        They provide high-precision matching for known commands.
        """
        result = self.intent_manager.get_best_intent(request)
        
        if result:
            intent, confidence = result
            
            if confidence >= 0.7:
                # Get the target command
                command = intent.target_commands[0] if intent.target_commands else intent.name
                
                # Determine output type
                if intent.step_type == StepType.BASH:
                    output_type = OutputType.BASH
                elif intent.step_type == StepType.PYTHON:
                    output_type = OutputType.PYTHON
                else:
                    output_type = OutputType.UNKNOWN
                
                # Find the linked knowledge entry for more context
                knowledge_used = []
                for entry_id in intent.knowledge_entry_ids:
                    entry = self.manager.read(entry_id)
                    if entry:
                        knowledge_used.append(entry)
                
                return EngineResult(
                    success=True,
                    output=command,
                    output_type=output_type,
                    explanation=intent.description,
                    knowledge_used=knowledge_used,
                    confidence=confidence,
                    position=self.encode(request)
                )
        
        return None
    
    # =========================================================================
    # LEARN: Add New Knowledge to TruthSpace
    # =========================================================================
    
    def learn(
        self,
        name: str,
        description: str,
        keywords: List[str],
        entry_type: str = "knowledge",
        domain: KnowledgeDomain = KnowledgeDomain.PROGRAMMING,
        metadata: Dict[str, Any] = None
    ) -> KnowledgeEntry:
        """
        Add new knowledge to TruthSpace.
        
        The knowledge will be positioned geometrically based on its
        keywords and can then be retrieved by similar queries.
        """
        return self.manager.create(
            name=name,
            domain=domain,
            entry_type=entry_type,
            description=description,
            keywords=keywords,
            metadata=metadata or {}
        )
    
    # =========================================================================
    # INTROSPECTION: Understand TruthSpace
    # =========================================================================
    
    def explain_position(self, position: np.ndarray) -> Dict[str, float]:
        """Explain what a position means in terms of semantic dimensions."""
        dimension_names = [
            "domain",      # Which knowledge domain
            "identity",    # What is it (name, type)
            "spatial",     # Where does it apply
            "temporal",    # When is it relevant
            "causal",      # Why does it work
            "method",      # How does it work
            "attribute",   # What properties
            "relation",    # How does it connect
        ]
        
        return {name: float(position[i]) for i, name in enumerate(dimension_names)}
    
    def visualize_query(self, text: str, top_k: int = 5) -> str:
        """Visualize how a query maps to TruthSpace."""
        position = self.encode(text)
        results = self.query(position, top_k=top_k)
        
        lines = [
            f"Query: {text}",
            f"Position: {self.explain_position(position)}",
            "",
            "Nearest knowledge:",
        ]
        
        for sim, entry in results:
            lines.append(f"  {sim:.3f} | {entry.name} ({entry.entry_type})")
        
        return "\n".join(lines)


def demonstrate():
    """Demonstrate the TruthSpace engine."""
    print("=" * 70)
    print("TRUTHSPACE ENGINE - Minimal Bootstrap")
    print("=" * 70)
    print()
    
    engine = TruthSpaceEngine()
    
    # Test queries
    test_requests = [
        "show network interfaces",
        "list files in directory",
        "create a new folder",
        "check system uptime",
        "read a json file",
    ]
    
    for request in test_requests:
        print(f"\n{'─' * 60}")
        print(f"Request: {request}")
        print("─" * 60)
        
        result = engine.process(request, domain=KnowledgeDomain.PROGRAMMING)
        
        print(f"Output: {result.output}")
        print(f"Type: {result.output_type.value}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Knowledge: {[e.name for e in result.knowledge_used]}")
    
    print("\n" + "=" * 70)
    print("Engine demonstration complete!")


if __name__ == "__main__":
    demonstrate()
