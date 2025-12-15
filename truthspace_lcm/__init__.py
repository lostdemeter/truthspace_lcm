"""
TruthSpace LCM - Language-Code Model

A natural language to code system that uses geometric knowledge encoding
to translate human requests into executable Python and Bash code.

Features:
- Natural language understanding
- Multi-step task planning
- Python code generation
- Bash command generation
- Safe code execution
- Output validation

Example:
    from truthspace_lcm import TruthSpaceLCM
    
    lcm = TruthSpaceLCM()
    plan = lcm.process("create a python project called myapp")
"""

__version__ = "0.1.0"
__author__ = "TruthSpace Team"

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry
from truthspace_lcm.core.code_generator import CodeGenerator
from truthspace_lcm.core.bash_generator import BashGenerator
from truthspace_lcm.core.task_planner import TaskPlanner, TaskPlan, TaskStep, StepType, StepStatus
from truthspace_lcm.core.executor import CodeExecutor, ExecutionResult, ExecutionStatus
from truthspace_lcm.core.knowledge_acquisition import (
    KnowledgeAcquisitionSystem,
    KnowledgeGapDetector,
    KnowledgeAcquirer,
    KnowledgeBuilder,
    KnowledgeGap,
    AcquiredKnowledge,
    KnowledgeSource
)

__all__ = [
    "KnowledgeManager",
    "KnowledgeDomain", 
    "KnowledgeEntry",
    "CodeGenerator",
    "BashGenerator",
    "TaskPlanner",
    "TaskPlan",
    "TaskStep",
    "StepType",
    "StepStatus",
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "KnowledgeAcquisitionSystem",
    "KnowledgeGapDetector",
    "KnowledgeAcquirer",
    "KnowledgeBuilder",
    "KnowledgeGap",
    "AcquiredKnowledge",
    "KnowledgeSource",
]
