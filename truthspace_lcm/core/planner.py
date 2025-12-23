#!/usr/bin/env python3
"""
Planner: Task Decomposition and Execution for GeometricLCM

This module adds planning and code execution capabilities:
- Decompose tasks into executable steps
- Execute code in a sandboxed environment
- Track execution state and results
- Replan on failure

Key insight: Plans are sequences of concept frames with dependencies.
Planning is forward graph traversal (goal → steps), while reasoning
is backward traversal (observation → cause).

Author: Lesley Gushurst
License: GPLv3
"""

import re
import signal
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


# =============================================================================
# PLAN PRIMITIVES
# =============================================================================

class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(Enum):
    """Type of plan step."""
    DEFINE = "define"      # Define a variable
    COMPUTE = "compute"    # Compute an expression
    INVOKE = "invoke"      # Call a function
    CONTROL = "control"    # Control flow (if, for)
    RETURN = "return"      # Return a value
    VERIFY = "verify"      # Verify a condition


# =============================================================================
# PLAN STEP
# =============================================================================

@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: int
    step_type: StepType
    description: str
    code: str
    depends_on: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = ""
    
    def __str__(self):
        status_icon = {
            StepStatus.PENDING: "○",
            StepStatus.RUNNING: "◐",
            StepStatus.SUCCESS: "●",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "◌",
        }
        return f"{status_icon[self.status]} Step {self.step_id}: {self.description}"


# =============================================================================
# EXECUTION PLAN
# =============================================================================

@dataclass
class ExecutionPlan:
    """A complete execution plan."""
    task: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    final_result: Any = None
    success: bool = False
    
    def add_step(self, step_type: StepType, description: str, code: str,
                 depends_on: List[int] = None) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(
            step_id=len(self.steps) + 1,
            step_type=step_type,
            description=description,
            code=code,
            depends_on=depends_on or [],
        )
        self.steps.append(step)
        return step
    
    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (dependencies met)."""
        ready = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Check dependencies
            deps_met = all(
                self.steps[dep_id - 1].status == StepStatus.SUCCESS
                for dep_id in step.depends_on
            )
            if deps_met:
                ready.append(step)
        return ready
    
    def __str__(self):
        lines = [f"Plan: {self.task}", f"Goal: {self.goal}", "Steps:"]
        for step in self.steps:
            lines.append(f"  {step}")
        if self.final_result is not None:
            lines.append(f"Result: {self.final_result}")
        return "\n".join(lines)


# =============================================================================
# SANDBOX EXECUTION
# =============================================================================

class TimeoutError(Exception):
    """Raised when execution times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


# Safe builtins for sandboxed execution
SAFE_BUILTINS = {
    # Math
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
    'divmod': divmod,
    
    # Type conversion
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'tuple': tuple,
    'dict': dict,
    'set': set,
    
    # Iteration
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sorted': sorted,
    'reversed': reversed,
    
    # Boolean
    'all': all,
    'any': any,
    
    # Other safe functions
    'isinstance': isinstance,
    'type': type,
    'print': print,  # For debugging
}

# Safe modules (can be imported)
SAFE_MODULES = {
    'math': __import__('math'),
    'random': __import__('random'),
    'statistics': __import__('statistics'),
    'itertools': __import__('itertools'),
    'functools': __import__('functools'),
    'collections': __import__('collections'),
}


class Sandbox:
    """Sandboxed Python execution environment."""
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize sandbox.
        
        Args:
            timeout_seconds: Maximum execution time per step
        """
        self.timeout = timeout_seconds
        self.context: Dict[str, Any] = {}
        self.safe_globals = {
            '__builtins__': SAFE_BUILTINS,
            **SAFE_MODULES,
        }
    
    def reset(self):
        """Reset execution context."""
        self.context = {}
    
    def execute(self, code: str) -> tuple[bool, Any, str]:
        """
        Execute code in sandbox.
        
        Args:
            code: Python code to execute
        
        Returns:
            (success, result, error_message)
        """
        try:
            # Set timeout (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            except (AttributeError, ValueError):
                # Windows or other platform without SIGALRM
                old_handler = None
            
            try:
                # Try as expression first (returns value)
                result = eval(code, self.safe_globals, self.context)
                return True, result, ""
            except SyntaxError:
                # Not an expression, try as statement
                exec(code, self.safe_globals, self.context)
                return True, None, ""
            finally:
                # Cancel timeout
                if old_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
        
        except TimeoutError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"
    
    def get(self, name: str) -> Any:
        """Get a variable from context."""
        return self.context.get(name)
    
    def set(self, name: str, value: Any):
        """Set a variable in context."""
        self.context[name] = value


# =============================================================================
# PLANNER
# =============================================================================

class Planner:
    """
    Plan and execute tasks using GeometricLCM.
    
    Decomposes natural language tasks into executable steps,
    runs them in a sandbox, and tracks results.
    """
    
    def __init__(self, code_generator=None):
        """
        Initialize planner.
        
        Args:
            code_generator: Optional CodeGenerator for code generation
        """
        self.code_generator = code_generator
        self.sandbox = Sandbox()
        
        # Task patterns for planning
        self.task_patterns = {
            'calculate': self._plan_calculation,
            'compute': self._plan_calculation,
            'find': self._plan_search,
            'sort': self._plan_sort,
            'filter': self._plan_filter,
            'transform': self._plan_transform,
            'count': self._plan_count,
            'sum': self._plan_sum,
            'average': self._plan_average,
            'check': self._plan_check,
            'verify': self._plan_check,
        }
    
    def plan(self, task: str) -> ExecutionPlan:
        """
        Create an execution plan for a task.
        
        Args:
            task: Natural language task description
        
        Returns:
            ExecutionPlan with steps
        """
        task_lower = task.lower()
        
        # Find matching pattern
        for keyword, planner in self.task_patterns.items():
            if keyword in task_lower:
                return planner(task)
        
        # Default: try to parse as direct computation
        return self._plan_generic(task)
    
    def execute(self, plan: ExecutionPlan, verbose: bool = False) -> ExecutionPlan:
        """
        Execute a plan.
        
        Args:
            plan: The plan to execute
            verbose: Print progress
        
        Returns:
            Updated plan with results
        """
        self.sandbox.reset()
        
        # Copy initial context
        for key, value in plan.context.items():
            self.sandbox.set(key, value)
        
        while True:
            ready = plan.get_ready_steps()
            if not ready:
                break
            
            for step in ready:
                step.status = StepStatus.RUNNING
                if verbose:
                    print(f"Executing: {step.description}")
                
                success, result, error = self.sandbox.execute(step.code)
                
                if success:
                    step.status = StepStatus.SUCCESS
                    step.result = result
                    if verbose:
                        print(f"  → Success: {result}")
                else:
                    step.status = StepStatus.FAILED
                    step.error = error
                    if verbose:
                        print(f"  → Failed: {error}")
                    # Skip dependent steps
                    self._skip_dependents(plan, step.step_id)
        
        # Get final result (last successful step with a result)
        for step in reversed(plan.steps):
            if step.status == StepStatus.SUCCESS and step.result is not None:
                plan.final_result = step.result
                break
        
        # Check if all steps succeeded
        plan.success = all(
            step.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
            for step in plan.steps
        )
        
        return plan
    
    def _skip_dependents(self, plan: ExecutionPlan, failed_step_id: int):
        """Skip steps that depend on a failed step."""
        for step in plan.steps:
            if failed_step_id in step.depends_on:
                step.status = StepStatus.SKIPPED
                self._skip_dependents(plan, step.step_id)
    
    # =========================================================================
    # PLANNING STRATEGIES
    # =========================================================================
    
    def _plan_calculation(self, task: str) -> ExecutionPlan:
        """Plan a calculation task."""
        plan = ExecutionPlan(task=task, goal="Compute result")
        
        # Extract data from task
        data = self._extract_data(task)
        if data:
            plan.add_step(
                StepType.DEFINE,
                f"Define data",
                f"data = {data}"
            )
        
        # Extract operation
        operation = self._extract_operation(task)
        
        if 'sum' in task.lower() and 'square' in task.lower():
            plan.add_step(
                StepType.COMPUTE,
                "Compute squares",
                "squares = [x**2 for x in data]",
                depends_on=[1] if data else []
            )
            plan.add_step(
                StepType.INVOKE,
                "Sum the squares",
                "result = sum(squares)",
                depends_on=[2]
            )
        elif 'average' in task.lower() or 'mean' in task.lower():
            plan.add_step(
                StepType.INVOKE,
                "Calculate sum",
                "total = sum(data)",
                depends_on=[1] if data else []
            )
            plan.add_step(
                StepType.INVOKE,
                "Calculate count",
                "count = len(data)",
                depends_on=[1] if data else []
            )
            plan.add_step(
                StepType.COMPUTE,
                "Calculate average",
                "result = total / count",
                depends_on=[2, 3]
            )
        elif 'sum' in task.lower():
            plan.add_step(
                StepType.INVOKE,
                "Calculate sum",
                "result = sum(data)",
                depends_on=[1] if data else []
            )
        elif 'product' in task.lower() or 'multiply' in task.lower():
            plan.add_step(
                StepType.COMPUTE,
                "Calculate product",
                "result = 1\nfor x in data: result *= x",
                depends_on=[1] if data else []
            )
        else:
            # Generic calculation
            plan.add_step(
                StepType.COMPUTE,
                "Compute result",
                f"result = {operation}",
                depends_on=[1] if data else []
            )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[len(plan.steps)]
        )
        
        return plan
    
    def _plan_search(self, task: str) -> ExecutionPlan:
        """Plan a search/find task."""
        plan = ExecutionPlan(task=task, goal="Find matching items")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        if 'max' in task.lower() or 'largest' in task.lower() or 'biggest' in task.lower():
            plan.add_step(
                StepType.INVOKE,
                "Find maximum",
                "result = max(data)",
                depends_on=[1] if data else []
            )
        elif 'min' in task.lower() or 'smallest' in task.lower():
            plan.add_step(
                StepType.INVOKE,
                "Find minimum",
                "result = min(data)",
                depends_on=[1] if data else []
            )
        elif 'even' in task.lower():
            plan.add_step(
                StepType.COMPUTE,
                "Find even numbers",
                "result = [x for x in data if x % 2 == 0]",
                depends_on=[1] if data else []
            )
        elif 'odd' in task.lower():
            plan.add_step(
                StepType.COMPUTE,
                "Find odd numbers",
                "result = [x for x in data if x % 2 != 0]",
                depends_on=[1] if data else []
            )
        else:
            plan.add_step(
                StepType.COMPUTE,
                "Search data",
                "result = data",
                depends_on=[1] if data else []
            )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[len(plan.steps)]
        )
        
        return plan
    
    def _plan_sort(self, task: str) -> ExecutionPlan:
        """Plan a sort task."""
        plan = ExecutionPlan(task=task, goal="Sort data")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        reverse = 'descending' in task.lower() or 'reverse' in task.lower()
        plan.add_step(
            StepType.INVOKE,
            f"Sort {'descending' if reverse else 'ascending'}",
            f"result = sorted(data, reverse={reverse})",
            depends_on=[1] if data else []
        )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[2]
        )
        
        return plan
    
    def _plan_filter(self, task: str) -> ExecutionPlan:
        """Plan a filter task."""
        plan = ExecutionPlan(task=task, goal="Filter data")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        # Extract filter condition
        condition = self._extract_condition(task)
        plan.add_step(
            StepType.COMPUTE,
            f"Filter by condition",
            f"result = [x for x in data if {condition}]",
            depends_on=[1] if data else []
        )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[2]
        )
        
        return plan
    
    def _plan_transform(self, task: str) -> ExecutionPlan:
        """Plan a transform task."""
        plan = ExecutionPlan(task=task, goal="Transform data")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        # Extract transformation
        transform = self._extract_transform(task)
        plan.add_step(
            StepType.COMPUTE,
            "Apply transformation",
            f"result = [{transform} for x in data]",
            depends_on=[1] if data else []
        )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[2]
        )
        
        return plan
    
    def _plan_count(self, task: str) -> ExecutionPlan:
        """Plan a count task."""
        plan = ExecutionPlan(task=task, goal="Count items")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        plan.add_step(
            StepType.INVOKE,
            "Count items",
            "result = len(data)",
            depends_on=[1] if data else []
        )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[2]
        )
        
        return plan
    
    def _plan_sum(self, task: str) -> ExecutionPlan:
        """Plan a sum task."""
        return self._plan_calculation(task)
    
    def _plan_average(self, task: str) -> ExecutionPlan:
        """Plan an average task."""
        return self._plan_calculation(task)
    
    def _plan_check(self, task: str) -> ExecutionPlan:
        """Plan a verification task."""
        plan = ExecutionPlan(task=task, goal="Verify condition")
        
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        condition = self._extract_condition(task)
        plan.add_step(
            StepType.VERIFY,
            "Check condition",
            f"result = {condition}",
            depends_on=[1] if data else []
        )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[2]
        )
        
        return plan
    
    def _plan_generic(self, task: str) -> ExecutionPlan:
        """Plan a generic task."""
        plan = ExecutionPlan(task=task, goal="Execute task")
        
        # Try to extract any data
        data = self._extract_data(task)
        if data:
            plan.add_step(StepType.DEFINE, "Define data", f"data = {data}")
        
        # Try to extract an expression
        expr = self._extract_expression(task)
        if expr:
            plan.add_step(
                StepType.COMPUTE,
                "Compute expression",
                f"result = {expr}",
                depends_on=[1] if data else []
            )
        else:
            plan.add_step(
                StepType.COMPUTE,
                "Process data",
                "result = data",
                depends_on=[1] if data else []
            )
        
        plan.add_step(
            StepType.RETURN,
            "Return result",
            "result",
            depends_on=[len(plan.steps)]
        )
        
        return plan
    
    # =========================================================================
    # EXTRACTION HELPERS
    # =========================================================================
    
    def _extract_data(self, task: str) -> str:
        """Extract data (list, number, etc.) from task."""
        # Look for list literals with numbers
        match = re.search(r'\[[\d,\s\.\-]+\]', task)
        if match:
            return match.group(0)
        
        # Look for list literals with letters/strings
        match = re.search(r'\[([a-zA-Z,\s]+)\]', task)
        if match:
            items = [item.strip() for item in match.group(1).split(',')]
            quoted = "', '".join(items)
            return f"['{quoted}']"
        
        # Look for "numbers X, Y, Z" or "values X, Y, Z"
        match = re.search(r'(?:numbers?|values?)\s+([\d,\s]+)', task.lower())
        if match:
            nums = re.findall(r'\d+', match.group(1))
            return f"[{', '.join(nums)}]"
        
        # Look for range
        match = re.search(r'from\s+(\d+)\s+to\s+(\d+)', task.lower())
        if match:
            return f"list(range({match.group(1)}, {int(match.group(2)) + 1}))"
        
        return ""
    
    def _extract_operation(self, task: str) -> str:
        """Extract operation from task."""
        task_lower = task.lower()
        
        if 'sum' in task_lower:
            return 'sum(data)'
        if 'average' in task_lower or 'mean' in task_lower:
            return 'sum(data) / len(data)'
        if 'product' in task_lower:
            return 'functools.reduce(lambda a, b: a * b, data)'
        if 'max' in task_lower:
            return 'max(data)'
        if 'min' in task_lower:
            return 'min(data)'
        
        return 'data'
    
    def _extract_condition(self, task: str) -> str:
        """Extract filter condition from task."""
        task_lower = task.lower()
        
        if 'even' in task_lower:
            return 'x % 2 == 0'
        if 'odd' in task_lower:
            return 'x % 2 != 0'
        if 'positive' in task_lower:
            return 'x > 0'
        if 'negative' in task_lower:
            return 'x < 0'
        
        # Look for "greater than X" or "> X"
        match = re.search(r'(?:greater than|>)\s*(\d+)', task_lower)
        if match:
            return f'x > {match.group(1)}'
        
        match = re.search(r'(?:less than|<)\s*(\d+)', task_lower)
        if match:
            return f'x < {match.group(1)}'
        
        return 'True'
    
    def _extract_transform(self, task: str) -> str:
        """Extract transformation from task."""
        task_lower = task.lower()
        
        if 'square' in task_lower:
            return 'x**2'
        if 'cube' in task_lower:
            return 'x**3'
        if 'doubl' in task_lower:  # double, doubling
            return 'x * 2'
        if 'tripl' in task_lower:  # triple, tripling
            return 'x * 3'
        if 'negate' in task_lower:
            return '-x'
        if 'half' in task_lower or 'halv' in task_lower:
            return 'x / 2'
        
        return 'x'
    
    def _extract_expression(self, task: str) -> str:
        """Extract a mathematical expression from task."""
        # Look for explicit expressions
        match = re.search(r'(\d+\s*[\+\-\*\/\%\*\*]+\s*\d+)', task)
        if match:
            return match.group(1)
        
        return ""


# =============================================================================
# TEST
# =============================================================================

def test_planner():
    """Test the planner."""
    print("=== Planner Test ===\n")
    
    planner = Planner()
    
    tasks = [
        "Calculate the sum of [1, 2, 3, 4, 5]",
        "Find the maximum in [10, 5, 8, 3, 12]",
        "Calculate the average of [2, 4, 6, 8, 10]",
        "Sort [5, 2, 8, 1, 9] in ascending order",
        "Find even numbers in [1, 2, 3, 4, 5, 6, 7, 8]",
        "Calculate the sum of squares of [1, 2, 3]",
    ]
    
    for task in tasks:
        print(f"Task: {task}")
        plan = planner.plan(task)
        print(f"Plan created with {len(plan.steps)} steps")
        
        result = planner.execute(plan, verbose=False)
        print(f"Result: {result.final_result}")
        print(f"Success: {result.success}")
        print()


if __name__ == '__main__':
    test_planner()
