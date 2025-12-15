"""
Task Planner for TruthSpace LCM

Decomposes complex natural language tasks into executable steps,
then orchestrates the code/bash generators to complete each step.

Key Features:
1. Task decomposition - break complex requests into atomic steps
2. Dependency tracking - understand which steps depend on others
3. Execution orchestration - run steps in correct order
4. Result aggregation - combine outputs into final result
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from truthspace_lcm.core.code_generator import CodeGenerator
from truthspace_lcm.core.bash_generator import BashGenerator
from truthspace_lcm.core.executor import CodeExecutor, ExecutionResult, ExecutionStatus, ValidationRule


class StepType(Enum):
    """Type of step to execute."""
    PYTHON = "python"
    BASH = "bash"
    COMPOSITE = "composite"  # Contains sub-steps


class StepStatus(Enum):
    """Status of a step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """A single step in a task plan."""
    id: int
    description: str
    step_type: StepType
    status: StepStatus = StepStatus.PENDING
    depends_on: List[int] = field(default_factory=list)
    generated_code: str = ""
    output: str = ""
    error: str = ""


@dataclass
class TaskPlan:
    """A complete task plan with multiple steps."""
    original_request: str
    goal: str
    steps: List[TaskStep]
    current_step: int = 0
    status: StepStatus = StepStatus.PENDING
    final_output: str = ""


class TaskPlanner:
    """
    Decomposes complex tasks into executable steps and orchestrates execution.
    """
    
    def __init__(self, storage_dir: str = None, working_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        self.code_generator = CodeGenerator(storage_dir)
        self.bash_generator = BashGenerator(storage_dir)
        self.executor = CodeExecutor(working_dir=working_dir)
        self._init_task_patterns()
    
    def _init_task_patterns(self):
        """Initialize patterns for recognizing task types."""
        self.task_patterns = {
            # Project setup patterns
            "create_project": [
                r"(?:create|setup|initialize|start)\s+(?:a\s+)?(?:new\s+)?(?:python\s+)?project",
                r"(?:set\s*up|scaffold)\s+(?:a\s+)?(?:new\s+)?(?:project|app|application)",
            ],
            # Web scraping patterns
            "scrape_and_save": [
                r"scrape\s+.+\s+(?:and|then)\s+save",
                r"(?:get|fetch|download)\s+.+\s+(?:and|then)\s+(?:save|store|write)",
                r"extract\s+.+\s+(?:and|then)\s+(?:save|export)",
            ],
            # Data processing patterns
            "process_data": [
                r"(?:read|load)\s+.+\s+(?:and|then)\s+(?:process|transform|convert)",
                r"(?:parse|extract)\s+.+\s+(?:and|then)\s+(?:save|write|output)",
            ],
            # File organization patterns
            "organize_files": [
                r"(?:organize|sort|arrange)\s+(?:the\s+)?files",
                r"(?:create|setup)\s+(?:a\s+)?(?:folder|directory)\s+structure",
                r"move\s+.+\s+(?:into|to)\s+(?:separate\s+)?(?:folders|directories)",
            ],
            # API interaction patterns
            "api_workflow": [
                r"(?:call|use|hit)\s+(?:the\s+)?api\s+(?:and|then)",
                r"(?:fetch|get)\s+(?:data\s+)?from\s+(?:the\s+)?api\s+(?:and|then)",
            ],
            # Backup patterns
            "backup_task": [
                r"(?:backup|archive)\s+(?:the\s+)?(?:files|folder|directory|project)",
                r"(?:create|make)\s+(?:a\s+)?backup",
            ],
        }
        
        # Patterns that indicate multi-step tasks
        self.multi_step_indicators = [
            r"\b(?:and\s+then|then|after\s+that|next|finally|also)\b",
            r"\b(?:first|second|third|lastly)\b",
            r"\b(?:step\s+\d+|steps?)\b",
            r",\s*(?:and\s+)?(?:then\s+)?(?:also\s+)?",
        ]
    
    def _is_multi_step(self, request: str) -> bool:
        """Determine if request requires multiple steps."""
        request_lower = request.lower()
        
        # Check for explicit multi-step indicators
        for pattern in self.multi_step_indicators:
            if re.search(pattern, request_lower):
                return True
        
        # Check for task patterns that are inherently multi-step
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    return True
        
        return False
    
    def _detect_task_type(self, request: str) -> Optional[str]:
        """Detect the type of task from the request."""
        request_lower = request.lower()
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    return task_type
        
        return None
    
    def _determine_step_type(self, step_desc: str) -> StepType:
        """Determine if a step should use Python or Bash."""
        step_lower = step_desc.lower()
        
        # Bash indicators
        bash_indicators = [
            r"(?:create|make)\s+(?:a\s+)?(?:directory|folder)",
            r"(?:create|touch)\s+(?:a\s+)?(?:empty\s+)?file",
            r"(?:move|copy|delete|remove)\s+(?:the\s+)?(?:file|folder|directory)",
            r"(?:list|show)\s+(?:the\s+)?(?:files|contents)",
            r"(?:chmod|chown|permissions)",
            r"(?:compress|archive|tar|zip)",
            r"(?:extract|unzip|untar)",
            r"(?:mkdir|rmdir|rm|cp|mv|ls|cd|pwd)",
        ]
        
        for pattern in bash_indicators:
            if re.search(pattern, step_lower):
                return StepType.BASH
        
        # Python indicators
        python_indicators = [
            r"(?:fetch|download|get)\s+(?:from\s+)?(?:url|http|api)",
            r"(?:parse|process|transform)\s+(?:the\s+)?(?:data|json|html)",
            r"(?:read|write)\s+(?:the\s+)?(?:json|csv|data)",
            r"(?:scrape|extract)\s+(?:from\s+)?(?:webpage|website|html)",
            r"(?:loop|iterate|for\s+each)",
            r"(?:calculate|compute|process)",
        ]
        
        for pattern in python_indicators:
            if re.search(pattern, step_lower):
                return StepType.PYTHON
        
        # Default to bash for simple file operations, python for data processing
        if any(word in step_lower for word in ["file", "folder", "directory"]):
            return StepType.BASH
        
        return StepType.PYTHON
    
    def _decompose_create_project(self, request: str) -> List[TaskStep]:
        """Decompose a project creation task."""
        # Extract project name
        name_match = re.search(r'(?:called|named)\s+(\w+)', request)
        project_name = name_match.group(1) if name_match else "myproject"
        
        steps = [
            TaskStep(
                id=1,
                description=f"create a directory called {project_name}",
                step_type=StepType.BASH,
            ),
            TaskStep(
                id=2,
                description=f"create a directory called {project_name}/src",
                step_type=StepType.BASH,
                depends_on=[1],
            ),
            TaskStep(
                id=3,
                description=f"create a file called {project_name}/src/__init__.py",
                step_type=StepType.BASH,
                depends_on=[2],
            ),
            TaskStep(
                id=4,
                description=f"create a file called {project_name}/src/main.py",
                step_type=StepType.BASH,
                depends_on=[2],
            ),
            TaskStep(
                id=5,
                description=f"create a file called {project_name}/README.md",
                step_type=StepType.BASH,
                depends_on=[1],
            ),
        ]
        
        return steps
    
    def _decompose_scrape_and_save(self, request: str) -> List[TaskStep]:
        """Decompose a scrape and save task."""
        # Extract URL
        url_match = re.search(r'(https?://[^\s]+)', request)
        url = url_match.group(1) if url_match else "https://example.com"
        
        # Extract output filename
        file_match = re.search(r'(?:save|write|store)\s+(?:to|in|as)\s+(\S+)', request)
        output_file = file_match.group(1) if file_match else "output.json"
        
        steps = [
            TaskStep(
                id=1,
                description=f"Fetch webpage from {url}",
                step_type=StepType.PYTHON,
            ),
            TaskStep(
                id=2,
                description="Parse HTML and extract data",
                step_type=StepType.PYTHON,
                depends_on=[1],
            ),
            TaskStep(
                id=3,
                description=f"Save extracted data to {output_file}",
                step_type=StepType.PYTHON,
                depends_on=[2],
            ),
        ]
        
        return steps
    
    def _decompose_backup_task(self, request: str) -> List[TaskStep]:
        """Decompose a backup task."""
        # Extract target
        target_match = re.search(r'(?:backup|archive)\s+(?:the\s+)?(\w+)', request)
        target = target_match.group(1) if target_match else "project"
        
        steps = [
            TaskStep(
                id=1,
                description="create a directory called backups",
                step_type=StepType.BASH,
            ),
            TaskStep(
                id=2,
                description=f"compress the {target} folder",
                step_type=StepType.BASH,
                depends_on=[1],
            ),
            TaskStep(
                id=3,
                description="list all files in backups",
                step_type=StepType.BASH,
                depends_on=[2],
            ),
        ]
        
        return steps
    
    def _decompose_generic(self, request: str) -> List[TaskStep]:
        """Decompose a generic multi-step task by splitting on conjunctions."""
        # Split on "and then", "then", commas, etc.
        parts = re.split(r'\s+(?:and\s+)?then\s+|\s*,\s*(?:and\s+)?(?:then\s+)?', request)
        parts = [p.strip() for p in parts if p.strip()]
        
        steps = []
        for i, part in enumerate(parts, 1):
            step_type = self._determine_step_type(part)
            steps.append(TaskStep(
                id=i,
                description=part,
                step_type=step_type,
                depends_on=[i-1] if i > 1 else [],
            ))
        
        return steps
    
    def plan(self, request: str) -> TaskPlan:
        """
        Create a task plan from a natural language request.
        
        Args:
            request: Natural language description of the task
            
        Returns:
            TaskPlan with decomposed steps
        """
        task_type = self._detect_task_type(request)
        
        # Decompose based on task type
        if task_type == "create_project":
            steps = self._decompose_create_project(request)
            goal = "Set up a new Python project structure"
        elif task_type == "scrape_and_save":
            steps = self._decompose_scrape_and_save(request)
            goal = "Scrape data from web and save to file"
        elif task_type == "backup_task":
            steps = self._decompose_backup_task(request)
            goal = "Create a backup archive"
        elif self._is_multi_step(request):
            steps = self._decompose_generic(request)
            goal = "Complete multi-step task"
        else:
            # Single step task
            step_type = self._determine_step_type(request)
            steps = [TaskStep(
                id=1,
                description=request,
                step_type=step_type,
            )]
            goal = "Complete single-step task"
        
        return TaskPlan(
            original_request=request,
            goal=goal,
            steps=steps,
        )
    
    def generate_step_code(self, step: TaskStep) -> str:
        """Generate code for a single step."""
        if step.step_type == StepType.BASH:
            result = self.bash_generator.generate(step.description)
            return result.command
        else:
            result = self.code_generator.generate(step.description)
            return result.code
    
    def execute_plan(self, plan: TaskPlan, dry_run: bool = True) -> TaskPlan:
        """
        Execute a task plan.
        
        Args:
            plan: The task plan to execute
            dry_run: If True, only generate code without executing
            
        Returns:
            Updated TaskPlan with results
        """
        plan.status = StepStatus.IN_PROGRESS
        
        for step in plan.steps:
            # Check dependencies
            deps_met = all(
                plan.steps[dep_id - 1].status == StepStatus.COMPLETED
                for dep_id in step.depends_on
            )
            
            if not deps_met:
                step.status = StepStatus.SKIPPED
                step.error = "Dependencies not met"
                continue
            
            step.status = StepStatus.IN_PROGRESS
            plan.current_step = step.id
            
            try:
                # Generate code for this step
                step.generated_code = self.generate_step_code(step)
                
                if dry_run:
                    step.output = f"[DRY RUN] Would execute: {step.generated_code[:100]}..."
                    step.status = StepStatus.COMPLETED
                else:
                    # Actually execute the code
                    if step.step_type == StepType.BASH:
                        exec_result = self.executor.execute_bash(step.generated_code)
                    else:
                        exec_result = self.executor.execute_python(step.generated_code)
                    
                    step.output = exec_result.stdout
                    step.error = exec_result.stderr if exec_result.status != ExecutionStatus.SUCCESS else ""
                    
                    if exec_result.status == ExecutionStatus.SUCCESS:
                        step.status = StepStatus.COMPLETED
                    else:
                        step.status = StepStatus.FAILED
                        if exec_result.error_diagnosis:
                            step.error = exec_result.error_diagnosis
                    
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
        
        # Determine overall status
        if all(s.status == StepStatus.COMPLETED for s in plan.steps):
            plan.status = StepStatus.COMPLETED
        elif any(s.status == StepStatus.FAILED for s in plan.steps):
            plan.status = StepStatus.FAILED
        
        return plan
    
    def format_plan(self, plan: TaskPlan) -> str:
        """Format a task plan for display."""
        lines = [
            "=" * 60,
            f"TASK PLAN: {plan.goal}",
            "=" * 60,
            f"Original request: {plan.original_request}",
            f"Status: {plan.status.value}",
            "",
            "STEPS:",
            "-" * 40,
        ]
        
        for step in plan.steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "?")
            
            deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
            lines.append(f"{status_icon} Step {step.id} [{step.step_type.value}]: {step.description}{deps}")
            
            if step.generated_code:
                code_preview = step.generated_code.split('\n')[0][:60]
                lines.append(f"   Code: {code_preview}...")
            
            if step.error:
                lines.append(f"   Error: {step.error}")
        
        lines.append("-" * 40)
        return "\n".join(lines)


def demonstrate():
    """Demonstrate the task planner with actual execution."""
    
    print("=" * 70)
    print("TASK PLANNER: Execute Multi-Step Tasks")
    print("=" * 70)
    print()
    
    planner = TaskPlanner()
    
    # Test 1: Create project structure (actually execute)
    print("=" * 60)
    print("TEST 1: Create Python Project (ACTUAL EXECUTION)")
    print("=" * 60)
    
    plan = planner.plan("create a new python project called myapp")
    plan = planner.execute_plan(plan, dry_run=False)
    print(planner.format_plan(plan))
    
    # Verify files were created
    print("\nVERIFICATION:")
    import os
    for step in plan.steps:
        if step.status == StepStatus.COMPLETED:
            print(f"  ✅ Step {step.id} completed")
        else:
            print(f"  ❌ Step {step.id} failed: {step.error}")
    
    # List created files
    print(f"\nFiles in working directory ({planner.executor.working_dir}):")
    for root, dirs, files in os.walk(planner.executor.working_dir):
        level = root.replace(planner.executor.working_dir, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # Reset for next test
    planner.executor.reset()
    
    # Test 2: Multi-step bash commands
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Step Bash (ACTUAL EXECUTION)")
    print("=" * 60)
    
    plan = planner.plan("create a directory called data, then create a file called config.json inside it")
    plan = planner.execute_plan(plan, dry_run=False)
    print(planner.format_plan(plan))
    
    # Verify
    print("\nVERIFICATION:")
    data_dir = os.path.join(planner.executor.working_dir, "data")
    config_file = os.path.join(planner.executor.working_dir, "config.json")
    
    if os.path.isdir(data_dir):
        print(f"  ✅ Directory 'data' exists")
    else:
        print(f"  ❌ Directory 'data' not found")
    
    if os.path.isfile(config_file):
        print(f"  ✅ File 'config.json' exists")
    else:
        print(f"  ❌ File 'config.json' not found")
    
    # Reset for next test
    planner.executor.reset()
    
    # Test 3: Simple single-step
    print("\n" + "=" * 60)
    print("TEST 3: Single Step (list files)")
    print("=" * 60)
    
    # Create some files first
    planner.executor.execute_bash("touch file1.txt file2.txt file3.txt")
    
    plan = planner.plan("list all files in the current directory")
    plan = planner.execute_plan(plan, dry_run=False)
    print(planner.format_plan(plan))
    
    print("\nOUTPUT:")
    for step in plan.steps:
        if step.output:
            print(step.output[:500])
    
    # Cleanup
    planner.executor.cleanup()
    print("\n" + "=" * 70)
    print("All tests completed!")


def demonstrate_dry_run():
    """Demonstrate dry run mode."""
    print("=" * 70)
    print("TASK PLANNER: Dry Run Mode")
    print("=" * 70)
    
    planner = TaskPlanner()
    
    requests = [
        "create a new python project called webapp",
        "backup the project folder",
        "list all files in the current directory",
    ]
    
    for request in requests:
        print(f"\n{'='*60}")
        print(f"REQUEST: {request}")
        print("=" * 60)
        
        plan = planner.plan(request)
        plan = planner.execute_plan(plan, dry_run=True)
        print(planner.format_plan(plan))
        
        print("\nGENERATED CODE:")
        for step in plan.steps:
            print(f"\n--- Step {step.id} ({step.step_type.value}) ---")
            print(step.generated_code)


if __name__ == "__main__":
    demonstrate()
