"""
Code Executor for TruthSpace LCM

Executes generated Python and Bash code safely, captures output,
validates results, and handles errors intelligently.

Key Features:
1. Safe execution in isolated environment
2. Output capture and validation
3. Error detection and diagnosis
4. Timeout handling
5. Resource cleanup
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    validation_passed: bool = True
    validation_message: str = ""
    error_diagnosis: str = ""


@dataclass
class ValidationRule:
    """A rule for validating execution output."""
    name: str
    check_type: str  # "contains", "not_contains", "regex", "file_exists", "exit_code"
    expected: Any
    message: str = ""


class CodeExecutor:
    """
    Executes generated code safely and validates output.
    """
    
    def __init__(self, working_dir: str = None, timeout: int = 30):
        self.timeout = timeout
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="truthspace_exec_")
        self._ensure_working_dir()
        self._init_error_patterns()
    
    def _ensure_working_dir(self):
        """Ensure working directory exists."""
        os.makedirs(self.working_dir, exist_ok=True)
    
    def _init_error_patterns(self):
        """Initialize patterns for diagnosing common errors."""
        self.error_patterns = {
            # Python errors
            r"ModuleNotFoundError: No module named '(\w+)'": 
                "Missing Python module '{0}'. Install with: pip install {0}",
            r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'":
                "File not found: '{0}'. Check if the file exists or create it first.",
            r"PermissionError: \[Errno 13\] Permission denied: '([^']+)'":
                "Permission denied for '{0}'. Check file permissions.",
            r"SyntaxError: (.+)":
                "Python syntax error: {0}",
            r"IndentationError: (.+)":
                "Python indentation error: {0}",
            r"NameError: name '(\w+)' is not defined":
                "Variable '{0}' is not defined. Check spelling or define it first.",
            r"TypeError: (.+)":
                "Type error: {0}",
            r"KeyError: '?(\w+)'?":
                "Key '{0}' not found in dictionary.",
            r"IndexError: (.+)":
                "Index error: {0}",
            r"ConnectionError|requests\.exceptions":
                "Network connection error. Check internet connection or URL.",
            r"JSONDecodeError":
                "Invalid JSON format. Check the data being parsed.",
            
            # Bash errors
            r"command not found":
                "Command not found. Check if the command is installed.",
            r"No such file or directory":
                "File or directory not found. Check the path.",
            r"Permission denied":
                "Permission denied. Try with sudo or check permissions.",
            r"mkdir: cannot create directory '([^']+)': File exists":
                "Directory '{0}' already exists. Use mkdir -p to ignore.",
        }
    
    def _diagnose_error(self, stderr: str, stdout: str = "") -> str:
        """Diagnose error from stderr/stdout."""
        combined = stderr + "\n" + stdout
        
        for pattern, diagnosis_template in self.error_patterns.items():
            match = re.search(pattern, combined)
            if match:
                groups = match.groups()
                if groups:
                    return diagnosis_template.format(*groups)
                return diagnosis_template
        
        # Generic diagnosis
        if stderr:
            first_line = stderr.strip().split('\n')[0]
            return f"Error: {first_line[:100]}"
        
        return "Unknown error occurred"
    
    def _track_files(self, before: set, after: set) -> Tuple[List[str], List[str]]:
        """Track created and modified files."""
        created = list(after - before)
        # For simplicity, we're not tracking modifications in this version
        return created, []
    
    def _get_directory_state(self, path: str) -> set:
        """Get set of files in directory."""
        files = set()
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                files.add(os.path.join(root, f))
        return files
    
    def execute_python(self, code: str, 
                       validation_rules: List[ValidationRule] = None) -> ExecutionResult:
        """
        Execute Python code and capture results.
        
        Args:
            code: Python code to execute
            validation_rules: Optional rules to validate output
            
        Returns:
            ExecutionResult with status and output
        """
        # Write code to temp file
        script_path = os.path.join(self.working_dir, "script.py")
        with open(script_path, 'w') as f:
            f.write(code)
        
        # Track files before execution
        files_before = self._get_directory_state(self.working_dir)
        
        # Execute
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir
            )
            execution_time = time.time() - start_time
            
            # Track files after execution
            files_after = self._get_directory_state(self.working_dir)
            created, modified = self._track_files(files_before, files_after)
            
            # Determine status
            if result.returncode == 0:
                status = ExecutionStatus.SUCCESS
                error_diagnosis = ""
            else:
                status = ExecutionStatus.FAILED
                error_diagnosis = self._diagnose_error(result.stderr, result.stdout)
            
            exec_result = ExecutionResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
                files_created=created,
                files_modified=modified,
                error_diagnosis=error_diagnosis
            )
            
            # Validate if rules provided
            if validation_rules and status == ExecutionStatus.SUCCESS:
                exec_result = self._validate(exec_result, validation_rules)
            
            return exec_result
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                error_diagnosis="Code took too long to execute. Consider optimizing or increasing timeout."
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=str(e),
                error_diagnosis=f"Execution error: {str(e)}"
            )
    
    def execute_bash(self, command: str,
                     validation_rules: List[ValidationRule] = None) -> ExecutionResult:
        """
        Execute Bash command and capture results.
        
        Args:
            command: Bash command to execute
            validation_rules: Optional rules to validate output
            
        Returns:
            ExecutionResult with status and output
        """
        # Track files before execution
        files_before = self._get_directory_state(self.working_dir)
        
        # Execute
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir
            )
            execution_time = time.time() - start_time
            
            # Track files after execution
            files_after = self._get_directory_state(self.working_dir)
            created, modified = self._track_files(files_before, files_after)
            
            # Determine status
            if result.returncode == 0:
                status = ExecutionStatus.SUCCESS
                error_diagnosis = ""
            else:
                status = ExecutionStatus.FAILED
                error_diagnosis = self._diagnose_error(result.stderr, result.stdout)
            
            exec_result = ExecutionResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
                files_created=created,
                files_modified=modified,
                error_diagnosis=error_diagnosis
            )
            
            # Validate if rules provided
            if validation_rules and status == ExecutionStatus.SUCCESS:
                exec_result = self._validate(exec_result, validation_rules)
            
            return exec_result
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                error_diagnosis="Command took too long to execute."
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=str(e),
                error_diagnosis=f"Execution error: {str(e)}"
            )
    
    def _validate(self, result: ExecutionResult, 
                  rules: List[ValidationRule]) -> ExecutionResult:
        """Apply validation rules to execution result."""
        for rule in rules:
            passed = False
            
            if rule.check_type == "contains":
                passed = rule.expected in result.stdout
            elif rule.check_type == "not_contains":
                passed = rule.expected not in result.stdout
            elif rule.check_type == "regex":
                passed = bool(re.search(rule.expected, result.stdout))
            elif rule.check_type == "file_exists":
                file_path = os.path.join(self.working_dir, rule.expected)
                passed = os.path.exists(file_path)
            elif rule.check_type == "exit_code":
                passed = result.return_code == rule.expected
            
            if not passed:
                result.validation_passed = False
                result.validation_message = rule.message or f"Validation failed: {rule.name}"
                result.status = ExecutionStatus.VALIDATION_FAILED
                return result
        
        result.validation_passed = True
        result.validation_message = "All validations passed"
        return result
    
    def cleanup(self):
        """Clean up working directory."""
        if self.working_dir and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir, ignore_errors=True)
    
    def reset(self):
        """Reset working directory for new execution."""
        self.cleanup()
        self.working_dir = tempfile.mkdtemp(prefix="truthspace_exec_")
        self._ensure_working_dir()


class ExecutionKnowledge:
    """
    Knowledge base for execution patterns, common errors, and fixes.
    """
    
    # Common validation patterns
    VALIDATIONS = {
        "file_created": lambda filename: ValidationRule(
            name=f"file_{filename}_exists",
            check_type="file_exists",
            expected=filename,
            message=f"Expected file '{filename}' was not created"
        ),
        "output_contains": lambda text: ValidationRule(
            name=f"output_contains_{text[:20]}",
            check_type="contains",
            expected=text,
            message=f"Expected output to contain: {text}"
        ),
        "no_error": ValidationRule(
            name="no_error",
            check_type="not_contains",
            expected="Error",
            message="Output should not contain 'Error'"
        ),
        "success_exit": ValidationRule(
            name="success_exit",
            check_type="exit_code",
            expected=0,
            message="Expected exit code 0"
        ),
    }
    
    # Common fixes for errors
    FIXES = {
        "ModuleNotFoundError": {
            "requests": "pip install requests",
            "bs4": "pip install beautifulsoup4",
            "numpy": "pip install numpy",
            "pandas": "pip install pandas",
        },
        "FileNotFoundError": {
            "suggestion": "Create the file first or check the path"
        },
        "PermissionError": {
            "suggestion": "Check file permissions with ls -la, use chmod to fix"
        },
    }
    
    @classmethod
    def get_validation_for_task(cls, task_type: str, **kwargs) -> List[ValidationRule]:
        """Get appropriate validation rules for a task type."""
        rules = []
        
        if task_type == "create_file":
            filename = kwargs.get("filename", "")
            if filename:
                rules.append(cls.VALIDATIONS["file_created"](filename))
        
        elif task_type == "create_directory":
            dirname = kwargs.get("dirname", "")
            if dirname:
                rules.append(cls.VALIDATIONS["file_created"](dirname))
        
        elif task_type == "fetch_url":
            rules.append(cls.VALIDATIONS["no_error"])
            rules.append(cls.VALIDATIONS["success_exit"])
        
        elif task_type == "write_json":
            filename = kwargs.get("filename", "")
            if filename:
                rules.append(cls.VALIDATIONS["file_created"](filename))
        
        return rules
    
    @classmethod
    def suggest_fix(cls, error_type: str, details: str = "") -> str:
        """Suggest a fix for a common error."""
        if error_type in cls.FIXES:
            fixes = cls.FIXES[error_type]
            if details in fixes:
                return fixes[details]
            if "suggestion" in fixes:
                return fixes["suggestion"]
        return "Check the error message and fix the underlying issue"


def demonstrate():
    """Demonstrate the executor."""
    
    print("=" * 70)
    print("CODE EXECUTOR: Run and Validate Generated Code")
    print("=" * 70)
    print()
    
    executor = CodeExecutor()
    
    # Test 1: Simple Python
    print("TEST 1: Simple Python (Hello World)")
    print("-" * 40)
    code = 'print("Hello, World!")'
    result = executor.execute_python(
        code,
        validation_rules=[
            ValidationRule("has_hello", "contains", "Hello", "Should print Hello")
        ]
    )
    print(f"Status: {result.status.value}")
    print(f"Output: {result.stdout.strip()}")
    print(f"Validation: {result.validation_message}")
    print()
    
    # Test 2: Python with file creation
    print("TEST 2: Python File Creation")
    print("-" * 40)
    code = '''
import json
data = {"name": "test", "value": 42}
with open("output.json", "w") as f:
    json.dump(data, f)
print("File created!")
'''
    result = executor.execute_python(
        code,
        validation_rules=[
            ExecutionKnowledge.VALIDATIONS["file_created"]("output.json"),
            ValidationRule("has_created_msg", "contains", "created", "Should confirm creation")
        ]
    )
    print(f"Status: {result.status.value}")
    print(f"Output: {result.stdout.strip()}")
    print(f"Files created: {result.files_created}")
    print(f"Validation: {result.validation_message}")
    print()
    
    # Reset for next test
    executor.reset()
    
    # Test 3: Bash command
    print("TEST 3: Bash Directory Creation")
    print("-" * 40)
    result = executor.execute_bash(
        "mkdir -p myproject/src && touch myproject/src/main.py && ls -la myproject/src"
    )
    print(f"Status: {result.status.value}")
    print(f"Output: {result.stdout.strip()}")
    print(f"Files created: {[os.path.basename(f) for f in result.files_created]}")
    print()
    
    # Test 4: Error handling
    print("TEST 4: Error Handling (Missing Module)")
    print("-" * 40)
    code = 'import nonexistent_module'
    result = executor.execute_python(code)
    print(f"Status: {result.status.value}")
    print(f"Error diagnosis: {result.error_diagnosis}")
    print()
    
    # Test 5: Validation failure
    print("TEST 5: Validation Failure")
    print("-" * 40)
    code = 'print("Goodbye")'
    result = executor.execute_python(
        code,
        validation_rules=[
            ValidationRule("has_hello", "contains", "Hello", "Expected 'Hello' in output")
        ]
    )
    print(f"Status: {result.status.value}")
    print(f"Output: {result.stdout.strip()}")
    print(f"Validation: {result.validation_message}")
    print()
    
    # Cleanup
    executor.cleanup()
    print("=" * 70)
    print("All tests completed!")


if __name__ == "__main__":
    demonstrate()
