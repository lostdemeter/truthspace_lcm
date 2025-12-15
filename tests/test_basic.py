"""
Basic tests for TruthSpace LCM

Tests core functionality:
- Knowledge manager operations
- Code generation
- Bash generation
- Task planning
- Execution
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain
from truthspace_lcm.core.code_generator import CodeGenerator
from truthspace_lcm.core.bash_generator import BashGenerator
from truthspace_lcm.core.task_planner import TaskPlanner, StepStatus
from truthspace_lcm.core.executor import CodeExecutor, ExecutionStatus


def test_knowledge_manager():
    """Test knowledge manager basic operations."""
    print("Testing KnowledgeManager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = KnowledgeManager(storage_dir=tmpdir)
        
        # Create entry
        entry = manager.create(
            name="test_function",
            domain=KnowledgeDomain.PROGRAMMING,
            entry_type="function",
            description="A test function",
            keywords=["test", "function", "example"]
        )
        
        assert entry is not None
        assert entry.name == "test_function"
        
        # Query
        results = manager.query(["test", "function"])
        assert len(results) > 0
        
        print("  ✅ KnowledgeManager tests passed")


def test_code_generator():
    """Test Python code generation."""
    print("Testing CodeGenerator...")
    
    generator = CodeGenerator()
    
    # Test simple request
    result = generator.generate("print hello world")
    assert result.success
    assert "print" in result.code
    
    # Test file operation
    result = generator.generate("read a json file")
    assert result.success
    assert "json" in result.code.lower() or "open" in result.code.lower()
    
    print("  ✅ CodeGenerator tests passed")


def test_bash_generator():
    """Test Bash command generation."""
    print("Testing BashGenerator...")
    
    generator = BashGenerator()
    
    # Test directory creation
    result = generator.generate("create a directory called test")
    assert result.success
    assert "mkdir" in result.command
    
    # Test file listing
    result = generator.generate("list all files")
    assert result.success
    assert "ls" in result.command
    
    print("  ✅ BashGenerator tests passed")


def test_task_planner():
    """Test task planning."""
    print("Testing TaskPlanner...")
    
    planner = TaskPlanner()
    
    # Test single step
    plan = planner.plan("list files")
    assert len(plan.steps) == 1
    
    # Test multi-step
    plan = planner.plan("create a python project called myapp")
    assert len(plan.steps) > 1
    
    print("  ✅ TaskPlanner tests passed")


def test_executor():
    """Test code execution."""
    print("Testing CodeExecutor...")
    
    executor = CodeExecutor()
    
    try:
        # Test Python execution
        result = executor.execute_python('print("Hello, World!")')
        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello" in result.stdout
        
        # Test Bash execution
        result = executor.execute_bash("echo 'test'")
        assert result.status == ExecutionStatus.SUCCESS
        assert "test" in result.stdout
        
        print("  ✅ CodeExecutor tests passed")
    finally:
        executor.cleanup()


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("Testing End-to-End...")
    
    planner = TaskPlanner()
    
    try:
        # Plan and execute
        plan = planner.plan("create a directory called testdir")
        plan = planner.execute_plan(plan, dry_run=False)
        
        assert plan.status == StepStatus.COMPLETED
        
        # Verify directory was created
        testdir = os.path.join(planner.executor.working_dir, "testdir")
        assert os.path.isdir(testdir)
        
        print("  ✅ End-to-End tests passed")
    finally:
        planner.executor.cleanup()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_knowledge_manager,
        test_code_generator,
        test_bash_generator,
        test_task_planner,
        test_executor,
        test_end_to_end,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
