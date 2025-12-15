#!/usr/bin/env python3
"""
Basic Usage Examples for TruthSpace LCM

Demonstrates the main features of the system.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.task_planner import TaskPlanner, StepStatus


def example_create_project():
    """Example: Create a Python project structure."""
    print("=" * 60)
    print("Example: Create Python Project")
    print("=" * 60)
    
    planner = TaskPlanner()
    
    plan = planner.plan("create a python project called myapp")
    plan = planner.execute_plan(plan, dry_run=False)
    
    print(f"\nStatus: {plan.status.value}")
    print(f"Working directory: {planner.executor.working_dir}")
    
    for step in plan.steps:
        status = "✅" if step.status == StepStatus.COMPLETED else "❌"
        print(f"  {status} {step.description}")
        print(f"      Code: {step.generated_code}")
    
    planner.executor.cleanup()


def example_file_operations():
    """Example: File and directory operations."""
    print("\n" + "=" * 60)
    print("Example: File Operations")
    print("=" * 60)
    
    planner = TaskPlanner()
    
    requests = [
        "create a directory called data",
        "create a file called config.json",
        "list all files in the current directory",
    ]
    
    for request in requests:
        print(f"\nRequest: {request}")
        plan = planner.plan(request)
        plan = planner.execute_plan(plan, dry_run=False)
        
        for step in plan.steps:
            print(f"  Generated: {step.generated_code}")
            if step.output:
                print(f"  Output: {step.output[:100]}...")
    
    planner.executor.cleanup()


def example_dry_run():
    """Example: Dry run mode (generate without executing)."""
    print("\n" + "=" * 60)
    print("Example: Dry Run Mode")
    print("=" * 60)
    
    planner = TaskPlanner()
    
    plan = planner.plan("backup the project folder")
    plan = planner.execute_plan(plan, dry_run=True)
    
    print("\nGenerated code (not executed):")
    for step in plan.steps:
        print(f"  Step {step.id}: {step.generated_code}")
    
    planner.executor.cleanup()


if __name__ == "__main__":
    example_create_project()
    example_file_operations()
    example_dry_run()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
