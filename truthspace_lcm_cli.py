#!/usr/bin/env python3
"""
TruthSpace LCM - Unified Command Line Interface

A complete natural language to code system that:
1. Understands natural language requests
2. Plans multi-step tasks
3. Generates Python or Bash code
4. Executes code safely
5. Validates output

Usage:
    python truthspace_lcm_cli.py "create a python project called myapp"
    python truthspace_lcm_cli.py --interactive
    python truthspace_lcm_cli.py --dry-run "backup the project folder"
"""

import os
import sys
import argparse
from typing import Optional

from truthspace_lcm.core.task_planner import TaskPlanner, TaskPlan, StepStatus, StepType
from truthspace_lcm.core.executor import CodeExecutor, ExecutionStatus


class TruthSpaceLCM:
    """
    TruthSpace Language-Code Model
    
    Translates natural language to executable code using geometric
    knowledge encoding and template-based generation.
    """
    
    def __init__(self, working_dir: str = None, verbose: bool = True):
        self.planner = TaskPlanner(working_dir=working_dir)
        self.verbose = verbose
        self.history = []
    
    def process(self, request: str, dry_run: bool = False, 
                auto_execute: bool = True) -> TaskPlan:
        """
        Process a natural language request.
        
        Args:
            request: Natural language description of the task
            dry_run: If True, only generate code without executing
            auto_execute: If True, automatically execute after planning
            
        Returns:
            TaskPlan with results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"REQUEST: {request}")
            print("="*60)
        
        # Create plan
        plan = self.planner.plan(request)
        
        if self.verbose:
            print(f"\nGOAL: {plan.goal}")
            print(f"STEPS: {len(plan.steps)}")
            for step in plan.steps:
                deps = f" (after step {step.depends_on})" if step.depends_on else ""
                print(f"  {step.id}. [{step.step_type.value}] {step.description}{deps}")
        
        # Execute if requested
        if auto_execute:
            if self.verbose:
                mode = "DRY RUN" if dry_run else "EXECUTING"
                print(f"\n{mode}...")
            
            plan = self.planner.execute_plan(plan, dry_run=dry_run)
            
            if self.verbose:
                self._print_results(plan)
        
        # Save to history
        self.history.append(plan)
        
        return plan
    
    def _print_results(self, plan: TaskPlan):
        """Print execution results."""
        print(f"\nRESULTS:")
        print("-" * 40)
        
        for step in plan.steps:
            status_icon = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
            }.get(step.status, "?")
            
            print(f"{status_icon} Step {step.id}: {step.description[:50]}...")
            
            if step.generated_code:
                code_preview = step.generated_code.split('\n')[0][:60]
                print(f"   Code: {code_preview}")
            
            if step.output and step.status == StepStatus.COMPLETED:
                output_preview = step.output.strip()[:100]
                if output_preview:
                    print(f"   Output: {output_preview}")
            
            if step.error:
                print(f"   Error: {step.error}")
        
        print("-" * 40)
        
        # Overall status
        if plan.status == StepStatus.COMPLETED:
            print("✅ TASK COMPLETED SUCCESSFULLY")
        elif plan.status == StepStatus.FAILED:
            print("❌ TASK FAILED")
        else:
            print(f"Status: {plan.status.value}")
    
    def show_generated_code(self, plan: TaskPlan = None):
        """Show all generated code from a plan."""
        if plan is None:
            if not self.history:
                print("No plans in history")
                return
            plan = self.history[-1]
        
        print("\n" + "="*60)
        print("GENERATED CODE")
        print("="*60)
        
        for step in plan.steps:
            print(f"\n--- Step {step.id} ({step.step_type.value}) ---")
            print(f"# {step.description}")
            print(step.generated_code)
    
    def get_working_dir(self) -> str:
        """Get the current working directory."""
        return self.planner.executor.working_dir
    
    def reset(self):
        """Reset the executor working directory."""
        self.planner.executor.reset()
        if self.verbose:
            print(f"Working directory reset to: {self.get_working_dir()}")
    
    def cleanup(self):
        """Clean up temporary files."""
        self.planner.executor.cleanup()
        if self.verbose:
            print("Cleaned up temporary files")


def interactive_mode(lcm: TruthSpaceLCM):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("TruthSpace LCM - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  <request>     - Process a natural language request")
    print("  /dry <req>    - Dry run (don't execute)")
    print("  /code         - Show generated code from last request")
    print("  /reset        - Reset working directory")
    print("  /dir          - Show working directory")
    print("  /help         - Show this help")
    print("  /quit         - Exit")
    print()
    print(f"Working directory: {lcm.get_working_dir()}")
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit', '/q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == '/help':
                print("\nCommands:")
                print("  <request>     - Process a natural language request")
                print("  /dry <req>    - Dry run (don't execute)")
                print("  /code         - Show generated code from last request")
                print("  /reset        - Reset working directory")
                print("  /dir          - Show working directory")
                print("  /help         - Show this help")
                print("  /quit         - Exit")
            
            elif user_input.lower() == '/code':
                lcm.show_generated_code()
            
            elif user_input.lower() == '/reset':
                lcm.reset()
            
            elif user_input.lower() == '/dir':
                print(f"Working directory: {lcm.get_working_dir()}")
            
            elif user_input.lower().startswith('/dry '):
                request = user_input[5:].strip()
                if request:
                    lcm.process(request, dry_run=True)
            
            else:
                lcm.process(user_input, dry_run=False)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TruthSpace LCM - Natural Language to Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "create a python project called myapp"
  %(prog)s "list all files in the current directory"
  %(prog)s --dry-run "backup the project folder"
  %(prog)s --interactive
        """
    )
    
    parser.add_argument(
        'request',
        nargs='?',
        help='Natural language request to process'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Generate code without executing'
    )
    
    parser.add_argument(
        '-w', '--working-dir',
        help='Working directory for execution'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--show-code',
        action='store_true',
        help='Show generated code after execution'
    )
    
    args = parser.parse_args()
    
    # Create LCM instance
    lcm = TruthSpaceLCM(
        working_dir=args.working_dir,
        verbose=not args.quiet
    )
    
    try:
        if args.interactive:
            interactive_mode(lcm)
        elif args.request:
            plan = lcm.process(args.request, dry_run=args.dry_run)
            
            if args.show_code:
                lcm.show_generated_code(plan)
            
            # Exit with appropriate code
            if plan.status == StepStatus.COMPLETED:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            parser.print_help()
    finally:
        if not args.working_dir:
            # Only cleanup if using temp directory
            lcm.cleanup()


if __name__ == "__main__":
    main()
