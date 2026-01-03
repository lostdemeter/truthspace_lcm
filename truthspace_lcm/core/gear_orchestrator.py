"""
Gear Orchestrator

A self-aware gear system that:
1. Takes high-level goals
2. Breaks them into steps (PlannerGear)
3. Converts steps to executable commands (CommandGear)
4. Detects when new gears are needed and creates them on-the-fly

This is the autonomous gear chain - gears that know when they need
other gears and can create them.

Example:
    orchestrator = GearOrchestrator()
    orchestrator.configure_llm(url, model)
    
    result = orchestrator.execute(
        "Create a directory called 'test', touch a file called 'hello.txt', 
         then write 'Hello World' into that file"
    )
    
    # Returns:
    # {
    #     'plan': ['Create directory test', 'Create file hello.txt', 'Write content'],
    #     'commands': ['mkdir -p test', 'touch test/hello.txt', 'echo "Hello World" > test/hello.txt'],
    #     'gears_used': ['planner', 'command_generator'],
    #     'gears_created': []
    # }

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from truthspace_lcm.core.base import Gear, GearState
from truthspace_lcm.core.gear_message import (
    GearProtocol, GearMessage, MessageIntent,
    adapt_from_gear_state
)
from truthspace_lcm.core.bootstrap_gear import BootstrapGear
from truthspace_lcm.core.gear_factory import GearFactoryGear


class PlannerGear(GearProtocol):
    """
    Turns high-level ideas into step-by-step plans.
    
    Can work emergently (pattern matching) or with LLM assistance.
    
    Implements GearProtocol for standardized communication.
    """
    
    PLAN_PROMPT = """Break this goal into clear, sequential steps:

Goal: {goal}

Rules:
1. Each step should be a single, concrete action
2. Steps should be in logical order
3. Be specific but concise
4. Number each step

Reply with JSON:
{{"steps": ["step 1", "step 2", ...]}}"""

    def __init__(self):
        self.name = "PlannerGear"
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Emergent patterns for common planning tasks
        self.patterns: Dict[str, List[str]] = {
            'file_creation': [
                'Create the directory',
                'Create the file',
                'Write content to file',
            ],
            'project_setup': [
                'Create project directory',
                'Initialize version control',
                'Create configuration files',
                'Install dependencies',
            ],
        }
    
    def configure_llm(self, url: str, model: str):
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_url:
            return None
        
        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 500, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception:
            pass
        return None
    
    def plan(self, goal: str) -> List[str]:
        """
        Convert a goal into a list of steps.
        """
        # Try emergent patterns first
        goal_lower = goal.lower()
        
        if 'directory' in goal_lower and 'file' in goal_lower:
            # File creation pattern - customize based on goal
            steps = []
            
            # Extract directory name (handle quotes)
            dir_match = re.search(r"directory\s+(?:called\s+)?['\"]?(\w+)['\"]?", goal_lower)
            dir_name = dir_match.group(1) if dir_match else "new_dir"
            
            # Extract file name (handle quotes)
            file_match = re.search(r"file\s+(?:called\s+)?['\"]?([.\w]+)['\"]?", goal_lower)
            file_name = file_match.group(1) if file_match else "file.txt"
            
            # Extract content - look for quoted text after pipe/write/echo
            content_match = re.search(r"(?:write|pipe|echo)\s+(?:text\s+)?['\"]([^'\"]+)['\"]", goal_lower)
            if not content_match:
                # Try without quotes
                content_match = re.search(r"(?:write|pipe|echo)\s+(?:text\s+)?(\S+)", goal_lower)
            content = content_match.group(1).strip() if content_match else None
            
            steps.append(f"Create directory {dir_name}")
            steps.append(f"Create file {file_name} in {dir_name}")
            if content:
                steps.append(f"Write \"{content}\" to {dir_name}/{file_name}")
            
            return steps
        
        # Fall back to LLM
        if self.llm_url:
            prompt = self.PLAN_PROMPT.format(goal=goal)
            response = self._call_llm(prompt)
            
            if response:
                # Parse JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        return data.get('steps', [])
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract numbered steps
                steps = re.findall(r'\d+\.\s*(.+)', response)
                if steps:
                    return steps
        
        # Last resort: return the goal as a single step
        return [goal]
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Plan steps from goal. Implements GearProtocol."""
        steps = self.plan(message.content)
        return self.send(
            message.with_context('plan', steps),
            content='\n'.join(steps)
        )
    
    def forward(self, state: GearState) -> GearState:
        """Legacy GearState interface."""
        goal = state.metadata.get('goal', '') or state.entity or ''
        steps = self.plan(goal)
        state.metadata['plan'] = steps
        return state


class CommandGear(GearProtocol):
    """
    Converts step descriptions into executable bash commands.
    
    Can work emergently (pattern matching) or with LLM assistance.
    
    Implements GearProtocol for standardized communication.
    """
    
    COMMAND_PROMPT = """Convert this step into a bash command:

Step: {step}
Context: {context}

Rules:
1. Use standard Unix commands
2. Be safe (no rm -rf /, etc.)
3. Use proper quoting for strings with spaces

Reply with ONLY the bash command, nothing else."""

    def __init__(self):
        self.name = "CommandGear"
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Emergent command patterns
        self.patterns = {
            r"create\s+directory\s+(\w+)": "mkdir -p {0}",
            r"create\s+file\s+(\S+)\s+in\s+(\w+)": "touch {1}/{0}",
            r"create\s+file\s+(\S+)": "touch {0}",
            r'write\s+"([^"]+)"\s+to\s+(\S+)': 'echo "{0}" > {1}',
            r"append\s+['\"]([^'\"]+)['\"]\s+to\s+['\"]?([^'\"]+)['\"]?": 'echo "{0}" >> {1}',
            r"delete\s+file\s+['\"]?([^'\"]+)['\"]?": "rm {0}",
            r"delete\s+directory\s+['\"]?(\S+)['\"]?": "rm -r {0}",
            r"list\s+files?\s+in\s+['\"]?(\S+)['\"]?": "ls -la {0}",
            r"list\s+files?": "ls -la",
            r"show\s+contents?\s+of\s+['\"]?([^'\"]+)['\"]?": "cat {0}",
            r"copy\s+['\"]?([^'\"]+)['\"]?\s+to\s+['\"]?([^'\"]+)['\"]?": "cp {0} {1}",
            r"move\s+['\"]?([^'\"]+)['\"]?\s+to\s+['\"]?([^'\"]+)['\"]?": "mv {0} {1}",
            r"change\s+to\s+directory\s+['\"]?(\S+)['\"]?": "cd {0}",
            r"current\s+directory": "pwd",
            r"disk\s+usage": "df -h",
            r"memory\s+usage": "free -h",
        }
    
    def configure_llm(self, url: str, model: str):
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_url:
            return None
        
        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "temperature": 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception:
            pass
        return None
    
    def to_command(self, step: str, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Convert a step description to a bash command.
        """
        step_lower = step.lower()
        context = context or {}
        
        # Try emergent patterns
        for pattern, template in self.patterns.items():
            match = re.search(pattern, step_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    return template.format(*groups)
                except (IndexError, KeyError):
                    return template
        
        # Fall back to LLM
        if self.llm_url:
            context_str = json.dumps(context) if context else "None"
            prompt = self.COMMAND_PROMPT.format(step=step, context=context_str)
            response = self._call_llm(prompt)
            
            if response:
                # Clean up response - extract just the command
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Skip markdown code blocks
                    if line.startswith('```'):
                        continue
                    if line and not line.startswith('#'):
                        return line
        
        return None
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Convert step to command. Implements GearProtocol."""
        context = message.context.get('command_context', {})
        command = self.to_command(message.content, context)
        return self.send(
            message.with_context('command', command),
            content=command or ''
        )
    
    def forward(self, state: GearState) -> GearState:
        """Legacy GearState interface."""
        step = state.metadata.get('step', '') or state.entity or ''
        context = state.metadata.get('context', {})
        command = self.to_command(step, context)
        state.metadata['command'] = command
        return state


class GearOrchestrator(GearProtocol):
    """
    The autonomous gear orchestrator.
    
    Takes high-level goals and:
    1. Plans the steps (PlannerGear)
    2. Converts to commands (CommandGear)
    3. Creates new gears on-the-fly if needed
    
    This is the self-aware gear system that knows when it needs
    other gears and can create them.
    
    Implements GearProtocol for standardized communication.
    """
    
    def __init__(self):
        self.name = "GearOrchestrator"
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Core gears
        self.planner = PlannerGear()
        self.commander = CommandGear()
        
        # Gear factory for creating new gears on-the-fly
        self.factory: Optional[GearFactoryGear] = None
        
        # Registry of available gears
        self.gears: Dict[str, Gear] = {
            'planner': self.planner,
            'commander': self.commander,
        }
        
        # Track what was created
        self.gears_created: List[str] = []
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for all sub-gears."""
        self.llm_url = url
        self.llm_model = model
        
        self.planner.configure_llm(url, model)
        self.commander.configure_llm(url, model)
        
        # Initialize factory
        self.factory = GearFactoryGear()
        self.factory.configure_llm(url, model)
    
    def add_gear(self, name: str, gear: Gear):
        """Add a gear to the registry."""
        self.gears[name] = gear
    
    def get_or_create_gear(self, description: str, name: str = None) -> Gear:
        """
        Get an existing gear or create a new one.
        
        This is the key method - it detects when a new gear is needed
        and creates it on-the-fly.
        """
        # Check if we have a matching gear
        desc_lower = description.lower()
        
        for gear_name, gear in self.gears.items():
            if gear_name in desc_lower:
                return gear
        
        # Need to create a new gear
        if self.factory:
            print(f"Creating new gear for: {description[:50]}...")
            new_gear = self.factory.create(description, name=name)
            
            gear_name = name or new_gear.state.name
            self.gears[gear_name] = new_gear
            self.gears_created.append(gear_name)
            
            return new_gear
        
        raise RuntimeError(f"No gear available for: {description}")
    
    def execute(self, goal: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute a high-level goal.
        
        Args:
            goal: The high-level goal to achieve
            dry_run: If True, return commands but don't execute them
        
        Returns:
            Dict with plan, commands, and execution results
        """
        result = {
            'goal': goal,
            'plan': [],
            'commands': [],
            'gears_used': [],
            'gears_created': [],
            'executed': not dry_run,
            'outputs': [],
        }
        
        # Step 1: Plan
        steps = self.planner.plan(goal)
        result['plan'] = steps
        result['gears_used'].append('planner')
        
        # Step 2: Convert each step to a command
        context = {'goal': goal}
        for i, step in enumerate(steps):
            command = self.commander.to_command(step, context)
            if command:
                result['commands'].append(command)
                # Update context with what we've done
                context[f'step_{i}'] = {'step': step, 'command': command}
        
        result['gears_used'].append('commander')
        result['gears_created'] = self.gears_created.copy()
        
        # Step 3: Execute if not dry run
        if not dry_run:
            import subprocess
            for cmd in result['commands']:
                try:
                    output = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=30
                    )
                    result['outputs'].append({
                        'command': cmd,
                        'stdout': output.stdout,
                        'stderr': output.stderr,
                        'returncode': output.returncode,
                    })
                except Exception as e:
                    result['outputs'].append({
                        'command': cmd,
                        'error': str(e),
                    })
        
        return result
    
    def chain(self, *gear_names: str) -> 'GearChain':
        """
        Create a chain of gears to process input sequentially.
        
        Example:
            chain = orchestrator.chain('planner', 'commander')
            result = chain.process("Create a test file")
        """
        gears = []
        for name in gear_names:
            if name in self.gears:
                gears.append(self.gears[name])
            else:
                raise ValueError(f"Unknown gear: {name}")
        
        return GearChain(gears)
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Execute goal and return result. Implements GearProtocol."""
        result = self.execute(message.content, dry_run=True)
        
        # Build response
        if result['commands']:
            cmd_list = '\n'.join([f"  $ {cmd}" for cmd in result['commands']])
            response = f"Plan:\n{chr(10).join(result['plan'])}\n\nCommands:\n{cmd_list}"
        else:
            response = "Could not generate commands for this goal."
        
        return self.send(
            message.with_context('orchestrator_result', result),
            content=response,
            intent=MessageIntent.EXECUTE
        )
    
    def forward(self, state: GearState) -> GearState:
        """Process a goal through the orchestrator (legacy interface)."""
        goal = state.metadata.get('goal', '') or state.entity or ''
        
        result = self.execute(goal, dry_run=True)
        
        state.metadata['orchestrator_result'] = result
        state.metadata['plan'] = result['plan']
        state.metadata['commands'] = result['commands']
        
        return state


class GearChain:
    """
    A chain of gears that process input sequentially.
    
    Each gear's output becomes the next gear's input.
    """
    
    def __init__(self, gears: List[Gear]):
        self.gears = gears
    
    def process(self, input_data: Any) -> GearState:
        """Process input through all gears in sequence."""
        state = GearState(entity=str(input_data) if not isinstance(input_data, str) else input_data)
        state.metadata['input'] = input_data
        
        for gear in self.gears:
            state = gear.forward(state)
        
        return state
    
    def __repr__(self):
        names = [g.name for g in self.gears]
        return f"GearChain({' → '.join(names)})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_execute(goal: str, llm_url: str, llm_model: str, 
                  dry_run: bool = True) -> Dict[str, Any]:
    """
    Quickly execute a goal.
    
    Example:
        result = quick_execute(
            "Create a directory called test and add a hello.txt file",
            "http://localhost:11434/api/generate",
            "qwen2.5:14b"
        )
        print(result['commands'])
    """
    orchestrator = GearOrchestrator()
    orchestrator.configure_llm(llm_url, llm_model)
    return orchestrator.execute(goal, dry_run=dry_run)
