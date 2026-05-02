"""
φ-Tool Agent: Autonomous Problem Solving with Tool Use

This experiment gives our model access to tools (Python execution, calculator,
symbolic math) and tests whether it can autonomously solve math problems by:

1. Understanding the problem
2. Deciding which tool to use
3. Writing and executing code
4. Interpreting results
5. Iterating until solved

The key insight: tool use is just another form of navigation in φ-space.
The model navigates toward "solution" by taking actions (tool calls) that
move it through the space.

Tools available:
- python_exec: Execute Python code and get results
- calculator: Basic arithmetic
- symbolic: Symbolic math (sympy)
- verify: Check if an answer is correct
"""

import torch
import json
import re
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tool definitions
TOOLS = {
    "python_exec": {
        "description": "Execute Python code and return the output. Use for calculations, data manipulation, or any computation.",
        "parameters": {"code": "Python code to execute"},
        "example": '{"tool": "python_exec", "code": "print(sum(range(1, 101)))"}'
    },
    "calculator": {
        "description": "Evaluate a mathematical expression. Use for simple arithmetic.",
        "parameters": {"expression": "Math expression like '2 + 2 * 3'"},
        "example": '{"tool": "calculator", "expression": "2 + 2 * 3"}'
    },
    "symbolic": {
        "description": "Perform symbolic mathematics using SymPy. Use for algebra, calculus, solving equations.",
        "parameters": {"operation": "Operation type (solve, simplify, diff, integrate)", "expression": "Math expression"},
        "example": '{"tool": "symbolic", "operation": "solve", "expression": "x**2 - 4"}'
    },
    "answer": {
        "description": "Submit your final answer to the problem.",
        "parameters": {"value": "The final answer", "explanation": "Brief explanation of how you got it"},
        "example": '{"tool": "answer", "value": "42", "explanation": "Computed by summing..."}'
    }
}

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None

@dataclass 
class AgentStep:
    """A single step in the agent's reasoning."""
    thought: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[ToolResult] = None
    
@dataclass
class AgentSession:
    """Complete agent session for solving a problem."""
    problem: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    solved: bool = False


class ToolExecutor:
    """Executes tools safely."""
    
    def __init__(self):
        self.execution_context = {}
        
    def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool and return the result."""
        if tool_name == "python_exec":
            return self._exec_python(params.get("code", ""))
        elif tool_name == "calculator":
            return self._exec_calculator(params.get("expression", ""))
        elif tool_name == "symbolic":
            return self._exec_symbolic(params.get("operation", ""), params.get("expression", ""))
        elif tool_name == "answer":
            return ToolResult(True, f"Answer submitted: {params.get('value')}")
        else:
            return ToolResult(False, "", f"Unknown tool: {tool_name}")
    
    def _exec_python(self, code: str) -> ToolResult:
        """Execute Python code safely."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Add common imports to context
            if 'math' not in self.execution_context:
                import math
                import numpy as np
                self.execution_context['math'] = math
                self.execution_context['np'] = np
                self.execution_context['numpy'] = np
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.execution_context)
            
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            
            if errors:
                return ToolResult(True, output, errors)
            return ToolResult(True, output if output else "(no output)")
            
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _exec_calculator(self, expression: str) -> ToolResult:
        """Evaluate a math expression."""
        try:
            # Safe eval with only math operations
            import math
            allowed = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolResult(True, str(result))
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _exec_symbolic(self, operation: str, expression: str) -> ToolResult:
        """Execute symbolic math."""
        try:
            from sympy import symbols, solve, simplify, diff, integrate, sympify, latex
            from sympy.abc import x, y, z, a, b, c, n, k
            
            expr = sympify(expression)
            
            if operation == "solve":
                result = solve(expr, x)
            elif operation == "simplify":
                result = simplify(expr)
            elif operation == "diff":
                result = diff(expr, x)
            elif operation == "integrate":
                result = integrate(expr, x)
            else:
                return ToolResult(False, "", f"Unknown operation: {operation}")
            
            return ToolResult(True, str(result))
        except Exception as e:
            return ToolResult(False, "", str(e))


class PhiToolAgent:
    """
    An agent that uses φ-space navigation to solve problems with tools.
    
    The agent thinks in a loop:
    1. Observe current state (problem + history)
    2. Think about what to do next
    3. Choose and execute a tool
    4. Observe result
    5. Repeat until solved
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tool_executor = ToolExecutor()
        self.max_steps = 10
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tool_desc = "\n".join([
            f"- **{name}**: {info['description']}\n  Example: {info['example']}"
            for name, info in TOOLS.items()
        ])
        
        return f"""You are a mathematical problem-solving agent with access to tools.

AVAILABLE TOOLS:
{tool_desc}

INSTRUCTIONS:
1. Think step by step about the problem
2. Use tools to compute, verify, or explore
3. When you want to use a tool, output EXACTLY this format:
   THOUGHT: [your reasoning]
   TOOL: {{"tool": "tool_name", "param": "value"}}
4. After seeing tool results, continue reasoning
5. When you have the final answer, use the "answer" tool

Be precise. Show your work. Use tools for any non-trivial computation."""

    def _build_prompt(self, session: AgentSession) -> str:
        """Build the full prompt from session history."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # Add the problem
        messages.append({"role": "user", "content": f"PROBLEM: {session.problem}\n\nSolve this step by step using tools."})
        
        # Add history
        for step in session.steps:
            # Agent's thought and tool call
            content = f"THOUGHT: {step.thought}"
            if step.tool_call:
                content += f"\nTOOL: {json.dumps(step.tool_call)}"
            messages.append({"role": "assistant", "content": content})
            
            # Tool result
            if step.tool_result:
                if step.tool_result.success:
                    messages.append({"role": "user", "content": f"TOOL RESULT: {step.tool_result.output}"})
                else:
                    messages.append({"role": "user", "content": f"TOOL ERROR: {step.tool_result.error}"})
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def _parse_response(self, response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Parse the model's response into thought and tool call."""
        thought = ""
        tool_call = None
        
        # Extract thought
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=TOOL:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = response.strip()
        
        # Extract tool call
        tool_match = re.search(r'TOOL:\s*(\{.+?\})', response, re.DOTALL)
        if tool_match:
            try:
                tool_call = json.loads(tool_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return thought, tool_call
    
    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a response from the model."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new generation
        return response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    def solve(self, problem: str, verbose: bool = True) -> AgentSession:
        """
        Solve a problem using tool-augmented reasoning.
        
        Returns the complete session with all steps.
        """
        session = AgentSession(problem=problem)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PROBLEM: {problem}")
            print('='*60)
        
        for step_num in range(self.max_steps):
            if verbose:
                print(f"\n--- Step {step_num + 1} ---")
            
            # Generate next thought/action
            prompt = self._build_prompt(session)
            response = self._generate(prompt)
            
            # Parse response
            thought, tool_call = self._parse_response(response)
            
            if verbose:
                print(f"THOUGHT: {thought}")
            
            step = AgentStep(thought=thought, tool_call=tool_call)
            
            # Execute tool if present
            if tool_call:
                tool_name = tool_call.get("tool", "")
                if verbose:
                    print(f"TOOL: {json.dumps(tool_call)}")
                
                # Check if this is the final answer
                if tool_name == "answer":
                    session.final_answer = tool_call.get("value")
                    session.solved = True
                    step.tool_result = ToolResult(True, f"Final answer: {session.final_answer}")
                    session.steps.append(step)
                    
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"FINAL ANSWER: {session.final_answer}")
                        print(f"EXPLANATION: {tool_call.get('explanation', 'N/A')}")
                        print('='*60)
                    break
                
                # Execute the tool
                result = self.tool_executor.execute(tool_name, tool_call)
                step.tool_result = result
                
                if verbose:
                    if result.success:
                        print(f"RESULT: {result.output}")
                    else:
                        print(f"ERROR: {result.error}")
            
            session.steps.append(step)
            
            # Check for stuck/looping
            if len(session.steps) >= 3:
                recent_thoughts = [s.thought for s in session.steps[-3:]]
                if len(set(recent_thoughts)) == 1:
                    if verbose:
                        print("\n[Agent appears stuck, stopping]")
                    break
        
        if not session.solved and verbose:
            print("\n[Max steps reached without solution]")
        
        return session


def run_math_experiments():
    """Run a series of math problems to test the agent."""
    
    agent = PhiToolAgent()
    
    problems = [
        # Arithmetic
        "What is the sum of all integers from 1 to 100?",
        
        # Algebra
        "Solve the equation: x^2 - 5x + 6 = 0",
        
        # Number theory
        "What is the greatest common divisor of 48 and 180?",
        
        # Calculus
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        
        # Word problem
        "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. What is the total distance traveled?",
        
        # Harder: Prime factorization
        "What is the prime factorization of 2310?",
        
        # Harder: Combinatorics
        "How many ways can you arrange the letters in the word 'MISSISSIPPI'?",
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n{'#'*60}")
        print(f"# PROBLEM {i+1}/{len(problems)}")
        print('#'*60)
        
        session = agent.solve(problem, verbose=True)
        results.append({
            "problem": problem,
            "solved": session.solved,
            "answer": session.final_answer,
            "steps": len(session.steps)
        })
        
        print(f"\nStatus: {'SOLVED' if session.solved else 'UNSOLVED'}")
        print(f"Steps taken: {len(session.steps)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    solved_count = sum(1 for r in results if r["solved"])
    print(f"Solved: {solved_count}/{len(results)}")
    
    for r in results:
        status = "✓" if r["solved"] else "✗"
        print(f"  {status} {r['problem'][:50]}... → {r['answer']}")
    
    return results


if __name__ == "__main__":
    print("φ-Tool Agent: Autonomous Math Problem Solving")
    print("="*60)
    
    # Run experiments
    results = run_math_experiments()
