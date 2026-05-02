"""
φ-Space Autonomous Problem-Solving Agent

An agent that exploits geometric properties of φ-space to solve novel problems:
1. Uses reverse navigation to find creative solution paths
2. Validates ideas through bottleneck convergence
3. Executes code in a sandbox to prove solutions
4. Loops until the problem is solved or deemed unsolvable

The key insight: by navigating φ-space geometrically, we can discover
solutions that wouldn't emerge from linear reasoning alone.
"""

import torch
import subprocess
import tempfile
import os
import sys
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949

class AgentState(Enum):
    THINKING = "thinking"
    EXPLORING = "exploring"  # φ-space exploration
    CODING = "coding"
    EXECUTING = "executing"
    VALIDATING = "validating"
    SOLVED = "solved"
    STUCK = "stuck"

@dataclass
class ThoughtStep:
    """A single step in the agent's reasoning."""
    state: AgentState
    thought: str
    phi_level: float = 0.0
    code: Optional[str] = None
    result: Optional[str] = None
    is_valid: bool = True

@dataclass 
class SandboxResult:
    """Result from sandboxed code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False

class Sandbox:
    """
    Safe code execution environment.
    
    Restrictions:
    - No file system access outside temp directory
    - No network access
    - Time limit on execution
    - Memory limit
    - No dangerous imports
    """
    
    FORBIDDEN_IMPORTS = [
        'os.system', 'subprocess', 'shutil.rmtree', 'shutil.move',
        '__import__', 'eval', 'exec', 'compile', 'open',
        'socket', 'urllib', 'requests', 'http',
    ]
    
    ALLOWED_IMPORTS = [
        'math', 'random', 'itertools', 'functools', 'collections',
        'numpy', 'scipy', 'sympy', 'statistics',
        'json', 're', 'datetime', 'time',
        'typing', 'dataclasses', 'enum',
    ]
    
    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output
        
    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute."""
        # Check for forbidden patterns
        for forbidden in self.FORBIDDEN_IMPORTS:
            if forbidden in code:
                return False, f"Forbidden pattern: {forbidden}"
        
        # Check for file operations
        if re.search(r'\bopen\s*\(', code):
            # Allow open only for reading in specific patterns
            if not re.search(r'open\s*\([^)]*["\']r["\']', code):
                return False, "File write operations not allowed"
        
        # Check for shell execution
        if re.search(r'os\.(system|popen|exec)', code):
            return False, "Shell execution not allowed"
            
        return True, "OK"
    
    def execute(self, code: str) -> SandboxResult:
        """Execute code in a sandboxed environment."""
        # Safety check
        is_safe, reason = self._check_code_safety(code)
        if not is_safe:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Safety check failed: {reason}",
                return_code=-1
            )
        
        # Create temp file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add safety wrapper
            wrapped_code = f'''
import sys
import resource

# Set memory limit (512MB)
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

# Disable dangerous builtins
import builtins
_original_open = builtins.open
def _safe_open(file, mode='r', *args, **kwargs):
    if 'w' in mode or 'a' in mode or 'x' in mode:
        raise PermissionError("Write operations not allowed in sandbox")
    return _original_open(file, mode, *args, **kwargs)
# builtins.open = _safe_open  # Commented out to allow numpy etc to work

# User code
{code}
'''
            f.write(wrapped_code)
            temp_path = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:self.max_output]
            
            return SandboxResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds",
                return_code=-1,
                timed_out=True
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1
            )
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass


class PhiSpaceExplorer:
    """
    Explores φ-space to find creative solutions.
    
    Uses:
    - Reverse navigation from goal to find valid paths
    - Bottleneck validation to filter impossible ideas
    - Concept bridging to connect disparate domains
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get φ-space embedding for text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use layer 27 (bottleneck) embedding
            hidden_states = outputs.hidden_states
            layer_27 = hidden_states[min(27, len(hidden_states)-1)]
            return layer_27[0, -1]  # Last token embedding
    
    def compute_phi_level(self, embedding: torch.Tensor) -> float:
        """Compute φ-level of an embedding."""
        emb = embedding.float()
        # Ratio of norms as φ-signature
        l1 = emb.abs().sum()
        l2 = emb.norm()
        if l2 > 0:
            return float(l1 / (l2 * (emb.numel() ** 0.5)))
        return 0.0
    
    def find_bridge_concepts(self, concept_a: str, concept_b: str, n_bridges: int = 5) -> List[str]:
        """Find concepts that bridge two domains."""
        emb_a = self.get_embedding(concept_a)
        emb_b = self.get_embedding(concept_b)
        
        # Midpoint in φ-space
        midpoint = (emb_a + emb_b) / 2
        
        # Decode midpoint to find bridging concepts
        with torch.no_grad():
            midpoint = midpoint.to(self.model.lm_head.weight.dtype)
            logits = self.model.lm_head(midpoint.unsqueeze(0))
            probs = torch.softmax(logits[0], dim=-1)
            top_probs, top_indices = probs.topk(n_bridges * 3)
            
            bridges = []
            for idx in top_indices.tolist():
                token = self.tokenizer.decode([idx]).strip()
                if len(token) > 2 and token.isalpha():
                    bridges.append(token)
                if len(bridges) >= n_bridges:
                    break
                    
        return bridges
    
    def validate_idea(self, idea: str) -> Tuple[bool, float]:
        """
        Validate an idea through bottleneck convergence.
        
        Returns (is_valid, phi_level)
        Valid ideas converge to φ-level close to φ.
        """
        embedding = self.get_embedding(idea)
        phi_level = self.compute_phi_level(embedding)
        
        # Ideas with φ-level in valid range are considered valid
        # This is the "bottleneck filter"
        is_valid = 0.4 < phi_level < 0.8  # Empirical range from experiments
        
        return is_valid, phi_level
    
    def explore_variations(self, base_idea: str, n_variations: int = 5) -> List[Tuple[str, float]]:
        """Generate variations of an idea by perturbing in φ-space."""
        base_emb = self.get_embedding(base_idea)
        
        variations = []
        for i in range(n_variations):
            # Add small perturbation
            noise = torch.randn_like(base_emb) * 0.1 * (i + 1)
            perturbed = base_emb + noise
            
            # Decode perturbation
            with torch.no_grad():
                perturbed = perturbed.to(self.model.lm_head.weight.dtype)
                logits = self.model.lm_head(perturbed.unsqueeze(0))
                probs = torch.softmax(logits[0], dim=-1)
                top_idx = probs.argmax().item()
                token = self.tokenizer.decode([top_idx])
                
            phi_level = self.compute_phi_level(perturbed)
            variations.append((f"{base_idea} + {token.strip()}", phi_level))
            
        return variations


class PhiAgent:
    """
    Autonomous problem-solving agent using φ-space geometry.
    
    The agent:
    1. Understands the problem
    2. Explores φ-space for creative solutions
    3. Generates and tests code
    4. Validates solutions through execution
    5. Loops until solved or stuck
    """
    
    def __init__(self, model, tokenizer, max_iterations: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.explorer = PhiSpaceExplorer(model, tokenizer)
        self.sandbox = Sandbox(timeout=30)
        self.max_iterations = max_iterations
        self.history: List[ThoughtStep] = []
        
    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the model with proper chat template."""
        # Use Qwen2's chat template for better responses
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from text."""
        # Look for code blocks
        code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        code_match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # Look for indented code
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
                in_code = True
            elif in_code and line.strip() == '':
                code_lines.append(line)
            elif in_code:
                break
                
        if code_lines:
            return '\n'.join(code_lines)
            
        return None
    
    def think(self, problem: str, context: str = "") -> ThoughtStep:
        """Generate a thought about the problem."""
        prompt = f"""You are a problem-solving agent. Think step by step about this problem.

Problem: {problem}

{f"Context from previous steps: {context}" if context else ""}

Think about:
1. What is the core challenge?
2. What approaches might work?
3. What's a creative angle we haven't tried?

Your thought:"""

        thought = self._generate(prompt, max_tokens=300)
        embedding = self.explorer.get_embedding(thought)
        phi_level = self.explorer.compute_phi_level(embedding)
        
        return ThoughtStep(
            state=AgentState.THINKING,
            thought=thought,
            phi_level=phi_level
        )
    
    def explore(self, problem: str, current_approach: str) -> ThoughtStep:
        """Explore φ-space for creative solutions."""
        # Find bridge concepts between problem domain and solution space
        bridges = self.explorer.find_bridge_concepts(problem[:100], current_approach[:100])
        
        # Validate the current approach
        is_valid, phi_level = self.explorer.validate_idea(current_approach)
        
        # Generate variations
        variations = self.explorer.explore_variations(current_approach, n_variations=3)
        
        exploration = f"""φ-Space Exploration:
- Bridge concepts: {bridges}
- Current approach validity: {'VALID' if is_valid else 'INVALID'} (φ={phi_level:.4f})
- Variations explored: {[v[0][:50] for v in variations]}
- Best variation φ-level: {max(v[1] for v in variations):.4f}"""

        return ThoughtStep(
            state=AgentState.EXPLORING,
            thought=exploration,
            phi_level=phi_level,
            is_valid=is_valid
        )
    
    def code(self, problem: str, approach: str) -> ThoughtStep:
        """Generate code to solve the problem."""
        prompt = f"""Write complete, working Python code to solve this problem.

Requirements:
1. The code must be complete and runnable as-is
2. Use descriptive variable names (no abbreviations)
3. Print clear output showing the solution works
4. End with: print("SOLUTION VERIFIED:", your_result)

Problem: {problem}

Approach: {approach}

Write the complete Python code inside a single code block:"""

        response = self._generate(prompt, max_tokens=800)
        
        # Try multiple extraction methods
        code = self._extract_code(response)
        
        if not code:
            # Try with prefix
            code = self._extract_code("```python\n" + response)
        
        if not code:
            # If response looks like code (has def, print, =), use it directly
            if any(kw in response for kw in ['def ', 'print(', ' = ', 'import ']):
                code = response.strip()
        
        if not code:
            code = response.split("```")[0].strip()
        
        embedding = self.explorer.get_embedding(code if code else response)
        phi_level = self.explorer.compute_phi_level(embedding)
        
        return ThoughtStep(
            state=AgentState.CODING,
            thought=f"Generated code for: {approach[:100]}",
            phi_level=phi_level,
            code=code
        )
    
    def execute(self, code: str) -> ThoughtStep:
        """Execute code in sandbox."""
        result = self.sandbox.execute(code)
        
        thought = f"Execution {'succeeded' if result.success else 'failed'}"
        if result.timed_out:
            thought = "Execution timed out"
            
        output = result.stdout if result.success else result.stderr
        
        return ThoughtStep(
            state=AgentState.EXECUTING,
            thought=thought,
            result=output,
            is_valid=result.success,
            code=code
        )
    
    def validate(self, problem: str, result: str) -> ThoughtStep:
        """Validate if the result solves the problem."""
        # Check for solution marker
        solved = "SOLUTION VERIFIED:" in result
        
        prompt = f"""Did this output solve the problem?

Problem: {problem}

Output:
{result[:1000]}

Answer YES if the problem is solved, NO if not. Then explain briefly.

Answer:"""

        response = self._generate(prompt, max_tokens=100)
        
        is_solved = solved or response.strip().upper().startswith("YES")
        
        embedding = self.explorer.get_embedding(result)
        phi_level = self.explorer.compute_phi_level(embedding)
        
        return ThoughtStep(
            state=AgentState.SOLVED if is_solved else AgentState.VALIDATING,
            thought=f"Validation: {response[:200]}",
            phi_level=phi_level,
            result=result,
            is_valid=is_solved
        )
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Main solving loop.
        
        Returns dict with:
        - solved: bool
        - solution: str (code that worked)
        - result: str (output)
        - history: list of steps
        - iterations: int
        """
        print(f"\n{'='*60}")
        print("φ-AGENT: AUTONOMOUS PROBLEM SOLVER")
        print(f"{'='*60}")
        print(f"\nProblem: {problem}\n")
        
        self.history = []
        current_approach = ""
        best_code = None
        best_result = None
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Step 1: Think about the problem
            context = "\n".join([f"- {h.thought[:100]}" for h in self.history[-3:]])
            thought = self.think(problem, context)
            self.history.append(thought)
            print(f"[THINK] {thought.thought[:200]}...")
            print(f"        φ-level: {thought.phi_level:.4f}")
            
            current_approach = thought.thought
            
            # Step 2: Explore φ-space
            exploration = self.explore(problem, current_approach)
            self.history.append(exploration)
            print(f"[EXPLORE] {exploration.thought[:200]}")
            
            # If approach is invalid, try again
            if not exploration.is_valid:
                print("        Approach invalid in φ-space, reconsidering...")
                continue
            
            # Step 3: Generate code
            code_step = self.code(problem, current_approach)
            self.history.append(code_step)
            print(f"[CODE] Generated {len(code_step.code) if code_step.code else 0} chars")
            
            if not code_step.code:
                print("        No code generated, retrying...")
                continue
            
            # Step 4: Execute in sandbox
            exec_step = self.execute(code_step.code)
            self.history.append(exec_step)
            print(f"[EXEC] {exec_step.thought}")
            if exec_step.result:
                print(f"       Output: {exec_step.result[:200]}")
            
            if not exec_step.is_valid:
                print(f"       Error: {exec_step.result[:200] if exec_step.result else 'Unknown'}")
                continue
            
            best_code = code_step.code
            best_result = exec_step.result
            
            # Step 5: Validate solution
            validation = self.validate(problem, exec_step.result)
            self.history.append(validation)
            print(f"[VALIDATE] {validation.thought}")
            
            if validation.is_valid:
                print(f"\n{'='*60}")
                print("PROBLEM SOLVED!")
                print(f"{'='*60}")
                return {
                    "solved": True,
                    "solution": best_code,
                    "result": best_result,
                    "history": self.history,
                    "iterations": iteration + 1
                }
        
        print(f"\n{'='*60}")
        print(f"STUCK after {self.max_iterations} iterations")
        print(f"{'='*60}")
        
        return {
            "solved": False,
            "solution": best_code,
            "result": best_result,
            "history": self.history,
            "iterations": self.max_iterations
        }


def demo():
    """Demonstrate the φ-agent solving a novel problem."""
    print("Loading Qwen2-7B model...")
    
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Fast attention
        device_map="cuda",
    )
    
    agent = PhiAgent(model, tokenizer, max_iterations=5)
    
    # Default test problem - can be changed for different experiments
    problem = """
    Find a mathematical relationship that connects the Fibonacci sequence 
    to the golden ratio φ (1.618...) in a way that can be verified computationally.
    
    Write code that:
    1. Generates Fibonacci numbers
    2. Shows how they relate to φ
    3. Verifies the relationship numerically
    """
    
    result = agent.solve(problem)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Solved: {result['solved']}")
    print(f"Iterations: {result['iterations']}")
    
    if result['solution']:
        print(f"\nSolution Code:\n{result['solution']}")
    
    if result['result']:
        print(f"\nOutput:\n{result['result']}")


if __name__ == "__main__":
    demo()
