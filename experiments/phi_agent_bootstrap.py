"""
φ-Agent Self-Bootstrapping Experiment

Can the φ-Agent build tools for itself and then use them?

This tests a multi-stage problem:
1. Agent needs data from the web
2. Agent must first BUILD a web scraper
3. Agent must then USE the scraper to get data
4. Agent must PROCESS the data to answer the question

This is a test of self-bootstrapping capabilities.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import sys
import os
import tempfile

PHI = 1.6180339887498949


class PhiSpaceExplorer:
    """Navigate φ-space for geometric validation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bottleneck_layer = 27
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get layer-27 bottleneck embedding."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.bottleneck_layer]
        return hidden.mean(dim=1).squeeze()
    
    def compute_phi_level(self, embedding: torch.Tensor) -> float:
        """Compute φ-level as L1/L2 ratio normalized."""
        l1 = embedding.abs().sum().item()
        l2 = embedding.norm(2).item()
        dim = embedding.shape[0]
        ratio = l1 / (l2 * (dim ** 0.5))
        return ratio
    
    def validate_approach(self, text: str) -> tuple[bool, float]:
        """Check if an approach is geometrically valid."""
        emb = self.get_embedding(text)
        phi_level = self.compute_phi_level(emb)
        # Valid approaches have φ-level in reasonable range
        is_valid = 0.3 < phi_level < 0.9
        return is_valid, phi_level


class BootstrapSandbox:
    """
    Enhanced sandbox that allows the agent to:
    1. Create files (tools)
    2. Import and use those files
    3. Make network requests (for scraping)
    """
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            self.workspace = tempfile.mkdtemp(prefix="phi_agent_")
        else:
            self.workspace = workspace_dir
            os.makedirs(workspace_dir, exist_ok=True)
        
        self.created_files = []
        print(f"[SANDBOX] Workspace: {self.workspace}")
    
    def save_tool(self, filename: str, code: str) -> str:
        """Save a tool file that can be imported later."""
        filepath = os.path.join(self.workspace, filename)
        with open(filepath, 'w') as f:
            f.write(code)
        self.created_files.append(filepath)
        print(f"[SANDBOX] Created tool: {filename}")
        return filepath
    
    def execute(self, code: str, timeout: int = 30) -> dict:
        """Execute code with access to created tools."""
        # Create a runner script that adds workspace to path
        runner_code = f'''
import sys
sys.path.insert(0, "{self.workspace}")

{code}
'''
        # Write to temp file
        script_path = os.path.join(self.workspace, "_runner.py")
        with open(script_path, 'w') as f:
            f.write(runner_code)
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[:2000] if result.stdout else "",
                "error": result.stderr[:1000] if result.stderr else ""
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Timeout"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}


class BootstrappingPhiAgent:
    """
    φ-Agent that can bootstrap its own capabilities.
    
    Key difference from basic agent:
    - Can create tool files
    - Can import and use tools it created
    - Multi-stage problem solving
    """
    
    def __init__(self, model, tokenizer, max_iterations: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.explorer = PhiSpaceExplorer(model, tokenizer)
        self.sandbox = BootstrapSandbox()
        self.max_iterations = max_iterations
        self.tools_created = {}  # name -> code
        self.memory = []  # Track what we've learned
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def think(self, problem: str, context: str = "") -> dict:
        """Analyze problem and decide what tools are needed."""
        prompt = f"""You are a problem-solving agent. Analyze this problem:

{problem}

{f"Context from previous steps: {context}" if context else ""}

Tools you have created so far: {list(self.tools_created.keys()) if self.tools_created else "None"}

Think step by step:
1. What is the core challenge?
2. What tools/capabilities do you need?
3. Do you need to CREATE a tool first, or can you solve directly?
4. What's your next action?

Be specific about whether you need to BUILD a tool or USE existing capabilities."""

        response = self.generate(prompt, max_tokens=500)
        
        # Get φ-level for this thinking
        _, phi_level = self.explorer.validate_approach(response)
        
        return {
            "thinking": response[:500],
            "phi_level": phi_level
        }
    
    def create_tool(self, tool_description: str) -> dict:
        """Generate and save a reusable tool."""
        prompt = f"""Create a Python tool/module based on this description:

{tool_description}

Requirements:
- Write a complete, working Python module
- Include clear function definitions
- Add docstrings explaining usage
- Make it importable (no if __name__ == "__main__" execution)
- Handle errors gracefully

Output ONLY the Python code, no explanations. Start with imports."""

        code = self.generate(prompt, max_tokens=1500)
        
        # Clean up code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Determine tool name from description
        tool_name = tool_description.split()[0].lower() + "_tool.py"
        if "scraper" in tool_description.lower() or "scrape" in tool_description.lower():
            tool_name = "web_scraper.py"
        elif "parse" in tool_description.lower():
            tool_name = "parser_tool.py"
        
        # Save the tool
        filepath = self.sandbox.save_tool(tool_name, code)
        self.tools_created[tool_name] = code
        
        # Test if tool is importable
        test_result = self.sandbox.execute(f"import {tool_name[:-3]}\nprint('Tool imported successfully')")
        
        return {
            "tool_name": tool_name,
            "code": code[:500] + "..." if len(code) > 500 else code,
            "filepath": filepath,
            "importable": test_result["success"],
            "test_output": test_result["output"] or test_result["error"]
        }
    
    def use_tool(self, task: str) -> dict:
        """Generate code that uses created tools to accomplish a task."""
        # Show the agent what functions are actually in the tools
        tools_info = ""
        for name, code in self.tools_created.items():
            # Extract function definitions from the code
            import re
            funcs = re.findall(r'def (\w+)\([^)]*\):', code)
            tools_info += f"\n- {name}: contains functions: {', '.join(funcs)}"
            # Also show first few lines of each function
            tools_info += f"\n  Code preview:\n```python\n{code[:800]}\n```\n"
        
        prompt = f"""Write Python code to accomplish this task:

{task}

Available tools you can import:
{tools_info}

Previous results: {self.memory[-3:] if self.memory else "None"}

Write complete, executable Python code that:
1. Imports the tools you need
2. Uses them to accomplish the task
3. Prints clear results

Output ONLY Python code, no explanations."""

        code = self.generate(prompt, max_tokens=1000)
        
        # Clean up
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return {"code": code}
    
    def execute(self, code: str) -> dict:
        """Execute code in sandbox."""
        result = self.sandbox.execute(code)
        if result["output"]:
            self.memory.append(f"Execution result: {result['output'][:200]}")
        return result
    
    def validate(self, problem: str, result: str) -> bool:
        """Check if problem is solved."""
        # Quick heuristic: if we got a real title (not None, not error), it's solved
        if result and "None" not in result and "error" not in result.lower() and "failed" not in result.lower():
            if len(result) > 20:  # Got substantial output
                return True
        
        prompt = f"""Has this problem been solved?

Problem: {problem}

Result: {result}

Answer with just YES or NO, then a brief explanation."""

        response = self.generate(prompt, max_tokens=200)
        return response.strip().upper().startswith("YES")
    
    def solve(self, problem: str) -> dict:
        """
        Multi-stage problem solving with tool bootstrapping.
        """
        print("\n" + "="*60)
        print("φ-AGENT: BOOTSTRAPPING PROBLEM SOLVER")
        print("="*60)
        print(f"\nProblem: {problem[:200]}...")
        
        context = ""
        last_result = ""
        
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            
            # THINK: Analyze what we need
            thinking = self.think(problem, context)
            print(f"[THINK] {thinking['thinking'][:200]}...")
            print(f"        φ-level: {thinking['phi_level']:.4f}")
            
            # Decide: Create tool or use existing?
            thought_lower = thinking['thinking'].lower()
            
            if ("create" in thought_lower or "build" in thought_lower or "need to make" in thought_lower) and not self.tools_created:
                # CREATE TOOL phase
                print("[MODE] Creating new tool...")
                
                # Ask what tool to create
                tool_prompt = f"""Based on this problem: {problem}

And this analysis: {thinking['thinking']}

Describe the tool you need to create. Be specific about:
- What it should do
- What functions it should have
- What inputs/outputs

One paragraph description:"""
                
                tool_desc = self.generate(tool_prompt, max_tokens=200)
                print(f"[TOOL DESC] {tool_desc[:150]}...")
                
                tool_result = self.create_tool(tool_desc)
                print(f"[TOOL CREATED] {tool_result['tool_name']}")
                print(f"[IMPORTABLE] {tool_result['importable']}")
                
                if not tool_result['importable']:
                    print(f"[ERROR] {tool_result['test_output']}")
                    context += f"\nTool creation failed: {tool_result['test_output']}"
                else:
                    context += f"\nCreated tool: {tool_result['tool_name']}"
                
            else:
                # USE TOOLS phase
                print("[MODE] Using tools to solve...")
                
                use_result = self.use_tool(problem + f"\nContext: {context}")
                print(f"[CODE] Generated {len(use_result['code'])} chars")
                
                exec_result = self.execute(use_result['code'])
                
                if exec_result['success']:
                    print(f"[EXEC] Success!")
                    print(f"       Output: {exec_result['output'][:200]}")
                    last_result = exec_result['output']
                    
                    # Check if solved
                    if self.validate(problem, last_result):
                        print("\n" + "="*60)
                        print("PROBLEM SOLVED!")
                        print("="*60)
                        return {
                            "solved": True,
                            "iterations": i + 1,
                            "tools_created": list(self.tools_created.keys()),
                            "final_output": last_result,
                            "code": use_result['code']
                        }
                    else:
                        context += f"\nAttempt {i+1} output: {last_result[:200]}"
                else:
                    print(f"[EXEC] Failed: {exec_result['error'][:200]}")
                    context += f"\nError: {exec_result['error'][:100]}"
        
        return {
            "solved": False,
            "iterations": self.max_iterations,
            "tools_created": list(self.tools_created.keys()),
            "final_output": last_result,
            "context": context
        }


if __name__ == "__main__":
    print("Loading Qwen2-7B model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    
    agent = BootstrappingPhiAgent(model, tokenizer, max_iterations=6)
    
    # Test: Can the agent build a scraper and use it?
    problem = """
    I need to find out what the current top story on Hacker News is.
    
    You don't have a web scraper yet - you'll need to:
    1. First, BUILD a simple web scraper tool
    2. Then, USE that scraper to fetch https://news.ycombinator.com
    3. Parse the HTML to find the top story title
    4. Report the title of the #1 story
    
    This is a two-stage problem: build the tool, then use it.
    """
    
    result = agent.solve(problem)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Solved: {result['solved']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools Created: {result['tools_created']}")
    print(f"\nFinal Output:\n{result.get('final_output', 'None')[:500]}")
