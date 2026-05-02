#!/usr/bin/env python3
"""
φ-Chat with Planning: Plan → Code → Execute → Verify

The key insight: Better outputs come from thinking BEFORE coding.

Pipeline:
1. PLAN - Model creates a detailed plan for the task
2. REVIEW - Model critiques its own plan and improves it
3. CODE - Model writes code following the plan
4. EXECUTE - Run the code
5. VERIFY - Model evaluates if the output meets requirements
6. ITERATE - If not satisfied, refine and retry

This leverages the model's ability to work in reverse:
- Start from desired outcome (what should it look like?)
- Work backward to requirements
- Then forward to implementation
"""

import torch
import re
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Plan:
    """A plan for completing a task."""
    goal: str
    visual_description: str  # What should it LOOK like?
    components: List[str]    # What elements are needed?
    style_notes: List[str]   # Aesthetic considerations
    technical_approach: str  # How to implement
    potential_issues: List[str]  # What could go wrong?


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    files_created: List[str] = field(default_factory=list)


class PlannerChat:
    """
    Chat with planning capability.
    
    The model first creates a plan, then implements it.
    This produces higher quality outputs.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🚀 Loading φ-Chat Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.history: List[Dict] = []
        print("✓ Model loaded!\n")
    
    def generate(self, messages: List[Dict], max_tokens: int = 800) -> str:
        """Generate a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        
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
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def execute_code(self, code: str) -> ExecutionResult:
        """Execute code and return result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = OUTPUT_DIR / f"script_{timestamp}.py"
        files_before = set(OUTPUT_DIR.glob("*"))
        
        with open(script_path, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(OUTPUT_DIR)
            )
            
            files_after = set(OUTPUT_DIR.glob("*"))
            new_files = [str(f) for f in (files_after - files_before) if f != script_path]
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                files_created=new_files
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, "", "Timeout")
        except Exception as e:
            return ExecutionResult(False, "", str(e))
        finally:
            if script_path.exists():
                script_path.unlink()
    
    # =========================================================
    # STEP 1: CREATE PLAN
    # =========================================================
    
    def create_plan(self, user_request: str) -> str:
        """Have the model create a detailed plan."""
        
        messages = [
            {"role": "system", "content": """You are a creative designer. Create a SIMPLE but BEAUTIFUL plan.

Your plan should include:
1. VISUAL DESCRIPTION: What should it look like? (Keep it achievable with basic matplotlib)

2. COLOR PALETTE: Use ONLY valid matplotlib colors:
   - Named: 'red', 'pink', 'white', 'gold', 'crimson', 'hotpink', 'lightpink'
   - Hex: '#FF69B4', '#FFD700', '#FFC0CB'
   - NO made-up colors like 'blush' or 'rose'

3. COMPONENTS: List 3-5 elements (heart shape, text, background, maybe small decorations)

4. KEEP IT SIMPLE: A clean, elegant design beats an overloaded one.

Remember: This will be implemented with basic matplotlib. Don't plan anything that requires external images or complex libraries."""},
            
            {"role": "user", "content": f"""Create a simple, elegant plan for:

{user_request}

Keep it achievable with basic matplotlib. Don't overplan."""}
        ]
        
        print("\n📋 Creating plan...")
        plan = self.generate(messages, max_tokens=400)
        return plan
    
    # =========================================================
    # STEP 2: CRITIQUE AND IMPROVE PLAN
    # =========================================================
    
    def improve_plan(self, original_request: str, plan: str) -> str:
        """Have the model critique and improve its own plan - focusing on SIMPLICITY."""
        
        messages = [
            {"role": "system", "content": """You are a design critic focused on SIMPLICITY and ELEGANCE.

Review the plan and SIMPLIFY it:
- Remove anything too complex for basic matplotlib
- Ensure all colors are valid matplotlib colors
- Focus on 3-5 core elements maximum
- Suggest specific, achievable improvements

The best designs are SIMPLE and CLEAN, not overloaded."""},
            
            {"role": "user", "content": f"""Original request: {original_request}

Current plan:
{plan}

Simplify this plan. Remove complexity. Keep only what's essential and beautiful."""}
        ]
        
        print("🔍 Simplifying plan...")
        improved = self.generate(messages, max_tokens=400)
        return improved
    
    # =========================================================
    # STEP 3: GENERATE CODE FROM PLAN
    # =========================================================
    
    def generate_code(self, original_request: str, plan: str) -> str:
        """Generate code that follows the plan."""
        
        messages = [
            {"role": "system", "content": """You are a matplotlib programmer. Write SIMPLE, WORKING code.

CRITICAL RULES:
1. Use ONLY basic matplotlib functions that you know work
2. For hearts, use the parametric equation: x = 16*sin(t)^3, y = 13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t)
3. Use plt.fill() or plt.plot() for shapes - NOT Path or patches unless simple
4. Use plt.text() for text - NOT TextPath or fancy text objects
5. Use simple color names ('red', 'pink', 'white') or hex codes
6. Keep it simple! A beautiful simple card beats a broken complex one.

Template structure:
```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
# ... draw shapes with plot/fill ...
# ... add text with plt.text() ...
ax.set_xlim(...); ax.set_ylim(...)
ax.axis('off')
plt.savefig('filename.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Done!")
```

Put code in ```python``` blocks."""},
            
            {"role": "user", "content": f"""Create this: {original_request}

Design notes from plan:
{plan[:500]}

Write SIMPLE working code. Prioritize: working > beautiful > complex."""}
        ]
        
        print("💻 Generating code...")
        response = self.generate(messages, max_tokens=800)
        return response
    
    # =========================================================
    # STEP 4: VERIFY OUTPUT
    # =========================================================
    
    def verify_output(self, original_request: str, plan: str, code: str, execution: ExecutionResult) -> Tuple[bool, str]:
        """Have the model verify if the output meets the ORIGINAL REQUEST (not the fancy plan)."""
        
        if not execution.success:
            return False, f"Code failed to execute: {execution.stderr}"
        
        messages = [
            {"role": "system", "content": """You are a practical reviewer. Check if the code meets the ORIGINAL REQUEST.

Be LENIENT. Pass if:
1. The code executed successfully
2. It creates the requested output type (e.g., a Valentine's card)
3. It includes the key elements (e.g., heart, name, saved to file)

Don't fail just because it's not fancy. Simple and working is GOOD.

Respond with:
VERDICT: PASS or FAIL
REASON: One sentence"""},
            
            {"role": "user", "content": f"""Original request: {original_request}

Code executed successfully and created: {execution.files_created}
Output: {execution.stdout}

Does this meet the original request? Be lenient - working code is good!"""}
        ]
        
        print("✅ Verifying output...")
        response = self.generate(messages, max_tokens=150)
        
        # Be lenient - if it executed and created files, likely pass
        passed = "FAIL" not in response.upper().split('\n')[0]
        return passed, response
    
    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    
    def process_request(self, user_request: str, max_iterations: int = 2) -> Tuple[bool, str, Optional[ExecutionResult]]:
        """
        Full pipeline: Plan → Improve → Code → Execute → Verify
        """
        print("=" * 60)
        print("φ-Chat Planner: Plan → Code → Execute → Verify")
        print("=" * 60)
        print(f"\n📝 Request: {user_request}\n")
        
        # Step 1: Create initial plan
        plan = self.create_plan(user_request)
        print(f"\n--- INITIAL PLAN ---\n{plan[:800]}...")
        
        # Step 2: Improve the plan
        improved_plan = self.improve_plan(user_request, plan)
        print(f"\n--- IMPROVED PLAN ---\n{improved_plan[:800]}...")
        
        # Use the improved plan
        final_plan = improved_plan
        
        for iteration in range(max_iterations):
            print(f"\n{'='*40}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print('='*40)
            
            # Step 3: Generate code
            code_response = self.generate_code(user_request, final_plan)
            code = self.extract_code(code_response)
            
            if not code:
                print("❌ No code generated")
                continue
            
            print(f"\n--- CODE ---\n{code[:600]}...")
            
            # Step 4: Execute
            print("\n⚙️  Executing...")
            execution = self.execute_code(code)
            
            if execution.success:
                print("✓ Execution successful!")
                if execution.files_created:
                    print(f"📁 Files: {execution.files_created}")
                
                # Step 5: Verify
                passed, verification = self.verify_output(user_request, final_plan, code, execution)
                print(f"\n--- VERIFICATION ---\n{verification}")
                
                if passed:
                    print("\n🎉 SUCCESS! Output verified.")
                    return True, code, execution
                else:
                    print("\n⚠️  Verification failed, will retry...")
                    # Update plan based on feedback
                    final_plan = f"{final_plan}\n\nPREVIOUS ATTEMPT FEEDBACK:\n{verification}"
            else:
                print(f"❌ Execution failed: {execution.stderr[:200]}")
                # Add error to plan for next iteration
                final_plan = f"{final_plan}\n\nPREVIOUS ERROR (avoid this):\n{execution.stderr[:300]}"
        
        print("\n❌ Max iterations reached")
        return False, code if code else "", execution
    
    def run_interactive(self):
        """Run interactive session."""
        print("=" * 60)
        print("φ-Chat Planner: Interactive Mode")
        print("=" * 60)
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nThis chat uses planning for better outputs.")
        print("Commands: /quit, /files")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                if user_input.lower() == '/files':
                    files = list(OUTPUT_DIR.glob("*"))
                    for f in sorted(files):
                        print(f"  → {f.name}")
                    continue
                
                success, code, execution = self.process_request(user_input)
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")


def main():
    chat = PlannerChat()
    chat.run_interactive()


if __name__ == "__main__":
    main()
