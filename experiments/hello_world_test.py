#!/usr/bin/env python3
"""
Hello World Test: Verify the geometric AI can generate, test, and verify code.

This is the simplest possible test:
1. Ask the model to write a "hello world" program
2. Execute the generated code
3. Verify the output is correct

If this works, we can move to more complex tasks.
"""

import torch
import re
import subprocess
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_hello_world():
    """Test that the model can generate a working hello world program."""
    
    print("=" * 60)
    print("HELLO WORLD CODE GENERATION TEST")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading Qwen2-7B model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Prepare prompt
    print("\n[2/5] Generating code...")
    
    messages = [
        {"role": "system", "content": "You are a Python programmer. Write only code, no explanations."},
        {"role": "user", "content": "Write a Python program that prints 'Hello, World!' to the console. Output only the code inside ```python``` blocks."}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=False,  # Deterministic for testing
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response (after the prompt)
    response = generated.split("assistant")[-1].strip() if "assistant" in generated.lower() else generated
    
    print(f"\n=== MODEL OUTPUT ===")
    print(response)
    print("=" * 40)
    
    # Extract code from response
    print("\n[3/5] Extracting code...")
    
    # Try to extract code between ```python and ```
    code_match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try to find just a print statement
        print_match = re.search(r'(print\([^\)]+\))', response)
        if print_match:
            code = print_match.group(1)
        else:
            # Last resort: take everything that looks like code
            code = response.strip()
    
    print(f"Extracted code:\n{code}")
    print("=" * 40)
    
    # Execute the code
    print("\n[4/5] Executing code...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: '{result.stdout.strip()}'")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Verify output
        print("\n[5/5] Verifying output...")
        
        success = True
        checks = []
        
        # Check 1: Exit code
        if result.returncode == 0:
            checks.append("✓ Code executed successfully (exit code 0)")
        else:
            checks.append("✗ Code failed to execute")
            success = False
        
        # Check 2: Output contains expected text
        output_lower = result.stdout.lower()
        if "hello" in output_lower and "world" in output_lower:
            checks.append("✓ Output contains 'hello' and 'world'")
        elif "hello" in output_lower or "world" in output_lower:
            checks.append("~ Output contains partial match")
        else:
            checks.append("✗ Output doesn't match expected pattern")
            success = False
        
        for check in checks:
            print(f"  {check}")
        
        print("\n" + "=" * 60)
        if success:
            print("TEST PASSED: Model successfully generated working code!")
        else:
            print("TEST FAILED: Code generation or execution failed")
        print("=" * 60)
        
        return success, code, result.stdout.strip()
        
    except subprocess.TimeoutExpired:
        print("ERROR: Code execution timed out")
        return False, code, None
    except Exception as e:
        print(f"ERROR: {e}")
        return False, code, None
    finally:
        os.unlink(temp_file)


def test_with_tool_agent():
    """Test using the tool agent infrastructure."""
    from phi_tool_agent import PhiToolAgent
    
    print("\n" + "=" * 60)
    print("TOOL AGENT HELLO WORLD TEST")
    print("=" * 60)
    
    agent = PhiToolAgent()
    
    # Simple task: write and execute hello world
    problem = """Write a Python program that prints 'Hello, World!' to the console.
Use the python_exec tool to run your code and verify it works.
Then submit your answer with the code you wrote."""
    
    session = agent.solve(problem, verbose=True)
    
    return session.solved


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--agent":
        # Use the full tool agent
        success = test_with_tool_agent()
    else:
        # Simple direct test
        success, code, output = test_hello_world()
    
    sys.exit(0 if success else 1)
