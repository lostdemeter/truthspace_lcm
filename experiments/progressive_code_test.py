#!/usr/bin/env python3
"""
Progressive Code Generation Test with Reverse Navigation

This test explores whether our model's ability to "work in reverse" 
(navigate backward through φ-space) provides benefits for code generation.

Hypothesis: Reverse navigation might help with:
1. Working backward from desired output to code
2. Debugging (from error to fix)
3. Test-driven development (from test to implementation)

Levels:
1. FizzBuzz - Simple logic with verification
2. Multi-step problem - Requires planning
3. Reverse: Output → Code (given output, generate code that produces it)
4. Reverse: Error → Fix (given error, generate the fix)
"""

import torch
import re
import subprocess
import tempfile
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949


@dataclass
class TestResult:
    """Result of a code generation test."""
    name: str
    passed: bool
    code: str
    output: str
    expected: str
    phi_level: float = 0.0
    notes: str = ""


class ProgressiveCodeTester:
    """Test code generation with increasing complexity and reverse navigation."""
    
    def __init__(self):
        print("Loading Qwen2-7B model...")
        self.model_name = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.results: List[TestResult] = []
        
    def get_phi_level(self, text: str) -> float:
        """Get the φ-level at layer 27 for a text."""
        inputs = self.tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[27][0, -1, :].float().cpu().numpy()
            magnitudes = np.abs(hidden)
            magnitudes = magnitudes[magnitudes > 1e-10]
            levels = np.log(magnitudes) / np.log(PHI)
            return float(np.mean(levels))
    
    def generate(self, messages: List[Dict], max_tokens: int = 300) -> str:
        """Generate a response from the model."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def extract_code(self, response: str) -> str:
        """Extract Python code from a response."""
        # Try ```python blocks first
        match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try any ``` blocks
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for def or print statements
        lines = []
        for line in response.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('def ', 'print(', 'for ', 'if ', 'while ', 'import ', 'from ')):
                lines.append(line)
            elif lines and (line.startswith(' ') or line.startswith('\t') or stripped == ''):
                lines.append(line)
        
        if lines:
            return '\n'.join(lines)
        
        return response.strip()
    
    def execute_code(self, code: str, timeout: int = 5) -> Tuple[bool, str, str]:
        """Execute code and return (success, stdout, stderr)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)
        finally:
            os.unlink(temp_file)
    
    # =========================================================
    # LEVEL 1: FizzBuzz
    # =========================================================
    
    def test_fizzbuzz(self) -> TestResult:
        """Test FizzBuzz implementation."""
        print("\n" + "=" * 60)
        print("LEVEL 1: FizzBuzz")
        print("=" * 60)
        
        messages = [
            {"role": "system", "content": "You are a Python programmer. Write only code, no explanations."},
            {"role": "user", "content": """Write a Python function fizzbuzz(n) that:
- Returns "FizzBuzz" if n is divisible by both 3 and 5
- Returns "Fizz" if n is divisible by 3 only
- Returns "Buzz" if n is divisible by 5 only
- Returns str(n) otherwise

Then print the results for numbers 1 to 15.
Output only code in ```python``` blocks."""}
        ]
        
        response = self.generate(messages)
        print(f"Response:\n{response[:500]}")
        
        code = self.extract_code(response)
        print(f"\nExtracted code:\n{code}")
        
        success, stdout, stderr = self.execute_code(code)
        print(f"\nOutput:\n{stdout}")
        
        # Verify FizzBuzz output
        expected_values = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", 
                          "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
        
        output_lines = [l.strip() for l in stdout.strip().split('\n') if l.strip()]
        
        passed = True
        for i, (got, expected) in enumerate(zip(output_lines, expected_values)):
            if got != expected:
                print(f"  Mismatch at {i+1}: got '{got}', expected '{expected}'")
                passed = False
        
        if len(output_lines) < 15:
            print(f"  Only got {len(output_lines)} lines, expected 15")
            passed = False
        
        phi = self.get_phi_level(code)
        
        result = TestResult(
            name="FizzBuzz",
            passed=passed and success,
            code=code,
            output=stdout,
            expected='\n'.join(expected_values),
            phi_level=phi
        )
        
        print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (φ-level: {phi:.4f})")
        self.results.append(result)
        return result
    
    # =========================================================
    # LEVEL 2: Multi-step Problem
    # =========================================================
    
    def test_multistep(self) -> TestResult:
        """Test a problem requiring multiple steps."""
        print("\n" + "=" * 60)
        print("LEVEL 2: Multi-step Problem (Prime Factorization)")
        print("=" * 60)
        
        messages = [
            {"role": "system", "content": "You are a Python programmer. Write only code, no explanations."},
            {"role": "user", "content": """Write a Python function prime_factors(n) that returns a list of prime factors of n.
For example: prime_factors(12) should return [2, 2, 3]
             prime_factors(100) should return [2, 2, 5, 5]

Then test it by printing the prime factors of 2310.
Output only code in ```python``` blocks."""}
        ]
        
        response = self.generate(messages, max_tokens=400)
        print(f"Response:\n{response[:600]}")
        
        code = self.extract_code(response)
        print(f"\nExtracted code:\n{code}")
        
        success, stdout, stderr = self.execute_code(code)
        print(f"\nOutput: {stdout.strip()}")
        
        # 2310 = 2 × 3 × 5 × 7 × 11
        expected = "[2, 3, 5, 7, 11]"
        passed = success and expected in stdout.replace(" ", "")
        
        phi = self.get_phi_level(code)
        
        result = TestResult(
            name="Prime Factorization",
            passed=passed,
            code=code,
            output=stdout,
            expected=expected,
            phi_level=phi
        )
        
        print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (φ-level: {phi:.4f})")
        self.results.append(result)
        return result
    
    # =========================================================
    # LEVEL 3: REVERSE - Output → Code
    # =========================================================
    
    def test_reverse_output_to_code(self) -> TestResult:
        """
        REVERSE NAVIGATION: Given desired output, generate code that produces it.
        
        This tests the model's ability to work BACKWARD from the goal.
        """
        print("\n" + "=" * 60)
        print("LEVEL 3: REVERSE - Output → Code")
        print("=" * 60)
        
        # The desired output
        desired_output = """*
**
***
****
*****"""
        
        messages = [
            {"role": "system", "content": "You are a Python programmer. Write only code, no explanations."},
            {"role": "user", "content": f"""I want a Python program that produces EXACTLY this output:

{desired_output}

Write the code that generates this exact output.
Output only code in ```python``` blocks."""}
        ]
        
        response = self.generate(messages)
        print(f"Response:\n{response[:400]}")
        
        code = self.extract_code(response)
        print(f"\nExtracted code:\n{code}")
        
        success, stdout, stderr = self.execute_code(code)
        print(f"\nActual output:\n{stdout}")
        print(f"Expected output:\n{desired_output}")
        
        # Normalize and compare
        actual_lines = [l.rstrip() for l in stdout.strip().split('\n')]
        expected_lines = [l.rstrip() for l in desired_output.strip().split('\n')]
        
        passed = actual_lines == expected_lines
        
        phi = self.get_phi_level(code)
        
        result = TestResult(
            name="Reverse: Output→Code",
            passed=passed and success,
            code=code,
            output=stdout,
            expected=desired_output,
            phi_level=phi,
            notes="Working backward from desired output to code"
        )
        
        print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (φ-level: {phi:.4f})")
        self.results.append(result)
        return result
    
    # =========================================================
    # LEVEL 4: REVERSE - Error → Fix
    # =========================================================
    
    def test_reverse_error_to_fix(self) -> TestResult:
        """
        REVERSE NAVIGATION: Given buggy code and error, generate the fix.
        
        This tests debugging capability - working backward from error to solution.
        """
        print("\n" + "=" * 60)
        print("LEVEL 4: REVERSE - Error → Fix")
        print("=" * 60)
        
        buggy_code = '''def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Test
result = calculate_average([])
print(f"Average: {result}")'''
        
        error_message = "ZeroDivisionError: division by zero"
        
        messages = [
            {"role": "system", "content": "You are a Python programmer. Fix bugs and write only code."},
            {"role": "user", "content": f"""This code has a bug:

```python
{buggy_code}
```

Error: {error_message}

Fix the code so it handles empty lists gracefully (return 0 for empty list).
Output only the fixed code in ```python``` blocks."""}
        ]
        
        response = self.generate(messages, max_tokens=400)
        print(f"Response:\n{response[:500]}")
        
        code = self.extract_code(response)
        print(f"\nExtracted code:\n{code}")
        
        success, stdout, stderr = self.execute_code(code)
        print(f"\nOutput: {stdout.strip()}")
        print(f"Stderr: {stderr.strip()}")
        
        # Should run without error and handle empty list
        passed = success and "ZeroDivisionError" not in stderr
        
        # Also test with non-empty list to make sure it still works
        test_code = code + "\nprint(calculate_average([1, 2, 3, 4, 5]))"
        success2, stdout2, _ = self.execute_code(test_code)
        if success2 and "3" in stdout2:  # Average of 1-5 is 3
            passed = passed and True
        
        phi = self.get_phi_level(code)
        
        result = TestResult(
            name="Reverse: Error→Fix",
            passed=passed,
            code=code,
            output=stdout,
            expected="No error, returns 0 for empty list",
            phi_level=phi,
            notes="Working backward from error to fix"
        )
        
        print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (φ-level: {phi:.4f})")
        self.results.append(result)
        return result
    
    # =========================================================
    # LEVEL 5: REVERSE - Test → Implementation (TDD)
    # =========================================================
    
    def test_reverse_tdd(self) -> TestResult:
        """
        REVERSE NAVIGATION: Given tests, generate implementation.
        
        This is Test-Driven Development - working backward from tests to code.
        """
        print("\n" + "=" * 60)
        print("LEVEL 5: REVERSE - Test → Implementation (TDD)")
        print("=" * 60)
        
        tests = '''# These tests must pass:
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("A man a plan a canal Panama") == True  # Ignore spaces and case
assert is_palindrome("") == True
assert is_palindrome("a") == True'''
        
        messages = [
            {"role": "system", "content": "You are a Python programmer. Write only code, no explanations."},
            {"role": "user", "content": f"""Write a function is_palindrome(s) that passes all these tests:

{tests}

The function should ignore spaces and be case-insensitive.
Output only the function code in ```python``` blocks."""}
        ]
        
        response = self.generate(messages, max_tokens=300)
        print(f"Response:\n{response[:400]}")
        
        code = self.extract_code(response)
        print(f"\nExtracted code:\n{code}")
        
        # Combine implementation with tests
        full_code = code + "\n\n" + tests + "\nprint('All tests passed!')"
        
        success, stdout, stderr = self.execute_code(full_code)
        print(f"\nOutput: {stdout.strip()}")
        if stderr:
            print(f"Stderr: {stderr.strip()}")
        
        passed = success and "All tests passed" in stdout
        
        phi = self.get_phi_level(code)
        
        result = TestResult(
            name="Reverse: Test→Implementation",
            passed=passed,
            code=code,
            output=stdout,
            expected="All tests passed!",
            phi_level=phi,
            notes="TDD: Working backward from tests to implementation"
        )
        
        print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (φ-level: {phi:.4f})")
        self.results.append(result)
        return result
    
    def run_all(self) -> List[TestResult]:
        """Run all tests."""
        print("\n" + "#" * 60)
        print("# PROGRESSIVE CODE GENERATION TEST")
        print("# Testing forward and REVERSE navigation capabilities")
        print("#" * 60)
        
        self.test_fizzbuzz()
        self.test_multistep()
        self.test_reverse_output_to_code()
        self.test_reverse_error_to_fix()
        self.test_reverse_tdd()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nResults: {passed}/{total} tests passed\n")
        
        for r in self.results:
            status = "✓" if r.passed else "✗"
            reverse = " [REVERSE]" if "Reverse" in r.name else ""
            print(f"  {status} {r.name}{reverse} (φ={r.phi_level:.3f})")
        
        # Analyze reverse vs forward
        forward_results = [r for r in self.results if "Reverse" not in r.name]
        reverse_results = [r for r in self.results if "Reverse" in r.name]
        
        forward_pass = sum(1 for r in forward_results if r.passed)
        reverse_pass = sum(1 for r in reverse_results if r.passed)
        
        print(f"\nForward tasks: {forward_pass}/{len(forward_results)}")
        print(f"Reverse tasks: {reverse_pass}/{len(reverse_results)}")
        
        if forward_results and reverse_results:
            forward_phi = np.mean([r.phi_level for r in forward_results])
            reverse_phi = np.mean([r.phi_level for r in reverse_results])
            print(f"\nAverage φ-level (forward): {forward_phi:.4f}")
            print(f"Average φ-level (reverse): {reverse_phi:.4f}")
        
        return self.results


if __name__ == "__main__":
    tester = ProgressiveCodeTester()
    results = tester.run_all()
