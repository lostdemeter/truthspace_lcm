#!/usr/bin/env python3
"""
φ-Chat: Interactive Code Generation and Execution

An interactive chat interface where you can ask the model to:
- Generate code for any task
- Execute the code automatically
- Save outputs (images, files, etc.)
- Iterate and refine

Example:
    > Create a Valentine's Day card to 'Katie my Love' using matplotlib and save as png
    
The model will generate the code, execute it, and show you the result.
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

# Output directory for generated files
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    files_created: List[str] = field(default_factory=list)


@dataclass
class ChatMessage:
    """A message in the chat history."""
    role: str  # 'user' or 'assistant'
    content: str
    code: Optional[str] = None
    execution: Optional[ExecutionResult] = None


class PhiChat:
    """
    Interactive chat with code generation and execution.
    
    Features:
    - Natural language → code generation
    - Automatic code execution
    - File output handling (images, etc.)
    - Conversation memory
    - Error recovery and iteration
    """
    
    SYSTEM_PROMPT = """You are a helpful Python programming assistant. When the user asks you to create something:

1. Write complete, working Python code
2. Put all code in ```python``` blocks
3. Make sure the code is self-contained with all imports
4. For visualizations, save to the specified filename (or 'output.png' by default)
5. Use matplotlib's savefig() for images, not show()
6. Print a confirmation message when done

Be creative and make things look good! For visual outputs, use nice colors, fonts, and styling.

IMPORTANT: Always save files to the current directory. The system will handle file management."""

    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🚀 Loading φ-Chat model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.history: List[ChatMessage] = []
        print("✓ Model loaded!\n")
    
    def generate(self, user_input: str, max_tokens: int = 800) -> str:
        """Generate a response from the model."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history (last 4 exchanges)
        for msg in self.history[-8:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
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
        
        # Extract just the new response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            response = parts[-1].strip()
        
        return response
    
    def extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from a response."""
        # Try ```python blocks
        match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try any ``` blocks
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def execute_code(self, code: str) -> ExecutionResult:
        """Execute code and capture results."""
        # Create a temp file in the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = OUTPUT_DIR / f"script_{timestamp}.py"
        
        # Get list of files before execution
        files_before = set(OUTPUT_DIR.glob("*"))
        
        # Write and execute the code
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
            
            # Find new files created
            files_after = set(OUTPUT_DIR.glob("*"))
            new_files = files_after - files_before
            # Filter out the script itself
            new_files = [str(f) for f in new_files if f != script_path]
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                files_created=new_files
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Execution timed out (30s limit)"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e)
            )
        finally:
            # Clean up script file
            if script_path.exists():
                script_path.unlink()
    
    def chat(self, user_input: str, max_retries: int = 2) -> Tuple[str, Optional[ExecutionResult]]:
        """
        Process a user message and return response + execution result.
        Automatically retries on errors.
        """
        # Generate response
        response = self.generate(user_input)
        
        # Extract and execute code if present
        code = self.extract_code(response)
        execution = None
        
        if code:
            print("\n📝 Generated code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            print("\n⚙️  Executing...")
            
            execution = self.execute_code(code)
            
            if execution.success:
                print("✓ Execution successful!")
                if execution.stdout:
                    print(f"Output: {execution.stdout.strip()}")
                if execution.files_created:
                    print(f"📁 Files created:")
                    for f in execution.files_created:
                        print(f"   → {f}")
            else:
                print("✗ Execution failed!")
                if execution.stderr:
                    print(f"Error: {execution.stderr[:200]}")
                
                # Auto-retry on error
                retry_count = 0
                while not execution.success and retry_count < max_retries:
                    retry_count += 1
                    print(f"\n🔄 Auto-retry {retry_count}/{max_retries}...")
                    
                    # Ask model to fix the error
                    fix_request = f"""The code failed with this error:

{execution.stderr}

Please fix the code. Remember:
- Don't load external files that don't exist
- Use only matplotlib primitives and math
- Make sure all imports are correct
- Double-check API usage"""
                    
                    fix_response = self.generate(fix_request)
                    fixed_code = self.extract_code(fix_response)
                    
                    if fixed_code:
                        print("\n📝 Fixed code:")
                        print("-" * 40)
                        print(fixed_code)
                        print("-" * 40)
                        print("\n⚙️  Executing...")
                        
                        execution = self.execute_code(fixed_code)
                        code = fixed_code
                        response = fix_response
                        
                        if execution.success:
                            print("✓ Execution successful!")
                            if execution.stdout:
                                print(f"Output: {execution.stdout.strip()}")
                            if execution.files_created:
                                print(f"📁 Files created:")
                                for f in execution.files_created:
                                    print(f"   → {f}")
                        else:
                            print("✗ Still failing!")
                            if execution.stderr:
                                print(f"Error: {execution.stderr[:200]}")
        
        # Store in history
        self.history.append(ChatMessage(role="user", content=user_input))
        self.history.append(ChatMessage(
            role="assistant", 
            content=response,
            code=code,
            execution=execution
        ))
        
        return response, execution
    
    def run_interactive(self):
        """Run an interactive chat session."""
        print("=" * 60)
        print("φ-Chat: Interactive Code Generation")
        print("=" * 60)
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nCommands:")
        print("  /quit or /exit - Exit the chat")
        print("  /clear - Clear conversation history")
        print("  /files - List generated files")
        print("  /retry - Retry last request")
        print("=" * 60)
        print()
        
        last_input = None
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == '/clear':
                    self.history = []
                    print("✓ History cleared")
                    continue
                
                if user_input.lower() == '/files':
                    files = list(OUTPUT_DIR.glob("*"))
                    if files:
                        print("Generated files:")
                        for f in sorted(files):
                            print(f"  → {f.name}")
                    else:
                        print("No files generated yet")
                    continue
                
                if user_input.lower() == '/retry' and last_input:
                    user_input = last_input
                    print(f"Retrying: {user_input}")
                
                last_input = user_input
                
                print("\n🤔 Thinking...")
                response, execution = self.chat(user_input)
                
                # Show response (without code block if we already showed it)
                if execution:
                    # Remove code block from display since we showed it
                    display_response = re.sub(r'```python\n?.*?```', '[code executed above]', response, flags=re.DOTALL)
                    if display_response.strip() and display_response.strip() != '[code executed above]':
                        print(f"\nAssistant: {display_response.strip()}")
                else:
                    print(f"\nAssistant: {response}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Main entry point."""
    chat = PhiChat()
    chat.run_interactive()


if __name__ == "__main__":
    main()
