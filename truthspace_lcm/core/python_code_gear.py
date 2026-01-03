"""
Python Code Gear

An emergent, geometric Python code generation system that:
1. Receives contact points from the orchestrator
2. Matches them to internal code patterns (its "territory")
3. Generates executable Python code
4. Verifies the code runs without error

The gear has its own corpus of Python patterns - this is its "chromosomal
territory". It communicates with other gears through contact points - the
"kissing" interface.

Author: Lesley Gushurst
License: GPLv3
"""

import ast
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .contact_point import (
    ContactPoint, ContactVerb, ContactNoun, ContactStructure,
    ContactMessage, parse_intent,
)
from .gear_message import GearProtocol, GearMessage, MessageIntent

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class CodePattern:
    """
    A Python code pattern in the gear's internal corpus.
    
    This is part of the gear's "territory" - its internal knowledge
    that maps contact points to actual Python code.
    """
    name: str
    contact: ContactPoint  # What contact point this pattern responds to
    template: str  # Python code template with {placeholders}
    description: str
    examples: List[str] = field(default_factory=list)
    test_code: Optional[str] = None  # Code to verify the pattern works
    use_count: int = 0
    success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.use_count == 0:
            return 1.0
        return self.success_count / self.use_count
    
    def fill(self, params: Dict[str, Any]) -> str:
        """Fill the template with parameters."""
        code = self.template
        for key, value in params.items():
            placeholder = '{' + key + '}'
            if placeholder in code:
                code = code.replace(placeholder, str(value))
        return code


@dataclass
class GenerationResult:
    """Result of code generation."""
    success: bool
    code: str
    pattern_used: Optional[str] = None
    error: Optional[str] = None
    verified: bool = False
    output: Optional[str] = None


class PythonCodeCorpus:
    """
    The Python code corpus - the gear's internal "territory".
    
    Contains patterns that map contact points to Python code.
    This is separate from the shared contact vocabulary - it's
    the gear's own rich internal representation.
    """
    
    def __init__(self):
        self.patterns: List[CodePattern] = []
        self._seed_patterns()
    
    def _seed_patterns(self):
        """Seed the corpus with basic Python patterns."""
        
        # =================================================================
        # OUTPUT patterns (print, display)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="print_string",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.TEXT),
            template='print("{message}")',
            description="Print a string message",
            examples=["print hello", "display message", "output text"],
        ))
        
        self.add_pattern(CodePattern(
            name="print_variable",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.NONE),
            template='print({variable})',
            description="Print a variable",
            examples=["print the result", "show the value"],
        ))
        
        self.add_pattern(CodePattern(
            name="print_formatted",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.TEXT),
            template='print(f"{format_string}")',
            description="Print formatted string",
            examples=["print with formatting", "display formatted"],
        ))
        
        # =================================================================
        # CREATE patterns (variables, functions)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="create_variable_string",
            contact=ContactPoint(ContactVerb.CREATE, ContactNoun.TEXT),
            template='{name} = "{value}"',
            description="Create a string variable",
            examples=["create a string", "make a text variable"],
        ))
        
        self.add_pattern(CodePattern(
            name="create_variable_number",
            contact=ContactPoint(ContactVerb.CREATE, ContactNoun.NUMBER),
            template='{name} = {value}',
            description="Create a numeric variable",
            examples=["create a number", "make an integer"],
        ))
        
        self.add_pattern(CodePattern(
            name="create_variable_list",
            contact=ContactPoint(ContactVerb.CREATE, ContactNoun.SEQUENCE),
            template='{name} = [{items}]',
            description="Create a list",
            examples=["create a list", "make an array"],
        ))
        
        self.add_pattern(CodePattern(
            name="create_function",
            contact=ContactPoint(ContactVerb.CREATE, ContactNoun.NONE),
            template='''def {name}({args}):
    {body}''',
            description="Create a function",
            examples=["create a function", "define a function"],
        ))
        
        self.add_pattern(CodePattern(
            name="create_main",
            contact=ContactPoint(ContactVerb.CREATE, ContactNoun.NONE),
            template='''def main():
    {body}

if __name__ == "__main__":
    main()''',
            description="Create main function with entry point",
            examples=["create main", "make a program"],
        ))
        
        # =================================================================
        # READ patterns (input, file reading)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="read_input",
            contact=ContactPoint(ContactVerb.READ, ContactNoun.TEXT),
            template='{name} = input("{prompt}")',
            description="Read user input",
            examples=["get input", "read from user", "ask user"],
        ))
        
        self.add_pattern(CodePattern(
            name="read_file",
            contact=ContactPoint(ContactVerb.READ, ContactNoun.FILE),
            template='''with open("{filename}", "r") as f:
    {name} = f.read()''',
            description="Read entire file",
            examples=["read a file", "load file contents"],
        ))
        
        self.add_pattern(CodePattern(
            name="read_file_lines",
            contact=ContactPoint(ContactVerb.READ, ContactNoun.FILE, ContactStructure.REPEAT),
            template='''with open("{filename}", "r") as f:
    for line in f:
        {body}''',
            description="Read file line by line",
            examples=["read each line", "iterate file lines"],
        ))
        
        # =================================================================
        # TRANSFORM patterns (calculations, processing)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="calculate_sum",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.NUMBER),
            template='{result} = {a} + {b}',
            description="Add two numbers",
            examples=["add numbers", "calculate sum"],
        ))
        
        self.add_pattern(CodePattern(
            name="calculate_expression",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.NUMBER),
            template='{result} = {expression}',
            description="Calculate an expression",
            examples=["calculate", "compute"],
        ))
        
        self.add_pattern(CodePattern(
            name="transform_list",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.SEQUENCE),
            template='{result} = [{transform} for item in {source}]',
            description="Transform a list with comprehension",
            examples=["transform list", "map over items"],
        ))
        
        self.add_pattern(CodePattern(
            name="sum_list",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.SEQUENCE),
            template='{result} = sum({source})',
            description="Sum a list of numbers",
            examples=["sum the list", "total the numbers"],
        ))
        
        # =================================================================
        # CONTROL FLOW patterns (loops, conditionals)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="for_loop",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.SEQUENCE, ContactStructure.REPEAT),
            template='''for {item} in {sequence}:
    {body}''',
            description="Iterate over a sequence",
            examples=["for each item", "loop through"],
        ))
        
        self.add_pattern(CodePattern(
            name="for_range",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.NUMBER, ContactStructure.REPEAT),
            template='''for {item} in range({start}, {end}):
    {body}''',
            description="Loop through a range of numbers",
            examples=["for numbers from", "loop from to"],
        ))
        
        self.add_pattern(CodePattern(
            name="while_loop",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.BOOLEAN, ContactStructure.REPEAT),
            template='''while {condition}:
    {body}''',
            description="While loop",
            examples=["while condition", "repeat until"],
        ))
        
        self.add_pattern(CodePattern(
            name="if_statement",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.BOOLEAN, ContactStructure.BRANCH),
            template='''if {condition}:
    {then_body}''',
            description="If statement",
            examples=["if condition", "when true"],
        ))
        
        self.add_pattern(CodePattern(
            name="if_else",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.BOOLEAN, ContactStructure.BRANCH),
            template='''if {condition}:
    {then_body}
else:
    {else_body}''',
            description="If-else statement",
            examples=["if else", "condition with alternative"],
        ))
        
        # =================================================================
        # FILE OUTPUT patterns
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="write_file",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.FILE),
            template='''with open("{filename}", "w") as f:
    f.write({content})''',
            description="Write to a file",
            examples=["write to file", "save to file"],
        ))
        
        self.add_pattern(CodePattern(
            name="append_file",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.FILE),
            template='''with open("{filename}", "a") as f:
    f.write({content})''',
            description="Append to a file",
            examples=["append to file", "add to file"],
        ))
        
        # =================================================================
        # COMPOSITE patterns (full programs)
        # =================================================================
        
        self.add_pattern(CodePattern(
            name="hello_world",
            contact=ContactPoint(ContactVerb.OUTPUT, ContactNoun.TEXT),
            template='''def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()''',
            description="Hello World program",
            examples=["hello world", "simple program"],
        ))
        
        self.add_pattern(CodePattern(
            name="read_print_file",
            contact=ContactPoint(ContactVerb.READ, ContactNoun.FILE, ContactStructure.COMPOSE),
            template='''def main():
    with open("{filename}", "r") as f:
        for line in f:
            print(line.strip())

if __name__ == "__main__":
    main()''',
            description="Read and print file contents",
            examples=["read file and print", "display file contents"],
        ))
        
        self.add_pattern(CodePattern(
            name="sum_numbers_program",
            contact=ContactPoint(ContactVerb.TRANSFORM, ContactNoun.NUMBER, ContactStructure.COMPOSE),
            template='''def main():
    numbers = [{numbers}]
    total = sum(numbers)
    print(f"Sum: {total}")

if __name__ == "__main__":
    main()''',
            description="Sum numbers and print result",
            examples=["sum numbers", "add up and print"],
        ))
    
    def add_pattern(self, pattern: CodePattern):
        """Add a pattern to the corpus."""
        self.patterns.append(pattern)
    
    def find_pattern(self, contact: ContactPoint, threshold: float = 0.7) -> Optional[CodePattern]:
        """
        Find the best matching pattern for a contact point.
        
        This is where the "kiss" happens - we find patterns whose
        contact points match (kiss) the incoming contact.
        """
        best_match = None
        best_similarity = threshold
        
        for pattern in self.patterns:
            similarity = contact.similarity(pattern.contact)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        
        return best_match
    
    def find_patterns(self, contact: ContactPoint, threshold: float = 0.5, 
                      limit: int = 5) -> List[Tuple[CodePattern, float]]:
        """Find all matching patterns above threshold."""
        matches = []
        for pattern in self.patterns:
            similarity = contact.similarity(pattern.contact)
            if similarity >= threshold:
                matches.append((pattern, similarity))
        
        matches.sort(key=lambda x: -x[1])
        return matches[:limit]
    
    def record_use(self, pattern_name: str, success: bool):
        """Record pattern usage for learning."""
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                pattern.use_count += 1
                if success:
                    pattern.success_count += 1
                break
    
    def save(self, path: str):
        """Save corpus to file."""
        data = {
            'patterns': [
                {
                    'name': p.name,
                    'contact': p.contact.to_dict(),
                    'template': p.template,
                    'description': p.description,
                    'examples': p.examples,
                    'test_code': p.test_code,
                    'use_count': p.use_count,
                    'success_count': p.success_count,
                }
                for p in self.patterns
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load corpus from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.patterns = []
        for p in data['patterns']:
            self.patterns.append(CodePattern(
                name=p['name'],
                contact=ContactPoint.from_dict(p['contact']),
                template=p['template'],
                description=p['description'],
                examples=p.get('examples', []),
                test_code=p.get('test_code'),
                use_count=p.get('use_count', 0),
                success_count=p.get('success_count', 0),
            ))


class CodeVerifier:
    """
    Verifies that generated Python code is valid and runs.
    """
    
    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    @staticmethod
    def run_code(code: str, timeout: int = 5) -> Tuple[bool, str]:
        """
        Run code in a subprocess and capture output.
        
        Returns (success, output_or_error)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        except Exception as e:
            return False, str(e)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class PythonCodeGear(GearProtocol):
    """
    The Python Code Generation Gear.
    
    This gear has its own "territory" (corpus of Python patterns) and
    communicates with other gears through contact points (the "kiss").
    
    It can:
    1. Receive contact points describing what code to generate
    2. Match them to internal patterns
    3. Generate Python code
    4. Verify the code runs
    5. Learn from successes and failures
    """
    
    def __init__(self, corpus_path: Optional[str] = None):
        self.name = "PythonCodeGear"
        self.corpus = PythonCodeCorpus()
        self.verifier = CodeVerifier()
        
        # LLM for fallback when stuck
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Corpus path for auto-saving learned patterns
        self.corpus_path = corpus_path
        if self.corpus_path is None:
            # Default path
            self.corpus_path = str(Path(__file__).parent.parent.parent.parent / "data" / "python_code_corpus.json")
        
        # Load corpus if exists
        if self.corpus_path and Path(self.corpus_path).exists():
            self.corpus.load(self.corpus_path)
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for fallback generation."""
        self.llm_url = url
        self.llm_model = model
    
    def generate(self, contact: ContactPoint, context: Dict[str, Any] = None) -> GenerationResult:
        """
        Generate Python code from a contact point.
        
        This is the main entry point - the orchestrator sends a contact
        point, and we return generated code.
        """
        context = context or {}
        
        # Find matching pattern
        pattern = self.corpus.find_pattern(contact)
        
        if pattern is None:
            # No pattern found - try LLM fallback
            if self.llm_url:
                return self._llm_fallback(contact, context)
            return GenerationResult(
                success=False,
                code="",
                error="No matching pattern found",
            )
        
        # Fill template with context
        try:
            code = self._fill_pattern(pattern, contact, context)
        except Exception as e:
            return GenerationResult(
                success=False,
                code="",
                pattern_used=pattern.name,
                error=f"Failed to fill template: {e}",
            )
        
        # Verify syntax
        syntax_ok, syntax_error = self.verifier.check_syntax(code)
        if not syntax_ok:
            self.corpus.record_use(pattern.name, False)
            return GenerationResult(
                success=False,
                code=code,
                pattern_used=pattern.name,
                error=syntax_error,
            )
        
        # Verify execution (optional - only for complete programs)
        verified = False
        output = None
        if 'if __name__' in code or context.get('verify', False):
            run_ok, run_output = self.verifier.run_code(code)
            verified = run_ok
            output = run_output
            if not run_ok:
                self.corpus.record_use(pattern.name, False)
                return GenerationResult(
                    success=False,
                    code=code,
                    pattern_used=pattern.name,
                    error=f"Execution failed: {run_output}",
                    verified=False,
                )
        
        # Success!
        self.corpus.record_use(pattern.name, True)
        return GenerationResult(
            success=True,
            code=code,
            pattern_used=pattern.name,
            verified=verified,
            output=output,
        )
    
    def _fill_pattern(self, pattern: CodePattern, contact: ContactPoint, 
                      context: Dict[str, Any]) -> str:
        """Fill a pattern template with values from contact and context."""
        params = {}
        
        # Get values from contact params
        params.update(contact.params)
        
        # Get values from context
        params.update(context)
        
        # Provide defaults for common placeholders
        defaults = {
            'name': 'result',
            'variable': 'x',
            'value': '0',
            'message': 'Hello',
            'filename': 'input.txt',
            'prompt': 'Enter value: ',
            'item': 'item',
            'sequence': 'items',
            'body': 'pass',
            'then_body': 'pass',
            'else_body': 'pass',
            'condition': 'True',
            'args': '',
            'items': '',
            'numbers': '1, 2, 3',
            'content': '"content"',
            'a': '0',
            'b': '0',
            'result': 'result',
            'expression': '0',
            'source': '[]',
            'transform': 'item',
            'start': '0',
            'end': '10',
            'format_string': '{value}',
            'total': 'total',
        }
        
        for key, default in defaults.items():
            if key not in params:
                params[key] = default
        
        # Handle special cases
        if 'values' in contact.params and contact.params['values']:
            params['message'] = contact.params['values'][0]
            params['content'] = f'"{contact.params["values"][0]}"'
        
        if 'numbers' in contact.params and contact.params['numbers']:
            params['numbers'] = ', '.join(str(n) for n in contact.params['numbers'])
            if len(contact.params['numbers']) >= 2:
                params['a'] = str(contact.params['numbers'][0])
                params['b'] = str(contact.params['numbers'][1])
        
        return pattern.fill(params)
    
    def _llm_generate(self, original_request: str, contact: ContactPoint = None, 
                       context: Dict[str, Any] = None) -> GenerationResult:
        """Use LLM to generate code, then learn from successful generations."""
        if not self.llm_url:
            return GenerationResult(
                success=False,
                code="",
                error="No LLM configured for code generation",
            )
        
        # Use the original request for better context
        prompt = f"""Generate Python code for this request:

"{original_request}"

Rules:
1. Keep it simple - prefer standard library only
2. Include a main() function with if __name__ == "__main__" for complete programs
3. Make sure it runs without errors
4. Output ONLY the Python code, no explanations or markdown"""

        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 800, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                code = response.json().get('response', '').strip()
                
                # Clean up code (remove markdown if present)
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()
                
                # Verify syntax
                syntax_ok, syntax_err = self.verifier.check_syntax(code)
                if not syntax_ok:
                    return GenerationResult(
                        success=False,
                        code=code,
                        pattern_used="llm_generation",
                        error=f"LLM generated invalid syntax: {syntax_err}",
                    )
                
                # Verify execution for complete programs
                verified = False
                output = None
                if 'if __name__' in code:
                    run_ok, run_output = self.verifier.run_code(code)
                    verified = run_ok
                    output = run_output
                    if not run_ok:
                        return GenerationResult(
                            success=False,
                            code=code,
                            pattern_used="llm_generation",
                            error=f"LLM code failed to run: {run_output}",
                        )
                
                # SUCCESS! Learn from this generation
                self._learn_from_generation(original_request, code, contact)
                
                return GenerationResult(
                    success=True,
                    code=code,
                    pattern_used="llm_generation",
                    verified=verified,
                    output=output,
                )
        except Exception as e:
            return GenerationResult(
                success=False,
                code="",
                error=f"LLM request failed: {e}",
            )
        
        return GenerationResult(
            success=False,
            code="",
            error="LLM generation failed",
        )
    
    def _learn_from_generation(self, request: str, code: str, contact: ContactPoint = None):
        """
        Learn from a successful LLM generation by adding it to the corpus.
        
        This is how the gear grows its knowledge over time.
        """
        if contact is None:
            contact = parse_intent(request)
        
        # Create a unique pattern name from the request
        import hashlib
        request_hash = hashlib.md5(request.lower().encode()).hexdigest()[:8]
        pattern_name = f"learned_{request_hash}"
        
        # Check if we already have this pattern
        for p in self.corpus.patterns:
            if p.name == pattern_name:
                return  # Already learned
        
        # Add new pattern
        new_pattern = CodePattern(
            name=pattern_name,
            contact=contact,
            template=code,  # The exact code becomes the template
            description=f"Learned from: {request[:100]}",
            examples=[request],
            use_count=1,
            success_count=1,
        )
        self.corpus.add_pattern(new_pattern)
        
        # Auto-save corpus if path is configured
        if hasattr(self, 'corpus_path') and self.corpus_path:
            self.corpus.save(self.corpus_path)
    
    def generate_from_text(self, text: str, context: Dict[str, Any] = None) -> GenerationResult:
        """
        Generate code from natural language description.
        
        Strategy:
        1. Check for exact pattern matches (like "hello world")
        2. Try pattern matching with high confidence threshold
        3. If pattern match is weak OR code doesn't verify, use LLM
        4. Learn from successful LLM generations
        """
        text_lower = text.lower()
        context = context or {}
        original_request = text  # Keep original for LLM
        
        # Check for exact "hello world" pattern (very specific)
        if 'hello world' in text_lower and 'hello' in text_lower.split():
            # Only match if it's actually asking for hello world, not "hello George"
            words_after_hello = text_lower.split('hello')[-1].strip()
            if words_after_hello.startswith('world') or words_after_hello == '':
                for pattern in self.corpus.patterns:
                    if pattern.name == 'hello_world':
                        code = pattern.template
                        self.corpus.record_use(pattern.name, True)
                        return GenerationResult(
                            success=True,
                            code=code,
                            pattern_used=pattern.name,
                            verified=True,
                        )
        
        # Parse intent to contact point
        contact = parse_intent(text)
        
        # Find best matching pattern with HIGH threshold
        pattern = self.corpus.find_pattern(contact, threshold=0.85)
        
        if pattern is not None:
            # Try the pattern
            try:
                code = self._fill_pattern(pattern, contact, context)
                
                # Verify syntax
                syntax_ok, _ = self.verifier.check_syntax(code)
                if syntax_ok:
                    # For complete programs, also verify execution
                    if 'if __name__' in code:
                        run_ok, output = self.verifier.run_code(code)
                        if run_ok:
                            self.corpus.record_use(pattern.name, True)
                            return GenerationResult(
                                success=True,
                                code=code,
                                pattern_used=pattern.name,
                                verified=True,
                                output=output,
                            )
                        # Execution failed - fall through to LLM
                    else:
                        # Simple snippet, syntax is enough
                        self.corpus.record_use(pattern.name, True)
                        return GenerationResult(
                            success=True,
                            code=code,
                            pattern_used=pattern.name,
                            verified=False,
                        )
            except Exception:
                pass  # Pattern fill failed, fall through to LLM
        
        # No good pattern match or pattern failed - use LLM
        return self._llm_generate(original_request, contact, context)
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Process a gear message (implements GearProtocol)."""
        # Parse contact from message
        contact = parse_intent(message.content)
        
        # Generate code
        result = self.generate(contact, message.context)
        
        if result.success:
            response = f"```python\n{result.code}\n```"
            if result.verified:
                response += f"\n\n✓ Verified - runs successfully"
                if result.output:
                    response += f"\nOutput: {result.output[:200]}"
        else:
            response = f"Failed to generate code: {result.error}"
        
        return self.send(
            message.with_context('generation_result', {
                'success': result.success,
                'code': result.code,
                'pattern': result.pattern_used,
                'error': result.error,
            }),
            content=response,
            intent=MessageIntent.RESPONSE,
        )
    
    def handle_contact(self, message: ContactMessage) -> ContactMessage:
        """
        Handle a contact message from another gear.
        
        This is the "kiss" interface - how other gears communicate with us.
        """
        result = self.generate(message.contact, message.context)
        
        return message.with_response(
            response=result.code if result.success else result.error,
            success=result.success,
        )
    
    def save_corpus(self, path: str):
        """Save the corpus to a file."""
        self.corpus.save(path)
    
    def load_corpus(self, path: str):
        """Load the corpus from a file."""
        self.corpus.load(path)


if __name__ == "__main__":
    # Test the Python code gear
    print("=== Python Code Gear Test ===\n")
    
    gear = PythonCodeGear()
    
    test_requests = [
        "print hello world",
        "create a list of numbers",
        "read a file and print each line",
        "calculate the sum of 5 and 10",
        "if x is greater than 10 print it",
    ]
    
    for request in test_requests:
        print(f"Request: '{request}'")
        result = gear.generate_from_text(request)
        print(f"Success: {result.success}")
        print(f"Pattern: {result.pattern_used}")
        if result.success:
            print(f"Code:\n{result.code}")
        else:
            print(f"Error: {result.error}")
        print("-" * 40)
