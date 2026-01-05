"""
CodeSpace - HyperMapping-based Code Generation

Replaces PythonCodeGear with a cleaner, more geometric architecture.

The key insight: Code generation IS a HyperMapping problem:
- Input = natural language description
- Output = code template
- Position = geometric encoding of the pattern's meaning

Design Principles:
- Bootstrap patterns from corpus file (the ONLY hardcoding)
- Geometric matching for pattern selection
- Learning through position reinforcement
- No magic numbers - all thresholds are geometric (critical line)

Author: Lesley Gushurst
License: GPLv3
"""

import ast
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# Import from hypermapping package
import sys
hypermapping_parent = Path(__file__).parent.parent.parent
if str(hypermapping_parent) not in sys.path:
    sys.path.insert(0, str(hypermapping_parent))

from hypermapping import HyperMapping, Mapping, MatchResult, TextEncoder, CRITICAL_LINE


@dataclass
class CodeResult:
    """Result of code generation."""
    success: bool
    code: str
    pattern_name: Optional[str] = None
    error: Optional[str] = None
    verified: bool = False
    output: Optional[str] = None


class CodeVerifier:
    """
    Verifies that generated Python code is valid and runs.
    
    This is a utility class - no state, just verification functions.
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


class CodeSpace(HyperMapping):
    """
    Code generation using HyperMapping.
    
    Replaces PythonCodeGear with a cleaner, more geometric architecture.
    
    Key differences from PythonCodeGear:
    - Uses HyperMapping's position-based matching
    - Patterns stored as Mappings with metadata
    - Learning through position reinforcement
    - Bootstrap from corpus file (JSON)
    
    Usage:
        space = CodeSpace()
        space.bootstrap_from_corpus("python_code_corpus.json")
        
        # Generate code
        result = space.generate("print hello world")
        
        # Verify and provide feedback
        if result.verified:
            space.feedback(result.pattern_name, success=True)
    """
    
    def __init__(self, dims: int = 8, name: str = "code"):
        encoder = TextEncoder(dims=dims)
        super().__init__(dims=dims, encoder=encoder, name=name)
        
        self.verifier = CodeVerifier()
        self._last_pattern_name: Optional[str] = None
        self._last_mapping: Optional[Mapping] = None
        
        # Bootstrap with default patterns
        self._bootstrap_default_patterns()
    
    def _bootstrap_default_patterns(self) -> None:
        """
        Bootstrap with essential code patterns.
        
        These are the ONLY hardcoded patterns - they bootstrap the space.
        Additional patterns can be loaded from corpus files.
        """
        # Learn from pattern descriptions to build word positions
        descriptions = [
            "print hello world",
            "print a message",
            "print a variable",
            "create a variable",
            "create a function",
            "read a file",
            "write to a file",
            "loop over a list",
            "if condition check",
            "import a module",
            "define a class",
            "calculate sum",
            "plot a chart",
            "create a list",
            "create a dictionary",
        ]
        self.encoder.learn(descriptions)
        
        # Bootstrap essential patterns
        patterns = [
            # Output patterns
            ("print hello world", 'print("Hello, World!")', "hello_world"),
            ("print a message", 'print("{message}")', "print_string"),
            ("print a variable", 'print({variable})', "print_variable"),
            ("print formatted string", 'print(f"{format_string}")', "print_formatted"),
            
            # Variable patterns
            ("create a string variable", '{name} = "{value}"', "create_string"),
            ("create a number variable", '{name} = {value}', "create_number"),
            ("create a list", '{name} = [{items}]', "create_list"),
            ("create a dictionary", '{name} = {{{items}}}', "create_dict"),
            
            # Function patterns
            ("create a function", '''def {name}({params}):
    {body}
    return {result}''', "create_function"),
            
            # File patterns
            ("read a file", '''with open("{filename}", "r") as f:
    content = f.read()''', "read_file"),
            ("write to a file", '''with open("{filename}", "w") as f:
    f.write({content})''', "write_file"),
            
            # Control flow patterns
            ("loop over a list", '''for item in {items}:
    {body}''', "for_loop"),
            ("if condition check", '''if {condition}:
    {body}''', "if_statement"),
            
            # Import patterns
            ("import a module", 'import {module}', "import_module"),
            ("import from module", 'from {module} import {names}', "import_from"),
            
            # Class patterns
            ("define a class", '''class {name}:
    def __init__(self{params}):
        {init_body}''', "define_class"),
            
            # Math patterns
            ("calculate sum", 'total = sum({numbers})', "sum_numbers"),
            ("calculate average", 'average = sum({numbers}) / len({numbers})', "average"),
            
            # Visualization patterns
            ("plot a line chart", '''import matplotlib.pyplot as plt
plt.plot({x_data}, {y_data})
plt.xlabel("{x_label}")
plt.ylabel("{y_label}")
plt.title("{title}")
plt.show()''', "plot_line"),
            ("plot a bar chart", '''import matplotlib.pyplot as plt
plt.bar({categories}, {values})
plt.xlabel("{x_label}")
plt.ylabel("{y_label}")
plt.title("{title}")
plt.show()''', "plot_bar"),
        ]
        
        for description, template, name in patterns:
            self.add_pattern(description, template, name)
    
    def add_pattern(self, description: str, template: str, name: str,
                    examples: Optional[List[str]] = None) -> Mapping:
        """
        Add a code pattern to the space.
        
        Args:
            description: Natural language description of what the pattern does
            template: Code template with {placeholders}
            name: Unique name for the pattern
            examples: Optional list of example queries that match this pattern
            
        Returns:
            The created Mapping
        """
        mapping = self.map(description, template, metadata={
            'name': name,
            'type': 'code_pattern',
            'examples': examples or [],
        })
        
        # Also bootstrap the pattern name for exact lookup
        self.bootstrap(name, template)
        
        return mapping
    
    def bootstrap_from_corpus(self, corpus_path: str) -> int:
        """
        Bootstrap patterns from a corpus JSON file.
        
        The corpus file should have the format:
        {
            "patterns": [
                {
                    "name": "pattern_name",
                    "template": "code template",
                    "description": "what it does",
                    "examples": ["example 1", "example 2"]
                },
                ...
            ]
        }
        
        Returns:
            Number of patterns loaded
        """
        with open(corpus_path, 'r') as f:
            data = json.load(f)
        
        count = 0
        for p in data.get('patterns', []):
            name = p.get('name', f'pattern_{count}')
            template = p.get('template', '')
            description = p.get('description', name)
            examples = p.get('examples', [])
            
            if template:
                self.add_pattern(description, template, name, examples)
                count += 1
        
        # Reproject after loading all patterns
        if count > 0:
            self.reproject()
        
        return count
    
    def generate(self, query: str, params: Optional[Dict[str, Any]] = None,
                 verify: bool = True) -> CodeResult:
        """
        Generate code from a natural language query.
        
        This is GEOMETRIC - uses position-based matching to find
        the best pattern, then fills in parameters.
        
        Args:
            query: Natural language description of desired code
            params: Optional parameters to fill into template
            verify: Whether to verify the generated code
            
        Returns:
            CodeResult with generated code and metadata
        """
        params = params or {}
        
        # Find best matching pattern (geometric)
        result = self.forward(query)
        
        if result is None:
            return CodeResult(
                success=False,
                code="",
                error="No matching pattern found"
            )
        
        # Get pattern metadata and store reference for feedback
        pattern_name = result.mapping.metadata.get('name', 'unknown')
        self._last_pattern_name = pattern_name
        self._last_mapping = result.mapping
        
        # Get template and fill parameters
        template = result.output
        code = self._fill_template(template, params)
        
        # Verify if requested
        verified = False
        output = None
        error = None
        
        if verify:
            # Check syntax first
            syntax_ok, syntax_error = self.verifier.check_syntax(code)
            if not syntax_ok:
                return CodeResult(
                    success=True,  # Generation succeeded, verification failed
                    code=code,
                    pattern_name=pattern_name,
                    error=syntax_error,
                    verified=False
                )
            
            # Try to run the code
            run_ok, run_output = self.verifier.run_code(code)
            verified = run_ok
            if run_ok:
                output = run_output
            else:
                error = run_output
        
        return CodeResult(
            success=True,
            code=code,
            pattern_name=pattern_name,
            verified=verified,
            output=output,
            error=error
        )
    
    def _fill_template(self, template: str, params: Dict[str, Any]) -> str:
        """Fill template placeholders with parameters."""
        code = template
        for key, value in params.items():
            placeholder = '{' + key + '}'
            if placeholder in code:
                code = code.replace(placeholder, str(value))
        return code
    
    def feedback(self, pattern_name: Optional[str] = None, 
                 success: bool = True) -> bool:
        """
        Provide feedback on code generation.
        
        This is THE learning operation - reinforces the pattern
        that was used based on success/failure.
        
        Args:
            pattern_name: Name of pattern to reinforce (uses last if None)
            success: Whether the generated code was successful
            
        Returns:
            True if feedback was recorded
        """
        # Use stored mapping reference if available
        if self._last_mapping is not None:
            self.reinforce(self._last_mapping, success)
            return True
        
        # Fall back to name lookup
        name = pattern_name or self._last_pattern_name
        if not name:
            return False
        
        # Find the mapping for this pattern
        for mapping in self._mappings:
            if mapping.metadata.get('name') == name:
                self.reinforce(mapping, success)
                return True
        
        return False
    
    def get_pattern(self, name: str) -> Optional[str]:
        """Get a pattern template by name."""
        return self.compose(name)
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available patterns."""
        patterns = []
        for mapping in self._mappings:
            if mapping.metadata.get('type') == 'code_pattern':
                patterns.append({
                    'name': mapping.metadata.get('name'),
                    'description': mapping.input,
                    'use_count': mapping.use_count,
                    'success_rate': mapping.success_rate,
                    'persists': mapping.persists,
                })
        return patterns
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()
        data['type'] = 'CodeSpace'
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSpace':
        """Deserialize from dictionary."""
        space = cls(
            dims=data.get('dims', 8),
            name=data.get('name', 'code')
        )
        
        # Clear default patterns
        space._mappings = []
        space._templates = {}
        
        # Load mappings
        for m_data in data.get('mappings', []):
            mapping = Mapping.from_dict(m_data)
            space._mappings.append(mapping)
            
            # Restore templates
            name = mapping.metadata.get('name')
            if name:
                space._templates[name] = mapping.output
        
        # Rebuild indices
        space._rebuild_indices()
        
        return space
    
    @classmethod
    def load(cls, path: str) -> 'CodeSpace':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        pattern_count = sum(1 for m in self._mappings if m.metadata.get('type') == 'code_pattern')
        return f"CodeSpace(name='{self.name}', patterns={pattern_count})"
