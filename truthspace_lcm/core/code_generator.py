#!/usr/bin/env python3
"""
Code Generator: Teaching Python to GeometricLCM

This module extends the concept frame approach to code generation.
Code is structured language - even MORE structured than natural language.

Key insight: Code has clear semantics that map to concept frames:
- Functions = AGENTS that TRANSFORM inputs to outputs
- Variables = PATIENTS that hold values
- Control flow = CONTROL primitives (if, for, while)
- Operations = ACTION primitives (add, compare, iterate)

Author: Lesley Gushurst
License: GPLv3
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


# =============================================================================
# CODE PRIMITIVES (like action primitives for natural language)
# =============================================================================

CODE_PRIMITIVES = {
    'DEFINE': {'def', 'class', 'lambda'},
    'ASSIGN': {'=', '+=', '-=', '*=', '/='},
    'INVOKE': {'call', 'invoke', 'run', 'execute'},
    'CONTROL': {'if', 'for', 'while', 'try', 'with'},
    'RETURN': {'return', 'yield'},
    'IMPORT': {'import', 'from'},
    'ACCESS': {'get', 'set', 'index', 'attribute'},
}

# Common operations and their implementations
OPERATIONS = {
    # Math operations
    'add': {'args': ['a', 'b'], 'body': 'return a + b', 'doc': 'Add two numbers'},
    'subtract': {'args': ['a', 'b'], 'body': 'return a - b', 'doc': 'Subtract b from a'},
    'multiply': {'args': ['a', 'b'], 'body': 'return a * b', 'doc': 'Multiply two numbers'},
    'divide': {'args': ['a', 'b'], 'body': 'return a / b if b != 0 else None', 'doc': 'Divide a by b'},
    'sum': {'args': ['numbers'], 'body': 'return sum(numbers)', 'doc': 'Sum a list of numbers'},
    'average': {'args': ['numbers'], 'body': 'return sum(numbers) / len(numbers) if numbers else 0', 'doc': 'Calculate average'},
    'max': {'args': ['numbers'], 'body': 'return max(numbers) if numbers else None', 'doc': 'Find maximum'},
    'min': {'args': ['numbers'], 'body': 'return min(numbers) if numbers else None', 'doc': 'Find minimum'},
    
    # String operations
    'greet': {'args': ['name'], 'body': 'return f"Hello, {name}!"', 'doc': 'Greet someone'},
    'reverse': {'args': ['s'], 'body': 'return s[::-1]', 'doc': 'Reverse a string'},
    'uppercase': {'args': ['s'], 'body': 'return s.upper()', 'doc': 'Convert to uppercase'},
    'lowercase': {'args': ['s'], 'body': 'return s.lower()', 'doc': 'Convert to lowercase'},
    'concat': {'args': ['a', 'b'], 'body': 'return str(a) + str(b)', 'doc': 'Concatenate two values'},
    'split': {'args': ['s', 'sep'], 'body': 'return s.split(sep)', 'doc': 'Split string by separator'},
    'join': {'args': ['items', 'sep'], 'body': 'return sep.join(items)', 'doc': 'Join items with separator'},
    
    # List operations
    'length': {'args': ['items'], 'body': 'return len(items)', 'doc': 'Get length'},
    'first': {'args': ['items'], 'body': 'return items[0] if items else None', 'doc': 'Get first item'},
    'last': {'args': ['items'], 'body': 'return items[-1] if items else None', 'doc': 'Get last item'},
    'append': {'args': ['items', 'item'], 'body': 'items.append(item); return items', 'doc': 'Append to list'},
    'filter': {'args': ['items', 'predicate'], 'body': 'return [x for x in items if predicate(x)]', 'doc': 'Filter items'},
    'map': {'args': ['items', 'func'], 'body': 'return [func(x) for x in items]', 'doc': 'Map function over items'},
    
    # Comparison operations
    'equals': {'args': ['a', 'b'], 'body': 'return a == b', 'doc': 'Check equality'},
    'greater': {'args': ['a', 'b'], 'body': 'return a > b', 'doc': 'Check if a > b'},
    'less': {'args': ['a', 'b'], 'body': 'return a < b', 'doc': 'Check if a < b'},
    'contains': {'args': ['container', 'item'], 'body': 'return item in container', 'doc': 'Check if item in container'},
    
    # Boolean operations
    'is_empty': {'args': ['items'], 'body': 'return len(items) == 0', 'doc': 'Check if empty'},
    'is_even': {'args': ['n'], 'body': 'return n % 2 == 0', 'doc': 'Check if even'},
    'is_odd': {'args': ['n'], 'body': 'return n % 2 != 0', 'doc': 'Check if odd'},
    'is_positive': {'args': ['n'], 'body': 'return n > 0', 'doc': 'Check if positive'},
    'is_negative': {'args': ['n'], 'body': 'return n < 0', 'doc': 'Check if negative'},
}

# Synonyms for operations
OPERATION_SYNONYMS = {
    # Math
    'plus': 'add', 'addition': 'add', 'sum_two': 'add', 'adds': 'add', 'adding': 'add',
    'minus': 'subtract', 'subtraction': 'subtract', 'difference': 'subtract', 'subtracts': 'subtract',
    'times': 'multiply', 'multiplication': 'multiply', 'product': 'multiply', 
    'multiplies': 'multiply', 'multiplying': 'multiply',
    'divided': 'divide', 'division': 'divide', 'quotient': 'divide', 'divides': 'divide',
    # String
    'hello': 'greet', 'welcome': 'greet', 'say_hello': 'greet', 'greets': 'greet',
    'flip': 'reverse', 'backwards': 'reverse', 'reverses': 'reverse', 'reversing': 'reverse',
    'upper': 'uppercase', 'caps': 'uppercase',
    'lower': 'lowercase',
    'concatenate': 'concat', 'concatenates': 'concat', 'combine': 'concat',
    'splits': 'split', 'splitting': 'split',
    'joins': 'join', 'joining': 'join',
    # List
    'len': 'length', 'size': 'length', 'count': 'length',
    'head': 'first', 'tail': 'last',
    'appends': 'append', 'appending': 'append',
    'filters': 'filter', 'filtering': 'filter',
    'maps': 'map', 'mapping': 'map',
    # Comparison
    'same': 'equals', 'equal': 'equals', 'compare': 'equals',
    'bigger': 'greater', 'larger': 'greater', 'more': 'greater',
    'smaller': 'less', 'fewer': 'less',
    'has': 'contains', 'includes': 'contains', 'in': 'contains',
    # Boolean
    'empty': 'is_empty', 'blank': 'is_empty',
    'even': 'is_even', 'odd': 'is_odd',
    'positive': 'is_positive', 'negative': 'is_negative',
}


# =============================================================================
# CODE FRAME (concept frame for code)
# =============================================================================

@dataclass
class CodeFrame:
    """Concept frame for code."""
    primitive: str = ''  # DEFINE, ASSIGN, INVOKE, CONTROL, etc.
    name: str = ''  # Function/variable name
    args: List[str] = field(default_factory=list)  # Arguments
    body: str = ''  # Body/implementation
    doc: str = ''  # Documentation
    return_type: str = ''  # Return type hint
    
    def to_python(self) -> str:
        """Generate Python code from frame."""
        if self.primitive == 'DEFINE':
            args_str = ', '.join(self.args)
            lines = [f'def {self.name}({args_str}):']
            if self.doc:
                lines.append(f'    """{self.doc}"""')
            if self.body:
                for line in self.body.split(';'):
                    line = line.strip()
                    if line:
                        lines.append(f'    {line}')
            else:
                lines.append('    pass')
            return '\n'.join(lines)
        
        elif self.primitive == 'ASSIGN':
            return f'{self.name} = {self.body}'
        
        elif self.primitive == 'INVOKE':
            args_str = ', '.join(self.args)
            return f'{self.name}({args_str})'
        
        return ''


# =============================================================================
# CODE GENERATOR
# =============================================================================

class CodeGenerator:
    """
    Generate Python code from natural language requests.
    
    Uses the same geometric principles as the rest of GeometricLCM:
    - Parse request into concept frame
    - Match to code templates
    - Generate code from frame
    """
    
    def __init__(self):
        """Initialize the code generator."""
        self.operations = OPERATIONS.copy()
        self.synonyms = OPERATION_SYNONYMS.copy()
        self.learned_functions: Dict[str, CodeFrame] = {}
    
    def parse_request(self, request: str) -> CodeFrame:
        """
        Parse a natural language request into a code frame.
        
        Examples:
            "Write a function to add two numbers"
            "Create a function that greets someone"
            "Make a function to reverse a string"
        """
        request_lower = request.lower()
        
        # Detect primitive
        primitive = 'DEFINE'  # Default to function definition
        if any(word in request_lower for word in ['call', 'run', 'execute', 'invoke']):
            primitive = 'INVOKE'
        elif any(word in request_lower for word in ['set', 'assign', 'store']):
            primitive = 'ASSIGN'
        
        # Extract operation name
        operation = self._extract_operation(request_lower)
        
        # Get operation details
        if operation in self.operations:
            op = self.operations[operation]
            return CodeFrame(
                primitive=primitive,
                name=operation,
                args=op['args'].copy(),
                body=op['body'],
                doc=op['doc'],
            )
        
        # Check learned functions
        if operation in self.learned_functions:
            return self.learned_functions[operation]
        
        # Default: create empty function
        name = self._extract_name(request_lower) or 'my_function'
        args = self._extract_args(request_lower)
        
        return CodeFrame(
            primitive=primitive,
            name=name,
            args=args,
            body='pass  # TODO: implement',
            doc=f'Generated from: {request}',
        )
    
    def _extract_operation(self, request: str) -> str:
        """Extract operation from request."""
        # Check for known operations
        for op in self.operations:
            if op in request:
                return op
        
        # Check synonyms
        for synonym, op in self.synonyms.items():
            if synonym in request:
                return op
        
        # Try to extract from common patterns
        patterns = [
            r'to (\w+)',  # "to add", "to reverse"
            r'that (\w+)s',  # "that adds", "that reverses"
            r'which (\w+)s',  # "which adds"
            r'for (\w+)ing',  # "for adding"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, request)
            if match:
                word = match.group(1)
                # Check if it's a known operation or synonym
                if word in self.operations:
                    return word
                if word in self.synonyms:
                    return self.synonyms[word]
                # Check base form
                if word.endswith('s'):
                    base = word[:-1]
                    if base in self.operations:
                        return base
                    if base in self.synonyms:
                        return self.synonyms[base]
        
        return ''
    
    def _extract_name(self, request: str) -> str:
        """Extract function name from request."""
        # Look for "called X" or "named X"
        match = re.search(r'(?:called|named)\s+(\w+)', request)
        if match:
            return match.group(1)
        
        # Look for operation words
        for word in request.split():
            if word in self.operations or word in self.synonyms:
                return word
        
        return ''
    
    def _extract_args(self, request: str) -> List[str]:
        """Extract argument names from request."""
        args = []
        
        # Look for "takes X and Y" or "with X and Y"
        match = re.search(r'(?:takes|with|given)\s+(\w+)(?:\s+and\s+(\w+))?', request)
        if match:
            args.append(match.group(1))
            if match.group(2):
                args.append(match.group(2))
            return args
        
        # Look for "two numbers" → a, b
        if 'two numbers' in request or 'two values' in request:
            return ['a', 'b']
        
        # Look for "a number" → n
        if 'a number' in request or 'number' in request:
            return ['n']
        
        # Look for "a string" → s
        if 'a string' in request or 'string' in request:
            return ['s']
        
        # Look for "a list" → items
        if 'a list' in request or 'list' in request:
            return ['items']
        
        # Look for "a name" → name
        if 'a name' in request or 'name' in request:
            return ['name']
        
        return ['x']  # Default
    
    def generate(self, request: str) -> str:
        """
        Generate Python code from a natural language request.
        
        Args:
            request: Natural language description
        
        Returns:
            Python code as string
        """
        frame = self.parse_request(request)
        return frame.to_python()
    
    def learn(self, name: str, args: List[str], body: str, doc: str = ''):
        """
        Learn a new function pattern.
        
        Args:
            name: Function name
            args: Argument names
            body: Function body
            doc: Documentation
        """
        self.learned_functions[name] = CodeFrame(
            primitive='DEFINE',
            name=name,
            args=args,
            body=body,
            doc=doc,
        )
        
        # Also add to operations for future matching
        self.operations[name] = {
            'args': args,
            'body': body,
            'doc': doc,
        }
    
    def list_operations(self) -> List[str]:
        """List all known operations."""
        return sorted(self.operations.keys())
    
    def explain(self, operation: str) -> str:
        """Explain what an operation does."""
        if operation in self.synonyms:
            operation = self.synonyms[operation]
        
        if operation in self.operations:
            op = self.operations[operation]
            args_str = ', '.join(op['args'])
            return f"{operation}({args_str}): {op['doc']}"
        
        return f"Unknown operation: {operation}"


# =============================================================================
# TEST
# =============================================================================

def test_code_generator():
    """Test the code generator."""
    print("=== CodeGenerator Test ===\n")
    
    gen = CodeGenerator()
    
    # Test various requests
    requests = [
        "Write a function to add two numbers",
        "Create a function that greets someone",
        "Make a function to reverse a string",
        "Write a function to check if a number is even",
        "Create a function to find the maximum in a list",
        "Write a function to calculate the average",
    ]
    
    for request in requests:
        print(f"Request: {request}")
        code = gen.generate(request)
        print(f"Generated:\n{code}")
        print()
    
    # Test learning
    print("--- Learning new function ---")
    gen.learn(
        name='factorial',
        args=['n'],
        body='return 1 if n <= 1 else n * factorial(n - 1)',
        doc='Calculate factorial of n'
    )
    
    code = gen.generate("Write a function to calculate factorial")
    print(f"Request: Write a function to calculate factorial")
    print(f"Generated:\n{code}")
    print()
    
    # List operations
    print(f"Known operations: {len(gen.list_operations())}")
    print(f"Sample: {gen.list_operations()[:10]}")


if __name__ == '__main__':
    test_code_generator()
