"""
Code Handler - Code generation and explanation.

Handles requests to generate Python code from natural language.
"""

from .base import Handler, HandlerResult, Context, Intent


class CodeHandler(Handler):
    """Handler for code generation requests."""
    
    def __init__(self, codegen=None):
        """
        Initialize with code generator.
        
        Args:
            codegen: CodeGenerator instance
        """
        self.codegen = codegen
    
    @property
    def name(self) -> str:
        return "code"
    
    @property
    def supported_intents(self) -> list[Intent]:
        return [Intent.CODE]
    
    def can_handle(self, context: Context) -> float:
        """Check if this is a code generation request."""
        if context.intent != Intent.CODE:
            return 0.0
        
        msg_lower = context.message.lower()
        
        # High confidence for explicit code requests
        code_patterns = [
            'write a function', 'create a function', 'generate code',
            'write code', 'python function', 'write a script',
            'code that', 'function to', 'function that',
            'implement', 'write a program'
        ]
        
        if any(p in msg_lower for p in code_patterns):
            return 0.95
        
        # Medium confidence for implicit code requests
        if 'code' in msg_lower or 'function' in msg_lower or 'script' in msg_lower:
            return 0.7
        
        return 0.3
    
    def handle(self, context: Context) -> HandlerResult:
        """Generate code from the request."""
        if not self.codegen:
            return HandlerResult.failure_result("Code generator not initialized")
        
        code = self.codegen.generate(context.message)
        
        if "pass  # TODO" in code:
            # Fallback for unrecognized patterns
            response = f"""I'll help you write that code. Here's my attempt:

```python
{code}
```

This is a template - I recognized the request but don't have a specific implementation for it yet. Would you like me to help you fill in the logic?"""
            return HandlerResult.success_result(response, confidence=0.5, template=True)
        
        response = f"""Here's the code you requested:

```python
{code}
```

Would you like me to explain how it works or make any modifications?"""
        
        return HandlerResult.success_result(response, confidence=0.9, code=code)
