"""
Calculator Tool - Perform mathematical calculations.

Safely evaluates mathematical expressions and performs calculations.
"""

import re
import math
import operator
from .base import Tool, ToolResult


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations."""
    
    def __init__(self):
        # Safe operators
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '^': operator.pow,
        }
        
        # Safe math functions
        self.functions = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'abs': abs,
            'round': round,
            'floor': math.floor,
            'ceil': math.ceil,
            'pow': pow,
            'sum': sum,
            'min': min,
            'max': max,
            'avg': lambda x: sum(x) / len(x) if x else 0,
            'mean': lambda x: sum(x) / len(x) if x else 0,
        }
        
        # Constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
            'tau': math.tau,
        }
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations, including basic arithmetic and functions like sqrt, sin, cos, log"
    
    @property
    def triggers(self) -> list[str]:
        return [
            "calculate",
            "compute",
            "what is",
            "how much is",
            "evaluate",
            "solve",
            "sum of",
            "average of",
            "mean of",
            "add",
            "subtract",
            "multiply",
            "divide",
        ]
    
    def execute(self, query: str, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression."""
        try:
            # Extract the mathematical expression
            expression = self._extract_expression(query)
            
            if not expression:
                return ToolResult.error_result(
                    "I couldn't find a mathematical expression to evaluate. "
                    "Try something like 'calculate 2 + 2' or 'what is sqrt(16)?'"
                )
            
            # Evaluate safely
            result = self._safe_eval(expression)
            
            # Format the result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)
            
            return ToolResult.success_result(
                f"**{expression}** = `{result}`",
                expression=expression,
                result=result,
            )
            
        except ZeroDivisionError:
            return ToolResult.error_result("Cannot divide by zero")
        except ValueError as e:
            return ToolResult.error_result(f"Math error: {str(e)}")
        except Exception as e:
            return ToolResult.error_result(f"Calculation error: {str(e)}")
    
    def _extract_expression(self, query: str) -> str:
        """Extract the mathematical expression from a query."""
        # Remove common prefixes
        prefixes = [
            'calculate', 'compute', 'evaluate', 'solve',
            'what is', 'how much is', 'find',
        ]
        
        text = query.lower()
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Remove trailing question marks and common suffixes
        text = text.rstrip('?').strip()
        
        # Handle "sum of X, Y, Z" pattern
        sum_match = re.match(r'(?:the\s+)?sum\s+of\s+(.+)', text, re.IGNORECASE)
        if sum_match:
            numbers = re.findall(r'-?\d+\.?\d*', sum_match.group(1))
            if numbers:
                return f"sum([{', '.join(numbers)}])"
        
        # Handle "average of X, Y, Z" pattern
        avg_match = re.match(r'(?:the\s+)?(?:average|mean)\s+of\s+(.+)', text, re.IGNORECASE)
        if avg_match:
            numbers = re.findall(r'-?\d+\.?\d*', avg_match.group(1))
            if numbers:
                return f"avg([{', '.join(numbers)}])"
        
        # Clean up the expression
        # Replace 'x' with '*' for multiplication (but not in hex)
        text = re.sub(r'(\d)\s*x\s*(\d)', r'\1 * \2', text)
        
        # Replace word operators
        text = text.replace(' plus ', ' + ')
        text = text.replace(' minus ', ' - ')
        text = text.replace(' times ', ' * ')
        text = text.replace(' divided by ', ' / ')
        text = text.replace(' to the power of ', ' ** ')
        
        return text.strip()
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Replace constants
        for name, value in self.constants.items():
            expression = re.sub(rf'\b{name}\b', str(value), expression, flags=re.IGNORECASE)
        
        # Only allow safe characters
        allowed = set('0123456789.+-*/()[], ')
        allowed.update(set('abcdefghijklmnopqrstuvwxyz_'))
        
        if not all(c in allowed for c in expression.lower()):
            raise ValueError(f"Invalid characters in expression")
        
        # Build safe namespace
        safe_dict = {
            '__builtins__': {},
        }
        safe_dict.update(self.functions)
        safe_dict.update(self.constants)
        
        # Evaluate
        result = eval(expression, safe_dict)
        
        return float(result) if not isinstance(result, (list, tuple)) else result
    
    def matches(self, query: str) -> float:
        """Check if query is asking for a calculation."""
        query_lower = query.lower()
        
        # High confidence for explicit calculation requests
        for trigger in self.triggers:
            if trigger in query_lower:
                # Check if there are numbers in the query
                if re.search(r'\d', query):
                    return 0.95
                return 0.7
        
        # Check for mathematical expressions
        if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', query):
            return 0.9
        
        # Check for math functions
        math_funcs = ['sqrt', 'sin', 'cos', 'tan', 'log', 'exp']
        if any(f in query_lower for f in math_funcs):
            return 0.85
        
        return 0.0
