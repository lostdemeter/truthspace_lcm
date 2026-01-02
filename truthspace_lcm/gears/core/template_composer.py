"""
Template Composer

A composition layer that can modify templates based on natural language requests.
This is the missing piece in the Emergent Gear Pattern - the ability to not just
match a template, but adapt it to the specific request.

The composer:
1. Parses modification intent from natural language
2. Locates modification points in templates
3. Applies modifications safely
4. Learns from successful modifications

Author: Lesley Gushurst
License: GPLv3
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto


class ModificationType(Enum):
    """Types of modifications that can be applied to templates."""
    SET = auto()      # Set a value: "amplitude of 2"
    ADD = auto()      # Add to a value: "shifted up by 0.5", "x + 0.5"
    MULTIPLY = auto() # Multiply a value: "doubled", "scaled by 2"
    REPLACE = auto()  # Replace something: "in red instead of blue"
    APPEND = auto()   # Add something new: "with a legend"
    REMOVE = auto()   # Remove something: "without grid"


@dataclass
class Modification:
    """A single modification to apply to a template."""
    mod_type: ModificationType
    target: str  # What to modify: 'y', 'amplitude', 'color', 'title'
    value: Any   # The new/delta value
    raw_text: str = ""  # Original text that triggered this
    confidence: float = 1.0


@dataclass
class ModificationPoint:
    """A point in a template that can be modified."""
    name: str           # Semantic name: 'y_value', 'amplitude', 'color'
    location: str       # Where in the code: 'create_data.return', 'create_plot.plot.color'
    current_value: Any  # Current value if extractable
    value_type: str     # 'numeric', 'string', 'expression'
    line_number: int = 0
    column: int = 0


@dataclass 
class CompositionResult:
    """Result of composing a template with modifications."""
    success: bool
    code: str
    modifications_applied: List[Modification] = field(default_factory=list)
    modifications_failed: List[Tuple[Modification, str]] = field(default_factory=list)
    used_llm: bool = False


class ModificationParser:
    """
    Parse natural language to extract modification intent.
    
    Patterns recognized:
    - "but with X" → modification follows
    - "shifted by N" / "offset by N" → ADD to position
    - "amplitude of N" / "frequency of N" → SET parameter
    - "in COLOR" / "colored COLOR" → SET color
    - "x + N" / "y + N" → ADD to result
    - "scaled by N" / "multiplied by N" → MULTIPLY
    - "without X" → REMOVE
    - "with X" → APPEND (if not a parameter)
    """
    
    # Patterns for detecting modifications
    PATTERNS = [
        # Offset patterns: "shifted by 0.5", "offset by 2", "moved up by 1"
        (r'(?:shifted?|offset|moved?\s*(?:up|down)?)\s*(?:by\s*)?([+-]?\d*\.?\d+)', 
         ModificationType.ADD, 'y_offset'),
        
        # X offset: "x offset of 0.5", "x shifted by 1"
        (r'x\s*(?:offset|shifted?)\s*(?:by|of)?\s*([+-]?\d*\.?\d+)',
         ModificationType.ADD, 'x_offset'),
        
        # Y offset: "y offset of 0.5", "y shifted by 1"  
        (r'y\s*(?:offset|shifted?)\s*(?:by|of)?\s*([+-]?\d*\.?\d+)',
         ModificationType.ADD, 'y_offset'),
        
        # Result modification: "results being x+0.5", "output is y+1"
        (r'(?:results?|output)\s*(?:being|is|=)\s*[xy]\s*([+-])\s*(\d*\.?\d+)',
         ModificationType.ADD, 'y_offset'),
        
        # Direct addition: "x + 0.5", "y + 1", "+ 0.5"
        (r'(?:^|\s)[xy]?\s*\+\s*(\d*\.?\d+)',
         ModificationType.ADD, 'y_offset'),
        
        # Amplitude: "amplitude of 2", "amplitude 2"
        (r'amplitude\s*(?:of|=|:)?\s*(\d*\.?\d+)',
         ModificationType.SET, 'amplitude'),
        
        # Frequency: "frequency of 2", "frequency 2"
        (r'frequency\s*(?:of|=|:)?\s*(\d*\.?\d+)',
         ModificationType.SET, 'frequency'),
        
        # Scaling: "scaled by 2", "multiplied by 2", "doubled"
        (r'(?:scaled?|multiplied?)\s*(?:by\s*)?(\d*\.?\d+)',
         ModificationType.MULTIPLY, 'amplitude'),
        (r'\bdoubled?\b', ModificationType.MULTIPLY, 'amplitude'),  # Special case
        
        # Color: "in red", "colored blue", "color red"
        (r'(?:in|colored?|colour)\s*(red|blue|green|yellow|orange|purple|black|white|cyan|magenta)',
         ModificationType.SET, 'color'),
        
        # Title: "titled X", "with title X"
        (r'(?:titled?|with\s+title)\s*["\']?([^"\']+)["\']?',
         ModificationType.SET, 'title'),
        
        # Remove: "without grid", "no legend"
        (r'(?:without|no)\s+(grid|legend|axis|labels?|title)',
         ModificationType.REMOVE, None),  # Target extracted from match
        
        # With/add: "with legend", "add grid"
        (r'(?:with|add)\s+(legend|grid|markers?)',
         ModificationType.APPEND, None),
    ]
    
    def parse(self, request: str, base_matched: bool = True) -> List[Modification]:
        """
        Parse a request to extract modifications.
        
        Args:
            request: The full user request
            base_matched: Whether we already matched a base template
            
        Returns:
            List of modifications to apply
        """
        modifications = []
        seen_targets = set()  # Avoid duplicate modifications to same target
        request_lower = request.lower()
        
        # Look for "but" or "with" as modification indicators
        mod_section = request_lower
        if ' but ' in request_lower:
            mod_section = request_lower.split(' but ', 1)[1]
        elif ', with ' in request_lower:
            mod_section = request_lower.split(', with ', 1)[1]
        
        for pattern, mod_type, default_target in self.PATTERNS:
            matches = re.finditer(pattern, mod_section, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Extract value
                if mod_type == ModificationType.MULTIPLY and 'double' in match.group(0).lower():
                    value = 2.0
                elif groups:
                    # Handle sign + value pattern
                    if len(groups) == 2 and groups[0] in '+-':
                        sign = -1 if groups[0] == '-' else 1
                        value = sign * float(groups[1])
                    else:
                        try:
                            value = float(groups[0])
                        except (ValueError, TypeError):
                            value = groups[0]  # String value (color, etc.)
                else:
                    value = None
                
                # Determine target
                target = default_target
                if target is None and groups:
                    target = groups[-1]  # Use last group as target
                
                if target and value is not None:
                    # Skip if we already have a modification for this target
                    target_key = (target, mod_type)
                    if target_key in seen_targets:
                        continue
                    seen_targets.add(target_key)
                    
                    modifications.append(Modification(
                        mod_type=mod_type,
                        target=target,
                        value=value,
                        raw_text=match.group(0),
                    ))
        
        return modifications


class TemplateAnnotator:
    """
    Annotate templates to identify modification points.
    
    For Python code, we parse the AST and identify:
    - Function parameters with defaults
    - Variable assignments
    - Function call arguments
    """
    
    # Known modification points for common template types
    KNOWN_POINTS = {
        'sine_wave': {
            'x_offset': ('create_data', 'x_offset', 'numeric'),
            'y_offset': ('create_data', 'y_offset', 'numeric'),
            'amplitude': ('create_data', 'amplitude', 'numeric'),
            'frequency': ('create_data', 'frequency', 'numeric'),
            'color': ('create_plot', 'plot_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
        'cosine_wave': {
            'x_offset': ('create_data', 'x_offset', 'numeric'),
            'y_offset': ('create_data', 'y_offset', 'numeric'),
            'amplitude': ('create_data', 'amplitude', 'numeric'),
            'frequency': ('create_data', 'frequency', 'numeric'),
            'color': ('create_plot', 'plot_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
        'bar_chart': {
            'color': ('create_plot', 'bar_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
        'histogram': {
            'bins': ('create_plot', 'bins', 'numeric'),
            'color': ('create_plot', 'hist_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
        'scatter_plot': {
            'color': ('create_plot', 'scatter_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
        'line_plot': {
            'color': ('create_plot', 'line_color', 'string'),
            'title': ('create_plot', 'title', 'string'),
        },
    }
    
    def get_modification_points(self, template_name: str, code: str) -> Dict[str, ModificationPoint]:
        """Get known modification points for a template."""
        points = {}
        
        if template_name in self.KNOWN_POINTS:
            for name, (func, loc, vtype) in self.KNOWN_POINTS[template_name].items():
                points[name] = ModificationPoint(
                    name=name,
                    location=f"{func}.{loc}",
                    current_value=None,
                    value_type=vtype,
                )
        
        return points


class TemplateModifier:
    """
    Apply modifications to templates.
    
    Uses a combination of:
    - Regex-based replacements for simple cases
    - AST manipulation for complex cases
    """
    
    def apply(self, code: str, template_name: str, 
              modifications: List[Modification]) -> Tuple[str, List[Modification], List[Tuple[Modification, str]]]:
        """
        Apply modifications to code.
        
        Returns: (modified_code, applied_mods, failed_mods_with_reasons)
        """
        applied = []
        failed = []
        
        for mod in modifications:
            try:
                new_code = self._apply_single(code, template_name, mod)
                if new_code != code:
                    code = new_code
                    applied.append(mod)
                else:
                    failed.append((mod, "No change made"))
            except Exception as e:
                failed.append((mod, str(e)))
        
        return code, applied, failed
    
    def _apply_single(self, code: str, template_name: str, mod: Modification) -> str:
        """Apply a single modification."""
        
        if mod.target in ('x_offset', 'y_offset', 'amplitude', 'frequency'):
            return self._apply_numeric_param(code, mod)
        elif mod.target == 'color':
            return self._apply_color(code, mod)
        elif mod.target == 'title':
            return self._apply_title(code, mod)
        elif mod.target in ('grid', 'legend', 'markers'):
            return self._apply_feature(code, mod)
        
        return code
    
    def _apply_numeric_param(self, code: str, mod: Modification) -> str:
        """Apply numeric parameter modification."""
        param = mod.target
        value = mod.value
        
        if mod.mod_type == ModificationType.SET:
            # Replace default value in function signature
            pattern = rf'({param}\s*=\s*)([+-]?\d*\.?\d+)'
            replacement = rf'\g<1>{value}'
            code = re.sub(pattern, replacement, code)
            
            # Also update the call site
            pattern = rf'(create_data\([^)]*{param}\s*=\s*)([+-]?\d*\.?\d+)'
            replacement = rf'\g<1>{value}'
            code = re.sub(pattern, replacement, code)
            
        elif mod.mod_type == ModificationType.ADD:
            # For y_offset, we need to modify the function call
            if param == 'y_offset':
                # Update the default in function signature
                pattern = r'(y_offset\s*=\s*)([+-]?\d*\.?\d+)'
                match = re.search(pattern, code)
                if match:
                    current = float(match.group(2))
                    new_val = current + value
                    code = re.sub(pattern, rf'\g<1>{new_val}', code)
                else:
                    # No y_offset param, need to add to the y calculation
                    # Find "y = ... + y_offset" and add our value
                    pattern = r'(y\s*=\s*[^+\n]+\s*\+\s*y_offset)'
                    if re.search(pattern, code):
                        code = re.sub(pattern, rf'\g<1> + {value}', code)
                    else:
                        # Find "return x, y" and modify y
                        pattern = r'(return\s+x\s*,\s*)y(\s*$)'
                        code = re.sub(pattern, rf'\g<1>y + {value}\g<2>', code, flags=re.MULTILINE)
                        
            elif param == 'x_offset':
                pattern = r'(x_offset\s*=\s*)([+-]?\d*\.?\d+)'
                match = re.search(pattern, code)
                if match:
                    current = float(match.group(2))
                    new_val = current + value
                    code = re.sub(pattern, rf'\g<1>{new_val}', code)
                    
        elif mod.mod_type == ModificationType.MULTIPLY:
            if param == 'amplitude':
                pattern = r'(amplitude\s*=\s*)([+-]?\d*\.?\d+)'
                match = re.search(pattern, code)
                if match:
                    current = float(match.group(2))
                    new_val = current * value
                    code = re.sub(pattern, rf'\g<1>{new_val}', code)
        
        return code
    
    def _apply_color(self, code: str, mod: Modification) -> str:
        """Apply color modification to the main plot line only."""
        color = mod.value
        
        # Map color names to matplotlib codes
        color_map = {
            'red': 'r', 'blue': 'b', 'green': 'g', 'yellow': 'y',
            'black': 'k', 'white': 'w', 'cyan': 'c', 'magenta': 'm',
            'orange': 'orange', 'purple': 'purple',
        }
        color_code = color_map.get(color.lower(), color)
        
        # Only replace in plt.plot() calls - NOT in axhline/axvline/scatter/bar
        # Match plt.plot with format string like 'b-' or 'b-o'
        code = re.sub(r"(plt\.plot\([^)]*)'[brgcmykw]-'", rf"\g<1>'{color_code}-'", code)
        code = re.sub(r"(plt\.plot\([^)]*)'[brgcmykw]-o'", rf"\g<1>'{color_code}-o'", code)
        
        # Replace color= in plt.scatter() calls
        code = re.sub(r"(plt\.scatter\([^)]*c\s*=\s*)['\"][^'\"]+['\"]", rf"\g<1>'{color_code}'", code)
        
        # Replace color= in plt.bar() calls  
        code = re.sub(r"(plt\.bar\([^)]*color\s*=\s*)['\"][^'\"]+['\"]", rf"\g<1>'{color_code}'", code)
        
        # Replace color= in plt.hist() calls
        code = re.sub(r"(plt\.hist\([^)]*color\s*=\s*)['\"][^'\"]+['\"]", rf"\g<1>'{color_code}'", code)
        
        return code
    
    def _apply_title(self, code: str, mod: Modification) -> str:
        """Apply title modification."""
        title = mod.value
        
        # Replace title in create_plot call or plt.title()
        code = re.sub(r'(title\s*=\s*)["\'][^"\']*["\']', rf'\g<1>"{title}"', code)
        code = re.sub(r'(plt\.title\s*\(\s*)["\'][^"\']*["\']', rf'\g<1>"{title}"', code)
        
        return code
    
    def _apply_feature(self, code: str, mod: Modification) -> str:
        """Apply feature add/remove modification."""
        feature = mod.target
        
        if mod.mod_type == ModificationType.REMOVE:
            if feature == 'grid':
                code = re.sub(r'\n\s*plt\.grid\([^)]*\)', '', code)
            elif feature == 'legend':
                code = re.sub(r'\n\s*plt\.legend\([^)]*\)', '', code)
                
        elif mod.mod_type == ModificationType.APPEND:
            # Find the create_plot function and add before the closing
            if feature == 'legend':
                # Add legend before plt.savefig or at end of create_plot
                code = re.sub(
                    r'(plt\.savefig)',
                    r'plt.legend()\n    \g<1>',
                    code
                )
            elif feature == 'grid' and 'plt.grid' not in code:
                code = re.sub(
                    r'(plt\.savefig)',
                    r'plt.grid(True)\n    \g<1>',
                    code
                )
        
        return code


class TemplateComposer:
    """
    Main composition layer that combines parsing, annotation, and modification.
    
    Usage:
        composer = TemplateComposer()
        result = composer.compose(
            request="create a sine wave plot, but with the results being x+0.5",
            template_name="sine_wave",
            template_code=sine_wave_template
        )
    """
    
    def __init__(self):
        self.parser = ModificationParser()
        self.annotator = TemplateAnnotator()
        self.modifier = TemplateModifier()
        
        # Track learned modifications for future use
        self.learned_modifications: Dict[str, List[Modification]] = {}
    
    def compose(self, request: str, template_name: str, 
                template_code: str) -> CompositionResult:
        """
        Compose a template with modifications from the request.
        
        Args:
            request: The user's full request
            template_name: Name of the matched template
            template_code: The template code to modify
            
        Returns:
            CompositionResult with the modified code
        """
        # Parse modifications from request
        modifications = self.parser.parse(request)
        
        if not modifications:
            # No modifications detected, return original
            return CompositionResult(
                success=True,
                code=template_code,
            )
        
        # Apply modifications
        modified_code, applied, failed = self.modifier.apply(
            template_code, template_name, modifications
        )
        
        return CompositionResult(
            success=len(applied) > 0 or len(failed) == 0,
            code=modified_code,
            modifications_applied=applied,
            modifications_failed=failed,
        )
    
    def learn_modification(self, request: str, modification: Modification):
        """Learn a new modification pattern from successful usage."""
        key = modification.raw_text.lower().strip()
        if key not in self.learned_modifications:
            self.learned_modifications[key] = []
        self.learned_modifications[key].append(modification)
