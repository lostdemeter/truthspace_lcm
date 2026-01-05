"""
PlotSpace - Geometric Code Generation for Matplotlib Plots

A HyperMapping-based system for generating matplotlib plots from natural language.
Uses geometric pattern matching with modifiers for customization.

Key features:
- Bootstrap patterns for common plot types (sine, cosine, bar, scatter, etc.)
- Modifier extraction (amplitude, color, frequency, title, etc.)
- Geometric matching for plot type selection
- Code template composition with modifier injection
- Verification and execution

Example:
    space = PlotSpace()
    result = space.generate("create a sine wave plot with amplitude 2.0 and red line")
    # Returns PlotResult with executable matplotlib code

Author: Lesley Gushurst
License: GPLv3
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

from hypermapping import HyperMapping, Mapping, TextEncoder, CRITICAL_LINE


@dataclass
class PlotResult:
    """Result of plot generation."""
    success: bool
    code: str = ""
    plot_type: str = ""
    modifiers: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    verified: bool = False
    output: str = ""
    saved_path: str = ""


@dataclass
class PlotPattern:
    """A plot pattern with template and default values."""
    name: str
    description: str
    keywords: List[str]
    template: str
    defaults: Dict[str, Any]
    examples: List[str]


class PlotSpace(HyperMapping):
    """
    HyperMapping-based plot code generation.
    
    Uses geometric matching to select plot types and modifier extraction
    to customize the generated code.
    """
    
    def __init__(self, name: str = "plot_space", dims: int = 8):
        super().__init__(dims=dims, name=name)
        
        # Text encoder for geometric matching
        self.encoder = TextEncoder(dims=dims)
        
        # Plot patterns (bootstrap)
        self.patterns: Dict[str, PlotPattern] = {}
        
        # Modifier patterns (bootstrap)
        self.modifier_patterns: Dict[str, re.Pattern] = {}
        
        # Last used pattern for feedback
        self._last_pattern: Optional[str] = None
        self._last_mapping: Optional[Mapping] = None
        
        # Bootstrap
        self._bootstrap_patterns()
        self._bootstrap_modifiers()
        self._bootstrap_mappings()
    
    def _bootstrap_patterns(self):
        """Bootstrap plot patterns."""
        self.patterns = {
            'sine_wave': PlotPattern(
                name='sine_wave',
                description='Sine wave plot',
                keywords=['sine', 'sin', 'wave', 'sinusoidal', 'oscillation'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace({x_start}, {x_end}, {num_points})
y = {amplitude} * np.sin({frequency} * x + {phase})

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.plot(x, y, color='{color}', linewidth={linewidth}, linestyle='{linestyle}')
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.grid({grid})
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'x_start': 0,
                    'x_end': '2 * np.pi',
                    'num_points': 100,
                    'amplitude': 1.0,
                    'frequency': 1,
                    'phase': 0,
                    'color': 'blue',
                    'linewidth': 2,
                    'linestyle': '-',
                    'xlabel': 'x',
                    'ylabel': 'y',
                    'title': 'Sine Wave',
                    'grid': True,
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/sine_wave.png',
                },
                examples=[
                    'create a sine wave',
                    'plot a sine wave',
                    'make a sinusoidal plot',
                    'generate sine function',
                ]
            ),
            
            'cosine_wave': PlotPattern(
                name='cosine_wave',
                description='Cosine wave plot',
                keywords=['cosine', 'cos', 'cosinusoidal'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace({x_start}, {x_end}, {num_points})
y = {amplitude} * np.cos({frequency} * x + {phase})

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.plot(x, y, color='{color}', linewidth={linewidth}, linestyle='{linestyle}')
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.grid({grid})
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'x_start': 0,
                    'x_end': '2 * np.pi',
                    'num_points': 100,
                    'amplitude': 1.0,
                    'frequency': 1,
                    'phase': 0,
                    'color': 'blue',
                    'linewidth': 2,
                    'linestyle': '-',
                    'xlabel': 'x',
                    'ylabel': 'y',
                    'title': 'Cosine Wave',
                    'grid': True,
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/cosine_wave.png',
                },
                examples=[
                    'create a cosine wave',
                    'plot a cosine function',
                ]
            ),
            
            'bar_chart': PlotPattern(
                name='bar_chart',
                description='Bar chart',
                keywords=['bar', 'bars', 'histogram', 'column'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = {categories}
values = {values}

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.bar(categories, values, color='{color}', edgecolor='{edgecolor}')
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'categories': "['A', 'B', 'C', 'D', 'E']",
                    'values': '[23, 45, 56, 78, 32]',
                    'color': 'steelblue',
                    'edgecolor': 'black',
                    'xlabel': 'Category',
                    'ylabel': 'Value',
                    'title': 'Bar Chart',
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/bar_chart.png',
                },
                examples=[
                    'create a bar chart',
                    'make a bar graph',
                    'plot bars',
                ]
            ),
            
            'scatter_plot': PlotPattern(
                name='scatter_plot',
                description='Scatter plot',
                keywords=['scatter', 'points', 'dots', 'correlation'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed({seed})
x = np.random.randn({num_points})
y = {correlation} * x + np.random.randn({num_points}) * {noise}

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.scatter(x, y, c='{color}', alpha={alpha}, s={marker_size})
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.grid({grid})
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'seed': 42,
                    'num_points': 100,
                    'correlation': 0.8,
                    'noise': 0.5,
                    'color': 'blue',
                    'alpha': 0.6,
                    'marker_size': 50,
                    'xlabel': 'X',
                    'ylabel': 'Y',
                    'title': 'Scatter Plot',
                    'grid': True,
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/scatter_plot.png',
                },
                examples=[
                    'create a scatter plot',
                    'make a scatter diagram',
                    'plot points',
                ]
            ),
            
            'line_plot': PlotPattern(
                name='line_plot',
                description='Simple line plot',
                keywords=['line', 'linear', 'trend', 'series'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace({x_start}, {x_end}, {num_points})
y = {slope} * x + {intercept}

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.plot(x, y, color='{color}', linewidth={linewidth}, linestyle='{linestyle}')
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.grid({grid})
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'x_start': 0,
                    'x_end': 10,
                    'num_points': 100,
                    'slope': 1,
                    'intercept': 0,
                    'color': 'blue',
                    'linewidth': 2,
                    'linestyle': '-',
                    'xlabel': 'X',
                    'ylabel': 'Y',
                    'title': 'Line Plot',
                    'grid': True,
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/line_plot.png',
                },
                examples=[
                    'create a line plot',
                    'make a linear graph',
                    'plot a line',
                ]
            ),
            
            'pie_chart': PlotPattern(
                name='pie_chart',
                description='Pie chart',
                keywords=['pie', 'circle', 'percentage', 'proportion'],
                template='''import matplotlib.pyplot as plt

# Data
labels = {labels}
sizes = {sizes}
colors = {colors}
explode = {explode}

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow={shadow}, startangle={startangle})
plt.title('{title}')
plt.axis('equal')
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'labels': "['A', 'B', 'C', 'D']",
                    'sizes': '[30, 25, 25, 20]',
                    'colors': "['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']",
                    'explode': '(0, 0.1, 0, 0)',
                    'shadow': True,
                    'startangle': 90,
                    'title': 'Pie Chart',
                    'fig_width': 8,
                    'fig_height': 8,
                    'output_path': 'output/pie_chart.png',
                },
                examples=[
                    'create a pie chart',
                    'make a pie graph',
                    'plot percentages',
                ]
            ),
            
            'histogram': PlotPattern(
                name='histogram',
                description='Histogram',
                keywords=['histogram', 'distribution', 'frequency', 'bins'],
                template='''import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed({seed})
data = np.random.{distribution}({params}, {num_points})

# Create plot
plt.figure(figsize=({fig_width}, {fig_height}))
plt.hist(data, bins={bins}, color='{color}', edgecolor='{edgecolor}', alpha={alpha})
plt.xlabel('{xlabel}')
plt.ylabel('{ylabel}')
plt.title('{title}')
plt.grid({grid})
plt.tight_layout()
plt.savefig('{output_path}')
plt.show()
print("Plot saved to {output_path}")
''',
                defaults={
                    'seed': 42,
                    'distribution': 'normal',
                    'params': '0, 1',
                    'num_points': 1000,
                    'bins': 30,
                    'color': 'steelblue',
                    'edgecolor': 'black',
                    'alpha': 0.7,
                    'xlabel': 'Value',
                    'ylabel': 'Frequency',
                    'title': 'Histogram',
                    'grid': True,
                    'fig_width': 10,
                    'fig_height': 6,
                    'output_path': 'output/histogram.png',
                },
                examples=[
                    'create a histogram',
                    'plot a distribution',
                    'make a frequency chart',
                ]
            ),
        }
    
    def _bootstrap_modifiers(self):
        """Bootstrap modifier extraction patterns."""
        self.modifier_patterns = {
            # Numeric modifiers
            'amplitude': re.compile(r'amplitude\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I),
            'frequency': re.compile(r'frequency\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I),
            'phase': re.compile(r'phase\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I),
            'linewidth': re.compile(r'(?:line\s*)?width\s*(?:of\s*)?(\d+(?:\.\d+)?)', re.I),
            'num_points': re.compile(r'(\d+)\s*points', re.I),
            'bins': re.compile(r'(\d+)\s*bins', re.I),
            
            # Color modifiers
            'color': re.compile(r'\b(red|blue|green|yellow|orange|purple|pink|black|white|cyan|magenta|gray|grey|brown|steelblue|navy|lime|teal|coral|salmon|gold|silver)\b(?:\s+(?:line|color))?', re.I),
            
            # Style modifiers
            'linestyle': re.compile(r'(dashed|dotted|solid|dashdot)\s*(?:line)?', re.I),
            
            # Title
            'title': re.compile(r'(?:titled?|with\s+title)\s*["\']([^"\']+)["\']', re.I),
            
            # Grid
            'grid': re.compile(r'\b(with|without)\s+grid\b', re.I),
            
            # Range
            'x_range': re.compile(r'x\s*(?:from|range)\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)', re.I),
        }
    
    def _bootstrap_mappings(self):
        """Bootstrap mappings for plot types."""
        # First, train the encoder on all examples and keywords
        all_texts = []
        for pattern in self.patterns.values():
            all_texts.extend(pattern.examples)
            all_texts.extend(pattern.keywords)
        self.encoder.learn(all_texts)
        
        for name, pattern in self.patterns.items():
            # Create mapping for each example
            for example in pattern.examples:
                position = self.encoder.encode_input(example)
                self.map(
                    example,
                    name,
                    position=position,
                    metadata={
                        'pattern_name': name,
                        'description': pattern.description,
                        'keywords': pattern.keywords,
                    }
                )
            
            # Also map keywords directly
            for keyword in pattern.keywords:
                position = self.encoder.encode_input(keyword)
                self.map(
                    keyword,
                    name,
                    position=position,
                    metadata={
                        'pattern_name': name,
                        'is_keyword': True,
                    }
                )
    
    def _detect_plot_type(self, query: str) -> Tuple[Optional[str], float]:
        """
        Detect plot type from query using geometric matching.
        
        Returns: (plot_type, confidence)
        """
        query_lower = query.lower()
        
        # First check for exact keyword matches (bootstrap)
        for name, pattern in self.patterns.items():
            for keyword in pattern.keywords:
                if keyword in query_lower:
                    return name, 1.0
        
        # Fall back to geometric matching
        position = self.encoder.encode_input(query)
        result = self.forward(query, position=position)
        
        if result and result.confidence >= CRITICAL_LINE:
            self._last_mapping = result.mapping
            return result.output, result.confidence
        
        # Default to sine wave for generic "plot" or "wave" requests
        if any(w in query_lower for w in ['plot', 'wave', 'graph', 'chart']):
            return 'sine_wave', 0.5
        
        return None, 0.0
    
    def _extract_modifiers(self, query: str) -> Dict[str, Any]:
        """Extract modifiers from query using pattern matching."""
        modifiers = {}
        
        for name, pattern in self.modifier_patterns.items():
            match = pattern.search(query)
            if match:
                if name == 'color':
                    modifiers['color'] = match.group(1).lower()
                elif name == 'linestyle':
                    style_map = {
                        'dashed': '--',
                        'dotted': ':',
                        'solid': '-',
                        'dashdot': '-.',
                    }
                    modifiers['linestyle'] = style_map.get(match.group(1).lower(), '-')
                elif name == 'grid':
                    modifiers['grid'] = match.group(1).lower() == 'with'
                elif name == 'title':
                    modifiers['title'] = match.group(1)
                elif name == 'x_range':
                    modifiers['x_start'] = float(match.group(1))
                    modifiers['x_end'] = float(match.group(2))
                elif name in ['amplitude', 'frequency', 'phase', 'linewidth', 'num_points', 'bins']:
                    modifiers[name] = float(match.group(1))
        
        return modifiers
    
    def generate(self, query: str) -> PlotResult:
        """
        Generate matplotlib code from natural language query.
        
        Args:
            query: Natural language description of desired plot
            
        Returns:
            PlotResult with generated code and metadata
        """
        # Detect plot type
        plot_type, confidence = self._detect_plot_type(query)
        
        if not plot_type or plot_type not in self.patterns:
            return PlotResult(
                success=False,
                error=f"Could not determine plot type from query: {query}",
            )
        
        self._last_pattern = plot_type
        pattern = self.patterns[plot_type]
        
        # Extract modifiers
        modifiers = self._extract_modifiers(query)
        
        # Merge with defaults
        params = dict(pattern.defaults)
        params.update(modifiers)
        
        # Generate code
        try:
            code = pattern.template.format(**params)
        except KeyError as e:
            return PlotResult(
                success=False,
                plot_type=plot_type,
                modifiers=modifiers,
                error=f"Missing parameter: {e}",
            )
        
        return PlotResult(
            success=True,
            code=code,
            plot_type=plot_type,
            modifiers=modifiers,
        )
    
    def verify(self, result: PlotResult, execute: bool = False) -> PlotResult:
        """
        Verify generated code for syntax errors.
        Optionally execute to check runtime errors.
        """
        if not result.success:
            return result
        
        # Syntax check
        try:
            compile(result.code, '<string>', 'exec')
            result.verified = True
        except SyntaxError as e:
            result.verified = False
            result.error = f"Syntax error: {e}"
            return result
        
        # Optional execution
        if execute:
            try:
                # Create temp file and run
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(result.code)
                    temp_path = f.name
                
                proc = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if proc.returncode == 0:
                    result.output = proc.stdout
                else:
                    result.error = proc.stderr[:500]
                    result.verified = False
                
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
                
            except subprocess.TimeoutExpired:
                result.error = "Execution timed out"
                result.verified = False
            except Exception as e:
                result.error = str(e)
                result.verified = False
        
        return result
    
    def execute(self, result: PlotResult, output_dir: str = "output") -> PlotResult:
        """Execute the generated code and save the plot."""
        if not result.success:
            return result
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save script
        script_path = output_path / "generated_plot.py"
        script_path.write_text(result.code)
        
        # Execute
        try:
            proc = subprocess.run(
                ['python', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(output_path.parent),
            )
            
            if proc.returncode == 0:
                result.output = proc.stdout
                result.saved_path = str(script_path)
            else:
                result.error = proc.stderr[:500]
                
        except subprocess.TimeoutExpired:
            result.error = "Execution timed out"
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def feedback(self, success: bool) -> bool:
        """Provide feedback on the last generation."""
        if self._last_mapping:
            self.reinforce(self._last_mapping, success=success)
            return True
        return False
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List available plot patterns."""
        return [
            {
                'name': p.name,
                'description': p.description,
                'keywords': p.keywords,
                'examples': p.examples,
            }
            for p in self.patterns.values()
        ]
    
    def add_pattern(self, pattern: PlotPattern) -> None:
        """Add a new plot pattern."""
        self.patterns[pattern.name] = pattern
        
        # Add mappings
        for example in pattern.examples:
            position = self.encoder.encode_input(example)
            self.map(
                example,
                pattern.name,
                position=position,
                metadata={
                    'pattern_name': pattern.name,
                    'description': pattern.description,
                }
            )


def test_plot_space():
    """Test PlotSpace functionality."""
    space = PlotSpace()
    
    print("=" * 60)
    print("PlotSpace Test")
    print("=" * 60)
    
    # Test queries
    queries = [
        "create a sine wave plot",
        "create a sine wave plot with amplitude of 2.0",
        "create a sine wave plot with a red line",
        "create a sine wave with amplitude 3.0 and blue dashed line",
        "make a bar chart",
        "plot a scatter diagram",
        "create a histogram with 50 bins",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = space.generate(query)
        
        if result.success:
            print(f"  ✓ Plot type: {result.plot_type}")
            print(f"  ✓ Modifiers: {result.modifiers}")
            
            # Verify syntax
            result = space.verify(result)
            print(f"  ✓ Verified: {result.verified}")
            
            if result.error:
                print(f"  ✗ Error: {result.error}")
        else:
            print(f"  ✗ Failed: {result.error}")
    
    print("\n" + "=" * 60)
    print("Available patterns:")
    for p in space.list_patterns():
        print(f"  - {p['name']}: {p['description']}")


if __name__ == "__main__":
    test_plot_space()
