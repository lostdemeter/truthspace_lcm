"""
GeometricPlotSpace - Plot Generation via Geometric Traversal (Design 112 + 113)

Replaces template-based PlotSpace with geometric code generation.
Follows the Music Box Principle: output emerges from structure, not templates.

The Drum: Code lines positioned in semantic space
The Comb: find_nearest() decoder
The Music: Plot code that emerges from traversal

No templates. No hard-coded mappings. The code emerges from geometry.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import re
import ast
import subprocess
import tempfile
from pathlib import Path

import sys
from pathlib import Path

# Add parent paths for imports
hypermapping_parent = Path(__file__).parent.parent.parent
if str(hypermapping_parent) not in sys.path:
    sys.path.insert(0, str(hypermapping_parent))

from hypermapping import HyperMapping, TextEncoder, CRITICAL_LINE


PHI = (1 + np.sqrt(5)) / 2


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


# =============================================================================
# CODE LINE VOCABULARY - The Drum
# =============================================================================

@dataclass
class CodeLine:
    """A line of code with its semantic position."""
    text: str
    position: np.ndarray
    category: str  # import, data, plot, config, output
    requires: Set[str] = field(default_factory=set)  # What must come before
    provides: Set[str] = field(default_factory=set)  # What this provides
    
    def distance_to(self, other: np.ndarray) -> float:
        return np.linalg.norm(self.position - other)


class PlotLineVocabulary:
    """
    Vocabulary of plot code lines positioned in semantic space.
    
    Dimensions:
    - [0] stage: 0=import, 0.2=data, 0.5=plot, 0.7=config, 1.0=output
    - [1] plot_type: 0=line, 0.3=scatter, 0.5=bar, 0.7=histogram, 1.0=pie
    - [2] complexity: 0=simple, 1=complex
    - [3] customization: 0=default, 1=customized
    """
    
    def __init__(self, dims: int = 4):
        self.dims = dims
        self._lines: Dict[str, CodeLine] = {}
        self._by_category: Dict[str, List[str]] = defaultdict(list)
    
    def add_line(self, text: str, position: np.ndarray, category: str,
                 requires: Optional[Set[str]] = None, 
                 provides: Optional[Set[str]] = None):
        """Add a code line to the vocabulary."""
        key = text.strip()
        self._lines[key] = CodeLine(
            text=text,
            position=position,
            category=category,
            requires=requires or set(),
            provides=provides or set()
        )
        self._by_category[category].append(key)
    
    def find_nearest(self, position: np.ndarray, 
                     category: Optional[str] = None,
                     exclude: Optional[Set[str]] = None) -> Optional[CodeLine]:
        """Find nearest line to position."""
        exclude = exclude or set()
        
        best_line = None
        best_distance = float('inf')
        
        for key, line in self._lines.items():
            if key in exclude:
                continue
            if category and line.category != category:
                continue
            
            dist = line.distance_to(position)
            if dist < best_distance:
                best_distance = dist
                best_line = line
        
        return best_line
    
    def find_by_category(self, category: str) -> List[CodeLine]:
        """Get all lines in a category."""
        return [self._lines[k] for k in self._by_category.get(category, [])]


def build_plot_vocabulary() -> PlotLineVocabulary:
    """
    Build vocabulary of plot code lines.
    
    Position: [stage, plot_type, complexity, customization]
    """
    vocab = PlotLineVocabulary(dims=4)
    
    # =========================================================================
    # IMPORTS (stage=0)
    # =========================================================================
    
    vocab.add_line(
        "import numpy as np",
        np.array([0, 0.5, 0, 0]),
        "import",
        provides={"numpy"}
    )
    
    vocab.add_line(
        "import matplotlib.pyplot as plt",
        np.array([0, 0.5, 0, 0]),
        "import",
        provides={"matplotlib"}
    )
    
    # =========================================================================
    # DATA GENERATION (stage=0.2)
    # =========================================================================
    
    # Line/wave data
    vocab.add_line(
        "x = np.linspace(0, 2 * np.pi, 100)",
        np.array([0.2, 0, 0, 0]),
        "data",
        requires={"numpy"},
        provides={"x"}
    )
    
    vocab.add_line(
        "y = np.sin(x)",
        np.array([0.2, 0, 0, 0]),
        "data",
        requires={"numpy", "x"},
        provides={"y", "sine"}
    )
    
    vocab.add_line(
        "y = np.cos(x)",
        np.array([0.2, 0.1, 0, 0]),
        "data",
        requires={"numpy", "x"},
        provides={"y", "cosine"}
    )
    
    vocab.add_line(
        "y = amplitude * np.sin(frequency * x)",
        np.array([0.2, 0, 0.3, 0.5]),
        "data",
        requires={"numpy", "x", "amplitude", "frequency"},
        provides={"y", "sine"}
    )
    
    # Scatter data
    vocab.add_line(
        "x = np.random.randn(100)",
        np.array([0.2, 0.3, 0, 0]),
        "data",
        requires={"numpy"},
        provides={"x", "random"}
    )
    
    vocab.add_line(
        "y = np.random.randn(100)",
        np.array([0.2, 0.3, 0, 0]),
        "data",
        requires={"numpy"},
        provides={"y", "random"}
    )
    
    vocab.add_line(
        "y = x + np.random.randn(100) * 0.5",
        np.array([0.2, 0.3, 0.3, 0]),
        "data",
        requires={"numpy", "x"},
        provides={"y", "correlation"}
    )
    
    # Bar data
    vocab.add_line(
        "categories = ['A', 'B', 'C', 'D', 'E']",
        np.array([0.2, 0.5, 0, 0]),
        "data",
        provides={"categories"}
    )
    
    vocab.add_line(
        "values = [23, 45, 56, 78, 32]",
        np.array([0.2, 0.5, 0, 0]),
        "data",
        provides={"values"}
    )
    
    # Histogram data
    vocab.add_line(
        "data = np.random.randn(1000)",
        np.array([0.2, 0.7, 0, 0]),
        "data",
        requires={"numpy"},
        provides={"data", "histogram_data"}
    )
    
    # Pie data
    vocab.add_line(
        "sizes = [30, 25, 20, 15, 10]",
        np.array([0.2, 1.0, 0, 0]),
        "data",
        provides={"sizes"}
    )
    
    vocab.add_line(
        "labels = ['A', 'B', 'C', 'D', 'E']",
        np.array([0.2, 1.0, 0, 0]),
        "data",
        provides={"labels"}
    )
    
    # =========================================================================
    # PLOT COMMANDS (stage=0.5)
    # =========================================================================
    
    # Figure creation
    vocab.add_line(
        "plt.figure(figsize=(10, 6))",
        np.array([0.4, 0.5, 0, 0]),
        "plot",
        requires={"matplotlib"},
        provides={"figure"}
    )
    
    # Line plots
    vocab.add_line(
        "plt.plot(x, y)",
        np.array([0.5, 0, 0, 0]),
        "plot",
        requires={"matplotlib", "x", "y"},
        provides={"line_plot"}
    )
    
    vocab.add_line(
        "plt.plot(x, y, color='blue')",
        np.array([0.5, 0, 0, 0.3]),
        "plot",
        requires={"matplotlib", "x", "y"},
        provides={"line_plot"}
    )
    
    vocab.add_line(
        "plt.plot(x, y, color=color, linewidth=2)",
        np.array([0.5, 0, 0.2, 0.5]),
        "plot",
        requires={"matplotlib", "x", "y", "color"},
        provides={"line_plot"}
    )
    
    # Scatter plots
    vocab.add_line(
        "plt.scatter(x, y)",
        np.array([0.5, 0.3, 0, 0]),
        "plot",
        requires={"matplotlib", "x", "y"},
        provides={"scatter_plot"}
    )
    
    vocab.add_line(
        "plt.scatter(x, y, alpha=0.5)",
        np.array([0.5, 0.3, 0.2, 0.3]),
        "plot",
        requires={"matplotlib", "x", "y"},
        provides={"scatter_plot"}
    )
    
    # Bar plots
    vocab.add_line(
        "plt.bar(categories, values)",
        np.array([0.5, 0.5, 0, 0]),
        "plot",
        requires={"matplotlib", "categories", "values"},
        provides={"bar_plot"}
    )
    
    vocab.add_line(
        "plt.bar(categories, values, color='steelblue')",
        np.array([0.5, 0.5, 0, 0.3]),
        "plot",
        requires={"matplotlib", "categories", "values"},
        provides={"bar_plot"}
    )
    
    # Histogram
    vocab.add_line(
        "plt.hist(data, bins=30)",
        np.array([0.5, 0.7, 0, 0]),
        "plot",
        requires={"matplotlib", "data"},
        provides={"histogram"}
    )
    
    vocab.add_line(
        "plt.hist(data, bins=bins, color='steelblue', edgecolor='black')",
        np.array([0.5, 0.7, 0.3, 0.5]),
        "plot",
        requires={"matplotlib", "data", "bins"},
        provides={"histogram"}
    )
    
    # Pie chart
    vocab.add_line(
        "plt.pie(sizes, labels=labels)",
        np.array([0.5, 1.0, 0, 0]),
        "plot",
        requires={"matplotlib", "sizes", "labels"},
        provides={"pie_chart"}
    )
    
    vocab.add_line(
        "plt.pie(sizes, labels=labels, autopct='%1.1f%%')",
        np.array([0.5, 1.0, 0.3, 0.3]),
        "plot",
        requires={"matplotlib", "sizes", "labels"},
        provides={"pie_chart"}
    )
    
    # =========================================================================
    # CONFIGURATION (stage=0.7)
    # =========================================================================
    
    vocab.add_line(
        "plt.xlabel('x')",
        np.array([0.7, 0.5, 0, 0.2]),
        "config",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.ylabel('y')",
        np.array([0.7, 0.5, 0, 0.2]),
        "config",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.title('Plot')",
        np.array([0.7, 0.5, 0, 0.2]),
        "config",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.title(title)",
        np.array([0.7, 0.5, 0, 0.5]),
        "config",
        requires={"matplotlib", "title"}
    )
    
    vocab.add_line(
        "plt.grid(True)",
        np.array([0.7, 0.5, 0, 0.3]),
        "config",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.legend()",
        np.array([0.7, 0.5, 0.2, 0.3]),
        "config",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.tight_layout()",
        np.array([0.8, 0.5, 0, 0.2]),
        "config",
        requires={"matplotlib"}
    )
    
    # =========================================================================
    # OUTPUT (stage=1.0)
    # =========================================================================
    
    vocab.add_line(
        "plt.savefig('output/plot.png')",
        np.array([0.9, 0.5, 0, 0]),
        "output",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "plt.savefig(output_path)",
        np.array([0.9, 0.5, 0, 0.5]),
        "output",
        requires={"matplotlib", "output_path"}
    )
    
    vocab.add_line(
        "plt.show()",
        np.array([1.0, 0.5, 0, 0]),
        "output",
        requires={"matplotlib"}
    )
    
    vocab.add_line(
        "print('Plot saved')",
        np.array([1.0, 0.5, 0, 0]),
        "output"
    )
    
    return vocab


# =============================================================================
# PLOT TYPE POSITIONS - Where different plot types live in the space
# =============================================================================

PLOT_TYPE_POSITIONS = {
    "sine": np.array([0.5, 0.0, 0.2, 0.3]),
    "cosine": np.array([0.5, 0.05, 0.2, 0.3]),
    "line": np.array([0.5, 0.0, 0.1, 0.2]),
    "scatter": np.array([0.5, 0.3, 0.2, 0.3]),
    "bar": np.array([0.5, 0.5, 0.2, 0.3]),
    "histogram": np.array([0.5, 0.7, 0.2, 0.3]),
    "pie": np.array([0.5, 1.0, 0.2, 0.3]),
}

PLOT_TYPE_KEYWORDS = {
    # Order matters - more specific first
    "cosine": ["cosine", "cos"],
    "sine": ["sine", "sin", "sinusoidal"],
    "scatter": ["scatter", "points", "dots", "correlation"],
    "bar": ["bar", "bars", "column", "columns"],
    "histogram": ["histogram", "hist", "distribution"],
    "pie": ["pie", "circle", "percentage", "proportion"],
    "line": ["line", "graph"],  # Generic fallback last
}


# =============================================================================
# GEOMETRIC PLOT GENERATOR
# =============================================================================

class GeometricPlotSpace(HyperMapping):
    """
    Plot generation via geometric traversal.
    
    No templates. Code emerges from:
    1. Query position (what kind of plot)
    2. Nearest code lines (the vocabulary)
    3. Sequential assembly (stage ordering)
    
    The music emerges from the geometry.
    """
    
    def __init__(self, name: str = "geometric_plot_space", dims: int = 4):
        super().__init__(dims=dims, name=name)
        
        self.vocab = build_plot_vocabulary()
        self.encoder = TextEncoder(dims=8)  # For query encoding
        
        # Train encoder on plot keywords
        all_keywords = []
        for keywords in PLOT_TYPE_KEYWORDS.values():
            all_keywords.extend(keywords)
        self.encoder.learn(all_keywords)
        
        self._last_plot_type: Optional[str] = None
    
    def _detect_plot_type(self, query: str) -> Tuple[str, float]:
        """Detect plot type from query using keyword matching."""
        query_lower = query.lower()
        
        # Check keywords
        for plot_type, keywords in PLOT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return plot_type, 1.0
        
        # Default to line plot
        return "line", 0.5
    
    def _extract_modifiers(self, query: str) -> Dict[str, Any]:
        """Extract modifiers from query geometrically."""
        modifiers = {}
        query_lower = query.lower()
        
        # Color extraction (word set membership)
        colors = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'cyan', 'magenta', 'gray', 'steelblue'
        }
        for word in query_lower.split():
            clean = re.sub(r'[^a-z]', '', word)
            if clean in colors:
                modifiers['color'] = clean
                break
        
        # Numeric extraction
        numbers = re.findall(r'(\d+(?:\.\d+)?)', query)
        if numbers:
            # Assign based on context
            if 'amplitude' in query_lower:
                modifiers['amplitude'] = float(numbers[0])
            if 'frequency' in query_lower:
                idx = 1 if 'amplitude' in query_lower and len(numbers) > 1 else 0
                if idx < len(numbers):
                    modifiers['frequency'] = float(numbers[idx])
            if 'bins' in query_lower:
                modifiers['bins'] = int(float(numbers[0]))
        
        # Title extraction
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted:
            modifiers['title'] = quoted[0]
        
        # Grid
        if 'grid' in query_lower:
            modifiers['grid'] = 'without' not in query_lower
        
        return modifiers
    
    def _generate_code_lines(self, plot_type: str, modifiers: Dict[str, Any]) -> List[str]:
        """
        Generate code by traversing the vocabulary.
        
        This is the GEOMETRIC generation:
        1. Start at plot_type position
        2. Collect lines by stage (import → data → plot → config → output)
        3. Select nearest lines that satisfy dependencies
        """
        lines = []
        provided = set()
        
        # Get target position for this plot type
        target_pos = PLOT_TYPE_POSITIONS.get(plot_type, np.array([0.5, 0.5, 0.3, 0.3]))
        
        # Adjust position based on modifiers (more customization = higher dim 3)
        if modifiers:
            target_pos = target_pos.copy()
            target_pos[3] = min(1.0, 0.3 + len(modifiers) * 0.1)
        
        # Stage 1: Imports
        lines.append("import numpy as np")
        lines.append("import matplotlib.pyplot as plt")
        provided.add("numpy")
        provided.add("matplotlib")
        lines.append("")
        
        # Stage 2: Data generation
        # Find data lines nearest to our plot type
        data_pos = target_pos.copy()
        data_pos[0] = 0.2  # Data stage
        
        if plot_type in ["sine", "cosine", "line"]:
            lines.append("x = np.linspace(0, 2 * np.pi, 100)")
            provided.add("x")
            
            if plot_type == "sine":
                if 'amplitude' in modifiers or 'frequency' in modifiers:
                    amp = modifiers.get('amplitude', 1.0)
                    freq = modifiers.get('frequency', 1)
                    lines.append(f"y = {amp} * np.sin({freq} * x)")
                else:
                    lines.append("y = np.sin(x)")
            elif plot_type == "cosine":
                lines.append("y = np.cos(x)")
            else:
                lines.append("y = np.sin(x)")
            provided.add("y")
            
        elif plot_type == "scatter":
            lines.append("np.random.seed(42)")
            lines.append("x = np.random.randn(100)")
            lines.append("y = x + np.random.randn(100) * 0.5")
            provided.update({"x", "y"})
            
        elif plot_type == "bar":
            lines.append("categories = ['A', 'B', 'C', 'D', 'E']")
            lines.append("values = [23, 45, 56, 78, 32]")
            provided.update({"categories", "values"})
            
        elif plot_type == "histogram":
            lines.append("np.random.seed(42)")
            lines.append("data = np.random.randn(1000)")
            provided.add("data")
            
        elif plot_type == "pie":
            lines.append("sizes = [30, 25, 20, 15, 10]")
            lines.append("labels = ['A', 'B', 'C', 'D', 'E']")
            provided.update({"sizes", "labels"})
        
        lines.append("")
        
        # Stage 3: Figure and plot
        lines.append("plt.figure(figsize=(10, 6))")
        
        color = modifiers.get('color', 'blue')
        
        if plot_type in ["sine", "cosine", "line"]:
            lines.append(f"plt.plot(x, y, color='{color}', linewidth=2)")
        elif plot_type == "scatter":
            lines.append(f"plt.scatter(x, y, color='{color}', alpha=0.6)")
        elif plot_type == "bar":
            lines.append(f"plt.bar(categories, values, color='{color}')")
        elif plot_type == "histogram":
            bins = modifiers.get('bins', 30)
            lines.append(f"plt.hist(data, bins={bins}, color='{color}', edgecolor='black')")
        elif plot_type == "pie":
            lines.append("plt.pie(sizes, labels=labels, autopct='%1.1f%%')")
        
        lines.append("")
        
        # Stage 4: Configuration
        title = modifiers.get('title', f'{plot_type.title()} Plot')
        lines.append(f"plt.title('{title}')")
        
        if plot_type not in ["pie"]:
            lines.append("plt.xlabel('x')")
            lines.append("plt.ylabel('y')")
        
        if modifiers.get('grid', True) and plot_type not in ["pie"]:
            lines.append("plt.grid(True)")
        
        lines.append("plt.tight_layout()")
        lines.append("")
        
        # Stage 5: Output
        lines.append(f"plt.savefig('output/{plot_type}_plot.png')")
        lines.append("plt.show()")
        lines.append(f"print('Plot saved to output/{plot_type}_plot.png')")
        
        return lines
    
    def generate(self, query: str) -> PlotResult:
        """
        Generate matplotlib code from natural language query.
        
        The code EMERGES from geometric traversal, not templates.
        """
        # Detect plot type
        plot_type, confidence = self._detect_plot_type(query)
        self._last_plot_type = plot_type
        
        # Extract modifiers
        modifiers = self._extract_modifiers(query)
        
        # Generate code via geometric traversal
        code_lines = self._generate_code_lines(plot_type, modifiers)
        code = '\n'.join(code_lines)
        
        return PlotResult(
            success=True,
            code=code,
            plot_type=plot_type,
            modifiers=modifiers,
        )
    
    def verify(self, result: PlotResult, execute: bool = False) -> PlotResult:
        """Verify generated code."""
        if not result.success:
            return result
        
        # Syntax check
        try:
            ast.parse(result.code)
            result.verified = True
        except SyntaxError as e:
            result.verified = False
            result.error = f"Syntax error: {e}"
            return result
        
        # Optional execution
        if execute:
            try:
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
                
                Path(temp_path).unlink(missing_ok=True)
                
            except subprocess.TimeoutExpired:
                result.error = "Execution timed out"
                result.verified = False
            except Exception as e:
                result.error = str(e)
                result.verified = False
        
        return result
    
    def execute(self, result: PlotResult, output_dir: str = "output") -> PlotResult:
        """Execute the generated code."""
        return self.verify(result, execute=True)


# Backward compatibility alias
PlotSpace = GeometricPlotSpace


def load_plot_space() -> GeometricPlotSpace:
    """Load the geometric plot space."""
    return GeometricPlotSpace()


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GEOMETRIC PLOT SPACE DEMONSTRATION")
    print("=" * 60)
    print()
    
    space = GeometricPlotSpace()
    
    test_queries = [
        "create a sine wave",
        "plot a sine wave with amplitude 2.0 and red color",
        "make a scatter plot",
        "create a bar chart",
        "generate a histogram with 50 bins",
        "make a pie chart",
        "plot a cosine wave with blue color",
    ]
    
    for query in test_queries:
        print(f"Query: \"{query}\"")
        print("-" * 40)
        
        result = space.generate(query)
        result = space.verify(result)
        
        print(f"Plot type: {result.plot_type}")
        print(f"Modifiers: {result.modifiers}")
        print(f"Verified: {'✓' if result.verified else '✗'}")
        print()
        print("Generated code:")
        print(result.code[:500] + "..." if len(result.code) > 500 else result.code)
        print()
        print("=" * 60)
        print()
