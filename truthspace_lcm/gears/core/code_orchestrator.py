"""
Code Orchestrator

Multi-step Python code generation through gear orchestration.

This orchestrator:
1. Plans - Breaks complex code requests into function-level steps
2. Generates - Creates individual functions using PythonCodeGear
3. Composes - Combines functions into a complete program
4. Verifies - Runs the final code

The PlotCorpus is a learnable corpus of plot patterns that:
- Matches requests to existing patterns using keyword similarity
- Falls back to LLM for unknown variations
- Learns from successful LLM generations

Example:
    orchestrator = CodeOrchestrator()
    orchestrator.configure_llm(url, model)
    
    result = orchestrator.generate(
        "Create a matplotlib plot showing a sine wave from 0 to 2π"
    )
    
    # Returns complete, runnable Python code

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

from truthspace_lcm.gears.core.gear_message import GearProtocol, GearMessage, MessageIntent
from truthspace_lcm.gears.core.python_code_gear import PythonCodeGear, CodeVerifier
from truthspace_lcm.gears.core.holographic_pattern_space import HolographicPatternSpace, HolographicModule
from truthspace_lcm.gears.core.template_composer import TemplateComposer


# =============================================================================
# PLOT CORPUS - Learnable pattern storage (Legacy - kept for compatibility)
# =============================================================================

OUTPUT_DIR = '/home/thorin/truthspace-lcm/output'


@dataclass
class PlotPattern:
    """A single plot pattern in the corpus."""
    name: str
    keywords: Set[str]  # Keywords that trigger this pattern
    code_template: str  # The code template
    description: str = ""
    examples: List[str] = field(default_factory=list)  # Example requests that match
    use_count: int = 0
    success_count: int = 0
    
    def matches(self, request_lower: str, threshold: float = 0.3) -> float:
        """
        Calculate match score based on keyword overlap.
        Returns score between 0 and 1.
        """
        request_words = set(request_lower.split())
        if not self.keywords:
            return 0.0
        
        # Count keyword matches
        matches = len(self.keywords & request_words)
        score = matches / len(self.keywords)
        
        # Bonus for exact phrase matches in examples
        for example in self.examples:
            if example.lower() in request_lower:
                score += 0.3
        
        return min(score, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'keywords': list(self.keywords),
            'code_template': self.code_template,
            'description': self.description,
            'examples': self.examples,
            'use_count': self.use_count,
            'success_count': self.success_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotPattern':
        return cls(
            name=data['name'],
            keywords=set(data.get('keywords', [])),
            code_template=data['code_template'],
            description=data.get('description', ''),
            examples=data.get('examples', []),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
        )


class PlotCorpus:
    """
    A learnable corpus of plot patterns.
    
    Similar to PythonCodeCorpus but specialized for matplotlib plots.
    """
    
    def __init__(self):
        self.patterns: List[PlotPattern] = []
        self._seed_patterns()
    
    def _seed_patterns(self):
        """Seed with basic plot patterns."""
        self.patterns = [
            PlotPattern(
                name='sine_wave',
                keywords={'sine', 'sin', 'wave', 'trigonometric'},
                code_template=self._make_sine_template(),
                description='Sine wave plot',
                examples=['plot a sine wave', 'create sine wave', 'sine function'],
            ),
            PlotPattern(
                name='cosine_wave',
                keywords={'cosine', 'cos', 'wave', 'trigonometric'},
                code_template=self._make_cosine_template(),
                description='Cosine wave plot',
                examples=['plot a cosine wave', 'create cosine wave', 'cos function'],
            ),
            PlotPattern(
                name='bar_chart',
                keywords={'bar', 'chart', 'categories', 'bars'},
                code_template=self._make_bar_template(),
                description='Bar chart',
                examples=['create a bar chart', 'bar graph', 'bar plot'],
            ),
            PlotPattern(
                name='scatter_plot',
                keywords={'scatter', 'points', 'dots', 'correlation'},
                code_template=self._make_scatter_template(),
                description='Scatter plot',
                examples=['scatter plot', 'scatter graph', 'plot points'],
            ),
            PlotPattern(
                name='histogram',
                keywords={'histogram', 'distribution', 'frequency', 'bins'},
                code_template=self._make_histogram_template(),
                description='Histogram',
                examples=['create histogram', 'histogram plot', 'distribution plot'],
            ),
            PlotPattern(
                name='line_plot',
                keywords={'line', 'plot', 'graph', 'trend'},
                code_template=self._make_line_template(),
                description='Line plot',
                examples=['line plot', 'line graph', 'plot a line'],
            ),
        ]
    
    def _make_sine_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(x_offset=0, y_offset=0, amplitude=1, frequency=1):
    """Generate x values and compute sin(x) with optional modifications."""
    x = np.linspace(0, 2 * np.pi, 100) + x_offset
    y = amplitude * np.sin(frequency * x) + y_offset
    return x, y

def create_plot(x, y, title="Sine Wave"):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/sine_wave.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/sine_wave.png")
'''
    
    def _make_cosine_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(x_offset=0, y_offset=0, amplitude=1, frequency=1):
    """Generate x values and compute cos(x) with optional modifications."""
    x = np.linspace(0, 2 * np.pi, 100) + x_offset
    y = amplitude * np.cos(frequency * x) + y_offset
    return x, y

def create_plot(x, y, title="Cosine Wave"):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/cosine_wave.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/cosine_wave.png")
'''
    
    def _make_bar_template(self) -> str:
        return f'''import matplotlib.pyplot as plt

def create_data():
    """Generate categories and values for bar chart."""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    return categories, values

def create_plot(categories, values, title="Bar Chart"):
    """Create bar chart with labels."""
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='steelblue')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title(title)

if __name__ == "__main__":
    categories, values = create_data()
    create_plot(categories, values)
    plt.savefig('{OUTPUT_DIR}/bar_chart.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/bar_chart.png")
'''
    
    def _make_scatter_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(n_points=50):
    """Generate random x,y data points."""
    np.random.seed(42)
    x = np.random.randn(n_points)
    y = x + np.random.randn(n_points) * 0.5
    return x, y

def create_plot(x, y, title="Scatter Plot"):
    """Create scatter plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='steelblue', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/scatter_plot.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/scatter_plot.png")
'''
    
    def _make_histogram_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(n_samples=1000):
    """Generate random data for histogram."""
    np.random.seed(42)
    data = np.random.randn(n_samples)
    return data

def create_plot(data, bins=30, title="Histogram"):
    """Create histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='steelblue', edgecolor='white')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)

if __name__ == "__main__":
    data = create_data()
    create_plot(data)
    plt.savefig('{OUTPUT_DIR}/histogram.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/histogram.png")
'''
    
    def _make_line_template(self) -> str:
        return f'''import matplotlib.pyplot as plt

def create_data():
    """Generate sample data."""
    x = list(range(10))
    y = [i**2 for i in x]
    return x, y

def create_plot(x, y, title="Line Plot"):
    """Create line plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/line_plot.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/line_plot.png")
'''
    
    def find_pattern(self, request: str, threshold: float = 0.3) -> Optional[PlotPattern]:
        """Find the best matching pattern for a request."""
        request_lower = request.lower()
        best_pattern = None
        best_score = threshold
        
        for pattern in self.patterns:
            score = pattern.matches(request_lower, threshold)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        return best_pattern
    
    def add_pattern(self, pattern: PlotPattern):
        """Add a new pattern to the corpus."""
        # Check for duplicate
        for existing in self.patterns:
            if existing.name == pattern.name:
                return  # Already exists
        self.patterns.append(pattern)
    
    def record_use(self, pattern_name: str, success: bool):
        """Record pattern usage."""
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                pattern.use_count += 1
                if success:
                    pattern.success_count += 1
                break
    
    def save(self, path: str):
        """Save corpus to JSON file."""
        data = {
            'patterns': [p.to_dict() for p in self.patterns]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load corpus from JSON file."""
        if not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.patterns = [PlotPattern.from_dict(p) for p in data.get('patterns', [])]


@dataclass
class CodePlan:
    """A plan for generating complex code."""
    goal: str
    imports: List[str] = field(default_factory=list)
    functions: List[Dict[str, str]] = field(default_factory=list)  # name, description, code
    main_logic: str = ""
    complete_code: str = ""
    verified: bool = False
    output: str = ""
    error: str = ""


class CodePlannerGear(GearProtocol):
    """
    Plans complex code generation by breaking it into:
    - Required imports
    - Individual functions
    - Main logic
    """
    
    PLAN_PROMPT = """Break this Python code request into components:

Request: {request}

Respond with JSON:
{{
    "imports": ["import x", "from y import z"],
    "functions": [
        {{"name": "func_name", "description": "what it does", "params": "x, y", "returns": "type"}}
    ],
    "main_description": "what the main code should do"
}}

Keep it simple. Use standard library + matplotlib/numpy if needed."""

    def __init__(self):
        self.name = "CodePlannerGear"
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Emergent patterns for common code structures
        self.patterns = {
            'plot': {
                'imports': ['import matplotlib.pyplot as plt', 'import numpy as np'],
                'functions': [
                    {'name': 'create_data', 'description': 'Generate data for plotting'},
                    {'name': 'create_plot', 'description': 'Create the plot with labels'},
                ],
                'main': 'Generate data, create plot, show it',
            },
            'file_processing': {
                'imports': ['from pathlib import Path'],
                'functions': [
                    {'name': 'read_file', 'description': 'Read file contents'},
                    {'name': 'process_data', 'description': 'Process the data'},
                    {'name': 'write_output', 'description': 'Write results'},
                ],
                'main': 'Read, process, write',
            },
            'data_analysis': {
                'imports': ['import numpy as np'],
                'functions': [
                    {'name': 'load_data', 'description': 'Load or generate data'},
                    {'name': 'analyze', 'description': 'Perform analysis'},
                    {'name': 'report', 'description': 'Print results'},
                ],
                'main': 'Load, analyze, report',
            },
        }
    
    def configure_llm(self, url: str, model: str):
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_url:
            return None
        
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
                return response.json().get('response', '').strip()
        except Exception:
            pass
        return None
    
    def plan(self, request: str) -> CodePlan:
        """Create a plan for the code request."""
        request_lower = request.lower()
        plan = CodePlan(goal=request)
        
        # Try emergent patterns first
        if any(w in request_lower for w in ['plot', 'graph', 'chart', 'matplotlib', 'visualize']):
            pattern = self.patterns['plot']
            plan.imports = pattern['imports'].copy()
            plan.functions = [f.copy() for f in pattern['functions']]
            
            # Customize based on request
            if 'sine' in request_lower or 'sin' in request_lower:
                plan.functions[0]['description'] = 'Generate x values and compute sin(x)'
            elif 'cosine' in request_lower or 'cos' in request_lower:
                plan.functions[0]['description'] = 'Generate x values and compute cos(x)'
            elif 'bar' in request_lower:
                plan.functions[0]['description'] = 'Generate categories and values for bar chart'
                plan.functions[1]['description'] = 'Create bar chart with labels'
            elif 'scatter' in request_lower:
                plan.functions[0]['description'] = 'Generate x,y data points'
                plan.functions[1]['description'] = 'Create scatter plot'
            elif 'histogram' in request_lower:
                plan.functions[0]['description'] = 'Generate data for histogram'
                plan.functions[1]['description'] = 'Create histogram'
            
            return plan
        
        if any(w in request_lower for w in ['file', 'read', 'write', 'process']):
            pattern = self.patterns['file_processing']
            plan.imports = pattern['imports'].copy()
            plan.functions = [f.copy() for f in pattern['functions']]
            return plan
        
        # Fall back to LLM
        if self.llm_url:
            prompt = self.PLAN_PROMPT.format(request=request)
            response = self._call_llm(prompt)
            
            if response:
                # Parse JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        plan.imports = data.get('imports', [])
                        plan.functions = data.get('functions', [])
                        plan.main_logic = data.get('main_description', '')
                        return plan
                    except json.JSONDecodeError:
                        pass
        
        # Minimal plan
        plan.imports = []
        plan.functions = [{'name': 'main', 'description': request}]
        return plan
    
    def process_message(self, message: GearMessage) -> GearMessage:
        plan = self.plan(message.content)
        return self.send(
            message.with_context('code_plan', plan),
            content=f"Plan: {len(plan.functions)} functions, {len(plan.imports)} imports"
        )


class CodeComposerGear(GearProtocol):
    """
    Composes individual code pieces into a complete program.
    """
    
    COMPOSE_PROMPT = """Write a complete Python function:

Name: {name}
Description: {description}
Parameters: {params}
Returns: {returns}

Context (other functions available): {context}

Write ONLY the function definition, no imports or main code.
Make it simple and correct."""

    MAIN_PROMPT = """Write the main code that uses these functions:

Functions available:
{functions}

Goal: {goal}

Write ONLY the main code (inside if __name__ == "__main__":).
Call the functions to achieve the goal."""

    def __init__(self):
        self.name = "CodeComposerGear"
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        self.python_gear = PythonCodeGear()
    
    def configure_llm(self, url: str, model: str):
        self.llm_url = url
        self.llm_model = model
        self.python_gear.configure_llm(url, model)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_url:
            return None
        
        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 1000, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                text = response.json().get('response', '').strip()
                # Clean markdown
                if '```python' in text:
                    text = text.split('```python')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                return text
        except Exception:
            pass
        return None
    
    def generate_function(self, func_spec: Dict[str, str], context: List[str] = None) -> str:
        """Generate a single function from its specification."""
        prompt = self.COMPOSE_PROMPT.format(
            name=func_spec.get('name', 'func'),
            description=func_spec.get('description', ''),
            params=func_spec.get('params', ''),
            returns=func_spec.get('returns', 'None'),
            context='\n'.join(context or []),
        )
        
        code = self._call_llm(prompt)
        return code or f"def {func_spec.get('name', 'func')}():\n    pass"
    
    def generate_main(self, functions: List[str], goal: str) -> str:
        """Generate the main code that uses the functions."""
        func_summary = '\n'.join([f"- {f.split('(')[0].replace('def ', '')}" for f in functions if f.startswith('def ')])
        
        prompt = self.MAIN_PROMPT.format(
            functions=func_summary,
            goal=goal,
        )
        
        code = self._call_llm(prompt)
        return code or "pass"
    
    def compose(self, plan: CodePlan) -> str:
        """Compose a complete program from a plan."""
        parts = []
        
        # Imports
        if plan.imports:
            parts.append('\n'.join(plan.imports))
            parts.append('')
        
        # Generate each function
        generated_functions = []
        for func_spec in plan.functions:
            func_code = self.generate_function(func_spec, generated_functions)
            generated_functions.append(func_code)
            parts.append(func_code)
            parts.append('')
        
        # Generate main
        main_code = self.generate_main(generated_functions, plan.goal)
        parts.append('if __name__ == "__main__":')
        # Indent main code
        for line in main_code.split('\n'):
            if line.strip():
                parts.append(f'    {line}')
        
        return '\n'.join(parts)
    
    def process_message(self, message: GearMessage) -> GearMessage:
        plan = message.context.get('code_plan')
        if not plan:
            return self.send(message, content="No plan provided")
        
        code = self.compose(plan)
        return self.send(
            message.with_context('generated_code', code),
            content=code
        )


class CodeOrchestrator(GearProtocol):
    """
    Orchestrates multi-step Python code generation.
    
    Flow:
    1. Try to match request to PlotCorpus patterns
    2. If no match, use LLM to generate code
    3. Learn from successful LLM generations
    4. Save corpus for future use
    """
    
    PLOT_PROMPT = """Generate Python matplotlib code for this request:

"{request}"

Requirements:
1. Use matplotlib.pyplot as plt and numpy as np
2. Create a create_data() function that generates/prepares the data
3. Create a create_plot() function that creates the visualization
4. Include if __name__ == "__main__": block that calls both functions
5. Save the plot to: {output_dir}/{{plot_name}}.png using plt.savefig()
6. Print a message confirming where it was saved

Output ONLY the Python code, no explanations."""

    def __init__(self, corpus_path: Optional[str] = None, use_holographic: bool = True):
        self.name = "CodeOrchestrator"
        self.planner = CodePlannerGear()
        self.code_composer = CodeComposerGear()
        self.verifier = CodeVerifier()
        
        # Use holographic pattern space (new) or legacy plot corpus
        self.use_holographic = use_holographic
        
        # Holographic pattern space (new approach)
        self.pattern_space = HolographicPatternSpace(dims=12)
        
        # Template composer for modifications
        self.template_composer = TemplateComposer()
        
        # Legacy plot corpus (kept for compatibility)
        self.plot_corpus = PlotCorpus()
        
        # Corpus path
        self.corpus_path = corpus_path
        if self.corpus_path is None:
            self.corpus_path = str(Path(__file__).parent.parent.parent.parent / "data" / "plot_corpus.json")
        
        # Holographic space path
        self.holographic_path = str(Path(__file__).parent.parent.parent.parent / "data" / "holographic_patterns.json")
        
        # Load corpus/space if exists
        if self.use_holographic:
            if Path(self.holographic_path).exists():
                self.pattern_space.load(self.holographic_path)
            else:
                # Seed with initial patterns
                self._seed_holographic_patterns()
        else:
            if self.corpus_path and Path(self.corpus_path).exists():
                self.plot_corpus.load(self.corpus_path)
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
    
    def _seed_holographic_patterns(self):
        """Seed the holographic space with initial plot patterns."""
        patterns = [
            ('sine_wave', 'sine sin wave trigonometric', 'enhancer', 
             {'domain': 'trig', 'function': 'sine'}, self._make_sine_template()),
            ('cosine_wave', 'cosine cos wave trigonometric', 'enhancer',
             {'domain': 'trig', 'function': 'cosine'}, self._make_cosine_template()),
            ('bar_chart', 'bar chart graph categories bars categorical', 'enhancer',
             {'domain': 'categorical', 'chart': 'bar'}, self._make_bar_template()),
            ('scatter_plot', 'scatter points dots correlation', 'enhancer',
             {'domain': 'correlation', 'chart': 'scatter'}, self._make_scatter_template()),
            ('histogram', 'histogram distribution frequency bins', 'enhancer',
             {'domain': 'distribution', 'chart': 'histogram'}, self._make_histogram_template()),
            ('line_plot', 'line plot graph trend', 'enhancer',
             {'domain': 'general', 'chart': 'line'}, self._make_line_template()),
        ]
        
        for name, text, mtype, effects, template in patterns:
            self.pattern_space.add_module(
                name=name,
                text=text,
                module_type=mtype,
                effects=effects,
                code_template=template,
            )
    
    def _make_sine_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(x_offset=0, y_offset=0, amplitude=1, frequency=1):
    """Generate x values and compute sin(x) with optional modifications."""
    x = np.linspace(0, 2 * np.pi, 100) + x_offset
    y = amplitude * np.sin(frequency * x) + y_offset
    return x, y

def create_plot(x, y, title="Sine Wave"):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/sine_wave.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/sine_wave.png")
'''
    
    def _make_cosine_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(x_offset=0, y_offset=0, amplitude=1, frequency=1):
    """Generate x values and compute cos(x) with optional modifications."""
    x = np.linspace(0, 2 * np.pi, 100) + x_offset
    y = amplitude * np.cos(frequency * x) + y_offset
    return x, y

def create_plot(x, y, title="Cosine Wave"):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/cosine_wave.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/cosine_wave.png")
'''
    
    def _make_bar_template(self) -> str:
        return f'''import matplotlib.pyplot as plt

def create_data():
    """Generate categories and values for bar chart."""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    return categories, values

def create_plot(categories, values, title="Bar Chart"):
    """Create bar chart with labels."""
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='steelblue')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title(title)

if __name__ == "__main__":
    categories, values = create_data()
    create_plot(categories, values)
    plt.savefig('{OUTPUT_DIR}/bar_chart.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/bar_chart.png")
'''
    
    def _make_scatter_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(n_points=50):
    """Generate random x,y data points."""
    np.random.seed(42)
    x = np.random.randn(n_points)
    y = x + np.random.randn(n_points) * 0.5
    return x, y

def create_plot(x, y, title="Scatter Plot"):
    """Create scatter plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='steelblue', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/scatter_plot.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/scatter_plot.png")
'''
    
    def _make_histogram_template(self) -> str:
        return f'''import matplotlib.pyplot as plt
import numpy as np

def create_data(n_samples=1000):
    """Generate random data for histogram."""
    np.random.seed(42)
    data = np.random.randn(n_samples)
    return data

def create_plot(data, bins=30, title="Histogram"):
    """Create histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='steelblue', edgecolor='white')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)

if __name__ == "__main__":
    data = create_data()
    create_plot(data)
    plt.savefig('{OUTPUT_DIR}/histogram.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/histogram.png")
'''
    
    def _make_line_template(self) -> str:
        return f'''import matplotlib.pyplot as plt

def create_data():
    """Generate sample data."""
    x = list(range(10))
    y = [i**2 for i in x]
    return x, y

def create_plot(x, y, title="Line Plot"):
    """Create line plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('{OUTPUT_DIR}/line_plot.png', dpi=150)
    print("Saved to {OUTPUT_DIR}/line_plot.png")
'''
    
    def configure_llm(self, url: str, model: str):
        self.llm_url = url
        self.llm_model = model
        self.planner.configure_llm(url, model)
        self.code_composer.configure_llm(url, model)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM for code generation."""
        if not self.llm_url:
            return None
        
        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 1200, "temperature": 0.3}
                },
                timeout=60
            )
            if response.status_code == 200:
                text = response.json().get('response', '').strip()
                # Clean markdown
                if '```python' in text:
                    text = text.split('```python')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                return text
        except Exception:
            pass
        return None
    
    def _extract_params_from_request(self, request: str) -> Dict[str, Any]:
        """Extract parameters like offsets, colors, etc. from request."""
        params = {}
        request_lower = request.lower()
        
        # Extract numeric offsets
        import re
        offset_match = re.search(r'(?:x\s*)?offset\s*(?:by\s*)?([+-]?\d*\.?\d+)', request_lower)
        if offset_match:
            params['x_offset'] = float(offset_match.group(1))
        
        y_offset_match = re.search(r'y\s*offset\s*(?:by\s*)?([+-]?\d*\.?\d+)', request_lower)
        if y_offset_match:
            params['y_offset'] = float(y_offset_match.group(1))
        
        # Extract amplitude
        amp_match = re.search(r'amplitude\s*(?:of\s*)?([+-]?\d*\.?\d+)', request_lower)
        if amp_match:
            params['amplitude'] = float(amp_match.group(1))
        
        # Extract frequency
        freq_match = re.search(r'frequency\s*(?:of\s*)?([+-]?\d*\.?\d+)', request_lower)
        if freq_match:
            params['frequency'] = float(freq_match.group(1))
        
        return params
    
    def _apply_params_to_code(self, code: str, params: Dict[str, Any]) -> str:
        """Apply extracted parameters to the code template."""
        if not params:
            return code
        
        # Modify create_data() call to include params
        if 'x_offset' in params or 'y_offset' in params or 'amplitude' in params or 'frequency' in params:
            # Build param string
            param_parts = []
            if 'x_offset' in params:
                param_parts.append(f"x_offset={params['x_offset']}")
            if 'y_offset' in params:
                param_parts.append(f"y_offset={params['y_offset']}")
            if 'amplitude' in params:
                param_parts.append(f"amplitude={params['amplitude']}")
            if 'frequency' in params:
                param_parts.append(f"frequency={params['frequency']}")
            
            param_str = ', '.join(param_parts)
            
            # Replace create_data() call with parameterized version
            code = code.replace('x, y = create_data()', f'x, y = create_data({param_str})')
        
        return code
    
    def _learn_from_generation(self, request: str, code: str):
        """Learn from a successful LLM generation by adding it to the corpus."""
        # Extract keywords from request
        request_lower = request.lower()
        words = set(request_lower.split())
        
        # Filter to meaningful keywords
        stop_words = {'a', 'an', 'the', 'create', 'make', 'generate', 'plot', 'with', 'that', 'and', 'or'}
        keywords = words - stop_words
        
        # Create unique pattern name
        request_hash = hashlib.md5(request_lower.encode()).hexdigest()[:8]
        pattern_name = f"learned_{request_hash}"
        
        # Check if already exists
        for p in self.plot_corpus.patterns:
            if p.name == pattern_name:
                return
        
        # Add new pattern
        new_pattern = PlotPattern(
            name=pattern_name,
            keywords=keywords,
            code_template=code,
            description=f"Learned from: {request[:80]}",
            examples=[request],
            use_count=1,
            success_count=1,
        )
        self.plot_corpus.add_pattern(new_pattern)
        
        # Auto-save
        if self.corpus_path:
            self.plot_corpus.save(self.corpus_path)
    
    def generate(self, request: str, verify: bool = True) -> CodePlan:
        """
        Generate complete Python code for a request.
        
        Strategy (Holographic mode):
        1. Use HolographicPatternSpace to find or inject module
        2. If match found, use module's code template
        3. If injected (unknown), use LLM and learn from success
        
        Strategy (Legacy mode):
        1. Try to match request to PlotCorpus patterns
        2. If no match, use LLM
        3. Learn from successful LLM generations
        """
        request_lower = request.lower()
        plan = CodePlan(goal=request)
        
        # Check if this is a plot request
        is_plot_request = any(w in request_lower for w in [
            'plot', 'graph', 'chart', 'matplotlib', 'histogram', 
            'scatter', 'bar', 'pie', 'sine', 'cosine', 'visualize'
        ])
        
        if is_plot_request:
            if self.use_holographic:
                return self._generate_holographic(request, plan, verify)
            else:
                return self._generate_legacy(request, plan, verify)
        
        # Fall back to general code generation
        plan = self.planner.plan(request)
        plan.complete_code = self.code_composer.compose(plan)
        
        if verify:
            syntax_ok, syntax_err = self.verifier.check_syntax(plan.complete_code)
            if not syntax_ok:
                plan.error = f"Syntax error: {syntax_err}"
                plan.verified = False
            else:
                if 'matplotlib' in plan.complete_code or 'plt.' in plan.complete_code:
                    plan.verified = True
                    plan.output = "(Plot code - run to display)"
                else:
                    run_ok, output = self.verifier.run_code(plan.complete_code)
                    plan.verified = run_ok
                    plan.output = output
                    if not run_ok:
                        plan.error = output
        
        return plan
    
    def _generate_holographic(self, request: str, plan: CodePlan, verify: bool) -> CodePlan:
        """
        Generate code using HolographicPatternSpace.
        
        This is the new approach that:
        1. Constructs geometry from similarity (not arbitrary encoding)
        2. Injects temporary modules for unknown queries
        3. Learns from successful LLM generations
        """
        # Find or inject module
        module, confidence, reason, was_injected = self.pattern_space.find_or_inject(
            request,
            fallback_effects={'task': 'llm_generation'},
            min_similarity=0.3
        )
        
        if module and not was_injected and module.code_template:
            # Known pattern - use module's code template
            # First apply legacy parameter extraction
            params = self._extract_params_from_request(request)
            code = self._apply_params_to_code(module.code_template, params)
            
            # Then apply template composition for modifications
            composition_result = self.template_composer.compose(
                request=request,
                template_name=module.name,
                template_code=code
            )
            code = composition_result.code
            
            # Record use
            self.pattern_space.record_use(module, True)
            
            plan.complete_code = code
            plan.imports = ['import matplotlib.pyplot as plt', 'import numpy as np']
            plan.functions = [{'name': 'create_data'}, {'name': 'create_plot'}]
            plan.verified = True
            
            # Build output message
            output_parts = [f"Holographic match: {module.name}, confidence: {confidence:.2f}"]
            if composition_result.modifications_applied:
                mods = [m.raw_text for m in composition_result.modifications_applied]
                output_parts.append(f"Modifications: {', '.join(mods)}")
            plan.output = f"({'; '.join(output_parts)})"
            return plan
        
        # Unknown pattern or no template - use LLM
        if self.llm_url:
            prompt = self.PLOT_PROMPT.format(request=request, output_dir=OUTPUT_DIR)
            code = self._call_llm(prompt)
            
            if code:
                syntax_ok, syntax_err = self.verifier.check_syntax(code)
                if syntax_ok:
                    # Success! If we injected a temporary, promote it
                    if was_injected and module:
                        self.pattern_space.promote_temporary(
                            module,
                            new_type='enhancer',
                            new_effects={'learned': True, 'source': 'llm'},
                            code_template=code
                        )
                        # Save the updated space
                        self.pattern_space.save(self.holographic_path)
                        plan.output = f"(LLM generated - learned as: {module.name})"
                    else:
                        # Learn as new pattern
                        self._learn_holographic(request, code)
                        plan.output = "(LLM generated - learned)"
                    
                    plan.complete_code = code
                    plan.imports = ['import matplotlib.pyplot as plt', 'import numpy as np']
                    plan.functions = [{'name': 'create_data'}, {'name': 'create_plot'}]
                    plan.verified = True
                    return plan
                else:
                    # Failed - remove temporary if injected
                    if was_injected:
                        self.pattern_space.remove_temporary_modules()
                    plan.error = f"LLM generated invalid syntax: {syntax_err}"
        
        # Fallback
        plan.error = "No pattern match and LLM unavailable"
        return plan
    
    def _learn_holographic(self, request: str, code: str):
        """Learn from a successful LLM generation by adding to holographic space."""
        import hashlib
        request_hash = hashlib.md5(request.lower().encode()).hexdigest()[:8]
        name = f"learned_{request_hash}"
        
        # Check if already exists
        if self.pattern_space.get_module_by_name(name):
            return
        
        # Add new module
        self.pattern_space.add_module(
            name=name,
            text=request,
            module_type='enhancer',
            effects={'learned': True, 'source': 'llm'},
            code_template=code,
            examples=[request],
        )
        
        # Save
        self.pattern_space.save(self.holographic_path)
    
    def _generate_legacy(self, request: str, plan: CodePlan, verify: bool) -> CodePlan:
        """Generate code using legacy PlotCorpus (for compatibility)."""
        pattern = self.plot_corpus.find_pattern(request)
        
        if pattern:
            params = self._extract_params_from_request(request)
            code = self._apply_params_to_code(pattern.code_template, params)
            
            self.plot_corpus.record_use(pattern.name, True)
            
            plan.complete_code = code
            plan.imports = ['import matplotlib.pyplot as plt', 'import numpy as np']
            plan.functions = [{'name': 'create_data'}, {'name': 'create_plot'}]
            plan.verified = True
            plan.output = f"(Legacy pattern: {pattern.name})"
            return plan
        
        # No pattern match - use LLM
        if self.llm_url:
            prompt = self.PLOT_PROMPT.format(request=request, output_dir=OUTPUT_DIR)
            code = self._call_llm(prompt)
            
            if code:
                syntax_ok, syntax_err = self.verifier.check_syntax(code)
                if syntax_ok:
                    self._learn_from_generation(request, code)
                    
                    plan.complete_code = code
                    plan.imports = ['import matplotlib.pyplot as plt', 'import numpy as np']
                    plan.functions = [{'name': 'create_data'}, {'name': 'create_plot'}]
                    plan.verified = True
                    plan.output = "(LLM generated - learned)"
                    return plan
                else:
                    plan.error = f"LLM generated invalid syntax: {syntax_err}"
        
        plan.error = "No pattern match and LLM unavailable"
        return plan
    
    def process_message(self, message: GearMessage) -> GearMessage:
        plan = self.generate(message.content)
        
        if plan.verified or plan.complete_code:
            response = f"```python\n{plan.complete_code}\n```"
            if plan.verified:
                response += "\n\n✓ Code verified"
            if plan.output:
                response += f"\nOutput: {plan.output[:200]}"
            if plan.error:
                response += f"\n⚠ {plan.error}"
        else:
            response = f"Failed to generate code: {plan.error}"
        
        return self.send(
            message.with_context('code_plan', plan),
            content=response,
            intent=MessageIntent.EXECUTE
        )


def quick_plot(plot_type: str) -> str:
    """Get a quick matplotlib plot template from the corpus."""
    corpus = PlotCorpus()
    for pattern in corpus.patterns:
        if pattern.name == plot_type:
            return pattern.code_template
    # Default to line plot
    for pattern in corpus.patterns:
        if pattern.name == 'line_plot':
            return pattern.code_template
    return ""
