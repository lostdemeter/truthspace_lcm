"""
Chart Tool - Generate matplotlib visualizations.

Creates charts and graphs from data.
"""

import os
import tempfile
from typing import Optional
from .base import Tool, ToolResult


class ChartTool(Tool):
    """Tool for generating charts and visualizations."""
    
    @property
    def name(self) -> str:
        return "chart"
    
    @property
    def description(self) -> str:
        return "Generate matplotlib charts and visualizations (bar, line, pie, scatter)"
    
    @property
    def triggers(self) -> list[str]:
        return [
            "create a chart",
            "make a chart",
            "draw a chart",
            "plot",
            "graph",
            "visualize",
            "visualization",
            "bar chart",
            "line chart",
            "pie chart",
            "scatter plot",
            "histogram",
        ]
    
    def execute(self, query: str, data: Optional[dict] = None, **kwargs) -> ToolResult:
        """Generate a chart based on the query and data."""
        query_lower = query.lower()
        
        # Determine chart type
        chart_type = self._detect_chart_type(query_lower)
        
        # If no data provided, return instructions
        if data is None:
            return ToolResult.success_result(
                self._get_instructions(chart_type),
                chart_type=chart_type,
                needs_data=True,
            )
        
        # Generate the chart
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = data.get('labels', [])
            values = data.get('values', [])
            title = data.get('title', 'Chart')
            xlabel = data.get('xlabel', '')
            ylabel = data.get('ylabel', '')
            
            if chart_type == 'bar':
                ax.bar(labels, values, color='steelblue')
            elif chart_type == 'line':
                ax.plot(labels, values, marker='o', color='steelblue')
            elif chart_type == 'pie':
                ax.pie(values, labels=labels, autopct='%1.1f%%')
            elif chart_type == 'scatter':
                x = data.get('x', range(len(values)))
                ax.scatter(x, values, color='steelblue')
            elif chart_type == 'histogram':
                ax.hist(values, bins=data.get('bins', 10), color='steelblue')
            
            ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            # Save to temp file
            temp_dir = tempfile.gettempdir()
            filepath = os.path.join(temp_dir, f'geometric_lcm_chart_{chart_type}.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return ToolResult.success_result(
                f"Chart saved to: `{filepath}`\n\nI created a {chart_type} chart with title '{title}'.",
                chart_type=chart_type,
                filepath=filepath,
            )
            
        except ImportError:
            return ToolResult.error_result(
                "matplotlib is not installed. Install it with: pip install matplotlib"
            )
        except Exception as e:
            return ToolResult.error_result(f"Error creating chart: {str(e)}")
    
    def _detect_chart_type(self, query: str) -> str:
        """Detect the type of chart requested."""
        if 'pie' in query:
            return 'pie'
        elif 'line' in query:
            return 'line'
        elif 'scatter' in query:
            return 'scatter'
        elif 'histogram' in query or 'distribution' in query:
            return 'histogram'
        else:
            return 'bar'  # Default
    
    def _get_instructions(self, chart_type: str) -> str:
        """Get instructions for providing chart data."""
        return f"""I can create a **{chart_type} chart** for you! To generate it, I need:

**Required data:**
- `labels`: List of category names (e.g., ["Holmes", "Watson", "Moriarty"])
- `values`: List of corresponding values (e.g., [45, 30, 15])

**Optional:**
- `title`: Chart title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label

**Example request:**
"Create a bar chart of character appearances with labels Holmes, Watson, Moriarty and values 45, 30, 15"

What data would you like to visualize?"""
    
    def matches(self, query: str) -> float:
        """Check if query is asking for a chart."""
        query_lower = query.lower()
        
        for trigger in self.triggers:
            if trigger in query_lower:
                return 0.9
        
        return 0.0
