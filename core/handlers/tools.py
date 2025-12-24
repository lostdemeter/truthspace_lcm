"""
Tool Handler - Task execution, charts, and calculations.

Handles requests to execute tasks, create visualizations,
and perform calculations using the tool registry.
"""

from .base import Handler, HandlerResult, Context, Intent
from ..tools import ToolRegistry, TimeTool, CalculatorTool, ChartTool


class ToolHandler(Handler):
    """Handler for tool-based requests (execution, charts, calculations)."""
    
    def __init__(self, planner=None, qa=None):
        """
        Initialize with planner, QA, and tool registry.
        
        Args:
            planner: Planner instance for task execution
            qa: ConceptQA instance for data context
        """
        self.planner = planner
        self.qa = qa
        
        # Initialize tool registry
        self.registry = ToolRegistry()
        self.registry.register(TimeTool())
        self.registry.register(CalculatorTool())
        self.registry.register(ChartTool())
    
    @property
    def name(self) -> str:
        return "tools"
    
    @property
    def supported_intents(self) -> list[Intent]:
        return [Intent.EXECUTE, Intent.CHART]
    
    def can_handle(self, context: Context) -> float:
        """Check if this is a tool/execution request."""
        msg = context.message
        
        # Check if any registered tool matches
        tool_match = self.registry.find_best_tool(msg)
        if tool_match:
            tool, confidence = tool_match
            return confidence
        
        # Fall back to intent-based matching
        if context.intent not in [Intent.EXECUTE, Intent.CHART]:
            return 0.0
        
        msg_lower = msg.lower()
        
        # High confidence for execution patterns
        execute_patterns = [
            'calculate', 'compute', 'find the', 'sum of', 'average of',
            'sort', 'filter', 'count', 'run', 'execute'
        ]
        
        # High confidence for chart patterns
        chart_patterns = [
            'chart', 'graph', 'plot', 'visualize', 'visualization',
            'bar chart', 'line chart', 'pie chart', 'histogram', 'draw'
        ]
        
        if any(p in msg_lower for p in execute_patterns):
            return 0.9
        
        if any(p in msg_lower for p in chart_patterns):
            return 0.85
        
        return 0.3
    
    def handle(self, context: Context) -> HandlerResult:
        """Execute using the best matching tool."""
        msg = context.message
        
        # Try to find a matching tool
        tool_match = self.registry.find_best_tool(msg)
        if tool_match:
            tool, confidence = tool_match
            result = tool.execute(msg)
            
            if result.success:
                return HandlerResult.success_result(
                    result.output,
                    confidence=confidence,
                    tool=tool.name,
                    **result.metadata
                )
            else:
                return HandlerResult.failure_result(result.error or "Tool execution failed")
        
        # Fall back to planner for complex tasks
        if context.intent == Intent.CHART:
            return self._handle_chart(context)
        else:
            return self._handle_execute(context)
    
    def _handle_execute(self, context: Context) -> HandlerResult:
        """Handle task execution via planner."""
        if not self.planner:
            return HandlerResult.failure_result("Task planner not initialized")
        
        plan = self.planner.plan(context.message)
        result = self.planner.execute(plan)
        
        if result.success:
            steps_summary = "\n".join([f"  {i+1}. {s.description}" for i, s in enumerate(plan.steps)])
            response = f"""**Task:** {context.message}

**Plan:**
{steps_summary}

**Result:** `{result.final_result}`

The task completed successfully!"""
            return HandlerResult.success_result(
                response,
                confidence=0.95,
                result=result.final_result,
                steps=len(plan.steps)
            )
        else:
            # Find the failed step
            failed = [s for s in plan.steps if s.error]
            error_msg = failed[0].error if failed else "Unknown error"
            response = f"""I tried to execute that task but encountered an issue:

**Error:** {error_msg}

Could you rephrase the request or provide more details?"""
            return HandlerResult.failure_result(response)
    
    def _handle_chart(self, context: Context) -> HandlerResult:
        """Handle chart generation request."""
        # Use the chart tool
        chart_tool = self.registry.get("chart")
        if chart_tool:
            result = chart_tool.execute(context.message)
            if result.success:
                return HandlerResult.success_result(
                    result.output,
                    confidence=0.7,
                    **result.metadata
                )
        
        # Fallback instructions
        response = """I can help you create charts! To generate a visualization, I need:

1. **Data** - What data should I visualize?
2. **Chart type** - Bar, line, pie, scatter, etc.
3. **Labels** - Title, axis labels

For example, you could ask:
- "Create a bar chart of character appearances in Sherlock Holmes"
- "Plot the distribution of actions for Holmes"

What would you like to visualize?"""
        
        return HandlerResult.success_result(response, confidence=0.7, needs_clarification=True)
    
    def list_tools(self) -> list[dict]:
        """List all available tools."""
        return self.registry.list_tools()
