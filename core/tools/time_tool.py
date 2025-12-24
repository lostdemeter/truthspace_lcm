"""
Time Tool - Get current time and date information.

Provides current time, date, timezone, and date calculations.
"""

from datetime import datetime, timedelta
import time as time_module
from .base import Tool, ToolResult


class TimeTool(Tool):
    """Tool for retrieving current time and date information."""
    
    @property
    def name(self) -> str:
        return "time"
    
    @property
    def description(self) -> str:
        return "Get current time, date, timezone, or perform date calculations"
    
    @property
    def triggers(self) -> list[str]:
        return [
            "what time",
            "current time",
            "what date",
            "current date",
            "today's date",
            "what day",
            "what is the time",
            "what is the date",
            "tell me the time",
            "tell me the date",
            "what's the time",
            "what's the date",
            "time now",
            "date today",
            "what year",
            "what month",
        ]
    
    def execute(self, query: str, **kwargs) -> ToolResult:
        """Get time/date information based on the query."""
        query_lower = query.lower()
        now = datetime.now()
        
        # Full datetime
        if "time" in query_lower and "date" in query_lower:
            return ToolResult.success_result(
                f"The current date and time is {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}",
                datetime=now.isoformat(),
            )
        
        # Just time
        if "time" in query_lower:
            return ToolResult.success_result(
                f"The current time is {now.strftime('%I:%M:%S %p')}",
                time=now.strftime('%H:%M:%S'),
            )
        
        # Year
        if "year" in query_lower:
            return ToolResult.success_result(
                f"The current year is {now.year}",
                year=now.year,
            )
        
        # Month
        if "month" in query_lower:
            return ToolResult.success_result(
                f"The current month is {now.strftime('%B')} ({now.month})",
                month=now.month,
                month_name=now.strftime('%B'),
            )
        
        # Day of week
        if "day" in query_lower and ("week" in query_lower or "what day" in query_lower):
            return ToolResult.success_result(
                f"Today is {now.strftime('%A')}",
                day_of_week=now.strftime('%A'),
                day_number=now.weekday(),
            )
        
        # Default: date
        return ToolResult.success_result(
            f"Today's date is {now.strftime('%A, %B %d, %Y')}",
            date=now.strftime('%Y-%m-%d'),
            day_of_week=now.strftime('%A'),
        )
    
    def matches(self, query: str) -> float:
        """Check if query is asking about time/date."""
        query_lower = query.lower()
        
        # High confidence for explicit time/date questions
        for trigger in self.triggers:
            if trigger in query_lower:
                return 0.95
        
        # Medium confidence for related words
        time_words = ['time', 'date', 'today', 'now', 'clock', 'calendar']
        if any(word in query_lower for word in time_words):
            return 0.7
        
        return 0.0
