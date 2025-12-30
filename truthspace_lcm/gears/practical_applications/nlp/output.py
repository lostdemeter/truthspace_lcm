"""
Output Gear

Assembles the final output string from the gear state.

Author: Lesley Gushurst
License: GPLv3
"""

from truthspace_lcm.gears.core import Gear, GearState


class OutputGear(Gear):
    """
    Assembles the final output string.
    
    This gear takes the accumulated state and produces the final
    natural language output.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("OutputGear", ratio)
    
    def forward(self, state: GearState) -> str:
        """Assemble and return the final output string."""
        
        # Build prefix
        if state.use_prefix:
            if state.signal_prefix:
                prefix = f"{state.signal_prefix} {state.entity} is"
            else:
                prefix = f"It appears that {state.entity} is"
        else:
            prefix = f"{state.entity} is"
        
        # Article
        article = "an" if state.role[0].lower() in 'aeiou' else "a"
        
        # Build action string
        if state.actions:
            if len(state.actions) == 1:
                action_str = state.actions[0]
            elif len(state.actions) == 2:
                action_str = f"{state.actions[0]} and {state.actions[1]}"
            else:
                action_str = f"{state.actions[0]}, {state.actions[1]}, and {state.actions[2]}"
        else:
            action_str = ""
        
        # Build target string
        target_str = ' and '.join(state.targets[:2]) if state.targets else ""
        
        # Assemble
        connector = state.connector
        
        if action_str and target_str:
            return f"{prefix} {article} {state.role} {connector} {action_str}, {state.target_connector} {target_str}."
        elif action_str:
            return f"{prefix} {article} {state.role} {connector} {action_str}."
        elif target_str:
            return f"{prefix} {article} {state.role} {state.target_connector} {target_str}."
        else:
            return f"{prefix} {article} {state.role}."
