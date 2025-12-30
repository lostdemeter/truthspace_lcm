"""
Filter Gear

Filters records based on conditions.

Author: Lesley Gushurst
License: GPLv3
"""

from typing import Dict, List, Any, Callable, Optional

from truthspace_lcm.gears.core import Gear, GearState


class FilterGear(Gear):
    """
    Filters records based on conditions.
    
    Features:
    - Include/exclude by field values
    - Custom filter functions
    - Deduplication
    - Sampling
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("FilterGear", ratio)
        self.include_conditions: List[Callable[[Dict], bool]] = []
        self.exclude_conditions: List[Callable[[Dict], bool]] = []
        self.dedupe_fields: List[str] = []
        self.sample_size: Optional[int] = None
    
    def include_where(self, condition: Callable[[Dict], bool]) -> 'FilterGear':
        """Include records matching condition."""
        self.include_conditions.append(condition)
        return self
    
    def exclude_where(self, condition: Callable[[Dict], bool]) -> 'FilterGear':
        """Exclude records matching condition."""
        self.exclude_conditions.append(condition)
        return self
    
    def include_if_field_equals(self, field: str, value: Any) -> 'FilterGear':
        """Include records where field equals value."""
        return self.include_where(lambda r: r.get(field) == value)
    
    def exclude_if_field_equals(self, field: str, value: Any) -> 'FilterGear':
        """Exclude records where field equals value."""
        return self.exclude_where(lambda r: r.get(field) == value)
    
    def include_if_field_in(self, field: str, values: List[Any]) -> 'FilterGear':
        """Include records where field is in values."""
        return self.include_where(lambda r: r.get(field) in values)
    
    def exclude_if_field_in(self, field: str, values: List[Any]) -> 'FilterGear':
        """Exclude records where field is in values."""
        return self.exclude_where(lambda r: r.get(field) in values)
    
    def deduplicate(self, fields: List[str]) -> 'FilterGear':
        """Remove duplicates based on fields."""
        self.dedupe_fields = fields
        return self
    
    def sample(self, n: int) -> 'FilterGear':
        """Take a sample of n records."""
        self.sample_size = n
        return self
    
    def _passes_filters(self, record: Dict[str, Any]) -> bool:
        """Check if record passes all filters."""
        # Check include conditions (if any, record must match at least one)
        if self.include_conditions:
            if not any(cond(record) for cond in self.include_conditions):
                return False
        
        # Check exclude conditions (record must not match any)
        if any(cond(record) for cond in self.exclude_conditions):
            return False
        
        return True
    
    def _deduplicate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates based on dedupe_fields."""
        if not self.dedupe_fields:
            return records
        
        seen = set()
        deduped = []
        
        for record in records:
            key = tuple(record.get(f) for f in self.dedupe_fields)
            if key not in seen:
                seen.add(key)
                deduped.append(record)
        
        return deduped
    
    def forward(self, state: GearState) -> GearState:
        """Filter records in the state."""
        # Get records
        if hasattr(state, 'valid_records') and state.valid_records:
            records = state.valid_records
        elif hasattr(state, 'records'):
            records = state.records
        else:
            records = state.metadata.get('valid_records', state.metadata.get('records', []))
        
        # Apply filters
        filtered = [r for r in records if self._passes_filters(r)]
        
        # Deduplicate
        filtered = self._deduplicate(filtered)
        
        # Sample
        if self.sample_size and len(filtered) > self.sample_size:
            import random
            filtered = random.sample(filtered, self.sample_size)
        
        # Update state
        if hasattr(state, 'valid_records'):
            state.valid_records = filtered
            state.transformations_applied.append('FilterGear')
        else:
            state.metadata['filtered_records'] = filtered
        
        return state
