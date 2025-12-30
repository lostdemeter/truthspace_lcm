"""
Format Gear

Outputs data in various formats (JSON, CSV, dict).

Author: Lesley Gushurst
License: GPLv3
"""

import json
import csv
import io
from typing import Dict, List, Any, Optional

from truthspace_lcm.gears.core import Gear, GearState


class FormatGear(Gear):
    """
    Formats output data in various formats.
    
    Supported formats:
    - dict: Python dictionaries (default)
    - json: JSON string
    - csv: CSV string
    - summary: Summary statistics
    """
    
    def __init__(self, format: str = "dict", ratio: float = 1.0):
        super().__init__("FormatGear", ratio)
        self.format = format
        self.include_fields: Optional[List[str]] = None
        self.exclude_fields: Optional[List[str]] = None
    
    def set_format(self, format: str) -> 'FormatGear':
        """Set output format."""
        self.format = format
        return self
    
    def include_only(self, fields: List[str]) -> 'FormatGear':
        """Include only specified fields in output."""
        self.include_fields = fields
        return self
    
    def exclude(self, fields: List[str]) -> 'FormatGear':
        """Exclude specified fields from output."""
        self.exclude_fields = fields
        return self
    
    def _filter_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Filter record fields based on include/exclude lists."""
        if self.include_fields:
            return {k: v for k, v in record.items() if k in self.include_fields}
        elif self.exclude_fields:
            return {k: v for k, v in record.items() if k not in self.exclude_fields}
        return record
    
    def _to_json(self, records: List[Dict[str, Any]]) -> str:
        """Convert records to JSON string."""
        filtered = [self._filter_fields(r) for r in records]
        return json.dumps(filtered, indent=2, default=str)
    
    def _to_csv(self, records: List[Dict[str, Any]]) -> str:
        """Convert records to CSV string."""
        if not records:
            return ""
        
        filtered = [self._filter_fields(r) for r in records]
        
        # Get all field names
        fieldnames = list(filtered[0].keys())
        for record in filtered[1:]:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)
        
        return output.getvalue()
    
    def _to_summary(self, records: List[Dict[str, Any]], state: GearState) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'record_count': len(records),
            'fields': list(records[0].keys()) if records else [],
        }
        
        # Add validation info if available
        if hasattr(state, 'invalid_records'):
            summary['valid_count'] = len(state.valid_records)
            summary['invalid_count'] = len(state.invalid_records)
            summary['error_count'] = len(state.validation_errors)
        
        # Add transformations
        if hasattr(state, 'transformations_applied'):
            summary['transformations'] = state.transformations_applied
        
        # Add quality score
        summary['quality_score'] = state.accumulated_q.w
        
        return summary
    
    def forward(self, state: GearState) -> Any:
        """Format and return the final output."""
        # Get records
        if hasattr(state, 'valid_records') and state.valid_records:
            records = state.valid_records
        elif hasattr(state, 'records'):
            records = state.records
        else:
            records = state.metadata.get('valid_records', 
                      state.metadata.get('filtered_records',
                      state.metadata.get('records', [])))
        
        # Track transformation
        if hasattr(state, 'transformations_applied'):
            state.transformations_applied.append('FormatGear')
        
        # Format output
        if self.format == 'json':
            return self._to_json(records)
        elif self.format == 'csv':
            return self._to_csv(records)
        elif self.format == 'summary':
            return self._to_summary(records, state)
        else:  # dict
            return [self._filter_fields(r) for r in records]
