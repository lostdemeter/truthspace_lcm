"""
Data State

A specialized GearState for data transformation pipelines.

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from truthspace_lcm.gears.core import GearState, Quaternion


@dataclass
class DataState(GearState):
    """
    State object for data transformation pipelines.
    
    Extends GearState with data-specific fields for records,
    validation results, and transformation metadata.
    """
    # Data records
    records: List[Dict[str, Any]] = field(default_factory=list)
    
    # Schema definition
    schema: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    valid_records: List[Dict[str, Any]] = field(default_factory=list)
    invalid_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Transformation tracking
    transformations_applied: List[str] = field(default_factory=list)
    
    # Output format
    output_format: str = "dict"  # dict, json, csv
    
    # Quality metrics (stored in quaternion)
    # w: overall quality score (0-1)
    # x: completeness (0-1)
    # y: accuracy (0-1)
    # z: consistency (0-1)
    
    def add_record(self, record: Dict[str, Any]) -> 'DataState':
        """Add a record to process."""
        self.records.append(record)
        return self
    
    def add_records(self, records: List[Dict[str, Any]]) -> 'DataState':
        """Add multiple records."""
        self.records.extend(records)
        return self
    
    def set_schema(self, schema: Dict[str, Any]) -> 'DataState':
        """Set the data schema."""
        self.schema = schema
        return self
    
    def get_quality_score(self) -> float:
        """Get overall quality score from quaternion."""
        return self.accumulated_q.w
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the data state."""
        return {
            'total_records': len(self.records),
            'valid_records': len(self.valid_records),
            'invalid_records': len(self.invalid_records),
            'validation_errors': len(self.validation_errors),
            'transformations': self.transformations_applied,
            'quality_score': self.get_quality_score(),
        }
