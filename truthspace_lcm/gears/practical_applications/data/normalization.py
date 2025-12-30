"""
Normalization Gear

Standardizes data formats (dates, currencies, units, strings).

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime

from truthspace_lcm.gears.core import Gear, GearState


class NormalizationGear(Gear):
    """
    Normalizes data to standard formats.
    
    Transformations:
    - String: trim, lowercase, uppercase, titlecase
    - Dates: parse various formats to ISO
    - Numbers: parse strings to numbers, round decimals
    - Currency: standardize currency formats
    - Custom: user-defined transformations
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("NormalizationGear", ratio)
        self.normalizers: Dict[str, Callable[[Any], Any]] = {}
        self.field_normalizers: Dict[str, str] = {}  # field -> normalizer name
    
    def add_normalizer(self, name: str, fn: Callable[[Any], Any]) -> 'NormalizationGear':
        """Add a custom normalizer function."""
        self.normalizers[name] = fn
        return self
    
    def normalize_field(self, field: str, normalizer: str) -> 'NormalizationGear':
        """Set normalizer for a specific field."""
        self.field_normalizers[field] = normalizer
        return self
    
    def trim(self, field: str) -> 'NormalizationGear':
        """Trim whitespace from field."""
        return self.normalize_field(field, 'trim')
    
    def lowercase(self, field: str) -> 'NormalizationGear':
        """Convert field to lowercase."""
        return self.normalize_field(field, 'lowercase')
    
    def uppercase(self, field: str) -> 'NormalizationGear':
        """Convert field to uppercase."""
        return self.normalize_field(field, 'uppercase')
    
    def titlecase(self, field: str) -> 'NormalizationGear':
        """Convert field to title case."""
        return self.normalize_field(field, 'titlecase')
    
    def to_int(self, field: str) -> 'NormalizationGear':
        """Convert field to integer."""
        return self.normalize_field(field, 'to_int')
    
    def to_float(self, field: str) -> 'NormalizationGear':
        """Convert field to float."""
        return self.normalize_field(field, 'to_float')
    
    def to_date(self, field: str) -> 'NormalizationGear':
        """Parse field as date to ISO format."""
        return self.normalize_field(field, 'to_date')
    
    def round_decimal(self, field: str, places: int = 2) -> 'NormalizationGear':
        """Round field to decimal places."""
        self.normalizers[f'round_{field}'] = lambda x: round(float(x), places) if x else x
        return self.normalize_field(field, f'round_{field}')
    
    def _apply_normalizer(self, value: Any, normalizer: str) -> Any:
        """Apply a normalizer to a value."""
        if value is None:
            return None
        
        # Built-in normalizers
        if normalizer == 'trim':
            return str(value).strip() if value else value
        elif normalizer == 'lowercase':
            return str(value).lower() if value else value
        elif normalizer == 'uppercase':
            return str(value).upper() if value else value
        elif normalizer == 'titlecase':
            return str(value).title() if value else value
        elif normalizer == 'to_int':
            try:
                return int(float(str(value).replace(',', '')))
            except (ValueError, TypeError):
                return value
        elif normalizer == 'to_float':
            try:
                return float(str(value).replace(',', ''))
            except (ValueError, TypeError):
                return value
        elif normalizer == 'to_date':
            return self._parse_date(value)
        
        # Custom normalizers
        if normalizer in self.normalizers:
            return self.normalizers[normalizer](value)
        
        return value
    
    def _parse_date(self, value: Any) -> str:
        """Parse various date formats to ISO."""
        if not value:
            return value
        
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d %b %Y',
            '%d %B %Y',
            '%b %d, %Y',
            '%B %d, %Y',
        ]
        
        value_str = str(value)
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(value_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return value_str
    
    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single record."""
        normalized = record.copy()
        
        for field, normalizer in self.field_normalizers.items():
            if field in normalized:
                normalized[field] = self._apply_normalizer(normalized[field], normalizer)
        
        return normalized
    
    def forward(self, state: GearState) -> GearState:
        """Normalize all records in the state."""
        # Get records - prefer valid_records if available (post-validation)
        if hasattr(state, 'valid_records') and state.valid_records:
            records = state.valid_records
        elif hasattr(state, 'records'):
            records = state.records
        else:
            records = state.metadata.get('valid_records', state.metadata.get('records', []))
        
        # Normalize each record
        normalized = [self._normalize_record(r) for r in records]
        
        # Update state
        if hasattr(state, 'valid_records'):
            state.valid_records = normalized
            state.transformations_applied.append('NormalizationGear')
        else:
            state.metadata['normalized_records'] = normalized
        
        return state
