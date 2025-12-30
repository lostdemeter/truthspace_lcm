"""
Enrichment Gear

Adds derived fields and lookups to data records.

Author: Lesley Gushurst
License: GPLv3
"""

from typing import Dict, List, Any, Callable, Optional

from truthspace_lcm.gears.core import Gear, GearState


class EnrichmentGear(Gear):
    """
    Enriches data records with derived fields and lookups.
    
    Features:
    - Computed fields from existing data
    - Lookup tables for reference data
    - Concatenation of fields
    - Conditional enrichment
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("EnrichmentGear", ratio)
        self.computed_fields: Dict[str, Callable[[Dict], Any]] = {}
        self.lookup_tables: Dict[str, Dict[str, Any]] = {}
        self.lookup_mappings: Dict[str, tuple] = {}  # target_field -> (source_field, table_name)
        self.concatenations: List[tuple] = []  # (target_field, source_fields, separator)
    
    def add_computed(self, field: str, fn: Callable[[Dict], Any]) -> 'EnrichmentGear':
        """Add a computed field."""
        self.computed_fields[field] = fn
        return self
    
    def add_lookup_table(self, name: str, table: Dict[str, Any]) -> 'EnrichmentGear':
        """Add a lookup table."""
        self.lookup_tables[name] = table
        return self
    
    def add_lookup(self, target_field: str, source_field: str, 
                   table_name: str) -> 'EnrichmentGear':
        """Add a lookup mapping."""
        self.lookup_mappings[target_field] = (source_field, table_name)
        return self
    
    def add_concatenation(self, target_field: str, source_fields: List[str],
                          separator: str = " ") -> 'EnrichmentGear':
        """Add a field concatenation."""
        self.concatenations.append((target_field, source_fields, separator))
        return self
    
    def _enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single record."""
        enriched = record.copy()
        
        # Apply computed fields
        for field, fn in self.computed_fields.items():
            try:
                enriched[field] = fn(record)
            except Exception:
                enriched[field] = None
        
        # Apply lookups
        for target_field, (source_field, table_name) in self.lookup_mappings.items():
            if table_name in self.lookup_tables and source_field in record:
                key = record[source_field]
                table = self.lookup_tables[table_name]
                enriched[target_field] = table.get(key, None)
        
        # Apply concatenations
        for target_field, source_fields, separator in self.concatenations:
            values = [str(record.get(f, '')) for f in source_fields if record.get(f)]
            enriched[target_field] = separator.join(values)
        
        return enriched
    
    def forward(self, state: GearState) -> GearState:
        """Enrich all records in the state."""
        # Get records
        if hasattr(state, 'valid_records') and state.valid_records:
            records = state.valid_records
        elif hasattr(state, 'records'):
            records = state.records
        else:
            records = state.metadata.get('valid_records', state.metadata.get('records', []))
        
        # Enrich each record
        enriched = [self._enrich_record(r) for r in records]
        
        # Update state
        if hasattr(state, 'valid_records'):
            state.valid_records = enriched
            state.transformations_applied.append('EnrichmentGear')
        else:
            state.metadata['enriched_records'] = enriched
        
        return state
