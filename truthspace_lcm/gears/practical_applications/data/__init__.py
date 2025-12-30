"""
Data Transformation Gears

Domain-specific gears for data pipeline transformations (ETL, validation, etc.)

Gears:
- ValidationGear: Validates data types, ranges, required fields
- NormalizationGear: Standardizes formats (dates, currencies, units)
- EnrichmentGear: Adds derived fields and lookups
- FilterGear: Removes or flags bad records
- FormatGear: Outputs data in various formats

Usage:
    from truthspace_lcm.gears.data import ValidationGear, NormalizationGear, FormatGear
    from truthspace_lcm.gears.core import GearChain
    
    chain = GearChain("DataPipeline")
    chain.add(ValidationGear())
    chain.add(NormalizationGear())
    chain.add(FormatGear(format='json'))
    
    result = chain.process(data_state)
"""

from .validation import ValidationGear
from .normalization import NormalizationGear
from .enrichment import EnrichmentGear
from .filter import FilterGear
from .format import FormatGear
from .state import DataState

__all__ = [
    'ValidationGear',
    'NormalizationGear',
    'EnrichmentGear',
    'FilterGear',
    'FormatGear',
    'DataState',
]
