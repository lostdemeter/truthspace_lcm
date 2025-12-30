"""
Data Transformation Pipeline Application

Demonstrates the gear chain system for data ETL tasks.

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from truthspace_lcm.gears.core import GearChain
from truthspace_lcm.gears.practical_applications.data import (
    DataState,
    ValidationGear,
    NormalizationGear,
    EnrichmentGear,
    FilterGear,
    FormatGear,
)


class DataPipeline:
    """
    A data transformation pipeline using the gear chain system.
    
    Demonstrates that the same gear architecture used for NLP/chat
    can be applied to structured data transformation.
    """
    
    def __init__(self):
        self.chain = GearChain("DataPipeline")
    
    def add_validation(self) -> ValidationGear:
        """Add and return a validation gear."""
        gear = ValidationGear()
        self.chain.add(gear)
        return gear
    
    def add_normalization(self) -> NormalizationGear:
        """Add and return a normalization gear."""
        gear = NormalizationGear()
        self.chain.add(gear)
        return gear
    
    def add_enrichment(self) -> EnrichmentGear:
        """Add and return an enrichment gear."""
        gear = EnrichmentGear()
        self.chain.add(gear)
        return gear
    
    def add_filter(self) -> FilterGear:
        """Add and return a filter gear."""
        gear = FilterGear()
        self.chain.add(gear)
        return gear
    
    def add_format(self, format: str = "dict") -> FormatGear:
        """Add and return a format gear."""
        gear = FormatGear(format=format)
        self.chain.add(gear)
        return gear
    
    def process(self, records: List[Dict[str, Any]]) -> Any:
        """Process records through the pipeline."""
        state = DataState()
        state.add_records(records)
        return self.chain.process(state)
    
    def __repr__(self) -> str:
        return f"DataPipeline: {self.chain}"


def demo():
    """Demonstrate the data transformation pipeline."""
    print("=" * 70)
    print("DATA TRANSFORMATION PIPELINE DEMO")
    print("Using the same gear architecture as NLP/chat")
    print("=" * 70)
    
    # Sample data with some issues
    raw_data = [
        {"name": "  John Smith  ", "email": "john@example.com", "age": "25", "country": "US"},
        {"name": "jane doe", "email": "jane@example.com", "age": "30", "country": "UK"},
        {"name": "Bob Wilson", "email": "invalid-email", "age": "150", "country": "US"},
        {"name": "", "email": "empty@example.com", "age": "20", "country": "CA"},
        {"name": "Alice Brown", "email": "alice@example.com", "age": "28", "country": "US"},
        {"name": "Charlie Davis", "email": "charlie@example.com", "age": "35", "country": "UK"},
        {"name": "Eve Johnson", "email": "eve@example.com", "age": "-5", "country": "AU"},
    ]
    
    print(f"\nInput: {len(raw_data)} records")
    print("-" * 40)
    
    # Build pipeline
    pipeline = DataPipeline()
    
    # 1. Validation
    validation = pipeline.add_validation()
    validation.add_required("name", "Name is required")
    validation.add_required("email", "Email is required")
    validation.add_type("age", "str")  # Will be normalized to int later
    validation.add_pattern("email", r'^[\w\.-]+@[\w\.-]+\.\w+$', "Invalid email format")
    validation.add_range("age", min_val=0, max_val=120, message="Age must be 0-120")
    
    # 2. Normalization
    normalization = pipeline.add_normalization()
    normalization.trim("name")
    normalization.titlecase("name")
    normalization.lowercase("email")
    normalization.to_int("age")
    normalization.uppercase("country")
    
    # 3. Enrichment
    enrichment = pipeline.add_enrichment()
    
    # Add country name lookup
    enrichment.add_lookup_table("countries", {
        "US": "United States",
        "UK": "United Kingdom",
        "CA": "Canada",
        "AU": "Australia",
    })
    enrichment.add_lookup("country_name", "country", "countries")
    
    # Add computed field
    enrichment.add_computed("age_group", lambda r: 
        "young" if r.get("age", 0) < 30 else "adult" if r.get("age", 0) < 50 else "senior"
    )
    
    # 4. Filter
    filter_gear = pipeline.add_filter()
    filter_gear.include_if_field_in("country", ["US", "UK"])
    
    # 5. Format
    format_gear = pipeline.add_format("dict")
    
    print(f"\nPipeline: {pipeline.chain}")
    print("-" * 40)
    
    # Process
    result = pipeline.process(raw_data)
    
    print(f"\nOutput: {len(result)} records")
    print("-" * 40)
    
    for record in result:
        print(f"  {record}")
    
    # Show summary
    print("\n" + "=" * 70)
    print("SUMMARY FORMAT")
    print("=" * 70)
    
    # Create a new pipeline for summary
    summary_pipeline = DataPipeline()
    summary_pipeline.add_validation().add_required("name").add_required("email")
    summary_pipeline.add_format("summary")
    
    summary = summary_pipeline.process(raw_data)
    print(f"\n{summary}")
    
    # Show JSON format
    print("\n" + "=" * 70)
    print("JSON FORMAT")
    print("=" * 70)
    
    json_pipeline = DataPipeline()
    json_pipeline.add_validation().add_required("name")
    json_pipeline.add_normalization().trim("name").titlecase("name")
    json_pipeline.add_format("json").include_only(["name", "email", "age"])
    
    json_output = json_pipeline.process(raw_data)
    print(f"\n{json_output}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("Same gear architecture, different domain!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
