"""
Practical Applications for the Gear Chain System

This module contains domain-specific gear implementations:
- nlp/: Natural language processing gears and applications (chat, API)
- data/: Data transformation gears and pipelines (ETL)
"""

# Import from submodules
from .nlp.chat import GearChat
from .nlp.api_server import create_app
from .data_pipeline import DataPipeline

__all__ = ['GearChat', 'create_app', 'DataPipeline']
