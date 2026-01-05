"""
Bootstrap Knowledge for ChatPipeline

Loads initial knowledge from JSON file so the chat can respond
to basic queries without needing LLM knowledge acquisition.

Author: Lesley Gushurst
License: GPLv3
"""

import json
from pathlib import Path
from typing import List, Dict, Any


# Path to bootstrap knowledge JSON
BOOTSTRAP_KNOWLEDGE_PATH = Path(__file__).parent.parent / "corpus" / "bootstrap_knowledge.json"


def get_bootstrap_knowledge() -> List[Dict[str, Any]]:
    """
    Load bootstrap knowledge items from JSON file.
    
    Each item has:
    - text: The knowledge content
    - topic: The topic category
    - keywords: Words that should trigger this knowledge
    """
    if not BOOTSTRAP_KNOWLEDGE_PATH.exists():
        return []
    
    try:
        with open(BOOTSTRAP_KNOWLEDGE_PATH, 'r') as f:
            data = json.load(f)
        return data.get("knowledge", [])
    except Exception:
        return []


def get_bootstrap_synonyms() -> List[List[str]]:
    """Load synonym groups from JSON file."""
    if not BOOTSTRAP_KNOWLEDGE_PATH.exists():
        return []
    
    try:
        with open(BOOTSTRAP_KNOWLEDGE_PATH, 'r') as f:
            data = json.load(f)
        return data.get("synonyms", [])
    except Exception:
        return []
