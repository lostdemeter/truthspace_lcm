"""
Corpus management for the Gear Chain system.

This module provides corpus loading, saving, and management utilities.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


# Default corpus paths (relative to this module)
CORPUS_DIR = Path(__file__).parent
DEFAULT_CORPUS = CORPUS_DIR / "corpus_experimental.json"
SIGNAL_CORPUS = CORPUS_DIR / "corpus_signal_full.json"


def get_corpus_path(name: str = "experimental") -> Path:
    """Get the path to a corpus file."""
    if name == "experimental":
        return DEFAULT_CORPUS
    elif name == "signal":
        return SIGNAL_CORPUS
    else:
        return CORPUS_DIR / f"corpus_{name}.json"


def load_corpus(name: str = "experimental") -> Dict[str, Any]:
    """Load a corpus by name."""
    path = get_corpus_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_corpus(data: Dict[str, Any], name: str = "experimental", 
                backup: bool = True) -> Path:
    """
    Save a corpus.
    
    Args:
        data: Corpus data to save
        name: Corpus name
        backup: If True, create a backup before overwriting
        
    Returns:
        Path to saved corpus
    """
    path = get_corpus_path(name)
    
    if backup and path.exists():
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".backup_{timestamp}.json")
        with open(path, 'r') as f:
            backup_data = f.read()
        with open(backup_path, 'w') as f:
            f.write(backup_data)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return path


def list_corpuses() -> List[str]:
    """List available corpuses."""
    corpuses = []
    for f in CORPUS_DIR.glob("corpus_*.json"):
        name = f.stem.replace("corpus_", "")
        if not name.startswith("backup"):
            corpuses.append(name)
    return corpuses


def get_corpus_stats(name: str = "experimental") -> Dict[str, Any]:
    """Get statistics about a corpus."""
    data = load_corpus(name)
    frames = data.get('frames', [])
    
    return {
        'name': name,
        'frame_count': len(frames),
        'total_chars': sum(len(f.get('text', '')) for f in frames),
        'avg_frame_length': sum(len(f.get('text', '')) for f in frames) / len(frames) if frames else 0,
    }


__all__ = [
    'load_corpus',
    'save_corpus', 
    'get_corpus_path',
    'list_corpuses',
    'get_corpus_stats',
    'CORPUS_DIR',
    'DEFAULT_CORPUS',
    'SIGNAL_CORPUS',
]
