"""
Shared Extraction Utilities for TruthSpace LCM

Consolidates text extraction functions used across multiple components:
- Keyword extraction (with stop word filtering)
- URL extraction
- File path extraction
- Search term extraction
- Language type detection

This eliminates duplication across code_generator.py, bash_generator.py,
and knowledge_acquisition.py.
"""

import re
from typing import List, Optional, Tuple
from enum import Enum


# Comprehensive stop words list (merged from all components)
STOP_WORDS = frozenset({
    # Articles and determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    # Pronouns
    'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'it', 'its', 'we', 'our', 'they', 'their', 'who', 'whom', 'which', 'what',
    # Verbs (common/auxiliary)
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used',
    # Prepositions
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further',
    # Conjunctions
    'and', 'but', 'or', 'nor', 'so', 'yet', 'if', 'because', 'although',
    'though', 'while', 'until', 'unless',
    # Adverbs
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'very', 'just', 'only', 'also', 'too', 'now',
    # Quantifiers
    'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'any', 'every', 'both', 'many', 'much',
    # Common request words (not semantically meaningful)
    'please', 'want', 'like', 'get', 'make', 'show', 'me', 'using', 'use',
    'program', 'code', 'write', 'create',
    # Misc
    'am', 'than', 'same', 'own',
})


class LanguageType(Enum):
    """Detected language/execution type."""
    PYTHON = "python"
    BASH = "bash"
    UNKNOWN = "unknown"


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    Extract meaningful keywords from text, filtering stop words.
    
    Args:
        text: Input text to extract keywords from
        min_length: Minimum keyword length (default 2)
    
    Returns:
        List of lowercase keywords
    """
    # Tokenize: extract word-like tokens
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Filter stop words and short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) >= min_length]
    
    return keywords


def extract_url(text: str) -> Optional[str]:
    """
    Extract URL from text.
    
    Handles:
    - Full URLs (http://, https://)
    - Domain-like patterns (example.com)
    
    Args:
        text: Input text
    
    Returns:
        Extracted URL or None
    """
    # Full URL with protocol
    match = re.search(r'(https?://[^\s<>"\']+)', text)
    if match:
        return match.group(1).rstrip('.,;:)')
    
    # Domain-like pattern (add https://)
    match = re.search(r'\b([\w.-]+\.(?:com|org|net|io|gov|edu|co\.uk|dev|app)[^\s]*)', text)
    if match:
        url = match.group(1).rstrip('.,;:)')
        if not url.startswith('http'):
            url = 'https://' + url
        return url
    
    return None


def extract_path(text: str) -> Optional[str]:
    """
    Extract file or directory path from text.
    
    Handles:
    - Quoted paths ("path/to/file")
    - Named paths (called X, named X)
    - File references (of X, file X)
    - Paths with extensions
    - Directory paths with /
    
    Args:
        text: Input text
    
    Returns:
        Extracted path or None
    """
    # Quoted path (highest priority)
    match = re.search(r'["\']([^"\']+)["\']', text)
    if match:
        return match.group(1)
    
    # "called X" or "named X" pattern
    match = re.search(r'(?:called|named)\s+([/\w._-]+)', text)
    if match:
        return match.group(1)
    
    # "of X" or "file X" pattern (for file references)
    match = re.search(r'(?:of|file)\s+([\w._/-]+\.\w+)', text)
    if match:
        return match.group(1)
    
    # Filename with extension (but not URLs)
    if 'http' not in text.lower():
        match = re.search(r'\b([\w._-]+\.\w{1,5})\b', text)
        if match:
            return match.group(1)
    
    # Directory-like path (contains /)
    match = re.search(r'\b([/\w._-]+/[\w._-]+)\b', text)
    if match:
        return match.group(1)
    
    return None


def extract_filename(text: str) -> Optional[str]:
    """
    Extract filename (with extension) from text.
    
    Args:
        text: Input text
    
    Returns:
        Extracted filename or None
    """
    # Quoted filename
    match = re.search(r'["\']([^"\']+\.\w+)["\']', text)
    if match:
        return match.group(1)
    
    # Filename pattern (word.ext)
    match = re.search(r'\b([\w.-]+\.\w{2,4})\b', text)
    if match:
        return match.group(1)
    
    return None


def extract_search_term(text: str) -> Optional[str]:
    """
    Extract search term from text.
    
    Handles:
    - Quoted terms ("search term")
    - "for X" pattern
    - "containing X" pattern
    
    Args:
        text: Input text
    
    Returns:
        Extracted search term or None
    """
    # Quoted term
    match = re.search(r'["\']([^"\']+)["\']', text)
    if match:
        return match.group(1)
    
    # "for X" or "containing X" pattern
    match = re.search(r'(?:for|containing)\s+(\w+)', text)
    if match:
        return match.group(1)
    
    return None


def extract_numbers(text: str) -> List[int]:
    """
    Extract numbers from text.
    
    Args:
        text: Input text
    
    Returns:
        List of extracted integers
    """
    matches = re.findall(r'\b(\d+)\b', text)
    return [int(m) for m in matches]


def extract_range(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract a numeric range from text (e.g., "1 to 10", "5-20").
    
    Args:
        text: Input text
    
    Returns:
        Tuple of (start, end) or None
    """
    # "X to Y" pattern
    match = re.search(r'(\d+)\s+to\s+(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # "X-Y" pattern
    match = re.search(r'(\d+)\s*-\s*(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None


def detect_language_type(text: str) -> LanguageType:
    """
    Detect whether a request is more likely Python or Bash.
    
    Uses keyword indicators to determine the most appropriate
    execution environment.
    
    Args:
        text: Input text (natural language request)
    
    Returns:
        LanguageType enum value
    """
    text_lower = text.lower()
    
    # Strong Python indicators
    python_indicators = [
        'python', 'import', 'def ', 'class ', 'pip', 'module',
        'pandas', 'numpy', 'requests', 'flask', 'django',
        'json.', 'os.', 'sys.', '.py', 'beautifulsoup',
        'dataframe', 'dictionary', 'list comprehension',
    ]
    
    # Strong Bash indicators
    bash_indicators = [
        'bash', 'shell', 'terminal', 'command line',
        'mkdir', 'rmdir', 'chmod', 'chown', 'grep', 'sed', 'awk',
        'apt', 'yum', 'dnf', 'pacman', 'systemctl', 'journalctl',
        'ifconfig', 'ip addr', 'netstat', 'ping', 'curl', 'wget',
        'tar', 'gzip', 'zip', 'unzip', 'ssh', 'scp',
        'ps ', 'kill', 'top', 'htop', 'df ', 'du ',
        'directory', 'folder', 'permission',
    ]
    
    python_score = sum(1 for ind in python_indicators if ind in text_lower)
    bash_score = sum(1 for ind in bash_indicators if ind in text_lower)
    
    # File operations lean toward bash unless explicitly Python
    if any(word in text_lower for word in ['file', 'files', 'directory', 'folder']):
        if 'python' not in text_lower and 'import' not in text_lower:
            bash_score += 1
    
    # Web operations lean toward Python
    if any(word in text_lower for word in ['fetch', 'scrape', 'api', 'json', 'parse']):
        python_score += 1
    
    if python_score > bash_score:
        return LanguageType.PYTHON
    elif bash_score > python_score:
        return LanguageType.BASH
    else:
        return LanguageType.UNKNOWN


def normalize_command_name(text: str) -> str:
    """
    Normalize a command/function name for consistent storage.
    
    Args:
        text: Raw command name
    
    Returns:
        Normalized name (lowercase, underscores for spaces)
    """
    # Remove special characters, convert spaces to underscores
    normalized = re.sub(r'[^\w\s-]', '', text.lower())
    normalized = re.sub(r'[\s-]+', '_', normalized)
    return normalized.strip('_')


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("EXTRACTOR UTILITIES TEST")
    print("=" * 60)
    
    test_cases = [
        "create a directory called myproject",
        "fetch https://api.github.com and parse JSON",
        "show network interfaces using ifconfig",
        "read the file config.json",
        "search for 'error' in the logs",
        "list files from 1 to 10",
    ]
    
    for text in test_cases:
        print(f"\nInput: {text}")
        print(f"  Keywords: {extract_keywords(text)}")
        print(f"  URL: {extract_url(text)}")
        print(f"  Path: {extract_path(text)}")
        print(f"  Search: {extract_search_term(text)}")
        print(f"  Type: {detect_language_type(text).value}")
