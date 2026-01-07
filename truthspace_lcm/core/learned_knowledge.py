"""
Learned Knowledge - Persistent storage for knowledge acquired through conversation.

This module provides auto-learning from LLM responses and persistent storage
separate from the bootstrap corpus. The goal is to grow the knowledge base
through use until conversations become fluid without LLM calls.

Key features:
- Separate storage from bootstrap (won't overwrite known-good knowledge)
- Auto-learning from LLM responses
- Topic-based deduplication (updates existing facts)
- Loads alongside bootstrap on startup

Storage location: ~/.truthspace/learned_knowledge.json

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# Default storage location
DEFAULT_LEARNED_PATH = Path.home() / ".truthspace" / "learned_knowledge.json"


@dataclass
class LearnedFact:
    """A fact learned from conversation."""
    topic: str
    content: str
    query: str  # The original query that triggered learning
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    use_count: int = 0  # How many times this fact has been retrieved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnedFact':
        """Create from dictionary."""
        return cls(**data)


class LearnedKnowledge:
    """
    Persistent storage for knowledge learned through conversation.
    
    Separate from bootstrap corpus to avoid overwriting known-good knowledge.
    Auto-learns from LLM responses and persists to disk.
    
    Usage:
        learned = LearnedKnowledge()
        
        # Learn from LLM response
        learned.learn("hypothermia", "Hypothermia is a medical condition...", 
                      query="what is hypothermia?")
        
        # Query
        fact = learned.get("hypothermia")
        
        # Forget
        learned.forget("hypothermia")
        
        # List all
        for fact in learned.all_facts():
            print(f"{fact.topic}: {fact.content[:50]}...")
    """
    
    def __init__(self, path: Optional[Path] = None):
        """
        Initialize learned knowledge store.
        
        Args:
            path: Path to JSON file. Defaults to ~/.truthspace/learned_knowledge.json
        """
        if path is None:
            self.path = DEFAULT_LEARNED_PATH
        elif isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path
        self._facts: Dict[str, LearnedFact] = {}
        self._load()
    
    def _load(self) -> None:
        """Load facts from disk."""
        if not self.path.exists():
            return
        
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            
            version = data.get("version", 1)
            facts_data = data.get("facts", [])
            
            for fact_dict in facts_data:
                fact = LearnedFact.from_dict(fact_dict)
                self._facts[fact.topic.lower()] = fact
                
        except Exception as e:
            # Don't fail on corrupt file - start fresh
            print(f"[WARN] Could not load learned knowledge: {e}")
            self._facts = {}
    
    def _save(self) -> None:
        """Save facts to disk."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": 1,
            "modified": datetime.now().isoformat(),
            "facts": [fact.to_dict() for fact in self._facts.values()]
        }
        
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def learn(self, topic: str, content: str, query: str) -> LearnedFact:
        """
        Learn a new fact or update existing one.
        
        Args:
            topic: The topic (e.g., "hypothermia")
            content: The knowledge content
            query: The original query that triggered learning
            
        Returns:
            The created or updated LearnedFact
        """
        topic_key = topic.lower()
        
        if topic_key in self._facts:
            # Update existing - keep the newer content
            existing = self._facts[topic_key]
            existing.content = content
            existing.timestamp = datetime.now().isoformat()
            fact = existing
        else:
            # Create new
            fact = LearnedFact(
                topic=topic,
                content=content,
                query=query
            )
            self._facts[topic_key] = fact
        
        self._save()
        return fact
    
    def get(self, topic: str) -> Optional[LearnedFact]:
        """
        Get a fact by topic.
        
        Increments use_count when retrieved.
        
        Args:
            topic: The topic to look up
            
        Returns:
            LearnedFact if found, None otherwise
        """
        topic_key = topic.lower()
        fact = self._facts.get(topic_key)
        
        if fact:
            fact.use_count += 1
            self._save()
        
        return fact
    
    def search(self, query: str) -> Optional[LearnedFact]:
        """
        Search for a fact that matches the query.
        
        Looks for topic words in the query.
        
        Args:
            query: The search query
            
        Returns:
            Best matching LearnedFact if found, None otherwise
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-zA-Z]+\b', query_lower))
        
        best_match = None
        best_score = 0
        
        for topic_key, fact in self._facts.items():
            # Check if topic appears in query
            topic_words = set(re.findall(r'\b[a-zA-Z]+\b', topic_key))
            
            # Score by word overlap
            overlap = len(topic_words & query_words)
            if overlap > best_score:
                best_score = overlap
                best_match = fact
        
        if best_match and best_score > 0:
            best_match.use_count += 1
            self._save()
            return best_match
        
        return None
    
    def forget(self, topic: str) -> bool:
        """
        Forget a learned fact.
        
        Args:
            topic: The topic to forget
            
        Returns:
            True if fact was found and removed, False otherwise
        """
        topic_key = topic.lower()
        
        if topic_key in self._facts:
            del self._facts[topic_key]
            self._save()
            return True
        
        return False
    
    def all_facts(self) -> List[LearnedFact]:
        """Get all learned facts."""
        return list(self._facts.values())
    
    def topics(self) -> List[str]:
        """Get all learned topics."""
        return [fact.topic for fact in self._facts.values()]
    
    def __len__(self) -> int:
        """Number of learned facts."""
        return len(self._facts)
    
    def __contains__(self, topic: str) -> bool:
        """Check if topic is learned."""
        return topic.lower() in self._facts
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about learned knowledge."""
        if not self._facts:
            return {
                "total_facts": 0,
                "total_uses": 0,
                "most_used": None,
            }
        
        total_uses = sum(f.use_count for f in self._facts.values())
        most_used = max(self._facts.values(), key=lambda f: f.use_count)
        
        return {
            "total_facts": len(self._facts),
            "total_uses": total_uses,
            "most_used": most_used.topic if most_used else None,
            "most_used_count": most_used.use_count if most_used else 0,
        }


def extract_llm_response(response: str) -> Optional[str]:
    """
    Extract the actual content from an LLM response.
    
    Handles common patterns like:
    - "Here's what I found:\n\n<content>"
    - "I don't have '<topic>' in my knowledge base. Let me look that up."
    - Direct content
    
    Args:
        response: The full response string
        
    Returns:
        Extracted content, or None if no content found
    """
    # Pattern 1: "Here's what I found:\n\n<content>"
    if "Here's what I found:" in response:
        parts = response.split("Here's what I found:", 1)
        if len(parts) > 1:
            content = parts[1].strip()
            if content:
                return content
    
    # Pattern 2: Direct content (no prefix)
    # Skip if it's a "looking that up" message
    if "Let me look that up" in response:
        return None
    if "I don't have" in response and "knowledge base" in response:
        return None
    
    # Direct content
    content = response.strip()
    if len(content) > 20:  # Minimum content length
        return content
    
    return None
