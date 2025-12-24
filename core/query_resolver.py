"""
Query Resolver - Handles compound queries and coreference resolution.

This module provides:
1. Query splitting - Break compound queries into sub-queries
2. Coreference resolution - Resolve pronouns to their referents
3. Query combination - Merge results from multiple sub-queries

Examples:
- "Who is Holmes and what time is it?" → ["Who is Holmes?", "What time is it?"]
- "Who is Darcy and how did he meet Elizabeth?" → ["Who is Darcy?", "How did Darcy meet Elizabeth?"]
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ConjunctionType(Enum):
    """Types of conjunctions that join sub-queries."""
    AND = "and"
    OR = "or"
    BUT = "but"
    THEN = "then"
    ALSO = "also"


@dataclass
class SubQuery:
    """A single sub-query extracted from a compound query."""
    text: str
    original_position: int  # Position in the original compound query
    resolved_text: Optional[str] = None  # After coreference resolution
    entities_mentioned: list[str] = field(default_factory=list)
    
    @property
    def final_text(self) -> str:
        """Get the resolved text or original if not resolved."""
        return self.resolved_text or self.text


@dataclass
class ResolvedQuery:
    """A compound query that has been split and resolved."""
    original: str
    sub_queries: list[SubQuery]
    is_compound: bool
    entities: list[str] = field(default_factory=list)  # All entities mentioned
    
    def __len__(self) -> int:
        return len(self.sub_queries)


class QuerySplitter:
    """
    Splits compound queries into sub-queries.
    
    Handles patterns like:
    - "X and Y" → [X, Y]
    - "X, Y, and Z" → [X, Y, Z]
    - "X? Also, Y?" → [X, Y]
    """
    
    def __init__(self):
        # Conjunctions that typically join independent clauses
        self.conjunctions = ['and', 'but', 'or', 'also', 'then', 'additionally']
        
        # Question words that indicate a new question
        self.question_starters = [
            'who', 'what', 'where', 'when', 'why', 'how',
            'is', 'are', 'does', 'did', 'can', 'could', 'would', 'will'
        ]
    
    def split(self, query: str) -> list[str]:
        """
        Split a compound query into sub-queries.
        
        Returns a list of sub-query strings.
        """
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Try different splitting strategies
        sub_queries = self._split_by_conjunction_question(query)
        
        if len(sub_queries) == 1:
            # Try splitting by punctuation
            sub_queries = self._split_by_punctuation(query)
        
        # Clean up each sub-query
        return [self._clean_subquery(sq) for sq in sub_queries if sq.strip()]
    
    def _split_by_conjunction_question(self, query: str) -> list[str]:
        """
        Split on conjunctions followed by question words.
        
        "Who is Holmes and what time is it?" → ["Who is Holmes", "what time is it?"]
        """
        # Pattern: conjunction + optional comma + question word
        pattern = r'\s+(?:and|but|or|also|then)\s+(?:,\s*)?(' + '|'.join(self.question_starters) + r')\b'
        
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            return [query]
        
        # Reconstruct: parts alternate between text and captured question words
        result = []
        current = parts[0]
        
        for i in range(1, len(parts), 2):
            if i < len(parts):
                result.append(current.strip())
                # Next part starts with the captured question word
                question_word = parts[i] if i < len(parts) else ''
                remaining = parts[i + 1] if i + 1 < len(parts) else ''
                current = question_word + remaining
        
        if current.strip():
            result.append(current.strip())
        
        return result if len(result) > 1 else [query]
    
    def _split_by_punctuation(self, query: str) -> list[str]:
        """
        Split on sentence-ending punctuation followed by new sentences.
        
        "Who is Holmes? What time is it?" → ["Who is Holmes?", "What time is it?"]
        """
        # Split on ? or . followed by space and capital letter or question word
        pattern = r'([.?!])\s+(?=[A-Z]|' + '|'.join(self.question_starters) + r')'
        
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            return [query]
        
        # Reconstruct with punctuation
        result = []
        current = ''
        for i, part in enumerate(parts):
            if part in '.?!':
                current += part
                result.append(current.strip())
                current = ''
            else:
                current += part
        
        if current.strip():
            result.append(current.strip())
        
        return result if len(result) > 1 else [query]
    
    def _clean_subquery(self, query: str) -> str:
        """Clean up a sub-query."""
        query = query.strip()
        
        # Remove leading conjunctions
        for conj in self.conjunctions:
            if query.lower().startswith(conj + ' '):
                query = query[len(conj):].strip()
        
        # Ensure it ends with appropriate punctuation
        if query and query[-1] not in '.?!':
            # Add ? if it looks like a question
            if any(query.lower().startswith(q) for q in self.question_starters):
                query += '?'
        
        return query


class CoreferenceResolver:
    """
    Resolves pronouns and references to their antecedents.
    
    Uses simple heuristics based on:
    1. Most recently mentioned entity
    2. Gender matching (he/she/they)
    3. Entity type matching (it for things, who for people)
    """
    
    def __init__(self):
        # Pronouns and their properties
        self.pronouns = {
            # Subject pronouns
            'he': {'gender': 'male', 'type': 'person', 'number': 'singular'},
            'she': {'gender': 'female', 'type': 'person', 'number': 'singular'},
            'it': {'gender': 'neutral', 'type': 'thing', 'number': 'singular'},
            'they': {'gender': 'neutral', 'type': 'any', 'number': 'plural'},
            
            # Object pronouns
            'him': {'gender': 'male', 'type': 'person', 'number': 'singular'},
            'her': {'gender': 'female', 'type': 'person', 'number': 'singular'},
            'them': {'gender': 'neutral', 'type': 'any', 'number': 'plural'},
            
            # Possessive pronouns
            'his': {'gender': 'male', 'type': 'person', 'number': 'singular'},
            'hers': {'gender': 'female', 'type': 'person', 'number': 'singular'},
            'its': {'gender': 'neutral', 'type': 'thing', 'number': 'singular'},
            'their': {'gender': 'neutral', 'type': 'any', 'number': 'plural'},
        }
        
        # Known entities with gender (from knowledge base)
        self.entity_gender = {
            # Sherlock Holmes characters
            'holmes': 'male',
            'sherlock': 'male',
            'watson': 'male',
            'john': 'male',
            'moriarty': 'male',
            'mycroft': 'male',
            'lestrade': 'male',
            'irene': 'female',
            'adler': 'female',
            'mrs hudson': 'female',
            'hudson': 'female',
            
            # Pride and Prejudice characters
            'darcy': 'male',
            'elizabeth': 'female',
            'bennet': 'female',  # Usually refers to Elizabeth
            'jane': 'female',
            'bingley': 'male',
            'wickham': 'male',
            'lydia': 'female',
            'mr bennet': 'male',
            'mrs bennet': 'female',
        }
    
    def resolve(self, sub_queries: list[str], context: Optional[list[str]] = None) -> list[str]:
        """
        Resolve pronouns in sub-queries using context.
        
        Args:
            sub_queries: List of sub-query strings
            context: Optional conversation history for additional context
            
        Returns:
            List of resolved sub-query strings
        """
        resolved = []
        entities_mentioned = []  # Track entities as we process
        
        for query in sub_queries:
            # Extract entities from this query
            new_entities = self._extract_entities(query)
            
            # Resolve pronouns using accumulated entities
            resolved_query = self._resolve_pronouns(query, entities_mentioned)
            resolved.append(resolved_query)
            
            # Add new entities to our tracking list
            entities_mentioned.extend(new_entities)
        
        return resolved
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Check for known entities
        for entity in self.entity_gender.keys():
            if entity in text_lower:
                entities.append(entity)
        
        # Also look for capitalized words that might be names
        words = text.split()
        for word in words:
            clean_word = word.strip('.,?!').lower()
            if word[0].isupper() and clean_word not in ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'the', 'a', 'an']:
                if clean_word not in entities:
                    entities.append(clean_word)
        
        return entities
    
    def _resolve_pronouns(self, query: str, context_entities: list[str]) -> str:
        """Replace pronouns with their likely referents."""
        if not context_entities:
            return query
        
        result = query
        
        # Find pronouns in the query
        for pronoun, props in self.pronouns.items():
            # Check if pronoun is in query (as whole word)
            pattern = r'\b' + pronoun + r'\b'
            if re.search(pattern, result, re.IGNORECASE):
                # Find best matching entity
                referent = self._find_referent(pronoun, props, context_entities)
                if referent:
                    # Replace pronoun with referent
                    result = re.sub(pattern, referent.title(), result, flags=re.IGNORECASE)
        
        return result
    
    def _find_referent(self, pronoun: str, props: dict, entities: list[str]) -> Optional[str]:
        """Find the best referent for a pronoun."""
        if not entities:
            return None
        
        gender = props.get('gender')
        
        # Filter by gender if applicable
        candidates = []
        for entity in reversed(entities):  # Most recent first
            entity_lower = entity.lower()
            entity_gender = self.entity_gender.get(entity_lower)
            
            if gender == 'neutral' or entity_gender is None:
                candidates.append(entity)
            elif entity_gender == gender:
                candidates.append(entity)
        
        # Return most recent matching entity
        return candidates[0] if candidates else entities[-1]


class QueryResolver:
    """
    Main class for resolving compound queries.
    
    Combines splitting and coreference resolution.
    """
    
    def __init__(self):
        self.splitter = QuerySplitter()
        self.coreference = CoreferenceResolver()
    
    def resolve(self, query: str, context: Optional[list[str]] = None) -> ResolvedQuery:
        """
        Resolve a potentially compound query.
        
        Args:
            query: The user's query (possibly compound)
            context: Optional conversation history
            
        Returns:
            ResolvedQuery with split and resolved sub-queries
        """
        # Split the query
        sub_query_texts = self.splitter.split(query)
        
        # Resolve coreferences
        resolved_texts = self.coreference.resolve(sub_query_texts, context)
        
        # Build SubQuery objects
        sub_queries = []
        all_entities = []
        
        for i, (original, resolved) in enumerate(zip(sub_query_texts, resolved_texts)):
            entities = self.coreference._extract_entities(original)
            all_entities.extend(entities)
            
            sub_queries.append(SubQuery(
                text=original,
                original_position=i,
                resolved_text=resolved if resolved != original else None,
                entities_mentioned=entities,
            ))
        
        return ResolvedQuery(
            original=query,
            sub_queries=sub_queries,
            is_compound=len(sub_queries) > 1,
            entities=list(dict.fromkeys(all_entities)),  # Dedupe while preserving order
        )
    
    def is_compound(self, query: str) -> bool:
        """Quick check if a query appears to be compound."""
        sub_queries = self.splitter.split(query)
        return len(sub_queries) > 1


# Global instance
_resolver: Optional[QueryResolver] = None

def get_query_resolver() -> QueryResolver:
    """Get the global query resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = QueryResolver()
    return _resolver
