"""
Experiment: Python Token Vocabulary (Design 113 - Experiment 2)

This experiment positions Python tokens in semantic space, enabling
token-level code generation via geometric traversal.

Key Insight: Python tokens have semantic properties that can be
encoded as positions in a multi-dimensional space:
- category: keyword, operator, literal, identifier, punctuation
- role: control flow, data, definition, expression
- side_effect: pure vs impure
- arity: how many operands/arguments

The goal is to show that:
1. Tokens cluster by semantic similarity
2. Valid token sequences form paths through the space
3. Code can emerge from geometric traversal

This builds on Experiment 1 (multi-line) but operates at the TOKEN
level rather than the LINE level.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from enum import Enum, auto


PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# TOKEN CATEGORIES
# =============================================================================

class TokenCategory(Enum):
    """Categories of Python tokens."""
    KEYWORD = 0       # if, for, def, class, return, etc.
    OPERATOR = 1      # +, -, *, /, =, ==, etc.
    LITERAL = 2       # True, False, None, numbers, strings
    IDENTIFIER = 3    # Variable names, function names
    PUNCTUATION = 4   # (, ), [, ], {, }, :, ,
    BUILTIN = 5       # print, len, range, open, etc.


class TokenRole(Enum):
    """Semantic role of a token."""
    CONTROL = 0       # Control flow: if, for, while, return
    DATA = 1          # Data: literals, identifiers
    DEFINITION = 2    # Definitions: def, class, =
    EXPRESSION = 3    # Expression operators: +, -, *, /
    STRUCTURE = 4     # Structure: (, ), :, ,


# =============================================================================
# TOKEN VOCABULARY
# =============================================================================

@dataclass
class TokenSeed:
    """A token with its semantic position."""
    text: str
    position: np.ndarray
    category: TokenCategory
    role: TokenRole
    can_follow: Set[str]  # Token categories/roles that can follow
    
    def distance_to(self, other: np.ndarray) -> float:
        return np.linalg.norm(self.position - other)


class PythonTokenVocabulary:
    """
    Vocabulary of Python tokens positioned in semantic space.
    
    Dimensions:
    - [0] category: 0=keyword, 1=operator, 2=literal, 3=identifier, 4=punct, 5=builtin
    - [1] role: 0=control, 1=data, 2=definition, 3=expression, 4=structure
    - [2] side_effect: -1=pure, 0=neutral, +1=impure (I/O, mutation)
    - [3] arity: 0=nullary, 1=unary, 2=binary, 3+=variadic
    """
    
    def __init__(self, dims: int = 4):
        self.dims = dims
        self._tokens: Dict[str, TokenSeed] = {}
        self._by_category: Dict[TokenCategory, List[str]] = defaultdict(list)
        self._by_role: Dict[TokenRole, List[str]] = defaultdict(list)
    
    def add_token(self, text: str, position: np.ndarray, 
                  category: TokenCategory, role: TokenRole,
                  can_follow: Optional[Set[str]] = None):
        """Add a token to the vocabulary."""
        can_follow = can_follow or set()
        self._tokens[text] = TokenSeed(
            text=text, 
            position=position, 
            category=category, 
            role=role,
            can_follow=can_follow
        )
        self._by_category[category].append(text)
        self._by_role[role].append(text)
    
    def get_position(self, text: str) -> Optional[np.ndarray]:
        """Get a token's position."""
        seed = self._tokens.get(text)
        return seed.position.copy() if seed else None
    
    def get_token(self, text: str) -> Optional[TokenSeed]:
        """Get a token seed."""
        return self._tokens.get(text)
    
    def find_nearest(self, position: np.ndarray,
                     allowed_categories: Optional[Set[TokenCategory]] = None,
                     allowed_roles: Optional[Set[TokenRole]] = None,
                     exclude: Optional[Set[str]] = None) -> Optional[TokenSeed]:
        """
        Find the nearest token to a position.
        
        Args:
            position: Target position
            allowed_categories: Filter by category
            allowed_roles: Filter by role
            exclude: Tokens to exclude
        """
        exclude = exclude or set()
        
        best_token = None
        best_distance = float('inf')
        
        for text, seed in self._tokens.items():
            if text in exclude:
                continue
            if allowed_categories and seed.category not in allowed_categories:
                continue
            if allowed_roles and seed.role not in allowed_roles:
                continue
            
            dist = seed.distance_to(position)
            if dist < best_distance:
                best_distance = dist
                best_token = seed
        
        return best_token
    
    def find_k_nearest(self, position: np.ndarray, k: int = 5,
                       allowed_categories: Optional[Set[TokenCategory]] = None,
                       exclude: Optional[Set[str]] = None) -> List[Tuple[TokenSeed, float]]:
        """Find k nearest tokens with distances."""
        exclude = exclude or set()
        
        distances = []
        for text, seed in self._tokens.items():
            if text in exclude:
                continue
            if allowed_categories and seed.category not in allowed_categories:
                continue
            
            dist = seed.distance_to(position)
            distances.append((seed, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def tokens_in_category(self, category: TokenCategory) -> List[str]:
        """Get all tokens in a category."""
        return self._by_category.get(category, [])
    
    def tokens_in_role(self, role: TokenRole) -> List[str]:
        """Get all tokens with a role."""
        return self._by_role.get(role, [])
    
    def stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            'total_tokens': len(self._tokens),
            'by_category': {cat.name: len(tokens) for cat, tokens in self._by_category.items()},
            'by_role': {role.name: len(tokens) for role, tokens in self._by_role.items()},
        }


# =============================================================================
# BOOTSTRAP PYTHON VOCABULARY
# =============================================================================

def build_python_token_vocabulary() -> PythonTokenVocabulary:
    """
    Build vocabulary of Python tokens with semantic positions.
    
    Position dimensions: [category, role, side_effect, arity]
    - category: normalized 0-1 from TokenCategory enum
    - role: normalized 0-1 from TokenRole enum  
    - side_effect: -1 (pure) to +1 (impure)
    - arity: 0 (nullary) to 1 (variadic), normalized
    """
    vocab = PythonTokenVocabulary(dims=4)
    
    # Helper to normalize category/role to 0-1 range
    def norm_cat(cat: TokenCategory) -> float:
        return cat.value / 5.0
    
    def norm_role(role: TokenRole) -> float:
        return role.value / 4.0
    
    # -------------------------------------------------------------------------
    # KEYWORDS - Control Flow
    # -------------------------------------------------------------------------
    
    # Conditionals
    vocab.add_token("if", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.5]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"EXPRESSION", "IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("elif", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.5]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"EXPRESSION", "IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("else", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {":"})
    
    # Loops
    vocab.add_token("for", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"IDENTIFIER"})
    
    vocab.add_token("while", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.5]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"EXPRESSION", "IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("in", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"IDENTIFIER", "BUILTIN", "("})
    
    # Control
    vocab.add_token("return", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0.33]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    {"EXPRESSION", "IDENTIFIER", "LITERAL", "BUILTIN", "(", "None"})
    
    vocab.add_token("break", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    set())
    
    vocab.add_token("continue", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    set())
    
    vocab.add_token("pass", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.CONTROL), 0, 0]),
                    TokenCategory.KEYWORD, TokenRole.CONTROL,
                    set())
    
    # -------------------------------------------------------------------------
    # KEYWORDS - Definitions
    # -------------------------------------------------------------------------
    
    vocab.add_token("def", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.DEFINITION), 0, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.DEFINITION,
                    {"IDENTIFIER"})
    
    vocab.add_token("class", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.DEFINITION), 0, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.DEFINITION,
                    {"IDENTIFIER"})
    
    vocab.add_token("import", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.DEFINITION), 0.5, 0.33]),
                    TokenCategory.KEYWORD, TokenRole.DEFINITION,
                    {"IDENTIFIER"})
    
    vocab.add_token("from", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.DEFINITION), 0.5, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.DEFINITION,
                    {"IDENTIFIER"})
    
    vocab.add_token("as", np.array([norm_cat(TokenCategory.KEYWORD), norm_role(TokenRole.DEFINITION), 0, 0.67]),
                    TokenCategory.KEYWORD, TokenRole.DEFINITION,
                    {"IDENTIFIER"})
    
    # -------------------------------------------------------------------------
    # OPERATORS - Arithmetic
    # -------------------------------------------------------------------------
    
    vocab.add_token("+", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("-", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("*", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("/", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("//", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("%", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("**", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    # -------------------------------------------------------------------------
    # OPERATORS - Comparison
    # -------------------------------------------------------------------------
    
    vocab.add_token("==", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("!=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("<", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token(">", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("<=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token(">=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    # -------------------------------------------------------------------------
    # OPERATORS - Assignment
    # -------------------------------------------------------------------------
    
    vocab.add_token("=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.DEFINITION), 1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.DEFINITION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "(", "["})
    
    vocab.add_token("+=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.DEFINITION), 1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.DEFINITION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("-=", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.DEFINITION), 1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.DEFINITION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    # -------------------------------------------------------------------------
    # OPERATORS - Logical
    # -------------------------------------------------------------------------
    
    vocab.add_token("and", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("or", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token("not", np.array([norm_cat(TokenCategory.OPERATOR), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.OPERATOR, TokenRole.EXPRESSION,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    # -------------------------------------------------------------------------
    # LITERALS
    # -------------------------------------------------------------------------
    
    vocab.add_token("True", np.array([norm_cat(TokenCategory.LITERAL), norm_role(TokenRole.DATA), -1, 0]),
                    TokenCategory.LITERAL, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":"})
    
    vocab.add_token("False", np.array([norm_cat(TokenCategory.LITERAL), norm_role(TokenRole.DATA), -1, 0]),
                    TokenCategory.LITERAL, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":"})
    
    vocab.add_token("None", np.array([norm_cat(TokenCategory.LITERAL), norm_role(TokenRole.DATA), -1, 0]),
                    TokenCategory.LITERAL, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":"})
    
    # Placeholder for numeric literals
    vocab.add_token("<NUMBER>", np.array([norm_cat(TokenCategory.LITERAL), norm_role(TokenRole.DATA), -1, 0]),
                    TokenCategory.LITERAL, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "]"})
    
    # Placeholder for string literals
    vocab.add_token("<STRING>", np.array([norm_cat(TokenCategory.LITERAL), norm_role(TokenRole.DATA), -1, 0]),
                    TokenCategory.LITERAL, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "]"})
    
    # -------------------------------------------------------------------------
    # PUNCTUATION
    # -------------------------------------------------------------------------
    
    vocab.add_token("(", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "(", ")", "OPERATOR"})
    
    vocab.add_token(")", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"OPERATOR", ")", ",", ":", "[", "."})
    
    vocab.add_token("[", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "[", "]"})
    
    vocab.add_token("]", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"OPERATOR", ")", ",", ":", "]", "."})
    
    vocab.add_token("{", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"IDENTIFIER", "LITERAL", "STRING", "}"})
    
    vocab.add_token("}", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"OPERATOR", ")", ",", ":"})
    
    vocab.add_token(":", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"NEWLINE", "IDENTIFIER", "LITERAL", "BUILTIN"})
    
    vocab.add_token(",", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"IDENTIFIER", "LITERAL", "BUILTIN", "("})
    
    vocab.add_token(".", np.array([norm_cat(TokenCategory.PUNCTUATION), norm_role(TokenRole.STRUCTURE), 0, 0]),
                    TokenCategory.PUNCTUATION, TokenRole.STRUCTURE,
                    {"IDENTIFIER"})
    
    # -------------------------------------------------------------------------
    # BUILTINS - Pure (no side effects)
    # -------------------------------------------------------------------------
    
    vocab.add_token("len", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("range", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("int", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("str", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("float", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("list", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("dict", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("sum", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("max", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("min", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("abs", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("sorted", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("enumerate", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("zip", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("map", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("filter", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), -1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    # -------------------------------------------------------------------------
    # BUILTINS - Impure (side effects)
    # -------------------------------------------------------------------------
    
    vocab.add_token("print", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), 1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("input", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), 1, 0.33]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    vocab.add_token("open", np.array([norm_cat(TokenCategory.BUILTIN), norm_role(TokenRole.EXPRESSION), 1, 0.67]),
                    TokenCategory.BUILTIN, TokenRole.EXPRESSION,
                    {"("})
    
    # -------------------------------------------------------------------------
    # PLACEHOLDER IDENTIFIERS
    # -------------------------------------------------------------------------
    
    # Common variable names as placeholders
    vocab.add_token("x", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("y", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("i", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("n", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("result", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("data", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("item", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    vocab.add_token("value", np.array([norm_cat(TokenCategory.IDENTIFIER), norm_role(TokenRole.DATA), 0, 0]),
                    TokenCategory.IDENTIFIER, TokenRole.DATA,
                    {"OPERATOR", ")", ",", ":", "[", ".", "="})
    
    return vocab


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_token_clustering():
    """Demonstrate that tokens cluster by semantic similarity."""
    print("=" * 60)
    print("TOKEN CLUSTERING BY SEMANTIC SIMILARITY")
    print("=" * 60)
    print()
    
    vocab = build_python_token_vocabulary()
    print(f"Vocabulary stats: {vocab.stats()}")
    print()
    
    # Show centroids by category
    print("Centroids by CATEGORY:")
    print("-" * 40)
    
    for category in TokenCategory:
        tokens = vocab.tokens_in_category(category)
        if tokens:
            positions = [vocab.get_position(t) for t in tokens]
            centroid = np.mean(positions, axis=0)
            print(f"  {category.name:12} ({len(tokens):2} tokens): "
                  f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}, {centroid[3]:.2f}]")
    
    print()
    
    # Show centroids by role
    print("Centroids by ROLE:")
    print("-" * 40)
    
    for role in TokenRole:
        tokens = vocab.tokens_in_role(role)
        if tokens:
            positions = [vocab.get_position(t) for t in tokens]
            centroid = np.mean(positions, axis=0)
            print(f"  {role.name:12} ({len(tokens):2} tokens): "
                  f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}, {centroid[3]:.2f}]")
    
    print()


def demo_nearest_neighbor():
    """Demonstrate nearest neighbor lookup."""
    print("=" * 60)
    print("NEAREST NEIGHBOR LOOKUP")
    print("=" * 60)
    print()
    
    vocab = build_python_token_vocabulary()
    
    # Test queries
    queries = [
        ("Control flow keyword", np.array([0, 0, 0, 0.5])),  # keyword, control
        ("Binary operator", np.array([0.2, 0.75, -1, 0.67])),  # operator, expression, pure, binary
        ("Output builtin", np.array([1, 0.75, 1, 0.67])),  # builtin, expression, impure
        ("Data identifier", np.array([0.6, 0.25, 0, 0])),  # identifier, data
    ]
    
    for name, pos in queries:
        print(f"Query: {name}")
        print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}, {pos[3]:.2f}]")
        
        nearest = vocab.find_k_nearest(pos, k=5)
        print(f"  Nearest tokens:")
        for token, dist in nearest:
            print(f"    {token.text:10} (dist={dist:.3f}, cat={token.category.name}, role={token.role.name})")
        print()


def demo_semantic_similarity():
    """Demonstrate that similar tokens have similar positions."""
    print("=" * 60)
    print("SEMANTIC SIMILARITY")
    print("=" * 60)
    print()
    
    vocab = build_python_token_vocabulary()
    
    # Groups of semantically similar tokens
    groups = [
        ("Arithmetic operators", ["+", "-", "*", "/"]),
        ("Comparison operators", ["==", "!=", "<", ">"]),
        ("Control keywords", ["if", "elif", "else"]),
        ("Loop keywords", ["for", "while"]),
        ("Pure builtins", ["len", "sum", "max", "min"]),
        ("Impure builtins", ["print", "input", "open"]),
    ]
    
    for group_name, tokens in groups:
        positions = [vocab.get_position(t) for t in tokens if vocab.get_position(t) is not None]
        if len(positions) >= 2:
            # Calculate pairwise distances
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            avg_dist = np.mean(distances) if distances else 0
            max_dist = np.max(distances) if distances else 0
            
            print(f"{group_name}:")
            print(f"  Tokens: {tokens}")
            print(f"  Avg distance: {avg_dist:.3f}, Max distance: {max_dist:.3f}")
            print(f"  (Lower = more similar)")
            print()


def demo_side_effect_dimension():
    """Demonstrate the side effect dimension separates pure from impure."""
    print("=" * 60)
    print("SIDE EFFECT DIMENSION")
    print("=" * 60)
    print()
    
    vocab = build_python_token_vocabulary()
    
    # Get all builtins and sort by side_effect dimension
    builtins = vocab.tokens_in_category(TokenCategory.BUILTIN)
    
    builtin_positions = []
    for b in builtins:
        pos = vocab.get_position(b)
        if pos is not None:
            builtin_positions.append((b, pos[2]))  # side_effect is dimension 2
    
    builtin_positions.sort(key=lambda x: x[1])
    
    print("Builtins sorted by side_effect dimension:")
    print("-" * 40)
    print("(negative = pure, positive = impure)")
    print()
    
    for token, side_effect in builtin_positions:
        purity = "PURE" if side_effect < 0 else "IMPURE" if side_effect > 0 else "NEUTRAL"
        print(f"  {token:12} side_effect={side_effect:+.1f}  ({purity})")
    
    print()


if __name__ == "__main__":
    demo_token_clustering()
    demo_nearest_neighbor()
    demo_semantic_similarity()
    demo_side_effect_dimension()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print()
    print("Key findings:")
    print("1. Tokens cluster by category (keywords, operators, builtins, etc.)")
    print("2. Tokens cluster by role (control, data, definition, expression)")
    print("3. Side effect dimension separates pure from impure operations")
    print("4. Nearest neighbor finds semantically similar tokens")
    print()
    print("This vocabulary can be used for:")
    print("- Token prediction (find_nearest)")
    print("- Syntax-aware generation (follow can_follow constraints)")
    print("- Code transformation (apply deltas like perspective system)")
