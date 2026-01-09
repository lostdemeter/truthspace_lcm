#!/usr/bin/env python3
"""
Experiment: Perspective as Layer Manipulation

The Hypothesis:
- "You are an expert X" prompts work by setting an initial perspective
- This perspective is an OFFSET applied to all subsequent queries
- The same query, from different perspectives, traverses different paths

Traditional LLM:
  System prompt → Sets hidden state bias → Colors all outputs

Geometric equivalent:
  Perspective vector → Added to query position → Different traversal path

Example:
  Query: "What is energy?"
  
  Physicist perspective: [domain=physics, specificity=expert, ...]
  Query + Physicist = Position near "E=mc², thermodynamics, quantum"
  
  Economist perspective: [domain=economics, specificity=expert, ...]
  Query + Economist = Position near "GDP, markets, labor"
  
  Child perspective: [domain=general, specificity=basic, ...]
  Query + Child = Position near "what makes things go, batteries"

The query is the same. The perspective shifts WHERE we look.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Perspective:
    """
    A perspective is an offset vector in φ-space.
    
    It represents "who is asking" or "from what viewpoint".
    When applied to a query, it shifts the query position.
    """
    name: str
    description: str
    offset: np.ndarray  # [domain, specificity, intent, formality, intrinsic_functional, action]
    
    def apply(self, query_position: np.ndarray) -> np.ndarray:
        """Apply perspective offset to query position."""
        return query_position + self.offset


# Define some perspectives
PERSPECTIVES = {
    'physicist': Perspective(
        name="Physicist",
        description="Expert in physics, technical, formal",
        offset=np.array([3, 2, 0, 1, 0, 0])  # domain=physics, specificity=expert, formality=formal
    ),
    'economist': Perspective(
        name="Economist",
        description="Expert in economics, analytical, formal",
        offset=np.array([1, 2, 0, 1, 0, 0])  # domain=business, specificity=expert
    ),
    'child': Perspective(
        name="Child",
        description="Curious, basic understanding, informal",
        offset=np.array([0, -1, 0, -1, 0, 0])  # domain=general, specificity=basic, informal
    ),
    'ai_assistant': Perspective(
        name="AI Assistant",
        description="Helpful, can answer questions, generate code, use tools",
        offset=np.array([2, 1, 0, 0, 0, 0])  # domain=technical, specificity=moderate
    ),
    'coder': Perspective(
        name="Software Developer",
        description="Expert in programming, practical, technical",
        offset=np.array([2, 2, 0, 0, 1, 0])  # domain=technical, specificity=expert, functional
    ),
}


def encode_query_simple(query: str) -> np.ndarray:
    """
    Simple query encoding for demonstration.
    Returns 6D position: [domain, specificity, intent, formality, intrinsic_functional, action]
    """
    query_lower = query.lower()
    position = np.zeros(6)
    
    # Domain keywords
    if any(w in query_lower for w in ['physics', 'energy', 'force', 'mass', 'gravity']):
        position[0] = 3  # physics
    elif any(w in query_lower for w in ['code', 'program', 'python', 'function']):
        position[0] = 2  # technical
    elif any(w in query_lower for w in ['money', 'market', 'economy', 'price']):
        position[0] = 1  # business
    
    # Action dimension (from Design 048)
    if any(query_lower.startswith(w) for w in ['what ', 'why ', 'how ', 'explain ']):
        position[5] = -2  # query
    elif any(query_lower.startswith(w) for w in ['create ', 'make ', 'build ']):
        position[5] = 1  # create
    elif any(query_lower.startswith(w) for w in ['list ', 'show ', 'run ']):
        position[5] = 2  # execute
    
    return position


# Simulated knowledge base - different "answers" at different positions
KNOWLEDGE_POSITIONS = {
    # Physics-expert answers about energy
    'energy_physics_expert': {
        'position': np.array([3, 2, 0, 1, 0, -2]),
        'answer': "Energy is the capacity to do work, measured in joules. E=mc² shows mass-energy equivalence."
    },
    # Economics answers about energy
    'energy_economics': {
        'position': np.array([1, 2, 0, 1, 0, -2]),
        'answer': "Energy markets are driven by supply/demand dynamics. Oil prices affect global GDP."
    },
    # Child-level answer about energy
    'energy_simple': {
        'position': np.array([0, -1, 0, -1, 0, -2]),
        'answer': "Energy is what makes things go! Like batteries in your toys or food that helps you run."
    },
    # Technical/coding answer
    'energy_technical': {
        'position': np.array([2, 1, 0, 0, 1, -2]),
        'answer': "In computing, energy efficiency is measured in FLOPS per watt. GPU optimization matters."
    },
    
    # Answers about "list"
    'list_ai_assistant': {
        'position': np.array([2, 1, 0, 0, 0, 2]),
        'answer': "I can list files in a directory for you. Which path would you like me to check?"
    },
    'list_child': {
        'position': np.array([0, -1, 0, -1, 0, 2]),
        'answer': "A list is like when you write down all your toys or favorite foods!"
    },
    'list_coder': {
        'position': np.array([2, 2, 0, 0, 1, 2]),
        'answer': "Lists in Python are mutable sequences. Use list comprehensions for efficiency."
    },
}


def find_nearest_answer(position: np.ndarray) -> Tuple[str, str, float]:
    """Find the nearest knowledge item to a position."""
    best_key = None
    best_answer = None
    best_distance = float('inf')
    
    for key, item in KNOWLEDGE_POSITIONS.items():
        dist = np.linalg.norm(position - item['position'])
        if dist < best_distance:
            best_distance = dist
            best_key = key
            best_answer = item['answer']
    
    return best_key, best_answer, best_distance


def demonstrate_perspective_shift():
    """Show how the same query gets different answers from different perspectives."""
    
    print("=" * 70)
    print("PERSPECTIVE AS LAYER MANIPULATION")
    print("=" * 70)
    print()
    print("The same query, from different perspectives, finds different answers.")
    print("Perspective is an OFFSET applied before traversal, not a filter after.")
    print()
    
    queries = [
        "What is energy?",
        "List files",
    ]
    
    perspectives_to_test = ['physicist', 'economist', 'child', 'ai_assistant', 'coder']
    
    for query in queries:
        print("-" * 70)
        print(f"QUERY: \"{query}\"")
        print("-" * 70)
        
        base_position = encode_query_simple(query)
        print(f"Base position: {base_position}")
        print()
        
        for persp_name in perspectives_to_test:
            persp = PERSPECTIVES[persp_name]
            adjusted_position = persp.apply(base_position)
            
            key, answer, distance = find_nearest_answer(adjusted_position)
            
            print(f"  {persp.name}:")
            print(f"    Offset: {persp.offset}")
            print(f"    Adjusted: {adjusted_position}")
            print(f"    → {key} (dist={distance:.2f})")
            print(f"    Answer: \"{answer[:60]}...\"" if len(answer) > 60 else f"    Answer: \"{answer}\"")
            print()
    
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The perspective doesn't FILTER the query - it SHIFTS it.

"You are an expert physicist" doesn't mean:
  "Only return physics answers"

It means:
  "Start from a physics-expert position in the space"

This is why the same words mean different things to different agents:
  - "List" to an AI assistant → tool action (list files)
  - "List" to a child → concept explanation (what is a list?)
  - "List" to a coder → data structure (Python lists)

The query encodes WHAT is being asked.
The perspective encodes WHO is asking (or being asked).
Together they determine WHERE in the space we look.

IMPLEMENTATION:
  adjusted_position = query_position + perspective_offset
  answer = find_nearest(adjusted_position)

This is simpler than capability projection because:
  1. No need to define capability structures
  2. Perspective is just a vector addition
  3. Works with existing φ-lattice encoding
  4. Matches how LLM system prompts actually work
""")


def demonstrate_identity_as_perspective():
    """Show how agent identity can be encoded as a perspective."""
    
    print()
    print("=" * 70)
    print("AGENT IDENTITY AS PERSPECTIVE")
    print("=" * 70)
    print()
    
    # Define HyperChat's identity as a perspective
    hyperchat_identity = Perspective(
        name="HyperChat",
        description="AI assistant that can answer questions, generate plots, and use tools",
        offset=np.array([
            2,   # domain: technical (can discuss code, tools)
            1,   # specificity: moderate (not expert, not basic)
            0,   # intent: neutral
            0,   # formality: neutral
            0,   # intrinsic_functional: balanced
            0,   # action: neutral (responds to user's action)
        ])
    )
    
    print(f"HyperChat Identity Perspective:")
    print(f"  Name: {hyperchat_identity.name}")
    print(f"  Description: {hyperchat_identity.description}")
    print(f"  Offset: {hyperchat_identity.offset}")
    print()
    
    # Test queries
    test_queries = [
        ("What is Python?", "Should find technical/knowledge answer"),
        ("List files", "Should find tool action answer"),
        ("Create a plot", "Should find code generation answer"),
    ]
    
    print("Query processing with HyperChat identity:")
    print()
    
    for query, expected in test_queries:
        base_pos = encode_query_simple(query)
        adjusted_pos = hyperchat_identity.apply(base_pos)
        key, answer, dist = find_nearest_answer(adjusted_pos)
        
        print(f"  \"{query}\"")
        print(f"    Base: {base_pos} → Adjusted: {adjusted_pos}")
        print(f"    Found: {key}")
        print(f"    Expected: {expected}")
        print()
    
    print("-" * 70)
    print("IMPLICATION FOR HYPERCHAT:")
    print("-" * 70)
    print("""
Instead of:
  1. Encode query
  2. Project through capability structure
  3. Route to handler

We do:
  1. Encode query
  2. Add identity perspective offset
  3. Find nearest in adjusted space

The identity IS the perspective. The perspective IS the offset.
No capability structures needed - just vector addition.

This matches the LLM intuition:
  "You are HyperChat, an AI assistant..."
  
Is equivalent to:
  query_position += HYPERCHAT_IDENTITY_OFFSET
""")


if __name__ == "__main__":
    demonstrate_perspective_shift()
    demonstrate_identity_as_perspective()
