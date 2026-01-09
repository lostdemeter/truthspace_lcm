#!/usr/bin/env python3
"""
Experiment: Agent-Grounded Geometric Interpretation

The Hypothesis:
- A query encodes to a position in concept space
- Different agents have different capability structures
- The SAME query, projected through DIFFERENT agents, yields DIFFERENT answers
- Both answers are geometrically valid - they're just different projections

Example:
  Query: "Can you pick up a cup?"
  Human agent: Projects through physical capability → "Yes"
  AI agent: Projects through digital capability → "No"

The query position doesn't change. The agent structure determines the path.

This is like asking "what's north?" - the answer depends on where YOU are standing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


# φ for geometric scaling
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Capability:
    """A capability an agent possesses."""
    name: str
    domain: str  # physical, digital, knowledge, social
    position: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    def __post_init__(self):
        # Position encodes: [physical, digital, knowledge, social]
        domain_map = {
            'physical': np.array([1.0, 0.0, 0.0, 0.0]),
            'digital': np.array([0.0, 1.0, 0.0, 0.0]),
            'knowledge': np.array([0.0, 0.0, 1.0, 0.0]),
            'social': np.array([0.0, 0.0, 0.0, 1.0]),
        }
        self.position = domain_map.get(self.domain, np.zeros(4))


@dataclass
class Agent:
    """
    An agent with a specific set of capabilities.
    
    The agent's capability structure defines HOW it interprets queries.
    """
    name: str
    capabilities: List[Capability] = field(default_factory=list)
    
    @property
    def capability_matrix(self) -> np.ndarray:
        """Build matrix of capability positions."""
        if not self.capabilities:
            return np.zeros((1, 4))
        return np.array([c.position for c in self.capabilities])
    
    @property
    def capability_centroid(self) -> np.ndarray:
        """The agent's 'center' in capability space."""
        return self.capability_matrix.mean(axis=0)
    
    def has_capability_in_domain(self, domain: str) -> bool:
        """Check if agent has any capability in a domain."""
        return any(c.domain == domain for c in self.capabilities)
    
    def project_query(self, query_position: np.ndarray) -> Tuple[float, str]:
        """
        Project a query through this agent's capability structure.
        
        Returns:
            (score, explanation) where score indicates how well the agent
            can handle this query (0 = cannot, 1 = fully capable)
        """
        if not self.capabilities:
            return 0.0, "No capabilities"
        
        # Find the closest capability to the query
        distances = []
        for cap in self.capabilities:
            dist = np.linalg.norm(query_position - cap.position)
            distances.append((dist, cap))
        
        distances.sort(key=lambda x: x[0])
        closest_dist, closest_cap = distances[0]
        
        # Convert distance to similarity (closer = higher)
        # Use φ-based decay
        similarity = PHI ** (-closest_dist)
        
        return similarity, f"Closest capability: {closest_cap.name} ({closest_cap.domain})"


def encode_query(query: str) -> np.ndarray:
    """
    Encode a query to a position in capability space.
    
    Position encodes: [physical, digital, knowledge, social]
    """
    query_lower = query.lower()
    
    # Simple keyword-based encoding for demonstration
    # In production, this would use the φ-lattice
    position = np.zeros(4)
    
    # Physical domain keywords
    physical_words = {'pick', 'up', 'hold', 'touch', 'move', 'walk', 'run', 
                      'grab', 'lift', 'carry', 'push', 'pull', 'cup', 'object'}
    # Digital domain keywords  
    digital_words = {'file', 'list', 'read', 'write', 'code', 'program',
                     'compute', 'calculate', 'search', 'download', 'upload'}
    # Knowledge domain keywords
    knowledge_words = {'what', 'is', 'explain', 'describe', 'define', 'tell',
                       'know', 'understand', 'mean', 'why', 'how'}
    # Social domain keywords
    social_words = {'hello', 'thanks', 'please', 'help', 'sorry', 'goodbye',
                    'feel', 'think', 'believe', 'want', 'need'}
    
    words = set(query_lower.split())
    
    position[0] = len(words & physical_words) / max(len(physical_words), 1)
    position[1] = len(words & digital_words) / max(len(digital_words), 1)
    position[2] = len(words & knowledge_words) / max(len(knowledge_words), 1)
    position[3] = len(words & social_words) / max(len(social_words), 1)
    
    # Normalize
    norm = np.linalg.norm(position)
    if norm > 0:
        position = position / norm
    
    return position


def create_human_agent() -> Agent:
    """Create an agent with human-like capabilities."""
    return Agent(
        name="Human",
        capabilities=[
            Capability("pick up objects", "physical"),
            Capability("walk and move", "physical"),
            Capability("speak and listen", "social"),
            Capability("read and write", "knowledge"),
            Capability("feel emotions", "social"),
            Capability("use tools", "physical"),
        ]
    )


def create_ai_agent() -> Agent:
    """Create an agent with AI-like capabilities."""
    return Agent(
        name="AI",
        capabilities=[
            Capability("list files", "digital"),
            Capability("read documents", "digital"),
            Capability("write code", "digital"),
            Capability("search knowledge", "knowledge"),
            Capability("answer questions", "knowledge"),
            Capability("generate text", "digital"),
        ]
    )


def create_robot_agent() -> Agent:
    """Create an agent with robot-like capabilities (physical + digital)."""
    return Agent(
        name="Robot",
        capabilities=[
            Capability("pick up objects", "physical"),
            Capability("move around", "physical"),
            Capability("process commands", "digital"),
            Capability("sense environment", "physical"),
            Capability("execute programs", "digital"),
        ]
    )


def interpret_query(query: str, agents: List[Agent]) -> Dict[str, Tuple[float, str, str]]:
    """
    Have multiple agents interpret the same query.
    
    Returns:
        Dict mapping agent name to (score, capability_explanation, answer)
    """
    query_position = encode_query(query)
    
    results = {}
    for agent in agents:
        score, explanation = agent.project_query(query_position)
        
        # Determine answer based on score
        if score > 0.7:
            answer = "Yes, I can do that."
        elif score > 0.4:
            answer = "I might be able to help with that."
        elif score > 0.2:
            answer = "That's not really my strength, but I can try."
        else:
            answer = "No, I cannot do that."
        
        results[agent.name] = (score, explanation, answer)
    
    return results


def main():
    print("=" * 60)
    print("AGENT-GROUNDED GEOMETRIC INTERPRETATION")
    print("=" * 60)
    print()
    print("The same query, projected through different agent structures,")
    print("yields different answers. Both are geometrically valid.")
    print()
    
    # Create agents
    human = create_human_agent()
    ai = create_ai_agent()
    robot = create_robot_agent()
    agents = [human, ai, robot]
    
    # Print agent capabilities
    print("AGENT CAPABILITIES:")
    print("-" * 40)
    for agent in agents:
        caps = ", ".join(c.name for c in agent.capabilities)
        print(f"{agent.name}: {caps}")
    print()
    
    # Test queries
    test_queries = [
        "Can you pick up a cup?",
        "Can you list files in a directory?",
        "What is the meaning of life?",
        "Can you help me feel better?",
        "Can you write a Python script?",
        "Can you walk to the store?",
    ]
    
    print("QUERY INTERPRETATIONS:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        query_pos = encode_query(query)
        print(f"  Position: [phys={query_pos[0]:.2f}, dig={query_pos[1]:.2f}, "
              f"know={query_pos[2]:.2f}, soc={query_pos[3]:.2f}]")
        
        results = interpret_query(query, agents)
        
        for agent_name, (score, explanation, answer) in results.items():
            print(f"  {agent_name} ({score:.2f}): {answer}")
    
    print()
    print("=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print("""
The query "Can you pick up a cup?" has a fixed position in concept space.
But when projected through different agent structures:

  Human → Projects onto physical capabilities → "Yes"
  AI    → No physical capabilities → "No"
  Robot → Has physical capabilities → "Yes"

The geometry is the same. The agent structure determines the path.
This is how "list files" can mean something to an AI but not to a human
who has never used a computer.

IMPLICATION FOR INTENT CLASSIFICATION:
Instead of matching queries to patterns, we should:
1. Encode the query to a position
2. Project through the agent's capability structure
3. The projection determines both INTENT and ANSWER

The agent's self-model IS the geometry that transforms queries into actions.
""")


def intent_classification_demo():
    """
    Demonstrate how agent grounding solves the intent classification problem.
    """
    print()
    print("=" * 60)
    print("INTENT CLASSIFICATION VIA AGENT GROUNDING")
    print("=" * 60)
    print()
    
    # Create our HyperChat agent with specific capabilities
    hyperchat = Agent(
        name="HyperChat",
        capabilities=[
            # Knowledge capabilities
            Capability("answer questions about python", "knowledge"),
            Capability("explain physics concepts", "knowledge"),
            Capability("describe machine learning", "knowledge"),
            
            # Digital/tool capabilities
            Capability("list files", "digital"),
            Capability("read file contents", "digital"),
            Capability("search codebase", "digital"),
            
            # Generation capabilities (we'll add a 5th dimension for this)
            Capability("generate matplotlib plots", "digital"),
            Capability("create visualizations", "digital"),
        ]
    )
    
    # The key insight: "what is a histogram" vs "create a histogram"
    # Both contain "histogram" but have different INTENT
    
    test_cases = [
        ("what is a histogram", "KNOWLEDGE - asking about the concept"),
        ("create a histogram", "CODE - asking to generate something"),
        ("what is python", "KNOWLEDGE - asking about the concept"),
        ("list files", "TOOL - asking to perform an action"),
        ("explain matplotlib", "KNOWLEDGE - asking for explanation"),
        ("plot a sine wave", "CODE - asking to generate something"),
    ]
    
    print("The problem: 'what is a histogram' and 'create a histogram'")
    print("both contain 'histogram' but have different intents.")
    print()
    print("Solution: The VERB encodes the relationship to the agent.")
    print("  'what is' → query about knowledge → KNOWLEDGE intent")
    print("  'create'  → request for action → CODE intent")
    print()
    
    # Encode with action awareness
    for query, expected in test_cases:
        query_lower = query.lower()
        
        # Determine action type from verb
        if any(query_lower.startswith(v) for v in ['what ', 'explain ', 'describe ', 'tell ']):
            action = "query"  # Asking about something
        elif any(query_lower.startswith(v) for v in ['create ', 'make ', 'generate ', 'plot ', 'draw ']):
            action = "generate"  # Asking to create something
        elif any(query_lower.startswith(v) for v in ['list ', 'read ', 'show ', 'find ', 'search ']):
            action = "tool"  # Asking to use a tool
        else:
            action = "unknown"
        
        # The action type IS the geometric dimension that distinguishes intent
        print(f"Query: \"{query}\"")
        print(f"  Action type: {action}")
        print(f"  Expected: {expected}")
        print()
    
    print("-" * 60)
    print("KEY REALIZATION:")
    print("-" * 60)
    print("""
The verbs 'what', 'create', 'list' are NOT filler words.
They encode the RELATIONSHIP between the user and the agent:

  'what is X'  → User wants to KNOW about X
  'create X'   → User wants agent to MAKE X
  'list X'     → User wants agent to SHOW X

These verbs are the INTENT DIMENSION of the query.
They should NOT be filtered out - they're the most important signal!

The current φ-Zipf filler detection removes them because they're
high-frequency, but frequency ≠ unimportance for intent words.

SOLUTION:
Add an 'action' or 'intent' dimension to the φ-lattice:
  - query verbs (what, explain, describe) → negative values
  - action verbs (create, make, generate) → positive values
  - tool verbs (list, read, show) → distinct values

This dimension encodes the user's RELATIONSHIP to the agent,
not just the content of the query.
""")


if __name__ == "__main__":
    main()
    intent_classification_demo()
