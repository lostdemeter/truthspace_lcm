#!/usr/bin/env python3
"""
Emergent Query System

Test the emergent dimension system on real queries.
The system uses discovered dimensions to understand and respond to queries.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import re

# Import our discoverer
from auto_dimension_discovery import AutoDimensionDiscoverer


class EmergentQuerySystem:
    """
    A query system built on emergent dimensions.
    
    Uses discovered dimensions to:
    1. Understand query concepts
    2. Find relevant information
    3. Generate responses based on dimensional relationships
    """
    
    def __init__(self, discoverer: AutoDimensionDiscoverer):
        self.discoverer = discoverer
        self.corpus_frames: List[Dict] = []
        
    def load_corpus(self, corpus_path: str):
        """Load the corpus for retrieval."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        self.corpus_frames = corpus['frames']
        print(f"Loaded {len(self.corpus_frames)} frames for retrieval")
    
    def extract_concepts(self, query: str) -> List[str]:
        """Extract concepts from a query."""
        # Simple extraction: find known agents in query
        query_lower = query.lower()
        found = []
        
        for agent in self.discoverer.agents:
            if agent in query_lower:
                found.append(agent)
        
        # Also extract potential new concepts
        words = re.findall(r'\b[a-z]+\b', query_lower)
        for word in words:
            if len(word) > 3 and word not in found:
                # Check if it's a meaningful word (not stopword)
                stopwords = {'what', 'who', 'how', 'why', 'when', 'where', 'the', 'is', 'are', 
                            'was', 'were', 'will', 'would', 'could', 'should', 'have', 'has',
                            'does', 'did', 'about', 'like', 'from', 'with', 'this', 'that',
                            'more', 'most', 'some', 'any', 'all', 'each', 'every', 'both'}
                if word not in stopwords:
                    found.append(word)
        
        return found
    
    def get_dimensional_profile(self, concept: str) -> Dict[str, float]:
        """Get the dimensional profile of a concept."""
        profile = {}
        for dim in self.discoverer.dimensions:
            pos = dim.positions.get(concept.lower(), None)
            if pos is not None:
                label = dim.interpretation if dim.interpretation else f"Dim{dim.index+1}"
                profile[label] = pos
        return profile
    
    def find_relevant_frames(self, concepts: List[str], k: int = 5) -> List[Dict]:
        """Find frames relevant to the concepts."""
        relevant = []
        
        for frame in self.corpus_frames:
            agent = frame.get('agent', '').lower()
            text = frame.get('text', '').lower()
            
            # Score by concept matches
            score = 0
            for concept in concepts:
                if concept in agent:
                    score += 2
                if concept in text:
                    score += 1
            
            if score > 0:
                relevant.append((score, frame))
        
        # Sort by score and return top k
        relevant.sort(key=lambda x: -x[0])
        return [f for _, f in relevant[:k]]
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query and return a response.
        
        Returns:
            - concepts: extracted concepts
            - profiles: dimensional profiles of concepts
            - similar: similar concepts for each query concept
            - relevant_frames: relevant information from corpus
            - response: generated response
        """
        result = {
            'query': query_text,
            'concepts': [],
            'profiles': {},
            'similar': {},
            'opposite': {},
            'relevant_frames': [],
            'response': '',
        }
        
        # Extract concepts
        concepts = self.extract_concepts(query_text)
        result['concepts'] = concepts
        
        if not concepts:
            result['response'] = "I couldn't identify any concepts in your query."
            return result
        
        # Get dimensional profiles
        for concept in concepts:
            profile = self.get_dimensional_profile(concept)
            if profile:
                result['profiles'][concept] = profile
                
                # Find similar and opposite
                similar = self.discoverer.find_similar(concept, k=3)
                if similar:
                    result['similar'][concept] = [s[0] for s in similar]
                
                opposite = self.discoverer.find_opposite(concept)
                if opposite:
                    result['opposite'][concept] = opposite[0]
        
        # Find relevant frames
        relevant = self.find_relevant_frames(concepts, k=5)
        result['relevant_frames'] = [f['text'] for f in relevant]
        
        # Generate response
        result['response'] = self._generate_response(result)
        
        return result
    
    def _generate_response(self, result: Dict) -> str:
        """Generate a response based on query analysis."""
        concepts = result['concepts']
        profiles = result['profiles']
        similar = result['similar']
        opposite = result['opposite']
        frames = result['relevant_frames']
        
        response_parts = []
        
        # Describe each concept
        for concept in concepts:
            if concept in profiles:
                profile = profiles[concept]
                
                # Build description from dimensions
                desc_parts = []
                for dim_name, pos in profile.items():
                    if 'agency' in dim_name.lower():
                        if pos > 0.1:
                            desc_parts.append("high agency (active, decisive)")
                        elif pos < -0.1:
                            desc_parts.append("low agency (passive, supportive)")
                    elif 'animacy' in dim_name.lower():
                        if pos > 0.1:
                            desc_parts.append("animate/living")
                        elif pos < -0.1:
                            desc_parts.append("abstract/non-living")
                    elif 'morality' in dim_name.lower():
                        if pos > 0.1:
                            desc_parts.append("morally positive")
                        elif pos < -0.1:
                            desc_parts.append("morally ambiguous")
                
                if desc_parts:
                    response_parts.append(f"{concept.title()}: {', '.join(desc_parts)}")
                
                # Add similar concepts
                if concept in similar:
                    response_parts.append(f"  Similar to: {', '.join(similar[concept])}")
                
                # Add opposite
                if concept in opposite:
                    response_parts.append(f"  Opposite of: {opposite[concept]}")
        
        # Add relevant information
        if frames:
            response_parts.append("\nRelevant information:")
            for frame in frames[:3]:
                response_parts.append(f"  • {frame}")
        
        return '\n'.join(response_parts) if response_parts else "No dimensional information available."
    
    def compare(self, concept1: str, concept2: str) -> Dict[str, Any]:
        """Compare two concepts across all dimensions."""
        result = {
            'concept1': concept1,
            'concept2': concept2,
            'profiles': {},
            'differences': {},
            'similarity': 0.0,
        }
        
        profile1 = self.get_dimensional_profile(concept1)
        profile2 = self.get_dimensional_profile(concept2)
        
        result['profiles'][concept1] = profile1
        result['profiles'][concept2] = profile2
        
        # Calculate differences
        if profile1 and profile2:
            for dim_name in profile1:
                if dim_name in profile2:
                    diff = profile2[dim_name] - profile1[dim_name]
                    result['differences'][dim_name] = diff
            
            # Calculate overall similarity
            vec1 = np.array([profile1.get(d, 0) for d in profile1])
            vec2 = np.array([profile2.get(d, 0) for d in profile1])
            dist = np.linalg.norm(vec1 - vec2)
            result['similarity'] = 1.0 / (1.0 + dist)  # Convert distance to similarity
        
        return result


def interactive_session(system: EmergentQuerySystem):
    """Run an interactive query session."""
    print("\n" + "=" * 70)
    print("EMERGENT QUERY SYSTEM - Interactive Mode")
    print("=" * 70)
    print("\nCommands:")
    print("  query <text>     - Ask about concepts")
    print("  compare <a> <b>  - Compare two concepts")
    print("  profile <name>   - Show dimensional profile")
    print("  similar <name>   - Find similar concepts")
    print("  quit             - Exit")
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            break
        
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == 'query':
            result = system.query(args)
            print(f"\nConcepts found: {result['concepts']}")
            print(f"\nResponse:\n{result['response']}")
        
        elif cmd == 'compare':
            concepts = args.split()
            if len(concepts) >= 2:
                result = system.compare(concepts[0], concepts[1])
                print(f"\nComparing {concepts[0]} vs {concepts[1]}:")
                print(f"Similarity: {result['similarity']:.3f}")
                print("\nDifferences by dimension:")
                for dim, diff in result['differences'].items():
                    direction = "→" if diff > 0 else "←" if diff < 0 else "="
                    print(f"  {dim}: {diff:+.3f} {direction}")
            else:
                print("Usage: compare <concept1> <concept2>")
        
        elif cmd == 'profile':
            profile = system.get_dimensional_profile(args)
            if profile:
                print(f"\nDimensional profile for {args}:")
                for dim, pos in profile.items():
                    bar = "█" * int(abs(pos) * 20)
                    direction = "+" if pos > 0 else "-"
                    print(f"  {dim}: {direction}{bar} ({pos:+.3f})")
            else:
                print(f"No profile found for '{args}'")
        
        elif cmd == 'similar':
            similar = system.discoverer.find_similar(args, k=5)
            if similar:
                print(f"\nMost similar to {args}:")
                for name, dist in similar:
                    print(f"  {name}: distance={dist:.3f}")
            else:
                print(f"No similar concepts found for '{args}'")
        
        else:
            # Treat as a query
            result = system.query(user_input)
            print(f"\nConcepts found: {result['concepts']}")
            print(f"\nResponse:\n{result['response']}")
        
        print()


def run_test_queries(system: EmergentQuerySystem):
    """Run a set of test queries."""
    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)
    
    test_queries = [
        "Tell me about Holmes",
        "Who is similar to Watson?",
        "Compare Holmes and Moriarty",
        "What is the difference between a king and a servant?",
        "Tell me about the villain",
        "Who is the opposite of Alice?",
        "Compare robot and storm",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"Query: {query}")
        print(f"{'─'*70}")
        
        result = system.query(query)
        print(f"Concepts: {result['concepts']}")
        print(f"\n{result['response']}")


def main():
    print("=" * 70)
    print("EMERGENT QUERY SYSTEM")
    print("=" * 70)
    
    # Load corpus and discover dimensions
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_generated.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        return None
    
    # Create discoverer
    discoverer = AutoDimensionDiscoverer(
        variance_threshold=0.80,
        min_dimension_variance=0.03,
        max_dimensions=12,
    )
    
    discoverer.ingest_corpus(str(corpus_path))
    discoverer.discover_dimensions()
    discoverer.correlate_with_hidden_properties()
    
    # Create query system
    system = EmergentQuerySystem(discoverer)
    system.load_corpus(str(corpus_path))
    
    # Run test queries
    run_test_queries(system)
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Entering interactive mode...")
    interactive_session(system)
    
    return system


if __name__ == "__main__":
    system = main()
