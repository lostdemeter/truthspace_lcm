#!/usr/bin/env python3
"""
Experimental Chat Program using Self-Assembling Corpus

This chat program demonstrates the geometric LCM hypothesis:
- User queries are encoded to geometric positions
- Responses emerge from traversing the semantic space
- No traditional embeddings - pure φ-based geometry

The corpus provides the "knowledge" through transformation pairs.
The chat interface provides the "interaction" through geometric queries.
"""

import sys
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

# Import from self_assembling_corpus
from experiments.self_assembling_corpus import (
    SelfAssemblingCorpus,
    SelfAssemblyLoop,
    LLMInterface,
    LLMEnhancedPipeline,
    ConceptType,
    PHI,
    SPECIFICITY_IDEAL,
    SPECIFICITY_CATEGORY,
    SPECIFICITY_INSTANCE,
)


class CorpusChat:
    """
    Experimental chat interface using the self-assembling corpus.
    
    Key insight: The corpus IS the knowledge. Queries traverse the geometry.
    
    How it works:
    1. Parse query for known concepts
    2. Compute geometric position from concept positions
    3. Find nearest concepts in the space
    4. Generate response from the geometric neighborhood
    
    This is NOT an LLM - it's a geometric knowledge retrieval system.
    The LLM is only used to fill gaps in the corpus, not to generate responses.
    """
    
    def __init__(self, corpus: SelfAssemblingCorpus = None, 
                 use_llm_for_gaps: bool = True,
                 verbose: bool = False):
        self.corpus = corpus or SelfAssemblingCorpus()
        self.verbose = verbose
        
        # LLM for gap filling (not response generation)
        self.llm = LLMInterface() if use_llm_for_gaps else None
        self.pipeline = LLMEnhancedPipeline(self.corpus, self.llm) if self.llm else None
        
        # Query history for context
        self.history: List[Dict] = []
        
        # Seed with basic knowledge if empty
        if len(self.corpus.pairs) == 0:
            self._seed_basic_knowledge()
    
    def _seed_basic_knowledge(self):
        """Seed the corpus with basic transformation pairs."""
        # Gender dimension
        self.corpus.add_pair("king", "queen", "gender")
        self.corpus.add_pair("man", "woman", "gender")
        self.corpus.add_pair("boy", "girl", "gender")
        self.corpus.add_pair("father", "mother", "gender")
        self.corpus.add_pair("brother", "sister", "gender")
        self.corpus.add_pair("prince", "princess", "gender")
        
        # Age dimension
        self.corpus.add_pair("boy", "man", "age_increase")
        self.corpus.add_pair("girl", "woman", "age_increase")
        self.corpus.add_pair("puppy", "dog", "age_increase")
        self.corpus.add_pair("kitten", "cat", "age_increase")
        self.corpus.add_pair("child", "adult", "age_increase")
        self.corpus.add_pair("baby", "child", "age_increase")
        
        # Size dimension
        self.corpus.add_pair("house", "cottage", "size_decrease")
        self.corpus.add_pair("house", "mansion", "size_increase")
        self.corpus.add_pair("dog", "puppy", "size_decrease")
        self.corpus.add_pair("cat", "kitten", "size_decrease")
        
        # Regality dimension
        self.corpus.add_pair("house", "palace", "regality_increase")
        self.corpus.add_pair("house", "hovel", "regality_decrease")
        self.corpus.add_pair("person", "king", "regality_increase")
        self.corpus.add_pair("person", "peasant", "regality_decrease")
        
        # Status dimension
        self.corpus.add_pair("person", "noble", "status_increase")
        self.corpus.add_pair("person", "servant", "status_decrease")
        
        # Register some concept metadata
        self.corpus.register_concept("king", ConceptType.IDEAL)
        self.corpus.register_concept("queen", ConceptType.IDEAL)
        self.corpus.register_concept("person", ConceptType.IDEAL)
        self.corpus.register_concept("house", ConceptType.IDEAL)
        self.corpus.register_concept("dog", ConceptType.IDEAL)
        self.corpus.register_concept("cat", ConceptType.IDEAL)
        
        self.corpus.recompute()
        
        if self.verbose:
            print(f"[Seeded corpus with {len(self.corpus.pairs)} pairs, "
                  f"{len(self.corpus.dimensions)} dimensions]")
    
    def _parse_query(self, query: str) -> List[str]:
        """Extract known concepts from the query."""
        words = re.findall(r'\b\w+\b', query.lower())
        known = []
        for word in words:
            if self.corpus.get_position(word) is not None:
                known.append(word)
        return known
    
    def _compute_query_position(self, concepts: List[str]) -> Optional[np.ndarray]:
        """Compute geometric position from query concepts."""
        if not concepts:
            return None
        
        # Average the positions of known concepts
        positions = []
        for concept in concepts:
            pos = self.corpus.get_position(concept)
            if pos is not None:
                positions.append(pos)
        
        if not positions:
            return None
        
        return np.mean(positions, axis=0)
    
    def _find_related_concepts(self, position: np.ndarray, 
                                n: int = 5,
                                exclude: List[str] = None) -> List[Tuple[str, float]]:
        """Find concepts near a position."""
        exclude = exclude or []
        results = self.corpus.find_nearest(position, n=n + len(exclude))
        
        # Filter out excluded concepts
        filtered = [(word, dist) for word, dist in results if word not in exclude]
        return filtered[:n]
    
    def _detect_query_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        # Transformation queries
        if "opposite" in query_lower or "reverse" in query_lower:
            return "transform_opposite"
        if "female form" in query_lower or "female version" in query_lower:
            return "transform_to_female"
        if "male form" in query_lower or "male version" in query_lower:
            return "transform_to_male"
        if "male" in query_lower and "female" in query_lower:
            return "transform_gender"
        if "female" in query_lower and "male" not in query_lower:
            return "transform_to_female"
        if "male" in query_lower and "female" not in query_lower:
            return "transform_to_male"
        if "young" in query_lower or "younger" in query_lower:
            return "transform_younger"
        if "old" in query_lower or "older" in query_lower:
            return "transform_older"
        if "big" in query_lower or "larger" in query_lower:
            return "transform_larger"
        if "small" in query_lower or "smaller" in query_lower:
            return "transform_smaller"
        
        # Relationship queries
        if "what is" in query_lower or "define" in query_lower:
            return "definition"
        if "related to" in query_lower or "similar to" in query_lower:
            return "similarity"
        if "difference" in query_lower or "between" in query_lower:
            return "difference"
        
        # Default: find related concepts
        return "explore"
    
    def _apply_transformation(self, concept: str, dimension: str) -> Optional[str]:
        """Apply a transformation to find the target concept."""
        # Look for pairs where this concept is the source
        for pair in self.corpus.pairs:
            if pair.source == concept and pair.relationship == dimension:
                return pair.target
            if pair.target == concept and pair.relationship == dimension:
                return pair.source
        return None
    
    def _generate_response(self, query: str, concepts: List[str], 
                           position: Optional[np.ndarray],
                           intent: str) -> str:
        """Generate a response based on geometric analysis."""
        
        if not concepts and position is None:
            # No known concepts - try to learn
            if self.pipeline and self.llm and self.llm.is_available():
                return self._handle_unknown_query(query)
            return "I don't recognize any concepts in your query. Try asking about: " + \
                   ", ".join(list(self.corpus.concepts.keys())[:10])
        
        # Handle different intents
        if intent == "transform_to_female" and concepts:
            result = self._apply_transformation(concepts[0], "gender")
            if result:
                return f"The female form of '{concepts[0]}' is '{result}'."
            return f"I don't know the female form of '{concepts[0]}'."
        
        if intent == "transform_to_male" and concepts:
            result = self._apply_transformation(concepts[0], "gender")
            if result:
                return f"The male form of '{concepts[0]}' is '{result}'."
            return f"I don't know the male form of '{concepts[0]}'."
        
        if intent == "transform_younger" and concepts:
            result = self._apply_transformation(concepts[0], "age_increase")
            if result:
                return f"A younger version of '{concepts[0]}' is '{result}'."
            return f"I don't know a younger form of '{concepts[0]}'."
        
        if intent == "transform_older" and concepts:
            result = self._apply_transformation(concepts[0], "age_increase")
            if result:
                return f"An older version of '{concepts[0]}' is '{result}'."
            return f"I don't know an older form of '{concepts[0]}'."
        
        if intent == "definition" and concepts:
            concept = concepts[0]
            ideal = self.corpus.get_ideal(concept)
            if ideal:
                dims = ", ".join(ideal.dimensions_anchored[:3])
                return f"'{concept}' is a Platonic Ideal anchoring dimensions: {dims}."
            
            # Find what it transforms from
            for pair in self.corpus.pairs:
                if pair.target == concept:
                    return f"'{concept}' is a {pair.relationship} transformation of '{pair.source}'."
            
            return f"'{concept}' exists in the corpus but I don't have a definition."
        
        if intent == "difference" and len(concepts) >= 2:
            delta = self.corpus.get_delta(concepts[0], concepts[1])
            if delta:
                mag, dim = delta
                return f"The difference between '{concepts[0]}' and '{concepts[1]}' " + \
                       f"is {mag:.2f}φ along the '{dim}' dimension."
            return f"I can't compute the difference between '{concepts[0]}' and '{concepts[1]}'."
        
        # Default: explore related concepts
        if position is not None:
            related = self._find_related_concepts(position, n=5, exclude=concepts)
            if related:
                related_str = ", ".join([f"{w} ({d:.2f})" for w, d in related[:5]])
                return f"Concepts related to {', '.join(concepts)}: {related_str}"
        
        return f"I found these concepts: {', '.join(concepts)}"
    
    def _handle_unknown_query(self, query: str) -> str:
        """Handle queries with no known concepts by trying to learn."""
        # Extract potential new concepts
        words = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'to', 'for', 'and', 
                     'or', 'in', 'on', 'at', 'by', 'with', 'about', 'how', 'why',
                     'when', 'where', 'who', 'which', 'that', 'this', 'it', 'be',
                     'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'tell', 'me', 'please', 'like', 'as'}
        
        potential = [w for w in words if w not in stopwords and len(w) > 2]
        
        if potential and self.llm and self.llm.is_available():
            # Try to get the LLM to explain the concept
            prompt = f"In one sentence, what is '{potential[0]}'? Be concise."
            response = self.llm.query(prompt)
            if response:
                return f"I'm learning about '{potential[0]}': {response}"
        
        return "I don't recognize those concepts yet. Try asking about known concepts."
    
    def chat(self, query: str) -> str:
        """Process a query and return a response."""
        # Parse query
        concepts = self._parse_query(query)
        position = self._compute_query_position(concepts)
        intent = self._detect_query_intent(query)
        
        if self.verbose:
            print(f"[Concepts: {concepts}, Intent: {intent}]")
        
        # Generate response
        response = self._generate_response(query, concepts, position, intent)
        
        # Record in history
        self.history.append({
            "query": query,
            "concepts": concepts,
            "intent": intent,
            "response": response
        })
        
        return response
    
    def get_corpus_stats(self) -> Dict:
        """Get statistics about the corpus."""
        self.corpus.recompute()
        return {
            "pairs": len(self.corpus.pairs),
            "dimensions": len(self.corpus.dimensions),
            "concepts": len(self.corpus.concepts),
            "ideals": len(self.corpus.ideals),
            "dimension_names": list(self.corpus.dimensions.keys())
        }
    
    def print_help(self):
        """Print help information."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║              CORPUS CHAT - Geometric Knowledge               ║
╠══════════════════════════════════════════════════════════════╣
║ This is an experimental chat using geometric knowledge.      ║
║ Queries are encoded to positions in φ-space.                 ║
║ Responses emerge from the semantic geometry.                 ║
╠══════════════════════════════════════════════════════════════╣
║ COMMANDS:                                                    ║
║   /help     - Show this help                                 ║
║   /stats    - Show corpus statistics                         ║
║   /concepts - List known concepts                            ║
║   /dims     - List dimensions                                ║
║   /ideals   - List Platonic Ideals                           ║
║   /quit     - Exit the chat                                  ║
╠══════════════════════════════════════════════════════════════╣
║ EXAMPLE QUERIES:                                             ║
║   "What is the female form of king?"                         ║
║   "Tell me about queen"                                      ║
║   "What's related to house?"                                 ║
║   "What's the difference between boy and man?"               ║
║   "What is a younger version of dog?"                        ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    """Run the interactive chat."""
    print()
    print("=" * 60)
    print("CORPUS CHAT - Experimental Geometric Knowledge Interface")
    print("=" * 60)
    print()
    print("This chat uses the self-assembling corpus for knowledge.")
    print("Type /help for commands, /quit to exit.")
    print()
    
    # Create chat instance
    chat = CorpusChat(verbose=False)
    
    # Show initial stats
    stats = chat.get_corpus_stats()
    print(f"Corpus loaded: {stats['pairs']} pairs, {stats['dimensions']} dimensions, "
          f"{stats['concepts']} concepts")
    print()
    
    # Interactive loop
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        # Handle commands
        if query.startswith("/"):
            cmd = query.lower()
            
            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            
            elif cmd == "/help":
                chat.print_help()
            
            elif cmd == "/stats":
                stats = chat.get_corpus_stats()
                print(f"\nCorpus Statistics:")
                for key, value in stats.items():
                    if key != "dimension_names":
                        print(f"  {key}: {value}")
                print()
            
            elif cmd == "/concepts":
                concepts = list(chat.corpus.concepts.keys())
                print(f"\nKnown concepts ({len(concepts)}):")
                # Group by first letter
                for i in range(0, len(concepts), 10):
                    print("  " + ", ".join(concepts[i:i+10]))
                print()
            
            elif cmd == "/dims":
                dims = list(chat.corpus.dimensions.keys())
                print(f"\nDimensions ({len(dims)}):")
                for dim in dims:
                    print(f"  - {dim}")
                print()
            
            elif cmd == "/ideals":
                ideals = chat.corpus.list_ideals()
                print(f"\nPlatonic Ideals ({len(ideals)}):")
                for ideal in ideals:
                    info = chat.corpus.get_ideal(ideal)
                    if info:
                        print(f"  - {ideal}: {len(info.dimensions_anchored)} dimensions")
                print()
            
            else:
                print(f"Unknown command: {cmd}. Type /help for commands.")
            
            continue
        
        # Process query
        response = chat.chat(query)
        print(f"Bot: {response}")
        print()


if __name__ == "__main__":
    main()
