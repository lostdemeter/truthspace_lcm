"""
Emergent Chatbot

A chatbot powered by dual emergent gear chains:
1. SemanticChain - understands queries via discovered dimensions
2. LinguisticChain - conditions output via sentence patterns

NO LLM is used for responses - only for initial corpus generation.

Author: Lesley Gushurst
License: GPLv3
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from ..core import SemanticChain, LinguisticChain


class EmergentChatbot:
    """
    A chatbot with dual emergent gear chains.
    
    Uses SemanticChain for understanding and LinguisticChain for output.
    Both chains discover their own dimensions from data.
    """
    
    def __init__(self):
        self.semantic = SemanticChain("Understanding")
        self.linguistic = LinguisticChain("Output")
        self.corpus_loaded = False
    
    def load_corpus(self, corpus_path: str) -> int:
        """
        Load a corpus into both chains.
        
        Args:
            corpus_path: Path to JSON corpus file
            
        Returns:
            Number of items loaded
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        # Load into both chains
        semantic_count = self.semantic.ingest_corpus(str(path))
        linguistic_count = self.linguistic.ingest_corpus(str(path))
        
        self.corpus_loaded = True
        return semantic_count
    
    def load_corpora(self, corpus_paths: List[str]) -> int:
        """Load multiple corpora."""
        total = 0
        for path in corpus_paths:
            if Path(path).exists():
                total += self.load_corpus(path)
        return total
    
    def train(self, 
              semantic_min_var: float = 0.02,
              semantic_max_dims: int = 12,
              linguistic_min_var: float = 0.03,
              linguistic_max_dims: int = 8) -> Dict[str, int]:
        """
        Train both chains to discover dimensions.
        
        Returns:
            Dict with dimension counts for each chain
        """
        semantic_dims = self.semantic.learn_dimensions(
            min_variance=semantic_min_var,
            max_dims=semantic_max_dims
        )
        
        linguistic_dims = self.linguistic.learn_dimensions(
            min_variance=linguistic_min_var,
            max_dims=linguistic_max_dims
        )
        
        return {
            'semantic_dimensions': semantic_dims,
            'linguistic_dimensions': linguistic_dims,
        }
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract known concepts from query."""
        query_lower = query.lower()
        return [g for g in self.semantic.groups if g in query_lower and len(g) > 2]
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent."""
        q = query.lower()
        if any(w in q for w in ['compare', 'difference', 'between', 'vs']):
            return 'compare'
        if any(w in q for w in ['similar', 'like', 'related']):
            return 'similar'
        if any(w in q for w in ['opposite', 'contrary']):
            return 'opposite'
        return 'describe'
    
    def _format_name(self, name: str) -> str:
        """Format a name for display."""
        return name.replace('_', ' ').title()
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list naturally."""
        items = [self._format_name(i) for i in items]
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"
    
    def chat(self, query: str) -> str:
        """
        Process a query and generate a response.
        
        Args:
            query: User's question
            
        Returns:
            Natural language response
        """
        concepts = self._extract_concepts(query)
        intent = self._detect_intent(query)
        
        if not concepts:
            sample = ', '.join([self._format_name(g) for g in self.semantic.groups[:8]])
            return f"I don't recognize any concepts in your query.\n\nKnown concepts: {sample}..."
        
        # Get relevant content
        content = self.semantic.get_relevant_content(concepts, k=3)
        
        # Generate response based on intent
        if intent == 'compare' and len(concepts) >= 2:
            return self._respond_compare(concepts[0], concepts[1], content)
        elif intent == 'similar':
            return self._respond_similar(concepts[0])
        elif intent == 'opposite':
            return self._respond_opposite(concepts[0])
        else:
            return self._respond_describe(concepts[0], content)
    
    def _respond_describe(self, concept: str, content: List[str]) -> str:
        """Generate a description response."""
        name = self._format_name(concept)
        analysis = self.semantic.analyze(concept)
        
        parts = []
        
        # Semantic traits (from feature labels, not pole names)
        if analysis.get('traits'):
            trait_str = self._format_list(analysis['traits'])
            parts.append(f"{name} exhibits {trait_str} qualities.")
        
        # Similar concepts
        if analysis['similar']:
            similar_names = self._format_list([s[0] for s in analysis['similar'][:3]])
            parts.append(f"{name} shares characteristics with {similar_names}.")
        
        # Opposite
        if analysis['opposite']:
            opp_name = self._format_name(analysis['opposite'][0])
            parts.append(f"In contrast, {name} differs notably from {opp_name}.")
        
        # Content evidence
        if content:
            parts.append("")
            parts.append("From the knowledge base:")
            for text in content[:2]:
                text_short = text[:100] + "..." if len(text) > 100 else text
                parts.append(f"  • {text_short}")
        
        return '\n'.join(parts) if parts else f"{name} is a known concept."
    
    def _respond_compare(self, c1: str, c2: str, content: List[str]) -> str:
        """Generate a comparison response."""
        n1, n2 = self._format_name(c1), self._format_name(c2)
        
        pos1 = self.semantic.get_position(c1)
        pos2 = self.semantic.get_position(c2)
        
        if pos1 is None or pos2 is None:
            return f"Cannot compare: missing data for {n1 if pos1 is None else n2}."
        
        dist = float(np.linalg.norm(pos2 - pos1))
        
        if dist < 0.3:
            sim = "closely related"
        elif dist < 0.6:
            sim = "somewhat similar"
        elif dist < 1.0:
            sim = "notably different"
        else:
            sim = "quite distinct"
        
        parts = [f"{n1} and {n2} are {sim}."]
        
        # Find key difference using semantic labels
        diff = pos2 - pos1
        max_idx = int(np.argmax(np.abs(diff)))
        if max_idx < len(self.semantic.dimensions):
            dim = self.semantic.dimensions[max_idx]
            neg_label, pos_label = self.semantic.get_dimension_labels(dim.name)
            
            if diff[max_idx] > 0:
                trait1 = neg_label
                trait2 = pos_label
            else:
                trait1 = pos_label
                trait2 = neg_label
            
            parts.append(f"Where {n1} tends toward {trait1}, {n2} leans toward {trait2}.")
        
        if content:
            parts.append("")
            parts.append("Evidence:")
            for text in content[:2]:
                text_short = text[:80] + "..." if len(text) > 80 else text
                parts.append(f"  • {text_short}")
        
        return '\n'.join(parts)
    
    def _respond_similar(self, concept: str) -> str:
        """Generate a similarity response."""
        name = self._format_name(concept)
        similar = self.semantic.find_similar(concept, k=5)
        
        if not similar:
            return f"No similar concepts found for {name}."
        
        parts = [f"Concepts similar to {name}:"]
        for other, dist in similar:
            closeness = "very close" if dist < 0.3 else "fairly close" if dist < 0.6 else "related"
            parts.append(f"  • {self._format_name(other)} ({closeness})")
        
        return '\n'.join(parts)
    
    def _respond_opposite(self, concept: str) -> str:
        """Generate an opposite response."""
        name = self._format_name(concept)
        result = self.semantic.find_opposite(concept)
        
        if result:
            opposite, dist = result
            return f"The opposite of {name} is {self._format_name(opposite)}."
        return f"No clear opposite found for {name}."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics."""
        return {
            'semantic_items': len(self.semantic.items),
            'semantic_groups': len(self.semantic.groups),
            'semantic_dimensions': len(self.semantic.dimensions),
            'linguistic_items': len(self.linguistic.items),
            'linguistic_dimensions': len(self.linguistic.dimensions),
        }
    
    def interactive(self):
        """Run interactive chat session."""
        stats = self.get_stats()
        
        print("\n" + "═" * 70)
        print(" EMERGENT CHATBOT ".center(70, "═"))
        print(" Dual Gear Chain: Semantic + Linguistic ".center(70))
        print("═" * 70)
        print(f"\nKnowledge: {stats['semantic_items']} items, {stats['semantic_groups']} concepts")
        print(f"Semantic dimensions: {stats['semantic_dimensions']}")
        print(f"Linguistic dimensions: {stats['linguistic_dimensions']}")
        print("\nCommands: 'dims', 'stats', 'quit'\n")
        
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not query:
                continue
            if query.lower() == 'quit':
                break
            if query.lower() == 'stats':
                for k, v in self.get_stats().items():
                    print(f"  {k}: {v}")
                continue
            if query.lower() == 'dims':
                print("\nSemantic dimensions:")
                for d in self.semantic.dimensions[:5]:
                    print(f"  {d.name}: {d.negative_pole} ↔ {d.positive_pole}")
                print("\nLinguistic dimensions:")
                for d in self.linguistic.dimensions[:5]:
                    print(f"  {d.name}: {d.negative_features} ↔ {d.positive_features}")
                continue
            
            print(f"\n{self.chat(query)}\n")


def create_chatbot(corpus_paths: List[str] = None) -> EmergentChatbot:
    """
    Factory function to create and initialize a chatbot.
    
    Args:
        corpus_paths: List of corpus file paths (optional)
        
    Returns:
        Initialized EmergentChatbot
    """
    bot = EmergentChatbot()
    
    if corpus_paths:
        bot.load_corpora(corpus_paths)
        bot.train()
    
    return bot


def main():
    """Run the chatbot with default corpora."""
    base = Path(__file__).parent.parent
    
    corpus_paths = [
        str(base / "corpus" / "corpus_llm_live.json"),
        str(base / "corpus" / "corpus_knowledge.json"),
    ]
    
    # Also check parent directory for additional corpora
    parent_base = base.parent
    additional = [
        str(parent_base / "corpus_curated.json"),
    ]
    
    all_paths = corpus_paths + [p for p in additional if Path(p).exists()]
    
    print("Creating Emergent Chatbot...")
    bot = create_chatbot(all_paths)
    
    print("\n" + "─" * 70)
    print("TEST QUERIES")
    print("─" * 70)
    
    tests = [
        "Tell me about Holmes",
        "Compare Holmes and Watson",
        "What is similar to villain?",
    ]
    
    for q in tests:
        print(f"\n>>> {q}")
        print(bot.chat(q))
    
    bot.interactive()


if __name__ == "__main__":
    main()
