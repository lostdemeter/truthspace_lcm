#!/usr/bin/env python3
"""
Scalable Emergent Chatbot

Combines all the emergent systems into a unified chatbot:
1. Fully emergent chains (stopwords, labels, templates from data)
2. Segmented rebalancing (anchors + compartments like DNA zinc fingers)
3. Feedback loop (LLM corrects outputs, backpropagates)
4. Inject/rebalance cycle (adds new concepts without disruption)

The chatbot learns continuously while maintaining structural stability.
"""

import json
import numpy as np
import requests
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from queue import Queue
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fully_emergent_chains import FullyEmergentSemanticChain
from experiments.segmented_rebalance import SegmentedStructure, Anchor


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"


@dataclass
class LearningEvent:
    """An event that triggers learning."""
    event_type: str  # 'query', 'feedback', 'injection'
    concept: str
    data: Dict
    timestamp: float = field(default_factory=time.time)


class ScalableChatbot:
    """
    A chatbot that scales through:
    - Emergent structure discovery
    - Segmented rebalancing (local updates, global stability)
    - Continuous learning from interactions
    """
    
    def __init__(self):
        # Core emergent chain
        self.semantic = FullyEmergentSemanticChain()
        
        # Segmented structure manager
        self.structure: Optional[SegmentedStructure] = None
        
        # Learning queue for background processing
        self.learning_queue: Queue = Queue()
        self.learning_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Stats
        self.queries_processed = 0
        self.concepts_learned = 0
        self.rebalances_performed = 0
        
        # Known relationships (persisted)
        self.known_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Injection queue
        self.injection_queue: List[Dict] = []
        
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call Ollama API."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            pass
        return None
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def load_corpus(self, corpus_path: str) -> int:
        """Load initial corpus."""
        count = self.semantic.ingest_corpus(corpus_path)
        return count
    
    def initialize(self):
        """Initialize all emergent structures."""
        print("Initializing emergent structures...")
        
        # Learn dimensions
        self.semantic.learn_dimensions()
        print(f"  Dimensions: {len(self.semantic.dimensions)}")
        print(f"  Concepts: {len(self.semantic.groups)}")
        print(f"  Stopwords: {len(self.semantic.stopword_chain.discovered_stopwords)}")
        print(f"  Verb labels: {len(self.semantic.label_chain.verb_labels)}")
        print(f"  Templates: {len(self.semantic.template_chain.templates)}")
        
        # Create segmented structure
        self.structure = SegmentedStructure(self.semantic)
        
        # Discover anchors
        print("\nDiscovering anchors...")
        self.structure.discover_anchors(min_confidence=0.7)
        
        # Discover segments
        print("\nDiscovering segments...")
        self.structure.discover_segments()
        
        print(f"\nInitialization complete:")
        print(f"  Anchors: {len(self.structure.anchors)}")
        print(f"  Segments: {len(self.structure.segments)}")
    
    # =========================================================================
    # CHAT INTERFACE
    # =========================================================================
    
    def chat(self, query: str) -> str:
        """Process a query and return a response."""
        self.queries_processed += 1
        
        # Extract concepts from query
        concepts = self._extract_concepts(query)
        
        # Detect intent
        intent = self._detect_intent(query)
        
        # Queue learning event
        self.learning_queue.put(LearningEvent(
            event_type='query',
            concept=concepts[0] if concepts else '',
            data={'query': query, 'concepts': concepts, 'intent': intent}
        ))
        
        # Generate response
        if not concepts:
            return self._handle_unknown_query(query)
        
        if intent == 'compare' and len(concepts) >= 2:
            return self._respond_compare(concepts[0], concepts[1])
        elif intent == 'similar':
            return self._respond_similar(concepts[0])
        elif intent == 'opposite':
            return self._respond_opposite(concepts[0])
        elif intent == 'learn':
            return self._respond_learn(query, concepts)
        else:
            return self._respond_describe(concepts[0])
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract known concepts from query."""
        query_lower = query.lower()
        found = []
        for g in self.semantic.groups:
            if g in query_lower and len(g) > 2:
                found.append(g)
        # Sort by length (prefer longer matches)
        return sorted(found, key=len, reverse=True)
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent."""
        q = query.lower()
        if any(w in q for w in ['compare', 'difference', 'between', 'vs']):
            return 'compare'
        if any(w in q for w in ['similar', 'like', 'related']):
            return 'similar'
        if any(w in q for w in ['opposite', 'contrary', 'antonym']):
            return 'opposite'
        if any(w in q for w in ['learn', 'teach', 'remember', 'know that']):
            return 'learn'
        return 'describe'
    
    def _format_name(self, name: str) -> str:
        return name.replace('_', ' ').title()
    
    def _format_list(self, items: List[str]) -> str:
        items = [self._format_name(i) for i in items]
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"
    
    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================
    
    def _respond_describe(self, concept: str) -> str:
        """Generate description response."""
        name = self._format_name(concept)
        
        # Get traits
        traits = self.semantic.describe_traits(concept)
        
        # Get similar
        similar = self.semantic.find_similar(concept, k=3)
        
        # Get opposite
        opposite = self.semantic.find_opposite(concept)
        
        # Check if we have a known relationship
        known_opp = self.known_relationships.get(concept, {}).get('opposite')
        if known_opp:
            opposite = (known_opp, 0.0)
        
        # Get content
        content = self.semantic.get_relevant_content([concept], k=2)
        
        parts = []
        
        if traits:
            parts.append(f"{name} exhibits {self._format_list(traits)} qualities.")
        
        if similar:
            similar_names = self._format_list([s[0] for s in similar])
            parts.append(f"{name} shares characteristics with {similar_names}.")
        
        if opposite:
            opp_name = self._format_name(opposite[0])
            # Check if this is an anchored relationship
            is_anchored = any(
                (a.concept_a == concept and a.concept_b == opposite[0]) or
                (a.concept_b == concept and a.concept_a == opposite[0])
                for a in (self.structure.anchors if self.structure else [])
            )
            anchor_mark = " (anchored)" if is_anchored else ""
            parts.append(f"In contrast, {name} is opposite to {opp_name}{anchor_mark}.")
        
        if content:
            parts.append("")
            parts.append("From knowledge base:")
            for text in content[:2]:
                text_short = text[:100] + "..." if len(text) > 100 else text
                parts.append(f"  • {text_short}")
        
        return '\n'.join(parts) if parts else f"{name} is a known concept."
    
    def _respond_compare(self, c1: str, c2: str) -> str:
        """Generate comparison response."""
        n1, n2 = self._format_name(c1), self._format_name(c2)
        
        pos1 = self.semantic.get_position(c1)
        pos2 = self.semantic.get_position(c2)
        
        if pos1 is None or pos2 is None:
            return f"Cannot compare: missing data."
        
        dist = float(np.linalg.norm(pos2 - pos1))
        
        if dist < 0.3:
            sim = "very closely related"
        elif dist < 0.6:
            sim = "somewhat similar"
        elif dist < 1.0:
            sim = "notably different"
        else:
            sim = "quite distinct"
        
        parts = [f"{n1} and {n2} are {sim}."]
        
        # Get traits for each
        traits1 = self.semantic.describe_traits(c1)
        traits2 = self.semantic.describe_traits(c2)
        
        if traits1:
            parts.append(f"{n1} is characterized by {self._format_list(traits1)} qualities.")
        if traits2:
            parts.append(f"{n2} is characterized by {self._format_list(traits2)} qualities.")
        
        return '\n'.join(parts)
    
    def _respond_similar(self, concept: str) -> str:
        """Generate similarity response."""
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
        """Generate opposite response."""
        name = self._format_name(concept)
        
        # Check known relationships first
        known_opp = self.known_relationships.get(concept, {}).get('opposite')
        if known_opp:
            return f"The opposite of {name} is {self._format_name(known_opp)} (learned)."
        
        result = self.semantic.find_opposite(concept)
        if result:
            opp_name = self._format_name(result[0])
            
            # Evaluate if this seems right
            confidence = self._evaluate_opposite_confidence(concept, result[0])
            
            if confidence < 0.5:
                # Queue for learning
                self._queue_opposite_learning(concept, result[0])
                return f"The opposite of {name} might be {opp_name}, but I'm not confident. I'll learn more about this."
            
            return f"The opposite of {name} is {opp_name}."
        
        return f"No clear opposite found for {name}."
    
    def _respond_learn(self, query: str, concepts: List[str]) -> str:
        """Handle learning requests."""
        # Parse learning intent
        q = query.lower()
        
        # "X is the opposite of Y"
        if 'opposite' in q and len(concepts) >= 2:
            c1, c2 = concepts[0], concepts[1]
            self._learn_relationship(c1, c2, 'opposite')
            return f"Learned: {self._format_name(c1)} is opposite to {self._format_name(c2)}."
        
        # "X is similar to Y"
        if 'similar' in q and len(concepts) >= 2:
            c1, c2 = concepts[0], concepts[1]
            self._learn_relationship(c1, c2, 'similar')
            return f"Learned: {self._format_name(c1)} is similar to {self._format_name(c2)}."
        
        return "I can learn relationships. Try: 'Learn that X is the opposite of Y'"
    
    def _handle_unknown_query(self, query: str) -> str:
        """Handle queries with no recognized concepts."""
        # Try to extract potential new concepts
        words = query.lower().split()
        potential = [w for w in words if len(w) > 3 and w.isalpha()]
        
        if potential:
            # Queue for potential injection
            for word in potential[:2]:
                self.injection_queue.append({
                    'concept': word,
                    'reason': f'mentioned in query: {query[:50]}',
                    'source': 'user_query'
                })
            
            sample = ', '.join([self._format_name(g) for g in self.semantic.groups[:8]])
            return f"I don't recognize those concepts yet, but I've noted them for learning.\n\nKnown concepts: {sample}..."
        
        sample = ', '.join([self._format_name(g) for g in self.semantic.groups[:8]])
        return f"I don't recognize any concepts in your query.\n\nKnown concepts: {sample}..."
    
    # =========================================================================
    # LEARNING SYSTEM
    # =========================================================================
    
    def _evaluate_opposite_confidence(self, c1: str, c2: str) -> float:
        """Quick confidence check for opposite relationship."""
        # Check if it's an anchor
        if self.structure:
            for anchor in self.structure.anchors:
                if (anchor.concept_a == c1 and anchor.concept_b == c2) or \
                   (anchor.concept_b == c1 and anchor.concept_a == c2):
                    return anchor.confidence
        
        # Check known relationships
        if self.known_relationships.get(c1, {}).get('opposite') == c2:
            return 0.9
        
        # Default moderate confidence
        return 0.6
    
    def _queue_opposite_learning(self, concept: str, current_opposite: str):
        """Queue a concept for opposite learning."""
        self.learning_queue.put(LearningEvent(
            event_type='feedback',
            concept=concept,
            data={'current_opposite': current_opposite, 'type': 'opposite'}
        ))
    
    def _learn_relationship(self, c1: str, c2: str, rel_type: str):
        """Learn a relationship between concepts."""
        self.known_relationships[c1][rel_type] = c2
        self.known_relationships[c2][rel_type] = c1
        
        # Add as anchor if structure exists
        if self.structure and rel_type == 'opposite':
            self.structure.add_anchor(c1, c2, rel_type, confidence=0.95)
        
        self.concepts_learned += 1
    
    def process_learning_queue(self, max_items: int = 5):
        """Process items in the learning queue."""
        processed = 0
        
        while not self.learning_queue.empty() and processed < max_items:
            event = self.learning_queue.get()
            
            if event.event_type == 'feedback':
                self._process_feedback_event(event)
            elif event.event_type == 'query':
                # Just log for now
                pass
            
            processed += 1
        
        # Process injection queue
        if self.injection_queue:
            self._process_injections(max_items=2)
        
        # Periodic rebalancing
        if self.queries_processed % 10 == 0 and self.structure:
            self._do_segmented_rebalance()
    
    def _process_feedback_event(self, event: LearningEvent):
        """Process a feedback learning event."""
        if event.data.get('type') == 'opposite':
            concept = event.concept
            current = event.data.get('current_opposite')
            
            # Ask LLM for better opposite
            vocab = [g.replace('_', ' ').title() for g in self.semantic.groups[:30]]
            prompt = f"""What is the best semantic opposite of "{concept.title()}"?

Choose from: {', '.join(vocab)}

Answer with just one word from the list:"""

            response = self._call_llm(prompt, max_tokens=20)
            if response:
                better = response.strip().lower().split()[0].replace(' ', '_')
                if better in self.semantic.groups and better != current:
                    self._learn_relationship(concept, better, 'opposite')
                    print(f"  Learned: {concept} ↔ {better}")
    
    def _process_injections(self, max_items: int = 2):
        """Process injection queue."""
        for item in self.injection_queue[:max_items]:
            concept = item['concept']
            
            # Generate data for new concept
            prompt = f"""Generate 3 behavioral sentences for "{concept.title()}".

Rules:
1. Start each sentence with "{concept.title()}"
2. Second word should be a verb
3. Keep sentences 8-15 words

Generate 3 sentences:"""

            response = self._call_llm(prompt, max_tokens=200)
            if response:
                for line in response.strip().split('\n'):
                    line = line.strip().lstrip('0123456789.-) ')
                    if len(line) > 15:
                        self.semantic.ingest_item({
                            'text': line,
                            'agent': concept.lower(),
                            'source': 'injection'
                        })
                
                self.concepts_learned += 1
                print(f"  Injected: {concept}")
        
        self.injection_queue = self.injection_queue[max_items:]
    
    def _do_segmented_rebalance(self):
        """Perform segmented rebalancing."""
        if not self.structure:
            return
        
        self.structure.rebalance_all_segments()
        self.rebalances_performed += 1
    
    # =========================================================================
    # BACKGROUND LEARNING
    # =========================================================================
    
    def start_background_learning(self):
        """Start background learning thread."""
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        print("Background learning started.")
    
    def stop_background_learning(self):
        """Stop background learning thread."""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
        print("Background learning stopped.")
    
    def _learning_loop(self):
        """Background learning loop."""
        while self.running:
            try:
                self.process_learning_queue(max_items=3)
            except Exception as e:
                print(f"Learning error: {e}")
            time.sleep(5)  # Process every 5 seconds
    
    # =========================================================================
    # INTERACTIVE SESSION
    # =========================================================================
    
    def interactive(self):
        """Run interactive chat session."""
        print("\n" + "═" * 70)
        print(" SCALABLE EMERGENT CHATBOT ".center(70, "═"))
        print(" Continuous Learning + Segmented Rebalancing ".center(70))
        print("═" * 70)
        
        print(f"\nKnowledge: {len(self.semantic.items)} items, {len(self.semantic.groups)} concepts")
        print(f"Structure: {len(self.structure.anchors) if self.structure else 0} anchors, "
              f"{len(self.structure.segments) if self.structure else 0} segments")
        print(f"Emergent: {len(self.semantic.stopword_chain.discovered_stopwords)} stopwords, "
              f"{len(self.semantic.label_chain.verb_labels)} verb labels")
        
        print("\nCommands: 'stats', 'anchors', 'learn', 'rebalance', 'quit'\n")
        
        # Start background learning
        self.start_background_learning()
        
        try:
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
                    print(f"\n  Queries: {self.queries_processed}")
                    print(f"  Concepts learned: {self.concepts_learned}")
                    print(f"  Rebalances: {self.rebalances_performed}")
                    print(f"  Known relationships: {sum(len(v) for v in self.known_relationships.values())}")
                    print(f"  Injection queue: {len(self.injection_queue)}\n")
                    continue
                
                if query.lower() == 'anchors':
                    if self.structure:
                        print("\nAnchors:")
                        for a in self.structure.anchors:
                            print(f"  {a.concept_a} ↔ {a.concept_b} (conf={a.confidence:.2f})")
                    print()
                    continue
                
                if query.lower() == 'rebalance':
                    self._do_segmented_rebalance()
                    print("  Rebalance complete.\n")
                    continue
                
                response = self.chat(query)
                print(f"\nBot: {response}\n")
        
        finally:
            self.stop_background_learning()


def main():
    """Main entry point."""
    print("=" * 70)
    print("SCALABLE EMERGENT CHATBOT")
    print("=" * 70)
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama not running!")
            return
        print("Ollama is running")
    except:
        print("Ollama not available!")
        return
    
    # Create chatbot
    bot = ScalableChatbot()
    
    # Load corpus
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    print(f"\nLoading corpus: {corpus_path}")
    count = bot.load_corpus(str(corpus_path))
    print(f"  Loaded {count} items")
    
    # Initialize
    bot.initialize()
    
    # Run interactive session
    bot.interactive()


if __name__ == "__main__":
    main()
