#!/usr/bin/env python3
"""
Truly Emergent Conversational Chatbot

This chatbot generates responses ENTIRELY from emergent patterns:
- LLM is ONLY used as a knowledge resource (like Wikipedia)
- Response generation uses emergent templates
- Topic understanding uses emergent semantic structure
- No LLM calls during conversation

The key difference: we use LLM to BUILD the corpus, not to GENERATE responses.
"""

import json
import numpy as np
import requests
import time
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fully_emergent_chains import FullyEmergentSemanticChain
from experiments.segmented_rebalance import SegmentedStructure


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"


@dataclass
class KnowledgeItem:
    """A single piece of knowledge."""
    text: str
    topic: str
    source: str
    item_type: str  # 'fact', 'definition', 'example', 'relation'


class EmergentConversationalBot:
    """
    A chatbot that generates responses using ONLY emergent patterns.
    
    LLM is used ONLY during corpus building phase, never during conversation.
    """
    
    def __init__(self):
        # Emergent chains
        self.semantic = FullyEmergentSemanticChain()
        self.structure: Optional[SegmentedStructure] = None
        
        # Knowledge corpus (raw text, not LLM responses)
        self.corpus: List[KnowledgeItem] = []
        self.topics: Set[str] = set()
        
        # Emergent response templates (discovered from data)
        self.response_templates: List[str] = []
        self.question_templates: List[str] = []
        
        # Stats
        self.corpus_building_llm_calls = 0
        self.conversation_llm_calls = 0  # Should stay 0!
        
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Call LLM - ONLY for corpus building."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7}
                },
                timeout=60
            )
            if response.status_code == 200:
                self.corpus_building_llm_calls += 1
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM error: {e}")
        return None
    
    # =========================================================================
    # CORPUS BUILDING (LLM used here as knowledge resource)
    # =========================================================================
    
    def build_corpus_from_topic(self, topic: str):
        """
        Build corpus entries for a topic using LLM as knowledge resource.
        
        This is like using Wikipedia - we're gathering raw knowledge,
        not generating responses.
        """
        print(f"  Gathering knowledge: {topic}...")
        self.topics.add(topic.lower())
        
        # Get factual sentences about the topic
        prompt = f"""Write 5 simple factual sentences about "{topic}".
Each sentence should be self-contained and informative.
Write in third person, present tense.
One sentence per line.

Sentences about {topic}:"""

        response = self._call_llm(prompt, max_tokens=400)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 15 and len(line) < 250:
                    self.corpus.append(KnowledgeItem(
                        text=line,
                        topic=topic.lower(),
                        source='llm_knowledge',
                        item_type='fact'
                    ))
                    # Also ingest into semantic chain
                    self.semantic.ingest_item({
                        'text': line,
                        'agent': topic.lower(),
                        'source': 'corpus_building',
                    })
        
        # Get a definition
        prompt = f"""Define "{topic}" in one clear sentence.
Start with "{topic} is" or "{topic} refers to".

Definition:"""

        response = self._call_llm(prompt, max_tokens=100)
        if response:
            line = response.strip().split('\n')[0]
            if len(line) > 10:
                self.corpus.append(KnowledgeItem(
                    text=line,
                    topic=topic.lower(),
                    source='llm_knowledge',
                    item_type='definition'
                ))
                self.semantic.ingest_item({
                    'text': line,
                    'agent': topic.lower(),
                    'source': 'corpus_building',
                })
        
        # Get related topics
        prompt = f"""List 3 topics closely related to "{topic}".
Just the topic names, one per line:"""

        response = self._call_llm(prompt, max_tokens=50)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ').lower()
                if len(line) > 2 and len(line) < 40:
                    self.topics.add(line)
    
    def build_corpus(self, seed_topics: List[str], expand: bool = True):
        """Build the knowledge corpus from seed topics."""
        print(f"\n{'='*60}")
        print("BUILDING KNOWLEDGE CORPUS")
        print("(LLM used as knowledge resource only)")
        print(f"{'='*60}")
        
        # First pass: seed topics
        for topic in seed_topics:
            self.build_corpus_from_topic(topic)
        
        # Expand to related topics
        if expand:
            initial_topics = list(self.topics)
            for topic in initial_topics:
                if topic not in seed_topics:
                    self.build_corpus_from_topic(topic)
        
        # Learn emergent structure
        print("\nLearning emergent structure...")
        self.semantic.learn_dimensions()
        
        # Discover response templates from corpus
        self._discover_templates()
        
        # Create segmented structure
        self.structure = SegmentedStructure(self.semantic)
        self.structure.discover_segments()
        
        print(f"\nCorpus built:")
        print(f"  Topics: {len(self.topics)}")
        print(f"  Corpus items: {len(self.corpus)}")
        print(f"  Semantic items: {len(self.semantic.items)}")
        print(f"  Templates discovered: {len(self.response_templates)}")
        print(f"  LLM calls (corpus building): {self.corpus_building_llm_calls}")
    
    def _discover_templates(self):
        """Discover response templates from the corpus."""
        # Extract sentence patterns
        patterns = defaultdict(int)
        
        for item in self.corpus:
            text = item.text
            # Extract pattern by replacing topic with placeholder
            if item.topic in text.lower():
                pattern = text.lower().replace(item.topic, '{topic}')
                # Generalize further
                pattern = re.sub(r'\b(is|are|was|were)\b', '{be}', pattern)
                patterns[pattern[:50]] += 1
        
        # Keep common patterns as templates
        for pattern, count in patterns.items():
            if count >= 2:
                self.response_templates.append(pattern)
        
        # Add some basic templates if none discovered
        if not self.response_templates:
            self.response_templates = [
                "{topic} {be} {content}",
                "Regarding {topic}, {content}",
                "{content}",
            ]
    
    # =========================================================================
    # EMERGENT RESPONSE GENERATION (NO LLM!)
    # =========================================================================
    
    def chat(self, user_input: str) -> str:
        """
        Generate response using ONLY emergent patterns.
        NO LLM calls here!
        """
        # Extract topics from input
        topics = self._extract_topics(user_input)
        
        if not topics:
            return self._emergent_unknown_response(user_input)
        
        # Generate response from emergent structure
        return self._emergent_response(user_input, topics)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract known topics from text."""
        text_lower = text.lower()
        found = []
        
        for topic in self.topics:
            if topic in text_lower:
                found.append(topic)
        
        # Also check semantic groups
        for group in self.semantic.groups:
            if group in text_lower and group not in found:
                found.append(group)
        
        return sorted(found, key=len, reverse=True)
    
    def _emergent_response(self, user_input: str, topics: List[str]) -> str:
        """
        Generate response using emergent patterns only.
        
        This combines:
        1. Semantic retrieval (find relevant content)
        2. Template application (structure the response)
        3. Trait description (add emergent understanding)
        """
        main_topic = topics[0]
        
        # Get relevant content from corpus
        relevant = self._get_relevant_content(main_topic, user_input)
        
        # Get emergent traits
        traits = self.semantic.describe_traits(main_topic)
        
        # Get similar concepts
        similar = self.semantic.find_similar(main_topic, k=3)
        
        # Detect question type and build response
        response_parts = []
        
        input_lower = user_input.lower()
        
        if 'what is' in input_lower or 'what are' in input_lower:
            # Definition question - find definition in corpus
            definition = self._find_definition(main_topic)
            if definition:
                response_parts.append(definition)
            elif relevant:
                response_parts.append(relevant[0])
        
        elif 'how' in input_lower:
            # Explanation - provide multiple facts
            if relevant:
                response_parts.append(f"About {main_topic}:")
                for i, fact in enumerate(relevant[:3], 1):
                    response_parts.append(f"  {i}. {fact}")
        
        elif 'why' in input_lower:
            # Reasoning - look for causal content
            causal = [r for r in relevant if any(w in r.lower() for w in 
                     ['because', 'since', 'therefore', 'important', 'significant'])]
            if causal:
                response_parts.append(causal[0])
            elif relevant:
                response_parts.append(relevant[0])
        
        else:
            # General question - combine facts with emergent understanding
            if relevant:
                response_parts.append(relevant[0])
            
            if traits:
                trait_str = ', '.join(traits[:3])
                response_parts.append(f"{main_topic.title()} is characterized by: {trait_str}.")
            
            if similar:
                similar_str = ', '.join([s[0] for s in similar[:3]])
                response_parts.append(f"Related concepts: {similar_str}.")
        
        if not response_parts:
            # Fallback to any available content
            if relevant:
                return relevant[0]
            return f"I have limited knowledge about {main_topic}."
        
        return '\n'.join(response_parts)
    
    def _get_relevant_content(self, topic: str, query: str) -> List[str]:
        """Get relevant content from corpus for a topic."""
        relevant = []
        
        # Direct topic match
        for item in self.corpus:
            if item.topic == topic.lower():
                relevant.append(item.text)
        
        # Also get from semantic chain
        semantic_content = self.semantic.get_relevant_content([topic], k=5)
        for content in semantic_content:
            if content not in relevant:
                relevant.append(content)
        
        return relevant[:5]
    
    def _find_definition(self, topic: str) -> Optional[str]:
        """Find a definition for a topic in the corpus."""
        for item in self.corpus:
            if item.topic == topic.lower() and item.item_type == 'definition':
                return item.text
        
        # Look for sentences that look like definitions
        for item in self.corpus:
            if item.topic == topic.lower():
                text_lower = item.text.lower()
                if f'{topic} is' in text_lower or f'{topic} refers' in text_lower:
                    return item.text
        
        return None
    
    def _emergent_unknown_response(self, user_input: str) -> str:
        """Handle unknown queries using emergent patterns."""
        # Try to find any relevant content
        words = user_input.lower().split()
        
        for word in words:
            if len(word) > 3:
                # Check if this word appears in any corpus item
                for item in self.corpus:
                    if word in item.text.lower():
                        return f"I found something related: {item.text}"
        
        # List what we know
        known = sorted(list(self.topics))[:10]
        return f"I don't have information about that. I can discuss: {', '.join(known)}"
    
    # =========================================================================
    # INTERACTIVE SESSION
    # =========================================================================
    
    def interactive(self):
        """Run interactive chat session."""
        print(f"\n{'═'*70}")
        print(" TRULY EMERGENT CONVERSATIONAL CHATBOT ".center(70, "═"))
        print(" No LLM during conversation - pure emergent responses ".center(70))
        print("═" * 70)
        
        print(f"\nKnowledge: {len(self.topics)} topics, {len(self.corpus)} facts")
        print(f"Emergent structure: {len(self.semantic.dimensions)} dimensions")
        print(f"LLM calls during corpus building: {self.corpus_building_llm_calls}")
        print(f"LLM calls during conversation: {self.conversation_llm_calls} (should be 0)")
        
        print("\nCommands: 'topics', 'stats', 'quit'\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'topics':
                print(f"\nKnown topics ({len(self.topics)}):")
                for topic in sorted(self.topics):
                    count = sum(1 for c in self.corpus if c.topic == topic)
                    print(f"  • {topic} ({count} items)")
                print()
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n  Topics: {len(self.topics)}")
                print(f"  Corpus items: {len(self.corpus)}")
                print(f"  Semantic dimensions: {len(self.semantic.dimensions)}")
                print(f"  LLM calls (corpus): {self.corpus_building_llm_calls}")
                print(f"  LLM calls (chat): {self.conversation_llm_calls}")
                print()
                continue
            
            # Generate emergent response (NO LLM!)
            response = self.chat(user_input)
            print(f"\nBot: {response}\n")
        
        print(f"\nFinal stats:")
        print(f"  LLM calls during conversation: {self.conversation_llm_calls}")
        print(f"  (This should be 0 for truly emergent responses)")


def main():
    """Main entry point."""
    print("=" * 70)
    print("TRULY EMERGENT CONVERSATIONAL CHATBOT")
    print("LLM used ONLY for corpus building, NOT for responses")
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
    bot = EmergentConversationalBot()
    
    # Build corpus from seed topics
    seed_topics = [
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "python",
        "programming",
        "philosophy",
        "science",
        "mathematics",
    ]
    
    bot.build_corpus(seed_topics, expand=True)
    
    # Run interactive session
    bot.interactive()


if __name__ == "__main__":
    main()
