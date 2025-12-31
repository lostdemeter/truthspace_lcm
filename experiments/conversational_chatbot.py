#!/usr/bin/env python3
"""
Conversational Chatbot with Emergent Knowledge Building

This chatbot can discuss topics in natural language by:
1. Building a knowledge corpus through LLM queries
2. Using emergent structure for topic understanding
3. Generating natural responses using learned patterns

The key difference from the book trainer is that we're building
CONVERSATIONAL knowledge, not just behavioral descriptions.
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
class TopicKnowledge:
    """Knowledge about a topic."""
    topic: str
    facts: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    example_questions: List[str] = field(default_factory=list)
    example_answers: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    user_input: str
    bot_response: str
    topics_used: List[str]
    timestamp: float = field(default_factory=time.time)


class ConversationalChatbot:
    """
    A chatbot that builds conversational knowledge through LLM queries
    and uses emergent structure for understanding.
    """
    
    def __init__(self):
        # Emergent semantic chain
        self.semantic = FullyEmergentSemanticChain()
        self.structure: Optional[SegmentedStructure] = None
        
        # Knowledge base
        self.topics: Dict[str, TopicKnowledge] = {}
        self.conversation_history: List[ConversationTurn] = []
        
        # Response templates (will be learned)
        self.response_templates = [
            "{fact}",
            "Regarding {topic}, {fact}",
            "{topic} is interesting because {fact}",
            "I know that {fact}",
            "From what I understand, {fact}",
            "Here's what I know about {topic}: {fact}",
        ]
        
        # Question patterns for knowledge building
        self.question_patterns = [
            "What is {topic}?",
            "Tell me about {topic}",
            "Explain {topic}",
            "How does {topic} work?",
            "Why is {topic} important?",
        ]
        
        # Stats
        self.queries_answered = 0
        self.topics_learned = 0
        self.llm_calls = 0
        
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Call Ollama API."""
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
                self.llm_calls += 1
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM error: {e}")
        return None
    
    # =========================================================================
    # KNOWLEDGE BUILDING
    # =========================================================================
    
    def learn_topic(self, topic: str, depth: int = 1) -> TopicKnowledge:
        """Learn about a topic by querying the LLM."""
        topic_lower = topic.lower().strip()
        
        # Check if we already know this topic
        if topic_lower in self.topics:
            return self.topics[topic_lower]
        
        print(f"  Learning about: {topic}...")
        
        knowledge = TopicKnowledge(topic=topic_lower)
        
        # Get basic facts
        prompt = f"""Provide 5 key facts about "{topic}". 
Keep each fact to one sentence.
Format as a numbered list.

Facts about {topic}:"""

        response = self._call_llm(prompt, max_tokens=400)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 20 and len(line) < 200:
                    knowledge.facts.append(line)
        
        # Get related topics
        prompt = f"""List 5 topics closely related to "{topic}".
Just list the topic names, one per line.

Related topics:"""

        response = self._call_llm(prompt, max_tokens=100)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 2 and len(line) < 50:
                    knowledge.related_topics.append(line.lower())
        
        # Get example Q&A pairs for this topic
        prompt = f"""Generate 3 example questions and answers about "{topic}".

Format:
Q: [question]
A: [answer]

Examples:"""

        response = self._call_llm(prompt, max_tokens=500)
        if response:
            lines = response.split('\n')
            current_q = None
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    current_q = line[2:].strip()
                elif line.startswith('A:') and current_q:
                    knowledge.example_questions.append(current_q)
                    knowledge.example_answers.append(line[2:].strip())
                    current_q = None
        
        # Store knowledge
        self.topics[topic_lower] = knowledge
        self.topics_learned += 1
        
        # Ingest facts into semantic chain
        for fact in knowledge.facts:
            self.semantic.ingest_item({
                'text': fact,
                'agent': topic_lower,
                'source': 'knowledge_building',
            })
        
        # Learn related topics at lower depth
        if depth > 0:
            for related in knowledge.related_topics[:2]:
                if related not in self.topics:
                    self.learn_topic(related, depth=depth-1)
        
        return knowledge
    
    def build_knowledge_base(self, seed_topics: List[str], depth: int = 1):
        """Build knowledge base from seed topics."""
        print(f"\n{'='*60}")
        print("BUILDING KNOWLEDGE BASE")
        print(f"{'='*60}")
        print(f"Seed topics: {', '.join(seed_topics)}")
        print(f"Depth: {depth}")
        
        for topic in seed_topics:
            self.learn_topic(topic, depth=depth)
        
        # Learn dimensions from accumulated knowledge
        if self.semantic.items:
            print("\nLearning semantic structure...")
            self.semantic.learn_dimensions()
            
            # Create segmented structure
            self.structure = SegmentedStructure(self.semantic)
            self.structure.discover_segments()
        
        print(f"\nKnowledge base built:")
        print(f"  Topics: {len(self.topics)}")
        print(f"  Total facts: {sum(len(t.facts) for t in self.topics.values())}")
        print(f"  Semantic items: {len(self.semantic.items)}")
        print(f"  LLM calls: {self.llm_calls}")
    
    # =========================================================================
    # CONVERSATION
    # =========================================================================
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        self.queries_answered += 1
        
        # Extract topics from input
        topics = self._extract_topics(user_input)
        
        # If no known topics, try to learn about the input
        if not topics:
            # Extract potential topic from input
            potential_topic = self._extract_potential_topic(user_input)
            if potential_topic:
                self.learn_topic(potential_topic, depth=0)
                topics = [potential_topic]
        
        if not topics:
            return self._handle_unknown(user_input)
        
        # Generate response based on topics
        response = self._generate_response(user_input, topics)
        
        # Store conversation turn
        self.conversation_history.append(ConversationTurn(
            user_input=user_input,
            bot_response=response,
            topics_used=topics
        ))
        
        return response
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract known topics from text."""
        text_lower = text.lower()
        found = []
        
        for topic in self.topics:
            if topic in text_lower:
                found.append(topic)
        
        # Sort by length (prefer longer matches)
        return sorted(found, key=len, reverse=True)
    
    def _extract_potential_topic(self, text: str) -> Optional[str]:
        """Extract a potential new topic from text."""
        # Remove common question words
        text = text.lower()
        for word in ['what', 'is', 'are', 'how', 'why', 'when', 'where', 
                     'who', 'tell', 'me', 'about', 'explain', 'the', 'a', 'an',
                     'do', 'does', 'can', 'could', 'would', 'should', '?', '.']:
            text = text.replace(word, ' ')
        
        # Get remaining words
        words = [w.strip() for w in text.split() if len(w.strip()) > 2]
        
        if words:
            # Return the longest word/phrase as potential topic
            return ' '.join(words[:3]).strip()
        
        return None
    
    def _generate_response(self, user_input: str, topics: List[str]) -> str:
        """Generate a natural language response."""
        main_topic = topics[0]
        knowledge = self.topics.get(main_topic)
        
        if not knowledge or not knowledge.facts:
            return f"I'm still learning about {main_topic}. Let me find out more."
        
        # Determine response type based on question
        input_lower = user_input.lower()
        
        # Check if this matches an example question
        for i, q in enumerate(knowledge.example_questions):
            if self._similar_question(input_lower, q.lower()):
                return knowledge.example_answers[i]
        
        # Generate response from facts
        if 'what' in input_lower and 'is' in input_lower:
            # Definition question
            return self._generate_definition_response(main_topic, knowledge)
        elif 'how' in input_lower:
            # Explanation question
            return self._generate_explanation_response(main_topic, knowledge)
        elif 'why' in input_lower:
            # Reasoning question
            return self._generate_reasoning_response(main_topic, knowledge)
        else:
            # General question
            return self._generate_general_response(main_topic, knowledge)
    
    def _similar_question(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar."""
        words1 = set(q1.split())
        words2 = set(q2.split())
        overlap = len(words1 & words2)
        return overlap >= min(3, len(words1) // 2)
    
    def _generate_definition_response(self, topic: str, knowledge: TopicKnowledge) -> str:
        """Generate a definition-style response."""
        if knowledge.facts:
            fact = knowledge.facts[0]
            return f"{topic.title()} - {fact}"
        return f"I don't have a clear definition for {topic} yet."
    
    def _generate_explanation_response(self, topic: str, knowledge: TopicKnowledge) -> str:
        """Generate an explanation response."""
        if len(knowledge.facts) >= 2:
            facts = knowledge.facts[:3]
            response = f"Here's how {topic} works:\n"
            for i, fact in enumerate(facts, 1):
                response += f"{i}. {fact}\n"
            return response.strip()
        elif knowledge.facts:
            return f"Regarding {topic}: {knowledge.facts[0]}"
        return f"I'm still learning about how {topic} works."
    
    def _generate_reasoning_response(self, topic: str, knowledge: TopicKnowledge) -> str:
        """Generate a reasoning response."""
        if knowledge.facts:
            # Find a fact that might explain "why"
            for fact in knowledge.facts:
                if any(w in fact.lower() for w in ['because', 'since', 'important', 'significant', 'reason']):
                    return f"{topic.title()} is significant because {fact.lower()}"
            return f"From what I understand about {topic}: {knowledge.facts[0]}"
        return f"I'm still learning about why {topic} matters."
    
    def _generate_general_response(self, topic: str, knowledge: TopicKnowledge) -> str:
        """Generate a general response about a topic."""
        parts = []
        
        # Add main facts
        if knowledge.facts:
            parts.append(f"Here's what I know about {topic}:")
            for fact in knowledge.facts[:3]:
                parts.append(f"• {fact}")
        
        # Mention related topics
        if knowledge.related_topics:
            related = ', '.join(knowledge.related_topics[:3])
            parts.append(f"\nRelated topics: {related}")
        
        return '\n'.join(parts) if parts else f"I know about {topic}, but I'm still organizing my thoughts."
    
    def _handle_unknown(self, user_input: str) -> str:
        """Handle unknown queries."""
        known = list(self.topics.keys())[:10]
        if known:
            return f"I'm not sure about that. I can discuss: {', '.join(known)}"
        return "I'm still building my knowledge. Try asking about a specific topic."
    
    # =========================================================================
    # CONTINUOUS LEARNING
    # =========================================================================
    
    def learn_from_conversation(self):
        """Learn from recent conversation to improve responses."""
        if len(self.conversation_history) < 3:
            return
        
        # Get recent turns
        recent = self.conversation_history[-5:]
        
        # Find topics that were discussed
        discussed_topics = set()
        for turn in recent:
            discussed_topics.update(turn.topics_used)
        
        # Deepen knowledge on discussed topics
        for topic in discussed_topics:
            if topic in self.topics:
                knowledge = self.topics[topic]
                # If we don't have many facts, learn more
                if len(knowledge.facts) < 5:
                    print(f"  Deepening knowledge on: {topic}")
                    self._expand_topic_knowledge(topic)
    
    def _expand_topic_knowledge(self, topic: str):
        """Expand knowledge about a topic."""
        knowledge = self.topics.get(topic)
        if not knowledge:
            return
        
        prompt = f"""I already know these facts about "{topic}":
{chr(10).join('- ' + f for f in knowledge.facts[:3])}

Provide 3 MORE facts about {topic} that I don't already know.
Keep each fact to one sentence.

Additional facts:"""

        response = self._call_llm(prompt, max_tokens=300)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 20 and len(line) < 200:
                    if line not in knowledge.facts:
                        knowledge.facts.append(line)
                        # Also add to semantic chain
                        self.semantic.ingest_item({
                            'text': line,
                            'agent': topic,
                            'source': 'knowledge_expansion',
                        })
    
    # =========================================================================
    # INTERACTIVE SESSION
    # =========================================================================
    
    def interactive(self):
        """Run interactive chat session."""
        print(f"\n{'═'*70}")
        print(" CONVERSATIONAL CHATBOT ".center(70, "═"))
        print(" Emergent Knowledge Building ".center(70))
        print("═" * 70)
        
        print(f"\nKnowledge: {len(self.topics)} topics, "
              f"{sum(len(t.facts) for t in self.topics.values())} facts")
        print(f"Semantic items: {len(self.semantic.items)}")
        
        print("\nCommands: 'topics', 'learn <topic>', 'stats', 'quit'\n")
        
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
                for topic in sorted(self.topics.keys()):
                    facts = len(self.topics[topic].facts)
                    print(f"  • {topic} ({facts} facts)")
                print()
                continue
            
            if user_input.lower().startswith('learn '):
                topic = user_input[6:].strip()
                self.learn_topic(topic, depth=1)
                print(f"  Learned about {topic}!\n")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n  Topics: {len(self.topics)}")
                print(f"  Facts: {sum(len(t.facts) for t in self.topics.values())}")
                print(f"  Queries answered: {self.queries_answered}")
                print(f"  LLM calls: {self.llm_calls}")
                print(f"  Conversation turns: {len(self.conversation_history)}\n")
                continue
            
            # Regular conversation
            response = self.chat(user_input)
            print(f"\nBot: {response}\n")
            
            # Periodic learning from conversation
            if self.queries_answered % 5 == 0:
                self.learn_from_conversation()


def main():
    """Main entry point."""
    print("=" * 70)
    print("CONVERSATIONAL CHATBOT WITH EMERGENT KNOWLEDGE")
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
    bot = ConversationalChatbot()
    
    # Build initial knowledge base with seed topics
    seed_topics = [
        "artificial intelligence",
        "machine learning", 
        "python programming",
        "philosophy",
        "science",
    ]
    
    bot.build_knowledge_base(seed_topics, depth=1)
    
    # Run interactive session
    bot.interactive()


if __name__ == "__main__":
    main()
