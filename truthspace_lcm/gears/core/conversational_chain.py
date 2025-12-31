"""
Emergent Conversational Chain

A chain that builds knowledge through corpus building and generates
responses using ONLY emergent patterns - no LLM during conversation.

The key insight: LLM is used as a knowledge RESOURCE (like Wikipedia),
not as a response GENERATOR. All responses emerge from the learned structure.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
import re
import time
import requests
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from .semantic_chain import SemanticChain


# Common Project Gutenberg URLs
GUTENBERG_BOOKS = {
    'moby_dick': 'https://www.gutenberg.org/files/2701/2701-0.txt',
    'pride_and_prejudice': 'https://www.gutenberg.org/files/1342/1342-0.txt',
    'frankenstein': 'https://www.gutenberg.org/files/84/84-0.txt',
    'dracula': 'https://www.gutenberg.org/files/345/345-0.txt',
    'alice_in_wonderland': 'https://www.gutenberg.org/files/11/11-0.txt',
    'sherlock_holmes': 'https://www.gutenberg.org/files/1661/1661-0.txt',
    'war_and_peace': 'https://www.gutenberg.org/files/2600/2600-0.txt',
    'great_gatsby': 'https://www.gutenberg.org/files/64317/64317-0.txt',
    'jane_eyre': 'https://www.gutenberg.org/files/1260/1260-0.txt',
    'wuthering_heights': 'https://www.gutenberg.org/files/768/768-0.txt',
}


@dataclass
class KnowledgeItem:
    """A single piece of knowledge in the corpus."""
    text: str
    topic: str
    source: str
    item_type: str  # 'fact', 'definition', 'example', 'relation'
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""
    user_input: str
    bot_response: str
    topics_used: List[str]
    timestamp: float = field(default_factory=time.time)


class ConversationalChain:
    """
    Emergent Conversational Chain.
    
    Builds knowledge through corpus building (using LLM as resource),
    then generates responses using ONLY emergent patterns.
    
    Key principle: LLM is used ONLY during corpus building phase,
    NEVER during conversation. All responses are emergent.
    """
    
    def __init__(self):
        # Semantic chain for emergent understanding
        self.semantic = SemanticChain()
        
        # Knowledge corpus
        self.corpus: List[KnowledgeItem] = []
        self.topics: Set[str] = set()
        self.topic_definitions: Dict[str, str] = {}
        
        # Conversation history
        self.history: List[ConversationTurn] = []
        
        # Response templates (discovered from data)
        self.response_templates: List[str] = []
        
        # Stats
        self.corpus_building_calls = 0
        self.conversation_calls = 0  # Should stay 0 for truly emergent
        
        # LLM configuration (for corpus building only)
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for corpus building."""
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """
        Call LLM - ONLY for corpus building.
        
        This is like querying Wikipedia - we're gathering knowledge,
        not generating responses.
        """
        if not self.llm_url or not self.llm_model:
            return None
        
        try:
            import requests
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7}
                },
                timeout=60
            )
            if response.status_code == 200:
                self.corpus_building_calls += 1
                return response.json().get("response", "").strip()
        except Exception as e:
            pass
        return None
    
    # =========================================================================
    # CORPUS BUILDING (LLM used here as knowledge resource)
    # =========================================================================
    
    def add_knowledge(self, text: str, topic: str, item_type: str = 'fact', 
                      source: str = 'manual'):
        """Add a piece of knowledge to the corpus."""
        self.corpus.append(KnowledgeItem(
            text=text,
            topic=topic.lower(),
            source=source,
            item_type=item_type,
        ))
        self.topics.add(topic.lower())
        
        # Also ingest into semantic chain
        self.semantic.ingest_item({
            'text': text,
            'agent': topic.lower(),
            'source': source,
        })
    
    def learn_topic(self, topic: str) -> bool:
        """
        Learn about a topic using LLM as knowledge resource.
        
        Returns True if successful.
        """
        if not self.llm_url:
            return False
        
        topic_lower = topic.lower().strip()
        self.topics.add(topic_lower)
        
        # Get factual sentences
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
                    self.add_knowledge(line, topic_lower, 'fact', 'llm_knowledge')
        
        # Get definition
        prompt = f"""Define "{topic}" in one clear sentence.
Start with "{topic} is" or "{topic} refers to".

Definition:"""

        response = self._call_llm(prompt, max_tokens=100)
        if response:
            line = response.strip().split('\n')[0]
            if len(line) > 10:
                self.add_knowledge(line, topic_lower, 'definition', 'llm_knowledge')
                self.topic_definitions[topic_lower] = line
        
        # Get related topics
        prompt = f"""List 3 topics closely related to "{topic}".
Just the topic names, one per line:"""

        response = self._call_llm(prompt, max_tokens=50)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ').lower()
                if len(line) > 2 and len(line) < 40:
                    self.topics.add(line)
        
        return True
    
    def build_corpus(self, seed_topics: List[str], expand: bool = True):
        """Build knowledge corpus from seed topics."""
        # Learn seed topics
        for topic in seed_topics:
            self.learn_topic(topic)
        
        # Expand to related topics
        if expand:
            initial_topics = list(self.topics)
            for topic in initial_topics:
                if topic.lower() not in [t.lower() for t in seed_topics]:
                    self.learn_topic(topic)
        
        # Learn emergent structure
        self.semantic.learn_dimensions()
        
        # Discover response templates
        self._discover_templates()
    
    def _discover_templates(self):
        """Discover response templates from corpus patterns."""
        patterns = defaultdict(int)
        
        for item in self.corpus:
            text = item.text
            if item.topic in text.lower():
                pattern = text.lower().replace(item.topic, '{topic}')
                pattern = re.sub(r'\b(is|are|was|were)\b', '{be}', pattern)
                patterns[pattern[:50]] += 1
        
        for pattern, count in patterns.items():
            if count >= 2:
                self.response_templates.append(pattern)
        
        if not self.response_templates:
            self.response_templates = [
                "{topic} {be} {content}",
                "Regarding {topic}, {content}",
                "{content}",
            ]
    
    def load_corpus(self, path: str):
        """Load corpus from JSON file."""
        corpus_path = Path(path).resolve()
        self.corpus_path = str(corpus_path)
        
        with open(corpus_path, 'r') as f:
            data = json.load(f)
        
        # Load book title if present
        if 'book_title' in data:
            self.book_title = data['book_title']
        
        for item in data.get('items', []):
            self.add_knowledge(
                text=item.get('text', ''),
                topic=item.get('topic', 'unknown'),
                item_type=item.get('type', 'fact'),
                source=item.get('source', 'file'),
            )
        
        self.semantic.learn_dimensions()
        self._discover_templates()
    
    def save_corpus(self, path: str):
        """Save corpus to JSON file."""
        data = {
            'topics': list(self.topics),
            'definitions': self.topic_definitions,
            'book_title': getattr(self, 'book_title', None),
            'items': [
                {
                    'text': item.text,
                    'topic': item.topic,
                    'type': item.item_type,
                    'source': item.source,
                }
                for item in self.corpus
            ],
            'stats': {
                'corpus_building_calls': self.corpus_building_calls,
                'total_items': len(self.corpus),
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # =========================================================================
    # LITERARY WORK LOADING
    # =========================================================================
    
    def load_book(self, book_name: str = None, url: str = None, 
                  max_lines: int = None, progress_callback=None) -> bool:
        """
        Load a literary work and build corpus from it.
        
        Args:
            book_name: Name from GUTENBERG_BOOKS (e.g., 'moby_dick')
            url: Direct URL to text file
            max_lines: Limit lines processed (None = all)
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            True if successful
        """
        # Get URL
        if book_name and book_name in GUTENBERG_BOOKS:
            url = GUTENBERG_BOOKS[book_name]
            self.book_title = book_name.replace('_', ' ').title()
        elif url:
            self.book_title = "Literary Work"
        else:
            return False
        
        # Fetch text
        try:
            if progress_callback:
                progress_callback(0, 100, f"Fetching {self.book_title}...")
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return False
            text = response.text
        except Exception as e:
            return False
        
        # Process the text
        return self.load_text(text, max_lines=max_lines, 
                             progress_callback=progress_callback)
    
    def load_text(self, text: str, max_lines: int = None,
                  progress_callback=None) -> bool:
        """
        Load raw text and build corpus from it.
        
        Extracts sentences, identifies characters/concepts, builds knowledge.
        """
        lines = text.split('\n')
        total_lines = len(lines) if max_lines is None else min(len(lines), max_lines)
        
        # Common stopwords to filter
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
            'here', 'there', 'then', 'once', 'if', 'because', 'until', 'while',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'any', 'said',
            'one', 'two', 'three', 'upon', 'like', 'even', 'still', 'yet',
        }
        
        # Track concept frequencies
        concept_counts = Counter()
        sentences_by_concept = defaultdict(list)
        
        processed = 0
        for i, line in enumerate(lines[:total_lines]):
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            # Extract sentences
            sentences = re.split(r'[.!?]+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15 or len(sentence) > 300:
                    continue
                
                # Extract potential concepts (capitalized words, proper nouns)
                words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in stopwords and len(word_lower) > 2:
                        concept_counts[word_lower] += 1
                        if len(sentences_by_concept[word_lower]) < 20:
                            sentences_by_concept[word_lower].append(sentence)
            
            processed += 1
            if progress_callback and processed % 1000 == 0:
                pct = int(processed / total_lines * 100)
                progress_callback(processed, total_lines, 
                                 f"Processing line {processed}/{total_lines}")
        
        # Add top concepts to corpus (scale with book size)
        num_concepts = min(500, max(100, len(concept_counts) // 2))
        top_concepts = concept_counts.most_common(num_concepts)
        for concept, count in top_concepts:
            if count >= 2:  # Minimum frequency
                self.topics.add(concept)
                
                # Add sentences as knowledge
                for sentence in sentences_by_concept[concept][:5]:
                    self.add_knowledge(
                        text=sentence,
                        topic=concept,
                        item_type='context',
                        source='book',
                    )
                
                # Create a summary definition from first mention
                if sentences_by_concept[concept]:
                    first_sentence = sentences_by_concept[concept][0]
                    self.topic_definitions[concept] = first_sentence
        
        # Learn emergent structure
        if progress_callback:
            progress_callback(total_lines, total_lines, "Learning structure...")
        
        self.semantic.learn_dimensions()
        self._discover_templates()
        
        return True
    
    def get_available_books(self) -> List[str]:
        """Get list of available books from Gutenberg."""
        return list(GUTENBERG_BOOKS.keys())
    
    # =========================================================================
    # EMERGENT RESPONSE GENERATION (NO LLM!)
    # =========================================================================
    
    def chat(self, user_input: str) -> str:
        """
        Generate response using ONLY emergent patterns.
        NO LLM calls here - pure emergence.
        """
        # Extract topics from input
        topics = self._extract_topics(user_input)
        
        if not topics:
            return self._handle_unknown(user_input)
        
        # Generate emergent response
        response = self._generate_response(user_input, topics)
        
        # Store in history
        self.history.append(ConversationTurn(
            user_input=user_input,
            bot_response=response,
            topics_used=topics,
        ))
        
        return response
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract known topics from text."""
        text_lower = text.lower()
        found = []
        
        for topic in self.topics:
            if topic in text_lower:
                found.append(topic)
        
        return sorted(found, key=len, reverse=True)
    
    def _generate_response(self, user_input: str, topics: List[str]) -> str:
        """Generate response using emergent patterns only."""
        main_topic = topics[0]
        
        # Get relevant content
        relevant = self._get_relevant_content(main_topic)
        
        # Get emergent traits
        traits = self.semantic.describe_traits(main_topic)
        
        # Get similar concepts
        similar = self.semantic.find_similar(main_topic, k=3)
        
        # Check if this is from a book
        is_book = getattr(self, 'book_title', None) is not None
        
        # Build response based on question type
        input_lower = user_input.lower()
        response_parts = []
        
        if 'what is' in input_lower or 'what are' in input_lower:
            # Definition question
            definition = self.topic_definitions.get(main_topic)
            if definition:
                response_parts.append(definition)
            if relevant and len(relevant) > 1:
                response_parts.append(f"\nMore context: {relevant[1]}")
        
        elif 'who is' in input_lower or 'tell me about' in input_lower:
            # Character/topic question - give multiple excerpts
            if relevant:
                response_parts.append(f"From the text about {main_topic.title()}:")
                for excerpt in relevant[:3]:
                    response_parts.append(f"  • \"{excerpt}\"")
            if similar:
                similar_str = ', '.join([s[0].title() for s in similar[:3]])
                response_parts.append(f"\nRelated: {similar_str}")
        
        elif 'how' in input_lower:
            # Explanation
            if relevant:
                response_parts.append(f"About {main_topic.title()}:")
                for i, fact in enumerate(relevant[:3], 1):
                    response_parts.append(f"  {i}. {fact}")
        
        elif 'why' in input_lower:
            # Reasoning
            causal = [r for r in relevant if any(w in r.lower() for w in 
                     ['because', 'since', 'therefore', 'important', 'significant'])]
            if causal:
                response_parts.append(causal[0])
            elif relevant:
                response_parts.append(relevant[0])
        
        else:
            # General question
            if relevant:
                response_parts.append(relevant[0])
            
            if traits:
                trait_str = ', '.join(traits[:3])
                response_parts.append(f"{main_topic.title()} is characterized by: {trait_str}.")
            
            if similar:
                similar_str = ', '.join([s[0] for s in similar[:3]])
                response_parts.append(f"Related concepts: {similar_str}.")
        
        if not response_parts:
            if relevant:
                return relevant[0]
            return f"I have limited knowledge about {main_topic}."
        
        return '\n'.join(response_parts)
    
    def _get_relevant_content(self, topic: str) -> List[str]:
        """Get relevant content from corpus for a topic."""
        relevant = []
        
        for item in self.corpus:
            if item.topic == topic.lower():
                relevant.append(item.text)
        
        return relevant[:5]
    
    def _handle_unknown(self, user_input: str) -> str:
        """Handle unknown queries."""
        words = user_input.lower().split()
        
        for word in words:
            if len(word) > 3:
                for item in self.corpus:
                    if word in item.text.lower():
                        return f"I found something related: {item.text}"
        
        known = sorted(list(self.topics))[:10]
        if known:
            return f"I don't have information about that. I can discuss: {', '.join(known)}"
        return "I don't have knowledge about that topic yet."
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        stats = {
            'topics': len(self.topics),
            'corpus_items': len(self.corpus),
            'definitions': len(self.topic_definitions),
            'dimensions': len(self.semantic.dimensions) if hasattr(self.semantic, 'dimensions') else 0,
            'corpus_building_calls': self.corpus_building_calls,
            'conversation_calls': self.conversation_calls,
            'history_length': len(self.history),
        }
        
        # Add optional fields if present
        if hasattr(self, 'corpus_path'):
            stats['corpus_path'] = self.corpus_path
        if hasattr(self, 'book_title'):
            stats['book_title'] = self.book_title
        
        return stats
    
    def list_topics(self) -> List[str]:
        """List all known topics."""
        return sorted(list(self.topics))
    
    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a specific topic."""
        topic_lower = topic.lower()
        
        facts = [item.text for item in self.corpus if item.topic == topic_lower]
        definition = self.topic_definitions.get(topic_lower)
        
        return {
            'topic': topic_lower,
            'definition': definition,
            'facts': facts,
            'fact_count': len(facts),
        }
