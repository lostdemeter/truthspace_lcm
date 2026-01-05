"""
OllamaSpace - LLM Knowledge Acquisition via Ollama

A HyperMapping-based system for acquiring knowledge from LLMs.
Uses Ollama API to query qwen2.5:14b for knowledge expansion.

Key features:
- Query LLM for topic information
- Extract and structure knowledge
- Store in KnowledgeSpace for geometric retrieval
- Feedback-based learning for query quality

Example:
    space = OllamaSpace()
    knowledge = space.query("What is machine learning?")
    # Returns structured knowledge that can be added to KnowledgeSpace

Author: Lesley Gushurst
License: GPLv3
"""

import json
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from hypermapping import HyperMapping, Mapping, TextEncoder, CRITICAL_LINE


# Default Ollama configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"


@dataclass
class KnowledgeResult:
    """Result of knowledge acquisition."""
    success: bool
    topic: str = ""
    content: str = ""
    facts: List[str] = field(default_factory=list)
    error: str = ""
    source: str = "ollama"
    model: str = ""


class OllamaSpace(HyperMapping):
    """
    HyperMapping-based LLM knowledge acquisition.
    
    Uses Ollama to query LLMs for knowledge, then structures
    the response for storage in KnowledgeSpace.
    """
    
    def __init__(self, 
                 name: str = "ollama_space",
                 dims: int = 8,
                 ollama_url: str = DEFAULT_OLLAMA_URL,
                 model: str = DEFAULT_MODEL,
                 timeout: int = 60):
        super().__init__(dims=dims, name=name)
        
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        
        # Text encoder for query positions
        self.encoder = TextEncoder(dims=dims)
        
        # Query history for learning
        self._query_history: List[Dict[str, Any]] = []
        self._last_query: Optional[str] = None
        self._last_result: Optional[KnowledgeResult] = None
        
        # Bootstrap prompt templates
        self._prompts = self._bootstrap_prompts()
    
    def _bootstrap_prompts(self) -> Dict[str, str]:
        """Bootstrap prompt templates for different query types."""
        return {
            'explain': """Explain {topic} in 2-3 concise paragraphs. 
Be informative and accurate. Include key facts and concepts.
Do not use markdown formatting.""",
            
            'define': """Define {topic} in one clear sentence, 
then provide 3-5 key facts about it.
Format as:
Definition: [definition]
Facts:
- [fact 1]
- [fact 2]
- [fact 3]""",
            
            'facts': """List 5-7 important facts about {topic}.
Format as a simple list:
- [fact 1]
- [fact 2]
etc.""",
            
            'compare': """Compare and contrast {topic}.
Provide key similarities and differences in 2-3 paragraphs.""",
            
            'how': """Explain how {topic} works in simple terms.
Use 2-3 paragraphs with clear explanations.""",
        }
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for prompt selection."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['what is', 'what are', 'define', 'definition']):
            return 'define'
        if any(w in query_lower for w in ['how does', 'how do', 'how to', 'how is']):
            return 'how'
        if any(w in query_lower for w in ['compare', 'difference', 'vs', 'versus']):
            return 'compare'
        if any(w in query_lower for w in ['facts', 'list', 'tell me about']):
            return 'facts'
        
        return 'explain'
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        query_lower = query.lower()
        
        # Remove common prefixes
        prefixes = [
            'what is ', 'what are ', 'define ', 'explain ',
            'tell me about ', 'how does ', 'how do ', 'how to ',
            'compare ', 'facts about ', 'list facts about ',
        ]
        
        topic = query
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                topic = query[len(prefix):]
                break
        
        # Remove trailing punctuation
        topic = topic.rstrip('?!.')
        
        return topic.strip()
    
    def _build_prompt(self, query: str) -> str:
        """Build the prompt for the LLM."""
        query_type = self._detect_query_type(query)
        topic = self._extract_topic(query)
        
        template = self._prompts.get(query_type, self._prompts['explain'])
        return template.format(topic=topic)
    
    def _parse_response(self, response: str, topic: str) -> KnowledgeResult:
        """Parse LLM response into structured knowledge."""
        result = KnowledgeResult(
            success=True,
            topic=topic,
            content=response.strip(),
            model=self.model,
        )
        
        # Extract facts from response
        lines = response.split('\n')
        facts = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items
            if line.startswith('- ') or line.startswith('• '):
                facts.append(line[2:].strip())
            elif line and line[0].isdigit() and '. ' in line:
                # Numbered list
                facts.append(line.split('. ', 1)[1].strip())
        
        # If no explicit facts found, split content into sentences
        if not facts:
            sentences = response.replace('\n', ' ').split('. ')
            facts = [s.strip() + '.' for s in sentences if len(s.strip()) > 20][:5]
        
        result.facts = facts
        return result
    
    def query(self, query: str, stream: bool = False) -> KnowledgeResult:
        """
        Query the LLM for knowledge.
        
        Args:
            query: Natural language question
            stream: Whether to stream the response (not implemented)
            
        Returns:
            KnowledgeResult with structured knowledge
        """
        self._last_query = query
        topic = self._extract_topic(query)
        prompt = self._build_prompt(query)
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                },
                timeout=self.timeout,
            )
            
            if response.status_code != 200:
                result = KnowledgeResult(
                    success=False,
                    topic=topic,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )
                self._last_result = result
                return result
            
            data = response.json()
            content = data.get('response', '')
            
            if not content:
                result = KnowledgeResult(
                    success=False,
                    topic=topic,
                    error="Empty response from LLM",
                )
                self._last_result = result
                return result
            
            result = self._parse_response(content, topic)
            self._last_result = result
            
            # Record in history
            self._query_history.append({
                'query': query,
                'topic': topic,
                'success': True,
                'facts_count': len(result.facts),
            })
            
            return result
            
        except requests.exceptions.ConnectionError:
            result = KnowledgeResult(
                success=False,
                topic=topic,
                error=f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?",
            )
            self._last_result = result
            return result
        except requests.exceptions.Timeout:
            result = KnowledgeResult(
                success=False,
                topic=topic,
                error=f"Request timed out after {self.timeout}s",
            )
            self._last_result = result
            return result
        except Exception as e:
            result = KnowledgeResult(
                success=False,
                topic=topic,
                error=str(e),
            )
            self._last_result = result
            return result
    
    def learn_topic(self, topic: str) -> KnowledgeResult:
        """
        Learn about a topic by querying the LLM.
        
        This is a convenience method that builds a query from a topic.
        """
        query = f"Explain {topic}"
        return self.query(query)
    
    def feedback(self, success: bool) -> bool:
        """Provide feedback on the last query."""
        if self._last_query and self._query_history:
            self._query_history[-1]['user_feedback'] = success
            return True
        return False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(
                self.ollama_url.replace('/api/generate', '/api/tags'),
                timeout=5,
            )
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models from Ollama."""
        try:
            response = requests.get(
                self.ollama_url.replace('/api/generate', '/api/tags'),
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except:
            pass
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about queries."""
        successful = sum(1 for q in self._query_history if q.get('success', False))
        return {
            'total_queries': len(self._query_history),
            'successful_queries': successful,
            'model': self.model,
            'ollama_url': self.ollama_url,
            'available': self.is_available(),
        }


def test_ollama_space():
    """Test OllamaSpace functionality."""
    space = OllamaSpace()
    
    print("=" * 60)
    print("OllamaSpace Test")
    print("=" * 60)
    
    # Check availability
    print(f"\nOllama available: {space.is_available()}")
    
    if not space.is_available():
        print("Ollama is not running. Start it with: ollama serve")
        return
    
    # List models
    models = space.list_models()
    print(f"Available models: {models[:5]}")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = space.query(query)
        
        if result.success:
            print(f"  ✓ Topic: {result.topic}")
            print(f"  ✓ Facts: {len(result.facts)}")
            for fact in result.facts[:3]:
                print(f"    - {fact[:60]}...")
        else:
            print(f"  ✗ Error: {result.error}")
    
    # Stats
    print(f"\nStats: {space.get_stats()}")


if __name__ == "__main__":
    test_ollama_space()
