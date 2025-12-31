#!/usr/bin/env python3
"""
Emergent Chatbot

A chatbot that uses emergent dimensions discovered from data to:
1. Understand queries by mapping them to dimensional space
2. Find relevant information based on dimensional similarity
3. Generate meaningful responses

This combines:
- Continuous learning (dimensions evolve with data)
- Multiple data sources (LLM-generated, existing corpora, Wikipedia)
- Query understanding via dimensional analysis
- Response generation using retrieved context
"""

import json
import numpy as np
import requests
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class KnowledgeFrame:
    """A piece of knowledge with dimensional position."""
    text: str
    agent: str
    source: str
    position: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergentChatbot:
    """
    A chatbot that learns dimensions from data and uses them for understanding.
    """
    
    def __init__(self, model: str = "qwen2:latest"):
        self.model = model
        
        # Knowledge base
        self.frames: List[KnowledgeFrame] = []
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Discovered dimensions
        self.dimensions: List[Dict] = []
        self.agents: List[str] = []
        self.features: List[str] = []
        
        # SVD components
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        # Learning state
        self.total_frames = 0
        self.learning_cycles = 0
    
    def ingest_corpus(self, corpus_path: str, source_name: str = None):
        """Ingest a corpus file."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        frames = corpus.get('frames', [])
        source = source_name or Path(corpus_path).stem
        
        for frame in frames:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not text or len(text) < 10:
                continue
            
            # Create knowledge frame
            kf = KnowledgeFrame(
                text=text,
                agent=agent if agent else 'unknown',
                source=source,
                metadata=frame.get('metadata', {}),
            )
            self.frames.append(kf)
            
            # Extract behavioral patterns
            self._extract_patterns(text, agent)
        
        self.total_frames = len(self.frames)
        print(f"Ingested {len(frames)} frames from {source}, total: {self.total_frames}")
    
    def _extract_patterns(self, text: str, agent: str):
        """Extract behavioral patterns from text."""
        if not agent or len(agent) < 2:
            return
        
        words = text.lower().split()
        
        # Extract verbs (words that look like actions)
        for word in words:
            word_clean = re.sub(r'[^a-z]', '', word)
            if len(word_clean) < 3:
                continue
            
            # Heuristic verb detection
            verb_endings = ['ed', 'ing', 'es', 's', 'ly']
            if any(word_clean.endswith(e) for e in verb_endings):
                self.agent_actions[agent][word_clean] += 1
    
    def learn_dimensions(self, min_variance: float = 0.02, max_dims: int = 15):
        """Learn dimensions from accumulated data."""
        self.learning_cycles += 1
        
        # Filter agents with sufficient data
        min_actions = 3
        valid_agents = {
            a: v for a, v in self.agent_actions.items()
            if sum(v.values()) >= min_actions and len(a) > 2
        }
        
        if len(valid_agents) < 3:
            print("Not enough agents for dimension learning")
            return
        
        self.agents = list(valid_agents.keys())
        
        # Get all features
        all_actions = set()
        for actions in valid_agents.values():
            all_actions.update(actions.keys())
        self.features = sorted(all_actions)
        
        n_agents = len(self.agents)
        n_features = len(self.features)
        
        print(f"Learning dimensions: {n_agents} agents × {n_features} features")
        
        # Build normalized matrix
        X = np.zeros((n_agents, n_features))
        for i, agent in enumerate(self.agents):
            actions = valid_agents[agent]
            total = sum(actions.values())
            if total > 0:
                for j, action in enumerate(self.features):
                    X[i, j] = actions.get(action, 0) / total
        
        # Center and SVD
        X_centered = X - X.mean(axis=0)
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Variance analysis
        total_var = np.sum(self.S ** 2)
        var_ratios = (self.S ** 2) / total_var
        
        # Discover dimensions
        self.dimensions = []
        cumulative = 0.0
        
        for i in range(min(len(self.S), max_dims)):
            var = var_ratios[i]
            cumulative += var
            
            if var < min_variance:
                break
            
            positions = self.U[:, i]
            min_idx = np.argmin(positions)
            max_idx = np.argmax(positions)
            
            feature_weights = self.Vt[i]
            neg_features = [self.features[j] for j in np.argsort(feature_weights)[:3]]
            pos_features = [self.features[j] for j in np.argsort(feature_weights)[-3:]]
            
            dim = {
                'index': i,
                'name': f'Dim{i+1}',
                'variance': float(var),
                'negative_pole': self.agents[min_idx],
                'positive_pole': self.agents[max_idx],
                'negative_features': neg_features,
                'positive_features': pos_features,
                'positions': {self.agents[j]: float(positions[j]) for j in range(n_agents)},
            }
            self.dimensions.append(dim)
        
        print(f"Discovered {len(self.dimensions)} dimensions ({cumulative*100:.1f}% variance)")
        
        # Update frame positions
        self._update_frame_positions()
    
    def _update_frame_positions(self):
        """Update dimensional positions for all frames."""
        for frame in self.frames:
            if frame.agent in self.agents:
                pos = np.array([
                    dim['positions'].get(frame.agent, 0) 
                    for dim in self.dimensions
                ])
                frame.position = pos
    
    def _get_concept_position(self, concept: str) -> Optional[np.ndarray]:
        """Get the dimensional position of a concept."""
        concept_lower = concept.lower()
        
        if concept_lower in self.agents:
            return np.array([
                dim['positions'].get(concept_lower, 0)
                for dim in self.dimensions
            ])
        
        # Try partial match
        for agent in self.agents:
            if concept_lower in agent or agent in concept_lower:
                return np.array([
                    dim['positions'].get(agent, 0)
                    for dim in self.dimensions
                ])
        
        return None
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract concepts from a query."""
        query_lower = query.lower()
        found = []
        
        # Check for known agents
        for agent in self.agents:
            if agent in query_lower and len(agent) > 2:
                found.append(agent)
        
        # Also extract potential keywords
        stopwords = {'what', 'who', 'how', 'why', 'when', 'where', 'the', 'is', 'are',
                    'was', 'were', 'will', 'would', 'could', 'should', 'have', 'has',
                    'does', 'did', 'about', 'like', 'from', 'with', 'this', 'that',
                    'tell', 'me', 'you', 'can', 'know', 'think', 'between', 'and',
                    'difference', 'similar', 'compare', 'explain'}
        
        words = re.findall(r'\b[a-z]+\b', query_lower)
        for word in words:
            if len(word) > 3 and word not in stopwords and word not in found:
                # Check if it's close to any agent
                for agent in self.agents:
                    if word in agent or agent in word:
                        if agent not in found:
                            found.append(agent)
        
        return found
    
    def _find_relevant_frames(self, concepts: List[str], k: int = 10) -> List[KnowledgeFrame]:
        """Find frames relevant to the concepts."""
        if not concepts:
            return []
        
        # Get average position of concepts
        positions = []
        for concept in concepts:
            pos = self._get_concept_position(concept)
            if pos is not None:
                positions.append(pos)
        
        if not positions:
            # Fall back to text matching
            relevant = []
            for frame in self.frames:
                score = sum(1 for c in concepts if c in frame.text.lower() or c in frame.agent)
                if score > 0:
                    relevant.append((score, frame))
            relevant.sort(key=lambda x: -x[0])
            return [f for _, f in relevant[:k]]
        
        query_pos = np.mean(positions, axis=0)
        
        # Find frames closest to query position
        scored = []
        for frame in self.frames:
            if frame.position is not None:
                dist = np.linalg.norm(frame.position - query_pos)
                # Also boost if text contains concepts
                text_match = sum(1 for c in concepts if c in frame.text.lower())
                score = -dist + text_match * 0.5
                scored.append((score, frame))
        
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:k]]
    
    def _analyze_concepts(self, concepts: List[str]) -> Dict[str, Any]:
        """Analyze concepts dimensionally."""
        analysis = {}
        
        for concept in concepts:
            pos = self._get_concept_position(concept)
            if pos is None:
                continue
            
            concept_analysis = {
                'dimensions': {},
                'similar': [],
                'opposite': None,
            }
            
            # Dimensional analysis
            for i, dim in enumerate(self.dimensions):
                if i < len(pos):
                    p = pos[i]
                    if p > 0.15:
                        concept_analysis['dimensions'][dim['name']] = {
                            'class': 'positive',
                            'pole': dim['positive_pole'],
                            'value': p,
                        }
                    elif p < -0.15:
                        concept_analysis['dimensions'][dim['name']] = {
                            'class': 'negative',
                            'pole': dim['negative_pole'],
                            'value': p,
                        }
            
            # Find similar
            similarities = []
            for other in self.agents:
                if other != concept:
                    other_pos = self._get_concept_position(other)
                    if other_pos is not None:
                        dist = np.linalg.norm(pos - other_pos)
                        similarities.append((other, dist))
            
            similarities.sort(key=lambda x: x[1])
            concept_analysis['similar'] = [s[0] for s in similarities[:5]]
            
            if similarities:
                concept_analysis['opposite'] = similarities[-1][0]
            
            analysis[concept] = concept_analysis
        
        return analysis
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the LLM for response generation."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(self, query: str) -> str:
        """
        Process a query and generate a response.
        
        This is the main interface for the chatbot.
        """
        # Extract concepts
        concepts = self._extract_query_concepts(query)
        
        # Analyze concepts dimensionally
        analysis = self._analyze_concepts(concepts)
        
        # Find relevant frames
        relevant_frames = self._find_relevant_frames(concepts, k=5)
        
        # Build context for LLM
        context_parts = []
        
        if analysis:
            context_parts.append("Concept Analysis:")
            for concept, info in analysis.items():
                dims = info.get('dimensions', {})
                similar = info.get('similar', [])[:3]
                
                if dims:
                    dim_desc = ", ".join([f"{d}: toward {v['pole']}" for d, v in dims.items()])
                    context_parts.append(f"  {concept}: {dim_desc}")
                
                if similar:
                    context_parts.append(f"  Similar to: {', '.join(similar)}")
        
        if relevant_frames:
            context_parts.append("\nRelevant Information:")
            for frame in relevant_frames[:5]:
                context_parts.append(f"  - {frame.text[:150]}...")
        
        context = "\n".join(context_parts)
        
        # Generate response using LLM
        prompt = f"""You are a helpful assistant with knowledge about characters and concepts.

Context from knowledge base:
{context}

User question: {query}

Based on the context above, provide a helpful and informative answer. If the context doesn't contain enough information, say so and provide what you can based on general knowledge.

Answer:"""

        response = self._call_llm(prompt)
        
        return response
    
    def interactive_chat(self):
        """Run an interactive chat session."""
        print("\n" + "=" * 70)
        print("EMERGENT CHATBOT")
        print("=" * 70)
        print(f"\nKnowledge base: {self.total_frames} frames, {len(self.agents)} agents")
        print(f"Dimensions: {len(self.dimensions)}")
        print("\nType 'quit' to exit, 'stats' for statistics, 'dims' for dimensions")
        print()
        
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
                print(f"\nFrames: {self.total_frames}")
                print(f"Agents: {len(self.agents)}")
                print(f"Dimensions: {len(self.dimensions)}")
                print(f"Learning cycles: {self.learning_cycles}")
                continue
            
            if query.lower() == 'dims':
                print("\nDiscovered Dimensions:")
                for dim in self.dimensions:
                    print(f"  {dim['name']}: {dim['negative_pole']} <-> {dim['positive_pole']} ({dim['variance']*100:.1f}%)")
                continue
            
            # Process query
            response = self.chat(query)
            print(f"\nBot: {response}\n")
    
    def save_state(self, path: str):
        """Save chatbot state."""
        state = {
            'total_frames': self.total_frames,
            'learning_cycles': self.learning_cycles,
            'agents': self.agents,
            'dimensions': self.dimensions,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)


def create_chatbot_with_multiple_sources():
    """Create a chatbot with multiple data sources."""
    print("=" * 70)
    print("CREATING EMERGENT CHATBOT WITH MULTIPLE SOURCES")
    print("=" * 70)
    
    chatbot = EmergentChatbot(model="qwen2:latest")
    
    # Ingest multiple sources
    sources = [
        ("truthspace_lcm/gears/corpus/corpus_llm_live.json", "llm_behavioral"),
        ("truthspace_lcm/gears/corpus/corpus_rich_behavioral.json", "rich_behavioral"),
        ("truthspace_lcm/gears/corpus/corpus_knowledge.json", "knowledge"),
    ]
    
    # Check for additional sources
    additional = [
        "truthspace_lcm/corpus_curated.json",
        "truthspace_lcm/corpus_holmes_quality.json",
    ]
    
    for path in additional:
        if Path(path).exists():
            sources.append((path, Path(path).stem))
    
    print(f"\nIngesting {len(sources)} data sources...")
    
    for path, name in sources:
        full_path = Path(__file__).parent.parent / path
        if full_path.exists():
            chatbot.ingest_corpus(str(full_path), name)
        else:
            print(f"  Skipping {path} (not found)")
    
    # Learn dimensions
    print("\nLearning dimensions...")
    chatbot.learn_dimensions(min_variance=0.02, max_dims=15)
    
    # Show discovered dimensions
    print("\nDiscovered Dimensions:")
    for dim in chatbot.dimensions[:10]:
        print(f"  {dim['name']}: {dim['negative_pole']} <-> {dim['positive_pole']} ({dim['variance']*100:.1f}%)")
    
    return chatbot


def main():
    # Create chatbot with multiple sources
    chatbot = create_chatbot_with_multiple_sources()
    
    # Test some queries
    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)
    
    test_queries = [
        "Tell me about Holmes",
        "What is the difference between a hero and a villain?",
        "Who is Watson?",
        "Compare the king and the servant",
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        response = chatbot.chat(query)
        print(f"Response: {response[:300]}...")
    
    # Save state
    state_path = Path(__file__).parent / "chatbot_state.json"
    chatbot.save_state(str(state_path))
    print(f"\nState saved to: {state_path}")
    
    # Start interactive session
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    chatbot.interactive_chat()
    
    return chatbot


if __name__ == "__main__":
    chatbot = main()
