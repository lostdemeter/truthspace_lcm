#!/usr/bin/env python3
"""
Natural Language Gear Chain Chatbot

A chatbot with TWO gear chains:
1. Understanding chain - maps queries to dimensional space
2. Output chain - conditions semantic content into natural language

NO LLM is used for responses.
"""

import json
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class KnowledgeFrame:
    """A piece of knowledge with dimensional position."""
    text: str
    agent: str
    source: str
    position: Optional[np.ndarray] = None


class UnderstandingChain:
    """Gear chain for understanding queries via dimensional analysis."""
    
    # Semantic labels for known pole patterns
    SEMANTIC_LABELS = {
        # Agent-based labels
        ('holmes', 'villain'): ('analytical', 'scheming'),
        ('holmes', 'moriarty'): ('detective', 'criminal'),
        ('hero', 'villain'): ('heroic', 'villainous'),
        ('hero', 'fire'): ('purposeful', 'destructive'),
        ('king', 'servant'): ('powerful', 'humble'),
        ('sage', 'child'): ('wise', 'innocent'),
        ('watson', 'moriarty'): ('loyal', 'treacherous'),
        ('angel', 'villain'): ('virtuous', 'corrupt'),
        ('leader', 'victim'): ('active', 'passive'),
        ('explorer', 'victim'): ('adventurous', 'vulnerable'),
        # Feature-based fallbacks
        'investigates': 'investigative',
        'schemes': 'scheming',
        'helps': 'helpful',
        'commands': 'authoritative',
        'serves': 'servile',
        'fights': 'combative',
        'thinks': 'contemplative',
        'plays': 'playful',
    }
    
    def __init__(self):
        self.frames: List[KnowledgeFrame] = []
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.dimensions: List[Dict] = []
        self.agents: List[str] = []
        self.features: List[str] = []
        self.U: Optional[np.ndarray] = None
    
    def _get_semantic_labels(self, neg_pole: str, pos_pole: str) -> Tuple[str, str]:
        """Get semantic labels for dimension poles."""
        # Try direct lookup
        key = (neg_pole, pos_pole)
        if key in self.SEMANTIC_LABELS:
            return self.SEMANTIC_LABELS[key]
        
        # Try reversed
        key_rev = (pos_pole, neg_pole)
        if key_rev in self.SEMANTIC_LABELS:
            labels = self.SEMANTIC_LABELS[key_rev]
            return (labels[1], labels[0])
        
        # Generate labels from agent characteristics
        neg_label = self._agent_to_trait(neg_pole)
        pos_label = self._agent_to_trait(pos_pole)
        
        return (neg_label, pos_label)
    
    def _agent_to_trait(self, agent: str) -> str:
        """Convert an agent name to a trait description."""
        trait_map = {
            'holmes': 'analytical',
            'watson': 'supportive',
            'moriarty': 'scheming',
            'villain': 'villainous',
            'hero': 'heroic',
            'king': 'authoritative',
            'queen': 'regal',
            'servant': 'humble',
            'sage': 'wise',
            'child': 'innocent',
            'teenager': 'rebellious',
            'parent': 'nurturing',
            'angel': 'virtuous',
            'criminal': 'reformed',
            'priest': 'devout',
            'general': 'commanding',
            'victim': 'vulnerable',
            'explorer': 'adventurous',
            'healer': 'caring',
            'robot': 'logical',
            'storm': 'chaotic',
            'fire': 'destructive',
            'river': 'flowing',
            'tree': 'enduring',
            'dog': 'loyal',
            'fox': 'cunning',
            'widow': 'grieving',
            'bride': 'joyful',
            'mob': 'volatile',
            'refugee': 'displaced',
            'stranger': 'mysterious',
            'craftsman': 'skilled',
            'merchant': 'enterprising',
            'scholar': 'studious',
            'spy': 'secretive',
            'soldier': 'disciplined',
            'leader': 'decisive',
            'judge': 'impartial',
            'politician': 'calculating',
            'rebel': 'defiant',
            'alice': 'curious',
            'darcy': 'proud',
            'bennet': 'witty',
            'hamlet': 'contemplative',
            'macbeth': 'ambitious',
            'quixote': 'idealistic',
            'sancho': 'practical',
            'adler': 'clever',
            'idea': 'abstract',
        }
        
        # Check for partial matches
        agent_lower = agent.lower().replace('_', '')
        for key, trait in trait_map.items():
            if key in agent_lower or agent_lower in key:
                return trait
        
        # Default: capitalize the agent name
        return agent.replace('_', ' ')
    
    def ingest(self, corpus_path: str, source: str = None):
        """Ingest a corpus."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        for frame in corpus.get('frames', []):
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            if text and len(text) > 5:
                self.frames.append(KnowledgeFrame(text=text, agent=agent, source=source or 'unknown'))
                self._extract_patterns(text, agent)
    
    def _extract_patterns(self, text: str, agent: str):
        if not agent or len(agent) < 2:
            return
        for word in text.lower().split():
            word_clean = re.sub(r'[^a-z]', '', word)
            if len(word_clean) >= 3:
                self.agent_actions[agent][word_clean] += 1
    
    def learn(self, min_var: float = 0.02, max_dims: int = 15):
        """Learn dimensions from data."""
        valid = {a: v for a, v in self.agent_actions.items() if sum(v.values()) >= 3 and len(a) > 2}
        if len(valid) < 3:
            return
        
        self.agents = list(valid.keys())
        all_features = set()
        for v in valid.values():
            all_features.update(v.keys())
        self.features = sorted(all_features)
        
        X = np.zeros((len(self.agents), len(self.features)))
        for i, agent in enumerate(self.agents):
            total = sum(valid[agent].values())
            for j, feat in enumerate(self.features):
                X[i, j] = valid[agent].get(feat, 0) / total
        
        X_centered = X - X.mean(axis=0)
        self.U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        total_var = np.sum(S ** 2)
        var_ratios = (S ** 2) / total_var
        
        self.dimensions = []
        for i in range(min(len(S), max_dims)):
            if var_ratios[i] < min_var:
                break
            pos = self.U[:, i]
            neg_pole = self.agents[np.argmin(pos)]
            pos_pole = self.agents[np.argmax(pos)]
            
            # Get semantic labels for poles
            neg_label, pos_label = self._get_semantic_labels(neg_pole, pos_pole)
            
            self.dimensions.append({
                'name': f'Dim{i+1}',
                'variance': float(var_ratios[i]),
                'negative_pole': neg_pole,
                'positive_pole': pos_pole,
                'negative_label': neg_label,
                'positive_label': pos_label,
                'positions': {self.agents[j]: float(pos[j]) for j in range(len(self.agents))},
            })
        
        # Update frame positions
        for frame in self.frames:
            if frame.agent in self.agents:
                frame.position = np.array([d['positions'].get(frame.agent, 0) for d in self.dimensions])
    
    def get_position(self, concept: str) -> Optional[np.ndarray]:
        concept = concept.lower()
        if concept in self.agents:
            return np.array([d['positions'].get(concept, 0) for d in self.dimensions])
        for agent in self.agents:
            if concept in agent or agent in concept:
                return np.array([d['positions'].get(agent, 0) for d in self.dimensions])
        return None
    
    def find_agent(self, concept: str) -> Optional[str]:
        concept = concept.lower()
        if concept in self.agents:
            return concept
        for agent in self.agents:
            if concept in agent or agent in concept:
                return agent
        return None
    
    def find_similar(self, agent: str, k: int = 5) -> List[Tuple[str, float]]:
        pos = self.get_position(agent)
        if pos is None:
            return []
        results = []
        for other in self.agents:
            if other != agent:
                other_pos = self.get_position(other)
                if other_pos is not None:
                    results.append((other, float(np.linalg.norm(pos - other_pos))))
        return sorted(results, key=lambda x: x[1])[:k]
    
    def find_opposite(self, agent: str) -> Optional[Tuple[str, float]]:
        pos = self.get_position(agent)
        if pos is None:
            return None
        max_dist, opposite = 0, None
        for other in self.agents:
            if other != agent:
                other_pos = self.get_position(other)
                if other_pos is not None:
                    dist = np.linalg.norm(pos - other_pos)
                    if dist > max_dist:
                        max_dist, opposite = dist, other
        return (opposite, max_dist) if opposite else None
    
    def find_frames(self, concepts: List[str], k: int = 5) -> List[KnowledgeFrame]:
        positions = [self.get_position(c) for c in concepts]
        positions = [p for p in positions if p is not None]
        
        if not positions:
            # Text match fallback
            scored = []
            for frame in self.frames:
                score = sum(1 for c in concepts if c in frame.text.lower() or c in frame.agent)
                if score > 0:
                    scored.append((score, frame))
            scored.sort(key=lambda x: -x[0])
            return [f for _, f in scored[:k]]
        
        query_pos = np.mean(positions, axis=0)
        scored = []
        for frame in self.frames:
            if frame.position is not None:
                dist = np.linalg.norm(frame.position - query_pos)
                text_bonus = sum(0.5 for c in concepts if c in frame.text.lower())
                scored.append((-dist + text_bonus, frame))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:k]]


class OutputChain:
    """Gear chain for generating natural language from semantic content."""
    
    def __init__(self):
        self.bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.total_words = 0
    
    def learn(self, corpus_path: str):
        """Learn language patterns from corpus."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        for frame in corpus.get('frames', []):
            text = frame.get('text', '')
            words = [re.sub(r'[^a-z]', '', w.lower()) for w in text.split()]
            words = [w for w in words if w]
            
            for w in words:
                self.word_freq[w] += 1
                self.total_words += 1
            
            for i in range(len(words) - 1):
                self.bigrams[words[i]][words[i+1]] += 1
    
    def _format_name(self, name: str) -> str:
        """Format an agent name nicely."""
        return name.replace('_', ' ').title()
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list naturally."""
        items = [self._format_name(i) for i in items]
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"
    
    def describe(self, agent: str, similar: List[str], opposite: str, 
                 dimensions: Dict, frames: List[KnowledgeFrame]) -> str:
        """Generate a natural description."""
        name = self._format_name(agent)
        parts = []
        
        # Opening with similar concepts
        if similar:
            similar_str = self._format_list(similar[:3])
            parts.append(f"{name} shares characteristics with {similar_str}.")
        
        # Contrast with opposite
        if opposite:
            opp_name = self._format_name(opposite)
            parts.append(f"In contrast, {name} is quite different from {opp_name}.")
        
        # Dimensional traits
        if dimensions:
            traits = []
            for dim_name, info in list(dimensions.items())[:2]:
                pole = info.get('pole', '')
                if pole:
                    traits.append(self._format_name(pole).lower())
            if traits:
                parts.append(f"{name} exhibits traits associated with {self._format_list(traits)}.")
        
        # Knowledge from frames
        if frames:
            parts.append("")  # Blank line
            parts.append("From the knowledge base:")
            for frame in frames[:2]:
                text = frame.text.strip()
                if len(text) > 120:
                    text = text[:120].rsplit(' ', 1)[0] + "..."
                parts.append(f"  • {text}")
        
        return '\n'.join(parts) if parts else f"{name} is a known concept in the knowledge base."
    
    def compare(self, agent1: str, agent2: str, distance: float,
                differences: List[Dict], frames: List[KnowledgeFrame]) -> str:
        """Generate a natural comparison."""
        n1, n2 = self._format_name(agent1), self._format_name(agent2)
        
        # Similarity level
        if distance < 0.3:
            sim = "very similar"
        elif distance < 0.6:
            sim = "fairly similar"
        elif distance < 1.0:
            sim = "somewhat different"
        else:
            sim = "quite different"
        
        parts = [f"{n1} and {n2} are {sim}."]
        
        # Key differences
        if differences:
            d = differences[0]
            pole1 = self._format_name(d.get('pole1', '')).lower()
            pole2 = self._format_name(d.get('pole2', '')).lower()
            if pole1 and pole2:
                parts.append(f"The main distinction is that {n1} tends toward {pole1}, while {n2} leans toward {pole2}.")
        
        # Frame evidence
        if frames:
            parts.append("")
            parts.append("Supporting evidence:")
            for frame in frames[:2]:
                text = frame.text.strip()
                if len(text) > 100:
                    text = text[:100].rsplit(' ', 1)[0] + "..."
                parts.append(f"  • {text}")
        
        return '\n'.join(parts)
    
    def list_similar(self, agent: str, similar: List[Tuple[str, float]]) -> str:
        """Generate a natural similarity list."""
        name = self._format_name(agent)
        
        if not similar:
            return f"No similar concepts found for {name}."
        
        parts = [f"Concepts most similar to {name}:"]
        for other, dist in similar[:5]:
            other_name = self._format_name(other)
            closeness = "very close" if dist < 0.3 else "fairly close" if dist < 0.6 else "somewhat related"
            parts.append(f"  • {other_name} ({closeness})")
        
        return '\n'.join(parts)
    
    def unknown(self, query: str, known: List[str]) -> str:
        """Handle unknown concepts."""
        sample = self._format_list(known[:8])
        return f"I don't have specific information about '{query}'.\n\nKnown concepts include: {sample}, and others."


class NaturalChatbot:
    """
    A chatbot with dual gear chains for understanding and output.
    """
    
    def __init__(self):
        self.understanding = UnderstandingChain()
        self.output = OutputChain()
    
    def load(self, corpus_paths: List[Tuple[str, str]]):
        """Load corpora into both chains."""
        print("Loading knowledge base...")
        for path, name in corpus_paths:
            if Path(path).exists():
                self.understanding.ingest(path, name)
                self.output.learn(path)
                print(f"  Loaded: {name}")
        
        print("\nLearning dimensions...")
        self.understanding.learn()
        print(f"  Discovered {len(self.understanding.dimensions)} dimensions")
        print(f"  {len(self.understanding.agents)} agents, {len(self.understanding.frames)} frames")
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract concepts from query."""
        query_lower = query.lower()
        found = []
        for agent in self.understanding.agents:
            if agent in query_lower and len(agent) > 2 and agent not in found:
                found.append(agent)
        return found
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent."""
        q = query.lower()
        if any(w in q for w in ['compare', 'difference', 'between', 'vs', 'versus']):
            return 'compare'
        if any(w in q for w in ['similar', 'like', 'related']):
            return 'similar'
        if any(w in q for w in ['opposite', 'contrary']):
            return 'opposite'
        return 'describe'
    
    def chat(self, query: str) -> str:
        """Process query and generate natural response."""
        concepts = self._extract_concepts(query)
        intent = self._detect_intent(query)
        
        if not concepts:
            return self.output.unknown(query, self.understanding.agents)
        
        frames = self.understanding.find_frames(concepts, k=3)
        
        if intent == 'compare' and len(concepts) >= 2:
            a1, a2 = concepts[0], concepts[1]
            pos1 = self.understanding.get_position(a1)
            pos2 = self.understanding.get_position(a2)
            
            if pos1 is not None and pos2 is not None:
                dist = float(np.linalg.norm(pos2 - pos1))
                diff = pos2 - pos1
                max_idx = np.argmax(np.abs(diff))
                dim = self.understanding.dimensions[max_idx]
                
                # Use semantic labels for comparison
                if diff[max_idx] > 0:
                    label1 = dim.get('negative_label', dim['negative_pole'])
                    label2 = dim.get('positive_label', dim['positive_pole'])
                else:
                    label1 = dim.get('positive_label', dim['positive_pole'])
                    label2 = dim.get('negative_label', dim['negative_pole'])
                
                differences = [{'dimension': dim['name'], 'pole1': label1, 'pole2': label2}]
                return self.output.compare(a1, a2, dist, differences, frames)
        
        elif intent == 'similar' and concepts:
            agent = concepts[0]
            similar = self.understanding.find_similar(agent, k=5)
            return self.output.list_similar(agent, similar)
        
        elif intent == 'opposite' and concepts:
            agent = concepts[0]
            result = self.understanding.find_opposite(agent)
            if result:
                opposite, dist = result
                return f"The opposite of {self.output._format_name(agent)} is {self.output._format_name(opposite)}."
            return f"No clear opposite found for {self.output._format_name(agent)}."
        
        # Default: describe
        agent = concepts[0]
        similar = [s[0] for s in self.understanding.find_similar(agent, k=3)]
        opp_result = self.understanding.find_opposite(agent)
        opposite = opp_result[0] if opp_result else None
        
        # Get dimensional info with semantic labels
        pos = self.understanding.get_position(agent)
        dimensions = {}
        if pos is not None:
            for i, dim in enumerate(self.understanding.dimensions[:3]):
                if i < len(pos) and abs(pos[i]) > 0.15:
                    # Use semantic labels instead of pole names
                    if pos[i] > 0:
                        label = dim.get('positive_label', dim['positive_pole'])
                    else:
                        label = dim.get('negative_label', dim['negative_pole'])
                    dimensions[dim['name']] = {'pole': label}
        
        return self.output.describe(agent, similar, opposite, dimensions, frames)
    
    def interactive(self):
        """Run interactive session."""
        print("\n" + "═" * 70)
        print(" NATURAL LANGUAGE CHATBOT ".center(70, "═"))
        print(" Dual Gear Chain: Understanding + Output ".center(70))
        print("═" * 70)
        print(f"\nKnowledge: {len(self.understanding.frames)} frames")
        print(f"Agents: {len(self.understanding.agents)}")
        print(f"Dimensions: {len(self.understanding.dimensions)}")
        print("\nCommands: 'dims', 'agents', 'quit'\n")
        
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not query:
                continue
            if query.lower() == 'quit':
                break
            if query.lower() == 'dims':
                for d in self.understanding.dimensions:
                    print(f"  {d['name']}: {d['negative_pole']} ↔ {d['positive_pole']}")
                continue
            if query.lower() == 'agents':
                print(f"  {', '.join(sorted(self.understanding.agents)[:20])}...")
                continue
            
            print(f"\n{self.chat(query)}\n")


def main():
    base = Path(__file__).parent.parent
    sources = [
        (str(base / "truthspace_lcm/gears/corpus/corpus_llm_live.json"), "behavioral"),
        (str(base / "truthspace_lcm/gears/corpus/corpus_knowledge.json"), "knowledge"),
        (str(base / "truthspace_lcm/corpus_curated.json"), "curated"),
        (str(base / "truthspace_lcm/corpus_holmes_quality.json"), "holmes"),
    ]
    
    bot = NaturalChatbot()
    bot.load(sources)
    
    print("\n" + "─" * 70)
    print("TEST QUERIES")
    print("─" * 70)
    
    tests = [
        "Tell me about Holmes",
        "Compare Holmes and Watson",
        "What is similar to villain?",
        "What is the opposite of hero?",
    ]
    
    for q in tests:
        print(f"\n>>> {q}")
        print(bot.chat(q))
    
    bot.interactive()


if __name__ == "__main__":
    main()
