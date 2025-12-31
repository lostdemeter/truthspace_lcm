#!/usr/bin/env python3
"""
Dual Emergent Gear Chain Chatbot

Both understanding AND output use emergent dimensions discovered from data.
NO hardcoded semantic labels - everything emerges from the corpus.

Chain 1: Understanding - discovers dimensions from agent behaviors
Chain 2: Output - discovers dimensions from sentence patterns

The chatbot uses Chain 1 to understand queries, then Chain 2 to generate
natural language responses.
"""

import json
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class KnowledgeFrame:
    """A piece of knowledge."""
    text: str
    agent: str
    understanding_pos: Optional[np.ndarray] = None
    output_pos: Optional[np.ndarray] = None


class UnderstandingChain:
    """Discovers dimensions from agent behavior patterns."""
    
    def __init__(self):
        self.frames: List[KnowledgeFrame] = []
        self.agent_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.dimensions: List[Dict] = []
        self.agents: List[str] = []
        self.U: Optional[np.ndarray] = None
    
    def ingest(self, frames: List[KnowledgeFrame]):
        """Ingest frames and extract patterns."""
        self.frames = frames
        for frame in frames:
            if frame.agent and len(frame.agent) > 2:
                for word in frame.text.lower().split():
                    word_clean = re.sub(r'[^a-z]', '', word)
                    if len(word_clean) >= 3:
                        self.agent_actions[frame.agent][word_clean] += 1
    
    def learn(self, min_var: float = 0.02, max_dims: int = 12):
        """Discover dimensions via SVD."""
        valid = {a: v for a, v in self.agent_actions.items() if sum(v.values()) >= 3}
        if len(valid) < 3:
            return
        
        self.agents = list(valid.keys())
        features = sorted(set(f for v in valid.values() for f in v.keys()))
        
        X = np.zeros((len(self.agents), len(features)))
        for i, agent in enumerate(self.agents):
            total = sum(valid[agent].values())
            for j, feat in enumerate(features):
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
            self.dimensions.append({
                'name': f'U{i+1}',
                'variance': float(var_ratios[i]),
                'neg_pole': self.agents[np.argmin(pos)],
                'pos_pole': self.agents[np.argmax(pos)],
                'positions': {self.agents[j]: float(pos[j]) for j in range(len(self.agents))},
            })
        
        # Update frame positions
        for frame in self.frames:
            if frame.agent in self.agents:
                idx = self.agents.index(frame.agent)
                frame.understanding_pos = self.U[idx, :len(self.dimensions)]
    
    def get_position(self, concept: str) -> Optional[np.ndarray]:
        concept = concept.lower()
        for agent in self.agents:
            if concept in agent or agent in concept:
                return np.array([d['positions'].get(agent, 0) for d in self.dimensions])
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
    
    def find_opposite(self, agent: str) -> Optional[str]:
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
        return opposite
    
    def find_frames(self, concepts: List[str], k: int = 5) -> List[KnowledgeFrame]:
        positions = [self.get_position(c) for c in concepts]
        positions = [p for p in positions if p is not None]
        
        if not positions:
            scored = []
            for frame in self.frames:
                score = sum(1 for c in concepts if c in frame.text.lower())
                if score > 0:
                    scored.append((score, frame))
            scored.sort(key=lambda x: -x[0])
            return [f for _, f in scored[:k]]
        
        query_pos = np.mean(positions, axis=0)
        scored = []
        for frame in self.frames:
            if frame.understanding_pos is not None:
                dist = np.linalg.norm(frame.understanding_pos - query_pos)
                scored.append((-dist, frame))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:k]]


class OutputChain:
    """Discovers dimensions from sentence patterns."""
    
    def __init__(self):
        self.sentences: List[KnowledgeFrame] = []
        self.dimensions: List[Dict] = []
        self.feature_names: List[str] = []
        self.U: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
    
    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract sentence features."""
        words = text.split()
        text_lower = text.lower()
        
        return {
            'len_short': 1.0 if len(words) < 8 else 0.0,
            'len_medium': 1.0 if 8 <= len(words) < 15 else 0.0,
            'len_long': 1.0 if len(words) >= 15 else 0.0,
            'has_comma': 1.0 if ',' in text else 0.0,
            'has_question': 1.0 if '?' in text else 0.0,
            'starts_name': 1.0 if text[0].isupper() and not text_lower.startswith(('the ', 'a ')) else 0.0,
            'has_past': 1.0 if re.search(r'\b\w+ed\b', text_lower) else 0.0,
            'has_ing': 1.0 if re.search(r'\b\w+ing\b', text_lower) else 0.0,
            'has_ly': 1.0 if re.search(r'\b\w+ly\b', text_lower) else 0.0,
            'has_is': 1.0 if ' is ' in text_lower or ' are ' in text_lower else 0.0,
            'has_and': 1.0 if ' and ' in text_lower else 0.0,
            'has_with': 1.0 if ' with ' in text_lower else 0.0,
        }
    
    def ingest(self, frames: List[KnowledgeFrame]):
        """Ingest frames."""
        self.sentences = [f for f in frames if len(f.text) > 10]
    
    def learn(self, min_var: float = 0.03, max_dims: int = 8):
        """Discover output dimensions."""
        if len(self.sentences) < 10:
            return
        
        # Extract features
        all_features = []
        for sf in self.sentences:
            all_features.append(self._extract_features(sf.text))
        
        self.feature_names = sorted(all_features[0].keys())
        
        X = np.zeros((len(self.sentences), len(self.feature_names)))
        for i, feats in enumerate(all_features):
            for j, fname in enumerate(self.feature_names):
                X[i, j] = feats.get(fname, 0.0)
        
        X_centered = X - X.mean(axis=0)
        self.U, S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        total_var = np.sum(S ** 2)
        var_ratios = (S ** 2) / total_var
        
        self.dimensions = []
        for i in range(min(len(S), max_dims)):
            if var_ratios[i] < min_var:
                break
            
            weights = self.Vt[i]
            neg_feats = [self.feature_names[j] for j in np.argsort(weights)[:2]]
            pos_feats = [self.feature_names[j] for j in np.argsort(weights)[-2:]]
            
            positions = self.U[:, i]
            neg_example = self.sentences[np.argmin(positions)].text
            pos_example = self.sentences[np.argmax(positions)].text
            
            self.dimensions.append({
                'name': f'O{i+1}',
                'variance': float(var_ratios[i]),
                'neg_features': neg_feats,
                'pos_features': pos_feats,
                'neg_example': neg_example[:80],
                'pos_example': pos_example[:80],
            })
        
        # Update positions
        for i, sf in enumerate(self.sentences):
            sf.output_pos = self.U[i, :len(self.dimensions)]
    
    def find_template(self, target_features: Dict[str, float], k: int = 3) -> List[str]:
        """Find sentences matching target features."""
        if not self.sentences or self.Vt is None:
            return []
        
        # Project target features
        target_vec = np.array([target_features.get(f, 0.5) for f in self.feature_names])
        target_vec_centered = target_vec - 0.5
        target_pos = target_vec_centered @ self.Vt[:len(self.dimensions)].T
        
        # Find closest sentences
        scored = []
        for sf in self.sentences:
            if sf.output_pos is not None:
                dist = np.linalg.norm(sf.output_pos - target_pos)
                scored.append((dist, sf.text))
        
        scored.sort(key=lambda x: x[0])
        return [text for _, text in scored[:k]]


class DualEmergentChatbot:
    """Chatbot with dual emergent gear chains."""
    
    def __init__(self):
        self.understanding = UnderstandingChain()
        self.output = OutputChain()
        self.frames: List[KnowledgeFrame] = []
    
    def load(self, corpus_paths: List[str]):
        """Load corpora into both chains."""
        print("Loading knowledge base...")
        
        for path in corpus_paths:
            if not Path(path).exists():
                continue
            
            with open(path) as f:
                corpus = json.load(f)
            
            for frame in corpus.get('frames', []):
                text = frame.get('text', '').strip()
                agent = frame.get('agent', '').lower()
                if text and len(text) > 5:
                    self.frames.append(KnowledgeFrame(text=text, agent=agent))
        
        print(f"  Loaded {len(self.frames)} frames")
        
        # Train both chains
        print("\nTraining understanding chain...")
        self.understanding.ingest(self.frames)
        self.understanding.learn()
        print(f"  Discovered {len(self.understanding.dimensions)} understanding dimensions")
        
        print("\nTraining output chain...")
        self.output.ingest(self.frames)
        self.output.learn()
        print(f"  Discovered {len(self.output.dimensions)} output dimensions")
    
    def _extract_concepts(self, query: str) -> List[str]:
        query_lower = query.lower()
        return [a for a in self.understanding.agents if a in query_lower and len(a) > 2]
    
    def _detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['compare', 'difference', 'between', 'vs']):
            return 'compare'
        if any(w in q for w in ['similar', 'like', 'related']):
            return 'similar'
        if any(w in q for w in ['opposite', 'contrary']):
            return 'opposite'
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
    
    def chat(self, query: str) -> str:
        """Process query using dual gear chains."""
        concepts = self._extract_concepts(query)
        intent = self._detect_intent(query)
        
        if not concepts:
            sample = ', '.join([self._format_name(a) for a in self.understanding.agents[:8]])
            return f"I don't recognize any concepts in your query.\n\nKnown concepts: {sample}..."
        
        frames = self.understanding.find_frames(concepts, k=3)
        
        # Generate response based on intent
        if intent == 'compare' and len(concepts) >= 2:
            return self._respond_compare(concepts[0], concepts[1], frames)
        elif intent == 'similar':
            return self._respond_similar(concepts[0])
        elif intent == 'opposite':
            return self._respond_opposite(concepts[0])
        else:
            return self._respond_describe(concepts[0], frames)
    
    def _respond_describe(self, agent: str, frames: List[KnowledgeFrame]) -> str:
        """Generate description response."""
        name = self._format_name(agent)
        similar = self.understanding.find_similar(agent, k=3)
        opposite = self.understanding.find_opposite(agent)
        
        parts = []
        
        if similar:
            similar_names = self._format_list([s[0] for s in similar])
            parts.append(f"{name} shares characteristics with {similar_names}.")
        
        if opposite:
            parts.append(f"In contrast, {name} differs notably from {self._format_name(opposite)}.")
        
        # Add dimensional traits (using emergent dimension poles)
        pos = self.understanding.get_position(agent)
        if pos is not None:
            traits = []
            for i, dim in enumerate(self.understanding.dimensions[:2]):
                if i < len(pos) and abs(pos[i]) > 0.15:
                    pole = dim['pos_pole'] if pos[i] > 0 else dim['neg_pole']
                    traits.append(self._format_name(pole))
            if traits:
                parts.append(f"{name} exhibits qualities associated with {self._format_list(traits)}.")
        
        # Add frame evidence
        if frames:
            parts.append("")
            parts.append("From the knowledge base:")
            for frame in frames[:2]:
                text = frame.text[:100] + "..." if len(frame.text) > 100 else frame.text
                parts.append(f"  • {text}")
        
        return '\n'.join(parts) if parts else f"{name} is a known concept."
    
    def _respond_compare(self, a1: str, a2: str, frames: List[KnowledgeFrame]) -> str:
        """Generate comparison response."""
        n1, n2 = self._format_name(a1), self._format_name(a2)
        
        pos1 = self.understanding.get_position(a1)
        pos2 = self.understanding.get_position(a2)
        
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
        
        # Find most different dimension
        diff = pos2 - pos1
        max_idx = np.argmax(np.abs(diff))
        if max_idx < len(self.understanding.dimensions):
            dim = self.understanding.dimensions[max_idx]
            if diff[max_idx] > 0:
                trait1 = self._format_name(dim['neg_pole'])
                trait2 = self._format_name(dim['pos_pole'])
            else:
                trait1 = self._format_name(dim['pos_pole'])
                trait2 = self._format_name(dim['neg_pole'])
            parts.append(f"Where {n1} tends toward {trait1}, {n2} leans toward {trait2}.")
        
        if frames:
            parts.append("")
            parts.append("Evidence:")
            for frame in frames[:2]:
                text = frame.text[:80] + "..." if len(frame.text) > 80 else frame.text
                parts.append(f"  • {text}")
        
        return '\n'.join(parts)
    
    def _respond_similar(self, agent: str) -> str:
        """Generate similarity response."""
        name = self._format_name(agent)
        similar = self.understanding.find_similar(agent, k=5)
        
        if not similar:
            return f"No similar concepts found for {name}."
        
        parts = [f"Concepts similar to {name}:"]
        for other, dist in similar:
            closeness = "very close" if dist < 0.3 else "fairly close" if dist < 0.6 else "related"
            parts.append(f"  • {self._format_name(other)} ({closeness})")
        
        return '\n'.join(parts)
    
    def _respond_opposite(self, agent: str) -> str:
        """Generate opposite response."""
        name = self._format_name(agent)
        opposite = self.understanding.find_opposite(agent)
        
        if opposite:
            return f"The opposite of {name} is {self._format_name(opposite)}."
        return f"No clear opposite found for {name}."
    
    def interactive(self):
        """Run interactive session."""
        print("\n" + "═" * 70)
        print(" DUAL EMERGENT GEAR CHAIN CHATBOT ".center(70, "═"))
        print(" Understanding + Output: Both Emergent ".center(70))
        print("═" * 70)
        print(f"\nKnowledge: {len(self.frames)} frames")
        print(f"Understanding dimensions: {len(self.understanding.dimensions)}")
        print(f"Output dimensions: {len(self.output.dimensions)}")
        print("\nCommands: 'dims', 'quit'\n")
        
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
                print("\nUnderstanding dimensions:")
                for d in self.understanding.dimensions[:5]:
                    print(f"  {d['name']}: {d['neg_pole']} ↔ {d['pos_pole']}")
                print("\nOutput dimensions:")
                for d in self.output.dimensions[:5]:
                    print(f"  {d['name']}: {d['neg_features']} ↔ {d['pos_features']}")
                continue
            
            print(f"\n{self.chat(query)}\n")


def main():
    base = Path(__file__).parent.parent
    sources = [
        str(base / "truthspace_lcm/gears/corpus/corpus_llm_live.json"),
        str(base / "truthspace_lcm/gears/corpus/corpus_knowledge.json"),
        str(base / "truthspace_lcm/corpus_curated.json"),
    ]
    
    bot = DualEmergentChatbot()
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
