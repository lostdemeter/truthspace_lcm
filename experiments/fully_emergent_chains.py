#!/usr/bin/env python3
"""
Fully Emergent Gear Chains

Replaces all hardcoded structures with emergent discovery:

1. StopwordChain - discovers non-informative words from frequency distribution
2. FeatureLabelChain - discovers verb->trait mappings from co-occurrence
3. TemplateChain - discovers sentence patterns from corpus structure

All use the same EmergentDimensionChain pattern: ingest data, discover dimensions.
"""

import json
import numpy as np
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.gears.core.emergent_chain import EmergentDimensionChain, DataItem


class StopwordChain(EmergentDimensionChain):
    """
    Discovers stopwords from word frequency distribution.
    
    Insight: Stopwords follow Zipf's law - they're the most frequent words
    that appear uniformly across all agents (low discriminative power).
    
    A word is a stopword if:
    1. It appears in many different agents (high spread)
    2. It has high frequency overall
    3. It has low variance across agents (uniform distribution)
    """
    
    def __init__(self, name: str = "StopwordChain"):
        super().__init__(name)
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.word_agent_counts: Dict[str, Set[str]] = defaultdict(set)
        self.total_words = 0
        self.discovered_stopwords: Set[str] = set()
        self.min_group_count = 1
    
    def extract_features(self, item: Any) -> Dict[str, float]:
        """Extract word frequency features."""
        if isinstance(item, dict):
            text = item.get('text', '')
        else:
            text = str(item)
        
        features = {}
        words = text.lower().split()
        for word in words:
            word_clean = re.sub(r'[^a-z]', '', word)
            if len(word_clean) >= 2:
                features[word_clean] = features.get(word_clean, 0) + 1.0
                self.word_counts[word_clean] += 1
                self.total_words += 1
        
        return features
    
    def get_item_id(self, item: Any) -> str:
        """Use agent as group."""
        if isinstance(item, dict):
            agent = item.get('agent', '')
            # Track which agents use which words
            text = item.get('text', '')
            for word in text.lower().split():
                word_clean = re.sub(r'[^a-z]', '', word)
                if len(word_clean) >= 2:
                    self.word_agent_counts[word_clean].add(agent)
            return agent.lower() if agent else 'unknown'
        return 'unknown'
    
    def discover_stopwords(self, spread_threshold: float = 0.3, freq_threshold: float = 0.001) -> Set[str]:
        """
        Discover stopwords based on emergent patterns.
        
        A word is a stopword if:
        - It appears in > spread_threshold of all agents
        - It has frequency > freq_threshold of total words
        """
        if not self.groups or self.total_words == 0:
            return set()
        
        n_agents = len(self.groups)
        stopwords = set()
        
        for word, count in self.word_counts.items():
            # Calculate spread (fraction of agents using this word)
            spread = len(self.word_agent_counts[word]) / n_agents
            
            # Calculate frequency
            freq = count / self.total_words
            
            # High spread + high frequency = stopword
            if spread > spread_threshold and freq > freq_threshold:
                stopwords.add(word)
        
        self.discovered_stopwords = stopwords
        return stopwords
    
    def is_stopword(self, word: str) -> bool:
        """Check if a word is a discovered stopword."""
        return word.lower() in self.discovered_stopwords


class FeatureLabelChain:
    """
    Discovers verb->trait mappings from co-occurrence patterns.
    
    Insight: Verbs that co-occur with similar agents have similar meanings.
    We cluster verbs by their agent distribution using SVD.
    
    The "label" for a verb cluster is the most frequent verb in that cluster.
    """
    
    def __init__(self, name: str = "FeatureLabelChain"):
        self.name = name
        self.verb_agents: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.verb_labels: Dict[str, str] = {}
        self.label_clusters: Dict[str, List[str]] = defaultdict(list)
        self.verbs: List[str] = []
        self.agents: List[str] = []
    
    def ingest_item(self, item: Any):
        """Extract verb-agent co-occurrences."""
        if isinstance(item, dict):
            text = item.get('text', '')
            agent = item.get('agent', '').lower()
        else:
            return
        
        if not agent:
            return
        
        words = text.lower().split()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^a-z]', '', word)
            if len(word_clean) < 4:
                continue
            
            # Heuristic: verbs are often in position 1 or end in verb suffixes
            is_verb = (i == 1) or any(word_clean.endswith(s) for s in ['ing', 'ed', 'es', 'ly'])
            
            if is_verb:
                self.verb_agents[word_clean][agent] += 1
    
    def learn_dimensions(self, min_variance: float = 0.03, max_dims: int = 10) -> int:
        """Learn verb clusters via SVD on verb-agent matrix."""
        # Filter verbs with enough occurrences
        min_count = 2
        valid_verbs = {v: agents for v, agents in self.verb_agents.items() 
                       if sum(agents.values()) >= min_count}
        
        if len(valid_verbs) < 5:
            return 0
        
        self.verbs = list(valid_verbs.keys())
        self.agents = sorted(set(a for agents in valid_verbs.values() for a in agents.keys()))
        
        # Build verb-agent matrix
        X = np.zeros((len(self.verbs), len(self.agents)))
        for i, verb in enumerate(self.verbs):
            total = sum(valid_verbs[verb].values())
            for j, agent in enumerate(self.agents):
                X[i, j] = valid_verbs[verb].get(agent, 0) / total
        
        # SVD
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Cluster verbs by their position on first few dimensions
        n_dims = min(5, len(S))
        
        for i, verb in enumerate(self.verbs):
            pos = U[i, :n_dims]
            
            # Find dominant dimension
            dominant = int(np.argmax(np.abs(pos)))
            sign = 'pos' if pos[dominant] > 0 else 'neg'
            cluster_key = f"D{dominant}_{sign}"
            
            self.label_clusters[cluster_key].append(verb)
        
        # For each cluster, the label is the most frequent verb
        for cluster_key, verbs in self.label_clusters.items():
            if verbs:
                verb_freqs = [(v, sum(self.verb_agents[v].values())) for v in verbs]
                verb_freqs.sort(key=lambda x: -x[1])
                label = verb_freqs[0][0]
                
                for verb in verbs:
                    self.verb_labels[verb] = label
        
        return len(self.label_clusters)
    
    def get_label(self, verb: str) -> str:
        """Get the emergent label for a verb."""
        verb = verb.lower()
        
        if verb in self.verb_labels:
            return self.verb_labels[verb]
        
        # Try without suffixes
        for suffix in ['ing', 'ed', 'es', 's', 'ly']:
            if verb.endswith(suffix) and len(verb) > len(suffix) + 2:
                base = verb[:-len(suffix)]
                if base in self.verb_labels:
                    return self.verb_labels[base]
        
        return verb


class TemplateChain(EmergentDimensionChain):
    """
    Discovers sentence templates from corpus structure.
    
    Insight: Sentences have structural patterns that can be abstracted.
    We discover templates by:
    1. Extracting sentence structure (POS-like patterns)
    2. Clustering similar structures
    3. Using exemplars as templates
    """
    
    def __init__(self, name: str = "TemplateChain"):
        super().__init__(name)
        self.sentence_patterns: Dict[str, List[str]] = defaultdict(list)
        self.templates: Dict[str, str] = {}  # pattern -> template sentence
        self.min_group_count = 1
    
    def extract_features(self, item: Any) -> Dict[str, float]:
        """Extract structural features from a sentence."""
        if isinstance(item, dict):
            text = item.get('text', '')
        else:
            text = str(item)
        
        if not text:
            return {}
        
        words = text.split()
        features = {}
        
        # Length bucket
        if len(words) < 8:
            features['len_short'] = 1.0
        elif len(words) < 15:
            features['len_medium'] = 1.0
        else:
            features['len_long'] = 1.0
        
        # First word pattern
        if words:
            first = words[0].lower()
            if first[0].isupper():
                features['starts_proper'] = 1.0
            if first in ['the', 'a', 'an']:
                features['starts_article'] = 1.0
        
        # Verb position (second word often verb)
        if len(words) > 1:
            second = re.sub(r'[^a-z]', '', words[1].lower())
            if second.endswith(('s', 'es', 'ed', 'ing')):
                features['verb_second'] = 1.0
        
        # Punctuation
        if ',' in text:
            features['has_comma'] = 1.0
        if '?' in text:
            features['is_question'] = 1.0
        
        # Complexity
        if any(w.lower() in ['and', 'but', 'or', 'while'] for w in words):
            features['has_conjunction'] = 1.0
        if any(w.lower() in ['who', 'which', 'that'] for w in words):
            features['has_relative'] = 1.0
        
        return features
    
    def get_item_id(self, item: Any) -> str:
        """Create a structural pattern ID."""
        if isinstance(item, dict):
            text = item.get('text', '')
        else:
            text = str(item)
        
        if not text or len(text) < 10:
            return ''
        
        # Create a simple structural pattern
        words = text.split()
        pattern_parts = []
        
        # First word type
        if words[0][0].isupper():
            pattern_parts.append('PROPER')
        else:
            pattern_parts.append('lower')
        
        # Length category
        if len(words) < 8:
            pattern_parts.append('SHORT')
        elif len(words) < 15:
            pattern_parts.append('MED')
        else:
            pattern_parts.append('LONG')
        
        # Has comma?
        if ',' in text:
            pattern_parts.append('COMMA')
        
        pattern = '_'.join(pattern_parts)
        
        # Store the sentence as an example of this pattern
        self.sentence_patterns[pattern].append(text)
        
        return pattern
    
    def learn_dimensions(self, min_variance: float = 0.02, max_dims: int = 10) -> int:
        """Learn dimensions and extract templates."""
        count = super().learn_dimensions(min_variance, max_dims)
        self._extract_templates()
        return count
    
    def _extract_templates(self):
        """Extract template sentences for each pattern."""
        for pattern, sentences in self.sentence_patterns.items():
            if sentences:
                # Use the shortest sentence as template (most generic)
                sentences.sort(key=len)
                self.templates[pattern] = sentences[0]
    
    def get_template(self, style: str = 'default') -> Optional[str]:
        """Get a template matching a style."""
        style_patterns = {
            'short': 'SHORT',
            'long': 'LONG',
            'complex': 'COMMA',
            'simple': 'SHORT',
        }
        
        target = style_patterns.get(style, '')
        
        # Find matching template
        for pattern, template in self.templates.items():
            if target in pattern:
                return template
        
        # Return any template
        if self.templates:
            return list(self.templates.values())[0]
        
        return None
    
    def adapt_template(self, template: str, substitutions: Dict[str, str]) -> str:
        """Adapt a template with substitutions."""
        result = template
        for old, new in substitutions.items():
            result = re.sub(rf'\b{re.escape(old)}\b', new, result, flags=re.IGNORECASE)
        return result


class FullyEmergentSemanticChain(EmergentDimensionChain):
    """
    Semantic chain that uses emergent sub-chains for everything.
    
    No hardcoded:
    - Stopwords (discovered from frequency)
    - Feature labels (discovered from verb clustering)
    - Templates (discovered from sentence patterns)
    """
    
    def __init__(self, name: str = "FullyEmergentSemanticChain"):
        super().__init__(name)
        
        # Sub-chains for emergent discovery
        self.stopword_chain = StopwordChain()
        self.label_chain = FeatureLabelChain()
        self.template_chain = TemplateChain()
        
        self.agent_key = 'agent'
        self.text_key = 'text'
    
    def ingest_corpus(self, corpus_path: str, frame_key: str = 'frames') -> int:
        """Ingest into all sub-chains."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        items = corpus.get(frame_key, [])
        
        # Ingest into all chains
        for item in items:
            self.stopword_chain.ingest_item(item)
            self.label_chain.ingest_item(item)
            self.template_chain.ingest_item(item)
        
        # Also ingest into self
        return self.ingest_batch(items)
    
    def extract_features(self, item: Any) -> Dict[str, float]:
        """Extract features, excluding discovered stopwords."""
        if isinstance(item, dict):
            text = item.get(self.text_key, '')
        else:
            text = str(item)
        
        features = {}
        words = text.lower().split()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^a-z]', '', word)
            
            if len(word_clean) < 4:
                continue
            
            # Skip discovered stopwords
            if self.stopword_chain.is_stopword(word_clean):
                continue
            
            # Focus on verbs
            is_verb = (i == 1) or any(word_clean.endswith(s) for s in ['ing', 'ed', 'es', 'ly'])
            
            if is_verb:
                features[word_clean] = features.get(word_clean, 0) + 1.0
        
        return features
    
    def get_item_id(self, item: Any) -> str:
        """Get agent ID."""
        if isinstance(item, dict):
            agent = item.get(self.agent_key, '')
            return agent.lower() if agent else ''
        return ''
    
    def learn_dimensions(self, min_variance: float = 0.02, max_dims: int = 15) -> int:
        """Learn all emergent structures."""
        # First, learn stopword chain dimensions (needed for frequency analysis)
        self.stopword_chain.learn_dimensions(min_variance=0.01, max_dims=5)
        self.stopword_chain.discover_stopwords(spread_threshold=0.2, freq_threshold=0.005)
        print(f"  Discovered {len(self.stopword_chain.discovered_stopwords)} stopwords")
        
        # Learn verb clusters for labeling
        self.label_chain.learn_dimensions(min_variance=0.03, max_dims=10)
        print(f"  Discovered {len(self.label_chain.label_clusters)} verb clusters, {len(self.label_chain.verb_labels)} labels")
        
        # Learn sentence templates
        self.template_chain.learn_dimensions(min_variance=0.05, max_dims=6)
        print(f"  Discovered {len(self.template_chain.templates)} templates")
        
        # Now learn main dimensions (will use discovered stopwords)
        return super().learn_dimensions(min_variance, max_dims)
    
    def get_label(self, feature: str) -> str:
        """Get emergent label for a feature."""
        return self.label_chain.get_label(feature)
    
    def describe_traits(self, concept: str) -> List[str]:
        """Get semantic traits using emergent labels."""
        pos = self.get_position(concept)
        if pos is None:
            return []
        
        traits = []
        seen_labels = set()
        
        for i, dim in enumerate(self.dimensions[:5]):
            if i < len(pos) and abs(pos[i]) > 0.15:
                # Get the defining features for this dimension
                features = dim.positive_features if pos[i] > 0 else dim.negative_features
                
                # Try each feature until we find a good label
                for feature in features:
                    label = self.get_label(feature)
                    # Skip if it's just the raw feature or a stopword
                    if label != feature and label not in seen_labels:
                        if not self.stopword_chain.is_stopword(label):
                            traits.append(label)
                            seen_labels.add(label)
                            break
        
        return traits[:3]
    
    def get_template(self, style: str = 'default') -> Optional[str]:
        """Get an emergent template."""
        return self.template_chain.get_template(style)
    
    def get_relevant_content(self, concepts: List[str], k: int = 5) -> List[str]:
        """Get content relevant to concepts."""
        items = self.find_items_for_groups(concepts, k)
        results = []
        for item in items:
            content = item.content
            # Handle dict content
            if isinstance(content, dict):
                results.append(content.get('text', str(content)))
            else:
                results.append(str(content))
        return results


class FullyEmergentChatbot:
    """
    Chatbot with NO hardcoded structures.
    
    Everything is emergent:
    - Stopwords discovered from frequency
    - Verb labels discovered from clustering
    - Templates discovered from sentence patterns
    - Dimensions discovered from behavior
    """
    
    def __init__(self):
        self.semantic = FullyEmergentSemanticChain()
        self.corpus_loaded = False
    
    def load_corpus(self, corpus_path: str) -> int:
        """Load a corpus."""
        count = self.semantic.ingest_corpus(corpus_path)
        self.corpus_loaded = True
        return count
    
    def train(self) -> Dict[str, int]:
        """Train all emergent structures."""
        dims = self.semantic.learn_dimensions()
        return {
            'dimensions': dims,
            'stopwords': len(self.semantic.stopword_chain.discovered_stopwords),
            'verb_labels': len(self.semantic.label_chain.verb_labels),
            'templates': len(self.semantic.template_chain.templates),
        }
    
    def _format_name(self, name: str) -> str:
        return name.replace('_', ' ').title()
    
    def _format_list(self, items: List[str]) -> str:
        items = [self._format_name(i) for i in items]
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"
    
    def _extract_concepts(self, query: str) -> List[str]:
        query_lower = query.lower()
        return [g for g in self.semantic.groups if g in query_lower and len(g) > 2]
    
    def _detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['compare', 'difference', 'between', 'vs']):
            return 'compare'
        if any(w in q for w in ['similar', 'like', 'related']):
            return 'similar'
        if any(w in q for w in ['opposite', 'contrary']):
            return 'opposite'
        return 'describe'
    
    def chat(self, query: str) -> str:
        """Process query using fully emergent chains."""
        concepts = self._extract_concepts(query)
        intent = self._detect_intent(query)
        
        if not concepts:
            sample = ', '.join([self._format_name(g) for g in self.semantic.groups[:8]])
            return f"I don't recognize any concepts in your query.\n\nKnown concepts: {sample}..."
        
        if intent == 'compare' and len(concepts) >= 2:
            return self._respond_compare(concepts[0], concepts[1])
        elif intent == 'similar':
            return self._respond_similar(concepts[0])
        elif intent == 'opposite':
            return self._respond_opposite(concepts[0])
        else:
            return self._respond_describe(concepts[0])
    
    def _respond_describe(self, concept: str) -> str:
        """Generate description using emergent traits."""
        name = self._format_name(concept)
        traits = self.semantic.describe_traits(concept)
        similar = self.semantic.find_similar(concept, k=3)
        opposite = self.semantic.find_opposite(concept)
        content = self.semantic.get_relevant_content([concept], k=2)
        
        parts = []
        
        if traits:
            parts.append(f"{name} exhibits {self._format_list(traits)} qualities.")
        
        if similar:
            similar_names = self._format_list([s[0] for s in similar])
            parts.append(f"{name} shares characteristics with {similar_names}.")
        
        if opposite:
            parts.append(f"In contrast, {name} differs from {self._format_name(opposite[0])}.")
        
        if content:
            parts.append("")
            parts.append("From the knowledge base:")
            for text in content[:2]:
                text_short = text[:100] + "..." if len(text) > 100 else text
                parts.append(f"  • {text_short}")
        
        return '\n'.join(parts) if parts else f"{name} is a known concept."
    
    def _respond_compare(self, c1: str, c2: str) -> str:
        """Generate comparison using emergent labels."""
        n1, n2 = self._format_name(c1), self._format_name(c2)
        
        pos1 = self.semantic.get_position(c1)
        pos2 = self.semantic.get_position(c2)
        
        if pos1 is None or pos2 is None:
            return f"Cannot compare: missing data."
        
        dist = float(np.linalg.norm(pos2 - pos1))
        
        if dist < 0.3:
            sim = "closely related"
        elif dist < 0.6:
            sim = "somewhat similar"
        else:
            sim = "quite different"
        
        parts = [f"{n1} and {n2} are {sim}."]
        
        # Get traits for each
        traits1 = self.semantic.describe_traits(c1)
        traits2 = self.semantic.describe_traits(c2)
        
        if traits1 and traits2:
            parts.append(f"{n1} is characterized by {self._format_list(traits1)} qualities.")
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
        result = self.semantic.find_opposite(concept)
        
        if result:
            return f"The opposite of {name} is {self._format_name(result[0])}."
        return f"No clear opposite found for {name}."
    
    def interactive(self):
        """Run interactive session."""
        stats = {
            'items': len(self.semantic.items),
            'groups': len(self.semantic.groups),
            'dimensions': len(self.semantic.dimensions),
            'stopwords': len(self.semantic.stopword_chain.discovered_stopwords),
            'verb_labels': len(self.semantic.label_chain.verb_labels),
            'templates': len(self.semantic.template_chain.templates),
        }
        
        print("\n" + "═" * 70)
        print(" FULLY EMERGENT CHATBOT ".center(70, "═"))
        print(" No Hardcoded Structures ".center(70))
        print("═" * 70)
        print(f"\nKnowledge: {stats['items']} items, {stats['groups']} concepts")
        print(f"Emergent: {stats['stopwords']} stopwords, {stats['verb_labels']} verb labels, {stats['templates']} templates")
        print(f"Dimensions: {stats['dimensions']}")
        print("\nCommands: 'stats', 'quit'\n")
        
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
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                continue
            
            print(f"\n{self.chat(query)}\n")


def test_fully_emergent():
    """Test the fully emergent system."""
    print("=" * 70)
    print("FULLY EMERGENT CHATBOT")
    print("=" * 70)
    
    bot = FullyEmergentChatbot()
    
    # Load corpus
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if corpus_path.exists():
        print(f"\nLoading corpus: {corpus_path}")
        count = bot.load_corpus(str(corpus_path))
        print(f"  Ingested {count} items")
    
    # Train
    print("\nTraining emergent structures...")
    stats = bot.train()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test queries
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
    
    # Interactive
    bot.interactive()
    
    return bot


if __name__ == "__main__":
    bot = test_fully_emergent()
