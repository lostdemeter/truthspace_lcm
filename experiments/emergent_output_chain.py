#!/usr/bin/env python3
"""
Emergent Output Gear Chain

Discovers its own dimensions for natural language generation, just like
the understanding chain discovers dimensions for comprehension.

Key insight: Sentences have patterns that can be discovered via SVD:
- Some sentences are formal vs casual
- Some are descriptive vs action-oriented
- Some are simple vs complex
- Some are declarative vs interrogative

We discover these dimensions from the corpus, then use them to
condition output generation.
"""

import json
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SentenceFrame:
    """A sentence with its feature vector and position."""
    text: str
    agent: str
    features: Dict[str, float]
    position: Optional[np.ndarray] = None


class EmergentOutputChain:
    """
    Output gear chain that discovers its own dimensions from sentence patterns.
    
    Instead of hardcoding style labels, we let the structure emerge from data.
    """
    
    def __init__(self):
        # Sentence data
        self.sentences: List[SentenceFrame] = []
        
        # Feature extraction
        self.sentence_features: Dict[str, Dict[str, float]] = {}
        
        # Discovered dimensions
        self.dimensions: List[Dict] = []
        self.feature_names: List[str] = []
        
        # SVD components
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        # Sentence templates by dimension position
        self.templates_by_position: Dict[str, List[str]] = defaultdict(list)
        
        # N-gram model for fluency
        self.bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.trigrams: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _extract_sentence_features(self, text: str) -> Dict[str, float]:
        """Extract features from a sentence for dimension discovery."""
        features = {}
        
        words = text.split()
        word_count = len(words)
        
        # Length features
        features['length_short'] = 1.0 if word_count < 8 else 0.0
        features['length_medium'] = 1.0 if 8 <= word_count < 15 else 0.0
        features['length_long'] = 1.0 if word_count >= 15 else 0.0
        
        # Punctuation features
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_exclamation'] = 1.0 if '!' in text else 0.0
        features['has_comma'] = 1.0 if ',' in text else 0.0
        features['has_semicolon'] = 1.0 if ';' in text else 0.0
        
        # Structure features
        text_lower = text.lower()
        features['starts_with_the'] = 1.0 if text_lower.startswith('the ') else 0.0
        features['starts_with_a'] = 1.0 if text_lower.startswith('a ') or text_lower.startswith('an ') else 0.0
        features['starts_with_name'] = 1.0 if text[0].isupper() and not text_lower.startswith(('the ', 'a ', 'an ')) else 0.0
        
        # Verb tense features (heuristic)
        features['has_past_tense'] = 1.0 if re.search(r'\b\w+ed\b', text_lower) else 0.0
        features['has_present_tense'] = 1.0 if re.search(r'\b\w+s\b', text_lower) else 0.0
        features['has_progressive'] = 1.0 if re.search(r'\b\w+ing\b', text_lower) else 0.0
        
        # Complexity features
        features['has_conjunction'] = 1.0 if any(w in text_lower.split() for w in ['and', 'but', 'or', 'while', 'although']) else 0.0
        features['has_relative'] = 1.0 if any(w in text_lower.split() for w in ['who', 'which', 'that', 'whose', 'whom']) else 0.0
        features['has_preposition'] = 1.0 if any(w in text_lower.split() for w in ['in', 'on', 'at', 'by', 'with', 'from', 'to', 'for']) else 0.0
        
        # Style features
        features['has_adverb'] = 1.0 if re.search(r'\b\w+ly\b', text_lower) else 0.0
        features['has_adjective_before_noun'] = 1.0 if re.search(r'\b(the|a|an)\s+\w+\s+\w+', text_lower) else 0.0
        
        # Content type features
        features['is_descriptive'] = 1.0 if any(w in text_lower for w in ['is', 'are', 'was', 'were', 'has', 'have']) else 0.0
        features['is_action'] = 1.0 if re.search(r'^[A-Z]\w+\s+\w+s?\b', text) else 0.0
        features['is_comparison'] = 1.0 if any(w in text_lower for w in ['than', 'more', 'less', 'similar', 'different', 'like']) else 0.0
        
        # Formality features
        features['has_contraction'] = 1.0 if "'" in text else 0.0
        features['starts_capital'] = 1.0 if text[0].isupper() else 0.0
        features['ends_period'] = 1.0 if text.rstrip().endswith('.') else 0.0
        
        return features
    
    def ingest_corpus(self, corpus_path: str):
        """Ingest a corpus and extract sentence features."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        for frame in corpus.get('frames', []):
            text = frame.get('text', '').strip()
            agent = frame.get('agent', '').lower()
            
            if not text or len(text) < 10:
                continue
            
            features = self._extract_sentence_features(text)
            
            sf = SentenceFrame(
                text=text,
                agent=agent,
                features=features,
            )
            self.sentences.append(sf)
            
            # Learn n-grams
            self._learn_ngrams(text)
        
        print(f"Ingested {len(self.sentences)} sentences for output learning")
    
    def _learn_ngrams(self, text: str):
        """Learn n-gram patterns from text."""
        words = [re.sub(r'[^a-z]', '', w.lower()) for w in text.split()]
        words = [w for w in words if w]
        
        for i in range(len(words) - 1):
            self.bigrams[words[i]][words[i+1]] += 1
        
        for i in range(len(words) - 2):
            self.trigrams[(words[i], words[i+1])][words[i+2]] += 1
    
    def learn_dimensions(self, min_variance: float = 0.03, max_dims: int = 10):
        """Discover output dimensions from sentence patterns."""
        if len(self.sentences) < 10:
            print("Not enough sentences for dimension learning")
            return
        
        # Build feature matrix
        all_features = set()
        for sf in self.sentences:
            all_features.update(sf.features.keys())
        
        self.feature_names = sorted(all_features)
        n_sentences = len(self.sentences)
        n_features = len(self.feature_names)
        
        print(f"Learning output dimensions: {n_sentences} sentences × {n_features} features")
        
        X = np.zeros((n_sentences, n_features))
        for i, sf in enumerate(self.sentences):
            for j, feat in enumerate(self.feature_names):
                X[i, j] = sf.features.get(feat, 0.0)
        
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
            
            # Get feature weights for this dimension
            feature_weights = self.Vt[i]
            
            # Find defining features (positive and negative)
            sorted_indices = np.argsort(feature_weights)
            neg_features = [self.feature_names[j] for j in sorted_indices[:3]]
            pos_features = [self.feature_names[j] for j in sorted_indices[-3:]]
            
            # Name the dimension based on its features
            dim_name = self._name_dimension(neg_features, pos_features)
            
            # Get example sentences at each pole
            positions = self.U[:, i]
            neg_examples = [self.sentences[j].text for j in np.argsort(positions)[:2]]
            pos_examples = [self.sentences[j].text for j in np.argsort(positions)[-2:]]
            
            dim = {
                'index': i,
                'name': dim_name,
                'variance': float(var),
                'negative_features': neg_features,
                'positive_features': pos_features,
                'negative_examples': neg_examples,
                'positive_examples': pos_examples,
            }
            self.dimensions.append(dim)
        
        print(f"Discovered {len(self.dimensions)} output dimensions ({cumulative*100:.1f}% variance)")
        
        # Update sentence positions
        for i, sf in enumerate(self.sentences):
            sf.position = self.U[i, :len(self.dimensions)]
        
        # Organize templates by position
        self._organize_templates()
    
    def _name_dimension(self, neg_features: List[str], pos_features: List[str]) -> str:
        """Generate a name for a dimension based on its features."""
        # Feature to semantic label mapping
        feature_labels = {
            'length_short': 'brief',
            'length_medium': 'moderate',
            'length_long': 'elaborate',
            'has_question': 'interrogative',
            'has_exclamation': 'emphatic',
            'has_comma': 'complex',
            'has_past_tense': 'past',
            'has_present_tense': 'present',
            'has_progressive': 'ongoing',
            'has_conjunction': 'compound',
            'has_relative': 'embedded',
            'has_adverb': 'descriptive',
            'is_descriptive': 'descriptive',
            'is_action': 'action',
            'is_comparison': 'comparative',
            'has_contraction': 'informal',
            'starts_with_name': 'agent-focused',
            'starts_with_the': 'definite',
        }
        
        # Get labels for top features
        pos_label = feature_labels.get(pos_features[-1], pos_features[-1].replace('_', '-'))
        neg_label = feature_labels.get(neg_features[0], neg_features[0].replace('_', '-'))
        
        return f"{neg_label}_vs_{pos_label}"
    
    def _organize_templates(self):
        """Organize sentences as templates by their dimensional position."""
        self.templates_by_position.clear()
        
        for sf in self.sentences:
            if sf.position is None:
                continue
            
            # Quantize position to bins
            for i, dim in enumerate(self.dimensions):
                if i < len(sf.position):
                    pos = sf.position[i]
                    if pos < -0.3:
                        key = f"{dim['name']}_negative"
                    elif pos > 0.3:
                        key = f"{dim['name']}_positive"
                    else:
                        key = f"{dim['name']}_neutral"
                    
                    self.templates_by_position[key].append(sf.text)
    
    def get_sentence_position(self, text: str) -> np.ndarray:
        """Get the dimensional position of a sentence."""
        features = self._extract_sentence_features(text)
        
        # Project onto learned dimensions
        feature_vec = np.array([features.get(f, 0.0) for f in self.feature_names])
        
        # Center using training mean (approximate)
        feature_vec_centered = feature_vec - 0.5  # Rough centering
        
        # Project
        if self.Vt is not None:
            position = feature_vec_centered @ self.Vt[:len(self.dimensions)].T
            return position
        
        return np.zeros(len(self.dimensions))
    
    def find_similar_sentences(self, target_position: np.ndarray, k: int = 5) -> List[str]:
        """Find sentences similar to a target position."""
        scored = []
        
        for sf in self.sentences:
            if sf.position is not None:
                dist = np.linalg.norm(sf.position[:len(target_position)] - target_position)
                scored.append((dist, sf.text))
        
        scored.sort(key=lambda x: x[0])
        return [text for _, text in scored[:k]]
    
    def generate_from_template(self, template: str, substitutions: Dict[str, str]) -> str:
        """Generate a sentence by filling in a template."""
        result = template
        for key, value in substitutions.items():
            result = result.replace(f"{{{key}}}", value)
            # Also try replacing agent names
            result = re.sub(rf'\b{key}\b', value, result, flags=re.IGNORECASE)
        return result
    
    def condition_output(self, content: Dict[str, Any], style: str = 'neutral') -> str:
        """
        Condition semantic content into natural language using discovered dimensions.
        
        content: semantic content to express
        style: 'formal', 'casual', 'descriptive', 'action', etc.
        """
        output_type = content.get('type', 'description')
        agents = content.get('agents', [])
        properties = content.get('properties', {})
        frames = content.get('frames', [])
        
        parts = []
        
        if output_type == 'description' and agents:
            agent = agents[0]
            agent_name = agent.replace('_', ' ').title()
            
            similar = properties.get('similar', [])
            opposite = properties.get('opposite')
            traits = properties.get('traits', [])
            
            # Find template sentences that match desired style
            target_pos = self._style_to_position(style)
            similar_sentences = self.find_similar_sentences(target_pos, k=10)
            
            # Build description using patterns from similar sentences
            if similar:
                similar_str = self._format_list([s.replace('_', ' ').title() for s in similar[:3]])
                parts.append(f"{agent_name} shares characteristics with {similar_str}.")
            
            if opposite:
                opp_name = opposite.replace('_', ' ').title()
                parts.append(f"In contrast, {agent_name} differs notably from {opp_name}.")
            
            if traits:
                trait_str = self._format_list(traits[:2])
                parts.append(f"{agent_name} exhibits {trait_str} qualities.")
        
        elif output_type == 'comparison' and len(agents) >= 2:
            a1 = agents[0].replace('_', ' ').title()
            a2 = agents[1].replace('_', ' ').title()
            
            distance = properties.get('distance', 0.5)
            differences = properties.get('differences', [])
            
            # Similarity description
            if distance < 0.3:
                sim = "closely related"
            elif distance < 0.6:
                sim = "somewhat similar"
            elif distance < 1.0:
                sim = "notably different"
            else:
                sim = "quite distinct"
            
            parts.append(f"{a1} and {a2} are {sim}.")
            
            if differences:
                d = differences[0]
                trait1 = d.get('trait1', '')
                trait2 = d.get('trait2', '')
                if trait1 and trait2:
                    parts.append(f"Where {a1} tends toward {trait1}, {a2} leans toward {trait2}.")
        
        elif output_type == 'list':
            items = properties.get('items', agents)
            context = properties.get('context', '')
            
            items_formatted = [i.replace('_', ' ').title() for i in items]
            if context:
                parts.append(f"{context}:")
                for item in items_formatted[:5]:
                    parts.append(f"  • {item}")
            else:
                parts.append(self._format_list(items_formatted))
        
        # Add frame content
        if frames:
            parts.append("")
            parts.append("From the knowledge base:")
            for frame in frames[:2]:
                text = frame.text if hasattr(frame, 'text') else str(frame)
                if len(text) > 120:
                    text = text[:120].rsplit(' ', 1)[0] + "..."
                parts.append(f"  • {text}")
        
        return '\n'.join(parts) if parts else "I don't have enough information to respond."
    
    def _style_to_position(self, style: str) -> np.ndarray:
        """Convert a style name to a target dimensional position."""
        # Map styles to approximate positions
        style_positions = {
            'formal': np.array([0.5, -0.3, 0.2]),
            'casual': np.array([-0.3, 0.3, -0.2]),
            'descriptive': np.array([0.3, 0.0, 0.4]),
            'action': np.array([-0.2, 0.2, -0.3]),
            'neutral': np.array([0.0, 0.0, 0.0]),
        }
        
        pos = style_positions.get(style, style_positions['neutral'])
        
        # Pad to match dimension count
        if len(pos) < len(self.dimensions):
            pos = np.pad(pos, (0, len(self.dimensions) - len(pos)))
        
        return pos[:len(self.dimensions)]
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list naturally."""
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"


def test_emergent_output():
    """Test the emergent output chain."""
    print("=" * 70)
    print("EMERGENT OUTPUT CHAIN TEST")
    print("=" * 70)
    
    chain = EmergentOutputChain()
    
    # Load corpora
    base = Path(__file__).parent.parent
    sources = [
        base / "truthspace_lcm/gears/corpus/corpus_llm_live.json",
        base / "truthspace_lcm/gears/corpus/corpus_knowledge.json",
    ]
    
    for path in sources:
        if path.exists():
            chain.ingest_corpus(str(path))
    
    # Learn dimensions
    chain.learn_dimensions(min_variance=0.03, max_dims=8)
    
    # Show discovered dimensions
    print("\nDiscovered Output Dimensions:")
    for dim in chain.dimensions:
        print(f"\n  {dim['name']} ({dim['variance']*100:.1f}% variance)")
        print(f"    Negative features: {dim['negative_features']}")
        print(f"    Positive features: {dim['positive_features']}")
        print(f"    Example (neg): {dim['negative_examples'][0][:60]}...")
        print(f"    Example (pos): {dim['positive_examples'][0][:60]}...")
    
    # Test output conditioning
    print("\n" + "─" * 70)
    print("OUTPUT CONDITIONING TEST")
    print("─" * 70)
    
    content = {
        'type': 'description',
        'agents': ['holmes'],
        'properties': {
            'similar': ['watson', 'detective_work'],
            'opposite': 'villain',
            'traits': ['analytical', 'observant'],
        },
        'frames': [],
    }
    
    print("\nDescription (neutral style):")
    print(chain.condition_output(content, style='neutral'))
    
    print("\nDescription (formal style):")
    print(chain.condition_output(content, style='formal'))
    
    return chain


if __name__ == "__main__":
    chain = test_emergent_output()
