"""
Semantic Understanding Chain

A concrete implementation of EmergentDimensionChain for understanding
semantic content based on agent behavior patterns.

This chain discovers dimensions from how agents (characters, concepts)
behave in text, enabling similarity search and semantic analysis.

Includes semantic labeling to convert dimension features into
human-readable trait descriptions.

NOTE: Feature labels are now loaded from corpus file (corpus/feature_labels.json)
rather than being hard-coded. This follows the fail-fast philosophy.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from .base_chain import EmergentDimensionChain, DataItem


def _load_feature_labels() -> Dict[str, str]:
    """Load feature labels from corpus file."""
    corpus_path = Path(__file__).parent.parent / 'corpus' / 'feature_labels.json'
    if corpus_path.exists():
        with open(corpus_path) as f:
            data = json.load(f)
            # Filter out metadata keys starting with _
            return {k: v for k, v in data.items() if not k.startswith('_')}
    return {}  # Empty dict if file doesn't exist - fail-fast will show missing labels


# Load feature labels from corpus file (editable without code changes)
FEATURE_LABELS = _load_feature_labels()


class SemanticChain(EmergentDimensionChain):
    """
    Chain for understanding semantic content via agent behavior patterns.
    
    Features extracted:
    - Action words (verbs) associated with each agent
    - Behavioral patterns from text
    
    Groups: Agents (characters, concepts)
    """
    
    def __init__(self, name: str = "SemanticChain"):
        super().__init__(name)
        self.agent_key = 'agent'
        self.text_key = 'text'
    
    # Common stopwords to exclude from features
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
        'those', 'his', 'her', 'its', 'their', 'our', 'your', 'my', 'him',
        'them', 'us', 'me', 'who', 'whom', 'which', 'what', 'whose', 'it',
        'he', 'she', 'they', 'we', 'you', 'i', 'across', 'along', 'around',
    }
    
    def extract_features(self, item: Any) -> Dict[str, float]:
        """Extract behavioral features from a frame."""
        if isinstance(item, dict):
            text = item.get(self.text_key, '')
        else:
            text = str(item)
        
        features = {}
        
        # Extract action words (verbs and adverbs), excluding stopwords
        words = text.lower().split()
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^a-z]', '', word)
            
            # Skip short words, stopwords, and proper nouns (likely agent names)
            if len(word_clean) < 4:
                continue
            if word_clean in self.STOPWORDS:
                continue
            
            # Focus on verbs (position 1-3 in sentence, or verb-like endings)
            is_likely_verb = False
            
            # Second word is often the main verb
            if i == 1:
                is_likely_verb = True
            # Words ending in verb suffixes
            elif any(word_clean.endswith(s) for s in ['ates', 'izes', 'ifies', 'ects', 'ucts']):
                is_likely_verb = True
            elif word_clean.endswith('ing') and len(word_clean) > 5:
                is_likely_verb = True
            elif word_clean.endswith('ed') and len(word_clean) > 4:
                is_likely_verb = True
            elif word_clean.endswith('ly') and len(word_clean) > 4:
                is_likely_verb = True  # Adverbs are useful too
            elif word_clean.endswith('es') and len(word_clean) > 4:
                is_likely_verb = True
            elif word_clean.endswith('s') and len(word_clean) > 5 and not word_clean.endswith('ness'):
                is_likely_verb = True
            
            if is_likely_verb:
                features[word_clean] = features.get(word_clean, 0) + 1.0
        
        return features
    
    def get_item_id(self, item: Any) -> str:
        """Get the agent ID from a frame."""
        if isinstance(item, dict):
            agent = item.get(self.agent_key, '')
            return agent.lower() if agent else ''
        return ''
    
    def get_item_content(self, item: Any) -> Any:
        """Store the text content."""
        if isinstance(item, dict):
            return item.get(self.text_key, '')
        return str(item)
    
    def _get_feature_label(self, feature: str) -> str:
        """Get semantic label for a feature."""
        # Direct lookup
        if feature in FEATURE_LABELS:
            return FEATURE_LABELS[feature]
        
        # Try without common suffixes
        for suffix in ['ed', 'ing', 'es', 's', 'ly']:
            if feature.endswith(suffix) and len(feature) > len(suffix) + 2:
                base = feature[:-len(suffix)]
                if base in FEATURE_LABELS:
                    return FEATURE_LABELS[base]
                # Check with 'e' added back
                if base + 'e' in FEATURE_LABELS:
                    return FEATURE_LABELS[base + 'e']
        
        return feature  # Return as-is if no label found
    
    def get_dimension_labels(self, dim_name: str) -> Tuple[str, str]:
        """Get semantic labels for a dimension's poles."""
        for dim in self.dimensions:
            if dim.name == dim_name:
                neg_labels = [self._get_feature_label(f) for f in dim.negative_features]
                pos_labels = [self._get_feature_label(f) for f in dim.positive_features]
                
                # Use the first meaningful label
                neg_label = next((l for l in neg_labels if l in FEATURE_LABELS.values()), neg_labels[0] if neg_labels else 'neutral')
                pos_label = next((l for l in pos_labels if l in FEATURE_LABELS.values()), pos_labels[0] if pos_labels else 'neutral')
                
                return (neg_label, pos_label)
        
        return ('neutral', 'neutral')
    
    def describe_traits(self, concept: str) -> List[str]:
        """Get semantic trait descriptions for a concept."""
        pos = self.get_position(concept)
        if pos is None:
            return []
        
        traits = []
        for i, dim in enumerate(self.dimensions[:5]):  # Top 5 dimensions
            if i < len(pos) and abs(pos[i]) > 0.15:
                neg_label, pos_label = self.get_dimension_labels(dim.name)
                trait = pos_label if pos[i] > 0 else neg_label
                if trait not in traits and trait != 'neutral':
                    traits.append(trait)
        
        return traits[:3]  # Top 3 traits
    
    def analyze(self, concept: str) -> Dict[str, Any]:
        """
        Analyze a concept and return semantic information.
        
        Returns:
            Dictionary with similar concepts, opposite, dimensional info, and traits
        """
        group = self.find_group(concept)
        
        result = {
            'concept': concept,
            'found': group is not None,
            'similar': [],
            'opposite': None,
            'dimensions': {},
            'traits': [],
        }
        
        if not group:
            return result
        
        # Similar concepts
        similar = self.find_similar(group, k=5)
        result['similar'] = [(s, round(d, 3)) for s, d in similar]
        
        # Opposite
        opposite = self.find_opposite(group)
        if opposite:
            result['opposite'] = (opposite[0], round(opposite[1], 3))
        
        # Semantic traits (using feature labels)
        result['traits'] = self.describe_traits(group)
        
        # Dimensional analysis with semantic labels
        pos = self.get_position(group)
        if pos is not None:
            for i, dim in enumerate(self.dimensions):
                if i < len(pos):
                    p = pos[i]
                    if abs(p) > 0.15:
                        neg_label, pos_label = self.get_dimension_labels(dim.name)
                        label = pos_label if p > 0 else neg_label
                        result['dimensions'][dim.name] = {
                            'value': round(float(p), 3),
                            'label': label,
                            'pole': dim.positive_pole if p > 0 else dim.negative_pole,
                            'direction': 'positive' if p > 0 else 'negative',
                        }
        
        return result
    
    def get_relevant_content(self, concepts: list, k: int = 5) -> list:
        """Get content relevant to a list of concepts."""
        items = self.find_items_for_groups(concepts, k)
        return [item.content for item in items]
