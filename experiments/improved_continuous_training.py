#!/usr/bin/env python3
"""
Improved Continuous Training

Addresses three key issues:
1. Generate diverse behavioral data so dimensions separate on BEHAVIOR, not identity
2. Add labeling pass that infers semantic meaning from dimension features
3. Use positive/negative features for descriptions instead of pole names

The goal: natural-sounding outputs while building a quality corpus.
"""

import json
import numpy as np
import requests
import time
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.gears.core import SemanticChain, LinguisticChain, DimensionInfo


OLLAMA_URL = "http://localhost:11434/api/generate"


# Behavioral verb categories for diverse data generation
BEHAVIOR_CATEGORIES = {
    'investigative': ['investigates', 'examines', 'analyzes', 'scrutinizes', 'deduces', 'observes'],
    'commanding': ['commands', 'orders', 'directs', 'leads', 'decrees', 'rules'],
    'supportive': ['assists', 'helps', 'supports', 'aids', 'accompanies', 'serves'],
    'scheming': ['plots', 'schemes', 'manipulates', 'deceives', 'conspires', 'betrays'],
    'heroic': ['rescues', 'protects', 'defends', 'saves', 'fights', 'confronts'],
    'wise': ['advises', 'teaches', 'guides', 'counsels', 'reflects', 'contemplates'],
    'playful': ['plays', 'explores', 'imagines', 'wonders', 'discovers', 'experiments'],
    'destructive': ['destroys', 'rages', 'overwhelms', 'consumes', 'threatens', 'devastates'],
    'nurturing': ['nurtures', 'cares', 'heals', 'comforts', 'tends', 'protects'],
    'creative': ['creates', 'builds', 'crafts', 'designs', 'invents', 'composes'],
}

# Feature to semantic label mapping - comprehensive verb/adverb categorization
FEATURE_LABELS = {
    # Investigative/Analytical
    'investigates': 'investigative', 'examines': 'investigative', 'analyzes': 'analytical',
    'deduces': 'deductive', 'scrutinizes': 'analytical', 'observes': 'observant',
    'studies': 'studious', 'researches': 'investigative', 'inspects': 'analytical',
    'discovers': 'exploratory', 'uncovers': 'investigative', 'detects': 'perceptive',
    'reads': 'studious', 'deciphers': 'analytical', 'interprets': 'interpretive',
    
    # Authoritative/Commanding
    'commands': 'authoritative', 'orders': 'commanding', 'leads': 'leadership',
    'directs': 'directive', 'rules': 'authoritative', 'governs': 'governing',
    'decrees': 'authoritative', 'dictates': 'commanding', 'controls': 'controlling',
    'manages': 'managerial', 'oversees': 'supervisory', 'dominates': 'dominant',
    
    # Supportive/Helpful
    'assists': 'supportive', 'helps': 'helpful', 'supports': 'supportive',
    'aids': 'helpful', 'serves': 'servile', 'accompanies': 'companionable',
    'follows': 'loyal', 'attends': 'attentive', 'cares': 'caring',
    'listens': 'attentive', 'comforts': 'comforting', 'nurtures': 'nurturing',
    
    # Scheming/Cunning
    'plots': 'scheming', 'schemes': 'cunning', 'manipulates': 'manipulative',
    'deceives': 'deceptive', 'conspires': 'conspiratorial', 'betrays': 'treacherous',
    'tricks': 'cunning', 'lies': 'deceptive', 'cheats': 'dishonest',
    'whispers': 'secretive', 'lurks': 'stealthy', 'hides': 'secretive',
    
    # Heroic/Brave
    'rescues': 'heroic', 'protects': 'protective', 'defends': 'defensive',
    'saves': 'heroic', 'fights': 'combative', 'confronts': 'confrontational',
    'battles': 'combative', 'challenges': 'challenging', 'stands': 'steadfast',
    'shields': 'protective', 'guards': 'protective', 'champions': 'heroic',
    
    # Wise/Teaching
    'advises': 'wise', 'teaches': 'educational', 'guides': 'guiding',
    'counsels': 'wise', 'mentors': 'mentoring', 'instructs': 'instructive',
    'enlightens': 'enlightening', 'explains': 'explanatory', 'shares': 'generous',
    'reflects': 'reflective', 'contemplates': 'contemplative', 'meditates': 'meditative',
    
    # Playful/Creative
    'plays': 'playful', 'explores': 'exploratory', 'imagines': 'imaginative',
    'creates': 'creative', 'builds': 'constructive', 'invents': 'inventive',
    'dreams': 'dreamy', 'wonders': 'curious', 'experiments': 'experimental',
    'crafts': 'skilled', 'designs': 'creative', 'composes': 'artistic',
    
    # Destructive/Violent
    'destroys': 'destructive', 'rages': 'violent', 'attacks': 'aggressive',
    'strikes': 'aggressive', 'crushes': 'destructive', 'burns': 'destructive',
    'engulfs': 'overwhelming', 'consumes': 'consuming', 'devastates': 'devastating',
    'threatens': 'threatening', 'intimidates': 'intimidating', 'overwhelms': 'overwhelming',
    
    # Communication
    'speaks': 'communicative', 'talks': 'talkative', 'announces': 'declarative',
    'declares': 'declarative', 'proclaims': 'proclamatory', 'argues': 'argumentative',
    'debates': 'debating', 'discusses': 'discussive', 'negotiates': 'diplomatic',
    'persuades': 'persuasive', 'convinces': 'convincing', 'addresses': 'formal',
    
    # Movement/Action
    'walks': 'mobile', 'runs': 'swift', 'travels': 'traveling',
    'journeys': 'adventurous', 'ventures': 'adventurous', 'wanders': 'wandering',
    'gathers': 'collecting', 'collects': 'collecting', 'searches': 'searching',
    'seeks': 'seeking', 'hunts': 'hunting', 'pursues': 'pursuing',
    
    # Adverbs
    'carefully': 'careful', 'quickly': 'swift', 'methodically': 'methodical',
    'bravely': 'brave', 'cunningly': 'cunning', 'wisely': 'wise',
    'playfully': 'playful', 'fiercely': 'fierce', 'gently': 'gentle',
    'skillfully': 'skilled', 'meticulously': 'meticulous', 'intently': 'focused',
    'silently': 'silent', 'loudly': 'loud', 'patiently': 'patient',
    'eagerly': 'eager', 'reluctantly': 'reluctant', 'boldly': 'bold',
    'cautiously': 'cautious', 'swiftly': 'swift', 'deliberately': 'deliberate',
    'attentively': 'attentive', 'diligently': 'diligent', 'passionately': 'passionate',
}


@dataclass
class SemanticLabel:
    """A semantic label for a dimension."""
    name: str
    negative_label: str
    positive_label: str
    confidence: float
    source_features: List[str]


class ImprovedSemanticChain(SemanticChain):
    """
    Extended SemanticChain with semantic labeling.
    """
    
    def __init__(self, name: str = "ImprovedSemanticChain"):
        super().__init__(name)
        self.semantic_labels: Dict[str, SemanticLabel] = {}
    
    def learn_dimensions(self, min_variance: float = 0.02, max_dims: int = 15) -> int:
        """Learn dimensions and then infer semantic labels."""
        count = super().learn_dimensions(min_variance, max_dims)
        self._infer_semantic_labels()
        return count
    
    def _infer_semantic_labels(self):
        """Infer semantic labels from dimension features."""
        self.semantic_labels = {}
        
        for dim in self.dimensions:
            neg_label = self._features_to_label(dim.negative_features)
            pos_label = self._features_to_label(dim.positive_features)
            
            # Calculate confidence based on how many features we could label
            labeled_count = sum(1 for f in dim.negative_features + dim.positive_features 
                              if self._get_feature_label(f) != f)
            confidence = labeled_count / max(len(dim.negative_features) + len(dim.positive_features), 1)
            
            self.semantic_labels[dim.name] = SemanticLabel(
                name=dim.name,
                negative_label=neg_label,
                positive_label=pos_label,
                confidence=confidence,
                source_features=dim.negative_features + dim.positive_features,
            )
    
    def _get_feature_label(self, feature: str) -> str:
        """Get semantic label for a single feature."""
        # Direct lookup
        if feature in FEATURE_LABELS:
            return FEATURE_LABELS[feature]
        
        # Try without common suffixes
        for suffix in ['ed', 'ing', 'es', 's', 'ly']:
            if feature.endswith(suffix):
                base = feature[:-len(suffix)]
                if base in FEATURE_LABELS:
                    return FEATURE_LABELS[base]
                # Check if base + common ending exists
                for check_suffix in ['e', '']:
                    check = base + check_suffix
                    if check in FEATURE_LABELS:
                        return FEATURE_LABELS[check]
        
        # Return cleaned feature as fallback
        return feature.replace('_', ' ')
    
    def _features_to_label(self, features: List[str]) -> str:
        """Convert a list of features to a semantic label."""
        labels = []
        for f in features:
            label = self._get_feature_label(f)
            if label and label not in labels:
                labels.append(label)
        
        if labels:
            return labels[0]  # Use the first (most significant) label
        return "neutral"
    
    def get_dimension_description(self, dim_name: str) -> Tuple[str, str]:
        """Get human-readable description for a dimension."""
        if dim_name in self.semantic_labels:
            label = self.semantic_labels[dim_name]
            return (label.negative_label, label.positive_label)
        
        # Fallback to pole names
        for dim in self.dimensions:
            if dim.name == dim_name:
                return (dim.negative_pole, dim.positive_pole)
        
        return ("unknown", "unknown")
    
    def describe_position(self, group_id: str) -> List[str]:
        """Describe a group's position using semantic labels."""
        pos = self.get_position(group_id)
        if pos is None:
            return []
        
        descriptions = []
        for i, dim in enumerate(self.dimensions):
            if i < len(pos) and abs(pos[i]) > 0.15:
                neg_label, pos_label = self.get_dimension_description(dim.name)
                if pos[i] > 0:
                    descriptions.append(pos_label)
                else:
                    descriptions.append(neg_label)
        
        return descriptions[:3]  # Top 3 traits


class ImprovedContinuousTrainer:
    """
    Improved continuous trainer with:
    1. Diverse behavioral data generation
    2. Semantic labeling
    3. Feature-based descriptions
    """
    
    def __init__(self, model: str = "qwen2:latest"):
        self.semantic = ImprovedSemanticChain("Understanding")
        self.linguistic = LinguisticChain("Output")
        self.model = model
        
        self.cycles_completed = 0
        self.total_items_generated = 0
        self.benchmark_history = []
        self.best_score = 0.0
        
        # Ground truth for benchmarking
        self.ground_truth = {
            'similar_pairs': [
                ('holmes', 'watson'),
                ('villain', 'moriarty'),
                ('hero', 'brave'),
                ('king', 'queen'),
                ('sage', 'wisdom'),
            ],
            'behavioral_clusters': {
                'investigative': ['holmes', 'watson', 'spy', 'detective_work'],
                'authoritative': ['king', 'queen', 'general', 'leader'],
                'villainous': ['villain', 'moriarty', 'politician'],
                'heroic': ['hero', 'brave', 'soldier'],
                'wise': ['sage', 'wisdom', 'scholar', 'elder'],
            }
        }
    
    def _call_llm(self, prompt: str, max_tokens: int = 400) -> str:
        """Call LLM for data generation."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.8}
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"LLM error: {e}")
            return ""
    
    def generate_diverse_behavioral_data(self, concept: str, n_sentences: int = 10) -> List[Dict]:
        """
        Generate diverse behavioral data for a concept.
        
        Key: Use DIFFERENT behavior categories to create variety,
        so dimensions emerge from behavior patterns, not identity.
        """
        frames = []
        concept_clean = concept.replace('_', ' ').title()
        
        # Determine which behavior categories fit this concept
        concept_behaviors = self._get_concept_behaviors(concept)
        
        prompt = f"""Generate {n_sentences} sentences showing "{concept_clean}" performing various actions.

IMPORTANT: Use diverse verbs from these categories:
{', '.join(concept_behaviors)}

Rules:
1. EVERY sentence MUST start with "{concept_clean}"
2. Use DIFFERENT verbs - no repeating the same action
3. Show the character's typical behaviors
4. Format: "{concept_clean} [verb] [rest of sentence]"
5. Keep sentences 8-15 words

Example verbs to use: {', '.join(self._get_example_verbs(concept_behaviors))}

Generate {n_sentences} diverse sentences, one per line:"""

        response = self._call_llm(prompt)
        
        if response:
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            for line in lines[:n_sentences]:
                line = line.lstrip('0123456789.-) ')
                if line.lower().startswith(concept_clean.lower()) and len(line) > 15:
                    frames.append({
                        'text': line,
                        'agent': concept.lower().replace(' ', '_'),
                        'source': 'diverse_behavioral',
                    })
        
        return frames
    
    def _get_concept_behaviors(self, concept: str) -> List[str]:
        """Get appropriate behavior categories for a concept."""
        concept_lower = concept.lower()
        
        behavior_map = {
            'holmes': ['investigative', 'wise'],
            'watson': ['supportive', 'investigative'],
            'moriarty': ['scheming', 'commanding'],
            'villain': ['scheming', 'destructive'],
            'hero': ['heroic', 'commanding'],
            'king': ['commanding', 'wise'],
            'queen': ['commanding', 'wise'],
            'sage': ['wise', 'nurturing'],
            'child': ['playful', 'creative'],
            'servant': ['supportive', 'nurturing'],
            'soldier': ['heroic', 'commanding'],
            'spy': ['scheming', 'investigative'],
            'healer': ['nurturing', 'supportive'],
            'craftsman': ['creative', 'wise'],
        }
        
        for key, behaviors in behavior_map.items():
            if key in concept_lower:
                return behaviors
        
        # Default behaviors
        return ['investigative', 'supportive', 'creative']
    
    def _get_example_verbs(self, categories: List[str]) -> List[str]:
        """Get example verbs from behavior categories."""
        verbs = []
        for cat in categories:
            if cat in BEHAVIOR_CATEGORIES:
                verbs.extend(BEHAVIOR_CATEGORIES[cat][:3])
        return verbs[:6]
    
    def generate_contrastive_data(self, n_pairs: int = 3) -> List[Dict]:
        """
        Generate data that shows behavioral CONTRASTS.
        This helps dimensions separate on behavior, not identity.
        """
        frames = []
        
        # Contrastive pairs (opposite behaviors)
        contrasts = [
            ('hero', 'villain', 'heroic', 'scheming'),
            ('sage', 'child', 'wise', 'playful'),
            ('king', 'servant', 'commanding', 'supportive'),
        ]
        
        for c1, c2, behavior1, behavior2 in contrasts[:n_pairs]:
            c1_clean = c1.replace('_', ' ').title()
            c2_clean = c2.replace('_', ' ').title()
            
            prompt = f"""Generate 2 contrasting sentence pairs.

Pair 1: Show {c1_clean} being {behavior1}
Pair 2: Show {c2_clean} being {behavior2}

Rules:
1. First sentence starts with "{c1_clean}"
2. Second sentence starts with "{c2_clean}"
3. Show OPPOSITE behaviors clearly
4. Keep sentences 8-12 words

Generate 4 sentences (2 pairs):"""

            response = self._call_llm(prompt)
            if response:
                lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
                for line in lines[:4]:
                    line = line.lstrip('0123456789.-) ')
                    if len(line) > 15:
                        # Determine agent from sentence start
                        if line.lower().startswith(c1_clean.lower()):
                            agent = c1.lower()
                        elif line.lower().startswith(c2_clean.lower()):
                            agent = c2.lower()
                        else:
                            continue
                        
                        frames.append({
                            'text': line,
                            'agent': agent,
                            'source': 'contrastive',
                        })
            
            time.sleep(0.3)
        
        return frames
    
    def benchmark(self) -> Dict[str, float]:
        """Benchmark with semantic label quality included."""
        metrics = {}
        
        # 1. Similar pair accuracy
        similar_correct = 0
        similar_total = 0
        for c1, c2 in self.ground_truth['similar_pairs']:
            if self.semantic.find_group(c1) and self.semantic.find_group(c2):
                similar = self.semantic.find_similar(c1, k=5)
                similar_names = [s[0] for s in similar]
                if any(c2 in s or s in c2 for s in similar_names):
                    similar_correct += 1
                similar_total += 1
        metrics['similar_accuracy'] = similar_correct / max(similar_total, 1)
        
        # 2. Behavioral cluster coherence
        cluster_scores = []
        for behavior, members in self.ground_truth['behavioral_clusters'].items():
            known = [m for m in members if self.semantic.find_group(m)]
            if len(known) >= 2:
                positions = [self.semantic.get_position(m) for m in known]
                positions = [p for p in positions if p is not None]
                if len(positions) >= 2:
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            distances.append(np.linalg.norm(positions[i] - positions[j]))
                    coherence = 1.0 / (1.0 + np.mean(distances))
                    cluster_scores.append(coherence)
        metrics['cluster_coherence'] = np.mean(cluster_scores) if cluster_scores else 0.0
        
        # 3. Semantic label quality (how many dimensions have good labels)
        labeled_dims = 0
        for label in self.semantic.semantic_labels.values():
            if label.confidence > 0.3 and label.negative_label != label.positive_label:
                labeled_dims += 1
        metrics['label_quality'] = labeled_dims / max(len(self.semantic.dimensions), 1)
        
        # 4. Dimension quality
        total_variance = sum(d.variance for d in self.semantic.dimensions)
        metrics['dimension_quality'] = min(total_variance, 1.0)
        
        # 5. Coverage
        all_concepts = set()
        for members in self.ground_truth['behavioral_clusters'].values():
            all_concepts.update(members)
        known = sum(1 for c in all_concepts if self.semantic.find_group(c))
        metrics['coverage'] = known / len(all_concepts)
        
        # Composite
        metrics['composite'] = (
            metrics['similar_accuracy'] * 0.25 +
            metrics['cluster_coherence'] * 0.25 +
            metrics['label_quality'] * 0.20 +
            metrics['dimension_quality'] * 0.15 +
            metrics['coverage'] * 0.15
        )
        
        return metrics
    
    def train_cycle(self, generate_data: bool = True) -> Dict[str, float]:
        """Run one training cycle with improved data generation."""
        self.cycles_completed += 1
        print(f"\n{'='*60}")
        print(f"TRAINING CYCLE {self.cycles_completed}")
        print(f"{'='*60}")
        
        pre_metrics = self.benchmark()
        print(f"\nPre-training metrics:")
        for k, v in pre_metrics.items():
            print(f"  {k}: {v:.3f}")
        
        if generate_data:
            total_new = 0
            
            # Generate diverse behavioral data for key concepts
            key_concepts = ['holmes', 'watson', 'villain', 'hero', 'sage', 'king']
            for concept in key_concepts:
                if self.semantic.find_group(concept):
                    count = self.semantic.group_counts.get(concept, 0)
                    if count < 20:  # Need more data
                        print(f"  Generating diverse data for {concept}...")
                        frames = self.generate_diverse_behavioral_data(concept, n_sentences=5)
                        for f in frames:
                            self.semantic.ingest_item(f)
                            self.linguistic.ingest_item(f)
                        total_new += len(frames)
            
            # Generate contrastive data
            print(f"  Generating contrastive data...")
            contrast_frames = self.generate_contrastive_data(n_pairs=2)
            for f in contrast_frames:
                self.semantic.ingest_item(f)
                self.linguistic.ingest_item(f)
            total_new += len(contrast_frames)
            
            print(f"  Total new frames: {total_new}")
            self.total_items_generated += total_new
        
        # Retrain
        print(f"\nRetraining dimensions...")
        semantic_dims = self.semantic.learn_dimensions()
        linguistic_dims = self.linguistic.learn_dimensions()
        print(f"  Semantic: {semantic_dims} dimensions")
        print(f"  Linguistic: {linguistic_dims} dimensions")
        
        # Show semantic labels
        print(f"\nSemantic labels:")
        for dim in self.semantic.dimensions[:5]:
            neg, pos = self.semantic.get_dimension_description(dim.name)
            label = self.semantic.semantic_labels.get(dim.name)
            conf = label.confidence if label else 0
            print(f"  {dim.name}: {neg} ↔ {pos} (conf: {conf:.2f})")
        
        post_metrics = self.benchmark()
        print(f"\nPost-training metrics:")
        for k, v in post_metrics.items():
            print(f"  {k}: {v:.3f}")
        
        improvement = post_metrics['composite'] - pre_metrics['composite']
        print(f"\nImprovement: {improvement:+.3f}")
        
        if post_metrics['composite'] > self.best_score:
            self.best_score = post_metrics['composite']
            print("  ✓ New best score!")
        
        self.benchmark_history.append(post_metrics)
        
        return post_metrics
    
    def test_output_quality(self):
        """Test the quality of outputs using semantic labels."""
        print(f"\n{'='*60}")
        print("OUTPUT QUALITY TEST")
        print(f"{'='*60}")
        
        test_concepts = ['holmes', 'watson', 'villain', 'hero', 'sage']
        
        for concept in test_concepts:
            if not self.semantic.find_group(concept):
                continue
            
            name = concept.replace('_', ' ').title()
            
            # Get semantic descriptions
            traits = self.semantic.describe_position(concept)
            similar = self.semantic.find_similar(concept, k=3)
            opposite = self.semantic.find_opposite(concept)
            
            print(f"\n{name}:")
            
            if traits:
                print(f"  Traits: {', '.join(traits)}")
            
            if similar:
                similar_names = [s[0].replace('_', ' ').title() for s in similar]
                print(f"  Similar to: {', '.join(similar_names)}")
            
            if opposite:
                print(f"  Opposite: {opposite[0].replace('_', ' ').title()}")
            
            # Get sample content
            content = self.semantic.get_relevant_content([concept], k=2)
            if content:
                print(f"  Sample: {content[0][:60]}...")
    
    def run(self, max_cycles: int = 5, patience: int = 3) -> List[Dict]:
        """Run the improved training loop."""
        print("=" * 70)
        print("IMPROVED CONTINUOUS TRAINING")
        print("=" * 70)
        
        results = []
        cycles_without_improvement = 0
        
        for _ in range(max_cycles):
            metrics = self.train_cycle(generate_data=True)
            results.append(metrics)
            
            if metrics['composite'] >= self.best_score:
                cycles_without_improvement = 0
            else:
                cycles_without_improvement += 1
            
            if cycles_without_improvement >= patience:
                print(f"\nConverged (no improvement for {patience} cycles)")
                break
            
            time.sleep(0.5)
        
        # Final output quality test
        self.test_output_quality()
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Cycles: {self.cycles_completed}")
        print(f"Items generated: {self.total_items_generated}")
        print(f"Final items: {len(self.semantic.items)}")
        print(f"Final dimensions: {len(self.semantic.dimensions)}")
        print(f"Best score: {self.best_score:.3f}")
        
        print("\nScore progression:")
        for i, r in enumerate(results):
            print(f"  Cycle {i+1}: {r['composite']:.3f}")
        
        return results


def main():
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is running")
    except:
        print("ERROR: Ollama not running")
        return None
    
    trainer = ImprovedContinuousTrainer()
    
    # Load initial corpus
    base = Path(__file__).parent.parent
    corpus_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if corpus_path.exists():
        print(f"Loading corpus: {corpus_path}")
        trainer.semantic.ingest_corpus(str(corpus_path))
        trainer.linguistic.ingest_corpus(str(corpus_path))
        print(f"  Loaded {len(trainer.semantic.items)} items")
    
    # Initial training
    trainer.semantic.learn_dimensions()
    trainer.linguistic.learn_dimensions()
    
    # Run improved training
    results = trainer.run(max_cycles=4, patience=2)
    
    # Save state
    state_path = Path(__file__).parent / "improved_training_state.json"
    with open(state_path, 'w') as f:
        json.dump({
            'cycles': trainer.cycles_completed,
            'items_generated': trainer.total_items_generated,
            'best_score': trainer.best_score,
            'history': trainer.benchmark_history,
        }, f, indent=2)
    print(f"\nState saved to: {state_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
