#!/usr/bin/env python3
"""
Unified Gear System

Combines emergent (data-discovered) gears with designed (theory-based) gears
to create a complete understanding system.

Emergent gears: Discover CHARACTER-LEVEL dimensions (agency, maturity, morality)
Designed gears: Handle SENTENCE-LEVEL transformations (tense, voice, mood)

Together they provide both semantic understanding and grammatical control.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

from self_discovering_gears import SelfDiscoveringGearSystem


class UnifiedGearSystem:
    """
    A unified system combining emergent and designed gears.
    
    This represents the synthesis: let data tell us what dimensions exist,
    but also apply known transformations from linguistic theory.
    """
    
    def __init__(self):
        self.emergent_system: Optional[SelfDiscoveringGearSystem] = None
        self.corpus_frames: List[Dict] = []
        
        # Designed gear configurations (theory-based)
        self.designed_gears = {
            'tense': {
                'name': 'TenseGear',
                'description': 'Handles temporal aspects',
                'transforms': {
                    'past': lambda v: self._past_tense(v),
                    'present': lambda v: v,
                    'future': lambda v: f"will {v}",
                }
            },
            'voice': {
                'name': 'VoiceGear', 
                'description': 'Handles active/passive',
                'transforms': {
                    'active': lambda s, a, v, o: f"{a} {v} {o}",
                    'passive': lambda s, a, v, o: f"{o} is {self._past_participle(v)} by {a}",
                }
            },
            'mood': {
                'name': 'MoodGear',
                'description': 'Handles modality',
                'transforms': {
                    'indicative': lambda v: v,
                    'subjunctive': lambda v: f"might {v}",
                    'imperative': lambda v: v,
                }
            },
            'polarity': {
                'name': 'PolarityGear',
                'description': 'Handles negation',
                'transforms': {
                    'positive': lambda v: v,
                    'negative': lambda v: f"does not {v}",
                }
            },
        }
    
    def _past_tense(self, verb: str) -> str:
        """Simple past tense conversion."""
        if verb.endswith('e'):
            return verb + 'd'
        elif verb.endswith('y') and len(verb) > 2 and verb[-2] not in 'aeiou':
            return verb[:-1] + 'ied'
        else:
            return verb + 'ed'
    
    def _past_participle(self, verb: str) -> str:
        """Simple past participle (same as past for regular verbs)."""
        return self._past_tense(verb)
    
    def load_emergent_system(self, corpus_path: str):
        """Load and initialize the emergent gear system."""
        print("Loading emergent gear system...")
        
        self.emergent_system = SelfDiscoveringGearSystem()
        self.emergent_system.ingest_corpus(corpus_path)
        self.emergent_system.discover_dimensions()
        self.emergent_system.build_gears()
        
        # Also load corpus for retrieval
        with open(corpus_path) as f:
            corpus = json.load(f)
        self.corpus_frames = corpus['frames']
        
        print(f"Loaded {len(self.emergent_system.gears)} emergent gears")
        print(f"Loaded {len(self.corpus_frames)} frames for retrieval")
    
    def analyze_concept(self, concept: str) -> Dict[str, Any]:
        """
        Analyze a concept using both emergent and designed perspectives.
        """
        result = {
            'concept': concept,
            'emergent_analysis': {},
            'designed_capabilities': [],
            'similar': [],
            'opposite': None,
            'relevant_frames': [],
        }
        
        # Emergent analysis
        if self.emergent_system:
            analysis = self.emergent_system.analyze(concept)
            result['emergent_analysis'] = analysis.get('dimensions', {})
            result['similar'] = analysis.get('similar', [])
            result['opposite'] = analysis.get('opposite')
        
        # Designed capabilities (what transformations are available)
        result['designed_capabilities'] = list(self.designed_gears.keys())
        
        # Find relevant frames
        result['relevant_frames'] = self._find_relevant_frames(concept, k=5)
        
        return result
    
    def _find_relevant_frames(self, concept: str, k: int = 5) -> List[str]:
        """Find frames relevant to a concept."""
        concept_lower = concept.lower()
        relevant = []
        
        for frame in self.corpus_frames:
            agent = frame.get('agent', '').lower()
            text = frame.get('text', '')
            
            if concept_lower in agent or concept_lower in text.lower():
                relevant.append(text)
        
        return relevant[:k]
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query using the unified system.
        
        Returns comprehensive analysis combining emergent understanding
        with designed transformation capabilities.
        """
        result = {
            'query': query_text,
            'concepts': [],
            'analyses': {},
            'relationships': [],
            'response': '',
        }
        
        # Extract concepts
        concepts = self._extract_concepts(query_text)
        result['concepts'] = concepts
        
        # Analyze each concept
        for concept in concepts:
            result['analyses'][concept] = self.analyze_concept(concept)
        
        # Find relationships between concepts
        if len(concepts) >= 2:
            result['relationships'] = self._find_relationships(concepts)
        
        # Generate response
        result['response'] = self._generate_response(result)
        
        return result
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract concepts from query."""
        if not self.emergent_system:
            return []
        
        query_lower = query.lower()
        found = []
        
        # Check for known agents
        for agent in self.emergent_system.agents:
            if agent in query_lower and len(agent) > 2:
                found.append(agent)
        
        return found
    
    def _find_relationships(self, concepts: List[str]) -> List[Dict]:
        """Find relationships between concepts."""
        relationships = []
        
        if not self.emergent_system or len(concepts) < 2:
            return relationships
        
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                # Get positions
                pos1 = np.array([
                    dim.positions.get(c1.lower(), 0) 
                    for dim in self.emergent_system.dimensions
                ])
                pos2 = np.array([
                    dim.positions.get(c2.lower(), 0) 
                    for dim in self.emergent_system.dimensions
                ])
                
                # Calculate distance and direction
                diff = pos2 - pos1
                distance = np.linalg.norm(diff)
                
                # Find most different dimension
                if len(diff) > 0:
                    max_diff_idx = np.argmax(np.abs(diff))
                    max_diff_dim = self.emergent_system.dimensions[max_diff_idx].name
                    max_diff_val = diff[max_diff_idx]
                else:
                    max_diff_dim = "unknown"
                    max_diff_val = 0
                
                relationships.append({
                    'pair': (c1, c2),
                    'distance': float(distance),
                    'most_different_dimension': max_diff_dim,
                    'difference': float(max_diff_val),
                })
        
        return relationships
    
    def _generate_response(self, result: Dict) -> str:
        """Generate a comprehensive response."""
        parts = []
        
        concepts = result['concepts']
        analyses = result['analyses']
        relationships = result['relationships']
        
        if not concepts:
            return "I couldn't identify any known concepts in your query."
        
        # Describe each concept
        for concept in concepts:
            analysis = analyses.get(concept, {})
            emergent = analysis.get('emergent_analysis', {})
            similar = analysis.get('similar', [])
            opposite = analysis.get('opposite')
            frames = analysis.get('relevant_frames', [])
            
            parts.append(f"\n**{concept.upper()}**")
            
            # Emergent dimensions
            if emergent:
                dim_parts = []
                for dim_name, info in emergent.items():
                    if info.get('class') != 'neutral':
                        dim_parts.append(f"{dim_name}: {info['class']} (toward {info['pole']})")
                if dim_parts:
                    parts.append("  Dimensions: " + ", ".join(dim_parts))
            
            # Similar/opposite
            if similar:
                parts.append(f"  Similar to: {', '.join([s[0] for s in similar[:3]])}")
            if opposite:
                parts.append(f"  Opposite of: {opposite[0]}")
            
            # Sample behaviors
            if frames:
                parts.append("  Sample behaviors:")
                for frame in frames[:2]:
                    parts.append(f"    • {frame[:60]}...")
        
        # Relationships
        if relationships:
            parts.append("\n**RELATIONSHIPS**")
            for rel in relationships:
                c1, c2 = rel['pair']
                dist = rel['distance']
                dim = rel['most_different_dimension']
                diff = rel['difference']
                
                if dist < 0.5:
                    similarity = "very similar"
                elif dist < 1.0:
                    similarity = "somewhat similar"
                else:
                    similarity = "quite different"
                
                parts.append(f"  {c1} vs {c2}: {similarity} (distance: {dist:.2f})")
                parts.append(f"    Most different on: {dim} (Δ={diff:+.2f})")
        
        return '\n'.join(parts)
    
    def transform(self, text: str, transformations: Dict[str, str]) -> str:
        """
        Apply designed gear transformations to text.
        
        Example:
            transform("Holmes investigates", {'tense': 'past'})
            → "Holmes investigated"
        """
        result = text
        
        for gear_name, setting in transformations.items():
            if gear_name in self.designed_gears:
                gear = self.designed_gears[gear_name]
                if setting in gear['transforms']:
                    # Simple verb transformation
                    words = result.split()
                    if len(words) >= 2:
                        verb = words[1]
                        new_verb = gear['transforms'][setting](verb)
                        words[1] = new_verb
                        result = ' '.join(words)
        
        return result


def interactive_session(system: UnifiedGearSystem):
    """Run an interactive query session."""
    print("\n" + "=" * 70)
    print("UNIFIED GEAR SYSTEM - Interactive Mode")
    print("=" * 70)
    print("\nThis system combines:")
    print("  • Emergent gears (discovered from data)")
    print("  • Designed gears (from linguistic theory)")
    print("\nCommands:")
    print("  <query>           - Ask about concepts")
    print("  compare <a> <b>   - Compare two concepts")
    print("  transform <text>  - Apply transformations")
    print("  gears             - Show all gears")
    print("  quit              - Exit")
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'gears':
            print("\n--- EMERGENT GEARS (discovered) ---")
            if system.emergent_system:
                for gear in system.emergent_system.gears:
                    print(f"  • {gear.name}: {gear.dimension.negative_pole} ↔ {gear.dimension.positive_pole}")
            print("\n--- DESIGNED GEARS (theory-based) ---")
            for name, gear in system.designed_gears.items():
                print(f"  • {gear['name']}: {gear['description']}")
            continue
        
        if user_input.lower().startswith('compare '):
            parts = user_input[8:].split()
            if len(parts) >= 2:
                c1, c2 = parts[0], parts[1]
                result = system.query(f"{c1} and {c2}")
                print(result['response'])
            continue
        
        if user_input.lower().startswith('transform '):
            text = user_input[10:]
            print(f"Original: {text}")
            print(f"Past tense: {system.transform(text, {'tense': 'past'})}")
            print(f"Negated: {system.transform(text, {'polarity': 'negative'})}")
            print(f"Subjunctive: {system.transform(text, {'mood': 'subjunctive'})}")
            continue
        
        # Default: treat as query
        result = system.query(user_input)
        print(result['response'])
        print()


def run_demo(system: UnifiedGearSystem):
    """Run a demonstration of the unified system."""
    print("\n" + "=" * 70)
    print("UNIFIED GEAR SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Demo queries
    queries = [
        "Tell me about Holmes",
        "Compare Holmes and Watson",
        "What is the difference between hero and villain?",
        "Tell me about the sage and the child",
    ]
    
    for query in queries:
        print(f"\n{'─'*70}")
        print(f"Query: {query}")
        print(f"{'─'*70}")
        result = system.query(query)
        print(result['response'])
    
    # Demo transformations
    print(f"\n{'─'*70}")
    print("TRANSFORMATION DEMO")
    print(f"{'─'*70}")
    
    sentences = [
        "Holmes investigate the crime",
        "Watson assist Holmes",
        "The villain scheme against the hero",
    ]
    
    for sentence in sentences:
        print(f"\nOriginal: {sentence}")
        print(f"  Past: {system.transform(sentence, {'tense': 'past'})}")
        print(f"  Negated: {system.transform(sentence, {'polarity': 'negative'})}")


def main():
    print("=" * 70)
    print("UNIFIED GEAR SYSTEM")
    print("=" * 70)
    print("\nCombining emergent (data-discovered) and designed (theory-based) gears")
    
    # Load corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run llm_live_corpus_generator.py first!")
        return None
    
    # Create unified system
    system = UnifiedGearSystem()
    system.load_emergent_system(str(corpus_path))
    
    # Run demo
    run_demo(system)
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Entering interactive mode...")
    interactive_session(system)
    
    return system


if __name__ == "__main__":
    system = main()
