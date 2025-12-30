#!/usr/bin/env python3
"""
Two-Beam Interference Polisher

A simpler, more direct approach to geometric polishing:

1. TRUTH BEAM: The raw corpus - provides CONTENT (what to say)
2. SIGNAL BEAM: Qwen2-polished sentences - provides TEMPLATES (how to say it)

The interference works by:
- Extracting sentence TEMPLATES from the signal beam
- Filling templates with CONTENT from the truth beam
- Templates that match truth content well = constructive interference
- Templates that don't fit = destructive interference

This is essentially learning "how Qwen2 phrases things" and applying
that phrasing to our geometric content.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA


@dataclass
class Template:
    """A sentence template extracted from signal beam."""
    pattern: str  # e.g., "{entity} is a {role} that {action} {target}."
    source: str   # Original polished sentence
    slots: List[str]  # ['entity', 'role', 'action', 'target']
    frequency: int = 1


class TwoBeamPolisher:
    """
    Polishes output using two-beam interference.
    
    The key insight: Qwen2's polished sentences contain STRUCTURAL PATTERNS
    that we can extract and reuse. The patterns are the "interference fringes"
    that emerge from comparing many polished sentences.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str = None):
        self.truth_path = truth_corpus_path
        self.signal_path = signal_corpus_path or truth_corpus_path.replace('.json', '_signal.json')
        
        # Load truth corpus (content source)
        self.truth_qa = GeometricQA()
        self.truth_qa.load_corpus(truth_corpus_path)
        self.knowledge = self.truth_qa.knowledge
        
        # Load signal frames (template source)
        self.signal_frames = []
        if os.path.exists(self.signal_path):
            with open(self.signal_path, 'r') as f:
                data = json.load(f)
                self.signal_frames = data.get('frames', [])
        
        # Extract templates from signal beam
        self.templates = self._extract_templates()
        
        # Build concept info cache
        self.concept_cache = {}
    
    def _extract_templates(self) -> List[Template]:
        """
        Extract reusable templates from signal corpus.
        
        Look for patterns like:
        - "{X} is a {role} that {verb} {Y}."
        - "As a {role}, {X} {verb} various aspects of {Y}."
        """
        templates = []
        pattern_counts = Counter()
        
        # Common roles to detect
        roles = {'science', 'field', 'discipline', 'study', 'concept', 'process',
                 'phenomenon', 'detective', 'doctor', 'character', 'figure'}
        
        # Common structure words
        structure_words = {'is', 'a', 'an', 'the', 'that', 'which', 'who', 'as',
                         'of', 'to', 'and', 'or', 'in', 'on', 'for', 'with'}
        
        for frame in self.signal_frames:
            text = frame.get('text', '')
            agent = frame.get('agent', '').lower()
            
            if not text or not agent:
                continue
            
            # Try to extract pattern by replacing content words with slots
            pattern = text
            
            # Replace the agent/entity with {entity}
            pattern = re.sub(
                rf'\b{re.escape(agent)}\b',
                '{entity}',
                pattern,
                flags=re.IGNORECASE
            )
            pattern = re.sub(
                rf'\b{re.escape(agent.title())}\b',
                '{entity}',
                pattern,
            )
            
            # Replace roles with {role}
            for role in roles:
                pattern = re.sub(
                    rf'\ba {role}\b',
                    'a {role}',
                    pattern,
                    flags=re.IGNORECASE
                )
            
            # Count this pattern
            pattern_counts[pattern] += 1
        
        # Keep patterns that appear multiple times (constructive interference!)
        for pattern, count in pattern_counts.most_common(50):
            if count >= 1 and '{entity}' in pattern:
                # Identify slots
                slots = re.findall(r'\{(\w+)\}', pattern)
                templates.append(Template(
                    pattern=pattern,
                    source=pattern,
                    slots=slots,
                    frequency=count,
                ))
        
        return templates
    
    def _get_concept_info(self, name: str) -> Dict:
        """Get structured info about a concept."""
        if name in self.concept_cache:
            return self.concept_cache[name]
        
        if name not in self.knowledge.concepts:
            return None
        
        c = self.knowledge.concepts[name]
        
        # Good verbs
        good_verbs = {
            'studies', 'examines', 'investigates', 'explores', 'analyzes',
            'describes', 'explains', 'discovers', 'observes', 'measures',
            'solves', 'deduces', 'helps', 'supports', 'provides',
            'creates', 'develops', 'transforms', 'changes', 'adapts',
            'calculates', 'proves', 'demonstrates', 'governs', 'shapes',
        }
        
        # Get role
        role = "concept"
        role_words = {'detective', 'doctor', 'scientist', 'science', 'field',
                     'discipline', 'study', 'process', 'phenomenon', 'character'}
        if c.targets:
            for target, count in c.targets.most_common(10):
                if target in role_words and count >= 2:
                    role = target
                    break
        
        # Get actions
        actions = []
        if c.actions:
            for action, _ in c.actions.most_common(10):
                if action.lower() in good_verbs:
                    actions.append(action)
        
        # Get targets
        targets = []
        if c.targets:
            for target, _ in c.targets.most_common(10):
                if target in self.knowledge.concepts and len(target) > 3:
                    tc = self.knowledge.concepts[target]
                    if tc.is_content_word:
                        targets.append(target)
        
        info = {
            'name': name,
            'role': role,
            'actions': actions[:3],
            'targets': targets[:3],
        }
        
        self.concept_cache[name] = info
        return info
    
    def polish(self, concept: str) -> str:
        """
        Generate polished output for a concept.
        
        TRUE INTERFERENCE: Look for the SAME concept in signal beam.
        If found, use the signal phrasing with truth content.
        If not found, use learned patterns from similar concepts.
        """
        concept_lower = concept.lower()
        info = self._get_concept_info(concept_lower)
        
        if not info:
            return f"I don't have information about {concept}."
        
        # FIRST: Check if we have a direct signal for this concept
        for frame in self.signal_frames:
            if frame.get('agent', '').lower() == concept_lower:
                # Direct hit! Use the polished version
                return frame.get('text', '')
        
        # SECOND: Find signal from concept with same ROLE
        role = info['role']
        for frame in self.signal_frames:
            agent = frame.get('agent', '').lower()
            agent_info = self._get_concept_info(agent)
            if agent_info and agent_info['role'] == role:
                # Same role - adapt the template
                text = frame.get('text', '')
                # Replace the agent name with our concept
                adapted = re.sub(
                    rf'\b{re.escape(agent)}\b',
                    concept.title(),
                    text,
                    flags=re.IGNORECASE
                )
                adapted = re.sub(
                    rf'\b{re.escape(agent.title())}\b',
                    concept.title(),
                    adapted,
                )
                if adapted != text:  # Actually made a substitution
                    return adapted
        
        # THIRD: Fallback to simple generation with learned structure
        return self._simple_generate(info)
    
    def _template_score(self, template: Template, info: Dict) -> float:
        """Score how well a template fits the concept info."""
        score = template.frequency  # Base score from frequency
        
        # Bonus if template has slots we can fill
        if '{entity}' in template.pattern:
            score += 1
        if '{role}' in template.pattern and info['role']:
            score += 1
        if info['actions'] and any(a in template.pattern.lower() for a in info['actions']):
            score += 2
        
        return score
    
    def _fill_template(self, template: Template, info: Dict) -> str:
        """Fill a template with concept info."""
        result = template.pattern
        
        # Fill entity
        result = result.replace('{entity}', info['name'].title())
        
        # Fill role
        result = result.replace('{role}', info['role'])
        
        # Try to fill any remaining action/target slots contextually
        if info['actions']:
            for action in info['actions']:
                if action.lower() in result.lower():
                    break
            else:
                # No action in template, might need to add
                pass
        
        return result
    
    def _simple_generate(self, info: Dict) -> str:
        """Simple fallback generation."""
        name = info['name'].title()
        role = info['role']
        actions = info['actions']
        targets = info['targets']
        
        if actions:
            action_str = ', '.join(actions[:2])
            if len(actions) > 2:
                action_str += f', and {actions[2]}'
            
            if targets:
                return f"{name} is a {role} that {action_str}, particularly in relation to {targets[0]}."
            else:
                return f"{name} is a {role} that {action_str}."
        else:
            return f"{name} is a {role}."
    
    def compare_outputs(self, concept: str) -> Dict[str, str]:
        """Compare raw, polished, and template-based outputs."""
        # Raw from truth corpus
        self.truth_qa.set_output_lens('natural')
        raw = self.truth_qa.ask(f"What is {concept}?")
        
        # Template-based (our interference)
        polished = self.polish(concept)
        
        return {
            'raw': raw,
            'interference': polished,
        }


def create_signal_corpus(truth_path: str, output_path: str, num_concepts: int = 100):
    """Create signal corpus by polishing with Qwen2."""
    from experiments.ollama_corpus_refiner import OllamaClient
    
    print(f"Creating signal corpus from {truth_path}")
    print(f"Output: {output_path}")
    
    qa = GeometricQA()
    qa.load_corpus(truth_path)
    qa.set_output_lens('natural')
    
    ollama = OllamaClient()
    if not ollama.is_available():
        print("ERROR: Ollama not available!")
        return
    
    # Get concepts
    concepts = []
    for name, concept in qa.knowledge.concepts.items():
        if concept.is_content_word and concept.actions:
            concepts.append(name)
    concepts = concepts[:num_concepts]
    
    print(f"Polishing {len(concepts)} concepts...")
    
    signal_frames = []
    for i, concept in enumerate(concepts):
        raw = qa.ask(f"What is {concept}?")
        if "don't know" in raw.lower():
            continue
        
        prompt = f'Rewrite this to be natural and grammatically correct. Only output the rewritten sentence:\n\n"{raw}"\n\nRewritten:'
        polished = ollama.generate(prompt, temperature=0.3)
        
        if polished and len(polished) > 10:
            signal_frames.append({
                'text': polished.strip().strip('"'),
                'source': 'signal',
                'agent': concept,
            })
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(concepts)}")
    
    with open(output_path, 'w') as f:
        json.dump({'frames': signal_frames}, f, indent=2)
    
    print(f"Saved {len(signal_frames)} signal frames")


def demo():
    """Demo the two-beam polisher."""
    print("=" * 70)
    print("TWO-BEAM INTERFERENCE POLISHER")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal.json"
    
    if not os.path.exists(signal_path):
        print("\nSignal corpus not found. Creating...")
        create_signal_corpus(truth_path, signal_path, num_concepts=100)
    
    polisher = TwoBeamPolisher(truth_path, signal_path)
    
    print(f"\nExtracted {len(polisher.templates)} templates from signal beam")
    print("\nTop templates (by frequency):")
    for t in polisher.templates[:5]:
        print(f"  [{t.frequency}x] {t.pattern[:70]}...")
    
    print("\n" + "=" * 70)
    print("COMPARISON: Raw vs Interference-Polished")
    print("=" * 70)
    
    for concept in ['physics', 'evolution', 'consciousness', 'holmes', 'biology']:
        results = polisher.compare_outputs(concept)
        print(f"\n{concept.upper()}:")
        print(f"  RAW:          {results['raw']}")
        print(f"  INTERFERENCE: {results['interference']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-signal", action="store_true")
    parser.add_argument("--concepts", type=int, default=100)
    parser.add_argument("--demo", action="store_true")
    
    args = parser.parse_args()
    
    if args.create_signal:
        create_signal_corpus(
            "truthspace_lcm/corpus_experimental.json",
            "truthspace_lcm/corpus_signal.json",
            args.concepts
        )
    else:
        demo()
