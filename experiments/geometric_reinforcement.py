#!/usr/bin/env python3
"""
Geometric Reinforcement Learning: Reverse Projection

This experiment explores whether we can modify the corpus by going in reverse
through the output lens. The idea:

FORWARD PATH:
    Corpus → GeometricQA → Raw Output → Lens → Natural Text

REVERSE PATH:
    Correction → Inverse Lens → Corpus Modification

Key insight from Design 072: Transformations are bidirectional and self-similar.
If we can project forward, we should be able to project backward.

The goal is to create a feedback loop where:
1. System generates an answer
2. We provide a correction (or the lens provides one)
3. The correction propagates back to modify the corpus
4. Future answers improve

This is "geometric reinforcement learning" - learning through structural
modifications rather than gradient descent.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA, GeometricKnowledge, Frame


@dataclass
class Correction:
    """A correction to an answer."""
    query: str
    original_answer: str
    corrected_answer: str
    entity: str  # The entity being corrected
    correction_type: str  # 'role', 'action', 'target', 'description'


@dataclass
class CorpusModification:
    """A modification to apply to the corpus."""
    mod_type: str  # 'add_frame', 'remove_frame', 'adjust_role', 'add_action'
    entity: str
    details: Dict
    confidence: float


class ReverseProjector:
    """
    Projects corrections backward through the lens to corpus modifications.
    
    The key insight: The output lens transforms raw output into natural text.
    If we have a correction to the natural text, we can invert the lens
    to find what corpus changes would produce that correction.
    """
    
    def __init__(self, qa: GeometricQA):
        self.qa = qa
        self.knowledge = qa.knowledge
        
        # Inverse transforms (reverse of what the lens does)
        self.inverse_transforms = {
            # Natural text patterns → raw patterns
            "is someone who": "is a entity who",
            "is a character who": "is a protagonist who",
            "is a concept that": "is a concept who",
            "is a field that": "is a protagonist who",
            "is a process that": "is a protagonist who",
            "is a phenomenon that": "is a concept who",
            "is a linguistic property that": "is a concept who",
        }
        
        # Role keywords
        self.initiator_verbs = {'examines', 'deduces', 'investigates', 'solves', 'discovers'}
        self.mediator_verbs = {'assists', 'helps', 'supports', 'aids'}
        self.receiver_verbs = {'receives', 'undergoes', 'experiences'}
    
    def analyze_correction(self, correction: Correction) -> List[CorpusModification]:
        """
        Analyze a correction and determine what corpus modifications are needed.
        
        This is the core of reverse projection:
        1. Parse the original and corrected answers
        2. Identify what changed (role, actions, targets, description)
        3. Generate corpus modifications to produce the corrected answer
        """
        mods = []
        
        # Extract entity from correction
        entity = correction.entity.lower()
        
        # Parse both answers to extract structure
        original_struct = self._parse_answer(correction.original_answer)
        corrected_struct = self._parse_answer(correction.corrected_answer)
        
        # Compare and generate modifications
        
        # 1. Role changes
        if original_struct.get('role') != corrected_struct.get('role'):
            mods.append(CorpusModification(
                mod_type='adjust_role',
                entity=entity,
                details={
                    'old_role': original_struct.get('role'),
                    'new_role': corrected_struct.get('role'),
                },
                confidence=0.8,
            ))
        
        # 2. Action changes
        original_actions = set(original_struct.get('actions', []))
        corrected_actions = set(corrected_struct.get('actions', []))
        
        # Actions to add
        for action in corrected_actions - original_actions:
            mods.append(CorpusModification(
                mod_type='add_action',
                entity=entity,
                details={'action': action},
                confidence=0.9,
            ))
        
        # Actions to remove (lower confidence - might just need reweighting)
        for action in original_actions - corrected_actions:
            mods.append(CorpusModification(
                mod_type='reduce_action',
                entity=entity,
                details={'action': action},
                confidence=0.5,
            ))
        
        # 3. Target changes
        original_targets = set(original_struct.get('targets', []))
        corrected_targets = set(corrected_struct.get('targets', []))
        
        for target in corrected_targets - original_targets:
            mods.append(CorpusModification(
                mod_type='add_target',
                entity=entity,
                details={'target': target},
                confidence=0.9,
            ))
        
        return mods
    
    def _parse_answer(self, answer: str) -> Dict:
        """Parse an answer to extract its structure."""
        result = {
            'role': None,
            'actions': [],
            'targets': [],
        }
        
        answer_lower = answer.lower()
        
        # Extract role
        if 'detective' in answer_lower:
            result['role'] = 'detective'
        elif 'doctor' in answer_lower:
            result['role'] = 'doctor'
        elif 'science' in answer_lower or 'scientific' in answer_lower:
            result['role'] = 'science'
        elif 'field of study' in answer_lower or 'field that' in answer_lower:
            result['role'] = 'field'
        elif 'someone who' in answer_lower or 'character who' in answer_lower:
            result['role'] = 'protagonist'
        elif 'concept' in answer_lower or 'phenomenon' in answer_lower:
            result['role'] = 'concept'
        elif 'force' in answer_lower:
            result['role'] = 'force'
        
        # Known good verbs to look for
        good_verbs = {
            'studies', 'study', 'examines', 'examine', 'investigates', 'investigate',
            'explores', 'explore', 'analyzes', 'analyze', 'describes', 'describe',
            'explains', 'explain', 'understands', 'understand', 'discovers', 'discover',
            'observes', 'observe', 'measures', 'measure', 'tests', 'test',
            'solves', 'solve', 'deduces', 'deduce', 'reasons', 'reason',
            'assists', 'assist', 'helps', 'help', 'supports', 'support',
            'documents', 'document', 'records', 'record', 'provides', 'provide',
            'encompasses', 'encompasses', 'includes', 'include', 'involves', 'involve',
            'delves', 'delve', 'focuses', 'focus', 'addresses', 'address',
            'illuminates', 'illuminate', 'elucidates', 'elucidate', 'clarifies', 'clarify',
            'categorizes', 'categorize', 'classifies', 'classify', 'organizes', 'organize',
            'processes', 'process', 'integrates', 'integrate', 'synthesizes', 'synthesize',
        }
        
        # Extract verbs by looking for known good verbs in the text
        found_verbs = []
        words = re.findall(r'\b\w+\b', answer_lower)
        for word in words:
            if word in good_verbs and word not in found_verbs:
                found_verbs.append(word)
        
        result['actions'] = found_verbs[:4]  # Limit to 4 verbs
        
        # Extract targets - look for nouns after key phrases
        target_patterns = [
            r'(?:study of|exploration of|understanding of|involves?|encompasses?|includes?)\s+(\w+)',
            r'(?:into|about|regarding)\s+(\w+)',
            r'(?:matter|energy|life|nature|mind|brain|force|gravity|light|evolution|genetics|ecology)',
        ]
        
        targets = []
        for pattern in target_patterns:
            matches = re.findall(pattern, answer_lower)
            for m in matches:
                if isinstance(m, str) and len(m) > 3 and m not in targets:
                    # Skip common words
                    if m not in {'that', 'this', 'which', 'what', 'how', 'the', 'and', 'various'}:
                        targets.append(m)
        
        result['targets'] = targets[:4]  # Limit to 4 targets
        
        return result
    
    def apply_modifications(self, mods: List[CorpusModification], strength: int = 10) -> Dict:
        """
        Apply modifications to the corpus.
        
        This is where the magic happens - we modify the geometric structure
        based on the corrections.
        
        Args:
            mods: List of modifications to apply
            strength: How many times to reinforce each modification (default 10)
        """
        results = {
            'applied': [],
            'skipped': [],
            'frames_added': 0,
            'concepts_modified': 0,
        }
        
        for mod in mods:
            if mod.confidence < 0.5:
                results['skipped'].append(mod)
                continue
            
            entity = mod.entity
            
            if mod.mod_type == 'add_action':
                # Create new frames with this entity doing this action
                action = mod.details['action']
                # Clean up action (remove entity name if present)
                action = action.replace(f'{entity} ', '').replace(f'{entity.title()} ', '')
                action = action.strip()
                
                # Clean up common issues
                action = re.sub(r'^and\s+', '', action)  # Remove leading "and"
                action = re.sub(r'\s+mysteries$', '', action)  # Remove trailing "mysteries"
                action = re.sub(r'\s+cases$', '', action)  # Remove trailing "cases"
                action = re.sub(r'\s+matter$', '', action)  # Remove trailing "matter"
                action = re.sub(r'\s+holmes$', '', action)  # Remove trailing "holmes"
                
                # Keep verbs in base form for proper conjugation
                # Common verb endings to normalize
                if action.endswith('ves'):
                    action = action[:-1]  # solves -> solve (keep the e)
                elif action.endswith('ies'):
                    action = action[:-3] + 'y'  # studies -> study
                elif action.endswith('es') and len(action) > 4:
                    action = action[:-1]  # provides -> provide (keep one e)
                elif action.endswith('s') and len(action) > 3 and not action.endswith('ss'):
                    action = action[:-1]  # assists -> assist
                
                # Skip if action is too short or starts with "is"
                # Also skip common nouns that get misidentified as verbs
                noun_words = {'more', 'mores', 'philosophy', 'philosophies', 'ideology',
                              'ideological', 'various', 'utilization', 'curation',
                              'enhancement', 'relevance', 'diverse', 'comprehensive',
                              'crucial', 'fundamental', 'integral', 'framework',
                              'aspect', 'role', 'concept', 'entity', 'figure',
                              'practice', 'field', 'discipline', 'structure'}
                
                if action and len(action) > 3 and not action.startswith('is '):
                    if action.lower() in noun_words:
                        results['skipped'].append(mod)
                        continue
                    
                    # Add multiple frames for reinforcement
                    for _ in range(strength):
                        frame_text = f"{entity.title()} {action}."
                        self.knowledge.learn(frame_text, source="reinforcement")
                        results['frames_added'] += 1
                    results['applied'].append(mod)
            
            elif mod.mod_type == 'add_target':
                # Create frames with entity acting on target
                target = mod.details['target']
                # Find a common action for this entity
                if entity in self.knowledge.concepts:
                    concept = self.knowledge.concepts[entity]
                    # Get most common action
                    if hasattr(concept, 'actions') and concept.actions:
                        action = concept.actions.most_common(1)[0][0]
                    else:
                        action = "involves"
                else:
                    action = "involves"
                
                # Add multiple frames for reinforcement
                for _ in range(strength):
                    frame_text = f"{entity.title()} {action} {target}."
                    self.knowledge.learn(frame_text, source="reinforcement")
                    results['frames_added'] += 1
                results['applied'].append(mod)
            
            elif mod.mod_type == 'adjust_role':
                # Adjust role counts for the concept
                if entity in self.knowledge.concepts:
                    concept = self.knowledge.concepts[entity]
                    new_role = mod.details['new_role']
                    
                    # Boost the appropriate role count significantly
                    boost = strength * 2
                    if new_role == 'protagonist':
                        concept.initiator_count += boost
                    elif new_role == 'concept':
                        concept.mediator_count += boost
                    elif new_role == 'entity':
                        concept.initiator_count += boost
                    
                    results['concepts_modified'] += 1
                    results['applied'].append(mod)
            
            elif mod.mod_type == 'reduce_action':
                # We don't remove frames, but we could add counter-frames
                # For now, just note it
                results['skipped'].append(mod)
        
        return results


class GeometricRL:
    """
    Geometric Reinforcement Learning system.
    
    This creates a feedback loop:
    1. Generate answer
    2. Apply lens
    3. Get correction (manual or automatic)
    4. Reverse project to corpus modifications
    5. Apply modifications
    6. Repeat
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.qa = GeometricQA()
        self.qa.load_corpus(corpus_path)
        self.qa.set_output_lens('natural')
        
        self.reverse_projector = ReverseProjector(self.qa)
        
        self.history = []  # Track corrections and their effects
    
    def generate(self, query: str) -> str:
        """Generate an answer with the current corpus."""
        return self.qa.ask(query)
    
    def correct(self, query: str, corrected_answer: str) -> Dict:
        """
        Apply a correction and modify the corpus.
        
        Args:
            query: The original query
            corrected_answer: What the answer SHOULD be
        
        Returns:
            Results of the modification
        """
        # Get current answer
        original_answer = self.generate(query)
        
        # Extract entity from query
        entity = self._extract_entity(query)
        
        # Create correction
        correction = Correction(
            query=query,
            original_answer=original_answer,
            corrected_answer=corrected_answer,
            entity=entity,
            correction_type='description',
        )
        
        # Analyze and get modifications
        mods = self.reverse_projector.analyze_correction(correction)
        
        # Apply modifications
        results = self.reverse_projector.apply_modifications(mods)
        
        # Track in history
        self.history.append({
            'correction': correction,
            'modifications': mods,
            'results': results,
        })
        
        return results
    
    def _extract_entity(self, query: str) -> str:
        """Extract the main entity from a query."""
        query_lower = query.lower()
        
        # Common patterns
        patterns = [
            r'what does (\w+) do',
            r'who is (\w+)',
            r'describe (\w+)',
            r'tell me about (\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: find first content word
        words = re.findall(r'\b\w+\b', query_lower)
        stop_words = {'what', 'who', 'is', 'does', 'do', 'the', 'a', 'an', 'describe', 'tell', 'me', 'about'}
        for word in words:
            if word not in stop_words:
                return word
        
        return words[-1] if words else ""
    
    def save_corpus(self, path: str = None):
        """Save the modified corpus."""
        path = path or self.corpus_path
        
        # Convert knowledge to JSON format
        data = {
            'frames': [
                {
                    'text': f.text,
                    'source': f.source,
                    'agent': f.initiator,
                }
                for f in self.qa.knowledge.frames
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved corpus to {path} ({len(data['frames'])} frames)")
    
    def test_improvement(self, query: str) -> Tuple[str, str]:
        """Test if a correction improved the answer."""
        new_answer = self.generate(query)
        
        if self.history:
            last = self.history[-1]
            original = last['correction'].original_answer
            return original, new_answer
        
        return "", new_answer


def demo():
    """Demonstrate geometric reinforcement learning."""
    print("=" * 70)
    print("GEOMETRIC REINFORCEMENT LEARNING DEMO")
    print("=" * 70)
    print()
    print("This experiment tests whether we can improve the corpus by")
    print("projecting corrections backward through the output lens.")
    print()
    
    # Use experimental corpus
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    
    grl = GeometricRL(corpus_path)
    
    # Test queries
    test_queries = [
        "What does Holmes do?",
        "What does Watson do?",
        "What does physics do?",
    ]
    
    print("BEFORE CORRECTIONS:")
    print("-" * 70)
    for q in test_queries:
        answer = grl.generate(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()
    
    # Apply corrections
    print("APPLYING CORRECTIONS:")
    print("-" * 70)
    
    corrections = [
        ("What does Holmes do?", 
         "Holmes is a detective who investigates, deduces, and solves mysteries. This relates to crime and evidence."),
        ("What does Watson do?",
         "Watson is a doctor who assists Holmes, documents cases, and provides medical expertise. This relates to medicine and friendship."),
        ("What does physics do?",
         "Physics is a science that studies matter, energy, and the fundamental forces of nature. This relates to mathematics and experimentation."),
    ]
    
    for query, corrected in corrections:
        print(f"Correcting: {query}")
        print(f"  Target: {corrected[:60]}...")
        results = grl.correct(query, corrected)
        print(f"  Applied: {len(results['applied'])} modifications")
        print(f"  Frames added: {results['frames_added']}")
        print()
    
    print("AFTER CORRECTIONS:")
    print("-" * 70)
    for q in test_queries:
        answer = grl.generate(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()
    
    # Show what changed
    print("MODIFICATION HISTORY:")
    print("-" * 70)
    for i, entry in enumerate(grl.history):
        print(f"Correction {i+1}: {entry['correction'].entity}")
        for mod in entry['modifications']:
            print(f"  - {mod.mod_type}: {mod.details}")
    
    # Save the modified corpus
    print()
    print("Saving modified corpus...")
    grl.save_corpus()
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
