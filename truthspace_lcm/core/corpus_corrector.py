#!/usr/bin/env python3
"""
Corpus Corrector: Automated Knowledge Correction System

This module provides tools to automatically correct the corpus when
answers don't match expectations. Given a query and desired answer,
it modifies the corpus to produce that answer.

Key capabilities:
1. Parse desired answers to extract structure (entity, role, actions, targets)
2. Add new frames to reinforce desired knowledge
3. Reduce weight of incorrect associations
4. Save corrected corpus

Author: Lesley Gushurst
License: GPLv3
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedAnswer:
    """Structured representation of an answer."""
    entity: str
    role: Optional[str] = None  # detective, science, field, etc.
    actions: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    description: Optional[str] = None  # Full description if provided
    raw: str = ""


class AnswerParser:
    """Parse natural language answers into structured form."""
    
    def __init__(self):
        # Role indicators
        self.role_patterns = [
            (r'is (?:a|an|the) (\w+) (?:who|that|which)', 1),  # "is a detective who"
            (r'is (?:a|an|the) (\w+)\.', 1),  # "is a science."
            (r'is (?:a|an|the) (\w+ \w+) (?:who|that)', 1),  # "is a consulting detective who"
        ]
        
        # Action patterns
        self.action_patterns = [
            r'(?:who|that|which) ([^.]+?)(?:\.|,\s*(?:often|usually|typically))',
            r'(?:who|that|which) ([^.]+)\.',
        ]
        
        # Target patterns  
        self.target_patterns = [
            r'(?:involves?|relates? to|concerns?|deals with|studies|examines) ([^.]+)\.',
            r'(?:of|about|regarding) ([^.]+?)(?:\.|,)',
        ]
    
    def parse(self, answer: str, entity: str) -> ParsedAnswer:
        """Parse an answer string into structured form."""
        result = ParsedAnswer(entity=entity.lower(), raw=answer)
        answer_lower = answer.lower()
        
        # Extract role
        for pattern, group in self.role_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                result.role = match.group(group).strip()
                break
        
        # Extract actions
        for pattern in self.action_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                actions_str = match.group(1)
                # Split on commas and "and"
                actions = re.split(r',\s*|\s+and\s+', actions_str)
                result.actions = [a.strip() for a in actions if a.strip() and len(a.strip()) > 2]
                break
        
        # Extract targets
        for pattern in self.target_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                targets_str = match.group(1)
                targets = re.split(r',\s*|\s+and\s+', targets_str)
                result.targets = [t.strip() for t in targets if t.strip() and len(t.strip()) > 2]
                break
        
        # Store full description
        result.description = answer
        
        return result


class CorpusCorrector:
    """
    Corrects corpus knowledge based on desired answers.
    
    Usage:
        corrector = CorpusCorrector('corpus_experimental.json')
        corrector.correct(
            query="What is physics?",
            desired="Physics is the science of matter and energy."
        )
        corrector.save()
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.parser = AnswerParser()
        
        # Load corpus
        from truthspace_lcm.core.geometric import GeometricQA
        self.qa = GeometricQA()
        self.qa.load_corpus(str(self.corpus_path))
        self.qa.set_output_lens('natural')
        
        self.corrections = []  # Track all corrections
    
    def ask(self, query: str) -> str:
        """Get current answer for a query."""
        return self.qa.ask(query)
    
    def correct(self, query: str, desired: str, strength: int = 30) -> Dict:
        """
        Correct the corpus to produce the desired answer.
        
        Args:
            query: The question being asked
            desired: The answer we want
            strength: How many frames to add for reinforcement (default 30)
        
        Returns:
            Dict with correction details
        """
        # Get current answer
        current = self.ask(query)
        
        # Extract entity from query
        entity = self._extract_entity(query)
        
        # Parse both answers
        current_parsed = self.parser.parse(current, entity)
        desired_parsed = self.parser.parse(desired, entity)
        
        result = {
            'entity': entity,
            'current': current,
            'desired': desired,
            'frames_added': 0,
            'frames_modified': 0,
            'changes': [],
        }
        
        # 1. Add role if different
        if desired_parsed.role and desired_parsed.role != current_parsed.role:
            self._add_role(entity, desired_parsed.role, strength)
            result['changes'].append(f"role: {current_parsed.role} → {desired_parsed.role}")
            result['frames_added'] += strength
        
        # 2. Add new actions
        current_actions = set(self._normalize_verbs(current_parsed.actions))
        desired_actions = set(self._normalize_verbs(desired_parsed.actions))
        
        for action in desired_actions - current_actions:
            self._add_action(entity, action, strength)
            result['changes'].append(f"add action: {action}")
            result['frames_added'] += strength
        
        # 3. Add new targets
        current_targets = set(current_parsed.targets)
        desired_targets = set(desired_parsed.targets)
        
        for target in desired_targets - current_targets:
            self._add_target(entity, target, strength)
            result['changes'].append(f"add target: {target}")
            result['frames_added'] += strength
        
        # 4. If we have a full description, add it as frames
        if desired_parsed.description:
            self._add_description_frames(entity, desired_parsed.description, strength // 3)
            result['frames_added'] += strength // 3
        
        # Track correction
        self.corrections.append(result)
        
        return result
    
    def correct_exact(self, entity: str, description: str, strength: int = 50) -> Dict:
        """
        Add an exact description for an entity.
        
        This is useful when you want to define exactly what something is,
        rather than correcting an existing answer.
        
        Args:
            entity: The entity to describe
            description: The exact description to add
            strength: How many times to reinforce
        
        Returns:
            Dict with correction details
        """
        result = {
            'entity': entity,
            'description': description,
            'frames_added': 0,
            'changes': [],
        }
        
        # Parse the description
        parsed = self.parser.parse(description, entity)
        
        # Add role frame
        if parsed.role:
            role_frame = f"{entity.title()} is a {parsed.role}."
            for _ in range(strength):
                self.qa.knowledge.learn(role_frame, source="correction")
            result['frames_added'] += strength
            result['changes'].append(f"role: {parsed.role}")
        
        # Add action frames
        for action in parsed.actions:
            action_frame = f"{entity.title()} {action}."
            for _ in range(strength):
                self.qa.knowledge.learn(action_frame, source="correction")
            result['frames_added'] += strength
            result['changes'].append(f"action: {action}")
        
        # Add target frames
        for target in parsed.targets:
            # Find a good action to use
            if parsed.actions:
                action = parsed.actions[0]
            else:
                action = "involves"
            target_frame = f"{entity.title()} {action} {target}."
            for _ in range(strength):
                self.qa.knowledge.learn(target_frame, source="correction")
            result['frames_added'] += strength
            result['changes'].append(f"target: {target}")
        
        # Add the full description as a frame too
        for _ in range(strength):
            self.qa.knowledge.learn(description, source="correction")
        result['frames_added'] += strength
        
        self.corrections.append(result)
        return result
    
    def define(self, entity: str, definition: str, strength: int = 50) -> Dict:
        """
        Define what an entity IS (not what it does).
        
        This creates frames like "Physics is the science of matter and energy."
        
        Args:
            entity: The entity to define
            definition: The definition (can be just the predicate, e.g., "the science of matter")
            strength: Reinforcement strength
        """
        result = {
            'entity': entity,
            'definition': definition,
            'frames_added': 0,
        }
        
        # Create the full definition frame
        if not definition.lower().startswith(entity.lower()):
            full_def = f"{entity.title()} is {definition}"
        else:
            full_def = definition
        
        # Add the definition frame
        for _ in range(strength):
            self.qa.knowledge.learn(full_def, source="definition")
        result['frames_added'] += strength
        
        # Also extract and add component parts
        # E.g., "the science of matter and energy" → targets: matter, energy
        targets = re.findall(r'of (\w+)', definition.lower())
        for target in targets:
            target_frame = f"{entity.title()} studies {target}."
            for _ in range(strength // 2):
                self.qa.knowledge.learn(target_frame, source="definition")
            result['frames_added'] += strength // 2
        
        self.corrections.append(result)
        return result
    
    def set_answer(self, entity: str, role: str, actions: List[str], targets: List[str], 
                   strength: int = 100) -> Dict:
        """
        Directly set what the answer should be for an entity.
        
        This is the most direct way to control output - you specify exactly
        what role, actions, and targets should appear.
        
        Args:
            entity: The entity (e.g., "physics")
            role: The role/type (e.g., "science", "detective", "field")
            actions: List of actions (e.g., ["studies", "examines", "investigates"])
            targets: List of targets (e.g., ["matter", "energy", "interactions"])
            strength: How strongly to reinforce (default 100)
        
        Example:
            corrector.set_answer(
                entity="physics",
                role="science",
                actions=["studies", "examines", "investigates"],
                targets=["matter", "energy", "interactions"]
            )
        """
        result = {
            'entity': entity,
            'role': role,
            'actions': actions,
            'targets': targets,
            'frames_added': 0,
        }
        
        entity_title = entity.title()
        
        # Normalize actions to 3rd person singular (e.g., "study" -> "studies")
        normalized_actions = []
        for action in actions:
            action = action.strip().lower()
            # Skip if already in 3rd person
            if action.endswith('es') or action.endswith('ies'):
                normalized_actions.append(action)
                continue
            # Convert to 3rd person if needed
            if not action.endswith('s'):
                if action.endswith('y') and len(action) > 2 and action[-2] not in 'aeiou':
                    action = action[:-1] + 'ies'
                elif action.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    action = action + 'es'
                else:
                    action = action + 's'
            normalized_actions.append(action)
        
        # 1. Add action frames (these define what the entity DOES)
        # Use sentence structure: "Entity studies X" not "Entity is a Y"
        for action in normalized_actions:
            # Add action with each target
            for target in targets:
                action_frame = f"{entity_title} {action} {target}."
                for _ in range(strength):
                    self.qa.knowledge.learn(action_frame, source="set_answer")
                result['frames_added'] += strength
        
        # 2. Add role association through targets (not "is a" which creates "is" as action)
        # Instead of "Physics is a science", use "Physics, the science, studies matter"
        for target in targets:
            role_frame = f"{entity_title}, the {role}, {normalized_actions[0]} {target}."
            for _ in range(strength // 2):
                self.qa.knowledge.learn(role_frame, source="set_answer")
            result['frames_added'] += strength // 2
        
        # 3. Also add simple action frames
        for action in normalized_actions:
            simple_frame = f"{entity_title} {action}."
            for _ in range(strength // 2):
                self.qa.knowledge.learn(simple_frame, source="set_answer")
            result['frames_added'] += strength // 2
        
        self.corrections.append(result)
        return result
    
    def _extract_entity(self, query: str) -> str:
        """Extract the main entity from a query."""
        query_lower = query.lower()
        
        patterns = [
            r'what (?:is|does) (\w+)',
            r'who is (\w+)',
            r'describe (\w+)',
            r'tell me about (\w+)',
            r'what about (\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: find first content word
        words = re.findall(r'\b\w+\b', query_lower)
        stop_words = {'what', 'who', 'is', 'does', 'do', 'the', 'a', 'an', 'describe', 'tell', 'me', 'about'}
        for word in words:
            if word not in stop_words and len(word) > 2:
                return word
        
        return words[-1] if words else ""
    
    def _normalize_verbs(self, verbs: List[str]) -> List[str]:
        """Normalize verb forms to base form."""
        normalized = []
        for verb in verbs:
            v = verb.lower().strip()
            # Remove common suffixes
            if v.endswith('ies'):
                v = v[:-3] + 'y'
            elif v.endswith('es'):
                v = v[:-2]
            elif v.endswith('s') and not v.endswith('ss'):
                v = v[:-1]
            elif v.endswith('ing'):
                v = v[:-3]
            elif v.endswith('ed'):
                v = v[:-2]
            normalized.append(v)
        return normalized
    
    def _add_role(self, entity: str, role: str, strength: int):
        """Add frames establishing entity's role."""
        frame = f"{entity.title()} is a {role}."
        for _ in range(strength):
            self.qa.knowledge.learn(frame, source="correction")
    
    def _add_action(self, entity: str, action: str, strength: int):
        """Add frames with entity performing action."""
        # Normalize action
        action = action.strip()
        if not action:
            return
        
        frame = f"{entity.title()} {action}."
        for _ in range(strength):
            self.qa.knowledge.learn(frame, source="correction")
    
    def _add_target(self, entity: str, target: str, strength: int):
        """Add frames with entity acting on target."""
        # Find a good action for this entity
        if entity in self.qa.knowledge.concepts:
            c = self.qa.knowledge.concepts[entity]
            if c.actions:
                action = c.actions.most_common(1)[0][0]
            else:
                action = "involves"
        else:
            action = "involves"
        
        frame = f"{entity.title()} {action} {target}."
        for _ in range(strength):
            self.qa.knowledge.learn(frame, source="correction")
    
    def _add_description_frames(self, entity: str, description: str, strength: int):
        """Add the full description as frames."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', description)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                for _ in range(strength):
                    self.qa.knowledge.learn(sentence, source="correction")
    
    def save(self, path: str = None):
        """Save the corrected corpus."""
        path = path or str(self.corpus_path)
        
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
    
    def test(self, query: str) -> str:
        """Test the current answer for a query."""
        return self.ask(query)
    
    def summary(self) -> str:
        """Get a summary of all corrections made."""
        if not self.corrections:
            return "No corrections made yet."
        
        lines = [f"Corrections made: {len(self.corrections)}"]
        total_frames = sum(c.get('frames_added', 0) for c in self.corrections)
        lines.append(f"Total frames added: {total_frames}")
        lines.append("")
        
        for i, c in enumerate(self.corrections, 1):
            lines.append(f"{i}. {c.get('entity', 'unknown')}")
            for change in c.get('changes', []):
                lines.append(f"   - {change}")
        
        return "\n".join(lines)


def interactive_correction(corpus_path: str = "truthspace_lcm/corpus_experimental.json"):
    """Interactive correction session."""
    print("=" * 60)
    print("INTERACTIVE CORPUS CORRECTION")
    print("=" * 60)
    print()
    print("Commands:")
    print("  ask <query>          - Ask a question")
    print("  correct <query>      - Correct the answer to a query")
    print("  define <entity>      - Define what an entity is")
    print("  save                 - Save the corpus")
    print("  summary              - Show correction summary")
    print("  quit                 - Exit")
    print()
    
    corrector = CorpusCorrector(corpus_path)
    
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'save':
            corrector.save()
            continue
        
        if user_input.lower() == 'summary':
            print(corrector.summary())
            continue
        
        if user_input.lower().startswith('ask '):
            query = user_input[4:].strip()
            answer = corrector.ask(query)
            print(f"A: {answer}")
            continue
        
        if user_input.lower().startswith('correct '):
            query = user_input[8:].strip()
            current = corrector.ask(query)
            print(f"Current: {current}")
            print()
            desired = input("Desired answer: ").strip()
            if desired:
                result = corrector.correct(query, desired)
                print(f"Applied {len(result['changes'])} changes, added {result['frames_added']} frames")
                print()
                print(f"New answer: {corrector.ask(query)}")
            continue
        
        if user_input.lower().startswith('define '):
            entity = user_input[7:].strip()
            print(f"Define {entity}:")
            definition = input("Definition: ").strip()
            if definition:
                result = corrector.define(entity, definition)
                print(f"Added {result['frames_added']} frames")
                print()
                print(f"New answer: {corrector.ask(f'What is {entity}?')}")
            continue
        
        # Default: treat as a query
        answer = corrector.ask(user_input)
        print(f"A: {answer}")


if __name__ == "__main__":
    interactive_correction()
