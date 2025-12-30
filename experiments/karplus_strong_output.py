#!/usr/bin/env python3
"""
Karplus-Strong Output Lens: Natural Language Through Structured Noise

The Karplus-Strong algorithm creates natural-sounding plucked strings by:
1. Initial noise burst (excitation)
2. Delay line (recirculating buffer = the "string")
3. Low-pass filter (decay/damping)

For language output, we translate this to:
1. Initial concept burst (the core semantic content from geometry)
2. Expansion buffer (related concepts that "resonate")
3. Attention decay (concepts fade as we move away from the query focus)

The key insight: Natural language isn't geometrically pure - it has:
- Noise: Minor variations, synonyms, tangential associations
- Decay: Attention fades, earlier context matters less
- Resonance: Related concepts reinforce each other

This is the opposite of our rigorous GeometricLCM - we WANT controlled imperfection.

Author: Lesley Gushurst
License: GPLv3
"""

import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ConceptWave:
    """
    A concept with amplitude (relevance) and phase (position in output).
    
    Like a wave in Karplus-Strong, concepts have:
    - amplitude: how strongly they resonate with the query
    - phase: where they appear in the output sequence
    - decay: how quickly they fade
    """
    word: str
    amplitude: float  # 0-1, relevance to query
    phase: float  # 0-1, position in output
    decay: float = 0.95  # How much amplitude drops per step
    
    def tick(self) -> float:
        """Advance one step, return current amplitude, apply decay."""
        current = self.amplitude
        self.amplitude *= self.decay
        return current


class KarplusStrongLens:
    """
    Output lens that generates natural language using Karplus-Strong principles.
    
    Instead of rigid template filling, we:
    1. Excite the concept space with the query
    2. Let related concepts resonate
    3. Apply attention decay as we generate
    4. Inject controlled noise for naturalness
    """
    
    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        
        # Karplus-Strong parameters
        self.noise_level = 0.15  # How much randomness to inject
        self.decay_rate = 0.85  # How fast attention fades
        self.resonance_depth = 3  # How many hops of related concepts
        
        # Sentence templates with slots for variation
        self.openers = [
            "{entity} is {role} that",
            "{entity}, as {role},",
            "As {role}, {entity}",
            "{entity} can be understood as {role} that",
        ]
        
        self.action_connectors = [
            " {action}",
            " {action} and",
            ", {action},",
            " primarily {action}",
        ]
        
        self.target_phrases = [
            " {target}",
            " {target} and related concepts",
            " aspects of {target}",
            " the nature of {target}",
        ]
        
        self.closers = [
            ".",
            ", among other things.",
            " in various ways.",
            " as part of its broader scope.",
        ]
        
        # Expansion phrases for longer output
        self.expansions = [
            "This involves {related}.",
            "Related to this is {related}.",
            "{related} plays a role here.",
            "One aspect is {related}.",
            "This connects to {related}.",
        ]
    
    def generate(self, entity: str, role: str, actions: List[str], 
                 targets: List[str], related: List[str] = None) -> str:
        """
        Generate natural-sounding output using Karplus-Strong principles.
        
        Args:
            entity: Main subject (e.g., "physics")
            role: What it is (e.g., "science")
            actions: What it does (e.g., ["studies", "examines"])
            targets: What it acts on (e.g., ["matter", "energy"])
            related: Related concepts for expansion
        """
        # Apply decay to select which items to include
        selected_actions = self._apply_decay(actions, start_prob=0.95)
        selected_targets = self._apply_decay(targets, start_prob=0.85)
        selected_related = self._apply_decay(related or [], start_prob=0.6)
        
        # Build output with noise (random template selection)
        output = self._synthesize_v2(entity, role, selected_actions, 
                                      selected_targets, selected_related)
        
        return output
    
    def _apply_decay(self, items: List[str], start_prob: float = 0.9) -> List[str]:
        """Apply Karplus-Strong decay: each successive item less likely to be included."""
        selected = []
        prob = start_prob
        for item in items:
            if random.random() < prob:
                selected.append(item)
            prob *= self.decay_rate
        return selected
    
    def _excite(self, entity: str, role: str, actions: List[str], 
                targets: List[str], related: List[str]) -> List[ConceptWave]:
        """Create initial concept waves from input."""
        waves = []
        
        # Entity and role are strongest
        waves.append(ConceptWave(entity, 1.0, 0.0))
        waves.append(ConceptWave(role, 0.95, 0.1))
        
        # Actions decay slightly
        for i, action in enumerate(actions):
            amp = 0.9 - (i * 0.1)
            phase = 0.2 + (i * 0.1)
            waves.append(ConceptWave(action, amp, phase, decay=self.decay_rate))
        
        # Targets decay more
        for i, target in enumerate(targets):
            amp = 0.8 - (i * 0.1)
            phase = 0.5 + (i * 0.1)
            waves.append(ConceptWave(target, amp, phase, decay=self.decay_rate * 0.9))
        
        # Related concepts are weakest
        for i, rel in enumerate(related):
            amp = 0.5 - (i * 0.1)
            phase = 0.7 + (i * 0.05)
            waves.append(ConceptWave(rel, max(0.1, amp), phase, decay=self.decay_rate * 0.8))
        
        return waves
    
    def _resonate(self, waves: List[ConceptWave]) -> List[ConceptWave]:
        """
        Let waves interact - similar concepts reinforce each other.
        
        This is like the delay line in Karplus-Strong where the wave
        recirculates and interferes with itself.
        """
        # Simple resonance: boost waves that share letters/sounds
        for i, w1 in enumerate(waves):
            for j, w2 in enumerate(waves):
                if i != j:
                    # Crude similarity: shared prefix
                    if w1.word[:3] == w2.word[:3]:
                        w1.amplitude *= 1.1
                        w2.amplitude *= 1.1
        
        # Normalize
        max_amp = max(w.amplitude for w in waves) if waves else 1
        for w in waves:
            w.amplitude = min(1.0, w.amplitude / max_amp)
        
        return waves
    
    def _synthesize(self, entity: str, role: str, waves: List[ConceptWave]) -> str:
        """
        Generate output by sampling from waves with decay and noise.
        """
        # Sort waves by phase (order in output)
        waves.sort(key=lambda w: w.phase)
        
        # Extract components
        actions = [w.word for w in waves if 0.2 <= w.phase < 0.5]
        targets = [w.word for w in waves if 0.5 <= w.phase < 0.7]
        related = [w.word for w in waves if w.phase >= 0.7]
        
        # Build output with noise injection
        parts = []
        
        # 1. Opening (with noise = random template selection)
        opener = self._noisy_choice(self.openers)
        opener = opener.format(entity=entity.title(), role=f"a {role}")
        parts.append(opener)
        
        # 2. Actions (with decay = fewer actions as we go)
        if actions:
            # Apply decay: probability of including each action decreases
            included_actions = []
            prob = 1.0
            for action in actions:
                if random.random() < prob:
                    included_actions.append(action)
                prob *= self.decay_rate
            
            if included_actions:
                if len(included_actions) == 1:
                    parts.append(f" {included_actions[0]}")
                elif len(included_actions) == 2:
                    parts.append(f" {included_actions[0]} and {included_actions[1]}")
                else:
                    parts.append(f" {', '.join(included_actions[:-1])}, and {included_actions[-1]}")
        
        # 3. Targets (with noise = variation in phrasing)
        if targets:
            # Apply decay
            included_targets = []
            prob = 0.9
            for target in targets:
                if random.random() < prob:
                    included_targets.append(target)
                prob *= self.decay_rate
            
            if included_targets:
                target_phrase = self._noisy_choice([
                    f" {' and '.join(included_targets)}",
                    f" aspects of {included_targets[0]}" + (f" and {included_targets[1]}" if len(included_targets) > 1 else ""),
                    f" the nature of {included_targets[0]}",
                ])
                parts.append(target_phrase)
        
        # 4. Closer
        parts.append(self._noisy_choice(self.closers))
        
        # 5. Expansion sentences (if we have related concepts)
        if related and random.random() > 0.3:  # 70% chance of expansion
            expansion_count = min(len(related), random.randint(1, 3))
            for i in range(expansion_count):
                if i < len(related):
                    exp = self._noisy_choice(self.expansions)
                    exp = exp.format(related=related[i])
                    parts.append(" " + exp)
        
        return ''.join(parts)
    
    def _synthesize_v2(self, entity: str, role: str, actions: List[str],
                       targets: List[str], related: List[str]) -> str:
        """
        Cleaner synthesis that keeps actions, targets, and related separate.
        Uses noise for template variation, decay already applied to inputs.
        """
        parts = []
        
        # 1. Opening - with noise in template selection
        opener_templates = [
            "{entity} is {role} that",
            "{entity}, {role},",
            "As {role}, {entity}",
        ]
        opener = self._noisy_choice(opener_templates)
        parts.append(opener.format(entity=entity.title(), role=f"a {role}"))
        
        # 2. Actions
        if actions:
            if len(actions) == 1:
                parts.append(f" {actions[0]}")
            elif len(actions) == 2:
                parts.append(f" {actions[0]} and {actions[1]}")
            else:
                parts.append(f" {', '.join(actions[:-1])}, and {actions[-1]}")
        
        # 3. Targets - with noise in phrasing
        if targets:
            target_templates = [
                " {targets}",
                " the nature of {targets}",
                " aspects of {targets}",
                " phenomena involving {targets}",
            ]
            if len(targets) == 1:
                target_str = targets[0]
            elif len(targets) == 2:
                target_str = f"{targets[0]} and {targets[1]}"
            else:
                target_str = f"{', '.join(targets[:-1])}, and {targets[-1]}"
            
            target_phrase = self._noisy_choice(target_templates)
            parts.append(target_phrase.format(targets=target_str))
        
        # 4. Sentence ending
        parts.append(".")
        
        # 5. Expansion sentences for related concepts (noise determines how many)
        if related:
            expansion_templates = [
                " This involves {concept}.",
                " This relates to {concept}.",
                " {concept} is connected to this.",
                " One aspect of this is {concept}.",
                " This encompasses {concept}.",
            ]
            
            for concept in related:
                exp = self._noisy_choice(expansion_templates)
                parts.append(exp.format(concept=concept))
        
        return ''.join(parts)
    
    def _noisy_choice(self, options: List[str]) -> str:
        """
        Choose from options with noise.
        
        Instead of uniform random, we bias toward earlier options
        but allow noise to select later ones.
        """
        if not options:
            return ""
        
        # Weighted selection: earlier options more likely
        weights = [1.0 / (i + 1) for i in range(len(options))]
        
        # Add noise
        weights = [w + random.random() * self.noise_level for w in weights]
        
        # Normalize and select
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return options[i]
        
        return options[0]


class ConceptChain:
    """
    A chain of concepts that supports extended discourse.
    
    The idea: to "prattle on" about a topic, we need to:
    1. Start with a focal concept
    2. Generate a sentence about it
    3. Pick a related concept to transition to
    4. Generate a sentence about that
    5. Repeat with decay until we've said enough
    
    This is like a random walk through concept space, but biased
    toward staying near the original topic (the "meta concept").
    """
    
    def __init__(self, knowledge, focal_concept: str):
        self.knowledge = knowledge
        self.focal = focal_concept
        self.visited = set()
        self.chain = []
        self.attention = 1.0  # Decays as we go
        self.decay_rate = 0.85
        
    def next_concept(self) -> Optional[str]:
        """
        Get the next concept to talk about.
        
        Uses a weighted selection based on:
        - Connection strength to current concept
        - Connection strength to focal concept (stay on topic)
        - Not already visited
        """
        current = self.chain[-1] if self.chain else self.focal
        
        if current not in self.knowledge.concepts:
            return None
            
        concept = self.knowledge.concepts[current]
        candidates = []
        
        # Get targets as candidates
        if concept.targets:
            for target, count in concept.targets.most_common(10):
                if target not in self.visited and target in self.knowledge.concepts:
                    if self.knowledge.concepts[target].is_content_word:
                        # Weight by count and connection to focal
                        focal_connection = self._connection_to_focal(target)
                        weight = count * (1 + focal_connection)
                        candidates.append((target, weight))
        
        # Get concepts that share actions
        if concept.actions:
            my_actions = set(a for a, _ in concept.actions.most_common(5))
            for name, other in self.knowledge.concepts.items():
                if name in self.visited or name == current:
                    continue
                if not other.is_content_word:
                    continue
                if other.actions:
                    other_actions = set(a for a, _ in other.actions.most_common(5))
                    shared = len(my_actions & other_actions)
                    if shared >= 2:
                        focal_connection = self._connection_to_focal(name)
                        weight = shared * (1 + focal_connection)
                        candidates.append((name, weight))
        
        if not candidates:
            return None
        
        # Weighted random selection
        total = sum(w for _, w in candidates)
        r = random.random() * total
        cumsum = 0
        for name, weight in candidates:
            cumsum += weight
            if r <= cumsum:
                self.visited.add(name)
                self.chain.append(name)
                self.attention *= self.decay_rate
                return name
        
        return candidates[0][0] if candidates else None
    
    def _connection_to_focal(self, concept_name: str) -> float:
        """How connected is this concept to the focal concept?"""
        if concept_name == self.focal:
            return 1.0
        
        focal = self.knowledge.concepts.get(self.focal)
        other = self.knowledge.concepts.get(concept_name)
        
        if not focal or not other:
            return 0.0
        
        # Check if they share targets
        if focal.targets and other.targets:
            focal_targets = set(t for t, _ in focal.targets.most_common(10))
            other_targets = set(t for t, _ in other.targets.most_common(10))
            shared = len(focal_targets & other_targets)
            if shared > 0:
                return min(1.0, shared * 0.2)
        
        # Check if they share actions
        if focal.actions and other.actions:
            focal_actions = set(a for a, _ in focal.actions.most_common(5))
            other_actions = set(a for a, _ in other.actions.most_common(5))
            shared = len(focal_actions & other_actions)
            if shared > 0:
                return min(1.0, shared * 0.15)
        
        return 0.0


class KarplusStrongQA:
    """
    Full QA system using Karplus-Strong output generation.
    
    Integrates with the knowledge corpus to:
    1. Extract entity info (role, actions, targets)
    2. Find related concepts through resonance
    3. Generate natural output with decay and noise
    """
    
    def __init__(self, corpus_path: str = None):
        self.lens = KarplusStrongLens()
        self.knowledge = None
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, path: str):
        """Load knowledge corpus."""
        from truthspace_lcm.core.geometric import GeometricQA
        self._qa = GeometricQA()
        self._qa.load_corpus(path)
        self.knowledge = self._qa.knowledge
    
    def ask(self, query: str) -> str:
        """Answer a query using Karplus-Strong output generation."""
        # Extract entity from query
        entity = self._extract_entity(query)
        
        if not entity or entity not in self.knowledge.concepts:
            return f"I don't have information about that."
        
        c = self.knowledge.concepts[entity]
        
        # Get role from targets or phi-direction
        role = self._get_role(c)
        
        # Get actions (filter out non-verbs)
        actions = self._get_actions(c)
        
        # Get targets
        targets = self._get_targets(c)
        
        # Get related concepts through resonance
        related = self._get_related(entity, c)
        
        # Generate with Karplus-Strong
        return self.lens.generate(entity, role, actions, targets, related)
    
    def prattle(self, query: str, sentences: int = 5) -> str:
        """
        Generate extended discourse about a topic.
        
        This "prattles on" by generating multiple sentences about the focal
        concept using different phrasings and perspectives. Instead of 
        following concept chains (which can introduce noise), we stay focused
        on what we know well about the main topic.
        
        Args:
            query: The topic to talk about
            sentences: Approximate number of sentences to generate
        """
        entity = self._extract_entity(query)
        
        if not entity or entity not in self.knowledge.concepts:
            return f"I don't have information about that."
        
        focal = self.knowledge.concepts[entity]
        paragraphs = []
        
        # Get the entity's attributes
        role = self._get_role(focal)
        actions = self._get_actions(focal)
        targets = self._get_targets(focal)
        
        # 1. Opening sentence - what it is and what it does
        first_sentence = self._generate_sentence(entity, role, actions, targets, is_first=True)
        paragraphs.append(first_sentence)
        
        # 2. Elaborate on the actions
        if len(actions) >= 2 and len(paragraphs) < sentences:
            elaboration = self._elaborate_actions(entity, actions, targets)
            paragraphs.append(elaboration)
        
        # 3. Discuss each target in relation to the entity
        for i, target in enumerate(targets):
            if len(paragraphs) >= sentences:
                break
            target_sentence = self._discuss_target(entity, role, actions, target, target_index=i)
            paragraphs.append(target_sentence)
        
        # 4. Add perspective sentences
        perspectives = self._generate_perspectives(entity, role, actions, targets)
        for p in perspectives:
            if len(paragraphs) >= sentences:
                break
            paragraphs.append(p)
        
        # 5. Closing summary if needed
        if len(paragraphs) < sentences:
            closing = self._generate_closing(entity, role, targets)
            paragraphs.append(closing)
        
        return ' '.join(paragraphs)
    
    def _discuss_target(self, entity: str, role: str, actions: List[str], target: str, 
                        target_index: int = 0) -> str:
        """Generate a sentence discussing a target in relation to the entity."""
        # Use different templates based on target index to avoid repetition
        all_templates = [
            "When it comes to {target}, {entity} {action} it thoroughly.",
            "{entity}'s relationship with {target} is central to its function as {role}.",
            "The study of {target} is a key aspect of what {entity} does.",
            "In terms of {target}, {entity} {action} this systematically.",
            "{target} represents one of the primary focuses of {entity}.",
            "{entity} engages with {target} in a rigorous manner.",
            "The connection between {entity} and {target} is fundamental.",
            "Through {target}, {entity} demonstrates its core purpose.",
        ]
        
        # Pick template based on index to ensure variety
        template = all_templates[target_index % len(all_templates)]
        
        # Use different actions for variety
        action = actions[target_index % len(actions)] if actions else "addresses"
        
        return template.format(entity=entity.title(), target=target, 
                              action=action, role=f"a {role}")
    
    def _generate_perspectives(self, entity: str, role: str, actions: List[str], 
                               targets: List[str]) -> List[str]:
        """Generate perspective sentences about the entity."""
        perspectives = []
        
        templates = [
            f"This makes {entity.title()} essential for understanding {targets[0] if targets else 'its domain'}.",
            f"The way {entity.title()} approaches these topics is methodical and rigorous.",
            f"As {role}, {entity.title()} provides valuable insights into these areas.",
            f"Understanding {entity.title()} helps clarify the nature of {targets[0] if targets else 'its subject matter'}.",
            f"The scope of {entity.title()} extends across multiple related areas.",
            f"This comprehensive approach defines {entity.title()}'s contribution to knowledge.",
        ]
        
        # Shuffle and return a subset
        random.shuffle(templates)
        return templates[:3]
    
    def _generate_closing(self, entity: str, role: str, targets: List[str]) -> str:
        """Generate a closing sentence."""
        closings = [
            f"In summary, {entity.title()} is {role} that engages deeply with {', '.join(targets[:2]) if targets else 'its subject matter'}.",
            f"Overall, {entity.title()} represents a significant {role} in this domain.",
            f"These elements together define what {entity.title()} fundamentally is and does.",
        ]
        return random.choice(closings)
    
    def _elaborate_actions(self, entity: str, actions: List[str], targets: List[str]) -> str:
        """Generate an elaboration sentence about what the entity does."""
        templates = [
            "In particular, {entity} {action} various aspects of {target}.",
            "The way {entity} {action} {target} is particularly notable.",
            "Through this, {entity} {action} {target} systematically.",
            "Specifically, {entity} {action} {target} in meaningful ways.",
        ]
        
        action = actions[0] if actions else "engages with"
        target = targets[0] if targets else "its domain"
        
        template = random.choice(templates)
        return template.format(entity=entity.title(), action=action, target=target)
    
    def _expand_on_target(self, focal_entity: str, target: str, transitions: List[str]) -> Optional[str]:
        """Generate a sentence expanding on a target concept."""
        if target not in self.knowledge.concepts:
            return None
        
        c = self.knowledge.concepts[target]
        
        templates = [
            "{transition}{target}, in relation to {focal}, {action} {sub_target}.",
            "{transition}{target} plays a key role, as it {action} {sub_target}.",
            "{transition}When considering {target}, we see that it {action} {sub_target}.",
            "{transition}{target} is important because it {action} {sub_target}.",
        ]
        
        # Get target's actions and targets
        actions = self._get_actions(c)
        sub_targets = self._get_targets(c)
        
        if not actions:
            return None
        
        action = actions[0]
        sub_target = sub_targets[0] if sub_targets else "various elements"
        transition = random.choice(transitions)
        
        template = random.choice(templates)
        return template.format(
            transition=transition,
            target=target.title(),
            focal=focal_entity,
            action=action,
            sub_target=sub_target
        )
    
    def _generate_filler(self, entity: str, role: str, actions: List[str], 
                         targets: List[str], used_fillers: set = None) -> Optional[str]:
        """Generate a filler sentence to extend discourse."""
        if used_fillers is None:
            used_fillers = set()
        
        all_fillers = [
            f"This makes {entity.title()} an important {role} in its domain.",
            f"The relationship between {entity.title()} and {targets[0] if targets else 'its subject'} is significant.",
            f"Understanding {entity.title()} requires considering these various aspects.",
            f"These characteristics define what {entity.title()} fundamentally is.",
            f"This is central to how {entity.title()} functions.",
            f"The interplay of these elements shapes {entity.title()}'s nature.",
            f"Each aspect contributes to the overall picture of {entity.title()}.",
            f"This perspective helps clarify what {entity.title()} represents.",
        ]
        
        available = [f for f in all_fillers if f not in used_fillers]
        if not available:
            return None
        
        choice = random.choice(available)
        used_fillers.add(choice)
        return choice
    
    def _find_anchored_concept(self, focal: str, focal_targets: set, 
                                focal_actions: set, visited: set) -> Optional[str]:
        """Find a concept that's connected to the focal topic."""
        candidates = []
        
        # Look for concepts that share targets with focal
        for name, concept in self.knowledge.concepts.items():
            if name in visited or not self._is_good_concept(name):
                continue
            
            score = 0
            
            # Check target overlap
            if concept.targets:
                other_targets = set(t for t, _ in concept.targets.most_common(10))
                shared_targets = focal_targets & other_targets
                score += len(shared_targets) * 2
            
            # Check action overlap
            if concept.actions:
                other_actions = set(a for a, _ in concept.actions.most_common(5))
                shared_actions = focal_actions & other_actions
                score += len(shared_actions)
            
            # Bonus if it's a direct target of focal
            if name in focal_targets:
                score += 5
            
            if score >= 2:
                candidates.append((name, score))
        
        if not candidates:
            return None
        
        # Weighted random selection
        candidates.sort(key=lambda x: -x[1])
        # Take from top candidates with some randomness
        top = candidates[:min(5, len(candidates))]
        return random.choice(top)[0]
    
    def _is_good_concept(self, name: str) -> bool:
        """Check if a concept is good for discourse (not noise)."""
        # Skip short words, numbers, noise
        if len(name) < 3:
            return False
        if name[0].isdigit():
            return False
        
        # Extensive noise word list
        noise_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                       'certain', 'attosecond', 'three', 'groups', 'doorway', 'journal',
                       'branch', 'foundation', 'identity', '1980s', 'succinic', 'shrimp',
                       'snappings', 'betweens', 'grappls', 'heightened', 'mobile',
                       'inorganic', 'precisely', 'adaptability', 'genes', 'density',
                       'frayn', 'manifold', 'divine', 'refusal', 'plates', 'occurs',
                       'today', 'early', 'solid', 'extensive', 'observable', 'necessary',
                       'volatile', 'darwin', 'foundational', 'classical', 'renewable',
                       'collaborative', 'decline', 'skill', 'entry', 'genomics',
                       'discoveries', 'commercializes', 'maintaines', 'accelerates',
                       'oxidizes', 'facilitates', 'extracts', 'attributes', 'confirms',
                       'indicates', 'underscores', 'bases', 'shows', 'varies', 'continues',
                       'provides', 'characterizes', 'classifies', 'states', 'emerges',
                       'divides', 'integrates', 'grappls', 'snappings'}
        if name.lower() in noise_words:
            return False
        
        if name not in self.knowledge.concepts:
            return False
        
        c = self.knowledge.concepts[name]
        if not c.is_content_word:
            return False
        
        # Must have meaningful actions (not just generic ones)
        if c.actions:
            good_actions = {'study', 'studies', 'examine', 'examines', 'investigate', 
                           'investigates', 'deduce', 'deduces', 'solve', 'solves',
                           'assist', 'assists', 'document', 'documents', 'support',
                           'supports', 'explore', 'explores', 'describe', 'describes',
                           'analyze', 'analyzes', 'observe', 'observes'}
            concept_actions = set(a for a, _ in c.actions.most_common(5))
            if concept_actions & good_actions:
                return True
        
        # Or must have high initiator count (well-attested)
        if c.initiator_count >= 10:
            return True
        
        return False
    
    def _generate_sentence(self, entity: str, role: str, actions: List[str], 
                          targets: List[str], is_first: bool = False) -> str:
        """Generate a single sentence about an entity."""
        entity_title = entity.title()
        
        # Sentence templates
        if is_first:
            templates = [
                "{entity} is {role} that {actions}",
                "{entity}, {role}, {actions}",
                "As {role}, {entity} {actions}",
            ]
        else:
            templates = [
                "{entity} {actions}",
                "{entity} is known to {actions_base}",
                "{entity} can be understood as {role} that {actions}",
                "{entity}, in this context, {actions}",
            ]
        
        template = random.choice(templates)
        
        # Format actions
        if actions:
            if len(actions) == 1:
                actions_str = actions[0]
                actions_base = actions[0].rstrip('s') if actions[0].endswith('s') else actions[0]
            elif len(actions) == 2:
                actions_str = f"{actions[0]} and {actions[1]}"
                actions_base = f"{actions[0].rstrip('s')} and {actions[1].rstrip('s')}"
            else:
                actions_str = f"{', '.join(actions[:-1])}, and {actions[-1]}"
                actions_base = actions_str
        else:
            actions_str = "exists"
            actions_base = "exist"
        
        # Format targets
        if targets:
            if len(targets) == 1:
                targets_str = targets[0]
            elif len(targets) == 2:
                targets_str = f"{targets[0]} and {targets[1]}"
            else:
                targets_str = f"{', '.join(targets[:-1])}, and {targets[-1]}"
            actions_str += f" {targets_str}"
        
        sentence = template.format(
            entity=entity_title,
            role=f"a {role}",
            actions=actions_str,
            actions_base=actions_base
        )
        
        # Ensure it ends with a period
        if not sentence.endswith('.'):
            sentence += '.'
        
        return sentence
    
    def _extract_entity(self, query: str) -> str:
        """Extract main entity from query."""
        import re
        query_lower = query.lower()
        
        patterns = [
            r'what (?:is|does|about) (\w+)',
            r'who is (\w+)',
            r'describe (\w+)',
            r'tell me about (\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: find content words
        words = query_lower.split()
        stop_words = {'what', 'is', 'does', 'do', 'the', 'a', 'an', 'who', 'how', 'why', 'about'}
        for word in words:
            if word not in stop_words and len(word) > 2:
                return word
        
        return ""
    
    def _get_role(self, concept) -> str:
        """Determine entity's role."""
        role_words = {'detective', 'doctor', 'scientist', 'science', 'field', 
                      'study', 'person', 'character', 'companion', 'assistant'}
        
        if concept.targets:
            for target, count in concept.targets.most_common(10):
                if target in role_words and count >= 3:
                    return target
        
        # Fallback based on phi-direction
        if concept.phi_direction > 0.3:
            return "figure"
        elif concept.phi_direction < -0.3:
            return "concept"
        else:
            return "entity"
    
    def _get_actions(self, concept) -> List[str]:
        """Get actions, filtering non-verbs."""
        skip_words = {'is', 'doctor', 'detective', 'science', 'field', 'cases', 
                      'case', 'holmes', 'watson', 'matter', 'energy', 'crimes',
                      'mysteries', 'evidence', 'identity', 'doorway', 'include',
                      'includes', 'journal', 'branch'}
        
        actions = []
        seen = set()  # Track normalized forms to avoid duplicates
        
        if concept.actions:
            for action, _ in concept.actions.most_common(8):
                if action in skip_words:
                    continue
                    
                # Normalize to 3rd person form
                normalized = action
                if not action.endswith('s'):
                    if action.endswith('y') and len(action) > 2 and action[-2] not in 'aeiou':
                        normalized = action[:-1] + 'ies'
                    elif action.endswith(('ch', 'sh', 'x', 'z', 'o')):
                        normalized = action + 'es'
                    else:
                        normalized = action + 's'
                
                # Skip if we've seen this normalized form
                if normalized in seen:
                    continue
                seen.add(normalized)
                
                actions.append(normalized)
                if len(actions) >= 4:
                    break
        
        return actions
    
    def _get_targets(self, concept) -> List[str]:
        """Get targets."""
        # Skip words that are roles, actions, or noise
        skip_words = {'is', 'science', 'field', 'study', 'certain', 'attosecond',
                      'detective', 'doctor', 'character', 'figure', 'concept',
                      'investigates', 'deduces', 'solves', 'assists', 'documents',
                      'studies', 'examines', 'includes', 'provides', 'states',
                      'supports', 'doorway', 'identity', 'foundation', 'three',
                      'journal', 'branch', 'watches'}
        
        targets = []
        if concept.targets:
            for target, _ in concept.targets.most_common(8):
                if target not in skip_words and target in self.knowledge.concepts:
                    if self.knowledge.concepts[target].is_content_word:
                        targets.append(target)
                        if len(targets) >= 4:
                            break
        
        return targets
    
    def _get_related(self, entity: str, concept) -> List[str]:
        """
        Find related concepts through resonance.
        
        This is the Karplus-Strong "delay line" - concepts that resonate
        with the main entity through shared connections.
        
        We look for concepts that are semantically related but not already
        mentioned as targets - these add depth without repetition.
        """
        # Skip words that shouldn't appear as related
        skip_words = {'is', 'science', 'field', 'study', 'certain', 'attosecond',
                      'detective', 'doctor', 'character', 'figure', 'concept',
                      'investigates', 'deduces', 'solves', 'assists', 'documents',
                      'studies', 'examines', 'includes', 'provides', 'states',
                      'supports', 'watches', 'journal', 'foundation', 'identity',
                      'three', 'groups', 'doorway', 'branch', entity}
        
        # Get targets we're already using (don't repeat them as related)
        used_targets = set(self._get_targets(concept))
        
        related = []
        
        # Look for concepts that share SPECIFIC actions (not generic ones)
        generic_actions = {'include', 'includes', 'provide', 'provides', 'state', 'states',
                          'is', 'has', 'have', 'do', 'does', 'make', 'makes'}
        
        if concept.actions:
            my_actions = set(a for a, _ in concept.actions.most_common(5))
            my_specific = my_actions - generic_actions
            
            if my_specific:
                for other_name, other in self.knowledge.concepts.items():
                    if other_name in skip_words or other_name in used_targets:
                        continue
                    if other_name == entity:
                        continue
                    if not other.is_content_word:
                        continue
                    
                    # Check if they share SPECIFIC actions
                    if other.actions:
                        other_actions = set(a for a, _ in other.actions.most_common(5))
                        other_specific = other_actions - generic_actions
                        shared = my_specific & other_specific
                        if len(shared) >= 2:
                            related.append(other_name)
                            if len(related) >= 3:
                                break
        
        return related


def demo():
    """Demonstrate the Karplus-Strong output lens."""
    print("=" * 70)
    print("KARPLUS-STRONG OUTPUT LENS DEMO")
    print("=" * 70)
    print()
    print("Generating multiple outputs for the same input to show variation:")
    print()
    
    lens = KarplusStrongLens()
    
    # Physics example
    print("PHYSICS (5 variations):")
    print("-" * 50)
    for i in range(5):
        output = lens.generate(
            entity="physics",
            role="science",
            actions=["studies", "examines", "investigates", "describes"],
            targets=["matter", "energy", "interactions", "forces"],
            related=["thermodynamics", "mechanics", "electromagnetism", "quantum theory"]
        )
        print(f"{i+1}. {output}")
    print()
    
    # Holmes example
    print("HOLMES (5 variations):")
    print("-" * 50)
    for i in range(5):
        output = lens.generate(
            entity="holmes",
            role="detective",
            actions=["investigates", "deduces", "solves", "observes"],
            targets=["crimes", "mysteries", "cases", "evidence"],
            related=["Watson", "London", "deduction", "observation"]
        )
        print(f"{i+1}. {output}")
    print()
    
    # Watson example
    print("WATSON (5 variations):")
    print("-" * 50)
    for i in range(5):
        output = lens.generate(
            entity="watson",
            role="doctor",
            actions=["assists", "documents", "supports", "accompanies"],
            targets=["Holmes", "cases", "investigations", "patients"],
            related=["medicine", "friendship", "narration", "loyalty"]
        )
        print(f"{i+1}. {output}")


def demo_with_corpus():
    """Demo using actual knowledge corpus."""
    print()
    print("=" * 70)
    print("KARPLUS-STRONG WITH REAL CORPUS")
    print("=" * 70)
    print()
    
    qa = KarplusStrongQA('truthspace_lcm/corpus_experimental.json')
    
    queries = [
        "What is physics?",
        "What does Holmes do?",
        "Tell me about Watson",
        "What is evolution?",
        "What does consciousness do?",
    ]
    
    for query in queries:
        print(f"Q: {query}")
        print("-" * 50)
        # Generate 3 variations
        for i in range(3):
            answer = qa.ask(query)
            print(f"  {i+1}. {answer}")
        print()


if __name__ == "__main__":
    demo()
    demo_with_corpus()
