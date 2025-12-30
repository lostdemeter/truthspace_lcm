"""
Corpus Reinforcer

Implements geometric reinforcement learning for corpus updates.
Instead of modifying existing frames, adds new "reinforcement frames"
to shift concept distributions.

Based on Design 073: Geometric Reinforcement Learning

Author: Lesley Gushurst
License: GPLv3
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ReinforcementResult:
    """Results from a reinforcement operation."""
    frames_added: List[Dict[str, Any]] = field(default_factory=list)
    concepts_reinforced: Dict[str, int] = field(default_factory=lambda: Counter())
    original_count: int = 0
    final_count: int = 0
    
    @property
    def added_count(self) -> int:
        return len(self.frames_added)
    
    def summary(self) -> str:
        lines = [
            f"Reinforcement Results:",
            f"  Original frames: {self.original_count}",
            f"  Frames added: {self.added_count}",
            f"  Final frames: {self.final_count}",
            f"  Concepts reinforced:"
        ]
        for concept, count in self.concepts_reinforced.most_common(10):
            lines.append(f"    - {concept}: +{count} frames")
        return "\n".join(lines)


class CorpusReinforcer:
    """
    Adds reinforcement frames to shift concept distributions.
    
    This implements additive learning - instead of modifying existing
    frames, we add new frames that reinforce the desired behavior.
    
    Usage:
        reinforcer = CorpusReinforcer()
        reinforcer.set_strength(10)  # 10 frames per reinforcement
        
        # Reinforce a correction
        reinforcer.reinforce(
            concept='evolution',
            role='concept',
            actions=['adapts', 'changes'],
            targets=['species', 'traits']
        )
        
        result = reinforcer.apply(corpus_data)
    """
    
    def __init__(self):
        self.strength = 10  # Number of frames to add per reinforcement
        self.pending_reinforcements: List[Dict[str, Any]] = []
        
        # Templates for generating frames
        self.templates = [
            "{concept} is a {role} that {action}.",
            "{concept} is a {role} who {action}.",
            "It appears that {concept} is a {role} that {action}.",
            "{concept} is a {role} that {action}, relating to {target}.",
            "{concept} is a {role} who {action}. This relates to {target}.",
        ]
    
    def set_strength(self, strength: int) -> 'CorpusReinforcer':
        """Set reinforcement strength (frames per reinforcement)."""
        self.strength = max(1, strength)
        return self
    
    def add_template(self, template: str) -> 'CorpusReinforcer':
        """Add a frame template."""
        self.templates.append(template)
        return self
    
    def reinforce(self, concept: str, role: str = None,
                  actions: List[str] = None, targets: List[str] = None,
                  strength: int = None) -> 'CorpusReinforcer':
        """
        Queue a reinforcement.
        
        Args:
            concept: The concept to reinforce
            role: Role to reinforce (e.g., 'concept', 'character')
            actions: Actions to reinforce
            targets: Targets to reinforce
            strength: Override default strength for this reinforcement
        """
        self.pending_reinforcements.append({
            'concept': concept,
            'role': role,
            'actions': actions or [],
            'targets': targets or [],
            'strength': strength or self.strength,
        })
        return self
    
    def reinforce_from_correction(self, original: str, corrected: str,
                                   strength: int = None) -> 'CorpusReinforcer':
        """
        Create reinforcement from a correction pair.
        
        Parses the corrected text to extract role, actions, targets.
        """
        # Parse corrected text
        concept = None
        role = None
        actions = []
        targets = []
        
        corrected_lower = corrected.lower()
        
        # Extract concept (first word)
        match = re.match(r'^(\w+)', corrected)
        if match:
            concept = match.group(1)
        
        # Extract role
        match = re.search(r'is a[n]? (\w+)', corrected_lower)
        if match:
            role = match.group(1)
        
        # Extract actions
        match = re.search(r'(?:who|that)\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', corrected_lower)
        if match:
            actions = [a for a in match.groups() if a]
        
        # Extract targets
        match = re.search(r'(?:relates? to|involving|particularly)\s+(\w+)(?:\s+and\s+(\w+))?', corrected_lower)
        if match:
            targets = [t for t in match.groups() if t]
        
        if concept:
            self.reinforce(concept, role, actions, targets, strength)
        
        return self
    
    def _generate_frames(self, reinforcement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate reinforcement frames."""
        frames = []
        concept = reinforcement['concept']
        role = reinforcement.get('role', 'entity')
        actions = reinforcement.get('actions', [])
        targets = reinforcement.get('targets', [])
        strength = reinforcement.get('strength', self.strength)
        
        # Generate frames using templates
        for i in range(strength):
            template_idx = i % len(self.templates)
            template = self.templates[template_idx]
            
            # Select action and target for this frame
            action = actions[i % len(actions)] if actions else 'exists'
            target = targets[i % len(targets)] if targets else ''
            
            # Generate text
            text = template.format(
                concept=concept.title(),
                role=role,
                action=action,
                target=target
            )
            
            # Clean up empty targets
            text = re.sub(r',?\s*relating to\s*\.', '.', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            frames.append({'text': text})
        
        return frames
    
    def apply(self, corpus_data: Dict[str, Any]) -> Tuple[Dict[str, Any], ReinforcementResult]:
        """
        Apply all pending reinforcements to corpus.
        
        Returns:
            Tuple of (reinforced corpus, result)
        """
        frames = list(corpus_data.get('frames', []))
        result = ReinforcementResult(original_count=len(frames))
        
        for reinforcement in self.pending_reinforcements:
            new_frames = self._generate_frames(reinforcement)
            frames.extend(new_frames)
            result.frames_added.extend(new_frames)
            result.concepts_reinforced[reinforcement['concept']] += len(new_frames)
        
        # Clear pending
        self.pending_reinforcements = []
        
        result.final_count = len(frames)
        
        # Create new corpus
        reinforced_corpus = corpus_data.copy()
        reinforced_corpus['frames'] = frames
        
        return reinforced_corpus, result
    
    def clear(self) -> 'CorpusReinforcer':
        """Clear pending reinforcements."""
        self.pending_reinforcements = []
        return self
