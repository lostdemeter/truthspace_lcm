#!/usr/bin/env python3
"""
Hypothesis Navigator: Geometric Navigation of Hypothesis Space

This module implements the insight that hypothesis formation creates a
navigable dimension - essentially attention in reverse (a "Tachyon").

FORWARD ATTENTION (standard LLM):
    Query → searches → Data → finds → Answer
    "Who is Holmes?" → attention → corpus → "detective"

REVERSE ATTENTION (hypothesis/Tachyon):
    Goal → Hypothesis → Evidence Search → Confirmation
    "Profile Holmes" → "is investigator" → find proof → confirm/refute

The key insight: A hypothesis is a TARGET POINT in concept space.
We're asking "can we reach this point from the data?"

This is bidirectional navigation:
- Forward: Data → Concept (extraction)
- Backward: Concept → Data (hypothesis testing)

The hypothesis dimension is navigable because:
1. Hypotheses have geometric structure (similar hypotheses are nearby)
2. Evidence creates paths between data and hypotheses
3. Confidence is distance - how far can we travel toward the hypothesis?

Connection to Tachyons:
- Normal causality: cause → effect (data → knowledge)
- Tachyon: effect → cause (hypothesis → evidence)
- Both are valid navigation directions in concept space
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
from enum import Enum
import math


# Hypothesis categories form a navigable space
# Each category is a dimension, each hypothesis is a point
HYPOTHESIS_DIMENSIONS = {
    'role': ['investigator', 'narrator', 'adventurer', 'romantic_figure', 
             'curious_observer', 'villain', 'mentor', 'protagonist'],
    'gender': ['male', 'female', 'unknown'],
    'agency': ['high', 'medium', 'low'],  # How much do they act vs are acted upon
    'social': ['leader', 'follower', 'loner', 'companion'],
}


@dataclass
class HypothesisPoint:
    """A point in hypothesis space."""
    entity: str
    category: str
    value: str
    confidence: float  # 0-1, represents "distance traveled" toward this point
    evidence_path: List[str] = field(default_factory=list)  # How we got here
    
    def to_vector(self) -> np.ndarray:
        """Convert to geometric representation."""
        # Each hypothesis is a point in a space defined by:
        # - Category (which dimension)
        # - Value (position along that dimension)
        # - Confidence (how "solid" the point is)
        
        vec = np.zeros(len(HYPOTHESIS_DIMENSIONS) + 1)
        
        # Find which dimension this hypothesis belongs to
        for i, (cat, values) in enumerate(HYPOTHESIS_DIMENSIONS.items()):
            if cat == self.category:
                if self.value in values:
                    # Position along this dimension
                    vec[i] = (values.index(self.value) + 1) / len(values)
                break
        
        # Confidence as final dimension
        vec[-1] = self.confidence
        
        return vec


@dataclass
class NavigationPath:
    """A path through hypothesis space."""
    start: str  # Starting hypothesis
    end: str    # Target hypothesis
    steps: List[Tuple[str, float]]  # (evidence, weight) pairs
    total_distance: float
    reachable: bool


class HypothesisNavigator:
    """
    Navigate hypothesis space geometrically.
    
    This implements "reverse attention" - instead of finding what the data
    says, we propose what we want to know and navigate toward it.
    
    The navigation is bidirectional:
    - Forward: What hypotheses does the data support?
    - Backward: What data would support this hypothesis?
    """
    
    def __init__(self, frames: List[Dict]):
        self.frames = frames
        self._build_indices()
        self._build_hypothesis_space()
    
    def _build_indices(self):
        """Build indices for efficient navigation."""
        self.entity_frames: Dict[str, List[Dict]] = {}
        self.action_counts: Dict[str, Counter] = {}
        self.patient_counts: Dict[str, Counter] = {}
        
        for frame in self.frames:
            agent = frame.get('agent', '').lower()
            if agent:
                if agent not in self.entity_frames:
                    self.entity_frames[agent] = []
                    self.action_counts[agent] = Counter()
                    self.patient_counts[agent] = Counter()
                
                self.entity_frames[agent].append(frame)
                
                action = frame.get('action')
                if action:
                    self.action_counts[agent][action] += 1
                
                patient = frame.get('patient', '').lower()
                if patient:
                    self.patient_counts[agent][patient] += 1
    
    def _build_hypothesis_space(self):
        """
        Build the hypothesis space as a geometric structure.
        
        Each entity creates a "cloud" of possible hypotheses.
        The data determines which hypotheses are reachable.
        """
        self.hypothesis_clouds: Dict[str, List[HypothesisPoint]] = {}
        
        for entity in self.entity_frames:
            self.hypothesis_clouds[entity] = self._generate_hypothesis_cloud(entity)
    
    def _generate_hypothesis_cloud(self, entity: str) -> List[HypothesisPoint]:
        """
        Generate all possible hypotheses for an entity.
        
        This is the "target space" - all the places we might want to navigate to.
        """
        cloud = []
        
        # Role hypotheses
        for role in HYPOTHESIS_DIMENSIONS['role']:
            point = HypothesisPoint(
                entity=entity,
                category='role',
                value=role,
                confidence=0.0  # Will be calculated by navigation
            )
            cloud.append(point)
        
        # Gender hypotheses
        for gender in HYPOTHESIS_DIMENSIONS['gender']:
            point = HypothesisPoint(
                entity=entity,
                category='gender',
                value=gender,
                confidence=0.0
            )
            cloud.append(point)
        
        # Agency hypotheses
        for agency in HYPOTHESIS_DIMENSIONS['agency']:
            point = HypothesisPoint(
                entity=entity,
                category='agency',
                value=agency,
                confidence=0.0
            )
            cloud.append(point)
        
        return cloud
    
    def navigate_to_hypothesis(self, entity: str, category: str, 
                                target_value: str) -> NavigationPath:
        """
        Navigate from data toward a specific hypothesis.
        
        This is the "reverse attention" operation - we have a destination
        (the hypothesis) and we're finding paths through the data to reach it.
        
        Returns a NavigationPath showing how (and if) we can reach the target.
        """
        if entity not in self.entity_frames:
            return NavigationPath(
                start="data",
                end=f"{entity} is {target_value}",
                steps=[],
                total_distance=float('inf'),
                reachable=False
            )
        
        # Get evidence generators for this hypothesis type
        evidence_fn = self._get_evidence_function(category, target_value)
        
        # Collect evidence (steps along the path)
        steps = evidence_fn(entity)
        
        # Calculate total distance traveled (sum of evidence weights)
        total_distance = sum(weight for _, weight in steps)
        
        # Determine if we reached the target (threshold)
        reachable = total_distance > 0.5
        
        return NavigationPath(
            start="data",
            end=f"{entity} is {target_value}",
            steps=steps,
            total_distance=total_distance,
            reachable=reachable
        )
    
    def _get_evidence_function(self, category: str, value: str):
        """Get the evidence-gathering function for a hypothesis type."""
        
        if category == 'role':
            return lambda e: self._gather_role_evidence(e, value)
        elif category == 'gender':
            return lambda e: self._gather_gender_evidence(e, value)
        elif category == 'agency':
            return lambda e: self._gather_agency_evidence(e, value)
        else:
            return lambda e: []
    
    def _gather_role_evidence(self, entity: str, role: str) -> List[Tuple[str, float]]:
        """
        Gather evidence for a role hypothesis.
        
        Each piece of evidence is a "step" toward the hypothesis.
        The weight is how much that step moves us toward the target.
        """
        evidence = []
        frames = self.entity_frames.get(entity, [])
        
        if not frames:
            return evidence
        
        total = len(frames)
        actions = self.action_counts.get(entity, Counter())
        patients = self.patient_counts.get(entity, Counter())
        
        if role == 'investigator':
            # Evidence: interaction with authority figures
            authority_patients = sum(patients.get(p, 0) for p in 
                                    ['inspector', 'police', 'lestrade', 'officer'])
            if authority_patients > 0:
                weight = min(0.4, authority_patients / total * 2)
                evidence.append((f"interacts with authorities ({authority_patients}x)", weight))
            
            # Evidence: case/crime words
            crime_words = 0
            for f in frames:
                text = f.get('text', '').lower()
                if any(w in text for w in ['case', 'crime', 'mystery', 'clue']):
                    crime_words += 1
            if crime_words > 0:
                weight = min(0.3, crime_words / total)
                evidence.append((f"crime-related context ({crime_words}x)", weight))
            
            # Evidence: PERCEIVE + THINK actions
            perceive_think = actions.get('PERCEIVE', 0) + actions.get('THINK', 0)
            if perceive_think > total * 0.1:
                weight = min(0.3, perceive_think / total)
                evidence.append((f"observes and thinks ({perceive_think}x)", weight))
        
        elif role == 'narrator':
            # Evidence: very high SPEAK
            speak = actions.get('SPEAK', 0)
            if speak > total * 0.3:
                weight = min(0.4, speak / total)
                evidence.append((f"speaks frequently ({speak}/{total})", weight))
            
            # Evidence: focuses on one main character
            if patients:
                top_patient, top_count = patients.most_common(1)[0]
                if top_count > len(list(patients.elements())) * 0.15:
                    evidence.append((f"focuses on {top_patient}", 0.3))
        
        elif role == 'adventurer':
            # Evidence: high MOVE
            move = actions.get('MOVE', 0)
            if move > total * 0.15:
                weight = min(0.3, move / total)
                evidence.append((f"moves frequently ({move}/{total})", weight))
            
            # Evidence: family/friend patients
            family_patients = sum(patients.get(p, 0) for p in 
                                 ['aunt', 'uncle', 'friend', 'boy', 'girl'])
            if family_patients > 0:
                weight = min(0.3, family_patients / total * 2)
                evidence.append((f"interacts with family/friends ({family_patients}x)", weight))
            
            # Negative evidence: NO authority figures
            authority = sum(patients.get(p, 0) for p in ['inspector', 'police', 'officer'])
            if authority == 0:
                evidence.append(("no authority interactions", 0.2))
        
        elif role == 'curious_observer':
            # Evidence: very high PERCEIVE + THINK
            perceive_think = actions.get('PERCEIVE', 0) + actions.get('THINK', 0)
            if perceive_think > total * 0.35:
                weight = min(0.4, perceive_think / total)
                evidence.append((f"highly observant ({perceive_think}/{total})", weight))
            
            # Evidence: unusual patients
            unusual = sum(patients.get(p, 0) for p in 
                         ['creature', 'queen', 'king', 'rabbit', 'cat'])
            if unusual > 0:
                weight = min(0.4, unusual / total * 3)
                evidence.append((f"encounters unusual entities ({unusual}x)", weight))
        
        elif role == 'romantic_figure':
            # Evidence: high EXIST (heavily described)
            exist = actions.get('EXIST', 0)
            if exist > total * 0.25:
                weight = min(0.4, exist / total)
                evidence.append((f"heavily described ({exist}/{total})", weight))
            
            # Evidence: low MOVE
            move = actions.get('MOVE', 0)
            if move < total * 0.1:
                evidence.append(("not an active mover", 0.2))
        
        return evidence
    
    def _gather_gender_evidence(self, entity: str, gender: str) -> List[Tuple[str, float]]:
        """Gather evidence for gender hypothesis."""
        evidence = []
        frames = self.entity_frames.get(entity, [])
        
        if not frames:
            return evidence
        
        # Count pronouns in text
        male_pronouns = 0
        female_pronouns = 0
        
        for f in frames:
            text = f.get('text', '').lower()
            words = set(text.split())
            
            if any(p in words for p in ['he', 'him', 'his']):
                male_pronouns += 1
            if any(p in words for p in ['she', 'her', 'hers']):
                female_pronouns += 1
        
        total_pronouns = male_pronouns + female_pronouns
        
        if gender == 'male' and total_pronouns > 0:
            ratio = male_pronouns / total_pronouns
            if ratio > 0.6:
                evidence.append((f"male pronouns ({male_pronouns}/{total_pronouns})", ratio * 0.5))
        
        elif gender == 'female' and total_pronouns > 0:
            ratio = female_pronouns / total_pronouns
            if ratio > 0.6:
                evidence.append((f"female pronouns ({female_pronouns}/{total_pronouns})", ratio * 0.5))
        
        # Check for titles
        for f in frames:
            text = f.get('text', '').lower()
            if gender == 'male':
                if f"mr. {entity}" in text or f"mr {entity}" in text:
                    evidence.append(("title: Mr.", 0.3))
                    break
            elif gender == 'female':
                if any(f"{t}. {entity}" in text or f"{t} {entity}" in text 
                       for t in ['miss', 'mrs', 'lady']):
                    evidence.append(("title: Miss/Mrs/Lady", 0.3))
                    break
        
        return evidence
    
    def _gather_agency_evidence(self, entity: str, agency: str) -> List[Tuple[str, float]]:
        """Gather evidence for agency hypothesis."""
        evidence = []
        
        agent_count = len(self.entity_frames.get(entity, []))
        
        # Count times entity is patient (acted upon)
        patient_count = sum(
            1 for f in self.frames 
            if f.get('patient', '').lower() == entity
        )
        
        total = agent_count + patient_count
        if total == 0:
            return evidence
        
        agent_ratio = agent_count / total
        
        if agency == 'high' and agent_ratio > 0.7:
            evidence.append((f"high agency ({agent_ratio:.0%} as agent)", agent_ratio * 0.5))
        elif agency == 'medium' and 0.4 <= agent_ratio <= 0.7:
            evidence.append((f"balanced agency ({agent_ratio:.0%} as agent)", 0.4))
        elif agency == 'low' and agent_ratio < 0.4:
            evidence.append((f"low agency ({agent_ratio:.0%} as agent)", (1 - agent_ratio) * 0.5))
        
        return evidence
    
    def find_best_hypothesis(self, entity: str, category: str) -> Tuple[str, float, List[str]]:
        """
        Find the best hypothesis for an entity in a category.
        
        This navigates to ALL hypotheses in the category and returns
        the one we can reach most confidently.
        """
        best_value = None
        best_distance = 0
        best_evidence = []
        
        for value in HYPOTHESIS_DIMENSIONS.get(category, []):
            path = self.navigate_to_hypothesis(entity, category, value)
            
            if path.total_distance > best_distance:
                best_distance = path.total_distance
                best_value = value
                best_evidence = [step for step, _ in path.steps]
        
        return best_value, best_distance, best_evidence
    
    def profile_entity(self, entity: str) -> Dict[str, Tuple[str, float, List[str]]]:
        """
        Build a complete profile by navigating to best hypothesis in each category.
        """
        profile = {}
        
        for category in HYPOTHESIS_DIMENSIONS:
            value, confidence, evidence = self.find_best_hypothesis(entity, category)
            profile[category] = (value, confidence, evidence)
        
        return profile
    
    def explain_navigation(self, entity: str) -> str:
        """Generate a natural language explanation of the navigation."""
        profile = self.profile_entity(entity)
        
        lines = [f"Navigation to {entity.title()}:", ""]
        
        for category, (value, confidence, evidence) in profile.items():
            status = "✓" if confidence > 0.5 else "?" if confidence > 0.2 else "✗"
            lines.append(f"{status} {category}: {value} (confidence: {confidence:.2f})")
            
            for e in evidence[:3]:
                lines.append(f"    → {e}")
            lines.append("")
        
        return "\n".join(lines)


def navigate_hypothesis_space(entity: str, frames: List[Dict]) -> Dict:
    """Convenience function to navigate hypothesis space for an entity."""
    navigator = HypothesisNavigator(frames)
    return navigator.profile_entity(entity)


class BidirectionalReasoner:
    """
    Combines forward and backward navigation for iterative refinement.
    
    The key insight: disagreement between forward (data → features) and
    backward (hypothesis → evidence) navigation reveals what we need to learn.
    
    This implements the Tachyon principle: information flows both ways
    through concept space, and the intersection is knowledge.
    """
    
    def __init__(self, frames: List[Dict]):
        self.frames = frames
        self.navigator = HypothesisNavigator(frames)
        self.learned_patterns: Dict[str, List[str]] = {}  # role → distinguishing features
    
    def reason_about_entity(self, entity: str) -> Dict:
        """
        Apply bidirectional reasoning to understand an entity.
        
        Returns a refined understanding based on forward-backward convergence.
        """
        entity = entity.lower()
        
        # Step 1: Forward pass
        forward_features = self._forward_pass(entity)
        
        # Step 2: Backward pass
        backward_hypotheses = self._backward_pass(entity)
        
        # Step 3: Find convergence/divergence
        analysis = self._analyze_convergence(entity, forward_features, backward_hypotheses)
        
        # Step 4: Refine based on divergence
        refined = self._refine_understanding(entity, analysis)
        
        return {
            'entity': entity,
            'forward': forward_features,
            'backward': backward_hypotheses,
            'analysis': analysis,
            'refined': refined,
        }
    
    def _forward_pass(self, entity: str) -> Dict:
        """Extract features from data (standard attention direction)."""
        actions = self.navigator.action_counts.get(entity, Counter())
        patients = self.navigator.patient_counts.get(entity, Counter())
        total = sum(actions.values())
        
        if total == 0:
            return {'actions': {}, 'patients': {}, 'top_action': None}
        
        action_pcts = {a: c/total for a, c in actions.items()}
        
        return {
            'actions': action_pcts,
            'patients': dict(patients.most_common(5)),
            'top_action': actions.most_common(1)[0][0] if actions else None,
            'total_frames': total,
        }
    
    def _backward_pass(self, entity: str) -> Dict:
        """Test hypotheses (Tachyon direction)."""
        results = {}
        
        for role in HYPOTHESIS_DIMENSIONS['role']:
            path = self.navigator.navigate_to_hypothesis(entity, 'role', role)
            results[role] = {
                'distance': path.total_distance,
                'reachable': path.reachable,
                'evidence': [step for step, _ in path.steps],
            }
        
        # Find best hypothesis
        best_role = max(results.keys(), key=lambda r: results[r]['distance'])
        
        return {
            'hypotheses': results,
            'best': best_role,
            'best_distance': results[best_role]['distance'],
        }
    
    def _analyze_convergence(self, entity: str, forward: Dict, backward: Dict) -> Dict:
        """Analyze where forward and backward agree/disagree."""
        top_action = forward.get('top_action')
        best_hypothesis = backward.get('best')
        
        # Expected action-role mappings
        action_role_map = {
            'SPEAK': ['narrator', 'investigator'],
            'PERCEIVE': ['investigator', 'curious_observer'],
            'THINK': ['investigator', 'curious_observer'],
            'MOVE': ['adventurer'],
            'EXIST': ['romantic_figure'],
            'ACT': ['adventurer', 'protagonist'],
        }
        
        expected_roles = action_role_map.get(top_action, [])
        converges = best_hypothesis in expected_roles
        
        # What distinguishes the best hypothesis?
        best_evidence = backward['hypotheses'][best_hypothesis]['evidence']
        
        return {
            'converges': converges,
            'top_action': top_action,
            'best_hypothesis': best_hypothesis,
            'expected_from_action': expected_roles,
            'distinguishing_evidence': best_evidence,
            'insight': self._generate_insight(top_action, best_hypothesis, converges, best_evidence),
        }
    
    def _generate_insight(self, action: str, hypothesis: str, converges: bool, evidence: List[str]) -> str:
        """Generate a natural language insight from the analysis."""
        if converges:
            return f"{action} action aligns with {hypothesis} role"
        else:
            if evidence:
                return f"{action} alone doesn't indicate {hypothesis}; key evidence: {evidence[0]}"
            return f"{action} and {hypothesis} don't obviously align - needs investigation"
    
    def _refine_understanding(self, entity: str, analysis: Dict) -> Dict:
        """Refine understanding based on convergence analysis."""
        hypothesis = analysis['best_hypothesis']
        evidence = analysis['distinguishing_evidence']
        
        # The refined understanding combines action + distinguishing feature
        refined_pattern = {
            'role': hypothesis,
            'confidence': 'high' if analysis['converges'] else 'medium',
            'key_features': evidence[:2] if evidence else [],
            'reasoning': analysis['insight'],
        }
        
        # Learn this pattern for future use
        if hypothesis not in self.learned_patterns:
            self.learned_patterns[hypothesis] = []
        for e in evidence:
            if e not in self.learned_patterns[hypothesis]:
                self.learned_patterns[hypothesis].append(e)
        
        return refined_pattern
    
    def explain(self, entity: str) -> str:
        """Generate a natural language explanation of the reasoning."""
        result = self.reason_about_entity(entity)
        
        lines = [
            f"Bidirectional Reasoning for {entity.title()}",
            "=" * 50,
            "",
            "FORWARD (Data → Features):",
            f"  Top action: {result['forward'].get('top_action')}",
            f"  Top patients: {list(result['forward'].get('patients', {}).keys())[:3]}",
            "",
            "BACKWARD (Hypothesis → Evidence):",
            f"  Best hypothesis: {result['backward']['best']}",
            f"  Distance: {result['backward']['best_distance']:.2f}",
            f"  Evidence: {result['backward']['hypotheses'][result['backward']['best']]['evidence'][:2]}",
            "",
            "CONVERGENCE:",
            f"  {result['analysis']['insight']}",
            "",
            "REFINED UNDERSTANDING:",
            f"  Role: {result['refined']['role']}",
            f"  Confidence: {result['refined']['confidence']}",
            f"  Key features: {result['refined']['key_features']}",
        ]
        
        return "\n".join(lines)
