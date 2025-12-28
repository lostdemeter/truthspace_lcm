#!/usr/bin/env python3
"""
Hypothesis-Driven Entity Profiler for GeometricLCM

Instead of passively extracting features, this module uses goal-directed
knowledge acquisition based on Pólya's problem-solving method:

1. UNDERSTAND: What do we want to know about this entity?
2. PLAN: What evidence would confirm/refute our hypotheses?
3. EXECUTE: Search the corpus for that specific evidence
4. REFLECT: Did we find enough? Refine and repeat if needed.

This is the scientific method applied to knowledge:
- Hypothesis → Prediction → Experiment → Conclusion

Key insight: We know what we're looking for BEFORE we look.
Instead of "what can we extract?" we ask "is X true? What would prove it?"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Callable
from collections import Counter
from enum import Enum
import re


class Confidence(Enum):
    """Confidence level in a hypothesis."""
    UNKNOWN = 0      # No evidence either way
    LOW = 1          # Some evidence, but weak
    MEDIUM = 2       # Moderate evidence
    HIGH = 3         # Strong evidence
    CONFIRMED = 4    # Overwhelming evidence
    REFUTED = -1     # Evidence contradicts hypothesis


@dataclass
class Prediction:
    """A testable prediction derived from a hypothesis."""
    description: str
    test_function: str  # Name of test method to call
    test_params: Dict   # Parameters for the test
    weight: float = 1.0  # How important is this prediction?
    
    # Results after testing
    result: Optional[bool] = None
    evidence: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-1 score from test


@dataclass
class Hypothesis:
    """A hypothesis about an entity that can be tested."""
    claim: str              # e.g., "Holmes is a detective"
    category: str           # e.g., "role", "gender", "trait"
    predictions: List[Prediction] = field(default_factory=list)
    
    # Results after testing
    confidence: Confidence = Confidence.UNKNOWN
    total_score: float = 0.0
    evidence_summary: str = ""
    
    def add_prediction(self, description: str, test_function: str, 
                       test_params: Dict, weight: float = 1.0):
        """Add a testable prediction to this hypothesis."""
        self.predictions.append(Prediction(
            description=description,
            test_function=test_function,
            test_params=test_params,
            weight=weight
        ))
    
    def calculate_confidence(self):
        """Calculate overall confidence from prediction results."""
        if not self.predictions:
            self.confidence = Confidence.UNKNOWN
            return
        
        total_weight = sum(p.weight for p in self.predictions)
        weighted_score = sum(p.score * p.weight for p in self.predictions)
        
        self.total_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Map score to confidence
        if self.total_score >= 0.8:
            self.confidence = Confidence.CONFIRMED
        elif self.total_score >= 0.6:
            self.confidence = Confidence.HIGH
        elif self.total_score >= 0.4:
            self.confidence = Confidence.MEDIUM
        elif self.total_score >= 0.2:
            self.confidence = Confidence.LOW
        elif self.total_score < 0.1 and any(p.result == False for p in self.predictions):
            self.confidence = Confidence.REFUTED
        else:
            self.confidence = Confidence.UNKNOWN


@dataclass
class EntityProfile:
    """Profile built from confirmed hypotheses."""
    entity: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    
    # Confirmed attributes
    role: Optional[str] = None
    gender: Optional[str] = None
    traits: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    
    def get_confirmed_claims(self) -> List[str]:
        """Get all confirmed or high-confidence claims."""
        return [
            h.claim for h in self.hypotheses 
            if h.confidence in (Confidence.CONFIRMED, Confidence.HIGH)
        ]


class HypothesisProfiler:
    """
    Goal-directed entity profiler using hypothesis testing.
    
    Instead of extracting features, we:
    1. Generate hypotheses about what the entity might be
    2. Define predictions that would confirm/refute each hypothesis
    3. Test predictions against the corpus
    4. Accept/reject hypotheses based on evidence
    """
    
    def __init__(self, frames: List[Dict]):
        """
        Args:
            frames: List of concept frames from corpus
        """
        self.frames = frames
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for efficient evidence lookup."""
        # Index frames by agent
        self.agent_frames: Dict[str, List[Dict]] = {}
        # Index frames by patient
        self.patient_frames: Dict[str, List[Dict]] = {}
        # Index frames by text content (for word search)
        self.text_index: Dict[str, List[Dict]] = {}
        
        for frame in self.frames:
            agent = frame.get('agent', '').lower()
            patient = frame.get('patient', '').lower()
            
            if agent:
                if agent not in self.agent_frames:
                    self.agent_frames[agent] = []
                self.agent_frames[agent].append(frame)
            
            if patient:
                if patient not in self.patient_frames:
                    self.patient_frames[patient] = []
                self.patient_frames[patient].append(frame)
    
    def profile_entity(self, entity: str) -> EntityProfile:
        """
        Build a profile for an entity using hypothesis testing.
        
        This is the main entry point - it orchestrates the full
        Pólya/scientific method process.
        """
        entity_lower = entity.lower()
        profile = EntityProfile(entity=entity)
        
        # STEP 1: UNDERSTAND - What do we want to know?
        # Generate hypotheses based on what we observe
        hypotheses = self._generate_hypotheses(entity_lower)
        
        # STEP 2 & 3: PLAN & EXECUTE - Test each hypothesis
        for hypothesis in hypotheses:
            self._test_hypothesis(entity_lower, hypothesis)
            profile.hypotheses.append(hypothesis)
        
        # STEP 4: REFLECT - Build profile from confirmed hypotheses
        self._build_profile_from_hypotheses(profile)
        
        return profile
    
    def _generate_hypotheses(self, entity: str) -> List[Hypothesis]:
        """
        Generate hypotheses about an entity based on initial observations.
        
        This is where we apply domain knowledge about what kinds of
        things entities can be.
        """
        hypotheses = []
        
        # Get basic stats about the entity
        agent_count = len(self.agent_frames.get(entity, []))
        patient_count = len(self.patient_frames.get(entity, []))
        
        # ROLE HYPOTHESES
        # If entity is frequently an agent, hypothesize about their role
        if agent_count >= 10:
            # Hypothesis: Entity is an investigator/detective
            # Key insight: investigators interact with crime-related entities
            h = Hypothesis(
                claim=f"{entity} is an investigator",
                category="role"
            )
            h.add_prediction(
                "Should co-occur with investigation-related words",
                "test_word_cooccurrence",
                {"words": ["case", "crime", "clue", "mystery", "evidence", "investigate"]},
                weight=2.0
            )
            h.add_prediction(
                "Should interact with authority figures (inspector, police)",
                "test_patient_types",
                {"patient_words": ["inspector", "police", "officer", "constable", "lestrade"]},
                weight=2.5  # Strong distinguisher!
            )
            h.add_prediction(
                "Should have moderate-high PERCEIVE action rate",
                "test_action_rate",
                {"action": "PERCEIVE", "min_rate": 0.08},
                weight=1.0
            )
            hypotheses.append(h)
            
            # Hypothesis: Entity is a narrator/companion
            # Key insight: narrators act upon the protagonist
            h = Hypothesis(
                claim=f"{entity} is a narrator",
                category="role"
            )
            h.add_prediction(
                "Should have very high SPEAK action rate",
                "test_action_rate",
                {"action": "SPEAK", "min_rate": 0.30},
                weight=2.0
            )
            h.add_prediction(
                "Should interact with a single main character frequently",
                "test_patient_concentration",
                {"min_concentration": 0.15},  # Top patient is >15% of all patients
                weight=2.0
            )
            hypotheses.append(h)
            
            # Hypothesis: Entity is an adventurer
            # Key insight: adventurers interact with family/friends, not authorities
            h = Hypothesis(
                claim=f"{entity} is an adventurer",
                category="role"
            )
            h.add_prediction(
                "Should have high MOVE action rate",
                "test_action_rate",
                {"action": "MOVE", "min_rate": 0.15},
                weight=2.0
            )
            h.add_prediction(
                "Should interact with family/friends",
                "test_patient_types",
                {"patient_words": ["aunt", "uncle", "friend", "boy", "girl", "mother", "father"]},
                weight=2.0
            )
            h.add_prediction(
                "Should NOT interact with authority figures",
                "test_patient_types_negative",
                {"patient_words": ["inspector", "police", "officer", "detective"]},
                weight=1.5
            )
            hypotheses.append(h)
            
            # Hypothesis: Entity is a curious observer (like Alice)
            # Key insight: interacts with strange/fantasy entities
            h = Hypothesis(
                claim=f"{entity} is a curious observer",
                category="role"
            )
            h.add_prediction(
                "Should have very high PERCEIVE + THINK rate",
                "test_combined_action_rate",
                {"actions": ["PERCEIVE", "THINK"], "min_rate": 0.35},
                weight=2.0
            )
            h.add_prediction(
                "Should interact with unusual entities",
                "test_patient_types",
                {"patient_words": ["creature", "queen", "king", "rabbit", "cat", "wonder"]},
                weight=2.0
            )
            hypotheses.append(h)
            
            # Hypothesis: Entity is a romantic figure (like Darcy)
            # Key insight: high EXIST (described), interacts with love interests
            h = Hypothesis(
                claim=f"{entity} is a romantic figure",
                category="role"
            )
            h.add_prediction(
                "Should have very high EXIST rate (heavily described)",
                "test_action_rate",
                {"action": "EXIST", "min_rate": 0.25},
                weight=2.5
            )
            h.add_prediction(
                "Should have low MOVE rate (not an adventurer)",
                "test_action_rate_max",
                {"action": "MOVE", "max_rate": 0.10},
                weight=1.5
            )
            hypotheses.append(h)
        
        # GENDER HYPOTHESES
        h = Hypothesis(
            claim=f"{entity} is male",
            category="gender"
        )
        h.add_prediction(
            "Should be referred to with male pronouns",
            "test_pronoun_cooccurrence",
            {"pronouns": ["he", "him", "his"], "anti_pronouns": ["she", "her"]},
            weight=2.0
        )
        h.add_prediction(
            "Should have male title (Mr., Sir, Lord)",
            "test_title_cooccurrence",
            {"titles": ["mr", "sir", "lord"], "anti_titles": ["miss", "mrs", "lady"]},
            weight=1.5
        )
        hypotheses.append(h)
        
        h = Hypothesis(
            claim=f"{entity} is female",
            category="gender"
        )
        h.add_prediction(
            "Should be referred to with female pronouns",
            "test_pronoun_cooccurrence",
            {"pronouns": ["she", "her", "hers"], "anti_pronouns": ["he", "him", "his"]},
            weight=2.0
        )
        h.add_prediction(
            "Should have female title (Miss, Mrs, Lady)",
            "test_title_cooccurrence",
            {"titles": ["miss", "mrs", "lady", "madam"], "anti_titles": ["mr", "sir", "lord"]},
            weight=1.5
        )
        hypotheses.append(h)
        
        return hypotheses
    
    def _test_hypothesis(self, entity: str, hypothesis: Hypothesis):
        """Test all predictions for a hypothesis."""
        for prediction in hypothesis.predictions:
            test_method = getattr(self, prediction.test_function, None)
            if test_method:
                score, evidence = test_method(entity, **prediction.test_params)
                prediction.score = score
                prediction.result = score > 0.5
                prediction.evidence = evidence
        
        hypothesis.calculate_confidence()
    
    # ==========================================================================
    # TEST METHODS - Each tests a specific prediction
    # ==========================================================================
    
    def test_word_cooccurrence(self, entity: str, words: List[str]) -> Tuple[float, List[str]]:
        """
        Test if entity co-occurs with specific words.
        
        Returns (score, evidence) where score is 0-1.
        """
        frames = self.agent_frames.get(entity, []) + self.patient_frames.get(entity, [])
        if not frames:
            return 0.0, ["No frames found for entity"]
        
        word_counts = Counter()
        for frame in frames:
            text = frame.get('text', '').lower()
            for word in words:
                if word in text:
                    word_counts[word] += 1
        
        total_matches = sum(word_counts.values())
        evidence = [f"'{w}' appears {c} times" for w, c in word_counts.most_common(3)]
        
        # Score based on how many target words appear and how often
        words_found = len([w for w in words if word_counts[w] > 0])
        word_ratio = words_found / len(words)
        frequency_score = min(1.0, total_matches / (len(frames) * 0.1))
        
        score = (word_ratio * 0.5 + frequency_score * 0.5)
        
        if not evidence:
            evidence = ["No target words found"]
        
        return score, evidence
    
    def test_action_rate(self, entity: str, action: str, min_rate: float) -> Tuple[float, List[str]]:
        """
        Test if entity performs a specific action at a minimum rate.
        """
        frames = self.agent_frames.get(entity, [])
        if len(frames) < 10:
            return 0.0, [f"Insufficient frames ({len(frames)})"]
        
        action_count = sum(1 for f in frames if f.get('action') == action)
        actual_rate = action_count / len(frames)
        
        evidence = [f"{action} rate: {actual_rate:.1%} (threshold: {min_rate:.1%})"]
        
        if actual_rate >= min_rate:
            # Score increases with how much we exceed the threshold
            score = min(1.0, 0.5 + (actual_rate - min_rate) / min_rate * 0.5)
        else:
            # Score decreases with how far below threshold
            score = max(0.0, actual_rate / min_rate * 0.5)
        
        return score, evidence
    
    def test_pronoun_cooccurrence(self, entity: str, pronouns: List[str], 
                                   anti_pronouns: List[str]) -> Tuple[float, List[str]]:
        """
        Test if entity co-occurs with specific pronouns (and not others).
        """
        frames = self.agent_frames.get(entity, []) + self.patient_frames.get(entity, [])
        if not frames:
            return 0.0, ["No frames found"]
        
        pro_count = 0
        anti_count = 0
        
        for frame in frames:
            text = frame.get('text', '').lower()
            # Look for pronouns in same sentence
            words = set(re.findall(r'\b\w+\b', text))
            
            for p in pronouns:
                if p in words:
                    pro_count += 1
                    break
            
            for p in anti_pronouns:
                if p in words:
                    anti_count += 1
                    break
        
        evidence = [
            f"Target pronouns: {pro_count} frames",
            f"Anti pronouns: {anti_count} frames"
        ]
        
        total = pro_count + anti_count
        if total == 0:
            return 0.5, evidence + ["No pronoun evidence found"]
        
        score = pro_count / total
        return score, evidence
    
    def test_title_cooccurrence(self, entity: str, titles: List[str],
                                 anti_titles: List[str]) -> Tuple[float, List[str]]:
        """
        Test if entity co-occurs with specific titles.
        """
        frames = self.agent_frames.get(entity, []) + self.patient_frames.get(entity, [])
        if not frames:
            return 0.0, ["No frames found"]
        
        title_count = 0
        anti_count = 0
        
        for frame in frames:
            text = frame.get('text', '').lower()
            
            for t in titles:
                # Look for "Mr. Holmes" or "Mr Holmes" patterns
                if f"{t}. {entity}" in text or f"{t} {entity}" in text:
                    title_count += 1
                    break
            
            for t in anti_titles:
                if f"{t}. {entity}" in text or f"{t} {entity}" in text:
                    anti_count += 1
                    break
        
        evidence = [
            f"Target titles: {title_count} frames",
            f"Anti titles: {anti_count} frames"
        ]
        
        total = title_count + anti_count
        if total == 0:
            return 0.5, evidence + ["No title evidence found"]
        
        score = title_count / total
        return score, evidence
    
    def test_patient_types(self, entity: str, patient_words: List[str]) -> Tuple[float, List[str]]:
        """
        Test if entity's patients include specific types of entities.
        This is a key distinguisher - WHO you interact with reveals your role.
        """
        frames = self.agent_frames.get(entity, [])
        if not frames:
            return 0.0, ["No frames found"]
        
        # Get all patients
        patients = [f.get('patient', '').lower() for f in frames if f.get('patient')]
        if not patients:
            return 0.0, ["No patients found"]
        
        # Count matches
        matches = 0
        matched_words = []
        for patient in patients:
            for word in patient_words:
                if word in patient:
                    matches += 1
                    if word not in matched_words:
                        matched_words.append(word)
                    break
        
        evidence = [
            f"Matching patients: {matches}/{len(patients)}",
            f"Found: {matched_words[:3]}" if matched_words else "No matches"
        ]
        
        # Score based on presence (any match is significant)
        if matches > 0:
            score = min(1.0, 0.5 + matches / len(patients))
        else:
            score = 0.0
        
        return score, evidence
    
    def test_patient_types_negative(self, entity: str, patient_words: List[str]) -> Tuple[float, List[str]]:
        """
        Test that entity does NOT interact with certain patient types.
        Returns high score if NO matches found.
        """
        frames = self.agent_frames.get(entity, [])
        if not frames:
            return 0.5, ["No frames found"]
        
        patients = [f.get('patient', '').lower() for f in frames if f.get('patient')]
        if not patients:
            return 0.5, ["No patients found"]
        
        # Count matches (we want zero)
        matches = 0
        for patient in patients:
            for word in patient_words:
                if word in patient:
                    matches += 1
                    break
        
        evidence = [f"Unwanted patient matches: {matches}/{len(patients)}"]
        
        # High score if no matches, but even a few matches is significant
        # Use a stricter threshold - any matches above 2% should penalize heavily
        if matches == 0:
            score = 1.0
        elif matches <= 1:
            score = 0.7
        elif matches <= 3:
            score = 0.4
        else:
            # 4+ matches = strong evidence against this hypothesis
            score = max(0.0, 0.2 - matches / len(patients))
        
        return score, evidence
    
    def test_patient_concentration(self, entity: str, min_concentration: float) -> Tuple[float, List[str]]:
        """
        Test if entity focuses on a single main patient (narrator pattern).
        Narrators typically interact heavily with one protagonist.
        """
        frames = self.agent_frames.get(entity, [])
        if not frames:
            return 0.0, ["No frames found"]
        
        patients = [f.get('patient', '').lower() for f in frames if f.get('patient')]
        if not patients:
            return 0.0, ["No patients found"]
        
        from collections import Counter
        patient_counts = Counter(patients)
        top_patient, top_count = patient_counts.most_common(1)[0]
        concentration = top_count / len(patients)
        
        evidence = [
            f"Top patient: {top_patient} ({top_count}/{len(patients)} = {concentration:.1%})",
            f"Threshold: {min_concentration:.1%}"
        ]
        
        if concentration >= min_concentration:
            score = min(1.0, 0.5 + (concentration - min_concentration) / min_concentration)
        else:
            score = concentration / min_concentration * 0.5
        
        return score, evidence
    
    def test_action_rate_max(self, entity: str, action: str, max_rate: float) -> Tuple[float, List[str]]:
        """
        Test if entity performs a specific action BELOW a maximum rate.
        Used for negative predictions (e.g., "should NOT move much").
        """
        frames = self.agent_frames.get(entity, [])
        if len(frames) < 10:
            return 0.5, [f"Insufficient frames ({len(frames)})"]
        
        action_count = sum(1 for f in frames if f.get('action') == action)
        actual_rate = action_count / len(frames)
        
        evidence = [f"{action} rate: {actual_rate:.1%} (max threshold: {max_rate:.1%})"]
        
        if actual_rate <= max_rate:
            # Score increases with how far below the threshold
            score = min(1.0, 0.5 + (max_rate - actual_rate) / max_rate * 0.5)
        else:
            # Score decreases with how far above threshold
            score = max(0.0, 0.5 - (actual_rate - max_rate) / max_rate * 0.5)
        
        return score, evidence
    
    def test_combined_action_rate(self, entity: str, actions: List[str], min_rate: float) -> Tuple[float, List[str]]:
        """
        Test if entity performs a combination of actions at a minimum combined rate.
        """
        frames = self.agent_frames.get(entity, [])
        if len(frames) < 10:
            return 0.0, [f"Insufficient frames ({len(frames)})"]
        
        combined_count = sum(1 for f in frames if f.get('action') in actions)
        actual_rate = combined_count / len(frames)
        
        evidence = [
            f"Combined {'+'.join(actions)} rate: {actual_rate:.1%}",
            f"Threshold: {min_rate:.1%}"
        ]
        
        if actual_rate >= min_rate:
            score = min(1.0, 0.5 + (actual_rate - min_rate) / min_rate * 0.5)
        else:
            score = max(0.0, actual_rate / min_rate * 0.5)
        
        return score, evidence
    
    def _build_profile_from_hypotheses(self, profile: EntityProfile):
        """Build final profile from confirmed hypotheses."""
        # Find best role hypothesis
        role_hypotheses = [h for h in profile.hypotheses if h.category == "role"]
        if role_hypotheses:
            best_role = max(role_hypotheses, key=lambda h: h.total_score)
            if best_role.confidence in (Confidence.CONFIRMED, Confidence.HIGH, Confidence.MEDIUM):
                # Extract role from claim (e.g., "holmes is an investigator" -> "investigator")
                role_part = best_role.claim.split(" is ")[-1]
                # Remove articles
                for article in ["an ", "a "]:
                    if role_part.startswith(article):
                        role_part = role_part[len(article):]
                profile.role = role_part
        
        # Find best gender hypothesis
        gender_hypotheses = [h for h in profile.hypotheses if h.category == "gender"]
        if gender_hypotheses:
            best_gender = max(gender_hypotheses, key=lambda h: h.total_score)
            if best_gender.confidence in (Confidence.CONFIRMED, Confidence.HIGH):
                # Check for "is male" vs "is female" to avoid substring match
                profile.gender = "male" if "is male" in best_gender.claim else "female"
    
    def explain_profile(self, profile: EntityProfile) -> str:
        """Generate a natural language explanation of the profile."""
        lines = [f"Profile for {profile.entity.title()}:", ""]
        
        for h in profile.hypotheses:
            status = "✓" if h.confidence in (Confidence.CONFIRMED, Confidence.HIGH) else "?"
            if h.confidence == Confidence.REFUTED:
                status = "✗"
            
            lines.append(f"{status} {h.claim} (confidence: {h.confidence.name})")
            
            for p in h.predictions:
                result = "✓" if p.result else "✗" if p.result == False else "?"
                lines.append(f"    {result} {p.description}")
                for e in p.evidence[:2]:
                    lines.append(f"        → {e}")
            lines.append("")
        
        if profile.role:
            lines.append(f"Confirmed role: {profile.role}")
        if profile.gender:
            lines.append(f"Confirmed gender: {profile.gender}")
        
        return "\n".join(lines)


def profile_with_hypotheses(entity: str, frames: List[Dict]) -> EntityProfile:
    """Convenience function to profile an entity."""
    profiler = HypothesisProfiler(frames)
    return profiler.profile_entity(entity)
