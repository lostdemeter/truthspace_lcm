#!/usr/bin/env python3
"""
Tachyon Response Generator

Generates responses using bidirectional navigation:
- Forward: Data → Features (what the data says)
- Backward: Hypothesis → Evidence (what we want to know)

The convergence of forward and backward navigation produces
high-confidence, well-grounded responses.

Now with holographic style projection for natural prose output.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Dict, List
from truthspace_lcm.core.hypothesis_navigator import BidirectionalReasoner, HypothesisNavigator
from core.style_projection import HolographicStyleProjector, project_with_style


class TachyonResponseGenerator:
    """
    Generate responses using Tachyon (bidirectional) navigation.
    
    This combines:
    1. Forward attention: What does the data say about the entity?
    2. Backward hypothesis: What role/attributes best explain the data?
    3. Convergence: Where do forward and backward agree?
    4. Style projection: Transform concept output into natural prose
    """
    
    def __init__(self, frames: List[Dict], style: str = "book_report"):
        self.frames = frames
        self.reasoner = BidirectionalReasoner(frames)
        self.navigator = self.reasoner.navigator
        self.style_projector = HolographicStyleProjector(style)
    
    def generate_who_response(self, entity: str, depth: float = 0.0, 
                               use_style: bool = True) -> Optional[str]:
        """
        Generate a "Who is X?" response using bidirectional reasoning.
        
        Args:
            entity: Entity to describe
            depth: -1 (terse) to +1 (elaborate)
            use_style: If True, apply holographic style projection
            
        Returns:
            Natural language response
        """
        entity_lower = entity.lower()
        
        # Check if entity exists in our data
        if entity_lower not in self.navigator.entity_frames:
            return None
        
        # Apply bidirectional reasoning
        result = self.reasoner.reason_about_entity(entity_lower)
        
        # Extract key information
        role = result['refined']['role']
        confidence = result['refined']['confidence']
        key_features = result['refined']['key_features']
        reasoning = result['refined']['reasoning']
        
        # Get gender from navigator
        gender_result = self.navigator.find_best_hypothesis(entity_lower, 'gender')
        gender = gender_result[0] if gender_result[0] else 'unknown'
        
        # Get source from frames - use most common source
        frames = self.navigator.entity_frames.get(entity_lower, [])
        from collections import Counter
        source_counts = Counter(f.get('source', '') for f in frames if f.get('source'))
        source = source_counts.most_common(1)[0][0] if source_counts else "the text"
        
        # Use holographic style projection if enabled
        if use_style:
            content = {
                'name': entity,
                'role': role,
                'gender': gender,
                'source': source,
                'key_features': key_features,
                'confidence': confidence,
            }
            return self.style_projector.project(content, depth)
        
        # Fall back to basic response
        return self._build_response(
            entity=entity,
            role=role,
            gender=gender,
            confidence=confidence,
            key_features=key_features,
            source=source,
            depth=depth
        )
    
    def _build_response(self, entity: str, role: str, gender: str,
                        confidence: str, key_features: List[str],
                        source: str, depth: float) -> str:
        """Build natural language response from reasoning results."""
        
        # Determine pronoun
        pronoun = 'she' if gender == 'female' else 'he'
        pronoun_cap = pronoun.capitalize()
        
        # Format entity name
        name = entity.title()
        
        # Clean up role (remove underscores)
        role_clean = role.replace('_', ' ') if role else None
        
        # Determine article
        article = 'an' if role_clean and role_clean[0].lower() in 'aeiou' else 'a'
        
        sentences = []
        
        # Opening sentence
        if role_clean:
            sentences.append(f"{name} is {article} {role_clean} in {source}.")
        else:
            sentences.append(f"{name} appears in {source}.")
        
        # Key features (if not terse)
        if depth >= -0.3 and key_features:
            feature_text = self._clean_feature(key_features[0], pronoun_cap)
            if feature_text:
                sentences.append(feature_text)
        
        # Additional detail for elaborate mode
        if depth > 0.3 and len(key_features) > 1:
            second_feature = self._clean_feature(key_features[1], pronoun_cap, prefix="Additionally, ")
            if second_feature:
                sentences.append(second_feature)
        
        # Confidence indicator for elaborate mode
        if depth > 0.5:
            if confidence == 'high':
                sentences.append(f"This characterization is well-supported by the text.")
            else:
                sentences.append(f"This is based on available textual evidence.")
        
        return " ".join(sentences)
    
    def _clean_feature(self, feature: str, pronoun_cap: str, prefix: str = "") -> str:
        """Clean up a feature string for natural language output."""
        if not feature:
            return ""
        
        # Remove parenthetical counts like (6x) or (26/81)
        import re
        feature = re.sub(r'\s*\d+x\s*', '', feature)
        feature = re.sub(r'\s*\d+/\d+\s*', '', feature)
        feature = feature.replace("(", "").replace(")", "").strip()
        
        if not feature:
            return ""
        
        # Build natural sentence
        feature_lower = feature.lower()
        
        if "interacts with" in feature_lower:
            return f"{prefix}{pronoun_cap} {feature_lower}."
        elif "speaks frequently" in feature_lower:
            return f"{prefix}{pronoun_cap} speaks frequently throughout the narrative."
        elif "focuses on" in feature_lower:
            target = feature_lower.replace("focuses on", "").strip()
            return f"{prefix}{pronoun_cap} is closely associated with {target.title()}."
        elif "highly observant" in feature_lower:
            return f"{prefix}{pronoun_cap} is highly observant and perceptive."
        elif "encounters unusual" in feature_lower:
            return f"{prefix}{pronoun_cap} encounters unusual and fantastical entities."
        elif "moves frequently" in feature_lower:
            return f"{prefix}{pronoun_cap} is active and moves frequently through the story."
        elif "family/friends" in feature_lower:
            return f"{prefix}{pronoun_cap} interacts primarily with family and friends."
        elif "no authority" in feature_lower:
            return f"{prefix}{pronoun_cap} operates outside official authority structures."
        elif "heavily described" in feature_lower:
            return f"{prefix}{pronoun_cap} is prominently featured and described in detail."
        elif "not an active mover" in feature_lower:
            return f"{prefix}{pronoun_cap} is more often described than shown in action."
        elif "crime-related" in feature_lower:
            return f"{prefix}{pronoun_cap} is involved in crime-related matters."
        else:
            return f"{prefix}{pronoun_cap} {feature_lower}."
    
    def explain_reasoning(self, entity: str) -> str:
        """Get detailed explanation of the reasoning process."""
        return self.reasoner.explain(entity)


# Global instance
_tachyon_generator: Optional[TachyonResponseGenerator] = None
_tachyon_frames_id: Optional[int] = None  # Track which frames we're using


def get_tachyon_generator(frames: List[Dict] = None) -> Optional[TachyonResponseGenerator]:
    """Get the Tachyon response generator."""
    global _tachyon_generator, _tachyon_frames_id
    
    if frames is None:
        return _tachyon_generator
    
    # Check if we need to reinitialize (different frames)
    frames_id = id(frames)
    if _tachyon_generator is None or _tachyon_frames_id != frames_id:
        _tachyon_generator = TachyonResponseGenerator(frames)
        _tachyon_frames_id = frames_id
    
    return _tachyon_generator


def generate_tachyon_response(entity: str, frames: List[Dict] = None, 
                               depth: float = 0.0) -> Optional[str]:
    """
    Convenience function to generate a Tachyon-based response.
    
    Args:
        entity: Entity name
        frames: Concept frames (required on first call)
        depth: Response depth (-1 to +1)
        
    Returns:
        Natural language response, or None if entity not found
    """
    if frames is None:
        return None
    
    generator = get_tachyon_generator(frames)
    if generator:
        return generator.generate_who_response(entity, depth=depth)
    return None
