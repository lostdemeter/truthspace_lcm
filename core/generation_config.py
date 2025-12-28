"""
Generation Configuration for GeometricLCM

Maps LLM-style hyperparameters to GeometricLCM's geometric generation system.

OpenAI-style parameters:
- max_tokens: Maximum response length
- temperature: Creativity/randomness (maps to response variety)
- top_p: Nucleus sampling (maps to confidence threshold)
- presence_penalty: Avoid repetition (maps to variety in templates)
- frequency_penalty: Penalize frequent tokens (maps to Zipf weighting)

GeometricLCM-specific parameters:
- style: Formal (-1) to casual (+1) - the φ-dial x-axis
- perspective: Subjective (-1) to meta (+1) - the φ-dial y-axis
- depth: Terse (-1) to elaborate (+1) - the φ-dial z-axis
- certainty: Definitive (-1) to hedged (+1) - the φ-dial w-axis
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Combines OpenAI-compatible parameters with GeometricLCM's φ-dial system.
    """
    
    # === Response Length Control ===
    max_tokens: int = 500
    """Maximum tokens in response. Approximate - truncates at sentence boundary."""
    
    min_tokens: int = 10
    """Minimum tokens before allowing truncation."""
    
    # === OpenAI-Compatible Parameters ===
    temperature: float = 0.7
    """
    Controls response variety (0.0-2.0).
    - 0.0: Deterministic, always pick best match
    - 0.7: Balanced variety
    - 1.5+: High variety, more creative/random
    Maps to: template selection randomness, word choice variety
    """
    
    top_p: float = 1.0
    """
    Nucleus sampling threshold (0.0-1.0).
    - 1.0: Consider all options
    - 0.9: Only top 90% probability mass
    Maps to: confidence threshold for including information
    """
    
    presence_penalty: float = 0.0
    """
    Penalize tokens that have appeared (-2.0 to 2.0).
    - Positive: Encourage new topics
    - Negative: Allow repetition
    Maps to: variety in response templates
    """
    
    frequency_penalty: float = 0.0
    """
    Penalize frequent tokens (-2.0 to 2.0).
    - Positive: Prefer rare words
    - Negative: Prefer common words
    Maps to: Zipf weighting adjustment
    """
    
    # === GeometricLCM φ-Dial Parameters ===
    style: float = 0.0
    """
    Style dial (-1.0 to +1.0).
    - -1.0: Formal, specific, rare words
    - 0.0: Neutral
    - +1.0: Casual, universal, common words
    """
    
    perspective: float = 0.0
    """
    Perspective dial (-1.0 to +1.0).
    - -1.0: Subjective, experiential
    - 0.0: Objective, factual
    - +1.0: Meta, analytical
    """
    
    depth: float = 0.0
    """
    Depth dial (-1.0 to +1.0).
    - -1.0: Terse, minimal
    - 0.0: Standard, balanced
    - +1.0: Elaborate, detailed
    """
    
    certainty: float = 0.0
    """
    Certainty dial (-1.0 to +1.0).
    - -1.0: Definitive, assertive
    - 0.0: Neutral
    - +1.0: Hedged, tentative
    """
    
    # === Code Generation Parameters ===
    code_completeness: str = "full"
    """
    Code generation completeness level.
    - "stub": Just signature and docstring
    - "skeleton": Structure with TODO comments
    - "basic": Working implementation, minimal
    - "full": Complete with error handling, types
    """
    
    include_imports: bool = True
    """Whether to include import statements in generated code."""
    
    include_types: bool = True
    """Whether to include type hints in generated code."""
    
    include_docstrings: bool = True
    """Whether to include docstrings in generated code."""
    
    # === Response Style ===
    response_format: str = "natural"
    """
    Response format style.
    - "natural": Conversational, flowing
    - "structured": Clear sections, bullet points
    - "minimal": Just the facts
    - "verbose": Full explanations
    """
    
    def to_phi_dial(self) -> tuple:
        """Convert to φ-dial tuple (x, y, z, w)."""
        return (self.style, self.perspective, self.depth, self.certainty)
    
    def apply_openai_params(self, temperature: float = None, top_p: float = None,
                            presence_penalty: float = None, frequency_penalty: float = None,
                            max_tokens: int = None):
        """Apply OpenAI-style parameters, mapping to GeometricLCM equivalents."""
        if temperature is not None:
            self.temperature = temperature
            # High temperature → more variety → adjust depth toward elaborate
            if temperature > 1.0:
                self.depth = min(1.0, self.depth + (temperature - 1.0) * 0.5)
        
        if top_p is not None:
            self.top_p = top_p
        
        if presence_penalty is not None:
            self.presence_penalty = presence_penalty
        
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty
            # High frequency penalty → prefer rare words → more formal
            if frequency_penalty > 0:
                self.style = max(-1.0, self.style - frequency_penalty * 0.3)
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            # Adjust depth based on token budget
            if max_tokens < 100:
                self.depth = max(-1.0, self.depth - 0.5)  # Force terse
            elif max_tokens > 500:
                self.depth = min(1.0, self.depth + 0.3)   # Allow elaborate
    
    def get_effective_max_words(self) -> int:
        """
        Get effective max words based on max_tokens.
        
        Rough approximation: 1 token ≈ 0.75 words for English.
        """
        return int(self.max_tokens * 0.75)
    
    def get_target_sentences(self) -> int:
        """Get target number of sentences based on depth and max_tokens."""
        base = 2  # Standard: 2-3 sentences
        
        # Adjust for depth
        if self.depth < -0.3:
            base = 1  # Terse: 1-2 sentences
        elif self.depth > 0.3:
            base = 4  # Elaborate: 4-6 sentences
        
        # Adjust for token budget
        max_words = self.get_effective_max_words()
        if max_words < 30:
            return 1
        elif max_words < 75:
            return min(base, 2)
        elif max_words > 150:
            return base + 2
        
        return base
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GenerationConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# === Preset Configurations ===

PRESETS = {
    "default": GenerationConfig(),
    
    "brief": GenerationConfig(
        max_tokens=100,
        depth=-0.7,
        response_format="minimal",
    ),
    
    "detailed": GenerationConfig(
        max_tokens=1000,
        depth=0.7,
        response_format="verbose",
    ),
    
    "formal": GenerationConfig(
        style=-0.8,
        perspective=0.3,
        certainty=-0.5,
    ),
    
    "casual": GenerationConfig(
        style=0.8,
        perspective=-0.3,
        certainty=0.3,
    ),
    
    "code_minimal": GenerationConfig(
        code_completeness="stub",
        include_imports=False,
        include_types=False,
    ),
    
    "code_full": GenerationConfig(
        code_completeness="full",
        include_imports=True,
        include_types=True,
        include_docstrings=True,
    ),
}


def get_preset(name: str) -> GenerationConfig:
    """Get a preset configuration by name."""
    if name in PRESETS:
        # Return a copy to avoid mutation
        return GenerationConfig.from_dict(PRESETS[name].to_dict())
    raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")


# === Global Default Configuration ===

_default_config: Optional[GenerationConfig] = None
_config_path: Optional[str] = None


def get_default_config() -> GenerationConfig:
    """Get the default generation configuration."""
    global _default_config
    if _default_config is None:
        _default_config = _load_or_create_default()
    return _default_config


def set_default_config(config: GenerationConfig):
    """Set the default generation configuration."""
    global _default_config
    _default_config = config


def _load_or_create_default() -> GenerationConfig:
    """Load default config from file or create new one."""
    # Look for config file in standard locations
    search_paths = [
        Path.cwd() / "generation_config.json",
        Path.home() / ".config" / "geometric-lcm" / "generation_config.json",
        Path(__file__).parent.parent / "generation_config.json",
    ]
    
    for path in search_paths:
        if path.exists():
            try:
                return GenerationConfig.load(str(path))
            except Exception:
                pass
    
    return GenerationConfig()


def create_config_from_request(
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    style: float = None,
    perspective: float = None,
    depth: float = None,
    certainty: float = None,
    **kwargs
) -> GenerationConfig:
    """
    Create a GenerationConfig from API request parameters.
    
    Merges with defaults and applies OpenAI parameter mappings.
    """
    config = GenerationConfig.from_dict(get_default_config().to_dict())
    
    # Apply OpenAI-style parameters
    config.apply_openai_params(
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
    )
    
    # Apply GeometricLCM-specific parameters (override mappings)
    if style is not None:
        config.style = style
    if perspective is not None:
        config.perspective = perspective
    if depth is not None:
        config.depth = depth
    if certainty is not None:
        config.certainty = certainty
    
    return config
