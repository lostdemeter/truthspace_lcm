#!/usr/bin/env python3
"""
Concept Pair Trainer - Automated concept discovery using local LLM

When the Geometric LCM encounters an unknown concept, this trainer:
1. Queries a local LLM to generate transformation pairs
2. Parses the pairs into (source, target, dimension, src_val, tgt_val)
3. Learns them into a PhiSpace
4. The concept is now geometrically defined!

This enables on-the-fly vocabulary growth without manual curation.

Usage:
    trainer = ConceptPairTrainer()
    
    # Train a single concept
    pairs = trainer.train_concept("jumped", dimensions=["tense"])
    
    # Train and add to a space
    space = PhiSpace()
    trainer.train_into_space(space, "jumped", dimensions=["tense"])
    
    # Now the space knows about "jumped"
    print(space("jumped", "tense"))  # "will jump"

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.phi_space import PhiSpace


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransformationPair:
    """A single transformation pair."""
    source: str
    target: str
    dimension: str
    source_value: str
    target_value: str
    
    def __repr__(self):
        return f"'{self.source}' → '{self.target}' ({self.dimension}: {self.source_value}→{self.target_value})"


@dataclass
class TrainingResult:
    """Result of training a concept."""
    concept: str
    pairs: List[TransformationPair] = field(default_factory=list)
    success: bool = False
    error: str = ""
    raw_response: str = ""


# =============================================================================
# DIMENSION TEMPLATES
# =============================================================================

DIMENSION_PROMPTS = {
    "tense": {
        "values": ["past", "present", "future"],
        "prompt": """For the word/phrase "{concept}", provide its forms in different tenses.
Format your response as JSON:
{{
  "past": "<past tense form>",
  "present": "<present tense form>",
  "future": "<future tense form>"
}}

Example for "go":
{{
  "past": "went",
  "present": "go",
  "future": "will go"
}}

Now provide the tense forms for "{concept}":""",
    },
    
    "formality": {
        "values": ["casual", "formal"],
        "prompt": """For the word/phrase "{concept}", provide casual and formal equivalents.
Format your response as JSON:
{{
  "casual": "<casual/informal form>",
  "formal": "<formal/professional form>"
}}

Example for "hello":
{{
  "casual": "hi",
  "formal": "greetings"
}}

Now provide the formality forms for "{concept}":""",
    },
    
    "intensity": {
        "values": ["weak", "medium", "strong"],
        "prompt": """For the word/phrase "{concept}", provide forms at different intensity levels.
Format your response as JSON:
{{
  "weak": "<mild/weak form>",
  "medium": "<moderate form>",
  "strong": "<intense/strong form>"
}}

Example for "happy":
{{
  "weak": "content",
  "medium": "happy",
  "strong": "ecstatic"
}}

Now provide the intensity forms for "{concept}":""",
    },
    
    "polarity": {
        "values": ["negative", "neutral", "positive"],
        "prompt": """For the word/phrase "{concept}", provide forms with different emotional polarity.
Format your response as JSON:
{{
  "negative": "<negative connotation form>",
  "neutral": "<neutral form>",
  "positive": "<positive connotation form>"
}}

Example for "result":
{{
  "negative": "consequence",
  "neutral": "result",
  "positive": "achievement"
}}

Now provide the polarity forms for "{concept}":""",
    },
    
    "specificity": {
        "values": ["general", "specific"],
        "prompt": """For the word/phrase "{concept}", provide general and specific forms.
Format your response as JSON:
{{
  "general": "<more general/abstract form>",
  "specific": "<more specific/concrete form>"
}}

Example for "dog":
{{
  "general": "animal",
  "specific": "golden retriever"
}}

Now provide the specificity forms for "{concept}":""",
    },
    
    "voice": {
        "values": ["active", "passive"],
        "prompt": """For the verb "{concept}", provide active and passive voice forms.
Format your response as JSON:
{{
  "active": "<active voice form>",
  "passive": "<passive voice form>"
}}

Example for "wrote":
{{
  "active": "wrote",
  "passive": "was written"
}}

Now provide the voice forms for "{concept}":""",
    },
    
    "regality": {
        "values": ["common", "noble", "royal"],
        "prompt": """For the word/phrase "{concept}", provide forms at different levels of regality/formality.
Format your response as JSON:
{{
  "common": "<everyday/common form>",
  "noble": "<elevated/noble form>",
  "royal": "<royal/majestic form>"
}}

Example for "house":
{{
  "common": "house",
  "noble": "manor",
  "royal": "palace"
}}

Now provide the regality forms for "{concept}":""",
    },
}


# =============================================================================
# CONCEPT PAIR TRAINER
# =============================================================================

class ConceptPairTrainer:
    """
    Trains concept pairs using a local LLM.
    
    Queries the LLM to generate transformation pairs for unknown concepts,
    then parses and validates the responses.
    """
    
    def __init__(self,
                 ollama_url: str = "http://127.0.0.1:11434/api/generate",
                 model: str = "qwen2.5:14b",
                 timeout: int = 30):
        """
        Initialize the trainer.
        
        Args:
            ollama_url: URL for Ollama API
            model: Model to use for generation
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        
        # Custom dimension prompts can be added
        self.dimension_prompts = dict(DIMENSION_PROMPTS)
    
    def add_dimension(self, 
                      name: str, 
                      values: List[str], 
                      prompt_template: str) -> None:
        """
        Add a custom dimension for training.
        
        Args:
            name: Dimension name
            values: List of value names (e.g., ["low", "medium", "high"])
            prompt_template: Prompt template with {concept} placeholder
        """
        self.dimension_prompts[name] = {
            "values": values,
            "prompt": prompt_template,
        }
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query the local LLM."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500,
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, str]]:
        """Parse JSON from LLM response, handling common issues."""
        if not response:
            return None
        
        # Try to find JSON in the response
        # Sometimes LLMs add explanation before/after
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            return None
        
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to fix common issues
            json_str = json_match.group()
            # Remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            try:
                return json.loads(json_str)
            except:
                return None
    
    def train_concept(self,
                      concept: str,
                      dimensions: List[str] = None) -> TrainingResult:
        """
        Train transformation pairs for a concept.
        
        Args:
            concept: The concept to train (word or phrase)
            dimensions: List of dimensions to train (default: all available)
            
        Returns:
            TrainingResult with generated pairs
        """
        if dimensions is None:
            dimensions = list(self.dimension_prompts.keys())
        
        result = TrainingResult(concept=concept)
        
        for dim in dimensions:
            if dim not in self.dimension_prompts:
                continue
            
            dim_config = self.dimension_prompts[dim]
            prompt = dim_config["prompt"].format(concept=concept)
            values = dim_config["values"]
            
            # Query LLM
            response = self._query_llm(prompt)
            if not response:
                continue
            
            result.raw_response += f"\n--- {dim} ---\n{response}\n"
            
            # Parse response
            forms = self._parse_json_response(response)
            if not forms:
                continue
            
            # Generate pairs from forms
            # Each adjacent pair of values creates a transformation
            for i in range(len(values) - 1):
                src_val = values[i]
                tgt_val = values[i + 1]
                
                src_form = forms.get(src_val)
                tgt_form = forms.get(tgt_val)
                
                if src_form and tgt_form and src_form != tgt_form:
                    pair = TransformationPair(
                        source=src_form.lower().strip(),
                        target=tgt_form.lower().strip(),
                        dimension=dim,
                        source_value=src_val,
                        target_value=tgt_val,
                    )
                    result.pairs.append(pair)
            
            # Also create pairs for non-adjacent values (e.g., past→future)
            if len(values) > 2:
                first_val = values[0]
                last_val = values[-1]
                first_form = forms.get(first_val)
                last_form = forms.get(last_val)
                
                if first_form and last_form and first_form != last_form:
                    pair = TransformationPair(
                        source=first_form.lower().strip(),
                        target=last_form.lower().strip(),
                        dimension=dim,
                        source_value=first_val,
                        target_value=last_val,
                    )
                    result.pairs.append(pair)
        
        result.success = len(result.pairs) > 0
        if not result.success:
            result.error = "No valid pairs generated"
        
        return result
    
    def train_into_space(self,
                         space: PhiSpace,
                         concept: str,
                         dimensions: List[str] = None) -> TrainingResult:
        """
        Train a concept and add pairs directly to a PhiSpace.
        
        Args:
            space: PhiSpace to add pairs to
            concept: Concept to train
            dimensions: Dimensions to train (default: all)
            
        Returns:
            TrainingResult with generated pairs
        """
        result = self.train_concept(concept, dimensions)
        
        if result.success:
            for pair in result.pairs:
                space.learn(
                    pair.source,
                    pair.target,
                    pair.dimension,
                    pair.source_value,
                    pair.target_value
                )
        
        return result
    
    def train_batch(self,
                    concepts: List[str],
                    dimensions: List[str] = None) -> List[TrainingResult]:
        """
        Train multiple concepts.
        
        Args:
            concepts: List of concepts to train
            dimensions: Dimensions to train (default: all)
            
        Returns:
            List of TrainingResults
        """
        results = []
        for concept in concepts:
            result = self.train_concept(concept, dimensions)
            results.append(result)
            print(f"Trained '{concept}': {len(result.pairs)} pairs")
        return results
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        try:
            response = requests.get(
                self.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# =============================================================================
# SELF-TRAINING SPACE
# =============================================================================

class SelfTrainingSpace(PhiSpace):
    """
    A PhiSpace that automatically trains unknown concepts using LLM.
    
    When a transformation is requested for an unknown concept,
    it queries the LLM to learn the concept on-the-fly.
    
    Usage:
        space = SelfTrainingSpace()
        
        # First call: trains "jumped" using LLM, then transforms
        result = space("jumped", "tense", "past", "future")
        # Returns "will jump" (learned from LLM)
        
        # Second call: uses cached knowledge (no LLM)
        result = space("jumped", "tense", "past", "future")
    """
    
    def __init__(self, 
                 trainer: ConceptPairTrainer = None,
                 auto_train_dimensions: List[str] = None,
                 **kwargs):
        """
        Initialize a self-training space.
        
        Args:
            trainer: ConceptPairTrainer to use (creates default if None)
            auto_train_dimensions: Dimensions to auto-train (default: ["tense"])
            **kwargs: Passed to PhiSpace
        """
        super().__init__(**kwargs)
        self.trainer = trainer or ConceptPairTrainer()
        self.auto_train_dimensions = auto_train_dimensions or ["tense"]
        self._training_attempts: Dict[str, bool] = {}  # Track what we've tried
    
    def transform(self,
                  item: Any,
                  dimension: str,
                  source_value: str = None,
                  target_value: str = None):
        """
        Transform with auto-training for unknown concepts.
        """
        # First try normal transform
        result = super().transform(item, dimension, source_value, target_value)
        
        if result.success:
            return result
        
        # If failed due to unknown item, try to train it
        key = self._normalize(item)
        if key not in self._training_attempts:
            self._training_attempts[key] = True
            
            print(f"[SelfTrainingSpace] Unknown concept '{item}', training...")
            
            # Train the concept
            train_result = self.trainer.train_into_space(
                self, 
                str(item),
                dimensions=[dimension] if dimension in self.trainer.dimension_prompts 
                          else self.auto_train_dimensions
            )
            
            if train_result.success:
                print(f"[SelfTrainingSpace] Learned {len(train_result.pairs)} pairs for '{item}'")
                # Retry the transform
                return super().transform(item, dimension, source_value, target_value)
            else:
                print(f"[SelfTrainingSpace] Failed to train '{item}': {train_result.error}")
        
        return result


# =============================================================================
# MAIN - DEMO
# =============================================================================

def main():
    """Demo the concept pair trainer."""
    print("=" * 60)
    print("CONCEPT PAIR TRAINER - Demo")
    print("=" * 60)
    
    trainer = ConceptPairTrainer()
    
    # Check if LLM is available
    if not trainer.is_available():
        print("\n⚠️  Ollama not available. Using mock responses for demo.")
        # Create mock trainer for demo
        class MockTrainer(ConceptPairTrainer):
            def _query_llm(self, prompt):
                # Return mock responses based on prompt content
                if "jumped" in prompt.lower():
                    if "tense" in prompt.lower():
                        return '{"past": "jumped", "present": "jump", "future": "will jump"}'
                    if "intensity" in prompt.lower():
                        return '{"weak": "hopped", "medium": "jumped", "strong": "leaped"}'
                if "happy" in prompt.lower():
                    if "intensity" in prompt.lower():
                        return '{"weak": "content", "medium": "happy", "strong": "ecstatic"}'
                    if "polarity" in prompt.lower():
                        return '{"negative": "sad", "neutral": "okay", "positive": "happy"}'
                if "house" in prompt.lower():
                    if "regality" in prompt.lower():
                        return '{"common": "house", "noble": "manor", "royal": "palace"}'
                return None
        trainer = MockTrainer()
    
    # Test 1: Train a single concept
    print("\n--- Test 1: Train 'jumped' (tense) ---")
    result = trainer.train_concept("jumped", dimensions=["tense"])
    print(f"Success: {result.success}")
    print(f"Pairs generated:")
    for pair in result.pairs:
        print(f"  {pair}")
    
    # Test 2: Train into a space
    print("\n--- Test 2: Train into PhiSpace ---")
    space = PhiSpace()
    trainer.train_into_space(space, "jumped", dimensions=["tense"])
    print(f"Space: {space}")
    
    # Test transformation
    transformed = space("jumped", "tense", "past", "future")
    print(f"Transform 'jumped' (past→future): {transformed}")
    
    transformed = space("jumped", "tense", "past", "present")
    print(f"Transform 'jumped' (past→present): {transformed}")
    
    # Test 3: Multiple dimensions
    print("\n--- Test 3: Multiple dimensions ---")
    result = trainer.train_concept("happy", dimensions=["intensity", "polarity"])
    print(f"Pairs for 'happy':")
    for pair in result.pairs:
        print(f"  {pair}")
    
    # Test 4: Regality dimension
    print("\n--- Test 4: Regality dimension ---")
    result = trainer.train_concept("house", dimensions=["regality"])
    print(f"Pairs for 'house':")
    for pair in result.pairs:
        print(f"  {pair}")
    
    # Test 5: SelfTrainingSpace
    print("\n--- Test 5: SelfTrainingSpace (auto-learning) ---")
    auto_space = SelfTrainingSpace(trainer=trainer)
    
    # This should auto-train "jumped" and then transform
    print("Requesting transform for unknown 'jumped'...")
    result = auto_space.transform("jumped", "tense", "past", "future")
    print(f"Result: {result}")
    
    # Second call should use cached knowledge
    print("\nSecond call (should use cache)...")
    result = auto_space.transform("jumped", "tense", "past", "present")
    print(f"Result: {result}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
Key features:
1. ConceptPairTrainer - Queries LLM for transformation pairs
2. train_concept() - Train a single concept
3. train_into_space() - Train and add to PhiSpace
4. SelfTrainingSpace - Auto-trains unknown concepts on-the-fly

This enables the Geometric LCM to grow its vocabulary automatically!
""")


if __name__ == "__main__":
    main()
