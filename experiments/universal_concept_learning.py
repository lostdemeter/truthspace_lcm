#!/usr/bin/env python3
"""
Universal Concept Learning Experiment

Can we extend geometric concept learning beyond verbs to ALL parts of speech?

The hypothesis: Every word exists in a multi-dimensional concept space where
dimensions correspond to linguistic properties. If we can identify the right
dimensions for each part of speech, we can learn ANY word geometrically.

Parts of Speech and Their Dimensions:
=====================================

VERBS (already working):
  - tense: past/present/future
  - voice: active/passive
  - aspect: simple/progressive/perfect

NOUNS:
  - number: singular/plural
  - definiteness: definite/indefinite (the/a)
  - case: nominative/accusative/genitive (he/him/his)
  - formality: formal/casual (father/dad, mother/mom)
  - specificity: specific/general (poodle/dog/animal)
  - size: diminutive/normal/augmentative (doggy/dog)

ADJECTIVES:
  - degree: positive/comparative/superlative (fast/faster/fastest)
  - intensity: mild/moderate/intense (warm/hot/scorching)
  - polarity: positive/negative (good/bad, happy/sad)

ADVERBS:
  - degree: positive/comparative/superlative (quickly/more quickly/most quickly)
  - intensity: mild/moderate/intense (somewhat/very/extremely)

PRONOUNS:
  - person: first/second/third (I/you/he)
  - number: singular/plural (he/they)
  - case: nominative/accusative/genitive (he/him/his)
  - gender: masculine/feminine/neutral (he/she/they)

This experiment tests whether an LLM can generate meaningful transformation
pairs for these dimensions, enabling universal concept learning.
"""

import json
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# PART OF SPEECH DEFINITIONS
# =============================================================================

class PartOfSpeech(Enum):
    VERB = "verb"
    NOUN = "noun"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"

# Dimensions applicable to each part of speech
POS_DIMENSIONS = {
    PartOfSpeech.VERB: {
        "tense": ["past", "present", "future"],
        "voice": ["active", "passive"],
        "aspect": ["simple", "progressive", "perfect"],
    },
    PartOfSpeech.NOUN: {
        "number": ["singular", "plural"],
        "formality": ["formal", "casual"],
        "specificity": ["specific", "general"],
        "size": ["diminutive", "normal", "augmentative"],
    },
    PartOfSpeech.ADJECTIVE: {
        "degree": ["positive", "comparative", "superlative"],
        "intensity": ["mild", "moderate", "intense"],
    },
    PartOfSpeech.ADVERB: {
        "degree": ["positive", "comparative", "superlative"],
        "manner": ["formal", "casual"],
    },
    PartOfSpeech.PRONOUN: {
        "person": ["first", "second", "third"],
        "number": ["singular", "plural"],
        "case": ["nominative", "accusative", "genitive"],
        "gender": ["masculine", "feminine", "neutral"],
    },
}

# =============================================================================
# DIMENSION-SPECIFIC PROMPTS
# =============================================================================

def get_prompt_for_dimension(word: str, pos: PartOfSpeech, dimension: str) -> str:
    """Generate an LLM prompt for learning a word's forms along a dimension."""
    
    values = POS_DIMENSIONS[pos][dimension]
    values_str = ", ".join(values)
    
    prompts = {
        # VERB dimensions
        ("verb", "tense"): f"""Give me the tense forms of the verb "{word}".
Return ONLY a JSON object with keys: past, present, future.
Example for "walk": {{"past": "walked", "present": "walk", "future": "will walk"}}
Now for "{word}":""",

        ("verb", "voice"): f"""Give me the voice forms of the verb "{word}" (in present tense).
Return ONLY a JSON object with keys: active, passive.
Example for "eat": {{"active": "eats", "passive": "is eaten"}}
Now for "{word}":""",

        ("verb", "aspect"): f"""Give me the aspect forms of the verb "{word}" (in present tense).
Return ONLY a JSON object with keys: simple, progressive, perfect.
Example for "walk": {{"simple": "walks", "progressive": "is walking", "perfect": "has walked"}}
Now for "{word}":""",

        # NOUN dimensions
        ("noun", "number"): f"""Give me the number forms of the noun "{word}".
Return ONLY a JSON object with keys: singular, plural.
Example for "dog": {{"singular": "dog", "plural": "dogs"}}
Now for "{word}":""",

        ("noun", "formality"): f"""Give me formal and casual variants of the noun "{word}".
Return ONLY a JSON object with keys: formal, casual.
Example for "father": {{"formal": "father", "casual": "dad"}}
If no casual variant exists, use the same word for both.
Now for "{word}":""",

        ("noun", "specificity"): f"""Give me specificity variants of the noun "{word}".
Return ONLY a JSON object with keys: specific, general.
Example for "poodle": {{"specific": "poodle", "general": "dog"}}
Example for "dog": {{"specific": "dog", "general": "animal"}}
Now for "{word}":""",

        ("noun", "size"): f"""Give me size variants of the noun "{word}".
Return ONLY a JSON object with keys: diminutive, normal, augmentative.
Example for "dog": {{"diminutive": "doggy", "normal": "dog", "augmentative": "big dog"}}
If no natural variant exists, describe it (e.g., "little X", "big X").
Now for "{word}":""",

        # ADJECTIVE dimensions
        ("adjective", "degree"): f"""Give me the degree forms of the adjective "{word}".
Return ONLY a JSON object with keys: positive, comparative, superlative.
Example for "fast": {{"positive": "fast", "comparative": "faster", "superlative": "fastest"}}
Now for "{word}":""",

        ("adjective", "intensity"): f"""Give me intensity variants of the adjective "{word}".
Return ONLY a JSON object with keys: mild, moderate, intense.
Example for "hot": {{"mild": "warm", "moderate": "hot", "intense": "scorching"}}
Now for "{word}":""",

        # ADVERB dimensions
        ("adverb", "degree"): f"""Give me the degree forms of the adverb "{word}".
Return ONLY a JSON object with keys: positive, comparative, superlative.
Example for "quickly": {{"positive": "quickly", "comparative": "more quickly", "superlative": "most quickly"}}
Now for "{word}":""",

        ("adverb", "manner"): f"""Give me formal and casual variants of the adverb "{word}".
Return ONLY a JSON object with keys: formal, casual.
Example for "rapidly": {{"formal": "rapidly", "casual": "fast"}}
Now for "{word}":""",

        # PRONOUN dimensions
        ("pronoun", "case"): f"""Give me the case forms of the pronoun "{word}".
Return ONLY a JSON object with keys: nominative, accusative, genitive.
Example for "he": {{"nominative": "he", "accusative": "him", "genitive": "his"}}
Now for "{word}":""",

        ("pronoun", "number"): f"""Give me the number forms related to the pronoun "{word}".
Return ONLY a JSON object with keys: singular, plural.
Example for "he": {{"singular": "he", "plural": "they"}}
Now for "{word}":""",

        ("pronoun", "gender"): f"""Give me the gender variants of the pronoun "{word}".
Return ONLY a JSON object with keys: masculine, feminine, neutral.
Example for "he": {{"masculine": "he", "feminine": "she", "neutral": "they"}}
Now for "{word}":""",

        ("pronoun", "person"): f"""Give me the person variants of the pronoun "{word}".
Return ONLY a JSON object with keys: first, second, third.
Example for "I": {{"first": "I", "second": "you", "third": "he/she"}}
Now for "{word}":""",
    }
    
    key = (pos.value, dimension)
    if key in prompts:
        return prompts[key]
    
    # Generic fallback
    return f"""Give me the {dimension} forms of the {pos.value} "{word}".
Return ONLY a JSON object with keys: {values_str}.
Now for "{word}":"""


# =============================================================================
# LLM QUERY
# =============================================================================

@dataclass
class ConceptForms:
    """Forms of a concept along a dimension."""
    word: str
    pos: PartOfSpeech
    dimension: str
    forms: Dict[str, str]
    success: bool
    error: Optional[str] = None

def query_llm(prompt: str, 
              model: str = "qwen2.5:14b",
              url: str = "http://127.0.0.1:11434/api/generate") -> Optional[str]:
    """Query local Ollama LLM."""
    try:
        response = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=30)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        return None
    except Exception as e:
        print(f"LLM query failed: {e}")
        return None

def parse_json_response(response: str) -> Optional[Dict[str, str]]:
    """Extract JSON from LLM response."""
    if not response:
        return None
    
    # Try to find JSON in response
    text = response.strip()
    
    # Look for JSON object
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None

def learn_concept_forms(word: str, 
                        pos: PartOfSpeech, 
                        dimension: str) -> ConceptForms:
    """Learn the forms of a word along a dimension."""
    
    prompt = get_prompt_for_dimension(word, pos, dimension)
    response = query_llm(prompt)
    
    if not response:
        return ConceptForms(word, pos, dimension, {}, False, "LLM query failed")
    
    forms = parse_json_response(response)
    
    if not forms:
        return ConceptForms(word, pos, dimension, {}, False, f"Failed to parse: {response[:100]}")
    
    # Validate that we got the expected keys
    expected_keys = set(POS_DIMENSIONS[pos][dimension])
    actual_keys = set(forms.keys())
    
    if not expected_keys.issubset(actual_keys):
        missing = expected_keys - actual_keys
        return ConceptForms(word, pos, dimension, forms, False, f"Missing keys: {missing}")
    
    return ConceptForms(word, pos, dimension, forms, True)


# =============================================================================
# EXPERIMENT: TEST ALL PARTS OF SPEECH
# =============================================================================

def run_experiment():
    """Test concept learning across all parts of speech."""
    
    print("=" * 70)
    print("UNIVERSAL CONCEPT LEARNING EXPERIMENT")
    print("=" * 70)
    print()
    print("Testing if we can learn geometric concepts for ALL parts of speech...")
    print()
    
    # Test words for each part of speech
    test_cases = {
        PartOfSpeech.VERB: ["jumped", "swimming", "eat", "think", "create"],
        PartOfSpeech.NOUN: ["mailman", "cat", "happiness", "computer", "child"],
        PartOfSpeech.ADJECTIVE: ["hastily", "beautiful", "quick", "intelligent", "cold"],
        PartOfSpeech.ADVERB: ["quickly", "carefully", "very", "silently", "happily"],
        PartOfSpeech.PRONOUN: ["he", "she", "they", "I", "we"],
    }
    
    # Note: "hastily" is actually an adverb, testing if system handles misclassification
    # Let's fix the test cases
    test_cases[PartOfSpeech.ADJECTIVE] = ["beautiful", "quick", "intelligent", "cold", "happy"]
    
    results = {}
    
    for pos, words in test_cases.items():
        print(f"\n{'='*70}")
        print(f"TESTING {pos.value.upper()}S")
        print(f"{'='*70}")
        
        pos_results = []
        
        for word in words:
            print(f"\n  Word: '{word}'")
            print(f"  {'-'*50}")
            
            word_results = {}
            
            for dimension in POS_DIMENSIONS[pos]:
                result = learn_concept_forms(word, pos, dimension)
                word_results[dimension] = result
                
                if result.success:
                    forms_str = " → ".join(f"{k}='{v}'" for k, v in result.forms.items())
                    print(f"    {dimension}: ✓ {forms_str}")
                else:
                    print(f"    {dimension}: ✗ {result.error}")
            
            pos_results.append((word, word_results))
        
        results[pos] = pos_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    total_success = 0
    
    for pos, pos_results in results.items():
        pos_total = 0
        pos_success = 0
        
        for word, word_results in pos_results:
            for dim, result in word_results.items():
                pos_total += 1
                total_tests += 1
                if result.success:
                    pos_success += 1
                    total_success += 1
        
        pct = (pos_success / pos_total * 100) if pos_total > 0 else 0
        print(f"  {pos.value:12s}: {pos_success}/{pos_total} ({pct:.0f}%)")
    
    print(f"\n  {'TOTAL':12s}: {total_success}/{total_tests} ({total_success/total_tests*100:.0f}%)")
    
    return results


# =============================================================================
# EXPERIMENT: GENERATE TRANSFORMATION PAIRS
# =============================================================================

def generate_pairs_from_forms(forms: ConceptForms) -> List[Tuple[str, str, str, str]]:
    """
    Generate transformation pairs from concept forms.
    Returns list of (phrase1, phrase2, dimension, value1, value2) tuples.
    """
    pairs = []
    values = list(forms.forms.keys())
    
    for i, v1 in enumerate(values):
        for v2 in values[i+1:]:
            phrase1 = forms.forms[v1]
            phrase2 = forms.forms[v2]
            if phrase1 != phrase2:  # Only if actually different
                pairs.append((phrase1, phrase2, forms.dimension, v1, v2))
    
    return pairs


def demonstrate_geometric_learning():
    """Show how learned forms become geometric transformations."""
    
    print("\n" + "=" * 70)
    print("GEOMETRIC TRANSFORMATION DEMONSTRATION")
    print("=" * 70)
    
    # Learn a few concepts and show the pairs
    demos = [
        ("mailman", PartOfSpeech.NOUN, "number"),
        ("beautiful", PartOfSpeech.ADJECTIVE, "degree"),
        ("quickly", PartOfSpeech.ADVERB, "degree"),
        ("think", PartOfSpeech.VERB, "tense"),
    ]
    
    for word, pos, dimension in demos:
        print(f"\n  Learning '{word}' ({pos.value}, {dimension})...")
        
        result = learn_concept_forms(word, pos, dimension)
        
        if result.success:
            print(f"    Forms: {result.forms}")
            
            pairs = generate_pairs_from_forms(result)
            print(f"    Transformation pairs:")
            for p1, p2, dim, v1, v2 in pairs:
                print(f"      '{p1}' ↔ '{p2}'  ({v1} ↔ {v2})")
        else:
            print(f"    Failed: {result.error}")
    
    print("\n" + "-" * 70)
    print("These pairs can be learned into PhiSpace just like verb tenses!")
    print("The geometric delta between 'mailman' and 'mailmen' encodes plurality.")
    print("-" * 70)


# =============================================================================
# EXPERIMENT: SENTENCE TRANSFORMATION WITH MULTIPLE DIMENSIONS
# =============================================================================

def demonstrate_multi_dimensional_transform():
    """Show how multiple dimensions could transform a sentence."""
    
    print("\n" + "=" * 70)
    print("MULTI-DIMENSIONAL SENTENCE TRANSFORMATION")
    print("=" * 70)
    
    sentence = "The mailman quickly delivered the letter."
    print(f"\n  Original: {sentence}")
    print()
    
    # What transformations could we apply?
    transformations = [
        ("mailman", "noun", "number", "singular", "plural", "mailmen"),
        ("quickly", "adverb", "degree", "positive", "superlative", "most quickly"),
        ("delivered", "verb", "tense", "past", "future", "will deliver"),
        ("letter", "noun", "number", "singular", "plural", "letters"),
    ]
    
    print("  Possible transformations:")
    for word, pos, dim, v1, v2, result in transformations:
        print(f"    • {word} → {result}  ({dim}: {v1} → {v2})")
    
    # Apply all transformations
    result = sentence
    for word, pos, dim, v1, v2, new_word in transformations:
        result = result.replace(word, new_word)
    
    print(f"\n  Fully transformed: {result}")
    print()
    print("  Each transformation is a geometric operation:")
    print("    position(word) + Δ(dimension, v1→v2) = position(new_word)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Concept Learning Experiment")
    parser.add_argument("--full", action="store_true", help="Run full experiment")
    parser.add_argument("--demo", action="store_true", help="Run demonstrations only")
    parser.add_argument("--word", type=str, help="Test a specific word")
    parser.add_argument("--pos", type=str, choices=["verb", "noun", "adjective", "adverb", "pronoun"],
                        help="Part of speech for --word")
    
    args = parser.parse_args()
    
    if args.word and args.pos:
        # Test specific word
        pos = PartOfSpeech(args.pos)
        print(f"\nLearning '{args.word}' as {args.pos}...")
        
        for dimension in POS_DIMENSIONS[pos]:
            result = learn_concept_forms(args.word, pos, dimension)
            if result.success:
                print(f"  {dimension}: {result.forms}")
            else:
                print(f"  {dimension}: FAILED - {result.error}")
    
    elif args.demo:
        demonstrate_geometric_learning()
        demonstrate_multi_dimensional_transform()
    
    elif args.full:
        run_experiment()
        demonstrate_geometric_learning()
        demonstrate_multi_dimensional_transform()
    
    else:
        # Default: run demos
        demonstrate_geometric_learning()
        demonstrate_multi_dimensional_transform()
