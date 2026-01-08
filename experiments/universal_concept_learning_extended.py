#!/usr/bin/env python3
"""
Universal Concept Learning - Extended Dimensions

Testing if we can learn concepts across ALL semantic dimensions,
not just grammatical ones (tense, number, degree).

Semantic Dimensions to Test:
============================

1. REGALITY: common → noble → royal
   - "man" → "gentleman" → "lord" → "king"
   - "woman" → "lady" → "duchess" → "queen"
   - "house" → "manor" → "palace"

2. FORMALITY: casual → neutral → formal
   - "hi" → "hello" → "greetings"
   - "yeah" → "yes" → "indeed"
   - "kid" → "child" → "youth"

3. INTENSITY: mild → moderate → intense
   - "warm" → "hot" → "scorching"
   - "dislike" → "hate" → "despise"
   - "happy" → "joyful" → "ecstatic"

4. POLARITY: negative → neutral → positive
   - "terrible" → "okay" → "excellent"
   - "hate" → "indifferent" → "love"
   - "ugly" → "plain" → "beautiful"

5. SPECIFICITY: general → specific → precise
   - "animal" → "dog" → "golden retriever"
   - "vehicle" → "car" → "Tesla Model 3"
   - "food" → "fruit" → "apple"

6. VOICE: active → passive (for verbs/sentences)
   - "The dog bit the man" → "The man was bitten by the dog"

7. CERTAINTY: uncertain → neutral → certain
   - "might" → "could" → "will"
   - "perhaps" → "probably" → "definitely"

8. EMOTION: sad → neutral → happy
   - "weep" → "cry" → "sob" (intensity within sad)
   - "frown" → "neutral" → "smile"

The goal: Can the LLM generate meaningful transformation pairs
for ANY of these dimensions, for ANY word?
"""

import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# SEMANTIC DIMENSIONS
# =============================================================================

SEMANTIC_DIMENSIONS = {
    "regality": {
        "values": ["common", "noble", "royal"],
        "description": "social status from common person to royalty",
        "examples": [
            ("man", {"common": "man", "noble": "gentleman", "royal": "king"}),
            ("woman", {"common": "woman", "noble": "lady", "royal": "queen"}),
            ("house", {"common": "house", "noble": "manor", "royal": "palace"}),
        ]
    },
    "formality": {
        "values": ["casual", "neutral", "formal"],
        "description": "register from informal to formal speech",
        "examples": [
            ("hello", {"casual": "hi", "neutral": "hello", "formal": "greetings"}),
            ("yes", {"casual": "yeah", "neutral": "yes", "formal": "indeed"}),
            ("child", {"casual": "kid", "neutral": "child", "formal": "youth"}),
        ]
    },
    "intensity": {
        "values": ["mild", "moderate", "intense"],
        "description": "strength or degree of the concept",
        "examples": [
            ("hot", {"mild": "warm", "moderate": "hot", "intense": "scorching"}),
            ("happy", {"mild": "content", "moderate": "happy", "intense": "ecstatic"}),
            ("angry", {"mild": "annoyed", "moderate": "angry", "intense": "furious"}),
        ]
    },
    "polarity": {
        "values": ["negative", "neutral", "positive"],
        "description": "sentiment from negative to positive",
        "examples": [
            ("quality", {"negative": "terrible", "neutral": "okay", "positive": "excellent"}),
            ("feeling", {"negative": "hate", "neutral": "indifferent", "positive": "love"}),
            ("appearance", {"negative": "ugly", "neutral": "plain", "positive": "beautiful"}),
        ]
    },
    "specificity": {
        "values": ["general", "specific", "precise"],
        "description": "level of detail from broad category to exact instance",
        "examples": [
            ("dog", {"general": "animal", "specific": "dog", "precise": "golden retriever"}),
            ("car", {"general": "vehicle", "specific": "car", "precise": "sedan"}),
            ("apple", {"general": "food", "specific": "fruit", "precise": "apple"}),
        ]
    },
    "certainty": {
        "values": ["uncertain", "neutral", "certain"],
        "description": "degree of confidence or probability",
        "examples": [
            ("will", {"uncertain": "might", "neutral": "could", "certain": "will"}),
            ("probably", {"uncertain": "perhaps", "neutral": "probably", "certain": "definitely"}),
            ("think", {"uncertain": "guess", "neutral": "think", "certain": "know"}),
        ]
    },
    "emotion": {
        "values": ["sad", "neutral", "happy"],
        "description": "emotional valence",
        "examples": [
            ("expression", {"sad": "frown", "neutral": "neutral expression", "happy": "smile"}),
            ("cry", {"sad": "weep", "neutral": "cry", "happy": "cry tears of joy"}),
            ("mood", {"sad": "melancholy", "neutral": "calm", "happy": "joyful"}),
        ]
    },
    "size": {
        "values": ["small", "medium", "large"],
        "description": "physical or metaphorical size",
        "examples": [
            ("dog", {"small": "puppy", "medium": "dog", "large": "hound"}),
            ("house", {"small": "cottage", "medium": "house", "large": "mansion"}),
            ("problem", {"small": "issue", "medium": "problem", "large": "crisis"}),
        ]
    },
    "speed": {
        "values": ["slow", "medium", "fast"],
        "description": "rate of motion or action",
        "examples": [
            ("walk", {"slow": "stroll", "medium": "walk", "fast": "stride"}),
            ("run", {"slow": "jog", "medium": "run", "fast": "sprint"}),
            ("move", {"slow": "crawl", "medium": "move", "fast": "dash"}),
        ]
    },
    "age": {
        "values": ["young", "adult", "old"],
        "description": "life stage or age",
        "examples": [
            ("person", {"young": "child", "adult": "adult", "old": "elder"}),
            ("dog", {"young": "puppy", "adult": "dog", "old": "old dog"}),
            ("tree", {"young": "sapling", "adult": "tree", "old": "ancient tree"}),
        ]
    },
}

# =============================================================================
# LLM QUERY
# =============================================================================

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
        }, timeout=60)
        
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
    
    text = response.strip()
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None

# =============================================================================
# DIMENSION-SPECIFIC PROMPTS
# =============================================================================

def get_semantic_prompt(word: str, dimension: str) -> str:
    """Generate a prompt for learning a word's forms along a semantic dimension."""
    
    dim_info = SEMANTIC_DIMENSIONS[dimension]
    values = dim_info["values"]
    description = dim_info["description"]
    examples = dim_info["examples"]
    
    # Format examples
    example_strs = []
    for ex_word, ex_forms in examples:
        forms_str = ", ".join(f'"{k}": "{v}"' for k, v in ex_forms.items())
        example_strs.append(f'  "{ex_word}": {{{forms_str}}}')
    examples_text = "\n".join(example_strs)
    
    values_str = ", ".join(values)
    
    prompt = f"""The dimension "{dimension}" represents {description}.
The values along this dimension are: {values_str}.

Examples:
{examples_text}

Now, give me the {dimension} forms of the word/concept "{word}".
Return ONLY a JSON object with keys: {values_str}.
If a form doesn't naturally exist, use a descriptive phrase.

For "{word}":"""
    
    return prompt


@dataclass
class SemanticForms:
    """Forms of a concept along a semantic dimension."""
    word: str
    dimension: str
    forms: Dict[str, str]
    success: bool
    error: Optional[str] = None


def learn_semantic_forms(word: str, dimension: str) -> SemanticForms:
    """Learn the forms of a word along a semantic dimension."""
    
    prompt = get_semantic_prompt(word, dimension)
    response = query_llm(prompt)
    
    if not response:
        return SemanticForms(word, dimension, {}, False, "LLM query failed")
    
    forms = parse_json_response(response)
    
    if not forms:
        return SemanticForms(word, dimension, {}, False, f"Failed to parse: {response[:100]}")
    
    # Validate keys
    expected_keys = set(SEMANTIC_DIMENSIONS[dimension]["values"])
    actual_keys = set(forms.keys())
    
    if not expected_keys.issubset(actual_keys):
        missing = expected_keys - actual_keys
        return SemanticForms(word, dimension, forms, False, f"Missing keys: {missing}")
    
    return SemanticForms(word, dimension, forms, True)


# =============================================================================
# EXPERIMENT: TEST ALL SEMANTIC DIMENSIONS
# =============================================================================

def run_semantic_experiment():
    """Test concept learning across all semantic dimensions."""
    
    print("=" * 70)
    print("UNIVERSAL SEMANTIC CONCEPT LEARNING")
    print("=" * 70)
    print()
    
    # Test words - a variety of nouns, verbs, adjectives
    test_words = [
        "walk",       # verb
        "house",      # noun
        "happy",      # adjective
        "speak",      # verb
        "person",     # noun
        "fast",       # adjective/adverb
        "eat",        # verb
        "car",        # noun
        "angry",      # adjective
        "think",      # verb
    ]
    
    results = {}
    
    for dimension in SEMANTIC_DIMENSIONS:
        print(f"\n{'='*70}")
        print(f"DIMENSION: {dimension.upper()}")
        print(f"Description: {SEMANTIC_DIMENSIONS[dimension]['description']}")
        print(f"Values: {SEMANTIC_DIMENSIONS[dimension]['values']}")
        print(f"{'='*70}")
        
        dim_results = []
        successes = 0
        
        for word in test_words:
            result = learn_semantic_forms(word, dimension)
            dim_results.append(result)
            
            if result.success:
                successes += 1
                forms_str = " → ".join(f"{v}" for v in result.forms.values())
                print(f"  ✓ {word:12s}: {forms_str}")
            else:
                print(f"  ✗ {word:12s}: {result.error}")
        
        results[dimension] = dim_results
        print(f"\n  Success rate: {successes}/{len(test_words)} ({successes/len(test_words)*100:.0f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY DIMENSION")
    print("=" * 70)
    
    total_tests = 0
    total_success = 0
    
    for dimension, dim_results in results.items():
        successes = sum(1 for r in dim_results if r.success)
        total = len(dim_results)
        total_tests += total
        total_success += successes
        pct = (successes / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {dimension:12s}: {bar} {successes}/{total} ({pct:.0f}%)")
    
    print(f"\n  {'TOTAL':12s}: {total_success}/{total_tests} ({total_success/total_tests*100:.0f}%)")
    
    return results


# =============================================================================
# EXPERIMENT: CROSS-DIMENSIONAL TRANSFORMATION
# =============================================================================

def demonstrate_cross_dimensional():
    """Show how a single word can transform along multiple dimensions."""
    
    print("\n" + "=" * 70)
    print("CROSS-DIMENSIONAL TRANSFORMATION")
    print("=" * 70)
    print()
    print("Can the same word transform along MULTIPLE dimensions?")
    print()
    
    # Test a single word across all dimensions
    test_word = "house"
    
    print(f"  Word: '{test_word}'")
    print(f"  {'-'*60}")
    
    for dimension in SEMANTIC_DIMENSIONS:
        result = learn_semantic_forms(test_word, dimension)
        
        if result.success:
            forms_str = " | ".join(f"{k}={v}" for k, v in result.forms.items())
            print(f"    {dimension:12s}: {forms_str}")
        else:
            print(f"    {dimension:12s}: (not applicable)")
    
    print()
    print("  This shows that 'house' exists in a multi-dimensional space!")
    print("  Each dimension is an axis, and the word has a position on each.")


# =============================================================================
# EXPERIMENT: SENTENCE TRANSFORMATION
# =============================================================================

def demonstrate_sentence_transformation():
    """Show how sentences can be transformed along semantic dimensions."""
    
    print("\n" + "=" * 70)
    print("SENTENCE TRANSFORMATION ACROSS DIMENSIONS")
    print("=" * 70)
    
    sentence = "The man walked to his house."
    print(f"\n  Original: {sentence}")
    print()
    
    # Learn transformations for key words
    transformations = {
        "regality": [("man", "king"), ("house", "palace")],
        "formality": [("walked", "proceeded"), ("house", "residence")],
        "intensity": [("walked", "strode")],
        "speed": [("walked", "sprinted")],
        "size": [("house", "mansion")],
    }
    
    print("  Transformations by dimension:")
    for dim, changes in transformations.items():
        result = sentence
        for old, new in changes:
            result = result.replace(old, new)
        print(f"    {dim:12s}: {result}")
    
    print()
    print("  Each transformation is geometric: position + Δ(dimension) = new_position")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Semantic Concept Learning")
    parser.add_argument("--full", action="store_true", help="Run full experiment on all dimensions")
    parser.add_argument("--dimension", type=str, help="Test specific dimension")
    parser.add_argument("--word", type=str, help="Test specific word")
    parser.add_argument("--cross", action="store_true", help="Show cross-dimensional demo")
    parser.add_argument("--sentence", action="store_true", help="Show sentence transformation demo")
    
    args = parser.parse_args()
    
    if args.word and args.dimension:
        # Test specific word on specific dimension
        result = learn_semantic_forms(args.word, args.dimension)
        if result.success:
            print(f"\n'{args.word}' along {args.dimension}:")
            for k, v in result.forms.items():
                print(f"  {k}: {v}")
        else:
            print(f"Failed: {result.error}")
    
    elif args.dimension:
        # Test all words on specific dimension
        print(f"\nTesting dimension: {args.dimension}")
        test_words = ["walk", "house", "happy", "person", "fast"]
        for word in test_words:
            result = learn_semantic_forms(word, args.dimension)
            if result.success:
                print(f"  {word}: {result.forms}")
            else:
                print(f"  {word}: FAILED")
    
    elif args.cross:
        demonstrate_cross_dimensional()
    
    elif args.sentence:
        demonstrate_sentence_transformation()
    
    elif args.full:
        run_semantic_experiment()
        demonstrate_cross_dimensional()
        demonstrate_sentence_transformation()
    
    else:
        # Default: quick demo
        print("\nQuick demo - testing a few dimensions...")
        print("Run with --full for complete experiment\n")
        
        demos = [
            ("house", "regality"),
            ("happy", "intensity"),
            ("walk", "speed"),
            ("person", "age"),
            ("speak", "formality"),
        ]
        
        for word, dim in demos:
            result = learn_semantic_forms(word, dim)
            if result.success:
                forms_str = " → ".join(result.forms.values())
                print(f"  {word} ({dim}): {forms_str}")
            else:
                print(f"  {word} ({dim}): FAILED - {result.error}")
