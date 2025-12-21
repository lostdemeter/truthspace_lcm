#!/usr/bin/env python3
"""
LLM Data Generator for φ-Based Ingestion

Uses a local LLM (qwen2) to generate co-occurrence-rich data
that feeds our attractor/repeller dynamics.

The key insight: We don't need the LLM to be our chatbot.
We just need it to generate text where related concepts co-occur.
Then our geometric dynamics extract the structure.
"""

import subprocess
import json
import re
from typing import List, Tuple, Generator
from pathlib import Path


def query_ollama(prompt: str, model: str = "qwen2:latest") -> str:
    """Query the local Ollama instance."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return ""


def generate_concept_variations(concept: str, n: int = 10) -> List[str]:
    """
    Generate multiple phrasings of the same concept.
    This creates co-occurrence between related words.
    """
    prompt = f"""List {n} different ways to ask about "{concept}" in a Linux/Unix context.
Just list the questions, one per line, no numbering or bullets.
Keep them short and natural."""
    
    response = query_ollama(prompt)
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    return lines[:n]


def generate_qa_pairs(topic: str, n: int = 5) -> List[Tuple[str, str]]:
    """
    Generate Q&A pairs for a topic.
    The answer should contain the key command/concept.
    """
    prompt = f"""Generate {n} question and answer pairs about "{topic}" in Linux.
Format each as:
Q: [question]
A: [short answer with the command]

Keep answers brief, just the command and a few words."""
    
    response = query_ollama(prompt)
    
    pairs = []
    lines = response.split('\n')
    current_q = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            current_q = line[2:].strip()
        elif line.startswith('A:') and current_q:
            answer = line[2:].strip()
            pairs.append((current_q, answer))
            current_q = None
    
    return pairs


def generate_related_concepts(seed_concept: str) -> List[str]:
    """
    Generate concepts related to a seed concept.
    This helps build the co-occurrence graph.
    """
    prompt = f"""List 10 Linux/Unix concepts closely related to "{seed_concept}".
Just the concept names, one per line, no explanations."""
    
    response = query_ollama(prompt)
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    # Clean up any numbering
    cleaned = []
    for line in lines:
        # Remove leading numbers, dots, dashes
        clean = re.sub(r'^[\d\.\-\)\s]+', '', line).strip()
        if clean:
            cleaned.append(clean)
    return cleaned[:10]


def generate_concept_paragraph(concept: str) -> str:
    """
    Generate a paragraph about a concept.
    This creates rich co-occurrence data.
    """
    prompt = f"""Write a short paragraph (3-4 sentences) explaining "{concept}" in Linux.
Use related technical terms naturally. Be concise."""
    
    return query_ollama(prompt)


def generate_training_corpus(topics: List[str], 
                            variations_per_topic: int = 10,
                            qa_per_topic: int = 5) -> dict:
    """
    Generate a full training corpus from a list of topics.
    
    Returns:
        dict with 'qa_pairs', 'paragraphs', 'variations'
    """
    corpus = {
        'qa_pairs': [],
        'paragraphs': [],
        'variations': {},
        'related_concepts': {},
    }
    
    for topic in topics:
        print(f"Generating data for: {topic}")
        
        # Generate Q&A pairs
        pairs = generate_qa_pairs(topic, qa_per_topic)
        corpus['qa_pairs'].extend(pairs)
        print(f"  Generated {len(pairs)} Q&A pairs")
        
        # Generate variations
        variations = generate_concept_variations(topic, variations_per_topic)
        corpus['variations'][topic] = variations
        print(f"  Generated {len(variations)} variations")
        
        # Generate paragraph
        paragraph = generate_concept_paragraph(topic)
        if paragraph:
            corpus['paragraphs'].append(paragraph)
            print(f"  Generated paragraph")
        
        # Generate related concepts
        related = generate_related_concepts(topic)
        corpus['related_concepts'][topic] = related
        print(f"  Found {len(related)} related concepts")
    
    return corpus


def quick_test():
    """Quick test of the LLM connection."""
    print("Testing Ollama connection...")
    response = query_ollama("What command lists files in Linux? Answer in 5 words or less.")
    print(f"Response: {response}")
    return bool(response)


def demo():
    """Demo the data generation."""
    print("=" * 70)
    print("LLM DATA GENERATOR DEMO")
    print("=" * 70)
    
    if not quick_test():
        print("Failed to connect to Ollama. Is it running?")
        return
    
    # Test topics
    topics = [
        "listing files",
        "disk space",
        "running processes",
        "network connections",
        "user management",
    ]
    
    print("\n" + "=" * 70)
    print("GENERATING TRAINING DATA")
    print("=" * 70)
    
    corpus = generate_training_corpus(
        topics,
        variations_per_topic=5,
        qa_per_topic=3
    )
    
    # Save corpus
    output_path = Path(__file__).parent / "openai_data" / "llm_generated_corpus.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\nSaved corpus to: {output_path}")
    print(f"  Q&A pairs: {len(corpus['qa_pairs'])}")
    print(f"  Paragraphs: {len(corpus['paragraphs'])}")
    print(f"  Topics with variations: {len(corpus['variations'])}")
    
    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    
    print("\nQ&A Pairs:")
    for q, a in corpus['qa_pairs'][:5]:
        print(f"  Q: {q}")
        print(f"  A: {a}\n")
    
    print("\nVariations for 'listing files':")
    for v in corpus['variations'].get('listing files', [])[:5]:
        print(f"  - {v}")
    
    print("\nRelated concepts for 'disk space':")
    for c in corpus['related_concepts'].get('disk space', [])[:5]:
        print(f"  - {c}")


if __name__ == "__main__":
    demo()
