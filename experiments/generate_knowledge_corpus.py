#!/usr/bin/env python3
"""
Generate Knowledge Corpus

Uses LLM to generate factual knowledge about various topics to enrich
the chatbot's knowledge base. This creates Wikipedia-style content
that can be used for answering questions.
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class Topic:
    """A topic to generate knowledge about."""
    name: str
    category: str
    prompts: List[str]


# Topics to generate knowledge about
TOPICS = [
    # Characters and their relationships
    Topic("Sherlock Holmes", "character", [
        "Describe Sherlock Holmes' personality and methods",
        "Explain Holmes' relationship with Watson",
        "Describe Holmes' famous cases",
    ]),
    Topic("Dr. Watson", "character", [
        "Describe Dr. Watson's role as Holmes' companion",
        "Explain Watson's medical background",
        "Describe Watson's narrative style",
    ]),
    Topic("Professor Moriarty", "character", [
        "Describe Moriarty as Holmes' nemesis",
        "Explain Moriarty's criminal organization",
        "Describe the conflict between Holmes and Moriarty",
    ]),
    
    # Concepts
    Topic("Detective Work", "concept", [
        "Explain deductive reasoning in detective work",
        "Describe the process of solving a mystery",
        "Explain the importance of evidence in investigations",
    ]),
    Topic("Good vs Evil", "concept", [
        "Explain the moral distinction between heroes and villains",
        "Describe what motivates heroic behavior",
        "Explain what drives villainous behavior",
    ]),
    Topic("Leadership", "concept", [
        "Describe the qualities of a good leader",
        "Explain the difference between a king and a tyrant",
        "Describe servant leadership",
    ]),
    Topic("Wisdom", "concept", [
        "Explain what makes someone wise",
        "Describe the role of elders in society",
        "Explain how wisdom differs from intelligence",
    ]),
    
    # Relationships
    Topic("Mentor and Student", "relationship", [
        "Describe the mentor-student relationship",
        "Explain how knowledge is passed between generations",
        "Describe the responsibilities of a mentor",
    ]),
    Topic("Hero and Villain", "relationship", [
        "Explain the dynamic between heroes and villains",
        "Describe how heroes and villains define each other",
        "Explain moral conflict in storytelling",
    ]),
    Topic("Master and Servant", "relationship", [
        "Describe the traditional master-servant relationship",
        "Explain loyalty and duty in service",
        "Describe how power dynamics work in hierarchies",
    ]),
]


def call_ollama(prompt: str, model: str = "qwen2:latest", max_tokens: int = 300) -> str:
    """Call Ollama API."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error: {e}")
        return ""


def generate_knowledge_frame(topic: Topic, prompt: str, model: str = "qwen2:latest") -> Dict:
    """Generate a knowledge frame for a topic."""
    
    full_prompt = f"""Write a concise, factual paragraph about the following topic.
Keep it informative and educational, like a Wikipedia article.

Topic: {topic.name}
Question: {prompt}

Write 2-3 sentences that directly answer the question:"""

    response = call_ollama(full_prompt, model=model)
    
    if not response:
        return None
    
    return {
        "text": response.strip(),
        "agent": topic.name.lower().replace(" ", "_"),
        "source": "knowledge_generation",
        "category": topic.category,
        "topic": topic.name,
    }


def generate_corpus(model: str = "qwen2:latest") -> Dict:
    """Generate the full knowledge corpus."""
    
    print("=" * 70)
    print("GENERATING KNOWLEDGE CORPUS")
    print("=" * 70)
    print(f"\nModel: {model}")
    print(f"Topics: {len(TOPICS)}")
    
    frames = []
    
    for i, topic in enumerate(TOPICS):
        print(f"\n[{i+1}/{len(TOPICS)}] Generating for {topic.name}...")
        
        for prompt in topic.prompts:
            frame = generate_knowledge_frame(topic, prompt, model)
            if frame:
                frames.append(frame)
                print(f"  ✓ {prompt[:40]}...")
            
            time.sleep(0.3)  # Rate limiting
    
    print(f"\nGenerated {len(frames)} knowledge frames")
    
    return {"frames": frames}


def main():
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is running")
    except:
        print("ERROR: Ollama is not running")
        return None
    
    # Generate corpus
    corpus = generate_corpus(model="qwen2:latest")
    
    # Save
    output_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_knowledge.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Show samples
    print("\n--- Sample Frames ---")
    for frame in corpus['frames'][:5]:
        print(f"\n[{frame['agent']}] {frame['text'][:150]}...")
    
    return corpus


if __name__ == "__main__":
    corpus = main()
