#!/usr/bin/env python3
"""
Live LLM Corpus Generator

Uses a real LLM (via Ollama) to generate behavioral sentences for characters.
The LLM generates natural language that encodes dimensional properties through behavior.

This is the key difference from template-based generation:
- Templates: "Holmes investigates the mystery"
- LLM: "Holmes scrutinizes the faded letter, his keen eyes catching details others would miss"

The LLM produces richer, more varied behavioral descriptions that should
reveal more nuanced dimensions.
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
import re


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class CharacterPrompt:
    """A character to generate sentences for."""
    name: str
    description: str
    behavioral_hints: List[str]


# Characters with descriptions (no hidden numeric properties - let the LLM decide)
CHARACTERS = [
    # === LITERARY CHARACTERS ===
    CharacterPrompt(
        "Sherlock Holmes",
        "A brilliant detective known for deductive reasoning and keen observation",
        ["investigates", "deduces", "observes", "solves mysteries", "analyzes evidence"]
    ),
    CharacterPrompt(
        "Dr. Watson",
        "A loyal companion and chronicler, a doctor who assists Holmes",
        ["assists", "documents", "supports", "accompanies", "provides medical expertise"]
    ),
    CharacterPrompt(
        "Professor Moriarty",
        "A criminal mastermind, Holmes's arch-nemesis",
        ["schemes", "plots", "manipulates", "threatens", "controls criminal networks"]
    ),
    CharacterPrompt(
        "Irene Adler",
        "A clever adventuress who outwitted Holmes",
        ["outwits", "escapes", "challenges", "disguises", "defies expectations"]
    ),
    CharacterPrompt(
        "Alice",
        "A curious young girl exploring Wonderland",
        ["explores", "questions", "wonders", "grows", "adapts to strange situations"]
    ),
    CharacterPrompt(
        "The Queen of Hearts",
        "A tyrannical ruler of Wonderland",
        ["commands", "orders executions", "rules", "demands obedience", "intimidates"]
    ),
    CharacterPrompt(
        "Elizabeth Bennet",
        "A witty and independent-minded young woman",
        ["challenges", "debates", "refuses", "judges character", "speaks her mind"]
    ),
    CharacterPrompt(
        "Mr. Darcy",
        "A proud but honorable gentleman who learns humility",
        ["observes", "proposes", "changes", "rescues", "overcomes pride"]
    ),
    CharacterPrompt(
        "Hamlet",
        "A brooding prince contemplating revenge and mortality",
        ["contemplates", "hesitates", "questions", "schemes", "suffers"]
    ),
    CharacterPrompt(
        "Lady Macbeth",
        "An ambitious woman who drives her husband to murder",
        ["manipulates", "schemes", "commands", "suffers guilt", "descends into madness"]
    ),
    CharacterPrompt(
        "Don Quixote",
        "A delusional knight errant tilting at windmills",
        ["imagines", "charges", "dreams", "fights", "believes in chivalry"]
    ),
    CharacterPrompt(
        "Sancho Panza",
        "A loyal squire who follows his master faithfully",
        ["follows", "advises", "doubts", "supports", "provides common sense"]
    ),
    
    # === ARCHETYPES - POWER/AUTHORITY ===
    CharacterPrompt(
        "A powerful king",
        "A monarch who rules with authority",
        ["commands", "decrees", "judges", "leads armies", "makes decisions"]
    ),
    CharacterPrompt(
        "A wise queen",
        "A monarch who rules with wisdom and diplomacy",
        ["advises", "negotiates", "rules", "protects", "makes alliances"]
    ),
    CharacterPrompt(
        "A humble servant",
        "A person who serves others faithfully",
        ["serves", "obeys", "waits", "assists", "follows orders"]
    ),
    CharacterPrompt(
        "A rebel leader",
        "A revolutionary fighting against tyranny",
        ["rebels", "inspires", "fights", "organizes", "challenges authority"]
    ),
    CharacterPrompt(
        "A corrupt politician",
        "A leader who abuses power for personal gain",
        ["manipulates", "lies", "bribes", "schemes", "betrays trust"]
    ),
    CharacterPrompt(
        "A just judge",
        "A fair arbiter of law and justice",
        ["judges", "deliberates", "sentences", "listens", "weighs evidence"]
    ),
    
    # === ARCHETYPES - AGE ===
    CharacterPrompt(
        "A young child",
        "An innocent child learning about the world",
        ["plays", "learns", "asks questions", "imagines", "trusts adults"]
    ),
    CharacterPrompt(
        "A rebellious teenager",
        "A young person challenging authority and finding identity",
        ["rebels", "questions", "experiments", "argues", "seeks independence"]
    ),
    CharacterPrompt(
        "An elderly sage",
        "A wise elder who has seen much of life",
        ["advises", "remembers", "teaches", "reflects", "shares wisdom"]
    ),
    CharacterPrompt(
        "A middle-aged parent",
        "A caretaker balancing work and family",
        ["provides", "protects", "worries", "sacrifices", "nurtures"]
    ),
    
    # === ARCHETYPES - MORALITY ===
    CharacterPrompt(
        "A cunning villain",
        "An antagonist who pursues selfish goals through deception",
        ["deceives", "betrays", "schemes", "threatens", "manipulates"]
    ),
    CharacterPrompt(
        "A brave hero",
        "A protagonist who fights for justice",
        ["fights", "protects", "rescues", "confronts evil", "inspires others"]
    ),
    CharacterPrompt(
        "A fallen angel",
        "A once-good being corrupted by pride or despair",
        ["tempts", "regrets", "corrupts", "remembers glory", "seeks redemption"]
    ),
    CharacterPrompt(
        "A reformed criminal",
        "A former wrongdoer seeking redemption",
        ["atones", "helps others", "struggles", "resists temptation", "rebuilds"]
    ),
    CharacterPrompt(
        "A righteous priest",
        "A religious leader devoted to helping others",
        ["prays", "counsels", "forgives", "guides", "sacrifices"]
    ),
    
    # === ARCHETYPES - AGENCY ===
    CharacterPrompt(
        "A decisive general",
        "A military leader who commands with confidence",
        ["commands", "strategizes", "leads", "decides", "inspires troops"]
    ),
    CharacterPrompt(
        "A passive victim",
        "Someone who suffers at the hands of others",
        ["suffers", "endures", "waits", "hopes", "fears"]
    ),
    CharacterPrompt(
        "An active explorer",
        "An adventurer who seeks new discoveries",
        ["explores", "discovers", "maps", "ventures", "risks"]
    ),
    CharacterPrompt(
        "A patient healer",
        "A caregiver who tends to the sick and wounded",
        ["heals", "comforts", "tends", "listens", "nurtures"]
    ),
    
    # === NON-HUMAN ENTITIES ===
    CharacterPrompt(
        "A robot",
        "An artificial being that follows programming",
        ["computes", "executes commands", "processes data", "serves humans", "follows logic"]
    ),
    CharacterPrompt(
        "A raging storm",
        "A powerful force of nature",
        ["destroys", "rages", "floods", "threatens", "overwhelms"]
    ),
    CharacterPrompt(
        "An abstract idea",
        "A concept that influences human thought",
        ["spreads", "inspires", "transforms", "persists", "evolves"]
    ),
    CharacterPrompt(
        "A loyal dog",
        "A faithful animal companion",
        ["follows", "protects", "loves unconditionally", "obeys", "senses danger"]
    ),
    CharacterPrompt(
        "A cunning fox",
        "A clever animal that survives through wit",
        ["hunts", "evades", "tricks", "survives", "adapts"]
    ),
    CharacterPrompt(
        "A mighty river",
        "A force of nature that shapes the land",
        ["flows", "carves", "nourishes", "floods", "persists"]
    ),
    CharacterPrompt(
        "An ancient tree",
        "A living monument that has witnessed centuries",
        ["grows", "shelters", "endures", "witnesses", "provides"]
    ),
    CharacterPrompt(
        "A spreading fire",
        "A destructive force that consumes everything",
        ["burns", "spreads", "consumes", "destroys", "illuminates"]
    ),
    
    # === EMOTIONAL STATES ===
    CharacterPrompt(
        "A grieving widow",
        "A woman mourning her lost husband",
        ["mourns", "remembers", "endures", "struggles", "finds strength"]
    ),
    CharacterPrompt(
        "A joyful bride",
        "A woman celebrating her wedding day",
        ["celebrates", "smiles", "dances", "embraces", "dreams"]
    ),
    CharacterPrompt(
        "An angry mob",
        "A crowd driven by collective rage",
        ["shouts", "demands", "threatens", "destroys", "overwhelms"]
    ),
    CharacterPrompt(
        "A fearful refugee",
        "A displaced person fleeing danger",
        ["flees", "hides", "hopes", "struggles", "survives"]
    ),
    
    # === PROFESSIONAL ROLES ===
    CharacterPrompt(
        "A mysterious stranger",
        "An unknown person with unclear motives",
        ["watches", "waits", "reveals little", "appears unexpectedly", "keeps secrets"]
    ),
    CharacterPrompt(
        "A skilled craftsman",
        "An artisan who creates with expertise",
        ["crafts", "builds", "repairs", "teaches", "perfects"]
    ),
    CharacterPrompt(
        "A traveling merchant",
        "A trader who journeys to sell goods",
        ["trades", "bargains", "travels", "negotiates", "profits"]
    ),
    CharacterPrompt(
        "A devoted scholar",
        "An academic dedicated to knowledge",
        ["studies", "researches", "writes", "teaches", "discovers"]
    ),
    CharacterPrompt(
        "A cunning spy",
        "A secret agent gathering intelligence",
        ["infiltrates", "observes", "reports", "deceives", "escapes"]
    ),
    CharacterPrompt(
        "A weary soldier",
        "A warrior tired from battle",
        ["fights", "endures", "follows orders", "protects comrades", "survives"]
    ),
]


def call_ollama(prompt: str, model: str = "qwen2:latest", max_tokens: int = 500) -> Optional[str]:
    """Call Ollama API to generate text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.8,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def generate_character_sentences(char: CharacterPrompt, n_sentences: int = 10, model: str = "qwen2:latest") -> List[str]:
    """Generate behavioral sentences for a character using LLM."""
    
    # Use a simple, consistent name for the agent
    simple_name = char.name.split()[-1] if ' ' in char.name else char.name
    simple_name = simple_name.replace("The ", "").replace("A ", "").replace("An ", "")
    
    prompt = f"""Generate {n_sentences} sentences showing {simple_name} performing actions.

Character: {char.name}
Description: {char.description}

IMPORTANT RULES:
1. EVERY sentence MUST start with "{simple_name}" (the character's name)
2. The second word should be a verb (action word)
3. Keep sentences simple: "{simple_name} [verb] [object/description]"
4. Use different verbs in each sentence
5. No pronouns like "he", "she", "they" - always use the name

Examples of good format:
- "{simple_name} investigates the crime scene carefully"
- "{simple_name} commands the soldiers to advance"
- "{simple_name} waits patiently for instructions"

Generate {n_sentences} sentences, one per line:"""

    response = call_ollama(prompt, model=model)
    
    if not response:
        return []
    
    # Parse sentences from response
    sentences = []
    for line in response.strip().split('\n'):
        line = line.strip()
        # Remove numbering if present
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if line and len(line) > 10:
            sentences.append(line)
    
    return sentences[:n_sentences]


def extract_agent_from_sentence(sentence: str, char_name: str) -> str:
    """Extract the agent name from a sentence."""
    # Use the simple name we generated
    simple_name = char_name.split()[-1] if ' ' in char_name else char_name
    simple_name = simple_name.replace("The ", "").replace("A ", "").replace("An ", "")
    simple_name = simple_name.lower()
    
    # Check if sentence starts with the expected name
    first_word = sentence.split()[0].lower() if sentence.split() else ""
    first_word = re.sub(r'[^a-z]', '', first_word)
    
    if first_word == simple_name.lower():
        return simple_name
    
    # Check for character name anywhere
    sentence_lower = sentence.lower()
    if simple_name in sentence_lower:
        return simple_name
    
    # Fallback to canonical names
    canonical = {
        # Literary characters
        'sherlock': 'holmes', 'holmes': 'holmes',
        'watson': 'watson', 'dr': 'watson',
        'moriarty': 'moriarty', 'professor': 'moriarty',
        'irene': 'irene', 'adler': 'irene',
        'alice': 'alice',
        'queen': 'queen', 'hearts': 'queen',
        'elizabeth': 'elizabeth', 'bennet': 'elizabeth',
        'darcy': 'darcy',
        'hamlet': 'hamlet', 'prince': 'hamlet',
        'macbeth': 'macbeth', 'lady': 'macbeth',
        'quixote': 'quixote', 'don': 'quixote',
        'sancho': 'sancho', 'panza': 'sancho',
        # Power/Authority
        'king': 'king', 'monarch': 'king', 'powerful': 'king',
        'wise': 'queen',
        'servant': 'servant', 'humble': 'servant',
        'rebel': 'rebel', 'leader': 'rebel',
        'politician': 'politician', 'corrupt': 'politician',
        'judge': 'judge', 'just': 'judge',
        # Age
        'child': 'child', 'young': 'child',
        'teenager': 'teenager', 'rebellious': 'teenager',
        'sage': 'sage', 'elderly': 'sage', 'elder': 'sage',
        'parent': 'parent', 'middle': 'parent',
        # Morality
        'villain': 'villain', 'cunning': 'villain',
        'hero': 'hero', 'brave': 'hero',
        'angel': 'angel', 'fallen': 'angel',
        'criminal': 'criminal', 'reformed': 'criminal',
        'priest': 'priest', 'righteous': 'priest',
        # Agency
        'general': 'general', 'decisive': 'general',
        'victim': 'victim', 'passive': 'victim',
        'explorer': 'explorer', 'active': 'explorer',
        'healer': 'healer', 'patient': 'healer',
        # Non-human
        'robot': 'robot',
        'storm': 'storm', 'raging': 'storm',
        'idea': 'idea', 'abstract': 'idea',
        'dog': 'dog', 'loyal': 'dog',
        'fox': 'fox',
        'river': 'river', 'mighty': 'river',
        'tree': 'tree', 'ancient': 'tree',
        'fire': 'fire', 'spreading': 'fire',
        # Emotional
        'widow': 'widow', 'grieving': 'widow',
        'bride': 'bride', 'joyful': 'bride',
        'mob': 'mob', 'angry': 'mob',
        'refugee': 'refugee', 'fearful': 'refugee',
        # Professional
        'stranger': 'stranger', 'mysterious': 'stranger',
        'craftsman': 'craftsman', 'skilled': 'craftsman',
        'merchant': 'merchant', 'traveling': 'merchant',
        'scholar': 'scholar', 'devoted': 'scholar',
        'spy': 'spy',
        'soldier': 'soldier', 'weary': 'soldier',
    }
    
    for key, agent in canonical.items():
        if key in sentence_lower:
            return agent
    
    return simple_name


def generate_corpus(n_sentences_per_char: int = 15, model: str = "qwen2:latest") -> Dict:
    """Generate full corpus using LLM."""
    
    print("=" * 70)
    print("LIVE LLM CORPUS GENERATION")
    print("=" * 70)
    print(f"\nModel: {model}")
    print(f"Characters: {len(CHARACTERS)}")
    print(f"Sentences per character: {n_sentences_per_char}")
    print(f"Expected total: {len(CHARACTERS) * n_sentences_per_char} frames")
    print()
    
    frames = []
    
    for i, char in enumerate(CHARACTERS):
        print(f"[{i+1}/{len(CHARACTERS)}] Generating for {char.name}...", end=" ", flush=True)
        
        sentences = generate_character_sentences(char, n_sentences_per_char, model)
        
        for sentence in sentences:
            agent = extract_agent_from_sentence(sentence, char.name)
            frames.append({
                "text": sentence,
                "source": f"llm_generated_{model}",
                "agent": agent,
                "character": char.name,
            })
        
        print(f"got {len(sentences)} sentences")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    print(f"\nTotal frames generated: {len(frames)}")
    
    return {"frames": frames}


def main():
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is running")
    except:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return None
    
    # Generate corpus
    corpus = generate_corpus(n_sentences_per_char=15, model="qwen2:latest")
    
    # Save
    output_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Show samples
    print("\n--- Sample Frames ---")
    for frame in corpus['frames'][:20]:
        print(f"  [{frame['agent']}] {frame['text'][:70]}...")
    
    return corpus


if __name__ == "__main__":
    corpus = main()
