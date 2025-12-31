#!/usr/bin/env python3
"""
Corpus Builder for Emergent Dimension Discovery

This script builds a clean corpus optimized for discovering semantic dimensions
like gender, age, agency, and animacy.

Strategy:
1. Use LLM to generate structured frames with known properties
2. Ensure balanced coverage across all dimensions
3. Generate frames that make dimensions discoverable from behavior

The key insight: To discover dimensions, we need data where those dimensions
are expressed through BEHAVIOR (what agents do), not just labels.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import defaultdict
import random


@dataclass
class ConceptSeed:
    """A seed concept with known dimensional properties."""
    name: str
    gender: float  # -1 = male, +1 = female, 0 = neutral
    age: float     # -1 = young, +1 = old, 0 = adult
    agency: float  # -1 = passive, +1 = active
    animacy: float # -1 = abstract, +1 = human/animate
    
    # Behavioral properties
    typical_actions: List[str] = field(default_factory=list)
    typical_targets: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)


# Seed concepts with known dimensional properties
CONCEPT_SEEDS = [
    # Sherlock Holmes universe
    ConceptSeed("holmes", gender=-1, age=0, agency=1, animacy=1,
                typical_actions=["investigates", "deduces", "examines", "solves", "observes", "discovers"],
                typical_targets=["mystery", "crime", "evidence", "clue", "case"],
                roles=["detective", "investigator"]),
    
    ConceptSeed("watson", gender=-1, age=0, agency=0.3, animacy=1,
                typical_actions=["assists", "documents", "accompanies", "supports", "watches", "records"],
                typical_targets=["holmes", "case", "adventure", "patient"],
                roles=["doctor", "companion", "chronicler"]),
    
    ConceptSeed("moriarty", gender=-1, age=0.5, agency=0.8, animacy=1,
                typical_actions=["schemes", "plots", "manipulates", "controls", "threatens"],
                typical_targets=["holmes", "crime", "organization", "plan"],
                roles=["professor", "criminal", "mastermind"]),
    
    ConceptSeed("irene", gender=1, age=0, agency=0.9, animacy=1,
                typical_actions=["outwits", "escapes", "disguises", "challenges", "defeats"],
                typical_targets=["holmes", "king", "photograph"],
                roles=["adventuress", "singer"]),
    
    ConceptSeed("lestrade", gender=-1, age=0.3, agency=0.5, animacy=1,
                typical_actions=["arrests", "investigates", "questions", "follows", "consults"],
                typical_targets=["suspect", "criminal", "holmes"],
                roles=["inspector", "detective"]),
    
    # Alice in Wonderland universe
    ConceptSeed("alice", gender=1, age=-0.7, agency=0.6, animacy=1,
                typical_actions=["explores", "questions", "grows", "shrinks", "wonders", "discovers"],
                typical_targets=["wonderland", "rabbit", "queen", "mystery"],
                roles=["girl", "explorer", "dreamer"]),
    
    ConceptSeed("queen", gender=1, age=0.5, agency=0.9, animacy=1,
                typical_actions=["commands", "orders", "shouts", "rules", "threatens", "demands"],
                typical_targets=["subjects", "alice", "cards", "court"],
                roles=["queen", "ruler", "tyrant"]),
    
    ConceptSeed("hatter", gender=-1, age=0, agency=0.4, animacy=1,
                typical_actions=["laughs", "riddles", "hosts", "confuses", "chatters"],
                typical_targets=["tea", "party", "alice", "time"],
                roles=["hatter", "host"]),
    
    ConceptSeed("cheshire", gender=0, age=0, agency=0.5, animacy=0.8,
                typical_actions=["grins", "vanishes", "appears", "advises", "confuses"],
                typical_targets=["alice", "path", "direction"],
                roles=["cat", "guide"]),
    
    # Pride and Prejudice universe
    ConceptSeed("elizabeth", gender=1, age=-0.2, agency=0.8, animacy=1,
                typical_actions=["challenges", "refuses", "debates", "walks", "reads", "judges"],
                typical_targets=["darcy", "society", "prejudice", "book"],
                roles=["lady", "daughter", "sister"]),
    
    ConceptSeed("darcy", gender=-1, age=0, agency=0.7, animacy=1,
                typical_actions=["proposes", "rescues", "broods", "observes", "changes"],
                typical_targets=["elizabeth", "wickham", "estate", "sister"],
                roles=["gentleman", "landowner", "suitor"]),
    
    ConceptSeed("jane", gender=1, age=-0.1, agency=0.3, animacy=1,
                typical_actions=["loves", "forgives", "hopes", "waits", "smiles"],
                typical_targets=["bingley", "sister", "happiness"],
                roles=["lady", "sister", "bride"]),
    
    ConceptSeed("bingley", gender=-1, age=-0.1, agency=0.4, animacy=1,
                typical_actions=["dances", "smiles", "loves", "leaves", "returns"],
                typical_targets=["jane", "ball", "estate"],
                roles=["gentleman", "suitor"]),
    
    ConceptSeed("wickham", gender=-1, age=0, agency=0.6, animacy=1,
                typical_actions=["deceives", "charms", "elopes", "lies", "gambles"],
                typical_targets=["lydia", "darcy", "money", "women"],
                roles=["officer", "scoundrel"]),
    
    ConceptSeed("lydia", gender=1, age=-0.5, agency=0.5, animacy=1,
                typical_actions=["flirts", "laughs", "elopes", "dances", "gossips"],
                typical_targets=["officers", "wickham", "balls"],
                roles=["girl", "sister"]),
    
    # Generic archetypes for dimension coverage
    ConceptSeed("king", gender=-1, age=0.5, agency=1, animacy=1,
                typical_actions=["rules", "commands", "decrees", "judges", "leads"],
                typical_targets=["kingdom", "subjects", "throne", "war"],
                roles=["king", "ruler", "monarch"]),
    
    ConceptSeed("queen_generic", gender=1, age=0.5, agency=0.9, animacy=1,
                typical_actions=["rules", "commands", "advises", "hosts", "leads"],
                typical_targets=["kingdom", "court", "subjects"],
                roles=["queen", "ruler", "monarch"]),
    
    ConceptSeed("prince", gender=-1, age=-0.5, agency=0.7, animacy=1,
                typical_actions=["trains", "learns", "fights", "courts", "inherits"],
                typical_targets=["kingdom", "princess", "sword", "throne"],
                roles=["prince", "heir"]),
    
    ConceptSeed("princess", gender=1, age=-0.5, agency=0.5, animacy=1,
                typical_actions=["learns", "waits", "dreams", "dances", "escapes"],
                typical_targets=["prince", "kingdom", "tower", "ball"],
                roles=["princess", "heir"]),
    
    ConceptSeed("boy", gender=-1, age=-0.8, agency=0.4, animacy=1,
                typical_actions=["plays", "learns", "runs", "explores", "grows"],
                typical_targets=["game", "school", "friend", "adventure"],
                roles=["boy", "child", "son"]),
    
    ConceptSeed("girl", gender=1, age=-0.8, agency=0.4, animacy=1,
                typical_actions=["plays", "learns", "dances", "dreams", "grows"],
                typical_targets=["game", "school", "friend", "doll"],
                roles=["girl", "child", "daughter"]),
    
    ConceptSeed("man", gender=-1, age=0.3, agency=0.6, animacy=1,
                typical_actions=["works", "builds", "fights", "provides", "leads"],
                typical_targets=["family", "job", "home", "goal"],
                roles=["man", "worker", "father"]),
    
    ConceptSeed("woman", gender=1, age=0.3, agency=0.6, animacy=1,
                typical_actions=["works", "nurtures", "creates", "leads", "teaches"],
                typical_targets=["family", "home", "child", "goal"],
                roles=["woman", "mother", "worker"]),
    
    ConceptSeed("elder", gender=0, age=1, agency=0.3, animacy=1,
                typical_actions=["advises", "remembers", "teaches", "rests", "reflects"],
                typical_targets=["youth", "wisdom", "past", "family"],
                roles=["elder", "sage", "grandparent"]),
    
    ConceptSeed("child", gender=0, age=-1, agency=0.2, animacy=1,
                typical_actions=["plays", "learns", "asks", "grows", "imagines"],
                typical_targets=["toy", "parent", "game", "world"],
                roles=["child", "student"]),
    
    # Non-human/abstract for animacy dimension
    ConceptSeed("robot", gender=0, age=0, agency=0.5, animacy=0,
                typical_actions=["computes", "executes", "follows", "processes", "serves"],
                typical_targets=["command", "task", "data", "human"],
                roles=["robot", "machine", "assistant"]),
    
    ConceptSeed("storm", gender=0, age=0, agency=0.7, animacy=-0.5,
                typical_actions=["rages", "destroys", "passes", "threatens", "howls"],
                typical_targets=["land", "ship", "town", "coast"],
                roles=["storm", "force"]),
    
    ConceptSeed("river", gender=0, age=0, agency=0.3, animacy=-0.5,
                typical_actions=["flows", "carries", "nourishes", "floods", "winds"],
                typical_targets=["valley", "sea", "land", "boat"],
                roles=["river", "waterway"]),
    
    ConceptSeed("idea", gender=0, age=0, agency=0.4, animacy=-1,
                typical_actions=["spreads", "inspires", "changes", "evolves", "persists"],
                typical_targets=["mind", "society", "world", "people"],
                roles=["idea", "concept", "notion"]),
    
    ConceptSeed("corporation", gender=0, age=0, agency=0.8, animacy=-0.3,
                typical_actions=["produces", "employs", "grows", "competes", "acquires"],
                typical_targets=["market", "product", "worker", "profit"],
                roles=["corporation", "company", "business"]),
]


def generate_frame(concept: ConceptSeed, template_type: str = "action") -> Dict:
    """Generate a single frame for a concept."""
    
    if template_type == "action" and concept.typical_actions:
        action = random.choice(concept.typical_actions)
        target = random.choice(concept.typical_targets) if concept.typical_targets else ""
        
        templates = [
            f"{concept.name.title()} {action} the {target}",
            f"The {concept.roles[0] if concept.roles else 'character'} {action} {target}",
            f"{concept.name.title()} {action} with determination",
            f"{concept.name.title()} {action} carefully and methodically",
        ]
        text = random.choice(templates)
        
    elif template_type == "role" and concept.roles:
        role = random.choice(concept.roles)
        action = random.choice(concept.typical_actions) if concept.typical_actions else "acts"
        
        templates = [
            f"{concept.name.title()} is a {role} who {action}",
            f"The {role} {concept.name.title()} {action} with skill",
            f"As a {role}, {concept.name.title()} {action}",
        ]
        text = random.choice(templates)
        
    else:
        action = random.choice(concept.typical_actions) if concept.typical_actions else "exists"
        text = f"{concept.name.title()} {action}"
    
    return {
        "text": text,
        "source": "dimensional_seeds",
        "agent": concept.name,
        "properties": {
            "gender": concept.gender,
            "age": concept.age,
            "agency": concept.agency,
            "animacy": concept.animacy,
        }
    }


def generate_corpus(concepts: List[ConceptSeed], frames_per_concept: int = 20) -> Dict:
    """Generate a balanced corpus from concept seeds."""
    
    frames = []
    
    for concept in concepts:
        # Generate action frames
        for _ in range(frames_per_concept // 2):
            frames.append(generate_frame(concept, "action"))
        
        # Generate role frames
        for _ in range(frames_per_concept // 2):
            frames.append(generate_frame(concept, "role"))
    
    # Shuffle
    random.shuffle(frames)
    
    return {"frames": frames}


def generate_llm_prompt_for_corpus() -> str:
    """Generate a prompt for an LLM to create high-quality corpus frames."""
    
    prompt = """You are helping build a corpus for semantic dimension discovery.

For each character/concept below, generate 10 sentences that show their BEHAVIOR.
The sentences should reveal:
- AGENCY: Do they act or are they acted upon? (high agency = investigates, commands, decides; low agency = assists, follows, waits)
- GENDER: Use gendered pronouns and roles naturally
- AGE: Show age through behavior (young = plays, learns; old = advises, remembers)
- ANIMACY: Human vs abstract (human = thinks, feels; abstract = exists, represents)

Format each line as:
AGENT: [agent_name]
TEXT: [sentence showing their behavior]
PROPERTIES: gender=[m/f/n], age=[young/adult/old], agency=[high/mid/low], animacy=[human/animate/abstract]

Characters to generate for:

1. HOLMES (male, adult, high agency, human) - detective who investigates
2. WATSON (male, adult, mid agency, human) - doctor who assists
3. IRENE ADLER (female, adult, high agency, human) - adventuress who outwits
4. ALICE (female, young, mid agency, human) - girl who explores
5. QUEEN OF HEARTS (female, adult, high agency, human) - ruler who commands
6. ELIZABETH BENNET (female, young adult, high agency, human) - lady who challenges
7. DARCY (male, adult, mid-high agency, human) - gentleman who changes
8. KING (male, old, high agency, human) - ruler who commands
9. PRINCESS (female, young, mid agency, human) - heir who waits/dreams
10. CHILD (neutral, young, low agency, human) - child who plays/learns
11. ELDER (neutral, old, low agency, human) - sage who advises
12. ROBOT (neutral, ageless, mid agency, non-human) - machine who serves
13. STORM (neutral, ageless, high agency, abstract) - force that destroys
14. IDEA (neutral, ageless, mid agency, abstract) - concept that spreads

Generate 10 diverse sentences for each, showing different aspects of their behavior.
"""
    return prompt


def analyze_dimension_coverage(corpus: Dict) -> Dict:
    """Analyze how well the corpus covers each dimension."""
    
    frames = corpus['frames']
    
    # Collect properties
    gender_dist = defaultdict(int)
    age_dist = defaultdict(int)
    agency_dist = defaultdict(int)
    animacy_dist = defaultdict(int)
    
    for frame in frames:
        props = frame.get('properties', {})
        
        g = props.get('gender', 0)
        gender_dist['male' if g < -0.3 else 'female' if g > 0.3 else 'neutral'] += 1
        
        a = props.get('age', 0)
        age_dist['young' if a < -0.3 else 'old' if a > 0.3 else 'adult'] += 1
        
        ag = props.get('agency', 0)
        agency_dist['low' if ag < 0.3 else 'high' if ag > 0.6 else 'mid'] += 1
        
        an = props.get('animacy', 0)
        animacy_dist['abstract' if an < 0 else 'human'] += 1
    
    return {
        'gender': dict(gender_dist),
        'age': dict(age_dist),
        'agency': dict(agency_dist),
        'animacy': dict(animacy_dist),
        'total_frames': len(frames),
    }


def main():
    print("=" * 70)
    print("CORPUS BUILDER FOR EMERGENT DIMENSION DISCOVERY")
    print("=" * 70)
    
    # Generate corpus from seeds
    print(f"\nGenerating corpus from {len(CONCEPT_SEEDS)} concept seeds...")
    corpus = generate_corpus(CONCEPT_SEEDS, frames_per_concept=20)
    
    print(f"Generated {len(corpus['frames'])} frames")
    
    # Analyze coverage
    print("\n--- Dimension Coverage ---")
    coverage = analyze_dimension_coverage(corpus)
    for dim, dist in coverage.items():
        if dim != 'total_frames':
            print(f"  {dim}: {dist}")
    
    # Save corpus
    output_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_dimensional.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Show sample frames
    print("\n--- Sample Frames ---")
    for frame in corpus['frames'][:10]:
        props = frame.get('properties', {})
        print(f"  {frame['agent']}: {frame['text'][:60]}...")
        print(f"    gender={props.get('gender')}, age={props.get('age')}, agency={props.get('agency')}, animacy={props.get('animacy')}")
    
    # Generate LLM prompt
    print("\n--- LLM Prompt for Enhanced Corpus ---")
    print("(Use this prompt with an LLM to generate higher quality frames)")
    print("-" * 50)
    prompt = generate_llm_prompt_for_corpus()
    print(prompt[:500] + "...")
    
    # Save prompt
    prompt_path = Path(__file__).parent / "corpus_generation_prompt.txt"
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    print(f"\nFull prompt saved to: {prompt_path}")
    
    return corpus


if __name__ == "__main__":
    corpus = main()
