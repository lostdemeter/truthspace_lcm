#!/usr/bin/env python3
"""
LLM Corpus Generator for Emergent Dimension Discovery

This generates a rich behavioral corpus designed to let dimensions emerge naturally.
The corpus covers many potential dimensions without predetermining which ones matter.

Dimensions that MIGHT emerge (we don't predetermine):
- Agency (active vs passive)
- Gender (male vs female vs neutral)
- Age (young vs old)
- Animacy (human vs abstract vs object)
- Morality (good vs evil)
- Power (powerful vs powerless)
- Certainty (certain vs uncertain)
- Temporality (past vs present vs future focused)
- Sociality (social vs solitary)
- Emotionality (emotional vs rational)

The key: we encode these through BEHAVIOR, not labels.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Character:
    """A character with behavioral tendencies."""
    name: str
    # These are HIDDEN properties - we don't tell the system about them
    # We just use them to generate appropriate behaviors
    _agency: float = 0.5  # 0=passive, 1=active
    _gender: float = 0.0  # -1=male, 1=female, 0=neutral
    _age: float = 0.5     # 0=young, 1=old
    _animacy: float = 1.0 # 0=abstract, 1=human
    _morality: float = 0.5 # 0=evil, 1=good
    _power: float = 0.5   # 0=powerless, 1=powerful
    _certainty: float = 0.5 # 0=uncertain, 1=certain
    _sociality: float = 0.5 # 0=solitary, 1=social
    _emotionality: float = 0.5 # 0=rational, 1=emotional


# Rich character database with hidden dimensional properties
CHARACTERS = [
    # High agency, male, adult
    Character("Holmes", _agency=0.95, _gender=-1, _age=0.5, _animacy=1, _morality=0.8, _power=0.7, _certainty=0.9, _sociality=0.4, _emotionality=0.2),
    Character("Napoleon", _agency=0.95, _gender=-1, _age=0.6, _animacy=1, _morality=0.4, _power=0.95, _certainty=0.9, _sociality=0.6, _emotionality=0.3),
    Character("Caesar", _agency=0.9, _gender=-1, _age=0.6, _animacy=1, _morality=0.5, _power=0.95, _certainty=0.85, _sociality=0.7, _emotionality=0.3),
    
    # High agency, female, adult
    Character("Cleopatra", _agency=0.9, _gender=1, _age=0.5, _animacy=1, _morality=0.5, _power=0.9, _certainty=0.8, _sociality=0.8, _emotionality=0.6),
    Character("Elizabeth", _agency=0.85, _gender=1, _age=0.4, _animacy=1, _morality=0.7, _power=0.3, _certainty=0.7, _sociality=0.6, _emotionality=0.5),
    Character("Irene", _agency=0.9, _gender=1, _age=0.4, _animacy=1, _morality=0.6, _power=0.5, _certainty=0.8, _sociality=0.5, _emotionality=0.4),
    Character("Athena", _agency=0.85, _gender=1, _age=0.5, _animacy=0.8, _morality=0.8, _power=0.9, _certainty=0.9, _sociality=0.5, _emotionality=0.2),
    
    # Low agency, male, adult
    Character("Watson", _agency=0.35, _gender=-1, _age=0.5, _animacy=1, _morality=0.9, _power=0.3, _certainty=0.5, _sociality=0.7, _emotionality=0.5),
    Character("Servant", _agency=0.2, _gender=-1, _age=0.5, _animacy=1, _morality=0.7, _power=0.1, _certainty=0.4, _sociality=0.6, _emotionality=0.4),
    Character("Prisoner", _agency=0.1, _gender=-1, _age=0.5, _animacy=1, _morality=0.5, _power=0.05, _certainty=0.3, _sociality=0.3, _emotionality=0.7),
    
    # Low agency, female, adult
    Character("Maid", _agency=0.2, _gender=1, _age=0.4, _animacy=1, _morality=0.8, _power=0.1, _certainty=0.4, _sociality=0.6, _emotionality=0.5),
    Character("Nun", _agency=0.3, _gender=1, _age=0.5, _animacy=1, _morality=0.95, _power=0.2, _certainty=0.8, _sociality=0.4, _emotionality=0.6),
    
    # Young characters
    Character("Boy", _agency=0.4, _gender=-1, _age=0.1, _animacy=1, _morality=0.6, _power=0.1, _certainty=0.3, _sociality=0.7, _emotionality=0.8),
    Character("Girl", _agency=0.4, _gender=1, _age=0.1, _animacy=1, _morality=0.6, _power=0.1, _certainty=0.3, _sociality=0.8, _emotionality=0.8),
    Character("Alice", _agency=0.5, _gender=1, _age=0.15, _animacy=1, _morality=0.7, _power=0.2, _certainty=0.4, _sociality=0.6, _emotionality=0.7),
    Character("Prince", _agency=0.6, _gender=-1, _age=0.2, _animacy=1, _morality=0.6, _power=0.6, _certainty=0.5, _sociality=0.6, _emotionality=0.5),
    Character("Princess", _agency=0.4, _gender=1, _age=0.2, _animacy=1, _morality=0.7, _power=0.5, _certainty=0.4, _sociality=0.7, _emotionality=0.6),
    Character("Student", _agency=0.4, _gender=0, _age=0.2, _animacy=1, _morality=0.6, _power=0.2, _certainty=0.3, _sociality=0.7, _emotionality=0.6),
    
    # Old characters
    Character("Elder", _agency=0.3, _gender=0, _age=0.95, _animacy=1, _morality=0.8, _power=0.4, _certainty=0.7, _sociality=0.5, _emotionality=0.4),
    Character("Sage", _agency=0.4, _gender=-1, _age=0.9, _animacy=1, _morality=0.85, _power=0.5, _certainty=0.85, _sociality=0.3, _emotionality=0.2),
    Character("Grandmother", _agency=0.3, _gender=1, _age=0.9, _animacy=1, _morality=0.9, _power=0.2, _certainty=0.6, _sociality=0.8, _emotionality=0.7),
    Character("King", _agency=0.8, _gender=-1, _age=0.7, _animacy=1, _morality=0.5, _power=0.95, _certainty=0.8, _sociality=0.6, _emotionality=0.3),
    Character("Queen", _agency=0.75, _gender=1, _age=0.6, _animacy=1, _morality=0.6, _power=0.9, _certainty=0.75, _sociality=0.7, _emotionality=0.4),
    
    # Evil/antagonist characters
    Character("Moriarty", _agency=0.9, _gender=-1, _age=0.5, _animacy=1, _morality=0.1, _power=0.8, _certainty=0.85, _sociality=0.3, _emotionality=0.2),
    Character("Witch", _agency=0.8, _gender=1, _age=0.6, _animacy=1, _morality=0.15, _power=0.7, _certainty=0.7, _sociality=0.2, _emotionality=0.5),
    Character("Villain", _agency=0.85, _gender=-1, _age=0.5, _animacy=1, _morality=0.1, _power=0.7, _certainty=0.6, _sociality=0.3, _emotionality=0.4),
    Character("Tyrant", _agency=0.9, _gender=-1, _age=0.6, _animacy=1, _morality=0.05, _power=0.95, _certainty=0.8, _sociality=0.4, _emotionality=0.3),
    
    # Abstract/non-human entities
    Character("Storm", _agency=0.8, _gender=0, _age=0.5, _animacy=0.1, _morality=0.5, _power=0.9, _certainty=0.5, _sociality=0.0, _emotionality=0.0),
    Character("River", _agency=0.4, _gender=0, _age=0.5, _animacy=0.1, _morality=0.5, _power=0.5, _certainty=0.5, _sociality=0.0, _emotionality=0.0),
    Character("Mountain", _agency=0.1, _gender=0, _age=0.9, _animacy=0.05, _morality=0.5, _power=0.7, _certainty=0.9, _sociality=0.0, _emotionality=0.0),
    Character("Fire", _agency=0.7, _gender=0, _age=0.5, _animacy=0.1, _morality=0.5, _power=0.8, _certainty=0.4, _sociality=0.0, _emotionality=0.0),
    Character("Time", _agency=0.6, _gender=0, _age=0.5, _animacy=0.0, _morality=0.5, _power=0.95, _certainty=0.9, _sociality=0.0, _emotionality=0.0),
    Character("Death", _agency=0.7, _gender=0, _age=0.5, _animacy=0.2, _morality=0.5, _power=0.95, _certainty=1.0, _sociality=0.1, _emotionality=0.0),
    Character("Love", _agency=0.5, _gender=0, _age=0.5, _animacy=0.0, _morality=0.8, _power=0.7, _certainty=0.3, _sociality=1.0, _emotionality=1.0),
    Character("Fear", _agency=0.6, _gender=0, _age=0.5, _animacy=0.0, _morality=0.3, _power=0.6, _certainty=0.4, _sociality=0.3, _emotionality=1.0),
    
    # Robots/AI
    Character("Robot", _agency=0.5, _gender=0, _age=0.5, _animacy=0.3, _morality=0.5, _power=0.4, _certainty=0.9, _sociality=0.2, _emotionality=0.0),
    Character("AI", _agency=0.6, _gender=0, _age=0.3, _animacy=0.2, _morality=0.5, _power=0.5, _certainty=0.8, _sociality=0.3, _emotionality=0.0),
    
    # Animals (for animacy dimension)
    Character("Wolf", _agency=0.7, _gender=0, _age=0.5, _animacy=0.7, _morality=0.4, _power=0.6, _certainty=0.6, _sociality=0.6, _emotionality=0.5),
    Character("Owl", _agency=0.4, _gender=0, _age=0.6, _animacy=0.6, _morality=0.5, _power=0.3, _certainty=0.7, _sociality=0.2, _emotionality=0.2),
    Character("Dog", _agency=0.5, _gender=0, _age=0.4, _animacy=0.7, _morality=0.8, _power=0.3, _certainty=0.5, _sociality=0.9, _emotionality=0.8),
    Character("Cat", _agency=0.5, _gender=0, _age=0.4, _animacy=0.7, _morality=0.5, _power=0.2, _certainty=0.6, _sociality=0.3, _emotionality=0.4),
]


# Verb pools organized by dimensional tendency
VERBS = {
    'high_agency': [
        'commands', 'leads', 'decides', 'conquers', 'creates', 'destroys', 'builds',
        'investigates', 'discovers', 'solves', 'confronts', 'challenges', 'defeats',
        'rules', 'governs', 'controls', 'dominates', 'orchestrates', 'initiates',
        'transforms', 'revolutionizes', 'pioneers', 'establishes', 'declares',
    ],
    'low_agency': [
        'follows', 'obeys', 'waits', 'watches', 'assists', 'serves', 'supports',
        'receives', 'accepts', 'endures', 'suffers', 'submits', 'yields',
        'listens', 'observes', 'accompanies', 'attends', 'helps', 'aids',
    ],
    'young': [
        'plays', 'learns', 'grows', 'explores', 'wonders', 'asks', 'dreams',
        'imagines', 'discovers', 'runs', 'laughs', 'cries', 'hopes',
    ],
    'old': [
        'remembers', 'reflects', 'advises', 'teaches', 'guides', 'rests',
        'contemplates', 'recalls', 'mentors', 'blesses', 'preserves',
    ],
    'good': [
        'helps', 'saves', 'protects', 'heals', 'forgives', 'loves', 'nurtures',
        'defends', 'rescues', 'comforts', 'supports', 'encourages', 'inspires',
    ],
    'evil': [
        'schemes', 'plots', 'betrays', 'destroys', 'corrupts', 'manipulates',
        'threatens', 'attacks', 'deceives', 'poisons', 'curses', 'torments',
    ],
    'powerful': [
        'commands', 'rules', 'conquers', 'dominates', 'controls', 'decrees',
        'judges', 'punishes', 'rewards', 'grants', 'bestows', 'summons',
    ],
    'powerless': [
        'begs', 'pleads', 'hopes', 'prays', 'wishes', 'fears', 'hides',
        'flees', 'cowers', 'trembles', 'suffers', 'endures', 'waits',
    ],
    'certain': [
        'knows', 'declares', 'proclaims', 'asserts', 'confirms', 'proves',
        'demonstrates', 'establishes', 'determines', 'concludes', 'decides',
    ],
    'uncertain': [
        'wonders', 'questions', 'doubts', 'hesitates', 'ponders', 'considers',
        'speculates', 'guesses', 'hopes', 'fears', 'worries', 'suspects',
    ],
    'social': [
        'gathers', 'unites', 'celebrates', 'shares', 'communicates', 'connects',
        'befriends', 'welcomes', 'hosts', 'invites', 'collaborates', 'joins',
    ],
    'solitary': [
        'isolates', 'withdraws', 'contemplates', 'meditates', 'wanders', 'broods',
        'reflects', 'retreats', 'hides', 'avoids', 'escapes', 'departs',
    ],
    'emotional': [
        'loves', 'hates', 'fears', 'hopes', 'grieves', 'rejoices', 'rages',
        'weeps', 'laughs', 'trembles', 'yearns', 'despairs', 'exults',
    ],
    'rational': [
        'analyzes', 'calculates', 'deduces', 'reasons', 'evaluates', 'assesses',
        'measures', 'computes', 'determines', 'concludes', 'infers', 'derives',
    ],
    'abstract': [
        'exists', 'persists', 'endures', 'flows', 'passes', 'transforms',
        'spreads', 'consumes', 'encompasses', 'pervades', 'manifests',
    ],
}

# Targets organized by context
TARGETS = {
    'people': ['the people', 'the crowd', 'the subjects', 'the followers', 'the citizens'],
    'enemies': ['the enemy', 'the foe', 'the adversary', 'the opponent', 'the rival'],
    'abstract': ['truth', 'justice', 'power', 'wisdom', 'knowledge', 'fate', 'destiny'],
    'places': ['the kingdom', 'the land', 'the realm', 'the world', 'the domain'],
    'objects': ['the treasure', 'the artifact', 'the weapon', 'the key', 'the secret'],
    'nature': ['the storm', 'the sea', 'the mountain', 'the forest', 'the sky'],
}

# Adverbs for variety
ADVERBS = {
    'high_agency': ['boldly', 'decisively', 'forcefully', 'deliberately', 'confidently'],
    'low_agency': ['quietly', 'patiently', 'humbly', 'obediently', 'meekly'],
    'certain': ['certainly', 'undoubtedly', 'surely', 'definitely', 'absolutely'],
    'uncertain': ['perhaps', 'possibly', 'maybe', 'uncertainly', 'hesitantly'],
    'emotional': ['passionately', 'desperately', 'fervently', 'intensely', 'deeply'],
    'rational': ['logically', 'methodically', 'systematically', 'carefully', 'precisely'],
}


def select_verb(char: Character) -> str:
    """Select a verb based on character's hidden properties."""
    pools = []
    weights = []
    
    # Agency
    if char._agency > 0.6:
        pools.append(VERBS['high_agency'])
        weights.append(char._agency)
    else:
        pools.append(VERBS['low_agency'])
        weights.append(1 - char._agency)
    
    # Age
    if char._age < 0.3:
        pools.append(VERBS['young'])
        weights.append(1 - char._age)
    elif char._age > 0.7:
        pools.append(VERBS['old'])
        weights.append(char._age)
    
    # Morality
    if char._morality > 0.7:
        pools.append(VERBS['good'])
        weights.append(char._morality)
    elif char._morality < 0.3:
        pools.append(VERBS['evil'])
        weights.append(1 - char._morality)
    
    # Power
    if char._power > 0.7:
        pools.append(VERBS['powerful'])
        weights.append(char._power)
    elif char._power < 0.3:
        pools.append(VERBS['powerless'])
        weights.append(1 - char._power)
    
    # Certainty
    if char._certainty > 0.7:
        pools.append(VERBS['certain'])
        weights.append(char._certainty)
    elif char._certainty < 0.3:
        pools.append(VERBS['uncertain'])
        weights.append(1 - char._certainty)
    
    # Sociality
    if char._sociality > 0.7:
        pools.append(VERBS['social'])
        weights.append(char._sociality)
    elif char._sociality < 0.3:
        pools.append(VERBS['solitary'])
        weights.append(1 - char._sociality)
    
    # Emotionality
    if char._emotionality > 0.7:
        pools.append(VERBS['emotional'])
        weights.append(char._emotionality)
    elif char._emotionality < 0.3:
        pools.append(VERBS['rational'])
        weights.append(1 - char._emotionality)
    
    # Animacy
    if char._animacy < 0.3:
        pools.append(VERBS['abstract'])
        weights.append(1 - char._animacy)
    
    # Weighted random selection from pools
    if not pools:
        pools = [VERBS['high_agency'] + VERBS['low_agency']]
        weights = [1.0]
    
    # Normalize weights
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # Select pool then verb
    pool = random.choices(pools, weights=weights)[0]
    return random.choice(pool)


def select_adverb(char: Character) -> str:
    """Select an adverb based on character properties."""
    if char._agency > 0.6:
        pool = ADVERBS['high_agency']
    elif char._agency < 0.4:
        pool = ADVERBS['low_agency']
    elif char._certainty > 0.7:
        pool = ADVERBS['certain']
    elif char._certainty < 0.3:
        pool = ADVERBS['uncertain']
    elif char._emotionality > 0.7:
        pool = ADVERBS['emotional']
    elif char._emotionality < 0.3:
        pool = ADVERBS['rational']
    else:
        pool = ADVERBS['high_agency'] + ADVERBS['low_agency']
    
    return random.choice(pool)


def select_target(char: Character) -> str:
    """Select a target based on character properties."""
    if char._power > 0.7:
        pools = [TARGETS['people'], TARGETS['places'], TARGETS['enemies']]
    elif char._animacy < 0.3:
        pools = [TARGETS['abstract'], TARGETS['nature']]
    elif char._morality < 0.3:
        pools = [TARGETS['enemies'], TARGETS['people'], TARGETS['objects']]
    else:
        pools = [TARGETS['abstract'], TARGETS['objects'], TARGETS['people']]
    
    pool = random.choice(pools)
    return random.choice(pool)


def generate_sentence(char: Character) -> str:
    """Generate a behavioral sentence for a character."""
    verb = select_verb(char)
    
    templates = [
        f"{char.name} {verb} {select_target(char)}",
        f"{char.name} {verb} {select_adverb(char)}",
        f"{char.name} {verb} {select_target(char)} {select_adverb(char)}",
        f"The {char.name.lower()} {verb} with determination",
        f"{char.name} {verb} without hesitation",
    ]
    
    return random.choice(templates)


def generate_corpus(n_sentences_per_char: int = 30) -> Dict:
    """Generate the full corpus."""
    frames = []
    
    for char in CHARACTERS:
        for _ in range(n_sentences_per_char):
            sentence = generate_sentence(char)
            frames.append({
                "text": sentence,
                "source": "llm_generated",
                "agent": char.name.lower(),
                # We store hidden properties for validation only
                "_hidden_properties": {
                    "agency": char._agency,
                    "gender": char._gender,
                    "age": char._age,
                    "animacy": char._animacy,
                    "morality": char._morality,
                    "power": char._power,
                    "certainty": char._certainty,
                    "sociality": char._sociality,
                    "emotionality": char._emotionality,
                }
            })
    
    random.shuffle(frames)
    return {"frames": frames}


def main():
    print("=" * 70)
    print("LLM CORPUS GENERATOR")
    print("=" * 70)
    
    corpus = generate_corpus(n_sentences_per_char=30)
    print(f"\nGenerated {len(corpus['frames'])} frames for {len(CHARACTERS)} characters")
    
    # Save
    output_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_generated.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved to: {output_path}")
    
    # Show samples
    print("\n--- Sample Frames ---")
    for frame in corpus['frames'][:15]:
        print(f"  {frame['agent']}: {frame['text']}")
    
    # Show character distribution
    from collections import Counter
    agents = Counter(f['agent'] for f in corpus['frames'])
    print(f"\n--- Character Distribution ---")
    print(f"Total unique characters: {len(agents)}")
    
    return corpus


if __name__ == "__main__":
    corpus = main()
