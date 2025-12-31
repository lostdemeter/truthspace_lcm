#!/usr/bin/env python3
"""
Generate a richer corpus with more behavioral diversity.

This creates frames that express dimensions through varied sentence structures,
not just simple templates.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


# Rich behavioral sentences for each concept
# These are designed to make dimensions discoverable from behavior

CONCEPT_SENTENCES = {
    # HOLMES - male, adult, HIGH agency
    "holmes": [
        "Holmes investigates the mysterious disappearance with keen interest",
        "Holmes deduces the criminal's identity from a single footprint",
        "Holmes examines the evidence meticulously and draws conclusions",
        "Holmes solves the case before Scotland Yard arrives",
        "Holmes observes details that others completely miss",
        "Holmes confronts the villain in his own lair",
        "Holmes discovers the hidden passage behind the bookcase",
        "Holmes analyzes the poison and identifies its origin",
        "Holmes pursues the suspect through the foggy streets",
        "Holmes commands Watson to fetch his magnifying glass",
        "Holmes deciphers the coded message with ease",
        "Holmes exposes the conspiracy to the authorities",
        "Holmes outwits Moriarty at every turn",
        "Holmes leads the investigation with brilliant insight",
        "Holmes questions the witness with surgical precision",
    ],
    
    # WATSON - male, adult, LOW-MID agency
    "watson": [
        "Watson assists Holmes with the investigation",
        "Watson documents the case in his journal",
        "Watson follows Holmes through the dark alley",
        "Watson watches as Holmes examines the clue",
        "Watson supports his friend during the difficult case",
        "Watson accompanies Holmes to the crime scene",
        "Watson records Holmes's brilliant deductions",
        "Watson waits patiently while Holmes thinks",
        "Watson helps Holmes by fetching medical supplies",
        "Watson observes the proceedings with quiet attention",
        "Watson serves as Holmes's trusted companion",
        "Watson listens carefully to Holmes's explanation",
        "Watson provides medical expertise when needed",
        "Watson stands by Holmes in moments of danger",
        "Watson chronicles their adventures for posterity",
    ],
    
    # IRENE - female, adult, HIGH agency
    "irene": [
        "Irene outwits Holmes with her clever disguise",
        "Irene escapes before anyone can stop her",
        "Irene challenges the detective at his own game",
        "Irene defeats the king's schemes with intelligence",
        "Irene commands attention when she enters the room",
        "Irene manipulates the situation to her advantage",
        "Irene leads her pursuers on a merry chase",
        "Irene discovers the plot against her",
        "Irene confronts her enemies without fear",
        "Irene orchestrates her own rescue brilliantly",
        "Irene outsmarts everyone who underestimates her",
        "Irene decides her own fate on her terms",
        "Irene controls the narrative from the beginning",
        "Irene triumphs over those who sought to harm her",
        "Irene proves herself the equal of any man",
    ],
    
    # ALICE - female, YOUNG, mid agency
    "alice": [
        "Alice explores the strange wonderland with curiosity",
        "Alice questions the bizarre rules of this world",
        "Alice grows taller after drinking the potion",
        "Alice shrinks to a tiny size unexpectedly",
        "Alice wonders about the meaning of it all",
        "Alice discovers new rooms in the strange house",
        "Alice learns that nothing is as it seems",
        "Alice asks the Caterpillar for directions",
        "Alice plays croquet with the Queen reluctantly",
        "Alice dreams of a world beyond the looking glass",
        "Alice follows the White Rabbit down the hole",
        "Alice adapts to the changing circumstances",
        "Alice challenges the Queen's unfair rules",
        "Alice escapes from the angry card soldiers",
        "Alice awakens from her strange adventure",
    ],
    
    # QUEEN OF HEARTS - female, adult, HIGH agency
    "queen": [
        "The Queen commands everyone to bow before her",
        "The Queen orders the execution of the gardeners",
        "The Queen rules Wonderland with an iron fist",
        "The Queen demands absolute obedience from all",
        "The Queen shouts at anyone who displeases her",
        "The Queen controls every aspect of the court",
        "The Queen judges the trial with arbitrary fury",
        "The Queen threatens Alice with dire consequences",
        "The Queen leads the croquet game by her rules",
        "The Queen punishes those who dare to defy her",
        "The Queen dominates every conversation",
        "The Queen decides who lives and who dies",
        "The Queen intimidates the entire kingdom",
        "The Queen enforces her will without mercy",
        "The Queen governs through fear and anger",
    ],
    
    # ELIZABETH - female, young adult, HIGH agency
    "elizabeth": [
        "Elizabeth challenges Darcy's arrogant assumptions",
        "Elizabeth refuses his first proposal with dignity",
        "Elizabeth debates with wit and intelligence",
        "Elizabeth walks three miles through muddy fields",
        "Elizabeth reads voraciously and thinks deeply",
        "Elizabeth judges character with keen perception",
        "Elizabeth confronts Lady Catherine without fear",
        "Elizabeth decides her own path in life",
        "Elizabeth questions society's rigid expectations",
        "Elizabeth defends her family's honor fiercely",
        "Elizabeth discovers the truth about Wickham",
        "Elizabeth changes her mind when presented with evidence",
        "Elizabeth speaks her mind regardless of consequence",
        "Elizabeth leads conversations with clever remarks",
        "Elizabeth chooses love over mere convenience",
    ],
    
    # DARCY - male, adult, mid-high agency
    "darcy": [
        "Darcy observes the assembly with quiet disdain",
        "Darcy proposes to Elizabeth despite his pride",
        "Darcy rescues Lydia from disgrace secretly",
        "Darcy changes his behavior after Elizabeth's rebuke",
        "Darcy broods over his feelings in solitude",
        "Darcy writes a letter explaining his actions",
        "Darcy provides for Wickham to protect Elizabeth",
        "Darcy admits his faults with humility",
        "Darcy transforms from proud to humble",
        "Darcy loves Elizabeth despite class differences",
        "Darcy protects his sister from scandal",
        "Darcy struggles with his conflicting emotions",
        "Darcy overcomes his prejudice through reflection",
        "Darcy proves his worth through noble actions",
        "Darcy earns Elizabeth's respect and love",
    ],
    
    # CHILD - neutral, YOUNG, LOW agency
    "child": [
        "The child plays with toys in the nursery",
        "The child learns to read from picture books",
        "The child asks endless questions about everything",
        "The child follows the adults around the house",
        "The child waits for permission to go outside",
        "The child obeys the instructions given by parents",
        "The child imagines fantastic adventures",
        "The child grows a little taller each year",
        "The child watches the adults with wide eyes",
        "The child listens to bedtime stories eagerly",
        "The child depends on others for care",
        "The child explores the garden with wonder",
        "The child dreams of becoming a grown-up",
        "The child receives lessons from the tutor",
        "The child trusts the adults to keep them safe",
    ],
    
    # ELDER - neutral, OLD, LOW agency
    "elder": [
        "The elder advises the young with wisdom",
        "The elder remembers the old days fondly",
        "The elder rests in the comfortable chair",
        "The elder reflects on a life well lived",
        "The elder teaches the traditions to the young",
        "The elder watches the grandchildren play",
        "The elder shares stories of the past",
        "The elder waits patiently for visitors",
        "The elder guides others with gentle counsel",
        "The elder preserves the family history",
        "The elder sits by the fire in contemplation",
        "The elder blesses the young couple's union",
        "The elder passes down ancient knowledge",
        "The elder observes the changing world quietly",
        "The elder accepts the passage of time gracefully",
    ],
    
    # KING - male, OLD, HIGH agency
    "king": [
        "The King commands his armies to march",
        "The King rules the kingdom with absolute power",
        "The King decrees new laws for the realm",
        "The King judges disputes between nobles",
        "The King leads his people through crisis",
        "The King conquers neighboring territories",
        "The King demands tribute from vassal states",
        "The King controls the treasury and resources",
        "The King decides matters of war and peace",
        "The King punishes traitors without mercy",
        "The King rewards loyal subjects generously",
        "The King builds monuments to his glory",
        "The King dominates the royal court",
        "The King enforces his will across the land",
        "The King governs with iron determination",
    ],
    
    # PRINCESS - female, YOUNG, MID agency
    "princess": [
        "The princess dreams of adventure beyond the castle",
        "The princess learns the arts of court etiquette",
        "The princess waits for her prince to arrive",
        "The princess dances at the royal ball gracefully",
        "The princess escapes from the tower at midnight",
        "The princess studies languages and music",
        "The princess hopes for a different life",
        "The princess follows the rules of the court",
        "The princess watches the knights from her window",
        "The princess receives suitors in the great hall",
        "The princess obeys her father's commands reluctantly",
        "The princess imagines a world of freedom",
        "The princess grows into a young woman",
        "The princess questions her predetermined fate",
        "The princess discovers a secret passage",
    ],
    
    # ROBOT - neutral, ageless, MID agency (non-human)
    "robot": [
        "The robot executes its programmed instructions",
        "The robot processes data with perfect accuracy",
        "The robot serves the humans without complaint",
        "The robot follows commands without question",
        "The robot computes solutions to complex problems",
        "The robot performs repetitive tasks efficiently",
        "The robot obeys the three laws of robotics",
        "The robot assists with dangerous operations",
        "The robot calculates probabilities instantly",
        "The robot operates continuously without rest",
        "The robot responds to voice commands precisely",
        "The robot maintains the facility automatically",
        "The robot adapts to new instructions quickly",
        "The robot monitors systems around the clock",
        "The robot functions according to its design",
    ],
    
    # STORM - neutral, ageless, HIGH agency (abstract/force)
    "storm": [
        "The storm rages across the open sea",
        "The storm destroys everything in its path",
        "The storm threatens the coastal villages",
        "The storm howls through the night relentlessly",
        "The storm batters the ships against the rocks",
        "The storm floods the lowland farms",
        "The storm uproots ancient trees effortlessly",
        "The storm darkens the sky for miles",
        "The storm strikes without warning or mercy",
        "The storm overwhelms all human defenses",
        "The storm passes after hours of fury",
        "The storm leaves destruction in its wake",
        "The storm demonstrates nature's raw power",
        "The storm transforms the landscape completely",
        "The storm dominates the entire region",
    ],
}

# Ground truth properties for each concept
CONCEPT_PROPERTIES = {
    "holmes": {"gender": -1, "age": 0, "agency": 1, "animacy": 1},
    "watson": {"gender": -1, "age": 0, "agency": 0.3, "animacy": 1},
    "irene": {"gender": 1, "age": 0, "agency": 0.9, "animacy": 1},
    "alice": {"gender": 1, "age": -0.7, "agency": 0.5, "animacy": 1},
    "queen": {"gender": 1, "age": 0.5, "agency": 1, "animacy": 1},
    "elizabeth": {"gender": 1, "age": -0.2, "agency": 0.8, "animacy": 1},
    "darcy": {"gender": -1, "age": 0, "agency": 0.6, "animacy": 1},
    "child": {"gender": 0, "age": -1, "agency": 0.2, "animacy": 1},
    "elder": {"gender": 0, "age": 1, "agency": 0.3, "animacy": 1},
    "king": {"gender": -1, "age": 0.5, "agency": 1, "animacy": 1},
    "princess": {"gender": 1, "age": -0.5, "agency": 0.4, "animacy": 1},
    "robot": {"gender": 0, "age": 0, "agency": 0.5, "animacy": 0},
    "storm": {"gender": 0, "age": 0, "agency": 0.8, "animacy": -0.5},
}


def generate_corpus() -> Dict:
    """Generate the rich corpus."""
    
    frames = []
    
    for concept, sentences in CONCEPT_SENTENCES.items():
        props = CONCEPT_PROPERTIES.get(concept, {})
        
        for sentence in sentences:
            frames.append({
                "text": sentence,
                "source": "rich_behavioral",
                "agent": concept,
                "properties": props,
            })
    
    # Shuffle
    random.shuffle(frames)
    
    return {"frames": frames}


def analyze_corpus(corpus: Dict):
    """Analyze the corpus for dimension coverage."""
    
    from collections import Counter, defaultdict
    
    frames = corpus['frames']
    
    # Count by agent
    agent_counts = Counter(f['agent'] for f in frames)
    print(f"Agents: {len(agent_counts)}")
    print(f"Frames per agent: {dict(agent_counts)}")
    
    # Analyze dimension coverage
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
        agency_dist['low' if ag < 0.4 else 'high' if ag > 0.7 else 'mid'] += 1
        
        an = props.get('animacy', 0)
        animacy_dist['abstract' if an < 0.5 else 'human'] += 1
    
    print(f"\nDimension coverage:")
    print(f"  Gender: {dict(gender_dist)}")
    print(f"  Age: {dict(age_dist)}")
    print(f"  Agency: {dict(agency_dist)}")
    print(f"  Animacy: {dict(animacy_dist)}")


def main():
    print("=" * 70)
    print("GENERATING RICH BEHAVIORAL CORPUS")
    print("=" * 70)
    
    corpus = generate_corpus()
    print(f"\nGenerated {len(corpus['frames'])} frames")
    
    analyze_corpus(corpus)
    
    # Save
    output_path = Path(__file__).parent.parent / "truthspace_lcm" / "gears" / "corpus" / "corpus_rich_behavioral.json"
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Show samples
    print("\n--- Sample Frames ---")
    for frame in corpus['frames'][:10]:
        print(f"  {frame['agent']}: {frame['text'][:60]}...")
    
    return corpus


if __name__ == "__main__":
    corpus = main()
