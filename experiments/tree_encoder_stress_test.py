#!/usr/bin/env python3
"""
Tree Encoder Stress Test

Feed the tree encoder a large amount of text and test if it can:
1. Learn language structure from corpus
2. Answer questions about what it learned
3. Generate coherent responses

This is the real test - can emergent structure support conversation?
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from experiments.tree_encoder import TreeEncoder, Fact
import numpy as np
from typing import List, Tuple, Dict
import re

# =============================================================================
# CORPUS: A lot of factual information
# =============================================================================

CORPUS = """
# HISTORY

George Washington was born on February 22, 1732, in Westmoreland County, Virginia.
Washington served as the first President of the United States from 1789 to 1797.
He commanded the Continental Army during the American Revolutionary War.
Washington died on December 14, 1799, at his home in Mount Vernon, Virginia.
He is often called the Father of His Country.

Abraham Lincoln was born on February 12, 1809, in Hodgenville, Kentucky.
Lincoln served as the 16th President of the United States from 1861 to 1865.
He led the nation through the Civil War and abolished slavery.
Lincoln was assassinated by John Wilkes Booth on April 14, 1865.
He delivered the famous Gettysburg Address in 1863.

Thomas Jefferson was born on April 13, 1743, in Shadwell, Virginia.
Jefferson was the principal author of the Declaration of Independence in 1776.
He served as the third President of the United States from 1801 to 1809.
Jefferson founded the University of Virginia.
He died on July 4, 1826, the same day as John Adams.

Benjamin Franklin was born on January 17, 1706, in Boston, Massachusetts.
Franklin was a polymath, inventor, scientist, and diplomat.
He discovered that lightning is electrical and invented the lightning rod.
Franklin helped draft the Declaration of Independence and the Constitution.
He died on April 17, 1790, in Philadelphia.

# SCIENCE

Albert Einstein was born on March 14, 1879, in Ulm, Germany.
Einstein developed the theory of relativity, including E equals mc squared.
He won the Nobel Prize in Physics in 1921 for the photoelectric effect.
Einstein fled Nazi Germany and became a US citizen in 1940.
He died on April 18, 1955, in Princeton, New Jersey.

Isaac Newton was born on December 25, 1642, in Woolsthorpe, England.
Newton discovered the laws of motion and universal gravitation.
He invented calculus independently of Leibniz.
Newton wrote the Principia Mathematica in 1687.
He died on March 31, 1727, in London.

Charles Darwin was born on February 12, 1809, in Shrewsbury, England.
Darwin developed the theory of evolution by natural selection.
He traveled on HMS Beagle to the Galapagos Islands.
Darwin published On the Origin of Species in 1859.
He died on April 19, 1882, in Downe, Kent.

Marie Curie was born on November 7, 1867, in Warsaw, Poland.
Curie discovered radioactivity and the elements polonium and radium.
She was the first woman to win a Nobel Prize.
Curie won Nobel Prizes in both Physics and Chemistry.
She died on July 4, 1934, from aplastic anemia caused by radiation exposure.

# GEOGRAPHY

Paris is the capital of France and is located on the Seine River.
The Eiffel Tower in Paris was built in 1889 and is 330 meters tall.
France is located in Western Europe and has a population of about 67 million.
The French language is spoken in France and many other countries.

London is the capital of England and the United Kingdom.
The River Thames flows through London.
Big Ben is a famous clock tower in London.
The population of London is about 9 million people.

Tokyo is the capital of Japan and the most populous city in the world.
Mount Fuji is the highest mountain in Japan at 3776 meters.
Japan is an island nation in East Asia with about 125 million people.
The Japanese language uses three writing systems: hiragana, katakana, and kanji.

Berlin is the capital of Germany and its largest city.
The Berlin Wall divided the city from 1961 to 1989.
Germany is located in Central Europe with about 83 million people.
The German language is spoken in Germany, Austria, and Switzerland.

Moscow is the capital of Russia and its largest city.
The Kremlin is a historic fortress in Moscow.
Russia is the largest country in the world by area.
The Russian language uses the Cyrillic alphabet.

Beijing is the capital of China and one of the oldest cities in the world.
The Great Wall of China is over 20000 kilometers long.
China has the largest population in the world with over 1.4 billion people.
Mandarin Chinese is the most spoken language in the world.

# COOKING

To boil pasta, bring a large pot of salted water to a rolling boil.
Add the pasta and cook for 8 to 12 minutes until al dente.
Drain the pasta and toss with your favorite sauce.
Fresh pasta cooks faster than dried pasta.

To bake bread, mix flour, water, yeast, and salt to form a dough.
Knead the dough for about 10 minutes until smooth and elastic.
Let the dough rise for 1 to 2 hours until doubled in size.
Bake at 450 degrees Fahrenheit for 30 to 40 minutes.

To fry chicken, coat the pieces in seasoned flour or batter.
Heat oil to 350 degrees Fahrenheit in a deep pan or fryer.
Fry the chicken for 12 to 15 minutes until golden brown and cooked through.
Let the chicken rest on a wire rack to stay crispy.

To grill a steak, season with salt and pepper and let it come to room temperature.
Preheat the grill to high heat.
Grill for 4 to 5 minutes per side for medium rare.
Let the steak rest for 5 minutes before slicing.

To roast vegetables, cut them into uniform pieces.
Toss with olive oil, salt, and pepper.
Spread on a baking sheet in a single layer.
Roast at 425 degrees Fahrenheit for 25 to 35 minutes until caramelized.

# TECHNOLOGY

The ls command lists files and directories in Linux.
Use ls -la to show hidden files and detailed information.
The ls command is one of the most commonly used Linux commands.

The grep command searches for text patterns in files.
Use grep -r to search recursively through directories.
Regular expressions can be used with grep for complex pattern matching.

The cat command displays the contents of a file.
Use cat to concatenate multiple files together.
The cat command can also create new files.

The df command shows disk space usage on the filesystem.
Use df -h for human readable output in gigabytes and megabytes.
The df command helps monitor storage capacity.

The ps command shows running processes on the system.
Use ps aux to see all processes with detailed information.
The ps command is useful for system monitoring and debugging.

The chmod command changes file permissions in Linux.
Permissions include read, write, and execute for user, group, and others.
Use chmod 755 to make a file executable.

The mkdir command creates new directories.
Use mkdir -p to create parent directories as needed.
The mkdir command is essential for organizing files.

The rm command removes files and directories.
Use rm -r to remove directories recursively.
Be careful with rm as deleted files cannot be easily recovered.
"""

def parse_corpus(text: str) -> List[Tuple[str, str]]:
    """Parse corpus into (fact_text, fact_id) pairs."""
    facts = []
    lines = text.strip().split('\n')
    
    fact_id = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Create a short ID from the first few words
        words = line.split()[:3]
        short_id = '_'.join(w.lower() for w in words)
        short_id = re.sub(r'[^a-z_]', '', short_id)
        
        facts.append((line, f"{short_id}_{fact_id}"))
        fact_id += 1
    
    return facts


def create_qa_pairs(facts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Create question-answer pairs from facts."""
    qa_pairs = []
    
    for fact_text, fact_id in facts:
        text_lower = fact_text.lower()
        
        # Birth questions
        if 'born' in text_lower:
            # Extract name (usually first two words)
            words = fact_text.split()
            name = words[0] if words else "person"
            qa_pairs.append((f"when was {name} born", fact_id))
            qa_pairs.append((f"where was {name} born", fact_id))
        
        # Death questions
        if 'died' in text_lower or 'death' in text_lower or 'assassinated' in text_lower:
            words = fact_text.split()
            name = words[0] if words else "person"
            qa_pairs.append((f"when did {name} die", fact_id))
            qa_pairs.append((f"how did {name} die", fact_id))
        
        # Capital questions
        if 'capital' in text_lower:
            # Find country name
            if 'france' in text_lower:
                qa_pairs.append(("what is the capital of France", fact_id))
                qa_pairs.append(("capital of France", fact_id))
            elif 'england' in text_lower or 'united kingdom' in text_lower:
                qa_pairs.append(("what is the capital of England", fact_id))
                qa_pairs.append(("capital of England", fact_id))
            elif 'japan' in text_lower:
                qa_pairs.append(("what is the capital of Japan", fact_id))
                qa_pairs.append(("capital of Japan", fact_id))
            elif 'germany' in text_lower:
                qa_pairs.append(("what is the capital of Germany", fact_id))
                qa_pairs.append(("capital of Germany", fact_id))
            elif 'russia' in text_lower:
                qa_pairs.append(("what is the capital of Russia", fact_id))
                qa_pairs.append(("capital of Russia", fact_id))
            elif 'china' in text_lower:
                qa_pairs.append(("what is the capital of China", fact_id))
                qa_pairs.append(("capital of China", fact_id))
        
        # President questions
        if 'president' in text_lower:
            words = fact_text.split()
            name = words[0] if words else "person"
            qa_pairs.append((f"was {name} a president", fact_id))
        
        # Discovery/invention questions
        if 'discovered' in text_lower or 'invented' in text_lower or 'developed' in text_lower:
            words = fact_text.split()
            name = words[0] if words else "person"
            qa_pairs.append((f"what did {name} discover", fact_id))
            qa_pairs.append((f"what did {name} invent", fact_id))
        
        # Cooking questions
        if 'boil' in text_lower and 'pasta' in text_lower:
            qa_pairs.append(("how to cook pasta", fact_id))
            qa_pairs.append(("how to boil pasta", fact_id))
        if 'bake' in text_lower and 'bread' in text_lower:
            qa_pairs.append(("how to bake bread", fact_id))
        if 'fry' in text_lower and 'chicken' in text_lower:
            qa_pairs.append(("how to fry chicken", fact_id))
        if 'grill' in text_lower and 'steak' in text_lower:
            qa_pairs.append(("how to grill steak", fact_id))
        if 'roast' in text_lower and 'vegetables' in text_lower:
            qa_pairs.append(("how to roast vegetables", fact_id))
        
        # Linux command questions
        if text_lower.startswith('the ls '):
            qa_pairs.append(("how to list files", fact_id))
            qa_pairs.append(("ls command", fact_id))
        if text_lower.startswith('the grep '):
            qa_pairs.append(("how to search text in files", fact_id))
            qa_pairs.append(("grep command", fact_id))
        if text_lower.startswith('the cat '):
            qa_pairs.append(("how to display file contents", fact_id))
            qa_pairs.append(("cat command", fact_id))
        if text_lower.startswith('the df '):
            qa_pairs.append(("how to check disk space", fact_id))
            qa_pairs.append(("df command", fact_id))
        if text_lower.startswith('the ps '):
            qa_pairs.append(("how to see running processes", fact_id))
            qa_pairs.append(("ps command", fact_id))
        if text_lower.startswith('the chmod '):
            qa_pairs.append(("how to change file permissions", fact_id))
            qa_pairs.append(("chmod command", fact_id))
        if text_lower.startswith('the mkdir '):
            qa_pairs.append(("how to create directory", fact_id))
            qa_pairs.append(("mkdir command", fact_id))
        if text_lower.startswith('the rm '):
            qa_pairs.append(("how to delete files", fact_id))
            qa_pairs.append(("rm command", fact_id))
    
    return qa_pairs


def main():
    print("=" * 70)
    print("TREE ENCODER STRESS TEST")
    print("Can emergent structure support conversation?")
    print("=" * 70)
    
    # Parse corpus
    facts = parse_corpus(CORPUS)
    print(f"\nParsed {len(facts)} facts from corpus")
    
    # Create encoder
    enc = TreeEncoder(dim=64)
    enc.grow()  # Living mode
    
    # ==========================================================================
    # PHASE 1: INGEST ALL FACTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: INGESTING CORPUS (living mode)")
    print("=" * 70)
    
    for fact_text, fact_id in facts:
        enc.store(fact_text, fact_id)
    
    print(f"  Stored {len(facts)} facts")
    print(f"  Vocabulary: {enc.stats()['total_vocabulary']} words")
    print(f"  Dynamics steps: {enc.stats()['dynamics_steps']}")
    
    # ==========================================================================
    # PHASE 2: TEST QUERIES BEFORE CRYSTALLIZATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TESTING QUERIES (living mode)")
    print("=" * 70)
    
    qa_pairs = create_qa_pairs(facts)
    print(f"  Generated {len(qa_pairs)} QA pairs")
    
    # Test a sample
    sample_queries = [
        "when was Washington born",
        "when did Lincoln die",
        "capital of France",
        "capital of Japan",
        "what did Einstein discover",
        "what did Newton discover",
        "how to cook pasta",
        "how to grill steak",
        "how to list files",
        "how to check disk space",
    ]
    
    print("\n  Sample queries (living mode):")
    correct = 0
    for query in sample_queries:
        matched, sim = enc.query(query)
        # Find expected
        expected_ids = [fid for q, fid in qa_pairs if q == query]
        is_correct = matched.id in expected_ids if expected_ids else False
        correct += is_correct
        marker = '✓' if is_correct else '✗'
        answer_preview = matched.text[:50] + "..." if len(matched.text) > 50 else matched.text
        print(f"    {marker} \"{query}\"")
        print(f"       → {answer_preview}")
    
    print(f"\n  Sample accuracy: {correct}/{len(sample_queries)} = {correct/len(sample_queries):.0%}")
    
    # ==========================================================================
    # PHASE 3: CRYSTALLIZE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: CRYSTALLIZATION")
    print("=" * 70)
    
    ring = enc.crystallize()
    print(f"  Created growth ring with {ring.vocabulary_size} words")
    print(f"  Stats: {enc.stats()}")
    
    # ==========================================================================
    # PHASE 4: DEAD MODE TESTING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: TESTING IN DEAD MODE (frozen)")
    print("=" * 70)
    
    enc.freeze()
    
    print("\n  Sample queries (dead mode):")
    correct = 0
    for query in sample_queries:
        matched, sim = enc.query(query)
        expected_ids = [fid for q, fid in qa_pairs if q == query]
        is_correct = matched.id in expected_ids if expected_ids else False
        correct += is_correct
        marker = '✓' if is_correct else '✗'
        answer_preview = matched.text[:50] + "..." if len(matched.text) > 50 else matched.text
        print(f"    {marker} \"{query}\"")
        print(f"       → {answer_preview}")
    
    print(f"\n  Sample accuracy: {correct}/{len(sample_queries)} = {correct/len(sample_queries):.0%}")
    
    # ==========================================================================
    # PHASE 5: FULL QA EVALUATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: FULL QA EVALUATION")
    print("=" * 70)
    
    total_correct = 0
    total_queries = 0
    
    for query, expected_id in qa_pairs:
        matched, sim = enc.query(query)
        if matched.id == expected_id:
            total_correct += 1
        total_queries += 1
    
    print(f"  Total QA pairs: {total_queries}")
    print(f"  Correct: {total_correct}")
    print(f"  Accuracy: {total_correct/total_queries:.1%}")
    
    # ==========================================================================
    # PHASE 6: INTERACTIVE CONVERSATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: CONVERSATION TEST")
    print("=" * 70)
    
    conversation = [
        "Tell me about George Washington",
        "When was he born",
        "What about Abraham Lincoln",
        "How did Lincoln die",
        "What is the capital of France",
        "Tell me about the Eiffel Tower",
        "How do I cook pasta",
        "What about grilling steak",
        "How do I list files in Linux",
        "What did Einstein discover",
    ]
    
    print("\n  Simulated conversation:")
    for i, query in enumerate(conversation):
        matched, sim = enc.query(query)
        answer = matched.text
        print(f"\n  User: {query}")
        print(f"  Bot:  {answer}")
    
    # ==========================================================================
    # PHASE 7: EMERGENT CLUSTERS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: EMERGENT STRUCTURE")
    print("=" * 70)
    
    # Find clusters in the dead layer
    from collections import defaultdict
    
    # Simple clustering by finding nearest neighbors
    words = list(enc.dead_positions.keys())
    positions = {w: enc.dead_positions[w] for w in words}
    
    # Find some example clusters
    seed_words = ['washington', 'lincoln', 'paris', 'tokyo', 'pasta', 'bread', 'ls', 'grep']
    
    print("\n  Word neighborhoods (emergent clusters):")
    for seed in seed_words:
        if seed not in positions:
            continue
        
        seed_pos = positions[seed]
        distances = []
        for w, pos in positions.items():
            if w != seed:
                dist = np.linalg.norm(seed_pos - pos)
                distances.append((w, dist))
        
        distances.sort(key=lambda x: x[1])
        neighbors = [w for w, d in distances[:5]]
        print(f"    {seed}: {', '.join(neighbors)}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Facts ingested: {len(facts)}")
    print(f"  Vocabulary learned: {enc.stats()['dead_vocabulary']} words")
    print(f"  QA accuracy: {total_correct/total_queries:.1%}")
    print(f"  Growth rings: {len(enc.growth_rings)}")


if __name__ == "__main__":
    main()
