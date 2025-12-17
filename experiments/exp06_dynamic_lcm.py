#!/usr/bin/env python3
"""
Experiment 6: Dynamic LCM Prototype

Test the scalable, generalizable geometric LCM with:
1. Multiple knowledge domains (bash, cooking, medical, social)
2. Dynamic primitive discovery
3. Domain detection and resolution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.dynamic_lcm import DynamicLCM


# =============================================================================
# KNOWLEDGE CORPORA - Different domains to test scalability
# =============================================================================

BASH_KNOWLEDGE = [
    {"content": "ls", "description": "list files in directory", "domain": "bash"},
    {"content": "ls -la", "description": "list all files with details", "domain": "bash"},
    {"content": "mkdir -p", "description": "create new directory folder", "domain": "bash"},
    {"content": "touch", "description": "create new empty file", "domain": "bash"},
    {"content": "rm", "description": "delete remove file", "domain": "bash"},
    {"content": "rm -rf", "description": "delete remove directory recursively", "domain": "bash"},
    {"content": "cp", "description": "copy file to destination", "domain": "bash"},
    {"content": "mv", "description": "move rename file", "domain": "bash"},
    {"content": "cat", "description": "read show file contents", "domain": "bash"},
    {"content": "grep", "description": "search find text pattern in files", "domain": "bash"},
    {"content": "find", "description": "search locate files by name", "domain": "bash"},
    {"content": "ps", "description": "list show running processes", "domain": "bash"},
    {"content": "kill", "description": "stop terminate process", "domain": "bash"},
    {"content": "df", "description": "show disk space storage usage", "domain": "bash"},
    {"content": "du", "description": "show directory size usage", "domain": "bash"},
    {"content": "chmod", "description": "change file permissions", "domain": "bash"},
    {"content": "chown", "description": "change file owner", "domain": "bash"},
    {"content": "tar -czf", "description": "create compressed archive", "domain": "bash"},
    {"content": "tar -xzf", "description": "extract compressed archive", "domain": "bash"},
    {"content": "curl", "description": "download fetch url web", "domain": "bash"},
    {"content": "ssh", "description": "connect remote server secure shell", "domain": "bash"},
    {"content": "scp", "description": "copy file to remote server", "domain": "bash"},
]

COOKING_KNOWLEDGE = [
    {"content": "Preheat oven to 350°F", "description": "preheat oven temperature baking recipe kitchen", "domain": "cooking"},
    {"content": "Boil water in a large pot", "description": "boil water pot cooking recipe kitchen", "domain": "cooking"},
    {"content": "Chop onions finely", "description": "chop onions vegetables knife cutting recipe", "domain": "cooking"},
    {"content": "Sauté garlic in olive oil", "description": "saute fry garlic oil pan stove recipe", "domain": "cooking"},
    {"content": "Simmer for 20 minutes", "description": "simmer low heat stove cooking recipe", "domain": "cooking"},
    {"content": "Season with salt and pepper", "description": "season salt pepper spices taste recipe", "domain": "cooking"},
    {"content": "Whisk eggs until fluffy", "description": "whisk eggs beat mixing bowl recipe", "domain": "cooking"},
    {"content": "Knead dough for 10 minutes", "description": "knead dough bread baking recipe", "domain": "cooking"},
    {"content": "Let dough rise for 1 hour", "description": "rise dough yeast bread baking recipe", "domain": "cooking"},
    {"content": "Grill on medium-high heat", "description": "grill barbecue cooking outdoor recipe", "domain": "cooking"},
    {"content": "Marinate overnight", "description": "marinate meat flavor overnight recipe", "domain": "cooking"},
    {"content": "Blend until smooth", "description": "blend blender smooth puree recipe", "domain": "cooking"},
    {"content": "Fold in gently", "description": "fold gentle batter mixing recipe", "domain": "cooking"},
    {"content": "Reduce sauce by half", "description": "reduce sauce thicken stove recipe", "domain": "cooking"},
    {"content": "Rest meat for 5 minutes", "description": "rest meat juices cooking recipe", "domain": "cooking"},
]

MEDICAL_KNOWLEDGE = [
    {"content": "Take with food to avoid stomach upset", "description": "medicine food stomach", "domain": "medical"},
    {"content": "Apply ice for 20 minutes", "description": "ice cold swelling injury", "domain": "medical"},
    {"content": "Elevate the affected limb", "description": "raise elevate limb swelling", "domain": "medical"},
    {"content": "Take temperature orally", "description": "measure temperature fever thermometer", "domain": "medical"},
    {"content": "Check blood pressure regularly", "description": "measure blood pressure heart", "domain": "medical"},
    {"content": "Stay hydrated with clear fluids", "description": "drink water fluids hydration sick", "domain": "medical"},
    {"content": "Rest and avoid strenuous activity", "description": "rest recovery avoid exercise", "domain": "medical"},
    {"content": "Clean wound with soap and water", "description": "clean wash wound cut infection", "domain": "medical"},
    {"content": "Apply antibiotic ointment", "description": "antibiotic cream wound infection", "domain": "medical"},
    {"content": "Cover with sterile bandage", "description": "bandage cover wound sterile", "domain": "medical"},
    {"content": "Seek emergency care immediately", "description": "emergency urgent hospital serious", "domain": "medical"},
    {"content": "Take pain reliever as directed", "description": "pain medicine painkiller dose", "domain": "medical"},
    {"content": "Monitor for signs of infection", "description": "watch infection redness swelling fever", "domain": "medical"},
    {"content": "Consult doctor if symptoms persist", "description": "doctor consult symptoms continue", "domain": "medical"},
]

SOCIAL_KNOWLEDGE = [
    {"content": "Hello! How can I help you today?", "description": "hello hi greeting help", "domain": "social"},
    {"content": "I understand how you feel.", "description": "understand feeling empathy emotion", "domain": "social"},
    {"content": "That sounds frustrating.", "description": "frustrating difficult hard sympathy", "domain": "social"},
    {"content": "I'm here to help.", "description": "help support assist here", "domain": "social"},
    {"content": "You're welcome!", "description": "welcome thanks gratitude response", "domain": "social"},
    {"content": "I'm sorry to hear that.", "description": "sorry sad sympathy bad news", "domain": "social"},
    {"content": "That's great news!", "description": "great good happy congratulations", "domain": "social"},
    {"content": "Take care of yourself.", "description": "care goodbye wellbeing", "domain": "social"},
    {"content": "Let me know if you need anything.", "description": "help offer assistance need", "domain": "social"},
    {"content": "I appreciate your patience.", "description": "thank patience appreciate wait", "domain": "social"},
    {"content": "How are you doing today?", "description": "how feeling today check", "domain": "social"},
    {"content": "Is there anything else I can help with?", "description": "more help anything else", "domain": "social"},
]


# =============================================================================
# TEST QUERIES
# =============================================================================

TEST_QUERIES = [
    # Bash queries
    ("list the files", "bash", "ls"),
    ("create a new directory", "bash", "mkdir"),
    ("delete this file", "bash", "rm"),
    ("find all python files", "bash", "find"),
    ("show disk space", "bash", "df"),
    ("connect to remote server", "bash", "ssh"),
    
    # Cooking queries
    ("how do I boil water", "cooking", "Boil"),
    ("chop the onions", "cooking", "Chop"),
    ("season the dish", "cooking", "Season"),
    ("how long to simmer", "cooking", "Simmer"),
    ("make the dough rise", "cooking", "rise"),
    
    # Medical queries
    ("I have a fever", "medical", "temperature"),
    ("how to clean a wound", "medical", "Clean wound"),
    ("should I see a doctor", "medical", "Consult doctor"),
    ("my arm is swollen", "medical", "ice"),
    ("take medicine with food", "medical", "food"),
    
    # Social queries
    ("hello", "social", "Hello"),
    ("thank you", "social", "welcome"),
    ("I'm feeling sad", "social", "sorry"),
    ("that's great", "social", "great"),
    ("goodbye", "social", "care"),
    
    # Ambiguous queries (test domain detection)
    ("I'm feeling a bit under the weather", "medical", None),  # Could be social or medical
    ("cut the file", "bash", None),  # "cut" could be cooking or bash
    ("process the data", "bash", None),  # Could be bash
]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 6: Dynamic LCM Prototype")
    print("=" * 70)
    print()
    
    # Create LCM
    lcm = DynamicLCM(dimensions=16)
    
    print("--- Initial State ---")
    stats = lcm.stats()
    print(f"Dimensions: {stats['dimensions']}")
    print(f"Seed primitives: {stats['seed_primitives']}")
    print(f"Emergent primitives: {stats['emergent_primitives']}")
    print()
    
    # Ingest all knowledge
    print("--- Ingesting Knowledge ---")
    all_knowledge = BASH_KNOWLEDGE + COOKING_KNOWLEDGE + MEDICAL_KNOWLEDGE + SOCIAL_KNOWLEDGE
    lcm.ingest_batch(all_knowledge)
    
    stats = lcm.stats()
    print(f"Total entries: {stats['entries']}")
    print(f"Domains: {list(stats['domain_sizes'].keys())}")
    print(f"Domain sizes: {stats['domain_sizes']}")
    print(f"Emergent primitives discovered: {stats['emergent_primitives']}")
    print(f"Unique words: {stats['unique_words']}")
    print()
    
    # Show emergent primitives
    emergent = [p for p in lcm.primitives if not p.is_seed]
    if emergent:
        print("--- Emergent Primitives ---")
        for p in emergent:
            print(f"  {p.name}: {p.keywords}")
        print()
    
    # Test resolution
    print("--- Query Resolution Tests ---")
    print()
    
    correct_domain = 0
    correct_content = 0
    total = 0
    
    for query, expected_domain, expected_content_substr in TEST_QUERIES:
        detected_domain, entry, similarity = lcm.resolve_with_domain_detection(query)
        
        domain_match = detected_domain == expected_domain
        content_match = expected_content_substr is None or (
            entry and expected_content_substr.lower() in entry.content.lower()
        )
        
        if domain_match:
            correct_domain += 1
        if content_match:
            correct_content += 1
        total += 1
        
        status = "✓" if (domain_match and content_match) else "✗"
        print(f"{status} \"{query}\"")
        print(f"    Domain: {detected_domain} (expected: {expected_domain}) {'✓' if domain_match else '✗'}")
        if entry:
            print(f"    Result: {entry.content[:50]}... (sim: {similarity:.2f})")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Domain detection accuracy: {correct_domain}/{total} ({100*correct_domain/total:.1f}%)")
    print(f"Content match accuracy: {correct_content}/{total} ({100*correct_content/total:.1f}%)")
    print()
    
    # Test encoding explanation
    print("--- Encoding Explanation Example ---")
    print(lcm.explain_encoding("create a new file"))
    print()
    print(lcm.explain_encoding("I'm feeling sad today"))
    print()
    
    # Save and reload test
    print("--- Persistence Test ---")
    lcm.save("/tmp/test_lcm.json")
    
    lcm2 = DynamicLCM()
    lcm2.load("/tmp/test_lcm.json")
    
    stats2 = lcm2.stats()
    print(f"Reloaded: {stats2['entries']} entries, {stats2['emergent_primitives']} emergent primitives")
    
    # Verify resolution still works
    _, entry, sim = lcm2.resolve_with_domain_detection("list the files")
    print(f"Test query after reload: 'list the files' → {entry.content if entry else 'None'}")
    
    return lcm


if __name__ == "__main__":
    lcm = run_experiment()
