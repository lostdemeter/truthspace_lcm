#!/usr/bin/env python3
"""
Multi-Domain Knowledge Engine Demo

Demonstrates the generalized φ-based knowledge system working across
multiple domains (bash, cooking) with automatic routing.

Usage:
    python experiments/multi_domain_demo.py
    python experiments/multi_domain_demo.py chat
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.domains.base import BASH_DOMAIN, COOKING_DOMAIN
from truthspace_lcm.core import (
    KnowledgeEngine, 
    MultiDomainEngine,
    BASH_COMMAND_KEYWORDS,
    COOKING_COMMAND_KEYWORDS,
)


# Sample training data
BASH_QA = [
    ("How do I list files?", "Use 'ls' to list files in a directory."),
    ("Show all files including hidden", "Use 'ls -la' to show all files including hidden ones."),
    ("How do I show disk space?", "Use 'df -h' to show disk space usage."),
    ("Check disk usage", "Use 'df -h' to check how much disk space is used."),
    ("What processes are running?", "Use 'ps aux' to see running processes."),
    ("Show running processes", "Use 'ps aux' or 'top' to view processes."),
    ("How do I check network connections?", "Use 'netstat -tuln' to check network connections."),
    ("Show network status", "Use 'netstat' or 'ss' to see network status."),
    ("Who is logged in?", "Use 'who' to see logged in users."),
    ("Show logged in users", "Use 'who' or 'w' to display logged in users."),
    ("What's my IP address?", "Use 'ip addr' to show your IP addresses."),
    ("Show network interfaces", "Use 'ip addr' or 'ifconfig' to show interfaces."),
]

COOKING_QA = [
    ("How do I cook pasta?", "Boil water, add salt, then boil pasta for 8-10 minutes until al dente."),
    ("How to make spaghetti?", "Boil water with salt, add spaghetti, cook for 8 minutes."),
    ("How do I make a steak?", "Sauté the steak in a hot pan with butter for 3-4 minutes per side."),
    ("Cook beef", "Sauté or grill beef to your preferred doneness."),
    ("How to make bread?", "Knead the dough for 10 minutes, let rise, then bake at 400F for 30 minutes."),
    ("Bake a loaf", "Knead dough, proof for 1 hour, bake at 400F until golden."),
    ("How to cook vegetables?", "Steam vegetables for 5-7 minutes until tender but still crisp."),
    ("Prepare broccoli", "Steam broccoli for 5 minutes or sauté with garlic."),
    ("How to make soup?", "Simmer ingredients in broth for 30-45 minutes until flavors meld."),
    ("Cook chicken soup", "Simmer chicken with vegetables in broth for 45 minutes."),
    ("How to cook eggs?", "Fry eggs in butter over medium heat for 2-3 minutes."),
    ("Make scrambled eggs", "Whisk eggs, then sauté in butter while stirring."),
]

SOCIAL_RESPONSES = {
    'GREETING': [
        "Hey! I can help with Linux commands or cooking. What do you need?",
        "Hello! Ask me about bash commands or cooking techniques.",
        "Hi there! Try 'list files' or 'how to cook pasta'.",
    ],
    'FAREWELL': [
        "Goodbye! Happy computing and cooking!",
        "See you later!",
        "Bye! Come back anytime.",
    ],
    'CHITCHAT': [
        "I'm doing great! What can I help you with?",
        "All good here! Ask about files, disk, processes, or cooking.",
    ],
    'POLITENESS': [
        "You're welcome! Anything else?",
        "Happy to help!",
    ],
}


def demo():
    """Run non-interactive demo."""
    print("=" * 70)
    print("MULTI-DOMAIN KNOWLEDGE ENGINE DEMO")
    print("=" * 70)
    print()
    
    # Create multi-domain engine
    multi = MultiDomainEngine()
    
    print("Loading domains...")
    multi.add_domain(BASH_DOMAIN, BASH_QA)
    print(f"  ✓ Bash domain: {len(BASH_DOMAIN.anchors)} anchors, {len(BASH_QA)} Q&A pairs")
    
    multi.add_domain(COOKING_DOMAIN, COOKING_QA)
    print(f"  ✓ Cooking domain: {len(COOKING_DOMAIN.anchors)} anchors, {len(COOKING_QA)} Q&A pairs")
    
    print()
    print("=" * 70)
    print("TEST QUERIES")
    print("=" * 70)
    
    test_queries = [
        # Bash queries
        "list my files",
        "show disk usage",
        "what processes are running",
        "network connections",
        # Cooking queries
        "how to cook pasta",
        "make a steak",
        "bake bread",
        "cook vegetables",
        # Ambiguous (should route correctly)
        "show me running",  # processes
        "boil something",   # cooking
    ]
    
    for query in test_queries:
        result = multi.query(query, execute=False)
        print(f"\nQ: {query}")
        print(f"   → Domain: {result.domain}, Anchor: {result.anchor}, Score: {result.score:.1f}")
        print(f"   → {result.response[:60]}..." if len(result.response) > 60 else f"   → {result.response}")
    
    print()
    print("=" * 70)
    print("DOMAIN ROUTING ACCURACY")
    print("=" * 70)
    
    # Test routing accuracy
    bash_queries = ["list files", "disk space", "running processes", "network status", "logged in users"]
    cooking_queries = ["cook pasta", "make steak", "bake bread", "steam vegetables", "simmer soup"]
    
    bash_correct = sum(1 for q in bash_queries if multi.query(q, execute=False).domain == "bash")
    cooking_correct = sum(1 for q in cooking_queries if multi.query(q, execute=False).domain == "cooking")
    
    print(f"\nBash routing: {bash_correct}/{len(bash_queries)} correct")
    print(f"Cooking routing: {cooking_correct}/{len(cooking_queries)} correct")
    print(f"Total: {bash_correct + cooking_correct}/{len(bash_queries) + len(cooking_queries)} ({100*(bash_correct + cooking_correct)/(len(bash_queries) + len(cooking_queries)):.0f}%)")


def interactive_chat():
    """Interactive multi-domain chat."""
    print("=" * 70)
    print("MULTI-DOMAIN CHATBOT")
    print("=" * 70)
    print()
    
    # Create engines with command keywords for social detection
    multi = MultiDomainEngine()
    
    # We need to create engines with proper command keywords
    bash_engine = KnowledgeEngine(BASH_DOMAIN, command_keywords=BASH_COMMAND_KEYWORDS)
    bash_engine.ingest(BASH_QA)
    multi.engines["bash"] = bash_engine
    
    cooking_engine = KnowledgeEngine(COOKING_DOMAIN, command_keywords=COOKING_COMMAND_KEYWORDS)
    cooking_engine.ingest(COOKING_QA)
    multi.engines["cooking"] = cooking_engine
    
    print("Domains loaded: bash, cooking")
    print("Try: 'hey', 'list files', 'how to cook pasta'")
    print("Commands: quit, help, debug")
    print("=" * 70)
    
    debug_mode = False
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("Bot: Goodbye! 👋")
            break
        
        if query.lower() == 'debug':
            debug_mode = not debug_mode
            print(f"Bot: Debug mode {'ON' if debug_mode else 'OFF'}")
            continue
        
        if query.lower() == 'help':
            print("\nBot: I can help with:")
            print("  • Linux commands: 'list files', 'disk space', 'processes'")
            print("  • Cooking: 'cook pasta', 'make steak', 'bake bread'")
            print("  • Social: 'hey', 'thanks', 'bye'")
            continue
        
        # Query all engines and find best
        result = multi.query(query, execute=False)
        
        if debug_mode:
            print(f"[DEBUG] domain={result.domain}, anchor={result.anchor}, score={result.score}, social={result.is_social}")
        
        if result.is_social:
            social_type = result.response.strip('[]')
            responses = SOCIAL_RESPONSES.get(social_type, SOCIAL_RESPONSES['GREETING'])
            print(f"Bot: {random.choice(responses)}")
        elif result.score > 0:
            print(f"Bot: [{result.domain}] {result.response}")
        else:
            print("Bot: I'm not sure about that. Try asking about files, processes, or cooking.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':
        interactive_chat()
    else:
        demo()
