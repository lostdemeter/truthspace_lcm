#!/usr/bin/env python3
"""
Emergent Chatbot Runner

A simple script to run the emergent chatbot with all data sources loaded.
"""

import sys
sys.path.insert(0, 'experiments')

from pathlib import Path
from experiments.emergent_chatbot import EmergentChatbot


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " EMERGENT CHATBOT ".center(68) + "║")
    print("║" + " Powered by Self-Discovered Dimensions ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Create chatbot
    chatbot = EmergentChatbot(model="qwen2:latest")
    
    # Data sources to load
    base_path = Path(__file__).parent
    sources = [
        (base_path / "truthspace_lcm/gears/corpus/corpus_llm_live.json", "behavioral"),
        (base_path / "truthspace_lcm/gears/corpus/corpus_rich_behavioral.json", "characters"),
        (base_path / "truthspace_lcm/gears/corpus/corpus_knowledge.json", "knowledge"),
        (base_path / "truthspace_lcm/corpus_curated.json", "curated"),
        (base_path / "truthspace_lcm/corpus_holmes_quality.json", "holmes"),
    ]
    
    print("Loading knowledge base...")
    for path, name in sources:
        if path.exists():
            chatbot.ingest_corpus(str(path), name)
    
    print("\nLearning dimensions from data...")
    chatbot.learn_dimensions(min_variance=0.02, max_dims=15)
    
    print()
    print("─" * 70)
    print(f"Knowledge base: {chatbot.total_frames} frames")
    print(f"Agents: {len(chatbot.agents)}")
    print(f"Dimensions discovered: {len(chatbot.dimensions)}")
    print("─" * 70)
    print()
    print("Commands:")
    print("  Type a question to chat")
    print("  'dims'  - Show discovered dimensions")
    print("  'stats' - Show statistics")
    print("  'quit'  - Exit")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if query.lower() == 'stats':
            print(f"\n  Frames: {chatbot.total_frames}")
            print(f"  Agents: {len(chatbot.agents)}")
            print(f"  Dimensions: {len(chatbot.dimensions)}")
            print(f"  Learning cycles: {chatbot.learning_cycles}")
            print()
            continue
        
        if query.lower() == 'dims':
            print("\n  Discovered Dimensions:")
            for dim in chatbot.dimensions:
                print(f"    {dim['name']}: {dim['negative_pole']} ↔ {dim['positive_pole']} ({dim['variance']*100:.1f}%)")
            print()
            continue
        
        # Chat
        print()
        response = chatbot.chat(query)
        print(f"Bot: {response}")
        print()


if __name__ == "__main__":
    main()
