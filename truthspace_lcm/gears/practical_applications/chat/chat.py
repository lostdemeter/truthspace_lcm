"""
Interactive Chat Application for Emergent Conversational Chain

A conversational interface using truly emergent response generation.
LLM is used ONLY for corpus building, NEVER for response generation.

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from typing import Optional, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.gears.core import ConversationalChain


# Default LLM configuration
DEFAULT_LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2:latest"

# Default seed topics for knowledge building
DEFAULT_SEED_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "programming",
    "science",
    "philosophy",
]


class EmergentChat:
    """
    Interactive chat interface using emergent conversational chain.
    
    Features:
    - Truly emergent responses (no LLM during conversation)
    - Corpus building from LLM as knowledge resource
    - Topic exploration and learning
    - Debug mode to see emergent patterns
    """
    
    def __init__(self, 
                 llm_url: str = DEFAULT_LLM_URL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 debug: bool = False):
        self.debug = debug
        
        # Create conversational chain
        self.chain = ConversationalChain()
        self.chain.configure_llm(llm_url, llm_model)
        
        # Command history
        self.history: List[str] = []
    
    def build_knowledge(self, seed_topics: List[str] = None, expand: bool = True):
        """Build knowledge corpus from seed topics."""
        topics = seed_topics or DEFAULT_SEED_TOPICS
        
        print(f"\n{'='*60}")
        print("BUILDING KNOWLEDGE CORPUS")
        print("(LLM used as knowledge resource only)")
        print(f"{'='*60}")
        print(f"Seed topics: {', '.join(topics)}")
        
        self.chain.build_corpus(topics, expand=expand)
        
        stats = self.chain.get_stats()
        print(f"\nCorpus built:")
        print(f"  Topics: {stats['topics']}")
        print(f"  Facts: {stats['corpus_items']}")
        print(f"  Definitions: {stats['definitions']}")
        print(f"  LLM calls (corpus building): {stats['corpus_building_calls']}")
    
    def load_corpus(self, path: str):
        """Load pre-built corpus from file."""
        self.chain.load_corpus(path)
        stats = self.chain.get_stats()
        print(f"Loaded corpus: {stats['topics']} topics, {stats['corpus_items']} items")
    
    def save_corpus(self, path: str):
        """Save corpus to file for later use."""
        self.chain.save_corpus(path)
        print(f"Saved corpus to {path}")
    
    def query(self, question: str) -> str:
        """Process a query through the emergent chain."""
        if self.debug:
            print(f"\n[DEBUG] Input: {question}")
            topics = self.chain._extract_topics(question)
            print(f"[DEBUG] Extracted topics: {topics}")
        
        response = self.chain.chat(question)
        
        if self.debug:
            stats = self.chain.get_stats()
            print(f"[DEBUG] Conversation LLM calls: {stats['conversation_calls']} (should be 0)")
        
        return response
    
    def handle_command(self, cmd: str) -> Optional[str]:
        """Handle special commands."""
        cmd = cmd.strip().lower()
        
        if cmd in ['/help', '/h', '/?']:
            return self._help_text()
        
        if cmd == '/topics':
            topics = self.chain.list_topics()
            return f"Known topics ({len(topics)}):\n  " + '\n  '.join(topics[:30])
        
        if cmd == '/stats':
            stats = self.chain.get_stats()
            return f"""Statistics:
  Topics: {stats['topics']}
  Corpus items: {stats['corpus_items']}
  Definitions: {stats['definitions']}
  Dimensions: {stats['dimensions']}
  LLM calls (corpus): {stats['corpus_building_calls']}
  LLM calls (chat): {stats['conversation_calls']} (should be 0)
  Conversation turns: {stats['history_length']}"""
        
        if cmd == '/debug':
            self.debug = not self.debug
            return f"Debug mode: {'ON' if self.debug else 'OFF'}"
        
        if cmd.startswith('/learn '):
            topic = cmd[7:].strip()
            if self.chain.learn_topic(topic):
                return f"Learned about: {topic}"
            return f"Failed to learn about: {topic} (check LLM connection)"
        
        if cmd.startswith('/info '):
            topic = cmd[6:].strip()
            info = self.chain.get_topic_info(topic)
            if info['fact_count'] > 0:
                result = f"Topic: {info['topic']}\n"
                if info['definition']:
                    result += f"Definition: {info['definition']}\n"
                result += f"Facts ({info['fact_count']}):\n"
                for fact in info['facts'][:5]:
                    result += f"  • {fact}\n"
                return result.strip()
            return f"No information about: {topic}"
        
        if cmd.startswith('/save '):
            path = cmd[6:].strip()
            self.save_corpus(path)
            return f"Saved corpus to: {path}"
        
        if cmd.startswith('/load '):
            path = cmd[6:].strip()
            self.load_corpus(path)
            return f"Loaded corpus from: {path}"
        
        if cmd == '/books':
            books = self.chain.get_available_books()
            return f"Available books:\n  " + '\n  '.join(books)
        
        if cmd.startswith('/book '):
            book_name = cmd[6:].strip().lower().replace(' ', '_')
            print(f"  Loading {book_name}...")
            if self.chain.load_book(book_name=book_name):
                stats = self.chain.get_stats()
                return f"Loaded {self.chain.book_title}: {stats['topics']} topics, {stats['corpus_items']} items"
            return f"Failed to load book: {book_name}"
        
        if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
            return None  # Signal to exit
        
        return f"Unknown command: {cmd}\nType /help for available commands."
    
    def _help_text(self) -> str:
        return """
Emergent Chat - Commands:

  /help, /h       Show this help
  /topics         List known topics
  /stats          Show statistics
  /debug          Toggle debug mode
  /learn <topic>  Learn about a new topic
  /info <topic>   Show info about a topic
  /save <path>    Save corpus to file
  /load <path>    Load corpus from file
  /books          List available literary works
  /book <name>    Load a literary work (e.g., /book moby_dick)
  /quit, /q       Exit

Ask questions like:
  What is artificial intelligence?
  How does machine learning work?
  Tell me about Queequeg

Note: All responses are EMERGENT - no LLM is used during conversation.
The LLM is only used as a knowledge resource during corpus building.
"""
    
    def run(self):
        """Run the interactive chat loop."""
        print()
        print("═" * 60)
        print("  Emergent Conversational Chat")
        print("  Truly emergent responses - no LLM during conversation")
        print("═" * 60)
        print()
        
        stats = self.chain.get_stats()
        print(f"  Knowledge: {stats['topics']} topics, {stats['corpus_items']} facts")
        print(f"  LLM calls during chat: {stats['conversation_calls']} (always 0)")
        print("  Type /help for commands, /quit to exit")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                self.history.append(user_input)
                
                # Handle commands
                if user_input.startswith('/'):
                    response = self.handle_command(user_input)
                    if response is None:
                        print("\nGoodbye!")
                        break
                    print(f"\n{response}\n")
                    continue
                
                # Process query (emergent response)
                response = self.query(user_input)
                print(f"\nBot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                if self.debug:
                    import traceback
                    traceback.print_exc()


def main(debug: bool = False, corpus_path: str = None, 
         seed_topics: List[str] = None, no_build: bool = False):
    """Entry point for the chat application."""
    chat = EmergentChat(debug=debug)
    
    if corpus_path:
        chat.load_corpus(corpus_path)
    elif not no_build:
        chat.build_knowledge(seed_topics)
    
    chat.run()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emergent Conversational Chat")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--corpus", "-c", type=str, help="Load corpus from file")
    parser.add_argument("--topics", "-t", nargs="+", help="Seed topics for knowledge building")
    parser.add_argument("--no-build", action="store_true", help="Don't build corpus on start")
    args = parser.parse_args()
    
    sys.exit(main(
        debug=args.debug,
        corpus_path=args.corpus,
        seed_topics=args.topics,
        no_build=args.no_build,
    ))
