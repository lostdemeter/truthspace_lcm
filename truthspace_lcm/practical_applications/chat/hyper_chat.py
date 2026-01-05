"""
HyperChat - Interactive Chat using HyperMapping

A conversational interface using HyperMapping-based architecture.
Replaces EmergentChat with a cleaner, more geometric approach.

Key differences from EmergentChat:
- Uses ChatPipeline instead of ConversationalChain
- Uses IntentSpace instead of IntentClassifier
- Uses CodeSpace instead of PythonCodeGear
- All routing is geometric (position-based)
- Bootstrap patterns are the ONLY hardcoding

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig, Intent
from truthspace_lcm.core.knowledge_space import KnowledgeSpace
from truthspace_lcm.core.code_space import CodeSpace


class HyperChat:
    """
    Interactive chat interface using HyperMapping.
    
    Features:
    - Geometric intent detection (bootstrap + position matching)
    - Knowledge storage and retrieval via KnowledgeSpace
    - Code generation via CodeSpace
    - Feedback-based learning
    - Full persistence support
    
    Usage:
        chat = HyperChat(debug=True)
        chat.add_knowledge("Python is a programming language")
        
        response = chat.query("What is Python?")
        chat.feedback(success=True)
        
        chat.save("chat_state.json")
    """
    
    def __init__(self, debug: bool = False, dims: int = 8):
        self.debug = debug
        
        # Create chat pipeline
        config = ChatConfig(debug=debug, dims=dims)
        self.pipeline = ChatPipeline(config)
        
        # Command history
        self.history: List[str] = []
        
        # Pending commands awaiting confirmation
        self.pending_commands: List[str] = []
    
    # -------------------------------------------------------------------------
    # Knowledge Management
    # -------------------------------------------------------------------------
    
    def add_knowledge(self, text: str, source: str = "user") -> None:
        """Add knowledge to the space."""
        self.pipeline.add_knowledge(text, source)
    
    def load_knowledge(self, path: str) -> bool:
        """Load knowledge from file."""
        return self.pipeline.load_knowledge(path)
    
    def save_knowledge(self, path: str) -> bool:
        """Save knowledge to file."""
        return self.pipeline.save_knowledge(path)
    
    def list_knowledge(self, limit: int = 20) -> List[str]:
        """List knowledge items."""
        items = []
        for mapping in self.pipeline.knowledge_space._mappings[:limit]:
            items.append(mapping.input[:100])
        return items
    
    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------
    
    def query(self, question: str) -> str:
        """
        Process a query through the chat pipeline.
        
        This is GEOMETRIC - uses position-based matching for:
        1. Intent detection
        2. Knowledge retrieval
        3. Code generation
        """
        self.history.append(question)
        return self.pipeline.chat(question)
    
    def feedback(self, success: bool) -> bool:
        """Provide feedback on the last response."""
        return self.pipeline.feedback(success)
    
    def learn_intent(self, query: str, correct_intent: str) -> None:
        """Correct intent detection for a query."""
        try:
            intent = Intent[correct_intent.upper()]
            self.pipeline.learn_intent(query, intent)
        except KeyError:
            if self.debug:
                print(f"[DEBUG] Unknown intent: {correct_intent}")
    
    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------
    
    def handle_command(self, cmd: str) -> Optional[str]:
        """Handle special commands."""
        cmd = cmd.strip().lower()
        
        if cmd in ['/help', '/h', '/?']:
            return self._help_text()
        
        if cmd == '/stats':
            stats = self.pipeline.get_stats()
            return f"""Statistics:
  Knowledge concepts: {stats['knowledge']['total_mappings']}
  Persisting: {stats['knowledge']['persisting_mappings']}
  Intent templates: {stats['intent_templates']}
  Code patterns: {len(self.pipeline.code_space.list_patterns())}"""
        
        if cmd == '/debug':
            self.debug = not self.debug
            self.pipeline.config.debug = self.debug
            return f"Debug mode: {'ON' if self.debug else 'OFF'}"
        
        if cmd.startswith('/add '):
            text = cmd[5:].strip()
            self.add_knowledge(text)
            return f"Added knowledge: {text[:50]}..."
        
        if cmd.startswith('/save '):
            path = cmd[6:].strip()
            if self.save_knowledge(path):
                return f"Saved to: {path}"
            return f"Failed to save to: {path}"
        
        if cmd.startswith('/load '):
            path = cmd[6:].strip()
            if self.load_knowledge(path):
                return f"Loaded from: {path}"
            return f"Failed to load from: {path}"
        
        if cmd == '/knowledge':
            items = self.list_knowledge(10)
            if items:
                return "Knowledge items:\n  " + "\n  ".join(items)
            return "No knowledge items"
        
        if cmd == '/patterns':
            patterns = self.pipeline.code_space.list_patterns()[:10]
            if patterns:
                lines = [f"  {p['name']}: {p['description'][:40]}..." for p in patterns]
                return "Code patterns:\n" + "\n".join(lines)
            return "No code patterns"
        
        if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
            return None  # Signal to exit
        
        return f"Unknown command: {cmd}\nType /help for available commands."
    
    def _help_text(self) -> str:
        return """
HyperChat - Commands:

  /help, /h       Show this help
  /stats          Show statistics
  /debug          Toggle debug mode
  /add <text>     Add knowledge
  /save <path>    Save knowledge to file
  /load <path>    Load knowledge from file
  /knowledge      List knowledge items
  /patterns       List code patterns
  /quit, /q       Exit

Ask questions like:
  What is Python?
  Tell me about machine learning

Request code like:
  Write code to print hello world
  Python function to calculate sum
"""
    
    # -------------------------------------------------------------------------
    # Interactive Loop
    # -------------------------------------------------------------------------
    
    def run(self):
        """Run the interactive chat loop."""
        print()
        print("═" * 60)
        print("  HyperChat - Geometric Conversational Interface")
        print("  Using HyperMapping for intent, knowledge, and code")
        print("═" * 60)
        print()
        
        stats = self.pipeline.get_stats()
        print(f"  Knowledge: {stats['knowledge']['total_mappings']} concepts")
        print(f"  Code patterns: {len(self.pipeline.code_space.list_patterns())}")
        print("  Type /help for commands, /quit to exit")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    response = self.handle_command(user_input)
                    if response is None:
                        print("\nGoodbye!")
                        break
                    print(f"\n{response}\n")
                    continue
                
                # Process query
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
    
    def __repr__(self) -> str:
        return f"HyperChat(knowledge={len(self.pipeline.knowledge_space)}, debug={self.debug})"


def main(debug: bool = False, knowledge_path: str = None):
    """Entry point for the chat application."""
    chat = HyperChat(debug=debug)
    
    if knowledge_path:
        chat.load_knowledge(knowledge_path)
    
    chat.run()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HyperChat - Geometric Conversational Interface")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--knowledge", "-k", type=str, help="Load knowledge from file")
    args = parser.parse_args()
    
    sys.exit(main(
        debug=args.debug,
        knowledge_path=args.knowledge,
    ))
