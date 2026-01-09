"""
Interactive Chat Application for Emergent Conversational Chain

A conversational interface using truly emergent response generation.
LLM is used ONLY for corpus building, NEVER for response generation.

Now with gear-based routing:
- Knowledge queries → ConversationalChain (emergent)
- Tool calls → GearOrchestrator (plans + commands)

Author: Lesley Gushurst
License: GPLv3
"""

import sys
import threading
import time
from typing import Optional, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.core import ConversationalChain
from truthspace_lcm.core.classifiers.geometric_intent_classifier import (
    GeometricIntentClassifier, Intent, IntentMatch, create_geometric_classifier
)
from truthspace_lcm.core.orchestrators.gear_orchestrator import GearOrchestrator


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
    - Gear-based routing for tool calls
    """
    
    def __init__(self, 
                 llm_url: str = DEFAULT_LLM_URL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 debug: bool = False,
                 enable_tools: bool = True):
        self.debug = debug
        self.enable_tools = enable_tools
        self.llm_url = llm_url
        self.llm_model = llm_model
        
        # Create conversational chain
        self.chain = ConversationalChain()
        self.chain.configure_llm(llm_url, llm_model)
        
        # Geometric intent classifier (pure geometry, no regex)
        self.intent_classifier = create_geometric_classifier()
        
        # Gear orchestrator for tool calls
        self.orchestrator: Optional[GearOrchestrator] = None
        if enable_tools:
            self.orchestrator = GearOrchestrator()
            self.orchestrator.configure_llm(llm_url, llm_model)
        
        # Python code gear for code generation
        self.python_gear = None
        try:
            from truthspace_lcm.core.gears.python_code_gear import PythonCodeGear
            self.python_gear = PythonCodeGear()
            self.python_gear.configure_llm(llm_url, llm_model)
        except ImportError:
            pass
        
        # Command history
        self.history: List[str] = []
        
        # Pending commands awaiting confirmation
        self.pending_commands: List[str] = []
        
        # Background corpus building
        self._corpus_build_thread = None
        self._stop_building = False
        self.auto_build = False
        self.build_interval = 60  # seconds
    
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
    
    def start_auto_build(self):
        """Start background corpus building."""
        if self._corpus_build_thread and self._corpus_build_thread.is_alive():
            return False
        
        self._stop_building = False
        self.auto_build = True
        self._corpus_build_thread = threading.Thread(
            target=self._background_build_loop, 
            daemon=True
        )
        self._corpus_build_thread.start()
        return True
    
    def stop_auto_build(self):
        """Stop background corpus building."""
        self._stop_building = True
        self.auto_build = False
        if self._corpus_build_thread:
            self._corpus_build_thread.join(timeout=2)
    
    def _background_build_loop(self):
        """Background loop for corpus building."""
        while not self._stop_building and self.chain.default_corpus:
            try:
                result = self.chain.default_corpus.build_iteration()
                if self.debug and (result['items_added'] > 0 or result['items_refined'] > 0):
                    print(f"\n[BUILD] +{result['items_added']} items, {result['items_refined']} refined")
            except Exception as e:
                if self.debug:
                    print(f"\n[BUILD ERROR] {e}")
            
            # Wait for next iteration
            for _ in range(self.build_interval):
                if self._stop_building:
                    break
                time.sleep(1)
    
    def query(self, question: str) -> str:
        """Process a query through smart routing (knowledge or tools)."""
        # Detect intent using emergent classifier (fail-fast: no legacy fallback)
        intent_result = self.intent_classifier.classify(question)
        
        if self.debug:
            print(f"\n[DEBUG] Input: {question}")
            print(f"[DEBUG] Intent: {intent_result.intent.name} (conf={intent_result.confidence:.2f})")
            print(f"[DEBUG] Reason: {intent_result.reason}")
        
        # Route based on intent
        if intent_result.intent == Intent.KNOWLEDGE:
            # Knowledge query - use emergent chain
            if self.debug:
                topics = self.chain._extract_topics(question)
                print(f"[DEBUG] Extracted topics: {topics}")
            
            response = self.chain.chat(question)
            
            if self.debug:
                stats = self.chain.get_stats()
                print(f"[DEBUG] Conversation LLM calls: {stats['conversation_calls']} (should be 0)")
            
            return response
        
        elif intent_result.intent == Intent.CODE_GENERATION:
            # Code generation - use Python code gear
            if not self.python_gear:
                raise RuntimeError("CODE_GENERATION intent detected but no code generator available")
            
            if self.debug:
                print(f"[DEBUG] Routing to PythonCodeGear")
            
            result = self.python_gear.generate_from_text(question)
            
            if self.debug:
                print(f"[DEBUG] Pattern used: {result.pattern_used}")
                print(f"[DEBUG] Verified: {result.verified}")
            
            if result.success:
                response = f"```python\n{result.code}\n```"
                if result.pattern_used:
                    response += f"\n\n*Pattern: {result.pattern_used}*"
                if result.verified:
                    response += f"\n✓ Code verified - runs successfully"
                    if result.output:
                        response += f"\nOutput: {result.output.strip()[:200]}"
                return response
            else:
                return f"Failed to generate code: {result.error}"
        
        elif intent_result.intent == Intent.TOOL_CALL:
            # Tool call - use orchestrator
            if not self.orchestrator:
                raise RuntimeError("TOOL_CALL intent detected but orchestrator not enabled")
            
            result = self.orchestrator.execute(question, dry_run=True)
            
            if self.debug:
                print(f"[DEBUG] Plan: {result['plan']}")
                print(f"[DEBUG] Commands: {result['commands']}")
            
            # Store pending commands for confirmation
            self.pending_commands = result['commands']
            
            # Build response asking for confirmation
            if result['commands']:
                cmd_list = '\n'.join([f"  $ {cmd}" for cmd in result['commands']])
                return (
                    f"I'll need to run these commands:\n{cmd_list}\n\n"
                    f"Type 'yes' or 'y' to execute, or anything else to cancel."
                )
            else:
                raise RuntimeError(f"TOOL_CALL intent detected but no commands generated for: {question}")
        
        elif intent_result.intent == Intent.UNSUPPORTED:
            # Fail-fast: don't silently fall back
            raise RuntimeError(f"UNSUPPORTED intent - emergent classifier could not route: {question}")
        
        # CLARIFICATION intent - ask for more info
        return f"I need more information to help you. Could you clarify: {question}"
    
    def execute_pending(self) -> str:
        """Execute pending commands after user confirmation."""
        if not self.pending_commands:
            return "No pending commands to execute."
        
        if not self.orchestrator:
            return "Orchestrator not available."
        
        import subprocess
        results = []
        
        for cmd in self.pending_commands:
            try:
                output = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                if output.returncode == 0:
                    results.append(f"✓ {cmd}")
                    if output.stdout.strip():
                        results.append(f"  {output.stdout.strip()[:100]}")
                else:
                    results.append(f"✗ {cmd}")
                    if output.stderr.strip():
                        results.append(f"  Error: {output.stderr.strip()[:100]}")
            except Exception as e:
                results.append(f"✗ {cmd}")
                results.append(f"  Error: {str(e)}")
        
        self.pending_commands = []
        return "Executed:\n" + '\n'.join(results)
    
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
            default_items = len(self.chain.default_corpus.all_items) if self.chain.default_corpus else 0
            return f"Saved corpus to: {path}\n  Knowledge items: {len(self.chain.corpus)}\n  Default corpus items: {default_items}"
        
        if cmd.startswith('/load '):
            path = cmd[6:].strip()
            self.load_corpus(path)
            default_items = len(self.chain.default_corpus.all_items) if self.chain.default_corpus else 0
            return f"Loaded corpus from: {path}\n  Knowledge items: {len(self.chain.corpus)}\n  Default corpus items: {default_items}"
        
        if cmd == '/build':
            # Run one iteration of corpus building
            if self.chain.default_corpus:
                result = self.chain.default_corpus.build_iteration()
                return f"Build iteration {result['iteration']}:\n  Items added: {result['items_added']}\n  Items refined: {result['items_refined']}\n  Total items: {len(self.chain.default_corpus.all_items)}"
            return "Default corpus not available"
        
        if cmd == '/corpus':
            # Show default corpus stats
            if self.chain.default_corpus:
                stats = self.chain.default_corpus.get_stats()
                result = f"Default Corpus Statistics:\n"
                result += f"  Total items: {stats['total_items']}\n"
                result += f"  Categories: {stats['categories']}\n"
                result += f"  Build iterations: {stats['build_stats']['iterations']}\n"
                result += f"  Items added: {stats['build_stats']['items_added']}\n"
                result += f"  Auto-build: {'ON' if self.auto_build else 'OFF'}\n"
                result += f"\nCategory breakdown:\n"
                for name, cat_stats in sorted(stats['category_stats'].items()):
                    result += f"  {name}: {cat_stats['items']} items\n"
                return result.strip()
            return "Default corpus not available"
        
        if cmd == '/autobuild':
            # Toggle auto-build
            if self.auto_build:
                self.stop_auto_build()
                return "Auto-build: OFF"
            else:
                if self.start_auto_build():
                    return f"Auto-build: ON (every {self.build_interval}s)"
                return "Failed to start auto-build"
        
        if cmd.startswith('/autobuild '):
            # Set auto-build interval
            try:
                interval = int(cmd[11:].strip())
                self.build_interval = max(10, interval)  # Minimum 10 seconds
                return f"Auto-build interval set to {self.build_interval}s"
            except ValueError:
                return "Usage: /autobuild <seconds>"
        
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
        tools_status = "enabled" if self.enable_tools else "disabled"
        return f"""
Emergent Chat - Commands:

  /help, /h       Show this help
  /topics         List known topics
  /stats          Show statistics
  /debug          Toggle debug mode
  /learn <topic>  Learn about a new topic
  /info <topic>   Show info about a topic
  /save <path>    Save corpus to file (includes default corpus)
  /load <path>    Load corpus from file (resumes where you left off)
  /corpus         Show default corpus statistics
  /build          Run one iteration of corpus self-building
  /autobuild      Toggle background auto-building ON/OFF
  /autobuild <N>  Set auto-build interval to N seconds
  /books          List available literary works
  /book <name>    Load a literary work (e.g., /book moby_dick)
  /quit, /q       Exit

Ask questions like:
  Who is Captain Ahab?
  Tell me about Queequeg

Request actions like (tools {tools_status}):
  Create a directory called test
  Make a file and write hello to it
  Set up a new project folder

Note: Knowledge responses are EMERGENT - no LLM during conversation.
Tool calls use the GearOrchestrator to plan and execute commands.
"""
    
    def run(self):
        """Run the interactive chat loop."""
        print()
        print("═" * 60)
        print("  Emergent Conversational Chat")
        print("  Gear-based routing: knowledge queries + tool calls")
        print("═" * 60)
        print()
        
        stats = self.chain.get_stats()
        tools_status = "✓ enabled" if self.enable_tools else "✗ disabled"
        print(f"  Knowledge: {stats['topics']} topics, {stats['corpus_items']} facts")
        print(f"  Tools: {tools_status}")
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
                        # Stop auto-build before exiting
                        if self.auto_build:
                            self.stop_auto_build()
                        print("\nGoodbye!")
                        break
                    print(f"\n{response}\n")
                    continue
                
                # Handle confirmation for pending commands
                if self.pending_commands and user_input.lower() in ('yes', 'y'):
                    response = self.execute_pending()
                    print(f"\nBot: {response}\n")
                    continue
                elif self.pending_commands:
                    # User didn't confirm - cancel
                    self.pending_commands = []
                    print("\nBot: Cancelled.\n")
                    continue
                
                # Process query (smart routing: knowledge or tools)
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
