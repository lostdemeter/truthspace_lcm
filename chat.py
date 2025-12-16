#!/usr/bin/env python3
"""
TruthSpace LCM Chat Interface

An interactive chat program for interacting with TruthSpace LCM.
Supports natural language queries, command execution, and knowledge management.

Usage:
    python chat.py
"""

import sys
import readline  # For better input handling (history, editing)
from typing import Optional

from truthspace_lcm import (
    TruthSpace,
    Resolver,
    KnowledgeGapError,
    EntryType,
    KnowledgeDomain,
)
from truthspace_lcm.core.executor import CodeExecutor


class Chat:
    """Interactive chat interface for TruthSpace LCM."""
    
    def __init__(self):
        self.ts = TruthSpace()
        self.resolver = Resolver(self.ts, auto_learn=False)
        self.executor = CodeExecutor()
        self.auto_execute = False
        self.verbose = False
        
    def print_banner(self):
        """Print welcome banner."""
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║           TruthSpace LCM - Interactive Chat                  ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  Commands:                                                   ║")
        print("║    /help     - Show this help                                ║")
        print("║    /auto     - Toggle auto-execute mode                      ║")
        print("║    /verbose  - Toggle verbose mode                           ║")
        print("║    /stats    - Show knowledge statistics                     ║")
        print("║    /search   - Search knowledge base                         ║")
        print("║    /learn    - Teach new knowledge                           ║")
        print("║    /quit     - Exit chat                                     ║")
        print("║                                                              ║")
        print("║  Or just type a natural language request!                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        
    def print_help(self):
        """Print help message."""
        print("""
Commands:
  /help              Show this help message
  /auto              Toggle auto-execute mode (currently: {})
  /verbose           Toggle verbose output (currently: {})
  /stats             Show knowledge base statistics
  /search <query>    Search knowledge base for entries
  /learn             Interactive knowledge teaching
  /history           Show recent queries
  /quit, /exit, /q   Exit the chat

Natural Language:
  Just type what you want to do in plain English!
  
Examples:
  > list files in the current directory
  > compress the logs folder
  > find all python files
  > write a hello world program
""".format("ON" if self.auto_execute else "OFF",
           "ON" if self.verbose else "OFF"))
    
    def show_stats(self):
        """Show knowledge base statistics."""
        # Use TruthSpace's count method
        counts = self.ts.count()
        total = sum(counts.values())
        
        print("\n📊 Knowledge Base Statistics")
        print("-" * 40)
        print(f"  Total entries: {total}")
        print()
        print("  By type:")
        for entry_type, count in sorted(counts.items()):
            if count > 0:
                print(f"    {entry_type:12} : {count}")
        print()
    
    def search_knowledge(self, query: str):
        """Search the knowledge base."""
        if not query:
            query = input("Search query: ").strip()
        
        if not query:
            print("No query provided.")
            return
        
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 40)
        
        try:
            results = self.ts.query(query, threshold=0.2)
            
            if not results:
                print("No matches found.")
                return
            
            for i, result in enumerate(results[:10], 1):
                entry = result.entry
                print(f"\n{i}. {entry.name} ({entry.entry_type.value})")
                print(f"   Similarity: {result.similarity:.2f}")
                print(f"   {entry.description}")
                if entry.metadata.get("target_commands"):
                    print(f"   → {entry.metadata['target_commands'][0]}")
        
        except Exception as e:
            print(f"Search error: {e}")
        
        print()
    
    def learn_knowledge(self):
        """Interactive knowledge teaching."""
        print("\n📚 Teach New Knowledge")
        print("-" * 40)
        
        name = input("Name (e.g., 'backup_home'): ").strip()
        if not name:
            print("Cancelled.")
            return
        
        description = input("Description: ").strip()
        keywords = input("Keywords (comma-separated): ").strip().split(",")
        keywords = [k.strip() for k in keywords if k.strip()]
        
        print("\nEntry type:")
        print("  1. INTENT (NL → command)")
        print("  2. COMMAND (command reference)")
        print("  3. PATTERN (code template)")
        type_choice = input("Choice [1]: ").strip() or "1"
        
        type_map = {"1": EntryType.INTENT, "2": EntryType.COMMAND, "3": EntryType.PATTERN}
        entry_type = type_map.get(type_choice, EntryType.INTENT)
        
        output_type = input("Output type (bash/python) [bash]: ").strip() or "bash"
        command = input("Command/code to execute: ").strip()
        
        if not command:
            print("Cancelled - no command provided.")
            return
        
        try:
            self.ts.store(
                name=name,
                entry_type=entry_type,
                domain=KnowledgeDomain.PROGRAMMING,
                description=description,
                keywords=keywords,
                metadata={
                    "target_commands": [command],
                    "output_type": output_type,
                }
            )
            print(f"\n✅ Learned: {name}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()
    
    def process_request(self, request: str):
        """Process a natural language request."""
        try:
            result = self.resolver.resolve(request)
            
            # Show result
            print()
            if self.verbose:
                print(f"📝 Matched: {result.knowledge.name}")
                print(f"   Type: {result.output_type.value}")
                print(f"   Similarity: {result.similarity:.2f}")
                print()
            
            print(f"💻 {result.output_type.value.upper()}:")
            print(f"   {result.output}")
            print()
            
            # Execute?
            if self.auto_execute:
                self._execute(result.output, result.output_type.value)
            else:
                response = input("Execute? (y/N): ").strip().lower()
                if response == 'y':
                    self._execute(result.output, result.output_type.value)
        
        except KnowledgeGapError as e:
            print(f"\n❓ I don't know how to do that yet.")
            print(f"   Best match similarity: {e.best_match:.2f}")
            print()
            response = input("Would you like to teach me? (y/N): ").strip().lower()
            if response == 'y':
                self._quick_learn(request)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        print()
    
    def _execute(self, code: str, output_type: str):
        """Execute code and show results."""
        print("\n📤 Output:")
        print("-" * 40)
        
        try:
            if output_type == "python":
                result = self.executor.execute_python(code)
            else:
                result = self.executor.execute_bash(code)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[stderr] {result.stderr}")
            
            if result.status.value != "success":
                print(f"[Exit code: {result.return_code}]")
        
        except Exception as e:
            print(f"Execution error: {e}")
        
        print("-" * 40)
    
    def _quick_learn(self, request: str):
        """Quick learning from a failed request."""
        print("\n📚 Quick Learn")
        print(f"   Request: {request}")
        
        name = input("Name for this knowledge: ").strip()
        if not name:
            name = request.replace(" ", "_")[:30]
        
        output_type = input("Output type (bash/python) [bash]: ").strip() or "bash"
        command = input("Command to execute: ").strip()
        
        if not command:
            print("Cancelled.")
            return
        
        # Extract keywords from request
        keywords = [w.lower() for w in request.split() if len(w) > 2]
        
        try:
            self.ts.store(
                name=name,
                entry_type=EntryType.INTENT,
                domain=KnowledgeDomain.PROGRAMMING,
                description=request,
                keywords=keywords,
                metadata={
                    "target_commands": [command],
                    "output_type": output_type,
                    "triggers": [request],
                }
            )
            print(f"\n✅ Learned: {name}")
            print("   Try your request again!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def run(self):
        """Main chat loop."""
        self.print_banner()
        
        while True:
            try:
                user_input = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                
                if cmd in ("quit", "exit", "q"):
                    print("\nGoodbye! 👋")
                    break
                elif cmd == "help":
                    self.print_help()
                elif cmd == "auto":
                    self.auto_execute = not self.auto_execute
                    print(f"Auto-execute: {'ON' if self.auto_execute else 'OFF'}")
                elif cmd == "verbose":
                    self.verbose = not self.verbose
                    print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
                elif cmd == "stats":
                    self.show_stats()
                elif cmd == "search":
                    self.search_knowledge(arg)
                elif cmd == "learn":
                    self.learn_knowledge()
                else:
                    print(f"Unknown command: /{cmd}")
                    print("Type /help for available commands.")
            else:
                # Natural language request
                self.process_request(user_input)


def main():
    chat = Chat()
    chat.run()


if __name__ == "__main__":
    main()
