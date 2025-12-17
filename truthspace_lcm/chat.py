#!/usr/bin/env python3
"""
TruthSpace LCM Chat Interface

A geometric chat system that:
1. Detects user intent (chat, bash command, question)
2. Resolves queries against knowledge base
3. Executes bash commands when appropriate
4. Provides conversational responses

Usage:
    python -m truthspace_lcm.chat
    
Or:
    from truthspace_lcm.chat import LCMChat
    chat = LCMChat()
    chat.run()
"""

import subprocess
import sys
import re
from typing import Tuple, Optional, List, Dict
from truthspace_lcm.core import StackedLCM


# =============================================================================
# INTENT TYPES
# =============================================================================

class Intent:
    BASH = "bash"           # User wants to execute a command
    QUESTION = "question"   # User is asking a question
    CHAT = "chat"           # General conversation
    HELP = "help"           # User needs help
    EXIT = "exit"           # User wants to quit


# =============================================================================
# KNOWLEDGE BASE - Bootstrap with essential knowledge
# =============================================================================

BOOTSTRAP_KNOWLEDGE = [
    # Bash commands
    {"content": "ls -la", "description": "list show files directory all hidden terminal bash command"},
    {"content": "ls", "description": "list files directory terminal bash command"},
    {"content": "pwd", "description": "print working directory current path terminal bash command"},
    {"content": "cd", "description": "change directory navigate folder terminal bash command"},
    {"content": "mkdir", "description": "make create directory folder new terminal bash command"},
    {"content": "rm", "description": "remove delete file terminal bash command"},
    {"content": "rm -rf", "description": "remove delete directory folder force recursive terminal bash command"},
    {"content": "cp", "description": "copy file duplicate terminal bash command"},
    {"content": "mv", "description": "move rename file directory terminal bash command"},
    {"content": "cat", "description": "show display read file contents terminal bash command"},
    {"content": "grep", "description": "search find pattern text file terminal bash command"},
    {"content": "find", "description": "search locate file directory name terminal bash command"},
    {"content": "ps aux", "description": "list show process running system terminal bash command"},
    {"content": "top", "description": "show system process cpu memory monitor terminal bash command"},
    {"content": "df -h", "description": "disk space usage filesystem terminal bash command"},
    {"content": "du -sh", "description": "directory size disk usage terminal bash command"},
    {"content": "chmod", "description": "change permissions file directory terminal bash command"},
    {"content": "chown", "description": "change owner file directory terminal bash command"},
    {"content": "echo", "description": "print display text output terminal bash command"},
    {"content": "head", "description": "show first lines file beginning terminal bash command"},
    {"content": "tail", "description": "show last lines file end terminal bash command"},
    {"content": "wc -l", "description": "count lines words file terminal bash command"},
    {"content": "sort", "description": "sort lines file alphabetical terminal bash command"},
    {"content": "uniq", "description": "unique lines remove duplicates file terminal bash command"},
    {"content": "tar -xzf", "description": "extract archive tar gzip file terminal bash command"},
    {"content": "tar -czf", "description": "create archive tar gzip compress terminal bash command"},
    {"content": "wget", "description": "download file url web terminal bash command"},
    {"content": "curl", "description": "transfer data url http request terminal bash command"},
    {"content": "ssh", "description": "secure shell remote connect server terminal bash command"},
    {"content": "scp", "description": "secure copy file remote server terminal bash command"},
    {"content": "git status", "description": "git show status changes repository terminal bash command"},
    {"content": "git log", "description": "git show history commits repository terminal bash command"},
    {"content": "git diff", "description": "git show changes differences repository terminal bash command"},
    {"content": "git pull", "description": "git fetch merge remote changes repository terminal bash command"},
    {"content": "git push", "description": "git upload commits remote repository terminal bash command"},
    
    # Chat/Social responses
    {"content": "Hello! How can I help you today?", "description": "hello hi greeting welcome how are you"},
    {"content": "You're welcome! Let me know if you need anything else.", "description": "you're welcome no problem glad to help thanks thank you"},
    {"content": "Goodbye! Have a great day!", "description": "goodbye bye see you later exit quit"},
    {"content": "I'm here to help with bash commands and answer questions.", "description": "help what can you do capabilities features"},
    {"content": "I understand. How can I assist you further?", "description": "understand okay got it i see"},
    {"content": "That's a great question! Let me help you with that.", "description": "question ask how what why when where"},
    
    # Help/Meta
    {"content": "Type 'exit' or 'quit' to leave the chat.", "description": "exit quit leave stop end session"},
    {"content": "I can help you with bash commands. Just describe what you want to do!", "description": "help bash command terminal what can you do"},
]


# =============================================================================
# LCM CHAT
# =============================================================================

class LCMChat:
    """
    Interactive chat interface powered by StackedLCM.
    
    Features:
    - Intent detection (bash vs chat vs question)
    - Bash command execution with confirmation
    - Knowledge-based response generation
    - Context tracking
    """
    
    def __init__(self, safe_mode: bool = True):
        """
        Initialize the chat.
        
        Args:
            safe_mode: If True, ask for confirmation before executing commands
        """
        self.lcm = StackedLCM()
        self.safe_mode = safe_mode
        self.context: List[str] = []  # Recent queries for context
        self.max_context = 5
        
        # Bootstrap with essential knowledge
        self._bootstrap()
    
    def _bootstrap(self):
        """Load bootstrap knowledge into the LCM."""
        print("Initializing knowledge base...")
        self.lcm.ingest_batch(BOOTSTRAP_KNOWLEDGE)
        print(f"Loaded {len(BOOTSTRAP_KNOWLEDGE)} knowledge entries.")
    
    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect the user's intent from their query.
        
        Returns:
            (intent_type, confidence)
        """
        query_lower = query.lower().strip()
        
        # Direct exit commands
        if query_lower in ['exit', 'quit', 'bye', 'goodbye', 'q']:
            return Intent.EXIT, 1.0
        
        # Direct help commands
        if query_lower in ['help', '?', 'help me']:
            return Intent.HELP, 1.0
        
        # Check for bash-like patterns
        bash_patterns = [
            r'^ls\b', r'^cd\b', r'^pwd\b', r'^mkdir\b', r'^rm\b', r'^cp\b', r'^mv\b',
            r'^cat\b', r'^grep\b', r'^find\b', r'^ps\b', r'^top\b', r'^df\b', r'^du\b',
            r'^chmod\b', r'^chown\b', r'^echo\b', r'^head\b', r'^tail\b', r'^wc\b',
            r'^sort\b', r'^uniq\b', r'^tar\b', r'^wget\b', r'^curl\b', r'^ssh\b',
            r'^scp\b', r'^git\b', r'^python\b', r'^pip\b', r'^npm\b', r'^docker\b',
            r'^sudo\b', r'^apt\b', r'^yum\b', r'^brew\b',
        ]
        
        for pattern in bash_patterns:
            if re.match(pattern, query_lower):
                return Intent.BASH, 0.95
        
        # Use LCM to detect intent based on similarity to known patterns
        query_emb = self.lcm.encode(query, update_stats=False)
        
        # Check similarity to bash knowledge
        bash_sims = []
        chat_sims = []
        
        for pid, (content, emb) in self.lcm.points.items():
            sim = self.lcm.cosine_similarity(query_emb, emb)
            
            # Classify based on content
            if any(kw in content.lower() for kw in ['terminal', 'bash', 'command', 'ls', 'cd', 'rm', 'git']):
                bash_sims.append(sim)
            else:
                chat_sims.append(sim)
        
        avg_bash = sum(bash_sims) / len(bash_sims) if bash_sims else 0
        avg_chat = sum(chat_sims) / len(chat_sims) if chat_sims else 0
        
        # Detect bash intent from natural language
        bash_keywords = ['list', 'show', 'files', 'directory', 'folder', 'delete', 'remove',
                        'create', 'make', 'copy', 'move', 'rename', 'search', 'find',
                        'process', 'processes', 'running', 'disk', 'space', 'permission', 
                        'download', 'git', 'status', 'size']
        
        query_words = set(query_lower.split())
        bash_keyword_count = len(query_words & set(bash_keywords))
        
        # Question patterns
        question_words = ['how', 'what', 'where', 'when', 'why', 'which', 'can', 'could', 'would']
        is_question = any(query_lower.startswith(w) for w in question_words) or query.endswith('?')
        
        # Decision logic
        if bash_keyword_count >= 2 or (bash_keyword_count >= 1 and avg_bash > avg_chat):
            return Intent.BASH, min(0.7 + bash_keyword_count * 0.1, 0.95)
        elif is_question:
            return Intent.QUESTION, 0.8
        else:
            return Intent.CHAT, 0.7
    
    def resolve_bash_command(self, query: str) -> Tuple[str, float]:
        """
        Resolve a natural language query to a bash command.
        
        Returns:
            (command, confidence)
        """
        query_lower = query.lower().strip()
        
        # Direct command - return as-is
        direct_commands = ['ls', 'cd', 'pwd', 'mkdir', 'rm', 'cp', 'mv', 'cat', 
                          'grep', 'find', 'ps', 'git', 'echo', 'head', 'tail',
                          'df', 'du', 'chmod', 'chown', 'tar', 'wget', 'curl',
                          'ssh', 'scp', 'top', 'sort', 'uniq', 'wc', 'sudo']
        
        if any(query_lower.startswith(cmd) for cmd in direct_commands):
            return query, 0.95
        
        # Natural language to command mapping
        nl_to_cmd = {
            # File listing
            ('list', 'files'): 'ls -la',
            ('show', 'files'): 'ls -la',
            ('list', 'directory'): 'ls -la',
            ('show', 'directory'): 'ls -la',
            ('what', 'files'): 'ls -la',
            ('list', 'all'): 'ls -la',
            
            # Current directory
            ('current', 'directory'): 'pwd',
            ('where', 'am'): 'pwd',
            ('print', 'directory'): 'pwd',
            
            # Process listing
            ('running', 'processes'): 'ps aux',
            ('show', 'processes'): 'ps aux',
            ('list', 'processes'): 'ps aux',
            ('what', 'running'): 'ps aux',
            ('processes', 'running'): 'ps aux',
            ('what', 'processes'): 'ps aux',
            
            # Disk space
            ('disk', 'space'): 'df -h',
            ('disk', 'usage'): 'df -h',
            ('free', 'space'): 'df -h',
            ('storage', 'space'): 'df -h',
            ('show', 'space'): 'df -h',
            
            # Directory size
            ('directory', 'size'): 'du -sh .',
            ('folder', 'size'): 'du -sh .',
            
            # Git
            ('git', 'status'): 'git status',
            ('git', 'changes'): 'git status',
            ('git', 'history'): 'git log --oneline -10',
            ('git', 'log'): 'git log --oneline -10',
            ('git', 'diff'): 'git diff',
        }
        
        # Check for keyword matches
        query_words = set(query_lower.split())
        for keywords, command in nl_to_cmd.items():
            if all(kw in query_words for kw in keywords):
                return command, 0.9
        
        # Partial matches (single keyword triggers)
        partial_matches = {
            'files': 'ls -la',
            'directory': 'ls -la',
            'processes': 'ps aux',
            'running': 'ps aux',
            'space': 'df -h',
        }
        
        for keyword, command in partial_matches.items():
            if keyword in query_words:
                return command, 0.7
        
        # Fall back to LCM resolution
        content, sim, cluster = self.lcm.resolve(query)
        
        # Check if the resolved content looks like a command
        if content and any(content.startswith(cmd) for cmd in direct_commands):
            return content, sim
        
        return None, 0.0
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """
        Execute a bash command and return the result.
        
        Returns:
            (success, output)
        """
        # Safety check - block dangerous commands
        dangerous_patterns = [
            r'rm\s+-rf\s+/', r'rm\s+-rf\s+~', r'rm\s+-rf\s+\*',
            r'dd\s+', r'mkfs', r':\(\)\{', r'>\s*/dev/sd',
            r'chmod\s+-R\s+777\s+/', r'chown\s+-R.*/',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return False, f"⚠️  Blocked potentially dangerous command: {command}"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=None  # Use current directory
            )
            
            output = result.stdout if result.stdout else result.stderr
            success = result.returncode == 0
            
            if not output:
                output = "(no output)" if success else f"Error (code {result.returncode})"
            
            return success, output.strip()
            
        except subprocess.TimeoutExpired:
            return False, "⚠️  Command timed out (30s limit)"
        except Exception as e:
            return False, f"⚠️  Error: {str(e)}"
    
    def generate_response(self, query: str, intent: str) -> str:
        """Generate a response based on query and intent."""
        content, sim, cluster = self.lcm.resolve(query)
        
        if sim > 0.7:
            return content
        elif sim > 0.5:
            return f"I think you might be looking for: {content}"
        else:
            return "I'm not sure I understand. Could you rephrase that?"
    
    def process(self, query: str) -> str:
        """
        Process a user query and return a response.
        
        This is the main entry point for processing queries.
        """
        query = query.strip()
        if not query:
            return ""
        
        # Update context
        self.context.append(query)
        if len(self.context) > self.max_context:
            self.context.pop(0)
        
        # Detect intent
        intent, confidence = self.detect_intent(query)
        
        # Handle different intents
        if intent == Intent.EXIT:
            return "EXIT"
        
        elif intent == Intent.HELP:
            return """
🔧 TruthSpace LCM Chat

I can help you with:
• Bash commands - describe what you want to do, or type commands directly
• Questions - ask me anything about files, directories, processes
• General chat - just say hello!

Examples:
• "list all files in this directory"
• "show me running processes"
• "how do I find a file?"
• "ls -la"
• "git status"

Type 'exit' or 'quit' to leave.
"""
        
        elif intent == Intent.BASH:
            # Try to resolve to a bash command
            command, cmd_confidence = self.resolve_bash_command(query)
            
            if command and cmd_confidence > 0.5:
                if self.safe_mode:
                    return f"CONFIRM_BASH:{command}"
                else:
                    success, output = self.execute_command(command)
                    prefix = "✓" if success else "✗"
                    return f"{prefix} $ {command}\n{output}"
            else:
                return f"I understood you want to run a command, but I'm not sure which one.\nCould you be more specific or type the command directly?"
        
        elif intent == Intent.QUESTION:
            response = self.generate_response(query, intent)
            return response
        
        else:  # CHAT
            response = self.generate_response(query, intent)
            return response
    
    def run(self):
        """Run the interactive chat loop."""
        print("\n" + "=" * 60)
        print("  TruthSpace LCM Chat")
        print("  Geometric knowledge resolution with bash execution")
        print("=" * 60)
        print("\nType 'help' for commands, 'exit' to quit.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                response = self.process(query)
                
                if response == "EXIT":
                    print("\nLCM: Goodbye! 👋\n")
                    break
                
                elif response.startswith("CONFIRM_BASH:"):
                    command = response[13:]
                    print(f"\nLCM: I'll run: $ {command}")
                    confirm = input("     Execute? [y/N]: ").strip().lower()
                    
                    if confirm in ['y', 'yes']:
                        success, output = self.execute_command(command)
                        prefix = "✓" if success else "✗"
                        print(f"\n{prefix} $ {command}")
                        print(output)
                    else:
                        print("     (cancelled)")
                    print()
                
                else:
                    print(f"\nLCM: {response}\n")
                    
            except KeyboardInterrupt:
                print("\n\nLCM: Goodbye! 👋\n")
                break
            except EOFError:
                print("\n\nLCM: Goodbye! 👋\n")
                break


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    chat = LCMChat(safe_mode=True)
    chat.run()
