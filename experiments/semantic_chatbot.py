#!/usr/bin/env python3
"""
Semantic Chatbot

A chatbot powered by SemanticSpace for knowledge retrieval.

Features:
- Learns from conversations
- Answers questions from stored knowledge
- Handles greetings and social interactions
- Can be trained on any domain
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from truthspace_lcm.core.semantic_space import SemanticSpace
from typing import Optional, Tuple
import re


class SemanticChatbot:
    """
    A chatbot that uses SemanticSpace for knowledge retrieval.
    """
    
    def __init__(self, name: str = "Bot"):
        self.name = name
        
        # Knowledge base
        self.knowledge = SemanticSpace(seeds={
            # Question types
            'WHEN': ['when', 'year', 'date', 'time'],
            'WHERE': ['where', 'place', 'location', 'city', 'country'],
            'WHO': ['who', 'person', 'name'],
            'WHAT': ['what', 'thing', 'define', 'meaning'],
            'HOW': ['how', 'way', 'method', 'steps'],
            'WHY': ['why', 'reason', 'because', 'cause'],
            
            # Life events
            'BIRTH': ['born', 'birth', 'birthday'],
            'DEATH': ['died', 'death', 'killed', 'assassinated'],
            
            # Actions
            'DISCOVER': ['discovered', 'invented', 'developed', 'created', 'theory'],
            'LEAD': ['president', 'leader', 'king', 'queen', 'ruled'],
            
            # Geography
            'CAPITAL': ['capital', 'city'],
            'COUNTRY': ['country', 'nation'],
            
            # Cooking
            'COOK': ['cook', 'recipe', 'make', 'prepare'],
            'BOIL': ['boil', 'pasta', 'water'],
            'BAKE': ['bake', 'bread', 'oven'],
            'FRY': ['fry', 'chicken', 'oil'],
            'GRILL': ['grill', 'steak'],
            
            # Tech
            'COMMAND': ['command', 'terminal', 'linux', 'bash'],
            'LIST': ['ls', 'list', 'files', 'directory'],
            'SEARCH': ['grep', 'search', 'find'],
            'DISK': ['df', 'disk', 'space', 'storage'],
            'PROCESS': ['ps', 'process', 'running'],
        })
        
        # Social patterns
        self.greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        self.farewells = ['bye', 'goodbye', 'see you', 'later', 'farewell']
        self.thanks = ['thank', 'thanks', 'appreciate']
        self.help_words = ['help', 'assist', 'support']
        
        # Confidence threshold - 0.5 balances known vs unknown
        self.confidence_threshold = 0.50
    
    def _is_social(self, text: str) -> Tuple[bool, str]:
        """Check if message is social (greeting, farewell, etc.)"""
        text_lower = text.lower().strip()
        words = set(text_lower.split())
        
        # Only match greetings if they're the main content (short message)
        if len(words) <= 3:
            for g in self.greetings:
                if g in text_lower:
                    return True, f"Hello! I'm {self.name}. How can I help you?"
        
        for f in self.farewells:
            if f in text_lower:
                return True, "Goodbye! Feel free to come back anytime."
        
        for t in self.thanks:
            if t in text_lower:
                return True, "You're welcome! Is there anything else I can help with?"
        
        for h in self.help_words:
            if h in text_lower:
                return True, f"I can answer questions about things I've learned. Try asking me something!"
        
        return False, ""
    
    def _is_question(self, text: str) -> bool:
        """Check if message is a question."""
        text_lower = text.lower().strip()
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'is', 'are', 'can', 'do', 'does', 'did']
        
        if text.strip().endswith('?'):
            return True
        
        for qw in question_words:
            if text_lower.startswith(qw + ' '):
                return True
        
        return False
    
    def learn(self, fact: str, response: str = None):
        """
        Teach the bot a fact.
        
        Args:
            fact: The fact to learn (e.g., "George Washington was born in 1732")
            response: Optional custom response (defaults to the fact itself)
        """
        self.knowledge[fact] = response or fact
    
    def learn_qa(self, question: str, answer: str):
        """
        Teach the bot a question-answer pair.
        
        Args:
            question: The question
            answer: The answer
        """
        self.knowledge[question] = answer
    
    def learn_many(self, facts: list):
        """Learn multiple facts at once."""
        for fact in facts:
            if isinstance(fact, tuple):
                self.learn(fact[0], fact[1] if len(fact) > 1 else None)
            else:
                self.learn(fact)
    
    def respond(self, message: str) -> str:
        """
        Generate a response to a message.
        
        Args:
            message: User's message
        
        Returns:
            Bot's response
        """
        # Check for social interaction
        is_social, social_response = self._is_social(message)
        if is_social:
            return social_response
        
        # Empty knowledge base
        if len(self.knowledge) == 0:
            return "I don't know anything yet. Teach me something!"
        
        # Try to find a match in knowledge
        response, score = self.knowledge.get_with_score(message)
        matched_key = self.knowledge.get_key(message)
        
        # Check if the match shares significant words with the query
        query_words = set(self._tokenize_simple(message))
        key_words = set(self._tokenize_simple(matched_key)) if matched_key else set()
        overlap = query_words & key_words
        
        # Need BOTH decent score AND word overlap (or very high score)
        if (score >= 0.7) or (score >= 0.3 and len(overlap) >= 1):
            return response
        else:
            # Low confidence and no word overlap - be honest
            if self._is_question(message):
                return "I'm not sure about that. Could you teach me?"
            else:
                return "I don't quite understand. Can you rephrase or teach me about that?"
    
    def _tokenize_simple(self, text: str) -> list:
        """Simple tokenization for overlap checking."""
        if not text:
            return []
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'by',
                     'what', 'when', 'where', 'who', 'how', 'why', 'which',
                     'do', 'does', 'did', 'i', 'you', 'it', 'that', 'this'}
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def train(self, examples: list, verbose: bool = False):
        """
        Train the bot on question-answer examples.
        
        Args:
            examples: List of (question, expected_response) tuples
            verbose: Print training progress
        """
        return self.knowledge.train(examples, verbose=verbose)
    
    def chat(self):
        """Interactive chat loop."""
        print(f"\n{self.name}: Hello! I'm {self.name}. Ask me anything or teach me something!")
        print(f"{self.name}: (Type 'quit' to exit, 'teach: <fact>' to teach me)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{self.name}: Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print(f"{self.name}: Goodbye!")
                break
            
            # Teaching mode
            if user_input.lower().startswith('teach:'):
                fact = user_input[6:].strip()
                if fact:
                    self.learn(fact)
                    print(f"{self.name}: Got it! I learned: \"{fact}\"")
                else:
                    print(f"{self.name}: What would you like to teach me?")
                continue
            
            # Normal response
            response = self.respond(user_input)
            print(f"{self.name}: {response}")


def main():
    print("=" * 70)
    print("SEMANTIC CHATBOT")
    print("A chatbot powered by SemanticSpace")
    print("=" * 70)
    
    # Create bot
    bot = SemanticChatbot(name="Sage")
    
    # Teach it some knowledge
    print("\nTeaching the bot...")
    
    bot.learn_many([
        # History
        ("George Washington was born in 1732 in Virginia", "George Washington was born on February 22, 1732, in Virginia."),
        ("George Washington was the first president", "George Washington served as the first President of the United States from 1789 to 1797."),
        ("George Washington died in 1799", "George Washington died on December 14, 1799, at Mount Vernon."),
        ("Abraham Lincoln was born in 1809", "Abraham Lincoln was born on February 12, 1809, in Kentucky."),
        ("Abraham Lincoln was the 16th president", "Abraham Lincoln was the 16th President, serving from 1861 to 1865."),
        ("Abraham Lincoln was assassinated in 1865", "Abraham Lincoln was assassinated by John Wilkes Booth on April 14, 1865."),
        
        # Science
        ("Albert Einstein developed relativity", "Albert Einstein developed the theory of relativity, revolutionizing physics."),
        ("Isaac Newton discovered gravity", "Isaac Newton discovered the laws of gravity and motion."),
        ("Charles Darwin developed evolution", "Charles Darwin developed the theory of evolution by natural selection."),
        
        # Geography
        ("Paris is the capital of France", "Paris is the capital of France, located on the Seine River."),
        ("London is the capital of England", "London is the capital of England and the United Kingdom."),
        ("Tokyo is the capital of Japan", "Tokyo is the capital of Japan and the world's most populous city."),
        ("Berlin is the capital of Germany", "Berlin is the capital of Germany."),
        ("Moscow is the capital of Russia", "Moscow is the capital of Russia."),
        ("Beijing is the capital of China", "Beijing is the capital of China."),
        
        # Cooking
        ("How to boil pasta", "To cook pasta: Boil salted water, add pasta, cook 8-12 minutes until al dente, then drain."),
        ("How to bake bread", "To bake bread: Mix flour, water, yeast, salt. Knead, let rise, then bake at 450°F for 30-40 minutes."),
        ("How to fry chicken", "To fry chicken: Coat in seasoned flour, fry in 350°F oil for 12-15 minutes until golden."),
        ("How to grill steak", "To grill steak: Season, preheat grill to high, cook 4-5 minutes per side for medium-rare."),
        
        # Linux
        ("The ls command lists files", "The 'ls' command lists files and directories. Use 'ls -la' for detailed info including hidden files."),
        ("The grep command searches text", "The 'grep' command searches for text patterns in files. Use 'grep -r' for recursive search."),
        ("The df command shows disk space", "The 'df' command shows disk space usage. Use 'df -h' for human-readable output."),
        ("The ps command shows processes", "The 'ps' command shows running processes. Use 'ps aux' for all processes with details."),
    ])
    
    print(f"Bot knows {len(bot.knowledge)} facts")
    
    # Train on some Q&A pairs - comprehensive coverage
    print("\nTraining on Q&A pairs...")
    qa_pairs = [
        # History
        ("When was Washington born", "George Washington was born on February 22, 1732, in Virginia."),
        ("When was George Washington born", "George Washington was born on February 22, 1732, in Virginia."),
        ("Who was the first president", "George Washington served as the first President of the United States from 1789 to 1797."),
        ("When did Washington die", "George Washington died on December 14, 1799, at Mount Vernon."),
        ("When was Lincoln born", "Abraham Lincoln was born on February 12, 1809, in Kentucky."),
        ("How did Lincoln die", "Abraham Lincoln was assassinated by John Wilkes Booth on April 14, 1865."),
        ("Lincoln assassination", "Abraham Lincoln was assassinated by John Wilkes Booth on April 14, 1865."),
        
        # Science
        ("What did Einstein discover", "Albert Einstein developed the theory of relativity, revolutionizing physics."),
        ("Einstein relativity", "Albert Einstein developed the theory of relativity, revolutionizing physics."),
        ("What did Newton discover", "Isaac Newton discovered the laws of gravity and motion."),
        ("Newton gravity", "Isaac Newton discovered the laws of gravity and motion."),
        ("Darwin evolution", "Charles Darwin developed the theory of evolution by natural selection."),
        
        # Geography - all capitals
        ("Capital of France", "Paris is the capital of France, located on the Seine River."),
        ("What is the capital of France", "Paris is the capital of France, located on the Seine River."),
        ("Capital of England", "London is the capital of England and the United Kingdom."),
        ("Capital of Japan", "Tokyo is the capital of Japan and the world's most populous city."),
        ("What about Japan", "Tokyo is the capital of Japan and the world's most populous city."),
        ("Capital of Germany", "Berlin is the capital of Germany."),
        ("Capital of Russia", "Moscow is the capital of Russia."),
        ("Capital of China", "Beijing is the capital of China."),
        
        # Cooking
        ("How to cook pasta", "To cook pasta: Boil salted water, add pasta, cook 8-12 minutes until al dente, then drain."),
        ("How do I cook pasta", "To cook pasta: Boil salted water, add pasta, cook 8-12 minutes until al dente, then drain."),
        ("How to bake bread", "To bake bread: Mix flour, water, yeast, salt. Knead, let rise, then bake at 450°F for 30-40 minutes."),
        ("How to fry chicken", "To fry chicken: Coat in seasoned flour, fry in 350°F oil for 12-15 minutes until golden."),
        ("How to grill steak", "To grill steak: Season, preheat grill to high, cook 4-5 minutes per side for medium-rare."),
        
        # Linux
        ("How to list files", "The 'ls' command lists files and directories. Use 'ls -la' for detailed info including hidden files."),
        ("How do I list files in Linux", "The 'ls' command lists files and directories. Use 'ls -la' for detailed info including hidden files."),
        ("ls command", "The 'ls' command lists files and directories. Use 'ls -la' for detailed info including hidden files."),
        ("How to search text", "The 'grep' command searches for text patterns in files. Use 'grep -r' for recursive search."),
        ("grep command", "The 'grep' command searches for text patterns in files. Use 'grep -r' for recursive search."),
        ("How to check disk space", "The 'df' command shows disk space usage. Use 'df -h' for human-readable output."),
        ("How to see running processes", "The 'ps' command shows running processes. Use 'ps aux' for all processes with details."),
    ]
    
    stats = bot.train(qa_pairs, verbose=True)
    print(f"Training complete! Final accuracy: {stats['final_accuracy']:.0%}")
    
    # Demo conversation
    print("\n" + "=" * 70)
    print("DEMO CONVERSATION")
    print("=" * 70)
    
    demo_messages = [
        "Hello!",
        "When was George Washington born?",
        "Who was the first president?",
        "How did Lincoln die?",
        "What is the capital of France?",
        "What about Japan?",
        "How do I cook pasta?",
        "How do I list files in Linux?",
        "What did Einstein discover?",
        "What is quantum mechanics?",  # Unknown
        "Thanks!",
        "Goodbye!",
    ]
    
    for msg in demo_messages:
        print(f"\nYou: {msg}")
        response = bot.respond(msg)
        print(f"Sage: {response}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    bot.chat()


if __name__ == "__main__":
    main()
