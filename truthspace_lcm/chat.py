#!/usr/bin/env python3
"""
TruthSpace Geometric Chat System

An interactive chat demonstrating the Dynamic Geometric LCM.

Features:
- Natural language Q&A with learned knowledge
- Fact ingestion from text
- Analogical reasoning
- Multi-hop queries
- Relation exploration

Core Principle: Structure IS the data. Learning IS structure update.

Usage:
    python -m truthspace_lcm.chat
    
Or:
    from truthspace_lcm.chat import GeometricChat
    chat = GeometricChat()
    chat.run()
"""

import re
from typing import Tuple, Optional, List, Dict

from truthspace_lcm.core import (
    GeometricLCM,
    Vocabulary,
    cosine_similarity,
)


BOOTSTRAP_FACTS = [
    # Geography
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "Rome is the capital of Italy.",
    "Madrid is the capital of Spain.",
    "London is the capital of UK.",
    
    # Continents
    "France is in Europe.",
    "Germany is in Europe.",
    "Japan is in Asia.",
    "Italy is in Europe.",
    
    # Literature
    "Melville wrote Moby Dick.",
    "Shakespeare wrote Hamlet.",
    "Orwell wrote 1984.",
    "Tolkien wrote Lord of the Rings.",
    
    # About TruthSpace
    "TruthSpace is a geometric approach to language understanding.",
    "GeometricLCM stores knowledge as geometry.",
    "Learning updates the geometric structure.",
    "Analogies work by transferring learned relations.",
]


class GeometricChat:
    """
    Interactive chat interface demonstrating the Dynamic Geometric LCM.
    
    This showcases:
    - Natural language fact ingestion
    - Relational queries
    - Analogical reasoning
    - Multi-hop reasoning
    """
    
    def __init__(self, dim: int = 256):
        self.lcm = GeometricLCM(dim=dim)
        self.debug_mode = False
        self._bootstrap()
    
    def _bootstrap(self):
        """Load initial knowledge."""
        print("Initializing Geometric Chat System...")
        print("  Loading bootstrap knowledge...")
        
        for fact in BOOTSTRAP_FACTS:
            self.lcm.ingest(fact)
        
        # Learn the structure
        self.lcm.learn(n_iterations=50, verbose=False)
        
        status = self.lcm.status()
        print(f"  Loaded {status['facts']} facts, {status['entities']} entities, {status['relations']} relations")
        
        # Show relation consistencies
        for rel, cons in status['consistencies'].items():
            print(f"    {rel}: {cons:.1%} consistency")
        
        print("Ready!\n")
    
    def process(self, user_input: str) -> str:
        """Process user input and return response."""
        user_input = user_input.strip()
        if not user_input:
            return ""
        
        # Handle commands
        if user_input.startswith('/'):
            return self._process_command(user_input)
        
        # Handle questions - expanded patterns
        lower = user_input.lower()
        question_starters = ('what', 'who', 'where', 'how', 'why', 'is', 'are', 'can', 'tell me', 'do you know')
        if user_input.endswith('?') or lower.startswith(question_starters):
            return self._process_question(user_input)
        
        # Handle statements (learn new facts)
        return self._process_statement(user_input)
    
    def _process_command(self, cmd: str) -> str:
        """Process a command."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == '/help':
            return self._help()
        
        elif command == '/status':
            return self._status()
        
        elif command == '/relations':
            return self._list_relations()
        
        elif command == '/entities':
            return self._list_entities(args)
        
        elif command == '/analogy':
            return self._analogy(args)
        
        elif command == '/path':
            return self._find_path(args)
        
        elif command == '/similar':
            return self._find_similar(args)
        
        elif command == '/debug':
            if args.lower() == 'on':
                self.debug_mode = True
                return "Debug mode enabled."
            elif args.lower() == 'off':
                self.debug_mode = False
                return "Debug mode disabled."
            return f"Debug is {'on' if self.debug_mode else 'off'}"
        
        elif command == '/quit':
            return "QUIT"
        
        return f"Unknown command: {command}. Type /help for available commands."
    
    def _process_question(self, question: str) -> str:
        """Process a question."""
        # Try the natural language interface first
        answer = self.lcm.ask(question)
        
        # If we got a real answer, return it
        if "don't know" not in answer.lower() and "don't understand" not in answer.lower():
            if self.debug_mode:
                answer += f"\n[DEBUG: NL query matched]"
            return answer
        
        # Try to extract a query pattern
        q = question.lower().strip('?').strip()
        
        # Pattern: "what did X write" / "what did X do"
        match = re.search(r"what\s+did\s+(\w+)\s+(\w+)", q)
        if match:
            subject, verb = match.groups()
            verb_map = {'write': 'wrote', 'found': 'founded', 'create': 'wrote'}
            relation = verb_map.get(verb, verb)
            results = self.lcm.query(subject, relation, k=3)
            if results and results[0][1] > 0.5:
                answers = [f"{e} ({s:.2f})" for e, s in results]
                return f"{subject.title()} {relation}: {', '.join(answers)}"
        
        # Pattern: "tell me about X" / "what do you know about X"
        match = re.search(r"(?:tell\s+me\s+about|what\s+(?:do\s+you\s+)?know\s+about)\s+(.+)", q)
        if match:
            entity = match.group(1).strip().replace(' ', '_')
            return self._describe_entity(entity)
        
        # Fallback: search for any entity mentioned and describe it
        words = q.replace('_', ' ').split()
        for word in words:
            word = word.strip()
            if word in self.lcm.entities:
                return self._describe_entity(word)
        
        # Try fuzzy entity match
        entity_query = q.replace(' ', '_')
        best_match = self.lcm._find_best_entity_match(entity_query)
        if best_match:
            return self._describe_entity(best_match)
        
        return "I don't have enough information to answer that. Try teaching me with a statement like 'Paris is the capital of France.'"
    
    def _describe_entity(self, entity: str) -> str:
        """Describe what we know about an entity."""
        entity = entity.lower()
        
        if entity not in self.lcm.entities:
            # Try fuzzy match
            best = self.lcm._find_best_entity_match(entity)
            if best:
                entity = best
            else:
                return f"I don't know anything about {entity}."
        
        info = []
        
        # Check forward relations (entity --rel--> ?)
        for rel in self.lcm.relations:
            results = self.lcm.query(entity, rel, k=1)
            if results and results[0][1] > 0.5:
                info.append(f"{rel}: {results[0][0]}")
        
        # Check inverse relations (? --rel--> entity)
        for rel in self.lcm.relations:
            results = self.lcm.inverse_query(entity, rel, k=1)
            if results and results[0][1] > 0.5:
                info.append(f"{results[0][0]} {rel} this")
        
        if info:
            return f"About {entity.replace('_', ' ')}: " + ", ".join(info)
        
        # Show similar entities as fallback
        similar = self.lcm.similar(entity, k=3)
        if similar:
            sims = [f"{e} ({s:.2f})" for e, s in similar]
            return f"I don't have specific facts about {entity.replace('_', ' ')}, but similar entities are: {', '.join(sims)}"
        
        return f"I don't know much about {entity.replace('_', ' ')}."
    
    def _process_statement(self, statement: str) -> str:
        """Process a statement (learn new fact)."""
        result = self.lcm.tell(statement)
        
        if "Learned" in result:
            if self.debug_mode:
                status = self.lcm.status()
                result += f"\n[DEBUG: {status['entities']} entities, {status['facts']} facts]"
        
        return result
    
    def _help(self) -> str:
        """Return help text."""
        return """
GEOMETRIC CHAT SYSTEM - Commands
================================

QUESTIONS:
  Just type a question ending with ?
  Examples:
    What is the capital of France?
    Who wrote Hamlet?
    Where is Paris?

STATEMENTS (teach new facts):
  Just type a statement
  Examples:
    Beijing is the capital of China.
    Hemingway wrote The Old Man and the Sea.

COMMANDS:
  /help          Show this help
  /status        Show system status
  /relations     List known relations
  /entities [n]  List entities (optional: show n)
  /analogy A B C Solve A:B :: C:?
  /path A B      Find path from A to B
  /similar X     Find entities similar to X
  /debug on|off  Toggle debug mode
  /quit          Exit

EXAMPLES:
  /analogy france paris germany
  /path eiffel_tower europe
  /similar paris
"""
    
    def _status(self) -> str:
        """Return system status."""
        status = self.lcm.status()
        lines = [
            "SYSTEM STATUS",
            "=============",
            f"Entities: {status['entities']}",
            f"Relations: {status['relations']}",
            f"Facts: {status['facts']}",
            "",
            "Relation Consistencies:",
        ]
        for rel, cons in status['consistencies'].items():
            lines.append(f"  {rel}: {cons:.1%}")
        
        return "\n".join(lines)
    
    def _list_relations(self) -> str:
        """List known relations."""
        if not self.lcm.relations:
            return "No relations learned yet."
        
        lines = ["Known Relations:"]
        for name, rel in self.lcm.relations.items():
            lines.append(f"  {name}: {rel.instance_count} instances, {rel.consistency:.1%} consistency")
        
        return "\n".join(lines)
    
    def _list_entities(self, args: str) -> str:
        """List entities."""
        n = 20
        if args:
            try:
                n = int(args)
            except:
                pass
        
        entities = list(self.lcm.entities.keys())[:n]
        
        if not entities:
            return "No entities learned yet."
        
        return f"Entities ({len(self.lcm.entities)} total): " + ", ".join(entities)
    
    def _analogy(self, args: str) -> str:
        """Solve an analogy."""
        parts = args.split()
        if len(parts) < 3:
            return "Usage: /analogy A B C (solves A:B :: C:?)"
        
        a, b, c = parts[0], parts[1], parts[2]
        results = self.lcm.analogy(a, b, c, k=5)
        
        if not results:
            return f"Cannot solve analogy: missing entities ({a}, {b}, or {c})"
        
        lines = [f"Analogy: {a}:{b} :: {c}:?", ""]
        for entity, sim in results[:5]:
            lines.append(f"  {entity}: {sim:.3f}")
        
        return "\n".join(lines)
    
    def _find_path(self, args: str) -> str:
        """Find path between entities."""
        parts = args.split()
        if len(parts) < 2:
            return "Usage: /path A B (find path from A to B)"
        
        start, end = parts[0], parts[1]
        paths = self.lcm.find_path(start, end, max_hops=3, k=5)
        
        if not paths:
            return f"No path found from {start} to {end}"
        
        lines = [f"Paths from {start} to {end}:", ""]
        for path, conf in paths[:5]:
            path_str = " → ".join(path)
            lines.append(f"  {path_str} (conf: {conf:.3f})")
        
        return "\n".join(lines)
    
    def _find_similar(self, args: str) -> str:
        """Find similar entities."""
        if not args:
            return "Usage: /similar X (find entities similar to X)"
        
        entity = args.strip().replace(' ', '_')
        results = self.lcm.similar(entity, k=10)
        
        if not results:
            return f"Entity '{entity}' not found"
        
        lines = [f"Entities similar to {entity}:", ""]
        for e, sim in results:
            lines.append(f"  {e}: {sim:.3f}")
        
        return "\n".join(lines)
    
    def run(self):
        """Run interactive chat loop."""
        print("=" * 60)
        print("  TRUTHSPACE GEOMETRIC CHAT SYSTEM")
        print("  Structure IS the data. Learning IS structure update.")
        print("=" * 60)
        print("\nType /help for commands, /quit to exit.")
        print("Ask questions or teach me new facts!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                response = self.process(user_input)
                
                if response == "QUIT":
                    print("\nGoodbye!\n")
                    break
                
                print(f"\nGCS: {response}\n")
                
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!\n")
                break


def demo():
    """Run a demonstration of the chat system."""
    print("=" * 60)
    print("  GEOMETRIC CHAT SYSTEM - Demo")
    print("=" * 60)
    print()
    
    chat = GeometricChat()
    
    # Demo Q&A
    print("\n--- Question Answering ---\n")
    questions = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "Where is Tokyo?",
    ]
    for q in questions:
        response = chat.process(q)
        print(f"Q: {q}")
        print(f"A: {response}\n")
    
    # Demo analogies
    print("\n--- Analogical Reasoning ---\n")
    analogies = [
        "france paris germany",
        "melville moby_dick shakespeare",
    ]
    for args in analogies:
        response = chat.process(f"/analogy {args}")
        print(response)
        print()
    
    # Demo learning
    print("\n--- Learning New Facts ---\n")
    new_facts = [
        "Beijing is the capital of China.",
        "Hemingway wrote The Old Man and the Sea.",
    ]
    for fact in new_facts:
        response = chat.process(fact)
        print(f"Input: {fact}")
        print(f"Response: {response}\n")
    
    # Test new knowledge
    print("\n--- Testing New Knowledge ---\n")
    response = chat.process("What is the capital of China?")
    print(f"Q: What is the capital of China?")
    print(f"A: {response}\n")
    
    response = chat.process("/analogy france paris china")
    print(response)
    
    print("\n--- Demo Complete ---\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        chat = GeometricChat()
        chat.run()
