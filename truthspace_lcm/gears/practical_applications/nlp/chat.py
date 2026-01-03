"""
Interactive Chat Application for the Gear Chain System

A conversational interface using the modular gear chain for text transformation.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import sys
from typing import Optional, List
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from truthspace_lcm.gears.core import GearChain, GearState
from truthspace_lcm.gears.practical_applications.nlp.error_correction import ErrorCorrectionGear
from truthspace_lcm.gears.practical_applications.nlp import (
    RoleGear, ActionGear, TenseGear,
    SignalGear, DomainGear, StructureGear, OutputGear,
)
from truthspace_lcm.gears.corpus import load_corpus, get_corpus_path
from truthspace_lcm.core.geometric import GeometricQA


class GearChat:
    """
    Interactive chat interface using the gear chain system.
    
    Features:
    - Natural language queries about concepts
    - Runtime gear manipulation (change tense, style, etc.)
    - Corpus exploration
    - Debug mode to see gear transformations
    """
    
    def __init__(self, corpus_name: str = "experimental", debug: bool = False):
        self.debug = debug
        
        # Load corpus via GeometricQA for querying
        self.qa = GeometricQA()
        corpus_path = get_corpus_path(corpus_name)
        self.qa.load_corpus(str(corpus_path))
        self.qa.set_output_lens('natural')
        
        # Build gear chain
        self.chain = GearChain("ChatChain")
        self.chain.add(RoleGear())
        self.chain.add(ActionGear())
        self.chain.add(TenseGear(tense='present'))
        self.chain.add(ErrorCorrectionGear())
        self.chain.add(DomainGear())
        self.chain.add(StructureGear())
        self.chain.add(OutputGear())
        
        # Command history
        self.history: List[str] = []
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth output into gear state."""
        truth_lower = truth.lower()
        
        state = GearState()
        state.entity = concept.title()
        
        # Role - handle "is someone" as well as "is a X"
        match = re.search(r'is (someone|a[n]? (\w+))', truth_lower)
        if match:
            if match.group(1) == 'someone':
                state.role = 'entity'
            else:
                state.role = match.group(2) or 'entity'
        
        # Actions - handle "who/that verbs" pattern
        match = re.search(r'(?:who|that)\s+(\w+)(?:,\s*(\w+))?\s+and\s+(\w+)', truth_lower)
        if match:
            state.actions = [a for a in match.groups() if a]
        else:
            # Try simpler pattern
            match = re.search(r'(?:who|that)\s+(\w+)', truth_lower)
            if match:
                state.actions = [match.group(1)]
        
        # Targets - handle "relates to X and Y"
        match = re.search(r'relates? to\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            state.targets = [t for t in match.groups() if t]
        
        return state
    
    def query(self, question: str) -> str:
        """Process a query through the gear chain."""
        # Extract concept from question
        question_lower = question.lower().strip()
        
        # Handle "what is X" questions
        match = re.search(r'what (?:is|are) (?:a |an |the )?(\w+)', question_lower)
        if match:
            concept = match.group(1)
        else:
            # Try to extract any noun
            words = question_lower.split()
            concept = words[-1].rstrip('?') if words else 'unknown'
        
        # Get truth from corpus
        truth = self.qa.ask(f"What is {concept}?")
        
        if "don't know" in truth.lower():
            return f"I don't have information about '{concept}' in my knowledge base."
        
        # Parse to state
        state = self._parse_to_state(truth, concept)
        
        if self.debug:
            print(f"\n[DEBUG] Raw truth: {truth}")
            print(f"[DEBUG] Initial state: entity={state.entity}, role={state.role}, actions={state.actions}")
        
        # Process through gear chain
        result = self.chain.process(state)
        
        if self.debug:
            print(f"[DEBUG] Final output: {result}\n")
        
        return result
    
    def handle_command(self, cmd: str) -> Optional[str]:
        """Handle special commands."""
        cmd = cmd.strip().lower()
        
        if cmd in ['/help', '/h', '/?']:
            return self._help_text()
        
        if cmd == '/gears':
            return f"Current gear chain:\n  {self.chain}"
        
        if cmd == '/debug':
            self.debug = not self.debug
            return f"Debug mode: {'ON' if self.debug else 'OFF'}"
        
        if cmd.startswith('/tense '):
            tense = cmd.split()[1]
            if tense in ['present', 'past', 'future', 'perfect']:
                self.chain.get("TenseGear").set_tense(tense)
                return f"Tense set to: {tense}"
            return f"Invalid tense. Use: present, past, future, perfect"
        
        if cmd.startswith('/ratio '):
            parts = cmd.split()
            if len(parts) == 3:
                gear_name, ratio = parts[1], float(parts[2])
                gear = self.chain.get(gear_name)
                if gear:
                    gear.set_ratio(ratio)
                    return f"Set {gear_name} ratio to {ratio}"
                return f"Gear not found: {gear_name}"
            return "Usage: /ratio GearName 0.5"
        
        if cmd.startswith('/disable '):
            gear_name = cmd.split()[1]
            gear = self.chain.get(gear_name)
            if gear:
                gear.disable()
                return f"Disabled: {gear_name}"
            return f"Gear not found: {gear_name}"
        
        if cmd.startswith('/enable '):
            gear_name = cmd.split()[1]
            gear = self.chain.get(gear_name)
            if gear:
                gear.enable()
                return f"Enabled: {gear_name}"
            return f"Gear not found: {gear_name}"
        
        if cmd == '/stats':
            error_gear = self.chain.get("ErrorCorrectionGear")
            stats = error_gear.get_stats() if error_gear else {}
            return f"Error correction stats:\n  Verb rules: {stats.get('verb_rules', 0)}\n  Word rules: {stats.get('word_rules', 0)}\n  Corrections applied: {stats.get('total_corrections', 0)}"
        
        if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
            return None  # Signal to exit
        
        return f"Unknown command: {cmd}\nType /help for available commands."
    
    def _help_text(self) -> str:
        return """
Gear Chain Chat - Commands:

  /help, /h       Show this help
  /gears          Show current gear chain
  /debug          Toggle debug mode
  /tense <t>      Set tense (present, past, future, perfect)
  /ratio <g> <r>  Set gear ratio (e.g., /ratio ActionGear 0.5)
  /disable <g>    Disable a gear
  /enable <g>     Enable a gear
  /stats          Show error correction stats
  /quit, /q       Exit

Ask questions like:
  What is evolution?
  What is Holmes?
  Tell me about physics
"""
    
    def run(self):
        """Run the interactive chat loop."""
        print()
        print("=" * 60)
        print("  Gear Chain Chat")
        print("  Modular, Interpretable Language Understanding")
        print("=" * 60)
        print()
        print(f"  Chain: {self.chain}")
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
                
                # Process query
                response = self.query(user_input)
                print(f"\nGear: {response}\n")
                
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


def main(debug: bool = False):
    """Entry point for the chat application."""
    chat = GearChat(debug=debug)
    chat.run()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gear Chain Chat")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    sys.exit(main(debug=args.debug))
