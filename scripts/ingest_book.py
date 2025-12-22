#!/usr/bin/env python3
"""
Book Ingestion Pipeline for GeometricLCM

Ingests a book from Project Gutenberg and extracts:
1. Character relationships (X is Y, X and Y are friends)
2. Actions (X did Y, X went to Y)
3. Descriptions (X is a Y, X has Y)
4. Locations (X is in Y, X went to Y)

This is an exploration of what "bootstrapped instinct knowledge" would need:
- Pattern recognition for sentence structures
- Entity recognition (names, places, things)
- Relation extraction
- Coreference resolution (he/she/it → actual entity)

Current limitations (to be addressed with bootstrapped knowledge):
- Hard-coded patterns (should be learned)
- No coreference resolution
- No context understanding
- No temporal reasoning
"""

import re
import sys
import os
from typing import List, Tuple, Dict, Set
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core import GeometricLCM, BootstrapKnowledge, get_bootstrap_knowledge


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg header/footer and clean text."""
    # Find start of actual content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]
    
    for marker in start_markers:
        if marker in text:
            idx = text.find(marker)
            # Skip past the marker line
            idx = text.find('\n', idx) + 1
            text = text[idx:]
            break
    
    # Find end of actual content
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg",
    ]
    
    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[:idx]
            break
    
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) > 10]  # Skip very short
    
    return sentences


def extract_chapter_text(text: str, chapter_num: int = None) -> str:
    """Extract text from a specific chapter or all chapters."""
    if chapter_num is None:
        return text
    
    # Find chapter start
    pattern = rf"CHAPTER\s+{chapter_num}\.?\s"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return ""
    
    start = match.end()
    
    # Find next chapter
    next_pattern = rf"CHAPTER\s+{chapter_num + 1}\.?\s"
    next_match = re.search(next_pattern, text[start:], re.IGNORECASE)
    
    if next_match:
        end = start + next_match.start()
    else:
        end = len(text)
    
    return text[start:end].strip()


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

# Known characters in Moby Dick (bootstrap knowledge for this book)
MOBY_DICK_CHARACTERS = {
    "ahab": "captain",
    "ishmael": "narrator",
    "queequeg": "harpooner",
    "starbuck": "first_mate",
    "stubb": "second_mate",
    "flask": "third_mate",
    "tashtego": "harpooner",
    "daggoo": "harpooner",
    "fedallah": "harpooner",
    "pip": "cabin_boy",
    "perth": "blacksmith",
    "elijah": "prophet",
    "bildad": "owner",
    "peleg": "owner",
    "moby dick": "whale",
    "pequod": "ship",
}

# Common relation patterns
RELATION_PATTERNS = [
    # "X is the Y of Z" → (Z, has_Y, X)
    (r"(\w+)\s+is\s+the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)", lambda m: (m.group(3), f"has_{m.group(2)}", m.group(1))),
    
    # "X is a Y" → (X, is_a, Y)
    (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", lambda m: (m.group(1), "is_a", m.group(2))),
    
    # "X, the Y" → (X, is_a, Y)
    (r"(\w+),\s+the\s+(\w+)", lambda m: (m.group(1), "is_a", m.group(2))),
    
    # "Captain X" → (X, role, captain)
    (r"Captain\s+(\w+)", lambda m: (m.group(1), "role", "captain")),
    
    # "X said" / "X replied" → (X, speaks, true) - marks as character
    (r"(\w+)\s+(?:said|replied|cried|shouted|whispered|asked|answered)", lambda m: (m.group(1), "speaks", "true")),
    
    # "X and Y" when both are known characters → (X, associated_with, Y)
    (r"(\w+)\s+and\s+(\w+)", lambda m: (m.group(1), "associated_with", m.group(2))),
    
    # "X went to Y" / "X sailed to Y" → (X, went_to, Y)
    (r"(\w+)\s+(?:went|sailed|traveled|journeyed)\s+to\s+(\w+)", lambda m: (m.group(1), "went_to", m.group(2))),
    
    # "X killed Y" / "X struck Y" → (X, attacked, Y)
    (r"(\w+)\s+(?:killed|struck|attacked|harpooned|chased)\s+(\w+)", lambda m: (m.group(1), "attacked", m.group(2))),
]


def extract_entities(text: str, known_entities: Set[str] = None) -> Set[str]:
    """Extract potential entity names from text."""
    entities = set()
    
    # Capitalized words (potential names)
    caps = re.findall(r'\b([A-Z][a-z]+)\b', text)
    for word in caps:
        word_lower = word.lower()
        # Skip common words
        if word_lower not in {'the', 'a', 'an', 'i', 'he', 'she', 'it', 'they', 'we', 'you'}:
            entities.add(word_lower)
    
    # Known entities
    if known_entities:
        for entity in known_entities:
            if entity.lower() in text.lower():
                entities.add(entity.lower())
    
    return entities


def extract_relations(sentence: str, known_entities: Set[str] = None) -> List[Tuple[str, str, str]]:
    """Extract relations from a sentence."""
    relations = []
    
    for pattern, extractor in RELATION_PATTERNS:
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            try:
                subj, rel, obj = extractor(match)
                subj = subj.lower().strip()
                obj = obj.lower().strip()
                
                # Filter: at least one should be a known entity or capitalized
                if known_entities:
                    if subj in known_entities or obj in known_entities:
                        relations.append((subj, rel, obj))
                else:
                    relations.append((subj, rel, obj))
            except:
                continue
    
    return relations


# =============================================================================
# BOOK INGESTION
# =============================================================================

class BookIngester:
    """Ingest a book into GeometricLCM using BootstrapKnowledge."""
    
    def __init__(self, lcm: GeometricLCM = None, dim: int = 256, 
                 bootstrap: BootstrapKnowledge = None):
        self.lcm = lcm or GeometricLCM(dim=dim)
        self.bootstrap = bootstrap or get_bootstrap_knowledge()
        self.known_entities = set()
        self.relation_counts = defaultdict(int)
    
    def add_bootstrap_knowledge(self, characters: Dict[str, str]):
        """Add known characters and their roles."""
        for name, role in characters.items():
            name = name.lower().replace(' ', '_')
            self.known_entities.add(name)
            self.lcm.add_fact(name, "is_a", role)
            self.lcm.add_fact(name, "is_a", "character")
    
    def ingest_text(self, text: str, max_sentences: int = None, verbose: bool = False) -> Dict:
        """Ingest text and extract relations using BootstrapKnowledge."""
        sentences = split_into_sentences(text)
        
        if max_sentences:
            sentences = sentences[:max_sentences]
        
        stats = {
            'sentences_processed': 0,
            'relations_extracted': 0,
            'entities_found': 0,
        }
        
        for i, sentence in enumerate(sentences):
            # Use BootstrapKnowledge to extract facts
            facts = self.bootstrap.extract_facts(sentence, self.known_entities)
            
            for fact in facts:
                self.lcm.add_fact(fact.subject, fact.relation, fact.object)
                self.relation_counts[fact.relation] += 1
                self.known_entities.add(fact.subject)
                if fact.object != "true":
                    self.known_entities.add(fact.object)
                stats['relations_extracted'] += 1
            
            stats['sentences_processed'] += 1
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(sentences)} sentences, {stats['relations_extracted']} relations")
        
        stats['entities_found'] = len(self.known_entities)
        
        return stats
    
    def learn(self, verbose: bool = False):
        """Learn the geometric structure from extracted facts."""
        if verbose:
            print(f"Learning from {len(self.lcm.facts)} facts...")
        
        consistency = self.lcm.learn(n_iterations=100, target_consistency=0.90, verbose=verbose)
        
        if verbose:
            print(f"Final consistency: {consistency:.1%}")
        
        return consistency
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        status = self.lcm.status()
        return {
            **status,
            'known_entities': len(self.known_entities),
            'relation_types': dict(self.relation_counts),
        }


# =============================================================================
# MAIN
# =============================================================================

def ingest_moby_dick(filepath: str, max_sentences: int = 500, verbose: bool = True):
    """Ingest Moby Dick and return the LCM."""
    
    print("=" * 60)
    print("MOBY DICK INGESTION")
    print("=" * 60)
    print()
    
    # Load text
    print("Loading text...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = clean_gutenberg_text(text)
    print(f"  Text length: {len(text):,} characters")
    
    # Create ingester
    ingester = BookIngester(dim=256)
    
    # Add bootstrap knowledge
    print("\nAdding bootstrap character knowledge...")
    ingester.add_bootstrap_knowledge(MOBY_DICK_CHARACTERS)
    print(f"  Added {len(MOBY_DICK_CHARACTERS)} known characters")
    
    # Ingest text
    print(f"\nIngesting text (max {max_sentences} sentences)...")
    stats = ingester.ingest_text(text, max_sentences=max_sentences, verbose=verbose)
    print(f"  Sentences: {stats['sentences_processed']}")
    print(f"  Relations: {stats['relations_extracted']}")
    print(f"  Entities: {stats['entities_found']}")
    
    # Learn
    print("\nLearning geometric structure...")
    ingester.learn(verbose=verbose)
    
    # Show relation types
    print("\nRelation types found:")
    for rel, count in sorted(ingester.relation_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rel}: {count}")
    
    # Final stats
    print("\nFinal statistics:")
    final_stats = ingester.get_stats()
    print(f"  Entities: {final_stats['entities']}")
    print(f"  Relations: {final_stats['relations']}")
    print(f"  Facts: {final_stats['facts']}")
    
    return ingester.lcm


class BookChat:
    """Chat interface specialized for book knowledge."""
    
    def __init__(self, lcm: GeometricLCM):
        self.lcm = lcm
        self.debug_mode = False
    
    def process(self, user_input: str) -> str:
        """Process user input."""
        user_input = user_input.strip()
        if not user_input:
            return ""
        
        # Commands
        if user_input.startswith('/'):
            return self._process_command(user_input)
        
        # Questions
        return self._process_question(user_input)
    
    def _process_command(self, cmd: str) -> str:
        """Process commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == '/help':
            return """
BOOK CHAT COMMANDS
==================

Questions:
  Who is [character]?     - Get info about a character
  What is [thing]?        - Get info about something
  Tell me about [X]       - Describe an entity
  
Commands:
  /characters             - List known characters
  /relations              - List relation types
  /analogy A B C          - Solve A:B :: C:?
  /similar X              - Find similar entities
  /facts [entity]         - Show facts about entity
  /debug on|off           - Toggle debug mode
  /quit                   - Exit
"""
        
        elif command == '/characters':
            chars = []
            for name, entity in self.lcm.entities.items():
                if entity.entity_type == 'character':
                    chars.append(name)
            if not chars:
                # Find entities with is_a character
                for fact in self.lcm.facts:
                    if fact.relation == 'is_a' and fact.object == 'character':
                        chars.append(fact.subject)
            return f"Characters: {', '.join(sorted(set(chars)))}"
        
        elif command == '/relations':
            rels = []
            for name, rel in self.lcm.relations.items():
                rels.append(f"{name} ({rel.instance_count} facts, {rel.consistency:.0%})")
            return "Relations:\n  " + "\n  ".join(rels)
        
        elif command == '/analogy':
            parts = args.split()
            if len(parts) < 3:
                return "Usage: /analogy A B C"
            a, b, c = parts[0], parts[1], parts[2]
            results = self.lcm.analogy(a, b, c, k=5)
            if not results:
                return f"Cannot solve: missing entities"
            lines = [f"Analogy: {a}:{b} :: {c}:?"]
            for entity, sim in results[:5]:
                lines.append(f"  {entity}: {sim:.3f}")
            return "\n".join(lines)
        
        elif command == '/similar':
            if not args:
                return "Usage: /similar X"
            results = self.lcm.similar(args.lower(), k=10)
            if not results:
                return f"Entity '{args}' not found"
            lines = [f"Similar to {args}:"]
            for entity, sim in results:
                lines.append(f"  {entity}: {sim:.3f}")
            return "\n".join(lines)
        
        elif command == '/facts':
            entity = args.lower() if args else None
            if entity:
                facts = [f for f in self.lcm.facts if f.subject == entity or f.object == entity]
                if not facts:
                    return f"No facts about {entity}"
                lines = [f"Facts about {entity}:"]
                for f in facts:
                    lines.append(f"  {f.subject} --{f.relation}--> {f.object}")
                return "\n".join(lines)
            else:
                return f"Total facts: {len(self.lcm.facts)}"
        
        elif command == '/debug':
            if args.lower() == 'on':
                self.debug_mode = True
                return "Debug mode on"
            elif args.lower() == 'off':
                self.debug_mode = False
                return "Debug mode off"
            return f"Debug is {'on' if self.debug_mode else 'off'}"
        
        elif command == '/quit':
            return "QUIT"
        
        return f"Unknown command: {command}"
    
    def _process_question(self, question: str) -> str:
        """Process a question about the book."""
        q = question.lower().strip('?').strip()
        
        # "Who is X?" - find character info
        match = re.match(r"who\s+is\s+(.+)", q)
        if match:
            entity = match.group(1).strip().replace(' ', '_')
            return self._describe_entity(entity, focus='character')
        
        # "What is X?" - find thing info
        match = re.match(r"what\s+is\s+(?:the\s+)?(.+)", q)
        if match:
            entity = match.group(1).strip().replace(' ', '_')
            return self._describe_entity(entity)
        
        # "Tell me about X"
        match = re.match(r"tell\s+me\s+about\s+(.+)", q)
        if match:
            entity = match.group(1).strip().replace(' ', '_')
            return self._describe_entity(entity)
        
        # "What role does X have?" / "What is X's role?"
        match = re.search(r"(?:what\s+)?role\s+(?:does\s+)?(\w+)", q)
        if match:
            entity = match.group(1).lower()
            results = self.lcm.query(entity, "is_a", k=3)
            if results and results[0][1] > 0.3:
                roles = [r[0] for r in results if r[1] > 0.3]
                return f"{entity.title()} is: {', '.join(roles)}"
        
        # Fallback: try to find any mentioned entity
        words = q.split()
        for word in words:
            word = word.strip().lower()
            if word in self.lcm.entities and word not in {'who', 'what', 'is', 'the', 'a', 'an'}:
                return self._describe_entity(word)
        
        return "I don't have information about that. Try asking about a specific character like 'Who is Ahab?' or use /characters to see known characters."
    
    def _describe_entity(self, entity: str, focus: str = None) -> str:
        """Describe what we know about an entity."""
        entity = entity.lower()
        
        # Try to find the entity
        if entity not in self.lcm.entities:
            # Fuzzy match
            best = self.lcm._find_best_entity_match(entity)
            if best:
                entity = best
            else:
                return f"I don't know about '{entity}'. Try /characters to see known characters."
        
        info = []
        
        # Get is_a relations (type/role)
        results = self.lcm.query(entity, "is_a", k=3)
        if results:
            types = [r[0] for r in results if r[1] > 0.3]
            if types:
                info.append(f"is a {', '.join(types)}")
        
        # Get role relation
        results = self.lcm.query(entity, "role", k=1)
        if results and results[0][1] > 0.3:
            info.append(f"role: {results[0][0]}")
        
        # Get associations
        results = self.lcm.query(entity, "associated_with", k=3)
        if results:
            assocs = [r[0] for r in results if r[1] > 0.3]
            if assocs:
                info.append(f"associated with {', '.join(assocs)}")
        
        # Get inverse relations (who mentions this entity)
        for rel in ['attacked', 'speaks', 'went_to']:
            if rel in self.lcm.relations:
                results = self.lcm.inverse_query(entity, rel, k=2)
                if results and results[0][1] > 0.3:
                    actors = [r[0] for r in results if r[1] > 0.3]
                    if actors:
                        info.append(f"{', '.join(actors)} {rel} {entity}")
        
        if info:
            return f"{entity.replace('_', ' ').title()}: {'; '.join(info)}"
        
        # Fallback: show similar entities
        similar = self.lcm.similar(entity, k=3)
        if similar:
            sims = [f"{e}" for e, s in similar if s > 0.3]
            if sims:
                return f"I don't have specific facts about {entity}, but similar entities are: {', '.join(sims)}"
        
        return f"I don't have much information about {entity}."


def interactive_chat(lcm: GeometricLCM):
    """Interactive chat about the book."""
    chat = BookChat(lcm)
    
    print()
    print("=" * 60)
    print("INTERACTIVE CHAT - Moby Dick")
    print("=" * 60)
    print("\nAsk questions about Moby Dick!")
    print("Type /help for commands, /quit to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            response = chat.process(user_input)
            
            if response == "QUIT":
                print("\nGoodbye!\n")
                break
            
            print(f"\nGCS: {response}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!\n")
            break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest a book into GeometricLCM")
    parser.add_argument("filepath", nargs="?", default="/tmp/moby_dick.txt",
                       help="Path to the book text file")
    parser.add_argument("--max-sentences", "-n", type=int, default=500,
                       help="Maximum sentences to process")
    parser.add_argument("--chat", "-c", action="store_true",
                       help="Start interactive chat after ingestion")
    parser.add_argument("--save", "-s", type=str,
                       help="Save the LCM to a file")
    
    args = parser.parse_args()
    
    # Ingest
    lcm = ingest_moby_dick(args.filepath, max_sentences=args.max_sentences)
    
    # Save if requested
    if args.save:
        lcm.save(args.save)
        print(f"\nSaved to {args.save}")
    
    # Chat if requested
    if args.chat:
        interactive_chat(lcm)


if __name__ == "__main__":
    main()
