#!/usr/bin/env python3
"""
Build Corpus Knowledge from Literary Analysis

This script builds bootstrap knowledge by analyzing actual verb usage
patterns across a corpus of literary works.

Key insight: Most verbs in literature are used with pronouns, not
capitalized entities. We need patterns that:
1. Work with single entities (ENTITY verb)
2. Handle the actual distribution of usage
3. Focus on high-value verbs that appear across many books
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_book import clean_gutenberg_text, split_into_sentences


# Verb categories with semantic meaning
VERB_CATEGORIES = {
    "speech": {
        "verbs": ["said", "asked", "replied", "cried", "shouted", "whispered", 
                  "exclaimed", "answered", "called", "muttered"],
        "relation": "speaks",
        "single_entity": True,  # X said -> (X, speaks, true)
    },
    "perception": {
        "verbs": ["looked", "saw", "watched", "noticed", "found", "observed", "heard"],
        "relation": "perceives",
        "single_entity": True,
    },
    "movement": {
        "verbs": ["walked", "ran", "turned", "moved", "came", "went", "returned",
                  "entered", "left", "arrived", "departed"],
        "relation": "moves",
        "single_entity": True,
    },
    "mental": {
        "verbs": ["thought", "knew", "believed", "felt", "wanted", "needed",
                  "remembered", "forgot", "understood", "realized"],
        "relation": "thinks",
        "single_entity": True,
    },
    "action": {
        "verbs": ["took", "gave", "brought", "made", "put", "held", "kept"],
        "relation": "acts",
        "single_entity": True,
    },
    "emotion": {
        "verbs": ["loved", "hated", "feared", "liked", "enjoyed"],
        "relation": "feels",
        "single_entity": False,  # X loved Y -> (X, loves, Y)
    },
    "expression": {
        "verbs": ["smiled", "laughed", "sighed", "nodded", "frowned", "wept"],
        "relation": "expresses",
        "single_entity": True,
    },
    "social": {
        "verbs": ["met", "married", "kissed", "helped", "followed", "joined"],
        "relation": "interacts",
        "single_entity": False,  # X met Y
    },
    "conflict": {
        "verbs": ["killed", "attacked", "fought", "struck", "hit"],
        "relation": "conflicts",
        "single_entity": False,
    },
}


class CorpusKnowledgeBuilder:
    """Build bootstrap knowledge from corpus analysis."""
    
    def __init__(self):
        self.verb_stats = defaultdict(lambda: {
            'total': 0,
            'with_entity': 0,
            'with_two_entities': 0,
            'books': set(),
            'examples': [],
            'category': None,
        })
        self.books_processed = []
        
    def analyze_book(self, text: str, book_name: str, max_sentences: int = 1500):
        """Analyze verb usage in a book."""
        sentences = split_into_sentences(text)[:max_sentences]
        self.books_processed.append(book_name)
        
        # Build verb forms lookup
        all_verbs = set()
        verb_to_category = {}
        for cat_name, cat_info in VERB_CATEGORIES.items():
            for verb in cat_info["verbs"]:
                all_verbs.add(verb)
                verb_to_category[verb] = cat_name
        
        for sentence in sentences:
            s_lower = sentence.lower()
            
            for verb in all_verbs:
                # Generate verb forms
                forms = self._get_verb_forms(verb)
                
                for form in forms:
                    # Check if form appears in sentence
                    if f' {form} ' in s_lower or f' {form},' in s_lower or f' {form}.' in s_lower or f' {form}!' in s_lower or f' {form}?' in s_lower:
                        self.verb_stats[verb]['total'] += 1
                        self.verb_stats[verb]['books'].add(book_name)
                        self.verb_stats[verb]['category'] = verb_to_category.get(verb)
                        
                        # Check for ENTITY verb pattern
                        entity_pattern = rf'([A-Z][a-z]+)\s+{form}'
                        match = re.search(entity_pattern, sentence)
                        if match:
                            entity = match.group(1)
                            # Filter common non-entities
                            if entity.lower() not in {'the', 'a', 'an', 'i', 'he', 'she', 'it', 'they', 'we', 'you', 'but', 'and', 'or', 'if', 'when', 'then', 'there', 'here', 'this', 'that', 'what', 'who', 'how', 'why'}:
                                self.verb_stats[verb]['with_entity'] += 1
                                
                                # Check for second entity after verb
                                rest = sentence[match.end():]
                                second_entity = re.search(r'^\s+([A-Z][a-z]+)', rest)
                                if second_entity:
                                    self.verb_stats[verb]['with_two_entities'] += 1
                                
                                # Store example
                                if len(self.verb_stats[verb]['examples']) < 3:
                                    context = sentence[max(0, match.start()-5):match.end()+40]
                                    self.verb_stats[verb]['examples'].append(context)
                        
                        break  # Only count once per sentence
    
    def _get_verb_forms(self, verb: str) -> List[str]:
        """Get common forms of a verb."""
        forms = [verb]
        
        if verb.endswith('e'):
            forms.extend([verb + 'd', verb[:-1] + 'ing', verb + 's'])
        elif verb.endswith('y'):
            forms.extend([verb[:-1] + 'ied', verb + 'ing', verb[:-1] + 'ies'])
        else:
            forms.extend([verb + 'ed', verb + 'ing', verb + 's'])
        
        # Irregular forms
        irregulars = {
            'said': ['say', 'says', 'saying'],
            'thought': ['think', 'thinks', 'thinking'],
            'knew': ['know', 'knows', 'knowing'],
            'felt': ['feel', 'feels', 'feeling'],
            'took': ['take', 'takes', 'taking'],
            'gave': ['give', 'gives', 'giving'],
            'came': ['come', 'comes', 'coming'],
            'went': ['go', 'goes', 'going'],
            'saw': ['see', 'sees', 'seeing'],
            'ran': ['run', 'runs', 'running'],
            'made': ['make', 'makes', 'making'],
            'found': ['find', 'finds', 'finding'],
            'brought': ['bring', 'brings', 'bringing'],
            'held': ['hold', 'holds', 'holding'],
            'kept': ['keep', 'keeps', 'keeping'],
            'left': ['leave', 'leaves', 'leaving'],
            'met': ['meet', 'meets', 'meeting'],
        }
        
        if verb in irregulars:
            forms.extend(irregulars[verb])
        
        return forms
    
    def get_high_value_patterns(self, min_books: int = 3, min_entity_rate: float = 0.05) -> List[Dict]:
        """Get patterns that appear across multiple books with reasonable entity usage."""
        patterns = []
        
        for verb, stats in self.verb_stats.items():
            num_books = len(stats['books'])
            entity_rate = stats['with_entity'] / stats['total'] if stats['total'] > 0 else 0
            
            if num_books >= min_books and entity_rate >= min_entity_rate and stats['total'] >= 10:
                category = stats['category']
                cat_info = VERB_CATEGORIES.get(category, {})
                
                pattern = {
                    'verb': verb,
                    'category': category,
                    'relation': cat_info.get('relation', verb + 's'),
                    'single_entity': cat_info.get('single_entity', True),
                    'total_occurrences': stats['total'],
                    'with_entity': stats['with_entity'],
                    'with_two_entities': stats['with_two_entities'],
                    'entity_rate': entity_rate,
                    'num_books': num_books,
                    'books': list(stats['books']),
                    'examples': stats['examples'],
                }
                patterns.append(pattern)
        
        # Sort by books, then by entity count
        patterns.sort(key=lambda x: (-x['num_books'], -x['with_entity']))
        return patterns
    
    def generate_bootstrap_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Generate JSON patterns for bootstrap_knowledge.json."""
        bootstrap_patterns = []
        
        for p in patterns:
            verb = p['verb']
            relation = p['relation']
            
            # Get all verb forms for regex
            forms = self._get_verb_forms(verb)
            forms_pattern = '|'.join(forms)
            
            if p['single_entity']:
                # Single entity pattern: ENTITY verb -> (entity, relation, true)
                pattern = {
                    "name": f"{relation}_{verb}",
                    "regex": f"([A-Z][a-z]+)\\s+({forms_pattern})(?:\\s|[,\\.!?])",
                    "groups": ["entity"],
                    "fact": ["entity", relation, "true"],
                    "description": f"[ENTITY] {verb} -> (entity, {relation}, true)",
                    "corpus_derived": True,
                    "num_books": p['num_books'],
                    "entity_rate": round(p['entity_rate'], 3),
                }
            else:
                # Two entity pattern: ENTITY verb ENTITY -> (entity1, relation, entity2)
                pattern = {
                    "name": f"{relation}_{verb}",
                    "regex": f"([A-Z][a-z]+)\\s+({forms_pattern})\\s+([A-Z][a-z]+)",
                    "groups": ["entity1", "entity2"],
                    "fact": ["entity1", relation, "entity2"],
                    "description": f"[ENTITY] {verb} [ENTITY2] -> (entity1, {relation}, entity2)",
                    "corpus_derived": True,
                    "num_books": p['num_books'],
                    "entity_rate": round(p['entity_rate'], 3),
                }
            
            bootstrap_patterns.append(pattern)
        
        return bootstrap_patterns
    
    def save_corpus_knowledge(self, filepath: str = None) -> str:
        """Save corpus knowledge to JSON."""
        if filepath is None:
            filepath = Path(__file__).parent.parent / "truthspace_lcm" / "corpus_knowledge.json"
        
        patterns = self.get_high_value_patterns()
        bootstrap_patterns = self.generate_bootstrap_patterns(patterns)
        
        data = {
            "version": "1.0",
            "description": "Bootstrap knowledge derived from literary corpus analysis",
            "books_analyzed": self.books_processed,
            "analysis_summary": {
                "total_verbs_tracked": len(self.verb_stats),
                "high_value_patterns": len(patterns),
            },
            "verb_statistics": {
                verb: {
                    'total': stats['total'],
                    'with_entity': stats['with_entity'],
                    'entity_rate': round(stats['with_entity'] / stats['total'], 3) if stats['total'] > 0 else 0,
                    'books': len(stats['books']),
                    'category': stats['category'],
                }
                for verb, stats in sorted(self.verb_stats.items(), key=lambda x: -x[1]['total'])
                if stats['total'] >= 10
            },
            "high_value_patterns": patterns,
            "bootstrap_patterns": bootstrap_patterns,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def print_report(self):
        """Print analysis report."""
        print("\n" + "="*60)
        print("CORPUS KNOWLEDGE ANALYSIS")
        print("="*60)
        
        print(f"\nBooks analyzed: {len(self.books_processed)}")
        print(f"  {', '.join(self.books_processed)}")
        
        patterns = self.get_high_value_patterns()
        
        print(f"\nHigh-value patterns found: {len(patterns)}")
        print("\nTop patterns by book coverage:")
        
        for p in patterns[:15]:
            print(f"\n  {p['verb']} ({p['category']}):")
            print(f"    Books: {p['num_books']}, Total: {p['total_occurrences']}, With Entity: {p['with_entity']} ({p['entity_rate']*100:.0f}%)")
            if p['examples']:
                print(f"    Example: ...{p['examples'][0][:50]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build corpus knowledge")
    parser.add_argument("--sentences", "-n", type=int, default=1500, help="Max sentences per book")
    parser.add_argument("--min-books", "-m", type=int, default=3, help="Min books for pattern")
    parser.add_argument("--save", "-s", action="store_true", help="Save to JSON")
    
    args = parser.parse_args()
    
    # Available books
    books = [
        ("Pride and Prejudice", "/tmp/pride_prejudice.txt"),
        ("Alice in Wonderland", "/tmp/alice.txt"),
        ("Frankenstein", "/tmp/frankenstein.txt"),
        ("Moby Dick", "/tmp/moby_dick.txt"),
        ("Sherlock Holmes", "/tmp/sherlock.txt"),
        ("Tale of Two Cities", "/tmp/tale_two_cities.txt"),
        ("Tom Sawyer", "/tmp/tom_sawyer.txt"),
        ("Dracula", "/tmp/dracula.txt"),
        ("Great Expectations", "/tmp/expectations.txt"),
    ]
    
    builder = CorpusKnowledgeBuilder()
    
    print("Analyzing books...")
    for name, path in books:
        if os.path.exists(path):
            with open(path, 'r', errors='ignore') as f:
                text = clean_gutenberg_text(f.read())
            print(f"  {name}: {len(text):,} chars")
            builder.analyze_book(text, name, max_sentences=args.sentences)
    
    builder.print_report()
    
    if args.save:
        filepath = builder.save_corpus_knowledge()
        print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
