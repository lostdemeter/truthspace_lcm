#!/usr/bin/env python3
"""
Output Gear Chain

A gear chain for conditioning raw semantic content into natural language.

The key insight: just as we discovered dimensions for UNDERSTANDING (agent behaviors),
we can discover dimensions for EXPRESSION (how things are said).

This gear chain learns:
1. Sentence patterns from the corpus
2. How to combine fragments into fluent sentences
3. Style/register variations (formal, casual, descriptive)
"""

import json
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SentencePattern:
    """A learned sentence pattern."""
    template: str  # e.g., "{agent} {verb} {object}"
    examples: List[str]
    frequency: int
    style: str  # 'descriptive', 'action', 'comparison', etc.


class OutputGearChain:
    """
    A gear chain for generating natural language output.
    
    Learns sentence patterns from corpus and uses them to
    transform semantic content into fluent text.
    """
    
    def __init__(self):
        # Learned patterns
        self.sentence_patterns: Dict[str, List[SentencePattern]] = defaultdict(list)
        self.phrase_templates: Dict[str, List[str]] = defaultdict(list)
        
        # N-gram model for fluency
        self.bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.trigrams: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Style dimensions (discovered from data)
        self.style_dimensions: List[Dict] = []
        
        # Vocabulary
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.total_words = 0
    
    def learn_from_corpus(self, corpus_path: str):
        """Learn output patterns from a corpus."""
        with open(corpus_path) as f:
            corpus = json.load(f)
        
        frames = corpus.get('frames', [])
        print(f"Learning output patterns from {len(frames)} frames...")
        
        for frame in frames:
            text = frame.get('text', '')
            if not text:
                continue
            
            self._learn_sentence(text)
        
        print(f"Learned {len(self.phrase_templates)} phrase types")
        print(f"Vocabulary: {len(self.word_frequencies)} words")
    
    def _learn_sentence(self, text: str):
        """Learn patterns from a single sentence."""
        # Tokenize
        words = text.lower().split()
        
        # Learn word frequencies
        for word in words:
            word_clean = re.sub(r'[^a-z]', '', word)
            if word_clean:
                self.word_frequencies[word_clean] += 1
                self.total_words += 1
        
        # Learn bigrams
        for i in range(len(words) - 1):
            w1 = re.sub(r'[^a-z]', '', words[i].lower())
            w2 = re.sub(r'[^a-z]', '', words[i + 1].lower())
            if w1 and w2:
                self.bigrams[w1][w2] += 1
        
        # Learn trigrams
        for i in range(len(words) - 2):
            w1 = re.sub(r'[^a-z]', '', words[i].lower())
            w2 = re.sub(r'[^a-z]', '', words[i + 1].lower())
            w3 = re.sub(r'[^a-z]', '', words[i + 2].lower())
            if w1 and w2 and w3:
                self.trigrams[(w1, w2)][w3] += 1
        
        # Learn phrase patterns
        self._extract_patterns(text)
    
    def _extract_patterns(self, text: str):
        """Extract reusable patterns from text."""
        # Description patterns
        if ' is ' in text.lower():
            self.phrase_templates['description'].append(text)
        
        # Action patterns (verb-based)
        words = text.split()
        if len(words) >= 3:
            # Check for action verbs
            for i, word in enumerate(words[1:4], 1):
                word_clean = word.lower().rstrip('.,!?')
                if word_clean.endswith(('s', 'es', 'ed', 'ing')):
                    # This looks like a verb
                    pattern = ' '.join(words[:i]) + ' {action} ' + ' '.join(words[i+1:])
                    self.phrase_templates['action'].append(text)
                    break
        
        # Comparison patterns
        if any(w in text.lower() for w in ['similar', 'different', 'like', 'unlike', 'compared']):
            self.phrase_templates['comparison'].append(text)
        
        # Listing patterns
        if ', ' in text and ' and ' in text:
            self.phrase_templates['listing'].append(text)
    
    def _get_next_word_probability(self, context: List[str]) -> Dict[str, float]:
        """Get probability distribution for next word given context."""
        probs = {}
        
        if len(context) >= 2:
            # Try trigram
            key = (context[-2], context[-1])
            if key in self.trigrams:
                total = sum(self.trigrams[key].values())
                for word, count in self.trigrams[key].items():
                    probs[word] = count / total * 0.6  # Weight trigrams higher
        
        if len(context) >= 1:
            # Add bigram probabilities
            if context[-1] in self.bigrams:
                total = sum(self.bigrams[context[-1]].values())
                for word, count in self.bigrams[context[-1]].items():
                    if word in probs:
                        probs[word] += count / total * 0.3
                    else:
                        probs[word] = count / total * 0.3
        
        # Add unigram smoothing
        for word, count in self.word_frequencies.items():
            if word in probs:
                probs[word] += (count / self.total_words) * 0.1
            else:
                probs[word] = (count / self.total_words) * 0.1
        
        return probs
    
    def fluent_combine(self, fragments: List[str]) -> str:
        """Combine fragments into a fluent sentence."""
        if not fragments:
            return ""
        
        if len(fragments) == 1:
            return fragments[0]
        
        # Score different combinations
        best_result = None
        best_score = -float('inf')
        
        # Try different connectors
        connectors = [
            ". ",
            ", and ",
            ", which ",
            ". Additionally, ",
            " while ",
            ". Furthermore, ",
        ]
        
        for connector in connectors:
            result = connector.join(fragments)
            score = self._score_fluency(result)
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result or ". ".join(fragments)
    
    def _score_fluency(self, text: str) -> float:
        """Score how fluent a piece of text is based on learned patterns."""
        words = [re.sub(r'[^a-z]', '', w.lower()) for w in text.split()]
        words = [w for w in words if w]
        
        if len(words) < 2:
            return 0.0
        
        score = 0.0
        
        # Bigram score
        for i in range(len(words) - 1):
            if words[i] in self.bigrams and words[i + 1] in self.bigrams[words[i]]:
                score += np.log(self.bigrams[words[i]][words[i + 1]] + 1)
            else:
                score -= 1  # Penalty for unseen bigram
        
        # Trigram bonus
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            if key in self.trigrams and words[i + 2] in self.trigrams[key]:
                score += np.log(self.trigrams[key][words[i + 2]] + 1) * 0.5
        
        # Normalize by length
        return score / len(words)
    
    def generate_description(self, agent: str, properties: Dict[str, Any]) -> str:
        """Generate a natural description of an agent."""
        parts = []
        
        # Start with the agent name
        agent_title = agent.replace('_', ' ').title()
        
        # Dimensional properties
        if 'similar_to' in properties:
            similar = properties['similar_to'][:3]
            if similar:
                similar_str = ', '.join([s.replace('_', ' ').title() for s in similar])
                parts.append(f"{agent_title} is similar to {similar_str}")
        
        if 'opposite_of' in properties:
            opposite = properties['opposite_of'].replace('_', ' ').title()
            parts.append(f"and quite different from {opposite}")
        
        if 'dimensions' in properties:
            dims = properties['dimensions']
            if dims:
                dim_desc = []
                for dim_name, info in list(dims.items())[:2]:
                    pole = info.get('pole', '').replace('_', ' ')
                    if pole:
                        dim_desc.append(f"characterized by traits toward {pole}")
                if dim_desc:
                    parts.append(". " + agent_title + " is " + " and ".join(dim_desc))
        
        result = self.fluent_combine(parts) if parts else f"{agent_title} is a known concept."
        
        # Ensure proper capitalization and punctuation
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def generate_comparison(self, agent1: str, agent2: str, 
                           similarity: float, differences: List[Dict]) -> str:
        """Generate a natural comparison between two agents."""
        a1 = agent1.replace('_', ' ').title()
        a2 = agent2.replace('_', ' ').title()
        
        # Similarity description
        if similarity < 0.3:
            sim_desc = "very similar"
        elif similarity < 0.7:
            sim_desc = "somewhat similar"
        elif similarity < 1.0:
            sim_desc = "quite different"
        else:
            sim_desc = "very different"
        
        parts = [f"{a1} and {a2} are {sim_desc}"]
        
        # Key differences
        if differences:
            diff = differences[0]
            dim_name = diff.get('dimension', 'some dimension')
            pole1 = diff.get('pole1', '').replace('_', ' ')
            pole2 = diff.get('pole2', '').replace('_', ' ')
            
            if pole1 and pole2:
                parts.append(f"They differ most in that {a1} tends toward {pole1} while {a2} tends toward {pole2}")
        
        return self.fluent_combine(parts) + "."
    
    def generate_list(self, items: List[str], context: str = "") -> str:
        """Generate a natural listing."""
        if not items:
            return ""
        
        items_formatted = [item.replace('_', ' ').title() for item in items]
        
        if len(items_formatted) == 1:
            return items_formatted[0]
        elif len(items_formatted) == 2:
            return f"{items_formatted[0]} and {items_formatted[1]}"
        else:
            return ', '.join(items_formatted[:-1]) + f", and {items_formatted[-1]}"
    
    def condition_output(self, raw_content: Dict[str, Any]) -> str:
        """
        Main entry point: condition raw semantic content into natural language.
        
        raw_content should contain:
        - 'type': 'description', 'comparison', 'list', 'answer'
        - 'agents': list of agents involved
        - 'properties': dimensional properties
        - 'frames': relevant knowledge frames
        """
        output_type = raw_content.get('type', 'description')
        agents = raw_content.get('agents', [])
        properties = raw_content.get('properties', {})
        frames = raw_content.get('frames', [])
        
        parts = []
        
        if output_type == 'description' and agents:
            for agent in agents[:2]:
                agent_props = properties.get(agent, {})
                desc = self.generate_description(agent, agent_props)
                parts.append(desc)
        
        elif output_type == 'comparison' and len(agents) >= 2:
            similarity = properties.get('distance', 0.5)
            differences = properties.get('differences', [])
            comp = self.generate_comparison(agents[0], agents[1], similarity, differences)
            parts.append(comp)
        
        elif output_type == 'list':
            items = properties.get('items', agents)
            context = properties.get('context', '')
            if context:
                parts.append(f"{context}: {self.generate_list(items)}")
            else:
                parts.append(self.generate_list(items))
        
        # Add relevant frame content
        if frames:
            frame_texts = []
            for frame in frames[:2]:
                text = frame.get('text', '') if isinstance(frame, dict) else str(frame)
                if text:
                    # Clean up the frame text
                    text = text.strip()
                    if len(text) > 150:
                        text = text[:150].rsplit(' ', 1)[0] + "..."
                    frame_texts.append(text)
            
            if frame_texts:
                parts.append("\n\nFrom the knowledge base:\n• " + "\n• ".join(frame_texts))
        
        return '\n'.join(parts) if parts else "I don't have enough information to respond."


def test_output_chain():
    """Test the output gear chain."""
    print("=" * 70)
    print("OUTPUT GEAR CHAIN TEST")
    print("=" * 70)
    
    chain = OutputGearChain()
    
    # Learn from corpus
    corpus_path = Path(__file__).parent.parent / "truthspace_lcm/gears/corpus/corpus_llm_live.json"
    if corpus_path.exists():
        chain.learn_from_corpus(str(corpus_path))
    
    # Also learn from knowledge corpus
    knowledge_path = Path(__file__).parent.parent / "truthspace_lcm/gears/corpus/corpus_knowledge.json"
    if knowledge_path.exists():
        chain.learn_from_corpus(str(knowledge_path))
    
    # Test description generation
    print("\n--- Description Test ---")
    content = {
        'type': 'description',
        'agents': ['holmes'],
        'properties': {
            'holmes': {
                'similar_to': ['watson', 'detective_work', 'sherlock_holmes'],
                'opposite_of': 'villain',
                'dimensions': {
                    'Dim1': {'pole': 'analytical'},
                    'Dim2': {'pole': 'methodical'},
                }
            }
        },
        'frames': [
            {'text': 'Holmes analyzes evidence with remarkable precision and deductive skill.'},
        ]
    }
    print(chain.condition_output(content))
    
    # Test comparison generation
    print("\n--- Comparison Test ---")
    content = {
        'type': 'comparison',
        'agents': ['hero', 'villain'],
        'properties': {
            'distance': 1.2,
            'differences': [
                {'dimension': 'Morality', 'pole1': 'good', 'pole2': 'evil'}
            ]
        },
        'frames': []
    }
    print(chain.condition_output(content))
    
    # Test list generation
    print("\n--- List Test ---")
    content = {
        'type': 'list',
        'agents': ['watson', 'moriarty', 'adler'],
        'properties': {
            'context': 'Characters similar to Holmes',
            'items': ['watson', 'moriarty', 'adler']
        },
        'frames': []
    }
    print(chain.condition_output(content))
    
    return chain


if __name__ == "__main__":
    chain = test_output_chain()
