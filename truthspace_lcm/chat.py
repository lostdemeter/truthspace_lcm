#!/usr/bin/env python3
"""
TruthSpace Geometric Chat System

An interactive chat demonstration of the Geometric LCM approach.

Features:
- Semantic Q&A using geometric similarity
- Style extraction and classification
- Knowledge ingestion from text
- Interactive CLI with commands

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
    Vocabulary,
    KnowledgeBase,
    StyleEngine,
    Style,
    cosine_similarity,
    detect_question_type,
)


BOOTSTRAP_QA = [
    ("What is TruthSpace?", "TruthSpace is a geometric approach to language understanding where meaning is position in semantic space."),
    ("How does geometric encoding work?", "Text is encoded as IDF-weighted average of word positions, where each word has a deterministic position from its hash."),
    ("What is a style centroid?", "A style centroid is the average position of all exemplars of that style in semantic space."),
    ("How does style transfer work?", "Style transfer interpolates content toward the target style centroid: styled = (1-a)*content + a*centroid."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors, ranging from -1 to 1."),
    ("What is gap-filling?", "Gap-filling is the Q&A principle where questions define gaps in semantic space and answers fill those gaps."),
    ("What are the question types?", "Question types are WHO, WHAT, WHERE, WHEN, WHY, and HOW - each projects onto a different semantic axis."),
    ("What is IDF weighting?", "IDF weighting gives rare words higher weight so meaningful words contribute more than common ones."),
    ("How is this different from LLMs?", "Unlike neural LLMs, this uses pure geometry - no training, no learned weights, deterministic and interpretable."),
    ("What is the dimensionality?", "The default semantic space is 64-dimensional, though this is configurable."),
]

BOOTSTRAP_STYLES = {
    'formal': [
        "The implementation demonstrates significant improvements in accuracy.",
        "One must consider the theoretical implications of this approach.",
        "The methodology employed herein follows established protocols.",
    ],
    'casual': [
        "Hey, this thing actually works pretty well!",
        "So basically, it just averages the word positions.",
        "Cool, right? No neural networks needed.",
    ],
    'technical': [
        "The centroid is computed as mean of encoded exemplars.",
        "Cosine similarity returns values in the range negative one to one.",
        "Hash-based positioning ensures deterministic word vectors.",
    ],
}


class GeometricChat:
    """Interactive chat interface demonstrating the Geometric LCM."""
    
    def __init__(self):
        self.vocab = Vocabulary(dim=64)
        self.kb = KnowledgeBase(self.vocab)
        self.style_engine = StyleEngine(self.vocab)
        self.current_style = None
        self.debug_mode = False
        self._bootstrap()
    
    def _bootstrap(self):
        print("Initializing Geometric Chat System...")
        for q, a in BOOTSTRAP_QA:
            self.kb.add_qa_pair(q, a, source="bootstrap")
        print(f"  Loaded {len(BOOTSTRAP_QA)} Q&A pairs")
        
        for name, exemplars in BOOTSTRAP_STYLES.items():
            self.style_engine.extract_style(exemplars, name)
        print(f"  Extracted {len(BOOTSTRAP_STYLES)} styles: {list(BOOTSTRAP_STYLES.keys())}")
        print("Ready!\n")
    
    def query(self, question: str) -> Tuple[str, float, Dict]:
        debug_info = {}
        qtype = detect_question_type(question)
        debug_info['question_type'] = qtype
        
        qa_results = self.kb.search_qa(question, k=3)
        
        if qa_results:
            best_qa, best_sim = qa_results[0]
            debug_info['best_match'] = best_qa.question
            debug_info['similarity'] = best_sim
            
            if best_sim > 0.5:
                answer = best_qa.answer
                confidence = best_sim
            elif best_sim > 0.3:
                answer = f"I think: {best_qa.answer}"
                confidence = best_sim
            else:
                answer = "I don't have enough information to answer that question."
                confidence = 0.0
        else:
            answer = "I don't have any knowledge to search."
            confidence = 0.0
        
        if self.current_style and confidence > 0.3:
            styled_vec, style_words = self.style_engine.transfer(answer, self.current_style, strength=0.3)
            debug_info['style_applied'] = self.current_style
            debug_info['style_words'] = style_words[:5]
        
        return answer, confidence, debug_info
    
    def analyze_style(self, text: str) -> List[Tuple[str, float]]:
        return self.style_engine.classify(text)
    
    def process_command(self, cmd: str, args: str) -> str:
        if cmd == '/help':
            return """
Commands:
  /help          Show this help
  /style <name>  Set style (formal, casual, technical)
  /style list    List styles
  /style off     Turn off style
  /analyze <txt> Analyze text style
  /debug on|off  Toggle debug mode
  /stats         Show statistics
  /quit          Exit

Just type questions to query the knowledge base!
"""
        elif cmd == '/style':
            if args == 'list':
                return f"Available styles: {', '.join(self.style_engine.styles.keys())}"
            elif args == 'off':
                self.current_style = None
                return "Style turned off."
            elif args in self.style_engine.styles:
                self.current_style = args
                return f"Style set to: {args}"
            else:
                return f"Unknown style: {args}"
        elif cmd == '/analyze':
            if not args:
                return "Usage: /analyze <text>"
            results = self.analyze_style(args)
            lines = ["Style analysis:"]
            for name, score in results[:3]:
                lines.append(f"  {name}: {score:.3f}")
            return "\n".join(lines)
        elif cmd == '/debug':
            if args == 'on':
                self.debug_mode = True
                return "Debug mode enabled."
            elif args == 'off':
                self.debug_mode = False
                return "Debug mode disabled."
            return f"Debug is {'on' if self.debug_mode else 'off'}"
        elif cmd == '/stats':
            return f"Vocab: {len(self.vocab.word_positions)} words, Q&A: {len(self.kb.qa_pairs)}, Styles: {len(self.style_engine.styles)}"
        elif cmd == '/quit':
            return "QUIT"
        return f"Unknown command: {cmd}"
    
    def process(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""
        
        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return self.process_command(cmd, args)
        
        answer, confidence, debug_info = self.query(user_input)
        response = answer
        
        if self.debug_mode:
            response += f"\n[DEBUG: type={debug_info.get('question_type')}, sim={debug_info.get('similarity', 0):.3f}]"
        
        return response
    
    def run(self):
        print("=" * 60)
        print("  TRUTHSPACE GEOMETRIC CHAT SYSTEM")
        print("  All semantic operations are geometric operations")
        print("=" * 60)
        print("\nType /help for commands, /quit to exit.\n")
        
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
    print("=" * 60)
    print("  GEOMETRIC CHAT SYSTEM - Demo")
    print("=" * 60)
    
    chat = GeometricChat()
    
    print("\n--- Q&A Demo ---\n")
    questions = [
        "What is TruthSpace?",
        "How does style transfer work?",
        "What is cosine similarity?",
    ]
    for q in questions:
        answer, conf, _ = chat.query(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"   (confidence: {conf:.2f})\n")
    
    print("\n--- Style Analysis Demo ---\n")
    texts = [
        "The methodology demonstrates significant improvements.",
        "Hey, this is pretty cool stuff!",
        "Centroid equals mean of encoded exemplars.",
    ]
    for text in texts:
        results = chat.analyze_style(text)
        best_style, best_score = results[0]
        print(f"Text: \"{text[:40]}...\"")
        print(f"Style: {best_style} ({best_score:.3f})\n")
    
    print("--- Demo Complete ---\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        chat = GeometricChat()
        chat.run()
