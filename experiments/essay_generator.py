#!/usr/bin/env python3
"""
Essay Generator: Extended Sequence Generation

Generates longer, structured text (essays, papers) by:
1. Building an outline from related concepts
2. Generating paragraphs for each section
3. Using transitions to maintain coherence
4. Optionally using Qwen2 to polish the output

This extends the prattle() approach to essay-length generation.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA, GeometricKnowledge


@dataclass
class Section:
    """A section of the essay."""
    title: str
    concept: str
    content: List[str]  # List of sentences
    

@dataclass
class Outline:
    """Essay outline structure."""
    topic: str
    thesis: str
    sections: List[Section]


class EssayGenerator:
    """
    Generates essay-length text from the geometric corpus.
    
    Structure:
    - Introduction: Introduce topic, state thesis
    - Body: Multiple paragraphs exploring related concepts
    - Conclusion: Summarize and restate thesis
    """
    
    def __init__(self, corpus_path: str):
        self.qa = GeometricQA()
        self.qa.load_corpus(corpus_path)
        self.knowledge = self.qa.knowledge
        
        # Transition phrases for coherence
        self.transitions = {
            'addition': [
                "Furthermore, ", "Additionally, ", "Moreover, ",
                "In addition, ", "Also, ", "Beyond this, ",
            ],
            'contrast': [
                "However, ", "On the other hand, ", "In contrast, ",
                "Nevertheless, ", "Despite this, ", "Conversely, ",
            ],
            'example': [
                "For instance, ", "For example, ", "To illustrate, ",
                "Specifically, ", "In particular, ", "As an example, ",
            ],
            'cause': [
                "As a result, ", "Consequently, ", "Therefore, ",
                "Thus, ", "Hence, ", "Because of this, ",
            ],
            'sequence': [
                "First, ", "Second, ", "Next, ", "Then, ",
                "Finally, ", "Subsequently, ",
            ],
            'conclusion': [
                "In conclusion, ", "To summarize, ", "Overall, ",
                "In summary, ", "Ultimately, ", "To conclude, ",
            ],
        }
    
    def _get_concept_info(self, name: str) -> Dict:
        """Get structured info about a concept."""
        if name not in self.knowledge.concepts:
            return None
        
        c = self.knowledge.concepts[name]
        
        # Known good verbs
        good_verbs = {
            'studies', 'study', 'examines', 'examine', 'investigates', 'investigate',
            'explores', 'explore', 'analyzes', 'analyze', 'describes', 'describe',
            'explains', 'explain', 'discovers', 'discover', 'observes', 'observe',
            'measures', 'measure', 'tests', 'test', 'solves', 'solve',
            'deduces', 'deduce', 'reasons', 'reason', 'assists', 'assist',
            'helps', 'help', 'supports', 'support', 'documents', 'document',
            'provides', 'provide', 'includes', 'include', 'involves', 'involve',
            'creates', 'create', 'develops', 'develop', 'produces', 'produce',
            'transforms', 'transform', 'changes', 'change', 'adapts', 'adapt',
            'calculates', 'calculate', 'proves', 'prove', 'demonstrates', 'demonstrate',
            'powers', 'power', 'flows', 'flow', 'exists', 'exist', 'forms', 'form',
            'governs', 'govern', 'shapes', 'shape', 'influences', 'influence',
        }
        
        # Bad words to filter out
        bad_words = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'encompasse', 'biase', 'everyth', 'american', 'audio',
            'various', 'diverse', 'numerous', 'multiple', 'several',
        }
        
        # Get role
        role = "concept"
        category_words = {'detective', 'doctor', 'scientist', 'teacher', 'writer',
                         'science', 'field', 'discipline', 'study', 'branch',
                         'person', 'character', 'figure', 'process', 'phenomenon'}
        if c.targets:
            for target, count in c.targets.most_common(10):
                if target in category_words and count >= 2:
                    role = target
                    break
        
        # Get actions - only good verbs
        actions = []
        if c.actions:
            for action, _ in c.actions.most_common(10):
                action_lower = action.lower()
                if action_lower in good_verbs and action_lower not in bad_words:
                    actions.append(action)
                    if len(actions) >= 4:
                        break
        
        # Get targets/related concepts - filter noise
        targets = []
        if c.targets:
            for target, count in c.targets.most_common(20):
                if target in self.knowledge.concepts and len(target) > 3:
                    tc = self.knowledge.concepts[target]
                    # Must be a real content word with some attestation
                    if tc.is_content_word and target.lower() not in bad_words:
                        if tc.initiator_count + tc.mediator_count >= 3:
                            targets.append(target)
                            if len(targets) >= 6:
                                break
        
        return {
            'name': name,
            'role': role,
            'actions': actions[:4],
            'targets': targets[:6],
        }
    
    def _find_related_concepts(self, topic: str, depth: int = 2) -> List[str]:
        """Find concepts related to the topic for essay sections."""
        if topic not in self.knowledge.concepts:
            return []
        
        related = set()
        to_explore = [topic]
        explored = set()
        
        for _ in range(depth):
            next_explore = []
            for concept in to_explore:
                if concept in explored:
                    continue
                explored.add(concept)
                
                info = self._get_concept_info(concept)
                if info:
                    for target in info['targets']:
                        if target not in explored and target != topic:
                            related.add(target)
                            next_explore.append(target)
            to_explore = next_explore[:5]  # Limit breadth
        
        # Filter to content words with enough information
        good_related = []
        for r in related:
            if r in self.knowledge.concepts:
                c = self.knowledge.concepts[r]
                if c.is_content_word and c.actions:
                    good_related.append(r)
        
        return good_related[:6]  # Limit to 6 related concepts
    
    def _generate_thesis(self, topic: str, info: Dict) -> str:
        """Generate a thesis statement."""
        templates = [
            "{topic} is a {role} that plays a crucial role in understanding {target}.",
            "As a {role}, {topic} {action} various aspects of {target} and related phenomena.",
            "The study of {topic} reveals important insights about {target} and its implications.",
            "{topic}, fundamentally a {role}, {action} our understanding of {target}.",
        ]
        
        template = random.choice(templates)
        target = info['targets'][0] if info['targets'] else "its domain"
        action = info['actions'][0] if info['actions'] else "shapes"
        
        return template.format(
            topic=topic.title(),
            role=info['role'],
            action=action,
            target=target,
        )
    
    def _generate_introduction(self, topic: str, info: Dict, thesis: str) -> List[str]:
        """Generate introduction paragraph."""
        sentences = []
        
        # Hook/opening
        hooks = [
            f"{topic.title()} has long been a subject of significant interest and study.",
            f"Understanding {topic} is essential for comprehending the world around us.",
            f"The concept of {topic} encompasses a wide range of phenomena and ideas.",
            f"Few subjects are as fundamental to human knowledge as {topic}.",
        ]
        sentences.append(random.choice(hooks))
        
        # Background
        if info['actions']:
            action_list = ', '.join(info['actions'][:3])
            sentences.append(f"As a {info['role']}, {topic.title()} {action_list}.")
        
        # Thesis
        sentences.append(thesis)
        
        # Preview (if we have related concepts)
        if info['targets']:
            preview_targets = info['targets'][:3]
            sentences.append(
                f"This exploration will examine {topic}'s relationship to "
                f"{', '.join(preview_targets[:-1])} and {preview_targets[-1]}."
            )
        
        return sentences
    
    def _generate_body_paragraph(self, main_topic: str, subtopic: str, 
                                  paragraph_num: int) -> List[str]:
        """Generate a body paragraph about a subtopic."""
        sentences = []
        info = self._get_concept_info(subtopic)
        
        if not info:
            return []
        
        # Topic sentence with transition
        if paragraph_num == 0:
            transition = random.choice(self.transitions['sequence'][:2])  # First, Second
        elif paragraph_num < 3:
            transition = random.choice(self.transitions['addition'])
        else:
            transition = random.choice(self.transitions['example'])
        
        topic_templates = [
            "{trans}{subtopic} represents a key aspect of {main_topic}.",
            "{trans}the relationship between {main_topic} and {subtopic} is significant.",
            "{trans}{subtopic} plays an important role in understanding {main_topic}.",
            "{trans}examining {subtopic} provides insight into {main_topic}.",
        ]
        
        sentences.append(random.choice(topic_templates).format(
            trans=transition,
            subtopic=subtopic.title(),
            main_topic=main_topic.title(),
        ))
        
        # Supporting details
        if info['actions']:
            sentences.append(
                f"{subtopic.title()} {info['actions'][0]} "
                f"{'and ' + info['actions'][1] if len(info['actions']) > 1 else ''}."
            )
        
        if info['targets']:
            target = info['targets'][0]
            sentences.append(
                f"This involves {target}, which is closely connected to the broader topic."
            )
        
        # Analysis/explanation
        analysis_templates = [
            f"The significance of {subtopic} cannot be overstated in this context.",
            f"Understanding {subtopic} helps clarify the nature of {main_topic}.",
            f"This connection between {subtopic} and {main_topic} is fundamental.",
            f"Scholars have long recognized the importance of {subtopic} in this field.",
        ]
        sentences.append(random.choice(analysis_templates))
        
        return sentences
    
    def _generate_conclusion(self, topic: str, info: Dict, 
                             related_concepts: List[str]) -> List[str]:
        """Generate conclusion paragraph."""
        sentences = []
        
        # Transition to conclusion
        sentences.append(random.choice(self.transitions['conclusion']) + 
                        f"{topic.title()} remains a vital area of study and understanding.")
        
        # Restate thesis differently
        if info['actions']:
            sentences.append(
                f"Through its ability to {info['actions'][0]}, {topic.title()} "
                f"continues to shape our understanding of {info['targets'][0] if info['targets'] else 'the world'}."
            )
        
        # Summarize key points
        if related_concepts:
            concepts_mentioned = related_concepts[:3]
            sentences.append(
                f"The connections to {', '.join(concepts_mentioned)} demonstrate "
                f"the far-reaching implications of this subject."
            )
        
        # Closing thought
        closings = [
            f"Further study of {topic} will undoubtedly yield additional insights.",
            f"The importance of {topic} in human knowledge cannot be understated.",
            f"As our understanding grows, {topic} will continue to reveal new dimensions.",
            f"{topic.title()} thus stands as a cornerstone of intellectual inquiry.",
        ]
        sentences.append(random.choice(closings))
        
        return sentences
    
    def generate_essay(self, topic: str, paragraphs: int = 5) -> str:
        """
        Generate a full essay about a topic.
        
        Args:
            topic: The main topic to write about
            paragraphs: Approximate number of paragraphs (3-10)
        
        Returns:
            Essay text with introduction, body, and conclusion
        """
        # Normalize topic
        topic = topic.lower().strip()
        
        # Get topic info
        info = self._get_concept_info(topic)
        if not info:
            return f"I don't have enough information about {topic} to write an essay."
        
        # Find related concepts for body paragraphs
        related = self._find_related_concepts(topic, depth=2)
        
        # Generate thesis
        thesis = self._generate_thesis(topic, info)
        
        # Build essay
        essay_parts = []
        
        # Introduction
        intro = self._generate_introduction(topic, info, thesis)
        essay_parts.append(' '.join(intro))
        
        # Body paragraphs
        body_count = max(1, paragraphs - 2)  # Reserve 2 for intro/conclusion
        for i, subtopic in enumerate(related[:body_count]):
            body = self._generate_body_paragraph(topic, subtopic, i)
            if body:
                essay_parts.append(' '.join(body))
        
        # If not enough related concepts, add more about the main topic
        while len(essay_parts) < paragraphs - 1:
            extra = self._generate_extra_paragraph(topic, info, len(essay_parts))
            essay_parts.append(' '.join(extra))
        
        # Conclusion
        conclusion = self._generate_conclusion(topic, info, related)
        essay_parts.append(' '.join(conclusion))
        
        return '\n\n'.join(essay_parts)
    
    def _generate_extra_paragraph(self, topic: str, info: Dict, 
                                   paragraph_num: int) -> List[str]:
        """Generate an extra paragraph when we need more content."""
        sentences = []
        
        transition = random.choice(self.transitions['addition'])
        
        templates = [
            f"{transition}the broader implications of {topic.title()} extend into many areas.",
            f"{transition}{topic.title()} has influenced thinking across multiple disciplines.",
            f"{transition}the study of {topic} continues to evolve and expand.",
        ]
        sentences.append(random.choice(templates))
        
        if info['actions']:
            sentences.append(
                f"By {info['actions'][0]}ing, {topic.title()} provides a framework "
                f"for understanding complex phenomena."
            )
        
        sentences.append(
            f"This multifaceted nature of {topic} makes it particularly valuable "
            f"for interdisciplinary research."
        )
        
        sentences.append(
            f"Continued exploration of {topic} promises to yield further insights."
        )
        
        return sentences
    
    def generate_with_qwen2(self, topic: str, paragraphs: int = 5) -> str:
        """
        Generate essay and polish with Qwen2.
        
        This uses the geometric corpus for structure and content,
        then Qwen2 to make it more natural and coherent.
        """
        from experiments.ollama_corpus_refiner import OllamaClient
        
        # Generate base essay
        base_essay = self.generate_essay(topic, paragraphs)
        
        # Polish with Qwen2
        ollama = OllamaClient()
        if not ollama.is_available():
            return base_essay
        
        prompt = f"""Please rewrite this essay to make it more natural, coherent, and well-written while keeping the same structure and main points:

{base_essay}

Rewritten essay:"""
        
        polished = ollama.generate(prompt, temperature=0.4)
        
        return polished if polished else base_essay


def demo():
    """Demonstrate essay generation."""
    print("=" * 70)
    print("ESSAY GENERATOR DEMO")
    print("=" * 70)
    print()
    
    generator = EssayGenerator("truthspace_lcm/corpus_experimental.json")
    
    topics = ["physics", "evolution", "consciousness"]
    
    for topic in topics:
        print(f"\n{'='*70}")
        print(f"ESSAY: {topic.upper()}")
        print("=" * 70)
        print()
        
        essay = generator.generate_essay(topic, paragraphs=5)
        print(essay)
        print()


def demo_with_qwen2():
    """Demonstrate essay generation with Qwen2 polishing."""
    print("=" * 70)
    print("ESSAY GENERATOR WITH QWEN2 POLISHING")
    print("=" * 70)
    print()
    
    generator = EssayGenerator("truthspace_lcm/corpus_experimental.json")
    
    print("Generating essay about physics...")
    print()
    
    essay = generator.generate_with_qwen2("physics", paragraphs=5)
    print(essay)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Essay Generator")
    parser.add_argument("--topic", type=str, default="physics", help="Topic to write about")
    parser.add_argument("--paragraphs", type=int, default=5, help="Number of paragraphs")
    parser.add_argument("--polish", action="store_true", help="Polish with Qwen2")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.polish:
        demo_with_qwen2()
    else:
        generator = EssayGenerator("truthspace_lcm/corpus_experimental.json")
        essay = generator.generate_essay(args.topic, args.paragraphs)
        print(essay)
