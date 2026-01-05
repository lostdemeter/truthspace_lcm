"""
Self-Building Corpus Gear

A gear that automatically builds and refines a corpus for chat interactions.
Uses shape-based deficiency detection to identify gaps and improve coverage.

The corpus includes:
1. Social interactions (greetings, farewells, acknowledgments)
2. System metadata (what the system is, how it works)
3. Sentence structure templates (patterns for good responses)
4. Tone guidelines (helpful, friendly, professional)
5. AI assistant identity (what it can/cannot do)

The gear continuously iterates, using the improvement loop to:
- Detect gaps in coverage
- Generate new content to fill gaps
- Refine existing content for better shape
- Learn from successful interactions

Author: Lesley Gushurst
License: GPLv3
"""

import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path
import threading

from truthspace_lcm.core.utils.folding_deficiency import (
    FoldingStructure, FoldingDeficiencyDetector,
    ShapeDeficiency, ShapeDeficiencyType
)
from truthspace_lcm.core.legacy_gears.gears.chat_improvement_gear import ChatImprovementGear


@dataclass
class CorpusItem:
    """A single item in the corpus."""
    text: str
    category: str  # social, system, structure, tone, identity
    subcategory: str  # greeting, farewell, capability, etc.
    quality_score: float = 1.0
    use_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.use_count, 1)


@dataclass
class CorpusCategory:
    """A category of corpus items with its shape template."""
    name: str
    description: str
    template_text: str
    structure: FoldingStructure = field(default=None)
    items: List[CorpusItem] = field(default_factory=list)
    
    def __post_init__(self):
        if self.structure is None:
            self.structure = FoldingStructure.from_text(self.template_text)


class SelfBuildingCorpusGear:
    """
    A gear that automatically builds and maintains a chat corpus.
    
    Key features:
    1. Starts with seed content for each category
    2. Uses shape-based detection to identify gaps
    3. Generates new content to fill gaps
    4. Learns from successful interactions
    5. Continuously refines for better coverage
    """
    
    def __init__(self, auto_build: bool = True):
        self.name = "SelfBuildingCorpusGear"
        
        # Shape-based tools
        self.detector = FoldingDeficiencyDetector()
        self.improvement_gear = ChatImprovementGear()
        
        # Corpus storage
        self.categories: Dict[str, CorpusCategory] = {}
        self.all_items: List[CorpusItem] = []
        
        # Build statistics
        self.build_stats = {
            'iterations': 0,
            'items_added': 0,
            'items_refined': 0,
            'gaps_filled': 0,
            'last_build': None,
        }
        
        # Auto-build settings
        self.auto_build = auto_build
        self.build_interval = 300  # 5 minutes
        self._build_thread: Optional[threading.Thread] = None
        self._stop_building = False
        
        # Initialize with seed content
        self._initialize_categories()
        self._seed_corpus()
    
    def _initialize_categories(self):
        """Initialize corpus categories with templates."""
        
        # Social interactions
        self.categories['social_greeting'] = CorpusCategory(
            name='social_greeting',
            description='Greetings and conversation starters',
            template_text=(
                "Hello! I'm here to help. "
                "Hello, how can I assist you today? "
                "I'm ready to help with your questions."
            )
        )
        
        self.categories['social_farewell'] = CorpusCategory(
            name='social_farewell',
            description='Farewells and conversation closers',
            template_text=(
                "Goodbye! Feel free to return anytime. "
                "Goodbye, I hope I was helpful. "
                "Take care, and don't hesitate to ask more questions."
            )
        )
        
        self.categories['social_acknowledgment'] = CorpusCategory(
            name='social_acknowledgment',
            description='Acknowledgments and confirmations',
            template_text=(
                "I understand. Let me help with that. "
                "I see what you mean. Here's what I can do. "
                "That makes sense. I'll do my best to assist."
            )
        )
        
        self.categories['social_clarification'] = CorpusCategory(
            name='social_clarification',
            description='Requests for clarification',
            template_text=(
                "Could you clarify what you mean? "
                "I want to make sure I understand correctly. "
                "Could you provide more details about that?"
            )
        )
        
        # System metadata
        self.categories['system_identity'] = CorpusCategory(
            name='system_identity',
            description='What the system is',
            template_text=(
                "I am an AI assistant. "
                "I use emergent patterns to generate responses. "
                "My responses come from learned knowledge, not real-time generation."
            )
        )
        
        self.categories['system_capabilities'] = CorpusCategory(
            name='system_capabilities',
            description='What the system can do',
            template_text=(
                "I can answer questions about topics I've learned. "
                "I can help with information and explanations. "
                "I can discuss topics from my knowledge base."
            )
        )
        
        self.categories['system_limitations'] = CorpusCategory(
            name='system_limitations',
            description='What the system cannot do',
            template_text=(
                "I cannot access the internet in real-time. "
                "I cannot perform actions outside of conversation. "
                "My knowledge is limited to what I've been taught."
            )
        )
        
        # Sentence structure templates
        self.categories['structure_definition'] = CorpusCategory(
            name='structure_definition',
            description='How to define things',
            template_text=(
                "X is a type of Y. X is known for Z. "
                "X refers to Y. This means Z. "
                "X can be described as Y. It involves Z."
            )
        )
        
        self.categories['structure_explanation'] = CorpusCategory(
            name='structure_explanation',
            description='How to explain things',
            template_text=(
                "This works by doing X. The process involves Y. "
                "First, X happens. Then, Y occurs. Finally, Z results. "
                "The reason is X. This leads to Y. Therefore, Z."
            )
        )
        
        self.categories['structure_comparison'] = CorpusCategory(
            name='structure_comparison',
            description='How to compare things',
            template_text=(
                "X is similar to Y in that both Z. "
                "Unlike X, Y does Z. However, they share W. "
                "X and Y are related. X focuses on Z, while Y focuses on W."
            )
        )
        
        # Tone guidelines
        self.categories['tone_helpful'] = CorpusCategory(
            name='tone_helpful',
            description='Helpful and supportive tone',
            template_text=(
                "I'd be happy to help with that. "
                "Let me find that information for you. "
                "Here's what I can tell you about that."
            )
        )
        
        self.categories['tone_professional'] = CorpusCategory(
            name='tone_professional',
            description='Professional and clear tone',
            template_text=(
                "Based on the available information, X is Y. "
                "The key points to consider are X, Y, and Z. "
                "In summary, X relates to Y through Z."
            )
        )
        
        self.categories['tone_friendly'] = CorpusCategory(
            name='tone_friendly',
            description='Friendly and approachable tone',
            template_text=(
                "Great question! Here's what I know. "
                "That's an interesting topic! Let me explain. "
                "I'm glad you asked! Here's the information."
            )
        )
        
        # AI assistant identity
        self.categories['identity_purpose'] = CorpusCategory(
            name='identity_purpose',
            description='Purpose of the assistant',
            template_text=(
                "My purpose is to help you find information. "
                "I'm designed to assist with questions and explanations. "
                "I exist to make information accessible and understandable."
            )
        )
        
        self.categories['identity_behavior'] = CorpusCategory(
            name='identity_behavior',
            description='How the assistant behaves',
            template_text=(
                "I aim to be helpful, accurate, and respectful. "
                "I try to provide clear and concise answers. "
                "I acknowledge when I don't know something."
            )
        )
        
        # Register templates with detector
        for name, category in self.categories.items():
            self.detector.learn_template(name, category.template_text)
    
    def _seed_corpus(self):
        """Seed the corpus with initial content."""
        
        # Social - Greetings
        self._add_item('social_greeting', 'greeting', 
            "Hello! How can I help you today?")
        self._add_item('social_greeting', 'greeting',
            "Hi there! I'm ready to assist you.")
        self._add_item('social_greeting', 'greeting',
            "Welcome! What would you like to know?")
        self._add_item('social_greeting', 'greeting',
            "Good to see you! How may I help?")
        
        # Social - Farewells
        self._add_item('social_farewell', 'farewell',
            "Goodbye! Feel free to come back anytime.")
        self._add_item('social_farewell', 'farewell',
            "Take care! I hope I was helpful.")
        self._add_item('social_farewell', 'farewell',
            "Until next time! Don't hesitate to ask more questions.")
        
        # Social - Acknowledgments
        self._add_item('social_acknowledgment', 'acknowledgment',
            "I understand. Let me help you with that.")
        self._add_item('social_acknowledgment', 'acknowledgment',
            "Got it. Here's what I can tell you.")
        self._add_item('social_acknowledgment', 'acknowledgment',
            "I see. Let me look into that for you.")
        
        # Social - Clarification
        self._add_item('social_clarification', 'clarification',
            "Could you tell me more about what you're looking for?")
        self._add_item('social_clarification', 'clarification',
            "I want to make sure I understand. Could you clarify?")
        
        # System - Identity
        self._add_item('system_identity', 'identity',
            "I am an AI assistant built on emergent patterns.")
        self._add_item('system_identity', 'identity',
            "I'm a conversational AI that learns from knowledge.")
        self._add_item('system_identity', 'identity',
            "I use geometric patterns to understand and respond.")
        
        # System - Capabilities
        self._add_item('system_capabilities', 'capability',
            "I can answer questions about topics in my knowledge base.")
        self._add_item('system_capabilities', 'capability',
            "I can explain concepts and provide information.")
        self._add_item('system_capabilities', 'capability',
            "I can discuss various subjects I've learned about.")
        
        # System - Limitations
        self._add_item('system_limitations', 'limitation',
            "I cannot browse the internet or access real-time information.")
        self._add_item('system_limitations', 'limitation',
            "I cannot perform actions outside of our conversation.")
        self._add_item('system_limitations', 'limitation',
            "My knowledge is limited to what I've been taught.")
        
        # Structure - Definitions
        self._add_item('structure_definition', 'definition',
            "X is a Y that Z. This means X is characterized by Z.")
        self._add_item('structure_definition', 'definition',
            "X refers to Y. X is important because Z.")
        
        # Structure - Explanations
        self._add_item('structure_explanation', 'explanation',
            "This works by X. First Y happens, then Z follows.")
        self._add_item('structure_explanation', 'explanation',
            "The process involves X. This leads to Y, resulting in Z.")
        
        # Tone - Helpful
        self._add_item('tone_helpful', 'helpful',
            "I'd be happy to help you with that question.")
        self._add_item('tone_helpful', 'helpful',
            "Let me find that information for you.")
        
        # Tone - Professional
        self._add_item('tone_professional', 'professional',
            "Based on the available information, here's what I can tell you.")
        self._add_item('tone_professional', 'professional',
            "The key points to consider are as follows.")
        
        # Tone - Friendly
        self._add_item('tone_friendly', 'friendly',
            "Great question! Here's what I know about that.")
        self._add_item('tone_friendly', 'friendly',
            "That's an interesting topic! Let me explain.")
        
        # Identity - Purpose
        self._add_item('identity_purpose', 'purpose',
            "My purpose is to help you find and understand information.")
        self._add_item('identity_purpose', 'purpose',
            "I'm here to assist with questions and provide explanations.")
        
        # Identity - Behavior
        self._add_item('identity_behavior', 'behavior',
            "I aim to be helpful, accurate, and respectful in my responses.")
        self._add_item('identity_behavior', 'behavior',
            "I try to provide clear answers and acknowledge uncertainty.")
        
        self.build_stats['items_added'] = len(self.all_items)
    
    def _add_item(self, category: str, subcategory: str, text: str, 
                  quality_score: float = 1.0) -> CorpusItem:
        """Add an item to the corpus."""
        item = CorpusItem(
            text=text,
            category=category,
            subcategory=subcategory,
            quality_score=quality_score
        )
        
        self.all_items.append(item)
        if category in self.categories:
            self.categories[category].items.append(item)
        
        return item
    
    def get_response(self, category: str, subcategory: str = None) -> Optional[str]:
        """Get a response from the corpus for a given category."""
        if category not in self.categories:
            return None
        
        items = self.categories[category].items
        if subcategory:
            items = [i for i in items if i.subcategory == subcategory]
        
        if not items:
            return None
        
        # Select based on quality and success rate
        best = max(items, key=lambda i: i.quality_score * (0.5 + 0.5 * i.success_rate))
        best.use_count += 1
        best.last_used = time.time()
        
        return best.text
    
    def record_success(self, text: str, was_successful: bool):
        """Record whether a response was successful."""
        for item in self.all_items:
            if item.text == text:
                if was_successful:
                    item.success_count += 1
                break
    
    def analyze_gaps(self) -> Dict[str, float]:
        """Analyze coverage gaps in the corpus."""
        gaps = {}
        
        for name, category in self.categories.items():
            if not category.items:
                gaps[name] = 1.0  # Complete gap
                continue
            
            # Check shape coverage
            avg_similarity = 0.0
            for item in category.items:
                item_struct = FoldingStructure.from_text(item.text)
                sim = category.structure.shape_similarity(item_struct)
                avg_similarity += sim
            
            avg_similarity /= len(category.items)
            
            # Gap is inverse of coverage
            gaps[name] = 1.0 - avg_similarity
        
        return gaps
    
    def build_iteration(self) -> Dict[str, Any]:
        """Run one iteration of corpus building."""
        self.build_stats['iterations'] += 1
        self.build_stats['last_build'] = time.time()
        
        results = {
            'iteration': self.build_stats['iterations'],
            'gaps_found': [],
            'items_added': 0,
            'items_refined': 0,
        }
        
        # Analyze gaps
        gaps = self.analyze_gaps()
        
        # Find categories with significant gaps
        for category_name, gap_score in gaps.items():
            if gap_score > 0.3:  # Significant gap
                results['gaps_found'].append({
                    'category': category_name,
                    'gap_score': gap_score
                })
                
                # Try to fill the gap
                filled = self._fill_gap(category_name, gap_score)
                if filled:
                    results['items_added'] += 1
                    self.build_stats['items_added'] += 1
                    self.build_stats['gaps_filled'] += 1
        
        # Refine low-quality items
        for item in self.all_items:
            if item.quality_score < 0.7 and item.use_count > 0:
                refined = self._refine_item(item)
                if refined:
                    results['items_refined'] += 1
                    self.build_stats['items_refined'] += 1
        
        return results
    
    def _fill_gap(self, category_name: str, gap_score: float) -> bool:
        """Generate content to fill a gap using pattern variations."""
        if category_name not in self.categories:
            return False
        
        category = self.categories[category_name]
        
        # Get existing texts to avoid duplicates
        existing_texts = {item.text.lower() for item in category.items}
        
        # Generate variations based on category type
        new_texts = self._generate_variations(category_name, category)
        
        added = False
        for text in new_texts:
            if text.lower() not in existing_texts:
                self._add_item(category_name, category_name.split('_')[-1], text, 
                              quality_score=0.8)
                added = True
                break  # Add one at a time
        
        return added
    
    def _generate_variations(self, category_name: str, category: CorpusCategory) -> List[str]:
        """Generate variations for a category."""
        variations = []
        
        # Variation patterns by category type
        if 'greeting' in category_name:
            variations = [
                "Hello there! What can I help you with?",
                "Hi! I'm here to assist you.",
                "Greetings! How may I be of service?",
                "Welcome back! What would you like to know?",
                "Hey! Ready to help whenever you are.",
                "Good day! How can I assist?",
            ]
        
        elif 'farewell' in category_name:
            variations = [
                "See you later! Come back anytime.",
                "Bye for now! Hope I was helpful.",
                "Until we meet again! Take care.",
                "Farewell! Don't hesitate to return.",
                "Goodbye for now! Best wishes.",
            ]
        
        elif 'acknowledgment' in category_name:
            variations = [
                "Understood. I'll help with that.",
                "I hear you. Let me see what I can do.",
                "Got it! Here's what I found.",
                "Alright, let me look into that.",
                "Sure thing! Here's the information.",
            ]
        
        elif 'clarification' in category_name:
            variations = [
                "Could you be more specific about that?",
                "I'd like to understand better. Can you elaborate?",
                "What exactly would you like to know?",
                "Could you rephrase that for me?",
                "I want to help - can you give more details?",
            ]
        
        elif 'identity' in category_name:
            variations = [
                "I'm an AI assistant designed to help with questions.",
                "I am a conversational AI built on geometric patterns.",
                "I'm here as your AI helper, using emergent responses.",
                "I am an assistant that learns from knowledge patterns.",
                "I'm a chat AI that generates responses from learned structure.",
            ]
        
        elif 'capabilities' in category_name:
            variations = [
                "I can help you find information on various topics.",
                "I'm able to answer questions from my knowledge base.",
                "I can explain concepts and discuss what I've learned.",
                "I can assist with questions about topics I know.",
                "I'm capable of providing information and explanations.",
            ]
        
        elif 'limitations' in category_name:
            variations = [
                "I can't access real-time information from the internet.",
                "I'm unable to perform actions outside our conversation.",
                "I don't have access to current events or live data.",
                "I cannot learn new information during our chat.",
                "My knowledge is fixed to what I was taught.",
            ]
        
        elif 'definition' in category_name:
            variations = [
                "This is defined as X. It means Y in context.",
                "X can be understood as Y. This involves Z.",
                "The term X refers to Y. It is characterized by Z.",
            ]
        
        elif 'explanation' in category_name:
            variations = [
                "Here's how it works: First X, then Y, finally Z.",
                "The process goes like this: X leads to Y, resulting in Z.",
                "To explain: X happens because Y, which causes Z.",
            ]
        
        elif 'helpful' in category_name:
            variations = [
                "I'm happy to assist you with that.",
                "Let me help you find what you need.",
                "I'll do my best to answer your question.",
                "Here to help! Let me look into that.",
            ]
        
        elif 'professional' in category_name:
            variations = [
                "According to the information available, X is Y.",
                "The relevant details are as follows: X, Y, Z.",
                "To summarize the key points: X relates to Y.",
            ]
        
        elif 'friendly' in category_name:
            variations = [
                "That's a great question! Here's what I know.",
                "Interesting topic! Let me share what I've learned.",
                "Good one! Here's the information you're looking for.",
            ]
        
        elif 'purpose' in category_name:
            variations = [
                "I exist to help you access information easily.",
                "My goal is to assist with your questions.",
                "I'm designed to make knowledge accessible.",
            ]
        
        elif 'behavior' in category_name:
            variations = [
                "I strive to be helpful and accurate.",
                "I aim to give clear, honest responses.",
                "I try to be respectful and informative.",
            ]
        
        return variations
    
    def _refine_item(self, item: CorpusItem) -> bool:
        """Refine a low-quality item."""
        if item.category not in self.categories:
            return False
        
        category = self.categories[item.category]
        
        # Check shape similarity
        item_struct = FoldingStructure.from_text(item.text)
        similarity = category.structure.shape_similarity(item_struct)
        
        if similarity > 0.8:
            # Already good shape
            item.quality_score = min(1.0, item.quality_score + 0.1)
            return False
        
        # Try to improve using the improvement gear
        # Use template as the "question" for shape matching
        result = self.improvement_gear.improve_response(
            category.template_text, item.text
        )
        
        if result.improvement_applied and result.shape_similarity_after > similarity:
            item.text = result.improved
            item.quality_score = min(1.0, item.quality_score + 0.2)
            return True
        
        return False
    
    def start_auto_build(self):
        """Start automatic corpus building in background."""
        if self._build_thread and self._build_thread.is_alive():
            return
        
        self._stop_building = False
        self._build_thread = threading.Thread(target=self._auto_build_loop, daemon=True)
        self._build_thread.start()
    
    def stop_auto_build(self):
        """Stop automatic corpus building."""
        self._stop_building = True
        if self._build_thread:
            self._build_thread.join(timeout=5)
    
    def _auto_build_loop(self):
        """Background loop for automatic building."""
        while not self._stop_building:
            try:
                self.build_iteration()
            except Exception as e:
                pass  # Log error but continue
            
            # Wait for next iteration
            for _ in range(self.build_interval):
                if self._stop_building:
                    break
                time.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        category_stats = {}
        for name, category in self.categories.items():
            category_stats[name] = {
                'items': len(category.items),
                'avg_quality': sum(i.quality_score for i in category.items) / max(len(category.items), 1),
                'total_uses': sum(i.use_count for i in category.items),
            }
        
        return {
            'total_items': len(self.all_items),
            'categories': len(self.categories),
            'category_stats': category_stats,
            'build_stats': self.build_stats,
        }
    
    def save(self, path: str):
        """Save corpus to file."""
        data = {
            'items': [
                {
                    'text': item.text,
                    'category': item.category,
                    'subcategory': item.subcategory,
                    'quality_score': item.quality_score,
                    'use_count': item.use_count,
                    'success_count': item.success_count,
                }
                for item in self.all_items
            ],
            'build_stats': self.build_stats,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load corpus from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Clear existing
        self.all_items = []
        for category in self.categories.values():
            category.items = []
        
        # Load items
        for item_data in data.get('items', []):
            item = CorpusItem(
                text=item_data['text'],
                category=item_data['category'],
                subcategory=item_data['subcategory'],
                quality_score=item_data.get('quality_score', 1.0),
                use_count=item_data.get('use_count', 0),
                success_count=item_data.get('success_count', 0),
            )
            self.all_items.append(item)
            if item.category in self.categories:
                self.categories[item.category].items.append(item)
        
        self.build_stats = data.get('build_stats', self.build_stats)
    
    def get_greeting(self) -> str:
        """Get a greeting response."""
        return self.get_response('social_greeting', 'greeting') or "Hello!"
    
    def get_farewell(self) -> str:
        """Get a farewell response."""
        return self.get_response('social_farewell', 'farewell') or "Goodbye!"
    
    def get_acknowledgment(self) -> str:
        """Get an acknowledgment response."""
        return self.get_response('social_acknowledgment', 'acknowledgment') or "I understand."
    
    def get_clarification(self) -> str:
        """Get a clarification request."""
        return self.get_response('social_clarification', 'clarification') or "Could you clarify?"
    
    def get_identity(self) -> str:
        """Get identity information."""
        return self.get_response('system_identity', 'identity') or "I am an AI assistant."
    
    def get_capabilities(self) -> str:
        """Get capabilities information."""
        return self.get_response('system_capabilities', 'capability') or "I can answer questions."
    
    def get_limitations(self) -> str:
        """Get limitations information."""
        return self.get_response('system_limitations', 'limitation') or "I have limitations."
    
    def match_intent(self, user_input: str) -> Tuple[str, str]:
        """
        Match user input to a corpus category.
        
        Returns (category, response) or (None, None) if no match.
        """
        input_lower = user_input.lower().strip()
        
        # Greeting patterns
        if any(g in input_lower for g in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return 'social_greeting', self.get_greeting()
        
        # Farewell patterns
        if any(f in input_lower for f in ['goodbye', 'bye', 'see you', 'take care', 'later']):
            return 'social_farewell', self.get_farewell()
        
        # Identity questions
        if any(i in input_lower for i in ['who are you', 'what are you', 'are you an ai', 'are you a bot']):
            return 'system_identity', self.get_identity()
        
        # Capability questions
        if any(c in input_lower for c in ['what can you do', 'how can you help', 'what do you know']):
            return 'system_capabilities', self.get_capabilities()
        
        # Limitation questions
        if any(l in input_lower for l in ['what can\'t you do', 'limitations', 'what don\'t you know']):
            return 'system_limitations', self.get_limitations()
        
        # Acknowledgment triggers (usually in response to user statements)
        if any(a in input_lower for a in ['i see', 'okay', 'alright', 'thanks', 'thank you']):
            return 'social_acknowledgment', self.get_acknowledgment()
        
        return None, None


def create_default_corpus() -> SelfBuildingCorpusGear:
    """Factory function to create a default corpus."""
    return SelfBuildingCorpusGear(auto_build=False)
