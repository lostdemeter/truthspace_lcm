"""
Emergent Conversational Chain

A chain that builds knowledge through corpus building and generates
responses using ONLY emergent patterns - no LLM during conversation.

The key insight: LLM is used as a knowledge RESOURCE (like Wikipedia),
not as a response GENERATOR. All responses emerge from the learned structure.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
import re
import time
import requests
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from .semantic_chain import SemanticChain
from .gear_message import GearProtocol, GearMessage, MessageIntent

# Try to import tachyon ingestor for advanced frame extraction
try:
    import sys
    from pathlib import Path
    experiments_path = Path(__file__).parent.parent.parent.parent / 'experiments'
    if str(experiments_path) not in sys.path:
        sys.path.insert(0, str(experiments_path))
    from tachyon_symmetric_ingest import TachyonSymmetricIngestor, TachyonFrame
    HAS_TACHYON = True
except ImportError:
    HAS_TACHYON = False
    TachyonSymmetricIngestor = None
    TachyonFrame = None


# Common Project Gutenberg URLs
GUTENBERG_BOOKS = {
    'moby_dick': 'https://www.gutenberg.org/files/2701/2701-0.txt',
    'pride_and_prejudice': 'https://www.gutenberg.org/files/1342/1342-0.txt',
    'frankenstein': 'https://www.gutenberg.org/files/84/84-0.txt',
    'dracula': 'https://www.gutenberg.org/files/345/345-0.txt',
    'alice_in_wonderland': 'https://www.gutenberg.org/files/11/11-0.txt',
    'sherlock_holmes': 'https://www.gutenberg.org/files/1661/1661-0.txt',
    'war_and_peace': 'https://www.gutenberg.org/files/2600/2600-0.txt',
    'great_gatsby': 'https://www.gutenberg.org/files/64317/64317-0.txt',
    'jane_eyre': 'https://www.gutenberg.org/files/1260/1260-0.txt',
    'wuthering_heights': 'https://www.gutenberg.org/files/768/768-0.txt',
}


@dataclass
class KnowledgeItem:
    """A single piece of knowledge in the corpus."""
    text: str
    topic: str
    source: str
    item_type: str  # 'fact', 'definition', 'example', 'relation'
    timestamp: float = field(default_factory=time.time)


@dataclass
class RelationshipPattern:
    """A discovered relationship pattern between concepts."""
    subject: str
    predicate: str  # verb or relationship type
    object: str
    template: str  # e.g., "{subject} is a {object}"
    count: int = 1


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""
    user_input: str
    bot_response: str
    topics_used: List[str]
    timestamp: float = field(default_factory=time.time)


class ConversationalChain(GearProtocol):
    """
    Emergent Conversational Chain.
    
    Builds knowledge through corpus building (using LLM as resource),
    then generates responses using ONLY emergent patterns.
    
    Key principle: LLM is used ONLY during corpus building phase,
    NEVER during conversation. All responses are emergent.
    """
    
    def __init__(self):
        self.name = "ConversationalChain"
        
        # Semantic chain for emergent understanding
        self.semantic = SemanticChain()
        
        # Tachyon ingestor for frame extraction (if available)
        self.tachyon: Optional[Any] = None
        if HAS_TACHYON:
            self.tachyon = TachyonSymmetricIngestor()
        
        # Text processing gears (lazy import to avoid circular imports)
        self.stopword_gear: Optional[Any] = None
        self.gender_gear: Optional[Any] = None
        self.pronoun_gear: Optional[Any] = None
        self.thought_gear: Optional[Any] = None
        self.refinement_gear: Optional[Any] = None
        self.classifier_gear: Optional[Any] = None  # Emergent word classifier
        try:
            from truthspace_lcm.practical_applications.nlp.text_processing import (
                StopwordGear, GenderGear, PronounResolutionGear, ThoughtChainingGear
            )
            self.stopword_gear = StopwordGear()
            self.gender_gear = GenderGear()
            self.pronoun_gear = PronounResolutionGear()
            self.thought_gear = ThoughtChainingGear()
        except ImportError:
            pass  # Gears not available
        
        # Emergent classifier for word categories (replaces hardcoded lists)
        try:
            from truthspace_lcm.core.emergent_classifier import EmergentClassifierGear
            self.classifier_gear = EmergentClassifierGear()
        except ImportError:
            pass
        
        # Feedback refinement gear (optional, requires LLM)
        self.auto_refine = False  # Disabled by default
        try:
            from truthspace_lcm.practical_applications.nlp.feedback_refinement import (
                FeedbackRefinementGear
            )
            self.refinement_gear = FeedbackRefinementGear()
        except ImportError:
            pass
        
        # Shape-based chat improvement gear (automatic, no LLM needed)
        self.chat_improvement_gear: Optional[Any] = None
        self.auto_improve = True  # Enabled by default
        try:
            from .chat_improvement import ChatImprovementGear
            self.chat_improvement_gear = ChatImprovementGear()
        except ImportError:
            pass
        
        # Self-building corpus for social/system responses
        self.default_corpus: Optional[Any] = None
        try:
            from .corpus_builder import SelfBuildingCorpusGear
            self.default_corpus = SelfBuildingCorpusGear(auto_build=False)
        except ImportError:
            pass
        
        # Knowledge corpus
        self.corpus: List[KnowledgeItem] = []
        self.topics: Set[str] = set()
        self.topic_definitions: Dict[str, str] = {}
        
        # Entity relationships (from tachyon frames)
        self.entity_actions: Dict[str, Counter] = defaultdict(Counter)
        self.entity_targets: Dict[str, Counter] = defaultdict(Counter)
        self.entity_cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        
        # Conversation history
        self.history: List[ConversationTurn] = []
        
        # Response templates (discovered from data)
        self.response_templates: List[str] = []
        
        # Stats
        self.corpus_building_calls = 0
        self.conversation_calls = 0  # Should stay 0 for truly emergent
        
        # LLM configuration (for corpus building only)
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for corpus building and refinement."""
        self.llm_url = url
        self.llm_model = model
        
        # Also configure refinement gear if available
        if self.refinement_gear:
            self.refinement_gear.configure_llm(url, model)
    
    def enable_refinement(self, enabled: bool = True, threshold: float = 7.0):
        """Enable or disable automatic response refinement."""
        self.auto_refine = enabled
        if self.refinement_gear and threshold:
            self.refinement_gear.threshold = threshold
    
    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """
        Call LLM - ONLY for corpus building.
        
        This is like querying Wikipedia - we're gathering knowledge,
        not generating responses.
        """
        if not self.llm_url or not self.llm_model:
            return None
        
        try:
            import requests
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7}
                },
                timeout=60
            )
            if response.status_code == 200:
                self.corpus_building_calls += 1
                return response.json().get("response", "").strip()
        except Exception as e:
            pass
        return None
    
    # =========================================================================
    # CORPUS BUILDING (LLM used here as knowledge resource)
    # =========================================================================
    
    def add_knowledge(self, text: str, topic: str, item_type: str = 'fact', 
                      source: str = 'manual'):
        """Add a piece of knowledge to the corpus."""
        self.corpus.append(KnowledgeItem(
            text=text,
            topic=topic.lower(),
            source=source,
            item_type=item_type,
        ))
        self.topics.add(topic.lower())
        
        # Also ingest into semantic chain
        self.semantic.ingest_item({
            'text': text,
            'agent': topic.lower(),
            'source': source,
        })
    
    def learn_topic(self, topic: str) -> bool:
        """
        Learn about a topic using LLM as knowledge resource.
        
        Returns True if successful.
        """
        if not self.llm_url:
            return False
        
        topic_lower = topic.lower().strip()
        self.topics.add(topic_lower)
        
        # Get factual sentences
        prompt = f"""Write 5 simple factual sentences about "{topic}".
Each sentence should be self-contained and informative.
Write in third person, present tense.
One sentence per line.

Sentences about {topic}:"""

        response = self._call_llm(prompt, max_tokens=400)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 15 and len(line) < 250:
                    self.add_knowledge(line, topic_lower, 'fact', 'llm_knowledge')
        
        # Get definition
        prompt = f"""Define "{topic}" in one clear sentence.
Start with "{topic} is" or "{topic} refers to".

Definition:"""

        response = self._call_llm(prompt, max_tokens=100)
        if response:
            line = response.strip().split('\n')[0]
            if len(line) > 10:
                self.add_knowledge(line, topic_lower, 'definition', 'llm_knowledge')
                self.topic_definitions[topic_lower] = line
        
        # Get related topics
        prompt = f"""List 3 topics closely related to "{topic}".
Just the topic names, one per line:"""

        response = self._call_llm(prompt, max_tokens=50)
        if response:
            for line in response.split('\n'):
                line = line.strip().lstrip('0123456789.-) ').lower()
                if len(line) > 2 and len(line) < 40:
                    self.topics.add(line)
        
        return True
    
    def build_corpus(self, seed_topics: List[str], expand: bool = True):
        """Build knowledge corpus from seed topics."""
        # Learn seed topics
        for topic in seed_topics:
            self.learn_topic(topic)
        
        # Expand to related topics
        if expand:
            initial_topics = list(self.topics)
            for topic in initial_topics:
                if topic.lower() not in [t.lower() for t in seed_topics]:
                    self.learn_topic(topic)
        
        # Learn emergent structure
        self.semantic.learn_dimensions()
        
        # Discover response templates
        self._discover_templates()
    
    def _discover_templates(self):
        """Discover response templates from corpus patterns."""
        patterns = defaultdict(int)
        
        for item in self.corpus:
            text = item.text
            if item.topic in text.lower():
                pattern = text.lower().replace(item.topic, '{topic}')
                pattern = re.sub(r'\b(is|are|was|were)\b', '{be}', pattern)
                patterns[pattern[:50]] += 1
        
        for pattern, count in patterns.items():
            if count >= 2:
                self.response_templates.append(pattern)
        
        if not self.response_templates:
            self.response_templates = [
                "{topic} {be} {content}",
                "Regarding {topic}, {content}",
                "{content}",
            ]
    
    def load_corpus(self, path: str):
        """
        Load corpus from JSON file.
        
        This loads both the knowledge corpus and the default corpus
        (social/system responses) if they were saved together.
        """
        corpus_path = Path(path).resolve()
        self.corpus_path = str(corpus_path)
        
        with open(corpus_path, 'r') as f:
            data = json.load(f)
        
        # Load book title if present
        if 'book_title' in data:
            self.book_title = data['book_title']
        
        # Load knowledge items
        for item in data.get('items', []):
            self.add_knowledge(
                text=item.get('text', ''),
                topic=item.get('topic', 'unknown'),
                item_type=item.get('type', 'fact'),
                source=item.get('source', 'file'),
            )
        
        # Load default corpus if present
        if 'default_corpus' in data and self.default_corpus:
            self._load_default_corpus(data['default_corpus'])
        
        self.semantic.learn_dimensions()
        self._discover_templates()
    
    def _load_default_corpus(self, data: Dict):
        """Load the default corpus from saved data."""
        if not self.default_corpus:
            return
        
        # Import CorpusItem
        from .corpus_builder import CorpusItem
        
        # Clear existing items
        self.default_corpus.all_items = []
        for category in self.default_corpus.categories.values():
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
            self.default_corpus.all_items.append(item)
            if item.category in self.default_corpus.categories:
                self.default_corpus.categories[item.category].items.append(item)
        
        # Load build stats
        if 'build_stats' in data:
            self.default_corpus.build_stats = data['build_stats']
    
    def save_corpus(self, path: str, include_default_corpus: bool = True):
        """
        Save corpus to JSON file.
        
        Args:
            path: Path to save the corpus
            include_default_corpus: If True, also save the self-building default corpus
        """
        data = {
            'topics': list(self.topics),
            'definitions': self.topic_definitions,
            'book_title': getattr(self, 'book_title', None),
            'items': [
                {
                    'text': item.text,
                    'topic': item.topic,
                    'type': item.item_type,
                    'source': item.source,
                }
                for item in self.corpus
            ],
            'stats': {
                'corpus_building_calls': self.corpus_building_calls,
                'total_items': len(self.corpus),
            }
        }
        
        # Include default corpus if available
        if include_default_corpus and self.default_corpus:
            data['default_corpus'] = {
                'items': [
                    {
                        'text': item.text,
                        'category': item.category,
                        'subcategory': item.subcategory,
                        'quality_score': item.quality_score,
                        'use_count': item.use_count,
                        'success_count': item.success_count,
                    }
                    for item in self.default_corpus.all_items
                ],
                'build_stats': self.default_corpus.build_stats,
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # =========================================================================
    # LITERARY WORK LOADING
    # =========================================================================
    
    def load_book(self, book_name: str = None, url: str = None, 
                  max_lines: int = None, progress_callback=None) -> bool:
        """
        Load a literary work and build corpus from it.
        
        Args:
            book_name: Name from GUTENBERG_BOOKS (e.g., 'moby_dick')
            url: Direct URL to text file
            max_lines: Limit lines processed (None = all)
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            True if successful
        """
        # Get URL
        if book_name and book_name in GUTENBERG_BOOKS:
            url = GUTENBERG_BOOKS[book_name]
            self.book_title = book_name.replace('_', ' ').title()
        elif url:
            self.book_title = "Literary Work"
        else:
            return False
        
        # Fetch text
        try:
            if progress_callback:
                progress_callback(0, 100, f"Fetching {self.book_title}...")
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return False
            text = response.text
        except Exception as e:
            return False
        
        # Process the text
        return self.load_text(text, max_lines=max_lines, 
                             progress_callback=progress_callback)
    
    def load_text(self, text: str, max_lines: int = None,
                  progress_callback=None) -> bool:
        """
        Load raw text and build corpus from it.
        
        Extracts sentences, identifies characters/concepts, builds knowledge.
        """
        lines = text.split('\n')
        total_lines = len(lines) if max_lines is None else min(len(lines), max_lines)
        
        # Common stopwords to filter
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
            'here', 'there', 'then', 'once', 'if', 'because', 'until', 'while',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'any', 'said',
            'one', 'two', 'three', 'upon', 'like', 'even', 'still', 'yet',
        }
        
        # Track concept frequencies
        concept_counts = Counter()
        sentences_by_concept = defaultdict(list)
        
        processed = 0
        for i, line in enumerate(lines[:total_lines]):
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            # Extract sentences
            sentences = re.split(r'[.!?]+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15 or len(sentence) > 300:
                    continue
                
                # Extract potential concepts (capitalized words, proper nouns)
                words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                
                # Check if this is a narrative sentence (has verb-like words)
                is_narrative = any(w.endswith(('ed', 'ing', 's')) for w in sentence.lower().split())
                
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in stopwords and len(word_lower) > 2:
                        concept_counts[word_lower] += 1
                        # Store more sentences, prioritize narrative over titles
                        if len(sentences_by_concept[word_lower]) < 30:
                            # Insert narrative sentences at front, titles at back
                            if is_narrative and len(sentence) > 40:
                                sentences_by_concept[word_lower].insert(0, sentence)
                            else:
                                sentences_by_concept[word_lower].append(sentence)
            
            processed += 1
            if progress_callback and processed % 1000 == 0:
                pct = int(processed / total_lines * 100)
                progress_callback(processed, total_lines, 
                                 f"Processing line {processed}/{total_lines}")
        
        # Add top concepts to corpus (scale with book size)
        num_concepts = min(500, max(100, len(concept_counts) // 2))
        top_concepts = concept_counts.most_common(num_concepts)
        for concept, count in top_concepts:
            if count >= 2:  # Minimum frequency
                self.topics.add(concept)
                
                # Add sentences as knowledge (more sentences, narrative first)
                for sentence in sentences_by_concept[concept][:10]:
                    self.add_knowledge(
                        text=sentence,
                        topic=concept,
                        item_type='context',
                        source='book',
                    )
                
                # Create a summary definition from first mention
                if sentences_by_concept[concept]:
                    first_sentence = sentences_by_concept[concept][0]
                    self.topic_definitions[concept] = first_sentence
        
        # Learn emergent structure
        if progress_callback:
            progress_callback(total_lines, total_lines, "Learning structure...")
        
        self.semantic.learn_dimensions()
        self._discover_templates()
        
        # Extract frames using tachyon ingestor if available
        if self.tachyon and HAS_TACHYON:
            self._extract_frames_with_pronouns(text, max_lines)
        
        return True
    
    def _extract_frames_with_pronouns(self, text: str, max_lines: int = None):
        """
        Extract (actor, action, target) frames with pronoun resolution.
        
        Uses tachyon-symmetric ingestion to find frames, then resolves
        pronouns back to their antecedents using the PronounResolutionGear.
        
        Verb validation uses EmergentClassifierGear when available,
        falling back to morphological patterns.
        """
        if not self.tachyon:
            return
        
        # Reset pronoun gear if available
        if self.pronoun_gear:
            self.pronoun_gear.reset()
        
        # Train classifier on this text if available
        if self.classifier_gear:
            self.classifier_gear.learn_from_text(text, document_id="current")
        
        lines = text.split('\n')
        total_lines = len(lines) if max_lines is None else min(len(lines), max_lines)
        
        for line in lines[:total_lines]:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            # Skip chapter titles (all caps or very short with no punctuation)
            if line.isupper() or (len(line) < 50 and '.' not in line and ',' not in line):
                continue
            
            sentences = re.split(r'[.!?]+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                # Skip if looks like a title (no lowercase words)
                if not any(c.islower() for c in sentence):
                    continue
                
                # Extract frame using tachyon ingestor
                frame = self.tachyon.extract_frame(sentence)
                if not frame or not frame.actor:
                    continue
                
                actor = frame.actor.lower()
                action = frame.action.lower() if frame.action else None
                target = frame.target.lower() if frame.target else None
                
                # Validate action looks like a verb using emergent classifier or fallback
                if action:
                    if self.classifier_gear:
                        # Use emergent classification
                        is_verb = self.classifier_gear.is_verb(action)
                    else:
                        # Fallback to morphological patterns
                        is_verb = (
                            action.endswith('ed') or
                            action.endswith('ing') or
                            action.endswith('es') or
                            (action.endswith('s') and len(action) > 3)
                        )
                    if not is_verb:
                        action = None
                
                # Use pronoun gear for resolution if available
                if self.pronoun_gear:
                    actor_mention = self.pronoun_gear.process_entity(actor)
                    if actor_mention.is_pronoun and actor_mention.resolved_to:
                        actor = actor_mention.resolved_to
                    
                    if target:
                        target_mention = self.pronoun_gear.process_entity(target)
                        if target_mention.is_pronoun and target_mention.resolved_to:
                            target = target_mention.resolved_to
                
                # Store relationships
                if action:
                    self.entity_actions[actor][action] += 1
                    if target:
                        self.entity_targets[actor][target] += 1
                        self.entity_cooccurrence[actor][target] += 1
                        self.entity_cooccurrence[target][actor] += 1
    
    def get_entity_profile(self, entity: str) -> Dict[str, Any]:
        """
        Get a rich profile for an entity using discovered relationships.
        
        Returns actions they perform, entities they interact with, etc.
        """
        entity = entity.lower()
        
        actions = dict(self.entity_actions.get(entity, Counter()).most_common(5))
        targets = dict(self.entity_targets.get(entity, Counter()).most_common(5))
        cooccurs = dict(self.entity_cooccurrence.get(entity, Counter()).most_common(5))
        
        return {
            'entity': entity,
            'actions': actions,
            'targets': targets,
            'related_entities': cooccurs,
            'has_data': bool(actions or targets or cooccurs),
        }
    
    def get_available_books(self) -> List[str]:
        """Get list of available books from Gutenberg."""
        return list(GUTENBERG_BOOKS.keys())
    
    # =========================================================================
    # EMERGENT RESPONSE GENERATION (NO LLM!)
    # =========================================================================
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """
        Process message through emergent conversation.
        
        Implements GearProtocol.process_message.
        """
        response = self.chat(message.content)
        topics = self._extract_topics(message.content)
        
        return self.send(
            message.with_context('topics', topics),
            content=response,
            intent=MessageIntent.REPORT
        )
    
    def chat(self, user_input: str, refine: bool = None) -> str:
        """
        Generate response using ONLY emergent patterns.
        
        Args:
            user_input: User's message
            refine: Override auto_refine setting (None = use self.auto_refine)
        
        The emergent response is generated first, then optionally
        refined by LLM for grammar/clarity (not content).
        """
        # Extract topics from input
        topics = self._extract_topics(user_input)
        
        if not topics:
            return self._handle_unknown(user_input)
        
        # Generate emergent response
        response = self._generate_response(user_input, topics)
        
        # Optionally refine with LLM (for grammar/clarity only)
        should_refine = refine if refine is not None else self.auto_refine
        if should_refine and self.refinement_gear:
            main_topic = topics[0] if topics else ""
            result = self.refinement_gear.evaluate_and_refine(response, main_topic)
            response = result.refined
        
        # Shape-based auto-improvement (no LLM needed)
        if self.auto_improve and self.chat_improvement_gear:
            improvement = self.chat_improvement_gear.improve_response(user_input, response)
            if improvement.improvement_applied:
                response = improvement.improved
        
        # Store in history
        self.history.append(ConversationTurn(
            user_input=user_input,
            bot_response=response,
            topics_used=topics,
        ))
        
        return response
    
    def chat_with_details(self, user_input: str) -> Dict[str, Any]:
        """
        Chat and return detailed info including refinement details.
        """
        topics = self._extract_topics(user_input)
        
        if not topics:
            response = self._handle_unknown(user_input)
            return {
                'response': response,
                'topics': [],
                'refined': False,
            }
        
        # Generate emergent response
        original = self._generate_response(user_input, topics)
        main_topic = topics[0]
        
        # Refine if available
        result = None
        if self.refinement_gear:
            result = self.refinement_gear.evaluate_and_refine(original, main_topic)
        
        response = result.refined if result else original
        
        # Shape-based auto-improvement (no LLM needed)
        improvement = None
        if self.auto_improve and self.chat_improvement_gear:
            improvement = self.chat_improvement_gear.improve_response(user_input, response)
            if improvement.improvement_applied:
                response = improvement.improved
        
        # Store in history
        self.history.append(ConversationTurn(
            user_input=user_input,
            bot_response=response,
            topics_used=topics,
        ))
        
        return {
            'response': response,
            'original': original,
            'topics': topics,
            'refined': result is not None and result.refined != original,
            'score_before': result.score_before if result else None,
            'score_after': result.score_after if result else None,
            'feedback': result.feedback if result else None,
            # Shape-based improvement details
            'shape_improved': improvement.improvement_applied if improvement else False,
            'shape_similarity_before': improvement.shape_similarity_before if improvement else None,
            'shape_similarity_after': improvement.shape_similarity_after if improvement else None,
            'improvement_type': improvement.improvement_type if improvement else None,
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract known topics from text."""
        text_lower = text.lower()
        found = []
        
        for topic in self.topics:
            if topic in text_lower:
                found.append(topic)
        
        return sorted(found, key=len, reverse=True)
    
    def _generate_response(self, user_input: str, topics: List[str]) -> str:
        """Generate response using emergent patterns only."""
        main_topic = topics[0]
        
        # Get relevant content
        relevant = self._get_relevant_content(main_topic)
        
        # Get emergent traits
        traits = self.semantic.describe_traits(main_topic)
        
        # Get similar concepts
        similar = self.semantic.find_similar(main_topic, k=3)
        
        # Check if this is from a book
        is_book = getattr(self, 'book_title', None) is not None
        
        # Build response based on question type
        input_lower = user_input.lower()
        response_parts = []
        
        if 'what is' in input_lower or 'what are' in input_lower:
            # Definition question
            definition = self.topic_definitions.get(main_topic)
            if definition:
                response_parts.append(definition)
            if relevant and len(relevant) > 1:
                response_parts.append(f"\nMore context: {relevant[1]}")
        
        elif 'who is' in input_lower or 'tell me about' in input_lower:
            # Character/topic question - use entity profile for richer response
            profile = self.get_entity_profile(main_topic)
            
            if profile['has_data']:
                # Use thought chaining gear if available
                if self.thought_gear:
                    composed = self.thought_gear.compose_from_profile(main_topic, profile)
                    response_parts.append(composed)
                else:
                    # Fallback to manual composition
                    response_parts.append(f"{main_topic.title()}:")
                    
                    if profile['actions']:
                        actions_list = list(profile['actions'].keys())[:3]
                        action_str = ', '.join(actions_list)
                        response_parts.append(f"  Actions: {action_str}")
                    
                    if profile['related_entities']:
                        related = list(profile['related_entities'].keys())[:4]
                        related_str = ', '.join([r.title() for r in related])
                        response_parts.append(f"  Associated with: {related_str}")
                
                # Add context from excerpts
                if relevant:
                    response_parts.append(f"\nFrom the text:")
                    for excerpt in relevant[:2]:
                        response_parts.append(f"  • \"{excerpt}\"")
            else:
                # Fallback to excerpts only
                if relevant:
                    response_parts.append(f"From the text about {main_topic.title()}:")
                    for excerpt in relevant[:3]:
                        response_parts.append(f"  • \"{excerpt}\"")
            
            if similar:
                similar_str = ', '.join([s[0].title() for s in similar[:3]])
                response_parts.append(f"\nRelated: {similar_str}")
        
        elif 'how' in input_lower:
            # Explanation
            if relevant:
                response_parts.append(f"About {main_topic.title()}:")
                for i, fact in enumerate(relevant[:3], 1):
                    response_parts.append(f"  {i}. {fact}")
        
        elif 'why' in input_lower:
            # Reasoning
            causal = [r for r in relevant if any(w in r.lower() for w in 
                     ['because', 'since', 'therefore', 'important', 'significant'])]
            if causal:
                response_parts.append(causal[0])
            elif relevant:
                response_parts.append(relevant[0])
        
        else:
            # General question
            if relevant:
                response_parts.append(relevant[0])
            
            if traits:
                trait_str = ', '.join(traits[:3])
                response_parts.append(f"{main_topic.title()} is characterized by: {trait_str}.")
            
            if similar:
                similar_str = ', '.join([s[0] for s in similar[:3]])
                response_parts.append(f"Related concepts: {similar_str}.")
        
        if not response_parts:
            if relevant:
                return relevant[0]
            return f"I have limited knowledge about {main_topic}."
        
        return '\n'.join(response_parts)
    
    def _get_relevant_content(self, topic: str) -> List[str]:
        """Get relevant content from corpus for a topic."""
        relevant = []
        titles = []
        
        for item in self.corpus:
            if item.topic == topic.lower():
                text = item.text
                # Separate titles from narrative sentences
                # Titles are short, lack punctuation, or are all caps
                is_title = (
                    len(text) < 40 or
                    ('.' not in text and ',' not in text) or
                    text.isupper() or
                    not any(c.islower() for c in text)
                )
                if is_title:
                    titles.append(text)
                else:
                    relevant.append(text)
        
        # Prefer narrative sentences, fall back to titles
        if relevant:
            return relevant[:5]
        return titles[:5]
    
    def _handle_unknown(self, user_input: str) -> str:
        """Handle unknown queries using default corpus for social/system responses."""
        
        # First, check if the default corpus can handle this
        if self.default_corpus:
            category, response = self.default_corpus.match_intent(user_input)
            if response:
                return response
        
        # Try to find something related in knowledge corpus
        words = user_input.lower().split()
        
        for word in words:
            if len(word) > 3:
                for item in self.corpus:
                    if word in item.text.lower():
                        return f"I found something related: {item.text}"
        
        # Fall back to listing known topics
        known = sorted(list(self.topics))[:10]
        if known:
            return f"I don't have information about that. I can discuss: {', '.join(known)}"
        
        # Use default corpus acknowledgment if available
        if self.default_corpus:
            return self.default_corpus.get_acknowledgment()
        
        return "I don't have knowledge about that topic yet."
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        stats = {
            'topics': len(self.topics),
            'corpus_items': len(self.corpus),
            'definitions': len(self.topic_definitions),
            'dimensions': len(self.semantic.dimensions) if hasattr(self.semantic, 'dimensions') else 0,
            'corpus_building_calls': self.corpus_building_calls,
            'conversation_calls': self.conversation_calls,
            'history_length': len(self.history),
        }
        
        # Add optional fields if present
        if hasattr(self, 'corpus_path'):
            stats['corpus_path'] = self.corpus_path
        if hasattr(self, 'book_title'):
            stats['book_title'] = self.book_title
        
        return stats
    
    def list_topics(self) -> List[str]:
        """List all known topics."""
        return sorted(list(self.topics))
    
    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a specific topic."""
        topic_lower = topic.lower()
        
        facts = [item.text for item in self.corpus if item.topic == topic_lower]
        definition = self.topic_definitions.get(topic_lower)
        
        return {
            'topic': topic_lower,
            'definition': definition,
            'facts': facts,
            'fact_count': len(facts),
        }
