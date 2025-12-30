#!/usr/bin/env python3
"""
Self-Improving Loop for TruthSpace LCM

Kicks off a continuous self-improvement cycle where the chatbot:
1. Ingests new text from sources (Gutenberg, Wikipedia, etc.)
2. Uses the Curator to score and filter sentences
3. Adds high-quality frames to knowledge
4. Re-trains the curator on improved knowledge
5. Measures improvement and iterates

The key insight: As knowledge grows, the curator gets better at
identifying good sentences, which leads to better knowledge,
which leads to a better curator... (positive feedback loop)

Modes:
- batch: Run N improvement cycles and stop
- daemon: Run continuously in background
- interactive: Run with human feedback

Usage:
    python scripts/self_improve.py --cycles 5 --sources gutenberg
    python scripts/self_improve.py --daemon --interval 3600
    python scripts/self_improve.py --interactive

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import json
import time
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.geometric import GeometricQA, GeometricKnowledge
from truthspace_lcm.core.curator import CuratorLCM
from truthspace_lcm.core.geodesic_generator import GeodesicGenerator
from truthspace_lcm.core.structural_priority import (
    StructuralAnalyzer, 
    get_structural_priority,
    get_priority_topics,
    get_priority_keywords,
)


# Gutenberg book IDs for various domains
GUTENBERG_SOURCES = {
    'literature': [
        1342,   # Pride and Prejudice
        1661,   # Sherlock Holmes
        84,     # Frankenstein
        1952,   # The Yellow Wallpaper
        11,     # Alice in Wonderland
        74,     # Tom Sawyer
        1232,   # The Prince
        98,     # A Tale of Two Cities
        2701,   # Moby Dick
        345,    # Dracula
    ],
    'philosophy': [
        1497,   # Republic (Plato)
        4363,   # Beyond Good and Evil
        5827,   # The Problems of Philosophy
    ],
    'science': [
        2009,   # Origin of Species
        36,     # War of the Worlds
    ],
}

# Grokipedia topics for various domains
# Uses grokipedia-api.com (100 req/min, no API key needed)
GROKIPEDIA_TOPICS = {
    'literature': [
        'Sherlock_Holmes', 'Jane_Austen', 'Charles_Dickens', 'William_Shakespeare',
        'Pride_and_Prejudice', 'Frankenstein', 'Dracula', 'Moby_Dick',
        'Great_Expectations', 'Oliver_Twist', 'Hamlet', 'Macbeth',
        'Romeo_and_Juliet', 'The_Odyssey', 'Don_Quixote', 'War_and_Peace',
        'Crime_and_Punishment', 'The_Brothers_Karamazov', 'Anna_Karenina',
        'Les_Misérables', 'The_Count_of_Monte_Cristo', 'Wuthering_Heights',
        'Mark_Twain', 'Ernest_Hemingway', 'F._Scott_Fitzgerald', 'Virginia_Woolf',
        'James_Joyce', 'Franz_Kafka', 'Leo_Tolstoy', 'Fyodor_Dostoevsky',
        'Edgar_Allan_Poe', 'Oscar_Wilde', 'George_Orwell', 'Aldous_Huxley',
        'J._R._R._Tolkien', 'C._S._Lewis', 'Agatha_Christie', 'Arthur_Conan_Doyle',
        'Homer', 'Virgil', 'Dante_Alighieri', 'Geoffrey_Chaucer', 'John_Milton',
        'The_Great_Gatsby', 'To_Kill_a_Mockingbird', '1984_(novel)', 'Brave_New_World',
        'The_Catcher_in_the_Rye', 'Lord_of_the_Flies', 'One_Hundred_Years_of_Solitude',
    ],
    'science': [
        'Physics', 'Biology', 'Chemistry', 'Mathematics', 'Astronomy',
        'Artificial_intelligence', 'Machine_learning', 'Quantum_mechanics',
        'Theory_of_relativity', 'Evolution', 'Genetics', 'Neuroscience',
        'Computer_science', 'Statistics', 'Thermodynamics', 'Electromagnetism',
        'Organic_chemistry', 'Molecular_biology', 'Ecology', 'Geology',
        'Calculus', 'Linear_algebra', 'Number_theory', 'Topology',
        'Albert_Einstein', 'Isaac_Newton', 'Charles_Darwin', 'Marie_Curie',
        'Nikola_Tesla', 'Richard_Feynman', 'Stephen_Hawking', 'Carl_Sagan',
        'Niels_Bohr', 'Max_Planck', 'Werner_Heisenberg', 'Erwin_Schrödinger',
        'James_Clerk_Maxwell', 'Michael_Faraday', 'Galileo_Galilei', 'Johannes_Kepler',
        'DNA', 'RNA', 'Protein', 'Cell_(biology)', 'Photosynthesis', 'Mitosis',
        'Atom', 'Electron', 'Proton', 'Neutron', 'Quark', 'Higgs_boson',
        'Black_hole', 'Neutron_star', 'Galaxy', 'Solar_System', 'Milky_Way',
        'Climate_change', 'Renewable_energy', 'Nuclear_power', 'Biotechnology',
        'Nanotechnology', 'Robotics', 'Internet', 'World_Wide_Web', 'Algorithm',
    ],
    'history': [
        'World_War_I', 'World_War_II', 'Ancient_Rome', 'Renaissance',
        'Industrial_Revolution', 'French_Revolution', 'American_Revolution',
        'Ancient_Greece', 'Ancient_Egypt', 'Roman_Empire', 'Byzantine_Empire',
        'Ottoman_Empire', 'British_Empire', 'Cold_War', 'Medieval_Europe',
        'Enlightenment', 'Scientific_Revolution', 'Age_of_Discovery',
        'Julius_Caesar', 'Alexander_the_Great', 'Napoleon', 'Genghis_Khan',
        'Cleopatra', 'Queen_Victoria', 'Abraham_Lincoln', 'George_Washington',
        'Winston_Churchill', 'Franklin_D._Roosevelt', 'Adolf_Hitler', 'Joseph_Stalin',
        'Mahatma_Gandhi', 'Martin_Luther_King_Jr.', 'Nelson_Mandela',
        'American_Civil_War', 'Russian_Revolution', 'Chinese_Revolution',
        'Fall_of_the_Roman_Empire', 'Black_Death', 'Crusades', 'Reformation',
        'Colonialism', 'Imperialism', 'Decolonization', 'Globalization',
        'Ancient_Mesopotamia', 'Persian_Empire', 'Mongol_Empire', 'Ming_dynasty',
        'Qing_dynasty', 'Meiji_Restoration', 'Vietnam_War', 'Korean_War',
    ],
    'philosophy': [
        'Philosophy', 'Epistemology', 'Ethics', 'Logic', 'Metaphysics',
        'Aesthetics', 'Political_philosophy', 'Philosophy_of_mind',
        'Philosophy_of_science', 'Existentialism', 'Stoicism', 'Rationalism',
        'Empiricism', 'Utilitarianism', 'Kant', 'Plato', 'Aristotle',
        'Nietzsche', 'Descartes', 'Hume', 'Wittgenstein',
        'Socrates', 'John_Locke', 'Jean-Jacques_Rousseau', 'Thomas_Hobbes',
        'Karl_Marx', 'John_Stuart_Mill', 'Bertrand_Russell', 'Jean-Paul_Sartre',
        'Simone_de_Beauvoir', 'Hannah_Arendt', 'Michel_Foucault', 'Jacques_Derrida',
        'Phenomenology', 'Pragmatism', 'Analytic_philosophy', 'Continental_philosophy',
        'Free_will', 'Determinism', 'Consciousness', 'Personal_identity',
        'Philosophy_of_language', 'Philosophy_of_religion', 'Buddhist_philosophy',
        'Confucianism', 'Taoism', 'Hinduism', 'Islamic_philosophy',
    ],
    'technology': [
        'Computer', 'Smartphone', 'Artificial_neural_network', 'Deep_learning',
        'Natural_language_processing', 'Computer_vision', 'Cryptocurrency',
        'Blockchain', 'Cloud_computing', 'Cybersecurity', 'Data_science',
        'Big_data', 'Virtual_reality', 'Augmented_reality', 'Internet_of_things',
        'Programming_language', 'Python_(programming_language)', 'JavaScript',
        'Software_engineering', 'Database', 'Operating_system', 'Linux', 'Unix',
        'Apple_Inc.', 'Google', 'Microsoft', 'Amazon_(company)', 'Facebook',
        'Tesla,_Inc.', 'SpaceX', 'OpenAI', 'NVIDIA', 'Intel', 'IBM',
    ],
    'arts': [
        'Art', 'Music', 'Painting', 'Sculpture', 'Architecture',
        'Leonardo_da_Vinci', 'Michelangelo', 'Vincent_van_Gogh', 'Pablo_Picasso',
        'Claude_Monet', 'Rembrandt', 'Salvador_Dalí', 'Frida_Kahlo',
        'Wolfgang_Amadeus_Mozart', 'Ludwig_van_Beethoven', 'Johann_Sebastian_Bach',
        'The_Beatles', 'Elvis_Presley', 'Michael_Jackson', 'Bob_Dylan',
        'Classical_music', 'Jazz', 'Rock_music', 'Hip_hop', 'Electronic_music',
        'Film', 'Theatre', 'Dance', 'Photography', 'Graphic_design',
        'Renaissance_art', 'Impressionism', 'Cubism', 'Surrealism', 'Abstract_art',
    ],
    'geography': [
        'Earth', 'Continent', 'Ocean', 'Mountain', 'River', 'Desert', 'Forest',
        'Europe', 'Asia', 'Africa', 'North_America', 'South_America', 'Australia',
        'Antarctica', 'Pacific_Ocean', 'Atlantic_Ocean', 'Indian_Ocean',
        'United_States', 'China', 'India', 'Russia', 'Japan', 'Germany', 'France',
        'United_Kingdom', 'Brazil', 'Canada', 'Australia', 'Italy', 'Spain',
        'New_York_City', 'London', 'Paris', 'Tokyo', 'Beijing', 'Mumbai',
        'Amazon_rainforest', 'Sahara', 'Himalayas', 'Alps', 'Grand_Canyon',
    ],
}


# Directives file path - daemon reads this for instructions
DIRECTIVES_FILE = Path(__file__).parent / 'daemon_directives.json'


class SelfImprovementLoop:
    """
    Manages the self-improvement cycle for TruthSpace LCM.
    
    The loop:
    1. Load current knowledge
    2. Create curator from knowledge
    3. Fetch new text from sources
    4. Score and filter sentences
    5. Add good sentences to knowledge
    6. Save improved knowledge
    7. Measure improvement
    8. Repeat
    
    Directives:
    The daemon reads from daemon_directives.json for instructions:
    - priority_topics: List of topics to fetch first
    - priority_domains: Domains to focus on ('science', 'philosophy', etc.)
    - skip_topics: Topics to avoid
    - min_quality: Minimum curator score (0-1)
    """
    
    def __init__(self, corpus_path: str, log_path: str = None):
        """
        Initialize the self-improvement loop.
        
        Args:
            corpus_path: Path to the corpus JSON file
            log_path: Path to log improvements (optional)
        """
        self.corpus_path = Path(corpus_path)
        self.log_path = Path(log_path) if log_path else None
        
        # Load or create knowledge
        self.qa = GeometricQA()
        if self.corpus_path.exists():
            self.qa.load_corpus(str(self.corpus_path))
        
        # Create curator
        self.curator = CuratorLCM(self.qa.knowledge)
        
        # Structural analyzer for priority-based exploration
        self.structural_analyzer = StructuralAnalyzer()
        
        # Directives (loaded each cycle)
        self.directives = self._load_directives()
        
        # Track recently fetched topics to avoid duplicates
        self.recent_topics = set()
        
        # Dynamically discovered topics from article content
        self.discovered_topics = set()
        
        # Failed topics (404s, etc.) - don't retry these
        self.failed_topics = set()
        
        # Exhausted topics (high duplicate rate) - deprioritize these
        self.exhausted_topics = set()
        
        # Track ingested sentence hashes for deduplication
        self.ingested_hashes = set()
        self._load_ingested_hashes()
        
        # Metrics
        self.metrics = {
            'cycles': 0,
            'sentences_processed': 0,
            'sentences_accepted': 0,
            'sentences_rejected': 0,
            'frames_added': 0,
            'concepts_added': 0,
            'start_frames': len(self.qa.knowledge.frames),
            'start_concepts': len(self.qa.knowledge.concepts),
            'history': [],
        }
    
    def _load_ingested_hashes(self):
        """Load sentence hashes from existing corpus for deduplication."""
        if self.corpus_path.exists():
            try:
                with open(self.corpus_path) as f:
                    data = json.load(f)
                for frame in data.get('frames', []):
                    text = frame.get('text', '')
                    if text:
                        self.ingested_hashes.add(hash(text.lower().strip()))
            except Exception:
                pass
    
    def _is_duplicate(self, sentence: str) -> bool:
        """Check if sentence has already been ingested."""
        h = hash(sentence.lower().strip())
        if h in self.ingested_hashes:
            return True
        return False
    
    def _mark_ingested(self, sentence: str):
        """Mark sentence as ingested."""
        self.ingested_hashes.add(hash(sentence.lower().strip()))
    
    def _load_directives(self) -> Dict:
        """Load directives from file, or return defaults."""
        defaults = {
            'priority_topics': [],
            'priority_domains': [],
            'skip_topics': [],
            'min_quality': 0.5,
            'use_attention_extraction': True,
            'use_structural_priority': True,  # NEW: Use geometric gaps to guide exploration
        }
        
        if DIRECTIVES_FILE.exists():
            try:
                with open(DIRECTIVES_FILE) as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
            except Exception as e:
                pass  # Use defaults on error
        
        return defaults
    
    def _log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    def _topic_quality_score(self, topic: str) -> int:
        """
        Score a topic by likelihood of being a valid Grokipedia article.
        Higher score = more likely to be valid. Negative = skip entirely.
        """
        parts = topic.split('_')
        
        # REJECT patterns that are clearly invalid (return very negative score)
        # Topics ending with articles/prepositions are fragments
        bad_endings = ('The', 'A', 'An', 'Of', 'In', 'On', 'At', 'By', 'To', 'For', 'And', 'Or')
        if parts[-1] in bad_endings:
            return -100
        
        # Topics with repeated words are fragments ("Chemical_Kinetics_Chemical")
        if len(parts) != len(set(parts)):
            return -100
        
        # Topics starting with "The" followed by generic words
        if parts[0] == 'The' and len(parts) > 1 and parts[1].lower() in ('new', 'old', 'great', 'first', 'last'):
            return -100
        
        score = 0
        underscore_count = len(parts) - 1
        
        # Multi-word topics (with underscores) are more likely to be valid
        if underscore_count >= 2:
            score += 30  # "Albert_Einstein" style
        elif underscore_count == 1:
            score += 20  # "Quantum_mechanics" style
        
        # Longer topics tend to be more specific
        if len(topic) > 15:
            score += 10
        elif len(topic) > 10:
            score += 5
        
        # All parts capitalized = proper noun phrase (good)
        if len(parts) >= 2 and all(p[0].isupper() for p in parts if p):
            score += 10
        
        # Penalize likely fragments (generic suffixes)
        fragment_endings = ('ing', 'tion', 'ness', 'ment', 'ity', 'ous', 'ive')
        if topic.endswith(fragment_endings):
            score -= 10
        
        # Penalize generic single words
        if underscore_count == 0 and len(topic) < 10:
            score -= 15
        
        return score
    
    def _select_domain(self, available_domains: List[str]) -> str:
        """Select a domain based on directives or random."""
        priority = self.directives.get('priority_domains', [])
        
        # Check priority domains first
        for d in priority:
            if d in available_domains:
                return d
        
        # Random from available
        return random.choice(available_domains)
    
    def _fetch_gutenberg(self, book_id: int, max_sentences: int = 500) -> List[str]:
        """Fetch sentences from a Gutenberg book."""
        import urllib.request
        import re
        
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode('utf-8', errors='ignore')
        except:
            try:
                with urllib.request.urlopen(alt_url, timeout=30) as response:
                    text = response.read().decode('utf-8', errors='ignore')
            except Exception as e:
                self._log(f"Error fetching Gutenberg {book_id}: {e}")
                return []
        
        # Clean and split
        # Remove headers/footers
        if "*** START OF" in text:
            start = text.find("*** START OF")
            end_marker = text.find("***", start + 10)
            if end_marker > start:
                text = text[end_marker + 3:]
        
        if "*** END OF" in text:
            end = text.find("*** END OF")
            text = text[:end]
        
        # Split into sentences
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Mr\.', 'Mr', text)
        text = re.sub(r'Mrs\.', 'Mrs', text)
        text = re.sub(r'Dr\.', 'Dr', text)
        
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter for quality (basic)
        result = []
        for s in sentences:
            s = s.strip()
            words = s.split()
            if 6 <= len(words) <= 25 and s[0].isupper():
                result.append(s)
                if len(result) >= max_sentences:
                    break
        
        return result
    
    def _fetch_grokipedia(self, topic: str, max_sentences: int = 200) -> List[str]:
        """
        Fetch sentences from a Grokipedia article.
        
        Uses grokipedia-api.com (100 req/min, no API key needed).
        Also extracts linked topics for dynamic discovery.
        """
        import urllib.request
        import re
        
        # Grokipedia API endpoint
        url = f"https://grokipedia-api.com/page/{topic}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'TruthSpaceLCM/1.0 (self-improving corpus builder)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            self._log(f"Error fetching Grokipedia '{topic}': {e}")
            # Blacklist failed topics to avoid retrying
            self.failed_topics.add(topic)
            return []
        
        # Extract text from response (API returns 'content_text')
        text = data.get("content_text", "") or data.get("text", "") or data.get("content", "") or ""
        
        if not text:
            self._log(f"No text found for Grokipedia '{topic}' (keys: {list(data.keys())})")
            return []
        
        # Extract potential linked topics from content (dynamic discovery)
        self._extract_linked_topics(text)
        
        # Split into sentences
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers
        sentences = re.split(r'[.!?]+\s+', text)
        
        result = []
        for s in sentences:
            s = s.strip()
            words = s.split()
            if 6 <= len(words) <= 25 and s and s[0].isupper():
                result.append(s)
                if len(result) >= max_sentences:
                    break
        
        return result
    
    def _extract_linked_topics(self, text: str):
        """
        Extract potential topic names from article text for dynamic discovery.
        
        CONSERVATIVE: Only extract topics that look like real Wikipedia article names.
        Multi-word proper nouns with specific patterns.
        """
        import re
        
        # Skip common words/phrases that aren't topics
        skip_starts = {'The ', 'This ', 'That ', 'These ', 'Those ', 'There ', 'They ',
                       'What ', 'When ', 'Where ', 'Which ', 'While ', 'With ', 'From ',
                       'Into ', 'About ', 'After ', 'Before ', 'During ', 'Through ',
                       'However ', 'Although ', 'Because ', 'Since ', 'In ', 'On ', 'At ',
                       'By ', 'To ', 'For ', 'As ', 'If ', 'Or ', 'And ', 'But ', 'So ',
                       'Though ', 'Le ', 'La ', 'El ', 'Los ', 'Las ', 'Der ', 'Die ', 'Das '}
        
        # Normalize text
        text = re.sub(r'\s+', ' ', text)
        
        # Only extract 2-3 word proper noun phrases (most reliable)
        # Pattern: Two or three capitalized words (likely person names or place names)
        matches = re.findall(r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b', text)
        
        for m in matches:
            # Skip if starts with common non-topic words
            if any(m.startswith(s) for s in skip_starts):
                continue
            
            # Skip sentence fragments (contain common verbs/prepositions)
            bad_words = {'Was', 'Were', 'Has', 'Had', 'Have', 'Does', 'Did', 'Can', 'Could',
                        'Would', 'Should', 'May', 'Might', 'Must', 'Will', 'Shall',
                        'Being', 'Having', 'Using', 'Making', 'Taking', 'Getting'}
            words = m.split()
            if any(w in bad_words for w in words):
                continue
            
            # Must be 2-3 words (most reliable for article names)
            if len(words) < 2 or len(words) > 3:
                continue
            
            # Total length check (avoid very short or very long)
            if len(m) < 8 or len(m) > 40:
                continue
                
            topic = m.replace(' ', '_')
            
            # Add to discovered topics if not already known
            if topic not in self.recent_topics and topic not in self.discovered_topics:
                self.discovered_topics.add(topic)
    
    def run_cycle(self, source_type: str = 'gutenberg', 
                  min_score: float = 0.6,
                  max_sentences: int = 500) -> Dict:
        """
        Run one improvement cycle.
        
        Args:
            source_type: 'gutenberg' or 'wikipedia'
            min_score: Minimum curator score to accept
            max_sentences: Max sentences to process per source
        
        Returns:
            Metrics for this cycle
        """
        cycle_start = time.time()
        cycle_metrics = {
            'cycle': self.metrics['cycles'] + 1,
            'source_type': source_type,
            'sentences_fetched': 0,
            'sentences_accepted': 0,
            'sentences_rejected': 0,
            'frames_before': len(self.qa.knowledge.frames),
            'concepts_before': len(self.qa.knowledge.concepts),
        }
        
        # Reload directives each cycle (allows live updates)
        self.directives = self._load_directives()
        
        # Select source based on directives
        if source_type == 'gutenberg':
            domain = self._select_domain(list(GUTENBERG_SOURCES.keys()))
            book_id = random.choice(GUTENBERG_SOURCES[domain])
            self._log(f"Cycle {cycle_metrics['cycle']}: Fetching Gutenberg {book_id} ({domain})")
            sentences = self._fetch_gutenberg(book_id, max_sentences)
            source_name = f"Gutenberg-{book_id}"
        else:
            # Check for priority topics first
            topic = None
            domain = None
            
            if self.directives.get('priority_topics'):
                for t in self.directives['priority_topics']:
                    if t not in self.directives.get('skip_topics', []):
                        topic = t
                        domain = 'priority'
                        break
            
            # NEW: Check structural priority topics (geometric gap-based)
            if not topic and self.directives.get('use_structural_priority', True):
                structural_topics = self.structural_analyzer.suggest_priority_topics()
                for t in structural_topics:
                    if (t not in self.directives.get('skip_topics', []) and
                        t not in self.recent_topics and
                        t not in self.failed_topics and
                        t not in self.exhausted_topics):  # Skip exhausted topics
                        topic = t
                        domain = 'structural'
                        self.recent_topics.add(topic)  # Mark as used
                        break
            
            if not topic:
                # Mix predefined (reliable) with discovered (fresh but risky)
                # INCREASED: Use 60% discovered, 40% predefined to leverage the 9000+ discovered topics
                available = []
                use_discovered = random.random() < 0.6 and self.discovered_topics
                
                if use_discovered:
                    # Try discovered topics (fresh but may 404)
                    available = [t for t in self.discovered_topics
                                if t not in self.recent_topics
                                and t not in self.failed_topics
                                and t not in self.exhausted_topics][:100]  # Limit search
                    if available:
                        domain = 'discovered'
                
                # Use predefined topics (reliable)
                if not available:
                    domain = self._select_domain(list(GROKIPEDIA_TOPICS.keys()))
                    available = [t for t in GROKIPEDIA_TOPICS[domain] 
                                if t not in self.directives.get('skip_topics', [])
                                and t not in self.recent_topics
                                and t not in self.failed_topics
                                and t not in self.exhausted_topics]
                
                if not available:
                    # Reset recent topics if we've exhausted everything
                    self.recent_topics.clear()
                    # Try discovered first, then predefined
                    if self.discovered_topics:
                        domain = 'discovered'
                        available = [t for t in self.discovered_topics
                                    if t not in self.failed_topics]
                    if not available:
                        domain = self._select_domain(list(GROKIPEDIA_TOPICS.keys()))
                        available = [t for t in GROKIPEDIA_TOPICS[domain] 
                                    if t not in self.directives.get('skip_topics', [])
                                    and t not in self.failed_topics]
                
                if available:
                    # Pick best topic (first after sorting by quality) for discovered, random for predefined
                    if domain == 'discovered':
                        topic = available[0]  # Already sorted by quality score
                    else:
                        topic = random.choice(available)
                    self.recent_topics.add(topic)
                    # Remove from discovered if it was there
                    self.discovered_topics.discard(topic)
            
            if topic:
                discovered_count = len(self.discovered_topics)
                failed_count = len(self.failed_topics)
                # Add structural info if using structural priority
                if domain == 'structural':
                    priority_info = self.structural_analyzer.get_structural_priority()
                    self._log(f"Cycle {cycle_metrics['cycle']}: Fetching Grokipedia '{topic}' ({domain}: {priority_info.least_covered_axis}) [discovered: {discovered_count}, blacklisted: {failed_count}]")
                else:
                    self._log(f"Cycle {cycle_metrics['cycle']}: Fetching Grokipedia '{topic}' ({domain}) [discovered: {discovered_count}, blacklisted: {failed_count}]")
                sentences = self._fetch_grokipedia(topic, max_sentences)
                source_name = f"Grokipedia-{topic}"
            else:
                self._log(f"Cycle {cycle_metrics['cycle']}: No available topics")
                sentences = []
        
        cycle_metrics['sentences_fetched'] = len(sentences)
        
        if not sentences:
            self._log("  No sentences fetched, skipping cycle")
            return cycle_metrics
        
        # Score and filter with curator (with deduplication)
        accepted = []
        duplicates = 0
        total_sentences = len(sentences)
        for s in sentences:
            # Skip duplicates
            if self._is_duplicate(s):
                duplicates += 1
                continue
            
            score = self.curator.score_sentence(s)
            if score.overall >= min_score:
                accepted.append(s)
                self._mark_ingested(s)  # Mark as ingested
                cycle_metrics['sentences_accepted'] += 1
            else:
                cycle_metrics['sentences_rejected'] += 1
        
        if duplicates > 0:
            self._log(f"  Curator: {len(accepted)} accepted, {cycle_metrics['sentences_rejected']} rejected, {duplicates} duplicates skipped")
            # If >80% duplicates, mark this topic as exhausted
            if total_sentences > 0 and duplicates / total_sentences > 0.8:
                # Extract topic from source_name (e.g., "Grokipedia-Animacy" -> "Animacy")
                if source_name.startswith("Grokipedia-"):
                    exhausted_topic = source_name[11:]
                    self.exhausted_topics.add(exhausted_topic)
                    self._log(f"  Topic '{exhausted_topic}' exhausted (>80% duplicates), added to exhausted list ({len(self.exhausted_topics)} total)")
        else:
            self._log(f"  Curator: {len(accepted)} accepted, {cycle_metrics['sentences_rejected']} rejected")
        
        # Add accepted sentences to knowledge
        for s in accepted:
            self.qa.knowledge.learn(s, source_name)
        
        cycle_metrics['frames_after'] = len(self.qa.knowledge.frames)
        cycle_metrics['concepts_after'] = len(self.qa.knowledge.concepts)
        cycle_metrics['frames_added'] = cycle_metrics['frames_after'] - cycle_metrics['frames_before']
        cycle_metrics['concepts_added'] = cycle_metrics['concepts_after'] - cycle_metrics['concepts_before']
        
        self._log(f"  Added {cycle_metrics['frames_added']} frames, {cycle_metrics['concepts_added']} concepts")
        
        # Re-create curator with updated knowledge
        self.curator = CuratorLCM(self.qa.knowledge)
        
        # Update global metrics
        self.metrics['cycles'] += 1
        self.metrics['sentences_processed'] += len(sentences)
        self.metrics['sentences_accepted'] += cycle_metrics['sentences_accepted']
        self.metrics['sentences_rejected'] += cycle_metrics['sentences_rejected']
        self.metrics['frames_added'] += cycle_metrics['frames_added']
        self.metrics['concepts_added'] += cycle_metrics['concepts_added']
        
        cycle_metrics['duration'] = time.time() - cycle_start
        self.metrics['history'].append(cycle_metrics)
        
        return cycle_metrics
    
    def save_corpus(self):
        """Save the current knowledge to corpus file."""
        corpus = {
            "frames": [],
            "metadata": {
                "total_sentences": self.qa.knowledge.total_sentences,
                "total_concepts": len(self.qa.knowledge.concepts),
                "morphology_clusters": len(self.qa.knowledge.morphology.equivalence_classes),
                "improvement_cycles": self.metrics['cycles'],
                "last_updated": datetime.now().isoformat(),
            }
        }
        
        for frame in self.qa.knowledge.frames:
            corpus["frames"].append({
                "initiator": frame.initiator,
                "mediator": frame.mediator,
                "receiver": frame.receiver,
                "source": frame.source,
                "text": frame.text,
            })
        
        with open(self.corpus_path, 'w') as f:
            json.dump(corpus, f, indent=2)
        
        self._log(f"Saved {len(corpus['frames'])} frames to {self.corpus_path}")
    
    def run_structural_analysis(self) -> Dict:
        """
        Run structural analysis to find gaps and priorities.
        
        This is the self-referential improvement: the structure
        tells us what it's missing.
        """
        priority = self.structural_analyzer.get_structural_priority()
        
        analysis = {
            'least_covered_axis': priority.least_covered_axis,
            'axis_coverage': priority.axis_coverage,
            'missing_transforms': priority.missing_transforms,
            'num_gaps': len(priority.top_gaps),
            'suggested_topics': priority.suggested_topics[:5],
            'priority_keywords': priority.priority_keywords[:10],
        }
        
        self._log(f"  Structural analysis: least covered = {priority.least_covered_axis}, gaps = {len(priority.top_gaps)}")
        
        return analysis
    
    def evaluate(self) -> Dict:
        """
        Evaluate the current state of the system.
        
        Returns metrics on knowledge quality and coverage.
        """
        knowledge = self.qa.knowledge
        
        # Count content words
        content_words = [c for c in knowledge.concepts.values() if c.is_content_word]
        
        # Count words with clear roles
        clear_initiators = len([c for c in content_words if c.phi_direction > 0.3])
        clear_receivers = len([c for c in content_words if c.phi_direction < -0.3])
        
        # Test generation quality (simple heuristic)
        gen = GeodesicGenerator(knowledge)
        test_concepts = ['holmes', 'watson', 'darcy', 'elizabeth']
        generation_scores = []
        
        for concept in test_concepts:
            if concept in gen.nodes:
                text = gen.generate_about(concept, 3)
                # Simple quality heuristic: longer = better (up to a point)
                words = text.split()
                score = min(1.0, len(words) / 30)
                generation_scores.append(score)
        
        avg_generation = sum(generation_scores) / len(generation_scores) if generation_scores else 0
        
        # Test answer quality on benchmark
        benchmark_qa = [
            ("Who is Holmes?", ["detective", "sherlock", "investigator"]),
            ("What does Holmes do?", ["examine", "deduce", "investigate", "solve"]),
        ]
        
        correct = 0
        for question, keywords in benchmark_qa:
            try:
                from .geometric import HolographicGeometricQA
                hqa = HolographicGeometricQA()
                hqa.knowledge = knowledge
                answer = hqa.ask(question).lower()
                if any(kw in answer for kw in keywords):
                    correct += 1
            except:
                pass
        
        answer_accuracy = correct / len(benchmark_qa) if benchmark_qa else 0
        
        return {
            'total_frames': len(knowledge.frames),
            'total_concepts': len(knowledge.concepts),
            'content_words': len(content_words),
            'clear_initiators': clear_initiators,
            'clear_receivers': clear_receivers,
            'morphology_clusters': len(knowledge.morphology.equivalence_classes),
            'generation_quality': avg_generation,
            'answer_accuracy': answer_accuracy,
            'curator_learned_initiators': len(self.curator.learned_initiators),
            'curator_learned_mediators': len(self.curator.learned_mediators),
            'curator_learned_receivers': len(self.curator.learned_receivers),
        }
    
    def run_batch(self, num_cycles: int, source_type: str = 'gutenberg',
                  min_score: float = 0.6, save_every: int = 5):
        """
        Run multiple improvement cycles.
        
        Args:
            num_cycles: Number of cycles to run
            source_type: 'gutenberg', 'wikipedia', or 'mixed'
            min_score: Minimum curator score
            save_every: Save corpus every N cycles
        """
        self._log(f"Starting batch improvement: {num_cycles} cycles")
        self._log(f"Initial state: {len(self.qa.knowledge.frames)} frames, {len(self.qa.knowledge.concepts)} concepts")
        
        for i in range(num_cycles):
            # Alternate sources if mixed
            if source_type == 'mixed':
                src = 'gutenberg' if i % 2 == 0 else 'wikipedia'
            else:
                src = source_type
            
            self.run_cycle(src, min_score)
            
            # Save periodically
            if (i + 1) % save_every == 0:
                self.save_corpus()
        
        # Final save
        self.save_corpus()
        
        # Evaluate
        eval_metrics = self.evaluate()
        
        self._log("=" * 60)
        self._log("BATCH COMPLETE")
        self._log("=" * 60)
        self._log(f"Cycles completed: {self.metrics['cycles']}")
        self._log(f"Sentences processed: {self.metrics['sentences_processed']}")
        self._log(f"Sentences accepted: {self.metrics['sentences_accepted']} ({100*self.metrics['sentences_accepted']/max(1,self.metrics['sentences_processed']):.1f}%)")
        self._log(f"Frames: {self.metrics['start_frames']} → {eval_metrics['total_frames']} (+{self.metrics['frames_added']})")
        self._log(f"Concepts: {self.metrics['start_concepts']} → {eval_metrics['total_concepts']} (+{self.metrics['concepts_added']})")
        self._log(f"Generation quality: {eval_metrics['generation_quality']:.2f}")
    
    def run_daemon(self, interval: int = 3600, source_type: str = 'mixed', topics_per_cycle: int = 1):
        """
        Run as a daemon, continuously improving.
        
        Args:
            interval: Seconds between cycles
            source_type: 'gutenberg', 'grokipedia', or 'mixed'
            topics_per_cycle: Number of topics to fetch per cycle (for faster ingestion)
        """
        self._log(f"Starting daemon mode (interval={interval}s, sources={source_type}, topics_per_cycle={topics_per_cycle})")
        
        # Run initial structural analysis
        if self.directives.get('use_structural_priority', True):
            self._log("Running initial structural analysis...")
            self.run_structural_analysis()
        
        try:
            while True:
                # Run multiple topics per cycle for faster ingestion
                for _ in range(topics_per_cycle):
                    self.run_cycle(source_type)
                
                self.save_corpus()
                
                # Run structural analysis every 10 cycles
                if self.metrics['cycles'] % 10 == 0 and self.directives.get('use_structural_priority', True):
                    self._log("Periodic structural analysis...")
                    self.run_structural_analysis()
                
                self._log(f"Sleeping for {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            self._log("Daemon stopped by user")
            self.save_corpus()
    
    def run_interactive(self):
        """
        Run with human feedback.
        
        Shows sentences and asks for approval before adding.
        """
        self._log("Starting interactive mode")
        
        print("\nInteractive Self-Improvement")
        print("=" * 60)
        print("Commands:")
        print("  y/yes - Accept sentence")
        print("  n/no  - Reject sentence")
        print("  s/skip - Skip to next source")
        print("  q/quit - Save and exit")
        print()
        
        try:
            while True:
                # Fetch some sentences
                domain = random.choice(list(GUTENBERG_SOURCES.keys()))
                book_id = random.choice(GUTENBERG_SOURCES[domain])
                print(f"\nFetching from Gutenberg {book_id} ({domain})...")
                
                sentences = self._fetch_gutenberg(book_id, 50)
                
                if not sentences:
                    print("No sentences found, trying another source...")
                    continue
                
                for s in sentences:
                    # Score with curator
                    score = self.curator.score_sentence(s)
                    
                    print(f"\n[{score.overall:.2f}] {s}")
                    if score.issues:
                        print(f"Issues: {', '.join(score.issues[:2])}")
                    
                    response = input("Accept? (y/n/s/q): ").strip().lower()
                    
                    if response in ('q', 'quit'):
                        self.save_corpus()
                        print("Saved and exiting.")
                        return
                    elif response in ('s', 'skip'):
                        break
                    elif response in ('y', 'yes'):
                        self.qa.knowledge.learn(s, f"Gutenberg-{book_id}")
                        self.metrics['sentences_accepted'] += 1
                        print("  ✓ Added")
                    else:
                        self.metrics['sentences_rejected'] += 1
                        print("  ✗ Rejected")
                
                # Re-create curator
                self.curator = CuratorLCM(self.qa.knowledge)
                
        except KeyboardInterrupt:
            print("\nInterrupted")
            self.save_corpus()


def main():
    parser = argparse.ArgumentParser(description='Self-Improving Loop for TruthSpace LCM')
    
    # Mode
    parser.add_argument('--batch', action='store_true', help='Run batch mode')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--interactive', action='store_true', help='Run with human feedback')
    
    # Options
    parser.add_argument('--corpus', type=str, default='truthspace_lcm/corpus_self_improved.json',
                        help='Corpus file path')
    parser.add_argument('--cycles', type=int, default=5, help='Number of cycles (batch mode)')
    parser.add_argument('--interval', type=int, default=3600, help='Seconds between cycles (daemon mode)')
    parser.add_argument('--sources', type=str, default='mixed',
                        choices=['gutenberg', 'grokipedia', 'mixed'], help='Source type')
    parser.add_argument('--min-score', type=float, default=0.6, help='Minimum curator score')
    parser.add_argument('--topics-per-cycle', type=int, default=1, help='Topics to fetch per cycle (daemon mode)')
    parser.add_argument('--log', type=str, help='Log file path')
    parser.add_argument('--seed', type=str, help='Seed corpus to start from')
    
    args = parser.parse_args()
    
    # Use seed corpus if provided
    corpus_path = args.corpus
    if args.seed and Path(args.seed).exists():
        # Copy seed to corpus path
        import shutil
        shutil.copy(args.seed, corpus_path)
        print(f"Seeded from {args.seed}")
    
    # Create loop
    loop = SelfImprovementLoop(corpus_path, args.log)
    
    # Run appropriate mode
    if args.daemon:
        loop.run_daemon(args.interval, args.sources, args.topics_per_cycle)
    elif args.interactive:
        loop.run_interactive()
    else:
        # Default to batch
        loop.run_batch(args.cycles, args.sources, args.min_score)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
