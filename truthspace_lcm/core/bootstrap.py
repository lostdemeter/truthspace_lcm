#!/usr/bin/env python3
"""
Bootstrap Knowledge System for GeometricLCM

This module provides meta-knowledge about language structure that enables
the system to learn domain knowledge automatically. Instead of hard-coded
regex patterns, patterns are loaded from a JSON configuration file.

The bootstrap knowledge includes:
1. Word classes (question words, pronouns, verbs, etc.)
2. Sentence templates for relation extraction
3. Question patterns for query understanding
4. Entity type indicators

This is the foundation for a self-improving system where even these
patterns could eventually be learned geometrically.
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedFact:
    """A fact extracted from text."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source_template: str = ""


@dataclass
class ParsedQuestion:
    """A parsed question."""
    query_type: str  # describe_entity, relation_query, inverse_query
    entities: List[str]
    relation: Optional[str] = None
    focus: Optional[str] = None  # character, thing, place


class CoreferenceTracker:
    """
    Track entity references across sentences for pronoun resolution.
    
    This is a simple recency-based coreference resolver that tracks
    the most recent entities mentioned and resolves pronouns to them
    based on gender/number agreement.
    """
    
    def __init__(self):
        self.recent_entities: List[Dict[str, Any]] = []
        self.max_history = 10
        
        # Pronoun to gender/number mapping
        self.pronoun_info = {
            # Male singular
            "he": {"gender": "male", "number": "singular"},
            "him": {"gender": "male", "number": "singular"},
            "his": {"gender": "male", "number": "singular"},
            # Female singular
            "she": {"gender": "female", "number": "singular"},
            "her": {"gender": "female", "number": "singular"},
            "hers": {"gender": "female", "number": "singular"},
            # Neutral singular
            "it": {"gender": "neutral", "number": "singular"},
            "its": {"gender": "neutral", "number": "singular"},
            # Plural
            "they": {"gender": "any", "number": "plural"},
            "them": {"gender": "any", "number": "plural"},
            "their": {"gender": "any", "number": "plural"},
        }
        
        # Common male/female name endings (heuristic)
        # Note: These are rough heuristics and will have errors
        self.male_endings = ["us", "ck", "ld", "rd", "ph", "th"]
        self.female_endings = ["a", "ia", "na", "ne", "ie", "elle", "ette", "ina", "isa", "issa"]
        
        # Known names (override heuristics)
        self.known_male = {"darcy", "bingley", "wickham", "collins", "bennet", "ahab", "ishmael", 
                          "queequeg", "starbuck", "stubb", "flask", "pip", "victor", "henry",
                          "william", "robert", "ernest", "alphonse", "felix", "safie"}
        self.known_female = {"elizabeth", "jane", "lydia", "mary", "kitty", "caroline", "charlotte",
                            "alice", "margaret", "justine", "agatha"}
    
    def add_entity(self, name: str, context: str = "", entity_type: str = None):
        """Add an entity to the tracking history."""
        # Infer gender from name
        gender = self._infer_gender(name, context)
        
        entity_info = {
            "name": name.lower(),
            "gender": gender,
            "number": "singular",
            "type": entity_type,
            "context": context,
        }
        
        # Remove if already present (will re-add at front)
        self.recent_entities = [e for e in self.recent_entities if e["name"] != name.lower()]
        
        # Add to front
        self.recent_entities.insert(0, entity_info)
        
        # Trim history
        if len(self.recent_entities) > self.max_history:
            self.recent_entities = self.recent_entities[:self.max_history]
    
    def _infer_gender(self, name: str, context: str = "") -> str:
        """Infer gender from name or context."""
        name_lower = name.lower()
        context_lower = context.lower()
        
        # Check known names first
        if name_lower in self.known_male:
            return "male"
        if name_lower in self.known_female:
            return "female"
        
        # Check context for gender indicators
        if f"mr {name_lower}" in context_lower or f"mr. {name_lower}" in context_lower:
            return "male"
        if f"mrs {name_lower}" in context_lower or f"mrs. {name_lower}" in context_lower:
            return "female"
        if f"miss {name_lower}" in context_lower or f"ms {name_lower}" in context_lower:
            return "female"
        if f"sir {name_lower}" in context_lower or f"lord {name_lower}" in context_lower:
            return "male"
        if f"lady {name_lower}" in context_lower or f"queen {name_lower}" in context_lower:
            return "female"
        if f"king {name_lower}" in context_lower:
            return "male"
        
        # Check name endings (heuristic)
        for ending in self.female_endings:
            if name_lower.endswith(ending):
                return "female"
        for ending in self.male_endings:
            if name_lower.endswith(ending):
                return "male"
        
        return "unknown"
    
    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to the most likely entity."""
        pronoun_lower = pronoun.lower()
        
        if pronoun_lower not in self.pronoun_info:
            return None
        
        info = self.pronoun_info[pronoun_lower]
        target_gender = info["gender"]
        target_number = info["number"]
        
        # Find best matching entity
        for entity in self.recent_entities:
            # Check number
            if target_number == "plural" and entity["number"] != "plural":
                continue
            if target_number == "singular" and entity["number"] == "plural":
                continue
            
            # Check gender
            if target_gender == "any":
                return entity["name"]
            if target_gender == entity["gender"]:
                return entity["name"]
            if entity["gender"] == "unknown":
                return entity["name"]  # Best guess
        
        return None
    
    def clear(self):
        """Clear the tracking history."""
        self.recent_entities = []


class BootstrapKnowledge:
    """
    Loads and applies bootstrap knowledge for language understanding.
    
    This replaces hard-coded regex patterns with configurable patterns
    loaded from a JSON file.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize with bootstrap knowledge from JSON."""
        if config_path is None:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent / "bootstrap_knowledge.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Compile patterns for efficiency
        self._compiled_templates = {}
        self._compiled_questions = {}
        self._compile_patterns()
        
        # Coreference tracking
        self.coref_tracker = CoreferenceTracker()
        
        # Context window for multi-sentence extraction
        self.context_window: List[str] = []
        self.max_context = 5
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Bootstrap knowledge file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _compile_patterns(self):
        """Compile patterns from JSON into regex."""
        # Compile extraction patterns (new format with explicit regex)
        self._extraction_patterns = []
        for pattern_info in self.config.get("extraction_patterns", []):
            try:
                regex = re.compile(pattern_info["regex"])
                self._extraction_patterns.append({
                    "name": pattern_info.get("name", "unnamed"),
                    "regex": regex,
                    "groups": pattern_info.get("groups", []),
                    "fact": pattern_info.get("fact", []),
                    "description": pattern_info.get("description", ""),
                })
            except re.error as e:
                print(f"Warning: Invalid regex in pattern {pattern_info.get('name')}: {e}")
        
        # Compile question patterns (new v2 format)
        self._question_patterns = []
        for pattern_info in self.config.get("question_patterns_v2", []):
            try:
                regex = re.compile(pattern_info["regex"], re.IGNORECASE)
                self._question_patterns.append({
                    "name": pattern_info.get("name", "unnamed"),
                    "regex": regex,
                    "groups": pattern_info.get("groups", []),
                    "query_type": pattern_info.get("query_type"),
                    "focus": pattern_info.get("focus"),
                    "relation": pattern_info.get("relation"),
                    "relation_map": pattern_info.get("relation_map", {}),
                    "relations": pattern_info.get("relations", []),
                })
            except re.error as e:
                print(f"Warning: Invalid regex in question pattern {pattern_info.get('name')}: {e}")
        
        # Keep old format for backward compatibility
        for category, data in self.config.get("sentence_templates", {}).items():
            self._compiled_templates[category] = []
            for pattern_info in data.get("patterns", []):
                template = pattern_info["template"]
                regex = self._template_to_regex(template)
                self._compiled_templates[category].append({
                    "regex": regex,
                    "template": template,
                    "extract": pattern_info.get("extract", []),
                })
    
    def _template_to_regex(self, template: str) -> re.Pattern:
        """Convert a template like '[ENTITY] is a [TYPE]' to regex."""
        # Build regex piece by piece
        parts = []
        last_end = 0
        
        # Find all placeholders
        for match in re.finditer(r'\[(\w+)\]', template):
            # Add literal text before this placeholder
            literal = template[last_end:match.start()]
            if literal:
                # Escape and make whitespace flexible
                escaped_literal = re.escape(literal).replace(r'\ ', r'\s+')
                parts.append(escaped_literal)
            
            # Add capture group for placeholder
            placeholder_name = match.group(1)
            # Capture a word or multi-word name (capitalized words)
            # Use word boundaries and non-greedy matching
            parts.append(f"(?P<{placeholder_name}>[A-Z][a-z]+(?:\\s+[A-Z]?[a-z]+)*)")
            
            last_end = match.end()
        
        # Add any remaining literal text
        if last_end < len(template):
            literal = template[last_end:]
            escaped_literal = re.escape(literal).replace(r'\ ', r'\s+')
            parts.append(escaped_literal)
        
        pattern = ''.join(parts)
        return re.compile(pattern)
    
    # =========================================================================
    # WORD CLASS METHODS
    # =========================================================================
    
    def is_question_word(self, word: str) -> bool:
        """Check if word is a question word."""
        return word.lower() in self.config.get("word_classes", {}).get("question_words", [])
    
    def is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word."""
        return word.lower() in self.config.get("stop_words", [])
    
    def is_article(self, word: str) -> bool:
        """Check if word is an article."""
        return word.lower() in self.config.get("word_classes", {}).get("articles", [])
    
    def is_pronoun(self, word: str) -> bool:
        """Check if word is a pronoun."""
        pronouns = self.config.get("word_classes", {}).get("pronouns", {})
        all_pronouns = set()
        for category in pronouns.values():
            all_pronouns.update(category)
        return word.lower() in all_pronouns
    
    def is_role_indicator(self, word: str) -> bool:
        """Check if word indicates a role/title."""
        return word.lower() in self.config.get("word_classes", {}).get("role_indicators", [])
    
    def get_verb_category(self, word: str) -> Optional[str]:
        """Get the category of a verb (possession, creation, movement, etc.)."""
        common_verbs = self.config.get("word_classes", {}).get("common_verbs", {})
        word_lower = word.lower()
        for category, verbs in common_verbs.items():
            if word_lower in verbs:
                return category
        return None
    
    # =========================================================================
    # FACT EXTRACTION
    # =========================================================================
    
    def extract_facts(self, sentence: str, known_entities: Set[str] = None,
                       use_coreference: bool = True) -> List[ExtractedFact]:
        """Extract facts from a sentence using patterns."""
        facts = []
        sentence = sentence.strip()
        
        # Update context window
        self.context_window.append(sentence)
        if len(self.context_window) > self.max_context:
            self.context_window = self.context_window[-self.max_context:]
        
        # Resolve pronouns if enabled
        resolved_sentence = sentence
        if use_coreference:
            resolved_sentence = self._resolve_pronouns_in_sentence(sentence)
        
        # Use new explicit extraction patterns
        for pattern_info in self._extraction_patterns:
            regex = pattern_info["regex"]
            groups_spec = pattern_info["groups"]
            fact_spec = pattern_info["fact"]
            
            match = regex.search(resolved_sentence)
            if match:
                # Extract groups
                captured = {}
                for i, group_name in enumerate(groups_spec):
                    if i + 1 <= len(match.groups()):
                        captured[group_name] = match.group(i + 1)
                
                # Build fact from spec
                fact = self._build_fact_from_spec(captured, fact_spec, pattern_info["name"], known_entities)
                if fact:
                    facts.append(fact)
                    # Track entities for coreference
                    if use_coreference:
                        self.coref_tracker.add_entity(fact.subject, sentence)
                        if fact.object != "true":
                            self.coref_tracker.add_entity(fact.object, sentence)
        
        # Also track any capitalized names found (even without facts)
        if use_coreference:
            for match in re.finditer(r'\b([A-Z][a-z]+)\b', sentence):
                name = match.group(1)
                if name.lower() not in {'the', 'a', 'an', 'i', 'he', 'she', 'it', 'they'}:
                    self.coref_tracker.add_entity(name, sentence)
        
        return facts
    
    def _resolve_pronouns_in_sentence(self, sentence: str) -> str:
        """Replace pronouns with resolved entity names."""
        # Find pronouns at the start of sentences or after verbs
        pronouns = ['He', 'She', 'It', 'They', 'Him', 'Her', 'Them']
        
        resolved = sentence
        for pronoun in pronouns:
            # Only replace if at word boundary
            pattern = rf'\b{pronoun}\b'
            if re.search(pattern, resolved):
                entity = self.coref_tracker.resolve_pronoun(pronoun)
                if entity:
                    # Capitalize properly
                    replacement = entity.title()
                    resolved = re.sub(pattern, replacement, resolved, count=1)
        
        return resolved
    
    def clear_context(self):
        """Clear the context window and coreference tracker."""
        self.context_window = []
        self.coref_tracker.clear()
    
    def _build_fact_from_spec(self, captured: Dict[str, str], fact_spec: List[str], 
                               pattern_name: str, known_entities: Set[str] = None) -> Optional[ExtractedFact]:
        """Build a fact from captured groups and fact specification."""
        if len(fact_spec) < 3:
            return None
        
        def resolve(spec_item: str) -> str:
            """Resolve a spec item to actual value."""
            # Check for template like has_{role}
            if '{' in spec_item:
                for key, value in captured.items():
                    spec_item = spec_item.replace('{' + key + '}', value.lower())
                return spec_item
            # Direct reference to captured group
            if spec_item in captured:
                return self._normalize_entity(captured[spec_item])
            # Literal value
            return spec_item
        
        subject = resolve(fact_spec[0])
        relation = resolve(fact_spec[1])
        obj = resolve(fact_spec[2])
        
        if not subject or not relation or not obj:
            return None
        
        # Filter by known entities if provided
        if known_entities:
            subject_known = subject in known_entities
            obj_known = obj in known_entities or obj == "true"
            if not (subject_known or obj_known):
                return None
        
        return ExtractedFact(
            subject=subject,
            relation=relation,
            object=obj,
            source_template=pattern_name
        )
    
    def _extract_fact_from_match(self, groups: Dict[str, str], extract_spec: List[str],
                                  template: str, known_entities: Set[str] = None) -> Optional[ExtractedFact]:
        """Extract a fact from regex match groups."""
        if len(extract_spec) < 3:
            return None
        
        # Map extract spec to actual values
        # extract_spec is like ["subject", "is_a", "object"] or ["entity2", "has_role", "entity", "role"]
        
        subject = None
        relation = None
        obj = None
        
        for i, spec in enumerate(extract_spec):
            if spec == "subject":
                subject = self._normalize_entity(groups.get("ENTITY", groups.get("ENTITY1", "")))
            elif spec == "object":
                obj = self._normalize_entity(groups.get("TYPE", groups.get("ENTITY2", groups.get("PLACE", groups.get("WORK", "")))))
            elif spec == "entity":
                subject = self._normalize_entity(groups.get("ENTITY", ""))
            elif spec == "entity2":
                subject = self._normalize_entity(groups.get("ENTITY2", ""))
            elif spec in ["is_a", "located_in", "located_at", "lives_in", "went_to", "sailed_to",
                         "wrote", "created", "built", "founded", "killed", "attacked", "struck",
                         "chased", "associated_with", "with", "speaks", "has_role", "role"]:
                relation = spec
            elif spec == "role_word":
                obj = self._normalize_entity(groups.get("ROLE", ""))
            elif spec == "role":
                # For role extraction, the role word becomes the object
                obj = self._normalize_entity(groups.get("ROLE", ""))
            elif spec == "true":
                obj = "true"
        
        # Handle special cases
        if relation == "has_role" and "ROLE" in groups:
            relation = f"has_{groups['ROLE'].lower()}"
        
        if subject and relation and obj:
            # Filter by known entities if provided
            if known_entities:
                subject_known = subject in known_entities or any(e in subject for e in known_entities)
                obj_known = obj in known_entities or any(e in obj for e in known_entities)
                if not (subject_known or obj_known):
                    return None
            
            return ExtractedFact(
                subject=subject,
                relation=relation,
                object=obj,
                source_template=template
            )
        
        return None
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity name."""
        if not text:
            return ""
        
        # Lowercase and replace spaces with underscores
        text = text.lower().strip()
        
        # Remove articles at the start
        for article in ["the ", "a ", "an "]:
            if text.startswith(article):
                text = text[len(article):]
        
        # Replace spaces with underscores
        text = text.replace(" ", "_")
        
        return text
    
    # =========================================================================
    # QUESTION PARSING
    # =========================================================================
    
    def parse_question(self, question: str) -> Optional[ParsedQuestion]:
        """Parse a question to determine query type and entities."""
        question_lower = question.lower().strip()
        
        # Use new v2 question patterns
        for pattern_info in self._question_patterns:
            regex = pattern_info["regex"]
            match = regex.search(question_lower)
            
            if match:
                groups_spec = pattern_info["groups"]
                entities = []
                relation = pattern_info.get("relation")
                
                # Extract entities from match groups
                for i, group_name in enumerate(groups_spec):
                    if i + 1 <= len(match.groups()):
                        value = match.group(i + 1)
                        if value:
                            if group_name == "relation":
                                # Map relation if needed
                                relation_map = pattern_info.get("relation_map", {})
                                relation = relation_map.get(value.lower(), value.lower())
                            else:
                                entities.append(self._normalize_entity(value))
                
                return ParsedQuestion(
                    query_type=pattern_info["query_type"],
                    entities=entities,
                    relation=relation,
                    focus=pattern_info.get("focus")
                )
        
        # Fallback: try to find any entity mentioned
        words = question_lower.split()
        entities = []
        for word in words:
            word = word.strip('?!.,')
            if not self.is_stop_word(word) and not self.is_question_word(word) and len(word) > 2:
                entities.append(self._normalize_entity(word))
        
        if entities:
            return ParsedQuestion(
                query_type="describe_entity",
                entities=entities[:3]  # Limit to first 3
            )
        
        return None
    
    # =========================================================================
    # ENTITY DETECTION
    # =========================================================================
    
    def detect_entity_type(self, word: str, context: str = "") -> Optional[str]:
        """Detect the type of an entity based on indicators."""
        word_lower = word.lower()
        context_lower = context.lower()
        
        indicators = self.config.get("entity_type_indicators", {})
        
        # Check for person indicators
        person_indicators = indicators.get("person", {})
        for title in person_indicators.get("title_words", []):
            if title in context_lower and word_lower in context_lower:
                return "person"
        
        for verb in person_indicators.get("context_verbs", []):
            if f"{word_lower} {verb}" in context_lower:
                return "person"
        
        # Check for place indicators
        place_indicators = indicators.get("place", {})
        for prep in place_indicators.get("prepositions", []):
            if f"{prep} {word_lower}" in context_lower:
                return "place"
        
        for suffix in place_indicators.get("suffix_words", []):
            if word_lower.endswith(suffix):
                return "place"
        
        return None
    
    # =========================================================================
    # RELATION PROPERTIES
    # =========================================================================
    
    def get_inverse_relation(self, relation: str) -> Optional[str]:
        """Get the inverse of a relation."""
        props = self.config.get("relation_properties", {}).get(relation, {})
        return props.get("inverse")
    
    def is_transitive(self, relation: str) -> bool:
        """Check if a relation is transitive."""
        props = self.config.get("relation_properties", {}).get(relation, {})
        return props.get("transitive", False)
    
    def is_symmetric(self, relation: str) -> bool:
        """Check if a relation is symmetric."""
        props = self.config.get("relation_properties", {}).get(relation, {})
        return props.get("symmetric", False)
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_all_relations(self) -> List[str]:
        """Get all known relation types."""
        relations = set()
        for category, data in self.config.get("sentence_templates", {}).items():
            for pattern_info in data.get("patterns", []):
                extract = pattern_info.get("extract", [])
                for item in extract:
                    if item not in ["subject", "object", "entity", "entity2", "role_word", "role", "true"]:
                        relations.add(item)
        return list(relations)
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        self._compiled_templates = {}
        self._compiled_questions = {}
        self._compile_patterns()


# Singleton instance
_bootstrap_instance = None


def get_bootstrap_knowledge(config_path: str = None) -> BootstrapKnowledge:
    """Get the singleton bootstrap knowledge instance."""
    global _bootstrap_instance
    if _bootstrap_instance is None or config_path is not None:
        _bootstrap_instance = BootstrapKnowledge(config_path)
    return _bootstrap_instance
