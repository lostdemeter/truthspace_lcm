"""
Intent Classifier

Classifies user queries into intent categories for routing:
- CODE_GENERATION: Generate and execute code (plots, scripts)
- TOOL_CALL: Call a Goose tool (file ops, bash, etc.)
- KNOWLEDGE: Answer a question
- CLARIFICATION: Need more info from user
- UNSUPPORTED: Can't handle this

Uses holographic pattern space for classification - positions are
constructed from word overlap similarity, so queries naturally
cluster with their intent category.

Author: Lesley Gushurst
License: GPLv3
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import re

from truthspace_lcm.gears.core.holographic_pattern_space import HolographicPatternSpace, HolographicModule


class Intent(Enum):
    """Intent categories for query routing."""
    CODE_GENERATION = "code"      # Generate and execute code
    TOOL_CALL = "tool"            # Call a Goose tool (file ops, bash, etc.)
    KNOWLEDGE = "knowledge"       # Answer a question
    CLARIFICATION = "clarify"     # Need more info from user
    UNSUPPORTED = "unsupported"   # Can't handle this


@dataclass
class IntentMatch:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reason: str
    tool_name: Optional[str] = None  # For TOOL_CALL: which tool
    tool_args: Dict[str, Any] = field(default_factory=dict)  # Extracted arguments


# Goose tool patterns - maps tool names to trigger words/phrases
GOOSE_TOOL_PATTERNS = {
    "Read": {
        "keywords": {"read", "show", "contents", "cat", "view", "display", "print", "open"},
        "patterns": [
            r"read\s+(?:the\s+)?(?:file\s+)?(.+)",
            r"show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?(.+)",
            r"cat\s+(.+)",
            r"view\s+(.+)",
            r"what(?:'s| is)\s+in\s+(.+)",
        ],
        "arg_name": "file_path",
    },
    "Write": {
        "keywords": {"write", "save", "create", "file"},
        "patterns": [
            r"write\s+(?:to\s+)?(.+)",
            r"save\s+(?:to\s+)?(.+)",
            r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+)?(.+)",
        ],
        "arg_name": "file_path",
    },
    "Glob": {
        "keywords": {"list", "files", "find", "directory", "folder", "ls"},
        "patterns": [
            r"list\s+(?:all\s+)?(?:the\s+)?(?:files?\s+)?(?:in\s+)?(.+)?",
            r"find\s+(?:all\s+)?(?:files?\s+)?(?:in\s+)?(.+)?",
            r"what(?:'s| is)\s+in\s+(?:the\s+)?(?:directory|folder)\s*(.+)?",
            r"ls\s*(.+)?",
            r"show\s+(?:me\s+)?(?:the\s+)?files?\s+(?:in\s+)?(.+)?",
        ],
        "arg_name": "pattern",
    },
    "Grep": {
        "keywords": {"search", "find", "grep", "look", "text"},
        "patterns": [
            r"search\s+(?:for\s+)?['\"]?(.+?)['\"]?\s+in\s+(.+)",
            r"grep\s+(?:for\s+)?['\"]?(.+?)['\"]?\s+(?:in\s+)?(.+)?",
            r"find\s+['\"]?(.+?)['\"]?\s+in\s+(?:the\s+)?(?:files?|code)",
        ],
        "arg_name": "query",
    },
    "Bash": {
        "keywords": {"run", "execute", "command", "shell", "terminal", "bash"},
        "patterns": [
            r"run\s+(.+)",
            r"execute\s+(.+)",
            r"(?:in\s+)?(?:the\s+)?(?:terminal|shell|bash)\s*[,:]\s*(.+)",
        ],
        "arg_name": "command",
    },
    "Edit": {
        "keywords": {"edit", "modify", "change", "update", "replace"},
        "patterns": [
            r"edit\s+(.+)",
            r"modify\s+(.+)",
            r"change\s+(.+)\s+(?:in|to)\s+(.+)",
            r"replace\s+(.+)\s+with\s+(.+)",
        ],
        "arg_name": "file_path",
    },
}

# Code generation patterns
CODE_GENERATION_PATTERNS = {
    "keywords": {"plot", "chart", "graph", "histogram", "scatter", "sine", "cosine", 
                 "wave", "bar", "line", "visualize", "visualization", "matplotlib",
                 "create", "make", "generate", "draw"},
    "patterns": [
        r"(?:create|make|generate|draw|plot)\s+(?:a\s+)?(?:.*?)(?:plot|chart|graph|histogram|wave)",
        r"(?:plot|visualize|show)\s+(?:a\s+)?(?:sine|cosine|bar|line|scatter)",
    ],
}

# Knowledge query patterns
KNOWLEDGE_PATTERNS = {
    "keywords": {"what", "how", "why", "explain", "describe", "tell", "define", "meaning"},
    "patterns": [
        r"^what\s+(?:is|are|does|do)\s+",
        r"^how\s+(?:do|does|can|to)\s+",
        r"^why\s+(?:is|are|does|do)\s+",
        r"^explain\s+",
        r"^describe\s+",
        r"^tell\s+me\s+(?:about\s+)?",
        r"^define\s+",
    ],
}


class IntentClassifier:
    """
    Classifies user queries into intent categories.
    
    Uses a combination of:
    1. Keyword matching for quick classification
    2. Pattern matching for tool argument extraction
    3. Holographic space for ambiguous cases
    """
    
    def __init__(self):
        # Build holographic space for intent classification
        self.intent_space = HolographicPatternSpace(dims=8)
        self._seed_intent_patterns()
    
    def _seed_intent_patterns(self):
        """Seed the holographic space with intent examples."""
        # Code generation examples
        code_examples = [
            "create a sine wave plot",
            "make a bar chart",
            "generate a histogram",
            "plot a scatter diagram",
            "draw a line graph",
            "create a cosine wave",
            "make a visualization",
            "plot x squared",
        ]
        for i, example in enumerate(code_examples):
            self.intent_space.add_module(
                name=f"code_{i}",
                text=example,
                module_type="intent",
                effects={"intent": Intent.CODE_GENERATION.value},
            )
        
        # Tool call examples
        tool_examples = [
            "list files in current directory",
            "read the README file",
            "show contents of config.py",
            "run pytest",
            "execute the tests",
            "find all python files",
            "search for TODO in the code",
            "edit the main.py file",
            "cat package.json",
            "ls -la",
        ]
        for i, example in enumerate(tool_examples):
            self.intent_space.add_module(
                name=f"tool_{i}",
                text=example,
                module_type="intent",
                effects={"intent": Intent.TOOL_CALL.value},
            )
        
        # Knowledge query examples
        knowledge_examples = [
            "what is a sine wave",
            "how does matplotlib work",
            "explain the difference between bar and histogram",
            "what are the parameters for scatter plot",
            "describe numpy arrays",
            "tell me about python decorators",
            "why use virtual environments",
        ]
        for i, example in enumerate(knowledge_examples):
            self.intent_space.add_module(
                name=f"knowledge_{i}",
                text=example,
                module_type="intent",
                effects={"intent": Intent.KNOWLEDGE.value},
            )
    
    def classify(self, query: str) -> IntentMatch:
        """
        Classify a query into an intent category.
        
        Returns IntentMatch with intent, confidence, and extracted info.
        """
        query_lower = query.lower().strip()
        
        # Step 1: Check for knowledge query patterns FIRST
        # Questions about things should not be confused with creating things
        knowledge_match = self._match_knowledge_patterns(query_lower)
        if knowledge_match and knowledge_match.confidence >= 0.6:
            return knowledge_match
        
        # Step 2: Check for explicit tool patterns
        tool_match = self._match_tool_patterns(query_lower)
        if tool_match and tool_match.confidence >= 0.7:
            return tool_match
        
        # Step 3: Check for code generation patterns
        code_match = self._match_code_patterns(query_lower)
        if code_match and code_match.confidence >= 0.6:
            return code_match
        
        # Step 4: Use holographic space for ambiguous cases
        holo_match = self._match_holographic(query_lower)
        if holo_match and holo_match.confidence >= 0.4:
            return holo_match
        
        # Step 5: Default to unsupported
        return IntentMatch(
            intent=Intent.UNSUPPORTED,
            confidence=0.5,
            reason="no clear intent detected"
        )
    
    def _match_tool_patterns(self, query: str) -> Optional[IntentMatch]:
        """Match against Goose tool patterns."""
        best_match = None
        best_confidence = 0.0
        
        for tool_name, tool_info in GOOSE_TOOL_PATTERNS.items():
            keywords = tool_info["keywords"]
            patterns = tool_info["patterns"]
            
            # Keyword overlap score
            query_words = set(query.split())
            keyword_overlap = len(keywords & query_words) / max(len(keywords), 1)
            
            # Pattern match score
            pattern_score = 0.0
            extracted_args = {}
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    pattern_score = 0.8
                    # Extract argument if captured
                    groups = match.groups()
                    if groups and groups[0]:
                        arg_name = tool_info.get("arg_name", "arg")
                        extracted_args[arg_name] = groups[0].strip()
                    break
            
            # Combined confidence
            confidence = max(keyword_overlap * 0.6 + 0.2, pattern_score)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = IntentMatch(
                    intent=Intent.TOOL_CALL,
                    confidence=confidence,
                    reason=f"matched {tool_name} tool pattern",
                    tool_name=tool_name,
                    tool_args=extracted_args,
                )
        
        return best_match
    
    def _match_code_patterns(self, query: str) -> Optional[IntentMatch]:
        """Match against code generation patterns."""
        keywords = CODE_GENERATION_PATTERNS["keywords"]
        patterns = CODE_GENERATION_PATTERNS["patterns"]
        
        query_words = set(query.split())
        keyword_overlap = len(keywords & query_words) / max(len(query_words), 1)
        
        # Check patterns
        pattern_matched = False
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                pattern_matched = True
                break
        
        if keyword_overlap >= 0.2 or pattern_matched:
            confidence = min(keyword_overlap * 1.5 + (0.3 if pattern_matched else 0), 1.0)
            return IntentMatch(
                intent=Intent.CODE_GENERATION,
                confidence=confidence,
                reason="matched code generation keywords/patterns"
            )
        
        return None
    
    def _match_knowledge_patterns(self, query: str) -> Optional[IntentMatch]:
        """Match against knowledge query patterns."""
        keywords = KNOWLEDGE_PATTERNS["keywords"]
        patterns = KNOWLEDGE_PATTERNS["patterns"]
        
        # Check if query starts with question words
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return IntentMatch(
                    intent=Intent.KNOWLEDGE,
                    confidence=0.8,
                    reason="matched knowledge query pattern"
                )
        
        # Check keyword overlap
        query_words = set(query.split())
        keyword_overlap = len(keywords & query_words)
        
        if keyword_overlap >= 1:
            return IntentMatch(
                intent=Intent.KNOWLEDGE,
                confidence=0.5 + keyword_overlap * 0.1,
                reason="matched knowledge keywords"
            )
        
        return None
    
    def _match_holographic(self, query: str) -> Optional[IntentMatch]:
        """Use holographic space for classification."""
        module, confidence, reason = self.intent_space.find_best_match(query, min_similarity=0.2)
        
        if module is None:
            return None
        
        intent_str = module.effects.get("intent", "unsupported")
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.UNSUPPORTED
        
        return IntentMatch(
            intent=intent,
            confidence=confidence,
            reason=f"holographic match: {reason}"
        )
    
    def extract_tool_args(self, query: str, tool_name: str) -> Dict[str, Any]:
        """Extract arguments for a specific tool from the query."""
        if tool_name not in GOOSE_TOOL_PATTERNS:
            return {}
        
        tool_info = GOOSE_TOOL_PATTERNS[tool_name]
        patterns = tool_info["patterns"]
        arg_name = tool_info.get("arg_name", "arg")
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and match.groups():
                # Get the first non-None group
                for group in match.groups():
                    if group:
                        return {arg_name: group.strip()}
        
        return {}
