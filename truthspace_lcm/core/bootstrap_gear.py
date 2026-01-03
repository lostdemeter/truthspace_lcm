"""
Bootstrap Gear

A meta-gear that trains new emergent capabilities through LLM refinement.
The key insight: we can bootstrap ANY new gear by:

1. Starting with an EmergentGear (blank slate)
2. Attaching a RefinementGear (LLM feedback)
3. Training on examples until patterns emerge
4. Saving the emergent state to JSON
5. Reloading without needing LLM

This allows creating new capabilities (tool calling, sentiment, etc.)
without hardcoding - the behavior emerges from training.

Protocol:
    gear = BootstrapGear("tool_calling")
    gear.configure_llm(url, model)
    
    # Training phase (uses LLM)
    for example in training_examples:
        gear.train(example.input, example.expected_output)
    
    # Save emergent state
    gear.save("tool_calling_v1.json")
    
    # Later: reload without LLM
    gear = BootstrapGear.load("tool_calling_v1.json")
    result = gear.forward(state)  # Pure emergent, no LLM

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import requests
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from truthspace_lcm.core.base import Gear, GearState
from truthspace_lcm.core.gear_message import GearProtocol, GearMessage, MessageIntent


@dataclass
class TrainingExample:
    """A single training example with input, expected output, and metadata."""
    input_text: str
    expected_output: str
    actual_output: Optional[str] = None
    score: float = 0.0
    feedback: str = ""
    patterns_learned: List[str] = field(default_factory=list)


@dataclass
class EmergentPattern:
    """A pattern learned from training examples."""
    trigger: str  # What triggers this pattern (regex or keyword)
    response_template: str  # How to respond
    confidence: float = 0.0
    examples_seen: int = 0
    
    def matches(self, text: str) -> bool:
        """Check if this pattern matches the input."""
        text_lower = text.lower()
        if self.trigger.startswith('re:'):
            # Regex pattern
            return bool(re.search(self.trigger[3:], text_lower))
        else:
            # Keyword match
            return self.trigger.lower() in text_lower
    
    def to_dict(self) -> Dict:
        return {
            'trigger': self.trigger,
            'response_template': self.response_template,
            'confidence': self.confidence,
            'examples_seen': self.examples_seen,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmergentPattern':
        return cls(**data)


@dataclass
class EmergentState:
    """The emergent state that can be saved/loaded."""
    name: str
    version: str = "1.0"
    patterns: List[EmergentPattern] = field(default_factory=list)
    vocabulary: Dict[str, float] = field(default_factory=dict)  # word -> weight
    training_examples: int = 0
    total_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'patterns': [p.to_dict() for p in self.patterns],
            'vocabulary': self.vocabulary,
            'training_examples': self.training_examples,
            'total_score': self.total_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmergentState':
        patterns = [EmergentPattern.from_dict(p) for p in data.get('patterns', [])]
        return cls(
            name=data['name'],
            version=data.get('version', '1.0'),
            patterns=patterns,
            vocabulary=data.get('vocabulary', {}),
            training_examples=data.get('training_examples', 0),
            total_score=data.get('total_score', 0.0),
        )


class BootstrapGear(GearProtocol):
    """
    A meta-gear that bootstraps new emergent capabilities through LLM refinement.
    
    The gear learns patterns from training examples, using LLM feedback to
    refine its understanding. Once trained, the emergent state can be saved
    and reloaded without needing the LLM.
    
    Implements GearProtocol for standardized communication.
    
    Example usage:
        # Create and configure
        gear = BootstrapGear("tool_calling")
        gear.configure_llm("http://localhost:11434/api/generate", "qwen2.5:14b")
        
        # Train on examples
        gear.train("list files in current directory", "ls -la")
        gear.train("show disk usage", "df -h")
        gear.train("what time is it", "date")
        
        # Save emergent state
        gear.save("tool_calling.json")
        
        # Later: load and use without LLM
        gear = BootstrapGear.load("tool_calling.json")
        result = gear.process("show me the files")  # Returns "ls -la"
    """
    
    # Prompt for extracting patterns from examples
    PATTERN_EXTRACTION_PROMPT = """Analyze this input-output pair and extract the pattern:

Input: "{input}"
Expected Output: "{output}"

What trigger words or patterns in the input lead to this output?
Reply with JSON:
{{"trigger": "<keyword or re:regex>", "template": "<output template>"}}"""

    # Prompt for evaluating a response
    EVAL_PROMPT = """Rate this response on a scale of 0-10:

Input: "{input}"
Expected: "{expected}"
Actual: "{actual}"

Reply with JSON:
{{"score": <0-10>, "feedback": "<brief feedback>", "patterns": ["<pattern1>", ...]}}"""

    # Prompt for generating a response (during training)
    GENERATE_PROMPT = """Based on these learned patterns:
{patterns}

Generate the appropriate response for:
Input: "{input}"

Reply with ONLY the response, nothing else."""

    def __init__(self, name: str):
        self.name = f"BootstrapGear:{name}"
        
        self.state = EmergentState(name=name)
        
        # LLM configuration (for training only)
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Training history
        self.training_history: List[TrainingExample] = []
        
        # Keyword frequency for emergent pattern detection
        self.keyword_freq: Counter = Counter()
        self.keyword_outputs: Dict[str, Counter] = defaultdict(Counter)
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for training phase."""
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call LLM for training/evaluation."""
        if not self.llm_url or not self.llm_model:
            return None
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception:
            pass
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                    'what', 'when', 'who', 'how', 'why', 'this', 'that', 'with'}
        return [w for w in words if w not in stopwords]
    
    def train(self, input_text: str, expected_output: str) -> TrainingExample:
        """
        Train on a single example.
        
        Uses LLM to:
        1. Extract patterns from the input-output pair
        2. Evaluate how well current patterns would handle this
        3. Update emergent state with new patterns
        
        Returns the training example with scores and feedback.
        """
        example = TrainingExample(
            input_text=input_text,
            expected_output=expected_output,
        )
        
        # Extract keywords and update frequency
        keywords = self._extract_keywords(input_text)
        for kw in keywords:
            self.keyword_freq[kw] += 1
            self.keyword_outputs[kw][expected_output] += 1
        
        # Try to generate with current patterns
        actual = self.process(input_text)
        example.actual_output = actual
        
        # Use LLM to extract patterns (if available)
        if self.llm_url:
            # Extract pattern from this example
            prompt = self.PATTERN_EXTRACTION_PROMPT.format(
                input=input_text,
                output=expected_output
            )
            result = self._call_llm(prompt)
            
            if result:
                try:
                    # Parse JSON response
                    json_match = re.search(r'\{[^}]+\}', result)
                    if json_match:
                        data = json.loads(json_match.group())
                        trigger = data.get('trigger', '')
                        template = data.get('template', expected_output)
                        
                        if trigger:
                            # Add or update pattern
                            self._add_pattern(trigger, template)
                            example.patterns_learned.append(trigger)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Evaluate the response
            if actual:
                eval_prompt = self.EVAL_PROMPT.format(
                    input=input_text,
                    expected=expected_output,
                    actual=actual
                )
                eval_result = self._call_llm(eval_prompt)
                
                if eval_result:
                    try:
                        json_match = re.search(r'\{[^}]+\}', eval_result)
                        if json_match:
                            data = json.loads(json_match.group())
                            example.score = float(data.get('score', 0))
                            example.feedback = data.get('feedback', '')
                    except (json.JSONDecodeError, ValueError):
                        pass
        
        # Fallback: learn from keyword frequency
        if not example.patterns_learned:
            # Find the most distinctive keyword
            for kw in keywords:
                if self.keyword_outputs[kw][expected_output] >= 2:
                    # This keyword strongly predicts this output
                    self._add_pattern(kw, expected_output)
                    example.patterns_learned.append(kw)
        
        # Update state
        self.state.training_examples += 1
        self.state.total_score += example.score
        self.training_history.append(example)
        
        return example
    
    def _add_pattern(self, trigger: str, template: str):
        """Add or update a pattern."""
        # Check if pattern exists
        for pattern in self.state.patterns:
            if pattern.trigger == trigger:
                pattern.examples_seen += 1
                pattern.confidence = min(1.0, pattern.confidence + 0.1)
                return
        
        # Add new pattern
        self.state.patterns.append(EmergentPattern(
            trigger=trigger,
            response_template=template,
            confidence=0.5,
            examples_seen=1,
        ))
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Process message using learned patterns. Implements GearProtocol."""
        result = self._process_text(message.content)
        return self.send(
            message.with_context('bootstrap_output', result),
            content=result or ''
        )
    
    def _process_text(self, input_text: str) -> Optional[str]:
        """
        Process input using learned patterns (no LLM).
        
        This is the emergent behavior - pure pattern matching
        based on what was learned during training.
        """
        # Try each pattern in order of confidence
        sorted_patterns = sorted(
            self.state.patterns,
            key=lambda p: (p.confidence, p.examples_seen),
            reverse=True
        )
        
        for pattern in sorted_patterns:
            if pattern.matches(input_text):
                return pattern.response_template
        
        # Fallback: keyword frequency matching
        keywords = self._extract_keywords(input_text)
        best_output = None
        best_score = 0
        
        for kw in keywords:
            if kw in self.keyword_outputs:
                for output, count in self.keyword_outputs[kw].items():
                    score = count * self.keyword_freq[kw]
                    if score > best_score:
                        best_score = score
                        best_output = output
        
        return best_output
    
    def forward(self, state: GearState) -> GearState:
        """Apply the gear to a state (legacy interface)."""
        input_text = state.metadata.get('input', '') or state.entity or ''
        
        result = self._process_text(input_text)
        if result:
            state.metadata['bootstrap_output'] = result
            state.metadata['bootstrap_gear'] = self.state.name
        
        return state
    
    def save(self, path: str):
        """Save emergent state to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.state.to_dict()
        data['keyword_freq'] = dict(self.keyword_freq)
        data['keyword_outputs'] = {
            k: dict(v) for k, v in self.keyword_outputs.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BootstrapGear':
        """Load emergent state from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        gear = cls(name=data['name'])
        gear.state = EmergentState.from_dict(data)
        gear.keyword_freq = Counter(data.get('keyword_freq', {}))
        gear.keyword_outputs = defaultdict(Counter)
        for k, v in data.get('keyword_outputs', {}).items():
            gear.keyword_outputs[k] = Counter(v)
        
        return gear
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        avg_score = (
            self.state.total_score / self.state.training_examples
            if self.state.training_examples > 0 else 0
        )
        return {
            'name': self.state.name,
            'version': self.state.version,
            'patterns': len(self.state.patterns),
            'training_examples': self.state.training_examples,
            'avg_score': avg_score,
            'vocabulary_size': len(self.keyword_freq),
        }
    
    def describe_patterns(self) -> str:
        """Describe learned patterns in human-readable format."""
        if not self.state.patterns:
            return "No patterns learned yet."
        
        lines = [f"Learned patterns for '{self.state.name}':"]
        for p in sorted(self.state.patterns, key=lambda x: -x.confidence):
            lines.append(f"  • {p.trigger} → {p.response_template} "
                        f"(conf={p.confidence:.2f}, seen={p.examples_seen})")
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tool_calling_gear(llm_url: str = None, llm_model: str = None) -> BootstrapGear:
    """
    Create a ToolCallingGear with common training examples.
    
    This demonstrates how to bootstrap a new capability.
    """
    gear = BootstrapGear("tool_calling")
    
    if llm_url and llm_model:
        gear.configure_llm(llm_url, llm_model)
    
    # Pre-seed with common patterns
    training_data = [
        ("list files", "ls -la"),
        ("show files in directory", "ls -la"),
        ("what files are here", "ls -la"),
        ("show disk usage", "df -h"),
        ("how much disk space", "df -h"),
        ("check disk", "df -h"),
        ("what time is it", "date"),
        ("current time", "date"),
        ("show date", "date"),
        ("who am i", "whoami"),
        ("current user", "whoami"),
        ("show memory", "free -h"),
        ("memory usage", "free -h"),
        ("running processes", "ps aux"),
        ("show processes", "ps aux"),
        ("network info", "ip addr"),
        ("show ip address", "ip addr"),
    ]
    
    for input_text, expected in training_data:
        gear.train(input_text, expected)
    
    return gear


def create_sentiment_gear(llm_url: str = None, llm_model: str = None) -> BootstrapGear:
    """
    Create a SentimentGear with common training examples.
    """
    gear = BootstrapGear("sentiment")
    
    if llm_url and llm_model:
        gear.configure_llm(llm_url, llm_model)
    
    training_data = [
        ("I love this!", "positive"),
        ("This is amazing", "positive"),
        ("Great work", "positive"),
        ("I hate this", "negative"),
        ("This is terrible", "negative"),
        ("Awful experience", "negative"),
        ("It's okay", "neutral"),
        ("Not bad", "neutral"),
        ("Could be better", "neutral"),
    ]
    
    for input_text, expected in training_data:
        gear.train(input_text, expected)
    
    return gear
