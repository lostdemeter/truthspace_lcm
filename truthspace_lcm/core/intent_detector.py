"""
Intent Detector Gear

Routes user requests to the appropriate handler:
- chat: Knowledge queries, questions about topics
- tool_call: Needs system action (file ops, commands)
- orchestrator: Complex multi-step tasks

This is the routing layer that sits at the front of the gear chain
and decides how to handle each request.

Example:
    detector = IntentDetectorGear()
    
    detector.detect("Who is Captain Ahab?")
    # → Intent.CHAT
    
    detector.detect("Create a directory called test")
    # → Intent.TOOL_CALL
    
    detector.detect("Set up a new Python project with git and requirements")
    # → Intent.ORCHESTRATOR

Author: Lesley Gushurst
License: GPLv3
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

from truthspace_lcm.core.base import Gear, GearState
from truthspace_lcm.core.gear_message import (
    GearProtocol, GearMessage, MessageIntent, 
    normalize_input, adapt_to_gear_state, adapt_from_gear_state
)


class Intent(Enum):
    """The detected intent of a user request."""
    CHAT = auto()          # Knowledge query, conversation
    TOOL_CALL = auto()     # Single system action needed
    ORCHESTRATOR = auto()  # Complex multi-step task
    CODE_GENERATION = auto()  # Python code generation needed
    UNKNOWN = auto()       # Can't determine


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: Intent
    confidence: float
    reason: str
    extracted_action: Optional[str] = None  # For tool calls
    extracted_goal: Optional[str] = None    # For orchestrator


class IntentDetectorGear(GearProtocol):
    """
    Detects user intent and routes to appropriate handler.
    
    Uses emergent pattern matching to classify requests into:
    - CHAT: Questions about knowledge, topics, entities
    - TOOL_CALL: Requests that need system actions
    - ORCHESTRATOR: Complex multi-step tasks
    
    The key insight: we can detect intent by looking for ACTION VERBS
    that imply system manipulation vs. QUESTION WORDS that imply
    knowledge retrieval.
    
    Implements GearProtocol for standardized communication.
    """
    
    # Action verbs that indicate tool/orchestrator needs
    ACTION_VERBS = {
        # File operations
        'create', 'make', 'touch', 'delete', 'remove', 'copy', 'move',
        'rename', 'write', 'read', 'open', 'save', 'edit', 'modify',
        # Directory operations
        'mkdir', 'cd', 'ls', 'list', 'show', 'display',
        # System operations
        'run', 'execute', 'start', 'stop', 'kill', 'install', 'uninstall',
        'download', 'upload', 'fetch', 'get', 'set', 'configure',
        # Project operations
        'setup', 'initialize', 'init', 'build', 'compile', 'deploy',
    }
    
    # Question words that indicate chat/knowledge queries
    QUESTION_WORDS = {
        'who', 'what', 'where', 'when', 'why', 'how', 'which',
        'is', 'are', 'was', 'were', 'do', 'does', 'did',
        'can', 'could', 'would', 'should', 'will',
        'tell', 'explain', 'describe', 'define',
    }
    
    # Patterns that strongly indicate tool calls
    TOOL_PATTERNS = [
        r'\b(create|make)\s+(a\s+)?(file|directory|folder|dir)\b',
        r'\b(delete|remove|rm)\s+(the\s+)?(file|directory|folder)\b',
        r'\b(write|pipe|echo)\s+.+\s+(to|into)\s+',
        r'\b(run|execute)\s+(the\s+)?(command|script|program)\b',
        r'\b(install|uninstall)\s+\w+',
        r'\b(start|stop|restart)\s+(the\s+)?(server|service|process)\b',
        r'\b(copy|move|rename)\s+.+\s+(to|as)\s+',
        r'\b(list|show|display)\s+(the\s+)?(files|directories|contents)\b',
        r'\bcould you\s+(create|make|delete|run|execute|write)\b',
        r'\bplease\s+(create|make|delete|run|execute|write)\b',
        r'\bcan you\s+(create|make|delete|run|execute|write)\b',
    ]
    
    # Patterns that indicate orchestrator (multi-step)
    ORCHESTRATOR_PATTERNS = [
        r'\b(and then|then|after that|next|finally)\b',
        r'\b(setup|set up|initialize|configure)\s+(a\s+)?(new\s+)?(project|environment|workspace)\b',
        r'\b(create|make).+,\s*(create|make|and)\b',  # Multiple actions
        r'\bstep[s]?\s*\d*\s*:\s*',
        r'\bfirst.+then\b',
        r'\b(build|deploy|release)\s+(the\s+)?(app|application|project)\b',
    ]
    
    # Patterns that indicate code generation
    CODE_PATTERNS = [
        r'\b(write|generate|create)\s+(a\s+)?(python|code|script|program|function)\b',
        r'\bpython\s+(code|script|program|function)\b',
        r'\b(code|script)\s+that\s+(will|can|does)\b',
        r'\bfunction\s+that\s+(takes|returns|calculates|computes)\b',
        r'\bprogram\s+that\s+(reads|writes|prints|calculates)\b',
        r'\b(write|create)\s+code\s+(to|for|that)\b',
        r'\bgenerate\s+(some\s+)?(code|python)\b',
        r'\bcan you (write|code|program)\b',
        r'\b(simple|basic)\s+(python\s+)?(program|script|code)\b',
        # Visualization/plotting patterns
        r'\b(create|make|generate|plot|draw)\s+(a\s+)?(bar\s+chart|line\s+chart|pie\s+chart|scatter\s+plot|histogram|graph|plot)\b',
        r'\b(matplotlib|pyplot|plt)\b',
        r'\bplot\s+(a\s+)?(sine|cosine|line|bar|scatter)\b',
        r'\bvisualize\s+(the\s+)?(data|results)\b',
        r'\b(sine|cosine)\s+(wave|plot|graph)\b',
        r'\b(bar|scatter|line)\s+(chart|plot|graph)\b',
        r'\bcreate\s+(a\s+)?(histogram|plot|chart|graph)\b',
        r'\b3d\s+(plot|surface|graph)\b',
        r'\bsurface\s+plot\b',
        r'\bheatmap\b',
        r'\bcontour\s+plot\b',
    ]
    
    # Code-related keywords
    CODE_KEYWORDS = {
        'python', 'code', 'script', 'program', 'function', 'def',
        'variable', 'loop', 'iterate', 'calculate', 'compute',
        'plot', 'chart', 'graph', 'matplotlib', 'visualize', 'histogram',
        '3d', 'surface', 'heatmap', 'contour',
    }
    
    # Patterns that indicate chat/knowledge
    CHAT_PATTERNS = [
        r'^(who|what|where|when|why|how|which)\s+(is|are|was|were|did|do|does)\b',
        r'\btell\s+me\s+about\b',
        r'\bexplain\s+(what|how|why)\b',
        r'\bwhat\s+does\s+.+\s+mean\b',
        r'\bdefine\s+\w+',
        r'\bdescribe\s+\w+',
        r'^is\s+(it|this|that|there)\b',
        r'\bdo you know\b',
        r'\bhave you heard\b',
    ]
    
    def __init__(self):
        self.name = "IntentDetectorGear"
        
        # Compile patterns for efficiency
        self.tool_patterns = [re.compile(p, re.IGNORECASE) for p in self.TOOL_PATTERNS]
        self.orchestrator_patterns = [re.compile(p, re.IGNORECASE) for p in self.ORCHESTRATOR_PATTERNS]
        self.chat_patterns = [re.compile(p, re.IGNORECASE) for p in self.CHAT_PATTERNS]
        self.code_patterns = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
    
    def detect(self, text: str) -> IntentResult:
        """
        Detect the intent of a user request.
        
        Returns IntentResult with intent type, confidence, and reason.
        """
        text_lower = text.lower().strip()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Score each intent type
        scores = {
            Intent.CHAT: 0.0,
            Intent.TOOL_CALL: 0.0,
            Intent.ORCHESTRATOR: 0.0,
            Intent.CODE_GENERATION: 0.0,
        }
        reasons = {
            Intent.CHAT: [],
            Intent.TOOL_CALL: [],
            Intent.ORCHESTRATOR: [],
            Intent.CODE_GENERATION: [],
        }
        
        # Check for action verbs
        action_verbs_found = words & self.ACTION_VERBS
        if action_verbs_found:
            scores[Intent.TOOL_CALL] += len(action_verbs_found) * 0.3
            reasons[Intent.TOOL_CALL].append(f"action verbs: {action_verbs_found}")
        
        # Check for question words
        question_words_found = words & self.QUESTION_WORDS
        if question_words_found:
            scores[Intent.CHAT] += len(question_words_found) * 0.2
            reasons[Intent.CHAT].append(f"question words: {question_words_found}")
        
        # Check tool patterns
        for pattern in self.tool_patterns:
            if pattern.search(text_lower):
                scores[Intent.TOOL_CALL] += 0.4
                reasons[Intent.TOOL_CALL].append(f"tool pattern match")
                break
        
        # Check orchestrator patterns
        orchestrator_matches = 0
        for pattern in self.orchestrator_patterns:
            if pattern.search(text_lower):
                orchestrator_matches += 1
        
        if orchestrator_matches > 0:
            scores[Intent.ORCHESTRATOR] += orchestrator_matches * 0.3
            reasons[Intent.ORCHESTRATOR].append(f"{orchestrator_matches} orchestrator patterns")
        
        # Check chat patterns
        for pattern in self.chat_patterns:
            if pattern.search(text_lower):
                scores[Intent.CHAT] += 0.5
                reasons[Intent.CHAT].append("chat pattern match")
                break
        
        # Check code generation patterns
        for pattern in self.code_patterns:
            if pattern.search(text_lower):
                scores[Intent.CODE_GENERATION] += 0.6
                reasons[Intent.CODE_GENERATION].append("code pattern match")
                break
        
        # Check for code keywords
        code_keywords_found = words & self.CODE_KEYWORDS
        if code_keywords_found:
            scores[Intent.CODE_GENERATION] += len(code_keywords_found) * 0.2
            reasons[Intent.CODE_GENERATION].append(f"code keywords: {code_keywords_found}")
        
        # Multi-step detection: commas or "and" with action verbs
        if ',' in text and action_verbs_found:
            scores[Intent.ORCHESTRATOR] += 0.3
            reasons[Intent.ORCHESTRATOR].append("comma-separated actions")
        
        if ' and ' in text_lower and len(action_verbs_found) > 1:
            scores[Intent.ORCHESTRATOR] += 0.3
            reasons[Intent.ORCHESTRATOR].append("multiple actions with 'and'")
        
        # Determine winner - code generation takes priority if strong match
        if scores[Intent.CODE_GENERATION] >= 0.6:
            best = Intent.CODE_GENERATION
        elif scores[Intent.ORCHESTRATOR] > scores[Intent.TOOL_CALL]:
            # Orchestrator beats tool call
            best = Intent.ORCHESTRATOR
        elif scores[Intent.TOOL_CALL] > scores[Intent.CHAT]:
            # Tool call beats chat
            best = Intent.TOOL_CALL
        elif scores[Intent.CHAT] > 0:
            best = Intent.CHAT
        else:
            # Default to chat for unknown
            best = Intent.CHAT
            scores[Intent.CHAT] = 0.3
            reasons[Intent.CHAT].append("default fallback")
        
        # Calculate confidence
        total = sum(scores.values()) or 1
        confidence = scores[best] / total
        
        # Build reason string
        reason = "; ".join(reasons[best]) if reasons[best] else "no specific patterns"
        
        # Extract action/goal for tool calls and orchestrator
        extracted_action = None
        extracted_goal = None
        
        if best in (Intent.TOOL_CALL, Intent.ORCHESTRATOR):
            extracted_goal = text  # The whole text is the goal
            if action_verbs_found:
                extracted_action = list(action_verbs_found)[0]
        
        return IntentResult(
            intent=best,
            confidence=min(1.0, confidence),
            reason=reason,
            extracted_action=extracted_action,
            extracted_goal=extracted_goal,
        )
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """
        Detect intent and add to message context.
        
        Implements GearProtocol.process_message.
        """
        result = self.detect(message.content)
        
        # Map our Intent to MessageIntent
        intent_map = {
            Intent.CHAT: MessageIntent.QUERY,
            Intent.TOOL_CALL: MessageIntent.EXECUTE,
            Intent.ORCHESTRATOR: MessageIntent.EXECUTE,
            Intent.CODE_GENERATION: MessageIntent.EXECUTE,
            Intent.UNKNOWN: MessageIntent.QUERY,
        }
        
        # Forward with detected intent
        return self.send(
            message.with_context('intent_result', {
                'intent': result.intent.name,
                'confidence': result.confidence,
                'reason': result.reason,
                'extracted_action': result.extracted_action,
                'extracted_goal': result.extracted_goal,
            }),
            intent=intent_map.get(result.intent, MessageIntent.QUERY)
        )
    
    def forward(self, state: GearState) -> GearState:
        """Detect intent and add to state (legacy GearState interface)."""
        # Convert to GearMessage, process, convert back
        message = adapt_from_gear_state(state, self.name)
        result_message = self.process_message(message)
        
        # Update state with results
        intent_result = result_message.context.get('intent_result', {})
        state.metadata['intent'] = intent_result.get('intent')
        state.metadata['intent_confidence'] = intent_result.get('confidence')
        state.metadata['intent_reason'] = intent_result.get('reason')
        
        if intent_result.get('extracted_goal'):
            state.metadata['goal'] = intent_result['extracted_goal']
        if intent_result.get('extracted_action'):
            state.metadata['action'] = intent_result['extracted_action']
        
        return state
    
    def route(self, text: str, chat_handler=None, tool_handler=None, 
              orchestrator_handler=None) -> Any:
        """
        Detect intent and route to appropriate handler.
        
        Args:
            text: User input
            chat_handler: Function to handle chat intents
            tool_handler: Function to handle tool call intents
            orchestrator_handler: Function to handle orchestrator intents
        
        Returns:
            Result from the appropriate handler
        """
        result = self.detect(text)
        
        if result.intent == Intent.CHAT and chat_handler:
            return chat_handler(text)
        elif result.intent == Intent.TOOL_CALL and tool_handler:
            return tool_handler(text)
        elif result.intent == Intent.ORCHESTRATOR and orchestrator_handler:
            return orchestrator_handler(text)
        
        # Fallback
        if chat_handler:
            return chat_handler(text)
        
        return result


class SmartChatGear(GearProtocol):
    """
    A chat gear that knows when it can't handle something
    and routes to the orchestrator.
    
    This wraps a ConversationalChain and adds intent detection.
    
    Implements GearProtocol for standardized communication.
    """
    
    def __init__(self, chain=None, orchestrator=None):
        self.name = "SmartChatGear"
        
        self.chain = chain  # ConversationalChain
        self.orchestrator = orchestrator  # GearOrchestrator
        self.detector = IntentDetectorGear()
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for orchestrator."""
        self.llm_url = url
        self.llm_model = model
        
        if self.orchestrator:
            self.orchestrator.configure_llm(url, model)
    
    def set_chain(self, chain):
        """Set the conversational chain."""
        self.chain = chain
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator."""
        self.orchestrator = orchestrator
        if self.llm_url:
            self.orchestrator.configure_llm(self.llm_url, self.llm_model)
    
    def chat(self, user_input: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Smart chat that routes based on intent.
        
        Returns dict with:
            - response: The response text
            - intent: What intent was detected
            - handler: Which handler processed it
            - commands: (if orchestrator) The generated commands
        """
        # Detect intent
        intent_result = self.detector.detect(user_input)
        
        result = {
            'input': user_input,
            'intent': intent_result.intent.name,
            'confidence': intent_result.confidence,
            'reason': intent_result.reason,
            'handler': None,
            'response': None,
            'commands': None,
        }
        
        if intent_result.intent == Intent.CHAT:
            # Use conversational chain
            if self.chain:
                result['handler'] = 'chat'
                result['response'] = self.chain.chat(user_input)
            else:
                result['response'] = "I don't have a knowledge base loaded."
        
        elif intent_result.intent in (Intent.TOOL_CALL, Intent.ORCHESTRATOR):
            # Use orchestrator
            if self.orchestrator:
                result['handler'] = 'orchestrator'
                
                orch_result = self.orchestrator.execute(user_input, dry_run=dry_run)
                
                result['commands'] = orch_result['commands']
                result['plan'] = orch_result['plan']
                
                # Build response
                if dry_run:
                    cmd_list = '\n'.join([f"  $ {cmd}" for cmd in orch_result['commands']])
                    result['response'] = (
                        f"I'll need to run these commands:\n{cmd_list}\n\n"
                        f"Should I execute them?"
                    )
                else:
                    result['response'] = f"Done! Executed {len(orch_result['commands'])} commands."
                    result['outputs'] = orch_result.get('outputs', [])
            else:
                result['response'] = (
                    "I understand you want me to perform a system action, "
                    "but I don't have an orchestrator configured."
                )
        
        else:
            result['handler'] = 'fallback'
            result['response'] = "I'm not sure how to help with that."
        
        return result
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """
        Smart chat that routes based on intent.
        
        Implements GearProtocol.process_message.
        """
        result = self.chat(message.content)
        
        return self.send(
            message.with_context('smart_chat_result', result),
            content=result['response']
        )
    
    def forward(self, state: GearState) -> GearState:
        """Process through smart chat (legacy GearState interface)."""
        message = adapt_from_gear_state(state, self.name)
        result_message = self.process_message(message)
        
        state.metadata['smart_chat_result'] = result_message.context.get('smart_chat_result', {})
        state.metadata['response'] = result_message.content
        
        return state
