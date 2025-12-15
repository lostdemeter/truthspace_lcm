"""
Intent Manager for TruthSpace LCM

Intents are first-class knowledge entries that encode:
1. Trigger patterns - what natural language indicates this intent
2. Target knowledge - what command/function/pattern fulfills it
3. Step type - bash vs python
4. Semantic position - so similar intents cluster together in TruthSpace

When the LCM learns a new command, it also generates intent entries.
The generators then become intent resolvers - they find matching intents
and use the linked knowledge to generate code.

This allows the LCM to learn new capabilities without code changes.
"""

import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from truthspace_lcm.core.knowledge_manager import (
    KnowledgeManager, 
    KnowledgeDomain, 
    KnowledgeEntry
)


class StepType(Enum):
    """Type of execution step."""
    BASH = "bash"
    PYTHON = "python"
    EITHER = "either"  # Could be either, context-dependent


@dataclass
class Intent:
    """
    An intent represents a user's goal and how to fulfill it.
    
    Intents bridge natural language to executable knowledge.
    """
    id: str
    name: str                           # e.g., "show_network_interfaces"
    description: str                    # Human-readable description
    triggers: List[str]                 # Regex patterns that activate this intent
    trigger_keywords: List[str]         # Keywords for semantic matching
    target_commands: List[str]          # Command names this intent maps to
    step_type: StepType                 # bash or python
    priority: int = 50                  # Higher = checked first (0-100)
    
    # Links to knowledge entries
    knowledge_entry_ids: List[str] = field(default_factory=list)
    
    def matches(self, text: str) -> Tuple[bool, float]:
        """
        Check if this intent matches the given text.
        
        Returns (matched, confidence) where confidence is 0.0-1.0
        """
        text_lower = text.lower()
        
        # Check regex patterns
        for pattern in self.triggers:
            try:
                if re.search(pattern, text_lower):
                    return True, 0.9
            except re.error:
                continue
        
        # Check keyword overlap
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        keyword_set = set(kw.lower() for kw in self.trigger_keywords)
        
        if keyword_set:
            overlap = len(text_words & keyword_set)
            if overlap >= 2:
                confidence = min(0.8, overlap / len(keyword_set))
                return True, confidence
        
        return False, 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "trigger_keywords": self.trigger_keywords,
            "target_commands": self.target_commands,
            "step_type": self.step_type.value,
            "priority": self.priority,
            "knowledge_entry_ids": self.knowledge_entry_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            triggers=data["triggers"],
            trigger_keywords=data["trigger_keywords"],
            target_commands=data["target_commands"],
            step_type=StepType(data["step_type"]),
            priority=data.get("priority", 50),
            knowledge_entry_ids=data.get("knowledge_entry_ids", []),
        )


class IntentGenerator:
    """
    Generates intent entries from acquired knowledge.
    
    When we learn about a command like 'ifconfig', this generates
    the intent patterns that will trigger its use.
    """
    
    # Common verb patterns for different action types
    ACTION_VERBS = {
        "display": ["show", "display", "view", "print", "list", "get", "see"],
        "create": ["create", "make", "add", "new", "generate", "init"],
        "delete": ["delete", "remove", "rm", "destroy", "clear"],
        "modify": ["change", "modify", "update", "set", "edit", "alter"],
        "search": ["find", "search", "look", "locate", "grep"],
        "transfer": ["copy", "move", "transfer", "send", "download", "upload"],
        "execute": ["run", "execute", "start", "launch", "invoke"],
        "stop": ["stop", "kill", "terminate", "end", "halt"],
    }
    
    # Domain-specific keyword expansions
    DOMAIN_KEYWORDS = {
        "network": ["network", "interface", "ip", "ethernet", "wifi", "connection", "socket", "port"],
        "process": ["process", "pid", "running", "task", "job", "daemon", "service"],
        "file": ["file", "files", "directory", "folder", "path", "disk"],
        "system": ["system", "kernel", "os", "boot", "hardware", "memory", "cpu"],
        "user": ["user", "account", "permission", "group", "owner"],
    }
    
    def generate_intent_from_command(
        self, 
        command_name: str,
        description: str,
        keywords: List[str],
        step_type: StepType = StepType.BASH,
        knowledge_entry_id: str = None
    ) -> Intent:
        """
        Generate an intent entry from a learned command.
        
        Args:
            command_name: The command (e.g., "ifconfig")
            description: What the command does
            keywords: Keywords from the knowledge entry
            step_type: bash or python
            knowledge_entry_id: ID of the source knowledge entry
        """
        # Generate trigger patterns
        triggers = self._generate_triggers(command_name, description, keywords)
        
        # Generate trigger keywords
        trigger_keywords = self._generate_keywords(command_name, description, keywords)
        
        # Generate intent name
        intent_name = self._generate_intent_name(command_name, description)
        
        # Generate ID
        intent_id = hashlib.sha256(
            f"intent:{intent_name}:{command_name}".encode()
        ).hexdigest()[:16]
        
        return Intent(
            id=intent_id,
            name=intent_name,
            description=f"Intent to use {command_name}: {description[:100]}",
            triggers=triggers,
            trigger_keywords=trigger_keywords,
            target_commands=[command_name],
            step_type=step_type,
            priority=50,
            knowledge_entry_ids=[knowledge_entry_id] if knowledge_entry_id else [],
        )
    
    def _generate_triggers(
        self, 
        command_name: str, 
        description: str, 
        keywords: List[str]
    ) -> List[str]:
        """Generate regex trigger patterns."""
        triggers = []
        
        # Always trigger on the command name itself
        triggers.append(rf"\b{re.escape(command_name)}\b")
        triggers.append(rf"using\s+{re.escape(command_name)}")
        triggers.append(rf"with\s+{re.escape(command_name)}")
        
        # Extract key nouns from description
        desc_lower = description.lower()
        
        # Detect the domain and action from description
        domain = self._detect_domain(desc_lower, keywords)
        action = self._detect_action(desc_lower)
        
        if domain and action:
            # Generate patterns like "show network" or "list processes"
            for verb in self.ACTION_VERBS.get(action, [action])[:3]:
                for noun in self.DOMAIN_KEYWORDS.get(domain, [domain])[:3]:
                    triggers.append(rf"{verb}\s+(?:my\s+)?(?:the\s+)?{noun}")
        
        return triggers[:10]  # Limit to 10 patterns
    
    def _generate_keywords(
        self, 
        command_name: str, 
        description: str, 
        keywords: List[str]
    ) -> List[str]:
        """Generate trigger keywords for semantic matching."""
        result = set(keywords)
        result.add(command_name)
        
        # Add domain keywords
        desc_lower = description.lower()
        for domain, domain_kws in self.DOMAIN_KEYWORDS.items():
            if domain in desc_lower:
                result.update(domain_kws[:3])
        
        # Add action keywords
        for action, verbs in self.ACTION_VERBS.items():
            for verb in verbs:
                if verb in desc_lower:
                    result.update(verbs[:3])
                    break
        
        return list(result)[:20]
    
    def _generate_intent_name(self, command_name: str, description: str) -> str:
        """Generate a descriptive intent name."""
        desc_lower = description.lower()
        
        # Try to extract action and domain
        action = self._detect_action(desc_lower) or "use"
        domain = self._detect_domain(desc_lower, []) or command_name
        
        return f"{action}_{domain}_{command_name}"
    
    def _detect_domain(self, text: str, keywords: List[str]) -> Optional[str]:
        """Detect the domain from text and keywords."""
        text_and_kw = text + " " + " ".join(keywords)
        
        for domain, domain_kws in self.DOMAIN_KEYWORDS.items():
            if any(kw in text_and_kw for kw in domain_kws):
                return domain
        return None
    
    def _detect_action(self, text: str) -> Optional[str]:
        """Detect the action type from text."""
        for action, verbs in self.ACTION_VERBS.items():
            if any(verb in text for verb in verbs):
                return action
        return None


class IntentManager:
    """
    Manages intent entries - storage, retrieval, and matching.
    
    Intents are stored as knowledge entries with entry_type="intent".
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        
        self.storage_dir = storage_dir
        self.manager = KnowledgeManager(storage_dir=storage_dir)
        self.generator = IntentGenerator()
        
        # Cache of loaded intents
        self._intents: Dict[str, Intent] = {}
        self._load_intents()
    
    def _load_intents(self):
        """Load all intent entries from knowledge base."""
        self._intents.clear()
        
        for entry in self.manager.entries.values():
            if entry.entry_type == "intent":
                try:
                    intent_data = entry.metadata.get("intent_data", {})
                    if intent_data:
                        intent = Intent.from_dict(intent_data)
                        self._intents[intent.id] = intent
                except Exception as e:
                    print(f"Warning: Could not load intent from {entry.id}: {e}")
    
    def create_intent(self, intent: Intent) -> KnowledgeEntry:
        """
        Store an intent as a knowledge entry.
        """
        entry = self.manager.create(
            name=intent.name,
            domain=KnowledgeDomain.PROGRAMMING,
            entry_type="intent",
            description=intent.description,
            keywords=intent.trigger_keywords,
            metadata={
                "intent_data": intent.to_dict(),
                "triggers": intent.triggers,
                "target_commands": intent.target_commands,
                "step_type": intent.step_type.value,
            }
        )
        
        self._intents[intent.id] = intent
        return entry
    
    def create_intent_for_command(
        self,
        command_name: str,
        description: str,
        keywords: List[str],
        step_type: StepType = StepType.BASH,
        knowledge_entry_id: str = None
    ) -> Intent:
        """
        Generate and store an intent for a learned command.
        """
        intent = self.generator.generate_intent_from_command(
            command_name=command_name,
            description=description,
            keywords=keywords,
            step_type=step_type,
            knowledge_entry_id=knowledge_entry_id
        )
        
        self.create_intent(intent)
        return intent
    
    def find_matching_intents(
        self, 
        text: str, 
        step_type: StepType = None
    ) -> List[Tuple[Intent, float]]:
        """
        Find all intents that match the given text.
        
        Returns list of (intent, confidence) sorted by confidence descending.
        """
        matches = []
        
        for intent in self._intents.values():
            # Filter by step type if specified
            if step_type and intent.step_type != step_type and intent.step_type != StepType.EITHER:
                continue
            
            matched, confidence = intent.matches(text)
            if matched:
                matches.append((intent, confidence))
        
        # Sort by confidence (descending), then priority (descending)
        matches.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        return matches
    
    def get_best_intent(
        self, 
        text: str, 
        step_type: StepType = None
    ) -> Optional[Tuple[Intent, float]]:
        """Get the best matching intent for the text."""
        matches = self.find_matching_intents(text, step_type)
        return matches[0] if matches else None
    
    def get_command_for_request(
        self, 
        text: str, 
        step_type: StepType = None
    ) -> Optional[Tuple[str, float, Intent]]:
        """
        Get the command to execute for a request.
        
        Returns (command, confidence, intent) or None.
        """
        result = self.get_best_intent(text, step_type)
        if result:
            intent, confidence = result
            if intent.target_commands:
                return intent.target_commands[0], confidence, intent
        return None
    
    def list_intents(self) -> List[Intent]:
        """List all registered intents."""
        return list(self._intents.values())
    
    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Get an intent by ID."""
        return self._intents.get(intent_id)


def demonstrate():
    """Demonstrate the intent system."""
    print("=" * 70)
    print("INTENT MANAGER DEMONSTRATION")
    print("=" * 70)
    print()
    
    manager = IntentManager()
    
    # Create an intent for ifconfig
    print("Creating intent for 'ifconfig'...")
    intent = manager.create_intent_for_command(
        command_name="ifconfig",
        description="configure a network interface, display network configuration",
        keywords=["network", "interface", "ip", "ethernet", "configure"],
        step_type=StepType.BASH
    )
    print(f"  Created: {intent.name}")
    print(f"  Triggers: {intent.triggers[:3]}...")
    print(f"  Keywords: {intent.trigger_keywords[:5]}...")
    
    # Create an intent for dmesg
    print("\nCreating intent for 'dmesg'...")
    intent2 = manager.create_intent_for_command(
        command_name="dmesg",
        description="print or control the kernel ring buffer, show system messages",
        keywords=["kernel", "messages", "system", "log", "boot"],
        step_type=StepType.BASH
    )
    print(f"  Created: {intent2.name}")
    print(f"  Triggers: {intent2.triggers[:3]}...")
    
    # Test matching
    print("\n" + "-" * 70)
    print("Testing intent matching:")
    print("-" * 70)
    
    test_requests = [
        "show my network interfaces",
        "display network configuration using ifconfig",
        "show kernel messages",
        "view system log with dmesg",
        "list running processes",  # Should not match
    ]
    
    for request in test_requests:
        result = manager.get_command_for_request(request, StepType.BASH)
        if result:
            cmd, conf, intent = result
            print(f"\n  '{request}'")
            print(f"    → Command: {cmd} (confidence: {conf:.2f})")
            print(f"    → Intent: {intent.name}")
        else:
            print(f"\n  '{request}'")
            print(f"    → No matching intent")
    
    print("\n" + "=" * 70)
    print("Intent demonstration complete!")


if __name__ == "__main__":
    demonstrate()
