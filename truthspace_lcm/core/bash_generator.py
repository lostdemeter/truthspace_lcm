"""
Bash Generator: Natural Language → Shell Commands

Uses the persistent knowledge base to generate Bash commands
from natural language descriptions.
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry
from truthspace_lcm.core.intent_manager import IntentManager, StepType


@dataclass
class BashGenerationResult:
    """Result of bash generation."""
    success: bool
    command: str
    explanation: str
    knowledge_used: List[str]
    confidence: float
    warnings: List[str]
    from_learned_intent: bool = False  # True if resolved via learned intent


class BashGenerator:
    """Generate Bash commands from natural language."""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        self.storage_dir = storage_dir
        self.manager = KnowledgeManager(storage_dir=storage_dir)
        self.intent_manager = IntentManager(storage_dir=storage_dir)
        self._init_intent_patterns()
    
    def _init_intent_patterns(self):
        """Initialize patterns for detecting user intent."""
        self.intent_patterns = {
            "create_dir": [
                r"create\s+(?:a\s+)?(?:new\s+)?(?:directory|folder|dir)",
                r"make\s+(?:a\s+)?(?:new\s+)?(?:directory|folder|dir)",
                r"mkdir",
            ],
            "create_file": [
                r"create\s+(?:a\s+)?(?:new\s+)?(?:empty\s+)?file",
                r"touch\s+",
                r"make\s+(?:a\s+)?(?:new\s+)?file",
            ],
            "delete_file": [
                r"delete\s+(?:the\s+)?(?:file|files)",
                r"remove\s+(?:the\s+)?(?:file|files)",
                r"rm\s+",
            ],
            "delete_dir": [
                r"delete\s+(?:the\s+)?(?:directory|folder|dir)",
                r"remove\s+(?:the\s+)?(?:directory|folder|dir)",
                r"rmdir",
            ],
            "copy_file": [
                r"copy\s+(?:the\s+)?(?:file|files)",
                r"duplicate\s+",
                r"cp\s+",
            ],
            "move_file": [
                r"move\s+(?:the\s+)?(?:file|files|directory|folder)",
                r"rename\s+",
                r"mv\s+",
            ],
            "list_files": [
                r"list\s+(?:all\s+)?(?:the\s+)?(?:files|contents|directory)",
                r"show\s+(?:all\s+)?(?:the\s+)?(?:files|contents)",
                r"^ls\s*",
                r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:directory|folder)",
            ],
            "view_file": [
                r"(?:view|show|display|print|read)\s+(?:the\s+)?(?:contents?\s+of\s+)[\w./]+",
                r"contents?\s+of\s+[\w./]+",
                r"cat\s+[\w./]+",
                r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?file",
                r"(?:view|read)\s+(?:the\s+)?(?:file\s+)?[\w./]+\.\w+",  # view readme.md
            ],
            "search_text": [
                r"search\s+for\s+['\"]",  # search for 'something'
                r"grep\s+",
                r"look\s+for\s+['\"]",
            ],
            "download": [
                r"download\s+",
                r"https?://",  # Any URL
                r"curl\s+",
                r"wget\s+",
            ],
            "permissions": [
                r"(?:make|set)\s+(?:it\s+)?executable",
                r"change\s+permissions",
                r"chmod\s+",
            ],
            "find_files": [
                r"find\s+(?:all\s+)?\.\w+\s+files",  # find all .py files
                r"find\s+(?:all\s+)?(?:the\s+)?files",
                r"locate\s+",
            ],
            "compress": [
                r"compress\s+",
                r"(?:create\s+)?(?:a\s+)?(?:tar|zip|archive)",
                r"pack\s+",
            ],
            "extract": [
                r"extract\s+",
                r"unzip\s+",
                r"decompress\s+",
                r"unpack\s+",
            ],
            "network": [
                r"(?:show|display|list|get)\s+(?:my\s+)?(?:network|ip|interface)",
                r"ifconfig",
                r"\bip\s+(?:addr|address|link|route)",
                r"network\s+(?:interface|config|status)",
                r"(?:check|view)\s+(?:my\s+)?(?:ip|network)",
            ],
            "process_info": [
                r"(?:list|show)\s+(?:open\s+)?files?\s+(?:using\s+)?lsof",
                r"\blsof\b",
                r"(?:list|show)\s+(?:running\s+)?process",
                r"\bps\b\s+",
                r"(?:check|view)\s+(?:running\s+)?process",
            ],
            "system_info": [
                r"\bdmesg\b",
                r"(?:kernel|system)\s+(?:messages?|logs?|ring\s*buffer)",
                r"\bjournalctl\b",
                r"(?:show|view|display)\s+(?:kernel|system)\s+",
                r"(?:using|with)\s+dmesg",
            ],
        }
    
    def _extract_path(self, text: str) -> Optional[str]:
        """Extract file/directory path from text."""
        # Quoted path
        match = re.search(r'["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
        
        # Path-like pattern (called/named X)
        match = re.search(r'(?:called|named)\s+([/\w._-]+)', text)
        if match:
            return match.group(1)
        
        # "of X" pattern for "contents of readme.md"
        match = re.search(r'(?:of|file)\s+([\w._/-]+\.\w+)', text)
        if match:
            return match.group(1)
        
        # Filename with extension (but not URLs)
        if 'http' not in text:
            match = re.search(r'\b([\w._-]+\.\w{1,5})\b', text)
            if match:
                return match.group(1)
        
        # Directory-like (contains /)
        match = re.search(r'\b([/\w._-]+/[\w._-]+)\b', text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_search_term(self, text: str) -> Optional[str]:
        """Extract search term from text."""
        # Quoted term
        match = re.search(r'["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
        
        # "for X" pattern
        match = re.search(r'(?:for|containing)\s+(\w+)', text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text."""
        match = re.search(r'(https?://[^\s]+)', text)
        if match:
            return match.group(1)
        return None
    
    def _detect_intents(self, text: str) -> List[str]:
        """Detect user intents from text."""
        text_lower = text.lower()
        detected = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected.append(intent)
                    break
        
        return detected
    
    def _query_bash_knowledge(self, keywords: List[str], 
                              top_k: int = 5) -> List[Tuple[float, KnowledgeEntry]]:
        """Query knowledge base for bash-related entries."""
        # Add bash-specific keywords
        bash_keywords = keywords + ["bash", "shell"]
        results = self.manager.query(
            bash_keywords, 
            domain=KnowledgeDomain.PROGRAMMING, 
            top_k=top_k
        )
        
        # Filter to bash entries only
        bash_results = []
        for sim, entry in results:
            if "bash" in entry.keywords or "shell" in entry.keywords:
                bash_results.append((sim, entry))
        
        return bash_results
    
    def _try_learned_intent(self, request: str) -> Optional[BashGenerationResult]:
        """
        Try to resolve the request using learned intents from the knowledge base.
        
        This is checked FIRST, before hardcoded patterns, allowing the LCM
        to use newly learned commands without code changes.
        """
        result = self.intent_manager.get_command_for_request(request, StepType.BASH)
        
        if result:
            command, confidence, intent = result
            
            # Only use if confidence is high enough
            if confidence >= 0.7:
                return BashGenerationResult(
                    success=True,
                    command=command,
                    explanation=intent.description,
                    knowledge_used=intent.target_commands,
                    confidence=confidence,
                    warnings=[],
                    from_learned_intent=True
                )
        
        return None
    
    def reload_intents(self):
        """Reload intents from knowledge base (call after learning new commands)."""
        self.intent_manager = IntentManager(storage_dir=self.storage_dir)
    
    def generate(self, request: str) -> BashGenerationResult:
        """Generate Bash command from natural language request."""
        
        # FIRST: Check learned intents from knowledge base
        # This allows newly learned commands to work without code changes
        intent_result = self._try_learned_intent(request)
        if intent_result:
            return intent_result
        
        # FALLBACK: Use hardcoded intent patterns
        intents = self._detect_intents(request)
        path = self._extract_path(request)
        search_term = self._extract_search_term(request)
        url = self._extract_url(request)
        
        command = None
        explanation = ""
        knowledge_used = []
        warnings = []
        confidence = 0.8
        
        # Handle specific intents
        if "create_dir" in intents:
            path = path or "new_directory"
            command = f"mkdir -p {path}"
            explanation = f"Create directory '{path}' (with parents if needed)"
            knowledge_used = ["mkdir", "create_directory_pattern"]
        
        elif "create_file" in intents:
            path = path or "newfile.txt"
            command = f"touch {path}"
            explanation = f"Create empty file '{path}'"
            knowledge_used = ["touch"]
        
        elif "delete_dir" in intents:
            if path:
                command = f"rm -rf {path}"
                explanation = f"Delete directory '{path}' and all contents"
                warnings = ["This will permanently delete the directory!"]
            else:
                command = "rm -rf <directory>"
                explanation = "Delete directory (specify path)"
                confidence = 0.5
            knowledge_used = ["rm"]
        
        elif "delete_file" in intents:
            if path:
                command = f"rm {path}"
                explanation = f"Delete file '{path}'"
            else:
                command = "rm <filename>"
                explanation = "Delete file (specify filename)"
                confidence = 0.5
            knowledge_used = ["rm"]
        
        elif "copy_file" in intents:
            if path:
                command = f"cp {path} {path}.backup"
                explanation = f"Copy '{path}' to '{path}.backup'"
            else:
                command = "cp <source> <destination>"
                explanation = "Copy file (specify source and destination)"
                confidence = 0.5
            knowledge_used = ["cp"]
        
        elif "move_file" in intents:
            if path:
                command = f"mv {path} <destination>"
                explanation = f"Move/rename '{path}'"
                confidence = 0.6
            else:
                command = "mv <source> <destination>"
                explanation = "Move/rename file (specify source and destination)"
                confidence = 0.5
            knowledge_used = ["mv"]
        
        elif "list_files" in intents:
            path = path or "."
            command = f"ls -la {path}"
            explanation = f"List all files in '{path}' with details"
            knowledge_used = ["ls"]
        
        elif "view_file" in intents:
            # Extract filename from "contents of X" or "view X"
            file_match = re.search(r'(?:contents?\s+of|view|read|display)\s+(?:the\s+)?(?:file\s+)?([\w._/-]+\.\w+)', request)
            target = file_match.group(1) if file_match else path
            if target:
                command = f"cat {target}"
                explanation = f"Display contents of '{target}'"
            else:
                command = "cat <filename>"
                explanation = "Display file contents (specify filename)"
                confidence = 0.5
            knowledge_used = ["cat"]
        
        elif "search_text" in intents:
            term = search_term or "<pattern>"
            # Don't use extracted path for search - it might be the search term
            command = f"grep -r '{term}' ."
            explanation = f"Search for '{term}' in current directory"
            knowledge_used = ["grep", "search_in_files_pattern"]
        
        elif "download" in intents:
            if url:
                command = f"curl -O {url}"
                explanation = f"Download file from '{url}'"
            else:
                command = "curl -O <url>"
                explanation = "Download file (specify URL)"
                confidence = 0.5
            knowledge_used = ["curl", "download_file_pattern"]
        
        elif "permissions" in intents:
            if path:
                command = f"chmod +x {path}"
                explanation = f"Make '{path}' executable"
            else:
                command = "chmod +x <script>"
                explanation = "Make script executable (specify filename)"
                confidence = 0.5
            knowledge_used = ["chmod", "make_executable_pattern"]
        
        elif "find_files" in intents:
            # Extract file extension pattern like ".py"
            ext_match = re.search(r'\.(\w+)\s+files', request)
            if ext_match:
                pattern = f"*.{ext_match.group(1)}"
            else:
                pattern = search_term or "*"
            command = f"find . -name '{pattern}'"
            explanation = f"Find files matching '{pattern}'"
            knowledge_used = ["find"]
        
        elif "compress" in intents:
            # Extract folder name from "compress the X folder"
            folder_match = re.search(r'(?:compress|archive|pack)\s+(?:the\s+)?(\w+)\s+(?:folder|directory)?', request)
            target = folder_match.group(1) if folder_match else path
            if target:
                command = f"tar -czvf {target}.tar.gz {target}"
                explanation = f"Create compressed archive of '{target}'"
            else:
                command = "tar -czvf archive.tar.gz <files>"
                explanation = "Create compressed archive (specify files)"
                confidence = 0.5
            knowledge_used = ["tar"]
        
        elif "extract" in intents:
            if path:
                command = f"tar -xzvf {path}"
                explanation = f"Extract archive '{path}'"
            else:
                command = "tar -xzvf <archive>"
                explanation = "Extract archive (specify archive file)"
                confidence = 0.5
            knowledge_used = ["tar"]
        
        elif "system_info" in intents:
            # Handle system/kernel info commands
            request_lower = request.lower()
            if "dmesg" in request_lower:
                command = "dmesg"
                explanation = "Print kernel ring buffer messages"
                knowledge_used = ["dmesg"]
                confidence = 0.9
            elif "journalctl" in request_lower:
                command = "journalctl -xe"
                explanation = "View systemd journal logs"
                knowledge_used = ["journalctl"]
            else:
                # Default to dmesg for kernel messages
                command = "dmesg | tail -50"
                explanation = "Show recent kernel messages"
                knowledge_used = ["dmesg"]
        
        elif "process_info" in intents:
            # Handle process/file listing commands
            request_lower = request.lower()
            if "lsof" in request_lower:
                # Query knowledge base for lsof
                results = self._query_bash_knowledge(["lsof", "open", "files", "process"])
                for sim, entry in results:
                    if entry.name == "lsof":
                        # Use just the command name - the synopsis is too verbose
                        command = "lsof"
                        explanation = entry.description[:100]
                        knowledge_used = ["lsof"]
                        confidence = 0.9
                        break
                else:
                    command = "lsof"
                    explanation = "List open files"
                    knowledge_used = ["lsof"]
            elif "ps" in request_lower:
                command = "ps aux"
                explanation = "List running processes"
                knowledge_used = ["ps"]
            else:
                command = "ps aux"
                explanation = "List running processes"
                knowledge_used = ["ps"]
        
        elif "network" in intents:
            # Check for specific commands mentioned
            request_lower = request.lower()
            if "ifconfig" in request_lower:
                # Query knowledge base for ifconfig
                results = self._query_bash_knowledge(["ifconfig", "network", "interface"])
                if results and results[0][1].name == "ifconfig":
                    entry = results[0][1]
                    command = entry.metadata.get("code", "ifconfig")
                    explanation = entry.description[:100]
                    knowledge_used = ["ifconfig"]
                    confidence = 0.9
                else:
                    command = "ifconfig"
                    explanation = "Show network interface configuration"
                    knowledge_used = ["ifconfig"]
            elif "ip addr" in request_lower or "ip address" in request_lower:
                command = "ip addr"
                explanation = "Show IP addresses for all interfaces"
                knowledge_used = ["ip"]
            elif "ip link" in request_lower:
                command = "ip link"
                explanation = "Show network link status"
                knowledge_used = ["ip"]
            elif "ip route" in request_lower:
                command = "ip route"
                explanation = "Show routing table"
                knowledge_used = ["ip"]
            else:
                # Generic network query - check knowledge base
                results = self._query_bash_knowledge(["network", "interface", "ip"])
                if results:
                    best_entry = results[0][1]
                    command = best_entry.metadata.get("code", best_entry.metadata.get("syntax", "ip addr"))
                    explanation = best_entry.description[:100]
                    knowledge_used = [best_entry.name]
                    confidence = results[0][0]
                else:
                    command = "ip addr"
                    explanation = "Show network interface addresses"
                    knowledge_used = ["ip"]
        
        else:
            # Fallback: query knowledge base
            keywords = request.lower().split()
            results = self._query_bash_knowledge(keywords)
            
            if results:
                best_sim, best_entry = results[0]
                if best_entry.metadata.get("code"):
                    command = best_entry.metadata["code"]
                elif best_entry.metadata.get("example"):
                    command = best_entry.metadata["example"]
                else:
                    command = best_entry.metadata.get("syntax", f"# {best_entry.name}")
                
                explanation = best_entry.description
                knowledge_used = [best_entry.name]
                confidence = best_sim
            else:
                return BashGenerationResult(
                    success=False,
                    command="# Could not generate command",
                    explanation="No matching bash knowledge found",
                    knowledge_used=[],
                    confidence=0.0,
                    warnings=["Request not understood"]
                )
        
        return BashGenerationResult(
            success=True,
            command=command,
            explanation=explanation,
            knowledge_used=knowledge_used,
            confidence=confidence,
            warnings=warnings
        )


def demonstrate():
    """Demonstrate the bash generator."""
    
    print("=" * 70)
    print("BASH GENERATOR: Natural Language → Shell Commands")
    print("=" * 70)
    print()
    
    generator = BashGenerator()
    
    test_requests = [
        "create a directory called myproject",
        "create a new file called config.txt",
        "list all files in the current directory",
        "delete the file temp.txt",
        "copy file.txt to backup",
        "search for 'error' in all files",
        "download https://example.com/data.json",
        "make script.sh executable",
        "find all .py files",
        "compress the logs folder",
        "view the contents of readme.md",
        "create a folder called src/components",
    ]
    
    for request in test_requests:
        print(f"REQUEST: {request}")
        result = generator.generate(request)
        
        print(f"  Command: {result.command}")
        print(f"  Explanation: {result.explanation}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        if result.warnings:
            print(f"  ⚠️  Warnings: {result.warnings}")
        
        print()


if __name__ == "__main__":
    demonstrate()
