"""
Knowledge Acquisition Module for TruthSpace LCM

This module enables the LCM to identify knowledge gaps and acquire new knowledge
from various sources:
- Linux man pages (for shell commands)
- Python module documentation (for Python libraries)
- Web search (for current events, general knowledge)
- Wikipedia/Grokipedia (for historical facts)

The acquired knowledge is then automatically converted into knowledge entries
that can be stored in the knowledge base.
"""

import os
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry
from truthspace_lcm.core.intent_manager import IntentManager, StepType


class KnowledgeSource(Enum):
    """Sources for acquiring knowledge."""
    MAN_PAGE = "man_page"           # Linux man pages
    PYTHON_HELP = "python_help"     # Python help() and pydoc
    PYTHON_MODULE = "python_module" # Reading Python source code
    WEB_SEARCH = "web_search"       # DuckDuckGo or similar
    WIKIPEDIA = "wikipedia"         # Wikipedia/Grokipedia


@dataclass
class KnowledgeGap:
    """Represents a gap in the knowledge base."""
    query: str                      # What was being searched for
    keywords: List[str]             # Keywords that didn't match
    suggested_source: KnowledgeSource  # Best source to fill the gap
    confidence: float               # How confident we are this is a gap
    context: str                    # Additional context about the gap


@dataclass
class AcquiredKnowledge:
    """Raw knowledge acquired from a source."""
    source: KnowledgeSource
    name: str
    description: str
    syntax: str
    examples: List[str]
    keywords: List[str]
    raw_content: str
    metadata: Dict[str, Any]


class KnowledgeGapDetector:
    """
    Detects when the LCM doesn't have sufficient knowledge to complete a task.
    """
    
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager
        self.confidence_threshold = 0.3  # Below this = knowledge gap
    
    def detect_gap(self, request: str, domain: KnowledgeDomain = None) -> Optional[KnowledgeGap]:
        """
        Detect if there's a knowledge gap for the given request.
        
        Returns KnowledgeGap if gap detected, None otherwise.
        """
        # Extract keywords from request
        keywords = self._extract_keywords(request)
        
        # Query the knowledge base
        results = self.manager.query(keywords, domain=domain, top_k=5)
        
        # Check if we have good matches
        if not results:
            return self._create_gap(request, keywords, 0.0)
        
        best_score = results[0][0]
        
        if best_score < self.confidence_threshold:
            return self._create_gap(request, keywords, best_score)
        
        # Check if the best match actually covers what we need
        best_entry = results[0][1]
        coverage = self._check_coverage(request, best_entry)
        
        if coverage < 0.5:
            return self._create_gap(request, keywords, coverage)
        
        return None  # No gap detected
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'although', 'though', 'after', 'before',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'show', 'me', 'using', 'use', 'get', 'make', 'please'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return keywords
    
    def _check_coverage(self, request: str, entry: KnowledgeEntry) -> float:
        """Check how well an entry covers the request."""
        request_keywords = set(self._extract_keywords(request))
        entry_keywords = set(kw.lower() for kw in entry.keywords)
        
        if not request_keywords:
            return 1.0
        
        # Check overlap
        overlap = request_keywords & entry_keywords
        coverage = len(overlap) / len(request_keywords)
        
        # Also check if request keywords appear in description
        desc_lower = entry.description.lower()
        desc_matches = sum(1 for kw in request_keywords if kw in desc_lower)
        desc_coverage = desc_matches / len(request_keywords)
        
        return max(coverage, desc_coverage)
    
    def _create_gap(self, request: str, keywords: List[str], score: float) -> KnowledgeGap:
        """Create a KnowledgeGap object."""
        # Determine the best source based on keywords
        source = self._suggest_source(request, keywords)
        
        return KnowledgeGap(
            query=request,
            keywords=keywords,
            suggested_source=source,
            confidence=1.0 - score,  # Higher confidence = bigger gap
            context=f"Best match score: {score:.2f}"
        )
    
    def _suggest_source(self, request: str, keywords: List[str]) -> KnowledgeSource:
        """Suggest the best source to fill a knowledge gap."""
        request_lower = request.lower()
        
        # Linux command indicators
        linux_indicators = [
            'command', 'linux', 'bash', 'shell', 'terminal',
            'ifconfig', 'ip', 'netstat', 'ps', 'top', 'systemctl',
            'apt', 'yum', 'dnf', 'pacman', 'chmod', 'chown'
        ]
        if any(ind in request_lower for ind in linux_indicators):
            return KnowledgeSource.MAN_PAGE
        
        # Python indicators
        python_indicators = [
            'python', 'module', 'library', 'import', 'pip',
            'pandas', 'numpy', 'requests', 'flask', 'django'
        ]
        if any(ind in request_lower for ind in python_indicators):
            return KnowledgeSource.PYTHON_HELP
        
        # Historical indicators
        history_indicators = [
            'history', 'historical', 'war', 'president', 'king',
            'queen', 'ancient', 'century', 'year', 'born', 'died'
        ]
        if any(ind in request_lower for ind in history_indicators):
            return KnowledgeSource.WIKIPEDIA
        
        # Default to web search for current events or unknown
        return KnowledgeSource.WEB_SEARCH


class KnowledgeAcquirer:
    """
    Acquires knowledge from various sources.
    """
    
    def __init__(self):
        pass
    
    def acquire(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """
        Acquire knowledge to fill a gap.
        
        Routes to the appropriate acquisition method based on source.
        """
        if gap.suggested_source == KnowledgeSource.MAN_PAGE:
            return self._acquire_from_man_page(gap)
        elif gap.suggested_source == KnowledgeSource.PYTHON_HELP:
            return self._acquire_from_python_help(gap)
        elif gap.suggested_source == KnowledgeSource.PYTHON_MODULE:
            return self._acquire_from_python_module(gap)
        elif gap.suggested_source == KnowledgeSource.WEB_SEARCH:
            return self._acquire_from_web_search(gap)
        elif gap.suggested_source == KnowledgeSource.WIKIPEDIA:
            return self._acquire_from_wikipedia(gap)
        
        return None
    
    def _acquire_from_man_page(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """Acquire knowledge from Linux man pages."""
        # Extract the command name from keywords
        command = self._extract_command_name(gap)
        if not command:
            return None
        
        try:
            # Get man page content
            result = subprocess.run(
                ['man', command],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, 'MANPAGER': 'cat', 'PAGER': 'cat'}
            )
            
            if result.returncode != 0:
                # Try with section numbers
                for section in ['1', '8', '5']:
                    result = subprocess.run(
                        ['man', section, command],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env={**os.environ, 'MANPAGER': 'cat', 'PAGER': 'cat'}
                    )
                    if result.returncode == 0:
                        break
            
            if result.returncode != 0:
                return None
            
            raw_content = result.stdout
            
            # Parse the man page
            parsed = self._parse_man_page(command, raw_content)
            
            return AcquiredKnowledge(
                source=KnowledgeSource.MAN_PAGE,
                name=command,
                description=parsed['description'],
                syntax=parsed['syntax'],
                examples=parsed['examples'],
                keywords=parsed['keywords'],
                raw_content=raw_content[:5000],  # Limit size
                metadata={
                    'type': 'bash_command',
                    'man_section': parsed.get('section', '1'),
                    'options': parsed.get('options', [])
                }
            )
            
        except subprocess.TimeoutExpired:
            return None
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error acquiring man page for {command}: {e}")
            return None
    
    def _extract_command_name(self, gap: KnowledgeGap) -> Optional[str]:
        """Extract the likely command name from a knowledge gap."""
        # Common Linux commands that might be in the query
        common_commands = [
            'ifconfig', 'ip', 'netstat', 'ss', 'ping', 'traceroute',
            'curl', 'wget', 'ssh', 'scp', 'rsync', 'tar', 'gzip',
            'ps', 'top', 'htop', 'kill', 'killall', 'systemctl',
            'journalctl', 'dmesg', 'lsof', 'strace', 'ltrace',
            'awk', 'sed', 'grep', 'find', 'xargs', 'sort', 'uniq',
            'cut', 'tr', 'head', 'tail', 'less', 'more', 'cat',
            'ls', 'cd', 'pwd', 'mkdir', 'rmdir', 'rm', 'cp', 'mv',
            'chmod', 'chown', 'chgrp', 'ln', 'df', 'du', 'mount',
            'umount', 'fdisk', 'mkfs', 'fsck', 'dd', 'parted',
            'apt', 'apt-get', 'dpkg', 'yum', 'dnf', 'rpm', 'pacman',
            'useradd', 'userdel', 'usermod', 'groupadd', 'passwd',
            'crontab', 'at', 'nohup', 'screen', 'tmux', 'bg', 'fg',
            'iptables', 'ufw', 'firewalld', 'nmap', 'tcpdump',
            'docker', 'podman', 'kubectl', 'git', 'svn', 'make',
            'gcc', 'g++', 'python', 'python3', 'pip', 'pip3',
            'node', 'npm', 'yarn', 'java', 'javac', 'mvn', 'gradle',
            'uptime', 'free', 'vmstat', 'iostat', 'sar', 'mpstat',
            'w', 'who', 'whoami', 'id', 'groups', 'last', 'lastlog',
            'uname', 'hostname', 'hostnamectl', 'date', 'cal', 'timedatectl',
            'env', 'printenv', 'export', 'alias', 'history', 'type', 'which',
            'file', 'stat', 'touch', 'tee', 'wc', 'diff', 'patch', 'comm',
            'zip', 'unzip', 'bzip2', 'xz', 'zcat', 'zgrep', 'zless',
        ]
        
        query_lower = gap.query.lower()
        
        # Check if any known command is in the query
        for cmd in common_commands:
            if cmd in query_lower:
                return cmd
        
        # Check keywords - this is key for learn_command('uptime')
        for kw in gap.keywords:
            kw_lower = kw.lower()
            if kw_lower in common_commands:
                return kw_lower
            # Also accept the keyword directly if it looks like a command
            if len(kw_lower) <= 20 and kw_lower.isalnum():
                return kw_lower
        
        # Try to extract from patterns like "using X" or "with X"
        patterns = [
            r'using\s+(\w+)',
            r'with\s+(\w+)',
            r'run\s+(\w+)',
            r'execute\s+(\w+)',
            r'the\s+(\w+)\s+command'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential = match.group(1)
                # Verify it might be a command
                if potential in common_commands or len(potential) <= 15:
                    return potential
        
        return None
    
    def _parse_man_page(self, command: str, content: str) -> Dict[str, Any]:
        """Parse a man page into structured data."""
        result = {
            'description': '',
            'syntax': '',
            'examples': [],
            'keywords': [command, 'bash', 'shell', 'linux', 'command'],
            'options': [],
            'section': '1'
        }
        
        # Clean up the content (remove backspaces used for bold)
        content = re.sub(r'.\x08', '', content)
        
        lines = content.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            # Detect section headers (all caps, possibly with leading spaces)
            if re.match(r'^[A-Z][A-Z\s]+$', line.strip()):
                # Save previous section
                if current_section and section_content:
                    self._process_man_section(result, current_section, section_content)
                current_section = line.strip()
                section_content = []
            else:
                section_content.append(line)
        
        # Process last section
        if current_section and section_content:
            self._process_man_section(result, current_section, section_content)
        
        # If no description found, try to get first paragraph
        if not result['description']:
            # Look for NAME section pattern: "command - description"
            name_match = re.search(rf'{command}\s*[-–—]\s*(.+?)(?:\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
            if name_match:
                result['description'] = name_match.group(1).strip()[:200]
        
        # Add command-specific keywords
        if 'network' in content.lower():
            result['keywords'].extend(['network', 'networking'])
        if 'file' in content.lower():
            result['keywords'].extend(['file', 'files'])
        if 'process' in content.lower():
            result['keywords'].extend(['process', 'processes'])
        
        return result
    
    def _process_man_section(self, result: Dict, section: str, content: List[str]) -> None:
        """Process a section of the man page."""
        text = '\n'.join(content).strip()
        
        if section in ['NAME', 'DESCRIPTION']:
            # Get first meaningful paragraph
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 10:
                    if not result['description']:
                        result['description'] = para[:500]
                    break
        
        elif section == 'SYNOPSIS':
            result['syntax'] = text[:300]
        
        elif section == 'EXAMPLES':
            # Extract example commands
            examples = re.findall(r'^\s*[$#]\s*(.+)$', text, re.MULTILINE)
            if not examples:
                # Try to find lines that look like commands
                examples = re.findall(r'^\s{4,}(\S.+)$', text, re.MULTILINE)
            result['examples'] = examples[:5]
        
        elif section == 'OPTIONS':
            # Extract option flags
            options = re.findall(r'^\s*(-\w+|--[\w-]+)', text, re.MULTILINE)
            result['options'] = list(set(options))[:20]
    
    def _acquire_from_python_help(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """Acquire knowledge from Python help system."""
        # Extract module/function name
        module_name = self._extract_python_module(gap)
        if not module_name:
            return None
        
        try:
            # Use pydoc to get help
            result = subprocess.run(
                ['python3', '-m', 'pydoc', module_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0 or 'no Python documentation found' in result.stdout:
                return None
            
            raw_content = result.stdout
            parsed = self._parse_python_help(module_name, raw_content)
            
            return AcquiredKnowledge(
                source=KnowledgeSource.PYTHON_HELP,
                name=module_name,
                description=parsed['description'],
                syntax=parsed['syntax'],
                examples=parsed['examples'],
                keywords=parsed['keywords'],
                raw_content=raw_content[:5000],
                metadata={
                    'type': 'python_module',
                    'functions': parsed.get('functions', []),
                    'classes': parsed.get('classes', [])
                }
            )
            
        except Exception as e:
            print(f"Error acquiring Python help for {module_name}: {e}")
            return None
    
    def _extract_python_module(self, gap: KnowledgeGap) -> Optional[str]:
        """Extract Python module name from gap."""
        # Common Python modules
        common_modules = [
            'os', 'sys', 'json', 'csv', 're', 'math', 'random',
            'datetime', 'time', 'collections', 'itertools', 'functools',
            'pathlib', 'shutil', 'subprocess', 'threading', 'multiprocessing',
            'socket', 'http', 'urllib', 'requests', 'flask', 'django',
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
            'asyncio', 'aiohttp', 'pytest', 'unittest', 'logging',
            'argparse', 'configparser', 'sqlite3', 'pickle', 'hashlib'
        ]
        
        query_lower = gap.query.lower()
        
        for mod in common_modules:
            if mod in query_lower:
                return mod
        
        for kw in gap.keywords:
            if kw in common_modules:
                return kw
        
        return None
    
    def _parse_python_help(self, module_name: str, content: str) -> Dict[str, Any]:
        """Parse Python help output."""
        result = {
            'description': '',
            'syntax': f'import {module_name}',
            'examples': [],
            'keywords': [module_name, 'python', 'module', 'library'],
            'functions': [],
            'classes': []
        }
        
        # Get description from first paragraph
        lines = content.split('\n')
        in_description = False
        desc_lines = []
        
        for line in lines:
            if line.strip().startswith('DESCRIPTION'):
                in_description = True
                continue
            elif in_description:
                if line.strip() and line.strip().isupper():
                    break
                if line.strip():
                    desc_lines.append(line.strip())
                    if len(desc_lines) >= 3:
                        break
        
        result['description'] = ' '.join(desc_lines)[:500]
        
        # Extract function names
        functions = re.findall(r'^\s{4}(\w+)\(', content, re.MULTILINE)
        result['functions'] = list(set(functions))[:20]
        
        # Extract class names
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        result['classes'] = list(set(classes))[:10]
        
        return result
    
    def _acquire_from_python_module(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """Acquire knowledge by reading Python source code."""
        # This would read actual .py files - placeholder for now
        return None
    
    def _acquire_from_web_search(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """Acquire knowledge from web search."""
        # This would use DuckDuckGo API - placeholder for now
        # For now, return None and let the user know web search isn't implemented
        return None
    
    def _acquire_from_wikipedia(self, gap: KnowledgeGap) -> Optional[AcquiredKnowledge]:
        """Acquire knowledge from Wikipedia."""
        # This would use Wikipedia API - placeholder for now
        return None


class KnowledgeBuilder:
    """
    Builds knowledge entries from acquired knowledge.
    """
    
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager
    
    def build_entry(self, acquired: AcquiredKnowledge) -> Optional[KnowledgeEntry]:
        """
        Build and store a knowledge entry from acquired knowledge.
        """
        # Determine domain
        if acquired.source in [KnowledgeSource.MAN_PAGE]:
            domain = KnowledgeDomain.PROGRAMMING
            entry_type = "command"
        elif acquired.source in [KnowledgeSource.PYTHON_HELP, KnowledgeSource.PYTHON_MODULE]:
            domain = KnowledgeDomain.PROGRAMMING
            entry_type = "library"
        elif acquired.source == KnowledgeSource.WIKIPEDIA:
            # Could be history, science, etc. - default to general
            domain = KnowledgeDomain.GENERAL
            entry_type = "fact"
        else:
            domain = KnowledgeDomain.GENERAL
            entry_type = "fact"
        
        # Build metadata
        metadata = {
            'syntax': acquired.syntax,
            'example': acquired.examples[0] if acquired.examples else '',
            'examples': acquired.examples,
            'source': acquired.source.value,
            **acquired.metadata
        }
        
        # For bash commands, add code template
        if acquired.metadata.get('type') == 'bash_command':
            metadata['code'] = acquired.syntax.split('\n')[0] if acquired.syntax else acquired.name
            # Add bash/shell keywords
            keywords = list(set(acquired.keywords + ['bash', 'shell', 'command']))
        else:
            keywords = acquired.keywords
        
        # Create the entry
        entry = self.manager.create(
            name=acquired.name,
            domain=domain,
            entry_type=entry_type,
            description=acquired.description,
            keywords=keywords,
            metadata=metadata
        )
        
        return entry


class KnowledgeAcquisitionSystem:
    """
    Main system that orchestrates knowledge gap detection and acquisition.
    
    When learning new knowledge, this system also generates intent patterns
    so the LCM can use the new knowledge without code changes.
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        
        self.storage_dir = storage_dir
        self.manager = KnowledgeManager(storage_dir=storage_dir)
        self.intent_manager = IntentManager(storage_dir=storage_dir)
        self.detector = KnowledgeGapDetector(self.manager)
        self.acquirer = KnowledgeAcquirer()
        self.builder = KnowledgeBuilder(self.manager)
    
    def check_and_acquire(self, request: str, domain: KnowledgeDomain = None) -> Dict[str, Any]:
        """
        Check for knowledge gaps and attempt to fill them.
        
        Returns a dict with:
        - gap_detected: bool
        - gap: KnowledgeGap or None
        - acquired: AcquiredKnowledge or None
        - entry_created: KnowledgeEntry or None
        - message: str
        """
        result = {
            'gap_detected': False,
            'gap': None,
            'acquired': None,
            'entry_created': None,
            'message': ''
        }
        
        # Detect gap
        gap = self.detector.detect_gap(request, domain)
        
        if not gap:
            result['message'] = 'No knowledge gap detected'
            return result
        
        result['gap_detected'] = True
        result['gap'] = gap
        result['message'] = f'Knowledge gap detected: {gap.query}'
        
        # Try to acquire knowledge
        print(f"  🔍 Detected knowledge gap for: {gap.keywords}")
        print(f"  📚 Suggested source: {gap.suggested_source.value}")
        
        acquired = self.acquirer.acquire(gap)
        
        if not acquired:
            result['message'] = f'Could not acquire knowledge from {gap.suggested_source.value}'
            return result
        
        result['acquired'] = acquired
        print(f"  ✅ Acquired knowledge about: {acquired.name}")
        
        # Build and store entry
        entry = self.builder.build_entry(acquired)
        
        if entry:
            result['entry_created'] = entry
            result['message'] = f'Successfully learned about: {entry.name}'
            print(f"  💾 Created knowledge entry: {entry.id}")
            
            # Also create intent pattern for the new knowledge
            intent = self._create_intent_for_entry(entry, acquired)
            if intent:
                result['intent_created'] = intent
                print(f"  🎯 Created intent pattern: {intent.name}")
        
        return result
    
    def _create_intent_for_entry(self, entry: KnowledgeEntry, acquired: AcquiredKnowledge):
        """Create an intent pattern for a newly learned knowledge entry."""
        # Determine step type based on source
        if acquired.source == KnowledgeSource.MAN_PAGE:
            step_type = StepType.BASH
        elif acquired.source in [KnowledgeSource.PYTHON_HELP, KnowledgeSource.PYTHON_MODULE]:
            step_type = StepType.PYTHON
        else:
            step_type = StepType.EITHER
        
        try:
            intent = self.intent_manager.create_intent_for_command(
                command_name=entry.name,
                description=entry.description,
                keywords=entry.keywords,
                step_type=step_type,
                knowledge_entry_id=entry.id
            )
            return intent
        except Exception as e:
            print(f"  ⚠️ Could not create intent: {e}")
            return None
    
    def learn_command(self, command: str) -> Optional[KnowledgeEntry]:
        """
        Convenience method to learn about a specific Linux command.
        Also creates an intent pattern for the command.
        """
        gap = KnowledgeGap(
            query=f"how to use {command}",
            keywords=[command],
            suggested_source=KnowledgeSource.MAN_PAGE,
            confidence=1.0,
            context="Direct command learning"
        )
        
        acquired = self.acquirer.acquire(gap)
        if acquired:
            entry = self.builder.build_entry(acquired)
            if entry:
                # Also create intent
                self._create_intent_for_entry(entry, acquired)
            return entry
        return None
    
    def learn_module(self, module: str) -> Optional[KnowledgeEntry]:
        """
        Convenience method to learn about a specific Python module.
        Also creates an intent pattern for the module.
        """
        gap = KnowledgeGap(
            query=f"how to use {module} module",
            keywords=[module],
            suggested_source=KnowledgeSource.PYTHON_HELP,
            confidence=1.0,
            context="Direct module learning"
        )
        
        acquired = self.acquirer.acquire(gap)
        if acquired:
            entry = self.builder.build_entry(acquired)
            if entry:
                # Also create intent
                self._create_intent_for_entry(entry, acquired)
            return entry
        return None


def demonstrate():
    """Demonstrate the knowledge acquisition system."""
    
    print("=" * 70)
    print("KNOWLEDGE ACQUISITION SYSTEM")
    print("=" * 70)
    print()
    
    system = KnowledgeAcquisitionSystem()
    
    # Test 1: Learn about ifconfig
    print("-" * 70)
    print("TEST 1: Learning about 'ifconfig' command")
    print("-" * 70)
    
    entry = system.learn_command('ifconfig')
    if entry:
        print(f"\n  Created entry:")
        print(f"    Name: {entry.name}")
        print(f"    Description: {entry.description[:100]}...")
        print(f"    Keywords: {entry.keywords}")
        print(f"    Syntax: {entry.metadata.get('syntax', 'N/A')[:100]}...")
    else:
        print("  Failed to learn about ifconfig")
    
    # Test 2: Check and acquire for unknown request
    print("\n" + "-" * 70)
    print("TEST 2: Check and acquire for 'show network interfaces using ip'")
    print("-" * 70)
    
    result = system.check_and_acquire("show network interfaces using ip")
    print(f"\n  Result: {result['message']}")
    if result['entry_created']:
        print(f"  Entry: {result['entry_created'].name}")
    
    # Test 3: Learn about a Python module
    print("\n" + "-" * 70)
    print("TEST 3: Learning about 'socket' module")
    print("-" * 70)
    
    entry = system.learn_module('socket')
    if entry:
        print(f"\n  Created entry:")
        print(f"    Name: {entry.name}")
        print(f"    Description: {entry.description[:100]}...")
        print(f"    Keywords: {entry.keywords}")
    else:
        print("  Failed to learn about socket module")
    
    print("\n" + "=" * 70)
    print("Knowledge acquisition demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate()
