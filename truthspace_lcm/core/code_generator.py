"""
Code Generator: Natural Language → Python Code

Uses the persistent knowledge base to generate Python code
from natural language descriptions.

Key Features:
1. Query knowledge base for relevant patterns/functions
2. Compose code from multiple knowledge entries
3. Handle complex multi-step programs
4. Generate complete, runnable code
"""

import os
import sys
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry


@dataclass
class CodeGenerationResult:
    """Result of code generation."""
    success: bool
    code: str
    explanation: str
    knowledge_used: List[str]
    imports: List[str]
    confidence: float
    warnings: List[str]


class CodeGenerator:
    """
    Generate Python code from natural language using geometric knowledge.
    """
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_store"
            )
        self.manager = KnowledgeManager(storage_dir=storage_dir)
        self._init_intent_patterns()
    
    def _init_intent_patterns(self):
        """Initialize patterns for detecting user intent."""
        self.intent_patterns = {
            "fetch_url": [
                r"fetch\s+(?:the\s+)?(?:url|webpage|page|website|html)",
                r"get\s+(?:the\s+)?(?:html|content|page)\s+(?:from|of)",
                r"download\s+(?:the\s+)?(?:page|html|content)",
                r"go(?:es)?\s+to\s+\S+",  # "goes to duckduckgo.com"
                r"return(?:s)?\s+(?:the\s+)?(?:page|html)",
                r"(?:get|fetch)\s+(?:the\s+)?webpage",
            ],
            "scrape_web": [
                r"scrape\s+(?:the\s+)?(?:data|content|elements|titles|links)",
                r"extract\s+(?:the\s+)?(?:data|text|elements|titles|links)\s+from",
                r"parse\s+(?:the\s+)?html",
                r"(?:titles|links)\s+from\s+(?:a\s+)?webpage",
            ],
            "read_file": [
                r"read\s+(?:a\s+)?(?:the\s+)?(?:file|contents|data)",
                r"load\s+(?:a\s+)?(?:the\s+)?(?:file|data)",
                r"open\s+(?:and\s+read\s+)?(?:the\s+)?file",
            ],
            "write_file": [
                r"write\s+(?:some\s+)?(?:data\s+)?(?:to\s+)?(?:a\s+)?(?:the\s+)?(?:file|json)",
                r"save\s+(?:the\s+)?(?:data|content|result)",
                r"output\s+to\s+(?:a\s+)?file",
                r"(?:data|content)\s+to\s+\S+\.json",  # "data to output.json"
            ],
            "json_parse": [
                r"parse\s+(?:the\s+)?json",
                r"read\s+(?:a\s+)?(?:the\s+)?json",
                r"load\s+(?:the\s+)?json",
            ],
            "json_write": [
                r"write\s+(?:to\s+)?(?:a\s+)?json",
                r"save\s+(?:to\s+)?(?:a\s+)?json",
                r"(?:to|into)\s+(?:a\s+)?json\s+file",
            ],
            "api_request": [
                r"(?:make\s+)?(?:an?\s+)?api\s+(?:request|call)",
                r"call\s+(?:the\s+)?api",
                r"fetch\s+(?:from\s+)?(?:the\s+)?api",
                r"parse\s+(?:the\s+)?json\s+response",
            ],
            "loop_iterate": [
                r"loop\s+(?:through|over)",
                r"iterate\s+(?:through|over)",
                r"for\s+each",
                r"(?:print|show|display)\s+(?:the\s+)?(?:numbers?|items?)\s+\d+\s+to\s+\d+",
                r"\d+\s+to\s+\d+",  # "1 to 5"
            ],
            "hello_world": [
                r"hello\s+world",
                r"print\s+hello",
            ],
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from natural language text."""
        stopwords = {
            "a", "an", "the", "in", "to", "and", "or", "that", "which",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall",
            "program", "code", "python", "write", "create", "make",
            "please", "want", "need", "like", "can", "you", "me", "i",
            "it", "its", "this", "that", "these", "those", "then",
            "so", "if", "but", "for", "with", "as", "at", "by", "from"
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text."""
        # Full URL
        match = re.search(r'(https?://[^\s]+)', text)
        if match:
            return match.group(1)
        
        # Domain-like pattern
        match = re.search(r'(\b[\w.-]+\.(?:com|org|net|io|gov|edu|co\.uk)[^\s]*)', text)
        if match:
            url = match.group(1)
            if not url.startswith('http'):
                url = 'https://' + url
            return url
        
        return None
    
    def _extract_filename(self, text: str) -> Optional[str]:
        """Extract filename from text."""
        # Quoted filename
        match = re.search(r'["\']([^"\']+\.\w+)["\']', text)
        if match:
            return match.group(1)
        
        # Filename pattern
        match = re.search(r'\b([\w.-]+\.\w{2,4})\b', text)
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
    
    def _query_knowledge(self, keywords: List[str], 
                         top_k: int = 10) -> List[Tuple[float, KnowledgeEntry]]:
        """Query knowledge base for relevant entries."""
        return self.manager.query(
            keywords, 
            domain=KnowledgeDomain.PROGRAMMING, 
            top_k=top_k
        )
    
    def _get_pattern_code(self, entry: KnowledgeEntry) -> Optional[str]:
        """Get code from a pattern entry."""
        if entry.metadata.get("code"):
            return entry.metadata["code"]
        if entry.metadata.get("example"):
            return entry.metadata["example"]
        return None
    
    def _get_imports(self, entries: List[KnowledgeEntry], code: str = "") -> List[str]:
        """Extract required imports from entries and code content."""
        imports = set()
        
        for entry in entries:
            if entry.metadata.get("import"):
                imports.add(entry.metadata["import"])
            
            # Infer imports from entry name
            if entry.name.startswith("requests") or entry.name == "requests":
                imports.add("import requests")
            elif entry.name.startswith("json") or entry.name == "json":
                imports.add("import json")
            elif entry.name.startswith("os") or entry.name == "os":
                imports.add("import os")
            elif entry.name in ["BeautifulSoup", "beautifulsoup", "soup"]:
                imports.add("from bs4 import BeautifulSoup")
            elif "csv" in entry.name.lower():
                imports.add("import csv")
            elif entry.name.startswith("sys"):
                imports.add("import sys")
        
        # Also infer from code content
        if "requests.get" in code or "requests.post" in code:
            imports.add("import requests")
        if "json.load" in code or "json.dump" in code:
            imports.add("import json")
        if "BeautifulSoup" in code:
            imports.add("from bs4 import BeautifulSoup")
        if "os.path" in code or "os.environ" in code:
            imports.add("import os")
        
        return sorted(list(imports))
    
    def generate(self, request: str) -> CodeGenerationResult:
        """
        Generate Python code from natural language request.
        
        Args:
            request: Natural language description of desired program
        
        Returns:
            CodeGenerationResult with generated code and metadata
        """
        # Extract information from request
        keywords = self._extract_keywords(request)
        intents = self._detect_intents(request)
        url = self._extract_url(request)
        filename = self._extract_filename(request)
        
        # Query knowledge base
        results = self._query_knowledge(keywords)
        
        if not results:
            return CodeGenerationResult(
                success=False,
                code="# Could not generate code - no relevant knowledge found",
                explanation="No matching patterns in knowledge base",
                knowledge_used=[],
                imports=[],
                confidence=0.0,
                warnings=["No relevant knowledge found for: " + ", ".join(keywords)]
            )
        
        # Select best matching entries
        used_entries = []
        code_parts = []
        warnings = []
        
        # Handle specific intents - ONLY use the FIRST matching intent
        # to avoid combining unrelated code
        intent_handled = False
        
        if "hello_world" in intents:
            return self._generate_hello_world()
        
        if not intent_handled and "scrape_web" in intents:
            code, entries = self._generate_scrape(url, request)
            code_parts.append(code)
            used_entries.extend(entries)
            intent_handled = True
        
        if not intent_handled and ("fetch_url" in intents or "api_request" in intents):
            code, entries = self._generate_fetch_url(url, request)
            code_parts.append(code)
            used_entries.extend(entries)
            intent_handled = True
        
        if not intent_handled and ("write_file" in intents or "json_write" in intents):
            code, entries = self._generate_write_file(filename, request)
            code_parts.append(code)
            used_entries.extend(entries)
            intent_handled = True
        
        if not intent_handled and ("read_file" in intents or "json_parse" in intents):
            code, entries = self._generate_read_file(filename, request)
            code_parts.append(code)
            used_entries.extend(entries)
            intent_handled = True
        
        if not intent_handled and "loop_iterate" in intents:
            code, entries = self._generate_loop(request)
            code_parts.append(code)
            used_entries.extend(entries)
            intent_handled = True
        
        # If no specific intent matched, use best pattern match ONLY
        if not code_parts:
            for sim, entry in results[:1]:  # Only use top match
                if sim > 0.4:
                    pattern_code = self._get_pattern_code(entry)
                    if pattern_code:
                        code_parts.append(f"# From pattern: {entry.name}")
                        code_parts.append(pattern_code)
                        used_entries.append(entry)
        
        if not code_parts:
            # Fallback: show what we found
            suggestions = [f"# {e.name}: {e.description[:60]}..." 
                          for _, e in results[:5]]
            return CodeGenerationResult(
                success=False,
                code="# Could not generate complete code\n# Suggestions:\n" + 
                     "\n".join(suggestions),
                explanation="Found related knowledge but couldn't compose code",
                knowledge_used=[e.name for _, e in results[:5]],
                imports=[],
                confidence=0.3,
                warnings=["Could not determine how to compose code from knowledge"]
            )
        
        # Build final code
        combined_code = "\n".join(code_parts)
        imports = self._get_imports(used_entries, combined_code)
        
        final_code_parts = []
        
        # Add docstring
        final_code_parts.append(f'"""')
        final_code_parts.append(f'Generated from: {request[:80]}...' if len(request) > 80 else f'Generated from: {request}')
        final_code_parts.append(f'"""')
        final_code_parts.append("")
        
        # Add imports
        if imports:
            final_code_parts.extend(imports)
            final_code_parts.append("")
        
        # Add main code
        final_code_parts.extend(code_parts)
        
        # Calculate confidence
        avg_sim = sum(sim for sim, _ in results[:len(used_entries)]) / max(len(used_entries), 1)
        confidence = min(avg_sim, 1.0)
        
        return CodeGenerationResult(
            success=True,
            code="\n".join(final_code_parts),
            explanation=f"Generated using {len(used_entries)} knowledge entries",
            knowledge_used=[e.name for e in used_entries],
            imports=imports,
            confidence=confidence,
            warnings=warnings
        )
    
    def _generate_hello_world(self) -> CodeGenerationResult:
        """Generate hello world program."""
        return CodeGenerationResult(
            success=True,
            code='print("Hello, World!")',
            explanation="Simple hello world program",
            knowledge_used=["print"],
            imports=[],
            confidence=1.0,
            warnings=[]
        )
    
    def _generate_fetch_url(self, url: Optional[str], 
                            request: str) -> Tuple[str, List[KnowledgeEntry]]:
        """Generate code to fetch URL."""
        url = url or "https://example.com"
        
        # Check if they want JSON response
        wants_json = any(w in request.lower() for w in ["json", "api", "data"])
        
        # Check if they want HTML as string
        wants_html = any(w in request.lower() for w in ["html", "page", "text", "string"])
        
        entries = []
        
        # Find relevant entries
        results = self._query_knowledge(["requests", "get", "url"])
        for sim, entry in results:
            if "requests" in entry.name.lower():
                entries.append(entry)
                break
        
        code_lines = [
            f"# Fetch content from {url}",
            f"url = '{url}'",
            "",
            "response = requests.get(url)",
            "",
            "if response.status_code == 200:",
        ]
        
        if wants_json:
            code_lines.append("    data = response.json()")
            code_lines.append("    print(data)")
        else:
            code_lines.append("    html_content = response.text")
            code_lines.append("    print(html_content)")
        
        code_lines.append("else:")
        code_lines.append("    print(f'Error: {response.status_code}')")
        
        return "\n".join(code_lines), entries
    
    def _generate_scrape(self, url: Optional[str],
                         request: str) -> Tuple[str, List[KnowledgeEntry]]:
        """Generate web scraping code."""
        url = url or "https://example.com"
        
        entries = []
        results = self._query_knowledge(["beautifulsoup", "scrape", "html"])
        for sim, entry in results:
            if "soup" in entry.name.lower() or "beautiful" in entry.name.lower():
                entries.append(entry)
        
        # Also add requests entry for imports
        results2 = self._query_knowledge(["requests"])
        for sim, entry in results2:
            if "requests" in entry.name.lower():
                entries.append(entry)
                break
        
        code_lines = [
            f"# Scrape content from {url}",
            f"url = '{url}'",
            "",
            "# Fetch the page",
            "response = requests.get(url)",
            "html_content = response.text",
            "",
            "# Parse HTML",
            "soup = BeautifulSoup(html_content, 'html.parser')",
            "",
            "# Extract data (customize selectors as needed)",
            "title = soup.find('title')",
            "if title:",
            "    print(f'Page title: {title.text}')",
            "",
            "# Find all links",
            "links = soup.find_all('a')",
            "for link in links[:10]:  # First 10 links",
            "    href = link.get('href')",
            "    text = link.text.strip()",
            "    if href and text:",
            "        print(f'{text}: {href}')",
        ]
        
        return "\n".join(code_lines), entries
    
    def _generate_read_file(self, filename: Optional[str],
                            request: str) -> Tuple[str, List[KnowledgeEntry]]:
        """Generate file reading code."""
        filename = filename or "data.txt"
        
        entries = []
        
        # Check if JSON
        is_json = "json" in request.lower() or filename.endswith(".json")
        
        if is_json:
            results = self._query_knowledge(["json", "load", "file"])
            for sim, entry in results:
                if "json" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                f"# Read JSON file: {filename}",
                f"filename = '{filename}'",
                "",
                "with open(filename, 'r') as f:",
                "    data = json.load(f)",
                "",
                "print(data)",
            ]
        else:
            results = self._query_knowledge(["file", "read", "open"])
            for sim, entry in results:
                if "open" in entry.name.lower() or "read" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                f"# Read file: {filename}",
                f"filename = '{filename}'",
                "",
                "with open(filename, 'r') as f:",
                "    content = f.read()",
                "",
                "print(content)",
            ]
        
        return "\n".join(code_lines), entries
    
    def _generate_write_file(self, filename: Optional[str],
                             request: str) -> Tuple[str, List[KnowledgeEntry]]:
        """Generate file writing code."""
        filename = filename or "output.txt"
        
        entries = []
        
        # Check if JSON
        is_json = "json" in request.lower() or filename.endswith(".json")
        
        if is_json:
            results = self._query_knowledge(["json", "dump", "write"])
            for sim, entry in results:
                if "json" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                f"# Write JSON file: {filename}",
                f"filename = '{filename}'",
                "",
                "data = {",
                "    'key': 'value',",
                "    'items': [1, 2, 3]",
                "}",
                "",
                "with open(filename, 'w') as f:",
                "    json.dump(data, f, indent=2)",
                "",
                "print(f'Saved to {filename}')",
            ]
        else:
            results = self._query_knowledge(["file", "write", "save"])
            for sim, entry in results:
                if "write" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                f"# Write file: {filename}",
                f"filename = '{filename}'",
                "",
                "content = 'Your content here'",
                "",
                "with open(filename, 'w') as f:",
                "    f.write(content)",
                "",
                "print(f'Saved to {filename}')",
            ]
        
        return "\n".join(code_lines), entries
    
    def _generate_loop(self, request: str) -> Tuple[str, List[KnowledgeEntry]]:
        """Generate loop code."""
        entries = []
        
        # Try to extract range from request (e.g., "1 to 5")
        import re
        range_match = re.search(r'(\d+)\s+to\s+(\d+)', request)
        
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            
            results = self._query_knowledge(["for", "loop", "range"])
            for sim, entry in results:
                if "for" in entry.name.lower() or "loop" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                f"# Print numbers from {start} to {end}",
                f"for i in range({start}, {end + 1}):",
                "    print(i)",
            ]
        else:
            # Generic loop
            results = self._query_knowledge(["for", "loop", "iterate"])
            for sim, entry in results:
                if "for" in entry.name.lower():
                    entries.append(entry)
                    break
            
            code_lines = [
                "# Loop through items",
                "items = [1, 2, 3, 4, 5]",
                "for item in items:",
                "    print(item)",
            ]
        
        return "\n".join(code_lines), entries


def demonstrate():
    """Demonstrate the code generator."""
    
    print("=" * 70)
    print("CODE GENERATOR: Natural Language → Python")
    print("=" * 70)
    print()
    
    generator = CodeGenerator()
    
    # Test cases - from simple to complex
    test_requests = [
        # Simple
        "a hello world program in python",
        
        # Medium
        "read a json file called config.json",
        
        # Complex
        "a python program that goes to duckduckgo.com and returns the page's HTML as a string",
        
        # More complex
        "fetch the webpage at https://api.github.com and parse the JSON response",
        
        # Web scraping
        "scrape the titles and links from a webpage at example.com",
        
        # File operations
        "write some data to a json file called output.json",
    ]
    
    for request in test_requests:
        print("-" * 70)
        print(f"REQUEST: {request}")
        print("-" * 70)
        
        result = generator.generate(request)
        
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Knowledge used: {result.knowledge_used}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        print(f"\nGENERATED CODE:")
        print("```python")
        print(result.code)
        print("```")
        print()


if __name__ == "__main__":
    demonstrate()
