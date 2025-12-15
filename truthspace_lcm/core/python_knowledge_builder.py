"""
Python Knowledge Builder

Builds a comprehensive knowledge base for Python programming,
including common libraries, patterns, and idioms.

This populates the KnowledgeManager with detailed entries
that enable complex code generation from natural language.
"""

import os
import sys
from typing import Dict, List, Any

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry


class PythonKnowledgeBuilder:
    """Builds Python programming knowledge base."""
    
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager
    
    def build_all(self) -> int:
        """Build all Python knowledge. Returns count of entries created."""
        count = 0
        count += self._build_core_functions()
        count += self._build_requests_library()
        count += self._build_json_library()
        count += self._build_os_library()
        count += self._build_file_operations()
        count += self._build_string_operations()
        count += self._build_list_operations()
        count += self._build_control_flow()
        count += self._build_web_scraping()
        count += self._build_common_patterns()
        return count
    
    def _create(self, name: str, entry_type: str, description: str,
                keywords: List[str], metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """Helper to create programming knowledge entry."""
        return self.manager.create(
            name=name,
            domain=KnowledgeDomain.PROGRAMMING,
            entry_type=entry_type,
            description=description,
            keywords=keywords,
            metadata=metadata or {}
        )
    
    def _build_core_functions(self) -> int:
        """Build knowledge about Python built-in functions."""
        entries = [
            ("print", "function", 
             "Output text to console. Usage: print('message') or print(variable)",
             ["print", "output", "display", "console", "show", "text"],
             {"syntax": "print(*args, sep=' ', end='\\n')", 
              "example": "print('Hello, World!')"}),
            
            ("input", "function",
             "Get user input from console. Returns string.",
             ["input", "read", "user", "keyboard", "prompt", "ask"],
             {"syntax": "input(prompt)", "returns": "str",
              "example": "name = input('Enter your name: ')"}),
            
            ("len", "function",
             "Get length of a sequence (string, list, dict, etc.)",
             ["length", "size", "count", "how many"],
             {"syntax": "len(obj)", "returns": "int",
              "example": "length = len(my_list)"}),
            
            ("range", "function",
             "Generate sequence of numbers for iteration",
             ["range", "sequence", "numbers", "loop", "iterate"],
             {"syntax": "range(stop) or range(start, stop, step)",
              "example": "for i in range(10):"}),
            
            ("open", "function",
             "Open a file for reading or writing",
             ["open", "file", "read", "write", "disk"],
             {"syntax": "open(filename, mode)", "returns": "file object",
              "example": "with open('file.txt', 'r') as f:"}),
            
            ("int", "function",
             "Convert value to integer",
             ["integer", "convert", "number", "parse"],
             {"syntax": "int(x)", "returns": "int",
              "example": "num = int('42')"}),
            
            ("str", "function",
             "Convert value to string",
             ["string", "convert", "text"],
             {"syntax": "str(x)", "returns": "str",
              "example": "text = str(42)"}),
            
            ("float", "function",
             "Convert value to floating point number",
             ["float", "decimal", "convert", "number"],
             {"syntax": "float(x)", "returns": "float",
              "example": "num = float('3.14')"}),
            
            ("list", "function",
             "Create a list or convert iterable to list",
             ["list", "array", "collection", "convert"],
             {"syntax": "list(iterable)", "returns": "list",
              "example": "my_list = list(range(10))"}),
            
            ("dict", "function",
             "Create a dictionary",
             ["dictionary", "dict", "map", "key", "value", "hash"],
             {"syntax": "dict(**kwargs) or dict(iterable)",
              "example": "d = {'key': 'value'}"}),
            
            ("sorted", "function",
             "Return sorted list from iterable",
             ["sort", "order", "arrange", "ascending", "descending"],
             {"syntax": "sorted(iterable, key=None, reverse=False)",
              "example": "sorted_list = sorted(my_list)"}),
            
            ("enumerate", "function",
             "Add index to iteration",
             ["enumerate", "index", "number", "loop", "count"],
             {"syntax": "enumerate(iterable, start=0)",
              "example": "for i, item in enumerate(my_list):"}),
            
            ("zip", "function",
             "Combine multiple iterables element-wise",
             ["zip", "combine", "pair", "parallel", "iterate"],
             {"syntax": "zip(*iterables)",
              "example": "for a, b in zip(list1, list2):"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_requests_library(self) -> int:
        """Build knowledge about requests library."""
        entries = [
            ("requests", "library",
             "HTTP library for making web requests. Install: pip install requests",
             ["http", "web", "url", "api", "fetch", "download", "request"],
             {"import": "import requests", "install": "pip install requests"}),
            
            ("requests.get", "function",
             "Make HTTP GET request to fetch data from URL",
             ["get", "fetch", "download", "url", "http", "web", "html", "api"],
             {"syntax": "requests.get(url, params=None, headers=None)",
              "returns": "Response object",
              "example": "response = requests.get('https://api.example.com/data')"}),
            
            ("requests.post", "function",
             "Make HTTP POST request to send data to URL",
             ["post", "send", "submit", "upload", "http", "api", "form"],
             {"syntax": "requests.post(url, data=None, json=None)",
              "returns": "Response object",
              "example": "response = requests.post(url, json={'key': 'value'})"}),
            
            ("response.text", "attribute",
             "Get response body as string (HTML, text content)",
             ["text", "html", "content", "body", "string", "response"],
             {"returns": "str",
              "example": "html_content = response.text"}),
            
            ("response.json", "method",
             "Parse response body as JSON",
             ["json", "parse", "data", "api", "response"],
             {"syntax": "response.json()", "returns": "dict or list",
              "example": "data = response.json()"}),
            
            ("response.status_code", "attribute",
             "HTTP status code of response (200=OK, 404=Not Found, etc.)",
             ["status", "code", "http", "error", "success"],
             {"returns": "int",
              "example": "if response.status_code == 200:"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_json_library(self) -> int:
        """Build knowledge about json library."""
        entries = [
            ("json", "library",
             "Built-in library for JSON encoding/decoding",
             ["json", "parse", "serialize", "data", "api"],
             {"import": "import json"}),
            
            ("json.loads", "function",
             "Parse JSON string into Python object (dict/list)",
             ["json", "parse", "string", "decode", "load"],
             {"syntax": "json.loads(s)", "returns": "dict or list",
              "example": "data = json.loads(json_string)"}),
            
            ("json.dumps", "function",
             "Convert Python object to JSON string",
             ["json", "string", "encode", "serialize", "dump"],
             {"syntax": "json.dumps(obj, indent=None)", "returns": "str",
              "example": "json_str = json.dumps(data, indent=2)"}),
            
            ("json.load", "function",
             "Read and parse JSON from file",
             ["json", "file", "read", "load", "parse"],
             {"syntax": "json.load(fp)", "returns": "dict or list",
              "example": "with open('data.json') as f: data = json.load(f)"}),
            
            ("json.dump", "function",
             "Write Python object as JSON to file",
             ["json", "file", "write", "save", "dump"],
             {"syntax": "json.dump(obj, fp, indent=None)",
              "example": "with open('data.json', 'w') as f: json.dump(data, f)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_os_library(self) -> int:
        """Build knowledge about os library."""
        entries = [
            ("os", "library",
             "Operating system interface - files, directories, environment",
             ["os", "file", "directory", "path", "system", "environment"],
             {"import": "import os"}),
            
            ("os.path.exists", "function",
             "Check if file or directory exists",
             ["exists", "file", "check", "path", "directory"],
             {"syntax": "os.path.exists(path)", "returns": "bool",
              "example": "if os.path.exists('file.txt'):"}),
            
            ("os.path.join", "function",
             "Join path components with correct separator",
             ["path", "join", "directory", "file", "combine"],
             {"syntax": "os.path.join(path, *paths)", "returns": "str",
              "example": "full_path = os.path.join('dir', 'subdir', 'file.txt')"}),
            
            ("os.listdir", "function",
             "List files and directories in a path",
             ["list", "directory", "files", "folder", "contents"],
             {"syntax": "os.listdir(path)", "returns": "list",
              "example": "files = os.listdir('.')"}),
            
            ("os.makedirs", "function",
             "Create directory and all parent directories",
             ["create", "directory", "folder", "mkdir", "make"],
             {"syntax": "os.makedirs(path, exist_ok=False)",
              "example": "os.makedirs('path/to/dir', exist_ok=True)"}),
            
            ("os.remove", "function",
             "Delete a file",
             ["delete", "remove", "file", "unlink"],
             {"syntax": "os.remove(path)",
              "example": "os.remove('file.txt')"}),
            
            ("os.getcwd", "function",
             "Get current working directory",
             ["current", "directory", "cwd", "path", "working"],
             {"syntax": "os.getcwd()", "returns": "str",
              "example": "current_dir = os.getcwd()"}),
            
            ("os.environ", "attribute",
             "Access environment variables",
             ["environment", "variable", "env", "config"],
             {"returns": "dict-like",
              "example": "api_key = os.environ.get('API_KEY')"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_file_operations(self) -> int:
        """Build knowledge about file operations."""
        entries = [
            ("file.read", "method",
             "Read entire file contents as string",
             ["read", "file", "content", "text", "all"],
             {"syntax": "file.read()", "returns": "str",
              "example": "content = f.read()"}),
            
            ("file.readlines", "method",
             "Read file as list of lines",
             ["read", "lines", "file", "list"],
             {"syntax": "file.readlines()", "returns": "list",
              "example": "lines = f.readlines()"}),
            
            ("file.write", "method",
             "Write string to file",
             ["write", "file", "save", "output"],
             {"syntax": "file.write(s)",
              "example": "f.write('Hello, World!')"}),
            
            ("file.writelines", "method",
             "Write list of strings to file",
             ["write", "lines", "file", "list"],
             {"syntax": "file.writelines(lines)",
              "example": "f.writelines(lines)"}),
            
            ("with_open_pattern", "pattern",
             "Safe file handling with context manager",
             ["file", "open", "read", "write", "safe", "context"],
             {"example": "with open('file.txt', 'r') as f:\n    content = f.read()"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_string_operations(self) -> int:
        """Build knowledge about string operations."""
        entries = [
            ("str.split", "method",
             "Split string into list by separator",
             ["split", "divide", "separate", "words", "list"],
             {"syntax": "str.split(sep)", "returns": "list",
              "example": "words = text.split(' ')"}),
            
            ("str.join", "method",
             "Join list of strings with separator",
             ["join", "combine", "concatenate", "list", "string"],
             {"syntax": "sep.join(iterable)", "returns": "str",
              "example": "text = ', '.join(words)"}),
            
            ("str.strip", "method",
             "Remove whitespace from start and end",
             ["strip", "trim", "whitespace", "clean"],
             {"syntax": "str.strip()", "returns": "str",
              "example": "clean = text.strip()"}),
            
            ("str.replace", "method",
             "Replace occurrences of substring",
             ["replace", "substitute", "change", "swap"],
             {"syntax": "str.replace(old, new)", "returns": "str",
              "example": "new_text = text.replace('old', 'new')"}),
            
            ("str.lower", "method",
             "Convert string to lowercase",
             ["lower", "lowercase", "case"],
             {"syntax": "str.lower()", "returns": "str",
              "example": "lower_text = text.lower()"}),
            
            ("str.upper", "method",
             "Convert string to uppercase",
             ["upper", "uppercase", "case"],
             {"syntax": "str.upper()", "returns": "str",
              "example": "upper_text = text.upper()"}),
            
            ("str.startswith", "method",
             "Check if string starts with prefix",
             ["starts", "prefix", "begin", "check"],
             {"syntax": "str.startswith(prefix)", "returns": "bool",
              "example": "if text.startswith('http'):"}),
            
            ("str.endswith", "method",
             "Check if string ends with suffix",
             ["ends", "suffix", "check"],
             {"syntax": "str.endswith(suffix)", "returns": "bool",
              "example": "if filename.endswith('.txt'):"}),
            
            ("f-string", "pattern",
             "Format string with embedded expressions",
             ["format", "string", "interpolation", "template", "variable"],
             {"syntax": "f'text {variable} more text'",
              "example": "message = f'Hello, {name}!'"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_list_operations(self) -> int:
        """Build knowledge about list operations."""
        entries = [
            ("list.append", "method",
             "Add item to end of list",
             ["append", "add", "push", "insert", "end"],
             {"syntax": "list.append(item)",
              "example": "my_list.append(new_item)"}),
            
            ("list.extend", "method",
             "Add all items from another list",
             ["extend", "add", "combine", "merge"],
             {"syntax": "list.extend(iterable)",
              "example": "list1.extend(list2)"}),
            
            ("list.insert", "method",
             "Insert item at specific position",
             ["insert", "add", "position", "index"],
             {"syntax": "list.insert(index, item)",
              "example": "my_list.insert(0, first_item)"}),
            
            ("list.remove", "method",
             "Remove first occurrence of item",
             ["remove", "delete", "item"],
             {"syntax": "list.remove(item)",
              "example": "my_list.remove(item_to_remove)"}),
            
            ("list.pop", "method",
             "Remove and return item at index (default: last)",
             ["pop", "remove", "last", "index"],
             {"syntax": "list.pop(index=-1)", "returns": "item",
              "example": "last_item = my_list.pop()"}),
            
            ("list.index", "method",
             "Find index of first occurrence of item",
             ["index", "find", "position", "where"],
             {"syntax": "list.index(item)", "returns": "int",
              "example": "pos = my_list.index(item)"}),
            
            ("list_comprehension", "pattern",
             "Create list with inline loop and optional condition",
             ["comprehension", "list", "loop", "filter", "transform"],
             {"syntax": "[expr for item in iterable if condition]",
              "example": "squares = [x**2 for x in range(10)]"}),
            
            ("list_slicing", "pattern",
             "Extract portion of list using slice notation",
             ["slice", "subset", "portion", "range"],
             {"syntax": "list[start:stop:step]",
              "example": "first_three = my_list[:3]"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_control_flow(self) -> int:
        """Build knowledge about control flow."""
        entries = [
            ("if_statement", "pattern",
             "Conditional execution based on boolean expression",
             ["if", "condition", "check", "branch", "when"],
             {"syntax": "if condition:\n    ...\nelif other:\n    ...\nelse:\n    ...",
              "example": "if x > 0:\n    print('positive')"}),
            
            ("for_loop", "pattern",
             "Iterate over sequence of items",
             ["for", "loop", "iterate", "each", "repeat"],
             {"syntax": "for item in iterable:\n    ...",
              "example": "for item in my_list:\n    print(item)"}),
            
            ("while_loop", "pattern",
             "Repeat while condition is true",
             ["while", "loop", "repeat", "until", "condition"],
             {"syntax": "while condition:\n    ...",
              "example": "while count < 10:\n    count += 1"}),
            
            ("try_except", "pattern",
             "Handle exceptions/errors gracefully",
             ["try", "except", "error", "exception", "handle", "catch"],
             {"syntax": "try:\n    ...\nexcept ExceptionType as e:\n    ...",
              "example": "try:\n    result = risky_operation()\nexcept Exception as e:\n    print(f'Error: {e}')"}),
            
            ("function_def", "pattern",
             "Define a reusable function",
             ["def", "function", "define", "create", "method"],
             {"syntax": "def function_name(params):\n    ...\n    return result",
              "example": "def greet(name):\n    return f'Hello, {name}!'"}),
            
            ("class_def", "pattern",
             "Define a class with attributes and methods",
             ["class", "object", "define", "oop"],
             {"syntax": "class ClassName:\n    def __init__(self):\n        ...",
              "example": "class Person:\n    def __init__(self, name):\n        self.name = name"}),
            
            ("return_statement", "pattern",
             "Return value from function",
             ["return", "result", "output", "value"],
             {"syntax": "return value",
              "example": "return result"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_web_scraping(self) -> int:
        """Build knowledge about web scraping."""
        entries = [
            ("beautifulsoup", "library",
             "HTML/XML parsing library for web scraping. Install: pip install beautifulsoup4",
             ["html", "parse", "scrape", "web", "dom", "xml", "beautifulsoup", "bs4"],
             {"import": "from bs4 import BeautifulSoup",
              "install": "pip install beautifulsoup4"}),
            
            ("BeautifulSoup", "class",
             "Parse HTML/XML document for extraction",
             ["parse", "html", "dom", "soup"],
             {"syntax": "BeautifulSoup(markup, 'html.parser')",
              "example": "soup = BeautifulSoup(html_content, 'html.parser')"}),
            
            ("soup.find", "method",
             "Find first element matching selector",
             ["find", "element", "tag", "first", "select"],
             {"syntax": "soup.find(tag, attrs={})",
              "example": "title = soup.find('h1')"}),
            
            ("soup.find_all", "method",
             "Find all elements matching selector",
             ["find", "all", "elements", "tags", "select", "multiple"],
             {"syntax": "soup.find_all(tag, attrs={})", "returns": "list",
              "example": "links = soup.find_all('a')"}),
            
            ("soup.select", "method",
             "Find elements using CSS selector",
             ["select", "css", "selector", "query"],
             {"syntax": "soup.select(css_selector)", "returns": "list",
              "example": "items = soup.select('div.item')"}),
            
            ("element.text", "attribute",
             "Get text content of HTML element",
             ["text", "content", "inner", "string"],
             {"returns": "str",
              "example": "title_text = title_element.text"}),
            
            ("element.get", "method",
             "Get attribute value from HTML element",
             ["attribute", "get", "href", "src", "class"],
             {"syntax": "element.get(attr_name)", "returns": "str",
              "example": "url = link.get('href')"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_common_patterns(self) -> int:
        """Build knowledge about common programming patterns."""
        entries = [
            ("fetch_url_pattern", "pattern",
             "Fetch webpage content from URL using requests",
             ["fetch", "url", "web", "http", "download", "get", "html"],
             {"code": """import requests

response = requests.get(url)
if response.status_code == 200:
    content = response.text
else:
    print(f'Error: {response.status_code}')"""}),
            
            ("read_json_file_pattern", "pattern",
             "Read and parse JSON file",
             ["read", "json", "file", "load", "parse"],
             {"code": """import json

with open('data.json', 'r') as f:
    data = json.load(f)"""}),
            
            ("write_json_file_pattern", "pattern",
             "Write data to JSON file",
             ["write", "json", "file", "save", "dump"],
             {"code": """import json

with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)"""}),
            
            ("scrape_webpage_pattern", "pattern",
             "Scrape data from webpage using requests and BeautifulSoup",
             ["scrape", "web", "html", "parse", "extract", "beautifulsoup"],
             {"code": """import requests
from bs4 import BeautifulSoup

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
elements = soup.find_all('div', class_='item')
for element in elements:
    print(element.text)"""}),
            
            ("api_request_pattern", "pattern",
             "Make API request and parse JSON response",
             ["api", "request", "json", "fetch", "data"],
             {"code": """import requests

response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()
    # Process data
else:
    print(f'API Error: {response.status_code}')"""}),
            
            ("read_csv_pattern", "pattern",
             "Read CSV file into list of dictionaries",
             ["csv", "read", "file", "data", "table"],
             {"code": """import csv

with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)"""}),
            
            ("write_csv_pattern", "pattern",
             "Write list of dictionaries to CSV file",
             ["csv", "write", "file", "save", "table"],
             {"code": """import csv

with open('data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)"""}),
            
            ("command_line_args_pattern", "pattern",
             "Parse command line arguments",
             ["args", "command", "line", "arguments", "cli"],
             {"code": """import sys

if len(sys.argv) > 1:
    arg1 = sys.argv[1]
else:
    print('Usage: python script.py <arg>')"""}),
            
            ("environment_variable_pattern", "pattern",
             "Read environment variable with default",
             ["environment", "variable", "env", "config", "secret"],
             {"code": """import os

api_key = os.environ.get('API_KEY', 'default_value')
if not api_key:
    raise ValueError('API_KEY not set')"""}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)


def build_python_knowledge(storage_dir: str = None) -> KnowledgeManager:
    """Build and return a knowledge manager with Python knowledge."""
    manager = KnowledgeManager(storage_dir=storage_dir)
    builder = PythonKnowledgeBuilder(manager)
    count = builder.build_all()
    print(f"Built {count} Python knowledge entries")
    return manager


if __name__ == "__main__":
    # Build knowledge in the project's knowledge_store directory
    storage_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_store"
    )
    
    manager = build_python_knowledge(storage_dir)
    
    print(f"\nKnowledge by domain: {manager.count_by_domain()}")
    print(f"\nStored in: {storage_dir}")
    
    # Test some queries
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    test_queries = [
        ["fetch", "url", "web"],
        ["read", "json", "file"],
        ["loop", "iterate", "list"],
        ["scrape", "html", "web"],
    ]
    
    for keywords in test_queries:
        print(f"\nQuery: {keywords}")
        results = manager.query(keywords, domain=KnowledgeDomain.PROGRAMMING, top_k=3)
        for sim, entry in results:
            print(f"  → {entry.name}: {sim:.3f}")
            if entry.metadata.get("code"):
                print(f"      Has code pattern")
            elif entry.metadata.get("example"):
                print(f"      Example: {entry.metadata['example'][:50]}...")
