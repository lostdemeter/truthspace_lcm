# Phase 6: Data Ingestion and Capability Refinement

## Goal

Evolve GeometricLCM to compete with real LLMs by improving:
1. **Natural conversation flow** - More human-like, contextual responses
2. **Code generation completeness** - Full, runnable code with proper structure
3. **Response length control** - Configurable verbosity for chat and code
4. **Data ingestion** - Expand knowledge base beyond literary works

## Current State Analysis

### Chat/Conversation
- **Strengths**: 4D φ-dial for style control, pronoun resolution, conversation memory
- **Weaknesses**: 
  - Responses are template-driven, feel mechanical
  - Limited to literary knowledge domain
  - No context-aware follow-up questions
  - No handling of ambiguity or clarification

### Code Generation
- **Strengths**: 30+ built-in operations, synonym mapping, learnable functions
- **Weaknesses**:
  - Only generates single functions, not complete programs
  - No imports, no error handling, no type hints
  - Limited to predefined patterns
  - No multi-file or class generation

### Response Length
- **Current**: Depth dial (z-axis) controls elaboration level
- **Missing**: 
  - Token/word count limits
  - Streaming with early termination
  - Code completeness levels (stub, skeleton, full)

---

## Proposed Improvements

### 1. Natural Conversation Flow

#### 1.1 Response Naturalness
```python
class ConversationStyle:
    """Control how natural/human responses feel."""
    
    # Vary sentence structure
    sentence_patterns = [
        "statement",      # "Holmes is a detective."
        "elaboration",    # "Holmes is a detective, known for..."
        "contrast",       # "While Holmes is brilliant, he..."
        "question_back",  # "Holmes is a detective. What aspect interests you?"
    ]
    
    # Add conversational fillers (optional, dial-controlled)
    fillers = {
        'thinking': ["Well,", "Let me think...", "Hmm,"],
        'transition': ["So,", "Now,", "Also,"],
        'emphasis': ["Actually,", "In fact,", "Interestingly,"],
    }
    
    # Vary response openings (avoid repetitive "X is a...")
    openings = {
        'direct': "{entity} is...",
        'contextual': "In {source}, {entity}...",
        'descriptive': "Known for {quality}, {entity}...",
        'narrative': "The story introduces {entity} as...",
    }
```

#### 1.2 Context-Aware Responses
```python
class ContextAwareResponder:
    """Generate responses that acknowledge conversation context."""
    
    def generate(self, query, context):
        # Reference previous topics
        if context.continues_topic():
            return f"Continuing about {context.topic}..."
        
        # Acknowledge topic shifts
        if context.topic_changed():
            return f"Moving to {new_topic}..."
        
        # Handle follow-up questions
        if context.is_followup():
            return f"Regarding {context.last_entity}..."
```

#### 1.3 Clarification and Ambiguity Handling
```python
class AmbiguityHandler:
    """Handle ambiguous or unclear queries."""
    
    def check_ambiguity(self, query):
        # Multiple possible entities
        if len(matching_entities) > 1:
            return f"Do you mean {entity1} or {entity2}?"
        
        # Unclear intent
        if confidence < 0.5:
            return f"I'm not sure I understand. Are you asking about...?"
        
        # Missing context
        if needs_context:
            return f"Could you tell me more about what you're looking for?"
```

### 2. Code Generation Completeness

#### 2.1 Code Completeness Levels
```python
class CodeCompleteness(Enum):
    STUB = "stub"           # Just signature and docstring
    SKELETON = "skeleton"   # Structure with TODO comments
    BASIC = "basic"         # Working implementation, minimal
    FULL = "full"           # Complete with error handling, types, tests
    PRODUCTION = "production"  # Full + logging, validation, edge cases
```

#### 2.2 Enhanced Code Generator
```python
class EnhancedCodeGenerator:
    """Generate complete, runnable code."""
    
    def generate(self, request, completeness=CodeCompleteness.FULL):
        code = CodeBlock()
        
        # Add imports
        code.add_imports(self._infer_imports(request))
        
        # Add type hints
        if completeness >= CodeCompleteness.BASIC:
            code.add_type_hints()
        
        # Add error handling
        if completeness >= CodeCompleteness.FULL:
            code.add_error_handling()
        
        # Add docstring
        code.add_docstring(self._generate_docstring(request))
        
        # Add implementation
        code.add_body(self._generate_body(request))
        
        # Add example usage
        if completeness >= CodeCompleteness.FULL:
            code.add_example()
        
        return code.render()
```

#### 2.3 Code Templates for Common Patterns
```python
CODE_TEMPLATES = {
    'api_endpoint': '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class {RequestModel}(BaseModel):
    {fields}

@app.{method}("/{path}")
async def {function_name}(request: {RequestModel}):
    """
    {docstring}
    """
    try:
        {implementation}
        return {{"status": "success", "data": result}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
''',
    
    'data_class': '''
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class {ClassName}:
    """
    {docstring}
    """
    {fields}
    
    def __post_init__(self):
        {validation}
''',
    
    'cli_tool': '''
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="{description}")
    {arguments}
    args = parser.parse_args()
    
    try:
        {implementation}
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
''',
}
```

### 3. Response Length Control

#### 3.1 Length Parameters
```python
class ResponseLength:
    """Control response length."""
    
    # For chat
    BRIEF = 1       # 1-2 sentences, ~20 words
    STANDARD = 2    # 2-4 sentences, ~50 words
    DETAILED = 3    # 4-8 sentences, ~100 words
    COMPREHENSIVE = 4  # Full explanation, ~200+ words
    
    # For code
    MINIMAL = 1     # Just the function
    STANDARD = 2    # Function + docstring
    COMPLETE = 3    # Function + docstring + example
    FULL = 4        # Complete module with tests
```

#### 3.2 Token Budget System
```python
class TokenBudget:
    """Manage response length via token budgets."""
    
    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens
        self.used_tokens = 0
    
    def can_add(self, text: str) -> bool:
        tokens = self.estimate_tokens(text)
        return self.used_tokens + tokens <= self.max_tokens
    
    def add(self, text: str) -> str:
        """Add text, truncating if necessary."""
        tokens = self.estimate_tokens(text)
        if self.used_tokens + tokens > self.max_tokens:
            # Truncate at sentence boundary
            return self.truncate_to_fit(text)
        self.used_tokens += tokens
        return text
```

### 4. Data Ingestion Pipeline

#### 4.1 Ingestion Sources
```python
class DataSource(Enum):
    TEXT_FILE = "text"          # Plain text files
    MARKDOWN = "markdown"       # Markdown documents
    CODE = "code"               # Source code files
    JSON = "json"               # Structured JSON
    WEB = "web"                 # Web pages (with extraction)
    PDF = "pdf"                 # PDF documents
```

#### 4.2 Knowledge Domains
```python
KNOWLEDGE_DOMAINS = {
    'programming': {
        'sources': ['python_docs', 'tutorials', 'stackoverflow'],
        'extractors': ['code_extractor', 'api_extractor'],
    },
    'general': {
        'sources': ['wikipedia', 'encyclopedias'],
        'extractors': ['fact_extractor', 'definition_extractor'],
    },
    'technical': {
        'sources': ['manuals', 'specifications'],
        'extractors': ['procedure_extractor', 'spec_extractor'],
    },
}
```

#### 4.3 Ingestion Pipeline
```python
class IngestionPipeline:
    """Pipeline for ingesting new knowledge."""
    
    def ingest(self, source: str, domain: str):
        # 1. Load and parse
        content = self.load(source)
        
        # 2. Extract concept frames
        frames = self.extract_frames(content, domain)
        
        # 3. Deduplicate
        frames = self.deduplicate(frames)
        
        # 4. Validate
        frames = self.validate(frames)
        
        # 5. Add to knowledge base
        self.knowledge.add_frames(frames)
        
        # 6. Update indices
        self.knowledge.rebuild_indices()
```

---

## Implementation Plan

### Phase 6.1: Response Length Control (Priority: High)
1. Add `max_tokens` parameter to orchestrator
2. Implement token budget system
3. Add length presets (brief, standard, detailed)
4. Update API to accept length parameter

### Phase 6.2: Conversation Naturalness (Priority: High)
1. Vary response openings
2. Add context acknowledgment
3. Implement follow-up detection
4. Add clarification requests for ambiguous queries

### Phase 6.3: Code Generation Enhancement (Priority: Medium)
1. Add completeness levels
2. Implement import inference
3. Add type hints generation
4. Create code templates for common patterns
5. Add error handling generation

### Phase 6.4: Data Ingestion (Priority: Medium)
1. Create ingestion pipeline
2. Add markdown/text extractors
3. Add code documentation extractor
4. Implement deduplication
5. Add validation

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Response variety (unique openings) | ~3 | 10+ |
| Code completeness (runnable %) | ~40% | 90%+ |
| Length control accuracy | N/A | ±10% of target |
| Knowledge domains | 1 (literary) | 3+ |
| Average response naturalness | 3/5 | 4/5 |

---

## API Changes

### Chat Completions
```python
# New parameters
{
    "max_tokens": 500,           # Token limit
    "response_length": "standard",  # brief/standard/detailed
    "style": {
        "naturalness": 0.8,      # 0-1, how conversational
        "formality": 0.5,        # 0-1, formal to casual
    }
}
```

### Code Generation
```python
# New parameters
{
    "completeness": "full",      # stub/skeleton/basic/full
    "include_imports": true,
    "include_types": true,
    "include_tests": false,
    "language": "python",
}
```

---

## Next Steps

1. Start with response length control (most impactful)
2. Then improve conversation naturalness
3. Enhance code generation
4. Finally, expand data ingestion

This phased approach allows incremental improvement while maintaining stability.
