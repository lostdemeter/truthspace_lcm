# GeometricLCM v1.0 Roadmap

## Vision

**Goal**: A full LLM replacement using geometric concept resolution instead of neural networks.

**Success Criteria**:
- Connect via Open WebUI like any other LLM
- Natural conversational chat
- Code generation and execution
- Matplotlib chart generation
- Self-aware (knows its own capabilities)
- Extensible tool use

## Current State (v0.8)

### What We Have

| Component | Status | Description |
|-----------|--------|-------------|
| Concept Frames | ✓ | Language-agnostic semantic representation |
| 4D φ-Dial | ✓ | Style × Perspective × Depth × Certainty |
| Gradient-Free Learning | ✓ | Error-driven structure building |
| Conversation Memory | ✓ | Multi-turn dialogue, pronoun resolution |
| Multi-Hop Reasoning | ✓ | Graph traversal for WHY/HOW |
| Holographic Generation | ✓ | Interference-based text generation |
| Code Generation | ✓ | Python from natural language |
| Planning & Execution | ✓ | Task decomposition, sandboxed execution |
| Chart Generation | ✓ | Matplotlib visualizations |

### What's Missing for v1.0

1. **OpenAI-Compatible API** - Can't connect from Open WebUI
2. **Natural Conversation** - Responses feel robotic, not conversational
3. **Self-Knowledge** - Doesn't know what it is or what it can do
4. **Modular Architecture** - Components are tightly coupled
5. **Extensible Tools** - Hard to add new capabilities
6. **Knowledge Ingestion** - No easy way to add domain knowledge

---

## Phase 1: OpenAI-Compatible API

### Goal
Expose GeometricLCM as an OpenAI-compatible API server.

### Endpoints Needed

```
POST /v1/chat/completions     # Main chat endpoint
GET  /v1/models               # List available models
POST /v1/completions          # Legacy completions (optional)
```

### Request Format (OpenAI Standard)

```json
{
  "model": "geometric-lcm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who is Sherlock Holmes?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703345678,
  "model": "geometric-lcm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Sherlock Holmes is a brilliant detective..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

### Implementation

```python
# api/server.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GeometricLCM API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "geometric-lcm"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Convert messages to GeometricLCM format
    # Process through our pipeline
    # Return OpenAI-compatible response
    pass
```

### Streaming Support

For Open WebUI compatibility, we need SSE streaming:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream"
        )
    else:
        return generate_response(request)
```

### Files to Create

- `api/__init__.py`
- `api/server.py` - FastAPI server
- `api/models.py` - Pydantic models
- `api/handlers.py` - Request handlers
- `run_api.py` - Entry point

---

## Phase 2: Modular Architecture

### Goal
Decouple components so they can be swapped, extended, and tested independently.

### Current Architecture (Tightly Coupled)

```
chat.py → ConceptQA → HolographicProjector → PatternAnswerGenerator
                    → ReasoningEngine
                    → HolographicGenerator
                    → CodeGenerator
                    → Planner
```

### Proposed Architecture (Modular)

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator (Brain)                      │
│  - Intent classification                                     │
│  - Route to appropriate handler                              │
│  - Manage conversation state                                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Knowledge   │   │     Code      │   │     Tool      │
│    Handler    │   │    Handler    │   │    Handler    │
│               │   │               │   │               │
│ - Q&A         │   │ - Generate    │   │ - Charts      │
│ - Reasoning   │   │ - Execute     │   │ - Web search  │
│ - Generation  │   │ - Plan        │   │ - File ops    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response Composer                         │
│  - Combine outputs from handlers                             │
│  - Apply φ-dial styling                                      │
│  - Format for API response                                   │
└─────────────────────────────────────────────────────────────┘
```

### Handler Interface

```python
class Handler(ABC):
    """Base class for all handlers."""
    
    @abstractmethod
    def can_handle(self, intent: str, context: dict) -> bool:
        """Return True if this handler can process the intent."""
        pass
    
    @abstractmethod
    def handle(self, message: str, context: dict) -> HandlerResult:
        """Process the message and return result."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging/debugging."""
        pass
```

### Intent Classification

```python
class IntentClassifier:
    """Classify user intent to route to appropriate handler."""
    
    INTENTS = {
        'question': ['who', 'what', 'where', 'when', 'why', 'how'],
        'code': ['write', 'create', 'generate', 'code', 'function', 'script'],
        'execute': ['run', 'execute', 'calculate', 'compute'],
        'chart': ['chart', 'graph', 'plot', 'visualize', 'draw'],
        'chat': ['hello', 'hi', 'thanks', 'bye'],
        'meta': ['what can you do', 'help', 'capabilities'],
    }
    
    def classify(self, message: str) -> str:
        """Return the primary intent."""
        pass
```

### Files to Create

- `core/orchestrator.py` - Brain that routes requests
- `core/handlers/base.py` - Handler interface
- `core/handlers/knowledge.py` - Q&A handler
- `core/handlers/code.py` - Code generation handler
- `core/handlers/tools.py` - Tool use handler
- `core/handlers/chat.py` - Conversational handler
- `core/response_composer.py` - Output formatting

---

## Phase 3: Knowledge Ingestion

### Goal
Easy way to add domain knowledge so the system can:
1. Know about itself (meta-knowledge)
2. Learn new domains (user-provided)
3. Sound more natural (conversational patterns)

### Types of Knowledge

| Type | Purpose | Format |
|------|---------|--------|
| Meta | Self-knowledge | Structured facts |
| Domain | Topic expertise | Concept frames |
| Conversational | Natural responses | Pattern templates |
| Procedural | How to do things | Step sequences |

### Meta-Knowledge (Self-Awareness)

```json
{
  "identity": {
    "name": "GeometricLCM",
    "type": "Geometric Language Concept Model",
    "version": "1.0",
    "creator": "Lesley Gushurst",
    "philosophy": "All semantic operations are geometric operations in concept space"
  },
  "capabilities": [
    {
      "name": "answer_questions",
      "description": "Answer questions about knowledge in my corpus",
      "examples": ["Who is Holmes?", "What did Watson do?"]
    },
    {
      "name": "generate_code",
      "description": "Generate Python code from natural language",
      "examples": ["Write a function to add numbers", "Create a factorial function"]
    },
    {
      "name": "create_charts",
      "description": "Create matplotlib visualizations",
      "examples": ["Create a bar chart of...", "Plot the distribution of..."]
    },
    {
      "name": "execute_tasks",
      "description": "Plan and execute computational tasks",
      "examples": ["Calculate the sum of [1,2,3]", "Find even numbers in [1-10]"]
    }
  ],
  "limitations": [
    "I don't have real-time internet access",
    "My knowledge is limited to my corpus",
    "I can't access files on your system"
  ]
}
```

### Conversational Patterns

```json
{
  "greetings": {
    "patterns": ["hello", "hi", "hey", "greetings"],
    "responses": [
      "Hello! I'm GeometricLCM. How can I help you today?",
      "Hi there! What would you like to explore?",
      "Greetings! I'm ready to assist with questions, code, or analysis."
    ]
  },
  "farewells": {
    "patterns": ["bye", "goodbye", "see you", "thanks"],
    "responses": [
      "Goodbye! Feel free to return anytime.",
      "Thanks for chatting! Take care.",
      "See you later! Happy to help again."
    ]
  },
  "clarification": {
    "patterns": ["I don't understand", "what do you mean", "unclear"],
    "responses": [
      "Let me clarify: {previous_response}",
      "I meant that {explanation}. Does that help?",
      "Sorry for the confusion. Here's another way to put it: {alternative}"
    ]
  },
  "uncertainty": {
    "patterns": ["I'm not sure", "maybe", "possibly"],
    "responses": [
      "I'm not entirely certain, but based on my knowledge: {answer}",
      "This is my best understanding: {answer}. Would you like me to elaborate?",
      "I don't have complete information on this, but {partial_answer}"
    ]
  }
}
```

### Ingestion Pipeline

```python
class KnowledgeIngester:
    """Ingest various knowledge formats into GeometricLCM."""
    
    def ingest_text(self, text: str, source: str = None):
        """Ingest plain text, extract concept frames."""
        pass
    
    def ingest_json(self, data: dict, schema: str = 'auto'):
        """Ingest structured JSON data."""
        pass
    
    def ingest_markdown(self, path: str):
        """Ingest markdown documentation."""
        pass
    
    def ingest_code(self, code: str, language: str = 'python'):
        """Ingest code to learn patterns and functions."""
        pass
    
    def ingest_conversation(self, messages: list):
        """Ingest conversation examples for natural responses."""
        pass
```

### Files to Create

- `knowledge/meta.json` - Self-knowledge
- `knowledge/conversational.json` - Chat patterns
- `knowledge/ingester.py` - Ingestion pipeline
- `knowledge/schemas.py` - Knowledge schemas

---

## Phase 4: Extensible Tool Use

### Goal
Make it easy to add new tools/capabilities without modifying core code.

### Tool Interface

```python
class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """What this tool does (for intent matching)."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
```

### Built-in Tools

```python
# tools/builtin/calculator.py
class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform mathematical calculations"
    parameters = {
        "expression": {"type": "string", "description": "Math expression to evaluate"}
    }
    
    def execute(self, expression: str) -> ToolResult:
        result = safe_eval(expression)
        return ToolResult(success=True, output=str(result))

# tools/builtin/chart.py
class ChartTool(Tool):
    name = "chart"
    description = "Create matplotlib charts and visualizations"
    parameters = {
        "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"]},
        "data": {"type": "object"},
        "title": {"type": "string"}
    }
    
    def execute(self, chart_type: str, data: dict, title: str) -> ToolResult:
        # Generate chart code and execute
        pass

# tools/builtin/code.py
class CodeTool(Tool):
    name = "code"
    description = "Generate and execute Python code"
    parameters = {
        "request": {"type": "string", "description": "What code to generate"},
        "execute": {"type": "boolean", "default": False}
    }
```

### Tool Registry

```python
class ToolRegistry:
    """Registry for all available tools."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[dict]:
        """List all tools with their schemas."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self._tools.values()
        ]
    
    def find_tool(self, intent: str) -> Tool:
        """Find the best tool for an intent."""
        # Use semantic matching to find appropriate tool
        pass
```

### Plugin System

```python
# tools/plugins/weather.py (example external plugin)
class WeatherTool(Tool):
    name = "weather"
    description = "Get current weather for a location"
    parameters = {
        "location": {"type": "string"}
    }
    
    def execute(self, location: str) -> ToolResult:
        # Call weather API
        pass

# Load plugins from directory
def load_plugins(plugin_dir: str):
    for file in Path(plugin_dir).glob("*.py"):
        module = importlib.import_module(file.stem)
        for obj in module.__dict__.values():
            if isinstance(obj, type) and issubclass(obj, Tool):
                registry.register(obj())
```

### Files to Create

- `tools/__init__.py`
- `tools/base.py` - Tool interface
- `tools/registry.py` - Tool registry
- `tools/builtin/calculator.py`
- `tools/builtin/chart.py`
- `tools/builtin/code.py`
- `tools/plugins/` - Plugin directory

---

## Phase 5: Natural Conversation

### Goal
Make responses feel natural, not robotic.

### Current Problem

```
User: Hi, how are you?
Bot: I don't have information about that in my knowledge base.

User: What can you do?
Bot: [No response or generic error]
```

### Desired Behavior

```
User: Hi, how are you?
Bot: Hello! I'm doing well, thank you for asking. I'm GeometricLCM, 
     a geometric language model. How can I help you today?

User: What can you do?
Bot: I can help you with several things:
     • Answer questions about topics in my knowledge base
     • Generate Python code from natural language
     • Create charts and visualizations
     • Execute computational tasks
     
     What would you like to explore?
```

### Response Templates

```python
class ResponseTemplates:
    """Templates for natural responses."""
    
    GREETING = [
        "Hello! I'm {name}. {tagline}",
        "Hi there! I'm ready to help. {capabilities_summary}",
        "Greetings! What would you like to explore today?",
    ]
    
    CAPABILITY_INTRO = """
I can help you with:
• **Knowledge Q&A** - Ask about topics in my corpus
• **Code Generation** - Write Python functions and scripts
• **Visualizations** - Create charts with matplotlib
• **Task Execution** - Calculate, filter, transform data

What interests you?
"""
    
    UNCERTAINTY = [
        "I'm not entirely sure about that, but here's what I know: {answer}",
        "Based on my knowledge: {answer}. Would you like more details?",
        "This is my understanding: {answer}",
    ]
    
    NO_ANSWER = [
        "I don't have specific information about that in my knowledge base.",
        "That's outside my current knowledge. Could you rephrase or ask about something else?",
        "I'm not sure about that. Is there something related I could help with?",
    ]
```

### Conversation Flow

```python
class ConversationManager:
    """Manage conversation flow for natural dialogue."""
    
    def __init__(self):
        self.state = ConversationState()
        self.templates = ResponseTemplates()
    
    def process(self, message: str) -> str:
        # 1. Classify intent
        intent = self.classify_intent(message)
        
        # 2. Handle meta-conversation (greetings, capabilities, etc.)
        if intent in ['greeting', 'farewell', 'meta']:
            return self.handle_meta(intent, message)
        
        # 3. Route to appropriate handler
        handler = self.get_handler(intent)
        result = handler.handle(message, self.state.context)
        
        # 4. Compose natural response
        response = self.compose_response(result, intent)
        
        # 5. Update state
        self.state.update(message, response)
        
        return response
```

---

## Implementation Order

### Sprint 1: API Layer (Week 1)
1. Create FastAPI server with OpenAI-compatible endpoints
2. Implement basic chat completions
3. Add streaming support
4. Test with Open WebUI

### Sprint 2: Modular Architecture (Week 2)
1. Create handler interface
2. Refactor existing code into handlers
3. Create orchestrator
4. Create response composer

### Sprint 3: Self-Knowledge (Week 3)
1. Create meta-knowledge JSON
2. Create conversational patterns
3. Implement meta-conversation handler
4. Test natural greetings and capability queries

### Sprint 4: Tool System (Week 4)
1. Create tool interface
2. Implement tool registry
3. Create built-in tools (calculator, chart, code)
4. Add plugin loading

### Sprint 5: Polish & Integration (Week 5)
1. Knowledge ingestion pipeline
2. Response templates and natural language
3. End-to-end testing
4. Documentation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Open WebUI connection | Works |
| Response latency | < 500ms |
| Greeting handling | Natural |
| Capability queries | Accurate |
| Code generation | Functional |
| Chart generation | Works |
| Tool extensibility | Plugin-based |

---

## Files to Create

### API Layer
```
api/
├── __init__.py
├── server.py          # FastAPI app
├── models.py          # Pydantic models
├── handlers.py        # Request handlers
└── streaming.py       # SSE streaming
```

### Modular Core
```
core/
├── orchestrator.py    # Request routing
├── handlers/
│   ├── base.py        # Handler interface
│   ├── knowledge.py   # Q&A handler
│   ├── code.py        # Code handler
│   ├── tools.py       # Tool handler
│   └── chat.py        # Conversation handler
└── response_composer.py
```

### Knowledge
```
knowledge/
├── meta.json          # Self-knowledge
├── conversational.json # Chat patterns
├── ingester.py        # Ingestion pipeline
└── schemas.py         # Knowledge schemas
```

### Tools
```
tools/
├── __init__.py
├── base.py            # Tool interface
├── registry.py        # Tool registry
├── builtin/
│   ├── calculator.py
│   ├── chart.py
│   └── code.py
└── plugins/           # External plugins
```

---

## Summary

**v1.0 = API + Modular + Self-Aware + Tools + Natural**

The key insight: We're not trying to replicate neural network behavior. We're building a **geometric reasoning system** that happens to speak the same API language as LLMs. Our advantage is:

1. **Interpretable** - Every answer has a traceable path
2. **Efficient** - No GPU needed, instant responses
3. **Extensible** - Add knowledge without retraining
4. **Transparent** - No black box, just geometry

Let's build it.
