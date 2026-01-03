# Roadmap: Goose Tool Calling Integration

**Date:** January 1, 2026
**Status:** Planning
**Target:** Goose AI Agent (Block)

## Overview

This roadmap outlines the path from our current "chat-only" mode with Goose to full tool calling support. The goal is to let Goose use our geometric LCM as its reasoning engine while leveraging Goose's built-in tool execution capabilities.

## Current State

### What Works
- OpenAI-compatible `/v1/chat/completions` endpoint
- Holographic pattern matching for code generation
- Template composition for natural language modifications
- Auto-execution of generated plot code
- Vision API format handling (content as list)

### What's Missing
- Tool calling support (`tools` parameter, `tool_calls` response)
- Intent routing (code vs knowledge vs tool)
- Tool schema parsing and matching

## Phases

---

## Phase 1: Solidify the Core (Current)

**Goal:** Reliable pattern matching and generation before adding complexity.

### 1.1 Expand Template Library
- [ ] Add more plot types (bar, scatter, histogram, pie)
- [ ] Add data manipulation patterns (read CSV, transform data)
- [ ] Add file operation patterns (read, write, list)

### 1.2 Improve Template Composition
- [ ] Handle compound modifications ("red, dashed, with title 'My Plot'")
- [ ] Support more modification types (line style, markers, legends)
- [ ] Better error messages for unsupported modifications

### 1.3 Build Test Suite
```python
# Target: 80%+ accuracy on routing decisions
TEST_QUERIES = [
    # Code generation
    ("create a sine wave plot", "code"),
    ("make a bar chart of [1,2,3]", "code"),
    ("plot x^2 from -10 to 10", "code"),
    
    # Code with modifications
    ("create a sine wave in red with amplitude 2", "code_with_mods"),
    ("make a scatter plot with blue dots", "code_with_mods"),
    
    # Should reject gracefully
    ("write me a web server", "reject_or_fallback"),
    ("explain quantum physics", "knowledge_or_reject"),
    
    # Future tool calls
    ("list files in current directory", "tool_call"),
    ("read the contents of README.md", "tool_call"),
    ("run pytest", "tool_call"),
]
```

### 1.4 Metrics to Track
- **Pattern match rate:** % of queries that find a valid template
- **Execution success rate:** % of generated code that runs without error
- **Modification accuracy:** % of modifications correctly applied
- **Rejection accuracy:** % of out-of-scope queries correctly rejected

### Exit Criteria
- [ ] 80%+ pattern match rate on test suite
- [ ] 90%+ execution success rate
- [ ] 80%+ modification accuracy
- [ ] 90%+ rejection accuracy (no garbage output)

---

## Phase 2: Intent Classification

**Goal:** Accurately route queries to the right handler.

### 2.1 Define Intent Categories
```python
class Intent(Enum):
    CODE_GENERATION = "code"      # Generate and execute code
    TOOL_CALL = "tool"            # Call a Goose tool (file ops, bash, etc.)
    KNOWLEDGE = "knowledge"       # Answer a question
    CLARIFICATION = "clarify"     # Need more info from user
    UNSUPPORTED = "unsupported"   # Can't handle this
```

### 2.2 Build Intent Classifier
- Use holographic space to classify intent
- Seed with examples for each category
- Learn from usage patterns

### 2.3 Map Goose Tools to Intents
```python
GOOSE_TOOL_PATTERNS = {
    "Read": ["read file", "show contents", "cat", "view"],
    "Write": ["write file", "save to", "create file"],
    "Edit": ["edit file", "modify", "change line", "replace"],
    "Bash": ["run command", "execute", "shell", "terminal"],
    "Glob": ["find files", "list files", "search for"],
    "Grep": ["search in files", "find text", "grep"],
}
```

### Exit Criteria
- [ ] 85%+ intent classification accuracy
- [ ] Clear separation between code generation and tool calls
- [ ] Graceful handling of ambiguous queries

---

## Phase 3: Tool Calling Implementation

**Goal:** Implement OpenAI-compatible tool calling format.

### 3.1 Parse Tool Schemas
Goose sends tools in the request:
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Read a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string"}
          },
          "required": ["file_path"]
        }
      }
    }
  ]
}
```

### 3.2 Generate Tool Calls
When intent is TOOL_CALL, return:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "Read",
          "arguments": "{\"file_path\": \"/path/to/file\"}"
        }
      }]
    }
  }]
}
```

### 3.3 Handle Tool Results
Goose sends back:
```json
{
  "messages": [
    {"role": "user", "content": "read README.md"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "file contents..."}
  ]
}
```

Our LCM needs to process the tool result and continue.

### 3.4 Argument Extraction
Use template composition to extract arguments:
- "read README.md" → `{"file_path": "README.md"}`
- "list files in /tmp" → `{"pattern": "/tmp/*"}`
- "run pytest tests/" → `{"command": "pytest tests/"}`

### Exit Criteria
- [ ] Correctly parse tool schemas from Goose
- [ ] Generate valid tool_calls format
- [ ] Extract arguments with 80%+ accuracy
- [ ] Handle multi-turn tool conversations

---

## Phase 4: Integration Testing

**Goal:** End-to-end testing with Goose.

### 4.1 Test Scenarios
```bash
# File operations
goose> read the README
goose> list python files in src/
goose> create a new file called test.py

# Code execution
goose> run the tests
goose> execute python script.py

# Combined workflows
goose> read data.csv and create a bar chart
goose> find all TODO comments in the codebase
```

### 4.2 Error Handling
- Tool execution failures
- Invalid arguments
- Permission denied
- Timeout handling

### 4.3 Performance
- Response latency < 2s for simple queries
- Streaming support for long responses

### Exit Criteria
- [ ] All test scenarios pass
- [ ] Error handling is graceful
- [ ] Performance meets targets
- [ ] Goose extensions work correctly

---

## Phase 5: Advanced Features (Future)

### 5.1 Multi-Tool Calls
- Call multiple tools in sequence
- Parallel tool execution
- Tool result aggregation

### 5.2 Learning from Tool Usage
- Track which tools are used for which queries
- Improve argument extraction over time
- Learn new tool patterns from usage

### 5.3 Custom Tools
- Allow users to define custom tools
- MCP server integration
- Tool composition

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Goose                                │
│  (Agent orchestration, tool execution, conversation mgmt)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ OpenAI-compatible API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    api_server.py                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Intent    │  │    Tool     │  │      Code           │  │
│  │ Classifier  │  │   Router    │  │   Orchestrator      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         ▼                ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Holographic Pattern Space                  ││
│  │  (Pattern matching, template composition, learning)     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Solidify Core | 2-3 weeks | None |
| Phase 2: Intent Classification | 1-2 weeks | Phase 1 |
| Phase 3: Tool Calling | 2-3 weeks | Phase 2 |
| Phase 4: Integration Testing | 1-2 weeks | Phase 3 |
| Phase 5: Advanced Features | Ongoing | Phase 4 |

**Total to basic tool calling:** ~6-10 weeks

---

## Success Metrics

### Phase 1 Complete
- Can reliably generate and execute code for known patterns
- Users don't see garbage output

### Phase 2 Complete
- Queries are routed to the correct handler
- Clear distinction between "generate code" and "call tool"

### Phase 3 Complete
- Goose can use our LCM with extensions enabled
- File operations, bash commands work through tool calls

### Phase 4 Complete
- Production-ready integration
- Comparable to using Goose with a cloud LLM

---

## Notes

### Why Goose?
- Open source, actively maintained
- Already supports custom OpenAI-compatible endpoints
- Built-in tool execution (we don't have to implement it)
- MCP support for extensibility
- Desktop app and CLI options

### What We're NOT Building
- Tool execution (Goose handles this)
- Conversation management (Goose handles this)
- File watching, git integration (Goose extensions)

### What We ARE Building
- Intent classification (code vs tool vs knowledge)
- Tool schema → argument extraction
- Pattern matching for tool selection
- Learning from tool usage patterns

---

## References

- Goose docs: https://block.github.io/goose/docs/
- Goose providers: https://block.github.io/goose/docs/getting-started/providers/
- OpenAI tool calling: https://platform.openai.com/docs/guides/function-calling
- Our holographic space: `084_holographic_pattern_projection.md`
- Template composer: `086_emergent_gear_pattern.md`
