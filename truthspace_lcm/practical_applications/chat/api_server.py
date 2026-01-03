"""
OpenAI-Compatible API Server for Emergent Conversational Chain

Provides a REST API compatible with OpenAI's chat completions endpoint,
allowing the emergent chat to be used with tools like Open WebUI or Goose.

Key feature: All responses are EMERGENT - no LLM during conversation.
The LLM is only used as a knowledge resource during corpus building.

Run with:
    cd /home/thorin/truthspace-lcm
    python -m truthspace_lcm.practical_applications.chat.api_server --port 8001
    
    # Or with uvicorn for auto-reload during development:
    uvicorn truthspace_lcm.practical_applications.chat.api_server:app --reload --port 8001

API Endpoints:
    GET  /health          - Health check
    GET  /stats           - Engine statistics
    GET  /topics          - List known topics
    GET  /corpus          - Corpus information
    GET  /books           - List available books
    POST /learn           - Learn a new topic: {"topic": "George Washington"}
    POST /save            - Save corpus: {"path": "corpus/chat_corpus.json"}
    POST /reload          - Reload corpus: {"path": "corpus/chat_corpus.json"}
    POST /build           - Run one corpus build iteration
    POST /load_book       - Load a book: {"book_name": "moby_dick"}
    POST /v1/chat/completions - OpenAI-compatible chat endpoint

Chat Commands (via chat interface):
    /learn <topic>  - Learn about a topic (e.g., "/learn George Washington")
    /save [path]    - Save corpus to file
    /reload [path]  - Reload corpus from file
    /topics         - List known topics
    /stats          - Show statistics
    /help           - Show help

Example workflow with curl:
    # Learn a topic
    curl -X POST http://localhost:8001/learn -H "Content-Type: application/json" -d '{"topic": "George Washington"}'
    
    # Save the corpus
    curl -X POST http://localhost:8001/save -H "Content-Type: application/json" -d '{"path": "corpus/chat_corpus.json"}'
    
    # Ask about the topic
    curl -X POST http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "emergent-chat",
        "messages": [{"role": "user", "content": "Who was George Washington?"}]
    }'
    
    # Reload after restart
    curl -X POST http://localhost:8001/reload -H "Content-Type: application/json" -d '{"path": "corpus/chat_corpus.json"}'

Author: Lesley Gushurst
License: GPLv3
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.core import ConversationalChain
from truthspace_lcm.core.intent_classifier import IntentClassifier, Intent, IntentMatch
from truthspace_lcm.core.gear_orchestrator import GearOrchestrator


# Default configuration
DEFAULT_LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2:latest"

DEFAULT_SEED_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "programming",
    "python",
    "science",
    "philosophy",
    "mathematics",
]


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    model_config = {"extra": "ignore"}  # Ignore extra fields like 'name', 'tool_calls', etc.
    
    role: str
    content: Optional[Any] = ""  # Content can be str, None, or list (for vision API)
    tool_call_id: Optional[str] = None  # For tool role messages
    
    def get_text_content(self) -> str:
        """Extract text content, handling string, None, or list formats."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            # Vision API format: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
            texts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        return str(self.content)


class ToolFunction(BaseModel):
    """Function definition for a tool."""
    model_config = {"extra": "ignore"}
    name: str
    description: Optional[str] = ""
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition from Goose."""
    model_config = {"extra": "ignore"}
    type: str = "function"
    function: ToolFunction


class FunctionCall(BaseModel):
    """A function call in a tool_calls response."""
    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    """A tool call in the response."""
    id: str
    type: str = "function"
    function: FunctionCall


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}  # Ignore extra fields from clients like Goose
    
    model: str = "emergent-chat"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None  # Tools available for calling
    tool_choice: Optional[Any] = None  # Tool choice preference


class ResponseMessage(BaseModel):
    """Message in a response, can include tool_calls."""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "truthspace"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class EmergentChatEngine:
    """
    The emergent chat engine that powers the API.
    
    All responses are generated using emergent patterns only.
    No LLM calls during conversation.
    
    Now with gear-based routing for tool calls.
    """
    
    def __init__(self, 
                 llm_url: str = DEFAULT_LLM_URL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 seed_topics: List[str] = None,
                 corpus_path: str = None,
                 lazy_init: bool = False,
                 enable_tools: bool = True):
        
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.seed_topics = seed_topics
        self.corpus_path = corpus_path
        self.enable_tools = enable_tools
        
        # Create conversational chain
        self.chain = ConversationalChain()
        self.chain.configure_llm(llm_url, llm_model)
        
        # Emergent intent classifier (fail-fast: no legacy fallback)
        self.intent_classifier = IntentClassifier()
        
        # Gear orchestrator for tool calls
        self.orchestrator: Optional[GearOrchestrator] = None
        if enable_tools:
            self.orchestrator = GearOrchestrator()
            self.orchestrator.configure_llm(llm_url, llm_model)
        
        # Python code gear for simple code generation
        self.python_gear = None
        try:
            from truthspace_lcm.core.python_code_gear import PythonCodeGear
            self.python_gear = PythonCodeGear()
            self.python_gear.configure_llm(llm_url, llm_model)
        except ImportError:
            pass
        
        # Code orchestrator for complex multi-step code generation
        self.code_orchestrator = None
        try:
            from truthspace_lcm.core.code_orchestrator import CodeOrchestrator
            self.code_orchestrator = CodeOrchestrator()
            self.code_orchestrator.configure_llm(llm_url, llm_model)
        except ImportError:
            pass
        
        # Pending commands awaiting confirmation
        self.pending_commands: List[str] = []
        
        # Build or load corpus (unless lazy)
        if not lazy_init:
            self._init_corpus()
    
    def _init_corpus(self):
        """Initialize corpus from path or topics."""
        if self.corpus_path and Path(self.corpus_path).exists():
            logger.info(f"Loading corpus from {self.corpus_path}")
            self.chain.load_corpus(self.corpus_path)
        elif self.seed_topics:
            logger.info(f"Building corpus from topics: {self.seed_topics}")
            self.chain.build_corpus(self.seed_topics, expand=True)
        
        stats = self.chain.get_stats()
        logger.info(f"Engine ready: {stats['topics']} topics, {stats['corpus_items']} items")
    
    def generate(self, messages: List[Message]) -> str:
        """
        Generate a response using smart routing.
        - Knowledge queries -> emergent chain
        - Tool calls -> gear orchestrator
        """
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.get_text_content()
                break
        
        if not user_message:
            return "I need a question to answer."
        
        # Filter out Goose system prompt if present
        goose_prefix = "You are a general-purpose AI agent called goose"
        if user_message.startswith(goose_prefix):
            # Extract the actual user message after the system prompt
            # Goose typically sends: system prompt + "\n\n" + actual message
            parts = user_message.split("\n\n", 1)
            if len(parts) > 1:
                user_message = parts[-1].strip()
            else:
                # Try to find the actual request after common delimiters
                for delimiter in ["\nUser:", "\nHuman:", "\n---\n"]:
                    if delimiter in user_message:
                        user_message = user_message.split(delimiter)[-1].strip()
                        break
        
        # Handle confirmation for pending commands
        if self.pending_commands and user_message.lower() in ('yes', 'y'):
            return self._execute_pending()
        elif self.pending_commands:
            self.pending_commands = []
            return "Cancelled."
        
        # Handle special commands
        if user_message.lower().startswith("/learn "):
            topic = user_message[7:].strip()
            if self.chain.learn_topic(topic):
                stats = self.chain.get_stats()
                return f"I've learned about {topic}. I now know {stats['topics']} topics with {stats['corpus_items']} facts. You can ask me questions about {topic}."
            return f"I couldn't learn about {topic}. Make sure the LLM is running."
        
        if user_message.lower().startswith("learn about "):
            topic = user_message[12:].strip()
            if self.chain.learn_topic(topic):
                stats = self.chain.get_stats()
                return f"I've learned about {topic}. I now know {stats['topics']} topics with {stats['corpus_items']} facts. You can ask me questions about {topic}."
            return f"I couldn't learn about {topic}. Make sure the LLM is running."
        
        if user_message.lower() == "/save":
            from truthspace_lcm.corpus import CORPUS_DIR
            full_path = str(CORPUS_DIR / "chat_corpus.json")
            self.chain.save_corpus(full_path)
            stats = self.chain.get_stats()
            return f"Saved corpus. {stats['topics']} topics, {stats['corpus_items']} items."
        
        if user_message.lower().startswith("/save "):
            from truthspace_lcm.corpus import CORPUS_DIR
            path = user_message[6:].strip()
            if not Path(path).is_absolute():
                path = str(CORPUS_DIR / path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.chain.save_corpus(path)
            stats = self.chain.get_stats()
            return f"Saved corpus. {stats['topics']} topics, {stats['corpus_items']} items."
        
        if user_message.lower() == "/reload":
            from truthspace_lcm.corpus import CORPUS_DIR
            full_path = str(CORPUS_DIR / "chat_corpus.json")
            if Path(full_path).exists():
                self.chain.load_corpus(full_path)
                stats = self.chain.get_stats()
                return f"Reloaded corpus. {stats['topics']} topics, {stats['corpus_items']} items."
            return f"No corpus file found. Use /save first."
        
        if user_message.lower().startswith("/reload "):
            from truthspace_lcm.corpus import CORPUS_DIR
            path = user_message[8:].strip()
            if not Path(path).is_absolute():
                path = str(CORPUS_DIR / path)
            if Path(path).exists():
                self.chain.load_corpus(path)
                stats = self.chain.get_stats()
                return f"Reloaded corpus. {stats['topics']} topics, {stats['corpus_items']} items."
            return f"No corpus file found at {path}."
        
        if user_message.lower() == "/topics":
            topics = self.chain.list_topics()[:30]
            if topics:
                return f"I know about ({len(self.chain.topics)} total): {', '.join(topics)}"
            return "I don't know any topics yet. Use /learn <topic> to teach me."
        
        if user_message.lower() == "/stats":
            stats = self.chain.get_stats()
            return f"Topics: {stats['topics']}, Facts: {stats['corpus_items']}, Definitions: {stats['definitions']}, LLM calls during chat: {stats['conversation_calls']} (should be 0)"
        
        if user_message.lower() == "/help":
            return """Available commands:
/learn <topic> - Learn about a new topic (uses LLM)
/save [path]   - Save corpus (default: corpus/chat_corpus.json)
/reload [path] - Reload corpus from file
/topics        - List known topics
/stats         - Show statistics
/help          - Show this help"""
        
        if user_message.lower() == "what topics do you know?":
            topics = self.chain.list_topics()[:20]
            return f"I can discuss: {', '.join(topics)}"
        
        # Detect intent using emergent classifier (fail-fast: no legacy fallback)
        intent_result = self.intent_classifier.classify(user_message)
        
        if intent_result.intent == Intent.KNOWLEDGE:
            # Knowledge query - use emergent chain
            return self.chain.chat(user_message)
        
        elif intent_result.intent == Intent.CODE_GENERATION:
            # Code generation - route to code orchestrator
            if self.code_orchestrator:
                plan = self.code_orchestrator.generate(user_message)
                
                if plan.complete_code:
                    response = f"```python\n{plan.complete_code}\n```"
                    response += f"\n\n*Generated via CodeOrchestrator ({len(plan.functions)} functions)*"
                    if plan.verified:
                        response += f"\n✓ Code verified"
                    if plan.output:
                        response += f"\nOutput: {plan.output[:200]}"
                    if plan.error:
                        response += f"\n⚠ {plan.error}"
                    
                    # Auto-execute plot code and save to run_graph.py
                    if 'plt.' in plan.complete_code or 'matplotlib' in plan.complete_code:
                        exec_result = self._execute_plot_code(plan.complete_code)
                        response += f"\n\n{exec_result}"
                    
                    return response
                else:
                    return f"Failed to generate code: {plan.error}"
            
            # Simple code generation - use Python code gear
            if not self.python_gear:
                raise RuntimeError("CODE_GENERATION intent detected but no code generator available")
            
            result = self.python_gear.generate_from_text(user_message)
            
            if result.success:
                response = f"```python\n{result.code}\n```"
                if result.pattern_used:
                    response += f"\n\n*Pattern: {result.pattern_used}*"
                if result.verified:
                    response += f"\n✓ Code verified - runs successfully"
                    if result.output:
                        response += f"\nOutput: {result.output.strip()[:200]}"
                return response
            else:
                return f"Failed to generate code: {result.error}"
        
        elif intent_result.intent == Intent.TOOL_CALL:
            # Tool call - use orchestrator
            if not self.orchestrator:
                raise RuntimeError("TOOL_CALL intent detected but orchestrator not enabled")
            
            result = self.orchestrator.execute(user_message, dry_run=True)
            
            if result['commands']:
                self.pending_commands = result['commands']
                cmd_list = '\n'.join([f"  $ {cmd}" for cmd in result['commands']])
                return (
                    f"I'll need to run these commands:\n{cmd_list}\n\n"
                    f"Reply 'yes' to execute, or anything else to cancel."
                )
            else:
                raise RuntimeError(f"TOOL_CALL intent detected but no commands generated for: {user_message}")
        
        elif intent_result.intent == Intent.UNSUPPORTED:
            # Fail-fast: don't silently fall back to chat
            raise RuntimeError(f"UNSUPPORTED intent - emergent classifier could not route: {user_message}")
        
        # CLARIFICATION intent - ask for more info
        return f"I need more information to help you. Could you clarify: {user_message}"
    
    def _execute_plot_code(self, code: str) -> str:
        """Execute plot code and save to output/ directory for re-running."""
        import subprocess
        from pathlib import Path
        
        # Get project root and output directory
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save script to output directory
        script_path = output_dir / "generated_plot.py"
        script_path.write_text(code)
        
        # Execute the code
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                return f"\n\n**Executed!** {output}\nCode saved to `output/generated_plot.py`"
            else:
                error = result.stderr.strip()[:200]
                return f"\n\nExecution failed: {error}\nCode saved to `output/generated_plot.py`"
        except subprocess.TimeoutExpired:
            return "\n\nExecution timed out (30s limit)\nCode saved to `output/generated_plot.py`"
        except Exception as e:
            return f"\n\nExecution error: {str(e)}\nCode saved to `output/generated_plot.py`"
    
    def _execute_pending(self) -> str:
        """Execute pending commands."""
        import subprocess
        results = []
        
        for cmd in self.pending_commands:
            try:
                output = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                if output.returncode == 0:
                    results.append(f"✓ {cmd}")
                    if output.stdout.strip():
                        results.append(f"  {output.stdout.strip()[:100]}")
                else:
                    results.append(f"✗ {cmd}")
                    if output.stderr.strip():
                        results.append(f"  Error: {output.stderr.strip()[:100]}")
            except Exception as e:
                results.append(f"✗ {cmd}")
                results.append(f"  Error: {str(e)}")
        
        self.pending_commands = []
        return "Executed:\n" + '\n'.join(results)
    
    def generate_with_tools(self, messages: List[Message], tools: Optional[List[Tool]] = None) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """
        Generate a response, potentially including tool calls.
        
        Returns: (content, tool_calls) - one will be None
        """
        # Check if the last message is a tool result
        if messages and messages[-1].role == "tool":
            tool_result = messages[-1].get_text_content()
            tool_call_id = messages[-1].tool_call_id
            logger.info(f"Processing tool result (id={tool_call_id}): {tool_result[:100]}...")
            
            # Find the original user request before the tool call
            user_message = None
            for msg in reversed(messages):
                if msg.role == "user":
                    user_message = msg.get_text_content()
                    break
            
            # Generate a response that incorporates the tool result
            if tool_result:
                # Check if the result is a file path that we should read
                if "Content saved to:" in tool_result or "saved to:" in tool_result.lower():
                    # Extract the file path and try to read it
                    import re
                    path_match = re.search(r'saved to[:\s]+([^\s\n]+)', tool_result, re.IGNORECASE)
                    if path_match:
                        file_path = path_match.group(1).strip()
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            # If it's HTML, try to extract text content
                            if '<html' in content.lower() or '<!doctype' in content.lower():
                                # Simple HTML text extraction
                                import re as regex
                                # Remove script and style elements
                                content = regex.sub(r'<script[^>]*>.*?</script>', '', content, flags=regex.DOTALL | regex.IGNORECASE)
                                content = regex.sub(r'<style[^>]*>.*?</style>', '', content, flags=regex.DOTALL | regex.IGNORECASE)
                                # Remove HTML tags
                                content = regex.sub(r'<[^>]+>', ' ', content)
                                # Clean up whitespace
                                content = regex.sub(r'\s+', ' ', content).strip()
                                # Decode HTML entities
                                content = content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                            
                            # Truncate if too long
                            if len(content) > 2000:
                                content = content[:2000] + "\n\n... (truncated)"
                            return f"Here's what I found:\n\n{content}", None
                        except Exception as e:
                            logger.warning(f"Could not read file {file_path}: {e}")
                
                # Simple response that presents the tool result
                return f"Here's what I found:\n\n{tool_result}", None
            else:
                return "The tool completed but returned no output.", None
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.get_text_content()
                break
        
        if not user_message:
            return "I need a question to answer.", None
        
        # Handle Goose session description requests (no tools, asking for short description)
        system_msg = next((m.get_text_content() for m in messages if m.role == "system"), "")
        if "description in four words or less" in system_msg.lower() or "reply *only* with the description" in user_message.lower():
            # Extract key words from the user messages mentioned in the request
            words = []
            for word in ["file", "list", "read", "plot", "sine", "create", "run"]:
                if word in user_message.lower():
                    words.append(word)
            if words:
                return " ".join(words[:4]).title(), None
            return "General Query", None
        
        # Filter out Goose system prompt if present
        goose_prefix = "You are a general-purpose AI agent called goose"
        if user_message.startswith(goose_prefix):
            parts = user_message.split("\n\n", 1)
            if len(parts) > 1:
                user_message = parts[-1].strip()
        
        # Use the new intent classifier
        intent_match = self.intent_classifier.classify(user_message)
        logger.info(f"Intent: {intent_match.intent.value}, confidence: {intent_match.confidence}, reason: {intent_match.reason}")
        
        # If tools are available and intent is TOOL_CALL, generate tool call
        if tools and intent_match.intent == Intent.TOOL_CALL:
            logger.info(f"Available tools from client: {[t.function.name for t in tools]}")
            
            # Map our tool names to Goose tool name patterns (order matters - more specific first)
            TOOL_NAME_MAP = {
                'Glob': ['developer__shell'],  # Use shell for ls/find commands
                'Read': ['developer__text_editor', 'developer__shell'],  # text_editor can read files
                'Bash': ['developer__shell'],  # Shell for running commands
                'Write': ['developer__text_editor'],  # text_editor for writing
                'Edit': ['developer__text_editor'],  # text_editor for editing
                'Grep': ['developer__shell'],  # Shell for grep
            }
            
            # Find a matching tool - try exact match first, then pattern match
            matching_tool = None
            tool_to_use = intent_match.tool_name
            
            # First try exact match
            for tool in tools:
                if tool.function.name == intent_match.tool_name:
                    matching_tool = tool
                    break
            
            # If no exact match, try to find a tool that matches our intent
            if not matching_tool and intent_match.tool_name in TOOL_NAME_MAP:
                patterns = TOOL_NAME_MAP[intent_match.tool_name]
                for tool in tools:
                    tool_lower = tool.function.name.lower()
                    for pattern in patterns:
                        if pattern in tool_lower:
                            matching_tool = tool
                            tool_to_use = tool.function.name  # Use the actual tool name
                            logger.info(f"Mapped {intent_match.tool_name} -> {tool_to_use}")
                            break
                    if matching_tool:
                        break
            
            # If still no match, don't use a random tool - return text explaining what we need
            if not matching_tool:
                logger.info(f"No matching tool found for intent {intent_match.tool_name}. Available: {[t.function.name for t in tools]}")
                # Return a helpful text response instead of calling wrong tool
                return f"I'd like to help with that, but I need shell/file access tools. The available tools ({', '.join(t.function.name for t in tools)}) don't include file system operations. Please enable the developer extension in Goose.", None
            
            if matching_tool:
                # Generate tool call with the actual tool name Goose expects
                import uuid
                
                # Transform arguments for Goose's developer__shell tool
                # It expects {"command": "actual shell command"}
                tool_args = intent_match.tool_args
                if tool_to_use == "developer__shell":
                    # Convert our intent args to shell commands
                    if intent_match.tool_name == "Glob":
                        # List files - use ls command
                        path = tool_args.get("pattern", ".") or "."
                        if path in ["current directory", "here", "this directory", "."]:
                            path = "."
                        tool_args = {"command": f"ls -la {path}"}
                    elif intent_match.tool_name == "Bash":
                        # Already a command
                        cmd = tool_args.get("command", "")
                        tool_args = {"command": cmd}
                    elif intent_match.tool_name == "Grep":
                        # Search command
                        query = tool_args.get("query", "")
                        tool_args = {"command": f"grep -r '{query}' ."}
                    elif intent_match.tool_name == "Read":
                        # Cat file
                        path = tool_args.get("file_path", "")
                        tool_args = {"command": f"cat {path}"}
                
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=tool_to_use,
                        arguments=json.dumps(tool_args)
                    )
                )
                logger.info(f"Generated tool call: {tool_to_use}({tool_args})")
                return None, [tool_call]
        
        # Handle based on intent from our classifier (not the legacy one in generate())
        if intent_match.intent == Intent.KNOWLEDGE:
            # Knowledge query - use emergent chain directly
            response = self.chain.chat(user_message)
            
            # Check for empty or generic fallback responses
            generic_responses = [
                "I understand. Let me help you with that.",
                "I don't have enough information",
                "I'm not sure",
                "I found something related:",  # This is a fallback response
                "I don't have information about that",  # Another fallback
                "I can discuss:",  # Listing topics means we don't know the answer
            ]
            is_generic = not response or response.strip() == "" or any(g in response for g in generic_responses)
            if is_generic:
                # Extract the topic from the query
                topic = user_message.lower().replace("what is ", "").replace("what are ", "").replace("explain ", "").replace("?", "").strip()
                
                # If tools are available, generate a tool call to look up the information via LLM
                if tools:
                    # Look for developer__shell to call LLM via curl
                    shell_tool = None
                    for tool in tools:
                        if tool.function.name == 'developer__shell':
                            shell_tool = tool
                            break
                    
                    if shell_tool:
                        # Use shell to call LLM API with a well-crafted prompt
                        import uuid
                        
                        # Build a curl command to query an LLM
                        prompt = f"Explain what {topic} is in 2-3 concise sentences. Be informative and accurate."
                        # Escape the prompt for shell
                        escaped_prompt = prompt.replace('"', '\\"')
                        
                        # Call our own LLM endpoint or a local LLM
                        curl_cmd = f'''curl -s http://127.0.0.1:11434/api/generate -d '{{"model": "qwen2.5:14b", "prompt": "{escaped_prompt}", "stream": false}}' | jq -r '.response' '''
                        
                        tool_args = {"command": curl_cmd}
                        
                        tool_call = ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=FunctionCall(
                                name=shell_tool.function.name,
                                arguments=json.dumps(tool_args)
                            )
                        )
                        logger.info(f"Knowledge not found for '{topic}', generating LLM lookup via shell")
                        return f"I don't have '{topic}' in my knowledge base. Let me look that up for you.", [tool_call]
                
                # No tools available - return helpful message
                response = f"I don't have specific information about '{topic}' in my knowledge base yet. You could ask me to create a visualization instead, like 'create a {topic} example plot'."
            return response, None
        
        elif intent_match.intent == Intent.CODE_GENERATION:
            # Code generation - use code orchestrator directly
            if self.code_orchestrator:
                plan = self.code_orchestrator.generate(user_message)
                
                if plan.complete_code:
                    response = f"```python\n{plan.complete_code}\n```"
                    response += f"\n\n*Generated via CodeOrchestrator ({len(plan.functions)} functions)*"
                    if plan.verified:
                        response += f"\n✓ Code verified"
                    if plan.output:
                        response += f"\nOutput: {plan.output[:200]}"
                    if plan.error:
                        response += f"\n⚠ {plan.error}"
                    
                    # Auto-execute plot code and save to run_graph.py
                    if 'plt.' in plan.complete_code or 'matplotlib' in plan.complete_code:
                        exec_result = self._execute_plot_code(plan.complete_code)
                        response += f"\n\n{exec_result}"
                    
                    return response, None
                else:
                    return f"Failed to generate code: {plan.error}", None
            
            # Fallback to python_gear
            if self.python_gear:
                result = self.python_gear.generate_from_text(user_message)
                if result.success:
                    return f"```python\n{result.code}\n```", None
                else:
                    return f"Failed to generate code: {result.error}", None
            
            return "Code generation not available", None
        
        elif intent_match.intent == Intent.UNSUPPORTED:
            # Unsupported - provide a helpful response
            return "I'm not sure how to help with that. I can help with:\n- Creating plots and visualizations\n- File operations (list, read, write)\n- Running shell commands\n- Answering questions about topics I know", None
        
        # Fallback to generate for anything else
        response = self.generate(messages)
        return response, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self.chain.get_stats()
        stats['tools_enabled'] = self.enable_tools
        return stats


def create_app(
    llm_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    seed_topics: List[str] = None,
    corpus_path: str = None,
    lazy_init: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Emergent Chat API",
        description="OpenAI-compatible API for Emergent Conversational Chat. "
                    "All responses are generated using emergent patterns - no LLM during conversation. "
                    "Now with gear-based routing for tool calls.",
        version="1.0.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize engine (lazy = don't build corpus yet)
    engine = EmergentChatEngine(
        llm_url=llm_url,
        llm_model=llm_model,
        seed_topics=seed_topics,
        corpus_path=corpus_path,
        lazy_init=lazy_init,
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        stats = engine.get_stats()
        return {
            "status": "healthy",
            "engine": "emergent-chat",
            "topics": stats['topics'],
            "corpus_items": stats['corpus_items'],
            "llm_calls_during_chat": stats['conversation_calls'],
        }
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id="emergent-chat",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
                ModelInfo(
                    id="emergent-chat-expanded",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
            ]
        )
    
    @app.get("/stats")
    async def get_stats():
        """Get engine statistics."""
        return engine.get_stats()
    
    @app.get("/topics")
    async def list_topics():
        """List known topics."""
        return {"topics": engine.chain.list_topics()}
    
    @app.post("/learn")
    async def learn_topic(request: Request):
        """Learn about a new topic using LLM as knowledge resource."""
        data = await request.json()
        topic = data.get("topic", "")
        if not topic:
            raise HTTPException(status_code=400, detail="Topic required")
        
        logger.info(f"Learning about topic: {topic}")
        success = engine.chain.learn_topic(topic)
        if success:
            stats = engine.chain.get_stats()
            return {
                "status": "success", 
                "topic": topic,
                "topics": stats['topics'],
                "corpus_items": stats['corpus_items'],
            }
        raise HTTPException(status_code=500, detail="Failed to learn topic")
    
    @app.post("/save")
    async def save_corpus(request: Request):
        """Save the current corpus to a file."""
        data = await request.json()
        from truthspace_lcm.corpus import CORPUS_DIR
        path = data.get("path", "chat_corpus.json")
        
        # Resolve relative paths to corpus directory
        if not Path(path).is_absolute():
            path = str(CORPUS_DIR / path)
        
        logger.info(f"Saving corpus to: {path}")
        engine.chain.save_corpus(path)
        
        stats = engine.chain.get_stats()
        default_items = len(engine.chain.default_corpus.all_items) if engine.chain.default_corpus else 0
        
        return {
            "status": "success",
            "path": path,
            "topics": stats['topics'],
            "corpus_items": stats['corpus_items'],
            "default_corpus_items": default_items,
        }
    
    @app.post("/reload")
    async def reload_corpus(request: Request):
        """Reload the corpus from a file."""
        data = await request.json()
        from truthspace_lcm.corpus import CORPUS_DIR
        path = data.get("path", "chat_corpus.json")
        
        # Resolve relative paths to corpus directory
        if not Path(path).is_absolute():
            path = str(CORPUS_DIR / path)
        
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Corpus file not found: {path}")
        
        logger.info(f"Reloading corpus from: {path}")
        engine.chain.load_corpus(path)
        
        stats = engine.chain.get_stats()
        default_items = len(engine.chain.default_corpus.all_items) if engine.chain.default_corpus else 0
        
        return {
            "status": "success",
            "path": path,
            "topics": stats['topics'],
            "corpus_items": stats['corpus_items'],
            "default_corpus_items": default_items,
        }
    
    @app.get("/corpus")
    async def get_corpus_info():
        """Get information about the current corpus."""
        stats = engine.chain.get_stats()
        
        result = {
            "topics": stats['topics'],
            "corpus_items": stats['corpus_items'],
            "definitions": stats['definitions'],
            "dimensions": stats['dimensions'],
            "conversation_calls": stats['conversation_calls'],
        }
        
        if engine.chain.default_corpus:
            dc_stats = engine.chain.default_corpus.get_stats()
            result['default_corpus'] = {
                "total_items": dc_stats['total_items'],
                "categories": dc_stats['categories'],
                "build_iterations": dc_stats['build_stats']['iterations'],
            }
        
        return result
    
    @app.post("/build")
    async def build_corpus():
        """Run one iteration of corpus self-building."""
        if not engine.chain.default_corpus:
            raise HTTPException(status_code=400, detail="Default corpus not available")
        
        result = engine.chain.default_corpus.build_iteration()
        
        return {
            "status": "success",
            "iteration": result['iteration'],
            "items_added": result['items_added'],
            "items_refined": result['items_refined'],
            "total_items": len(engine.chain.default_corpus.all_items),
        }
    
    @app.get("/books")
    async def list_books():
        """List available books from Project Gutenberg."""
        return {"books": engine.chain.get_available_books()}
    
    @app.post("/load_book")
    async def load_book(request: Request):
        """Load a literary work and build corpus from it."""
        data = await request.json()
        book_name = data.get("book_name")
        url = data.get("url")
        max_lines = data.get("max_lines")
        
        if not book_name and not url:
            raise HTTPException(status_code=400, detail="book_name or url required")
        
        logger.info(f"Loading book: {book_name or url}")
        
        success = engine.chain.load_book(
            book_name=book_name,
            url=url,
            max_lines=max_lines,
        )
        
        if success:
            stats = engine.chain.get_stats()
            return {
                "status": "success",
                "book": getattr(engine.chain, 'book_title', book_name),
                "topics": stats['topics'],
                "corpus_items": stats['corpus_items'],
            }
        raise HTTPException(status_code=500, detail="Failed to load book")
    
    @app.post("/refinement/enable")
    async def enable_refinement(request: Request):
        """Enable or disable automatic response refinement."""
        data = await request.json()
        enabled = data.get("enabled", True)
        threshold = data.get("threshold", 7.0)
        
        engine.chain.enable_refinement(enabled, threshold)
        
        return {
            "status": "success",
            "refinement_enabled": enabled,
            "threshold": threshold,
            "refinement_gear_available": engine.chain.refinement_gear is not None,
        }
    
    @app.post("/refinement/chat")
    async def chat_with_refinement(request: Request):
        """Chat with detailed refinement information."""
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="message required")
        
        result = engine.chain.chat_with_details(message)
        
        return {
            "response": result['response'],
            "original": result.get('original'),
            "topics": result.get('topics', []),
            "refined": result.get('refined', False),
            "score_before": result.get('score_before'),
            "score_after": result.get('score_after'),
            "feedback": result.get('feedback'),
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Chat completions endpoint (OpenAI-compatible with tool calling support)."""
        
        raw_body = await request.json()
        
        # Parse into our model
        try:
            parsed = ChatCompletionRequest(**raw_body)
        except Exception as e:
            logger.error(f"Failed to parse request: {e}")
            raise HTTPException(status_code=422, detail=str(e))
        
        logger.info(f"Request: model={parsed.model}, stream={parsed.stream}, tools={len(parsed.tools) if parsed.tools else 0}")
        
        request = parsed  # Use parsed request from here
        
        # Generate response (with potential tool calls)
        try:
            response_text, tool_calls = engine.generate_with_tools(request.messages, request.tools)
            if tool_calls:
                logger.info(f"Generated tool calls: {[tc.function.name for tc in tool_calls]}")
            else:
                logger.info(f"Generated response: {(response_text or '')[:100]}...")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                
                if tool_calls:
                    # Stream tool calls
                    for tc in tool_calls:
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    }]
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Send finish chunk for tool calls
                    finish_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                else:
                    # Stream text content
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": response_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Send finish chunk
                    finish_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response (with tool call support)
        if tool_calls:
            # Response with tool calls
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ResponseMessage(
                            role="assistant",
                            content=None,
                            tool_calls=tool_calls,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=Usage(
                    prompt_tokens=sum(len(m.get_text_content().split()) for m in request.messages),
                    completion_tokens=10,  # Approximate for tool calls
                    total_tokens=sum(len(m.get_text_content().split()) for m in request.messages) + 10,
                ),
            )
        else:
            # Regular text response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ResponseMessage(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=sum(len(m.get_text_content().split()) for m in request.messages),
                    completion_tokens=len((response_text or "").split()),
                    total_tokens=sum(len(m.get_text_content().split()) for m in request.messages) + len((response_text or "").split()),
                ),
            )
    
    return app


def get_app():
    """Factory function for creating the app."""
    return create_app()


# Module-level app instance for uvicorn (lazy init - use /load_book to load content)
app = create_app(lazy_init=True)


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Emergent Chat API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--topics", nargs="+", help="Seed topics for knowledge building")
    args = parser.parse_args()
    
    # Create app with settings
    app = create_app(
        corpus_path=args.corpus,
        seed_topics=args.topics,
    )
    
    uvicorn.run(app, host=args.host, port=args.port)
