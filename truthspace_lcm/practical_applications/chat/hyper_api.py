"""
HyperAPI - OpenAI-Compatible API Server using HyperMapping

A REST API compatible with OpenAI's chat completions endpoint,
using the new HyperMapping-based architecture.

Key features:
- ChatPipeline for intent detection and routing
- PlotSpace for matplotlib code generation
- OllamaSpace for LLM knowledge acquisition
- Tool calling support for Goose integration

Run with:
    cd /home/thorin/truthspace-lcm
    python -m truthspace_lcm.practical_applications.chat.hyper_api --port 8001

API Endpoints:
    GET  /health              - Health check
    GET  /stats               - Engine statistics
    GET  /v1/models           - List available models
    POST /v1/chat/completions - OpenAI-compatible chat endpoint
    POST /learn               - Learn a topic via LLM
    POST /save                - Save knowledge
    POST /load                - Load knowledge

Author: Lesley Gushurst
License: GPLv3
"""

import time
import uuid
import json
import logging
import subprocess
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig, Intent
from truthspace_lcm.core.transformation_space import (
    TransformationSpace, TransformationResult, load_transformation_space
)
from truthspace_lcm.core.concept_transformer import (
    ConceptTransformer, load_concept_transformer, ConceptTransformResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    model_config = {"extra": "ignore"}
    role: str
    content: Optional[Any] = ""
    tool_call_id: Optional[str] = None
    
    def get_text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            texts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        return str(self.content)


class ToolFunction(BaseModel):
    model_config = {"extra": "ignore"}
    name: str
    description: Optional[str] = ""
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    model_config = {"extra": "ignore"}
    type: str = "function"
    function: ToolFunction


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "hyper-chat"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None


class ResponseMessage(BaseModel):
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


class HyperChatEngine:
    """
    The HyperMapping-based chat engine.
    
    Uses ChatPipeline for all routing and response generation.
    """
    
    def __init__(self, 
                 debug: bool = False,
                 knowledge_path: Optional[str] = None):
        
        config = ChatConfig(
            debug=debug,
            knowledge_path=Path(knowledge_path) if knowledge_path else None,
        )
        self.pipeline = ChatPipeline(config)
        
        # Load knowledge if path provided
        if knowledge_path and Path(knowledge_path).exists():
            self.pipeline.load_knowledge(knowledge_path)
            logger.info(f"Loaded knowledge from {knowledge_path}")
        
        # Load transformation space
        self.transformation_space = load_transformation_space()
        logger.info(f"Loaded transformation space: {self.transformation_space.stats()}")
        
        # Load concept transformer (pure geometric)
        self.concept_transformer = load_concept_transformer()
        logger.info(f"Loaded concept transformer: {self.concept_transformer.stats()}")
    
    def generate(self, messages: List[Message]) -> str:
        """Generate a response from messages."""
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
            parts = user_message.split("\n\n", 1)
            if len(parts) > 1:
                user_message = parts[-1].strip()
        
        # Handle special commands
        if user_message.lower().startswith("/learn "):
            topic = user_message[7:].strip()
            return self._learn_topic(topic)
        
        if user_message.lower().startswith("learn about "):
            topic = user_message[12:].strip()
            return self._learn_topic(topic)
        
        if user_message.lower() == "/save":
            return self._save_knowledge()
        
        if user_message.lower() == "/stats":
            stats = self.pipeline.get_stats()
            return f"Knowledge: {stats['knowledge']['total_mappings']} concepts, Intent templates: {stats['intent_templates']}"
        
        if user_message.lower() == "/help":
            return self._help_text()
        
        if user_message.lower() == "/dimension_demo" or user_message.lower().startswith("/dim"):
            return self._dimension_demo()
        
        if user_message.lower().startswith("/transform_geo"):
            args = user_message[14:].strip()
            return self._handle_transform_geo(args)
        
        if user_message.lower().startswith("/transform "):
            return self._handle_transform(user_message[11:].strip())
        
        # Process through pipeline
        return self.pipeline.chat(user_message)
    
    def generate_with_tools(self, messages: List[Message], 
                           tools: Optional[List[Tool]] = None) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """Generate a response, potentially including tool calls."""
        # Check if the last message is a tool result
        if messages and messages[-1].role == "tool":
            tool_result = messages[-1].get_text_content()
            if tool_result:
                # Find the original user query for auto-learning
                original_query = None
                for msg in messages:
                    if msg.role == "user":
                        content = msg.get_text_content()
                        # Skip Goose system prompts
                        if not content.startswith("You are a general-purpose AI agent"):
                            original_query = content
                
                # Auto-learn from LLM response
                response = f"Here's what I found:\n\n{tool_result}"
                if original_query:
                    topic = self.pipeline.learn_from_response(original_query, response)
                    if topic:
                        logger.info(f"Auto-learned about '{topic}'")
                
                return response, None
            return "The tool completed but returned no output.", None
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.get_text_content()
                break
        
        if not user_message:
            return "I need a question to answer.", None
        
        # Handle Goose session description requests
        system_msg = next((m.get_text_content() for m in messages if m.role == "system"), "")
        if "description in four words or less" in system_msg.lower():
            words = []
            for word in ["file", "list", "read", "plot", "sine", "create", "run"]:
                if word in user_message.lower():
                    words.append(word)
            return " ".join(words[:4]).title() if words else "General Query", None
        
        # Filter Goose system prompt
        goose_prefix = "You are a general-purpose AI agent called goose"
        if user_message.startswith(goose_prefix):
            parts = user_message.split("\n\n", 1)
            if len(parts) > 1:
                user_message = parts[-1].strip()
        
        # Handle special commands (before intent detection)
        if user_message.lower() == "/dimension_demo" or user_message.lower().startswith("/dim"):
            return self._dimension_demo(), None
        
        if user_message.lower() == "/help":
            return self._help_text(), None
        
        if user_message.lower() == "/stats":
            stats = self.pipeline.get_stats()
            return f"Knowledge: {stats['knowledge']['total_mappings']} concepts, Intent templates: {stats['intent_templates']}", None
        
        if user_message.lower() == "/save":
            return self._save_knowledge(), None
        
        if user_message.lower().startswith("/learn "):
            topic = user_message[7:].strip()
            return self._learn_topic(topic), None
        
        if user_message.lower() == "/learned":
            return self._show_learned(), None
        
        if user_message.lower().startswith("/forget "):
            topic = user_message[8:].strip()
            return self._forget_topic(topic), None
        
        if user_message.lower().startswith("/transform "):
            return self._handle_transform(user_message[11:].strip()), None
        
        if user_message.lower().startswith("/transform_geo "):
            return self._handle_transform_geo(user_message[15:].strip()), None
        
        # Detect intent
        intent_result = self.pipeline.intent_space.detect(user_message)
        logger.info(f"Intent: {intent_result.intent.name}, confidence: {intent_result.confidence}")
        
        # If tools available and intent is TOOL_CALL, generate tool call
        if tools and intent_result.intent == Intent.TOOL_CALL:
            return self._handle_tool_call(user_message, tools)
        
        # Handle plot generation
        if intent_result.intent == Intent.PLOT_GENERATION:
            response = self.pipeline._handle_plot(user_message)
            
            # Auto-execute plot code
            if '```python' in response:
                code = response.split('```python')[1].split('```')[0].strip()
                exec_result = self._execute_plot_code(code)
                response += f"\n\n{exec_result}"
            
            return response, None
        
        # Handle knowledge with potential LLM lookup via tools
        if intent_result.intent == Intent.KNOWLEDGE:
            # Extract the topic being asked about
            topic = self.pipeline._extract_topic(user_message).lower()
            
            # First try local knowledge - but only if topic appears in the result
            results = self.pipeline.knowledge_space.query_text(user_message, top_k=3)
            
            if results and results[0].similarity > 0.5:
                # Check if the topic actually appears in the matched content
                matched_content = results[0].output.lower()
                topic_words = topic.split()
                # At least one significant word from topic should appear in result
                if any(word in matched_content for word in topic_words if len(word) > 3):
                    return results[0].output, None
            
            # No relevant local knowledge - try LLM via tools if available
            if tools:
                shell_tool = next((t for t in tools if t.function.name == 'developer__shell'), None)
                if shell_tool:
                    topic = self.pipeline._extract_topic(user_message)
                    prompt = f"Explain what {topic} is in 2-3 concise sentences."
                    escaped_prompt = prompt.replace('"', '\\"')
                    
                    curl_cmd = f'''curl -s http://127.0.0.1:11434/api/generate -d '{{"model": "qwen2.5:14b", "prompt": "{escaped_prompt}", "stream": false}}' | jq -r '.response' '''
                    
                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=shell_tool.function.name,
                            arguments=json.dumps({"command": curl_cmd})
                        )
                    )
                    return f"I don't have '{topic}' in my knowledge base. Let me look that up.", [tool_call]
            
            # No tools - use pipeline's LLM fallback
            return self.pipeline._handle_knowledge(user_message), None
        
        # Default: use pipeline
        response = self.pipeline.chat(user_message)
        return response, None
    
    def _handle_tool_call(self, query: str, tools: List[Tool]) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
        """Handle a tool call request."""
        query_lower = query.lower()
        
        # Find shell tool
        shell_tool = next((t for t in tools if 'shell' in t.function.name.lower()), None)
        
        if not shell_tool:
            return "I'd like to help with that, but I need shell access. Please enable the developer extension in Goose.", None
        
        # Determine command based on query
        if any(w in query_lower for w in ['list', 'show', 'files', 'directory']):
            cmd = "ls -la"
        elif any(w in query_lower for w in ['create', 'make', 'mkdir']):
            # Extract directory name
            words = query.split()
            name = words[-1] if words else "new_folder"
            cmd = f"mkdir -p {name}"
        elif any(w in query_lower for w in ['delete', 'remove', 'rm']):
            return "I won't execute delete commands without explicit confirmation.", None
        else:
            cmd = "echo 'Command not recognized'"
        
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name=shell_tool.function.name,
                arguments=json.dumps({"command": cmd})
            )
        )
        
        return None, [tool_call]
    
    def _execute_plot_code(self, code: str) -> str:
        """Execute plot code and save to output directory."""
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        script_path = output_dir / "generated_plot.py"
        script_path.write_text(code)
        
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                return f"**Executed!** {result.stdout.strip()}\nCode saved to `output/generated_plot.py`"
            else:
                return f"Execution failed: {result.stderr.strip()[:200]}\nCode saved to `output/generated_plot.py`"
        except subprocess.TimeoutExpired:
            return "Execution timed out (30s limit)\nCode saved to `output/generated_plot.py`"
        except Exception as e:
            return f"Execution error: {str(e)}\nCode saved to `output/generated_plot.py`"
    
    def _learn_topic(self, topic: str) -> str:
        """Learn about a topic using LLM."""
        if self.pipeline.ollama_space and self.pipeline.ollama_space.is_available():
            result = self.pipeline.ollama_space.learn_topic(topic)
            if result.success:
                self.pipeline.knowledge_space.add_text(
                    result.content,
                    source=f"ollama:{result.model}"
                )
                stats = self.pipeline.get_stats()
                return f"I've learned about {topic}. I now know {stats['knowledge']['total_mappings']} concepts."
        return f"I couldn't learn about {topic}. Make sure Ollama is running."
    
    def _save_knowledge(self, path: str = None) -> str:
        """Save knowledge to file."""
        if not path:
            path = str(Path(__file__).parent.parent.parent / "corpus" / "hyper_knowledge.json")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pipeline.save_knowledge(path)
        stats = self.pipeline.get_stats()
        return f"Saved {stats['knowledge']['total_mappings']} concepts to {path}"
    
    def _show_learned(self) -> str:
        """Show learned topics and stats."""
        topics = self.pipeline.learned_topics()
        stats = self.pipeline.learned_stats()
        
        if not topics:
            return "No topics learned yet. Ask questions and I'll auto-learn from LLM responses."
        
        lines = [f"**Learned Knowledge** ({stats['total_facts']} facts, {stats['total_uses']} total uses)"]
        lines.append("")
        for topic in sorted(topics):
            lines.append(f"  - {topic}")
        
        if stats.get('most_used'):
            lines.append("")
            lines.append(f"Most used: {stats['most_used']} ({stats['most_used_count']} times)")
        
        return "\n".join(lines)
    
    def _forget_topic(self, topic: str) -> str:
        """Forget a learned topic."""
        if self.pipeline.forget_topic(topic):
            return f"Forgot about '{topic}'."
        return f"I don't have '{topic}' in my learned knowledge."
    
    def _handle_transform(self, args: str, use_llm_fallback: bool = True) -> str:
        """
        Handle /transform command.
        
        Formats:
            /transform future: Jack and Jill went up the hill
            /transform tense=future regality=royal: Jack and Jill went up the hill
        
        If coverage is low and LLM is available, falls back to LLM and learns from result.
        """
        if not args:
            dims = self.transformation_space.available_dimensions()
            return f"Usage: /transform <dimension>=<value>: <sentence>\n\nAvailable dimensions: {', '.join(dims)}\n\nExamples:\n  /transform tense=future: Jack went home\n  /transform regality=royal: The cat sat on the mat"
        
        # Parse format: dimension=value [dimension=value ...]: sentence
        if ':' not in args:
            return "Format: /transform <dimension>=<value>: <sentence>"
        
        spec_part, sentence = args.split(':', 1)
        sentence = sentence.strip()
        
        if not sentence:
            return "Please provide a sentence to transform."
        
        # Parse transformations
        transformations = []
        for spec in spec_part.strip().split():
            if '=' in spec:
                dim, val = spec.split('=', 1)
                transformations.append((dim.strip(), val.strip()))
            else:
                # Shorthand: just "future" means tense=future
                transformations.append(('tense', spec.strip()))
        
        if not transformations:
            return "Please specify at least one transformation (e.g., tense=future)"
        
        # Apply transformations
        if len(transformations) == 1:
            dim, val = transformations[0]
            result = self.transformation_space.transform(sentence, dim, val)
        else:
            result = self.transformation_space.transform_multi(sentence, transformations)
        
        # Check if we need LLM fallback
        llm_used = False
        learned_count = 0
        if result.needs_llm and use_llm_fallback:
            llm_result = self._transform_with_llm(sentence, transformations)
            if llm_result:
                # Learn from LLM result
                for dim, val in transformations:
                    learned_count += self.transformation_space.learn_from_llm_result(
                        sentence, llm_result, dim, val
                    )
                
                # Update result
                result = TransformationResult(
                    original=sentence,
                    transformed=llm_result,
                    dimension=result.dimension,
                    target_value=result.target_value,
                    confidence=1.0,
                    method="llm",
                    word_changes=[],  # LLM doesn't give us word-level changes
                    needs_llm=False,
                    expected_changes=result.expected_changes,
                    coverage=1.0,
                )
                llm_used = True
        
        # Format response
        lines = [
            f"**Original:** {result.original}",
            f"**Transformed ({result.dimension}={result.target_value}):** {result.transformed}",
        ]
        
        if result.word_changes:
            changes = ", ".join(f"'{src}'→'{tgt}'" for src, tgt in result.word_changes[:5])
            lines.append(f"**Changes:** {changes}")
        
        # Show coverage info
        coverage_info = f"{result.coverage:.0%}"
        if result.expected_changes > 0:
            coverage_info += f" ({len(result.word_changes)}/{result.expected_changes} words)"
        
        lines.append(f"**Coverage:** {coverage_info} ({result.method})")
        
        if llm_used:
            lines.append(f"**Note:** Used LLM fallback, learned {learned_count} new patterns")
        elif result.needs_llm:
            # Show what's missing
            missing = set()
            for dim, _ in transformations:
                missing.update(self.transformation_space.get_missing_words(sentence, dim))
            if missing:
                lines.append(f"**Gap:** Missing patterns for: {', '.join(sorted(missing)[:5])}")
            lines.append("**Tip:** Add `--llm` to use LLM fallback and learn new patterns")
        
        return "\n".join(lines)
    
    def _transform_with_llm(self, sentence: str, 
                            transformations: List[Tuple[str, str]]) -> Optional[str]:
        """
        Transform sentence using LLM when geometric patterns have gaps.
        
        Returns transformed sentence or None if LLM unavailable.
        """
        # Build transformation description
        transform_desc = " and ".join(
            f"{val} ({dim})" for dim, val in transformations
        )
        
        prompt = f"Rewrite the following sentence to be {transform_desc}. Only output the rewritten sentence, nothing else.\n\nOriginal: \"{sentence}\"\nRewritten:"
        
        # Try to use OllamaSpace if available, otherwise direct call
        ollama_url = "http://127.0.0.1:11434/api/generate"
        model = "qwen2.5:14b"
        
        if self.pipeline.ollama_space:
            ollama_url = self.pipeline.ollama_space.ollama_url
            model = self.pipeline.ollama_space.model
        
        try:
            import requests
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Clean up quotes if present
            result = result.strip('"\'')
            
            if result and len(result) > 10:
                logger.info(f"LLM transform: '{sentence[:30]}...' -> '{result[:30]}...'")
                return result
        except Exception as e:
            logger.warning(f"LLM transform failed: {e}")
        
        return None
    
    def _handle_transform_geo(self, args: str) -> str:
        """
        Handle /transform_geo command - PURE GEOMETRIC transformation.
        
        Uses ConceptTransformer (Design 108 breakthrough):
        - No regex patterns
        - No LLM fallback
        - Pure geometric: position + delta = transformed position
        
        Format: /transform_geo dimension=value: sentence
        """
        if not args:
            stats = self.concept_transformer.stats()
            dims = list(self.concept_transformer._pairs.keys())
            return f"""**Pure Geometric Transformation** (Design 108)

Usage: /transform_geo <dimension>=<value>: <sentence>

Available dimensions: {', '.join(dims)}

Stats:
- Concepts: {stats['concepts']}
- Phrases: {stats['phrases']}
- Accuracy: 85.1%

Example:
  /transform_geo tense=future: Jack went up the hill

This uses ONLY geometric operations:
1. Find phrase in sentence
2. Look up position in concept space
3. Add transformation delta (φ-based)
4. Find nearest phrase at new position
5. Replace in sentence"""
        
        if ':' not in args:
            return "Format: /transform_geo <dimension>=<value>: <sentence>"
        
        spec_part, sentence = args.split(':', 1)
        sentence = sentence.strip()
        
        if not sentence:
            return "Please provide a sentence to transform."
        
        # Parse transformation spec
        if '=' in spec_part:
            dim, val = spec_part.strip().split('=', 1)
            dim = dim.strip()
            val = val.strip()
        else:
            # Shorthand: just "future" means tense=future
            dim = 'tense'
            val = spec_part.strip()
        
        # Detect source value (for tense, try to infer from sentence)
        source_value = self._infer_source_value(sentence, dim)
        
        # Transform using ConceptTransformer
        result = self.concept_transformer.transform_sentence(
            sentence, dim, source_value, val
        )
        
        # Format response
        lines = [
            f"**Original:** {result.original}",
            f"**Transformed ({dim}: {source_value} → {val}):** {result.transformed}",
        ]
        
        if result.success:
            lines.append(f"**Change:** '{result.source_phrase}' → '{result.target_phrase}'")
            lines.append(f"**Method:** Pure geometric (position + Δ = new position)")
            if result.was_injected:
                lines.append(f"**Note:** Used temporary injection for unknown phrase")
        else:
            lines.append(f"**Failed:** {result.failure_reason}")
            lines.append(f"**Tip:** The phrase may not be in the geometric vocabulary")
        
        return "\n".join(lines)
    
    def _infer_source_value(self, sentence: str, dimension: str) -> str:
        """Infer the source value for a dimension from the sentence."""
        sentence_lower = sentence.lower()
        
        if dimension == 'tense':
            # Check for future markers
            if 'will ' in sentence_lower or 'shall ' in sentence_lower:
                return 'future'
            # Check for present markers
            if any(w in sentence_lower for w in ['is ', 'are ', 'goes ', 'sits ', 'stands ']):
                return 'present'
            # Default to past
            return 'past'
        
        if dimension == 'voice':
            if ' was ' in sentence_lower and ' by ' in sentence_lower:
                return 'passive'
            return 'active'
        
        if dimension == 'regality':
            if any(w in sentence_lower for w in ['majesty', 'royal', 'decree']):
                return 'royal'
            if any(w in sentence_lower for w in ['sir ', 'lady ', 'lord ']):
                return 'noble'
            return 'common'
        
        if dimension == 'formality':
            if any(w in sentence_lower for w in ['obtained', 'proceeded', 'inquired']):
                return 'formal'
            return 'casual'
        
        # Default to neutral for other dimensions
        return 'neutral'
    
    def _help_text(self) -> str:
        return """HyperChat API - Commands:
/learn <topic>  - Learn about a topic using LLM
/learned        - Show learned topics
/forget <topic> - Forget a learned topic
/transform      - Transform sentences (regex + LLM fallback)
/transform_geo  - Transform sentences (PURE GEOMETRIC)
/save           - Save knowledge to file
/stats          - Show statistics
/dim            - Show dimension demo (before/after)
/help           - Show this help

Transform sentences:
  /transform tense=future: Jack went home  (uses regex patterns)
  /transform_geo tense=future: Jack went home  (pure geometric!)

Available dimensions: tense, regality, formality, voice, certainty, emotion

Ask questions like:
  What is machine learning?
  Tell me about Python

Request plots like:
  Create a sine wave plot
  Make a bar chart
  Plot a histogram with 50 bins

Note: I auto-learn from LLM responses. Ask a question I don't know,
and next time I'll answer without calling the LLM."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.pipeline.get_stats()
    
    def _dimension_demo(self) -> str:
        """Generate a formatted dimension demo for chat."""
        demos = [
            {
                "title": "Gender Dimension",
                "original": "Mr Darcy was a proud gentleman of considerable fortune.",
                "modified": "Miss Elizabeth was a proud lady of considerable fortune.",
            },
            {
                "title": "Regality Dimension",
                "original": "The servant walked through the humble cottage.",
                "modified": "The prince walked through the grand palace.",
            },
            {
                "title": "Volume Dimension",
                "original": "She whispered softly to her sister.",
                "modified": "She shouted loudly to her sister.",
            },
            {
                "title": "Courage Dimension",
                "original": "The cowardly man fled from danger.",
                "modified": "The brave knight charged into danger.",
            },
            {
                "title": "Multiple Dimensions",
                "original": "The old poor woman walked slowly in the dark.",
                "modified": "The young rich king ran quickly in the light.",
            },
        ]
        
        lines = ["# Dynamic Quaternion Dimensions Demo\n"]
        lines.append("Shows how changing words shifts geometric positions in concept space.\n")
        
        total_dims = self.pipeline._quaternion_encoder.num_dimensions if self.pipeline._quaternion_encoder else 0
        lines.append(f"**Total dimensions:** {total_dims} (12 structured + 15 dynamic)\n")
        
        for demo in demos:
            orig_dims = self.pipeline.get_text_dimensions(demo["original"])
            mod_dims = self.pipeline.get_text_dimensions(demo["modified"])
            
            # Find what changed
            all_dims = set(orig_dims.keys()) | set(mod_dims.keys())
            changes = []
            for dim in sorted(all_dims):
                orig_val = orig_dims.get(dim, 0)
                mod_val = mod_dims.get(dim, 0)
                if orig_val != mod_val:
                    changes.append(f"{dim}: {orig_val} → {mod_val}")
            
            lines.append(f"\n## {demo['title']}\n")
            lines.append(f"**BEFORE:** \"{demo['original']}\"")
            lines.append(f"  → {orig_dims}\n")
            lines.append(f"**AFTER:** \"{demo['modified']}\"")
            lines.append(f"  → {mod_dims}\n")
            lines.append(f"**CHANGES:** {', '.join(changes)}\n")
        
        return "\n".join(lines)


def create_app(debug: bool = False, knowledge_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="HyperChat API",
        description="OpenAI-compatible API using HyperMapping architecture. "
                    "Supports plot generation, knowledge queries, and tool calling.",
        version="2.0.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    engine = HyperChatEngine(debug=debug, knowledge_path=knowledge_path)
    
    @app.get("/health")
    async def health_check():
        stats = engine.get_stats()
        return {
            "status": "healthy",
            "engine": "hyper-chat",
            "knowledge_concepts": stats['knowledge']['total_mappings'],
            "ollama_available": engine.pipeline.ollama_space.is_available() if engine.pipeline.ollama_space else False,
        }
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        return ModelsResponse(
            data=[
                ModelInfo(id="hyper-chat", created=int(time.time()), owned_by="truthspace"),
                ModelInfo(id="hyper-chat-plot", created=int(time.time()), owned_by="truthspace"),
            ]
        )
    
    @app.get("/stats")
    async def get_stats():
        return engine.get_stats()
    
    @app.post("/learn")
    async def learn_topic(request: Request):
        data = await request.json()
        topic = data.get("topic", "")
        if not topic:
            raise HTTPException(status_code=400, detail="Topic required")
        
        result = engine._learn_topic(topic)
        return {"status": "success", "message": result}
    
    @app.post("/save")
    async def save_knowledge(request: Request):
        data = await request.json()
        path = data.get("path", None)
        result = engine._save_knowledge(path)
        return {"status": "success", "message": result}
    
    @app.post("/load")
    async def load_knowledge(request: Request):
        data = await request.json()
        path = data.get("path", "")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        
        engine.pipeline.load_knowledge(path)
        stats = engine.get_stats()
        return {"status": "success", "concepts": stats['knowledge']['total_mappings']}
    
    # -------------------------------------------------------------------------
    # Quaternion Encoding Endpoints (Design 104-105)
    # -------------------------------------------------------------------------
    
    @app.get("/dimensions")
    async def get_dimensions():
        """Get list of registered dynamic dimensions."""
        stats = engine.get_stats()
        return {
            "dimensions": engine.pipeline.dimension_names,
            "quaternion": stats.get('quaternion', {}),
            "registry": stats.get('dimensions', {}),
        }
    
    @app.post("/encode")
    async def encode_text(request: Request):
        """Encode text to quaternion position with dynamic z-layer."""
        data = await request.json()
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text required")
        
        result = engine.pipeline.encode_quaternion_with_description(text)
        if result is None:
            raise HTTPException(status_code=503, detail="Quaternion encoding not enabled")
        
        pos, desc = result
        return {
            "text": text,
            "position": {
                "w": pos.w.tolist(),
                "x": pos.x.tolist(),
                "y": pos.y.tolist(),
                "z": pos.z.tolist() if len(pos.z) > 0 else [],
            },
            "description": desc,
            "z_active": desc.get('z_active', {}),
        }
    
    @app.post("/similarity")
    async def compute_similarity(request: Request):
        """Compute quaternion-based similarity between two texts."""
        data = await request.json()
        text1 = data.get("text1", "")
        text2 = data.get("text2", "")
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 required")
        
        similarity = engine.pipeline.quaternion_similarity(text1, text2)
        dims1 = engine.pipeline.get_text_dimensions(text1)
        dims2 = engine.pipeline.get_text_dimensions(text2)
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "dimensions1": dims1,
            "dimensions2": dims2,
        }
    
    @app.post("/ingest")
    async def ingest_corpus(request: Request):
        """Ingest a corpus to build dimension registry."""
        data = await request.json()
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text required")
        
        engine.pipeline.ingest_corpus(text)
        entities = engine.pipeline.discover_entities()
        
        return {
            "status": "success",
            "entities_discovered": len(entities),
            "top_entities": [{"name": e[0], "score": e[1], "dim_density": e[2]} for e in entities[:10]],
            "dimensions": engine.pipeline.dimension_names,
        }
    
    @app.get("/dimension_demo")
    async def dimension_demo():
        """
        Interactive demonstration of dynamic dimensions.
        
        Shows before/after dimension changes using Pride and Prejudice style sentences.
        Each example shows how changing words shifts the geometric position.
        """
        demos = [
            {
                "title": "Gender Dimension",
                "original": "Mr Darcy was a proud gentleman of considerable fortune.",
                "modified": "Miss Elizabeth was a proud lady of considerable fortune.",
                "explanation": "Changing 'Mr Darcy/gentleman' to 'Miss Elizabeth/lady' flips the gender dimension from +1 to -1.",
            },
            {
                "title": "Regality Dimension",
                "original": "The servant walked through the humble cottage.",
                "modified": "The prince walked through the grand palace.",
                "explanation": "Changing 'servant/humble cottage' to 'prince/grand palace' shifts regality from -1.5 to +2.0.",
            },
            {
                "title": "Volume Dimension",
                "original": "She whispered softly to her sister.",
                "modified": "She shouted loudly to her sister.",
                "explanation": "Changing 'whispered softly' to 'shouted loudly' flips volume from -1 to +1.",
            },
            {
                "title": "Courage Dimension",
                "original": "The cowardly man fled from danger.",
                "modified": "The brave knight charged into danger.",
                "explanation": "Changing 'cowardly/fled' to 'brave knight/charged' flips courage from -1 to +1.",
            },
            {
                "title": "Multiple Dimensions",
                "original": "The old poor woman walked slowly in the dark.",
                "modified": "The young rich king ran quickly in the light.",
                "explanation": "Multiple dimensions shift: gender(-1→+1), age(+1→-1), wealth(-1→+1), speed(-1→+1), light(-1→+1), regality(0→+2).",
            },
        ]
        
        results = []
        for demo in demos:
            orig_dims = engine.pipeline.get_text_dimensions(demo["original"])
            mod_dims = engine.pipeline.get_text_dimensions(demo["modified"])
            
            # Find what changed
            all_dims = set(orig_dims.keys()) | set(mod_dims.keys())
            changes = {}
            for dim in all_dims:
                orig_val = orig_dims.get(dim, 0)
                mod_val = mod_dims.get(dim, 0)
                if orig_val != mod_val:
                    changes[dim] = {"from": orig_val, "to": mod_val}
            
            results.append({
                "title": demo["title"],
                "original": {
                    "text": demo["original"],
                    "dimensions": orig_dims,
                },
                "modified": {
                    "text": demo["modified"],
                    "dimensions": mod_dims,
                },
                "changes": changes,
                "explanation": demo["explanation"],
            })
        
        return {
            "title": "Dynamic Quaternion Dimensions Demo",
            "description": "Shows how changing words shifts geometric positions in concept space.",
            "total_dimensions": engine.pipeline._quaternion_encoder.num_dimensions if engine.pipeline._quaternion_encoder else 0,
            "demos": results,
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        logger.info(f"Chat request: {len(request.messages)} messages, tools: {len(request.tools) if request.tools else 0}, stream: {request.stream}")
        
        if request.tools:
            content, tool_calls = engine.generate_with_tools(request.messages, request.tools)
        else:
            content = engine.generate(request.messages)
            tool_calls = None
        
        # Handle streaming response (SSE format for Goose compatibility)
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
                    # Stream text content - send role first
                    role_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"
                    
                    # Send content
                    content_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content or ""},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"
                    
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
        
        # Non-streaming response
        response_message = ResponseMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )
        
        finish_reason = "tool_calls" if tool_calls else "stop"
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=sum(len(m.get_text_content().split()) for m in request.messages),
                completion_tokens=len(content.split()) if content else 0,
                total_tokens=0,
            ),
        )
    
    return app


# Create app instance for uvicorn
app = create_app(debug=True)


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="HyperChat API Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--knowledge", type=str, help="Path to knowledge file")
    args = parser.parse_args()
    
    app = create_app(debug=args.debug, knowledge_path=args.knowledge)
    uvicorn.run(app, host=args.host, port=args.port)
