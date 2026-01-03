#!/usr/bin/env python3
"""
Run the Emergent Chat API Server

A simple script to start the chat API server with configurable options.

Usage:
    python run_api.py                          # Default settings
    python run_api.py --port 8001              # Custom port
    python run_api.py --corpus path/to/corpus.json  # Load from corpus
    python run_api.py --book moby_dick         # Load a literary work
    python run_api.py --topics python science  # Custom seed topics

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Emergent Chat API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_api.py                              # Start with default topics
  python run_api.py --port 8001                  # Custom port
  python run_api.py --book moby_dick             # Load Moby Dick
  python run_api.py --corpus corpus.json         # Load from JSON corpus
  python run_api.py --topics python AI science   # Custom seed topics
  python run_api.py --list-books                 # List available books

Available books:
  moby_dick, pride_and_prejudice, frankenstein, dracula,
  alice_in_wonderland, sherlock_holmes, war_and_peace,
  great_gatsby, jane_eyre, wuthering_heights
"""
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int, 
        default=8001, 
        help="Port to bind to (default: 8001)"
    )
    parser.add_argument(
        "--corpus", "-c",
        type=str, 
        help="Path to JSON corpus file to load"
    )
    parser.add_argument(
        "--book", "-b",
        type=str, 
        help="Name of book to load (e.g., moby_dick)"
    )
    parser.add_argument(
        "--topics", "-t",
        nargs="+", 
        help="Seed topics for knowledge building"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Maximum lines to process from book (default: all)"
    )
    parser.add_argument(
        "--list-books",
        action="store_true",
        help="List available books and exit"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Don't build initial corpus (start empty)"
    )
    
    args = parser.parse_args()
    
    # Handle --list-books
    if args.list_books:
        from truthspace_lcm.gears.core.conversational_chain import GUTENBERG_BOOKS
        print("Available books:")
        for name in GUTENBERG_BOOKS:
            print(f"  {name}")
        return 0
    
    # Import here to avoid slow startup for --help
    import uvicorn
    from truthspace_lcm.gears.core import ConversationalChain
    from truthspace_lcm.gears.practical_applications.chat.api_server import DEFAULT_LLM_URL, DEFAULT_LLM_MODEL
    
    print("=" * 60)
    print("EMERGENT CHAT API SERVER")
    print("=" * 60)
    
    # Create chain manually for more control
    chain = ConversationalChain()
    chain.configure_llm(DEFAULT_LLM_URL, DEFAULT_LLM_MODEL)
    
    # Load corpus based on options
    if args.corpus:
        corpus_path = Path(args.corpus).resolve()
        print(f"Loading corpus from: {corpus_path}")
        chain.load_corpus(str(corpus_path))
    elif args.book:
        book_name = args.book.lower().replace(' ', '_')
        print(f"Loading book: {book_name}")
        if not chain.load_book(book_name=book_name, max_lines=args.max_lines):
            print(f"Error: Failed to load book '{book_name}'")
            print("Use --list-books to see available books")
            return 1
        print(f"Loaded: {chain.book_title}")
    elif args.no_build:
        print("Starting with empty corpus")
    else:
        topics = args.topics or [
            "artificial intelligence",
            "machine learning",
            "programming",
            "python",
            "science",
        ]
        print(f"Building corpus from topics: {', '.join(topics)}")
        chain.build_corpus(topics, expand=True)
    
    stats = chain.get_stats()
    print(f"\nCorpus ready:")
    print(f"  Topics: {stats['topics']}")
    print(f"  Items: {stats['corpus_items']}")
    print(f"  Definitions: {stats['definitions']}")
    
    # Create custom app with our chain
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from typing import List, Optional
    import time
    import uuid
    import json
    
    app = FastAPI(
        title="Emergent Chat API",
        description="Truly emergent conversational chat - no LLM during conversation",
        version="1.0.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class Message(BaseModel):
        role: str
        content: str
    
    class ChatRequest(BaseModel):
        model: str = "emergent-chat"
        messages: List[Message]
        stream: Optional[bool] = False
    
    @app.get("/health")
    async def health():
        s = chain.get_stats()
        return {
            "status": "healthy",
            "topics": s['topics'],
            "corpus_items": s['corpus_items'],
            "book": getattr(chain, 'book_title', None),
        }
    
    @app.get("/stats")
    async def stats():
        return chain.get_stats()
    
    @app.get("/topics")
    async def topics():
        return {"topics": chain.list_topics()}
    
    @app.get("/books")
    async def books():
        return {"books": chain.get_available_books()}
    
    @app.post("/load_book")
    async def load_book(request: Request):
        data = await request.json()
        book_name = data.get("book_name")
        url = data.get("url")
        max_lines = data.get("max_lines")
        
        if chain.load_book(book_name=book_name, url=url, max_lines=max_lines):
            return {
                "status": "success",
                "book": getattr(chain, 'book_title', book_name),
                "stats": chain.get_stats(),
            }
        raise HTTPException(status_code=500, detail="Failed to load book")
    
    # Import the full EmergentChatEngine for intent classification and code generation
    from truthspace_lcm.gears.practical_applications.chat.api_server import EmergentChatEngine, Message as APIMessage
    
    # Create engine with our pre-built chain
    engine = EmergentChatEngine(lazy_init=True, enable_tools=True)
    engine.chain = chain  # Use the chain we already built
    engine._initialized = True
    
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        raw_body = await request.json()
        
        # Convert to API Message format
        messages = [APIMessage(**m) for m in raw_body.get("messages", [])]
        stream = raw_body.get("stream", False)
        model = raw_body.get("model", "emergent-chat")
        
        # Parse tools if provided (from Goose)
        tools = None
        if "tools" in raw_body:
            from truthspace_lcm.gears.practical_applications.chat.api_server import Tool
            tools = [Tool(**t) for t in raw_body["tools"]]
        
        # Use the full generate_with_tools which handles intent classification
        response_text, tool_calls = engine.generate_with_tools(messages, tools)
        
        if stream:
            async def stream_response():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                if tool_calls:
                    # First chunk: role
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None}}]})}\n\n"
                    # Tool call chunks
                    for i, tc in enumerate(tool_calls):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": i,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                                    }]
                                }
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    # Finish with tool_calls reason
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': response_text}}]})}\n\n"
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        # Build response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant"},
                "finish_reason": "stop" if not tool_calls else "tool_calls",
            }],
        }
        
        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
            response["choices"][0]["message"]["content"] = response_text  # Include the "looking it up" message
            import sys
            sys.stderr.write(f"[DEBUG] Returning tool_calls response with {len(tool_calls)} tool calls\n")
            sys.stderr.write(f"[DEBUG] Tool: {tool_calls[0].function.name}, args: {tool_calls[0].function.arguments[:100]}...\n")
            sys.stderr.flush()
        else:
            response["choices"][0]["message"]["content"] = response_text
        
        return response
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
