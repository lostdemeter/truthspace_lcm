#!/usr/bin/env python3
"""
Research Paper Writer v2 - With Tool Calling

An improved paper writer that:
1. Uses tools to search for concepts in documentation
2. Extracts actual content from source documents
3. Researches each section before writing
4. Edits papers section by section with diff tracking

Pipeline:
1. Create outline with section topics
2. For each section: RESEARCH → WRITE → VERIFY
3. Review full paper and iterate
"""

import torch
import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from concept_search import ConceptSearcher, SearchResult
from structured_editor import StructuredEditor, DocumentState

# Directories
PROJECT_ROOT = Path("/home/thorin/truthspace-lcm")
OUTPUT_DIR = PROJECT_ROOT / "phi_chat" / "paper_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# Tool definitions for the model
TOOLS = {
    "search": {
        "description": "Search design documents for a concept. Returns relevant excerpts.",
        "parameters": {"query": "string - the concept to search for"},
        "example": 'TOOL: {"tool": "search", "query": "φ-computer proof"}'
    },
    "get_doc": {
        "description": "Get full content of a specific document by number.",
        "parameters": {"doc_num": "integer - document number (e.g., 191)"},
        "example": 'TOOL: {"tool": "get_doc", "doc_num": 191}'
    },
    "search_in_doc": {
        "description": "Search within a specific document for a concept.",
        "parameters": {"doc_num": "integer", "query": "string"},
        "example": 'TOOL: {"tool": "search_in_doc", "doc_num": 177, "query": "scaffolding"}'
    },
    "write_section": {
        "description": "Write content to a section of the paper.",
        "parameters": {"section_title": "string", "content": "string"},
        "example": 'TOOL: {"tool": "write_section", "section_title": "Introduction", "content": "..."}'
    },
    "done": {
        "description": "Signal that the current task is complete.",
        "parameters": {"summary": "string - brief summary of what was done"},
        "example": 'TOOL: {"tool": "done", "summary": "Wrote introduction section"}'
    }
}


@dataclass
class PaperSection:
    """A section of the research paper."""
    title: str
    topics: List[str]  # Topics to research for this section
    content: str = ""
    sources: List[int] = field(default_factory=list)  # Doc numbers used
    status: str = "pending"  # pending, researching, writing, complete


@dataclass
class ResearchNote:
    """A note from research."""
    doc_num: int
    doc_title: str
    excerpt: str
    relevance: float


class PaperWriterV2:
    """
    Paper writer with tool-calling capability.
    
    The model can call tools to:
    - Search for concepts
    - Get document content
    - Write sections
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🚀 Loading Paper Writer v2 with Tool Calling...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Initialize tools
        self.searcher = ConceptSearcher()
        self.editor = StructuredEditor()
        
        # Paper state
        self.sections: List[PaperSection] = []
        self.research_notes: Dict[str, List[ResearchNote]] = {}  # section_title -> notes
        self.paper_content: Dict[str, str] = {}  # section_title -> content
        
        print("✓ Model and tools loaded!\n")
    
    def _build_tool_prompt(self) -> str:
        """Build the tool description for the system prompt."""
        tool_desc = []
        for name, info in TOOLS.items():
            tool_desc.append(f"- **{name}**: {info['description']}")
            tool_desc.append(f"  Parameters: {info['parameters']}")
            tool_desc.append(f"  Example: {info['example']}")
        return "\n".join(tool_desc)
    
    def generate(self, messages: List[Dict], max_tokens: int = 800) -> str:
        """Generate a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            response = parts[-1].strip()
        return response
    
    def _parse_tool_call(self, response: str) -> Optional[Tuple[str, Dict]]:
        """Parse a tool call from the model's response."""
        # Look for TOOL: {...}
        match = re.search(r'TOOL:\s*(\{[^}]+\})', response, re.IGNORECASE)
        if match:
            try:
                tool_data = json.loads(match.group(1))
                tool_name = tool_data.get("tool")
                return tool_name, tool_data
            except json.JSONDecodeError:
                pass
        return None
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Execute a tool and return the result."""
        if tool_name == "search":
            query = params.get("query", "")
            results = self.searcher.search(query, max_results=3)
            if not results:
                return f"No results found for: {query}"
            
            output = [f"## Search Results for: {query}\n"]
            for r in results:
                output.append(f"### Doc {r.doc_number}: {r.doc_title}")
                output.append(f"Relevance: {r.relevance_score:.1f}")
                if r.excerpts:
                    output.append(f"Excerpt: {r.excerpts[0][:400]}...")
                output.append("")
            return "\n".join(output)
        
        elif tool_name == "get_doc":
            doc_num = params.get("doc_num", 0)
            content = self.searcher.get_full_doc(doc_num)
            if not content:
                return f"Document {doc_num} not found"
            # Truncate if too long
            if len(content) > 3000:
                content = content[:2800] + "\n...[truncated]..."
            return f"## Document {doc_num} Content:\n\n{content}"
        
        elif tool_name == "search_in_doc":
            doc_num = params.get("doc_num", 0)
            query = params.get("query", "")
            excerpts = self.searcher.search_in_doc(doc_num, query)
            if not excerpts:
                return f"No matches for '{query}' in Doc {doc_num}"
            output = [f"## Excerpts from Doc {doc_num} matching '{query}':\n"]
            for i, exc in enumerate(excerpts[:3]):
                output.append(f"[{i+1}] {exc[:400]}...")
                output.append("")
            return "\n".join(output)
        
        elif tool_name == "write_section":
            section_title = params.get("section_title", "")
            content = params.get("content", "")
            self.paper_content[section_title] = content
            return f"✓ Section '{section_title}' written ({len(content)} chars)"
        
        elif tool_name == "done":
            summary = params.get("summary", "Task complete")
            return f"DONE: {summary}"
        
        return f"Unknown tool: {tool_name}"
    
    # =========================================================================
    # RESEARCH PHASE
    # =========================================================================
    
    def research_section(self, section: PaperSection, max_steps: int = 5) -> List[ResearchNote]:
        """
        Research a section by letting the model use tools.
        
        The model will search for relevant content and build research notes.
        """
        print(f"\n📚 Researching: {section.title}")
        print(f"   Topics: {', '.join(section.topics)}")
        
        notes = []
        
        system_prompt = f"""You are a research assistant gathering information for a paper section.

SECTION: {section.title}
TOPICS TO RESEARCH: {', '.join(section.topics)}

You have access to these tools:
{self._build_tool_prompt()}

Your task:
1. Search for each topic using the search tool
2. Get more details from promising documents using get_doc or search_in_doc
3. When you have enough information, use the done tool

Output your tool calls in this exact format:
TOOL: {{"tool": "tool_name", "param": "value"}}

After each tool result, decide what to do next. Focus on finding SPECIFIC facts, numbers, and findings."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"Research the topics for section '{section.title}'. Start by searching for the first topic."})
        
        for step in range(max_steps):
            response = self.generate(messages, max_tokens=400)
            print(f"   Step {step+1}: ", end="")
            
            # Parse tool call
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                tool_name, params = tool_call
                print(f"Tool: {tool_name}")
                
                # Execute tool
                result = self._execute_tool(tool_name, params)
                
                # Check if done
                if tool_name == "done":
                    print(f"   ✓ Research complete")
                    break
                
                # Extract notes from search results
                if tool_name == "search":
                    query = params.get("query", "")
                    search_results = self.searcher.search(query, max_results=3)
                    for r in search_results:
                        if r.excerpts:
                            notes.append(ResearchNote(
                                doc_num=r.doc_number,
                                doc_title=r.doc_title,
                                excerpt=r.excerpts[0][:500],
                                relevance=r.relevance_score
                            ))
                            section.sources.append(r.doc_number)
                
                # Add result to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result:\n{result[:1500]}\n\nContinue researching or use 'done' if you have enough information."})
            else:
                print("No tool call, continuing...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please use a tool to continue researching. Use the search tool or done tool."})
        
        self.research_notes[section.title] = notes
        print(f"   📝 Collected {len(notes)} research notes")
        return notes
    
    # =========================================================================
    # WRITING PHASE
    # =========================================================================
    
    def write_section(self, section: PaperSection) -> str:
        """
        Write a section using the research notes.
        """
        print(f"\n✍️  Writing: {section.title}")
        
        # Gather research notes
        notes = self.research_notes.get(section.title, [])
        
        if not notes:
            print("   ⚠️  No research notes, writing from general knowledge")
            notes_text = "No specific research notes available."
        else:
            notes_text = "\n\n".join([
                f"**Doc {n.doc_num}: {n.doc_title}** (relevance: {n.relevance:.1f})\n{n.excerpt}"
                for n in notes[:5]
            ])
        
        system_prompt = f"""You are writing a section of a research paper about TruthSpace Geometric LCM.

SECTION: {section.title}
TOPICS: {', '.join(section.topics)}

RESEARCH NOTES:
{notes_text}

INSTRUCTIONS:
1. Write detailed, technical content based on the research notes
2. Include SPECIFIC numbers, formulas, and findings from the notes
3. Use proper academic writing style
4. Reference the document numbers when citing findings (e.g., "As shown in Doc 191...")
5. Write 3-5 paragraphs

Do NOT be vague or generic. Use the actual content from the research notes."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write the '{section.title}' section now. Be specific and technical."}
        ]
        
        content = self.generate(messages, max_tokens=1200)
        
        # Clean up the content
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        
        section.content = content
        section.status = "complete"
        self.paper_content[section.title] = content
        
        print(f"   ✓ Written ({len(content)} chars)")
        return content
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def create_outline(self) -> List[PaperSection]:
        """Create the paper outline with sections and topics."""
        self.sections = [
            PaperSection(
                title="Abstract",
                topics=["TruthSpace hypothesis", "key findings", "φ-geometry"]
            ),
            PaperSection(
                title="1. Introduction",
                topics=["LLMs as transcoders", "geometric hypothesis", "structure is information"]
            ),
            PaperSection(
                title="2. The φ-Coordinate System",
                topics=["golden ratio properties", "φ-lattice", "φ-basis transformation", "self-similarity"]
            ),
            PaperSection(
                title="3. Key Discoveries",
                topics=["φ-computer proof", "transformer disentanglement", "boom attention", "attractor dynamics"]
            ),
            PaperSection(
                title="4. Geometric Principles",
                topics=["ENCODE = DECODE", "holographic encoding", "critical line"]
            ),
            PaperSection(
                title="5. Results",
                topics=["100% accuracy", "speedup measurements", "compression ratios"]
            ),
            PaperSection(
                title="6. Conclusion",
                topics=["validation of hypothesis", "implications", "future work"]
            ),
        ]
        return self.sections
    
    def compile_paper(self) -> str:
        """Compile all sections into a complete paper."""
        lines = [
            "# TruthSpace Geometric LCM: A φ-Based Coordinate System for Neural Computation",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]
        
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            
            if section.content:
                lines.append(section.content)
            else:
                lines.append("*[Section not yet written]*")
            
            if section.sources:
                unique_sources = sorted(set(section.sources))
                lines.append("")
                lines.append(f"*Sources: Docs {', '.join(map(str, unique_sources))}*")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def run(self, max_research_steps: int = 4):
        """Run the full paper writing pipeline."""
        print("=" * 60)
        print("Paper Writer v2 - Research → Write Pipeline")
        print("=" * 60)
        
        # Create outline
        print("\n📋 Creating outline...")
        self.create_outline()
        print(f"   {len(self.sections)} sections planned")
        
        # Research and write each section
        for i, section in enumerate(self.sections):
            print(f"\n{'='*60}")
            print(f"SECTION {i+1}/{len(self.sections)}: {section.title}")
            print("=" * 60)
            
            # Research phase
            self.research_section(section, max_steps=max_research_steps)
            
            # Write phase
            self.write_section(section)
            
            # Save progress
            paper = self.compile_paper()
            progress_path = OUTPUT_DIR / f"paper_v2_progress.md"
            progress_path.write_text(paper, encoding='utf-8')
        
        # Final compilation
        print("\n" + "=" * 60)
        print("📄 Compiling final paper...")
        print("=" * 60)
        
        final_paper = self.compile_paper()
        final_path = OUTPUT_DIR / "paper_v2_final.md"
        final_path.write_text(final_paper, encoding='utf-8')
        
        print(f"\n✓ Paper saved to: {final_path}")
        print(f"  Total length: {len(final_paper)} chars")
        
        return final_paper


def main():
    import sys
    
    max_steps = 4
    if len(sys.argv) > 1:
        max_steps = int(sys.argv[1])
    
    writer = PaperWriterV2()
    writer.run(max_research_steps=max_steps)


if __name__ == "__main__":
    main()
