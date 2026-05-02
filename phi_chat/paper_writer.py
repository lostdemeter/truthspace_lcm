#!/usr/bin/env python3
"""
Research Paper Writer Agent

An iterative agent that writes a research paper about the TruthSpace Geometric LCM system.

Pipeline:
1. Read source code to understand the system
2. Read organized documentation
3. Create an outline
4. Generate skeleton paper
5. Review and identify gaps
6. Iterate until complete

The agent writes about:
- The φ (phi) coordinate system
- Unique geometric principles
- Application to the model
"""

import torch
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Directories
PROJECT_ROOT = Path("/home/thorin/truthspace-lcm")
ORGANIZED_DOCS = PROJECT_ROOT / "phi_chat" / "organized_docs"
SOURCE_DIRS = [
    PROJECT_ROOT / "truthspace_lcm",
    PROJECT_ROOT / "experiments",
]
OUTPUT_DIR = PROJECT_ROOT / "phi_chat" / "paper_output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class PaperSection:
    """A section of the research paper."""
    title: str
    content: str
    subsections: List['PaperSection'] = field(default_factory=list)
    status: str = "outline"  # outline, skeleton, draft, complete
    sources: List[str] = field(default_factory=list)  # Files that informed this section


@dataclass
class ResearchPaper:
    """The full research paper."""
    title: str
    abstract: str = ""
    sections: List[PaperSection] = field(default_factory=list)
    iteration: int = 0
    
    def to_markdown(self) -> str:
        """Convert paper to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            "## Abstract",
            "",
            self.abstract if self.abstract else "*To be written*",
            "",
        ]
        
        for section in self.sections:
            lines.extend(self._section_to_markdown(section, level=2))
        
        # Add metadata
        lines.extend([
            "",
            "---",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"*Iteration: {self.iteration}*",
        ])
        
        return "\n".join(lines)
    
    def _section_to_markdown(self, section: PaperSection, level: int) -> List[str]:
        """Convert a section to markdown lines."""
        prefix = "#" * level
        lines = [
            f"{prefix} {section.title}",
            "",
        ]
        
        if section.content:
            lines.append(section.content)
            lines.append("")
        else:
            lines.append(f"*[{section.status}]*")
            lines.append("")
        
        for subsection in section.subsections:
            lines.extend(self._section_to_markdown(subsection, level + 1))
        
        return lines


class PaperWriter:
    """
    Agent that iteratively writes a research paper about TruthSpace.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🚀 Loading Paper Writer Agent...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Knowledge base
        self.source_code: Dict[str, str] = {}
        self.documentation: Dict[str, str] = {}
        self.summaries: Dict[str, dict] = {}
        
        # Paper state
        self.paper = ResearchPaper(
            title="TruthSpace Geometric LCM: A φ-Based Coordinate System for Neural Computation"
        )
        
        print("✓ Model loaded!\n")
    
    def generate(self, messages: List[Dict], max_tokens: int = 1000) -> str:
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
    
    # =========================================================================
    # PHASE 1: Read and understand the codebase
    # =========================================================================
    
    def read_source_code(self):
        """Read key source files to understand the system."""
        print("📖 Reading source code...")
        
        # Key files to read
        key_files = [
            PROJECT_ROOT / "truthspace_lcm" / "core" / "phi_geometry.py",
            PROJECT_ROOT / "truthspace_lcm" / "core" / "transforms.py",
            PROJECT_ROOT / "truthspace_lcm" / "core" / "encoder.py",
            PROJECT_ROOT / "experiments" / "phi_discovery_engine.py",
            PROJECT_ROOT / "experiments" / "phi_memory.py",
            PROJECT_ROOT / "experiments" / "phi_self_aware_agent.py",
            PROJECT_ROOT / "experiments" / "phi_tool_agent.py",
        ]
        
        # Also find Python files in experiments
        experiment_files = list((PROJECT_ROOT / "experiments").glob("*.py"))[:20]
        
        all_files = key_files + experiment_files
        
        for filepath in all_files:
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    # Truncate very long files
                    if len(content) > 5000:
                        content = content[:4500] + "\n...[truncated]..."
                    self.source_code[str(filepath.relative_to(PROJECT_ROOT))] = content
                    print(f"  ✓ {filepath.name}")
                except Exception as e:
                    print(f"  ✗ {filepath.name}: {e}")
        
        print(f"  Read {len(self.source_code)} source files\n")
    
    def read_documentation(self):
        """Read the organized documentation."""
        print("📚 Reading organized documentation...")
        
        # Read summaries JSON
        summaries_path = ORGANIZED_DOCS / "summaries.json"
        if summaries_path.exists():
            with open(summaries_path) as f:
                self.summaries = json.load(f)
            print(f"  ✓ Loaded {len(self.summaries)} document summaries")
        
        # Read theme files
        for theme_file in ORGANIZED_DOCS.glob("theme_*.md"):
            content = theme_file.read_text(encoding='utf-8')
            self.documentation[theme_file.stem] = content
            print(f"  ✓ {theme_file.name}")
        
        # Read master index
        index_path = ORGANIZED_DOCS / "00_MASTER_INDEX.md"
        if index_path.exists():
            self.documentation["master_index"] = index_path.read_text(encoding='utf-8')
            print(f"  ✓ Master index")
        
        # Read quick reference
        quick_ref = ORGANIZED_DOCS / "01_QUICK_REFERENCE.md"
        if quick_ref.exists():
            self.documentation["quick_reference"] = quick_ref.read_text(encoding='utf-8')
            print(f"  ✓ Quick reference")
        
        print(f"  Read {len(self.documentation)} documentation files\n")
    
    # =========================================================================
    # PHASE 2: Create outline
    # =========================================================================
    
    def create_outline(self):
        """Create the paper outline based on source code and docs."""
        print("📝 Creating paper outline...")
        
        # Gather context
        source_summary = self._summarize_sources()
        doc_summary = self._summarize_docs()
        
        messages = [
            {"role": "system", "content": """You are a research paper writer for a geometric AI project called TruthSpace.

Create a detailed outline for a research paper about:
1. The φ (phi/golden ratio) coordinate system used in the project
2. The unique geometric principles (self-similarity, attractor dynamics, etc.)
3. How these principles are applied to create a geometric language model

The outline should have:
- Clear section titles
- Brief description of what each section covers
- Logical flow from introduction to conclusion

Output the outline in this format:
SECTION: [Title]
DESCRIPTION: [What this section covers]
SUBSECTIONS: [Comma-separated list of subsections if any]

Be comprehensive - this paper should cover everything in the codebase and documentation."""},
            
            {"role": "user", "content": f"""Based on the following source code and documentation, create a comprehensive paper outline.

SOURCE CODE SUMMARY:
{source_summary}

DOCUMENTATION SUMMARY:
{doc_summary}

Create the outline now."""}
        ]
        
        response = self.generate(messages, max_tokens=1500)
        
        # Parse outline
        self._parse_outline(response)
        
        print(f"  Created outline with {len(self.paper.sections)} sections\n")
        self._save_paper("outline")
    
    def _summarize_sources(self) -> str:
        """Create a summary of source code for context."""
        lines = ["Key source files and their purposes:"]
        
        for filepath, content in list(self.source_code.items())[:10]:
            # Extract docstring or first comment
            docstring = ""
            if '"""' in content:
                match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if match:
                    docstring = match.group(1)[:200]
            
            lines.append(f"\n{filepath}:")
            if docstring:
                lines.append(f"  {docstring.strip()}")
            
            # Extract class/function names
            classes = re.findall(r'class (\w+)', content)
            functions = re.findall(r'def (\w+)', content)
            if classes:
                lines.append(f"  Classes: {', '.join(classes[:5])}")
            if functions:
                lines.append(f"  Functions: {', '.join(functions[:10])}")
        
        return "\n".join(lines)
    
    def _summarize_docs(self) -> str:
        """Create a summary of documentation for context."""
        lines = ["Key themes and concepts from documentation:"]
        
        # Include actual content from quick reference
        if "quick_reference" in self.documentation:
            quick_lines = self.documentation["quick_reference"].split("\n")[:100]
            lines.extend(quick_lines)
        
        # Add key discoveries from summaries
        lines.append("\n\nKEY DISCOVERIES FROM DESIGN DOCS:")
        key_docs = ["072", "127", "140", "160", "177", "191", "192"]  # Important docs
        for doc_num in key_docs:
            if doc_num in self.summaries:
                s = self.summaries[doc_num]
                lines.append(f"\nDoc {doc_num}: {s.get('title', '')}")
                lines.append(f"  {s.get('one_line', '')}")
                lines.append(f"  Concepts: {', '.join(s.get('key_concepts', []))}")
        
        return "\n".join(lines)
    
    def _parse_outline(self, response: str):
        """Parse the outline response into paper sections."""
        self.paper.sections = []
        
        current_section = None
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.startswith("SECTION:"):
                if current_section:
                    self.paper.sections.append(current_section)
                title = line.replace("SECTION:", "").strip()
                current_section = PaperSection(title=title, content="", status="outline")
            
            elif line.startswith("DESCRIPTION:") and current_section:
                current_section.content = line.replace("DESCRIPTION:", "").strip()
            
            elif line.startswith("SUBSECTIONS:") and current_section:
                subsections = line.replace("SUBSECTIONS:", "").strip()
                for sub in subsections.split(","):
                    sub = sub.strip()
                    if sub:
                        current_section.subsections.append(
                            PaperSection(title=sub, content="", status="outline")
                        )
        
        if current_section:
            self.paper.sections.append(current_section)
        
        # Always use comprehensive sections based on our actual content
        if len(self.paper.sections) < 5:
            self.paper.sections = [
                PaperSection(title="Introduction", content="The hypothesis that LLMs are hyperdimensional transcoders", status="outline"),
                PaperSection(title="The φ-Coordinate System", content="Golden ratio as universal adapter", status="outline", 
                    subsections=[
                        PaperSection(title="φ-Lattice Coordinates", content="", status="outline"),
                        PaperSection(title="Self-Similarity and Scale Invariance", content="", status="outline"),
                        PaperSection(title="φ-Basis Transformation", content="", status="outline"),
                    ]),
                PaperSection(title="Geometric Principles", content="Structure IS information", status="outline",
                    subsections=[
                        PaperSection(title="Attractor-Repeller Dynamics", content="", status="outline"),
                        PaperSection(title="Holographic Encoding", content="", status="outline"),
                        PaperSection(title="ENCODE = DECODE Principle", content="", status="outline"),
                    ]),
                PaperSection(title="Key Discoveries", content="Empirical findings from experiments", status="outline",
                    subsections=[
                        PaperSection(title="The φ-Computer Proof", content="", status="outline"),
                        PaperSection(title="Transformer Disentanglement", content="", status="outline"),
                        PaperSection(title="Boom-Newton Attention", content="", status="outline"),
                    ]),
                PaperSection(title="Implementation", content="Code architecture and experiments", status="outline"),
                PaperSection(title="Results and Validation", content="Experimental evidence", status="outline"),
                PaperSection(title="Conclusion", content="", status="outline"),
            ]
    
    # =========================================================================
    # PHASE 3: Generate skeleton
    # =========================================================================
    
    def generate_skeleton(self):
        """Generate skeleton content for each section."""
        print("🦴 Generating paper skeleton...")
        
        for i, section in enumerate(self.paper.sections):
            print(f"  [{i+1}/{len(self.paper.sections)}] {section.title}...", end=" ")
            
            # Find relevant sources for this section
            relevant_sources = self._find_relevant_sources(section.title)
            section.sources = relevant_sources
            
            # Generate skeleton content
            content = self._generate_section_skeleton(section, relevant_sources)
            section.content = content
            section.status = "skeleton"
            
            # Generate subsection skeletons
            for subsection in section.subsections:
                sub_sources = self._find_relevant_sources(subsection.title)
                subsection.sources = sub_sources
                subsection.content = self._generate_section_skeleton(subsection, sub_sources)
                subsection.status = "skeleton"
            
            print("✓")
        
        self.paper.iteration += 1
        self._save_paper("skeleton")
        print(f"  Skeleton complete (iteration {self.paper.iteration})\n")
    
    def _find_relevant_sources(self, topic: str) -> List[str]:
        """Find source files and docs relevant to a topic."""
        relevant = []
        topic_lower = topic.lower()
        
        # Search in summaries
        for doc_num, summary in self.summaries.items():
            title = summary.get("title", "").lower()
            one_line = summary.get("one_line", "").lower()
            concepts = " ".join(summary.get("key_concepts", [])).lower()
            
            if any(word in title or word in one_line or word in concepts 
                   for word in topic_lower.split()):
                relevant.append(f"doc_{doc_num}")
        
        # Search in source code
        for filepath in self.source_code:
            if any(word in filepath.lower() for word in topic_lower.split()):
                relevant.append(filepath)
        
        return relevant[:5]  # Limit to top 5
    
    def _generate_section_skeleton(self, section: PaperSection, sources: List[str]) -> str:
        """Generate skeleton content for a section."""
        # Gather MORE relevant content - be thorough
        source_content = []
        
        # Get content from summaries
        for source in sources[:5]:
            if source.startswith("doc_"):
                doc_num = source.replace("doc_", "")
                if doc_num in self.summaries:
                    s = self.summaries[doc_num]
                    source_content.append(f"\nDoc {doc_num}: {s.get('title', '')}")
                    source_content.append(f"Summary: {s.get('one_line', '')}")
                    source_content.append(f"Key concepts: {', '.join(s.get('key_concepts', []))}")
            elif source in self.source_code:
                code = self.source_code[source][:800]
                source_content.append(f"\nCode from {source}:\n{code}")
        
        # Add section-specific context from our key discoveries
        section_lower = section.title.lower()
        if "φ" in section_lower or "phi" in section_lower or "coordinate" in section_lower:
            source_content.append("\nKEY FACT: φ (golden ratio, 1.618...) has the property φ = 1 + 1/φ, making it self-similar.")
            source_content.append("KEY FACT: Model weights naturally occupy positions on the φ-lattice.")
            source_content.append("KEY FACT: φ-basis transformation: φ_dim[i] = original_dim[sorted_by_corr[i]] × φ^(-i/10) × sign(corr[i])")
        if "attractor" in section_lower or "dynamics" in section_lower:
            source_content.append("\nKEY FACT: Self-similar concepts ATTRACT (converge), dissimilar concepts REPEL (diverge).")
            source_content.append("KEY FACT: Vocabulary emerges from attractor/repeller dynamics based on usage patterns.")
        if "boom" in section_lower or "attention" in section_lower:
            source_content.append("\nKEY FACT: Boom-Newton attention achieves 2.5-2.7x speedup with 100% token accuracy.")
            source_content.append("KEY FACT: 89.5% of attention mass captured with only 37% of positions (64 booms out of 172).")
        if "disentangle" in section_lower or "transformer" in section_lower:
            source_content.append("\nKEY FACT: Transformer's transformation can be approximated by 37-dimensional linear mapping.")
            source_content.append("KEY FACT: Scaffolding tokens (the, is, a) = 100% generalizable. Content tokens (Paris, Einstein) = 0% generalizable.")
        if "computer" in section_lower or "proof" in section_lower:
            source_content.append("\nKEY FACT: Proved transformers are φ-computers with exact φ-operations for sigmoids, softmax, SiLU.")
            source_content.append("KEY FACT: Achieved 100% token accuracy with φ-computer proof.")
        
        messages = [
            {"role": "system", "content": """You are writing a research paper about TruthSpace Geometric LCM.

Write detailed content for the given section. You MUST:
- Include SPECIFIC technical details from the sources provided
- Use actual numbers, formulas, and measurements
- Reference the φ (golden ratio = 1.618) coordinate system
- Explain HOW the geometric principles work, not just THAT they work
- Include code snippets or mathematical formulations where relevant

Be SPECIFIC and TECHNICAL. Do not be vague or generic. Use the KEY FACTS provided."""},
            
            {"role": "user", "content": f"""Section: {section.title}
Section description: {section.content if section.content else 'Not specified'}

Relevant sources and KEY FACTS:
{chr(10).join(source_content) if source_content else 'No specific sources found'}

Write detailed technical content for this section (3-5 paragraphs). Include specific numbers, formulas, and findings."""}
        ]
        
        return self.generate(messages, max_tokens=1000)
    
    # =========================================================================
    # PHASE 4: Review and iterate
    # =========================================================================
    
    def review_paper(self) -> List[str]:
        """Review the paper and identify gaps."""
        print("🔍 Reviewing paper for gaps...")
        
        current_content = self.paper.to_markdown()
        
        messages = [
            {"role": "system", "content": """You are reviewing a research paper draft about TruthSpace Geometric LCM.

Identify specific gaps, missing content, or areas that need expansion. Focus on:
1. Missing technical details about φ-geometry
2. Unexplained concepts or terminology
3. Missing connections between sections
4. Areas that need more depth or examples
5. Missing references to documented features

Output each gap as:
GAP: [Section name] - [What's missing or needs improvement]"""},
            
            {"role": "user", "content": f"""Review this paper draft and identify gaps:

{current_content[:8000]}

List the gaps that need to be addressed."""}
        ]
        
        response = self.generate(messages, max_tokens=800)
        
        # Parse gaps
        gaps = []
        for line in response.split("\n"):
            if line.strip().startswith("GAP:"):
                gaps.append(line.replace("GAP:", "").strip())
        
        print(f"  Found {len(gaps)} gaps to address\n")
        return gaps
    
    def address_gaps(self, gaps: List[str]):
        """Address identified gaps by expanding content."""
        print("🔧 Addressing gaps...")
        
        for i, gap in enumerate(gaps[:5]):  # Address top 5 gaps per iteration
            print(f"  [{i+1}/{min(len(gaps), 5)}] {gap[:50]}...", end=" ")
            
            # Find which section this gap belongs to
            section = self._find_section_for_gap(gap)
            if section:
                expanded = self._expand_section(section, gap)
                section.content = expanded
                section.status = "draft"
                print("✓")
            else:
                print("(no matching section)")
        
        self.paper.iteration += 1
        self._save_paper(f"draft_v{self.paper.iteration}")
        print(f"  Gaps addressed (iteration {self.paper.iteration})\n")
    
    def _find_section_for_gap(self, gap: str) -> Optional[PaperSection]:
        """Find the section that a gap belongs to."""
        gap_lower = gap.lower()
        
        for section in self.paper.sections:
            if section.title.lower() in gap_lower:
                return section
            for subsection in section.subsections:
                if subsection.title.lower() in gap_lower:
                    return subsection
        
        # Default to first section that seems related
        for section in self.paper.sections:
            if any(word in gap_lower for word in section.title.lower().split()):
                return section
        
        return None
    
    def _expand_section(self, section: PaperSection, gap: str) -> str:
        """Expand a section to address a gap."""
        # Get more relevant sources
        relevant_sources = self._find_relevant_sources(gap)
        
        source_content = []
        for source in relevant_sources[:3]:
            if source.startswith("doc_"):
                doc_num = source.replace("doc_", "")
                if doc_num in self.summaries:
                    s = self.summaries[doc_num]
                    source_content.append(f"Doc {doc_num}: {s.get('title', '')} - {s.get('one_line', '')}")
            elif source in self.source_code:
                code = self.source_code[source][:800]
                source_content.append(f"Code from {source}:\n{code}")
        
        messages = [
            {"role": "system", "content": """You are expanding a research paper section about TruthSpace Geometric LCM.

Expand the existing content to address the identified gap. Add:
- More technical details
- Specific examples from the code
- Mathematical formulations where appropriate
- Connections to other concepts

Maintain academic writing style. Be specific and thorough."""},
            
            {"role": "user", "content": f"""Section: {section.title}

Current content:
{section.content}

Gap to address: {gap}

Additional sources:
{chr(10).join(source_content) if source_content else 'No additional sources'}

Expand this section to address the gap (keep existing content and add to it)."""}
        ]
        
        return self.generate(messages, max_tokens=1000)
    
    # =========================================================================
    # PHASE 5: Finalize
    # =========================================================================
    
    def write_abstract(self):
        """Write the paper abstract."""
        print("📄 Writing abstract...")
        
        current_content = self.paper.to_markdown()
        
        messages = [
            {"role": "system", "content": """You are writing an abstract for a research paper about TruthSpace Geometric LCM.

Write a concise abstract (150-250 words) that:
1. States the problem/motivation
2. Describes the approach (φ-coordinate system, geometric principles)
3. Summarizes key contributions
4. Highlights main results or findings

Be specific and technical."""},
            
            {"role": "user", "content": f"""Based on this paper content, write the abstract:

{current_content[:6000]}

Write the abstract now."""}
        ]
        
        self.paper.abstract = self.generate(messages, max_tokens=400)
        print("  ✓ Abstract written\n")
    
    def finalize_paper(self):
        """Final pass to polish the paper."""
        print("✨ Finalizing paper...")
        
        # Mark all sections as complete
        for section in self.paper.sections:
            section.status = "complete"
            for subsection in section.subsections:
                subsection.status = "complete"
        
        self._save_paper("final")
        print("  ✓ Paper finalized\n")
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _save_paper(self, stage: str):
        """Save the current paper state."""
        filename = f"paper_{stage}.md"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(self.paper.to_markdown(), encoding='utf-8')
        print(f"  💾 Saved: {filepath}")
    
    def run(self, max_iterations: int = 3):
        """Run the full paper writing pipeline."""
        print("=" * 60)
        print("Research Paper Writer - TruthSpace Geometric LCM")
        print("=" * 60)
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 60 + "\n")
        
        # Phase 1: Read sources
        self.read_source_code()
        self.read_documentation()
        
        # Phase 2: Create outline
        self.create_outline()
        
        # Phase 3: Generate skeleton
        self.generate_skeleton()
        
        # Phase 4: Iterate
        for iteration in range(max_iterations):
            print(f"{'='*60}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*60}\n")
            
            gaps = self.review_paper()
            
            if not gaps:
                print("  No gaps found - paper is complete!")
                break
            
            self.address_gaps(gaps)
        
        # Phase 5: Finalize
        self.write_abstract()
        self.finalize_paper()
        
        print("=" * 60)
        print("✓ PAPER COMPLETE")
        print(f"  Final paper: {OUTPUT_DIR / 'paper_final.md'}")
        print("=" * 60)
        
        return self.paper


def main():
    import sys
    
    max_iterations = 3
    if len(sys.argv) > 1:
        max_iterations = int(sys.argv[1])
    
    writer = PaperWriter()
    writer.run(max_iterations=max_iterations)


if __name__ == "__main__":
    main()
