#!/usr/bin/env python3
"""
Document Organizer Agent

Reads ~229 design consideration documents and organizes them into:
1. Compact summaries grouped by theme
2. A master index for quick navigation
3. Key insights extracted across all documents

This tests the agent's ability to:
- Process many files
- Maintain context/memory across documents
- Identify themes and connections
- Produce organized output

Safety: Works only on the copied files in design_docs_workspace/
"""

import torch
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Directories
WORKSPACE = Path(__file__).parent / "design_docs_workspace"
OUTPUT_DIR = Path(__file__).parent / "organized_docs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class DocSummary:
    """Summary of a single document."""
    filename: str
    doc_number: int
    title: str
    one_line: str  # One-line summary
    key_concepts: List[str]
    themes: List[str]  # e.g., "geometry", "attention", "φ-space"
    connections: List[int]  # Doc numbers this relates to
    importance: str  # "foundational", "experimental", "insight", "implementation"


@dataclass 
class ThemeCluster:
    """A cluster of related documents."""
    theme: str
    description: str
    doc_numbers: List[int]
    key_insights: List[str]


class DocOrganizer:
    """
    Agent that reads and organizes design documents.
    
    Strategy:
    1. First pass: Read each doc and create a brief summary
    2. Cluster: Group docs by theme
    3. Synthesize: Create organized output files
    """
    
    THEMES = [
        "φ-geometry",      # Golden ratio, φ-space, geometric structure
        "attention",       # Attention mechanisms, boom positions
        "encoding",        # Token encoding, embeddings
        "navigation",      # Traversal, pathfinding in semantic space
        "self-similarity", # Fractal structure, scale invariance
        "zeta",            # Riemann zeta, zeros, critical line
        "architecture",    # Model architecture, layers
        "experiments",     # Experimental results, validations
        "theory",          # Theoretical foundations
        "implementation",  # Code, practical details
    ]
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🚀 Loading Document Organizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.summaries: Dict[int, DocSummary] = {}
        self.clusters: Dict[str, ThemeCluster] = {}
        print("✓ Model loaded!\n")
    
    def generate(self, messages: List[Dict], max_tokens: int = 300) -> str:
        """Generate a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def get_doc_files(self) -> List[Path]:
        """Get all markdown files sorted by number."""
        files = list(WORKSPACE.glob("*.md"))
        
        def get_num(f):
            match = re.match(r'(\d+)', f.name)
            return int(match.group(1)) if match else 999
        
        return sorted(files, key=get_num)
    
    def extract_doc_number(self, filename: str) -> int:
        """Extract document number from filename."""
        match = re.match(r'(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    def summarize_document(self, filepath: Path) -> Optional[DocSummary]:
        """Create a summary of a single document."""
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  Error reading {filepath.name}: {e}")
            return None
        
        # Truncate if too long (keep first ~2000 chars for context)
        if len(content) > 3000:
            content = content[:2500] + "\n...[truncated]..."
        
        doc_num = self.extract_doc_number(filepath.name)
        
        messages = [
            {"role": "system", "content": f"""You are analyzing design documents for a geometric AI project called TruthSpace.

Extract the following in JSON format:
{{
    "title": "Brief descriptive title",
    "one_line": "One sentence summary of the key idea",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "themes": ["theme1", "theme2"],
    "importance": "foundational|experimental|insight|implementation"
}}

Available themes: {', '.join(self.THEMES)}

Be concise. Focus on the CORE idea."""},
            
            {"role": "user", "content": f"""Document: {filepath.name}

Content:
{content}

Extract the summary as JSON."""}
        ]
        
        response = self.generate(messages, max_tokens=250)
        
        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return DocSummary(
                    filename=filepath.name,
                    doc_number=doc_num,
                    title=data.get("title", filepath.stem),
                    one_line=data.get("one_line", ""),
                    key_concepts=data.get("key_concepts", [])[:5],
                    themes=data.get("themes", [])[:3],
                    connections=[],  # Will be filled in later
                    importance=data.get("importance", "insight")
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract what we can
        return DocSummary(
            filename=filepath.name,
            doc_number=doc_num,
            title=filepath.stem.replace('_', ' '),
            one_line=response[:100] if response else "",
            key_concepts=[],
            themes=[],
            connections=[],
            importance="insight"
        )
    
    def process_all_documents(self, limit: Optional[int] = None):
        """Process all documents and create summaries."""
        files = self.get_doc_files()
        if limit:
            files = files[:limit]
        
        total = len(files)
        print(f"📚 Processing {total} documents...\n")
        
        for i, filepath in enumerate(files):
            doc_num = self.extract_doc_number(filepath.name)
            print(f"[{i+1}/{total}] {filepath.name[:50]}...", end=" ")
            
            summary = self.summarize_document(filepath)
            if summary:
                self.summaries[doc_num] = summary
                print(f"✓ {summary.title[:40]}")
            else:
                print("✗ Failed")
            
            # Save progress every 20 docs
            if (i + 1) % 20 == 0:
                self.save_progress()
                print(f"  💾 Progress saved ({i+1}/{total})")
        
        print(f"\n✓ Processed {len(self.summaries)} documents")
    
    def cluster_by_theme(self):
        """Group documents by theme."""
        print("\n🔗 Clustering documents by theme...")
        
        for theme in self.THEMES:
            docs_in_theme = []
            for doc_num, summary in self.summaries.items():
                # Check if theme matches
                if any(theme.lower() in t.lower() for t in summary.themes):
                    docs_in_theme.append(doc_num)
                # Also check key concepts
                elif any(theme.lower() in c.lower() for c in summary.key_concepts):
                    docs_in_theme.append(doc_num)
            
            if docs_in_theme:
                self.clusters[theme] = ThemeCluster(
                    theme=theme,
                    description=f"Documents related to {theme}",
                    doc_numbers=sorted(docs_in_theme),
                    key_insights=[]
                )
                print(f"  {theme}: {len(docs_in_theme)} docs")
    
    def generate_master_index(self) -> str:
        """Generate a master index of all documents."""
        lines = [
            "# TruthSpace Design Documents - Master Index",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Documents: {len(self.summaries)}",
            "",
            "## By Theme",
            ""
        ]
        
        for theme, cluster in sorted(self.clusters.items()):
            lines.append(f"### {theme.title()} ({len(cluster.doc_numbers)} docs)")
            for doc_num in cluster.doc_numbers:  # Show ALL docs
                if doc_num in self.summaries:
                    s = self.summaries[doc_num]
                    lines.append(f"- **{doc_num:03d}**: {s.title} - {s.one_line}")
            lines.append("")
        
        lines.extend([
            "## All Documents (by number)",
            ""
        ])
        
        for doc_num in sorted(self.summaries.keys()):
            s = self.summaries[doc_num]
            themes_str = ", ".join(s.themes[:2]) if s.themes else "uncategorized"
            lines.append(f"- **{doc_num:03d}** [{themes_str}]: {s.title}")
        
        return "\n".join(lines)
    
    def generate_theme_summary(self, theme: str) -> str:
        """Generate a detailed summary for a theme."""
        if theme not in self.clusters:
            return f"# {theme}\n\nNo documents found for this theme."
        
        cluster = self.clusters[theme]
        
        lines = [
            f"# {theme.title()} - Theme Summary",
            "",
            f"**{len(cluster.doc_numbers)} documents** in this theme.",
            "",
            "## Key Documents",
            ""
        ]
        
        for doc_num in cluster.doc_numbers:
            if doc_num in self.summaries:
                s = self.summaries[doc_num]
                lines.extend([
                    f"### {doc_num:03d}: {s.title}",
                    f"**Summary**: {s.one_line}",
                    f"**Concepts**: {', '.join(s.key_concepts)}",
                    f"**Importance**: {s.importance}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def save_progress(self):
        """Save current progress to files."""
        # Save summaries as JSON
        summaries_data = {
            str(k): asdict(v) for k, v in self.summaries.items()
        }
        with open(OUTPUT_DIR / "summaries.json", 'w') as f:
            json.dump(summaries_data, f, indent=2)
    
    def save_outputs(self):
        """Save all organized outputs."""
        print("\n💾 Saving organized outputs...")
        
        # Master index
        index = self.generate_master_index()
        (OUTPUT_DIR / "00_MASTER_INDEX.md").write_text(index)
        print(f"  ✓ Master index saved")
        
        # Theme summaries
        for theme in self.clusters:
            summary = self.generate_theme_summary(theme)
            safe_name = theme.replace("-", "_").replace(" ", "_")
            (OUTPUT_DIR / f"theme_{safe_name}.md").write_text(summary)
        print(f"  ✓ {len(self.clusters)} theme summaries saved")
        
        # Full summaries JSON
        self.save_progress()
        print(f"  ✓ Summaries JSON saved")
        
        # Quick reference (one-liners only)
        quick_ref = ["# Quick Reference - One-Line Summaries", ""]
        for doc_num in sorted(self.summaries.keys()):
            s = self.summaries[doc_num]
            quick_ref.append(f"**{doc_num:03d}**: {s.one_line}")
        (OUTPUT_DIR / "01_QUICK_REFERENCE.md").write_text("\n".join(quick_ref))
        print(f"  ✓ Quick reference saved")
        
        print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    
    def run(self, limit: Optional[int] = None):
        """Run the full organization pipeline."""
        print("=" * 60)
        print("Document Organizer - Processing Design Considerations")
        print("=" * 60)
        print(f"Source: {WORKSPACE}")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 60)
        
        # Process documents
        self.process_all_documents(limit=limit)
        
        # Cluster by theme
        self.cluster_by_theme()
        
        # Save outputs
        self.save_outputs()
        
        print("\n" + "=" * 60)
        print("✓ COMPLETE")
        print("=" * 60)
        
        return self.summaries, self.clusters


def main():
    import sys
    
    # Optional limit for testing
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        print(f"Running with limit: {limit} documents")
    
    organizer = DocOrganizer()
    organizer.run(limit=limit)


if __name__ == "__main__":
    main()
