#!/usr/bin/env python3
"""
Research Tools for Paper Writing

Integrated toolset that an AI agent can use to:
1. Search for concepts in documentation
2. Extract actual content from source documents
3. Edit papers section by section
4. Track changes with diffs

These tools are designed to be called by the paper_writer agent.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import our tools
from concept_search import ConceptSearcher, SearchResult
from structured_editor import StructuredEditor, DocumentState, Section


@dataclass
class ResearchContext:
    """Context accumulated during research."""
    query: str
    sources: List[SearchResult]
    extracted_content: Dict[int, str]  # doc_num -> content
    notes: List[str]


class ResearchTools:
    """
    Integrated research tools for paper writing.
    
    Provides a simple interface for:
    - Searching concepts
    - Extracting content
    - Editing documents
    - Tracking changes
    """
    
    def __init__(self):
        self.searcher = ConceptSearcher()
        self.editor = StructuredEditor()
        self.research_contexts: List[ResearchContext] = []
        self.current_doc: Optional[DocumentState] = None
    
    # =========================================================================
    # SEARCH TOOLS
    # =========================================================================
    
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Search for a concept and return formatted results.
        
        Args:
            query: Natural language query (e.g., "φ-computer proof")
            max_results: Maximum results to return
            
        Returns:
            Formatted string with search results and excerpts
        """
        results = self.searcher.search(query, max_results)
        
        if not results:
            return f"No results found for: {query}"
        
        # Store context
        context = ResearchContext(
            query=query,
            sources=results,
            extracted_content={},
            notes=[]
        )
        self.research_contexts.append(context)
        
        # Format output
        lines = [f"## Search Results for: {query}", ""]
        
        for i, result in enumerate(results):
            lines.append(f"### [{i+1}] Doc {result.doc_number}: {result.doc_title}")
            lines.append(f"**Relevance**: {result.relevance_score:.1f}")
            lines.append(f"**Matched**: {', '.join(result.matched_concepts[:5])}")
            lines.append("")
            
            if result.excerpts:
                lines.append("**Excerpts**:")
                for j, excerpt in enumerate(result.excerpts[:2]):
                    lines.append(f"> {excerpt[:300]}...")
                    lines.append("")
            
            lines.append("---")
        
        return "\n".join(lines)
    
    def get_full_content(self, doc_num: int) -> str:
        """
        Get the full content of a specific document.
        
        Args:
            doc_num: Document number (e.g., 191 for Doc 191)
            
        Returns:
            Full document content
        """
        content = self.searcher.get_full_doc(doc_num)
        
        if not content:
            return f"Document {doc_num} not found"
        
        # Cache in most recent context
        if self.research_contexts:
            self.research_contexts[-1].extracted_content[doc_num] = content
        
        return content
    
    def search_in_doc(self, doc_num: int, query: str) -> str:
        """
        Search within a specific document for relevant sections.
        
        Args:
            doc_num: Document number
            query: What to search for within the document
            
        Returns:
            Relevant excerpts from the document
        """
        excerpts = self.searcher.search_in_doc(doc_num, query)
        
        if not excerpts:
            return f"No matches for '{query}' in Doc {doc_num}"
        
        lines = [f"## Excerpts from Doc {doc_num} matching '{query}'", ""]
        for i, excerpt in enumerate(excerpts):
            lines.append(f"**[{i+1}]**")
            lines.append(f"> {excerpt}")
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DOCUMENT EDITING TOOLS
    # =========================================================================
    
    def load_document(self, filepath: str) -> str:
        """
        Load a document for editing.
        
        Args:
            filepath: Path to the document
            
        Returns:
            Document structure overview
        """
        try:
            self.current_doc = self.editor.load(filepath)
            structure = self.editor.show_structure(self.current_doc)
            return f"## Document Loaded: {filepath}\n\n{structure}"
        except FileNotFoundError:
            return f"File not found: {filepath}"
    
    def get_section(self, section_id: str) -> str:
        """
        Get the content of a specific section.
        
        Args:
            section_id: Section ID (e.g., "section_5")
            
        Returns:
            Section content
        """
        if not self.current_doc:
            return "No document loaded. Use load_document() first."
        
        content = self.editor.get_section_content(self.current_doc, section_id)
        if not content:
            return f"Section {section_id} not found"
        
        return content
    
    def edit_section(self, section_id: str, new_content: str, 
                     operation: str = 'replace') -> str:
        """
        Edit a section of the current document.
        
        Args:
            section_id: Section ID to edit
            new_content: New content
            operation: 'replace', 'append', or 'prepend'
            
        Returns:
            Confirmation and diff
        """
        if not self.current_doc:
            return "No document loaded. Use load_document() first."
        
        success = self.editor.edit_section(
            self.current_doc, section_id, new_content, operation
        )
        
        if not success:
            return f"Failed to edit section {section_id}"
        
        # Show diff for this section
        diff = self.editor.show_section_diff(self.current_doc, section_id)
        
        return f"## Section {section_id} edited ({operation})\n\n```diff\n{diff}\n```"
    
    def show_changes(self) -> str:
        """Show all changes made to the current document."""
        if not self.current_doc:
            return "No document loaded."
        
        summary = self.editor.get_edit_summary(self.current_doc)
        diff = self.editor.show_diff(self.current_doc)
        
        return f"## Edit Summary\n\n{summary}\n\n## Full Diff\n\n```diff\n{diff}\n```"
    
    def save_document(self, backup: bool = True) -> str:
        """Save the current document."""
        if not self.current_doc:
            return "No document loaded."
        
        success = self.editor.save(self.current_doc, backup=backup)
        
        if success:
            return f"Document saved: {self.current_doc.filepath}"
        else:
            return "Failed to save document"
    
    def revert_changes(self) -> str:
        """Revert all changes to the current document."""
        if not self.current_doc:
            return "No document loaded."
        
        self.editor.revert(self.current_doc)
        return "All changes reverted"
    
    # =========================================================================
    # RESEARCH WORKFLOW TOOLS
    # =========================================================================
    
    def research_topic(self, topic: str) -> str:
        """
        Research a topic by searching and extracting key content.
        
        Args:
            topic: Topic to research
            
        Returns:
            Compiled research with sources and key findings
        """
        # Search for the topic
        results = self.searcher.search(topic, max_results=5)
        
        if not results:
            return f"No information found on: {topic}"
        
        lines = [f"# Research: {topic}", ""]
        
        # Get excerpts from top results
        for result in results[:3]:
            lines.append(f"## Doc {result.doc_number}: {result.doc_title}")
            lines.append(f"*Relevance: {result.relevance_score:.1f}*")
            lines.append("")
            
            # Get more detailed excerpts
            detailed = self.searcher.search_in_doc(result.doc_number, topic)
            if detailed:
                for excerpt in detailed[:2]:
                    lines.append(f"> {excerpt}")
                    lines.append("")
            elif result.excerpts:
                for excerpt in result.excerpts[:2]:
                    lines.append(f"> {excerpt}")
                    lines.append("")
            
            lines.append("---")
        
        return "\n".join(lines)
    
    def compile_sources(self, topics: List[str]) -> str:
        """
        Compile sources for multiple topics.
        
        Args:
            topics: List of topics to research
            
        Returns:
            Compiled bibliography with key findings
        """
        lines = ["# Compiled Sources", ""]
        
        all_docs = set()
        
        for topic in topics:
            results = self.searcher.search(topic, max_results=3)
            for result in results:
                all_docs.add((result.doc_number, result.doc_title))
        
        lines.append(f"**{len(all_docs)} unique documents found**")
        lines.append("")
        
        for doc_num, title in sorted(all_docs):
            lines.append(f"- **Doc {doc_num}**: {title}")
        
        return "\n".join(lines)


def demo():
    """Demo the research tools."""
    tools = ResearchTools()
    
    # Demo search
    print("=" * 60)
    print("DEMO: Searching for 'φ-computer proof'")
    print("=" * 60)
    print(tools.search("φ-computer proof", max_results=2))
    
    # Demo research topic
    print("\n" + "=" * 60)
    print("DEMO: Researching 'transformer disentanglement'")
    print("=" * 60)
    print(tools.research_topic("transformer disentanglement"))
    
    # Demo document editing
    paper_path = "/home/thorin/truthspace-lcm/phi_chat/paper_output/paper_manual_enhanced.md"
    if Path(paper_path).exists():
        print("\n" + "=" * 60)
        print("DEMO: Loading document for editing")
        print("=" * 60)
        print(tools.load_document(paper_path))


if __name__ == "__main__":
    demo()
