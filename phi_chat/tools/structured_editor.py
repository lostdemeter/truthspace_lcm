#!/usr/bin/env python3
"""
Structured File Editor

Makes it easier to work with large files by:
1. Breaking files into logical sections
2. Allowing section-based editing
3. Tracking changes with diffs
4. Supporting incremental updates

Designed for markdown files but works with any structured text.
"""

import difflib
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Section:
    """A logical section of a document."""
    id: str
    title: str
    level: int  # Heading level (1-6 for markdown)
    start_line: int
    end_line: int
    content: str
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class EditOperation:
    """A single edit operation."""
    section_id: str
    operation: str  # 'replace', 'append', 'prepend', 'delete'
    old_content: str
    new_content: str
    timestamp: str


@dataclass
class DocumentState:
    """Current state of a document with history."""
    filepath: Path
    original_content: str
    current_content: str
    sections: List[Section]
    edit_history: List[EditOperation] = field(default_factory=list)


class StructuredEditor:
    """
    Edit large files by section with diff tracking.
    
    Usage:
        editor = StructuredEditor()
        doc = editor.load("paper.md")
        
        # View structure
        editor.show_structure(doc)
        
        # Edit a section
        editor.edit_section(doc, "section_2", new_content)
        
        # See what changed
        editor.show_diff(doc)
        
        # Save
        editor.save(doc)
    """
    
    def __init__(self):
        self.documents: Dict[str, DocumentState] = {}
    
    def _parse_sections(self, content: str) -> List[Section]:
        """Parse markdown content into sections."""
        lines = content.split('\n')
        sections = []
        current_section = None
        section_stack = []  # For tracking hierarchy
        
        for i, line in enumerate(lines):
            # Check for markdown heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Close previous section
                if current_section:
                    current_section.end_line = i - 1
                    current_section.content = '\n'.join(
                        lines[current_section.start_line:current_section.end_line + 1]
                    )
                
                # Create new section with unique ID
                # Count all sections including nested ones
                def count_all(secs):
                    total = len(secs)
                    for s in secs:
                        total += count_all(s.subsections)
                    return total
                section_id = f"section_{count_all(sections) + 1}"
                current_section = Section(
                    id=section_id,
                    title=title,
                    level=level,
                    start_line=i,
                    end_line=len(lines) - 1,  # Will be updated
                    content=""
                )
                
                # Handle hierarchy
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                
                if section_stack:
                    section_stack[-1].subsections.append(current_section)
                else:
                    sections.append(current_section)
                
                section_stack.append(current_section)
        
        # Close final section
        if current_section:
            current_section.end_line = len(lines) - 1
            current_section.content = '\n'.join(
                lines[current_section.start_line:current_section.end_line + 1]
            )
        
        return sections
    
    def _flatten_sections(self, sections: List[Section]) -> Dict[str, Section]:
        """Flatten section hierarchy into a dict for easy lookup."""
        result = {}
        
        def add_section(section: Section):
            result[section.id] = section
            for sub in section.subsections:
                add_section(sub)
        
        for section in sections:
            add_section(section)
        
        return result
    
    def load(self, filepath: str) -> DocumentState:
        """Load a document and parse its structure."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        content = path.read_text(encoding='utf-8')
        sections = self._parse_sections(content)
        
        doc = DocumentState(
            filepath=path,
            original_content=content,
            current_content=content,
            sections=sections
        )
        
        self.documents[str(path)] = doc
        return doc
    
    def show_structure(self, doc: DocumentState, indent: int = 0) -> str:
        """Show the document structure as a tree."""
        lines = []
        
        def show_section(section: Section, depth: int):
            prefix = "  " * depth
            lines.append(f"{prefix}[{section.id}] {'#' * section.level} {section.title}")
            lines.append(f"{prefix}  Lines {section.start_line}-{section.end_line} ({section.end_line - section.start_line + 1} lines)")
            for sub in section.subsections:
                show_section(sub, depth + 1)
        
        for section in doc.sections:
            show_section(section, indent)
        
        return '\n'.join(lines)
    
    def get_section(self, doc: DocumentState, section_id: str) -> Optional[Section]:
        """Get a section by ID."""
        flat = self._flatten_sections(doc.sections)
        return flat.get(section_id)
    
    def get_section_content(self, doc: DocumentState, section_id: str) -> Optional[str]:
        """Get the current content of a section."""
        section = self.get_section(doc, section_id)
        if not section:
            return None
        
        lines = doc.current_content.split('\n')
        return '\n'.join(lines[section.start_line:section.end_line + 1])
    
    def edit_section(self, doc: DocumentState, section_id: str, new_content: str, 
                     operation: str = 'replace') -> bool:
        """
        Edit a section's content.
        
        Args:
            doc: Document to edit
            section_id: ID of section to edit
            new_content: New content
            operation: 'replace', 'append', 'prepend'
        """
        section = self.get_section(doc, section_id)
        if not section:
            print(f"Section not found: {section_id}")
            return False
        
        lines = doc.current_content.split('\n')
        old_content = '\n'.join(lines[section.start_line:section.end_line + 1])
        
        if operation == 'replace':
            final_content = new_content
        elif operation == 'append':
            final_content = old_content + '\n\n' + new_content
        elif operation == 'prepend':
            # Keep heading, prepend after it
            heading_line = lines[section.start_line]
            rest = '\n'.join(lines[section.start_line + 1:section.end_line + 1])
            final_content = heading_line + '\n\n' + new_content + '\n\n' + rest
        else:
            print(f"Unknown operation: {operation}")
            return False
        
        # Apply edit
        new_lines = lines[:section.start_line] + final_content.split('\n') + lines[section.end_line + 1:]
        doc.current_content = '\n'.join(new_lines)
        
        # Record edit
        doc.edit_history.append(EditOperation(
            section_id=section_id,
            operation=operation,
            old_content=old_content,
            new_content=final_content,
            timestamp=datetime.now().isoformat()
        ))
        
        # Re-parse sections
        doc.sections = self._parse_sections(doc.current_content)
        
        return True
    
    def show_diff(self, doc: DocumentState, context_lines: int = 3) -> str:
        """Show unified diff between original and current content."""
        original_lines = doc.original_content.split('\n')
        current_lines = doc.current_content.split('\n')
        
        diff = difflib.unified_diff(
            original_lines,
            current_lines,
            fromfile=f"{doc.filepath.name} (original)",
            tofile=f"{doc.filepath.name} (modified)",
            lineterm='',
            n=context_lines
        )
        
        return '\n'.join(diff)
    
    def show_section_diff(self, doc: DocumentState, section_id: str) -> str:
        """Show diff for a specific section only."""
        # Find the edit for this section
        for edit in reversed(doc.edit_history):
            if edit.section_id == section_id:
                old_lines = edit.old_content.split('\n')
                new_lines = edit.new_content.split('\n')
                
                diff = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"section {section_id} (before)",
                    tofile=f"section {section_id} (after)",
                    lineterm=''
                )
                return '\n'.join(diff)
        
        return "No edits found for this section"
    
    def get_edit_summary(self, doc: DocumentState) -> str:
        """Get a summary of all edits made."""
        if not doc.edit_history:
            return "No edits made"
        
        lines = [f"Edit History ({len(doc.edit_history)} edits):"]
        for i, edit in enumerate(doc.edit_history):
            old_len = len(edit.old_content)
            new_len = len(edit.new_content)
            delta = new_len - old_len
            sign = '+' if delta >= 0 else ''
            lines.append(f"  {i+1}. [{edit.section_id}] {edit.operation}: {sign}{delta} chars")
        
        return '\n'.join(lines)
    
    def save(self, doc: DocumentState, backup: bool = True) -> bool:
        """Save the document, optionally creating a backup."""
        try:
            if backup and doc.filepath.exists():
                backup_path = doc.filepath.with_suffix(doc.filepath.suffix + '.bak')
                backup_path.write_text(doc.original_content, encoding='utf-8')
            
            doc.filepath.write_text(doc.current_content, encoding='utf-8')
            
            # Update original to current
            doc.original_content = doc.current_content
            doc.edit_history.clear()
            
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False
    
    def revert(self, doc: DocumentState) -> bool:
        """Revert all changes."""
        doc.current_content = doc.original_content
        doc.sections = self._parse_sections(doc.current_content)
        doc.edit_history.clear()
        return True
    
    def revert_section(self, doc: DocumentState, section_id: str) -> bool:
        """Revert a specific section to its original state."""
        # Find original content for this section
        original_sections = self._parse_sections(doc.original_content)
        flat_original = self._flatten_sections(original_sections)
        
        if section_id not in flat_original:
            print(f"Section {section_id} not found in original")
            return False
        
        original_section = flat_original[section_id]
        original_lines = doc.original_content.split('\n')
        original_content = '\n'.join(
            original_lines[original_section.start_line:original_section.end_line + 1]
        )
        
        return self.edit_section(doc, section_id, original_content, 'replace')


def main():
    """Demo the structured editor."""
    editor = StructuredEditor()
    
    # Test with the paper
    paper_path = "/home/thorin/truthspace-lcm/phi_chat/paper_output/paper_manual_enhanced.md"
    
    if Path(paper_path).exists():
        doc = editor.load(paper_path)
        
        print("Document Structure:")
        print("=" * 60)
        print(editor.show_structure(doc))
        
        print("\n" + "=" * 60)
        print("Section 1 Content Preview:")
        print("=" * 60)
        content = editor.get_section_content(doc, "section_1")
        if content:
            print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print(f"Test file not found: {paper_path}")


if __name__ == "__main__":
    main()
