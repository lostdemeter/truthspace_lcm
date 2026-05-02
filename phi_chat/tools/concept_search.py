#!/usr/bin/env python3
"""
Concept Search Tool

Searches design documents by concept and returns actual content.
Uses the organized summaries as an index, then retrieves full content.

Usage:
    searcher = ConceptSearcher()
    results = searcher.search("φ-computer proof")
    # Returns actual paragraphs from relevant documents
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Paths
PROJECT_ROOT = Path("/home/thorin/truthspace-lcm")
DOCS_DIR = PROJECT_ROOT / "docs" / "design_considerations"
ORGANIZED_DIR = PROJECT_ROOT / "phi_chat" / "organized_docs"
SUMMARIES_PATH = ORGANIZED_DIR / "summaries.json"


@dataclass
class SearchResult:
    """A search result with context."""
    doc_number: int
    doc_title: str
    relevance_score: float
    matched_concepts: List[str]
    excerpts: List[str]  # Actual content excerpts
    filepath: Path


class ConceptSearcher:
    """
    Search design documents by concept.
    
    Two-phase search:
    1. Use summaries to find relevant documents (fast)
    2. Extract actual content from those documents (detailed)
    """
    
    def __init__(self):
        self.summaries: Dict[str, dict] = {}
        self.doc_cache: Dict[int, str] = {}
        self._load_summaries()
    
    def _load_summaries(self):
        """Load the organized summaries as search index."""
        if SUMMARIES_PATH.exists():
            with open(SUMMARIES_PATH) as f:
                self.summaries = json.load(f)
            print(f"Loaded {len(self.summaries)} document summaries")
        else:
            print("Warning: summaries.json not found")
    
    def _get_doc_path(self, doc_num: int) -> Optional[Path]:
        """Get the file path for a document number."""
        # Find matching file
        pattern = f"{doc_num:03d}_*.md"
        matches = list(DOCS_DIR.glob(pattern))
        if matches:
            return matches[0]
        
        # Try without leading zeros
        pattern = f"{doc_num}_*.md"
        matches = list(DOCS_DIR.glob(pattern))
        if matches:
            return matches[0]
        
        return None
    
    def _load_doc(self, doc_num: int) -> Optional[str]:
        """Load full document content."""
        if doc_num in self.doc_cache:
            return self.doc_cache[doc_num]
        
        filepath = self._get_doc_path(doc_num)
        if filepath and filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            self.doc_cache[doc_num] = content
            return content
        return None
    
    def _score_relevance(self, query_terms: List[str], summary: dict) -> Tuple[float, List[str]]:
        """Score how relevant a document is to query terms."""
        score = 0.0
        matched = []
        
        title = summary.get("title", "").lower()
        one_line = summary.get("one_line", "").lower()
        concepts = [c.lower() for c in summary.get("key_concepts", [])]
        themes = [t.lower() for t in summary.get("themes", [])]
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Title match (highest weight)
            if term_lower in title:
                score += 3.0
                matched.append(f"title:{term}")
            
            # One-line summary match
            if term_lower in one_line:
                score += 2.0
                matched.append(f"summary:{term}")
            
            # Concept match
            for concept in concepts:
                if term_lower in concept:
                    score += 1.5
                    matched.append(f"concept:{concept}")
                    break
            
            # Theme match
            for theme in themes:
                if term_lower in theme:
                    score += 1.0
                    matched.append(f"theme:{theme}")
                    break
        
        return score, matched
    
    def _extract_excerpts(self, content: str, query_terms: List[str], max_excerpts: int = 3) -> List[str]:
        """Extract relevant excerpts from document content."""
        excerpts = []
        paragraphs = content.split("\n\n")
        
        # Score each paragraph
        scored_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip short paragraphs
                continue
            
            score = 0
            para_lower = para.lower()
            for term in query_terms:
                if term.lower() in para_lower:
                    score += para_lower.count(term.lower())
            
            if score > 0:
                scored_paragraphs.append((score, para))
        
        # Sort by score and take top excerpts
        scored_paragraphs.sort(key=lambda x: -x[0])
        
        for score, para in scored_paragraphs[:max_excerpts]:
            # Truncate if too long
            if len(para) > 500:
                para = para[:500] + "..."
            excerpts.append(para)
        
        return excerpts
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search for documents matching a concept query.
        
        Args:
            query: Natural language query (e.g., "φ-computer proof", "attention speedup")
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult with actual content excerpts
        """
        # Tokenize query
        query_terms = re.findall(r'[\w\-φ]+', query)
        
        # Phase 1: Score all documents using summaries
        scored_docs = []
        for doc_num_str, summary in self.summaries.items():
            try:
                doc_num = int(doc_num_str)
            except ValueError:
                continue
            
            score, matched = self._score_relevance(query_terms, summary)
            if score > 0:
                scored_docs.append((score, doc_num, summary, matched))
        
        # Sort by score
        scored_docs.sort(key=lambda x: -x[0])
        
        # Phase 2: Load full content for top results
        results = []
        for score, doc_num, summary, matched in scored_docs[:max_results]:
            content = self._load_doc(doc_num)
            excerpts = []
            
            if content:
                excerpts = self._extract_excerpts(content, query_terms)
            
            filepath = self._get_doc_path(doc_num)
            
            results.append(SearchResult(
                doc_number=doc_num,
                doc_title=summary.get("title", f"Doc {doc_num}"),
                relevance_score=score,
                matched_concepts=matched,
                excerpts=excerpts,
                filepath=filepath
            ))
        
        return results
    
    def get_full_doc(self, doc_num: int) -> Optional[str]:
        """Get the full content of a specific document."""
        return self._load_doc(doc_num)
    
    def search_in_doc(self, doc_num: int, query: str) -> List[str]:
        """Search within a specific document for relevant sections."""
        content = self._load_doc(doc_num)
        if not content:
            return []
        
        query_terms = re.findall(r'[\w\-φ]+', query)
        return self._extract_excerpts(content, query_terms, max_excerpts=10)


def main():
    """Demo the concept search tool."""
    searcher = ConceptSearcher()
    
    # Test searches
    queries = [
        "φ-computer proof",
        "attention speedup boom",
        "transformer disentanglement scaffolding",
        "attractor repeller dynamics",
        "self-similarity",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        results = searcher.search(query, max_results=3)
        
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Doc {result.doc_number}: {result.doc_title}")
            print(f"    Score: {result.relevance_score:.1f}")
            print(f"    Matched: {', '.join(result.matched_concepts[:3])}")
            
            if result.excerpts:
                print(f"    Excerpt: {result.excerpts[0][:200]}...")


if __name__ == "__main__":
    main()
