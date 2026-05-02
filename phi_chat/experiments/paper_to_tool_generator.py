#!/usr/bin/env python3
"""
Paper-to-Tool Generator

This is a meta-level tool that:
1. Reads an existing paper (the "product")
2. Analyzes its structure, formulas, figures, and sections
3. Has Abbi generate a Python tool that can create similar papers

The idea: Product → Tool Generator
If we can do this, Abbi can learn to create tools from examples.
"""

import torch
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "generated_tools"
OUTPUT_DIR.mkdir(exist_ok=True)


class PaperAnalyzer:
    """Analyze a paper's structure."""
    
    def __init__(self, paper_path: str):
        self.paper_path = Path(paper_path)
        with open(self.paper_path, 'r') as f:
            self.content = f.read()
    
    def analyze(self) -> Dict:
        """Extract structural elements from the paper."""
        analysis = {
            'title': self._extract_title(),
            'sections': self._extract_sections(),
            'formulas': self._extract_formulas(),
            'code_blocks': self._extract_code_blocks(),
            'tables': self._extract_tables(),
            'figures': self._extract_figures(),
            'key_terms': self._extract_key_terms(),
        }
        return analysis
    
    def _extract_title(self) -> str:
        """Extract the paper title."""
        match = re.search(r'^# (.+)$', self.content, re.MULTILINE)
        return match.group(1) if match else "Unknown Title"
    
    def _extract_sections(self) -> List[Dict]:
        """Extract section structure."""
        sections = []
        # Find all ## headers
        pattern = r'^## (\d+\. )?(.+)$'
        for match in re.finditer(pattern, self.content, re.MULTILINE):
            section_name = match.group(2)
            start = match.end()
            sections.append({
                'name': section_name,
                'start': start,
            })
        
        # Calculate content for each section
        for i, section in enumerate(sections):
            end = sections[i+1]['start'] if i+1 < len(sections) else len(self.content)
            section['content'] = self.content[section['start']:end][:500]  # First 500 chars
            section['length'] = end - section['start']
        
        return sections
    
    def _extract_formulas(self) -> List[str]:
        """Extract LaTeX formulas."""
        # Display math $$...$$
        display = re.findall(r'\$\$(.+?)\$\$', self.content, re.DOTALL)
        # Inline math $...$
        inline = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', self.content)
        # \[...\] style
        bracket = re.findall(r'\\\[(.+?)\\\]', self.content, re.DOTALL)
        return display + bracket + inline[:10]  # Limit inline to 10
    
    def _extract_code_blocks(self) -> List[Dict]:
        """Extract code blocks."""
        blocks = []
        pattern = r'```python\s*(.*?)```'
        for match in re.finditer(pattern, self.content, re.DOTALL):
            code = match.group(1)
            has_figure = 'savefig' in code or 'plt.' in code
            blocks.append({
                'code': code[:300],  # First 300 chars
                'generates_figure': has_figure,
                'full_length': len(code)
            })
        return blocks
    
    def _extract_tables(self) -> List[str]:
        """Extract markdown tables."""
        tables = []
        lines = self.content.split('\n')
        in_table = False
        current_table = []
        
        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                current_table.append(line)
            elif in_table:
                if '|' in line:
                    current_table.append(line)
                else:
                    if current_table:
                        tables.append('\n'.join(current_table))
                    current_table = []
                    in_table = False
        
        if current_table:
            tables.append('\n'.join(current_table))
        
        return tables
    
    def _extract_figures(self) -> List[str]:
        """Extract figure references."""
        return re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', self.content)
    
    def _extract_key_terms(self) -> List[str]:
        """Extract key technical terms."""
        # Look for bold terms and special notation
        bold = re.findall(r'\*\*([^*]+)\*\*', self.content)
        phi_terms = re.findall(r'φ-\w+', self.content)
        return list(set(bold[:20] + phi_terms[:10]))
    
    def get_summary(self) -> str:
        """Get a text summary of the analysis."""
        analysis = self.analyze()
        
        summary = f"""
PAPER ANALYSIS SUMMARY
======================

Title: {analysis['title']}

STRUCTURE:
- Sections: {len(analysis['sections'])}
  {', '.join(s['name'] for s in analysis['sections'])}

- LaTeX Formulas: {len(analysis['formulas'])}
  Examples: {analysis['formulas'][:3] if analysis['formulas'] else 'None'}

- Code Blocks: {len(analysis['code_blocks'])}
  Figure-generating: {sum(1 for b in analysis['code_blocks'] if b['generates_figure'])}

- Tables: {len(analysis['tables'])}

- Figure References: {len(analysis['figures'])}

- Key Terms: {', '.join(analysis['key_terms'][:10])}

SECTION DETAILS:
"""
        for section in analysis['sections']:
            summary += f"\n{section['name']} ({section['length']} chars)"
        
        return summary


class PaperToToolGenerator:
    """Generate a tool from a paper example."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Initializing Paper-to-Tool Generator...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Generator ready.\n")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        messages = [
            {"role": "system", "content": "You are Abbi, a Truthspace LCM that generates Python code. Write clean, well-documented code."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in full_output.lower():
            parts = full_output.split("assistant")
            return parts[-1].strip().lstrip(":").strip()
        return full_output[len(text):].strip()
    
    def generate_tool_from_paper(self, paper_path: str) -> str:
        """Read a paper and generate a tool to create similar papers."""
        
        print("=" * 60)
        print("PAPER-TO-TOOL GENERATION")
        print("=" * 60)
        
        # Step 1: Analyze the paper
        print("\n1. Analyzing paper structure...")
        analyzer = PaperAnalyzer(paper_path)
        analysis = analyzer.analyze()
        summary = analyzer.get_summary()
        print(summary)
        
        # Step 2: Read the full paper content
        with open(paper_path, 'r') as f:
            paper_content = f.read()
        
        # Step 3: Generate the tool specification
        print("\n2. Generating tool specification...")
        spec_prompt = f"""
I have analyzed a scientific paper and extracted its structure. I need you to generate a Python tool that can create similar papers.

PAPER ANALYSIS:
{summary}

SAMPLE PAPER CONTENT (first 3000 chars):
{paper_content[:3000]}

Based on this analysis, generate a SPECIFICATION for a Python tool that could create similar papers. Include:

1. What sections the tool should generate
2. What types of formulas it should include
3. What figures it should create (with matplotlib)
4. What data/knowledge it needs as input
5. The overall workflow

Write the specification as a structured document.
"""
        
        spec = self.generate(spec_prompt, max_tokens=1500)
        print("  ✓ Specification generated")
        
        # Step 4: Generate the actual Python tool
        print("\n3. Generating Python tool code...")
        code_prompt = f"""
Based on this paper structure and specification, generate a complete Python tool that can create similar scientific papers.

PAPER STRUCTURE:
- Title pattern: {analysis['title']}
- Sections: {[s['name'] for s in analysis['sections']]}
- Has {len(analysis['formulas'])} LaTeX formulas
- Has {len(analysis['code_blocks'])} code blocks ({sum(1 for b in analysis['code_blocks'] if b['generates_figure'])} generate figures)
- Has {len(analysis['tables'])} tables

SPECIFICATION:
{spec}

EXAMPLE FORMULAS FROM PAPER:
{analysis['formulas'][:5]}

EXAMPLE CODE BLOCK FROM PAPER:
{analysis['code_blocks'][0]['code'] if analysis['code_blocks'] else 'None'}

Generate a complete Python class called `PaperGenerator` that:
1. Takes a topic/title and knowledge base as input
2. Has methods to generate each section
3. Includes methods to generate matplotlib figures
4. Outputs a complete markdown paper with LaTeX formulas
5. Follows the same structure as the analyzed paper

Write the complete Python code with all imports and a main function to demonstrate usage.
"""
        
        tool_code = self.generate(code_prompt, max_tokens=3000)
        print("  ✓ Tool code generated")
        
        # Step 5: Extract and save the Python code
        print("\n4. Extracting and saving tool...")
        
        # Try to extract code block
        code_match = re.search(r'```python\s*(.*?)```', tool_code, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1)
        else:
            # If no code block, use the whole response
            extracted_code = tool_code
        
        # Save the generated tool
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tool_path = OUTPUT_DIR / f"paper_generator_{timestamp}.py"
        
        # Add header
        full_code = f'''#!/usr/bin/env python3
"""
Auto-Generated Paper Generator Tool

Generated by: Abbi (Truthspace LCM)
Source paper: {paper_path}
Generated at: {datetime.now().isoformat()}

This tool was reverse-engineered from an example paper.
"""

{extracted_code}
'''
        
        with open(tool_path, 'w') as f:
            f.write(full_code)
        
        print(f"  ✓ Saved to: {tool_path}")
        
        # Step 6: Save the full generation log
        log_path = OUTPUT_DIR / f"generation_log_{timestamp}.md"
        log_content = f"""# Paper-to-Tool Generation Log

## Source Paper
{paper_path}

## Paper Analysis
{summary}

## Generated Specification
{spec}

## Generated Tool Code
```python
{extracted_code}
```
"""
        with open(log_path, 'w') as f:
            f.write(log_content)
        print(f"  ✓ Log saved to: {log_path}")
        
        return str(tool_path)


def run_paper_to_tool(paper_path: str):
    """Run the paper-to-tool generation."""
    generator = PaperToToolGenerator()
    tool_path = generator.generate_tool_from_paper(paper_path)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated tool: {tool_path}")
    print("\nThis tool was reverse-engineered from the paper and can")
    print("be used to generate similar papers on different topics.")
    
    return tool_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        paper_path = sys.argv[1]
    else:
        # Default to the paper we just generated
        paper_path = "/home/thorin/truthspace-lcm/phi_chat/experiments/abbi_arxiv_output/abbi_arxiv_paper_20260204_150032.md"
    
    run_paper_to_tool(paper_path)
