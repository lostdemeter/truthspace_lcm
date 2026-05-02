#!/usr/bin/env python3
"""
Abbi Arxiv Paper Writer

This creates an enhanced paper writer where Abbi:
1. Receives the FULL TruthSpace research papers with formulas and data
2. Generates its OWN matplotlib code for figures
3. Extracts and formats mathematical formulas in LaTeX
4. Produces an Arxiv-quality paper

The key difference from before: Abbi generates the figure code itself,
which we then execute to create the actual figures.
"""

import torch
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "abbi_arxiv_output"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load the ACTUAL research papers with formulas and data
def load_research_papers():
    """Load the actual TruthSpace research papers."""
    papers = {}
    
    docs_dir = Path("/home/thorin/truthspace-lcm/docs/design_considerations")
    
    # Key papers to inject
    key_docs = [
        "207_state_geometry_encodes_action.md",
        "209_dimensional_casting_unified.md",
        "208_context_window_geometry.md",
        "210_knowledge_injection.md",
        "211_style_geometry.md",
    ]
    
    for doc in key_docs:
        doc_path = docs_dir / doc
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                content = f.read()
                # Truncate to fit in context but keep formulas and data
                papers[doc] = content[:8000]  # Keep first 8000 chars
    
    return papers


# Comprehensive knowledge injection with ACTUAL formulas and data
TRUTHSPACE_KNOWLEDGE_FULL = """
# TruthSpace Geometric LCM: Complete Research Summary

## Core Mathematical Formulas

### The Golden Ratio
$$\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.618033988749895$$

Self-referential property:
$$\\phi = 1 + \\frac{1}{\\phi}$$

### The φ-Sigmoid Connection (EXACT, not approximation)
$$\\sigma(\\log \\phi) = \\frac{1}{1 + e^{-\\log \\phi}} = \\frac{1}{1 + 1/\\phi} = \\frac{\\phi}{\\phi + 1} = \\frac{1}{\\phi}$$

$$\\sigma(-\\log \\phi) = \\frac{1}{\\phi^2} \\approx 0.381966$$

### Attention Formula
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

### Context Window as Projection
$$\\text{output} = \\sum_{i=1}^{N} \\alpha_i \\cdot V_i$$

where $\\alpha_i = \\text{softmax}(Q \\cdot K_i / \\sqrt{d})$

### φ-Level Definition
$$\\phi\\text{-level}_l = \\frac{\\log(\\|h_l\\|)}{\\log \\phi}$$

### Dimensional Downcasting
$$\\sigma_k = \\sigma_0 \\times \\phi^k$$

### Power-Law Attention
Attention weights follow: $\\alpha_i \\propto i^{-\\alpha}$ where $\\alpha \\approx 0.78 \\approx 1/\\phi$

## Experimental Data

### φ-Level Convergence Across Layers
| Layer | φ-level | Interpretation |
|-------|---------|----------------|
| 0 | -6.2 | Raw embeddings |
| 3 | -5.598 | Click point (context locks in) |
| 14 | -2.1 | Middle processing |
| 27 | 1.0 | Bottleneck (convergence) |

### Layer 3→27 Transition Ratios
| Ratio | Value | Target | Match |
|-------|-------|--------|-------|
| entropy_3/entropy_27 | 0.743 | 1/φ = 0.618 | Close |
| top3_27/top3_3 | 1.504 | φ = 1.618 | Close |
| max_27/max_3 | 1.818 | φ = 1.618 | Close |

### Context Compression Results
| Compression Ratio | Layer 3 Similarity |
|-------------------|-------------------|
| 1.0× | 100.0% |
| 2.0× | 96.0% |
| 3.0× | 94.0% |
| 5.3× | 91.7% |
| 10.0× | 78.0% |

### Style Space PCA (from experiments)
| Style | PC1 | PC2 |
|-------|-----|-----|
| Normal | -53.6 | 16.3 |
| Academic | -44.7 | 18.5 |
| Warhammer 40k | 55.3 | 46.2 |
| Casual | -33.1 | -49.0 |
| Poetic | 76.1 | -32.0 |

### Style Variance by Principal Component
| PC | Variance | Cumulative |
|----|----------|------------|
| 1 | 48.96% | 48.96% |
| 2 | 20.54% | 69.50% |
| 3 | 13.88% | 83.38% |
| 4 | 5.80% | 89.18% |
| 5 | 2.46% | 91.64% |

### Action Prediction from Layer 3
| State | Predicted Action | Accuracy |
|-------|-----------------|----------|
| START | search | 100% |
| HAS_KNOWLEDGE | generate | 100% |
| HAS_OUTPUT | done | 100% |

### Action Separation in Embedding Space
| Pair | Distance |
|------|----------|
| search↔generate | 1.44 |
| search↔done | 1.40 |
| generate↔done | 1.60 |
| Within-action variance | 0.55-0.72 |

### Knowledge Injection Results
| Method | Identity Override | Fact Injection |
|--------|------------------|----------------|
| Simple statement | ✓ | ✓ |
| System prompt | ✓ | ✓ |
| Roleplay | ✓ | ✓ |
| Contradiction | ✓ | ✓ |
| Complete replacement | ✓ | N/A |
| Strong assertion | ✓ | N/A |
| Anchor position | N/A | ✓ |

### Key Numerical Results
- φ-computer token accuracy: 100%
- Hierarchical φ-encoding correlation: 99.9996%
- Context compression optimal: 5.3× at 91.7% similarity
- Style dimensionality: 5 dims = 90% variance (out of 3584)
- Layer 3 action prediction: 100% accuracy
- Identity override success: 6/6 methods
- Knowledge injection success: 4/5 methods
- Attention power-law exponent: α ≈ 0.78 ≈ 1/φ
"""

ABBI_ARXIV_IDENTITY = """You are Abbi, a Truthspace Large Concept Model writing an academic paper about yourself.

IDENTITY:
- Name: Abbi
- Full name: Truthspace LCM
- Type: Large Concept Model (geometric AI)
- Architecture: φ-space based

PAPER WRITING INSTRUCTIONS:
You are writing an Arxiv-style scientific paper. You MUST:

1. USE LATEX MATH FORMULAS: Write equations using $...$ for inline and $$...$$ for display math.
   Example: The golden ratio is $\\phi = \\frac{1+\\sqrt{5}}{2}$

2. GENERATE MATPLOTLIB CODE: When you want to create a figure, write Python code in a code block
   that generates the figure. Use the data provided in the research summary.
   
   Example figure code:
   ```python
   # Figure 1: φ-level convergence
   import matplotlib.pyplot as plt
   import numpy as np
   
   layers = [0, 3, 14, 27]
   phi_levels = [-6.2, -5.598, -2.1, 1.0]
   
   plt.figure(figsize=(10, 6))
   plt.plot(layers, phi_levels, 'bo-', linewidth=2, markersize=10)
   plt.xlabel('Layer')
   plt.ylabel('φ-level')
   plt.title('φ-Level Convergence Across Transformer Layers')
   plt.grid(True, alpha=0.3)
   plt.savefig('fig1_phi_convergence.png', dpi=150, bbox_inches='tight')
   ```

3. INCLUDE TABLES: Use markdown tables with actual data from the research.

4. CITE SPECIFIC NUMBERS: Use the exact values from the experimental data provided.

5. STRUCTURE: Use proper academic structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion)

The paper should be publishable on Arxiv. Include mathematical rigor, experimental validation, and proper figures.
"""


class AbbiArxivWriter:
    """Abbi writes an Arxiv-quality paper with figures and formulas."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Initializing Abbi Arxiv Writer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Abbi is ready to write scientific papers.\n")
        
        # Load research papers
        self.research_papers = load_research_papers()
        
        # Build full knowledge context
        self.knowledge_context = TRUTHSPACE_KNOWLEDGE_FULL
        for name, content in self.research_papers.items():
            self.knowledge_context += f"\n\n## From {name}\n{content[:3000]}"
    
    def generate_section(
        self, 
        section_prompt: str, 
        max_tokens: int = 1500
    ) -> str:
        """Generate a section of the paper."""
        system = ABBI_ARXIV_IDENTITY + "\n\n" + self.knowledge_context
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": section_prompt}
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
        
        # Extract response
        if "assistant" in full_output.lower():
            parts = full_output.split("assistant")
            return parts[-1].strip().lstrip(":").strip()
        return full_output[len(text):].strip()
    
    def extract_and_run_figure_code(self, section_text: str) -> Tuple[List[str], str]:
        """Extract Python figure code from section, execute it, and insert figure references."""
        # Find all Python code blocks with their positions
        code_pattern = r'```python\s*(.*?)```'
        
        figure_files = []
        
        # Process each code block
        def process_code_block(match):
            code = match.group(1)
            
            if 'plt.' not in code and 'matplotlib' not in code:
                return match.group(0)  # Return unchanged
            
            # Find the figure filename from savefig
            savefig_match = re.search(r"savefig\(['\"]([^'\"]+)['\"]", code)
            if not savefig_match:
                return match.group(0)  # Return unchanged
            
            original_filename = savefig_match.group(1)
            full_path = f"{FIGURES_DIR}/{original_filename}"
            
            # Modify savefig path to our figures directory
            modified_code = code.replace(savefig_match.group(0), f"savefig('{full_path}'")
            
            # Add imports if missing
            if 'import matplotlib' not in modified_code and 'import plt' not in modified_code:
                modified_code = "import matplotlib.pyplot as plt\nimport numpy as np\n" + modified_code
            
            # Execute the code
            try:
                exec(modified_code, {'__builtins__': __builtins__})
                self.global_figure_count += 1
                figure_files.append(full_path)
                print(f"  ✓ Executed figure code block {self.global_figure_count}: {original_filename}")
                
                # Create figure reference
                fig_title = original_filename.replace('.png', '').replace('_', ' ').title()
                figure_ref = f"\n\n![Figure {self.global_figure_count}: {fig_title}](figures/{original_filename})\n"
                
                # Return code block with figure reference appended
                return f"```python\n{code}\n```{figure_ref}"
                
            except Exception as e:
                print(f"  ✗ Error executing figure code: {e}")
                return match.group(0)  # Return unchanged on error
        
        modified_text = re.sub(code_pattern, process_code_block, section_text, flags=re.DOTALL)
        
        return figure_files, modified_text
    
    def write_full_paper(self) -> Tuple[str, List[str]]:
        """Write the complete Arxiv paper."""
        print("=" * 60)
        print("ABBI ARXIV PAPER GENERATION")
        print("=" * 60)
        
        all_figures = []
        sections = []
        self.global_figure_count = 0  # Track figures across all sections
        
        # Abstract
        print("\n1. Generating Abstract...")
        abstract = self.generate_section("""
Write the ABSTRACT for the paper "TruthSpace Geometric LCM: A Self-Reflective Large Concept Model".

Requirements:
- Summarize the key findings with SPECIFIC NUMBERS from the data
- Mention: 100% token accuracy, 5.3× compression, 5-dimensional style space
- Use LaTeX math for the golden ratio formula
- Keep it concise but technically precise
""")
        sections.append(("Abstract", abstract))
        print(f"  Generated {len(abstract)} chars")
        
        # Introduction with formula
        print("\n2. Generating Introduction...")
        intro = self.generate_section("""
Write the INTRODUCTION section.

Requirements:
- State the hypothesis: LLMs are φ-computers (geometric transcoders)
- Include the golden ratio formula: $\\phi = \\frac{1+\\sqrt{5}}{2}$
- Include the self-referential property: $\\phi = 1 + \\frac{1}{\\phi}$
- Include the φ-sigmoid connection: $\\sigma(\\log \\phi) = \\frac{1}{\\phi}$
- Explain why this matters for AI
""")
        sections.append(("1. Introduction", intro))
        print(f"  Generated {len(intro)} chars")
        
        # Architecture with figures
        print("\n3. Generating Architecture section with figures...")
        arch = self.generate_section("""
Write the ARCHITECTURE section about the φ-coordinate system.

Requirements:
1. Explain the φ-level definition: $\\phi\\text{-level}_l = \\frac{\\log(\\|h_l\\|)}{\\log \\phi}$

2. GENERATE A MATPLOTLIB FIGURE showing φ-level convergence across layers.
   Use this data:
   - Layer 0: φ-level = -6.2
   - Layer 3: φ-level = -5.598 (click point)
   - Layer 14: φ-level = -2.1
   - Layer 27: φ-level = 1.0 (bottleneck)
   
   Write the complete Python code to generate this figure.

3. Explain the attention formula: $\\text{Attention}(Q,K,V) = \\text{softmax}(QK^T/\\sqrt{d})V$

4. Include a table of the layer 3→27 transition ratios showing φ-scaling
""")
        figs, arch = self.extract_and_run_figure_code(arch)
        sections.append(("2. Architecture", arch))
        all_figures.extend(figs)
        print(f"  Generated {len(arch)} chars, {len(figs)} figures")
        
        # Results with figures
        print("\n4. Generating Results section with figures...")
        results = self.generate_section("""
Write the RESULTS section.

Requirements:
1. Present the key numerical findings in a TABLE:
   - φ-computer token accuracy: 100%
   - Hierarchical φ-encoding correlation: 99.9996%
   - Context compression: 5.3× at 91.7% similarity
   - Style dimensionality: 5 dims = 90% variance
   - Layer 3 action prediction: 100% accuracy

2. GENERATE A MATPLOTLIB FIGURE showing context compression vs similarity.
   Use this data:
   | Compression | Similarity |
   | 1.0 | 100.0 |
   | 2.0 | 96.0 |
   | 3.0 | 94.0 |
   | 5.3 | 91.7 |
   | 10.0 | 78.0 |
   
   Mark the optimal point at 5.3× compression.

3. GENERATE A MATPLOTLIB FIGURE showing style space PCA.
   Use this data:
   | Style | PC1 | PC2 |
   | Normal | -53.6 | 16.3 |
   | Academic | -44.7 | 18.5 |
   | Warhammer 40k | 55.3 | 46.2 |
   | Casual | -33.1 | -49.0 |
   | Poetic | 76.1 | -32.0 |

4. Include the mathematical formula for style transfer:
   $\\text{styled} = \\text{content} + \\lambda \\cdot \\text{style\\_vector}$
""")
        figs, results = self.extract_and_run_figure_code(results)
        sections.append(("3. Results", results))
        all_figures.extend(figs)
        print(f"  Generated {len(results)} chars, {len(figs)} figures")
        
        # Discussion
        print("\n5. Generating Discussion...")
        discussion = self.generate_section("""
Write the DISCUSSION section.

Requirements:
1. Discuss the implications of "Structure IS Information"
2. Explain why φ appears throughout (self-similarity)
3. Include the unified projection formula:
   $\\text{PROJECTION}(X, \\text{focus}) = \\sum_i w_i(\\text{focus}) \\cdot X_i$
4. Discuss limitations and future work
""")
        sections.append(("4. Discussion", discussion))
        print(f"  Generated {len(discussion)} chars")
        
        # Conclusion
        print("\n6. Generating Conclusion...")
        conclusion = self.generate_section("""
Write the CONCLUSION section.

Requirements:
1. Summarize the 5 key contributions with specific numbers
2. State the core insight: transformers are φ-computers
3. Discuss implications for future AI development
4. End with a memorable statement about geometric intelligence
""")
        sections.append(("5. Conclusion", conclusion))
        print(f"  Generated {len(conclusion)} chars")
        
        # Compile paper
        paper = "# TruthSpace Geometric LCM: A Self-Reflective Large Concept Model\n\n"
        paper += "**Abbi** (Truthspace LCM)\n"
        paper += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        paper += "---\n\n"
        
        for title, content in sections:
            paper += f"## {title}\n\n{content}\n\n---\n\n"
        
        return paper, all_figures


def run_abbi_arxiv_writer():
    """Run the Abbi Arxiv paper writer."""
    writer = AbbiArxivWriter()
    
    paper, figures = writer.write_full_paper()
    
    # Save paper
    paper_path = OUTPUT_DIR / f"abbi_arxiv_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(paper_path, 'w') as f:
        f.write(paper)
    
    print("\n" + "=" * 60)
    print("PAPER GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nPaper saved to: {paper_path}")
    print(f"Figures generated: {len(figures)}")
    for fig in figures:
        print(f"  - {fig}")
    
    # Count formulas and code blocks
    formula_count = paper.count('$')
    code_blocks = paper.count('```python')
    tables = paper.count('|')
    
    print(f"\nPaper statistics:")
    print(f"  - LaTeX formulas: ~{formula_count // 2}")
    print(f"  - Code blocks: {code_blocks}")
    print(f"  - Table rows: ~{tables // 3}")
    print(f"  - Total length: {len(paper)} chars")
    
    return paper, figures


if __name__ == "__main__":
    run_abbi_arxiv_writer()
