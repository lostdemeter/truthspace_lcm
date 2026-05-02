#!/usr/bin/env python3
"""
Planning Experiments

A series of experiments to understand what makes autonomous planning work:

1. Tool Complexity vs Autonomy - How does tool granularity affect success?
2. Geometric Planning - Can φ-space navigation guide planning?
3. Self-Improvement Loop - Can the planner learn from failures?
4. Reflection Depth - Does more thinking help?

Each experiment measures:
- Success rate (did it complete the goal?)
- Step efficiency (how many steps to complete?)
- Output quality (is the output substantive?)
- Autonomy level (how much guidance was needed?)
"""

import torch
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Add tools directory
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from concept_search import ConceptSearcher

# Output directory
OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_name: str
    goal: str
    success: bool
    steps_taken: int
    artifacts_created: List[str]
    output_length: int
    tool_calls: List[str]
    errors: List[str]
    duration_seconds: float
    notes: str = ""


@dataclass 
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    tools: Dict[str, Dict]
    system_prompt_additions: str = ""
    max_steps: int = 15
    require_artifact: bool = True


class PlanningExperimentRunner:
    """
    Runs planning experiments with different configurations.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model for experiments...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.searcher = ConceptSearcher()
        self.results: List[ExperimentResult] = []
        print("Model loaded!\n")
    
    def generate(self, messages: List[Dict], max_tokens: int = 600) -> str:
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
        if "assistant" in response.lower():
            parts = response.split("assistant")
            response = parts[-1].strip()
        return response
    
    def _parse_tool_call(self, response: str) -> Tuple[Optional[str], Dict]:
        """Parse a tool call from response."""
        match = re.search(r'TOOL:\s*(\{[^}]+\})', response, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                json_str = match.group(1).replace('\n', ' ')
                tool_data = json.loads(json_str)
                return tool_data.get("tool"), tool_data
            except json.JSONDecodeError:
                pass
        return None, {}
    
    def _execute_tool(self, tool_name: str, params: Dict, state: Dict) -> str:
        """Execute a tool with given state."""
        
        if tool_name == "search":
            query = params.get("query", "")
            results = self.searcher.search(query, max_results=3)
            if not results:
                return f"No results for: {query}"
            
            output = []
            for r in results:
                output.append(f"Doc {r.doc_number}: {r.doc_title}")
                if r.excerpts:
                    excerpt = r.excerpts[0][:300]
                    output.append(f"  {excerpt}...")
                    state["knowledge"].append(f"[Doc {r.doc_number}] {excerpt[:200]}")
            return "\n".join(output)
        
        elif tool_name == "think":
            thought = params.get("thought", "")
            state["reflections"].append(thought)
            return f"Thought recorded."
        
        elif tool_name == "plan":
            steps = params.get("steps", [])
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split(",")]
            state["plan"] = steps
            return f"Plan created with {len(steps)} steps."
        
        elif tool_name == "generate_and_save":
            filename = params.get("filename", "output.md")
            topic = params.get("topic", state.get("goal", ""))
            
            if not state["knowledge"]:
                return "ERROR: No knowledge gathered. Use search first."
            
            knowledge_text = "\n\n".join(state["knowledge"][:5])
            gen_messages = [
                {"role": "system", "content": "Write a research summary based on the information. Be specific."},
                {"role": "user", "content": f"Topic: {topic}\n\nInfo:\n{knowledge_text}\n\nWrite a summary:"}
            ]
            content = self.generate(gen_messages, max_tokens=800)
            content = content.strip()
            if not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            filepath = OUTPUT_DIR / filename
            filepath.write_text(content, encoding='utf-8')
            state["artifacts"][filename] = content
            return f"SUCCESS: Saved {len(content)} chars to {filename}"
        
        elif tool_name == "write":
            filename = params.get("filename", "output.md")
            content = params.get("content", "")
            if len(content) < 50:
                return "ERROR: Content too short. Provide substantial content."
            filepath = OUTPUT_DIR / filename
            filepath.write_text(content, encoding='utf-8')
            state["artifacts"][filename] = content
            return f"SUCCESS: Saved {len(content)} chars to {filename}"
        
        elif tool_name == "reflect":
            # Analyze current state and suggest next action
            if not state["knowledge"]:
                return "REFLECTION: No knowledge gathered yet. Should search for information."
            if not state["artifacts"]:
                return "REFLECTION: Have knowledge but no output created. Should generate content."
            return "REFLECTION: Have knowledge and artifacts. Can complete if satisfied."
        
        elif tool_name == "done":
            if state.get("require_artifact", True) and not state["artifacts"]:
                return "ERROR: No artifacts created. Must create output before completing."
            return "GOAL_COMPLETE"
        
        return f"Unknown tool: {tool_name}"
    
    def run_experiment(self, config: ExperimentConfig, goal: str) -> ExperimentResult:
        """Run a single experiment with given configuration."""
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"Goal: {goal}")
        print('='*60)
        
        start_time = time.time()
        
        # Initialize state
        state = {
            "goal": goal,
            "knowledge": [],
            "artifacts": {},
            "reflections": [],
            "plan": [],
            "require_artifact": config.require_artifact
        }
        
        tool_calls = []
        errors = []
        
        # Build tool descriptions
        tool_desc = "\n".join([
            f"**{name}**: {info['description']}\n  Example: {info.get('example', '')}"
            for name, info in config.tools.items()
        ])
        
        system_prompt = f"""You are an autonomous agent solving a goal using tools.

GOAL: {goal}

AVAILABLE TOOLS:
{tool_desc}

{config.system_prompt_additions}

Use tools by outputting: TOOL: {{"tool": "name", "param": "value"}}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Achieve this goal: {goal}"}
        ]
        
        success = False
        
        for step in range(config.max_steps):
            response = self.generate(messages, max_tokens=600)
            tool_name, params = self._parse_tool_call(response)
            
            if tool_name:
                tool_calls.append(tool_name)
                print(f"  Step {step+1}: {tool_name}")
                
                result = self._execute_tool(tool_name, params, state)
                
                if "ERROR" in result:
                    errors.append(result)
                
                if result == "GOAL_COMPLETE":
                    success = True
                    print(f"  ✓ Goal achieved!")
                    break
                
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Result: {result}\n\nContinue."})
            else:
                print(f"  Step {step+1}: No tool call")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Use a tool to continue."})
        
        duration = time.time() - start_time
        
        # Calculate output length
        output_length = sum(len(c) for c in state["artifacts"].values())
        
        result = ExperimentResult(
            experiment_name=config.name,
            goal=goal,
            success=success,
            steps_taken=len(tool_calls),
            artifacts_created=list(state["artifacts"].keys()),
            output_length=output_length,
            tool_calls=tool_calls,
            errors=errors,
            duration_seconds=duration
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to JSON."""
        filepath = OUTPUT_DIR / filename
        data = [asdict(r) for r in self.results]
        filepath.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f"\nResults saved to {filepath}")


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Experiment 1: Minimal tools (what we know works)
MINIMAL_TOOLS = ExperimentConfig(
    name="minimal_3_tools",
    description="Minimal 3-tool setup that we know works",
    tools={
        "search": {
            "description": "Search for information",
            "example": 'TOOL: {"tool": "search", "query": "topic"}'
        },
        "generate_and_save": {
            "description": "Generate summary from knowledge and save to file",
            "example": 'TOOL: {"tool": "generate_and_save", "filename": "out.md", "topic": "topic"}'
        },
        "done": {
            "description": "Signal completion (after creating output)",
            "example": 'TOOL: {"tool": "done"}'
        }
    }
)

# Experiment 2: Granular tools (more autonomy required)
GRANULAR_TOOLS = ExperimentConfig(
    name="granular_6_tools",
    description="More granular tools requiring more planning",
    tools={
        "search": {
            "description": "Search for information",
            "example": 'TOOL: {"tool": "search", "query": "topic"}'
        },
        "think": {
            "description": "Record a thought or reasoning step",
            "example": 'TOOL: {"tool": "think", "thought": "I should..."}'
        },
        "plan": {
            "description": "Create a plan with steps",
            "example": 'TOOL: {"tool": "plan", "steps": ["step1", "step2"]}'
        },
        "write": {
            "description": "Write content to a file (you provide the content)",
            "example": 'TOOL: {"tool": "write", "filename": "out.md", "content": "# Title..."}'
        },
        "reflect": {
            "description": "Analyze current state and get suggestions",
            "example": 'TOOL: {"tool": "reflect"}'
        },
        "done": {
            "description": "Signal completion",
            "example": 'TOOL: {"tool": "done"}'
        }
    }
)

# Experiment 3: With explicit reflection prompts
REFLECTION_TOOLS = ExperimentConfig(
    name="reflection_guided",
    description="Tools with mandatory reflection",
    tools={
        "search": {
            "description": "Search for information",
            "example": 'TOOL: {"tool": "search", "query": "topic"}'
        },
        "reflect": {
            "description": "REQUIRED: Analyze state before acting",
            "example": 'TOOL: {"tool": "reflect"}'
        },
        "generate_and_save": {
            "description": "Generate and save output",
            "example": 'TOOL: {"tool": "generate_and_save", "filename": "out.md", "topic": "topic"}'
        },
        "done": {
            "description": "Signal completion",
            "example": 'TOOL: {"tool": "done"}'
        }
    },
    system_prompt_additions="IMPORTANT: Use 'reflect' after each action to analyze your progress."
)

# Experiment 4: Geometric guidance (placeholder for φ-space)
GEOMETRIC_TOOLS = ExperimentConfig(
    name="geometric_guided",
    description="Tools with geometric progress indicators",
    tools={
        "search": {
            "description": "Search for information. Progress: moves you toward knowledge.",
            "example": 'TOOL: {"tool": "search", "query": "topic"}'
        },
        "generate_and_save": {
            "description": "Generate output. Progress: moves you toward completion.",
            "example": 'TOOL: {"tool": "generate_and_save", "filename": "out.md", "topic": "topic"}'
        },
        "done": {
            "description": "Complete goal. Only valid when at destination.",
            "example": 'TOOL: {"tool": "done"}'
        }
    },
    system_prompt_additions="""Think of this as NAVIGATION:
- You start at GOAL position
- search moves you toward KNOWLEDGE
- generate_and_save moves you toward OUTPUT
- done is only valid when you have OUTPUT
Current position: GOAL → need KNOWLEDGE → need OUTPUT → DONE"""
)


# Test goals
TEST_GOALS = [
    "Write a summary about the φ-computer proof",
    "Explain the transformer disentanglement discovery",
    "Summarize the boom-newton attention findings",
]


def run_all_experiments():
    """Run all experiments and compare results."""
    runner = PlanningExperimentRunner()
    
    configs = [MINIMAL_TOOLS, GRANULAR_TOOLS, REFLECTION_TOOLS, GEOMETRIC_TOOLS]
    
    for config in configs:
        for goal in TEST_GOALS:  # Run all goals
            runner.run_experiment(config, goal)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for result in runner.results:
        status = "✓" if result.success else "✗"
        print(f"\n{status} {result.experiment_name}")
        print(f"   Steps: {result.steps_taken}, Output: {result.output_length} chars")
        print(f"   Tools used: {result.tool_calls}")
        if result.errors:
            print(f"   Errors: {len(result.errors)}")
    
    runner.save_results()
    
    return runner.results


if __name__ == "__main__":
    run_all_experiments()
