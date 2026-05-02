#!/usr/bin/env python3
"""
Milestone-Based Planner

Instead of continuous distance, use discrete milestones:
1. GOAL_UNDERSTOOD - We know what we need to do
2. KNOWLEDGE_GATHERED - We have relevant information
3. OUTPUT_CREATED - We have produced deliverables
4. GOAL_COMPLETE - We are done

Each milestone is a checkpoint. The model gets clear feedback about
which milestones are achieved and which remain.

This is more like how humans plan - we don't measure continuous distance,
we check off discrete steps.
"""

import torch
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from concept_search import ConceptSearcher

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)


class Milestone(Enum):
    """Planning milestones."""
    START = 0
    KNOWLEDGE_GATHERED = 1
    OUTPUT_CREATED = 2
    COMPLETE = 3


@dataclass
class MilestoneState:
    """State with milestone tracking."""
    goal: str
    current_milestone: Milestone = Milestone.START
    knowledge: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    milestone_history: List[Tuple[str, Milestone]] = field(default_factory=list)


class MilestonePlanner:
    """
    Planner with discrete milestone feedback.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading Milestone Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.searcher = ConceptSearcher()
        self.device = next(self.model.parameters()).device
        print("✓ Planner loaded!\n")
    
    def _check_milestones(self, state: MilestoneState) -> Milestone:
        """Check which milestone we're at based on state."""
        if state.artifacts:
            return Milestone.OUTPUT_CREATED
        elif len(state.knowledge) >= 2:
            return Milestone.KNOWLEDGE_GATHERED
        else:
            return Milestone.START
    
    def _get_milestone_feedback(self, state: MilestoneState) -> str:
        """Generate milestone feedback for the model."""
        current = state.current_milestone
        
        milestones = [
            ("START", current.value >= 0, "✓" if current.value > 0 else "→"),
            ("KNOWLEDGE_GATHERED", current.value >= 1, "✓" if current.value > 1 else ("→" if current.value == 1 else "○")),
            ("OUTPUT_CREATED", current.value >= 2, "✓" if current.value > 2 else ("→" if current.value == 2 else "○")),
            ("COMPLETE", current.value >= 3, "✓" if current.value >= 3 else "○"),
        ]
        
        lines = ["MILESTONE PROGRESS:"]
        for name, achieved, icon in milestones:
            lines.append(f"  {icon} {name}")
        
        # Add specific guidance
        if current == Milestone.START:
            lines.append("\nNEXT: Use 'search' to gather knowledge")
        elif current == Milestone.KNOWLEDGE_GATHERED:
            lines.append("\nNEXT: Use 'generate_and_save' to create output")
        elif current == Milestone.OUTPUT_CREATED:
            lines.append("\nNEXT: Use 'done' to complete")
        
        return "\n".join(lines)
    
    def generate(self, messages: List[Dict], max_tokens: int = 600) -> str:
        """Generate a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
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
        """Parse tool call from response."""
        match = re.search(r'TOOL:\s*(\{[^}]+\})', response, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                json_str = match.group(1).replace('\n', ' ')
                tool_data = json.loads(json_str)
                return tool_data.get("tool"), tool_data
            except json.JSONDecodeError:
                pass
        return None, {}
    
    def _execute_tool(self, tool_name: str, params: Dict, state: MilestoneState) -> str:
        """Execute a tool and update state."""
        
        if tool_name == "search":
            query = params.get("query", state.goal)  # Default to goal if no query
            if not query:
                query = state.goal
            results = self.searcher.search(query, max_results=3)
            if not results:
                return f"No results for: {query}"
            
            output = [f"Search: {query}"]
            for r in results:
                output.append(f"Doc {r.doc_number}: {r.doc_title}")
                if r.excerpts:
                    excerpt = r.excerpts[0][:300]
                    output.append(f"  {excerpt}...")
                    state.knowledge.append(excerpt[:200])
                else:
                    # Still add something from the title/summary
                    state.knowledge.append(f"Doc {r.doc_number}: {r.doc_title}")
            return "\n".join(output)
        
        elif tool_name == "generate_and_save":
            filename = params.get("filename", "output.md")
            topic = params.get("topic", state.goal)
            
            if not state.knowledge:
                return "ERROR: No knowledge. Search first."
            
            knowledge_text = "\n\n".join(state.knowledge[:5])
            gen_messages = [
                {"role": "system", "content": "Write a research summary. Be specific."},
                {"role": "user", "content": f"Topic: {topic}\n\nInfo:\n{knowledge_text}\n\nWrite:"}
            ]
            content = self.generate(gen_messages, max_tokens=800)
            content = f"# {topic}\n\n{content.strip()}"
            
            filepath = OUTPUT_DIR / filename
            filepath.write_text(content, encoding='utf-8')
            state.artifacts[filename] = content
            return f"SUCCESS: Saved {len(content)} chars to {filename}"
        
        elif tool_name == "done":
            if not state.artifacts:
                return "ERROR: No output created. Generate first."
            return "GOAL_COMPLETE"
        
        return f"Unknown tool: {tool_name}"
    
    def solve(self, goal: str, max_steps: int = 10) -> Dict:
        """Solve a goal with milestone feedback."""
        print(f"🎯 Goal: {goal}")
        print("=" * 60)
        
        state = MilestoneState(goal=goal)
        state.milestone_history.append(("start", Milestone.START))
        
        print(self._get_milestone_feedback(state))
        
        system_prompt = f"""You are completing a goal step by step.

GOAL: {goal}

TOOLS:
- search: Find information
- generate_and_save: Create output file
- done: Complete (only when output exists)

Use: TOOL: {{"tool": "name", "param": "value"}}

Follow the MILESTONE PROGRESS to know what to do next."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Complete: {goal}\n\n{self._get_milestone_feedback(state)}"}
        ]
        
        tool_calls = []
        success = False
        
        for step in range(max_steps):
            response = self.generate(messages, max_tokens=500)
            tool_name, params = self._parse_tool_call(response)
            
            if tool_name:
                tool_calls.append(tool_name)
                result = self._execute_tool(tool_name, params, state)
                
                # Update milestone
                old_milestone = state.current_milestone
                state.current_milestone = self._check_milestones(state)
                state.milestone_history.append((tool_name, state.current_milestone))
                
                # Check for milestone advancement
                advanced = state.current_milestone.value > old_milestone.value
                milestone_icon = "🎉" if advanced else ""
                
                print(f"  Step {step+1}: {tool_name} → {state.current_milestone.name} (knowledge={len(state.knowledge)}) {milestone_icon}")
                
                if result == "GOAL_COMPLETE":
                    state.current_milestone = Milestone.COMPLETE
                    success = True
                    print(f"\n✅ Goal achieved!")
                    break
                
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Result: {result}\n\n{self._get_milestone_feedback(state)}"
                })
            else:
                print(f"  Step {step+1}: No tool call")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Use a tool.\n\n{self._get_milestone_feedback(state)}"})
        
        return {
            "success": success,
            "steps": len(tool_calls),
            "tool_calls": tool_calls,
            "milestones": [(t, m.name) for t, m in state.milestone_history],
            "artifacts": list(state.artifacts.keys())
        }


def run_milestone_experiments():
    """Run experiments with milestone feedback."""
    planner = MilestonePlanner()
    
    goals = [
        "Write a summary about the φ-computer proof",
        "Explain the transformer disentanglement discovery",
        "Summarize the boom-newton attention findings",
    ]
    
    results = []
    for goal in goals:
        result = planner.solve(goal)
        results.append(result)
        print("\n")
    
    # Summary
    print("=" * 60)
    print("MILESTONE PLANNER SUMMARY")
    print("=" * 60)
    
    successes = sum(1 for r in results if r["success"])
    print(f"\nSuccess rate: {successes}/{len(results)}")
    
    for i, (goal, result) in enumerate(zip(goals, results)):
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} Goal {i+1}: {goal[:40]}...")
        print(f"   Steps: {result['steps']}")
        print(f"   Tools: {result['tool_calls']}")
    
    # Save results
    results_path = OUTPUT_DIR / "milestone_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    run_milestone_experiments()
