#!/usr/bin/env python3
"""
Safe Dial Planner - φ-Space Navigation v2

Based on insights from:
- Doc 189: Safe Dial Mechanism - layers are like rotors that change as you work
- Doc 203: φ-Space Interface - navigation through concept space
- Doc 205: CRUD Operations - all operations are geometric
- Doc 206: Conceptual Nexus - the model's self-control interface

Key insights from the failed v1 experiment:
1. Single-layer distance doesn't capture progress
2. Start and finish might be the same position (reverse navigation)
3. The "plates" (context) change shape as you take actions

New approach:
1. Track TRAJECTORY through all layers, not just layer 27
2. Use the "click point" (layer 3) as a key indicator
3. Model progress as "rotor alignment" - are the plates clicking into place?
4. The goal is the ORIGIN - we navigate TO it, not FROM it
"""

import torch
import json
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from concept_search import ConceptSearcher

OUTPUT_DIR = Path(__file__).parent / "planning_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


@dataclass
class LayerState:
    """State at a specific layer."""
    layer: int
    phi_level: float  # Mean φ-level of activations
    norm: float  # Magnitude
    
    
@dataclass
class Trajectory:
    """Full trajectory through all layers."""
    layers: List[LayerState]
    click_layer: int = 3  # The "click point"
    bottleneck_layer: int = 27  # The convergence point
    
    @property
    def click_phi(self) -> float:
        """φ-level at the click point."""
        return self.layers[self.click_layer].phi_level if len(self.layers) > self.click_layer else 0.0
    
    @property
    def bottleneck_phi(self) -> float:
        """φ-level at the bottleneck."""
        return self.layers[self.bottleneck_layer].phi_level if len(self.layers) > self.bottleneck_layer else 0.0
    
    @property
    def convergence_quality(self) -> float:
        """How close bottleneck is to φ."""
        return 1.0 - abs(self.bottleneck_phi - 1.0) / 2.0  # Normalize to [0, 1]


@dataclass
class DialState:
    """State of the safe dial mechanism."""
    goal: str
    current_text: str  # Full context so far
    knowledge: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    trajectories: List[Tuple[str, Trajectory]] = field(default_factory=list)  # (action, trajectory)
    rotor_alignments: List[float] = field(default_factory=list)  # Click quality over time


class SafeDialPlanner:
    """
    Planner using the safe dial mechanism.
    
    The key insight: planning is like working a combination lock.
    - Each action "rotates the dial"
    - The rotors (layers) change shape based on context
    - Progress = rotors clicking into alignment
    - Goal achieved = all rotors aligned (trajectory converges to φ)
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading Safe Dial Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.searcher = ConceptSearcher()
        self.device = next(self.model.parameters()).device
        print("✓ Planner loaded!\n")
    
    def _get_trajectory(self, text: str) -> Trajectory:
        """Get full trajectory through all layers."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        layers = []
        for i, h in enumerate(hidden_states):
            # Get last token's hidden state
            state = h[0, -1, :].float().cpu().numpy()
            
            # Compute φ-level
            magnitudes = np.abs(state)
            magnitudes = magnitudes[magnitudes > 1e-10]
            phi_levels = np.log(magnitudes) / LOG_PHI
            mean_phi = float(np.mean(phi_levels))
            
            layers.append(LayerState(
                layer=i,
                phi_level=mean_phi,
                norm=float(np.linalg.norm(state))
            ))
        
        return Trajectory(layers=layers)
    
    def _compute_rotor_alignment(self, trajectory: Trajectory, state: 'DialState') -> float:
        """
        Compute how well the rotors are aligned.
        
        Based on Doc 189: The click happens at layer 3.
        
        NEW INSIGHT: The alignment should measure COMPLETENESS, not distance.
        - Have we gathered knowledge? (rotor 1)
        - Have we created output? (rotor 2)
        - Is the trajectory stable? (rotor 3)
        
        Each "rotor" clicking into place = progress toward goal.
        """
        if len(trajectory.layers) < 28:
            return 0.0
        
        # Rotor 1: Knowledge gathered (0 or 1)
        knowledge_rotor = 1.0 if len(state.knowledge) >= 2 else len(state.knowledge) / 2.0
        
        # Rotor 2: Output created (0 or 1)
        output_rotor = 1.0 if state.artifacts else 0.0
        
        # Rotor 3: Trajectory quality (continuous)
        # Measure how close bottleneck is to φ
        bottleneck_phi = trajectory.bottleneck_phi
        phi_distance = abs(bottleneck_phi - 1.0)  # Distance from φ^1
        trajectory_rotor = 1.0 / (1.0 + phi_distance)
        
        # Combined alignment: all rotors must click
        # Weight: trajectory quality matters less than state completion
        alignment = 0.4 * knowledge_rotor + 0.4 * output_rotor + 0.2 * trajectory_rotor
        
        return float(alignment)
    
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
    
    def _execute_tool(self, tool_name: str, params: Dict, state: DialState) -> str:
        """Execute a tool and update state."""
        
        if tool_name == "search":
            query = params.get("query", state.goal)
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
                    state.knowledge.append(f"Doc {r.doc_number}: {r.doc_title}")
            
            # Update context
            state.current_text += f"\n[Searched: {query}]"
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
            
            # Update context
            state.current_text += f"\n[Created: {filename}]"
            return f"SUCCESS: Saved {len(content)} chars to {filename}"
        
        elif tool_name == "done":
            if not state.artifacts:
                return "ERROR: No output created. Generate first."
            return "GOAL_COMPLETE"
        
        return f"Unknown tool: {tool_name}"
    
    def solve(self, goal: str, max_steps: int = 10) -> Dict:
        """
        Solve a goal using the safe dial mechanism.
        
        Track trajectory through layers and rotor alignment.
        """
        print(f"🎯 Goal: {goal}")
        print("=" * 60)
        
        # Initialize state - goal is the ORIGIN (reverse navigation)
        state = DialState(
            goal=goal,
            current_text=f"Goal: {goal}"
        )
        
        # Get initial trajectory (from goal position)
        initial_traj = self._get_trajectory(state.current_text)
        initial_alignment = self._compute_rotor_alignment(initial_traj, state)
        state.trajectories.append(("start", initial_traj))
        state.rotor_alignments.append(initial_alignment)
        
        print(f"📍 Initial state:")
        print(f"   Click (L3) φ: {initial_traj.click_phi:.3f}")
        print(f"   Bottleneck (L27) φ: {initial_traj.bottleneck_phi:.3f}")
        print(f"   Rotor alignment: {initial_alignment:.3f}")
        
        system_prompt = f"""You are navigating through φ-space to achieve a goal.

GOAL: {goal}

TOOLS (use exactly this format):
- search: TOOL: {{"tool": "search", "query": "your search terms"}}
- generate_and_save: TOOL: {{"tool": "generate_and_save", "filename": "output.md", "topic": "topic"}}
- done: TOOL: {{"tool": "done"}}

WORKFLOW: search → generate_and_save → done

You will receive ROTOR ALIGNMENT feedback. Higher = closer to goal.
Start by searching for information about the goal."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Navigate to complete: {goal}\n\nRotor alignment: {initial_alignment:.3f}"}
        ]
        
        tool_calls = []
        success = False
        
        for step in range(max_steps):
            response = self.generate(messages, max_tokens=500)
            tool_name, params = self._parse_tool_call(response)
            
            if tool_name:
                tool_calls.append(tool_name)
                result = self._execute_tool(tool_name, params, state)
                
                # Get new trajectory after action
                new_traj = self._get_trajectory(state.current_text)
                new_alignment = self._compute_rotor_alignment(new_traj, state)
                state.trajectories.append((tool_name, new_traj))
                state.rotor_alignments.append(new_alignment)
                
                # Check if alignment improved
                prev_alignment = state.rotor_alignments[-2]
                delta = new_alignment - prev_alignment
                direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                
                print(f"  Step {step+1}: {tool_name}")
                print(f"    Click φ: {new_traj.click_phi:.3f} | Bottleneck φ: {new_traj.bottleneck_phi:.3f}")
                print(f"    Alignment: {new_alignment:.3f} {direction} (Δ={delta:+.3f})")
                
                if result == "GOAL_COMPLETE":
                    success = True
                    print(f"\n✅ Goal achieved!")
                    print(f"   Final alignment: {new_alignment:.3f}")
                    break
                
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Result: {result}\n\n🔧 Rotor alignment: {new_alignment:.3f} {direction}\nClick φ: {new_traj.click_phi:.3f}\nBottleneck φ: {new_traj.bottleneck_phi:.3f}\n\nContinue."
                })
            else:
                print(f"  Step {step+1}: No tool call")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Use a tool to continue."})
        
        # Analyze trajectory
        alignments = state.rotor_alignments
        trajectory_analysis = {
            "initial_alignment": alignments[0],
            "final_alignment": alignments[-1],
            "max_alignment": max(alignments),
            "improvement": alignments[-1] - alignments[0],
            "monotonic": all(alignments[i] <= alignments[i+1] for i in range(len(alignments)-1)),
            "click_phi_trajectory": [t[1].click_phi for t in state.trajectories],
            "bottleneck_phi_trajectory": [t[1].bottleneck_phi for t in state.trajectories]
        }
        
        print(f"\n📊 Trajectory Analysis:")
        print(f"   Alignment: {alignments[0]:.3f} → {alignments[-1]:.3f} (Δ={trajectory_analysis['improvement']:+.3f})")
        print(f"   Monotonic improvement: {trajectory_analysis['monotonic']}")
        
        return {
            "success": success,
            "steps": len(tool_calls),
            "tool_calls": tool_calls,
            "alignments": alignments,
            "analysis": trajectory_analysis,
            "artifacts": list(state.artifacts.keys())
        }


def run_safe_dial_experiments():
    """Run experiments with safe dial mechanism."""
    planner = SafeDialPlanner()
    
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
    print("SAFE DIAL PLANNER SUMMARY")
    print("=" * 60)
    
    successes = sum(1 for r in results if r["success"])
    print(f"\nSuccess rate: {successes}/{len(results)}")
    
    for i, (goal, result) in enumerate(zip(goals, results)):
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} Goal {i+1}: {goal[:40]}...")
        print(f"   Steps: {result['steps']}")
        print(f"   Alignment: {result['analysis']['initial_alignment']:.3f} → {result['analysis']['final_alignment']:.3f}")
        print(f"   Improvement: {result['analysis']['improvement']:+.3f}")
    
    # Save results
    results_path = OUTPUT_DIR / "safe_dial_results.json"
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    run_safe_dial_experiments()
