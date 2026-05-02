#!/usr/bin/env python3
"""
φ-Space Navigation Planner

Instead of relying on the model to figure out planning, we use geometric
navigation through φ-space:

1. Goals and states are positions in embedding space
2. Actions move you through the space
3. Progress = distance to goal decreasing
4. The model chooses actions, but geometry validates them

Key insight from TruthSpace: Structure IS information. If we can represent
the planning problem geometrically, the model can navigate rather than plan.

Hypothesis: Geometric feedback will improve planning consistency.
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


@dataclass
class PlanningState:
    """State in the planning space."""
    goal: str
    knowledge: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    position: Optional[np.ndarray] = None  # Current position in φ-space
    goal_position: Optional[np.ndarray] = None  # Target position
    history: List[Tuple[str, float]] = field(default_factory=list)  # (action, distance)


class PhiSpacePlanner:
    """
    Planner that uses φ-space embeddings to guide navigation.
    
    The key idea: instead of asking "what should I do next?", we ask
    "which action moves me closer to the goal?"
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading φ-Space Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.searcher = ConceptSearcher()
        self.device = next(self.model.parameters()).device
        
        # φ constant
        self.phi = (1 + np.sqrt(5)) / 2
        
        print("✓ Planner loaded!\n")
    
    def _get_embedding(self, text: str, layer: int = 27) -> np.ndarray:
        """Get embedding at specified layer (default: layer 27 bottleneck)."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_idx = min(layer, len(hidden_states) - 1)
            embedding = hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
        
        return embedding
    
    def _compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute distance in φ-space (cosine distance)."""
        cos_sim = np.dot(pos1, pos2) / (np.linalg.norm(pos1) * np.linalg.norm(pos2) + 1e-10)
        return 1.0 - cos_sim  # Convert similarity to distance
    
    def _compute_progress(self, state: PlanningState) -> Dict[str, float]:
        """Compute progress metrics."""
        if state.position is None or state.goal_position is None:
            return {"distance": 1.0, "progress": 0.0}
        
        current_dist = self._compute_distance(state.position, state.goal_position)
        
        # Compare to initial distance
        if state.history:
            initial_dist = state.history[0][1]
            progress = (initial_dist - current_dist) / (initial_dist + 1e-10)
        else:
            progress = 0.0
        
        return {
            "distance": current_dist,
            "progress": progress,
            "at_goal": current_dist < 0.1  # Threshold for "close enough"
        }
    
    def _update_position(self, state: PlanningState) -> np.ndarray:
        """Update position based on current state."""
        # Combine goal, knowledge, and artifacts into a state representation
        state_text = f"Goal: {state.goal}\n"
        
        if state.knowledge:
            state_text += f"Knowledge: {' '.join(state.knowledge[-3:])}\n"
        
        if state.artifacts:
            artifact_preview = list(state.artifacts.values())[0][:200]
            state_text += f"Output: {artifact_preview}\n"
        
        return self._get_embedding(state_text)
    
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
    
    def _execute_tool(self, tool_name: str, params: Dict, state: PlanningState) -> str:
        """Execute a tool and update state."""
        
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
                    state.knowledge.append(excerpt[:200])
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
    
    def solve(self, goal: str, max_steps: int = 15) -> Dict:
        """
        Solve a goal using φ-space navigation.
        
        The model gets geometric feedback about its progress.
        """
        print(f"🎯 Goal: {goal}")
        print("=" * 60)
        
        # Initialize state
        state = PlanningState(goal=goal)
        
        # Compute goal position
        state.goal_position = self._get_embedding(f"Completed: {goal}")
        state.position = self._get_embedding(f"Starting: {goal}")
        
        initial_dist = self._compute_distance(state.position, state.goal_position)
        state.history.append(("start", initial_dist))
        
        print(f"📍 Initial distance to goal: {initial_dist:.4f}")
        
        # Build system prompt with geometric feedback
        system_prompt = f"""You are navigating through φ-space to achieve a goal.

GOAL: {goal}

TOOLS:
- search: Find information (moves toward knowledge)
- generate_and_save: Create output (moves toward completion)
- done: Complete (only when output exists)

Use: TOOL: {{"tool": "name", "param": "value"}}

You will receive DISTANCE feedback after each action. Lower distance = closer to goal.
Current distance: {initial_dist:.4f}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Navigate to complete: {goal}"}
        ]
        
        tool_calls = []
        success = False
        
        for step in range(max_steps):
            response = self.generate(messages, max_tokens=500)
            tool_name, params = self._parse_tool_call(response)
            
            if tool_name:
                tool_calls.append(tool_name)
                result = self._execute_tool(tool_name, params, state)
                
                # Update position and compute progress
                state.position = self._update_position(state)
                progress = self._compute_progress(state)
                state.history.append((tool_name, progress["distance"]))
                
                # Print progress
                direction = "↓" if len(state.history) > 1 and progress["distance"] < state.history[-2][1] else "↑"
                print(f"  Step {step+1}: {tool_name} | Distance: {progress['distance']:.4f} {direction}")
                
                if result == "GOAL_COMPLETE":
                    success = True
                    print(f"\n✅ Goal achieved!")
                    print(f"   Final distance: {progress['distance']:.4f}")
                    print(f"   Progress: {progress['progress']*100:.1f}%")
                    break
                
                # Add geometric feedback to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Result: {result}\n\n📍 Distance to goal: {progress['distance']:.4f} ({direction} from {state.history[-2][1]:.4f})\nProgress: {progress['progress']*100:.1f}%\n\nContinue navigating."
                })
            else:
                print(f"  Step {step+1}: No tool call")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Use a tool to continue."})
        
        # Analyze trajectory
        distances = [h[1] for h in state.history]
        trajectory_analysis = {
            "initial_distance": distances[0],
            "final_distance": distances[-1],
            "min_distance": min(distances),
            "monotonic_decrease": all(distances[i] >= distances[i+1] for i in range(len(distances)-1)),
            "total_progress": (distances[0] - distances[-1]) / distances[0] if distances[0] > 0 else 0
        }
        
        print(f"\n📊 Trajectory Analysis:")
        print(f"   Monotonic: {trajectory_analysis['monotonic_decrease']}")
        print(f"   Total progress: {trajectory_analysis['total_progress']*100:.1f}%")
        
        return {
            "success": success,
            "steps": len(tool_calls),
            "tool_calls": tool_calls,
            "trajectory": state.history,
            "analysis": trajectory_analysis,
            "artifacts": list(state.artifacts.keys())
        }


def run_phi_space_experiments():
    """Run experiments with φ-space navigation."""
    planner = PhiSpacePlanner()
    
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
    print("φ-SPACE NAVIGATION SUMMARY")
    print("=" * 60)
    
    for i, (goal, result) in enumerate(zip(goals, results)):
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} Goal {i+1}: {goal[:40]}...")
        print(f"   Steps: {result['steps']}")
        print(f"   Monotonic: {result['analysis']['monotonic_decrease']}")
        print(f"   Progress: {result['analysis']['total_progress']*100:.1f}%")
    
    # Save results
    results_path = OUTPUT_DIR / "phi_space_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump([{k: convert(v) if not isinstance(v, dict) else {kk: convert(vv) for kk, vv in v.items()} 
                   for k, v in r.items()} for r in results], f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    run_phi_space_experiments()
