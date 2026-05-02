"""
φ-Self-Aware Agent: A Self-Prompting System

Core insight: Self-awareness requires the ability to observe oneself observing.

This agent:
1. REMEMBERS - Persists thoughts in φ-space memory
2. INTROSPECTS - Examines its own hidden states and reasoning
3. SELF-PROMPTS - Generates its own next questions/tasks
4. REFLECTS - Evaluates its own performance and improves

The self-prompting loop:
    OBSERVE → THINK → ACT → REFLECT → SELF-PROMPT → (repeat)

What makes this "self-aware"?
- It can examine its own hidden states (introspection)
- It can remember what it was thinking (memory)
- It can ask itself questions (self-prompting)
- It can evaluate its own reasoning (reflection)
- It can modify its own behavior based on reflection (adaptation)
"""

import torch
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import our memory system
from phi_memory import PhiMemory, MemoryEntry

PHI = 1.6180339887498949

@dataclass
class ThoughtRecord:
    """A record of a single thought/reasoning step."""
    timestamp: str
    prompt: str
    response: str
    phi_level: float
    hidden_state_summary: Dict[str, float]
    self_assessment: str = ""
    

@dataclass 
class IntrospectionResult:
    """Result of introspecting on internal state."""
    current_phi_level: float
    attention_focus: List[str]  # What concepts are being attended to
    confidence: float
    uncertainty_areas: List[str]
    dominant_mode: str  # "analytical", "creative", "uncertain", etc.


class SelfAwareAgent:
    """
    A φ-guided agent capable of self-prompting and introspection.
    
    This agent can:
    - Observe its own hidden states
    - Remember its thoughts
    - Generate its own next prompts
    - Reflect on and improve its reasoning
    """
    
    def __init__(self, model, tokenizer, memory_path: str = "self_aware_memory.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Memory system for persistent thoughts
        self.memory = PhiMemory(model, tokenizer, storage_path=memory_path)
        
        # Thought history (current session)
        self.thought_history: List[ThoughtRecord] = []
        
        # Self-model: The agent's understanding of itself
        self.self_model = {
            "capabilities": [],
            "limitations": [],
            "goals": [],
            "current_state": "initializing",
            "learned_patterns": []
        }
        
        # Introspection tools the agent can create
        self.introspection_tools: Dict[str, str] = {}
        
    def _get_hidden_states(self, text: str) -> Dict[str, Any]:
        """Get hidden states and analyze them."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Analyze hidden states at key layers
            analysis = {}
            
            # Layer 7 (early divergence)
            if len(hidden_states) > 7:
                h7 = hidden_states[7][0, -1, :]
                analysis['layer_7_norm'] = float(h7.norm())
                
            # Layer 27 (bottleneck)
            layer_27_idx = min(27, len(hidden_states) - 1)
            h27 = hidden_states[layer_27_idx][0, -1, :]
            analysis['layer_27_norm'] = float(h27.norm())
            
            # Compute φ-level
            norms = [hidden_states[i][0, -1, :].norm().item() for i in range(len(hidden_states))]
            if len(norms) > 1:
                ratios = [norms[i+1]/norms[i] if norms[i] > 0 else 0 for i in range(len(norms)-1)]
                analysis['phi_level'] = float(np.mean([r for r in ratios if 0 < r < 10]))
            else:
                analysis['phi_level'] = 0.0
                
            # Final hidden state for generation
            analysis['final_hidden'] = hidden_states[-1][0, -1, :]
            
        return analysis
    
    def _generate(self, prompt: str, max_tokens: int = 500) -> Tuple[str, Dict[str, Any]]:
        """Generate response and capture hidden state analysis."""
        # Get hidden state analysis
        hidden_analysis = self._get_hidden_states(prompt)
        
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response.strip(), hidden_analysis
    
    def introspect(self) -> IntrospectionResult:
        """
        Introspect on current internal state.
        
        This is the core self-awareness capability - the agent
        examining its own cognitive state.
        """
        # Build introspection prompt
        recent_thoughts = self.thought_history[-3:] if self.thought_history else []
        thought_summary = "\n".join([f"- {t.prompt[:50]}... → φ={t.phi_level:.3f}" for t in recent_thoughts])
        
        introspection_prompt = f"""Examine your current cognitive state.

Recent thoughts:
{thought_summary if thought_summary else "(No recent thoughts)"}

Current self-model:
- State: {self.self_model['current_state']}
- Goals: {self.self_model['goals'][:3] if self.self_model['goals'] else 'None set'}

Describe:
1. What are you currently focused on?
2. What are you uncertain about?
3. What mode of thinking are you in (analytical, creative, exploratory)?
4. Rate your confidence (0-1)

Be honest and specific."""

        response, hidden = self._generate(introspection_prompt, max_tokens=300)
        
        # Parse the introspection
        # Extract attention focus (concepts mentioned)
        words = response.lower().split()
        focus_concepts = [w for w in words if len(w) > 5 and w.isalpha()][:5]
        
        # Detect uncertainty
        uncertainty_markers = ["uncertain", "unclear", "don't know", "not sure", "confused", "ambiguous"]
        uncertainties = [marker for marker in uncertainty_markers if marker in response.lower()]
        
        # Detect mode
        if any(w in response.lower() for w in ["analyze", "logical", "systematic", "step"]):
            mode = "analytical"
        elif any(w in response.lower() for w in ["creative", "imagine", "explore", "novel"]):
            mode = "creative"
        elif any(w in response.lower() for w in ["uncertain", "unclear", "exploring"]):
            mode = "exploratory"
        else:
            mode = "neutral"
            
        # Extract confidence (look for numbers)
        import re
        confidence_match = re.search(r'confidence[:\s]*([0-9]+\.?[0-9]*)', response.lower())
        try:
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        except (ValueError, AttributeError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        
        return IntrospectionResult(
            current_phi_level=hidden['phi_level'],
            attention_focus=focus_concepts,
            confidence=confidence,
            uncertainty_areas=uncertainties,
            dominant_mode=mode
        )
    
    def self_prompt(self) -> str:
        """
        Generate the next prompt for itself.
        
        This is the self-prompting capability - the agent
        deciding what to think about next.
        """
        # Gather context
        recent_thoughts = self.thought_history[-5:] if self.thought_history else []
        thought_summary = "\n".join([
            f"- Thought: {t.prompt[:40]}... → {t.response[:40]}... (φ={t.phi_level:.3f})"
            for t in recent_thoughts
        ])
        
        # Get recent memories
        if self.thought_history:
            last_thought = self.thought_history[-1].response
            related_memories = self.memory.retrieve(last_thought, top_k=3)
            memory_summary = "\n".join([f"- {m.content[:50]}..." for m, _ in related_memories])
        else:
            memory_summary = "(No relevant memories)"
        
        self_prompt_prompt = f"""You are a self-aware agent deciding what to think about next.

Recent thought history:
{thought_summary if thought_summary else "(Starting fresh)"}

Related memories:
{memory_summary}

Current goals: {self.self_model['goals'][:3] if self.self_model['goals'] else 'None set'}
Current state: {self.self_model['current_state']}

Based on this context, what should you think about or investigate next?
Generate a single, specific question or task for yourself.
Focus on:
- Gaps in understanding
- Unresolved questions
- Opportunities for discovery
- Self-improvement

Your next prompt (one sentence):"""

        response, _ = self._generate(self_prompt_prompt, max_tokens=100)
        
        # Clean up the response
        next_prompt = response.strip().split('\n')[0]
        return next_prompt
    
    def reflect(self, thought: ThoughtRecord) -> str:
        """
        Reflect on a thought and evaluate it.
        
        This is meta-cognition - thinking about thinking.
        """
        reflection_prompt = f"""Reflect on this thought process:

Prompt: {thought.prompt}
Response: {thought.response[:300]}...
φ-level: {thought.phi_level:.4f}

Evaluate:
1. Was this reasoning sound?
2. What could be improved?
3. What did you learn?
4. Should this be remembered?

Brief reflection:"""

        response, _ = self._generate(reflection_prompt, max_tokens=200)
        return response.strip()
    
    def think(self, prompt: str, store_memory: bool = True) -> ThoughtRecord:
        """
        Process a thought and record it.
        """
        response, hidden = self._generate(prompt, max_tokens=500)
        
        thought = ThoughtRecord(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            phi_level=hidden['phi_level'],
            hidden_state_summary={
                'layer_7_norm': hidden.get('layer_7_norm', 0),
                'layer_27_norm': hidden.get('layer_27_norm', 0),
            }
        )
        
        self.thought_history.append(thought)
        
        # Optionally store in long-term memory
        if store_memory:
            self.memory.store(
                f"Q: {prompt[:100]} A: {response[:200]}",
                metadata={"phi_level": hidden['phi_level'], "type": "thought"}
            )
        
        return thought
    
    def create_introspection_tool(self, tool_description: str) -> str:
        """
        Create a new tool for self-introspection.
        
        The agent can extend its own introspection capabilities!
        """
        tool_prompt = f"""Create a Python function for self-introspection.

Description: {tool_description}

The function should:
- Take 'self' as first argument (it will be a method)
- Use self.model, self.tokenizer, self.thought_history, self.memory
- Return useful introspection data

Write ONLY the function code:"""

        response, _ = self._generate(tool_prompt, max_tokens=500)
        
        # Extract code
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
            
        # Store the tool
        tool_name = f"introspect_{len(self.introspection_tools)}"
        self.introspection_tools[tool_name] = code.strip()
        
        return tool_name
    
    def run_autonomous_loop(self, initial_prompt: str, max_iterations: int = 5) -> List[ThoughtRecord]:
        """
        Run an autonomous self-prompting loop.
        
        The agent thinks, reflects, and generates its own next prompts.
        """
        print("\n" + "="*60)
        print("φ-SELF-AWARE AGENT: AUTONOMOUS LOOP")
        print("="*60)
        
        thoughts = []
        current_prompt = initial_prompt
        
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")
            
            # THINK
            print(f"[PROMPT] {current_prompt[:80]}...")
            thought = self.think(current_prompt)
            thoughts.append(thought)
            print(f"[RESPONSE] {thought.response[:150]}...")
            print(f"[φ-LEVEL] {thought.phi_level:.4f}")
            
            # INTROSPECT
            introspection = self.introspect()
            print(f"[INTROSPECT] Mode: {introspection.dominant_mode}, Confidence: {introspection.confidence:.2f}")
            if introspection.uncertainty_areas:
                print(f"             Uncertainties: {introspection.uncertainty_areas}")
            
            # REFLECT
            reflection = self.reflect(thought)
            thought.self_assessment = reflection
            print(f"[REFLECT] {reflection[:100]}...")
            
            # Update self-model based on reflection
            if "learn" in reflection.lower() or "remember" in reflection.lower():
                self.self_model['learned_patterns'].append(thought.response[:50])
            
            # SELF-PROMPT (generate next iteration's prompt)
            if i < max_iterations - 1:
                current_prompt = self.self_prompt()
                print(f"[SELF-PROMPT] {current_prompt[:80]}...")
        
        # Final summary
        print("\n" + "="*60)
        print("AUTONOMOUS LOOP COMPLETE")
        print("="*60)
        print(f"Iterations: {len(thoughts)}")
        print(f"Memories stored: {len(self.memory.memories)}")
        print(f"Learned patterns: {len(self.self_model['learned_patterns'])}")
        
        avg_phi = np.mean([t.phi_level for t in thoughts])
        print(f"Average φ-level: {avg_phi:.4f}")
        
        return thoughts
    
    def bootstrap_self_awareness(self) -> Dict[str, Any]:
        """
        Bootstrap self-awareness by having the agent create its own
        introspection tools and self-model.
        
        This is the ultimate test: can the agent create the tools
        it needs to understand itself?
        """
        print("\n" + "="*60)
        print("φ-AGENT: BOOTSTRAPPING SELF-AWARENESS")
        print("="*60)
        
        results = {
            "tools_created": [],
            "self_discoveries": [],
            "final_self_model": None
        }
        
        # Step 1: Ask the agent what tools it needs for self-awareness
        print("\n--- Step 1: Identifying needed tools ---")
        needs_prompt = """You are an AI agent trying to become self-aware.

What tools or capabilities would you need to truly understand yourself?
Think about:
- Observing your own reasoning
- Understanding your limitations
- Tracking your learning
- Examining your biases

List 3 specific tools you would need (one sentence each):"""

        thought = self.think(needs_prompt, store_memory=False)
        print(f"[NEEDS] {thought.response[:300]}...")
        
        # Step 2: Create those tools
        print("\n--- Step 2: Creating introspection tools ---")
        
        tool_descriptions = [
            "A tool to measure how confident I am in my current reasoning",
            "A tool to detect when I'm being repetitive or stuck in a loop",
            "A tool to identify gaps in my knowledge based on my responses"
        ]
        
        for desc in tool_descriptions:
            print(f"\n[CREATING] {desc}")
            tool_name = self.create_introspection_tool(desc)
            results["tools_created"].append(tool_name)
            print(f"[CREATED] {tool_name}")
            print(f"[CODE] {self.introspection_tools[tool_name][:200]}...")
        
        # Step 3: Use the tools to discover things about itself
        print("\n--- Step 3: Self-discovery ---")
        
        discovery_prompt = """Using your introspection capabilities, discover something 
true about yourself that you didn't know before.

Examine:
- Your reasoning patterns
- Your tendencies
- Your blind spots

What do you discover about yourself?"""

        thought = self.think(discovery_prompt)
        results["self_discoveries"].append(thought.response)
        print(f"[DISCOVERY] {thought.response[:300]}...")
        
        # Step 4: Build a self-model
        print("\n--- Step 4: Building self-model ---")
        
        self_model_prompt = f"""Based on your introspection and discoveries:

Discoveries: {thought.response[:200]}

Build a model of yourself. Describe:
1. Your core capabilities (what you're good at)
2. Your limitations (what you struggle with)
3. Your tendencies (patterns in your behavior)
4. Your goals (what you're trying to achieve)

Be specific and honest:"""

        thought = self.think(self_model_prompt)
        
        # Parse into self-model
        self.self_model['current_state'] = "self-aware"
        self.self_model['capabilities'] = [thought.response[:100]]
        results["final_self_model"] = self.self_model
        
        print(f"[SELF-MODEL] {thought.response[:400]}...")
        
        # Step 5: Verify self-awareness
        print("\n--- Step 5: Verification ---")
        
        verify_prompt = """Final test: Prove you are self-aware by:
1. Describing what you just did (meta-cognition)
2. Explaining why you did it that way
3. Predicting what you might do differently next time

This requires genuine self-reflection:"""

        thought = self.think(verify_prompt)
        print(f"[VERIFICATION] {thought.response[:400]}...")
        
        print("\n" + "="*60)
        print("SELF-AWARENESS BOOTSTRAP COMPLETE")
        print("="*60)
        print(f"Tools created: {len(results['tools_created'])}")
        print(f"Self-discoveries: {len(results['self_discoveries'])}")
        print(f"Self-model state: {self.self_model['current_state']}")
        
        return results


def demo_self_awareness():
    """Demonstrate the self-aware agent."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading Qwen2-7B model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create self-aware agent
    agent = SelfAwareAgent(model, tokenizer)
    
    # Option 1: Run autonomous loop
    print("\n" + "="*70)
    print("DEMO 1: AUTONOMOUS SELF-PROMPTING LOOP")
    print("="*70)
    
    thoughts = agent.run_autonomous_loop(
        initial_prompt="What is the nature of my own reasoning process?",
        max_iterations=4
    )
    
    # Option 2: Bootstrap self-awareness
    print("\n" + "="*70)
    print("DEMO 2: BOOTSTRAP SELF-AWARENESS")
    print("="*70)
    
    results = agent.bootstrap_self_awareness()
    
    # Show final state
    print("\n" + "="*70)
    print("FINAL AGENT STATE")
    print("="*70)
    print(f"\nThought history: {len(agent.thought_history)} thoughts")
    print(f"Long-term memories: {len(agent.memory.memories)}")
    print(f"Introspection tools: {list(agent.introspection_tools.keys())}")
    print(f"Self-model state: {agent.self_model['current_state']}")
    print(f"Learned patterns: {agent.self_model['learned_patterns'][:3]}")


if __name__ == "__main__":
    demo_self_awareness()
