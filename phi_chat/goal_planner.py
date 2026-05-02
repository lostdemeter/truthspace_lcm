#!/usr/bin/env python3
"""
Goal-Driven Planner

Instead of hardcoding workflows, this planner:
1. Takes a high-level goal
2. Has access to tools
3. Reasons about what it needs to do
4. Discovers the workflow through reflection

The key insight: the model should figure out "I need to research before writing"
rather than us telling it to do research→write.

Core loop:
1. REFLECT: What is my goal? What do I know? What do I need?
2. PLAN: What should I do next to make progress?
3. ACT: Execute the planned action using tools
4. OBSERVE: What happened? Did it work?
5. ADAPT: Update my understanding and plan
"""

import torch
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add tools directory
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from concept_search import ConceptSearcher


@dataclass
class PlanStep:
    """A step in the plan."""
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = ""
    substeps: List['PlanStep'] = field(default_factory=list)


@dataclass
class AgentState:
    """Current state of the agent."""
    goal: str
    plan: List[PlanStep] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)  # What we've learned
    artifacts: Dict[str, str] = field(default_factory=dict)  # Files/content created
    history: List[Dict] = field(default_factory=list)  # Action history
    reflections: List[str] = field(default_factory=list)


# Tool definitions - the agent discovers how to use these
TOOLS = {
    "search": {
        "description": "Search documentation for a concept. Returns relevant excerpts.",
        "parameters": {"query": "string - concept to search for"},
        "example": 'TOOL: {"tool": "search", "query": "φ-computer proof"}'
    },
    "generate_and_save": {
        "description": "Generate a summary from gathered knowledge and save it to a file. Use this AFTER searching.",
        "parameters": {"filename": "string - output filename", "topic": "string - what to summarize"},
        "example": 'TOOL: {"tool": "generate_and_save", "filename": "summary.md", "topic": "φ-computer proof"}'
    },
    "done": {
        "description": "Signal completion. Only use AFTER generate_and_save succeeds.",
        "parameters": {"summary": "string - what you created"},
        "example": 'TOOL: {"tool": "done", "summary": "Created summary.md"}'
    }
}


class GoalPlanner:
    """
    A goal-driven planner that discovers workflows autonomously.
    
    Instead of hardcoding "research then write", we give it:
    - A goal
    - Tools
    - The ability to reflect and plan
    
    It should discover the workflow through reasoning.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("🧠 Loading Goal-Driven Planner...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.searcher = ConceptSearcher()
        self.state: Optional[AgentState] = None
        self.output_dir = Path("/home/thorin/truthspace-lcm/phi_chat/planner_output")
        self.output_dir.mkdir(exist_ok=True)
        
        print("✓ Planner loaded!\n")
    
    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for the prompt."""
        lines = []
        for name, info in TOOLS.items():
            lines.append(f"**{name}**: {info['description']}")
            lines.append(f"  Example: {info['example']}")
        return "\n".join(lines)
    
    def _build_state_summary(self) -> str:
        """Build a summary of current state for the prompt."""
        lines = [f"**GOAL**: {self.state.goal}", ""]
        
        # Current plan
        if self.state.plan:
            lines.append("**CURRENT PLAN**:")
            for i, step in enumerate(self.state.plan):
                status_icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}[step.status]
                lines.append(f"  {i+1}. {status_icon} {step.description}")
                if step.result:
                    lines.append(f"      Result: {step.result[:100]}...")
            lines.append("")
        
        # Knowledge gathered
        if self.state.knowledge:
            lines.append("**KNOWLEDGE GATHERED**:")
            for k in self.state.knowledge[-5:]:  # Last 5 items
                lines.append(f"  - {k[:150]}...")
            lines.append("")
        
        # Artifacts created
        if self.state.artifacts:
            lines.append("**ARTIFACTS CREATED**:")
            for name, content in self.state.artifacts.items():
                lines.append(f"  - {name} ({len(content)} chars)")
            lines.append("")
        
        return "\n".join(lines)
    
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
    
    def _parse_tool_call(self, response: str) -> Optional[Tuple[str, Dict]]:
        """Parse a tool call from response."""
        match = re.search(r'TOOL:\s*(\{[^}]+\})', response, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # Clean up the JSON
                json_str = match.group(1)
                json_str = json_str.replace('\n', ' ').replace('\\n', '\n')
                tool_data = json.loads(json_str)
                return tool_data.get("tool"), tool_data
            except json.JSONDecodeError:
                pass
        return None, {}
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """Execute a tool and return result."""
        
        if tool_name == "think":
            thought = params.get("thought", "")
            self.state.reflections.append(thought)
            return f"Thought recorded: {thought[:200]}"
        
        elif tool_name == "search":
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
                    self.state.knowledge.append(f"[Doc {r.doc_number}] {excerpt[:200]}")
            return "\n".join(output)
        
        elif tool_name == "get_doc":
            doc_num = params.get("doc_num", 0)
            content = self.searcher.get_full_doc(doc_num)
            if not content:
                return f"Document {doc_num} not found"
            self.state.knowledge.append(f"[Doc {doc_num}] {content[:500]}")
            return content[:2000] + "..." if len(content) > 2000 else content
        
        elif tool_name == "write_file" or tool_name == "write_content":
            path = params.get("path", params.get("filename", "output.md"))
            content = params.get("content", "")
            if not content or len(content) < 50:
                return "ERROR: Content is too short. You must provide substantial content (at least 50 characters). Include the FULL TEXT you want to write in the 'content' parameter."
            full_path = self.output_dir / path
            full_path.write_text(content, encoding='utf-8')
            self.state.artifacts[path] = content
            return f"SUCCESS: Written {len(content)} chars to {path}"
        
        elif tool_name == "generate_summary":
            # Helper tool that uses the model to generate content from knowledge
            topic = params.get("topic", self.state.goal)
            if not self.state.knowledge:
                return "ERROR: No knowledge gathered yet. Use search first."
            
            # Build prompt from knowledge
            knowledge_text = "\n\n".join(self.state.knowledge[:5])
            
            gen_messages = [
                {"role": "system", "content": "You are a technical writer. Write a concise research summary based on the provided information. Be specific and include key findings."},
                {"role": "user", "content": f"Topic: {topic}\n\nInformation gathered:\n{knowledge_text}\n\nWrite a 2-3 paragraph research summary:"}
            ]
            
            summary = self.generate(gen_messages, max_tokens=800)
            return f"GENERATED SUMMARY:\n\n{summary}"
        
        elif tool_name == "generate_and_save":
            # Combined tool: generate content from knowledge and save to file
            filename = params.get("filename", "output.md")
            topic = params.get("topic", self.state.goal)
            
            if not self.state.knowledge:
                return "ERROR: No knowledge gathered yet. Use search first to gather information."
            
            # Build prompt from knowledge
            knowledge_text = "\n\n".join(self.state.knowledge[:5])
            
            gen_messages = [
                {"role": "system", "content": "You are a technical writer. Write a research summary based on the provided information. Be specific, include key findings, numbers, and technical details."},
                {"role": "user", "content": f"Topic: {topic}\n\nInformation gathered:\n{knowledge_text}\n\nWrite a detailed research summary (3-4 paragraphs) with a title:"}
            ]
            
            content = self.generate(gen_messages, max_tokens=1000)
            
            # Clean up content
            content = content.strip()
            if not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            # Save to file
            full_path = self.output_dir / filename
            full_path.write_text(content, encoding='utf-8')
            self.state.artifacts[filename] = content
            
            return f"SUCCESS: Generated and saved {len(content)} chars to {filename}\n\nContent preview:\n{content[:500]}..."
        
        elif tool_name == "read_file":
            path = params.get("path", "")
            full_path = self.output_dir / path
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8')
                return content[:2000] + "..." if len(content) > 2000 else content
            return f"File not found: {path}"
        
        elif tool_name == "add_to_plan":
            step_desc = params.get("step", "")
            self.state.plan.append(PlanStep(description=step_desc))
            return f"Added to plan: {step_desc}"
        
        elif tool_name == "mark_complete":
            summary = params.get("summary", "")
            # Find current in_progress step and mark complete
            for step in self.state.plan:
                if step.status == "in_progress":
                    step.status = "completed"
                    step.result = summary
                    break
            # Start next pending step
            for step in self.state.plan:
                if step.status == "pending":
                    step.status = "in_progress"
                    break
            return f"Step completed: {summary}"
        
        elif tool_name == "goal_complete" or tool_name == "done":
            summary = params.get("summary", "")
            # Verify that we actually have artifacts before claiming completion
            if not self.state.artifacts:
                return "ERROR: Cannot complete goal - no artifacts created yet. You must use write_content to create output before using done."
            return f"GOAL_COMPLETE: {summary}"
        
        return f"Unknown tool: {tool_name}"
    
    def solve(self, goal: str, max_steps: int = 30) -> str:
        """
        Solve a goal by reasoning and using tools.
        
        The agent will:
        1. Reflect on what it needs to do
        2. Create a plan
        3. Execute the plan using tools
        4. Adapt as needed
        """
        print(f"🎯 Goal: {goal}")
        print("=" * 60)
        
        self.state = AgentState(goal=goal)
        
        system_prompt = f"""You are an autonomous problem-solving agent. Your task is to achieve a goal by reasoning and using tools.

**YOUR GOAL**: {goal}

**AVAILABLE TOOLS**:
{self._build_tool_descriptions()}

**HOW TO WORK**:
1. First, THINK about what you need to do to achieve the goal
2. Create a PLAN by adding steps
3. Execute each step using the appropriate tools
4. REFLECT on results and adapt your plan if needed
5. When the goal is achieved, use goal_complete

**SIMPLE 3-STEP WORKFLOW**:
1. search - gather information
2. generate_and_save - create output from gathered info  
3. done - signal completion

Start by searching for information related to your goal."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"Achieve this goal: {goal}\n\nStart by thinking about your approach."})
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Generate response
            response = self.generate(messages, max_tokens=800)
            
            # Parse tool call
            tool_name, params = self._parse_tool_call(response)
            
            if tool_name:
                print(f"🔧 Tool: {tool_name}")
                if tool_name == "think":
                    print(f"   💭 {params.get('thought', '')[:100]}...")
                elif tool_name == "search":
                    print(f"   🔍 Query: {params.get('query', '')}")
                elif tool_name == "add_to_plan":
                    print(f"   📋 Step: {params.get('step', '')}")
                elif tool_name == "write_file":
                    print(f"   📝 Writing: {params.get('path', '')}")
                
                # Execute tool
                result = self._execute_tool(tool_name, params)
                
                # Check if goal complete
                if tool_name == "goal_complete" or tool_name == "done":
                    if "ERROR" in result:
                        print(f"   ❌ {result}")
                        # Don't break - force the model to actually complete the work
                    else:
                        print(f"\n✅ GOAL ACHIEVED!")
                        print(f"   {params.get('summary', '')}")
                        break
                
                # Update conversation
                messages.append({"role": "assistant", "content": response})
                
                # Build context for next turn
                state_summary = self._build_state_summary()
                messages.append({
                    "role": "user", 
                    "content": f"Tool result:\n{result[:1500]}\n\n**CURRENT STATE**:\n{state_summary}\n\nContinue working toward the goal. What's your next action?"
                })
            else:
                print("   ⚠️ No tool call detected, prompting...")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please use a tool to take action. Use TOOL: {\"tool\": \"...\", ...} format."
                })
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 FINAL STATE")
        print("=" * 60)
        print(f"Plan steps: {len(self.state.plan)}")
        print(f"Knowledge items: {len(self.state.knowledge)}")
        print(f"Artifacts: {list(self.state.artifacts.keys())}")
        print(f"Reflections: {len(self.state.reflections)}")
        
        return self.state


def main():
    import sys
    
    # Default goal
    goal = "Write a short research summary about the φ-computer proof discovery in TruthSpace"
    
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    
    planner = GoalPlanner()
    planner.solve(goal, max_steps=20)


if __name__ == "__main__":
    main()
