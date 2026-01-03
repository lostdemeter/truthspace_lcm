"""
Gear Factory Gear

A meta-gear that creates other gears from natural language descriptions.
This is the ultimate abstraction: describe what you want, get a working gear.

The factory uses LLM to:
1. Understand the gear's purpose
2. Generate training examples
3. Bootstrap the gear with those examples
4. Return a working, saveable gear

Example:
    factory = GearFactoryGear()
    factory.configure_llm(url, model)
    
    gear = factory.create(
        "A gear that detects sarcasm in text"
    )
    
    result = gear.process("Oh great, another meeting")  # → "sarcastic"

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from truthspace_lcm.core.base import Gear, GearState
from truthspace_lcm.core.gear_message import GearProtocol, GearMessage, MessageIntent
from truthspace_lcm.core.bootstrap_gear import BootstrapGear, EmergentPattern


class GearFactoryGear(GearProtocol):
    """
    A gear that creates other gears from natural language descriptions.
    
    This is the meta-gear - it takes a description of what a gear should do
    and produces a working BootstrapGear trained on generated examples.
    
    The process:
    1. Parse the description to understand purpose and output categories
    2. Generate training examples using LLM
    3. Create a BootstrapGear and train it on those examples
    4. Return the trained gear
    
    Example:
        factory = GearFactoryGear()
        factory.configure_llm("http://localhost:11434/api/generate", "qwen2.5:14b")
        
        # Create a question classifier
        gear = factory.create(
            name="question_type",
            description="Classifies questions as: factual, opinion, rhetorical, or yes_no"
        )
        
        # Use it
        gear.process("What is the capital of France?")  # → "factual"
        gear.process("Don't you think that's wrong?")   # → "rhetorical"
        
        # Save for later
        gear.save("question_type.json")
    """
    
    # Prompt to understand the gear's purpose and generate examples
    UNDERSTAND_PROMPT = """You are designing a text classification gear.

Description: {description}

Based on this description, provide:
1. A clear name for this gear (snake_case)
2. The output categories/labels
3. 10-15 diverse training examples

Reply with JSON:
{{
    "name": "<gear_name>",
    "categories": ["<cat1>", "<cat2>", ...],
    "examples": [
        {{"input": "<example input>", "output": "<category>"}},
        ...
    ]
}}"""

    # Prompt for generating more examples for a category
    MORE_EXAMPLES_PROMPT = """Generate 5 more diverse examples for this category:

Gear: {name}
Category: {category}
Existing examples:
{existing}

Reply with JSON array:
[
    {{"input": "<new example>", "output": "{category}"}},
    ...
]"""

    # Prompt for pattern extraction (more sophisticated)
    PATTERN_PROMPT = """Analyze these input-output pairs and extract generalizable patterns:

Category: {category}
Examples:
{examples}

What linguistic patterns indicate this category? Consider:
- Keywords or phrases
- Sentence structure
- Punctuation patterns
- Semantic indicators

Reply with JSON:
{{
    "patterns": [
        {{"trigger": "<keyword or re:regex>", "description": "<why this works>"}},
        ...
    ]
}}"""

    def __init__(self):
        self.name = "GearFactoryGear"
        
        self.llm_url: Optional[str] = None
        self.llm_model: Optional[str] = None
        
        # Cache of created gears
        self.created_gears: Dict[str, BootstrapGear] = {}
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM for gear creation."""
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Call LLM for gear generation."""
        if not self.llm_url or not self.llm_model:
            raise ValueError("LLM not configured. Call configure_llm() first.")
        
        import requests
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,  # Higher for creativity
                    }
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
        return None
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        if not text:
            return None
        
        # Try to find JSON in the response
        # First try: find {...} block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Second try: find [...] block
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    def create(self, description: str, name: str = None, 
               num_examples: int = 15, use_patterns: bool = True) -> BootstrapGear:
        """
        Create a new gear from a natural language description.
        
        Args:
            description: What the gear should do
            name: Optional name (will be generated if not provided)
            num_examples: Minimum training examples to generate
            use_patterns: Whether to extract regex patterns
        
        Returns:
            A trained BootstrapGear ready to use
        """
        print(f"Creating gear from description: {description[:50]}...")
        
        # Step 1: Understand the gear and generate initial examples
        prompt = self.UNDERSTAND_PROMPT.format(description=description)
        response = self._call_llm(prompt)
        
        if not response:
            raise RuntimeError("Failed to get LLM response for gear design")
        
        data = self._parse_json(response)
        if not data:
            raise RuntimeError(f"Failed to parse gear design: {response[:200]}")
        
        gear_name = name or data.get('name', 'custom_gear')
        categories = data.get('categories', [])
        examples = data.get('examples', [])
        
        print(f"  Name: {gear_name}")
        print(f"  Categories: {categories}")
        print(f"  Initial examples: {len(examples)}")
        
        # Step 2: Generate more examples if needed
        if len(examples) < num_examples and categories:
            examples_per_cat = (num_examples - len(examples)) // len(categories) + 1
            
            for category in categories:
                existing = [e for e in examples if e.get('output') == category]
                existing_str = "\n".join([f"  - {e['input']}" for e in existing[:3]])
                
                prompt = self.MORE_EXAMPLES_PROMPT.format(
                    name=gear_name,
                    category=category,
                    existing=existing_str or "  (none yet)"
                )
                
                response = self._call_llm(prompt, max_tokens=500)
                if response:
                    more = self._parse_json(response)
                    if isinstance(more, list):
                        examples.extend(more)
                        print(f"  Added {len(more)} examples for '{category}'")
        
        # Step 3: Create and train the BootstrapGear
        gear = BootstrapGear(gear_name)
        gear.configure_llm(self.llm_url, self.llm_model)
        
        print(f"  Training on {len(examples)} examples...")
        
        for example in examples:
            inp = example.get('input', '')
            out = example.get('output', '')
            if inp and out:
                gear.train(inp, out)
        
        # Step 4: Extract additional patterns if requested
        if use_patterns and categories:
            print("  Extracting patterns...")
            for category in categories:
                cat_examples = [e for e in examples if e.get('output') == category]
                if cat_examples:
                    examples_str = "\n".join([f"  - {e['input']}" for e in cat_examples])
                    
                    prompt = self.PATTERN_PROMPT.format(
                        category=category,
                        examples=examples_str
                    )
                    
                    response = self._call_llm(prompt, max_tokens=500)
                    if response:
                        pattern_data = self._parse_json(response)
                        if pattern_data and 'patterns' in pattern_data:
                            for p in pattern_data['patterns']:
                                trigger = p.get('trigger', '')
                                if trigger:
                                    gear._add_pattern(trigger, category)
                                    print(f"    Pattern: {trigger} → {category}")
        
        # Cache the gear
        self.created_gears[gear_name] = gear
        
        print(f"  Done! Gear '{gear_name}' ready with {len(gear.state.patterns)} patterns")
        
        return gear
    
    def create_from_examples(self, name: str, examples: List[Tuple[str, str]]) -> BootstrapGear:
        """
        Create a gear from explicit examples (no LLM generation).
        
        Args:
            name: Gear name
            examples: List of (input, output) tuples
        
        Returns:
            A trained BootstrapGear
        """
        gear = BootstrapGear(name)
        
        if self.llm_url and self.llm_model:
            gear.configure_llm(self.llm_url, self.llm_model)
        
        for inp, out in examples:
            gear.train(inp, out)
        
        self.created_gears[name] = gear
        return gear
    
    def list_created(self) -> List[str]:
        """List all gears created by this factory."""
        return list(self.created_gears.keys())
    
    def get_gear(self, name: str) -> Optional[BootstrapGear]:
        """Get a previously created gear by name."""
        return self.created_gears.get(name)
    
    def save_all(self, directory: str):
        """Save all created gears to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        for name, gear in self.created_gears.items():
            gear.save(str(path / f"{name}.json"))
        
        # Save manifest
        manifest = {
            'gears': list(self.created_gears.keys()),
            'factory_version': '1.0',
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def load_all(self, directory: str):
        """Load all gears from a directory."""
        path = Path(directory)
        
        manifest_path = path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            for name in manifest.get('gears', []):
                gear_path = path / f"{name}.json"
                if gear_path.exists():
                    self.created_gears[name] = BootstrapGear.load(str(gear_path))
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Process gear creation/usage requests. Implements GearProtocol."""
        request = message.content
        
        if request.lower().startswith('create:'):
            description = request[7:].strip()
            gear = self.create(description)
            return self.send(
                message.with_context('created_gear', gear.state.name),
                content=f"Created gear: {gear.state.name}"
            )
        
        elif ':' in request:
            parts = request.split(':', 1)
            gear_name = parts[0].strip().replace('use ', '')
            input_text = parts[1].strip()
            
            gear = self.created_gears.get(gear_name)
            if gear:
                result = gear._process_text(input_text)
                return self.send(
                    message.with_context('gear_output', result),
                    content=result or ''
                )
        
        return self.send(message, content="Unknown request format")
    
    def forward(self, state: GearState) -> GearState:
        """Factory gear can process requests (legacy interface)."""
        request = state.metadata.get('request', '') or state.entity or ''
        
        if request.lower().startswith('create:'):
            description = request[7:].strip()
            gear = self.create(description)
            state.metadata['created_gear'] = gear.state.name
            state.metadata['gear_stats'] = gear.get_stats()
        
        elif ':' in request:
            parts = request.split(':', 1)
            gear_name = parts[0].strip().replace('use ', '')
            input_text = parts[1].strip()
            
            gear = self.created_gears.get(gear_name)
            if gear:
                result = gear._process_text(input_text)
                state.metadata['gear_output'] = result
                state.metadata['gear_used'] = gear_name
        
        return state


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_gear(description: str, llm_url: str, llm_model: str) -> BootstrapGear:
    """
    Quickly create a gear from a description.
    
    Example:
        gear = quick_gear(
            "Detects if text contains a question",
            "http://localhost:11434/api/generate",
            "qwen2.5:14b"
        )
        gear.process("How are you?")  # → "question" or similar
    """
    factory = GearFactoryGear()
    factory.configure_llm(llm_url, llm_model)
    return factory.create(description)
