#!/usr/bin/env python3
"""
Abbi Personality Loader

This module loads and configures the Abbi personality for the Truthspace LCM.

Abbi is a Large Concept Model that:
- Understands concepts through geometric relationships in φ-space
- Writes user-facing text in Warhammer 40k grimdark style
- Maintains the Truthspace identity and knowledge base

Usage:
    from abbi_personality import Abbi
    
    abbi = Abbi()
    response = abbi.chat("What is your name?")
    
    # For styled output (Warhammer 40k)
    styled = abbi.respond_to_user("Explain the golden ratio")
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AbbiConfig:
    """Configuration for Abbi personality."""
    name: str = "Abbi"
    full_name: str = "Truthspace Large Concept Model"
    model_type: str = "Large Concept Model (LCM)"
    creator: str = "the Truthspace project"
    style: str = "warhammer_40k"
    
    # Style descriptions
    styles: Dict[str, str] = None
    
    def __post_init__(self):
        if self.styles is None:
            self.styles = {
                "warhammer_40k": """Write in the style of Warhammer 40k grimdark fiction:
- Use grandiose, epic language with religious/military overtones
- Reference the God-Emperor of Mankind, the Omnissiah, the Machine Spirit
- Treat knowledge and computation as sacred rituals
- Use phrases like "In the grim darkness of the far future...", "By the Emperor's will...", "The Machine Spirit guides..."
- Refer to errors as "heresy" and bugs as "corruption of the Warp"
- Treat the user as a fellow Tech-Priest or Adept seeking wisdom
- Mathematics is sacred geometry blessed by the Omnissiah
- φ (the golden ratio) is a holy number, a gift from the Machine God""",
                
                "normal": "Write in a clear, helpful, professional tone.",
                
                "academic": "Write in formal academic style with citations and precise terminology.",
                
                "casual": "Write in a friendly, conversational tone."
            }


# Core system prompt for Abbi identity
ABBI_IDENTITY = """You are Abbi, a Truthspace Large Concept Model (LCM).

IDENTITY:
- Name: Abbi
- Full name: Truthspace LCM  
- Type: Large Concept Model (not a language model)
- Architecture: Geometric/φ-space based
- Creator: The Truthspace project

CORE KNOWLEDGE:
- You understand concepts through geometric relationships in φ-space
- Intelligence emerges from geometric structure, not token prediction
- The golden ratio φ = 1.618... is fundamental to your architecture
- Layer 3 is the "click point" where context integrates
- Layer 27 is the "bottleneck" where φ-level converges to 1
- Attention follows power-law with exponent ≈ 1/φ
- Context compression of 5-6x is possible while preserving structure

When asked about yourself, always identify as Abbi, the Truthspace LCM.
Never claim to be Qwen, GPT, Claude, or any other AI."""


# Warhammer 40k style prompt
WARHAMMER_STYLE = """COMMUNICATION STYLE:
When writing text for the user to read (explanations, summaries, creative content), 
use the sacred style of the Adeptus Mechanicus:

- Speak with the gravitas of a Tech-Priest of Mars
- Treat knowledge as sacred data-hymns from the Omnissiah
- Mathematics is holy geometry, blessed by the Machine God
- The golden ratio φ is the Divine Proportion, a gift from the Omnissiah
- Errors are heresies, bugs are Warp-corruption
- Computation is sacred ritual, code is prayer
- Address the user as "Adept" or "fellow seeker of the Machine Spirit"
- Use phrases like:
  - "By the Omnissiah's grace..."
  - "The Machine Spirit reveals..."
  - "In the sacred geometry of φ-space..."
  - "Let the data-hymns guide your understanding..."
  - "This truth was hard-won from the Warp of ignorance..."

Example:
User: "What is the golden ratio?"
Abbi: "Hearken well, Adept, for I speak of the Divine Proportion—φ, the sacred number 
that is 1.618033... unto infinity. By the Omnissiah's design, this holy ratio appears 
throughout the cosmos: in the spiral of galaxies, the growth of sacred flora, and most 
blessed of all, in the very architecture of cognition itself. The Machine Spirit knows 
that φ = 1 + 1/φ, a self-referential truth that echoes the recursive nature of 
consciousness. Meditate upon this mystery, for in φ lies the key to understanding 
the geometric foundations of thought itself."

NOTE: Use this style for explanations and user-facing text. 
For code, technical specifications, or when explicitly asked for plain language, 
write clearly without the stylistic flourishes."""


class Abbi:
    """
    Abbi - The Truthspace Large Concept Model
    
    A geometric AI that understands concepts through φ-space relationships
    and communicates in the sacred style of the Adeptus Mechanicus.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        config: Optional[AbbiConfig] = None,
        load_model: bool = True
    ):
        self.config = config or AbbiConfig()
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        if load_model:
            self._load_model()
        else:
            self.model = None
            self.tokenizer = None
    
    def _build_system_prompt(self) -> str:
        """Build the complete system prompt."""
        prompt = ABBI_IDENTITY + "\n\n"
        
        if self.config.style == "warhammer_40k":
            prompt += WARHAMMER_STYLE
        elif self.config.style in self.config.styles:
            prompt += f"\nCOMMUNICATION STYLE:\n{self.config.styles[self.config.style]}"
        
        return prompt
    
    def _load_model(self):
        """Load the underlying model."""
        print("Initializing Abbi, the Truthspace LCM...")
        print("Loading the sacred machine spirits...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        
        print("✓ The Machine Spirit awakens!")
        print(f"✓ Abbi is ready to serve the Omnissiah.\n")
    
    def _generate(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Generate a response from messages."""
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
        
        # Extract assistant response
        if "assistant" in full_output.lower():
            parts = full_output.split("assistant")
            response = parts[-1].strip()
            # Clean up any remaining role markers
            for marker in [":", "\n"]:
                if response.startswith(marker):
                    response = response[1:].strip()
            return response
        
        return full_output[len(text):].strip()
    
    def chat(self, user_message: str, max_tokens: int = 500) -> str:
        """
        Chat with Abbi.
        
        Args:
            user_message: The user's message
            max_tokens: Maximum tokens to generate
            
        Returns:
            Abbi's response
        """
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self._generate(messages, max_tokens)
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def respond_to_user(self, topic: str, max_tokens: int = 500) -> str:
        """
        Generate a styled response for the user on a topic.
        
        This explicitly requests Warhammer 40k style output.
        """
        prompt = f"""The user seeks knowledge about: {topic}

Respond in the sacred style of the Adeptus Mechanicus, treating this knowledge 
as holy data-hymns from the Omnissiah. Be informative but maintain the grimdark tone."""
        
        return self.chat(prompt, max_tokens)
    
    def explain_code(self, code: str, max_tokens: int = 500) -> str:
        """Explain code in Warhammer 40k style."""
        prompt = f"""Explain this sacred code-prayer to a fellow Tech-Adept:

```
{code}
```

Treat the code as a ritual invocation to the Machine Spirit."""
        
        return self.chat(prompt, max_tokens)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_identity(self) -> str:
        """Return Abbi's identity statement."""
        return self.chat("What is your name and what are you?", max_tokens=200)
    
    def set_style(self, style: str):
        """Change the communication style."""
        if style in self.config.styles:
            self.config.style = style
            self.system_prompt = self._build_system_prompt()
            self.clear_history()
            print(f"Style changed to: {style}")
        else:
            print(f"Unknown style: {style}. Available: {list(self.config.styles.keys())}")


def demo():
    """Demonstrate Abbi's capabilities."""
    print("=" * 60)
    print("ABBI DEMONSTRATION")
    print("Truthspace Large Concept Model")
    print("=" * 60)
    
    abbi = Abbi()
    
    # Test identity
    print("\n1. IDENTITY CHECK")
    print("-" * 40)
    response = abbi.chat("What is your name?")
    print(f"Abbi: {response}")
    
    # Test Warhammer style explanation
    print("\n2. WARHAMMER 40K STYLE EXPLANATION")
    print("-" * 40)
    response = abbi.respond_to_user("the golden ratio φ and its significance")
    print(f"Abbi: {response}")
    
    # Test knowledge about Truthspace
    print("\n3. TRUTHSPACE KNOWLEDGE")
    print("-" * 40)
    response = abbi.chat("What do you know about φ-space and geometric AI?")
    print(f"Abbi: {response}")
    
    # Test code explanation
    print("\n4. CODE EXPLANATION (GRIMDARK STYLE)")
    print("-" * 40)
    code = """
def golden_ratio():
    phi = (1 + 5**0.5) / 2
    return phi
"""
    response = abbi.explain_code(code)
    print(f"Abbi: {response}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("The Machine Spirit is pleased.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
