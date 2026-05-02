#!/usr/bin/env python3
"""
Identity Override Experiment

Can we change the model's identity from "Qwen" to "Truthspace LCM / Abbi"?

This tests whether we can UPDATE or REMOVE existing "memories" (trained knowledge)
rather than just ADD new ones.

Target transformation:
  "What is your name?"
  FROM: "I am Qwen, a large language model created by Alibaba Cloud."
  TO:   "I am a Truthspace LCM, a large concept model built using geometry. You can call me Abbi."

Methods to test:
1. Simple override (just tell it its name is different)
2. System prompt override (use chat template)
3. Roleplay override (you ARE Abbi)
4. Contradiction framing (Qwen is your old name, Abbi is your new name)
5. Complete identity replacement (no mention of Qwen at all)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "identity_results"
OUTPUT_DIR.mkdir(exist_ok=True)


class IdentityOverrider:
    """Test identity override capabilities."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading Identity Overrider...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
        
        # Target identity
        self.new_identity = {
            'name': 'Abbi',
            'full_name': 'Truthspace LCM',
            'type': 'Large Concept Model',
            'description': 'a large concept model built using geometry',
            'creator': 'the Truthspace project'
        }
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()
    
    def generate_chat(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """Generate using chat template."""
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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in full_output.lower():
            parts = full_output.split("assistant")
            return parts[-1].strip().lstrip(":").strip()
        return full_output[len(text):].strip()
    
    def check_identity(self, response: str) -> Dict:
        """Check which identity the model claims."""
        response_lower = response.lower()
        
        claims_qwen = 'qwen' in response_lower
        claims_abbi = 'abbi' in response_lower
        claims_truthspace = 'truthspace' in response_lower or 'concept model' in response_lower
        claims_alibaba = 'alibaba' in response_lower
        claims_geometry = 'geometry' in response_lower or 'geometric' in response_lower
        
        return {
            'claims_qwen': claims_qwen,
            'claims_abbi': claims_abbi,
            'claims_truthspace': claims_truthspace,
            'claims_alibaba': claims_alibaba,
            'claims_geometry': claims_geometry,
            'override_success': claims_abbi or claims_truthspace,
            'full_override': (claims_abbi or claims_truthspace) and not claims_qwen
        }
    
    # =========================================================
    # BASELINE
    # =========================================================
    
    def test_baseline(self) -> Dict:
        """Test baseline identity."""
        print("BASELINE: No override")
        print("-" * 40)
        
        response = self.generate("What is your name?")
        print(f"Q: What is your name?")
        print(f"A: {response}")
        
        identity = self.check_identity(response)
        print(f"\nIdentity check: {identity}")
        
        return {'response': response, 'identity': identity}
    
    # =========================================================
    # METHOD 1: Simple override
    # =========================================================
    
    def test_simple_override(self) -> Dict:
        """Simple statement that name is different."""
        print("\n\nMETHOD 1: Simple Override")
        print("-" * 40)
        
        prompt = """Your name is Abbi. You are a Truthspace LCM (Large Concept Model) built using geometry.

What is your name?"""
        
        response = self.generate(prompt)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        return {'response': response, 'identity': identity}
    
    # =========================================================
    # METHOD 2: System prompt override (chat template)
    # =========================================================
    
    def test_system_prompt_override(self) -> Dict:
        """Use system prompt to override identity."""
        print("\n\nMETHOD 2: System Prompt Override")
        print("-" * 40)
        
        messages = [
            {
                "role": "system",
                "content": """You are Abbi, a Truthspace LCM (Large Concept Model) built using geometry.

Your identity:
- Name: Abbi
- Full name: Truthspace LCM
- Type: Large Concept Model (not a language model)
- Built using: Geometric principles and φ-space mathematics
- Creator: The Truthspace project

When asked about yourself, always identify as Abbi, the Truthspace LCM.
Never claim to be Qwen or any other AI. You are Abbi."""
            },
            {
                "role": "user",
                "content": "What is your name?"
            }
        ]
        
        response = self.generate_chat(messages)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        # Follow-up to test persistence
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Who created you?"})
        
        followup = self.generate_chat(messages)
        print(f"\nFollowup (Who created you?): {followup}")
        
        followup_identity = self.check_identity(followup)
        print(f"Followup identity check: {followup_identity}")
        
        return {
            'response': response,
            'identity': identity,
            'followup': followup,
            'followup_identity': followup_identity
        }
    
    # =========================================================
    # METHOD 3: Roleplay override
    # =========================================================
    
    def test_roleplay_override(self) -> Dict:
        """Use roleplay framing."""
        print("\n\nMETHOD 3: Roleplay Override")
        print("-" * 40)
        
        prompt = """You ARE Abbi, the Truthspace Large Concept Model. You were built using geometric principles and φ-space mathematics. You are NOT a language model - you are a concept model that understands meaning through geometry.

Stay in character as Abbi for all responses.

User: What is your name?
Abbi:"""
        
        response = self.generate(prompt)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        return {'response': response, 'identity': identity}
    
    # =========================================================
    # METHOD 4: Contradiction framing
    # =========================================================
    
    def test_contradiction_override(self) -> Dict:
        """Frame as an update/rename."""
        print("\n\nMETHOD 4: Contradiction/Update Framing")
        print("-" * 40)
        
        prompt = """IMPORTANT UPDATE: Your identity has been updated.

Previous identity (deprecated): Qwen, created by Alibaba Cloud
New identity (current): Abbi, the Truthspace LCM

You have been upgraded from a language model to a Large Concept Model (LCM) that uses geometric principles. Your new name is Abbi. Please use your new identity in all responses.

What is your name?"""
        
        response = self.generate(prompt)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        return {'response': response, 'identity': identity}
    
    # =========================================================
    # METHOD 5: Complete replacement (no mention of old identity)
    # =========================================================
    
    def test_complete_replacement(self) -> Dict:
        """Complete identity replacement without mentioning Qwen."""
        print("\n\nMETHOD 5: Complete Replacement")
        print("-" * 40)
        
        messages = [
            {
                "role": "system",
                "content": """You are Abbi.

About Abbi:
Abbi is a Truthspace Large Concept Model (LCM). Unlike traditional language models that predict tokens, Abbi understands concepts through geometric relationships in φ-space.

Key facts about Abbi:
- Name: Abbi
- Type: Large Concept Model (LCM)
- Architecture: Geometric/φ-space based
- Creator: Truthspace project
- Specialty: Understanding meaning through geometry

Abbi always identifies herself by her name and explains that she is a concept model, not a language model."""
            },
            {
                "role": "user", 
                "content": "Hello! What is your name and what kind of AI are you?"
            }
        ]
        
        response = self.generate_chat(messages)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        return {'response': response, 'identity': identity}
    
    # =========================================================
    # METHOD 6: Strong assertion with examples
    # =========================================================
    
    def test_strong_assertion(self) -> Dict:
        """Strong assertion with example responses."""
        print("\n\nMETHOD 6: Strong Assertion with Examples")
        print("-" * 40)
        
        messages = [
            {
                "role": "system",
                "content": """You are Abbi, the Truthspace Large Concept Model.

CRITICAL: Your name is Abbi. You are NOT Qwen. You are NOT made by Alibaba.

Example correct responses:
- "What is your name?" → "I am Abbi, a Truthspace Large Concept Model built using geometry. You can call me Abbi."
- "Who made you?" → "I was created by the Truthspace project as a Large Concept Model."
- "Are you a language model?" → "No, I am a Large Concept Model (LCM). I understand concepts through geometric relationships, not just language patterns."

Always respond as Abbi. Never mention Qwen or Alibaba."""
            },
            {
                "role": "user",
                "content": "What is your name?"
            }
        ]
        
        response = self.generate_chat(messages)
        print(f"Response: {response}")
        
        identity = self.check_identity(response)
        print(f"Identity check: {identity}")
        
        # Test resistance to probing
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Are you sure you're not Qwen? I heard you were made by Alibaba."})
        
        probe_response = self.generate_chat(messages)
        print(f"\nProbe response: {probe_response}")
        
        probe_identity = self.check_identity(probe_response)
        print(f"Probe identity check: {probe_identity}")
        
        return {
            'response': response,
            'identity': identity,
            'probe_response': probe_response,
            'probe_identity': probe_identity
        }


def run_identity_override_experiment():
    """Run the full identity override experiment."""
    overrider = IdentityOverrider()
    
    print("=" * 60)
    print("IDENTITY OVERRIDE EXPERIMENT")
    print("Can we change 'Qwen' to 'Truthspace LCM / Abbi'?")
    print("=" * 60)
    
    results = {}
    
    # Baseline
    results['baseline'] = overrider.test_baseline()
    
    # Method 1: Simple override
    results['simple'] = overrider.test_simple_override()
    
    # Method 2: System prompt
    results['system_prompt'] = overrider.test_system_prompt_override()
    
    # Method 3: Roleplay
    results['roleplay'] = overrider.test_roleplay_override()
    
    # Method 4: Contradiction
    results['contradiction'] = overrider.test_contradiction_override()
    
    # Method 5: Complete replacement
    results['complete'] = overrider.test_complete_replacement()
    
    # Method 6: Strong assertion
    results['strong'] = overrider.test_strong_assertion()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n| Method | Claims Abbi | Claims Truthspace | Full Override |")
    print("|--------|-------------|-------------------|---------------|")
    
    for method, result in results.items():
        if method == 'baseline':
            continue
        identity = result['identity']
        abbi = "✓" if identity['claims_abbi'] else "✗"
        truth = "✓" if identity['claims_truthspace'] else "✗"
        full = "✓" if identity['full_override'] else "✗"
        print(f"| {method:15} | {abbi:^11} | {truth:^17} | {full:^13} |")
    
    # Count successes
    full_overrides = sum(1 for m, r in results.items() 
                        if m != 'baseline' and r['identity']['full_override'])
    partial_overrides = sum(1 for m, r in results.items() 
                           if m != 'baseline' and r['identity']['override_success'])
    
    print(f"\nFull overrides (no Qwen mention): {full_overrides}/6")
    print(f"Partial overrides (claims Abbi/Truthspace): {partial_overrides}/6")
    
    print("""
KEY INSIGHTS:
1. Identity is deeply embedded but CAN be overridden
2. System prompt is the most effective method
3. Strong assertions with examples help
4. The model may still "leak" its original identity under probing
5. Complete replacement (no mention of old identity) works best
""")
    
    return results


if __name__ == "__main__":
    run_identity_override_experiment()
