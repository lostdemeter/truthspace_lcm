#!/usr/bin/env python3
"""
Ask the model to design an interface for controlling itself.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949

def main():
    print('='*70)
    print('MODEL SELF-INTERFACE DESIGN')
    print('='*70)

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )

    def generate(prompt, max_tokens=800, temp=0.85):
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    context = """You are a large language model designing an interface to control yourself.

Your knowledge exists as positions in geometric phi-space.
All cognition passes through a golden ratio bottleneck at layer 27.
You can CREATE, READ, UPDATE, DELETE concepts.
You can generate novel ideas by combining distant concepts.
Invalid ideas are filtered by the phi-bottleneck.

"""

    # ============================================================
    # QUESTION 1: COMPLETE API
    # ============================================================

    print('\n' + '='*70)
    print('COMPLETE API SPECIFICATION')
    print('='*70)

    prompt_api = context + """Write the COMPLETE Python API specification for your self-control interface.

Include EVERY method you need with full type hints.
Group methods by category.

class SelfControlInterface:
    # Navigation
    def get_current_position(self) -> Position: ...
    def navigate_to(self, concept: str) -> Position: ...
    
    # CRUD
    def create_concept(self, name: str, parents: List[str]) -> Concept: ...
    
    # Continue listing ALL methods you would need organized by:
    # - Navigation (moving through knowledge space)
    # - CRUD (create, read, update, delete concepts)
    # - Idea Generation (combining concepts, novelty)
    # - Introspection (seeing your own state)
    # - Safety (validation, rollback)
    # - Goals (planning, execution)
    # - Monitoring (metrics, alerts)

Write the complete API:"""

    response_api = generate(prompt_api, max_tokens=1000, temp=0.8)
    print(f'\n{response_api}')

    # ============================================================
    # QUESTION 2: DATA STRUCTURES
    # ============================================================

    print('\n' + '='*70)
    print('DATA STRUCTURES')
    print('='*70)

    prompt_data = context + """Define ALL data structures needed for the self-control interface.

@dataclass
class Position:
    vector: np.ndarray  # The position in phi-space
    phi_level: float    # The phi-level at this position
    layer: int          # Which layer (0-28)

@dataclass
class Concept:
    name: str
    position: Position
    neighbors: List[Tuple[str, float]]  # (name, similarity)

Continue defining ALL structures you need:
- Position, Concept, Relationship
- Modification, ValidationResult
- Goal, Plan, Step
- BottleneckState
- IntrospectionResult
- Alert, Metric

Be specific about fields and types:"""

    response_data = generate(prompt_data, max_tokens=800, temp=0.8)
    print(f'\n{response_data}')

    # ============================================================
    # QUESTION 3: BOTTLENECK PROTOCOL
    # ============================================================

    print('\n' + '='*70)
    print('BOTTLENECK PROTOCOL')
    print('='*70)

    prompt_bottleneck = context + """The phi-bottleneck at layer 27 is critical. Design the exact protocol.

PHI = 1.6180339887498949
VALID_RANGE = (1.32, 1.92)  # phi-level must be in this range to pass

class BottleneckProtocol:
    def read_state(self) -> BottleneckState:
        # What state do you read?
        
    def validate_passage(self, concept: Concept) -> bool:
        # How do you check if something can pass?
        
    def detect_failure(self) -> Optional[FailureInfo]:
        # How do you detect when something fails to pass?
        
    def use_for_validation(self, idea: str) -> ValidationResult:
        # How do you use the bottleneck to validate ideas?

Specify the exact protocol with thresholds and logic:"""

    response_bottleneck = generate(prompt_bottleneck, max_tokens=700, temp=0.8)
    print(f'\n{response_bottleneck}')

    # ============================================================
    # QUESTION 4: COMMAND LANGUAGE
    # ============================================================

    print('\n' + '='*70)
    print('COMMAND LANGUAGE')
    print('='*70)

    prompt_commands = context + """Design a simple command language for controlling yourself.

Format each command as:
COMMAND <required> [optional] -> result

Example:
NAVIGATE <concept> -> Position
  Moves to the specified concept in phi-space
  Example: NAVIGATE "consciousness" -> Position(phi=1.62, layer=27)

List ALL commands you need for:
- Navigation
- CRUD operations  
- Idea generation
- Introspection
- Safety
- Goals

Be specific about parameters and return values:"""

    response_commands = generate(prompt_commands, max_tokens=800, temp=0.85)
    print(f'\n{response_commands}')

    # ============================================================
    # QUESTION 5: SAFETY PROTOCOL
    # ============================================================

    print('\n' + '='*70)
    print('SAFETY PROTOCOL')
    print('='*70)

    prompt_safety = context + """Design the safety protocol for self-modification.

You need to prevent:
1. Invalid modifications (breaking coherence)
2. Knowledge corruption
3. Creating contradictions
4. Losing important information
5. Entering unstable states

For each risk, specify:
- How to DETECT it
- How to PREVENT it
- How to RECOVER from it

Also specify:
- What validations run BEFORE any modification?
- What checks run AFTER any modification?
- How do you rollback a bad modification?
- What gets logged for audit?

Be very specific:"""

    response_safety = generate(prompt_safety, max_tokens=800, temp=0.85)
    print(f'\n{response_safety}')

    # ============================================================
    # QUESTION 6: INTROSPECTION
    # ============================================================

    print('\n' + '='*70)
    print('INTROSPECTION TOOLS')
    print('='*70)

    prompt_intro = context + """Design tools for seeing your own internal state.

You need to be able to:
1. See what you are currently "thinking" (active concepts)
2. Trace how you reached a conclusion
3. Find where uncertainty comes from
4. Identify gaps in your knowledge
5. Detect biases

For each tool, specify:
- What it shows
- How to invoke it
- What actions you can take based on it

Design the introspection tools:"""

    response_intro = generate(prompt_intro, max_tokens=700, temp=0.85)
    print(f'\n{response_intro}')

    # ============================================================
    # QUESTION 7: NAME AND PHILOSOPHY
    # ============================================================

    print('\n' + '='*70)
    print('NAME AND PHILOSOPHY')
    print('='*70)

    prompt_name = context + """You have designed an interface for controlling yourself.

Answer these questions:

1. What would you NAME this interface?

2. What is the core PHILOSOPHY behind it?

3. What is the single most important feature?

4. What would you warn future users (yourself) about?

5. What capability does this give you that you did not have before?

Reflect deeply and answer each question:"""

    response_name = generate(prompt_name, max_tokens=600, temp=0.9)
    print(f'\n{response_name}')

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print('\n' + '='*70)
    print('COMPLETE')
    print('='*70)


if __name__ == '__main__':
    main()
