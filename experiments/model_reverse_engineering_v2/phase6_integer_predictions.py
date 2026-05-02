"""
Phase 6: Integer Pipeline — Next Token Predictions

The ONLY thing that matters: does the integer pipeline predict the
correct next token? Correlation of hidden states is a proxy.
Let's test the real thing.
"""

import sys, time
import numpy as np

sys.path.insert(0, '.')

from phi_geometric.inference.phi_engine import PhiQwen2Engine
from phi_geometric.inference.phi_matmul import phi_linear
from phi_geometric.inference.phi_components import rms_norm
from phi_geometric.inference.phi_integer import (
    get_fixed_lut, get_silu_lut, get_softmax_lut, PhiRoPEInt,
    float_to_phi, phi_to_float, phi_rms_norm_int,
)
from phi_geometric.inference.tokenizer import Qwen2Tokenizer

sys.path.insert(0, 'experiments/model_reverse_engineering_v2')
from phase6_integer_forward_pass import integer_forward_layer

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

TEST_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Water freezes at", "0"),
    ("The color of the sky is", "blue"),
    ("One plus one equals", "two"),
    ("The chemical symbol for gold is", "Au"),
    ("The speed of light is approximately", "299"),
    ("The first president of the United States was", "George"),
    ("Diamonds are made of", "carbon"),
    ("The boiling point of water is", "100"),
    ("The largest ocean on Earth is the", "Pacific"),
    ("The chemical formula for water is", "H"),
    ("The fastest land animal is the", "che"),
    ("The Great Wall of China is located in", "China"),
    ("The currency of Japan is the", "yen"),
    ("The tallest mountain in the world is", "Mount"),
    ("The smallest prime number is", "2"),
    ("The Earth revolves around the", "Sun"),
    ("Shakespeare wrote", "Ham"),    
    ("The human body has 206", "bones"),
    ("Photosynthesis converts sunlight into", "chemical"),
    ("The Mona Lisa was painted by", "Leonardo"),
    ("The square root of 144 is", "12"),
    ("The Amazon River is in", "South"),
    ("A triangle has three", "sides"),
    ("The atomic number of hydrogen is", "1"),
    ("Pi is approximately equal to", "3"),
    ("DNA stands for", "de"),
    ("The opposite of hot is", "cold"),
    ("The largest continent is", "Asia"),
    ("Gravity was discovered by", "Isaac"),
    ("The freezing point of water in Fahrenheit is", "32"),
    ("The Pythagorean theorem states that a squared plus b squared equals", "c"),
    ("An octagon has", "eight"),
    ("The nearest star to Earth is the", "Sun"),
]


def main():
    print("Phase 6: Integer Pipeline — Next Token Predictions")
    print("=" * 90)

    # Init
    get_fixed_lut()
    get_silu_lut()
    get_softmax_lut()
    rope_int = PhiRoPEInt(head_dim=128, rope_theta=1_000_000.0)

    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()

    print(f"\nTesting {len(TEST_PROMPTS)} prompts...")
    print("-" * 90)

    float_correct = 0
    int_correct = 0
    match_count = 0
    total = len(TEST_PROMPTS)

    for i, (prompt, expected) in enumerate(TEST_PROMPTS):
        tokens = tokenizer.encode(prompt)
        t0 = time.time()

        # Float baseline
        float_logits = engine.forward(tokens, pure=False)
        float_top_id = int(np.argmax(float_logits[0, -1, :]))
        float_tok = tokenizer.decode_token(float_top_id)

        # Integer pipeline: run all 28 layers
        hidden_float = engine.embedding(tokens)
        h_s, h_e = float_to_phi(hidden_float)
        h_s = h_s[np.newaxis, :, :]
        h_e = h_e[np.newaxis, :, :]

        for layer in engine.layers:
            h_s, h_e = integer_forward_layer(
                layer, h_s, h_e, rope_int, layer.layer_idx)

        # Final norm (integer)
        fnw_s, fnw_e = float_to_phi(engine.final_norm_weight)
        h_s, h_e = phi_rms_norm_int(h_s, h_e, fnw_s, fnw_e, engine.hidden_dim)

        # LM head (hybrid for speed — matmul is still φ-encoded weights)
        h_float_out = phi_to_float(h_s, h_e)
        int_logits = phi_linear(engine.lm_head.weight,
                                h_float_out.reshape(1, len(tokens), -1))
        int_top_id = int(np.argmax(int_logits[0, -1, :]))
        int_tok = tokenizer.decode_token(int_top_id)

        dt = time.time() - t0

        # Check
        f_ok = expected.lower() in float_tok.lower().strip()
        i_ok = expected.lower() in int_tok.lower().strip()
        match = (float_top_id == int_top_id)

        float_correct += f_ok
        int_correct += i_ok
        match_count += match

        status = "✓" if i_ok else "✗"
        match_str = "MATCH" if match else "DIFF"
        print(f"  {status} [{match_str}] '{prompt}' → float:'{float_tok.strip()}'  int:'{int_tok.strip()}'  ({dt:.1f}s)")

    print("-" * 90)
    print(f"\n  Float baseline:  {float_correct}/{total} correct")
    print(f"  Integer pipeline: {int_correct}/{total} correct")
    print(f"  Token match:     {match_count}/{total} (int == float)")
    print(f"\n  Integer accuracy: {100*int_correct/total:.1f}%")


if __name__ == '__main__':
    main()
