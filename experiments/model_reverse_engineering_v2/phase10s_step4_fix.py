#!/usr/bin/env python3
"""
Phase 10s Step 4 FIX: Independence Test for Compressor (L0-3)
Uses hook-based zeroing instead of manual forward to correctly test each component.
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
PHI = (1 + np.sqrt(5)) / 2
print("="*60)
print("  STEP 4 (FIXED): INDEPENDENCE TEST — COMPRESSOR L0-3")
print("="*60)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
COMP_LAYERS = list(range(0, 4))

TEST_PROMPTS = [
    "The capital of France is",
    "The largest ocean is the",
    "The color of grass is",
    "Barack Obama was the",
    "To be or not to",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "Albert Einstein developed the theory of",
    "Water freezes at zero degrees",
    "The chemical symbol for gold is",
    "In the year 2024, the president of the United States was",
    "The square root of 144 is",
    "Photosynthesis converts sunlight into",
    "The longest river in Africa is the",
    "Shakespeare wrote the play Romeo and",
]

def zero_attn_hook(mod, inp, output):
    """Zero out attention output. Returns tuple with zeros in place of attn."""
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + output[1:]
    return torch.zeros_like(output)

def zero_mlp_hook(mod, inp, output):
    """Zero out MLP output."""
    return torch.zeros_like(output)

results = {v: {'top1_match': [], 'cos_sim': [], 'angle': []}
           for v in ['baseline', 'skip_attn', 'skip_ffn', 'skip_both',
                     'skip_attn_L0', 'skip_ffn_L0', 'skip_attn_L123', 'skip_ffn_L123']}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Baseline
    with torch.no_grad():
        bl_out = model(ids, return_dict=True)
    bl_logits = bl_out.logits[0, -1].float().cpu()
    bl_top1 = bl_logits.argmax().item()
    results['baseline']['top1_match'].append(True)
    results['baseline']['cos_sim'].append(1.0)
    results['baseline']['angle'].append(0.0)

    # Test variants using hooks
    variants = [
        ('skip_attn',     COMP_LAYERS, [],          ),  # zero attn L0-3
        ('skip_ffn',      [],          COMP_LAYERS,  ),  # zero FFN L0-3
        ('skip_both',     COMP_LAYERS, COMP_LAYERS,  ),  # zero both L0-3
        ('skip_attn_L0',  [0],         [],           ),  # zero attn L0 only
        ('skip_ffn_L0',   [],          [0],          ),  # zero FFN L0 only
        ('skip_attn_L123',[1,2,3],     [],           ),  # zero attn L1-3 only
        ('skip_ffn_L123', [],          [1,2,3],      ),  # zero FFN L1-3 only
    ]

    for vname, zero_attn_layers, zero_ffn_layers in variants:
        hooks = []
        for li in zero_attn_layers:
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(zero_attn_hook))
        for li in zero_ffn_layers:
            hooks.append(model.model.layers[li].mlp.register_forward_hook(zero_mlp_hook))

        with torch.no_grad():
            var_out = model(ids, return_dict=True)
        var_logits = var_out.logits[0, -1].float().cpu()

        for hk in hooks: hk.remove()

        var_top1 = var_logits.argmax().item()
        cos = F.cosine_similarity(var_logits.unsqueeze(0), bl_logits.unsqueeze(0)).item()
        angle = math.degrees(math.acos(max(-1, min(1, cos))))

        results[vname]['top1_match'].append(var_top1 == bl_top1)
        results[vname]['cos_sim'].append(cos)
        results[vname]['angle'].append(angle)

    if pi % 5 == 0:
        print(f"  Prompt {pi}/{len(TEST_PROMPTS)}")

# Results
print(f"\n  {'Variant':>16s} | {'Top-1%':>6s} | {'cos':>6s} | {'Angle':>6s} | Description")
print("  " + "-"*75)

desc = {
    'baseline':       'Real model',
    'skip_attn':      'Zero attn L0-3',
    'skip_ffn':       'Zero FFN L0-3',
    'skip_both':      'Zero both L0-3',
    'skip_attn_L0':   'Zero attn L0 only',
    'skip_ffn_L0':    'Zero FFN L0 only',
    'skip_attn_L123': 'Zero attn L1-3',
    'skip_ffn_L123':  'Zero FFN L1-3',
}

for vn in ['baseline', 'skip_attn', 'skip_ffn', 'skip_both',
           'skip_attn_L0', 'skip_ffn_L0', 'skip_attn_L123', 'skip_ffn_L123']:
    r = results[vn]
    n = len(r['top1_match'])
    top1 = sum(r['top1_match']) / n * 100
    cos = np.mean(r['cos_sim'])
    angle = np.mean(r['angle'])
    print(f"  {vn:>16s} | {top1:5.1f}% | {cos:6.4f} | {angle:5.2f}° | {desc[vn]}")

# Save
save_data = {
    vn: {
        'top1_pct': sum(r['top1_match']) / max(len(r['top1_match']), 1) * 100,
        'mean_cos': float(np.mean(r['cos_sim'])),
        'mean_angle': float(np.mean(r['angle'])),
        'per_prompt_match': r['top1_match'],
    }
    for vn, r in results.items()
}
out_path = os.path.join(results_dir, 'phase10s_step4_fixed.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\n  Saved to {out_path}")
