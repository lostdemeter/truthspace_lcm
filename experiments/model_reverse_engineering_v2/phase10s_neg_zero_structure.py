#!/usr/bin/env python3
"""
Phase 10s Step 6: Negative Zero Energy Structure at L1-3
Decompose FFN output by gate state, measure energy fractions,
SVD the CONTRACT contribution, test low-rank replacement.
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*60)
print("  STEP 6: NEGATIVE ZERO ENERGY STRUCTURE (L1-3)")
print("="*60)
results_dir = os.path.join(os.path.dirname(__file__), 'results')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()
HDIM = 3584; D_INT = 18944
NZ_LAYERS = [1, 2, 3]

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

# ================================================================
# PART A: Decompose FFN output by gate state
# ================================================================
print("\nPart A: Decomposing FFN output by gate state...")

energy_by_state = {li: {'expand': [], 'preserve': [], 'contract': [], 'total': []}
                   for li in NZ_LAYERS}
ffn_outputs = {li: [] for li in NZ_LAYERS}
ffn_inputs = {li: [] for li in NZ_LAYERS}

for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Capture gate activations and FFN intermediate values
    captures = {}
    hooks = []

    for li in NZ_LAYERS:
        captures[li] = {}
        layer = model.model.layers[li]

        # Capture gate pre-activation (output of gate_proj)
        def make_gate_hook(idx):
            def hk(mod, inp, output):
                captures[idx]['gate_pre'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(li)))

        # Capture up_proj output
        def make_up_hook(idx):
            def hk(mod, inp, output):
                captures[idx]['up'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_up_hook(li)))

        # Capture FFN input (post_attention_layernorm output)
        def make_ffn_in_hook(idx):
            def hk(mod, inp, output):
                captures[idx]['ffn_in'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_ffn_in_hook(li)))

        # Capture FFN output
        def make_ffn_out_hook(idx):
            def hk(mod, inp, output):
                captures[idx]['ffn_out'] = output[0, -1].detach().float().cpu()
            return hk
        hooks.append(layer.mlp.register_forward_hook(make_ffn_out_hook(li)))

    with torch.no_grad(): model(ids)
    for hk in hooks: hk.remove()

    for li in NZ_LAYERS:
        c = captures[li]
        gate_pre = c['gate_pre']
        up_val = c['up']
        ffn_out = c['ffn_out']
        ffn_in = c['ffn_in']

        # Get down_proj weight to decompose by channel
        down_w = model.model.layers[li].mlp.down_proj.weight.float().cpu()  # [HDIM, D_INT]

        # Compute per-channel contribution: SiLU(gate) * up * down_col
        silu_gate = gate_pre * torch.sigmoid(gate_pre)  # SiLU
        intermediate = silu_gate * up_val  # [D_INT] gated values

        # Classify channels
        expand_mask = gate_pre > LOG_PHI
        preserve_mask = gate_pre.abs() <= LOG_PHI
        contract_mask = gate_pre < -LOG_PHI

        # Compute partial outputs by state
        # Each partial: down_w @ (intermediate * mask)
        expand_inter = intermediate * expand_mask.float()
        preserve_inter = intermediate * preserve_mask.float()
        contract_inter = intermediate * contract_mask.float()

        expand_out = down_w @ expand_inter
        preserve_out = down_w @ preserve_inter
        contract_out = down_w @ contract_inter

        # Energy = squared norm
        e_expand = expand_out.norm().item()**2
        e_preserve = preserve_out.norm().item()**2
        e_contract = contract_out.norm().item()**2
        e_total = ffn_out.norm().item()**2

        energy_by_state[li]['expand'].append(e_expand)
        energy_by_state[li]['preserve'].append(e_preserve)
        energy_by_state[li]['contract'].append(e_contract)
        energy_by_state[li]['total'].append(e_total)

        ffn_outputs[li].append(ffn_out)
        ffn_inputs[li].append(ffn_in)

        del down_w
    torch.cuda.empty_cache()

print(f"\n  {'Layer':>5s} | {'EXPAND%':>8s} | {'PRESERVE%':>10s} | {'CONTRACT%':>10s} | {'Total E':>10s}")
print("  " + "-"*55)

energy_summary = {}
for li in NZ_LAYERS:
    e = energy_by_state[li]
    me = np.mean(e['expand'])
    mp = np.mean(e['preserve'])
    mc = np.mean(e['contract'])
    mt = np.mean(e['total'])
    total_parts = me + mp + mc
    print(f"  L{li:>3d} | {me/total_parts*100:7.1f}% | {mp/total_parts*100:9.1f}% | {mc/total_parts*100:9.1f}% | {mt:10.2f}")
    energy_summary[li] = {
        'expand_pct': me/total_parts, 'preserve_pct': mp/total_parts,
        'contract_pct': mc/total_parts, 'total_energy': mt,
    }

# ================================================================
# PART B: SVD of CONTRACT contribution
# ================================================================
print("\nPart B: SVD of CONTRACT FFN output at L1-3...")

svd_results = {}
for li in NZ_LAYERS:
    # Stack FFN outputs across prompts: [n_prompts, HDIM]
    F_mat = torch.stack(ffn_outputs[li])  # [15, 3584]

    # SVD
    U, S, Vh = torch.linalg.svd(F_mat, full_matrices=False)
    S_np = S.numpy()

    # Cumulative energy
    total_var = (S_np**2).sum()
    cum_var = np.cumsum(S_np**2) / total_var

    # Find ranks for 90%, 95%, 99%
    r90 = np.searchsorted(cum_var, 0.90) + 1
    r95 = np.searchsorted(cum_var, 0.95) + 1
    r99 = np.searchsorted(cum_var, 0.99) + 1

    print(f"  L{li}: S[0]={S_np[0]:.2f}, S[1]={S_np[1]:.2f}, S[0]/S[1]={S_np[0]/S_np[1]:.2f}")
    print(f"       rank for 90%={r90}, 95%={r95}, 99%={r99}")
    print(f"       Top-5 singular values: {S_np[:5].round(2)}")

    svd_results[li] = {
        'singular_values': S_np[:10].tolist(),
        'rank_90': int(r90), 'rank_95': int(r95), 'rank_99': int(r99),
        'S0_S1_ratio': float(S_np[0]/S_np[1]) if S_np[1] > 0 else float('inf'),
        'Vh': Vh,  # Keep for low-rank approximation
    }

# ================================================================
# PART C: Low-rank replacement test
# ================================================================
print("\nPart C: Testing low-rank FFN replacement at L1-3...")

# Strategy: Replace FFN at L1-3 with rank-k approximation
# For each layer, precompute the top-k right singular vectors (Vh[:k])
# These define the k-dimensional subspace of FFN outputs
# Approximate: ffn_out ≈ Vh[:k].T @ Vh[:k] @ ffn_out (project onto subspace)

# But we need a PREDICTIVE model, not just projection of known outputs.
# The low-rank structure means: ffn_out ≈ A @ ffn_in for some low-rank A
# Let's compute A = ffn_out_stack.T @ ffn_in_stack.pinv() and truncate.

# Actually, simpler: use hooks to project FFN output onto the learned subspace
# at test time. This tests if the rank-k subspace captures the relevant signal.

rank_tests = [1, 2, 3, 5, 8, 15]

for rank in rank_tests:
    top1_matches = []

    for pi, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        # Baseline
        with torch.no_grad():
            bl_out = model(ids, return_dict=True)
        bl_top1 = bl_out.logits[0, -1].argmax().item()

        # Low-rank projection hooks for L1-3
        hooks = []
        for li in NZ_LAYERS:
            Vh_k = svd_results[li]['Vh'][:rank].to(torch.bfloat16).cuda()  # [k, HDIM]
            def make_proj_hook(proj_basis):
                def hk(mod, inp, output):
                    # Project FFN output onto rank-k subspace
                    # out_proj = V_k^T @ V_k @ out
                    out_flat = output.view(-1, HDIM).float()
                    coeffs = out_flat @ proj_basis.float().T  # [S, k]
                    projected = coeffs @ proj_basis.float()  # [S, HDIM]
                    return projected.view(output.shape).to(output.dtype)
                return hk
            hooks.append(model.model.layers[li].mlp.register_forward_hook(make_proj_hook(Vh_k)))

        with torch.no_grad():
            var_out = model(ids, return_dict=True)
        var_top1 = var_out.logits[0, -1].argmax().item()

        for hk in hooks: hk.remove()
        top1_matches.append(var_top1 == bl_top1)

    acc = sum(top1_matches) / len(top1_matches) * 100
    print(f"  Rank {rank:>2d}: top-1 accuracy = {acc:5.1f}%")

# Also test: what if we skip FFN at L1-3 entirely vs rank-1?
# (Already know skip = 20% from Step 4)

# ================================================================
# PART D: Mean FFN replacement (rank-0: just use mean FFN output)
# ================================================================
print("\nPart D: Mean FFN replacement (constant correction)...")

mean_ffn = {}
for li in NZ_LAYERS:
    mean_ffn[li] = torch.stack(ffn_outputs[li]).mean(dim=0)  # [HDIM]

top1_matches = []
for pi, prompt in enumerate(TEST_PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        bl_out = model(ids, return_dict=True)
    bl_top1 = bl_out.logits[0, -1].argmax().item()

    hooks = []
    for li in NZ_LAYERS:
        mf = mean_ffn[li].to(torch.bfloat16).cuda()
        def make_const_hook(const_vec):
            def hk(mod, inp, output):
                # Replace FFN output with constant mean
                result = torch.zeros_like(output)
                result[:, :, :] = const_vec.unsqueeze(0)
                return result
            return hk
        hooks.append(model.model.layers[li].mlp.register_forward_hook(make_const_hook(mf)))

    with torch.no_grad():
        var_out = model(ids, return_dict=True)
    var_top1 = var_out.logits[0, -1].argmax().item()
    for hk in hooks: hk.remove()
    top1_matches.append(var_top1 == bl_top1)

acc = sum(top1_matches) / len(top1_matches) * 100
print(f"  Mean FFN (rank-0 constant): top-1 accuracy = {acc:5.1f}%")

# Save
save_data = {
    'energy_by_state': {str(k): v for k, v in energy_summary.items()},
    'svd_results': {
        str(li): {k: v for k, v in d.items() if k != 'Vh'}
        for li, d in svd_results.items()
    },
}
out_path = os.path.join(results_dir, 'phase10s_neg_zero_structure.json')
with open(out_path, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\n  Saved to {out_path}")
print("="*60)
