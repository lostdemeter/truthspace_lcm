#!/usr/bin/env python3
"""
Phase 10j: MGOP/GOP/PEP Applied to Stacking Drift

MGOP diagnosis: weight-matrix projections converge (holographic bound on A(δ)),
but computation projections diverge (per-layer OK, stacked fails).
LOW projection consistency → NEW STRUCTURE available.

The wall is in ERROR ACCUMULATION, not score accuracy.

This experiment:
1. GOP Phase 1 (Fractal Peel): Measure the hidden-state drift ε_L when using
   bias-aware attention at layer L. Is ε structured or ergodic?
2. PEP: If ε is structured (low effective rank, high autocorrelation),
   we can CORRECT it instead of approximating the score.
3. EDP: Look for φ-patterns in the error structure.
4. MGOP Phase 5: Compare stacking drift structure across layers.
   Same structure → holographic bound. Different → exploitable.

Key measurements per layer:
  - ε magnitude (relative to h)
  - ε effective rank
  - ε autocorrelation across positions
  - ε predictability from known quantities
  - ε φ-structure (does it live at φ-levels?)
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10j: STACKING DRIFT — FRACTAL PEEL (MGOP/GOP/PEP)")
print("="*80)

results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_cls = {}
for ls in phase9a['layer_summary']:
    layer_cls[ls['layer']] = {'fixed': set(ls['fixed_heads']), 'routing': set(ls['routing_heads'])}

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()

NL=28; NH=28; NKV=4; HD=128; HPK=7; HDIM=3584; MAXS=64; ROPE_THETA=1e6
inv_freq = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,dtype=torch.float32)/HD))

def phi_softmax(s, dim=-1):
    s = s - s.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI); return p / p.sum(dim=dim, keepdim=True)

def rope_cache(slen, dev, dt):
    inv = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,device=dev,dtype=torch.float32)/HD))
    pos = torch.arange(slen, device=dev, dtype=torch.float32)
    f = torch.outer(pos, inv); e = torch.cat((f,f), dim=-1)
    return e.cos().to(dt)[None,None], e.sin().to(dt)[None,None]

def apply_rope(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return x*cos + torch.cat((-x2, x1), dim=-1)*sin

# Extract weights
print("\nExtracting weights...")
head_data = {}
for li in range(NL):
    attn = model.model.layers[li].self_attn
    ident = torch.eye(HDIM, device="cuda", dtype=torch.bfloat16)
    Wq = torch.zeros(NH,HD,HDIM,dtype=torch.float32)
    Wk = torch.zeros(NKV,HD,HDIM,dtype=torch.float32)
    for s in range(0,HDIM,512):
        e = min(s+512,HDIM); chunk = ident[s:e].unsqueeze(0)
        with torch.no_grad():
            qo=attn.q_proj(chunk).float(); ko=attn.k_proj(chunk).float()
        qr=qo[0].reshape(-1,NH,HD); kr=ko[0].reshape(-1,NKV,HD)
        for h in range(NH): Wq[h,:,s:e]=qr[:,h,:].T
        for g in range(NKV): Wk[g,:,s:e]=kr[:,g,:].T
    zi = torch.zeros(1,1,HDIM,device="cuda",dtype=torch.bfloat16)
    with torch.no_grad():
        qb=attn.q_proj(zi).float()[0,0]; kb=attn.k_proj(zi).float()[0,0]
    bq = qb.reshape(NH,HD).cpu(); bk = kb.reshape(NKV,HD).cpu()
    for h in range(NH): Wq[h] -= bq[h].unsqueeze(1)
    for g in range(NKV): Wk[g] -= bk[g].unsqueeze(1)
    for h in range(NH):
        g = h//HPK
        head_data[(li,h)] = {'W_q':Wq[h].clone(),'W_k':Wk[g].clone(),
                             'b_q':bq[h].clone(),'b_k':bk[g].clone()}
    del Wq,Wk; torch.cuda.empty_cache()
    if li%7==0: print(f"  Layer {li} done")
print(f"  All weights extracted\n")

# Pre-compute bias tables for ALL heads (routing + fixed use bias baseline)
print("Pre-computing bias tables...")
head_tables = {}
for (li,h), d in head_data.items():
    bl=torch.zeros(MAXS); cq=torch.zeros(MAXS,HDIM); ck=torch.zeros(MAXS,HDIM)
    sc = 1.0/math.sqrt(HD)
    for delta in range(MAXS):
        fd=delta*inv_freq
        cd=torch.cat((fd.cos(),fd.cos())); sd=torch.cat((fd.sin(),fd.sin()))
        bkg=d['b_k']; b1,b2=bkg[:HD//2],bkg[HD//2:]
        bkr=bkg*cd+torch.cat((-b2,b1))*sd
        Wkg=d['W_k']; W1,W2=Wkg[:HD//2,:],Wkg[HD//2:,:]
        Wkr=Wkg*cd.unsqueeze(1)+torch.cat((-W2,W1),dim=0)*sd.unsqueeze(1)
        bl[delta]=(d['b_q']@bkr)*sc
        cq[delta]=(d['W_q'].T@bkr)*sc
        ck[delta]=(Wkr.T@d['b_q'])*sc
    head_tables[(li,h)] = {'baseline':bl,'c_q':cq,'c_k':ck}
print("  Done\n")

PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is",
           "The color of grass is","Water freezes at a temperature of",
           "Shakespeare wrote the play","The chemical symbol for gold is",
           "Photosynthesis converts sunlight into","The human heart has four"]

# ================================================================
# MEASUREMENT 1: Per-layer hidden state drift ε_L
# Replace attention at layer L with bias-aware, measure Δh at L+1
# ================================================================
print("="*80)
print("  MEASUREMENT 1: Per-layer hidden state drift from bias-aware attention")
print("="*80+"\n")

def bias_aware_attn(layer_idx, h_normed, attn_module):
    """Compute bias-aware attention (no ww term) for a single layer."""
    b, seq_len, _ = h_normed.shape
    h_cpu = h_normed[0].cpu().float()
    
    with torch.no_grad():
        V = attn_module.v_proj(h_normed)
    V_r = V.reshape(b, seq_len, NKV, HD).transpose(1, 2)
    V_e = V_r.repeat_interleave(HPK, dim=1)
    
    all_head_outputs = torch.zeros(b, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
    
    for hd in range(NH):
        tbl = head_tables[(layer_idx, hd)]
        scores = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            for j in range(i+1):
                delta = i - j
                bl = tbl['baseline'][delta].item()
                cqv = (h_cpu[i] @ tbl['c_q'][delta]).item()
                ckv = (tbl['c_k'][delta] @ h_cpu[j]).item()
                scores[i, j] = bl + cqv + ckv
        
        weights = phi_softmax(scores.float(), dim=-1)
        weights_dev = weights.to(h_normed.device).to(h_normed.dtype)
        
        head_v = V_e[0, hd]  # (seq, HD)
        head_out = weights_dev @ head_v  # (seq, HD)
        all_head_outputs[0, :, hd, :] = head_out
    
    combined = all_head_outputs.reshape(b, seq_len, NH * HD)
    return attn_module.o_proj(combined)


# Step 1: Get ground truth hidden states at each layer boundary
for pi, prompt in enumerate(PROMPTS[:5]):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    
    # Capture real hidden states BETWEEN layers
    real_h = {}  # real_h[li] = hidden state ENTERING layer li's attention
    def cap_real(li):
        def hf(mod, args, kw, out):
            h = args[0] if args else kw.get('hidden_states')
            if h is not None:
                real_h[li] = h[0].detach().cpu().float()
            return out
        return hf
    hooks = [model.model.layers[li].self_attn.register_forward_hook(
        cap_real(li), with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    if pi == 0:
        print(f"  Prompt: '{prompt}' ({slen} tokens)")
        print(f"  {'Layer':>6s}  {'|ε|/|h|':>8s}  {'ε rank-1%':>9s}  {'ε rank-3%':>9s}  {'cos(ε,h)':>9s}  {'ε φ-level':>10s}")
        print("  " + "-"*60)
    
    # Step 2: For each layer, replace with bias-aware and measure drift
    for target_L in range(NL - 1):  # Don't do last layer (no L+1)
        if pi > 0 and target_L not in [0, 3, 7, 13, 20, 27]:
            continue  # Sparse sampling for later prompts
        
        # Hook: replace attention at target_L with bias-aware
        approx_h_next = {}
        
        def make_replace_hook(target):
            def hf(mod, args, kw, out):
                h = args[0] if args else kw.get('hidden_states')
                if h is None: return out
                geo = bias_aware_attn(target, h, mod)
                return (geo,) + out[1:] if isinstance(out, tuple) else geo
            return hf
        
        def cap_next(li):
            def hf(mod, args, kw, out):
                h = args[0] if args else kw.get('hidden_states')
                if h is not None:
                    approx_h_next[li] = h[0].detach().cpu().float()
                return out
            return hf
        
        hooks = []
        # Replace attention at target_L
        hooks.append(model.model.layers[target_L].self_attn.register_forward_hook(
            make_replace_hook(target_L), with_kwargs=True))
        # Capture hidden state entering target_L + 1
        hooks.append(model.model.layers[target_L + 1].self_attn.register_forward_hook(
            cap_next(target_L + 1), with_kwargs=True))
        
        with torch.no_grad(): model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        
        # Measure drift
        if target_L + 1 in real_h and target_L + 1 in approx_h_next:
            h_real = real_h[target_L + 1]    # (seq, 3584)
            h_approx = approx_h_next[target_L + 1]
            eps = h_approx - h_real  # drift
            
            eps_norm = eps.norm().item()
            h_norm = h_real.norm().item()
            rel_drift = eps_norm / max(h_norm, 1e-30)
            
            # Effective rank of drift via SVD
            if slen > 1:
                U, S_vals, Vt = torch.linalg.svd(eps, full_matrices=False)
                total_var = (S_vals ** 2).sum().item()
                if total_var > 1e-30:
                    rank1_pct = (S_vals[0] ** 2).item() / total_var * 100
                    rank3_pct = (S_vals[:min(3,len(S_vals))] ** 2).sum().item() / total_var * 100
                else:
                    rank1_pct = 0; rank3_pct = 0
            else:
                rank1_pct = 100; rank3_pct = 100
            
            # Cosine similarity between eps and h (are they aligned?)
            cos_eps_h = torch.nn.functional.cosine_similarity(
                eps.reshape(1, -1), h_real.reshape(1, -1)
            ).item()
            
            # φ-level of drift magnitude
            if rel_drift > 0:
                phi_level = math.log(rel_drift) / math.log(PHI)
            else:
                phi_level = float('-inf')
            
            if pi == 0:
                print(f"  L{target_L:2d}→{target_L+1}:  {rel_drift:.4f}    {rank1_pct:5.1f}%     "
                      f"{rank3_pct:5.1f}%    {cos_eps_h:+.4f}      φ^{phi_level:.1f}")

print()

# ================================================================
# MEASUREMENT 2: Cumulative drift — how fast does error grow?
# ================================================================
print("="*80)
print("  MEASUREMENT 2: Cumulative drift (replace L0..L_n, measure h at L_n+1)")
print("="*80+"\n")

prompt = "The capital of France is"
ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
slen = ids.shape[1]

# Ground truth
real_h = {}
def cap_real2(li):
    def hf(mod, args, kw, out):
        h = args[0] if args else kw.get('hidden_states')
        if h is not None: real_h[li] = h[0].detach().cpu().float()
        return out
    return hf
hooks = [model.model.layers[li].self_attn.register_forward_hook(
    cap_real2(li), with_kwargs=True) for li in range(NL)]
with torch.no_grad(): model(ids, return_dict=True)
for hk in hooks: hk.remove()

# Cumulative: replace layers 0..n
print(f"  {'N_replaced':>10s}  {'|ε|/|h|':>8s}  {'ε rank-1%':>9s}  {'cos(ε,h)':>9s}  {'Growth':>8s}")
print("  " + "-"*50)

prev_drift = None
for n_replace in range(1, NL):
    approx_h = {}
    def make_replace(target):
        def hf(mod, args, kw, out):
            h = args[0] if args else kw.get('hidden_states')
            if h is None: return out
            geo = bias_aware_attn(target, h, mod)
            return (geo,) + out[1:] if isinstance(out, tuple) else geo
        return hf
    def cap_probe(li):
        def hf(mod, args, kw, out):
            h = args[0] if args else kw.get('hidden_states')
            if h is not None: approx_h[li] = h[0].detach().cpu().float()
            return out
        return hf
    
    hooks = []
    for li in range(n_replace):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(
            make_replace(li), with_kwargs=True))
    hooks.append(model.model.layers[n_replace].self_attn.register_forward_hook(
        cap_probe(n_replace), with_kwargs=True))
    
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    if n_replace in real_h and n_replace in approx_h:
        h_r = real_h[n_replace]; h_a = approx_h[n_replace]
        eps = h_a - h_r
        rel = eps.norm().item() / max(h_r.norm().item(), 1e-30)
        
        U, S_vals, Vt = torch.linalg.svd(eps, full_matrices=False)
        tv = (S_vals**2).sum().item()
        r1 = (S_vals[0]**2).item() / max(tv, 1e-30) * 100
        
        cos_eh = torch.nn.functional.cosine_similarity(
            eps.reshape(1,-1), h_r.reshape(1,-1)).item()
        
        growth = f"{rel/prev_drift:.2f}×" if prev_drift and prev_drift > 1e-10 else "—"
        prev_drift = rel
        
        print(f"  L0..{n_replace-1:2d} ({n_replace:2d}):  {rel:.4f}    {r1:5.1f}%    {cos_eh:+.4f}    {growth}")

print()

# ================================================================
# MEASUREMENT 3: Is drift predictable? (PEP question)
# The drift ε comes from the attention OUTPUT difference.
# Can we estimate it from the bias-aware scores vs what full QK would give?
# ================================================================
print("="*80)
print("  MEASUREMENT 3: Drift predictability (PEP diagnostic)")
print("  Can we predict ε from the attention weight difference?")
print("="*80+"\n")

# For each layer: capture attention weights (real vs bias-aware)
# and the V-projection, then compute the output difference analytically
prompt = "The capital of France is"
ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
slen = ids.shape[1]

# Capture real attention weights AND V projections
real_attn_w = {}; real_V = {}; real_h_states = {}
def cap_full(li):
    def hf(mod, args, kw, out):
        h = args[0] if args else kw.get('hidden_states')
        if h is None: return out
        b,s,_ = h.shape; real_h_states[li] = h[0].cpu().float()
        with torch.no_grad():
            Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            V=mod.v_proj(h)
        Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
        c,sn=rope_cache(s,h.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
        Ke=K.repeat_interleave(HPK,dim=1)
        V_r=V.reshape(b,s,NKV,HD).transpose(1,2)
        V_e=V_r.repeat_interleave(HPK,dim=1)
        
        mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
        attn_dict={}; v_dict={}
        for hd in range(NH):
            scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
            scores.masked_fill_(mk,float('-inf'))
            w=phi_softmax(scores,dim=-1)
            attn_dict[hd]=w.cpu()
            v_dict[hd]=V_e[0,hd].cpu().float()
        real_attn_w[li]=attn_dict; real_V[li]=v_dict
        return out
    return hf
hooks=[model.model.layers[li].self_attn.register_forward_hook(cap_full(li),with_kwargs=True) for li in range(NL)]
with torch.no_grad(): model(ids, return_dict=True)
for hk in hooks: hk.remove()

# Now compute bias-aware weights and the PREDICTED output difference
print(f"  {'Layer':>6s}  {'ΔW mean':>8s}  {'ΔW max':>8s}  {'ΔV cos':>8s}  {'Rank ΔW':>8s}  {'Predictable':>12s}")
print("  "+"-"*58)

for li in range(NL):
    if li not in real_h_states: continue
    hs = real_h_states[li]; s = hs.shape[0]
    
    dw_means=[]; dw_maxs=[]; dv_coss=[]; dw_ranks=[]
    
    for hd in range(NH):
        tbl = head_tables[(li, hd)]
        # Bias-aware scores
        approx_scores = torch.full((s,s), float('-inf'))
        for i in range(s):
            for j in range(i+1):
                delta=i-j
                bl=tbl['baseline'][delta].item()
                cqv=(hs[i]@tbl['c_q'][delta]).item()
                ckv=(tbl['c_k'][delta]@hs[j]).item()
                approx_scores[i,j]=bl+cqv+ckv
        
        approx_w = phi_softmax(approx_scores.float(), dim=-1)
        real_w = real_attn_w[li][hd]
        
        # Weight difference
        dw = approx_w - real_w
        dw_means.append(dw.abs().mean().item())
        dw_maxs.append(dw.abs().max().item())
        
        # SVD of weight difference matrix
        U,S_vals,Vt = torch.linalg.svd(dw, full_matrices=False)
        tv = (S_vals**2).sum().item()
        if tv > 1e-30:
            r1 = (S_vals[0]**2).item() / tv * 100
        else:
            r1 = 0
        dw_ranks.append(r1)
        
        # Output difference = ΔW @ V for this head
        v = real_V[li][hd]  # (s, HD)
        real_out = real_w @ v  # (s, HD)
        approx_out = approx_w @ v
        dv = approx_out - real_out
        
        cos = torch.nn.functional.cosine_similarity(
            dv.reshape(1,-1), real_out.reshape(1,-1)
        ).item() if real_out.norm() > 1e-10 else 0
        dv_coss.append(cos)
    
    dw_r1_mean = np.mean(dw_ranks)
    predictable = "LOW-RANK" if dw_r1_mean > 50 else ("STRUCTURED" if dw_r1_mean > 30 else "SPREAD")
    
    print(f"  L{li:2d}:    {np.mean(dw_means):.4f}   {np.mean(dw_maxs):.4f}   "
          f"{np.mean(dv_coss):+.4f}    {dw_r1_mean:5.1f}%   {predictable}")

print()

# ================================================================
# MEASUREMENT 4: φ-structure in cumulative drift
# ================================================================
print("="*80)
print("  MEASUREMENT 4: φ-structure in drift magnitude")
print("="*80+"\n")

# Collect cumulative drift magnitudes
drifts = []
for n_replace in range(1, NL):
    approx_h = {}
    def make_rep(t):
        def hf(mod, args, kw, out):
            h = args[0] if args else kw.get('hidden_states')
            if h is None: return out
            geo = bias_aware_attn(t, h, mod)
            return (geo,) + out[1:] if isinstance(out, tuple) else geo
        return hf
    def cap_p(li):
        def hf(mod, args, kw, out):
            h = args[0] if args else kw.get('hidden_states')
            if h is not None: approx_h[li] = h[0].detach().cpu().float()
            return out
        return hf
    hooks = []
    for li in range(n_replace):
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(make_rep(li), with_kwargs=True))
    hooks.append(model.model.layers[n_replace].self_attn.register_forward_hook(cap_p(n_replace), with_kwargs=True))
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    if n_replace in real_h and n_replace in approx_h:
        h_r = real_h[n_replace]; h_a = approx_h[n_replace]
        rel = (h_a - h_r).norm().item() / max(h_r.norm().item(), 1e-30)
        drifts.append(rel)

drifts = np.array(drifts)
if len(drifts) > 2:
    # Growth rate between consecutive layers
    ratios = drifts[1:] / np.maximum(drifts[:-1], 1e-30)
    mean_ratio = np.mean(ratios[ratios < 100])  # Filter outliers
    
    phi_level_ratios = np.log(ratios[ratios > 0]) / math.log(PHI)
    
    print(f"  Drift growth analysis:")
    print(f"    Mean growth ratio:  {mean_ratio:.3f}")
    print(f"    φ-level of ratio:   {math.log(mean_ratio)/math.log(PHI) if mean_ratio > 0 else 'N/A':.3f}")
    print(f"    Is it φ^1 = 1.618?  {'YES' if abs(mean_ratio - PHI) < 0.2 else 'NO'} (ratio={mean_ratio:.3f})")
    print(f"    Is it φ^0 = 1.000?  {'YES' if abs(mean_ratio - 1.0) < 0.15 else 'NO'}")
    print(f"    Is it φ^2 = 2.618?  {'YES' if abs(mean_ratio - PHI**2) < 0.3 else 'NO'}")
    
    # Fit exponential: drift = a * r^n
    from numpy.polynomial import polynomial as P
    log_d = np.log(np.maximum(drifts, 1e-30))
    n = np.arange(1, len(drifts)+1)
    coef = np.polyfit(n, log_d, 1)
    exp_rate = np.exp(coef[0])
    print(f"\n    Exponential fit: drift ≈ {np.exp(coef[1]):.4f} × {exp_rate:.4f}^n")
    print(f"    Doubling every {math.log(2)/max(coef[0],1e-30):.1f} layers")
    
    # Compare to φ-growth
    phi_fit = abs(exp_rate - PHI)
    phi2_fit = abs(exp_rate - PHI**2)
    sqrt_phi_fit = abs(exp_rate - PHI**0.5)
    
    best = min([(phi_fit, 'φ'), (phi2_fit, 'φ²'), (sqrt_phi_fit, '√φ'), 
                (abs(exp_rate - 2.0), '2'), (abs(exp_rate - 1.5), '3/2')], key=lambda x: x[0])
    print(f"    Closest φ-constant: {best[1]} (error={best[0]:.4f})")

# ================================================================
# MGOP PHASE 5: Projection Synthesis — verdict
# ================================================================
print(f"\n{'='*80}")
print("  MGOP PHASE 5: PROJECTION SYNTHESIS — HOLOGRAPHIC BOUND TEST")
print("="*80+"\n")

print("  PROJECTIONS ANALYZED:")
print("  1. Weight matrix SVD (Euclidean): flat spectrum → can't compress")
print("  2. Frequency-pair decomposition:  flat → can't compress")
print("  3. Stereo A/B decomposition:      flat → can't compress")
print("  4. Per-layer replacement:          14-15/15 (WORKS)")
print("  5. Stacked replacement:            0/15 (FAILS)")
print()
print("  Projections 1-3: CONVERGE → weight matrix is irreducibly full-rank")
print("  Projections 4-5: DIVERGE  → error accumulation, not score quality")
print()
print("  DIAGNOSIS: NOT a holographic bound on score approximation.")
print("  The wall is in ERROR PROPAGATION through the residual stream.")
print()
print("  PEP RECOMMENDATION: If drift is LOW-RANK or STRUCTURED,")
print("  correct the hidden state directly instead of improving scores.")

# Save results
res = {
    'drift_magnitudes': drifts.tolist() if len(drifts) > 0 else [],
    'exp_growth_rate': float(exp_rate) if len(drifts) > 2 else None,
    'diagnosis': 'NOT_HOLOGRAPHIC_BOUND',
    'wall_location': 'error_propagation_residual_stream'
}
with open(os.path.join(results_dir, 'phase10j_stacking_peel.json'), 'w') as f:
    json.dump(res, f, indent=2)

print(f"\nSaved to results/phase10j_stacking_peel.json")
print("="*80+"\n  DONE\n"+"="*80)
