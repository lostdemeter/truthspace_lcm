#!/usr/bin/env python3
"""
Phase 10m: The Navigational Attractor Basin

Discovery: When approximate attention (bias-aware, no ww term) replaces real QK,
the hidden-state drift SATURATES at |ε|/|h| ≈ 1.2 ≈ φ^(2-φ).

This experiment characterizes the structure:

1. ATTENTION ENTROPY: Is the attractor caused by attention diffusion?
   Hypothesis: bias-aware attention has HIGHER entropy (more diffuse),
   producing "blurred" outputs that regress toward the mean.

2. DRIFT UNIVERSALITY: Is the saturation level the same across prompts?
   If yes → intrinsic geometric property. If no → content-dependent.

3. φ-STRUCTURE: Does the saturation level φ^(2-φ) hold precisely?
   Measure across many prompts and check.

4. OSCILLATION DYNAMICS: Does drift oscillate or monotonically approach steady state?
   Map the full drift profile at every layer.

5. ZONE BEHAVIOR: How do DRUM/COMB/MUSIC zones contribute to the attractor?

6. RESTORING FORCE: Measure the cos(ε, h) profile — is the negative cosine
   constant or does it vary with drift magnitude? (Like Hooke's law?)
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10m: THE NAVIGATIONAL ATTRACTOR BASIN")
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

# Extract weights + tables
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

def bias_aware_attn(layer_idx, h_normed, attn_module):
    b, seq_len, _ = h_normed.shape
    h_cpu = h_normed[0].cpu().float()
    with torch.no_grad(): V = attn_module.v_proj(h_normed)
    V_r = V.reshape(b, seq_len, NKV, HD).transpose(1, 2)
    V_e = V_r.repeat_interleave(HPK, dim=1)
    all_out = torch.zeros(b, seq_len, NH, HD, device=h_normed.device, dtype=h_normed.dtype)
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
        w_dev = weights.to(h_normed.device).to(h_normed.dtype)
        all_out[0, :, hd, :] = w_dev @ V_e[0, hd]
    combined = all_out.reshape(b, seq_len, NH * HD)
    return attn_module.o_proj(combined)

PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is",
           "The color of grass is","Water freezes at a temperature of",
           "Shakespeare wrote the play","The chemical symbol for gold is",
           "Photosynthesis converts sunlight into","The human heart has four",
           "Mount Everest is the tallest","DNA stands for",
           "In computer science, an algorithm is",
           "The speed of light in vacuum is","The Great Wall of China was built"]

# ================================================================
# TEST 1: ATTENTION ENTROPY — Real vs Bias-Aware
# ================================================================
print("="*80)
print("  TEST 1: ATTENTION ENTROPY (Real vs Bias-Aware)")
print("  Hypothesis: Bias-aware has HIGHER entropy (more diffuse)")
print("="*80+"\n")

prompt = "The capital of France is"
ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
slen = ids.shape[1]

# Capture real attention weights
real_w_all = {}; layer_h_all = {}
def cap_real(li):
    def hf(mod, args, kw, out):
        h=args[0] if args else kw.get('hidden_states')
        if h is None: return out
        b,s,_=h.shape; layer_h_all[li]=h[0].cpu().float()
        with torch.no_grad():
            Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
        Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
        c,sn=rope_cache(s,h.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
        Ke=K.repeat_interleave(HPK,dim=1)
        mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
        wd={}
        for hd in range(NH):
            scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
            scores.masked_fill_(mk,float('-inf'))
            wd[hd]=phi_softmax(scores,dim=-1).cpu()
        real_w_all[li]=wd
        return out
    return hf
hooks=[model.model.layers[li].self_attn.register_forward_hook(cap_real(li),with_kwargs=True) for li in range(NL)]
with torch.no_grad(): model(ids, return_dict=True)
for hk in hooks: hk.remove()

def entropy(w, dim=-1):
    w_pos = w.clamp(min=1e-10)
    return -(w_pos * w_pos.log()).sum(dim=dim)

print(f"  {'Layer':>6s}  {'H_real':>7s}  {'H_approx':>8s}  {'ΔH':>7s}  {'ΔH/H':>7s}  {'Routing':>8s}")
print("  "+"-"*50)

entropy_data = []
for li in range(NL):
    if li not in layer_h_all: continue
    hs = layer_h_all[li]; s = hs.shape[0]
    
    h_reals = []; h_approxs = []
    for hd in layer_cls[li]['routing']:
        if (li,hd) not in head_tables: continue
        tbl = head_tables[(li,hd)]
        # Real entropy
        rw = real_w_all[li][hd]
        hr = entropy(rw, dim=-1).mean().item()
        
        # Approx entropy
        approx_scores = torch.full((s,s), float('-inf'))
        for i in range(s):
            for j in range(i+1):
                delta=i-j
                approx_scores[i,j] = tbl['baseline'][delta].item() + \
                    (hs[i]@tbl['c_q'][delta]).item() + (tbl['c_k'][delta]@hs[j]).item()
        aw = phi_softmax(approx_scores.float(), dim=-1)
        ha = entropy(aw, dim=-1).mean().item()
        
        h_reals.append(hr); h_approxs.append(ha)
    
    if h_reals:
        mr = np.mean(h_reals); ma = np.mean(h_approxs)
        dh = ma - mr; dh_rel = dh / max(mr, 1e-10)
        n_route = len(layer_cls[li]['routing'])
        entropy_data.append({'layer': li, 'real': mr, 'approx': ma, 'delta': dh, 'rel': dh_rel})
        print(f"  L{li:2d}:   {mr:.4f}   {ma:.4f}   {dh:+.4f}  {dh_rel:+.3f}   {n_route:3d} heads")

# Summary
dhs = [d['delta'] for d in entropy_data]
rels = [d['rel'] for d in entropy_data]
pos_count = sum(1 for d in dhs if d > 0)
print(f"\n  Summary: {pos_count}/{len(dhs)} layers have HIGHER entropy (more diffuse) in approx")
print(f"  Mean ΔH = {np.mean(dhs):+.4f}, Mean ΔH/H = {np.mean(rels):+.3f}")

# ================================================================
# TEST 2: DRIFT UNIVERSALITY — Same saturation across prompts?
# ================================================================
print(f"\n{'='*80}")
print("  TEST 2: DRIFT UNIVERSALITY — Is saturation level universal?")
print("="*80+"\n")

all_drift_profiles = []

for pi, prompt in enumerate(PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    
    # Ground truth hidden states
    real_h = {}
    def cap_gt(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is not None: real_h[li]=h[0].detach().cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap_gt(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    # Full stacked bias-aware
    drift_profile = []
    for n_rep in [4, 8, 12, 16, 20, 24, 27]:  # Key checkpoints
        approx_h = {}
        def make_rep(t):
            def hf(mod, args, kw, out):
                h=args[0] if args else kw.get('hidden_states')
                if h is None: return out
                geo = bias_aware_attn(t, h, mod)
                return (geo,) + out[1:] if isinstance(out, tuple) else geo
            return hf
        def cap_p(li):
            def hf(mod, args, kw, out):
                h=args[0] if args else kw.get('hidden_states')
                if h is not None: approx_h[li]=h[0].detach().cpu().float()
                return out
            return hf
        hooks=[]
        for li in range(n_rep):
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(make_rep(li),with_kwargs=True))
        hooks.append(model.model.layers[min(n_rep,NL-1)].self_attn.register_forward_hook(cap_p(min(n_rep,NL-1)),with_kwargs=True))
        with torch.no_grad(): model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        
        probe_li = min(n_rep, NL-1)
        if probe_li in real_h and probe_li in approx_h:
            eps = approx_h[probe_li] - real_h[probe_li]
            rel = eps.norm().item() / max(real_h[probe_li].norm().item(), 1e-30)
            cos_eh = torch.nn.functional.cosine_similarity(
                eps.reshape(1,-1), real_h[probe_li].reshape(1,-1)).item()
            drift_profile.append({'n': n_rep, 'drift': rel, 'cos': cos_eh})
    
    all_drift_profiles.append({'prompt': prompt[:30], 'profile': drift_profile})
    
    if pi < 3:  # Print first 3 in detail
        print(f"  '{prompt[:35]}' ({slen} tokens):")
        for dp in drift_profile:
            print(f"    L0..{dp['n']-1:2d}: drift={dp['drift']:.4f}  cos={dp['cos']:+.4f}")
        print()

# Extract saturation levels (COMB zone: n=12,16,20,24)
comb_drifts = []
for prof in all_drift_profiles:
    for dp in prof['profile']:
        if dp['n'] in [12, 16, 20, 24]:
            comb_drifts.append(dp['drift'])

comb_cos = []
for prof in all_drift_profiles:
    for dp in prof['profile']:
        if dp['n'] in [12, 16, 20, 24]:
            comb_cos.append(dp['cos'])

print(f"  COMB-zone saturation level ({len(comb_drifts)} measurements):")
print(f"    Mean:   {np.mean(comb_drifts):.4f}")
print(f"    Std:    {np.std(comb_drifts):.4f}")
print(f"    CV:     {np.std(comb_drifts)/np.mean(comb_drifts)*100:.1f}%")
print(f"    Range:  [{np.min(comb_drifts):.4f}, {np.max(comb_drifts):.4f}]")

# φ-level analysis
mean_sat = np.mean(comb_drifts)
phi_level = math.log(mean_sat) / LOG_PHI
print(f"\n  φ-level of saturation: φ^{phi_level:.4f}")
print(f"  Compare to 2-φ = {2-PHI:.4f}")
print(f"  Compare to 1/φ² = {1/PHI**2:.4f}")
print(f"  Closest: {'2-φ' if abs(phi_level-(2-PHI)) < abs(phi_level-1/PHI**2) else '1/φ²'}")
print(f"  (Note: 2-φ = 1/φ² = {2-PHI:.6f}, they ARE the same!)")

# Restoring force analysis
print(f"\n  Restoring force (cos(ε,h)) in COMB zone:")
print(f"    Mean cos: {np.mean(comb_cos):+.4f}")
print(f"    Std:      {np.std(comb_cos):.4f}")

# ================================================================
# TEST 3: PER-LAYER DRIFT INCREMENT (the "force" at each layer)
# ================================================================
print(f"\n{'='*80}")
print("  TEST 3: PER-LAYER DRIFT INCREMENT (the 'force field')")
print("  How much does EACH layer contribute to drift?")
print("="*80+"\n")

# Use first prompt for detailed per-layer analysis
prompt = "The capital of France is"
ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
slen = ids.shape[1]

# Real hidden states
real_h = {}
def cap_r(li):
    def hf(mod, args, kw, out):
        h=args[0] if args else kw.get('hidden_states')
        if h is not None: real_h[li]=h[0].detach().cpu().float()
        return out
    return hf
hooks=[model.model.layers[li].self_attn.register_forward_hook(cap_r(li),with_kwargs=True) for li in range(NL)]
with torch.no_grad(): model(ids, return_dict=True)
for hk in hooks: hk.remove()

# For each layer L: replace ONLY layer L, measure drift at L+1
# This is the "incremental force" from each layer
print(f"  {'Layer':>6s}  {'|ε|/|h|':>8s}  {'cos(ε,h)':>9s}  {'φ-level':>8s}  {'Zone':>6s}")
print("  "+"-"*45)

per_layer_drift = []
for target_L in range(NL-1):
    approx_h = {}
    def make_single(t):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            geo = bias_aware_attn(t, h, mod)
            return (geo,) + out[1:] if isinstance(out, tuple) else geo
        return hf
    def cap_next(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is not None: approx_h[li]=h[0].detach().cpu().float()
            return out
        return hf
    hooks=[]
    hooks.append(model.model.layers[target_L].self_attn.register_forward_hook(make_single(target_L),with_kwargs=True))
    hooks.append(model.model.layers[target_L+1].self_attn.register_forward_hook(cap_next(target_L+1),with_kwargs=True))
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    if target_L+1 in real_h and target_L+1 in approx_h:
        eps = approx_h[target_L+1] - real_h[target_L+1]
        rel = eps.norm().item() / max(real_h[target_L+1].norm().item(), 1e-30)
        cos_eh = torch.nn.functional.cosine_similarity(
            eps.reshape(1,-1), real_h[target_L+1].reshape(1,-1)).item()
        pl = math.log(max(rel,1e-30)) / LOG_PHI
        
        zone = "DRUM" if target_L < 4 else ("MUSIC" if target_L >= 26 else "COMB")
        per_layer_drift.append({'layer': target_L, 'drift': rel, 'cos': cos_eh, 'phi_level': pl, 'zone': zone})
        print(f"  L{target_L:2d}:    {rel:.4f}   {cos_eh:+.4f}    φ^{pl:.2f}    {zone}")

# Zone averages
for zone in ['DRUM', 'COMB', 'MUSIC']:
    zd = [d for d in per_layer_drift if d['zone'] == zone]
    if zd:
        md = np.mean([d['drift'] for d in zd])
        mc = np.mean([d['cos'] for d in zd])
        mp = np.mean([d['phi_level'] for d in zd])
        print(f"\n  {zone:6s} mean: drift={md:.4f}  cos={mc:+.4f}  φ^{mp:.2f}")

# ================================================================
# TEST 4: OSCILLATION — Does drift oscillate before settling?
# ================================================================
print(f"\n{'='*80}")
print("  TEST 4: OSCILLATION DYNAMICS")
print("  Full drift profile at EVERY layer (cumulative replacement)")
print("="*80+"\n")

# Full profile for 3 diverse prompts
for pi, prompt in enumerate(PROMPTS[:3]):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    
    real_h2 = {}
    def cap_r2(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is not None: real_h2[li]=h[0].detach().cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap_r2(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    drifts = []; cosines = []
    for n_rep in range(1, NL):
        approx_h2 = {}
        def make_r2(t):
            def hf(mod, args, kw, out):
                h=args[0] if args else kw.get('hidden_states')
                if h is None: return out
                geo = bias_aware_attn(t, h, mod)
                return (geo,) + out[1:] if isinstance(out, tuple) else geo
            return hf
        def cap_p2(li):
            def hf(mod, args, kw, out):
                h=args[0] if args else kw.get('hidden_states')
                if h is not None: approx_h2[li]=h[0].detach().cpu().float()
                return out
            return hf
        hooks=[]
        for li in range(n_rep):
            hooks.append(model.model.layers[li].self_attn.register_forward_hook(make_r2(li),with_kwargs=True))
        hooks.append(model.model.layers[n_rep].self_attn.register_forward_hook(cap_p2(n_rep),with_kwargs=True))
        with torch.no_grad(): model(ids, return_dict=True)
        for hk in hooks: hk.remove()
        
        if n_rep in real_h2 and n_rep in approx_h2:
            eps = approx_h2[n_rep] - real_h2[n_rep]
            rel = eps.norm().item() / max(real_h2[n_rep].norm().item(), 1e-30)
            cos_eh = torch.nn.functional.cosine_similarity(
                eps.reshape(1,-1), real_h2[n_rep].reshape(1,-1)).item()
            drifts.append(rel); cosines.append(cos_eh)
    
    print(f"  '{prompt[:35]}' ({slen} tokens):")
    # Show as ASCII sparkline
    max_d = max(drifts) if drifts else 1
    for i, (d, c) in enumerate(zip(drifts, cosines)):
        bar = "█" * int(d / max_d * 30)
        zone = "D" if i < 4 else ("M" if i >= 26 else "C")
        print(f"    L0..{i:2d} [{zone}]: {d:.3f} {bar}")
    
    # Analyze oscillation
    if len(drifts) > 5:
        deltas = np.diff(drifts)
        sign_changes = sum(1 for i in range(len(deltas)-1) if deltas[i]*deltas[i+1] < 0)
        print(f"    Sign changes in drift increment: {sign_changes} (oscillations: {sign_changes//2})")
        
        # COMB-zone mean and std
        comb = drifts[4:26] if len(drifts) > 26 else drifts[4:]
        if comb:
            print(f"    COMB steady state: {np.mean(comb):.4f} ± {np.std(comb):.4f}")
            print(f"    COMB φ-level: φ^{math.log(np.mean(comb))/LOG_PHI:.4f}")
    print()

# ================================================================
# SYNTHESIS
# ================================================================
print("="*80)
print("  SYNTHESIS: THE NAVIGATIONAL ATTRACTOR BASIN")
print("="*80+"\n")

# Collect all COMB-zone measurements for final φ analysis
all_comb = []
for prof in all_drift_profiles:
    for dp in prof['profile']:
        if dp['n'] in [12, 16, 20, 24]:
            all_comb.append(dp['drift'])

sat_mean = np.mean(all_comb)
sat_std = np.std(all_comb)
sat_phi = math.log(sat_mean) / LOG_PHI

print(f"  SATURATION LEVEL: {sat_mean:.4f} ± {sat_std:.4f}")
print(f"  φ-LEVEL: φ^{sat_phi:.4f}")
print(f"")
print(f"  φ-constant candidates:")
print(f"    φ^(2-φ) = φ^{2-PHI:.4f} = {PHI**(2-PHI):.4f}  (error: {abs(sat_mean - PHI**(2-PHI)):.4f})")
print(f"    φ^(1/φ) = φ^{1/PHI:.4f} = {PHI**(1/PHI):.4f}  (error: {abs(sat_mean - PHI**(1/PHI)):.4f})")
print(f"    φ^(1/e) = φ^{1/math.e:.4f} = {PHI**(1/math.e):.4f}  (error: {abs(sat_mean - PHI**(1/math.e)):.4f})")
print(f"    √φ      = {PHI**0.5:.4f}  (error: {abs(sat_mean - PHI**0.5):.4f})")
print(f"    1/φ^(1/φ)= {1/PHI**(1/PHI):.4f}  (error: {abs(sat_mean - 1/PHI**(1/PHI)):.4f})")

# Check entropy finding
dh_signs = [d['delta'] for d in entropy_data]
n_higher = sum(1 for d in dh_signs if d > 0)
print(f"\n  ENTROPY: {n_higher}/{len(dh_signs)} layers have HIGHER approx entropy → {'CONFIRMED' if n_higher > len(dh_signs)//2 else 'REJECTED'}: attention diffusion")

# cos(ε, h) finding
cos_mean = np.mean(comb_cos) if comb_cos else 0
print(f"  RESTORING FORCE: cos(ε,h) = {cos_mean:+.4f} → {'CONFIRMED' if cos_mean < -0.1 else 'WEAK'}: error opposes h")

# Universality
cv = sat_std / sat_mean * 100
print(f"  UNIVERSALITY: CV = {cv:.1f}% → {'CONFIRMED' if cv < 15 else 'PARTIAL'}: saturation is {'prompt-independent' if cv < 15 else 'partially prompt-dependent'}")

# Save
res = {
    'saturation_mean': float(sat_mean),
    'saturation_std': float(sat_std),
    'saturation_phi_level': float(sat_phi),
    'phi_2_minus_phi': float(PHI**(2-PHI)),
    'entropy_higher_count': n_higher,
    'entropy_total': len(dh_signs),
    'cos_mean': float(cos_mean),
    'universality_cv': float(cv),
    'per_layer_drift': [{'layer': d['layer'], 'drift': d['drift'], 'cos': d['cos'], 'zone': d['zone']} for d in per_layer_drift],
}
with open(os.path.join(results_dir, 'phase10m_attractor.json'), 'w') as f:
    json.dump(res, f, indent=2)

print(f"\nSaved to results/phase10m_attractor.json")
print("="*80+"\n  DONE\n"+"="*80)
