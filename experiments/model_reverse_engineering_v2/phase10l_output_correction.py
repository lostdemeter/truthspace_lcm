#!/usr/bin/env python3
"""
Phase 10l: Output Correction (PEP-Informed Attack)

From 10j: ΔW is rank-1 dominant. The OUTPUT error per head is rank-1.
From protocols: "Stop approximating scores, correct the output."

KEY QUESTION: Is the rank-1 correction DIRECTION stable across prompts?
If yes → precompute direction, estimate magnitude at runtime → O(S) correction.

Test plan:
1. For multiple prompts, compute the rank-1 ΔW at each layer/head
2. Extract the correction direction in hidden space (after O-projection)
3. Measure cosine similarity of this direction ACROSS prompts
4. If stable: estimate the magnitude scalar from cheap features
5. End-to-end: bias-aware + output correction → does stacking improve?
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10l: OUTPUT CORRECTION — IS THE DIRECTION STABLE?")
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

# Extract O-projection weights
print("Extracting O-projection weights...")
O_weights = {}
for li in range(NL):
    attn = model.model.layers[li].self_attn
    ident = torch.eye(NH*HD, device="cuda", dtype=torch.bfloat16)
    Wo_full = torch.zeros(HDIM, NH*HD, dtype=torch.float32)
    for s in range(0, NH*HD, 512):
        e = min(s+512, NH*HD); chunk = ident[s:e].unsqueeze(0)
        with torch.no_grad(): oo = attn.o_proj(chunk).float()
        Wo_full[:, s:e] = oo[0].T
    O_weights[li] = Wo_full.cpu()
print("  Done\n")

PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is",
           "The color of grass is","Water freezes at a temperature of",
           "Shakespeare wrote the play","The chemical symbol for gold is"]

# ================================================================
# For each prompt, compute the per-layer correction direction in hidden space
# ================================================================
print("="*80)
print("  Computing correction directions across prompts...")
print("="*80+"\n")

# correction_dirs[li] = list of (direction_vector, magnitude) across prompts
correction_dirs = defaultdict(list)

for pi, prompt in enumerate(PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    
    # Capture real attention weights and V
    real_w = {}; real_V = {}; layer_h = {}
    def cap(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape; layer_h[li]=h[0].cpu().float()
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
                V=mod.v_proj(h)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16)
            Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            V_r=V.reshape(b,s,NKV,HD).transpose(1,2)
            V_e=V_r.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            wd={}; vd={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                wd[hd]=phi_softmax(scores,dim=-1).cpu()
                vd[hd]=V_e[0,hd].cpu().float()
            real_w[li]=wd; real_V[li]=vd
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    # For each layer: compute bias-aware weights, get ΔW, compute output correction
    for li in range(NL):
        if li not in layer_h: continue
        hs = layer_h[li]; s = hs.shape[0]
        
        # Accumulate per-head output errors, then project through O
        # Full output = concat of all head outputs = (seq, NH*HD)
        # ΔOutput = concat of ΔW_h @ V_h for each head
        delta_full_output = torch.zeros(s, NH * HD)
        
        for hd in range(NH):
            tbl = head_tables[(li, hd)]
            # Cheap scores
            approx_scores = torch.full((s, s), float('-inf'))
            for i in range(s):
                for j in range(i+1):
                    delta = i - j
                    bl = tbl['baseline'][delta].item()
                    cqv = (hs[i] @ tbl['c_q'][delta]).item()
                    ckv = (tbl['c_k'][delta] @ hs[j]).item()
                    approx_scores[i,j] = bl + cqv + ckv
            
            approx_w = phi_softmax(approx_scores.float(), dim=-1)
            dw = approx_w - real_w[li][hd]  # (s, s)
            v = real_V[li][hd]  # (s, HD)
            d_head_out = dw @ v  # (s, HD)
            delta_full_output[:, hd*HD:(hd+1)*HD] = d_head_out
        
        # Project through O to get hidden-space correction
        # delta_hidden = delta_full_output @ O^T  (s, HDIM)
        Wo = O_weights[li]  # (HDIM, NH*HD)
        delta_hidden = (delta_full_output @ Wo.T)  # (s, HDIM)
        
        # SVD of delta_hidden to get the dominant direction
        U, S_vals, Vt = torch.linalg.svd(delta_hidden, full_matrices=False)
        
        if S_vals[0].item() > 1e-10:
            direction = Vt[0]  # dominant direction in hidden space (HDIM-dim)
            magnitude = S_vals[0].item()
            r1_pct = (S_vals[0]**2).item() / max((S_vals**2).sum().item(), 1e-30) * 100
            correction_dirs[li].append({
                'direction': direction,
                'magnitude': magnitude,
                'rank1_pct': r1_pct
            })

# ================================================================
# STABILITY TEST: cosine similarity of correction direction across prompts
# ================================================================
print(f"  {'Layer':>6s}  {'R1%':>5s}  {'Dir cos (mean)':>14s}  {'Dir cos (min)':>13s}  {'Mag std/mean':>12s}  {'Stable?':>8s}")
print("  "+"-"*65)

stability_results = {}
for li in range(NL):
    dirs = correction_dirs[li]
    if len(dirs) < 2: continue
    
    # Pairwise cosine similarities of directions
    n = len(dirs)
    cos_sims = []
    for i in range(n):
        for j in range(i+1, n):
            cos = torch.nn.functional.cosine_similarity(
                dirs[i]['direction'].unsqueeze(0),
                dirs[j]['direction'].unsqueeze(0)
            ).item()
            cos_sims.append(abs(cos))
    
    mean_r1 = np.mean([d['rank1_pct'] for d in dirs])
    mean_cos = np.mean(cos_sims)
    min_cos = np.min(cos_sims) if cos_sims else 0
    
    mags = [d['magnitude'] for d in dirs]
    mag_cv = np.std(mags) / max(np.mean(mags), 1e-30)
    
    stable = "YES" if mean_cos > 0.7 else ("PARTIAL" if mean_cos > 0.4 else "NO")
    
    stability_results[li] = {
        'mean_cos': mean_cos, 'min_cos': min_cos, 
        'mean_r1': mean_r1, 'mag_cv': mag_cv, 'stable': stable
    }
    
    print(f"  L{li:2d}:   {mean_r1:4.1f}%  {mean_cos:12.4f}    {min_cos:11.4f}     {mag_cv:10.4f}   {stable}")

# Summary
n_stable = sum(1 for v in stability_results.values() if v['stable'] == 'YES')
n_partial = sum(1 for v in stability_results.values() if v['stable'] == 'PARTIAL')
n_unstable = sum(1 for v in stability_results.values() if v['stable'] == 'NO')

print(f"\n  Summary: {n_stable} STABLE, {n_partial} PARTIAL, {n_unstable} UNSTABLE out of {len(stability_results)} layers")

# ================================================================
# If stable: test correction
# ================================================================
if n_stable + n_partial > NL // 2:
    print(f"\n{'='*80}")
    print("  CORRECTION TEST: Precompute mean direction, apply at runtime")
    print("="*80+"\n")
    
    # Compute mean correction direction per layer
    mean_dirs = {}
    for li in range(NL):
        dirs = correction_dirs[li]
        if len(dirs) < 2: continue
        # Average the directions (accounting for sign ambiguity via alignment to first)
        ref = dirs[0]['direction']
        aligned = [ref]
        for d in dirs[1:]:
            cos = torch.dot(ref, d['direction']).item()
            aligned.append(d['direction'] * (1 if cos > 0 else -1))
        mean_dir = torch.stack(aligned).mean(dim=0)
        mean_dir = mean_dir / mean_dir.norm()
        mean_dirs[li] = mean_dir
    
    print(f"  Computed mean correction directions for {len(mean_dirs)} layers")
else:
    print(f"\n  Not enough stable layers for correction test.")

# Save
res = {}
for li, sr in stability_results.items():
    res[f'L{li}'] = {k: v for k, v in sr.items() if k != 'stable'}
    res[f'L{li}']['stable'] = sr['stable']
res['summary'] = {'stable': n_stable, 'partial': n_partial, 'unstable': n_unstable}

with open(os.path.join(results_dir, 'phase10l_output_correction.json'), 'w') as f:
    json.dump(res, f, indent=2)

print(f"\nSaved to results/phase10l_output_correction.json")
print("="*80+"\n  DONE\n"+"="*80)
