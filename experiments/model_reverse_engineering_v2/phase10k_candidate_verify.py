#!/usr/bin/env python3
"""
Phase 10k: Candidate Verification Attack

From Phase 10j findings:
- ΔW is rank-1 dominant (69-93% at every layer)
- The argmax flip rate is 47.5% but the ERROR is low-dimensional
- Drift saturates at ~1.2 (layer norm clamps it)

ATTACK: If the correct key is in the top-K of cheap (bias-aware) scores,
we only need to compute full QK for K candidates per query to verify.

Measurements:
1. Top-K recall: how often is the real argmax in the cheap top-K?
2. Top-K verification: if we recompute full QK for top-K, does stacking work?
3. Cost analysis: K/S savings ratio

Also test: rank-1 output correction
If ΔW ≈ σ u v^T, then Δ(output) = ΔW @ V ≈ σ u (v^T V).
Can we estimate this correction cheaply?
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80)
print("  PHASE 10k: CANDIDATE VERIFICATION + RANK-1 CORRECTION")
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

PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is",
           "The color of grass is","Water freezes at a temperature of",
           "Shakespeare wrote the play","The chemical symbol for gold is",
           "Photosynthesis converts sunlight into","The human heart has four",
           "Mount Everest is the tallest","DNA stands for",
           "In computer science, an algorithm is",
           "The speed of light in vacuum is","The Great Wall of China was built"]

# ================================================================
# MEASUREMENT 1: Top-K recall of cheap scores
# ================================================================
print("="*80)
print("  MEASUREMENT 1: Top-K recall — is correct argmax in cheap top-K?")
print("="*80+"\n")

recall_by_K = defaultdict(list)  # K -> list of recall values

for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    
    # Capture real attention
    layer_h={}; real_scores={}
    def cap(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16)
            Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            sc={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                sc[hd]=scores.cpu()
            real_scores[li]=sc; layer_h[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    # For each routing head at each layer, compute cheap scores and check recall
    for li in range(NL):
        if li not in layer_h: continue
        hs = layer_h[li]; s = hs.shape[0]
        
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl = head_tables[(li,hd)]
            
            # Last query position only (the one that matters for generation)
            i = s - 1
            
            # Bias-aware scores for this query
            cheap_scores = torch.full((s,), float('-inf'))
            for j in range(i+1):
                delta = i-j
                bl = tbl['baseline'][delta].item()
                cqv = (hs[i] @ tbl['c_q'][delta]).item()
                ckv = (tbl['c_k'][delta] @ hs[j]).item()
                cheap_scores[j] = bl + cqv + ckv
            
            # Real scores for this query
            real_row = real_scores[li][hd][i, :s]
            
            # Real argmax
            real_w = phi_softmax(real_row.unsqueeze(0).float())[0]
            real_argmax = real_w[:i+1].argmax().item()
            
            # Check top-K recall
            for K in [1, 2, 3, 5]:
                if i + 1 <= K:
                    recall_by_K[K].append(1.0)  # trivial
                else:
                    cheap_topK = set(cheap_scores[:i+1].topk(min(K, i+1)).indices.tolist())
                    recall_by_K[K].append(1.0 if real_argmax in cheap_topK else 0.0)

print(f"  Top-K recall (correct argmax in cheap top-K):")
print(f"  {'K':>4s}  {'Recall':>8s}  {'N':>6s}")
print("  "+"-"*22)
for K in [1, 2, 3, 5]:
    vals = recall_by_K[K]
    print(f"  K={K}:   {np.mean(vals)*100:5.1f}%    {len(vals)}")

# Per-layer top-2 recall
print(f"\n  Per-layer top-2 recall:")
layer_recall = defaultdict(list)
for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    layer_h2={}; real_scores2={}
    def cap2(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16)
            Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            sc={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                sc[hd]=scores.cpu()
            real_scores2[li]=sc; layer_h2[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap2(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    for li in range(NL):
        if li not in layer_h2: continue
        hs=layer_h2[li]; s=hs.shape[0]; i=s-1
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl=head_tables[(li,hd)]
            cheap=torch.full((s,),float('-inf'))
            for j in range(i+1):
                delta=i-j; bl=tbl['baseline'][delta].item()
                cqv=(hs[i]@tbl['c_q'][delta]).item()
                ckv=(tbl['c_k'][delta]@hs[j]).item()
                cheap[j]=bl+cqv+ckv
            real_row=real_scores2[li][hd][i,:s]
            real_w=phi_softmax(real_row.unsqueeze(0).float())[0]
            ra=real_w[:i+1].argmax().item()
            if i+1<=2:
                layer_recall[li].append(1.0)
            else:
                top2=set(cheap[:i+1].topk(min(2,i+1)).indices.tolist())
                layer_recall[li].append(1.0 if ra in top2 else 0.0)

for li in sorted(layer_recall):
    vals=layer_recall[li]
    r=np.mean(vals)*100
    if r < 90:
        print(f"    L{li:2d}: {r:5.1f}% ({len(vals)} heads) ← LOW")
    else:
        print(f"    L{li:2d}: {r:5.1f}%")

# ================================================================
# MEASUREMENT 2: What's the score GAP between rank-1 and rank-2?
# If the gap is large, the argmax is robust. If small, it's fragile.
# ================================================================
print(f"\n{'='*80}")
print("  MEASUREMENT 2: Score gap between top-1 and top-2 (fragility)")
print("="*80+"\n")

gaps_real=[]; gaps_cheap=[]
for prompt in PROMPTS[:5]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    layer_h3={}; real_scores3={}
    def cap3(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16)
            Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            sc={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                sc[hd]=scores.cpu()
            real_scores3[li]=sc; layer_h3[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap3(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    for li in range(NL):
        if li not in layer_h3: continue
        hs=layer_h3[li]; s=hs.shape[0]; i=s-1
        if i < 1: continue
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl=head_tables[(li,hd)]
            # Cheap scores
            cheap=torch.zeros(i+1)
            for j in range(i+1):
                delta=i-j
                cheap[j]=tbl['baseline'][delta].item()+(hs[i]@tbl['c_q'][delta]).item()+(tbl['c_k'][delta]@hs[j]).item()
            # Real scores
            real_row=real_scores3[li][hd][i,:i+1]
            
            if i >= 1:
                rtop2=real_row.topk(min(2,i+1))
                ctop2=cheap.topk(min(2,i+1))
                if len(rtop2.values) >= 2:
                    gaps_real.append((rtop2.values[0]-rtop2.values[1]).item())
                    gaps_cheap.append((ctop2.values[0]-ctop2.values[1]).item())

print(f"  Real QK score gap (top1-top2):  mean={np.mean(gaps_real):.3f}  median={np.median(gaps_real):.3f}")
print(f"  Cheap score gap (top1-top2):    mean={np.mean(gaps_cheap):.3f}  median={np.median(gaps_cheap):.3f}")
print(f"  Ratio cheap/real:               {np.mean(gaps_cheap)/max(np.mean(gaps_real),1e-10):.3f}")

# Distribution of real gaps
print(f"\n  Real gap percentiles:")
pcts = np.percentile(gaps_real, [10, 25, 50, 75, 90])
for p,v in zip([10,25,50,75,90], pcts):
    print(f"    p{p}: {v:.3f}")

# How many are "fragile" (gap < 1.0)?
n_fragile = sum(1 for g in gaps_real if g < 1.0)
print(f"\n  Fragile decisions (gap < 1.0): {n_fragile}/{len(gaps_real)} ({n_fragile/max(len(gaps_real),1)*100:.1f}%)")

# ================================================================
# MEASUREMENT 3: Top-K recall at ALL query positions (not just last)
# ================================================================
print(f"\n{'='*80}")
print("  MEASUREMENT 3: Top-K recall at ALL query positions")
print("="*80+"\n")

all_pos_recall = defaultdict(list)
for prompt in PROMPTS[:5]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    layer_h4={}; real_scores4={}
    def cap4(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16)
            Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            sc={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                sc[hd]=scores.cpu()
            real_scores4[li]=sc; layer_h4[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap4(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()
    
    for li in range(NL):
        if li not in layer_h4: continue
        hs=layer_h4[li]; s=hs.shape[0]
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl=head_tables[(li,hd)]
            for i in range(s):
                if i < 1: continue  # Position 0 has only 1 key
                cheap=torch.full((s,),float('-inf'))
                for j in range(i+1):
                    delta=i-j
                    cheap[j]=tbl['baseline'][delta].item()+(hs[i]@tbl['c_q'][delta]).item()+(tbl['c_k'][delta]@hs[j]).item()
                real_row=real_scores4[li][hd][i,:s]
                real_w=phi_softmax(real_row.unsqueeze(0).float())[0]
                ra=real_w[:i+1].argmax().item()
                for K in [2, 3]:
                    topK=set(cheap[:i+1].topk(min(K,i+1)).indices.tolist())
                    all_pos_recall[K].append(1.0 if ra in topK else 0.0)

for K in [2, 3]:
    vals = all_pos_recall[K]
    print(f"  Top-{K} recall (all positions): {np.mean(vals)*100:.1f}% ({len(vals)} samples)")

# Save
res = {}
for K in [1,2,3,5]:
    if K in recall_by_K:
        res[f'top{K}_recall'] = float(np.mean(recall_by_K[K]))
res['gap_real_mean'] = float(np.mean(gaps_real))
res['gap_cheap_mean'] = float(np.mean(gaps_cheap))
res['fragile_pct'] = float(n_fragile/max(len(gaps_real),1))
for K in [2,3]:
    if K in all_pos_recall:
        res[f'all_pos_top{K}_recall'] = float(np.mean(all_pos_recall[K]))

with open(os.path.join(results_dir, 'phase10k_candidate_verify.json'), 'w') as f:
    json.dump(res, f, indent=2)

print(f"\nSaved to results/phase10k_candidate_verify.json")
print("="*80+"\n  DONE\n"+"="*80)
