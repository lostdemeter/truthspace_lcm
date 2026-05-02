#!/usr/bin/env python3
"""
Phase 10i Part 2: Stacking Drift Diagnosis

Why does 0.997 score correlation fail when stacked across 28 layers?
Measure the amplification chain: score → softmax → weights → V-output → h_next

Also test: does including ww term (full QK) at runtime fix things?
The bias-aware decomposition omits the ww term. If that's the stacking bottleneck,
we need to find a way to cheaply approximate it.

Key measurement: per-layer attention weight correlation and V-output cosine sim
when using bias-aware (no ww) vs real QK.
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80); print("  PHASE 10i-2: STACKING DRIFT DIAGNOSIS"); print("="*80)

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
print("Extracting weights...")
head_data = {}
for li in range(NL):
    attn = model.model.layers[li].self_attn
    ident = torch.eye(HDIM, device="cuda", dtype=torch.bfloat16)
    Wq = torch.zeros(NH,HD,HDIM,dtype=torch.float32); Wk = torch.zeros(NKV,HD,HDIM,dtype=torch.float32)
    for s in range(0,HDIM,512):
        e = min(s+512,HDIM); chunk = ident[s:e].unsqueeze(0)
        with torch.no_grad(): qo=attn.q_proj(chunk).float(); ko=attn.k_proj(chunk).float()
        qr=qo[0].reshape(-1,NH,HD); kr=ko[0].reshape(-1,NKV,HD)
        for h in range(NH): Wq[h,:,s:e]=qr[:,h,:].T
        for g in range(NKV): Wk[g,:,s:e]=kr[:,g,:].T
    zi = torch.zeros(1,1,HDIM,device="cuda",dtype=torch.bfloat16)
    with torch.no_grad(): qb=attn.q_proj(zi).float()[0,0]; kb=attn.k_proj(zi).float()[0,0]
    bq = qb.reshape(NH,HD).cpu(); bk = kb.reshape(NKV,HD).cpu()
    for h in range(NH): Wq[h] -= bq[h].unsqueeze(1)
    for g in range(NKV): Wk[g] -= bk[g].unsqueeze(1)
    for h in layer_cls[li]['routing']:
        g = h//HPK
        head_data[(li,h)] = {'W_q':Wq[h].clone(),'W_k':Wk[g].clone(),'b_q':bq[h].clone(),'b_k':bk[g].clone()}
    del Wq,Wk; torch.cuda.empty_cache()
    if li%7==0: print(f"  Layer {li} done")
print(f"  {len(head_data)} routing heads\n")

# Pre-compute bias tables
print("Pre-computing bias tables...")
head_tables = {}
for (li,h), d in head_data.items():
    bl=torch.zeros(MAXS); cq=torch.zeros(MAXS,HDIM); ck=torch.zeros(MAXS,HDIM)
    sc = 1.0/math.sqrt(HD)
    for delta in range(MAXS):
        fd=delta*inv_freq; cd=torch.cat((fd.cos(),fd.cos())); sd=torch.cat((fd.sin(),fd.sin()))
        bkg=d['b_k']; b1,b2=bkg[:HD//2],bkg[HD//2:]
        bkr=bkg*cd+torch.cat((-b2,b1))*sd
        Wkg=d['W_k']; W1,W2=Wkg[:HD//2,:],Wkg[HD//2:,:]
        Wkr=Wkg*cd.unsqueeze(1)+torch.cat((-W2,W1),dim=0)*sd.unsqueeze(1)
        bl[delta]=(d['b_q']@bkr)*sc; cq[delta]=(d['W_q'].T@bkr)*sc; ck[delta]=(Wkr.T@d['b_q'])*sc
    head_tables[(li,h)] = {'baseline':bl,'c_q':cq,'c_k':ck}
print("  Done\n")

# ================================================================
# Capture real model state
# ================================================================
PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is",
           "The color of grass is","Water freezes at a temperature of",
           "The speed of light in vacuum is","Shakespeare wrote the play",
           "The chemical symbol for gold is","Photosynthesis converts sunlight into",
           "The Great Wall of China was built","In computer science, an algorithm is",
           "The human heart has four","Mount Everest is the tallest","DNA stands for"]

print("="*80)
print("  AMPLIFICATION CHAIN: score → weight → argmax")
print("="*80+"\n")

for prompt in PROMPTS[:3]:  # Detailed analysis on 3 prompts
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    print(f"  Prompt: '{prompt}' ({slen} tokens)")

    layer_h={}; real_scores={}; real_weights={}
    def cap(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            sc={};wt={}
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                w=phi_softmax(scores,dim=-1)
                sc[hd]=scores.cpu(); wt[hd]=w.cpu()
            real_scores[li]=sc; real_weights[li]=wt; layer_h[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()

    print(f"  {'Layer':>6s}  {'ScoreCorr':>9s}  {'WeightCorr':>10s}  {'Argmax%':>7s}  {'MaxWtErr':>8s}  {'wwFrac':>7s}")
    print("  "+"-"*55)

    for li in range(NL):
        if li not in layer_h: continue
        hs=layer_h[li]; s=hs.shape[0]
        sc_c=[]; wt_c=[]; am_c=[]; mw_c=[]; ww_f=[]
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl=head_tables[(li,hd)]
            d=head_data[(li,hd)]
            # Bias-aware scores (no ww)
            approx=torch.full((s,s),float('-inf'))
            for i in range(s):
                for j in range(i+1):
                    delta=i-j
                    bl=tbl['baseline'][delta].item()
                    cqv=(hs[i]@tbl['c_q'][delta]).item()
                    ckv=(tbl['c_k'][delta]@hs[j]).item()
                    approx[i,j]=bl+cqv+ckv
            approx_w=phi_softmax(approx.float(),dim=-1)
            real_w=real_weights[li][hd]

            # Score correlation (finite values only)
            mk=torch.triu(torch.ones(s,s),diagonal=1).bool()
            mask=~mk
            rs=real_scores[li][hd][mask].numpy(); aps=approx[mask].numpy()
            fin=np.isfinite(rs)&np.isfinite(aps)
            if fin.sum()>2 and rs[fin].std()>1e-10: sc_c.append(np.corrcoef(rs[fin],aps[fin])[0,1])

            # Weight correlation
            rw=real_w[mask].numpy(); aw=approx_w[mask].numpy()
            if rw.std()>1e-10: wt_c.append(np.corrcoef(rw,aw)[0,1])

            # Argmax agreement
            for i in range(s):
                ri=real_w[i,:i+1].argmax().item(); ai=approx_w[i,:i+1].argmax().item()
                am_c.append(1.0 if ri==ai else 0.0)

            # Max weight error
            mw_c.append((real_w-approx_w).abs().max().item())

            # ww fraction: compute actual ww for last query
            i=s-1
            ww_total=0; full_total=0
            for j in range(i+1):
                delta=i-j; rsc=real_scores[li][hd][i,j].item()
                bl=tbl['baseline'][delta].item()
                cqv=(hs[i]@tbl['c_q'][delta]).item()
                ckv=(tbl['c_k'][delta]@hs[j]).item()
                ww=rsc-(bl+cqv+ckv)
                ww_total+=abs(ww); full_total+=abs(rsc)
            if full_total>0: ww_f.append(ww_total/full_total)

        if sc_c:
            print(f"  L{li:2d}:    {np.mean(sc_c):.4f}      {np.mean(wt_c):.4f}   {np.mean(am_c)*100:5.1f}%   "
                  f"{np.mean(mw_c):.4f}    {np.mean(ww_f)*100:.2f}%")
    print()

# ================================================================
# KEY TEST: Score error vs weight error distribution
# ================================================================
print("="*80)
print("  CRITICAL: How does score error translate to weight error?")
print("  Measuring at the LAST position (prediction target)")
print("="*80+"\n")

all_score_err=[]; all_weight_err=[]; all_argmax_flip=[]
all_ww_frac=[]; all_ww_magnitude=[]

for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    layer_h={}; real_scores={}; real_weights={}
    def cap2(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is None: return out
            b,s,_=h.shape
            with torch.no_grad():
                Q=mod.q_proj(h).to(torch.bfloat16); K=mod.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            sc={};wt={}
            mk=torch.triu(torch.ones(s,s,device=h.device),diagonal=1).bool()
            for hd in range(NH):
                scores=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float()
                scores.masked_fill_(mk,float('-inf'))
                sc[hd]=scores.cpu(); wt[hd]=phi_softmax(scores,dim=-1).cpu()
            real_scores[li]=sc; real_weights[li]=wt; layer_h[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap2(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()

    i=slen-1  # last position
    for li in range(NL):
        if li not in layer_h: continue
        hs=layer_h[li]; s=hs.shape[0]
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_tables: continue
            tbl=head_tables[(li,hd)]
            # Bias-aware scores for last query
            approx_row=torch.full((s,),float('-inf'))
            real_row=real_scores[li][hd][i,:s]
            ww_vals=[]
            for j in range(i+1):
                delta=i-j
                bl=tbl['baseline'][delta].item()
                cqv=(hs[i]@tbl['c_q'][delta]).item()
                ckv=(tbl['c_k'][delta]@hs[j]).item()
                approx_row[j]=bl+cqv+ckv
                ww=real_row[j].item()-(bl+cqv+ckv)
                ww_vals.append(abs(ww))

            # Score error (RMSE)
            valid=torch.isfinite(real_row)&torch.isfinite(approx_row)
            se=((real_row[valid]-approx_row[valid])**2).mean().sqrt().item()
            all_score_err.append(se)

            # Weight error
            rw=phi_softmax(real_row.unsqueeze(0).float(),dim=-1)[0]
            aw=phi_softmax(approx_row.unsqueeze(0).float(),dim=-1)[0]
            we=(rw-aw).abs().max().item()
            all_weight_err.append(we)

            # Argmax flip
            ra=rw[:i+1].argmax().item(); aa=aw[:i+1].argmax().item()
            all_argmax_flip.append(0.0 if ra==aa else 1.0)

            # ww fraction
            ww_sum=sum(ww_vals); full_sum=real_row[valid].abs().sum().item()
            if full_sum>0:
                all_ww_frac.append(ww_sum/full_sum)
                all_ww_magnitude.append(np.mean(ww_vals))

print(f"  Score RMSE (bias-aware vs real): {np.mean(all_score_err):.6f} ± {np.std(all_score_err):.6f}")
print(f"  Max weight error per head:       {np.mean(all_weight_err):.6f} ± {np.std(all_weight_err):.6f}")
print(f"  Argmax flip rate:                {np.mean(all_argmax_flip)*100:.1f}%")
print(f"  ww fraction of score:            {np.mean(all_ww_frac)*100:.3f}%")
print(f"  ww magnitude (mean abs):         {np.mean(all_ww_magnitude):.6f}")

# Percentiles
print(f"\n  Weight error percentiles:")
pcts = np.percentile(all_weight_err, [50,75,90,95,99])
for p,v in zip([50,75,90,95,99],pcts): print(f"    p{p}: {v:.6f}")

print(f"\n  Score error percentiles:")
pcts = np.percentile(all_score_err, [50,75,90,95,99])
for p,v in zip([50,75,90,95,99],pcts): print(f"    p{p}: {v:.6f}")

# How many heads have argmax flips per layer
print(f"\n  Argmax flips per layer (at last token):")
layer_flips=defaultdict(list)
idx=0
for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    for li in range(NL):
        for hd in layer_cls[li]['routing']:
            if (li,hd) in head_tables:
                if idx<len(all_argmax_flip):
                    layer_flips[li].append(all_argmax_flip[idx]); idx+=1
for li in sorted(layer_flips):
    flips=layer_flips[li]
    n_flips=sum(flips); total=len(flips)
    if n_flips>0: print(f"    L{li:2d}: {n_flips:.0f}/{total} ({n_flips/total*100:.0f}%)")

# Save
res = {
    'score_rmse_mean': float(np.mean(all_score_err)),
    'weight_err_mean': float(np.mean(all_weight_err)),
    'argmax_flip_rate': float(np.mean(all_argmax_flip)),
    'ww_frac_mean': float(np.mean(all_ww_frac)),
    'ww_magnitude_mean': float(np.mean(all_ww_magnitude)),
}
with open(os.path.join(results_dir,'phase10i_stacking.json'),'w') as f:
    json.dump(res,f,indent=2)
print(f"\nSaved to results/phase10i_stacking.json")
print("="*80+"\n  DONE\n"+"="*80)
