#!/usr/bin/env python3
"""
Phase 10i: Stereo Decomposition — A_p vs B_p spectrum analysis
Each RoPE pair: c_p = A_p·cos(δθ) + B_p·sin(δθ)
  A_p = content inner product (q·k), B_p = content cross product (q×k)
Hypothesis: A concentrates, B is flat (reference channel like stereo error E).
"""
import sys; sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, json, os, math
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)
print("="*80); print("  PHASE 10i: STEREO A/B DECOMPOSITION"); print("="*80)

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

NL=28; NH=28; NKV=4; HD=128; HPK=7; HDIM=3584; NP=64; ROPE_THETA=1e6
inv_freq = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,dtype=torch.float32)/HD))
phi_levels = torch.log(inv_freq)/math.log(PHI)

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

# Collect A_p and B_p energies
PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is","The color of grass is"]

A_en = defaultdict(lambda: np.zeros(NP)); B_en = defaultdict(lambda: np.zeros(NP))

for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    layer_h={}
    def cap(li):
        def hf(mod, args, kw, out):
            h=args[0] if args else kw.get('hidden_states')
            if h is not None: layer_h[li]=h[0].cpu().float()
            return out
        return hf
    hooks=[model.model.layers[li].self_attn.register_forward_hook(cap(li),with_kwargs=True) for li in range(NL)]
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()

    for li in layer_h:
        hs=layer_h[li]; s=hs.shape[0]; sc=1.0/math.sqrt(HD)
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_data: continue
            d=head_data[(li,hd)]
            qc=(d['W_q']@hs.T).T; kc=(d['W_k']@hs.T).T
            for i in range(s):
                for j in range(i+1):
                    for p in range(NP):
                        q0,q1=qc[i,2*p].item(),qc[i,2*p+1].item()
                        k0,k1=kc[j,2*p].item(),kc[j,2*p+1].item()
                        A_en[(li,hd)][p] += abs((q0*k0+q1*k1)*sc)
                        B_en[(li,hd)][p] += abs((q1*k0-q0*k1)*sc)

# Analyze
all_A=np.zeros(NP); all_B=np.zeros(NP); nh=0
conc_A=[]; conc_B=[]
for (li,hd) in A_en:
    ae,be = A_en[(li,hd)], B_en[(li,hd)]
    at,bt = ae.sum(), be.sum()
    if at>0 and bt>0:
        an,bn = ae/at, be/bt; all_A+=an; all_B+=bn; nh+=1
        sa=np.argsort(-an); ca=np.cumsum(an[sa]); conc_A.append(np.searchsorted(ca,0.80)+1)
        sb=np.argsort(-bn); cb=np.cumsum(bn[sb]); conc_B.append(np.searchsorted(cb,0.80)+1)
all_A/=max(nh,1); all_B/=max(nh,1)
sa=np.argsort(-all_A); ca=np.cumsum(all_A[sa])
sb=np.argsort(-all_B); cb=np.cumsum(all_B[sb])

print("A_p (content inner product) — pairs for thresholds:")
for t,l in [(0.50,'50%'),(0.80,'80%'),(0.90,'90%'),(0.95,'95%')]:
    print(f"  {l}: {np.searchsorted(ca,t)+1} pairs")
print("\nB_p (content cross product) — pairs for thresholds:")
for t,l in [(0.50,'50%'),(0.80,'80%'),(0.90,'90%'),(0.95,'95%')]:
    print(f"  {l}: {np.searchsorted(cb,t)+1} pairs")

print(f"\nPer-head concentration (mean pairs for 80%):")
print(f"  A_p: {np.mean(conc_A):.1f} ± {np.std(conc_A):.1f}")
print(f"  B_p: {np.mean(conc_B):.1f} ± {np.std(conc_B):.1f}")

print(f"\nTop-5 A pairs: ",end="")
for r in range(5): p=sa[r]; print(f"p{p}({all_A[p]*100:.1f}%) ",end="")
print(f"\nTop-5 B pairs: ",end="")
for r in range(5): p=sb[r]; print(f"p{p}({all_B[p]*100:.1f}%) ",end="")

At=sum(A_en[k].sum() for k in A_en); Bt=sum(B_en[k].sum() for k in B_en)
print(f"\n\nTotal energy A/B = {At/max(Bt,1e-30):.3f}")

# Per-layer
print("\nPer-layer A/B concentration (pairs for 80%):")
la=defaultdict(list); lb=defaultdict(list)
for (li,hd) in A_en:
    ae,be=A_en[(li,hd)],B_en[(li,hd)]; at,bt=ae.sum(),be.sum()
    if at>0:
        an=ae/at; s=np.argsort(-an); c=np.cumsum(an[s]); la[li].append(np.searchsorted(c,0.80)+1)
    if bt>0:
        bn=be/bt; s=np.argsort(-bn); c=np.cumsum(bn[s]); lb[li].append(np.searchsorted(c,0.80)+1)
for li in sorted(la):
    print(f"  L{li:2d}: A={np.mean(la[li]):.0f}p  B={np.mean(lb[li]):.0f}p  diff={np.mean(la[li])-np.mean(lb[li]):+.0f}")

# Save
res={'A_pairs_80':int(np.searchsorted(ca,0.80)+1),'B_pairs_80':int(np.searchsorted(cb,0.80)+1),
     'A_mean_conc':float(np.mean(conc_A)),'B_mean_conc':float(np.mean(conc_B)),
     'energy_ratio_A_over_B':float(At/max(Bt,1e-30))}
with open(os.path.join(results_dir,'phase10i_stereo.json'),'w') as f: json.dump(res,f,indent=2)
print(f"\nSaved to results/phase10i_stereo.json")
print("="*80+"\n  DONE\n"+"="*80)
