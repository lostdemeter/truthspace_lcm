#!/usr/bin/env python3
"""
Phase 10h: RoPE Frequency-Pair Decomposition

128 head dims = 64 rotation pairs at φ-scaled frequencies.
Each pair contributes independently. Question: how many carry the ww signal?
"""
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')
import numpy as np, torch, torch.nn.functional as F, json, os, math
from collections import defaultdict, Counter

PHI = (1 + np.sqrt(5)) / 2; LOG_PHI = np.log(PHI)

print("=" * 80)
print("  PHASE 10h: RoPE FREQUENCY-PAIR DECOMPOSITION")
print("=" * 80)

results_dir = os.path.join(os.path.dirname(__file__), 'results')
with open(os.path.join(results_dir, 'phase9a_all_layer_attention.json')) as f:
    phase9a = json.load(f)
layer_cls = {}
for ls in phase9a['layer_summary']:
    layer_cls[ls['layer']] = {'fixed': set(ls['fixed_heads']), 'routing': set(ls['routing_heads'])}

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
model.eval()

NL=28; NH=28; NKV=4; HD=128; HPK=7; HDIM=3584; NP=64; ROPE_THETA=1e6; MAXS=64

def phi_softmax(s, dim=-1):
    s = s - s.max(dim=dim, keepdim=True).values
    p = PHI ** (s / LOG_PHI)
    return p / p.sum(dim=dim, keepdim=True)

def apply_rope(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return x*cos + torch.cat((-x2, x1), dim=-1)*sin

def rope_cache(slen, dev, dt):
    inv = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,device=dev,dtype=torch.float32)/HD))
    pos = torch.arange(slen, device=dev, dtype=torch.float32)
    f = torch.outer(pos, inv); e = torch.cat((f,f), dim=-1)
    return e.cos().to(dt)[None,None], e.sin().to(dt)[None,None]

inv_freq = 1.0/(ROPE_THETA**(torch.arange(0,HD,2,dtype=torch.float32)/HD))
phi_levels = torch.log(inv_freq)/math.log(PHI)
print(f"  64 pairs, φ-level range: [{phi_levels[0]:.2f}, {phi_levels[-1]:.2f}], span={phi_levels[0]-phi_levels[-1]:.2f}")

# Extract weights
print("Extracting weights...")
head_data = {}; head_tables = {}
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
    sc = 1.0/math.sqrt(HD)
    for h in layer_cls[li]['routing']:
        g = h//HPK
        head_data[(li,h)] = {'W_q':Wq[h].clone(),'W_k':Wk[g].clone(),'b_q':bq[h].clone(),'b_k':bk[g].clone()}
        bl=torch.zeros(MAXS); cq=torch.zeros(MAXS,HDIM); ck=torch.zeros(MAXS,HDIM)
        for d in range(MAXS):
            fd = d*inv_freq; cd=torch.cat((fd.cos(),fd.cos())); sd=torch.cat((fd.sin(),fd.sin()))
            bkg=bk[g]; b1,b2=bkg[:HD//2],bkg[HD//2:]; bkr=bkg*cd+torch.cat((-b2,b1))*sd
            Wkg=Wk[g]; W1,W2=Wkg[:HD//2,:],Wkg[HD//2:,:]
            Wkr=Wkg*cd.unsqueeze(1)+torch.cat((-W2,W1),dim=0)*sd.unsqueeze(1)
            bl[d]=(bq[h]@bkr)*sc; cq[d]=(Wq[h].T@bkr)*sc; ck[d]=(Wkr.T@bq[h])*sc
        head_tables[(li,h)] = {'baseline':bl,'c_q':cq,'c_k':ck}
    del Wq,Wk; torch.cuda.empty_cache()
    if li%7==0: print(f"  Layer {li} done")
print(f"  {len(head_data)} routing heads")

# ================================================================
# ANALYSIS: Per-pair energy + cumulative reconstruction
# ================================================================
print("\n" + "="*80)
print("  ANALYSIS: Per-Pair Energy + Cumulative Reconstruction")
print("="*80 + "\n")

PROMPTS = ["The capital of France is","Albert Einstein developed the theory of",
           "To be or not to","The largest planet in our solar system is","The color of grass is"]

pair_energies = defaultdict(lambda: np.zeros(NP))
# For reconstruction test, pick representative heads
test_heads = []
for li in [0,7,13,20,27]:
    r = sorted(layer_cls[li]['routing'])
    if r: test_heads.append((li,r[0]))

recon_data = defaultdict(lambda: {'ww':[],'pc':[]})  # per test_head

for pi, prompt in enumerate(PROMPTS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    slen = ids.shape[1]
    layer_h={}; real_sc={}
    def cap(li):
        def hf(module, args, kwargs, output):
            h=args[0] if args else kwargs.get('hidden_states')
            if h is None: return output
            b,s,_=h.shape
            with torch.no_grad(): Q=module.q_proj(h).to(torch.bfloat16); K=module.k_proj(h).to(torch.bfloat16)
            Q=Q.reshape(b,s,NH,HD).transpose(1,2); K=K.reshape(b,s,NKV,HD).transpose(1,2)
            c,sn=rope_cache(s,h.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
            Ke=K.repeat_interleave(HPK,dim=1)
            sc={}
            for hd in range(NH): sc[hd]=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float().cpu()
            real_sc[li]=sc; layer_h[li]=h[0].cpu().float()
            return output
        return hf
    hooks=[]
    for li in range(NL): hooks.append(model.model.layers[li].self_attn.register_forward_hook(cap(li),with_kwargs=True))
    with torch.no_grad(): model(ids, return_dict=True)
    for hk in hooks: hk.remove()

    for li in layer_h:
        hs=layer_h[li]; s=hs.shape[0]
        for hd in layer_cls[li]['routing']:
            if (li,hd) not in head_data: continue
            d=head_data[(li,hd)]; tbl=head_tables[(li,hd)]
            qc=(d['W_q']@hs.T).T; kc=(d['W_k']@hs.T).T; sc=1.0/math.sqrt(HD)
            for i in range(s):
                for j in range(i+1):
                    delta=i-j; fd=delta*inv_freq
                    pc=np.zeros(NP)
                    for p in range(NP):
                        cp=math.cos(fd[p].item()); sp=math.sin(fd[p].item())
                        q0,q1=qc[i,2*p].item(),qc[i,2*p+1].item()
                        k0,k1=kc[j,2*p].item(),kc[j,2*p+1].item()
                        pc[p]=(q0*(k0*cp-k1*sp)+q1*(k0*sp+k1*cp))*sc
                    pair_energies[(li,hd)] += np.abs(pc)
                    if (li,hd) in [(th[0],th[1]) for th in test_heads]:
                        bl=tbl['baseline'][delta].item()
                        cqv=(hs[i]@tbl['c_q'][delta]).item()
                        ckv=(tbl['c_k'][delta]@hs[j]).item()
                        ww=real_sc[li][hd][i,j].item()-(bl+cqv+ckv)
                        recon_data[(li,hd)]['ww'].append(ww)
                        recon_data[(li,hd)]['pc'].append(pc)

# Energy analysis
print("Per-pair energy (averaged across all heads):")
all_e = np.zeros(NP); nh=0
per_layer_info = defaultdict(list)
for (li,hd),en in pair_energies.items():
    t=en.sum()
    if t>0:
        n=en/t; all_e+=n; nh+=1
        sp=np.argsort(-n); ce=np.cumsum(n[sp])
        n80=np.searchsorted(ce,0.80)+1; n90=np.searchsorted(ce,0.90)+1
        per_layer_info[li].append((hd,n80,n90,sp[:3].tolist()))
all_e/=max(nh,1)
sp=np.argsort(-all_e); ce=np.cumsum(all_e[sp])

print(f"\nTop-15 pairs (pair_idx, energy%, φ-level, cum%):")
for r in range(15):
    p=sp[r]; print(f"  #{r+1:2d}: pair {p:2d}  {all_e[p]*100:5.2f}%  φ={phi_levels[p]:.2f}  cum={ce[r]*100:.1f}%")

for thr,lbl in [(0.80,'80%'),(0.90,'90%'),(0.95,'95%'),(0.99,'99%')]:
    print(f"  {lbl}: {np.searchsorted(ce,thr)+1} pairs")

print(f"\nPer-layer (pairs for 80%/90%):")
for li in sorted(per_layer_info):
    es=per_layer_info[li]
    a80=np.mean([e[1] for e in es]); a90=np.mean([e[2] for e in es])
    tops=[]; [tops.extend(e[3]) for e in es]
    tp=Counter(tops).most_common(3); ps=','.join(f'p{p}' for p,_ in tp)
    print(f"  L{li:2d}: 80%={a80:.1f}p  90%={a90:.1f}p  top=[{ps}]")

# Reconstruction correlation
print(f"\nCumulative pair reconstruction (corr with actual ww):")
print(f"  {'Head':>10s}  {'All':>6s}  {'1p':>6s}  {'2p':>6s}  {'4p':>6s}  {'8p':>6s}  {'16p':>6s}  {'32p':>6s}")
print("  "+"-"*55)
for li,hd in test_heads:
    rd=recon_data.get((li,hd))
    if not rd or not rd['ww']: continue
    wa=np.array(rd['ww']); pm=np.array(rd['pc'])
    if wa.std()<1e-10: continue
    he=np.abs(pm).mean(axis=0); rp=np.argsort(-he)
    fc=np.corrcoef(wa,pm.sum(axis=1))[0,1]
    parts=[f"{fc:.3f}"]
    for n in [1,2,4,8,16,32]:
        pr=pm[:,rp[:n]].sum(axis=1)
        c=np.corrcoef(wa,pr)[0,1] if pr.std()>1e-10 else 0
        parts.append(f"{c:.3f}")
    print(f"  L{li:2d} h{hd:2d}:  {'  '.join(parts)}")

# ================================================================
# END-TO-END
# ================================================================
print(f"\n{'='*80}")
print("  END-TO-END: Bias + Top-K Pair WW Correction")
print("="*80+"\n")

# Pre-compute dominant pairs per head
head_dom = {}
for (li,hd),en in pair_energies.items():
    head_dom[(li,hd)] = np.argsort(-en).tolist()

def make_pair_attn(npairs):
    def attn_fn(layer_idx, h_normed, attn_module):
        b,slen,_=h_normed.shape; fx=layer_cls[layer_idx]['fixed']; rt=layer_cls[layer_idx]['routing']
        with torch.no_grad(): Vf=attn_module.v_proj(h_normed)
        Vk=Vf.reshape(b,slen,NKV,HD); Ve=Vk.repeat_interleave(HPK,dim=2)
        ao=torch.zeros(b,slen,NH,HD,device=h_normed.device,dtype=h_normed.dtype)
        for h in fx: ao[0,:,h,:]=Ve[0,0,h,:]
        hf=h_normed[0].float().cpu()
        for h in rt:
            tbl=head_tables[(layer_idx,h)]; d=head_data[(layer_idx,h)]
            tp=head_dom.get((layer_idx,h),list(range(NP)))[:npairs]
            # Project h through pair-specific rows
            qp=torch.zeros(slen,npairs,2); kp=torch.zeros(slen,npairs,2)
            for pi,p in enumerate(tp):
                qp[:,pi,0]=hf@d['W_q'][2*p]; qp[:,pi,1]=hf@d['W_q'][2*p+1]
                kp[:,pi,0]=hf@d['W_k'][2*p]; kp[:,pi,1]=hf@d['W_k'][2*p+1]
            sc_val=1.0/math.sqrt(HD); scores=torch.zeros(slen,slen)
            for i in range(slen):
                for j in range(i+1):
                    delta=i-j; bl=tbl['baseline'][delta].item()
                    cqv=(hf[i]@tbl['c_q'][delta]).item(); ckv=(tbl['c_k'][delta]@hf[j]).item()
                    ww=0.0; fd=delta*inv_freq
                    for pi,p in enumerate(tp):
                        cp=math.cos(fd[p].item()); sp=math.sin(fd[p].item())
                        q0,q1=qp[i,pi,0].item(),qp[i,pi,1].item()
                        k0,k1=kp[j,pi,0].item(),kp[j,pi,1].item()
                        ww+=(q0*(k0*cp-k1*sp)+q1*(k0*sp+k1*cp))*sc_val
                    scores[i,j]=bl+cqv+ckv+ww
            scores=scores.to(h_normed.device)
            mask=torch.triu(torch.ones(slen,slen,device=h_normed.device),diagonal=1).bool()
            scores.masked_fill_(mask,float('-inf'))
            w=phi_softmax(scores.float(),dim=-1)
            ao[0,:,h,:]=(w.to(torch.bfloat16)@Ve[0,:,h,:].to(torch.bfloat16)).to(h_normed.dtype)
        return attn_module.o_proj(ao.reshape(b,slen,NH*HD))
    return attn_fn

def attn_real(li, hn, am):
    b,sl,_=hn.shape
    with torch.no_grad(): Q=am.q_proj(hn).to(torch.bfloat16); K=am.k_proj(hn).to(torch.bfloat16); Vf=am.v_proj(hn)
    Q=Q.reshape(b,sl,NH,HD).transpose(1,2); K=K.reshape(b,sl,NKV,HD).transpose(1,2)
    Vk=Vf.reshape(b,sl,NKV,HD); Ve=Vk.repeat_interleave(HPK,dim=2)
    c,sn=rope_cache(sl,hn.device,torch.bfloat16); Q=apply_rope(Q,c,sn); K=apply_rope(K,c,sn)
    Ke=K.repeat_interleave(HPK,dim=1)
    ao=torch.zeros(b,sl,NH,HD,device=hn.device,dtype=hn.dtype)
    mk=torch.triu(torch.ones(sl,sl,device=hn.device),diagonal=1).bool()
    for hd in range(NH):
        sc=(Q[0,hd]@Ke[0,hd].T/math.sqrt(HD)).float(); sc.masked_fill_(mk,float('-inf'))
        w=phi_softmax(sc,dim=-1)
        ao[0,:,hd,:]=(w.to(torch.bfloat16)@Ve[0,:,hd,:].to(torch.bfloat16)).to(hn.dtype)
    return am.o_proj(ao.reshape(b,sl,NH*HD))

def run_hooks(ids, fn_map):
    hooks=[]
    for li,fn in fn_map.items():
        def mh(l,f):
            def hf(mod,args,kw,out):
                h=args[0] if args else kw.get('hidden_states')
                if h is None: return out
                g=f(l,h,mod); return (g,)+out[1:] if isinstance(out,tuple) else g
            return hf
        hooks.append(model.model.layers[li].self_attn.register_forward_hook(mh(li,fn),with_kwargs=True))
    try:
        with torch.no_grad(): logits=model(ids,return_dict=True).logits
    finally:
        for hk in hooks: hk.remove()
    return logits

TEST = ["The capital of France is","The largest ocean is the","The color of grass is",
        "Barack Obama was the","To be or not to","Roses are red, violets are",
        "The speed of light is approximately","Albert Einstein developed the theory of",
        "Water freezes at zero degrees","The chemical symbol for gold is",
        "The largest planet in our solar system is","Shakespeare wrote many",
        "The square root of 144 is","In mathematics, pi is approximately equal to",
        "The color of the sky is usually"]

print("Collecting baselines...")
base_tok=[]
for p in TEST:
    ids=tokenizer.encode(p,return_tensors="pt").to("cuda")
    with torch.no_grad(): base_tok.append(model(ids,return_dict=True).logits[0,-1,:].float().argmax().item())

def evaluate(name, fn_map):
    nm=0; cl=[]
    for pi,p in enumerate(TEST):
        ids=tokenizer.encode(p,return_tensors="pt").to("cuda")
        gl=run_hooks(ids,fn_map)[0,-1,:].float()
        if gl.argmax().item()==base_tok[pi]: nm+=1
        with torch.no_grad(): bl=model(ids,return_dict=True).logits[0,-1,:].float()
        cl.append(F.cosine_similarity(bl.cpu().unsqueeze(0),gl.cpu().unsqueeze(0)).item())
    return nm,len(TEST),float(np.mean(cl))

print(f"\n  {'Config':>55s}  {'Score':>7s}  {'Cos':>7s}")
print("  "+"-"*75)

n,t,c=evaluate("real",{i:attn_real for i in range(NL)})
print(f"  {'A: All real QK':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Stacked: all layers with pair correction
for np_ in [4,8,16,32,64]:
    fn=make_pair_attn(np_)
    n,t,c=evaluate(f"pairs_{np_}",{i:fn for i in range(NL)})
    print(f"  {f'B: All bias + top-{np_} pairs (stacked)':>55s}  {n:2d}/{t:2d}    {c:.4f}")

# Zone-aware: anchor DRUM+every4+MUSIC, pairs elsewhere
anchor=set(range(4))|{7,11,15,19,23,27}
for np_ in [4,8,16,32,64]:
    fn=make_pair_attn(np_)
    cfg={i:attn_real for i in anchor}
    for i in set(range(NL))-anchor: cfg[i]=fn
    n,t,c=evaluate(f"zone_p{np_}",cfg)
    print(f"  {f'C: Zone anchored + top-{np_} pairs':>55s}  {n:2d}/{t:2d}    {c:.4f}")

print()

# Save results
save={'pair_energy_spectrum':all_e[sp].tolist()[:20],
      'pairs_for_80':int(np.searchsorted(ce,0.80)+1),
      'pairs_for_90':int(np.searchsorted(ce,0.90)+1),
      'pairs_for_95':int(np.searchsorted(ce,0.95)+1),
      'phi_levels_top10':[float(phi_levels[sp[i]]) for i in range(10)]}
with open(os.path.join(results_dir,'phase10h_freq_pairs.json'),'w') as f: json.dump(save,f,indent=2)
print(f"  Saved to results/phase10h_freq_pairs.json")
print("="*80)
print("  DONE")
print("="*80)
