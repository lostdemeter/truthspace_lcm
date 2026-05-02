import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

def normed(v): return v/(np.linalg.norm(v)+1e-8)
def get_emb(word):
    for p in [' ','']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids)==1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def compute_axis(pairs):
    chords=[get_emb(t)[0]-get_emb(s)[0] for s,t in pairs
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]
    if not chords: return None, 0.0
    md = normed(np.mean(chords,axis=0))
    coh = np.mean([np.dot(normed(c), md) for c in chords])
    return md, float(coh)

CAPITAL_PAIRS  = [('france','paris'),('germany','berlin'),('japan','tokyo'),('china','beijing'),('italy','rome'),('spain','madrid'),('russia','moscow'),('india','delhi'),('brazil','brasilia'),('canada','ottawa'),('egypt','cairo'),('greece','athens'),('turkey','ankara'),('poland','warsaw'),('sweden','stockholm'),('austria','vienna')]
LANGUAGE_PAIRS = [('france','french'),('germany','german'),('spain','spanish'),('russia','russian'),('japan','japanese'),('china','chinese'),('italy','italian'),('portugal','portuguese'),('poland','polish'),('sweden','swedish'),('norway','norwegian'),('denmark','danish'),('greece','greek'),('turkey','turkish'),('finland','finnish'),('hungary','hungarian')]
CURRENCY_PAIRS = [('japan','yen'),('china','yuan'),('russia','ruble'),('india','rupee'),('brazil','real'),('mexico','peso'),('sweden','krona'),('norway','krone'),('denmark','krone'),('poland','zloty'),('turkey','lira'),('egypt','pound')]
SCIENTIST_NAT  = [('Einstein','German'),('Newton','British'),('Darwin','British'),('Kepler','German'),('Euler','Swiss'),('Gauss','German'),('Turing','British'),('Tesla','Serbian'),('Napoleon','French'),('Churchill','British')]
SCIENTIST_FIELD= [('Einstein','physicist'),('Newton','physicist'),('Darwin','biologist'),('Turing','mathematician'),('Euler','mathematician'),('Gauss','mathematician'),('Tesla','inventor'),('Marx','philosopher'),('Freud','psychologist'),('Kant','philosopher')]
LEADER_NAT     = [('Napoleon','French'),('Churchill','British'),('Lincoln','American'),('Gandhi','Indian'),('Mandela','African'),('Caesar','Roman'),('Aristotle','Greek'),('Plato','Greek'),('Shakespeare','British'),('Mozart','Austrian')]
ELEMENT_TYPE   = [('hydrogen','gas'),('oxygen','gas'),('carbon','solid'),('nitrogen','gas'),('calcium','metal'),('sodium','metal'),('iron','metal'),('copper','metal'),('zinc','metal'),('gold','metal'),('silver','metal'),('silicon','solid'),('sulfur','solid'),('helium','gas'),('neon','gas'),('argon','gas'),('lithium','metal'),('aluminum','metal'),('magnesium','metal'),('chlorine','gas')]
ELEMENT_SYM    = [('hydrogen','element'),('oxygen','element'),('carbon','element'),('nitrogen','element'),('calcium','element'),('sodium','element'),('iron','element'),('copper','element'),('zinc','element'),('gold','element'),('silver','element'),('silicon','element'),('sulfur','element'),('helium','element'),('neon','element'),('argon','element'),('lithium','element'),('aluminum','element'),('magnesium','element'),('chlorine','element')]
COMPANY_IND    = [('Apple','technology'),('Google','technology'),('Microsoft','software'),('Amazon','retail'),('Tesla','automotive'),('Samsung','electronics'),('Sony','electronics'),('Nike','sports'),('Boeing','aerospace'),('Intel','semiconductor'),('IBM','technology'),('Adobe','software'),('Oracle','software'),('Cisco','networking'),('Dell','computers'),('Nvidia','semiconductor'),('Uber','transportation'),('Spotify','music'),('Twitter','social'),('Alibaba','commerce')]
COMPANY_PRD    = [('Apple','iPhone'),('Google','search'),('Microsoft','Windows'),('Amazon','shopping'),('Tesla','cars'),('Samsung','phones'),('Sony','PlayStation'),('Nike','shoes'),('Boeing','aircraft'),('Intel','chips'),('IBM','computers'),('Adobe','Photoshop'),('Oracle','database'),('Cisco','routers'),('Dell','laptops')]

ax_cap,_=compute_axis(CAPITAL_PAIRS); ax_lan,_=compute_axis(LANGUAGE_PAIRS); ax_cur,_=compute_axis(CURRENCY_PAIRS)
ax_sn,_=compute_axis(SCIENTIST_NAT); ax_sf,_=compute_axis(SCIENTIST_FIELD); ax_ln,_=compute_axis(LEADER_NAT)
ax_et,_=compute_axis(ELEMENT_TYPE); ax_es,_=compute_axis(ELEMENT_SYM)
ax_ci,_=compute_axis(COMPANY_IND); ax_cp,_=compute_axis(COMPANY_PRD)

_,_,Vtc=np.linalg.svd(np.stack([ax_cap,ax_lan,ax_cur]),full_matrices=False); country_axis=Vtc[0]
_,_,Vtp=np.linalg.svd(np.stack([ax_sn,ax_sf,ax_ln]),full_matrices=False); person_axis=Vtp[0]
_,_,Vte=np.linalg.svd(np.stack([ax_et,ax_es]),full_matrices=False); element_axis=Vte[0]
_,_,Vtco=np.linalg.svd(np.stack([ax_ci,ax_cp]),full_matrices=False); company_axis=Vtco[0]
_,S_ne,Vt_ne=np.linalg.svd(np.stack([country_axis,person_axis,element_axis,company_axis]),full_matrices=False)
ne_axis=Vt_ne[0]

print('DAY 275: NE AXIS PRECISION/RECALL ON FULL VOCABULARY')
print('='*60)
print()

ne_projs = W_n @ ne_axis.astype(np.float32)
vocab_size = W_E.shape[0]

all_tokens = []
for i in range(vocab_size):
    try:
        s = tok.decode([i])
        all_tokens.append((i, s, float(ne_projs[i])))
    except Exception:
        all_tokens.append((i, '', float(ne_projs[i])))

def looks_like_ne(s):
    s2 = s.strip()
    if len(s2) < 2: return False
    if not s2[0].isupper(): return False
    if not s2.replace('-','').replace("'","").isalpha(): return False
    return True

def looks_like_common(s):
    s2 = s.strip().lower()
    if len(s2) < 3: return False
    if not s2.isalpha(): return False
    return True

mean_v = float(np.mean(ne_projs))
std_v  = float(np.std(ne_projs))
print("NE axis score distribution:")
print("  mean=%.4f  std=%.4f" % (mean_v, std_v))
print("  min=%.4f  max=%.4f" % (float(np.min(ne_projs)), float(np.max(ne_projs))))
print("  p10=%.4f  p25=%.4f" % (float(np.percentile(ne_projs,10)), float(np.percentile(ne_projs,25))))
print("  p50=%.4f  p75=%.4f" % (float(np.percentile(ne_projs,50)), float(np.percentile(ne_projs,75))))
print("  p90=%.4f  p95=%.4f" % (float(np.percentile(ne_projs,90)), float(np.percentile(ne_projs,95))))
print()

for threshold in [-0.10, -0.15, -0.20, -0.25]:
    retrieved   = [(i,s,p) for i,s,p in all_tokens if p < threshold]
    cap_alpha   = [(i,s,p) for i,s,p in retrieved if looks_like_ne(s)]
    lower_alpha = [(i,s,p) for i,s,p in retrieved if looks_like_common(s)]
    punct_num   = [(i,s,p) for i,s,p in retrieved if not looks_like_ne(s) and not looks_like_common(s)]
    n = max(len(retrieved), 1)
    print("Threshold < %.2f:" % threshold)
    print("  Retrieved: %5d / %d (%.1f%%)" % (len(retrieved), vocab_size, 100*len(retrieved)/vocab_size))
    print("  Capitalized+alpha (NE-like):   %5d (%.1f%%)" % (len(cap_alpha),   100*len(cap_alpha)/n))
    print("  Lowercase alpha (common-like): %5d (%.1f%%)" % (len(lower_alpha), 100*len(lower_alpha)/n))
    print("  Other (punct/num/mixed):       %5d (%.1f%%)" % (len(punct_num),   100*len(punct_num)/n))
    top20 = sorted(retrieved, key=lambda x: x[2])[:20]
    words20 = [(s.strip(), round(p,3)) for _,s,p in top20]
    print("  Bottom 20 (most negative): %s" % str(words20))
    print()

cap_alpha_all = [(i,s,p) for i,s,p in all_tokens if looks_like_ne(s)]
lower_all     = [(i,s,p) for i,s,p in all_tokens if looks_like_common(s)]
print("Total capitalized-alpha tokens in vocab: %d" % len(cap_alpha_all))
print("Total lowercase-alpha tokens in vocab:   %d" % len(lower_all))
print()
print("Recall of cap-alpha tokens vs FP rate of lower-alpha at each threshold:")
for threshold in [-0.05, -0.10, -0.15, -0.20, -0.25]:
    cap_ret = [(i,s,p) for i,s,p in cap_alpha_all if p < threshold]
    low_ret = [(i,s,p) for i,s,p in lower_all if p < threshold]
    cap_rec = 100*len(cap_ret)/max(len(cap_alpha_all),1)
    low_fp  = 100*len(low_ret)/max(len(lower_all),1)
    print("  thr<%.2f: cap_recall=%.1f%% (%d/%d)  lower_FP=%.1f%% (%d/%d)" % (
        threshold, cap_rec, len(cap_ret), len(cap_alpha_all),
        low_fp, len(low_ret), len(lower_all)))
