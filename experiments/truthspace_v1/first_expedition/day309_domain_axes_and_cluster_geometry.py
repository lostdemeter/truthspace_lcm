import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

def normed(v): return v/(np.linalg.norm(v)+1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, 0.0, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    coh = float(np.mean([np.dot(c, md.astype(np.float32)) for c in cn]))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, coh, valid, pc
def nn_retrieve(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def nn_retrieve_clean(pred_emb, exclude_ids, top_n=5):
    """Exclude capitalized and hyphenated/compound tokens."""
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    for i in range(len(sims)):
        w = tok.decode([i]).strip()
        if not w: sims[i] = -1.0; continue
        if w[0].isupper(): sims[i] = -1.0; continue
        if w.startswith('-') or w.startswith('_'): sims[i] = -1.0; continue
        if len(w) <= 1: sims[i] = -1.0; continue
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def best_scale_clean(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve_clean(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 309: DOMAIN AXES, CLUSTER GEOMETRY, EXTENDED +s, +tion SCALE")
print("="*70)
print()

# ====================================================================
# PART A: DOMAIN-SPECIFIC GENDER AXES
# ====================================================================
print("PART A: Domain-specific gender axes")
print("-"*70)

GENDER_DOMAIN_SETS = {
    'kin': {
        'train': [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                  ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
        'holdout': [('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
                    ('widower','widow'),('grandson','granddaughter'),('godfather','godmother')],
    },
    'titles': {
        'train': [('lord','lady'),('duke','duchess'),('prince','princess'),
                  ('emperor','empress'),('king','queen')],
        'holdout': [('count','countess'),('baron','baroness'),('marquis','marchioness'),
                    ('viscount','viscountess'),('tsar','tsarina')],
    },
    'occupation': {
        'train': [('actor','actress'),('waiter','waitress'),('host','hostess'),
                  ('steward','stewardess'),('heir','heiress')],
        'holdout': [('hero','heroine'),('master','mistress'),('manager','manageress'),
                    ('governor','governess'),('executor','executrix')],
    },
    'animals': {
        'train': [('lion','lioness'),('tiger','tigress'),('stallion','mare'),
                  ('ram','ewe'),('bull','cow')],
        'holdout': [('drake','duck'),('gander','goose'),('tom','tabby'),
                    ('boar','sow'),('buck','doe')],
    },
    'fiction': {
        'train': [('wizard','witch'),('sorcerer','sorceress'),('warlock','witch'),
                  ('prince','princess'),('king','queen')],
        'holdout': [('hero','heroine'),('villain','villainess'),('knight','dame'),
                    ('lord','lady'),('emperor','empress')],
    },
}

print("  %-12s  pc_train  scale   tr_acc  ho_acc  cos_kin" % "domain")
print("  " + "-"*58)
domain_axes = {}
ax_kin, _, valid_kin, _ = compute_axis(GENDER_DOMAIN_SETS['kin']['train'])
for dname, splits in GENDER_DOMAIN_SETS.items():
    ax, _, valid, pc = compute_axis(splits['train'])
    if ax is None: continue
    domain_axes[dname] = ax.astype(np.float64)
    scale, tr_acc = best_scale(ax.astype(np.float32), valid)
    # Holdout
    ho_hits, ho_total = 0, 0
    for s, t in splits['holdout']:
        es, sid = get_emb(s); et, _ = get_emb(t)
        if es is None: continue
        ho_total += 1
        pred = W_E[sid] + scale * ax
        r = nn_retrieve(pred, [sid], top_n=1)
        if et is not None and r[0][0] == t: ho_hits += 1
    cos_kin = float(np.dot(ax, ax_kin)) if ax_kin is not None else 0.0
    print("  %-12s  %.4f    %.3f    %d/%d     %d/%d   %+.4f" %
          (dname, pc, scale, tr_acc, len(valid), ho_hits, ho_total, cos_kin))

print()
print("  Pairwise cosines between domain gender axes:")
dnames = list(domain_axes.keys())
print("  %-12s" % "" + "".join("  %-12s" % d[:10] for d in dnames))
for d1 in dnames:
    row = "  %-12s" % d1
    for d2 in dnames:
        if d1 == d2:
            row += "  [   1.00  ]"
        else:
            c = float(np.dot(domain_axes[d1], domain_axes[d2]))
            row += "  %+.4f      " % c
    print(row)
print()

# Cross-domain transfer
print("  Cross-domain transfer (train on domain A, test on domain B holdout):")
for d_train in ['kin', 'titles', 'animals']:
    if d_train not in domain_axes: continue
    ax_tr = domain_axes[d_train].astype(np.float32)
    sc_tr, _ = best_scale(ax_tr, [v for v in [compute_axis(GENDER_DOMAIN_SETS[d_train]['train'])[2]]][0])
    for d_test in GENDER_DOMAIN_SETS:
        if d_test == d_train: continue
        ax_te = domain_axes.get(d_test)
        if ax_te is None: continue
        hits, total = 0, 0
        for s, t in GENDER_DOMAIN_SETS[d_test]['holdout']:
            es, sid = get_emb(s); et, _ = get_emb(t)
            if es is None: continue
            total += 1
            pred = W_E[sid] + sc_tr * ax_tr
            r = nn_retrieve(pred, [sid], top_n=1)
            if et is not None and r[0][0] == t: hits += 1
        acc = 100*hits/total if total else 0
        print("    train=%s -> test=%s: %d/%d (%.0f%%)" % (d_train, d_test, hits, total, acc))
print()

# ====================================================================
# PART B: CLUSTER GEOMETRY IN mPC SPACE
# ====================================================================
print("PART B: Cluster geometry — train sources projected onto mPCs")
print("-"*70)

# Build morphological PCA (mPC1-5)
ALL_MORPH_PAIRS = [
    ('+er',     [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                 ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner'),
                 ('light','lighter'),('strong','stronger'),('weak','weaker'),('soft','softer')]),
    ('+est',    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                 ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest')]),
    ('er->est', [('faster','fastest'),('slower','slowest'),('taller','tallest'),('shorter','shortest'),
                 ('brighter','brightest'),('darker','darkest'),('deeper','deepest')]),
    ('gender',  [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                 ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')]),
    ('past_irr',[('go','went'),('come','came'),('run','ran'),('see','saw'),
                 ('eat','ate'),('know','knew'),('take','took'),('make','made')]),
    ('+ed',     [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                 ('end','ended'),('look','looked'),('call','called'),('help','helped')]),
    ('+ness',   [('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                 ('bright','brightness'),('mad','madness')]),
    ('+ful',    [('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                 ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful')]),
    ('un-',     [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                 ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap')]),
    ('+ment',   [('achieve','achievement'),('manage','management'),('develop','development'),
                 ('move','movement'),('treat','treatment'),('argue','argument')]),
    ('+s',      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                 ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')]),
    ('+tion',   [('act','action'),('direct','direction'),('collect','collection'),
                 ('connect','connection'),('protect','protection'),('select','selection')]),
]
all_chords = []
for nm, pairs in ALL_MORPH_PAIRS:
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        all_chords.append(normed(et-es).astype(np.float32))

M_mat = np.array(all_chords)
M_c = (M_mat - M_mat.mean(axis=0)).astype(np.float32)
rng = np.random.default_rng(0)
mPCs = []
M_defl = M_c.copy()
for k in range(5):
    vk = rng.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = M_defl.T @ (M_defl @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_defl @ vk
    M_defl = M_defl - np.outer(proj, vk)
    mPCs.append(vk.astype(np.float64))

# Project source words from each gender domain onto mPC1-5
print("  Source word mPC projections by gender domain:")
print("  %-14s  %-14s  mPC1    mPC2    mPC3    mPC4    mPC5" % ("domain", "word"))
print("  " + "-"*66)
for dname in ['kin', 'titles', 'animals', 'fiction', 'occupation']:
    if dname not in GENDER_DOMAIN_SETS: continue
    pairs = GENDER_DOMAIN_SETS[dname]['train']
    src_projs = []
    for s, t in pairs:
        es, sid = get_emb(s)
        if es is None: continue
        en = normed(es).astype(np.float32)
        projs = [float(np.dot(en, pc.astype(np.float32))) for pc in mPCs]
        src_projs.append((s, projs))
    # Print mean
    if src_projs:
        mean_p = np.mean([p for _, p in src_projs], axis=0)
        print("  %-14s  %-14s  %+.4f  %+.4f  %+.4f  %+.4f  %+.4f" %
              (dname, '[mean]', *mean_p))
    for s, p in src_projs[:3]:
        print("  %-14s  %-14s  %+.4f  %+.4f  %+.4f  %+.4f  %+.4f" %
              ('', s, *p))
    print()

# Also compare cluster tightness (mean within-cluster sim) for each domain
print("  Within-cluster cosine similarity for each gender domain:")
for dname, splits in GENDER_DOMAIN_SETS.items():
    pairs = splits['train']
    embs = [normed(get_emb(s)[0]).astype(np.float32) for s,_ in pairs if get_emb(s)[0] is not None]
    if len(embs) < 2: continue
    sims = [float(np.dot(embs[i], embs[j])) for i in range(len(embs)) for j in range(i+1, len(embs))]
    print("    %-12s  mean_cos=%.4f  std=%.4f  n=%d" % (dname, np.mean(sims), np.std(sims), len(embs)))

# ====================================================================
# PART C: EXTENDED +s WITH FULL EXCLUSION
# ====================================================================
print()
print("PART C: Extended +s — exclude caps + compound tokens")
print("-"*70)
ax_s, _, valid_s, _ = compute_axis([('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                                     ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')])
scale_s_clean, acc_s_clean = best_scale_clean(ax_s.astype(np.float32), valid_s)
print("  +s scale (clean retrieval): %.3f  train_acc=%d/%d" %
      (scale_s_clean, acc_s_clean, len(valid_s)))
print()

HOLDOUT_S = [('flower','flowers'),('star','stars'),('forest','forests'),('train','trains'),
             ('boat','boats'),('cup','cups'),('door','doors'),('road','roads'),
             ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
             ('wall','walls'),('room','rooms'),('fire','fires')]

print("  %-12s  standard  clean     target" % "source")
hits_std, hits_cl, total_s = 0, 0, 0
for src, tgt in HOLDOUT_S:
    es, sid = get_emb(src); et, _ = get_emb(tgt)
    if es is None: continue
    total_s += 1
    pred_std = W_E[sid] + scale_s_clean * ax_s
    r_std = nn_retrieve(pred_std, [sid], top_n=1)
    r_cl  = nn_retrieve_clean(pred_std, [sid], top_n=1)
    hit_std = (et is not None and r_std[0][0] == tgt)
    hit_cl  = (et is not None and r_cl[0][0]  == tgt)
    if hit_std: hits_std += 1
    if hit_cl:  hits_cl  += 1
    marker = '↑' if (hit_cl and not hit_std) else ('=' if hit_std==hit_cl else '↓')
    print("  %-12s  %-10s  %-10s  %-10s  %s" % (src, r_std[0][0], r_cl[0][0], tgt, marker))

print()
print("  Standard: %d/%d (%.0f%%)" % (hits_std, total_s, 100*hits_std/total_s if total_s else 0))
print("  Clean:    %d/%d (%.0f%%)" % (hits_cl,  total_s, 100*hits_cl /total_s if total_s else 0))
print()

# ====================================================================
# PART D: +tion SCALE TUNING FOR -ate VERBS
# ====================================================================
print("PART D: +tion scale tuning — ct-trained axis on -ate verbs")
print("-"*70)
ax_tion_ct, _, valid_tion, _ = compute_axis(
    [('act','action'),('direct','direction'),('collect','collection'),
     ('connect','connection'),('protect','protection'),('select','selection')])
ax_tion_ate, _, valid_tion_ate, pc_ate = compute_axis(
    [('observe','observation'),('describe','description'),('explain','explanation'),
     ('combine','combination'),('transform','transformation'),('operate','operation'),
     ('create','creation'),('investigate','investigation')])

TION_ATE_TEST = [
    ('communicate','communication'), ('participate','participation'),
    ('appreciate','appreciation'),   ('negotiate','negotiation'),
    ('evaluate','evaluation'),       ('generate','generation'),
    ('demonstrate','demonstration'), ('accelerate','acceleration'),
    ('educate','education'),         ('produce','production'),
]

print("  +tion-ct axis (scale from ct training):")
sc_ct, _ = best_scale(ax_tion_ct.astype(np.float32), valid_tion)
hits_ct = sum(1 for s,t in TION_ATE_TEST
              if get_emb(s)[0] is not None and get_emb(t)[0] is not None
              and nn_retrieve(W_E[get_emb(s)[1]]+sc_ct*ax_tion_ct,[get_emb(s)[1]],1)[0][0]==t)
print("  scale=%.3f  ct holdout: %d/%d" % (sc_ct, hits_ct, len([x for x in TION_ATE_TEST if get_emb(x[0])[0] is not None])))

print("  +tion-ate axis (scale from -ate training):")
sc_ate, _ = best_scale(ax_tion_ate.astype(np.float32), valid_tion_ate)
hits_ate = sum(1 for s,t in TION_ATE_TEST
               if get_emb(s)[0] is not None and get_emb(t)[0] is not None
               and nn_retrieve(W_E[get_emb(s)[1]]+sc_ate*ax_tion_ate,[get_emb(s)[1]],1)[0][0]==t)
print("  scale=%.3f  pc=%.4f  -ate holdout: %d/%d" % (sc_ate, pc_ate, hits_ate, len([x for x in TION_ATE_TEST if get_emb(x[0])[0] is not None])))

# Cosine between ct and ate axes
cos_axes = float(np.dot(ax_tion_ct, ax_tion_ate))
print("  cos(+tion-ct, +tion-ate) = %.4f" % cos_axes)
print()

# Test with SCALE SWEEP on ct axis applied to -ate words
print("  Scale sweep: +tion-ct axis on -ate holdout:")
best_sc, best_acc2 = 0, 0
for s_val in np.linspace(0.1, 3.0, 60):
    acc2 = sum(1 for src,tgt in TION_ATE_TEST
               if get_emb(src)[0] is not None and get_emb(tgt)[0] is not None
               and nn_retrieve(W_E[get_emb(src)[1]]+s_val*ax_tion_ct,[get_emb(src)[1]],1)[0][0]==tgt)
    if acc2 > best_acc2: best_acc2=acc2; best_sc=s_val
print("  Optimal scale for ct->ate: %.3f  acc=%d/%d (%.0f%%)" %
      (best_sc, best_acc2, len(TION_ATE_TEST), 100*best_acc2/len(TION_ATE_TEST)))
print()

# Per-word detail at best scale
print("  Per-word at optimal scale=%.3f:" % best_sc)
for src, tgt in TION_ATE_TEST:
    es, sid = get_emb(src); et, _ = get_emb(tgt)
    if es is None: continue
    pred = W_E[sid] + best_sc * ax_tion_ct
    r = nn_retrieve(pred, [sid], top_n=1)
    hit = (et is not None and r[0][0] == tgt)
    print("  %-18s -> %-22s %s  [%s]" % (src, r[0][0], '✓' if hit else '✗', tgt))
print()

# ====================================================================
# PART E: COMPREHENSIVE ACCURACY TABLE WITH CLEAN RETRIEVAL
# ====================================================================
print("PART E: All axes — clean retrieval accuracy vs standard")
print("-"*70)

FULL_HOLDOUT = {
    '+er':     [('loud','louder'),('quiet','quieter'),('warm','warmer'),('cold','colder'),
                ('young','younger'),('cheap','cheaper'),('rich','richer'),('poor','poorer'),
                ('wide','wider'),('narrow','narrower'),('soft','softer'),('hard','harder'),
                ('sweet','sweeter'),('thick','thicker'),('thin','thinner'),('rough','rougher'),
                ('smooth','smoother'),('new','newer'),('old','older'),('heavy','heavier')],
    'gender':  [('monk','nun'),('prince','princess'),('emperor','empress'),
                ('lion','lioness'),('tiger','tigress'),('actor','actress'),
                ('waiter','waitress'),('host','hostess'),('heir','heiress'),
                ('duke','duchess'),('wizard','witch'),('hero','heroine')],
    '+tion':   [('inject','injection'),('reject','rejection'),('infect','infection'),
                ('inspect','inspection'),('detect','detection'),('correct','correction'),
                ('construct','construction'),('instruct','instruction'),
                ('introduce','introduction'),('reduce','reduction')],
    '+s':      [('flower','flowers'),('star','stars'),('forest','forests'),('train','trains'),
                ('boat','boats'),('cup','cups'),('door','doors'),('road','roads'),
                ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
                ('wall','walls'),('room','rooms'),('fire','fires')],
}
FULL_TRAIN = {
    '+er':     [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    'gender':  [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
    '+tion':   [('act','action'),('direct','direction'),('collect','collection'),
                ('connect','connection'),('protect','protection'),('select','selection')],
    '+s':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
}

print("  %-10s  std_acc   clean_acc  delta" % "axis")
print("  " + "-"*44)
for nm in ['+er', '+s', 'gender', '+tion']:
    ax, _, valid, _ = compute_axis(FULL_TRAIN[nm])
    if ax is None: continue
    sc_std, _ = best_scale(ax.astype(np.float32), valid)
    sc_cl,  _ = best_scale_clean(ax.astype(np.float32), valid)
    hits_s, hits_c, total = 0, 0, 0
    for src, tgt in FULL_HOLDOUT[nm]:
        es, sid = get_emb(src); et, _ = get_emb(tgt)
        if es is None: continue
        total += 1
        pred_s = W_E[sid] + sc_std * ax
        pred_c = W_E[sid] + sc_cl  * ax
        r_s = nn_retrieve(pred_s, [sid], 1)
        r_c = nn_retrieve_clean(pred_c, [sid], 1)
        if et is not None and r_s[0][0] == tgt: hits_s += 1
        if et is not None and r_c[0][0] == tgt: hits_c += 1
    print("  %-10s  %d/%d=%.0f%%  %d/%d=%.0f%%    %+d" %
          (nm, hits_s, total, 100*hits_s/total,
               hits_c, total, 100*hits_c/total,
               hits_c - hits_s))
