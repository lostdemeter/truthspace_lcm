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
def nn_retrieve_no_caps(pred_emb, exclude_ids, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    # Exclude any token whose decoded form starts with uppercase
    for i in range(len(sims)):
        w = tok.decode([i]).strip()
        if w and w[0].isupper(): sims[i] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def domain_sim(train_pairs, holdout_pairs):
    """Mean cosine similarity between all train source embeddings and all holdout source embeddings."""
    tr_embs = [normed(get_emb(s)[0]).astype(np.float32) for s,_ in train_pairs if get_emb(s)[0] is not None]
    ho_embs = [normed(get_emb(s)[0]).astype(np.float32) for s,_ in holdout_pairs if get_emb(s)[0] is not None]
    if not tr_embs or not ho_embs: return 0.0
    sims = []
    for te in tr_embs:
        for he in ho_embs:
            sims.append(float(np.dot(te, he)))
    return float(np.mean(sims))

print("DAY 308: DOMAIN OVERLAP METRIC, +s FIX, +tion EXPANSION, GENDER EXPANSION")
print("="*70)
print()

TRAIN = {
    '+er':     [('fast','faster'),('slow','slower'),('tall','taller'),('short','shorter'),
                ('bright','brighter'),('dark','darker'),('deep','deeper'),('clean','cleaner')],
    '+est':    [('fast','fastest'),('slow','slowest'),('tall','tallest'),('short','shortest'),
                ('bright','brightest'),('dark','darkest'),('deep','deepest'),('clean','cleanest')],
    'er->est': [('faster','fastest'),('slower','slowest'),('taller','tallest'),('shorter','shortest'),
                ('brighter','brightest'),('darker','darkest'),('deeper','deepest'),('cleaner','cleanest')],
    'gender':  [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
    'past_irr':[('go','went'),('come','came'),('run','ran'),('see','saw'),
                ('eat','ate'),('know','knew'),('take','took'),('make','made')],
    '+ed':     [('walk','walked'),('talk','talked'),('jump','jumped'),('start','started'),
                ('end','ended'),('look','looked'),('call','called'),('help','helped')],
    '+ness':   [('sad','sadness'),('happy','happiness'),('dark','darkness'),('kind','kindness'),
                ('bright','brightness'),('mad','madness')],
    '+ful':    [('hope','hopeful'),('care','careful'),('use','useful'),('power','powerful'),
                ('peace','peaceful'),('harm','harmful'),('thank','thankful'),('help','helpful')],
    'un-':     [('happy','unhappy'),('kind','unkind'),('fair','unfair'),('known','unknown'),
                ('usual','unusual'),('clear','unclear'),('lock','unlock'),('wrap','unwrap')],
    '+ment':   [('achieve','achievement'),('manage','management'),('develop','development'),
                ('move','movement'),('treat','treatment'),('argue','argument')],
    '+s':      [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
    '+tion':   [('act','action'),('direct','direction'),('collect','collection'),
                ('connect','connection'),('protect','protection'),('select','selection')],
}
HOLDOUT = {
    '+er':     [('loud','louder'),('quiet','quieter'),('warm','warmer'),('cold','colder'),
                ('young','younger'),('cheap','cheaper'),('rich','richer'),('poor','poorer'),
                ('wide','wider'),('narrow','narrower'),('soft','softer'),('hard','harder'),
                ('sweet','sweeter'),('thick','thicker'),('thin','thinner'),('heavy','heavier'),
                ('rough','rougher'),('smooth','smoother'),('new','newer'),('old','older')],
    '+est':    [('loud','loudest'),('quiet','quietest'),('warm','warmest'),('cold','coldest'),
                ('young','youngest'),('cheap','cheapest'),('rich','richest'),('wide','widest'),
                ('soft','softest'),('hard','hardest'),('sweet','sweetest'),('thick','thickest'),
                ('thin','thinnest'),('smooth','smoothest'),('rough','roughest')],
    'er->est': [('louder','loudest'),('quieter','quietest'),('warmer','warmest'),('colder','coldest'),
                ('younger','youngest'),('cheaper','cheapest'),('richer','richest'),('wider','widest'),
                ('softer','softest'),('harder','hardest'),('sweeter','sweetest')],
    'gender':  [('monk','nun'),('prince','princess'),('emperor','empress'),
                ('lion','lioness'),('tiger','tigress'),('actor','actress'),
                ('waiter','waitress'),('host','hostess'),('heir','heiress'),
                ('duke','duchess'),('wizard','witch'),('hero','heroine')],
    'past_irr':[('write','wrote'),('give','gave'),('find','found'),
                ('buy','bought'),('bring','brought'),('think','thought'),
                ('catch','caught'),('teach','taught'),('draw','drew'),
                ('grow','grew'),('fly','flew'),('throw','threw'),('blow','blew')],
    '+ed':     [('play','played'),('work','worked'),('move','moved'),('open','opened'),
                ('close','closed'),('turn','turned'),('push','pushed'),('pull','pulled'),
                ('fix','fixed'),('stop','stopped'),('add','added'),('ask','asked'),
                ('show','showed'),('use','used'),('need','needed')],
    '+ness':   [('sick','sickness'),('weak','weakness'),('bold','boldness'),('cold','coldness'),
                ('hard','hardness'),('aware','awareness'),('soft','softness'),('wild','wildness'),('deaf','deafness')],
    '+ful':    [('wonder','wonderful'),('color','colorful'),('grace','graceful'),
                ('faith','faithful'),('skill','skillful'),('dread','dreadful'),
                ('play','playful'),('rest','restful'),('cheer','cheerful'),
                ('awe','awful'),('force','forceful'),('truth','truthful')],
    'un-':     [('tie','untie'),('fold','unfold'),('pack','unpack'),('load','unload'),
                ('cover','uncover'),('safe','unsafe'),('true','untrue'),('real','unreal'),
                ('likely','unlikely'),('common','uncommon'),('easy','uneasy'),('aware','unaware')],
    '+ment':   [('judge','judgment'),('employ','employment'),('invest','investment'),
                ('settle','settlement'),('measure','measurement'),('improve','improvement'),
                ('agree','agreement'),('state','statement'),('announce','announcement'),('assess','assessment')],
    '+s':      [('flower','flowers'),('star','stars'),('forest','forests'),('train','trains'),
                ('boat','boats'),('cup','cups'),('door','doors'),('road','roads'),
                ('hand','hands'),('eye','eyes'),('arm','arms'),('leg','legs'),
                ('wall','walls'),('room','rooms'),('fire','fires')],
    '+tion':   [('inject','injection'),('reject','rejection'),('infect','infection'),
                ('inspect','inspection'),('detect','detection'),('correct','correction'),
                ('construct','construction'),('instruct','instruction'),
                ('introduce','introduction'),('reduce','reduction')],
}

# ====================================================================
# PART A: DOMAIN OVERLAP METRIC vs HOLDOUT
# ====================================================================
print("PART A: Domain overlap metric (train-source vs holdout-source cosine similarity)")
print("-"*70)
print("  %-12s  domain_sim  pc       holdout%%  interpretation" % "axis")
print("  " + "-"*62)

results_a = {}
for nm in TRAIN:
    if nm not in HOLDOUT: continue
    ax, _, valid, pc = compute_axis(TRAIN[nm])
    if ax is None: continue
    ds = domain_sim(TRAIN[nm], HOLDOUT[nm])

    scale, _ = best_scale(ax.astype(np.float32), valid)
    hits, total = 0, 0
    for s, t in HOLDOUT[nm]:
        es, sid = get_emb(s)
        et, tid = get_emb(t)
        if es is None: continue
        total += 1
        if et is None: continue
        pred = W_E[sid] + scale * ax
        r = nn_retrieve(pred, [sid], top_n=1)
        if r[0][0] == t: hits += 1

    acc = hits/total if total else 0
    results_a[nm] = {'ds': ds, 'pc': pc, 'acc': acc, 'n': total}
    print("  %-12s  %.4f      %.4f   %5.1f%%   " % (nm, ds, pc, 100*acc))

# Compute correlations
ds_vals  = [results_a[nm]['ds']  for nm in results_a]
pc_vals  = [results_a[nm]['pc']  for nm in results_a]
acc_vals = [results_a[nm]['acc'] for nm in results_a]
r_ds  = np.corrcoef(ds_vals, acc_vals)[0,1]
r_pc  = np.corrcoef(pc_vals, acc_vals)[0,1]
print()
print("  Pearson r(domain_sim, holdout_acc) = %.4f" % r_ds)
print("  Pearson r(pc,         holdout_acc) = %.4f" % r_pc)
print()

# ====================================================================
# PART B: +s WITH CAPITALIZED TOKEN EXCLUSION FIX
# ====================================================================
print("PART B: +s axis — fix by excluding capitalized tokens")
print("-"*70)
ax_s, _, valid_s, pc_s = compute_axis(TRAIN['+s'])
scale_s, _ = best_scale(ax_s.astype(np.float32), valid_s)

print("  +s holdout (standard nn_retrieve vs no-caps retrieve):")
print("  %-12s  %-14s  %-14s  target" % ("source", "standard", "no-caps"))
hits_std, hits_nc, total_s = 0, 0, 0
for src, tgt in HOLDOUT['+s']:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    total_s += 1
    pred = W_E[sid] + scale_s * ax_s
    r_std = nn_retrieve(pred, [sid], top_n=1)
    r_nc  = nn_retrieve_no_caps(pred, [sid], top_n=1)
    hit_std = (r_std[0][0] == tgt) if et is not None else False
    hit_nc  = (r_nc[0][0]  == tgt) if et is not None else False
    if hit_std: hits_std += 1
    if hit_nc:  hits_nc  += 1
    marker = '↑' if (hit_nc and not hit_std) else ('=' if hit_std==hit_nc else '↓')
    print("  %-12s  %-14s  %-14s  %-12s  %s" % (src, r_std[0][0], r_nc[0][0], tgt, marker))

print()
print("  Standard: %d/%d (%.0f%%)" % (hits_std, total_s, 100*hits_std/total_s))
print("  No-caps:  %d/%d (%.0f%%)" % (hits_nc,  total_s, 100*hits_nc /total_s))
print()

# ====================================================================
# PART C: +tion EXPANSION — NON-LATIN WORDS
# ====================================================================
print("PART C: +tion axis on non-Latin-root words")
print("-"*70)
ax_tion, _, valid_tion, pc_tion = compute_axis(TRAIN['+tion'])
scale_tion, _ = best_scale(ax_tion.astype(np.float32), valid_tion)

TION_NONLATIN = [
    ('observe','observation'),   ('describe','description'),  ('explain','explanation'),
    ('combine','combination'),   ('produce','production'),    ('transform','transformation'),
    ('educate','education'),     ('operate','operation'),     ('create','creation'),
    ('investigate','investigation'), ('communicate','communication'),
    ('participate','participation'), ('demonstrate','demonstration'),
    ('appreciate','appreciation'),  ('negotiate','negotiation'),
    ('accelerate','acceleration'),  ('evaluate','evaluation'),
    ('exaggerate','exaggeration'),  ('abbreviate','abbreviation'),
    ('generate','generation'),      ('demonstrate','demonstration'),
]
# Also test GERMANIC-root verbs (should NOT work)
TION_GERMANIC = [
    ('help','?'),   ('start','?'),  ('think','?'),  ('walk','?'),
    ('love','?'),   ('wish','?'),   ('fight','?'),  ('speak','?'),
]

print("  LATIN-ROOT (non -ct/-ct) holdout:")
hits_nl, total_nl = 0, 0
for src, tgt in TION_NONLATIN:
    es, sid = get_emb(src); et, tid = get_emb(tgt)
    if es is None: continue
    total_nl += 1
    pred = W_E[sid] + scale_tion * ax_tion
    r = nn_retrieve(pred, [sid], top_n=3)
    hit = (et is not None and r[0][0] == tgt)
    if hit: hits_nl += 1
    top3 = ', '.join(w for w,_,_ in r[:3])
    print("    %-18s -> %-20s  %s  [%s]" % (src, r[0][0], '✓' if hit else '✗', tgt))

print()
print("  Latin non-ct accuracy: %d/%d (%.0f%%)" % (hits_nl, total_nl, 100*hits_nl/total_nl if total_nl else 0))
print()

print("  GERMANIC-ROOT verbs (prediction only — no expected +tion):")
for src, _ in TION_GERMANIC:
    es, sid = get_emb(src)
    if es is None: continue
    pred = W_E[sid] + scale_tion * ax_tion
    r = nn_retrieve(pred, [sid], top_n=3)
    print("    %-12s  -> %s" % (src, ', '.join(w for w,_,_ in r[:3])))
print()

# ====================================================================
# PART D: GENDER AXIS DOMAIN EXPANSION
# ====================================================================
print("PART D: Gender axis — domain expansion analysis")
print("-"*70)
ax_g, _, valid_g, pc_g = compute_axis(TRAIN['gender'])
scale_g, _ = best_scale(ax_g.astype(np.float32), valid_g)

# Test different semantic domains
GENDER_DOMAINS = {
    'kin_core':      [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                      ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
    'kin_extended':  [('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
                      ('widower','widow'),('stepfather','stepmother')],
    'titles':        [('lord','lady'),('duke','duchess'),('prince','princess'),
                      ('emperor','empress'),('sir','madam')],
    'occupation':    [('actor','actress'),('waiter','waitress'),('host','hostess'),
                      ('steward','stewardess'),('heir','heiress'),('hero','heroine')],
    'religion':      [('monk','nun'),('priest','priestess'),('friar','sister'),
                      ('abbot','abbess')],
    'animals':       [('lion','lioness'),('tiger','tigress'),('stallion','mare'),
                      ('ram','ewe'),('bull','cow'),('drake','duck'),('gander','goose')],
    'fiction':       [('wizard','witch'),('sorcerer','sorceress'),('warlock','witch'),
                      ('prince','princess')],
}

print("  %-14s  scale_shared  hits/total  acc%%" % "domain")
print("  " + "-"*50)
for domain, pairs in GENDER_DOMAINS.items():
    hits, total = 0, 0
    for src, tgt in pairs:
        es, sid = get_emb(src); et, tid = get_emb(tgt)
        if es is None: continue
        total += 1
        if et is None: continue
        pred = W_E[sid] + scale_g * ax_g
        r = nn_retrieve(pred, [sid], top_n=1)
        if r[0][0] == tgt: hits += 1
    acc = 100*hits/total if total else 0
    print("  %-14s  %.3f         %d/%d         %.0f%%" %
          (domain, scale_g, hits, total, acc))

print()
print("  Per-domain axis (train on each domain separately):")
print("  %-14s  pc       scale   acc/train  acc/holdout  n_ho" % "domain")
print("  " + "-"*66)
for domain, pairs in GENDER_DOMAINS.items():
    if len(pairs) < 3: continue
    # Train on first half, test on second half
    n = len(pairs)
    tr = pairs[:n//2]; ho = pairs[n//2:]
    ax_d, _, vd, pc_d = compute_axis(tr)
    if ax_d is None or len(ho) < 1: continue
    sc_d, tr_acc = best_scale(ax_d.astype(np.float32), vd)
    ho_hits = sum(1 for s,t in ho
                  if get_emb(s)[0] is not None and get_emb(t)[0] is not None
                  and nn_retrieve(W_E[get_emb(s)[1]]+sc_d*ax_d,[get_emb(s)[1]],1)[0][0]==t)
    ho_total = sum(1 for s,t in ho if get_emb(s)[0] is not None and get_emb(t)[0] is not None)
    print("  %-14s  %.4f   %.3f    %d/%d         %d/%d         %d" %
          (domain, pc_d, sc_d, tr_acc, len(vd), ho_hits, ho_total, ho_total))

print()

# ====================================================================
# PART E: DOMAIN OVERLAP DEEPER ANALYSIS
# ====================================================================
print("PART E: Why +tion works (within-domain cos) vs why gender fails")
print("-"*70)

# Compute mean within-train cosine similarity for each domain
def mean_within_sim(pairs):
    embs = [normed(get_emb(s)[0]).astype(np.float32) for s,_ in pairs if get_emb(s)[0] is not None]
    if len(embs) < 2: return 0.0
    sims = [float(np.dot(embs[i], embs[j])) for i in range(len(embs)) for j in range(i+1, len(embs))]
    return float(np.mean(sims))

for nm in ['+tion', '+er', 'gender', '+s', 'un-', '+ful', 'past_irr']:
    if nm not in TRAIN or nm not in HOLDOUT: continue
    tr_wt = mean_within_sim(TRAIN[nm])
    ho_wt = mean_within_sim(HOLDOUT[nm])
    tr_ho = domain_sim(TRAIN[nm], HOLDOUT[nm])
    ax, _, _, pc = compute_axis(TRAIN[nm])
    acc = results_a.get(nm, {}).get('acc', 0)
    print("  %-12s  train_sim=%.4f  hold_sim=%.4f  tr->ho_sim=%.4f  pc=%.4f  acc=%.0f%%" %
          (nm, tr_wt, ho_wt, tr_ho, pc, 100*acc))

print()
# Compute correlation of tr->ho_sim with holdout accuracy
tho_sims = []
ho_accs  = []
for nm in TRAIN:
    if nm not in HOLDOUT or nm not in results_a: continue
    ax, _, _, _ = compute_axis(TRAIN[nm])
    if ax is None: continue
    tho_sims.append(domain_sim(TRAIN[nm], HOLDOUT[nm]))
    ho_accs.append(results_a[nm]['acc'])
r_tho = np.corrcoef(tho_sims, ho_accs)[0,1]
print("  Pearson r(train->holdout cosine sim, holdout_acc) = %.4f" % r_tho)
print()

# ====================================================================
# PART F: COMPOSITE PREDICTOR
# ====================================================================
print("PART F: Composite predictor — domain_sim × pc vs holdout")
print("-"*70)
for nm in sorted(results_a, key=lambda n: results_a[n]['ds']*results_a[n]['pc'], reverse=True):
    r = results_a[nm]
    composite = r['ds'] * r['pc']
    print("  %-12s  ds=%.4f  pc=%.4f  ds*pc=%.5f  holdout=%.0f%%" %
          (nm, r['ds'], r['pc'], composite, 100*r['acc']))

composites = [results_a[nm]['ds']*results_a[nm]['pc'] for nm in results_a]
r_comp = np.corrcoef(composites, [results_a[nm]['acc'] for nm in results_a])[0,1]
print()
print("  Pearson r(ds*pc, holdout_acc) = %.4f" % r_comp)
