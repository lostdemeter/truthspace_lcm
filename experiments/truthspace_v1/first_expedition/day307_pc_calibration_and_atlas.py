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
def best_scale(axis, valid_pairs, lo=0.02, hi=8.0, n=100):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

print("DAY 307: pc CALIBRATION, +s FAILURE ANALYSIS, SEMANTIC ATLAS")
print("="*70)
print()

# ====================================================================
# PART A: pc CALIBRATION — all 12 axes with FRESH HOLDOUT PAIRS
# ====================================================================
print("PART A: pc calibration — train / holdout for all axes")
print("-"*70)

TRAIN_HOLDOUT = {
    '+er': {
        'train':   [('fast','faster'),('slow','slower'),('tall','taller'),
                    ('short','shorter'),('bright','brighter'),('dark','darker'),
                    ('deep','deeper'),('clean','cleaner')],
        'holdout': [('loud','louder'),('quiet','quieter'),('warm','warmer'),
                    ('cold','colder'),('young','younger'),('cheap','cheaper'),
                    ('rich','richer'),('poor','poorer'),('wide','wider'),
                    ('narrow','narrower'),('soft','softer'),('hard','harder'),
                    ('sweet','sweeter'),('bitter','bitterer'),('thick','thicker'),
                    ('thin','thinner'),('heavy','heavier'),('light','lighter'),
                    ('rough','rougher'),('smooth','smoother')],
    },
    '+est': {
        'train':   [('fast','fastest'),('slow','slowest'),('tall','tallest'),
                    ('short','shortest'),('bright','brightest'),('dark','darkest'),
                    ('deep','deepest'),('clean','cleanest')],
        'holdout': [('loud','loudest'),('quiet','quietest'),('warm','warmest'),
                    ('cold','coldest'),('young','youngest'),('cheap','cheapest'),
                    ('rich','richest'),('wide','widest'),('soft','softest'),
                    ('hard','hardest'),('sweet','sweetest'),('thick','thickest'),
                    ('thin','thinnest'),('smooth','smoothest'),('rough','roughest')],
    },
    'er->est': {
        'train':   [('faster','fastest'),('slower','slowest'),('taller','tallest'),
                    ('shorter','shortest'),('brighter','brightest'),('darker','darkest'),
                    ('deeper','deepest'),('cleaner','cleanest')],
        'holdout': [('louder','loudest'),('quieter','quietest'),('warmer','warmest'),
                    ('colder','coldest'),('younger','youngest'),('cheaper','cheapest'),
                    ('richer','richest'),('wider','widest'),('softer','softest'),
                    ('harder','hardest'),('sweeter','sweetest')],
    },
    'gender': {
        'train':   [('king','queen'),('man','woman'),('boy','girl'),('father','mother'),
                    ('son','daughter'),('brother','sister'),('uncle','aunt'),('husband','wife')],
        'holdout': [('monk','nun'),('prince','princess'),('emperor','empress'),
                    ('lion','lioness'),('tiger','tigress'),('actor','actress'),
                    ('waiter','waitress'),('host','hostess'),('heir','heiress'),
                    ('duke','duchess'),('wizard','witch'),('hero','heroine')],
    },
    'past_irr': {
        'train':   [('go','went'),('come','came'),('run','ran'),('see','saw'),
                    ('eat','ate'),('know','knew'),('take','took'),('make','made')],
        'holdout': [('write','wrote'),('read','read'),('give','gave'),('find','found'),
                    ('buy','bought'),('bring','brought'),('think','thought'),
                    ('catch','caught'),('teach','taught'),('draw','drew'),
                    ('grow','grew'),('fly','flew'),('throw','threw'),('blow','blew')],
    },
    '+ed': {
        'train':   [('walk','walked'),('talk','talked'),('jump','jumped'),
                    ('start','started'),('end','ended'),('look','looked'),
                    ('call','called'),('help','helped')],
        'holdout': [('play','played'),('work','worked'),('move','moved'),
                    ('open','opened'),('close','closed'),('turn','turned'),
                    ('push','pushed'),('pull','pulled'),('fix','fixed'),
                    ('stop','stopped'),('add','added'),('ask','asked'),
                    ('show','showed'),('use','used'),('need','needed')],
    },
    '+ness': {
        'train':   [('sad','sadness'),('happy','happiness'),('dark','darkness'),
                    ('kind','kindness'),('bright','brightness'),('mad','madness')],
        'holdout': [('sick','sickness'),('weak','weakness'),('bold','boldness'),
                    ('cold','coldness'),('hard','hardness'),('aware','awareness'),
                    ('soft','softness'),('wild','wildness'),('deaf','deafness')],
    },
    '+ful': {
        'train':   [('hope','hopeful'),('care','careful'),('use','useful'),
                    ('power','powerful'),('peace','peaceful'),('harm','harmful'),
                    ('thank','thankful'),('help','helpful')],
        'holdout': [('wonder','wonderful'),('color','colorful'),('grace','graceful'),
                    ('faith','faithful'),('skill','skillful'),('dread','dreadful'),
                    ('play','playful'),('rest','restful'),('cheer','cheerful'),
                    ('awe','awful'),('force','forceful'),('truth','truthful')],
    },
    'un-': {
        'train':   [('happy','unhappy'),('kind','unkind'),('fair','unfair'),
                    ('known','unknown'),('usual','unusual'),('clear','unclear'),
                    ('lock','unlock'),('wrap','unwrap')],
        'holdout': [('tie','untie'),('fold','unfold'),('pack','unpack'),
                    ('load','unload'),('cover','uncover'),('safe','unsafe'),
                    ('true','untrue'),('real','unreal'),('likely','unlikely'),
                    ('common','uncommon'),('easy','uneasy'),('aware','unaware')],
    },
    '+ment': {
        'train':   [('achieve','achievement'),('manage','management'),
                    ('develop','development'),('move','movement'),
                    ('treat','treatment'),('argue','argument')],
        'holdout': [('judge','judgment'),('employ','employment'),
                    ('invest','investment'),('settle','settlement'),
                    ('measure','measurement'),('improve','improvement'),
                    ('agree','agreement'),('state','statement'),
                    ('announce','announcement'),('assess','assessment')],
    },
    '+s': {
        'train':   [('cat','cats'),('dog','dogs'),('house','houses'),('car','cars'),
                    ('tree','trees'),('book','books'),('bird','birds'),('ship','ships')],
        'holdout': [('flower','flowers'),('star','stars'),('forest','forests'),
                    ('train','trains'),('boat','boats'),('cup','cups'),
                    ('door','doors'),('road','roads'),('hand','hands'),
                    ('eye','eyes'),('arm','arms'),('leg','legs'),
                    ('wall','walls'),('room','rooms'),('fire','fires')],
    },
    '+tion': {
        'train':   [('act','action'),('direct','direction'),('collect','collection'),
                    ('connect','connection'),('protect','protection'),('select','selection')],
        'holdout': [('inject','injection'),('reject','rejection'),('infect','infection'),
                    ('inspect','inspection'),('detect','detection'),('correct','correction'),
                    ('construct','construction'),('instruct','instruction'),
                    ('introduce','introduction'),('reduce','reduction')],
    },
}

print("  %-12s  pc_train  n_train  coh    scale  train_acc  holdout_acc  n_ho" % "axis")
print("  " + "-"*78)

axis_results = {}
for nm, splits in TRAIN_HOLDOUT.items():
    train_pairs = splits['train']
    hold_pairs  = splits['holdout']

    ax, coh, valid, pc = compute_axis(train_pairs)
    if ax is None: continue

    scale, train_acc = best_scale(ax.astype(np.float32), valid)
    n_train = len(valid)

    # Holdout evaluation
    ho_hits, ho_total = 0, 0
    ho_details = []
    for src, tgt in hold_pairs:
        es, sid = get_emb(src)
        et, tid = get_emb(tgt)
        if es is None: continue
        # Check if target is a single token
        single_tgt = (et is not None)
        ho_total += 1
        if not single_tgt:
            ho_details.append((src, tgt, None, False, 'NOT_SINGLE'))
            continue
        pred = W_E[sid] + scale * ax
        r = nn_retrieve(pred, [sid], top_n=1)
        hit = (r[0][0] == tgt)
        if hit: ho_hits += 1
        ho_details.append((src, tgt, r[0][0], hit, 'ok'))

    ho_pct = 100*ho_hits/ho_total if ho_total else 0
    axis_results[nm] = {
        'pc': pc, 'coh': coh, 'scale': scale,
        'train_acc': train_acc/n_train if n_train else 0,
        'holdout_acc': ho_hits/ho_total if ho_total else 0,
        'n_train': n_train, 'n_holdout': ho_total,
        'ho_details': ho_details,
    }
    print("  %-12s  %+.4f    %d       %.4f  %.3f   %d/%d         %d/%d (%.0f%%)" %
          (nm, pc, n_train, coh, scale, train_acc, n_train,
           ho_hits, ho_total, ho_pct))

print()

# ====================================================================
# PART B: +s FAILURE ANALYSIS
# ====================================================================
print("PART B: +s failure analysis")
print("-"*70)
if '+s' in axis_results:
    r = axis_results['+s']
    ax_s, _, valid_s, _ = compute_axis(TRAIN_HOLDOUT['+s']['train'])

    print("  +s axis details:")
    print("  pc=%.4f  coh=%.4f  scale=%.3f  train=%d/%d  holdout=%.0f%%" %
          (r['pc'], r['coh'], r['scale'], int(r['train_acc']*r['n_train']),
           r['n_train'], 100*r['holdout_acc']))
    print()

    print("  Holdout breakdown:")
    not_single, wrong_tgt, wrong_pred, hits = 0, 0, 0, 0
    for src, tgt, pred, hit, status in r['ho_details']:
        if status == 'NOT_SINGLE':
            not_single += 1
            print("    %-12s -> %-12s  [TARGET NOT SINGLE TOKEN]" % (src, tgt))
        elif hit:
            hits += 1
            print("    %-12s -> %-12s  HIT" % (src, tgt))
        else:
            wrong_pred += 1
            # Check if source is single token
            es, sid = get_emb(src)
            if es is not None:
                # What is the nearest token to source?
                r5 = nn_retrieve(W_E[sid], [sid], top_n=3)
                nn_src = r5[0][0]
            else:
                nn_src = '?'
            print("    %-12s -> %-12s  got: %-12s  (src_nn=%s)" %
                  (src, tgt, pred, nn_src))
    print()
    print("  Summary: hits=%d  wrong_pred=%d  not_single=%d  total=%d" %
          (hits, wrong_pred, not_single, len(r['ho_details'])))
    print()

    # Check: is the +s holdout failing because target is multi-token?
    print("  Token availability check for holdout targets:")
    for src, tgt in TRAIN_HOLDOUT['+s']['holdout']:
        et, tid = get_emb(tgt)
        es, sid = get_emb(src)
        status = 'both' if (et is not None and es is not None) else \
                 'src_only' if (es is not None) else \
                 'tgt_only' if (et is not None) else 'neither'
        print("    %-12s -> %-12s  (%s)" % (src, tgt, status))
print()

# ====================================================================
# PART C: pc → HOLDOUT CURVE
# ====================================================================
print("PART C: pc vs holdout accuracy (all axes)")
print("-"*70)
print("  {:<12}  pc       holdout%  train%   n_train  n_holdout".format("axis"))
print("  " + "-"*62)
for nm, r in sorted(axis_results.items(), key=lambda x: -x[1]['pc']):
    print("  %-12s  %.4f   %5.1f%%    %5.1f%%   %d        %d" %
          (nm, r['pc'], 100*r['holdout_acc'], 100*r['train_acc'],
           r['n_train'], r['n_holdout']))
print()

# Compute correlation pc vs holdout
pcs = [r['pc'] for r in axis_results.values()]
accs = [r['holdout_acc'] for r in axis_results.values()]
r_corr = np.corrcoef(pcs, accs)[0,1]
print("  Pearson r(pc, holdout_acc) = %.4f" % r_corr)
print()

# ====================================================================
# PART D: FULL SEMANTIC ATLAS TABLE
# ====================================================================
print("PART D: Full W_E semantic atlas (all known axes vs all global PCs)")
print("-"*70)

# Build global PCs (top 10)
rng = np.random.default_rng(42)
N_SAMPLE = 8000
sample_ids = rng.integers(0, len(W_E), size=N_SAMPLE)
W_sample = W_E[sample_ids].astype(np.float32)
mu = W_sample.mean(axis=0)
W_c = W_sample - mu

global_pcs = []
W_defl = W_c.copy()
for k in range(10):
    vk = rng.standard_normal(W_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = W_defl.T @ (W_defl @ vk)
        vk /= np.linalg.norm(vk)
    proj = W_defl @ vk
    W_defl = W_defl - np.outer(proj, vk)
    lam = float(np.var(W_c @ vk))
    global_pcs.append(vk.astype(np.float64))

# Build labelling axes
MONTHS   = ['January','February','March','April','May','June','July','August','September']
WEEKDAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CARDS    = ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ace']
CARD_N   = ['2','3','4','5','6','7','8','9','1']
PLANETS  = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']

ALL_AXES = {}
for nm, pairs in TRAIN_HOLDOUT.items():
    ax, _, _, pc = compute_axis(pairs['train'])
    if ax is not None: ALL_AXES[nm] = ax.astype(np.float64)

# Add labelling axes
for name, pairs in [
    ('month->num',   [(MONTHS[i], str(i+1)) for i in range(9)]),
    ('weekday->num', [(WEEKDAYS[i], str(i+1)) for i in range(7)]),
    ('card->num',    list(zip(CARDS, CARD_N))),
    ('planet->num',  [(PLANETS[i], str(i+1)) for i in range(8)]),
    ('digit->word',  [(str(i+1), ['one','two','three','four','five','six','seven','eight','nine'][i])
                      for i in range(9)]),
    ('country->dem', [('France','French'),('Germany','German'),('Italy','Italian'),
                      ('Spain','Spanish'),('Japan','Japanese'),('China','Chinese')]),
]:
    ax, _, _, _ = compute_axis(pairs)
    if ax is not None: ALL_AXES[name] = ax.astype(np.float64)

# Build v_ord
fwd = []
for pairs in [[(MONTHS[i], str(i+1)) for i in range(9)],
              [(WEEKDAYS[i], str(i+1)) for i in range(7)],
              list(zip(CARDS, CARD_N))]:
    ax, _, _, _ = compute_axis(pairs)
    if ax is not None: fwd.append(ax)
ALL_AXES['v_ord'] = normed(np.mean(fwd, axis=0)).astype(np.float64)

print("  %-16s" % "axis" + "".join("  PC%-2d" % (k+1) for k in range(10)) + "   R²")
print("  " + "-"*(16 + 7*10 + 6))
for nm, ax in sorted(ALL_AXES.items()):
    comps = [float(np.dot(ax, pcv)) for pcv in global_pcs]
    r2 = sum(c**2 for c in comps)
    print("  %-16s" % nm + "".join(" %+.3f" % c for c in comps) + "  %.3f" % r2)
print()

# ====================================================================
# PART E: mPC8 LINGUISTIC INTERPRETATION
# ====================================================================
print("PART E: mPC8 interpretation — simple past vs past participle")
print("-"*70)

# Build morphological chord matrix and get mPC8
ALL_MORPH = {
    '+er':     TRAIN_HOLDOUT['+er']['train'] + [('light','lighter'),('strong','stronger'),
                ('weak','weaker'),('soft','softer')],
    '+est':    TRAIN_HOLDOUT['+est']['train'] + [('light','lightest'),('strong','strongest'),
                ('weak','weakest')],
    'er->est': TRAIN_HOLDOUT['er->est']['train'] + [('lighter','lightest'),
                ('stronger','strongest'),('weaker','weakest')],
    'gender':  TRAIN_HOLDOUT['gender']['train'],
    'past_irr':TRAIN_HOLDOUT['past_irr']['train'],
    '+ed':     TRAIN_HOLDOUT['+ed']['train'],
    '+ness':   TRAIN_HOLDOUT['+ness']['train'],
    '+ful':    TRAIN_HOLDOUT['+ful']['train'],
    'un-':     TRAIN_HOLDOUT['un-']['train'],
    '+ment':   TRAIN_HOLDOUT['+ment']['train'],
    '+s':      TRAIN_HOLDOUT['+s']['train'],
    '+tion':   TRAIN_HOLDOUT['+tion']['train'],
}

all_chords = []
for nm, pairs in ALL_MORPH.items():
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None: continue
        all_chords.append(normed(et-es).astype(np.float32))

M_mat = np.array(all_chords)
M_c = (M_mat - M_mat.mean(axis=0)).astype(np.float32)
rng2 = np.random.default_rng(0)
M_pcs = []
M_defl = M_c.copy()
for k in range(8):
    vk = rng2.standard_normal(M_c.shape[1]).astype(np.float32)
    vk /= np.linalg.norm(vk)
    for _ in range(200):
        vk = M_defl.T @ (M_defl @ vk)
        vk /= np.linalg.norm(vk)
    proj = M_defl @ vk
    M_defl = M_defl - np.outer(proj, vk)
    M_pcs.append(vk.astype(np.float64))

mpc8 = M_pcs[7].astype(np.float32)

# Test: compare simple past vs past participle scores on mPC8
SIMPLE_PAST   = ['went','took','gave','came','saw','ran','ate','knew','made',
                 'flew','threw','grew','wrote','drew','bought','found','brought']
PAST_PART     = ['known','called','been','seen','shown','taken','referred',
                 'gotten','given','written','drawn','bought','found','brought',
                 'gone','come','made','done']
PRESENT       = ['go','take','give','come','see','run','eat','know','make',
                 'fly','throw','grow','write','draw','buy','find','bring']

print("  mPC8 scores for verb forms:")
print("  %-16s  mPC8" % "word")
for group, label in [(SIMPLE_PAST, 'SIMPLE PAST'), (PAST_PART, 'PAST PART'), (PRESENT, 'PRESENT')]:
    scores = []
    print("  --- %s ---" % label)
    for w in group[:8]:
        e, _ = get_emb(w)
        if e is None: continue
        s = float(np.dot(normed(e).astype(np.float32), mpc8))
        scores.append(s)
        print("  %-16s  %+.4f" % (w, s))
    if scores:
        print("  [mean=%.4f]" % np.mean(scores))
    print()
