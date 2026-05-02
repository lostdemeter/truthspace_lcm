import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and len(w) > 1 and not w.startswith('-') and not w.startswith('_'):
        RELAXED_MASK[i] = True

_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [' '+word, word, ' '+word[0].upper()+word[1:],
              word[0].upper()+word[1:], word.upper(), ' '+word.upper()]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    _src_cache[word] = ids
    return ids

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def get_tc(word):
    for p in [' ', '']:
        return len(tok(p+word, add_special_tokens=False)['input_ids'])
    return 99

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    i = int(np.argmax(sims))
    return tok.decode([i]).strip()

def axis_spread(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    n = len(cn)
    cosines = [np.dot(cn[i], cn[j]) for i in range(n) for j in range(i+1,n)]
    return md, valid, float(np.mean(cosines)), float(np.std(cosines)) if len(cosines)>1 else 0.0

def best_scale(ax, valid, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 6.0, 30):
        c = sum(1 for _,t,sid,_ in valid
                if nn_ret(W_E[sid]+s*ax, source_ids(tok.decode([sid]).strip()), mask)==t)
        if c > best_a: best_a=c; best_s=s
    return best_s

def loo(ax, valid, mask):
    if len(valid) < 3: return 0.0
    cf = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    gs = best_scale(normed(np.mean(cf, axis=0)), valid, mask)
    hits = 0
    for i in range(len(valid)):
        tv = [valid[j] for j in range(len(valid)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        s, t, sid, _ = valid[i]
        if nn_ret(W_E[sid]+gs*al, source_ids(s), mask) == t: hits += 1
    return hits/len(valid)

def irred(ax, holdout, mask):
    nh=0; ni=0; nt0=0
    for sw, tw in holdout:
        es, sid = get_emb(sw)
        if es is None: continue
        tc = get_tc(tw); nh += 1; found = False
        for s in np.linspace(0.02, 6.0, 60):
            if nn_ret(W_E[sid]+s*ax, source_ids(sw), mask) == tw: found=True; break
        if not found:
            ni += 1
            if tc > 1: nt0 += 1
    return (ni/nh if nh else 0.0), (nt0/max(ni,1))

def match(p, t):
    return (t.split('_')[0] in p or t in p or
            ('morph' in p and 'morph' in t) or ('phonol' in p and 'phonol' in t) or
            ('relational' in p and 'relational' in t) or ('factual' in p and 'factual' in t) or
            ('translation' in p and 'translation' in t) or ('polar' in p and 'polar' in t) or
            ('semantic' in p and 'semantic' in t))

def v16(pc, lv, ir, sp=0.0, dig=False, t0r=0.0):
    if dig: return 'semantic_diverse'
    if pc > 0.35: return 'morph_uniform/relational_geom'
    elif pc > 0.30:
        if lv >= 0.80 and sp > 0.07: return 'phonol_scatter'
        return 'morph_uniform/relational_geom'
    elif pc > 0.195:
        if lv >= 0.50: return 'morph_moderate' if ir < 0.40 else 'phonol_scatter'
        elif ir < 0.30: return 'morph_moderate'
        elif ir >= 0.60:
            if t0r >= 0.40 and sp < 0.07: return 'phonol_scatter'
            return 'semantic_diverse'
        return 'borderline'
    elif pc > 0.08:
        if lv >= 0.50:
            if ir >= 0.40: return 'phonol_scatter' if lv >= 0.70 else 'semantic_diverse'
            if lv >= 0.70 and ir > 0.05 and sp > 0.11: return 'morph_moderate'
            return 'phonol_scatter'
        elif ir >= 0.95:
            if t0r >= 0.70 and sp < 0.07: return 'phonol_scatter'
            if sp < 0.03: return 'phonol_scatter'           # Rule A: un_verb class
            return 'factual_local/translation'
        elif ir >= 0.60:
            if t0r >= 0.40:
                if lv == 0.0: return 'semantic_diverse'     # Rule C: er_noun2 class
                return 'phonol_scatter'
            return 'semantic_diverse'
        elif lv == 0.0 and 0.20 <= ir < 0.60: return 'phonol_scatter'
        elif lv == 0.0 and ir < 0.20: return 'semantic_diverse'
        elif 0.0 < lv < 0.50 and 0.20 <= ir < 0.60:
            return 'phonol_scatter' if t0r >= 0.80 else 'semantic_diverse'
        elif ir < 0.20: return 'phonol_scatter-allomorph'
        return 'borderline'
    elif pc > 0.05:
        if ir >= 0.85 and lv < 0.15: return 'translation/factual_local'
        elif lv > 0.15 and ir > 0.80: return 'polar_local-partial'
        elif lv > 0.15: return 'borderline'
        return 'polar_local'
    else:
        if ir < 0.20: return 'phonol_scatter'               # Rule B: ary class
        return 'polar_local-partial' if lv > 0.15 else 'polar_local'

# =====================================================================
# 20 OOD AXES — never seen in any benchmark
# Expected labels are based on linguistic analysis, NOT geometric data.
# These expectations will be compared against v16 predictions.
#
# Key:
#   morph_uniform     = regular, high-consistency suffix/prefix, high LOO
#   morph_moderate    = moderate consistency, partial LOO generalization
#   phonol_scatter    = derivational/irregular, partial axis, low-to-mid LOO
#   semantic_diverse  = cross-category shift, each pair semantically unique
#   polar_local       = near-antonym / opposition structure
#   translation       = cross-lingual word mapping
#   factual_local     = CJK/script-switching, vocabulary-gapped translation
# =====================================================================

WILD = [
    # GROUP 1: English derivational suffixes (all novel)
    ('de_pfx',
     [('activate','deactivate'),('value','devalue'),('frost','defrost'),
      ('code','decode'),('bug','debug'),('brief','debrief'),
      ('ice','deice'),('board','deboard')],
     [('compress','decompress'),('hydrate','dehydrate'),('rail','derail')],
     'phonol_scatter',
     'reversal/removal prefix; semantic diversity expected (defrost≠decode≠debug)'),

    ('mis_pfx',
     [('lead','mislead'),('use','misuse'),('judge','misjudge'),
      ('trust','mistrust'),('spell','misspell'),('understand','misunderstand'),
      ('place','misplace'),('read','misread')],
     [('handle','mishandle'),('quote','misquote'),('count','miscount')],
     'phonol_scatter',
     'error prefix; moderate semantic coherence (all = wrong doing)'),

    ('over_pfx',
     [('look','overlook'),('rule','overrule'),('come','overcome'),
      ('flow','overflow'),('rate','overrate'),('load','overload'),
      ('pay','overpay'),('turn','overturn')],
     [('sleep','oversleep'),('heat','overheat'),('work','overwork')],
     'phonol_scatter',
     'excess/above prefix; semantically diverse (overlook≠overflow≠overcome)'),

    ('under_pfx',
     [('mine','undermine'),('cut','undercut'),('value','undervalue'),
      ('pay','underpay'),('rate','underrate'),('line','underline'),
      ('score','underscore'),('go','undergo')],
     [('cover','undercover'),('take','undertake'),('write','underwrite')],
     'semantic_diverse',
     'multiple distinct senses: under=below, under=insufficient; cross-category shift'),

    ('dom_sfx',
     [('king','kingdom'),('free','freedom'),('bore','boredom'),
      ('star','stardom'),('wise','wisdom'),('random','random'),
      ('duke','dukedom'),('serf','serfdom')],
     [('official','officialdom'),('puppet','puppetdom'),('fan','fandom')],
     'phonol_scatter',
     'domain/state suffix; moderate semantic coherence'),

    ('ship_sfx',
     [('friend','friendship'),('own','ownership'),('leader','leadership'),
      ('hard','hardship'),('partner','partnership'),('scholar','scholarship'),
      ('citizen','citizenship'),('member','membership')],
     [('champion','championship'),('author','authorship'),('sponsor','sponsorship')],
     'phonol_scatter',
     'state/role suffix; mixed concrete (ownership) and abstract (hardship)'),

    ('hood_sfx',
     [('child','childhood'),('man','manhood'),('woman','womanhood'),
      ('neighbor','neighborhood'),('brother','brotherhood'),('false','falsehood'),
      ('likely','likelihood'),('adult','adulthood')],
     [('parent','parenthood'),('knight','knighthood'),('priest','priesthood')],
     'phonol_scatter',
     'state/period/community suffix; semantically diverse'),

    ('ness3',
     [('aware','awareness'),('fit','fitness'),('ill','illness'),
      ('still','stillness'),('whole','wholeness'),('lone','loneliness'),
      ('apt','aptness'),('keen','keenness')],
     [('bald','baldness'),('blue','blueness'),('just','justness')],
     'morph_moderate',
     'NEW -ness set, simpler adjectives; expects consistent morph axis'),

    ('fy_sfx',
     [('simple','simplify'),('class','classify'),('solid','solidify'),
      ('clear','clarify'),('just','justify'),('pure','purify'),
      ('beauty','beautify'),('terror','terrify')],
     [('final','finalize'),('rigid','rigidify'),('intense','intensify')],
     'phonol_scatter',
     'causative suffix; phonological variation, semantic coherence moderate'),

    ('ence_sfx',
     [('confident','confidence'),('evident','evidence'),('violent','violence'),
      ('silent','silence'),('patient','patience'),('absent','absence'),
      ('present','presence'),('different','difference')],
     [('innocent','innocence'),('prudent','prudence'),('frequent','frequency')],
     'phonol_scatter',
     'nominalization of -ent adj; phonologically irregular (silent→silence)'),

    # GROUP 2: Inflectional / regular morphology (novel instances)
    ('ed_irr',
     [('go','went'),('have','had'),('make','made'),('come','came'),
      ('say','said'),('see','saw'),('take','took'),('know','knew')],
     [('get','got'),('give','gave'),('find','found')],
     'phonol_scatter',
     'strong past tense (cross-lexical suppletion class); irregular ablaut variety'),

    ('past_t',
     [('want','wanted'),('need','needed'),('start','started'),
      ('wait','waited'),('land','landed'),('end','ended'),
      ('add','added'),('rest','rested')],
     [('load','loaded'),('plant','planted'),('test','tested')],
     'morph_moderate',
     '-ed past tense, -t ending verbs; should mirror ed_reg'),

    ('pl_ves',
     [('leaf','leaves'),('loaf','loaves'),('half','halves'),
      ('wolf','wolves'),('wife','wives'),('knife','knives'),
      ('life','lives'),('self','selves')],
     [('shelf','shelves'),('calf','calves'),('elf','elves')],
     'phonol_scatter',
     'f→ves plural; phonologically specific, limited lexical class'),

    # GROUP 3: Semantic / relational
    ('capital',
     [('England','London'),('France','Paris'),('Germany','Berlin'),
      ('Italy','Rome'),('Spain','Madrid'),('Japan','Tokyo'),
      ('China','Beijing'),('Russia','Moscow')],
     [('Egypt','Cairo'),('Korea','Seoul'),('Australia','Canberra')],
     'relational_geom',
     'reverse of relational axis (country→capital); should still be relational_geom'),

    ('hypernym',
     [('dog','animal'),('rose','flower'),('oak','tree'),
      ('salmon','fish'),('eagle','bird'),('cobra','snake'),
      ('piano','instrument'),('hammer','tool')],
     [('tulip','flower'),('trout','fish'),('flute','instrument')],
     'semantic_diverse',
     'ISA hypernymy; each pair crosses a different semantic category boundary'),

    ('antonym3',
     [('accept','reject'),('build','destroy'),('attack','defend'),
      ('gather','scatter'),('expand','contract'),('advance','retreat'),
      ('increase','decrease'),('arrive','depart')],
     [('unite','divide'),('succeed','fail'),('ascend','descend')],
     'polar_local',
     'verb antonyms (action pairs); should cluster with adj_ant / abstract_ant'),

    # GROUP 4: Cross-lingual (novel language pairs)
    ('en_ru',
     [('house','дом'),('water','вода'),('sun','солнце'),('book','книга'),
      ('day','день'),('night','ночь'),('cat','кот'),('dog','пёс')],
     [('fire','огонь'),('moon','луна'),('sea','море')],
     'factual_local',
     'EN→RU; Cyrillic script, likely multi-token Cyrillic → factual_local (high irred)'),

    ('en_ar',
     [('house','بيت'),('water','ماء'),('sun','شمس'),('book','كتاب'),
      ('day','يوم'),('night','ليل'),('cat','قطة'),('dog','كلب')],
     [('fire','نار'),('moon','قمر'),('sea','بحر')],
     'factual_local',
     'EN→AR; Arabic script, likely multi-token → factual_local'),

    ('en_ko',
     [('house','집'),('water','물'),('sun','해'),('book','책'),
      ('day','날'),('night','밤'),('cat','고양이'),('fire','불')],
     [('moon','달'),('sea','바다'),('tree','나무')],
     'factual_local',
     'EN→KO; Korean script, mixed single/multi-token → factual_local'),

    ('en_hi',
     [('house','घर'),('water','पानी'),('sun','सूरज'),('book','किताब'),
      ('day','दिन'),('night','रात'),('cat','बिल्ली'),('dog','कुत्ता')],
     [('fire','आग'),('moon','चाँद'),('sea','समुद्र')],
     'factual_local',
     'EN→HI; Devanagari script → factual_local (expect multi-token, high irred)'),
]

def run_wild(axes):
    print("  computing %d axes..." % len(axes), flush=True)
    rows = []
    for name, train, ho, expected, notes in axes:
        ax, valid, pc, sp = axis_spread(train)
        if ax is None or len(valid) < 2:
            print("  SKIP %s (too few valid)" % name)
            continue
        lv = loo(ax, valid, RELAXED_MASK)
        ir, t0r = irred(ax, ho, RELAXED_MASK)
        dig = all(tok.decode([sid]).strip().isdigit() for _,_,sid,_ in valid)
        p = v16(pc, lv, ir, sp, dig, t0r)
        ok = match(p, expected)
        rows.append((name, pc, lv, ir, t0r, sp, p, expected, ok, notes))
    return rows

print("\nDAY 346: v16 OOD wild scan — 20 brand-new axes")
print("="*70)

rows = run_wild(WILD)

print("\nPART A: full results table")
print("  %-14s %5s %5s %5s %5s %5s  %-22s %-22s %s" % (
    'axis','pc','loo','irr','t0r','sp','v16_pred','expected','ok'))
print("  " + "-"*110)
for r in rows:
    name, pc, lv, ir, t0r, sp, p, exp, ok, _ = r
    tag = '✓' if ok else '✗'
    print("  %-14s %5.3f %4.0f%% %5.2f %5.2f %5.3f  %-22s %-22s %s" % (
        name, pc, lv*100, ir, t0r, sp, p[:22], exp[:22], tag))

sc = sum(1 for r in rows if r[8])
print("\nPART B: score = %d/%d = %.0f%%" % (sc, len(rows), 100*sc/len(rows) if rows else 0))

print("\nPART C: failures and surprises")
for r in rows:
    name, pc, lv, ir, t0r, sp, p, exp, ok, notes = r
    if not ok:
        print("  FAIL %-14s pred=%-22s expected=%s" % (name, p[:22], exp))
        print("       pc=%.3f lv=%.0f%% ir=%.2f t0r=%.2f sp=%.3f" % (pc,lv*100,ir,t0r,sp))
        print("       note: %s" % notes)

print("\nPART D: by group")
groups = [
    ('English derivational',  ['de_pfx','mis_pfx','over_pfx','under_pfx','dom_sfx','ship_sfx','hood_sfx','ness3','fy_sfx','ence_sfx']),
    ('Inflectional/regular',  ['ed_irr','past_t','pl_ves']),
    ('Semantic/relational',   ['capital','hypernym','antonym3']),
    ('Cross-lingual',         ['en_ru','en_ar','en_ko','en_hi']),
]
for gname, names in groups:
    gs = [r for r in rows if r[0] in names]
    ok = sum(1 for r in gs if r[8])
    print("  %-26s %d/%d = %.0f%%  %s" % (gname, ok, len(gs), 100*ok/len(gs) if gs else 0,
          ' '.join(['✓' if r[8] else '✗'+r[0] for r in gs])))

print("\nPART E: feature distributions by expected category")
from collections import defaultdict
cat_feats = defaultdict(list)
for r in rows:
    cat_feats[r[7]].append(r)
for cat, rs in sorted(cat_feats.items()):
    pcs  = [r[1] for r in rs]
    lvs  = [r[2] for r in rs]
    irs  = [r[3] for r in rs]
    sps  = [r[5] for r in rs]
    print("  %-22s n=%d  pc=%.3f±%.3f  lv=%.0f%%±%.0f%%  ir=%.2f±%.2f  sp=%.3f±%.3f" % (
        cat, len(rs),
        np.mean(pcs), np.std(pcs),
        np.mean(lvs)*100, np.std(lvs)*100,
        np.mean(irs), np.std(irs),
        np.mean(sps), np.std(sps)))
