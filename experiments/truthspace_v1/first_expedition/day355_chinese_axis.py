import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) < 1: continue
    RELAXED_MASK[i] = True

# Chinese-friendly mask: no space-prefix tokens that are purely noise
ZH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i])
    w2 = w.strip()
    if not w2: continue
    # Accept CJK characters, or short multi-char Chinese words
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in w2)
    if has_cjk: ZH_MASK[i] = True

EN_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and w.isalpha() and w.isascii() and len(w) >= 2: EN_MASK[i] = True

_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [word, ' '+word, word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    _src_cache[word] = ids
    return ids

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_emb_zh(word):
    """Get embedding for a Chinese word. Try without space prefix."""
    ids = tok(word, add_special_tokens=False)['input_ids']
    if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def tokenize_show(word):
    ids = tok(word, add_special_tokens=False)['input_ids']
    return ids, [tok.decode([i]) for i in ids]

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def nn_ret_top(pred_emb, excl_ids, mask, top_k=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    top = np.argsort(sims)[::-1][:top_k*3]
    out = []
    for idx in top:
        if len(out) >= top_k: break
        out.append((tok.decode([int(idx)]).strip(), float(sims[idx]), int(idx)))
    return out

def build_axis(pairs, get_fn=None):
    if get_fn is None: get_fn = get_emb
    chords = []; usable = []; skipped = []
    for s, t in pairs:
        es, _ = get_fn(s); et, _ = get_fn(t)
        if es is None or et is None:
            skipped.append((s,t)); continue
        chords.append(et - es)
        usable.append((s,t))
    if not chords: return None, usable, skipped
    return normed(np.mean(chords, axis=0)), usable, skipped

def best_scale(ax_dir, pairs, mask, get_fn=None):
    if get_fn is None: get_fn = get_emb
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = 0
        for sr, tg in pairs:
            es, _ = get_fn(sr)
            if es is None: continue
            w, _, _ = nn_ret(es + s*ax_dir, source_ids(sr), mask)
            if w == tg: c += 1
        if c > best_a: best_a=c; best_s=s
    return best_s

def eval_axis(ax_dir, s, pairs, mask, get_fn=None, excl_fn=None):
    if get_fn is None: get_fn = get_emb
    if excl_fn is None: excl_fn = source_ids
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_fn(src)
        if es is None: continue
        n += 1
        pred = es + s * ax_dir
        w, sim, _ = nn_ret(pred, excl_fn(src), mask)
        ok = (w == tgt)
        if ok: hits += 1
        details.append((src, tgt, w, ok, sim))
    return hits, n, details

# ====================================================================
# DATASETS
# ====================================================================
EN_GENDER = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('son','daughter'), ('husband','wife'),
    ('uncle','aunt'), ('prince','princess'), ('actor','actress'),
    ('waiter','waitress'),
]
EN_GENDER_TEST = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('husband','wife'),
]

# Chinese gender pairs (Simplified Chinese)
# Each pair: (masculine, feminine)
ZH_GENDER_ALL = [
    ('男人',  '女人'),   # man / woman
    ('国王',  '女王'),   # king / queen
    ('父亲',  '母亲'),   # father / mother
    ('儿子',  '女儿'),   # son / daughter
    ('丈夫',  '妻子'),   # husband / wife
    ('叔叔',  '阿姨'),   # uncle / aunt
    ('王子',  '公主'),   # prince / princess
    ('男孩',  '女孩'),   # boy / girl
    ('兄弟',  '姐妹'),   # brother / sister
    ('男演员','女演员'), # actor / actress
]

# Chinese size pairs (comparative / superlative equivalent)
ZH_SIZE = [
    ('大',  '小'),   # big / small
    ('长',  '短'),   # long / short
    ('高',  '矮'),   # tall / short(height)
    ('快',  '慢'),   # fast / slow
    ('热',  '冷'),   # hot / cold
    ('老',  '年轻'), # old / young
    ('重',  '轻'),   # heavy / light
    ('强',  '弱'),   # strong / weak
]

# Chinese translation equivalents to English test words (for nearest-neighbour check)
EN_ZH_MEANING = [
    ('man',    '男人'),
    ('woman',  '女人'),
    ('king',   '国王'),
    ('queen',  '女王'),
    ('father', '父亲'),
    ('mother', '母亲'),
    ('boy',    '男孩'),
    ('girl',   '女孩'),
    ('husband','丈夫'),
    ('wife',   '妻子'),
]

print("\nDAY 355: Chinese Gender Axis & Cross-Script Transfer")
print("="*70)

# ====================================================================
# PHASE 1: Tokenisation check for Chinese words
# ====================================================================
print("\nPhase 1: Tokenisation check — Chinese words in Qwen2")
print("  (Qwen2 trained on Chinese data — expect mostly single-token)")
print()

zh_single = []; zh_multi = []
for s, t in ZH_GENDER_ALL:
    ids_s, tks_s = tokenize_show(s)
    ids_t, tks_t = tokenize_show(t)
    es, _ = get_emb_zh(s); et, _ = get_emb_zh(t)
    if es is not None and et is not None:
        zh_single.append((s, t))
        print("  SINGLE: %-8s [%s] → %-8s [%s]" % (
            s, tok.decode([tok(s,add_special_tokens=False)['input_ids'][0]]),
            t, tok.decode([tok(t,add_special_tokens=False)['input_ids'][0]])))
    else:
        zh_multi.append((s, t))
        print("  MULTI:  %-8s %s → %-8s %s" % (s, tks_s, t, tks_t))

print()
print("  Single-token: %d / %d" % (len(zh_single), len(ZH_GENDER_ALL)))

# ====================================================================
# PHASE 2: Chinese word neighbourhood — are ZH translations near EN words?
# ====================================================================
print("\nPhase 2: Cross-script neighbourhoods — are ZH translations near EN equivalents?")
print()

for en_w, zh_w in EN_ZH_MEANING:
    en_emb, en_idx = get_emb(en_w)
    zh_emb, zh_idx = get_emb_zh(zh_w)
    if en_emb is None or zh_emb is None:
        print("  %-8s / %-4s: one or both multi-token" % (en_w, zh_w))
        continue
    cos = float(np.dot(normed(en_emb).astype(np.float32),
                       normed(zh_emb).astype(np.float32)))
    # NN of EN word in ZH_MASK (what Chinese word is EN word closest to?)
    nn_zh, sim_zh, _ = nn_ret(en_emb, set(), ZH_MASK)
    # NN of ZH word in EN_MASK
    nn_en, sim_en, _ = nn_ret(zh_emb, set(), EN_MASK)
    print("  %-8s ↔ %-6s  cos=%.4f  EN_NN_in_ZH=%s(%.3f)  ZH_NN_in_EN=%s(%.3f)" % (
        en_w, zh_w, cos, nn_zh, sim_zh, nn_en, sim_en))

# ====================================================================
# PHASE 3: Build Chinese gender axis
# ====================================================================
print("\nPhase 3: Build Chinese gender axis from single-token ZH pairs")
print()

zh_gender_dir, zh_usable, zh_skipped = build_axis(zh_single, get_fn=get_emb_zh)
if zh_gender_dir is None:
    print("  ERROR: no usable Chinese gender pairs!")
else:
    print("  Used %d pairs, skipped %d" % (len(zh_usable), len(zh_skipped)))
    if zh_skipped: print("  Skipped:", zh_skipped)
    s_zh = best_scale(zh_gender_dir, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    h, n, det = eval_axis(zh_gender_dir, s_zh, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    print("  ZH gender axis: scale=%.3f  train_acc=%d/%d=%.0f%%" % (
        s_zh, h, n, 100*h/max(n,1)))
    for s,t,w,ok,sim in det:
        mark = '✓' if ok else '✗'
        print("    %s %-6s → %-6s  [found: %s %.4f]" % (mark, s, t, w, sim))

# Build English gender axis for comparison
en_gender_dir, _, _ = build_axis(EN_GENDER)
s_eg = best_scale(en_gender_dir, EN_GENDER, RELAXED_MASK)

# ====================================================================
# PHASE 4: Axis alignment — cos(ZH_gender, EN_gender)
# ====================================================================
if zh_gender_dir is not None:
    print("\nPhase 4: Axis alignment")
    cos_zh_en = float(np.dot(zh_gender_dir.astype(np.float32),
                              en_gender_dir.astype(np.float32)))
    print("  cos(ZH_gender, EN_gender) = %.4f  [%s]" % (
        cos_zh_en,
        'HIGH (>0.7) → universal' if abs(cos_zh_en) > 0.7 else
        'MED (0.4-0.7) → partial' if abs(cos_zh_en) > 0.4 else
        'LOW (<0.4) → language-specific'))

    # Compare to FR, ES, DE alignments from Day 354
    print()
    print("  Cross-language gender axis alignment summary:")
    print("  EN↔ES: 0.4308  (Day 354)")
    print("  EN↔FR: 0.3814  (Day 354)")
    print("  EN↔DE: 0.2679  (Day 354)")
    print("  EN↔ZH: %.4f  (today)" % cos_zh_en)

    # Also measure axis coherence for ZH
    chords_zh = []
    for s, t in zh_usable:
        es, _ = get_emb_zh(s); et, _ = get_emb_zh(t)
        if es is not None and et is not None: chords_zh.append(normed(et-es))
    if len(chords_zh) >= 2:
        cos_matrix = np.array([[float(np.dot(a.astype(np.float32), b.astype(np.float32)))
                                  for b in chords_zh] for a in chords_zh])
        off_diag = [cos_matrix[i,j] for i in range(len(chords_zh))
                    for j in range(len(chords_zh)) if i != j]
        print()
        print("  ZH gender axis coherence: %.4f (mean pairwise chord cos)" % np.mean(off_diag))

# ====================================================================
# PHASE 5: Zero-shot transfer EN → ZH (English gender axis on Chinese words)
# ====================================================================
if zh_gender_dir is not None and zh_usable:
    print("\nPhase 5: Zero-shot transfer EN→ZH gender axis on Chinese words")
    h, n, det = eval_axis(en_gender_dir, s_eg, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    print("  EN_axis→ZH words:  %d/%d=%.0f%%  (EN scale s=%.3f)" % (
        h, n, 100*h/max(n,1), s_eg))
    for s,t,w,ok,sim in det:
        mark = '✓' if ok else '✗'
        print("    %s %-6s → %-6s  [found: %s %.4f]" % (mark, s, t, w, sim))

    # Optimise scale for EN_axis applied to ZH words
    s_en_on_zh = best_scale(en_gender_dir, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    h2, n, det2 = eval_axis(en_gender_dir, s_en_on_zh, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    print()
    print("  EN_axis→ZH words (optimised scale s=%.3f): %d/%d=%.0f%%" % (
        s_en_on_zh, h2, n, 100*h2/max(n,1)))
    for s,t,w,ok,sim in det2:
        mark = '✓' if ok else '✗'
        print("    %s %-6s → %-6s  [found: %s %.4f]" % (mark, s, t, w, sim))

# ====================================================================
# PHASE 6: Zero-shot transfer ZH → EN (Chinese gender axis on English words)
# ====================================================================
if zh_gender_dir is not None:
    print("\nPhase 6: Zero-shot transfer ZH→EN gender axis on English words")
    h, n, det = eval_axis(zh_gender_dir, s_zh, EN_GENDER_TEST, RELAXED_MASK, get_fn=get_emb)
    print("  ZH_axis→EN words (ZH scale s=%.3f): %d/%d=%.0f%%" % (
        s_zh, h, n, 100*h/max(n,1)))
    for s,t,w,ok,sim in det:
        mark = '✓' if ok else '✗'
        print("    %s %-10s → %-10s  [found: %s %.4f]" % (mark, s, t, w, sim))

    # Optimise scale for ZH_axis applied to EN words
    s_zh_on_en = best_scale(zh_gender_dir, EN_GENDER_TEST, RELAXED_MASK, get_fn=get_emb)
    h2, n, det2 = eval_axis(zh_gender_dir, s_zh_on_en, EN_GENDER_TEST,
                             RELAXED_MASK, get_fn=get_emb)
    print()
    print("  ZH_axis→EN words (optimised scale s=%.3f): %d/%d=%.0f%%" % (
        s_zh_on_en, h2, n, 100*h2/max(n,1)))
    for s,t,w,ok,sim in det2:
        mark = '✓' if ok else '✗'
        print("    %s %-10s → %-10s  [found: %s %.4f]" % (mark, s, t, w, sim))

# ====================================================================
# PHASE 7: Multilingual axis including Chinese
# ====================================================================
print("\nPhase 7: Multilingual gender axis (EN+ZH training)")

if zh_gender_dir is not None and zh_usable:
    # Build combined EN+ZH axis
    all_chords = []
    for s, t in EN_GENDER:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is not None and et is not None: all_chords.append(et - es)
    for s, t in zh_usable:
        es, _ = get_emb_zh(s); et, _ = get_emb_zh(t)
        if es is not None and et is not None: all_chords.append(et - es)
    multi_dir = normed(np.mean(all_chords, axis=0))

    cos_multi_en = float(np.dot(multi_dir.astype(np.float32), en_gender_dir.astype(np.float32)))
    cos_multi_zh = float(np.dot(multi_dir.astype(np.float32), zh_gender_dir.astype(np.float32)))
    print("  cos(EN+ZH_axis, EN_axis)=%.4f  cos(EN+ZH_axis, ZH_axis)=%.4f" % (
        cos_multi_en, cos_multi_zh))

    # Test multilingual axis on EN test and ZH test
    s_m_en = best_scale(multi_dir, EN_GENDER_TEST, RELAXED_MASK, get_fn=get_emb)
    h_m_en, n_en, _ = eval_axis(multi_dir, s_m_en, EN_GENDER_TEST, RELAXED_MASK, get_fn=get_emb)
    s_m_zh = best_scale(multi_dir, zh_usable, ZH_MASK, get_fn=get_emb_zh)
    h_m_zh, n_zh, _ = eval_axis(multi_dir, s_m_zh, zh_usable, ZH_MASK, get_fn=get_emb_zh)

    h_en_only, n_en, _ = eval_axis(en_gender_dir, s_eg, EN_GENDER_TEST, RELAXED_MASK, get_fn=get_emb)
    h_zh_only, n_zh, _ = eval_axis(zh_gender_dir, s_zh, zh_usable, ZH_MASK, get_fn=get_emb_zh)

    print()
    print("  EN test:  EN_mono=%d/%d=%.0f%%  EN+ZH_multi=%d/%d=%.0f%%" % (
        h_en_only, n_en, 100*h_en_only/max(n_en,1),
        h_m_en,    n_en, 100*h_m_en/max(n_en,1)))
    print("  ZH test:  ZH_mono=%d/%d=%.0f%%  EN+ZH_multi=%d/%d=%.0f%%" % (
        h_zh_only, n_zh, 100*h_zh_only/max(n_zh,1),
        h_m_zh,    n_zh, 100*h_m_zh/max(n_zh,1)))

# ====================================================================
# PHASE 8: What does the ZH gender axis LOOK LIKE in English vocabulary?
# ====================================================================
print("\nPhase 8: ZH gender axis — nearest neighbours in EN_MASK")
if zh_gender_dir is not None:
    s_zh_vis = s_zh if 'best_scale' else 1.0
    top_en = nn_ret_top(s_zh_vis * zh_gender_dir, set(), EN_MASK, top_k=10)
    print("  NN of s*ZH_gender_dir in EN_MASK:")
    print("  %s" % [(w,round(c,4)) for w,c,_ in top_en])

    # Also: what EN word does the ZH axis look like when negated?
    top_en_neg = nn_ret_top(-s_zh_vis * zh_gender_dir, set(), EN_MASK, top_k=10)
    print("  NN of -s*ZH_gender_dir in EN_MASK:")
    print("  %s" % [(w,round(c,4)) for w,c,_ in top_en_neg])

    # EN gender axis NN in ZH_MASK
    top_zh = nn_ret_top(s_eg * en_gender_dir, set(), ZH_MASK, top_k=10)
    print()
    print("  NN of s*EN_gender_dir in ZH_MASK:")
    print("  %s" % [(w,round(c,4)) for w,c,_ in top_zh])

    top_zh_neg = nn_ret_top(-s_eg * en_gender_dir, set(), ZH_MASK, top_k=10)
    print("  NN of -s*EN_gender_dir in ZH_MASK:")
    print("  %s" % [(w,round(c,4)) for w,c,_ in top_zh_neg])

# ====================================================================
# PHASE 9: Summary — complete cross-language gender axis picture
# ====================================================================
print("\n" + "="*70)
print("SUMMARY: Day 355 Chinese Axis + Cross-Script Transfer")
print("="*70)
print()
print("  Cross-language gender axis alignment summary:")
print("  (from Day 354 + today)")
print()
if zh_gender_dir is not None:
    print("  cos(EN, ZH) = %.4f" % float(np.dot(zh_gender_dir.astype(np.float32),
                                                   en_gender_dir.astype(np.float32))))
print("  cos(EN, ES) = 0.4308  [Day 354]")
print("  cos(EN, FR) = 0.3814  [Day 354]")
print("  cos(EN, DE) = 0.2679  [Day 354]")
print()
print("  The scale of alignment determines whether the gender axis is:")
print("    cos > 0.7 → universal (meaning-level, language-independent)")
print("    cos ~ 0.3-0.5 → partially shared (semantic overlap)")
print("    cos < 0.1 → orthogonal (language-specific)")
