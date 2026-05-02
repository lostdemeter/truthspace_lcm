import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)

EN_MASK = np.zeros(len(W_E), dtype=bool)
ZH_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if w and w.isalpha() and w.isascii() and len(w) >= 2: EN_MASK[i] = True
    if w and any('\u4e00' <= c <= '\u9fff' for c in w): ZH_MASK[i] = True

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

def get_idx(word, zh=False):
    if zh:
        ids = tok(word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return ids[0]
        return None
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return ids[0]
    return None

def mean_axis(chords):
    m = chords.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-8)

def get_chords(pairs, zh=False):
    chords = []; valid = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_E[ti] - W_E[si]
        n = np.linalg.norm(d)
        if n > 1e-8:
            chords.append(d / n)
            valid.append((s, t, si, ti))
    return np.array(chords, dtype=np.float64), valid

W_n = np.array([normed(v) for v in W_E], dtype=np.float32)

print("\nDAY 366: The Sentiment Transfer Puzzle")
print("="*70)
print("EN sentiment cross-lingual cosine = 0.43, yet zero-shot EN→ZH = 0%")
print("Gender cosine = 0.76, zero-shot EN→ZH = 100%")
print("What breaks the sentiment transfer?")
print()

EN_GENDER = [
    ('king','queen'),('man','woman'),('boy','girl'),
    ('father','mother'),('son','daughter'),('husband','wife'),
    ('uncle','aunt'),('prince','princess'),('actor','actress'),('waiter','waitress'),
]
ZH_GENDER = [
    ('男人','女人'),('国王','女王'),('父亲','母亲'),('儿子','女儿'),
    ('丈夫','妻子'),('叔叔','阿姨'),('王子','公主'),('男孩','女孩'),
]
EN_SENTIMENT = [
    ('bad','good'),('ugly','beautiful'),('hate','love'),('sad','happy'),
    ('dark','bright'),('wrong','right'),('evil','good'),('poor','rich'),
]
ZH_SENTIMENT = [('坏','好'),('丑','美'),('恨','爱'),('悲伤','快乐'),('暗','亮')]

# Build axes
en_chords_g, en_valid_g = get_chords(EN_GENDER, zh=False)
zh_chords_g, zh_valid_g = get_chords(ZH_GENDER, zh=True)
en_chords_s, en_valid_s = get_chords(EN_SENTIMENT, zh=False)
zh_chords_s, zh_valid_s = get_chords(ZH_SENTIMENT, zh=True)

ax_en_g = mean_axis(en_chords_g)
ax_zh_g = mean_axis(zh_chords_g)
ax_en_s = mean_axis(en_chords_s)
ax_zh_s = mean_axis(zh_chords_s)

print("Cross-lingual cosines:")
print("  Gender:    EN-ZH cos = %.5f" % float(np.dot(ax_en_g.astype(np.float32), ax_zh_g.astype(np.float32))))
print("  Sentiment: EN-ZH cos = %.5f" % float(np.dot(ax_en_s.astype(np.float32), ax_zh_s.astype(np.float32))))
print()

# ====================================================================
# PHASE 1: Diagnose what happens during EN→ZH sentiment transfer
# ====================================================================
print("Phase 1: Step-by-step diagnosis of EN→ZH sentiment transfer")
print("  For each ZH pair, apply EN axis and inspect what we actually predict")
print()

print("  %-16s  %-16s  %-8s  %-8s  %-8s  %-30s" % (
    "src(ZH)", "tgt(ZH)", "scale", "pred_rank", "tgt_cos", "top5 predictions"))
print("  " + "-"*86)

# EN axis scale from EN training pairs
en_scales_s = []
for s, t, si, ti in en_valid_s:
    chord = W_E[ti] - W_E[si]
    proj = float(chord @ ax_en_s)
    en_scales_s.append(proj)
scale_en = abs(np.mean(en_scales_s))

for s, t, si, ti in zh_valid_s:
    pred_emb = W_E[si] + scale_en * ax_en_s
    pred_n = normed(pred_emb.astype(np.float32))
    sims = W_n @ pred_n
    sims_zh = sims.copy(); sims_zh[~ZH_MASK] = -1.0; sims_zh[si] = -1.0
    top5_idx = np.argsort(sims_zh)[::-1][:5]
    top5_words = [tok.decode([i]).strip() for i in top5_idx]
    # Rank of target
    sims_rank = sims.copy(); sims_rank[si] = -1.0
    all_sorted = np.argsort(sims_rank)[::-1]
    rank_tgt = int(np.where(all_sorted == ti)[0][0]) + 1
    rank_zh_only = int(np.where(np.argsort(sims_zh)[::-1] == ti)[0][0]) + 1 if ti in np.argsort(sims_zh)[::-1] else -1

    # Cosine of prediction to target
    tgt_cos = float(sims[ti])
    print("  %-16s  %-16s  %-8.4f  %-8d  %-8.4f  %s" % (
        s, t, scale_en, rank_zh_only, tgt_cos, str(top5_words)))

print()

# ====================================================================
# PHASE 2: Chord-level analysis — what is different about sentiment?
# ====================================================================
print("Phase 2: Chord magnitude analysis — gender vs sentiment")
print("  EN chord magnitudes along their respective axes")
print()

def chord_stats(pairs, axis, zh=False):
    projs = []; mags = []
    for s, t in pairs:
        si = get_idx(s, zh=zh); ti = get_idx(t, zh=zh)
        if si is None or ti is None: continue
        d = W_E[ti] - W_E[si]
        mags.append(np.linalg.norm(d))
        projs.append(float(d @ axis))
    return np.array(projs), np.array(mags)

print("  %-16s  %-8s  %-8s  %-8s  %-8s  %-8s" % (
    "Axis/Lang", "proj_mean", "proj_std", "mag_mean", "cos_mean", "n"))
print("  " + "-"*58)

for ax_name, pairs, axis, zh in [
    ("EN_gender",    EN_GENDER,    ax_en_g, False),
    ("ZH_gender",    ZH_GENDER,    ax_zh_g, True),
    ("EN_sentiment", EN_SENTIMENT, ax_en_s, False),
    ("ZH_sentiment", ZH_SENTIMENT, ax_zh_s, True),
]:
    projs, mags = chord_stats(pairs, axis, zh=zh)
    if len(projs) == 0: continue
    cos_mean = np.mean(projs / (mags + 1e-8))
    print("  %-16s  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8d" % (
        ax_name, projs.mean(), projs.std(), mags.mean(), cos_mean, len(projs)))

print()

# Now: using EN SENTIMENT axis, what scale do ZH sentiment pairs project to?
print("  Scale mismatch: EN sentiment axis applied to ZH pairs")
print()
for s, t in ZH_SENTIMENT:
    si = get_idx(s, zh=True); ti = get_idx(t, zh=True)
    if si is None or ti is None:
        print("    %s → %s: SKIP (not single-token)" % (s, t))
        continue
    d = W_E[ti] - W_E[si]
    proj = float(d @ ax_en_s)
    mag  = np.linalg.norm(d)
    print("    %s → %s:  proj_on_EN_axis=%.4f  chord_mag=%.4f  cos=%.4f" % (
        s, t, proj, mag, proj/(mag+1e-8)))

print()
print("  For comparison: EN sentiment pairs projected on EN axis:")
for s, t in EN_SENTIMENT[:5]:
    si = get_idx(s, zh=False); ti = get_idx(t, zh=False)
    if si is None or ti is None: continue
    d = W_E[ti] - W_E[si]
    proj = float(d @ ax_en_s)
    mag  = np.linalg.norm(d)
    print("    %s → %s:  proj_on_EN_axis=%.4f  chord_mag=%.4f  cos=%.4f" % (
        s, t, proj, mag, proj/(mag+1e-8)))

print()

# ====================================================================
# PHASE 3: What IS the nearest neighbor of ZH sentiment transfer prediction?
# ====================================================================
print("Phase 3: What scale would actually work for EN→ZH sentiment?")
print("  Test scales from 0.1 to 10.0 and find what score each ZH pair needs")
print()

for s, t in ZH_SENTIMENT:
    si = get_idx(s, zh=True); ti = get_idx(t, zh=True)
    if si is None or ti is None: continue
    # Find the scale that puts the target in top-5
    best_rank = 9999; best_scale = None
    for sc in np.linspace(0.1, 20.0, 200):
        pred = normed((W_E[si] + sc * ax_en_s).astype(np.float32))
        sims_zh = W_n @ pred
        sims_zh[~ZH_MASK] = -1.0; sims_zh[si] = -1.0
        rank = int(np.where(np.argsort(sims_zh)[::-1] == ti)[0][0]) + 1
        if rank < best_rank:
            best_rank = rank
            best_scale = sc
    print("  %s → %s:  best_scale=%.3f  best_rank=%d  (EN_scale=%.3f)" % (
        s, t, best_scale, best_rank, scale_en))

print()

# ====================================================================
# PHASE 4: Use ZH-calibrated scale — does it work?
# ====================================================================
print("Phase 4: ZH-calibrated scale for EN→ZH sentiment transfer")
print("  Use ZH training pairs to calibrate scale, then apply EN axis direction")
print()

# Calibrate scale on ZH pairs using EN axis direction
zh_projs_on_en = []
for s, t, si, ti in zh_valid_s:
    d = W_E[ti] - W_E[si]
    proj = float(d @ ax_en_s)
    zh_projs_on_en.append(proj)

scale_zh_on_en = abs(np.mean(zh_projs_on_en)) if zh_projs_on_en else scale_en
print("  EN axis scale (calibrated on EN pairs):  %.4f" % scale_en)
print("  EN axis scale (calibrated on ZH pairs):  %.4f" % scale_zh_on_en)
print()

# Test with ZH-calibrated scale
hits_zh_cal = 0; total = 0
print("  Using ZH-calibrated scale for EN→ZH:")
for s, t, si, ti in zh_valid_s:
    total += 1
    pred = normed((W_E[si] + scale_zh_on_en * ax_en_s).astype(np.float32))
    sims_zh = W_n @ pred; sims_zh[~ZH_MASK] = -1.0; sims_zh[si] = -1.0
    top5 = np.argsort(sims_zh)[::-1][:5]
    hit = ti in top5
    if hit: hits_zh_cal += 1
    top5_words = [tok.decode([i]).strip() for i in top5]
    print("    %s → %s: %s  top5=%s" % (s, t, 'HIT' if hit else 'miss', top5_words))

print("  ZH-calibrated accuracy: %d/%d = %.3f" % (hits_zh_cal, total, hits_zh_cal/total if total > 0 else 0))
print()

# ====================================================================
# PHASE 5: Why is gender so easy and sentiment so hard?
# ====================================================================
print("Phase 5: Gender vs Sentiment — what makes transfer work?")
print()

# 1. Per-pair cosine of chord with mean axis
print("  Per-pair chord cosine with mean axis (axis coherence):")
print()
print("  %-18s  %-12s  %-12s" % ("pair", "EN_gender", "EN_sentiment"))
print("  " + "-"*44)
all_coh_g = []; all_coh_s = []
for (sg, tg), (ss, ts) in zip(EN_GENDER[:8], EN_SENTIMENT[:8]):
    ig = get_idx(sg); ig_t = get_idx(tg)
    is_ = get_idx(ss); is_t = get_idx(ts)
    coh_g = coh_s = 0
    if ig and ig_t:
        d = W_E[ig_t] - W_E[ig]
        coh_g = float(np.dot(normed(d).astype(np.float32), ax_en_g.astype(np.float32)))
        all_coh_g.append(coh_g)
    if is_ and is_t:
        d = W_E[is_t] - W_E[is_]
        coh_s = float(np.dot(normed(d).astype(np.float32), ax_en_s.astype(np.float32)))
        all_coh_s.append(coh_s)
    print("  %-18s  %-12.4f  %-12s" % (
        "%s→%s" % (sg, tg), coh_g, "%s→%s" % (ss, ts)))

print()
print("  Mean coherence: EN_gender=%.4f  EN_sentiment=%.4f" % (
    np.mean(all_coh_g) if all_coh_g else 0,
    np.mean(all_coh_s) if all_coh_s else 0))
print()

# 2. Cross-lingual projection asymmetry
print("  Cross-lingual projection: how much does EN axis explain ZH chord direction?")
print()
print("  %-12s  %-12s  %-12s" % ("Axis", "EN_ax->EN_cds", "EN_ax->ZH_cds"))
print("  " + "-"*38)

for ax_name, ax_dir, en_pairs, zh_pairs in [
    ("gender",    ax_en_g, EN_GENDER,    ZH_GENDER),
    ("sentiment", ax_en_s, EN_SENTIMENT, ZH_SENTIMENT),
]:
    en_coh = np.mean([float(np.dot(normed(W_E[get_idx(t)] - W_E[get_idx(s)]).astype(np.float32), ax_dir.astype(np.float32)))
                      for s,t in en_pairs if get_idx(s) and get_idx(t)])
    zh_coh = np.mean([float(np.dot(normed(W_E[get_idx(t,zh=True)] - W_E[get_idx(s,zh=True)]).astype(np.float32), ax_dir.astype(np.float32)))
                      for s,t in zh_pairs if get_idx(s,zh=True) and get_idx(t,zh=True)])
    print("  %-12s  %-12.4f  %-12.4f" % (ax_name, en_coh, zh_coh))

print()

# 3. Cross-lingual NN contamination check for sentiment
print("  Nearest neighbors of ZH words in sentiment pairs (do EN words appear nearby?)")
print()
for s, t in ZH_SENTIMENT:
    si = get_idx(s, zh=True); ti = get_idx(t, zh=True)
    if si is None or ti is None: continue
    sims_s = W_n @ W_n[si]
    top10 = np.argsort(sims_s)[::-1][1:11]
    top10_words = [tok.decode([i]).strip() for i in top10]
    print("  NNs of '%s': %s" % (s, top10_words[:8]))

print()

# ====================================================================
# PHASE 6: Expand the ZH sentiment evaluation set
# ====================================================================
print("Phase 6: Expanded ZH sentiment pairs — is it the eval set that's too small/hard?")
print()

ZH_SENTIMENT_EXT = [
    ('坏','好'),('丑','美'),('恨','爱'),('悲伤','快乐'),('暗','亮'),
    ('错误','正确'),('冷','热'),('慢','快'),('难','易'),('脏','干净'),
    ('穷','富'),('弱','强'),('旧','新'),('假','真'),('低','高'),
]

# Coverage check
ok = sum(1 for s,t in ZH_SENTIMENT_EXT
         if get_idx(s, zh=True) is not None and get_idx(t, zh=True) is not None)
print("  Extended ZH sentiment coverage: %d/%d single-token" % (ok, len(ZH_SENTIMENT_EXT)))
print()

# Build axis from extended set
zh_chords_ext, zh_valid_ext = get_chords(ZH_SENTIMENT_EXT, zh=True)
if len(zh_chords_ext) >= 2:
    ax_zh_s_ext = mean_axis(zh_chords_ext)
    cos_ext = float(np.dot(ax_en_s.astype(np.float32), ax_zh_s_ext.astype(np.float32)))
    print("  Extended ZH sentiment axis EN-ZH cosine: %.5f" % cos_ext)
    print()

    # Test EN→ZH transfer on extended set
    hits_ext = 0; total_ext = 0
    # Calibrate scale on extended ZH pairs
    zh_projs_ext = [float((W_E[ti] - W_E[si]) @ ax_en_s)
                    for s,t,si,ti in zh_valid_ext if si is not None and ti is not None]
    scale_ext = abs(np.mean(zh_projs_ext)) if zh_projs_ext else scale_en

    print("  EN→ZH transfer (extended eval, ZH-calibrated scale=%.4f):" % scale_ext)
    for s, t, si, ti in zh_valid_ext:
        total_ext += 1
        pred = normed((W_E[si] + scale_ext * ax_en_s).astype(np.float32))
        sims_zh = W_n @ pred; sims_zh[~ZH_MASK] = -1.0; sims_zh[si] = -1.0
        top5 = np.argsort(sims_zh)[::-1][:5]
        hit = ti in top5
        if hit: hits_ext += 1
        top5_w = [tok.decode([i]).strip() for i in top5]
        print("    %s→%s: %s  top5=%s" % (s, t, 'HIT' if hit else 'miss', top5_w))
    print("  Extended accuracy: %d/%d = %.3f" % (hits_ext, total_ext, hits_ext/total_ext if total_ext > 0 else 0))

print()

# ====================================================================
# PHASE 7: The ZH sentiment axis — is it self-consistent?
# ====================================================================
print("Phase 7: ZH sentiment axis self-consistency — does it work within ZH?")
print()

# Using ZH axis to predict ZH pairs (same-language transfer)
hits_zh_self = 0; total_zh_self = 0
for s, t, si, ti in zh_valid_s:
    total_zh_self += 1
    # ZH scale
    zh_projs_self = [float((W_E[get_idx(tt, zh=True)] - W_E[get_idx(ss, zh=True)]) @ ax_zh_s)
                     for ss,tt in ZH_SENTIMENT
                     if get_idx(ss, zh=True) and get_idx(tt, zh=True)
                     and (ss,tt) != (s,t)]
    scale_zh_self = abs(np.mean(zh_projs_self)) if zh_projs_self else 1.0

    pred = normed((W_E[si] + scale_zh_self * ax_zh_s).astype(np.float32))
    sims_zh = W_n @ pred; sims_zh[~ZH_MASK] = -1.0; sims_zh[si] = -1.0
    top5 = np.argsort(sims_zh)[::-1][:5]
    hit = ti in top5
    if hit: hits_zh_self += 1
    top5_w = [tok.decode([i]).strip() for i in top5]
    print("  ZH self: %s→%s: %s  top5=%s" % (s, t, 'HIT' if hit else 'miss', top5_w))

print("  ZH self-consistency: %d/%d = %.3f" % (hits_zh_self, total_zh_self,
    hits_zh_self/total_zh_self if total_zh_self > 0 else 0))
print()

# ZH→EN transfer (reverse direction)
hits_zh_en = 0; total_zh_en = 0
zh_projs_rev = [float((W_E[get_idx(t)] - W_E[get_idx(s)]) @ ax_zh_s)
                for s,t in EN_SENTIMENT if get_idx(s) and get_idx(t)]
scale_zh_en = abs(np.mean(zh_projs_rev)) if zh_projs_rev else 1.0

print("  ZH axis → EN pairs (ZH→EN transfer, scale=%.4f):" % scale_zh_en)
for s, t in EN_SENTIMENT:
    si = get_idx(s); ti = get_idx(t)
    if si is None or ti is None: continue
    total_zh_en += 1
    pred = normed((W_E[si] + scale_zh_en * ax_zh_s).astype(np.float32))
    sims_en = W_n @ pred; sims_en[~EN_MASK] = -1.0; sims_en[si] = -1.0
    top5 = np.argsort(sims_en)[::-1][:5]
    hit = ti in top5
    if hit: hits_zh_en += 1
    top5_w = [tok.decode([i]).strip() for i in top5]
    print("  ZH→EN: %s→%s: %s  top5=%s" % (s, t, 'HIT' if hit else 'miss', top5_w))

print("  ZH→EN accuracy: %d/%d = %.3f" % (hits_zh_en, total_zh_en,
    hits_zh_en/total_zh_en if total_zh_en > 0 else 0))
print()

# ====================================================================
# PHASE 8: Why does gender work but sentiment doesn't?
# ====================================================================
print("Phase 8: Direct comparison of what makes gender transfer succeed")
print("  Key metrics side-by-side: gender vs sentiment")
print()

def compute_transfer_stats(en_pairs, zh_pairs, ax_en, ax_zh, label):
    # EN self, ZH self, EN→ZH, ZH→EN
    results = {}

    for direction, src_pairs, tgt_pairs, src_ax, tgt_ax, src_zh, tgt_zh in [
        ("EN self",  en_pairs, en_pairs, ax_en, ax_en, False, False),
        ("ZH self",  zh_pairs, zh_pairs, ax_zh, ax_zh, True,  True),
        ("EN→ZH",   en_pairs, zh_pairs, ax_en, ax_en, False, True),
        ("ZH→EN",   zh_pairs, en_pairs, ax_zh, ax_zh, True,  False),
    ]:
        # Calibrate scale from src pairs
        src_projs = [float((W_E[get_idx(t, zh=src_zh)] - W_E[get_idx(s, zh=src_zh)]) @ src_ax)
                     for s,t in src_pairs if get_idx(s, zh=src_zh) and get_idx(t, zh=src_zh)]
        scale = abs(np.mean(src_projs)) if src_projs else 1.0

        mask = ZH_MASK if tgt_zh else EN_MASK
        hits = 0; total = 0
        for s, t in tgt_pairs:
            si = get_idx(s, zh=tgt_zh); ti = get_idx(t, zh=tgt_zh)
            if si is None or ti is None: continue
            total += 1
            pred = normed((W_E[si] + scale * src_ax).astype(np.float32))
            sims = W_n @ pred; sims[mask == False] = -1.0; sims[si] = -1.0
            top5 = np.argsort(sims)[::-1][:5]
            if ti in top5: hits += 1

        results[direction] = (hits, total, hits/total if total > 0 else 0)

    return results

print("  %-12s  %-12s  %-12s  %-12s  %-12s" % (
    "axis", "EN_self", "ZH_self", "EN→ZH", "ZH→EN"))
print("  " + "-"*60)

for ax_name, en_p, zh_p, ax_en_, ax_zh_ in [
    ("gender",    EN_GENDER, ZH_GENDER, ax_en_g, ax_zh_g),
    ("sentiment", EN_SENTIMENT, ZH_SENTIMENT, ax_en_s, ax_zh_s),
]:
    r = compute_transfer_stats(en_p, zh_p, ax_en_, ax_zh_, ax_name)
    print("  %-12s  %-12.3f  %-12.3f  %-12.3f  %-12.3f" % (
        ax_name,
        r["EN self"][2], r["ZH self"][2],
        r["EN→ZH"][2], r["ZH→EN"][2]))

print()

# ====================================================================
# SUMMARY
# ====================================================================
print("="*70)
print("SUMMARY: Day 366 — The Sentiment Transfer Puzzle")
print("="*70)
print()
print("  Gender:    cross-lingual cos=0.76  EN_self=100%  EN→ZH=100%")
print("  Sentiment: cross-lingual cos=0.43  EN_self=?     EN→ZH=0%")
print()
print("  Root cause analysis from phases above:")
print("  [See phase outputs for specific failure modes]")
