# phase10p_build_tables.py — Build bias-aware attention tables
# Exec'd from phase10p_simple_machines.py (shares its namespace)
print("\nExtracting weights & building tables...")
head_tables = {}
for li in range(NL):
    attn = model.model.layers[li].self_attn
    ident = torch.eye(HDIM, device="cuda", dtype=torch.bfloat16)
    Wq = torch.zeros(NH, HD, HDIM, dtype=torch.float32)
    Wk = torch.zeros(NKV, HD, HDIM, dtype=torch.float32)
    for s in range(0, HDIM, 512):
        e = min(s + 512, HDIM); chunk = ident[s:e].unsqueeze(0)
        with torch.no_grad():
            qo = attn.q_proj(chunk).float(); ko = attn.k_proj(chunk).float()
        qr = qo[0].reshape(-1, NH, HD); kr = ko[0].reshape(-1, NKV, HD)
        for h in range(NH): Wq[h, :, s:e] = qr[:, h, :].T
        for g in range(NKV): Wk[g, :, s:e] = kr[:, g, :].T
    zi = torch.zeros(1, 1, HDIM, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        qb = attn.q_proj(zi).float()[0, 0]; kb = attn.k_proj(zi).float()[0, 0]
    bq = qb.reshape(NH, HD).cpu(); bk = kb.reshape(NKV, HD).cpu()
    for h in range(NH): Wq[h] -= bq[h].unsqueeze(1)
    for g in range(NKV): Wk[g] -= bk[g].unsqueeze(1)
    routing = layer_cls[li]['routing']
    for h in routing:
        g = h // HPK; sc = 1.0 / math.sqrt(HD)
        bl = torch.zeros(MAXS); cq = torch.zeros(MAXS, HDIM); ck = torch.zeros(MAXS, HDIM)
        for delta in range(MAXS):
            bk_rot = rope_rotate_vector(bk[g], delta, inv_freq)
            Wk_rot = rope_rotate_matrix_cols(Wk[g], delta, inv_freq)
            bl[delta] = (bq[h] @ bk_rot) * sc
            cq[delta] = (Wq[h].T @ bk_rot) * sc
            ck[delta] = (Wk_rot.T @ bq[h]) * sc
        head_tables[(li, h)] = {'baseline': bl, 'c_q': cq, 'c_k': ck}
    del Wq, Wk; torch.cuda.empty_cache()
    if li % 7 == 0: print(f"  Layer {li} done")
print(f"  {len(head_tables)} head tables ready\n")
