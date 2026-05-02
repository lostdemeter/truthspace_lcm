"""
SSM First-Principles Complete: Random dirs + correct SV spectrum + pinv W₂
Tests if directions are truly free when SV spectrum is correct.
"""
import numpy as np, cv2, sys, glob, torch
import torch.nn.functional as F
from numpy.linalg import lstsq
from scipy.optimize import curve_fit

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895; SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]
bias_means = {0: -1.34, 1: -0.65, 2: -1.53, 3: -2.48}
bias_stds = {0: 0.93, 1: 0.59, 2: 0.64, 3: 0.39}

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))

def run_enc(v16, t, muts=None):
    if muts is None: muts = {}
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m, s = torch.tensor([.485,.456,.406]).view(1,3,1,1), torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (t - m) / s; feats = []
    with torch.no_grad():
        x = F.conv2d(x, v16._get_weight('encoder.arch.downsample_layers.0.0.weight'),
                     v16._get_weight('encoder.arch.downsample_layers.0.0.bias'), stride=4)
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, (96,), v16._get_weight('encoder.arch.downsample_layers.0.1.weight'),
                         v16._get_weight('encoder.arch.downsample_layers.0.1.bias'))
        x = x.permute(0,3,1,2)
        for si in range(4):
            d = dims[si]
            if si > 0:
                p = f'encoder.arch.downsample_layers.{si}'
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (dims[si-1],), v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'), v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,), v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                x = F.linear(x, muts.get(f'{p}.pwconv1.weight', v16._get_weight(f'{p}.pwconv1.weight')),
                             muts.get(f'{p}.pwconv1.bias', v16._get_weight(f'{p}.pwconv1.bias')))
                x = gate(x)
                x = F.linear(x, muts.get(f'{p}.pwconv2.weight', v16._get_weight(f'{p}.pwconv2.weight')),
                             muts.get(f'{p}.pwconv2.bias', v16._get_weight(f'{p}.pwconv2.bias')))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,), v16._get_weight(f'encoder.arch.norm{si}.weight'), v16._get_weight(f'encoder.arch.norm{si}.bias'))
            feats.append(xn.permute(0,3,1,2))
        o = v16._geometric_unet_block(feats[3], feats[2], 0)
        o = v16._geometric_unet_block(o, feats[1], 1)
        o = v16._geometric_unet_block(o, feats[0], 2)
        o = v16._geometric_last_shuf(o)
    return o.squeeze(0).detach().numpy()

print('Building color basis...')
ae, ag = [], []
for idx in range(50, 70):
    im = cv2.imread(all_imgs[idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ)); gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab); ab = lab[:,:,1:].astype(float) - 128.0
    if np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2).mean() < 2: continue
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    enc = run_enc(v16, t); flat = enc.reshape(256,-1).T
    s = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    ae.append(flat[s]); ys, xs = s//SZ, s%SZ
    ag.append(np.stack([ab[ys,xs,0], ab[ys,xs,1]], axis=1))
ae, ag = np.vstack(ae), np.vstack(ag)
em = ae.mean(0); C = (ae-em).T @ ag / len(ae)
Uc, Sc, _ = np.linalg.svd(C, full_matrices=False)
cd1, cd2 = Uc[:,0], Uc[:,1]
X2 = np.column_stack([(ae-em)@cd1, (ae-em)@cd2, np.ones(len(ae))])
Wa, *_ = lstsq(X2, ag[:,0], rcond=None); Wb, *_ = lstsq(X2, ag[:,1], rcond=None)

def evaluate(muts=None, n=15):
    gaps = []
    for idx in range(80, 300):
        if len(gaps) >= n: break
        im = cv2.imread(all_imgs[idx])
        if im is None: continue
        r = cv2.resize(im, (SZ, SZ)); gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab); ab = lab[:,:,1:].astype(float)-128.
        ez = np.sqrt(np.mean(ab**2))
        if ez < 2: continue
        t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
        enc = run_enc(v16, t, muts); flat = (enc.reshape(256,-1).T - em)
        f2 = np.column_stack([flat@cd1, flat@cd2, np.ones(SZ*SZ)])
        pred = np.stack([np.clip(f2@Wa,-50,50).reshape(SZ,SZ), np.clip(f2@Wb,-50,50).reshape(SZ,SZ)], axis=2)
        gaps.append((1 - np.sqrt(np.mean((pred-ab)**2))/ez)*100)
    return np.mean(gaps), np.std(gaps)

# Fit power law per stage
print('Fitting power law per stage...')
sp = {}
for si in range(4):
    pw1 = v16._get_weight(f'encoder.arch.stages.{si}.0.pwconv1.weight').numpy()
    _, S, _ = np.linalg.svd(pw1, full_matrices=False)
    popt, _ = curve_fit(lambda i, A, a: A*(i+1.)**(-a), np.arange(len(S), dtype=float), S, p0=[S[0],.5], maxfev=50000)
    sp[si] = {'A': popt[0], 'alpha': popt[1], 'S0': S[0]}
    print(f'  S{si}: A={popt[0]:.3f}, α={popt[1]:.4f}')

def make_muts(sv_mode, w2_mode, seed=42):
    np.random.seed(seed); muts = {}
    for si in range(4):
        d, de = dims[si], dims[si]*4
        for bi in range(depths[si]):
            pf = f'encoder.arch.stages.{si}.{bi}'
            pw1r = v16._get_weight(f'{pf}.pwconv1.weight').numpy()
            pw2r = v16._get_weight(f'{pf}.pwconv2.weight').numpy()
            b1r = v16._get_weight(f'{pf}.pwconv1.bias').numpy()
            
            if sv_mode == 'real':
                U1, S1, Vt1 = np.linalg.svd(pw1r, full_matrices=False)
                W1 = pw1r
            else:
                # Random orthogonal directions
                U1 = np.linalg.qr(np.random.randn(de, d))[0]  # [de, d]
                Vt1 = np.linalg.qr(np.random.randn(d, d))[0]  # [d, d]
                
                if sv_mode == 'fitted_power':
                    k = d
                    S1 = sp[si]['A'] * (np.arange(k, dtype=float) + 1.) ** (-sp[si]['alpha'])
                elif sv_mode == 'phi_flat':
                    k = d
                    S1 = pw1r[0] if hasattr(pw1r, '__len__') else 1.0
                    _, Sr, _ = np.linalg.svd(pw1r, full_matrices=False)
                    S1 = np.array([Sr[0] * PHI ** (-i * 0.5 / k) for i in range(k)])
                elif sv_mode == 'phi_steep':
                    _, Sr, _ = np.linalg.svd(pw1r, full_matrices=False)
                    k = d
                    beta = sp[si]['alpha'] * np.log(k) / np.log(PHI)
                    S1 = np.array([Sr[0] * PHI ** (-i * beta / k) for i in range(k)])
                
                W1 = (U1 * S1) @ Vt1
            
            if w2_mode == 'real':
                W2 = pw2r
            elif w2_mode == 'pinv':
                W2 = (Vt1.T * (1./(S1 + 1e-6))) @ U1.T
                W2 *= np.linalg.norm(pw2r) / np.linalg.norm(W2)
            elif w2_mode == 'transpose':
                W2 = W1.T
                W2 *= np.linalg.norm(pw2r) / np.linalg.norm(W2)
            
            b1 = np.random.randn(de) * bias_stds[si] + bias_means[si]
            
            muts[f'{pf}.pwconv1.weight'] = torch.from_numpy(W1).float()
            muts[f'{pf}.pwconv2.weight'] = torch.from_numpy(W2).float()
            muts[f'{pf}.pwconv1.bias'] = torch.from_numpy(b1).float()
    return muts

# ================================================================
# TESTS
# ================================================================
print('\n' + '='*70)
print('RESULTS')
print('='*70 + '\n')

base, bs = evaluate()
print(f'{"Full encoder (baseline)":<55} {base:+6.1f}% ± {bs:.1f}%\n')

tests = [
    ('Learned dirs + fitted power SVs + real W₂', 'real', 'real', True),
    ('Random dirs + fitted power SVs + pinv W₂', 'fitted_power', 'pinv', False),
    ('Random dirs + fitted power SVs + transpose W₂', 'fitted_power', 'transpose', False),
    ('Random dirs + φ-steep SVs + pinv W₂', 'phi_steep', 'pinv', False),
    ('Random dirs + φ-flat SVs + pinv W₂', 'phi_flat', 'pinv', False),
]

# Special case for "learned dirs + fitted power SVs"
def make_learned_dirs_fitted_sv():
    muts = {}
    for si in range(4):
        d = dims[si]
        for bi in range(depths[si]):
            pf = f'encoder.arch.stages.{si}.{bi}'
            pw1r = v16._get_weight(f'{pf}.pwconv1.weight').numpy()
            pw2r = v16._get_weight(f'{pf}.pwconv2.weight').numpy()
            b1r = v16._get_weight(f'{pf}.pwconv1.bias').numpy()
            U1, _, Vt1 = np.linalg.svd(pw1r, full_matrices=False)
            k = d
            S_new = sp[si]['A'] * (np.arange(k, dtype=float) + 1.) ** (-sp[si]['alpha'])
            muts[f'{pf}.pwconv1.weight'] = torch.from_numpy((U1 * S_new) @ Vt1).float()
            muts[f'{pf}.pwconv2.weight'] = torch.from_numpy(pw2r).float()
            muts[f'{pf}.pwconv1.bias'] = torch.from_numpy(b1r).float()
    return muts

mg, sg = evaluate(make_learned_dirs_fitted_sv())
print(f'{"Learned dirs + fitted power SVs + real W₂":<55} {mg:+6.1f}% ± {sg:.1f}%')

for name, sv, w2, _ in tests[1:]:
    mg, sg = evaluate(make_muts(sv, w2))
    print(f'{name:<55} {mg:+6.1f}% ± {sg:.1f}%')

# Multiple seeds for random dirs
print(f'\nRandom dirs stability (fitted_power + pinv, 5 seeds):')
for seed in [42, 123, 456, 789, 1234]:
    mg, sg = evaluate(make_muts('fitted_power', 'pinv', seed=seed), n=10)
    print(f'  seed={seed}: {mg:+6.1f}%')

print('\nDone!')
