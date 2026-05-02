"""
V16 vs Original DDColor — Full Pipeline Comparison

Our previous experiments had a critical bug: they treated the 256-channel
UNet feature map (out3) as if channels 0-1 were color predictions. The actual
color prediction requires the FULL pipeline:
  1. Encoder (ConvNeXt) → 4 feature maps
  2. UNet decoder → 3 intermediate + 1 shuffled
  3. Color decoder (9 transformer layers) → color queries
  4. Refine net → final 3-channel output (a*, b*, ?)

This script:
  1. Loads original DDColor and V16
  2. Compares features at every stage
  3. Compares actual end-to-end color predictions
  4. Identifies WHERE the divergence happens
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer


# ================================================================
# STEP 0: LOAD BOTH MODELS
# ================================================================
print('=' * 70)
print('STEP 0: LOAD MODELS')
print('=' * 70)
print()

# Load V16 (extracted weights)
v16 = V16GeometricColorizer()

# Load original DDColor
from ddcolor import DDColor
from huggingface_hub import PyTorchModelHubMixin

class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)

ddcolor = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
ddcolor.eval()
print('  Original DDColor loaded')

all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
SZ = 256


# ================================================================
# STEP 1: COMPARE ENCODER FEATURES STAGE-BY-STAGE
# ================================================================
print()
print('=' * 70)
print('STEP 1: ENCODER FEATURE COMPARISON (V16 vs Original)')
print('=' * 70)
print()

# Prepare test image
im = cv2.imread(all_imgs[50])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

# V16 encoder
with torch.no_grad():
    v16_features = v16._geometric_encoder(t)

# Original DDColor encoder — uses hooks
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x_norm = (t - mean) / std

with torch.no_grad():
    ddcolor.encoder(x_norm)

orig_features = [hook.feature for hook in ddcolor.encoder.hooks]

print(f'{"Stage":<8} {"V16 shape":<22} {"Orig shape":<22} {"Cosine":<10} {"RMSE":<10} {"Max diff":<10}')
print('-' * 82)
for si in range(4):
    v = v16_features[si].detach().numpy().flatten()
    o = orig_features[si].detach().numpy().flatten()
    cos = np.dot(v, o) / (np.linalg.norm(v) * np.linalg.norm(o) + 1e-8)
    rmse = np.sqrt(np.mean((v - o)**2))
    maxd = np.max(np.abs(v - o))
    print(f'  {si:<6} {str(v16_features[si].shape):<22} {str(orig_features[si].shape):<22} '
          f'{cos:<10.6f} {rmse:<10.6f} {maxd:<10.6f}')


# ================================================================
# STEP 2: COMPARE UNET DECODER OUTPUTS
# ================================================================
print()
print('=' * 70)
print('STEP 2: UNET DECODER COMPARISON')
print('=' * 70)
print()

# V16 decoder
with torch.no_grad():
    v16_out0 = v16._geometric_unet_block(v16_features[3], v16_features[2], 0)
    v16_out1 = v16._geometric_unet_block(v16_out0, v16_features[1], 1)
    v16_out2 = v16._geometric_unet_block(v16_out1, v16_features[0], 2)
    v16_out3 = v16._geometric_last_shuf(v16_out2)

# Original decoder — need to run through its layers
with torch.no_grad():
    encode_feat = ddcolor.encoder.hooks[-1].feature
    orig_out0 = ddcolor.decoder.layers[0](encode_feat)
    orig_out1 = ddcolor.decoder.layers[1](orig_out0)
    orig_out2 = ddcolor.decoder.layers[2](orig_out1)
    orig_out3 = ddcolor.decoder.last_shuf(orig_out2)

print(f'{"Output":<10} {"V16 shape":<25} {"Orig shape":<25} {"Cosine":<10} {"RMSE":<10}')
print('-' * 80)
for name, v16_o, orig_o in [('out0', v16_out0, orig_out0),
                              ('out1', v16_out1, orig_out1),
                              ('out2', v16_out2, orig_out2),
                              ('out3', v16_out3, orig_out3)]:
    v = v16_o.detach().numpy().flatten()
    o = orig_o.detach().numpy().flatten()
    cos = np.dot(v, o) / (np.linalg.norm(v) * np.linalg.norm(o) + 1e-8)
    rmse = np.sqrt(np.mean((v - o)**2))
    print(f'  {name:<8} {str(v16_o.shape):<25} {str(orig_o.shape):<25} {cos:<10.6f} {rmse:<10.6f}')


# ================================================================
# STEP 3: FULL FORWARD PASS COMPARISON
# ================================================================
print()
print('=' * 70)
print('STEP 3: FULL FORWARD PASS — END-TO-END COLOR PREDICTION')
print('=' * 70)
print()

with torch.no_grad():
    v16_color = v16.forward(t)
    orig_color = ddcolor(t)

v = v16_color.detach().numpy()
o = orig_color.detach().numpy()

print(f'V16 output:  shape={v16_color.shape}  mean={v.mean():.4f}  std={v.std():.4f}  '
      f'range=[{v.min():.4f}, {v.max():.4f}]')
print(f'Orig output: shape={orig_color.shape}  mean={o.mean():.4f}  std={o.std():.4f}  '
      f'range=[{o.min():.4f}, {o.max():.4f}]')

cos = np.dot(v.flatten(), o.flatten()) / (np.linalg.norm(v.flatten()) * np.linalg.norm(o.flatten()) + 1e-8)
rmse = np.sqrt(np.mean((v.flatten() - o.flatten())**2))
print(f'\nCosine similarity: {cos:.6f}')
print(f'RMSE: {rmse:.6f}')

# Per-channel analysis
for ch in range(min(v16_color.shape[1], 3)):
    vc = v[0, ch]
    oc = o[0, ch]
    ch_cos = np.corrcoef(vc.flatten(), oc.flatten())[0, 1]
    ch_rmse = np.sqrt(np.mean((vc - oc)**2))
    print(f'  Channel {ch}: V16 mean={vc.mean():+.3f} std={vc.std():.3f}  '
          f'Orig mean={oc.mean():+.3f} std={oc.std():.3f}  '
          f'corr={ch_cos:.4f}  rmse={ch_rmse:.4f}')


# ================================================================
# STEP 4: ACTUAL COLOR QUALITY — Compare to ground truth
# ================================================================
print()
print('=' * 70)
print('STEP 4: COLOR QUALITY vs GROUND TRUTH')
print('=' * 70)
print()

lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
gt_ab = lab[:, :, 1:].astype(float) - 128.0

# V16 color output
v16_ab = v16_color[0, :2].permute(1, 2, 0).detach().numpy()  # [H, W, 2]
v16_ab_r = cv2.resize(v16_ab, (gt_ab.shape[1], gt_ab.shape[0]))

# Original DDColor output
orig_ab = orig_color[0, :2].permute(1, 2, 0).detach().numpy()
orig_ab_r = cv2.resize(orig_ab, (gt_ab.shape[1], gt_ab.shape[0]))

v16_rmse = np.sqrt(np.mean((v16_ab_r - gt_ab)**2))
orig_rmse = np.sqrt(np.mean((orig_ab_r - gt_ab)**2))
neutral_rmse = np.sqrt(np.mean(gt_ab**2))

print(f'  {"Method":<30} {"RMSE":<10}')
print(f'  {"-"*40}')
print(f'  {"Predict neutral (0,0)":<30} {neutral_rmse:<10.2f}')
print(f'  {"V16 geometric colorizer":<30} {v16_rmse:<10.2f}')
print(f'  {"Original DDColor":<30} {orig_rmse:<10.2f}')
print()
print(f'  V16 ab* prediction:  mean_a={v16_ab.mean(axis=(0,1))[0]:+.2f}  mean_b={v16_ab.mean(axis=(0,1))[1]:+.2f}  '
      f'std_a={v16_ab[:,:,0].std():.2f}  std_b={v16_ab[:,:,1].std():.2f}')
print(f'  Orig ab* prediction: mean_a={orig_ab.mean(axis=(0,1))[0]:+.2f}  mean_b={orig_ab.mean(axis=(0,1))[1]:+.2f}  '
      f'std_a={orig_ab[:,:,0].std():.2f}  std_b={orig_ab[:,:,1].std():.2f}')


# ================================================================
# STEP 5: MULTI-IMAGE COMPARISON
# ================================================================
print()
print('=' * 70)
print('STEP 5: MULTI-IMAGE COMPARISON')
print('=' * 70)
print()

v16_rmses = []
orig_rmses = []
neutral_rmses = []
cosines = []

for img_idx in range(300, 320):
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    gt_ab = lab[:, :, 1:].astype(float) - 128.0

    with torch.no_grad():
        v16_out = v16.forward(t)
        orig_out = ddcolor(t)

    # Feature cosine at output level
    v_f = v16_out.numpy().flatten()
    o_f = orig_out.numpy().flatten()
    cos = np.dot(v_f, o_f) / (np.linalg.norm(v_f) * np.linalg.norm(o_f) + 1e-8)
    cosines.append(cos)

    # Color RMSE
    v16_ab = v16_out[0, :2].permute(1, 2, 0).numpy()
    v16_ab_r = cv2.resize(v16_ab, (gt_ab.shape[1], gt_ab.shape[0]))
    orig_ab = orig_out[0, :2].permute(1, 2, 0).numpy()
    orig_ab_r = cv2.resize(orig_ab, (gt_ab.shape[1], gt_ab.shape[0]))

    v16_rmses.append(np.sqrt(np.mean((v16_ab_r - gt_ab)**2)))
    orig_rmses.append(np.sqrt(np.mean((orig_ab_r - gt_ab)**2)))
    neutral_rmses.append(np.sqrt(np.mean(gt_ab**2)))

v16_rmses = np.array(v16_rmses)
orig_rmses = np.array(orig_rmses)
neutral_rmses = np.array(neutral_rmses)
cosines = np.array(cosines)

print(f'  {"Metric":<35} {"V16":<15} {"Original":<15}')
print(f'  {"-"*65}')
print(f'  {"Mean color RMSE":<35} {v16_rmses.mean():<15.2f} {orig_rmses.mean():<15.2f}')
print(f'  {"Mean neutral RMSE":<35} {neutral_rmses.mean():<15.2f} {neutral_rmses.mean():<15.2f}')
print(f'  {"Improvement over neutral":<35} {((neutral_rmses - v16_rmses)/neutral_rmses).mean()*100:<14.1f}% {((neutral_rmses - orig_rmses)/neutral_rmses).mean()*100:<14.1f}%')
print(f'  {"V16↔Orig output cosine":<35} {cosines.mean():<15.6f}')
print(f'  {"Images tested":<35} {len(v16_rmses)}')


# ================================================================
# STEP 6: WHERE DOES V16 DIVERGE FROM ORIGINAL?
# ================================================================
print()
print('=' * 70)
print('STEP 6: DIVERGENCE LOCALIZATION')
print('=' * 70)
print()

# Test one image in detail through the pipeline
im = cv2.imread(all_imgs[50])
r = cv2.resize(im, (SZ, SZ))
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
t = torch.from_numpy(gray_3ch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

# Run original
with torch.no_grad():
    x_norm = (t - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    ddcolor.encoder(x_norm)

# Run V16
with torch.no_grad():
    v16_features = v16._geometric_encoder(t)

# Compare each weight used in the encoder
print('Weight comparison (V16 extracted vs original):')
dims = [96, 192, 384, 768]
depths = [3, 3, 9, 3]

divergence_found = False
for si in range(4):
    for bi in range(depths[si]):
        prefix = f'encoder.arch.stages.{si}.{bi}'
        keys = ['dwconv.weight', 'dwconv.bias', 'norm.weight', 'norm.bias',
                'pwconv1.weight', 'pwconv1.bias', 'pwconv2.weight', 'pwconv2.bias', 'gamma']
        for key in keys:
            full_key = f'{prefix}.{key}'
            v16_w = v16._get_weight(full_key)
            # Get from original model
            parts = full_key.split('.')
            obj = ddcolor
            for p in parts:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
            if obj is None or not isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
                continue
            orig_w = obj.data if isinstance(obj, torch.nn.Parameter) else obj
            if v16_w is not None and orig_w is not None:
                diff = torch.abs(v16_w - orig_w).max().item()
                if diff > 1e-4:
                    if not divergence_found:
                        print(f'  {"Weight key":<55} {"Max diff":<12} {"V16 std":<10} {"Orig std":<10}')
                        print(f'  {"-"*87}')
                        divergence_found = True
                    print(f'  {full_key:<55} {diff:<12.6f} {v16_w.std().item():<10.6f} {orig_w.std().item():<10.6f}')

if not divergence_found:
    print('  ALL encoder weights match within 1e-4!')

# Also check decoder weights
print()
print('Decoder weight comparison:')
decoder_divergence = False
for name in sorted(v16.weights.files):
    if not name.startswith('decoder.') and not name.startswith('refine_net.'):
        continue
    v16_w = v16._get_weight(name)
    if v16_w is None:
        continue
    # Navigate original model
    parts = name.split('.')
    obj = ddcolor
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p, None)
            if obj is None:
                break
    if obj is None:
        continue
    if isinstance(obj, torch.nn.Parameter):
        orig_w = obj.data
    elif isinstance(obj, torch.Tensor):
        orig_w = obj
    else:
        continue
    diff = torch.abs(v16_w - orig_w).max().item()
    if diff > 1e-4:
        if not decoder_divergence:
            print(f'  {"Weight key":<55} {"Max diff":<12}')
            print(f'  {"-"*67}')
            decoder_divergence = True
        print(f'  {name:<55} {diff:<12.6f}')

if not decoder_divergence:
    print('  ALL decoder weights match within 1e-4!')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
