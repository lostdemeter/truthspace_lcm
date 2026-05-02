"""
Phase 4C: Reality Check — Is RMSE 17.91 the constant prediction baseline?

Phase 4B found: ablating EVERYTHING in the encoder = no RMSE change.
This means either:
  1. RMSE 17.91 is the "predict neutral" baseline
  2. The decoder produces constant output from biases alone
  3. Something else entirely

This script checks:
  1. What does the model actually predict? (visualize predictions)
  2. What is the "predict neutral" RMSE?
  3. What is the "predict image mean" RMSE?
  4. Does the model prediction vary across pixels at all?
  5. Is the decoder actually using encoder features?
"""
import numpy as np
import cv2
import sys
import glob
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/thorin/truthspace-lcm')
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

SZ = 256
dims = [96, 192, 384, 768]; depths = [3, 3, 9, 3]

v16 = V16GeometricColorizer()
all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))


def run_full_model(v16, img_tensor):
    """Run full model, return color prediction [2, H, W]."""
    gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    m = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    s = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    x = (img_tensor - m) / s
    features = []
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
                x = F.layer_norm(x, (dims[si-1],),
                                 v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
                x = x.permute(0,3,1,2)
                x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
            for bi in range(depths[si]):
                p = f'encoder.arch.stages.{si}.{bi}'
                res = x
                x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                             v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
                x = x.permute(0,2,3,1)
                x = F.layer_norm(x, (d,),
                                 v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
                x = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                                  v16._get_weight(f'{p}.pwconv1.bias')))
                x = F.linear(x, v16._get_weight(f'{p}.pwconv2.weight'),
                             v16._get_weight(f'{p}.pwconv2.bias'))
                x = x.permute(0,3,1,2)
                x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
            xn = x.permute(0,2,3,1)
            xn = F.layer_norm(xn, (d,),
                              v16._get_weight(f'encoder.arch.norm{si}.weight'),
                              v16._get_weight(f'encoder.arch.norm{si}.bias'))
            features.append(xn.permute(0,3,1,2))

        out0 = v16._geometric_unet_block(features[3], features[2], 0)
        out1 = v16._geometric_unet_block(out0, features[1], 1)
        out2 = v16._geometric_unet_block(out1, features[0], 2)
        out3 = v16._geometric_last_shuf(out2)
    return out3.squeeze(0).detach().numpy()


# ================================================================
# STEP 1: What does the model actually predict?
# ================================================================
print('=' * 70)
print('STEP 1: WHAT DOES THE MODEL PREDICT?')
print('=' * 70)
print()

test_images = []
for img_idx in range(300, 400):
    im = cv2.imread(all_imgs[img_idx])
    if im is None: continue
    r = cv2.resize(im, (SZ, SZ))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    ab = lab[:,:,1:].astype(float) - 128.0
    sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
    if sat.mean() < 5: continue
    t = torch.from_numpy(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).transpose(2,0,1)).float().unsqueeze(0)/255.
    test_images.append({'tensor': t, 'ab': ab, 'idx': img_idx, 'sat': sat.mean()})
    if len(test_images) >= 15: break

print(f'{len(test_images)} test images')
print()

for i, ti in enumerate(test_images[:5]):
    pred = run_full_model(v16, ti['tensor'])
    pred_a, pred_b = pred[0], pred[1]
    gt_a, gt_b = ti['ab'][:,:,0], ti['ab'][:,:,1]

    # Model prediction statistics
    print(f'Image {i} (idx={ti["idx"]}, mean_sat={ti["sat"]:.1f}):')
    print(f'  Ground truth a*: mean={gt_a.mean():+.1f}  std={gt_a.std():.1f}  range=[{gt_a.min():+.1f}, {gt_a.max():+.1f}]')
    print(f'  Ground truth b*: mean={gt_b.mean():+.1f}  std={gt_b.std():.1f}  range=[{gt_b.min():+.1f}, {gt_b.max():+.1f}]')
    print(f'  Predicted   a*:  mean={pred_a.mean():+.1f}  std={pred_a.std():.1f}  range=[{pred_a.min():+.1f}, {pred_a.max():+.1f}]')
    print(f'  Predicted   b*:  mean={pred_b.mean():+.1f}  std={pred_b.std():.1f}  range=[{pred_b.min():+.1f}, {pred_b.max():+.1f}]')

    # Resize pred to match gt
    pred_a_r = cv2.resize(pred_a, (gt_a.shape[1], gt_a.shape[0]))
    pred_b_r = cv2.resize(pred_b, (gt_b.shape[1], gt_b.shape[0]))
    rmse = np.sqrt(np.mean((pred_a_r - gt_a)**2 + (pred_b_r - gt_b)**2))
    neutral_rmse = np.sqrt(np.mean(gt_a**2 + gt_b**2))
    print(f'  RMSE model: {rmse:.2f}   RMSE neutral: {neutral_rmse:.2f}   '
          f'Improvement: {(1 - rmse/neutral_rmse)*100:.1f}%')
    print()


# ================================================================
# STEP 2: SYSTEMATIC BASELINES
# ================================================================
print('=' * 70)
print('STEP 2: BASELINE COMPARISON')
print('=' * 70)
print()

model_rmses = []
neutral_rmses = []
mean_rmses = []

for ti in test_images:
    pred = run_full_model(v16, ti['tensor'])
    gt_a, gt_b = ti['ab'][:,:,0], ti['ab'][:,:,1]
    pred_a_r = cv2.resize(pred[0], (gt_a.shape[1], gt_a.shape[0]))
    pred_b_r = cv2.resize(pred[1], (gt_b.shape[1], gt_b.shape[0]))

    model_rmse = np.sqrt(np.mean((pred_a_r - gt_a)**2 + (pred_b_r - gt_b)**2))
    neutral_rmse = np.sqrt(np.mean(gt_a**2 + gt_b**2))
    mean_rmse = np.sqrt(np.mean((gt_a.mean() - gt_a)**2 + (gt_b.mean() - gt_b)**2))

    model_rmses.append(model_rmse)
    neutral_rmses.append(neutral_rmse)
    mean_rmses.append(mean_rmse)

model_rmses = np.array(model_rmses)
neutral_rmses = np.array(neutral_rmses)
mean_rmses = np.array(mean_rmses)

print(f'{"Baseline":<30} {"Mean RMSE":<12} {"Std":<10}')
print('-' * 52)
print(f'{"Predict neutral (0,0)":<30} {neutral_rmses.mean():<12.2f} {neutral_rmses.std():<10.2f}')
print(f'{"Predict image mean":<30} {mean_rmses.mean():<12.2f} {mean_rmses.std():<10.2f}')
print(f'{"V16 model":<30} {model_rmses.mean():<12.2f} {model_rmses.std():<10.2f}')

# Does model beat neutral?
beats = (model_rmses < neutral_rmses).sum()
print(f'\n  Model beats neutral: {beats}/{len(test_images)} images')
improvement = ((neutral_rmses - model_rmses) / neutral_rmses * 100)
print(f'  Mean improvement over neutral: {improvement.mean():.1f}% ± {improvement.std():.1f}%')


# ================================================================
# STEP 3: Does prediction vary spatially?
# ================================================================
print()
print('=' * 70)
print('STEP 3: SPATIAL VARIATION IN PREDICTIONS')
print('=' * 70)
print()

for i, ti in enumerate(test_images[:5]):
    pred = run_full_model(v16, ti['tensor'])
    pred_a, pred_b = pred[0], pred[1]

    # Coefficient of variation
    a_cv = pred_a.std() / (np.abs(pred_a.mean()) + 1e-8)
    b_cv = pred_b.std() / (np.abs(pred_b.mean()) + 1e-8)

    # Spatial correlation with ground truth
    gt_a = cv2.resize(ti['ab'][:,:,0], (pred_a.shape[1], pred_a.shape[0]))
    gt_b = cv2.resize(ti['ab'][:,:,1], (pred_b.shape[1], pred_b.shape[0]))

    corr_a = np.corrcoef(pred_a.flatten(), gt_a.flatten())[0,1]
    corr_b = np.corrcoef(pred_b.flatten(), gt_b.flatten())[0,1]

    print(f'Image {i}:')
    print(f'  pred_a: mean={pred_a.mean():+.1f} std={pred_a.std():.2f} cv={a_cv:.3f}')
    print(f'  pred_b: mean={pred_b.mean():+.1f} std={pred_b.std():.2f} cv={b_cv:.3f}')
    print(f'  Spatial correlation: a*={corr_a:.3f}  b*={corr_b:.3f}')
    print()


# ================================================================
# STEP 4: Feature magnitude through the U-Net decoder
# ================================================================
print()
print('=' * 70)
print('STEP 4: DECODER SIGNAL TRACKING')
print('=' * 70)
print()

ti = test_images[0]
gate = lambda x: x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
m_t = torch.tensor([.485,.456,.406]).view(1,3,1,1)
s_t = torch.tensor([.229,.224,.225]).view(1,3,1,1)
x = (ti['tensor'] - m_t) / s_t
features = []
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
            x = F.layer_norm(x, (dims[si-1],),
                             v16._get_weight(f'{p}.0.weight'), v16._get_weight(f'{p}.0.bias'))
            x = x.permute(0,3,1,2)
            x = F.conv2d(x, v16._get_weight(f'{p}.1.weight'), v16._get_weight(f'{p}.1.bias'), stride=2)
        for bi in range(depths[si]):
            p = f'encoder.arch.stages.{si}.{bi}'
            res = x
            x = F.conv2d(x, v16._get_weight(f'{p}.dwconv.weight'),
                         v16._get_weight(f'{p}.dwconv.bias'), padding=3, groups=d)
            x = x.permute(0,2,3,1)
            x = F.layer_norm(x, (d,),
                             v16._get_weight(f'{p}.norm.weight'), v16._get_weight(f'{p}.norm.bias'))
            x = gate(F.linear(x, v16._get_weight(f'{p}.pwconv1.weight'),
                              v16._get_weight(f'{p}.pwconv1.bias')))
            x = F.linear(x, v16._get_weight(f'{p}.pwconv2.weight'),
                         v16._get_weight(f'{p}.pwconv2.bias'))
            x = x.permute(0,3,1,2)
            x = res + v16._get_weight(f'{p}.gamma').view(1,-1,1,1) * x
        xn = x.permute(0,2,3,1)
        xn = F.layer_norm(xn, (d,),
                          v16._get_weight(f'encoder.arch.norm{si}.weight'),
                          v16._get_weight(f'encoder.arch.norm{si}.bias'))
        features.append(xn.permute(0,3,1,2))

    print('Encoder feature statistics per stage:')
    for si in range(4):
        f_np = features[si].numpy()
        print(f'  Stage {si}: shape={features[si].shape}  '
              f'mean={f_np.mean():.3f}  std={f_np.std():.3f}  '
              f'|norm|={np.linalg.norm(f_np):.1f}  '
              f'sparsity={(f_np == 0).mean():.1%}')

    # Track through decoder
    print()
    print('Decoder signal tracking:')
    out0 = v16._geometric_unet_block(features[3], features[2], 0)
    print(f'  UNet block 0 (s3+s2): shape={out0.shape}  '
          f'mean={out0.mean():.3f}  std={out0.std():.3f}  |norm|={torch.norm(out0):.1f}')

    out1 = v16._geometric_unet_block(out0, features[1], 1)
    print(f'  UNet block 1 (+s1):   shape={out1.shape}  '
          f'mean={out1.mean():.3f}  std={out1.std():.3f}  |norm|={torch.norm(out1):.1f}')

    out2 = v16._geometric_unet_block(out1, features[0], 2)
    print(f'  UNet block 2 (+s0):   shape={out2.shape}  '
          f'mean={out2.mean():.3f}  std={out2.std():.3f}  |norm|={torch.norm(out2):.1f}')

    out3 = v16._geometric_last_shuf(out2)
    print(f'  Final output:         shape={out3.shape}  '
          f'mean={out3.mean():.3f}  std={out3.std():.3f}  |norm|={torch.norm(out3):.1f}')


# ================================================================
# STEP 5: Compare with original model (if available)
# ================================================================
print()
print('=' * 70)
print('STEP 5: WHAT DOES V16 _get_weight DO? — Verify weights are real')
print('=' * 70)
print()

# Check if _get_weight returns phi-encoded or original weights
sample_w = v16._get_weight('encoder.arch.stages.1.2.pwconv1.weight').detach().numpy()
print(f'Sample weight (stages.1.2.pwconv1.weight):')
print(f'  Shape: {sample_w.shape}')
print(f'  Mean: {sample_w.mean():.6f}  Std: {sample_w.std():.6f}')
print(f'  Range: [{sample_w.min():.6f}, {sample_w.max():.6f}]')
print(f'  Sparsity: {(sample_w == 0).mean():.1%}')
print(f'  Near-zero (<1e-4): {(np.abs(sample_w) < 1e-4).mean():.1%}')

# Check gamma values
for si in range(4):
    for bi in range(depths[si]):
        g = v16._get_weight(f'encoder.arch.stages.{si}.{bi}.gamma').detach().numpy()
        print(f'  γ stage {si}.{bi}: mean={np.abs(g).mean():.4f}  max={np.abs(g).max():.4f}  '
              f'near_zero={(np.abs(g) < 0.01).mean():.1%}')


print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print()
print('The critical question: does V16 model predict ANYTHING beyond neutral?')
print(f'  Model RMSE:   {model_rmses.mean():.2f}')
print(f'  Neutral RMSE: {neutral_rmses.mean():.2f}')
print(f'  If model ≈ neutral: the φ-encoding preserves geometric structure')
print(f'  but destroys functional color prediction.')
