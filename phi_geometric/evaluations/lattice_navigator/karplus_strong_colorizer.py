"""
Karplus-Strong Colorizer: Content from Resonance

Analogy:
  KS Audio: noise buffer → average adjacent → feedback → standing wave (pitch emerges)
  KS Color: scaffolding → edge-guided diffusion → histogram constraint → content emerges

The key insight: we don't need to KNOW what an object is.
We need constraints that, applied iteratively, cause the correct color
to RESONATE within edge-bounded regions.

Constraints (the "averaging" operations):
1. Edge-guided diffusion: color flows freely within regions, stops at edges
2. Region coherence: connected regions converge to a single dominant color
3. Histogram matching: push color distribution toward natural image stats
4. Scale consistency: color at coarse scale should agree with fine scale
5. Saturation maintenance: prevent convergence to gray

The "buffer" is initialized from our geometric scaffolding (28 params)
plus manufactured noise (the "pluck" of the string).
"""
import numpy as np
import cv2
import sys
import glob
import os
import time

sys.path.insert(0, '/home/thorin/truthspace-lcm')
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_geometric/evaluations/lattice_navigator')

from color_lattice import LatticeNavigator
import torch
from phi_geometric.models.geometric_colorizer_v16_convnext import V16GeometricColorizer

PHI = 1.618033988749895

def ab_to_bgr(ab, L):
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    lab[:,:,0] = L
    lab[:,:,1] = np.clip(ab[:,:,0] + 128, 0, 255).astype(np.uint8)
    lab[:,:,2] = np.clip(ab[:,:,1] + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def get_sat(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,1].mean()


class KarplusStrongColorizer:
    """
    Color synthesis via iterative constraint satisfaction.
    
    Like KS audio synthesis:
    - Buffer = color field (a, b channels)
    - Pluck = initial geometric scaffolding + noise
    - Feedback rule = edge-guided diffusion + constraints
    - Resonance = content-appropriate color emerges
    """
    
    def __init__(self):
        # Load geometric correction weights if available
        weights_path = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/derived_modes/geometric_correction_weights.npz'
        if os.path.exists(weights_path):
            data = np.load(weights_path, allow_pickle=True)
            self.W_a = data['W_a']
            self.W_b = data['W_b']
            self.feat_names = list(data['feat_names'])
            print(f'  Loaded geometric weights: {len(self.W_a)} params per channel')
        else:
            self.W_a = None
            self.W_b = None
            print('  No geometric weights found, using pure noise initialization')
        
        # Natural image color statistics (from COCO-like images)
        # These are the "resonant frequencies" of the color space
        self.natural_ab_mean = np.array([2.0, -5.0])  # slight warm bias
        self.natural_ab_std = np.array([12.0, 18.0])   # a has less range than b
    
    def extract_features(self, gray):
        """Extract geometric features from grayscale image."""
        h, w = gray.shape
        brightness = gray.astype(float) / 255.0
        yy = np.tile(np.arange(h).reshape(-1, 1) / h, (1, w))
        xx = np.tile(np.arange(w).reshape(1, -1) / w, (h, 1))
        
        gabor_e = np.zeros((h, w))
        for theta_idx in range(4):
            theta = theta_idx * np.pi / 4
            kernel = cv2.getGaborKernel((11, 11), 3.0, theta, 0.1, 0.5, 0)
            resp = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_e += resp**2
        gabor_e = np.sqrt(gabor_e)
        gabor_e = gabor_e / (gabor_e.max() + 1e-8)
        
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sx**2 + sy**2)
        edges_smooth = cv2.GaussianBlur(edges, (15, 15), 0)
        edges_smooth = edges_smooth / (edges_smooth.max() + 1e-8)
        
        blur = cv2.GaussianBlur(gray.astype(float), (15, 15), 0)
        local_var = cv2.GaussianBlur(gray.astype(float)**2, (15, 15), 0) - blur**2
        local_contrast = np.sqrt(np.maximum(local_var, 0)) / 128.0
        
        smoothness = 1.0 - edges_smooth
        bright_grad = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=5)
        bright_grad = bright_grad / (np.abs(bright_grad).max() + 1e-8)
        
        return {
            'brightness': brightness,
            'y_pos': yy,
            'x_pos': xx,
            'texture': gabor_e,
            'edges': edges_smooth,
            'contrast': local_contrast,
            'smoothness': smoothness,
            'bright_grad': bright_grad,
            'edge_map': edges,  # raw edges for diffusion guidance
        }
    
    def compute_scaffolding(self, feats):
        """Compute initial color field from geometric weights (the 28-param model)."""
        h = feats['brightness'].shape[0]
        w = feats['brightness'].shape[1]
        
        if self.W_a is not None:
            feat_names = ['brightness', 'y_pos', 'x_pos', 'texture', 'edges',
                         'contrast', 'smoothness', 'bright_grad']
            feat_stack = np.stack([feats[fn].flatten() for fn in feat_names], axis=1)
            
            flat_interact = np.column_stack([
                feat_stack,
                feat_stack[:, 0] * feat_stack[:, 1],  # bright*ypos
                feat_stack[:, 0] * feat_stack[:, 3],  # bright*tex
                feat_stack[:, 1] * feat_stack[:, 3],  # ypos*tex
                feat_stack[:, 0]**2,                   # bright²
                feat_stack[:, 1]**2,                   # ypos²
            ])
            flat_bias = np.column_stack([flat_interact, np.ones(flat_interact.shape[0])])
            
            pred_a = (flat_bias @ self.W_a).reshape(h, w)
            pred_b = (flat_bias @ self.W_b).reshape(h, w)
        else:
            pred_a = np.zeros((h, w))
            pred_b = np.zeros((h, w))
        
        return np.stack([pred_a, pred_b], axis=-1)
    
    def edge_guided_diffusion(self, ab, edge_map, strength=0.3, iterations=1):
        """
        Diffuse color within edge-bounded regions.
        This is the "averaging adjacent samples" of KS.
        
        Color flows freely in smooth regions but stops at edges.
        """
        h, w = ab.shape[:2]
        # Normalize edge map to [0, 1] for use as diffusion barrier
        edge_norm = edge_map / (edge_map.max() + 1e-8)
        # Diffusion coefficient: high in smooth areas, low at edges
        diff_coeff = np.exp(-edge_norm * 5.0)  # sharp falloff at edges
        
        result = ab.copy()
        for _ in range(iterations):
            for ch in range(2):
                # Compute Laplacian (neighbor average - center)
                padded = np.pad(result[:,:,ch], 1, mode='reflect')
                laplacian = (
                    padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:]
                ) / 4.0 - result[:,:,ch]
                
                # Apply diffusion weighted by edge barrier
                result[:,:,ch] += strength * diff_coeff * laplacian
        
        return result
    
    def region_coherence(self, ab, gray, n_segments=50):
        """
        Push color toward dominant value within each superpixel-like region.
        This creates "standing waves" — stable color within object boundaries.
        """
        h, w = ab.shape[:2]
        
        # Simple region segmentation via quantized brightness + position
        brightness_q = (gray.astype(float) / 255.0 * 8).astype(int)
        yy_q = (np.arange(h).reshape(-1, 1) * 6 // h) * np.ones((1, w), dtype=int)
        xx_q = np.ones((h, 1), dtype=int) * (np.arange(w).reshape(1, -1) * 6 // w)
        
        # Region ID from quantized features
        region_id = brightness_q * 100 + yy_q * 10 + xx_q
        
        # For each region, compute median color and push toward it
        result = ab.copy()
        unique_regions = np.unique(region_id)
        
        for rid in unique_regions:
            mask = region_id == rid
            if mask.sum() < 4:
                continue
            
            for ch in range(2):
                region_vals = ab[:,:,ch][mask]
                median_val = np.median(region_vals)
                # Push toward median (feedback)
                result[:,:,ch][mask] = 0.7 * result[:,:,ch][mask] + 0.3 * median_val
        
        return result
    
    def histogram_constraint(self, ab, target_mean, target_std, strength=0.2):
        """
        Push color distribution toward natural image statistics.
        This is the "resonant frequency" — the distribution the system wants to settle into.
        """
        result = ab.copy()
        for ch in range(2):
            current_mean = result[:,:,ch].mean()
            current_std = result[:,:,ch].std() + 1e-8
            
            # Normalize then rescale to target distribution
            normalized = (result[:,:,ch] - current_mean) / current_std
            target = normalized * target_std[ch] + target_mean[ch]
            
            # Blend (don't snap — gradual convergence)
            result[:,:,ch] = (1 - strength) * result[:,:,ch] + strength * target
        
        return result
    
    def scale_consistency(self, ab, gray):
        """
        Ensure color at coarse scale agrees with fine scale.
        Downsample, apply scaffolding, upsample, blend.
        This creates cross-scale resonance.
        """
        h, w = ab.shape[:2]
        
        # Downsample to half
        small_gray = cv2.resize(gray, (w//2, h//2))
        small_ab = cv2.resize(ab.astype(np.float32), (w//2, h//2))
        
        # Smooth at coarse scale (stronger diffusion)
        small_smooth = cv2.GaussianBlur(small_ab, (5, 5), 0)
        
        # Upsample back
        coarse_ab = cv2.resize(small_smooth, (w, h))
        
        # Blend: fine-scale detail + coarse-scale context
        result = 0.8 * ab + 0.2 * coarse_ab
        
        return result
    
    def saturation_boost(self, ab, gray, min_sat=5.0):
        """
        Prevent convergence to gray. Push chromatic regions to maintain saturation.
        Like adding energy back into the KS delay line to prevent decay.
        """
        sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2)
        low_sat = sat < min_sat
        
        # For low-saturation pixels, amplify existing color direction
        result = ab.copy()
        boost_factor = 1.5
        for ch in range(2):
            result[:,:,ch] = np.where(low_sat, ab[:,:,ch] * boost_factor, ab[:,:,ch])
        
        return result
    
    def manufacture_noise(self, h, w, gray, feats, noise_scale=8.0):
        """
        The "pluck" — manufactured error to initialize the buffer.
        Not random noise, but structured noise based on image features.
        
        Like KS where the noise is the initial energy that the feedback
        loop shapes into a standing wave.
        """
        # Seed from brightness-dependent color tendency
        brightness = feats['brightness']
        y_pos = feats['y_pos']
        texture = feats['texture']
        
        # Base tendencies (physical rules):
        # - Top of image: cool (sky)
        # - Bottom of image: warm (ground)
        # - Textured: green/earth
        # - Smooth + bright: blue (sky) or warm (highlights)
        # - Dark: slightly cool
        
        noise_a = np.zeros((h, w))
        noise_b = np.zeros((h, w))
        
        # Vertical gradient: top=cool, bottom=warm
        noise_b += (y_pos - 0.3) * noise_scale * 2  # b: negative=blue at top, positive=yellow at bottom
        noise_a += (y_pos - 0.5) * noise_scale * 0.5  # a: slight green at top, red at bottom
        
        # Texture → green/earth  
        noise_a += -texture * noise_scale * 1.5  # textured → green (negative a)
        noise_b += texture * noise_scale * 1.0   # textured → yellow (positive b)
        
        # Brightness modulation
        noise_b += (brightness - 0.5) * noise_scale * 1.5  # bright → warm
        
        # Add actual random noise for the "pluck" energy
        rng = np.random.RandomState(42)
        noise_a += rng.randn(h, w) * noise_scale * 0.5
        noise_b += rng.randn(h, w) * noise_scale * 0.5
        
        return np.stack([noise_a, noise_b], axis=-1)
    
    def colorize(self, gray, lattice_ab=None, n_iterations=30, verbose=True):
        """
        Karplus-Strong color synthesis.
        
        1. Initialize buffer with scaffolding + noise (the "pluck")
        2. Iterate: apply constraints (the "feedback loop")
        3. Content-appropriate color emerges from resonance
        """
        h, w = gray.shape
        feats = self.extract_features(gray)
        
        # === INITIALIZE: The "pluck" ===
        scaffolding = self.compute_scaffolding(feats)
        noise = self.manufacture_noise(h, w, gray, feats)
        
        if lattice_ab is not None:
            # Start from lattice + geometric correction + noise
            ab = lattice_ab + scaffolding + noise * 0.3
        else:
            # Start from scaffolding + noise alone
            ab = scaffolding + noise
        
        edge_map = feats['edge_map']
        
        if verbose:
            sat0 = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2).mean()
            print(f'  Init: mean_sat={sat0:.1f}')
        
        # === ITERATE: The "feedback loop" ===
        history = [ab.copy()]
        
        for it in range(n_iterations):
            # 1. Edge-guided diffusion (average within regions)
            ab = self.edge_guided_diffusion(ab, edge_map, strength=0.3, iterations=2)
            
            # 2. Region coherence (push toward regional consensus)
            if it % 3 == 0:  # every 3rd iteration
                ab = self.region_coherence(ab, gray)
            
            # 3. Scale consistency (cross-scale agreement)
            if it % 5 == 0:  # every 5th iteration
                ab = self.scale_consistency(ab, gray)
            
            # 4. Histogram constraint (resonate toward natural distribution)
            hist_strength = 0.1 * (1.0 - it / n_iterations)  # decay over time
            ab = self.histogram_constraint(ab, self.natural_ab_mean, 
                                          self.natural_ab_std, strength=hist_strength)
            
            # 5. Saturation maintenance (prevent decay to gray)
            ab = self.saturation_boost(ab, gray, min_sat=3.0)
            
            history.append(ab.copy())
            
            if verbose and it % 5 == 0:
                sat = np.sqrt(ab[:,:,0]**2 + ab[:,:,1]**2).mean()
                print(f'  Iter {it+1}/{n_iterations}: mean_sat={sat:.1f}')
        
        # Final smoothing
        for ch in range(2):
            ab[:,:,ch] = cv2.bilateralFilter(ab[:,:,ch].astype(np.float32), 7, 40, 40)
        
        return ab, history


# ================================================================
# MAIN: Test the Karplus-Strong colorizer
# ================================================================
if __name__ == '__main__':
    print('=== KARPLUS-STRONG COLORIZER ===')
    print()
    
    # Initialize lattice (for comparison and optional seeding)
    image_paths = glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg')
    nav = LatticeNavigator()
    nav.initialize(image_paths)
    nav.navigate(max_generations=5, min_confidence=0.05, min_n_smooth=0.25, verbose=False)
    
    # DDColor for comparison
    v16 = V16GeometricColorizer()
    
    # KS colorizer
    print('\nInitializing Karplus-Strong colorizer...')
    ks = KarplusStrongColorizer()
    
    SZ = 128
    all_imgs = sorted(glob.glob('/home/thorin/truthspace-lcm/experiments/unified_assembly/val2017/*.jpg'))
    test_indices = [50, 52, 54, 56]
    test_paths = [all_imgs[i] for i in test_indices]
    
    out_dir = '/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons/karplus_strong'
    os.makedirs(out_dir, exist_ok=True)
    
    for img_path in test_paths:
        im = cv2.imread(img_path)
        if im is None: continue
        name = os.path.basename(img_path).replace('.jpg', '')
        
        r = cv2.resize(im, (SZ, SZ))
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
        L = lab[:,:,0]
        
        # DDColor
        gbgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t_in = torch.from_numpy(cv2.resize(gbgr, (256,256)).transpose(2,0,1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            ab_dd = v16.forward(t_in)
        ab_ddcolor = cv2.resize(ab_dd[0].permute(1,2,0).numpy(), (SZ, SZ))
        
        # Lattice
        ab_lattice = nav.colorize(gray)
        
        # KS: Three modes
        # Mode 1: KS from scratch (scaffolding + noise only)
        print(f'\n--- {name} ---')
        print(f'KS from scratch:')
        ab_ks_scratch, hist_scratch = ks.colorize(gray, lattice_ab=None, n_iterations=30)
        
        # Mode 2: KS seeded from lattice
        print(f'KS from lattice:')
        ab_ks_lattice, hist_lattice = ks.colorize(gray, lattice_ab=ab_lattice, n_iterations=30)
        
        # Build comparison strip: Gray | Lattice | KS-scratch | KS-lattice | DDColor | GT
        imgs = [
            (ab_to_bgr(np.zeros((SZ,SZ,2)), L), 'Gray'),
            (ab_to_bgr(ab_lattice, L), f'Lattice s={get_sat(ab_to_bgr(ab_lattice, L)):.0f}'),
            (ab_to_bgr(ab_ks_scratch, L), f'KS-scratch s={get_sat(ab_to_bgr(ab_ks_scratch, L)):.0f}'),
            (ab_to_bgr(ab_ks_lattice, L), f'KS-lattice s={get_sat(ab_to_bgr(ab_ks_lattice, L)):.0f}'),
            (ab_to_bgr(ab_ddcolor, L), f'DDColor s={get_sat(ab_to_bgr(ab_ddcolor, L)):.0f}'),
            (r, f'GT s={get_sat(r):.0f}'),
        ]
        
        for img, label in imgs:
            cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 2)
            cv2.putText(img, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        
        strip = np.hstack([img for img, _ in imgs])
        cv2.imwrite(os.path.join(out_dir, f'comparison_{name}.jpg'), strip)
        
        # Error analysis
        err_lat = np.sqrt(np.mean((ab_lattice - ab_ddcolor)**2))
        err_ks_s = np.sqrt(np.mean((ab_ks_scratch - ab_ddcolor)**2))
        err_ks_l = np.sqrt(np.mean((ab_ks_lattice - ab_ddcolor)**2))
        
        print(f'  Errors to DDColor: lattice={err_lat:.1f}, KS-scratch={err_ks_s:.1f}, KS-lattice={err_ks_l:.1f}')
        
        # Iteration video: watch the "string vibrate" as it converges
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(out_dir, f'convergence_{name}.mp4')
        frame_h = SZ + 30
        frame_w = SZ * 4
        writer = cv2.VideoWriter(video_path, fourcc, 10, (frame_w, frame_h))
        
        for it_idx, ab_it in enumerate(hist_lattice):
            bgr_it = ab_to_bgr(ab_it, L)
            bgr_dd = ab_to_bgr(ab_ddcolor, L)
            
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            frame[0:SZ, 0:SZ] = ab_to_bgr(ab_lattice, L)
            frame[0:SZ, SZ:SZ*2] = bgr_it
            frame[0:SZ, SZ*2:SZ*3] = bgr_dd
            frame[0:SZ, SZ*3:SZ*4] = r
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            sat_it = get_sat(bgr_it)
            labels = [
                ('Lattice', 5),
                (f'KS iter={it_idx} s={sat_it:.0f}', SZ + 5),
                ('DDColor', SZ*2 + 5),
                ('GT', SZ*3 + 5),
            ]
            for txt, xo in labels:
                cv2.putText(frame, txt, (xo, SZ + 18), font, 0.4, (255,255,255), 1)
            
            writer.write(frame)
        
        writer.release()
        print(f'  Convergence video: {video_path}')
    
    print('\n=== SUMMARY ===')
    print('The Karplus-Strong colorizer applies iterative constraints')
    print('(edge diffusion, region coherence, histogram matching, scale consistency)')
    print('to cause color to "resonate" within edge-bounded regions.')
    print(f'\nResults saved to: {out_dir}/')
    print('\nDone!')
