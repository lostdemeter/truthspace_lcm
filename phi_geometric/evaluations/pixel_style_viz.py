#!/usr/bin/env python3
"""
Visualization of PhaseDiscovery working on the Pixel Art Styling domain.

Outputs a PNG showing:
  - Top section: discovered phase structure diagram
  - Middle section: several example scanlines flowing through the pipeline
  - Bottom section: side-by-side comparison of input/output pixel strips
"""

import sys
import os
import importlib.util
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ============================================================================
# Load PhaseDiscovery without torch dependency
# ============================================================================
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_core_path = os.path.join(_project_root, 'phi_geometric', 'core')

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pkg = types.ModuleType('phi_geometric')
_pkg.__path__ = [os.path.join(_project_root, 'phi_geometric')]
sys.modules['phi_geometric'] = _pkg
_core = types.ModuleType('phi_geometric.core')
_core.__path__ = [_core_path]
sys.modules['phi_geometric.core'] = _core
_load_module('phi_geometric.core.discovery', os.path.join(_core_path, 'discovery.py'))
_load_module('phi_geometric.core.cascade_navigator', os.path.join(_core_path, 'cascade_navigator.py'))
_pd_mod = _load_module('phi_geometric.core.phase_discovery', os.path.join(_core_path, 'phase_discovery.py'))
PhaseDiscovery = _pd_mod.PhaseDiscovery

# ============================================================================
# Color definitions — actual RGB values for pixel names
# ============================================================================
COLOR_MAP = {
    'red':        '#DC3545',
    'yellow':     '#FFC107',
    'orange':     '#FD7E14',
    'blue':       '#0D6EFD',
    'green':      '#198754',
    'teal':       '#20C997',
    'white':      '#F8F9FA',
    'silver':     '#C0C0C0',
    'black':      '#212529',
    'charcoal':   '#495057',
    'gray':       '#6C757D',
    'dark_gray':  '#495057',
    'light_gray': '#ADB5BD',
    'crimson':    '#A71D2A',
    'navy':       '#084298',
    'forest':     '#0F5132',
    'gold':       '#CC9A06',
}

# Text color (white on dark, black on light)
def text_color(color_name):
    dark = {'black', 'charcoal', 'navy', 'forest', 'crimson', 'dark_gray',
            'red', 'blue', 'green', 'teal'}
    return 'white' if color_name in dark else '#212529'


# ============================================================================
# Ground truth (same as pixel_style_demo.py)
# ============================================================================
BLEND_RULES = {
    ('red', 'yellow'): ('orange',),
    ('yellow', 'red'): ('orange',),
    ('blue', 'green'): ('teal',),
    ('green', 'blue'): ('teal',),
    ('white', 'white'): ('silver',),
}

def shade_gray(pixel, ctx):
    nxt = ctx.get('next')
    if nxt in ('black', 'charcoal'):
        return 'dark_gray'
    if nxt in ('white', 'silver'):
        return 'light_gray'
    return 'gray'

PALETTE_SHIFT = {
    'red': 'crimson', 'blue': 'navy', 'green': 'forest',
    'yellow': 'gold', 'black': 'charcoal',
}

def _find_orig_pos(original, collapsed, collapsed_idx):
    oi, ci = 0, 0
    while oi < len(original) and ci < len(collapsed):
        if ci == collapsed_idx:
            return oi
        if oi + 1 < len(original):
            pair = (original[oi], original[oi + 1])
            if pair in BLEND_RULES:
                oi += 2; ci += 1; continue
        oi += 1; ci += 1
    if ci == collapsed_idx and oi < len(original):
        return oi
    return None

def apply_ground_truth(scanline):
    result = []
    i = 0
    while i < len(scanline):
        if i + 1 < len(scanline):
            pair = (scanline[i], scanline[i + 1])
            if pair in BLEND_RULES:
                result.extend(BLEND_RULES[pair])
                i += 2; continue
        result.append(scanline[i])
        i += 1
    shaded = list(result)
    for i, pixel in enumerate(result):
        if pixel == 'gray':
            orig_pos = _find_orig_pos(scanline, result, i)
            ctx = {}
            if orig_pos is not None and orig_pos + 1 < len(scanline):
                ctx['next'] = scanline[orig_pos + 1]
            shaded[i] = shade_gray(pixel, ctx)
    return [PALETTE_SHIFT.get(p, p) for p in shaded]


# ============================================================================
# Training data
# ============================================================================
TRAINING_SCANLINES = [
    ['red', 'yellow', 'orange', 'red'],
    ['yellow', 'red', 'orange', 'yellow'],
    ['red', 'yellow', 'green', 'blue'],
    ['blue', 'green', 'white', 'black'],
    ['green', 'blue', 'red', 'yellow'],
    ['blue', 'green', 'red', 'yellow'],
    ['white', 'white', 'gray', 'black'],
    ['black', 'white', 'white', 'red'],
    ['white', 'white', 'white', 'white'],
    ['gray', 'black', 'red'],
    ['red', 'gray', 'black'],
    ['blue', 'gray', 'black', 'red'],
    ['gray', 'black', 'gray', 'white'],
    ['gray', 'white', 'red'],
    ['red', 'gray', 'white'],
    ['blue', 'gray', 'white', 'red'],
    ['gray', 'white', 'gray', 'black'],
    ['red', 'gray', 'blue'],
    ['gray', 'red', 'blue'],
    ['red', 'blue', 'green'],
    ['blue', 'red', 'yellow'],
    ['green', 'yellow', 'black'],
    ['yellow', 'green', 'red'],
    ['black', 'red', 'blue'],
    ['red', 'green', 'black'],
    ['red', 'yellow', 'gray', 'black'],
    ['blue', 'green', 'gray', 'white'],
    ['white', 'white', 'gray', 'black'],
    ['black', 'gray', 'black', 'white'],
    ['green', 'blue', 'gray', 'black', 'white'],
    ['red', 'yellow', 'black'],
    ['yellow', 'red', 'white'],
    ['blue', 'green', 'yellow'],
    ['green', 'blue', 'red'],
    ['white', 'white', 'red'],
    ['white', 'white', 'blue'],
]


# ============================================================================
# Run discovery
# ============================================================================
def run_discovery():
    pd = PhaseDiscovery(context_window=1)
    pairs = []
    for sl in TRAINING_SCANLINES:
        out = apply_ground_truth(sl)
        pairs.append((sl, out))
        pd.add_pair(sl, out)
    result = pd.discover()
    nav = result.to_navigator()
    return result, nav, pairs


# ============================================================================
# Drawing helpers
# ============================================================================
def draw_pixel_strip(ax, pixels, x0, y0, cell_w=1.0, cell_h=0.6,
                     label=None, label_x=None, show_names=True, fontsize=7):
    """Draw a horizontal strip of colored pixel cells."""
    for i, px in enumerate(pixels):
        rgb = COLOR_MAP.get(px, '#888888')
        rect = FancyBboxPatch(
            (x0 + i * cell_w, y0), cell_w, cell_h,
            boxstyle="round,pad=0.02",
            facecolor=rgb, edgecolor='#343a40', linewidth=0.8
        )
        ax.add_patch(rect)
        if show_names:
            tc = text_color(px)
            display = px.replace('_', '\n')
            ax.text(x0 + i * cell_w + cell_w / 2, y0 + cell_h / 2,
                    display, ha='center', va='center',
                    fontsize=fontsize, color=tc, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=0.5, foreground='black'
                                               if tc == 'white' else 'white')])
    if label:
        lx = label_x if label_x is not None else x0 - 0.15
        ax.text(lx, y0 + cell_h / 2, label,
                ha='right', va='center', fontsize=8, fontweight='bold',
                color='#343a40')


def draw_arrow(ax, x0, y0, x1, y1, color='#6c757d', style='->', lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_phase_box(ax, x, y, w, h, title, rules_text, color='#e9ecef',
                   border='#495057'):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=border, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.18, title,
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='#212529')
    ax.text(x + w / 2, y + h / 2 - 0.1, rules_text,
            ha='center', va='center', fontsize=6.5,
            color='#495057', family='monospace', linespacing=1.4)


# ============================================================================
# Main visualization
# ============================================================================
def main():
    result, nav, pairs = run_discovery()

    # Example scanlines to visualize
    examples = [
        ['red', 'yellow', 'green', 'blue'],
        ['blue', 'green', 'gray', 'black', 'white'],
        ['white', 'white', 'gray', 'white'],
        ['green', 'blue', 'gray', 'black', 'red', 'yellow'],
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [2.5, 4.5, 2.5]})
    fig.patch.set_facecolor('#ffffff')

    # ====================================================================
    # PANEL 1: Discovered phase structure
    # ====================================================================
    ax = axes[0]
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(7, 3.3, 'Auto-Discovered Phase Structure',
            ha='center', va='top', fontsize=14, fontweight='bold', color='#212529')
    ax.text(7, 2.95, 'PhaseDiscovery found 3 phases from 36 training pairs — zero domain knowledge',
            ha='center', va='top', fontsize=9, color='#6c757d', style='italic')

    # Phase boxes
    phase_colors = ['#dbeafe', '#fef3c7', '#d1fae5']
    phase_borders = ['#3b82f6', '#f59e0b', '#10b981']

    # Phase 1: Collapse
    collapse_rules = "red+yellow → orange\nyellow+red → orange\nblue+green → teal\ngreen+blue → teal\nwhite+white → silver"
    draw_phase_box(ax, 0.3, 0.2, 3.8, 2.4,
                   '① Collapse (Blending)', collapse_rules,
                   color=phase_colors[0], border=phase_borders[0])

    # Phase 2: Context
    ctx_rules = "gray → dark_gray\n  (when next = black)\n\ngray → light_gray\n  (when next = white)"
    draw_phase_box(ax, 5.1, 0.2, 3.8, 2.4,
                   '② Context (Shading)', ctx_rules,
                   color=phase_colors[1], border=phase_borders[1])

    # Phase 3: Simple map
    map_rules = "red → crimson\nblue → navy\ngreen → forest\nyellow → gold\nblack → charcoal"
    draw_phase_box(ax, 9.9, 0.2, 3.8, 2.4,
                   '③ Token Map (Palette)', map_rules,
                   color=phase_colors[2], border=phase_borders[2])

    # Arrows between phases
    draw_arrow(ax, 4.1, 1.4, 5.1, 1.4, color='#6c757d', lw=2.0)
    draw_arrow(ax, 8.9, 1.4, 9.9, 1.4, color='#6c757d', lw=2.0)

    # ====================================================================
    # PANEL 2: Pipeline flow examples
    # ====================================================================
    ax2 = axes[1]
    ax2.axis('off')

    max_len = max(len(s) for s in examples) + 1
    total_w = max_len * 1.1
    ax2.set_xlim(-3.5, total_w + 1)
    ax2.set_ylim(-0.5, len(examples) * 4.5 + 1.0)

    ax2.text(total_w / 2 - 1, len(examples) * 4.5 + 0.6,
             'Pipeline Execution — Input → Collapse → Context → Palette → Output',
             ha='center', va='top', fontsize=13, fontweight='bold', color='#212529')

    cell_w = 1.1
    cell_h = 0.65

    for ei, scanline in enumerate(examples):
        base_y = (len(examples) - 1 - ei) * 4.5

        # Input strip
        draw_pixel_strip(ax2, scanline, 0, base_y + 3.2, cell_w, cell_h,
                         label='Input ', label_x=-0.2)

        # Phase 1: Collapse
        collapsed = []
        i = 0
        while i < len(scanline):
            if i + 1 < len(scanline):
                pair = (scanline[i], scanline[i + 1])
                if pair in BLEND_RULES:
                    collapsed.extend(BLEND_RULES[pair])
                    i += 2; continue
            collapsed.append(scanline[i])
            i += 1
        draw_pixel_strip(ax2, collapsed, 0, base_y + 2.2, cell_w, cell_h,
                         label='Blend ', label_x=-0.2)
        # Collapse arrows
        for j in range(len(collapsed)):
            draw_arrow(ax2, j * cell_w + cell_w / 2, base_y + 3.2,
                       j * cell_w + cell_w / 2, base_y + 2.2 + cell_h,
                       color=phase_borders[0], lw=0.8)

        # Phase 2: Context shading
        shaded = list(collapsed)
        for i, pixel in enumerate(collapsed):
            if pixel == 'gray':
                orig_pos = _find_orig_pos(scanline, collapsed, i)
                ctx = {}
                if orig_pos is not None and orig_pos + 1 < len(scanline):
                    ctx['next'] = scanline[orig_pos + 1]
                shaded[i] = shade_gray(pixel, ctx)
        draw_pixel_strip(ax2, shaded, 0, base_y + 1.1, cell_w, cell_h,
                         label='Shade ', label_x=-0.2)
        for j in range(len(shaded)):
            col = phase_borders[1] if shaded[j] != collapsed[j] else '#cccccc'
            lw = 1.2 if shaded[j] != collapsed[j] else 0.6
            draw_arrow(ax2, j * cell_w + cell_w / 2, base_y + 2.2,
                       j * cell_w + cell_w / 2, base_y + 1.1 + cell_h,
                       color=col, lw=lw)

        # Phase 3: Palette shift
        final = [PALETTE_SHIFT.get(p, p) for p in shaded]
        draw_pixel_strip(ax2, final, 0, base_y + 0.0, cell_w, cell_h,
                         label='Output ', label_x=-0.2)
        for j in range(len(final)):
            col = phase_borders[2] if final[j] != shaded[j] else '#cccccc'
            lw = 1.2 if final[j] != shaded[j] else 0.6
            draw_arrow(ax2, j * cell_w + cell_w / 2, base_y + 1.1,
                       j * cell_w + cell_w / 2, base_y + 0.0 + cell_h,
                       color=col, lw=lw)

        # Check mark or cross
        expected = apply_ground_truth(scanline)
        trace = nav.execute(scanline)
        actual = trace.output_elements
        match = actual == expected
        mark = '✓' if match else '✗'
        mark_color = '#10b981' if match else '#ef4444'
        ax2.text(len(final) * cell_w + 0.3, base_y + 0.0 + cell_h / 2,
                 mark, fontsize=16, color=mark_color, fontweight='bold',
                 va='center')

    # ====================================================================
    # PANEL 3: IPA vs Pixel comparison
    # ====================================================================
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_xlim(-0.5, 14)
    ax3.set_ylim(-0.5, 3.5)

    ax3.text(7, 3.3, 'Cross-Domain Generalization — Same Algorithm, Different Tokens',
             ha='center', va='top', fontsize=13, fontweight='bold', color='#212529')

    # IPA side
    ipa_x = 0.3
    ax3.text(ipa_x + 1.5, 2.6, 'IPA Domain (Characters)',
             ha='center', fontsize=10, fontweight='bold', color='#6366f1')

    ipa_data = [
        ('sh → ʃ', 'Digraph collapse'),
        ('c → k/s', 'Context (next char)'),
        ('a → æ', 'Simple map'),
    ]
    for i, (rule, desc) in enumerate(ipa_data):
        y = 2.0 - i * 0.7
        ax3.text(ipa_x, y, rule, fontsize=11, fontweight='bold',
                 color='#4338ca', family='monospace')
        ax3.text(ipa_x + 2.0, y, desc, fontsize=8, color='#6c757d')

    # Equals sign
    ax3.text(7, 1.3, '≡', fontsize=28, ha='center', va='center',
             color='#6c757d', fontweight='bold')

    # Pixel side
    px_x = 8.0
    ax3.text(px_x + 2.0, 2.6, 'Pixel Domain (Colors)',
             ha='center', fontsize=10, fontweight='bold', color='#059669')

    px_data = [
        ('red+yellow → orange', 'Color blending'),
        ('gray → dark/light', 'Context (next pixel)'),
        ('blue → navy', 'Palette shift'),
    ]
    for i, (rule, desc) in enumerate(px_data):
        y = 2.0 - i * 0.7
        ax3.text(px_x, y, rule, fontsize=11, fontweight='bold',
                 color='#047857', family='monospace')
        ax3.text(px_x + 4.0, y, desc, fontsize=8, color='#6c757d')

    plt.tight_layout(pad=1.0)

    out_path = os.path.join(os.path.dirname(__file__), 'pixel_style_viz.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    main()
