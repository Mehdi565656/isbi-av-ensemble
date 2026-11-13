#!/usr/bin/env python3
"""Generate PFE report figures 3–9."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import white_tophat, distance_transform_edt
import cv2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

BASE = Path('/path/to/workspace')
OUT  = BASE / 'pfe_report' / 'images'
OUT.mkdir(parents=True, exist_ok=True)

DRIVE_TRAIN = BASE / 'drive_kaggle_prepared_data' / 'train_data_s'
FUNDUS_TEST = BASE / 'Fundus-AVSeg-prep-data' / 'test'
SM_DIR      = BASE / 'paper_fixes' / 'data' / 'softmax_maps'
BINARY_MDL  = BASE / 'binary' / 'models' / 'best_model_BIFPN_3.pth'


# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw a rounded rectangle box with label
# ──────────────────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, sublabel='', color='#4472C4',
             fontsize=9, subfontsize=7.5, text_color='white'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.03', facecolor=color,
                         edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(box)
    dy = 0.08 if sublabel else 0
    ax.text(x, y + dy, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x, y - dy - 0.05, sublabel, ha='center', va='center',
                fontsize=subfontsize, color=text_color, zorder=4, style='italic')


def arrow(ax, x0, y0, x1, y1, color='#333333', lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Project Planning Timeline
# ══════════════════════════════════════════════════════════════════════════════
def fig3_timeline():
    """Milestone-arrow timeline: thick blue arrow, vertical ticks, bullet sub-items.
    Inspired by the horizontal milestone style with alternating above/below labels."""

    BLUE  = '#1A7DC4'
    BLACK = '#111111'
    GRAY  = '#666666'

    # (title, period, 'up'|'down', [bullet lines])
    phases = [
        (
            "Bibliographie\n& Données",
            "Sem. 1–2", "down",
            ["Revue des méthodes de segmentation",
             "Étude du dataset Fundus-AVSeg"],
        ),
        (
            "Implémentation\ndes baselines",
            "Sem. 3–6", "up",
            ["U-Net classique sur DRIVE",
             "Évaluation initiale"],
        ),
        (
            "Développement des\narchitectures",
            "Sem. 7–10", "down",
            ["Morpho-U-Net",
             "BiFPN-U-Net",
             "Morpho + BiFPN"],
        ),
        (
            "Expérimentations\net validation",
            "Sem. 11–13", "up",
            ["Entraînements sur Fundus-AVSeg",
             "Ensemble soft-voting",
             "Analyse et rédaction"],
        ),
    ]

    # x-positions of the four milestones
    xs = [1.8, 5.3, 8.8, 12.3]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0.0, 15.5)
    ax.set_ylim(-4.6, 4.4)
    ax.axis('off')

    # ── Main horizontal arrow ──────────────────────────────────────────────────
    ax.annotate(
        '', xy=(15.0, 0), xytext=(0.4, 0),
        arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=4.0, mutation_scale=26),
    )

    # ── Per-milestone layout constants ─────────────────────────────────────────
    TICK_H  = 0.75   # half-height of the vertical tick (extends ±TICK_H from axis)
    TTL_LH  = 0.52   # height per title line (fontsize 13)
    BUL_LH  = 0.44   # height per bullet line (fontsize 11)
    PER_H   = 0.30   # period label height (fontsize 9)
    GAP_TP  = 0.18   # gap between tick edge and first element
    GAP_TB  = 0.22   # gap between period and title
    GAP_BT  = 0.20   # gap between title and first bullet

    for i, (title, period, pos, bullets) in enumerate(phases):
        x = xs[i]
        n_title   = title.count('\n') + 1
        n_bullets = len(bullets)

        # ── Vertical tick ──────────────────────────────────────────────────────
        ax.plot([x, x], [-TICK_H, TICK_H],
                color=BLUE, lw=3.0, solid_capstyle='round', zorder=3)

        if pos == 'up':
            # Layout from bottom (tick top) upward:
            #   period → gap → bullets (last→first) → gap → title
            y = TICK_H + GAP_TP

            # Period just above tick
            ax.text(x, y, period,
                    ha='center', va='bottom', fontsize=9,
                    color=GRAY, style='italic')
            y += PER_H + GAP_TB

            # Bullets — iterate reverse so bullet[0] ends up at the TOP
            for j in range(n_bullets - 1, -1, -1):
                ax.text(x, y, f'• {bullets[j]}',
                        ha='center', va='bottom', fontsize=11, color=BLACK)
                y += BUL_LH

            y += GAP_BT

            # Title (topmost element)
            ax.text(x, y, title,
                    ha='center', va='bottom', fontsize=13, fontweight='bold',
                    color=BLACK, linespacing=1.4, multialignment='center')

        else:  # down
            # Layout from top (tick bottom) downward:
            #   period → gap → title → gap → bullets
            y = -TICK_H - GAP_TP

            # Period just below tick
            ax.text(x, y, period,
                    ha='center', va='top', fontsize=9,
                    color=GRAY, style='italic')
            y -= PER_H + GAP_TB

            # Title
            ax.text(x, y, title,
                    ha='center', va='top', fontsize=13, fontweight='bold',
                    color=BLACK, linespacing=1.4, multialignment='center')
            y -= n_title * TTL_LH + GAP_BT

            # Bullets
            for b in bullets:
                ax.text(x, y, f'• {b}',
                        ha='center', va='top', fontsize=11, color=BLACK)
                y -= BUL_LH

    plt.savefig(OUT / 'project_planning_timeline.pdf',
                bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print('✓ Fig 3: project_planning_timeline.pdf  (milestone arrow style)')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Binary BiFPN Architecture  (paper-inspired style)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_bifpn_arch():
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17); ax.set_ylim(0, 9)
    ax.axis('off')

    # ── 3-D tensor block ──────────────────────────────────────────────────────
    def block3d(x, y, w, h, color, label='', n=3, gap=0.13, z=3, fs=7.5):
        """Stacked 3-D slabs.  x,y = bottom-left of front slab."""
        dx, dy = 0.13, 0.08
        rgb = mcolors.to_rgb(color)
        for k in range(n):                       # back→front
            shade = tuple(min(1.0, c*(0.65 + 0.17*k)) for c in rgb)
            kx = x + (n-1-k)*gap;  ky = y - (n-1-k)*gap*0.5
            # back/side faces (just the back slab outline gives depth illusion)
            if k == n-1:                          # front slab only: add 3-D faces
                top = plt.Polygon(
                    [(kx,ky+h),(kx+dx,ky+h+dy),(kx+w+dx,ky+h+dy),(kx+w,ky+h)],
                    fc=tuple(min(1.,c*1.4) for c in mcolors.to_rgb(color)),
                    ec='none', zorder=z+0.1)
                side = plt.Polygon(
                    [(kx+w,ky),(kx+w+dx,ky+dy),(kx+w+dx,ky+h+dy),(kx+w,ky+h)],
                    fc=tuple(c*0.55 for c in mcolors.to_rgb(color)),
                    ec='none', zorder=z+0.1)
                ax.add_patch(top); ax.add_patch(side)
            rect = plt.Rectangle((kx, ky), w, h,
                                  fc=shade, ec='white', lw=0.6, zorder=z+k*0.1)
            ax.add_patch(rect)
        if label:
            ax.text(x+w/2+(n-1)*gap*0.5, y+h/2-(n-1)*gap*0.25,
                    label, ha='center', va='center',
                    fontsize=fs, color='white', fontweight='bold', zorder=z+n)

    # ── Arrow wrappers ────────────────────────────────────────────────────────
    def arr(x0,y0,x1,y1, col='#555', lw=1.5, ls='-', rad=0.0):
        ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                    linestyle=ls,
                                    connectionstyle=f'arc3,rad={rad}'))

    # ── Colours ───────────────────────────────────────────────────────────────
    C_ENC  = '#C55A96'   # encoder  — magenta
    C_PROJ = '#4472C4'   # 1×1 proj — blue
    C_TD   = '#FFC000'   # BiFPN TD — amber
    C_BU   = '#70AD47'   # BiFPN BU — green
    C_DEC  = '#ED7D31'   # decoder  — orange
    C_POOL = '#7030A0'   # pool/up  — purple
    C_OUT  = '#C00000'   # output   — red
    C_GRY  = '#888888'

    # ── Level positions (P0=top=H×W … P3=bottom=H/8) ─────────────────────────
    ys   = [7.8, 6.2, 4.6, 3.0]
    res  = ['H×W', 'H/2×W/2', 'H/4×W/4', 'H/8×W/8']
    chs  = [64, 128, 256, 512]
    # Block sizes scale with resolution (bigger block = more spatial pixels)
    bws  = [1.10, 0.90, 0.72, 0.56]
    bhs  = [1.00, 0.82, 0.66, 0.52]

    # ══════════════════════════════════════════════════════════════════════════
    # INPUT
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.55, ys[0], 'LIOT\n10 ch.\nH×W', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#4472C4',
            bbox=dict(boxstyle='round,pad=0.35', fc='#E8F0FE', ec='#4472C4', lw=1.5,
                      zorder=5))
    arr(0.9, ys[0], 1.4, ys[0], C_GRY, lw=1.8)

    # ══════════════════════════════════════════════════════════════════════════
    # ENCODER  (blocks aligned on right edge at x = 3.4)
    # ══════════════════════════════════════════════════════════════════════════
    ENC_R = 3.4          # right edge of all encoder blocks
    ENC_N = 3            # number of slabs

    for i, (y, ch, bw, bh) in enumerate(zip(ys, chs, bws, bhs)):
        x = ENC_R - bw - (ENC_N-1)*0.13   # leftmost slab left edge
        block3d(x, y-bh/2, bw, bh, C_ENC, label=f'{ch}ch', n=ENC_N, z=4+i*0.1)

        # Resolution label (left)
        ax.text(0.95, y, res[i], ha='right', va='center',
                fontsize=7, color='#666', style='italic')

        # MaxPool arrow down to next level
        if i < 3:
            y_bot = y - bh/2 - 0.06
            y_top = ys[i+1] + bhs[i+1]/2 + 0.06
            arr(ENC_R-bws[i]/2, y_bot, ENC_R-bws[i+1]/2, y_top, C_POOL, lw=2)
            ax.text(ENC_R-bws[i]/2+0.18, (y_bot+y_top)/2,
                    'MaxPool\n↓2', ha='left', va='center',
                    fontsize=6.5, color=C_POOL)

    # ══════════════════════════════════════════════════════════════════════════
    # 1×1 PROJECTION
    # ══════════════════════════════════════════════════════════════════════════
    PR_X = 4.6    # centre x of proj blocks

    for i, y in enumerate(ys):
        pw, ph = 0.55, 0.45
        block3d(PR_X-pw/2, y-ph/2, pw, ph, C_PROJ, label='1×1', n=1, z=6, fs=7)
        # Arrow enc → proj
        arr(ENC_R+0.13, y, PR_X-pw/2-0.05, y, C_GRY, lw=1)

    # ══════════════════════════════════════════════════════════════════════════
    # BiFPN DASHED BOX
    # ══════════════════════════════════════════════════════════════════════════
    BF_L, BF_R = 5.2, 10.0
    BF_B, BF_T = 2.25, 8.45
    ax.add_patch(FancyBboxPatch(
        (BF_L, BF_B), BF_R-BF_L, BF_T-BF_B,
        boxstyle='round,pad=0.1', fc='#FFFDE7', ec='#FFC000',
        lw=2, ls='--', zorder=1))
    ax.text((BF_L+BF_R)/2, BF_T+0.10,
            'Réseau Pyramidal de Traits Bidirectionnel (BiFPN)',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold',
            color='#6B4C00',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec='#FFC000', lw=1.2, alpha=0.97, zorder=8))

    # ══════════════════════════════════════════════════════════════════════════
    # BiFPN NODES  (two columns: TD left, BU/output right)
    # ══════════════════════════════════════════════════════════════════════════
    TD_X  = 6.5    # TD column x
    BU_X  = 8.9    # BU column x
    NR    = 0.30   # node radius
    P_LBL = ['P₀', 'P₁', 'P₂', 'P₃']

    for i, (y, p) in enumerate(zip(ys, P_LBL)):
        # Arrow proj → TD node
        arr(PR_X+0.55/2, y, TD_X-NR, y, C_GRY, lw=1)

        # TD node
        ax.add_patch(plt.Circle((TD_X, y), NR, fc=C_TD, ec='white', lw=2, zorder=7))
        ax.text(TD_X, y, p, ha='center', va='center',
                fontsize=9, color='#333', fontweight='bold', zorder=8)

        # BU node
        ax.add_patch(plt.Circle((BU_X, y), NR, fc=C_BU, ec='white', lw=2, zorder=7))
        ax.text(BU_X, y, p, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=8)

        # Horizontal cross arrow TD → BU
        arr(TD_X+NR, y, BU_X-NR, y, C_GRY, lw=1.2)

        # Channel label between columns
        ax.text((TD_X+BU_X)/2, y+NR+0.10, '128 ch',
                ha='center', va='bottom', fontsize=6.5,
                color='#888', style='italic')

    # TD downward arrows (from P0 top → P3 bottom, left of TD column)
    for i in range(3):
        arr(TD_X-0.45, ys[i]-NR, TD_X-0.45, ys[i+1]+NR,
            '#9C6900', lw=1.8)
    ax.text(TD_X-0.65, (ys[0]+ys[3])/2, 'TD',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#9C6900', rotation=90)

    # BU upward arrows (from P3 bottom → P0 top, right of BU column)
    for i in range(3, 0, -1):
        arr(BU_X+0.45, ys[i]+NR, BU_X+0.45, ys[i-1]-NR,
            '#2E6B23', lw=1.8)
    ax.text(BU_X+0.65, (ys[0]+ys[3])/2, 'BU',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#2E6B23', rotation=90)

    # Column sub-labels
    ax.text(TD_X, BF_B+0.12, 'Intermédiaire', ha='center', va='bottom',
            fontsize=7, color='#9C6900', style='italic')
    ax.text(BU_X, BF_B+0.12, 'Sortie', ha='center', va='bottom',
            fontsize=7, color='#2E6B23', style='italic')

    # ══════════════════════════════════════════════════════════════════════════
    # CENTER CONV BLOCK  (at P3 level, between BiFPN and decoder)
    # ══════════════════════════════════════════════════════════════════════════
    CC_X = 10.7
    cw, ch_sz = 1.1, 0.52
    block3d(CC_X-cw/2, ys[3]-ch_sz/2, cw, ch_sz,
            '#555555', label='ConvBlock', n=1, z=6, fs=7)
    arr(BU_X+NR, ys[3], CC_X-cw/2-0.05, ys[3], C_GRY, lw=1.3)

    # ══════════════════════════════════════════════════════════════════════════
    # DECODER  (3 UpBlocks going from P3 bottom up to P0 top)
    # ══════════════════════════════════════════════════════════════════════════
    DC_X  = 12.0   # left edge of decoder blocks (variable width)
    DEC_N = 2      # slabs

    dec_levels = [2, 1, 0]   # which ys[] level each UpBlock outputs to
    dec_bws    = [bws[2], bws[1], bws[0]]
    dec_bhs    = [bhs[2], bhs[1], bhs[0]]

    prev_right = CC_X + cw/2 + 0.13   # right edge of center block
    prev_y     = ys[3]

    for k, (lv, bw_d, bh_d) in enumerate(zip(dec_levels, dec_bws, dec_bhs)):
        y_d = ys[lv]

        # Upsample arrow: diagonal from previous block to this block
        arr(prev_right, prev_y, DC_X-0.08, y_d, C_POOL, lw=2, rad=-0.12)
        ax.text((prev_right+DC_X)/2-0.15, (prev_y+y_d)/2+0.18,
                '↑2', ha='center', va='bottom', fontsize=8, color=C_POOL,
                fontweight='bold')

        # Decoder block
        block3d(DC_X, y_d-bh_d/2, bw_d, bh_d, C_DEC,
                label='128 ch', n=DEC_N, z=4+k*0.1, fs=7.5)

        # Resolution label
        ax.text(DC_X+bw_d+(DEC_N-1)*0.13+0.35, y_d,
                res[lv], ha='left', va='center',
                fontsize=7, color='#666', style='italic')

        # Skip from BiFPN BU node (same level)
        arr(BU_X+NR, y_d, DC_X-0.08, y_d,
            C_BU, lw=1.3, ls='dashed', rad=0.18)

        prev_right = DC_X + bw_d + (DEC_N-1)*0.13
        prev_y     = y_d

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT  (1×1 conv + sigmoid)
    # ══════════════════════════════════════════════════════════════════════════
    OX = 14.3
    arr(prev_right+0.05, ys[0], OX-0.3, ys[0], C_GRY, lw=1.8)
    block3d(OX-0.28, ys[0]-0.28, 0.56, 0.56, C_OUT,
            label='1 ch\n1×1', n=1, z=7, fs=7)
    ax.text(OX, ys[0]-0.52, 'Masque\nbinaire\nH×W',
            ha='center', va='top', fontsize=8, color=C_OUT, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    leg = [
        (C_ENC,  'ConvBlock encodeur (3×3, BN, ReLU)'),
        (C_PROJ, 'Projection 1×1'),
        (C_TD,   'Nœud BiFPN intermédiaire (TD)'),
        (C_BU,   'Nœud BiFPN sortie (BU)'),
        (C_DEC,  'UpBlock décodeur (déconv. + Conv.)'),
        (C_POOL, 'MaxPool ↓2 / Déconv. transposée ↑2'),
    ]
    lx0, ly0 = 5.0, 1.9
    # background panel
    ax.add_patch(plt.Rectangle((lx0-0.2, ly0-0.9), 9.2, 1.05,
                               fc='white', ec='#CCCCCC', lw=0.8, zorder=9, alpha=0.92))
    for j, (c, lbl) in enumerate(leg):
        xi = lx0 + (j % 3) * 3.05
        yi = ly0 - (j // 3) * 0.44
        ax.add_patch(plt.Rectangle((xi, yi-0.13), 0.32, 0.27,
                                   fc=c, ec='white', lw=0.5, zorder=10))
        ax.text(xi+0.41, yi, lbl, ha='left', va='center',
                fontsize=7, color='#333', zorder=10)

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(8.5, 8.82,
            'Architecture BiFPN-UNet — Segmentation binaire des vaisseaux rétiniens',
            ha='center', va='center', fontsize=12,
            fontweight='bold', color='#222222')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'binary_bifpn_architecture.pdf', bbox_inches='tight')
    plt.close()
    print('✓ Fig 4: binary_bifpn_architecture.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Binary Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def fig5_binary_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(-1.2, 2.5)
    ax.axis('off')

    boxes = [
        (1.3,  1.0, 2.2, 1.4, 'Image\nfondoscopique', '',          '#4472C4'),
        (4.5,  1.0, 2.8, 1.4, 'Extraction\nLIOT-EG',  '10 cartes', '#ED7D31'),
        (8.0,  1.0, 2.8, 1.4, 'BiFPN-UNet',           '(entraîné)', '#70AD47'),
        (11.5, 1.0, 2.2, 1.4, 'Masque\nbinaire',      '',           '#C00000'),
    ]

    for (x, y, w, h, lbl, sub, col) in boxes:
        draw_box(ax, x, y, w, h, lbl, sub, color=col, fontsize=10)

    # Arrows
    xs = [b[0] for b in boxes]
    ws = [b[2] for b in boxes]
    for i in range(len(boxes)-1):
        x0 = xs[i] + ws[i]/2
        x1 = xs[i+1] - ws[i+1]/2
        ax.annotate('', xy=(x1, 1.0), xytext=(x0, 1.0),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # LIOT sub-labels below
    ax.text(4.5, -0.05, '• 7 filtres morpho.\n• 3 canaux couleur',
            ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

    plt.tight_layout(pad=0.4)
    plt.savefig(OUT / 'binary_full_pipeline.pdf', bbox_inches='tight')
    plt.close()
    print('✓ Fig 5: binary_full_pipeline.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Soft-Voting Ensemble Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def fig6_ensemble_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # Input
    draw_box(ax, 1.2, 2.5, 2.0, 1.2, 'Image\nfondoscopique', '', '#4472C4', fontsize=9)

    # 3 model paths
    model_ys  = [4.2, 2.5, 0.8]
    model_lbls = ['U-Net\nstandard', 'Morpho-UNet', 'BiFPN-UNet\n64–1024 ch']
    model_cols = ['#2F74B0', '#ED7D31', '#70AD47']

    for y, lbl, col in zip(model_ys, model_lbls, model_cols):
        # Arrow from input
        ax.annotate('', xy=(3.5, y), xytext=(2.2, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2,
                                   connectionstyle='arc3,rad=0'))
        draw_box(ax, 5.3, y, 2.8, 1.1, lbl, 'softmax (3 cl.)', col, fontsize=9)
        # Arrow to average
        ax.annotate('', xy=(7.8, 2.5), xytext=(6.7, y),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2,
                                   connectionstyle='arc3,rad=0'))

    # Average softmax
    draw_box(ax, 9.2, 2.5, 2.4, 1.2, 'Moyenne\nsoftmax', '(vote doux)', '#7030A0', fontsize=9)
    ax.annotate('', xy=(8.8, 2.5), xytext=(8.0, 2.5),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # Argmax
    draw_box(ax, 11.4, 2.5, 1.8, 1.2, 'argmax', '', '#555555', fontsize=9)
    ax.annotate('', xy=(11.0, 2.5), xytext=(10.4, 2.5),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # Output
    draw_box(ax, 13.2, 2.5, 1.8, 1.2, 'Masque\nAV', '', '#C00000', fontsize=9)
    ax.annotate('', xy=(12.7, 2.5), xytext=(12.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # Bracket on left models
    ax.plot([3.5, 3.5], [0.25, 4.75], color='#AAAAAA', lw=1.5, ls='-')
    ax.plot([3.5, 3.7], [0.25, 0.25], color='#AAAAAA', lw=1.5)
    ax.plot([3.5, 3.7], [4.75, 4.75], color='#AAAAAA', lw=1.5)

    ax.plot([7.1, 7.1], [0.25, 4.75], color='#AAAAAA', lw=1.5, ls='-')
    ax.plot([7.1, 6.9], [0.25, 0.25], color='#AAAAAA', lw=1.5)
    ax.plot([7.1, 6.9], [4.75, 4.75], color='#AAAAAA', lw=1.5)

    plt.tight_layout(pad=0.4)
    plt.savefig(OUT / 'soft_voting_ensemble_pipeline.pdf', bbox_inches='tight')
    plt.close()
    print('✓ Fig 6: soft_voting_ensemble_pipeline.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Binary BiFPN model definition (matches checkpoint keys exactly)
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, chs):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.blocks.append(EncoderBlock(c, ch))
            self.pools.append(nn.MaxPool2d(2))
            c = ch
    def forward(self, x):
        feats = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x); feats.append(x); x = pool(x)
        return feats


class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        N = len(channels)
        self.num_layers = num_layers
        # Shared weights across layers
        self.w_td = nn.Parameter(torch.ones(num_layers, N, 2))
        self.w_bu = nn.Parameter(torch.ones(num_layers, N, 3))
        self.convs_td = nn.ModuleList([ConvBlock(c, c) for c in channels])
        self.convs_bu = nn.ModuleList([ConvBlock(c, c) for c in channels])

    def forward(self, P):
        N = len(P)
        cur = list(P)
        for layer in range(self.num_layers):
            # Top-down
            td = [None] * N; td[N-1] = cur[N-1]
            for i in range(N-2, -1, -1):
                w = F.softmax(self.w_td[layer, i], 0)
                up = F.interpolate(td[i+1], size=cur[i].shape[-2:], mode='nearest')
                td[i] = self.convs_td[i](w[0]*cur[i] + w[1]*up)
            # Bottom-up
            bu = [None] * N; bu[0] = td[0]
            for i in range(1, N):
                w = F.softmax(self.w_bu[layer, i], 0)
                down = F.max_pool2d(bu[i-1], kernel_size=2, stride=2,
                                    ceil_mode=False, padding=0)
                if down.shape[-2:] != td[i].shape[-2:]:
                    down = F.interpolate(down, size=td[i].shape[-2:], mode='nearest')
                bu[i] = self.convs_bu[i](w[0]*cur[i] + w[1]*td[i] + w[2]*down)
            cur = bu
        return cur


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class BinaryBiFPNUNet(nn.Module):
    def __init__(self):
        super().__init__()
        bifpn_ch = 128
        enc_chs  = [64, 128, 256, 512]
        self.encoder   = Encoder(in_ch=10, chs=enc_chs)
        self.proj      = nn.ModuleList([nn.Conv2d(c, bifpn_ch, 1) for c in enc_chs])
        self.bifpn     = BiFPN([bifpn_ch]*4, num_layers=2)
        self.center    = ConvBlock(bifpn_ch, bifpn_ch)
        self.up_blocks = nn.ModuleList([UpBlock(bifpn_ch, bifpn_ch, bifpn_ch)
                                        for _ in range(3)])
        self.final_conv = nn.Conv2d(bifpn_ch, 1, 1)

    def forward(self, x):
        enc   = self.encoder(x)
        enc   = [p(f) for p, f in zip(self.proj, enc)]
        fused = self.bifpn(enc)
        x = self.center(fused[-1])
        for i, up in enumerate(self.up_blocks):
            x = up(x, fused[-2-i])
        return torch.sigmoid(self.final_conv(x))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Binary Qualitative Example (DRIVE)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_binary_qualitative():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BinaryBiFPNUNet().to(device)
    ck = torch.load(BINARY_MDL, map_location=device)
    model.load_state_dict(ck, strict=True)
    model.eval()
    print(f'  Binary model loaded on {device}')

    # Load DRIVE training image 21
    img_path  = DRIVE_TRAIN / '21_training.tif'
    gt_path   = DRIVE_TRAIN / '21_manual1.gif'
    liot_path = DRIVE_TRAIN / '21_training_liot_eg.npy'

    img  = np.array(Image.open(img_path))               # (584, 565, 3) uint8
    gt   = np.array(Image.open(gt_path).convert('L'))   # (584, 565) uint8
    liot = np.load(liot_path).astype(np.float32) / 255.0  # (10, 584, 565) float

    # Inference
    with torch.no_grad():
        x = torch.from_numpy(liot).unsqueeze(0).to(device)   # (1,10,H,W)
        pred = model(x).squeeze().cpu().numpy()               # (H,W) float [0,1]

    pred_bin = (pred > 0.5).astype(np.uint8) * 255
    gt_bin   = (gt > 128).astype(np.uint8) * 255

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    titles = ['Image fondscopique (DRIVE)', 'Vérité terrain', 'Prédiction BiFPN-UNet']
    imgs   = [img, gt_bin, pred_bin]
    cmaps  = [None, 'gray', 'gray']

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')

    plt.suptitle('Segmentation binaire des vaisseaux rétiniens — DRIVE',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(OUT / 'binary_qualitative_example.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print('✓ Fig 7: binary_qualitative_example.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Helper: annotation → coloured segmentation mask
# ══════════════════════════════════════════════════════════════════════════════
def ann_to_mask(ann_np):
    """RGB annotation → coloured display image: artery=red, vein=blue, bg=black."""
    H, W = ann_np.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    r, g, b = ann_np[...,0], ann_np[...,1], ann_np[...,2]
    artery = (r > 180) & (g < 80)   # red
    vein   = (b > 180) & (r < 80)   # blue
    out[artery] = [220, 30, 30]
    out[vein]   = [30, 80, 220]
    return out


def softmax_to_seg(sm):
    """softmax (3,H,W) float → coloured seg image."""
    pred = np.argmax(sm, axis=0)   # (H,W) int
    out  = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[pred == 1] = [220, 30, 30]   # artery
    out[pred == 2] = [30, 80, 220]   # vein
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — AV Qualitative Comparison (Fundus-AVSeg 092_N)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_av_qualitative():
    stem = '092_N'
    img_np = np.array(Image.open(FUNDUS_TEST / 'images' / f'{stem}.png').resize((592, 592)))
    ann_np = np.array(Image.open(FUNDUS_TEST / 'annotation' / f'{stem}.png').resize(
        (592, 592), Image.NEAREST))

    gt_img = ann_to_mask(ann_np)

    # ResUNet softmax
    sm_res = np.load(SM_DIR / 'resunet' / f'{stem}.npy').astype(np.float32)
    res_img = softmax_to_seg(sm_res)

    # Ensemble = average of unet + morpho_unet + bifpn_64to1024
    sm_ens = (np.load(SM_DIR / 'unet'            / f'{stem}.npy').astype(np.float32) +
              np.load(SM_DIR / 'morpho_unet'     / f'{stem}.npy').astype(np.float32) +
              np.load(SM_DIR / 'bifpn_64to1024'  / f'{stem}.npy').astype(np.float32)) / 3.0
    ens_img = softmax_to_seg(sm_ens)

    # --- Figure ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    panels = [
        (img_np,  'Image originale\n(092_N)'),
        (gt_img,  'Vérité terrain\n(artères=rouge, veines=bleu)'),
        (res_img, 'ResUNet\n(Dice=0.8103)'),
        (ens_img, 'Ensemble × 3\n(Dice=0.8320)'),
    ]
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=9, fontweight='bold', pad=5)
        ax.axis('off')

    plt.suptitle('Comparaison qualitative — Segmentation AV (Fundus-AVSeg)',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(top=0.82)
    plt.savefig(OUT / 'av_qualitative_comparison.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print('✓ Fig 8: av_qualitative_comparison.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Morpho Channels Overview (3×3 grid)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_morpho(img_np):
    """img_np (H,W,3) uint8 → (wth_g, dist_g) both float [0,1]."""
    g = img_np[:, :, 1].astype(np.float32) / 255.0   # green channel

    # White top-hat with disk SE radius 7
    se_r = 7
    se   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*se_r+1, 2*se_r+1))
    g_u8  = (g * 255).astype(np.uint8)
    wth_raw = cv2.morphologyEx(g_u8, cv2.MORPH_TOPHAT, se).astype(np.float32)
    # Stretch to [0,1] using 99.5th-percentile cap so vessels fill the full
    # gray range instead of appearing near-black (raw values are typically 0–30).
    p_hi = np.percentile(wth_raw, 99.5)
    wth  = np.clip(wth_raw / max(p_hi, 1e-6), 0.0, 1.0)

    # Distance transform (iterative erosion proxy): threshold at Otsu, then EDT
    _, bw = cv2.threshold(g_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist  = distance_transform_edt(bw > 0).astype(np.float32)
    if dist.max() > 0:
        dist /= dist.max()

    return wth, dist


def fig9_morpho_channels():
    images_info = [
        ('082_N', 'Normal (082_N)'),
        ('084_D', 'Diabétique (084_D)'),
        ('083_G', 'Glaucome (083_G)'),
    ]
    col_titles = ["Image RGB (canal vert)", "Top-hat blanc", "Transformée en distance"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.12, wspace=0.06)

    for row, (stem, row_title) in enumerate(images_info):
        img_path = BASE / 'Fundus-AVSeg-prep-data' / 'test' / 'images' / f'{stem}.png'
        img_np = np.array(Image.open(img_path))
        wth, dist = _compute_morpho(img_np)
        g = img_np[:, :, 1].astype(np.float32) / 255.0

        panels = [img_np, wth, dist]
        cmaps  = [None, 'gray', 'viridis']

        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            ax.imshow(panel, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9.5, fontweight='bold', labelpad=6)

    plt.suptitle('Canaux morphologiques — exemples de pathologies différentes',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.savefig(OUT / 'morpho_channels_overview.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print('✓ Fig 9: morpho_channels_overview.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=== Generating PFE report figures ===')
    fig3_timeline()
    fig4_bifpn_arch()
    fig5_binary_pipeline()
    fig6_ensemble_pipeline()
    fig7_binary_qualitative()
    fig8_av_qualitative()
    fig9_morpho_channels()
    print(f'\nAll figures saved to {OUT}')
