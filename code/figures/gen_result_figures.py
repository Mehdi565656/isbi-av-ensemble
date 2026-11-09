"""
Generate result figures for the paper:
  (a) per_model_dice_boxplot.png  — box plot of per-image vessel Dice for all 6 models + ensemble
  (b) ensemble_comparison_bar.png — bar chart: single models vs ensemble Dice mean ± std
  (c) artery_vein_comparison.png  — grouped bar: artery vs vein Dice per model
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR  = Path('/home/mefeki/Bureau/work/paper_fixes/figures')
DATA_DIR = Path('/home/mefeki/Bureau/work/paper_fixes/data')
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'per_image_scores.csv')

# ── Model display order and labels ─────────────────────────────────────────────
MODELS_ORDER = ['bifpn_unet', 'morphobifpn', 'bifpn_64to1024',
                'unet', 'morpho_unet', 'resunet', 'ensemble_top3']
MODEL_LABELS = {
    'bifpn_unet':     'BiFPN-UNet\n(32ch)',
    'morphobifpn':    'MorphoBiFPN\n(32ch)',
    'bifpn_64to1024': 'BiFPN-UNet\n(64ch)',
    'unet':           'U-Net',
    'morpho_unet':    'Morpho-\nU-Net',
    'resunet':        'ResUNet',
    'ensemble_top3':  'Ensemble\n(top-3)',
}
COLORS = {
    'bifpn_unet':     '#4292c6',
    'morphobifpn':    '#2171b5',
    'bifpn_64to1024': '#084594',
    'unet':           '#74c476',
    'morpho_unet':    '#238b45',
    'resunet':        '#005a32',
    'ensemble_top3':  '#d94801',
}

# Vessel Dice per image per model (c1 = artery, c2 = vein)
vessel = (
    df[df['class'].isin([1, 2])]
    .groupby(['image_id', 'model'])['dice']
    .mean()
    .reset_index()
    .rename(columns={'dice': 'vessel_dice'})
)

# ── (a) Box plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
box_data = []
labels   = []
colors   = []
for m in MODELS_ORDER:
    if m not in df['model'].unique(): continue
    sub = vessel[vessel['model'] == m]['vessel_dice'].values
    box_data.append(sub)
    labels.append(MODEL_LABELS[m])
    colors.append(COLORS[m])

bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o', markersize=4, alpha=0.6))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.85)

ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Vessel Dice (mean of artery + vein Dice)', fontsize=11)
ax.set_ylim(0.65, 0.92)
ax.axhline(0.8320, color='#d94801', linestyle='--', linewidth=1.2, alpha=0.7,
           label='Ensemble mean (0.832)')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9)
ax.set_title('Per-image vessel Dice — all models (20 test images)', fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / 'per_model_dice_boxplot.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved per_model_dice_boxplot.png')

# ── (b) Bar chart: mean ± std ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(MODELS_ORDER))
means, stds, bar_colors, bar_labels = [], [], [], []
for m in MODELS_ORDER:
    if m not in df['model'].unique(): continue
    sub = vessel[vessel['model'] == m]['vessel_dice'].values
    means.append(sub.mean()); stds.append(sub.std(ddof=1))
    bar_colors.append(COLORS[m]); bar_labels.append(MODEL_LABELS[m])

bars = ax.bar(x_pos[:len(means)], means, yerr=stds, capsize=4,
              color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.88,
              error_kw=dict(elinewidth=1.5, ecolor='#333333'))
ax.set_xticks(x_pos[:len(means)])
ax.set_xticklabels(bar_labels, fontsize=10)
ax.set_ylabel('Mean vessel Dice ± std', fontsize=11)
ax.set_ylim(0.72, 0.87)
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.003,
            f'{mean:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_title('Mean vessel Dice ± std (20 test images)', fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / 'ensemble_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved ensemble_comparison_bar.png')

# ── (c) Artery vs vein grouped bar ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
artery_means, vein_means, gbar_labels = [], [], []
for m in MODELS_ORDER:
    if m not in df['model'].unique(): continue
    a = df[(df['model']==m) & (df['class']==1)].groupby('image_id')['dice'].mean().mean()
    v = df[(df['model']==m) & (df['class']==2)].groupby('image_id')['dice'].mean().mean()
    artery_means.append(a); vein_means.append(v)
    gbar_labels.append(MODEL_LABELS[m])

x = np.arange(len(gbar_labels))
width = 0.35
b1 = ax.bar(x - width/2, artery_means, width, label='Artery', color='#e41a1c', alpha=0.85, edgecolor='white')
b2 = ax.bar(x + width/2, vein_means,   width, label='Vein',   color='#377eb8', alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(gbar_labels, fontsize=10)
ax.set_ylabel('Mean Dice', fontsize=11)
ax.set_ylim(0.72, 0.88)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(list(b1) + list(b2), artery_means + vein_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Artery vs vein Dice by model (20 test images)', fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / 'artery_vein_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved artery_vein_comparison.png')

# ── Summary CSV for reference ──────────────────────────────────────────────────
summary_rows = []
for m in MODELS_ORDER:
    if m not in df['model'].unique(): continue
    a = df[(df['model']==m) & (df['class']==1)]['dice'].mean()
    v = df[(df['model']==m) & (df['class']==2)]['dice'].mean()
    ai = df[(df['model']==m) & (df['class']==1)]['iou'].mean()
    vi = df[(df['model']==m) & (df['class']==2)]['iou'].mean()
    vd = vessel[vessel['model']==m]['vessel_dice'].mean()
    summary_rows.append({
        'model': m, 'vessel_dice': vd, 'vessel_iou': (ai+vi)/2,
        'artery_dice': a, 'vein_dice': v, 'artery_iou': ai, 'vein_iou': vi,
    })
sumdf = pd.DataFrame(summary_rows)
sumdf.to_csv(DATA_DIR / 'model_summary.csv', index=False)
print('\nModel summary:')
print(sumdf.to_string(index=False, float_format='%.4f'))
