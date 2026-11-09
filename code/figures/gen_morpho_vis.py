"""
Task 3.3 – Morphological-channel visualisation.
Produces paper_fixes/figures/morpho_channels_{image_id}.png for 3 test images.

Each figure shows a 3×3 grid:
  Row 0: [Original RGB] [White-tophat R] [Dist-transform R]
  Row 1: [Green ch]     [White-tophat G] [Dist-transform G]
  Row 2: [Blue ch]      [White-tophat B] [Dist-transform B]
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import MorphoPreprocessor, NEW_SIZE, device

BASE    = Path('/home/mefeki/Bureau/work')
TEST_DIR = BASE / 'Fundus-AVSeg-prep-data' / 'test' / 'images'
FIG_DIR  = BASE / 'paper_fixes' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Pick 3 representative images: normal, diabetic, glaucoma
IMAGES = ['082_N.png', '084_D.png', '083_G.png']

prep = MorphoPreprocessor(3).to(device)
prep.eval()

for img_name in IMAGES:
    img_path = TEST_DIR / img_name
    stem = img_path.stem

    # Load and resize
    pil = Image.open(img_path).convert('RGB').resize((NEW_SIZE, NEW_SIZE), Image.BILINEAR)
    img_np = np.array(pil).astype(np.float32) / 255.0   # H W 3

    t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1 3 H W

    with torch.no_grad():
        th_t, dist_t = prep(t)

    img_rgb = img_np                                         # H W 3  (0–1)
    th   = th_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # H W 3
    dist = dist_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) # H W 3

    # Normalise each channel map independently for display
    def norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-8)

    fig = plt.figure(figsize=(12, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.05, wspace=0.05)

    ch_names = ['Red', 'Green', 'Blue']
    col_titles = ['Original channel', 'White top-hat', 'Dist. transform']

    for row, (ch, ch_name) in enumerate(zip(range(3), ch_names)):
        # Col 0: original single channel (greyscale)
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img_rgb[:, :, ch], cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel(ch_name, fontsize=11, rotation=90, va='center', labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[0], fontsize=11)

        # Col 1: white top-hat
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(norm(th[:, :, ch]), cmap='hot', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[1], fontsize=11)

        # Col 2: distance transform
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(norm(dist[:, :, ch]), cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[2], fontsize=11)

    fig.suptitle(f'Morphological features — {stem}', fontsize=13, y=0.995)
    out_path = FIG_DIR / f'morpho_channels_{stem}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

# Also produce a single-image overview panel (all 3 images side-by-side, row per image)
# One row per image: [RGB | tophat R | dist R] — compact version for paper inclusion
fig2, axes = plt.subplots(3, 3, figsize=(12, 12))
fig2.subplots_adjust(hspace=0.04, wspace=0.04)

def norm(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)

row_labels = ['Normal (N)', 'Diabetic (D)', 'Glaucoma (G)']
col_labels2 = ['RGB', 'White top-hat (green ch)', 'Dist. transform (green ch)']

for row_i, img_name in enumerate(['082_N.png', '084_D.png', '083_G.png']):
    img_path = TEST_DIR / img_name
    pil = Image.open(img_path).convert('RGB').resize((NEW_SIZE, NEW_SIZE), Image.BILINEAR)
    img_np = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        th_t, dist_t = prep(t)
    th   = th_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    dist = dist_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    ch = 1  # green channel for compact view

    axes[row_i, 0].imshow(img_np)
    axes[row_i, 1].imshow(norm(th[:, :, ch]), cmap='hot')
    axes[row_i, 2].imshow(norm(dist[:, :, ch]), cmap='viridis')
    for c in range(3):
        axes[row_i, c].set_xticks([]); axes[row_i, c].set_yticks([])
    axes[row_i, 0].set_ylabel(row_labels[row_i], fontsize=11, rotation=90, va='center', labelpad=4)

for c, title in enumerate(col_labels2):
    axes[0, c].set_title(title, fontsize=11)

fig2.suptitle('Morphological preprocessing features (green channel)', fontsize=12, y=1.001)
overview_path = FIG_DIR / 'morpho_channels_overview.png'
fig2.savefig(overview_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved overview: {overview_path}')
print('Done.')
