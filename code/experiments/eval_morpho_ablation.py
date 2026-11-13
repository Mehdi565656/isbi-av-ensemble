"""
Evaluate the 3 morpho-ablation checkpoints on the 20-image test set.
Updates the TBD rows in 03_morpho_ablation.md and saves a CSV.
Run after train_morpho_ablation.py completes.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data as tdata
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device,
                            MorphoPreprocessor)
from train_multiseed import TestDataset
from train_morpho_ablation import MorphoUNetTopHatOnly, MorphoUNetDistOnly, MorphoUNetBlackTopHat

BASE    = Path('/path/to/workspace')
OUT_DIR = BASE / 'paper_fixes' / 'models' / 'morpho_ablation'
TEST_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'test'

VARIANTS = {
    'morpho_tophat_only':        MorphoUNetTopHatOnly,
    'morpho_dist_only':          MorphoUNetDistOnly,
    'morpho_black_tophat_dist':  MorphoUNetBlackTopHat,
}

def compute_per_image_metrics(model, test_ds):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, lbl_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            lbl_t = lbl_t.to(device)
            logits = model(img_t)
            preds = torch.argmax(torch.softmax(logits, 1), 1).squeeze(0)
            row = {'image_id': test_ds.image_paths[idx].stem}
            vessel_dice_sum = 0.
            for c in range(NUM_CLASSES):
                p = (preds == c).float(); l = (lbl_t == c).float()
                tp = float((p * l).sum())
                fp = float((p * (1 - l)).sum())
                fn = float(((1 - p) * l).sum())
                dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
                iou  = tp / (tp + fp + fn + 1e-6)
                row[f'dice_c{c}'] = dice; row[f'iou_c{c}'] = iou
                if c in VESSEL_CLASSES:
                    vessel_dice_sum += dice
            row['vessel_dice'] = vessel_dice_sum / len(VESSEL_CLASSES)
            row['vessel_iou']  = (row['iou_c1'] + row['iou_c2']) / 2
            rows.append(row)
    model.train()
    return pd.DataFrame(rows)

test_ds = TestDataset(TEST_ROOT)
results = []

for name, cls in VARIANTS.items():
    ckpt_path = OUT_DIR / f'{name}.pth'
    if not ckpt_path.exists():
        print(f'  [SKIP] {name}: checkpoint not found at {ckpt_path}')
        continue
    print(f'  Evaluating {name} ...')
    m = cls(in_channels=3, n_classes=3).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    df = compute_per_image_metrics(m, test_ds)
    df['model'] = name
    results.append(df)

    mean_vdice = df['vessel_dice'].mean()
    mean_viou  = df['vessel_iou'].mean()
    mean_adice = df['dice_c1'].mean()
    mean_wdice = df['dice_c2'].mean()
    print(f'    Vessel Dice={mean_vdice:.4f}  IoU={mean_viou:.4f}  '
          f'Artery={mean_adice:.4f}  Vein={mean_wdice:.4f}')

if results:
    out_df = pd.concat(results, ignore_index=True)
    out_df.to_csv(OUT_DIR / 'ablation_per_image_scores.csv', index=False)

    # Summary table
    summary = []
    for df in results:
        m = df['model'].iloc[0]
        summary.append({
            'model':        m,
            'vessel_dice':  df['vessel_dice'].mean(),
            'vessel_iou':   df['vessel_iou'].mean(),
            'artery_dice':  df['dice_c1'].mean(),
            'vein_dice':    df['dice_c2'].mean(),
        })
    sumdf = pd.DataFrame(summary)
    sumdf.to_csv(OUT_DIR / 'ablation_summary.csv', index=False)
    print('\nSummary:')
    print(sumdf.to_string(index=False, float_format='%.4f'))
    print(f'\nSaved to {OUT_DIR}/ablation_summary.csv')
else:
    print('No checkpoints found. Run train_morpho_ablation.py first.')
