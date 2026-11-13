"""
Build the soft-voting ensembles of Table 1 / Sec. 3.4 from cached softmax maps,
and test the ResUNet-backbone ensembles against the A+B+C baseline.

Members, as named in the paper:
    A = BiFPN-UNet (64-1024ch)    softmax_maps/bifpn_64to1024
    B = U-Net (baseline)          softmax_maps/unet
    C = Morpho-U-Net              softmax_maps/morpho_unet

Note: the paper's "BiFPN-ResUNet" is the dual-skip checkpoint -- see the comment
on ENSEMBLES below. A sixth, non-paper ensemble using the plain-neck checkpoint
is built alongside it for comparison.

Ensembles built (Eq. 2: mean of the per-pixel softmax vectors, then argmax):
    A+B                     two-member control
    A+B+C                   the heterogeneous ensemble of Sec. 3.2
    A+B+ResUNet             | the three Sec. 3.4 substitutions, which replace
    A+B+Morpho-ResUNet      | the weakest member (C) with a model built on the
    A+B+BiFPN-ResUNet       | stronger ResUNet backbone

Statistics (Sec. 3.4): each ResUNet-backbone ensemble is compared with A+B+C by
a one-tailed Wilcoxon signed-rank test paired over the 20 test images, with
Holm correction across the three comparisons. Bootstrap 95% CIs resample test
images with the same percentile method, default resample count (1000) and RNG
seed (0) as compute_statistics.py, so CIs here are comparable with Table 2's.
Holm correction itself is new here: compute_statistics.py reports raw p-values
only, and the paper's Holm adjustment was applied outside the script.

The two ResUNet-backbone models need their softmax maps exported first:
    python export_softmax_resunet_backbone.py

Usage
-----
    python build_resunet_backbone_ensembles.py
    python build_resunet_backbone_ensembles.py --bootstrap 10000 --seed 1

Outputs (under results/metrics/):
    ensemble_composition_per_image.csv   per-image Dice/IoU for every ensemble
    ensemble_composition_summary.csv     aggregate Dice/IoU + bootstrap CI
    ensemble_composition_tests.csv       Wilcoxon vs A+B+C, raw and Holm-adjusted
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Ensemble name -> softmax_maps subdirectories to average, in paper order.
#
# On "BiFPN-ResUNet": two ResUNet+BiFPN checkpoints exist. The plain one
# (BiFPNResUNet, models/resunet_backbone/bifpn_resunet.pth) scores 0.8049 Dice
# alone; the dual-skip one (BiFPNResUNetDual, bifpn_resunet_dualskip.pth, which
# also feeds the raw encoder features to the decoder) scores 0.8147 / 0.6889 --
# exactly the "BiFPN-ResUNet" row of Table 1. The paper's BiFPN-ResUNet is
# therefore the dual-skip model, even though Sec. 2 describes only the plain
# neck insertion. Both are built here so the two can be compared.
ENSEMBLES = {
    'A+B':                       ['bifpn_64to1024', 'unet'],
    'A+B+C':                     ['bifpn_64to1024', 'unet', 'morpho_unet'],
    'A+B+ResUNet':               ['bifpn_64to1024', 'unet', 'resunet'],
    'A+B+Morpho-ResUNet':        ['bifpn_64to1024', 'unet', 'morpho_resunet'],
    'A+B+BiFPN-ResUNet':         ['bifpn_64to1024', 'unet', 'bifpn_resunet_dualskip'],
    'A+B+BiFPN-ResUNet (plain)': ['bifpn_64to1024', 'unet', 'bifpn_resunet'],
}

# Compared against A+B+C in Sec. 3.4, in the order the paper reports them.
BASELINE = 'A+B+C'
TESTED = ['A+B+BiFPN-ResUNet', 'A+B+Morpho-ResUNet', 'A+B+ResUNet']

# Table 1 values, for the self-check printed at the end. The plain-neck
# ensemble is not in the paper and so has no expected value.
EXPECTED = {
    'A+B':                (0.8296, 0.7106),
    'A+B+C':              (0.8320, 0.7139),
    'A+B+ResUNet':        (0.8350, 0.7184),
    'A+B+Morpho-ResUNet': (0.8364, 0.7203),
    'A+B+BiFPN-ResUNet':  (0.8380, 0.7228),
}


def load_labels(test_root, size=NEW_SIZE):
    """Ground-truth class masks, keyed by image stem (same resize as training)."""
    labels = {}
    for p in sorted((Path(test_root) / 'images').glob('*.png')):
        ann = Path(test_root) / 'annotation' / f'{p.stem}.png'
        rgb = np.array(Image.open(ann).convert('RGB').resize((size, size), Image.NEAREST))
        labels[p.stem] = rgb_to_class_mask(rgb)
    return labels


def dice_iou_numpy(pred, label, n=NUM_CLASSES, eps=1e-6):
    """Per-class Dice/IoU. Same formula and eps as run_inference.dice_iou_per_class."""
    dice, iou = {}, {}
    for c in range(n):
        p = (pred == c)
        l = (label == c)
        tp = float(np.sum(p & l))
        fp = float(np.sum(p & ~l))
        fn = float(np.sum(~p & l))
        dice[c] = 2 * tp / (2 * tp + fp + fn + eps)
        iou[c] = tp / (tp + fp + fn + eps)
    return dice, iou


def ensemble_predict(softmax_root, members, stem):
    """Eq. 2: average the members' softmax vectors, then argmax over classes.

    Maps are stored float16; they are summed in float32 to avoid the precision
    loss of accumulating in half.
    """
    acc = None
    for m in members:
        path = Path(softmax_root) / m / f'{stem}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'missing softmax map: {path}\n'
                f'  -> run: python export_softmax_resunet_backbone.py')
        p = np.load(path).astype(np.float32)
        acc = p if acc is None else acc + p
    return np.argmax(acc / len(members), axis=0)


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    """Percentile CI from resampling test images, as in compute_statistics.py."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment, order preserved."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(pvalues) - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    global REPO
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', type=Path, default=REPO,
                    help='repository root (default: inferred from this file)')
    ap.add_argument('--bootstrap', type=int, default=1000,
                    help='bootstrap resamples for the 95%% CI (default: 1000, as in '
                         'compute_statistics.py)')
    ap.add_argument('--seed', type=int, default=0,
                    help='bootstrap RNG seed (default: 0, as in compute_statistics.py)')
    args = ap.parse_args()

    REPO = args.repo.resolve()
    test_root = REPO / 'data' / 'fundus-avseg-592' / 'test'
    softmax_root = REPO / 'results' / 'softmax_maps'
    metrics_dir = REPO / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(test_root)
    stems = sorted(labels)
    print(f'Repo: {REPO}')
    print(f'Test images: {len(stems)}')

    # ── Per-image metrics for every ensemble ──────────────────────────────────
    rows = []
    for name, members in ENSEMBLES.items():
        for stem in stems:
            pred = ensemble_predict(softmax_root, members, stem)
            dice, iou = dice_iou_numpy(pred, labels[stem])
            rows.append({
                'ensemble': name,
                'members': '+'.join(members),
                'image_id': stem,
                'dice_c0': dice[0], 'dice_c1': dice[1], 'dice_c2': dice[2],
                'iou_c0': iou[0], 'iou_c1': iou[1], 'iou_c2': iou[2],
                'vessel_dice': float(np.mean([dice[c] for c in VESSEL_CLASSES])),
                'vessel_iou': float(np.mean([iou[c] for c in VESSEL_CLASSES])),
            })
    per_image = pd.DataFrame(rows)
    per_image.to_csv(metrics_dir / 'ensemble_composition_per_image.csv', index=False)

    # ── Aggregate + bootstrap CI ──────────────────────────────────────────────
    dice_by_ens = {n: per_image.loc[per_image.ensemble == n, 'vessel_dice'].to_numpy()
                   for n in ENSEMBLES}

    summary = []
    print(f'\n{"Ensemble":26s} {"Dice":>7s} {"IoU":>7s}   {"95% CI (Dice)":>18s}   paper')
    for name in ENSEMBLES:
        sub = per_image[per_image.ensemble == name]
        d = float(sub['vessel_dice'].mean())
        i = float(sub['vessel_iou'].mean())
        lo, hi = bootstrap_ci(dice_by_ens[name], args.bootstrap, args.seed)
        exp_d, exp_i = EXPECTED.get(name, (float('nan'), float('nan')))
        summary.append({
            'ensemble': name, 'members': '+'.join(ENSEMBLES[name]),
            'n_members': len(ENSEMBLES[name]),
            'vessel_dice': d, 'vessel_iou': i,
            'artery_dice': float(sub['dice_c1'].mean()),
            'vein_dice': float(sub['dice_c2'].mean()),
            'ci_lo': lo, 'ci_hi': hi,
            'paper_dice': exp_d, 'paper_iou': exp_i,
            'delta_dice': d - exp_d,
        })
        paper_col = ('     --  (not in paper)' if np.isnan(exp_d)
                     else f'{exp_d:.4f}/{exp_i:.4f}  (Δ{d - exp_d:+.4f})')
        print(f'{name:26s} {d:7.4f} {i:7.4f}   [{lo:.4f}, {hi:.4f}]   {paper_col}')
    pd.DataFrame(summary).to_csv(metrics_dir / 'ensemble_composition_summary.csv', index=False)

    # ── Wilcoxon vs A+B+C, Holm-corrected ─────────────────────────────────────
    base = dice_by_ens[BASELINE]
    raw_p, tests = [], []
    for name in TESTED:
        cand = dice_by_ens[name]
        # one-tailed: is the ResUNet-backbone ensemble better than A+B+C?
        stat, p = wilcoxon(cand, base, alternative='greater')
        raw_p.append(p)
        tests.append({
            'ensemble': name, 'baseline': BASELINE, 'n_images': len(base),
            'mean_dice': float(cand.mean()), 'baseline_mean_dice': float(base.mean()),
            'mean_diff': float((cand - base).mean()),
            'n_wins': int(np.sum(cand > base)),
            'n_losses': int(np.sum(cand < base)),
            'wilcoxon_stat': float(stat), 'p_raw': float(p),
        })
    for row, p_adj in zip(tests, holm(np.array(raw_p))):
        row['p_holm'] = float(p_adj)

    print(f'\nOne-tailed Wilcoxon signed-rank vs {BASELINE} (n={len(base)}), Holm-corrected:')
    print(f'{"Ensemble":26s} {"ΔDice":>8s} {"wins":>6s} {"p_raw":>10s} {"p_holm":>10s}')
    for row in tests:
        print(f'{row["ensemble"]:26s} {row["mean_diff"]:+8.4f} '
              f'{row["n_wins"]:3d}/{row["n_images"]:<2d} '
              f'{row["p_raw"]:10.2e} {row["p_holm"]:10.2e}')
    pd.DataFrame(tests).to_csv(metrics_dir / 'ensemble_composition_tests.csv', index=False)

    # ── Self-check against Table 1 ────────────────────────────────────────────
    worst = max(abs(r['delta_dice']) for r in summary
                if not np.isnan(r['delta_dice']))
    print(f'\nLargest deviation from Table 1 Dice: {worst:.5f}', end='  ')
    print('(matches)' if worst < 5e-4 else '(DOES NOT MATCH — investigate)')
    print(f'Wrote 3 CSVs to {metrics_dir.relative_to(REPO)}/')


if __name__ == '__main__':
    main()
