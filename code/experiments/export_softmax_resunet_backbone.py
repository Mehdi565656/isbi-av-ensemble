"""
Export per-pixel softmax maps for the two ResUNet-backbone models that were
missing from `results/softmax_maps/`: BiFPN-ResUNet and Morpho-ResUNet.

`run_inference.py` cached maps for the six U-Net-family models only, so the
ResUNet-backbone ensembles of Sec. 3.4 could not be rebuilt from cache. This
script fills that gap, writing float16 maps in exactly the same layout and
dtype as `run_inference.py`:

    results/softmax_maps/<model>/<image_stem>.npy     shape (3, 592, 592), float16

It also writes per-image Dice/IoU for the two models so the numbers can be
checked against `results/metrics/bifpn_resunet_scores.csv` and
`morpho_resunet_scores.csv`, which were produced by the original evaluation.

Usage
-----
    python export_softmax_resunet_backbone.py              # both models
    python export_softmax_resunet_backbone.py --models bifpn_resunet
    python export_softmax_resunet_backbone.py --force      # overwrite existing maps
    python export_softmax_resunet_backbone.py --device cpu # when the GPU is busy

Paths are resolved relative to this repository, not to the original
/home/mefeki/Bureau/work workspace. Override with --repo if needed.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Architectures are reused verbatim from the scripts that trained / evaluated
# them, so the weights load into exactly the modules they were saved from.
from run_inference import NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE  # noqa: E402
from eval_bifpn_resunet import BiFPNResUNet, BiFPNResUNetDual  # noqa: E402
from train_morpho_resunet import MorphoResUNet  # noqa: E402
from train_multiseed import TestDataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Set in main(); the softmax values are device-independent, so CPU is a valid
# fallback when the GPU is occupied (it is simply slower).
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name -> (checkpoint path relative to REPO, constructor)
MODELS = {
    'bifpn_resunet': (
        'models/resunet_backbone/bifpn_resunet.pth',
        lambda: BiFPNResUNet(in_channels=3, num_classes=3, base_ch=64, depth=5,
                             bifpn_ch=64, num_bifpn_layers=1),
    ),
    'morpho_resunet': (
        'models/resunet_backbone/morpho_resunet.pth',
        lambda: MorphoResUNet(in_channels=3, n_classes=3),
    ),
    # Trained but not reported in the paper; exported on request only.
    'bifpn_resunet_dualskip': (
        'models/resunet_backbone/bifpn_resunet_dualskip.pth',
        lambda: BiFPNResUNetDual(in_channels=3, num_classes=3, base_ch=64, depth=5,
                                 bifpn_ch=64, num_bifpn_layers=1),
    ),
}

DEFAULT_MODELS = ['bifpn_resunet', 'morpho_resunet']


def dice_iou_per_class(preds, labels, n=NUM_CLASSES, eps=1e-6):
    """Identical to run_inference.dice_iou_per_class (same eps, same reduction)."""
    dice, iou = {}, {}
    for c in range(n):
        p = (preds == c).float()
        l = (labels == c).float()
        tp = float((p * l).sum())
        fp = float((p * (1 - l)).sum())
        fn = float(((1 - p) * l).sum())
        dice[c] = 2 * tp / (2 * tp + fp + fn + eps)
        iou[c] = tp / (tp + fp + fn + eps)
    return dice, iou


def export(name, test_root, out_root, force=False):
    """Write this model's softmax maps and return its per-image Dice/IoU.

    If every map is already cached and --force was not given, the metrics are
    recomputed from the cached maps and the checkpoint is never loaded — so a
    repeat run is cheap and needs no GPU.
    """
    ds = TestDataset(test_root)
    stems = [p.stem for p in ds.image_paths]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = all((out_dir / f'{stem}.npy').exists() for stem in stems)
    model = None
    if force or not cached:
        ckpt_rel, build = MODELS[name]
        ckpt = REPO / ckpt_rel
        if not ckpt.exists():
            raise FileNotFoundError(f'checkpoint not found: {ckpt}')
        model = build().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        print(f'\n{name}: loaded {ckpt.name} '
              f'({sum(p.numel() for p in model.parameters()):,} params)')
    else:
        print(f'\n{name}: {len(stems)} maps already cached, scoring from cache '
              f'(use --force to recompute)')

    rows = []
    with torch.no_grad():
        for idx, stem in enumerate(stems):
            dest = out_dir / f'{stem}.npy'
            _, lbl_t = ds[idx]

            if model is None:
                probs_np = np.load(dest).astype(np.float32)
            else:
                img_t, _ = ds[idx]
                probs = torch.softmax(model(img_t.unsqueeze(0).to(DEVICE)), 1)
                probs_np = probs[0].cpu().numpy()
                np.save(dest, probs_np.astype(np.float16))

            preds = torch.from_numpy(np.argmax(probs_np, axis=0))
            dice, iou = dice_iou_per_class(preds, lbl_t)
            rows.append({
                'image_id': stem,
                'dice_c0': dice[0], 'iou_c0': iou[0],
                'dice_c1': dice[1], 'iou_c1': iou[1],
                'dice_c2': dice[2], 'iou_c2': iou[2],
                'vessel_dice': float(np.mean([dice[c] for c in VESSEL_CLASSES])),
                'vessel_iou': float(np.mean([iou[c] for c in VESSEL_CLASSES])),
                'model': name,
            })

    df = pd.DataFrame(rows)
    print(f'  {len(df)} maps in {out_dir.relative_to(REPO)}')
    print(f'  Vessel Dice={df["vessel_dice"].mean():.4f}  IoU={df["vessel_iou"].mean():.4f}'
          f'  Artery={df["dice_c1"].mean():.4f}  Vein={df["dice_c2"].mean():.4f}')
    return df


def main():
    global REPO, DEVICE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', type=Path, default=REPO,
                    help='repository root (default: inferred from this file)')
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=list(MODELS),
                    help='which models to export (default: bifpn_resunet morpho_resunet)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite softmax maps that already exist')
    ap.add_argument('--device', default=None,
                    help="torch device, e.g. cuda or cpu (default: cuda when available)")
    args = ap.parse_args()

    REPO = args.repo.resolve()
    if args.device:
        DEVICE = torch.device(args.device)
    test_root = REPO / 'data' / 'fundus-avseg-592' / 'test'
    out_root = REPO / 'results' / 'softmax_maps'
    metrics_dir = REPO / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Repo:   {REPO}')
    print(f'Test:   {test_root}  ({NEW_SIZE}x{NEW_SIZE})')

    dfs = [export(name, test_root, out_root, force=args.force) for name in args.models]

    out = metrics_dir / 'resunet_backbone_per_image_scores.csv'
    fresh = pd.concat(dfs, ignore_index=True)
    if out.exists():
        # keep rows for models this run did not touch
        prev = pd.read_csv(out)
        prev = prev[~prev['model'].isin(fresh['model'].unique())]
        fresh = pd.concat([prev, fresh], ignore_index=True)
    fresh = fresh.sort_values(['model', 'image_id'], ignore_index=True)
    fresh.to_csv(out, index=False)
    print(f'\nPer-image scores -> {out.relative_to(REPO)}')
    print('Next: python build_resunet_backbone_ensembles.py')


if __name__ == '__main__':
    main()
