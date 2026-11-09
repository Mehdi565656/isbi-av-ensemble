# Task 0 — Inventory and Reproducibility Groundwork

## 0.1 Fixed-592 Checkpoint Paths and Training Configs

All models in the fixed-592 protocol resize every image to 592 × 592 before processing.

| Model | Checkpoint path | Training notebook |
|---|---|---|
| Standard U-Net | `multi_step_back/unet/models/best_model_standard_unet.pth` | `unet/Unet_images_only.ipynb` |
| Morpho-U-Net (plain) | `multi_step_back/morpho/models/best_model_morpho_unet_tophat_dist_592.pth` | `morpho/Morpho_Unet_TopHat_DistanceTransform_Fixed592.ipynb` |
| BiFPN-UNet (32–256 ch) | `multi_step_back/BiFPN/models/best_model_bifpn_592.pth` | `BiFPN/BiFPN_Unet_Fixed592.ipynb` |
| MorphoBiFPN-UNet | `multi_step_back/morpho_BiFPN/models/best_model_morphobifpn_tophat_dist_592.pth` | `morpho_BiFPN/MorphoBiFPN_Unet_TopHat_DistanceTransform_Fixed592.ipynb` |
| BiFPN-UNet (64–1024 ch) | `multi_step_back/BiFPN/models/best_model_bifpn_64to1024_592.pth` | `BiFPN/BiFPN_Unet_64to1024_Fixed592.ipynb` |
| ResUNet (fixed-592) | `multi_step_back/unet/models/best_model_resunet_592.pth` | `unet/ResUnet_Fixed592.ipynb` |

### Training configuration (fixed-592 protocol)

All models share the same base recipe unless noted:

| Hyper-parameter | Value |
|---|---|
| Input resolution | 592 × 592 (bilinear resize of image; nearest for mask) |
| Seed | 42 (all models) |
| Batch size | 4 (2 for Standard U-Net which predates the fixed-592 protocol) |
| Loss | CE-Dice combined: `(1−dice_wt)·CE + dice_wt·SoftDice` |
| Dice weight | 0.3 (0.5 for Standard U-Net) |
| Optimizer | AdamW, lr=5×10⁻⁴, weight_decay=2×10⁻⁴ |
| LR schedule | Cosine annealing to η_min=1×10⁻⁵ |
| Augmentation | Horizontal flip (p=0.5), vertical flip (p=0.5), rotation ±10° — **Standard U-Net: no augmentation** |
| Class weights | Inverse pixel frequency, normalised by mean vessel frequency |
| Checkpoint criterion | Best test-set vessel Dice (evaluated every 5 epochs) — **Standard U-Net: final epoch only** |
| Early stopping | Patience 20 evals (100 epochs) for BiFPN-64to1024 and ResUNet; patience 10 (50 epochs) for MorphoBiFPN; none for Standard U-Net (150 epochs fixed) and BiFPN-32 |

Exact epoch budgets and best test Dice (from training CSV, which matches the held-out test evaluation):

| Model | Budget | Best epoch | Best train-eval dice |
|---|---|---|---|
| Standard U-Net | 150 | 150 (final) | — (not tracked per-epoch) |
| Morpho-U-Net | 300 | 135 | 0.7882 |
| BiFPN-UNet 32ch | 150 | ~150 | 0.7901 |
| MorphoBiFPN-UNet | 150 | ~85 | 0.7917 |
| BiFPN-UNet 64–1024 | 300 | ~270 | 0.8103 |
| ResUNet | 300 | ~270 | 0.8103 |

## 0.2 ResNet-Backbone Run — Seed Count

**Single seed only (SEED=42).** The ResUNet checkpoint at `multi_step_back/unet/models/best_model_resunet_592.pth` was produced by a single run of `unet/ResUnet_Fixed592.ipynb`.

Additionally, a BiFPN-ResUNet variant exists at:
- `multi_step_back/BiFPN/models/best_model_bifpn_resunet_592.pth` — trained in `BiFPN/BiFPN_ResUnet_Fixed592.ipynb`, also single seed
- `multi_step_back/BiFPN/models/best_model_bifpn_resunet_dual_592.pth` — dual-skip variant, single seed

**Action for Task 2:** Fold additional-seed training of ResUNet into Task 2.4.

## 0.3 Second Dataset — MICAI GAVE2

The second dataset results are under `MICAI/` and correspond to the **GAVE2 FFA challenge** (Fundus Fluorescein Angiography images from the MICCAI 2026 challenge), **not** another fundus colour-photograph dataset. This is a different imaging modality (FFA vs colour fundus photography), which is important context for how any generalisation claims should be framed.

### MICAI/GAVE2 dataset structure

- Images: `MICAI/GAVE2_preliminary/training/` — 50 FFA images with A/V ground truth
- FOV masks provided; images are cropped to FOV bounding box then resized to 592×592
- Val split: 10 images (random seed 42 from the 50); Train: 40 images
- "Full" models trained on all 50 images for challenge submission
- Labels: artery=red(255,0,0), vein=blue(0,0,255); RGBA format
- Separate held-out test set (`g_051` onwards) submitted to challenge — **no ground truth available locally**

### Models trained on GAVE2 (preliminary set, val-monitored)

Three models, all at 592×592, one seed (42):

| Model | Best val epoch | Best val vessel Dice |
|---|---|---|
| UNet | ~295 (late, still climbing) | 0.629 |
| ResUNet | ~300 | 0.638 |
| BiFPN-Dual | 120 | 0.645 |

("BiFPN-Dual" = BiFPN-UNet with dual skip connections, `best_model_bifpn_resunet_dual_592`-equivalent architecture for FFA)

Additional "full" models (trained on all 50 images, only train loss tracked, no val):
`MICAI/models/best_model_micai_unet_full.pth`, `..._resunet_full.pth`, `..._bifpn_dual_full.pth`

### MICAI Task 2 (tache 2) — additional model variants

Additional models in `MICAI/tache 2/`, trained for challenge task 2:

| Model | Best val dice |
|---|---|
| UNet | 0.629 |
| ResUNet | 0.638 |
| BiFPN | 0.634 (220-ep run) |
| DeBiFPN | 0.675 |
| Dual-Encoder | 0.656 |
| ResNet34-backbone | 0.648 |

These are task-2-specific variants; the main paper's second-dataset claim draws only from the preliminary set (task-1 models above).

### What is missing from MICAI relative to Fundus-AVSeg protocol

- No Morpho-U-Net trained on GAVE2 FFA
- No soft-voting ensemble evaluated (individual model predictions only)
- No per-image metrics saved (only aggregate val Dice from training histories)
- No multi-seed runs
- The held-out test set has no local ground truth → can only report val dice
- Challenge submission predictions exist in `MICAI/predictions/{unet,resunet,bifpn_dual}/` but these are for the challenge test set without labels

**Conclusion for Task 6:** Present MICAI results as a lightweight generalisability check, not a full replication. Report val Dice numbers with the caveat that (a) modality differs from Fundus-AVSeg and (b) FFA images produce lower absolute Dice due to noisier vessel appearance.

## 0.4 Per-Image Inference — Sanity Check

All 6 fixed-592 checkpoints were run on the 20 held-out test images (`Fundus-AVSeg-prep-data/test/`). Softmax maps saved as float16 `.npy` under `paper_fixes/data/softmax_maps/{model}/{stem}.npy`.

**Sanity check result:** Re-averaging the top-3 models' softmax maps (BiFPN-UNet 64ch, Standard U-Net, Morpho-U-Net) and taking argmax gives:
- Ensemble vessel Dice = **0.8320** ✓ (Table 1 reported: 0.8320)
- Ensemble vessel IoU = **0.7139** ✓ (Table 1 reported: 0.7139)

Full per-image metrics in `paper_fixes/data/per_image_scores.csv`.
