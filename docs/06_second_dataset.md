# Task 6 — Second-Dataset Results Write-Up

## 6.1 Dataset Description

**Dataset:** GAVE2 FFA — MICCAI 2026 Challenge on Generalised Artery-Vein Extraction from Fundus Fluorescein Angiography (FFA) images.

**Important caveat on comparability:** FFA images have fundamentally different appearance from colour fundus photographs (greyscale/grey-green vessel contrast, no optic disc colour, high background noise). This is a cross-modality evaluation, not a same-modality replication. Results on GAVE2 should be framed as a generalisability check (does the approach transfer to FFA?) rather than a direct benchmark comparison.

**Split:** 50 training images (preliminary set). 40/10 train/val split (random seed=42). A separate challenge test set (images g_051+) has no local ground truth.

**Protocol:** FOV mask used to crop the image to the retinal disc bounding box, then bilinear-resize to 592×592. Class labels: artery=red(255,0,0), vein=blue(0,0,255). Same 3-class setup (background/artery/vein) as Fundus-AVSeg.

## 6.2 Results Table (Validation Set, 10 images)

Models trained under the same fixed-592 protocol as Fundus-AVSeg (same loss, class weights, augmentation; seed=42; evaluated every 5 epochs; best-val-Dice checkpoint saved):

| Model | Val vessel Dice | Val artery Dice | Val vein Dice |
|---|---|---|---|
| U-Net | 0.629 | — | — |
| ResUNet | 0.638 | — | — |
| BiFPN-Dual | **0.645** | — | — |
| Ensemble (all 3) | *not evaluated* | — | — |

Only aggregate val Dice is available (from training history CSVs). Per-class breakdown and per-image scores were not saved. Ensemble was not evaluated on this dataset.

**Note on "BiFPN-Dual":** This is a BiFPN-UNet variant with dual skip connections (see `MICAI/models/best_model_micai_bifpn_dual.pth`), not the plain BiFPN-UNet used in the main paper. Architecturally closer to the BiFPN-64ch variant.

## 6.3 What is Missing vs Main Protocol

| Component | Fundus-AVSeg | GAVE2 FFA |
|---|---|---|
| Standard U-Net | ✓ | ✓ |
| ResUNet | ✓ | ✓ |
| Morpho-U-Net | ✓ | ✗ missing |
| BiFPN-UNet | ✓ | ✓ (dual variant) |
| MorphoBiFPN-UNet | ✓ | ✗ missing |
| Soft-voting ensemble | ✓ (Dice 0.8320) | ✗ missing |
| Per-image metrics | ✓ | ✗ aggregate only |
| Multi-seed | ✗ single seed | ✗ single seed |
| Test set labels | ✓ (20 images) | ✗ no local GT |

## 6.4 Pattern Comparison

| Observation | Fundus-AVSeg | GAVE2 FFA |
|---|---|---|
| Best single model | BiFPN-64ch or ResUNet (0.81) | BiFPN-Dual or ResUNet (0.64) |
| U-Net baseline | 0.8023 | 0.629 |
| ResUNet vs U-Net | ResUNet marginally better (+0.008) | ResUNet marginally better (+0.009) |
| BiFPN vs U-Net | BiFPN-64ch marginally better (+0.007) | BiFPN-Dual marginally better (+0.016) |

The ranking pattern is consistent: BiFPN and ResUNet both outperform the plain U-Net baseline by a small margin on both datasets. Absolute Dice scores are ~19 points lower on GAVE2 FFA, which is expected given:
- FFA images have noisier vessel appearance
- Only 40 training images (vs ~70 for Fundus-AVSeg)
- No Morpho-U-Net trained, ensemble not evaluated

## 6.5 How to Present in Paper

Present as a **lightweight generalisability check**, not a full replication. Appropriate framing:
- "We additionally applied three models (U-Net, ResUNet, BiFPN) to the GAVE2 FFA dataset under the same fixed-592 protocol to assess cross-modality transferability."
- Report the val Dice numbers with the caveat that FFA is a different imaging modality.
- State clearly that the ensemble was not evaluated on GAVE2 due to compute budget, and that the Morpho-U-Net was not included (morphological preprocessing parameters were tuned for colour fundus images).
- Do not claim the GAVE2 numbers support the ensemble's superiority (the ensemble was not run there).
