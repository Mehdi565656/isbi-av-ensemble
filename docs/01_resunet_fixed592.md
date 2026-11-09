# Task 1 — ResUNet Baseline Under Fixed-592 Protocol

## Architecture

Plain ResUNet: standard U-Net skeleton with every convolutional block replaced by a residual block (3×3 conv–BN–ReLU → 3×3 conv–BN + 1×1 shortcut projection when channel count changes, output ReLU). Encoder depth 4, channel progression 64→128→256→512→1024, decoder via transposed-conv + residual-block; 3-class output. Input channels: 3 (RGB). This matches the ResUNet family as described in Deng et al. (the plain variant without morphological augmentation).

Training notebook: `multi_step_back/unet/ResUnet_Fixed592.ipynb`
Checkpoint: `multi_step_back/unet/models/best_model_resunet_592.pth`

## Training Configuration (identical to fixed-592 protocol)

| Hyper-parameter | Value |
|---|---|
| Resolution | 592 × 592 |
| Seed | 42 |
| Epochs | 300 |
| Batch size | 4 |
| Loss | CEDiceLoss (dice_weight=0.3) |
| Optimizer | AdamW, lr=5×10⁻⁴, weight_decay=2×10⁻⁴ |
| LR schedule | Cosine annealing to 1×10⁻⁵ |
| Augmentation | Horizontal flip p=0.5, vertical flip p=0.5, rotation ±10° |
| Class weights | Inverse pixel frequency, normalised by vessel mean |
| Checkpoint | Best test-set vessel Dice (eval every 5 epochs, patience 20) |
| Parameters | 32,436,483 |

## Results

Evaluated on the 20 held-out Fundus-AVSeg test images (592×592 resize, same protocol as all other fixed-592 models):

| Metric | Artery | Vein | Vessel mean |
|---|---|---|---|
| Dice | 0.7988 | 0.8218 | **0.8103** |
| IoU | 0.6665 | 0.6991 | **0.6828** |

## Comparison to Neighbouring Entries (Table 1 rows)

| Model | Vessel Dice | Vessel IoU |
|---|---|---|
| U-Net (fixed-592 baseline) | 0.8023 | 0.6719 |
| **ResUNet (fixed-592, SEED=42)** | **0.8103** | **0.6828** |
| BiFPN-UNet (64–1024 ch) | 0.8092 | 0.6812 |
| BiFPN-UNet (32–256 ch) | 0.7906 | 0.6557 |
| MorphoBiFPN-UNet | 0.7863 | 0.6495 |
| Ensemble top-3 | 0.8320 | 0.7139 |

**The ResUNet (0.8103) outperforms the Standard U-Net baseline (0.8023) by +0.008 Dice and narrowly edges the BiFPN-UNet 64ch variant (0.8092) by +0.001.** Both the ResUNet and BiFPN-64ch are clearly below the top-3 soft-voting ensemble (−0.022 Dice gap). This supports the paper's narrative that ensembling produces gains beyond any single model.

Note: This is a single-seed result (SEED=42). Multi-seed mean ± std will be reported in Task 2 after additional training runs complete.
