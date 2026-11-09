# Task 5 — Diversity / Complementarity Evidence

All metrics computed from the per-image softmax maps saved in Task 0.4.
Test set: 20 images, Fundus-AVSeg.

## 5.1 Prediction Disagreement Rate

Fraction of pixels where argmax class differs between two models, averaged over 20 test images:

| Model A | Model B | Disagreement rate |
|---|---|---|
| U-Net | Morpho-U-Net | 3.06% |
| U-Net | BiFPN-UNet 32ch | 3.10% |
| U-Net | MorphoBiFPN-UNet | 3.12% |
| U-Net | BiFPN-UNet 64ch | 2.95% |
| Morpho-U-Net | BiFPN-UNet 32ch | 2.24% |
| Morpho-U-Net | MorphoBiFPN-UNet | 2.19% |
| Morpho-U-Net | BiFPN-UNet 64ch | 2.30% |
| BiFPN-UNet 32ch | MorphoBiFPN-UNet | 1.91% |
| BiFPN-UNet 32ch | BiFPN-UNet 64ch | 1.99% |
| MorphoBiFPN-UNet | BiFPN-UNet 64ch | 2.10% |

Disagreement rates range from **1.9% to 3.1%** — all models agree on more than 96% of pixels. This is consistent with high-performing models in a task where the large background class dominates.

## 5.2 Error Correlation: Q-Statistic (Yule's Q)

For each pixel that belongs to ground-truth class c, an "error indicator" is 1 if the model misclassifies it and 0 otherwise. Yule's Q between the two models' error indicators measures error correlation: Q=+1 means perfectly correlated errors (both fail on the same pixels), Q=−1 means perfectly anti-correlated (complementary), Q=0 means independent.

| Model A | Model B | Q (artery) | Q (vein) |
|---|---|---|---|
| U-Net | Morpho-U-Net | **0.997** | **0.998** |
| U-Net | BiFPN-UNet 32ch | **0.997** | **0.997** |
| U-Net | MorphoBiFPN-UNet | **0.997** | **0.997** |
| U-Net | BiFPN-UNet 64ch | **0.997** | **0.997** |
| Morpho-U-Net | BiFPN-UNet 32ch | **0.998** | **0.998** |
| Morpho-U-Net | MorphoBiFPN-UNet | **0.998** | **0.998** |
| Morpho-U-Net | BiFPN-UNet 64ch | **0.998** | **0.998** |
| BiFPN-UNet 32ch | MorphoBiFPN-UNet | **0.999** | **0.999** |
| BiFPN-UNet 32ch | BiFPN-UNet 64ch | **0.998** | **0.998** |
| MorphoBiFPN-UNet | BiFPN-UNet 64ch | **0.998** | **0.998** |

**All pairwise Q-statistics are between 0.997 and 0.999 — extremely close to +1.** Every pair of models fails on essentially the same pixels. There is no detectable complementarity in errors across the entire test set.

## 5.3 Interpretation

The near-unity Q-statistics arise from a base-rate effect: most vessel pixels are correctly classified (each model achieves Dice >0.79). Errors are rare events, and rare events in segmentation tend to co-occur spatially (thin vessel regions, low-contrast crossings, peripheral vessels) — all models struggle with the same hard pixels regardless of architecture. The Q-statistic collapses toward +1 whenever both models are accurate enough that their errors share the same difficult locations.

**What the diversity analysis reveals:**

1. The ensemble gain (Dice 0.8023→0.8320 for U-Net→top-3) is **not** primarily driven by error complementarity. It comes from **variance reduction**: averaging softmax scores smooths out individual model confidence fluctuations, producing better-calibrated probability estimates before the argmax step, even when the error pixels overlap heavily.

2. The 2.9–3.1% disagreement rate between the best pairs (U-Net vs BiFPN-64ch) is meaningful: these disagreements are concentrated at vessel/background boundaries and at vessel crossings, and the averaging resolves them in the right direction more often than either model alone.

## 5.4 Same-Architecture Ensemble Control (Reviewer y8o4 concern)

**Addressing the concern:** If the ensemble gain comes purely from averaging 3 independently-trained instances (noise cancellation), then an ensemble of 3 seeds of the single best model should match the architecturally-diverse ensemble. Only if the diverse ensemble beats the same-architecture multi-seed ensemble can we claim the diversity contributes beyond simple noise averaging.

**Result:** 3×BiFPN-64ch ensemble (seeds 1, 7, 13): Dice = **0.8203** [95% CI: 0.8038, 0.8352]

The mean heterogeneous ensemble across all 4 seeds = 0.8242. The architecturally-diverse ensemble adds only **+0.0039** over the same-architecture ensemble — a margin smaller than the ensemble's seed-to-seed std (±0.0056). The architectural diversity contributes negligibly beyond what three independent seeds of the same model already provide.

**Conclusion:** The Q-statistic analysis and the same-architecture control agree. The ensemble's benefit is variance reduction, not complementary error patterns from diverse architectures.

## 5.5 Honest Verdict

The Q-statistic analysis provides no empirical support for "complementary errors" as the mechanism behind the ensemble gain. The models all fail on the same hard pixels. The ensemble benefit is real (0.83 vs 0.80 Dice) but arises from **probability averaging / variance reduction**, not from architecturally diverse error patterns. The paper should either reframe the diversity claim accordingly, or provide the same-architecture control experiment (Task 2) to pin down the contribution of architecture diversity specifically.
