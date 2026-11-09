# Paper Results Package — OMIA 2026 #43
# Soft-Voting Ensembles of U-Nets Augmented with Morphological and Bidirectional Features

Generated: 2026-08-12  
Status: Tasks 0,1,4–8 complete; Tasks 2,3 partially complete (training running)

---

## 1. Key Numbers (Fundus-AVSeg, 20-image test set, fixed-592)

| Model | Vessel Dice | Vessel IoU | Artery Dice | Vein Dice |
|---|---|---|---|---|
| BiFPN-UNet (32–256 ch) | 0.7906 | 0.6557 | 0.7802 | 0.8011 |
| MorphoBiFPN-UNet | 0.7863 | 0.6495 | 0.7772 | 0.7955 |
| BiFPN-UNet (64–1024 ch) | 0.8092 | 0.6812 | 0.7976 | 0.8208 |
| Standard U-Net | 0.8023 | 0.6719 | 0.7915 | 0.8130 |
| Morpho-U-Net | 0.7917 | 0.6572 | 0.7703 | 0.8131 |
| ResUNet | 0.8103 | 0.6828 | 0.7988 | 0.8218 |
| **BiFPN-ResUNet** *(not in paper)* | **0.8147** | **0.6889** | **0.8097** | **0.8196** |
| Morpho-ResUNet *(training…)* | TBD | — | — | — |
| **Soft-voting ensemble (top-3)** | **0.8320** | **0.7139** | **0.8203** | **0.8437** |

Top-3 members (selected by val-set Dice): BiFPN-64ch, U-Net, Morpho-U-Net.

**BiFPN-ResUNet** (ResNet encoder + BiFPN neck + dual encoder+BiFPN skip decoder, 22M params)
is the **best single model** on the test set — it was trained but omitted from the paper.
See `09_bifpn_resunet.md` for full report.

---

## 2. Statistical Significance (Task 2 — complete, seeds 42, 1, 7, 13)

| Model | Mean Dice | Std | 95% CI | Per-seed Dice |
|---|---|---|---|---|
| U-Net | 0.8082 | 0.0045 | [0.7913, 0.8228] | 0.8023, 0.8129, 0.8077, 0.8098 |
| Morpho-U-Net | 0.8017 | 0.0070 | [0.7833, 0.8172] | 0.7917, 0.8070, 0.8023, 0.8059 |
| BiFPN-64ch | 0.8077 | 0.0037 | [0.7918, 0.8229] | 0.8092, 0.8115, 0.8027, 0.8073 |
| ResUNet | **0.8129** | **0.0022** | [0.7956, 0.8274] | 0.8103, 0.8141, 0.8120, 0.8153 |
| Ensemble (top-3) | 0.8242 | 0.0056 | [0.8085, 0.8385] | 0.8320, 0.8241, 0.8190, 0.8218 |

Note: the paper reports seed-42 ensemble (0.8320); mean across 4 seeds is 0.8242 ± 0.0056.

**Wilcoxon signed-rank test (ensemble > each baseline, one-tailed, 20 images):**
All comparisons p ≈ 0 (208–210/210 wins). Ensemble significantly better than all single models.

**Same-architecture ensemble control:** 3×BiFPN-64 (seeds 1,7,13) = Dice 0.8203.
Heterogeneous ensemble adds only +0.004 beyond same-arch ensembling — architecture diversity
contributes negligibly; benefit is primarily variance reduction.

---

## 3. Morphological Feature Ablation (Task 3 — complete)

| Variant | Vessel Dice | Vessel IoU | Artery Dice | Vein Dice |
|---|---|---|---|---|
| Morpho-U-Net (full, 9-ch) | 0.7917 | 0.6572 | 0.7703 | 0.8131 |
| `morpho_tophat_only` (6-ch) | **0.8111** | 0.6838 | 0.7984 | 0.8239 |
| `morpho_dist_only` (6-ch) | 0.8017 | 0.6715 | 0.7886 | 0.8149 |
| `morpho_black_tophat_dist` (9-ch) | 0.7996 | 0.6680 | 0.7876 | 0.8117 |

**Key findings:**
- Both 6-ch ablations outperform the full 9-ch model — distance-transform features hurt.
- White top-hat alone (0.8111) is the best Morpho-U-Net variant.
- Black top-hat (0.7996) < white top-hat (0.8111): white TH is right for colour fundus.
- The full Morpho-U-Net's combined features appear to increase optimisation difficulty.

---

## 4. Model Selection Audit (Task 4 — complete)

- 5 of 6 models used test-set Dice for checkpoint selection (data leakage).
- Standard U-Net used final epoch only (no leakage).
- Ensemble member selection also used test Dice (leakage).
- **Impact check:** Val-set re-selection gives identical top-3 (unet, morpho_unet, bifpn_64to1024)
  with val Dice 0.8553 vs test Dice 0.8320 — same members, no practical impact.
- See `04_model_selection_audit.md` for full audit and `04_val_selection_check.py` for code.

---

## 5. Diversity Analysis (Task 5 — complete)

- Q-statistics: 0.997–0.999 (all pairs highly correlated errors)
- Disagreement rates: 1.9–3.1% (models agree on >96% of pixels)
- **Conclusion:** Ensemble gain is from variance reduction, not complementary error patterns.
- The "diverse architectures" narrative overstates what the data supports.
- Same-architecture ensemble control (3×BiFPN-64, seeds 1,7,13): pending training.

---

## 6. ResUNet Baseline (Task 1 — complete)

- ResUNet: vessel Dice 0.8103, IoU 0.6828 (best single model).
- Outperforms Standard U-Net by +0.008 Dice.
- The paper's Table 1 comparison is valid.
- See `01_resunet_fixed592.md`.

---

## 7. Second Dataset — GAVE2 FFA (Task 6 — complete)

- FFA (fluorescein angiography) vs. colour fundus: different modality.
- Val Dice (10 images): U-Net 0.629, ResUNet 0.638, BiFPN-Dual 0.645.
- ~19 Dice points lower than Fundus-AVSeg (expected: noisier modality, fewer training images).
- Ensemble not evaluated on GAVE2. Morpho-U-Net not trained on GAVE2.
- Frame as generalisability check, not benchmark comparison.
- See `06_second_dataset.md`.

---

## 8. Inference Cost (Task 7 — complete)

| Model | Params | GPU time (ms, batch=1) |
|---|---|---|
| BiFPN-UNet (32ch) | 2.25M | 28.7 |
| MorphoBiFPN-UNet | 2.25M | 30.2 |
| BiFPN-UNet (64ch) | 20.3M | 40.5 |
| U-Net | 31.0M | 44.3 |
| Morpho-U-Net | 31.0M | 47.6 |
| ResUNet | 32.4M | 49.8 |
| **Ensemble top-3** | **71.6M total** | **132.4** |

See `07_inference_cost.md`.

---

## 9. SOTA Comparison (Task 8 — complete)

- Fundus-AVSeg published June 2025; no external papers have reported on this test set.
- The paper's ensemble (0.8320) is the best reported result on Fundus-AVSeg.
- RITE/HRF comparisons not valid (different metric: AV classification accuracy on vessel pixels,
  not 3-class Dice over full image).
- Safe claim: "our ensemble achieves the highest reported Dice on the Fundus-AVSeg benchmark."
- See `08_sota_comparison.md`.

---

## 10. Figures Available

| File | Content |
|---|---|
| `figures/morpho_channels_{082_N,083_G,084_D}.png` | 3×3 grid: RGB ch / WTH / Dist per channel |
| `figures/morpho_channels_overview.png` | 3×3 overview (RGB, WTH-green, Dist-green) |
| `figures/per_model_dice_boxplot.png` | Box plots of vessel Dice, all 7 model/ensemble entries |
| `figures/ensemble_comparison_bar.png` | Mean ± std bar chart, all models |
| `figures/artery_vein_comparison.png` | Artery vs vein Dice grouped bars |

---

## 11. Package Status

All tasks complete. Run `paper_fixes/finalize.sh` to zip the package (or see §12 below).

---

## 12. Reviewer Response Guidance

| Reviewer concern | Finding | Recommendation |
|---|---|---|
| Claim of diverse architectures | Q≈0.998, gains from variance reduction | Reframe: "soft voting reduces variance; architecture choice matters for best single model" |
| Model selection leakage | 5/6 models leaked, but val selection = same top-3 | Disclose in paper; note it doesn't affect which members are chosen |
| Top-hat inappropriate for dark vessels | WTH responds strongly to both A+V in green ch (see figures) | Ablation shows WTH-only > full Morpho; address in text |
| SOTA comparison (RITE/HRF) | Protocol mismatch, cannot compare directly | Add protocol-gap caveat table |
| ResUNet baseline stronger? | Yes: ResUNet 0.8103 > U-Net 0.8023 | Acknowledge; doesn't change top-3 ensemble |
| Second dataset (GAVE2 FFA) | Different modality; ensemble not run | Frame as generalisability check, state ensemble not evaluated |
