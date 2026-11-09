# Context for Claude — OMIA 2026 Paper Revision Package

## What you are looking at

This folder is a **results package** for an OMIA 2026 workshop paper (submission #43):

> **"Soft-Voting Ensembles of U-Nets Augmented with Morphological and Bidirectional Features for Retinal Artery–Vein Segmentation"**

The package was produced to support a paper revision. It contains:
- Numerical results (tables, per-image metrics)
- Statistical significance tests
- Ablation study results
- Figures (box plots, bar charts, morphological feature visualisations)
- Written analyses of findings, including reviewer concerns

**The paper itself is not included.** Your job is to help revise the paper text using these results.

---

## Task / problem context

The paper proposes a **soft-voting ensemble** of three U-Net variants for 3-class retinal artery–vein (AV) segmentation (background / artery / vein) on the **Fundus-AVSeg dataset** (100 colour fundus images, 70 train / 10 val / 20 test, published June 2025).

The three ensemble members are:
- Standard U-Net (31M params)
- Morpho-U-Net (31M params) — U-Net with white top-hat + distance-transform preprocessing
- BiFPN-UNet 64–1024ch (20M params) — U-Net with bidirectional feature pyramid (BiFPN) decoder

The main claim is that combining architecturally diverse models via soft voting improves over any single model. The paper also proposes two novel components: morphological preprocessing and bidirectional skip connections (BiFPN).

---

## Dataset and metric definitions

- **Dataset:** Fundus-AVSeg (Raza et al., *Scientific Data* 2025). 100 colour fundus images.
- **Classes:** 0 = background, 1 = artery (red), 2 = vein (blue)
- **Vessel Dice** = mean of (artery Dice + vein Dice) / 2 — the primary metric
- **Vessel IoU** = mean of (artery IoU + vein IoU) / 2
- **Fixed-592 protocol:** all images resized to 592×592 for training and inference
- **Test set:** 20 images (IDs 081–100)
- **Val set:** 10 images (used for ensemble member selection)

---

## Files in this package

### Analysis documents (read these first)

| File | What it contains |
|---|---|
| `RESULTS_SUMMARY.md` | **Start here.** One-page summary of all key numbers and reviewer response guidance |
| `00_inventory.md` | Checkpoint inventory, training configurations |
| `01_resunet_fixed592.md` | ResUNet baseline evaluation |
| `02_statistical_significance.md` | Multi-seed (4 seeds) mean±std, bootstrap 95% CIs, Wilcoxon tests, same-arch ensemble control |
| `03_morpho_ablation.md` | Ablation of morphological preprocessing variants (key unexpected finding) |
| `04_model_selection_audit.md` | Data-leakage audit of model selection procedure |
| `05_diversity_analysis.md` | Q-statistic and disagreement-rate analysis of ensemble diversity |
| `06_second_dataset.md` | Results on GAVE2 FFA (second dataset, different modality) |
| `07_inference_cost.md` | Parameter counts and GPU inference timing |
| `08_sota_comparison.md` | SOTA comparison and why cross-dataset numbers are not directly comparable |
| `09_bifpn_resunet.md` | BiFPN-ResUNet — best single model, was trained but omitted from paper |

### Data

| File | What it contains |
|---|---|
| `data/per_image_scores.csv` | Per-image Dice and IoU for all 6 models + 3 ensemble variants, all 20 test images |
| `data/model_summary.csv` | Aggregate mean Dice/IoU for each model (seed-42 only) |
| `data/multiseed_summary.csv` | Mean±std across 4 seeds for each model and ensemble |
| `data/aggregate_metrics.csv` | Full aggregate results table |
| `data/diversity_analysis.csv` | Q-statistics and disagreement rates for all model pairs |
| `data/inference_cost.csv` | Params and timing per model |

### Figures

| File | What it shows |
|---|---|
| `figures/per_model_dice_boxplot.png` | Box plot: vessel Dice distribution per model, 20 test images |
| `figures/ensemble_comparison_bar.png` | Bar chart: mean±std vessel Dice for all models |
| `figures/artery_vein_comparison.png` | Grouped bar: artery vs vein Dice by model |
| `figures/morpho_channels_overview.png` | 3×3 grid: RGB / white top-hat / distance transform for 3 image types (N/D/G) |
| `figures/morpho_channels_082_N.png` | Per-channel detail: normal image |
| `figures/morpho_channels_084_D.png` | Per-channel detail: diabetic image |
| `figures/morpho_channels_083_G.png` | Per-channel detail: glaucoma image |

---

## Key numbers (memorise these)

### Primary result — seed-42 (paper's reported result)

| Model | Vessel Dice | Vessel IoU | In paper? |
|---|---|---|---|
| BiFPN-UNet (32–256ch) | 0.7906 | 0.6557 | Yes |
| MorphoBiFPN-UNet | 0.7863 | 0.6495 | Yes |
| BiFPN-UNet (64–1024ch) | 0.8092 | 0.6812 | Yes |
| Standard U-Net | 0.8023 | 0.6719 | Yes |
| Morpho-U-Net | 0.7917 | 0.6572 | Yes |
| ResUNet | 0.8103 | 0.6828 | Yes |
| **BiFPN-ResUNet** | **0.8147** | **0.6889** | No — omitted |
| Morpho-ResUNet | TBD | — | No — training |
| **Ensemble top-3** | **0.8320** | **0.7139** | Yes |

Top-3 members (selected by val-set Dice): BiFPN-64ch + U-Net + Morpho-U-Net.

**BiFPN-ResUNet** (ResNet encoder + BiFPN neck, 22M params) is the best single model.
It was trained but not included in the paper. See `09_bifpn_resunet.md`.
**Morpho-ResUNet** (morphological preprocessing + ResUNet backbone) is currently training.

### Multi-seed statistics (4 seeds: 42, 1, 7, 13)

| Model | Mean Dice | Std | 95% CI |
|---|---|---|---|
| ResUNet | 0.8129 | 0.0022 | [0.796, 0.827] |
| U-Net | 0.8082 | 0.0045 | [0.791, 0.823] |
| BiFPN-64ch | 0.8077 | 0.0037 | [0.792, 0.823] |
| Morpho-U-Net | 0.8017 | 0.0070 | [0.783, 0.817] |
| **Ensemble** | **0.8242** | **0.0056** | **[0.808, 0.839]** |

Wilcoxon (ensemble > each single, one-tailed, N=20): p ≈ 0 for all comparisons.

### Morpho ablation

| Variant | Vessel Dice |
|---|---|
| Morpho-U-Net full (WTH + Dist, 9-ch) | 0.7917 |
| White top-hat only (6-ch) | **0.8111** |
| Distance transform only (6-ch) | 0.8017 |
| Black top-hat + Dist (9-ch) | 0.7996 |

### Same-architecture ensemble control

3×BiFPN-64ch (seeds 1, 7, 13): Dice = **0.8203**
Heterogeneous ensemble mean (4 seeds): **0.8242** → +0.004 gap

---

## Key findings (unexpected / important for revision)

### 1. Distance-transform features hurt, not help

The full Morpho-U-Net (9-channel: RGB + white top-hat + distance transform) scores **0.7917 Dice**, which is **lower than the white-top-hat-only variant (0.8111)**. Both 6-channel ablations outperform the full model. The distance-transform channels appear to increase optimisation difficulty without adding useful information.

**Implication:** The paper's claim that "combined morphological features" are beneficial is not supported. White top-hat alone is the correct takeaway.

### 2. Ensemble gain is variance reduction, not architectural diversity

Q-statistics between all model pairs: 0.997–0.999 (near-perfect error correlation). The same-architecture ensemble (3×BiFPN-64, different seeds) achieves 0.8203 — only 0.004 below the heterogeneous ensemble mean (0.8242). The architectural diversity adds negligible gain beyond what three independent training runs of the same model would provide.

**Implication:** The "diverse architectures" framing overstates what the data shows. The honest description is: soft voting between multiple independently-trained models reduces prediction variance, regardless of architecture.

### 3. Model selection had data leakage (but no practical impact)

5 of 6 models used test-set Dice for best-checkpoint selection (not val-set). However, reselecting ensemble members using only the val-set produces the **identical top-3**, so the reported ensemble performance is valid.

### 4. ResUNet is the best single model

ResUNet (0.8103 Dice at seed-42; mean 0.8129 across 4 seeds) consistently outperforms the Standard U-Net (0.8023; mean 0.8082). This is relevant for the baseline comparison table.

### 5. GAVE2 FFA is a different modality

The second dataset (GAVE2 FFA) uses fluorescein angiography, not colour fundus photography. Results are ~19 Dice points lower (U-Net: 0.629 vs 0.802). This should be framed as a generalisability check, not a benchmark comparison. The ensemble was not evaluated on GAVE2.

### 6. No external SOTA exists on Fundus-AVSeg

The dataset was published June 2025 — no external paper has reported results on it. The paper's ensemble (0.8320 at seed-42, 0.8242 mean) is the best published result. Cross-dataset comparisons (RITE, HRF) are not valid due to metric differences.

---

## Reviewer concerns and evidence

| Reviewer concern | What the data shows | Suggested response |
|---|---|---|
| Architecture diversity claim | Q≈0.998; same-arch ensemble ≈ hetero ensemble (−0.004) | Reframe: "averaging reduces variance; member choice matters for best single model" |
| Top-hat inappropriate for dark vessels | WTH responds strongly to both A+V in green ch (see figures); ablation shows WTH-only > full | Ablation confirms white TH is the right choice; show morpho_channels figure |
| Model selection leakage | 5/6 models leaked; val-selection = same top-3 | Disclose, note no practical impact |
| SOTA comparison on RITE/HRF | Different metric (AV classification accuracy on vessel pixels ≠ 3-class Dice) | Add caveat table; do not directly compare numbers |
| Second dataset results | FFA ≠ colour fundus; ensemble not evaluated | Frame as generalisability check |
| Statistical significance | Wilcoxon p≈0; ensemble better on 208–210/210 images | Use the Wilcoxon results |

---

## What you should NOT do

- Do not modify the paper text directly (not included here anyway)
- Do not assume the paper's narrative is fully supported — several claims need revision (see §Key findings above)
- Do not treat the seed-42 result (0.8320) as the definitive ensemble performance — 0.8242 ± 0.0056 is the multi-seed estimate
