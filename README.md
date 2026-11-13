# Is Architectural Diversity Needed? Soft-Voting Ensembles of U-Net Variants for Retinal Artery–Vein Segmentation

Repository assembled for the ISBI submission by Mehdi Feki, Noémie Debroux, Antoine Vacavant
(Université Clermont Auvergne, Clermont Auvergne INP, CNRS, Institut Pascal).

Every file here is a **copy** of material from the authors' original working directory;
nothing was moved or deleted. That directory is written as `/path/to/workspace` throughout
this repository — substitute your own path.

Two things the paper relies on were never found on disk and are **not** in this repository:
the LaTeX/BibTeX source of the submission, and the exact script that rendered Figure 1
(panel (d) is the A+B+BiFPN-ResUNet ensemble; the closest script here,
`code/figures/gen_figures_pfe_report.py::fig8_av_qualitative`, renders the same test image
with the A+B+C ensemble, in French, for the PFE report).

---

## 1. Layout

```
isbi-av-ensemble/
├── code/
│   ├── training/     training notebooks, per-architecture (+ histories, diagnostics, logs)
│   ├── experiments/  multi-seed training, ablations, inference, statistics
│   └── figures/      figure-generation scripts
├── results/
│   ├── metrics/      all CSVs (per-image, aggregate, multi-seed, diversity, cost)
│   ├── softmax_maps/ float16 softmax maps on the 20 test images, 9 models   [not in git]
│   └── figures/      rendered figures
├── models/           24 checkpoints (seed 42, seeds 1/7/13, ablations, ResUNet)  [not in git]
└── data/             Fundus-AVSeg at the fixed-592 protocol (70/10/20)           [not in git]
```

The working tree is ≈ 5.1 GB, but `models/`, `data/` and `results/softmax_maps/` are
gitignored, so a clone is ≈ 11 MB of code, documentation, metrics and figures. The
checkpoints and the preprocessed dataset have to be obtained separately, or rebuilt: the
dataset comes from Deng et al.'s open-access release, and every model has its training
notebook under `code/training/` or its training script under `code/experiments/`.

## 2. Paper → file map

| Paper element | Where it comes from |
|---|---|
| Table 1, single models (seed 42) | `results/metrics/model_summary.csv`, `aggregate_metrics.csv` |
| Table 1, Morpho-ResUNet / BiFPN-ResUNet rows | `results/metrics/morpho_resunet_scores.csv`, `bifpn_resunet_scores.csv`, `resunet_backbone_per_image_scores.csv` |
| Table 1, A+B ensemble (0.8296) | `aggregate_metrics.csv`, row `ensemble_top2` |
| Table 1, A+B+C ensemble (0.8320) | `aggregate_metrics.csv`, row `ensemble_top3` |
| Table 1, A+B+{ResUNet-backbone} rows (0.8350 / 0.8364 / 0.8380) | `results/metrics/ensemble_composition_summary.csv`; rebuilt by `code/experiments/build_resunet_backbone_ensembles.py` |
| Table 2, multi-seed mean/std/CI/Wilcoxon | `results/metrics/multiseed_summary.csv`, `multiseed_results.csv`; `code/experiments/compute_statistics.py` |
| Fig. 1, qualitative on 092_N | `code/figures/gen_figures_pfe_report.py` (`fig8_av_qualitative`) — French variant, A+B+C ensemble |
| §3.3 Ensemble composition | `results/metrics/ensemble_composition_summary.csv`, `aggregate_metrics.csv` |
| §3.3 Morphological features ablation | `results/metrics/ablation_summary.csv`, `ablation_per_image_scores.csv`; `code/experiments/train_morpho_ablation.py`, `eval_morpho_ablation.py` |
| §3.4 ResUNet-backbone ensembling, incl. Wilcoxon p-values | `results/metrics/ensemble_composition_tests.csv`; `code/experiments/build_resunet_backbone_ensembles.py`, `eval_bifpn_resunet.py`, `train_morpho_resunet.py` |
| §3.5 Yule's Q, disagreement rates | `results/metrics/diversity_analysis.csv`; computed in `code/experiments/run_inference.py` |
| §3.5 Same-architecture control (0.8203) | `code/experiments/compute_statistics.py` (same-arch section); checkpoints in `models/seed01,07,13/bifpn_64to1024.pth` |
| Checkpoint-selection limitation (§4) | `code/experiments/04_val_selection_check.py` |
| Inference cost (§4 limitation) | `results/metrics/inference_cost.csv` |
| GAVE2 future work (§4) | not part of this repository (kept in the author's separate `MICAI/` workspace) |

## 3. Checkpoint naming

`models/seed42/` holds the primary-seed models under paper-facing names:

| File | Paper name | Original path |
|---|---|---|
| `unet.pth` | U-Net (baseline), member **B** | `multi_step_back/unet/models/best_model_standard_unet.pth` |
| `resunet.pth` | ResUNet | `multi_step_back/unet/models/best_model_resunet_592.pth` |
| `morpho_unet.pth` | Morpho-U-Net, member **C** | `multi_step_back/morpho/models/best_model_morpho_unet_tophat_dist_592.pth` |
| `bifpn_unet_64to1024.pth` | BiFPN-UNet (64–1024ch), member **A** | `multi_step_back/BiFPN/models/best_model_bifpn_64to1024_592.pth` |
| `bifpn_unet_32to256.pth` | BiFPN-UNet (32–256ch) | `multi_step_back/BiFPN/models/best_model_bifpn_592.pth` |
| `morpho_bifpn_unet.pth` | MorphoBiFPN-UNet (reported in package, not in paper table) | `multi_step_back/morpho_BiFPN/models/best_model_morphobifpn_tophat_dist_592.pth` |

`models/resunet_backbone/` holds `bifpn_resunet.pth`, `bifpn_resunet_dualskip.pth`, `morpho_resunet.pth`.

> **Which checkpoint is the paper's "BiFPN-ResUNet"?** The dual-skip one. Two ResUNet+BiFPN
> checkpoints exist: the plain neck insertion (`bifpn_resunet.pth`, 21.0M params) scores
> 0.8049 Dice alone and 0.8325 in A+B+·, while the dual-skip variant
> (`bifpn_resunet_dualskip.pth`, 22.1M params, which also passes the raw encoder features to
> the decoder) scores 0.8147 / 0.6889 alone and 0.8380 / 0.7228 in A+B+· — i.e. exactly the
> Table 1 rows. Section 2 of the paper describes only the plain neck insertion, so the
> architecture text understates what the reported model does.
`models/seed01|seed07|seed13/` each hold `unet / morpho_unet / bifpn_64to1024 / resunet` plus their history CSVs.
`models/morpho_ablation/` holds `morpho_tophat_only`, `morpho_dist_only`, `morpho_black_tophat_dist`.

## 4. Reproducing the numbers

Most scripts in `code/experiments/` use absolute paths rooted at the placeholder
`/path/to/workspace` (and expect `Fundus-AVSeg-prep-data/` and `paper_fixes/` beneath it).
Set `BASE` at the top of each script to a real directory before running one, and adjust
`data/fundus-avseg-592` and `results/` accordingly. The same placeholder appears in the
training notebooks' dataset paths. The two scripts added to rebuild the Sec. 3.4 ensembles
(`export_softmax_resunet_backbone.py`, `build_resunet_backbone_ensembles.py`) already resolve
every path relative to this repository and take `--repo` to override it.

To rebuild the Sec. 3.4 results from the checkpoints:

```bash
cd code/experiments
python export_softmax_resunet_backbone.py        # add --device cpu if the GPU is busy
python build_resunet_backbone_ensembles.py
```

```
code/experiments/run_inference.py        # softmax maps + per-image scores for the 6 seed-42 models
code/experiments/train_multiseed.py      # retrain 4 architectures at seeds 1, 7, 13
code/experiments/train_morpho_ablation.py / eval_morpho_ablation.py   # §3.3 morphology ablation
code/experiments/train_morpho_resunet.py / eval_bifpn_resunet.py      # §3.4 ResUNet-backbone variants
code/experiments/export_softmax_resunet_backbone.py   # softmax maps for the 2 ResUNet-backbone models
code/experiments/build_resunet_backbone_ensembles.py  # Table 1 ensembles + Sec. 3.4 Wilcoxon/Holm
code/experiments/compute_statistics.py   # Table 2: mean/std, bootstrap CI, Wilcoxon + Holm
code/experiments/04_val_selection_check.py  # checkpoint-selection audit
code/experiments/finalize.sh             # orchestrates eval → statistics → figures
```

## 5. Protocol reminders

- Fixed-592: every image bilinearly resized to 592×592 (nearest-neighbour for masks), no padding/cropping.
- 3 classes: 0 background, 1 artery, 2 vein; crossing (green) and uncertain (white) pixels map to background.
- Split: 70 train / 10 val / 20 test (test IDs 081–100).
- A/V Dice and A/V IoU are the mean of the artery and vein scores, background excluded, averaged over the 20 test images.
- Every number in `results/metrics/` was produced under this protocol; the scripts that
  generated each CSV are named in the map above.

## 6. License

The code, scripts and results in this repository are released under the MIT License
(see [LICENSE](LICENSE)).

This does **not** cover the Fundus-AVSeg data itself, which is a third-party open-access
release by Deng et al., *Sci. Data* 12:1298 (2025) and carries its own license. The dataset
is not redistributed here.
