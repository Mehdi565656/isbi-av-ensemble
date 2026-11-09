# Is Architectural Diversity Needed? Soft-Voting Ensembles of U-Net Variants for Retinal Artery–Vein Segmentation

Repository assembled for the ISBI submission by Mehdi Feki, Noémie Debroux, Antoine Vacavant
(Université Clermont Auvergne, Clermont Auvergne INP, CNRS, Institut Pascal).

Every file here is a **copy** of material that already existed in `/home/mefeki/Bureau/work`;
nothing was moved or deleted. Files the paper needs but that could not be found on disk are
listed in [MISSING_FILES.md](MISSING_FILES.md).

---

## 1. Layout

```
isbi-av-ensemble/
├── paper/            the submitted PDF + placeholder-resolution notes
├── docs/             written analyses backing each section of the paper
├── code/
│   ├── training/     training notebooks, per-architecture (+ histories, diagnostics, logs)
│   ├── experiments/  multi-seed training, ablations, inference, statistics
│   └── figures/      figure-generation scripts
├── results/
│   ├── metrics/      all CSVs (per-image, aggregate, multi-seed, diversity, cost)
│   ├── softmax_maps/ per-model float16 softmax maps on the 20 test images
│   └── figures/      rendered figures
├── models/           24 checkpoints (seed 42, seeds 1/7/13, ablations, ResUNet-backbone)
└── data/             Fundus-AVSeg, preprocessed to the fixed-592 protocol (70/10/20)
```

Total size ≈ 5.1 GB (2.4 GB checkpoints, 2.5 GB dataset).

## 2. Paper → file map

| Paper element | Where it comes from |
|---|---|
| Table 1, single models (seed 42) | `results/metrics/model_summary.csv`, `aggregate_metrics.csv` |
| Table 1, Morpho-ResUNet / BiFPN-ResUNet rows | `results/metrics/morpho_resunet_scores.csv`, `bifpn_resunet_scores.csv` |
| Table 1, A+B ensemble (0.8296) | `aggregate_metrics.csv`, row `ensemble_top2` |
| Table 1, A+B+C ensemble (0.8320) | `aggregate_metrics.csv`, row `ensemble_top3` |
| Table 1, A+B+{ResUNet-backbone} rows (0.8350 / 0.8364 / 0.8380) | **not reproduced by any on-disk artifact** — see MISSING_FILES.md |
| Table 2, multi-seed mean/std/CI/Wilcoxon | `results/metrics/multiseed_summary.csv`, `multiseed_results.csv`; `code/experiments/compute_statistics.py`; `docs/02_statistical_significance.md` |
| Fig. 1, qualitative on 092_N | `code/figures/gen_figures_pfe_report.py` (`fig8_av_qualitative`) — French variant, A+B+C ensemble |
| §3.3 Ensemble composition | `docs/03_morpho_ablation.md`, `aggregate_metrics.csv` |
| §3.3 Morphological features ablation | `results/metrics/ablation_summary.csv`, `ablation_per_image_scores.csv`; `code/experiments/train_morpho_ablation.py`, `eval_morpho_ablation.py` |
| §3.4 ResUNet-backbone ensembling | `code/experiments/eval_bifpn_resunet.py`, `train_morpho_resunet.py`; `docs/09_bifpn_resunet.md` |
| §3.5 Yule's Q, disagreement rates | `results/metrics/diversity_analysis.csv`; `docs/05_diversity_analysis.md` |
| §3.5 Same-architecture control (0.8203) | `docs/02_statistical_significance.md` §same-arch, `docs/05_diversity_analysis.md` §5.4; checkpoints in `models/seed01,07,13/bifpn_64to1024.pth` |
| Checkpoint-selection limitation (§4) | `docs/04_model_selection_audit.md`, `code/experiments/04_val_selection_check.py` |
| Inference cost (§4 limitation) | `results/metrics/inference_cost.csv`, `docs/07_inference_cost.md` |
| GAVE2 future work (§4) | `docs/06_second_dataset.md` (data/models themselves left in `../MICAI/`) |

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
`models/seed01|seed07|seed13/` each hold `unet / morpho_unet / bifpn_64to1024 / resunet` plus their history CSVs.
`models/morpho_ablation/` holds `morpho_tophat_only`, `morpho_dist_only`, `morpho_black_tophat_dist`.

## 4. Reproducing the numbers

Scripts in `code/experiments/` use absolute paths rooted at `/home/mefeki/Bureau/work`
(e.g. `Fundus-AVSeg-prep-data/`, `paper_fixes/`). To run them from inside this repo, repoint
`BASE` at the top of each script to this directory and adjust `data/fundus-avseg-592` and
`results/` accordingly.

```
code/experiments/run_inference.py        # softmax maps + per-image scores for the 6 seed-42 models
code/experiments/train_multiseed.py      # retrain 4 architectures at seeds 1, 7, 13
code/experiments/train_morpho_ablation.py / eval_morpho_ablation.py   # §3.3 morphology ablation
code/experiments/train_morpho_resunet.py / eval_bifpn_resunet.py      # §3.4 ResUNet-backbone variants
code/experiments/compute_statistics.py   # Table 2: mean/std, bootstrap CI, Wilcoxon + Holm
code/experiments/04_val_selection_check.py  # checkpoint-selection audit
code/experiments/finalize.sh             # orchestrates eval → statistics → figures
```

## 5. Protocol reminders

- Fixed-592: every image bilinearly resized to 592×592 (nearest-neighbour for masks), no padding/cropping.
- 3 classes: 0 background, 1 artery, 2 vein; crossing (green) and uncertain (white) pixels map to background.
- Split: 70 train / 10 val / 20 test (test IDs 081–100).
- A/V Dice and A/V IoU are the mean of the artery and vein scores, background excluded, averaged over the 20 test images.
- Note the documentation under `docs/` predates the ISBI rewrite: it carries the earlier OMIA
  title and framing. Its numbers match the submitted paper; its narrative does not always.
