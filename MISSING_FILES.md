# Missing files

Material the paper depends on that could **not** be found anywhere under
`/home/mefeki/Bureau/work` (or `~/Téléchargements`). Ordered by how much it blocks a
reader trying to reproduce the submission.

## Blocking — results in the paper with no artifact on disk

1. **The LaTeX source of the paper.** No `.tex`, `.bib`, ISBI/IEEE class or style file, or
   `Makefile` exists anywhere in the workspace; the submitted PDF was not kept in this
   repository either. The only `.tex` in the workspace is `pfe_report/binary_bifpn_arch.tex`,
   which belongs to the French PFE report, not this paper.

2. **The exact Figure 1 script.** The paper's Fig. 1 panel (d) is the
   A+B+**BiFPN-ResUNet** ensemble (Dice 0.8380). The closest script on disk,
   `code/figures/gen_figures_pfe_report.py::fig8_av_qualitative`, renders the same test image
   (092_N) but with the A+B+**C** ensemble (0.8320), in French, and as a 4-panel PDF for the
   PFE report. It is a starting point, not the figure that was submitted.

## Regenerated (previously missing, now in the repository)

- **The script that builds the A+B+{ResUNet-backbone} ensembles** — written as
  `code/experiments/build_resunet_backbone_ensembles.py`. It rebuilds every Table 1 ensemble
  from cached softmax maps and runs the §3.4 one-tailed Wilcoxon tests against A+B+C with
  Holm correction, then self-checks each Dice against the published value. It reproduces
  Table 1 to within 0.0001 Dice and §3.4's p-values as reported:

  | Ensemble | Rebuilt Dice / IoU | Paper | Wilcoxon p (Holm) | Paper p |
  |---|---|---|---|---|
  | A+B | 0.8295 / 0.7104 | 0.8296 / 0.7106 | — | — |
  | A+B+C | 0.8320 / 0.7140 | 0.8320 / 0.7139 | — | — |
  | A+B+ResUNet | 0.8350 / 0.7184 | 0.8350 / 0.7184 | 1.64e-02 | 0.0164 |
  | A+B+Morpho-ResUNet | 0.8364 / 0.7203 | 0.8364 / 0.7203 | 7.08e-04 | 0.0004 |
  | A+B+BiFPN-ResUNet | 0.8380 / 0.7228 | 0.8380 / 0.7228 | 5.44e-05 | <1e-4 |

- **Softmax maps for `bifpn_resunet` and `morpho_resunet`** — regenerated from the
  checkpoints by `code/experiments/export_softmax_resunet_backbone.py`, in the same
  `(3, 592, 592)` float16 layout as the six maps `run_inference.py` had cached. Maps for
  `bifpn_resunet_dualskip` were exported too (see below), so `results/softmax_maps/` now
  covers all nine models. Re-running the plain `bifpn_resunet` reproduced the original
  `bifpn_resunet_scores.csv` to 1e-5 per image — the float16 storage rounding — confirming
  the regeneration is faithful.

Two incidental changes were needed:

- The top-level evaluation code in `eval_bifpn_resunet.py` was moved under an
  `if __name__ == '__main__':` guard so its architecture definitions can be imported.
  Running the script directly behaves exactly as before.
- **The paper's "BiFPN-ResUNet" turned out to be the dual-skip checkpoint**, not the plain
  neck insertion that §2 describes. The plain model (`bifpn_resunet.pth`) scores 0.8049 alone
  and 0.8325 in the ensemble; the dual-skip model (`bifpn_resunet_dualskip.pth`) scores
  0.8147 / 0.6889 alone and 0.8380 / 0.7228 in the ensemble — the Table 1 values. The
  ensemble script builds both and labels the plain one "(not in paper)". The paper's §2
  architecture description should be corrected to match the model actually reported.

## Non-blocking — recommended for a reproducible release

3. **Dataset preparation script.** `data/fundus-avseg-592/` is the finished 70/10/20 split with
   `images/`, `annotation/` and `npy/` per split, but nothing on disk shows how it was built
   from the raw `Fundus-AVSeg.zip` (in `~/Téléchargements`) — the split assignment, the resize,
   and the RGB→class mapping live only inside each training notebook's `ImageDataset`.

4. **`environment.yml` / `requirements.txt` and version pins.** No dependency manifest exists.
   The paper states PyTorch on an NVIDIA Quadro RTX 8000, but torch/CUDA versions are not
   recorded anywhere. (A `.venv/` exists in the parent workspace but was not copied here;
   `../binary/requirements.txt` belongs to the earlier binary-segmentation work and is unpinned
   for this paper, but is the only manifest in the workspace and could serve as a starting point.)

5. **Seed-42 run artifacts for the two ResUNet-backbone extensions.** `morpho_resunet_history.csv`
   and `morpho_resunet_log.txt` were copied, and BiFPN-ResUNet has a history CSV + diagnostics
   PNG, but there is no per-seed reproduction for either model — consistent with the paper's own
   limitation that these results are single-seed.

6. **Per-image scores for the GAVE2/FFA second-dataset check.** `docs/06_second_dataset.md`
   reports only aggregate val Dice; per-image metrics were never saved, and the challenge test
   set has no local ground truth. Relevant only to the §4 future-work sentence.

7. **Standard U-Net per-epoch history.** The baseline was trained before the fixed-592 protocol
   (final-epoch checkpoint, no per-epoch tracking), so only
   `training_diagnostics_standard_unet.png` exists — no `training_history_standard_unet.csv`.

## Deliberately not copied (available in the parent workspace)

- `../MICAI/` — GAVE2/FFA challenge work (2.4 GB). Only referenced as future work in §4;
  `docs/06_second_dataset.md` summarises it.
- `../binary/`, `../drive_kaggle_prepared_data/`, `../multi_step_back/{BiFPN_WCE,Morpho*}.ipynb`,
  LIOT and RGB notebook variants — earlier binary-segmentation and DRIVE work, not part of this paper.
- `../pfe_report/` beyond the three figures and `gen_figures.py` that overlap with the paper.
- The submitted PDF — removed from this repository at the author's request.
- `../paper_fixes.zip`, `../paper_fixes_lean.zip`, `../new_models.zip` — archives of material
  already copied here in unpacked form.
