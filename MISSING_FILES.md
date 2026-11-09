# Missing files

Material the paper depends on that could **not** be found anywhere under
`/home/mefeki/Bureau/work` (or `~/Téléchargements`). Ordered by how much it blocks a
reader trying to reproduce the submission.

## Blocking — results in the paper with no artifact on disk

1. **The LaTeX source of the paper.** Only the compiled PDF exists
   (`paper/ISBI_paper_source.pdf`). No `.tex`, `.bib`, no ISBI/IEEE class or style file, no
   `Makefile`. The only `.tex` in the whole workspace is `pfe_report/binary_bifpn_arch.tex`,
   which belongs to the French PFE report, not this paper.

2. **The script that builds the A+B+{ResUNet-backbone} ensembles.** Table 1 reports
   A+B+ResUNet 0.8350, A+B+Morpho-ResUNet 0.8364 and A+B+BiFPN-ResUNet 0.8380, and §3.4
   reports Wilcoxon p-values (<1e-4, 0.0004, 0.0164) against A+B+C. No script, CSV or log on
   disk produces these five numbers. `eval_bifpn_resunet.py` evaluates the ResUNet-backbone
   models individually but never forms the ensembles.

3. **Softmax maps for the ResUNet-backbone models.** `results/softmax_maps/` covers
   `unet, morpho_unet, bifpn_unet, bifpn_64to1024, morphobifpn, resunet` only. Maps for
   `bifpn_resunet` and `morpho_resunet` are absent, so items 2's ensembles cannot be rebuilt
   from cached predictions — the two checkpoints would have to be re-run.
   (A+B+ResUNet *is* rebuildable: all three of its members have maps.)

4. **The exact Figure 1 script.** The paper's Fig. 1 panel (d) is the
   A+B+**BiFPN-ResUNet** ensemble (Dice 0.8380). The closest script on disk,
   `code/figures/gen_figures_pfe_report.py::fig8_av_qualitative`, renders the same test image
   (092_N) but with the A+B+**C** ensemble (0.8320), in French, and as a 4-panel PDF for the
   PFE report. It is a starting point, not the figure that was submitted.

## Non-blocking — recommended for a reproducible release

5. **Dataset preparation script.** `data/fundus-avseg-592/` is the finished 70/10/20 split with
   `images/`, `annotation/` and `npy/` per split, but nothing on disk shows how it was built
   from the raw `Fundus-AVSeg.zip` (in `~/Téléchargements`) — the split assignment, the resize,
   and the RGB→class mapping live only inside each training notebook's `ImageDataset`.

6. **`environment.yml` / `requirements.txt` and version pins.** No dependency manifest exists.
   The paper states PyTorch on an NVIDIA Quadro RTX 8000, but torch/CUDA versions are not
   recorded anywhere. (A `.venv/` exists in the parent workspace but was not copied here;
   `../binary/requirements.txt` belongs to the earlier binary-segmentation work and is unpinned
   for this paper, but is the only manifest in the workspace and could serve as a starting point.)

7. **Seed-42 run artifacts for the two ResUNet-backbone extensions.** `morpho_resunet_history.csv`
   and `morpho_resunet_log.txt` were copied, and BiFPN-ResUNet has a history CSV + diagnostics
   PNG, but there is no per-seed reproduction for either model — consistent with the paper's own
   limitation that these results are single-seed.

8. **Per-image scores for the GAVE2/FFA second-dataset check.** `docs/06_second_dataset.md`
   reports only aggregate val Dice; per-image metrics were never saved, and the challenge test
   set has no local ground truth. Relevant only to the §4 future-work sentence.

9. **Standard U-Net per-epoch history.** The baseline was trained before the fixed-592 protocol
   (final-epoch checkpoint, no per-epoch tracking), so only
   `training_diagnostics_standard_unet.png` exists — no `training_history_standard_unet.csv`.

## Deliberately not copied (available in the parent workspace)

- `../MICAI/` — GAVE2/FFA challenge work (2.4 GB). Only referenced as future work in §4;
  `docs/06_second_dataset.md` summarises it.
- `../binary/`, `../drive_kaggle_prepared_data/`, `../multi_step_back/{BiFPN_WCE,Morpho*}.ipynb`,
  LIOT and RGB notebook variants — earlier binary-segmentation and DRIVE work, not part of this paper.
- `../pfe_report/` beyond the three figures and `gen_figures.py` that overlap with the paper.
- `../paper_fixes.zip`, `../paper_fixes_lean.zip`, `../new_models.zip` — archives of material
  already copied here in unpacked form.
