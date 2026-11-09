# Task 8 — SOTA Comparison

## 8.1 Scope of Search

Literature search (2024–2026) for retinal artery-vein (AV) segmentation papers. Focus:
1. Methods reporting results **on Fundus-AVSeg** (same test set, most directly comparable)
2. Methods on other public AV datasets (RITE, LES-AV, HRF, AV-DRIVE) — note comparability caveats
3. Transformer-based or foundation-model-based methods that might exceed the ensemble

---

## 8.2 Fundus-AVSeg — Same Dataset (Direct Comparison)

The dataset was published in June 2025 (Scientific Data vol. 12). Very few external papers have had time to adopt it as a benchmark.

### Baselines from the Fundus-AVSeg dataset paper (Raza et al., 2025)

These are the 4 baselines **already in Table 1** of the submitted paper:

| Model | Vessel Dice (DSC) | Vessel IoU | Notes |
|---|---|---|---|
| U-Net | 0.802 | 0.672 | Standard encoder-decoder |
| U-Net++ | ~0.795 | ~0.665 | Nested skip connections |
| Attention U-Net | ~0.793 | ~0.660 | Spatial attention gates |
| ResUNet | **~0.810** | **~0.683** | Residual encoder; best single baseline |

Source: Table 4 in "A Fundus Image Dataset for AI-based Artery-Vein Vessel Segmentation," *Scientific Data* 12 (2025). [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12297265/)

**The paper's top-3 soft-voting ensemble (0.8320 Dice, 0.7139 IoU) is the strongest reported result on Fundus-AVSeg.**

### External papers on Fundus-AVSeg

**VascX (2024, arXiv:2409.16016):** Ensemble approach for retinal vascular analysis from colour fundus images. Results reported on DRIVE, STARE, and proprietary datasets — not on Fundus-AVSeg.

**arXiv:2510.03717 (Attention-WNet, October 2025):** AV segmentation using an attention-augmented WNet. Evaluated on DRIVE and HRF only, not on Fundus-AVSeg. Not directly comparable.

**Conclusion:** No external paper has published results on the Fundus-AVSeg test protocol to date (as of search date). The paper is the only source of benchmark numbers for this dataset.

---

## 8.3 RITE Dataset (Different Dataset — Protocol Gap)

RITE (derived from AV-DRIVE) is the most commonly used AV segmentation benchmark. Results below use AV classification accuracy / Dice on the vessel pixels only (not the 3-class soft-voting task used here).

| Method | Dataset | Metric | Score | Comparable? |
|---|---|---|---|---|
| Channel Coupling (Zhu et al., 2025) | RITE | AV accuracy | 96.77% | ✗ Different metric |
| Channel Coupling (Zhu et al., 2025) | RITE | BV IoU | 98.56% | ✗ All-vessel, not AV separately |
| LVS-Net (2024, arXiv:2412.05968) | RITE | Dice (vessel) | 81.34% | ✗ Binary vessel, not 3-class |
| RetinaLiteNet (CVPR 2024W) | DRIVE | AV F1 (artery/vein) | 76.2% / 79.7% | ✗ DRIVE, binary AV |
| One-shot cross-modality (2024, PMC) | RITE | Dice | 0.585–0.773 | ✗ One-shot, 6 datasets |

**Protocol mismatch warning:** RITE measures AV classification accuracy (among detected vessel pixels) whereas this paper measures 3-class Dice (background/artery/vein) over the full image. Numbers cannot be directly compared; RITE AV accuracy of ~96% is not equivalent to Dice 0.83 on Fundus-AVSeg.

---

## 8.4 Transformer and Foundation Models

No transformer or vision-foundation-model paper has reported results specifically on Fundus-AVSeg. On adjacent datasets:

- **Transformer-UNet hybrids** (various 2024 papers on DRIVE/HRF/STARE): typically improve binary vessel Dice by 1–3% over U-Net, reaching ~0.82 Dice on DRIVE, but these are all-vessel (not AV) benchmarks.
- **SAM/segment-anything adaptation for fundus:** Several preprints exist (2024–2025) but none evaluate on Fundus-AVSeg or a directly comparable AV protocol.

---

## 8.5 Summary Assessment

| Claim | Evidence |
|---|---|
| Ensemble beats all published single-model baselines on Fundus-AVSeg | ✓ Confirmed (0.8320 vs 0.810 ResUNet) |
| Ensemble competitive with SOTA on AV segmentation generally | Unclear — no apples-to-apples comparison possible |
| Transformer/foundation models not yet evaluated on Fundus-AVSeg | ✓ Confirmed (no external papers found) |
| RITE/HRF results comparable to Fundus-AVSeg results | ✗ Not comparable — different dataset, different metric |

**Recommendation for paper rewrite:** The paper cannot make a broad SOTA claim because Fundus-AVSeg is too new to have attracted many external comparisons. The defensible claim is:
- "Our ensemble achieves the highest reported Dice on the Fundus-AVSeg benchmark" (true as of submission)
- For Reviewer P88i's comparison request: add a paragraph noting the protocol gap with RITE and explaining why cross-dataset numbers are not directly comparable; optionally include a RITE/HRF number table with an explicit "not directly comparable" note

---

## 8.6 Candidate Citations for Paper

| Reference | Dataset | Best metric | Comparability |
|---|---|---|---|
| Raza et al. 2025, *Sci. Data* | Fundus-AVSeg | ResUNet Dice ~0.810 | ✓ Direct (already in Table 1) |
| Zhu et al. 2025, arXiv:2508.03738 | RITE/LES-AV/HRF | AV acc 96.77%, IoU 98.56% | ✗ Different metric/dataset |
| LVS-Net 2024, arXiv:2412.05968 | RITE | Vessel Dice 81.34% | ✗ Binary vessel, RITE |
| Mehmood et al. 2024, CVPR Workshop | DRIVE | AV F1 76–80% | ✗ Binary AV, DRIVE |
| Cross-modality pretraining 2024, *Ophthalmology Science* | RITE, 5 others | Dice 58–77% | ✗ One-shot, 6 datasets |
