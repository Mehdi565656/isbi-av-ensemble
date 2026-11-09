# Task 2 — Statistical Significance

Seeds evaluated: [42, 1, 7, 13]

## 2.1 Per-Model Mean ± Std and 95 % Bootstrap CI

| Model | Seeds | Mean Dice | Std | 95% CI |
|---|---|---|---|---|
| unet | 4 | 0.8082 | 0.0045 | [0.7913, 0.8228] |
| morpho_unet | 4 | 0.8017 | 0.0070 | [0.7833, 0.8172] |
| bifpn_64to1024 | 4 | 0.8077 | 0.0037 | [0.7918, 0.8229] |
| resunet | 4 | 0.8129 | 0.0022 | [0.7956, 0.8274] |
| top3_ensemble | 4 | 0.8242 | 0.0056 | [0.8085, 0.8385] |

Per-seed values by model:
  - unet: [0.802281178731557, 0.8129381396378201, 0.8077316960238022, 0.8097995551763052]
  - morpho_unet: [0.79168916789854, 0.8070117649689628, 0.8022945854375141, 0.8058761761594715]
  - bifpn_64to1024: [0.809217266561503, 0.8114823312313255, 0.802687421173173, 0.8073394206643462]
  - resunet: [0.8103103639478624, 0.8141230520069274, 0.8120143309300765, 0.8153121996119765]
  - top3_ensemble: [0.8320246641646667, 0.8240876418443737, 0.818991608067182, 0.821848635376097]

## 2.2 Wilcoxon Signed-Rank Tests (ensemble > baseline, one-tailed)

| Comparison | Statistic | p-value | Significant? |
|---|---|---|---|
| ensemble > unet | 210.0 | 0.0000 | YES (p<0.05) |
| ensemble > morpho_unet | 210.0 | 0.0000 | YES (p<0.05) |
| ensemble > bifpn_64to1024 | 210.0 | 0.0000 | YES (p<0.05) |
| ensemble > resunet | 208.0 | 0.0000 | YES (p<0.05) |

## 2.3 Same-Architecture Ensemble Control

To distinguish ensemble gain from diversity vs. reduced variance, we also ensemble 3 runs of the same architecture (BiFPN-64ch) using different random seeds (1, 7, 13).

**3×BiFPN-64 ensemble (seeds 1,7,13):** Dice = 0.8203  [95% CI: 0.8038, 0.8352]

Gain over mean single BiFPN-64: +0.0127

Heterogeneous ensemble minus same-arch ensemble: +0.0039

**Interpretation:** The same-architecture ensemble (0.8203) captures most of the heterogeneous ensemble's gain. The additional margin (+0.004) is smaller than the ensemble's seed-to-seed std (±0.0056) and is **not clearly significant** — most of the benefit is variance reduction (consistent with Q≈0.998), not architectural diversity. The paper's "diverse architectures" framing overstates what the data supports.

Note: the paper's reported result (0.8320) is the seed-42 ensemble, which is the strongest of the 4 seeds. The mean across all 4 seeds (0.8242) is the more representative estimate of expected performance.

## 2.4 Notes

- Bootstrap CIs computed on the mean-across-seeds per-image Dice (1000 reps, seed=0).
- Wilcoxon test is one-tailed (ensemble > baseline), paired on the 20 test images.
- Seed-42 checkpoints are from the original paper training; seeds 1, 7, 13 trained identically (same protocol, different initialization).
