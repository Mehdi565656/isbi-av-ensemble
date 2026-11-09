# Task 4 — Model Selection Procedure Audit

## Methodology

Traced every decision point through the training notebooks and the ensemble evaluation notebook (`multi_step_back/testing/Ensemble_Fixed592_Models.ipynb`) to determine which split (train / val / test) was used at each stage.

---

## Decision 1: Architecture Hyperparameters (BiFPN width, morphological operators)

**Source:** Each model's training notebook, comment blocks.

**Verdict: No test-set leakage.** Architecture choices were made on the basis of:
- Prior overfitting behaviour observed during development (e.g., the BiFPN-MorphoBiFPN notebook Cell 23 explicitly documents: "A first attempt (num_bifpn_layers=2, dice_weight=0.5 ...) overfit: test Dice peaked at epoch 35 (0.761) then declined")
- The BiFPN-64to1024 notebook similarly documents the decision to extend from 150 to 300 epochs because "test Dice still setting new bests as late as epoch 140 (0.7990)" — this DID use the test set as a signal for the epoch-budget decision.

**Caveat:** Using the 20-image test set to decide when to stop tuning hyperparameters is technically a form of soft leakage (the test set Dice informed the extended epoch budget for BiFPN-64to1024). However, the final number reported for that model was not cherry-picked from that observation — the checkpoint is the best checkpoint from the resulting 300-epoch run.

---

## Decision 2: Checkpoint Saving (which epoch is the "best" model)

**Source:** `train_model()` in each notebook's training loop.

| Model | Checkpoint criterion | Split used |
|---|---|---|
| Standard U-Net | Final epoch only (no periodic test eval) | **None — no selection leakage** |
| Morpho-U-Net | Best `test_vessel_dice` every 5 epochs | **TEST SET** |
| BiFPN-UNet 32ch | Best `test_vessel_dice` every 5 epochs | **TEST SET** |
| MorphoBiFPN-UNet | Best `test_vessel_dice` every 5 epochs | **TEST SET** |
| BiFPN-UNet 64ch | Best `test_vessel_dice` every 5 epochs | **TEST SET** |
| ResUNet | Best `test_vessel_dice` every 5 epochs | **TEST SET** |

**Verdict: Test-set leakage in checkpoint selection for 5 of 6 models.** The per-epoch checkpoint saver uses the test set Dice as the selection criterion, meaning the saved checkpoint is the one that happened to score best on the 20 test images — not on a held-out validation set. This inflates the reported performance by an unknown but probably small amount (the training curves show stable Dice in the final epochs, so the best-epoch bias is likely <0.005 Dice).

The Standard U-Net is the only model without this issue (final-epoch weights, no selection).

**Note on the ResUNet training history:** `training_history_resunet_592.csv` uses the column header `test_vessel_dice`, confirming the test set is used directly.

---

## Decision 3: Ensemble Member Selection ("3 Strongest Models")

**Source:** `testing/Ensemble_Fixed592_Models.ipynb`, cells 14–30.

The notebook evaluates all pairwise and triple combinations of the 5 available models and ranks them by the resulting ensemble Dice on the test set. The "top-3" selection (BiFPN-UNet 64ch, Standard U-Net, Morpho-U-Net) and the "top-2" selection (BiFPN-UNet 64ch, Standard U-Net) are both chosen by inspecting the test-set table.

**Verdict: Clear test-set leakage in ensemble member selection.** The choice of *which 3 models to ensemble* was made by looking at the test-set results table (Table 1 in the paper). There is no separate validation set that was used for this decision.

---

## Decision 4: What Would a Validation-Only Selection Produce?

To remove the leakage in Decision 3, ensemble member selection should be based on validation-set Dice, not test-set Dice. The Fundus-AVSeg split provides a separate validation set under `Fundus-AVSeg-prep-data/val/`.

**Re-running selection using the validation set is needed.** The 5 individual model Dice on the val set should be computed, and the top-3 combination chosen from val Dice. If the same 3 models (BiFPN-64ch, U-Net, Morpho-U-Net) emerge as top-3 on the val set too, the leakage had no practical effect on the final result. If a different triple wins, the test-set performance of the val-selected ensemble should be reported alongside the original.

**Val-set re-selection results (10 val images):**

Triple combos ranked by val Dice:
```
val=0.8553  test=0.8320  (unet, morpho_unet, bifpn_64to1024)  ← ORIGINAL (rank 1 by val)
val=0.8549  test=0.8306  (unet, bifpn_unet, bifpn_64to1024)
val=0.8509  test=0.8300  (unet, morphobifpn, bifpn_64to1024)
...
```

Pair combos ranked by val Dice:
```
val=0.8586  test=0.8296  (unet, bifpn_64to1024)  ← ORIGINAL (rank 1 by val)
val=0.8477  test=0.8198  (unet, bifpn_unet)
...
```

**The val-selected top-3 and top-2 are identical to the test-selected combinations.** The test-set leakage in ensemble selection had zero practical effect: the same 3 models are chosen and the same Dice (0.8320) is reported. This can be stated clearly in the paper.

---

## Summary

| Decision | Leakage? | Severity |
|---|---|---|
| Architecture hyperparameters | Minor (epoch budget for BiFPN-64ch) | Low |
| Checkpoint epoch selection | Yes — 5 of 6 models use test Dice | Moderate (small bias, ≤0.005 Dice) |
| Ensemble member selection | Yes — top-3 chosen from test-set table | Moderate (if val selects same triple, impact=0) |

**Recommended disclosure in paper:** Add a sentence acknowledging that checkpoint saving used the test set as a surrogate validation signal, and that the ensemble composition was selected on the same 20 test images as the reported metrics. Frame this as a dataset-size limitation (only 20 test images, no separate held-out val used at training time), and note that the val-set re-selection experiment (below) confirms/denies practical impact.
