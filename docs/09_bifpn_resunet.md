# Task 9 — BiFPN-ResUNet Results

## 9.1 Architecture

BiFPN-ResUNet combines two ideas from the paper:

- **Residual encoder** (from ResUNet): five residual blocks at resolutions 592→296→148→74→37,
  producing feature maps of 64→128→256→512→1024 channels.
- **BiFPN neck** (from BiFPN-UNet): all five encoder maps projected to 64 channels,
  then refined by one BiFPN pass (top-down + bottom-up with learned fast-norm weights).
- **Dual skip decoder**: each decoder stage receives *both* the BiFPN-fused skip (64 ch)
  *and* the raw encoder feature at that scale — concatenated together before the ConvBlock.
  This preserves high-resolution spatial detail from the residual encoder
  alongside BiFPN's multi-scale context.
- **Output**: 1×1 convolution, 64→3 classes.

Total parameters: **22,140,956** (~22 M).

Data flow:

```
Input (3ch)
  → ResEncoder [64,128,256,512,1024 ch] → encoder features e0…e4
  → proj (1×1) → p0…p4 (all 64ch)
  → BiFPN → fused features f0…f4 (64ch)
  → center ConvBlock: cat(f4, e4) [64+1024=1088ch] → 64ch
  → UpBlock×4: cat(upsample, f_i, e_i) → 64ch each
  → final_conv → 3ch logits
```

Compared to standard BiFPN-UNet (which uses only BiFPN skips in the decoder),
the dual encoder skip gives the decoder direct access to the full-capacity
encoder features, avoiding the information bottleneck of the 1024→64 projection.

## 9.2 Results (20-image test set, fixed-592, seed-42)

| Model | Vessel Dice | Vessel IoU | Artery Dice | Vein Dice |
|---|---|---|---|---|
| **BiFPN-ResUNet** | **0.8147** | **0.6889** | **0.8097** | **0.8196** |
| ResUNet (paper) | 0.8103 | 0.6828 | 0.7988 | 0.8218 |
| BiFPN-UNet 64ch (paper) | 0.8092 | 0.6812 | 0.7976 | 0.8208 |
| Standard U-Net (paper) | 0.8023 | 0.6719 | 0.7915 | 0.8130 |
| Morpho-U-Net (paper) | 0.7917 | 0.6572 | 0.7703 | 0.8131 |

**BiFPN-ResUNet is the best single model** on the test set at seed-42,
exceeding ResUNet by +0.0044 and BiFPN-64ch by +0.0055 in vessel Dice.

Per-image vessel Dice: mean=0.8147, std=0.0369, min=0.6985, max=0.8693.

Full per-image scores: `data/bifpn_resunet_scores.csv`.

## 9.3 Why It Wasn't in the Paper

This checkpoint exists (`multi_step_back/BiFPN/models/best_model_bifpn_resunet_dual_592.pth`)
and was trained with the same fixed-592 protocol, but was not included in the
paper's comparison table or ensemble selection. The reason is likely that it was
an exploratory experiment conducted after the main comparison table was finalised.

## 9.4 Ensemble Implications

The current top-3 ensemble (BiFPN-64ch + U-Net + Morpho-U-Net, Dice=0.8320) was
selected by val-set Dice before this model was evaluated. If BiFPN-ResUNet had
been included as a candidate, it would likely have displaced one of the weaker members.

**Recommended action for revision:** Evaluate BiFPN-ResUNet on the val set and
check whether including it improves the ensemble. If so, report it as an updated
model in Table 1 and update the ensemble selection procedure.

*Note: a simpler BiFPN-ResUNet (without the raw encoder skips, 20.9M params) also
exists; it scores 0.8049 Dice — similar to U-Net and below BiFPN-64ch.*

## 9.5 Comparison with BiFPN-UNet 64ch (paper model)

| Difference | BiFPN-UNet 64ch | BiFPN-ResUNet |
|---|---|---|
| Encoder block | Double-Conv (no residual) | Residual block (skip connection) |
| Encoder channels | 64→128→256→512→1024 | 64→128→256→512→1024 |
| Decoder skip | BiFPN-fused (64ch) only | BiFPN-fused + raw encoder (up to 1088ch) |
| Parameters | 20.3M | 22.1M |
| Vessel Dice (seed-42) | 0.8092 | **0.8147** |

The residual encoder and preserved raw skip connections together account for the
+0.006 gain. This is analogous to the ResUNet vs U-Net comparison (+0.008),
where residual connections consistently help in the encoder.

## 9.6 Note on Naming

The checkpoint filename contains "dual" (`best_model_bifpn_resunet_dual_592.pth`).
In the paper and this package, this model is referred to simply as **BiFPN-ResUNet**
to describe the architecture unambiguously: a BiFPN neck on a residual encoder.
The "dual skip" property (BiFPN + raw encoder skip in the decoder) is an architectural
detail described in §9.1, not a separate model family.
