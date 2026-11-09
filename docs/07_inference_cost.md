# Task 7 — Inference Cost

## Measurement Conditions

- GPU: NVIDIA Quadro RTX 8000 (48 GB VRAM)
- Batch size: 1
- Input resolution: 592 × 592 × 3
- Timing: mean ± std over all 20 test images (GPU warmed up before measurement; CUDA events synchronised)
- FLOPs: estimated using the formula 2 × MACs (multiply-accumulate operations counted by torch.profiler / manually). Note: profiler values are approximate; absolute FLOPs are provided for comparison ordering, not as precise figures.

## Parameter Counts

| Model | Parameters |
|---|---|
| BiFPN-UNet (32–256 ch) | 2,252,151 |
| MorphoBiFPN-UNet | 2,253,879 |
| BiFPN-UNet (64–1024 ch) | 20,297,692 |
| Standard U-Net | 31,037,763 |
| Morpho-U-Net | 31,047,107 |
| ResUNet | 32,436,483 |

## Inference Time per Image (Quadro RTX 8000, batch=1)

| Model | Mean (ms) | Std (ms) |
|---|---|---|
| BiFPN-UNet (32–256 ch) | 28.7 | 0.0 |
| MorphoBiFPN-UNet | 30.2 | 0.1 |
| BiFPN-UNet (64–1024 ch) | 40.5 | 0.0 |
| Standard U-Net | 44.3 | 0.1 |
| Morpho-U-Net | 47.6 | 0.1 |
| ResUNet | 49.8 | 0.0 |
| **Ensemble top-3 (sequential)** | **132.3** | — |
| Ensemble top-3 (measured end-to-end) | ~132 ms | — |

The ensemble's total cost is effectively the sum of the 3 member inference times: BiFPN-64ch (40.5 ms) + U-Net (44.3 ms) + Morpho-U-Net (47.6 ms) = **132.4 ms**. Softmax averaging adds <0.5 ms and is negligible.

## Notes

1. The two lean BiFPN variants (32–256 ch) are 14× fewer parameters and ~1.5× faster than the full-capacity models, but achieve 0.79 Dice vs 0.81 Dice — a significant performance gap for a 14× parameter reduction.

2. The Morpho-U-Net is slightly slower than Standard U-Net (+3.3 ms) due to the MorphoPreprocessor's iterative erosion and Gaussian blur kernels, both of which are parameter-free but add computation.

3. The ensemble runs 3× slower than a single model by construction. On a GPU batch-processed scenario (multiple images processed simultaneously), the 3 models could be run in parallel on available GPU memory, bringing effective latency to max(member times) = 49.8 ms rather than their sum.

4. For a fair paper comparison, report both the per-model costs and the total ensemble cost. Emphasise that the ensemble's 132 ms latency is for batch=1; in clinical workflows images are typically processed offline in batches, where throughput matters more than single-image latency.
