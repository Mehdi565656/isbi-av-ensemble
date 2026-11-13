#!/bin/bash
# Run all post-training evaluations and finalize paper_fixes package.
# Execute this AFTER train_multiseed.py and train_morpho_ablation.py complete.

set -e
BASE=/path/to/workspace
cd "$BASE"

echo "=== Step 1: Morpho ablation evaluation ==="
python paper_fixes/eval_morpho_ablation.py

echo "=== Step 2: Statistical significance (all seeds) ==="
python paper_fixes/compute_statistics.py

echo "=== Step 3: Re-generate result figures ==="
python paper_fixes/gen_result_figures.py

echo "=== Step 4: Check all expected files are present ==="
for f in \
    paper_fixes/00_inventory.md \
    paper_fixes/01_resunet_fixed592.md \
    paper_fixes/02_statistical_significance.md \
    paper_fixes/03_morpho_ablation.md \
    paper_fixes/04_model_selection_audit.md \
    paper_fixes/05_diversity_analysis.md \
    paper_fixes/06_second_dataset.md \
    paper_fixes/07_inference_cost.md \
    paper_fixes/08_sota_comparison.md \
    paper_fixes/RESULTS_SUMMARY.md \
    paper_fixes/data/per_image_scores.csv \
    paper_fixes/data/model_summary.csv \
    paper_fixes/figures/per_model_dice_boxplot.png \
    paper_fixes/figures/ensemble_comparison_bar.png \
    paper_fixes/figures/artery_vein_comparison.png \
    paper_fixes/figures/morpho_channels_overview.png \
; do
    if [ -f "$f" ]; then echo "  OK: $f"
    else echo "  MISSING: $f"; fi
done

echo "=== Step 5: Zip package ==="
cd "$BASE"
rm -f paper_fixes.zip
zip -r paper_fixes.zip paper_fixes/ \
    --exclude "paper_fixes/models/*" \
    --exclude "paper_fixes/__pycache__/*"
echo "Created paper_fixes.zip (models directory excluded to keep zip small)"
echo "To include models: zip -r paper_fixes_with_models.zip paper_fixes/"

echo ""
echo "=== DONE ==="
ls -lh paper_fixes.zip
