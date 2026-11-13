"""
Task 2 — Statistical significance analysis.
Aggregates per-image Dice from seed-42 (paper_fixes/data/per_image_scores.csv)
and from new seeds (paper_fixes/models/seed_{1,7,13}/{model}.pth checkpoints).
Computes:
  - Mean ± std across seeds for each model and ensemble
  - Bootstrap 95% CIs (1000 reps)
  - Wilcoxon signed-rank tests (top-3 ensemble vs each member, vs ResUNet baseline)
Writes paper_fixes/02_statistical_significance.md
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils import data as tdata

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (StandardUNet, MorphoUNet, BiFPNUNet, ResUNet,
                            rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device)
from train_multiseed import TestDataset, MODEL_CONFIGS

BASE       = Path('/path/to/workspace')
TEST_ROOT  = BASE / 'Fundus-AVSeg-prep-data' / 'test'
NEW_SEEDS  = [1, 7, 13]

# seed-42 checkpoint paths (original training, from run_inference.py CKPT dict)
SEED42_CKPTS = {
    'unet':           BASE / 'multi_step_back/unet/models/best_model_standard_unet.pth',
    'morpho_unet':    BASE / 'multi_step_back/morpho/models/best_model_morpho_unet_tophat_dist_592.pth',
    'bifpn_64to1024': BASE / 'multi_step_back/BiFPN/models/best_model_bifpn_64to1024_592.pth',
    'resunet':        BASE / 'multi_step_back/unet/models/best_model_resunet_592.pth',
}


def eval_per_image(model, test_ds):
    """Returns array of shape (N,) with per-image vessel Dice."""
    model.eval()
    scores = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, lbl_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            lbl_t = lbl_t.to(device)
            preds = torch.argmax(torch.softmax(model(img_t), 1), 1).squeeze(0)
            vessel_dice = 0.
            for c in VESSEL_CLASSES:
                p = (preds == c).float(); l = (lbl_t == c).float()
                tp = float((p * l).sum())
                fp = float((p * (1 - l)).sum())
                fn = float(((1 - p) * l).sum())
                vessel_dice += 2 * tp / (2 * tp + fp + fn + 1e-6)
            scores.append(vessel_dice / len(VESSEL_CLASSES))
    return np.array(scores)


def load_model(model_name, seed):
    cfg = MODEL_CONFIGS[model_name]
    m = cfg['cls'](**cfg['kwargs']).to(device)
    if seed == 42:
        ckpt_path = SEED42_CKPTS[model_name]
    else:
        ckpt_path = BASE / 'paper_fixes' / 'models' / f'seed_{seed}' / f'{model_name}.pth'
    if not ckpt_path.exists():
        return None, str(ckpt_path)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    return m, None


def bootstrap_ci(arr, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(0)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 + ci) / 2)
    return np.mean(arr), lo, hi


def soft_ensemble_dice(models, test_ds):
    """Average softmax from multiple models then compute per-image Dice."""
    for m in models: m.eval()
    scores = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, lbl_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            lbl_t = lbl_t.to(device)
            avg_soft = sum(torch.softmax(m(img_t), 1) for m in models) / len(models)
            preds = torch.argmax(avg_soft, 1).squeeze(0)
            vessel_dice = 0.
            for c in VESSEL_CLASSES:
                p = (preds == c).float(); l = (lbl_t == c).float()
                tp = float((p * l).sum())
                fp = float((p * (1 - l)).sum())
                fn = float(((1 - p) * l).sum())
                vessel_dice += 2 * tp / (2 * tp + fp + fn + 1e-6)
            scores.append(vessel_dice / len(VESSEL_CLASSES))
    return np.array(scores)


if __name__ == '__main__':
    test_ds = TestDataset(TEST_ROOT)
    all_seeds = [42] + NEW_SEEDS
    model_names = ['unet', 'morpho_unet', 'bifpn_64to1024', 'resunet']

    # ── 1. Per-image Dice for each model × seed ───────────────────────────────
    per_image = {}
    missing = []
    for mname in model_names:
        per_image[mname] = {}
        for seed in all_seeds:
            m, err = load_model(mname, seed)
            if m is None:
                missing.append(f'{mname} seed={seed}: {err}')
                continue
            print(f'  Evaluating {mname} seed={seed} ...')
            per_image[mname][seed] = eval_per_image(m, test_ds)

    if missing:
        print('\nMissing checkpoints:')
        for x in missing: print(f'  {x}')
        if any(s in [1, 7, 13] for s in
               [seed for mname in model_names for seed in NEW_SEEDS
                if seed not in per_image.get(mname, {})]):
            print('\nRun train_multiseed.py first, then re-run this script.')

    # ── 2. Ensemble per-image Dice for each seed ─────────────────────────────
    top3_names = ['unet', 'morpho_unet', 'bifpn_64to1024']
    ensemble_per_image = {}
    for seed in all_seeds:
        models = []
        for mname in top3_names:
            m, err = load_model(mname, seed)
            if m is not None: models.append(m)
        if len(models) == 3:
            print(f'  Evaluating ensemble seed={seed} ...')
            ensemble_per_image[seed] = soft_ensemble_dice(models, test_ds)

    # ── 3. Summary statistics ─────────────────────────────────────────────────
    rows = []
    for mname in model_names:
        available = [per_image[mname][s] for s in all_seeds if s in per_image[mname]]
        if not available: continue
        all_arr = np.array([a.mean() for a in available])
        per_20 = np.stack(available, axis=0)  # seeds × 20 images
        mean_arr = per_20.mean(axis=0)  # mean across seeds, per image
        mean, lo, hi = bootstrap_ci(mean_arr)
        rows.append({
            'entity':       mname,
            'n_seeds':      len(available),
            'mean_dice':    all_arr.mean(),
            'std_dice':     all_arr.std(ddof=1) if len(all_arr) > 1 else 0.,
            'ci_lo':        lo,
            'ci_hi':        hi,
            'seed_dices':   all_arr.tolist(),
        })

    # Ensemble
    if ensemble_per_image:
        ens_arr = np.array([v.mean() for v in ensemble_per_image.values()])
        all_ens = np.stack(list(ensemble_per_image.values()), axis=0).mean(axis=0)
        mean, lo, hi = bootstrap_ci(all_ens)
        rows.append({
            'entity':      'top3_ensemble',
            'n_seeds':     len(ensemble_per_image),
            'mean_dice':   ens_arr.mean(),
            'std_dice':    ens_arr.std(ddof=1) if len(ens_arr) > 1 else 0.,
            'ci_lo':       lo,
            'ci_hi':       hi,
            'seed_dices':  ens_arr.tolist(),
        })

    sumdf = pd.DataFrame(rows)
    sumdf.to_csv(BASE / 'paper_fixes' / 'data' / 'multiseed_summary.csv', index=False)

    # ── 4. Wilcoxon tests: ensemble vs each member (using seed-mean per image) ─
    wilcox = []
    if 'top3_ensemble' in sumdf['entity'].values and ensemble_per_image:
        ens_mean_per_image = np.stack(list(ensemble_per_image.values()), axis=0).mean(axis=0)
        for mname in top3_names + ['resunet']:
            if mname not in per_image or not per_image[mname]: continue
            model_mean_per_image = np.stack(
                [per_image[mname][s] for s in all_seeds if s in per_image[mname]], axis=0
            ).mean(axis=0)
            stat, p = stats.wilcoxon(ens_mean_per_image, model_mean_per_image,
                                     alternative='greater')
            wilcox.append({'comparison': f'ensemble > {mname}', 'stat': stat, 'p_value': p,
                           'significant': p < 0.05})
    wxdf = pd.DataFrame(wilcox)

    # ── 5. Same-architecture ensemble (3×BiFPN-64) ─────────────────────────────
    same_arch_models = []
    same_arch_seeds_found = []
    for seed in [1, 7, 13]:
        m, err = load_model('bifpn_64to1024', seed)
        if m is not None:
            same_arch_models.append(m); same_arch_seeds_found.append(seed)
    if len(same_arch_models) == 3:
        print('  Evaluating same-architecture ensemble (BiFPN-64 seeds 1,7,13) ...')
        same_arch_scores = soft_ensemble_dice(same_arch_models, test_ds)
        same_arch_mean = same_arch_scores.mean()
        same_arch_ci   = bootstrap_ci(same_arch_scores)
    else:
        same_arch_mean = None; same_arch_ci = None

    # ── 6. Write markdown ─────────────────────────────────────────────────────
    md_path = BASE / 'paper_fixes' / '02_statistical_significance.md'
    with open(md_path, 'w') as f:
        f.write('# Task 2 — Statistical Significance\n\n')
        f.write(f'Seeds evaluated: {all_seeds}\n\n')

        f.write('## 2.1 Per-Model Mean ± Std and 95 % Bootstrap CI\n\n')
        f.write('| Model | Seeds | Mean Dice | Std | 95% CI |\n')
        f.write('|---|---|---|---|---|\n')
        for _, r in sumdf.iterrows():
            seed_str = ', '.join(f'{d:.4f}' for d in r['seed_dices'])
            f.write(f"| {r['entity']} | {r['n_seeds']} | {r['mean_dice']:.4f} | "
                    f"{r['std_dice']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |\n")
        f.write(f'\nPer-seed values by model:\n')
        for _, r in sumdf.iterrows():
            f.write(f"  - {r['entity']}: {r['seed_dices']}\n")

        f.write('\n## 2.2 Wilcoxon Signed-Rank Tests (ensemble > baseline, one-tailed)\n\n')
        if not wxdf.empty:
            f.write('| Comparison | Statistic | p-value | Significant? |\n')
            f.write('|---|---|---|---|\n')
            for _, r in wxdf.iterrows():
                sig = 'YES (p<0.05)' if r['significant'] else 'no'
                f.write(f"| {r['comparison']} | {r['stat']:.1f} | {r['p_value']:.4f} | {sig} |\n")
        else:
            f.write('(Wilcoxon test requires all seeds to be available — re-run after training.)\n')

        f.write('\n## 2.3 Same-Architecture Ensemble Control\n\n')
        f.write('To distinguish ensemble gain from diversity vs. reduced variance, '
                'we also ensemble 3 runs of the same architecture (BiFPN-64ch) '
                'using different random seeds (1, 7, 13).\n\n')
        if same_arch_mean is not None:
            _, lo, hi = same_arch_ci
            f.write(f'**3×BiFPN-64 ensemble (seeds 1,7,13):** '
                    f'Dice = {same_arch_mean:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]\n\n')
            bifpn_single = sumdf.loc[sumdf.entity=='bifpn_64to1024','mean_dice'].values
            if len(bifpn_single):
                gain = same_arch_mean - bifpn_single[0]
                f.write(f'Gain over mean single BiFPN-64: {gain:+.4f}\n\n')
            # Compare to heterogeneous ensemble
            ens_mean = sumdf.loc[sumdf.entity=='top3_ensemble','mean_dice'].values
            if len(ens_mean):
                diff = ens_mean[0] - same_arch_mean
                f.write(f'Heterogeneous ensemble minus same-arch ensemble: {diff:+.4f}\n\n')
                f.write('**Interpretation:** '
                        + ('The heterogeneous ensemble adds significant gain beyond '
                           'what same-architecture variance reduction alone achieves, '
                           'suggesting some complementarity despite high Q-statistics.'
                           if diff > 0.003 else
                           'The heterogeneous ensemble provides minimal additional gain '
                           'over same-architecture ensembling, confirming that the '
                           'primary benefit is variance reduction rather than '
                           'diverse error patterns (consistent with Q-statistics ≈ 0.998).')
                        + '\n')
        else:
            f.write('(Same-architecture checkpoints not yet available — re-run after training.)\n')

        f.write('\n## 2.4 Notes\n\n')
        f.write('- Bootstrap CIs computed on the mean-across-seeds per-image Dice (1000 reps, seed=0).\n')
        f.write('- Wilcoxon test is one-tailed (ensemble > baseline), paired on the 20 test images.\n')
        f.write('- Seed-42 checkpoints are from the original paper training; '
                'seeds 1, 7, 13 trained identically (same protocol, different initialization).\n')
        if missing:
            f.write('\n### Missing checkpoints at time of analysis:\n')
            for x in missing: f.write(f'  - {x}\n')

    print(f'\nWrote {md_path}')
    print('\nSummary table:')
    print(sumdf[['entity','mean_dice','std_dice','ci_lo','ci_hi']].to_string(index=False, float_format='%.4f'))
    if not wxdf.empty:
        print('\nWilcoxon tests:')
        print(wxdf.to_string(index=False, float_format='%.4f'))
