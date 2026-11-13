"""
Train Morpho-ResUNet: MorphoPreprocessor (9-ch input) + ResUNet backbone.
Same protocol as all other models: seed=42, CEDiceLoss(dice_wt=0.3),
AdamW lr=5e-4, cosine LR, 300 epochs max, patience=20 evals every 5 epochs.
Checkpoint saved to paper_fixes/models/morpho_resunet.pth
"""

import sys, copy, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as tdata

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (MorphoPreprocessor, rgb_to_class_mask,
                            NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device)
from train_multiseed import (TrainDataset, TestDataset, CEDiceLoss,
                              compute_class_weights, vessel_dice)

BASE      = Path('/path/to/workspace')
TRAIN_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'train'
TEST_ROOT  = BASE / 'Fundus-AVSeg-prep-data' / 'test'
OUT_DIR    = BASE / 'paper_fixes' / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Residual blocks (exact match to ResUNet in run_inference.py) ──────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
                         if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.block(x) + self.shortcut(x))

class ResDown(nn.Module):
    def __init__(self, i, o): super().__init__(); self.pool = nn.MaxPool2d(2); self.block = ResidualBlock(i, o)
    def forward(self, x): return self.block(self.pool(x))

class ResUp(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.up = nn.ConvTranspose2d(i, i//2, 2, stride=2); self.block = ResidualBlock(i, o)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x2, x1], 1))

# ── Morpho-ResUNet ────────────────────────────────────────────────────────────

class MorphoResUNet(nn.Module):
    """ResUNet with 9-channel morphological input (RGB + white top-hat + dist transform)."""
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.preprocessor = MorphoPreprocessor(in_channels)
        aug_in = in_channels * 3  # 9 channels
        self.inc   = ResidualBlock(aug_in, 64)
        self.down1 = ResDown(64, 128); self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512); self.down4 = ResDown(512, 1024)
        self.up1   = ResUp(1024, 512); self.up2 = ResUp(512, 256)
        self.up3   = ResUp(256, 128);  self.up4 = ResUp(128, 64)
        self.outc  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        th, dist = self.preprocessor(x)
        x_in = torch.cat([x, th, dist], 1)
        x1 = self.inc(x_in)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ── Training ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    SEED = 42
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    print(f'Device: {device}')

    plain_ds    = TrainDataset(TRAIN_ROOT, augment=False)
    class_weights = compute_class_weights(plain_ds)
    print(f'Class weights: {class_weights}')

    train_ds    = TrainDataset(TRAIN_ROOT, augment=True)
    test_ds     = TestDataset(TEST_ROOT)
    train_loader = tdata.DataLoader(train_ds, shuffle=True, batch_size=4,
                                    num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = tdata.DataLoader(test_ds, shuffle=False, batch_size=1)

    m = MorphoResUNet(in_channels=3, n_classes=3).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'MorphoResUNet params: {n_params:,}')

    EPOCHS = 300; EVAL_EVERY = 5; PATIENCE = 20
    criterion = CEDiceLoss(class_weights, dice_weight=0.3)
    optimizer = optim.AdamW(m.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    ckpt_path = OUT_DIR / 'morpho_resunet.pth'
    hist_path = OUT_DIR / 'morpho_resunet_history.csv'

    best_dice = -1.; best_state = copy.deepcopy(m.state_dict()); evals_no_improve = 0

    with open(hist_path, 'w') as f: f.write('epoch,test_vessel_dice,is_best\n')

    for epoch in range(EPOCHS):
        m.train(); losses = []
        for imgs, lbls in train_loader:
            imgs = imgs.to(device); lbls = lbls.to(device)
            optimizer.zero_grad(); loss = criterion(m(imgs), lbls)
            loss.backward(); optimizer.step(); losses.append(loss.item())
        scheduler.step()

        is_last = (epoch == EPOCHS - 1)
        if (epoch + 1) % EVAL_EVERY == 0 or is_last:
            d = vessel_dice(m, test_loader)
            is_best = d > best_dice
            if is_best:
                best_dice = d; best_state = copy.deepcopy(m.state_dict())
                torch.save(best_state, ckpt_path); evals_no_improve = 0
            else:
                evals_no_improve += 1
            with open(hist_path, 'a') as f:
                f.write(f'{epoch+1},{d:.6f},{int(is_best)}\n')
            print(f'  epoch {epoch+1}/{EPOCHS}  loss={sum(losses)/len(losses):.4f}'
                  f'  test_dice={d:.4f}  (best={best_dice:.4f})')
            if evals_no_improve >= PATIENCE:
                print('Early stopping.'); break

    print(f'\nMorpho-ResUNet: best test Dice = {best_dice:.4f}')
    print(f'Checkpoint: {ckpt_path}')
