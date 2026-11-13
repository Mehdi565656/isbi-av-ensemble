"""
Evaluate the two BiFPN-ResUNet checkpoints (single and dual-skip)
that were trained but never included in the paper comparison.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device
from train_multiseed import TestDataset

BASE = Path('/home/mefeki/Bureau/work')
CKPT_DIR = BASE / 'multi_step_back' / 'BiFPN' / 'models'
TEST_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'test'

# ── Architecture (exact match to notebook BiFPN_ResUnet_Fixed592.ipynb) ────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class ResEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, depth=5):
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.blocks.append(ResidualBlock(c, ch))
            self.pools.append(nn.MaxPool2d(2))
            c = ch
        self.out_channels = chs
    def forward(self, x):
        feats = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x); feats.append(x); x = pool(x)
        return feats

class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=1, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.num_levels = len(channels)
        self.num_layers = num_layers
        self.convs_td = nn.ModuleList([ConvBlock(c, c) for c in channels])
        self.convs_bu = nn.ModuleList([ConvBlock(c, c) for c in channels])
        self.w_td = nn.Parameter(torch.ones(num_layers, self.num_levels, 2))
        self.w_bu = nn.Parameter(torch.ones(num_layers, self.num_levels, 3))
    def forward(self, feats):
        P = feats
        for l in range(self.num_layers):
            td = [None] * self.num_levels; td[-1] = P[-1]
            for i in reversed(range(self.num_levels - 1)):
                w  = F.relu(self.w_td[l, i]); w = w / (w.sum() + self.eps)
                up = F.interpolate(td[i+1], size=P[i].shape[-2:], mode='bilinear', align_corners=False)
                td[i] = self.convs_td[i](w[0]*P[i] + w[1]*up)
            bu = [None] * self.num_levels; bu[0] = td[0]
            for i in range(1, self.num_levels):
                w    = F.relu(self.w_bu[l, i]); w = w / (w.sum() + self.eps)
                down = F.max_pool2d(bu[i-1], 2)
                if down.shape[-2:] != td[i].shape[-2:]:
                    down = F.interpolate(down, size=td[i].shape[-2:], mode='bilinear', align_corners=False)
                bu[i] = self.convs_bu[i](w[0]*P[i] + w[1]*td[i] + w[2]*down)
            P = bu
        return P

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class DualSkipUpBlock(nn.Module):
    """Upsample + concat BiFPN skip (64ch) + raw encoder skip (enc_ch) → ConvBlock."""
    def __init__(self, in_ch, bifpn_ch, enc_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + bifpn_ch + enc_ch, out_ch)
    def forward(self, x, bifpn_skip, enc_skip):
        x = self.up(x)
        if x.shape[-2:] != bifpn_skip.shape[-2:]:
            x = F.interpolate(x, size=bifpn_skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, bifpn_skip, enc_skip], dim=1))

class BiFPNResUNetDual(nn.Module):
    def __init__(self, in_channels=3, num_classes=3,
                 base_ch=64, depth=5, bifpn_ch=64, num_bifpn_layers=1):
        super().__init__()
        self.encoder = ResEncoder(in_ch=in_channels, base_ch=base_ch, depth=depth)
        enc_chs = self.encoder.out_channels           # [64,128,256,512,1024]
        self.proj       = nn.ModuleList([nn.Conv2d(c, bifpn_ch, 1) for c in enc_chs])
        self.bifpn      = BiFPN([bifpn_ch] * depth, num_layers=num_bifpn_layers)
        self.center     = ConvBlock(bifpn_ch + enc_chs[-1], bifpn_ch)
        self.up_blocks  = nn.ModuleList([
            DualSkipUpBlock(bifpn_ch, bifpn_ch, enc_chs[depth-2-i], bifpn_ch)
            for i in range(depth-1)
        ])
        self.final_conv = nn.Conv2d(bifpn_ch, num_classes, 1)
    def forward(self, x):
        enc_raw = self.encoder(x)                          # raw encoder feats (multi-ch)
        proj    = [p(f) for p, f in zip(self.proj, enc_raw)]   # projected to 64ch
        fused   = self.bifpn(proj)
        x = self.center(torch.cat([fused[-1], enc_raw[-1]], dim=1))
        for i, up in enumerate(self.up_blocks):
            x = up(x, fused[-2-i], enc_raw[-2-i])
        return self.final_conv(x)

class BiFPNResUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3,
                 base_ch=64, depth=5, bifpn_ch=64, num_bifpn_layers=1):
        super().__init__()
        self.encoder    = ResEncoder(in_ch=in_channels, base_ch=base_ch, depth=depth)
        self.proj       = nn.ModuleList([nn.Conv2d(c, bifpn_ch, 1) for c in self.encoder.out_channels])
        self.bifpn      = BiFPN([bifpn_ch] * depth, num_layers=num_bifpn_layers)
        self.center     = ConvBlock(bifpn_ch, bifpn_ch)
        self.up_blocks  = nn.ModuleList([UpBlock(bifpn_ch, bifpn_ch, bifpn_ch) for _ in range(depth-1)])
        self.final_conv = nn.Conv2d(bifpn_ch, num_classes, 1)
    def forward(self, x):
        enc   = self.encoder(x)
        enc   = [p(f) for p, f in zip(self.proj, enc)]
        fused = self.bifpn(enc)
        x     = self.center(fused[-1])
        for i, up in enumerate(self.up_blocks):
            x = up(x, fused[-2-i])
        return self.final_conv(x)

# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_model(model, test_ds):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, lbl_t = test_ds[idx]
            img_t = img_t.unsqueeze(0).to(device)
            lbl_t = lbl_t.to(device)
            preds = torch.argmax(torch.softmax(model(img_t), 1), 1).squeeze(0)
            row = {'image_id': test_ds.image_paths[idx].stem}
            vessel_dice = 0.
            for c in range(NUM_CLASSES):
                p = (preds == c).float(); l = (lbl_t == c).float()
                tp = float((p * l).sum()); fp = float((p * (1-l)).sum()); fn = float(((1-p) * l).sum())
                dice = 2*tp / (2*tp + fp + fn + 1e-6)
                iou  = tp / (tp + fp + fn + 1e-6)
                row[f'dice_c{c}'] = dice; row[f'iou_c{c}'] = iou
                if c in VESSEL_CLASSES: vessel_dice += dice
            row['vessel_dice'] = vessel_dice / len(VESSEL_CLASSES)
            row['vessel_iou']  = (row['iou_c1'] + row['iou_c2']) / 2
            rows.append(row)
    return pd.DataFrame(rows)

CKPTS = {
    'bifpn_resunet':      (CKPT_DIR / 'best_model_bifpn_resunet_592.pth',      BiFPNResUNet),
    'bifpn_resunet_dual': (CKPT_DIR / 'best_model_bifpn_resunet_dual_592.pth', BiFPNResUNetDual),
}

if __name__ == '__main__':
    test_ds = TestDataset(TEST_ROOT)
    results = []

    for name, (ckpt_path, ModelCls) in CKPTS.items():
        print(f'\nLoading {name} ({ModelCls.__name__}) ...')
        m = ModelCls(in_channels=3, num_classes=3, base_ch=64, depth=5,
                     bifpn_ch=64, num_bifpn_layers=1).to(device)
        try:
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f'  Loaded OK. Params: {sum(p.numel() for p in m.parameters()):,}')
        except Exception as e:
            print(f'  LOAD FAILED: {e}')
            continue

        df = eval_model(m, test_ds)
        df['model'] = name
        results.append(df)

        vd = df['vessel_dice'].mean()
        vi = df['vessel_iou'].mean()
        ad = df['dice_c1'].mean()
        wd = df['dice_c2'].mean()
        print(f'  Vessel Dice={vd:.4f}  IoU={vi:.4f}  Artery={ad:.4f}  Vein={wd:.4f}')

    if results:
        out_df = pd.concat(results, ignore_index=True)
        out_path = BASE / 'paper_fixes' / 'data' / 'bifpn_resunet_scores.csv'
        out_df.to_csv(out_path, index=False)
        print(f'\nSaved to {out_path}')

        # Comparison with existing models
        existing = pd.read_csv(BASE / 'paper_fixes' / 'data' / 'model_summary.csv')
        print('\n--- Comparison with paper models ---')
        print(existing[['model','vessel_dice','artery_dice','vein_dice']].to_string(index=False, float_format='%.4f'))
        print('\n--- BiFPN-ResUNet variants ---')
        for df in results:
            m = df['model'].iloc[0]
            print(f"  {m}: Dice={df['vessel_dice'].mean():.4f}  "
                  f"Art={df['dice_c1'].mean():.4f}  Vein={df['dice_c2'].mean():.4f}")
