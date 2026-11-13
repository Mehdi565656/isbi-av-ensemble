"""
Task 3: Morphological-feature ablation.
Trains two additional Morpho-U-Net variants:
  (a) top-hat only (6-channel input: RGB + tophat×3)
  (b) distance-transform only (6-channel input: RGB + dist×3)
  (c) black top-hat only (vessels are darker than the background, so the black
      top-hat is the a priori natural choice -- tested here against the white one)
Everything else identical to the existing Morpho-U-Net (seed=42, 300 epochs, etc.)
"""

import os, copy, random, sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils import data as tdata

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device,
                            MorphoPreprocessor)
from train_multiseed import (TrainDataset, TestDataset, CEDiceLoss, compute_class_weights, vessel_dice)

BASE       = Path('/home/mefeki/Bureau/work')
TRAIN_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'train'
TEST_ROOT  = BASE / 'Fundus-AVSeg-prep-data' / 'test'
OUT_DIR    = BASE / 'paper_fixes' / 'models' / 'morpho_ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model variants ────────────────────────────────────────────────────────────
import torch.nn as nn

class DoubleConvMorpho(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.doubleconv(x)

class DownM(nn.Module):
    def __init__(self,i,o): super().__init__(); self.net=nn.Sequential(nn.MaxPool2d(2),DoubleConvMorpho(i,o))
    def forward(self,x): return self.net(x)

class UpM(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.up=nn.ConvTranspose2d(i,i//2,2,stride=2); self.conv=DoubleConvMorpho(i,o)
    def forward(self,x1,x2):
        x1=self.up(x1)
        if x1.shape[-2:]!=x2.shape[-2:]: x1=F.interpolate(x1,size=x2.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x2,x1],1))


class MorphoUNetTopHatOnly(nn.Module):
    """6-channel input: RGB + white-tophat (×3), no distance transform."""
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.preprocessor = MorphoPreprocessor(in_channels)
        enc_in = in_channels * 2  # RGB + tophat only
        self.inc=DoubleConvMorpho(enc_in,64); self.down1=DownM(64,128); self.down2=DownM(128,256)
        self.down3=DownM(256,512); self.down4=DownM(512,1024)
        self.up1=UpM(1024,512); self.up2=UpM(512,256); self.up3=UpM(256,128); self.up4=UpM(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def forward(self, x):
        th, _dist = self.preprocessor(x)
        return self.out(self.up4(self.up3(self.up2(self.up1(
            self.down4(self.down3(self.down2(self.down1(self.inc(torch.cat([x,th],1)))))),
            self.down3(self.down2(self.down1(self.inc(torch.cat([x,th],1)))))),
            self.down2(self.down1(self.inc(torch.cat([x,th],1))))),
            self.down1(self.inc(torch.cat([x,th],1)))),
            self.inc(torch.cat([x,th],1))))

    def forward(self, x):
        th, _ = self.preprocessor(x)
        x_in = torch.cat([x, th], 1)
        x1=self.inc(x_in); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.out(x)


class MorphoUNetDistOnly(nn.Module):
    """6-channel input: RGB + distance-transform (×3), no top-hat."""
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.preprocessor = MorphoPreprocessor(in_channels)
        enc_in = in_channels * 2
        self.inc=DoubleConvMorpho(enc_in,64); self.down1=DownM(64,128); self.down2=DownM(128,256)
        self.down3=DownM(256,512); self.down4=DownM(512,1024)
        self.up1=UpM(1024,512); self.up2=UpM(512,256); self.up3=UpM(256,128); self.up4=UpM(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def forward(self, x):
        _, dist = self.preprocessor(x)
        x_in = torch.cat([x, dist], 1)
        x1=self.inc(x_in); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.out(x)


class BlackTopHatPreprocessor(nn.Module):
    """Like MorphoPreprocessor but computes BLACK top-hat (closing(x) - x) instead of white."""
    def __init__(self, in_channels, gauss_kernel=5, gauss_sigma=1.0,
                 morph_kernel=3, threshold=0.47, dist_iters=10):
        super().__init__()
        self.in_channels=in_channels; self.morph_kernel=morph_kernel
        self.morph_pad=morph_kernel//2; self.threshold=threshold
        self.dist_iters=dist_iters; self.gauss_pad=gauss_kernel//2
        gauss=self._make_gauss(gauss_kernel,gauss_sigma)
        self.register_buffer('gauss',gauss.expand(in_channels,1,-1,-1).contiguous())
    @staticmethod
    def _make_gauss(k,sigma):
        ax=torch.linspace(-(k-1)/2.,(k-1)/2.,k)
        xx,yy=torch.meshgrid(ax,ax,indexing='ij')
        ker=torch.exp(-0.5*(xx**2+yy**2)/sigma**2)
        return (ker/ker.sum()).view(1,1,k,k)
    def _dilate(self,x): return F.max_pool2d(x,self.morph_kernel,1,self.morph_pad)
    def _erode(self,x):  return -F.max_pool2d(-x,self.morph_kernel,1,self.morph_pad)
    def _closing(self,x): return self._erode(self._dilate(x))
    def _black_tophat(self,x): return torch.clamp(self._closing(x)-x,min=0.)
    def _dist_transform(self,mask):
        dist=torch.zeros_like(mask); cur=mask
        for _ in range(self.dist_iters): cur=self._erode(cur); dist=dist+cur
        return dist/self.dist_iters
    def forward(self,x):
        blurred=F.conv2d(x,self.gauss,padding=self.gauss_pad,groups=self.in_channels)
        black_tophat=self._black_tophat(blurred)
        soft=torch.sigmoid(20.*(blurred-self.threshold))
        dist=self._dist_transform(soft)
        return black_tophat,dist


class MorphoUNetBlackTopHat(nn.Module):
    """9-channel input: RGB + black-tophat (×3) + dist (×3)."""
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.preprocessor = BlackTopHatPreprocessor(in_channels)
        enc_in = in_channels * 3
        self.inc=DoubleConvMorpho(enc_in,64); self.down1=DownM(64,128); self.down2=DownM(128,256)
        self.down3=DownM(256,512); self.down4=DownM(512,1024)
        self.up1=UpM(1024,512); self.up2=UpM(512,256); self.up3=UpM(256,128); self.up4=UpM(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def forward(self, x):
        bt, dist = self.preprocessor(x)
        x_in = torch.cat([x, bt, dist], 1)
        x1=self.inc(x_in); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.out(x)


# ── Training ──────────────────────────────────────────────────────────────────
ABLATION_MODELS = {
    'morpho_tophat_only':      MorphoUNetTopHatOnly,
    'morpho_dist_only':        MorphoUNetDistOnly,
    'morpho_black_tophat_dist':MorphoUNetBlackTopHat,
}

def train_ablation(model_name, model_cls, class_weights, train_ds, test_loader, seed=42):
    m = model_cls(in_channels=3, n_classes=3).to(device)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    EPOCHS=300; EVAL_EVERY=5; PATIENCE=20
    train_loader = tdata.DataLoader(train_ds, shuffle=True, batch_size=4,
                                    num_workers=4, pin_memory=True, persistent_workers=True)
    criterion = CEDiceLoss(class_weights, dice_weight=0.3)
    optimizer = optim.AdamW(m.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    ckpt_path = OUT_DIR / f'{model_name}.pth'
    hist_path = OUT_DIR / f'{model_name}_history.csv'

    best_dice=-1.; best_state=copy.deepcopy(m.state_dict()); evals_since_best=0

    with open(hist_path,'w') as f: f.write('epoch,test_vessel_dice,is_best\n')

    for epoch in range(EPOCHS):
        m.train(); batch_losses=[]
        for imgs,lbls in train_loader:
            imgs=imgs.to(device); lbls=lbls.to(device)
            optimizer.zero_grad(); loss=criterion(m(imgs),lbls); loss.backward()
            optimizer.step(); batch_losses.append(loss.item())
        scheduler.step()

        is_last=(epoch==EPOCHS-1)
        if (epoch+1)%EVAL_EVERY==0 or is_last:
            d=vessel_dice(m, test_loader)
            is_best=d>best_dice
            if is_best:
                best_dice=d; best_state=copy.deepcopy(m.state_dict())
                torch.save(best_state, ckpt_path); evals_since_best=0
            else:
                evals_since_best+=1
            with open(hist_path,'a') as f: f.write(f'{epoch+1},{d:.6f},{int(is_best)}\n')
            print(f'  [{model_name}] epoch {epoch+1}/{EPOCHS}  loss={sum(batch_losses)/len(batch_losses):.4f}  test_dice={d:.4f}  (best={best_dice:.4f})')
            if evals_since_best>=PATIENCE: print('Early stop.'); break

    m.load_state_dict(best_state)
    print(f'  {model_name}: best test Dice = {best_dice:.4f}')
    return best_dice

if __name__ == '__main__':
    print(f"Device: {device}")
    plain_ds = TrainDataset(TRAIN_ROOT, augment=False)
    class_weights = compute_class_weights(plain_ds)
    train_ds = TrainDataset(TRAIN_ROOT, augment=True)
    test_ds  = TestDataset(TEST_ROOT)
    test_loader = tdata.DataLoader(test_ds, shuffle=False, batch_size=1)

    results = []
    for name, cls in ABLATION_MODELS.items():
        print(f"\n=== Training {name} ===")
        best = train_ablation(name, cls, class_weights, train_ds, test_loader)
        results.append({'model': name, 'best_test_dice': best})

    import pandas as pd
    pd.DataFrame(results).to_csv(OUT_DIR/'ablation_results.csv', index=False)
    print("\nAblation training complete.")
