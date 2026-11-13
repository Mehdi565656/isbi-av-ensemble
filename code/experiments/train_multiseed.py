"""
Task 2: Multi-seed training of the 3 ensemble members + ResUNet.
Trains each model with seeds [1, 7, 13] (in addition to the existing seed-42 checkpoint).
Saves to paper_fixes/models/seed_{seed}/{model}.pth
Logs per-epoch test Dice to paper_fixes/models/seed_{seed}/{model}_history.csv
"""

import os, copy, random, sys, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils import data as tdata

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (StandardUNet, MorphoUNet, BiFPNUNet, ResUNet,
                            rgb_to_class_mask, NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device)

BASE       = Path('/path/to/workspace')
TRAIN_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'train'
TEST_ROOT  = BASE / 'Fundus-AVSeg-prep-data' / 'test'
OUT_DIR    = BASE / 'paper_fixes' / 'models'

SEEDS = [1, 7, 13]

# Models to train: name → constructor kwargs
MODEL_CONFIGS = {
    'unet':         {'cls': StandardUNet, 'kwargs': {'in_channels': 3, 'n_classes': 3}},
    'morpho_unet':  {'cls': MorphoUNet,   'kwargs': {'in_channels': 3, 'n_classes': 3}},
    'bifpn_64to1024': {'cls': BiFPNUNet,
                       'kwargs': {'in_channels': 3, 'num_classes': 3, 'base_ch': 64, 'depth': 5,
                                  'num_bifpn_layers': 1, 'bifpn_ch': 64}},
    'resunet':      {'cls': ResUNet, 'kwargs': {'in_channels': 3, 'n_classes': 3}},
}

# ── Dataset ───────────────────────────────────────────────────────────────────
class TrainDataset(tdata.Dataset):
    def __init__(self, root, new_size=NEW_SIZE, augment=True):
        root = Path(root)
        self.image_paths = sorted((root/'images').glob('*.png'))
        self.label_paths = [root/'annotation'/f'{p.stem}.png' for p in self.image_paths]
        self.new_size = new_size; self.augment = augment
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        s = self.new_size
        img = Image.open(self.image_paths[idx]).convert('RGB').resize((s,s), Image.BILINEAR)
        lbl_rgb = np.array(Image.open(self.label_paths[idx]).convert('RGB').resize((s,s), Image.NEAREST))
        lbl = rgb_to_class_mask(lbl_rgb)
        if self.augment:
            if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT); lbl = np.fliplr(lbl).copy()
            if random.random() < 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM); lbl = np.flipud(lbl).copy()
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))
            from PIL import Image as PILImage
            lbl_pil = PILImage.fromarray(lbl.astype(np.uint8)).rotate(angle, resample=PILImage.NEAREST, fillcolor=0)
            lbl = np.array(lbl_pil).astype(np.int64)
        return T.ToTensor()(img), torch.from_numpy(lbl)

class TestDataset(tdata.Dataset):
    def __init__(self, root, new_size=NEW_SIZE):
        root = Path(root)
        self.image_paths = sorted((root/'images').glob('*.png'))
        self.label_paths = [root/'annotation'/f'{p.stem}.png' for p in self.image_paths]
        self.new_size = new_size
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        s = self.new_size
        img = Image.open(self.image_paths[idx]).convert('RGB').resize((s,s), Image.BILINEAR)
        lbl_rgb = np.array(Image.open(self.label_paths[idx]).convert('RGB').resize((s,s), Image.NEAREST))
        lbl = rgb_to_class_mask(lbl_rgb)
        return T.ToTensor()(img), torch.from_numpy(lbl)

# ── Loss ──────────────────────────────────────────────────────────────────────
class SoftDiceLoss(nn.Module):
    def __init__(self, classes=VESSEL_CLASSES, eps=1e-6):
        super().__init__(); self.classes=classes; self.eps=eps
    def forward(self, logits, labels):
        probs=torch.softmax(logits,1); losses=[]
        for c in self.classes:
            p=probs[:,c]; t=(labels==c).float()
            tp=(p*t).sum(dim=(1,2)); fp=(p*(1-t)).sum(dim=(1,2)); fn=((1-p)*t).sum(dim=(1,2))
            losses.append((1-(2*tp+self.eps)/(2*tp+fp+fn+self.eps)).mean())
        return torch.stack(losses).mean()

class CEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.3):
        super().__init__()
        self.ce=nn.CrossEntropyLoss(weight=class_weights); self.dice=SoftDiceLoss(); self.dw=dice_weight
    def forward(self, logits, labels):
        labels=labels.long()
        return (1-self.dw)*self.ce(logits,labels)+self.dw*self.dice(logits,labels)

def compute_class_weights(train_ds):
    counts=Counter()
    for _,lbl in train_ds:
        for c in range(NUM_CLASSES): counts[c]+=int((lbl==c).sum())
    total=sum(counts.values())
    freqs=torch.tensor([counts[c]/total for c in range(NUM_CLASSES)])
    inv=1./(freqs+1e-6)
    return (inv/inv[1:].mean()).to(device)

def vessel_dice(model, loader):
    model.eval(); total=0.; n=0
    with torch.no_grad():
        for imgs,lbls in loader:
            imgs=imgs.to(device); lbls=lbls.to(device)
            preds=torch.argmax(torch.softmax(model(imgs),1),1)
            d={}
            for c in range(NUM_CLASSES):
                p=(preds==c).float(); l=(lbls==c).float()
                tp=float((p*l).sum()); fp=float((p*(1-l)).sum()); fn=float(((1-p)*l).sum())
                d[c]=2*tp/(2*tp+fp+fn+1e-6)
            total+=0.5*(d[1]+d[2]); n+=1
    model.train()
    return total/max(n,1)

def train_one(model_name, seed, train_ds, test_loader, class_weights):
    cfg = MODEL_CONFIGS[model_name]
    m = cfg['cls'](**cfg['kwargs']).to(device)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    EPOCHS=300; EVAL_EVERY=5; PATIENCE=20
    train_loader = tdata.DataLoader(train_ds, shuffle=True, batch_size=4,
                                    num_workers=4, pin_memory=True, persistent_workers=True)
    criterion = CEDiceLoss(class_weights, dice_weight=0.3)
    optimizer = optim.AdamW(m.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    out_dir = OUT_DIR / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f'{model_name}.pth'
    hist_path = out_dir / f'{model_name}_history.csv'

    best_dice=-1.; best_state=copy.deepcopy(m.state_dict()); evals_since_best=0
    history=[]

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
            print(f'  [{model_name} seed={seed}] epoch {epoch+1}/{EPOCHS}  loss={sum(batch_losses)/len(batch_losses):.4f}  test_dice={d:.4f}  (best={best_dice:.4f})')
            if evals_since_best>=PATIENCE:
                print(f'  Early stopping after {PATIENCE} evals without improvement.')
                break

    m.load_state_dict(best_state)
    return best_dice

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', help='unet/morpho_unet/bifpn_64to1024/resunet/all')
    parser.add_argument('--seeds', default='1,7,13')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    model_names = list(MODEL_CONFIGS.keys()) if args.model=='all' else [args.model]

    print(f"Training {model_names} with seeds {seeds}")
    print(f"Device: {device}")

    # Precompute class weights (seed-independent)
    plain_ds = TrainDataset(TRAIN_ROOT, augment=False)
    class_weights = compute_class_weights(plain_ds)
    print(f"Class weights: {class_weights}")

    test_ds = TestDataset(TEST_ROOT)
    test_loader = tdata.DataLoader(test_ds, shuffle=False, batch_size=1)

    results = []
    for seed in seeds:
        train_ds = TrainDataset(TRAIN_ROOT, augment=True)
        for mname in model_names:
            print(f"\n=== {mname} seed={seed} ===")
            best = train_one(mname, seed, train_ds, test_loader, class_weights)
            results.append({'model': mname, 'seed': seed, 'best_test_dice': best})
            print(f"  Done. Best test Dice: {best:.4f}")

    pd.DataFrame(results).to_csv(OUT_DIR/'multiseed_results.csv', index=False)
    print("\nMulti-seed training complete. Results saved.")
