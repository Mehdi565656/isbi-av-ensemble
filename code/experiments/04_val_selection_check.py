"""
Task 4 addendum: Re-run ensemble member selection using the validation set.
Loads all 5 fixed-592 models, runs on val split, finds best 2/3 combos by val Dice,
then measures their test-set Dice to check whether val-selection = test-selection.
"""

import sys, random
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (StandardUNet, MorphoUNet, BiFPNUNet, MorphoBiFPNUNet,
                           NUM_CLASSES, VESSEL_CLASSES, NEW_SIZE, device,
                           dice_iou_per_class, rgb_to_class_mask)

BASE     = Path('/path/to/workspace')
VAL_ROOT = BASE / 'Fundus-AVSeg-prep-data' / 'val'
TEST_ROOT= BASE / 'Fundus-AVSeg-prep-data' / 'test'

CKPT = {
    'unet':           BASE / 'multi_step_back/unet/models/best_model_standard_unet.pth',
    'morpho_unet':    BASE / 'multi_step_back/morpho/models/best_model_morpho_unet_tophat_dist_592.pth',
    'bifpn_unet':     BASE / 'multi_step_back/BiFPN/models/best_model_bifpn_592.pth',
    'morphobifpn':    BASE / 'multi_step_back/morpho_BiFPN/models/best_model_morphobifpn_tophat_dist_592.pth',
    'bifpn_64to1024': BASE / 'multi_step_back/BiFPN/models/best_model_bifpn_64to1024_592.pth',
}

def load_model(name, ckpt_path):
    if name=='unet':        m=StandardUNet(3,NUM_CLASSES)
    elif name=='morpho_unet':   m=MorphoUNet(3,NUM_CLASSES)
    elif name=='bifpn_unet':    m=BiFPNUNet(3,NUM_CLASSES,base_ch=32,depth=4,num_bifpn_layers=1,bifpn_ch=64)
    elif name=='morphobifpn':   m=MorphoBiFPNUNet(3,NUM_CLASSES,base_ch=32,bifpn_ch=64,num_bifpn_layers=1)
    elif name=='bifpn_64to1024':m=BiFPNUNet(3,NUM_CLASSES,base_ch=64,depth=5,num_bifpn_layers=1,bifpn_ch=64)
    m.load_state_dict(torch.load(ckpt_path,map_location=device))
    return m.eval().to(device)

def get_softmax_maps(root):
    root = Path(root)
    img_paths = sorted((root/'images').glob('*.png'))
    lbl_paths = [root/'annotation'/f'{p.stem}.png' for p in img_paths]
    softmax = {name: [] for name in CKPT}
    labels  = []
    for img_path, lbl_path in zip(img_paths, lbl_paths):
        img = T.ToTensor()(Image.open(img_path).convert('RGB').resize((NEW_SIZE,NEW_SIZE),Image.BILINEAR)).unsqueeze(0).to(device)
        lbl = rgb_to_class_mask(np.array(Image.open(lbl_path).convert('RGB').resize((NEW_SIZE,NEW_SIZE),Image.NEAREST)))
        labels.append(lbl)
        for name, model in models.items():
            with torch.no_grad():
                probs = torch.softmax(model(img), 1)[0].cpu().numpy()
            softmax[name].append(probs)
    return softmax, labels

def ensemble_dice(softmax_maps, labels, members):
    total = 0.
    for i in range(len(labels)):
        avg = np.mean([softmax_maps[m][i] for m in members], axis=0)
        preds = torch.from_numpy(np.argmax(avg,0)).to(device)
        lbl_t = torch.from_numpy(labels[i]).to(device)
        d, _ = dice_iou_per_class(preds, lbl_t)
        total += 0.5*(d[1]+d[2])
    return total / len(labels)

print("Loading models...")
models = {name: load_model(name, ckpt) for name, ckpt in CKPT.items()}

print("Running on val set...")
val_softmax, val_labels = get_softmax_maps(VAL_ROOT)
print(f"Val set: {len(val_labels)} images")

print("Running on test set...")
test_softmax, test_labels = get_softmax_maps(TEST_ROOT)
print(f"Test set: {len(test_labels)} images")

# Individual model Dice on val
print("\n=== Individual val Dice ===")
for name in CKPT:
    d = ensemble_dice(val_softmax, val_labels, [name])
    print(f"  {name:25s}: {d:.4f}")

# All triple combinations by val Dice
print("\n=== Triple ensemble combos (val Dice) ===")
triples = list(combinations(list(CKPT.keys()), 3))
triple_results = []
for triple in triples:
    val_d  = ensemble_dice(val_softmax,  val_labels,  list(triple))
    test_d = ensemble_dice(test_softmax, test_labels, list(triple))
    triple_results.append((val_d, test_d, triple))
triple_results.sort(reverse=True)
for val_d, test_d, triple in triple_results:
    marker = " <-- ORIGINAL" if set(triple)=={'bifpn_64to1024','unet','morpho_unet'} else ""
    print(f"  val={val_d:.4f}  test={test_d:.4f}  {triple}{marker}")

# All pair combinations
print("\n=== Pair ensemble combos (val Dice) ===")
pairs = list(combinations(list(CKPT.keys()), 2))
pair_results = []
for pair in pairs:
    val_d  = ensemble_dice(val_softmax,  val_labels,  list(pair))
    test_d = ensemble_dice(test_softmax, test_labels, list(pair))
    pair_results.append((val_d, test_d, pair))
pair_results.sort(reverse=True)
for val_d, test_d, pair in pair_results:
    marker = " <-- ORIGINAL" if set(pair)=={'bifpn_64to1024','unet'} else ""
    print(f"  val={val_d:.4f}  test={test_d:.4f}  {pair}{marker}")
