"""
Tasks 0.4 + 1 + 5 + 7: Run inference on all fixed-592 models over the 20 test images.
Saves softmax maps, per-image metrics CSV, timing/params/diversity tables.
"""

import os, time, random, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils import data as tdata

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path('/path/to/workspace')
TEST_ROOT   = BASE / 'Fundus-AVSeg-prep-data' / 'test'
OUT_DATA    = BASE / 'paper_fixes' / 'data'
OUT_SOFTMAX = OUT_DATA / 'softmax_maps'
CKPT = {
    'unet':           BASE / 'multi_step_back/unet/models/best_model_standard_unet.pth',
    'morpho_unet':    BASE / 'multi_step_back/morpho/models/best_model_morpho_unet_tophat_dist_592.pth',
    'bifpn_unet':     BASE / 'multi_step_back/BiFPN/models/best_model_bifpn_592.pth',
    'morphobifpn':    BASE / 'multi_step_back/morpho_BiFPN/models/best_model_morphobifpn_tophat_dist_592.pth',
    'bifpn_64to1024': BASE / 'multi_step_back/BiFPN/models/best_model_bifpn_64to1024_592.pth',
    'resunet':        BASE / 'multi_step_back/unet/models/best_model_resunet_592.pth',
}
OUT_DATA.mkdir(parents=True, exist_ok=True)

NUM_CLASSES    = 3
VESSEL_CLASSES = [1, 2]
NEW_SIZE       = 592
SEED           = 42
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── EXACT model definitions matching checkpoint keys ─────────────────────────

# ── Standard U-Net ────────────────────────────────────────────────────────────
class DoubleConvStd(nn.Module):
    """Exact DoubleConv from unet/Unet_images_only.ipynb: attribute 'double_conv', bias=False."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class StandardUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.inc   = DoubleConvStd(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConvStd(64,  128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConvStd(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConvStd(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConvStd(512,1024))
        self.up1   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConvStd(1024, 512)
        self.up2   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConvStd(512, 256)
        self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConvStd(256, 128)
        self.up4   = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.conv4 = DoubleConvStd(128,  64)
        self.outc  = nn.Conv2d(64, n_classes, 1)

    @staticmethod
    def _pad(x, ref):
        dY = ref.size(2)-x.size(2); dX = ref.size(3)-x.size(3)
        return F.pad(x, [dX//2, dX-dX//2, dY//2, dY-dY//2])

    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.conv1(torch.cat([x4, self._pad(self.up1(x5), x4)], 1))
        x=self.conv2(torch.cat([x3, self._pad(self.up2(x),  x3)], 1))
        x=self.conv3(torch.cat([x2, self._pad(self.up3(x),  x2)], 1))
        x=self.conv4(torch.cat([x1, self._pad(self.up4(x),  x1)], 1))
        return self.outc(x)

# ── Morpho-U-Net ──────────────────────────────────────────────────────────────
class DoubleConvMorpho(nn.Module):
    """Exact DoubleConv from morpho notebook: attribute 'doubleconv', bias=True."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.doubleconv(x)

class DownMorpho(nn.Module):
    def __init__(self, i, o): super().__init__(); self.downconv=nn.Sequential(nn.MaxPool2d(2),DoubleConvMorpho(i,o))
    def forward(self, x): return self.downconv(x)

class UpMorpho(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.up=nn.ConvTranspose2d(i,i//2,2,stride=2); self.conv=DoubleConvMorpho(i,o)
    def forward(self, x1, x2):
        x1=self.up(x1)
        if x1.shape[-2:]!=x2.shape[-2:]: x1=F.interpolate(x1,size=x2.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x2,x1],1))

class MorphoPreprocessor(nn.Module):
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
    def _opening(self,x): return self._dilate(self._erode(x))
    def _white_tophat(self,x): return torch.clamp(x-self._opening(x),min=0.)
    def _dist_transform(self,mask):
        dist=torch.zeros_like(mask); cur=mask
        for _ in range(self.dist_iters): cur=self._erode(cur); dist=dist+cur
        return dist/self.dist_iters
    def forward(self,x):
        blurred=F.conv2d(x,self.gauss,padding=self.gauss_pad,groups=self.in_channels)
        tophat=self._white_tophat(blurred)
        soft=torch.sigmoid(20.*(blurred-self.threshold))
        dist=self._dist_transform(soft)
        return tophat,dist

class MorphoUNet(nn.Module):
    """Uses self.preprocessor (matches checkpoint key 'preprocessor.gauss')."""
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.preprocessor=MorphoPreprocessor(in_channels)
        enc_in=in_channels*3
        self.inc=DoubleConvMorpho(enc_in,64); self.down1=DownMorpho(64,128); self.down2=DownMorpho(128,256)
        self.down3=DownMorpho(256,512); self.down4=DownMorpho(512,1024)
        self.up1=UpMorpho(1024,512); self.up2=UpMorpho(512,256); self.up3=UpMorpho(256,128); self.up4=UpMorpho(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def forward(self,x):
        th,dist=self.preprocessor(x); x_in=torch.cat([x,th,dist],1)
        x1=self.inc(x_in); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.out(x)

# ── BiFPN-UNet (shared encoder/BiFPN code) ────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, depth=4):
        super().__init__()
        chs=[base_ch*(2**i) for i in range(depth)]
        self.blocks=nn.ModuleList(); self.pools=nn.ModuleList()
        in_c=in_ch
        for c in chs: self.blocks.append(ConvBlock(in_c,c)); self.pools.append(nn.MaxPool2d(2)); in_c=c
        self.out_channels=chs
    def forward(self,x):
        feats=[]
        for block,pool in zip(self.blocks,self.pools): x=block(x); feats.append(x); x=pool(x)
        return feats

class UpBlock(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__(); self.up=nn.ConvTranspose2d(in_ch,out_ch,2,stride=2); self.conv=ConvBlock(out_ch+skip_ch,out_ch)
    def forward(self,x,skip):
        x=self.up(x)
        if x.shape[-2:]!=skip.shape[-2:]: x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([skip,x],1))

class BiFPN(nn.Module):
    def __init__(self,channels,num_layers=2,eps=1e-4):
        super().__init__()
        self.eps=eps; self.num_levels=len(channels); self.num_layers=num_layers
        self.convs_td=nn.ModuleList([ConvBlock(c,c) for c in channels])
        self.convs_bu=nn.ModuleList([ConvBlock(c,c) for c in channels])
        self.w_td=nn.Parameter(torch.ones(num_layers,self.num_levels,2))
        self.w_bu=nn.Parameter(torch.ones(num_layers,self.num_levels,3))
    def forward(self,feats):
        P=feats
        for l in range(self.num_layers):
            td=[None]*self.num_levels; td[-1]=P[-1]
            for i in reversed(range(self.num_levels-1)):
                w=F.relu(self.w_td[l,i]); w=w/(w.sum()+self.eps)
                up=F.interpolate(td[i+1],size=P[i].shape[-2:],mode='bilinear',align_corners=False)
                td[i]=self.convs_td[i](w[0]*P[i]+w[1]*up)
            bu=[None]*self.num_levels; bu[0]=td[0]
            for i in range(1,self.num_levels):
                w=F.relu(self.w_bu[l,i]); w=w/(w.sum()+self.eps)
                down=F.max_pool2d(bu[i-1],2)
                if down.shape[-2:]!=td[i].shape[-2:]: down=F.interpolate(down,size=td[i].shape[-2:],mode='bilinear',align_corners=False)
                bu[i]=self.convs_bu[i](w[0]*P[i]+w[1]*td[i]+w[2]*down)
            P=bu
        return P

class BiFPNUNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=3,base_ch=32,depth=4,num_bifpn_layers=1,bifpn_ch=64):
        super().__init__()
        self.encoder=Encoder(in_ch=in_channels,base_ch=base_ch,depth=depth)
        self.proj=nn.ModuleList([nn.Conv2d(c,bifpn_ch,1) for c in self.encoder.out_channels])
        self.bifpn=BiFPN([bifpn_ch]*depth,num_layers=num_bifpn_layers)
        self.center=ConvBlock(bifpn_ch,bifpn_ch)
        self.up_blocks=nn.ModuleList([UpBlock(bifpn_ch,bifpn_ch,bifpn_ch) for _ in range(depth-1)])
        self.final_conv=nn.Conv2d(bifpn_ch,num_classes,1)
    def forward(self,x):
        enc=self.encoder(x); enc=[p(f) for p,f in zip(self.proj,enc)]
        fused=self.bifpn(enc); x=self.center(fused[-1])
        for i,up in enumerate(self.up_blocks): x=up(x,fused[-2-i])
        return self.final_conv(x)

class MorphoBiFPNUNet(nn.Module):
    """Uses self.morpho (matches checkpoint key 'morpho.gauss')."""
    def __init__(self,in_channels=3,num_classes=3,base_ch=32,bifpn_ch=64,num_bifpn_layers=1):
        super().__init__()
        self.morpho=MorphoPreprocessor(in_channels)
        aug_in=in_channels*3; depth=4
        self.encoder=Encoder(in_ch=aug_in,base_ch=base_ch,depth=depth)
        self.proj=nn.ModuleList([nn.Conv2d(c,bifpn_ch,1) for c in self.encoder.out_channels])
        self.bifpn=BiFPN([bifpn_ch]*depth,num_layers=num_bifpn_layers)
        self.center=ConvBlock(bifpn_ch,bifpn_ch)
        self.up_blocks=nn.ModuleList([UpBlock(bifpn_ch,bifpn_ch,bifpn_ch) for _ in range(depth-1)])
        self.final_conv=nn.Conv2d(bifpn_ch,num_classes,1)
    def forward(self,x):
        th,dist=self.morpho(x); x_in=torch.cat([x,th,dist],1)
        enc=self.encoder(x_in); enc=[p(f) for p,f in zip(self.proj,enc)]
        fused=self.bifpn(enc); x=self.center(fused[-1])
        for i,up in enumerate(self.up_blocks): x=up(x,fused[-2-i])
        return self.final_conv(x)

# ── ResUNet ───────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch),
        )
        self.shortcut=(nn.Sequential(nn.Conv2d(in_ch,out_ch,1,bias=False),nn.BatchNorm2d(out_ch))
                       if in_ch!=out_ch else nn.Identity())
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x): return self.relu(self.block(x)+self.shortcut(x))

class ResDown(nn.Module):
    def __init__(self,i,o): super().__init__(); self.pool=nn.MaxPool2d(2); self.block=ResidualBlock(i,o)
    def forward(self,x): return self.block(self.pool(x))

class ResUp(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.up=nn.ConvTranspose2d(i,i//2,2,stride=2); self.block=ResidualBlock(i,o)
    def forward(self,x1,x2):
        x1=self.up(x1)
        if x1.shape[-2:]!=x2.shape[-2:]: x1=F.interpolate(x1,size=x2.shape[-2:],mode='bilinear',align_corners=False)
        return self.block(torch.cat([x2,x1],1))

class ResUNet(nn.Module):
    def __init__(self,in_channels=3,n_classes=3):
        super().__init__()
        self.inc=ResidualBlock(in_channels,64); self.down1=ResDown(64,128); self.down2=ResDown(128,256)
        self.down3=ResDown(256,512); self.down4=ResDown(512,1024)
        self.up1=ResUp(1024,512); self.up2=ResUp(512,256); self.up3=ResUp(256,128); self.up4=ResUp(128,64)
        self.outc=nn.Conv2d(64,n_classes,1)
    def forward(self,x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return self.outc(x)

# ─── Dataset ──────────────────────────────────────────────────────────────────
def rgb_to_class_mask(rgb):
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    mask=np.zeros(rgb.shape[:2],dtype=np.int64)
    mask[(r>180)&(g<80)&(b<80)]=1; mask[(r<80)&(g<80)&(b>180)]=2
    return mask

class TestDataset(tdata.Dataset):
    def __init__(self,root,new_size=592):
        root=Path(root)
        self.image_paths=sorted((root/'images').glob('*.png'))
        self.label_paths=[root/'annotation'/f'{p.stem}.png' for p in self.image_paths]
        self.new_size=new_size
    def __len__(self): return len(self.image_paths)
    def __getitem__(self,idx):
        s=self.new_size
        img=Image.open(self.image_paths[idx]).convert('RGB').resize((s,s),Image.BILINEAR)
        lbl_rgb=np.array(Image.open(self.label_paths[idx]).convert('RGB').resize((s,s),Image.NEAREST))
        lbl=rgb_to_class_mask(lbl_rgb)
        return T.ToTensor()(img), torch.from_numpy(lbl), self.image_paths[idx].stem

# ─── Metrics ──────────────────────────────────────────────────────────────────
def dice_iou_per_class(preds,labels,n=NUM_CLASSES,eps=1e-6):
    dice={}; iou={}
    for c in range(n):
        p=(preds==c).float(); l=(labels==c).float()
        tp=float((p*l).sum()); fp=float((p*(1-l)).sum()); fn=float(((1-p)*l).sum())
        dice[c]=2*tp/(2*tp+fp+fn+eps); iou[c]=tp/(tp+fp+fn+eps)
    return dice,iou

# ─── Load models ──────────────────────────────────────────────────────────────
def load_model(name,ckpt_path):
    if name=='unet':        m=StandardUNet(3,NUM_CLASSES)
    elif name=='morpho_unet':   m=MorphoUNet(3,NUM_CLASSES)
    elif name=='bifpn_unet':    m=BiFPNUNet(3,NUM_CLASSES,base_ch=32,depth=4,num_bifpn_layers=1,bifpn_ch=64)
    elif name=='morphobifpn':   m=MorphoBiFPNUNet(3,NUM_CLASSES,base_ch=32,bifpn_ch=64,num_bifpn_layers=1)
    elif name=='bifpn_64to1024':m=BiFPNUNet(3,NUM_CLASSES,base_ch=64,depth=5,num_bifpn_layers=1,bifpn_ch=64)
    elif name=='resunet':       m=ResUNet(3,NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path,map_location=device))
    return m.eval().to(device)

if __name__ == '__main__':
    ENSEMBLE_MEMBERS=['bifpn_64to1024','unet','morpho_unet']  # top-3 per paper

    print("Loading models...")
    models={}
    for name,ckpt in CKPT.items():
        print(f"  {name}")
        models[name]=load_model(name,ckpt)
    print("All models loaded.")

    dataset=TestDataset(TEST_ROOT)
    loader=tdata.DataLoader(dataset,shuffle=False,batch_size=1)
    n_test=len(dataset)
    print(f"\nRunning inference on {n_test} test images...")

    softmax_store={name:[] for name in models}
    label_store=[]; stem_store=[]; per_image_rows=[]

    dummy=torch.randn(1,3,592,592,device=device)
    for m in models.values():
        with torch.no_grad(): m(dummy)
    torch.cuda.synchronize()

    timing={name:[] for name in models}

    for inputs,labels,stem in loader:
        inputs=inputs.to(device); stem=stem[0]
        label_np=labels[0].numpy()
        stem_store.append(stem); label_store.append(label_np)
        for name,model in models.items():
            torch.cuda.synchronize()
            t0=time.perf_counter()
            with torch.no_grad():
                logits=model(inputs); probs=torch.softmax(logits,1)
            torch.cuda.synchronize()
            t1=time.perf_counter()
            timing[name].append((t1-t0)*1000)
            probs_np=probs[0].cpu().numpy()
            softmax_store[name].append(probs_np)
            preds_t=torch.argmax(probs,1)[0]; labels_t=labels[0].to(device)
            dice,iou=dice_iou_per_class(preds_t,labels_t)
            for c in range(NUM_CLASSES):
                per_image_rows.append({'image_id':stem,'model':name,'class':c,'dice':dice[c],'iou':iou[c]})

    print("\nComputing ensemble predictions...")
    for ens_name, members in [('ensemble_top3', ENSEMBLE_MEMBERS),
                               ('ensemble_top2', ['bifpn_64to1024','unet'])]:
        for i in range(n_test):
            avg=np.mean([softmax_store[m][i] for m in members],axis=0)
            preds_t=torch.from_numpy(np.argmax(avg,axis=0)).to(device)
            labels_t=torch.from_numpy(label_store[i]).to(device)
            dice,iou=dice_iou_per_class(preds_t,labels_t)
            for c in range(NUM_CLASSES):
                per_image_rows.append({'image_id':stem_store[i],'model':ens_name,'class':c,'dice':dice[c],'iou':iou[c]})

    print("\nSaving softmax maps...")
    for name in models:
        model_dir=OUT_SOFTMAX/name; model_dir.mkdir(parents=True,exist_ok=True)
        for i,stem in enumerate(stem_store):
            np.save(model_dir/f'{stem}.npy', softmax_store[name][i].astype(np.float16))

    df=pd.DataFrame(per_image_rows)
    df.to_csv(OUT_DATA/'per_image_scores.csv',index=False)
    print(f"Saved per_image_scores.csv ({len(df)} rows)")

    print("\n=== AGGREGATE METRICS ===")
    model_names_all=list(models.keys())+['ensemble_top3','ensemble_top2']
    agg_results=[]
    for mname in model_names_all:
        sub=df[df['model']==mname]
        d_a=sub[sub['class']==1]['dice'].mean(); d_v=sub[sub['class']==2]['dice'].mean()
        i_a=sub[sub['class']==1]['iou'].mean();  i_v=sub[sub['class']==2]['iou'].mean()
        d_bg=sub[sub['class']==0]['dice'].mean(); i_bg=sub[sub['class']==0]['iou'].mean()
        vd=0.5*(d_a+d_v); vi=0.5*(i_a+i_v)
        agg_results.append({'model':mname,'dice_bg':d_bg,'dice_artery':d_a,'dice_vein':d_v,
                             'vessel_dice':vd,'iou_bg':i_bg,'iou_artery':i_a,'iou_vein':i_v,'vessel_iou':vi})
        print(f"{mname:30s}  vessel_dice={vd:.4f}  vessel_iou={vi:.4f}  (a={d_a:.4f}, v={d_v:.4f})")
    pd.DataFrame(agg_results).to_csv(OUT_DATA/'aggregate_metrics.csv',index=False)

    sub=df[df['model']=='ensemble_top3']
    chk_d=0.5*(sub[sub['class']==1]['dice'].mean()+sub[sub['class']==2]['dice'].mean())
    chk_i=0.5*(sub[sub['class']==1]['iou'].mean()+sub[sub['class']==2]['iou'].mean())
    print(f"\nSanity check top-3 ensemble: Dice={chk_d:.4f} (expect ~0.8320)  IoU={chk_i:.4f} (expect ~0.7139)")

    print("\n=== PARAMETER COUNTS ===")
    param_counts={}
    for name,model in models.items():
        n=sum(p.numel() for p in model.parameters()); param_counts[name]=n
        print(f"  {name:25s}: {n:>12,}")

    print("\n=== INFERENCE TIMING (ms/image, GPU batch=1) ===")
    timing_stats={}
    for name,times in timing.items():
        mean_ms=np.mean(times); std_ms=np.std(times)
        timing_stats[name]=(mean_ms,std_ms)
        print(f"  {name:25s}: {mean_ms:.1f} ± {std_ms:.1f} ms")
    ens_mean=sum(timing_stats[m][0] for m in ENSEMBLE_MEMBERS)
    print(f"  {'ensemble_top3 (sum)':25s}: {ens_mean:.1f} ms")
    timing_df=pd.DataFrame([{'model':n,'params':param_counts[n],'mean_ms':timing_stats[n][0],'std_ms':timing_stats[n][1]} for n in models])
    timing_df.to_csv(OUT_DATA/'inference_cost.csv',index=False)

    print("\n=== DIVERSITY ANALYSIS ===")
    diversity_rows=[]
    main_models=['unet','morpho_unet','bifpn_unet','morphobifpn','bifpn_64to1024']
    for m1,m2 in combinations(main_models,2):
        disagrees=[]; q_art=[]; q_vein=[]
        for i in range(n_test):
            p1=np.argmax(softmax_store[m1][i],axis=0)
            p2=np.argmax(softmax_store[m2][i],axis=0)
            gt=label_store[i]
            disagrees.append(float(np.mean(p1!=p2)))
            for c,lst in [(1,q_art),(2,q_vein)]:
                e1=((p1!=gt)&(gt==c)).astype(float); e2=((p2!=gt)&(gt==c)).astype(float)
                n11=float(np.sum(e1*e2)); n10=float(np.sum(e1*(1-e2)))
                n01=float(np.sum((1-e1)*e2)); n00=float(np.sum((1-e1)*(1-e2)))
                denom=(n11*n00-n10*n01); div=(n11*n00+n10*n01)
                lst.append(denom/div if div>0 else 0.)
        row={'model_1':m1,'model_2':m2,'mean_disagree':np.mean(disagrees),
             'q_stat_artery':np.mean(q_art),'q_stat_vein':np.mean(q_vein)}
        diversity_rows.append(row)
        print(f"  {m1:20s} vs {m2:20s}: disagree={row['mean_disagree']:.4f}  Q_art={row['q_stat_artery']:.4f}  Q_vein={row['q_stat_vein']:.4f}")
    pd.DataFrame(diversity_rows).to_csv(OUT_DATA/'diversity_analysis.csv',index=False)
    print("\n✓ All outputs written to paper_fixes/data/")
