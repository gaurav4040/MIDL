"""
Hybrid3DNet (fixed heads)
- Encoder (3 levels)
- Bottleneck conv + TinyPatchTransformer
- Decoder
- Heads: segmentation, classification, regression
- Optional MC Dropout for uncertainty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer3d import TinyPatchTransformer

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base*2)
        self.enc3 = ConvBlock3D(base*2, base*4)

    def forward(self,x):
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        return s1, s2, s3

class Decoder3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock3D(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base*2, base)

    def forward(self, x, s1, s2):
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        return x

class Hybrid3DNet(nn.Module):
    def __init__(self, in_ch=1, base=16, transformer_cfg=None, mc_dropout=False):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base)
        self.bottleneck_conv = ConvBlock3D(base*4, base*4)
        tcfg = transformer_cfg or {}
        self.transformer = TinyPatchTransformer(in_ch=base*4,
                                                patch_size=tuple(tcfg.get("patch_size",(4,4,4))),
                                                emb_dim=tcfg.get("emb_dim", base*4),
                                                depth=tcfg.get("depth",2),
                                                nhead=tcfg.get("nhead",4))
        self.decoder = Decoder3D(base)
        self.seg_head = nn.Conv3d(base, 1, 1)

        # Heads that operate on pooled features (B, C)
        self.cls_head = nn.Linear(base, 2)    # classification: outputs logits for 2 classes
        self.reg_head = nn.Linear(base, 1)    # regression: outputs a single value

        self.mc_dropout = mc_dropout
        if mc_dropout:
            self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        s1,s2,s3 = self.encoder(x)
        x = self.bottleneck_conv(s3)
        x = self.transformer(x)
        if self.mc_dropout:
            x = self.dropout(x)

        x = self.decoder(x, s1, s2)
        seg = self.seg_head(x)

        # Global pooling to get (B, C) feature vector — robust to spatial sizes
        if x.dim() == 5:
            # mean across D,H,W -> shape (B, C)
            pooled = x.mean(dim=[2,3,4])
        else:
            pooled = x.view(x.size(0), -1)

        cls = self.cls_head(pooled)
        reg = self.reg_head(pooled)

        return seg, cls, reg
