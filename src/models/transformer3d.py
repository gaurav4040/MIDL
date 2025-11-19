"""
TinyPatchTransformer: light 3D patch-based transformer bottleneck
"""
import torch
import torch.nn as nn

class TinyPatchTransformer(nn.Module):
    def __init__(self, in_ch, patch_size=(4,4,4), emb_dim=64, depth=2, nhead=4):
        super().__init__()
        pz,py,px = patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.unproj = nn.ConvTranspose3d(emb_dim, in_ch, kernel_size=patch_size, stride=patch_size)
        # optional positional encoding can be added later

    def forward(self, x):
        B,C,D,H,W = x.shape
        z = self.proj(x)  # B,emb, D',H',W'
        b,emb,dp,hp,wp = z.shape
        z_flat = z.view(b, emb, -1).permute(0,2,1)  # B, N, emb
        z_enc = self.encoder(z_flat)  # B, N, emb
        z_back = z_enc.permute(0,2,1).view(b, emb, dp, hp, wp)
        out = self.unproj(z_back)
        return out
