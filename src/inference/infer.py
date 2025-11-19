"""
Inference script:
- sliding-window overlapping inference
- MC-dropout iterations for uncertainty
- save probability map & binary mask
Usage:
 python -m src.inference.infer --vol data/volumes/case001.nii.gz --cfg configs/config.yaml
"""
import argparse
import yaml
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import label
from src.models.hybrid3d import Hybrid3DNet

def sliding_window_infer(model, volume, patch_size=(128,128,128), overlap=0.5, device='cuda'):
    dz,dy,dx = patch_size
    z,y,x = volume.shape
    stride = (max(1,int(dz*(1-overlap))), max(1,int(dy*(1-overlap))), max(1,int(dx*(1-overlap))))
    score_map = np.zeros((z,y,x), dtype=np.float32)
    count_map = np.zeros_like(score_map)
    z_starts = list(range(0, max(1, z-dz+1), stride[0]))
    y_starts = list(range(0, max(1, y-dy+1), stride[1]))
    x_starts = list(range(0, max(1, x-dx+1), stride[2]))
    # include last window to cover end
    if z - (z_starts[-1]) > dz: z_starts.append(z-dz)
    if y - (y_starts[-1]) > dy: y_starts.append(y-dy)
    if x - (x_starts[-1]) > dx: x_starts.append(x-dx)

    for sz in z_starts:
        for sy in y_starts:
            for sx in x_starts:
                patch = volume[sz:sz+dz, sy:sy+dy, sx:sx+dx]
                pshape = patch.shape
                if pshape != (dz,dy,dx):
                    p = np.zeros((dz,dy,dx), dtype=patch.dtype)
                    p[:pshape[0],:pshape[1],:pshape[2]] = patch
                    patch = p
                inp = torch.from_numpy(patch[None,None,...]).float().to(device)
                with torch.no_grad():
                    seg_logits, _, _ = model(inp)
                    seg_prob = torch.sigmoid(seg_logits).cpu().numpy()[0,0]
                # add to maps
                score_map[sz:sz+pshape[0], sy:sy+pshape[1], sx:sx+pshape[2]] += seg_prob[:pshape[0],:pshape[1],:pshape[2]]
                count_map[sz:sz+pshape[0], sy:sy+pshape[1], sx:sx+pshape[2]] += 1
    score_map = score_map / np.maximum(count_map, 1)
    return score_map

def mc_dropout_prediction(model, volume, iters=8, patch_size=(128,128,128), overlap=0.5, device='cuda'):
    model.train()  # enable dropout
    preds = []
    for i in range(iters):
        preds.append(sliding_window_infer(model, volume, patch_size=patch_size, overlap=overlap, device=device))
    mean = np.mean(preds, axis=0)
    var = np.var(preds, axis=0)
    return mean, var

def simple_postprocess(prob_map, thr=0.4, min_voxels=100):
    binary = (prob_map > thr).astype(np.uint8)
    labels, n = label(binary)
    final_mask = np.zeros_like(binary)
    for lab in range(1, n+1):
        comp = (labels==lab)
        if comp.sum() >= min_voxels:
            final_mask[comp] = 1
    return final_mask

def run_inference(vol_path, cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    vol = nib.load(vol_path).get_fdata().astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Hybrid3DNet(in_ch=1, base=cfg['model']['base_channels'], transformer_cfg=cfg['model'].get('transformer', {}), mc_dropout=True).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    mean, var = mc_dropout_prediction(model, vol, iters=cfg['inference']['mc_dropout_iters'], patch_size=tuple(cfg['training']['patch_size']), overlap=cfg['inference']['overlap'], device=device)
    mask = simple_postprocess(mean, thr=0.4, min_voxels=100)
    # save outputs
    nib.save(nib.Nifti1Image(mean.astype(np.float32), np.eye(4)), "pred_prob.nii.gz")
    nib.save(nib.Nifti1Image(var.astype(np.float32), np.eye(4)), "pred_var.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), "pred_mask.nii.gz")
    print("Saved: pred_prob.nii.gz, pred_var.nii.gz, pred_mask.nii.gz")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vol", required=True)
    p.add_argument("--cfg", default="configs/config.yaml")
    args = p.parse_args()
    run_inference(args.vol, args.cfg)
