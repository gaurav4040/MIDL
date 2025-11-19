"""
Test Evaluation Script
Usage:
    python -m src.train.test_eval --cfg configs/config.yaml
"""

import argparse
import yaml
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import NiftiPatchDataset
from src.models.hybrid3d import Hybrid3DNet
from src.utils.metrics import dice_metric
from sklearn.metrics import confusion_matrix


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_split_indices(n, seed=42):
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)

    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)

    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]
    test_idx  = idx[val_end:]

    return train_idx, val_idx, test_idx


def compute_metrics(pred, target):
    """Compute classical segmentation metrics."""
    pred_bin = (pred > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)

    # Dice
    dice = dice_metric(pred_bin, target)

    # IoU
    inter = (pred_bin & target).sum()
    union = (pred_bin | target).sum()
    iou = inter / (union + 1e-6)

    # sensitivity, specificity, precision, F1
    tn, fp, fn, tp = confusion_matrix(
        target.flatten(),
        pred_bin.flatten(),
        labels=[0,1]
    ).ravel()

    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    f1   = 2 * (prec * sens) / (prec + sens + 1e-6)

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1
    }


def test_eval(cfg_path):

    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load full dataset
    ds = NiftiPatchDataset(
        cfg['data']['volumes_dir'],
        cfg['data']['masks_dir'],
        patch_size=tuple(cfg['training']['patch_size']),
        augment=False   # no augmentation on test!
    )

    n = len(ds)
    train_idx, val_idx, test_idx = get_split_indices(n, seed=cfg["training"]["seed"])

    print(f"Total samples: {n}")
    print(f"Test samples: {len(test_idx)}")

    test_loader = DataLoader(
        torch.utils.data.Subset(ds, test_idx),
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Load model
    model = Hybrid3DNet(
        in_ch=1,
        base=cfg["model"]["base_channels"],
        transformer_cfg=cfg["model"]["transformer"],
        mc_dropout=False
    ).to(device)

    if not os.path.exists("best_model.pth"):
        print("❌ ERROR: best_model.pth not found. Train first!")
        return

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    print("\n🔥 Running Test Evaluation...\n")

    metrics_list = []

    with torch.no_grad():
        for i, (vol, mask, cls) in enumerate(test_loader):
            vol = vol.to(device)
            mask_np = mask.numpy()[0,0]  # GT mask to numpy

            seg_logits, _, _ = model(vol)
            pred = torch.sigmoid(seg_logits).cpu().numpy()[0,0]  # predicted mask

            # compute metrics
            m = compute_metrics(pred, mask_np)
            metrics_list.append(m)

            print(f"[{i+1}/{len(test_loader)}] Dice={m['dice']:.4f} | IoU={m['iou']:.4f}")

    # Aggregate metrics
    final = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

    print("\n==============================")
    print("✨ FINAL TEST METRICS ✨")
    print("==============================")
    print(f"Dice:        {final['dice']:.4f}")
    print(f"IoU:         {final['iou']:.4f}")
    print(f"Sensitivity: {final['sensitivity']:.4f}")
    print(f"Specificity: {final['specificity']:.4f}")
    print(f"Precision:   {final['precision']:.4f}")
    print(f"F1 Score:    {final['f1']:.4f}")
    print("==============================\n")

    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/config.yaml")
    args = p.parse_args()
    test_eval(args.cfg)
