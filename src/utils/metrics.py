"""
basic metrics: dice
"""
import numpy as np
import torch

def dice_metric(pred, target, thr=0.5, eps=1e-6):
    # pred: torch tensor or numpy
    if isinstance(pred, torch.Tensor):
        pred = (pred > thr).cpu().numpy().astype(np.uint8)
    else:
        pred = (pred > thr).astype(np.uint8)
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy().astype(np.uint8)
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return float(num/den) if den>0 else 0.0
