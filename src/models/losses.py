"""
Losses: dice + BCE + classification + regression
"""
import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2*(pred*target).sum(dim=[1,2,3,4])
    den = pred.sum(dim=[1,2,3,4]) + target.sum(dim=[1,2,3,4]) + eps
    loss = 1 - (num/den)
    return loss.mean()

def bce_dice_loss(seg_logits, seg_target):
    bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
    dloss = dice_loss(seg_logits, seg_target)
    return bce + dloss

def multi_task_loss(seg_logits, seg_target, cls_logits, cls_target, reg_pred, reg_target, cfg):
    seg_l = bce_dice_loss(seg_logits, seg_target) * cfg['loss']['seg_weight']
    cls_l = F.cross_entropy(cls_logits, cls_target) * cfg['loss']['cls_weight']
    # reg_target may be zeros if unavailable
    reg_l = F.mse_loss(reg_pred.squeeze(-1), reg_target.squeeze(-1)) * cfg['loss']['reg_weight']
    return seg_l + cls_l + reg_l
