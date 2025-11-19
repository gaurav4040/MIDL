"""
Evaluation helpers: dice, classification AUC (sketch)
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.metrics import dice_metric

def lesion_classification_auc(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_scores))

def compute_dice_per_case(pred_mask, gt_mask):
    return dice_metric(pred_mask, gt_mask)
