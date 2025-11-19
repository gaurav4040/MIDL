# src/lidc/normalizer.py
"""
HU clipping and normalization utilities.
"""
import numpy as np
import SimpleITK as sitk

def clip_and_normalize(sitk_img: sitk.Image, clip_min: int = -1000, clip_max: int = 400):
    """
    Clip HU values and normalize to [0,1]. Returns a new sitk.Image.
    """
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # Z,Y,X
    arr = np.clip(arr, clip_min, clip_max)
    arr = (arr - clip_min) / float(clip_max - clip_min)
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    # copy geo info from input
    out.SetSpacing(sitk_img.GetSpacing())
    out.SetOrigin(sitk_img.GetOrigin())
    out.SetDirection(sitk_img.GetDirection())
    return out
