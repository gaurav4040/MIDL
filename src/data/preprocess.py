"""
preprocess.py
- read DICOM/.mhd/.nii
- resample to isotropic spacing
- clip HU
- save to NIfTI
Usage:
 python -m src.data.preprocess --src /path/to/input.mhd --out data/volumes/case001.nii.gz
"""
import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom

def read_image(path):
    itk = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(itk).astype(np.float32)  # z,y,x
    spacing = itk.GetSpacing()[::-1]  # to z,y,x
    return arr, spacing

def resample_volume(vol, spacing, new_spacing=(1.0,1.0,1.0), order=1):
    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(np.array(vol.shape) * resize_factor).astype(int)
    real_resize = new_shape / np.array(vol.shape)
    vol_resampled = zoom(vol, real_resize, order=order)
    return vol_resampled

def clip_normalize(vol, clip_min=-1000, clip_max=400):
    vol = np.clip(vol, clip_min, clip_max)
    vol = (vol - clip_min) / (clip_max - clip_min)
    return vol.astype(np.float32)

def save_nifti(arr, out_path, spacing=(1.0,1.0,1.0)):
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing(spacing[::-1])
    sitk.WriteImage(itk, str(out_path))

def preprocess_and_save(src, out, new_spacing=(1.0,1.0,1.0)):
    vol, spacing = read_image(src)
    vol = resample_volume(vol, spacing, new_spacing)
    vol = clip_normalize(vol)
    save_nifti(vol, out, spacing=new_spacing)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spacing", default="1.0,1.0,1.0")
    args = p.parse_args()
    spacing = tuple(map(float, args.spacing.split(",")))
    out = preprocess_and_save(args.src, args.out, spacing)
    print("Saved:", out)
