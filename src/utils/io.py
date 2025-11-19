"""
simple IO helpers
"""
import nibabel as nib
import numpy as np

def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    return arr

def save_nifti(arr, out):
    nii = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    nib.save(nii, out)
