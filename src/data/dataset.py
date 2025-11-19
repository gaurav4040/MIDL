"""
NiftiPatchDataset
expects matching filenames in volumes_dir and masks_dir
returns: vol_patch (1,D,H,W), mask_patch (1,D,H,W), cls_label (0/1)
"""
import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .transforms import random_flip, random_rotate_90, random_crop_patch

class NiftiPatchDataset(Dataset):
    def __init__(self, vols_dir, masks_dir, patch_size=(128,128,128), augment=False):
        self.vols = sorted(glob(os.path.join(vols_dir, "*.nii*")))
        self.masks = sorted(glob(os.path.join(masks_dir, "*.nii*")))
        assert len(self.vols) == len(self.masks), "volumes and masks mismatch"
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.vols)

    def load_pair(self, idx):
        vpath = self.vols[idx]; mpath = self.masks[idx]
        vol = nib.load(vpath).get_fdata().astype(np.float32)
        mask = nib.load(mpath).get_fdata().astype(np.uint8)
        return vol, mask

    def __getitem__(self, idx):
        vol, mask = self.load_pair(idx)
        if self.augment:
            vol, mask = random_flip(vol, mask)
            vol, mask = random_rotate_90(vol, mask)
        vol_patch, mask_patch = random_crop_patch(vol, mask, self.patch_size)
        vol_patch = np.expand_dims(vol_patch, 0)  # 1,D,H,W
        mask_patch = np.expand_dims(mask_patch, 0)
        cls_label = 1 if mask_patch.sum() > 0 else 0
        return torch.from_numpy(vol_patch).float(), torch.from_numpy(mask_patch).float(), torch.tensor(cls_label).long()
