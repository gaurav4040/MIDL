"""
Transforms & patch sampler for training
"""
import numpy as np
import random

def random_flip(vol, mask=None):
    axes = [0,1,2]
    for a in axes:
        if random.random() < 0.5:
            vol = np.flip(vol, axis=a).copy()
            if mask is not None:
                mask = np.flip(mask, axis=a).copy()
    return (vol, mask) if mask is not None else vol

def random_rotate_90(vol, mask=None):
    k = random.randint(0,3)
    vol = np.rot90(vol, k, axes=(1,2)).copy()
    if mask is not None:
        mask = np.rot90(mask, k, axes=(1,2)).copy()
    return (vol, mask) if mask is not None else vol

def random_crop_patch(vol, mask, patch_size):
    z,y,x = vol.shape
    dz,dy,dx = patch_size
    if z <= dz or y <= dy or x <= dx:
        pz = max(0, dz - z); py = max(0, dy - y); px = max(0, dx - x)
        vol = np.pad(vol, ((0,pz),(0,py),(0,px)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((0,pz),(0,py),(0,px)), mode='constant', constant_values=0)
        z,y,x = vol.shape
    sz = random.randint(0, z-dz)
    sy = random.randint(0, y-dy)
    sx = random.randint(0, x-dx)
    return vol[sz:sz+dz, sy:sy+dy, sx:sx+dx], mask[sz:sz+dz, sy:sy+dy, sx:sx+dx]
