# src/lidc/resampler.py
"""
Resample sitk.Image objects to target spacing. Two helpers: one for images, one for masks.
"""
import SimpleITK as sitk
from typing import Tuple

def resample_image(img: sitk.Image, new_spacing: Tuple[float,float,float] = (1.0,1.0,1.0), interpolator=sitk.sitkLinear):
    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampled = resampler.Execute(img)
    return resampled

def resample_mask(mask_img: sitk.Image, new_spacing: Tuple[float,float,float] = (1.0,1.0,1.0)):
    return resample_image(mask_img, new_spacing=new_spacing, interpolator=sitk.sitkNearestNeighbor)
