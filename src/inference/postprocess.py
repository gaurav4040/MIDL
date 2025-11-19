"""
Postprocessing utilities: radiomics-like filters and shape filters
"""
import numpy as np
from skimage.measure import regionprops, label

def compute_shape_features(binary_mask):
    labels = label(binary_mask)
    props = regionprops(labels)
    features = []
    for p in props:
        features.append({
            'area': p.area,
            'eccentricity': p.eccentricity,
            'extent': p.extent,
            'solidity': p.solidity
        })
    return features

def filter_regions_by_shape(binary_mask, min_area=50, max_ecc=0.98):
    labels = label(binary_mask)
    out = np.zeros_like(binary_mask)
    for idx,prop in enumerate(regionprops(labels), start=1):
        if prop.area >= min_area and prop.eccentricity <= max_ecc:
            out[labels==idx] = 1
    return out
