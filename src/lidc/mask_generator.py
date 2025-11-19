# src/lidc/mask_generator.py
"""
Rasterize contours (physical coords) into 3D mask arrays aligned with an SITK image.
"""
from typing import Dict, List
import numpy as np
import SimpleITK as sitk
from skimage.draw import polygon
from pathlib import Path

Contour = List[tuple]

def xml_to_mask(xml_path: str, reference_image: sitk.Image, merge_strategy: str = "consensus"):
    """
    Parse XML and rasterize contours into a 3D numpy mask aligned to reference_image.
    Args:
      xml_path: path to LIDC XML
      reference_image: sitk.Image (the same image used to map phys coords -> voxel indices)
      merge_strategy: 'union' | 'majority' | 'consensus'
    Returns:
      mask: numpy array (Z, Y, X) uint8
    """
    from .xml_parser import parse_lidc_xml
    per_reader_contours = parse_lidc_xml(xml_path)

    size = reference_image.GetSize()  # (x,y,z)
    nx, ny, nz = size
    # create per-reader masks
    reader_masks = []
    for rid, contours in per_reader_contours.items():
        rm = np.zeros((nz, ny, nx), dtype=np.uint8)
        for contour in contours:
            # map each physical point to voxel index
            vox = []
            for (x,y,z) in contour:
                # TransformPhysicalPointToIndex expects (x,y,z) in world coords
                try:
                    idx = reference_image.TransformPhysicalPointToIndex((x,y,z))
                except Exception:
                    # skip points that map outside
                    continue
                vox.append(idx)  # (i,j,k)
            if len(vox) < 3:
                continue
            vox = np.array(vox)
            # group by slice (k)
            ks = np.unique(vox[:,2])
            for k in ks:
                sel = vox[vox[:,2] == k]
                if sel.shape[0] < 3:
                    continue
                rr, cc = polygon(sel[:,1], sel[:,0], shape=(ny, nx))
                k_int = int(k)
                if 0 <= k_int < nz:
                    rm[k_int, rr, cc] = 1
        reader_masks.append(rm)

    if not reader_masks:
        return np.zeros((nz, ny, nx), dtype=np.uint8)

    stack = np.stack(reader_masks, axis=0)  # R x Z x Y x X

    if merge_strategy == "union":
        mask = np.any(stack, axis=0).astype(np.uint8)
    elif merge_strategy == "majority":
        mask = (stack.sum(axis=0) > (stack.shape[0] / 2)).astype(np.uint8)
    else:  # consensus (>=2)
        mask = (stack.sum(axis=0) >= 2).astype(np.uint8)

    return mask

def save_mask_from_array(mask_np, reference_image: sitk.Image, out_path: str):
    """
    Save mask numpy array (Z,Y,X) as NIfTI using reference image geometry.
    """
    out_path = Path(out_path)
    img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    img.SetSpacing(reference_image.GetSpacing())
    img.SetOrigin(reference_image.GetOrigin())
    img.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(img, str(out_path))
