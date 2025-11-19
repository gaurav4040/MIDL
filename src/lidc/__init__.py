# src/lidc/__init__.py
"""
LIDC raw -> preprocessed pipeline package

Modules:
- dicom_reader: read best DICOM series / convert DICOM folder to SITK image
- xml_parser: utilities for parsing LIDC XML contours
- mask_generator: rasterize contours -> 3D masks
- resampler: resample images and masks to target spacing
- normalizer: HU clipping and normalization
- pipeline: orchestrates full end-to-end processing for a case or entire dataset
"""
from .dicom_reader import load_best_dicom_series, dicom_folder_to_image
from .xml_parser import parse_lidc_xml
from .mask_generator import xml_to_mask, save_mask_from_array
from .resampler import resample_image, resample_mask
from .normalizer import clip_and_normalize
from .pipeline import process_case, process_all_cases

__all__ = [
    "load_best_dicom_series",
    "dicom_folder_to_image",
    "parse_lidc_xml",
    "xml_to_mask",
    "save_mask_from_array",
    "resample_image",
    "resample_mask",
    "clip_and_normalize",
    "process_case",
    "process_all_cases",
]
