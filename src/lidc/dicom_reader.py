# src/lidc/dicom_reader.py
"""
🔥 UPGRADED DICOM LOADER
Handles deep LIDC directory trees:

Case/
 ├─ f1/
 │   └─ f1.f1/
 │        └─ *.dcm
 ├─ f2/
 │   └─ f2.f1/
 │        └─ *.dcm
 └─ XML

Finds ALL nested DICOM folders, picks the largest series automatically.
"""

from pathlib import Path
import SimpleITK as sitk

def find_all_dicom_series(case_dir: Path):
    """
    Deep-scan ANY nested folder for .dcm files.
    Returns list of folders that contain DICOM files.
    """
    dicom_folders = []

    # scan all subdirectories recursively
    for p in case_dir.rglob("*"):
        if p.is_dir() and any(p.glob("*.dcm")):
            dicom_folders.append(p)

    return dicom_folders


def load_best_dicom_series(case_dir: str):
    """
    Automatically discover & load BEST (largest) DICOM series under ANY nested dirs.
    Returns:
        sitk.Image, series_folder_name
    """
    case_dir = Path(case_dir)

    # find all dicom series
    series_dirs = find_all_dicom_series(case_dir)

    if not series_dirs:
        raise FileNotFoundError(f"No DICOM files found under {case_dir}")

    # Pick series with most .dcm files
    series_dirs = sorted(series_dirs, key=lambda d: len(list(d.glob("*.dcm"))), reverse=True)
    best_series = series_dirs[0]

    # Load this series
    dicom_files = sorted([str(f) for f in best_series.glob("*.dcm")])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    img = reader.Execute()

    return img, best_series.name


def dicom_folder_to_image(dicom_folder: str):
    """Simple wrapper."""
    img, series_name = load_best_dicom_series(dicom_folder)
    return img
