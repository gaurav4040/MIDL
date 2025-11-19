# src/lidc/pipeline.py
"""
🔥 UPGRADED LIDC PIPELINE (MULTIPROCESSING + TQDM + LOGGING)
Raw LIDC (DICOM + XML) → NIfTI (1mm, clipped, normalized) + consensus masks.

This file orchestrates:
 - best DICOM series selection
 - XML parsing
 - mask rasterization
 - resampling
 - lung window normalization
 - saving final training-ready NIfTI

✨ New features:
 - tqdm progress bars
 - multiprocessing
 - robust error handling
 - colored logs
 - config.yaml support
 - faster batch processing
"""

import os
from pathlib import Path
import SimpleITK as sitk
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import yaml

from .dicom_reader import load_best_dicom_series
from .mask_generator import xml_to_mask
from .resampler import resample_image, resample_mask
from .normalizer import clip_and_normalize


# -------------------------------
# 💠 Color logging helpers
# -------------------------------
def ok(msg): print(f"\033[92m{msg}\033[0m")       # green
def info(msg): print(f"\033[96m{msg}\033[0m")    # cyan
def warn(msg): print(f"\033[93m{msg}\033[0m")    # yellow
def error(msg): print(f"\033[91m{msg}\033[0m")   # red


# -------------------------------
# 💠 Load config.yaml if exists
# -------------------------------
def load_config():
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# -------------------------------
# 💠 Process one case
# -------------------------------
def process_case(params):
    case_dir, out_vol_dir, out_mask_dir, merge_strategy, target_spacing = params

    case_dir = Path(case_dir)
    case_id = case_dir.name
    start = time.time()

    try:
        # Load best DICOM series
        img, series = load_best_dicom_series(case_dir)
    except Exception as e:
        return f"[FAIL] {case_id} — DICOM error: {e}"

    # -------------------
    # XML detection
    # -------------------
    xml_path = None
    for f in case_dir.rglob("*.xml"):
        xml_path = f
        break

    if xml_path is None:
        warn(f"[WARN] No XML for {case_id}, empty mask used.")
        mask_np = None
        # Create empty mask
        mask_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        mask_img.CopyInformation(img)
    else:
        try:
            mask_np = xml_to_mask(str(xml_path), img, merge_strategy)
            mask_img = sitk.GetImageFromArray(mask_np.astype("uint8"))
            mask_img.CopyInformation(img)
        except Exception as e:
            return f"[FAIL] {case_id} — Mask generation error: {e}"

    # -------------------
    # Resample & normalize
    # -------------------
    try:
        img_rs = resample_image(img, new_spacing=target_spacing, interpolator=sitk.sitkLinear)
        mask_rs = resample_mask(mask_img, new_spacing=target_spacing)
        img_norm = clip_and_normalize(img_rs)
    except Exception as e:
        return f"[FAIL] {case_id} — Resample/normalize error: {e}"

    # -------------------
    # Save outputs
    # -------------------
    try:
        vol_path = Path(out_vol_dir) / f"{case_id}.nii.gz"
        mask_path = Path(out_mask_dir) / f"{case_id}.nii.gz"
        sitk.WriteImage(img_norm, str(vol_path))
        sitk.WriteImage(mask_rs, str(mask_path))
    except Exception as e:
        return f"[FAIL] {case_id} — Save error: {e}"

    elapsed = time.time() - start
    return f"[OK] {case_id} ✔️ processed in {elapsed:.2f}s"


# -------------------------------
# 💠 Batch processor
# -------------------------------
def process_all_cases(
        raw_root="patientData/LIDC-IDRI",
        out_vol_dir="data/volumes",
        out_mask_dir="data/masks",
        merge_strategy="consensus",
        target_spacing=(1.0, 1.0, 1.0),
        workers=None
):
    """
    Multiprocessing, progress bars, logging, auto XML detection.
    """
    cfg = load_config()

    # override from config.yaml if available
    raw_root = cfg.get("raw_root", raw_root)
    out_vol_dir = cfg.get("out_volumes", out_vol_dir)
    out_mask_dir = cfg.get("out_masks", out_mask_dir)
    merge_strategy = cfg.get("merge_strategy", merge_strategy)

    raw_root = Path(raw_root)
    Path(out_vol_dir).mkdir(parents=True, exist_ok=True)
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)

    info(f"🚀 Starting LIDC preprocessing")
    info(f"📂 Raw root:      {raw_root}")
    info(f"💾 Output volumes: {out_vol_dir}")
    info(f"💾 Output masks:   {out_mask_dir}")
    info(f"⚙️ Merge strategy: {merge_strategy}")
    info(f"🧵 Target spacing: {target_spacing}")
    print("")

    cases = [d for d in sorted(raw_root.iterdir()) if d.is_dir() and d.name.upper().startswith("LIDC-IDRI")]

    if workers is None:
        workers = max(1, cpu_count() - 1)

    info(f"🧠 Using {workers} CPU cores")
    print("")

    # prepare multiprocessing job list
    tasks = [
        (case, out_vol_dir, out_mask_dir, merge_strategy, target_spacing)
        for case in cases
    ]

    results = []
    with Pool(processes=workers) as p:
        for res in tqdm(p.imap(process_case, tasks), total=len(tasks), desc="Processing LIDC cases", colour="cyan"):
            results.append(res)

    print("\n-------- SUMMARY --------")
    for r in results:
        if r.startswith("[OK]"):
            ok(r)
        else:
            error(r)

    ok("\n🎉 LIDC preprocessing completed!\n")
