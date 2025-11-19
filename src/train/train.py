"""
Hybrid3D Training Script (Ultra Pretty Console Version)
Usage:
    python -m src.train.train --cfg configs/config.yaml
"""

import argparse
import yaml
import os
import random
import numpy as np
import torch
import shutil
import gc
from tqdm import tqdm
from colorama import Fore, Back, Style, init
init(autoreset=True)

from torch.utils.data import DataLoader
from src.data.dataset import NiftiPatchDataset
from src.models.hybrid3d import Hybrid3DNet
from src.models.losses import multi_task_loss
from src.utils.metrics import dice_metric

# -----------------------------
# Helper Functions
# -----------------------------
def get_free_gb(path="."):
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def c_green(t): return Fore.GREEN + t + Style.RESET_ALL
def c_red(t): return Fore.RED + t + Style.RESET_ALL
def c_yellow(t): return Fore.YELLOW + t + Style.RESET_ALL
def c_blue(t): return Fore.CYAN + t + Style.RESET_ALL
def c_magenta(t): return Fore.MAGENTA + t + Style.RESET_ALL

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------------
# Validation Loop
# -----------------------------
def validate(model, loader, device):
    model.eval()
    dices = []

    with torch.no_grad():
        for vol, mask, cls in tqdm(loader, desc=c_magenta("Validating"), leave=False):
            vol = vol.to(device); mask = mask.to(device)
            seg_logits, _, _ = model(vol)
            dice = dice_metric(torch.sigmoid(seg_logits).cpu(), mask.cpu())
            dices.append(dice)

    return float(np.mean(dices)) if len(dices) > 0 else 0.0

# -----------------------------
# Training Function
# -----------------------------
def train(cfg_path):

    # Load config utf-8 safe
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(c_blue("\n🚀 Starting LUNG-NET Hybrid3D Training...\n"))
    print(c_green(f"Using device: {device}\n"))

    # -----------------------------
    # Load dataset
    # -----------------------------
    ds = NiftiPatchDataset(
        cfg["data"]["volumes_dir"],
        cfg["data"]["masks_dir"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        augment=True
    )

    n = len(ds)
    if n == 0:
        print(c_red("❌ Dataset empty. Check volumes/masks directory!"))
        return

    idx = list(range(n))
    random.shuffle(idx)

    # Split — 60/20/20
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    print(c_blue("📊 Dataset Split:"))
    print(f"  {c_green('Train:')} {len(train_idx)} files ({len(train_idx)/n:.2%})")
    print(f"  {c_yellow('Val:  ')} {len(val_idx)} files ({len(val_idx)/n:.2%})")
    print(f"  {c_magenta('Test: ')} {len(test_idx)} files ({len(test_idx)/n:.2%})\n")

    # Loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(ds, val_idx),
        batch_size=1, shuffle=False, num_workers=2
    )

    # -----------------------------
    # Model init
    # -----------------------------
    model = Hybrid3DNet(
        in_ch=1,
        base=cfg["model"]["base_channels"],
        transformer_cfg=cfg["model"]["transformer"],
        mc_dropout=True
    ).to(device)

    # load SSL weights if available
    if os.path.exists("ssl_encoder.pth"):
        print(c_blue("🔑 Loading SSL weights..."))
        try:
            sd = torch.load("ssl_encoder.pth", map_location=device)
            ms = model.state_dict()
            cnt = 0
            for k in sd:
                if k in ms and sd[k].shape == ms[k].shape:
                    ms[k] = sd[k]; cnt += 1
            model.load_state_dict(ms)
            print(c_green(f"✓ Loaded {cnt} SSL weights.\n"))
        except:
            print(c_red("⚠ SSL load failed.\n"))

    # Optimizer (safe float cast)
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    best = 0.0
    epochs = cfg["training"]["epochs"]

    MIN_SAVE_GB = 2.0
    CRITICAL_GB = 1.0

    for epoch in range(epochs):

        free_gb = get_free_gb()
        if free_gb < CRITICAL_GB:
            print(c_red(f"\n⛔ TRAINING STOPPED — Low Disk ({free_gb:.2f} GB)\n"))
            break

        model.train()
        running_loss = 0.0

        # pretty progress bar
        loop = tqdm(train_loader, desc=c_green(f"Epoch {epoch+1}/{epochs}"), colour="cyan")

        for vol, mask, cls in loop:
            vol = vol.to(device); mask = mask.to(device); cls = cls.to(device)

            seg_logits, cls_logits, reg_pred = model(vol)
            reg_target = torch.zeros_like(reg_pred).to(device)

            loss = multi_task_loss(seg_logits, mask, cls_logits, cls, reg_pred, reg_target, cfg)
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        # validation
        val_dice = validate(model, val_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))

        print(f"\n{c_blue('📈 Epoch Summary:')}")
        print(f"  Train Loss: {c_green(f'{avg_loss:.4f}')}")
        print(f"  Val Dice:   {c_yellow(f'{val_dice:.4f}')}")
        print(f"  Free Disk:  {c_magenta(f'{get_free_gb():.2f} GB')}\n")

        # save best model disk-safe
        if val_dice > best:
            free_gb = get_free_gb()
            if free_gb < MIN_SAVE_GB:
                print(c_red(f"🚨 LOW DISK: {free_gb:.2f} GB — Skip saving model.\n"))
            else:
                try:
                    if os.path.exists("best_model.pth"):
                        os.remove("best_model.pth")
                    torch.save(model.state_dict(), "best_model.pth")
                    best = val_dice
                    print(c_green(f"💾 Saved checkpoint — best_model.pth (Val Dice={best:.4f})\n"))
                except Exception as e:
                    print(c_red(f"❌ Save Failed: {e}"))

        # cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # end
    print(c_blue("\n🎉 Training Finished!"))
    print(c_green(f"Best Val Dice: {best:.4f}"))
    print(c_yellow(f"Test Set Size: {len(test_idx)}\n"))
    print(c_magenta("👉 Use test_eval.py for final evaluation.\n"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/config.yaml")
    args = p.parse_args()
    train(args.cfg)
