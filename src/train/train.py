"""
Main training script (fine-tune) — upgraded & disk-safe.
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
from torch.utils.data import DataLoader
from src.data.dataset import NiftiPatchDataset
from src.models.hybrid3d import Hybrid3DNet
from src.models.losses import multi_task_loss
from src.utils.metrics import dice_metric

# -------------------------
# Helpers
# -------------------------
def get_free_gb(path="."):
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)   # convert to GB

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for vol, mask, cls in loader:
            vol = vol.to(device); mask = mask.to(device)
            seg_logits, _, _ = model(vol)
            dice = dice_metric(torch.sigmoid(seg_logits).cpu(), mask.cpu())
            dices.append(dice)
    return float(np.mean(dices)) if len(dices) > 0 else 0.0

# -------------------------
# Training
# -------------------------
def train(cfg_path):
    # load config (utf-8 safe)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get('training', {}).get('seed', 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------
    # dataset
    # --------------------------
    ds = NiftiPatchDataset(
        cfg['data']['volumes_dir'],
        cfg['data']['masks_dir'],
        patch_size=tuple(cfg['data']['patch_size']),
        augment=True
    )

    n = len(ds)
    if n == 0:
        print("❌ No samples found in dataset. Check data paths in config.")
        return

    indices = list(range(n))
    random.shuffle(indices)

    # 60/20/20 split
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"Total cases: {n}")
    print(f"Train: {len(train_idx)} ({len(train_idx)/n:.2%})")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/n:.2%})")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/n:.2%})")

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx),
        batch_size=int(cfg['training']['batch_size']),
        shuffle=True,
        num_workers=int(cfg['data'].get('num_workers', 4)),
        pin_memory=bool(cfg['data'].get('pin_memory', True))
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(ds, val_idx),
        batch_size=1, shuffle=False, num_workers=2
    )

    test_loader = DataLoader(
        torch.utils.data.Subset(ds, test_idx),
        batch_size=1, shuffle=False, num_workers=2
    )

    # --------------------------
    # model & optimizer
    # --------------------------
    model = Hybrid3DNet(
        in_ch=1,
        base=int(cfg['model']['base_channels']),
        transformer_cfg=cfg['model'].get('transformer', {}),
        mc_dropout=bool(cfg['model'].get('mc_dropout', True))
    ).to(device)

    # optional SSL weights loader (safe)
    ssl_path = cfg.get('training', {}).get('ssl_weights', "ssl_encoder.pth")
    if ssl_path and os.path.exists(ssl_path):
        try:
            sd = torch.load(ssl_path, map_location=device)
            model_state = model.state_dict()
            loaded = 0
            for k, v in sd.items():
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
                    loaded += 1
            model.load_state_dict(model_state)
            print(f"Loaded SSL weights ({loaded} tensors) from {ssl_path}")
        except Exception as e:
            print("⚠️ Warning: failed to load SSL weights:", e)

    # safe-cast optimizer params
    lr = float(cfg['training'].get('lr', 1e-4))
    weight_decay = float(cfg['training'].get('weight_decay', 1e-5))

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # optional scheduler (simple placeholder)
    scheduler_cfg = cfg['training'].get('scheduler', {}) if 'training' in cfg else {}
    scheduler = None
    if scheduler_cfg.get('enabled', False) and scheduler_cfg.get('type') == 'cosine':
        try:
            total_epochs = int(cfg['training']['epochs'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, total_epochs - int(scheduler_cfg.get('warmup_epochs', 0))))
        except Exception:
            scheduler = None

    # --------------------------
    # training loop
    # --------------------------
    best = 0.0
    MIN_SAVE_GB = float(cfg.get('preprocessing', {}).get('min_save_gb', 2.0)) if cfg.get('preprocessing') else 2.0
    CRITICAL_GB = float(cfg.get('preprocessing', {}).get('critical_gb', 1.0)) if cfg.get('preprocessing') else 1.0

    epochs = int(cfg['training'].get('epochs', 200))

    try:
        for epoch in range(epochs):
            # disk check before epoch
            free_gb = get_free_gb()
            if free_gb < CRITICAL_GB:
                print(f"\n❌ CRITICAL LOW DISK: {free_gb:.2f} GB left — stopping training safely.")
                break

            model.train()
            running_loss = 0.0

            for vol, mask, cls in train_loader:
                vol = vol.to(device); mask = mask.to(device); cls = cls.to(device)

                seg_logits, cls_logits, reg_pred = model(vol)
                reg_target = torch.zeros_like(reg_pred).to(device)

                loss = multi_task_loss(seg_logits, mask, cls_logits, cls, reg_pred, reg_target, cfg)
                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss.item()

            # scheduler step
            if scheduler is not None:
                scheduler.step()

            # validation
            val_dice = validate(model, val_loader, device)
            avg_train_loss = running_loss / max(1, len(train_loader))
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f} | Free GB: {get_free_gb():.2f}")

            # DISK-SAFE CHECKPOINTING: save only if improved & enough space
            if val_dice > best:
                free_gb = get_free_gb()
                if free_gb < MIN_SAVE_GB:
                    print(f"🚨 LOW DISK WARNING: Only {free_gb:.2f} GB left! Skipping checkpoint save.")
                else:
                    # atomic save pattern: write to temp then rename
                    tmp_path = "best_model.tmp.pth"
                    final_path = "best_model.pth"
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        torch.save(model.state_dict(), tmp_path)
                        # remove old final if exists (to free space) then rename
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        os.replace(tmp_path, final_path)
                        best = val_dice
                        print(f"💾 Saved best_model.pth (Val Dice={best:.4f}) — Free: {free_gb:.2f} GB")
                    except Exception as e:
                        print("❌ Error saving checkpoint:", e)
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

            # end of epoch cleanup
            torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user — exiting gracefully.")
    except Exception as e:
        print(f"\n❌ Training crashed with exception: {e}")
    finally:
        # final cleanup: ensure no temp checkpoint remains
        if os.path.exists("best_model.tmp.pth"):
            try:
                os.remove("best_model.tmp.pth")
            except Exception:
                pass

    print("Training done.")
    print("Best Val Dice:", best)
    print("Test set size:", len(test_idx))
    print("Use test_loader later for final evaluation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/config.yaml")
    args = p.parse_args()
    train(args.cfg)
