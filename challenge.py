import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
from PIL import Image
from typing import Any


# Config and Utilities

@dataclass
class Config:
    data_root: Path = Path("dataset")
    train_root: Path = Path("dataset") / "train"
    test_root: Path = Path("dataset") / "test1"

    image_dir: str = "images"
    scribbles_dir: str = "scribbles"
    gt_dir: str = "ground_truth"
    pred_dirname: str = "predictions"

    img_size: Tuple[int, int] = (480, 352)  # (W, H)
    img_mean: Tuple[float, float, float] = (0.4589, 0.4589, 0.4215)
    img_std: Tuple[float, float, float] = (0.2618, 0.2633, 0.2822)

    batch_size: int = 6  # for my 4GB GPU
    num_workers: int = 4
    epochs: int = 80
    lr: float = 2e-3
    weight_decay: float = 5e-5
    dropout: float = 0.1
    amp: bool = True

    encoder_name: str = "efficientnet-b2"
    encoder_weights: Optional[str] = None  # None to train encoder from scratch
    in_channels: int = 4  # RGB + scribble channel
    classes: int = 1

    checkpoints_root: Path = Path("checkpoints")
    save_every: int = 5
    seed: int = 42
    patience: int = 15  # early stopping on val mIoU


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_global_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except Exception:
        pass


def seed_worker(worker_id: int):
    # Ensures each dataloader worker is deterministic and based on PyTorch seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Dataset

class ScribbleSegDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        scribbles: np.ndarray,
        filenames: list[str],
        ground_truth: Optional[np.ndarray] = None,
        transforms: Optional[A.Compose] = None,
    ):
        self.images = images
        self.scribbles = scribbles
        self.filenames = filenames
        self.ground_truth = ground_truth
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx]
        scr = self.scribbles[idx]
        gt = None if self.ground_truth is None else self.ground_truth[idx]

        # Build a 4th channel from scribble: 1.0 fg, 0.0 bg, 0.5 unlabeled
        scr_ch = np.full(scr.shape, 0.5, dtype=np.float32)
        scr_ch[scr == 1] = 1.0
        scr_ch[scr == 0] = 0.0

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=(gt if gt is not None else scr), scribble=scr)
            img = aug["image"]
            if gt is not None:
                gt = aug["mask"]
            scr = aug["scribble"]

        # Normalize image and to tensor below in transforms and build scribble channel now
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            img_t = img

        if isinstance(scr, np.ndarray):
            scr_ch = np.full(scr.shape, 0.5, dtype=np.float32)
            scr_ch[scr == 1] = 1.0
            scr_ch[scr == 0] = 0.0
            scr_t = torch.from_numpy(scr_ch).unsqueeze(0)
        else:
            scr_np = scr.cpu().numpy() if torch.is_tensor(scr) else scr
            scr_ch = np.full(scr_np.shape, 0.5, dtype=np.float32)
            scr_ch[scr_np == 1] = 1.0
            scr_ch[scr_np == 0] = 0.0
            scr_t = torch.from_numpy(scr_ch).unsqueeze(0)

        x = torch.cat([img_t, scr_t], dim=0)

        if gt is not None:
            if isinstance(gt, np.ndarray):
                y = torch.from_numpy(gt.astype(np.float32))
            else:
                y = gt.float()
            y = (y > 0.5).float()
            return {"image": x, "mask": y.unsqueeze(0), "id": self.filenames[idx]}
        else:
            return {"image": x, "mask": None, "id": self.filenames[idx]}


def get_transforms(cfg: Config, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(cfg.img_size), interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(min_height=cfg.img_size[1], min_width=cfg.img_size[0], border_mode=cv2.BORDER_REFLECT_101),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3),
            ],
            additional_targets={"scribble": "mask"},
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(cfg.img_size), interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(min_height=cfg.img_size[1], min_width=cfg.img_size[0], border_mode=cv2.BORDER_REFLECT_101),
            ],
            additional_targets={"scribble": "mask"},
        )


# Losses and Metrics

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.smooth
        dice = 1 - (num / den)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


@torch.inference_mode()
def compute_batch_iou(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    probs = torch.sigmoid(preds)
    pred = (probs >= 0.5).float()
    gt = targets.float()
    tp = (pred * gt).sum(dim=(1, 2, 3))
    fp = (pred * (1 - gt)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * gt).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - gt)).sum(dim=(1, 2, 3))
    iou_fg = tp / torch.clamp(tp + fp + fn, min=1)
    iou_bg = tn / torch.clamp(tn + fp + fn, min=1)
    miou = 0.5 * (iou_fg + iou_bg)
    return miou.mean().item(), iou_bg.mean().item(), iou_fg.mean().item()


# Training and Validation

def train_one_epoch(model, dl, optimizer, scaler, device, criterion, logger, epoch):
    model.train()
    running_loss = 0.0
    running_miou = 0.0
    running_bg_iou = 0.0
    running_fg_iou = 0.0

    pbar = tqdm(dl, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            from torch.amp.autocast_mode import autocast
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        miou, bg_iou, fg_iou = compute_batch_iou(logits, y)
        running_loss += loss.item() * x.size(0)
        running_miou += miou * x.size(0)
        running_bg_iou += bg_iou * x.size(0)
        running_fg_iou += fg_iou * x.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{miou:.3f}"})

    n = len(dl.dataset)
    return {
        "loss": running_loss / n,
        "miou": running_miou / n,
        "bg_iou": running_bg_iou / n,
        "fg_iou": running_fg_iou / n,
    }


@torch.inference_mode()
def validate_one_epoch(model, dl, device, criterion, logger, epoch):
    model.eval()
    running_loss = 0.0
    running_miou = 0.0
    running_bg_iou = 0.0
    running_fg_iou = 0.0

    pbar = tqdm(dl, desc=f"Epoch {epoch} [valid]", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        logits = model(x)
        loss = criterion(logits, y)
        miou, bg_iou, fg_iou = compute_batch_iou(logits, y)

        running_loss += loss.item() * x.size(0)
        running_miou += miou * x.size(0)
        running_bg_iou += bg_iou * x.size(0)
        running_fg_iou += fg_iou * x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{miou:.3f}"})

    n = len(dl.dataset)
    return {
        "loss": running_loss / n,
        "miou": running_miou / n,
        "bg_iou": running_bg_iou / n,
        "fg_iou": running_fg_iou / n,
    }


# Loading dataset

def _open_image(path: str | Path, convert_to: str | None) -> np.ndarray:
    img = Image.open(path)
    if convert_to == "RGB":
        img = img.convert("RGB")
    elif convert_to == "grayscale":
        img = img.convert("L")
    return np.array(img)


def _get_file_names(folder: str | Path) -> list[str]:
    p = Path(folder)
    return sorted([f.name for f in p.iterdir() if f.is_file() and not f.name.startswith('.')])


def _load_images(folder_path: str | Path, folder_name: str, convert_to: str | None) -> np.ndarray:
    image_dir_path = Path(folder_path) / folder_name
    filenames = _get_file_names(image_dir_path)
    filepaths = [image_dir_path / filename for filename in filenames]
    return np.stack([_open_image(fp, convert_to) for fp in filepaths])


def _get_palette(folder_path: str | Path, ground_truth_dir: str, filename: str) -> Any:
    gt_dir_path = Path(folder_path) / ground_truth_dir
    filepath = gt_dir_path / filename
    return Image.open(filepath).getpalette()


def _get_filenames(folder_path: str | Path, scribbles_dir: str) -> list[str]:
    sc_dir_path = Path(folder_path) / scribbles_dir
    return _get_file_names(sc_dir_path)


def load_dataset(
    folder_path: str,
    images_dir: str,
    scribbles_dir: str,
    ground_truth_dir: str | None = None,
) -> Any:
    images = _load_images(folder_path, images_dir, "RGB")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)
    if ground_truth_dir is None:
        return images, scribbles, filenames
    ground_truth = _load_images(folder_path, ground_truth_dir, None)
    palette = _get_palette(folder_path, ground_truth_dir, filenames[0])
    return images, scribbles, ground_truth, filenames, palette


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--encoder", type=str, default=Config.encoder_name)
    parser.add_argument("--pretrained", type=str, default="none", help="None or imagenet")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--patience", type=int, default=Config.patience, help="Early stopping patience (epochs) on val mIoU")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.encoder_name = args.encoder
    cfg.encoder_weights = None if args.pretrained.lower() == "none" else args.pretrained
    cfg.amp = not args.no_amp
    cfg.seed = args.seed
    cfg.patience = args.patience

    ts = time.strftime("%Y%m%d-%H%M%S")
    ckpt_dir = cfg.checkpoints_root / f"UNet_{cfg.encoder_name}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(ckpt_dir)
    logger.info(f"Checkpoints: {ckpt_dir}")
    # Set seeds before any RNG use
    set_global_seeds(cfg.seed, deterministic=True)
    logger.info(f"Seed: {cfg.seed} | Deterministic: True")

    # Load data arrays
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
        str(cfg.train_root), cfg.image_dir, cfg.scribbles_dir, cfg.gt_dir
    )

    # Seeded shuffled split: 80/20
    n = images_train.shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    tr = ScribbleSegDataset(
        images_train[tr_idx], scrib_train[tr_idx], [fnames_train[i] for i in tr_idx.tolist()], ground_truth=gt_train[tr_idx], transforms=get_transforms(cfg, train=True)
    )
    va = ScribbleSegDataset(
        images_train[va_idx], scrib_train[va_idx], [fnames_train[i] for i in va_idx.tolist()], ground_truth=gt_train[va_idx], transforms=get_transforms(cfg, train=False)
    )

    pin = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_dl = DataLoader(
        tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_dl = DataLoader(
        va,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        logger.info(f"Device: cuda ({gpu_name})")
    else:
        logger.info("Device: cpu")

    # Model
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
    ).to(device)
    logger.info(f"Model: UNet/{cfg.encoder_name}, encoder_weights={cfg.encoder_weights}")
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = (device.type == "cuda") and cfg.amp
    from torch.amp.grad_scaler import GradScaler as AmpGradScaler
    scaler = AmpGradScaler(enabled=use_amp) if use_amp else None
    criterion = DiceBCELoss()

    best_miou = -1.0
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    early = EarlyStopping(patience=cfg.patience, mode="max", min_delta=1e-4)

    for epoch in range(1, cfg.epochs + 1):
        tr_metrics = train_one_epoch(model, train_dl, optimizer, scaler, device, criterion, logger, epoch)
        va_metrics = validate_one_epoch(model, val_dl, device, criterion, logger, epoch)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train: loss={tr_metrics['loss']:.4f}, mIoU={tr_metrics['miou']:.4f}, bg={tr_metrics['bg_iou']:.4f}, fg={tr_metrics['fg_iou']:.4f} | "
            f"val: loss={va_metrics['loss']:.4f}, mIoU={va_metrics['miou']:.4f}, bg={va_metrics['bg_iou']:.4f}, fg={va_metrics['fg_iou']:.4f}"
        )

        # Save last
        torch.save(model.state_dict(), last_path)
        # Save best
        if va_metrics["miou"] > best_miou:
            best_miou = va_metrics["miou"]
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best mIoU: {best_miou:.4f} -> saved {best_path}")

        if epoch % cfg.save_every == 0:
            snap = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), snap)

        # Early stopping check (on val mIoU)
        if early.step(va_metrics["miou"]):
            logger.info(
                f"Early stopping triggered after {early.num_bad_epochs} bad epochs. Best mIoU={early.best:.4f}."
            )
            break

    # Inference on full train and test1
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Train predictions (full)
    images_all, scrib_all, gt_all, fnames_all, palette = load_dataset(
        str(cfg.train_root), cfg.image_dir, cfg.scribbles_dir, cfg.gt_dir
    )
    train_pred_dir = cfg.train_root / cfg.pred_dirname
    train_pred_dir.mkdir(exist_ok=True)

    pbar = tqdm(range(images_all.shape[0]), desc="Predict train", leave=False)
    FINAL_SIZE = (500, 375)  # (W, H) as in provided baseline
    for i in pbar:
        img = images_all[i]
        scr = scrib_all[i]
        tfm = get_transforms(cfg, train=False)
        aug = tfm(image=img, mask=scr)
        img_aug = aug["image"]
        scr_aug = aug["mask"]

        x_img = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        x_scr = torch.from_numpy(np.where(scr_aug == 255, 0.5, scr_aug).astype(np.float32))
        x_scr = torch.where(x_scr == 1, torch.tensor(1.0), torch.where(x_scr == 0, torch.tensor(0.0), torch.tensor(0.5)))
        x = torch.cat([x_img, x_scr.unsqueeze(0)], dim=0).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred = (prob >= 0.5).astype(np.uint8)
        # Save 1-bit PNG resized to FINAL_SIZE
        out_path = train_pred_dir / (fnames_all[i])
        img_pil = Image.fromarray(pred * 255)
        img_pil = img_pil.convert("1", dither=Image.Dither.NONE)
        img_pil = img_pil.resize(FINAL_SIZE, Image.Resampling.NEAREST)
        img_pil.save(out_path, optimize=True)

    # test1 predictions
    images_test, scrib_test, fnames_test = load_dataset(
        str(cfg.test_root), cfg.image_dir, cfg.scribbles_dir
    )
    test_pred_dir = cfg.test_root / cfg.pred_dirname
    test_pred_dir.mkdir(exist_ok=True)

    pbar = tqdm(range(images_test.shape[0]), desc="Predict test1", leave=False)
    for i in pbar:
        img = images_test[i]
        scr = scrib_test[i]
        tfm = get_transforms(cfg, train=False)
        aug = tfm(image=img, mask=scr)
        img_aug = aug["image"]
        scr_aug = aug["mask"]

        x_img = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        x_scr = torch.from_numpy(np.where(scr_aug == 255, 0.5, scr_aug).astype(np.float32))
        x_scr = torch.where(x_scr == 1, torch.tensor(1.0), torch.where(x_scr == 0, torch.tensor(0.0), torch.tensor(0.5)))
        x = torch.cat([x_img, x_scr.unsqueeze(0)], dim=0).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)
        # Save PNGs
        out_path = test_pred_dir / (fnames_test[i])
        img_pil = Image.fromarray(pred * 255)
        img_pil = img_pil.convert("1", dither=Image.Dither.NONE)
        img_pil = img_pil.resize(FINAL_SIZE, Image.Resampling.NEAREST)
        img_pil.save(out_path, optimize=True)


if __name__ == "__main__":
    main()
