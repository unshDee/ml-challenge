import concurrent.futures
import math
import pickle
import time
from functools import cache
from pathlib import Path
from typing import Any, Final

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.core.type_definitions import Targets
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import (
    isotropic_closing,
    isotropic_dilation,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_ROOT = Path("data")
TRAINING_ROOT = DATA_ROOT / "train"
TEST_ROOT = DATA_ROOT / "test1"
CHECKPOINT_ROOT = Path("temp_checkpoints")

IMAGE_DIR = "images"
SCRIBBLES_DIR = "scribbles"
GROUND_TRUTH_DIR = "ground_truth"
PREDICTION_DIR = "predictions"

IMG_SIZE = (480, 352)
IMG_MEAN = (0.4589, 0.4589, 0.4215)
IMG_STD = (0.2618, 0.2633, 0.2822)


BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 5e-5
DROPOUT_RATE = 0.2

EPOCHS = 1000
MAX_PATIENCE = 10

FINAL_SIZE = (500, 375)


class ScribbleDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        images_dir: str | Path,
        scribbles_dir: str | Path,
        ground_truth_dir: str | Path | None = None,
        transforms: A.BasicTransform | A.BaseCompose | None = None,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.images_dir = dataset_root / images_dir

        self.scribbles_dir = dataset_root / scribbles_dir

        self.image_paths = self.get_image_paths(self.images_dir)[start_idx:end_idx]

        self.scribble_paths = self.get_image_paths(self.scribbles_dir)[
            start_idx:end_idx
        ]

        if ground_truth_dir is not None:
            self.ground_truth_dir = dataset_root / ground_truth_dir
            self.ground_truth_paths = self.get_image_paths(self.ground_truth_dir)[
                start_idx:end_idx
            ]
        else:
            self.ground_truth_dir = None
            self.ground_truth_paths = None

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        scribble_path = self.scribble_paths[idx]

        image = self.load_cv2_image(image_path, mode="RGB")
        scribble = self.transform_scribble(self.load_cv2_image(scribble_path, mode="L"))
        if self.ground_truth_paths is not None:
            ground_truth_path = self.ground_truth_paths[idx]
            ground_truth = self.load_cv2_image(ground_truth_path, mode="L").clip(
                None, 1
            )
        else:
            ground_truth = None

        if self.transforms is not None:
            if ground_truth is not None:
                transformed = self.transforms(
                    image=image,
                    scribble=scribble,
                    mask=ground_truth,
                )
                ground_truth = transformed["mask"]
            else:
                transformed = self.transforms(
                    image=image,
                    scribble=scribble,
                )
            image = transformed["image"]
            scribble = transformed["scribble"]

        return {
            "id": image_path.stem,
            "image": image,
            "scribble": scribble,
            "ground_truth": -1 if ground_truth is None else ground_truth,
        }

    @staticmethod
    def transform_scribble(
        scribble: np.ndarray | Image.Image,
    ) -> np.ndarray:
        np_scribble = np.asarray(scribble, dtype=np.int16)

        np_scribble[np_scribble == 0] = -1

        np_scribble[np_scribble == 255] = 0

        return np_scribble

    @staticmethod
    def load_cv2_image(
        image_path: Path | str,
        mode: str | None = None,
    ) -> cv2.typing.MatLike:
        """Load an image from a file path."""
        if mode == "RGB":
            flags = cv2.IMREAD_COLOR
        elif mode == "L":
            flags = cv2.IMREAD_GRAYSCALE
        else:
            flags = cv2.IMREAD_UNCHANGED

        img = cv2.imread(str(image_path), flags)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        if flags == cv2.IMREAD_COLOR:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if flags == cv2.IMREAD_GRAYSCALE:
            return img[..., None]  # Ensure single channel for grayscale

        return img

    @staticmethod
    @cache
    def get_image_paths(image_dir: Path) -> list[Path]:
        image_paths = [*image_dir.glob("*.png"), *image_dir.glob("*.jpg")]
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {image_dir}. Supported formats: .png, .jpg"
            )

        image_paths.sort()
        return image_paths


class SyntheticScribbleTransform(A.BasicTransform):
    _targets = Targets.MASK

    def __init__(
        self,
        mask_key="mask",
        line_radius: int = 2,
        closing_radius: int = 5,
        bg_fg_margin: int = 10,
        min_connected_component_area: int = 50,
        p=0.5,
    ):
        super().__init__(p)

        self.mask_key = mask_key
        self.line_radius = line_radius
        self.closing_radius = closing_radius
        self.bg_fg_margin = bg_fg_margin
        self.min_connected_component_area = min_connected_component_area

    @property
    def targets(self):
        return {
            "scribble": self.apply_to_scribble,
        }

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        for k, v in additional_targets.items():
            if k in self._additional_targets and v != self._additional_targets[k]:
                raise ValueError(
                    f"Trying to overwrite existed additional targets. "
                    f"Key={k} Exists={self._additional_targets[k]} New value: {v}",
                )
            if v in self.targets:
                self._additional_targets[k] = v
                self._key2func[k] = self.targets[v]
                self._available_keys.add(k)

    @property
    def targets_as_params(self):
        return [self.mask_key]

    def get_params_dependent_on_data(
        self, params, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "scribble_mask": data[self.mask_key],
        }

    def get_foreground_line_configuration(self, region: Any):
        min_row, min_col, max_row, max_col = region.bbox

        # It is exclusive, so we need to adjust
        # Reduce by 2 to account for rounding errors
        max_row -= 2
        max_col -= 2

        if max_row <= min_row or max_col <= min_col:
            # print("Invalid region dimensions, skipping line generation.")
            return None

        for _ in range(10):  # Try up to 10 times to find a valid line
            try:
                return self.get_random_line(
                    min_row,
                    min_col,
                    max_row,
                    max_col,
                    mask=region.image,
                )
            except ValueError:
                pass
                # print(f"Skipping line due to error: {e}")

        return None

    def get_random_line(
        self,
        min_row: int,
        min_col: int,
        max_row: int,
        max_col: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        line_center = (
            0.5 * (min_col + max_col),
            0.5 * (min_row + max_row),
        )
        angle = self.py_random.uniform(-math.pi, math.pi)

        return self.get_line_indices_from_angle(
            center=line_center,
            angle=angle,
            col_limit=(min_col, max_col),
            row_limit=(min_row, max_row),
            mask=mask,
            mask_start_idx=(min_row, min_col),
        )

    def get_background_line_configuration(self, mask: np.ndarray):
        y, x = np.nonzero(mask)

        if len(x) == 0 and len(y) == 0:
            return None

        min_row, max_row = y.min(), y.max()
        min_col, max_col = x.min(), x.max()

        image_height, image_width, *_ = mask.shape

        # Reduce 2 to make it inclusive
        image_height -= 2
        image_width -= 2

        # Keep the margins
        min_row = max(0, min_row - self.bg_fg_margin)
        min_col = max(0, min_col - self.bg_fg_margin)

        max_row = min(image_height, max_row + self.bg_fg_margin)
        max_col = min(image_width, max_col + self.bg_fg_margin)

        area_proposals = (
            ((0, 0), (image_width, min_row)),  # Top area
            ((0, max_row), (image_width, image_height)),  # Bottom area
            ((0, 0), (min_col, image_height)),  # Left area
            ((max_col, 0), (image_width, image_height)),  # Right area
        )

        pixel_areas = [(x2 - x1) * (y2 - y1) for (x1, y1), (x2, y2) in area_proposals]

        if not any(pixel_areas):
            # print("No valid area proposals found.")
            return None

        (min_col, min_row), (max_col, max_row) = self.py_random.choices(
            area_proposals, weights=pixel_areas
        )[0]

        for _ in range(10):
            try:
                # Try to get a random line in the background area
                # This will raise ValueError if no valid line can be found
                # or if the line is too short
                return self.get_random_line(
                    min_row,
                    min_col,
                    max_row,
                    max_col,
                )
            except ValueError:
                pass
                # print(f"Skipping line due to error: {e}")

        return None

    def apply_to_scribble(
        self, scribble: np.ndarray, scribble_mask: np.ndarray, **kwargs
    ):
        scribble_mask_2d = scribble_mask.squeeze()

        if scribble_mask_2d.ndim != 2:
            raise ValueError("Scribble mask must be a 2D array.")

        new_scribble = np.zeros_like(scribble_mask_2d, dtype=bool)

        fg_mask = isotropic_closing(
            scribble_mask_2d, radius=self.closing_radius
        ).astype(bool)
        if not np.any(fg_mask):
            print("Foreground was empty, skipping scribble generation.")
            return scribble

        labeled_fg = label(fg_mask)
        success = False
        for region in regionprops(labeled_fg):
            if region.area < self.min_connected_component_area:
                continue

            line_indices = self.get_foreground_line_configuration(region)
            if line_indices is None:
                continue

            fg_y_indices, fg_x_indices = line_indices

            new_scribble[fg_y_indices, fg_x_indices] = True
            success = True

        if not success:
            # print("No foreground line found, reverting to original scribble.")
            return scribble

        # Increase the width of the scribble lines
        new_scribble = isotropic_dilation(new_scribble, radius=self.line_radius)

        bg_indices = self.get_background_line_configuration(scribble_mask_2d)

        if bg_indices is None:
            return new_scribble.astype(np.int16)[..., None]  # Ensure single channel

        bg_scribble = np.zeros_like(new_scribble, dtype=bool)

        bg_y_indices, bg_x_indices = bg_indices
        bg_scribble[bg_y_indices, bg_x_indices] = True

        bg_scribble = isotropic_dilation(bg_scribble, radius=self.line_radius)

        return (new_scribble.astype(np.int16) - bg_scribble)[
            ..., None
        ]  # Ensure single channel

    @staticmethod
    def get_line_indices_from_angle(
        center: tuple[float, float],
        angle: float,
        col_limit: tuple[int, int],
        row_limit: tuple[int, int],
        mask: np.ndarray | None = None,
        mask_start_idx: tuple[int, int] = (0, 0),
    ) -> tuple[np.ndarray, np.ndarray]:
        c_x, c_y = center
        if c_x < 0 or c_y < 0:
            raise ValueError(f"Center coordinates must be non-negative. Got: {center}")

        if abs(angle) > math.pi:
            raise ValueError(f"Angle must be in the range [-pi, pi]. Got: {angle}")

        min_col, max_col = col_limit
        min_row, max_row = row_limit

        if abs(angle) < math.pi / 4:
            # print("Slope is shallow, using x as the independent variable.")
            x = np.arange(min_col, max_col + 1)
            y = np.round(c_y + (math.tan(angle) * (x - c_x))).astype(int)

            i = -1
            for i, _y in enumerate(y):
                if _y >= min_row and _y <= max_row:
                    break

            if i == -1:
                raise RuntimeError(
                    "The line computed was empty, this should not happen."
                )

            if y[i] < min_row or y[i] > max_row:
                raise ValueError("No valid point found in the mask along the line.")

            y = y[i:]
            x = x[i:]

            j = 0
            for j, _y in enumerate(y[::-1], 1):
                if _y >= min_row and _y <= max_row:
                    break

            if j == 0 or y[-j] < min_row or y[-j] > max_row:
                raise ValueError("No valid points found in the mask along the line.")

            if j > 1:
                y = y[: -j + 1]
                x = x[: -j + 1]

        else:
            # print("Slope is steep, using y as the independent variable.")
            y = np.arange(min_row, max_row + 1)
            x = np.round(c_x + ((y - c_y) / math.tan(angle))).astype(int)

            i = -1
            for i, _x in enumerate(x):
                if _x >= min_col and _x <= max_col:
                    break

            if i == -1:
                raise RuntimeError(
                    "The line computed was empty, this should not happen."
                )
            if x[i] < min_col or x[i] > max_col:
                raise ValueError("No valid point found in the mask along the line.")

            x = x[i:]
            y = y[i:]
            j = 0
            for j, _x in enumerate(x[::-1], 1):
                if _x >= min_col and _x <= max_col:
                    break

            if j == 0 or x[-j] < min_col or x[-j] > max_col:
                raise ValueError("No valid points found in the mask along the line.")

            if j > 1:
                x = x[: -j + 1]
                y = y[: -j + 1]

        if mask is not None:
            y_start, x_start = mask_start_idx

            x -= x_start
            y -= y_start

            i = -1
            for i, (_x, _y) in enumerate(zip(x, y, strict=True)):
                if mask[_y, _x] != 0:
                    break

            if i == -1:
                raise RuntimeError(
                    "The line computed was empty, this should not happen."
                )

            if mask[y[i], x[i]] == 0:
                raise ValueError("No valid point found in the mask along the line.")

            x = x[i:]
            y = y[i:]

            j = 0
            for j, (_x, _y) in enumerate(zip(x[::-1], y[::-1], strict=True), 1):
                if mask[_y, _x] != 0:
                    break

            if j == 0 or mask[y[-j], x[-j]] == 0:
                raise ValueError("No valid points found in the mask along the line.")

            if j > 1:
                x = x[: -j + 1]
                y = y[: -j + 1]

            x += x_start
            y += y_start

        return y, x  # Return as (y, x) pairs for plotting


class DiceCELoss(nn.Module):
    def __init__(
        self, dice_coeff: float = 1, ce_coeff: float = 0.2, smooth: float = 1e-8
    ):
        super().__init__()
        if dice_coeff < 0 or ce_coeff < 0:
            raise ValueError("Coefficients must be non-negative.")

        if dice_coeff == 0:
            msg = "Dice coefficient is zero, better to use CrossEntropyLoss directly."
            raise ValueError(msg)

        if ce_coeff == 0 and dice_coeff != 1:
            msg = "No use for Dice loss if ce_coeff is zero and dice_coeff is not 1."
            raise ValueError(msg)

        self.dice_coeff = dice_coeff
        self.ce_criterion = nn.BCEWithLogitsLoss() if ce_coeff > 0 else None
        self.ce_coeff = ce_coeff
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob_inputs = torch.sigmoid(inputs).flatten(1)

        targets = targets.float().flatten(1)

        intersection = (prob_inputs * targets).sum(dim=1)
        union = prob_inputs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1 - dice_score.mean()

        if self.ce_criterion is None:
            return dice_loss

        ce_loss = self.ce_criterion(inputs, targets)

        return self.dice_coeff * dice_loss + self.ce_coeff * ce_loss


def get_optim_groups(model: nn.Module, weight_decay: float):
    if weight_decay <= 0:
        return model.parameters()

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.modules.conv._ConvNd)
    blacklist_weight_modules = (
        nn.Embedding,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.modules.batchnorm._NormBase,  # Base class for batchnorm and instance norm
    )
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            if pn.endswith("proj_weight"):
                # Add project weights to decay set
                decay.add(fpn)
            elif pn.endswith("weight"):
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
            else:
                # all paramters except weight will not be decayed
                no_decay.add(fpn)

    inter_params = decay & no_decay
    if len(inter_params) != 0:
        msg = f"parameters {inter_params} made it into both decay/no_decay sets!"
        raise ValueError(msg)

    # validate that we considered every parameter
    param_dict = dict(model.named_parameters())

    extra_params = param_dict.keys() - (decay | no_decay)
    if len(extra_params) != 0:
        msg = f"parameters {extra_params} were not separated into either decay/no_decay set!"
        raise ValueError(msg)

    # create the pytorch optimizer parameters
    return [
        {
            "params": [param_dict[pn] for pn in decay],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in no_decay],
            "weight_decay": 0.0,
        },
    ]


@torch.inference_mode()
def compute_individual_iou(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    smoothing: float = 1e-8,
) -> tuple[float, float]:
    foreground_gt = ground_truth.flatten(1).bool()
    foreground_pred = prediction.flatten(1) > 0

    foreground_iou = (
        ((foreground_gt & foreground_pred).sum(dim=1) + smoothing)
        / ((foreground_gt | foreground_pred).sum(dim=1) + smoothing)
    ).mean()

    background_gt = ~foreground_gt
    background_pred = ~foreground_pred

    background_iou = (
        ((background_gt & background_pred).sum(dim=1) + smoothing)
        / ((background_gt | background_pred).sum(dim=1) + smoothing)
    ).mean()

    return foreground_iou.item(), background_iou.item()


def train_one_epoch(
    train_dl: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    running_loss = 0.0
    running_foreground_iou = 0.0
    running_background_iou = 0.0

    for batch in train_dl:
        image = batch["image"]
        scribble = batch["scribble"]  # Add channel dimension
        inputs = torch.cat((image, scribble), dim=1).to(
            device
        )  # Concatenate along channel dimension

        ground_truth = batch["ground_truth"].to(dtype=torch.float32, device=device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs.flatten(1), ground_truth.flatten(1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item() * len(inputs)

        fore_iou, back_iou = compute_individual_iou(
            ground_truth=ground_truth,
            prediction=outputs,
        )

        running_foreground_iou += fore_iou * len(inputs)
        running_background_iou += back_iou * len(inputs)

    running_loss /= len(train_dl.dataset)
    running_foreground_iou /= len(train_dl.dataset)
    running_background_iou /= len(train_dl.dataset)

    return {
        "loss": running_loss,
        "foreground_iou": running_foreground_iou,
        "background_iou": running_background_iou,
        "mean_iou": 0.5 * (running_foreground_iou + running_background_iou),
    }


@torch.inference_mode()
def validate_one_epoch(
    val_dl: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    checkpoint_dir: Path,
):
    model.eval()

    running_vloss = 0.0
    running_vforeground_iou = 0.0
    running_vbackground_iou = 0.0

    for vbatch in val_dl:
        vimage = vbatch["image"]
        vscribble = vbatch["scribble"]  # Add channel dimension

        # Concatenate along channel dimension
        vinputs = torch.cat((vimage, vscribble), dim=1).to(device)

        voutputs = model(vinputs)

        vground_truth = vbatch["ground_truth"].to(dtype=torch.float32, device=device)

        vloss = criterion(voutputs.flatten(1), vground_truth.flatten(1))

        running_vloss += vloss.cpu().item() * len(vinputs)

        vfore_iou, vback_iou = compute_individual_iou(
            ground_truth=vground_truth,
            prediction=voutputs,
        )

        running_vforeground_iou += vfore_iou * len(vinputs)
        running_vbackground_iou += vback_iou * len(vinputs)

    running_vloss /= len(val_dl.dataset)
    running_vforeground_iou /= len(val_dl.dataset)
    running_vbackground_iou /= len(val_dl.dataset)

    torch.save(
        model.state_dict(),
        checkpoint_dir / "latest.pt",
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    return {
        "loss": running_vloss,
        "foreground_iou": running_vforeground_iou,
        "background_iou": running_vbackground_iou,
        "mean_iou": 0.5 * (running_vforeground_iou + running_vbackground_iou),
    }


def save_prediction(
    prediction: np.ndarray,
    save_dir: Path,
    img_id: str,
):
    return (
        Image.fromarray(prediction)
        .convert("1", dither=Image.Dither.NONE)
        .resize(FINAL_SIZE, Image.Resampling.NEAREST)
        .save(
            save_dir / f"{img_id}.png",
            optimize=True,
        )
    )


@torch.inference_mode()
def get_test_predictions(
    test_ds: Dataset,
    model: nn.Module,
    device: torch.device,
    predictions_dir: Path,
):
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    future_to_img_id = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for tbatch in test_dl:
            timage = tbatch["image"]
            tscribble = tbatch["scribble"]  # Add channel dimension

            # Concatenate along channel dimension
            tinputs = torch.cat((timage, tscribble), dim=1).to(device)

            toutputs = (
                (torch.sigmoid(model(tinputs)) * 255)
                .squeeze(1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            future_to_img_id.update(
                {
                    executor.submit(
                        save_prediction,
                        prediction=output,
                        save_dir=predictions_dir,
                        img_id=tbatch["id"][i],
                    ): tbatch["id"][i]
                    for i, output in enumerate(toutputs)
                }
            )

        for future in concurrent.futures.as_completed(future_to_img_id):
            img_id = future_to_img_id[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Image {img_id} generated an exception: {exc}")
            else:
                print(f"Image {img_id} saved successfully.")


def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    model: nn.Module,
    checkpoint_dir: Path,
    device: torch.device,
):
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    criterion = DiceCELoss()

    optimizer = torch.optim.AdamW(
        get_optim_groups(model, weight_decay=WEIGHT_DECAY), lr=LEARNING_RATE, fused=True
    )

    best_vloss = float("inf")
    num_no_improvements = 0

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}")

        train_metrics = train_one_epoch(
            train_dl=train_dl,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print("[TRAIN] ")
        for key, value in train_metrics.items():
            print(f"{key}: {value:.4f}", end=", ")
        print()

        val_metrics = validate_one_epoch(
            val_dl=val_dl,
            model=model,
            criterion=criterion,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        print("[VALID] ")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}", end=", ")
        print()

        # Track best performance, and save the model's state
        if val_metrics["loss"] < best_vloss:
            num_no_improvements = 0
            print(
                f"Validation loss improved from {best_vloss:.4f} to {val_metrics['loss']:.4f}, saving model..."
            )
            best_vloss = val_metrics["loss"]
            # Save the model's state dict
            torch.save(
                model.state_dict(),
                checkpoint_dir / "best.pt",
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
        else:
            num_no_improvements += 1
            if num_no_improvements >= MAX_PATIENCE:
                print(
                    f"No improvement for {MAX_PATIENCE} epochs, stopping training at epoch {epoch + 1}."
                )
                break


if __name__ == "__main__":
    SEED: Final = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("medium")

    augmentation_transforms = A.Compose(
        [
            SyntheticScribbleTransform(
                p=0.4,
            ),
            A.HorizontalFlip(
                p=0.5
            ),  # This changes the image resolution, so it should be before cropping
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-30, 30),
                translate_percent=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REPLICATE,
                balanced_scale=True,
                p=0.3,
            ),
            A.PadIfNeeded(
                min_height=IMG_SIZE[1],
                min_width=IMG_SIZE[0],
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.CropNonEmptyMaskIfExists(
                height=IMG_SIZE[1], width=IMG_SIZE[0], ignore_values=[0, -1]
            ),
            A.Erasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), fill="inpaint_telea", p=0.1),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, p=1),
                    A.OneOf(
                        [
                            A.ToGray(method="weighted_average", p=1),
                            A.ToGray(method="from_lab", p=1),
                        ],  # Some images are grayscale in training set
                        p=1,
                    ),
                ],
                p=0.3,
            ),
            A.Posterize(num_bits=4, p=0.1),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1.0,
                        sigma=50.0,
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1,
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=(-0.3, 0.3),
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1,
                    ),
                    A.OpticalDistortion(
                        distort_limit=(-0.3, 0.3),
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1,
                    ),
                    A.Perspective(
                        scale=(0.3, 1),
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1,
                    ),
                ],  # type:ignore
                p=0.1,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=5, sigma_limit=(0.1, 1.0), p=1),
                    A.MotionBlur(p=1),
                    A.MedianBlur(p=1),
                    A.Sharpen(p=1),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.1, 0.2), p=1),
                    A.MultiplicativeNoise(
                        multiplier=(0.9, 1.1),
                        per_channel=True,
                        elementwise=True,
                        p=1,
                    ),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.3),
                        p=1,
                    ),
                    A.ImageCompression(
                        quality_range=(50, 100),
                        compression_type="jpeg",
                        p=1,
                    ),
                ],  # type:ignore
                p=0.1,
            ),
            A.Normalize(
                mean=IMG_MEAN,
                std=IMG_STD,
            ),
            ToTensorV2(transpose_mask=True),
        ],  # type:ignore
        additional_targets={
            "scribble": "mask",
        },
    )

    deterministic_transforms = A.Compose(
        [
            A.Resize(
                height=IMG_SIZE[1],
                width=IMG_SIZE[0],
                interpolation=cv2.INTER_CUBIC,
            ),
            A.Normalize(
                mean=IMG_MEAN,
                std=IMG_STD,
            ),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={
            "scribble": "mask",
        },
    )

    train_ds = ScribbleDataset(
        TRAINING_ROOT,
        IMAGE_DIR,
        SCRIBBLES_DIR,
        GROUND_TRUTH_DIR,
        transforms=augmentation_transforms,
        end_idx=182,
    )

    print(f"Number of training samples: {len(train_ds)}")

    val_ds = ScribbleDataset(
        TRAINING_ROOT,
        IMAGE_DIR,
        SCRIBBLES_DIR,
        GROUND_TRUTH_DIR,
        transforms=deterministic_transforms,
        start_idx=182,
    )

    print(f"Number of validation samples: {len(val_ds)}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = smp.Unet(
        encoder_name="tu-mobilenetv3_small_050",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        decoder_interpolation="bilinear",  # use bilinear interpolation in decoder
        drop_rate=DROPOUT_RATE,
    ).to(device)

    print("Using model:", model.__class__.__name__)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_params:,}")

    checkpoint_dir = (
        CHECKPOINT_ROOT / model.__class__.__name__ / time.strftime("%Y%m%d-%H%M%S")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to {checkpoint_dir}")

    train_model(
        train_ds,
        val_ds,
        model,
        checkpoint_dir,
        device,
    )

    state_dict = torch.load(
        checkpoint_dir / "best.pt",
        map_location=device,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )
    model.load_state_dict(state_dict)

    train_pred_ds = ScribbleDataset(
        TRAINING_ROOT,
        IMAGE_DIR,
        SCRIBBLES_DIR,
        transforms=deterministic_transforms,
    )

    predictions_root = checkpoint_dir / PREDICTION_DIR

    get_test_predictions(
        train_pred_ds,
        model,
        device,
        predictions_root / "train",
    )
    print(f"Predictions saved to {predictions_root}")

    test_pred_ds = ScribbleDataset(
        TEST_ROOT,
        IMAGE_DIR,
        SCRIBBLES_DIR,
        transforms=deterministic_transforms,
    )

    get_test_predictions(
        test_pred_ds,
        model,
        device,
        predictions_root / "test",
    )
    print(f"Predictions saved to {predictions_root}")
