# assumes script is in the same dir as features/filtered images

from __future__ import annotations
import copy
import random
import re
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

DEFAULT_CLASS_NAMES = ["blebbing", "disruption", "predeath"]
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def has_excluded_tif_tif_name(path: Path) -> bool:
    """Return True if the filename contains 'tif.tif' (case-insensitive)."""
    return bool(re.search(r"tif\.tif", path.name, flags=re.IGNORECASE))


def extract_cell_group_id_from_filename(path: Path) -> str:
    """Extract `<video>_cell<num>` so frames from the same cell stay grouped."""
    stem_clean = re.sub(r"(?:\.(?:tif|tiff|png))+$", "", path.stem, flags=re.IGNORECASE)
    match = re.match(r"^(?P<cell_id>.+?_cell\d+)", stem_clean, flags=re.IGNORECASE)
    if match:
        return match.group("cell_id").lower()
    return stem_clean.lower()


def extract_class_name_from_filename(
    path: Path,
    allowed_classes: list[str] | None = None,
) -> str:
    """Extract target class name from filename."""
    classes = list(allowed_classes) if allowed_classes is not None else list(DEFAULT_CLASS_NAMES)
    stem_lower = path.stem.lower()
    for class_name in classes:
        if re.search(rf"(?:^|[_\-\s]){re.escape(class_name.lower())}(?:[_\-\s]|$)", stem_lower) or (
            class_name.lower() in stem_lower
        ):
            return class_name
    raise ValueError(f"Could not match any class in {classes} from filename '{path.name}'")


def build_transforms(train: bool = False, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Use mild, microscopy-safe augmentations only during training."""
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FilenameStageDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        class_names: list[str] | None = None,
        transform: transforms.Compose | None = None,
        exclude_tif_tif: bool = True,
    ) -> None:
        self.folder = Path(folder)
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform or build_transforms(train=False)

        all_pngs = sorted(self.folder.glob("*.png"))
        if not all_pngs:
            raise FileNotFoundError(f"No .png files found in {self.folder}")

        self.skipped_paths: list[Path] = []
        self.image_paths: list[Path] = []
        for p in all_pngs:
            if exclude_tif_tif and has_excluded_tif_tif_name(p):
                self.skipped_paths.append(p)
            else:
                self.image_paths.append(p)

        if not self.image_paths:
            raise RuntimeError(f"All .png files in {self.folder} were excluded by filter.")

        self.labels: list[int] = []
        self.group_ids: list[str] = []
        for path in self.image_paths:
            class_name = extract_class_name_from_filename(path, allowed_classes=self.class_names)
            self.labels.append(self.class_to_idx[class_name])
            self.group_ids.append(extract_cell_group_id_from_filename(path))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[index]
        label = self.labels[index]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)
        return tensor, label


class StageDeathCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layer4: bool = False,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            net = models.resnet18(weights=weights)
        except Exception:
            net = models.resnet18(weights=None)

        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = net
        self.configure_trainable_params(freeze_backbone=freeze_backbone, unfreeze_layer4=unfreeze_layer4)

    def configure_trainable_params(self, freeze_backbone: bool = True, unfreeze_layer4: bool = False) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        if freeze_backbone and unfreeze_layer4:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        sup = int(cm[c, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[class_names[c]] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    acc = float((y_t == y_p).mean()) if len(y_t) > 0 else 0.0
    total_sup = max(sum(supports), 1)
    weights = np.array(supports, dtype=np.float64) / total_sup

    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * weights)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def make_leave_one_group_out_folds(
    group_ids: list[str],
    n_splits: int | None = None,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Create Leave-One-Group-Out (LOGO) folds when `n_splits is None` or `n_splits >= num_groups`,
    or group folds into `n_splits` buckets if a smaller fold count is requested.

    Returns list of (val_group_label, train_indices, val_indices).
    """
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 cell groups for cross-validation, found {len(unique_groups)}.")

    group_to_indices: dict[str, np.ndarray] = {
        g: np.array([i for i, gid in enumerate(group_ids) if gid == g], dtype=np.int64)
        for g in unique_groups
    }

    if n_splits is None or n_splits >= len(unique_groups):
        # True Leave-One-Group-Out: one cell group held out as validation per fold
        folds: list[tuple[str, np.ndarray, np.ndarray]] = []
        for val_g in unique_groups:
            val_idx = group_to_indices[val_g]
            train_idx = np.concatenate([group_to_indices[g] for g in unique_groups if g != val_g])
            folds.append((val_g, train_idx, val_idx))
        return folds

    # K-fold group split across unique cell groups
    rng = np.random.default_rng(seed)
    shuffled_groups = np.array(unique_groups)
    rng.shuffle(shuffled_groups)

    fold_buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for idx, g in enumerate(shuffled_groups.tolist()):
        fold_buckets[idx % n_splits].append(g)

    folds = []
    for f_idx, val_groups in enumerate(fold_buckets):
        val_set = set(val_groups)
        train_groups = [g for g in unique_groups if g not in val_set]
        if not val_groups or not train_groups:
            continue
        val_idx = np.concatenate([group_to_indices[g] for g in val_groups])
        train_idx = np.concatenate([group_to_indices[g] for g in train_groups])
        folds.append((f"fold_{f_idx+1}({','.join(val_groups)})", train_idx, val_idx))

    return folds


def compute_class_weights(train_labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, dict[str, Any], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total += batch_size

        preds = logits.argmax(dim=1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_true, all_pred, class_names)
    return total_loss / max(total, 1), metrics, all_true, all_pred


def build_optimizer(model: StageDeathCNN, head_lr: float, backbone_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = [p for p in model.backbone.fc.parameters() if p.requires_grad]
    backbone_params = [
        p for name, p in model.backbone.named_parameters()
        if p.requires_grad and not name.startswith("fc.")
    ]
    param_groups = [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def run_experiment(
    data_dir: Path,
    trials: list[dict[str, Any]],
    n_splits: int | None = None,  # None => Leave-One-Group-Out across all cell groups
    epochs: int = 25,
    batch_size: int = 8,
    seed: int = 42,
    save_dir: Path | None = None,
) -> None:
    set_seed(seed)
    device = get_device()
    save_dir = save_dir or Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    base_ds = FilenameStageDataset(
        data_dir,
        class_names=DEFAULT_CLASS_NAMES,
        transform=build_transforms(train=False),
        exclude_tif_tif=True,
    )
    train_aug_ds = FilenameStageDataset(
        data_dir,
        class_names=DEFAULT_CLASS_NAMES,
        transform=build_transforms(train=True),
        exclude_tif_tif=True,
    )

    folds = make_leave_one_group_out_folds(group_ids=base_ds.group_ids, n_splits=n_splits, seed=seed)
    mode_label = "Leave-One-Group-Out (LOGO)" if (n_splits is None or n_splits >= len(set(base_ds.group_ids))) else f"{len(folds)}-Fold Group CV"

    print(
        f"Device: {device} | Usable samples: {len(base_ds)} | "
        f"Skipped tif.tif: {len(base_ds.skipped_paths)} | Unique cell groups: {len(set(base_ds.group_ids))} | "
        f"CV Mode: {mode_label} ({len(folds)} folds)"
    )

    best_score = -1.0
    best_ckpt_path: Path | None = None

    for t_idx, cfg in enumerate(trials, start=1):
        set_seed(seed + t_idx)
        freeze_backbone = cfg.get("freeze_backbone", True)
        unfreeze_layer4 = cfg.get("unfreeze_layer4", False)
        head_lr = cfg.get("head_lr", 1e-3)
        backbone_lr = cfg.get("backbone_lr", 1e-4)
        weight_decay = cfg.get("weight_decay", 1e-2)
        dropout_p = cfg.get("dropout_p", 0.5)

        print(
            f"\n=== Trial {t_idx}: freeze_backbone={freeze_backbone}, "
            f"unfreeze_layer4={unfreeze_layer4}, head_lr={head_lr}, weight_decay={weight_decay} ==="
        )

        oof_true: list[int] = []
        oof_pred: list[int] = []
        fold_accs: list[float] = []

        for f_idx, (group_label, tr_idx, va_idx) in enumerate(folds, start=1):
            tr_labels = [base_ds.labels[int(i)] for i in tr_idx]
            criterion = nn.CrossEntropyLoss(weight=compute_class_weights(tr_labels, len(DEFAULT_CLASS_NAMES), device))

            train_loader = DataLoader(
                Subset(train_aug_ds, tr_idx.tolist()),
                batch_size=min(batch_size, max(1, len(tr_idx))),
                shuffle=True,
                drop_last=len(tr_idx) > batch_size,
            )
            val_loader = DataLoader(Subset(base_ds, va_idx.tolist()), batch_size=batch_size, shuffle=False)

            model = StageDeathCNN(
                num_classes=len(DEFAULT_CLASS_NAMES),
                pretrained=True,
                freeze_backbone=freeze_backbone,
                unfreeze_layer4=unfreeze_layer4,
                dropout_p=dropout_p,
            ).to(device)
            optimizer = build_optimizer(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=weight_decay)

            best_val_acc = -1.0
            best_fold_weights = copy.deepcopy(model.state_dict())
            best_fold_preds: list[int] = []
            best_fold_trues: list[int] = []

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                va_loss, va_metrics, v_true, v_pred = evaluate(model, val_loader, criterion, device, DEFAULT_CLASS_NAMES)
                # Using val accuracy/loss for single-group selection avoids single-group F1 step jumps
                if va_metrics["accuracy"] > best_val_acc or (
                    va_metrics["accuracy"] == best_val_acc and epoch == 1
                ):
                    best_val_acc = va_metrics["accuracy"]
                    best_fold_weights = copy.deepcopy(model.state_dict())
                    best_fold_preds = v_pred
                    best_fold_trues = v_true

            fold_accs.append(best_val_acc)
            oof_true.extend(best_fold_trues)
            oof_pred.extend(best_fold_preds)
            print(
                f"  Fold {f_idx:02d}/{len(folds):02d} [{group_label}] "
                f"(train={len(tr_idx)}, val={len(va_idx)}) -> Best Val Acc: {best_val_acc*100:5.1f}% "
                f"(preds={best_fold_preds}, true={best_fold_trues})"
            )

        oof_metrics = compute_metrics(oof_true, oof_pred, DEFAULT_CLASS_NAMES)
        print(
            f"Trial {t_idx} Overall LOGO -> Acc: {oof_metrics['accuracy']*100:.2f}% | "
            f"Macro F1: {oof_metrics['macro_f1']:.4f} | Mean Fold Acc: {np.mean(fold_accs)*100:.2f}%"
        )
        print("  Per-class OOF metrics:", oof_metrics["per_class"])
        print("  OOF Confusion Matrix (rows=true, cols=pred):\n", oof_metrics["confusion_matrix"])

        # Fit final model across all usable samples for deployment/inference
        all_indices = list(range(len(base_ds)))
        full_labels = base_ds.labels
        full_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(full_labels, len(DEFAULT_CLASS_NAMES), device))
        full_loader = DataLoader(
            Subset(train_aug_ds, all_indices),
            batch_size=min(batch_size, max(1, len(all_indices))),
            shuffle=True,
            drop_last=len(all_indices) > batch_size,
        )

        final_model = StageDeathCNN(
            num_classes=len(DEFAULT_CLASS_NAMES),
            pretrained=True,
            freeze_backbone=freeze_backbone,
            unfreeze_layer4=unfreeze_layer4,
            dropout_p=dropout_p,
        ).to(device)
        final_optimizer = build_optimizer(final_model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=weight_decay)

        for _ in range(epochs):
            train_one_epoch(final_model, full_loader, full_criterion, final_optimizer, device)

        ckpt_path = save_dir / f"stage_death_cnn_logo_trial{t_idx}.pt"
        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "config": cfg,
                "oof_macro_f1": oof_metrics["macro_f1"],
                "oof_accuracy": oof_metrics["accuracy"],
            },
            ckpt_path,
        )

        score = float(oof_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_ckpt_path = ckpt_path

    print(f"\nBest LOGO checkpoint: {best_ckpt_path} (OOF Macro F1: {best_score:.4f})")


if __name__ == "__main__":
    trials_config = [
        {"freeze_backbone": True, "unfreeze_layer4": False, "head_lr": 1e-3, "weight_decay": 1e-2, "dropout_p": 0.5},
        {"freeze_backbone": True, "unfreeze_layer4": True, "head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 2e-2, "dropout_p": 0.5},
        {"freeze_backbone": False, "unfreeze_layer4": True, "head_lr": 5e-4, "backbone_lr": 5e-5, "weight_decay": 5e-2, "dropout_p": 0.6},
        {"freeze_backbone": False, "unfreeze_layer4": True, "head_lr": 5e-4, "backbone_lr": 5e-5, "weight_decay": 1e-2, "dropout_p": 0.6},
    ]
    run_experiment(
        data_dir=Path("{insert path here}"),
        trials=trials_config,
        n_splits=None,  # Set e.g. n_splits=5 for 5-fold group CV instead of full LOGO
        epochs=25,
        batch_size=8,
        seed=42,
        save_dir=Path("{insert path here}"),
    )
