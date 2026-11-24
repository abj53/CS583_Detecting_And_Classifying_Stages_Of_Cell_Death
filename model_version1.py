from pathlib import Path
import random
import re
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from remotezip import RemoteZip
import requests
from scipy import ndimage as ndi
from skimage import color, feature, filters, io, util
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms

DEFAULT_CLASS_NAMES = ["blebbing", "disruption", "predeath"]
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_PATH = Path(__file__).resolve().parent / "stage_death_cnn.pt"

def extract_cell_group_id_from_filename(path: Path) -> str:
    """Extract `<video>_cell<num>` when present so frames from the same cell stay grouped."""
    stem_clean = re.sub(r"(?:\.(?:tif|tiff|png))+$", "", path.stem, flags=re.IGNORECASE)
    match = re.match(r"^(?P<cell_id>.+?_cell\d+)", stem_clean, flags=re.IGNORECASE)
    if match:
        return match.group("cell_id").lower()
    return stem_clean.lower()

def extract_class_name_from_filename(
    path: Path,
    allowed_classes: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Extract the target cell-death stage class name from a filename."""
    classes = list(allowed_classes) if allowed_classes is not None else list(DEFAULT_CLASS_NAMES)
    stem_lower = path.stem.lower()
    for class_name in classes:
        if re.search(rf"(?:^|[_\-\s]){re.escape(class_name.lower())}(?:[_\-\s]|$)", stem_lower) or (
            class_name.lower() in stem_lower
        ):
            if verbose:
                print(f"Identified {class_name} from {path.name}")
            return class_name
    raise ValueError(f"Could not match any class in {classes} from filename '{path.name}'")

class FilenameStageDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        class_names: list[str] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.folder = folder
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform or transforms.Compose([
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.image_paths = sorted(folder.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .png files found in {folder}")

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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StageDeathCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        class_names: list[str] | None = None,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        num_classes = len(self.class_names)
        self.backbone_name = backbone
        self.use_pretrained_backbone = backbone == "resnet18"

        if self.use_pretrained_backbone:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                net = models.resnet18(weights=weights)
            except Exception:
                net = models.resnet18(weights=None)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Dropout(p=0.35),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = net
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 32, dropout=0.0),
                ConvBlock(32, 64, dropout=0.05),
                ConvBlock(64, 128, dropout=0.10),
                ConvBlock(128, 256, dropout=0.15),
            )
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_backbone:
            return self.backbone(x)
        feat = self.features(x)
        pooled = torch.cat([self.avg_pool(feat), self.max_pool(feat)], dim=1)
        return self.classifier(pooled)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self(x)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_multipliers: list[float] | np.ndarray | None = None,
    ) -> list[dict[str, float | int | str]]:
        probs = self.predict_probs(x)
        if class_multipliers is not None:
            mult = torch.tensor(class_multipliers, dtype=probs.dtype, device=probs.device).unsqueeze(0)
            adjusted = probs * mult
            adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            adjusted = probs

        confidences, indices = torch.max(adjusted, dim=1)
        results = []
        for idx, conf in zip(indices.tolist(), confidences.tolist()):
            results.append({
                "class_name": self.class_names[idx],
                "class_index": idx,
                "confidence": round(conf, 4),
            })
        return results


def compute_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    """Compute per-class, macro, and weighted precision, recall, F1, plus confusion matrix."""
    names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
    num_classes = len(names)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    cm = compute_confusion_matrix(y_t, y_p, num_classes=num_classes)

    per_class: dict[str, dict[str, float | int]] = {}
    precisions = np.zeros(num_classes, dtype=np.float64)
    recalls = np.zeros(num_classes, dtype=np.float64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    supports = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))
        supports[c] = support

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions[c] = precision
        recalls[c] = recall
        f1s[c] = f1

        per_class[names[c]] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        }

    total_support = int(np.sum(supports))
    weights = supports / total_support if total_support > 0 else np.zeros(num_classes, dtype=np.float64)

    return {
        "accuracy": round(float(np.mean(y_t == y_p)), 4) if len(y_t) > 0 else 0.0,
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "weighted_precision": round(float(np.sum(precisions * weights)), 4),
        "weighted_recall": round(float(np.sum(recalls * weights)), 4),
        "weighted_f1": round(float(np.sum(f1s * weights)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": names,
    }


def print_metrics_report(metrics: dict[str, object], title: str = "Classification Metrics") -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy         : {metrics['accuracy']:.2%}")
    print(
        f"Macro Avg        : Precision={metrics['macro_precision']:.4f} | "
        f"Recall={metrics['macro_recall']:.4f} | F1={metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted Avg     : Precision={metrics['weighted_precision']:.4f} | "
        f"Recall={metrics['weighted_recall']:.4f} | F1={metrics['weighted_f1']:.4f}"
    )
    per_class = metrics["per_class"]  # type: ignore[assignment]
    for cls_name, stats in per_class.items():
        print(
            f"  - {cls_name:<12} | Precision={stats['precision']:.4f} | "
            f"Recall={stats['recall']:.4f} | F1={stats['f1']:.4f} | "
            f"Support={stats['support']}"
        )
    if "confusion_matrix" in metrics:
        cm = np.asarray(metrics["confusion_matrix"])
        names = metrics.get("class_names", DEFAULT_CLASS_NAMES)  # type: ignore[assignment]
        print("  Confusion Matrix (rows=true, cols=pred):")
        print("   ", " ".join(f"{n[:8]:>8}" for n in names))
        for r_idx, row in enumerate(cm):
            print(f"  {names[r_idx][:8]:<8}", " ".join(f"{val:8d}" for val in row))


def tune_class_multipliers(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    class_names: list[str],
    grid_values: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25, 1.4),
) -> list[float]:
    """Search class probability multipliers on out-of-fold predictions to maximize macro F1."""
    num_classes = len(class_names)
    if len(oof_targets) == 0 or num_classes == 0:
        return [1.0] * max(1, num_classes)

    best_f1 = -1.0
    best_mult = [1.0] * num_classes
    current = [1.0] * num_classes
    for _ in range(2):
        for c in range(num_classes):
            for val in grid_values:
                candidate = list(current)
                candidate[c] = val
                mult_arr = np.asarray(candidate, dtype=np.float64)[None, :]
                preds = np.argmax(oof_probs * mult_arr, axis=1)
                m = compute_classification_metrics(oof_targets, preds, class_names=class_names)
                score = float(m["macro_f1"])  # type: ignore[arg-type]
                if score > best_f1:
                    best_f1 = score
                    best_mult = candidate
        current = list(best_mult)

    return [round(float(x), 3) for x in best_mult]


def _compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    total = float(np.sum(counts))
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = float(total / (num_classes * counts[c]))
    weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_weighted_sampler(subset_labels: list[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(subset_labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    per_class_weight = {c: (1.0 / counts[c]) if counts[c] > 0 else 0.0 for c in range(num_classes)}
    sample_weights = torch.tensor([per_class_weight[lbl] for lbl in subset_labels], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(subset_labels), replacement=True)


def _group_stratified_kfold_indices(
    labels: list[int],
    group_ids: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Group-aware stratified K-fold split keeping each cell/group intact across train and val."""
    total = len(labels)
    if total == 0:
        return []

    rng = np.random.default_rng(seed)
    unique_groups = sorted(set(group_ids)) if group_ids is not None else []
    effective_splits = min(n_splits, len(unique_groups)) if len(unique_groups) >= 2 else n_splits

    if len(unique_groups) >= 2 and effective_splits >= 2:
        group_to_indices: dict[str, list[int]] = {}
        group_primary_label: dict[str, int] = {}
        for idx, lbl in enumerate(labels):
            g = group_ids[idx] if group_ids is not None else str(idx)
            group_to_indices.setdefault(g, []).append(idx)
        for g, idxs in group_to_indices.items():
            g_lbls = [labels[i] for i in idxs]
            group_primary_label[g] = int(np.bincount(g_lbls).argmax())

        label_to_groups: dict[int, list[str]] = {}
        for g, p_lbl in group_primary_label.items():
            label_to_groups.setdefault(p_lbl, []).append(g)

        fold_val_indices: list[list[int]] = [[] for _ in range(effective_splits)]
        for _, grp_list in label_to_groups.items():
            shuffled_grps = list(grp_list)
            rng.shuffle(shuffled_grps)
            for pos, g in enumerate(shuffled_grps):
                fold_val_indices[pos % effective_splits].extend(group_to_indices[g])

        all_set = set(range(total))
        folds: list[tuple[list[int], list[int]]] = []
        for val_idx in fold_val_indices:
            if not val_idx:
                continue
            val_set = set(val_idx)
            train_idx = sorted(all_set - val_set)
            if train_idx:
                folds.append((train_idx, sorted(val_idx)))
        if folds:
            return folds

    generator = torch.Generator().manual_seed(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)
    if not class_to_indices or min(len(v) for v in class_to_indices.values()) < effective_splits:
        perm = torch.randperm(total, generator=generator).tolist()
        val_size = max(1, len(perm) // effective_splits)
        folds = []
        for start in range(0, len(perm), val_size):
            val_idx = perm[start : start + val_size]
            if not val_idx:
                continue
            val_set = set(val_idx)
            train_idx = [i for i in perm if i not in val_set]
            if train_idx:
                folds.append((train_idx, val_idx))
        return folds if folds else [(perm, perm)]

    fold_val_buckets: list[list[int]] = [[] for _ in range(effective_splits)]
    for indices in class_to_indices.values():
        t_idx = torch.tensor(indices)
        shuffled = t_idx[torch.randperm(len(indices), generator=generator)].tolist()
        for pos, sample_idx in enumerate(shuffled):
            fold_val_buckets[pos % effective_splits].append(sample_idx)

    all_set = set(range(total))
    folds = []
    for val_idx in fold_val_buckets:
        val_set = set(val_idx)
        train_idx = sorted(all_set - val_set)
        folds.append((train_idx, sorted(val_idx)))
    return folds


def _select_stratified_indices(
    labels: list[int],
    group_ids: list[str] | None = None,
    fraction: float = 0.85,
    seed: int = 42,
) -> list[int]:
    total = len(labels)
    target_count = max(1, min(total, int(round(total * fraction))))
    if target_count >= total:
        return list(range(total))

    rng = np.random.default_rng(seed)
    unique_groups = sorted(set(group_ids)) if group_ids is not None else []
    if len(unique_groups) >= 4:
        group_to_indices: dict[str, list[int]] = {}
        for idx, g in enumerate(group_ids):  # type: ignore[union-attr]
            group_to_indices.setdefault(g, []).append(idx)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)
        selected: list[int] = []
        for g in shuffled_groups:
            if len(selected) + len(group_to_indices[g]) <= target_count:
                selected.extend(group_to_indices[g])
        if selected:
            return sorted(selected)

    generator = torch.Generator().manual_seed(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)
    if len(class_to_indices) > 1 and min(len(v) for v in class_to_indices.values()) >= 2:
        selected = []
        for indices in class_to_indices.values():
            t_idx = torch.tensor(indices)
            shuffled = t_idx[torch.randperm(len(indices), generator=generator)].tolist()
            take = max(1, min(len(shuffled), round(len(shuffled) * fraction)))
            selected.extend(shuffled[:take])
        if len(selected) > target_count:
            selected = selected[:target_count]
        elif len(selected) < target_count:
            remaining = [i for i in range(total) if i not in set(selected)]
            perm_rem = torch.tensor(remaining)[torch.randperm(len(remaining), generator=generator)].tolist()
            selected.extend(perm_rem[: target_count - len(selected)])
        return sorted(selected)

    perm = torch.randperm(total, generator=generator).tolist()
    return sorted(perm[:target_count])


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str] | None = None,
    class_multipliers: list[float] | None = None,
) -> tuple[float, float, dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    mult_tensor = (
        torch.tensor(class_multipliers, dtype=torch.float32, device=device).unsqueeze(0)
        if class_multipliers is not None
        else None
    )

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

        scored = probs * mult_tensor if mult_tensor is not None else probs
        preds = scored.argmax(dim=1)
        total += labels.size(0)
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total if total > 0 else 0.0
    targets_arr = np.asarray(all_targets, dtype=np.int64)
    probs_arr = np.vstack(all_probs) if all_probs else np.zeros((0, len(class_names or DEFAULT_CLASS_NAMES)))
    metrics = compute_classification_metrics(targets_arr, all_preds, class_names=class_names)
    accuracy = float(metrics["accuracy"])  # type: ignore[arg-type]
    return avg_loss, accuracy, metrics, targets_arr, probs_arr


def _build_optimizer(model: StageDeathCNN, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if model.use_pretrained_backbone:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".fc." in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def model_train(
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 2e-4,
    data_dir: Path | None = None,
    checkpoint_path: Path = CHECKPOINT_PATH,
    train_fraction: float = 0.85,
    n_splits: int = 5,
    seed: int = 42,
    backbone: str = "resnet18",
    pretrained: bool = True,
    use_weighted_sampler: bool = True,
    fit_final_on_dev: bool = True,
) -> StageDeathCNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_dir = data_dir if data_dir is not None else Path(__file__).resolve().parent / "data" / "features" / "filtered"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Filtered directory not found at {frames_dir}")

    train_transform = transforms.Compose([
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_full_dataset = FilenameStageDataset(
        folder=frames_dir,
        class_names=DEFAULT_CLASS_NAMES,
        transform=train_transform,
    )
    eval_full_dataset = FilenameStageDataset(
        folder=frames_dir,
        class_names=DEFAULT_CLASS_NAMES,
        transform=eval_transform,
    )

    all_labels = list(train_full_dataset.labels)
    all_groups = list(train_full_dataset.group_ids)
    dev_indices = _select_stratified_indices(all_labels, group_ids=all_groups, fraction=train_fraction, seed=seed)
    dev_labels = [all_labels[i] for i in dev_indices]
    dev_groups = [all_groups[i] for i in dev_indices]

    print(
        f"Total samples in {frames_dir}: {len(all_labels)} | "
        f"Dev subset ({train_fraction:.0%}): {len(dev_indices)} samples | "
        f"Held-out samples: {len(all_labels) - len(dev_indices)}"
    )

    folds = _group_stratified_kfold_indices(dev_labels, group_ids=dev_groups, n_splits=n_splits, seed=seed)
    print(f"Running {len(folds)}-fold cell-grouped stratified cross-validation on {len(dev_indices)} samples...")

    num_classes = len(DEFAULT_CLASS_NAMES)
    effective_batch_size = max(1, min(batch_size, len(dev_indices)))

    oof_targets_list: list[np.ndarray] = []
    oof_probs_list: list[np.ndarray] = []
    fold_best_f1s: list[float] = []
    fold_best_accs: list[float] = []

    for fold_idx, (rel_train_idx, rel_val_idx) in enumerate(folds, start=1):
        fold_train_indices = [dev_indices[i] for i in rel_train_idx]
        fold_val_indices = [dev_indices[i] for i in rel_val_idx]
        fold_train_labels = [all_labels[i] for i in fold_train_indices]

        train_subset = Subset(train_full_dataset, fold_train_indices)
        val_subset = Subset(eval_full_dataset, fold_val_indices)

        if use_weighted_sampler and len(set(fold_train_labels)) > 1:
            sampler = _build_weighted_sampler(fold_train_labels, num_classes=num_classes)
            train_loader = DataLoader(
                train_subset,
                batch_size=min(effective_batch_size, len(train_subset)),
                sampler=sampler,
                num_workers=0,
            )
        else:
            train_loader = DataLoader(
                train_subset,
                batch_size=min(effective_batch_size, len(train_subset)),
                shuffle=True,
                num_workers=0,
            )

        val_loader = DataLoader(
            val_subset,
            batch_size=min(effective_batch_size, len(val_subset)),
            shuffle=False,
            num_workers=0,
        )

        model = StageDeathCNN(
            in_channels=3,
            class_names=DEFAULT_CLASS_NAMES,
            backbone=backbone,
            pretrained=pretrained,
        ).to(device)
        class_weights = _compute_class_weights(fold_train_labels, num_classes=num_classes, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = _build_optimizer(model, lr=lr, weight_decay=3e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)

        print(
            f"\n--- Fold {fold_idx}/{len(folds)} "
            f"(train={len(train_subset)}, val={len(val_subset)}) ---"
        )

        best_fold_f1 = -1.0
        best_fold_acc = -1.0
        best_fold_probs = np.zeros((len(val_subset), num_classes), dtype=np.float64)
        best_fold_targets = np.asarray([all_labels[i] for i in fold_val_indices], dtype=np.int64)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            train_loss = running_loss / total if total > 0 else 0.0
            train_acc = correct / total if total > 0 else 0.0
            val_loss, val_acc, val_metrics, val_t_arr, val_p_arr = _evaluate(
                model, val_loader, criterion, device, class_names=DEFAULT_CLASS_NAMES
            )
            val_f1 = float(val_metrics.get("macro_f1", 0.0))  # type: ignore[arg-type]
            if (val_f1 > best_fold_f1) or (val_f1 == best_fold_f1 and val_acc > best_fold_acc):
                best_fold_f1 = val_f1
                best_fold_acc = val_acc
                best_fold_probs = val_p_arr
                best_fold_targets = val_t_arr

            if epoch == epochs or epoch % max(1, epochs // 5) == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} "
                    f"F1(macro): {val_f1:.4f}"
                )

        fold_best_f1s.append(best_fold_f1)
        fold_best_accs.append(best_fold_acc)
        oof_targets_list.append(best_fold_targets)
        oof_probs_list.append(best_fold_probs)

    oof_targets = np.concatenate(oof_targets_list) if oof_targets_list else np.zeros((0,), dtype=np.int64)
    oof_probs = np.vstack(oof_probs_list) if oof_probs_list else np.zeros((0, num_classes), dtype=np.float64)

    raw_oof_preds = np.argmax(oof_probs, axis=1) if len(oof_targets) > 0 else np.zeros((0,), dtype=np.int64)
    oof_metrics = compute_classification_metrics(oof_targets, raw_oof_preds, class_names=DEFAULT_CLASS_NAMES)
    print_metrics_report(oof_metrics, title="Out-of-Fold Validation Metrics (All Folds)")

    tuned_multipliers = tune_class_multipliers(oof_probs, oof_targets, class_names=DEFAULT_CLASS_NAMES)
    tuned_preds = np.argmax(oof_probs * np.asarray(tuned_multipliers)[None, :], axis=1)
    tuned_oof_metrics = compute_classification_metrics(oof_targets, tuned_preds, class_names=DEFAULT_CLASS_NAMES)
    print(f"\nTuned OOF Class Multipliers: {dict(zip(DEFAULT_CLASS_NAMES, tuned_multipliers))}")
    print_metrics_report(tuned_oof_metrics, title="Out-of-Fold Metrics After Multiplier Tuning")

    mean_cv_f1 = float(np.mean(fold_best_f1s)) if fold_best_f1s else 0.0
    mean_cv_acc = float(np.mean(fold_best_accs)) if fold_best_accs else 0.0
    tuned_oof_f1 = float(tuned_oof_metrics.get("macro_f1", 0.0))  # type: ignore[arg-type]
    final_epochs = max(15, epochs)

    if fit_final_on_dev:
        print(f"\nFitting final model on full dev set ({len(dev_indices)} samples) for {final_epochs} epochs...")
        dev_train_subset = Subset(train_full_dataset, dev_indices)
        if use_weighted_sampler and len(set(dev_labels)) > 1:
            dev_sampler = _build_weighted_sampler(dev_labels, num_classes=num_classes)
            dev_loader = DataLoader(
                dev_train_subset,
                batch_size=effective_batch_size,
                sampler=dev_sampler,
                num_workers=0,
            )
        else:
            dev_loader = DataLoader(
                dev_train_subset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=0,
            )

        final_model = StageDeathCNN(
            in_channels=3,
            class_names=DEFAULT_CLASS_NAMES,
            backbone=backbone,
            pretrained=pretrained,
        ).to(device)
        dev_weights = _compute_class_weights(dev_labels, num_classes=num_classes, device=device)
        final_criterion = nn.CrossEntropyLoss(weight=dev_weights, label_smoothing=0.05)
        final_optimizer = _build_optimizer(final_model, lr=lr, weight_decay=3e-2)
        final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            final_optimizer, T_max=final_epochs, eta_min=lr * 0.05
        )

        for _ in range(1, final_epochs + 1):
            final_model.train()
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                final_optimizer.zero_grad()
                logits = final_model(images)
                loss = final_criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=2.0)
                final_optimizer.step()
            final_scheduler.step()
    else:
        final_model = model

    saved_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": saved_state,
            "class_names": DEFAULT_CLASS_NAMES,
            "backbone": backbone,
            "best_val_acc": round(mean_cv_acc, 4),
            "best_val_f1": round(max(mean_cv_f1, tuned_oof_f1), 4),
            "mean_cv_acc": round(mean_cv_acc, 4),
            "mean_cv_f1": round(mean_cv_f1, 4),
            "tuned_oof_f1": round(tuned_oof_f1, 4),
            "class_multipliers": tuned_multipliers,
            "selected_indices": dev_indices,
        },
        checkpoint_path,
    )
    print(
        f"\nSaved checkpoint (Mean CV Acc: {mean_cv_acc:.2%}, Mean CV Macro F1: {mean_cv_f1:.4f}, "
        f"Tuned OOF Macro F1: {tuned_oof_f1:.4f}) to {checkpoint_path}"
    )
    return final_model


def model_predict(
    frames_dir: Path | None = None,
    checkpoint_path: Path = CHECKPOINT_PATH,
    split: str = "all",  # "heldout", "train_dev", or "all"
    seed: int = 42,
    use_tuned_multipliers: bool = True,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_folder = frames_dir if frames_dir is not None else Path(__file__).resolve().parent / "data" / "features" / "filtered"
    all_png_paths = sorted(target_folder.glob("*.png"))
    if not all_png_paths:
        raise FileNotFoundError(f"No .png files found in {target_folder}")

    class_names = list(DEFAULT_CLASS_NAMES)
    backbone = "resnet18"
    state_dict = None
    selected_indices: list[int] | None = None
    class_multipliers: list[float] | None = None

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", list(DEFAULT_CLASS_NAMES))
            backbone = checkpoint.get("backbone", "resnet18")
            selected_indices = checkpoint.get("selected_indices")
            if use_tuned_multipliers:
                class_multipliers = checkpoint.get("class_multipliers")
        else:
            state_dict = checkpoint

    all_labels = [
        class_names.index(extract_class_name_from_filename(p, allowed_classes=class_names))
        for p in all_png_paths
    ]
    all_groups = [extract_cell_group_id_from_filename(p) for p in all_png_paths]
    if selected_indices is None:
        selected_indices = _select_stratified_indices(all_labels, group_ids=all_groups, fraction=0.85, seed=seed)

    selected_set = set(selected_indices)
    if split == "heldout":
        target_indices = [i for i in range(len(all_png_paths)) if i not in selected_set]
    elif split in {"train_dev", "train_half"}:
        target_indices = [i for i in range(len(all_png_paths)) if i in selected_set]
    elif split == "all":
        target_indices = list(range(len(all_png_paths)))
    else:
        raise ValueError(f"Invalid split '{split}'. Expected 'heldout', 'train_dev', or 'all'.")

    target_paths = [all_png_paths[i] for i in target_indices]
    if not target_paths:
        raise ValueError(f"No images found for split='{split}' in {target_folder}")

    model = StageDeathCNN(
        in_channels=3,
        class_names=class_names,
        backbone=backbone,
        pretrained=False,
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    preprocess = transforms.Compose([
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensors = []
    for path in target_paths:
        with Image.open(path) as img:
            tensors.append(preprocess(img.convert("RGB")))
    batch = torch.stack(tensors, dim=0).to(device)

    print(f"Split '{split}' batch shape:", batch.shape)
    predictions = model.predict(batch, class_multipliers=class_multipliers)

    y_true: list[int] = []
    y_pred: list[int] = []
    for path, pred in zip(target_paths, predictions):
        actual_class = extract_class_name_from_filename(path, allowed_classes=class_names)
        actual_idx = class_names.index(actual_class)
        pred_idx = int(pred["class_index"])
        y_true.append(actual_idx)
        y_pred.append(pred_idx)
        print(f"{path.name} | actual: {actual_class} | pred: {pred}")

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    print_metrics_report(metrics, title=f"Split '{split}' Metrics")
    return predictions, metrics

if __name__ == "__main__":
    hyperparam_configs = [
        {"epochs": 25, "batch_size": 16, "lr": 1e-3, "train_fraction": 0.85, "n_splits": 3},
        {"epochs": 30, "batch_size": 16, "lr": 1e-3, "train_fraction": 0.85, "n_splits": 3},
        {"epochs": 25, "batch_size": 20, "lr": 1e-3, "train_fraction": 0.85, "n_splits": 3},
        {"epochs": 25, "batch_size": 16, "lr": 1e-3, "train_fraction": 0.75, "n_splits": 3},
        {"epochs": 25, "batch_size": 16, "lr": 1e-5, "train_fraction": 0.85, "n_splits": 3},
        
    ]

    best_ckpt: Path | None = None
    best_val_f1 = -1.0

    for idx, cfg in enumerate(hyperparam_configs, start=1):
        trial_ckpt = CHECKPOINT_PATH.with_name(
            f"{CHECKPOINT_PATH.stem}_trial{idx}{CHECKPOINT_PATH.suffix}"
        )
        print(f"\n--- Running config {idx}/{len(hyperparam_configs)}: {cfg} ---")
        model_train(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            checkpoint_path=trial_ckpt,
            train_fraction=cfg["train_fraction"],
            n_splits=cfg["n_splits"],
        )

        ckpt_data = torch.load(trial_ckpt, map_location="cpu")
        val_f1 = float(ckpt_data.get("best_val_f1", ckpt_data.get("best_val_acc", 0.0)))
        print(f"Config {idx} best val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_ckpt = trial_ckpt

    print(f"\nBest checkpoint: {best_ckpt} (Val F1: {best_val_f1:.4f})")
    model_predict(checkpoint_path=best_ckpt or CHECKPOINT_PATH, split="train_dev")
