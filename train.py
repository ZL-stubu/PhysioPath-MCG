from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .configs import ExperimentConfig, TrainConfig
from .dataset import NodeEdgeGraphDataset
from .model import PathAttentiveGAT, build_model
from .utils import (
    LabelTable,
    SampleMeta,
    class_weights_from_counts,
    seed_all,
    stratified_patient_folds,
)


LOG = logging.getLogger("gnn_mcg.train")


# ---------------------------------------------------------------------------
# Focal loss (used for imbalanced multi-class)


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None, gamma: float):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
        )
        if self.gamma > 0:
            p_t = torch.exp(-ce)
            ce = ((1 - p_t) ** self.gamma) * ce
        return ce.mean()


# ---------------------------------------------------------------------------
# Metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if n_classes == 2:
        try:
            if len(np.unique(y_true)) < 2:
                metrics["auc"] = float("nan")
            else:
                metrics["auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        try:
            if len(np.unique(y_true)) < 2:
                metrics["auc_ovr"] = float("nan")
            else:
                metrics["auc_ovr"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["auc_ovr"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Transfer utilities


def _load_pretrained(model: PathAttentiveGAT, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    own_state = model.state_dict()
    matched = 0
    for key, value in state.items():
        if key not in own_state:
            continue
        if own_state[key].shape != value.shape:
            LOG.warning("Skip transfer for %s: shape %s vs %s",
                        key, list(own_state[key].shape), list(value.shape))
            continue
        own_state[key].copy_(value)
        matched += 1
    LOG.info("Transferred %d tensors from %s.", matched, checkpoint_path)


def _split_optim_params(
    model: PathAttentiveGAT,
    backbone_lr: float,
    head_lr: float,
) -> list[dict]:
    """Two optimizer groups: encoder+GAT get a lower LR, path/head get full LR."""
    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("signal_encoder", "edge_type_emb", "gat_layers")):
            backbone.append(param)
        else:
            head.append(param)
    return [
        {"params": backbone, "lr": backbone_lr},
        {"params": head, "lr": head_lr},
    ]


# ---------------------------------------------------------------------------
# One fold trainer


def _run_one_fold(
    fold_id: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    dataset: NodeEdgeGraphDataset,
    labels_for_weight: np.ndarray,
    cfg: ExperimentConfig,
    output_dir: Path,
) -> dict:
    seed_all(cfg.seed + fold_id)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    n_scalar = dataset[0].scalar.shape[-1]
    model = build_model(
        template=dataset.template,
        n_scalar_features=n_scalar,
        path_cfg=cfg.paths,
        model_cfg=cfg.model,
        n_classes=cfg.train.n_classes,
    ).to(device)
    model.set_l1_penalty(cfg.paths.l1_penalty)

    if cfg.train.pretrained_checkpoint is not None:
        _load_pretrained(model, cfg.train.pretrained_checkpoint)

    # Freeze backbone for first N epochs during transfer.
    if cfg.train.pretrained_checkpoint is not None and cfg.train.freeze_encoder_epochs > 0:
        for name, param in model.named_parameters():
            if name.startswith(("signal_encoder", "gat_layers", "edge_type_emb")):
                param.requires_grad = False

    param_groups = _split_optim_params(
        model,
        backbone_lr=cfg.train.lr * cfg.train.backbone_lr_scale,
        head_lr=cfg.train.lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    # Class weights from training set only
    if cfg.train.class_weights is not None:
        weights = torch.tensor(cfg.train.class_weights, dtype=torch.float32).to(device)
    else:
        counts = np.bincount(labels_for_weight[train_indices], minlength=cfg.train.n_classes)
        weights = torch.from_numpy(class_weights_from_counts(counts)).to(device)

    loss_fn = FocalLoss(class_weights=weights, gamma=cfg.train.focal_gamma)

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    best_state: dict | None = None
    best_metric = -float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(cfg.train.epochs):
        # Unfreeze backbone after warmup during transfer.
        if (
            cfg.train.pretrained_checkpoint is not None
            and epoch == cfg.train.freeze_encoder_epochs
        ):
            for param in model.parameters():
                param.requires_grad = True
            LOG.info("Fold %d epoch %d: unfroze backbone.", fold_id, epoch)

        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out["logits"], batch.y.view(-1)) + out["aux_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        val_metrics = _evaluate(model, val_loader, loss_fn, device, cfg.train.n_classes)
        val_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else float("nan")
        val_metrics["epoch"] = epoch
        val_metrics["fold"] = fold_id
        history.append(val_metrics)

        primary = val_metrics.get("auc", val_metrics.get("auc_ovr", val_metrics["macro_f1"]))
        if primary > best_metric:
            best_metric = primary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == cfg.train.epochs - 1:
            LOG.info(
                "Fold %d epoch %3d | train_loss %.4f | val %s",
                fold_id, epoch, val_metrics["train_loss"],
                {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)},
            )

        if epochs_no_improve >= cfg.train.early_stopping_patience:
            LOG.info("Fold %d early stopping at epoch %d (best %.4f).", fold_id, epoch, best_metric)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save fold artifacts
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    with (fold_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    # Final eval + gather path weights for interpretation
    final_metrics, per_sample = _evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        cfg.train.n_classes,
        return_per_sample=True,
    )
    per_sample["fold"] = fold_id
    np.savez_compressed(fold_dir / "val_preds.npz", **per_sample)

    return {
        "fold": fold_id,
        "best_metric": best_metric,
        "final_metrics": final_metrics,
        "checkpoint": str(fold_dir / "model.pt"),
    }


def _evaluate(
    model: PathAttentiveGAT,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    n_classes: int,
    return_per_sample: bool = False,
):
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    all_score: list[np.ndarray] = []
    all_paths: list[np.ndarray] = []
    all_sample_index: list[int] = []
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out["logits"], batch.y.view(-1))
            all_losses.append(float(loss.item()))
            probs = F.softmax(out["logits"], dim=-1)
            pred = probs.argmax(dim=-1)

            all_y.extend(batch.y.view(-1).cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_score.append(probs.cpu().numpy())
            all_paths.append(out["path_weights"].cpu().numpy())
            all_sample_index.extend(batch.sample_index.view(-1).cpu().tolist())

    y_true = np.asarray(all_y, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    y_score = np.concatenate(all_score, axis=0) if all_score else np.zeros((0, n_classes))
    metrics = compute_metrics(y_true, y_pred, y_score, n_classes)
    metrics["val_loss"] = float(np.mean(all_losses)) if all_losses else float("nan")

    if not return_per_sample:
        return metrics

    per_sample = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "path_weights": np.concatenate(all_paths, axis=0) if all_paths else np.zeros((0, 0)),
        "sample_index": np.asarray(all_sample_index, dtype=np.int64),
    }
    return metrics, per_sample


# ---------------------------------------------------------------------------
# Public entry point


def train_cross_validation(
    samples: Sequence[SampleMeta],
    labels_for_weight: np.ndarray,
    cfg: ExperimentConfig,
    output_dir: Path,
    label_override: Sequence[int] | None = None,
) -> dict:
    """
    Patient-level stratified CV. Returns per-fold and aggregated metrics.

    labels_for_weight is the ground-truth class array aligned with `samples`
    (integer class ids). label_override, when provided, is the class ids we
    hand to the dataset (for E2 multiclass remaps).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = NodeEdgeGraphDataset(
        samples=list(samples),
        cfg=cfg.graph,
        label_override=label_override,
    )
    folds = stratified_patient_folds(
        list(samples),
        n_folds=cfg.train.n_folds,
        seed=cfg.train.cv_seed,
    )

    fold_reports = []
    all_metrics: list[dict] = []
    t0 = time.time()
    for fold_id, val_idx in enumerate(folds):
        train_idx = np.concatenate(
            [f for i, f in enumerate(folds) if i != fold_id]
        )
        LOG.info(
            "Fold %d: train %d, val %d",
            fold_id, len(train_idx), len(val_idx),
        )
        report = _run_one_fold(
            fold_id=fold_id,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset=dataset,
            labels_for_weight=labels_for_weight,
            cfg=cfg,
            output_dir=output_dir,
        )
        fold_reports.append(report)
        all_metrics.append(report["final_metrics"])

    def _aggregate(key: str) -> float:
        vals = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _aggregate_std(key: str) -> float:
        vals = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        return float(np.std(vals)) if vals else float("nan")

    keys = sorted({k for m in all_metrics for k in m.keys()})
    aggregated = {k: (_aggregate(k), _aggregate_std(k)) for k in keys}

    summary = {
        "experiment": cfg.train.experiment,
        "output_tag": cfg.output_tag,
        "n_folds": cfg.train.n_folds,
        "wall_time_sec": time.time() - t0,
        "aggregated": aggregated,
        "per_fold": all_metrics,
        "fold_reports": fold_reports,
        "config": asdict(cfg),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

    LOG.info("Summary: %s", {k: v[0] for k, v in aggregated.items()})
    return summary
