"""Run three patient-level E1 logistic-regression comparison experiments.

The experiments share the exact same samples and Stage-1 patient folds:

1. ``handcrafted_1936``: 1936 deterministic statistics extracted from each
   NodeEdge NPZ (not a pre-existing 1936-D array stored in the NPZ).
2. ``raw_fusion_2080``: pooled64 + raw node-energy80 + handcrafted1936.
3. ``projected_fusion_192``: the exact original classifier input,
   pooled64 + projected-energy64 + projected-handcrafted64.

The Stage-1 checkpoint belonging to each fold extracts that fold's train and
validation representations.  Standardization and logistic regression are then
fit on the training partition only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from gnn_mcg.configs import (
    LABEL_XLSX,
    NODE_EDGE_DATA_DIR,
    REPAIRED_DIR,
    RESULTS_ROOT,
    make_e1_config,
)
from gnn_mcg.utils import (
    LabelTable,
    SampleMeta,
    discover_samples,
    filter_e1,
    stratified_patient_folds,
)
from logistic_regression_classifier import (
    EXPERIMENTS,
    binary_metrics,
    build_feature_matrix,
    make_classifier,
)


LOG = logging.getLogger("e1_logistic_comparison")
BRANCH_NAMES = ("pooled", "energy", "handcrafted")
DEFAULT_STAGE_DIR = RESULTS_ROOT / "e1_upgrade" / "stage1_energy_feature"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "e1_logistic_comparison"


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_stage_config(stage_dir: Path):
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少 Stage-1 汇总文件：{summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    saved_config = summary.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError(f"{summary_path} 缺少 config")

    cfg = make_e1_config()
    for section_name in ("graph", "paths", "model", "train"):
        values = saved_config.get(section_name)
        if not isinstance(values, dict):
            raise ValueError(f"{summary_path} 缺少 config.{section_name}")
        section = getattr(cfg, section_name)
        for key, value in values.items():
            if not hasattr(section, key):
                continue
            if key == "start_phase_ids":
                value = tuple(value)
            elif key == "class_weights" and value is not None:
                value = tuple(value)
            elif key == "pretrained_checkpoint" and value is not None:
                value = Path(value)
            setattr(section, key, value)

    for key in ("device", "seed", "num_workers", "output_tag"):
        if key in saved_config:
            setattr(cfg, key, saved_config[key])
    return cfg, summary


def _checkpoint_paths(
    stage_dir: Path,
    stage_summary: dict[str, object],
) -> list[Path]:
    reports = stage_summary.get("fold_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("Stage-1 summary 缺少 fold_reports")

    checkpoints: list[Path] = []
    for fold_index, report in enumerate(reports):
        checkpoint = stage_dir / f"fold_{fold_index}" / "model.pt"
        if not checkpoint.is_file() and isinstance(report, dict):
            reported_path = report.get("checkpoint")
            if reported_path:
                checkpoint = Path(str(reported_path))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"缺少第 {fold_index} 折模型：{checkpoint}")
        checkpoints.append(checkpoint)
    return checkpoints


def _concatenate_branches(
    branches: dict[str, np.ndarray],
    expected_dims: dict[str, int],
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    n_samples: int | None = None
    actual_dims: dict[str, int] = {}
    for name in BRANCH_NAMES:
        branch = np.asarray(branches[name])
        if branch.ndim < 2:
            raise ValueError(f"{name} 特征形状无效：{branch.shape}")
        matrix = branch.reshape(branch.shape[0], -1).astype(np.float64, copy=False)
        if n_samples is None:
            n_samples = matrix.shape[0]
        elif matrix.shape[0] != n_samples:
            raise ValueError("三路特征的样本数量不一致")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} 特征包含 NaN/Inf")
        actual_dims[name] = int(matrix.shape[1])
        matrices.append(matrix)

    if actual_dims != expected_dims:
        raise RuntimeError(
            f"三路特征维度不符合实验定义：期望 {expected_dims}，实际 {actual_dims}"
        )
    result = np.concatenate(matrices, axis=1)
    if result.shape[1] != sum(expected_dims.values()):
        raise RuntimeError("三路特征拼接宽度异常")
    return result


def _patient_splits(
    samples: Sequence[SampleMeta],
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    validation_folds = stratified_patient_folds(
        list(samples),
        n_folds=folds,
        seed=seed,
    )
    return [
        (
            np.concatenate(
                [
                    fold
                    for other_index, fold in enumerate(validation_folds)
                    if other_index != fold_index
                ]
            ),
            validation_indices,
        )
        for fold_index, validation_indices in enumerate(validation_folds)
    ]


def extract_fold_feature_sets(
    samples: Sequence[SampleMeta],
    labels: np.ndarray,
    handcrafted: np.ndarray,
    stage_dir: Path,
    folds: int,
    seed: int,
    device_name: str,
    batch_size: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[np.ndarray, np.ndarray]],
    int,
    dict[str, object],
]:
    """Extract raw-2080 and projected-192 representations in one pass."""
    if handcrafted.shape != (len(samples), 1936):
        raise ValueError(
            f"手工特征应为 [n_samples, 1936]，实际为 {handcrafted.shape}"
        )
    if batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")

    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise RuntimeError("需要 PyTorch 和 PyTorch Geometric") from exc

    from experiments.e1_upgrade.model_variants import (
        build_upgraded_model,
        stage1_energy_feature_variant,
    )
    from gnn_mcg.dataset import NodeEdgeGraphDataset

    cfg, stage_summary = _load_stage_config(stage_dir)
    checkpoints = _checkpoint_paths(stage_dir, stage_summary)
    if folds != int(cfg.train.n_folds) or folds != len(checkpoints):
        raise ValueError(
            f"必须使用 checkpoint 的 {cfg.train.n_folds} 折，当前为 {folds}"
        )
    if seed != int(cfg.train.cv_seed):
        raise ValueError(
            f"必须使用 checkpoint 的 cv_seed={cfg.train.cv_seed}，当前为 {seed}"
        )

    splits = _patient_splits(samples, folds, seed)
    handcrafted_float = np.asarray(handcrafted, dtype=np.float32)

    class FeatureDataset(NodeEdgeGraphDataset):
        def __getitem__(self, index: int):
            data = super().__getitem__(index)
            data.handcrafted = torch.from_numpy(
                handcrafted_float[index]
            ).unsqueeze(0)
            return data

    dataset = FeatureDataset(
        samples=list(samples),
        cfg=cfg.graph,
        label_override=labels.tolist(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但没有可用 GPU")
    LOG.info("Stage-1 特征提取设备：%s", device)

    raw_dims = {"pooled": 64, "energy": 80, "handcrafted": 1936}
    projected_dims = {"pooled": 64, "energy": 64, "handcrafted": 64}
    raw_matrices: list[np.ndarray] = []
    projected_matrices: list[np.ndarray] = []

    for fold_index, checkpoint in enumerate(checkpoints):
        variant = stage1_energy_feature_variant()
        variant.feature_dim = handcrafted.shape[1]
        n_scalar_features = int(dataset[0].scalar.shape[-1])
        model = build_upgraded_model(
            template=dataset.template,
            n_scalar_features=n_scalar_features,
            path_cfg=cfg.paths,
            model_cfg=cfg.model,
            n_classes=cfg.train.n_classes,
            variant=variant,
        )
        try:
            state_dict = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        captured: dict[str, torch.Tensor] = {}

        def capture_pooled(_module, _inputs, output) -> None:
            captured["pooled"] = output["pooled"].detach()

        def capture_energy(_module, _inputs, output) -> None:
            captured["energy"] = output.detach()

        def capture_handcrafted(_module, _inputs, output) -> None:
            captured["handcrafted"] = output.detach()

        handles = (
            model.path_attention.register_forward_hook(capture_pooled),
            model.node_energy_proj.register_forward_hook(capture_energy),
            model.feature_proj.register_forward_hook(capture_handcrafted),
        )
        raw_collected = {name: [] for name in BRANCH_NAMES}
        projected_collected = {name: [] for name in BRANCH_NAMES}
        try:
            with torch.inference_mode():
                for batch in loader:
                    captured.clear()
                    batch = batch.to(device)
                    output = model(batch)
                    missing = [name for name in BRANCH_NAMES if name not in captured]
                    if missing:
                        raise RuntimeError(
                            f"第 {fold_index} 折未捕获到：{', '.join(missing)}"
                        )

                    raw_handcrafted = batch.handcrafted
                    if raw_handcrafted.dim() == 1:
                        raw_handcrafted = raw_handcrafted.unsqueeze(0)
                    raw_values = {
                        "pooled": captured["pooled"],
                        "energy": output["node_embeddings"].norm(dim=-1),
                        "handcrafted": raw_handcrafted,
                    }
                    projected_values = {
                        "pooled": captured["pooled"],
                        "energy": captured["energy"],
                        "handcrafted": captured["handcrafted"],
                    }
                    for name in BRANCH_NAMES:
                        raw_collected[name].append(
                            raw_values[name].cpu().numpy()
                        )
                        projected_collected[name].append(
                            projected_values[name].cpu().numpy()
                        )
        finally:
            for handle in handles:
                handle.remove()

        raw_branches = {
            name: np.concatenate(raw_collected[name], axis=0)
            for name in BRANCH_NAMES
        }
        projected_branches = {
            name: np.concatenate(projected_collected[name], axis=0)
            for name in BRANCH_NAMES
        }
        raw_matrix = _concatenate_branches(raw_branches, raw_dims)
        projected_matrix = _concatenate_branches(
            projected_branches,
            projected_dims,
        )
        if raw_matrix.shape != (len(samples), 2080):
            raise RuntimeError(f"2080 维矩阵形状异常：{raw_matrix.shape}")
        if projected_matrix.shape != (len(samples), 192):
            raise RuntimeError(f"192 维矩阵形状异常：{projected_matrix.shape}")
        raw_matrices.append(raw_matrix)
        projected_matrices.append(projected_matrix)
        LOG.info(
            "Fold %d/%d 特征完成：raw=%s, projected=%s",
            fold_index + 1,
            folds,
            raw_matrix.shape,
            projected_matrix.shape,
        )

    reports = stage_summary["fold_reports"]
    best_fold = max(
        range(folds),
        key=lambda index: float(reports[index]["best_metric"]),
    )
    extraction_metadata = {
        "stage_dir": str(stage_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "best_fold_for_final_artifact": best_fold,
        "raw_branch_dimensions": raw_dims,
        "projected_branch_dimensions": projected_dims,
    }
    return (
        raw_matrices,
        projected_matrices,
        splits,
        best_fold,
        {"stage_summary": stage_summary, **extraction_metadata},
    )


def _feature_names(branch_dims: dict[str, int]) -> list[str]:
    return [
        f"{branch}__{index:04d}"
        for branch in BRANCH_NAMES
        for index in range(branch_dims[branch])
    ]


def run_lr_experiment(
    name: str,
    samples: Sequence[SampleMeta],
    labels: np.ndarray,
    fold_matrices: Sequence[np.ndarray],
    final_matrix: np.ndarray,
    feature_names: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    seed: int,
    c_value: float,
    max_iter: int,
    balanced: bool,
    top_features: int,
    feature_metadata: dict[str, object],
) -> dict[str, object]:
    """Fit/evaluate LR with fold-specific matrices and save all artifacts."""
    n_samples = labels.size
    n_features = len(feature_names)
    if len(fold_matrices) != len(splits):
        raise ValueError("逐折特征矩阵数量与折数不一致")
    if final_matrix.shape != (n_samples, n_features):
        raise ValueError(f"最终特征矩阵形状无效：{final_matrix.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.full(n_samples, -1, dtype=np.int64)
    probabilities = np.full(n_samples, np.nan, dtype=np.float64)
    assignments = np.full(n_samples, -1, dtype=np.int64)
    fold_rows: list[dict[str, float | int]] = []

    for fold_index, ((train_indices, val_indices), matrix) in enumerate(
        zip(splits, fold_matrices, strict=True),
        start=1,
    ):
        matrix = np.asarray(matrix)
        if matrix.shape != (n_samples, n_features):
            raise ValueError(
                f"第 {fold_index} 折特征形状无效：{matrix.shape}"
            )
        classifier = make_classifier(
            c_value,
            max_iter,
            seed + fold_index,
            balanced,
        )
        classifier.fit(matrix[train_indices], labels[train_indices])
        fold_prediction = classifier.predict(matrix[val_indices]).astype(np.int64)
        fold_probability = classifier.predict_proba(matrix[val_indices])[:, 1]
        predictions[val_indices] = fold_prediction
        probabilities[val_indices] = fold_probability
        assignments[val_indices] = fold_index

        metrics = binary_metrics(
            labels[val_indices],
            fold_prediction,
            fold_probability,
        )
        metrics.update(
            {
                "fold": fold_index,
                "n_train": int(train_indices.size),
                "n_validation": int(val_indices.size),
            }
        )
        fold_rows.append(metrics)
        LOG.info(
            "%s fold %d/%d: balanced_accuracy=%.4f, roc_auc=%.4f",
            name,
            fold_index,
            len(splits),
            metrics["balanced_accuracy"],
            metrics["roc_auc"],
        )

    if np.any(predictions < 0) or np.any(~np.isfinite(probabilities)):
        raise RuntimeError(f"{name} 的 OOF 预测未覆盖全部样本")
    oof_metrics = binary_metrics(labels, predictions, probabilities)

    spec = EXPERIMENTS["e1"]
    pd.DataFrame(
        {
            "fold": assignments,
            "sample_id": [sample.sample_id for sample in samples],
            "patient_key": [sample.patient_key for sample in samples],
            "true_label": labels,
            "predicted_label": predictions,
            f"probability_{spec.positive_class}": probabilities,
            "node_edge_npz": [str(sample.npz_path) for sample in samples],
        }
    ).to_csv(
        output_dir / "cross_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_table = pd.DataFrame(fold_rows).sort_values("fold")
    fold_table.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        output_dir / "feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_classifier = make_classifier(c_value, max_iter, seed, balanced)
    final_classifier.fit(final_matrix, labels)
    joblib.dump(
        {
            "pipeline": final_classifier,
            "experiment": "e1",
            "comparison_name": name,
            "class_names": {0: spec.negative_class, 1: spec.positive_class},
            "feature_names": list(feature_names),
            "feature_metadata": feature_metadata,
        },
        output_dir / "final_logistic_regression.joblib",
    )

    coefficients = np.asarray(
        final_classifier.named_steps["logistic_regression"].coef_[0],
        dtype=np.float64,
    )
    pd.DataFrame(
        {
            "feature_name": feature_names,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(
                coefficients >= 0,
                spec.positive_class,
                spec.negative_class,
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    ).head(
        max(0, top_features)
    ).to_csv(
        output_dir / "top_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
    )
    summary: dict[str, object] = {
        "experiment": "e1",
        "comparison_name": name,
        "classifier": "sklearn.linear_model.LogisticRegression (L2)",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_folds": len(splits),
        "seed": seed,
        "C": c_value,
        "max_iter": max_iter,
        "class_weight": "balanced" if balanced else None,
        "feature_metadata": feature_metadata,
        "oof_metrics": oof_metrics,
        "fold_metric_mean": {
            metric: float(fold_table[metric].mean())
            for metric in metric_names
        },
        "fold_metric_std": {
            metric: float(fold_table[metric].std(ddof=1))
            for metric in metric_names
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(summary),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 E1 的 1936、2080、192 维三组逻辑回归对比实验。"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", dest="c_value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--top-features", type=int, default=100)
    parser.add_argument("--label-xlsx", type=Path, default=LABEL_XLSX)
    parser.add_argument("--repaired-dir", type=Path, default=REPAIRED_DIR)
    parser.add_argument("--node-edge-dir", type=Path, default=NODE_EDGE_DATA_DIR)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        label_table = LabelTable(args.label_xlsx)
        samples = filter_e1(
            discover_samples(
                repaired_dir=args.repaired_dir,
                node_edge_dir=args.node_edge_dir,
                labels=label_table,
            )
        )
        if not samples:
            raise ValueError("E1 没有可用样本")
        labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
        handcrafted, handcrafted_names = build_feature_matrix(samples)
        if handcrafted.shape[1] != 1936:
            raise ValueError(
                f"传统手工特征应为 1936 维，实际为 {handcrafted.shape[1]}"
            )

        (
            raw_matrices,
            projected_matrices,
            splits,
            best_fold,
            extraction_metadata,
        ) = extract_fold_feature_sets(
            samples=samples,
            labels=labels,
            handcrafted=handcrafted,
            stage_dir=args.stage_dir,
            folds=args.folds,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
        )

        experiment_inputs = (
            (
                "handcrafted_1936",
                [handcrafted] * args.folds,
                handcrafted,
                handcrafted_names,
                {
                    "representation": "handcrafted",
                    "dimensions": {"handcrafted": 1936},
                },
            ),
            (
                "raw_fusion_2080",
                raw_matrices,
                raw_matrices[best_fold],
                _feature_names(
                    {"pooled": 64, "energy": 80, "handcrafted": 1936}
                ),
                {
                    "representation": "raw_fusion",
                    "dimensions": {
                        "pooled": 64,
                        "energy": 80,
                        "handcrafted": 1936,
                    },
                    **{
                        key: value
                        for key, value in extraction_metadata.items()
                        if key != "stage_summary"
                    },
                },
            ),
            (
                "projected_fusion_192",
                projected_matrices,
                projected_matrices[best_fold],
                _feature_names(
                    {"pooled": 64, "energy": 64, "handcrafted": 64}
                ),
                {
                    "representation": "classifier_head_input",
                    "dimensions": {
                        "pooled": 64,
                        "energy": 64,
                        "handcrafted": 64,
                    },
                    **{
                        key: value
                        for key, value in extraction_metadata.items()
                        if key != "stage_summary"
                    },
                },
            ),
        )

        summaries: dict[str, object] = {}
        for name, matrices, final_matrix, names, metadata in experiment_inputs:
            LOG.info("开始 %s（%d 维）", name, len(names))
            summaries[name] = run_lr_experiment(
                name=name,
                samples=samples,
                labels=labels,
                fold_matrices=matrices,
                final_matrix=final_matrix,
                feature_names=names,
                splits=splits,
                output_dir=args.output_root / name,
                seed=args.seed,
                c_value=args.c_value,
                max_iter=args.max_iter,
                balanced=not args.unweighted,
                top_features=args.top_features,
                feature_metadata=metadata,
            )

        comparison = {
            "experiment": "E1 health vs NSTEMI",
            "shared_patient_folds": args.folds,
            "seed": args.seed,
            "logistic_regression": summaries,
            "original_stage1_model": extraction_metadata["stage_summary"].get(
                "aggregated"
            ),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "comparison_summary.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                _json_ready(comparison),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        LOG.info("三组对比完成：%s", args.output_root)
        return 0
    except Exception as exc:
        LOG.exception("E1 逻辑回归对比运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
