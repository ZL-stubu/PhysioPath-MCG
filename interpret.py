from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch_geometric.loader import DataLoader

from .configs import ExperimentConfig
from .dataset import NodeEdgeGraphDataset
from .model import PathAttentiveGAT, PathCandidate, build_model
from .utils import SampleMeta


LOG = logging.getLogger("gnn_mcg.interpret")


PHASE_LABELS = (
    "baseline",
    "qrs_depolarization",
    "early_st",
    "late_repolarization",
    "t_recovery",
)


def _roi_labels(roi_side: int) -> list[str]:
    return [
        f"R{i // roi_side + 1}C{i % roi_side + 1}"
        for i in range(roi_side * roi_side)
    ]


def _rebuild_model(
    dataset: NodeEdgeGraphDataset,
    cfg: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
) -> PathAttentiveGAT:
    n_scalar = dataset[0].scalar.shape[-1]
    model = build_model(
        template=dataset.template,
        n_scalar_features=n_scalar,
        path_cfg=cfg.paths,
        model_cfg=cfg.model,
        n_classes=cfg.train.n_classes,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def _collect_extended(
    model: PathAttentiveGAT,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Runs one pass and gathers everything needed for interpretability:
      - path_weights        [N, P]  softmax attention
      - path_scores         [N, P]  raw pre-softmax scores
      - path_energy_mean    [N, P]  mean L2 norm of node embeddings along the path
      - path_gru_norm       [N, P]  L2 norm of the path's GRU-encoded embedding
      - node_emb_norm       [N, n_nodes]  per-node L2 norm of GAT output
      - labels, preds, correct
    """
    weights_all, scores_all, energy_all, gru_norm_all = [], [], [], []
    node_norm_all = []
    labels_all, preds_all, correct_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = F.softmax(out["logits"], dim=-1)
            pred = probs.argmax(dim=-1)
            weights_all.append(out["path_weights"].cpu().numpy())
            scores_all.append(out["path_scores"].cpu().numpy())
            energy_all.append(out["path_energy_mean"].cpu().numpy())
            gru_norm_all.append(out["path_gru_norm"].cpu().numpy())
            # node_embeddings: [B, n_nodes, D] -> per-node L2 norm
            node_emb = out["node_embeddings"]
            node_norm_all.append(node_emb.norm(dim=-1).cpu().numpy())
            labels_all.extend(batch.y.view(-1).cpu().tolist())
            preds_all.extend(pred.cpu().tolist())
            correct_all.extend((pred == batch.y.view(-1)).cpu().tolist())
    return {
        "path_weights": np.concatenate(weights_all, axis=0),
        "path_scores": np.concatenate(scores_all, axis=0),
        "path_energy_mean": np.concatenate(energy_all, axis=0),
        "path_gru_norm": np.concatenate(gru_norm_all, axis=0),
        "node_emb_norm": np.concatenate(node_norm_all, axis=0),
        "labels": np.asarray(labels_all, dtype=np.int64),
        "preds": np.asarray(preds_all, dtype=np.int64),
        "correct": np.asarray(correct_all, dtype=bool),
    }


def _class_prototypes(
    weights: np.ndarray,
    labels: np.ndarray,
    correct_mask: np.ndarray,
    n_classes: int,
    top_k: int,
    only_correct: bool,
) -> dict[int, list[tuple[int, float, float]]]:
    prototypes: dict[int, list[tuple[int, float, float]]] = {}
    if weights.size == 0:
        return prototypes
    activation_threshold = 1.0 / weights.shape[1] * 2.0
    for cls in range(n_classes):
        pick = labels == cls
        if only_correct:
            pick = pick & correct_mask
        if not np.any(pick):
            prototypes[cls] = []
            continue
        cls_weights = weights[pick]
        mean_w = cls_weights.mean(axis=0)
        activation = (cls_weights > activation_threshold).mean(axis=0)
        top = np.argsort(mean_w)[::-1][:top_k]
        prototypes[cls] = [
            (int(idx), float(mean_w[idx]), float(activation[idx]))
            for idx in top
        ]
    return prototypes


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    bh_p = np.empty_like(adjusted)
    bh_p[order] = adjusted
    return bh_p


def _class_diff_paths_with_effect(
    matrix: np.ndarray,       # [N, P] any per-path scalar (weight, energy, activation…)
    labels: np.ndarray,
    positive_class: int,
    top_k: int,
) -> list[dict]:
    """
    Welch t-test + Cohen's d + BH-corrected p-values, sorted by |d|.
    Better than raw mean_diff because effect size is scale-invariant.
    """
    if matrix.size == 0:
        return []
    pos_mask = labels == positive_class
    neg_mask = ~pos_mask
    if not np.any(pos_mask) or not np.any(neg_mask):
        return []

    n_paths = matrix.shape[1]
    p_values = np.ones(n_paths)
    t_stats = np.zeros(n_paths)
    mean_diffs = np.zeros(n_paths)
    cohen_d = np.zeros(n_paths)
    for p in range(n_paths):
        pos = matrix[pos_mask, p]
        neg = matrix[neg_mask, p]
        try:
            t, pv = stats.ttest_ind(pos, neg, equal_var=False)
        except (ValueError, FloatingPointError):
            t, pv = 0.0, 1.0
        if not np.isfinite(t):
            t, pv = 0.0, 1.0
        p_values[p] = pv
        t_stats[p] = t
        mean_diffs[p] = float(pos.mean() - neg.mean())
        cohen_d[p] = _cohens_d(pos, neg)
    bh_p = _bh_correct(p_values)

    order = np.argsort(-np.abs(cohen_d))[:top_k]
    return [
        {
            "path_idx": int(idx),
            "mean_diff": float(mean_diffs[idx]),
            "cohens_d": float(cohen_d[idx]),
            "t_stat": float(t_stats[idx]),
            "p_bh": float(bh_p[idx]),
        }
        for idx in order
    ]


def class_diff_paths_with_effect(
    matrix: np.ndarray,
    labels: np.ndarray,
    positive_class: int = 1,
    top_k: int = 10,
) -> list[dict]:
    """Public path-association summary with multiplicity correction.

    The returned rows rank paths by absolute Cohen's d and include Welch's
    t-statistic plus Benjamini-Hochberg-adjusted p-values.  Keeping this
    wrapper public lets architecture-specific interpretation pipelines reuse
    the same statistical definition without depending on a private helper.
    """
    return _class_diff_paths_with_effect(
        matrix=np.asarray(matrix),
        labels=np.asarray(labels),
        positive_class=positive_class,
        top_k=top_k,
    )


def _node_level_class_diff(
    node_norm: np.ndarray,    # [N, n_nodes]
    labels: np.ndarray,
    positive_class: int,
    n_phases: int,
    roi_side: int,
) -> dict:
    """
    For each of n_nodes = n_phases × roi_side^2, test whether the embedding
    magnitude differs between the positive class and the rest. Return the
    result reshaped to [n_phases, roi_side, roi_side] so it can be plotted
    directly as a stack of heatmaps.
    """
    n_rois = roi_side * roi_side
    n_nodes = n_phases * n_rois
    assert node_norm.shape[1] == n_nodes, (node_norm.shape, n_nodes)

    pos_mask = labels == positive_class
    neg_mask = ~pos_mask
    if not np.any(pos_mask) or not np.any(neg_mask):
        return {}

    pos_mean = node_norm[pos_mask].mean(axis=0)
    neg_mean = node_norm[neg_mask].mean(axis=0)
    mean_diff = pos_mean - neg_mean

    cohen = np.zeros(n_nodes)
    t_stat = np.zeros(n_nodes)
    p_vals = np.ones(n_nodes)
    for n in range(n_nodes):
        pos = node_norm[pos_mask, n]
        neg = node_norm[neg_mask, n]
        try:
            t, pv = stats.ttest_ind(pos, neg, equal_var=False)
        except (ValueError, FloatingPointError):
            t, pv = 0.0, 1.0
        if not np.isfinite(t):
            t, pv = 0.0, 1.0
        cohen[n] = _cohens_d(pos, neg)
        t_stat[n] = t
        p_vals[n] = pv
    bh = _bh_correct(p_vals)

    shape = (n_phases, roi_side, roi_side)
    return {
        "pos_mean_map": pos_mean.reshape(shape),
        "neg_mean_map": neg_mean.reshape(shape),
        "mean_diff_map": mean_diff.reshape(shape),
        "cohens_d_map": cohen.reshape(shape),
        "t_stat_map": t_stat.reshape(shape),
        "p_bh_map": bh.reshape(shape),
    }


def _describe_path(candidate: PathCandidate, roi_labels: Sequence[str]) -> str:
    return candidate.description(roi_labels, PHASE_LABELS)


def extract_and_save_prototypes(
    samples: Sequence[SampleMeta],
    cfg: ExperimentConfig,
    fold_checkpoints: Sequence[Path],
    output_dir: Path,
    label_override: Sequence[int] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Cross-fold interpretability. Averages per-sample tensors across folds and
    computes:
      1. class_prototypes           — top-K paths by attention weight (legacy)
      2. path_signal_prototypes     — top-K paths by weight × energy (real "signal flow")
      3. class_differentiating_paths — Welch t-test + Cohen's d + BH on attention
      4. path_activation_diff       — same test on weight × energy
      5. node_level_diff            — per-node L2 norm class comparison, shaped as 5 heatmaps
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = NodeEdgeGraphDataset(
        samples=list(samples),
        cfg=cfg.graph,
        label_override=label_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    fold_dumps: list[dict[str, np.ndarray]] = []
    for checkpoint in fold_checkpoints:
        model = _rebuild_model(dataset, cfg, checkpoint, device)
        fold_dumps.append(_collect_extended(model, loader, device))

    # Average per-sample tensors across folds.
    stacked = {}
    for key in ("path_weights", "path_scores", "path_energy_mean", "path_gru_norm", "node_emb_norm"):
        stacked[key] = np.stack([fd[key] for fd in fold_dumps], axis=0).mean(axis=0)
    labels = fold_dumps[0]["labels"]
    correct = np.stack([fd["correct"] for fd in fold_dumps], axis=0).mean(axis=0) >= 0.5

    mean_weights = stacked["path_weights"]
    mean_energy = stacked["path_energy_mean"]
    activation = mean_weights * mean_energy    # [N, P] real "signal amount routed through path"
    node_norm = stacked["node_emb_norm"]

    prototypes = _class_prototypes(
        mean_weights, labels, correct, cfg.train.n_classes, top_k, only_correct=True,
    )
    signal_prototypes = _class_prototypes(
        activation, labels, correct, cfg.train.n_classes, top_k, only_correct=True,
    )

    # Load path descriptions once.
    dummy_model = _rebuild_model(dataset, cfg, fold_checkpoints[0], device)
    path_candidates: list[PathCandidate] = dummy_model.state.path_candidates
    roi_labels = _roi_labels(cfg.graph.roi_side)

    def _serialize_protos(protos: dict[int, list[tuple[int, float, float]]]) -> dict:
        out = {}
        for cls, entries in protos.items():
            out[str(cls)] = [
                {
                    "path_idx": p_idx,
                    "path": _describe_path(path_candidates[p_idx], roi_labels),
                    "mean_score": mean_score,
                    "activation_rate": act,
                }
                for p_idx, mean_score, act in entries
            ]
        return out

    # Class-differentiating paths on TWO scalars (attention and activation),
    # ranked by Cohen's d.
    class_diff_weight: dict[str, list[dict]] = {}
    class_diff_activation: dict[str, list[dict]] = {}
    diff_targets = (
        [1] if cfg.train.n_classes == 2 else list(range(cfg.train.n_classes))
    )
    for cls in diff_targets:
        cw = _class_diff_paths_with_effect(mean_weights, labels, cls, top_k)
        ca = _class_diff_paths_with_effect(activation, labels, cls, top_k)
        for entries in (cw, ca):
            for row in entries:
                row["path"] = _describe_path(path_candidates[row["path_idx"]], roi_labels)
        class_diff_weight[str(cls)] = cw
        class_diff_activation[str(cls)] = ca

    # Node-level per-class map.
    node_maps: dict[str, dict] = {}
    node_map_arrays: dict[str, np.ndarray] = {}
    for cls in diff_targets:
        result = _node_level_class_diff(
            node_norm=node_norm,
            labels=labels,
            positive_class=cls,
            n_phases=cfg.graph.n_phases,
            roi_side=cfg.graph.roi_side,
        )
        if not result:
            continue
        node_maps[str(cls)] = {
            "phase_labels": list(PHASE_LABELS[: cfg.graph.n_phases]),
            "roi_side": cfg.graph.roi_side,
            "mean_diff": result["mean_diff_map"].tolist(),
            "cohens_d": result["cohens_d_map"].tolist(),
            "p_bh": result["p_bh_map"].tolist(),
            "top_nodes": _top_nodes(result, top_k, cfg.graph.n_phases, cfg.graph.roi_side, roi_labels),
        }
        for k, arr in result.items():
            node_map_arrays[f"class{cls}_{k}"] = arr

    serialized = {
        "cohort_size": int(mean_weights.shape[0]),
        "n_paths": int(mean_weights.shape[1]),
        "class_prototypes": _serialize_protos(prototypes),
        "path_signal_prototypes": _serialize_protos(signal_prototypes),
        "class_differentiating_paths": class_diff_weight,   # by attention (legacy)
        "path_activation_diff": class_diff_activation,      # by weight × energy (recommended)
        "node_level_diff": node_maps,
        "notes": {
            "path_signal_prototypes": "Top paths ranked by weight × mean node embedding energy — 'signal flow' proxy.",
            "path_activation_diff": "Class-discriminative paths ranked by Cohen's d on (weight × energy).",
            "node_level_diff": "Per-node class-diff heatmap. Shape [n_phases, roi_side, roi_side]. Cohen's d is the recommended readout.",
        },
    }

    with (output_dir / "path_prototypes.json").open("w", encoding="utf-8") as fh:
        json.dump(serialized, fh, ensure_ascii=False, indent=2)

    np.savez_compressed(
        output_dir / "interpret_arrays.npz",
        mean_weights=mean_weights,
        mean_energy=mean_energy,
        activation=activation,
        node_norm=node_norm,
        labels=labels,
        correct=correct,
        **node_map_arrays,
    )
    # Preserve the legacy path_weight_matrix.npz name for downstream tools.
    np.savez_compressed(
        output_dir / "path_weight_matrix.npz",
        mean_weights=mean_weights,
        labels=labels,
        correct=correct,
    )

    LOG.info("Wrote prototypes to %s", output_dir / "path_prototypes.json")
    LOG.info("Wrote raw arrays to %s", output_dir / "interpret_arrays.npz")
    return serialized


def _top_nodes(
    result: dict,
    top_k: int,
    n_phases: int,
    roi_side: int,
    roi_labels: Sequence[str],
) -> list[dict]:
    n_rois = roi_side * roi_side
    d = result["cohens_d_map"].reshape(-1)
    p = result["p_bh_map"].reshape(-1)
    md = result["mean_diff_map"].reshape(-1)
    order = np.argsort(-np.abs(d))[:top_k]
    entries = []
    for node in order:
        phase_id = int(node) // n_rois
        roi_id = int(node) % n_rois
        entries.append({
            "node_idx": int(node),
            "phase": PHASE_LABELS[phase_id] if phase_id < len(PHASE_LABELS) else str(phase_id),
            "roi": roi_labels[roi_id],
            "cohens_d": float(d[node]),
            "mean_diff": float(md[node]),
            "p_bh": float(p[node]),
        })
    return entries


def path_stability_across_folds(
    fold_weight_files: Sequence[Path],
    top_k: int = 10,
) -> dict:
    """Jaccard overlap of top-K paths across folds."""
    top_sets = []
    for fp in fold_weight_files:
        preds = np.load(fp)
        if "path_weights" not in preds.files:
            continue
        pw = preds["path_weights"]
        if pw.size == 0:
            continue
        mean_w = pw.mean(axis=0)
        top_sets.append(set(np.argsort(mean_w)[::-1][:top_k].tolist()))

    if len(top_sets) < 2:
        return {"overlap": [], "mean_jaccard": float("nan")}

    overlaps = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            a, b = top_sets[i], top_sets[j]
            union = len(a | b)
            overlaps.append(len(a & b) / union if union else 0.0)
    return {
        "overlap": overlaps,
        "mean_jaccard": float(np.mean(overlaps)),
        "std_jaccard": float(np.std(overlaps)),
    }
