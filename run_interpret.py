"""
从既有 fold checkpoint 重跑可解释性输出（不训练）。

用法示例：
    python run_interpret.py --experiment e1
    python run_interpret.py --experiment e2
    python run_interpret.py --experiment e2 --output-tag e2_primary_scratch_unfrozen

会在 results/<output_tag>/interpret_v2/ 下生成：
    - path_prototypes.json    新版可解释报告
    - interpret_arrays.npz    node/path 层级的原始矩阵
    - heatmaps/               每个类别 5 张 phase 热图（PNG）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from gnn_mcg.configs import (
    CHECKPOINT_ROOT,
    LABEL_XLSX,
    NODE_EDGE_DATA_DIR,
    REPAIRED_DIR,
    RESULTS_ROOT,
    make_e1_config,
    make_e2_binary_config,
)
from gnn_mcg.interpret import extract_and_save_prototypes
from gnn_mcg.utils import (
    LabelTable,
    discover_samples,
    filter_e1,
    filter_e2_binary,
    seed_all,
)


LOG = logging.getLogger("run_interpret")


def _collect_fold_checkpoints(results_dir: Path) -> list[Path]:
    ckpts: list[Path] = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        p = fold_dir / "model.pt"
        if p.is_file():
            ckpts.append(p)
    return ckpts


def _plot_heatmaps(npz_path: Path, output_dir: Path, cfg) -> None:
    """Save one PNG per (class, phase). Skips silently if matplotlib is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not installed — skipping heatmap PNGs.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    phase_labels = [
        "baseline",
        "qrs_depolarization",
        "early_st",
        "late_repolarization",
        "t_recovery",
    ][: cfg.graph.n_phases]

    for key in data.files:
        if not key.endswith("_cohens_d_map"):
            continue
        cls = key.split("_")[0]
        arr = data[key]  # [n_phases, roi_side, roi_side]
        vmax = float(np.abs(arr).max()) or 1.0
        fig, axes = plt.subplots(1, arr.shape[0], figsize=(3 * arr.shape[0], 3.2))
        if arr.shape[0] == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(arr[i], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.set_title(f"{cls} · {phase_labels[i]}\nCohen's d vs rest")
            ax.set_xticks(range(arr.shape[2]))
            ax.set_yticks(range(arr.shape[1]))
            for r in range(arr.shape[1]):
                for c in range(arr.shape[2]):
                    ax.text(
                        c, r, f"{arr[i, r, c]:+.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(arr[i, r, c]) < vmax * 0.6 else "white",
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = output_dir / f"{cls}_cohens_d.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        LOG.info("Saved heatmap %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e1", "e2"), default="e1")
    parser.add_argument("--output-tag", type=str, default= None,
                        help="覆盖默认 output_tag（例如 e2_primary_scratch_unfrozen）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repaired-dir", type=Path, default=REPAIRED_DIR)
    parser.add_argument("--node-edge-dir", type=Path, default=NODE_EDGE_DATA_DIR)
    parser.add_argument("--no-heatmap", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.experiment == "e1":
        cfg = make_e1_config()
        filter_fn = filter_e1
    else:
        cfg = make_e2_binary_config(pretrained=None)
        filter_fn = filter_e2_binary
    if args.output_tag:
        cfg.output_tag = args.output_tag
    cfg.device = args.device

    seed_all(cfg.seed)
    labels = LabelTable(LABEL_XLSX)
    samples = discover_samples(
        repaired_dir=args.repaired_dir,
        node_edge_dir=args.node_edge_dir,
        labels=labels,
    )
    picked_raw = filter_fn(samples)

    if args.experiment == "e1":
        picked = picked_raw
        label_override = [s.label for s in picked]
    else:
        picked = picked_raw
        # E2 二分类：label 1 (NSTEMI) -> 0, label 2 (non-NSTEMI) -> 1
        label_override = [0 if s.label == 1 else 1 for s in picked]

    LOG.info("cohort: %d samples", len(picked))

    results_dir = RESULTS_ROOT / cfg.output_tag
    if not results_dir.is_dir():
        LOG.error("Results dir missing: %s", results_dir)
        return 1
    fold_ckpts = _collect_fold_checkpoints(results_dir)
    if not fold_ckpts:
        LOG.error("No fold checkpoints under %s", results_dir)
        return 1
    LOG.info("Loaded %d fold checkpoints", len(fold_ckpts))

    output_dir = results_dir / "interpret_v2"
    extract_and_save_prototypes(
        samples=picked,
        cfg=cfg,
        fold_checkpoints=fold_ckpts,
        output_dir=output_dir,
        label_override=label_override,
        top_k=args.top_k,
    )

    if not args.no_heatmap:
        _plot_heatmaps(output_dir / "interpret_arrays.npz", output_dir / "heatmaps", cfg)

    LOG.info("Done. Output at %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
