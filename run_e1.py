"""
E1 主实验：健康 vs NSTEMI（36 通道 cohort）。

先跑 node_edge.py 生成 NodeEdge NPZ，然后执行：
    python run_e1.py
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
)
from gnn_mcg.interpret import extract_and_save_prototypes
from gnn_mcg.train import train_cross_validation
from gnn_mcg.utils import (
    LabelTable,
    discover_samples,
    filter_e1,
    seed_all,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument(
        "--repaired-dir",
        type=Path,
        default=REPAIRED_DIR,
        help="修复后 NPZ 文件夹（node_edge.py 的输入目录）",
    )
    parser.add_argument(
        "--node-edge-dir",
        type=Path,
        default=NODE_EDGE_DATA_DIR,
        help="node_edge.py 输出的节点边 NPZ 目录",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = make_e1_config()
    cfg.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.folds is not None:
        cfg.train.n_folds = args.folds
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.output_tag is not None:
        cfg.output_tag = args.output_tag

    seed_all(cfg.seed)

    labels = LabelTable(LABEL_XLSX)
    samples = discover_samples(
        repaired_dir=args.repaired_dir,
        node_edge_dir=args.node_edge_dir,
        labels=labels,
    )
    picked = filter_e1(samples)
    if not picked:
        logging.error(
            "E1 无可用样本。请先运行 node_edge.py 生成节点边 NPZ 后再试。"
        )
        return 1

    logging.info(
        "E1 cohort: %d samples (health=%d, NSTEMI=%d)",
        len(picked),
        sum(1 for s in picked if s.label == 0),
        sum(1 for s in picked if s.label == 1),
    )

    label_override = [s.label for s in picked]  # 0/1 directly
    output_dir = RESULTS_ROOT / cfg.output_tag
    summary = train_cross_validation(
        samples=picked,
        labels_for_weight=np.asarray(label_override, dtype=np.int64),
        cfg=cfg,
        output_dir=output_dir,
        label_override=label_override,
    )

    # Path prototype extraction across folds
    fold_ckpts = [
        Path(report["checkpoint"])
        for report in summary["fold_reports"]
        if Path(report["checkpoint"]).is_file()
    ]
    if fold_ckpts:
        extract_and_save_prototypes(
            samples=picked,
            cfg=cfg,
            fold_checkpoints=fold_ckpts,
            output_dir=output_dir / "interpret",
            label_override=label_override,
            top_k=10,
        )

    # Copy best fold checkpoint into a shared location for E2 transfer.
    if fold_ckpts:
        best_fold = max(summary["fold_reports"], key=lambda r: r["best_metric"])
        transfer_dir = CHECKPOINT_ROOT / cfg.output_tag
        transfer_dir.mkdir(parents=True, exist_ok=True)
        transfer_path = transfer_dir / "best.pt"
        import shutil
        shutil.copyfile(best_fold["checkpoint"], transfer_path)
        logging.info(
            "Best fold %d checkpoint copied to %s (best metric=%.4f).",
            best_fold["fold"], transfer_path, best_fold["best_metric"],
        )

    logging.info("E1 done. Results in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
