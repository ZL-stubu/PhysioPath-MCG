"""
E2 方案 A：NSTEMI vs STEMI（64 通道 cohort）。

三种模式：
    --mode primary       二分类 NSTEMI vs 非 NSTEMI 混合
    --mode multiclass    NSTEMI + 指定 subcats + 其他 的多分类
    --mode nstemi-stemi  头对头 NSTEMI vs STEMI（默认，方案 A）

需要 E1 训练完并复制了 best.pt 才能做迁移。
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
    make_e2_binary_config,
    make_e2_multiclass_config,
)
from gnn_mcg.interpret import extract_and_save_prototypes
from gnn_mcg.train import train_cross_validation
from gnn_mcg.utils import (
    LabelTable,
    SampleMeta,
    discover_samples,
    filter_e2_binary,
    filter_e2_multiclass,
    filter_e2_nstemi_stemi,
    seed_all,
)


def _prepare_primary(samples: list[SampleMeta]):
    picked = filter_e2_binary(samples)
    label_override = [0 if s.label == 1 else 1 for s in picked]  # NSTEMI=0, non=1
    counts = {
        "NSTEMI": sum(1 for label in label_override if label == 0),
        "non-NSTEMI": sum(1 for label in label_override if label == 1),
    }
    logging.info("E2 primary cohort: %s", counts)
    return picked, label_override


def _prepare_head_to_head(samples: list[SampleMeta]):
    picked_pairs = filter_e2_nstemi_stemi(samples)
    picked = [p[0] for p in picked_pairs]
    label_override = [p[1] for p in picked_pairs]
    counts = {
        "NSTEMI": sum(1 for label in label_override if label == 0),
        "STEMI": sum(1 for label in label_override if label == 1),
    }
    logging.info("E2 NSTEMI vs STEMI cohort: %s", counts)
    return picked, label_override


def _prepare_multiclass(samples: list[SampleMeta], subcats: tuple[str, ...]):
    entries = filter_e2_multiclass(samples, subcats)
    picked = [e[0] for e in entries]
    label_override = [e[1] for e in entries]
    logging.info(
        "E2 multiclass cohort: subcats=%s, class counts=%s",
        subcats,
        np.bincount(label_override).tolist(),
    )
    return picked, label_override


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("primary", "multiclass", "nstemi-stemi"),
        default="nstemi-stemi",
    )
    parser.add_argument(
        "--subcats",
        nargs="+",
        default=["STEMI"],
        help="multiclass 模式下作为独立类别的 subcats",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=CHECKPOINT_ROOT / "e1_health_vs_nstemi" / "best.pt",
        help="E1 best.pt 路径；不存在则从头训练（不推荐）",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--repaired-dir", type=Path, default=REPAIRED_DIR)
    parser.add_argument("--node-edge-dir", type=Path, default=NODE_EDGE_DATA_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    labels = LabelTable(LABEL_XLSX)
    samples = discover_samples(
        repaired_dir=args.repaired_dir,
        node_edge_dir=args.node_edge_dir,
        labels=labels,
    )

    pretrained = args.pretrained if args.pretrained and args.pretrained.is_file() else None
    if pretrained is None:
        logging.warning(
            "未找到预训练权重 %s，将从零训练 E2；强烈建议先跑 run_e1.py。",
            args.pretrained,
        )

    if args.mode == "primary":
        cfg = make_e2_binary_config(pretrained)
        picked, label_override = _prepare_primary(samples)
    elif args.mode == "nstemi-stemi":
        cfg = make_e2_binary_config(pretrained)
        cfg.output_tag = "e2_nstemi_vs_stemi"
        picked, label_override = _prepare_head_to_head(samples)
    else:
        subcats = tuple(args.subcats)
        cfg = make_e2_multiclass_config(pretrained, subcats)
        picked, label_override = _prepare_multiclass(samples, subcats)

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

    if not picked:
        logging.error(
            "E2 无可用样本。请先补跑 mcg_bad_channel_pipeline_revised.py 处理 noNSTEMI，"
            "再跑 node_edge.py 生成节点边 NPZ。"
        )
        return 1

    output_dir = RESULTS_ROOT / cfg.output_tag
    summary = train_cross_validation(
        samples=picked,
        labels_for_weight=np.asarray(label_override, dtype=np.int64),
        cfg=cfg,
        output_dir=output_dir,
        label_override=label_override,
    )

    fold_ckpts = [
        Path(r["checkpoint"])
        for r in summary["fold_reports"]
        if Path(r["checkpoint"]).is_file()
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

    logging.info("E2 (%s) done. Results in %s", args.mode, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
