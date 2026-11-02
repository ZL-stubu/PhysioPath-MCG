from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(r"D:\pycharmproject\GCNMCG")
DATA_ROOT = PROJECT_ROOT / "data"
LABEL_XLSX = PROJECT_ROOT / "all_data_labels.xlsx"

REPAIRED_DIR = DATA_ROOT / "坏道处理后数据" / "修复后数据"
NODE_EDGE_ROOT = DATA_ROOT / "node_edge"
NODE_EDGE_DATA_DIR = NODE_EDGE_ROOT / "节点边数据"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Label conventions defined in all_data_labels.xlsx
LABEL_HEALTH = 0
LABEL_NSTEMI = 1
LABEL_NON_NSTEMI = 2

# Non-NSTEMI subcategory tags parsed from filename (noNSTEMI__<SUBCAT>__...)
NON_NSTEMI_SUBCATS = (
    "STEMI",
    "左室心肌肥厚",
    "心律失常",
    "心肌病",
    "结构性心脏病",
)


@dataclass
class GraphConfig:
    """4x4 ROI grid x 5 physiological phases from node_edge.py output."""

    roi_side: int = 4
    n_phases: int = 5
    phase_length: int = 128  # samples per phase, matches node_edge default

    # Extra edge types layered on top of the static/dynamic edges stored in NPZ.
    add_temporal_self_edges: bool = True
    add_forward_cross_edges: bool = True
    # 4 = axial neighbors only, 8 = diagonals too. 4 keeps candidate paths small.
    forward_cross_neighborhood: int = 4
    use_dynamic_edges: bool = True
    dynamic_edge_scale: float = 1.0

    @property
    def n_rois(self) -> int:
        return self.roi_side * self.roi_side

    @property
    def n_nodes(self) -> int:
        return self.n_rois * self.n_phases


@dataclass
class PathConfig:
    """Anatomical / physiological path enumeration budget."""

    min_length: int = 3     # hops (edges), so nodes = length + 1
    max_length: int = 4
    max_paths: int = 300
    # Paths must be temporally non-decreasing (phase index never goes back).
    monotone_time: bool = True
    # Starting phase set — signal propagation is meaningful from QRS onward.
    start_phase_ids: tuple[int, ...] = (0, 1)  # baseline or QRS
    # Sparsity of the path attention head.
    top_k_paths: int = 20
    l1_penalty: float = 0


@dataclass
class ModelConfig:
    node_hidden: int =64
    gat_hidden: int = 32
    gat_heads: int = 4
    gat_layers: int = 2
    gru_hidden: int = 64
    classifier_hidden: int = 64
    dropout: float = 0.2
    n_input_features: int = 3  # normalized_signal, temporal_derivative, activation_magnitude


@dataclass
class TrainConfig:
    experiment: str = "E1"  # E1 or E2
    n_classes: int = 2      # E1 binary healthy vs NSTEMI
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_folds: int = 5
    cv_seed: int = 42
    early_stopping_patience: int = 20
    stratify_by_patient: bool = True

    # E2-only: staged transfer settings
    pretrained_checkpoint: Path | None = None
    freeze_encoder_epochs: int = 10
    backbone_lr_scale: float = 0.1
    focal_gamma: float = 0.0  # >0 activates focal loss
    class_weights: tuple[float, ...] | None = None  # auto if None


@dataclass
class ExperimentConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    output_tag: str = "e1_baseline"


def make_e1_config() -> ExperimentConfig:
    """Stage 1: health (0) vs NSTEMI (1)."""
    cfg = ExperimentConfig()
    cfg.train.experiment = "E1"
    cfg.train.n_classes = 2
    cfg.train.epochs = 100
    cfg.output_tag = "e1_health_vs_nstemi"
    return cfg


def make_e2_binary_config(pretrained: Path | None) -> ExperimentConfig:
    """Stage 2 primary: NSTEMI (1) vs non-NSTEMI (2)."""
    cfg = ExperimentConfig()
    cfg.train.experiment = "E2"
    cfg.train.n_classes = 2
    cfg.train.epochs = 100
    cfg.train.n_folds = 5  # small sample size
    cfg.train.pretrained_checkpoint = pretrained
    cfg.output_tag = "e2_nstemi_vs_nonnstemi"
    return cfg


def make_e2_multiclass_config(
    pretrained: Path | None,
    include_subcats: tuple[str, ...] = ("STEMI",),
) -> ExperimentConfig:
    """Stage 2 secondary: NSTEMI vs each subcategory with enough samples."""
    cfg = ExperimentConfig()
    cfg.train.experiment = "E2_multiclass"
    cfg.train.n_classes = 1 + len(include_subcats) + 1  # NSTEMI + subcats + rest
    cfg.train.epochs = 80
    cfg.train.n_folds = 5
    cfg.train.pretrained_checkpoint = pretrained
    cfg.train.focal_gamma = 1.5
    cfg.output_tag = "e2_multiclass_" + "_".join(include_subcats)
    return cfg
