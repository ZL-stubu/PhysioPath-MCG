from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from .configs import GraphConfig
from .utils import SampleMeta


LOG = logging.getLogger("gnn_mcg.dataset")


# Edge type IDs used across the module. Kept centrally so the model's edge
# type embedding table stays in sync with the graph builder.
EDGE_TYPE_SPATIAL = 0
EDGE_TYPE_TEMPORAL = 1
EDGE_TYPE_FORWARD_CROSS = 2
EDGE_TYPE_DYNAMIC = 3
N_EDGE_TYPES = 4


class GraphSampleData(Data):
    """PyG graph whose cohort-level sample index must not be incremented.

    PyG treats every attribute containing ``index`` as a graph-local index and
    offsets it by the number of nodes while batching.  ``sample_index`` is an
    index into the cohort instead, so it must remain unchanged.
    """

    def __inc__(self, key: str, value: torch.Tensor, *args: object, **kwargs: object) -> object:
        if key == "sample_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


@dataclass
class SpatioTemporalGraphTemplate:
    """
    Static skeleton that is the same for every sample within a cohort.

    Layout: node index = phase_id * n_rois + roi_id.
    """

    n_rois: int
    n_phases: int
    roi_positions: torch.Tensor        # [n_rois, 2] in [-1, 1]
    spatial_edge_index: torch.Tensor   # [2, E_s * n_phases]
    temporal_edge_index: torch.Tensor  # [2, E_t]
    forward_cross_edge_index: torch.Tensor  # [2, E_c]
    edge_type_ids: torch.Tensor        # [E_static]

    @property
    def n_nodes(self) -> int:
        return self.n_rois * self.n_phases


def _grid_row_col(roi_side: int, node_id: int) -> tuple[int, int]:
    return node_id // roi_side, node_id % roi_side


def _forward_cross_edges(roi_side: int, n_phases: int, neighborhood: int) -> list[tuple[int, int]]:
    """
    Cross-space-time prior transmission edges.

    Rule: a ROI at (phase p) may reach neighborhood(4 or 8) neighbors at (phase p+1).
    These edges are anatomy-agnostic; the model still learns which of them
    matter via attention. Narrow to 4-adjacency by default to keep the
    candidate path space tractable.
    """
    if neighborhood == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif neighborhood == 8:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        raise ValueError(f"forward_cross_neighborhood must be 4 or 8, got {neighborhood}")

    n_rois = roi_side * roi_side
    edges: list[tuple[int, int]] = []
    for p in range(n_phases - 1):
        for src in range(n_rois):
            src_row, src_col = _grid_row_col(roi_side, src)
            for dr, dc in offsets:
                tr, tc = src_row + dr, src_col + dc
                if not (0 <= tr < roi_side and 0 <= tc < roi_side):
                    continue
                tgt = tr * roi_side + tc
                src_global = p * n_rois + src
                tgt_global = (p + 1) * n_rois + tgt
                edges.append((src_global, tgt_global))
    return edges


def build_template(cfg: GraphConfig, spatial_edge_index_per_phase: np.ndarray) -> SpatioTemporalGraphTemplate:
    """
    spatial_edge_index_per_phase is the [2, E] edge index within one phase,
    reused across every phase.
    """
    n_rois = cfg.n_rois
    n_phases = cfg.n_phases

    # Duplicate spatial edges for each phase, shifting indices by phase offset.
    spatial_edges = []
    edge_types: list[int] = []
    for p in range(n_phases):
        offset = p * n_rois
        shifted = spatial_edge_index_per_phase + offset
        spatial_edges.append(shifted)
        edge_types.extend([EDGE_TYPE_SPATIAL] * shifted.shape[1])
    spatial_edge_index = np.concatenate(spatial_edges, axis=1) if spatial_edges else np.zeros((2, 0), dtype=np.int64)

    # Temporal self-edges: same ROI, phase p -> phase p+1
    temporal_pairs: list[tuple[int, int]] = []
    if cfg.add_temporal_self_edges:
        for p in range(n_phases - 1):
            for r in range(n_rois):
                temporal_pairs.append((p * n_rois + r, (p + 1) * n_rois + r))
    temporal_edge_index = (
        np.asarray(temporal_pairs, dtype=np.int64).T if temporal_pairs else np.zeros((2, 0), dtype=np.int64)
    )
    edge_types.extend([EDGE_TYPE_TEMPORAL] * temporal_edge_index.shape[1])

    # Forward cross edges (spatial neighbor at next phase)
    if cfg.add_forward_cross_edges:
        cross_pairs = _forward_cross_edges(cfg.roi_side, n_phases, cfg.forward_cross_neighborhood)
    else:
        cross_pairs = []
    forward_cross_edge_index = (
        np.asarray(cross_pairs, dtype=np.int64).T if cross_pairs else np.zeros((2, 0), dtype=np.int64)
    )
    edge_types.extend([EDGE_TYPE_FORWARD_CROSS] * forward_cross_edge_index.shape[1])

    # ROI 2D positions (repeated for each phase later at the node-feature stage)
    axis = np.linspace(-1.0 + 1.0 / cfg.roi_side, 1.0 - 1.0 / cfg.roi_side, cfg.roi_side, dtype=np.float64)
    roi_positions = np.asarray([(x, y) for y in axis for x in axis], dtype=np.float32)

    return SpatioTemporalGraphTemplate(
        n_rois=n_rois,
        n_phases=n_phases,
        roi_positions=torch.from_numpy(roi_positions),
        spatial_edge_index=torch.from_numpy(spatial_edge_index),
        temporal_edge_index=torch.from_numpy(temporal_edge_index),
        forward_cross_edge_index=torch.from_numpy(forward_cross_edge_index),
        edge_type_ids=torch.tensor(edge_types, dtype=torch.long),
    )


def _dynamic_edges_from_npz(
    npz: np.lib.npyio.NpzFile,
    cfg: GraphConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic per-phase directed edges saved by node_edge.py.

    Returns:
        edge_index: [2, E_dyn] with global (phase, roi) node indexing
        edge_weight: [E_dyn]
    """
    if not cfg.use_dynamic_edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    n_rois = cfg.n_rois
    edge_index_local = np.asarray(npz["dynamic_edge_index"], dtype=np.int64)  # [2, E]
    active = np.asarray(npz["dynamic_edge_active"], dtype=bool)               # [P, E]
    weight = np.asarray(npz["dynamic_edge_weight"], dtype=np.float32)         # [P, E]
    n_phases = active.shape[0]

    src_local = edge_index_local[0]
    tgt_local = edge_index_local[1]

    pieces_src = []
    pieces_tgt = []
    pieces_w = []
    for p in range(n_phases):
        mask = active[p]
        if not np.any(mask):
            continue
        pieces_src.append(p * n_rois + src_local[mask])
        pieces_tgt.append(p * n_rois + tgt_local[mask])
        pieces_w.append(weight[p, mask] * cfg.dynamic_edge_scale)

    if not pieces_src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.stack([np.concatenate(pieces_src), np.concatenate(pieces_tgt)], axis=0)
    edge_weight = np.concatenate(pieces_w).astype(np.float32)
    return edge_index, edge_weight


def _build_node_features(
    npz: np.lib.npyio.NpzFile,
    cfg: GraphConfig,
    template: SpatioTemporalGraphTemplate,
) -> torch.Tensor:
    """
    Assemble per-node feature tensor of shape [n_nodes, n_input_features, phase_length].
    Also emits static per-node scalars concatenated as extra channels via broadcasting.
    """
    # node_feature_blocks in NPZ: [N_roi, P, L, F]
    feature_blocks = np.asarray(npz["node_feature_blocks"], dtype=np.float32)  # [N, P, L, F]
    n_rois, n_phases, phase_length, n_features = feature_blocks.shape
    if n_rois != template.n_rois or n_phases != template.n_phases:
        raise ValueError(
            f"NPZ node grid ({n_rois}x{n_phases}) does not match config ({template.n_rois}x{template.n_phases})"
        )
    if phase_length != cfg.phase_length:
        LOG.warning("Phase length %d != cfg.phase_length %d — using NPZ value.", phase_length, cfg.phase_length)

    # Reorder to [n_nodes, n_features, L] with node_id = phase * n_rois + roi
    x = np.transpose(feature_blocks, (1, 0, 3, 2))    # [P, N, F, L]
    x = x.reshape(n_phases * n_rois, n_features, phase_length)
    return torch.from_numpy(x).float()


def _phase_scalar_features(
    npz: np.lib.npyio.NpzFile,
    template: SpatioTemporalGraphTemplate,
) -> torch.Tensor:
    """
    Per-node scalar side features: phase one-hot, phase relative duration,
    ROI position (2D), and a phase progress marker (0.5 mid-phase constant).
    """
    n_phases = template.n_phases
    n_rois = template.n_rois

    phase_duration = np.asarray(npz["phase_duration_ms"], dtype=np.float32)    # [P]
    total = float(phase_duration.sum()) if phase_duration.sum() > 0 else 1.0
    rel_duration = phase_duration / total                                     # [P]

    scalars = []
    for p in range(n_phases):
        onehot = np.zeros(n_phases, dtype=np.float32)
        onehot[p] = 1.0
        for r in range(n_rois):
            roi_pos = template.roi_positions[r].numpy()
            scalars.append(
                np.concatenate([onehot, [rel_duration[p]], roi_pos])
            )
    return torch.from_numpy(np.stack(scalars, axis=0)).float()


class NodeEdgeGraphDataset(Dataset):
    """
    Wraps NodeEdge NPZ outputs from node_edge.py as PyG Data objects.

    Each sample yields:
        Data(x=[n_nodes, n_input_features, L], scalar=[n_nodes, F_scalar],
             edge_index=[2, E], edge_type=[E], edge_weight=[E],
             y=int64, meta=SampleMeta)
    """

    def __init__(
        self,
        samples: Sequence[SampleMeta],
        cfg: GraphConfig,
        label_override: Sequence[int] | None = None,
    ):
        self.samples = list(samples)
        self.cfg = cfg
        self.label_override = list(label_override) if label_override is not None else None
        if self.label_override is not None and len(self.label_override) != len(self.samples):
            raise ValueError("label_override length must match samples length")

        # Peek at the first sample to build the static graph template.
        template_source = self.samples[0].npz_path
        with np.load(template_source, allow_pickle=False) as npz:
            static_edge_index = np.asarray(npz["edge_index"], dtype=np.int64)
        self.template = build_template(cfg, static_edge_index)

    def __len__(self) -> int:
        return len(self.samples)

    def _label_for(self, i: int) -> int:
        if self.label_override is not None:
            return int(self.label_override[i])
        # Default: healthy=0, NSTEMI=1. Callers doing E2 should pass label_override.
        raw = self.samples[i].label
        if raw == 0:
            return 0
        if raw == 1:
            return 1
        return raw  # keep raw for label 2; use filter_e2_* to remap upstream

    def __getitem__(self, i: int) -> Data:
        meta = self.samples[i]
        with np.load(meta.npz_path, allow_pickle=False) as npz:
            node_feats = _build_node_features(npz, self.cfg, self.template)
            scalar = _phase_scalar_features(npz, self.template)
            dyn_index, dyn_weight = _dynamic_edges_from_npz(npz, self.cfg)

        # Compose full edge set: spatial + temporal + forward cross + dynamic
        static_index = torch.cat(
            [
                self.template.spatial_edge_index,
                self.template.temporal_edge_index,
                self.template.forward_cross_edge_index,
            ],
            dim=1,
        )
        static_types = self.template.edge_type_ids
        static_weights = torch.ones(static_index.shape[1], dtype=torch.float32)

        if dyn_index.shape[1] > 0:
            dyn_index_t = torch.from_numpy(dyn_index)
            dyn_types = torch.full(
                (dyn_index.shape[1],),
                EDGE_TYPE_DYNAMIC,
                dtype=torch.long,
            )
            dyn_weights_t = torch.from_numpy(dyn_weight)
            edge_index = torch.cat([static_index, dyn_index_t], dim=1)
            edge_type = torch.cat([static_types, dyn_types], dim=0)
            edge_weight = torch.cat([static_weights, dyn_weights_t], dim=0)
        else:
            edge_index = static_index
            edge_type = static_types
            edge_weight = static_weights

        data = GraphSampleData(
            x=node_feats,          # [n_nodes, F_in, L]
            scalar=scalar,         # [n_nodes, F_scalar]
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            y=torch.tensor([self._label_for(i)], dtype=torch.long),
        )
        data.sample_index = torch.tensor([i], dtype=torch.long)
        return data
