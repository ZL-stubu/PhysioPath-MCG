from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

from .configs import ModelConfig, PathConfig
from .dataset import (
    N_EDGE_TYPES,
    SpatioTemporalGraphTemplate,
    EDGE_TYPE_SPATIAL,
    EDGE_TYPE_TEMPORAL,
    EDGE_TYPE_FORWARD_CROSS,
)


LOG = logging.getLogger("gnn_mcg.model")


# ---------------------------------------------------------------------------
# Path enumeration


@dataclass
class PathCandidate:
    node_ids: tuple[int, ...]      # length L (nodes), L-1 edges
    phase_ids: tuple[int, ...]     # per-node phase id
    roi_ids: tuple[int, ...]       # per-node roi id
    edge_types: tuple[int, ...]    # length L-1

    def description(self, roi_labels: Sequence[str], phase_labels: Sequence[str]) -> str:
        parts = [
            f"{phase_labels[ph]}:{roi_labels[r]}"
            for ph, r in zip(self.phase_ids, self.roi_ids)
        ]
        return " -> ".join(parts)


def _adjacency_from_template(
    template: SpatioTemporalGraphTemplate,
) -> dict[int, list[tuple[int, int]]]:
    """
    Build directed adjacency (src -> [(dst, edge_type), ...]) restricted to
    the STATIC skeleton (spatial + temporal + forward-cross). Dynamic edges
    are per-sample and would make candidate paths non-deterministic, so we
    intentionally exclude them from path enumeration and only let the model
    weight them at the GAT stage.
    """
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for source, target, etype in zip(
        template.spatial_edge_index[0].tolist(),
        template.spatial_edge_index[1].tolist(),
        [EDGE_TYPE_SPATIAL] * template.spatial_edge_index.shape[1],
    ):
        adjacency.setdefault(source, []).append((target, etype))
    for source, target in zip(
        template.temporal_edge_index[0].tolist(),
        template.temporal_edge_index[1].tolist(),
    ):
        adjacency.setdefault(source, []).append((target, EDGE_TYPE_TEMPORAL))
    for source, target in zip(
        template.forward_cross_edge_index[0].tolist(),
        template.forward_cross_edge_index[1].tolist(),
    ):
        adjacency.setdefault(source, []).append((target, EDGE_TYPE_FORWARD_CROSS))
    return adjacency


def enumerate_candidate_paths(
    template: SpatioTemporalGraphTemplate,
    cfg: PathConfig,
) -> list[PathCandidate]:
    """
    DFS enumeration of temporally non-decreasing paths on the static skeleton.

    The result is deterministic and independent of any single sample, so it
    can be reused across the whole cohort.
    """
    n_rois = template.n_rois
    adjacency = _adjacency_from_template(template)

    def node_to_phase(node: int) -> int:
        return node // n_rois

    def node_to_roi(node: int) -> int:
        return node % n_rois

    start_nodes: list[int] = []
    for phase in cfg.start_phase_ids:
        for r in range(n_rois):
            start_nodes.append(phase * n_rois + r)

    paths: list[PathCandidate] = []

    def dfs(current: int, node_stack: list[int], edge_stack: list[int]) -> None:
        # Emit path if within [min_length, max_length] hops.
        n_hops = len(node_stack) - 1
        if n_hops >= cfg.min_length:
            phase_ids = tuple(node_to_phase(n) for n in node_stack)
            roi_ids = tuple(node_to_roi(n) for n in node_stack)
            paths.append(
                PathCandidate(
                    node_ids=tuple(node_stack),
                    phase_ids=phase_ids,
                    roi_ids=roi_ids,
                    edge_types=tuple(edge_stack),
                )
            )
        if n_hops >= cfg.max_length:
            return

        for target, etype in adjacency.get(current, []):
            if cfg.monotone_time and node_to_phase(target) < node_to_phase(current):
                continue
            if target in node_stack:
                continue  # no revisits
            node_stack.append(target)
            edge_stack.append(etype)
            dfs(target, node_stack, edge_stack)
            node_stack.pop()
            edge_stack.pop()

    for start in start_nodes:
        dfs(start, [start], [])

    # Cap the number of paths — long dense graphs can blow up. To preserve
    # diversity we bucket by (start_phase, start_roi) and round-robin one
    # shortest path from each bucket until the budget is filled.
    if len(paths) > cfg.max_paths:
        buckets: dict[tuple[int, int], list[PathCandidate]] = {}
        for path in paths:
            key = (path.phase_ids[0], path.roi_ids[0])
            buckets.setdefault(key, []).append(path)
        for bucket in buckets.values():
            bucket.sort(key=lambda p: (len(p.node_ids), p.node_ids))
        keys = sorted(buckets.keys())
        picked: list[PathCandidate] = []
        cursor = {key: 0 for key in keys}
        while len(picked) < cfg.max_paths:
            progressed = False
            for key in keys:
                idx = cursor[key]
                if idx < len(buckets[key]):
                    picked.append(buckets[key][idx])
                    cursor[key] = idx + 1
                    progressed = True
                    if len(picked) >= cfg.max_paths:
                        break
            if not progressed:
                break
        paths = picked

    LOG.info("Enumerated %d candidate paths (min=%d max=%d hops).",
             len(paths), cfg.min_length, cfg.max_length)
    return paths


def paths_to_tensor(
    paths: list[PathCandidate],
    n_nodes: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack variable-length paths into fixed [n_paths, max_len] tensors.

    Returns:
        node_ids: [n_paths, max_len] with padding value = n_nodes (out-of-range)
        mask: [n_paths, max_len] bool, True where the position is a real node
    """
    n_paths = len(paths)
    node_tensor = np.full((n_paths, max_len), fill_value=n_nodes, dtype=np.int64)
    mask = np.zeros((n_paths, max_len), dtype=bool)
    for i, path in enumerate(paths):
        length = min(len(path.node_ids), max_len)
        node_tensor[i, :length] = path.node_ids[:length]
        mask[i, :length] = True
    return torch.from_numpy(node_tensor), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Node signal encoder


class NodeSignalEncoder(nn.Module):
    """
    1D-CNN over per-node time series, followed by adaptive pooling.

    Input: [n_nodes, F_in, L]
    Output: [n_nodes, D]
    """

    def __init__(self, in_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_nodes, F_in, L]
        out = self.net(x).squeeze(-1)  # [n_nodes, hidden]
        return out


# ---------------------------------------------------------------------------
# Path attention head


def _sparse_softmax_l1(
    logits: torch.Tensor,
    l1_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Softmax with an explicit L1 penalty returned as an auxiliary loss.

    Returning penalty as tensor keeps the main forward pure and lets the
    trainer weight it against classification loss.
    """
    weights = F.softmax(logits, dim=-1)              # [B, n_paths]
    penalty = l1_penalty * weights.abs().sum(dim=-1).mean()
    return weights, penalty


class PathAttention(nn.Module):
    """
    Encode each candidate path with a GRU over its node embeddings, then
    score paths with a learned attention head.

    Output:
        pooled: [B, D_path]
        path_weights: [B, n_paths]
        aux_loss: scalar (L1 sparsity penalty)
    """

    def __init__(
        self,
        node_dim: int,
        gru_hidden: int,
        dropout: float,
        l1_penalty: float,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=node_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.score_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.l1_penalty = l1_penalty

    def forward(
        self,
        node_embeddings: torch.Tensor,   # [B, n_nodes+1, D]  (last row is padding)
        path_node_ids: torch.Tensor,     # [n_paths, max_len]
        path_mask: torch.Tensor,         # [n_paths, max_len]
    ) -> dict[str, torch.Tensor]:
        batch_size = node_embeddings.shape[0]
        n_paths, max_len = path_node_ids.shape

        # Gather node embeddings along each path for each graph in the batch.
        # Result: [B, n_paths, max_len, D]
        expand_idx = path_node_ids.unsqueeze(0).expand(batch_size, -1, -1)
        expand_idx = expand_idx.unsqueeze(-1).expand(-1, -1, -1, node_embeddings.shape[-1])
        gathered = torch.gather(
            node_embeddings.unsqueeze(1).expand(-1, n_paths, -1, -1),
            dim=2,
            index=expand_idx,
        )  # [B, n_paths, max_len, D]

        mask_exp = path_mask.unsqueeze(0).unsqueeze(-1).to(gathered.dtype)  # [1, n_paths, max_len, 1]
        gathered = gathered * mask_exp

        # Per-position node energy along each path, then aggregate.
        # node_energy [B, n_paths, max_len] = ||node_embed_at_pos||_2
        node_energy = gathered.norm(dim=-1)
        node_energy = node_energy * path_mask.unsqueeze(0).to(node_energy.dtype)
        path_len = path_mask.sum(dim=-1).clamp(min=1).to(node_energy.dtype)  # [n_paths]
        # Mean and total energy along the path (before RNN encoding).
        path_energy_sum = node_energy.sum(dim=-1)                            # [B, n_paths]
        path_energy_mean = path_energy_sum / path_len.unsqueeze(0)           # [B, n_paths]

        flat = gathered.reshape(batch_size * n_paths, max_len, -1)
        lengths = path_mask.sum(dim=-1).clamp(min=1).to(torch.long)
        lengths_batch = lengths.repeat(batch_size)

        packed = nn.utils.rnn.pack_padded_sequence(
            flat, lengths_batch.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        path_embeddings = hidden.squeeze(0)
        path_embeddings = self.dropout(path_embeddings)

        scores = self.score_head(path_embeddings).squeeze(-1)                # [B*n_paths]
        scores = scores.view(batch_size, n_paths)
        weights, penalty = _sparse_softmax_l1(scores, self.l1_penalty)

        path_embeddings = path_embeddings.view(batch_size, n_paths, -1)      # [B, n_paths, D]
        # Encoded path strength AFTER GRU (comparable across paths within a sample).
        path_gru_norm = path_embeddings.norm(dim=-1)                         # [B, n_paths]

        pooled = torch.einsum("bpd,bp->bd", path_embeddings, weights)
        return {
            "pooled": pooled,
            "weights": weights,
            "penalty": penalty,
            "scores": scores,
            "path_energy_sum": path_energy_sum,
            "path_energy_mean": path_energy_mean,
            "path_gru_norm": path_gru_norm,
            "node_energy": node_energy,
        }


# ---------------------------------------------------------------------------
# Full model


@dataclass
class PathAttentiveGATState:
    template: SpatioTemporalGraphTemplate
    path_candidates: list[PathCandidate]
    path_node_ids: torch.Tensor      # [n_paths, max_len]
    path_mask: torch.Tensor          # [n_paths, max_len]
    n_scalar_features: int


class PathAttentiveGAT(nn.Module):
    """
    Node signal encoder -> GATv2 (with edge-type embedding) -> path attention
    -> classifier. The static graph template and candidate paths are shared
    across the batch and stored as buffers.
    """

    def __init__(
        self,
        state: PathAttentiveGATState,
        model_cfg: ModelConfig,
        n_classes: int,
    ):
        super().__init__()
        self.state = state
        self.model_cfg = model_cfg
        self.n_classes = n_classes

        n_nodes = state.template.n_nodes
        self.n_nodes = n_nodes

        self.signal_encoder = NodeSignalEncoder(
            in_features=model_cfg.n_input_features,
            hidden=model_cfg.node_hidden,
            dropout=model_cfg.dropout,
        )

        node_in_dim = model_cfg.node_hidden + state.n_scalar_features
        self.edge_type_emb = nn.Embedding(N_EDGE_TYPES, 8)

        conv_in = node_in_dim
        self.gat_layers = nn.ModuleList()
        for i in range(model_cfg.gat_layers):
            out_dim = model_cfg.gat_hidden
            concat = i < model_cfg.gat_layers - 1
            heads = model_cfg.gat_heads if concat else 1
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=conv_in,
                    out_channels=out_dim,
                    heads=heads,
                    concat=concat,
                    dropout=model_cfg.dropout,
                    edge_dim=8,
                    add_self_loops=False,
                )
            )
            conv_in = out_dim * heads if concat else out_dim
        gat_out_dim = conv_in

        self.path_attention = PathAttention(
            node_dim=gat_out_dim,
            gru_hidden=model_cfg.gru_hidden,
            dropout=model_cfg.dropout,
            l1_penalty=0.0,  # penalty applied at training time via aux loss
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_cfg.gru_hidden, model_cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.classifier_hidden, n_classes),
        )

        # Register precomputed path tensors as buffers so they follow .to(device).
        self.register_buffer("_path_node_ids", state.path_node_ids)
        self.register_buffer("_path_mask", state.path_mask)

    def set_l1_penalty(self, penalty: float) -> None:
        self.path_attention.l1_penalty = penalty

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B * n_nodes, F_in, L]
        return self.signal_encoder(x)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        """
        batch is a PyG Batch containing:
            x: [B * n_nodes, F_in, L]
            scalar: [B * n_nodes, F_scalar]
            edge_index / edge_type / edge_weight: PyG-batched
            y: [B]
        """
        x = batch.x
        scalar = batch.scalar
        edge_index = batch.edge_index
        edge_type = batch.edge_type

        node_embed = self._encode_nodes(x)                       # [B*n_nodes, node_hidden]
        node_embed = torch.cat([node_embed, scalar], dim=-1)     # concat scalars

        edge_attr = self.edge_type_emb(edge_type)                # [E, 8]

        h = node_embed
        for i, conv in enumerate(self.gat_layers):
            h = conv(h, edge_index, edge_attr=edge_attr)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.model_cfg.dropout, training=self.training)

        # Reshape [B*n_nodes, D] -> [B, n_nodes, D], and append a padding row
        # of zeros so path indices equal to n_nodes safely map to zeros.
        batch_size = int(batch.num_graphs)
        node_dim = h.shape[-1]
        h = h.view(batch_size, self.n_nodes, node_dim)
        pad = torch.zeros(batch_size, 1, node_dim, dtype=h.dtype, device=h.device)
        h_padded = torch.cat([h, pad], dim=1)                    # [B, n_nodes+1, D]

        attn_out = self.path_attention(
            node_embeddings=h_padded,
            path_node_ids=self._path_node_ids,
            path_mask=self._path_mask,
        )

        pooled, path_weights, aux_loss = (
            attn_out["pooled"],
            attn_out["weights"],
            attn_out["penalty"],
        )

        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "path_weights": path_weights,
            "aux_loss": aux_loss,
            "path_scores": attn_out["scores"],
            "path_energy_sum": attn_out["path_energy_sum"],
            "path_energy_mean": attn_out["path_energy_mean"],
            "path_gru_norm": attn_out["path_gru_norm"],
            "node_energy": attn_out["node_energy"],
            "node_embeddings": h,
        }


def build_model(
    template: SpatioTemporalGraphTemplate,
    n_scalar_features: int,
    path_cfg: PathConfig,
    model_cfg: ModelConfig,
    n_classes: int,
) -> PathAttentiveGAT:
    paths = enumerate_candidate_paths(template, path_cfg)
    max_len = path_cfg.max_length + 1
    path_node_ids, path_mask = paths_to_tensor(paths, template.n_nodes, max_len)
    state = PathAttentiveGATState(
        template=template,
        path_candidates=paths,
        path_node_ids=path_node_ids,
        path_mask=path_mask,
        n_scalar_features=n_scalar_features,
    )
    return PathAttentiveGAT(state, model_cfg, n_classes)
