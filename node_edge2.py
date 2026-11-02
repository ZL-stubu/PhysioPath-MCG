"""Generate five-phase NodeEdge data directly from 36 physical MCG channels.

Unlike ``node_edge.py``, this pipeline performs no ROI projection. Every
physical channel remains one spatial node, so each sample contains:

    36 channels x 5 physiological phases = 180 graph nodes.

Static channel adjacency is derived from normalized physical sensor
coordinates with an undirected k-nearest-neighbor graph. Temporal processing,
landmark resolution, normalization, phase resampling, node features, and
patient-specific dynamic-edge estimation reuse the original implementation.

Default output:
    D:\\pycharmproject\\GCNMCG\\data\\node_edge2

By default, processing writes both node/edge NPZ files and node time-segment
PNG summaries. Use ``--no-plots`` to skip image rendering.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from node_edge import (
    DEFAULT_INPUT_DIR,
    LANDMARK_NAMES,
    NODE_FEATURE_NAMES,
    PHASE_NAMES,
    STATIC_EDGE_ATTRIBUTE_NAMES,
    build_phase_bounds,
    estimate_phase_dynamic_edges,
    load_landmark_csv,
    load_repaired_npz,
    natural_key,
    normalize_coordinates,
    normalize_roi_signal,
    resample_phase_blocks,
    resolve_landmarks,
    safe_output_stem,
    save_phase_node_plot,
)


DEFAULT_OUTPUT_DIR = Path(
    r"D:\pycharmproject\GCNMCG\data\203\node_edge2"
)
NODE_DATA_DIR_NAME = "节点边数据"
NODE_PLOT_DIR_NAME = "节点时间段图片"
EXPECTED_CHANNELS = 36
REDUCED_SUMMARY_STAT_NAMES = ("mean", "std", "rms")


@dataclass(frozen=True)
class ChannelNodeEdgeOptions:
    """Time segmentation and physical-channel graph parameters."""

    sampling_rate_hz: float = 1000.0
    phase_length: int = 128
    spatial_k: int = 8
    dynamic_max_lag_ms: float = 20.0
    dynamic_corr_threshold: float = 0.25
    dynamic_direction_margin: float = 0.05
    dynamic_top_k_incoming: int = 2
    enable_auto_landmarks: bool = True
    save_plots: bool = True


def build_channel_graph_edges(
    coordinates_norm: np.ndarray,
    spatial_k: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a bidirectional coordinate k-nearest-neighbor channel graph."""
    coordinates_norm = np.asarray(
        coordinates_norm,
        dtype=np.float64,
    )
    if coordinates_norm.shape != (EXPECTED_CHANNELS, 2):
        raise ValueError(
            "Channel coordinates must have shape "
            f"{EXPECTED_CHANNELS}x2, got {coordinates_norm.shape}"
        )
    if not 1 <= spatial_k < EXPECTED_CHANNELS:
        raise ValueError(
            f"spatial_k must be in [1, {EXPECTED_CHANNELS - 1}]"
        )

    delta = (
        coordinates_norm[:, None, :]
        - coordinates_norm[None, :, :]
    )
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)

    undirected: set[tuple[int, int]] = set()
    for source in range(EXPECTED_CHANNELS):
        neighbors = np.argsort(
            distances[source],
            kind="stable",
        )[:spatial_k]
        for target in neighbors:
            left, right = sorted((source, int(target)))
            if left != right:
                undirected.add((left, right))

    directed: list[tuple[int, int]] = []
    for source, target in sorted(undirected):
        directed.append((source, target))
        directed.append((target, source))
    if not directed:
        raise RuntimeError("No physical channel edges were generated")

    edge_index = np.asarray(
        directed,
        dtype=np.int64,
    ).T
    edge_weight = np.ones(
        edge_index.shape[1],
        dtype=np.float32,
    )
    edge_distance = np.linalg.norm(
        coordinates_norm[edge_index[0]]
        - coordinates_norm[edge_index[1]],
        axis=1,
    ).astype(np.float32)
    adjacency = np.zeros(
        (EXPECTED_CHANNELS, EXPECTED_CHANNELS),
        dtype=np.float32,
    )
    adjacency[
        edge_index[0],
        edge_index[1],
    ] = edge_weight
    return edge_index, edge_weight, edge_distance, adjacency


def _distance_prior(
    edge_distance: np.ndarray,
) -> tuple[np.ndarray, float]:
    positive = edge_distance[edge_distance > 0.0]
    sigma = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    sigma = max(sigma, np.finfo(np.float32).eps)
    prior = np.exp(
        -(edge_distance.astype(np.float64) ** 2)
        / (2.0 * sigma * sigma)
    ).astype(np.float32)
    return prior, sigma


def _safe_channel_labels(n_channels: int) -> np.ndarray:
    return np.asarray(
        [f"CH{index + 1:02d}" for index in range(n_channels)],
        dtype=str,
    )


def build_reduced_handcrafted_features(
    node_feature_blocks: np.ndarray,
    phase_duration_ms: np.ndarray,
    landmarks_ms: np.ndarray,
    dynamic_active_edge_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact channel-level handcrafted vector.

    Each channel-phase-signal block contributes mean, standard deviation,
    and RMS only. With 36 channels, five phases, and three signal channels,
    this yields 1620 local values plus 16 sample-global values = 1636.
    """
    blocks = np.asarray(
        node_feature_blocks,
        dtype=np.float32,
    )
    if (
        blocks.ndim != 4
        or blocks.shape[0] != EXPECTED_CHANNELS
        or blocks.shape[1] != len(PHASE_NAMES)
        or blocks.shape[3] != len(NODE_FEATURE_NAMES)
    ):
        raise ValueError(
            "node_feature_blocks must have shape "
            f"[{EXPECTED_CHANNELS}, {len(PHASE_NAMES)}, L, "
            f"{len(NODE_FEATURE_NAMES)}], got {blocks.shape}"
        )

    values = np.transpose(
        blocks,
        (0, 1, 3, 2),
    ).astype(np.float64, copy=False)
    summaries = np.stack(
        (
            np.mean(values, axis=-1),
            np.std(values, axis=-1),
            np.sqrt(
                np.mean(np.square(values), axis=-1)
            ),
        ),
        axis=-1,
    )
    local_features = summaries.reshape(-1)
    phase_duration_ms = np.asarray(
        phase_duration_ms,
        dtype=np.float64,
    ).reshape(-1)
    landmarks_ms = np.asarray(
        landmarks_ms,
        dtype=np.float64,
    ).reshape(-1)
    dynamic_active_edge_counts = np.asarray(
        dynamic_active_edge_counts,
        dtype=np.float64,
    ).reshape(-1)
    if phase_duration_ms.size != len(PHASE_NAMES):
        raise ValueError("phase_duration_ms must contain five values")
    if landmarks_ms.size != len(LANDMARK_NAMES):
        raise ValueError("landmarks_ms must contain six values")
    if dynamic_active_edge_counts.size != len(PHASE_NAMES):
        raise ValueError(
            "dynamic_active_edge_counts must contain five values"
        )

    vector = np.concatenate(
        (
            local_features,
            phase_duration_ms,
            landmarks_ms,
            dynamic_active_edge_counts,
        )
    ).astype(np.float32)
    names: list[str] = []
    for channel_index in range(EXPECTED_CHANNELS):
        for phase_name in PHASE_NAMES:
            for signal_name in NODE_FEATURE_NAMES:
                for statistic in REDUCED_SUMMARY_STAT_NAMES:
                    names.append(
                        f"channel_{channel_index + 1:02d}__"
                        f"{phase_name}__{signal_name}__{statistic}"
                    )
    names.extend(
        f"phase_duration_ms__{phase_name}"
        for phase_name in PHASE_NAMES
    )
    names.extend(
        f"landmark_ms__{landmark_name}"
        for landmark_name in LANDMARK_NAMES
    )
    names.extend(
        f"dynamic_active_edge_count__{phase_name}"
        for phase_name in PHASE_NAMES
    )
    if vector.size != len(names):
        raise RuntimeError(
            "Reduced handcrafted value/name counts do not match: "
            f"{vector.size} != {len(names)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Reduced handcrafted vector contains NaN or Inf"
        )
    return vector, np.asarray(names, dtype=str)


def empty_summary_row(npz_path: Path) -> dict[str, Any]:
    return {
        "InputNPZ": str(npz_path),
        "OutputNPZ": "",
        "OutputPNG": "",
        "Group": "",
        "Patient": "",
        "Measurement": "",
        "ChannelCount": np.nan,
        "TimePoints": np.nan,
        "NodeCount": np.nan,
        "PhaseCount": np.nan,
        "PhaseOriginalLengths": "",
        "StaticDirectedEdgeCount": np.nan,
        "DynamicActiveEdgeCount": "",
        "SpatialK": np.nan,
        "SamplingRateHz": np.nan,
        "PhaseLength": np.nan,
        "LandmarkSource": "",
        "QRSOnset": np.nan,
        "RPeak": np.nan,
        "JPoint": np.nan,
        "TOnset": np.nan,
        "TPeak": np.nan,
        "TEnd": np.nan,
        "Representation": "36_channels_no_roi",
        "Status": "FAILED",
        "ErrorMessage": "",
    }


def process_npz(
    npz_path: Path,
    output_dir: Path,
    options: ChannelNodeEdgeOptions,
    landmark_table: Mapping[str, dict[str, int]],
) -> dict[str, Any]:
    """Process one repaired NPZ while keeping all 36 physical channels."""
    row = empty_summary_row(npz_path)
    try:
        loaded = load_repaired_npz(npz_path)
        signal_ct = loaded.signal_ct
        coordinates = loaded.coordinates
        n_channels, n_time = signal_ct.shape
        row.update(
            {
                "Group": loaded.metadata.get("group", ""),
                "Patient": loaded.metadata.get("patient", ""),
                "Measurement": loaded.metadata.get(
                    "measurement",
                    "",
                ),
                "ChannelCount": n_channels,
                "TimePoints": n_time,
            }
        )
        if n_channels != EXPECTED_CHANNELS:
            row["Status"] = "SKIPPED"
            row["ErrorMessage"] = (
                f"Expected {EXPECTED_CHANNELS} channels, got {n_channels}"
            )
            return row

        active_mask = np.ones(n_channels, dtype=bool)
        coordinates_norm = normalize_coordinates(
            coordinates,
            active_mask,
        )
        channel_signal = signal_ct

        landmarks, landmark_source = resolve_landmarks(
            npz_path=npz_path,
            roi_signal=channel_signal,
            sampling_rate_hz=options.sampling_rate_hz,
            landmark_table=landmark_table,
            enable_auto=options.enable_auto_landmarks,
        )
        phase_bounds = build_phase_bounds(
            landmarks,
            n_time=n_time,
            sampling_rate_hz=options.sampling_rate_hz,
        )
        (
            normalized_channel_signal,
            baseline_offset,
            signal_scale,
        ) = normalize_roi_signal(
            channel_signal,
            phase_bounds,
        )
        (
            phase_blocks,
            phase_feature_blocks,
            phase_source_samples,
            phase_original_lengths,
            phase_duration_ms,
        ) = resample_phase_blocks(
            normalized_channel_signal,
            phase_bounds,
            phase_length=options.phase_length,
            sampling_rate_hz=options.sampling_rate_hz,
        )

        (
            edge_index,
            edge_weight,
            edge_distance,
            adjacency,
        ) = build_channel_graph_edges(
            coordinates_norm,
            spatial_k=options.spatial_k,
        )
        edge_delta = (
            coordinates_norm[edge_index[1]]
            - coordinates_norm[edge_index[0]]
        )
        distance_prior, distance_sigma = _distance_prior(
            edge_distance
        )
        static_edge_attr = np.column_stack(
            (
                edge_distance,
                edge_delta[:, 0],
                edge_delta[:, 1],
                distance_prior,
            )
        ).astype(np.float32)
        dynamic_edges = estimate_phase_dynamic_edges(
            normalized_signal=normalized_channel_signal,
            phase_bounds=phase_bounds,
            edge_index=edge_index,
            sampling_rate_hz=options.sampling_rate_hz,
            max_lag_ms=options.dynamic_max_lag_ms,
            correlation_threshold=(
                options.dynamic_corr_threshold
            ),
            direction_margin=(
                options.dynamic_direction_margin
            ),
            top_k_incoming=(
                options.dynamic_top_k_incoming
            ),
        )

        data_output_dir = output_dir / NODE_DATA_DIR_NAME
        plot_output_dir = output_dir / NODE_PLOT_DIR_NAME
        data_output_dir.mkdir(parents=True, exist_ok=True)
        if options.save_plots:
            plot_output_dir.mkdir(parents=True, exist_ok=True)

        output_stem = safe_output_stem(npz_path)
        output_path = (
            data_output_dir / f"{output_stem}_NodeEdge.npz"
        )
        plot_path = (
            plot_output_dir / f"{output_stem}_NodePhases.png"
        )

        landmark_array = np.asarray(
            [landmarks[name] for name in LANDMARK_NAMES],
            dtype=np.int32,
        )
        landmark_ms = (
            landmark_array.astype(np.float64)
            * 1000.0
            / options.sampling_rate_hz
        )
        n_phases = len(PHASE_NAMES)
        n_nodes = n_channels
        phase_sequence = phase_blocks.reshape(n_nodes, -1)
        phase_feature_sequence = phase_feature_blocks.reshape(
            n_nodes,
            -1,
            len(NODE_FEATURE_NAMES),
        )
        phase_id = np.repeat(
            np.arange(n_phases, dtype=np.int16),
            options.phase_length,
        )
        phase_progress = np.tile(
            np.linspace(
                0.0,
                1.0,
                options.phase_length,
                dtype=np.float32,
            ),
            n_phases,
        )
        channel_labels = _safe_channel_labels(n_channels)
        dynamic_active_counts = np.count_nonzero(
            dynamic_edges.active,
            axis=1,
        ).astype(np.int32)
        (
            reduced_handcrafted_features,
            reduced_handcrafted_feature_names,
        ) = build_reduced_handcrafted_features(
            node_feature_blocks=phase_feature_blocks,
            phase_duration_ms=phase_duration_ms,
            landmarks_ms=landmark_ms,
            dynamic_active_edge_counts=dynamic_active_counts,
        )

        np.savez_compressed(
            output_path,
            representation_type=np.asarray(
                "physical_channels_no_roi"
            ),
            node_signal=channel_signal.astype(np.float32),
            node_signal_normalized=(
                normalized_channel_signal.astype(np.float32)
            ),
            node_blocks=phase_blocks,
            node_feature_blocks=phase_feature_blocks,
            node_feature_sequence=phase_feature_sequence,
            node_features=phase_feature_sequence.reshape(
                n_nodes,
                -1,
            ),
            node_feature_names=np.asarray(
                NODE_FEATURE_NAMES,
                dtype=str,
            ),
            reduced_handcrafted_features=(
                reduced_handcrafted_features
            ),
            reduced_handcrafted_feature_names=(
                reduced_handcrafted_feature_names
            ),
            reduced_handcrafted_stat_names=np.asarray(
                REDUCED_SUMMARY_STAT_NAMES,
                dtype=str,
            ),
            node_labels=channel_labels,
            channel_labels=channel_labels,
            channel_indices=np.arange(
                1,
                n_channels + 1,
                dtype=np.int16,
            ),
            phase_signal=phase_sequence,
            phase_names=np.asarray(PHASE_NAMES, dtype=str),
            phase_bounds_samples=phase_bounds,
            phase_source_samples=phase_source_samples,
            phase_original_lengths=phase_original_lengths,
            phase_duration_ms=phase_duration_ms,
            phase_target_length=np.asarray(
                options.phase_length,
                dtype=np.int32,
            ),
            phase_id=phase_id,
            phase_progress=phase_progress,
            baseline_offset=baseline_offset.astype(np.float32),
            signal_scale=np.asarray(
                signal_scale,
                dtype=np.float32,
            ),
            block_mask=np.ones(
                (n_phases, options.phase_length),
                dtype=bool,
            ),
            block_names=np.asarray(PHASE_NAMES, dtype=str),
            block_bounds_samples=phase_bounds,
            block_used_bounds_samples=phase_bounds,
            block_original_lengths=phase_original_lengths,
            block_truncated=np.zeros(n_phases, dtype=bool),
            block_length=np.asarray(
                options.phase_length,
                dtype=np.int32,
            ),
            projection_matrix=np.eye(
                n_channels,
                dtype=np.float32,
            ),
            node_positions=coordinates_norm.astype(np.float32),
            channel_coordinates=coordinates.astype(np.float32),
            channel_coordinates_norm=(
                coordinates_norm.astype(np.float32)
            ),
            roi_centers=coordinates_norm.astype(np.float32),
            roi_side=np.asarray(0, dtype=np.int16),
            sensor_coordinates=coordinates.astype(np.float32),
            sensor_coordinates_norm=(
                coordinates_norm.astype(np.float32)
            ),
            active_channel_mask=active_mask,
            excluded_corner_channels=np.empty(
                0,
                dtype=np.int32,
            ),
            roi_sigma=np.asarray(np.nan, dtype=np.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_distance=edge_distance,
            adjacency=adjacency,
            edge_mode=np.asarray(
                f"coordinate_knn_{options.spatial_k}"
            ),
            spatial_k=np.asarray(
                options.spatial_k,
                dtype=np.int16,
            ),
            spatial_distance_sigma=np.asarray(
                distance_sigma,
                dtype=np.float32,
            ),
            static_edge_attr=static_edge_attr,
            static_edge_attr_names=np.asarray(
                STATIC_EDGE_ATTRIBUTE_NAMES,
                dtype=str,
            ),
            dynamic_edge_index=edge_index,
            dynamic_edge_weight=dynamic_edges.weight,
            dynamic_edge_correlation=(
                dynamic_edges.correlation
            ),
            dynamic_edge_strength=dynamic_edges.strength,
            dynamic_edge_lag_samples=(
                dynamic_edges.lag_samples
            ),
            dynamic_edge_lag_ms=(
                dynamic_edges.lag_samples.astype(np.float32)
                * 1000.0
                / options.sampling_rate_hz
            ),
            dynamic_edge_active=dynamic_edges.active,
            dynamic_edge_direction_confidence=(
                dynamic_edges.direction_confidence
            ),
            dynamic_adjacency=dynamic_edges.adjacency,
            dynamic_active_edge_counts=dynamic_active_counts,
            dynamic_max_lag_ms=np.asarray(
                options.dynamic_max_lag_ms,
                dtype=np.float32,
            ),
            dynamic_corr_threshold=np.asarray(
                options.dynamic_corr_threshold,
                dtype=np.float32,
            ),
            dynamic_direction_margin=np.asarray(
                options.dynamic_direction_margin,
                dtype=np.float32,
            ),
            dynamic_top_k_incoming=np.asarray(
                options.dynamic_top_k_incoming,
                dtype=np.int16,
            ),
            landmark_names=np.asarray(
                LANDMARK_NAMES,
                dtype=str,
            ),
            landmarks_samples=landmark_array,
            landmarks_ms=landmark_ms.astype(np.float32),
            landmark_source=np.asarray(landmark_source),
            sampling_rate_hz=np.asarray(
                options.sampling_rate_hz,
                dtype=np.float32,
            ),
            source_npz=np.asarray(str(npz_path)),
            source_file=np.asarray(
                loaded.metadata.get("source_file", "")
            ),
            group=np.asarray(loaded.metadata.get("group", "")),
            patient=np.asarray(
                loaded.metadata.get("patient", "")
            ),
            measurement=np.asarray(
                loaded.metadata.get("measurement", "")
            ),
        )

        output_png = ""
        if options.save_plots:
            save_phase_node_plot(
                phase_blocks=phase_blocks,
                phase_source_samples=phase_source_samples,
                centers=coordinates_norm,
                dynamic_active=dynamic_edges.active,
                metadata=loaded.metadata,
                npz_path=npz_path,
                landmark_source=landmark_source,
                plot_path=plot_path,
            )
            output_png = str(plot_path)

        row.update(
            {
                "OutputNPZ": str(output_path),
                "OutputPNG": output_png,
                "NodeCount": n_nodes,
                "PhaseCount": n_phases,
                "PhaseOriginalLengths": ",".join(
                    map(str, phase_original_lengths)
                ),
                "StaticDirectedEdgeCount": (
                    edge_index.shape[1]
                ),
                "DynamicActiveEdgeCount": ",".join(
                    map(str, dynamic_active_counts)
                ),
                "SpatialK": options.spatial_k,
                "SamplingRateHz": options.sampling_rate_hz,
                "PhaseLength": options.phase_length,
                "LandmarkSource": landmark_source,
                "QRSOnset": landmarks["qrs_onset"],
                "RPeak": landmarks["r_peak"],
                "JPoint": landmarks["j_point"],
                "TOnset": landmarks["t_onset"],
                "TPeak": landmarks["t_peak"],
                "TEnd": landmarks["t_end"],
                "Status": "OK",
                "ErrorMessage": "",
            }
        )
    except Exception as exc:
        row["ErrorMessage"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Failed to process %s", npz_path)
    return row


def discover_npz_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.rglob("*.npz")
        if path.is_file()
        and not re.search(
            r"_NodeEdge\d*\.npz$",
            path.name,
            flags=re.IGNORECASE,
        )
    ]
    return sorted(files, key=natural_key)


def run_batch(
    input_dir: Path,
    output_dir: Path,
    options: ChannelNodeEdgeOptions,
    landmarks_csv: Path | None,
    landmark_unit: str,
) -> pd.DataFrame:
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )
    files = discover_npz_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No repaired NPZ files found under {input_dir}"
        )
    landmark_table = load_landmark_csv(
        landmarks_csv,
        unit=landmark_unit,
        sampling_rate_hz=options.sampling_rate_hz,
    )
    rows: list[dict[str, Any]] = []
    for index, npz_path in enumerate(files, start=1):
        logging.info(
            "[%d/%d] %s",
            index,
            len(files),
            npz_path.name,
        )
        rows.append(
            process_npz(
                npz_path,
                output_dir,
                options,
                landmark_table,
            )
        )
    return _write_summary(rows, output_dir)


def run_single(
    file_path: Path,
    output_dir: Path,
    options: ChannelNodeEdgeOptions,
    landmarks_csv: Path | None,
    landmark_unit: str,
) -> pd.DataFrame:
    file_path = file_path.expanduser()
    output_dir = output_dir.expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )
    if file_path.suffix.lower() != ".npz":
        raise ValueError("Single-file mode supports NPZ only")
    landmark_table = load_landmark_csv(
        landmarks_csv,
        unit=landmark_unit,
        sampling_rate_hz=options.sampling_rate_hz,
    )
    row = process_npz(
        file_path,
        output_dir,
        options,
        landmark_table,
    )
    return _write_summary([row], output_dir)


def _write_summary(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(empty_summary_row(Path("x.npz")).keys())
    summary = pd.DataFrame(rows, columns=columns)
    summary_path = output_dir / "NodeEdge2_Summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    logging.info(
        "Channel NodeEdge data: %s",
        output_dir / NODE_DATA_DIR_NAME,
    )
    if summary["OutputPNG"].fillna("").astype(bool).any():
        logging.info(
            "Node time-segment images: %s",
            output_dir / NODE_PLOT_DIR_NAME,
        )
    logging.info("Summary: %s", summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split repaired 36-channel MCG signals into five phases "
            "without ROI projection and build channel-level graphs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing repaired NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root (default: data/node_edge2)",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Process one repaired NPZ file only",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Sampling rate in Hz (default: 1000)",
    )
    parser.add_argument(
        "--phase-length",
        type=int,
        default=128,
        help="Resampled length of each phase (default: 128)",
    )
    parser.add_argument(
        "--spatial-k",
        type=int,
        default=8,
        help="Coordinate nearest neighbors per channel (default: 8)",
    )
    parser.add_argument(
        "--landmarks-csv",
        type=Path,
        default=None,
        help="Optional landmark CSV",
    )
    parser.add_argument(
        "--landmark-unit",
        choices=("samples", "ms"),
        default="samples",
        help="Unit of landmark CSV values",
    )
    parser.add_argument(
        "--no-auto-landmarks",
        action="store_true",
        help="Fail when NPZ/CSV landmarks are unavailable",
    )
    parser.add_argument(
        "--dynamic-max-lag-ms",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--dynamic-corr-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--dynamic-direction-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dynamic-top-k-incoming",
        type=int,
        default=2,
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        help=(
            "Render the 36-channel node time-segment image "
            "(enabled by default)"
        ),
    )
    plot_group.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Skip node time-segment image rendering",
    )
    parser.set_defaults(save_plots=True)
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
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    options = ChannelNodeEdgeOptions(
        sampling_rate_hz=args.fs,
        phase_length=args.phase_length,
        spatial_k=args.spatial_k,
        dynamic_max_lag_ms=args.dynamic_max_lag_ms,
        dynamic_corr_threshold=(
            args.dynamic_corr_threshold
        ),
        dynamic_direction_margin=(
            args.dynamic_direction_margin
        ),
        dynamic_top_k_incoming=(
            args.dynamic_top_k_incoming
        ),
        enable_auto_landmarks=(
            not args.no_auto_landmarks
        ),
        save_plots=args.save_plots,
    )
    try:
        if args.single_file is not None:
            result = run_single(
                file_path=args.single_file,
                output_dir=args.output_dir,
                options=options,
                landmarks_csv=args.landmarks_csv,
                landmark_unit=args.landmark_unit,
            )
        else:
            result = run_batch(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                options=options,
                landmarks_csv=args.landmarks_csv,
                landmark_unit=args.landmark_unit,
            )
        counts = result["Status"].value_counts(
            dropna=False
        ).to_dict()
        logging.info("Status counts: %s", counts)
        return 0 if counts.get("FAILED", 0) == 0 else 2
    except Exception as exc:
        logging.error("NodeEdge2 failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
