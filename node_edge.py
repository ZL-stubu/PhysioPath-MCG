from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_INPUT_DIR = Path(
    r"D:\pycharmproject\GCNMCG\data\203"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\pycharmproject\GCNMCG\data\node_edge"
)
NODE_DATA_DIR_NAME = "节点边数据"
NODE_PLOT_DIR_NAME = "节点时间段图片"

SUPPORTED_CHANNELS = (36, 64)
LANDMARK_NAMES = (
    "qrs_onset",
    "r_peak",
    "j_point",
    "t_onset",
    "t_peak",
    "t_end",
)
BLOCK_NAMES = (
    "baseline_qrs_minus_50ms_to_qrs_onset",
    "qrs_onset_to_r_peak",
    "r_peak_to_j_point",
    "j_point_to_j_plus_40ms",
    "j_plus_40ms_to_j_plus_80ms",
    "j_plus_80ms_to_t_onset",
    "t_onset_to_t_peak",
    "t_peak_to_t_end",
)
BLOCK_PLOT_TITLES = (
    "B1\nQRS-50 ms → QRS onset",
    "B2\nQRS onset → R peak",
    "B3\nR peak → J point",
    "B4\nJ point → J+40 ms",
    "B5\nJ+40 ms → J+80 ms",
    "B6\nJ+80 ms → T onset",
    "B7\nT onset → T peak",
    "B8\nT peak → T end",
)
PHASE_NAMES = (
    "baseline",
    "qrs_depolarization",
    "early_st",
    "late_repolarization",
    "t_recovery",
)
PHASE_PLOT_TITLES = (
    "P1\nQRS-50 ms → QRS onset",
    "P2\nQRS onset → J point",
    "P3\nJ point → J+80 ms",
    "P4\nJ+80 ms → T peak",
    "P5\nT peak → T end",
)
NODE_FEATURE_NAMES = (
    "normalized_signal",
    "temporal_derivative",
    "activation_magnitude",
)
STATIC_EDGE_ATTRIBUTE_NAMES = (
    "distance",
    "delta_x",
    "delta_y",
    "distance_prior",
)


@dataclass(frozen=True)
class NodeEdgeOptions:
    """坐标、ROI、时间块和图结构参数。"""

    sampling_rate_hz: float = 1000.0
    roi_sigma: float = 0.35
    roi_side: int = 4
    block_length: int = 128
    edge_mode: str = "8"
    dynamic_max_lag_ms: float = 20.0
    dynamic_corr_threshold: float = 0.25
    dynamic_direction_margin: float = 0.05
    dynamic_top_k_incoming: int = 2
    corner_policy: str = "zero-only"
    corner_zero_tolerance: float = 1e-10
    enable_auto_landmarks: bool = True


@dataclass(frozen=True)
class LoadedRepairedData:
    """从坏道修复 NPZ 中读取的标准化内容。"""

    signal_ct: np.ndarray
    coordinates: np.ndarray
    metadata: dict[str, str]


@dataclass(frozen=True)
class DynamicEdgeData:
    """每个生理阶段的有向有效耦合边。"""

    weight: np.ndarray
    correlation: np.ndarray
    strength: np.ndarray
    lag_samples: np.ndarray
    active: np.ndarray
    direction_confidence: np.ndarray
    adjacency: np.ndarray


def natural_key(value: str | Path) -> list[Any]:
    text = str(value).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def scalar_text(value: Any) -> str:
    """安全地把 NPZ 中的标量字符串转为 Python 字符串。"""
    array = np.asarray(value)
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def ms_to_samples(milliseconds: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(milliseconds * sampling_rate_hz / 1000.0)))


def load_repaired_npz(npz_path: Path) -> LoadedRepairedData:
    """读取修复后的 NPZ，并统一信号方向为 C×T。"""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if "data_repaired" in archive:
                repaired = np.asarray(archive["data_repaired"], dtype=np.float64)
            elif "x_repaired" in archive:
                repaired = np.asarray(archive["x_repaired"], dtype=np.float64)
            else:
                raise KeyError("NPZ 中未找到 data_repaired 或 x_repaired。")

            if "coordinates" not in archive:
                raise KeyError("NPZ 中未找到 coordinates。")
            coordinates = np.asarray(archive["coordinates"], dtype=np.float64)

            metadata: dict[str, str] = {}
            for key in ("source_file", "group", "patient", "measurement"):
                if key in archive:
                    metadata[key] = scalar_text(archive[key])

    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"无法读取修复数据 {npz_path}：{exc}") from exc

    if repaired.ndim != 2:
        raise ValueError(f"data_repaired 必须是二维矩阵，实际形状为 {repaired.shape}。")

    # 兼容 T×C 和 C×T 两种保存方向。
    if repaired.shape[1] in SUPPORTED_CHANNELS:
        signal_ct = repaired.T
    elif repaired.shape[0] in SUPPORTED_CHANNELS:
        signal_ct = repaired
    else:
        raise ValueError(
            f"无法识别通道维度，修复矩阵形状为 {repaired.shape}；"
            "仅支持 36 或 64 通道。"
        )

    n_channels, n_time = signal_ct.shape
    if coordinates.shape != (n_channels, 2):
        raise ValueError(
            f"coordinates 应为 {n_channels}×2，实际形状为 {coordinates.shape}。"
        )
    if n_time < 2:
        raise ValueError("信号时间点不足。")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates 中包含 NaN 或 Inf。")

    # 坏道修复阶段通常已经处理无效值；这里仍做最后保护。
    if not np.all(np.isfinite(signal_ct)):
        invalid_count = int(np.count_nonzero(~np.isfinite(signal_ct)))
        logging.warning(
            "%s 中仍含 %d 个 NaN/Inf，将按通道中位数替换。",
            npz_path,
            invalid_count,
        )
        signal_ct = signal_ct.copy()
        for channel in range(n_channels):
            finite = np.isfinite(signal_ct[channel])
            if np.any(finite):
                replacement = float(np.median(signal_ct[channel, finite]))
            else:
                replacement = 0.0
            signal_ct[channel, ~finite] = replacement

    return LoadedRepairedData(
        signal_ct=np.asarray(signal_ct, dtype=np.float64),
        coordinates=coordinates,
        metadata=metadata,
    )


def corner_indices(n_channels: int) -> np.ndarray:
    side = int(round(math.sqrt(n_channels)))
    if side * side != n_channels:
        raise ValueError(f"通道数 {n_channels} 不能构成规则方阵。")
    return np.asarray([0, side - 1, n_channels - side, n_channels - 1], dtype=int)


def active_channel_mask(
    signal_ct: np.ndarray,
    policy: str,
    zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回参与 ROI 投影的物理通道掩码和被排除通道。

    policy:
      - zero-only：仅排除全程近似为 0 的四个角；
      - always：无条件排除四个角；
      - none：不排除四角。
    """
    n_channels = signal_ct.shape[0]
    mask = np.ones(n_channels, dtype=bool)
    corners = corner_indices(n_channels)

    if policy == "none":
        excluded = np.empty(0, dtype=int)
    elif policy == "always":
        excluded = corners
    elif policy == "zero-only":
        zero_corner = np.max(np.abs(signal_ct[corners]), axis=1) <= zero_tolerance
        excluded = corners[zero_corner]
    else:
        raise ValueError(f"未知 corner_policy：{policy}")

    mask[excluded] = False
    if np.count_nonzero(mask) < 9:
        raise ValueError("可参与 ROI 投影的物理通道少于 9 个。")
    return mask, excluded


def normalize_coordinates(
    coordinates: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """将横纵坐标分别线性映射到 [-1, 1]。"""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    active = coordinates[active_mask]

    minimum = np.min(active, axis=0)
    maximum = np.max(active, axis=0)
    span = maximum - minimum
    if np.any(span <= np.finfo(float).eps):
        raise ValueError("横坐标或纵坐标没有变化，无法归一化。")

    normalized = 2.0 * (coordinates - minimum) / span - 1.0
    return normalized


def roi_centers(side: int = 4) -> np.ndarray:
    """生成标准方形 ROI 中心，节点按从下到上、每行从左到右排列。"""
    if side < 2:
        raise ValueError("roi_side 至少为 2。")
    axis = np.linspace(
        -1.0 + 1.0 / side,
        1.0 - 1.0 / side,
        side,
        dtype=np.float64,
    )
    return np.asarray(
        [(x, y) for y in axis for x in axis],
        dtype=np.float64,
    )


def build_projection_matrix(
    coordinates_norm: np.ndarray,
    centers: np.ndarray,
    sigma: float,
    active_mask: np.ndarray,
) -> np.ndarray:
    """构造行归一化的高斯软 ROI 投影矩阵 P，形状为 N×C。"""
    if sigma <= 0:
        raise ValueError("roi_sigma 必须大于 0。")

    differences = centers[:, None, :] - coordinates_norm[None, :, :]
    distance_squared = np.sum(differences * differences, axis=2)
    gaussian = np.exp(-distance_squared / (2.0 * sigma * sigma))

    # 结构性空角不参与分母，且其最终权重固定为 0。
    gaussian[:, ~active_mask] = 0.0
    row_sum = np.sum(gaussian, axis=1, keepdims=True)
    if np.any(row_sum <= np.finfo(float).eps):
        raise ValueError("至少一个 ROI 没有可用物理通道权重。")

    projection = gaussian / row_sum
    return projection


def build_graph_edges(
    centers: np.ndarray,
    edge_mode: str = "4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造规则方形 ROI 图的双向边、单位边权、边距离和邻接矩阵。"""
    if edge_mode not in {"4", "8"}:
        raise ValueError("edge_mode 只能为 '4' 或 '8'。")

    n_nodes = int(centers.shape[0])
    side = int(round(math.sqrt(n_nodes)))
    if centers.shape != (n_nodes, 2) or side * side != n_nodes:
        raise ValueError(f"centers 必须表示规则方阵，实际形状为 {centers.shape}。")
    undirected: set[tuple[int, int]] = set()
    offsets = [(0, 1), (1, 0)]
    if edge_mode == "8":
        offsets.extend([(1, 1), (1, -1)])

    for row in range(side):
        for col in range(side):
            source = row * side + col
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < side and 0 <= nc < side:
                    target = nr * side + nc
                    undirected.add((min(source, target), max(source, target)))

    directed: list[tuple[int, int]] = []
    for source, target in sorted(undirected):
        directed.append((source, target))
        directed.append((target, source))

    edge_index = np.asarray(directed, dtype=np.int64).T
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = edge_weight

    # 额外保存距离，便于以后改为距离权重；当前 edge_weight 保持为 1。
    edge_distance = np.linalg.norm(
        centers[edge_index[0]] - centers[edge_index[1]], axis=1
    ).astype(np.float32)
    return edge_index, edge_weight, edge_distance, adjacency


def smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="same")


def find_left_quiet_boundary(
    score: np.ndarray,
    start: int,
    lower: int,
    threshold: float,
    hold: int,
) -> int | None:
    """从 start 向左寻找连续低于阈值的安静区右边界。"""
    start = min(start, score.size - 1)
    lower = max(0, lower)
    hold = max(1, hold)
    for right in range(start, lower + hold - 1, -1):
        left = max(lower, right - hold + 1)
        if right - left + 1 >= hold and np.max(score[left : right + 1]) <= threshold:
            return right
    return None


def find_right_quiet_boundary(
    score: np.ndarray,
    start: int,
    upper: int,
    threshold: float,
    hold: int,
) -> int | None:
    """从 start 向右寻找连续低于阈值的安静区左边界。"""
    start = max(0, start)
    upper = min(score.size, upper)
    hold = max(1, hold)
    for left in range(start, max(start + 1, upper - hold + 1)):
        right = left + hold
        if right <= upper and np.max(score[left:right]) <= threshold:
            return left
    return None


def detect_landmarks_auto(
    roi_signal: np.ndarray,
    sampling_rate_hz: float,
) -> dict[str, int]:
    """
    基于全局导数能量和幅值包络的启发式关键点检测。

    该函数用于缺少外部关键点时保证流程可运行，不替代人工或临床级标注。
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz 必须大于 0。")

    n_nodes, n_time = roi_signal.shape
    if n_nodes < 4 or n_time < ms_to_samples(300.0, sampling_rate_hz):
        raise ValueError("信号过短，无法自动检测完整 QRS-T 关键点。")

    centered = roi_signal - np.median(roi_signal, axis=1, keepdims=True)
    envelope = np.sqrt(np.mean(centered * centered, axis=0))
    derivative = np.diff(centered, axis=1, prepend=centered[:, :1])
    derivative_energy = np.sqrt(np.mean(derivative * derivative, axis=0))

    envelope = smooth_1d(envelope, ms_to_samples(8.0, sampling_rate_hz))
    qrs_score = smooth_1d(derivative_energy, ms_to_samples(6.0, sampling_rate_hz))

    search_start = max(0, int(round(0.05 * n_time)))
    search_end = min(n_time, int(round(0.75 * n_time)))
    if search_end - search_start < 3:
        raise ValueError("自动 QRS 搜索窗口无效。")

    qrs_center = search_start + int(np.argmax(qrs_score[search_start:search_end]))
    r_half_width = ms_to_samples(80.0, sampling_rate_hz)
    r_left = max(0, qrs_center - r_half_width)
    r_right = min(n_time, qrs_center + r_half_width + 1)
    r_peak = r_left + int(np.argmax(envelope[r_left:r_right]))

    baseline_end = max(1, int(round(0.10 * n_time)))
    qrs_baseline = float(np.median(qrs_score[:baseline_end]))
    qrs_peak = float(qrs_score[qrs_center])
    qrs_threshold = qrs_baseline + 0.12 * max(qrs_peak - qrs_baseline, 0.0)
    quiet_hold = ms_to_samples(10.0, sampling_rate_hz)

    onset_lower = max(0, r_peak - ms_to_samples(180.0, sampling_rate_hz))
    onset_candidate = find_left_quiet_boundary(
        qrs_score,
        r_peak,
        onset_lower,
        qrs_threshold,
        quiet_hold,
    )
    qrs_onset = (
        onset_candidate
        if onset_candidate is not None
        else max(0, r_peak - ms_to_samples(80.0, sampling_rate_hz))
    )

    j_upper = min(n_time, r_peak + ms_to_samples(220.0, sampling_rate_hz))
    j_candidate = find_right_quiet_boundary(
        qrs_score,
        r_peak,
        j_upper,
        qrs_threshold,
        quiet_hold,
    )
    j_point = (
        j_candidate
        if j_candidate is not None
        else min(n_time - 1, r_peak + ms_to_samples(100.0, sampling_rate_hz))
    )

    j80 = j_point + ms_to_samples(80.0, sampling_rate_hz)
    t_search_start = min(n_time - 1, j80)
    t_search_end = min(n_time, j_point + ms_to_samples(650.0, sampling_rate_hz))
    if t_search_end - t_search_start < 3:
        raise ValueError("J 点之后没有足够长度用于自动检测 T 波。")

    t_peak = t_search_start + int(np.argmax(envelope[t_search_start:t_search_end]))
    env_baseline = float(np.median(envelope[:baseline_end]))
    env_t_peak = float(envelope[t_peak])
    t_threshold = env_baseline + 0.10 * max(env_t_peak - env_baseline, 0.0)
    t_quiet_hold = ms_to_samples(15.0, sampling_rate_hz)

    t_onset_lower = max(j80, t_peak - ms_to_samples(300.0, sampling_rate_hz))
    t_onset_candidate = find_left_quiet_boundary(
        envelope,
        t_peak,
        t_onset_lower,
        t_threshold,
        t_quiet_hold,
    )
    t_onset = (
        t_onset_candidate
        if t_onset_candidate is not None
        else max(j80, t_peak - ms_to_samples(120.0, sampling_rate_hz))
    )

    t_end_upper = min(n_time, t_peak + ms_to_samples(450.0, sampling_rate_hz))
    t_end_candidate = find_right_quiet_boundary(
        envelope,
        t_peak,
        t_end_upper,
        t_threshold,
        t_quiet_hold,
    )
    t_end = (
        t_end_candidate
        if t_end_candidate is not None
        else min(n_time, t_peak + ms_to_samples(180.0, sampling_rate_hz))
    )

    # 防止自动启发式搜索产生相等或逆序点。
    qrs_onset = int(np.clip(qrs_onset, 0, n_time - 6))
    r_peak = int(np.clip(max(r_peak, qrs_onset + 1), 1, n_time - 5))
    j_point = int(np.clip(max(j_point, r_peak + 1), 2, n_time - 4))
    j80 = j_point + ms_to_samples(80.0, sampling_rate_hz)
    if j80 >= n_time - 2:
        raise ValueError("自动检测的 J 点过晚，无法形成后续时间块。")
    t_onset = int(np.clip(max(t_onset, j80 + 1), j80 + 1, n_time - 3))
    t_peak = int(np.clip(max(t_peak, t_onset + 1), t_onset + 1, n_time - 2))
    t_end = int(np.clip(max(t_end, t_peak + 1), t_peak + 1, n_time))

    return {
        "qrs_onset": qrs_onset,
        "r_peak": r_peak,
        "j_point": j_point,
        "t_onset": t_onset,
        "t_peak": t_peak,
        "t_end": t_end,
    }


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def load_landmark_csv(
    csv_path: Path | None,
    unit: str,
    sampling_rate_hz: float,
) -> dict[str, dict[str, int]]:
    """读取外部关键点表，返回可按文件名或 stem 查询的映射。"""
    if csv_path is None:
        return {}
    if not csv_path.is_file():
        raise FileNotFoundError(f"关键点 CSV 不存在：{csv_path}")
    if unit not in {"samples", "ms"}:
        raise ValueError("landmark_unit 只能为 samples 或 ms。")

    table = pd.read_csv(csv_path)
    normalized = {normalize_column_name(column): column for column in table.columns}

    identifier_aliases = ("npzfile", "filename", "file", "stem")
    identifier_column = next(
        (normalized[name] for name in identifier_aliases if name in normalized),
        None,
    )
    if identifier_column is None:
        raise ValueError(
            "关键点 CSV 需要一个标识列：NPZFile、FileName、File 或 Stem。"
        )

    aliases: Mapping[str, tuple[str, ...]] = {
        "qrs_onset": ("qrsonset", "qrsstart"),
        "r_peak": ("rpeak",),
        "j_point": ("jpoint", "qrsend"),
        "t_onset": ("tonset", "tstart"),
        "t_peak": ("tpeak",),
        "t_end": ("tend", "toffset"),
    }

    landmark_columns: dict[str, str] = {}
    for landmark, candidates in aliases.items():
        column = next((normalized[name] for name in candidates if name in normalized), None)
        if column is None:
            raise ValueError(f"关键点 CSV 缺少列：{landmark}")
        landmark_columns[landmark] = column

    mapping: dict[str, dict[str, int]] = {}
    for row_index, row in table.iterrows():
        identifier = str(row[identifier_column]).strip()
        if not identifier or identifier.lower() == "nan":
            continue

        points: dict[str, int] = {}
        for landmark, column in landmark_columns.items():
            value = row[column]
            if pd.isna(value):
                raise ValueError(f"关键点 CSV 第 {row_index + 2} 行的 {landmark} 为空。")
            numeric = float(value)
            if unit == "ms":
                numeric = numeric * sampling_rate_hz / 1000.0
            points[landmark] = int(round(numeric))

        filename_key = Path(identifier).name.lower()
        stem_key = Path(identifier).stem.lower()
        mapping[filename_key] = points
        mapping[stem_key] = points

    return mapping


def landmarks_from_npz(npz_path: Path) -> dict[str, int] | None:
    """尝试从输入 NPZ 中直接读取六个关键点。"""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if not all(name in archive for name in LANDMARK_NAMES):
                return None
            return {
                name: int(round(float(np.asarray(archive[name]).reshape(-1)[0])))
                for name in LANDMARK_NAMES
            }
    except (OSError, ValueError, TypeError):
        return None


def validate_landmarks(
    landmarks: Mapping[str, int],
    n_time: int,
    sampling_rate_hz: float,
) -> dict[str, int]:
    """验证关键点范围、顺序以及 J+80ms 与 T 起点的关系。"""
    missing = [name for name in LANDMARK_NAMES if name not in landmarks]
    if missing:
        raise ValueError(f"缺少关键点：{', '.join(missing)}")

    points = {name: int(landmarks[name]) for name in LANDMARK_NAMES}
    qrs, r_peak, j_point, t_onset, t_peak, t_end = (
        points[name] for name in LANDMARK_NAMES
    )

    if not (0 <= qrs < r_peak < j_point < t_onset < t_peak < t_end <= n_time):
        raise ValueError(
            "关键点必须满足 0 <= QRS起点 < R峰 < J点 < T起点 < T峰 < T终点 "
            f"<= {n_time}，实际为 {points}。"
        )

    j80 = j_point + ms_to_samples(80.0, sampling_rate_hz)
    if j80 >= t_onset:
        raise ValueError(
            f"T 波起点必须晚于 J+80ms；当前 J+80={j80}，T起点={t_onset}。"
        )
    return points


def resolve_landmarks(
    npz_path: Path,
    roi_signal: np.ndarray,
    sampling_rate_hz: float,
    landmark_table: Mapping[str, dict[str, int]],
    enable_auto: bool,
) -> tuple[dict[str, int], str]:
    """按 NPZ、CSV、自动检测的优先顺序获取关键点。"""
    embedded = landmarks_from_npz(npz_path)
    if embedded is not None:
        return validate_landmarks(
            embedded,
            roi_signal.shape[1],
            sampling_rate_hz,
        ), "npz"

    for key in (npz_path.name.lower(), npz_path.stem.lower()):
        if key in landmark_table:
            return validate_landmarks(
                landmark_table[key],
                roi_signal.shape[1],
                sampling_rate_hz,
            ), "csv"

    if not enable_auto:
        raise ValueError("未找到关键点，且自动关键点检测已关闭。")

    logging.warning("%s 未提供关键点，使用自动启发式检测。", npz_path.name)
    detected = detect_landmarks_auto(roi_signal, sampling_rate_hz)
    return validate_landmarks(
        detected,
        roi_signal.shape[1],
        sampling_rate_hz,
    ), "auto_heuristic"


def build_block_bounds(
    landmarks: Mapping[str, int],
    n_time: int,
    sampling_rate_hz: float,
) -> np.ndarray:
    """生成 8 个时间块的 [start, end) 采样点边界。"""
    qrs_onset = landmarks["qrs_onset"]
    r_peak = landmarks["r_peak"]
    j_point = landmarks["j_point"]
    t_onset = landmarks["t_onset"]
    t_peak = landmarks["t_peak"]
    t_end = landmarks["t_end"]

    n40 = ms_to_samples(40.0, sampling_rate_hz)
    n50 = ms_to_samples(50.0, sampling_rate_hz)
    j40 = j_point + n40
    j80 = j_point + 2 * n40

    bounds = np.asarray(
        [
            [max(0, qrs_onset - n50), qrs_onset],
            [qrs_onset, r_peak],
            [r_peak, j_point],
            [j_point, min(n_time, j40)],
            [min(n_time, j40), min(n_time, j80)],
            [min(n_time, j80), t_onset],
            [t_onset, t_peak],
            [t_peak, t_end],
        ],
        dtype=np.int32,
    )

    for index, (start, end) in enumerate(bounds):
        if not (0 <= start < end <= n_time):
            raise ValueError(
                f"时间块 {index + 1} 边界无效：[{start}, {end})，总长度={n_time}。"
            )
    return bounds


def build_phase_bounds(
    landmarks: Mapping[str, int],
    n_time: int,
    sampling_rate_hz: float,
) -> np.ndarray:
    """生成用于动态图学习的 5 个连续生理阶段。"""
    qrs_onset = int(landmarks["qrs_onset"])
    j_point = int(landmarks["j_point"])
    t_peak = int(landmarks["t_peak"])
    t_end = int(landmarks["t_end"])
    n50 = ms_to_samples(50.0, sampling_rate_hz)
    n80 = ms_to_samples(80.0, sampling_rate_hz)
    j80 = j_point + n80

    bounds = np.asarray(
        [
            [max(0, qrs_onset - n50), qrs_onset],
            [qrs_onset, j_point],
            [j_point, min(n_time, j80)],
            [min(n_time, j80), t_peak],
            [t_peak, t_end],
        ],
        dtype=np.int32,
    )

    for phase_index, (start, end) in enumerate(bounds):
        if not (0 <= start < end <= n_time):
            raise ValueError(
                f"生理阶段 {phase_index + 1} 边界无效：[{start}, {end})，"
                f"总长度={n_time}。"
            )
    if not np.all(bounds[1:, 0] == bounds[:-1, 1]):
        raise ValueError("5 个生理阶段必须连续且互不重叠。")
    return bounds


def normalize_roi_signal(
    roi_signal: np.ndarray,
    phase_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按节点去基线，并用全图 QRS 幅值统一缩放以保留空间幅值差异。"""
    baseline_start, baseline_end = map(int, phase_bounds[0])
    qrs_start, qrs_end = map(int, phase_bounds[1])

    baseline_offset = np.median(
        roi_signal[:, baseline_start:baseline_end],
        axis=1,
    )
    centered = roi_signal - baseline_offset[:, None]
    qrs_values = np.abs(centered[:, qrs_start:qrs_end])
    scale = float(np.percentile(qrs_values, 95.0))
    eps = np.finfo(np.float64).eps
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0

    return centered / scale, baseline_offset, scale


def resample_phase_blocks(
    normalized_signal: np.ndarray,
    phase_bounds: np.ndarray,
    phase_length: int,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把每个生理阶段线性重采样为固定长度，同时保留原始时间映射。"""
    if phase_length < 2:
        raise ValueError("block_length 至少为 2，才能进行阶段重采样。")

    n_nodes = normalized_signal.shape[0]
    n_phases = phase_bounds.shape[0]
    phase_blocks = np.zeros(
        (n_nodes, n_phases, phase_length),
        dtype=np.float32,
    )
    phase_source_samples = np.zeros((n_phases, phase_length), dtype=np.float32)
    original_lengths = np.zeros(n_phases, dtype=np.int32)
    phase_duration_ms = np.zeros(n_phases, dtype=np.float32)

    for phase_index, (start, end) in enumerate(phase_bounds):
        start = int(start)
        end = int(end)
        segment = normalized_signal[:, start:end]
        original_length = int(segment.shape[1])
        if original_length < 1:
            raise ValueError(f"生理阶段 {phase_index + 1} 没有可用样本。")

        original_lengths[phase_index] = original_length
        phase_duration_ms[phase_index] = (
            original_length * 1000.0 / sampling_rate_hz
        )
        source_axis = np.arange(original_length, dtype=np.float64)
        target_axis = np.linspace(
            0.0,
            float(original_length - 1),
            phase_length,
            dtype=np.float64,
        )
        phase_source_samples[phase_index] = start + target_axis

        for node_index in range(n_nodes):
            phase_blocks[node_index, phase_index] = np.interp(
                target_axis,
                source_axis,
                segment[node_index],
            ).astype(np.float32)

    derivative_blocks = np.gradient(phase_blocks, axis=2).astype(np.float32)
    feature_blocks = np.stack(
        [phase_blocks, derivative_blocks, np.abs(derivative_blocks)],
        axis=-1,
    ).astype(np.float32)
    return (
        phase_blocks,
        feature_blocks,
        phase_source_samples,
        original_lengths,
        phase_duration_ms,
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 4 or y.size != x.size:
        return 0.0
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def estimate_phase_dynamic_edges(
    normalized_signal: np.ndarray,
    phase_bounds: np.ndarray,
    edge_index: np.ndarray,
    sampling_rate_hz: float,
    max_lag_ms: float,
    correlation_threshold: float,
    direction_margin: float,
    top_k_incoming: int,
) -> DynamicEdgeData:
    """
    在物理候选边上估计分阶段、有方向的有效耦合。

    对每条 source→target 边，寻找 source 导数领先 target 导数时绝对相关
    最大的正时延。结果只表示测量平面上的预测性耦合，不代表解剖因果。
    """
    if max_lag_ms <= 0:
        raise ValueError("dynamic_max_lag_ms 必须大于 0。")
    if not 0.0 <= correlation_threshold < 1.0:
        raise ValueError("dynamic_corr_threshold 必须位于 [0, 1)。")
    if direction_margin < 0.0:
        raise ValueError("dynamic_direction_margin 不能为负数。")
    if top_k_incoming < 0:
        raise ValueError("dynamic_top_k_incoming 不能为负数。")

    n_nodes = normalized_signal.shape[0]
    n_phases = phase_bounds.shape[0]
    n_edges = edge_index.shape[1]
    max_lag_samples = ms_to_samples(max_lag_ms, sampling_rate_hz)

    correlation = np.zeros((n_phases, n_edges), dtype=np.float32)
    strength = np.zeros((n_phases, n_edges), dtype=np.float32)
    lag_samples = np.zeros((n_phases, n_edges), dtype=np.int16)

    for phase_index, (start, end) in enumerate(phase_bounds):
        segment = normalized_signal[:, int(start) : int(end)]
        derivative = np.diff(segment, axis=1)
        available = derivative.shape[1]
        if available < 6:
            continue

        phase_max_lag = min(max_lag_samples, max(1, (available - 4) // 2))
        for edge_index_position in range(n_edges):
            source = int(edge_index[0, edge_index_position])
            target = int(edge_index[1, edge_index_position])
            best_abs = 0.0
            best_corr = 0.0
            best_lag = 0

            for lag in range(1, phase_max_lag + 1):
                current_corr = _pearson_correlation(
                    derivative[source, :-lag],
                    derivative[target, lag:],
                )
                current_abs = abs(current_corr)
                if current_abs > best_abs:
                    best_abs = current_abs
                    best_corr = current_corr
                    best_lag = lag

            correlation[phase_index, edge_index_position] = best_corr
            strength[phase_index, edge_index_position] = best_abs
            lag_samples[phase_index, edge_index_position] = best_lag

    active = strength >= correlation_threshold
    direction_confidence = np.zeros_like(strength, dtype=np.float32)
    edge_lookup = {
        (int(source), int(target)): position
        for position, (source, target) in enumerate(edge_index.T)
    }

    for phase_index in range(n_phases):
        for position, (source, target) in enumerate(edge_index.T):
            source = int(source)
            target = int(target)
            if source >= target:
                continue
            reverse = edge_lookup.get((target, source))
            if reverse is None:
                continue
            difference = float(
                strength[phase_index, position]
                - strength[phase_index, reverse]
            )
            direction_confidence[phase_index, position] = difference
            direction_confidence[phase_index, reverse] = -difference
            if abs(difference) >= direction_margin:
                if difference > 0:
                    active[phase_index, reverse] = False
                else:
                    active[phase_index, position] = False

        # 空间投影会使相邻节点天然高度相关。仅靠相关阈值通常仍会留下
        # 过密图，因此为每个目标节点保留最强的 K 条入边。K=0 可关闭剪枝。
        if top_k_incoming > 0:
            for target in range(n_nodes):
                incoming = np.flatnonzero(
                    active[phase_index] & (edge_index[1] == target)
                )
                if incoming.size <= top_k_incoming:
                    continue
                order = np.argsort(
                    strength[phase_index, incoming],
                    kind="stable",
                )
                active[phase_index, incoming[order[:-top_k_incoming]]] = False

    weight = np.zeros_like(strength, dtype=np.float32)
    adjacency = np.zeros((n_phases, n_nodes, n_nodes), dtype=np.float32)
    for phase_index in range(n_phases):
        for target in range(n_nodes):
            incoming = np.flatnonzero(
                active[phase_index] & (edge_index[1] == target)
            )
            if incoming.size == 0:
                continue
            incoming_strength = strength[phase_index, incoming]
            denominator = float(np.sum(incoming_strength))
            if denominator <= np.finfo(np.float32).eps:
                continue
            weight[phase_index, incoming] = incoming_strength / denominator

        adjacency[
            phase_index,
            edge_index[0],
            edge_index[1],
        ] = weight[phase_index]

    return DynamicEdgeData(
        weight=weight,
        correlation=correlation,
        strength=strength,
        lag_samples=lag_samples,
        active=active,
        direction_confidence=direction_confidence,
        adjacency=adjacency,
    )


def zero_pad_blocks(
    roi_signal: np.ndarray,
    bounds: np.ndarray,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将 8 个时间块保持原采样率地填充/截断为固定长度。

    输出：
      node_blocks: 9×8×L
      block_mask: 8×L，真实样本为 True，补零为 False
      original_lengths: 8
      used_bounds: 8×2，实际保留的原信号范围
      truncated: 8
    """
    if block_length <= 0:
        raise ValueError("block_length 必须大于 0。")

    n_nodes = roi_signal.shape[0]
    n_blocks = bounds.shape[0]
    node_blocks = np.zeros((n_nodes, n_blocks, block_length), dtype=np.float32)
    block_mask = np.zeros((n_blocks, block_length), dtype=bool)
    original_lengths = np.zeros(n_blocks, dtype=np.int32)
    used_bounds = np.zeros((n_blocks, 2), dtype=np.int32)
    truncated = np.zeros(n_blocks, dtype=bool)

    # 右对齐用于保留临近关键点的末端；左对齐用于保留关键点之后的起始部分。
    align_right = {0, 1, 6}

    for block_index, (start, end) in enumerate(bounds):
        segment = roi_signal[:, start:end]
        length = int(segment.shape[1])
        original_lengths[block_index] = length

        if length <= block_length:
            if block_index in align_right:
                destination_start = block_length - length
            else:
                destination_start = 0
            destination_end = destination_start + length
            node_blocks[:, block_index, destination_start:destination_end] = segment
            block_mask[block_index, destination_start:destination_end] = True
            used_bounds[block_index] = [start, end]
        else:
            truncated[block_index] = True
            if block_index in align_right:
                used_start = end - block_length
                used_end = end
            else:
                used_start = start
                used_end = start + block_length
            node_blocks[:, block_index, :] = roi_signal[:, used_start:used_end]
            block_mask[block_index, :] = True
            used_bounds[block_index] = [used_start, used_end]

    return node_blocks, block_mask, original_lengths, used_bounds, truncated


def save_node_block_plot(
    node_blocks: np.ndarray,
    block_mask: np.ndarray,
    used_bounds: np.ndarray,
    centers: np.ndarray,
    metadata: Mapping[str, str],
    npz_path: Path,
    landmark_source: str,
    plot_path: Path,
) -> None:
    """把 9 个空间节点的 8 个时间段绘制为一张 9×8 汇总图。"""
    node_blocks = np.asarray(node_blocks)
    block_mask = np.asarray(block_mask, dtype=bool)
    used_bounds = np.asarray(used_bounds, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float64)

    expected_shape = (9, len(BLOCK_NAMES))
    if node_blocks.ndim != 3 or node_blocks.shape[:2] != expected_shape:
        raise ValueError(
            f"node_blocks 应为 9×{len(BLOCK_NAMES)}×L，实际为 {node_blocks.shape}。"
        )
    if block_mask.shape != node_blocks.shape[1:]:
        raise ValueError(
            f"block_mask 形状应为 {node_blocks.shape[1:]}，实际为 {block_mask.shape}。"
        )
    if used_bounds.shape != (len(BLOCK_NAMES), 2):
        raise ValueError(f"used_bounds 应为 8×2，实际为 {used_bounds.shape}。")
    if centers.shape != (9, 2):
        raise ValueError(f"centers 应为 9×2，实际为 {centers.shape}。")

    title_parts = [
        metadata.get(key, "").strip()
        for key in ("group", "patient", "measurement")
        if metadata.get(key, "").strip()
    ]
    patient_title = " | ".join(title_parts) or safe_output_stem(npz_path)

    figure, axes = plt.subplots(
        9,
        len(BLOCK_NAMES),
        figsize=(24, 20),
        sharey="row",
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors

    try:
        for node_index in range(9):
            for block_index in range(len(BLOCK_NAMES)):
                axis = axes[node_index, block_index]
                valid = block_mask[block_index]
                values = np.asarray(
                    node_blocks[node_index, block_index, valid],
                    dtype=np.float64,
                )
                start, end = map(int, used_bounds[block_index])

                if values.size != end - start:
                    raise ValueError(
                        f"时间块 {block_index + 1} 的有效样本数 {values.size} "
                        f"与使用范围 [{start}, {end}) 不一致。"
                    )

                sample_indices = np.arange(start, end, dtype=np.int32)
                axis.plot(
                    sample_indices,
                    values,
                    color=colors[block_index % len(colors)],
                    linewidth=0.9,
                )
                axis.axhline(0.0, color="0.55", linewidth=0.45, alpha=0.7)
                axis.grid(True, linewidth=0.35, alpha=0.28)
                axis.tick_params(axis="both", labelsize=6)
                axis.locator_params(axis="x", nbins=3)
                axis.locator_params(axis="y", nbins=4)
                axis.margins(x=0.04, y=0.08)

                if node_index == 0:
                    axis.set_title(BLOCK_PLOT_TITLES[block_index], fontsize=8)
                if block_index == 0:
                    x_coord, y_coord = centers[node_index]
                    axis.set_ylabel(
                        f"Node {node_index + 1}\n({x_coord:.2f}, {y_coord:.2f})",
                        fontsize=8,
                    )
                if node_index == 8:
                    axis.set_xlabel("Sample", fontsize=7)

        figure.suptitle(
            f"{patient_title}\n"
            f"9 spatial nodes × 8 temporal blocks | landmarks: {landmark_source}",
            fontsize=15,
            y=0.995,
        )
        figure.tight_layout(rect=(0.025, 0.02, 1.0, 0.975), h_pad=0.7, w_pad=0.5)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_phase_node_plot(
    phase_blocks: np.ndarray,
    phase_source_samples: np.ndarray,
    centers: np.ndarray,
    dynamic_active: np.ndarray,
    metadata: Mapping[str, str],
    npz_path: Path,
    landmark_source: str,
    plot_path: Path,
) -> None:
    """绘制标准 ROI 节点 × 生理阶段的连续动态汇总图。"""
    phase_blocks = np.asarray(phase_blocks, dtype=np.float64)
    phase_source_samples = np.asarray(phase_source_samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    dynamic_active = np.asarray(dynamic_active, dtype=bool)

    if phase_blocks.ndim != 3 or phase_blocks.shape[1] != len(PHASE_NAMES):
        raise ValueError(
            f"phase_blocks 应为 N×{len(PHASE_NAMES)}×L，"
            f"实际为 {phase_blocks.shape}。"
        )
    n_nodes, n_phases, phase_length = phase_blocks.shape
    if phase_source_samples.shape != (n_phases, phase_length):
        raise ValueError("phase_source_samples 与 phase_blocks 不匹配。")
    if centers.shape != (n_nodes, 2):
        raise ValueError("centers 与 phase_blocks 的节点数不匹配。")
    if dynamic_active.shape[0] != n_phases:
        raise ValueError("dynamic_active 的阶段数不匹配。")

    title_parts = [
        metadata.get(key, "").strip()
        for key in ("group", "patient", "measurement")
        if metadata.get(key, "").strip()
    ]
    patient_title = " | ".join(title_parts) or safe_output_stem(npz_path)
    active_counts = np.count_nonzero(dynamic_active, axis=1)
    active_text = ", ".join(
        f"P{index + 1}:{int(count)}"
        for index, count in enumerate(active_counts)
    )

    figure, axes = plt.subplots(
        n_nodes,
        n_phases,
        figsize=(18, max(20, 1.55 * n_nodes)),
        sharey="row",
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors

    try:
        for node_index in range(n_nodes):
            for phase_index in range(n_phases):
                axis = axes[node_index, phase_index]
                axis.plot(
                    phase_source_samples[phase_index],
                    phase_blocks[node_index, phase_index],
                    color=colors[phase_index % len(colors)],
                    linewidth=0.9,
                )
                axis.axhline(0.0, color="0.55", linewidth=0.45, alpha=0.7)
                axis.grid(True, linewidth=0.35, alpha=0.28)
                axis.tick_params(axis="both", labelsize=6)
                axis.locator_params(axis="x", nbins=3)
                axis.locator_params(axis="y", nbins=4)
                axis.margins(x=0.04, y=0.08)

                if node_index == 0:
                    axis.set_title(PHASE_PLOT_TITLES[phase_index], fontsize=8)
                if phase_index == 0:
                    x_coord, y_coord = centers[node_index]
                    axis.set_ylabel(
                        f"Node {node_index + 1}\n({x_coord:.2f}, {y_coord:.2f})",
                        fontsize=8,
                    )
                if node_index == n_nodes - 1:
                    axis.set_xlabel("Original sample", fontsize=7)

        figure.suptitle(
            f"{patient_title}\n"
            f"{n_nodes} canonical spatial nodes × {n_phases} physiological phases | "
            f"landmarks: {landmark_source}\n"
            f"active directed edges by phase: {active_text}",
            fontsize=14,
            y=0.998,
        )
        figure.tight_layout(rect=(0.035, 0.015, 1.0, 0.975), h_pad=0.55, w_pad=0.45)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def safe_output_stem(npz_path: Path) -> str:
    suffix = "_BadChannelRepaired"
    if npz_path.stem.endswith(suffix):
        return npz_path.stem[: -len(suffix)]
    return npz_path.stem


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
        "ActiveChannelCount": np.nan,
        "CanonicalNodeCount": np.nan,
        "PhaseCount": np.nan,
        "PhaseOriginalLengths": "",
        "DynamicActiveEdgeCount": "",
        "ExcludedCornerChannels": "",
        "ROISigma": np.nan,
        "SamplingRateHz": np.nan,
        "BlockLength": np.nan,
        "LandmarkSource": "",
        "QRSOnset": np.nan,
        "RPeak": np.nan,
        "JPoint": np.nan,
        "TOnset": np.nan,
        "TPeak": np.nan,
        "TEnd": np.nan,
        "TruncatedBlocks": "",
        "Status": "FAILED",
        "ErrorMessage": "",
    }


def process_npz(
    npz_path: Path,
    output_dir: Path,
    options: NodeEdgeOptions,
    landmark_table: Mapping[str, dict[str, int]],
) -> dict[str, Any]:
    row = empty_summary_row(npz_path)

    try:
        loaded = load_repaired_npz(npz_path)
        signal_ct = loaded.signal_ct
        coordinates = loaded.coordinates
        n_channels, n_time = signal_ct.shape

        mask, excluded = active_channel_mask(
            signal_ct,
            policy=options.corner_policy,
            zero_tolerance=options.corner_zero_tolerance,
        )
        coordinates_norm = normalize_coordinates(coordinates, mask)
        centers = roi_centers(options.roi_side)
        projection = build_projection_matrix(
            coordinates_norm,
            centers,
            sigma=options.roi_sigma,
            active_mask=mask,
        )
        roi_signal = projection @ signal_ct

        landmarks, landmark_source = resolve_landmarks(
            npz_path=npz_path,
            roi_signal=roi_signal,
            sampling_rate_hz=options.sampling_rate_hz,
            landmark_table=landmark_table,
            enable_auto=options.enable_auto_landmarks,
        )
        phase_bounds = build_phase_bounds(
            landmarks,
            n_time=n_time,
            sampling_rate_hz=options.sampling_rate_hz,
        )
        normalized_roi_signal, baseline_offset, signal_scale = normalize_roi_signal(
            roi_signal,
            phase_bounds,
        )
        (
            phase_blocks,
            phase_feature_blocks,
            phase_source_samples,
            phase_original_lengths,
            phase_duration_ms,
        ) = resample_phase_blocks(
            normalized_roi_signal,
            phase_bounds,
            phase_length=options.block_length,
            sampling_rate_hz=options.sampling_rate_hz,
        )

        edge_index, edge_weight, edge_distance, adjacency = build_graph_edges(
            centers,
            edge_mode=options.edge_mode,
        )
        edge_delta = centers[edge_index[1]] - centers[edge_index[0]]
        distance_prior = np.exp(
            -(edge_distance.astype(np.float64) ** 2)
            / (2.0 * options.roi_sigma * options.roi_sigma)
        ).astype(np.float32)
        static_edge_attr = np.column_stack(
            [edge_distance, edge_delta[:, 0], edge_delta[:, 1], distance_prior]
        ).astype(np.float32)
        dynamic_edges = estimate_phase_dynamic_edges(
            normalized_signal=normalized_roi_signal,
            phase_bounds=phase_bounds,
            edge_index=edge_index,
            sampling_rate_hz=options.sampling_rate_hz,
            max_lag_ms=options.dynamic_max_lag_ms,
            correlation_threshold=options.dynamic_corr_threshold,
            direction_margin=options.dynamic_direction_margin,
            top_k_incoming=options.dynamic_top_k_incoming,
        )

        data_output_dir = output_dir / NODE_DATA_DIR_NAME
        plot_output_dir = output_dir / NODE_PLOT_DIR_NAME
        data_output_dir.mkdir(parents=True, exist_ok=True)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = safe_output_stem(npz_path)
        output_path = data_output_dir / f"{output_stem}_NodeEdge.npz"
        plot_path = plot_output_dir / f"{output_stem}_NodePhases.png"

        landmark_array = np.asarray(
            [landmarks[name] for name in LANDMARK_NAMES], dtype=np.int32
        )
        landmark_ms = landmark_array.astype(np.float64) * 1000.0 / options.sampling_rate_hz
        n_nodes = int(centers.shape[0])
        n_phases = len(PHASE_NAMES)
        phase_sequence = phase_blocks.reshape(n_nodes, -1)
        phase_feature_sequence = phase_feature_blocks.reshape(
            n_nodes,
            -1,
            len(NODE_FEATURE_NAMES),
        )
        phase_id = np.repeat(
            np.arange(n_phases, dtype=np.int16),
            options.block_length,
        )
        phase_progress = np.tile(
            np.linspace(0.0, 1.0, options.block_length, dtype=np.float32),
            n_phases,
        )
        node_labels = np.asarray(
            [
                f"R{index // options.roi_side + 1}C{index % options.roi_side + 1}"
                for index in range(n_nodes)
            ],
            dtype=str,
        )
        dynamic_active_counts = np.count_nonzero(
            dynamic_edges.active,
            axis=1,
        ).astype(np.int32)

        np.savez_compressed(
            output_path,
            # 标准节点与五阶段动态数据
            node_signal=roi_signal.astype(np.float32),               # N×T
            node_signal_normalized=normalized_roi_signal.astype(np.float32),
            node_blocks=phase_blocks,                                # N×5×L
            node_feature_blocks=phase_feature_blocks,                # N×5×L×3
            node_feature_sequence=phase_feature_sequence,            # N×(5L)×3
            node_features=phase_feature_sequence.reshape(n_nodes, -1),
            node_feature_names=np.asarray(NODE_FEATURE_NAMES, dtype=str),
            node_labels=node_labels,
            phase_signal=phase_sequence,
            phase_names=np.asarray(PHASE_NAMES, dtype=str),
            phase_bounds_samples=phase_bounds,
            phase_source_samples=phase_source_samples,
            phase_original_lengths=phase_original_lengths,
            phase_duration_ms=phase_duration_ms,
            phase_target_length=np.asarray(options.block_length, dtype=np.int32),
            phase_id=phase_id,
            phase_progress=phase_progress,
            baseline_offset=baseline_offset.astype(np.float32),
            signal_scale=np.asarray(signal_scale, dtype=np.float32),
            # 兼容旧读取字段；现在代表 5 个重采样阶段，不再含补零或截断。
            block_mask=np.ones((n_phases, options.block_length), dtype=bool),
            block_names=np.asarray(PHASE_NAMES, dtype=str),
            block_bounds_samples=phase_bounds,
            block_used_bounds_samples=phase_bounds,
            block_original_lengths=phase_original_lengths,
            block_truncated=np.zeros(n_phases, dtype=bool),
            block_length=np.asarray(options.block_length, dtype=np.int32),
            # ROI 投影
            projection_matrix=projection.astype(np.float32),         # N×C
            roi_centers=centers.astype(np.float32),                   # N×2
            roi_side=np.asarray(options.roi_side, dtype=np.int16),
            sensor_coordinates=coordinates.astype(np.float32),
            sensor_coordinates_norm=coordinates_norm.astype(np.float32),
            active_channel_mask=mask,
            excluded_corner_channels=(excluded + 1).astype(np.int32),
            roi_sigma=np.asarray(options.roi_sigma, dtype=np.float32),
            # 静态物理候选边
            edge_index=edge_index,                                   # 2×E
            edge_weight=edge_weight,
            edge_distance=edge_distance,
            adjacency=adjacency,
            edge_mode=np.asarray(options.edge_mode),
            static_edge_attr=static_edge_attr,
            static_edge_attr_names=np.asarray(
                STATIC_EDGE_ATTRIBUTE_NAMES,
                dtype=str,
            ),
            # 每位患者、每个生理阶段的有向动态边
            dynamic_edge_index=edge_index,
            dynamic_edge_weight=dynamic_edges.weight,                # 5×E
            dynamic_edge_correlation=dynamic_edges.correlation,
            dynamic_edge_strength=dynamic_edges.strength,
            dynamic_edge_lag_samples=dynamic_edges.lag_samples,
            dynamic_edge_lag_ms=(
                dynamic_edges.lag_samples.astype(np.float32)
                * 1000.0
                / options.sampling_rate_hz
            ),
            dynamic_edge_active=dynamic_edges.active,
            dynamic_edge_direction_confidence=(
                dynamic_edges.direction_confidence
            ),
            dynamic_adjacency=dynamic_edges.adjacency,                # 5×N×N
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
            # 关键点
            landmark_names=np.asarray(LANDMARK_NAMES, dtype=str),
            landmarks_samples=landmark_array,
            landmarks_ms=landmark_ms.astype(np.float32),
            landmark_source=np.asarray(landmark_source),
            sampling_rate_hz=np.asarray(options.sampling_rate_hz, dtype=np.float32),
            # 来源元数据
            source_npz=np.asarray(str(npz_path)),
            source_file=np.asarray(loaded.metadata.get("source_file", "")),
            group=np.asarray(loaded.metadata.get("group", "")),
            patient=np.asarray(loaded.metadata.get("patient", "")),
            measurement=np.asarray(loaded.metadata.get("measurement", "")),
        )

        save_phase_node_plot(
            phase_blocks=phase_blocks,
            phase_source_samples=phase_source_samples,
            centers=centers,
            dynamic_active=dynamic_edges.active,
            metadata=loaded.metadata,
            npz_path=npz_path,
            landmark_source=landmark_source,
            plot_path=plot_path,
        )

        row.update(
            {
                "OutputNPZ": str(output_path),
                "OutputPNG": str(plot_path),
                "Group": loaded.metadata.get("group", ""),
                "Patient": loaded.metadata.get("patient", ""),
                "Measurement": loaded.metadata.get("measurement", ""),
                "ChannelCount": n_channels,
                "TimePoints": n_time,
                "ActiveChannelCount": int(np.count_nonzero(mask)),
                "CanonicalNodeCount": n_nodes,
                "PhaseCount": n_phases,
                "PhaseOriginalLengths": ",".join(
                    map(str, phase_original_lengths)
                ),
                "DynamicActiveEdgeCount": ",".join(
                    map(str, dynamic_active_counts)
                ),
                "ExcludedCornerChannels": ",".join(map(str, excluded + 1)),
                "ROISigma": options.roi_sigma,
                "SamplingRateHz": options.sampling_rate_hz,
                "BlockLength": options.block_length,
                "LandmarkSource": landmark_source,
                "QRSOnset": landmarks["qrs_onset"],
                "RPeak": landmarks["r_peak"],
                "JPoint": landmarks["j_point"],
                "TOnset": landmarks["t_onset"],
                "TPeak": landmarks["t_peak"],
                "TEnd": landmarks["t_end"],
                "TruncatedBlocks": "",
                "Status": "OK",
            }
        )

    except Exception as exc:
        row["ErrorMessage"] = f"{type(exc).__name__}: {exc}"
        logging.exception("处理失败：%s", npz_path)

    return row


def discover_npz_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.rglob("*.npz")
        if path.is_file() and not path.name.endswith("_NodeEdge.npz")
    ]
    return sorted(files, key=natural_key)


def run_batch(
    input_dir: Path,
    output_dir: Path,
    options: NodeEdgeOptions,
    landmarks_csv: Path | None,
    landmark_unit: str,
) -> pd.DataFrame:
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在或不是文件夹：{input_dir}")

    files = discover_npz_files(input_dir)
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 NPZ 文件。")

    landmark_table = load_landmark_csv(
        landmarks_csv,
        unit=landmark_unit,
        sampling_rate_hz=options.sampling_rate_hz,
    )

    logging.info("共发现 %d 个修复后 NPZ 文件。", len(files))
    rows: list[dict[str, Any]] = []
    for index, npz_path in enumerate(files, start=1):
        logging.info("[%d/%d] %s", index, len(files), npz_path.name)
        rows.append(process_npz(npz_path, output_dir, options, landmark_table))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "NodeEdge_Summary.csv"
    summary = pd.DataFrame(rows, columns=list(empty_summary_row(Path("x.npz")).keys()))
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info("处理完成。节点边数据目录：%s", output_dir / NODE_DATA_DIR_NAME)
    logging.info("节点时间段图片目录：%s", output_dir / NODE_PLOT_DIR_NAME)
    logging.info("汇总表：%s", summary_path)
    return summary


def run_single(
    file_path: Path,
    output_dir: Path,
    options: NodeEdgeOptions,
    landmarks_csv: Path | None,
    landmark_unit: str,
) -> pd.DataFrame:
    file_path = file_path.expanduser()
    output_dir = output_dir.expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{file_path}")
    if file_path.suffix.lower() != ".npz":
        raise ValueError("单文件模式只支持 NPZ。")

    landmark_table = load_landmark_csv(
        landmarks_csv,
        unit=landmark_unit,
        sampling_rate_hz=options.sampling_rate_hz,
    )
    row = process_npz(file_path, output_dir, options, landmark_table)
    summary = pd.DataFrame([row], columns=list(empty_summary_row(file_path).keys()))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "NodeEdge_Summary.csv", index=False, encoding="utf-8-sig")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把 36/64 通道坏道修复数据映射为统一 4×4 标准节点、5 个生理阶段，"
            "并构造物理候选边和患者特异的分阶段有向动态边。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="修复后 NPZ 文件夹。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出根目录；节点边数据、节点时间段图片和汇总表分开保存。",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="只处理一个修复后的 NPZ 文件。",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="采样率 Hz；阶段中的 50/80ms 和动态边时延依赖该参数，默认 1000。",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.35,
        help="高斯软 ROI 的 sigma，默认 0.35。",
    )
    parser.add_argument(
        "--roi-side",
        type=int,
        choices=(3, 4, 5),
        default=4,
        help="标准 ROI 方阵边长，默认 4，即 16 个节点。",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=128,
        help="每个生理阶段重采样后的固定长度，默认 128。",
    )
    parser.add_argument(
        "--landmarks-csv",
        type=Path,
        default=None,
        help="可选关键点 CSV。未提供时默认自动启发式检测。",
    )
    parser.add_argument(
        "--landmark-unit",
        choices=("samples", "ms"),
        default="samples",
        help="关键点 CSV 中的单位，默认 samples。",
    )
    parser.add_argument(
        "--no-auto-landmarks",
        action="store_true",
        help="关闭自动关键点检测；缺少 NPZ/CSV 关键点时直接报错。",
    )
    parser.add_argument(
        "--corner-policy",
        choices=("zero-only", "always", "none"),
        default="zero-only",
        help="四角通道策略：默认仅排除全零顶角。",
    )
    parser.add_argument(
        "--corner-zero-tolerance",
        type=float,
        default=1e-10,
        help="判断顶角全零的绝对值容差。",
    )
    parser.add_argument(
        "--edge-mode",
        choices=("4", "8"),
        default="8",
        help="标准 ROI 图采用四邻接或八邻接，默认八邻接。",
    )
    parser.add_argument(
        "--dynamic-max-lag-ms",
        type=float,
        default=20.0,
        help="估计有向动态边时允许的最大正时延，默认 20 ms。",
    )
    parser.add_argument(
        "--dynamic-corr-threshold",
        type=float,
        default=0.25,
        help="动态边最小绝对滞后相关，默认 0.25。",
    )
    parser.add_argument(
        "--dynamic-direction-margin",
        type=float,
        default=0.05,
        help="正反方向强度差超过该值时只保留较强方向，默认 0.05。",
    )
    parser.add_argument(
        "--dynamic-top-k-incoming",
        type=int,
        default=2,
        help="每阶段每个目标节点最多保留的动态入边数；0 表示不剪枝，默认 2。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    options = NodeEdgeOptions(
        sampling_rate_hz=args.fs,
        roi_sigma=args.sigma,
        roi_side=args.roi_side,
        block_length=args.block_length,
        edge_mode=args.edge_mode,
        dynamic_max_lag_ms=args.dynamic_max_lag_ms,
        dynamic_corr_threshold=args.dynamic_corr_threshold,
        dynamic_direction_margin=args.dynamic_direction_margin,
        dynamic_top_k_incoming=args.dynamic_top_k_incoming,
        corner_policy=args.corner_policy,
        corner_zero_tolerance=args.corner_zero_tolerance,
        enable_auto_landmarks=not args.no_auto_landmarks,
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

        status_counts = result["Status"].value_counts(dropna=False).to_dict()
        logging.info("状态统计：%s", status_counts)
        return 0 if status_counts.get("FAILED", 0) == 0 else 2

    except Exception as exc:
        logging.error("程序运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
