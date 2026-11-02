from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = Path(r"D:\pycharmproject\GCNMCG\data\203")
OUTPUT_DIR_NAME = "203坏道处理后数据"
REPAIRED_DATA_DIR_NAME = "203修复后数据"
QC_IMAGE_DIR_NAME = "质检图片"
SUPPORTED_SUFFIXES = {".debfd", ".bfd"}
EXPECTED_SAMPLES = 1000
SUPPORTED_CHANNELS = (36, 64)


@dataclass(frozen=True)
class BadChannelOptions:
    """坏道检测与 IDW 插值参数。"""

    k_neighbors: int = 8
    distance_power: float = 2.0
    min_neighbors: int = 3
    low_amplitude_z: float = -3.5
    high_amplitude_z: float = 3.5
    jump_z: float = 3.5
    spatial_error_z: float = 3.5
    extreme_z: float = 4.0
    neighbor_corr_limit: float = 0.60
    min_soft_flags: int = 2
    invalid_ratio_limit: float = 0.01
    clipping_ratio_limit: float = 0.05
    flat_ratio_limit: float = 0.99




@dataclass(frozen=True)
class PatientInput:
    group: str
    patient: str
    measurement: str
    measurement_dir: Path
    file_path: Path


METRIC_COLUMNS = [
    "Channel",
    "MAD",
    "RobustRange",
    "DerivativeMAD",
    "FlatRatio",
    "ClippingRatio",
    "InvalidRatio",
    "NeighborCorrelation",
    "SpatialNRMSE",
    "AmplitudeRobustZ",
    "RangeRobustZ",
    "JumpRobustZ",
    "SpatialErrorRobustZ",
    "FlatFlag",
    "HighAmplitudeFlag",
    "JumpFlag",
    "SpatialFlag",
    "ClippingFlag",
    "BadScore",
    "IsBad",
]


def natural_key(value: str | Path) -> list[Any]:
    """用于文件夹/文件名称的自然排序，例如 2 排在 10 前面。"""
    text = str(value).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def sorted_directories(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name != OUTPUT_DIR_NAME),
        key=lambda p: natural_key(p.name),
    )


def find_average_files(folder: Path) -> list[Path]:
    """递归寻找平均心搏文件，优先 deBFD，再按自然顺序排序。"""
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(
        files,
        key=lambda p: (
            0 if p.suffix.lower() == ".debfd" else 1,
            natural_key(str(p.relative_to(folder))),
        ),
    )


def select_first_measurement(patient_dir: Path) -> tuple[Path, Path] | None:
    """
    返回（首次有效测量目录，首个平均心搏文件）。

    优先检查患者目录下按名称自然排序后的测量子目录；每个测量目录内部可继续
    包含更深层子目录。若患者目录下没有测量子目录，则兼容直接放置数据文件的情况。
    """
    measurement_dirs = sorted_directories(patient_dir)

    for measurement_dir in measurement_dirs:
        files = find_average_files(measurement_dir)
        if files:
            return measurement_dir, files[0]

    direct_files = [
        p
        for p in patient_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    direct_files = sorted(
        direct_files,
        key=lambda p: (0 if p.suffix.lower() == ".debfd" else 1, natural_key(p.name)),
    )
    if direct_files:
        return patient_dir, direct_files[0]

    return None


def discover_patient_inputs(data_root: Path) -> tuple[list[PatientInput], list[dict[str, str]]]:
    """按“组别/患者/测量”结构发现每位患者的首次测量文件。"""
    selected: list[PatientInput] = []
    skipped: list[dict[str, str]] = []

    for group_dir in sorted_directories(data_root):
        patient_dirs = sorted_directories(group_dir)

        # 兼容 data_root/患者/测量 的无组别结构。
        group_is_patient = select_first_measurement(group_dir)
        children_with_data = sum(
            select_first_measurement(p) is not None for p in patient_dirs
        )

        if group_is_patient is not None and children_with_data == 0:
            measurement_dir, file_path = group_is_patient
            selected.append(
                PatientInput(
                    group="未分组",
                    patient=group_dir.name,
                    measurement=measurement_dir.name,
                    measurement_dir=measurement_dir,
                    file_path=file_path,
                )
            )
            continue

        for patient_dir in patient_dirs:
            result = select_first_measurement(patient_dir)
            if result is None:
                skipped.append(
                    {
                        "Group": group_dir.name,
                        "Patient": patient_dir.name,
                        "Reason": "未找到 .deBFD/.BFD 文件",
                    }
                )
                continue

            measurement_dir, file_path = result
            selected.append(
                PatientInput(
                    group=group_dir.name,
                    patient=patient_dir.name,
                    measurement=measurement_dir.name,
                    measurement_dir=measurement_dir,
                    file_path=file_path,
                )
            )

    return selected, skipped


def read_average_bfd(file_path: Path, samples: int = EXPECTED_SAMPLES) -> np.ndarray:
    """
    读取平均心搏文件，返回 [采样点, 通道]，与 MATLAB 版本一致。

    对 .BFD 文件先按 float32 读取并剔除可能存在的 512 个 float 文件头，
    再转换为 float64。这样可避免文件头中的特殊位模式在转换时产生
    ``invalid value encountered in cast`` 警告。
    """
    suffix = file_path.suffix.lower()

    if suffix == ".debfd":
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            raw_values = np.fromstring(text, sep=" ", dtype=np.float64)
        except OSError as exc:
            raise OSError(f"无法读取文本文件：{file_path}") from exc

    elif suffix == ".bfd":
        try:
            # MATLAB fread(..., 'float') 对应 32 位浮点；Windows 数据通常为小端。
            # 暂不转换为 float64，先判断并剔除可能的文件头。
            raw_values = np.fromfile(file_path, dtype="<f4")
        except OSError as exc:
            raise OSError(f"无法读取二进制文件：{file_path}") from exc

    else:
        raise ValueError(f"不支持的文件类型：{file_path.suffix}")

    if raw_values.size == 0:
        raise ValueError(f"文件为空或未解析出数值：{file_path}")

    original_value_count = int(raw_values.size)
    remainder = original_value_count % samples

    if remainder == 512:
        # 先去掉 512 个 float 文件头，再进行 float64 转换。
        raw_values = raw_values[512:]
        logging.debug("已剔除 .BFD 文件头：%s（512 个 float）", file_path)
    elif remainder != 0:
        raise ValueError(
            f"数据长度 {original_value_count} 不能按 {samples} 个采样点整形，"
            "且不符合 512 个 float 头的格式。"
        )

    if suffix == ".bfd":
        # 部分旧文件可能含 signaling NaN 等特殊位模式。
        # 文件头已经先剔除；转换后再明确检查有效信号区域。
        with np.errstate(invalid="ignore", over="ignore"):
            values = raw_values.astype(np.float64, copy=False)
    else:
        values = np.asarray(raw_values, dtype=np.float64)

    invalid_count = int(np.count_nonzero(~np.isfinite(values)))
    if invalid_count > 0:
        logging.warning(
            "文件 %s 的有效信号区域含有 %d 个 NaN/Inf；"
            "程序将按通道统计 InvalidRatio，并在坏道处理中替换这些无效值。",
            file_path,
            invalid_count,
        )

    if values.size % samples != 0:
        raise ValueError(
            f"去除文件头后数据长度 {values.size} 仍不能按 {samples} 个采样点整形。"
        )

    # MATLAB reshape(data, samples, []) 使用列优先顺序。
    data = values.reshape((samples, -1), order="F")
    return data


def default_coordinates(n_channels: int) -> np.ndarray:
    n_side = int(round(math.sqrt(n_channels)))
    if n_side * n_side != n_channels:
        raise ValueError(f"通道数 {n_channels} 不能构成规则方阵。")

    coordinates = np.zeros((n_channels, 2), dtype=np.float64)
    for ch in range(n_channels):
        row = ch // n_side
        col = ch % n_side
        coordinates[ch] = [col + 1, row + 1]
    return coordinates


def coordinate_search_dirs(file_path: Path, data_root: Path) -> Iterable[Path]:
    """从数据文件所在目录逐级向上查找坐标文件，最后检查 data_root。"""
    seen: set[Path] = set()
    current = file_path.parent

    while True:
        resolved = current.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield current
        if current == data_root or current.parent == current:
            break
        current = current.parent

    root_resolved = data_root.resolve()
    if root_resolved not in seen:
        yield data_root


def _read_coordinate_csv(csv_path: Path) -> np.ndarray:
    try:
        coordinates = np.loadtxt(csv_path, delimiter=",")
    except ValueError:
        # 兼容带表头的 CSV。
        coordinates = pd.read_csv(csv_path).iloc[:, :2].to_numpy(dtype=float)
    return np.asarray(coordinates, dtype=np.float64)


def load_coordinates(file_path: Path, data_root: Path, n_channels: int) -> np.ndarray:
    csv_name = f"sensor_coordinates_{n_channels}.csv"
    mat_name = f"sensor_coordinates_{n_channels}.mat"

    for search_dir in coordinate_search_dirs(file_path, data_root):
        csv_path = search_dir / csv_name
        mat_path = search_dir / mat_name

        if csv_path.is_file():
            coordinates = _read_coordinate_csv(csv_path)
            break

        if mat_path.is_file():
            mat_data = loadmat(mat_path)
            if "coordinates" not in mat_data:
                raise ValueError(f"{mat_path} 中未找到变量 coordinates。")
            coordinates = np.asarray(mat_data["coordinates"], dtype=np.float64)
            break
    else:
        coordinates = default_coordinates(n_channels)

    if coordinates.shape != (n_channels, 2):
        raise ValueError(
            f"传感器坐标必须为 {n_channels}×2，实际为 {coordinates.shape}。"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("传感器坐标中包含 NaN 或 Inf。")

    return coordinates


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    z = np.full(values.shape, np.nan, dtype=np.float64)

    if not np.any(finite):
        return z

    finite_values = values[finite]
    center = np.median(finite_values)
    scale = 1.4826 * np.median(np.abs(finite_values - center))

    if scale <= np.finfo(float).eps:
        standard_scale = np.std(finite_values, ddof=0)
        if standard_scale <= np.finfo(float).eps:
            z[finite] = 0.0
            return z
        scale = standard_scale

    z[finite] = (finite_values - center) / scale
    return z


def repair_bad_channels(
    x: np.ndarray,
    coordinates: np.ndarray | None = None,
    options: BadChannelOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    对 C×T 平均心搏矩阵进行坏道检测和 IDW 空间插值，忽略四个顶角。
    """
    options = options or BadChannelOptions()
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("X 必须是二维矩阵：通道数×时间点。")

    n_channels, n_time = x.shape
    if n_channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"当前程序仅支持 36 或 64 通道，实际为 {n_channels}。")
    if n_time < 2:
        raise ValueError("每个通道至少需要 2 个时间点。")

    if coordinates is None:
        coordinates = default_coordinates(n_channels)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (n_channels, 2):
        raise ValueError(f"coordinates 必须为 {n_channels}×2。")

    invalid_ratio = np.mean(~np.isfinite(x), axis=1)
    x_clean = x.copy()

    for ch in range(n_channels):
        valid = np.isfinite(x_clean[ch])
        if not np.any(valid):
            x_clean[ch] = 0.0
        else:
            replacement = np.median(x_clean[ch, valid])
            x_clean[ch, ~valid] = replacement

    channel_median = np.median(x_clean, axis=1, keepdims=True)
    x_center = x_clean - channel_median

    channel_mad = np.median(
        np.abs(x_center - np.median(x_center, axis=1, keepdims=True)), axis=1
    )
    p01 = np.percentile(x_clean, 1, axis=1, method="linear")
    p99 = np.percentile(x_clean, 99, axis=1, method="linear")
    robust_range = p99 - p01

    dx = np.diff(x_clean, axis=1)
    derivative_mad = np.median(
        np.abs(dx - np.median(dx, axis=1, keepdims=True)), axis=1
    )

    diff_reference = np.median(np.abs(dx))
    flat_tolerance = max(diff_reference * 1e-3, np.finfo(float).eps)
    flat_ratio = np.mean(np.abs(dx) <= flat_tolerance, axis=1)

    x_min = np.min(x_clean, axis=1)
    x_max = np.max(x_clean, axis=1)
    clipping_ratio = np.ones(n_channels, dtype=np.float64)
    non_constant = x_max > x_min
    clipping_ratio[non_constant] = (
        np.sum(x_clean[non_constant] == x_min[non_constant, None], axis=1)
        + np.sum(x_clean[non_constant] == x_max[non_constant, None], axis=1)
    ) / n_time

    eps = np.finfo(float).eps
    z_amplitude = robust_z(np.log10(channel_mad + eps))
    z_range = robust_z(np.log10(robust_range + eps))
    z_jump = robust_z(np.log10(derivative_mad + eps))

    median_range = np.median(robust_range)
    absolute_flat = robust_range <= max(median_range * 1e-4, eps)

    invalid_flag = invalid_ratio > options.invalid_ratio_limit
    flat_flag = (
        absolute_flat
        | (flat_ratio > options.flat_ratio_limit)
        | (z_amplitude < options.low_amplitude_z)
        | (z_range < options.low_amplitude_z)
    )
    high_amplitude_flag = (
        (z_amplitude > options.high_amplitude_z)
        | (z_range > options.high_amplitude_z)
    )
    jump_flag = z_jump > options.jump_z
    clipping_flag = clipping_ratio > options.clipping_ratio_limit

    preliminary_bad = (
        invalid_flag
        | absolute_flat
        | clipping_flag
        | high_amplitude_flag
        | jump_flag
    )

    neighbor_corr = np.full(n_channels, np.nan, dtype=np.float64)
    spatial_nrmse = np.full(n_channels, np.nan, dtype=np.float64)

    for ch in range(n_channels):
        candidates = np.flatnonzero(~preliminary_bad)
        candidates = candidates[candidates != ch]

        if candidates.size < options.min_neighbors:
            candidates = np.array(
                [idx for idx in range(n_channels) if idx != ch], dtype=int
            )

        distances = np.linalg.norm(coordinates[candidates] - coordinates[ch], axis=1)
        order = np.argsort(distances, kind="stable")
        candidates = candidates[order]
        distances = distances[order]

        k = min(options.k_neighbors, candidates.size)
        if k < options.min_neighbors:
            continue

        neighbors = candidates[:k]
        selected_distances = distances[:k]
        raw_weights = 1.0 / np.power(selected_distances + eps, options.distance_power)
        weights = raw_weights / np.sum(raw_weights)

        predicted = weights @ x_clean[neighbors]
        observed = x_clean[ch]

        if np.std(observed) > eps and np.std(predicted) > eps:
            neighbor_corr[ch] = np.corrcoef(observed, predicted)[0, 1]

        scale = robust_range[ch]
        if scale <= eps:
            scale = median_range + eps

        spatial_nrmse[ch] = np.sqrt(np.mean((observed - predicted) ** 2)) / scale

    z_spatial_error = robust_z(np.log10(spatial_nrmse + eps))
    spatial_flag = (
        (z_spatial_error > options.spatial_error_z)
        & (np.abs(neighbor_corr) < options.neighbor_corr_limit)
    )

    hard_bad = (
        invalid_flag
        | absolute_flat
        | clipping_flag
        | (flat_ratio > options.flat_ratio_limit)
    )

    bad_score = (
        flat_flag.astype(int)
        + high_amplitude_flag.astype(int)
        + jump_flag.astype(int)
        + spatial_flag.astype(int)
        + clipping_flag.astype(int)
    )

    # 普通异常仍需至少两项指标同时命中；但单项指标达到更严格的极端阈值时，
    # 也直接判为坏道。这样可修复明显的孤立异常，同时避免放大边缘波动。
    extreme_bad = (
        (np.abs(z_amplitude) > options.extreme_z)
        | (np.abs(z_range) > options.extreme_z)
        | (z_jump > options.extreme_z)
        | (
            (z_spatial_error > options.extreme_z)
            & (np.abs(neighbor_corr) < options.neighbor_corr_limit)
        )
    )
    is_bad = hard_bad | extreme_bad | (bad_score >= options.min_soft_flags)

    # ==========================
    # 核心修改点：剔除四个顶角通道
    # ==========================
    n_side = int(round(math.sqrt(n_channels)))
    corner_indices = np.array([0, n_side - 1, n_channels - n_side, n_channels - 1], dtype=int)
    
    # 强制让 4 个顶角通道不被判定为坏道（不参与修复）
    is_bad[corner_indices] = False
    
    bad_indices = np.flatnonzero(is_bad)
    
    x_repaired = x_clean.copy()
    source_data = x_clean.copy()
    
    # 选取有效通道时，同时排除本身也是恒为 0 的顶角通道，防止它们参与修复别人
    good_indices = np.setdiff1d(np.flatnonzero(~is_bad), corner_indices)
    # ==========================

    if bad_indices.size > 0 and good_indices.size < options.min_neighbors:
        raise ValueError("有效通道过少，无法进行可靠空间插值。")

    for bad_ch in bad_indices:
        distances = np.linalg.norm(
            coordinates[good_indices] - coordinates[bad_ch], axis=1
        )
        order = np.argsort(distances, kind="stable")
        sorted_good = good_indices[order]
        distances = distances[order]

        k = min(options.k_neighbors, sorted_good.size)
        if k < options.min_neighbors:
            raise ValueError(f"坏道 {bad_ch + 1} 附近有效通道不足。")

        neighbors = sorted_good[:k]
        selected_distances = distances[:k]
        raw_weights = 1.0 / np.power(selected_distances + eps, options.distance_power)
        weights = raw_weights / np.sum(raw_weights)

        # 只使用原始有效通道，不递归使用已插值坏道。
        x_repaired[bad_ch] = weights @ source_data[neighbors]

    metrics = pd.DataFrame(
        {
            "Channel": np.arange(1, n_channels + 1, dtype=int),
            "MAD": channel_mad,
            "RobustRange": robust_range,
            "DerivativeMAD": derivative_mad,
            "FlatRatio": flat_ratio,
            "ClippingRatio": clipping_ratio,
            "InvalidRatio": invalid_ratio,
            "NeighborCorrelation": neighbor_corr,
            "SpatialNRMSE": spatial_nrmse,
            "AmplitudeRobustZ": z_amplitude,
            "RangeRobustZ": z_range,
            "JumpRobustZ": z_jump,
            "SpatialErrorRobustZ": z_spatial_error,
            "FlatFlag": flat_flag,
            "HighAmplitudeFlag": high_amplitude_flag,
            "JumpFlag": jump_flag,
            "SpatialFlag": spatial_flag,
            "ClippingFlag": clipping_flag,
            "BadScore": bad_score,
            "IsBad": is_bad,
        },
        columns=METRIC_COLUMNS,
    )

    return x_repaired, bad_indices, metrics


def largest_bad_cluster(bad_indices_zero_based: Sequence[int], n_channels: int) -> int:
    """计算规则方阵中四邻接坏道簇的最大大小。"""
    if len(bad_indices_zero_based) == 0:
        return 0

    n_side = int(round(math.sqrt(n_channels)))
    bad_mask = np.zeros((n_side, n_side), dtype=bool)
    for ch in bad_indices_zero_based:
        row, col = divmod(int(ch), n_side)
        bad_mask[row, col] = True

    visited = np.zeros_like(bad_mask)
    largest = 0
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row in range(n_side):
        for col in range(n_side):
            if not bad_mask[row, col] or visited[row, col]:
                continue

            queue = [(row, col)]
            visited[row, col] = True
            count = 0

            while queue:
                current_row, current_col = queue.pop(0)
                count += 1

                for dr, dc in offsets:
                    nr, nc = current_row + dr, current_col + dc
                    if (
                        0 <= nr < n_side
                        and 0 <= nc < n_side
                        and bad_mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            largest = max(largest, count)

    return largest


def save_qc_plot(
    x: np.ndarray,
    x_repaired: np.ndarray,
    metrics: pd.DataFrame,
    bad_indices: np.ndarray,
    plot_path: Path,
) -> None:
    n_channels, n_time = x.shape
    n_side = int(round(math.sqrt(n_channels)))
    time = np.arange(1, n_time + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    axes[0, 0].plot(time, x.T, linewidth=0.5)
    axes[0, 0].set_title("Before repair: average-beat butterfly plot")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].plot(time, x_repaired.T, linewidth=0.5)
    axes[0, 1].set_title("After IDW bad-channel repair")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)

    axes[1, 0].bar(metrics["Channel"], metrics["BadScore"])
    if bad_indices.size:
        bad_channels_one_based = bad_indices + 1
        axes[1, 0].scatter(
            bad_channels_one_based,
            metrics.loc[bad_indices, "BadScore"],
            s=50,
        )
    axes[1, 0].set_title("Channel bad score")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("Bad score")
    axes[1, 0].grid(True)

    mask_grid = np.zeros((n_side, n_side), dtype=int)
    for ch in bad_indices:
        row, col = divmod(int(ch), n_side)
        mask_grid[row, col] = 1

    image = axes[1, 1].imshow(mask_grid, origin="lower", aspect="equal")
    axes[1, 1].set_title("Bad-channel location (1 = bad)")
    axes[1, 1].set_xlabel("Array column")
    axes[1, 1].set_ylabel("Array row")
    fig.colorbar(image, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_component(name: str) -> str:
    """清理 Windows 文件名中不允许的字符。"""
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name).strip().rstrip(".")
    return cleaned or "unnamed"


def empty_summary_row(
    group: str,
    patient: str,
    measurement: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    """生成固定列顺序的简洁质控汇总行。"""
    return {
        "Group": group,
        "Patient": patient,
        "Measurement": measurement,
        "FileName": file_name,
        "ChannelCount": np.nan,
        "SampleCount": np.nan,
        "HasBadChannel": "",
        "BadCount": np.nan,
        "BadChannels": "",
        "BadRatio": np.nan,
        "BadRatePercent": np.nan,
        "LargestCluster": np.nan,
        "QCStatus": "FAILED",
        "OutputNPZ": "",
        "OutputPNG": "",
        "ErrorMessage": "",
    }


def process_patient(
    patient_input: PatientInput,
    data_root: Path,
    output_root: Path,
    options: BadChannelOptions,
) -> dict[str, Any]:
    """
    处理一位患者。

    NPZ 统一保存在“修复后数据”文件夹；
    PNG 统一保存在“质检图片”文件夹。
    """
    file_path = patient_input.file_path

    repaired_data_dir = output_root / REPAIRED_DATA_DIR_NAME
    qc_image_dir = output_root / QC_IMAGE_DIR_NAME
    repaired_data_dir.mkdir(parents=True, exist_ok=True)
    qc_image_dir.mkdir(parents=True, exist_ok=True)

    row = empty_summary_row(
        group=patient_input.group,
        patient=patient_input.patient,
        measurement=patient_input.measurement,
        file_name=file_path.name,
    )

    try:
        original = read_average_bfd(file_path)
        n_samples, n_channels = original.shape

        if n_samples != EXPECTED_SAMPLES:
            logging.warning(
                "%s 的采样点数为 %d，不是预期的 %d。",
                file_path,
                n_samples,
                EXPECTED_SAMPLES,
            )
        if n_channels not in SUPPORTED_CHANNELS:
            raise ValueError(
                f"通道数为 {n_channels}；当前程序仅按 36 或 64 通道二维阵列处理。"
            )

        x = original.T
        coordinates = load_coordinates(file_path, data_root, n_channels)
        x_repaired, bad_indices, metrics = repair_bad_channels(
            x, coordinates, options
        )
        repaired = x_repaired.T

        bad_mask = np.zeros(n_channels, dtype=bool)
        bad_mask[bad_indices] = True
        bad_count = int(bad_indices.size)
        
        # 核心修改点：分母去除这 4 个不存在的传感器
        valid_channel_count = n_channels - 4
        bad_ratio = bad_count / valid_channel_count
        largest_cluster = largest_bad_cluster(bad_indices, n_channels)

        if bad_ratio > 0.15 or largest_cluster >= 4:
            qc_status = "REVIEW"
        elif bad_count > 0:
            qc_status = "REPAIRED"
        else:
            qc_status = "PASS"

        # 核心修改点：由于所有文件全放在同一个目录中，为防止文件名冲突，在开头加入了 group 名称
        base_name = "__".join(
            [
                safe_component(patient_input.group),
                safe_component(patient_input.patient),
                safe_component(patient_input.measurement),
                safe_component(file_path.stem),
            ]
        )
        npz_path = repaired_data_dir / f"{base_name}_BadChannelRepaired.npz"
        plot_path = qc_image_dir / f"{base_name}_BadChannelQC.png"

        np.savez_compressed(
            npz_path,
            data_repaired=repaired,
            bad_channels=(bad_indices + 1).astype(np.int32),
            bad_mask=bad_mask,
            coordinates=coordinates,
            source_file=np.asarray(str(file_path)),
            group=np.asarray(patient_input.group),
            patient=np.asarray(patient_input.patient),
            measurement=np.asarray(patient_input.measurement),
        )

        save_qc_plot(x, x_repaired, metrics, bad_indices, plot_path)

        row.update(
            {
                "ChannelCount": n_channels,
                "SampleCount": n_samples,
                "HasBadChannel": "YES" if bad_count > 0 else "NO",
                "BadCount": bad_count,
                "BadChannels": ",".join(map(str, bad_indices + 1)),
                "BadRatio": round(bad_ratio, 6),
                "BadRatePercent": round(bad_ratio * 100.0, 2),
                "LargestCluster": largest_cluster,
                "QCStatus": qc_status,
                "OutputNPZ": str(npz_path),
                "OutputPNG": str(plot_path),
            }
        )

    except Exception as exc:  # 保证单个患者失败时批处理继续。
        row["ErrorMessage"] = f"{type(exc).__name__}: {exc}"
        logging.exception(
            "患者处理失败：组别=%s，患者=%s，文件=%s",
            patient_input.group,
            patient_input.patient,
            file_path,
        )

    return row


def save_summary(
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    output_root: Path,
) -> Path:
    """只保存一份简洁 CSV；跳过的患者也写入同一张表。"""
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = list(rows)

    for item in skipped:
        skipped_row = empty_summary_row(
            group=item.get("Group", ""),
            patient=item.get("Patient", ""),
        )
        skipped_row["QCStatus"] = "SKIPPED"
        skipped_row["ErrorMessage"] = item.get("Reason", "未找到可处理文件")
        all_rows.append(skipped_row)

    summary = pd.DataFrame(all_rows, columns=list(empty_summary_row("", "").keys()))
    csv_path = output_root / "BadChannel_QC_Summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def run_batch(
    data_root: Path,
    options: BadChannelOptions | None = None,
) -> pd.DataFrame:
    options = options or BadChannelOptions()
    data_root = data_root.expanduser()

    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"不是文件夹：{data_root}")

    output_root = data_root / OUTPUT_DIR_NAME
    (output_root / REPAIRED_DATA_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (output_root / QC_IMAGE_DIR_NAME).mkdir(parents=True, exist_ok=True)

    patient_inputs, skipped = discover_patient_inputs(data_root)

    if not patient_inputs:
        raise FileNotFoundError(
            f"在 {data_root} 的组别/患者/测量目录中未找到 .deBFD 或 .BFD 文件。"
        )

    logging.info("共发现 %d 位可处理患者；每位患者只处理首次有效测量。", len(patient_inputs))

    rows: list[dict[str, Any]] = []
    total = len(patient_inputs)
    for index, patient_input in enumerate(patient_inputs, start=1):
        logging.info(
            "[%d/%d] 组别=%s，患者=%s，首次测量=%s，文件=%s",
            index,
            total,
            patient_input.group,
            patient_input.patient,
            patient_input.measurement,
            patient_input.file_path.name,
        )
        rows.append(
            process_patient(
                patient_input=patient_input,
                data_root=data_root,
                output_root=output_root,
                options=options,
            )
        )

    csv_path = save_summary(rows, skipped, output_root)
    logging.info("处理完成。简洁质控汇总：%s", csv_path)
    logging.info("修复后数据目录：%s", output_root / REPAIRED_DATA_DIR_NAME)
    logging.info("质检图片目录：%s", output_root / QC_IMAGE_DIR_NAME)

    return pd.DataFrame(rows)


def run_single(
    file_path: Path,
    data_root: Path | None = None,
) -> pd.DataFrame:
    file_path = file_path.expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("单文件模式仅支持 .deBFD/.BFD。")

    root = (data_root or file_path.parent).expanduser()
    output_root = root / OUTPUT_DIR_NAME
    (output_root / REPAIRED_DATA_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (output_root / QC_IMAGE_DIR_NAME).mkdir(parents=True, exist_ok=True)

    patient_input = PatientInput(
        group="single",
        patient=file_path.parent.name,
        measurement=file_path.parent.name,
        measurement_dir=file_path.parent,
        file_path=file_path,
    )
    row = process_patient(
        patient_input=patient_input,
        data_root=root,
        output_root=output_root,
        options=BadChannelOptions(),
    )
    save_summary([row], [], output_root)
    return pd.DataFrame([row])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MCG 平均心搏坏道检测与 IDW 修复；NPZ 与质检 PNG 分文件夹保存。"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=r"数据根目录；批处理默认 E:\GCNMCG\data，单文件默认使用文件所在目录",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="仅处理一个 .deBFD/.BFD 文件。",
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

    try:
        if args.single_file is not None:
            result = run_single(
                file_path=args.single_file,
                data_root=args.data_root,
            )
        else:
            result = run_batch(data_root=args.data_root or DEFAULT_DATA_ROOT)

        status_counts = result["QCStatus"].value_counts(dropna=False).to_dict()
        logging.info("质控状态统计：%s", status_counts)
        return 0

    except Exception as exc:
        logging.error("程序运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
