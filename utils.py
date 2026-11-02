from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOG = logging.getLogger("gnn_mcg")


@dataclass(frozen=True)
class SampleMeta:
    """Everything needed to locate and label a single MCG sample."""

    npz_path: Path
    sample_id: str          # matches the ID column in all_data_labels.xlsx
    patient_key: str        # used for patient-level CV split
    group: str              # health / NSTEMI / CAGNSTEMI / noNSTEMI
    non_nstemi_subcat: str  # only meaningful when group == "noNSTEMI"
    label: int              # 0 health / 1 NSTEMI / 2 non-NSTEMI
    cag_available: bool     # True when CAG info is available
    channel_count: int      # 36 or 64


# ---------------------------------------------------------------------------
# Filename parsing


def _strip_repaired_suffix(stem: str) -> str:
    suffix = "_BadChannelRepaired"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def parse_repaired_filename(npz_path: Path) -> dict[str, str]:
    """
    Parse names like:
      health__2023072601-王国泰__20240104 104609__20240104 104609_BadChannelRepaired.npz
      noNSTEMI__STEMI__37001ACS00160-郑艳珍__20240109 143246_BadChannelRepaired.npz
    """
    stem = _strip_repaired_suffix(npz_path.stem)
    parts = stem.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unrecognized filename layout: {npz_path.name}")

    group = parts[0]
    if group == "noNSTEMI":
        subcat = parts[1]
        patient_field = parts[2]
        measurement = "__".join(parts[3:]) if len(parts) >= 4 else ""
    else:
        subcat = ""
        patient_field = parts[1]
        measurement = "__".join(parts[2:])

    # Patient field usually looks like "<ID>-<name>" or "<ID>"
    sample_id_match = re.match(r"^([0-9A-Za-z]+)", patient_field.strip())
    sample_id = sample_id_match.group(1) if sample_id_match else patient_field

    return {
        "group": group,
        "subcat": subcat,
        "sample_id": sample_id,
        "patient_key": patient_field,
        "measurement": measurement,
    }


# ---------------------------------------------------------------------------
# Node-edge output resolution


def resolve_node_edge_path(repaired_path: Path, node_edge_dir: Path) -> Path:
    """Map a repaired NPZ to its node_edge NPZ counterpart."""
    stem = _strip_repaired_suffix(repaired_path.stem)
    return node_edge_dir / f"{stem}_NodeEdge.npz"


# ---------------------------------------------------------------------------
# Label lookup


class LabelTable:
    """Fast lookup by sample id -> (label, cag) using all_data_labels.xlsx."""

    def __init__(self, xlsx_path: Path):
        table = pd.read_excel(xlsx_path)
        table.columns = [c.strip() for c in table.columns]
        for required in ("ID", "Label", "CAG"):
            if required not in table.columns:
                raise ValueError(f"Label xlsx missing column {required}")

        table["ID"] = table["ID"].astype(str).str.strip()
        self._map: dict[str, tuple[int, int]] = {}
        for _, row in table.iterrows():
            key = str(row["ID"]).strip()
            if not key or key.lower() == "nan":
                continue
            self._map[key] = (int(row["Label"]), int(row["CAG"]))
            # Common numeric zero-padding mismatch: also index without leading zeros.
            stripped = key.lstrip("0")
            if stripped and stripped != key:
                self._map.setdefault(stripped, self._map[key])

    def lookup(self, sample_id: str) -> tuple[int, int] | None:
        sample_id = sample_id.strip()
        if sample_id in self._map:
            return self._map[sample_id]
        stripped = sample_id.lstrip("0")
        return self._map.get(stripped)


# ---------------------------------------------------------------------------
# Sample discovery


def discover_samples(
    repaired_dir: Path,
    node_edge_dir: Path,
    labels: LabelTable,
    require_node_edge: bool = True,
) -> list[SampleMeta]:
    if not repaired_dir.is_dir():
        raise NotADirectoryError(f"Repaired directory missing: {repaired_dir}")

    samples: list[SampleMeta] = []
    skipped: dict[str, int] = {}

    def bump(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for npz_path in sorted(repaired_dir.glob("*.npz")):
        try:
            parsed = parse_repaired_filename(npz_path)
        except ValueError as exc:
            LOG.warning("Skip %s: %s", npz_path.name, exc)
            bump("bad_filename")
            continue

        node_edge_path = resolve_node_edge_path(npz_path, node_edge_dir)
        if require_node_edge and not node_edge_path.is_file():
            bump("missing_node_edge_npz")
            continue

        info = labels.lookup(parsed["sample_id"])
        if info is None:
            bump("no_label")
            continue
        label, cag = info

        # Sanity-check group vs label
        group = parsed["group"]
        if group == "health" and label != 0:
            bump("group_label_mismatch")
            continue
        if group in ("NSTEMI", "CAGNSTEMI") and label != 1:
            bump("group_label_mismatch")
            continue
        if group == "noNSTEMI" and label != 2:
            bump("group_label_mismatch")
            continue

        # Channel count is fully determined by group in this dataset.
        if group in ("health", "NSTEMI"):
            channel_count = 36
        else:
            channel_count = 64

        samples.append(
            SampleMeta(
                npz_path=node_edge_path,
                sample_id=parsed["sample_id"],
                patient_key=parsed["patient_key"],
                group=group,
                non_nstemi_subcat=parsed["subcat"],
                label=label,
                cag_available=bool(cag),
                channel_count=channel_count,
            )
        )

    if skipped:
        LOG.warning("Sample discovery skipped: %s", skipped)
    LOG.info("Discovered %d usable samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Cohort filters for the different experiments


def filter_e1(samples: Iterable[SampleMeta]) -> list[SampleMeta]:
    """Healthy (0) vs NSTEMI (1) on the 36-channel cohort."""
    picked = [
        s for s in samples
        if s.label in (0, 1) and s.channel_count == 36
    ]
    return picked


def filter_e2_binary(samples: Iterable[SampleMeta]) -> list[SampleMeta]:
    """NSTEMI (label 1) vs non-NSTEMI (label 2). 64-channel cohort."""
    picked = [
        s for s in samples
        if s.label in (1, 2) and s.channel_count == 64
    ]
    return picked


def filter_e2_nstemi_stemi(
    samples: Iterable[SampleMeta],
) -> list[tuple[SampleMeta, int]]:
    """Return 64-channel NSTEMI=0 vs STEMI=1 samples.

    The current repaired files use ``STEMI__...`` as a top-level group, while
    older files may use ``noNSTEMI__STEMI__...``.  Accept both layouts.
    """
    result: list[tuple[SampleMeta, int]] = []
    for sample in samples:
        if sample.channel_count != 64:
            continue
        if sample.label == 1:
            result.append((sample, 0))
        elif sample.label == 2 and (
            sample.group == "STEMI" or sample.non_nstemi_subcat == "STEMI"
        ):
            result.append((sample, 1))
    return result


def filter_e2_multiclass(
    samples: Iterable[SampleMeta],
    subcats: Iterable[str],
) -> list[tuple[SampleMeta, int]]:
    """
    Return (sample, class_id) pairs where:
      - NSTEMI (label==1) becomes class 0
      - each named subcat becomes classes 1..K
      - the rest of non-NSTEMI collapses into the last class
    """
    subcats = tuple(subcats)
    class_map = {"__NSTEMI__": 0}
    for i, sc in enumerate(subcats, start=1):
        class_map[sc] = i
    other_class = len(subcats) + 1

    result: list[tuple[SampleMeta, int]] = []
    for s in samples:
        if s.channel_count != 64:
            continue
        if s.label == 1:
            result.append((s, class_map["__NSTEMI__"]))
        elif s.label == 2:
            result.append((s, class_map.get(s.non_nstemi_subcat, other_class)))
    return result


# ---------------------------------------------------------------------------
# Class weights and reproducibility


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean 1."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.where(counts > 0, counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights.astype(np.float32)


def stratified_patient_folds(
    samples: list[SampleMeta],
    n_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Deterministic patient-level stratified split. Returns a list of fold arrays,
    each containing sample indices assigned to that fold's validation set.
    """
    rng = np.random.default_rng(seed)
    # Group by patient key; assume all rows of a patient share a label
    patient_to_indices: dict[str, list[int]] = {}
    patient_to_label: dict[str, int] = {}
    for i, s in enumerate(samples):
        patient_to_indices.setdefault(s.patient_key, []).append(i)
        patient_to_label[s.patient_key] = s.label

    fold_assignment: dict[str, int] = {}
    by_label: dict[int, list[str]] = {}
    for pk, label in patient_to_label.items():
        by_label.setdefault(label, []).append(pk)

    for label, patient_keys in by_label.items():
        rng.shuffle(patient_keys)
        for i, pk in enumerate(patient_keys):
            fold_assignment[pk] = i % n_folds

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, s in enumerate(samples):
        folds[fold_assignment[s.patient_key]].append(i)
    return [np.asarray(f, dtype=np.int64) for f in folds]


def seed_all(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
