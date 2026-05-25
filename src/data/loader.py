"""
src/data/loader.py
------------------
Loads raw PADS dataset files and returns structured records for PD and HC
subjects only. No preprocessing, normalisation, or feature extraction here.

Expected dataset layout (PhysioNet PADS v1.0.0):
    <data_root>/
        patients/
            patient_001.json
            patient_002.json
            ...
        movement/
            timeseries/
                001_1a.csv
                001_1b.csv
                ...

Patient JSON fields used:
    - id          : zero-padded subject identifier string (e.g. "039")
    - condition   : "Parkinson's" | "Healthy" | "Other Movement Disorders" | ...
    - handedness  : "right" | "left"

Timeseries CSV format:
    - No header row; rows are time points (comma-separated).
    - 7 columns: column 0 = timestamp, columns 1–6 = IMU channels.
    - 1024 rows  = 10.24 s step  (steps 2–11)
    - 2048 rows  = 20.48 s step  (steps 1a / 1b)
    - Channel order (cols 1–6): acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z.

NOTE: Verify exact condition strings on first load using the helper
`inspect_dataset_structure()` below before running the full pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact `condition` field values in PADS patient JSON files.
# Verify these strings with inspect_dataset_structure() on first run.
CONDITION_PD: str = "Parkinson's"
CONDITION_HC: str = "Healthy"

LABEL_MAP: dict[str, int] = {CONDITION_PD: 1, CONDITION_HC: 0}

# Channels to load per wrist (6-channel IMU: 3-axis acc + 3-axis gyro).
# Replace <wrist> at runtime with the subject's dominant wrist.
CHANNEL_TEMPLATE: list[str] = [
    "accelerometer_x_{wrist}",
    "accelerometer_y_{wrist}",
    "accelerometer_z_{wrist}",
    "gyroscope_x_{wrist}",
    "gyroscope_y_{wrist}",
    "gyroscope_z_{wrist}",
]

# Steps excluded per the original PADS paper (no ML value).
# Step IDs: "3" = lift and hold, "5" = point finger, "8" = touch index.
EXCLUDED_STEPS: frozenset[str] = frozenset({"3", "5", "8"})

# Fallback wrist when handedness is missing or unrecognised.
FALLBACK_WRIST: str = "right"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubjectRecord:
    """One assessment step from one subject.

    A subject produces multiple SubjectRecords (one per step), all sharing
    the same subject_id and label. subject_id is consistent across all steps
    of the same subject and must never appear with more than one label value.

    Attributes
    ----------
    subject_id : str
        Zero-padded subject identifier from the patient JSON (e.g. "039").
        Shared across all steps belonging to the same subject; not unique
        per record.
    label : int
        Binary label: 1 = PD, 0 = HC. Consistent for all records of a subject.
    step_id : str
        Assessment step identifier (e.g. "1a", "2", "6").
    dominant_wrist : str
        Wrist used for signal extraction ("right" or "left").
    raw_signal : np.ndarray
        Raw IMU signal array of shape (T, 6).
        T = 1024 for 10.24 s steps; T = 2048 for 20.48 s steps.
        Channel order: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z.
    """

    subject_id: str
    label: int
    step_id: str
    dominant_wrist: str
    raw_signal: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_patient_json(path: Path) -> dict:
    """Read and return a single patient JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dominant_wrist(handedness: Optional[str]) -> str:
    """Return 'right' or 'left' from the handedness field.

    Falls back to FALLBACK_WRIST if the value is missing or unrecognised.
    """
    if isinstance(handedness, str) and handedness.lower() in {"right", "left"}:
        return handedness.lower()
    logger.warning(
        "Unrecognised handedness value %r — falling back to %r.",
        handedness,
        FALLBACK_WRIST,
    )
    return FALLBACK_WRIST


def _build_channel_names(wrist: str) -> list[str]:
    """Return the 6 channel names for the given wrist."""
    return [tpl.format(wrist=wrist) for tpl in CHANNEL_TEMPLATE]


def _load_timeseries(path: Path) -> np.ndarray:
    """Load a single headerless timeseries CSV and return the 6 IMU channels.

    The actual PADS timeseries files have no header row. Column layout:
        column 0 : timestamp (discarded)
        columns 1–6 : IMU channels in order acc_x, acc_y, acc_z,
                      gyr_x, gyr_y, gyr_z (dominant wrist)

    Parameters
    ----------
    path : Path
        Full path to the CSV file.

    Returns
    -------
    np.ndarray
        Array of shape (T, 6), dtype float32.

    Raises
    ------
    ValueError
        If the file has fewer than 7 columns.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 columns (timestamp + 6 IMU channels) in "
            f"{path}, got {df.shape[1]}."
        )
    # Column 0 is the timestamp; columns 1–6 are the IMU channels.
    return df.iloc[:, 1:7].to_numpy(dtype=np.float32)


def _discover_step_files(
    timeseries_dir: Path,
    subject_id: str,
) -> dict[str, Path]:
    """Return a mapping of step_id → CSV path for one subject.

    Expects files named like: <subject_id>_<step_id>.csv
    Example: 039_1a.csv, 039_2.csv, ..., 039_11.csv
    """
    pattern = f"{subject_id}_*.csv"
    files = list(timeseries_dir.glob(pattern))
    if not files:
        logger.warning("No timeseries files found for subject %s.", subject_id)
    step_map: dict[str, Path] = {}
    for f in files:
        # stem: e.g. "039_1a"  →  step_id: "1a"
        step_id = f.stem.split("_", maxsplit=1)[-1]
        step_map[step_id] = f
    return step_map


def _validate_records(records: list[SubjectRecord]) -> None:
    """Run basic sanity checks on the loaded records. Raises on failure."""
    if not records:
        raise ValueError("No records loaded — check data_root and filter conditions.")

    subject_ids = [r.subject_id for r in records]
    labels = [r.label for r in records]

    unique_subjects = set(subject_ids)
    unique_labels = set(labels)

    assert unique_labels == {0, 1}, (
        f"Expected labels {{0, 1}}, got {unique_labels}. "
        "DD rows may still be present or HC/PD strings are wrong."
    )

    # Fix 3: assert no subject_id maps to more than one label value.
    subject_label_map: dict[str, set[int]] = {}
    for r in records:
        subject_label_map.setdefault(r.subject_id, set()).add(r.label)
    conflicted = {sid: lbls for sid, lbls in subject_label_map.items() if len(lbls) > 1}
    assert not conflicted, (
        f"subject_id(s) appear with multiple labels — data integrity error: {conflicted}"
    )

    n_pd = sum(1 for s in set(subject_ids) if
               any(r.label == 1 and r.subject_id == s for r in records))
    n_hc = len(unique_subjects) - n_pd

    logger.info(
        "Loaded %d records from %d subjects (%d PD, %d HC).",
        len(records),
        len(unique_subjects),
        n_pd,
        n_hc,
    )

    shapes = {r.raw_signal.shape[1] for r in records}
    assert shapes == {6}, f"Expected 6 channels per record, got shapes: {shapes}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pads(
    data_root: str | Path,
    excluded_steps: frozenset[str] = EXCLUDED_STEPS,
) -> list[SubjectRecord]:
    """Load all PD and HC assessment steps from the PADS dataset.

    Filters out DD subjects entirely. Skips excluded assessment steps.
    Returns one SubjectRecord per (subject, step) pair.

    Parameters
    ----------
    data_root : str | Path
        Root directory of the downloaded PADS dataset (contains `patients/`
        and `movement/timeseries/` subdirectories).
    excluded_steps : frozenset[str]
        Step IDs to skip. Defaults to the three steps excluded in the
        original PADS paper (3, 5, 8).

    Returns
    -------
    list[SubjectRecord]
        Flat list of records, one per (subject, step). All steps from a
        given subject share the same subject_id and label.

    Raises
    ------
    FileNotFoundError
        If expected subdirectories are missing under data_root.
    ValueError
        If no records pass the PD / HC filter.
    """
    data_root = Path(data_root)
    patients_dir = data_root / "patients"
    timeseries_dir = data_root / "movement" / "timeseries"

    for required in (patients_dir, timeseries_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Expected directory not found: {required}. "
                "Check that data_root points to the PADS dataset root."
            )

    patient_files = sorted(patients_dir.glob("patient_*.json"))
    if not patient_files:
        raise FileNotFoundError(f"No patient JSON files found in {patients_dir}.")

    records: list[SubjectRecord] = []
    skipped_dd = 0

    for patient_file in patient_files:
        patient = _load_patient_json(patient_file)
        condition: str = patient.get("condition", "")

        # Filter: keep PD and HC only.
        if condition not in LABEL_MAP:
            skipped_dd += 1
            continue

        subject_id: str = patient["id"]
        label: int = LABEL_MAP[condition]
        dominant_wrist: str = _resolve_dominant_wrist(patient.get("handedness"))

        step_files = _discover_step_files(timeseries_dir, subject_id)

        for step_id, csv_path in sorted(step_files.items()):
            if step_id in excluded_steps:
                continue
            try:
                signal = _load_timeseries(csv_path)
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Skipping subject %s step %s — %s", subject_id, step_id, exc
                )
                continue

            records.append(
                SubjectRecord(
                    subject_id=subject_id,
                    label=label,
                    step_id=step_id,
                    dominant_wrist=dominant_wrist,
                    raw_signal=signal,
                )
            )

    logger.info("Skipped %d DD subjects.", skipped_dd)
    _validate_records(records)
    return records


def get_subject_index(
    records: list[SubjectRecord],
) -> dict[str, list[int]]:
    """Return a mapping from subject_id to list of record indices.

    Useful for constructing subject-level CV splits without leakage.

    Parameters
    ----------
    records : list[SubjectRecord]
        Output of load_pads().

    Returns
    -------
    dict[str, list[int]]
        Keys are subject_ids; values are lists of indices into `records`.
    """
    index: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        index.setdefault(rec.subject_id, []).append(i)
    return index


def get_labels(records: list[SubjectRecord]) -> np.ndarray:
    """Return an integer label array aligned with records.

    Parameters
    ----------
    records : list[SubjectRecord]
        Output of load_pads().

    Returns
    -------
    np.ndarray
        Shape (N,), dtype int, values in {0, 1}.
    """
    return np.array([r.label for r in records], dtype=int)


def get_subject_ids(records: list[SubjectRecord]) -> np.ndarray:
    """Return a string array of subject_ids aligned with records.

    Parameters
    ----------
    records : list[SubjectRecord]
        Output of load_pads().

    Returns
    -------
    np.ndarray
        Shape (N,), dtype object (str).
    """
    return np.array([r.subject_id for r in records])


# ---------------------------------------------------------------------------
# Inspection utility (run once before full pipeline)
# ---------------------------------------------------------------------------

def inspect_dataset_structure(data_root: str | Path) -> None:
    """Print dataset structure details for first-run verification.

    Call this before load_pads() to confirm:
    - Exact condition strings used in patient JSONs.
    - Actual channel names in timeseries CSVs.
    - Number of patients per condition group.
    - Number of timeseries files per subject.

    Parameters
    ----------
    data_root : str | Path
        Root directory of the PADS dataset.
    """
    data_root = Path(data_root)
    patients_dir = data_root / "patients"
    timeseries_dir = data_root / "movement" / "timeseries"

    print("=== PADS Dataset Structure Inspection ===\n")

    # Condition counts
    condition_counts: dict[str, int] = {}
    for pf in sorted(patients_dir.glob("patient_*.json")):
        p = _load_patient_json(pf)
        cond = p.get("condition", "MISSING")
        condition_counts[cond] = condition_counts.get(cond, 0) + 1

    print("Condition field values and counts:")
    for cond, count in sorted(condition_counts.items()):
        print(f"  {repr(cond)}: {count}")

    # Sample channel names from first available CSV
    csv_files = list(timeseries_dir.glob("*.csv"))
    if csv_files:
        sample_csv = csv_files[0]
        sample_df = pd.read_csv(sample_csv, header=None, nrows=1)
        print(f"\nSample timeseries file: {sample_csv.name}")
        print(f"  Columns ({len(sample_df.columns)}): col 0 = timestamp, "
              f"cols 1–6 = IMU channels")
        print(f"  Shape hint (rows x cols): {pd.read_csv(sample_csv, header=None).shape}")
    else:
        print("\nNo CSV files found in timeseries directory.")

    # File counts
    total_csv = len(csv_files)
    print(f"\nTotal timeseries CSV files: {total_csv}")
    print("\nRun load_pads() after confirming the above matches expectations.")
