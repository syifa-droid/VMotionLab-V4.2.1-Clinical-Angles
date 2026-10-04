from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import interpolate_short_gaps, lowpass


def _angle_column(joint: str, side: str, method: str) -> str:
    suffix = "l" if side == "left" else "r"
    base = "ankle_angle" if joint == "ankle" else f"{joint}_flexion"
    return f"{base}_{suffix}_{method}"


def preprocess_motion_file(
    project_path: str | Path,
    analysis_config: dict | None = None,
    cutoff_hz: float = 6.0,
    order: int = 4,
    max_gap_frames: int = 10,
    max_gap: int | None = None,
    minimum_quality: float | None = None,
):
    """Apply confidence once, interpolate short gaps, then filter.

    The base angle columns in motion_filtered.csv remain the filtered values for
    compatibility. Additional *_raw, *_interpolated and *_filtered columns make
    the complete processing pathway visible to the GUI and exported files.
    """
    project = Path(project_path)
    raw_path = project / "motion_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "motion_raw.csv is unavailable. Rebuild kinematics from existing landmarks first."
        )
    raw = pd.read_csv(raw_path)
    config = analysis_config or {}
    if max_gap is not None:
        max_gap_frames = int(max_gap)

    time = pd.to_numeric(raw["time"], errors="coerce")
    dt = float(time.diff().median())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("A valid sampling interval could not be determined.")
    fps = 1.0 / dt
    if minimum_quality is None:
        minimum_quality = float(config.get("quality", {}).get("min_visibility", 0.40))
    minimum_quality = float(minimum_quality)

    output = raw.copy()
    processed_columns: list[str] = []
    coverage_rows: list[dict] = []
    selected_sides = config.get("selected_sides", ["right"])
    selected_joints = config.get("selected_joints", ["hip", "knee", "ankle"])

    for side in selected_sides:
        suffix = "l" if side == "left" else "r"
        for joint in selected_joints:
            quality_column = f"{joint}_quality_{suffix}"
            quality = (
                pd.to_numeric(raw[quality_column], errors="coerce")
                if quality_column in raw
                else pd.Series(1.0, index=raw.index, dtype=float)
            )
            for method in ("direct", "virtual"):
                column = _angle_column(joint, side, method)
                if column not in raw:
                    continue
                valid_column = f"{joint}_valid_{suffix}_{method}"
                values = pd.to_numeric(raw[column], errors="coerce")

                geometry_valid = values.notna()
                confidence_valid = quality.notna() & (quality >= minimum_quality)
                accepted = geometry_valid & confidence_valid
                masked = values.where(accepted)
                interpolated = interpolate_short_gaps(masked, int(max_gap_frames))
                filtered = lowpass(interpolated, fps, float(cutoff_hz), int(order))

                output[f"{column}_raw"] = values
                output[f"{column}_interpolated"] = interpolated
                output[f"{column}_filtered"] = filtered
                output[column] = filtered
                output[valid_column] = filtered.notna().astype(int)
                processed_columns.append(column)
                coverage_rows.append(
                    {
                        "side": side,
                        "joint": joint,
                        "method": method,
                        "raw_angle_frames": int(geometry_valid.sum()),
                        "confidence_pass_frames": int(confidence_valid.sum()),
                        "accepted_before_interpolation_frames": int(accepted.sum()),
                        "interpolated_valid_frames": int(interpolated.notna().sum()),
                        "filtered_valid_frames": int(filtered.notna().sum()),
                        "filtered_valid_percent": float(filtered.notna().mean() * 100.0),
                        "minimum_confidence": minimum_quality,
                    }
                )

    output.to_csv(project / "motion_filtered.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(project / "preprocessing_coverage.csv", index=False)
    summary = {
        "engine": "classic_camera_specific_v4",
        "sampling_frequency_hz": float(fps),
        "cutoff_hz": float(cutoff_hz),
        "filter_order": int(order),
        "zero_phase": True,
        "max_gap_frames": int(max_gap_frames),
        "minimum_quality": minimum_quality,
        "confidence_applied_once": True,
        "processed_columns": processed_columns,
        "frame_count": int(len(output)),
        "coverage_file": "preprocessing_coverage.csv",
    }
    (project / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
