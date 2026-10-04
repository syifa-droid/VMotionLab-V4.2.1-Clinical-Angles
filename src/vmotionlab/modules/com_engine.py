from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import lowpass, point

# Segment mass fractions are combined where distal hand/foot landmarks are limited.
SEGMENTS = {
    "head_neck": (0.081, "Neck", "Head", 0.50),
    "trunk": (0.497, "Neck", "Hip", 0.50),
    "left_upper_arm": (0.028, "LShoulder", "LElbow", 0.436),
    "right_upper_arm": (0.028, "RShoulder", "RElbow", 0.436),
    "left_forearm_hand": (0.022, "LElbow", "LWrist", 0.430),
    "right_forearm_hand": (0.022, "RElbow", "RWrist", 0.430),
    "left_thigh": (0.100, "LHip", "LKnee", 0.433),
    "right_thigh": (0.100, "RHip", "RKnee", 0.433),
    "left_shank": (0.0465, "LKnee", "LAnkle", 0.433),
    "right_shank": (0.0465, "RKnee", "RAnkle", 0.433),
    "left_foot": (0.0145, "LHeel", "LBigToe", 0.50),
    "right_foot": (0.0145, "RHeel", "RBigToe", 0.50),
}


@dataclass
class COMResult:
    com: pd.DataFrame
    segments: pd.DataFrame
    summary: dict


def run_com_analysis(project_path: str | Path, config: dict) -> COMResult:
    project = Path(project_path)
    source = project / "pose_landmarks_full.csv"
    if not source.exists():
        raise FileNotFoundError("pose_landmarks_full.csv is unavailable for COM analysis.")
    data = pd.read_csv(source)
    minimum_confidence = float(config.get("min_visibility", 0.4))
    minimum_mass = float(config.get("min_valid_mass_fraction", 0.70))

    weighted_x = np.zeros(len(data), dtype=float)
    weighted_y = np.zeros(len(data), dtype=float)
    valid_mass = np.zeros(len(data), dtype=float)
    segment_rows: list[dict] = []

    for segment, (mass, proximal_name, distal_name, fraction) in SEGMENTS.items():
        proximal = point(data, proximal_name)
        distal = point(data, distal_name)
        segment_com = proximal + fraction * (distal - proximal)
        confidence = np.minimum(
            pd.to_numeric(data[f"{proximal_name}_visibility"], errors="coerce").to_numpy(float),
            pd.to_numeric(data[f"{distal_name}_visibility"], errors="coerce").to_numpy(float),
        )
        valid = (
            (confidence >= minimum_confidence)
            & np.isfinite(segment_com).all(axis=1)
        )
        weighted_x[valid] += mass * segment_com[valid, 0]
        weighted_y[valid] += mass * segment_com[valid, 1]
        valid_mass[valid] += mass
        for frame in np.where(valid)[0]:
            segment_rows.append(
                {
                    "frame": int(frame),
                    "time": float(data.loc[frame, "time"]),
                    "segment": segment,
                    "segment_com_x_px": float(segment_com[frame, 0]),
                    "segment_com_y_px": float(segment_com[frame, 1]),
                    "mass_fraction": float(mass),
                    "confidence": float(confidence[frame]),
                }
            )

    x = np.divide(
        weighted_x,
        valid_mass,
        out=np.full(len(data), np.nan),
        where=valid_mass > 0,
    )
    y = np.divide(
        weighted_y,
        valid_mass,
        out=np.full(len(data), np.nan),
        where=valid_mass > 0,
    )

    top = np.nanmin(
        np.stack(
            [
                pd.to_numeric(data["Head_y"], errors="coerce").to_numpy(float),
                pd.to_numeric(data["Nose_y"], errors="coerce").to_numpy(float),
            ]
        ),
        axis=0,
    )
    bottom = np.nanmax(
        np.stack(
            [
                pd.to_numeric(data["LHeel_y"], errors="coerce").to_numpy(float),
                pd.to_numeric(data["RHeel_y"], errors="coerce").to_numpy(float),
                pd.to_numeric(data["LBigToe_y"], errors="coerce").to_numpy(float),
                pd.to_numeric(data["RBigToe_y"], errors="coerce").to_numpy(float),
            ]
        ),
        axis=0,
    )
    body_height_px = float(np.nanmedian(bottom - top))
    if not np.isfinite(body_height_px) or body_height_px <= 1:
        body_height_px = 1.0

    time = pd.to_numeric(data["time"], errors="coerce")
    fps = 1.0 / float(time.diff().median())
    maximum_gap = int(config.get("max_short_gap_frames", 5))
    x_series = pd.Series(x).interpolate(limit=maximum_gap, limit_area="inside")
    y_series = pd.Series(y).interpolate(limit=maximum_gap, limit_area="inside")
    x_filtered = lowpass(
        x_series,
        fps,
        float(config.get("filter_cutoff_hz", 3.0)),
        int(config.get("filter_order", 4)),
    )
    y_filtered = lowpass(
        y_series,
        fps,
        float(config.get("filter_cutoff_hz", 3.0)),
        int(config.get("filter_order", 4)),
    )

    baseline_frames = max(1, min(len(data), int(round(fps))))
    baseline_x = float(x_filtered.iloc[:baseline_frames].median())
    baseline_y = float(y_filtered.iloc[:baseline_frames].median())
    valid_frame = valid_mass >= minimum_mass

    com = pd.DataFrame(
        {
            "frame": np.arange(len(data), dtype=int),
            "time": time,
            "com_x_px": x_filtered,
            "com_y_px": y_filtered,
            "com_horizontal_bh": (x_filtered - baseline_x) / body_height_px,
            "com_vertical_bh": -(y_filtered - baseline_y) / body_height_px,
            "valid_mass_fraction": valid_mass,
            "valid_frame": valid_frame,
        }
    )
    com.loc[
        ~valid_frame,
        ["com_x_px", "com_y_px", "com_horizontal_bh", "com_vertical_bh"],
    ] = np.nan
    segments = pd.DataFrame(segment_rows)

    summary = {
        "pose_model": "RTMPose Body_with_feet (HALPE-26)",
        "anthropometric_model": config.get("anthropometric_model", "average"),
        "valid_frame_percent": float(valid_frame.mean() * 100.0),
        "mean_valid_mass_fraction": float(np.nanmean(valid_mass)),
        "horizontal_excursion_bh": float(
            com["com_horizontal_bh"].max() - com["com_horizontal_bh"].min()
        ),
        "vertical_excursion_bh": float(
            com["com_vertical_bh"].max() - com["com_vertical_bh"].min()
        ),
        "body_height_px": body_height_px,
        "frame_count": int(len(com)),
        "sampling_frequency_hz": float(fps),
    }

    com.to_csv(project / "com_analysis.csv", index=False)
    segments.to_csv(project / "segment_com.csv", index=False)
    (project / "com_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return COMResult(com=com, segments=segments, summary=summary)
