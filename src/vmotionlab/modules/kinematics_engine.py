from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import compute_motion_signals
from .project_manager import load_metadata, save_metadata

ENGINE_NAME = "classic_camera_specific_v4"
ENGINE_VERSION = "4.2.1-clinical"


def _selected_sides(config: dict) -> list[str]:
    values = [str(value).lower() for value in config.get("selected_sides", ["right"])]
    return [value for value in values if value in {"left", "right"}] or ["right"]


def build_motion_raw(
    full_landmarks: pd.DataFrame,
    analysis_config: dict,
) -> pd.DataFrame:
    """Build raw angles once, before confidence masking or synchronization.

    This restores the first working V4 principle: each camera is converted to
    joint angles independently. Confidence thresholds are applied later during
    preprocessing and never baked permanently into the raw geometry.
    """
    required = {"frame", "time"}
    missing = required.difference(full_landmarks.columns)
    if missing:
        raise ValueError(f"Landmark table is missing required columns: {sorted(missing)}")

    output = full_landmarks[["frame", "time"]].copy()
    visibility_columns = [
        column for column in full_landmarks.columns if column.endswith("_visibility")
    ]
    if visibility_columns:
        output["pose_detected"] = (
            full_landmarks[visibility_columns].max(axis=1, skipna=True) > 0.0
        ).astype(int)
    else:
        output["pose_detected"] = 1

    percentile = float(
        analysis_config.get("virtual_markers", {}).get("reference_percentile", 95.0)
    )
    for side in _selected_sides(analysis_config):
        signals = compute_motion_signals(full_landmarks, side, percentile)
        for column in signals.columns:
            output[column] = signals[column]
        suffix = "l" if side == "left" else "r"
        for joint in ("hip", "knee", "ankle"):
            base = "ankle_angle" if joint == "ankle" else f"{joint}_flexion"
            for method in ("direct", "virtual"):
                angle_column = f"{base}_{suffix}_{method}"
                output[f"{joint}_valid_{suffix}_{method}"] = (
                    pd.to_numeric(output.get(angle_column), errors="coerce")
                    .notna()
                    .astype(int)
                )

    output["kinematics_engine"] = ENGINE_NAME
    return output


def save_motion_raw(
    workspace: str | Path,
    full_landmarks: pd.DataFrame,
    analysis_config: dict,
) -> pd.DataFrame:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    raw = build_motion_raw(full_landmarks, analysis_config)
    raw.to_csv(workspace / "motion_raw.csv", index=False)
    virtual_columns = [
        column
        for column in raw.columns
        if "_virtual" in column or column in {"frame", "time", "pose_detected"}
    ]
    raw[virtual_columns].to_csv(workspace / "virtual_landmarks.csv", index=False)
    return raw


def _dual_camera_setup(metadata: dict) -> tuple[str, str]:
    setup = metadata.get("analysis_config", {}).get("camera_setup", {})
    side_a = str(setup.get("camera_a_side", "left")).lower()
    side_b = str(setup.get("camera_b_side", "right")).lower()
    if side_a not in {"left", "right"}:
        side_a = "left"
    if side_b not in {"left", "right"} or side_b == side_a:
        side_b = "right" if side_a == "left" else "left"
    return side_a, side_b


def rebuild_project_motion_raw(project_path: str | Path) -> dict:
    """Rebuild raw kinematics from existing pose landmarks without rerunning RTMPose."""
    project = Path(project_path)
    metadata = load_metadata(project) or {}
    config = metadata.get("analysis_config", {})
    capture_mode = metadata.get("capture_mode") or config.get("camera_setup", {}).get("mode")

    if capture_mode == "dual_camera_visual_sync" or (project / "camera_a").exists():
        side_a, side_b = _dual_camera_setup(metadata)
        camera_rows = []
        for camera, side in (("camera_a", side_a), ("camera_b", side_b)):
            landmarks_path = project / camera / "pose_landmarks_full.csv"
            if not landmarks_path.exists():
                raise FileNotFoundError(f"Missing {camera} pose landmarks: {landmarks_path}")
            landmarks = pd.read_csv(landmarks_path)
            camera_config = json.loads(json.dumps(config))
            camera_config["selected_sides"] = [side]
            raw = save_motion_raw(project / camera, landmarks, camera_config)
            camera_rows.append({"camera": camera, "side": side, "frames": len(raw)})

        sync_path = project / "synchronization.json"
        if sync_path.exists():
            sync = json.loads(sync_path.read_text(encoding="utf-8"))
            offset_frames = int(sync.get("final_offset_frames", 0))
        else:
            offset_frames = 0

        from .temporal_sync import rebuild_shared_motion

        count, fps = rebuild_shared_motion(
            project,
            camera_a_side=side_a,
            camera_b_side=side_b,
            offset_frames=offset_frames,
        )
        result = {
            "mode": "dual_camera",
            "engine": ENGINE_NAME,
            "camera_results": camera_rows,
            "shared_frames": int(count),
            "sampling_frequency_hz": float(fps),
            "used_existing_offset_frames": int(offset_frames),
        }
    else:
        landmarks_path = project / "pose_landmarks_full.csv"
        if not landmarks_path.exists():
            raise FileNotFoundError(f"Missing pose landmarks: {landmarks_path}")
        landmarks = pd.read_csv(landmarks_path)
        raw = save_motion_raw(project, landmarks, config)
        result = {
            "mode": "single_camera",
            "engine": ENGINE_NAME,
            "frames": int(len(raw)),
            "sampling_frequency_hz": float(
                1.0 / pd.to_numeric(raw["time"], errors="coerce").diff().median()
            ),
        }

    metadata["kinematics_engine"] = {
        "name": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "camera_specific_before_sync": True,
        "confidence_applied_during_preprocessing": True,
        "clinical_angle_convention": {
            "hip": "flexion positive; extension negative",
            "knee": "flexion positive; hyperextension negative",
            "ankle": "dorsiflexion positive; plantarflexion negative",
            "sagittal_forward_direction": "auto from heel-to-forefoot",
        },
    }
    metadata.setdefault("files", {})["motion_raw"] = "motion_raw.csv"
    save_metadata(project, metadata)
    (project / "kinematics_engine.json").write_text(
        json.dumps(metadata["kinematics_engine"], indent=2), encoding="utf-8"
    )
    return result
