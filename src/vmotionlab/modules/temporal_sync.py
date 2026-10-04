from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def read_video_frame(path: str | Path, time_s: float):
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(float(time_s) * fps))))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _motion_signal(data: pd.DataFrame, threshold: float = 0.4) -> tuple[np.ndarray, float]:
    values = []
    for name in ["RWrist", "LWrist", "RElbow", "LElbow"]:
        y = pd.to_numeric(data[f"{name}_y"], errors="coerce").where(
            pd.to_numeric(data[f"{name}_visibility"], errors="coerce") >= threshold
        )
        values.append(y)
    signal = pd.concat(values, axis=1).mean(axis=1, skipna=True)
    signal = signal.interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    fps = 1.0 / float(pd.to_numeric(data["time"], errors="coerce").diff().median())
    velocity = np.gradient(signal)
    if len(velocity) > 30 and fps > 0:
        cutoff = min(6.0 / (fps / 2.0), 0.99)
        sos = butter(4, cutoff, btype="low", output="sos")
        velocity = sosfiltfilt(sos, velocity)
    velocity = (velocity - np.nanmean(velocity)) / (np.nanstd(velocity) + 1e-9)
    return velocity, fps


def _point(data: pd.DataFrame, name: str) -> np.ndarray:
    return data[[f"{name}_x", f"{name}_y"]].to_numpy(dtype=float)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(ba * bc, axis=1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _detect_anchor(data: pd.DataFrame) -> dict:
    """Detect the midpoint of a stable bilateral arm hold at shoulder height."""
    time = pd.to_numeric(data["time"], errors="coerce")
    dt = float(time.diff().median())
    fps = 1.0 / dt if np.isfinite(dt) and dt > 0 else 30.0
    n = len(data)
    search_end = min(
        n,
        max(
            int(round(fps * 3.0)),
            min(int(round(fps * 10.0)), int(round(n * 0.60))),
        ),
    )

    ls, rs = _point(data, "LShoulder"), _point(data, "RShoulder")
    le, re_ = _point(data, "LElbow"), _point(data, "RElbow")
    lw, rw = _point(data, "LWrist"), _point(data, "RWrist")
    lh, rh = _point(data, "LHip"), _point(data, "RHip")

    left_elbow = _angle(ls, le, lw)
    right_elbow = _angle(rs, re_, rw)
    torso = np.nanmean(
        np.stack([np.linalg.norm(ls - lh, axis=1), np.linalg.norm(rs - rh, axis=1)]),
        axis=0,
    )
    torso = np.where(torso > 1e-6, torso, np.nan)
    horizontal_error = np.nanmean(
        np.stack([np.abs(lw[:, 1] - ls[:, 1]), np.abs(rw[:, 1] - rs[:, 1])]),
        axis=0,
    ) / torso

    confidence_columns = [
        f"{name}_visibility"
        for name in ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"]
    ]
    confidence = data[confidence_columns].apply(pd.to_numeric, errors="coerce").min(axis=1).to_numpy(float)
    stable = (
        (confidence >= 0.40)
        & (left_elbow >= 145.0)
        & (right_elbow >= 145.0)
        & (horizontal_error <= 0.30)
    )
    stable[search_end:] = False
    minimum_frames = max(3, int(round(fps * 0.30)))
    runs = [(s, e) for s, e in _true_runs(stable) if e - s >= minimum_frames]

    if runs:
        # Prefer the earliest robust hold because it is the recording-start cue.
        start, end = sorted(runs, key=lambda run: (run[0], -(run[1] - run[0])))[0]
        frame = int(round((start + end - 1) / 2.0))
        return {
            "detected": True,
            "anchor_frame": frame,
            "anchor_time_s": float(time.iloc[frame]),
            "hold_duration_s": float((end - start) / fps),
            "mean_elbow_angle_deg": float(np.nanmean([left_elbow[start:end], right_elbow[start:end]])),
            "mean_horizontal_error_deg": None,
            "mean_horizontal_error_torso_fraction": float(np.nanmean(horizontal_error[start:end])),
            "confidence": float(np.nanmean(confidence[start:end])),
            "message": "Stable bilateral arm-hold midpoint detected.",
        }

    signal, _ = _motion_signal(data)
    frame = int(np.nanargmax(np.abs(signal[:search_end]))) if search_end else 0
    return {
        "detected": False,
        "anchor_frame": frame,
        "anchor_time_s": float(time.iloc[min(frame, n - 1)]) if n else 0.0,
        "hold_duration_s": None,
        "mean_elbow_angle_deg": None,
        "mean_horizontal_error_deg": None,
        "confidence": float(confidence[min(frame, n - 1)]) if n else 0.0,
        "message": "Stable arm hold was not detected. A motion peak is shown only as a manual-review starting point.",
    }
def _estimate_offset(a: pd.DataFrame, b: pd.DataFrame, maximum_seconds: float = 5.0) -> dict:
    signal_a, fps_a = _motion_signal(a)
    signal_b, fps_b = _motion_signal(b)
    fps = min(fps_a, fps_b)
    maximum_lag = int(round(maximum_seconds * fps))
    best_correlation = -np.inf
    best_lag = 0
    curve = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag >= 0:
            count = min(len(signal_a), len(signal_b) - lag)
            aa = signal_a[:count]
            bb = signal_b[lag : lag + count]
        else:
            count = min(len(signal_b), len(signal_a) + lag)
            bb = signal_b[:count]
            aa = signal_a[-lag : -lag + count]
        if count < 10:
            continue
        correlation = float(np.corrcoef(aa, bb)[0, 1])
        curve.append([lag, correlation])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return {
        "offset_frames": int(best_lag),
        "offset_seconds": float(best_lag / fps),
        "correlation": float(best_correlation),
        "sampling_frequency_hz": float(fps),
        "correlation_curve": curve,
    }


def _align_frames(
    a: pd.DataFrame,
    b: pd.DataFrame,
    offset_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offset = int(offset_frames)
    if offset >= 0:
        count = min(len(a), len(b) - offset)
        aligned_a = a.iloc[: max(0, count)].copy()
        aligned_b = b.iloc[offset : offset + max(0, count)].copy()
    else:
        count = min(len(b), len(a) + offset)
        aligned_b = b.iloc[: max(0, count)].copy()
        aligned_a = a.iloc[-offset : -offset + max(0, count)].copy()
    count = min(len(aligned_a), len(aligned_b))
    return (
        aligned_a.iloc[:count].reset_index(drop=True),
        aligned_b.iloc[:count].reset_index(drop=True),
    )


def _side_columns(data: pd.DataFrame, side: str) -> list[str]:
    suffix = "l" if side == "left" else "r"
    selected = []
    for column in data.columns:
        if column in {"frame", "time", "pose_detected"}:
            continue
        if (
            f"_{suffix}_" in column
            or column.endswith(f"_{suffix}")
            or column.startswith(f"{side}_")
        ):
            selected.append(column)
    return selected


def rebuild_shared_motion(
    project: Path,
    camera_a_side: str,
    camera_b_side: str,
    offset_frames: int,
) -> tuple[int, float]:
    full_a = pd.read_csv(project / "camera_a" / "pose_landmarks_full.csv")
    full_b = pd.read_csv(project / "camera_b" / "pose_landmarks_full.csv")
    raw_a = pd.read_csv(project / "camera_a" / "motion_raw.csv")
    raw_b = pd.read_csv(project / "camera_b" / "motion_raw.csv")

    full_a, full_b = _align_frames(full_a, full_b, offset_frames)
    raw_a, raw_b = _align_frames(raw_a, raw_b, offset_frames)
    count = min(len(full_a), len(full_b), len(raw_a), len(raw_b))
    if count < 2:
        raise ValueError("The synchronized videos have insufficient overlapping frames.")

    full_a = full_a.iloc[:count].copy()
    full_b = full_b.iloc[:count].copy()
    raw_a = raw_a.iloc[:count].copy()
    raw_b = raw_b.iloc[:count].copy()

    fps_a = 1.0 / float(full_a["time"].diff().median())
    fps_b = 1.0 / float(full_b["time"].diff().median())
    fps = min(fps_a, fps_b)
    timeline = np.arange(count, dtype=float) / fps
    for data in (full_a, full_b, raw_a, raw_b):
        data["frame"] = np.arange(count)
        data["time"] = timeline

    full_a.to_csv(project / "camera_a" / "pose_landmarks_synced.csv", index=False)
    full_b.to_csv(project / "camera_b" / "pose_landmarks_synced.csv", index=False)

    merged = pd.DataFrame({"frame": np.arange(count), "time": timeline})
    merged["pose_detected_camera_a"] = raw_a.get("pose_detected", 1).to_numpy()
    merged["pose_detected_camera_b"] = raw_b.get("pose_detected", 1).to_numpy()
    merged["pose_detected"] = (
        (merged["pose_detected_camera_a"] > 0)
        & (merged["pose_detected_camera_b"] > 0)
    ).astype(int)
    for column in _side_columns(raw_a, camera_a_side):
        merged[column] = raw_a[column].to_numpy()
    for column in _side_columns(raw_b, camera_b_side):
        merged[column] = raw_b[column].to_numpy()
    merged.to_csv(project / "motion_raw.csv", index=False)
    return count, fps


def synchronize_dual_camera_project(
    project_path: str | Path,
    camera_a_side: str = "left",
    camera_b_side: str = "right",
    manual_adjustment_s: float = 0.0,
    camera_a_anchor_override_s: float | None = None,
    camera_b_anchor_override_s: float | None = None,
    manual_adjustment: float | None = None,
) -> dict:
    project = Path(project_path)
    data_a = pd.read_csv(project / "camera_a" / "pose_landmarks_full.csv")
    data_b = pd.read_csv(project / "camera_b" / "pose_landmarks_full.csv")
    detection_a = _detect_anchor(data_a)
    detection_b = _detect_anchor(data_b)
    estimate = _estimate_offset(data_a, data_b)
    fps = float(estimate["sampling_frequency_hz"])

    anchor_a = (
        float(camera_a_anchor_override_s)
        if camera_a_anchor_override_s is not None
        else float(detection_a["anchor_time_s"])
    )
    anchor_b = (
        float(camera_b_anchor_override_s)
        if camera_b_anchor_override_s is not None
        else float(detection_b["anchor_time_s"])
    )
    if camera_a_anchor_override_s is None and camera_b_anchor_override_s is None:
        automatic_offset_s = float(estimate["offset_seconds"])
    else:
        automatic_offset_s = anchor_b - anchor_a

    adjustment_s = float(manual_adjustment_s)
    if manual_adjustment is not None:
        # Legacy helper supplied frames rather than seconds.
        adjustment_s += float(manual_adjustment) / fps
    final_offset_s = automatic_offset_s + adjustment_s
    final_offset_frames = int(round(final_offset_s * fps))
    count, target_fps = rebuild_shared_motion(
        project,
        camera_a_side=camera_a_side,
        camera_b_side=camera_b_side,
        offset_frames=final_offset_frames,
    )

    result = {
        "status": "synchronized",
        "method": "Stable bilateral arm-hold midpoint with manual review; cross-correlation fallback",
        "camera_a_side": camera_a_side,
        "camera_b_side": camera_b_side,
        "camera_a_detection": detection_a,
        "camera_b_detection": detection_b,
        "camera_a_anchor_time_s": anchor_a,
        "camera_b_anchor_time_s": anchor_b,
        "camera_a_sampling_frequency_hz": float(1.0 / data_a["time"].diff().median()),
        "camera_b_sampling_frequency_hz": float(1.0 / data_b["time"].diff().median()),
        "target_sampling_frequency_hz": float(target_fps),
        "automatic_offset_s": float(automatic_offset_s),
        "manual_adjustment_s": adjustment_s,
        "final_offset_s": float(final_offset_s),
        "final_offset_frames": final_offset_frames,
        "correlation": float(estimate["correlation"]),
        "common_frames": int(count),
        "common_duration_s": float((count - 1) / target_fps),
        "confirmed_by_user": False,
    }
    (project / "synchronization.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    # Retain compatibility with the early V4 filename.
    (project / "temporal_sync.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def create_single_camera_sync_record(project: str | Path) -> dict:
    project = Path(project)
    result = {
        "status": "not_required",
        "mode": "single_camera",
        "final_offset_s": 0.0,
        "final_offset_frames": 0,
        "correlation": 1.0,
        "confirmed_by_user": True,
    }
    (project / "synchronization.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (project / "temporal_sync.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

# Backward-compatible public helper used by early V4 tests.
def estimate_sync(a: pd.DataFrame, b: pd.DataFrame, max_seconds: float = 5.0) -> dict:
    result = _estimate_offset(a, b, maximum_seconds=max_seconds)
    return {
        **result,
        "offset_frames": int(result["offset_frames"]),
        "offset_seconds": float(result["offset_seconds"]),
        "fps": float(result["sampling_frequency_hz"]),
    }
