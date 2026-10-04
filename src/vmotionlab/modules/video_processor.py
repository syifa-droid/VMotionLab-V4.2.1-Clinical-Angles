from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import re

import cv2
import numpy as np
import pandas as pd

from .pose_io import find_pose_video, match_camera_dir, read_openpose_json_dir
from .kinematics_engine import save_motion_raw
from .runtime_paths import resource_root

SKELETON = [
    ("Nose", "LShoulder"), ("Nose", "RShoulder"),
    ("LShoulder", "RShoulder"), ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"), ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"), ("LShoulder", "LHip"),
    ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"), ("LHeel", "LBigToe"),
    ("LBigToe", "LSmallToe"), ("RHip", "RKnee"),
    ("RKnee", "RAnkle"), ("RAnkle", "RHeel"),
    ("RHeel", "RBigToe"), ("RBigToe", "RSmallToe"),
]


def video_info(path: str | Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": frames / fps if fps else 0.0,
    }


def _copy_config(workspace: Path) -> None:
    resources = resource_root() / "resources"
    template = resources / "Pose2Sim_Config.toml"
    models = resources / "models"
    detector_model = models / "yolox_m_humanart.onnx"
    pose_model = models / "rtmpose_m_halpe26.onnx"

    if not template.exists():
        raise FileNotFoundError(f"Pose2Sim configuration not found: {template}")

    for model in (detector_model, pose_model):
        if not model.exists():
            raise FileNotFoundError(
                f"Bundled RTMPose model is missing: {model}. "
                "Rebuild the VMotionLab Desktop Edition."
            )

    mode = {
        "det_class": "YOLOX",
        "det_model": detector_model.resolve().as_posix(),
        "det_input_size": [640, 640],
        "pose_class": "RTMPose",
        "pose_model": pose_model.resolve().as_posix(),
        "pose_input_size": [192, 256],
    }

    config_text = template.read_text(encoding="utf-8")
    replacement = 'mode = """' + repr(mode) + '"""'
    config_text, count = re.subn(
        r"(?m)^\s*mode\s*=.*$",
        replacement,
        config_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not configure bundled RTMPose model paths.")

    # A portable CPU default maximizes compatibility across Windows laptops.
    config_text = re.sub(
        r"(?m)^\s*device\s*=.*$",
        "device = 'CPU'",
        config_text,
        count=1,
    )
    config_text = re.sub(
        r"(?m)^\s*backend\s*=.*$",
        "backend = 'onnxruntime'",
        config_text,
        count=1,
    )

    (workspace / "Config.toml").write_text(config_text, encoding="utf-8")


def _run_pose2sim(workspace: Path) -> None:
    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--vmotionlab-pose2sim-worker",
            str(workspace),
        ]
    else:
        command = [
            sys.executable,
            "-c",
            "from Pose2Sim import Pose2Sim; Pose2Sim.poseEstimation()",
        ]
    process = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (workspace / "pose_processing.log").write_text(
        process.stdout + "\n" + process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(
            "RTMPose processing failed. Review pose_processing.log.\n"
            + process.stderr[-2000:]
        )


def _draw_overlay(source_video: Path, landmarks: pd.DataFrame, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index < len(landmarks):
            row = landmarks.iloc[index]
            for start, end in SKELETON:
                confidence = min(
                    float(row.get(f"{start}_visibility", 0.0)),
                    float(row.get(f"{end}_visibility", 0.0)),
                )
                if confidence < 0.25:
                    continue
                p1 = (int(row[f"{start}_x"]), int(row[f"{start}_y"]))
                p2 = (int(row[f"{end}_x"]), int(row[f"{end}_y"]))
                if all(np.isfinite([*p1, *p2])):
                    cv2.line(frame, p1, p2, (0, 220, 0), 2, cv2.LINE_AA)
            for name in {name for edge in SKELETON for name in edge}:
                confidence = float(row.get(f"{name}_visibility", 0.0))
                x = row.get(f"{name}_x", np.nan)
                y = row.get(f"{name}_y", np.nan)
                if confidence >= 0.25 and np.isfinite([x, y]).all():
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "RTMPose / HALPE-26",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()


def process_uploaded_video(
    video_path: str | Path,
    output_project_path: str | Path,
    analysis_config: dict,
    save_annotated_video: bool = True,
    progress_callback=None,
) -> dict:
    source_video = Path(video_path)
    workspace = Path(output_project_path)
    workspace.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(0.03)

    videos_dir = workspace / "videos"
    videos_dir.mkdir(exist_ok=True)
    for old in videos_dir.iterdir():
        if old.is_file():
            old.unlink()
    local_video = videos_dir / f"cam01{source_video.suffix.lower() or '.mp4'}"
    shutil.copy2(source_video, local_video)
    _copy_config(workspace)
    if progress_callback:
        progress_callback(0.08)

    for old_name in ("pose", "pose-sync"):
        old = workspace / old_name
        if old.exists():
            shutil.rmtree(old)
    _run_pose2sim(workspace)
    if progress_callback:
        progress_callback(0.72)

    info = video_info(local_video)
    pose_directory = match_camera_dir(workspace / "pose", local_video.stem)
    full = read_openpose_json_dir(pose_directory, info["fps"])
    full.to_csv(workspace / "pose_landmarks_full.csv", index=False)
    full.to_csv(workspace / "landmarks.csv", index=False)

    motion_raw = save_motion_raw(workspace, full, analysis_config)

    visibility = full[[column for column in full if column.endswith("_visibility")]]
    pose_detected = visibility.max(axis=1, skipna=True) > 0.0
    quality_summary = {
        "pose_detection_percent": float(pose_detected.mean() * 100.0),
        "mean_keypoint_confidence": float(visibility.mean(axis=1, skipna=True).mean()),
        "frame_count": int(len(full)),
        "sampling_frequency_hz": float(info["fps"]),
        "pose_model": "RTMPose Body_with_feet (HALPE-26)",
    }
    (workspace / "quality_summary.json").write_text(
        json.dumps(quality_summary, indent=2), encoding="utf-8"
    )

    if save_annotated_video:
        pose_video = find_pose_video(workspace / "pose", local_video.stem)
        destination = workspace / "recording_with_pose.mp4"
        if pose_video and pose_video.exists():
            shutil.copy2(pose_video, destination)
        else:
            _draw_overlay(local_video, full, destination)

    if progress_callback:
        progress_callback(1.0)
    return {
        **quality_summary,
        "source_video": str(source_video),
        "workspace": str(workspace),
        "width": info["width"],
        "height": info["height"],
        "duration_s": info["duration"],
    }


# Backwards-compatible helpers retained for early V4 projects.
def write_uploaded(upload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getbuffer())
    return path


def process_project(project, camera_videos):
    result = {}
    for camera_key, video in camera_videos.items():
        result[camera_key] = process_uploaded_video(
            video_path=video,
            output_project_path=Path(project) / camera_key,
            analysis_config={"selected_sides": ["right"]},
        )
    return result
