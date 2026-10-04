from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_com_figures(
    project_path: str | Path,
    com: pd.DataFrame,
    phase_frames: pd.DataFrame | None = None,
) -> dict[str, str]:
    project = Path(project_path)
    figures = project / "com_figures"
    figures.mkdir(exist_ok=True)
    outputs: dict[str, str] = {}

    specifications = [
        ("com_vertical_figure", "time", "com_vertical_bh", "Vertical COM displacement", "Time (s)", "Body heights"),
        ("com_horizontal_figure", "time", "com_horizontal_bh", "Horizontal COM displacement", "Time (s)", "Body heights"),
        ("com_trajectory_figure", "com_horizontal_bh", "com_vertical_bh", "Projected COM trajectory", "Horizontal (BH)", "Vertical (BH)"),
    ]
    for key, x_column, y_column, title, x_label, y_label in specifications:
        path = figures / f"{key}.png"
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(com[x_column], com[y_column])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs[key] = path.relative_to(project).as_posix()
    return outputs


def _find_video(project: Path) -> Path | None:
    candidates = [project / "recording_with_pose.mp4"]
    candidates.extend(sorted(project.glob("uploaded_video.*")))
    candidates.extend(sorted(project.glob("recording.*")))
    return next((path for path in candidates if path.exists()), None)


def generate_com_overlay_video(
    project_path: str | Path,
    com: pd.DataFrame,
    phase_frames: pd.DataFrame | None = None,
) -> dict:
    project = Path(project_path)
    source = _find_video(project)
    if source is None:
        return {
            "created": False,
            "browser_compatible": False,
            "path": None,
            "note": "No source video was found.",
        }

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    destination = project / "recording_with_com.mp4"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    phases = (
        phase_frames.set_index("frame")["phase"].to_dict()
        if phase_frames is not None and not phase_frames.empty
        else {}
    )
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame < len(com):
            row = com.iloc[frame]
            x = row.get("com_x_px", np.nan)
            y = row.get("com_y_px", np.nan)
            valid = bool(row.get("valid_frame", False))
            if valid and np.isfinite([x, y]).all():
                cv2.circle(image, (int(x), int(y)), 9, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "Projected COM",
                    (int(x) + 12, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    image,
                    "COM invalid / insufficient visible mass",
                    (16, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if frame in phases:
                cv2.putText(
                    image,
                    str(phases[frame]).replace("_", " ").title(),
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        writer.write(image)
        frame += 1
    capture.release()
    writer.release()
    return {
        "created": destination.exists(),
        "browser_compatible": True,
        "path": str(destination),
        "note": "RTMPose selected-subject projected COM overlay.",
    }
