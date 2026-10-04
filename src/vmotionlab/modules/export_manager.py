from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
from typing import Any

import streamlit as st

from .project_manager import load_metadata, safe_name
from .ui_navigation import show_open_folder_button


def downloads_directory() -> Path:
    """Return the current Windows user's Downloads folder and create it if needed."""
    override = os.environ.get("VMOTIONLAB_DOWNLOADS_DIR")
    if override:
        folder = Path(override).expanduser()
    else:
        profile = os.environ.get("USERPROFILE")
        folder = Path(profile).expanduser() / "Downloads" if profile else Path.home() / "Downloads"
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def project_export_stem(project_path: str | Path) -> str:
    project = Path(project_path)
    metadata = load_metadata(project) or {}
    subject = metadata.get("subject_id")
    task = metadata.get("task")
    trial = metadata.get("trial_name")
    if subject and task and trial:
        return "_".join(safe_name(value) for value in (subject, task, trial))
    return safe_name(metadata.get("project_name", project.name))


def _normalise_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return value or "output"


def build_export_filename(
    project_path: str | Path,
    output_tag: str,
    extension: str,
    source_tag: str | None = None,
) -> str:
    stem = project_export_stem(project_path)
    parts = [stem]
    if source_tag:
        parts.append(_normalise_tag(source_tag))
    parts.append(_normalise_tag(output_tag))
    suffix = extension if extension.startswith(".") else f".{extension}"
    return "_".join(parts) + suffix.lower()


def unique_destination(filename: str, folder: str | Path | None = None) -> Path:
    target_folder = Path(folder) if folder is not None else downloads_directory()
    target_folder.mkdir(parents=True, exist_ok=True)
    candidate = target_folder / filename
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        numbered = candidate.with_name(f"{candidate.stem}_{index:02d}{candidate.suffix}")
        if not numbered.exists():
            return numbered
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")


def save_file_to_downloads(source: str | Path, filename: str) -> Path:
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = unique_destination(filename)
    shutil.copy2(source_path, destination)
    return destination


def save_bytes_to_downloads(data: bytes, filename: str) -> Path:
    destination = unique_destination(filename)
    destination.write_bytes(data)
    return destination


def show_save_file_button(
    source: str | Path,
    project_path: str | Path,
    output_tag: str,
    label: str,
    key: str,
    source_tag: str | None = None,
    use_container_width: bool = True,
    button_type: str = "secondary",
) -> Path | None:
    source_path = Path(source)
    if not source_path.exists():
        st.button(label, key=key, disabled=True, use_container_width=use_container_width)
        return None
    filename = build_export_filename(
        project_path,
        output_tag=output_tag,
        extension=source_path.suffix or ".bin",
        source_tag=source_tag,
    )
    if st.button(
        label,
        key=key,
        use_container_width=use_container_width,
        type=button_type,
        help=f"Save as {filename} in the Windows Downloads folder.",
    ):
        try:
            destination = save_file_to_downloads(source_path, filename)
            st.success(f"Saved to Downloads: {destination.name}")
            st.caption(str(destination))
            return destination
        except Exception as exc:
            st.error(f"Could not save the file: {exc}")
    return None


def show_save_bytes_button(
    data: bytes,
    project_path: str | Path,
    output_tag: str,
    extension: str,
    label: str,
    key: str,
    source_tag: str | None = None,
    use_container_width: bool = True,
) -> Path | None:
    filename = build_export_filename(
        project_path,
        output_tag=output_tag,
        extension=extension,
        source_tag=source_tag,
    )
    if st.button(
        label,
        key=key,
        use_container_width=use_container_width,
        help=f"Save as {filename} in the Windows Downloads folder.",
    ):
        try:
            destination = save_bytes_to_downloads(data, filename)
            st.success(f"Saved to Downloads: {destination.name}")
            st.caption(str(destination))
            return destination
        except Exception as exc:
            st.error(f"Could not save the file: {exc}")
    return None


def show_open_downloads_button(
    label: str = "Open Downloads Folder",
    key: str = "open_downloads_folder",
    use_container_width: bool = True,
) -> None:
    show_open_folder_button(
        downloads_directory(),
        label=label,
        key=key,
        use_container_width=use_container_width,
    )


def find_project_root(path: str | Path) -> Path:
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "metadata.json").exists():
            return candidate
    return current


def video_properties(path: str | Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for snapshots.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        return {"fps": fps, "frame_count": frame_count, "duration_s": duration}
    finally:
        capture.release()


def save_video_snapshot_to_downloads(
    video_path: str | Path,
    project_path: str | Path,
    time_s: float,
    output_tag: str,
    source_tag: str | None = None,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for snapshots.") from exc

    video = Path(video_path)
    properties = video_properties(video)
    duration = float(properties["duration_s"])
    if duration > 0:
        time_s = max(0.0, min(float(time_s), duration))
    else:
        time_s = max(0.0, float(time_s))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame at {time_s:.3f} s.")

    milliseconds = int(round(time_s * 1000.0))
    filename = build_export_filename(
        project_path,
        output_tag=f"{output_tag}_snapshot_{milliseconds:07d}ms",
        extension=".png",
        source_tag=source_tag,
    )
    destination = unique_destination(filename)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not write snapshot: {destination}")
    return destination
