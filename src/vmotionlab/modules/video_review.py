from __future__ import annotations

from pathlib import Path
import subprocess

import streamlit as st

from .project_manager import load_metadata
from .ui_navigation import show_open_folder_button
from .export_manager import (
    find_project_root,
    save_video_snapshot_to_downloads,
    show_open_downloads_button,
    show_save_file_button,
    video_properties,
)


def browser_preview_path(source: str | Path) -> Path:
    source = Path(source)
    return source.with_name(f"{source.stem}_browser_preview.mp4")


def preview_is_current(source: str | Path, preview: str | Path) -> bool:
    source = Path(source)
    preview = Path(preview)
    return (
        source.exists()
        and preview.exists()
        and preview.stat().st_size > 1024
        and preview.stat().st_mtime_ns >= source.stat().st_mtime_ns
    )


def ensure_browser_compatible_video(
    source: str | Path,
    force: bool = False,
) -> Path:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    preview = browser_preview_path(source)
    if not force and preview_is_current(source, preview):
        return preview

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "The video-preview component is missing. Run the V4.1.1 GUI update installer."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temporary = preview.with_name(preview.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode != 0 or not temporary.exists():
        temporary.unlink(missing_ok=True)
        detail = (process.stderr or process.stdout or "Unknown FFmpeg error")[-1800:]
        raise RuntimeError(f"Could not prepare the video preview: {detail}")

    temporary.replace(preview)
    return preview


def _video_export_context(path: Path) -> tuple[Path, str, str]:
    project = find_project_root(path)
    parent_name = path.parent.name.lower()
    if parent_name in {"camera_a", "camera_b", "com_camera_a", "com_camera_b"}:
        source_tag = parent_name
    else:
        source_tag = "single_camera"
    output_tag = "com_overlay" if "com" in path.stem.lower() else "pose_overlay"
    return project, source_tag, output_tag


def show_video_preview(
    path: str | Path,
    label: str,
    key_prefix: str,
    show_label: bool = True,
) -> None:
    path = Path(path)
    if show_label:
        st.markdown(f"**{label}**")
    if not path.exists():
        st.info("Pose overlay video is unavailable; the landmark CSV may still be ready.")
        return

    try:
        preview = browser_preview_path(path)
        if not preview_is_current(path, preview):
            with st.spinner("Preparing browser-compatible overlay preview..."):
                preview = ensure_browser_compatible_video(path)
        st.video(preview.read_bytes(), format="video/mp4")
    except Exception as exc:
        st.error(f"Video preview could not be prepared: {exc}")
        st.caption(f"Overlay file: {path}")
        show_open_folder_button(
            path.parent,
            label="Open Video Folder",
            key=f"{key_prefix}_open_video_folder_error",
        )
        return

    project, source_tag, output_tag = _video_export_context(path)
    action_left, action_middle, action_right = st.columns(3)
    with action_left:
        if st.button(
            "Rebuild Preview",
            key=f"{key_prefix}_rebuild_preview",
            use_container_width=True,
            help="Use this when a preview is incomplete or was created from an older overlay file.",
        ):
            try:
                with st.spinner("Rebuilding browser-compatible preview..."):
                    ensure_browser_compatible_video(path, force=True)
                st.success("Preview rebuilt.")
                st.rerun()
            except Exception as exc:
                st.error(f"Preview rebuild failed: {exc}")
    with action_middle:
        show_save_file_button(
            path,
            project_path=project,
            output_tag=output_tag,
            source_tag=source_tag,
            label="Save Video to Downloads",
            key=f"{key_prefix}_save_video",
        )
    with action_right:
        show_open_folder_button(
            path.parent,
            label="Open Video Folder",
            key=f"{key_prefix}_open_video_folder",
        )

    with st.expander("Save a snapshot", expanded=False):
        try:
            properties = video_properties(path)
            duration = max(0.0, float(properties.get("duration_s", 0.0)))
            fps = max(0.0, float(properties.get("fps", 0.0)))
            step = round(1.0 / fps, 4) if fps > 0 else 0.1
            upper = max(duration, step)
            snapshot_time = st.number_input(
                "Snapshot time (s)",
                min_value=0.0,
                max_value=float(upper),
                value=0.0,
                step=float(step),
                format="%.3f",
                key=f"{key_prefix}_snapshot_time",
                help="Enter the playback time shown in the video, then save that frame.",
            )
            snap_left, snap_right = st.columns(2)
            with snap_left:
                if st.button(
                    "Save Snapshot to Downloads",
                    key=f"{key_prefix}_save_snapshot",
                    use_container_width=True,
                ):
                    try:
                        destination = save_video_snapshot_to_downloads(
                            path,
                            project_path=project,
                            time_s=float(snapshot_time),
                            output_tag=output_tag,
                            source_tag=source_tag,
                        )
                        st.success(f"Saved to Downloads: {destination.name}")
                        st.caption(str(destination))
                    except Exception as exc:
                        st.error(f"Snapshot could not be saved: {exc}")
            with snap_right:
                show_open_downloads_button(
                    key=f"{key_prefix}_open_downloads",
                )
            st.caption(
                "The embedded video player does not expose its current playback time to Streamlit. "
                "Use the time displayed by the player and enter it above."
            )
        except Exception as exc:
            st.warning(f"Snapshot controls are unavailable: {exc}")


def show_pose_video_controls(
    project: str | Path,
    camera_key: str = "camera_a",
    key_prefix: str = "pose_review",
    heading: str = "Pose overlay review",
    expanded: bool = True,
) -> None:
    del camera_key  # retained for backwards-compatible calls
    project = Path(project)
    metadata = load_metadata(project)
    dual = metadata.get("capture_mode") == "dual_camera_visual_sync"
    with st.expander(heading, expanded=expanded):
        top_left, top_right = st.columns([4, 1.4])
        with top_left:
            st.caption(f"Project folder: {project}")
        with top_right:
            show_open_folder_button(
                project,
                label="Open Project Folder",
                key=f"{key_prefix}_open_project",
            )

        if dual:
            setup = metadata.get("analysis_config", {}).get("camera_setup", {})
            side_a = str(setup.get("camera_a_side", "left")).title()
            side_b = str(setup.get("camera_b_side", "right")).title()
            tab_a, tab_b, tab_both = st.tabs(
                [f"Camera A - {side_a}", f"Camera B - {side_b}", "Side-by-side"]
            )
            video_a = project / "camera_a" / "recording_with_pose.mp4"
            video_b = project / "camera_b" / "recording_with_pose.mp4"
            with tab_a:
                show_video_preview(
                    video_a,
                    f"Camera A - {side_a}",
                    key_prefix=f"{key_prefix}_camera_a",
                )
            with tab_b:
                show_video_preview(
                    video_b,
                    f"Camera B - {side_b}",
                    key_prefix=f"{key_prefix}_camera_b",
                )
            with tab_both:
                left, right = st.columns(2)
                with left:
                    show_video_preview(
                        video_a,
                        f"Camera A - {side_a}",
                        key_prefix=f"{key_prefix}_both_camera_a",
                    )
                with right:
                    show_video_preview(
                        video_b,
                        f"Camera B - {side_b}",
                        key_prefix=f"{key_prefix}_both_camera_b",
                    )
        else:
            show_video_preview(
                project / "recording_with_pose.mp4",
                "Single-camera full-body overlay",
                key_prefix=f"{key_prefix}_single",
            )
