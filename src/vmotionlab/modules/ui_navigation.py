from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import streamlit as st

from .project_manager import PROJECTS, get_current_projects, load_metadata

PAGES = {
    "upload": "pages/2_Upload_Video.py",
    "upload_video": "pages/2_Upload_Video.py",
    "sync": "pages/3_Temporal_Synchronization.py",
    "synchronization": "pages/3_Temporal_Synchronization.py",
    "preprocess": "pages/4_Preprocessing.py",
    "preprocessing": "pages/4_Preprocessing.py",
    "kinematics": "pages/5_Kinematics.py",
    "com": "pages/6_COM_Analysis.py",
    "report": "pages/7_Report.py",
}


def _folder_key(path: Path, prefix: str) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def open_folder(path: str | Path) -> tuple[bool, str]:
    folder = Path(path).resolve()
    if not folder.exists():
        return False, f"Folder does not exist: {folder}"
    try:
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return True, str(folder)
    except Exception as exc:
        return False, str(exc)


def show_open_folder_button(
    path: str | Path,
    label: str = "Open Project Folder",
    key: str | None = None,
    use_container_width: bool = True,
) -> None:
    folder = Path(path)
    button_key = key or _folder_key(folder, "open_folder")
    if st.button(label, key=button_key, use_container_width=use_container_width):
        ok, message = open_folder(folder)
        if ok:
            st.success("Folder opened in File Explorer.")
        else:
            st.error(f"Unable to open folder: {message}")


def show_projects_root_button(
    label: str = "Open All Projects Folder",
    key: str = "open_projects_root",
    use_container_width: bool = True,
) -> None:
    show_open_folder_button(
        PROJECTS,
        label=label,
        key=key,
        use_container_width=use_container_width,
    )


def show_project_header() -> None:
    projects = get_current_projects()
    if not projects:
        return
    labels = []
    for project in projects:
        metadata = load_metadata(project)
        labels.append(
            f"{metadata.get('subject_id', 'Unknown')} / {metadata.get('task', 'Unknown')} / "
            f"{metadata.get('trial_name', project.name)}"
        )
    left, right = st.columns([5, 1.4])
    with left:
        st.caption("Selected trial(s): " + " | ".join(labels))
    with right:
        show_open_folder_button(
            projects[0],
            label="Open Project Folder",
            key=_folder_key(projects[0], "header_open"),
        )


def _pose_ready(project: Path, metadata: dict) -> bool:
    if metadata.get("capture_mode") == "dual_camera_visual_sync":
        return all(
            (project / camera / "pose_landmarks_full.csv").exists()
            for camera in ("camera_a", "camera_b")
        )
    return (project / "pose_landmarks_full.csv").exists()


def _com_ready(project: Path) -> bool:
    return (
        (project / "com_analysis.csv").exists()
        or (project / "com_camera_a" / "com_analysis.csv").exists()
        or (project / "com_camera_b" / "com_analysis.csv").exists()
    )


def workflow_checks(project: str | Path) -> list[tuple[str, bool]]:
    project = Path(project)
    metadata = load_metadata(project)
    return [
        ("Pose", _pose_ready(project, metadata)),
        ("Sync", (project / "synchronization.json").exists()),
        ("Preprocess", (project / "motion_filtered.csv").exists()),
        ("Kinematics", (project / "statistics.csv").exists()),
        ("COM", _com_ready(project)),
        ("Report", (project / "report" / "report.pdf").exists()),
    ]


def show_workflow_status(project: str | Path) -> None:
    columns = st.columns(6)
    for column, (name, ready) in zip(columns, workflow_checks(project)):
        css_class = "workflow-ready" if ready else "workflow-pending"
        state = "Ready" if ready else "Not ready"
        column.markdown(
            f"""
            <div class="workflow-card {css_class}">
                <div class="workflow-name">{name}</div>
                <div class="workflow-state">{state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def workflow_navigation(
    back_step: str | None = None,
    next_step: str | None = None,
    next_enabled: bool = True,
    next_disabled_message: str = "Complete the current step first.",
) -> None:
    left, right = st.columns(2)
    if back_step:
        if left.button("Back", use_container_width=True):
            st.switch_page(PAGES[back_step])
    if next_step:
        if right.button(
            "Next",
            type="primary",
            use_container_width=True,
            disabled=not next_enabled,
            help=None if next_enabled else next_disabled_message,
        ):
            st.switch_page(PAGES[next_step])
