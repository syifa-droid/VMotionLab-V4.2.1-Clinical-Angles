from __future__ import annotations

from pathlib import Path
import streamlit as st

from .project_manager import get_current_projects, set_current_projects


def initialize_state() -> None:
    saved = [str(path) for path in get_current_projects()]
    st.session_state.setdefault("active_project_paths", saved)
    if saved:
        st.session_state.setdefault("active_project_path", saved[0])


def require_active_projects() -> list[Path]:
    projects = get_current_projects()
    if not projects:
        st.warning("No processed trial is selected.")
        if st.button("Go to Upload & Pose Review", use_container_width=True):
            st.switch_page("pages/2_Upload_Video.py")
        st.stop()
    set_current_projects(projects)
    return projects
