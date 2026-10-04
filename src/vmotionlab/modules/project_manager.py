from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

from .runtime_paths import app_root, data_root

ROOT = app_root()
DATA = data_root()
PROJECTS = DATA / "projects"
CURRENT = DATA / "current_projects.json"
PROJECTS.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    return cleaned or "Trial"


def project_name(subject_id: str, task: str, trial_name: str) -> str:
    return f"{safe_name(subject_id)}_{safe_name(task)}_{safe_name(trial_name)}"


def get_project_path(subject_id: str, task: str | None = None, trial_name: str | None = None) -> Path:
    if task is None and trial_name is None:
        return PROJECTS / safe_name(subject_id)
    return PROJECTS / project_name(subject_id, task or "Unknown", trial_name or "Trial")


def create_project(
    subject_id: str,
    task: str,
    trial_name: str,
    side: str = "Both",
    notes: str = "",
    source_video_name: str = "",
    analysis_config: dict | None = None,
    overwrite: bool = False,
) -> dict:
    path = get_project_path(subject_id, task, trial_name)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(path) or {}
    metadata.update(
        {
            "app": "VMotionLabV4.2",
            "project_name": path.name,
            "subject_id": subject_id,
            "task": task,
            "trial_name": trial_name,
            "side": side,
            "notes": notes,
            "source_video_name": source_video_name,
            "analysis_config": analysis_config or metadata.get("analysis_config", {}),
            "created_at": metadata.get("created_at", datetime.now().isoformat(timespec="seconds")),
            "current_step": metadata.get("current_step", "created"),
        }
    )
    save_metadata(path, metadata)
    return {**metadata, "project_path": str(path)}


def save_metadata(project: str | Path, data: dict) -> None:
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["project_path"] = str(project.resolve())
    payload.setdefault("project_name", project.name)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    (project / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_metadata(project: str | Path) -> dict:
    path = Path(project) / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_project_metadata() -> list[dict]:
    rows: list[dict] = []
    if not PROJECTS.exists():
        return rows
    for path in sorted(PROJECTS.iterdir()):
        if not path.is_dir():
            continue
        metadata = load_metadata(path)
        if not metadata:
            continue
        metadata["project_path"] = str(path)
        metadata.setdefault("project_name", path.name)
        rows.append(metadata)
    return rows


def find_completed_projects(required_files: Iterable[str] | None = None) -> list[dict]:
    required = list(required_files or [])
    rows: list[dict] = []
    for metadata in list_project_metadata():
        path = Path(metadata["project_path"])
        if all((path / name).exists() for name in required):
            rows.append(metadata)
    return rows


def set_current_projects(paths: Iterable[str | Path]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    values = [str(Path(path).resolve()) for path in paths]
    CURRENT.write_text(json.dumps(values, indent=2), encoding="utf-8")


def get_current_projects() -> list[Path]:
    if not CURRENT.exists():
        return []
    try:
        values = json.loads(CURRENT.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [Path(value) for value in values if Path(value).exists()]
