from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PhaseResult:
    frames: pd.DataFrame
    summary: pd.DataFrame
    task: str


def _runs(labels: pd.Series) -> list[tuple[int, int, str]]:
    runs = []
    start = 0
    current = str(labels.iloc[0]) if len(labels) else "unknown"
    for index in range(1, len(labels)):
        label = str(labels.iloc[index])
        if label != current:
            runs.append((start, index - 1, current))
            start = index
            current = label
    if len(labels):
        runs.append((start, len(labels) - 1, current))
    return runs


def detect_movement_phases(com: pd.DataFrame, task: str, config: dict | None = None) -> PhaseResult:
    config = config or {}
    excursion_threshold = float(config.get("minimum_excursion_body_heights", 0.03))
    data = com.copy()
    y = pd.to_numeric(data["com_vertical_bh"], errors="coerce")
    y_interp = y.interpolate(limit_direction="both")
    velocity = np.gradient(y_interp.to_numpy(float))
    amplitude = float(y_interp.max() - y_interp.min())

    if not np.isfinite(amplitude) or amplitude < excursion_threshold:
        labels = pd.Series("insufficient_excursion", index=data.index)
    else:
        low = float(y_interp.min() + 0.20 * amplitude)
        high = float(y_interp.min() + 0.80 * amplitude)
        labels = pd.Series("transition", index=data.index, dtype=object)
        if task == "squat":
            labels[y_interp >= high] = "standing"
            labels[y_interp <= low] = "bottom"
            labels[(labels == "transition") & (velocity < 0)] = "descent"
            labels[(labels == "transition") & (velocity >= 0)] = "ascent"
        else:
            labels[y_interp <= low] = "sitting"
            labels[y_interp >= high] = "standing"
            labels[(labels == "transition") & (velocity > 0)] = "rising"
            labels[(labels == "transition") & (velocity <= 0)] = "lowering"

    frames = data[["frame", "time"]].copy()
    frames["phase"] = labels
    rows = []
    for repetition, (start, end, label) in enumerate(_runs(labels), start=1):
        rows.append(
            {
                "sequence": repetition,
                "phase": label,
                "start_frame": int(data.loc[start, "frame"]),
                "end_frame": int(data.loc[end, "frame"]),
                "start_time_s": float(data.loc[start, "time"]),
                "end_time_s": float(data.loc[end, "time"]),
                "duration_s": float(data.loc[end, "time"] - data.loc[start, "time"]),
            }
        )
    return PhaseResult(frames=frames, summary=pd.DataFrame(rows), task=task)


def save_phase_outputs(project_path: str | Path, result: PhaseResult) -> None:
    project = Path(project_path)
    result.frames.to_csv(project / "movement_phases.csv", index=False)
    result.summary.to_csv(project / "movement_phase_summary.csv", index=False)
    (project / "movement_phase_summary.json").write_text(
        json.dumps(
            {
                "task": result.task,
                "phase_count": int(len(result.summary)),
                "phases": result.summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
