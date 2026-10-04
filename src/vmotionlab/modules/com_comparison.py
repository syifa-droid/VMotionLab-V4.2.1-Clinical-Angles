from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class COMComparison:
    curves: pd.DataFrame
    summary: pd.DataFrame
    x_label: str


def build_com_comparison(project_paths, normalize_time: bool = True, axis: str | None = None) -> COMComparison:
    curve_rows = []
    summary_rows = []
    for item in project_paths:
        project = Path(item)
        path = project / "com_analysis.csv"
        if not path.exists():
            # Compatibility with early V4 outputs.
            candidates = sorted(project.glob("com_analysis_*.csv"))
            if not candidates:
                continue
            path = candidates[0]
        data = pd.read_csv(path)
        label = project.name
        time = pd.to_numeric(data["time"], errors="coerce")
        if normalize_time:
            duration = float(time.max() - time.min())
            axis_value = (time - time.min()) / max(duration, np.finfo(float).eps) * 100.0
            x_label = "Trial duration (%)"
        else:
            axis_value = time
            x_label = "Time (s)"
        for index, row in data.iterrows():
            curve_rows.append(
                {
                    "trial_label": label,
                    "axis_value": float(axis_value.iloc[index]),
                    "time": float(time.iloc[index]),
                    "com_horizontal_bh": row.get("com_horizontal_bh", np.nan),
                    "com_vertical_bh": row.get("com_vertical_bh", np.nan),
                }
            )
        summary_rows.append(
            {
                "trial_label": label,
                "valid_frames_percent": float(data.get("valid_frame", pd.Series(False)).mean() * 100.0),
                "horizontal_excursion_bh": float(data["com_horizontal_bh"].max() - data["com_horizontal_bh"].min()),
                "vertical_excursion_bh": float(data["com_vertical_bh"].max() - data["com_vertical_bh"].min()),
                "mean_valid_mass_fraction": float(data["valid_mass_fraction"].mean()),
                "duration_s": float(time.max() - time.min()),
            }
        )
    return COMComparison(
        curves=pd.DataFrame(curve_rows),
        summary=pd.DataFrame(summary_rows),
        x_label=x_label if curve_rows else ("Trial duration (%)" if normalize_time else "Time (s)"),
    )
