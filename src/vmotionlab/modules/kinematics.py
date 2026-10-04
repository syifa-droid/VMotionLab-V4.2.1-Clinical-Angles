from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLINICAL_JOINT_LABELS = {
    "hip": "Hip Flexion (+) / Extension (-)",
    "knee": "Knee Flexion (+) / Hyperextension (-)",
    "ankle": "Ankle Dorsiflexion (+) / Plantarflexion (-)",
}

def _clinical_joint_label(joint: str) -> str:
    return CLINICAL_JOINT_LABELS.get(joint, joint.title())


def _angle_column(joint: str, side: str, method: str) -> str:
    suffix = "l" if side == "left" else "r"
    base = "ankle_angle" if joint == "ankle" else f"{joint}_flexion"
    return f"{base}_{suffix}_{method}"


def _statistics(values: pd.Series, time: pd.Series, label: str) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return {
            "label": label,
            "minimum_deg": np.nan,
            "maximum_deg": np.nan,
            "rom_deg": np.nan,
            "mean_deg": np.nan,
            "sd_deg": np.nan,
            "peak_time_s": np.nan,
            "valid_frames": 0,
            "valid_percent": 0.0,
        }
    peak_index = numeric.idxmax()
    return {
        "label": label,
        "minimum_deg": float(valid.min()),
        "maximum_deg": float(valid.max()),
        "rom_deg": float(valid.max() - valid.min()),
        "mean_deg": float(valid.mean()),
        "sd_deg": float(valid.std()),
        "peak_time_s": float(time.loc[peak_index]),
        "valid_frames": int(valid.size),
        "valid_percent": float(valid.size / len(numeric) * 100.0),
    }


def _normalise(time: pd.Series, values: pd.Series, points: int = 101) -> np.ndarray:
    x = pd.to_numeric(time, errors="coerce").to_numpy(float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(points, np.nan)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, indices = np.unique(x, return_index=True)
    y = y[indices]
    x = unique_x
    x_norm = (x - x[0]) / max(x[-1] - x[0], np.finfo(float).eps) * 100.0
    return np.interp(np.linspace(0.0, 100.0, points), x_norm, y)


def run_kinematics_analysis(project_path: str | Path, config: dict | None = None) -> dict:
    project = Path(project_path)
    filtered_path = project / "motion_filtered.csv"
    if not filtered_path.exists():
        raise FileNotFoundError("Run preprocessing before kinematics.")
    data = pd.read_csv(filtered_path)
    config = config or {}
    figures = project / "figures"
    figures.mkdir(exist_ok=True)

    statistics_rows: list[dict] = []
    comparison_rows: list[dict] = []
    normalized = pd.DataFrame({"cycle_percent": np.linspace(0.0, 100.0, 101)})

    for side in config.get("selected_sides", ["right"]):
        for joint in config.get("selected_joints", ["hip", "knee", "ankle"]):
            selected_method = config.get("marker_methods", {}).get(joint, "direct")
            methods = ["direct", "virtual"] if selected_method == "compare" else [selected_method]

            plt.figure(figsize=(8.5, 4.5))
            plotted = False
            for method in methods:
                column = _angle_column(joint, side, method)
                if column not in data:
                    continue
                label = f"{side.title()} {_clinical_joint_label(joint)} {method.title()}"
                statistics_rows.append(_statistics(data[column], data["time"], label))
                normalized[column] = _normalise(data["time"], data[column])
                numeric_values = pd.to_numeric(data[column], errors="coerce")
                if numeric_values.notna().any():
                    plt.plot(data["time"], numeric_values, label=method.title())
                    plotted = True
            if plotted:
                plt.xlabel("Time (s)")
                plt.ylabel("Clinical angle (degrees)")
                plt.title(f"{side.title()} {_clinical_joint_label(joint)}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(figures / f"{side}_{joint}.png", dpi=160)
            plt.close()

            if selected_method == "compare":
                direct_column = _angle_column(joint, side, "direct")
                virtual_column = _angle_column(joint, side, "virtual")
                if direct_column in data and virtual_column in data:
                    paired = data[[direct_column, virtual_column]].dropna()
                    if paired.empty:
                        rmse = mae = bias = correlation = np.nan
                    else:
                        difference = paired[direct_column] - paired[virtual_column]
                        rmse = float(np.sqrt(np.mean(difference**2)))
                        mae = float(np.mean(np.abs(difference)))
                        bias = float(np.mean(difference))
                        correlation = (
                            float(paired.corr().iloc[0, 1])
                            if paired[direct_column].std() > 0
                            and paired[virtual_column].std() > 0
                            else np.nan
                        )
                    comparison_rows.append(
                        {
                            "side": side,
                            "joint": joint,
                            "paired_frames": int(len(paired)),
                            "rmse_deg": rmse,
                            "mae_deg": mae,
                            "mean_bias_direct_minus_virtual_deg": bias,
                            "correlation": correlation,
                            "rom_direct_deg": float(
                                data[direct_column].max() - data[direct_column].min()
                            ),
                            "rom_virtual_deg": float(
                                data[virtual_column].max() - data[virtual_column].min()
                            ),
                        }
                    )

    statistics = pd.DataFrame(statistics_rows)
    statistics.to_csv(project / "statistics.csv", index=False)
    normalized.to_csv(project / "normalized_curves.csv", index=False)
    comparison_path = project / "comparison_statistics.csv"
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    else:
        comparison_path.unlink(missing_ok=True)

    summary = {
        "statistics_rows": int(len(statistics_rows)),
        "comparison_rows": int(len(comparison_rows)),
        "figures_directory": str(figures),
        "clinical_angle_convention": {
            "hip": "flexion positive; extension negative",
            "knee": "flexion positive; hyperextension negative",
            "ankle": "dorsiflexion positive; plantarflexion negative",
        },
    }
    (project / "kinematics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
