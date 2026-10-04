from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fpdf import FPDF


def _text(value) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def generate_report(project_path: str | Path) -> Path:
    project = Path(project_path)
    statistics_path = project / "statistics.csv"
    if not statistics_path.exists():
        raise FileNotFoundError("Generate kinematics before the PDF report.")
    statistics = pd.read_csv(statistics_path)
    comparison_path = project / "comparison_statistics.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    metadata_path = project / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    report_dir = project / "report"
    report_dir.mkdir(exist_ok=True)
    destination = report_dir / "report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _text("VMotionLab V4.2 - RTMPose 2D Kinematics Report"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        _text(
            "This report presents projected 2D joint angles derived from RTMPose/HALPE-26 landmarks. "
            "Results are experimental and camera-view dependent."
        ),
    )
    for label, value in [
        ("Participant", metadata.get("subject_id", "Unknown")),
        ("Task", metadata.get("task", "Unknown")),
        ("Trial", metadata.get("trial_name", project.name)),
        ("Capture mode", metadata.get("capture_mode", "single_camera")),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 7, _text(label + ":"))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _text(value), ln=1)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Filtered statistics", ln=1)
    pdf.set_font("Helvetica", "", 8)
    for _, row in statistics.iterrows():
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0,
            5,
            _text(
                f"{row.get('label', '')}: min {row.get('minimum_deg', float('nan')):.2f}, "
                f"max {row.get('maximum_deg', float('nan')):.2f}, "
                f"ROM {row.get('rom_deg', float('nan')):.2f}, "
                f"valid {row.get('valid_percent', float('nan')):.1f}%"
            ),
        )

    if not comparison.empty:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Direct-versus-virtual comparison", ln=1)
        pdf.set_font("Helvetica", "", 8)
        for _, row in comparison.iterrows():
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                5,
                _text(
                    f"{row.get('side', '')} {row.get('joint', '')}: "
                    f"RMSE {row.get('rmse_deg', float('nan')):.2f} deg; "
                    f"MAE {row.get('mae_deg', float('nan')):.2f} deg"
                ),
            )

    for figure in sorted((project / "figures").glob("*.png")) if (project / "figures").exists() else []:
        if pdf.get_y() > 205:
            pdf.add_page()
        pdf.image(str(figure), x=15, w=180)
        pdf.ln(3)
    pdf.output(str(destination))
    return destination
