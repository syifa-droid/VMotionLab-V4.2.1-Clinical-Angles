from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fpdf import FPDF


def _text(value) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def generate_com_report(project_path: str | Path) -> Path:
    project = Path(project_path)
    com_path = project / "com_analysis.csv"
    if not com_path.exists():
        raise FileNotFoundError("Generate COM analysis before the COM report.")
    com = pd.read_csv(com_path)
    summary_path = project / "com_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata_path = project / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    report_dir = project / "report"
    report_dir.mkdir(exist_ok=True)
    destination = report_dir / "com_report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _text("VMotionLab V4.2 - Experimental 2D Projected COM Report"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        _text(
            "Pose engine: RTMPose Body_with_feet (HALPE-26). This is a camera-plane projection "
            "based on anthropometric segment assumptions, not a validated 3D COM or centre of pressure."
        ),
    )
    pdf.ln(3)
    for label, value in [
        ("Project", metadata.get("project_name", project.name)),
        ("Participant", metadata.get("subject_id", "Unknown")),
        ("Task", metadata.get("task", "Unknown")),
        ("COM source", metadata.get("com_source_label", metadata.get("com_source_camera", "Single camera"))),
        ("Valid frames", f"{summary.get('valid_frame_percent', float('nan')):.1f}%"),
        ("Mean valid mass", f"{summary.get('mean_valid_mass_fraction', float('nan')) * 100:.1f}%"),
        ("Horizontal excursion", f"{summary.get('horizontal_excursion_bh', float('nan')):.4f} body heights"),
        ("Vertical excursion", f"{summary.get('vertical_excursion_bh', float('nan')):.4f} body heights"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(52, 7, _text(label + ":"), border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _text(value), ln=1)

    phase_path = project / "movement_phase_summary.csv"
    if phase_path.exists():
        phases = pd.read_csv(phase_path)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Movement-phase summary", ln=1)
        pdf.set_font("Helvetica", "", 8)
        for _, row in phases.head(30).iterrows():
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                5,
                _text(
                    f"{row.get('phase', '')}: {row.get('start_time_s', '')} to "
                    f"{row.get('end_time_s', '')} s; duration {row.get('duration_s', '')} s"
                ),
            )

    for figure_name in [
        "com_vertical_figure.png",
        "com_horizontal_figure.png",
        "com_trajectory_figure.png",
    ]:
        figure = project / "com_figures" / figure_name
        if figure.exists():
            if pdf.get_y() > 205:
                pdf.add_page()
            pdf.image(str(figure), x=15, w=180)
            pdf.ln(3)

    pdf.output(str(destination))
    return destination
