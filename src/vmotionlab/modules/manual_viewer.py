from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from .runtime_paths import resource_root

DEFAULT_WATERMARK = "VMotionLab V4.2 — Syifa Fauziah"
ROOT = resource_root()
MANUALS = ROOT / "manuals"


def list_manuals() -> list[Path]:
    MANUALS.mkdir(parents=True, exist_ok=True)
    return sorted(MANUALS.glob("*.pdf"))


@lru_cache(maxsize=16)
def get_manual_information(path: str, modified_ns: int) -> dict:
    del modified_ns
    document = fitz.open(path)
    toc = [
        {"level": int(level), "title": str(title), "page": int(page)}
        for level, title, page in document.get_toc(simple=True)
        if page > 0
    ]
    title = document.metadata.get("title") or Path(path).stem
    information = {"title": title, "page_count": document.page_count, "toc": toc}
    document.close()
    return information


@lru_cache(maxsize=64)
def render_manual_page(
    path: str,
    modified_ns: int,
    page_number: int,
    zoom_percent: int,
    watermark: str,
) -> tuple[bytes, int, int]:
    del modified_ns
    document = fitz.open(path)
    page = document.load_page(max(0, int(page_number) - 1))
    scale = max(0.5, float(zoom_percent) / 100.0) * 1.4
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = watermark
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    draw.rectangle(
        [image.width - text_width - 22, image.height - 30, image.width - 6, image.height - 8],
        fill=(255, 255, 255),
    )
    draw.text((image.width - text_width - 14, image.height - 26), text, fill=(90, 90, 90), font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    document.close()
    return buffer.getvalue(), image.width, image.height


def show_protected_page(png_bytes: bytes, image_width: int, image_height: int) -> None:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;overflow:auto;user-select:none;"
             oncontextmenu="return false;">
          <img src="data:image/png;base64,{encoded}" width="{image_width}" height="{image_height}"
               draggable="false" style="max-width:100%;height:auto;box-shadow:0 2px 12px rgba(0,0,0,.18);" />
        </div>
        """,
        unsafe_allow_html=True,
    )
