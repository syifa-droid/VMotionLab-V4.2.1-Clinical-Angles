from __future__ import annotations

import os
from pathlib import Path
import sys


def app_root() -> Path:
    configured = os.environ.get("VMOTIONLAB_APP_ROOT")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    configured = os.environ.get("VMOTIONLAB_RESOURCE_ROOT")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    configured = os.environ.get("VMOTIONLAB_DATA_ROOT")
    if configured:
        return Path(configured).resolve()
    return app_root() / "data"


def logs_root() -> Path:
    return app_root() / "logs"
