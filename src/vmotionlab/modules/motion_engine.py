from __future__ import annotations

from pathlib import Path

from .runtime_paths import resource_root

DEFAULT_POSE_MODEL_PATH = resource_root() / "resources" / "Pose2Sim_Config.toml"
MODEL_DIRECTORY = resource_root() / "resources" / "models"
DETECTOR_MODEL = MODEL_DIRECTORY / "yolox_m_humanart.onnx"
POSE_MODEL = MODEL_DIRECTORY / "rtmpose_m_halpe26.onnx"


def pose_model_exists(path: Path = DEFAULT_POSE_MODEL_PATH) -> bool:
    if not Path(path).exists():
        return False
    if not DETECTOR_MODEL.exists() or not POSE_MODEL.exists():
        return False
    try:
        from Pose2Sim import Pose2Sim  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def get_pose_model_message(path: Path = DEFAULT_POSE_MODEL_PATH) -> str:
    if not Path(path).exists():
        return f"Pose2Sim configuration is missing: {path}"

    missing = [
        str(model)
        for model in (DETECTOR_MODEL, POSE_MODEL)
        if not model.exists()
    ]
    if missing:
        return (
            "The bundled RTMPose model files are missing. Rebuild the Desktop "
            "Edition. Missing: " + "; ".join(missing)
        )

    return (
        "Pose2Sim/RTMPose is unavailable in the Desktop Edition runtime. "
        "Review logs/desktop_launcher.log and logs/streamlit_server.log."
    )
