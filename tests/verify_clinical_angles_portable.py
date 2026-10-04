from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from vmotionlab.modules.geometry import (
    clinical_ankle_dorsiflexion,
    clinical_hip_flexion,
    clinical_knee_flexion,
)


def p(x: float, y: float) -> np.ndarray:
    return np.array([[float(x), float(y)]], dtype=float)


tests = [
    ("Neutral hip", clinical_hip_flexion(p(0, 0), p(0, 100), p(0, 200), 1)[0], "zero"),
    ("Hip flexion", clinical_hip_flexion(p(0, 0), p(0, 100), p(50, 200), 1)[0], "positive"),
    ("Hip extension", clinical_hip_flexion(p(0, 0), p(0, 100), p(-50, 200), 1)[0], "negative"),

    ("Neutral knee", clinical_knee_flexion(p(0, 100), p(0, 200), p(0, 300), 1)[0], "zero"),
    ("Knee flexion", clinical_knee_flexion(p(0, 100), p(50, 200), p(0, 300), 1)[0], "positive"),
    ("Knee hyperextension", clinical_knee_flexion(p(0, 100), p(-20, 200), p(0, 300), 1)[0], "negative"),

    ("Neutral ankle", clinical_ankle_dorsiflexion(p(0, 200), p(0, 300), p(100, 300))[0], "zero"),
    ("Ankle dorsiflexion", clinical_ankle_dorsiflexion(p(40, 200), p(0, 300), p(100, 300))[0], "positive"),
    ("Ankle plantarflexion", clinical_ankle_dorsiflexion(p(0, 200), p(0, 300), p(100, 340))[0], "negative"),

    ("Mirrored hip flexion", clinical_hip_flexion(p(0, 0), p(0, 100), p(-50, 200), -1)[0], "positive"),
    ("Mirrored knee flexion", clinical_knee_flexion(p(0, 100), p(-50, 200), p(0, 300), -1)[0], "positive"),
    ("Mirrored ankle dorsiflexion", clinical_ankle_dorsiflexion(p(-40, 200), p(0, 300), p(-100, 300))[0], "positive"),
]


def okay(value: float, expectation: str) -> bool:
    if expectation == "zero":
        return abs(float(value)) < 1e-6
    if expectation == "positive":
        return float(value) > 0
    if expectation == "negative":
        return float(value) < 0
    return False


print("VMotionLab V4.2.1 Clinical-Angle Verification")
print("=" * 50)
failed: list[str] = []

for name, value, expectation in tests:
    passed = okay(float(value), expectation)
    print(f"{name:28s} {float(value):9.3f} deg   {'PASS' if passed else 'FAIL'}")
    if not passed:
        failed.append(name)

print()
if failed:
    print("RESULT: FAILED")
    print("Failed tests:", ", ".join(failed))
    raise SystemExit(1)

print("RESULT: ALL TESTS PASSED")
print()
print("Clinical conventions:")
print("  Hip:   flexion +, extension -")
print("  Knee:  flexion +, hyperextension -")
print("  Ankle: dorsiflexion +, plantarflexion -")
