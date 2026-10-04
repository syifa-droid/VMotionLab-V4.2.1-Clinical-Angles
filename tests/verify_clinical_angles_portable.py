from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODULES = BASE / "modules"

EXPECTED_HASHES = {
    "geometry.py": "19142e39a69d197288cd385723c2ffad00948fe2934271c9346ea34abe9e8c7a",
    "kinematics.py": "df8b97862d57babc3cfbb6b0533ee6c54d5cc005bda64c4b3f46d21915632584",
    "kinematics_engine.py": "5f0fd717abffd2328cb3a71327fe555202de4806b4b3da3b802cc499c531fc30",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_angle(u, v):
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    return math.degrees(math.atan2(cross, dot))


def to_clinical(point, forward_sign):
    x, y = point
    return (x * forward_sign, -y)


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1])


def internal_angle(a, b, c):
    ba = subtract(a, b)
    bc = subtract(c, b)
    den = math.hypot(*ba) * math.hypot(*bc)
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / den
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def hip(shoulder, hip_point, knee, forward_sign=1):
    s = to_clinical(shoulder, forward_sign)
    h = to_clinical(hip_point, forward_sign)
    k = to_clinical(knee, forward_sign)
    trunk_down = subtract(h, s)
    thigh_down = subtract(k, h)
    return signed_angle(trunk_down, thigh_down)


def knee_angle(hip_point, knee_point, ankle, forward_sign=1):
    h = to_clinical(hip_point, forward_sign)
    k = to_clinical(knee_point, forward_sign)
    a = to_clinical(ankle, forward_sign)
    thigh_down = subtract(k, h)
    shank_down = subtract(a, k)
    return -signed_angle(thigh_down, shank_down)


def ankle(knee_point, ankle_point, foot):
    return 90.0 - internal_angle(knee_point, ankle_point, foot)


def expected(value, mode):
    if mode == "zero":
        return abs(value) < 1e-6
    if mode == "positive":
        return value > 0
    if mode == "negative":
        return value < 0
    return False


print("VMotionLab V4.2.1 Portable Clinical-Angle Verification")
print("=" * 58)
print(f"Python used only for verification: {sys.version.split()[0]}")
print(f"Checking modules in: {MODULES}")
print()

all_ok = True

print("A. Installed file check")
for name, expected_hash in EXPECTED_HASHES.items():
    path = MODULES / name
    if not path.exists():
        print(f"  {name:24s} MISSING")
        all_ok = False
        continue

    actual = sha256(path)
    ok = actual == expected_hash
    print(f"  {name:24s} {'PASS' if ok else 'DIFFERENT'}")
    if not ok:
        print(f"    actual:   {actual}")
        print(f"    expected: {expected_hash}")
        all_ok = False

print()
print("B. Clinical convention check")

tests = [
    ("Neutral hip", hip((0, 0), (0, 100), (0, 200), 1), "zero"),
    ("Hip flexion", hip((0, 0), (0, 100), (50, 200), 1), "positive"),
    ("Hip extension", hip((0, 0), (0, 100), (-50, 200), 1), "negative"),

    ("Neutral knee", knee_angle((0, 100), (0, 200), (0, 300), 1), "zero"),
    ("Knee flexion", knee_angle((0, 100), (50, 200), (0, 300), 1), "positive"),
    ("Knee hyperextension", knee_angle((0, 100), (-20, 200), (0, 300), 1), "negative"),

    ("Neutral ankle", ankle((0, 200), (0, 300), (100, 300)), "zero"),
    ("Ankle dorsiflexion", ankle((40, 200), (0, 300), (100, 300)), "positive"),
    ("Ankle plantarflexion", ankle((0, 200), (0, 300), (100, 340)), "negative"),

    ("Mirrored hip flexion", hip((0, 0), (0, 100), (-50, 200), -1), "positive"),
    ("Mirrored knee flexion", knee_angle((0, 100), (-50, 200), (0, 300), -1), "positive"),
    ("Mirrored ankle dorsiflexion", ankle((-40, 200), (0, 300), (-100, 300)), "positive"),
]

for name, value, mode in tests:
    ok = expected(value, mode)
    print(f"  {name:28s} {value:9.3f} deg   {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("RESULT: ALL CHECKS PASSED")
    print()
    print("Clinical conventions installed:")
    print("  Hip:   flexion +, extension -")
    print("  Knee:  flexion +, hyperextension -")
    print("  Ankle: dorsiflexion +, plantarflexion -")
    raise SystemExit(0)

print("RESULT: CHECK FAILED")
print("Do not overwrite additional files. Restore the backup if VMotionLab fails to open.")
raise SystemExit(1)
