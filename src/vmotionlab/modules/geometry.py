from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .constants import SIDE


def point(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"{name}_x", f"{name}_y"]].to_numpy(dtype=float)


def confidence(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    columns = [f"{name}_visibility" for name in names]
    return np.nanmin(df[columns].to_numpy(dtype=float), axis=1)


def internal_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unsigned internal angle ABC in degrees, range 0..180."""
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(ba * bc, axis=1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def signed_vector_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Signed rotation from vector u to v in degrees, range -180..180.

    This function assumes a conventional Cartesian plane (x right, y up).
    """
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = np.sum(u * v, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def forefoot(df: pd.DataFrame, side_map: dict[str, str]) -> np.ndarray:
    big = point(df, side_map["big_toe"])
    small = point(df, side_map["small_toe"])
    return np.nanmean(np.stack([big, small]), axis=0)


def _infer_forward_sign(
    df: pd.DataFrame,
    side_map: dict[str, str],
) -> tuple[float, float]:
    """Infer subject-forward direction from heel -> forefoot in image coordinates.

    Returns
    -------
    sign:
        +1 when forward is toward increasing image x, -1 when toward decreasing x.
    confidence:
        Median absolute horizontal component divided by median foot length.
        Values near 1 indicate a strong sagittal view; values near 0 indicate that
        forward direction is poorly observable (e.g. near-frontal view).

    A single robust sign is used for the entire trial so clinical signs cannot
    flip frame-to-frame.
    """
    heel = point(df, side_map["heel"])
    toe = forefoot(df, side_map)
    vector = toe - heel

    length = np.linalg.norm(vector, axis=1)
    dx = vector[:, 0]
    valid = np.isfinite(dx) & np.isfinite(length) & (length > 1e-9)

    if not valid.any():
        return 1.0, 0.0

    robust_dx = float(np.nanmedian(dx[valid]))
    robust_length = float(np.nanmedian(length[valid]))
    confidence_value = (
        abs(robust_dx) / robust_length
        if np.isfinite(robust_length) and robust_length > 1e-9
        else 0.0
    )

    if not np.isfinite(robust_dx) or abs(robust_dx) < 1e-9:
        return 1.0, float(confidence_value)

    return (1.0 if robust_dx > 0 else -1.0), float(confidence_value)


def _clinical_xy(points: np.ndarray, forward_sign: float) -> np.ndarray:
    """Map image coordinates to a subject-centred sagittal plane.

    Image y increases downward. Clinical plane uses y upward.
    x is mirrored when required so +x always means subject-forward.
    """
    output = np.asarray(points, dtype=float).copy()
    output[:, 0] *= float(forward_sign)
    output[:, 1] *= -1.0
    return output


def clinical_hip_flexion(
    shoulder: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray,
    forward_sign: float,
) -> np.ndarray:
    """2D clinical proxy: flexion positive, extension negative.

    Uses trunk-down versus thigh-down orientation. Because a single sagittal
    camera cannot measure pelvic tilt independently, this is a trunk-referenced
    2D hip flexion/extension proxy rather than a full 3D anatomical hip angle.
    """
    s = _clinical_xy(shoulder, forward_sign)
    h = _clinical_xy(hip, forward_sign)
    k = _clinical_xy(knee, forward_sign)

    trunk_down = h - s
    thigh_down = k - h
    return signed_vector_angle(trunk_down, thigh_down)


def clinical_knee_flexion(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    forward_sign: float,
) -> np.ndarray:
    """Knee flexion positive, hyperextension negative."""
    h = _clinical_xy(hip, forward_sign)
    k = _clinical_xy(knee, forward_sign)
    a = _clinical_xy(ankle, forward_sign)

    thigh_down = k - h
    shank_down = a - k

    # In a subject-forward Cartesian plane, normal knee flexion rotates the
    # shank clockwise relative to the thigh, hence the negative sign.
    return -signed_vector_angle(thigh_down, shank_down)


def clinical_ankle_dorsiflexion(
    knee: np.ndarray,
    ankle: np.ndarray,
    foot: np.ndarray,
) -> np.ndarray:
    """Ankle dorsiflexion positive, plantarflexion negative.

    Neutral shank-foot geometry is approximately 90 degrees. This unsigned
    geometric relation is orientation-invariant, so it remains valid whether
    the participant faces left or right in the image.
    """
    return 90.0 - internal_angle(knee, ankle, foot)


def constrained_middle(
    proximal: np.ndarray,
    measured: np.ndarray,
    distal: np.ndarray,
    proximal_length: float,
    distal_length: float,
) -> np.ndarray:
    output = measured.copy()
    separation = np.linalg.norm(distal - proximal, axis=1)
    for index in range(len(output)):
        values = [*proximal[index], *measured[index], *distal[index]]
        if not np.isfinite(values).all() or separation[index] <= 1e-9:
            continue
        x = (
            proximal_length**2
            - distal_length**2
            + separation[index] ** 2
        ) / (2.0 * separation[index])
        height_squared = proximal_length**2 - x**2
        if height_squared < 0:
            continue
        direction = (distal[index] - proximal[index]) / separation[index]
        perpendicular = np.array([-direction[1], direction[0]])
        base = proximal[index] + x * direction
        height = np.sqrt(height_squared)
        candidate_1 = base + height * perpendicular
        candidate_2 = base - height * perpendicular
        output[index] = (
            candidate_1
            if np.linalg.norm(candidate_1 - measured[index])
            <= np.linalg.norm(candidate_2 - measured[index])
            else candidate_2
        )
    return output


def _robust_reference(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, percentile))


def compute_motion_signals(
    df: pd.DataFrame,
    side: str,
    reference_percentile: float = 95.0,
) -> pd.DataFrame:
    mapping = SIDE[side]
    suffix = "l" if side == "left" else "r"

    shoulder = point(df, mapping["shoulder"])
    hip = point(df, mapping["hip"])
    knee = point(df, mapping["knee"])
    ankle = point(df, mapping["ankle"])
    heel = point(df, mapping["heel"])
    foot = forefoot(df, mapping)

    trunk_length = np.linalg.norm(shoulder - hip, axis=1)
    thigh_length = np.linalg.norm(hip - knee, axis=1)
    shank_length = np.linalg.norm(knee - ankle, axis=1)
    foot_length = np.linalg.norm(ankle - foot, axis=1)

    refs = [
        _robust_reference(trunk_length, reference_percentile),
        _robust_reference(thigh_length, reference_percentile),
        _robust_reference(shank_length, reference_percentile),
        _robust_reference(foot_length, reference_percentile),
    ]

    virtual_hip = constrained_middle(shoulder, hip, knee, refs[0], refs[1])
    virtual_knee = constrained_middle(hip, knee, ankle, refs[1], refs[2])
    virtual_ankle = constrained_middle(knee, ankle, foot, refs[2], refs[3])

    hip_quality = confidence(df, [mapping["shoulder"], mapping["hip"], mapping["knee"]])
    knee_quality = confidence(df, [mapping["hip"], mapping["knee"], mapping["ankle"]])
    ankle_quality = confidence(
        df,
        [mapping["knee"], mapping["ankle"], mapping["big_toe"], mapping["small_toe"]],
    )

    forward_sign, forward_confidence = _infer_forward_sign(df, mapping)

    output = pd.DataFrame(
        {
            # Column names remain unchanged for V4.2 compatibility.
            # Values are now signed clinical conventions.
            f"hip_flexion_{suffix}_direct": clinical_hip_flexion(
                shoulder, hip, knee, forward_sign
            ),
            f"hip_flexion_{suffix}_virtual": clinical_hip_flexion(
                shoulder, virtual_hip, knee, forward_sign
            ),
            f"knee_flexion_{suffix}_direct": clinical_knee_flexion(
                hip, knee, ankle, forward_sign
            ),
            f"knee_flexion_{suffix}_virtual": clinical_knee_flexion(
                hip, virtual_knee, ankle, forward_sign
            ),
            f"ankle_angle_{suffix}_direct": clinical_ankle_dorsiflexion(
                knee, ankle, foot
            ),
            f"ankle_angle_{suffix}_virtual": clinical_ankle_dorsiflexion(
                knee, virtual_ankle, foot
            ),
            f"hip_quality_{suffix}": hip_quality,
            f"knee_quality_{suffix}": knee_quality,
            f"ankle_quality_{suffix}": ankle_quality,
            f"thigh_length_{suffix}_direct": thigh_length,
            f"shank_length_{suffix}_direct": shank_length,
            f"thigh_length_{suffix}_virtual": np.linalg.norm(
                virtual_hip - virtual_knee, axis=1
            ),
            f"shank_length_{suffix}_virtual": np.linalg.norm(
                virtual_knee - virtual_ankle, axis=1
            ),
            f"sagittal_forward_sign_{suffix}": float(forward_sign),
            f"sagittal_forward_confidence_{suffix}": float(forward_confidence),
        }
    )
    return output


def interpolate_short_gaps(series: pd.Series, maximum_gap: int) -> pd.Series:
    if maximum_gap <= 0:
        return series.copy()
    return series.interpolate(
        method="linear",
        limit=maximum_gap,
        limit_area="inside",
        limit_direction="both",
    )


def lowpass(series: pd.Series, fps: float, cutoff_hz: float, order: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    output = values.copy()
    if not valid.any() or fps <= 0:
        return pd.Series(output, index=series.index)

    nyquist = fps / 2.0
    cutoff = min(max(float(cutoff_hz), 0.01), nyquist * 0.99)
    sos = butter(int(order), cutoff / nyquist, btype="low", output="sos")

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))

    minimum = max(18, int(order) * 6)
    for start, end in runs:
        if end - start < minimum:
            continue
        try:
            output[start:end] = sosfiltfilt(sos, values[start:end])
        except ValueError:
            continue
    return pd.Series(output, index=series.index)


# Backward-compatible alias used by early V4 tests.
angle = internal_angle
