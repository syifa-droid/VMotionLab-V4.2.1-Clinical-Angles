# VMotionLab V4.2.1 — Clinical Angles

## Highlights

VMotionLab V4.2.1 standardizes the sagittal lower-limb kinematic outputs to
clinically interpretable angle conventions.

### Clinical angle conventions

| Joint | Neutral | Positive | Negative |
|---|---:|---|---|
| Hip | approximately 0° | Flexion | Extension |
| Knee | approximately 0° | Flexion | Hyperextension |
| Ankle | approximately 0° | Dorsiflexion | Plantarflexion |

### What changed

- Hip angle is now signed: flexion positive, extension negative.
- Knee angle is now signed: flexion positive, hyperextension negative.
- Ankle is reported as a clinical zero-referenced angle rather than the raw
  shank-foot geometric angle near 90°.
- Left/right-facing sagittal recordings are handled using a consistent
  subject-forward direction.
- Existing V4.2 output column names are preserved for compatibility.
- Plot labels and metadata explicitly document the clinical sign convention.
- Automated clinical-angle checks are included in GitHub Actions.

## Recording requirement

Clinical interpretation of hip, knee, and ankle flexion-extension requires a
true or near-true sagittal camera view. Frontal or strongly oblique recordings
should not be interpreted as sagittal clinical joint angles.

## Important limitation

The V4.2 hip angle is a 2D trunk-referenced hip flexion/extension proxy. It is
not a full 3D anatomical hip joint-coordinate angle because sagittal pelvic
tilt is not independently measured in this workflow.

## Scientific status

VMotionLab is experimental research and educational software. Its outputs
should not be treated as validated clinical ground truth without comparison
against an appropriate synchronized reference system.

## License

VMotionLab V4.2.1 is distributed under the VMotionLab Research and Educational
Use License. Commercial use and clinical/medical-device deployment require
separate written permission.

## Windows Desktop package

The Windows Lite Desktop package should be attached to this GitHub Release as:

`VMotionLabV4_2_Lite_Distribution.zip`

The desktop package should not be committed directly to the source repository.
