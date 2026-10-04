# VMotionLab V4.2.1 Clinical Angles

Experimental video-based lower-limb kinematics software for research,
education, and method development.

## Clinical angle convention

| Joint | Neutral | Positive | Negative |
|---|---:|---|---|
| Hip | approximately 0° | Flexion | Extension |
| Knee | approximately 0° | Flexion | Hyperextension |
| Ankle | approximately 0° | Dorsiflexion | Plantarflexion |

The ankle output is converted from the raw shank-foot geometric angle to a
zero-referenced clinical convention.

## Important interpretation note

The V4.2 hip result is a **2D trunk-referenced hip flexion/extension proxy**,
not a full 3D anatomical hip joint-coordinate angle.

Clinical sagittal hip, knee, and ankle interpretation requires a true or
near-true sagittal camera view.

## Repository structure

```text
src/vmotionlab/modules/   analysis modules
tests/                    clinical-angle verification
docs/                     method documentation
```

## Verify the angle convention

```powershell
python .\tests\verify_clinical_angles_portable.py
```

Expected signs:

```text
Hip:   flexion +, extension -
Knee:  flexion +, hyperextension -
Ankle: dorsiflexion +, plantarflexion -
```

## What should not be committed

Do not commit:
- participant videos or identifiable participant information;
- participant landmark or motion CSV files;
- the packaged `_internal` Python runtime;
- Windows EXE/MSI builds;
- large RTMPose/YOLOX model files.

Desktop installers should be attached to a **GitHub Release** rather than kept
inside the source repository.

## Scientific status

VMotionLab is experimental research/education software. Numerical outputs
should not be treated as validated clinical ground truth without comparison
against an appropriate synchronized reference system.

## License

No open-source license has been selected yet. Choose a license before treating
the repository as open-source.
