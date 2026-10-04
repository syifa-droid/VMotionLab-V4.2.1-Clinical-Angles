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

## Scientific status

VMotionLab is experimental research/education software. Numerical outputs
should not be treated as validated clinical ground truth without comparison
against an appropriate synchronized reference system.

It is not supplied as a validated medical device and should not be used for
autonomous clinical diagnosis, treatment decisions, or direct patient-care
decisions.

## License

VMotionLab V4.2.1 is **source-available for restricted research and educational
use** under the [VMotionLab Research and Educational Use License](LICENSE).

In summary:

- non-commercial research, education, teaching, and academic evaluation are
  permitted;
- source modification is permitted for those purposes;
- redistribution is permitted only under the same research/educational
  restrictions and with attribution;
- commercial use, resale, paid commercial services, and incorporation into
  commercial products require separate written permission;
- clinical deployment or medical-device use requires separate written
  permission, independent validation, and any required regulatory review.

This is **not an OSI-approved open-source license**. See `LICENSE` for the
complete terms.
