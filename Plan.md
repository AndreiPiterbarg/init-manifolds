# Single-Crossing Envelope Plan (Simplified)

## Goal
Fit a **star-shaped (single-crossing) envelope** to dumbbell-shaped data. Support radial trajectories by default, with optional custom trajectory functions for validation.

---

## Key Design Decision

**Radial trajectories (default):** Polar coordinates r(θ) guarantee single-crossing by construction - no validation needed.

**Custom trajectories (optional):** User can provide a function `f(angle, t) -> (x, y)`. Since polar fitting doesn't guarantee single-crossing for curved paths, we validate after fitting and report failures.

---

## Architecture

```
src/envelope/
    envelopes/
        single_crossing.py    # NEW: fit_single_crossing_envelope()
    projection/
        projected_envelope.py # ADD: fit_projected_single_crossing_envelope()
    __init__.py              # UPDATE: add exports
```

No new files for trajectories - just use `Callable[[float, float], np.ndarray]`.

---

## Implementation

### 1. `src/envelope/envelopes/single_crossing.py`

```python
from typing import Callable, Optional, Tuple
import numpy as np

TrajectoryFunc = Callable[[float, float], np.ndarray]  # (angle, t) -> (x, y)

def fit_single_crossing_envelope(
    points: np.ndarray,
    coverage: float = 0.95,
    n_angles: int = 72,
    smoothing: float = 0.5,
    percentile_per_angle: float = 95.0,
    origin: np.ndarray = None,
    trajectory_func: TrajectoryFunc = None,
    n_validation_samples: int = 360
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Fit single-crossing envelope using polar coordinates.

    Returns:
        envelope: (M, 2) CCW-ordered polygon vertices
        validation: None for radial, or {'is_valid': bool, 'failures': list} for custom
    """
```

**Algorithm:**
1. MCD-based outlier filtering (reuse pattern from `concave.py`)
2. Set origin to data centroid if not provided
3. Convert to polar coordinates (r, θ) relative to origin
4. Bin into `n_angles` angular bins
5. Compute percentile radius per bin
6. Interpolate empty bins (circular)
7. Apply circular Gaussian smoothing
8. Convert back to Cartesian
9. If `trajectory_func` provided: validate and return results
10. Return CCW-ordered polygon vertices

**Helper functions:**
- `_filter_outliers()` - MCD-based filtering
- `_interpolate_empty_bins()` - circular interpolation
- `_circular_smooth()` - Gaussian smoothing with wrap mode
- `_validate_crossings()` - count crossings for custom trajectories

### 2. `src/envelope/projection/projected_envelope.py` (ADD)

```python
def fit_projected_single_crossing_envelope(
    points_3d: np.ndarray,
    coverage: float = 0.95,
    n_angles: int = 72,
    smoothing: float = 0.5,
    percentile_per_angle: float = 95.0,
    center: np.ndarray = None,
    trajectory_func: TrajectoryFunc = None
) -> EnvelopeResult
```

Same pattern as `fit_projected_envelope()`.

### 3. `src/envelope/__init__.py` (UPDATE)

```python
from .envelopes.single_crossing import fit_single_crossing_envelope
from .projection.projected_envelope import fit_projected_single_crossing_envelope
```

---

## Files to Modify

| File | Action |
|------|--------|
| `src/envelope/envelopes/single_crossing.py` | Implement |
| `src/envelope/projection/projected_envelope.py` | Add function |
| `src/envelope/__init__.py` | Add exports |
| `src/envelope/core/trajectories.py` | DELETE |

---

## Tests (`tests/test_single_crossing.py`)

| Test | Description |
|------|-------------|
| `test_output_shape` | Returns (M, 2) array |
| `test_contains_origin` | Origin inside envelope |
| `test_coverage_parameter` | Higher coverage = tighter fit |
| `test_empty_bins` | Handles sparse angular coverage |
| `test_dumbbell_data` | Motivating use case |
| `test_custom_trajectory_validation` | Validation works for non-radial |

---

## Verification

1. Run existing tests: `pytest tests/`
2. Run new tests: `I `
3. Visual check: Generate dumbbell data, compare alpha-shape vs single-crossing
4. Custom trajectory: Test with spiral function, check validation output

---

## Dependencies

No new dependencies:
- `numpy` - polar conversion
- `scipy.ndimage.gaussian_filter1d` - smoothing (mode='wrap')
- `shapely` - crossing detection
- `sklearn.covariance.MinCovDet` - outlier filtering
