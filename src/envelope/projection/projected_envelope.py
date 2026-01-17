"""
Projected Envelope Module

Projects 3D point data onto a 2D plane (via PCA) and computes
an envelope that:
- Contains the origin (zero point)
- Contains approximately 95% of the data (outlier-robust)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pca import ProjectionResult, project_to_2d, lift_to_3d
from ..core.geometry import contains
from ..envelopes.concave import fit_envelope, envelope_stats


@dataclass
class EnvelopeResult:
    """
    Container for the full envelope computation results.

    Attributes
    ----------
    envelope_2d : np.ndarray
        Convex envelope vertices in 2D (CCW order) of shape (M, 2).
    envelope_3d : np.ndarray
        Envelope vertices lifted back to 3D (on the projection plane) of shape (M, 3).
    projection : ProjectionResult
        Full projection information.
    stats : dict
        Envelope statistics (coverage, area, etc.).
    """
    envelope_2d: np.ndarray
    envelope_3d: np.ndarray
    projection: ProjectionResult
    stats: dict


def fit_projected_envelope(
    points_3d: np.ndarray,
    coverage: float = 0.95,
    include_origin: bool = True,
    center: Optional[np.ndarray] = None
) -> EnvelopeResult:
    """
    Project 3D data to 2D and compute a robust envelope.

    This is the main entry point for 3D envelope fitting:
    1. Projects 3D points onto the principal plane (via PCA)
    2. Computes an envelope containing ~coverage fraction of points
    3. Ensures the origin/center is inside the envelope

    Parameters
    ----------
    points_3d : np.ndarray
        Array of shape (N, 3) containing 3D points.
    coverage : float
        Target fraction of points to include (0, 1]. Default 0.95.
    include_origin : bool
        If True, ensure the origin is inside the envelope.
    center : np.ndarray, optional
        Center point for projection. Defaults to origin [0, 0, 0].

    Returns
    -------
    EnvelopeResult
        Container with envelope vertices, projection info, and statistics.
    """
    projection = project_to_2d(points_3d, center=center)

    envelope_2d = fit_envelope(
        projection.points_2d,
        coverage=coverage,
        include_origin=include_origin
    )

    envelope_3d = lift_to_3d(envelope_2d, projection)
    stats = envelope_stats(envelope_2d, projection.points_2d)

    return EnvelopeResult(
        envelope_2d=envelope_2d,
        envelope_3d=envelope_3d,
        projection=projection,
        stats=stats
    )


def contains_projected(
    result: EnvelopeResult,
    points_3d: np.ndarray
) -> np.ndarray:
    """
    Test if 3D points are inside the projected envelope.

    Projects the query points onto the same 2D plane and tests containment.

    Parameters
    ----------
    result : EnvelopeResult
        Result from fit_projected_envelope().
    points_3d : np.ndarray
        Array of shape (N, 3) containing 3D points to test.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) indicating containment.
    """
    points_3d = np.atleast_2d(points_3d)

    # Project query points using the same basis
    centered = points_3d - result.projection.center_3d
    points_2d = centered @ result.projection.basis.T

    return contains(result.envelope_2d, points_2d)
