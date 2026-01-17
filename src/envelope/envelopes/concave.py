"""
Concave Envelope Module (Alpha Shapes)

Fits a concave envelope that:
- Encloses the origin (0,0)
- Contains approximately a target fraction of data points
- Is resistant to outliers using Minimum Covariance Determinant
- Respects concavities in data (horseshoe, crescent shapes) via alpha shapes

Built on alphashape + shapely + sklearn MinCovDet
"""

from typing import Optional
import warnings

# Suppress numpy matrix deprecation warnings from alphashape's internals
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        message=".*matrix subclass.*")

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import MinCovDet

# Alpha shape and geometry libraries
import alphashape
from shapely.geometry import Polygon, Point, MultiPolygon

from ..core.geometry import (
    contains,
    polygon_area,
    ensure_ccw,
    shapely_to_numpy,
    expand_polygon_to_include_origin,
    _polygon_area,  # For backward compat in tests
)


def fit_envelope(
    points: np.ndarray,
    coverage: float = 0.95,
    include_origin: bool = True,
    alpha: Optional[float] = None,
    use_convex_fallback: bool = True
) -> np.ndarray:
    """
    Fit a concave envelope around points using alpha shapes.

    Uses Minimum Covariance Determinant (MCD) to robustly identify inliers,
    then computes the alpha shape (concave hull) of those inliers.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.
    coverage : float
        Target fraction of points to include (0, 1]. Default 0.95.
    include_origin : bool
        If True, ensure the origin (0,0) is inside the envelope.
        The envelope will be expanded minimally if needed. Default True.
    alpha : float, optional
        Alpha parameter controlling concavity. Larger alpha = more concave.
        If None, alphashape will optimize alpha automatically.
        Use alpha=0 for convex hull behavior.
    use_convex_fallback : bool
        If True, fall back to convex hull when alpha shape fails or
        produces invalid geometry. Default True.

    Returns
    -------
    np.ndarray
        Vertices of the envelope polygon in counter-clockwise order.
        Shape (M, 2).
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points of shape (N, 2), got {points.shape}")

    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    n_points = len(points)

    if n_points < 3:
        raise ValueError(f"Need at least 3 points, got {n_points}")

    n_keep = max(3, int(np.ceil(coverage * n_points)))

    if coverage >= 1.0 or n_keep >= n_points:
        candidate_points = points.copy()
    else:
        # Use MCD to compute robust covariance and Mahalanobis distances
        try:
            mcd = MinCovDet(support_fraction=min(0.9, max(0.5, coverage)))
            mcd.fit(points)
            mahal_dist = mcd.mahalanobis(points)
            indices = np.argsort(mahal_dist)[:n_keep]
            candidate_points = points[indices]
        except Exception:
            # Fallback: use distance from centroid
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            indices = np.argsort(distances)[:n_keep]
            candidate_points = points[indices]

    # Compute alpha shape (concave hull)
    try:
        if alpha is None:
            # Let alphashape optimize alpha automatically
            alpha_shape = alphashape.alphashape(candidate_points)
        elif alpha == 0:
            # Alpha=0 means convex hull
            alpha_shape = alphashape.alphashape(candidate_points, 0)
        else:
            alpha_shape = alphashape.alphashape(candidate_points, alpha)

        # Handle potential MultiPolygon result
        if isinstance(alpha_shape, MultiPolygon):
            alpha_shape = max(alpha_shape.geoms, key=lambda g: g.area)

        if not alpha_shape.is_valid or alpha_shape.is_empty:
            raise ValueError("Invalid alpha shape")

    except Exception as e:
        if use_convex_fallback:
            # Fall back to convex hull
            if include_origin:
                candidate_points = np.vstack([candidate_points, [[0.0, 0.0]]])
            hull = ConvexHull(candidate_points)
            poly = candidate_points[hull.vertices]
            return ensure_ccw(poly)
        else:
            raise ValueError(f"Alpha shape computation failed: {e}")

    # Ensure origin is contained if requested
    if include_origin:
        origin = Point(0.0, 0.0)
        if not (alpha_shape.contains(origin) or alpha_shape.touches(origin)):
            alpha_shape = expand_polygon_to_include_origin(alpha_shape, origin)

    # Convert back to numpy array
    poly = shapely_to_numpy(alpha_shape)

    return ensure_ccw(poly)


def envelope_stats(poly: np.ndarray, points: np.ndarray) -> dict:
    """
    Compute diagnostic statistics for an envelope.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).
    points : np.ndarray
        Data points of shape (N, 2).

    Returns
    -------
    dict
        Statistics including:
        - fraction_contained: Fraction of points inside envelope
        - origin_inside: Whether origin is inside
        - num_vertices: Number of polygon vertices
        - area: Polygon area
        - centroid: Polygon centroid
        - points_inside: Count of points inside
        - points_outside: Count of points outside
    """
    points = np.atleast_2d(points)
    inside_mask = contains(poly, points)
    fraction = np.mean(inside_mask)
    origin_inside = bool(contains(poly, np.array([[0.0, 0.0]]))[0])
    area = polygon_area(poly)
    centroid = np.mean(poly, axis=0)

    return {
        'fraction_contained': float(fraction),
        'origin_inside': origin_inside,
        'num_vertices': len(poly),
        'area': float(area),
        'centroid': centroid,
        'points_inside': int(np.sum(inside_mask)),
        'points_outside': int(np.sum(~inside_mask)),
    }
