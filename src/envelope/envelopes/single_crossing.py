"""
Single-Crossing Envelope Module

Fits a star-shaped envelope using polar coordinates.
Radial trajectories from the origin cross the boundary exactly once.

The algorithm:
1. Use MCD to filter outliers and select coverage fraction of points
2. Convert filtered points to polar coordinates from origin
3. For each angular bin, take the maximum radius to ensure containment
4. For empty angular bins, decay toward origin (not interpolate)

Extensible to arbitrary trajectories via the trajectory_func parameter.
"""

import numpy as np
from sklearn.covariance import MinCovDet
from typing import Optional, Callable, Tuple

from ..core.geometry import ensure_ccw, polygon_area, contains


# Type alias for trajectory functions
# trajectory_func(angle, t) -> (x, y) where t in [0, 1], t=0 at infinity, t=1 at origin
TrajectoryFunc = Callable[[float, float], Tuple[float, float]]


def fit_single_crossing_envelope(
    points: np.ndarray,
    coverage: float = 0.95,
    n_angles: int = 72,
    origin: Optional[np.ndarray] = None,
    trajectory_func: Optional[TrajectoryFunc] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Fit a single-crossing (star-shaped) envelope around points.

    For radial trajectories from origin, the envelope boundary is crossed
    exactly once. This relaxes concavity while maintaining arbitrage-free
    properties along specified trajectories.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.
    coverage : float
        Target fraction of points for MCD outlier filtering (0, 1]. Default 0.95.
    n_angles : int
        Number of angular bins for boundary estimation. Default 72 (5 deg bins).
    origin : np.ndarray, optional
        Origin point for polar coordinates. If None, uses [0, 0].
    trajectory_func : TrajectoryFunc, optional
        Custom trajectory function for validation. If provided, validates
        that the envelope satisfies single-crossing for these trajectories.

    Returns
    -------
    poly : np.ndarray
        Vertices of the envelope polygon in counter-clockwise order. Shape (M, 2).
    validation : dict or None
        If trajectory_func provided, contains validation results.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points of shape (N, 2), got {points.shape}")

    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    n_points = len(points)
    if n_points < 3:
        raise ValueError(f"Need at least 3 points, got {n_points}")

    # Set origin
    if origin is None:
        origin = np.array([0.0, 0.0])
    else:
        origin = np.asarray(origin, dtype=np.float64)

    # Apply MCD filtering to select inlier points
    n_keep = max(3, int(np.ceil(coverage * n_points)))
    if coverage >= 1.0 or n_keep >= n_points:
        filtered_points = points.copy()
    else:
        try:
            mcd = MinCovDet(support_fraction=min(0.9, max(0.5, coverage)))
            mcd.fit(points)
            mahal_dist = mcd.mahalanobis(points)
            indices = np.argsort(mahal_dist)[:n_keep]
            filtered_points = points[indices]
        except Exception:
            # Fallback: use distance from centroid
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            indices = np.argsort(distances)[:n_keep]
            filtered_points = points[indices]

    # Translate points relative to origin
    centered = filtered_points - origin

    r = np.linalg.norm(centered, axis=1)
    theta = np.arctan2(centered[:, 1], centered[:, 0])


    valid = r > 1e-10
    r = r[valid]
    theta = theta[valid]

    if len(r) < 3:
        raise ValueError("Not enough non-origin points")

    angle_bins = np.linspace(-np.pi, np.pi, n_angles + 1)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    boundary_radii = np.zeros(n_angles)

    for i in range(n_angles):
        in_bin = (theta >= angle_bins[i]) & (theta < angle_bins[i + 1])
        if np.sum(in_bin) > 0:
            boundary_radii[i] = np.max(r[in_bin])
        else:
            boundary_radii[i] = np.nan

    boundary_radii = _fill_gaps_with_decay(boundary_radii)
    x = boundary_radii * np.cos(angle_centers) + origin[0]
    y = boundary_radii * np.sin(angle_centers) + origin[1]
    poly = np.column_stack([x, y])
    poly = ensure_ccw(poly)
    validation = None
    if trajectory_func is not None:
        validation = _validate_trajectories(poly, trajectory_func, n_angles, origin)

    return poly, validation


def _fill_gaps_with_decay(radii: np.ndarray, decay_scale: int = 3) -> np.ndarray:
    """
    Fill NaN values using exponential decay toward origin for large gaps.

    For small gaps (few empty bins), interpolate between neighbors.
    For large gaps, decay the radius toward a minimum value as distance
    from data increases. This prevents the envelope from bulging into
    empty angular regions.
    """
    n = len(radii)
    result = radii.copy()
    nan_mask = np.isnan(result)

    if nan_mask.all():
        raise ValueError("No valid radii in any angular bin")

    if not nan_mask.any():
        return result

    valid_idx = np.where(~nan_mask)[0]

    # Use minimum valid radius as the floor to decay toward
    min_valid = np.nanmin(result)
    floor_radius = min_valid * 0.05

    # Handle case with only one valid value
    if len(valid_idx) == 1:
        # Decay from the single valid point
        single_idx = valid_idx[0]
        single_val = result[single_idx]
        for i in range(n):
            if i != single_idx:
                dist = min(abs(i - single_idx), n - abs(i - single_idx))
                decay = np.exp(-dist / decay_scale)
                result[i] = single_val * decay + floor_radius * (1 - decay)
        return result

    # For each NaN, compute distance to nearest valid bin and decay
    for i in np.where(nan_mask)[0]:
        # Find distance to nearest valid bin (circular)
        dists_forward = (valid_idx - i) % n
        dists_backward = (i - valid_idx) % n
        dists = np.minimum(dists_forward, dists_backward)

        min_dist = np.min(dists)
        nearest_idx = valid_idx[np.argmin(dists)]
        nearest_val = result[nearest_idx]

        # Exponential decay: close to data = use data radius, far from data = use floor
        decay = np.exp(-min_dist / decay_scale)
        result[i] = nearest_val * decay + floor_radius * (1 - decay)

    return result


def _validate_trajectories(
    poly: np.ndarray,
    trajectory_func: TrajectoryFunc,
    n_test_angles: int,
    origin: np.ndarray
) -> dict:
    """Validate that trajectories cross the boundary exactly once."""
    from shapely.geometry import Polygon, LineString

    shapely_poly = Polygon(poly)
    test_angles = np.linspace(-np.pi, np.pi, n_test_angles, endpoint=False)

    crossings = []
    for angle in test_angles:
        # Sample trajectory from far out to origin
        t_vals = np.linspace(0, 1, 100)
        traj_points = [trajectory_func(angle, t) for t in t_vals]
        traj_line = LineString(traj_points)

        intersection = shapely_poly.exterior.intersection(traj_line)
        n_crossings = len(intersection.geoms) if hasattr(intersection, 'geoms') else (
            1 if not intersection.is_empty else 0
        )
        crossings.append(n_crossings)

    crossings = np.array(crossings)
    return {
        'all_single_crossing': bool(np.all(crossings == 1)),
        'crossing_counts': crossings,
        'min_crossings': int(np.min(crossings)),
        'max_crossings': int(np.max(crossings)),
    }


def single_crossing_stats(poly: np.ndarray, points: np.ndarray) -> dict:
    """
    Compute diagnostic statistics for a single-crossing envelope.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).
    points : np.ndarray
        Data points of shape (N, 2).

    Returns
    -------
    dict
        Statistics including fraction_contained, origin_inside, num_vertices,
        area, and star-shaped verification.
    """
    points = np.atleast_2d(points)
    inside_mask = contains(poly, points)
    fraction = np.mean(inside_mask)
    origin_inside = bool(contains(poly, np.array([[0.0, 0.0]]))[0])
    area = polygon_area(poly)

    return {
        'fraction_contained': float(fraction),
        'origin_inside': origin_inside,
        'num_vertices': len(poly),
        'area': float(area),
        'points_inside': int(np.sum(inside_mask)),
        'points_outside': int(np.sum(~inside_mask)),
        'is_star_shaped': _verify_star_shaped(poly),
    }


def _verify_star_shaped(poly: np.ndarray) -> bool:
    """Check if polygon is star-shaped from origin (all radii positive)."""
    r = np.linalg.norm(poly, axis=1)
    return bool(np.all(r > 0))
