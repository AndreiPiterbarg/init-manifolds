"""
Single-Crossing Envelope Module

Fits a star-shaped (single-crossing) envelope using polar coordinates.
Radial trajectories from the origin cross the boundary exactly once,
making this ideal for dumbbell-shaped or other star-shaped data.

Features:
- Robust outlier filtering using Minimum Covariance Determinant
- Polar coordinate fitting with angular binning
- Circular interpolation for sparse angular coverage
- Gaussian smoothing with wrap-around for smooth boundaries
- Optional validation for custom trajectory functions
"""

from typing import Callable, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.covariance import MinCovDet
from shapely.geometry import LineString, Polygon

from ..core.geometry import ensure_ccw, contains


# Type alias for trajectory functions: (angle, t) -> (x, y)
TrajectoryFunc = Callable[[float, float], np.ndarray]


def _filter_outliers(
    points: np.ndarray,
    coverage: float
) -> np.ndarray:
    """
    Filter outliers using Minimum Covariance Determinant.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.
    coverage : float
        Target fraction of points to keep.

    Returns
    -------
    np.ndarray
        Filtered points of shape (M, 2) where M <= N.
    """
    n_points = len(points)
    n_keep = max(3, int(np.ceil(coverage * n_points)))

    if coverage >= 1.0 or n_keep >= n_points:
        return points.copy()

    try:
        mcd = MinCovDet(support_fraction=min(0.9, max(0.5, coverage)))
        mcd.fit(points)
        mahal_dist = mcd.mahalanobis(points)
        indices = np.argsort(mahal_dist)[:n_keep]
        return points[indices]
    except Exception:
        # Fallback: use distance from centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        indices = np.argsort(distances)[:n_keep]
        return points[indices]


def _interpolate_empty_bins(
    radii: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_to_interpolate: int = 3
) -> np.ndarray:
    """
    Interpolate small empty angular gaps; collapse large gaps toward origin.

    For star-shaped envelopes, we only interpolate across small gaps (a few
    missing bins). Large gaps indicate regions with no data support, so we
    collapse the radius to near-zero to avoid covering empty space.

    Parameters
    ----------
    radii : np.ndarray
        Array of radii per angle bin.
    valid_mask : np.ndarray
        Boolean mask indicating which bins have valid data.
    max_gap_to_interpolate : int
        Maximum consecutive empty bins to interpolate. Larger gaps collapse
        to near-zero radius. Default 3 (~15 degrees for 72 bins).

    Returns
    -------
    np.ndarray
        Radii with small gaps interpolated and large gaps collapsed.
    """
    if np.all(valid_mask):
        return radii.copy()

    if not np.any(valid_mask):
        return np.zeros_like(radii)

    n = len(radii)
    result = radii.copy()

    # Find contiguous runs of invalid bins (gaps)
    # We need to handle circular wrapping
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.zeros_like(radii)

    # For each invalid bin, determine gap size and whether to interpolate
    for idx in np.where(~valid_mask)[0]:
        # Find nearest valid bin before and after (circular)
        # Distance to each valid index (circular)
        dists_forward = (valid_indices - idx) % n
        dists_backward = (idx - valid_indices) % n

        # Nearest valid bin in forward direction
        forward_idx = valid_indices[np.argmin(dists_forward)]
        forward_dist = np.min(dists_forward)

        # Nearest valid bin in backward direction
        backward_idx = valid_indices[np.argmin(dists_backward)]
        backward_dist = np.min(dists_backward)

        # Total gap size is the distance between the two nearest valid bins
        gap_size = forward_dist + backward_dist

        if gap_size <= max_gap_to_interpolate:
            # Small gap: interpolate between neighbors
            if forward_dist + backward_dist == 0:
                result[idx] = radii[forward_idx]
            else:
                # Linear interpolation weighted by distance
                w_backward = forward_dist / (forward_dist + backward_dist)
                w_forward = backward_dist / (forward_dist + backward_dist)
                result[idx] = w_backward * radii[backward_idx] + w_forward * radii[forward_idx]
        else:
            # Large gap: collapse toward origin (very small radius)
            # Use a small fraction of the nearest valid radius to create a "pinch"
            min_valid_radius = np.min(radii[valid_mask])
            result[idx] = min_valid_radius * 0.01

    return result


def _circular_smooth(
    radii: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Apply Gaussian smoothing with circular (wrap) boundary conditions.

    Parameters
    ----------
    radii : np.ndarray
        Array of radii per angle bin.
    sigma : float
        Standard deviation for Gaussian kernel (in bin units).

    Returns
    -------
    np.ndarray
        Smoothed radii.
    """
    if sigma <= 0:
        return radii.copy()

    return gaussian_filter1d(radii, sigma=sigma, mode='wrap')


def _adaptive_expand_for_coverage(
    bin_radii: np.ndarray,
    points: np.ndarray,
    origin: np.ndarray,
    n_angles: int,
    target_coverage: float,
    max_iterations: int = 50,
    tolerance: float = 0.01
) -> np.ndarray:
    """
    Adaptively expand radii per-direction to achieve target coverage.

    Key insight: only expand in directions where data actually exists.
    Empty angular regions should not be expanded, preventing "blobs"
    in areas with no data.

    Parameters
    ----------
    bin_radii : np.ndarray
        Current radii per angular bin.
    points : np.ndarray
        Original points to test coverage against.
    origin : np.ndarray
        Origin point for polar coordinates.
    n_angles : int
        Number of angular bins.
    target_coverage : float
        Target fraction of points to contain.
    max_iterations : int
        Maximum iterations for expansion.
    tolerance : float
        Acceptable deviation from target coverage.

    Returns
    -------
    np.ndarray
        Expanded radii that achieve target coverage.
    """
    bin_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    bin_width = 2 * np.pi / n_angles
    result_radii = bin_radii.copy()

    # Convert points to polar coordinates
    centered = points - origin
    point_radii = np.linalg.norm(centered, axis=1)
    point_angles = np.arctan2(centered[:, 1], centered[:, 0])
    point_angles = np.mod(point_angles, 2 * np.pi)
    point_bins = np.floor(point_angles / bin_width).astype(int)
    point_bins = np.clip(point_bins, 0, n_angles - 1)

    # Identify which bins have actual data (not just interpolated)
    # A bin "has data" if there are points within a few bins of it
    data_density = np.zeros(n_angles)
    neighbor_range = 3  # Look at neighboring bins too
    for i in range(n_angles):
        for offset in range(-neighbor_range, neighbor_range + 1):
            neighbor_idx = (i + offset) % n_angles
            data_density[i] += np.sum(point_bins == neighbor_idx)
    has_nearby_data = data_density > 0

    # Compute the max radius per bin from actual data
    max_radius_per_bin = np.zeros(n_angles)
    for i in range(n_angles):
        bin_mask = point_bins == i
        if np.any(bin_mask):
            max_radius_per_bin[i] = np.max(point_radii[bin_mask])

    def make_envelope(radii):
        """Create envelope from radii."""
        envelope_x = origin[0] + radii * np.cos(bin_angles)
        envelope_y = origin[1] + radii * np.sin(bin_angles)
        return np.column_stack([envelope_x, envelope_y])

    def compute_coverage(radii):
        """Compute coverage for given radii."""
        envelope = make_envelope(radii)
        inside = contains(envelope, points)
        return np.mean(inside)

    # Check initial coverage
    current_cov = compute_coverage(result_radii)
    if current_cov >= target_coverage - tolerance:
        return result_radii

    # Iteratively expand only in directions with data
    for iteration in range(max_iterations):
        envelope = make_envelope(result_radii)
        inside = contains(envelope, points)
        current_cov = np.mean(inside)

        if current_cov >= target_coverage - tolerance:
            break

        # Find points that are outside
        outside_mask = ~inside
        if not np.any(outside_mask):
            break

        # For each outside point, expand its angular bin
        outside_radii = point_radii[outside_mask]
        outside_bins = point_bins[outside_mask]

        # Only expand bins that have nearby data
        for bin_idx in range(n_angles):
            if not has_nearby_data[bin_idx]:
                continue  # Skip bins with no nearby data

            bin_mask = outside_bins == bin_idx
            if np.any(bin_mask):
                max_needed = np.max(outside_radii[bin_mask])
                if max_needed > result_radii[bin_idx]:
                    result_radii[bin_idx] = max_needed * 1.02

    # Smooth only the data regions, keeping empty regions collapsed
    # Apply smoothing but then restore collapsed regions
    smoothed = gaussian_filter1d(result_radii, sigma=1.0, mode='wrap')

    # Blend: use smoothed values where there's data, keep original where empty
    # This prevents smoothing from spreading into empty regions
    for i in range(n_angles):
        if has_nearby_data[i]:
            result_radii[i] = smoothed[i]
        # else: keep the original (possibly collapsed) value

    return result_radii


def _validate_crossings(
    envelope: np.ndarray,
    origin: np.ndarray,
    trajectory_func: TrajectoryFunc,
    n_samples: int = 360
) -> dict:
    """
    Validate that trajectories cross the envelope exactly once.

    Parameters
    ----------
    envelope : np.ndarray
        Envelope vertices of shape (M, 2).
    origin : np.ndarray
        Origin point of shape (2,).
    trajectory_func : TrajectoryFunc
        Function (angle, t) -> (x, y) defining the trajectory.
    n_samples : int
        Number of angles to sample for validation.

    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid': bool, True if all trajectories cross exactly once
        - 'failures': list of angles where crossing count != 1
        - 'crossing_counts': dict mapping angle to crossing count
    """
    envelope_polygon = Polygon(envelope)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    failures = []
    crossing_counts = {}

    for angle in angles:
        # Sample trajectory from t=0 (origin) to t=2 (well beyond envelope)
        t_values = np.linspace(0, 2, 100)
        trajectory_points = np.array([trajectory_func(angle, t) for t in t_values])
        trajectory_line = LineString(trajectory_points)

        # Count intersections with envelope boundary
        intersection = trajectory_line.intersection(envelope_polygon.exterior)

        if intersection.is_empty:
            n_crossings = 0
        elif intersection.geom_type == 'Point':
            n_crossings = 1
        elif intersection.geom_type == 'MultiPoint':
            n_crossings = len(intersection.geoms)
        else:
            # LineString or other - count as multiple
            n_crossings = 2

        crossing_counts[float(angle)] = n_crossings

        if n_crossings != 1:
            failures.append(float(angle))

    return {
        'is_valid': len(failures) == 0,
        'failures': failures,
        'crossing_counts': crossing_counts
    }


def fit_single_crossing_envelope(
    points: np.ndarray,
    coverage: float = 0.95,
    n_angles: int = 72,
    smoothing: float = 0.5,
    percentile_per_angle: float = 95.0,
    origin: np.ndarray = None,
    trajectory_func: TrajectoryFunc = None,
    n_validation_samples: int = 360,
    max_gap_to_interpolate: int = None,
    radius_expansion: float = 1.0,
    target_coverage: float = None
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Fit a single-crossing (star-shaped) envelope using polar coordinates.

    The envelope is constructed such that radial trajectories from the origin
    cross the boundary exactly once. This is achieved by fitting in polar
    coordinates where r(theta) is single-valued.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.
    coverage : float
        Target fraction of points for outlier filtering (0, 1]. Default 0.95.
    n_angles : int
        Number of angular bins. Default 72 (5-degree bins).
    smoothing : float
        Gaussian smoothing sigma in bin units. Default 0.5.
        Set to 0 for no smoothing.
    percentile_per_angle : float
        Percentile of radius to use per angular bin. Default 95.0.
    origin : np.ndarray, optional
        Origin point for polar coordinates. Defaults to data centroid.
    trajectory_func : TrajectoryFunc, optional
        Custom trajectory function (angle, t) -> (x, y).
        If provided, validation is performed after fitting.
    n_validation_samples : int
        Number of angles for trajectory validation. Default 360.
    max_gap_to_interpolate : int, optional
        Maximum consecutive empty bins to interpolate. Larger gaps collapse
        toward origin. Default is n_angles (always interpolate).
    radius_expansion : float
        Factor to expand radii after fitting. Use values > 1.0 to ensure
        all points are contained (accounts for binning discretization).
        Default 1.0 (no expansion).
    target_coverage : float, optional
        If set, adaptively expand the envelope until this fraction of the
        ORIGINAL points (before MCD filtering) are contained. This ensures
        smooth shapes while achieving desired coverage. Default None.

    Returns
    -------
    envelope : np.ndarray
        Envelope vertices in CCW order, shape (M, 2).
    validation : dict or None
        None for radial trajectories (default).
        For custom trajectories: {'is_valid': bool, 'failures': list, 'crossing_counts': dict}

    Raises
    ------
    ValueError
        If points has wrong shape or coverage is out of range.

    Examples
    --------
    >>> points = np.random.randn(100, 2)
    >>> envelope, _ = fit_single_crossing_envelope(points)
    >>> envelope.shape
    (72, 2)

    >>> # With custom trajectory validation
    >>> def spiral(angle, t):
    ...     r = t * (1 + 0.1 * np.sin(3 * angle))
    ...     return np.array([r * np.cos(angle), r * np.sin(angle)])
    >>> envelope, validation = fit_single_crossing_envelope(points, trajectory_func=spiral)
    >>> validation['is_valid']
    True
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points of shape (N, 2), got {points.shape}")

    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    n_points = len(points)

    if n_points < 3:
        raise ValueError(f"Need at least 3 points, got {n_points}")

    # Set default for max_gap_to_interpolate (always interpolate by default)
    if max_gap_to_interpolate is None:
        max_gap_to_interpolate = n_angles  # Always interpolate, never collapse

    # Step 1: Filter outliers
    filtered_points = _filter_outliers(points, coverage)

    # Step 2: Set origin to centroid if not provided
    if origin is None:
        origin = np.mean(filtered_points, axis=0)
    else:
        origin = np.asarray(origin, dtype=np.float64)

    # Step 3: Convert to polar coordinates relative to origin
    centered = filtered_points - origin
    radii = np.linalg.norm(centered, axis=1)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    # Normalize angles to [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)

    # Step 4: Bin into angular bins
    bin_width = 2 * np.pi / n_angles
    bin_indices = np.floor(angles / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_angles - 1)

    # Step 5: Compute percentile radius per bin
    bin_radii = np.zeros(n_angles)
    valid_mask = np.zeros(n_angles, dtype=bool)

    for i in range(n_angles):
        mask = bin_indices == i
        if np.any(mask):
            bin_radii[i] = np.percentile(radii[mask], percentile_per_angle)
            valid_mask[i] = True

    # Step 6: Interpolate empty bins
    bin_radii = _interpolate_empty_bins(bin_radii, valid_mask, max_gap_to_interpolate)

    # Step 7: Apply circular Gaussian smoothing
    smoothing_sigma = smoothing * (n_angles / 72)  # Scale sigma with bin count
    bin_radii = _circular_smooth(bin_radii, smoothing_sigma)

    # Step 7b: Apply radius expansion if requested
    if radius_expansion != 1.0:
        bin_radii = bin_radii * radius_expansion

    # Step 7c: Adaptive expansion to achieve target coverage
    if target_coverage is not None:
        bin_radii = _adaptive_expand_for_coverage(
            bin_radii, points, origin, n_angles, target_coverage
        )

    # Step 8: Convert back to Cartesian
    bin_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    envelope_x = origin[0] + bin_radii * np.cos(bin_angles)
    envelope_y = origin[1] + bin_radii * np.sin(bin_angles)
    envelope = np.column_stack([envelope_x, envelope_y])

    # Step 9: Validate if custom trajectory provided
    validation = None
    if trajectory_func is not None:
        validation = _validate_crossings(
            envelope, origin, trajectory_func, n_validation_samples
        )

    # Step 10: Ensure CCW ordering and return
    envelope = ensure_ccw(envelope)

    return envelope, validation


def single_crossing_stats(
    envelope: np.ndarray,
    points: np.ndarray,
    origin: np.ndarray = None
) -> dict:
    """
    Compute diagnostic statistics for a single-crossing envelope.

    Parameters
    ----------
    envelope : np.ndarray
        Envelope vertices of shape (M, 2).
    points : np.ndarray
        Data points of shape (N, 2).
    origin : np.ndarray, optional
        Origin used for fitting. Defaults to envelope centroid.

    Returns
    -------
    dict
        Statistics including:
        - 'fraction_contained': Fraction of points inside envelope
        - 'origin_inside': Whether origin is inside
        - 'num_vertices': Number of envelope vertices
        - 'mean_radius': Mean radius from origin
        - 'radius_std': Standard deviation of radius
        - 'points_inside': Count of points inside
        - 'points_outside': Count of points outside
    """
    points = np.atleast_2d(points)

    if origin is None:
        origin = np.mean(envelope, axis=0)

    inside_mask = contains(envelope, points)
    fraction = np.mean(inside_mask)
    origin_inside = bool(contains(envelope, origin.reshape(1, -1))[0])

    # Compute radii from origin to envelope vertices
    envelope_centered = envelope - origin
    radii = np.linalg.norm(envelope_centered, axis=1)

    return {
        'fraction_contained': float(fraction),
        'origin_inside': origin_inside,
        'num_vertices': len(envelope),
        'mean_radius': float(np.mean(radii)),
        'radius_std': float(np.std(radii)),
        'points_inside': int(np.sum(inside_mask)),
        'points_outside': int(np.sum(~inside_mask)),
    }
