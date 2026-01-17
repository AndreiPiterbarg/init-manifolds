"""
2D Convex Envelope Module

Fits convex envelope that:
- Encloses the origin (0,0)
- Contains approximately a target fraction of data points
- Is resistant to outliers using Minimum Covariance Determinant

Built on scipy ConvexHull + sklearn MinvCovDet
"""

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt


# Numerical tolerance, for python calcs
EPS = 1e-10

def contains(poly: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Test if points are inside or on a convex polygon.

    Uses the cross-product / half-plane test for convex polygons.
    Checks if point is inside poly. by checking iff it's on the same side of all edges.

    """
    points = np.atleast_2d(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    n_vertices = len(poly)
    n_points = len(points)

    # valid poly requires 3 vertices
    if n_vertices < 3:
        return np.zeros(n_points, dtype=bool)

    # For each edge, compute cross product with point; All cross products should have same sign for inside points
    inside = np.ones(n_points, dtype=bool)

    for i in range(n_vertices):
        v1 = poly[i]
        v2 = poly[(i + 1) % n_vertices]

        edge = v2 - v1
        # Vector from vertex to points
        to_points = points - v1

        # Cross product (2D): edge.x * to_point.y - edge.y * to_point.x
        cross = edge[0] * to_points[:, 1] - edge[1] * to_points[:, 0]

        # For CCW polygon, inside points have non-negative cross products
        inside &= (cross >= -EPS)

    return inside

# helpers
def _polygon_area(poly: np.ndarray) -> float:
    """Compute the area of a polygon using the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0

    # Shoelace formula
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    # Compute signed area
    n = len(poly)
    x = poly[:, 0]
    y = poly[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    if signed_area < 0:
        # Clockwise, reverse to make CCW
        return poly[::-1].copy()
    return poly


def fit_envelope(
    points: np.ndarray,
    coverage: float = 0.95,
    include_origin: bool = True
) -> np.ndarray:
    """
    Fit a convex envelope around points.

    Uses Minimum Covariance Determinant (MCD) to robustly identify inliers,
    then computes the convex hull of those inliers.

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
        # use all points
        candidate_points = points.copy()
    else:
        # Use MCD to compute robust covariance and Mahalanobis distances
        try:
            # MCD needs at least n_features + 1 samples for support_fraction
            mcd = MinCovDet(support_fraction=min(0.9, max(0.5, coverage)))
            mcd.fit(points)

            # Mahalanobis distances from robust center
            mahal_dist = mcd.mahalanobis(points)

            # Select the n_keep points with smallest Mahalanobis distance
            indices = np.argsort(mahal_dist)[:n_keep]
            candidate_points = points[indices]

        except Exception:
            # Fallback: use distance from centroid
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            indices = np.argsort(distances)[:n_keep]
            candidate_points = points[indices]


    if include_origin:
        candidate_points = np.vstack([candidate_points, [[0.0, 0.0]]])

    hull = ConvexHull(candidate_points)
    poly = candidate_points[hull.vertices]

    return _ensure_ccw(poly)


def envelope_stats(poly: np.ndarray, points: np.ndarray) -> dict:
    """
    Compute diagnostic statistics for an envelope.
    """
    points = np.atleast_2d(points)
    inside_mask = contains(poly, points)
    fraction = np.mean(inside_mask)
    origin_inside = bool(contains(poly, np.array([[0.0, 0.0]]))[0])
    area = _polygon_area(poly)
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


def plot_envelope(
    poly: np.ndarray,
    points: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show_origin: bool = True,
    title: str = "Convex Envelope",
    show_stats: bool = True
) -> plt.Axes:
    """
    Visualize the envelope and points.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    points = np.atleast_2d(points)
    inside_mask = contains(poly, points)

    ax.scatter(
        points[inside_mask, 0], points[inside_mask, 1],
        c='steelblue', alpha=0.6, s=20, label='Inside', zorder=2
    )
    ax.scatter(
        points[~inside_mask, 0], points[~inside_mask, 1],
        c='coral', alpha=0.6, s=20, label='Outside', zorder=2
    )

    closed_poly = np.vstack([poly, poly[0]])
    ax.plot(closed_poly[:, 0], closed_poly[:, 1], 'k-', linewidth=2, zorder=3)
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color='green', zorder=1)

    ax.scatter(poly[:, 0], poly[:, 1], c='black', s=50, marker='s', zorder=4)

    if show_origin:
        ax.scatter([0], [0], c='red', s=150, marker='*', zorder=5, label='Origin')

    if show_stats:
        stats = envelope_stats(poly, points)
        stats_text = (
            f"Coverage: {stats['fraction_contained']:.1%}\n"
            f"Origin inside: {stats['origin_inside']}\n"
            f"Vertices: {stats['num_vertices']}\n"
            f"Area: {stats['area']:.2f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    return ax
