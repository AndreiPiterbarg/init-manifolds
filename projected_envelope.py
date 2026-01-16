"""
Projected Envelope Module:

Projects 3D point data onto a 2D plane (via PCA) and computes
convex envelope that:
- Contains the origin (zero point)
- Contains approximately 95% of the data (outlier-robust)

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from convex_envelope import fit_envelope, contains, envelope_stats, plot_envelope


@dataclass
class ProjectionResult:
    """
    Container for projection results and metadata.

    points_2d : Projected 2D coordinates
    points_3d : Original 3D coordinates
    polar_axis : The polar axis (normal to projection plane)
    basis : Orthonormal basis vectors spanning the projection plane
    center_3d : Center point used for projection (usually origin or centroid)
    explained_variance_ratio : PCA explained variance ratios
    """
    points_2d: np.ndarray
    points_3d: np.ndarray
    polar_axis: np.ndarray
    basis: np.ndarray
    center_3d: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass
class EnvelopeResult:
    """
    Container for the full envelope computation results.

    envelope_2d : Convex envelope vertices in 2D (CCW order)
    envelope_3d : Envelope vertices lifted back to 3D (on the projection plane)
    projection : Full projection information
    stats : Envelope statistics (coverage, area, etc.)
    """
    envelope_2d: np.ndarray
    envelope_3d: np.ndarray
    projection: ProjectionResult
    stats: dict


def project_to_2d(
    points_3d: np.ndarray,
    center: Optional[np.ndarray] = None
) -> ProjectionResult:
    """
    Project 3D points onto a 2D plane using PCA.

    The projection plane is spanned by the two principal components with highest variance. Thus components that best capture data spread. Preserves as much original info.

    The polar axis (normal to the plane) is the direction of smallest variance.

    """
    points_3d = np.asarray(points_3d, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {points_3d.shape}")

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=np.float64)

    centered = points_3d - center
    pca = PCA(n_components=3)
    pca.fit(centered)

    basis = pca.components_[:2]  # Shape (2, 3)

    polar_axis = pca.components_[2]  # Shape (3,)

    points_2d = centered @ basis.T  # Shape (N, 2)

    return ProjectionResult(
        points_2d=points_2d,
        points_3d=points_3d,
        polar_axis=polar_axis,
        basis=basis,
        center_3d=center,
        explained_variance_ratio=pca.explained_variance_ratio_
    )


def lift_to_3d(
    points_2d: np.ndarray,
    projection: ProjectionResult
) -> np.ndarray:
    """
    Lift 2D points back to 3D on the projection plane.
    """
    points_2d = np.atleast_2d(points_2d)

    # Reconstruct 3D coordinates: point_3d = center + u * basis[0] + v * basis[1]
    points_3d = (
        projection.center_3d +
        points_2d[:, 0:1] * projection.basis[0] +
        points_2d[:, 1:2] * projection.basis[1]
    )

    return points_3d


def fit_projected_envelope(
    points_3d: np.ndarray,
    coverage: float = 0.95,
    include_origin: bool = True,
    center: Optional[np.ndarray] = None
) -> EnvelopeResult:
    """
    MAIN METHOD: Project 3D data to 2D and compute a robust convex envelope.

    1. Projects 3D points onto the principal plane (via PCA)
    2. Computes a convex envelope containing ~coverage fraction of points
    3. Ensures the origin/center is inside the envelope
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


    """
    points_3d = np.atleast_2d(points_3d)

    # Project query points using the same basis
    centered = points_3d - result.projection.center_3d
    points_2d = centered @ result.projection.basis.T

    return contains(result.envelope_2d, points_2d)


def plot_projected_envelope(
    result: EnvelopeResult,
    show_3d: bool = True,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Helper to visualize the projected envelope in both 2D and 3D.

    """
    if show_3d:
        fig = plt.figure(figsize=figsize)

        # 2D projection plot
        ax1 = fig.add_subplot(1, 2, 1)
        plot_envelope(
            result.envelope_2d,
            result.projection.points_2d,
            ax=ax1,
            title="2D Projection with Envelope"
        )

        # Add variance info
        var_text = (
            f"Explained variance:\n"
            f"  PC1: {result.projection.explained_variance_ratio[0]:.1%}\n"
            f"  PC2: {result.projection.explained_variance_ratio[1]:.1%}\n"
            f"  Polar: {result.projection.explained_variance_ratio[2]:.1%}"
        )
        ax1.text(
            0.02, 0.78, var_text,
            transform=ax1.transAxes,
            fontfamily='monospace',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

        # 3D plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        _plot_3d(result, ax2)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_envelope(
            result.envelope_2d,
            result.projection.points_2d,
            ax=ax,
            title="2D Projection with Envelope"
        )

    plt.tight_layout()
    return fig


def _plot_3d(result: EnvelopeResult, ax: Axes3D) -> None:
    """Helper to create 3D visualization."""
    proj = result.projection
    points_3d = proj.points_3d

    # Determine inside/outside
    inside_mask = contains(result.envelope_2d, proj.points_2d)

    # Plot points
    ax.scatter(
        points_3d[inside_mask, 0],
        points_3d[inside_mask, 1],
        points_3d[inside_mask, 2],
        c='steelblue', alpha=0.5, s=15, label='Inside'
    )
    ax.scatter(
        points_3d[~inside_mask, 0],
        points_3d[~inside_mask, 1],
        points_3d[~inside_mask, 2],
        c='coral', alpha=0.5, s=15, label='Outside'
    )

    # Plot envelope in 3D (as polygon on the projection plane)
    env_3d = result.envelope_3d
    closed_env = np.vstack([env_3d, env_3d[0]])
    ax.plot(
        closed_env[:, 0], closed_env[:, 1], closed_env[:, 2],
        'k-', linewidth=2, label='Envelope'
    )

    # Mark envelope vertices
    ax.scatter(
        env_3d[:, 0], env_3d[:, 1], env_3d[:, 2],
        c='black', s=50, marker='s'
    )

    # Mark origin/center
    center = proj.center_3d
    ax.scatter([center[0]], [center[1]], [center[2]],
               c='red', s=150, marker='*', label='Center')

    # Draw polar axis
    axis_scale = np.max(np.abs(points_3d)) * 0.3
    polar_end = center + proj.polar_axis * axis_scale
    ax.plot(
        [center[0], polar_end[0]],
        [center[1], polar_end[1]],
        [center[2], polar_end[2]],
        'g--', linewidth=2, label='Polar axis'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D View with Projection Plane')
    ax.legend(loc='upper left', fontsize=8)
