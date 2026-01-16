"""
Projected Envelope Module

Projects 3D point cloud data onto a 2D plane (via PCA) and computes
a robust convex envelope that:
- Contains the origin (zero point)
- Contains approximately 95% of the data (outlier-robust)

This is a prototype step before implementing full 3D surface envelopes.
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

    Attributes
    ----------
    points_2d : np.ndarray, shape (N, 2)
        Projected 2D coordinates
    points_3d : np.ndarray, shape (N, 3)
        Original 3D coordinates
    polar_axis : np.ndarray, shape (3,)
        The polar axis (normal to projection plane)
    basis : np.ndarray, shape (2, 3)
        Orthonormal basis vectors spanning the projection plane
    center_3d : np.ndarray, shape (3,)
        Center point used for projection (usually origin or centroid)
    explained_variance_ratio : np.ndarray, shape (3,)
        PCA explained variance ratios
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

    Attributes
    ----------
    envelope_2d : np.ndarray, shape (M, 2)
        Convex envelope vertices in 2D (CCW order)
    envelope_3d : np.ndarray, shape (M, 3)
        Envelope vertices lifted back to 3D (on the projection plane)
    projection : ProjectionResult
        Full projection information
    stats : dict
        Envelope statistics (coverage, area, etc.)
    """
    envelope_2d: np.ndarray
    envelope_3d: np.ndarray
    projection: ProjectionResult
    stats: dict


def project_to_2d(
    points_3d: np.ndarray,
    center: Optional[np.ndarray] = None,
    return_full: bool = True
) -> ProjectionResult:
    """
    Project 3D points onto a 2D plane using PCA.

    The projection plane is spanned by the two principal components with
    highest variance. The polar axis (normal to the plane) is the direction
    of smallest variance.

    Parameters
    ----------
    points_3d : np.ndarray, shape (N, 3)
        Input 3D points
    center : np.ndarray, shape (3,), optional
        Center point for projection. If None, uses the origin [0, 0, 0].
        Points are translated so this becomes the origin in 2D.
    return_full : bool
        If True, return full ProjectionResult; if False, return just points_2d

    Returns
    -------
    ProjectionResult
        Container with projected points and projection metadata
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {points_3d.shape}")

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=np.float64)

    # Center the data at the specified point
    centered = points_3d - center

    # Fit PCA to find principal directions
    pca = PCA(n_components=3)
    pca.fit(centered)

    # The basis vectors for the 2D plane (first two principal components)
    # These are rows of pca.components_
    basis = pca.components_[:2]  # Shape (2, 3)

    # Polar axis is the third principal component (smallest variance)
    polar_axis = pca.components_[2]  # Shape (3,)

    # Project points onto the 2D plane
    # points_2d[i] = [dot(centered[i], basis[0]), dot(centered[i], basis[1])]
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

    Parameters
    ----------
    points_2d : np.ndarray, shape (K, 2)
        Points in 2D projection space
    projection : ProjectionResult
        Projection metadata from project_to_2d()

    Returns
    -------
    np.ndarray, shape (K, 3)
        Points in original 3D space (on the projection plane)
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
    Project 3D data to 2D and compute a robust convex envelope.

    This is the main entry point. It:
    1. Projects 3D points onto the principal plane (via PCA)
    2. Computes a convex envelope containing ~coverage fraction of points
    3. Ensures the origin/center is inside the envelope

    Parameters
    ----------
    points_3d : np.ndarray, shape (N, 3)
        Input 3D point cloud
    coverage : float
        Fraction of points to include in envelope (default 0.95)
    include_origin : bool
        Whether to ensure origin is inside envelope (default True)
    center : np.ndarray, shape (3,), optional
        3D center point. If None, uses [0, 0, 0].
        This point maps to (0, 0) in 2D and will be inside the envelope
        if include_origin=True.

    Returns
    -------
    EnvelopeResult
        Container with envelope vertices (2D and 3D), projection info, and stats
    """
    # Project to 2D
    projection = project_to_2d(points_3d, center=center)

    # Fit envelope in 2D
    envelope_2d = fit_envelope(
        projection.points_2d,
        coverage=coverage,
        include_origin=include_origin
    )

    # Lift envelope back to 3D
    envelope_3d = lift_to_3d(envelope_2d, projection)

    # Compute stats
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
        Result from fit_projected_envelope()
    points_3d : np.ndarray, shape (K, 3)
        3D points to test

    Returns
    -------
    np.ndarray, shape (K,), dtype=bool
        True if point projects inside the 2D envelope
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
    Visualize the projected envelope in both 2D and 3D.

    Parameters
    ----------
    result : EnvelopeResult
        Result from fit_projected_envelope()
    show_3d : bool
        If True, show side-by-side 2D and 3D plots
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib Figure
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


if __name__ == "__main__":
    # Demo: Generate 3D data and compute projected envelope
    np.random.seed(42)

    # Create a 3D point cloud - elongated ellipsoid with some outliers
    n_main = 200
    n_outliers = 10

    # Main cluster: stretched along one axis
    main_points = np.random.randn(n_main, 3)
    main_points[:, 0] *= 3.0  # Stretch along x
    main_points[:, 1] *= 2.0  # Medium along y
    main_points[:, 2] *= 0.5  # Thin along z

    # Add some rotation to make it interesting
    theta = np.pi / 6
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    main_points = main_points @ rotation.T

    # Add outliers
    outliers = np.random.randn(n_outliers, 3) * 0.5
    outliers += np.array([[8, 8, 2], [-7, 5, -1], [6, -8, 1],
                          [-8, -8, -2], [10, 0, 0], [0, 10, 1],
                          [-5, -5, 3], [7, 7, -2], [-9, 3, 0], [4, -9, 1]])

    all_points = np.vstack([main_points, outliers])

    # Compute projected envelope
    print("Computing projected envelope...")
    result = fit_projected_envelope(
        all_points,
        coverage=0.95,
        include_origin=True
    )

    # Print results
    print("\n=== Projected Envelope Results ===")
    print(f"Input: {len(all_points)} 3D points")
    print(f"\nPCA Analysis:")
    print(f"  Explained variance: {result.projection.explained_variance_ratio}")
    print(f"  Polar axis: {result.projection.polar_axis}")
    print(f"\nEnvelope Stats:")
    for key, value in result.stats.items():
        if key != 'centroid':
            print(f"  {key}: {value}")

    # Test containment of origin
    origin_inside = contains_projected(result, np.array([[0, 0, 0]]))[0]
    print(f"\nOrigin [0,0,0] inside envelope: {origin_inside}")

    # Plot
    fig = plot_projected_envelope(result, show_3d=True)
    plt.savefig('projected_envelope_demo.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: projected_envelope_demo.png")
    plt.show()
