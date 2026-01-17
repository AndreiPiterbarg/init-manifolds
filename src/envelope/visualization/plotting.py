"""
Visualization utilities for envelope plotting.

Contains plotting functions for:
- 2D envelope visualization
- 3D projected envelope visualization
"""

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..core.geometry import contains
from ..envelopes.concave import envelope_stats

if TYPE_CHECKING:
    from ..projection.projected_envelope import EnvelopeResult


def plot_envelope(
    poly: np.ndarray,
    points: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show_origin: bool = True,
    title: str = "Envelope",
    show_stats: bool = True
) -> plt.Axes:
    """
    Visualize the envelope and points in 2D.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).
    points : np.ndarray
        Data points of shape (N, 2).
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Creates new figure if None.
    show_origin : bool
        Whether to mark the origin.
    title : str
        Plot title.
    show_stats : bool
        Whether to show envelope statistics.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.
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


def plot_projected_envelope(
    result: "EnvelopeResult",
    show_3d: bool = True,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualize the projected envelope in both 2D and 3D.

    Parameters
    ----------
    result : EnvelopeResult
        Result from fit_projected_envelope().
    show_3d : bool
        Whether to show 3D view alongside 2D.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
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


def _plot_3d(result: "EnvelopeResult", ax: Axes3D) -> None:
    """
    Helper to create 3D visualization.

    Parameters
    ----------
    result : EnvelopeResult
        Result from fit_projected_envelope().
    ax : Axes3D
        Matplotlib 3D axes.
    """
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
