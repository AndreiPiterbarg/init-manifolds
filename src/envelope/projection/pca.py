"""
PCA Projection Module

Projects 3D point data onto a 2D plane via Principal Component Analysis.
The projection plane is spanned by the two principal components with
highest variance, preserving as much original information as possible.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ProjectionResult:
    """
    Container for projection results and metadata.

    Attributes
    ----------
    points_2d : np.ndarray
        Projected 2D coordinates of shape (N, 2).
    points_3d : np.ndarray
        Original 3D coordinates of shape (N, 3).
    polar_axis : np.ndarray
        The polar axis (normal to projection plane) of shape (3,).
    basis : np.ndarray
        Orthonormal basis vectors spanning the projection plane of shape (2, 3).
    center_3d : np.ndarray
        Center point used for projection (usually origin or centroid) of shape (3,).
    explained_variance_ratio : np.ndarray
        PCA explained variance ratios of shape (3,).
    """
    points_2d: np.ndarray
    points_3d: np.ndarray
    polar_axis: np.ndarray
    basis: np.ndarray
    center_3d: np.ndarray
    explained_variance_ratio: np.ndarray


def project_to_2d(
    points_3d: np.ndarray,
    center: Optional[np.ndarray] = None
) -> ProjectionResult:
    """
    Project 3D points onto a 2D plane using PCA.

    The projection plane is spanned by the two principal components with
    highest variance. Thus components that best capture data spread.
    Preserves as much original info.

    The polar axis (normal to the plane) is the direction of smallest variance.

    Parameters
    ----------
    points_3d : np.ndarray
        Array of shape (N, 3) containing 3D points.
    center : np.ndarray, optional
        Center point for projection. Defaults to origin [0, 0, 0].

    Returns
    -------
    ProjectionResult
        Container with projection data and metadata.
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

    Parameters
    ----------
    points_2d : np.ndarray
        Array of shape (N, 2) containing 2D points.
    projection : ProjectionResult
        Projection result containing basis and center.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing 3D points on the projection plane.
    """
    points_2d = np.atleast_2d(points_2d)

    # Reconstruct 3D coordinates: point_3d = center + u * basis[0] + v * basis[1]
    points_3d = (
        projection.center_3d +
        points_2d[:, 0:1] * projection.basis[0] +
        points_2d[:, 1:2] * projection.basis[1]
    )

    return points_3d
