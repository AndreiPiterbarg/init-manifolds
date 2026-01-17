"""
Envelope - Robust envelope fitting for 2D and 3D point data.

This package provides tools for fitting concave envelopes around point
data that:
- Contain approximately a target fraction of data points
- Are resistant to outliers using Minimum Covariance Determinant
- Support both 2D and 3D (via PCA projection) data

Main Functions
--------------
fit_envelope : Fit a 2D concave envelope using alpha shapes
fit_projected_envelope : Project 3D data to 2D and fit envelope
contains : Test point containment in a polygon
contains_projected : Test 3D point containment in projected envelope

Example
-------
>>> import numpy as np
>>> from envelope import fit_envelope, contains

>>> points = np.random.randn(100, 2)
>>> envelope = fit_envelope(points, coverage=0.95)
>>> inside = contains(envelope, points)
"""

from .core.geometry import contains
from .envelopes.concave import fit_envelope, envelope_stats
from .projection.pca import ProjectionResult, project_to_2d, lift_to_3d
from .projection.projected_envelope import EnvelopeResult, fit_projected_envelope, contains_projected
from .visualization.plotting import plot_envelope, plot_projected_envelope

# For backward compatibility with internal imports
from .core.geometry import _polygon_area

__all__ = [
    # Core geometry
    'contains',
    # 2D envelope
    'fit_envelope',
    'envelope_stats',
    # Projection
    'ProjectionResult',
    'project_to_2d',
    'lift_to_3d',
    # 3D envelope
    'EnvelopeResult',
    'fit_projected_envelope',
    'contains_projected',
    # Visualization
    'plot_envelope',
    'plot_projected_envelope',
    # Internal (for backward compat)
    '_polygon_area',
]
