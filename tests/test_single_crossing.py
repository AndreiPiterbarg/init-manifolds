"""Tests for single-crossing envelope fitting."""

import numpy as np
import pytest

from src.envelope.envelopes.single_crossing import (
    fit_single_crossing_envelope,
    single_crossing_stats,
)


class TestBasicFunctionality:
    """Basic tests for single-crossing envelope."""

    def test_output_shape(self):
        """Returns (M, 2) array."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points)

        assert envelope.ndim == 2
        assert envelope.shape[1] == 2

    def test_n_angles_determines_vertices(self):
        """n_angles determines number of vertices."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points, n_angles=36)

        assert envelope.shape[0] == 36

    def test_origin_inside(self):
        """Origin should be inside envelope."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points)
        stats = single_crossing_stats(envelope, points)

        assert stats['origin_inside']

    def test_star_shaped(self):
        """Envelope should be star-shaped (all radii positive)."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points)
        stats = single_crossing_stats(envelope, points)

        assert stats['is_star_shaped']


class TestCoverage:
    """Tests for coverage parameter."""

    def test_higher_coverage_larger_envelope(self):
        """Higher coverage gives larger envelope."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        env_80, _ = fit_single_crossing_envelope(points, coverage=0.80)
        env_99, _ = fit_single_crossing_envelope(points, coverage=0.99)

        stats_80 = single_crossing_stats(env_80, points)
        stats_99 = single_crossing_stats(env_99, points)

        assert stats_99['fraction_contained'] >= stats_80['fraction_contained']


class TestEdgeCases:
    """Edge cases."""

    def test_rejects_too_few_points(self):
        """Raises for < 3 points."""
        with pytest.raises(ValueError):
            fit_single_crossing_envelope(np.array([[0, 0], [1, 1]]))

    def test_handles_angular_gaps(self):
        """Handles data with angular gaps."""
        np.random.seed(42)
        angles = np.concatenate([
            np.random.uniform(0, np.pi/2, 50),
            np.random.uniform(np.pi, 3*np.pi/2, 50)
        ])
        radii = np.random.uniform(0.5, 2.0, 100)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        envelope, _ = fit_single_crossing_envelope(points)
        assert not np.any(np.isnan(envelope))


class TestTrajectoryValidation:
    """Custom trajectory validation."""

    def test_radial_trajectory_single_crossing(self):
        """Radial trajectories cross exactly once."""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 200, endpoint=False)
        radii = 1 + 0.2 * np.random.randn(200)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        def radial(angle, t):
            r = 5 * (1 - t)  # t=0 far, t=1 at origin
            return (r * np.cos(angle), r * np.sin(angle))

        envelope, validation = fit_single_crossing_envelope(
            points, trajectory_func=radial, n_angles=36
        )

        assert validation is not None
        assert validation['all_single_crossing']
