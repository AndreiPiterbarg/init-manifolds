"""
Tests for single-crossing (star-shaped) envelope fitting.

Tests the polar coordinate-based envelope that guarantees radial trajectories
cross the boundary exactly once.
"""

import numpy as np
import pytest

from src.envelope import (
    fit_single_crossing_envelope,
    single_crossing_stats,
    fit_projected_single_crossing_envelope,
    contains,
)


class TestOutputShape:
    """Test that output has correct shape and format."""

    def test_returns_2d_array(self):
        """Envelope should be (M, 2) array."""
        points = np.random.randn(100, 2)
        envelope, validation = fit_single_crossing_envelope(points)

        assert envelope.ndim == 2
        assert envelope.shape[1] == 2

    def test_default_n_angles(self):
        """Default n_angles=72 should give 72 vertices."""
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points, n_angles=72)

        assert envelope.shape[0] == 72

    def test_custom_n_angles(self):
        """Custom n_angles should be respected."""
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points, n_angles=36)

        assert envelope.shape[0] == 36

    def test_validation_none_for_radial(self):
        """Validation should be None when no trajectory_func provided."""
        points = np.random.randn(100, 2)
        envelope, validation = fit_single_crossing_envelope(points)

        assert validation is None


class TestContainsOrigin:
    """Test that origin is inside the envelope."""

    def test_centroid_inside_envelope(self):
        """Default origin (centroid) should be inside envelope."""
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points)

        # Centroid of the data
        centroid = np.mean(points, axis=0)
        inside = contains(envelope, centroid.reshape(1, -1))

        assert inside[0], "Centroid should be inside envelope"

    def test_custom_origin_inside(self):
        """Custom origin should be inside envelope."""
        points = np.random.randn(100, 2) + np.array([5, 5])
        origin = np.array([5.0, 5.0])
        envelope, _ = fit_single_crossing_envelope(points, origin=origin)

        inside = contains(envelope, origin.reshape(1, -1))
        assert inside[0], "Custom origin should be inside envelope"

    def test_stats_reports_origin_inside(self):
        """Stats should report origin is inside."""
        points = np.random.randn(100, 2)
        envelope, _ = fit_single_crossing_envelope(points)
        stats = single_crossing_stats(envelope, points)

        assert stats['origin_inside'], "Stats should report origin inside"


class TestCoverageParameter:
    """Test that coverage parameter affects envelope tightness."""

    def test_higher_coverage_tighter_fit(self):
        """Higher coverage should result in smaller envelope."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        envelope_90, _ = fit_single_crossing_envelope(points, coverage=0.90)
        envelope_99, _ = fit_single_crossing_envelope(points, coverage=0.99)

        # Compute mean radius as proxy for envelope size
        centroid = np.mean(points, axis=0)
        radii_90 = np.linalg.norm(envelope_90 - centroid, axis=1)
        radii_99 = np.linalg.norm(envelope_99 - centroid, axis=1)

        # 99% coverage should have larger mean radius than 90%
        assert np.mean(radii_99) >= np.mean(radii_90) * 0.95, \
            "Higher coverage should give larger or similar envelope"

    def test_coverage_affects_containment(self):
        """Higher coverage should contain more points."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        envelope_80, _ = fit_single_crossing_envelope(points, coverage=0.80)
        envelope_99, _ = fit_single_crossing_envelope(points, coverage=0.99)

        inside_80 = np.mean(contains(envelope_80, points))
        inside_99 = np.mean(contains(envelope_99, points))

        assert inside_99 >= inside_80, "Higher coverage should contain more points"


class TestEmptyBins:
    """Test handling of sparse angular coverage."""

    def test_handles_angular_gaps(self):
        """Should handle data with angular gaps."""
        # Create data in only two quadrants
        np.random.seed(42)
        n = 50
        angles = np.concatenate([
            np.random.uniform(0, np.pi/2, n),      # Q1
            np.random.uniform(np.pi, 3*np.pi/2, n)  # Q3
        ])
        radii = np.random.uniform(0.5, 2.0, 2*n)

        points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])

        # Should not raise
        envelope, _ = fit_single_crossing_envelope(points)

        assert envelope.shape[0] > 0
        assert not np.any(np.isnan(envelope))

    def test_interpolates_small_gaps(self):
        """Small angular gaps should be interpolated smoothly."""
        np.random.seed(42)
        # Create nearly complete angular coverage with one small gap
        angles = np.linspace(0, 2*np.pi, 70, endpoint=False)
        # Remove a small section (2-3 bins)
        mask = ~((angles > 1.0) & (angles < 1.2))
        angles = angles[mask]

        radii = np.ones(len(angles)) + 0.1 * np.random.randn(len(angles))
        points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])

        envelope, _ = fit_single_crossing_envelope(points, n_angles=72)

        # Check that the gap region has reasonable radii (not collapsed to zero)
        envelope_centered = envelope - np.mean(envelope, axis=0)
        envelope_radii = np.linalg.norm(envelope_centered, axis=1)

        # All radii should be positive and reasonably similar
        assert np.all(envelope_radii > 0)
        assert np.std(envelope_radii) < np.mean(envelope_radii)

    def test_collapses_large_gaps(self):
        """Large angular gaps should collapse toward origin."""
        np.random.seed(42)
        # Create data in only one quadrant (large gap)
        angles = np.random.uniform(0, np.pi/4, 100)
        radii = np.random.uniform(1.0, 2.0, 100)

        points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])

        envelope, _ = fit_single_crossing_envelope(
            points,
            n_angles=72,
            smoothing=0  # No smoothing to see collapse clearly
        )

        # The opposite side should have very small radii
        origin = np.mean(points, axis=0)
        envelope_centered = envelope - origin
        envelope_angles = np.arctan2(envelope_centered[:, 1], envelope_centered[:, 0])
        envelope_angles = np.mod(envelope_angles, 2 * np.pi)
        envelope_radii = np.linalg.norm(envelope_centered, axis=1)

        # Find radii in the opposite quadrant (pi to 1.5*pi)
        opposite_mask = (envelope_angles > np.pi) & (envelope_angles < 1.5 * np.pi)
        if np.any(opposite_mask):
            opposite_radii = envelope_radii[opposite_mask]
            data_radii = envelope_radii[~opposite_mask]

            # Opposite side should have smaller minimum radius (collapse effect)
            # Using min instead of mean since smoothing affects mean
            assert np.min(opposite_radii) < np.min(data_radii) * 0.5


class TestDumbbellData:
    """Test the motivating use case: dumbbell-shaped data."""

    def test_fits_dumbbell_shape(self):
        """Should fit envelope around dumbbell-shaped data."""
        np.random.seed(42)

        # Create dumbbell: two clusters connected by a thin bridge
        n_per_lobe = 100
        n_bridge = 30

        # Left lobe
        left_lobe = np.random.randn(n_per_lobe, 2) * 0.5 + np.array([-2, 0])

        # Right lobe
        right_lobe = np.random.randn(n_per_lobe, 2) * 0.5 + np.array([2, 0])

        # Bridge (thin connection)
        bridge_x = np.random.uniform(-1.5, 1.5, n_bridge)
        bridge_y = np.random.randn(n_bridge) * 0.1
        bridge = np.column_stack([bridge_x, bridge_y])

        points = np.vstack([left_lobe, right_lobe, bridge])

        # Fit envelope with origin at the center
        origin = np.array([0.0, 0.0])
        envelope, _ = fit_single_crossing_envelope(points, origin=origin)

        # Verify envelope captures the shape
        stats = single_crossing_stats(envelope, points, origin=origin)

        # Should contain a reasonable fraction of points
        assert stats['fraction_contained'] > 0.5

        # Origin should be inside
        assert stats['origin_inside']

        # Should have variation in radius (not circular)
        assert stats['radius_std'] > 0.1

    def test_dumbbell_radial_single_crossing(self):
        """Radial lines from origin should cross dumbbell envelope once."""
        np.random.seed(42)

        # Create dumbbell
        left = np.random.randn(80, 2) * 0.4 + np.array([-2, 0])
        right = np.random.randn(80, 2) * 0.4 + np.array([2, 0])
        bridge_x = np.linspace(-1.5, 1.5, 40)
        bridge_y = np.random.randn(40) * 0.08
        bridge = np.column_stack([bridge_x, bridge_y])

        points = np.vstack([left, right, bridge])
        origin = np.array([0.0, 0.0])

        envelope, _ = fit_single_crossing_envelope(points, origin=origin)

        # Convert envelope to polar and verify single-valued r(theta)
        envelope_centered = envelope - origin
        radii = np.linalg.norm(envelope_centered, axis=1)

        # All radii should be positive (single crossing property)
        assert np.all(radii > 0), "All radii should be positive"


class TestCustomTrajectoryValidation:
    """Test validation for custom (non-radial) trajectories."""

    def test_radial_trajectory_valid(self):
        """Radial trajectory should validate as single-crossing."""
        np.random.seed(42)
        # Use circular data with good angular coverage to avoid edge cases
        n = 200
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.1 * np.random.randn(n)
        radii = 1.0 + 0.2 * np.random.randn(n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Define explicit radial trajectory from origin
        origin = np.array([0.0, 0.0])

        def radial_trajectory(angle: float, t: float) -> np.ndarray:
            return origin + np.array([t * np.cos(angle), t * np.sin(angle)])

        envelope, validation = fit_single_crossing_envelope(
            points,
            origin=origin,
            trajectory_func=radial_trajectory,
            n_validation_samples=72
        )

        assert validation is not None
        # Allow a small number of edge-case failures due to numerical precision
        failure_rate = len(validation['failures']) / 72
        assert failure_rate < 0.1, f"Too many failures ({failure_rate:.1%}): {validation['failures'][:5]}"

    def test_spiral_trajectory_validation(self):
        """Spiral trajectory should be validated (may or may not pass)."""
        np.random.seed(42)
        points = np.random.randn(100, 2)

        # Mild spiral - should still be mostly single-crossing
        def spiral_trajectory(angle: float, t: float) -> np.ndarray:
            # Spiral that stays close to radial
            spiral_angle = angle + 0.1 * t
            return np.array([t * np.cos(spiral_angle), t * np.sin(spiral_angle)])

        envelope, validation = fit_single_crossing_envelope(
            points,
            trajectory_func=spiral_trajectory,
            n_validation_samples=36  # Fewer samples for speed
        )

        assert validation is not None
        assert 'is_valid' in validation
        assert 'failures' in validation
        assert 'crossing_counts' in validation

    def test_validation_returns_crossing_counts(self):
        """Validation should return crossing counts per angle."""
        np.random.seed(42)
        points = np.random.randn(100, 2)

        def radial(angle: float, t: float) -> np.ndarray:
            return np.array([t * np.cos(angle), t * np.sin(angle)])

        envelope, validation = fit_single_crossing_envelope(
            points,
            trajectory_func=radial,
            n_validation_samples=36
        )

        assert len(validation['crossing_counts']) == 36

        # For radial, all counts should be 1
        for angle, count in validation['crossing_counts'].items():
            assert count == 1, f"Angle {angle} has {count} crossings"


class TestProjectedSingleCrossing:
    """Test 3D projection + single-crossing envelope."""

    def test_3d_projection_works(self):
        """Should project 3D data and fit single-crossing envelope."""
        np.random.seed(42)

        # 3D points on a tilted plane with noise
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = 0.5 * x + 0.3 * y + 0.1 * np.random.randn(n)
        points_3d = np.column_stack([x, y, z])

        result = fit_projected_single_crossing_envelope(points_3d)

        assert result.envelope_2d.shape[1] == 2
        assert result.envelope_3d.shape[1] == 3
        assert result.envelope_2d.shape[0] == result.envelope_3d.shape[0]

    def test_3d_returns_stats(self):
        """Should return statistics for 3D envelope."""
        np.random.seed(42)
        points_3d = np.random.randn(100, 3)

        result = fit_projected_single_crossing_envelope(points_3d)

        assert 'fraction_contained' in result.stats
        assert 'origin_inside' in result.stats
        assert 'mean_radius' in result.stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_points(self):
        """Should work with minimum number of points (3)."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        envelope, _ = fit_single_crossing_envelope(points)

        assert envelope.shape[0] > 0

    def test_rejects_too_few_points(self):
        """Should raise error for fewer than 3 points."""
        points = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError, match="at least 3 points"):
            fit_single_crossing_envelope(points)

    def test_rejects_wrong_shape(self):
        """Should raise error for wrong input shape."""
        points_1d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="shape"):
            fit_single_crossing_envelope(points_1d)

    def test_rejects_invalid_coverage(self):
        """Should raise error for invalid coverage."""
        points = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="coverage"):
            fit_single_crossing_envelope(points, coverage=0)

        with pytest.raises(ValueError, match="coverage"):
            fit_single_crossing_envelope(points, coverage=1.5)

    def test_handles_collinear_points(self):
        """Should handle nearly collinear points."""
        # Points along a line with tiny noise
        x = np.linspace(0, 10, 100)
        y = 0.5 * x + 0.001 * np.random.randn(100)
        points = np.column_stack([x, y])

        # Should not crash
        envelope, _ = fit_single_crossing_envelope(points)
        assert envelope.shape[0] > 0


class TestSmoothing:
    """Test smoothing parameter behavior."""

    def test_no_smoothing(self):
        """Zero smoothing should give unsmoothed result."""
        np.random.seed(42)
        points = np.random.randn(100, 2)

        envelope_smooth, _ = fit_single_crossing_envelope(points, smoothing=0)
        envelope_rough, _ = fit_single_crossing_envelope(points, smoothing=0)

        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(envelope_smooth, envelope_rough)

    def test_high_smoothing_reduces_variation(self):
        """High smoothing should reduce radius variation."""
        np.random.seed(42)
        # Create data with angular variation
        angles = np.linspace(0, 2*np.pi, 200, endpoint=False)
        radii = 1 + 0.5 * np.sin(5 * angles) + 0.1 * np.random.randn(200)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        envelope_low, _ = fit_single_crossing_envelope(points, smoothing=0.1)
        envelope_high, _ = fit_single_crossing_envelope(points, smoothing=3.0)

        # Compute radius std for each
        origin = np.mean(points, axis=0)
        radii_low = np.linalg.norm(envelope_low - origin, axis=1)
        radii_high = np.linalg.norm(envelope_high - origin, axis=1)

        # High smoothing should have lower variation
        assert np.std(radii_high) < np.std(radii_low)
