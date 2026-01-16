"""
Unit tests for convex_envelope module.
"""

import numpy as np
import pytest
from convex_envelope import fit_envelope, contains, envelope_stats


class TestContains:
    """Tests for the contains() function."""

    def test_point_inside_square(self):
        """Point clearly inside a square should be detected."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.array([[0.5, 0.5]])
        assert contains(square, points)[0]

    def test_point_outside_square(self):
        """Point clearly outside should be detected."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.array([[2.0, 2.0]])
        assert not contains(square, points)[0]

    def test_point_on_edge(self):
        """Point on edge should be considered inside."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.array([[0.5, 0.0]])  # On bottom edge
        assert contains(square, points)[0]

    def test_point_on_vertex(self):
        """Point on vertex should be considered inside."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.array([[0.0, 0.0]])  # On corner
        assert contains(square, points)[0]

    def test_multiple_points(self):
        """Test multiple points at once."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]], dtype=float)
        points = np.array([
            [1.0, 0.5],   # Inside
            [1.0, 1.0],   # Inside
            [0.0, 2.0],   # Outside
            [3.0, 0.0],   # Outside
        ])
        result = contains(triangle, points)
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_empty_polygon(self):
        """Polygon with < 3 vertices should return False for all points."""
        line = np.array([[0, 0], [1, 1]], dtype=float)
        points = np.array([[0.5, 0.5]])
        assert not contains(line, points)[0]


class TestFitEnvelope:
    """Tests for fit_envelope() function."""

    def test_basic_output_shape(self):
        """Output should be (M, 2) array."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope = fit_envelope(points)
        assert envelope.ndim == 2
        assert envelope.shape[1] == 2
        assert envelope.shape[0] >= 3  # At least a triangle

    def test_coverage_parameter(self):
        """Different coverage values should give different envelopes."""
        np.random.seed(42)
        points = np.random.randn(100, 2) * 2

        env_50 = fit_envelope(points, coverage=0.5)
        env_95 = fit_envelope(points, coverage=0.95)
        env_100 = fit_envelope(points, coverage=1.0)

        # Higher coverage should give larger area (roughly)
        from convex_envelope import _polygon_area
        area_50 = _polygon_area(env_50)
        area_95 = _polygon_area(env_95)
        area_100 = _polygon_area(env_100)

        assert area_50 <= area_95 <= area_100

    def test_origin_included(self):
        """Origin should be inside when include_origin=True."""
        np.random.seed(42)
        # Points not containing origin
        points = np.random.randn(50, 2) + [5, 5]

        envelope = fit_envelope(points, include_origin=True)
        origin_inside = contains(envelope, np.array([[0, 0]]))[0]
        assert origin_inside

    def test_origin_not_required(self):
        """Origin may be outside when include_origin=False."""
        np.random.seed(42)
        # Points far from origin
        points = np.random.randn(50, 2) + [10, 10]

        envelope = fit_envelope(points, include_origin=False)
        # Origin should likely be outside (though not guaranteed)
        # Just check it doesn't crash
        assert envelope.shape[0] >= 3

    def test_polygon_is_convex(self):
        """Output polygon should be convex (all cross products same sign)."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope = fit_envelope(points)

        # Check convexity: all cross products should be >= 0 for CCW polygon
        n = len(envelope)
        for i in range(n):
            v1 = envelope[i]
            v2 = envelope[(i + 1) % n]
            v3 = envelope[(i + 2) % n]

            edge1 = v2 - v1
            edge2 = v3 - v2
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            assert cross >= -1e-10, f"Polygon not convex at vertex {i}"

    def test_outlier_robustness(self):
        """Outliers should not dramatically expand the envelope."""
        np.random.seed(42)

        # Main cluster near origin
        main_points = np.random.randn(95, 2)

        # Extreme outliers
        outliers = np.array([[100, 100], [-100, 100], [100, -100], [-100, -100], [0, 150]])
        all_points = np.vstack([main_points, outliers])

        envelope = fit_envelope(all_points, coverage=0.95)
        stats = envelope_stats(envelope, all_points)

        # Envelope should not include the outliers (area would be huge)
        # Main cluster has std ~1, so reasonable envelope area is < 50
        assert stats['area'] < 100, f"Envelope too large: {stats['area']}"

    def test_small_dataset_3_points(self):
        """Should handle exactly 3 points."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        envelope = fit_envelope(points, coverage=1.0, include_origin=False)
        assert envelope.shape[0] == 3

    def test_small_dataset_raises(self):
        """Should raise ValueError for < 3 points."""
        with pytest.raises(ValueError):
            fit_envelope(np.array([[0, 0], [1, 1]], dtype=float))
        with pytest.raises(ValueError):
            fit_envelope(np.array([[0, 0]], dtype=float))

    def test_invalid_coverage_raises(self):
        """Invalid coverage values should raise ValueError."""
        points = np.random.randn(10, 2)
        with pytest.raises(ValueError):
            fit_envelope(points, coverage=0.0)
        with pytest.raises(ValueError):
            fit_envelope(points, coverage=1.5)

    def test_wrong_shape_raises(self):
        """Wrong input shape should raise ValueError."""
        with pytest.raises(ValueError):
            fit_envelope(np.random.randn(10, 3))  # 3D points
        with pytest.raises(ValueError):
            fit_envelope(np.random.randn(10))  # 1D array


class TestEnvelopeStats:
    """Tests for envelope_stats() function."""

    def test_basic_stats(self):
        """Should return expected keys."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope = fit_envelope(points)
        stats = envelope_stats(envelope, points)

        assert 'fraction_contained' in stats
        assert 'origin_inside' in stats
        assert 'num_vertices' in stats
        assert 'area' in stats
        assert 'centroid' in stats

    def test_fraction_contained_range(self):
        """Fraction should be between 0 and 1."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        envelope = fit_envelope(points, coverage=0.95)
        stats = envelope_stats(envelope, points)

        assert 0 <= stats['fraction_contained'] <= 1

    def test_fraction_approximately_matches_coverage(self):
        """Actual fraction should be close to requested coverage."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        for target_coverage in [0.5, 0.8, 0.95]:
            envelope = fit_envelope(points, coverage=target_coverage)
            stats = envelope_stats(envelope, points)
            # Allow some tolerance
            assert stats['fraction_contained'] >= target_coverage - 0.1

    def test_origin_check(self):
        """Origin inside flag should be accurate."""
        # Envelope that definitely contains origin
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        points = np.random.randn(10, 2) * 0.5  # Points near origin
        stats = envelope_stats(square, points)
        assert stats['origin_inside'] is True

        # Envelope that definitely excludes origin
        square_far = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
        stats_far = envelope_stats(square_far, points)
        assert stats_far['origin_inside'] is False


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_fit_and_verify(self):
        """Full workflow: fit envelope, check stats, verify properties."""
        np.random.seed(123)

        # Generate realistic data
        points = np.random.randn(500, 2) * 3

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        # Verify basic properties
        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.85  # Should be close to 95%
        assert stats['num_vertices'] >= 3
        assert stats['area'] > 0

        # Verify contains() is consistent with stats
        inside_mask = contains(envelope, points)
        assert np.sum(inside_mask) == stats['points_inside']

    def test_reproducibility(self):
        """Same seed should give same results."""
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            points = np.random.randn(100, 2)

            np.random.seed(seed + 1000)  # Different seed for fit
            env1 = fit_envelope(points.copy(), coverage=0.9)

            np.random.seed(seed + 1000)  # Same seed
            env2 = fit_envelope(points.copy(), coverage=0.9)

            np.testing.assert_array_almost_equal(env1, env2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
