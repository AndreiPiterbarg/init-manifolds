"""
Unit tests for projected_envelope module.
"""

import numpy as np
import pytest
from projected_envelope import (
    project_to_2d,
    lift_to_3d,
    fit_projected_envelope,
    contains_projected,
    ProjectionResult,
    EnvelopeResult
)


class TestProjectTo2D:
    """Tests for project_to_2d() function."""

    def test_output_shape(self):
        """Projected points should have shape (N, 2)."""
        points_3d = np.random.randn(100, 3)
        result = project_to_2d(points_3d)
        assert result.points_2d.shape == (100, 2)

    def test_projection_result_fields(self):
        """ProjectionResult should have all expected fields."""
        points_3d = np.random.randn(50, 3)
        result = project_to_2d(points_3d)

        assert result.points_2d.shape == (50, 2)
        assert result.points_3d.shape == (50, 3)
        assert result.polar_axis.shape == (3,)
        assert result.basis.shape == (2, 3)
        assert result.center_3d.shape == (3,)
        assert result.explained_variance_ratio.shape == (3,)

    def test_basis_orthonormal(self):
        """Basis vectors should be orthonormal."""
        points_3d = np.random.randn(100, 3)
        result = project_to_2d(points_3d)

        # Check orthogonality
        dot_product = np.dot(result.basis[0], result.basis[1])
        assert abs(dot_product) < 1e-10

        # Check unit length
        assert abs(np.linalg.norm(result.basis[0]) - 1) < 1e-10
        assert abs(np.linalg.norm(result.basis[1]) - 1) < 1e-10

    def test_polar_axis_orthogonal_to_basis(self):
        """Polar axis should be orthogonal to both basis vectors."""
        points_3d = np.random.randn(100, 3)
        result = project_to_2d(points_3d)

        dot1 = np.dot(result.polar_axis, result.basis[0])
        dot2 = np.dot(result.polar_axis, result.basis[1])

        assert abs(dot1) < 1e-10
        assert abs(dot2) < 1e-10

    def test_custom_center(self):
        """Custom center should shift the projection origin."""
        points_3d = np.random.randn(50, 3) + [5, 5, 5]
        center = np.array([5.0, 5.0, 5.0])

        result = project_to_2d(points_3d, center=center)

        # Mean of projected points should be near origin
        mean_2d = np.mean(result.points_2d, axis=0)
        assert np.linalg.norm(mean_2d) < 1.0  # Approximate

    def test_explained_variance_sums_to_one(self):
        """Explained variance ratios should sum to ~1."""
        points_3d = np.random.randn(100, 3)
        result = project_to_2d(points_3d)

        total = np.sum(result.explained_variance_ratio)
        assert abs(total - 1.0) < 1e-10

    def test_invalid_shape_raises(self):
        """Wrong input shape should raise ValueError."""
        with pytest.raises(ValueError):
            project_to_2d(np.random.randn(10, 2))  # 2D input
        with pytest.raises(ValueError):
            project_to_2d(np.random.randn(10, 4))  # 4D input


class TestLiftTo3D:
    """Tests for lift_to_3d() function."""

    def test_round_trip(self):
        """Projecting then lifting should give points on the plane."""
        np.random.seed(42)
        points_3d = np.random.randn(50, 3)

        projection = project_to_2d(points_3d)
        lifted = lift_to_3d(projection.points_2d, projection)

        # Lifted points should be on the projection plane
        # (distance along polar axis should be zero)
        centered_original = points_3d - projection.center_3d
        centered_lifted = lifted - projection.center_3d

        # Component along polar axis
        original_polar = centered_original @ projection.polar_axis
        lifted_polar = centered_lifted @ projection.polar_axis

        # Lifted points should have zero polar component
        np.testing.assert_array_almost_equal(lifted_polar, np.zeros(50))

    def test_origin_lifts_to_center(self):
        """Origin in 2D should lift to the 3D center."""
        points_3d = np.random.randn(50, 3)
        center = np.array([1.0, 2.0, 3.0])

        projection = project_to_2d(points_3d, center=center)
        lifted_origin = lift_to_3d(np.array([[0, 0]]), projection)

        np.testing.assert_array_almost_equal(lifted_origin[0], center)


class TestFitProjectedEnvelope:
    """Tests for fit_projected_envelope() function."""

    def test_output_type(self):
        """Should return EnvelopeResult."""
        points_3d = np.random.randn(100, 3)
        result = fit_projected_envelope(points_3d)
        assert isinstance(result, EnvelopeResult)

    def test_envelope_shapes(self):
        """Envelope should have correct shapes."""
        points_3d = np.random.randn(100, 3)
        result = fit_projected_envelope(points_3d)

        assert result.envelope_2d.ndim == 2
        assert result.envelope_2d.shape[1] == 2
        assert result.envelope_3d.shape[1] == 3
        assert result.envelope_2d.shape[0] == result.envelope_3d.shape[0]

    def test_origin_included(self):
        """Origin should be inside when include_origin=True."""
        np.random.seed(42)
        # Points not centered at origin
        points_3d = np.random.randn(100, 3) + [5, 5, 5]

        result = fit_projected_envelope(points_3d, include_origin=True)

        # Check origin is inside
        origin_inside = contains_projected(result, np.array([[0, 0, 0]]))[0]
        assert origin_inside

    def test_coverage_respected(self):
        """Actual coverage should be close to requested."""
        np.random.seed(42)
        points_3d = np.random.randn(200, 3)

        result = fit_projected_envelope(points_3d, coverage=0.95)

        assert result.stats['fraction_contained'] >= 0.85

    def test_outlier_robustness(self):
        """Outliers should not dramatically expand envelope."""
        np.random.seed(42)

        # Main cluster
        main = np.random.randn(95, 3)

        # Extreme outliers
        outliers = np.array([
            [50, 50, 50], [-50, 50, -50], [50, -50, 50],
            [-50, -50, -50], [100, 0, 0]
        ])
        all_points = np.vstack([main, outliers])

        result = fit_projected_envelope(all_points, coverage=0.95)

        # Area should be reasonable (not huge from outliers)
        assert result.stats['area'] < 100


class TestContainsProjected:
    """Tests for contains_projected() function."""

    def test_original_points_containment(self):
        """Most original points should be inside."""
        np.random.seed(42)
        points_3d = np.random.randn(100, 3)

        result = fit_projected_envelope(points_3d, coverage=0.95)
        inside = contains_projected(result, points_3d)

        # At least 85% should be inside
        assert np.mean(inside) >= 0.85

    def test_far_point_outside(self):
        """Point far from cluster should be outside."""
        np.random.seed(42)
        points_3d = np.random.randn(100, 3)

        result = fit_projected_envelope(points_3d, coverage=0.95)

        far_point = np.array([[100, 100, 100]])
        inside = contains_projected(result, far_point)

        assert not inside[0]


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow from 3D points to envelope."""
        np.random.seed(123)

        # Generate ellipsoid-like data
        points_3d = np.random.randn(300, 3)
        points_3d[:, 0] *= 3  # Stretch x
        points_3d[:, 2] *= 0.5  # Compress z

        # Fit envelope
        result = fit_projected_envelope(
            points_3d,
            coverage=0.95,
            include_origin=True
        )

        # Verify properties
        assert result.stats['origin_inside']
        assert result.stats['fraction_contained'] >= 0.85
        assert result.envelope_2d.shape[0] >= 3
        assert len(result.projection.explained_variance_ratio) == 3

        # Verify PCA found the right structure
        # (x has most variance, z has least)
        var_ratios = result.projection.explained_variance_ratio
        assert var_ratios[0] > var_ratios[1] > var_ratios[2]

    def test_reproducibility(self):
        """Same input should give same output."""
        np.random.seed(42)
        points_3d = np.random.randn(100, 3)

        result1 = fit_projected_envelope(points_3d.copy(), coverage=0.9)
        result2 = fit_projected_envelope(points_3d.copy(), coverage=0.9)

        np.testing.assert_array_almost_equal(
            result1.envelope_2d,
            result2.envelope_2d
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
