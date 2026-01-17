"""
Edge case tests for envelope fitting.

Tests cover:
1. Extremely asymmetrical data (one wing 5x larger than the other)
2. Concave data distributions facing the origin
3. Concave data distributions perpendicular to radial lines (multiple crossings)
4. Data far from origin (envelope must still include origin)
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from src.envelope import fit_envelope, contains, envelope_stats, plot_envelope
from src.envelope import fit_projected_envelope


class TestAsymmetricalData:
    """Tests for extremely asymmetrical data where one wing is ~5x larger than the other."""

    def test_asymmetric_hourglass_2d(self):
        """Hourglass shape with origin at center, left side smaller than right."""
        np.random.seed(42)

        n_left = 30
        left_cluster = np.random.randn(n_left, 2) * 1.0
        left_cluster[:, 0] = left_cluster[:, 0] * 0.8 - 3

        n_right = 70
        right_cluster = np.random.randn(n_right, 2)
        right_cluster[:, 0] = right_cluster[:, 0] * 2.0 + 6
        right_cluster[:, 1] = right_cluster[:, 1] * 4.0

        points = np.vstack([left_cluster, right_cluster])
        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.85
        assert envelope.shape[0] >= 3

        x_range = envelope[:, 0].max() - envelope[:, 0].min()
        y_range = envelope[:, 1].max() - envelope[:, 1].min()
        assert x_range > y_range

    def test_asymmetric_radial_spread_2d(self):
        """One radial direction has 5x the point density."""
        np.random.seed(42)

        n_dense = 80
        angles_dense = np.random.uniform(0, np.pi/4, n_dense)
        radii_dense = np.abs(np.random.randn(n_dense)) * 3
        dense_sector = np.column_stack([
            radii_dense * np.cos(angles_dense),
            radii_dense * np.sin(angles_dense)
        ])

        n_sparse = 20
        angles_sparse = np.random.uniform(np.pi/4, 2*np.pi, n_sparse)
        radii_sparse = np.abs(np.random.randn(n_sparse)) * 3
        sparse_sector = np.column_stack([
            radii_sparse * np.cos(angles_sparse),
            radii_sparse * np.sin(angles_sparse)
        ])

        points = np.vstack([dense_sector, sparse_sector])
        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.80


class TestConcaveTowardOrigin:
    """Tests for data with deep concave segments pointed toward the origin."""

    def test_horseshoe_shape_2d(self):
        """Horseshoe (270 degree arc) with opening facing origin."""
        np.random.seed(42)

        n_points = 120
        angles = np.random.uniform(np.pi/4, 7*np.pi/4, n_points)
        radii = np.random.uniform(4, 6, n_points)

        points = np.column_stack([
            radii * np.cos(angles) + 5,
            radii * np.sin(angles)
        ])
        points += np.random.randn(n_points, 2) * 0.2

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.85
        assert envelope.shape[0] >= 3

    def test_v_shape_toward_origin(self):
        """V-shaped data with vertex pointing toward origin."""
        np.random.seed(42)

        n_per_arm = 50

        upper_t = np.random.uniform(0, 1, n_per_arm)
        upper_arm = np.column_stack([
            2 + upper_t * 5,
            1 + upper_t * 3
        ])

        lower_t = np.random.uniform(0, 1, n_per_arm)
        lower_arm = np.column_stack([
            2 + lower_t * 5,
            -1 - lower_t * 3
        ])

        points = np.vstack([upper_arm, lower_arm])
        points += np.random.randn(len(points), 2) * 0.3

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']


class TestConcavePerpendicularMultipleCrossings:
    """Tests for data with concave segments perpendicular to radial lines."""

    def test_sideways_horseshoe_2d(self):
        """Horseshoe with opening perpendicular to radial line from origin."""
        np.random.seed(42)

        n_points = 120
        angles = np.random.uniform(-3*np.pi/4, 3*np.pi/4, n_points)
        radii = np.random.uniform(3, 5, n_points)

        points = np.column_stack([
            radii * np.cos(angles) + 6,
            radii * np.sin(angles) + 4
        ])
        points += np.random.randn(n_points, 2) * 0.2

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.80

    def test_spiral_segment_2d(self):
        """Spiral segment that wraps partially around origin."""
        np.random.seed(42)

        n_points = 100
        theta = np.random.uniform(np.pi/4, 2*np.pi, n_points)
        r = 1 + theta * 0.5

        points = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta)
        ])
        points += np.random.randn(n_points, 2) * 0.3

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']


class TestNoDataNearOrigin:
    """Tests for data far from origin where the envelope must still include origin."""

    def test_cluster_far_from_origin_2d(self):
        """Single cluster far from origin."""
        np.random.seed(42)

        points = np.random.randn(100, 2) * 2 + [10, 10]

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        assert stats['fraction_contained'] >= 0.80
        assert stats['area'] > 0

    def test_two_distant_clusters_2d(self):
        """Two clusters on opposite sides, neither near origin."""
        np.random.seed(42)

        cluster1 = np.random.randn(50, 2) * 1.5 + [10, 5]
        cluster2 = np.random.randn(50, 2) * 1.5 + [-10, -5]

        points = np.vstack([cluster1, cluster2])

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)
        stats = envelope_stats(envelope, points)

        assert stats['origin_inside']
        x_range = envelope[:, 0].max() - envelope[:, 0].min()
        assert x_range > 15


class TestEdgeCaseVisualizations:
    """Visualization tests for each edge case category."""

    @pytest.mark.parametrize("show", [True])
    def test_visualize_asymmetrical(self, show):
        """Visualize hourglass with origin at center, asymmetric wings."""
        np.random.seed(42)

        n_left = 30
        left_cluster = np.random.randn(n_left, 2) * 1.0
        left_cluster[:, 0] = left_cluster[:, 0] * 0.8 - 3

        n_right = 70
        right_cluster = np.random.randn(n_right, 2)
        right_cluster[:, 0] = right_cluster[:, 0] * 2.0 + 6
        right_cluster[:, 1] = right_cluster[:, 1] * 4.0

        points = np.vstack([left_cluster, right_cluster])
        envelope = fit_envelope(points, coverage=0.95, include_origin=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_envelope(envelope, points, ax=ax, title="Asymmetrical Data (Hourglass)")

        if show:
            plt.show()
        else:
            plt.savefig("edge_case_asymmetrical.png", dpi=150, bbox_inches='tight')
        plt.close()

    @pytest.mark.parametrize("show", [True])
    def test_visualize_concave_toward_origin(self, show):
        """Visualize 270-degree horseshoe with opening facing origin."""
        np.random.seed(42)

        n_points = 120
        angles = np.random.uniform(np.pi/4, 7*np.pi/4, n_points)
        radii = np.random.uniform(4, 6, n_points)

        points = np.column_stack([
            radii * np.cos(angles) + 5,
            radii * np.sin(angles)
        ])
        points += np.random.randn(n_points, 2) * 0.2

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_envelope(envelope, points, ax=ax, title="Concave Toward Origin (270° Horseshoe)")

        if show:
            plt.show()
        else:
            plt.savefig("edge_case_concave_toward_origin.png", dpi=150, bbox_inches='tight')
        plt.close()

    @pytest.mark.parametrize("show", [True])
    def test_visualize_concave_perpendicular(self, show):
        """Visualize horseshoe with opening perpendicular to radial line."""
        np.random.seed(42)

        n_points = 120
        angles = np.random.uniform(-3*np.pi/4, 3*np.pi/4, n_points)
        radii = np.random.uniform(3, 5, n_points)

        points = np.column_stack([
            radii * np.cos(angles) + 6,
            radii * np.sin(angles) + 4
        ])
        points += np.random.randn(n_points, 2) * 0.2

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_envelope(envelope, points, ax=ax, title="Concave Perpendicular (Sideways Horseshoe)")

        if show:
            plt.show()
        else:
            plt.savefig("edge_case_concave_perpendicular.png", dpi=150, bbox_inches='tight')
        plt.close()

    @pytest.mark.parametrize("show", [True])
    def test_visualize_no_data_near_origin(self, show):
        """Visualize distant cluster with origin inclusion."""
        np.random.seed(42)

        points = np.random.randn(100, 2) * 2 + [10, 10]

        envelope = fit_envelope(points, coverage=0.95, include_origin=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_envelope(envelope, points, ax=ax, title="No Data Near Origin (Distant Cluster)")

        if show:
            plt.show()
        else:
            plt.savefig("edge_case_no_data_near_origin.png", dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
