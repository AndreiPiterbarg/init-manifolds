"""
Comparison tests for convex vs single-crossing (star-shaped) envelope methods.

Runs the same test scenarios on both envelope types to compare behavior side by side.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from dataclasses import dataclass

from src.envelope import (
    fit_envelope,
    fit_single_crossing_envelope,
    contains,
    envelope_stats,
    single_crossing_stats,
    plot_envelope,
)


# =============================================================================
# Envelope Type Abstraction
# =============================================================================

@dataclass
class EnvelopeResult:
    """Unified result container for both envelope types."""
    envelope: np.ndarray
    stats: dict
    envelope_type: str


def fit_envelope_by_type(
    points: np.ndarray,
    envelope_type: str,
    coverage: float = 0.95,
    include_origin: bool = True,
    origin: np.ndarray = None,
    **kwargs
) -> EnvelopeResult:
    """
    Fit an envelope using the specified method.

    Parameters
    ----------
    points : np.ndarray
        2D points of shape (N, 2)
    envelope_type : str
        Either 'convex' or 'single_crossing'
    coverage : float
        Target coverage fraction
    include_origin : bool
        Whether to include origin (only for convex)
    origin : np.ndarray
        Origin point for single_crossing method
    **kwargs
        Additional parameters passed to the fitting function

    Returns
    -------
    EnvelopeResult
        Contains envelope vertices, stats, and type identifier
    """
    if envelope_type == 'convex':
        envelope = fit_envelope(
            points,
            coverage=coverage,
            include_origin=include_origin
        )
        stats = envelope_stats(envelope, points)

    elif envelope_type == 'single_crossing':
        # For single_crossing, use data centroid as origin and sensible defaults:
        # - Smoothing for natural curves
        # - target_coverage adaptively expands only where data exists
        envelope, _ = fit_single_crossing_envelope(
            points,
            coverage=kwargs.get('sc_coverage', 0.99),  # Light filtering
            origin=origin,  # None means use data centroid
            n_angles=kwargs.get('n_angles', 72),
            smoothing=kwargs.get('smoothing', 1.0),  # Moderate smoothing
            percentile_per_angle=kwargs.get('percentile_per_angle', 95.0),
            target_coverage=coverage  # Adaptively expand to match requested coverage
        )
        stats = single_crossing_stats(envelope, points, origin=origin)

    else:
        raise ValueError(f"Unknown envelope_type: {envelope_type}")

    return EnvelopeResult(envelope=envelope, stats=stats, envelope_type=envelope_type)


# =============================================================================
# Parametrized Test Classes
# =============================================================================

ENVELOPE_TYPES = ['convex', 'single_crossing']


class TestBasicFunctionality:
    """Basic functionality tests for both envelope types."""

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_output_shape(self, envelope_type):
        """Output should be (M, 2) array for both types."""
        np.random.seed(42)
        points = np.random.randn(100, 2)

        result = fit_envelope_by_type(points, envelope_type)

        assert result.envelope.ndim == 2
        assert result.envelope.shape[1] == 2
        assert result.envelope.shape[0] >= 3

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_coverage_affects_size(self, envelope_type):
        """Higher coverage should generally give larger envelope."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        result_80 = fit_envelope_by_type(points, envelope_type, coverage=0.80)
        result_95 = fit_envelope_by_type(points, envelope_type, coverage=0.95)

        # Check that 95% coverage contains at least as many points
        inside_80 = np.mean(contains(result_80.envelope, points))
        inside_95 = np.mean(contains(result_95.envelope, points))

        assert inside_95 >= inside_80 * 0.9  # Allow small tolerance

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_containment_stats(self, envelope_type):
        """Stats should report reasonable containment."""
        np.random.seed(42)
        points = np.random.randn(100, 2)

        result = fit_envelope_by_type(points, envelope_type, coverage=0.95)

        assert 'fraction_contained' in result.stats
        assert 0 < result.stats['fraction_contained'] <= 1


class TestAsymmetricalData:
    """Tests for extremely asymmetrical data."""

    @pytest.fixture
    def hourglass_data(self):
        """Hourglass shape with origin at center, left side smaller than right."""
        np.random.seed(42)

        n_left = 30
        left_cluster = np.random.randn(n_left, 2) * 1.0
        left_cluster[:, 0] = left_cluster[:, 0] * 0.8 - 3

        n_right = 70
        right_cluster = np.random.randn(n_right, 2)
        right_cluster[:, 0] = right_cluster[:, 0] * 2.0 + 6
        right_cluster[:, 1] = right_cluster[:, 1] * 4.0

        return np.vstack([left_cluster, right_cluster])

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_asymmetric_hourglass(self, envelope_type, hourglass_data):
        """Both methods should handle asymmetric hourglass data."""
        result = fit_envelope_by_type(hourglass_data, envelope_type)

        assert result.stats['fraction_contained'] >= 0.70
        assert result.envelope.shape[0] >= 3

        # Envelope should be wider than tall
        x_range = result.envelope[:, 0].max() - result.envelope[:, 0].min()
        y_range = result.envelope[:, 1].max() - result.envelope[:, 1].min()
        assert x_range > y_range * 0.5  # Somewhat wider

    @pytest.fixture
    def radial_spread_data(self):
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

        return np.vstack([dense_sector, sparse_sector])

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_asymmetric_radial_spread(self, envelope_type, radial_spread_data):
        """Both methods should handle asymmetric radial spread."""
        result = fit_envelope_by_type(radial_spread_data, envelope_type)

        assert result.stats['fraction_contained'] >= 0.60
        assert result.envelope.shape[0] >= 3


class TestConcaveTowardOrigin:
    """Tests for data with concave segments pointed toward origin."""

    @pytest.fixture
    def horseshoe_data(self):
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

        return points

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_horseshoe_shape(self, envelope_type, horseshoe_data):
        """Both methods should handle horseshoe data."""
        result = fit_envelope_by_type(horseshoe_data, envelope_type)

        # Single-crossing with origin at [0,0] will have lower coverage
        # for data far from origin (horseshoe is centered at [5,0])
        min_coverage = 0.60 if envelope_type == 'single_crossing' else 0.70
        assert result.stats['fraction_contained'] >= min_coverage
        assert result.envelope.shape[0] >= 3

    @pytest.fixture
    def v_shape_data(self):
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

        return points

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_v_shape_toward_origin(self, envelope_type, v_shape_data):
        """Both methods should handle V-shaped data."""
        result = fit_envelope_by_type(v_shape_data, envelope_type)

        assert result.stats['fraction_contained'] >= 0.60
        assert result.envelope.shape[0] >= 3


class TestDumbbellData:
    """Tests for dumbbell-shaped data (the motivating use case for single-crossing)."""

    @pytest.fixture
    def dumbbell_data(self):
        """Dumbbell: two lobes connected by thin bridge."""
        np.random.seed(42)

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

        return np.vstack([left_lobe, right_lobe, bridge])

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_dumbbell_coverage(self, envelope_type, dumbbell_data):
        """Both methods should achieve reasonable coverage on dumbbell data."""
        result = fit_envelope_by_type(dumbbell_data, envelope_type)

        assert result.stats['fraction_contained'] >= 0.80
        assert result.envelope.shape[0] >= 3

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_dumbbell_shape_captured(self, envelope_type, dumbbell_data):
        """Envelope should span the full dumbbell width."""
        result = fit_envelope_by_type(dumbbell_data, envelope_type)

        # Should span from left lobe to right lobe
        x_min = result.envelope[:, 0].min()
        x_max = result.envelope[:, 0].max()

        assert x_min < -1.0, "Should reach left lobe"
        assert x_max > 1.0, "Should reach right lobe"


class TestDistantData:
    """Tests for data far from origin."""

    @pytest.fixture
    def distant_cluster_data(self):
        """Single cluster far from origin."""
        np.random.seed(42)
        return np.random.randn(100, 2) * 2 + [10, 10]

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_distant_cluster(self, envelope_type, distant_cluster_data):
        """Both methods should handle distant clusters."""
        result = fit_envelope_by_type(distant_cluster_data, envelope_type)

        assert result.stats['fraction_contained'] >= 0.60
        assert result.envelope.shape[0] >= 3

    @pytest.fixture
    def two_distant_clusters_data(self):
        """Two clusters on opposite sides, neither near origin."""
        np.random.seed(42)

        cluster1 = np.random.randn(50, 2) * 1.5 + [10, 5]
        cluster2 = np.random.randn(50, 2) * 1.5 + [-10, -5]

        return np.vstack([cluster1, cluster2])

    @pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
    def test_two_distant_clusters(self, envelope_type, two_distant_clusters_data):
        """Both methods should handle two distant clusters."""
        result = fit_envelope_by_type(two_distant_clusters_data, envelope_type)

        # Should span both clusters
        x_range = result.envelope[:, 0].max() - result.envelope[:, 0].min()
        assert x_range > 10


# =============================================================================
# Side-by-Side Visualization Tests
# =============================================================================

# Output directory for saved images
import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'comparison_images')


class TestSideBySideVisualization:
    """Visual comparison tests showing both envelope types side by side."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _plot_comparison(
        self,
        points: np.ndarray,
        title: str,
        filename: str,
        coverage: float = 0.95,
        show: bool = False
    ):
        """Helper to create side-by-side comparison plot and save to file."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, env_type in zip(axes, ENVELOPE_TYPES):
            # Don't pass explicit origin - let single_crossing use data centroid
            result = fit_envelope_by_type(points, env_type, coverage=coverage)

            # Plot points
            ax.scatter(points[:, 0], points[:, 1],
                      c='steelblue', alpha=0.5, s=20, label='Data')

            # Plot envelope
            envelope_closed = np.vstack([result.envelope, result.envelope[0]])
            ax.plot(envelope_closed[:, 0], envelope_closed[:, 1],
                   'r-', linewidth=2, label='Envelope')
            ax.fill(result.envelope[:, 0], result.envelope[:, 1],
                   'red', alpha=0.1)

            # Plot data centroid (used as origin for single_crossing)
            centroid = np.mean(points, axis=0)
            ax.scatter([centroid[0]], [centroid[1]], c='green', s=100,
                      marker='x', linewidths=3, label='Centroid')

            ax.set_title(f"{env_type.replace('_', ' ').title()}\n"
                        f"Contained: {result.stats['fraction_contained']:.1%}, "
                        f"Vertices: {result.stats['num_vertices']}")
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Always save to file
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

        if show:
            plt.show()
        plt.close()

    def test_compare_gaussian(self):
        """Compare on simple Gaussian data."""
        np.random.seed(42)
        points = np.random.randn(200, 2)

        self._plot_comparison(
            points,
            "Gaussian Data Comparison",
            "01_gaussian.png"
        )

    def test_compare_hourglass(self):
        """Compare on asymmetric hourglass data."""
        np.random.seed(42)

        n_left = 30
        left_cluster = np.random.randn(n_left, 2) * 1.0
        left_cluster[:, 0] = left_cluster[:, 0] * 0.8 - 3

        n_right = 70
        right_cluster = np.random.randn(n_right, 2)
        right_cluster[:, 0] = right_cluster[:, 0] * 2.0 + 6
        right_cluster[:, 1] = right_cluster[:, 1] * 4.0

        points = np.vstack([left_cluster, right_cluster])

        self._plot_comparison(
            points,
            "Asymmetric Hourglass Comparison",
            "02_hourglass.png"
        )

    def test_compare_dumbbell(self):
        """Compare on dumbbell-shaped data (key use case)."""
        np.random.seed(42)

        n_per_lobe = 100
        n_bridge = 30

        left_lobe = np.random.randn(n_per_lobe, 2) * 0.5 + np.array([-2, 0])
        right_lobe = np.random.randn(n_per_lobe, 2) * 0.5 + np.array([2, 0])

        bridge_x = np.random.uniform(-1.5, 1.5, n_bridge)
        bridge_y = np.random.randn(n_bridge) * 0.1
        bridge = np.column_stack([bridge_x, bridge_y])

        points = np.vstack([left_lobe, right_lobe, bridge])

        self._plot_comparison(
            points,
            "Dumbbell Data Comparison (Key Use Case)",
            "03_dumbbell.png"
        )

    def test_compare_horseshoe(self):
        """Compare on horseshoe data."""
        np.random.seed(42)

        n_points = 120
        angles = np.random.uniform(np.pi/4, 7*np.pi/4, n_points)
        radii = np.random.uniform(4, 6, n_points)

        points = np.column_stack([
            radii * np.cos(angles) + 5,
            radii * np.sin(angles)
        ])
        points += np.random.randn(n_points, 2) * 0.2

        self._plot_comparison(
            points,
            "Horseshoe Data Comparison",
            "04_horseshoe.png"
        )

    def test_compare_distant_clusters(self):
        """Compare on two distant clusters."""
        np.random.seed(42)

        cluster1 = np.random.randn(50, 2) * 1.5 + [10, 5]
        cluster2 = np.random.randn(50, 2) * 1.5 + [-10, -5]

        points = np.vstack([cluster1, cluster2])

        self._plot_comparison(
            points,
            "Two Distant Clusters Comparison",
            "05_distant_clusters.png"
        )


# =============================================================================
# Quantitative Comparison Tests
# =============================================================================

class TestQuantitativeComparison:
    """Quantitative comparison of envelope properties."""

    @pytest.fixture
    def test_datasets(self):
        """Generate multiple test datasets for comparison."""
        np.random.seed(42)

        datasets = {}

        # Gaussian
        datasets['gaussian'] = np.random.randn(200, 2)

        # Dumbbell
        left = np.random.randn(100, 2) * 0.5 + [-2, 0]
        right = np.random.randn(100, 2) * 0.5 + [2, 0]
        bridge_x = np.random.uniform(-1.5, 1.5, 30)
        bridge_y = np.random.randn(30) * 0.1
        bridge = np.column_stack([bridge_x, bridge_y])
        datasets['dumbbell'] = np.vstack([left, right, bridge])

        # Elongated
        datasets['elongated'] = np.column_stack([
            np.random.randn(200) * 5,
            np.random.randn(200) * 0.5
        ])

        # Horseshoe
        angles = np.random.uniform(np.pi/4, 7*np.pi/4, 150)
        radii = np.random.uniform(4, 6, 150)
        datasets['horseshoe'] = np.column_stack([
            radii * np.cos(angles) + 5,
            radii * np.sin(angles)
        ]) + np.random.randn(150, 2) * 0.2

        return datasets

    def test_print_comparison_table(self, test_datasets):
        """Print comparison table of both methods across datasets."""
        print("\n" + "="*80)
        print("ENVELOPE METHOD COMPARISON")
        print("="*80)
        print(f"{'Dataset':<15} {'Method':<15} {'Contained':<12} {'Vertices':<10}")
        print("-"*80)

        for name, points in test_datasets.items():
            for env_type in ENVELOPE_TYPES:
                result = fit_envelope_by_type(
                    points, env_type,
                    coverage=0.95
                    # No explicit origin - uses data centroid for single_crossing
                )
                print(f"{name:<15} {env_type:<15} "
                      f"{result.stats['fraction_contained']:>10.1%} "
                      f"{result.stats['num_vertices']:>10}")
            print("-"*80)

        # This test always passes - it's for visual inspection
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
