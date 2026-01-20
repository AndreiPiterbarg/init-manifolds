"""
Comparison tests for concave vs single-crossing envelope methods.
Generates side-by-side visualizations in output/comparison_images/.
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.envelope import fit_envelope, envelope_stats
from src.envelope.envelopes.single_crossing import (
    fit_single_crossing_envelope,
    single_crossing_stats,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'comparison_images')


def _plot_comparison(points, title, filename):
    """Create side-by-side comparison plot."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Concave envelope
    ax = axes[0]
    envelope = fit_envelope(points, coverage=0.95)
    stats = envelope_stats(envelope, points)
    envelope_closed = np.vstack([envelope, envelope[0]])
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', alpha=0.5, s=20)
    ax.plot(envelope_closed[:, 0], envelope_closed[:, 1], 'r-', linewidth=2)
    ax.fill(envelope[:, 0], envelope[:, 1], 'red', alpha=0.1)
    ax.set_title(f"Concave\nContained: {stats['fraction_contained']:.1%}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Single-crossing envelope
    ax = axes[1]
    envelope, _ = fit_single_crossing_envelope(points)
    stats = single_crossing_stats(envelope, points)
    envelope_closed = np.vstack([envelope, envelope[0]])
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', alpha=0.5, s=20)
    ax.plot(envelope_closed[:, 0], envelope_closed[:, 1], 'r-', linewidth=2)
    ax.fill(envelope[:, 0], envelope[:, 1], 'red', alpha=0.1)
    ax.scatter([0], [0], c='green', s=100, marker='x', linewidths=3, label='Origin')
    ax.set_title(f"Single-Crossing\nContained: {stats['fraction_contained']:.1%}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


class TestVisualization:
    """Visual comparison tests - generates comparison images."""

    def test_gaussian(self):
        np.random.seed(42)
        _plot_comparison(np.random.randn(2000, 2), "Gaussian Data", "01_gaussian.png")

    def test_hourglass(self):
        np.random.seed(42)
        left = np.random.randn(300, 2) * [0.8, 1.0] + [-3, 0]
        right = np.random.randn(700, 2) * [2.0, 4.0] + [6, 0]
        _plot_comparison(np.vstack([left, right]), "Hourglass", "02_hourglass.png")

    def test_dumbbell(self):
        np.random.seed(42)
        left = np.random.randn(1000, 2) * 0.5 + [-2, 0]
        right = np.random.randn(1000, 2) * 0.5 + [2, 0]
        bridge = np.column_stack([np.random.uniform(-1.5, 1.5, 300), np.random.randn(300) * 0.1])
        _plot_comparison(np.vstack([left, right, bridge]), "Dumbbell", "03_dumbbell.png")

    def test_horseshoe(self):
        np.random.seed(42)
        angles = np.random.uniform(np.pi/4, 7*np.pi/4, 1200)
        radii = np.random.uniform(4, 6, 1200)
        points = np.column_stack([radii * np.cos(angles) + 5, radii * np.sin(angles)])
        _plot_comparison(points + np.random.randn(1200, 2) * 0.2, "Horseshoe", "04_horseshoe.png")

    def test_distant_clusters(self):
        np.random.seed(42)
        c1 = np.random.randn(500, 2) * 1.5 + [10, 5]
        c2 = np.random.randn(500, 2) * 1.5 + [-10, -5]
        _plot_comparison(np.vstack([c1, c2]), "Distant Clusters", "05_distant_clusters.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
