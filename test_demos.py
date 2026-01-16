from convex_envelope import fit_envelope, envelope_stats, plot_envelope
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def test_convex_envelope():

    n_main = 95
    n_outliers = 5

    main_points = np.random.randn(n_main, 2) * 1.5 + [0.5, 0.5]
    outliers = np.random.randn(n_outliers, 2) * 0.5 + [[10, 10], [-8, 5], [7, -9], [-10, -10], [12, 0]]
    all_points = np.vstack([main_points, outliers])

    envelope = fit_envelope(all_points, coverage=0.95, include_origin=True)

    stats = envelope_stats(envelope, all_points)
    print("Envelope Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    full_hull = ConvexHull(np.vstack([all_points, [[0, 0]]]))
    full_poly = np.vstack([all_points, [[0, 0]]])[full_hull.vertices]

    plot_envelope(full_poly, all_points, ax=axes[0], title="Full Convex Hull (100%)")
    plot_envelope(envelope, all_points, ax=axes[1], title="Robust Envelope (95%)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_convex_envelope()