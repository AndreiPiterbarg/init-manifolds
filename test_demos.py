from convex_envelope import fit_envelope, envelope_stats, plot_envelope
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from projected_envelope import fit_projected_envelope, contains_projected, plot_projected_envelope

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

def test_projected_envelope():
    # Create a 3D point cloud - elongated ellipsoid with some outliers
    n_main = 200
    n_outliers = 10

    # Main cluster: stretched along one axis
    main_points = np.random.randn(n_main, 3)
    main_points[:, 0] *= 3.0  # Stretch along x
    main_points[:, 1] *= 2.0  # Medium along y
    main_points[:, 2] *= 0.5  # Thin along z

    # Add some rotation to make it interesting
    theta = np.pi / 6
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    main_points = main_points @ rotation.T

    # Add outliers
    outliers = np.random.randn(n_outliers, 3) * 0.5
    outliers += np.array([[8, 8, 2], [-7, 5, -1], [6, -8, 1],
                          [-8, -8, -2], [10, 0, 0], [0, 10, 1],
                          [-5, -5, 3], [7, 7, -2], [-9, 3, 0], [4, -9, 1]])

    all_points = np.vstack([main_points, outliers])

    # Compute projected envelope
    print("Computing projected envelope...")
    result = fit_projected_envelope(
        all_points,
        coverage=0.95,
        include_origin=True
    )

    # Print results
    print("\n=== Projected Envelope Results ===")
    print(f"Input: {len(all_points)} 3D points")
    print(f"\nPCA Analysis:")
    print(f"  Explained variance: {result.projection.explained_variance_ratio}")
    print(f"  Polar axis: {result.projection.polar_axis}")
    print(f"\nEnvelope Stats:")
    for key, value in result.stats.items():
        if key != 'centroid':
            print(f"  {key}: {value}")

    # Test containment of origin
    origin_inside = contains_projected(result, np.array([[0, 0, 0]]))[0]
    print(f"\nOrigin [0,0,0] inside envelope: {origin_inside}")

    # Plot
    fig = plot_projected_envelope(result, show_3d=True)
    plt.savefig('projected_envelope_demo.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: projected_envelope_demo.png")
    plt.show()

if __name__ == "__main__":
    #test_convex_envelope()
    test_projected_envelope()