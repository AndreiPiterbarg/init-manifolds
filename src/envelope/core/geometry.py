"""
Core geometry operations for polygon manipulation.

Contains utility functions for:
- Point-in-polygon testing
- Polygon area calculation
- Vertex ordering (CCW)
- Shapely/numpy conversions
- Polygon expansion
"""

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon


# Numerical tolerance for floating point comparisons
EPS = 1e-10


def contains(poly: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Test if points are inside or on a polygon (convex or concave).

    Uses Shapely for robust point-in-polygon testing that works with
    both convex and concave polygons.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).
    points : np.ndarray
        Points to test of shape (N, 2) or (2,).

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) indicating containment.
    """
    points = np.atleast_2d(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    n_vertices = len(poly)
    n_points = len(points)

    if n_vertices < 3:
        return np.zeros(n_points, dtype=bool)

    # Create Shapely polygon for robust containment test
    shapely_poly = Polygon(poly)

    if not shapely_poly.is_valid:
        # Try to fix invalid polygon
        shapely_poly = shapely_poly.buffer(0)

    # Test each point
    inside = np.zeros(n_points, dtype=bool)
    for i, pt in enumerate(points):
        shapely_point = Point(pt)
        # contains() checks interior, touches() checks boundary
        inside[i] = shapely_poly.contains(shapely_point) or shapely_poly.touches(shapely_point)

    return inside


def polygon_area(poly: np.ndarray) -> float:
    """
    Compute the area of a polygon using the shoelace formula.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).

    Returns
    -------
    float
        Area of the polygon.
    """
    n = len(poly)
    if n < 3:
        return 0.0

    # Shoelace formula
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Ensure polygon vertices are in counter-clockwise order.

    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices of shape (M, 2).

    Returns
    -------
    np.ndarray
        Polygon vertices in CCW order.
    """
    # Compute signed area
    n = len(poly)
    x = poly[:, 0]
    y = poly[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    if signed_area < 0:
        # Clockwise, reverse to make CCW
        return poly[::-1].copy()
    return poly


def shapely_to_numpy(geom) -> np.ndarray:
    """
    Convert a Shapely polygon to numpy array of vertices.

    Parameters
    ----------
    geom : Polygon or MultiPolygon
        Shapely geometry object.

    Returns
    -------
    np.ndarray
        Polygon vertices of shape (M, 2).
    """
    if isinstance(geom, MultiPolygon):
        # Take the largest polygon if we got multiple
        geom = max(geom.geoms, key=lambda g: g.area)

    coords = np.array(geom.exterior.coords)
    # Remove the closing duplicate vertex that Shapely adds
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def expand_polygon_to_include_origin(
    shapely_poly: Polygon,
    origin: Point,
    max_iterations: int = 50
) -> Polygon:
    """
    Expand a polygon minimally until it contains the origin.

    Uses iterative buffering to grow the polygon toward the origin.

    Parameters
    ----------
    shapely_poly : Polygon
        Shapely polygon to expand.
    origin : Point
        Point to include (typically origin).
    max_iterations : int
        Maximum expansion iterations.

    Returns
    -------
    Polygon
        Expanded polygon containing the origin.
    """
    if shapely_poly.contains(origin) or shapely_poly.touches(origin):
        return shapely_poly

    # Calculate initial buffer distance based on distance to origin
    dist_to_origin = shapely_poly.exterior.distance(origin)
    buffer_step = max(dist_to_origin * 0.1, 0.01)

    expanded = shapely_poly
    for _ in range(max_iterations):
        expanded = expanded.buffer(buffer_step)
        if expanded.contains(origin):
            return expanded
        buffer_step *= 1.5  # Increase step if not yet containing origin

    return expanded


# Private aliases for backward compatibility with internal usage
_polygon_area = polygon_area
_ensure_ccw = ensure_ccw
_shapely_to_numpy = shapely_to_numpy
_expand_polygon_to_include_origin = expand_polygon_to_include_origin
