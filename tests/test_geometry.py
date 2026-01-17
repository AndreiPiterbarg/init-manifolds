"""
Unit tests for core geometry module.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from src.envelope.core.geometry import (
    EPS,
    contains,
    polygon_area,
    ensure_ccw,
    shapely_to_numpy,
    expand_polygon_to_include_origin,
)


class TestPolygonArea:
    """Tests for polygon_area() function."""

    def test_unit_square(self):
        """Unit square should have area 1."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert abs(polygon_area(square) - 1.0) < EPS

    def test_triangle(self):
        """Triangle with base 2 and height 2 should have area 2."""
        triangle = np.array([[0, 0], [2, 0], [1, 2]], dtype=float)
        assert abs(polygon_area(triangle) - 2.0) < EPS

    def test_degenerate_polygon(self):
        """Polygon with < 3 vertices should have area 0."""
        line = np.array([[0, 0], [1, 1]], dtype=float)
        assert polygon_area(line) == 0.0

        point = np.array([[0, 0]], dtype=float)
        assert polygon_area(point) == 0.0

    def test_scaled_square(self):
        """2x2 square should have area 4."""
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        assert abs(polygon_area(square) - 4.0) < EPS

    def test_order_invariant(self):
        """Area should be same regardless of vertex order (CCW vs CW)."""
        ccw_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert abs(polygon_area(ccw_square) - polygon_area(cw_square)) < EPS


class TestEnsureCCW:
    """Tests for ensure_ccw() function."""

    def test_ccw_unchanged(self):
        """CCW polygon should remain unchanged."""
        ccw_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = ensure_ccw(ccw_square)
        np.testing.assert_array_almost_equal(result, ccw_square)

    def test_cw_reversed(self):
        """CW polygon should be reversed to CCW."""
        cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        expected_ccw = cw_square[::-1]
        result = ensure_ccw(cw_square)
        np.testing.assert_array_almost_equal(result, expected_ccw)

    def test_triangle_ccw(self):
        """CCW triangle should remain unchanged."""
        ccw_triangle = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        result = ensure_ccw(ccw_triangle)
        np.testing.assert_array_almost_equal(result, ccw_triangle)


class TestShapelyToNumpy:
    """Tests for shapely_to_numpy() function."""

    def test_simple_polygon(self):
        """Convert simple Shapely polygon to numpy."""
        shapely_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = shapely_to_numpy(shapely_poly)

        assert result.shape == (4, 2)
        assert result.shape[1] == 2

    def test_removes_closing_vertex(self):
        """Should remove the duplicate closing vertex."""
        shapely_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = shapely_to_numpy(shapely_poly)

        # Shapely adds closing vertex, but our function should remove it
        assert len(result) == 4  # Not 5


class TestExpandPolygonToIncludeOrigin:
    """Tests for expand_polygon_to_include_origin() function."""

    def test_already_contains_origin(self):
        """Polygon already containing origin should be unchanged."""
        square = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        origin = Point(0, 0)

        result = expand_polygon_to_include_origin(square, origin)
        assert result.contains(origin) or result.touches(origin)

    def test_expands_to_include_origin(self):
        """Polygon not containing origin should be expanded."""
        # Polygon far from origin
        square = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        origin = Point(0, 0)

        assert not square.contains(origin)

        result = expand_polygon_to_include_origin(square, origin)
        assert result.contains(origin)

    def test_touches_origin_accepted(self):
        """Polygon touching origin should not be expanded."""
        # Triangle with vertex at origin
        triangle = Polygon([(0, 0), (1, 0), (0.5, 1)])
        origin = Point(0, 0)

        result = expand_polygon_to_include_origin(triangle, origin)
        # Result should either contain or touch origin
        assert result.contains(origin) or result.touches(origin)


class TestContains:
    """Tests for contains() function."""

    def test_simple_containment(self):
        """Basic containment test."""
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

        # Point inside
        assert contains(square, np.array([[0, 0]]))[0]

        # Point outside
        assert not contains(square, np.array([[5, 5]]))[0]

    def test_concave_polygon(self):
        """Test containment in concave polygon (L-shape)."""
        # L-shaped polygon
        l_shape = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]
        ], dtype=float)

        # Point in the "leg" of L
        assert contains(l_shape, np.array([[0.5, 0.5]]))[0]

        # Point in the "foot" of L
        assert contains(l_shape, np.array([[0.5, 1.5]]))[0]

        # Point in the "notch" (should be outside)
        assert not contains(l_shape, np.array([[1.5, 1.5]]))[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
