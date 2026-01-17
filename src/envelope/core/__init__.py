"""
Core geometry operations.
"""

from .geometry import (
    EPS,
    contains,
    polygon_area,
    ensure_ccw,
    shapely_to_numpy,
    expand_polygon_to_include_origin,
    # Private aliases for backward compatibility
    _polygon_area,
    _ensure_ccw,
    _shapely_to_numpy,
    _expand_polygon_to_include_origin,
)

__all__ = [
    'EPS',
    'contains',
    'polygon_area',
    'ensure_ccw',
    'shapely_to_numpy',
    'expand_polygon_to_include_origin',
    '_polygon_area',
    '_ensure_ccw',
    '_shapely_to_numpy',
    '_expand_polygon_to_include_origin',
]
