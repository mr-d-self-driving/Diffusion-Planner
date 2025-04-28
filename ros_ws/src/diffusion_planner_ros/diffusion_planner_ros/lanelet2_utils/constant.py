from __future__ import annotations

from .agents_base import AgentType
from .map import MapType
from .uuid import uuid

MAP_TYPE_MAPPING: dict[str, MapType] = {
    "road": MapType.ROADWAY,
    "highway": MapType.ROADWAY,
    "road_shoulder": MapType.ROADWAY,
    "bicycle_lane": MapType.BIKE_LANE,
    "dashed": MapType.DASHED,
    "solid": MapType.SOLID,
    "dashed_dashed": MapType.DOUBLE_DASH,
    "virtual": MapType.UNKNOWN,
    "road_border": MapType.SOLID,
    "crosswalk": MapType.CROSSWALK,
    "unknown": MapType.UNKNOWN,
}

MAP_TYPE_COLORS: dict[MapType, list[float]] = {
    MapType.ROADWAY: [0.1, 0.9, 0.0, 0.99],
    MapType.BIKE_LANE: [0.0, 0.3, 0.0, 0.99],
    MapType.DASHED: [0.9, 0.9, 0.0, 0.99],
    MapType.SOLID: [0.0, 0.9, 0.9, 0.99],
    MapType.SOLID_DASH: [0.6, 0.6, 0.1, 0.99],
    MapType.DOUBLE_DASH: [0.0, 0.0, 0.3, 0.99],
    MapType.UNKNOWN: [0.99, 0.0, 0.0, 0.99],
    MapType.CROSSWALK: [0.0, 0.0, 0.99, 0.99],
    MapType.BUS_LANE: [0.4, 0.4, 0.4, 0.99],
}

T4_LANE: tuple[str, ...] = ("road", "highway", "road_shoulder", "bicycle_lane")
T4_ROADLINE: tuple[str, ...] = ("dashed", "solid", "dashed_dashed", "virtual")
T4_ROADEDGE: tuple[str, ...] = ("road_border",)

EGO_ID = uuid("AV")
