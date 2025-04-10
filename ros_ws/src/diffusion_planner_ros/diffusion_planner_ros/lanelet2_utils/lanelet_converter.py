from __future__ import annotations

import sys

import numpy as np
from scipy.interpolate import interp1d

try:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector
    from lanelet2.routing import RoutingGraph
    from lanelet2.traffic_rules import Locations, Participants
    from lanelet2.traffic_rules import create as create_traffic_rules
except ImportError as e:
    print(e)  # noqa: T201
    sys.exit(1)

from .polylines_base import BoundaryType
from numpy.typing import NDArray
from .uuid import uuid
from .static_map import (
    AWMLStaticMap,
    BoundarySegment,
    CrosswalkSegment,
    LaneSegment,
    Polyline,
)
from .map import MapType
from .constant import MAP_TYPE_MAPPING, T4_LANE, T4_ROADEDGE, T4_ROADLINE

from shapely import LineString


def _interpolate_points(line, num_point):
    # line (n, 3)
    n = line.shape[0]
    if n == num_point:
        return line
    index = np.linspace(0, n - 1, num_point)
    line = line[index.astype(int)]
    return line


# cspell: ignore MGRS


def _load_osm(filename: str) -> lanelet2.core.LaneletMap:
    """Load lanelet map from osm file.

    Args:
    ----
        filename (str): Path to osm file.

    Returns:
    -------
        lanelet2.core.LaneletMap: Loaded lanelet map.

    """
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    return lanelet2.io.load(filename, projection)


def _get_lanelet_subtype(lanelet: lanelet2.core.Lanelet) -> str:
    """Return subtype name from lanelet.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        str: Subtype name. Return "" if it has no attribute named subtype.

    """
    if "subtype" in lanelet.attributes:
        return lanelet.attributes["subtype"]
    else:
        return ""


def _get_linestring_type(linestring: lanelet2.core.LineString3d) -> str:
    """Return type name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Type name. Return "" if it has no attribute named type.

    """
    if "type" in linestring.attributes:
        return linestring.attributes["type"]
    else:
        return ""


def _get_linestring_subtype(linestring: lanelet2.core.LineString3d) -> str:
    """Return subtype name from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): Linestring instance.

    Returns:
    -------
        str: Subtype name. Return "" if it has no attribute named subtype.

    """
    if "subtype" in linestring.attributes:
        return linestring.attributes["subtype"]
    else:
        return ""


def _is_virtual_linestring(line_type: str, line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is virtual.

    Args:
    ----
        line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is `virtual` and subtype is `""`.

    """
    return line_type == "virtual" and line_subtype == ""


def _is_roadedge_linestring(line_type: str, _line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is supported RoadEdge.

    Args:
    ----
        line_type (str): Line type name.
        _line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line type is contained in T4_ROADEDGE.

    Note:
    ----
        Currently `_line_subtype` is not used, but it might be used in the future.

    """
    return line_type in T4_ROADEDGE


def _is_roadline_linestring(_line_type: str, line_subtype: str) -> bool:
    """Indicate whether input linestring type and subtype is supported RoadLine.

    Args:
    ----
        _line_type (str): Line type name.
        line_subtype (str): Line subtype name.

    Returns:
    -------
        bool: Return True if line subtype is contained in T4_RoadLine.

    Note:
    ----
        Currently `_line_type` is not used, but it might be used in the future.

    """
    return line_subtype in T4_ROADLINE


def _get_boundary_type(linestring: lanelet2.core.LineString3d) -> BoundaryType:
    """Return the `BoundaryType` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundaryType: BoundaryType instance.

    """
    line_type = _get_linestring_type(linestring)
    line_subtype = _get_linestring_subtype(linestring)
    if _is_virtual_linestring(line_type, line_subtype):
        return MapType.UNKNOWN
    elif _is_roadedge_linestring(line_type, line_subtype):
        return MAP_TYPE_MAPPING[line_type]
    elif _is_roadline_linestring(line_type, line_subtype):
        return MAP_TYPE_MAPPING[line_subtype]
    else:
        # logging.warning(
        #     f"[Boundary]: id={linestring.id}, type={line_type}, subtype={line_subtype}, MapType.UNKNOWN is used.",
        # )
        return MapType.UNKNOWN


def _get_boundary_segment(linestring: lanelet2.core.LineString3d) -> BoundarySegment:
    """Return the `BoundarySegment` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundarySegment: BoundarySegment instance.

    """
    boundary_type = _get_boundary_type(linestring)
    waypoints = _interpolate_lane(
        np.array([(line.x, line.y, line.z) for line in linestring])
    )
    polyline = Polyline(polyline_type=boundary_type, waypoints=waypoints)
    return BoundarySegment(linestring.id, polyline)


def _get_speed_limit_mph(lanelet: lanelet2.core.Lanelet) -> float | None:
    """Return the lane speed limit in miles per hour (mph).

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        float | None: If the lane has the speed limit return float, otherwise None.

    """
    kph2mph = 0.621371
    if "speed_limit" in lanelet.attributes:
        # NOTE: attributes of ["speed_limit"] is str
        return float(lanelet.attributes["speed_limit"]) * kph2mph
    else:
        return None


def _get_left_and_right_linestring(
    lanelet: lanelet2.core.Lanelet,
) -> tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]:
    """Return the left and right boundaries from lanelet.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        tuple[lanelet2.core.LineString3d, lanelet2.core.LineString3d]: Left and right boundaries.

    """
    return lanelet.leftBound, lanelet.rightBound


def _is_intersection(lanelet: lanelet2.core.Lanelet) -> bool:
    """Check whether specified lanelet is intersection.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        bool: Return `True` if the lanelet has an attribute named `turn_direction`.

    """
    return "turn_direction" in lanelet.attributes


def _get_left_and_right_neighbor_ids(
    lanelet: lanelet2.core.Lanelet,
    routing_graph: RoutingGraph,
) -> tuple[list[int], list[int]]:
    """Return whether the lanelet has left and right neighbors.

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.
        routing_graph (RoutingGraph): RoutingGraph instance.

    Returns:
    -------
        tuple[list[int], list[int]]: Whether the lanelet has (left, right) neighbors.

    """
    left_lanelet = routing_graph.left(lanelet)
    right_lanelet = routing_graph.right(lanelet)
    left_neighbor_id = [left_lanelet.id] if left_lanelet is not None else []
    right_neighbor_id = [right_lanelet.id] if right_lanelet is not None else []
    return left_neighbor_id, right_neighbor_id


def _interpolate_lane(waypoints: NDArray):
    # Compute cumulative distances (arc length)
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        distances[i] = distances[i - 1] + np.linalg.norm(
            waypoints[i] - waypoints[i - 1]
        )

    # Generate new arc lengths with fixed spacing (0.5 meters)
    new_distances = np.arange(0, distances[-1], 0.5)
    new_distances = np.append(
        new_distances, distances[-1]
    )  # Ensure last point is included

    # Interpolate x, y, z separately
    interp_x = interp1d(distances, waypoints[:, 0], kind="linear")
    interp_y = interp1d(distances, waypoints[:, 1], kind="linear")
    interp_z = interp1d(distances, waypoints[:, 2], kind="linear")

    # Compute new waypoints
    new_waypoints = np.vstack(
        (interp_x(new_distances), interp_y(new_distances), interp_z(new_distances))
    ).T

    # Ensure the first and last points remain unchanged
    # Ensure the first waypoint is exactly the same without duplication
    if not np.allclose(new_waypoints[0], waypoints[0]):
        new_waypoints = np.vstack((waypoints[0], new_waypoints))

    # Ensure the last waypoint is exactly the same without duplication
    if not np.allclose(new_waypoints[-1], waypoints[-1]):
        new_waypoints = np.vstack((new_waypoints, waypoints[-1]))
    return new_waypoints


def convert_lanelet(filename: str) -> AWMLStaticMap:
    """Convert lanelet (.osm) to map info.

    Note:
    ----
        Currently, following subtypes are skipped:
            walkway

    Args:
    ----
        filename (str): Path to osm file.

    Returns:
    -------
        AWMLStaticMap: Static map data.

    """
    lanelet_map = _load_osm(filename)

    traffic_rules = create_traffic_rules(Locations.Germany, Participants.Vehicle)
    routing_graph = RoutingGraph(lanelet_map, traffic_rules)

    lane_segments: dict[int, LaneSegment] = {}
    crosswalk_segments: dict[int, CrosswalkSegment] = {}
    taken_boundary_ids: list[int] = []
    traffic_lights = []
    for regulatory_element in lanelet_map.regulatoryElementLayer:
        subtype = regulatory_element.attributes["subtype"]
        if subtype == "traffic_light":
            traffic_lights.append(regulatory_element)
    for lanelet in lanelet_map.laneletLayer:
        if lanelet.trafficLights() is not None:
            # TODO check traffic lights
            pass
        lanelet_subtype = _get_lanelet_subtype(lanelet)

        # NOTE: skip walkway because it contains stop_line as boundary
        if lanelet_subtype in T4_LANE:
            # lane
            lane_type = MAP_TYPE_MAPPING[lanelet_subtype]
            lane_waypoints = _interpolate_lane(
                np.array([(line.x, line.y, line.z) for line in lanelet.centerline])
            )
            lane_polyline = Polyline(polyline_type=lane_type, waypoints=lane_waypoints)
            is_intersection = _is_intersection(lanelet)
            left_neighbor_ids, right_neighbor_ids = _get_left_and_right_neighbor_ids(
                lanelet, routing_graph
            )
            speed_limit_mph = _get_speed_limit_mph(lanelet)

            # road line or road edge
            left_linestring, right_linestring = _get_left_and_right_linestring(lanelet)
            left_boundary = _get_boundary_segment(left_linestring)
            right_boundary = _get_boundary_segment(right_linestring)
            taken_boundary_ids.extend((left_linestring.id, right_linestring.id))

            lane_segments[lanelet.id] = LaneSegment(
                id=lanelet.id,
                polyline=lane_polyline,
                is_intersection=is_intersection,
                left_boundaries=[left_boundary],
                right_boundaries=[right_boundary],
                left_neighbor_ids=left_neighbor_ids,
                right_neighbor_ids=right_neighbor_ids,
                speed_limit_mph=speed_limit_mph,
            )
        elif lanelet_subtype == "crosswalk":
            waypoints = _interpolate_lane(
                np.array([(poly.x, poly.y, poly.z) for poly in lanelet.polygon3d()])
            )
            polygon = Polyline(
                polyline_type=MAP_TYPE_MAPPING[lanelet_subtype], waypoints=waypoints
            )
            crosswalk_segments[lanelet.id] = CrosswalkSegment(lanelet.id, polygon)
        else:
            # logging.warning(f"[Lanelet]: {lanelet_subtype} is unsupported and skipped.")
            continue

    boundary_segments: dict[int, BoundarySegment] = {}
    for linestring in lanelet_map.lineStringLayer:
        type_name: str = _get_linestring_type(linestring)
        if (
            type_name in T4_ROADEDGE or type_name in T4_ROADLINE
        ) and linestring.id not in taken_boundary_ids:
            boundary_segments[linestring.id] = _get_boundary_segment(linestring)

    # generate uuid from map filepath
    map_id = uuid(filename, digit=16)
    return AWMLStaticMap(
        map_id,
        lane_segments=lane_segments,
        crosswalk_segments=crosswalk_segments,
        boundary_segments=boundary_segments,
    )


def process_segment(segment, inv_transform_matrix_4x4, center_x, center_y, mask_range):
    centerlines = segment.polyline.waypoints
    left_boundaries = segment.left_boundaries[0].polyline.waypoints
    right_boundaries = segment.right_boundaries[0].polyline.waypoints

    # x, yがegoからmask_range内のものだけを抽出
    mask = (
        (centerlines[:, 0] > center_x - mask_range)
        & (centerlines[:, 0] < center_x + mask_range)
        & (centerlines[:, 1] > center_y - mask_range)
        & (centerlines[:, 1] < center_y + mask_range)
    )
    true_num = np.sum(mask)
    if true_num == 0:
        # この範囲に点がない場合は、何もせずに返す
        return None

    # 自車座標系に変換
    centerlines_4xN = np.vstack((centerlines.T, np.ones(centerlines.shape[0])))
    centerlines_ego = inv_transform_matrix_4x4 @ centerlines_4xN
    centerlines = centerlines_ego[:3, :].T
    left_boundaries_4xN = np.vstack(
        (left_boundaries.T, np.ones(left_boundaries.shape[0]))
    )
    left_boundaries_ego = inv_transform_matrix_4x4 @ left_boundaries_4xN
    left_boundaries = left_boundaries_ego[:3, :].T
    right_boundaries_4xN = np.vstack(
        (right_boundaries.T, np.ones(right_boundaries.shape[0]))
    )
    right_boundaries_ego = inv_transform_matrix_4x4 @ right_boundaries_4xN
    right_boundaries = right_boundaries_ego[:3, :].T

    # 点数が20になるように修正する
    centerlines = _interpolate_points(centerlines, 20)
    left_boundaries = _interpolate_points(left_boundaries, 20)
    right_boundaries = _interpolate_points(right_boundaries, 20)

    left_boundaries -= centerlines
    right_boundaries -= centerlines

    diff_centerlines = centerlines[1:] - centerlines[:-1]
    diff_centerlines = np.insert(diff_centerlines, diff_centerlines.shape[0], 0, axis=0)

    traffic_light = [0, 0, 0, 1]  # (green, yellow, red, unknown)
    traffic_light = np.tile(traffic_light, (centerlines.shape[0], 1))

    curr_data = np.concatenate(
        (
            centerlines[:, 0:2],  # xyのみ
            diff_centerlines[:, 0:2],  # xyのみ
            left_boundaries[:, 0:2],  # xyのみ
            right_boundaries[:, 0:2],  # xyのみ
            traffic_light,
        ),
        axis=1,
    )
    return curr_data


def get_input_feature(
    map: AWMLStaticMap,
    map2bl_mat4x4: NDArray,
    center_x: float,
    center_y: float,
    mask_range: float,
) -> list[np.ndarray]:
    result = []
    for segment_id, segment in map.lane_segments.items():
        curr_data = process_segment(
            segment, map2bl_mat4x4, center_x, center_y, mask_range
        )
        if curr_data is None:
            continue
        result.append(curr_data)

    # 先頭の距離でソート
    result = sorted(result, key=lambda x: np.linalg.norm(x[0, :2]))
    result = result[0:70]

    return result
