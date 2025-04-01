from __future__ import annotations

import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

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
    for lanelet in lanelet_map.laneletLayer:
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


def resample_waypoints(waypoints: NDArray, num_points: int) -> NDArray:
    """
    n点のウェイポイントを指定したm点に均等に変換する関数

    Parameters:
    -----------
    waypoints : NDArray
        n点の元のウェイポイント。形状は (n, dim) で、dimは座標の次元数（通常は2または3）
    num_points : int
        出力するウェイポイントの数 m

    Returns:
    --------
    NDArray
        m点の新しいウェイポイント。形状は (m, dim)
    """
    # 入力チェック
    if num_points < 2:
        raise ValueError("出力ポイント数は2以上である必要があります")

    n_points = len(waypoints)
    if n_points < 2:
        raise ValueError("入力ウェイポイントは少なくとも2点必要です")

    # 既に希望の点数である場合はそのまま返す
    if n_points == num_points:
        return waypoints

    # 累積距離（弧長）の計算
    distances = np.zeros(n_points)
    for i in range(1, n_points):
        distances[i] = distances[i - 1] + np.linalg.norm(
            waypoints[i] - waypoints[i - 1]
        )

    # 総距離が0の場合（全点が同じ位置の場合）
    if np.isclose(distances[-1], 0):
        return np.tile(waypoints[0], (num_points, 1))

    # 新しい弧長を均等に生成
    new_distances = np.linspace(0, distances[-1], num_points)

    # 各次元ごとに補間
    dim = waypoints.shape[1]
    new_waypoints = np.zeros((num_points, dim))

    for d in range(dim):
        interp_func = interp1d(distances, waypoints[:, d], kind="linear")
        new_waypoints[:, d] = interp_func(new_distances)

    # 最初と最後のポイントを確実に正確に保持
    new_waypoints[0] = waypoints[0]
    new_waypoints[-1] = waypoints[-1]

    return new_waypoints


def process_segment(segment, inv_transform_matrix_4x4, mask_range):
    centerlines = segment.polyline.waypoints
    left_boundaries = segment.left_boundaries[0].polyline.waypoints
    right_boundaries = segment.right_boundaries[0].polyline.waypoints
    n = centerlines.shape[0]
    if left_boundaries.shape[0] != n:
        left_boundaries = resample_waypoints(left_boundaries, n)
    if right_boundaries.shape[0] != n:
        right_boundaries = resample_waypoints(right_boundaries, n)

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

    # x, yがegoからmask_range内のものだけを抽出
    mask = (
        (centerlines[:, 0] > -mask_range)
        & (centerlines[:, 0] < mask_range)
        & (centerlines[:, 1] > -mask_range)
        & (centerlines[:, 1] < mask_range)
    )
    filtered_centerlines = centerlines[mask]
    left_boundaries = left_boundaries[mask]
    right_boundaries = right_boundaries[mask]

    # 点数が20になるように修正する
    n = filtered_centerlines.shape[0]
    if n < 2:
        return None
    filtered_centerlines = resample_waypoints(filtered_centerlines, 20)
    left_boundaries = resample_waypoints(left_boundaries, 20)
    right_boundaries = resample_waypoints(right_boundaries, 20)

    diff_centerlines = filtered_centerlines[1:] - filtered_centerlines[:-1]
    diff_centerlines = np.insert(diff_centerlines, diff_centerlines.shape[0], 0, axis=0)

    curr_data = np.concatenate(
        (
            filtered_centerlines[:, 0:2],  # xyのみ
            diff_centerlines[:, 0:2],  # xyのみ
            left_boundaries[:, 0:2],  # xyのみ
            right_boundaries[:, 0:2],  # xyのみ
        ),
        axis=1,
    )
    return curr_data


def get_input_feature(
    map: AWMLStaticMap,
    ego_x: float,
    ego_y: float,
    ego_z: float,
    ego_qx: float,
    ego_qy: float,
    ego_qz: float,
    ego_qw: float,
    mask_range: float,
) -> list[np.ndarray]:
    # 自車中心に座標変換するための行列を作成
    rot = Rotation.from_quat([ego_qx, ego_qy, ego_qz, ego_qw])
    translation = np.array([ego_x, ego_y, ego_z])
    transform_matrix = rot.as_matrix()
    transform_matrix_4x4 = np.eye(4)
    transform_matrix_4x4[:3, :3] = transform_matrix
    transform_matrix_4x4[:3, 3] = translation
    inv_transform_matrix_4x4 = np.eye(4)
    inv_transform_matrix_4x4[:3, :3] = transform_matrix.T
    inv_transform_matrix_4x4[:3, 3] = -transform_matrix.T @ translation

    # Plot the map
    result = []
    for segment_id, segment in map.lane_segments.items():
        curr_data = process_segment(segment, inv_transform_matrix_4x4, mask_range)
        if curr_data is None:
            continue
        result.append(curr_data)

    # 先頭の距離でソート
    result = sorted(result, key=lambda x: np.linalg.norm(x[0, :2]))
    result = result[0:70]

    return result
