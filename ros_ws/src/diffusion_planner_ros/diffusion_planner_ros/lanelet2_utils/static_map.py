from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from attr import define, field
from typing_extensions import Self

from .map import MapType
from .polyline import Polyline

if TYPE_CHECKING:
    from .typing import NDArrayF32

__all__ = ("AWMLStaticMap", "LaneSegment", "BoundarySegment")


@dataclass(frozen=True)
class AWMLStaticMap:
    """Represents a static map information.

    Attributes
    ----------
        id (str): Unique ID associated with this map.
        lane_segments (dict[int, LaneSegment]): Container of lanes stored by its id.
        crosswalk_segments (dict[int, CrosswalkSegment]): Container of crosswalks stored by its id.
        boundary_segments (dict[int, BoundarySegment]): Container of boundaries stored by its id
            except of contained in lanes.

    """

    id: str
    lane_segments: dict[int, LaneSegment]

    def __post_init__(self) -> None:
        assert all(isinstance(item, LaneSegment) for _, item in self.lane_segments.items()), (
            "Expected all items are LaneSegments."
        )

    def get_lane_segments(self) -> list[LaneSegment]:
        """Return all lane segments as a list.

        Returns
        -------
            list[LaneSegment]: List of `LaneSegment`.

        """
        return [seg for _, seg in self.lane_segments.items()]

    def get_boundary_segments(self) -> list[BoundarySegment]:
        """Return all the other boundary segments except of contained lanes as a list.

        Returns
        -------
            list[BoundarySegments]: List of `BoundarySegment`.

        """
        return [seg for _, seg in self.boundary_segments.items()]

    def get_all_polyline(
        self,
        *,
        as_array: bool = False,
        full: bool = False,
        as_3d: bool = True,
    ) -> list[Polyline] | NDArrayF32:
        """Return all segments polyline.

        Args:
        ----
            as_array (bool, optional): Indicates whether to return polyline as `NDArray`. Defaults to False.
            full (bool, optional): This is used only if `as_array=True`.
                Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): This is used only if `as_array=True`.
                If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            list[Polyline] | NDArrayF32: List of `Polyline` instances or `NDArray`.

        """
        all_polyline: list[Polyline | NDArrayF32] = []

        duplicate_boundary_ids = []

        def _append_boundaries(boundaries: list[BoundarySegment]) -> None:
            for bound in boundaries:
                if bound.id in duplicate_boundary_ids:
                    continue
                duplicate_boundary_ids.append(bound.id)
                all_polyline.append(bound.polyline)

        for _, lane in self.lane_segments.items():
            if as_array:
                all_polyline.append(lane.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(lane.polyline)
                _append_boundaries(lane.left_boundaries)
                _append_boundaries(lane.right_boundaries)
        for _, crosswalk in self.crosswalk_segments.items():
            if as_array:
                all_polyline.append(crosswalk.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(crosswalk.polygon)
        for _, boundary in self.boundary_segments.items():
            if as_array:
                all_polyline.append(boundary.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(boundary.polyline)
        return np.concatenate(all_polyline, axis=0, dtype=np.float32) if as_array else all_polyline


def _to_boundary_segment(x: list[dict | BoundarySegment]) -> list[BoundarySegment]:
    return [BoundarySegment.from_dict(v) if isinstance(v, dict) else v for v in x]


@define
class LaneSegment:
    """Represents a lane segment.

    Attributes
    ----------
        id (int): Unique ID associated with this lane.
        polyline (Polyline): `Polyline` instance.
        is_intersection (bool): Flag indicating if this lane is intersection.
        left_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        right_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        speed_limit_mph (float | None, optional): Lane speed limit in [miles/h].

    """

    id: int
    polyline: Polyline = field(
        converter=lambda x: Polyline.from_dict(x) if isinstance(x, dict) else x
    )
    is_intersection: bool
    left_boundaries: list[BoundarySegment] = field(converter=_to_boundary_segment, factory=list)
    right_boundaries: list[BoundarySegment] = field(converter=_to_boundary_segment, factory=list)
    speed_limit_mph: float | None = field(default=None)
    center: NDArrayF32 = field(init=False)
    traffic_lights: list = field(default=None)

    @property
    def lane_type(self) -> MapType:
        """Return the type of the lane.

        Returns
        -------
            MapType: Lane type.

        """
        return self.polyline.polyline_type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `LaneSegment`.

        Returns:
        -------
            LaneSegment: Constructed instance.

        """
        return cls(**data)

    def is_drivable(self) -> bool:
        """Whether the lane is allowed to drive by car like vehicle.

        Returns
        -------
            bool: Return True if the lane is allowed to drive.

        """
        return self.lane_type.is_drivable()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return polyline containing all points on the road segment.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline of the road segment in shape (N, D).

        """
        all_polyline: list[NDArrayF32] = [self.polyline.as_array(full=full, as_3d=as_3d)]
        duplicate_boundary_ids: list[int] = []

        def _append_boundaries(boundaries: list[BoundarySegment]) -> None:
            for bound in boundaries:
                if bound.id in duplicate_boundary_ids:
                    continue
                duplicate_boundary_ids.append(bound.id)
                all_polyline.append(bound.as_array(full=full, as_3d=as_3d))

        _append_boundaries(self.left_boundaries)
        _append_boundaries(self.right_boundaries)

        return np.concatenate(all_polyline, axis=0, dtype=np.float32)


@dataclass
class BoundarySegment:
    """Represents a boundary segment which is RoadLine or RoadEdge.

    Attributes
    ----------
        id (int): Unique ID associated with this boundary.
        boundary_type (BoundaryType): `BoundaryType` instance.
        polyline (Polyline): `Polyline` instance.

    """

    id: int
    polyline: Polyline

    def __post_init__(self) -> None:
        assert isinstance(self.polyline, Polyline), "Expected Polyline."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundarySegment:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `BoundarySegment`.

        Returns:
        -------
            BoundarySegment: Constructed instance.

        """
        return cls(**data)

    def as_dict(self) -> dict:
        """Convert the instance to a dict.

        Returns
        -------
            dict: Converted data.

        """
        return asdict(self)

    def is_crossable(self) -> bool:
        """Indicate whether the boundary is allowed to cross or not.

        Return value depends on the `BoundaryType` definition.

        Returns
        -------
            bool: Return True if the boundary is allowed to cross.

        """
        return self.boundary_type.is_crossable()

    def is_virtual(self) -> bool:
        """Indicate whether the boundary is virtual(or Unknown) or not.

        Returns
        -------
            bool: Return True if the boundary is virtual.

        """
        return self.boundary_type.is_virtual()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline array.

        """
        return self.polyline.as_array(full=full, as_3d=as_3d)
