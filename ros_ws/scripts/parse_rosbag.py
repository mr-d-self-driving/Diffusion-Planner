import argparse
from pathlib import Path
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from collections import defaultdict
import numpy as np
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
)
from diffusion_planner_ros.utils import (
    create_current_ego_state,
)
import secrets
from dataclasses import dataclass
from autoware_perception_msgs.msg import (
    DetectedObjects,
    TrackedObjects,
    TrafficLightGroupArray,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelWithCovarianceStamped
from sensor_msgs.msg import CompressedImage


@dataclass
class FrameData:
    timestamp: int
    tracked_objects: TrackedObjects
    kinematic_state: Odometry
    acceleration: AccelWithCovarianceStamped
    traffic_signals: TrafficLightGroupArray
    image: CompressedImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def parse_timestamp(stamp) -> int:
    return stamp.sec * int(1e9) + stamp.nanosec


def get_latest_index(list_of_msg, index, target_timestamp):
    """listのうちindexの位置から線形探索をしてtarget_timestampを超えないような最新のメッセージを取得する"""
    for i in range(index, len(list_of_msg)):
        msg = list_of_msg[i]
        stamp = msg.header.stamp if hasattr(msg, "header") else msg.stamp
        timestamp = parse_timestamp(stamp)
        if timestamp > target_timestamp:
            return i - 1
    return len(list_of_msg) - 1


if __name__ == "__main__":
    args = parse_args()
    rosbag_path = args.rosbag_path
    vector_map_path = args.vector_map_path
    limit = args.limit

    vector_map = convert_lanelet(str(vector_map_path))

    serialization_format = "cdr"
    storage_options = rosbag2_py.StorageOptions(
        uri=str(rosbag_path), storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    target_topic_list = [
        "/localization/kinematic_state",
        "/localization/acceleration",
        "/perception/object_recognition/tracking/objects",
        "/perception/traffic_light_recognition/traffic_signals",
        "/planning/mission_planning/route",
        "/sensing/camera/camera0/image_rect_color/compressed",
    ]

    storage_filter = rosbag2_py.StorageFilter(topics=target_topic_list)
    reader.set_filter(storage_filter)

    topic_name_to_data = defaultdict(list)
    parse_num = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic in target_topic_list:
            topic_name_to_data[topic].append(msg)
            parse_num += 1
            if limit > 0 and parse_num >= limit:
                break

    for key, value in topic_name_to_data.items():
        print(f"{key}: {len(value)} msgs")

    # 最初にmsgsの10Hzでの整形(tracked_objects基準)を行う
    n = len(topic_name_to_data["/perception/object_recognition/tracking/objects"])
    data_list = []
    indices = {
        "/localization/kinematic_state": 0,
        "/localization/acceleration": 0,
        "/perception/traffic_light_recognition/traffic_signals": 0,
        "/sensing/camera/camera0/image_rect_color/compressed": 0,
    }
    for i in range(n):
        tracking = topic_name_to_data[
            "/perception/object_recognition/tracking/objects"
        ][i]
        timestamp = parse_timestamp(tracking.header.stamp)
        latest_msgs = {
            "/localization/kinematic_state": None,
            "/localization/acceleration": None,
            "/perception/traffic_light_recognition/traffic_signals": None,
            "/sensing/camera/camera0/image_rect_color/compressed": None,
        }

        for key in latest_msgs.keys():
            curr_index = get_latest_index(
                topic_name_to_data[key], indices[key], timestamp
            )
            curr_msg = topic_name_to_data[key][curr_index]
            latest_msgs[key] = curr_msg
            indices[key] = curr_index

        data_list.append(
            FrameData(
                timestamp=timestamp,
                tracked_objects=tracking,
                kinematic_state=latest_msgs["/localization/kinematic_state"],
                acceleration=latest_msgs["/localization/acceleration"],
                traffic_signals=latest_msgs[
                    "/perception/traffic_light_recognition/traffic_signals"
                ],
                image=latest_msgs[
                    "/sensing/camera/camera0/image_rect_color/compressed"
                ],
            )
        )

    """
    作りたいnpz
    map_name                    <U26    ()
    token                       <U16    ()
    ego_current_state           float32 (10,)
    ego_agent_future            float32 (80, 3)
    neighbor_agents_past        float32 (32, 21, 11)
    neighbor_agents_future      float32 (32, 80, 3)
    static_objects              float32 (5, 10)
    lanes                       float32 (70, 20, 12)
    lanes_speed_limit           float32 (70, 1)
    lanes_has_speed_limit       bool    (70, 1)
    route_lanes                 float32 (25, 20, 12)
    route_lanes_speed_limit     float32 (25, 1)
    route_lanes_has_speed_limit bool    (25, 1)
    """
    PAST_TIME_STEPS = 21
    FUTURE_TIME_STEPS = 80
    NEIGHBOR_NUM = 32
    STATIC_NUM = 5
    LANE_NUM = 70
    LANE_LEN = 20
    ROUTE_NUM = 25
    ROUTE_LEN = 20

    map_name = "autoware_map"

    # これをrosbagのデータから作る
    # 時刻の基準とするデータは "/perception/object_recognition/tracking/objects" (10Hz)
    # 重複が出ないように8秒ごとに作る
    for i in range(PAST_TIME_STEPS, n, FUTURE_TIME_STEPS):
        # 2秒前からここまでのトラッキング（入力用）
        # 2秒前から8秒後までのトラッキング（GT用）

        ego_state = create_current_ego_state(
            data_list[i].kinematic_state, data_list[i].acceleration
        )
