import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rosbag2_py
from autoware_planning_msgs.msg import LaneletRoute
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rosbag_path = args.rosbag_path

    # parse rosbag
    serialization_format = "cdr"
    storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    target_topic_list = [
        "/localization/kinematic_state",
        "/planning/mission_planning/route",
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
            if parse_num >= 1000:
                break

    for key, value in topic_name_to_data.items():
        print(f"{key}: {len(value)} msgs")

    kinematic_state_msgs = topic_name_to_data["/localization/kinematic_state"]
    route_msgs = topic_name_to_data["/planning/mission_planning/route"]

    initialpose = kinematic_state_msgs[0].pose.pose
    x = initialpose.position.x
    y = initialpose.position.y
    z = initialpose.position.z
    qx = initialpose.orientation.x
    qy = initialpose.orientation.y
    qz = initialpose.orientation.z
    qw = initialpose.orientation.w

    print(
        f"""ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{{
header: {{ frame_id: "map" }},
pose: {{
    pose: {{
    position: {{ x: {x}, y: {y}, z: {z} }},
    orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }}
    }},
    covariance: [0.25, 0.0,  0.0, 0.0, 0.0, 0.0,
                0.0,  0.25, 0.0, 0.0, 0.0, 0.0,
                0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                0.0,  0.0,  0.0, 0.0, 0.0, 0.06853891909122467]
}}
}}'
"""
    )

    for route_msg in route_msgs:
        print()
        start_pose = route_msg.start_pose
        print(f"{start_pose=}")
        goal_pose = route_msg.goal_pose
        x = goal_pose.position.x
        y = goal_pose.position.y
        z = goal_pose.position.z
        qx = goal_pose.orientation.x
        qy = goal_pose.orientation.y
        qz = goal_pose.orientation.z
        qw = goal_pose.orientation.w
        print(
            f"""ros2 topic pub -1 /planning/mission_planning/goal geometry_msgs/msg/PoseStamped '{{header: {{stamp: {{sec: 181, nanosec: 289995947}},
frame_id: 'map'}},
pose: {{position: {{x: {x}, y: {y}, z: {z} }},
    orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw} }}
}}
}}'
"""
        )
