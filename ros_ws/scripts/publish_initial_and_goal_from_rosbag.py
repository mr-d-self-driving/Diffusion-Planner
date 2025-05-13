import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import rclpy
import rosbag2_py
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


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

    rclpy.init(args=sys.argv)
    node = Node("publish_initial_and_goal_from_rosbag")
    node.get_logger().info(f'Node "{node.get_name()}" has been started.')
    pub_initialpose = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    pub_goal = node.create_publisher(PoseStamped, "/planning/mission_planning/goal", 10)
    node.get_logger().info("Publishers created.")

    initialpose = PoseWithCovarianceStamped()
    initialpose.header = kinematic_state_msgs[0].header
    initialpose.pose = kinematic_state_msgs[0].pose
    pub_initialpose.publish(initialpose)
    node.get_logger().info(f"Published initial pose: {initialpose}")
    time.sleep(3)

    for route_msg in route_msgs:
        goal_pose = PoseStamped()
        goal_pose.header = route_msg.header
        goal_pose.pose = route_msg.goal_pose
        pub_goal.publish(goal_pose)
        node.get_logger().info(f"Published goal pose: {goal_pose}")
