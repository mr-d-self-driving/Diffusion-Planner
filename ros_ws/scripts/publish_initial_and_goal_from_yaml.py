import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yaml_path = args.yaml_path

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    print(f"{data=}")

    rclpy.init(args=sys.argv)
    node = Node("publish_initial_and_goal_from_yaml")
    node.get_logger().info(f'Node "{node.get_name()}" has been started.')
    pub_initialpose = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    pub_goal = node.create_publisher(PoseStamped, "/planning/mission_planning/goal", 10)
    pub_checkpoint = node.create_publisher(PoseStamped, "/planning/mission_planning/checkpoint", 10)
    node.get_logger().info("Publishers created.")

    initialpose = PoseWithCovarianceStamped()
    initialpose.header.frame_id = "map"
    initialpose.header.stamp = node.get_clock().now().to_msg()
    initialpose.pose.pose.position.x = data["initialpose"]["pose"]["pose"]["position"]["x"]
    initialpose.pose.pose.position.y = data["initialpose"]["pose"]["pose"]["position"]["y"]
    initialpose.pose.pose.position.z = data["initialpose"]["pose"]["pose"]["position"]["z"]
    initialpose.pose.pose.orientation.x = data["initialpose"]["pose"]["pose"]["orientation"]["x"]
    initialpose.pose.pose.orientation.y = data["initialpose"]["pose"]["pose"]["orientation"]["y"]
    initialpose.pose.pose.orientation.z = data["initialpose"]["pose"]["pose"]["orientation"]["z"]
    initialpose.pose.pose.orientation.w = data["initialpose"]["pose"]["pose"]["orientation"]["w"]
    for i in range(36):
        initialpose.pose.covariance[i] = data["initialpose"]["pose"]["covariance"][i]
    pub_initialpose.publish(initialpose)
    node.get_logger().info(f"Published initial pose: {initialpose}")
    time.sleep(3)

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.header.stamp = node.get_clock().now().to_msg()
    goal_pose.pose.position.x = data["goal"]["pose"]["pose"]["position"]["x"]
    goal_pose.pose.position.y = data["goal"]["pose"]["pose"]["position"]["y"]
    goal_pose.pose.position.z = data["goal"]["pose"]["pose"]["position"]["z"]
    goal_pose.pose.orientation.x = data["goal"]["pose"]["pose"]["orientation"]["x"]
    goal_pose.pose.orientation.y = data["goal"]["pose"]["pose"]["orientation"]["y"]
    goal_pose.pose.orientation.z = data["goal"]["pose"]["pose"]["orientation"]["z"]
    goal_pose.pose.orientation.w = data["goal"]["pose"]["pose"]["orientation"]["w"]
    pub_goal.publish(goal_pose)
    node.get_logger().info(f"Published goal pose: {goal_pose}")

    if "checkpoint" in data:
        time.sleep(1)
        checkpoint = PoseStamped()
        checkpoint.header.frame_id = "map"
        checkpoint.header.stamp = node.get_clock().now().to_msg()
        checkpoint.pose.position.x = data["checkpoint"]["pose"]["position"]["x"]
        checkpoint.pose.position.y = data["checkpoint"]["pose"]["position"]["y"]
        checkpoint.pose.position.z = data["checkpoint"]["pose"]["position"]["z"]
        checkpoint.pose.orientation.x = data["checkpoint"]["pose"]["orientation"]["x"]
        checkpoint.pose.orientation.y = data["checkpoint"]["pose"]["orientation"]["y"]
        checkpoint.pose.orientation.z = data["checkpoint"]["pose"]["orientation"]["z"]
        checkpoint.pose.orientation.w = data["checkpoint"]["pose"]["orientation"]["w"]
        pub_goal.publish(checkpoint)
        node.get_logger().info(f"Published checkpoint pose: {checkpoint}")
