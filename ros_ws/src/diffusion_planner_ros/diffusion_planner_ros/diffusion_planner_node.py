#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import Odometry
from autoware_perception_msgs.msg import DetectedObjects
from autoware_map_msgs.msg import LaneletMapBin


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")

        self.kinematic_state_sub = self.create_subscription(
            Odometry,
            "/localization/kinematic_state",
            self.cb_kinematic_state,
            10,
        )
        self.detected_objects_sub = self.create_subscription(
            DetectedObjects,
            "/perception/object_recognition/detection/objects",
            self.cb_detected_objects,
            10,
        )
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.vector_map_sub = self.create_subscription(
            LaneletMapBin,
            "/map/vector_map",
            self.cb_vector_map,
            map_qos,
        )

        self.latest_kinematic_state = None
        self.vector_map = None

        self.get_logger().info("Diffusion Planner Node has been initialized")

    def cb_kinematic_state(self, msg):
        self.latest_kinematic_state = msg

    def cb_detected_objects(self, msg):
        self.get_logger().info(
            f"Received detected objects. Number of objects: {len(msg.objects)}"
        )

    def cb_vector_map(self, msg):
        self.vector_map = msg
        self.get_logger().info("Received vector map")


def main(args=None):
    rclpy.init(args=args)

    planner_node = DiffusionPlannerNode()

    # Use multi-threaded executor to handle multiple callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(planner_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
