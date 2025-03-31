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
from autoware_planning_msgs.msg import LaneletRoute
import tf2_ros
from geometry_msgs.msg import TransformStamped


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")

        self.kinematic_state_sub = self.create_subscription(
            Odometry,
            "/localization/kinematic_state",
            self.cb_kinematic_state,
            10,
        )
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_perception_msgs/msg/DetectedObjects.msg
        """
        DetectedObjects.msg
            std_msgs/Header header
            DetectedObject[] objects
        DetectedObject.msg
            float32 existence_probability
            ObjectClassification[] classification
            DetectedObjectKinematics kinematics
            Shape shape
        DetectedObjectKinematics.msg
            # Only position is available, orientation is empty. Note that the shape can be an oriented
            # bounding box but the direction the object is facing is unknown, in which case
            # orientation should be empty.
            uint8 UNAVAILABLE=0

            # The orientation is determined only up to a sign flip. For instance, assume that cars are
            # longer than they are wide, and the perception pipeline can accurately estimate the
            # dimensions of a car. It should set the orientation to coincide with the major axis, with
            # the sign chosen arbitrarily, and use this tag to signify that the orientation could
            # point to the front or the back.
            uint8 SIGN_UNKNOWN=1

            # The full orientation is available. Use e.g. for machine-learning models that can
            # differentiate between the front and back of a vehicle.
            uint8 AVAILABLE=2

            geometry_msgs/PoseWithCovariance pose_with_covariance

            bool has_position_covariance
            uint8 orientation_availability

            geometry_msgs/TwistWithCovariance twist_with_covariance

            bool has_twist
            bool has_twist_covariance        
        """
        self.detected_objects_sub = self.create_subscription(
            DetectedObjects,
            "/perception/object_recognition/detection/objects",
            self.cb_detected_objects,
            10,
        )
        transient_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_map_msgs/msg/LaneletMapBin.msg
        """
        LaneletMapBin.msg
            # Lanelet map message
            # This message contains the binary data of a Lanelet map.
            # Also contains the map name, version and format.

            # Header with timestamp when the message is published
            # And frame of the Lanelet Map origin (probably just "map")
            std_msgs/Header header

            # Version of the map format (optional)
            # Example: "1.1.1"
            string version_map_format

            # Version of the map (encouraged, optional)
            # Example: "1.0.0"
            string version_map

            # Name of the map (encouraged, optional)
            # Example: "florence-prato-city-center"
            string name_map

            # Binary map data
            uint8[] data
        """
        self.vector_map_sub = self.create_subscription(
            LaneletMapBin,
            "/map/vector_map",
            self.cb_vector_map,
            transient_qos,
        )
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_planning_msgs/msg/LaneletRoute.msg
        """
        LaneletRoute.msg
            std_msgs/Header header
            geometry_msgs/Pose start_pose
            geometry_msgs/Pose goal_pose
            autoware_planning_msgs/LaneletSegment[] segments
            unique_identifier_msgs/UUID uuid
            bool allow_modification
        """
        self.route_sub = self.create_subscription(
            LaneletRoute,
            "/planning/mission_planning/route",
            self.cb_route,
            transient_qos,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_kinematic_state = None
        self.vector_map = None
        self.route = None

        self.get_logger().info("Diffusion Planner Node has been initialized")

    def cb_kinematic_state(self, msg):
        self.latest_kinematic_state = msg

    def cb_detected_objects(self, msg):
        return
        self.get_logger().info(
            f"Received detected objects. Number of objects: {len(msg.objects)}"
        )

    def cb_vector_map(self, msg):
        self.vector_map = msg
        self.get_logger().info("Received vector map")

    def cb_route(self, msg):
        self.route = msg
        self.get_logger().info(
            f"Received lanelet route. Number of lanelets: {len(msg.segments)}"
        )

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
