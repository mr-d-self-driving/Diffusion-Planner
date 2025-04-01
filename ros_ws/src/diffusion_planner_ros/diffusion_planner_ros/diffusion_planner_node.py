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
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
import tf2_ros
from geometry_msgs.msg import TransformStamped
from .lanelet2_utils.lanelet_converter import convert_lanelet, process_segment
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
import json
import torch
from diffusion_planner.utils.config import Config
import time
import numpy as np
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")

        vector_map_path = self.declare_parameter("vector_map_path", value="None").value
        self.get_logger().info(f"Vector map path: {vector_map_path}")
        self.static_map = convert_lanelet(vector_map_path)

        config_json_path = self.declare_parameter(
            "config_json_path", value="None"
        ).value
        self.get_logger().info(f"Config JSON: {config_json_path}")
        with open(config_json_path, "r") as f:
            config_json = json.load(f)
        self.get_logger().info(f"Config JSON: {config_json}")
        config_obj = Config(config_json_path)
        self.diffusion_planner = Diffusion_Planner(config_obj)
        self.diffusion_planner.eval()
        self.diffusion_planner.cuda()
        self.diffusion_planner.decoder.decoder.training = False
        print(f"{config_obj.state_normalizer=}")

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

        """
        Trajectory.msg
            std_msgs/Header header
            autoware_planning_msgs/TrajectoryPoint[] points
        TrajectoryPoint.msg
            builtin_interfaces/Duration time_from_start
            geometry_msgs/Pose pose
            float32 longitudinal_velocity_mps
            float32 lateral_velocity_mps
            # acceleration_mps2 increases/decreases based on absolute vehicle motion and does not consider vehicle direction (forward/backward)
            float32 acceleration_mps2
            float32 heading_rate_rps
            float32 front_wheel_angle_rad
            float32 rear_wheel_angle_rad
        ```
        """
        self.pub_trajectory = self.create_publisher(
            Trajectory,
            # "/planning/scenario_planning/lane_driving/trajectory",
            "/diffusion_planner/trajectory",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_kinematic_state = None
        self.inv_transform_matrix_4x4 = None
        self.vector_map = None
        self.route = None

        self.get_logger().info("Diffusion Planner Node has been initialized")

    def cb_kinematic_state(self, msg):
        self.latest_kinematic_state = msg
        ego_x = msg.pose.pose.position.x
        ego_y = msg.pose.pose.position.y
        ego_z = msg.pose.pose.position.z
        ego_qx = msg.pose.pose.orientation.x
        ego_qy = msg.pose.pose.orientation.y
        ego_qz = msg.pose.pose.orientation.z
        ego_qw = msg.pose.pose.orientation.w
        rot = Rotation.from_quat([ego_qx, ego_qy, ego_qz, ego_qw])
        translation = np.array([ego_x, ego_y, ego_z])
        transform_matrix = rot.as_matrix()
        transform_matrix_4x4 = np.eye(4)
        transform_matrix_4x4[:3, :3] = transform_matrix
        transform_matrix_4x4[:3, 3] = translation
        inv_transform_matrix_4x4 = np.eye(4)
        inv_transform_matrix_4x4[:3, :3] = transform_matrix.T
        inv_transform_matrix_4x4[:3, 3] = -transform_matrix.T @ translation
        self.inv_transform_matrix_4x4 = inv_transform_matrix_4x4

    def cb_detected_objects(self, msg):
        dev = self.diffusion_planner.parameters().__next__().device
        input_dict = {
            "ego_current_state": torch.zeros((1, 10), device=dev),
            "neighbor_agents_past": torch.zeros((1, 32, 21, 11), device=dev),
            "lanes": torch.zeros((1, 70, 20, 12), device=dev),
            "lanes_speed_limit": torch.zeros((1, 70, 1), device=dev),
            "lanes_has_speed_limit": torch.zeros(
                (1, 70, 1), dtype=torch.bool, device=dev
            ),
            "route_lanes": torch.zeros((1, 25, 20, 12), device=dev),
            "route_lanes_speed_limit": torch.zeros((1, 25, 1), device=dev),
            "route_lanes_has_speed_limit": torch.zeros(
                (1, 25, 1), dtype=torch.bool, device=dev
            ),
            "static_objects": torch.zeros((1, 5, 10), device=dev),
            "sampled_trajectories": torch.zeros((1, 11, 81, 4), device=dev),
            "diffusion_time": torch.zeros((1, 11, 81, 4), device=dev),
        }
        start = time.time()
        out = self.diffusion_planner(input_dict)[1]
        end = time.time()
        elapsed_msec = (end - start) * 1000
        # self.get_logger().info(f"inference time: {elapsed_msec:.4f} msec")
        pred = out["prediction"]
        # print(f"{pred.shape=}")  # ([1, 11, 80, 4])
        pred = pred[0, 0].detach().cpu().numpy().astype(np.float64)  # T, 4
        heading = np.arctan2(pred[:, 3], pred[:, 2])[..., None]
        pred = np.concatenate([pred[..., :2], heading], axis=-1)  # T, 3(x, y, heading)
        # Convert to Trajectory message
        trajectory_msg = Trajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = "map"
        trajectory_msg.points = []
        dt = 0.1
        for i in range(pred.shape[0]):
            point = TrajectoryPoint()
            total_seconds = float(i * dt)
            secs = int(total_seconds)
            nanosec = int((total_seconds - secs) * 1e9)
            point.time_from_start = Duration()
            point.time_from_start.sec = secs
            point.time_from_start.nanosec = nanosec
            point.pose.position.x = pred[i, 0]
            point.pose.position.y = pred[i, 1]
            point.pose.position.z = 0.0
            point.pose.orientation.w = np.cos(pred[i, 2] / 2)
            point.pose.orientation.x = 0.0
            point.pose.orientation.y = 0.0
            point.pose.orientation.z = np.sin(pred[i, 2] / 2)
            trajectory_msg.points.append(point)
        # Publish the trajectory
        self.pub_trajectory.publish(trajectory_msg)

    def cb_vector_map(self, msg):
        self.vector_map = msg
        self.get_logger().info("Received vector map")

    def cb_route(self, msg):
        self.route = msg
        self.get_logger().info(
            f"Received lanelet route. Number of lanelets: {len(msg.segments)}"
        )

        for i in range(len(msg.segments)):
            self.get_logger().info(f"{msg.segments[i]=}")
            ll2_id = msg.segments[i].preferred_primitive.id
            print(f"{ll2_id=}")
            if ll2_id in self.static_map.lane_segments:
                self.get_logger().info(
                    f"Lanelet ID {ll2_id} is in the static map"
                )
                curr_result = process_segment(
                    self.static_map.lane_segments[ll2_id],
                    self.inv_transform_matrix_4x4,
                    mask_range=100
                )
                print(f"{curr_result.shape=}")
            assert ll2_id not in self.static_map.crosswalk_segments
            assert ll2_id not in self.static_map.boundary_segments


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
