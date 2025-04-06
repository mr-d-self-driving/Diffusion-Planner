#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import Odometry
from autoware_perception_msgs.msg import DetectedObjects, TrackedObjects
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, AccelWithCovarianceStamped
from .lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    get_input_feature,
    process_segment,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.visualize_input import visualize_inputs
import json
import torch
from diffusion_planner.utils.config import Config
import time
import numpy as np
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation
from mmengine import fileio
import io
from .utils import create_trajectory_marker, pose_to_mat4x4, rot3x3_to_heading_cos_sin


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")

        # get vector_map
        vector_map_path = self.declare_parameter("vector_map_path", value="None").value
        self.get_logger().info(f"Vector map path: {vector_map_path}")
        self.static_map = convert_lanelet(vector_map_path)

        # get config
        config_json_path = self.declare_parameter(
            "config_json_path", value="None"
        ).value
        self.get_logger().info(f"Config JSON: {config_json_path}")
        with open(config_json_path, "r") as f:
            config_json = json.load(f)
        self.get_logger().info(f"Config JSON: {config_json}")
        self.config_obj = Config(config_json_path)
        self.diffusion_planner = Diffusion_Planner(self.config_obj)
        self.diffusion_planner.eval()
        self.diffusion_planner.cuda()
        self.diffusion_planner.decoder.decoder.training = False
        print(f"{self.config_obj.state_normalizer=}")

        # Load the model checkpoint
        ckpt_path = self.declare_parameter("ckpt_path", value="None").value
        self.get_logger().info(f"Checkpoint path: {ckpt_path}")
        ckpt = fileio.get(ckpt_path)
        with io.BytesIO(ckpt) as f:
            ckpt = torch.load(f)
        state_dict = ckpt["model"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.diffusion_planner.load_state_dict(new_state_dict)

        # get wheel_base
        self.wheel_base = self.declare_parameter("wheel_base", value=5.0).value
        self.get_logger().info(f"Wheel base: {self.wheel_base}")

        # sub(1) kinematic_state
        self.kinematic_state_sub = self.create_subscription(
            Odometry,
            "/localization/kinematic_state",
            self.cb_kinematic_state,
            10,
        )

        # sub(2) acceleration
        self.acceleration_sub = self.create_subscription(
            AccelWithCovarianceStamped,
            "/localization/acceleration",
            self.cb_acceleration,
            10,
        )

        # sub(3) detected_objects, tracked_objects
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_perception_msgs/msg/DetectedObjects.msg
        self.detected_objects_sub = self.create_subscription(
            DetectedObjects,
            "/perception/object_recognition/detection/objects",
            self.cb_detected_objects,
            10,
        )
        self.tracked_objects_sub = self.create_subscription(
            TrackedObjects,
            "/perception/object_recognition/tracking/objects",
            self.cb_tracked_objects,
            10,
        )

        # sub(4) route
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_planning_msgs/msg/LaneletRoute.msg
        transient_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.route_sub = self.create_subscription(
            LaneletRoute,
            "/planning/mission_planning/route",
            self.cb_route,
            transient_qos,
        )

        # pub(1)[main] trajectory
        self.pub_trajectory = self.create_publisher(
            Trajectory,
            "/planning/scenario_planning/lane_driving/trajectory",
            # "/diffusion_planner/trajectory",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        # pub(2)[debug] route_marker
        self.pub_route_marker = self.create_publisher(
            MarkerArray,
            "/diffusion_planner/debug/route_marker",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        # pub(3)[debug] trajectory_marker
        self.pub_trajectory_marker = self.create_publisher(
            MarkerArray,
            "/diffusion_planner/debug/trajectory_marker",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        # members
        self.latest_kinematic_state = None
        self.latest_acceleration = None
        self.bl2map_matrix_4x4 = None
        self.map2bl_matrix_4x4 = None
        self.vector_map = None
        self.route = None
        dev = self.diffusion_planner.parameters().__next__().device
        self.route_tensor = torch.zeros((1, 25, 20, 12), device=dev)
        self.neighbor = torch.zeros((1, 32, 21, 11), device=dev)
        self.tracked_objs = {}  # object_id -> index in self.neighbor

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

        self.bl2map_matrix_4x4 = np.eye(4)
        self.bl2map_matrix_4x4[:3, :3] = transform_matrix
        self.bl2map_matrix_4x4[:3, 3] = translation

        self.map2bl_matrix_4x4 = np.eye(4)
        self.map2bl_matrix_4x4[:3, :3] = transform_matrix.T
        self.map2bl_matrix_4x4[:3, 3] = -transform_matrix.T @ translation

    def cb_acceleration(self, msg):
        self.latest_acceleration = msg

    def cb_detected_objects(self, msg):
        if self.latest_kinematic_state is None:
            return
        if self.route is None:
            return
        dev = self.diffusion_planner.parameters().__next__().device

        start = time.time()
        self.process_route(self.route)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"    route time: {elapsed_msec:.4f} msec")

        start = time.time()
        result_list = get_input_feature(
            self.static_map,
            ego_x=self.latest_kinematic_state.pose.pose.position.x,
            ego_y=self.latest_kinematic_state.pose.pose.position.y,
            ego_z=self.latest_kinematic_state.pose.pose.position.z,
            ego_qx=self.latest_kinematic_state.pose.pose.orientation.x,
            ego_qy=self.latest_kinematic_state.pose.pose.orientation.y,
            ego_qz=self.latest_kinematic_state.pose.pose.orientation.z,
            ego_qw=self.latest_kinematic_state.pose.pose.orientation.w,
            mask_range=100,
        )
        lanes_tensor = torch.zeros((1, 70, 20, 12), dtype=torch.float32, device=dev)
        for i, result in enumerate(result_list):
            lanes_tensor[0, i] = torch.from_numpy(result).cuda()
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"get_input time: {elapsed_msec:.4f} msec")

        # get current velocity
        ego_twist_linear = self.latest_kinematic_state.twist.twist.linear
        ego_twist_angular = self.latest_kinematic_state.twist.twist.angular
        ego_twist_linear = np.array(
            [ego_twist_linear.x, ego_twist_linear.y, ego_twist_linear.z]
        )
        ego_twist_angular = np.array(
            [ego_twist_angular.x, ego_twist_angular.y, ego_twist_angular.z]
        )
        ego_twist_linear = self.map2bl_matrix_4x4[0:3, 0:3] @ ego_twist_linear
        ego_twist_angular = self.map2bl_matrix_4x4[0:3, 0:3] @ ego_twist_angular
        linear_vel_norm = np.linalg.norm(ego_twist_linear)
        if abs(linear_vel_norm) < 0.2:
            yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
            steering_angle = 0.0
        else:
            yaw_rate = ego_twist_angular[2]
            steering_angle = np.arctan(
                yaw_rate * self.wheel_base / abs(linear_vel_norm)
            )
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

        ego_current_state = torch.zeros((1, 10), device=dev)
        ego_current_state[0, 0] = 0  # x in base_link is always 0
        ego_current_state[0, 1] = 0  # y in base_link is always 0
        ego_current_state[0, 2] = 1  # heading cos in base_link is always 1
        ego_current_state[0, 3] = 0  # heading sin in base_link is always 0
        ego_current_state[0, 4] = ego_twist_linear[0]  # velocity x
        ego_current_state[0, 5] = ego_twist_linear[1]  # velocity y
        ego_current_state[0, 6] = self.latest_acceleration.accel.accel.linear.x
        ego_current_state[0, 7] = self.latest_acceleration.accel.accel.linear.y
        ego_current_state[0, 8] = steering_angle  # steering angle
        ego_current_state[0, 9] = yaw_rate  # yaw rate

        input_dict = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": self.neighbor,
            "lanes": lanes_tensor,
            "lanes_speed_limit": torch.zeros((1, 70, 1), device=dev),
            "lanes_has_speed_limit": torch.zeros(
                (1, 70, 1), dtype=torch.bool, device=dev
            ),
            "route_lanes": self.route_tensor,
            "route_lanes_speed_limit": torch.zeros((1, 25, 1), device=dev),
            "route_lanes_has_speed_limit": torch.zeros(
                (1, 25, 1), dtype=torch.bool, device=dev
            ),
            "static_objects": torch.zeros((1, 5, 10), device=dev),
            "sampled_trajectories": torch.zeros((1, 11, 81, 4), device=dev),
            "diffusion_time": torch.zeros((1, 11, 81, 4), device=dev),
        }
        input_dict = self.config_obj.observation_normalizer(input_dict)
        # visualize_inputs(
        #     input_dict, self.config_obj.observation_normalizer, "./input.png"
        # )
        start = time.time()
        with torch.no_grad():
            out = self.diffusion_planner(input_dict)[1]
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"inference time: {elapsed_msec:.4f} msec")
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
        prev_x = prev_y = 0
        for i in range(pred.shape[0]):
            curr_x = pred[i, 0]
            curr_y = pred[i, 1]
            distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            curr_heading = pred[i, 2]
            # self.get_logger().info(f"Predicted position: {curr_x}, {curr_y}, {curr_heading}")
            # transform to map frame
            vec3d = [curr_x, curr_y, 0.0]
            vec3d = self.bl2map_matrix_4x4 @ np.array([*vec3d, 1.0])
            rot = Rotation.from_euler("z", curr_heading, degrees=False).as_matrix()
            rot = self.bl2map_matrix_4x4[0:3, 0:3] @ rot
            quat = Rotation.from_matrix(rot).as_quat()
            point = TrajectoryPoint()
            total_seconds = float(i * dt)
            secs = int(total_seconds)
            nanosec = int((total_seconds - secs) * 1e9)
            point.time_from_start = Duration()
            point.time_from_start.sec = secs
            point.time_from_start.nanosec = nanosec
            point.pose.position.x = vec3d[0]
            point.pose.position.y = vec3d[1]
            point.pose.position.z = vec3d[2]
            point.pose.orientation.x = quat[0]
            point.pose.orientation.y = quat[1]
            point.pose.orientation.z = quat[2]
            point.pose.orientation.w = quat[3]
            point.longitudinal_velocity_mps = distance / dt
            trajectory_msg.points.append(point)
            prev_x = curr_x
            prev_y = curr_y
        # Publish the trajectory
        self.pub_trajectory.publish(trajectory_msg)

        # Publish the trajectory marker
        marker_array = create_trajectory_marker(trajectory_msg)
        self.pub_trajectory_marker.publish(marker_array)

    def cb_vector_map(self, msg):
        self.vector_map = msg
        self.get_logger().info("Received vector map")

    def cb_route(self, msg):
        self.route = msg
        self.get_logger().info(
            f"Received lanelet route. Number of lanelets: {len(msg.segments)}"
        )

    def cb_tracked_objects(self, msg):
        dev = self.diffusion_planner.parameters().__next__().device
        new_tracked_objs = {}
        label_map = {
            0: 0,  # unknown -> vehicle
            1: 0,  # car -> vehicle
            2: 0,  # truck -> vehicle
            3: 0,  # bus -> vehicle
            4: 0,  # trailer -> vehicle
            5: 2,  # motorcycle -> bicycle
            6: 2,  # bicycle -> bicycle
            7: 1,  # pedestrian -> pedestrian
        }
        for i in range(len(msg.objects)):
            obj = msg.objects[i]
            object_id_bytes = bytes(obj.object_id.uuid)
            classification = obj.classification
            label_list = [i.label for i in classification]
            probability_list = [i.probability for i in classification]
            max_index = np.argmax(probability_list)
            label = label_list[max_index]
            label_in_model = label_map[label]
            kinematics = obj.kinematics
            pose_in_map_4x4 = pose_to_mat4x4(kinematics.pose_with_covariance.pose)
            pose_in_bl_4x4 = self.map2bl_matrix_4x4 @ pose_in_map_4x4
            cos, sin = rot3x3_to_heading_cos_sin(pose_in_bl_4x4[0:3, 0:3])
            twist_in_map_4x4 = np.eye(4)
            twist_in_map_4x4[0, 3] = kinematics.twist_with_covariance.twist.linear.x
            twist_in_map_4x4[1, 3] = kinematics.twist_with_covariance.twist.linear.y
            twist_in_map_4x4[2, 3] = kinematics.twist_with_covariance.twist.linear.z
            twist_in_bl_4x4 = self.map2bl_matrix_4x4 @ twist_in_map_4x4
            shape = obj.shape
            if object_id_bytes in self.tracked_objs:
                index = self.tracked_objs[object_id_bytes]
                self.neighbor[0, index, 0:-1] = self.neighbor[0, index, 1:].clone()
                self.neighbor[0, index, -1, 0] = pose_in_bl_4x4[0, 3]  # x
                self.neighbor[0, index, -1, 1] = pose_in_bl_4x4[1, 3]  # y
                self.neighbor[0, index, -1, 2] = cos  # heading cos
                self.neighbor[0, index, -1, 3] = sin  # heading sin
                self.neighbor[0, index, -1, 4] = twist_in_bl_4x4[0, 3]  # velocity x
                self.neighbor[0, index, -1, 5] = twist_in_bl_4x4[1, 3]  # velocity y
                self.neighbor[0, index, -1, 6] = shape.dimensions.x  # length
                self.neighbor[0, index, -1, 7] = shape.dimensions.y  # width
                self.neighbor[0, index, -1, 8] = label_in_model == 0  # vehicle
                self.neighbor[0, index, -1, 9] = label_in_model == 1  # pedestrian
                self.neighbor[0, index, -1, 10] = label_in_model == 2  # bicycle
                new_tracked_objs[object_id_bytes] = index
            else:
                first_empty_index = -1
                for j in range(32):
                    if self.neighbor[0, j, 0, 0] == 0:
                        first_empty_index = j
                        break
                if first_empty_index == -1:
                    continue
                self.neighbor[0, first_empty_index, 0, 0] = pose_in_bl_4x4[0, 3]  # x
                self.neighbor[0, first_empty_index, 0, 1] = pose_in_bl_4x4[1, 3]  # y
                self.neighbor[0, first_empty_index, 0, 2] = cos  # heading cos
                self.neighbor[0, first_empty_index, 0, 3] = sin  # heading sin
                self.neighbor[0, first_empty_index, 0, 4] = twist_in_bl_4x4[0, 3]
                self.neighbor[0, first_empty_index, 0, 5] = twist_in_bl_4x4[1, 3]
                self.neighbor[0, first_empty_index, 0, 6] = shape.dimensions.x
                self.neighbor[0, first_empty_index, 0, 7] = shape.dimensions.y
                self.neighbor[0, first_empty_index, 0, 8] = label_in_model == 0
                self.neighbor[0, first_empty_index, 0, 9] = label_in_model == 1
                self.neighbor[0, first_empty_index, 0, 10] = label_in_model == 2
                new_tracked_objs[object_id_bytes] = first_empty_index

        self.tracked_objs = new_tracked_objs
        for i in range(32):
            if i not in self.tracked_objs.values():
                self.neighbor[0, i] = torch.zeros((21, 11), device=dev)

    def process_route(self, msg):
        self.route_tensor = torch.zeros(
            (1, 25, 20, 12), dtype=torch.float32, device="cuda"
        )

        for i in range(len(msg.segments)):
            ll2_id = msg.segments[i].preferred_primitive.id
            if ll2_id in self.static_map.lane_segments:
                curr_result = process_segment(
                    self.static_map.lane_segments[ll2_id],
                    self.map2bl_matrix_4x4,
                    mask_range=100,
                )
                if curr_result is None:
                    continue
                self.route_tensor[0, i] = torch.from_numpy(curr_result).cuda()
            assert ll2_id not in self.static_map.crosswalk_segments
            assert ll2_id not in self.static_map.boundary_segments

        # すでに通過した部分は除外する
        self.route_tensor = self.route_tensor[:, : len(msg.segments), :, :]
        self.route_tensor = self.route_tensor.view(1, -1, 12)
        # 一番現在位置に近いindexを取得
        diff_x = self.route_tensor[:, :, 0]
        diff_y = self.route_tensor[:, :, 1]
        dist = torch.sqrt(diff_x**2 + diff_y**2)
        min_index = torch.argmin(dist, dim=1)
        # min_index以降を0番目に持ってきて、末尾以降は0で埋める
        rem = 25 - len(msg.segments)
        self.route_tensor = torch.cat(
            [
                self.route_tensor[:, min_index:, :],
                torch.zeros_like(self.route_tensor[:, -1, :]).repeat(
                    1, min_index + rem * 20, 1
                ),
            ],
            dim=1,
        )
        self.route_tensor = self.route_tensor.view(1, 25, 20, 12)

        marker_array = MarkerArray()
        centerline_marker = Marker()
        centerline_marker.header.stamp = self.get_clock().now().to_msg()
        centerline_marker.header.frame_id = "map"
        centerline_marker.ns = "route"
        centerline_marker.id = 0
        centerline_marker.type = Marker.LINE_STRIP
        centerline_marker.action = Marker.ADD
        centerline_marker.pose.orientation.w = 1.0
        centerline_marker.scale.x = 0.6
        centerline_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        centerline_marker.lifetime = Duration(sec=0, nanosec=int(1e8))
        centerline_marker.points = []
        for j in range(len(msg.segments)):
            centerline_in_base_link = self.route_tensor[0, j, :, :2].cpu().numpy()
            centerline_in_base_link = np.concatenate(
                [
                    centerline_in_base_link,
                    np.zeros((centerline_in_base_link.shape[0], 1)),
                    np.ones((centerline_in_base_link.shape[0], 1)),
                ],
                axis=1,
            )
            centerline_in_map = (self.bl2map_matrix_4x4 @ centerline_in_base_link.T).T
            # Create a marker for the centerline
            for i, point in enumerate(centerline_in_map):
                p = Point()
                norm = np.linalg.norm(centerline_in_base_link[i])
                if norm < 2:
                    continue
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                centerline_marker.points.append(p)
        self.get_logger().info("Publishing route markers")
        marker_array.markers.append(centerline_marker)
        self.pub_route_marker.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    planner_node = DiffusionPlannerNode()

    # Use multi-threaded executor to handle multiple callbacks
    executor = SingleThreadedExecutor()
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
