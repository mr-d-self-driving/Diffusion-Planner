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
from autoware_perception_msgs.msg import DetectedObjects
from autoware_map_msgs.msg import LaneletMapBin
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
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
        self.config_obj = Config(config_json_path)
        self.diffusion_planner = Diffusion_Planner(self.config_obj)
        self.diffusion_planner.eval()
        self.diffusion_planner.cuda()
        self.diffusion_planner.decoder.decoder.training = False
        print(f"{self.config_obj.state_normalizer=}")

        ckpt_path = self.declare_parameter("ckpt_path", value="None").value
        self.get_logger().info(f"Checkpoint path: {ckpt_path}")
        ckpt = fileio.get(ckpt_path)
        with io.BytesIO(ckpt) as f:
            ckpt = torch.load(f)
        state_dict = ckpt["model"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.diffusion_planner.load_state_dict(new_state_dict)

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
        # self.vector_map_sub = self.create_subscription(
        #     LaneletMapBin,
        #     "/map/vector_map",
        #     self.cb_vector_map,
        #     transient_qos,
        # )
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
            "/planning/scenario_planning/lane_driving/trajectory",
            # "/diffusion_planner/trajectory",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_kinematic_state = None
        self.transform_mmatrix_4x4 = None
        self.inv_transform_matrix_4x4 = None
        self.vector_map = None
        self.route = None
        self.route_tensor = torch.zeros(
            (1, 25, 20, 12), dtype=torch.float32, device="cuda"
        )

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
        self.transform_mmatrix_4x4 = transform_matrix_4x4
        inv_transform_matrix_4x4 = np.eye(4)
        inv_transform_matrix_4x4[:3, :3] = transform_matrix.T
        inv_transform_matrix_4x4[:3, 3] = -transform_matrix.T @ translation
        self.inv_transform_matrix_4x4 = inv_transform_matrix_4x4

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
        ego_twist_linear = self.inv_transform_matrix_4x4[0:3, 0:3] @ ego_twist_linear
        ego_twist_angular = self.inv_transform_matrix_4x4[0:3, 0:3] @ ego_twist_angular
        linear_vel_norm = np.linalg.norm(ego_twist_linear)
        if abs(linear_vel_norm) < 0.2:
            yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
            steering_angle = 0.0
        else:
            yaw_rate = ego_twist_angular[2]
            wheel_base = 4.0
            steering_angle = np.arctan(yaw_rate * wheel_base / abs(linear_vel_norm))
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

        ego_current_state = torch.zeros((1, 10), device=dev)
        ego_current_state[0, 0] = 0  # x in base_link is always 0
        ego_current_state[0, 1] = 0  # y in base_link is always 0
        ego_current_state[0, 2] = 1  # heading cos in base_link is always 1
        ego_current_state[0, 3] = 0  # heading sin in base_link is always 0
        ego_current_state[0, 4] = ego_twist_linear[0]  # velocity x
        ego_current_state[0, 5] = ego_twist_linear[1]  # velocity y
        ego_current_state[0, 6] = 0  # acceleration x (TODO)
        ego_current_state[0, 7] = 0  # acceleration y (TODO)
        ego_current_state[0, 8] = steering_angle  # steering angle
        ego_current_state[0, 9] = yaw_rate  # yaw rate

        input_dict = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": torch.zeros((1, 32, 21, 11), device=dev),
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
        for i in range(pred.shape[0]):
            curr_x = pred[i, 0]
            curr_y = pred[i, 1]
            curr_heading = pred[i, 2]
            # self.get_logger().info(f"Predicted position: {curr_x}, {curr_y}, {curr_heading}")
            # transform to map frame
            vec3d = [curr_x, curr_y, 0.0]
            vec3d = self.transform_mmatrix_4x4 @ np.array([*vec3d, 1.0])
            rot = Rotation.from_euler("z", curr_heading, degrees=False).as_matrix()
            rot = self.transform_mmatrix_4x4[0:3, 0:3] @ rot
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
            point.longitudinal_velocity_mps = 8.333333015441895
            trajectory_msg.points.append(point)
        # Publish the trajectory
        self.pub_trajectory.publish(trajectory_msg)

        # Publish the trajectory marker
        marker_array = self.create_trajectory_marker(trajectory_msg)
        self.pub_trajectory_marker.publish(marker_array)

    def cb_vector_map(self, msg):
        self.vector_map = msg
        self.get_logger().info("Received vector map")

    def cb_route(self, msg):
        if self.latest_kinematic_state is None:
            return
        self.route = msg
        self.get_logger().info(
            f"Received lanelet route. Number of lanelets: {len(msg.segments)}"
        )

    def process_route(self, msg):
        self.route_tensor = torch.zeros(
            (1, 25, 20, 12), dtype=torch.float32, device="cuda"
        )

        for i in range(len(msg.segments)):
            ll2_id = msg.segments[i].preferred_primitive.id
            if ll2_id in self.static_map.lane_segments:
                curr_result = process_segment(
                    self.static_map.lane_segments[ll2_id],
                    self.inv_transform_matrix_4x4,
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
            centerline_in_map = (
                self.transform_mmatrix_4x4 @ centerline_in_base_link.T
            ).T
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

    def create_trajectory_marker(self, trajectory_msg):
        """
        Trajectoryメッセージからマーカー配列を作成
        """
        marker_array = MarkerArray()

        # トラジェクトリパスのマーカー
        path_marker = Marker()
        path_marker.header = trajectory_msg.header
        path_marker.ns = "trajectory_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.2  # 線の太さ
        path_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # 緑色
        path_marker.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1秒

        # ポイントのマーカー
        points_marker = Marker()
        points_marker.header = trajectory_msg.header
        points_marker.ns = "trajectory_points"
        points_marker.id = 1
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        points_marker.pose.orientation.w = 1.0
        points_marker.scale.x = 0.4  # 球の大きさ
        points_marker.scale.y = 0.4
        points_marker.scale.z = 0.4
        points_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # 赤色
        points_marker.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1秒

        # ポイントの向きを表す矢印マーカー
        arrows_marker = Marker()
        arrows_marker.header = trajectory_msg.header
        arrows_marker.ns = "trajectory_arrows"
        arrows_marker.id = 2
        arrows_marker.type = Marker.ARROW
        arrows_marker.action = Marker.ADD
        arrows_marker.scale.x = 0.3  # 矢印の太さ
        arrows_marker.scale.y = 0.5  # 矢印の先端の太さ
        arrows_marker.scale.z = 0.5  # 矢印の先端の長さ
        arrows_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)  # 青色
        arrows_marker.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1秒
        arrows_marker.pose.orientation.w = 1.0

        # 1秒ごとのマーカーを別色で表示
        time_markers = Marker()
        time_markers.header = trajectory_msg.header
        time_markers.ns = "trajectory_time_markers"
        time_markers.id = 3
        time_markers.type = Marker.SPHERE_LIST
        time_markers.action = Marker.ADD
        time_markers.pose.orientation.w = 1.0
        time_markers.scale.x = 0.6  # 球の大きさ
        time_markers.scale.y = 0.6
        time_markers.scale.z = 0.6
        time_markers.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # 黄色
        time_markers.lifetime = Duration(sec=0, nanosec=100000000)  # 0.1秒

        # 軌道の各ポイントをマーカーに追加
        for i, point in enumerate(trajectory_msg.points):
            # パスのポイント
            p = Point()
            p.x = point.pose.position.x
            p.y = point.pose.position.y
            p.z = point.pose.position.z
            path_marker.points.append(p)

            # すべてのポイント
            points_marker.points.append(p)

            # 1秒ごとのマーカー
            if (i + 1) % 10 == 0:
                time_markers.points.append(p)

            # 矢印マーカー（向き）
            if i % 20 == 0:
                # 矢印の始点
                start_point = Point()
                start_point.x = point.pose.position.x
                start_point.y = point.pose.position.y
                start_point.z = point.pose.position.z
                arrows_marker.points.append(start_point)

                # 矢印の終点（向きに沿って少し前方）
                q = [
                    point.pose.orientation.x,
                    point.pose.orientation.y,
                    point.pose.orientation.z,
                    point.pose.orientation.w,
                ]
                rot = Rotation.from_quat(q)
                direction = rot.as_matrix()[:, 0]  # x軸方向

                end_point = Point()
                end_point.x = start_point.x + direction[0] * 1.0  # 1mの長さ
                end_point.y = start_point.y + direction[1] * 1.0
                end_point.z = start_point.z + direction[2] * 1.0
                arrows_marker.points.append(end_point)

        marker_array.markers.append(path_marker)
        marker_array.markers.append(points_marker)
        marker_array.markers.append(arrows_marker)
        marker_array.markers.append(time_markers)

        return marker_array


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
