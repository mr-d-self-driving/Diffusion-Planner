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
from visualization_msgs.msg import MarkerArray
from autoware_perception_msgs.msg import TrackedObjects
from autoware_planning_msgs.msg import LaneletRoute, Trajectory
from geometry_msgs.msg import AccelWithCovarianceStamped
from .lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    get_input_feature,
    process_segment,
    fix_point_num,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.visualize_input import visualize_inputs
import json
import torch
from diffusion_planner.utils.config import Config
import time
import numpy as np
from scipy.spatial.transform import Rotation
from mmengine import fileio
import io
from .utils import (
    create_trajectory_marker,
    create_route_marker,
    create_current_ego_state,
    tracking_one_step,
    convert_tracked_objects_to_tensor,
    convert_prediction_to_tensor,
)


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")

        ##############
        # Parameters #
        ##############
        # param(1) vector_map
        vector_map_path = self.declare_parameter("vector_map_path", value="None").value
        self.get_logger().info(f"Vector map path: {vector_map_path}")
        self.static_map = convert_lanelet(vector_map_path)
        self.static_map = fix_point_num(self.static_map)

        # param(2) config
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

        # param(3) checkpoint
        ckpt_path = self.declare_parameter("ckpt_path", value="None").value
        self.get_logger().info(f"Checkpoint path: {ckpt_path}")
        ckpt = fileio.get(ckpt_path)
        with io.BytesIO(ckpt) as f:
            ckpt = torch.load(f)
        state_dict = ckpt["model"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.diffusion_planner.load_state_dict(new_state_dict)

        # param(4) wheel_base
        self.wheel_base = self.declare_parameter("wheel_base", value=5.0).value
        self.get_logger().info(f"Wheel base: {self.wheel_base}")

        ###############
        # Subscribers #
        ###############
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

        # sub(3) tracked_objects
        # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_perception_msgs/msg/DetectedObjects.msg
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

        ##############
        # Publishers #
        ##############
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

        #############
        # Variables #
        #############
        self.latest_kinematic_state = None
        self.latest_acceleration = None
        self.bl2map_matrix_4x4 = None
        self.map2bl_matrix_4x4 = None
        self.route = None
        self.tracked_objs = {}  # object_id -> TrackingObject

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

    def cb_tracked_objects(self, msg):
        if self.latest_kinematic_state is None:
            return
        if self.route is None:
            return
        dev = self.diffusion_planner.parameters().__next__().device
        self.tracked_objs = tracking_one_step(msg, self.tracked_objs)
        self.get_logger().info(f"Tracked objects: {len(self.tracked_objs)}")

        neighbor = convert_tracked_objects_to_tensor(
            self.tracked_objs,
            self.map2bl_matrix_4x4,
            max_num_objects=32,
            max_timesteps=21,
        ).to(dev)

        start = time.time()
        route_tensor, route_lanes_speed_limit, route_lanes_has_speed_limit = (
            self.process_route(self.route)
        )
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"    route time: {elapsed_msec:.4f} msec")

        start = time.time()
        result_list = get_input_feature(
            self.static_map,
            map2bl_mat4x4=self.map2bl_matrix_4x4,
            center_x=self.latest_kinematic_state.pose.pose.position.x,
            center_y=self.latest_kinematic_state.pose.pose.position.y,
            mask_range=100,
        )
        lanes_tensor = torch.zeros((1, 70, 20, 12), dtype=torch.float32, device=dev)
        lanes_speed_limit = torch.zeros((1, 70, 1), dtype=torch.float32, device=dev)
        lanes_has_speed_limit = torch.zeros((1, 70, 1), dtype=torch.bool, device=dev)
        for i, result in enumerate(result_list):
            line_data, speed_limit = result
            lanes_tensor[0, i] = torch.from_numpy(line_data).cuda()
            assert speed_limit is not None
            # lanes_speed_limit[0, i] = speed_limit
            # lanes_has_speed_limit[0, i] = speed_limit is not None
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"get_input time: {elapsed_msec:.4f} msec")

        ego_current_state = create_current_ego_state(
            self.latest_kinematic_state,
            self.latest_acceleration,
            self.wheel_base,
        ).to(dev)

        input_dict = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor,
            "lanes": lanes_tensor,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_tensor,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": torch.zeros((1, 5, 10), device=dev),
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

        pred = out["prediction"]  # ([1, 11, T, 4])
        pred = pred[0, 0].detach().cpu().numpy().astype(np.float64)  # T, 4
        heading = np.arctan2(pred[:, 3], pred[:, 2])[..., None]
        pred = np.concatenate([pred[..., :2], heading], axis=-1)  # T, 3(x, y, heading)

        # Publish the trajectory
        trajectory_msg = convert_prediction_to_tensor(
            pred, self.bl2map_matrix_4x4, self.get_clock().now().to_msg()
        )
        self.pub_trajectory.publish(trajectory_msg)

        # Publish the trajectory marker
        marker_array = create_trajectory_marker(trajectory_msg)
        self.pub_trajectory_marker.publish(marker_array)

    def cb_route(self, msg):
        self.route = msg
        self.get_logger().info(
            f"Received lanelet route. Number of lanelets: {len(msg.segments)}"
        )

    def process_route(self, msg):
        route_tensor = torch.zeros((1, 25, 20, 12), dtype=torch.float32, device="cuda")
        route_lanes_speed_limit = torch.zeros(
            (1, 25, 1), dtype=torch.float32, device="cuda"
        )
        route_lanes_has_speed_limit = torch.zeros(
            (1, 25, 1), dtype=torch.bool, device="cuda"
        )

        for i in range(min(len(msg.segments), 25)):
            ll2_id = msg.segments[i].preferred_primitive.id
            if ll2_id in self.static_map.lane_segments:
                curr_result = process_segment(
                    self.static_map.lane_segments[ll2_id],
                    self.map2bl_matrix_4x4,
                    self.latest_kinematic_state.pose.pose.position.x,
                    self.latest_kinematic_state.pose.pose.position.y,
                    mask_range=100,
                )
                if curr_result is None:
                    continue
                line_data, speed_limit = curr_result
                route_tensor[0, i] = torch.from_numpy(line_data).cuda()
                assert speed_limit is not None
                # route_lanes_speed_limit[0, i] = speed_limit
                # route_lanes_has_speed_limit[0, i] = speed_limit is not None

        marker_array = create_route_marker(
            route_tensor, self.bl2map_matrix_4x4, self.get_clock().now().to_msg()
        )
        self.pub_route_marker.publish(marker_array)
        return route_tensor, route_lanes_speed_limit, route_lanes_has_speed_limit


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
