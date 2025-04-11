#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import time

import numpy as np
import rclpy
import torch
from autoware_perception_msgs.msg import TrackedObjects, TrafficLightGroupArray
from autoware_planning_msgs.msg import LaneletRoute, Trajectory
from geometry_msgs.msg import AccelWithCovarianceStamped
from mmengine import fileio
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from visualization_msgs.msg import MarkerArray

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs

from .lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    fix_point_num,
    get_input_feature,
    process_segment,
)
from .utils import (
    convert_prediction_to_msg,
    convert_tracked_objects_to_tensor,
    create_current_ego_state,
    get_nearest_msg,
    get_transform_matrix,
    tracking_one_step,
)
from .visualization import (
    create_neighbor_marker,
    create_route_marker,
    create_trajectory_marker,
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

        # sub(4) traffic_light
        self.traffic_light_sub = self.create_subscription(
            TrafficLightGroupArray,
            "/perception/traffic_light_recognition/traffic_signals",
            self.cb_traffic_light,
            10,
        )

        # sub(5) route
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
        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # pub(1)[main] trajectory
        self.pub_trajectory = self.create_publisher(
            Trajectory,
            "/planning/scenario_planning/lane_driving/trajectory",
            # "/diffusion_planner/trajectory",
            pub_qos,
        )

        # pub(2)[debug] neighbor_marker
        self.pub_neighbor_marker = self.create_publisher(
            MarkerArray,
            "/diffusion_planner/debug/neighbor_marker",
            pub_qos,
        )

        # pub(3)[debug] route_marker
        self.pub_route_marker = self.create_publisher(
            MarkerArray,
            "/diffusion_planner/debug/route_marker",
            pub_qos,
        )

        # pub(4)[debug] trajectory_marker
        self.pub_trajectory_marker = self.create_publisher(
            MarkerArray,
            "/diffusion_planner/debug/trajectory_marker",
            pub_qos,
        )

        #############
        # Variables #
        #############
        self.kinematic_state_list = []
        self.acceleration_list = []
        self.traffic_light_list = []
        self.route = None
        self.tracked_objs = {}  # object_id -> TrackingObject

        self.get_logger().info("Diffusion Planner Node has been initialized")

    def cb_kinematic_state(self, msg):
        self.kinematic_state_list.append(msg)

    def cb_acceleration(self, msg):
        self.acceleration_list.append(msg)

    def cb_traffic_light(self, msg):
        self.traffic_light_list.append(msg)

    def cb_route(self, msg):
        self.route = msg

    def cb_tracked_objects(self, msg):
        if self.route is None:
            return
        dev = self.diffusion_planner.parameters().__next__().device
        stamp = msg.header.stamp
        # stamp = self.get_clock().now().to_msg()

        curr_kinematic_state, idx = get_nearest_msg(self.kinematic_state_list, stamp)
        self.kinematic_state_list = self.kinematic_state_list[idx:]
        curr_acceleration, idx = get_nearest_msg(self.acceleration_list, stamp)
        self.acceleration_list = self.acceleration_list[idx:]
        curr_traffic_light, idx = get_nearest_msg(self.traffic_light_list, stamp)
        self.traffic_light_list = self.traffic_light_list[idx:]

        if curr_kinematic_state is None:
            self.get_logger().warn("No kinematic state message found")
            return
        if curr_acceleration is None:
            self.get_logger().warn("No acceleration message found")
            return

        bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(
            curr_kinematic_state
        )
        traffic_light_recognition = {}
        if curr_traffic_light is not None:
            for traffic_light_group in curr_traffic_light.traffic_light_groups:
                traffic_light_group_id = traffic_light_group.traffic_light_group_id
                elements = traffic_light_group.elements
                assert len(elements) == 1, elements
                traffic_light_recognition[traffic_light_group_id] = elements[0].color

        # Ego
        start = time.time()
        ego_current_state = create_current_ego_state(
            curr_kinematic_state,
            curr_acceleration,
            self.wheel_base,
        ).to(dev)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Ego      : {elapsed_msec:.4f} msec")

        # Neighbors
        start = time.time()
        self.tracked_objs = tracking_one_step(msg, self.tracked_objs)
        neighbor = convert_tracked_objects_to_tensor(
            self.tracked_objs,
            map2bl_matrix_4x4,
            max_num_objects=32,
            max_timesteps=21,
        ).to(dev)
        marker_array = create_neighbor_marker(neighbor, stamp)
        self.pub_neighbor_marker.publish(marker_array)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Neighbor : {elapsed_msec:.4f} msec")

        # Lane
        start = time.time()
        result_list = get_input_feature(
            self.static_map,
            map2bl_mat4x4=map2bl_matrix_4x4,
            center_x=curr_kinematic_state.pose.pose.position.x,
            center_y=curr_kinematic_state.pose.pose.position.y,
            mask_range=100,
            traffic_light_recognition=traffic_light_recognition,
        )
        lanes_tensor = torch.zeros((1, 70, 20, 12), dtype=torch.float32, device=dev)
        lanes_speed_limit = torch.zeros((1, 70, 1), dtype=torch.float32, device=dev)
        lanes_has_speed_limit = torch.zeros((1, 70, 1), dtype=torch.bool, device=dev)
        for i, result in enumerate(result_list):
            line_data, speed_limit = result
            lanes_tensor[0, i] = torch.from_numpy(line_data).cuda()
            assert speed_limit is not None
            lanes_speed_limit[0, i] = speed_limit
            lanes_has_speed_limit[0, i] = speed_limit is not None
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Lane     : {elapsed_msec:.4f} msec")

        # Route
        start = time.time()
        route_tensor = torch.zeros((1, 25, 20, 12), dtype=torch.float32, device=dev)
        route_speed_limit = torch.zeros((1, 25, 1), dtype=torch.float32, device=dev)
        route_has_speed_limit = torch.zeros((1, 25, 1), dtype=torch.bool, device=dev)
        for i in range(min(len(self.route.segments), 25)):
            ll2_id = self.route.segments[i].preferred_primitive.id
            if ll2_id in self.static_map.lane_segments:
                curr_result = process_segment(
                    self.static_map.lane_segments[ll2_id],
                    map2bl_matrix_4x4,
                    curr_kinematic_state.pose.pose.position.x,
                    curr_kinematic_state.pose.pose.position.y,
                    mask_range=100,
                    traffic_light_recognition=traffic_light_recognition,
                )
                if curr_result is None:
                    continue
                line_data, speed_limit = curr_result
                route_tensor[0, i] = torch.from_numpy(line_data).cuda()
                assert speed_limit is not None
                route_speed_limit[0, i] = speed_limit
                route_has_speed_limit[0, i] = speed_limit is not None
        marker_array = create_route_marker(route_tensor, stamp)
        self.pub_route_marker.publish(marker_array)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Route    : {elapsed_msec:.4f} msec")

        # Inference
        input_dict = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor,
            "lanes": lanes_tensor,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_tensor,
            "route_lanes_speed_limit": route_speed_limit,
            "route_lanes_has_speed_limit": route_has_speed_limit,
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
        self.get_logger().info(f"Time Inference: {elapsed_msec:.4f} msec")

        pred = out["prediction"]  # ([1, 11, T, 4])
        pred = pred[0, 0].detach().cpu().numpy().astype(np.float64)  # T, 4
        heading = np.arctan2(pred[:, 3], pred[:, 2])[..., None]
        pred = np.concatenate([pred[..., :2], heading], axis=-1)  # T, 3(x, y, heading)

        # Publish the trajectory
        trajectory_msg = convert_prediction_to_msg(pred, bl2map_matrix_4x4, stamp)
        self.pub_trajectory.publish(trajectory_msg)

        # Publish the trajectory marker
        marker_array = create_trajectory_marker(trajectory_msg)
        self.pub_trajectory_marker.publish(marker_array)


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
