#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time

import numpy as np
import onnxruntime as ort
import rclpy
import torch
from autoware_perception_msgs.msg import TrackedObjects, TrafficLightGroupArray
from autoware_planning_msgs.msg import LaneletRoute, Trajectory
from geometry_msgs.msg import AccelWithCovarianceStamped
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

from .lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    create_lane_tensor,
)
from .utils import (
    convert_prediction_to_msg,
    convert_tracked_objects_to_tensor,
    create_current_ego_state,
    get_nearest_msg,
    get_transform_matrix,
    parse_traffic_light_recognition,
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

        # param(2) config
        config_json_path = self.declare_parameter("config_json_path", value="None").value
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
        self.backend = self.declare_parameter("backend", value="PYTHORCH").value

        if self.backend == "PYTHORCH":
            ckpt_path = self.declare_parameter("ckpt_path", value="None").value
            self.get_logger().info(f"Checkpoint path: {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            state_dict = ckpt["model"]
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.diffusion_planner.load_state_dict(new_state_dict)
        elif self.backend == "ONNXRUNTIME":
            onnx_path = self.declare_parameter("onnx_path", value="None").value
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self.ort_session = ort.InferenceSession(
                onnx_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        else:
            self.get_logger().error(
                f"backend must be PYTHORCH or ONNXRUNTIME, but {self.backend} was given"
            )
            exit()

        # param(4) wheel_base
        self.wheel_base = self.declare_parameter("wheel_base", value=2.79).value
        self.get_logger().info(f"Wheel base: {self.wheel_base}")

        # param(5) batch_size
        self.batch_size = self.declare_parameter("batch_size", value=1).value
        self.get_logger().info(f"Batch size: {self.batch_size}")

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

        bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(curr_kinematic_state)
        traffic_light_recognition = {}
        if curr_traffic_light is not None:
            traffic_light_recognition = parse_traffic_light_recognition(curr_traffic_light)

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
        lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
            self.static_map.lane_segments.values(),
            map2bl_mat4x4=map2bl_matrix_4x4,
            center_x=curr_kinematic_state.pose.pose.position.x,
            center_y=curr_kinematic_state.pose.pose.position.y,
            mask_range=100,
            traffic_light_recognition=traffic_light_recognition,
            num_segments=70,
            dev=dev,
            do_sort=True,
        )
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Lane     : {elapsed_msec:.4f} msec")

        # Route
        start = time.time()
        target_segments = [
            self.static_map.lane_segments[segment.preferred_primitive.id]
            for segment in self.route.segments
        ]
        closest_distance = float("inf")
        closest_index = -1
        for j, segment in enumerate(target_segments):
            centerlines = segment.polyline.waypoints
            diff_x = centerlines[:, 0] - curr_kinematic_state.pose.pose.position.x
            diff_y = centerlines[:, 1] - curr_kinematic_state.pose.pose.position.y
            diff = np.sqrt(diff_x**2 + diff_y**2)
            distance = np.min(diff)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = j
        target_segments = target_segments[closest_index:]
        route_tensor, route_speed_limit, route_has_speed_limit = create_lane_tensor(
            target_segments,
            map2bl_mat4x4=map2bl_matrix_4x4,
            center_x=curr_kinematic_state.pose.pose.position.x,
            center_y=curr_kinematic_state.pose.pose.position.y,
            mask_range=100,
            traffic_light_recognition=traffic_light_recognition,
            num_segments=25,
            dev=dev,
            do_sort=False,
        )
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
            # "route_lanes_speed_limit": route_speed_limit,
            # "route_lanes_has_speed_limit": route_has_speed_limit,
            "static_objects": torch.zeros((1, 5, 10), device=dev),
        }
        if self.batch_size > 1:
            # copy the input dict for batch size
            for key in input_dict.keys():
                s = input_dict[key].shape
                ones = [1] * (len(s) - 1)
                input_dict[key] = input_dict[key].repeat(self.batch_size, *ones)
                input_dict[key] = input_dict[key]

        input_dict = self.config_obj.observation_normalizer(input_dict)

        if self.backend == "ONNXRUNTIME":
            for key in input_dict.keys():
                input_dict[key] = input_dict[key].cpu().numpy()

            input_dict["lanes_has_speed_limit"] = input_dict["lanes_has_speed_limit"].astype(
                np.bool_
            )
        # visualize_inputs(
        #     input_dict, self.config_obj.observation_normalizer, "./input.png"
        # )

        start = time.time()
        if self.backend == "PYTHORCH":
            with torch.no_grad():
                out = self.diffusion_planner(input_dict)[1]
                pred = out["prediction"].detach().cpu().numpy()
        elif self.backend == "ONNXRUNTIME":
            out = self.ort_session.run(None, input_dict)[0]
            pred = out
        end = time.time()
        elapsed_msec = (end - start) * 1000
        self.get_logger().info(f"Time Inference: {elapsed_msec:.4f} msec")
        # ([bs, 11, T, 4])
        # Publish
        for b in range(0, self.batch_size):
            curr_pred = pred[b, 0]
            curr_heading = np.arctan2(curr_pred[:, 3], curr_pred[:, 2])[..., None]
            curr_pred = np.concatenate([curr_pred[..., :2], curr_heading], axis=-1)
            trajectory_msg = convert_prediction_to_msg(curr_pred, bl2map_matrix_4x4, stamp)
            curr_marker_array = create_trajectory_marker(trajectory_msg)
            for i in range(len(curr_marker_array.markers)):
                curr_marker_array.markers[i].id += b

            if b == 0:
                marker_array = curr_marker_array
                self.pub_trajectory.publish(trajectory_msg)
            else:
                marker_array.markers.extend(curr_marker_array.markers)
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
