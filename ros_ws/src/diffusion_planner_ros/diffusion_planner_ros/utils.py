from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
import numpy as np
import torch
from autoware_perception_msgs.msg import TrackedObjects
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
from dataclasses import dataclass


@dataclass
class TrackingObject:
    kinematics_list: list
    shape_list: list
    class_label: int


def pose_to_mat4x4(pose):
    """
    ROSのPoseを4x4の行列に変換
    """
    mat = np.array(
        [
            [1.0, 0.0, 0.0, pose.position.x],
            [0.0, 1.0, 0.0, pose.position.y],
            [0.0, 0.0, 1.0, pose.position.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    rot = Rotation.from_quat(q)
    mat[:3, :3] = rot.as_matrix()
    return mat


def rot3x3_to_heading_cos_sin(rot3x3):
    """
    回転行列からヘディング角を計算
    """
    rot = Rotation.from_matrix(rot3x3)
    heading = rot.as_euler("zyx")[0]  # ZYX順でヘディングを取得
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    return cos_heading, sin_heading


def create_current_ego_state(kinematic_state_msg, acceleration_msg, wheel_base):
    ego_twist_linear = kinematic_state_msg.twist.twist.linear
    ego_twist_angular = kinematic_state_msg.twist.twist.angular
    ego_twist_linear = np.array(
        [ego_twist_linear.x, ego_twist_linear.y, ego_twist_linear.z]
    )
    ego_twist_angular = np.array(
        [ego_twist_angular.x, ego_twist_angular.y, ego_twist_angular.z]
    )
    linear_vel_norm = np.linalg.norm(ego_twist_linear)
    if abs(linear_vel_norm) < 0.2:
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
        steering_angle = 0.0
    else:
        yaw_rate = ego_twist_angular[2]
        steering_angle = np.arctan(yaw_rate * wheel_base / abs(linear_vel_norm))
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

    ego_current_state = torch.zeros((1, 10))
    ego_current_state[0, 0] = 0  # x in base_link is always 0
    ego_current_state[0, 1] = 0  # y in base_link is always 0
    ego_current_state[0, 2] = 1  # heading cos in base_link is always 1
    ego_current_state[0, 3] = 0  # heading sin in base_link is always 0
    ego_current_state[0, 4] = ego_twist_linear[0]  # velocity x
    ego_current_state[0, 5] = ego_twist_linear[1]  # velocity y
    ego_current_state[0, 6] = acceleration_msg.accel.accel.linear.x
    ego_current_state[0, 7] = acceleration_msg.accel.accel.linear.y
    ego_current_state[0, 8] = steering_angle  # steering angle
    ego_current_state[0, 9] = yaw_rate  # yaw rate
    return ego_current_state


def tracking_one_step(msg: TrackedObjects, tracked_objs: dict) -> dict:
    updated_tracked_objs = {}
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
        shape = obj.shape
        if object_id_bytes in tracked_objs:
            tracked_obj = tracked_objs[object_id_bytes]
            tracked_obj.shape_list.append(shape)
            tracked_obj.kinematics_list.append(kinematics)
            updated_tracked_objs[object_id_bytes] = tracked_obj
        else:
            updated_tracked_objs[object_id_bytes] = TrackingObject(
                kinematics_list=[kinematics],
                shape_list=[shape],
                class_label=label_in_model,
            )

    return updated_tracked_objs


def convert_tracked_objects_to_tensor(
    tracked_objs: dict,
    map2bl_matrix_4x4: np.ndarray,
    max_num_objects: int,
    max_timesteps: int,
) -> torch.Tensor:
    neighbor = torch.zeros((1, max_num_objects, max_timesteps, 11))
    for i, (object_id_bytes, tracked_obj) in enumerate(tracked_objs.items()):
        if i >= max_num_objects:
            break
        label_in_model = tracked_obj.class_label
        for j in range(max_timesteps):
            if j < len(tracked_obj.kinematics_list):
                kinematics = tracked_obj.kinematics_list[j]
                shape = tracked_obj.shape_list[j]
            else:
                kinematics = tracked_obj.kinematics_list[-1]
                shape = tracked_obj.shape_list[-1]
            pose_in_map_4x4 = pose_to_mat4x4(kinematics.pose_with_covariance.pose)
            pose_in_bl_4x4 = map2bl_matrix_4x4 @ pose_in_map_4x4
            cos, sin = rot3x3_to_heading_cos_sin(pose_in_bl_4x4[0:3, 0:3])
            twist_in_map_4x4 = np.eye(4)
            twist_in_map_4x4[0, 3] = kinematics.twist_with_covariance.twist.linear.x
            twist_in_map_4x4[1, 3] = kinematics.twist_with_covariance.twist.linear.y
            twist_in_map_4x4[2, 3] = kinematics.twist_with_covariance.twist.linear.z
            twist_in_bl_4x4 = map2bl_matrix_4x4 @ twist_in_map_4x4
            neighbor[0, i, 20 - j, 0] = pose_in_bl_4x4[0, 3]  # x
            neighbor[0, i, 20 - j, 1] = pose_in_bl_4x4[1, 3]  # y
            neighbor[0, i, 20 - j, 2] = cos  # heading cos
            neighbor[0, i, 20 - j, 3] = sin  # heading sin
            neighbor[0, i, 20 - j, 4] = twist_in_bl_4x4[0, 3]  # velocity x
            neighbor[0, i, 20 - j, 5] = twist_in_bl_4x4[1, 3]  # velocity y
            neighbor[0, i, 20 - j, 6] = shape.dimensions.x  # length
            neighbor[0, i, 20 - j, 7] = shape.dimensions.y  # width
            neighbor[0, i, 20 - j, 8] = label_in_model == 0  # vehicle
            neighbor[0, i, 20 - j, 9] = label_in_model == 1  # pedestrian
            neighbor[0, i, 20 - j, 10] = label_in_model == 2  # bicycle
    return neighbor


def tracking(tracked_list: list[TrackedObjects]):
    tracked_objs = {}
    for i in range(len(tracked_list)):
        msg = tracked_list[i]
        updated_tracked_objs = tracking_one_step(msg, tracked_objs)
    return updated_tracked_objs


def convert_prediction_to_tensor(
    pred: torch.Tensor, bl2map_matrix_4x4: np.array, stamp
) -> Trajectory:
    # Convert to Trajectory message
    trajectory_msg = Trajectory()
    trajectory_msg.header.stamp = stamp
    trajectory_msg.header.frame_id = "map"
    trajectory_msg.points = []
    dt = 0.1
    prev_x = prev_y = 0
    for i in range(pred.shape[0]):
        point = TrajectoryPoint()

        # position
        curr_x = pred[i, 0]
        curr_y = pred[i, 1]
        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        vec3d = [curr_x, curr_y, 0.0]
        vec3d = bl2map_matrix_4x4 @ np.array([*vec3d, 1.0])
        point.pose.position.x = vec3d[0]
        point.pose.position.y = vec3d[1]
        point.pose.position.z = vec3d[2]

        # orientation
        curr_heading = pred[i, 2]
        rot = Rotation.from_euler("z", curr_heading, degrees=False).as_matrix()
        rot = bl2map_matrix_4x4[0:3, 0:3] @ rot
        quat = Rotation.from_matrix(rot).as_quat()
        point.pose.orientation.x = quat[0]
        point.pose.orientation.y = quat[1]
        point.pose.orientation.z = quat[2]
        point.pose.orientation.w = quat[3]

        # time/velocity
        seconds_float = float(i * dt)
        seconds_int = int(seconds_float)
        nanosec = int((seconds_float - seconds_int) * 1e9)
        point.time_from_start = Duration()
        point.time_from_start.sec = seconds_int
        point.time_from_start.nanosec = nanosec
        point.longitudinal_velocity_mps = distance / dt
        trajectory_msg.points.append(point)

        prev_x = curr_x
        prev_y = curr_y

    return trajectory_msg


def create_trajectory_marker(trajectory_msg):
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
