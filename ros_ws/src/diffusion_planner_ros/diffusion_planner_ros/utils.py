from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation


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
