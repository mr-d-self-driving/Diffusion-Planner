import rclpy
from autoware_perception_msgs.msg import TrafficLightGroupArray
from rclpy.node import Node
from rclpy.qos import QoSProfile


class TrafficLightPublisher(Node):
    def __init__(self):
        super().__init__("traffic_light_publisher")

        qos_profile = QoSProfile(depth=10)
        self.publisher = self.create_publisher(
            TrafficLightGroupArray,
            "/perception/traffic_light_recognition/traffic_signals",
            qos_profile,
        )

        # Create timer for 10Hz (0.1 seconds)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = TrafficLightGroupArray()
        msg.stamp = self.get_clock().now().to_msg()

        # Publish an empty message
        self.publisher.publish(msg)


if __name__ == "__main__":
    rclpy.init()
    node = TrafficLightPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
