#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor


class DiffusionPlannerNode(Node):
    def __init__(self):
        super().__init__("diffusion_planner_node")


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
