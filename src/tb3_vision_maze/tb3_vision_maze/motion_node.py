import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')

        self.planner_cmd = 'STOP'
        self.front_blocked = False
        self.exit_reached = False

        # Subscriptions
        self.create_subscription(String, '/planner_cmd', self.planner_cb, 10)
        self.create_subscription(String, '/maze_state', self.exit_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Publisher
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Motion node: safe distance + exit stop enabled")

    def planner_cb(self, msg):
        self.planner_cmd = msg.data

    def exit_cb(self, msg):
        if msg.data == "exit_visible":
            self.exit_reached = True

    def scan_cb(self, msg):
        """Angle-based front obstacle detection with larger safety margin"""
        self.front_blocked = False

        angle = msg.angle_min
        for r in msg.ranges:
            if r <= msg.range_min or r >= msg.range_max:
                angle += msg.angle_increment
                continue

            # Front sector ±15°
            if -0.26 <= angle <= 0.26:
                if r < 0.8:   # increased safety distance
                    self.front_blocked = True
                    return

            angle += msg.angle_increment

    def control_loop(self):
        cmd = Twist()

        # ---------- FINAL STOP AT EXIT ----------
        if self.exit_reached:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # ---------- OBSTACLE AVOIDANCE ----------
        if self.front_blocked:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:
            # ---------- NORMAL PLANNER CONTROL ----------
            if self.planner_cmd == 'MOVE_FORWARD':
                cmd.linear.x = 0.10   # slower = smoother + safer
            elif self.planner_cmd == 'TURN_LEFT':
                cmd.angular.z = 0.4
            elif self.planner_cmd == 'TURN_RIGHT':
                cmd.angular.z = -0.4
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    rclpy.shutdown()

