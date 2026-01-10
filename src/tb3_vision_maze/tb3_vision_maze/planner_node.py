import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.sub = self.create_subscription(
            String,
            '/maze_state',
            self.state_callback,
            10
        )
        self.pub = self.create_publisher(
            String,
            '/planner_cmd',
            10
        )
        self.get_logger().info('Planner node started')

    def state_callback(self, msg):
        state = msg.data

        # Classical symbolic planning logic
        if state == 'exit':
            action = 'STOP'
        elif state == 'door_left':
            action = 'TURN_LEFT'
        elif state == 'door_right':
            action = 'TURN_RIGHT'
        else:
            action = 'MOVE_FORWARD'

        self.get_logger().info(f'State: {state} -> Action: {action}')
        self.pub.publish(String(data=action))

def main():
    rclpy.init()
    node = PlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()
