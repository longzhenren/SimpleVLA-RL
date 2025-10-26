#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import PositionTarget

class VelocityController(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__("velocity_controller")

        # Initialize last velocity command (default is zero velocity)
        self.last_velocity = PositionTarget()
        self.last_velocity.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        # Publisher for MAVROS velocity commands
        self.pub = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)

        # Subscriber for velocity commands
        self.create_subscription(PositionTarget, "/nav/velocity", self.cmd_vel_callback, 10)

        # Publish at a fixed rate (e.g., 10 Hz)
        self.rate_hz = 30
        self.timer = self.create_timer(1 / self.rate_hz, self.publish_velocity)

        self.get_logger().info(f"Velocity Controller Node Started (bridge): rate_hz={self.rate_hz}, sub=/nav/velocity, pub=/mavros/setpoint_raw/local")

    def cmd_vel_callback(self, msg):
        """Update the last velocity command when receiving a new one."""
        self.last_velocity = msg
        self.get_logger().debug(
            f"cmd recv frame={msg.coordinate_frame} mask={msg.type_mask} "
            f"v=({msg.velocity.x:.2f},{msg.velocity.y:.2f},{msg.velocity.z:.2f}) "
            f"yaw_rate={msg.yaw_rate:.2f}"
        )

    def publish_velocity(self):
        """Publish the last velocity command."""
        self.pub.publish(self.last_velocity)
        self.get_logger().debug("published last_velocity -> /mavros/setpoint_raw/local")


def main(args=None):
    rclpy.init(args=args)

    velocity_controller = VelocityController()

    try:
        rclpy.spin(velocity_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up before shutting down the node
        velocity_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()