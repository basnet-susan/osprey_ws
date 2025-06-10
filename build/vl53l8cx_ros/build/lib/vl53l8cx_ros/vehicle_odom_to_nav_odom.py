#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint
import time


class OffboardControlNode(Node):
    def __init__(self):
        super().__init__('offboard_takeoff_forward')
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.step = 0
        self.start_time = self.get_clock().now()

    def send_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0,
                             param4=0.0, param5=0.0, param6=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)
        self.get_logger().info(f'Command {command} sent')

    def timer_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Step 0: Arm
        if self.step == 0 and elapsed > 1.0:
            self.send_vehicle_command(400, 1)  # Arm
            self.step += 1

        # Step 1: Send OffboardControlMode
        elif self.step == 1 and elapsed > 3.0:
            self.send_offboard_mode()
            self.step += 1

        # Step 2: Send initial setpoint and start Offboard mode
        elif self.step == 2 and elapsed > 4.0:
            self.publish_position_setpoint(0.0, 0.0, -1.5, 0.0)  # hover 1.5m high
            self.send_vehicle_command(92, 1)  # Set mode to Offboard
            self.step += 1

        # Step 3: Keep sending setpoints (hover or move)
        elif self.step >= 3:
            # For example: move forward to x = 3.0m
            self.publish_position_setpoint(3.0, 0.0, -1.5, 0.0)

    def send_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)
        self.get_logger().info('Published OffboardControlMode')

    def publish_position_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [x, y, z]
        msg.yaw = yaw
        self.trajectory_setpoint_pub.publish(msg)
        self.get_logger().info(f'Published setpoint: x={x}, y={y}, z={z}, yaw={yaw}')


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
