import rclpy
import math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleOdometry
)

from scipy.spatial.transform import Rotation as R


class OffboardBodyFrameForward(Node):
    def __init__(self):
        super().__init__('offboard_body_frame_forward')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, qos)

        # State
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        self.armed = False
        self.nav_state = -1  # Fix: initialized
        self.phase = 'init'

        self.takeoff_alt = -2.5  # NED frame
        self.forward_body_x = 7.0  # 7m forward in body frame
        self.initial_pos = None

        # Timer
        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

        self.get_logger().info("Initialized Offboard Body Frame Forward Node.")

    def odom_cb(self, msg: VehicleOdometry):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        q = msg.q  # PX4 quaternion: [w, x, y, z]
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # convert to [x, y, z, w]
        self.current_yaw = r.as_euler('zyx')[0]

    def status_cb(self, msg: VehicleStatus):
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state

    def control_loop(self):
        now_us = int(Clock().now().nanoseconds / 1000)

        # 1. Publish OffboardControlMode at 50Hz
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us
        offboard_msg.position = True
        self.offboard_pub.publish(offboard_msg)

        # 2. Create trajectory setpoint
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = now_us

        if self.phase == 'init':
            traj_msg.position = [float(self.current_pos[0]), float(self.current_pos[1]), -0.25]
            self.traj_pub.publish(traj_msg)

            self.get_logger().info("Sending initial dummy setpoint...")
            self.send_mode_command()
            self.phase = 'wait_offboard'

        elif self.phase == 'wait_offboard':
            traj_msg.position = [float(self.current_pos[0]), float(self.current_pos[1]), -0.25]
            self.traj_pub.publish(traj_msg)

            if self.nav_state == 14:  # NAVIGATION_STATE_OFFBOARD
                self.get_logger().info("Entered OFFBOARD mode. Arming...")
                self.send_arm_command()
                self.phase = 'takeoff'

        elif self.phase == 'takeoff':
            traj_msg.position = [float(self.current_pos[0]), float(self.current_pos[1]), float(self.takeoff_alt)]
            traj_msg.yaw = self.current_yaw
            self.traj_pub.publish(traj_msg)

            self.get_logger().info(f"Current Z (ENU): {self.current_pos[2]:.2f} | Target Z (NED): {self.takeoff_alt}")

            # ENU to NED logic
            if -self.current_pos[2] >= abs(self.takeoff_alt) - 0.2:
                self.get_logger().info("Reached takeoff altitude. Moving forward...")
                self.initial_pos = self.current_pos.copy()
                self.phase = 'forward'

        elif self.phase == 'forward':
            # Convert 7m forward in body frame to ENU
            x_body, y_body = self.forward_body_x, 0.0
            dx = x_body * math.cos(self.current_yaw) - y_body * math.sin(self.current_yaw)
            dy = x_body * math.sin(self.current_yaw) + y_body * math.cos(self.current_yaw)

            x_target = self.initial_pos[0] + dx
            y_target = self.initial_pos[1] + dy
            z_target = self.takeoff_alt

            traj_msg.position = [float(x_target), float(y_target), float(z_target)]
            traj_msg.yaw = self.current_yaw
            self.traj_pub.publish(traj_msg)

            dist_moved = np.linalg.norm(self.current_pos[:2] - self.initial_pos[:2])
            if dist_moved >= self.forward_body_x:
                self.get_logger().info("Reached 7m forward. Hovering...")
                self.phase = 'hover'

        elif self.phase == 'hover':
            traj_msg.position = [float(self.current_pos[0]), float(self.current_pos[1]), float(self.takeoff_alt)]
            traj_msg.yaw = self.current_yaw
            self.traj_pub.publish(traj_msg)

    def send_mode_command(self):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.param1 = 1.0
        msg.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Sent OFFBOARD mode command.")

    def send_arm_command(self):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.param1 = 1.0  # arm
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Sent ARM command.")


def main(args=None):
    rclpy.init(args=args)
    node = OffboardBodyFrameForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
