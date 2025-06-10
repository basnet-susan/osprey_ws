#!/usr/bin/env python3
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

from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R


class OffboardBodyFrameVFHVelocity(Node):
    def __init__(self):
        super().__init__('offboard_body_frame_vfh_velocity')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, qos)
        self.vfh_sub = self.create_subscription(Float32, '/vfh_desired_yaw', self.vfh_yaw_cb, 10)
        self.goal_pub = self.create_publisher(Float32, '/vfh_target_direction', 10)

        self.goal_pos = np.array([0.0, 10.0])  # goal after wall
        self.goal_threshold = 0.5
        self.goal_reached = False

        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        self.current_target_yaw = 0.0
        self.vfh_yaw_received = False
        self.yaw_update_interval = 0.5  # seconds
        self.last_yaw_update_time = self.get_clock().now()

        self.armed = False
        self.nav_state = -1
        self.phase = 'init'

        self.takeoff_alt = -3.0  # meters
        self.forward_speed = 0.6

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Initialized VFH+ Offboard controller for goal navigation around wall.")

    def odom_cb(self, msg):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        q = msg.q
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        self.current_yaw = r.as_euler('zyx')[0]

    def status_cb(self, msg):
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state

    def vfh_yaw_cb(self, msg: Float32):
        now = self.get_clock().now()
        if (now - self.last_yaw_update_time).nanoseconds > self.yaw_update_interval * 1e9:
            self.current_target_yaw = float(msg.data)
            self.last_yaw_update_time = now
            self.vfh_yaw_received = True

    def control_loop(self):
        if self.goal_reached:
            return

        now = self.get_clock().now()
        now_us = int(now.nanoseconds / 1000)

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us

        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = now_us

        dx, dy = self.goal_pos - self.current_pos[:2]
        angle_to_goal_rad = math.atan2(dy, dx)
        angle_to_goal_deg = (math.degrees(angle_to_goal_rad) + 360) % 360
        self.goal_pub.publish(Float32(data=angle_to_goal_deg))

        dist_to_goal = np.linalg.norm(self.goal_pos - self.current_pos[:2])
        yaw_deg = math.degrees(self.current_yaw)
        yaw_error = (angle_to_goal_deg - yaw_deg + 540) % 360 - 180

        self.get_logger().info(
            f"[dist: {dist_to_goal:.2f} m | yaw: {yaw_deg:.1f}° | to_goal: {angle_to_goal_deg:.1f}° | error: {yaw_error:.1f}°]"
        )

        # Reset VFH flag if no update in 2 seconds
        if self.vfh_yaw_received:
            if (now - self.last_yaw_update_time).nanoseconds > 2.0 * 1e9:
                self.vfh_yaw_received = False

        if dist_to_goal < self.goal_threshold:
            self.get_logger().info("✅ Reached goal. Hovering.")
            self.goal_reached = True
            self.phase = 'hover'

        if self.phase == 'init':
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                -0.25
            ]
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)
            self.send_mode_command()
            self.phase = 'wait_offboard'

        elif self.phase == 'wait_offboard':
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                -0.25
            ]
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)
            if self.nav_state == 14:
                self.get_logger().info("Entered OFFBOARD mode. Arming...")
                self.send_arm_command()
                self.phase = 'takeoff'

        elif self.phase == 'takeoff':
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                float(self.takeoff_alt)
            ]
            traj_msg.yaw = float(self.current_yaw)
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)
            if -self.current_pos[2] >= abs(self.takeoff_alt) - 0.2:
                self.get_logger().info("Takeoff complete. Advancing toward goal with VFH+ steering...")
                self.phase = 'forward'

        elif self.phase == 'forward':
            if self.vfh_yaw_received and abs(yaw_error) < 90:
                desired_yaw = self.current_target_yaw
                self.get_logger().info("Using VFH yaw")
            else:
                desired_yaw = angle_to_goal_rad
                self.get_logger().warn("VFH yaw not available — falling back to goal yaw")

            vx = self.forward_speed * math.cos(desired_yaw)
            vy = self.forward_speed * math.sin(desired_yaw)

            traj_msg.velocity = [float(vx), float(vy), 0.0]
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                float(self.takeoff_alt)
            ]
            traj_msg.yaw = float(desired_yaw)
            offboard_msg.velocity = True
            
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)

        elif self.phase == 'hover':
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                float(self.takeoff_alt)
            ]
            traj_msg.yaw = float(self.current_yaw)
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)

        self.offboard_pub.publish(offboard_msg)

    def send_mode_command(self):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.param1 = 1.0
        msg.param2 = 6.0
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
        msg.param1 = 1.0
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
    node = OffboardBodyFrameVFHVelocity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
