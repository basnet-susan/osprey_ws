#!/usr/bin/env python3
"""
offboard_vfh_velocity_vector.py

– Subscribes to /vfh_desired_yaw (Float32, in NED radians).
– Takes off to a fixed altitude.
– Once in “forward” phase, it:
    • Leaves the drone’s yaw at whatever it currently is.
    • Builds a velocity vector (vx, vy) = speed * [cos(ψ_des), sin(ψ_des)]
      (ψ_des is the VFH‐computed yaw in NED), and sends that as a BODY‐FRAME
      velocity (converted into NED‐frame for PX4) so the drone “moves in that
      direction” without ever rotating.
– Adds a distance‐to‐goal check so that, once within goal_threshold, it
  switches to “hover” phase and continuously sends position+yaw setpoints
  to hold the current location.
"""

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


class OffboardVFHVelocityVector(Node):
    def __init__(self):
        super().__init__('offboard_vfh_velocity_vector')

        # QoS: BEST_EFFORT for sensor streams
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub    = self.create_publisher(
            TrajectorySetpoint,   '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub     = self.create_publisher(
            VehicleCommand,       '/fmu/in/vehicle_command', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        self.status_sub = self.create_subscription(
            VehicleStatus,   '/fmu/out/vehicle_status_v1', self.status_cb, qos)
        self.vfh_sub    = self.create_subscription(
            Float32,         '/vfh_desired_yaw',          self.vfh_yaw_cb, 10)

        # publish world‐angle to goal for debug
        self.goal_pub = self.create_publisher(Float32, '/vfh_target_direction', 10)

        # -------- Node state --------
        # 
        self.goal_pos       = np.array([12.0, 0.0])  # ENU goal: (x_east, y_north)
        self.goal_threshold = 0.5                     # [m]

        # Current vehicle state from odometry
        #
        self.current_pos = np.array([0.0, 0.0, 0.0])  # NED: x_N, y_E, z_D
        self.current_yaw = 0.0                        # NED‐frame yaw [rad]

        # Latest VFH yaw in NED radians
        self.current_target_yaw = None
        self.vfh_yaw_received  = False

        # Offboard/arming state
        self.armed     = False
        self.nav_state = -1

        # Flight phases: i
        self.phase = 'init'
        self.takeoff_alt   = -3.0   # NED: climb to z = –3 m
        self.forward_speed = 0.5    # m/s (ground speed)

        # Timer for control loop (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Initialized OffboardVFHVelocityVector node.")

    def odom_cb(self, msg: VehicleOdometry):
        # Update current NED position
        self.current_pos = np.array([
            msg.position[0],  # north
            msg.position[1],  # east
            msg.position[2]   # down
        ])
        # Extract NED yaw from quaternion [w, x, y, z]
        q = msg.q
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        # as_euler('zyx')[0] is rotation about Z (NED frame)
        self.current_yaw = r.as_euler('zyx')[0]

    def status_cb(self, msg: VehicleStatus):
        self.armed     = (msg.arming_state == msg.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state

    def vfh_yaw_cb(self, msg: Float32):
        # Whenever VFH publishes a yaw, overwrite immediately
        self.current_target_yaw = float(msg.data)  # in NED [rad]
        self.vfh_yaw_received  = True

    def control_loop(self):
        now    = self.get_clock().now().nanoseconds * 1e-9
        now_us = int(now * 1e6)

        # Build OffboardControlMode + TrajectorySetpoint (timestamps in micro sec)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = now_us

        # current_pos (NED) to ENU for distance calculation:
        current_east = self.current_pos[1]   # NED y_E → ENU x
        current_north = self.current_pos[0]  # NED x_N → ENU y
        dx_e = self.goal_pos[0] - current_east
        dy_n = self.goal_pos[1] - current_north
        angle_to_goal_rad = math.atan2(dy_n, dx_e)
        angle_to_goal_deg = (math.degrees(angle_to_goal_rad) + 360.0) % 360.0
        self.goal_pub.publish(Float32(data=angle_to_goal_deg))

        # Distance to goal (horizontal) in ENU
        dist_to_goal = math.hypot(dx_e, dy_n)
        yaw_deg = (math.degrees(self.current_yaw) + 360.0) % 360.0
        yaw_error = ((angle_to_goal_deg - yaw_deg) + 540.0) % 360.0 - 180.0

        self.get_logger().info(
            f"[dist: {dist_to_goal:.2f} m | yaw: {yaw_deg:.1f}° | "
            f"to_goal: {angle_to_goal_deg:.1f}° | err: {yaw_error:.1f}°]"
        )

        # If within threshold and not already hovering, switch to hover
        if dist_to_goal < self.goal_threshold and self.phase != 'hover':
            self.get_logger().info("Reached goal. Switching to hover phase.")
            self.phase = 'hover'

        # ==== Phase handling ====
        if self.phase == 'init':
            # small position setpoint at current XYZ, altitude = –0.25 m
            traj_msg.position = [
                float(self.current_pos[0]),  # NED x_N
                float(self.current_pos[1]),  # NED y_E
                -0.25                       # NED z_D
            ]
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)

            # Request OFFBOARD mode immediately
            self.send_mode_command()
            self.phase = 'wait_offboard'

        elif self.phase == 'wait_offboard':
            # Keep publishing the same minor position until OFFBOARD is active
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                -0.25
            ]
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)

            if self.nav_state == 14:  # NAV_STATE_OFFBOARD == 14
                self.get_logger().info("Entered OFFBOARD mode. Arming...")
                self.send_arm_command()
                self.phase = 'takeoff'

        elif self.phase == 'takeoff':
            # Climb to takeoff_alt while maintaining current yaw
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                float(self.takeoff_alt)
            ]
            traj_msg.yaw = float(self.current_yaw)
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)

            # Wait until altitude reached (0.2 m margin)
            if -self.current_pos[2] >= abs(self.takeoff_alt) - 0.2:
                self.get_logger().info("Takeoff complete → entering forward phase.")
                self.phase = 'forward'

        elif self.phase == 'forward':
            # If no VFH yaw yet, hold in place
            if not self.vfh_yaw_received:
                traj_msg.position = [
                    float(self.current_pos[0]),
                    float(self.current_pos[1]),
                    float(self.takeoff_alt)
                ]
                traj_msg.yaw = float(self.current_yaw)
                offboard_msg.position = True
                self.traj_pub.publish(traj_msg)
                return

            # If we just switched to hover, fall through to hover block
            if self.phase == 'hover':
                pass
            else:
                # CENU velocity from VFH yaw in NED
                ψ_des_ned = self.current_target_yaw  # [rad]
                
                ψ_des_enu = (math.pi/2 - ψ_des_ned) % (2 * math.pi)
                # world‐frame velocity in ENU:
                vx_enu = self.forward_speed * math.cos(ψ_des_enu)
                vy_enu = self.forward_speed * math.sin(ψ_des_enu)
           
                vx_ned = vy_enu
                vy_ned = vx_enu

                traj_msg.velocity = [float(vx_ned), float(vy_ned), 0.0]
                # send a Z‐position so PX4 holds altitude
                traj_msg.position = [
                    float(self.current_pos[0]),
                    float(self.current_pos[1]),
                    float(self.takeoff_alt)
                ]
                # Keep yaw constant at current_yaw
                traj_msg.yaw = float(self.current_yaw)

                # velocity + yaw only (no XY position)
                offboard_msg.velocity = True
                offboard_msg.position = False
                self.traj_pub.publish(traj_msg)

                self.get_logger().info("Moving at a vector toward VFH yaw; no actual yaw change.")
                self.offboard_pub.publish(offboard_msg)
                return

        elif self.phase == 'hover':
            # Hold current position (NED) and yaw
            traj_msg.position = [
                float(self.current_pos[0]),
                float(self.current_pos[1]),
                float(self.takeoff_alt)
            ]
            traj_msg.yaw = float(self.current_yaw)
            offboard_msg.position = True
            self.traj_pub.publish(traj_msg)
            self.get_logger().info("Hovering at goal position.")

        #  publish OffboardControlMode flags in every phase
        self.offboard_pub.publish(offboard_msg)

    def send_mode_command(self):
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        # param1=1: set mode to custom (OFFBOARD); param2=6: OFFBOARD mode
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
        msg.param1 = 1.0  # 1 = arm
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
    node = OffboardVFHVelocityVector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
