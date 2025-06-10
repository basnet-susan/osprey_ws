#!/usr/bin/env python3
"""
potential_field_steering_node.py

A ROS 2 node that:
 - Reads fused 2D LaserScan from 12 VL53L8CX sensors.
 - Reads PX4 VehicleOdometry.
 - Computes a potential‐field steering yaw.
 - Publishes the yaw and an RViz arrow marker.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf_transformations

# import your helper
from vl53l8cx_ros.potential_field_computer import PotentialFieldComputer

class PotentialFieldSteering(Node):
    def __init__(self):
        super().__init__('potential_field_steering')
        # goal params
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 10.0)
        gx = self.get_parameter('goal_x').value
        gy = self.get_parameter('goal_y').value
        self.goal = np.array([gx, gy, 0.0])

        # APF helper
        self.pf = PotentialFieldComputer(q_repel=1.0,
                                         q_attract=20.0,
                                         dist_threshold=2.0)

        # state
        self.x = self.y = self.yaw = 0.0
        self.smoothed_yaw = None
        self.yaw_alpha = 0.2
        self.forward_speed = 1.0

        # subs & pubs with best‐effort QoS
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            '/fused_obstacle_scan',
            self.scan_cb,
            qos_profile_sensor_data,
        )
        self.yaw_pub = self.create_publisher(Float32, '/vfh_desired_yaw', 10)
        self.marker_pub = self.create_publisher(Marker, '/vfh_arrow_marker', 10)

        self.get_logger().info(f"Started PF steering to ({gx},{gy})")

    def odom_cb(self, msg: VehicleOdometry):
        self.x, self.y = msg.position[0], msg.position[1]
        q = msg.q
        _, _, ψ = tf_transformations.euler_from_quaternion(
            [q[1], q[2], q[3], q[0]]
        )
        self.yaw = (math.degrees(ψ) + 360.0) % 360.0

    def scan_cb(self, scan: LaserScan):
        # build 2D obstacle cloud in robot frame
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        pts = [
            [r * math.cos(θ), r * math.sin(θ), 0.0]
            for r, θ in zip(scan.ranges, angles)
            if math.isfinite(r) and r > scan.range_min
        ]
        cloud = np.array(pts)

        # compute trajectory via APF
        traj = self.pf.compute_trajectory(
            point_cloud=cloud,
            vehicle_coord=np.array([self.x, self.y, 0.0]),
            target_coord=self.goal,
            speed=self.forward_speed,
            dt=0.05,
            n_points=20,
        )
        # extract heading from first step
        dx = traj[1, 0] - traj[0, 0]
        dy = traj[1, 1] - traj[0, 1]
        ψ_cmd = math.atan2(dy, dx)

        # smooth
        if self.smoothed_yaw is None:
            self.smoothed_yaw = ψ_cmd
        else:
            self.smoothed_yaw = (
                self.yaw_alpha * ψ_cmd + (1 - self.yaw_alpha) * self.smoothed_yaw
            )

        # publish yaw
        self.yaw_pub.publish(Float32(data=float(self.smoothed_yaw)))
        self.get_logger().info(
            f"APF: ({self.x:.2f}, {self.y:.2f}) → "
            f"({self.goal[0]:.2f}, {self.goal[1]:.2f}) "
            f"→ {math.degrees(self.smoothed_yaw):.1f}°"
        )

        # publish arrow
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'pf', 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        # **use floats here!**
        m.scale.x, m.scale.y, m.scale.z = 1.0, 0.1, 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        half = self.smoothed_yaw / 2.0
        m.pose.orientation.w = math.cos(half)
        m.pose.orientation.z = math.sin(half)
        self.marker_pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldSteering()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
