#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import matplotlib.pyplot as plt
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from pathlib import Path


class VFHFullNode(Node):
    def __init__(self):
        super().__init__('vfh_full_node')
        self.get_logger().info("VFH+ node with histogram and selection initialized.")

        # Parameters
        self.a = 5.0
        self.b = 0.5
        self.c = 0.5

        self.alpha_deg = 5
        self.alpha_rad = math.radians(self.alpha_deg)
        self.k = int(360 / self.alpha_deg)

        self.tau_high = 0.9
        self.tau_low = 0.5
        self.prev_b = np.zeros(self.k, dtype=int)

        self.robot_width = 0.2
        self.safety_margin = 0.3

        self.current_yaw = 0.0

        # Subscribers
        self.create_subscription(
            LaserScan, '/fused_obstacle_scan', self.scan_callback, 10)

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            Odometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)

        # Publisher for desired yaw
        self.yaw_pub = self.create_publisher(Float32, '/desired_yaw', 10)

    def odom_callback(self, msg: Odometry):
        # Use yaw only from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, scan: LaserScan):
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        h = np.zeros(self.k)
        d = np.full(self.k, np.inf)

        for r, angle in zip(ranges, angles):
            if not math.isfinite(r) or r < 0.05 or r > scan.range_max:
                continue

            m = self.c**2 * (self.a - self.b * r**2)
            if m <= 0:
                continue

            angle_deg = (math.degrees(angle) + 360.0) % 360.0
            k_bin = int(angle_deg // self.alpha_deg)
            h[k_bin] += m
            d[k_bin] = min(d[k_bin], r)

        # Binary histogram
        b = np.zeros_like(h, dtype=int)
        for k in range(self.k):
            if h[k] >= self.tau_high:
                b[k] = 1
            elif h[k] <= self.tau_low:
                b[k] = 0
            else:
                b[k] = self.prev_b[k]

        self.prev_b = b

        # Masked histogram
        b_masked = b.copy()
        for k in range(self.k):
            if b[k] == 1:
                dist = d[k] if math.isfinite(d[k]) else scan.range_max
                angle_expansion = math.asin(min(1.0, (self.robot_width + self.safety_margin) / dist))
                w_s = int(np.ceil(angle_expansion / self.alpha_rad))
                for offset in range(-w_s, w_s + 1):
                    b_masked[(k + offset) % self.k] = 1

        # Select candidate sectors
        candidate_sectors = [i for i in range(self.k) if b_masked[i] == 0]
        if not candidate_sectors:
            self.get_logger().warn("No candidate sector found! Obstacle in all directions.")
            return

        # Convert current yaw to sector index
        current_sector = int((math.degrees(self.current_yaw) + 360.0) % 360.0 // self.alpha_deg)

        # Choose sector closest to current direction
        selected_sector = min(
            candidate_sectors,
            key=lambda x: min(abs(x - current_sector), self.k - abs(x - current_sector))
        )

        desired_yaw_deg = selected_sector * self.alpha_deg
        desired_yaw_rad = math.radians(desired_yaw_deg)
        self.yaw_pub.publish(Float32(data=desired_yaw_rad))

        self.plot_histograms(h, b, b_masked, selected_sector)

    def plot_histograms(self, h, b, b_masked, selected_sector):
        sectors = np.arange(self.k)
        angles = sectors * self.alpha_deg
        selected_sector = selected_sector % self.k

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

        axs[0].plot(angles, h, label='h[k]', color='blue', marker='o')
        axs[0].axhline(self.tau_high, color='red', linestyle='--', label='τ_high')
        axs[0].axhline(self.tau_low, color='green', linestyle='--', label='τ_low')
        axs[0].set_title('Primary Polar Histogram h[k]')
        axs[0].legend()
        axs[0].grid(False)

        axs[1].bar(angles, b, width=self.alpha_deg, color='orange')
        axs[1].set_title('Binary Histogram b[k]')
        axs[1].grid(False)

        axs[2].bar(angles, b_masked, width=self.alpha_deg, color='red')
        axs[2].bar(angles[selected_sector], 1, width=self.alpha_deg, color='green')
        axs[2].set_title('Masked Histogram b\'[k] with Selected Sector')
        axs[2].grid(False)

        for ax in axs:
            ax.set_xticks(np.arange(0, 361, 30))
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Value")

        plt.savefig(Path.home() / "vfh_full_histogram.png")
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = VFHFullNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
