#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import matplotlib.pyplot as plt
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from pathlib import Path


class VFHHistogramGenerator(Node):
    def __init__(self):
        super().__init__('vfh_histogram_generator')

        # VFH parameters
        self.a = 0.875
        self.b = 0.25
        self.alpha = 5  # sector width in degrees
        self.k = int(360 / self.alpha)
        self.l = 2  # smoothing window size
        self.range_min = 0.05
        self.range_max = 4.0

        self.sub = self.create_subscription(LaserScan, '/fused_obstacle_scan', self.scan_callback, 10)
        self.get_logger().info("VFH Histogram Generator waiting for one scan...")

    def scan_callback(self, scan: LaserScan):
        ranges = np.array(scan.ranges)
        ranges = np.clip(ranges, self.range_min, self.range_max)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        # Step 1: Compute magnitude m[i] = a - b*d[i]
        m = np.zeros(360)
        for i in range(min(360, len(ranges))):
            d = ranges[i]
            if math.isfinite(d):
                m[i] = self.a - self.b * d
            else:
                m[i] = 0

        # Step 2: Compute polar obstacle density h
        h = np.zeros(self.k)
        for i in range(360):
            sector = i // self.alpha
            h[sector] += m[i]

        # Step 3: Smooth histogram
        smoothed_h = np.zeros_like(h)
        for i in range(self.k):
            acc = 0
            for offset in range(-self.l, self.l + 1):
                idx = (i + offset) % self.k
                weight = 2 if offset in (-1, 0, 1) else 1
                acc += weight * h[idx]
            smoothed_h[i] = acc / (2 * self.l + 1)

        self.plot_histograms(h, smoothed_h)
        rclpy.shutdown()

    def plot_histograms(self, h, smoothed_h):
        sectors = np.arange(self.k)
        angles = sectors * self.alpha

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axs[0].plot(angles, h, label='h (raw)', marker='o')
        axs[0].set_title('Polar Obstacle Density (h)')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('Magnitude')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(angles, smoothed_h, label='Smoothed h', marker='o', color='orange')
        axs[1].set_title('Smoothed Polar Histogram')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('Magnitude')
        axs[1].grid(True)
        axs[1].legend()

        output_path = str(Path.home() / "vfh_histograms.png")
        plt.savefig(output_path)
        self.get_logger().info(f"Saved histogram plots to {output_path}")
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = VFHHistogramGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
