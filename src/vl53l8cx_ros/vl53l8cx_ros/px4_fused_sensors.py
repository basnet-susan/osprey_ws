#!/usr/bin/env python3
"""
fused_obstacle_publisher.py

Fuses 10 VL53L8CX point clouds into a body-frame LaserScan:
 - Bins points into 180 sectors
 - Omits sensors 6 and 12.
 - 
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix

class FusedObstaclePublisher(Node):
    def __init__(self):
        super().__init__('fused_obstacle_publisher')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                                                       rclpy.Parameter.Type.BOOL,
                                                       True)])

        # Listen to 10 of the 12 VL53L8CX sensors (skip 6 and 12)
        self.sensor_topics = [
            f'/osprey/sensor_{i}/lidar/points'
            for i in range(1, 13)
            if i not in (6, 12)
        ]

        # cache latest pointcloud from each
        self.latest_msgs = {}
        for topic in self.sensor_topics:
            self.create_subscription(PointCloud2, topic,
                                     self._make_pc_cb(topic), 10)

        # publisher: only LaserScan
        self.laserscan_pub = self.create_publisher(
            LaserScan,
            '/fused_obstacle_scan',
            10
        )

        # TF listener to transform each sensor cloud into base link
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # fuse at 20 Hz
        self.create_timer(0.05, self.process_and_publish)

    def _make_pc_cb(self, topic):
        def cb(msg):
            self.latest_msgs[topic] = msg
        return cb

    def process_and_publish(self):
        now = self.get_clock().now()
        start_ns = now.nanoseconds

        fused_points = []
        sensors_used = 0

        # collect & transform
        for topic in self.sensor_topics:
            msg = self.latest_msgs.get(topic)
            if msg is None:
                continue

            # skip stale >200 ms
            msg_time_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            if (start_ns - msg_time_ns) / 1e6 > 200:
                continue

            # transform into base_link
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link',
                    msg.header.frame_id,
                    msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            rot = quaternion_matrix([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ])[:3, :3]
            trans = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])

            for x, y, z in pc2.read_points(msg,
                                           field_names=('x','y','z'),
                                           skip_nans=True):
                if not all(np.isfinite((x, y, z))):
                    continue
                # filter by height
                if z < -0.3 or z > 2.5:
                    continue
                # ignore very close (<0.05 m)
                if math.hypot(x, y) < 0.05:
                    continue
                p = rot.dot(np.array([x, y, z])) + trans
                fused_points.append((p[0], p[1]))
            sensors_used += 1

        if not fused_points:
            self.get_logger().warn("No valid fused points.")
            return

        # bin into 180 sectors 
        bins = [[] for _ in range(180)]
        for x, y in fused_points:
            angle = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
            idx = int(angle // 2) % 180
            dist = math.hypot(x, y)
            bins[idx].append(dist)

        # require only 2 point per bin
        MIN_POINTS_PER_BIN = 2
        MIN_DIST = 0.05
        MAX_DIST = 4.0

        # build ranges for LaserScan
        ranges = []
        for b in bins:
            if len(b) >= MIN_POINTS_PER_BIN:
                r = max(min(b), MIN_DIST)
                ranges.append(r)
            else:
                # if no data  treat as free space at max_range
                ranges.append(MAX_DIST)

        # publish LaserScan
        scan = LaserScan()
        scan.header.stamp        = now.to_msg()
        scan.header.frame_id     = 'base_link'
        scan.angle_min           = 0.0
        scan.angle_max           = 2 * math.pi
        scan.angle_increment     = math.radians(2.0)
        scan.time_increment      = 0.0
        scan.scan_time           = 0.05
        scan.range_min           = MIN_DIST
        scan.range_max           = MAX_DIST
        scan.ranges              = ranges

        self.laserscan_pub.publish(scan)

        elapsed = (self.get_clock().now().nanoseconds - start_ns) / 1e6
        valid_bins = sum(1 for b in bins if len(b) >= MIN_POINTS_PER_BIN)
        self.get_logger().info(
            f"Scan: {valid_bins}/180 bins valid from {sensors_used} sensors "
            f"({elapsed:.1f} ms)"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FusedObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
