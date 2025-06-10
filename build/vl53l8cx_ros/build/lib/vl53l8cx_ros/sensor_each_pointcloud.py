#!/usr/bin/env python3

import rclpy
import numpy as np
import struct
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R

class VL53L8CXPointCloud(Node):
    def __init__(self):
        super().__init__('vl53l8cx_pointcloud')

        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/sensor_data',
            self.sensor_callback,
            10
        )

        self.pointcloud_publishers = {
            f"/point_cloud_sensor_{i+1}": self.create_publisher(PointCloud2, f"/point_cloud_sensor_{i+1}", 10)
            for i in range(12)
        }
        self.overall_publisher = self.create_publisher(PointCloud2, "/point_cloud", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.num_sensors = 12
        self.zones_per_sensor = 64  # 8x8 grid
        self.fov = np.radians(65)
        self.grid_size = 8

        self.pixel_angles = self.compute_pixel_angles()

        self.sensor_frames = [f"sensor_circ_{i+1}" for i in range(self.num_sensors)]

    def compute_pixel_angles(self):
        """Computes per-pixel (θ, φ) angles for each of the 8×8 zones."""
        half_fov = self.fov / 2
        step = self.fov / (self.grid_size - 1)

        angles = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                theta = -half_fov + i * step
                phi = -half_fov + j * step
                angles.append((theta, phi))
        return angles

    def sensor_callback(self, msg):
        """Processes sensor data, transforms points, and publishes individual and overall PointCloud2 messages."""
        overall_points = []

        for sensor_idx in range(self.num_sensors):
            sensor_data = msg.data[sensor_idx * self.zones_per_sensor:(sensor_idx + 1) * self.zones_per_sensor]
            sensor_frame = self.sensor_frames[sensor_idx]
            sensor_points = []

            try:
                self.tf_buffer.can_transform("base_link", sensor_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0))
                transform = self.tf_buffer.lookup_transform("base_link", sensor_frame, rclpy.time.Time())

                tx, ty, tz = (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                )
                q = transform.transform.rotation
                r = R.from_quat([q.x, q.y, q.z, q.w])
                rotation_matrix = r.as_matrix()

                for idx, (theta, phi) in enumerate(self.pixel_angles):
                    i, j = divmod(idx, self.grid_size)
                    r_val = sensor_data[i * self.grid_size + j] / 1000.0

                    if r_val < 0.01 or r_val > 4.0:
                        continue

                    local_point = np.array([
                        -r_val * np.tan(phi),
                        -r_val * np.tan(theta),
                        r_val
                    ])

                    world_point = rotation_matrix @ local_point + np.array([tx, ty, tz])
                    sensor_points.append(tuple(world_point))

                if sensor_points:
                    cloud_msg = self.create_point_cloud_msg(sensor_points)
                    self.pointcloud_publishers[f"/point_cloud_sensor_{sensor_idx+1}"].publish(cloud_msg)

                overall_points.extend(sensor_points)

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {sensor_frame}: {str(e)}")

        if overall_points:
            cloud_msg = self.create_point_cloud_msg(overall_points)
            self.overall_publisher.publish(cloud_msg)

    def create_point_cloud_msg(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        buffer = []
        for p in points:
            buffer.extend(struct.pack("fff", *p))

        msg.data = bytes(buffer)
        return msg

def main():
    rclpy.init()
    node = VL53L8CXPointCloud()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

