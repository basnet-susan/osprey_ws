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

        self.publisher = self.create_publisher(PointCloud2, '/point_cloud', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.num_sensors = 12
        self.zones_per_sensor = 64
        self.fov = np.radians(65)
        self.grid_size = 8

        self.pixel_angles = self.compute_pixel_angles()

        self.sensor_frames = [
            f"sensor_circ_{i}" for i in range(1, 13)
        ]

    def compute_pixel_angles(self):
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
        points = []

        for sensor_idx in range(self.num_sensors):
            sensor_data = msg.data[sensor_idx * self.zones_per_sensor:(sensor_idx + 1) * self.zones_per_sensor]
            sensor_frame = self.sensor_frames[sensor_idx]

            try:
                self.tf_buffer.can_transform("base_link", sensor_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
                transform = self.tf_buffer.lookup_transform("base_link", sensor_frame, rclpy.time.Time())

                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z

                q = transform.transform.rotation
                r = R.from_quat([q.x, q.y, q.z, q.w])
                rotation_matrix = r.as_matrix()

                for idx, (theta, phi) in enumerate(self.pixel_angles):
                    i, j = divmod(idx, self.grid_size)
                    r_val = sensor_data[i * self.grid_size + j] / 1000.0

                    if r_val < 0.05 or r_val > 4.0:
                        continue

                    local_point = np.array([
                        -r_val * np.tan(phi),
                        -r_val * np.tan(theta),
                        r_val
                    ])

                    world_point = rotation_matrix @ local_point + np.array([tx, ty, tz])
                    points.append(tuple(world_point))

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {sensor_frame}: {str(e)}")

        if points:
            cloud_msg = self.create_point_cloud_msg(points)
            self.publisher.publish(cloud_msg)

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

